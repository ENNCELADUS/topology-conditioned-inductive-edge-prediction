"""Contracts for `src.distill.content_logit`.

The scoring core (`score_content_logit`) is exercised against a tiny
`FeatureStore` fixture and a stub `nn.Module` standing in for a real `V3_1`
checkpoint -- `score_universe._score_v3_1` (the helper this module reuses)
only requires ``model(batch)["logits"]``, so no real checkpoint or the
full feature-pack machinery is needed to test the scoring/IO layer in
isolation. The CLI `main()` (checkpoint loading, `--data-root` benchmark
loading, training-side legality re-check) is NOT exercised here: it needs a
real published `V3_1` checkpoint and a real benchmark data root, which are
heavy fixtures outside this module's scope. The artifact-assembly logic
`main()` performs (copy v2 rows verbatim, append `content_logit`, bump the
manifest format) is instead tested directly against `write_kd_targets`/
`load_kd_targets`, the same primitives `main()` calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.distill.content_logit import build_parser, score_content_logit
from src.model.egostitch.classifier.b0_v31 import V3_1
from torch import nn

pytestmark = pytest.mark.unit

_DIM = 4
_LENGTH = 3  # every node gets the same token length so padding is a no-op


def _write_feature_store(root: Path, node_tokens: dict[str, torch.Tensor]) -> None:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for node_id, tokens in node_tokens.items():
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tokens, root / rel_path)
        index[node_id] = rel_path
    (root / "metadata.json").write_text(
        json.dumps({"format": "torch_pt_per_node", "input_dim": _DIM})
    )
    (root / "index.json").write_text(json.dumps(index))


class _StubContentModel(nn.Module):
    """`mean(emb_a) - mean(emb_b)`: a deterministic, independently recomputable logit."""

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"logits": batch["emb_a"].mean(dim=(1, 2)) - batch["emb_b"].mean(dim=(1, 2))}


class _NonFiniteModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["emb_a"].size(0)
        logits = torch.zeros(batch_size)
        logits[0] = float("nan")
        return {"logits": logits}


def _tiny_store(tmp_path: Path) -> tuple[FeatureStore, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    node_ids = ["n0", "n1", "n2", "n3"]
    node_tokens = {node: torch.randn(_LENGTH, _DIM) for node in node_ids}
    root = tmp_path / "features"
    _write_feature_store(root, node_tokens)
    return FeatureStore(root), node_tokens


def test_score_content_logit_matches_direct_stub_computation(tmp_path: Path) -> None:
    store, node_tokens = _tiny_store(tmp_path)
    node_ids = ["n0", "n1", "n2", "n3"]
    pair_anchor_idx = np.array([0, 1, 2], dtype=np.int32)
    pair_partner_idx = np.array([1, 2, 3], dtype=np.int32)

    logits = score_content_logit(
        cast(V3_1, _StubContentModel()),
        node_ids,
        pair_anchor_idx,
        pair_partner_idx,
        store,
        device=torch.device("cpu"),
        token_budget=4096,
    )

    expected = np.array(
        [
            float(node_tokens[node_ids[a]].mean() - node_tokens[node_ids[b]].mean())
            for a, b in zip(pair_anchor_idx.tolist(), pair_partner_idx.tolist(), strict=True)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(logits, expected, atol=1e-5)
    assert logits.dtype == np.float32


def test_score_content_logit_raises_on_non_finite_logit(tmp_path: Path) -> None:
    store, _ = _tiny_store(tmp_path)
    node_ids = ["n0", "n1", "n2", "n3"]
    pair_anchor_idx = np.array([0, 1], dtype=np.int32)
    pair_partner_idx = np.array([1, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="non-finite"):
        score_content_logit(
            cast(V3_1, _NonFiniteModel()),
            node_ids,
            pair_anchor_idx,
            pair_partner_idx,
            store,
            device=torch.device("cpu"),
            token_budget=4096,
        )


def test_build_parser_accepts_the_documented_cli_signature() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--targets",
            "targets_v2",
            "--checkpoint",
            "best.pt",
            "--data-root",
            "data",
            "--strategy",
            "breadth_first",
            "--output",
            "targets_v3",
        ]
    )
    assert args.targets == Path("targets_v2")
    assert args.checkpoint == Path("best.pt")
    assert args.data_root == Path("data")
    assert args.strategy == "breadth_first"
    assert args.output == Path("targets_v3")
    assert args.device == "cpu"
    assert args.token_budget == 32_768


# --------------------------------------------------------------------------- artifact assembly


def _write_v2_artifact(path: Path) -> None:
    rng = np.random.default_rng(0)
    write_kd_targets(
        path,
        node_ids=["a", "b", "c"],
        pair_anchor_idx=np.array([0, 0, 1], dtype=np.int32),
        pair_partner_idx=np.array([1, 2, 2], dtype=np.int32),
        anchor_offsets=np.array([0, 2, 3, 3], dtype=np.int64),
        teacher_logit=np.array([0.5, -0.3, 1.2], dtype=np.float32),
        teacher_pooled_ab=rng.standard_normal((3, 4)).astype(np.float16),
        teacher_pooled_ba=rng.standard_normal((3, 4)).astype(np.float16),
        is_near=np.array([1, 0, 1], dtype=np.uint8),
        pair_label=np.array([1, 0, 0], dtype=np.int8),
        truth_graph_sha256="oracle-truth-sha",
        checkpoint_path=Path("oracle.pt"),
        checkpoint_sha256="oracle-sha",
        checkpoint_id="oracle123",
        k_near=2,
        k_rand=1,
        seed=0,
    )


def test_v3_artifact_assembly_copies_v2_rows_byte_identical_and_carries_provenance(
    tmp_path: Path,
) -> None:
    """Mirrors `content_logit.main()`'s artifact-assembly step directly.

    Builds a v2 artifact, computes a content_logit array independent of the
    v2 rows, writes a v3 artifact the same way `main()` does (v2's own
    checkpoint provenance stays primary; the content checkpoint gets its own
    manifest fields), and checks every array the row-copy contract requires
    to be untouched plus the ones that must change.
    """
    v2_dir = tmp_path / "v2"
    _write_v2_artifact(v2_dir)
    v2 = load_kd_targets(v2_dir)

    content_logit = np.array([0.11, 0.22, 0.33], dtype=np.float32)
    v3_dir = tmp_path / "v3"
    write_kd_targets(
        v3_dir,
        node_ids=v2.node_ids,
        pair_anchor_idx=v2.pair_anchor_idx,
        pair_partner_idx=v2.pair_partner_idx,
        anchor_offsets=v2.anchor_offsets,
        teacher_logit=v2.teacher_logit,
        teacher_pooled_ab=v2.teacher_pooled_ab,
        teacher_pooled_ba=v2.teacher_pooled_ba,
        is_near=v2.is_near,
        pair_label=v2.pair_label,
        content_logit=content_logit,
        truth_graph_sha256=str(v2.manifest["truth_graph_sha256"]),
        checkpoint_path=Path(str(v2.manifest["checkpoint_path"])),
        checkpoint_sha256=str(v2.manifest["checkpoint_sha256"]),
        checkpoint_id=str(v2.manifest["checkpoint_id"]),
        k_near=cast(int, v2.manifest["k_near"]),
        k_rand=cast(int, v2.manifest["k_rand"]),
        seed=cast(int, v2.manifest["seed"]),
    )
    manifest_path = v3_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_checkpoint_path"] = "content.pt"
    manifest["content_checkpoint_sha256"] = "content-sha"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    v3 = load_kd_targets(v3_dir)
    assert v3.node_ids == v2.node_ids
    np.testing.assert_array_equal(v3.pair_anchor_idx, v2.pair_anchor_idx)
    np.testing.assert_array_equal(v3.pair_partner_idx, v2.pair_partner_idx)
    np.testing.assert_array_equal(v3.anchor_offsets, v2.anchor_offsets)
    np.testing.assert_array_equal(v3.is_near, v2.is_near)
    np.testing.assert_array_equal(v3.pair_label, v2.pair_label)
    np.testing.assert_array_equal(v3.teacher_logit, v2.teacher_logit)
    np.testing.assert_array_equal(v3.teacher_pooled_ab, v2.teacher_pooled_ab)
    np.testing.assert_array_equal(v3.teacher_pooled_ba, v2.teacher_pooled_ba)
    assert v3.content_logit is not None
    np.testing.assert_array_equal(v3.content_logit, content_logit)
    assert v3.manifest["format"] == "kd_targets_v3"
    # Primary checkpoint_* fields keep v2's own (oracle) provenance.
    assert v3.manifest["checkpoint_id"] == "oracle123"
    assert v3.manifest["content_checkpoint_path"] == "content.pt"
