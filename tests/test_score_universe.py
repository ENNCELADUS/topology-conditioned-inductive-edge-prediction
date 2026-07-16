"""Tests for `src.score_universe` (scoring CLI, sharding, and the scores artifact).

All tests are synthetic: tiny fake feature roots and pair files built under
`tmp_path`. No dependence on the real `data/` package.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from src import score_universe
from src.data import packed_features
from src.data.artifacts import canonical_pair
from src.data.distributed_pairs import CompactPairBatch
from src.data.features import FeatureStore
from src.data.packed_features import PackedFeatureTable, build_packed_features
from src.model.B0 import V3_1

INPUT_DIM = 4


def _write_feature_store(root: Path, node_tokens: dict[str, torch.Tensor]) -> None:
    """Write a synthetic FeatureStore package (metadata.json/index.json/embeddings)."""
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for node_id, tokens in node_tokens.items():
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tokens, root / rel_path)
        index[node_id] = rel_path
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": INPUT_DIM, "max_sequence_length": 1024}
        )
    )
    (root / "index.json").write_text(json.dumps(index))


def _data_root_with_features(tmp_path: Path, node_tokens: dict[str, torch.Tensor]) -> Path:
    """Build a `<data_root>/features/frozen_node_features_1024/` package under tmp_path."""
    data_root = tmp_path / "data"
    features_root = data_root / "features" / "frozen_node_features_1024"
    _write_feature_store(features_root, node_tokens)
    return data_root


def _tiny_v3_1_config() -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0, "activation": "gelu", "norm": "layernorm"},
        "regularization": {"dropout": 0.0},
    }


def _write_checkpoint(
    path: Path, *, model: nn.Module, model_family: str, model_config: dict[str, object]
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_family": model_family,
            "model_config": model_config,
            "epoch": 0,
            "val_metrics": {},
            "seed": 0,
            "config": {},
        },
        path,
    )


def test_packed_scoring_caches_encoder_without_changing_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    assert isinstance(model, V3_1)
    model.eval()
    feature_root = tmp_path / "features"
    nodes = {
        "node_00": torch.randn(3, INPUT_DIM),
        "node_01": torch.randn(4, INPUT_DIM),
        "node_02": torch.randn(5, INPUT_DIM),
    }
    _write_feature_store(feature_root, nodes)
    pack_root = tmp_path / "pack"
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)
    build_packed_features(feature_root, pack_root, workers=1)
    pairs = [("node_00", "node_01"), ("node_02", "node_00")]

    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    node_index = table.manifest.node_index()
    compact = CompactPairBatch(
        row_ids=torch.tensor([0, 1]),
        node_a=torch.tensor([node_index[u] for u, _ in pairs]),
        node_b=torch.tensor([node_index[v] for _, v in pairs]),
        labels=torch.zeros(2),
        bucket_boundary=128,
        global_pair_count=2,
    )
    reference_batch = table.assemble(compact)
    reference_batch["emb_a"] = reference_batch["emb_a"].float()
    reference_batch["emb_b"] = reference_batch["emb_b"].float()
    with torch.inference_mode():
        reference = model(reference_batch)["logits"].numpy().reshape(-1)

    actual = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack_root,
        device=torch.device("cpu"),
        amp="off",
        token_budget=512,
    )

    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=0.0)


def test_load_bare_legacy_checkpoint_with_explicit_model_metadata(tmp_path: Path) -> None:
    model_config = _tiny_v3_1_config()
    source_model = score_universe.build_model("v3_1", model_config)
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(source_model.state_dict(), checkpoint_path)

    loaded_model, model_family, checkpoint_id = score_universe._load_checkpoint(
        checkpoint_path,
        model_family="v3_1",
        model_config=model_config,
    )

    assert model_family == "v3_1"
    assert checkpoint_id == score_universe._checkpoint_id(source_model.state_dict())
    for name, expected in source_model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[name], expected)


def test_cli_scores_bare_legacy_checkpoint_with_explicit_model_config(tmp_path: Path) -> None:
    model_config = _tiny_v3_1_config()
    source_model = score_universe.build_model("v3_1", model_config)
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(source_model.state_dict(), checkpoint_path)
    config_path = tmp_path / "legacy-config.yaml"
    config_path.write_text(json.dumps({"model": {"family": "v3_1", "config": model_config}}))

    nodes = {"node_00": torch.randn(3, INPUT_DIM), "node_01": torch.randn(4, INPUT_DIM)}
    data_root = _data_root_with_features(tmp_path, nodes)
    pairs_path = tmp_path / "pairs.tsv"
    _write_tsv(pairs_path, [("node_00", "node_01", 1)])
    output_path = tmp_path / "scores.npz"

    score_universe.main(
        [
            "score",
            "--checkpoint",
            str(checkpoint_path),
            "--model-family",
            "v3_1",
            "--model-config",
            str(config_path),
            "--pairs",
            f"file:{pairs_path}",
            "--data-root",
            str(data_root),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ]
    )

    scores = score_universe.load_scores(output_path)
    assert scores.meta["model_family"] == "v3_1"
    assert scores.meta["checkpoint_id"] == score_universe._checkpoint_id(source_model.state_dict())
    assert scores.logit.shape == (1,)


def _write_tsv(path: Path, rows: list[tuple[str, str, int | None]]) -> None:
    lines = []
    for u, v, label in rows:
        lines.append(f"{u}\t{v}" if label is None else f"{u}\t{v}\t{label}")
    path.write_text("\n".join(lines) + "\n")


class _FakeF0PairMLP(nn.Module):
    """Stand-in mirroring the pinned `src.model.b0_alt.F0PairMLP` batch contract.

    `src/model/b0_alt.py` is being built concurrently (Task 4) and did not exist at the
    time these tests were written. Per the Task-5 brief's contingency note, the f0_mlp
    scoring path in `score_universe` is exercised end-to-end using a local double that
    implements the identical pinned contract (batch keys `x_a`/`x_b`, optional `label`,
    returns `{"logits", "loss"}`), registered into `score_universe.MODEL_BUILDERS` for the
    duration of a single test via monkeypatch. This proves the F0-matrix batching, row
    order restoration, determinism, and label passthrough logic without depending on
    `b0_alt.py`'s landing order.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = dict(batch or {})
        merged.update(kwargs)
        x_a, x_b = merged["x_a"], merged["x_b"]
        features = torch.cat([x_a + x_b, (x_a - x_b).abs(), x_a * x_b], dim=-1)
        logits = self.net(features)
        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), merged["label"].float()
            )
        return output


def _build_fake_f0(config: dict[str, object]) -> nn.Module:
    return _FakeF0PairMLP(
        input_dim=cast(int, config["input_dim"]), hidden_dim=cast(int, config["hidden_dim"])
    )


# ---------------------------------------------------------------------------
# save_scores / load_scores roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_everything(tmp_path: Path) -> None:
    node_ids = ["node_a", "node_b", "node_c"]
    u_idx = np.array([0, 1, 2], dtype=np.int32)
    v_idx = np.array([1, 2, 0], dtype=np.int32)
    logit = np.array([2.0, -1.0, 0.0], dtype=np.float32)
    label = np.array([1, -1, 0], dtype=np.int8)
    meta = {
        "checkpoint_id": "deadbeefcafef00d",
        "model_family": "v3_1",
        "pairs_source": "file:/tmp/x.tsv",
        "strategy": "breadth_first",
        "num_rows": 3,
        "created_utc": "2026-07-09T00:00:00+00:00",
        "torch_version": torch.__version__,
    }
    out_path = tmp_path / "scores.npz"

    score_universe.save_scores(
        out_path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logit,
        label=label,
        row_start=0,
        meta=meta,
    )
    loaded = score_universe.load_scores(out_path)

    assert loaded.node_ids == node_ids
    np.testing.assert_array_equal(loaded.u_idx, u_idx)
    np.testing.assert_array_equal(loaded.v_idx, v_idx)
    np.testing.assert_array_equal(loaded.logit, logit)
    np.testing.assert_array_equal(loaded.label, label)
    assert loaded.meta == meta
    assert list(loaded.pairs()) == [
        ("node_a", "node_b"),
        ("node_b", "node_c"),
        ("node_c", "node_a"),
    ]
    np.testing.assert_allclose(loaded.probs(), 1.0 / (1.0 + np.exp(-logit.astype(np.float64))))


# ---------------------------------------------------------------------------
# f0_mlp end-to-end (via a local double registered under MODEL_BUILDERS)
# ---------------------------------------------------------------------------


def test_f0_mlp_end_to_end_preserves_order_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(score_universe.MODEL_BUILDERS, "f0_mlp", _build_fake_f0)

    nodes = [f"node_{i:02d}" for i in range(6)]
    node_tokens = {n: torch.randn(3 + i, INPUT_DIM) for i, n in enumerate(nodes)}
    data_root = _data_root_with_features(tmp_path, node_tokens)

    rows: list[tuple[str, str, int | None]] = []
    for i in range(20):
        u, v = nodes[i % 6], nodes[(i * 3 + 1) % 6]
        label: int | None = i % 2
        if i % 7 == 0:
            label = None
        rows.append((u, v, label))
    tsv_path = tmp_path / "pairs.tsv"
    _write_tsv(tsv_path, rows)

    model = _FakeF0PairMLP(input_dim=INPUT_DIM)
    model_config: dict[str, object] = {"input_dim": INPUT_DIM, "hidden_dim": 8}
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="f0_mlp", model_config=model_config
    )

    out_path = tmp_path / "scores.npz"
    f0_cache = tmp_path / "f0_cache.pt"

    def run() -> score_universe.ScoresArtifact:
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint_path),
                "--pairs",
                f"file:{tsv_path}",
                "--data-root",
                str(data_root),
                "--strategy",
                "breadth_first",
                "--output",
                str(out_path),
                "--device",
                "cpu",
                "--f0-cache",
                str(f0_cache),
            ]
        )
        return score_universe.load_scores(out_path)

    result_1 = run()
    # Pairs are stored canonicalized (repo-wide convention: (min(u, v), max(u, v))),
    # in the input row order.
    expected_pairs = [canonical_pair(u, v) for u, v, _ in rows]
    expected_labels = np.array(
        [(-1 if label is None else label) for _, _, label in rows], dtype=np.int8
    )

    assert list(result_1.pairs()) == expected_pairs
    np.testing.assert_array_equal(result_1.label, expected_labels)
    assert np.all(np.isfinite(result_1.logit))
    assert result_1.meta["model_family"] == "f0_mlp"
    assert result_1.meta["pairs_source"] == f"file:{tsv_path}"
    assert result_1.meta["num_rows"] == 20

    # Row-alignment proof: logit[i] must equal the model applied to row i's own
    # mean-pooled features (computed here independently of the CLI machinery).
    model.eval()
    with torch.inference_mode():
        for i, (u, v) in enumerate(expected_pairs):
            x_a = node_tokens[u].mean(dim=0, keepdim=True)
            x_b = node_tokens[v].mean(dim=0, keepdim=True)
            expected_logit = model({"x_a": x_a, "x_b": x_b})["logits"].item()
            assert result_1.logit[i] == pytest.approx(expected_logit, abs=1e-5)

    result_2 = run()
    np.testing.assert_array_equal(result_1.logit, result_2.logit)


# ---------------------------------------------------------------------------
# v3_1 tiny-config end-to-end (real V3_1, length-bucketed path)
# ---------------------------------------------------------------------------


def test_v3_1_tiny_config_end_to_end_restores_row_order(tmp_path: Path) -> None:
    nodes = [f"node_{i:02d}" for i in range(6)]
    # Distinct lengths spanning at least two bucket boundaries (128, 256).
    lengths = [5, 150, 8, 200, 12, 6]
    node_tokens = {
        n: torch.randn(length, INPUT_DIM) for n, length in zip(nodes, lengths, strict=True)
    }
    data_root = _data_root_with_features(tmp_path, node_tokens)

    rows: list[tuple[str, str, int | None]] = []
    for i in range(20):
        u, v = nodes[i % 6], nodes[(i * 5 + 2) % 6]
        label: int | None = i % 2
        if i % 6 == 0:
            label = None
        rows.append((u, v, label))
    tsv_path = tmp_path / "pairs.tsv"
    _write_tsv(tsv_path, rows)

    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )

    out_path = tmp_path / "scores.npz"

    def run() -> score_universe.ScoresArtifact:
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint_path),
                "--pairs",
                f"file:{tsv_path}",
                "--data-root",
                str(data_root),
                "--strategy",
                "breadth_first",
                "--output",
                str(out_path),
                "--device",
                "cpu",
            ]
        )
        return score_universe.load_scores(out_path)

    result_1 = run()
    expected_pairs = [canonical_pair(u, v) for u, v, _ in rows]
    expected_labels = np.array(
        [(-1 if label is None else label) for _, _, label in rows], dtype=np.int8
    )

    assert list(result_1.pairs()) == expected_pairs
    np.testing.assert_array_equal(result_1.label, expected_labels)
    assert np.all(np.isfinite(result_1.logit))

    # Row-alignment proof: the length-bucketed path processes rows out of input
    # order, so logit[i] must still equal a single-pair forward of row i's own
    # token sequences (computed here independently of the batching machinery).
    model.eval()
    with torch.inference_mode():
        for i, (u, v) in enumerate(expected_pairs):
            single = {
                "emb_a": node_tokens[u].unsqueeze(0),
                "emb_b": node_tokens[v].unsqueeze(0),
                "len_a": torch.tensor([node_tokens[u].size(0)], dtype=torch.int64),
                "len_b": torch.tensor([node_tokens[v].size(0)], dtype=torch.int64),
            }
            expected_logit = model(single)["logits"].item()
            assert result_1.logit[i] == pytest.approx(expected_logit, abs=1e-4)

    result_2 = run()
    np.testing.assert_array_equal(result_1.logit, result_2.logit)


# ---------------------------------------------------------------------------
# Sharding + merge == unsharded run
# ---------------------------------------------------------------------------


def test_shard_and_merge_matches_unsharded_run(tmp_path: Path) -> None:
    nodes = [f"node_{i:02d}" for i in range(8)]
    node_tokens = {n: torch.randn(6, INPUT_DIM) for n in nodes}
    data_root = _data_root_with_features(tmp_path, node_tokens)

    rows: list[tuple[str, str, int | None]] = []
    for i in range(23):
        u, v = nodes[i % 8], nodes[(i * 3 + 1) % 8]
        label: int | None = i % 2
        if i % 5 == 0:
            label = None
        rows.append((u, v, label))
    tsv_path = tmp_path / "pairs.tsv"
    _write_tsv(tsv_path, rows)

    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )

    # token-budget=1 forces batch size 1 for every example (cap = max(1, budget // (2*boundary))),
    # so per-example results cannot depend on which other rows share a batch. This keeps the
    # bit-identical comparison between the unsharded and sharded runs free of any incidental
    # floating-point batch-composition effects that would otherwise be an artifact of sharding
    # rather than of the scoring logic under test.
    common_args = [
        "--checkpoint",
        str(checkpoint_path),
        "--pairs",
        f"file:{tsv_path}",
        "--data-root",
        str(data_root),
        "--strategy",
        "breadth_first",
        "--device",
        "cpu",
        "--token-budget",
        "1",
    ]

    unsharded_out = tmp_path / "unsharded.npz"
    score_universe.main(["score", *common_args, "--output", str(unsharded_out)])
    unsharded = score_universe.load_scores(unsharded_out)

    sharded_output_base = tmp_path / "sharded.npz"
    shard_paths = []
    for shard in range(3):
        score_universe.main(
            [
                "score",
                *common_args,
                "--output",
                str(sharded_output_base),
                "--shard",
                str(shard),
                "--num-shards",
                "3",
            ]
        )
        shard_paths.append(tmp_path / f"sharded.shard-{shard}.npz")
    for shard_path in shard_paths:
        assert shard_path.exists()

    merged_out = tmp_path / "merged.npz"
    score_universe.main(
        ["merge", "--inputs", *[str(p) for p in shard_paths], "--output", str(merged_out)]
    )
    merged = score_universe.load_scores(merged_out)

    assert merged.node_ids == unsharded.node_ids
    np.testing.assert_array_equal(merged.u_idx, unsharded.u_idx)
    np.testing.assert_array_equal(merged.v_idx, unsharded.v_idx)
    np.testing.assert_array_equal(merged.logit, unsharded.logit)
    np.testing.assert_array_equal(merged.label, unsharded.label)
    assert merged.meta["checkpoint_id"] == unsharded.meta["checkpoint_id"]
    assert merged.meta["model_family"] == unsharded.meta["model_family"]
    assert merged.meta["pairs_source"] == unsharded.meta["pairs_source"]
    assert merged.meta["num_rows"] == unsharded.meta["num_rows"] == 23


# ---------------------------------------------------------------------------
# merge failure cases
# ---------------------------------------------------------------------------


def _write_fake_shard(
    path: Path,
    *,
    row_start: int,
    n_rows: int,
    num_rows: int,
    checkpoint_id: str = "abc123abc123abcd",
) -> None:
    node_ids = ["node_a", "node_b"]
    score_universe.save_scores(
        path,
        node_ids=node_ids,
        u_idx=np.zeros(n_rows, dtype=np.int32),
        v_idx=np.ones(n_rows, dtype=np.int32),
        logit=np.zeros(n_rows, dtype=np.float32),
        label=np.full(n_rows, -1, dtype=np.int8),
        row_start=row_start,
        meta={
            "checkpoint_id": checkpoint_id,
            "model_family": "v3_1",
            "pairs_source": "candidate",
            "strategy": "breadth_first",
            "num_rows": num_rows,
            "created_utc": "2026-07-09T00:00:00+00:00",
            "torch_version": torch.__version__,
        },
    )


def test_merge_overlapping_ranges_raises_clear_error(tmp_path: Path) -> None:
    shard0 = tmp_path / "s0.npz"
    shard1 = tmp_path / "s1.npz"
    _write_fake_shard(shard0, row_start=0, n_rows=10, num_rows=20)
    _write_fake_shard(shard1, row_start=8, n_rows=12, num_rows=20)

    with pytest.raises(ValueError, match="overlap"):
        score_universe.merge_scores([shard0, shard1])


def test_merge_gap_raises_clear_error(tmp_path: Path) -> None:
    shard0 = tmp_path / "s0.npz"
    shard1 = tmp_path / "s1.npz"
    _write_fake_shard(shard0, row_start=0, n_rows=10, num_rows=20)
    _write_fake_shard(shard1, row_start=12, n_rows=8, num_rows=20)

    with pytest.raises(ValueError, match="gap"):
        score_universe.merge_scores([shard0, shard1])


def test_merge_mismatched_checkpoint_id_raises_clear_error(tmp_path: Path) -> None:
    shard0 = tmp_path / "s0.npz"
    shard1 = tmp_path / "s1.npz"
    _write_fake_shard(shard0, row_start=0, n_rows=10, num_rows=20, checkpoint_id="aaaaaaaaaaaaaaaa")
    _write_fake_shard(
        shard1, row_start=10, n_rows=10, num_rows=20, checkpoint_id="bbbbbbbbbbbbbbbb"
    )

    with pytest.raises(ValueError, match="checkpoint_id"):
        score_universe.merge_scores([shard0, shard1])


# ---------------------------------------------------------------------------
# CLI arg errors
# ---------------------------------------------------------------------------


def test_cli_bad_pairs_argument_raises_system_exit(tmp_path: Path) -> None:
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )
    data_root = _data_root_with_features(tmp_path, {"node_00": torch.randn(3, INPUT_DIM)})

    with pytest.raises(SystemExit):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint_path),
                "--pairs",
                "not-a-valid-source",
                "--data-root",
                str(data_root),
                "--strategy",
                "breadth_first",
                "--output",
                str(tmp_path / "out.npz"),
                "--device",
                "cpu",
            ]
        )


def test_cli_missing_checkpoint_raises_system_exit(tmp_path: Path) -> None:
    data_root = _data_root_with_features(tmp_path, {"node_00": torch.randn(3, INPUT_DIM)})
    with pytest.raises(SystemExit):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(tmp_path / "does_not_exist.pt"),
                "--pairs",
                "file:" + str(tmp_path / "missing.tsv"),
                "--data-root",
                str(data_root),
                "--strategy",
                "breadth_first",
                "--output",
                str(tmp_path / "out.npz"),
                "--device",
                "cpu",
            ]
        )


# --------------------------------------------------------------------------- egostitch scoring


_TINY_EGOSTITCH_CONFIG: dict[str, object] = {
    "input_dim": INPUT_DIM,
    "d_p": 4,
    "d_z": 4,
    "d_h": 8,
    "slots": 3,
    "m_max": 8,
    "n_ground": 2,
    "decoder_layers": 1,
    "n_heads": 2,
    "gin_hidden": 8,
    "gin_layers": 2,
    "sinkhorn_iters": 3,
}
_EGOSTITCH_NODES = [f"e{i}" for i in range(5)]
# Includes a self-pair: it must route through the single-ego path.
_EGOSTITCH_PAIRS = [("e0", "e1"), ("e1", "e2"), ("e0", "e3"), ("e2", "e4"), ("e3", "e3")]


def _egostitch_setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Feature store + checkpoint + pairs TSV + covering b0-scores artifact."""
    torch.manual_seed(0)
    node_tokens = {node: torch.randn(3 + i, INPUT_DIM) for i, node in enumerate(_EGOSTITCH_NODES)}
    data_root = _data_root_with_features(tmp_path, node_tokens)

    model = score_universe.build_model("egostitch", dict(_TINY_EGOSTITCH_CONFIG))
    checkpoint_path = tmp_path / "egostitch.pt"
    _write_checkpoint(
        checkpoint_path,
        model=model,
        model_family="egostitch",
        model_config=dict(_TINY_EGOSTITCH_CONFIG),
    )

    pairs_path = tmp_path / "pairs.tsv"
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in _EGOSTITCH_PAIRS))

    b0_path = tmp_path / "b0_scores.npz"
    position = {node: i for i, node in enumerate(_EGOSTITCH_NODES)}
    score_universe.save_scores(
        b0_path,
        node_ids=list(_EGOSTITCH_NODES),
        u_idx=np.array([position[u] for u, _ in _EGOSTITCH_PAIRS], dtype=np.int32),
        v_idx=np.array([position[v] for _, v in _EGOSTITCH_PAIRS], dtype=np.int32),
        logit=np.linspace(-1.0, 1.0, len(_EGOSTITCH_PAIRS)).astype(np.float32),
        label=np.full(len(_EGOSTITCH_PAIRS), -1, dtype=np.int8),
        row_start=0,
        meta={
            "checkpoint_id": "cafecafecafecafe",
            "model_family": "v3_1",
            "pairs_source": "file",
            "strategy": "breadth_first",
            "num_rows": len(_EGOSTITCH_PAIRS),
            "created_utc": "2026-07-14T00:00:00+00:00",
            "torch_version": str(torch.__version__),
        },
    )
    return data_root, checkpoint_path, pairs_path, b0_path


def _egostitch_score_args(
    tmp_path: Path, data_root: Path, checkpoint: Path, pairs: Path, b0: Path, output: Path
) -> list[str]:
    return [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        f"file:{pairs}",
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--batch-pairs",
        "3",
        "--b0-scores",
        str(b0),
        "--s0-checkpoint-id",
        "cafecafecafecafe",
        "--f0-cache",
        str(tmp_path / "f0_cache.pt"),
        "--grounding-cache",
        str(tmp_path / "grounding.npz"),
        "--device",
        "cpu",
    ]


def test_build_model_egostitch_round_trips() -> None:
    from src.model.egostitch import EgoStitchStage1

    model = score_universe.build_model("egostitch", dict(_TINY_EGOSTITCH_CONFIG))
    assert isinstance(model, EgoStitchStage1)
    assert model.config.slots == 3


def test_egostitch_cli_scores_end_to_end_with_self_pair(tmp_path: Path) -> None:
    data_root, checkpoint, pairs, b0 = _egostitch_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_score_args(tmp_path, data_root, checkpoint, pairs, b0, output))
    artifact = score_universe.load_scores(output)
    assert len(artifact.logit) == len(_EGOSTITCH_PAIRS)
    assert bool(np.isfinite(artifact.logit).all())
    assert artifact.meta["model_family"] == "egostitch"
    assert artifact.meta["s0_checkpoint_id"] == "cafecafecafecafe"
    assert artifact.meta["score_precision"] == {
        "contract": "egostitch_pair_fp32_v1",
        "encode_autocast": "off",
        "pair_compute_dtype": "float32",
        "pair_autocast": False,
        "logit_storage_dtype": "float32",
    }
    resolution = artifact.meta["score_resolution"]
    assert isinstance(resolution, dict)
    assert resolution["n_rows"] == len(_EGOSTITCH_PAIRS)
    assert resolution["n_unique"] == int(np.unique(artifact.logit).size)
    density = artifact.meta["density_calibration"]
    assert isinstance(density, dict)
    assert density["pass"] == 1
    assert 0.0 <= float(density["rho_hat_eval"]) <= 1.0


def test_egostitch_cli_is_deterministic(tmp_path: Path) -> None:
    data_root, checkpoint, pairs, b0 = _egostitch_setup(tmp_path)
    out_a, out_b = tmp_path / "a.npz", tmp_path / "b.npz"
    for output in (out_a, out_b):
        score_universe.main(
            _egostitch_score_args(tmp_path, data_root, checkpoint, pairs, b0, output)
        )
    np.testing.assert_array_equal(
        score_universe.load_scores(out_a).logit, score_universe.load_scores(out_b).logit
    )


def test_egostitch_cli_requires_b0_scores(tmp_path: Path) -> None:
    data_root, checkpoint, pairs, b0 = _egostitch_setup(tmp_path)
    args = _egostitch_score_args(tmp_path, data_root, checkpoint, pairs, b0, tmp_path / "out.npz")
    idx = args.index("--b0-scores")
    del args[idx : idx + 2]
    with pytest.raises(ValueError, match="requires --b0-scores"):
        score_universe.main(args)


def test_egostitch_cli_rejects_s0_checkpoint_mismatch(tmp_path: Path) -> None:
    data_root, checkpoint, pairs, b0 = _egostitch_setup(tmp_path)
    args = _egostitch_score_args(tmp_path, data_root, checkpoint, pairs, b0, tmp_path / "out.npz")
    args[args.index("--s0-checkpoint-id") + 1] = "0000000000000000"
    with pytest.raises(ValueError, match="checkpoint_id mismatch"):
        score_universe.main(args)


def test_egostitch_cli_hard_fails_on_missing_s0_pair(tmp_path: Path) -> None:
    data_root, checkpoint, pairs, b0 = _egostitch_setup(tmp_path)
    # Rewrite the pairs file with a pair the b0 artifact does not cover.
    pairs.write_text(pairs.read_text() + "e1\te4\n")
    with pytest.raises(ValueError, match="missing pair"):
        score_universe.main(
            _egostitch_score_args(tmp_path, data_root, checkpoint, pairs, b0, tmp_path / "o.npz")
        )


# ---------------------------------------------------------------------------
# egostitch pair pass is fp32 even under bf16 amp (spec Sec 13.16)
# ---------------------------------------------------------------------------


def test_score_egostitch_pair_pass_is_fp32_even_under_bf16_amp(tmp_path: Path) -> None:
    """The bf16-autocast regression: only the per-node encode pass may use bf16.

    Before the fix, the pair-scoring loop (Sinkhorn stitch + decision head) ran
    inside the same `bf16` autocast context, so every logit landed on the bf16
    ulp grid regardless of the fp32 output dtype. This reproduces that bug's
    exact trigger (`--amp bf16` on CPU) and asserts full fp32 resolution.
    """
    torch.manual_seed(0)
    n_nodes = 16
    nodes = [f"e{i}" for i in range(n_nodes)]
    node_tokens = {node: torch.randn(3 + (i % 5), INPUT_DIM) for i, node in enumerate(nodes)}
    data_root = _data_root_with_features(tmp_path, node_tokens)
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")

    model = score_universe.build_model("egostitch", dict(_TINY_EGOSTITCH_CONFIG))
    model.eval()

    rng = np.random.default_rng(1)
    n_pairs = 60
    u_idx = rng.integers(0, n_nodes, size=n_pairs)
    v_idx = (u_idx + rng.integers(1, n_nodes, size=n_pairs)) % n_nodes
    v_idx[::5] = u_idx[::5]
    pairs = [(nodes[int(u)], nodes[int(v)]) for u, v in zip(u_idx, v_idx, strict=True)]
    s0_logits = rng.normal(size=n_pairs).astype(np.float32)

    logits = score_universe._score_egostitch(
        model,
        pairs,
        store,
        device=torch.device("cpu"),
        amp="bf16",
        batch_pairs=5,
        f0_cache=tmp_path / "f0_cache.pt",
        grounding_cache=tmp_path / "grounding.npz",
        s0_logits=s0_logits,
    )

    assert np.all(np.isfinite(logits))
    bf16_roundtrip = torch.from_numpy(logits).to(torch.bfloat16).to(torch.float32).numpy()
    is_self = u_idx == v_idx
    assert np.any(logits[~is_self] != bf16_roundtrip[~is_self])
    assert np.any(logits[is_self] != bf16_roundtrip[is_self])
    assert np.unique(logits).size == n_pairs


# ---------------------------------------------------------------------------
# pair-pass precision provenance (spec Sec 13.16)
# ---------------------------------------------------------------------------


def _egostitch_precision_meta(n_rows: int) -> dict[str, object]:
    return {
        "checkpoint_id": "abc123abc123abcd",
        "model_family": "egostitch",
        "pairs_source": "candidate",
        "strategy": "breadth_first",
        "num_rows": n_rows,
        "created_utc": "2026-07-16T00:00:00+00:00",
        "torch_version": str(torch.__version__),
        "score_precision": {
            "contract": "egostitch_pair_fp32_v1",
            "encode_autocast": "bf16",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
    }


def test_save_scores_rejects_egostitch_without_precision_provenance(tmp_path: Path) -> None:
    meta = _egostitch_precision_meta(2)
    del meta["score_precision"]
    with pytest.raises(ValueError, match="missing score_precision provenance"):
        score_universe.save_scores(
            tmp_path / "missing-precision.npz",
            node_ids=["node_a", "node_b"],
            u_idx=np.array([0, 0], dtype=np.int32),
            v_idx=np.array([1, 1], dtype=np.int32),
            logit=np.array([0.0, 1.0], dtype=np.float32),
            label=np.full(2, -1, dtype=np.int8),
            row_start=0,
            meta=meta,
        )


def test_save_scores_accepts_legitimate_low_unique_egostitch_logits(tmp_path: Path) -> None:
    n_rows = 10_000
    values = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    logit = np.tile(values, n_rows // len(values) + 1)[:n_rows].astype(np.float32)
    output = tmp_path / "low-unique.npz"
    score_universe.save_scores(
        output,
        node_ids=["node_a", "node_b"],
        u_idx=np.zeros(n_rows, dtype=np.int32),
        v_idx=np.ones(n_rows, dtype=np.int32),
        logit=logit,
        label=np.full(n_rows, -1, dtype=np.int8),
        row_start=0,
        meta=_egostitch_precision_meta(n_rows),
    )
    artifact = score_universe.load_scores(output)
    score_universe.validate_score_precision(artifact.logit, meta=artifact.meta, label=str(output))
    resolution = artifact.meta["score_resolution"]
    assert isinstance(resolution, dict)
    assert resolution["n_unique"] == 16
    assert resolution["unique_fraction"] == pytest.approx(0.0016)


def test_merge_recomputes_egostitch_resolution_diagnostics(tmp_path: Path) -> None:
    shard_paths = [tmp_path / "s0.npz", tmp_path / "s1.npz"]
    for shard, values in enumerate(
        (
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        )
    ):
        meta = _egostitch_precision_meta(4)
        score_universe.save_scores(
            shard_paths[shard],
            node_ids=["node_a", "node_b"],
            u_idx=np.zeros(2, dtype=np.int32),
            v_idx=np.ones(2, dtype=np.int32),
            logit=values,
            label=np.full(2, -1, dtype=np.int8),
            row_start=2 * shard,
            meta=meta,
        )

    output = tmp_path / "merged.npz"
    score_universe.main(
        ["merge", "--inputs", *[str(path) for path in shard_paths], "--output", str(output)]
    )
    merged = score_universe.load_scores(output)
    assert merged.meta["score_resolution"] == score_universe.score_resolution_diagnostics(
        merged.logit
    )
