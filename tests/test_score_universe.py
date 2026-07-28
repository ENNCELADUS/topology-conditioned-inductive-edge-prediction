"""Tests for `src.score_universe` (scoring CLI, sharding, and the scores artifact).

All tests are synthetic: tiny fake feature roots and pair files built under
`tmp_path`. No dependence on the real `data/` package.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
from src.data.feature_stats import compute_feature_stats
from src.data.features import FeatureStore, build_f0_matrix
from src.data.grounding import build_grounding_pool
from src.data.packed_features import PackedFeatureTable, build_packed_features
from src.model.B0 import V3_1
from src.model.egostitch.conditioning import GatedCrossAttention
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import E2ENodeState, E2EPairContext, EgoStitchE2E

INPUT_DIM = 4


def _write_feature_store(
    root: Path, node_tokens: dict[str, torch.Tensor], *, input_dim: int = INPUT_DIM
) -> None:
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
            {"format": "torch_pt_per_node", "input_dim": input_dim, "max_sequence_length": 1024}
        )
    )
    (root / "index.json").write_text(json.dumps(index))


def _data_root_with_features(
    tmp_path: Path, node_tokens: dict[str, torch.Tensor], *, input_dim: int = INPUT_DIM
) -> Path:
    """Build a `<data_root>/features/frozen_node_features_1024/` package under tmp_path."""
    data_root = tmp_path / "data"
    features_root = data_root / "features" / "frozen_node_features_1024"
    _write_feature_store(features_root, node_tokens, input_dim=input_dim)
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


def test_load_e2e_checkpoint_with_current_scaffold_without_relational_head(tmp_path: Path) -> None:
    legacy_config = dict(_TINY_E2E_CONFIG)
    legacy_config["w_rel"] = 0.0
    source_model = score_universe.build_model("egostitch_e2e", legacy_config)
    checkpoint_config = dict(legacy_config)
    checkpoint_config.pop("w_rel")
    checkpoint_path = tmp_path / "pre-rev31-e2e.pt"
    _write_checkpoint(
        checkpoint_path,
        model=source_model,
        model_family="egostitch_e2e",
        model_config=checkpoint_config,
    )

    loaded_model, model_family, checkpoint_id = score_universe._load_checkpoint(checkpoint_path)

    assert model_family == "egostitch_e2e"
    assert isinstance(loaded_model, EgoStitchE2E)
    assert loaded_model.rel_head is None
    assert checkpoint_id == score_universe._checkpoint_id(source_model.state_dict())
    for name, expected in source_model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[name], expected)


def test_load_pre_rev31_e2e_checkpoint_rejects_legacy_scaffold_shape(tmp_path: Path) -> None:
    legacy_config = dict(_TINY_E2E_CONFIG)
    legacy_config["w_rel"] = 0.0
    source_model = score_universe.build_model("egostitch_e2e", legacy_config)
    legacy_state = dict(source_model.state_dict())
    legacy_state["ste.embed.weight"] = legacy_state["ste.embed.weight"][:, :9].clone()
    for layer in range(cast(int, legacy_config["ste_layers"])):
        legacy_state.pop(f"ste.layers.{layer}.msg.3.weight")
        legacy_state.pop(f"ste.layers.{layer}.msg.3.bias")
    checkpoint_config = dict(legacy_config)
    checkpoint_config.pop("w_rel")
    checkpoint_path = tmp_path / "pre-rev31-e2e.pt"
    torch.save(
        {
            "model_state": legacy_state,
            "model_family": "egostitch_e2e",
            "model_config": checkpoint_config,
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            "rev-3\\.1 scaffold change.*FEAT_DIM 9 to 11.*EDGE_TYPES 3 to 4.*"
            "spec section 14\\.4\\.2.*pre-rev-3\\.1 e2e checkpoints are not loadable.*"
            "pre-rev-3\\.1 commit"
        ),
    ):
        score_universe._load_checkpoint(checkpoint_path)


def test_e2e_checkpoint_restores_the_feature_standardization(tmp_path: Path) -> None:
    """A checkpoint's registered mu/sigma buffers must ride the strict load.

    Scoring never sees the V_fit universe: the only way `normalize_features`
    can reproduce training-time behavior is if `set_feature_stats`'s buffers
    (`feature_mu`, `feature_sigma`, `feature_stats_ready`,
    `feature_stats_digest`) are part of `state_dict()` and survive
    `_load_checkpoint`'s strict `load_state_dict` untouched.
    """
    gen = np.random.default_rng(0)
    cfg = E2EConfig(feature_standardization="zscore_vfit_v1")
    trained = EgoStitchE2E(cfg)
    rows = (30.0 + 4.0 * gen.standard_normal((32, trained.generator_cfg.input_dim))).astype(
        np.float32
    )
    stats = compute_feature_stats(rows, [f"n{i}" for i in range(32)])
    trained.set_feature_stats(stats)

    checkpoint_path = tmp_path / "best.pt"
    _write_checkpoint(
        checkpoint_path,
        model=trained,
        model_family="egostitch_e2e",
        model_config={**asdict(cfg), "feature_stats_sha256": stats.digest},
    )

    restored, model_family, _ = score_universe._load_checkpoint(checkpoint_path)
    assert model_family == "egostitch_e2e"
    assert isinstance(restored, EgoStitchE2E)
    assert restored.feature_stats_digest_hex == stats.digest
    x = torch.randn(4, trained.generator_cfg.input_dim)
    torch.testing.assert_close(
        restored.generator.normalize_features(x), trained.generator.normalize_features(x)
    )


def test_rev31_checkpoints_still_load_as_row_layernorm(tmp_path: Path) -> None:
    """A checkpoint predating `feature_standardization` must not silently switch modes.

    `e2e_checkpoint_config` backfills the missing key as `"row_layernorm"`
    (never the current rev-3.2 default), so a rev-3.1 checkpoint keeps its
    original stateless per-row transform under strict loading.
    """
    legacy_cfg = E2EConfig(feature_standardization="row_layernorm")
    legacy = EgoStitchE2E(legacy_cfg)
    legacy_config = {
        key: value
        for key, value in asdict(legacy_cfg).items()
        if key not in {"feature_standardization", "feature_stats_sha256"}
    }
    checkpoint_path = tmp_path / "pre-rev31-e2e.pt"
    _write_checkpoint(
        checkpoint_path,
        model=legacy,
        model_family="egostitch_e2e",
        model_config=legacy_config,
    )

    restored, _, _ = score_universe._load_checkpoint(checkpoint_path)
    assert isinstance(restored, EgoStitchE2E)
    assert restored.generator.feature_standardization == "row_layernorm"
    for name, expected in legacy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_default_e2e_grounding_cache_path_is_namespaced_by_pool_configuration(
    tmp_path: Path,
) -> None:
    f0_cache = tmp_path / "f0_matrix.pt"
    universe = ["node-a", "node-b", "node-c"]
    full = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=50, node_ids=universe
    )
    cosine_pool = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=20, node_ids=universe
    )
    same_pool = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=50, node_ids=reversed(universe)
    )
    other_universe = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=50, node_ids=["node-a", "node-b", "node-d"]
    )

    assert full != cosine_pool
    assert full == same_pool
    assert full != other_universe
    assert "cosine_topk_v1" in full.name
    assert "n50" in full.name


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


def test_merge_aggregates_concurrent_score_profiles(tmp_path: Path) -> None:
    shard0 = tmp_path / "s0.npz"
    shard1 = tmp_path / "s1.npz"
    _write_fake_shard(shard0, row_start=0, num_rows=4, n_rows=2)
    _write_fake_shard(shard1, row_start=2, num_rows=4, n_rows=2)
    for path, wall in ((shard0, 3.0), (shard1, 5.0)):
        raw = score_universe._load_shard(path)
        meta = dict(raw.meta)
        meta["score_profile"] = {
            "wall_seconds": wall,
            "rows": 2,
            "unique_nodes": len(raw.node_ids),
            "measurement": "single_process_compute_wall_seconds",
        }
        score_universe.save_scores(
            path,
            node_ids=raw.node_ids,
            u_idx=raw.u_idx,
            v_idx=raw.v_idx,
            logit=raw.logit,
            label=raw.label,
            row_start=raw.row_start,
            meta=meta,
        )

    merged = score_universe.merge_scores([shard0, shard1])

    assert merged.meta["score_profile"] == {
        "wall_seconds": 5.0,
        "rows": 4,
        "unique_nodes": len(merged.node_ids),
        "measurement": "max_concurrent_shard_compute_wall_seconds",
    }


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
    assert isinstance(model.feature_norm, nn.Identity)


def test_legacy_egostitch_cached_scoring_matches_direct_forward(tmp_path: Path) -> None:
    data_root, _, _, _ = _egostitch_setup(tmp_path)
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    model = score_universe.build_model("egostitch", dict(_TINY_EGOSTITCH_CONFIG))
    model.eval()
    pairs = list(_EGOSTITCH_PAIRS)
    s0 = np.linspace(-1.0, 1.0, len(pairs), dtype=np.float32)
    cached = score_universe._score_egostitch(
        model,
        pairs,
        store,
        device=torch.device("cpu"),
        amp="off",
        batch_pairs=3,
        f0_cache=tmp_path / "legacy-f0.pt",
        grounding_cache=tmp_path / "legacy-grounding.npz",
        s0_logits=s0,
    )

    node_ids = sorted({node for pair in pairs for node in pair})
    matrix, index = build_f0_matrix(store, node_ids, cache_path=None)
    grounding = build_grounding_pool(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_ids,
        n_ground=model.config.n_ground,
        role_universe="test",
        cache_path=None,
    )
    pool_rows = torch.tensor(
        [[index[neighbor] for neighbor in grounding[node]] for node in node_ids],
        dtype=torch.long,
    )
    direct: list[float] = []
    with torch.inference_mode():
        for row, (u, v) in enumerate(pairs):
            u_row, v_row = index[u], index[v]
            batch = {
                "x_i": matrix[u_row].unsqueeze(0),
                "ground_i": matrix[pool_rows[u_row]].unsqueeze(0),
                "s0": torch.tensor([s0[row]]),
            }
            if u != v:
                batch.update(
                    {
                        "x_j": matrix[v_row].unsqueeze(0),
                        "ground_j": matrix[pool_rows[v_row]].unsqueeze(0),
                    }
                )
            direct.append(float(model(batch)["logits"].item()))

    np.testing.assert_allclose(cached, np.asarray(direct, dtype=np.float32), rtol=1e-5, atol=2e-6)


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


# --------------------------------------------------------------------------- egostitch_e2e scoring
# (design rev 3 four-logit decomposition; Task 14)

_TINY_E2E_CONFIG: dict[str, object] = {
    "d_model": 32,
    "encoder_layers": 1,
    "cross_attn_layers": 2,
    "n_heads": 4,
    "n_inj": 1,
    "ste_dim": 16,
    "ste_layers": 2,
    "xattn_heads": 4,
    # Explicit rather than relying on `E2EConfig`'s rev-3.2 default
    # (`"zscore_vfit_v1"`): these tests build checkpoints through
    # `build_model`/`_load_checkpoint` without ever calling
    # `set_feature_stats`, and the registered mode requires those buffers to
    # be populated before a forward pass. Pinning the stateless rev-3.1
    # transform keeps this shared fixture representative of what it always
    # tested (scoring/merge/precision plumbing, not feature standardization)
    # -- the registered-stats path is covered separately in
    # `test_e2e_checkpoint_restores_the_feature_standardization`.
    "feature_standardization": "row_layernorm",
}
# EgoStitchE2E's internal Stage-1 generator keeps its own pinned spec default
# (EgoStitchConfig().input_dim, spec Sec 13) regardless of E2EConfig, so the
# feature store backing these tests must use that exact node-token dimension.
_E2E_NODE_DIM = 1536
_E2E_NODES = [f"n{i}" for i in range(4)]
# Repeated endpoints exercise cache reuse; dedicated tests add self-pair rows.
_E2E_PAIRS = [("n0", "n1"), ("n1", "n2"), ("n0", "n2")]


def _egostitch_e2e_setup(
    tmp_path: Path, *, model_config: dict[str, object] | None = None
) -> tuple[Path, Path, Path]:
    """Feature store + checkpoint + pairs TSV for the `egostitch_e2e` family."""
    torch.manual_seed(0)
    node_tokens = {node: torch.randn(3 + i, _E2E_NODE_DIM) for i, node in enumerate(_E2E_NODES)}
    data_root = _data_root_with_features(tmp_path, node_tokens, input_dim=_E2E_NODE_DIM)

    config = dict(_TINY_E2E_CONFIG if model_config is None else model_config)
    model = score_universe.build_model("egostitch_e2e", config)
    checkpoint_path = tmp_path / "egostitch_e2e.pt"
    _write_checkpoint(
        checkpoint_path,
        model=model,
        model_family="egostitch_e2e",
        model_config=config,
    )

    pairs_path = tmp_path / "pairs.tsv"
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in _E2E_PAIRS))
    return data_root, checkpoint_path, pairs_path


def _egostitch_e2e_score_args(
    tmp_path: Path, data_root: Path, checkpoint: Path, pairs: Path, output: Path
) -> list[str]:
    evidence_artifact = tmp_path / "synthetic-file-score-evidence.json"
    evidence_artifact.write_text('{"scope": "synthetic-nonheldout"}\n', encoding="utf-8")
    evidence_record = {
        "path": str(evidence_artifact),
        "sha256": score_universe._file_sha256(evidence_artifact),
    }
    preregistration = tmp_path / "synthetic-file-score-preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "status": "BINDING",
                "binding_evidence": {
                    "schema_version": "egostitch_e2e_binding_evidence_v1",
                    "configs": {"synthetic": evidence_record},
                    "parameter_group_manifests": evidence_record,
                    "packs_and_validation_manifests": evidence_record,
                    "qualification_attempts": evidence_record,
                    "boundary_access_audit": evidence_record,
                    "runtime_and_peak_memory": evidence_record,
                },
            }
        ),
        encoding="utf-8",
    )
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
        "--preregistration",
        str(preregistration),
        "--token-budget",
        "8192",
        "--f0-cache",
        str(tmp_path / "f0_cache.pt"),
        "--device",
        "cpu",
    ]


def test_build_model_egostitch_e2e_round_trips() -> None:
    from src.model.egostitch.e2e_model import EgoStitchE2E

    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchE2E)
    assert model.cfg.d_model == 32


def test_egostitch_e2e_cli_scores_four_logit_decomposition(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    assert artifact.meta["model_family"] == "egostitch_e2e"
    assert len(artifact.logit) == len(_E2E_PAIRS)
    assert artifact.logit.dtype == np.float32
    assert artifact.f_logit is not None
    assert artifact.pair_content is not None
    assert artifact.pair_topology is not None
    for arr in (artifact.f_logit, artifact.pair_content, artifact.pair_topology):
        assert arr.dtype == np.float32
        assert arr.shape == artifact.logit.shape
        assert bool(np.isfinite(arr).all())

    assert artifact.meta["score_precision"] == {
        "contract": "egostitch_e2e_pair_fp32_v1",
        "pair_compute_dtype": "float32",
        "pair_autocast": False,
        "logit_storage_dtype": "float32",
    }

    resolution = artifact.meta["score_resolution"]
    assert isinstance(resolution, dict)
    assert set(resolution) == {"full", "f_logit", "pair_content", "pair_topology"}
    for name, arr in (
        ("full", artifact.logit),
        ("f_logit", artifact.f_logit),
        ("pair_content", artifact.pair_content),
        ("pair_topology", artifact.pair_topology),
    ):
        assert resolution[name] == score_universe.score_resolution_diagnostics(arr)

    # Zero-init sanity check (design rev 3 Sec 3.4/3.5): GatedCrossAttention's
    # gate parameter starts at exactly zero, so the topo/content pathways are
    # an exact-identity bypass regardless of which one a given arm nulls out.
    # All four logits must therefore be bit-for-bit identical at init.
    np.testing.assert_array_equal(artifact.logit, artifact.f_logit)
    np.testing.assert_array_equal(artifact.logit, artifact.pair_content)
    np.testing.assert_array_equal(artifact.logit, artifact.pair_topology)


def test_egostitch_e2e_permanent_null_publishes_active_primary_logit(tmp_path: Path) -> None:
    for permanent_null, active_key in (("all_head", "f_logit"), ("content_head", "pair_topology")):
        config = {**_TINY_E2E_CONFIG, "permanent_null": permanent_null}
        data_root, checkpoint, pairs = _egostitch_e2e_setup(
            tmp_path / permanent_null, model_config=config
        )
        _enable_e2e_conditioning_gates(checkpoint)
        output = tmp_path / f"{permanent_null}.npz"
        score_universe.main(
            _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output)
        )
        artifact = score_universe.load_scores(output)
        active = getattr(artifact, active_key)
        assert active is not None
        np.testing.assert_array_equal(artifact.logit, active)
        assert artifact.full_logit is not None
        assert not np.array_equal(artifact.logit, artifact.full_logit)
        resolution = cast(dict[str, object], artifact.meta["score_resolution"])
        assert resolution["full"] == score_universe.score_resolution_diagnostics(
            artifact.full_logit
        )
        score_universe.validate_artifact_precision(artifact)
        assert artifact.meta["permanent_null"] == permanent_null


@pytest.mark.parametrize("permanent_null", ["all_head", "content_head"])
def test_egostitch_e2e_null_arm_merge_preserves_true_full_logit(
    tmp_path: Path, permanent_null: str
) -> None:
    """Null-arm shards retain the distinct true full arm through merge/load."""
    config = {**_TINY_E2E_CONFIG, "permanent_null": permanent_null}
    data_root, checkpoint, pairs = _egostitch_e2e_setup(
        tmp_path / permanent_null, model_config=config
    )
    _enable_e2e_conditioning_gates(checkpoint)
    output = tmp_path / f"{permanent_null}.npz"
    for shard in range(2):
        score_universe.main(
            [
                *_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output),
                "--shard",
                str(shard),
                "--num-shards",
                "2",
            ]
        )
    shard_paths = [
        output.with_name(f"{output.stem}.shard-{shard}{output.suffix}") for shard in range(2)
    ]
    merged_path = tmp_path / f"{permanent_null}-merged.npz"
    score_universe.main(
        ["merge", "--inputs", *(str(path) for path in shard_paths), "--output", str(merged_path)]
    )

    merged = score_universe.load_scores(merged_path)
    assert merged.full_logit is not None
    assert not np.array_equal(merged.logit, merged.full_logit)
    score_universe.validate_artifact_precision(merged)


def test_egostitch_e2e_cli_is_deterministic(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    out_a, out_b = tmp_path / "a.npz", tmp_path / "b.npz"
    for output in (out_a, out_b):
        score_universe.main(
            _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output)
        )
    a, b = score_universe.load_scores(out_a), score_universe.load_scores(out_b)
    np.testing.assert_array_equal(a.logit, b.logit)
    assert a.f_logit is not None and b.f_logit is not None
    np.testing.assert_array_equal(a.f_logit, b.f_logit)


def _enable_e2e_conditioning_gates(checkpoint: Path) -> None:
    """Make the synthetic checkpoint sensitive to topology perturbations."""
    payload = torch.load(checkpoint, weights_only=False)
    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchE2E)
    model.load_state_dict(payload["model_state"])
    for module in [*model.trunk.topo_xattn, *model.trunk.cont_xattn]:
        assert isinstance(module, GatedCrossAttention)
        module.gate.data.fill_(1.0)
    payload["model_state"] = model.state_dict()
    torch.save(payload, checkpoint)


def _direct_e2e_control_scores(
    tmp_path: Path,
    data_root: Path,
    checkpoint: Path,
    pairs: list[tuple[str, str]],
    *,
    control: str,
    universe_pairs: list[tuple[str, str]] | None = None,
    row_start: int = 0,
) -> dict[str, np.ndarray]:
    payload = torch.load(checkpoint, weights_only=False)
    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchE2E)
    model.load_state_dict(payload["model_state"])
    model.eval()
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    return score_universe._score_egostitch_e2e(
        model,
        pairs,
        store,
        device=torch.device("cpu"),
        token_budget=4096,
        f0_cache=tmp_path / "direct-f0.npz",
        grounding_cache=tmp_path / "direct-grounding.npz",
        scaffold_control=control,
        universe_pairs=universe_pairs,
        row_start=row_start,
    )


def test_egostitch_e2e_rejects_superseded_v2_scaffold_shuffle(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    with pytest.raises(ValueError, match="14\\.4\\.5"):
        score_universe.main(
            [
                *_egostitch_e2e_score_args(
                    tmp_path, data_root, checkpoint, pairs, tmp_path / "rejected.npz"
                ),
                "--scaffold-control",
                "shuffle_within_pair",
            ]
        )


@pytest.mark.parametrize("control", ["shuffle_within_pair_v3", "rewire_checkerboard_v1"])
def test_egostitch_e2e_scaffold_control_is_ab_ba_and_shard_invariant(
    tmp_path: Path, control: str
) -> None:
    data_root, checkpoint, _ = _egostitch_e2e_setup(tmp_path)
    _enable_e2e_conditioning_gates(checkpoint)
    forward = _direct_e2e_control_scores(
        tmp_path, data_root, checkpoint, _E2E_PAIRS, control=control
    )
    reversed_pairs = [(v, u) for u, v in _E2E_PAIRS]
    reverse = _direct_e2e_control_scores(
        tmp_path, data_root, checkpoint, reversed_pairs, control=control
    )
    np.testing.assert_allclose(
        forward["full"],
        reverse["full"],
        atol=1e-6,
        rtol=1e-6,
    )

    shards: list[np.ndarray] = []
    for shard in range(3):
        shard_scores = _direct_e2e_control_scores(
            tmp_path,
            data_root,
            checkpoint,
            [_E2E_PAIRS[shard]],
            control=control,
            universe_pairs=_E2E_PAIRS,
            row_start=shard,
        )
        shards.append(shard_scores["full"])
    np.testing.assert_array_equal(forward["full"], np.concatenate(shards))

    reordered = [_E2E_PAIRS[2], _E2E_PAIRS[0], _E2E_PAIRS[1]]
    reordered_scores = _direct_e2e_control_scores(
        tmp_path, data_root, checkpoint, reordered, control=control
    )
    ordered_scores = dict(zip(_E2E_PAIRS, forward["full"], strict=True))
    restored = np.asarray([ordered_scores[pair] for pair in reordered], dtype=np.float32)
    np.testing.assert_array_equal(reordered_scores["full"], restored)


@pytest.mark.parametrize("control", ["shuffle_within_pair_v3", "rewire_checkerboard_v1"])
def test_egostitch_e2e_scaffold_control_uses_multi_pair_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    data_root, checkpoint, _ = _egostitch_e2e_setup(tmp_path)
    _enable_e2e_conditioning_gates(checkpoint)
    many_pairs = _E2E_PAIRS * 8
    calls: list[int] = []
    original = EgoStitchE2E.decompose_pair_context

    def _spy(self: EgoStitchE2E, context: E2EPairContext) -> dict[str, torch.Tensor]:
        calls.append(context.encoded_a.shape[0])
        return original(self, context)

    monkeypatch.setattr(EgoStitchE2E, "decompose_pair_context", _spy)
    _direct_e2e_control_scores(
        tmp_path,
        data_root,
        checkpoint,
        many_pairs,
        control=control,
    )
    assert max(calls) > 1
    assert len(calls) < len(many_pairs)


def test_egostitch_e2e_scoring_caches_unique_nodes_and_reuses_pair_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, checkpoint, _ = _egostitch_e2e_setup(tmp_path)
    pairs_path = tmp_path / "many-cache.tsv"
    many_pairs = (_E2E_PAIRS + [("n0", "n0")]) * 12
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in many_pairs))

    encoded_rows = 0
    context_calls = 0
    head_calls = 0
    original_encode = cast(Callable[..., E2ENodeState], EgoStitchE2E.encode_node_state)
    original_context = cast(
        Callable[..., E2EPairContext], EgoStitchE2E.build_pair_context_from_states
    )
    original_head = cast(Callable[..., torch.Tensor], EgoStitchE2E.score_pair_context)

    def encode_spy(self: EgoStitchE2E, *args: object, **kwargs: object) -> E2ENodeState:
        nonlocal encoded_rows
        encoded_rows += cast(torch.Tensor, args[0]).shape[0]
        return original_encode(self, *args, **kwargs)

    def context_spy(self: EgoStitchE2E, *args: object, **kwargs: object) -> E2EPairContext:
        nonlocal context_calls
        context_calls += 1
        return original_context(self, *args, **kwargs)

    def head_spy(self: EgoStitchE2E, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal head_calls
        head_calls += 1
        return original_head(self, *args, **kwargs)

    monkeypatch.setattr(EgoStitchE2E, "encode_node_state", encode_spy)
    monkeypatch.setattr(EgoStitchE2E, "build_pair_context_from_states", context_spy)
    monkeypatch.setattr(EgoStitchE2E, "score_pair_context", head_spy)
    score_universe.main(
        _egostitch_e2e_score_args(
            tmp_path, data_root, checkpoint, pairs_path, tmp_path / "cached.npz"
        )
    )
    assert encoded_rows == len({node for pair in many_pairs for node in pair})
    assert 0 < context_calls < len(many_pairs)
    assert head_calls == 4 * context_calls


def test_egostitch_e2e_cached_scoring_matches_uncached_decomposition(tmp_path: Path) -> None:
    """CPU cache/crop/repad scoring matches a direct production decomposition."""
    data_root, checkpoint, _ = _egostitch_e2e_setup(tmp_path)
    _enable_e2e_conditioning_gates(checkpoint)
    pairs = [("n0", "n3"), ("n3", "n3"), ("n1", "n2"), ("n0", "n0")]
    pairs_path = tmp_path / "mixed-self.tsv"
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in pairs))
    output = tmp_path / "cached-equivalence.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs_path, output)
    )
    cached = score_universe.load_scores(output)

    payload = cast(dict[str, object], torch.load(checkpoint, map_location="cpu"))
    model = score_universe.build_model(
        "egostitch_e2e", cast(dict[str, object], payload["model_config"])
    )
    assert isinstance(model, EgoStitchE2E)
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state"]))
    model.eval()
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    node_ids = sorted({node for pair in pairs for node in pair})
    matrix, index = build_f0_matrix(store, node_ids, cache_path=None)
    n_ground = min(model.generator_cfg.n_ground, len(node_ids) - 1)
    grounding = build_grounding_pool(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_ids,
        n_ground=n_ground,
        role_universe="test",
        cache_path=tmp_path / "direct-grounding.npz",
    )
    pool_rows = torch.tensor(
        [[index[neighbor] for neighbor in grounding[node]] for node in node_ids],
        dtype=torch.long,
    )
    rows_a = torch.tensor([index[u] for u, _ in pairs], dtype=torch.long)
    rows_b = torch.tensor([index[v] for _, v in pairs], dtype=torch.long)
    tokens_a = [store.load_tokens(u) for u, _ in pairs]
    tokens_b = [store.load_tokens(v) for _, v in pairs]
    batch = {
        "emb_a": torch.nn.utils.rnn.pad_sequence(tokens_a, batch_first=True),
        "emb_b": torch.nn.utils.rnn.pad_sequence(tokens_b, batch_first=True),
        "len_a": torch.tensor([len(tokens) for tokens in tokens_a], dtype=torch.long),
        "len_b": torch.tensor([len(tokens) for tokens in tokens_b], dtype=torch.long),
        "x_a": matrix.index_select(0, rows_a),
        "x_b": matrix.index_select(0, rows_b),
        "ground_a": matrix[pool_rows[rows_a]],
        "ground_b": matrix[pool_rows[rows_b]],
        "ground_id_a": pool_rows[rows_a],
        "ground_id_b": pool_rows[rows_b],
        "is_self": torch.tensor([u == v for u, v in pairs], dtype=torch.bool),
    }
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        direct = model.decompose(batch)

    assert cached.f_logit is not None
    assert cached.pair_content is not None
    assert cached.pair_topology is not None
    cached_arrays = {
        "full": cached.logit,
        "f_logit": cached.f_logit,
        "pair_content": cached.pair_content,
        "pair_topology": cached.pair_topology,
    }
    assert not np.array_equal(cached_arrays["full"], cached_arrays["f_logit"])
    for name, expected in direct.items():
        actual = cached_arrays[name]
        np.testing.assert_allclose(
            actual,
            expected.detach().float().numpy(),
            rtol=1e-5,
            atol=2e-6,
        )


def test_merge_rejects_conflicting_scaffold_control_provenance(tmp_path: Path) -> None:
    base_meta = {
        "checkpoint_id": "abc123abc123abcd",
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": "breadth_first",
        "num_rows": 2,
        "created_utc": "2026-07-17T00:00:00+00:00",
        "torch_version": str(torch.__version__),
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "scaffold_control": {
            "mode": "shuffle_within_pair",
            "seed": 0,
            "keying": "canonical_pair_v1",
        },
        "permanent_null": "none",
        "primary_logit": "full",
    }

    def write_shard(path: Path, row_start: int, control: dict[str, object]) -> None:
        meta = {**base_meta, "scaffold_control": control}
        values = np.array([float(row_start)], dtype=np.float32)
        score_universe.save_scores(
            path,
            node_ids=["a", "b"],
            u_idx=np.array([0], dtype=np.int32),
            v_idx=np.array([1], dtype=np.int32),
            logit=values,
            label=np.full(1, -1, dtype=np.int8),
            row_start=row_start,
            meta=meta,
            f_logit=values,
            pair_content=values,
            pair_topology=values,
        )

    left = tmp_path / "left.npz"
    write_shard(left, 0, cast(dict[str, object], base_meta["scaffold_control"]))
    for suffix, control in (
        ("mode", {"mode": "none", "seed": 0, "keying": "canonical_pair_v1"}),
        ("seed", {"mode": "shuffle_within_pair", "seed": 1, "keying": "canonical_pair_v1"}),
        ("keying", {"mode": "shuffle_within_pair", "seed": 0, "keying": "other"}),
    ):
        right = tmp_path / f"{suffix}.npz"
        write_shard(right, 1, control)
        with pytest.raises(ValueError, match="scaffold_control"):
            score_universe.merge_scores([left, right])

    for key, value in (("permanent_null", "all_head"), ("primary_logit", "f_logit")):
        right = tmp_path / f"{key}.npz"
        write_shard(right, 1, cast(dict[str, object], base_meta["scaffold_control"]))
        with np.load(right, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"][()]))
            meta[key] = value
            if key == "permanent_null":
                meta["primary_logit"] = "f_logit"
            else:
                meta["permanent_null"] = "all_head"
            score_universe.save_scores(
                right,
                node_ids=[str(node) for node in data["node_ids"].tolist()],
                u_idx=data["u_idx"],
                v_idx=data["v_idx"],
                logit=data["logit"],
                label=data["label"],
                row_start=int(data["row_start"][()]),
                meta=meta,
                f_logit=data["f_logit"],
                pair_content=data["pair_content"],
                pair_topology=data["pair_topology"],
                full_logit=data["logit"],
            )
        with pytest.raises(ValueError, match=key):
            score_universe.merge_scores([left, right])


def test_egostitch_e2e_merge_preserves_four_arrays_in_manifest_order(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    for shard in range(2):
        score_universe.main(
            [
                *_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output),
                "--shard",
                str(shard),
                "--num-shards",
                "2",
            ]
        )
    shard_paths = [
        output.with_name(f"{output.stem}.shard-{shard}{output.suffix}") for shard in range(2)
    ]
    merged_path = tmp_path / "merged.npz"
    score_universe.main(
        ["merge", "--inputs", *(str(path) for path in shard_paths), "--output", str(merged_path)]
    )

    unsharded_path = tmp_path / "unsharded.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, unsharded_path)
    )

    merged = score_universe.load_scores(merged_path)
    unsharded = score_universe.load_scores(unsharded_path)
    assert merged.f_logit is not None
    assert merged.pair_content is not None
    assert merged.pair_topology is not None
    assert unsharded.f_logit is not None
    assert unsharded.pair_content is not None
    assert unsharded.pair_topology is not None
    assert len(merged.f_logit) == len(_E2E_PAIRS)
    assert len(merged.pair_content) == len(_E2E_PAIRS)
    assert len(merged.pair_topology) == len(_E2E_PAIRS)

    # Merged rows are remapped onto the sorted node-id union but otherwise keep
    # each row's own quadruple aligned; comparing by pair identity against the
    # unsharded run confirms manifest order (row <-> quadruple) is preserved.
    # Ordinary sharded scoring may use different pair-batch shapes, so compare
    # within the same explicit fp32 tolerance as cached-vs-direct decomposition.
    merged_index = {pair: row for row, pair in enumerate(merged.pairs())}
    unsharded_index = {pair: row for row, pair in enumerate(unsharded.pairs())}
    assert set(merged_index) == set(unsharded_index) == set(_E2E_PAIRS)
    for pair in _E2E_PAIRS:
        m_row, u_row = merged_index[pair], unsharded_index[pair]
        assert merged.logit[m_row] == pytest.approx(unsharded.logit[u_row], rel=1e-5, abs=2e-6)
        assert merged.f_logit[m_row] == pytest.approx(unsharded.f_logit[u_row], rel=1e-5, abs=2e-6)
        assert merged.pair_content[m_row] == pytest.approx(
            unsharded.pair_content[u_row], rel=1e-5, abs=2e-6
        )
        assert merged.pair_topology[m_row] == pytest.approx(
            unsharded.pair_topology[u_row], rel=1e-5, abs=2e-6
        )


# --------------------------------------------------------------------------- artifact-aware
# score-precision validation entry point (Task-14 review follow-up)


def test_validate_artifact_precision_passes_on_well_formed_e2e_artifact(tmp_path: Path) -> None:
    """A caller with only a `ScoresArtifact` (no separate extra_arrays) validates cleanly."""
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    score_universe.validate_artifact_precision(artifact, label=str(output))


def test_validate_artifact_precision_raises_on_corrupted_e2e_artifact(tmp_path: Path) -> None:
    """A genuinely broken decomposition array (wrong dtype) is still caught."""
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    assert artifact.pair_content is not None
    corrupted = score_universe.ScoresArtifact(
        node_ids=artifact.node_ids,
        u_idx=artifact.u_idx,
        v_idx=artifact.v_idx,
        logit=artifact.logit,
        label=artifact.label,
        meta=artifact.meta,
        f_logit=artifact.f_logit,
        pair_content=artifact.pair_content.astype(np.float64),  # dtype broken on purpose
        pair_topology=artifact.pair_topology,
    )
    with pytest.raises(ValueError, match="must be stored as float32"):
        score_universe.validate_artifact_precision(corrupted, label="corrupted")


def test_validate_score_precision_old_idiom_on_e2e_artifact_points_to_new_entry_point(
    tmp_path: Path,
) -> None:
    """The old generic idiom on an egostitch_e2e artifact still raises, with a pointer.

    `validate_score_precision(artifact.logit, ...)` (as used by
    `src.experiments.g1_hardened_e2` for B0-family artifacts) still raises for an
    egostitch_e2e artifact — the four-array contract can't be checked without the
    extra arrays — but the error now points callers at validate_artifact_precision
    instead of reading as a false "artifact is missing arrays" report.
    """
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    with pytest.raises(ValueError, match="validate_artifact_precision"):
        score_universe.validate_score_precision(
            artifact.logit, meta=artifact.meta, label=str(output)
        )


# --------------------------------------------------------------------------- Task 13b:
# grounded-identity-match real grounding wiring in the e2e scorer


def test_egostitch_e2e_scorer_supplies_grounding_batch_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_score_egostitch_e2e` assembles real grounding-pool batch keys per pair.

    Spies on `EgoStitchE2E.encode_node_state` to capture the node-cache inputs,
    asserting the required ``ground_a``/``ground_b`` tensors and optional
    ``ground_id_a``/``ground_id_b`` identifiers are present with the expected
    shapes and dtypes for every scored batch.
    """
    from src.model.egostitch.config import E2EConfig
    from src.model.egostitch.e2e_model import EgoStitchE2E

    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"

    captured: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    original_encode = EgoStitchE2E.encode_node_state

    def _spy_encode(
        self: EgoStitchE2E,
        emb: torch.Tensor,
        length: torch.Tensor,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> E2ENodeState:
        captured.append((ground, ground_ids))
        return original_encode(self, emb, length, x, ground, ground_ids)

    monkeypatch.setattr(EgoStitchE2E, "encode_node_state", _spy_encode)

    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    assert captured, "node cache was never encoded"
    node_universe = len({node for pair in _E2E_PAIRS for node in pair})
    # `_egostitch_e2e_setup` builds the checkpoint from `_TINY_E2E_CONFIG`,
    # which does not override `n_ground` -- so the registered value is
    # `E2EConfig()`'s rev-3.1 default (spec Sec 14.4.4), not the internal
    # generator's own pinned `EgoStitchConfig()` default.
    expected_n_ground = min(E2EConfig().n_ground, node_universe - 1)
    for ground, ground_ids in captured:
        assert ground is not None and ground_ids is not None
        rows = ground.shape[0]
        assert ground.shape == (rows, expected_n_ground, _E2E_NODE_DIM)
        assert ground_ids.shape == (rows, expected_n_ground)
        assert ground_ids.dtype == torch.int64


# --------------------------------------------------------------------------- Task 13b review
# follow-up: silent n_ground clamp on small universes


def test_egostitch_e2e_scorer_warns_on_n_ground_clamp(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`_score_egostitch_e2e` warns when the registered `n_ground` gets clamped.

    The tiny 3-node fixture (`_E2E_PAIRS` over `{n0, n1, n2}`) can only support
    `len(node_ids) - 1 == 2` grounding candidates, while `EgoStitchE2E`'s
    `generator_cfg.n_ground` registers `E2EConfig()`'s rev-3.1 default (spec
    Sec 14.4.4; `_egostitch_e2e_setup`'s `_TINY_E2E_CONFIG` does not override
    it). Every call here silently clamps the registered default down to 2;
    that must now be logged, stating both the registered and effective
    values.
    """
    from src.model.egostitch.config import E2EConfig

    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"

    with caplog.at_level("WARNING", logger="src.score_universe"):
        score_universe.main(
            _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output)
        )

    node_universe = len({node for pair in _E2E_PAIRS for node in pair})
    registered_n_ground = E2EConfig().n_ground
    expected_n_ground = min(registered_n_ground, node_universe - 1)
    assert expected_n_ground < registered_n_ground, "fixture must actually trigger the clamp"

    clamp_records = [r for r in caplog.records if "n_ground clamped" in r.getMessage()]
    assert clamp_records, f"expected an n_ground-clamp warning; got records: {caplog.records}"
    message = clamp_records[0].getMessage()
    assert str(registered_n_ground) in message
    assert str(expected_n_ground) in message
    assert str(node_universe) in message
