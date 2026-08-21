"""Tests for `src.score_universe` (scoring CLI, sharding, and the scores artifact).

All tests are synthetic: tiny fake feature roots and pair files built under
`tmp_path`. No dependence on the real `data/` package.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import cast

import networkx as nx
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
from src.data.val_region import ValRegionParams, derive_val_region_split, val_universe_arrays
from src.model.egostitch.classifier.b0_v31 import V3_1, B0V31PairClassifier, GatedCrossAttention
from src.model.egostitch.composite import E2ENodeState, E2EPairContext, EgoStitchModel
from src.model.egostitch.config import E2EConfig, GeneratorConfig
from src.model.egostitch.generator.egostitch import EgoStitchImagineGenerator

INPUT_DIM = 4


def test_gpu_table_gather_uses_batch_local_width() -> None:
    encoded_table = torch.arange(3 * 7 * 2, dtype=torch.float32).reshape(3, 7, 2)
    length_table = torch.tensor([7, 2, 3])
    rows = torch.tensor([1, 2])

    encoded, lengths = score_universe._gather_padded_node_rows(
        encoded_table, length_table, rows, width=3
    )

    assert encoded.shape == (2, 3, 2)
    torch.testing.assert_close(encoded, encoded_table[rows, :3])
    torch.testing.assert_close(lengths, torch.tensor([2, 3]))
    with pytest.raises(ValueError, match="shorter than a selected node state"):
        score_universe._gather_padded_node_rows(encoded_table, length_table, rows, width=2)


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
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "model_family": model_family,
        "model_config": model_config,
        "epoch": 0,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    torch.save(payload, path)


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


def test_packed_scoring_includes_pair_latent_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model_config = _tiny_v3_1_config()
    model_config["pair_latent_gen"] = {
        "z_dim": 2,
        "cond_dim": 4,
        "hidden": 4,
        "seed_count": 2,
        "seed_dim": 3,
        "mc_samples": 2,
        "kl_free_bits": 0.05,
    }
    model = score_universe.build_model("v3_1", model_config)
    assert isinstance(model, V3_1)
    assert model.pair_latent_gen is not None
    model.pair_latent_gen.alpha.data.fill_(0.75)
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


def test_e2e_checkpoint_missing_w_rel_fails_loudly_instead_of_backfilling(
    tmp_path: Path,
) -> None:
    """A config-incomplete checkpoint raises rather than being reconstructed.

    The rev-3.1 `e2e_checkpoint_config` backfills were deleted with the content
    path (design 2026-08-02 §11). A checkpoint whose recorded config omits
    `w_rel` would now take `E2EConfig`'s default and build a relational head the
    stored weights never had, so strict loading must reject it. Failing loudly
    is the point: silently reinterpreting a checkpoint under a config it was
    never trained with is the outcome this guards against.
    """
    # Three-component refactor (design 2026-08-02 Sec 8): `w_rel` moved from
    # the flat `E2EConfig` onto the nested `encoder` sub-config.
    legacy_config = {
        **_TINY_E2E_CONFIG,
        "encoder": {**cast(dict[str, object], _TINY_E2E_CONFIG["encoder"]), "w_rel": 0.0},
    }
    source_model = score_universe.build_model("egostitch_e2e", legacy_config)
    checkpoint_config = {
        **legacy_config,
        "encoder": {
            key: value
            for key, value in cast(dict[str, object], legacy_config["encoder"]).items()
            if key != "w_rel"
        },
    }
    checkpoint_path = tmp_path / "config-incomplete-e2e.pt"
    _write_checkpoint(
        checkpoint_path,
        model=source_model,
        model_family="egostitch_e2e",
        model_config=checkpoint_config,
    )

    with pytest.raises(RuntimeError, match="rel_head"):
        score_universe._load_checkpoint(checkpoint_path)


def test_load_pre_rev31_e2e_checkpoint_rejects_legacy_scaffold_shape(tmp_path: Path) -> None:
    legacy_config = {
        **_TINY_E2E_CONFIG,
        "encoder": {**cast(dict[str, object], _TINY_E2E_CONFIG["encoder"]), "w_rel": 0.0},
    }
    source_model = score_universe.build_model("egostitch_e2e", legacy_config)
    legacy_state = dict(source_model.state_dict())
    legacy_state["encoder.embed.weight"] = legacy_state["encoder.embed.weight"][:, :9].clone()
    # `ste_layers` was renamed `layers` and moved onto the nested `encoder`
    # sub-config (design 2026-08-02 Sec 8).
    encoder_config = cast(dict[str, object], legacy_config["encoder"])
    for layer in range(cast(int, encoder_config["layers"])):
        legacy_state.pop(f"encoder.layers.{layer}.msg.3.weight")
        legacy_state.pop(f"encoder.layers.{layer}.msg.3.bias")
    checkpoint_config = {
        **legacy_config,
        "encoder": {key: value for key, value in encoder_config.items() if key != "w_rel"},
    }
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
    # Three-component refactor (design 2026-08-02 Sec 8): `feature_standardization`
    # and `feature_stats_sha256` moved from the flat `E2EConfig` onto the nested
    # `generator` sub-config.
    cfg = E2EConfig(generator=GeneratorConfig(feature_standardization="zscore_vfit_v1"))
    trained = EgoStitchModel(cfg)
    rows = (30.0 + 4.0 * gen.standard_normal((32, trained.generator_cfg.input_dim))).astype(
        np.float32
    )
    stats = compute_feature_stats(rows, [f"n{i}" for i in range(32)])
    trained.set_feature_stats(stats)

    checkpoint_path = tmp_path / "best.pt"
    model_config = asdict(cfg)
    model_config["generator"] = {**model_config["generator"], "feature_stats_sha256": stats.digest}
    _write_checkpoint(
        checkpoint_path,
        model=trained,
        model_family="egostitch_e2e",
        model_config=model_config,
    )

    restored, model_family, _ = score_universe._load_checkpoint(checkpoint_path)
    assert model_family == "egostitch_e2e"
    assert isinstance(restored, EgoStitchModel)
    assert restored.feature_stats_digest_hex == stats.digest
    x = torch.randn(4, trained.generator_cfg.input_dim)
    # `model.generator` used to be the `EgoStitchStage1` itself; the
    # three-component refactor (design §5) moves it to `model.generator.stage1`
    # (the composite's `generator` attribute is now the `EgoStitchImagineGenerator`
    # wrapper). `normalize_features` lives on `EgoStitchStage1`, not the wrapper.
    # Both `restored`/`trained` are built with the default (real) generator, so
    # narrow `model.generator`'s registry-widened type explicitly (design
    # 2026-08-02 §12 P3 introduces `NullGenerator`, which has no `.stage1`).
    assert isinstance(restored.generator, EgoStitchImagineGenerator)
    assert isinstance(trained.generator, EgoStitchImagineGenerator)
    torch.testing.assert_close(
        restored.generator.stage1.normalize_features(x),
        trained.generator.stage1.normalize_features(x),
    )


def test_checkpoint_missing_feature_standardization_never_silently_switches_modes(
    tmp_path: Path,
) -> None:
    """A checkpoint omitting `feature_standardization` raises, never reinterprets.

    This is the surviving half of the deleted rev-3.1 backfill's purpose. That
    backfill kept such a checkpoint loadable by pinning `"row_layernorm"`; with
    the backfills gone (design 2026-08-02 §11) the config's own rev-3.2 default
    would apply instead, which would score a row-LayerNorm checkpoint under
    z-scored features. Strict loading catches it because the two modes differ in
    state: only `zscore_vfit_v1` carries `generator.stage1.feature_norm.*` buffers.
    """
    legacy_cfg = E2EConfig(generator=GeneratorConfig(feature_standardization="row_layernorm"))
    legacy = EgoStitchModel(legacy_cfg)
    # Three-component refactor (design 2026-08-02 Sec 8): both keys now live
    # on the nested `generator` sub-config, not the top level.
    legacy_config = asdict(legacy_cfg)
    legacy_config["generator"] = {
        key: value
        for key, value in legacy_config["generator"].items()
        if key not in {"feature_standardization", "feature_stats_sha256"}
    }
    checkpoint_path = tmp_path / "config-incomplete-e2e.pt"
    _write_checkpoint(
        checkpoint_path,
        model=legacy,
        model_family="egostitch_e2e",
        model_config=legacy_config,
    )

    with pytest.raises(RuntimeError, match="feature_norm"):
        score_universe._load_checkpoint(checkpoint_path)


def test_default_e2e_grounding_cache_path_is_namespaced_by_pool_configuration(
    tmp_path: Path,
) -> None:
    f0_cache = tmp_path / "f0_matrix.pt"
    universe = ["node-a", "node-b", "node-c"]
    full = score_universe._default_grounding_cache_path(f0_cache, n_ground=50, node_ids=universe)
    narrow_pool = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=20, node_ids=universe
    )
    same_pool = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=50, node_ids=reversed(universe)
    )
    other_universe = score_universe._default_grounding_cache_path(
        f0_cache, n_ground=50, node_ids=["node-a", "node-b", "node-d"]
    )

    assert full != narrow_pool
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
    """Stand-in for the pinned `f0_mlp` batch contract (x_a/x_b/label -> logits/loss).

    `src/model/b0_alt.py` (`F0PairMLP`, the B0-alt baseline) was removed 2026-08-03 by
    owner decision; see `docs/results/E2-pair-to-topology-gap.md` for the closed result
    it produced. `score_universe.MODEL_BUILDERS` no longer ships a default `"f0_mlp"`
    entry, but the family's scoring path (F0-matrix batching, row order restoration,
    determinism, label passthrough) is decoupled from any concrete model, so it is
    still exercised end-to-end here using this local double, registered into
    `score_universe.MODEL_BUILDERS` for the duration of a single test via monkeypatch.
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
    # save_scores always stamps an explicit heldout marker (False here: family
    # v3_1 never claims the held-out E2E universe); the round-trip otherwise
    # preserves the input meta exactly.
    assert loaded.meta == {**meta, "heldout": False}
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


def test_balanced_shard_range_is_contiguous_covering_and_deterministic() -> None:
    costs = [100, 4, 9, 1, 7, 2, 20, 3, 5]
    ranges = [score_universe._balanced_shard_range(costs, shard, 4) for shard in range(4)]
    assert ranges == [score_universe._balanced_shard_range(costs, shard, 4) for shard in range(4)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(costs)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False))


def test_balanced_shard_range_evenly_splits_uniform_costs() -> None:
    assert [score_universe._balanced_shard_range([1] * 12, shard, 3) for shard in range(3)] == [
        (0, 4),
        (4, 8),
        (8, 12),
    ]


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


# --------------------------------------------------------------------------- retired egostitch
# The frozen-s0 `egostitch` family was excised with the two-stage cleanup
# (docs/superpowers/specs/2026-07-29-egostitch-e2e-two-stage-cleanup.md §6.2).
# Its evidence survives under docs/results/ and outputs/egostitch_stage1/, so
# these two tests pin that the retirement fails *closed* on a surviving
# artifact rather than silently admitting it.


def test_build_model_rejects_the_retired_egostitch_family() -> None:
    assert "egostitch" not in score_universe.MODEL_BUILDERS
    with pytest.raises(ValueError, match="unknown model_family 'egostitch'"):
        score_universe.build_model("egostitch", {})


def test_validate_score_precision_refuses_a_legacy_egostitch_artifact() -> None:
    with pytest.raises(ValueError, match="frozen-s0 'egostitch' family is retired"):
        score_universe.validate_score_precision(
            np.linspace(-1.0, 1.0, 4, dtype=np.float32),
            meta={"model_family": "egostitch"},
            label="legacy",
        )


# --------------------------------------------------------------------------- egostitch_e2e scoring
# (design rev 4 two-logit decomposition, content path removed; Task 14)

# Three-component refactor (design 2026-08-02 Sec 8): the flat `E2EConfig`
# fields are now nested under `generator`/`encoder`/`classifier`
# sub-mappings; `ste_dim`/`ste_layers` were renamed `dim`/`layers` on the
# `encoder` sub-config.
_TINY_E2E_CONFIG: dict[str, object] = {
    "generator": {
        # Explicit rather than relying on `GeneratorConfig`'s rev-3.2 default
        # (`"zscore_vfit_v1"`): these tests build checkpoints through
        # `build_model`/`_load_checkpoint` without ever calling
        # `set_feature_stats`, and the registered mode requires those buffers
        # to be populated before a forward pass. Pinning the stateless
        # rev-3.1 transform keeps this shared fixture representative of what
        # it always tested (scoring/merge/precision plumbing, not feature
        # standardization) -- the registered-stats path is covered
        # separately in `test_e2e_checkpoint_restores_the_feature_standardization`.
        "feature_standardization": "row_layernorm",
    },
    "encoder": {
        "dim": 16,
        "layers": 2,
    },
    "classifier": {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 4,
        "n_inj": 1,
        "xattn_heads": 4,
    },
}


def _tiny_e2e_config_with_permanent_null(permanent_null: str) -> dict[str, object]:
    """`_TINY_E2E_CONFIG` with `classifier.permanent_null` overridden.

    `permanent_null` moved from the flat `E2EConfig` onto the nested
    `classifier` sub-config (design 2026-08-02 Sec 8).
    """
    return {
        **_TINY_E2E_CONFIG,
        "classifier": {
            **cast(dict[str, object], _TINY_E2E_CONFIG["classifier"]),
            "permanent_null": permanent_null,
        },
    }


# EgoStitchModel's internal Stage-1 generator keeps its own pinned spec default
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
    """Build a `score` CLI arg list for a generic (not held-out-focused) e2e test.

    `val` (val_edges.txt) is retired, so every named source now requires the
    held-out test-access ledger. Writes `pairs` to `test_edges.txt` and mints
    a fresh, `output`-scoped `--run-metadata` (arm keyed by `output.stem`, so
    two different `output` paths never collide on the same ledger epoch even
    though they share one ledger file per directory).
    """
    test_pairs = data_root / "benchmark_2025_neurips" / "breadth_first" / "test_edges.txt"
    test_pairs.parent.mkdir(parents=True, exist_ok=True)
    test_pairs.write_bytes(pairs.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    run_metadata = output.parent / f"{output.stem}_run_metadata.json"
    run_metadata.write_text(
        json.dumps({"arm": f"score_args_{output.stem}", "seed": 0}), encoding="utf-8"
    )
    return [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        "test",
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--token-budget",
        "8192",
        "--f0-cache",
        str(tmp_path / "f0_cache.pt"),
        "--device",
        "cpu",
        "--run-metadata",
        str(run_metadata),
    ]


def test_build_model_egostitch_e2e_round_trips() -> None:
    from src.model.egostitch.composite import EgoStitchModel

    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchModel)
    assert model.cfg.classifier.d_model == 32


def test_egostitch_e2e_cli_scores_two_logit_decomposition(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    assert artifact.meta["model_family"] == "egostitch_e2e"
    assert len(artifact.logit) == len(_E2E_PAIRS)
    assert artifact.logit.dtype == np.float32
    assert artifact.f_logit is not None
    # Content path removed (design doc §9): new artifacts never populate the
    # legacy v3 pair_content/pair_topology arrays.
    assert artifact.pair_content is None
    assert artifact.pair_topology is None
    assert artifact.f_logit.dtype == np.float32
    assert artifact.f_logit.shape == artifact.logit.shape
    assert bool(np.isfinite(artifact.f_logit).all())

    assert artifact.meta["score_precision"] == {
        "contract": "egostitch_e2e_pair_fp32_v1",
        "pair_compute_dtype": "float32",
        "pair_autocast": False,
        "logit_storage_dtype": "float32",
    }

    resolution = artifact.meta["score_resolution"]
    assert isinstance(resolution, dict)
    assert set(resolution) == {"full", "f_logit"}
    for name, arr in (
        ("full", artifact.logit),
        ("f_logit", artifact.f_logit),
    ):
        assert resolution[name] == score_universe.score_resolution_diagnostics(arr)

    # Zero-init sanity check: GatedCrossAttention's gate parameter starts at
    # exactly zero, so the topo pathway is an exact-identity bypass regardless
    # of which head a given arm nulls out. Both logits must therefore be
    # bit-for-bit identical at init.
    np.testing.assert_array_equal(artifact.logit, artifact.f_logit)


def test_egostitch_e2e_permanent_null_publishes_active_primary_logit(tmp_path: Path) -> None:
    # content_head is deleted with the content path (design doc §9); "none"
    # and "all_head" are the only surviving permanent_null values.
    for permanent_null, active_key in (("all_head", "f_logit"),):
        config = _tiny_e2e_config_with_permanent_null(permanent_null)
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


@pytest.mark.parametrize("permanent_null", ["all_head"])
def test_egostitch_e2e_null_arm_merge_preserves_true_full_logit(
    tmp_path: Path, permanent_null: str
) -> None:
    """Null-arm shards retain the distinct true full arm through merge/load."""
    config = _tiny_e2e_config_with_permanent_null(permanent_null)
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
    # Merge destination must equal the shards' own unsharded `--output`
    # reference (the production convention `src/score_fanout.py` follows):
    # the held-out test-access ledger binds each shard's record to that exact
    # path, and `save_scores` re-validates the binding against wherever the
    # merged artifact is actually written.
    merged_path = output
    score_universe.main(
        ["merge", "--inputs", *(str(path) for path in shard_paths), "--output", str(merged_path)]
    )

    merged = score_universe.load_scores(merged_path)
    assert merged.full_logit is not None
    assert not np.array_equal(merged.logit, merged.full_logit)
    score_universe.validate_artifact_precision(merged)


# --------------------------------------------------------------------------- null-generator arm
# (Codex P1, 2026-08-03: `_score_egostitch_e2e` used to blanket-assert the real
# `egostitch_imagine` generator, so every `generator.name: "null"` checkpoint
# failed to score at all -- contradicting `build_e2e_parameter_groups`'s own
# refusal message, which tells the user to score that arm. Fixed by handling
# the null generator's `slots=None`/`projected_x=None` node state explicitly
# instead of asserting it away, while keeping the assert loud for a genuinely
# broken real-generator/checkpoint pairing.)


def _null_generator_e2e_config() -> dict[str, object]:
    """`_TINY_E2E_CONFIG` with `generator.name` overridden to `"null"`."""
    return {
        **_TINY_E2E_CONFIG,
        "generator": {
            **cast(dict[str, object], _TINY_E2E_CONFIG["generator"]),
            "name": "null",
        },
    }


def _null_generator_e2e_setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Feature store + a conditioned checkpoint + a matching null-generator checkpoint.

    The null-generator checkpoint's classifier weights are loaded directly
    from the conditioned checkpoint's classifier -- the same reasoning
    `tests/model/test_composite.py`'s
    `test_registry_driven_null_generator_config_matches_conditioned_model_f_logit`
    applies in-process: the two models are independently seeded, so their
    classifiers start with different weights unless reconciled;
    `load_state_dict` isolates "does the null-generator arm reproduce this
    classifier's baseline" from "do two independently-seeded models happen to
    match".

    Returns:
        ``(data_root, conditioned_checkpoint, null_checkpoint, pairs_path)``.
    """
    torch.manual_seed(0)
    node_tokens = {node: torch.randn(3 + i, _E2E_NODE_DIM) for i, node in enumerate(_E2E_NODES)}
    data_root = _data_root_with_features(tmp_path, node_tokens, input_dim=_E2E_NODE_DIM)

    conditioned_config = dict(_TINY_E2E_CONFIG)
    conditioned_model = score_universe.build_model("egostitch_e2e", conditioned_config)
    assert isinstance(conditioned_model, EgoStitchModel)
    conditioned_checkpoint = tmp_path / "conditioned.pt"
    _write_checkpoint(
        conditioned_checkpoint,
        model=conditioned_model,
        model_family="egostitch_e2e",
        model_config=conditioned_config,
    )

    null_config = _null_generator_e2e_config()
    null_model = score_universe.build_model("egostitch_e2e", null_config)
    assert isinstance(null_model, EgoStitchModel)
    null_model.classifier.load_state_dict(conditioned_model.classifier.state_dict())
    null_checkpoint = tmp_path / "null_generator.pt"
    _write_checkpoint(
        null_checkpoint,
        model=null_model,
        model_family="egostitch_e2e",
        model_config=null_config,
    )

    pairs_path = tmp_path / "pairs.tsv"
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in _E2E_PAIRS))
    return data_root, conditioned_checkpoint, null_checkpoint, pairs_path


def test_null_generator_checkpoint_scores_through_score_universe_and_produces_valid_v4_artifact(
    tmp_path: Path,
) -> None:
    """A `generator.name: "null"` checkpoint scores end to end through the real CLI.

    This is the defect itself: as shipped, the blanket
    `isinstance(model.generator, EgoStitchImagineGenerator)` assert made this
    call fail for every null-generator checkpoint. Scoring through
    `score_universe.main`'s real ``score`` command (not a hand-rolled call
    into ``_score_egostitch_e2e``) and loading the result back must now
    succeed and satisfy the v4 artifact schema.
    """
    data_root, _conditioned_checkpoint, null_checkpoint, pairs_path = _null_generator_e2e_setup(
        tmp_path
    )
    output = tmp_path / "null_generator_scores.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, null_checkpoint, pairs_path, output)
    )

    artifact = score_universe.load_scores(output)
    assert artifact.meta["model_family"] == "egostitch_e2e"
    assert artifact.meta["scores_meta_version"] == score_universe._SCORES_META_VERSION
    assert len(artifact.logit) == len(_E2E_PAIRS)
    assert artifact.logit.dtype == np.float32
    assert artifact.f_logit is not None
    assert artifact.f_logit.dtype == np.float32
    assert artifact.f_logit.shape == artifact.logit.shape
    assert bool(np.isfinite(artifact.logit).all())
    assert bool(np.isfinite(artifact.f_logit).all())
    score_universe.validate_artifact_precision(artifact)


def test_null_generator_full_and_f_logit_arrays_are_exactly_equal(tmp_path: Path) -> None:
    """The null-generator arm's `full` and `f_logit` arrays are bit-identical.

    Not a degenerate result: `EgoStitchModel.__init__` never constructs a
    `GraphEncoder` for `generator.name: "null"` (`self.encoder is None`), so
    `score_pair_context`'s `need_topo` clamps to `False` unconditionally --
    the `full` head and the `f_logit` head both run the classifier fully
    unconditioned on the identical `PairInputs`, so they must come out
    exactly equal. That equality is the correct null measurement (the
    topology pathway contributed nothing), not a special case to collapse
    into a single published array.
    """
    data_root, _conditioned_checkpoint, null_checkpoint, pairs_path = _null_generator_e2e_setup(
        tmp_path
    )
    output = tmp_path / "null_generator_scores.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, null_checkpoint, pairs_path, output)
    )

    artifact = score_universe.load_scores(output)
    assert artifact.f_logit is not None
    np.testing.assert_array_equal(artifact.logit, artifact.f_logit)


def test_null_generator_logits_match_conditioned_checkpoints_f_logit(tmp_path: Path) -> None:
    """The null-generator arm really is the pairwise baseline, at the artifact level.

    Artifact-level counterpart of `tests/model/test_composite.py`'s in-process
    `test_registry_driven_null_generator_config_matches_conditioned_model_f_logit`:
    scoring a null-generator checkpoint through the real CLI ``score`` command
    reproduces exactly what the *same* classifier weights produce as
    ``f_logit`` when a real, topology-conditioned checkpoint is scored
    through the identical command. `f_logit` is invariant to the grounding
    pool and to topology (the eval-time hard bypass nulls the topo head
    before the classifier ever attends to it), so this holds regardless of
    the two runs' independently-built grounding-pool caches.
    """
    data_root, conditioned_checkpoint, null_checkpoint, pairs_path = _null_generator_e2e_setup(
        tmp_path
    )
    conditioned_output = tmp_path / "conditioned_scores.npz"
    score_universe.main(
        _egostitch_e2e_score_args(
            tmp_path, data_root, conditioned_checkpoint, pairs_path, conditioned_output
        )
    )
    conditioned_artifact = score_universe.load_scores(conditioned_output)
    assert conditioned_artifact.f_logit is not None

    null_output = tmp_path / "null_generator_scores.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, null_checkpoint, pairs_path, null_output)
    )
    null_artifact = score_universe.load_scores(null_output)

    np.testing.assert_allclose(
        null_artifact.logit, conditioned_artifact.f_logit, rtol=0.0, atol=1e-5
    )


def test_real_generator_checkpoint_missing_slot_state_still_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real-generator checkpoint whose node state is missing slots still raises loudly.

    The null-generator fix (above) replaced a blanket assert with a branch on
    `is_real_generator`; this proves the branch did not weaken the real
    generator's own guarantee. Forcing `EgoStitchImagineGenerator.encode_node`
    to echo its input back (the null generator's own behavior) simulates a
    genuinely broken generator/checkpoint pairing without changing
    `model.generator`'s type -- `is_real_generator` still reads `True`, so
    the scorer must still assert instead of silently caching `slots=None` or
    crashing on a `NoneType` deep inside `_stack_cached`.
    """
    data_root, conditioned_checkpoint, _null_checkpoint, pairs_path = _null_generator_e2e_setup(
        tmp_path
    )
    del pairs_path
    payload = torch.load(conditioned_checkpoint, weights_only=False)
    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchModel)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def _echo_x(
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
        *,
        node_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del ground, ground_ids, node_rows
        return x

    monkeypatch.setattr(model.generator, "encode_node", _echo_x)

    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    with pytest.raises(AssertionError, match="not the supported null-generator arm"):
        score_universe._score_egostitch_e2e(
            model,
            _E2E_PAIRS,
            store,
            device=torch.device("cpu"),
            token_budget=4096,
            f0_cache=tmp_path / "missing-slots-f0.npz",
            grounding_cache=tmp_path / "missing-slots-grounding.npz",
        )


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


def test_egostitch_e2e_f0_cache_requires_exact_universe_match(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    cache_path = tmp_path / "f0_cache.pt"
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    expected_node_ids = sorted({node for pair in _E2E_PAIRS for node in pair})
    build_f0_matrix(store, expected_node_ids, cache_path=cache_path)

    output = tmp_path / "exact-cache.npz"
    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    artifact = score_universe.load_scores(output)
    assert artifact.node_ids == expected_node_ids


def test_egostitch_e2e_rejects_superset_f0_cache(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    cache_path = tmp_path / "f0_cache.pt"
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    build_f0_matrix(store, sorted(_E2E_NODES), cache_path=cache_path)

    with pytest.raises(ValueError, match="does not match the requested node_ids"):
        score_universe.main(
            _egostitch_e2e_score_args(
                tmp_path, data_root, checkpoint, pairs, tmp_path / "rejected.npz"
            )
        )


def test_egostitch_e2e_exact_f0_cache_is_shard_invariant(tmp_path: Path) -> None:
    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    cache_path = tmp_path / "f0_cache.pt"
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    expected_node_ids = sorted({node for pair in _E2E_PAIRS for node in pair})
    build_f0_matrix(store, expected_node_ids, cache_path=cache_path)

    unsharded_path = tmp_path / "unsharded-cache.npz"
    score_universe.main(
        _egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, unsharded_path)
    )
    unsharded = score_universe.load_scores(unsharded_path)

    shard_base = tmp_path / "sharded-cache.npz"
    shard_artifacts = []
    for shard in range(2):
        score_universe.main(
            [
                *_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, shard_base),
                "--shard",
                str(shard),
                "--num-shards",
                "2",
            ]
        )
        shard_artifacts.append(
            score_universe.load_scores(
                shard_base.with_name(f"{shard_base.stem}.shard-{shard}{shard_base.suffix}")
            )
        )

    for key in ("logit", "f_logit"):
        full = getattr(unsharded, key)
        parts = [getattr(artifact, key) for artifact in shard_artifacts]
        assert full is not None and all(part is not None for part in parts)
        np.testing.assert_allclose(
            np.concatenate(cast(list[np.ndarray], parts)),
            full,
            rtol=1e-5,
            atol=2e-6,
        )


def _enable_e2e_conditioning_gates(checkpoint: Path) -> None:
    """Make the synthetic checkpoint sensitive to topology perturbations."""
    payload = torch.load(checkpoint, weights_only=False)
    model = score_universe.build_model("egostitch_e2e", dict(_TINY_E2E_CONFIG))
    assert isinstance(model, EgoStitchModel)
    model.load_state_dict(payload["model_state"])
    # cont_xattn is deleted with the content path (design doc §9); only the
    # topo pathway remains to make the checkpoint sensitive to perturbations.
    # `model.trunk` (an `EgoStitchE2E` attribute) is now `model.classifier.trunk`
    # (design §5). `model.classifier`'s registry-widened type (design §12 P3)
    # no longer statically exposes `.trunk`, so narrow it explicitly --
    # `_TINY_E2E_CONFIG` always selects the default `b0_v31` classifier.
    assert isinstance(model.classifier, B0V31PairClassifier)
    for module in [*model.classifier.trunk.topo_xattn]:
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
    assert isinstance(model, EgoStitchModel)
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
    original = EgoStitchModel.decompose_pair_context

    def _spy(self: EgoStitchModel, context: E2EPairContext) -> dict[str, torch.Tensor]:
        calls.append(context.encoded_a.shape[0])
        return original(self, context)

    monkeypatch.setattr(EgoStitchModel, "decompose_pair_context", _spy)
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
    original_encode = cast(Callable[..., E2ENodeState], EgoStitchModel.encode_node_state)
    original_context = cast(
        Callable[..., E2EPairContext], EgoStitchModel.build_pair_context_from_states
    )
    original_head = cast(Callable[..., torch.Tensor], EgoStitchModel.score_pair_context)

    def encode_spy(self: EgoStitchModel, *args: object, **kwargs: object) -> E2ENodeState:
        nonlocal encoded_rows
        encoded_rows += cast(torch.Tensor, args[0]).shape[0]
        return original_encode(self, *args, **kwargs)

    def context_spy(self: EgoStitchModel, *args: object, **kwargs: object) -> E2EPairContext:
        nonlocal context_calls
        context_calls += 1
        return original_context(self, *args, **kwargs)

    def head_spy(self: EgoStitchModel, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal head_calls
        head_calls += 1
        return original_head(self, *args, **kwargs)

    monkeypatch.setattr(EgoStitchModel, "encode_node_state", encode_spy)
    monkeypatch.setattr(EgoStitchModel, "build_pair_context_from_states", context_spy)
    monkeypatch.setattr(EgoStitchModel, "score_pair_context", head_spy)
    score_universe.main(
        _egostitch_e2e_score_args(
            tmp_path, data_root, checkpoint, pairs_path, tmp_path / "cached.npz"
        )
    )
    assert encoded_rows == len({node for pair in many_pairs for node in pair})
    assert 0 < context_calls < len(many_pairs)
    # Two decomposition heads survive the content-path removal (design doc
    # §9): full (unnulled) and f_logit (all-head nulled).
    assert head_calls == 2 * context_calls


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
    assert isinstance(model, EgoStitchModel)
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
    assert cached.pair_content is None
    assert cached.pair_topology is None
    cached_arrays = {
        "full": cached.logit,
        "f_logit": cached.f_logit,
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
        # "val" (not "candidate"/"test") deliberately: this test is only about
        # merge's scaffold_control/permanent_null/primary_logit conflict
        # detection, and a held-out pairs_source would require a genuine
        # test-access ledger binding it has no reason to fabricate.
        "pairs_source": "val",
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
                full_logit=data["logit"],
            )
        with pytest.raises(ValueError, match=key):
            score_universe.merge_scores([left, right])


def test_egostitch_e2e_merge_preserves_two_arrays_in_manifest_order(tmp_path: Path) -> None:
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
    # Merge destination must equal the shards' own unsharded `--output`
    # reference (the production convention `src/score_fanout.py` follows):
    # the held-out test-access ledger binds each shard's record to that exact
    # path, and `save_scores` re-validates the binding against wherever the
    # merged artifact is actually written.
    merged_path = output
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
    assert merged.pair_content is None
    assert merged.pair_topology is None
    assert unsharded.f_logit is not None
    assert unsharded.pair_content is None
    assert unsharded.pair_topology is None
    assert len(merged.f_logit) == len(_E2E_PAIRS)

    # Merged rows are remapped onto the sorted node-id union but otherwise keep
    # each row's own pair aligned; comparing by pair identity against the
    # unsharded run confirms manifest order (row <-> pair) is preserved.
    # Ordinary sharded scoring may use different pair-batch shapes, so compare
    # within the same explicit fp32 tolerance as cached-vs-direct decomposition.
    merged_index = {pair: row for row, pair in enumerate(merged.pairs())}
    unsharded_index = {pair: row for row, pair in enumerate(unsharded.pairs())}
    assert set(merged_index) == set(unsharded_index) == set(_E2E_PAIRS)
    for pair in _E2E_PAIRS:
        m_row, u_row = merged_index[pair], unsharded_index[pair]
        assert merged.logit[m_row] == pytest.approx(unsharded.logit[u_row], rel=1e-5, abs=2e-6)
        assert merged.f_logit[m_row] == pytest.approx(unsharded.f_logit[u_row], rel=1e-5, abs=2e-6)


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
    assert artifact.f_logit is not None
    corrupted = score_universe.ScoresArtifact(
        node_ids=artifact.node_ids,
        u_idx=artifact.u_idx,
        v_idx=artifact.v_idx,
        logit=artifact.logit,
        label=artifact.label,
        meta=artifact.meta,
        f_logit=artifact.f_logit.astype(np.float64),  # dtype broken on purpose
    )
    with pytest.raises(ValueError, match="must be stored as float32"):
        score_universe.validate_artifact_precision(corrupted, label="corrupted")


def test_validate_score_precision_old_idiom_on_e2e_artifact_points_to_new_entry_point(
    tmp_path: Path,
) -> None:
    """The old generic idiom on an egostitch_e2e artifact still raises, with a pointer.

    `validate_score_precision(artifact.logit, ...)` (as used by
    `src.experiments.g1_hardened_e2` for B0-family artifacts) still raises for an
    egostitch_e2e artifact — the two-array contract can't be checked without the
    extra `f_logit` array — but the error now points callers at
    validate_artifact_precision instead of reading as a false "artifact is
    missing arrays" report.
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

    Spies on `EgoStitchModel.encode_node_state` to capture the node-cache inputs,
    asserting the required ``ground_a``/``ground_b`` tensors and optional
    ``ground_id_a``/``ground_id_b`` identifiers are present with the expected
    shapes and dtypes for every scored batch.
    """
    from src.model.egostitch.composite import EgoStitchModel
    from src.model.egostitch.config import E2EConfig

    data_root, checkpoint, pairs = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "scores.npz"

    captured: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    original_encode = EgoStitchModel.encode_node_state

    def _spy_encode(
        self: EgoStitchModel,
        emb: torch.Tensor,
        length: torch.Tensor,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
        node_rows: torch.Tensor | None = None,
    ) -> E2ENodeState:
        captured.append((ground, ground_ids))
        return original_encode(self, emb, length, x, ground, ground_ids, node_rows)

    monkeypatch.setattr(EgoStitchModel, "encode_node_state", _spy_encode)

    score_universe.main(_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs, output))

    assert captured, "node cache was never encoded"
    node_universe = len({node for pair in _E2E_PAIRS for node in pair})
    # `_egostitch_e2e_setup` builds the checkpoint from `_TINY_E2E_CONFIG`,
    # which does not override `n_ground` -- so the registered value is
    # `GeneratorConfig()`'s rev-3.1 default (spec Sec 14.4.4), not the internal
    # generator's own pinned `EgoStitchConfig()` default. `n_ground` moved
    # from the flat `E2EConfig` onto the nested `generator` sub-config
    # (design 2026-08-02 Sec 8).
    expected_n_ground = min(E2EConfig().generator.n_ground, node_universe - 1)
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
    `len(node_ids) - 1 == 2` grounding candidates, while `EgoStitchModel`'s
    `generator_cfg.n_ground` registers `GeneratorConfig()`'s rev-3.1 default
    (spec Sec 14.4.4; `_egostitch_e2e_setup`'s `_TINY_E2E_CONFIG` does not
    override it). Every call here silently clamps the registered default down
    to 2; that must now be logged, stating both the registered and effective
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
    # `n_ground` moved from the flat `E2EConfig` onto the nested `generator`
    # sub-config (design 2026-08-02 Sec 8).
    registered_n_ground = E2EConfig().generator.n_ground
    expected_n_ground = min(registered_n_ground, node_universe - 1)
    assert expected_n_ground < registered_n_ground, "fixture must actually trigger the clamp"

    clamp_records = [r for r in caplog.records if "n_ground clamped" in r.getMessage()]
    assert clamp_records, f"expected an n_ground-clamp warning; got records: {caplog.records}"
    message = clamp_records[0].getMessage()
    assert str(registered_n_ground) in message
    assert str(expected_n_ground) in message
    assert str(node_universe) in message


# --------------------------------------------------------------------------- val_region pairs
# source (fix 1: the real V_val validation-region universe, not val_edges.txt)

# A 12-node ring (with one self-loop, so a self row exercises the positive
# label branch too) gives one connected component small enough for a tiny
# `ValRegionParams` to grow a real, deterministic V_val region rather than
# through a hand-rolled substitute.
_VAL_REGION_NODES = [f"h{i}" for i in range(12)]
_VAL_REGION_TEST_PARAMS = ValRegionParams(
    edge_fraction=0.4,
    n_regions=2,
    salt="test-val-region|",
    bucket_sizes=(2, 3),
    buckets_per_size=2,
    negative_seed=0,
)


def _write_val_region_benchmark_package(tmp_path: Path, *, strategy: str = "breadth_first") -> Path:
    """Write split.pkl/train_graph.pkl/train_edges.txt/val_edges.txt/positive_edges.txt.

    A 12-node ring under `tmp_path/data`, split across `train_edges.txt` (the
    first half of the ring's edges, plus label-0 rows) and `val_edges.txt`
    (the other half, plus a self-loop row) -- mirroring the benchmark's own
    train+/val+ = train_graph convention closely enough to exercise
    `_load_val_region_split`'s real read-from-disk mirror.
    """
    strategy_dir = tmp_path / "data" / "benchmark_2025_neurips" / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    with (strategy_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": _VAL_REGION_NODES, "test": []}, handle)

    ring_edges = [
        (_VAL_REGION_NODES[i], _VAL_REGION_NODES[(i + 1) % len(_VAL_REGION_NODES)])
        for i in range(len(_VAL_REGION_NODES))
    ]
    self_loop_node = _VAL_REGION_NODES[0]
    graph = nx.Graph()
    graph.add_nodes_from(_VAL_REGION_NODES)
    graph.add_edges_from(ring_edges)
    graph.add_edge(self_loop_node, self_loop_node)
    with (strategy_dir / "train_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)

    negatives = [(_VAL_REGION_NODES[i], _VAL_REGION_NODES[i + 6]) for i in range(0, 6, 2)]
    (strategy_dir / "train_edges.txt").write_text(
        "".join(f"{u}\t{v}\t1\n" for u, v in ring_edges[:6])
        + "".join(f"{u}\t{v}\t0\n" for u, v in negatives),
        encoding="utf-8",
    )
    (strategy_dir / "val_edges.txt").write_text(
        "".join(f"{u}\t{v}\t1\n" for u, v in ring_edges[6:])
        + f"{self_loop_node}\t{self_loop_node}\t1\n",
        encoding="utf-8",
    )
    positive_pairs = sorted(
        {canonical_pair(u, v) for u, v in [*ring_edges, (self_loop_node, self_loop_node)]}
    )
    (tmp_path / "data" / "benchmark_2025_neurips" / "positive_edges.txt").write_text(
        "".join(f"{u}\t{v}\n" for u, v in positive_pairs), encoding="utf-8"
    )
    return tmp_path / "data"


def _val_region_e2e_setup(
    tmp_path: Path, *, model_config: dict[str, object] | None = None
) -> tuple[Path, Path]:
    """Feature store (12 ring nodes) + benchmark package + `egostitch_e2e` checkpoint."""
    data_root = _write_val_region_benchmark_package(tmp_path)
    torch.manual_seed(0)
    node_tokens = {node: torch.randn(3, _E2E_NODE_DIM) for node in _VAL_REGION_NODES}
    _write_feature_store(
        data_root / "features" / "frozen_node_features_1024", node_tokens, input_dim=_E2E_NODE_DIM
    )
    config = dict(_TINY_E2E_CONFIG if model_config is None else model_config)
    model = score_universe.build_model("egostitch_e2e", config)
    checkpoint_path = tmp_path / "egostitch_e2e_val_region.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="egostitch_e2e", model_config=config
    )
    return data_root, checkpoint_path


def test_resolve_val_region_pairs_matches_direct_derivation_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--pairs val_region` must reproduce a from-scratch derivation exactly.

    Both the universe (pairs, row order) and the labels must match.
    """
    monkeypatch.setattr(score_universe, "_VAL_REGION_PARAMS", _VAL_REGION_TEST_PARAMS)
    data_root = _write_val_region_benchmark_package(tmp_path)

    pairs, labels = score_universe._resolve_pairs("val_region", data_root, "breadth_first")

    strategy_dir = data_root / "benchmark_2025_neurips" / "breadth_first"
    with (strategy_dir / "split.pkl").open("rb") as handle:
        split_payload = pickle.load(handle)  # noqa: S301
    with (strategy_dir / "train_graph.pkl").open("rb") as handle:
        train_graph = pickle.load(handle)  # noqa: S301
    train_pairs, train_labels = score_universe._read_pairs_tsv(strategy_dir / "train_edges.txt")
    val_pairs_raw, val_labels_raw = score_universe._read_pairs_tsv(strategy_dir / "val_edges.txt")
    benchmark_negatives = [
        pair for pair, label in zip(train_pairs, train_labels.tolist(), strict=True) if label == 0
    ] + [
        pair
        for pair, label in zip(val_pairs_raw, val_labels_raw.tolist(), strict=True)
        if label == 0
    ]
    positive_pairs, _ = score_universe._read_pairs_tsv(
        data_root / "benchmark_2025_neurips" / "positive_edges.txt"
    )

    expected_split = derive_val_region_split(
        sorted(split_payload["train"]),
        list(train_graph.edges()),
        benchmark_negatives,
        frozenset(positive_pairs),
        params=_VAL_REGION_TEST_PARAMS,
    )
    expected_nodes = sorted(expected_split.v_val)
    expected_u, expected_v = val_universe_arrays(expected_split.v_val)
    expected_pairs = [
        (expected_nodes[u], expected_nodes[v])
        for u, v in zip(expected_u.tolist(), expected_v.tolist(), strict=True)
    ]
    expected_positives = frozenset(expected_split.val_positives)
    expected_labels = np.array(
        [1 if pair in expected_positives else 0 for pair in expected_pairs], dtype=np.int8
    )

    assert pairs == expected_pairs
    np.testing.assert_array_equal(labels, expected_labels)
    assert len(expected_split.val_positives) > 0, "fixture must produce a real V_val topology"


def test_val_region_pairs_source_is_never_a_heldout_universe() -> None:
    assert (
        score_universe._is_heldout_universe(
            {"model_family": "egostitch_e2e", "pairs_source": "val_region"}
        )
        is False
    )


def test_test_topology_pairs_source_is_a_heldout_universe() -> None:
    assert score_universe._is_heldout_universe(
        {"model_family": "egostitch_e2e", "pairs_source": "test_topology"}
    )


def test_val_region_scoring_never_writes_a_ledger_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--pairs val_region` needs no `--run-metadata` and never touches the test-access ledger."""
    monkeypatch.setattr(score_universe, "_VAL_REGION_PARAMS", _VAL_REGION_TEST_PARAMS)
    data_root, checkpoint = _val_region_e2e_setup(tmp_path)
    output = tmp_path / "val_region.npz"

    score_universe.main(
        [
            "score",
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            "val_region",
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--token-budget",
            "8192",
            "--f0-cache",
            str(tmp_path / "f0_cache.pt"),
            "--device",
            "cpu",
        ]
    )

    artifact = score_universe.load_scores(output)
    assert artifact.meta["pairs_source"] == "val_region"
    assert artifact.meta["heldout"] is False
    assert "test_access_ledger" not in artifact.meta
    assert not list(tmp_path.rglob("test_access_ledger.jsonl")), (
        "val_region scoring must never append to any test-access ledger"
    )


def test_val_pairs_source_no_longer_resolves(tmp_path: Path) -> None:
    """`val` (val_edges.txt) is retired; only `val_region` names the validation universe."""
    with pytest.raises(ValueError, match=r"unsupported --pairs value 'val'"):
        score_universe._resolve_pairs("val", tmp_path / "data", "breadth_first")


# --------------------------------------------------------------------------- cazi_mbn scoring
# (fix 3: the released `{"state_dict": ...}` checkpoint format and its own
# feature-standardization path)


def _write_cazi_config(
    tmp_path: Path,
    *,
    output_dir: Path,
    latent_dim: int,
    network_layers: int,
    missing_features: list[str] | None = None,
) -> Path:
    """Write a minimal but fully valid CAZI-MBN training YAML (JSON is valid YAML)."""
    config_path = tmp_path / "cazi.yaml"
    payload = {
        "model": {
            "order": 2,
            "topology_dim": 8,
            "latent_dim": latent_dim,
            "network_layers": network_layers,
            "heads": 2,
        },
        "data": {
            "root": "data",
            "strategy": "breadth_first",
            "f0_cache": str(tmp_path / "unused_f0_cache.pt"),
            "expected_missing_features": missing_features or [],
        },
        "optim": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "teacher_epochs": 1,
            "student_epochs": 1,
            "patience": 1,
        },
        "loss": {
            "discriminator_coef": 1.0,
            "regularization_coef": 1.0,
            "classification_coef": 1.0,
            "distillation_weight": 0.5,
            "supervised_weight": 0.5,
        },
        "runtime": {"batch_size": 4, "score_batch_size": 4},
        "seed": 0,
        "output_dir": str(output_dir),
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _write_cazi_checkpoint_and_config(
    tmp_path: Path,
    *,
    sequence_dim: int,
    latent_dim: int,
    network_layers: int,
    missing_features: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Released-format `student.pt` + its training YAML + a matching `feature_stats.npz`."""
    from src.baselines.cazi_mbn import CAZIStudent
    from src.data.feature_stats import compute_feature_stats, save_feature_stats

    torch.manual_seed(0)
    model = CAZIStudent(sequence_dim, latent_dim=latent_dim, network_layers=network_layers)
    model.eval()

    output_dir = tmp_path / "cazi_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    # `src.train_cazi_mbn` writes `student.pt` and `feature_stats.npz` into the
    # same run directory, and scoring resolves the statistics beside the
    # checkpoint so an `--output-dir` override cannot pick up another run's.
    checkpoint_path = output_dir / "student.pt"
    torch.save({"state_dict": model.state_dict(), "best_val_auroc": 0.75}, checkpoint_path)

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4, sequence_dim)).astype(np.float32)
    stats = compute_feature_stats(rows, [f"s{i}" for i in range(4)])
    save_feature_stats(stats, output_dir / "feature_stats.npz")

    config_path = _write_cazi_config(
        tmp_path,
        output_dir=output_dir,
        latent_dim=latent_dim,
        network_layers=network_layers,
        missing_features=missing_features,
    )
    return checkpoint_path, config_path, output_dir


def test_build_cazi_mbn_infers_architecture_from_checkpoint_state_dict() -> None:
    from src.baselines.cazi_mbn import CAZIStudent

    torch.manual_seed(0)
    reference = CAZIStudent(6, latent_dim=3, network_layers=2)
    inferred_config = score_universe._cazi_model_config_from_state(reference.state_dict())
    assert inferred_config == {"sequence_dim": 6, "latent_dim": 3, "network_layers": 2}
    rebuilt = score_universe.build_model("cazi_mbn", inferred_config)
    assert isinstance(rebuilt, CAZIStudent)
    rebuilt.load_state_dict(reference.state_dict())


def test_cazi_mbn_scoring_rejects_statistics_missing_beside_the_checkpoint(
    tmp_path: Path,
) -> None:
    """An `--output-dir` override moves both files; the YAML's path is not authoritative.

    Resolving `feature_stats.npz` from the config's own `output_dir` would either
    miss the file or silently standardize with a different run's statistics —
    wrong scores that raise nothing.
    """
    checkpoint_path, config_path, output_dir = _write_cazi_checkpoint_and_config(
        tmp_path, sequence_dim=6, latent_dim=3, network_layers=2
    )
    (output_dir / "feature_stats.npz").unlink()

    args = argparse.Namespace(checkpoint=checkpoint_path, model_config=config_path)
    with pytest.raises(ValueError, match="beside the checkpoint"):
        score_universe._resolve_cazi_context(args)


def test_cazi_mbn_checkpoint_requires_the_released_state_dict_wrapper(tmp_path: Path) -> None:
    """A bare `state_dict` (no `{"state_dict": ...}` wrapper) must fail with a clear reason."""
    from src.baselines.cazi_mbn import CAZIStudent

    torch.manual_seed(0)
    model = CAZIStudent(6, latent_dim=3, network_layers=2)
    checkpoint_path = tmp_path / "bad_student.pt"
    torch.save(model.state_dict(), checkpoint_path)  # no wrapper -- the release's own format

    with pytest.raises(ValueError, match="released.*state_dict.*format"):
        score_universe._load_checkpoint(checkpoint_path, model_family="cazi_mbn", model_config={})


def test_cazi_mbn_scoring_matches_manual_standardization_and_zeros_missing_features(
    tmp_path: Path,
) -> None:
    """CAZI scoring must reuse `_standardize_f0` and zero rows for known feature gaps."""
    from src.train_cazi_mbn import _standardize_f0

    sequence_dim = INPUT_DIM
    # `network_layers` doubles as the MoE output dimension (`CAZIStudent.pair_logits`
    # squeezes it to a scalar), so every real config pins it to 1
    # (`configs/cazi_mbn_breadth_first.yaml`) -- this fixture must too, or
    # `pair_logits` returns a per-pair vector instead of a scalar.
    checkpoint_path, config_path, output_dir = _write_cazi_checkpoint_and_config(
        tmp_path,
        sequence_dim=sequence_dim,
        latent_dim=3,
        network_layers=1,
        missing_features=["missing_node"],
    )
    node_tokens = {
        "a": torch.randn(5, sequence_dim),
        "b": torch.randn(3, sequence_dim),
        # "missing_node" deliberately absent from the feature store.
    }
    data_root = _data_root_with_features(tmp_path, node_tokens, input_dim=sequence_dim)
    pairs_path = tmp_path / "pairs.tsv"
    pairs_path.write_text("a\tmissing_node\t1\na\tb\t0\n", encoding="utf-8")
    output = tmp_path / "cazi_scores.npz"

    score_universe.main(
        [
            "score",
            "--checkpoint",
            str(checkpoint_path),
            "--model-family",
            "cazi_mbn",
            "--model-config",
            str(config_path),
            "--pairs",
            f"file:{pairs_path}",
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    artifact = score_universe.load_scores(output)
    assert artifact.meta["model_family"] == "cazi_mbn"

    # Independently recompute the expected logits the same way `_score_cazi_mbn`
    # does (build_f0_matrix + zero row for the missing node + `_standardize_f0`),
    # but through this test's own call, not through the scorer under test.
    from src.data.feature_stats import load_feature_stats
    from src.data.features import FeatureStore, build_f0_matrix

    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")
    present_f0, present_position = build_f0_matrix(store, ["a", "b"], cache_path=None)
    raw = torch.zeros((3, sequence_dim), dtype=torch.float32)
    position = {"a": 0, "b": 1, "missing_node": 2}
    raw[0] = present_f0[present_position["a"]]
    raw[1] = present_f0[present_position["b"]]
    stats = load_feature_stats(output_dir / "feature_stats.npz")
    sequence = _standardize_f0(raw, stats)

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    from src.baselines.cazi_mbn import CAZIStudent

    reference_model = CAZIStudent(sequence_dim, latent_dim=3, network_layers=1)
    reference_model.load_state_dict(checkpoint_payload["state_dict"])
    reference_model.eval()
    with torch.no_grad():
        expected_logits = reference_model.pair_logits(
            sequence,
            torch.tensor([position["a"], position["a"]], dtype=torch.long),
            torch.tensor([position["missing_node"], position["b"]], dtype=torch.long),
        )

    # `save_scores` stores rows canonicalized (min(u, v), max(u, v)), so match
    # `artifact.pairs()`'s own row order rather than assuming input order.
    row_by_pair = dict(zip(artifact.pairs(), artifact.logit.tolist(), strict=True))
    expected_by_pair = {
        canonical_pair("a", "missing_node"): float(expected_logits[0]),
        canonical_pair("a", "b"): float(expected_logits[1]),
    }
    for pair, expected_value in expected_by_pair.items():
        assert row_by_pair[pair] == pytest.approx(expected_value, abs=1e-5)


# --------------------------------------------------------------------------- oracle_struct scoring
# (fix 4: an airtight diagnostic-only boundary for the ground-truth-scaffold arm)

_TINY_ORACLE_E2E_CONFIG: dict[str, object] = {
    "generator": {
        "name": "oracle_struct",
        "feature_standardization": "row_layernorm",
    },
    "encoder": {
        "dim": 16,
        "layers": 2,
    },
    "classifier": {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 4,
        "n_inj": 1,
        "xattn_heads": 4,
    },
}

_TINY_FULL_ORACLE_E2E_CONFIG: dict[str, object] = {
    **_TINY_ORACLE_E2E_CONFIG,
    "generator": {
        "name": "full_ego_oracle",
        "feature_standardization": "row_layernorm",
    },
}


def _write_test_graph_pkl(data_root: Path, strategy: str, edges: list[tuple[str, str]]) -> Path:
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(edges)
    path = data_root / "benchmark_2025_neurips" / strategy / "test_graph.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    return path


def _write_run_metadata(tmp_path: Path, checkpoint: Path, *, arm: str = "oracle") -> Path:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "seed": 0,
                "checkpoint_id": score_universe._checkpoint_id(checkpoint_payload["model_state"]),
            }
        ),
        encoding="utf-8",
    )
    return run_metadata_path


def test_oracle_truth_graph_rejects_unsupported_pairs_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no diagnostic truth graph"):
        score_universe._oracle_truth_graph_for_scoring(
            "unsupported", tmp_path / "data", "breadth_first"
        )


def test_allow_oracle_diagnostic_flag_rejected_for_a_non_oracle_generator(tmp_path: Path) -> None:
    """The flag must be refused, not silently ignored, when it does not apply."""
    data_root, checkpoint, pairs_path = _egostitch_e2e_setup(tmp_path)
    output = tmp_path / "out.npz"
    args = [
        *_egostitch_e2e_score_args(tmp_path, data_root, checkpoint, pairs_path, output),
        "--allow-oracle-diagnostic",
    ]
    with pytest.raises(ValueError, match="valid only when"):
        score_universe.main(args)
    assert not output.exists()


def test_full_oracle_scoring_is_unreachable_without_the_diagnostic_flag(
    tmp_path: Path,
) -> None:
    data_root, checkpoint, pairs_path = _egostitch_e2e_setup(
        tmp_path, model_config=_TINY_FULL_ORACLE_E2E_CONFIG
    )
    strategy_dir = data_root / "benchmark_2025_neurips" / "breadth_first"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "test_edges.txt").write_bytes(pairs_path.read_bytes())
    truth_graph_path = _write_test_graph_pkl(data_root, "breadth_first", _E2E_PAIRS)
    run_metadata_path = _write_run_metadata(tmp_path, checkpoint, arm="full_ego_oracle")
    output = tmp_path / "full_oracle.npz"
    args = [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        "test",
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--token-budget",
        "8192",
        "--f0-cache",
        str(tmp_path / "full_f0_cache.pt"),
        "--device",
        "cpu",
        "--run-metadata",
        str(run_metadata_path),
    ]

    with pytest.raises(ValueError, match="--allow-oracle-diagnostic"):
        score_universe.main(args)
    assert not output.exists()

    with pytest.raises(ValueError, match="does not support scoring-time scaffold controls"):
        score_universe.main(
            [
                *args,
                "--allow-oracle-diagnostic",
                "--scaffold-control",
                "shuffle_within_pair_v3",
            ]
        )
    assert not output.exists()

    score_universe.main([*args, "--allow-oracle-diagnostic"])
    artifact = score_universe.load_scores(output)
    assert len(artifact.logit) == len(_E2E_PAIRS)
    assert artifact.meta["formal"] is False
    diagnostic = cast(dict[str, object], artifact.meta["oracle_diagnostic"])
    assert diagnostic["generator"] == "full_ego_oracle"
    assert diagnostic["truth_graph_sha256"] == score_universe._oracle_truth_graph_sha256(
        nx.Graph(_E2E_PAIRS)
    )
    assert truth_graph_path.is_file()
    sidecar = output.with_name(f"{output.stem}.telemetry.json")
    telemetry = json.loads(sidecar.read_text(encoding="utf-8"))
    assert telemetry["rows"] == len(_E2E_PAIRS)
    assert telemetry["ego_size"] == {
        "max": 3,
        "p50": 3.0,
        "p90": 3.0,
        "p95": 3.0,
        "p99": 3.0,
    }
    assert telemetry["budgets"]["cell_budget"] == score_universe._FULL_ORACLE_CELL_BUDGET
    assert telemetry["producer_wait_seconds"] >= 0.0
    assert telemetry["max_memory_allocated_bytes"] == 0
    score_profile = cast(dict[str, object], artifact.meta["score_profile"])
    assert score_profile["peak_memory_allocated_bytes"] == 0


def test_full_oracle_batched_scoring_matches_singleton_and_inline_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, checkpoint, _ = _egostitch_e2e_setup(
        tmp_path, model_config=_TINY_FULL_ORACLE_E2E_CONFIG
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = score_universe.build_model("egostitch_e2e", _TINY_FULL_ORACLE_E2E_CONFIG)
    model.load_state_dict(payload["model_state"])
    model.eval()
    truth_graph = nx.Graph(_E2E_PAIRS)
    store = FeatureStore(data_root / "features" / "frozen_node_features_1024")

    def score() -> dict[str, np.ndarray]:
        return score_universe._score_egostitch_e2e(
            model,
            _E2E_PAIRS,
            store,
            device=torch.device("cpu"),
            token_budget=8192,
            f0_cache=tmp_path / "full-oracle-equivalence-f0.pt",
            grounding_cache=tmp_path / "full-oracle-equivalence-grounding.npz",
            oracle_truth_graph=truth_graph,
        )

    monkeypatch.setattr(score_universe, "_FULL_ORACLE_MAX_BATCH_PAIRS", 1)
    monkeypatch.setattr(score_universe, "_FULL_ORACLE_PREFETCH_DEPTH", 0)
    singleton = score()

    monkeypatch.setattr(score_universe, "_FULL_ORACLE_MAX_BATCH_PAIRS", 1024)
    inline = score()
    monkeypatch.setattr(score_universe, "_FULL_ORACLE_PREFETCH_DEPTH", 4)
    prefetched = score()

    for key in score_universe._EGOSTITCH_E2E_ARRAY_KEYS:
        np.testing.assert_allclose(inline[key], singleton[key], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(prefetched[key], inline[key])


def test_full_oracle_balanced_shards_merge_to_unsharded_with_empty_shard(
    tmp_path: Path,
) -> None:
    pairs = [("n0", "n1"), ("n2", "n3"), ("n2", "n2")]
    data_root, checkpoint, pairs_path = _egostitch_e2e_setup(
        tmp_path, model_config=_TINY_FULL_ORACLE_E2E_CONFIG
    )
    pairs_path.write_text("".join(f"{u}\t{v}\n" for u, v in pairs), encoding="utf-8")
    strategy_dir = data_root / "benchmark_2025_neurips" / "breadth_first"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "test_edges.txt").write_bytes(pairs_path.read_bytes())
    truth = nx.Graph([("n0", "n1"), ("n2", "n3")])
    truth.add_edges_from(("n0", f"hub-{index:02d}") for index in range(68))
    _write_test_graph_pkl(data_root, "breadth_first", list(truth.edges))
    run_metadata = _write_run_metadata(tmp_path, checkpoint, arm="full_ego_oracle")
    common = [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        "test",
        "--data-root",
        str(data_root),
        "--token-budget",
        "8192",
        "--f0-cache",
        str(tmp_path / "balanced-f0.pt"),
        "--grounding-cache",
        str(tmp_path / "balanced-grounding.npz"),
        "--device",
        "cpu",
        "--run-metadata",
        str(run_metadata),
        "--allow-oracle-diagnostic",
    ]

    unsharded_path = tmp_path / "balanced-unsharded.npz"
    score_universe.main([*common, "--output", str(unsharded_path), "--scoring-run-id", "unsharded"])
    shard_base = tmp_path / "balanced-shards.npz"
    shard_paths: list[Path] = []
    for shard in range(2):
        score_universe.main(
            [
                *common,
                "--output",
                str(shard_base),
                "--shard",
                str(shard),
                "--num-shards",
                "2",
                "--scoring-run-id",
                "balanced-shards",
                "--rescore-reason",
                "balanced shard merge regression",
            ]
        )
        shard_paths.append(tmp_path / f"balanced-shards.shard-{shard}.npz")

    assert len(score_universe.load_scores(shard_paths[0]).logit) == 0
    merged_path = tmp_path / "balanced-merged.npz"
    score_universe.main(
        ["merge", "--inputs", *[str(path) for path in shard_paths], "--output", str(merged_path)]
    )
    unsharded = score_universe.load_scores(unsharded_path)
    merged = score_universe.load_scores(merged_path)
    assert list(merged.pairs()) == list(unsharded.pairs())
    np.testing.assert_allclose(merged.logit, unsharded.logit, rtol=1e-6, atol=1e-6)
    assert merged.f_logit is not None and unsharded.f_logit is not None
    np.testing.assert_allclose(merged.f_logit, unsharded.f_logit, rtol=1e-6, atol=1e-6)
    assert merged.meta["num_rows"] == unsharded.meta["num_rows"] == len(pairs)


def test_oracle_generator_scoring_is_unreachable_without_the_diagnostic_flag(
    tmp_path: Path,
) -> None:
    data_root, checkpoint, pairs_path = _egostitch_e2e_setup(
        tmp_path, model_config=_TINY_ORACLE_E2E_CONFIG
    )
    strategy_dir = data_root / "benchmark_2025_neurips" / "breadth_first"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "test_edges.txt").write_bytes(pairs_path.read_bytes())
    _write_test_graph_pkl(data_root, "breadth_first", _E2E_PAIRS)
    run_metadata_path = _write_run_metadata(tmp_path, checkpoint, arm="oracle_struct")
    output = tmp_path / "oracle.npz"

    base_args = [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        "test",
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--token-budget",
        "8192",
        "--f0-cache",
        str(tmp_path / "f0_cache.pt"),
        "--device",
        "cpu",
        "--run-metadata",
        str(run_metadata_path),
    ]

    with pytest.raises(ValueError, match="--allow-oracle-diagnostic"):
        score_universe.main(base_args)
    assert not output.exists()

    score_universe.main([*base_args, "--allow-oracle-diagnostic"])
    artifact = score_universe.load_scores(output)
    truth_graph = nx.Graph(_E2E_PAIRS)
    assert artifact.meta["oracle_diagnostic"] == {
        "generator": "oracle_struct",
        "truth_source": "test_graph",
        "diagnostic_only": True,
        "formal": False,
        "truth_graph_sha256": score_universe._oracle_truth_graph_sha256(truth_graph),
        "truth_graph_node_count": truth_graph.number_of_nodes(),
        "truth_graph_edge_count": truth_graph.number_of_edges(),
    }
    assert artifact.meta["formal"] is False
    # A truth-consuming diagnostic can never be the formal held-out claim,
    # even though `pairs_source == "test"` looks held-out-shaped.
    assert artifact.meta["heldout"] is False
    assert score_universe._is_heldout_universe(artifact.meta) is False
    assert np.isfinite(artifact.logit).all()


def test_oracle_val_region_truth_diagnostic_still_writes_no_ledger_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `val_region`-truth oracle diagnostic combines fixes 1 and 4: still no ledger access."""
    monkeypatch.setattr(score_universe, "_VAL_REGION_PARAMS", _VAL_REGION_TEST_PARAMS)
    data_root, checkpoint = _val_region_e2e_setup(tmp_path, model_config=_TINY_ORACLE_E2E_CONFIG)
    output = tmp_path / "oracle_val_region.npz"

    score_universe.main(
        [
            "score",
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            "val_region",
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--token-budget",
            "8192",
            "--f0-cache",
            str(tmp_path / "f0_cache.pt"),
            "--device",
            "cpu",
            "--allow-oracle-diagnostic",
        ]
    )

    artifact = score_universe.load_scores(output)
    oracle_diagnostic = cast(dict[str, object], artifact.meta["oracle_diagnostic"])
    assert oracle_diagnostic["truth_source"] == "val_region_g_val"
    assert artifact.meta["heldout"] is False
    assert not list(tmp_path.rglob("test_access_ledger.jsonl"))
