"""Tests for src.experiments.s3_set_residual.evaluate: paired test-bucket readouts.

All CPU, synthetic fixtures mirroring `tests/experiments/test_s2_evaluate.py`:
a small connected test graph, hand-sampled buckets, a tiny `FeatureStore`, a
hand-written B0 candidate-universe scores artifact (via `save_scores`), and a
tiny `SetResidualModel` checkpoint written directly against the
`load_residual_checkpoint` contract (no `train.py` involved).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.experiments.s2_latent_topology import evaluate as s2eval
from src.experiments.s3_set_residual import evaluate as s3eval
from src.experiments.s3_set_residual.model import ResidualConfig, SetResidualModel
from src.score_universe import save_scores

pytestmark = pytest.mark.unit

INPUT_DIM = 8
D_MODEL = 16
SAB_LAYERS = 2
HEADS = 2
PMA_SEEDS = 2
P_DIM = 6
HEAD_HIDDEN = 12
STRATEGY = "toy"
NODES = [f"node_{i:06d}" for i in range(16)]


# --------------------------------------------------------------------------- fixture builders


def _write_feature_root(tmp_path: Path, node_ids: list[str]) -> Path:
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((3, INPUT_DIM)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": INPUT_DIM, "max_sequence_length": 32}
        )
    )
    return root


def _make_test_graph() -> nx.Graph:
    base = nx.connected_watts_strogatz_graph(16, k=4, p=0.3, seed=1)
    return nx.relabel_nodes(base, dict(enumerate(NODES)))


def _make_buckets() -> dict[int, list[set[str]]]:
    rng = np.random.default_rng(3)
    return {
        size: [set(rng.choice(NODES, size=size, replace=False).tolist()) for _ in range(4)]
        for size in (4, 5)
    }


def _write_benchmark(tmp_path: Path, g: nx.Graph, buckets: dict[int, list[set[str]]]) -> Path:
    data_root = tmp_path / "data"
    strategy_dir = data_root / "benchmark_2025_neurips" / STRATEGY
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(g, f)
    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump(buckets, f)
    return data_root


def _all_canonical_pairs(node_ids: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, u in enumerate(node_ids):
        for v in node_ids[i:]:
            pairs.append((u, v))
    return pairs


def _write_b0_universe(
    tmp_path: Path,
    node_ids: list[str],
    *,
    drop_fraction: float = 0.0,
    name: str = "b0_universe.npz",
) -> Path:
    """Write a B0 candidate-universe artifact; `drop_fraction` deliberately omits some pairs."""
    pairs = _all_canonical_pairs(node_ids)
    rng = np.random.default_rng(11)
    if drop_fraction > 0.0:
        keep = rng.random(len(pairs)) >= drop_fraction
        pairs = [p for p, k in zip(pairs, keep.tolist(), strict=True) if k]
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    logits = rng.normal(size=len(pairs)).astype(np.float32)
    labels = np.zeros(len(pairs), dtype=np.int8)
    path = tmp_path / name
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logits,
        label=labels,
        row_start=0,
        meta={
            "checkpoint_id": "deadbeef",
            "model_family": "v3_1",
            "pairs_source": "candidate",
            "strategy": STRATEGY,
            "num_rows": len(pairs),
            "created_utc": "2026-08-18T00:00:00+00:00",
            "torch_version": "2.10.0",
        },
    )
    return path


def _build_context(tmp_path: Path, b0_universe: Path) -> s2eval.EvalContext:
    g = _make_test_graph()
    buckets = _make_buckets()
    data_root = _write_benchmark(tmp_path, g, buckets)
    store = FeatureStore(_write_feature_root(tmp_path, NODES))
    return s2eval.build_eval_context(
        data_root=data_root,
        strategy=STRATEGY,
        store=store,
        cache_dir=tmp_path / "cache",
        b0_universe=b0_universe,
    )


def _tiny_cfg(mode: str) -> ResidualConfig:
    return ResidualConfig(
        mode=mode,  # type: ignore[arg-type]
        d_in=INPUT_DIM,
        d_model=D_MODEL,
        sab_layers=SAB_LAYERS,
        heads=HEADS,
        pma_seeds=PMA_SEEDS,
        p_dim=P_DIM,
        head_hidden=HEAD_HIDDEN,
    )


def _write_checkpoint(run_dir: Path, cfg: ResidualConfig, *, seed: int = 0) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = SetResidualModel(cfg)
    # Perturb pair_out off zero-init so MODEL != B0 in general (mirrors model.py's
    # own test convention for exercising a "trained" checkpoint).
    with torch.no_grad():
        model.pair_out.weight.normal_(std=0.1)
        model.pair_out.bias.normal_(std=0.1)
    path = run_dir / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "mode": cfg.mode,
                "d_in": cfg.d_in,
                "d_model": cfg.d_model,
                "sab_layers": cfg.sab_layers,
                "heads": cfg.heads,
                "pma_seeds": cfg.pma_seeds,
                "p_dim": cfg.p_dim,
                "head_hidden": cfg.head_hidden,
            },
        },
        path,
    )
    return path


# --------------------------------------------------------------------------- _b0_logit_matrix


def test_b0_logit_matrix_returns_raw_logits_not_probs(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    node_index = {node_id: i for i, node_id in enumerate(ctx.nodes)}
    ordered = sorted(NODES[:4])

    mat, missing, missing_count = s3eval._b0_logit_matrix(ctx, artifact.logit, node_index, ordered)

    assert missing_count == 0
    assert not missing.any()
    u, v = ordered[0], ordered[2]
    expected_logit = dict(zip(artifact.pairs(), artifact.logit.tolist(), strict=True))[(u, v)]
    i, j = ordered.index(u), ordered.index(v)
    assert mat[i, j] == pytest.approx(expected_logit)
    assert mat[i, j] != pytest.approx(1.0 / (1.0 + np.exp(-expected_logit)))  # not sigmoided
    assert mat[i, j] == mat[j, i]
    assert np.diag(mat).sum() == 0.0


def test_b0_logit_matrix_flags_missing_pairs(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES, drop_fraction=0.3)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    present_pairs = set(artifact.pairs())
    node_index = {node_id: i for i, node_id in enumerate(ctx.nodes)}
    ordered = sorted(NODES[:6])

    mat, missing, missing_count = s3eval._b0_logit_matrix(ctx, artifact.logit, node_index, ordered)

    assert missing_count > 0
    iu, iv = np.triu_indices(len(ordered), k=1)
    for i, j in zip(iu.tolist(), iv.tolist(), strict=True):
        pair = (ordered[i], ordered[j]) if ordered[i] <= ordered[j] else (ordered[j], ordered[i])
        assert missing[i, j] == (pair not in present_pairs)


# --------------------------------------------------------------------------- pairs/delta round trip


def test_all_pairs_tensor_and_delta_to_matrix_round_trip() -> None:
    b_count, n = 3, 5
    device = torch.device("cpu")
    pairs = s3eval._all_pairs_tensor(b_count, n, device)
    iu, iv = np.triu_indices(n, k=1)
    assert pairs.shape == (b_count * len(iu), 3)

    delta = torch.arange(pairs.shape[0], dtype=torch.float32)
    mat = s3eval._delta_to_matrix(delta, b_count=b_count, n=n)

    assert mat.shape == (b_count, n, n)
    for row in range(b_count):
        assert np.allclose(np.diag(mat[row]), 0.0)
        assert np.allclose(mat[row], mat[row].T)
    # Row 0's first pair (i=0, j=1) is flat index 0.
    assert mat[0, 0, 1] == pytest.approx(0.0)
    assert mat[0, 1, 0] == pytest.approx(0.0)


def test_all_pairs_tensor_handles_singleton_sets() -> None:
    pairs = s3eval._all_pairs_tensor(2, 1, torch.device("cpu"))
    assert pairs.shape == (0, 3)


# --------------------------------------------------------------------------- AUPRC exclusion


def test_compute_set_readout_s3_excludes_missing_pairs_from_auprc() -> None:
    node_ids = ["a", "b", "c", "d"]
    true_adj = np.zeros((4, 4))
    for u, v in (("a", "b"), ("c", "d")):
        i, j = node_ids.index(u), node_ids.index(v)
        true_adj[i, j] = true_adj[j, i] = 1.0
    ref = nx.Graph()
    ref.add_nodes_from(node_ids)
    ref.add_edges_from([("a", "b"), ("c", "d")])
    rng = np.random.default_rng(4)
    raw = rng.random((4, 4))
    probs = (raw + raw.T) / 2.0
    np.fill_diagonal(probs, 0.0)

    missing_none = np.zeros((4, 4), dtype=bool)
    readout_full = s3eval._compute_set_readout_s3(
        probs=probs,
        true_adj=true_adj,
        node_ids=node_ids,
        ref_subgraph=ref,
        target_edges=2,
        missing_mask=missing_none,
    )
    assert readout_full.auprc is not None

    # Mask every non-(a,b) pair as missing, leaving a single-class AUPRC input -> None.
    missing_all_but_one = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(missing_all_but_one, False)
    ia, ib = node_ids.index("a"), node_ids.index("b")
    missing_all_but_one[ia, ib] = missing_all_but_one[ib, ia] = False
    readout_masked = s3eval._compute_set_readout_s3(
        probs=probs,
        true_adj=true_adj,
        node_ids=node_ids,
        ref_subgraph=ref,
        target_edges=2,
        missing_mask=missing_all_but_one,
    )
    assert readout_masked.auprc is None  # only one present pair, single-class


# --------------------------------------------------------------------------- ECE / calibration


def test_ece_zero_for_perfectly_calibrated_predictions() -> None:
    # Bin-average confidence must equal bin-average accuracy exactly: 0.0/1.0
    # predictions matching 0/1 labels land each pair in a bin whose mean
    # confidence and mean accuracy coincide, giving zero contribution.
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert s3eval._ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_miscalibrated_predictions() -> None:
    probs = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 0])
    assert s3eval._ece(probs, labels) == pytest.approx(0.9, abs=1e-9)


def test_calibration_block_implied_edge_ratio() -> None:
    probs = np.array([0.5, 0.5, 0.0, 0.0])
    labels = np.array([1, 0, 0, 0])
    block = s3eval._calibration_block(probs, labels, true_edges_total=2)
    assert block["implied_edge_count_ratio"] == pytest.approx(1.0 / 2.0)
    assert block["n_pairs"] == 4
    assert block["ece"] is not None


def test_calibration_block_handles_empty_pool() -> None:
    block = s3eval._calibration_block(np.array([]), np.array([]), true_edges_total=0)
    assert block["ece"] is None
    assert block["implied_edge_count_ratio"] is None
    assert block["n_pairs"] == 0


# --------------------------------------------------------------------------- paired-delta bootstrap


def test_ci_entry_is_deterministic_and_reports_raw_values() -> None:
    values = [0.1, -0.2, 0.05, 0.3]
    first = s3eval._ci_entry(values, arm_code=101, size=20, metric_code=1)
    second = s3eval._ci_entry(values, arm_code=101, size=20, metric_code=1)
    assert first == second
    assert first["values"] == values
    assert first["n"] == 4
    assert first["ci_lo"] <= first["mean"] <= first["ci_hi"]


def test_ci_entry_empty_values_are_none() -> None:
    entry = s3eval._ci_entry([], arm_code=101, size=20, metric_code=1)
    assert entry["mean"] is None
    assert entry["ci_lo"] is None
    assert entry["ci_hi"] is None
    assert entry["n"] == 0


def test_build_paired_deltas_pools_across_sizes() -> None:
    delta_raw = {
        "model_minus_b0__gs": {"4": [0.1, 0.2], "5": [0.3]},
        "model_minus_b0__auprc": {"4": [0.05], "5": []},
    }
    out = s3eval._build_paired_deltas(delta_raw, ["b0", "model"], [4, 5])
    assert out["model_minus_b0"]["gs"]["pooled"]["n"] == 3
    assert out["model_minus_b0"]["gs"]["pooled"]["values"] == [0.1, 0.2, 0.3]
    assert out["model_minus_b0"]["auprc"]["by_size"]["5"]["n"] == 0
    assert "shuf_minus_b0" not in out


def test_build_paired_deltas_includes_shuf_when_present() -> None:
    delta_raw = {"shuf_minus_b0__gs": {"4": [0.0]}}
    out = s3eval._build_paired_deltas(delta_raw, ["b0", "model", "shuf"], [4])
    assert "shuf_minus_b0" in out


# --------------------------------------------------------------------------- checkpoint loading


def test_load_residual_checkpoint_round_trips_config_and_weights(tmp_path: Path) -> None:
    cfg = _tiny_cfg("res")
    path = _write_checkpoint(tmp_path / "run", cfg)

    model, loaded_cfg, checkpoint_id = s3eval.load_residual_checkpoint(
        tmp_path / "run", torch.device("cpu")
    )

    assert loaded_cfg == cfg
    assert isinstance(checkpoint_id, str) and len(checkpoint_id) == 16
    original = torch.load(path, weights_only=True)["model_state"]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[key])


def test_load_residual_checkpoint_fills_missing_dims_with_defaults(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = ResidualConfig(mode="pair")
    model = SetResidualModel(cfg)
    torch.save({"model_state": model.state_dict(), "config": {"mode": "pair"}}, run_dir / "best.pt")

    _, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    assert loaded_cfg == ResidualConfig(mode="pair")


def test_load_residual_checkpoint_rejects_missing_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"config": {"mode": "res"}}, run_dir / "best.pt")

    with pytest.raises(ValueError, match="model_state"):
        s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))


def test_load_residual_checkpoint_rejects_unknown_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = ResidualConfig(mode="res")
    model = SetResidualModel(cfg)
    torch.save(
        {"model_state": model.state_dict(), "config": {"mode": "bogus"}}, run_dir / "best.pt"
    )

    with pytest.raises(ValueError, match="mode"):
        s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))


# --------------------------------------------------------------------------- run_s3_eval end-to-end


def test_run_s3_eval_zero_init_sanity_passes(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    payload = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=0,
        device="cpu",
        shuf=False,
        sanity_zero_init=True,
    )

    assert payload["sanity_zero_init"]["ran"] is True
    assert payload["sanity_zero_init"]["passed"] is True


def test_run_s3_eval_zero_init_sanity_raises_on_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    class _BrokenZeroModel:
        """Stands in for a "zero-init" model that (incorrectly) returns a nonzero delta."""

        def __init__(self, _cfg: ResidualConfig) -> None:
            pass

        def to(self, _device: torch.device) -> _BrokenZeroModel:
            return self

        def eval(self) -> _BrokenZeroModel:
            return self

        def __call__(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pairs: torch.Tensor,
            x_set: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.ones(pairs.shape[0], dtype=torch.float32)

    monkeypatch.setattr(s3eval, "SetResidualModel", _BrokenZeroModel)

    with pytest.raises(ValueError, match="sanity-zero-init"):
        s3eval.run_s3_eval(
            ctx=ctx,
            model=model,
            cfg=loaded_cfg,
            base_logit=artifact.logit,
            seed=0,
            device="cpu",
            shuf=False,
            sanity_zero_init=True,
        )


def test_run_s3_eval_pair_mode_shuf_equals_model_since_x_set_is_ignored(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("pair")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    payload = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=7,
        device="cpu",
        shuf=True,
        sanity_zero_init=False,
    )

    model_macro = payload["arms"]["model"]["macro"]
    shuf_macro = payload["arms"]["shuf"]["macro"]
    for metric in ("gs", "rd", "auprc"):
        if model_macro[metric] is None:
            assert shuf_macro[metric] is None
        else:
            assert shuf_macro[metric] == pytest.approx(model_macro[metric])


def test_run_s3_eval_res_mode_shuf_changes_backbone_output(tmp_path: Path) -> None:
    # Exercise the SHUF wiring directly at the model-forward level (not through
    # density-matched assembly, which can quantize a small perturbation away):
    # the same checkpoint's delta must differ between x_set=None and a permuted
    # x_set for "res" mode, since the backbone (unlike p_proj) reads x_set.
    cfg = _tiny_cfg("res")
    torch.manual_seed(0)
    model = SetResidualModel(cfg)
    with torch.no_grad():
        model.pair_out.weight.normal_(std=0.1)
        model.pair_out.bias.normal_(std=0.1)
    model.eval()

    b_count, n = 2, 6
    x = torch.randn(b_count, n, INPUT_DIM)
    mask = torch.ones(b_count, n)
    pairs = s3eval._all_pairs_tensor(b_count, n, torch.device("cpu"))
    # A full reversal is never the identity permutation for n=6, so this is a
    # deterministic (not RNG-luck-dependent) way to guarantee correspondence
    # is actually destroyed.
    x_shuf = x.flip(dims=(1,))

    with torch.no_grad():
        delta_plain = model(x, mask, pairs, x_set=None)
        delta_shuf = model(x, mask, pairs, x_set=x_shuf)

    assert not torch.allclose(delta_plain, delta_shuf)


def test_run_s3_eval_reports_missing_pairs_and_excludes_from_auprc(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES, drop_fraction=0.4)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    payload = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=0,
        device="cpu",
        shuf=False,
        sanity_zero_init=False,
    )

    assert payload["meta"]["missing_pairs_total"] > 0


def test_run_s3_eval_reports_fixed_threshold_readouts_per_set(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    payload = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=0,
        device="cpu",
        shuf=False,
        sanity_zero_init=False,
    )

    assert "full_region" not in payload
    for arm_name in ("b0", "model"):
        arm = payload["arms"][arm_name]
        assert arm["macro"]["gs_t05"] is not None
        assert arm["macro"]["rd_t05"] is not None
        for stat in ("degree", "clustering", "spectral"):
            assert arm["mmd_t05_macro"][stat] is not None


def test_run_s3_eval_deterministic_given_same_seed(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    ctx = _build_context(tmp_path, b0_path)
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)
    cfg = _tiny_cfg("diag")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))

    payload1 = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=3,
        device="cpu",
        shuf=True,
        sanity_zero_init=False,
    )
    payload2 = s3eval.run_s3_eval(
        ctx=ctx,
        model=model,
        cfg=loaded_cfg,
        base_logit=artifact.logit,
        seed=3,
        device="cpu",
        shuf=True,
        sanity_zero_init=False,
    )

    assert json.dumps(payload1, sort_keys=True) == json.dumps(payload2, sort_keys=True)


def test_run_s3_eval_requires_b0_lookup(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path / "with_b0", NODES)
    ctx_with_b0 = _build_context(tmp_path / "with_b0", b0_path)
    no_b0_root = tmp_path / "without_b0"
    g = _make_test_graph()
    buckets = _make_buckets()
    data_root = _write_benchmark(no_b0_root, g, buckets)
    store = FeatureStore(_write_feature_root(no_b0_root, NODES))
    ctx_without_b0 = s2eval.build_eval_context(
        data_root=data_root,
        strategy=STRATEGY,
        store=store,
        cache_dir=no_b0_root / "cache2",
        b0_universe=None,
    )
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)
    model, loaded_cfg, _ = s3eval.load_residual_checkpoint(run_dir, torch.device("cpu"))
    from src.score_universe import load_scores

    artifact = load_scores(b0_path)

    with pytest.raises(ValueError, match="b0_universe"):
        s3eval.run_s3_eval(
            ctx=ctx_without_b0,
            model=model,
            cfg=loaded_cfg,
            base_logit=artifact.logit,
            seed=0,
            device="cpu",
            shuf=False,
            sanity_zero_init=False,
        )
    assert ctx_with_b0.b0_lookup is not None  # sanity: the "with b0" context is usable


# --------------------------------------------------------------------------- CLI: run_eval


def _build_cli_parser() -> argparse.ArgumentParser:
    """A stand-in for Task 3's shared top-level parser (data-root/strategy/device/seed)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    s3eval.register_eval_args(parser)
    return parser


def test_run_eval_writes_report_json(tmp_path: Path) -> None:
    b0_path = _write_b0_universe(tmp_path, NODES)
    g = _make_test_graph()
    buckets = _make_buckets()
    data_root = _write_benchmark(tmp_path, g, buckets)
    feature_root = data_root / s3eval._FEATURES_SUBDIR
    feature_root.parent.mkdir(parents=True, exist_ok=True)
    written_root = _write_feature_root(tmp_path, NODES)
    written_root.rename(feature_root)
    cfg = _tiny_cfg("res")
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, cfg)

    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--data-root",
            str(data_root),
            "--strategy",
            STRATEGY,
            "--run-dir",
            str(run_dir),
            "--b0-universe",
            str(b0_path),
        ]
    )

    out = s3eval.run_eval(args)

    assert out == run_dir / "report.json"
    payload = json.loads(out.read_text())
    assert "arms" in payload
    assert payload["provenance"]["checkpoint_id"]


def test_run_eval_requires_run_dir_and_b0_universe() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    with pytest.raises(ValueError, match="--run-dir"):
        s3eval.run_eval(args)


def test_run_eval_aggregate_pools_reports(tmp_path: Path) -> None:
    report_a = tmp_path / "a" / "report.json"
    report_b = tmp_path / "b" / "report.json"
    for path, values in ((report_a, [0.1, 0.2]), (report_b, [0.3])):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "paired_deltas": {
                        "model_minus_b0": {
                            "gs": {"by_size": {"4": {"values": values}}},
                            "auprc": {"by_size": {"4": {"values": []}}},
                        }
                    }
                }
            )
        )

    parser = _build_cli_parser()
    args = parser.parse_args(["--aggregate", str(report_a), str(report_b)])

    out = s3eval.run_eval(args)

    assert out == report_a.parent / "pooled_report.json"
    pooled = json.loads(out.read_text())
    gs_pooled = pooled["paired_deltas"]["model_minus_b0"]["gs"]["pooled"]
    assert gs_pooled["n"] == 3
    assert sorted(gs_pooled["values"]) == [0.1, 0.2, 0.3]


def test_aggregate_reports_matches_run_eval_aggregate_path(tmp_path: Path) -> None:
    report_a = tmp_path / "report_a.json"
    report_a.write_text(
        json.dumps(
            {"paired_deltas": {"model_minus_b0": {"gs": {"by_size": {"4": {"values": [0.5]}}}}}}
        )
    )

    result = s3eval.aggregate_reports([report_a])

    assert result["source_reports"] == [str(report_a)]
    assert result["paired_deltas"]["model_minus_b0"]["gs"]["pooled"]["values"] == [0.5]


def test_aggregate_reports_same_mode_pools_and_records_mode(tmp_path: Path) -> None:
    report_a = tmp_path / "a" / "report.json"
    report_b = tmp_path / "b" / "report.json"
    for path, values in ((report_a, [0.1, 0.2]), (report_b, [0.3])):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "meta": {"mode": "res"},
                    "paired_deltas": {
                        "model_minus_b0": {"gs": {"by_size": {"4": {"values": values}}}}
                    },
                }
            )
        )

    result = s3eval.aggregate_reports([report_a, report_b])

    assert result["mode"] == "res"
    assert result["paired_deltas"]["model_minus_b0"]["gs"]["pooled"]["n"] == 3


def test_aggregate_reports_mixed_mode_raises(tmp_path: Path) -> None:
    report_a = tmp_path / "a" / "report.json"
    report_b = tmp_path / "b" / "report.json"
    for path, mode, values in ((report_a, "res", [0.1]), (report_b, "pair", [0.3])):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "meta": {"mode": mode},
                    "paired_deltas": {
                        "model_minus_b0": {"gs": {"by_size": {"4": {"values": values}}}}
                    },
                }
            )
        )

    with pytest.raises(ValueError, match="res"):
        s3eval.aggregate_reports([report_a, report_b])
