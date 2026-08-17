"""Tests for src.experiments.s2_latent_topology.evaluate: the S2 evaluation pipeline.

All CPU, synthetic fixtures: a 24-node test graph (2 self-loops), buckets
`{6: [4 sets], 8: [4 sets]}` sampled by hand, feature dim 8, and tiny
untrained checkpoints built directly against the pinned `ae.pt`/`prior.pt`/
`det.pt`/`marg.pt` contract (no `train.py` involved).
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.experiments.s2_latent_topology import evaluate as s2eval
from src.experiments.s2_latent_topology.model import (
    DetTwin,
    MargDegreeRegressor,
    RegionAutoencoder,
    SetDenoiser,
)
from src.score_universe import save_scores

pytestmark = pytest.mark.unit

INPUT_DIM = 8
TOKEN_LEN = 5
Z_DIM = 4
AE_DIM = 16
AE_LAYERS = 2
AE_RRWP_K = 3
AE_HEADS = 4
AE_SEEDS = 2
AE_D_MODEL = 32
SET_WIDTH = 32
SET_LAYERS = 2
SET_HEADS = 2
SET_PMA_SEEDS = 2
TIME_DIM = 8
MARG_HIDDEN = 16
STRATEGY = "toy"

_CFG: dict[str, object] = {
    "z_dim": Z_DIM,
    "ae_dim": AE_DIM,
    "ae_layers": AE_LAYERS,
    "ae_rrwp_k": AE_RRWP_K,
    "ae_heads": AE_HEADS,
    "ae_seeds": AE_SEEDS,
    "ae_d_model": AE_D_MODEL,
    "set_width": SET_WIDTH,
    "set_layers": SET_LAYERS,
    "set_heads": SET_HEADS,
    "set_pma_seeds": SET_PMA_SEEDS,
    "time_dim": TIME_DIM,
    "marg_hidden": MARG_HIDDEN,
}

NODES = [f"node_{i:06d}" for i in range(24)]


# --------------------------------------------------------------------------- fixture builders


def _write_feature_root(tmp_path: Path, node_ids: list[str]) -> Path:
    """Build a tiny synthetic FeatureStore root: metadata.json + index.json + per-node .pt."""
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((TOKEN_LEN, INPUT_DIM)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": INPUT_DIM, "max_sequence_length": 1024}
        )
    )
    return root


def _make_test_graph() -> nx.Graph:
    """A 24-node connected graph with two self-loops."""
    base = nx.connected_watts_strogatz_graph(24, k=4, p=0.3, seed=1)
    g = nx.relabel_nodes(base, dict(enumerate(NODES)))
    g.add_edge(NODES[0], NODES[0])
    g.add_edge(NODES[5], NODES[5])
    return g


def _make_buckets() -> dict[int, list[set[str]]]:
    """Hand-sampled buckets: 4 six-node sets and 4 eight-node sets."""
    rng = np.random.default_rng(3)
    return {
        size: [set(rng.choice(NODES, size=size, replace=False).tolist()) for _ in range(4)]
        for size in (6, 8)
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


def _write_checkpoints(tmp_path: Path) -> dict[str, Path]:
    """Save tiny untrained ae/prior/det/marg checkpoints matching the pinned contract."""
    torch.manual_seed(0)
    ae = RegionAutoencoder(
        in_dim=INPUT_DIM,
        z_dim=Z_DIM,
        dim=AE_DIM,
        layers=AE_LAYERS,
        rrwp_k=AE_RRWP_K,
        n_heads=AE_HEADS,
        seeds=AE_SEEDS,
        d_model=AE_D_MODEL,
    )
    prior = SetDenoiser(
        in_dim=INPUT_DIM,
        z_dim=Z_DIM,
        width=SET_WIDTH,
        sab_layers=SET_LAYERS,
        heads=SET_HEADS,
        pma_seeds=SET_PMA_SEEDS,
        time_dim=TIME_DIM,
    )
    det = DetTwin(
        in_dim=INPUT_DIM,
        z_dim=Z_DIM,
        width=SET_WIDTH,
        sab_layers=SET_LAYERS,
        heads=SET_HEADS,
        pma_seeds=SET_PMA_SEEDS,
    )
    marg = MargDegreeRegressor(in_dim=INPUT_DIM, hidden=MARG_HIDDEN)

    paths: dict[str, Path] = {}
    ae_path = tmp_path / "ae.pt"
    torch.save(
        {
            "model": ae.state_dict(),
            "config": dict(_CFG),
            "in_dim": INPUT_DIM,
            "latent_mean": torch.zeros(1 + Z_DIM),
            "latent_std": torch.ones(1 + Z_DIM),
            "best_epoch": 3,
            "val_loss": 0.1,
        },
        ae_path,
    )
    paths["ae"] = ae_path

    for name, model in (("prior", prior), ("det", det), ("marg", marg)):
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": dict(_CFG),
                "in_dim": INPUT_DIM,
                "best_epoch": 3,
                "val_loss": 0.2,
            },
            path,
        )
        paths[name] = path
    return paths


def _all_canonical_pairs(node_ids: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, u in enumerate(node_ids):
        for v in node_ids[i:]:
            pairs.append((u, v))
    return pairs


def _write_b0_universe(tmp_path: Path, node_ids: list[str]) -> Path:
    """Write a full candidate-universe B0 scores artifact over `node_ids`."""
    pairs = _all_canonical_pairs(node_ids)
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    rng = np.random.default_rng(11)
    logits = rng.normal(size=len(pairs)).astype(np.float32)
    labels = np.zeros(len(pairs), dtype=np.int8)
    path = tmp_path / "b0_universe.npz"
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
            "created_utc": "2026-08-16T00:00:00+00:00",
            "torch_version": "2.10.0",
        },
    )
    return path


def _build_context(tmp_path: Path, *, with_b0: bool) -> s2eval.EvalContext:
    g = _make_test_graph()
    buckets = _make_buckets()
    data_root = _write_benchmark(tmp_path, g, buckets)
    store = FeatureStore(_write_feature_root(tmp_path, NODES))
    b0_universe = _write_b0_universe(tmp_path, NODES) if with_b0 else None
    return s2eval.build_eval_context(
        data_root=data_root,
        strategy=STRATEGY,
        store=store,
        cache_dir=tmp_path / "cache",
        b0_universe=b0_universe,
    )


def _assert_finite_or_none(value: object) -> None:
    """Recursively assert every float in a JSON-ready payload is finite or `None`."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite float in payload: {value}"
        return
    if isinstance(value, dict):
        for v in value.values():
            _assert_finite_or_none(v)
        return
    if isinstance(value, list):
        for v in value:
            _assert_finite_or_none(v)
        return
    raise AssertionError(f"unexpected payload value type: {type(value)}")


# --------------------------------------------------------------------------- density matching


def test_density_matched_graph_has_exact_target_edge_count() -> None:
    node_ids = ["a", "b", "c", "d", "e"]
    true_adj = np.zeros((5, 5), dtype=np.float64)
    for u, v in (("a", "b"), ("a", "c"), ("b", "c")):
        i, j = node_ids.index(u), node_ids.index(v)
        true_adj[i, j] = 1.0
        true_adj[j, i] = 1.0
    ref_subgraph = nx.Graph()
    ref_subgraph.add_nodes_from(node_ids)
    ref_subgraph.add_edges_from([("a", "b"), ("a", "c"), ("b", "c")])

    rng = np.random.default_rng(5)
    raw = rng.random((5, 5))
    probs = (raw + raw.T) / 2.0
    np.fill_diagonal(probs, 0.0)
    assert len(np.unique(probs[np.triu_indices(5, k=1)])) == 10  # no ties

    readout = s2eval._compute_set_readout(
        probs=probs, true_adj=true_adj, node_ids=node_ids, ref_subgraph=ref_subgraph, target_edges=3
    )

    assert readout.g_pred.number_of_edges() == 3


# --------------------------------------------------------------------------- spearman / hub recall


def test_safe_spearman_hand_computed_values() -> None:
    perfect = s2eval._safe_spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    inverse = s2eval._safe_spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0]))
    constant = s2eval._safe_spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))

    assert perfect == pytest.approx(1.0)
    assert inverse == pytest.approx(-1.0)
    assert constant is None


def test_hub_recall_hand_computed_overlap() -> None:
    node_ids = [f"n{i:02d}" for i in range(20)]
    deg_hat = np.zeros(20)
    deg_hat[0] = 100.0
    deg_hat[1] = 90.0
    true_degree = np.zeros(20)
    true_degree[1] = 100.0
    true_degree[2] = 90.0

    # ceil(0.1 * 20) == 2; top_hat = {n00, n01}, top_true = {n01, n02} -> overlap 1/2.
    recall = s2eval._hub_recall(deg_hat, true_degree, node_ids)

    assert recall == pytest.approx(0.5)


def test_hub_recall_none_when_top_k_is_zero() -> None:
    node_ids = ["a", "b", "c"]
    deg_hat = np.array([1.0, 2.0, 3.0])
    true_degree = np.array([3.0, 2.0, 1.0])

    # ceil(0.1 * 3) == 1, so this is a sanity check that k > 0 here (not the None path);
    # None only triggers for an empty node set, which callers never construct.
    assert s2eval._hub_recall(deg_hat, true_degree, node_ids) is not None
    assert s2eval._hub_recall(np.array([]), np.array([]), []) is None


# --------------------------------------------------------------------------- single-class AUPRC


def test_all_positive_set_has_none_auprc() -> None:
    node_ids = ["a", "b", "c"]
    true_adj = np.ones((3, 3)) - np.eye(3)
    ref_subgraph = nx.complete_graph(node_ids)
    rng = np.random.default_rng(9)
    raw = rng.random((3, 3))
    probs = (raw + raw.T) / 2.0
    np.fill_diagonal(probs, 0.0)

    readout = s2eval._compute_set_readout(
        probs=probs, true_adj=true_adj, node_ids=node_ids, ref_subgraph=ref_subgraph, target_edges=3
    )

    assert readout.auprc is None


def test_process_arm_size_counts_single_class_auprc_as_none() -> None:
    # Two sets (a bucket needs >= 2 reference samples for the MMD floor split),
    # both complete graphs so every set's pair labels are single-class (all 1).
    node_ids = ["a", "b", "c"]
    true_adj = np.stack([np.ones((3, 3)) - np.eye(3)] * 2)
    ref_subgraph = nx.complete_graph(node_ids)
    rng = np.random.default_rng(13)
    raw = rng.random((2, 3, 3))
    probs = np.stack([(mat + mat.T) / 2.0 for mat in raw])
    for mat in probs:
        np.fill_diagonal(mat, 0.0)
    ref_descs = [s2eval._descriptors(ref_subgraph), s2eval._descriptors(ref_subgraph)]

    result = s2eval._process_arm_size(
        arm_code=s2eval._ARM_CODES["det"],
        size=3,
        seed=0,
        mean_probs=probs,
        draw0_probs=probs,
        ordered_sets=[node_ids, node_ids],
        true_adj=true_adj,
        ref_subgraphs=[ref_subgraph, ref_subgraph],
        ref_descs=ref_descs,
    )

    assert result["per_size"]["auprc"]["n"] == 0
    assert result["per_size"]["auprc"]["n_none"] == 2
    assert result["per_size"]["auprc"]["mean"] is None


def test_process_arm_size_skips_and_counts_zero_edge_sets() -> None:
    node_ids = ["a", "b", "c", "d"]
    true_adj = np.zeros((1, 4, 4))
    ref_subgraph = nx.Graph()
    ref_subgraph.add_nodes_from(node_ids)
    probs = np.zeros((1, 4, 4))
    ref_descs = [s2eval._descriptors(ref_subgraph)]

    result = s2eval._process_arm_size(
        arm_code=s2eval._ARM_CODES["ae"],
        size=4,
        seed=0,
        mean_probs=probs,
        draw0_probs=probs,
        ordered_sets=[node_ids],
        true_adj=true_adj,
        ref_subgraphs=[ref_subgraph],
        ref_descs=ref_descs,
    )

    assert result["per_size"]["n_skipped"] == 1
    assert result["per_size"]["n_sets"] == 1
    assert result["per_size"]["gs"]["n"] == 0


# --------------------------------------------------------------------------- MARG quota rounding


def test_largest_remainder_quota_sums_exactly_and_breaks_ties_by_order() -> None:
    deg_hat = np.array([1.0, 1.0, 1.0])

    quotas = s2eval._largest_remainder_quota(deg_hat, 4)

    assert int(quotas.sum()) == 4
    assert quotas.tolist() == [2, 1, 1]


def test_largest_remainder_quota_exact_case_needs_no_rounding() -> None:
    deg_hat = np.array([1.0, 2.0, 3.0, 4.0])

    quotas = s2eval._largest_remainder_quota(deg_hat, 10)

    assert quotas.tolist() == [1, 2, 3, 4]
    assert int(quotas.sum()) == 10


def test_largest_remainder_quota_zero_total_is_all_zero() -> None:
    quotas = s2eval._largest_remainder_quota(np.array([5.0, 5.0]), 0)
    assert quotas.tolist() == [0, 0]


# --------------------------------------------------------------------------- bootstrap CI


def test_bootstrap_ci_is_deterministic_given_same_seed() -> None:
    values = [0.1, 0.5, 0.9, 0.3, 0.7]

    first = s2eval._bootstrap_ci(values, seed=0, arm_code=1, size=20, metric_code=1)
    second = s2eval._bootstrap_ci(values, seed=0, arm_code=1, size=20, metric_code=1)

    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_bootstrap_ci_differs_across_metric_codes() -> None:
    values = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.6]

    a = s2eval._bootstrap_ci(values, seed=0, arm_code=1, size=20, metric_code=1)
    b = s2eval._bootstrap_ci(values, seed=0, arm_code=1, size=20, metric_code=2)

    assert a[0] == b[0]  # same mean
    assert a[1] != b[1] or a[2] != b[2]  # different bootstrap resample stream


# --------------------------------------------------------------------------- b0 lookup


def test_build_eval_context_b0_lookup_resolves_both_orders(tmp_path: Path) -> None:
    ctx = _build_context(tmp_path, with_b0=True)
    assert ctx.b0_probs is not None
    assert ctx.b0_lookup is not None

    n = len(ctx.nodes)
    pairs = _all_canonical_pairs(ctx.nodes)
    target = (ctx.nodes[2], ctx.nodes[5])
    expected_row = pairs.index(target)

    iu, iv = ctx.nodes.index(target[0]), ctx.nodes.index(target[1])
    row_forward = ctx.b0_lookup[iu * n + iv]
    row_backward = ctx.b0_lookup[iv * n + iu]

    assert row_forward == row_backward == expected_row
    assert ctx.b0_probs[expected_row] == pytest.approx(ctx.b0_probs[row_forward])
    assert 0.0 <= float(ctx.b0_probs[expected_row]) <= 1.0


def test_build_eval_context_without_b0_leaves_lookup_none(tmp_path: Path) -> None:
    ctx = _build_context(tmp_path, with_b0=False)
    assert ctx.b0_probs is None
    assert ctx.b0_lookup is None


# --------------------------------------------------------------------------- end-to-end


def test_run_s2_eval_end_to_end_and_deterministic(tmp_path: Path) -> None:
    ctx = _build_context(tmp_path, with_b0=True)
    checkpoints = _write_checkpoints(tmp_path)

    kwargs = {
        "ctx": ctx,
        "ae_path": checkpoints["ae"],
        "prior_path": checkpoints["prior"],
        "det_path": checkpoints["det"],
        "marg_path": checkpoints["marg"],
        "k_draws": 3,
        "flow_steps": 4,
        "seed": 0,
        "device": "cpu",
        "full_region": True,
    }

    payload1 = s2eval.run_s2_eval(**kwargs)  # type: ignore[arg-type]
    text1 = json.dumps(payload1, sort_keys=True)
    assert isinstance(text1, str)

    arms1 = payload1["arms"]
    assert isinstance(arms1, dict)
    for arm_name in ("gen", "unc", "shuf", "det", "ae", "b0", "marg"):
        assert arm_name in arms1
    for arm_name in ("gen", "unc", "shuf"):
        entry = arms1[arm_name]
        assert isinstance(entry, dict)
        assert "coherence" in entry
    assert "full_region" in payload1
    full_region = payload1["full_region"]
    assert isinstance(full_region, dict)
    ae_block = full_region["ae"]
    assert isinstance(ae_block, dict)
    assert ae_block["skipped"] is True

    _assert_finite_or_none(payload1)

    payload2 = s2eval.run_s2_eval(**kwargs)  # type: ignore[arg-type]
    text2 = json.dumps(payload2, sort_keys=True)
    assert text1 == text2


def test_run_s2_eval_without_b0_or_marg_omits_those_arms(tmp_path: Path) -> None:
    ctx = _build_context(tmp_path, with_b0=False)
    checkpoints = _write_checkpoints(tmp_path)

    payload = s2eval.run_s2_eval(
        ctx=ctx,
        ae_path=checkpoints["ae"],
        prior_path=checkpoints["prior"],
        det_path=checkpoints["det"],
        marg_path=checkpoints["marg"],
        k_draws=2,
        flow_steps=2,
        seed=1,
        device="cpu",
        full_region=False,
    )

    arms = payload["arms"]
    assert isinstance(arms, dict)
    assert "b0" not in arms
    assert "marg" not in arms
    assert "full_region" not in payload
    _assert_finite_or_none(payload)


# --------------------------------------------------------------------------- markdown tables


def test_render_tables_markdown_contains_every_arm_name(tmp_path: Path) -> None:
    ctx = _build_context(tmp_path, with_b0=True)
    checkpoints = _write_checkpoints(tmp_path)
    payload = s2eval.run_s2_eval(
        ctx=ctx,
        ae_path=checkpoints["ae"],
        prior_path=checkpoints["prior"],
        det_path=checkpoints["det"],
        marg_path=checkpoints["marg"],
        k_draws=2,
        flow_steps=2,
        seed=2,
        device="cpu",
        full_region=True,
    )

    markdown = s2eval.render_tables_markdown(payload)

    assert isinstance(markdown, str)
    for arm_name in ("gen", "unc", "shuf", "det", "ae", "b0", "marg"):
        assert arm_name in markdown
