"""Tests for src.experiments.g1_hardened_e2: the G1 hardened-E2 analysis pipeline."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.eval.composite import PerturbationCheckResult
from src.experiments import g1_hardened_e2 as g1
from src.score_universe import load_scores, save_scores

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- fixtures / builders

# Hand-built 8-node reference graph with known common-neighbor / degree structure.
#   edges: (n1,n2) (n1,n3) (n2,n3) (n4,n5) (n1,n4)
#   degrees: n1=3 n2=2 n3=2 n4=2 n5=1 n6=0 n7=0 n8=0
_NODES = [f"n{i}" for i in range(1, 9)]
_POSITIVE_EDGES = [("n1", "n2"), ("n1", "n3"), ("n2", "n3"), ("n4", "n5"), ("n1", "n4")]

# Feature vectors chosen so cosine similarity is hand-verifiable: n4 and n6 are
# exactly parallel (cos = 1, tying with every self-pair); n5 is close to but not
# parallel with n4/n6; n7 is anti-parallel to n1.
_FEATURE_VECTORS: dict[str, tuple[float, float]] = {
    "n1": (1.0, 0.0),
    "n2": (0.9, 0.1),
    "n3": (0.8, 0.2),
    "n4": (0.0, 1.0),
    "n5": (0.1, 0.9),
    "n6": (0.0, 1.0),
    "n7": (-1.0, 0.0),
    "n8": (0.5, 0.5),
}


def _make_reference_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    g.add_edges_from(_POSITIVE_EDGES)
    return g


def _canonical(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _universe_rows(
    node_ids: list[str], positive_edges: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Build the full candidate universe (all canonical pairs + self-pairs) for `node_ids`."""
    positive_set = {_canonical(u, v) for u, v in positive_edges}
    pairs: list[tuple[str, str]] = []
    for i, u in enumerate(node_ids):
        for v in node_ids[i:]:
            pairs.append((u, v))
    labels = np.array([1 if pair in positive_set else 0 for pair in pairs], dtype=np.int8)
    return pairs, labels


def _write_universe_npz(
    path: Path,
    *,
    node_ids: list[str],
    pairs: list[tuple[str, str]],
    logits: np.ndarray,
    labels: np.ndarray,
    strategy: str = "toy",
    pairs_source: str = "candidate",
    model_family: str = "v3_1",
    checkpoint_id: str = "deadbeefcafefeed",
) -> None:
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    meta = {
        "checkpoint_id": checkpoint_id,
        "model_family": model_family,
        "pairs_source": pairs_source,
        "strategy": strategy,
        "num_rows": len(pairs),
        "created_utc": "2026-07-09T00:00:00+00:00",
        "torch_version": "2.10.0",
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logits.astype(np.float32),
        label=labels.astype(np.int8),
        row_start=0,
        meta=meta,
    )


def _logit_for_prob(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


def _reference_universe_path(tmp_path: Path, *, strategy: str = "toy") -> Path:
    """Write the full 36-row candidate universe for the 8-node reference graph."""
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=len(pairs)).astype(np.float32)
    path = tmp_path / "universe.npz"
    _write_universe_npz(
        path, node_ids=_NODES, pairs=pairs, logits=logits, labels=labels, strategy=strategy
    )
    return path


def _write_benchmark(
    tmp_path: Path, strategy: str, g: nx.Graph, buckets: dict[int, list[set[str]]]
) -> Path:
    data_root = tmp_path / "data"
    strategy_dir = data_root / "benchmark_2025_neurips" / strategy
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(g, f)
    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump(buckets, f)
    return data_root


def _small_buckets(
    nodes: list[str], *, size: int = 5, n_samples: int = 4, seed: int = 0
) -> dict[int, list[set[str]]]:
    rng = np.random.default_rng(seed)
    return {
        size: [
            set(rng.choice(nodes, size=min(size, len(nodes)), replace=False).tolist())
            for _ in range(n_samples)
        ]
    }


def _write_feature_root(tmp_path: Path) -> Path:
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id, vec in _FEATURE_VECTORS.items():
        tensor = torch.tensor([list(vec)], dtype=torch.float32)  # (L=1, dim=2)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps({"format": "torch_pt_per_node", "input_dim": 2, "max_sequence_length": 8})
    )
    return root


_SMALL_CONFIG_KWARGS = {"degree_max": 8, "clustering_bins": 8, "spectral_bins": 8}


def _d(x: object) -> dict[str, Any]:
    """Cast a JSON-payload value known to be a dict, for concise test assertions."""
    return cast(dict[str, Any], x)


# --------------------------------------------------------------------------- validation


class TestValidateUniverseArtifact:
    def test_accepts_valid_universe(self, tmp_path: Path) -> None:
        path = _reference_universe_path(tmp_path)
        artifact = load_scores(path)
        g1.validate_universe_artifact(artifact, strategy="toy", n_test_nodes=8)

    def test_rejects_wrong_pairs_source(self, tmp_path: Path) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        path = tmp_path / "universe.npz"
        _write_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.zeros(len(pairs)),
            labels=labels,
            strategy="toy",
            pairs_source="test",
        )
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="pairs_source"):
            g1.validate_universe_artifact(artifact, strategy="toy", n_test_nodes=8)

    def test_rejects_wrong_strategy(self, tmp_path: Path) -> None:
        path = _reference_universe_path(tmp_path, strategy="other")
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="strategy"):
            g1.validate_universe_artifact(artifact, strategy="toy", n_test_nodes=8)

    def test_rejects_wrong_row_count(self, tmp_path: Path) -> None:
        path = _reference_universe_path(tmp_path)
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="row count"):
            g1.validate_universe_artifact(artifact, strategy="toy", n_test_nodes=100)

    def test_rejects_bad_labels(self, tmp_path: Path) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        labels = labels.copy()
        labels[0] = -1
        path = tmp_path / "universe.npz"
        _write_universe_npz(
            path, node_ids=_NODES, pairs=pairs, logits=np.zeros(len(pairs)), labels=labels
        )
        artifact = load_scores(path)
        with pytest.raises(ValueError, match="label"):
            g1.validate_universe_artifact(artifact, strategy="toy", n_test_nodes=8)


class TestValidateAltArtifact:
    def test_accepts_matching_alt(self, tmp_path: Path) -> None:
        universe_path = _reference_universe_path(tmp_path)
        universe = load_scores(universe_path)
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        alt_path = tmp_path / "alt.npz"
        rng = np.random.default_rng(1)
        _write_universe_npz(
            alt_path,
            node_ids=_NODES,
            pairs=pairs,
            logits=rng.normal(size=len(pairs)).astype(np.float32),
            labels=labels,
            model_family="f0_mlp",
        )
        alt = load_scores(alt_path)
        g1.validate_alt_artifact(alt, universe)

    def test_rejects_different_row_count(self, tmp_path: Path) -> None:
        universe = load_scores(_reference_universe_path(tmp_path))
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        alt_path = tmp_path / "alt.npz"
        _write_universe_npz(
            alt_path,
            node_ids=_NODES,
            pairs=pairs[:-1],
            logits=np.zeros(len(pairs) - 1),
            labels=labels[:-1],
        )
        alt = load_scores(alt_path)
        with pytest.raises(ValueError, match="rows"):
            g1.validate_alt_artifact(alt, universe)

    def test_rejects_reordered_pairs(self, tmp_path: Path) -> None:
        universe = load_scores(_reference_universe_path(tmp_path))
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        # Swap the first two rows: same set of pairs, different row order.
        reordered_pairs = [pairs[1], pairs[0], *pairs[2:]]
        reordered_labels = np.array([labels[1], labels[0], *labels[2:]], dtype=np.int8)
        alt_path = tmp_path / "alt.npz"
        _write_universe_npz(
            alt_path,
            node_ids=_NODES,
            pairs=reordered_pairs,
            logits=np.zeros(len(reordered_pairs)),
            labels=reordered_labels,
        )
        alt = load_scores(alt_path)
        with pytest.raises(ValueError, match="pair order"):
            g1.validate_alt_artifact(alt, universe)


# --------------------------------------------------------------------------- calibration


class TestCalibrateTau:
    def test_tau_is_mean_noise_floor_over_bucket_sizes(self) -> None:
        nf = {
            20: {"degree": 0.1, "clustering": 0.2, "spectral": 0.05},
            40: {"degree": 0.3, "clustering": 0.0, "spectral": 0.15},
        }
        tau = g1.calibrate_tau(nf)
        assert tau["degree"] == pytest.approx(0.2)
        assert tau["clustering"] == pytest.approx(0.1)
        assert tau["spectral"] == pytest.approx(0.1)

    def test_degenerate_zero_noise_floor_hits_floor(self) -> None:
        nf = {20: {"degree": 0.0, "clustering": 0.0, "spectral": 0.0}}
        tau = g1.calibrate_tau(nf)
        assert tau == {"degree": 1e-6, "clustering": 1e-6, "spectral": 1e-6}


# --------------------------------------------------------------------------- regime selectors


class TestSelectEasyUniform:
    def test_subset_of_negatives_and_correct_size(self) -> None:
        neg_idx = np.arange(10, 20, dtype=np.int64)
        selected = g1.select_easy_uniform(neg_idx, 4, seed=3)
        assert selected.size == 4
        assert set(selected.tolist()) <= set(neg_idx.tolist())

    def test_deterministic_given_seed(self) -> None:
        neg_idx = np.arange(10, 40, dtype=np.int64)
        a = g1.select_easy_uniform(neg_idx, 7, seed=5)
        b = g1.select_easy_uniform(neg_idx, 7, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_size_clamped_to_pool(self) -> None:
        neg_idx = np.array([1, 2, 3], dtype=np.int64)
        selected = g1.select_easy_uniform(neg_idx, 100, seed=0)
        assert set(selected.tolist()) == {1, 2, 3}


class TestWeightedSampleWithoutReplacement:
    def test_extreme_weight_dominates_across_seeds(self) -> None:
        weights = np.array([1.0, 1.0, 1e12])
        for seed in range(20):
            rng = np.random.default_rng(seed)
            picked = g1._weighted_sample_without_replacement(weights, 1, rng)
            assert picked.tolist() == [2]

    def test_n_at_or_above_pool_returns_everything(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        picked = g1._weighted_sample_without_replacement(weights, 5, rng)
        assert set(picked.tolist()) == {0, 1, 2}


class TestSelectDegreeCorrected:
    def test_never_selects_positive_rows_and_respects_size(self) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()
        degree_array = g1._degree_array(g, _NODES)

        selected = g1.select_degree_corrected(neg_idx, u_idx, v_idx, degree_array, 6, seed=7)
        assert selected.size == 6
        assert np.all(labels[selected] == 0)
        assert set(selected.tolist()) <= set(neg_idx.tolist())

    def test_matches_reference_efraimidis_spirakis_computation(self) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()
        degree_array = g1._degree_array(g, _NODES)
        seed, n = 7, 6

        weights = (degree_array[u_idx[neg_idx]] + 1.0) * (degree_array[v_idx[neg_idx]] + 1.0)
        rng = np.random.default_rng(g1._regime_seed_sequence(seed, "degree_corrected", n))
        expected_positions = g1._weighted_sample_without_replacement(weights, n, rng)
        expected = np.sort(neg_idx[expected_positions])

        got = g1.select_degree_corrected(neg_idx, u_idx, v_idx, degree_array, n, seed=seed)
        np.testing.assert_array_equal(got, expected)

    def test_deterministic_given_seed(self) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()
        degree_array = g1._degree_array(g, _NODES)
        a = g1.select_degree_corrected(neg_idx, u_idx, v_idx, degree_array, 5, seed=42)
        b = g1.select_degree_corrected(neg_idx, u_idx, v_idx, degree_array, 5, seed=42)
        np.testing.assert_array_equal(a, b)


class TestCommonNeighborAndAdamicAdar:
    def test_hand_computed_matrices(self) -> None:
        g = _make_reference_graph()
        cn, aa = g1.common_neighbor_and_adamic_adar(g, _NODES)
        # CN(u, u) = deg(u); CN(n2, n4) = 1 via common neighbor n1.
        assert cn[0, 0] == pytest.approx(3.0)  # n1
        assert cn[1, 1] == pytest.approx(2.0)  # n2
        assert cn[4, 4] == pytest.approx(1.0)  # n5
        assert cn[1, 3] == pytest.approx(1.0)  # (n2, n4) via n1
        assert cn[5, 5] == pytest.approx(0.0)  # n6 isolated
        # AA(n2, n2) = 1/log(deg n1=3) + 1/log(deg n3=2)
        expected_aa_n2n2 = 1.0 / math.log(3) + 1.0 / math.log(2)
        assert aa[1, 1] == pytest.approx(expected_aa_n2n2)


class TestSelectHardHeuristic:
    def test_top_4_matches_hand_computation(self) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()

        selected = g1.select_hard_heuristic(neg_idx, u_idx, v_idx, _NODES, g, 4)
        selected_pairs = {pairs[i] for i in selected.tolist()}
        # CN=3: (n1,n1); CN=2 tie broken by AA then pair order: (n2,n2) ties (n3,n3)
        # at AA=1/log3+1/log2, pair order n2<n3; then (n4,n4) at lower AA.
        assert selected_pairs == {("n1", "n1"), ("n2", "n2"), ("n3", "n3"), ("n4", "n4")}

    def test_top_6_discriminates_adamic_adar_from_index_order(self) -> None:
        # Within the CN=1 group {(n1,n5), (n5,n5), (n2,n4), (n3,n4)}, AA(n1,n5) =
        # AA(n5,n5) = 1/log(deg n4=2) > AA(n2,n4) = AA(n3,n4) = 1/log(deg n1=3), so
        # (n5,n5) must rank ahead of (n2,n4)/(n3,n4) even though its u_idx (4) is
        # larger than theirs (1, 2) -- this fails if AA is ignored (pure index order
        # would instead promote (n2,n4) into the top 6).
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()

        selected = g1.select_hard_heuristic(neg_idx, u_idx, v_idx, _NODES, g, 6)
        selected_pairs = {pairs[i] for i in selected.tolist()}
        assert selected_pairs == {
            ("n1", "n1"),
            ("n2", "n2"),
            ("n3", "n3"),
            ("n4", "n4"),
            ("n1", "n5"),
            ("n5", "n5"),
        }

    def test_never_selects_positive_rows(self) -> None:
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        g = _make_reference_graph()
        selected = g1.select_hard_heuristic(neg_idx, u_idx, v_idx, _NODES, g, 25)
        assert np.all(labels[selected] == 0)


class TestSelectHardFeature:
    def test_top_5_matches_hand_computation(self, tmp_path: Path) -> None:
        store = FeatureStore(_write_feature_root(tmp_path))
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)

        selected = g1.select_hard_feature(neg_idx, u_idx, v_idx, _NODES, store, 5)
        selected_pairs = {pairs[i] for i in selected.tolist()}
        # cos=1.0 tie group ordered by (u_idx, v_idx) asc: self-pairs n1..n4, then
        # (n4, n6) (exactly parallel feature vectors).
        assert selected_pairs == {
            ("n1", "n1"),
            ("n2", "n2"),
            ("n3", "n3"),
            ("n4", "n4"),
            ("n4", "n6"),
        }

    def test_never_selects_positive_rows(self, tmp_path: Path) -> None:
        store = FeatureStore(_write_feature_root(tmp_path))
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        neg_idx = np.flatnonzero(labels == 0)
        selected = g1.select_hard_feature(neg_idx, u_idx, v_idx, _NODES, store, 31)
        assert np.all(labels[selected] == 0)


# --------------------------------------------------------------------------- PA-null


class TestComputePaNullScores:
    def test_exact_values_on_reference_graph(self) -> None:
        g = _make_reference_graph()
        pairs, _ = _universe_rows(_NODES, _POSITIVE_EDGES)
        u_idx = np.array([_NODES.index(u) for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([_NODES.index(v) for _, v in pairs], dtype=np.int32)
        scores = g1.compute_pa_null_scores(g, _NODES, u_idx, v_idx)

        expected = {
            ("n1", "n1"): 9.0,
            ("n1", "n2"): 6.0,
            ("n1", "n5"): 3.0,
            ("n6", "n7"): 0.0,
            ("n5", "n5"): 1.0,
        }
        for pair, value in expected.items():
            i = pairs.index(pair)
            assert scores[i] == pytest.approx(value)


class TestNormalizeMinMax:
    def test_normalizes_into_unit_range(self) -> None:
        scores = np.array([0.0, 3.0, 9.0])
        normed = g1.normalize_min_max(scores)
        assert normed[0] == pytest.approx(0.0)
        assert normed[2] == pytest.approx(1.0)
        assert normed[1] == pytest.approx(3.0 / 9.0)

    def test_constant_scores_return_one_half(self) -> None:
        scores = np.array([5.0, 5.0, 5.0])
        normed = g1.normalize_min_max(scores)
        np.testing.assert_allclose(normed, 0.5)


class TestDegreeHeterogeneitySigma:
    def test_matches_hand_computed_value(self) -> None:
        g = _make_reference_graph()
        # degrees >= 1: n1=3, n2=2, n3=2, n4=2, n5=1 (n6,n7,n8 excluded, degree 0)
        expected = float(np.std(np.log([3.0, 2.0, 2.0, 2.0, 1.0])))
        assert g1.degree_heterogeneity_sigma(g) == pytest.approx(expected)

    def test_no_positive_degree_nodes_returns_zero(self) -> None:
        g = nx.Graph()
        g.add_nodes_from(["a", "b"])
        assert g1.degree_heterogeneity_sigma(g) == 0.0


# --------------------------------------------------------------------------- threshold sweep


class TestBuildThresholdGrid:
    def test_includes_operating_point_and_percentiles_deduped_sorted(self) -> None:
        probs = np.linspace(0.0, 1.0, 101)
        grid = g1.build_threshold_grid(probs, 0.5)
        assert grid == sorted(set(grid))
        assert 0.5 in grid


class TestComputeSelfLoopStats:
    def test_counts_and_rate(self) -> None:
        u_idx = np.array([0, 1, 2, 2], dtype=np.int32)
        v_idx = np.array([1, 1, 2, 2], dtype=np.int32)
        probs = np.array([0.9, 0.9, 0.9, 0.9])
        hits, rate = g1.compute_self_loop_stats(u_idx, v_idx, probs, 0.5)
        # self-pairs: (1,1) and (2,2) [appearing twice] -> 3 self-pair rows total, all hit.
        assert hits == 3
        assert rate == pytest.approx(1.0)

    def test_zero_self_pairs_gives_zero_rate(self) -> None:
        u_idx = np.array([0, 1], dtype=np.int32)
        v_idx = np.array([1, 2], dtype=np.int32)
        probs = np.array([0.9, 0.9])
        hits, rate = g1.compute_self_loop_stats(u_idx, v_idx, probs, 0.5)
        assert hits == 0
        assert rate == 0.0


class TestRunThresholdSweep:
    def test_planted_scores_recover_recall_and_density_near_one(self, tmp_path: Path) -> None:
        g = nx.erdos_renyi_graph(12, 0.3, seed=3)
        g = nx.relabel_nodes(g, {i: f"m{i:02d}" for i in g.nodes()})
        g.add_edge("m00", "m00")
        g.add_edge("m01", "m01")
        nodes = sorted(g.nodes())
        edge_set = {frozenset(e) for e in g.edges()}

        pairs: list[tuple[str, str]] = []
        for i, u in enumerate(nodes):
            for v in nodes[i:]:
                pairs.append((u, v))
        probs = np.array([0.9 if frozenset((u, v)) in edge_set else 0.1 for u, v in pairs])
        logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)
        labels = np.zeros(len(pairs), dtype=np.int8)
        path = tmp_path / "universe.npz"
        _write_universe_npz(path, node_ids=nodes, pairs=pairs, logits=logits, labels=labels)
        universe = load_scores(path)

        from src.eval.graph_metrics import MMDConfig

        config = MMDConfig(degree_max=12, clustering_bins=10, spectral_bins=10)
        buckets = _small_buckets(nodes, size=6, n_samples=3, seed=1)

        op_threshold, rows = g1.run_threshold_sweep(
            universe=universe, probs=universe.probs(), g_ref=g, buckets=buckets, config=config
        )
        op_row = next(r for r in rows if r.threshold == pytest.approx(op_threshold))
        assert op_row.recall == pytest.approx(1.0, abs=1e-6)
        assert op_row.relative_density == pytest.approx(1.0, abs=1e-6)
        assert op_row.self_loop_count == 2
        assert op_row.self_loop_rate == pytest.approx(2 / len(nodes))


# --------------------------------------------------------------------------- assembled rows


class TestAssembleTopNByScore:
    def test_selects_top_n_with_deterministic_tie_break(self) -> None:
        pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("a", "a")]
        u_idx = np.array([0, 0, 1, 0], dtype=np.int32)
        v_idx = np.array([1, 2, 2, 0], dtype=np.int32)
        scores = np.array([1.0, 1.0, 0.5, 1.0])  # three-way tie at 1.0
        g = g1.assemble_top_n_by_score(pairs, u_idx, v_idx, scores, 2, ["a", "b", "c"])
        # tie-break by (u_idx, v_idx) ascending among the 1.0 scores: (a,a) u=0,v=0
        # then (a,b) u=0,v=1 beat (a,c) u=0,v=2.
        assert g.has_edge("a", "a")
        assert g.has_edge("a", "b")
        assert not g.has_edge("a", "c")


# --------------------------------------------------------------------------- composite gating


class TestCompositeGating:
    def test_failed_perturbation_check_nulls_composite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        g = _make_reference_graph()
        buckets = _small_buckets(_NODES, size=5, n_samples=3, seed=2)
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)
        universe_path = _reference_universe_path(tmp_path)

        def _fake_perturbation_check(*args: object, **kwargs: object) -> PerturbationCheckResult:
            return PerturbationCheckResult(
                similarities={"degree_preserving_swap": [1.0, 0.5]},
                fractions=(0.0, 0.5),
                passed=False,
                failures=["forced failure for test"],
            )

        monkeypatch.setattr(g1, "perturbation_check", _fake_perturbation_check)

        payload = g1.run_g1_pipeline(
            universe_path=universe_path,
            alt_universe_path=None,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
        )
        assert _d(payload["metadata"])["composite_valid"] is False
        assembled = _d(payload["assembled"])
        assert _d(assembled["b0"])["composite"] is None
        assert _d(assembled["pa_null"])["composite"] is None

    def test_skip_perturbation_check_also_nulls_composite(self, tmp_path: Path) -> None:
        g = _make_reference_graph()
        buckets = _small_buckets(_NODES, size=5, n_samples=3, seed=2)
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)
        universe_path = _reference_universe_path(tmp_path)

        payload = g1.run_g1_pipeline(
            universe_path=universe_path,
            alt_universe_path=None,
            data_root=data_root,
            strategy="toy",
            output_dir=tmp_path / "out",
            seed=0,
            skip_perturbation_check=True,
        )
        metadata = _d(payload["metadata"])
        assert metadata["composite_valid"] is None
        assert _d(metadata["perturbation_check"])["skipped"] is True
        assert _d(_d(payload["assembled"])["b0"])["composite"] is None


# --------------------------------------------------------------------------- markdown tables


class TestRenderTablesMarkdown:
    def test_contains_regime_table_header(self) -> None:
        payload = {
            "regime_table": {"b0": {}, "b0_alt": None, "pa_null": {}},
            "threshold_sweep": {"operating_point": 0.5, "target_edges": 1, "rows": []},
            "assembled": {"b0": None, "b0_alt": None, "pa_null": None},
            "noise_floor": {},
        }
        markdown = g1.render_tables_markdown(payload)
        assert "## Regime table" in markdown
        assert "| scorer | regime | key |" in markdown


# --------------------------------------------------------------------------- benchmark I/O


class TestLoadTestGraphAndBuckets:
    def test_loads_pickled_graph_and_buckets(self, tmp_path: Path) -> None:
        g = _make_reference_graph()
        buckets = _small_buckets(_NODES)
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)
        loaded_graph = g1.load_test_graph(data_root / "benchmark_2025_neurips", "toy")
        loaded_buckets = g1.load_test_node_buckets(data_root / "benchmark_2025_neurips", "toy")
        assert set(loaded_graph.nodes()) == set(g.nodes())
        assert loaded_graph.number_of_edges() == g.number_of_edges()
        assert set(loaded_buckets.keys()) == set(buckets.keys())

    def test_rejects_non_graph_pickle(self, tmp_path: Path) -> None:
        strategy_dir = tmp_path / "toy"
        strategy_dir.mkdir(parents=True)
        with (strategy_dir / "test_graph.pkl").open("wb") as f:
            pickle.dump({"not": "a graph"}, f)
        with pytest.raises(TypeError):
            g1.load_test_graph(tmp_path, "toy")


# --------------------------------------------------------------------------- CLI end-to-end


class TestCliEndToEnd:
    def test_run_twice_byte_identical_and_tables_render(self, tmp_path: Path) -> None:
        g = _make_reference_graph()
        buckets = _small_buckets(_NODES, size=5, n_samples=3, seed=4)
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)
        universe_path = _reference_universe_path(tmp_path)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        common_args = [
            "--universe",
            str(universe_path),
            "--data-root",
            str(data_root),
            "--strategy",
            "toy",
            "--seed",
            "0",
        ]
        g1.main([*common_args, "--output-dir", str(out1)])
        g1.main([*common_args, "--output-dir", str(out2)])

        json1 = (out1 / "g1_results.json").read_bytes()
        json2 = (out2 / "g1_results.json").read_bytes()
        assert json1 == json2

        parsed = json.loads(json1)
        assert "metadata" in parsed
        assert "regime_table" in parsed

        tables = (out1 / "g1_tables.md").read_text()
        assert "## Regime table" in tables

    def test_bad_pairs_source_fails_with_clear_message(self, tmp_path: Path) -> None:
        g = _make_reference_graph()
        buckets = _small_buckets(_NODES, size=5, n_samples=3, seed=5)
        data_root = _write_benchmark(tmp_path, "toy", g, buckets)

        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        bad_path = tmp_path / "bad_universe.npz"
        _write_universe_npz(
            bad_path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.zeros(len(pairs)),
            labels=labels,
            strategy="toy",
            pairs_source="test",
        )

        with pytest.raises(ValueError, match="pairs_source"):
            g1.main(
                [
                    "--universe",
                    str(bad_path),
                    "--data-root",
                    str(data_root),
                    "--strategy",
                    "toy",
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )

    def test_missing_universe_file_raises_system_exit(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            g1.main(
                [
                    "--universe",
                    str(tmp_path / "does_not_exist.npz"),
                    "--data-root",
                    str(tmp_path / "data"),
                    "--strategy",
                    "toy",
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )


# --------------------------------------------------------------------------- real-data integration


@pytest.mark.integration
@pytest.mark.slow
class TestRealBenchmarkNoiseFloorAndCalibration:
    def test_noise_floor_and_tau_positive_on_real_breadth_first(self, benchmark_root: Path) -> None:
        from src.eval.graph_metrics import MMDConfig, noise_floor

        g_ref = g1.load_test_graph(benchmark_root, "breadth_first")
        buckets = g1.load_test_node_buckets(benchmark_root, "breadth_first")
        config = MMDConfig()

        nf = noise_floor(g_ref, buckets, config, seed=0)
        tau = g1.calibrate_tau(nf)

        assert set(tau.keys()) == {"degree", "clustering", "spectral"}
        for value in tau.values():
            assert value > 0.0
