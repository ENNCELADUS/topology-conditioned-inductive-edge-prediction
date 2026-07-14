"""Tests for src.eval.assembly: graph assembly and threshold utilities."""

import networkx as nx
import numpy as np
import pytest
from src.eval.assembly import SweepPoint, assemble_graph, density_matched_threshold, threshold_sweep
from src.eval.graph_metrics import STATISTICS, MMDConfig


@pytest.mark.unit
class TestAssembleGraph:
    def test_edges_included_above_threshold_only(self) -> None:
        pairs = [("a", "b"), ("b", "c")]
        probs = np.array([0.9, 0.3])
        g = assemble_graph(pairs, probs, threshold=0.5, nodes=["a", "b", "c", "d"])
        assert g.has_edge("a", "b")
        assert not g.has_edge("b", "c")

    def test_all_nodes_present_including_isolated(self) -> None:
        pairs = [("a", "b")]
        probs = np.array([0.9])
        g = assemble_graph(pairs, probs, threshold=0.5, nodes=["a", "b", "c"])
        assert set(g.nodes()) == {"a", "b", "c"}
        assert g.degree("c") == 0

    def test_self_pair_becomes_self_loop(self) -> None:
        pairs = [("a", "a")]
        probs = np.array([0.9])
        g = assemble_graph(pairs, probs, threshold=0.5, nodes=["a"])
        assert g.has_edge("a", "a")
        assert nx.number_of_selfloops(g) == 1

    def test_threshold_boundary_is_inclusive(self) -> None:
        pairs = [("a", "b")]
        probs = np.array([0.5])
        g = assemble_graph(pairs, probs, threshold=0.5, nodes=["a", "b"])
        assert g.has_edge("a", "b")

    def test_below_threshold_excluded(self) -> None:
        pairs = [("a", "b")]
        probs = np.array([0.4999])
        g = assemble_graph(pairs, probs, threshold=0.5, nodes=["a", "b"])
        assert not g.has_edge("a", "b")


@pytest.mark.unit
class TestDensityMatchedThreshold:
    def test_exact_no_ties(self) -> None:
        probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        t = density_matched_threshold(probs, target_edges=2)
        assert t == pytest.approx(0.7)
        assert int((probs >= t).sum()) == 2

    def test_target_at_or_above_n_includes_all(self) -> None:
        probs = np.array([0.9, 0.5, 0.1])
        t = density_matched_threshold(probs, target_edges=5)
        assert t == pytest.approx(0.1)
        assert int((probs >= t).sum()) == 3

    def test_target_zero_or_negative_excludes_all(self) -> None:
        probs = np.array([0.9, 0.5, 0.1])
        t = density_matched_threshold(probs, target_edges=0)
        assert int((probs >= t).sum()) == 0
        assert t > probs.max()

    def test_tie_group_exceeding_target_falls_back_above_max(self) -> None:
        # Top tie group (three 0.9's) alone exceeds target=2, so no data-derived
        # threshold keeps count <= target; falls back to just above max.
        probs = np.array([0.9, 0.9, 0.9, 0.5, 0.1])
        t = density_matched_threshold(probs, target_edges=2)
        assert int((probs >= t).sum()) == 0
        assert t > probs.max()

    def test_tie_group_exactly_matching_target(self) -> None:
        probs = np.array([0.9, 0.5, 0.5, 0.1])
        t = density_matched_threshold(probs, target_edges=3)
        assert t == pytest.approx(0.5)
        assert int((probs >= t).sum()) == 3


def _seeded_graph_pairs_probs_buckets() -> tuple[
    nx.Graph, list[tuple[str, str]], np.ndarray, dict[int, list[set[str]]]
]:
    g = nx.erdos_renyi_graph(30, 0.2, seed=5)
    g = nx.relabel_nodes(g, {i: f"n{i:02d}" for i in g.nodes()})
    nodes = list(g.nodes())
    rng = np.random.default_rng(5)
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    probs = rng.uniform(0.0, 1.0, size=len(pairs))
    buckets = {
        10: [set(rng.choice(nodes, size=10, replace=False).tolist()) for _ in range(4)],
    }
    return g, pairs, probs, buckets


@pytest.mark.unit
class TestThresholdSweep:
    def test_returns_sweep_point_per_threshold(self) -> None:
        g_ref, pairs, probs, buckets = _seeded_graph_pairs_probs_buckets()
        config = MMDConfig()
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        points = threshold_sweep(
            pairs, probs, thresholds=thresholds, g_ref=g_ref, buckets=buckets, config=config
        )
        assert len(points) == len(thresholds)
        for point in points:
            assert isinstance(point, SweepPoint)
            assert set(point.mmd_ratio) == set(STATISTICS)
            assert 0.0 <= point.graph_similarity <= 1.0

    def test_relative_density_non_increasing_in_threshold(self) -> None:
        g_ref, pairs, probs, buckets = _seeded_graph_pairs_probs_buckets()
        config = MMDConfig()
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        points = threshold_sweep(
            pairs, probs, thresholds=thresholds, g_ref=g_ref, buckets=buckets, config=config
        )
        densities = [p.relative_density for p in points]
        assert all(densities[i] >= densities[i + 1] - 1e-9 for i in range(len(densities) - 1))

    def test_recall_non_increasing_in_threshold(self) -> None:
        g_ref, pairs, probs, buckets = _seeded_graph_pairs_probs_buckets()
        config = MMDConfig()
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        points = threshold_sweep(
            pairs, probs, thresholds=thresholds, g_ref=g_ref, buckets=buckets, config=config
        )
        recalls = [p.recall for p in points]
        assert all(0.0 <= r <= 1.0 for r in recalls)
        assert all(recalls[i] >= recalls[i + 1] - 1e-9 for i in range(len(recalls) - 1))
