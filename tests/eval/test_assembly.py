"""Tests for src.eval.assembly: graph assembly and threshold utilities."""

import networkx as nx
import numpy as np
import pytest
from src.eval.assembly import (
    SweepPoint,
    assemble_degree_quota,
    assemble_graph,
    density_matched_threshold,
    estimated_density_matched_threshold,
    rank_matched_degree_quotas,
    threshold_sweep,
)
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


@pytest.mark.unit
class TestEstimatedDensityMatchedThreshold:
    def test_exact_equivalence_when_sample_covers_full_complement(self) -> None:
        rng = np.random.default_rng(11)
        for target in (0, 1, 3, 7, 15, 25, 1000):
            probs = rng.uniform(0.0, 1.0, size=20)
            exact, sample = probs[:8], probs[8:]
            expected = density_matched_threshold(probs, target_edges=target)
            actual = estimated_density_matched_threshold(
                exact, sample, sampled_total=len(sample), target_edges=target
            )
            assert actual == pytest.approx(expected)

    def test_exact_equivalence_tie_heavy(self) -> None:
        probs = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.9])
        exact, sample = probs[:3], probs[3:]
        for target in (0, 1, 2, 3, 4, 5, 6, 100):
            expected = density_matched_threshold(probs, target_edges=target)
            actual = estimated_density_matched_threshold(
                exact, sample, sampled_total=len(sample), target_edges=target
            )
            assert actual == pytest.approx(expected)

    def test_tie_atomic_across_arrays(self) -> None:
        # Value 0.5 appears in both arrays; combined weighted count for 0.5's
        # tie-group is 1 (exact) + 2 (sampled, weight sampled_total/len(sample)
        # = 2/1 = 2) = 3, on top of 0.9's weight-1 group -> total weight 4.
        exact = np.array([0.9, 0.5])
        sample = np.array([0.5])
        # target=4 admits the full weighted population -> the 0.5 group included.
        t = estimated_density_matched_threshold(exact, sample, sampled_total=2, target_edges=4)
        assert t == pytest.approx(0.5)
        # target=3 cannot fit the 0.5 group's weight-3 tie (cumulative would be
        # 4) -> only the top group (weight 1) admits.
        t2 = estimated_density_matched_threshold(exact, sample, sampled_total=2, target_edges=3)
        assert t2 == pytest.approx(0.9)

    def test_target_zero_or_negative_excludes_all(self) -> None:
        exact = np.array([0.9, 0.5, 0.1])
        sample = np.array([0.7, 0.3])
        overall_max = 0.9
        for target in (0, -5):
            t = estimated_density_matched_threshold(
                exact, sample, sampled_total=4, target_edges=target
            )
            assert t > overall_max
            assert t == pytest.approx(float(np.nextafter(overall_max, np.inf)))

    def test_target_very_large_includes_all(self) -> None:
        exact = np.array([0.9, 0.5, 0.1])
        sample = np.array([0.7, 0.3])
        t = estimated_density_matched_threshold(exact, sample, sampled_total=4, target_edges=10_000)
        assert t == pytest.approx(0.1)

    def test_sampled_total_zero_delegates_to_density_matched_threshold(self) -> None:
        exact = np.array([0.9, 0.5, 0.1, 0.3])
        expected = density_matched_threshold(exact, target_edges=2)
        actual = estimated_density_matched_threshold(
            exact, np.array([]), sampled_total=0, target_edges=2
        )
        assert actual == pytest.approx(expected)

    def test_empty_exact_probs_raises(self) -> None:
        with pytest.raises(ValueError, match="exact_probs"):
            estimated_density_matched_threshold(
                np.array([]), np.array([0.5]), sampled_total=1, target_edges=1
            )

    def test_negative_sampled_total_raises(self) -> None:
        with pytest.raises(ValueError, match="sampled_total"):
            estimated_density_matched_threshold(
                np.array([0.5]), np.array([]), sampled_total=-1, target_edges=1
            )

    def test_positive_sampled_total_with_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="sampled_probs"):
            estimated_density_matched_threshold(
                np.array([0.5]), np.array([]), sampled_total=5, target_edges=1
            )

    def test_sample_larger_than_sampled_total_raises(self) -> None:
        with pytest.raises(ValueError, match="sampled_total"):
            estimated_density_matched_threshold(
                np.array([0.5]), np.array([0.1, 0.2, 0.3]), sampled_total=2, target_edges=1
            )

    def test_statistical_sanity_large_population(self) -> None:
        rng = np.random.default_rng(42)
        population = rng.uniform(0.0, 1.0, size=200_000)
        exact_size = 20_000
        exact = population[:exact_size]
        rest = population[exact_size:]
        sample_size = int(0.1 * len(rest))
        sample = rng.choice(rest, size=sample_size, replace=False)

        target = 9_000
        t = estimated_density_matched_threshold(
            exact, sample, sampled_total=len(rest), target_edges=target
        )
        realized = int((population >= t).sum())
        assert abs(realized - target) / target < 0.05


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


@pytest.mark.unit
class TestRankMatchedDegreeQuotas:
    def test_rank_assignment(self) -> None:
        expected = {"a": 3.0, "b": 2.0, "c": 0.5}
        quotas = rank_matched_degree_quotas(expected, [2, 1, 2])
        assert quotas == {"a": 2, "b": 2, "c": 1}

    def test_expected_degree_ties_break_by_node_id(self) -> None:
        expected = {"b": 1.0, "a": 1.0}
        quotas = rank_matched_degree_quotas(expected, [3, 1])
        assert quotas == {"a": 3, "b": 1}

    def test_multiset_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="multiset"):
            rank_matched_degree_quotas({"a": 1.0}, [1, 2])

    def test_quota_sum_equals_reference_degree_sum(self) -> None:
        rng = np.random.default_rng(0)
        expected = {f"n{i}": float(rng.uniform(0, 5)) for i in range(10)}
        reference = [int(d) for d in rng.integers(0, 4, size=10)]
        quotas = rank_matched_degree_quotas(expected, reference)
        assert sum(quotas.values()) == sum(reference)


@pytest.mark.unit
class TestAssembleDegreeQuota:
    def _rows(
        self, pairs: list[tuple[str, str]], nodes: list[str], scores: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = {node: i for i, node in enumerate(nodes)}
        u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
        v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
        return u_idx, v_idx, np.array(scores, dtype=np.float64)

    def test_greedy_respects_quotas(self) -> None:
        nodes = ["a", "b", "c", "d"]
        pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")]
        u_idx, v_idx, scores = self._rows(pairs, nodes, [0.9, 0.8, 0.7, 0.6])
        quotas = {"a": 2, "b": 1, "c": 1, "d": 0}
        g, stats = assemble_degree_quota(pairs, u_idx, v_idx, scores, quotas, nodes)
        # (a,b) accepted; (a,c) accepted; (a,d) blocked (a exhausted + d zero);
        # (b,c) blocked (both exhausted).
        assert set(map(frozenset, g.edges())) == {frozenset({"a", "b"}), frozenset({"a", "c"})}
        assert stats.realized_edges == 2
        assert stats.target_edges == 2
        assert stats.shortfall == 0
        assert stats.residual_quota == 0

    def test_shortfall_disclosed_without_backfill(self) -> None:
        nodes = ["a", "b", "c"]
        pairs = [("a", "b")]
        u_idx, v_idx, scores = self._rows(pairs, nodes, [0.9])
        quotas = {"a": 1, "b": 1, "c": 2}  # c's quota can never be consumed
        g, stats = assemble_degree_quota(pairs, u_idx, v_idx, scores, quotas, nodes)
        assert g.number_of_edges() == 1
        assert stats.target_edges == 2
        assert stats.shortfall == 1
        assert stats.residual_quota == 2

    def test_score_ties_break_by_canonical_pair_order(self) -> None:
        nodes = ["a", "b", "c"]
        pairs = [("b", "c"), ("a", "b")]
        u_idx, v_idx, scores = self._rows(pairs, nodes, [0.5, 0.5])
        quotas = {"a": 1, "b": 1, "c": 1}
        g, _ = assemble_degree_quota(pairs, u_idx, v_idx, scores, quotas, nodes)
        # Tie at 0.5: canonical order prefers (a, b); b then exhausted for (b, c).
        assert set(map(frozenset, g.edges())) == {frozenset({"a", "b"})}

    def test_all_nodes_present_including_isolated(self) -> None:
        nodes = ["a", "b", "z"]
        pairs = [("a", "b")]
        u_idx, v_idx, scores = self._rows(pairs, nodes, [0.9])
        g, _ = assemble_degree_quota(pairs, u_idx, v_idx, scores, {"a": 1, "b": 1, "z": 0}, nodes)
        assert set(g.nodes()) == {"a", "b", "z"}

    def test_self_pair_rows_rejected(self) -> None:
        nodes = ["a", "b"]
        pairs = [("a", "a")]
        u_idx, v_idx, scores = self._rows(pairs, nodes, [0.9])
        with pytest.raises(ValueError, match="non-self"):
            assemble_degree_quota(pairs, u_idx, v_idx, scores, {"a": 1, "b": 1}, nodes)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        nodes = [f"n{i}" for i in range(8)]
        pairs = [(nodes[i], nodes[j]) for i in range(8) for j in range(i + 1, 8)]
        u_idx, v_idx, _ = self._rows(pairs, nodes, [0.0] * len(pairs))
        scores = rng.uniform(size=len(pairs))
        quotas = {node: int(q) for node, q in zip(nodes, rng.integers(0, 4, size=8), strict=True)}
        g1, s1 = assemble_degree_quota(pairs, u_idx, v_idx, scores, quotas, nodes)
        g2, s2 = assemble_degree_quota(pairs, u_idx, v_idx, scores.copy(), dict(quotas), nodes)
        assert set(map(frozenset, g1.edges())) == set(map(frozenset, g2.edges()))
        assert s1 == s2
