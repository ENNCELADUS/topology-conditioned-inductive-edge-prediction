"""Tests for src.eval.graph_metrics: descriptors, MMD, bucketed graph evaluation."""

import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from src.eval.graph_metrics import (
    STATISTICS,
    MMDConfig,
    bootstrap_mmd,
    clustering_histogram,
    compute_graph_similarity,
    compute_relative_density,
    degree_histogram,
    evaluate_assembled_graph,
    laplacian_spectrum_histogram,
    mmd_squared,
    noise_floor,
)


def _manual_mmd(samples1: list[np.ndarray], samples2: list[np.ndarray]) -> float:
    """Independent Gaussian-TV biased V-stat calculation for formula tests."""
    normalized1 = [sample / (float(sample.sum()) + 1e-6) for sample in samples1]
    normalized2 = [sample / (float(sample.sum()) + 1e-6) for sample in samples2]

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        support = max(len(x), len(y))
        x_padded = np.pad(x, (0, support - len(x)))
        y_padded = np.pad(y, (0, support - len(y)))
        tv = float(np.abs(x_padded - y_padded).sum() / 2.0)
        return float(np.exp(-(tv**2) / 2.0))

    def mean_kernel(left: list[np.ndarray], right: list[np.ndarray]) -> float:
        return float(sum(kernel(x, y) for x in left for y in right) / (len(left) * len(right)))

    return (
        mean_kernel(normalized1, normalized1)
        + mean_kernel(normalized2, normalized2)
        - 2.0 * mean_kernel(normalized1, normalized2)
    )


@pytest.mark.unit
class TestOfficialPringGraphMetrics:
    def test_graph_similarity_is_edge_dice(self) -> None:
        gt = nx.Graph([("a", "b"), ("b", "c")])
        pred = nx.Graph([("a", "b"), ("a", "c")])
        assert compute_graph_similarity(pred, gt) == pytest.approx(0.5)

    def test_graph_similarity_empty_graphs_are_identical(self) -> None:
        gt = nx.empty_graph(["a", "b"])
        pred = gt.copy()
        assert compute_graph_similarity(pred, gt) == pytest.approx(1.0)

    def test_relative_density_matches_official_formula_and_empty_guards(self) -> None:
        gt = nx.Graph([("a", "b"), ("b", "c")])
        pred = nx.Graph([("a", "b")])
        pred.add_node("c")
        assert compute_relative_density(pred, gt) == pytest.approx(0.5)

        empty = nx.empty_graph(["a", "b", "c"])
        assert compute_relative_density(empty, empty) == pytest.approx(1.0)
        assert compute_relative_density(gt, empty) == float("inf")

    def test_requires_matching_node_sets(self) -> None:
        with pytest.raises(ValueError, match="same set of nodes"):
            compute_graph_similarity(nx.empty_graph(["a"]), nx.empty_graph(["b"]))
        with pytest.raises(ValueError, match="same set of nodes"):
            compute_relative_density(nx.empty_graph(["a"]), nx.empty_graph(["b"]))


@pytest.mark.unit
class TestOfficialMmdSquared:
    def test_singleton_total_variation_formula(self) -> None:
        a = [np.array([1.0, 0.0])]
        b = [np.array([0.0, 1.0])]
        normalized_tv = 1.0 / (1.0 + 1e-6)
        expected = 2.0 - 2.0 * np.exp(-(normalized_tv**2) / 2.0)
        assert mmd_squared(a, b, MMDConfig()) == pytest.approx(expected)

    def test_identical_sets_are_zero(self) -> None:
        samples = [np.array([3.0, 1.0]), np.array([1.0, 3.0])]
        assert mmd_squared(samples, samples, MMDConfig()) == pytest.approx(0.0, abs=1e-12)

    def test_histograms_are_normalized_inside_mmd(self) -> None:
        a = [np.array([3.0, 1.0]), np.array([1.0, 3.0])]
        b = [np.array([6.0, 2.0]), np.array([2.0, 6.0])]
        assert mmd_squared(a, b, MMDConfig()) == pytest.approx(0.0, abs=1e-12)

    def test_asymmetric_multi_bin_biased_v_stat_matches_manual_formula(self) -> None:
        samples1 = [np.array([4.0, 1.0, 0.0]), np.array([0.0, 2.0, 3.0])]
        samples2 = [
            np.array([1.0, 3.0]),
            np.array([0.0, 1.0, 4.0]),
            np.array([2.0, 0.0, 1.0, 2.0]),
        ]
        expected = _manual_mmd(samples1, samples2)
        assert expected > 0.0
        assert mmd_squared(samples1, samples2, MMDConfig()) == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize(
        ("samples1", "samples2"),
        [([], [np.array([1.0])]), ([np.array([1.0])], [])],
    )
    def test_rejects_empty_sample_set(
        self, samples1: list[np.ndarray], samples2: list[np.ndarray]
    ) -> None:
        with pytest.raises(ValueError, match="two non-empty sample sets"):
            mmd_squared(samples1, samples2, MMDConfig())


@pytest.mark.unit
class TestOfficialDescriptors:
    def test_degree_histogram_keeps_full_support(self) -> None:
        hist = degree_histogram(nx.star_graph(70))
        assert hist.shape == (71,)
        assert hist[1] == 70
        assert hist[70] == 1

    def test_clustering_uses_one_hundred_bins(self) -> None:
        hist = clustering_histogram(nx.complete_graph(3))
        assert hist.shape == (100,)
        assert hist[-1] == 3

    def test_spectral_uses_two_hundred_pmf_bins(self) -> None:
        graph = nx.path_graph(5)
        hist = laplacian_spectrum_histogram(graph)
        assert hist.shape == (200,)
        assert hist.sum() == pytest.approx(1.0)

    def test_clustering_bin_geometry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            nx,
            "clustering",
            lambda graph: {0: 0.0, 1: 0.009, 2: 0.01, 3: 0.999, 4: 1.0},
        )
        hist = clustering_histogram(nx.Graph())
        assert hist[0] == 2
        assert hist[1] == 1
        assert hist[99] == 2
        assert hist.sum() == 5

    def test_spectral_bin_geometry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.eval.graph_metrics.eigvalsh",
            lambda matrix: np.array([0.0, 0.005, 0.015, 1.999, 2.0]),
        )
        graph = nx.path_graph(5)
        hist = laplacian_spectrum_histogram(graph)
        assert hist[0] == pytest.approx(2.0 / 5.0)
        assert hist[1] == pytest.approx(1.0 / 5.0)
        assert hist[199] == pytest.approx(2.0 / 5.0)
        assert hist.sum() == pytest.approx(1.0)

    def test_empty_spectral_histogram_is_all_zero(self) -> None:
        hist = laplacian_spectrum_histogram(nx.Graph())
        assert hist.shape == (200,)
        np.testing.assert_array_equal(hist, np.zeros(200))


def _seeded_er_graph_and_buckets(
    seed: int = 42,
) -> tuple[nx.Graph, dict[int, list[set[str]]]]:
    g = nx.erdos_renyi_graph(60, 0.15, seed=seed)
    g = nx.relabel_nodes(g, {i: f"node_{i:03d}" for i in g.nodes()})
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    buckets: dict[int, list[set[str]]] = {}
    for size in (10, 20):
        buckets[size] = [
            set(rng.choice(nodes, size=size, replace=False).tolist()) for _ in range(6)
        ]
    return g, buckets


@pytest.mark.unit
def test_bucket_report_uses_ratio_of_aggregate_means() -> None:
    g_ref, buckets = _seeded_er_graph_and_buckets()
    g_pred = nx.Graph()
    g_pred.add_nodes_from(g_ref.nodes())
    report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())

    for stat in STATISTICS:
        raw_mean = float(np.mean([report.per_size_raw_mmd2[size][stat] for size in buckets]))
        ref_mean = float(np.mean([report.per_size_reference_mmd2[size][stat] for size in buckets]))
        assert report.raw_mmd2[stat] == pytest.approx(raw_mean)
        assert report.reference_mmd2[stat] == pytest.approx(ref_mean)
        assert report.mmd_ratio[stat] == pytest.approx(raw_mean / max(ref_mean, 1e-12))


@pytest.mark.unit
class TestEvaluateAssembledGraph:
    def test_identical_graphs_give_zero_mmd_and_unit_density(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        report = evaluate_assembled_graph(g_ref.copy(), g_ref, buckets, MMDConfig())
        for stat in STATISTICS:
            assert report.raw_mmd2[stat] == pytest.approx(0.0, abs=1e-9)
            assert report.mmd_ratio[stat] == pytest.approx(0.0, abs=1e-9)
            assert report.reference_mmd2[stat] > 0.0
        assert report.relative_density == pytest.approx(1.0)
        assert report.self_loops_pred == 0
        assert report.self_loops_ref == 0

    def test_self_loops_participate_in_official_graph_similarity_and_density(self) -> None:
        g_ref = nx.empty_graph(["a", "b", "c"])
        g_ref.add_edge("a", "a")
        g_pred = nx.empty_graph(["a", "b", "c"])
        buckets = {3: [{"a", "b", "c"}, {"a", "b", "c"}]}
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())
        assert report.self_loops_pred == 0
        assert report.self_loops_ref == 1
        assert report.graph_similarity == pytest.approx(0.0)
        assert report.relative_density == pytest.approx(0.0)
        assert report.raw_mmd2["degree"] > 0.0

    def test_gs_rd_are_macro_means_over_every_sample(self) -> None:
        g_ref = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
        g_pred = nx.Graph([("a", "b")])
        g_pred.add_nodes_from(g_ref.nodes())
        buckets = {
            2: [{"a", "b"}, {"c", "d"}],
            3: [{"a", "b", "c"}, {"b", "c", "d"}, {"a", "c", "d"}],
        }
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())
        expected_gs = [1.0, 0.0, 2.0 / 3.0, 0.0, 0.0]
        expected_rd = [1.0, 0.0, 0.5, 0.0, 0.0]
        assert report.per_size_graph_similarity[2] == pytest.approx(expected_gs[:2])
        assert report.per_size_graph_similarity[3] == pytest.approx(expected_gs[2:])
        assert report.per_size_relative_density[2] == pytest.approx(expected_rd[:2])
        assert report.per_size_relative_density[3] == pytest.approx(expected_rd[2:])
        assert report.graph_similarity == pytest.approx(np.mean(expected_gs))
        assert report.relative_density == pytest.approx(np.mean(expected_rd))

    def test_per_size_keys_match_buckets(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        report = evaluate_assembled_graph(g_ref.copy(), g_ref, buckets, MMDConfig())
        assert set(report.per_size_raw_mmd2) == set(buckets)
        assert set(report.per_size_reference_mmd2) == set(buckets)
        for size in buckets:
            assert set(report.per_size_raw_mmd2[size]) == set(STATISTICS)
            assert set(report.per_size_reference_mmd2[size]) == set(STATISTICS)

    def test_bucket_requires_two_reference_samples(self) -> None:
        graph = nx.path_graph(["a", "b", "c"])
        with pytest.raises(ValueError, match="requires at least two reference samples"):
            evaluate_assembled_graph(graph, graph, {3: [{"a", "b", "c"}]}, MMDConfig())


@pytest.mark.unit
class TestNoiseFloor:
    def test_matches_evaluator_reference_denominator(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        config = MMDConfig()
        floor = noise_floor(g_ref, buckets, config)
        report = evaluate_assembled_graph(g_ref.copy(), g_ref, buckets, config)
        assert floor == report.per_size_reference_mmd2

    def test_uses_order_sensitive_even_odd_split(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])
        node_sets = [
            {"a", "b"},
            {"a", "b", "c"},
            {"a", "b", "c", "d"},
            {"a", "c", "d"},
        ]
        histograms = [degree_histogram(graph.subgraph(nodes)) for nodes in node_sets]
        expected_even_odd = _manual_mmd(histograms[::2], histograms[1::2])
        contiguous_halves = _manual_mmd(histograms[:2], histograms[2:])
        floor = noise_floor(graph, {2: node_sets}, MMDConfig())
        assert expected_even_odd != pytest.approx(contiguous_halves)
        assert floor[2]["degree"] == pytest.approx(expected_even_odd, abs=1e-15)

    @pytest.mark.parametrize("node_sets", [[], [{"a"}]])
    def test_rejects_fewer_than_two_samples(self, node_sets: list[set[str]]) -> None:
        graph = nx.Graph()
        graph.add_node("a")
        with pytest.raises(ValueError, match="requires at least two reference samples"):
            noise_floor(graph, {1: node_sets}, MMDConfig())


@pytest.mark.unit
class TestBootstrapMmd:
    def test_shape_and_identical_graph_ratio(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        result = bootstrap_mmd(g_ref.copy(), g_ref, buckets, MMDConfig(), seed=11, n_boot=20)
        assert set(result) == set(STATISTICS)
        for stat in STATISTICS:
            mean, std = result[stat]
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert mean == pytest.approx(0.0, abs=1e-9)
            assert std == pytest.approx(0.0, abs=1e-9)

    def test_replays_ratio_of_size_means_with_fixed_rng(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets(seed=17)
        g_pred = g_ref.copy()
        g_pred.remove_edges_from(list(g_pred.edges())[::3])
        seed = 29
        n_boot = 7
        result = bootstrap_mmd(g_pred, g_ref, buckets, MMDConfig(), seed=seed, n_boot=n_boot)

        histograms = {
            size: (
                [degree_histogram(g_pred.subgraph(sample)) for sample in node_sets],
                [degree_histogram(g_ref.subgraph(sample)) for sample in node_sets],
            )
            for size, node_sets in buckets.items()
        }
        rng = np.random.default_rng(seed)
        ratios: list[float] = []
        per_size_denominators: dict[int, list[float]] = {size: [] for size in buckets}
        for _ in range(n_boot):
            raw_by_size: list[float] = []
            reference_by_size: list[float] = []
            for size, node_sets in buckets.items():
                indices = rng.integers(0, len(node_sets), size=len(node_sets))
                pred_histograms, ref_histograms = histograms[size]
                pred_samples = [pred_histograms[i] for i in indices]
                ref_samples = [ref_histograms[i] for i in indices]
                raw_by_size.append(_manual_mmd(pred_samples, ref_samples))
                denominator = _manual_mmd(ref_samples[::2], ref_samples[1::2])
                reference_by_size.append(denominator)
                per_size_denominators[size].append(denominator)
            ratios.append(
                float(np.mean(raw_by_size)) / max(float(np.mean(reference_by_size)), 1e-12)
            )

        denominator_means = [float(np.mean(values)) for values in per_size_denominators.values()]
        assert all(value > 0.0 for value in denominator_means)
        assert denominator_means[0] != pytest.approx(denominator_means[1])
        assert result["degree"] == pytest.approx(
            (float(np.mean(ratios)), float(np.std(ratios))), abs=1e-15
        )

    @pytest.mark.parametrize("node_sets", [[], [{"a"}]])
    def test_rejects_fewer_than_two_samples(self, node_sets: list[set[str]]) -> None:
        graph = nx.Graph()
        graph.add_node("a")
        with pytest.raises(ValueError, match="requires at least two reference samples"):
            bootstrap_mmd(graph, graph, {1: node_sets}, MMDConfig(), seed=0, n_boot=1)


_HASH_SEED_SNIPPET = """
import numpy as np
import networkx as nx
from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph

g_ref = nx.watts_strogatz_graph(120, k=6, p=0.0, seed=7)
g_ref = nx.relabel_nodes(g_ref, {i: f"n{i:04d}" for i in g_ref.nodes()})
g_pred = g_ref.copy()
g_pred.remove_edges_from(list(g_pred.edges())[::7])
rng = np.random.default_rng(7)
nodes = list(g_ref.nodes())
buckets = {30: [set(rng.choice(nodes, size=30, replace=False).tolist()) for _ in range(6)]}
report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())
for stat in ("degree", "clustering", "spectral"):
    print(f"{stat}={report.raw_mmd2[stat]:.17g}")
"""


@pytest.mark.unit
class TestHashSeedDeterminism:
    def test_evaluate_assembled_graph_identical_across_hash_seeds(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        outputs = []
        for hash_seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(repo_root))
            proc = subprocess.run(  # noqa: S603 - fixed interpreter + literal snippet
                [sys.executable, "-c", _HASH_SEED_SNIPPET],
                capture_output=True,
                text=True,
                env=env,
                cwd=repo_root,
                check=True,
                timeout=120,
            )
            outputs.append(proc.stdout)
        assert outputs[0] == outputs[1] == outputs[2], (
            f"aggregate MMD depends on PYTHONHASHSEED:\n"
            f"seed 1 -> {outputs[0]}seed 2 -> {outputs[1]}seed 3 -> {outputs[2]}"
        )
