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
    degree_histogram,
    evaluate_assembled_graph,
    laplacian_spectrum_histogram,
    mmd_squared,
    noise_floor,
)


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

    def test_spectral_uses_two_hundred_raw_count_bins(self) -> None:
        graph = nx.path_graph(5)
        hist = laplacian_spectrum_histogram(graph)
        assert hist.shape == (200,)
        assert hist.sum() == pytest.approx(graph.number_of_nodes())


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
        ref_mean = float(
            np.mean([report.per_size_reference_mmd2[size][stat] for size in buckets])
        )
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

    def test_self_loops_reported_separately_and_stripped_from_topology(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        g_pred = g_ref.copy()
        first_node = next(iter(g_pred.nodes()))
        g_pred.add_edge(first_node, first_node)
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())
        assert report.self_loops_pred == 1
        assert report.self_loops_ref == 0
        assert report.relative_density == pytest.approx(1.0)
        for stat in STATISTICS:
            assert report.raw_mmd2[stat] == pytest.approx(0.0, abs=1e-9)

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


@pytest.mark.unit
class TestBootstrapMmd:
    def test_shape_and_identical_graph_ratio(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        result = bootstrap_mmd(g_ref.copy(), g_ref, buckets, MMDConfig(), seed=11, n_boot=20)
        assert set(result) == set(buckets)
        for size in buckets:
            for stat in STATISTICS:
                mean, std = result[size][stat]
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert mean == pytest.approx(0.0, abs=1e-9)
                assert std == pytest.approx(0.0, abs=1e-9)


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
