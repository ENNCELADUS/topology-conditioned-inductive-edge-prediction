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
class TestDegreeHistogram:
    def test_triangle_k3(self) -> None:
        g = nx.complete_graph(3)  # all degree 2
        hist = degree_histogram(g, max_degree=5)
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert hist.shape == (6,)
        np.testing.assert_allclose(hist, expected)

    def test_star_graph(self) -> None:
        g = nx.star_graph(4)  # center degree 4, four leaves degree 1
        hist = degree_histogram(g, max_degree=5)
        # 4 nodes at degree 1, 1 node at degree 4, out of 5 total
        expected = np.array([0.0, 0.8, 0.0, 0.0, 0.2, 0.0])
        np.testing.assert_allclose(hist, expected)

    def test_path_graph(self) -> None:
        g = nx.path_graph(4)  # degrees [1,2,2,1]
        hist = degree_histogram(g, max_degree=5)
        expected = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(hist, expected)

    def test_degree_clipped_at_max(self) -> None:
        g = nx.star_graph(10)  # center degree 10
        hist = degree_histogram(g, max_degree=3)
        assert hist.shape == (4,)
        # center's degree 10 clips into bin 3; 10 leaves at degree 1
        expected = np.array([0.0, 10.0 / 11.0, 0.0, 1.0 / 11.0])
        np.testing.assert_allclose(hist, expected)

    def test_empty_graph_returns_zero_vector(self) -> None:
        g = nx.Graph()
        hist = degree_histogram(g, max_degree=5)
        np.testing.assert_allclose(hist, np.zeros(6))


@pytest.mark.unit
class TestClusteringHistogram:
    def test_triangle_all_closed(self) -> None:
        g = nx.complete_graph(3)  # all clustering coefficients = 1.0
        hist = clustering_histogram(g, bins=4)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(hist, expected)

    def test_path_no_triangles(self) -> None:
        g = nx.path_graph(4)  # all clustering coefficients = 0.0
        hist = clustering_histogram(g, bins=4)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(hist, expected)

    def test_empty_graph_returns_zero_vector(self) -> None:
        g = nx.Graph()
        hist = clustering_histogram(g, bins=4)
        np.testing.assert_allclose(hist, np.zeros(4))


@pytest.mark.unit
class TestLaplacianSpectrumHistogram:
    def test_triangle_k3(self) -> None:
        # K3 normalized Laplacian eigenvalues: 0, 3/2, 3/2 (classical: 0 and n/(n-1)).
        g = nx.complete_graph(3)
        hist = laplacian_spectrum_histogram(g, bins=4)
        # bins over [0,2]: [0,0.5),[0.5,1.0),[1.0,1.5),[1.5,2.0] -> 0 in bin0, both 1.5s in bin2
        # (floating point normalized-Laplacian eigenvalues land fractionally below 1.5).
        expected = np.array([1.0 / 3.0, 0.0, 2.0 / 3.0, 0.0])
        np.testing.assert_allclose(hist, expected, atol=1e-9)

    def test_star_graph(self) -> None:
        # K_{1,4} normalized Laplacian eigenvalues: 0, 1, 1, 1, 2 (classical star spectrum).
        g = nx.star_graph(4)
        hist = laplacian_spectrum_histogram(g, bins=4)
        expected = np.array([1.0 / 5.0, 0.0, 3.0 / 5.0, 1.0 / 5.0])
        np.testing.assert_allclose(hist, expected, atol=1e-9)

    def test_no_nodes_all_mass_in_bin_zero(self) -> None:
        g = nx.Graph()
        hist = laplacian_spectrum_histogram(g, bins=10)
        expected = np.zeros(10)
        expected[0] = 1.0
        np.testing.assert_allclose(hist, expected)

    def test_no_edges_all_mass_in_bin_zero(self) -> None:
        g = nx.Graph()
        g.add_nodes_from(["a", "b", "c"])
        hist = laplacian_spectrum_histogram(g, bins=10)
        expected = np.zeros(10)
        expected[0] = 1.0
        np.testing.assert_allclose(hist, expected)

    def test_l1_normalized(self) -> None:
        g = nx.erdos_renyi_graph(15, 0.3, seed=3)
        hist = laplacian_spectrum_histogram(g, bins=20)
        assert hist.sum() == pytest.approx(1.0)


def _random_descriptor_population(
    rng: np.random.Generator, n: int, dim: int, center: np.ndarray, noise: float
) -> list[np.ndarray]:
    """Build a list of `n` descriptor-like vectors clustered around `center`."""
    raw = center[None, :] + rng.normal(scale=noise, size=(n, dim))
    raw = np.clip(raw, 0.0, None)
    raw = raw / raw.sum(axis=1, keepdims=True)
    return list(raw)


@pytest.mark.unit
class TestMmdSquared:
    def test_identical_sets_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        a = _random_descriptor_population(rng, 10, 6, np.array([1, 2, 3, 2, 1, 0.5]), 0.1)
        b = [row.copy() for row in a]
        config = MMDConfig()
        result = mmd_squared(a, b, config)
        assert result.canonical == pytest.approx(0.0, abs=1e-12)
        for v in result.by_scale.values():
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_different_populations_exceed_ten_times_same_population_baseline(self) -> None:
        rng = np.random.default_rng(1)
        center_a = np.array([5.0, 1.0, 0.5, 0.2, 0.1, 0.1])
        center_b = np.array([0.1, 0.1, 0.2, 0.5, 1.0, 5.0])
        a1 = _random_descriptor_population(rng, 30, 6, center_a, 0.05)
        a2 = _random_descriptor_population(rng, 30, 6, center_a, 0.05)
        b = _random_descriptor_population(rng, 30, 6, center_b, 0.05)
        config = MMDConfig()
        baseline = mmd_squared(a1, a2, config).canonical
        cross = mmd_squared(a1, b, config).canonical
        assert baseline >= 0.0
        assert cross > 10 * max(baseline, 1e-12)

    def test_median_zero_guard(self) -> None:
        # All vectors identical across a and b -> all pairwise distances 0 -> guard to 1.0.
        a = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        b = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        config = MMDConfig()
        result = mmd_squared(a, b, config)
        assert result.median_bandwidth == pytest.approx(1.0)
        assert result.canonical == pytest.approx(0.0, abs=1e-12)

    def test_canonical_matches_scale_one(self) -> None:
        rng = np.random.default_rng(2)
        a = _random_descriptor_population(rng, 8, 4, np.array([1, 1, 1, 1]), 0.2)
        b = _random_descriptor_population(rng, 8, 4, np.array([2, 1, 0.5, 0.2]), 0.2)
        config = MMDConfig(bandwidth_scales=(1.0, 2.0))
        result = mmd_squared(a, b, config)
        assert result.canonical == pytest.approx(result.by_scale[1.0])


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
class TestEvaluateAssembledGraph:
    def test_identical_graphs_give_zero_mmd_and_unit_density(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        g_pred = g_ref.copy()
        config = MMDConfig()
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
        for stat in STATISTICS:
            assert report.aggregate[stat] == pytest.approx(0.0, abs=1e-9)
        assert report.relative_density == pytest.approx(1.0)
        assert report.self_loops_pred == 0
        assert report.self_loops_ref == 0

    def test_self_loops_reported_separately_and_stripped_from_topology(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        g_pred = g_ref.copy()
        # Add self-loops to g_pred; they must not affect topology metrics, only the count.
        first_node = next(iter(g_pred.nodes()))
        g_pred.add_edge(first_node, first_node)
        config = MMDConfig()
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
        assert report.self_loops_pred == 1
        assert report.self_loops_ref == 0
        assert report.relative_density == pytest.approx(1.0)

    def test_per_size_keys_match_buckets(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        g_pred = g_ref.copy()
        config = MMDConfig()
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
        assert set(report.per_size.keys()) == set(buckets.keys())
        for size in buckets:
            assert set(report.per_size[size].keys()) == set(STATISTICS)


@pytest.mark.unit
class TestNoiseFloor:
    def test_floor_is_small_but_nonzero(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        config = MMDConfig()
        floor = noise_floor(g_ref, buckets, config, seed=7, n_splits=5)
        assert set(floor.keys()) == set(buckets.keys())
        for size in buckets:
            for stat in STATISTICS:
                value = floor[size][stat]
                assert value >= 0.0
                assert value < 0.5  # small relative to a clearly-different-population MMD


@pytest.mark.unit
class TestBootstrapMmd:
    def test_shape_and_type(self) -> None:
        g_ref, buckets = _seeded_er_graph_and_buckets()
        g_pred = g_ref.copy()
        config = MMDConfig()
        result = bootstrap_mmd(g_pred, g_ref, buckets, config, seed=11, n_boot=20)
        assert set(result.keys()) == set(buckets.keys())
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
config = MMDConfig(degree_max=10, clustering_bins=20, spectral_bins=20)
report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
for stat in ("degree", "clustering", "spectral"):
    print(f"{stat}={report.aggregate[stat]:.17g}")
"""


@pytest.mark.unit
class TestHashSeedDeterminism:
    """Evaluation numbers must be bit-reproducible across Python hash randomization.

    Bucket node sets are Python sets, whose iteration order varies with
    PYTHONHASHSEED between processes. networkx subgraph views iterate the filter
    node set when it is much smaller than the graph, which permutes the Laplacian
    nodelist; eigenvalues then shift by ~1e-15 and can flip histogram bins when an
    eigenvalue sits exactly on a bin edge (e.g. 1.0). The evaluation layer must
    canonicalize node order so reported metrics do not depend on the hash seed.
    """

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
