"""Tests for the official-Graph-Similarity perturbation check."""

import networkx as nx
import numpy as np
import pytest
from src.eval.composite import perturbation_check
from src.eval.graph_metrics import MMDConfig


def _seeded_ws_graph_and_buckets() -> tuple[nx.Graph, dict[int, list[set[str]]]]:
    # p=0.0 (a pure ring lattice) starts maximally structured/clustered, giving
    # degree-preserving swaps a long runway to keep diverging out to fraction 0.5
    # without saturating near the configuration-model baseline partway through.
    g = nx.watts_strogatz_graph(300, k=6, p=0.0, seed=123)
    g = nx.relabel_nodes(g, {i: f"n{i:04d}" for i in g.nodes()})
    rng = np.random.default_rng(123)
    nodes = list(g.nodes())
    buckets: dict[int, list[set[str]]] = {}
    for size in (20, 40):
        buckets[size] = [
            set(rng.choice(nodes, size=size, replace=False).tolist()) for _ in range(8)
        ]
    return g, buckets


@pytest.mark.slow
@pytest.mark.unit
class TestPerturbationCheck:
    def test_passes_and_similarity_decreases_with_perturbation(self) -> None:
        g_ref, buckets = _seeded_ws_graph_and_buckets()
        config = MMDConfig()
        result = perturbation_check(
            g_ref,
            buckets,
            config,
            n_trials=10,
            seed=99,
        )
        assert result.passed, f"perturbation check failed: {result.failures}"
        for mode in result.similarities:
            sims = result.similarities[mode]
            # baseline (fraction 0.0) should be the highest similarity.
            assert sims[0] == max(sims)
            assert sims[-1] < 0.8 * sims[0]

    def test_never_mutates_reference_graph(self) -> None:
        g_ref, buckets = _seeded_ws_graph_and_buckets()
        edges_before = set(g_ref.edges())
        config = MMDConfig()
        perturbation_check(g_ref, buckets, config, seed=99)
        assert set(g_ref.edges()) == edges_before
