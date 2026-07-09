"""Tests for src.eval.composite: graph-similarity composite and perturbation check."""

import networkx as nx
import numpy as np
import pytest
from src.eval.composite import (
    CompositeDefinition,
    graph_similarity,
    perturbation_check,
)
from src.eval.graph_metrics import MMDConfig


@pytest.mark.unit
class TestCompositeDefinition:
    def test_default_construction_does_not_raise(self) -> None:
        definition = CompositeDefinition()
        assert definition.statistics == ("degree", "clustering", "spectral")
        assert sum(definition.weights.values()) == pytest.approx(1.0)

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            CompositeDefinition(weights={"degree": 0.5, "clustering": 0.5, "spectral": 0.5})

    def test_non_positive_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            CompositeDefinition(scales={"degree": 0.0, "clustering": 1.0, "spectral": 1.0})


@pytest.mark.unit
class TestGraphSimilarity:
    def test_identical_graph_similarity_is_one(self) -> None:
        definition = CompositeDefinition()
        mmd2 = {"degree": 0.0, "clustering": 0.0, "spectral": 0.0}
        assert graph_similarity(mmd2, definition) == pytest.approx(1.0)

    def test_hand_computed_value(self) -> None:
        definition = CompositeDefinition(
            weights={"degree": 1 / 3, "clustering": 1 / 3, "spectral": 1 / 3},
            scales={"degree": 1.0, "clustering": 1.0, "spectral": 1.0},
        )
        mmd2 = {"degree": 0.1, "clustering": 0.2, "spectral": 0.3}
        # exp(-(0.1+0.2+0.3)/3) = exp(-0.2)
        assert graph_similarity(mmd2, definition) == pytest.approx(np.exp(-0.2))

    def test_higher_mmd_gives_lower_similarity(self) -> None:
        definition = CompositeDefinition()
        low = graph_similarity({"degree": 0.01, "clustering": 0.01, "spectral": 0.01}, definition)
        high = graph_similarity({"degree": 1.0, "clustering": 1.0, "spectral": 1.0}, definition)
        assert low > high


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
        config = MMDConfig(degree_max=30, clustering_bins=20, spectral_bins=20)
        # Scales tuned smaller than the (uncalibrated) default so the composite is
        # sensitive enough on this modest synthetic graph/bucket size to visibly
        # separate perturbation levels; production tau_k are calibrated separately.
        definition = CompositeDefinition(scales={"degree": 0.3, "clustering": 0.3, "spectral": 0.3})
        result = perturbation_check(
            g_ref,
            buckets,
            config,
            definition,
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
        config = MMDConfig(degree_max=30, clustering_bins=20, spectral_bins=20)
        definition = CompositeDefinition()
        perturbation_check(g_ref, buckets, config, definition, seed=99)
        assert set(g_ref.edges()) == edges_before
