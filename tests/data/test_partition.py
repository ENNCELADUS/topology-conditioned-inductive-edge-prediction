"""Tests for src.data.partition."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from src.data.artifacts import load_benchmark
from src.data.partition import build_g_struct, derive_training_interactions, strip_self_loops


def _chain_positives(n: int) -> list[tuple[str, str]]:
    return [(f"n{i}", f"n{i + 1}") for i in range(n)]


class TestDeriveTrainingInteractions:
    def test_topology_and_classification_share_every_nonself_interaction(self) -> None:
        interactions = derive_training_interactions(_chain_positives(20))
        assert interactions.topology_edges == interactions.positives

    def test_canonicalizes_and_deduplicates_without_splitting(self) -> None:
        positives = [("n2", "n1"), ("n1", "n2"), ("n3", "n4")]
        interactions = derive_training_interactions(positives)
        assert interactions.positives == frozenset({("n1", "n2"), ("n3", "n4")})

    def test_only_topology_projection_drops_self_pairs(self) -> None:
        interactions = derive_training_interactions([("n1", "n1"), ("n1", "n2")])
        assert interactions.positives == frozenset(
            {("n1", "n1"), ("n1", "n2")}
        )
        assert interactions.topology_edges == frozenset({("n1", "n2")})


class TestBuildGStruct:
    def test_isolated_train_nodes_are_present(self) -> None:
        train_nodes = ["n1", "n2", "n3", "n4"]
        training_interactions = [("n1", "n2")]
        g = build_g_struct(train_nodes, training_interactions)
        assert set(g.nodes()) == {"n1", "n2", "n3", "n4"}
        assert g.number_of_edges() == 1

    def test_self_loops_in_training_interactions_are_dropped(self) -> None:
        train_nodes = ["n1", "n2"]
        training_interactions = [("n1", "n2"), ("n1", "n1")]
        g = build_g_struct(train_nodes, training_interactions)
        assert list(nx.selfloop_edges(g)) == []
        assert g.number_of_edges() == 1


class TestStripSelfLoops:
    def test_leaves_input_graph_unmodified(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("a", "a")])
        result = strip_self_loops(g)
        assert nx.number_of_selfloops(g) == 1
        assert nx.number_of_selfloops(result) == 0
        assert result.number_of_edges() == 1


@pytest.mark.integration
class TestGStructRealData:
    def test_built_from_all_real_train_positives(self, benchmark_root: Path) -> None:
        benchmark = load_benchmark(benchmark_root, "breadth_first")
        train_pairs = benchmark.split.train_pairs
        train_plus = [
            pair
            for pair, label in zip(train_pairs.pairs, train_pairs.labels, strict=True)
            if label == 1
        ]
        interactions = derive_training_interactions(train_plus)
        g_struct = build_g_struct(benchmark.split.train_nodes, interactions.topology_edges)

        assert g_struct.number_of_nodes() == 8_072
        assert nx.number_of_selfloops(g_struct) == 0
        assert g_struct.number_of_edges() == len(interactions.topology_edges)
