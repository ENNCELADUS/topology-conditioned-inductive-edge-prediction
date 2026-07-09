"""Tests for src.data.partition."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from src.data.artifacts import load_benchmark
from src.data.partition import build_g_struct, derive_partition, strip_self_loops


def _chain_positives(n: int) -> list[tuple[str, str]]:
    return [(f"n{i}", f"n{i + 1}") for i in range(n)]


class TestDerivePartitionDeterminism:
    def test_same_seed_twice_gives_identical_partition(self) -> None:
        positives = _chain_positives(20)
        first = derive_partition(positives, seed=42)
        second = derive_partition(positives, seed=42)
        assert first.e_msg == second.e_msg
        assert first.e_sup == second.e_sup

    def test_different_seeds_give_different_partition(self) -> None:
        positives = _chain_positives(20)
        first = derive_partition(positives, seed=0)
        second = derive_partition(positives, seed=1)
        assert first.e_msg != second.e_msg

    def test_seed_is_recorded_on_the_result(self) -> None:
        partition = derive_partition(_chain_positives(5), seed=7)
        assert partition.seed == 7


class TestDerivePartitionSetAlgebra:
    def test_disjoint_and_union_equals_canonicalized_deduped_input(self) -> None:
        positives = [("n2", "n1"), ("n1", "n2"), ("n3", "n4")]
        partition = derive_partition(positives, seed=1)
        assert partition.e_msg & partition.e_sup == frozenset()
        assert partition.e_msg | partition.e_sup == frozenset({("n1", "n2"), ("n3", "n4")})

    def test_msg_fraction_boundary_on_small_n(self) -> None:
        positives = _chain_positives(5)
        partition = derive_partition(positives, seed=3, msg_fraction=0.8)
        assert len(partition.e_msg) == 4
        assert len(partition.e_sup) == 1


class TestBuildGStruct:
    def test_isolated_train_nodes_are_present(self) -> None:
        train_nodes = ["n1", "n2", "n3", "n4"]
        e_msg = [("n1", "n2")]
        g = build_g_struct(train_nodes, e_msg)
        assert set(g.nodes()) == {"n1", "n2", "n3", "n4"}
        assert g.number_of_edges() == 1

    def test_self_loops_in_e_msg_are_dropped(self) -> None:
        train_nodes = ["n1", "n2"]
        e_msg = [("n1", "n2"), ("n1", "n1")]
        g = build_g_struct(train_nodes, e_msg)
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
    def test_built_from_real_train_positives_seed_0(self, benchmark_root: Path) -> None:
        benchmark = load_benchmark(benchmark_root, "breadth_first")
        train_pairs = benchmark.split.train_pairs
        train_plus = [
            pair
            for pair, label in zip(train_pairs.pairs, train_pairs.labels, strict=True)
            if label == 1
        ]
        partition = derive_partition(train_plus, seed=0)
        g_struct = build_g_struct(benchmark.split.train_nodes, partition.e_msg)

        assert g_struct.number_of_nodes() == 8_072
        assert nx.number_of_selfloops(g_struct) == 0
        non_self_e_msg = {(u, v) for u, v in partition.e_msg if u != v}
        assert g_struct.number_of_edges() == len(non_self_e_msg)
