"""Exact L3 extraction, training boundary, and disjoint graph batching."""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from src.data.l3_patterns import L3Pattern, L3PatternCache, batch_patterns


def _edges(pattern: L3Pattern) -> set[tuple[str, str]]:
    return {
        tuple(sorted((pattern.nodes[i], pattern.nodes[j])))
        for i, j in pattern.edge_index.t().tolist()
    }


def test_overlap_union_deduplicates_edges_without_inducing_extras() -> None:
    graph = nx.Graph(
        [("u", "a"), ("a", "b"), ("a", "c"), ("b", "v"), ("c", "v"), ("b", "c")]
    )
    graph.add_edge("u", "v")
    graph.add_edge("a", "a")
    pattern = L3PatternCache(graph, set(graph)).extract("u", "v")
    assert pattern.nodes == ("u", "v", "a", "b", "c")
    assert pattern.num_paths == 2
    assert _edges(pattern) == {("a", "u"), ("a", "b"), ("a", "c"), ("b", "v"), ("c", "v")}
    assert pattern.edge_index.shape == (2, 10)
    without = graph.copy()
    without.remove_edge("u", "v")
    expected = L3PatternCache(without, set(graph)).extract("u", "v")
    assert torch.equal(pattern.edge_index, expected.edge_index)
    assert graph.has_edge("u", "v") and graph.has_edge("a", "a")


def test_zero_paths_and_self_pairs_retain_two_endpoint_slots() -> None:
    graph = nx.Graph([("u", "v"), ("v", "w"), ("w", "u"), ("u", "u")])
    cache = L3PatternCache(graph, {*graph, "isolated"})
    for a, b in [("u", "v"), ("u", "u"), ("u", "isolated")]:
        pattern = cache.extract(a, b)
        assert pattern.nodes == (a, b)
        assert pattern.num_paths == 0
        assert pattern.edge_index.shape == (2, 0)
        assert pattern.edge_index.dtype == torch.long


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_all_simple_paths_reference(seed: int) -> None:
    graph = nx.relabel_nodes(nx.gnp_random_graph(12, 0.4, seed=seed), str)
    cache = L3PatternCache(graph, set(graph), cache_size=0)
    for a in graph:
        for b in graph:
            if a == b:
                continue
            reference = graph.copy()
            if reference.has_edge(a, b):
                reference.remove_edge(a, b)
            paths = [
                path for path in nx.all_simple_paths(reference, a, b, cutoff=3) if len(path) == 4
            ]
            expected = {
                tuple(sorted((u, v)))
                for path in paths
                for u, v in zip(path[:-1], path[1:], strict=True)
            }
            pattern = cache.extract(a, b)
            assert pattern.num_paths == len(paths)
            assert _edges(pattern) == expected


def test_boundaries_reject_illegal_graph_nodes_and_queries() -> None:
    graph = nx.Graph([("train", "heldout")])
    with pytest.raises(ValueError, match="illegal nodes"):
        L3PatternCache(graph, {"train"})
    cache = L3PatternCache(nx.empty_graph(["train"]), {"train"})
    with pytest.raises(ValueError, match="legal training nodes"):
        cache.extract("train", "heldout")
    for graph_type in [nx.DiGraph, nx.MultiGraph]:
        with pytest.raises(ValueError, match="simple undirected"):
            L3PatternCache(graph_type(), set())
    with pytest.raises(ValueError, match="nonnegative"):
        L3PatternCache(nx.Graph(), set(), cache_size=-1)


def test_deterministic_under_insertion_order_eviction_and_graph_mutation() -> None:
    edges = [("u", "a"), ("a", "b"), ("b", "v"), ("u", "c"), ("c", "b")]
    graph = nx.Graph(edges)
    cache = L3PatternCache(graph, set(graph), cache_size=1)
    first = cache.extract("u", "v")
    graph.clear_edges()
    cache.extract("a", "c")  # Evict, then recompute from the immutable snapshot.
    again = cache.extract("u", "v")
    reversed_pattern = L3PatternCache(nx.Graph(edges[::-1]), set(graph)).extract("u", "v")
    for pattern in [again, reversed_pattern]:
        assert pattern.nodes == first.nodes
        assert pattern.num_paths == first.num_paths
        assert torch.equal(pattern.edge_index, first.edge_index)


def test_batch_disjoint_offsets_self_slots_and_featureless_failure() -> None:
    graph = nx.Graph([("u", "a"), ("a", "b"), ("b", "v")])
    cache = L3PatternCache(graph, set(graph))
    patterns = [cache.extract("u", "u"), cache.extract("u", "v")]
    features = {
        node: torch.tensor([index, index + 1], dtype=torch.bfloat16)
        for index, node in enumerate(graph)
    }
    x, edges, weights, batch = batch_patterns(patterns, features, torch.device("cpu"))
    assert x.dtype == weights.dtype == torch.float32
    assert x.shape == (6, 2)
    assert torch.equal(x[0], x[1])
    assert torch.equal(edges, patterns[1].edge_index + 2)
    assert torch.equal(weights, torch.ones(6))
    assert batch.tolist() == [0, 0, 1, 1, 1, 1]
    del features["a"]
    with pytest.raises(KeyError, match="a"):
        batch_patterns(patterns, features, torch.device("cpu"))
    with pytest.raises(ValueError, match="empty list"):
        batch_patterns([], features, torch.device("cpu"))
