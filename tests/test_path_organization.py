import itertools

import networkx as nx
import pytest
from src.experiments.path_organization import BASE, pair_features


def features(graph: nx.Graph, u: str = "u", v: str = "v") -> dict[str, float]:
    return pair_features({x: set(graph[x]) for x in graph}, u, v)


def test_proposed_equal_count_structures() -> None:
    independent = nx.Graph([("u", "a"), ("a", "b"), ("b", "v"), ("u", "c"), ("c", "d"), ("d", "v")])
    bottleneck = nx.Graph([("u", "a"), ("a", "b"), ("b", "v"), ("u", "c"), ("c", "b"), ("v", "d")])
    a, b = features(independent), features(bottleneck)
    assert [a[k] for k in BASE] == [b[k] for k in BASE]
    assert a["max_edge_share"] == 0.5
    assert b["max_edge_share"] == 1
    assert a["disjoint_path_pair_fraction"] == 1
    assert b["disjoint_path_pair_fraction"] == 0


def test_small_graphs_against_explicit_edge_deleted_paths() -> None:
    nodes = ["u", "v", "a", "b", "c"]
    possible = list(itertools.combinations(nodes, 2))
    for mask in range(2 ** len(possible)):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(e for i, e in enumerate(possible) if mask & (1 << i))
        observed = features(graph)
        assert observed == features(graph, "v", "u")
        deleted = graph.copy()
        if deleted.has_edge("u", "v"):
            deleted.remove_edge("u", "v")
        assert observed == features(deleted)
        paths = [p for p in nx.all_simple_paths(deleted, "u", "v", cutoff=3) if len(p) == 4]
        assert observed["l3"] == len(paths)
        pairs = list(itertools.combinations(paths, 2))
        disjoint = sum(not (set(p[1:-1]) & set(q[1:-1])) for p, q in pairs)
        expected = disjoint / len(pairs) if pairs else 0
        assert observed["disjoint_path_pair_fraction"] == expected
        triangles = sorted(nx.triangles(deleted, n) for n in ("u", "v"))
        assert [observed["triangles_min"], observed["triangles_max"]] == triangles


def test_self_pairs_rejected() -> None:
    with pytest.raises(ValueError, match="Self-pairs"):
        pair_features({"u": set()}, "u", "u")
