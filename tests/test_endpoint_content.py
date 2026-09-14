import itertools

import networkx as nx
import numpy as np
from src.experiments.endpoint_content import context_features, paired_intervals, sampled_rows


def test_context_against_explicit_edge_deletion() -> None:
    nodes = ["u", "v", "a", "b", "c"]
    edges = list(itertools.combinations(nodes, 2))
    for mask in range(2 ** len(edges)):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(e for i, e in enumerate(edges) if mask & (1 << i))
        got = context_features(g, [("u", "v")])[0]
        np.testing.assert_array_equal(got, context_features(g, [("v", "u")])[0])
        if g.has_edge("u", "v"):
            g.remove_edge("u", "v")
        tuples = []
        for u in ("u", "v"):
            tuples.append(
                np.log1p(
                    [
                        g.degree(u),
                        nx.triangles(g, u),
                        sum(g.degree(v) for v in g[u]) / max(1, g.degree(u)),
                    ]
                )
            )
        d, t, n = np.array(tuples).T
        expected = [
            *sorted(d),
            *sorted(t),
            *sorted(n),
            np.dot(d, t),
            np.dot(d, n),
            np.dot(t, n),
            np.sum(d * t * n),
        ]
        np.testing.assert_allclose(got, expected)


def test_sampler_reproducibility_and_boundary() -> None:
    g = nx.path_graph([f"n{i}" for i in range(50)])
    pairs, labels = sampled_rows(g, 42)
    again, other_labels = sampled_rows(g, 42)
    assert pairs == again
    np.testing.assert_array_equal(labels, other_labels)
    assert len(set(pairs)) == len(pairs)
    assert sum(labels) == g.number_of_edges()
    assert all(u != v and u in g and v in g for u, v in pairs)
    assert all(g.has_edge(u, v) == bool(y) for (u, v), y in zip(pairs, labels, strict=True))


def test_paired_node_resampling_preserves_constant_gain() -> None:
    pairs = np.array(list(itertools.combinations(range(30), 2)))
    labels = np.array([(u + v) % 2 for u, v in pairs])
    differences = np.tile([0.2, -0.3], (len(pairs), 1))
    intervals = paired_intervals(labels, differences, pairs)
    np.testing.assert_allclose(intervals, [[0.2, 0.2], [-0.3, -0.3]])
