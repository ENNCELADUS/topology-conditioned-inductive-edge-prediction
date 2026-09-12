"""Structural-coordinate tests: dense path vs exact path vs a NetworkX reference."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.data.struct_coords import (
    COORD_DIM,
    COORD_NAMES,
    FIELD_SLICES,
    StructCoordinateTable,
    coordinate_statistics,
    reference_pair_coords,
)


def _random_graph(seed: int, n: int = 40, p: float = 0.12) -> nx.Graph:
    graph = nx.gnp_random_graph(n, p, seed=seed)
    graph = nx.relabel_nodes(graph, {i: f"node_{i:06d}" for i in range(n)})
    graph.add_node("node_000099")  # an isolate survives every path
    return graph


def _all_pairs(graph: nx.Graph, rng: np.random.Generator, count: int) -> list[tuple[str, str]]:
    nodes = sorted(graph.nodes)
    pairs = [
        (nodes[int(i)], nodes[int(j)])
        for i, j in rng.integers(len(nodes), size=(count, 2)).tolist()
    ]
    pairs += [(node, node) for node in nodes[:3]]
    pairs += list(graph.edges)[:12]
    return pairs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_table_matches_the_networkx_reference_on_edges_non_edges_and_self_pairs(
    seed: int,
) -> None:
    graph = _random_graph(seed)
    table = StructCoordinateTable(graph)
    rng = np.random.default_rng(seed)
    pairs = _all_pairs(graph, rng, 40)
    observed = table.coords(pairs)
    assert observed.shape == (len(pairs), COORD_DIM)
    assert len(COORD_NAMES) == COORD_DIM
    for row, (u, v) in enumerate(pairs):
        expected = reference_pair_coords(graph, u, v)
        np.testing.assert_allclose(
            observed[row], expected, rtol=1e-4, atol=1e-5, err_msg=f"{u} {v}"
        )


def test_exact_path_agrees_with_the_dense_path_on_non_edges() -> None:
    graph = _random_graph(3)
    table = StructCoordinateTable(graph)
    rng = np.random.default_rng(3)
    nodes = table.nodes
    checked = 0
    for i, j in rng.integers(len(nodes), size=(60, 2)).tolist():
        if table.is_edge(i, j):
            continue
        dense = table.coords_by_index(np.asarray([i]), np.asarray([j]))[0]
        exact = table._exact_pair(i, j)  # noqa: SLF001 - the reference implementation
        np.testing.assert_allclose(dense, exact, rtol=1e-4, atol=1e-5)
        checked += 1
    assert checked > 20


def test_queried_edge_is_removed_before_measuring() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")])
    table = StructCoordinateTable(graph)
    coords = table.coords([("a", "c")])[0]
    relation = coords[FIELD_SLICES["relation"]]
    # Without the edge a-c, a and c share neighbours b and d and sit at distance 2.
    assert relation[0] == pytest.approx(np.log1p(2.0))
    assert relation[7] == 1.0  # dist_2
    endpoint_u = coords[FIELD_SLICES["endpoint_u"]]
    assert endpoint_u[0] == pytest.approx(np.log1p(2.0))  # degree of a drops from 3 to 2
    without = nx.Graph(graph)
    without.remove_edge("a", "c")
    np.testing.assert_allclose(coords, reference_pair_coords(without, "a", "c"), rtol=1e-5)


def test_swapping_endpoints_swaps_endpoint_fields_and_fixes_the_rest() -> None:
    graph = _random_graph(5)
    table = StructCoordinateTable(graph)
    pairs = [("node_000001", "node_000007"), *list(graph.edges)[:5]]
    forward = table.coords(pairs)
    backward = table.coords([(v, u) for u, v in pairs])
    np.testing.assert_allclose(
        forward[:, FIELD_SLICES["endpoint_u"]], backward[:, FIELD_SLICES["endpoint_v"]]
    )
    np.testing.assert_allclose(
        forward[:, FIELD_SLICES["endpoint_v"]], backward[:, FIELD_SLICES["endpoint_u"]]
    )
    np.testing.assert_allclose(
        forward[:, FIELD_SLICES["relation"]], backward[:, FIELD_SLICES["relation"]]
    )
    np.testing.assert_allclose(
        forward[:, FIELD_SLICES["context"]], backward[:, FIELD_SLICES["context"]]
    )


def test_self_pair_and_isolate_are_finite_and_well_defined() -> None:
    graph = _random_graph(7)
    table = StructCoordinateTable(graph)
    coords = table.coords([("node_000099", "node_000099"), ("node_000099", "node_000001")])
    assert np.isfinite(coords).all()
    relation = coords[1, FIELD_SLICES["relation"]]
    assert relation[-1] == 1.0  # disconnected
    self_row = coords[0, FIELD_SLICES["relation"]]
    assert self_row[7:].sum() == 0.0  # no distance category for a self pair


def test_rejects_self_loops_and_unknown_nodes() -> None:
    graph = nx.Graph([("a", "b"), ("b", "b")])
    with pytest.raises(ValueError, match="loopless"):
        StructCoordinateTable(graph)
    table = StructCoordinateTable(nx.Graph([("a", "b")]))
    with pytest.raises(KeyError):
        table.coords([("a", "zzz")])


def test_coordinate_statistics_floor_constant_columns() -> None:
    coords = np.zeros((5, COORD_DIM), dtype=np.float32)
    coords[:, 0] = np.arange(5)
    mean, std = coordinate_statistics(coords)
    assert mean[0] == pytest.approx(2.0)
    assert std[0] == pytest.approx(np.sqrt(2.0))
    assert std[1] == 1.0
    with pytest.raises(ValueError):
        coordinate_statistics(coords[:0])
