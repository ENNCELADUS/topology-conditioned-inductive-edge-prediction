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
    get_coord_spec,
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


def test_coordinate_statistics_pool_endpoints_and_floor_constant_columns() -> None:
    coords = np.zeros((5, COORD_DIM), dtype=np.float32)
    coords[:, 0] = np.arange(5)  # endpoint_u degree column; endpoint_v's stays zero
    relation_start = FIELD_SLICES["relation"].start
    coords[:, relation_start] = np.arange(5)
    mean, std = coordinate_statistics(coords)
    # Endpoint statistics are pooled over both endpoints: {0..4} and five zeros.
    assert mean[0] == pytest.approx(1.0)
    assert std[0] == pytest.approx(np.sqrt(2.0))
    np.testing.assert_array_equal(
        mean[FIELD_SLICES["endpoint_u"]], mean[FIELD_SLICES["endpoint_v"]]
    )
    np.testing.assert_array_equal(std[FIELD_SLICES["endpoint_u"]], std[FIELD_SLICES["endpoint_v"]])
    # Non-endpoint columns keep their own statistics; constant columns get unit scale.
    assert mean[relation_start] == pytest.approx(2.0)
    assert std[relation_start] == pytest.approx(np.sqrt(2.0))
    assert std[1] == 1.0
    with pytest.raises(ValueError):
        coordinate_statistics(coords[:0])


def test_sampled_pair_coordinates_keep_full_graph_context_and_order() -> None:
    graph = nx.Graph([("a", "b"), ("a", "c"), ("c", "d"), ("b", "d"), ("d", "e")])
    table = StructCoordinateTable(graph)
    pairs = [("a", "b"), ("b", "a"), ("a", "e"), ("e", "e"), ("a", "b")]
    u = np.asarray([table.index[a] for a, _ in pairs])
    v = np.asarray([table.index[b] for _, b in pairs])
    actual = table.coords_for_pairs(u, v)
    assert actual.dtype == np.float32
    for row, (a, b) in enumerate(pairs):
        np.testing.assert_allclose(
            actual[row], reference_pair_coords(graph, a, b), atol=1e-6, rtol=1e-5
        )
    # Even when the sampled graph is just {a,b}, the removed-edge degree is one.
    assert actual[0, 0] == pytest.approx(np.log(2))
    np.testing.assert_allclose(
        actual[0, FIELD_SLICES["endpoint_u"]], actual[1, FIELD_SLICES["endpoint_v"]]
    )
    assert table.coords_for_pairs(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    ).shape == (0, COORD_DIM)
    with pytest.raises(ValueError, match="outside"):
        table.coords_for_pairs(np.asarray([-1]), np.asarray([0]))


def test_compact_spec_matches_v1_columns_and_merges_distant_classes() -> None:
    from src.data.struct_coords import get_coord_spec

    graph = _random_graph(21, n=12)
    pairs = [(u, v) for u in graph for v in graph]
    legacy = StructCoordinateTable(graph).coords(pairs)
    compact = StructCoordinateTable(graph, spec="v2")
    spec = get_coord_spec("v2")
    actual = compact.coords(pairs)
    assert spec.fields == ("endpoint_u", "endpoint_v", "relation")
    assert actual.shape == (len(pairs), 11)
    for index, name in enumerate(spec.coord_names):
        expected = legacy[:, COORD_NAMES.index(name)]
        if name == "dist_4plus":
            expected = expected + legacy[:, COORD_NAMES.index("dist_inf")]
        np.testing.assert_array_equal(actual[:, index], expected)
    assert compact.coords([]).shape == (0, 11)
    np.testing.assert_array_equal(StructCoordinateTable(graph, spec="v1").coords(pairs), legacy)


def test_compact_statistics_pool_endpoint_roles() -> None:
    coords = np.zeros((5, 11), dtype=np.float32)
    coords[:, 0] = np.arange(5)
    mean, std = coordinate_statistics(coords, spec="v2")
    np.testing.assert_array_equal(mean[:2], mean[2:4])
    np.testing.assert_array_equal(std[:2], std[2:4])
    assert mean[0] == 1.0
    assert std[0] == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize("seed", [0, 21])
def test_v3_projects_v2_for_edges_non_edges_self_and_disconnected_pairs(seed: int) -> None:
    graph = _random_graph(seed, n=12, p=0.3)
    pairs = [(u, v) for u in graph for v in graph]
    old_spec = get_coord_spec("v2")
    spec = get_coord_spec("v3")
    table = StructCoordinateTable(graph, spec="v3")
    expected = StructCoordinateTable(graph, spec="v2").coords(pairs)
    columns = [old_spec.coord_names.index(name) for name in spec.coord_names]
    actual = table.coords(pairs)
    np.testing.assert_array_equal(actual, expected[:, columns])
    assert actual.shape == (len(pairs), 9)
    assert spec.fields == ("endpoint_u", "endpoint_v", "relation")
    assert spec.continuous_indices == tuple(range(6))
    assert spec.distance_indices == (6, 7, 8)
    assert spec.self_distance_class == 3
    assert table.coords([]).shape == (0, 9)
    swapped = table.coords([(v, u) for u, v in pairs])
    np.testing.assert_array_equal(actual[:, 0], swapped[:, 1])
    np.testing.assert_array_equal(actual[:, 1], swapped[:, 0])
    np.testing.assert_array_equal(actual[:, 2:], swapped[:, 2:])
    for row, (u, v) in enumerate(pairs):
        np.testing.assert_allclose(
            actual[row], table._exact_pair(table.index[u], table.index[v]), atol=1e-6
        )
        if u == v:
            assert actual[row, 6:].sum() == 0


def test_v3_removes_queried_edge_before_measuring_and_pools_endpoint_statistics() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")])
    actual = StructCoordinateTable(graph, spec="v3").coords([("a", "b")])
    graph.remove_edge("a", "b")
    expected = StructCoordinateTable(graph, spec="v3").coords([("a", "b")])
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0] == pytest.approx(np.log1p(2))
    assert actual[0, 1] == pytest.approx(np.log1p(1))
    assert actual[0, 6] == 1  # The removed edge leaves a common neighbour.
    coords = np.zeros((5, 9), dtype=np.float32)
    coords[:, 0] = np.arange(5)
    mean, std = coordinate_statistics(coords, spec="v3")
    assert mean[0] == mean[1] == 1
    assert std[0] == std[1] == pytest.approx(np.sqrt(2.0))
    np.testing.assert_array_equal(std[2:], np.ones(7))
