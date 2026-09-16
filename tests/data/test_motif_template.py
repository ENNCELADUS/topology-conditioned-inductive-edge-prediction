"""Fixed motif-template geometry, the compiler and the section 7.3 target tensor."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.data.motif_template import (
    ATTACH_L,
    ATTACH_R,
    C_SLOTS,
    CLOSURE_U,
    CLOSURE_V,
    EDGE_ENDPOINTS,
    EDGE_TYPES,
    INTERIOR,
    L_SLOTS,
    N_EDGES,
    N_SLOTS,
    R_SLOTS,
    SLOT_ROLES,
    SLOT_U,
    SLOT_V,
    SWAP_PERM,
    TYPE_ATTACH,
    TYPE_CLOSURE,
    TYPE_INTERIOR,
    MotifTemplateTable,
    role_permutation,
)


def test_geometry_is_26_slots_and_96_distinct_undirected_edges() -> None:
    assert (N_SLOTS, N_EDGES) == (26, 96)
    assert len(EDGE_ENDPOINTS) == 96
    assert len(EDGE_TYPES) == 96
    assert len(SLOT_ROLES) == 26
    undirected = {frozenset(edge) for edge in EDGE_ENDPOINTS}
    assert len(undirected) == 96
    assert all(len(edge) == 2 for edge in undirected)
    assert frozenset((SLOT_U, SLOT_V)) not in undirected


def test_edge_blocks_carry_the_declared_types_and_endpoints() -> None:
    assert [EDGE_ENDPOINTS[i] for i in range(*CLOSURE_U.indices(96))] == [
        (SLOT_U, c) for c in C_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*CLOSURE_V.indices(96))] == [
        (c, SLOT_V) for c in C_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*ATTACH_L.indices(96))] == [
        (SLOT_U, left) for left in L_SLOTS
    ]
    assert [EDGE_ENDPOINTS[i] for i in range(*ATTACH_R.indices(96))] == [
        (right, SLOT_V) for right in R_SLOTS
    ]
    assert [EDGE_ENDPOINTS[32 + 8 * i + j] for i in range(8) for j in range(8)] == [
        (L_SLOTS[i], R_SLOTS[j]) for i in range(8) for j in range(8)
    ]
    assert set(EDGE_TYPES[CLOSURE_U]) == set(EDGE_TYPES[CLOSURE_V]) == {TYPE_CLOSURE}
    assert set(EDGE_TYPES[ATTACH_L]) == set(EDGE_TYPES[ATTACH_R]) == {TYPE_ATTACH}
    assert set(EDGE_TYPES[INTERIOR]) == {TYPE_INTERIOR}


def test_swap_permutation_is_an_involution_that_exchanges_sides() -> None:
    perm = np.asarray(SWAP_PERM)
    assert perm.shape == (96,)
    assert sorted(perm.tolist()) == list(range(96))
    assert perm[perm].tolist() == list(range(96))
    # closure blocks exchange, attachment blocks exchange, interior transposes.
    assert perm[:8].tolist() == list(range(8, 16))
    assert perm[16:24].tolist() == list(range(24, 32))
    assert perm[32 + 8 * 3 + 5] == 32 + 8 * 5 + 3


def test_role_permutation_relabels_blocks_consistently() -> None:
    sigma_c = (1, 0, 2, 3, 4, 5, 6, 7)
    sigma_l = (7, 6, 5, 4, 3, 2, 1, 0)
    sigma_r = tuple(range(8))
    perm = np.asarray(role_permutation(sigma_c, sigma_l, sigma_r))
    assert sorted(perm.tolist()) == list(range(96))
    # A closure slot moves identically in both of its blocks.
    assert perm[0] == 1 and perm[8] == 9
    # An attachment slot moves with its whole interior row.
    assert perm[16] == 23
    assert perm[32 + 8 * 0 + 4] == 32 + 8 * 7 + 4
    # Slot roles are unchanged by a within-role relabelling.
    assert SLOT_ROLES[SLOT_U] == SLOT_ROLES[SLOT_V]


def _wedge_graph() -> nx.Graph:
    """Nodes u and v share witnesses w1 (degree 2) and w2 (degree 4); u-v is an edge."""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("u", "v"),
            ("u", "w1"),
            ("w1", "v"),
            ("u", "w2"),
            ("w2", "v"),
            ("w2", "x1"),
            ("w2", "x2"),
        ]
    )
    return graph


def test_queried_edge_is_removed_before_neighbourhoods_and_weights() -> None:
    table = MotifTemplateTable(_wedge_graph())
    row = table.compile_row("u", "v")
    # v is never a witness of (u, v) even though it is a neighbour of u.
    assert set(row.closure) == {"w1", "w2"}
    # Witness weights are 1/sqrt(d(w)) on the edge-deleted graph: d(w1)=2, d(w2)=4.
    order = {node: i for i, node in enumerate(row.closure)}
    assert row.weights[order["w1"]] == pytest.approx(2.0**-0.5)
    assert row.weights[8 + order["w1"]] == pytest.approx(2.0**-0.5)
    assert row.weights[order["w2"]] == pytest.approx(4.0**-0.5)
    # Witnesses come first by descending 1/d(w), i.e. ascending degree.
    assert row.closure[0] == "w1"


def test_endpoint_degrees_are_decremented_by_the_removed_edge() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("u", "v"), ("u", "a"), ("a", "b"), ("b", "v")])
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert row.closure == ()
    assert row.left == ("a",) and row.right == ("b",)
    # d(a) = 2 and d(b) = 2 on the edge-deleted graph.
    assert row.weights[16] == pytest.approx(2.0**-0.5)
    assert row.weights[24] == pytest.approx(2.0**-0.5)
    assert row.weights[32] == pytest.approx(1.0)


def test_only_at_most_eight_witnesses_survive_truncation() -> None:
    graph = nx.Graph()
    graph.add_edge("u", "v")
    for i in range(12):
        graph.add_edges_from([("u", f"w{i}"), (f"w{i}", "v")])
        for j in range(i):
            graph.add_edge(f"w{i}", f"pad{i}_{j}")
    table = MotifTemplateTable(graph)
    row = table.compile_row("u", "v")
    assert len(row.closure) == 8
    assert row.closure == tuple(f"w{i}" for i in range(8))
    assert float(row.weights[8:16].sum()) > 0.0


def test_swapping_endpoints_permutes_the_weight_vector_by_swap_perm() -> None:
    table = MotifTemplateTable(_wedge_graph())
    forward = table.compile_row("u", "v")
    reverse = table.compile_row("v", "u")
    np.testing.assert_allclose(reverse.weights, forward.weights[list(SWAP_PERM)], atol=0)
    assert reverse.closure == forward.closure
    assert reverse.left == forward.right and reverse.right == forward.left


def test_self_rows_get_an_explicit_empty_template() -> None:
    table = MotifTemplateTable(_wedge_graph())
    row = table.compile_row("u", "u")
    assert not row.weights.any()
    assert (row.closure, row.left, row.right) == ((), (), ())
