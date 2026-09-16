"""Fixed motif-template geometry, the compiler and the section 7.3 target tensor."""

from __future__ import annotations

import numpy as np
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
