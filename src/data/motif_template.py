"""Fixed motif-template geometry and the deterministic per-pair compiler.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``
sections 2 and 3. The template is a fixed 26-slot, 96-edge graph:
``{u, v} union C union L union R`` with ``|C| = |L| = |R| = 8``. Closure edges
``u-c`` and ``c-v`` carry a shared witness, attachment edges ``u-l`` and ``r-v``
carry the endpoints' bridge attachments, and the 64 interior edges ``l-r``
carry the bridge crossing. Every other adjacency entry, the diagonal and the
queried ``u-v`` entry are exactly zero.
"""

from __future__ import annotations

N_SLOTS = 26
N_EDGES = 96

SLOT_U = 0
SLOT_V = 1
C_SLOTS: tuple[int, ...] = tuple(range(2, 10))
L_SLOTS: tuple[int, ...] = tuple(range(10, 18))
R_SLOTS: tuple[int, ...] = tuple(range(18, 26))

ROLE_ENDPOINT = 0
ROLE_CLOSURE = 1
ROLE_BRIDGE = 2
N_ROLES = 3

TYPE_CLOSURE = 0
TYPE_ATTACH = 1
TYPE_INTERIOR = 2
N_EDGE_TYPES = 3

SLOT_ROLES: tuple[int, ...] = (ROLE_ENDPOINT,) * 2 + (ROLE_CLOSURE,) * 8 + (ROLE_BRIDGE,) * 16

CLOSURE_U = slice(0, 8)
CLOSURE_V = slice(8, 16)
ATTACH_L = slice(16, 24)
ATTACH_R = slice(24, 32)
INTERIOR = slice(32, 96)

EDGE_ENDPOINTS: tuple[tuple[int, int], ...] = (
    tuple((SLOT_U, c) for c in C_SLOTS)
    + tuple((c, SLOT_V) for c in C_SLOTS)
    + tuple((SLOT_U, left) for left in L_SLOTS)
    + tuple((right, SLOT_V) for right in R_SLOTS)
    + tuple((L_SLOTS[i], R_SLOTS[j]) for i in range(8) for j in range(8))
)

EDGE_TYPES: tuple[int, ...] = (TYPE_CLOSURE,) * 16 + (TYPE_ATTACH,) * 16 + (TYPE_INTERIOR,) * 64


def _interior_index(left: int, right: int) -> int:
    """Return the edge index of interior slot pair ``(left, right)``.

    Args:
        left: Left-bridge slot ordinal in ``range(8)``.
        right: Right-bridge slot ordinal in ``range(8)``.

    Returns:
        The index into `EDGE_ENDPOINTS` of the interior edge ``l-r``.
    """
    return 32 + 8 * left + right


def _build_swap_perm() -> tuple[int, ...]:
    """Return the edge permutation realising ``u<->v`` with ``L<->R`` (spec section 4).

    Returns:
        A length-96 permutation ``perm`` with ``perm[i]`` the new index of edge ``i``.
    """
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = 8 + k
        perm[8 + k] = k
        perm[16 + k] = 24 + k
        perm[24 + k] = 16 + k
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(right, left)
    return tuple(perm)


SWAP_PERM: tuple[int, ...] = _build_swap_perm()


def role_permutation(
    sigma_c: tuple[int, ...], sigma_l: tuple[int, ...], sigma_r: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the edge permutation induced by relabelling slots within each role.

    Args:
        sigma_c: Destination closure slot of each closure slot.
        sigma_l: Destination left-bridge slot of each left slot.
        sigma_r: Destination right-bridge slot of each right slot.

    Returns:
        A length-96 permutation ``perm`` with ``perm[i]`` the new index of edge ``i``.
    """
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = sigma_c[k]
        perm[8 + k] = 8 + sigma_c[k]
        perm[16 + k] = 16 + sigma_l[k]
        perm[24 + k] = 24 + sigma_r[k]
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(sigma_l[left], sigma_r[right])
    return tuple(perm)
