"""Deterministic KD anchor/context sampling over V_fit (LLP-style near + random sets).

Per anchor ``u``, the near set is every node within <=2 hops of ``u`` in ``g_fit``
(minus ``u`` itself) and the random set is drawn uniformly from the remaining
``V_fit`` universe -- the two context "arms" `src/distill/losses.py`'s LLP_R/LLP_D
rows are computed over. Selection is reproducible per anchor, independent of
process/rank/iteration order, via the same blake2b node-keyed RNG idiom
`src.train_egostitch._edge_target_rng` uses for edge-target sampling (spec
Sec 14.4.1): the anchor's node id, not its position, seeds its draw, so context
sets do not move if `node_ids` is re-sorted or re-sharded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

_HOP_CUTOFF = 2


def _anchor_rng(node_id: str, *, seed: int) -> np.random.Generator:
    """Anchor-keyed RNG mirroring `_edge_target_rng`'s blake2b XOR scheme.

    Context sets are sampled once (not per epoch), so only `seed` -- no
    epoch -- enters the key.
    """
    if not 0 <= seed < 2**64:
        raise ValueError("context sampler seed must fit an unsigned 64-bit integer")
    node_key = int.from_bytes(
        hashlib.blake2b(node_id.encode("utf-8"), digest_size=16).digest(), "little"
    )
    return np.random.default_rng(node_key ^ seed)


@dataclass(frozen=True)
class ContextSets:
    """CSR-style anchor -> partner context sets, as ordered pair-row arrays.

    Attributes:
        anchor_idx: Shape ``(P,)`` int32 row -> anchor position in `node_ids`.
        partner_idx: Shape ``(P,)`` int32 row -> partner position in `node_ids`.
        anchor_offsets: Shape ``(len(node_ids) + 1,)`` int64 CSR row starts;
            anchor ``i``'s rows are ``[anchor_offsets[i], anchor_offsets[i+1])``.
        is_near: Shape ``(P,)`` uint8, 1 for a <=2-hop partner, 0 for random.
        k_near: The requested near-context size (a per-anchor ceiling, not a
            guarantee -- anchors with fewer candidates get fewer rows).
        k_rand: The requested random-context size (same ceiling caveat).
        seed: The seed every anchor's RNG was keyed from.
    """

    anchor_idx: NDArray[np.int32]
    partner_idx: NDArray[np.int32]
    anchor_offsets: NDArray[np.int64]
    is_near: NDArray[np.uint8]
    k_near: int
    k_rand: int
    seed: int


def sample_context_sets(
    g_fit: nx.Graph,
    node_ids: Sequence[str],
    *,
    seed: int,
    k_near: int,
    k_rand: int,
) -> ContextSets:
    """Sample each anchor's near (<=2-hop) and random context sets over `node_ids`.

    Args:
        g_fit: The loopless V_fit training topology (or an oracle truth graph
            built from it). Every id in `node_ids` must be a node of `g_fit`.
        node_ids: The sorted V_fit universe; row order fixes both anchor
            identity (`node_ids[i]` is anchor ``i``) and the partner index
            space every returned array indexes into.
        seed: Deterministic seed for every anchor's RNG (see module docstring).
        k_near: Ceiling on near-context rows per anchor; an anchor with fewer
            <=2-hop candidates gets all of them, never padded.
        k_rand: Ceiling on random-context rows per anchor, drawn from
            `node_ids` minus the anchor and its selected near partners. Capped
            at the available pool the same way `k_near` is, for parity on
            small graphs (real V_fit universes are always far larger than
            `k_rand`, so this ceiling is the only case that fires in practice).

    Returns:
        The `ContextSets` CSR-style pair-row arrays.

    Raises:
        ValueError: If `seed`/`k_near`/`k_rand` are negative, `node_ids` is
            not sorted and duplicate-free, or `node_ids` contains a node
            absent from `g_fit`.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if k_near < 0 or k_rand < 0:
        raise ValueError("k_near and k_rand must be non-negative")
    ordered = list(node_ids)
    if ordered != sorted(ordered):
        raise ValueError("node_ids must be the sorted V_fit universe")
    if len(set(ordered)) != len(ordered):
        raise ValueError("node_ids must not contain duplicates")
    missing = [node_id for node_id in ordered if node_id not in g_fit]
    if missing:
        raise ValueError(f"node_ids contains node(s) absent from g_fit: {missing[:5]}")

    node_to_pos = {node_id: position for position, node_id in enumerate(ordered)}
    n_nodes = len(ordered)

    anchor_rows: list[int] = []
    partner_rows: list[int] = []
    near_flags: list[int] = []
    offsets = np.empty(n_nodes + 1, dtype=np.int64)
    offsets[0] = 0

    for position, anchor in enumerate(ordered):
        rng = _anchor_rng(anchor, seed=seed)
        near_positions = _sample_near_positions(g_fit, anchor, node_to_pos, rng, k_near=k_near)
        excluded = {position, *near_positions}
        rand_positions = _sample_random_positions(n_nodes, excluded, rng, k_rand=k_rand)

        group = [*near_positions, *rand_positions]
        anchor_rows.extend([position] * len(group))
        partner_rows.extend(group)
        near_flags.extend([1] * len(near_positions))
        near_flags.extend([0] * len(rand_positions))
        offsets[position + 1] = offsets[position] + len(group)

    return ContextSets(
        anchor_idx=np.asarray(anchor_rows, dtype=np.int32),
        partner_idx=np.asarray(partner_rows, dtype=np.int32),
        anchor_offsets=offsets,
        is_near=np.asarray(near_flags, dtype=np.uint8),
        k_near=k_near,
        k_rand=k_rand,
        seed=seed,
    )


def _sample_near_positions(
    g_fit: nx.Graph,
    anchor: str,
    node_to_pos: dict[str, int],
    rng: np.random.Generator,
    *,
    k_near: int,
) -> list[int]:
    """Return up to `k_near` <=2-hop partner positions, sampled without replacement."""
    hop_lengths = nx.single_source_shortest_path_length(g_fit, anchor, cutoff=_HOP_CUTOFF)
    candidates = sorted(node for node in hop_lengths if node != anchor)
    n_selected = min(k_near, len(candidates))
    if n_selected == 0:
        return []
    picks = rng.choice(len(candidates), size=n_selected, replace=False)
    return [node_to_pos[candidates[int(pick)]] for pick in picks]


def _sample_random_positions(
    n_nodes: int,
    excluded: set[int],
    rng: np.random.Generator,
    *,
    k_rand: int,
) -> list[int]:
    """Return up to `k_rand` uniformly random positions outside `excluded`.

    Rejection sampling avoids materializing the ``node_ids``-minus-excluded
    pool per anchor, which would cost O(n_nodes) per anchor (O(n_nodes^2)
    total) for a real V_fit-scale universe.
    """
    n_selected = min(k_rand, n_nodes - len(excluded))
    if n_selected <= 0:
        return []
    seen = set(excluded)
    selected: list[int] = []
    while len(selected) < n_selected:
        draw = rng.integers(0, n_nodes, size=(n_selected - len(selected)) * 2 + 4)
        for value in draw.tolist():
            if value not in seen:
                seen.add(value)
                selected.append(value)
                if len(selected) == n_selected:
                    break
    return selected


__all__ = ["ContextSets", "sample_context_sets"]
