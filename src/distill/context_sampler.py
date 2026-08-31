"""Reference-faithful LLP context-bank sampling for KD2.

Each anchor receives ``rw_step * hops`` ordered random-walk visits followed by
``rw_step * hops * ns_rate`` global draws. Walk visits are never deduplicated;
illegal visits are dropped rather than replaced. Global draws are uniform with
replacement from the anchor's legal feature-bearing node pool.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

DEFAULT_RW_STEP = 3
DEFAULT_HOPS = 2
DEFAULT_NS_RATE = 1
V_VAL_DIAGNOSTIC_SEED = 0x56414C


@dataclass(frozen=True)
class ContextBank:
    """CSR rows for an ordered subset of anchors in a shared node universe."""

    anchor_idx: NDArray[np.int32]
    anchor_offsets: NDArray[np.int64]
    partner_idx: NDArray[np.int32]
    is_near: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate array dtypes and CSR shape invariants."""
        if self.anchor_idx.ndim != 1 or self.anchor_idx.dtype != np.int32:
            raise ValueError("anchor_idx must be a one-dimensional int32 array")
        if self.anchor_offsets.ndim != 1 or self.anchor_offsets.dtype != np.int64:
            raise ValueError("anchor_offsets must be a one-dimensional int64 array")
        if self.partner_idx.ndim != 1 or self.partner_idx.dtype != np.int32:
            raise ValueError("partner_idx must be a one-dimensional int32 array")
        if self.is_near.ndim != 1 or self.is_near.dtype != np.bool_:
            raise ValueError("is_near must be a one-dimensional bool array")
        if len(self.anchor_offsets) != len(self.anchor_idx) + 1:
            raise ValueError("anchor_offsets length must equal anchor count plus one")
        if self.anchor_offsets[0] != 0:
            raise ValueError("anchor_offsets must start at zero")
        if np.any(np.diff(self.anchor_offsets) < 0):
            raise ValueError("anchor_offsets must be nondecreasing")
        if self.anchor_offsets[-1] != len(self.partner_idx):
            raise ValueError("anchor_offsets must end at the context-row count")
        if len(self.is_near) != len(self.partner_idx):
            raise ValueError("is_near length must equal partner_idx length")


def _anchor_rng(node_id: str, *, seed: int, epoch: int) -> np.random.Generator:
    """Return a node-keyed RNG with independent uint64 seed and epoch lanes."""
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    if not 0 <= epoch < 2**64:
        raise ValueError("epoch must fit an unsigned 64-bit integer")
    node_key = int.from_bytes(
        hashlib.blake2b(node_id.encode("utf-8"), digest_size=16).digest(), "little"
    )
    return np.random.default_rng(node_key ^ seed ^ (epoch << 64))


def sample_context_bank(
    truth_graph: nx.Graph,
    *,
    anchor_ids: Sequence[str],
    node_ids: Sequence[str],
    forbidden_internal: frozenset[str],
    seed: int,
    epoch: int,
    rw_step: int = DEFAULT_RW_STEP,
    hops: int = DEFAULT_HOPS,
    ns_rate: int = DEFAULT_NS_RATE,
) -> ContextBank:
    """Sample one epoch's strict-LLP context bank.

    ``truth_graph`` must be the loopless training-side KD truth graph. A walk
    visit is legal only when it is feature-bearing (present in ``node_ids``)
    and does not form a V_val-internal pair. Degree-zero walks stay at their
    current node, matching ``torch_cluster.random_walk`` padding semantics.
    """
    if rw_step < 0 or hops < 0 or ns_rate < 0:
        raise ValueError("rw_step, hops, and ns_rate must be non-negative")
    if any(u == v for u, v in nx.selfloop_edges(truth_graph)):
        raise ValueError("truth_graph must be loopless")

    ordered_nodes = list(node_ids)
    ordered_anchors = list(anchor_ids)
    if ordered_nodes != sorted(ordered_nodes) or len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError("node_ids must be sorted and duplicate-free")
    if len(set(ordered_anchors)) != len(ordered_anchors):
        raise ValueError("anchor_ids must be duplicate-free")

    node_to_idx = {node_id: idx for idx, node_id in enumerate(ordered_nodes)}
    missing_nodes = [node_id for node_id in ordered_nodes if node_id not in truth_graph]
    missing_anchors = [node_id for node_id in ordered_anchors if node_id not in node_to_idx]
    if missing_nodes:
        raise ValueError(f"node_ids contains node(s) absent from truth_graph: {missing_nodes[:5]}")
    if missing_anchors:
        raise ValueError(f"anchor_ids contains node(s) absent from node_ids: {missing_anchors[:5]}")

    p = rw_step * hops
    q = p * ns_rate
    all_positions = np.arange(len(ordered_nodes), dtype=np.int64)
    external_positions = np.asarray(
        [idx for idx, node_id in enumerate(ordered_nodes) if node_id not in forbidden_internal],
        dtype=np.int64,
    )
    if (
        q
        and any(anchor in forbidden_internal for anchor in ordered_anchors)
        and len(external_positions) == 0
    ):
        raise ValueError("V_val anchors have no legal random-context pool")
    if q and ordered_anchors and len(all_positions) == 0:
        raise ValueError("anchors have no legal random-context pool")

    partner_rows: list[int] = []
    near_rows: list[bool] = []
    offsets = np.empty(len(ordered_anchors) + 1, dtype=np.int64)
    offsets[0] = 0
    neighbor_cache: dict[str, tuple[str, ...]] = {}

    for anchor_position, anchor in enumerate(ordered_anchors):
        rng = _anchor_rng(anchor, seed=seed, epoch=epoch)
        for _ in range(rw_step):
            current = anchor
            for _ in range(hops):
                neighbors = neighbor_cache.get(current)
                if neighbors is None:
                    neighbors = tuple(sorted(truth_graph.neighbors(current)))
                    neighbor_cache[current] = neighbors
                if neighbors:
                    current = neighbors[int(rng.integers(len(neighbors)))]
                partner_position = node_to_idx.get(current)
                if partner_position is not None and not (
                    anchor in forbidden_internal and current in forbidden_internal
                ):
                    partner_rows.append(partner_position)
                    near_rows.append(True)

        random_pool = external_positions if anchor in forbidden_internal else all_positions
        if q:
            random_rows = rng.choice(random_pool, size=q, replace=True)
            partner_rows.extend(int(position) for position in random_rows)
            near_rows.extend([False] * q)
        offsets[anchor_position + 1] = len(partner_rows)

    partner_idx = np.asarray(partner_rows, dtype=np.int32)
    if len(partner_idx) and (partner_idx.min() < 0 or partner_idx.max() >= len(ordered_nodes)):
        raise AssertionError("sampler produced an out-of-universe partner index")
    return ContextBank(
        anchor_idx=np.asarray(
            [node_to_idx[node_id] for node_id in ordered_anchors], dtype=np.int32
        ),
        anchor_offsets=offsets,
        partner_idx=partner_idx,
        is_near=np.asarray(near_rows, dtype=np.bool_),
    )


def sample_context_banks(
    truth_graph: nx.Graph,
    *,
    anchor_ids: Sequence[str],
    node_ids: Sequence[str],
    forbidden_internal: frozenset[str],
    seed: int,
    n_banks: int,
    rw_step: int = DEFAULT_RW_STEP,
    hops: int = DEFAULT_HOPS,
    ns_rate: int = DEFAULT_NS_RATE,
) -> tuple[ContextBank, ...]:
    """Sample epoch-indexed banks with one deterministic bank per epoch."""
    if n_banks < 0:
        raise ValueError("n_banks must be non-negative")
    return tuple(
        sample_context_bank(
            truth_graph,
            anchor_ids=anchor_ids,
            node_ids=node_ids,
            forbidden_internal=forbidden_internal,
            seed=seed,
            epoch=epoch,
            rw_step=rw_step,
            hops=hops,
            ns_rate=ns_rate,
        )
        for epoch in range(n_banks)
    )


def sample_v_val_context_bank(
    truth_graph: nx.Graph,
    *,
    v_val: frozenset[str],
    node_ids: Sequence[str],
    rw_step: int = DEFAULT_RW_STEP,
    hops: int = DEFAULT_HOPS,
    ns_rate: int = DEFAULT_NS_RATE,
) -> ContextBank:
    """Build the fixed, diagnostics-only V_val context bank."""
    return sample_context_bank(
        truth_graph,
        anchor_ids=sorted(v_val.intersection(node_ids)),
        node_ids=node_ids,
        forbidden_internal=v_val,
        seed=V_VAL_DIAGNOSTIC_SEED,
        epoch=0,
        rw_step=rw_step,
        hops=hops,
        ns_rate=ns_rate,
    )


__all__ = [
    "DEFAULT_HOPS",
    "DEFAULT_NS_RATE",
    "DEFAULT_RW_STEP",
    "V_VAL_DIAGNOSTIC_SEED",
    "ContextBank",
    "sample_context_bank",
    "sample_context_banks",
    "sample_v_val_context_bank",
]
