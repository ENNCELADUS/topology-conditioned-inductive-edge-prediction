"""Ego-net structural targets from ``G_struct`` (spec Sec 6, Sec 9.3, Sec 13.4/13.6).

Everything structural derives from the message partition's ``G_struct`` — never
from ``train_graph.pkl`` (quarantined, spec Sec 9.3). Self-loops are already
absent (`build_g_struct` strips them). Leave-one-out (spec Sec 6) is satisfied
structurally: ``E_sup`` and ``E_msg`` are disjoint by construction
(`derive_partition`), so a supervision pair's endpoint can never appear among
its partner's ``G_struct`` targets — asserted here, not re-implemented.

Hub policy (spec Sec 13.4): when ``|N(u)| > K``, neighbors are stratified by
``deg_G_struct`` log2 bucket; largest-remainder proportional allocation capped
at K; per-target multiplicity label = stratum_size / allocated_count; the
degree NLL keeps the true ``|N(u)|``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray


class EgoTargets(NamedTuple):
    """One node batch of ego-net reconstruction targets.

    Attributes:
        features: Shape ``(B, K, d)`` float32 F0 rows of the selected targets
            (zero-padded past `mask`).
        mult: Shape ``(B, K)`` float32 multiplicity labels.
        adj: Shape ``(B, K, K)`` float32 binary adjacency among the selected
            targets on ``G_struct``.
        mask: Shape ``(B, K)`` bool target validity.
        degree: Shape ``(B,)`` float32 true simple degrees ``|N(u)|``.
        in_pool: Shape ``(B, K)`` bool — target is in the node's grounding pool.
        ego_stats: Shape ``(B, 4)`` float32 real-side ego-stat vectors
            (spec Sec 13.6: degree, local clustering, ego edge count,
            ego density — NetworkX implementations on ``G_struct``).
    """

    features: torch.Tensor
    mult: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor
    degree: torch.Tensor
    in_pool: torch.Tensor
    ego_stats: torch.Tensor


def _deg_bucket(degree: int) -> int:
    """Log2 degree bucket (spec Sec 13.4), aligned with the matching buckets."""
    return min(int(math.floor(math.log2(max(degree, 1)))), 4)


def _largest_remainder_allocation(sizes: list[int], total: int) -> list[int]:
    """Proportional allocation of `total` over strata, capped by stratum size."""
    n = sum(sizes)
    raw = [total * s / n for s in sizes]
    alloc = [min(int(math.floor(r)), s) for r, s in zip(raw, sizes, strict=True)]
    remaining = total - sum(alloc)
    # Distribute the remainder by largest fractional part (ties by index),
    # skipping saturated strata.
    order = sorted(range(len(sizes)), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    while remaining > 0:
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # every stratum saturated (total > n) — cannot happen
            break
    return alloc


class EgoTargetBuilder:
    """Builds `EgoTargets` batches for the node stream.

    Static per-node quantities (neighbor lists, ego stats) are computed once;
    only the hub subsample is re-drawn per call (seeded by the caller's rng).
    """

    def __init__(
        self,
        g_struct: nx.Graph,
        f0_matrix: NDArray[np.float32],
        node_index: Mapping[str, int],
        grounding_pool: Mapping[str, Sequence[str]],
        *,
        slots: int,
    ) -> None:
        """Precompute static per-node target ingredients.

        Args:
            g_struct: The message-partition structural graph (simple; from
                `src.data.partition.build_g_struct`).
            f0_matrix: The F0 mean-pool matrix.
            node_index: Node id -> `f0_matrix` row.
            grounding_pool: Node id -> grounding candidates ``G(u)``.
            slots: ``K`` — the slot count (targets are capped at K).

        Raises:
            ValueError: If a ``G_struct`` node is missing an F0 row.
        """
        missing = [n for n in g_struct.nodes() if n not in node_index]
        if missing:
            raise ValueError(f"nodes missing F0 rows: {sorted(missing)[:5]} ...")
        if nx.number_of_selfloops(g_struct) > 0:
            raise ValueError("G_struct must be self-loop-free (spec Sec 9.3)")
        self._g = g_struct
        self._f0 = f0_matrix
        self._index = dict(node_index)
        self._pool = {node: set(members) for node, members in grounding_pool.items()}
        self._slots = int(slots)
        self._neighbors: dict[str, list[str]] = {
            node: sorted(g_struct.neighbors(node)) for node in g_struct.nodes()
        }
        self._ego_stats: dict[str, tuple[float, float, float, float]] = {}
        self._feature_read_observer: Callable[[Sequence[str]], None] | None = None

    def set_feature_read_observer(self, observer: Callable[[Sequence[str]], None] | None) -> None:
        """Observe the exact target-node feature identities read by ``build``."""
        self._feature_read_observer = observer

    @property
    def graph(self) -> nx.Graph:
        """Return the protocol-clean message graph used to build all targets."""
        return self._g

    def _node_ego_stats(self, node: str) -> tuple[float, float, float, float]:
        """The pinned real-side ego-stat 4-vector (spec Sec 13.6), cached."""
        cached = self._ego_stats.get(node)
        if cached is not None:
            return cached
        neighbors = self._neighbors.get(node, [])
        degree = float(len(neighbors))
        clustering = float(nx.clustering(self._g, node)) if neighbors else 0.0
        ego = self._g.subgraph([node, *neighbors])
        stats = (degree, clustering, float(ego.number_of_edges()), float(nx.density(ego)))
        self._ego_stats[node] = stats
        return stats

    def _select_targets(self, node: str, rng: np.random.Generator) -> tuple[list[str], list[float]]:
        """Select up to K targets and their multiplicity labels (spec Sec 13.4)."""
        neighbors = self._neighbors.get(node, [])
        if len(neighbors) <= self._slots:
            return list(neighbors), [1.0] * len(neighbors)
        strata: dict[int, list[str]] = {}
        for v in neighbors:
            strata.setdefault(_deg_bucket(self._g.degree(v)), []).append(v)
        keys = sorted(strata)
        sizes = [len(strata[k]) for k in keys]
        alloc = _largest_remainder_allocation(sizes, self._slots)
        selected: list[str] = []
        mult: list[float] = []
        for key, count in zip(keys, alloc, strict=True):
            if count == 0:
                continue
            members = strata[key]
            chosen_idx = rng.choice(len(members), size=count, replace=False)
            label = len(members) / count
            for i in sorted(chosen_idx.tolist()):
                selected.append(members[i])
                mult.append(label)
        return selected, mult

    def build(self, node_ids: Sequence[str], rng: np.random.Generator) -> EgoTargets:
        """Build one target batch.

        Args:
            node_ids: The node-stream batch.
            rng: Seeded generator (the hub subsample is its only consumer).

        Returns:
            The `EgoTargets` batch (targets padded to ``K``).
        """
        batch = len(node_ids)
        k = self._slots
        d = self._f0.shape[1]
        features = np.zeros((batch, k, d), dtype=np.float32)
        mult = np.zeros((batch, k), dtype=np.float32)
        adj = np.zeros((batch, k, k), dtype=np.float32)
        mask = np.zeros((batch, k), dtype=bool)
        degree = np.zeros(batch, dtype=np.float32)
        in_pool = np.zeros((batch, k), dtype=bool)
        ego_stats = np.zeros((batch, 4), dtype=np.float32)

        for b, node in enumerate(node_ids):
            selected, labels = self._select_targets(node, rng)
            if self._feature_read_observer is not None:
                self._feature_read_observer(selected)
            degree[b] = float(len(self._neighbors.get(node, [])))
            ego_stats[b] = self._node_ego_stats(node)
            pool = self._pool.get(node, set())
            for t, (v, label) in enumerate(zip(selected, labels, strict=True)):
                features[b, t] = self._f0[self._index[v]]
                mult[b, t] = label
                mask[b, t] = True
                in_pool[b, t] = v in pool
            for t1 in range(len(selected)):
                for t2 in range(t1 + 1, len(selected)):
                    if self._g.has_edge(selected[t1], selected[t2]):
                        adj[b, t1, t2] = 1.0
                        adj[b, t2, t1] = 1.0

        return EgoTargets(
            features=torch.from_numpy(features),
            mult=torch.from_numpy(mult),
            adj=torch.from_numpy(adj),
            mask=torch.from_numpy(mask),
            degree=torch.from_numpy(degree),
            in_pool=torch.from_numpy(in_pool),
            ego_stats=torch.from_numpy(ego_stats),
        )
