"""Structural-stream subgraph sampler over the V_val-masked training graph.

Three subgraph kinds (BFS ball, wedge/triangle-seeded ball, two-ball bridge)
share one node budget: ``nodes - background_nodes`` locally expanded nodes plus
``background_nodes`` uniform draws. Every subgraph's randomness comes from one
blake2b-keyed generator per plan position, so a plan is identical on every
DDP rank and independent of world size.
Spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`` section 3.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.distill.context_sampler import _anchor_rng
from src.distill.struct_config import KINDS

_RETRIES = 32
_BRIDGE_MIN_DISTANCE = 3


@dataclass(frozen=True)
class StructSubgraph:
    """One sampled subgraph: kind, ordered node ids, and trailing background count."""

    kind: str
    nodes: tuple[str, ...]
    background: int


@dataclass(frozen=True)
class StructEpochPlan:
    """Every subgraph of one epoch, in plan order."""

    seed: int
    epoch: int
    subgraphs: tuple[StructSubgraph, ...]


class StructSampler:
    """Sample training subgraphs and expose their legal masks, targets and statistics."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        nodes: int,
        background_nodes: int,
        mix: Mapping[str, float],
        v_val: frozenset[str],
        exclude_nodes: frozenset[str],
    ) -> None:
        if nodes < 4 or not 0 <= background_nodes < nodes:
            raise ValueError(
                f"invalid struct sampler budget nodes={nodes} background={background_nodes}"
            )
        if set(mix) != set(KINDS):
            raise ValueError(f"mix must name exactly {list(KINDS)}")
        self.graph = graph
        self.nodes = nodes
        self.background_nodes = background_nodes
        self.local_budget = nodes - background_nodes
        self.v_val = v_val
        self.exclude_nodes = exclude_nodes
        self._kinds = tuple(KINDS)
        self._shares = np.asarray([float(mix[kind]) for kind in self._kinds], dtype=np.float64)
        self._universe: tuple[str, ...] = tuple(
            sorted(node for node in graph.nodes if node not in exclude_nodes)
        )
        if len(self._universe) < nodes:
            raise ValueError(
                f"structural graph has fewer than {nodes} sampleable nodes ({len(self._universe)})"
            )
        self._neighbors: dict[str, tuple[str, ...]] = {
            node: tuple(sorted(n for n in graph.neighbors(node) if n not in exclude_nodes))
            for node in self._universe
        }
        self._degree_ge1 = tuple(n for n in self._universe if len(self._neighbors[n]) >= 1)
        self._degree_ge2 = tuple(n for n in self._universe if len(self._neighbors[n]) >= 2)
        self._edges = tuple((u, v) for u in self._universe for v in self._neighbors[u] if u < v)
        if not self._degree_ge1:
            raise ValueError("structural graph has no edges among sampleable nodes")
        self._positive_count = len(self._edges)

    # ----------------------------------------------------------------- planning

    def plan(self, *, seed: int, epoch: int, count: int) -> StructEpochPlan:
        """Return ``count`` subgraphs for ``(seed, epoch)``; deterministic and rank-free."""
        if count < 0:
            raise ValueError("count must be non-negative")
        subgraphs = tuple(
            self._sample_one(seed=seed, epoch=epoch, position=t) for t in range(count)
        )
        return StructEpochPlan(seed=seed, epoch=epoch, subgraphs=subgraphs)

    def _sample_one(self, *, seed: int, epoch: int, position: int) -> StructSubgraph:
        rng = _anchor_rng(f"struct:{position}", seed=seed, epoch=epoch)
        kind = self._kinds[int(rng.choice(len(self._kinds), p=self._shares))]
        if kind == "bfs":
            local = self._ball([self._choice(rng, self._degree_ge1)], self.local_budget, rng)
        elif kind == "motif":
            local = self._ball(list(self._motif_seed(rng)), self.local_budget, rng)
        else:
            local = self._bridge(rng)
        local = self._fill(local, self.local_budget, rng)
        present = set(local)
        pool = [node for node in self._universe if node not in present]
        background = (
            [pool[int(i)] for i in rng.choice(len(pool), size=self.background_nodes, replace=False)]
            if self.background_nodes
            else []
        )
        return StructSubgraph(
            kind=kind, nodes=tuple(local + background), background=len(background)
        )

    @staticmethod
    def _choice(rng: np.random.Generator, pool: Sequence[str]) -> str:
        return pool[int(rng.integers(len(pool)))]

    def _ball(
        self,
        seeds: list[str],
        budget: int,
        rng: np.random.Generator,
        taken: set[str] | None = None,
    ) -> list[str]:
        """Randomised-neighbour-order BFS from ``seeds`` up to ``budget`` nodes."""
        order: list[str] = []
        seen: set[str] = set(taken or ())
        queue: deque[str] = deque()
        for seed in seeds:
            if seed not in seen:
                seen.add(seed)
                order.append(seed)
                queue.append(seed)
        while queue and len(order) < budget:
            current = queue.popleft()
            neighbours = list(self._neighbors[current])
            rng.shuffle(neighbours)
            for neighbour in neighbours:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
                if len(order) >= budget:
                    break
        return order[:budget]

    def _fill(self, local: list[str], budget: int, rng: np.random.Generator) -> list[str]:
        """Top up a short ball from fresh roots until ``budget`` is met."""
        while len(local) < budget:
            present = set(local)
            pool = [node for node in self._degree_ge1 if node not in present]
            if not pool:
                pool = [node for node in self._universe if node not in present]
            root = self._choice(rng, pool)
            local = local + self._ball([root], budget - len(local), rng, taken=present)
        return local[:budget]

    def _motif_seed(self, rng: np.random.Generator) -> tuple[str, str, str]:
        """An open wedge (centre, leaf, leaf) or a closed triangle, 50/50."""
        if rng.random() < 0.5:
            return self._wedge(rng)
        for _ in range(_RETRIES):
            u, v = self._edges[int(rng.integers(len(self._edges)))]
            common = sorted(set(self._neighbors[u]) & set(self._neighbors[v]))
            if common:
                return u, v, self._choice(rng, common)
        return self._wedge(rng)

    def _wedge(self, rng: np.random.Generator) -> tuple[str, str, str]:
        pool = self._degree_ge2 or self._degree_ge1
        for _ in range(_RETRIES):
            centre = self._choice(rng, pool)
            neighbours = self._neighbors[centre]
            if len(neighbours) < 2:
                continue
            i, j = rng.choice(len(neighbours), size=2, replace=False)
            left, right = neighbours[int(i)], neighbours[int(j)]
            if right not in self._neighbors[left]:
                return centre, left, right
        centre = self._choice(rng, pool)
        neighbours = self._neighbors[centre]
        return centre, neighbours[0], neighbours[-1]

    def _bridge(self, rng: np.random.Generator) -> list[str]:
        """Two balls of ``local_budget // 2`` from roots at distance >= 3 when possible."""
        half = self.local_budget // 2
        first = self._choice(rng, self._degree_ge1)
        second = first
        near = nx.single_source_shortest_path_length(
            self.graph, first, cutoff=_BRIDGE_MIN_DISTANCE - 1
        )
        for _ in range(_RETRIES):
            candidate = self._choice(rng, self._degree_ge1)
            if candidate != first and candidate not in near:
                second = candidate
                break
        if second == first:
            pool = [node for node in self._degree_ge1 if node != first]
            second = self._choice(rng, pool)
        ball_a = self._ball([first], half, rng)
        ball_b = self._ball([second], self.local_budget - len(ball_a), rng, taken=set(ball_a))
        return ball_a + ball_b

    # ----------------------------------------------------------------- tensors

    def legal_mask(self, subgraph: StructSubgraph) -> NDArray[np.float32]:
        """Symmetric 0/1 legal-pair mask with a zero diagonal."""
        nodes = subgraph.nodes
        in_val = np.asarray([node in self.v_val for node in nodes], dtype=bool)
        excluded = np.asarray([node in self.exclude_nodes for node in nodes], dtype=bool)
        mask = ~(in_val[:, None] & in_val[None, :]) & ~excluded[:, None] & ~excluded[None, :]
        np.fill_diagonal(mask, False)
        return np.asarray(mask, dtype=np.float32)

    def adjacency(self, subgraph: StructSubgraph) -> NDArray[np.float32]:
        """Induced loopless adjacency of the structural graph, multiplied by the legal mask."""
        nodes = subgraph.nodes
        index = {node: i for i, node in enumerate(nodes)}
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for i, node in enumerate(nodes):
            for neighbour in self._neighbors.get(node, ()):
                j = index.get(neighbour)
                if j is not None and j != i:
                    adjacency[i, j] = 1.0
        return adjacency * self.legal_mask(subgraph)

    # ----------------------------------------------------------------- telemetry

    def statistics(self, subgraph: StructSubgraph) -> dict[str, float]:
        """Per-subgraph sampler telemetry (spec section 3.4)."""
        mask = self.legal_mask(subgraph)
        adjacency = self.adjacency(subgraph)
        n = len(subgraph.nodes)
        degrees = adjacency.sum(axis=1)
        triangles = float(np.trace(adjacency @ adjacency @ adjacency) / 6.0)
        open_wedges = float((degrees * (degrees - 1.0) / 2.0).sum() - 3.0 * triangles)
        induced = nx.from_numpy_array(adjacency)
        stats = {
            "struct_nodes": float(n),
            "struct_components": float(nx.number_connected_components(induced)),
            "struct_legal_fraction": float(mask.sum() / max(1.0, n * (n - 1))),
            "struct_positive_fraction": float(adjacency.sum() / max(1.0, mask.sum())),
            "struct_triangles": triangles,
            "struct_open_wedges": open_wedges,
            "struct_background": float(subgraph.background),
        }
        for kind in self._kinds:
            stats[f"struct_kind_{kind}"] = float(subgraph.kind == kind)
        return stats

    def coverage(self, plan: StructEpochPlan) -> dict[str, float]:
        """Fraction of sampleable positives seen at least once, and their mean reuse."""
        counts: dict[tuple[str, str], int] = {}
        for subgraph in plan.subgraphs:
            adjacency = self.adjacency(subgraph)
            rows, cols = np.nonzero(np.triu(adjacency, 1))
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
                u, v = subgraph.nodes[i], subgraph.nodes[j]
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        seen = len(counts)
        return {
            "struct_positive_coverage": seen / max(1, self._positive_count),
            "struct_positive_reuse": (sum(counts.values()) / seen) if seen else 0.0,
        }


__all__ = ["StructEpochPlan", "StructSampler", "StructSubgraph"]
