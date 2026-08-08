"""Exact full-neighborhood oracle graphs for diagnostic experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import torch

from ...graph import ImaginedGraph
from ..base import NeighborhoodGenerator
from ..egostitch import GeneratorNodeState
from ..imagine import SlotSet


@dataclass(frozen=True)
class FullEgoGraph(ImaginedGraph):
    """One padded batch of exact, query-conditioned induced ego graphs."""

    def swapped(self) -> FullEgoGraph:
        """Exchange source/destination roles while sharing graph structure."""
        channel_order = torch.tensor([1, 0, 3, 2, 4], device=self.x.device)
        return FullEgoGraph(
            x=self.x.index_select(-1, channel_order),
            adj=self.adj,
            mask=self.mask,
            aux=self.aux,
            directed=not self.directed,
        )


class FullOracleGenerator(NeighborhoodGenerator[GeneratorNodeState, object, FullEgoGraph]):
    """Emit the complete true induced graph around each queried endpoint pair."""

    def __init__(self) -> None:
        super().__init__()
        self._graph: nx.Graph[str] | None = None
        self._node_ids: tuple[str, ...] | None = None
        self._neighbors: dict[str, frozenset[str]] | None = None

    def set_oracle_context(self, graph: nx.Graph[str], node_ids: Sequence[str]) -> None:
        """Bind a loopless truth graph and its context-local ordered node ids."""
        if graph.is_directed():
            raise ValueError("full oracle context graph must be undirected")
        if nx.number_of_selfloops(graph) != 0:
            raise ValueError("full oracle context graph must be loopless")

        ordered = tuple(node_ids)
        if len(set(ordered)) != len(ordered):
            raise ValueError("full oracle context node_ids contain duplicates")
        missing_from_graph = sorted(set(ordered) - set(graph.nodes))
        if missing_from_graph:
            raise ValueError(
                "full oracle context row ids are absent from the truth graph: "
                f"{missing_from_graph[:5]}"
            )

        self._graph = graph.copy()
        self._node_ids = ordered
        self._neighbors = {
            node_id: frozenset(self._graph.neighbors(node_id)) for node_id in self._graph.nodes
        }

    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
        *,
        node_rows: torch.Tensor | None = None,
    ) -> GeneratorNodeState:
        """Carry context-local row identity in the shared cacheable state type."""
        del ground
        batch_size = int(x.shape[0])
        if self._graph is None or self._node_ids is None:
            raise RuntimeError("full oracle context is not installed")
        if node_rows is None:
            raise ValueError("FullOracleGenerator.encode_node requires node_rows")
        if node_rows.shape != (batch_size,) or node_rows.dtype != torch.int64:
            raise ValueError(
                "node_rows must be a 1-D int64 tensor with one row per node: "
                f"got {tuple(node_rows.shape)} / {node_rows.dtype}"
            )
        rows = node_rows.to(device=x.device)
        if bool(((rows < 0) | (rows >= len(self._node_ids))).any()):
            missing = rows[(rows < 0) | (rows >= len(self._node_ids))][:5].tolist()
            raise ValueError(f"context-local node rows are missing: {missing}")

        # The cache machinery expects a GeneratorNodeState/SlotSet. The full
        # oracle reads only projected_x (the row identity), so K=1 is enough.
        slots = SlotSet(
            h=torch.zeros((batch_size, 1, 1), device=x.device),
            pi=torch.zeros((batch_size, 1), device=x.device),
            mult=torch.ones((batch_size, 1), device=x.device),
            gate=torch.zeros((batch_size, 1), device=x.device),
            pointer=torch.zeros((batch_size, 1, 1), device=x.device),
            adj=torch.zeros((batch_size, 1, 1), device=x.device),
            adj_logits=torch.zeros((batch_size, 1, 1), device=x.device),
        )
        return GeneratorNodeState(
            slots=slots,
            projected_x=rows.unsqueeze(-1),
            ground_ids=ground_ids,
        )

    def _node_id(self, row: int) -> str:
        if self._node_ids is None:
            raise RuntimeError("full oracle context is not installed")
        if row < 0 or row >= len(self._node_ids):
            raise ValueError(f"context-local node row is missing: {row}")
        return self._node_ids[row]

    def _query_graph(self, src: str, dst: str) -> tuple[list[str], set[tuple[str, str]]]:
        if self._graph is None or self._neighbors is None:
            raise RuntimeError("full oracle context is not installed")

        src_neighbors = set(self._neighbors[src])
        dst_neighbors = set(self._neighbors[dst])
        if src != dst:
            src_neighbors.discard(dst)
            dst_neighbors.discard(src)
            endpoints = [src, dst]
        else:
            endpoints = [src]
        nodes = endpoints + sorted((src_neighbors | dst_neighbors) - set(endpoints))
        node_set = set(nodes)
        edges = {
            (left, right)
            for left, right in self._graph.edges(node_set)
            if left in node_set and right in node_set and {left, right} != {src, dst}
        }
        return nodes, edges

    def stitch(
        self,
        state_a: GeneratorNodeState,
        state_b: GeneratorNodeState,
        is_self: torch.Tensor,
        *,
        perturbation: object | None = None,
    ) -> FullEgoGraph:
        """Build and batch-pad exact induced query graphs on the caller's device."""
        if perturbation is not None:
            raise ValueError("FullOracleGenerator does not support scaffold perturbations")
        if self._graph is None or self._neighbors is None:
            raise RuntimeError("full oracle context is not installed")
        neighbors = self._neighbors
        batch_size = int(state_a.projected_x.shape[0])
        if state_b.projected_x.shape != state_a.projected_x.shape:
            raise ValueError("state_a and state_b must have matching row-identity shapes")
        if is_self.shape != (batch_size,) or is_self.dtype != torch.bool:
            raise ValueError(
                f"is_self must be shape {(batch_size,)} and bool, "
                f"got {tuple(is_self.shape)} / {is_self.dtype}"
            )

        device = state_a.projected_x.device
        row_triples = torch.stack(
            (
                state_a.projected_x[:, 0],
                state_b.projected_x[:, 0].to(device=device),
                is_self.to(device=device, dtype=torch.int64),
            ),
            dim=1,
        ).to(device="cpu", dtype=torch.int64)

        query_graphs: list[tuple[str, str, list[str], set[tuple[str, str]]]] = []
        for src_row, dst_row, self_flag in row_triples.tolist():
            src = self._node_id(src_row)
            dst = self._node_id(dst_row)
            if bool(self_flag) != (src == dst):
                raise ValueError("is_self disagrees with endpoint row identities")
            nodes, edges = self._query_graph(src, dst)
            query_graphs.append((src, dst, nodes, edges))

        max_nodes = max((len(item[2]) for item in query_graphs), default=0)
        x = torch.zeros((batch_size, max_nodes, 5))
        adj = torch.zeros((batch_size, 1, max_nodes, max_nodes))
        mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)

        for batch_row, (src, dst, nodes, edges) in enumerate(query_graphs):
            index = {node_id: offset for offset, node_id in enumerate(nodes)}
            count = len(nodes)
            mask[batch_row, :count] = True
            x[batch_row, :count, 4] = 1.0
            x[batch_row, index[src], 0] = 1.0
            x[batch_row, index[dst], 1] = 1.0
            for node_id in neighbors[src] - {dst}:
                x[batch_row, index[node_id], 2] = 1.0
            for node_id in neighbors[dst] - {src}:
                x[batch_row, index[node_id], 3] = 1.0
            for left, right in edges:
                left_i, right_i = index[left], index[right]
                adj[batch_row, 0, left_i, right_i] = 1.0
                adj[batch_row, 0, right_i, left_i] = 1.0

        empty_plan = torch.empty((batch_size, 0, 0), device=device)
        return FullEgoGraph(
            x=x.to(device=device),
            adj=adj.to(device=device),
            mask=mask.to(device=device),
            aux={"plan": empty_plan, "log_plan": empty_plan},
            directed=True,
        )

    def graph_dims(self) -> tuple[int, int]:
        """Return the five node channels and single truth-edge relation."""
        return 5, 1

    def auxiliary_losses(
        self, graph: FullEgoGraph | None, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return no losses because the oracle graph has no learned generator state."""
        del graph, batch
        return {}
