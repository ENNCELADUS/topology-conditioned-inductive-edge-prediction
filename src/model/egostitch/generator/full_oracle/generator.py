"""Exact full-neighborhood oracle graphs for diagnostic experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

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
        self._context_to_graph_row: NDArray[np.int64] | None = None
        self._csr_indptr: NDArray[np.int64] | None = None
        self._csr_indices: NDArray[np.int64] | None = None

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
        graph_nodes = tuple(sorted(self._graph.nodes))
        graph_row = {node_id: row for row, node_id in enumerate(graph_nodes)}
        neighbor_rows = tuple(
            np.asarray(
                sorted(graph_row[node] for node in self._graph.neighbors(node_id)),
                dtype=np.int64,
            )
            for node_id in graph_nodes
        )
        degrees = np.fromiter((len(rows) for rows in neighbor_rows), dtype=np.int64)
        indptr = np.empty(len(graph_nodes) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(degrees, out=indptr[1:])
        self._context_to_graph_row = np.asarray(
            [graph_row[node_id] for node_id in ordered], dtype=np.int64
        )
        self._csr_indptr = indptr
        self._csr_indices = (
            np.concatenate(neighbor_rows) if int(indptr[-1]) else np.empty(0, dtype=np.int64)
        )

    def candidate_node_counts(
        self, node_rows_a: torch.Tensor, node_rows_b: torch.Tensor
    ) -> torch.Tensor:
        """Count the exact oracle candidate nodes for each context-row pair."""
        if self._node_ids is None or self._neighbors is None:
            raise RuntimeError("full oracle context is not installed")
        if node_rows_a.shape != node_rows_b.shape or node_rows_a.ndim != 1:
            raise ValueError("node row tensors must be matching one-dimensional tensors")
        rows_a = node_rows_a.detach().to(device="cpu", dtype=torch.int64).tolist()
        rows_b = node_rows_b.detach().to(device="cpu", dtype=torch.int64).tolist()
        if any(row < 0 or row >= len(self._node_ids) for row in (*rows_a, *rows_b)):
            raise ValueError("context-local node rows are out of range")
        counts = []
        for row_a, row_b in zip(rows_a, rows_b, strict=True):
            node_a = self._node_ids[row_a]
            node_b = self._node_ids[row_b]
            candidates = self._neighbors[node_a] | self._neighbors[node_b] | {node_a, node_b}
            counts.append(len(candidates))
        return torch.tensor(counts, device=node_rows_a.device, dtype=torch.int64)

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
        if (
            self._graph is None
            or self._neighbors is None
            or self._context_to_graph_row is None
            or self._csr_indptr is None
            or self._csr_indices is None
        ):
            raise RuntimeError("full oracle context is not installed")
        context_to_graph_row = self._context_to_graph_row
        indptr = self._csr_indptr
        csr_indices = self._csr_indices
        batch_size = int(state_a.projected_x.shape[0])
        if state_b.projected_x.shape != state_a.projected_x.shape:
            raise ValueError("state_a and state_b must have matching row-identity shapes")
        if is_self.shape != (batch_size,) or is_self.dtype != torch.bool:
            raise ValueError(
                f"is_self must be shape {(batch_size,)} and bool, "
                f"got {tuple(is_self.shape)} / {is_self.dtype}"
            )

        device = state_a.projected_x.device
        row_triples = (
            torch.stack(
                (
                    state_a.projected_x[:, 0],
                    state_b.projected_x[:, 0].to(device=device),
                    is_self.to(device=device, dtype=torch.int64),
                ),
                dim=1,
            )
            .to(device="cpu", dtype=torch.int64)
            .numpy()
        )
        src_context_rows = row_triples[:, 0]
        dst_context_rows = row_triples[:, 1]
        self_flags = row_triples[:, 2].astype(np.bool_, copy=False)
        invalid_context = (
            (src_context_rows < 0)
            | (src_context_rows >= len(context_to_graph_row))
            | (dst_context_rows < 0)
            | (dst_context_rows >= len(context_to_graph_row))
        )
        if bool(invalid_context.any()):
            missing = row_triples[invalid_context, :2].reshape(-1)[:5].tolist()
            raise ValueError(f"context-local node rows are missing: {missing}")
        if not np.array_equal(self_flags, src_context_rows == dst_context_rows):
            raise ValueError("is_self disagrees with endpoint row identities")

        src_graph_rows = context_to_graph_row[src_context_rows]
        dst_graph_rows = context_to_graph_row[dst_context_rows]
        graph_node_count = len(indptr) - 1
        batch_rows = np.arange(batch_size, dtype=np.int64)

        def expand_csr(rows: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
            """Return input-row owners and flattened CSR values without Python row loops."""
            lengths = indptr[rows + 1] - indptr[rows]
            total = int(lengths.sum())
            if total == 0:
                empty = np.empty(0, dtype=np.int64)
                return empty, empty
            owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
            group_starts = np.cumsum(lengths, dtype=np.int64) - lengths
            offsets = np.arange(total, dtype=np.int64) - np.repeat(group_starts, lengths)
            positions = np.repeat(indptr[rows], lengths) + offsets
            return owners, csr_indices[positions]

        src_neighbor_owners, src_neighbor_values = expand_csr(src_graph_rows)
        dst_neighbor_owners, dst_neighbor_values = expand_csr(dst_graph_rows)
        rest_owners = np.concatenate((src_neighbor_owners, dst_neighbor_owners))
        rest_values = np.concatenate((src_neighbor_values, dst_neighbor_values))
        if rest_values.size:
            rest_is_endpoint = (rest_values == src_graph_rows[rest_owners]) | (
                rest_values == dst_graph_rows[rest_owners]
            )
            rest_owners = rest_owners[~rest_is_endpoint]
            rest_values = rest_values[~rest_is_endpoint]
            order = np.lexsort((rest_values, rest_owners))
            rest_owners = rest_owners[order]
            rest_values = rest_values[order]
            unique = np.ones(len(rest_values), dtype=np.bool_)
            unique[1:] = (rest_owners[1:] != rest_owners[:-1]) | (
                rest_values[1:] != rest_values[:-1]
            )
            rest_owners = rest_owners[unique]
            rest_values = rest_values[unique]

        endpoint_counts = np.where(self_flags, 1, 2).astype(np.int64, copy=False)
        rest_counts = np.bincount(rest_owners, minlength=batch_size).astype(np.int64, copy=False)
        node_counts = endpoint_counts + rest_counts
        max_nodes = int(node_counts.max()) if batch_size else 0
        x = torch.zeros((batch_size, max_nodes, 5))
        adj = torch.zeros((batch_size, 1, max_nodes, max_nodes))
        mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)

        nonself_rows = batch_rows[~self_flags]
        endpoint_owners = np.concatenate((batch_rows, nonself_rows))
        endpoint_values = np.concatenate((src_graph_rows, dst_graph_rows[~self_flags]))
        endpoint_locals = np.concatenate(
            (np.zeros(batch_size, dtype=np.int64), np.ones(len(nonself_rows), dtype=np.int64))
        )
        if rest_values.size:
            rest_group_starts = np.cumsum(rest_counts, dtype=np.int64) - rest_counts
            rest_locals = endpoint_counts[rest_owners] + (
                np.arange(len(rest_values), dtype=np.int64)
                - np.repeat(rest_group_starts, rest_counts)
            )
        else:
            rest_locals = np.empty(0, dtype=np.int64)
        node_owners = np.concatenate((endpoint_owners, rest_owners))
        node_values = np.concatenate((endpoint_values, rest_values))
        node_locals = np.concatenate((endpoint_locals, rest_locals))

        node_keys = node_owners * graph_node_count + node_values
        key_order = np.argsort(node_keys)
        sorted_node_keys = node_keys[key_order]
        sorted_node_locals = node_locals[key_order]

        def lookup_locals(
            owners: NDArray[np.int64], values: NDArray[np.int64]
        ) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
            query_keys = owners * graph_node_count + values
            positions = np.searchsorted(sorted_node_keys, query_keys)
            found = positions < len(sorted_node_keys)
            safe_positions = np.minimum(positions, max(len(sorted_node_keys) - 1, 0))
            if len(sorted_node_keys):
                found &= sorted_node_keys[safe_positions] == query_keys
                return found, sorted_node_locals[safe_positions]
            return found, np.empty(len(query_keys), dtype=np.int64)

        owner_tensor = torch.from_numpy(node_owners)
        local_tensor = torch.from_numpy(node_locals)
        mask[owner_tensor, local_tensor] = True
        x[owner_tensor, local_tensor, 4] = 1.0
        batch_tensor = torch.from_numpy(batch_rows)
        x[batch_tensor, 0, 0] = 1.0
        x[batch_tensor, torch.from_numpy(np.where(self_flags, 0, 1)), 1] = 1.0

        for channel, owners, values, partners in (
            (2, src_neighbor_owners, src_neighbor_values, dst_graph_rows),
            (3, dst_neighbor_owners, dst_neighbor_values, src_graph_rows),
        ):
            keep = values != partners[owners]
            channel_owners = owners[keep]
            channel_values = values[keep]
            found, channel_locals = lookup_locals(channel_owners, channel_values)
            if not bool(found.all()):
                raise RuntimeError("full oracle CSR lookup escaped the query graph")
            x[
                torch.from_numpy(channel_owners),
                torch.from_numpy(channel_locals),
                channel,
            ] = 1.0

        edge_node_entries, edge_right_values = expand_csr(node_values)
        if edge_right_values.size:
            edge_owners = node_owners[edge_node_entries]
            edge_left_values = node_values[edge_node_entries]
            found, edge_right_locals = lookup_locals(edge_owners, edge_right_values)
            query_edge = (
                (edge_left_values == src_graph_rows[edge_owners])
                & (edge_right_values == dst_graph_rows[edge_owners])
            ) | (
                (edge_left_values == dst_graph_rows[edge_owners])
                & (edge_right_values == src_graph_rows[edge_owners])
            )
            keep = found & ~query_edge
            edge_count = int(keep.sum())
            if edge_count:
                adj[
                    torch.from_numpy(edge_owners[keep]),
                    torch.zeros(edge_count, dtype=torch.int64),
                    torch.from_numpy(node_locals[edge_node_entries[keep]]),
                    torch.from_numpy(edge_right_locals[keep]),
                ] = 1.0

        return self._finalize_graph(
            x=x,
            adj=adj,
            mask=mask,
            device=device,
            node_owners=node_owners,
            node_locals=node_locals,
            node_values=node_values,
        )

    def _finalize_graph(
        self,
        *,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
        node_owners: NDArray[np.int64],
        node_locals: NDArray[np.int64],
        node_values: NDArray[np.int64],
    ) -> FullEgoGraph:
        """Assemble the emitted graph from `stitch`'s CPU-built layout.

        The layout arrays give every valid slot's batch row (`node_owners`),
        local column (`node_locals`), and truth-graph row (`node_values`).
        This base implementation ignores them and emits the structural
        oracle graph unchanged; `FullEgoFeaturesGenerator` overrides it to
        swap the structural channels for gathered node features while
        reusing every line of the vectorized layout construction above.
        """
        del node_owners, node_locals, node_values
        empty_plan = torch.empty((x.shape[0], 0, 0), device=device)
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
