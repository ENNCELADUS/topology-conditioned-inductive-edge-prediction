"""Training-only unions of simple L3 paths for the L3-PPI surrogate.

No induced edges are added, and no paths are truncated. The caller supplies
the legal training node universe independently of the graph, so accidentally
passing a graph containing held-out nodes fails before extracting patterns.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import torch
from torch import Tensor


@dataclass(frozen=True)
class L3Pattern:
    """Endpoint slots first, followed by sorted intermediate nodes.

    ``edge_index`` contains both directions, with no self-loops. Self-pairs
    have two separate isolated slots referencing the same feature node ID.
    """

    nodes: tuple[str, ...]
    edge_index: Tensor
    num_paths: int


class L3PatternCache:
    """Exact extraction with an optional bounded LRU of ordered query pairs."""

    def __init__(
        self, graph: nx.Graph, allowed_nodes: set[str], cache_size: int = 100_000
    ) -> None:
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("L3 patterns require a simple undirected graph")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        illegal = set(graph) - allowed_nodes
        if illegal:
            raise ValueError(f"L3 training graph contains illegal nodes: {sorted(illegal)[:5]}")
        self.allowed_nodes = frozenset(allowed_nodes)
        self.cache_size = cache_size
        # Independent loopless adjacency snapshot; later graph mutations cannot
        # change existing or newly extracted patterns within a training run.
        self._neighbors = {
            node: frozenset(neighbor for neighbor in graph[node] if neighbor != node)
            for node in graph
        }
        self._cache: OrderedDict[tuple[str, str], L3Pattern] = OrderedDict()

    def extract(self, a: str, b: str) -> L3Pattern:
        """Return every simple ``a-i-j-b`` path as a deduplicated union graph."""
        if a not in self.allowed_nodes or b not in self.allowed_nodes:
            raise ValueError(f"L3 query endpoints must be legal training nodes: {(a, b)}")
        key = (a, b)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        intermediate: set[str] = set()
        edges: set[tuple[str, str]] = set()
        num_paths = 0
        if a != b:
            # Excluding the opposite endpoint removes the query edge and,
            # together with loopless adjacency, prevents all repeated vertices.
            left = self._neighbors.get(a, frozenset()) - {b}
            right = self._neighbors.get(b, frozenset()) - {a}
            for i in left:
                for j in self._neighbors[i] & right:
                    num_paths += 1
                    intermediate.update((i, j))
                    for source, target in ((a, i), (i, j), (j, b)):
                        edges.add((min(source, target), max(source, target)))

        nodes = (a, b, *sorted(intermediate))
        local = {node: index for index, node in enumerate(nodes)}
        directed = sorted(
            edge
            for source, target in edges
            for edge in ((local[source], local[target]), (local[target], local[source]))
        )
        edge_index = (
            torch.tensor(directed, dtype=torch.long).t().contiguous()
            if directed
            else torch.empty((2, 0), dtype=torch.long)
        )
        pattern = L3Pattern(nodes, edge_index, num_paths)
        if self.cache_size:
            self._cache[key] = pattern
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return pattern


def batch_patterns(
    patterns: list[L3Pattern], features: Mapping[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack disjoint graphs as FP32 features, edges, unit weights, graph IDs.

    A missing intermediate feature is an error, never a reason to omit a node
    or its paths. The caller passes ``len(patterns)`` to the graph readout.
    """
    if not patterns:
        raise ValueError("Cannot batch an empty list of L3 patterns")
    node_ids = [node for pattern in patterns for node in pattern.nodes]
    x = torch.stack([features[node] for node in node_ids]).to(device=device, dtype=torch.float32)
    edge_parts: list[Tensor] = []
    graph_parts: list[Tensor] = []
    offset = 0
    for graph_id, pattern in enumerate(patterns):
        edge_parts.append(pattern.edge_index.to(device=device) + offset)
        graph_parts.append(
            torch.full((len(pattern.nodes),), graph_id, dtype=torch.long, device=device)
        )
        offset += len(pattern.nodes)
    edge_index = torch.cat(edge_parts, dim=1)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    return x, edge_index, edge_weight, torch.cat(graph_parts)
