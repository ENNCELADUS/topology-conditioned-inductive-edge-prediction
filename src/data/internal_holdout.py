"""Current E2E internal topology holdouts (spec §§9.3 and 13.19).

The holdouts are derived solely from the seeded message graph.  This module also
materializes the complete, non-self pair/label universes used by qualification
and checkpoint selection so their exact contents can be pinned before binding.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import networkx as nx

from src.data.artifacts import canonical_pair

Pair = tuple[str, str]


@dataclass(frozen=True)
class PairLabelManifest:
    """A complete non-self pair universe and its canonical content digest."""

    nodes: tuple[str, ...]
    positive_edges: tuple[Pair, ...]
    pairs: tuple[Pair, ...]
    labels: tuple[int, ...]
    nodes_sha256: str
    positive_edges_sha256: str
    pair_labels_sha256: str

    @property
    def positive_count(self) -> int:
        """Return the number of positive rows in the complete universe."""
        return sum(self.labels)

    @property
    def prevalence(self) -> float:
        """Return positive prevalence, or zero for a universe with no pairs."""
        return self.positive_count / len(self.pairs) if self.pairs else 0.0


@dataclass(frozen=True)
class QuarantineCounts:
    """Counts of edges crossing each unordered node-partition pair."""

    message: dict[str, int]
    supervision: dict[str, int]


@dataclass(frozen=True)
class OverlapProof:
    """Explicit overlap counts required by the pre-binding audit."""

    node: dict[str, int]
    label_edge: dict[str, int]

    @property
    def all_zero(self) -> bool:
        """Return whether every recorded node and label-edge overlap is zero."""
        return not any((*self.node.values(), *self.label_edge.values()))


@dataclass(frozen=True)
class InternalHoldoutPartition:
    """Training and two isolated internal topology partitions."""

    v_fit: frozenset[str]
    v_qual: frozenset[str]
    v_select: frozenset[str]
    e_msg_fit: frozenset[Pair]
    e_sup_fit: frozenset[Pair]
    qual_manifest: PairLabelManifest
    select_manifest: PairLabelManifest
    quarantine_counts: QuarantineCounts
    overlap_proof: OverlapProof

    def build_g_fit(self) -> nx.Graph:
        """Return the loopless induced training message graph, including isolates."""
        graph = nx.Graph()
        graph.add_nodes_from(self.v_fit)
        graph.add_edges_from(self.e_msg_fit)
        return graph


def canonical_pair_label_sha256(pairs: Iterable[Pair], labels: Iterable[int]) -> str:
    """Hash canonical pair/label rows sorted independently of input order.

    The canonical byte representation is UTF-8 TSV, one ``u, v, label`` row per
    line with endpoints canonicalized and rows lexicographically sorted.
    """
    rows = []
    for pair, label in zip(pairs, labels, strict=True):
        if label not in (0, 1):
            raise ValueError(f"pair labels must be binary, got {label}")
        u, v = canonical_pair(*pair)
        rows.append((u, v, label))
    rows.sort()
    return _sha256_rows(f"{u}\t{v}\t{label}\n" for u, v, label in rows)


def build_pair_label_manifest(
    nodes: Iterable[str], positive_edges: Iterable[Pair]
) -> PairLabelManifest:
    """Build the complete non-self pair universe labeled by induced positives."""
    sorted_nodes = tuple(sorted(set(nodes)))
    node_set = frozenset(sorted_nodes)
    positives = frozenset(
        canonical_pair(u, v)
        for u, v in positive_edges
        if u != v and u in node_set and v in node_set
    )
    pairs = tuple(itertools.combinations(sorted_nodes, 2))
    labels = tuple(int(pair in positives) for pair in pairs)
    sorted_positives = tuple(sorted(positives))
    return PairLabelManifest(
        nodes=sorted_nodes,
        positive_edges=sorted_positives,
        pairs=pairs,
        labels=labels,
        nodes_sha256=_sha256_rows(f"{node}\n" for node in sorted_nodes),
        positive_edges_sha256=_sha256_rows(f"{u}\t{v}\n" for u, v in sorted_positives),
        pair_labels_sha256=canonical_pair_label_sha256(pairs, labels),
    )


def derive_internal_holdout(
    train_nodes: Iterable[str],
    e_msg: Iterable[Pair],
    e_sup: Iterable[Pair],
    *,
    holdout_size: int = 256,
) -> InternalHoldoutPartition:
    """Derive deterministic ``V_qual``, ``V_select``, and ``V_fit`` partitions.

    ``V_qual`` is a hashed-frontier BFS prefix of the largest loopless message
    component.  It is removed before deriving ``V_select`` by the same rule.
    Training message and supervision edges are then restricted to ``V_fit``.
    """
    if holdout_size <= 0:
        raise ValueError("holdout_size must be positive")

    node_set = frozenset(train_nodes)
    message = _canonical_edge_set(e_msg, node_set, drop_loops=True)
    supervision = _canonical_edge_set(e_sup, node_set, drop_loops=False)
    graph = nx.Graph()
    graph.add_nodes_from(node_set)
    graph.add_edges_from(message)

    v_qual = frozenset(_holdout_bfs(graph, holdout_size, "g5-v2-qual|"))
    remaining = graph.subgraph(node_set - v_qual).copy()
    v_select = frozenset(_holdout_bfs(remaining, holdout_size, "g5-v2-select|"))
    v_fit = node_set - v_qual - v_select

    e_msg_fit = frozenset((u, v) for u, v in message if u in v_fit and v in v_fit)
    e_sup_fit = frozenset((u, v) for u, v in supervision if u in v_fit and v in v_fit)
    e_msg_qual = frozenset((u, v) for u, v in message if u in v_qual and v in v_qual)
    e_msg_select = frozenset((u, v) for u, v in message if u in v_select and v in v_select)

    partitions = {"fit": v_fit, "qual": v_qual, "select": v_select}
    quarantine = QuarantineCounts(
        message=_cross_partition_counts(message, partitions),
        supervision=_cross_partition_counts(supervision, partitions),
    )
    proof = OverlapProof(
        node=_pairwise_overlap_counts(partitions),
        label_edge=_pairwise_overlap_counts(
            {"fit": e_msg_fit, "qual": e_msg_qual, "select": e_msg_select}
        ),
    )
    if not proof.all_zero:  # defensive invariant: disjoint node-induced labels
        raise AssertionError("internal holdout overlap proof is non-zero")

    return InternalHoldoutPartition(
        v_fit=v_fit,
        v_qual=v_qual,
        v_select=v_select,
        e_msg_fit=e_msg_fit,
        e_sup_fit=e_sup_fit,
        qual_manifest=build_pair_label_manifest(v_qual, e_msg_qual),
        select_manifest=build_pair_label_manifest(v_select, e_msg_select),
        quarantine_counts=quarantine,
        overlap_proof=proof,
    )


def _holdout_bfs(graph: nx.Graph, size: int, prefix: str) -> tuple[str, ...]:
    """Return a deterministic hashed-frontier BFS prefix of the largest component."""
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    if not components:
        raise ValueError("message graph has no nodes")
    component = min(components, key=lambda nodes: (-len(nodes), nodes))
    if len(component) < size:
        raise ValueError(
            f"largest remaining message component has {len(component)} nodes; need {size}"
        )

    def order_key(node: str) -> tuple[bytes, str]:
        return hashlib.sha256(f"{prefix}{node}".encode()).digest(), node

    allowed = frozenset(component)
    seed = min(component, key=order_key)
    visited = {seed}
    ordered = [seed]
    frontier = [seed]
    while frontier and len(ordered) < size:
        next_frontier = {
            neighbor
            for node in frontier
            for neighbor in graph.neighbors(node)
            if neighbor in allowed and neighbor not in visited
        }
        frontier = sorted(next_frontier, key=order_key)
        visited.update(frontier)
        ordered.extend(frontier[: size - len(ordered)])
    if len(ordered) != size:
        raise AssertionError("connected-component BFS ended before reaching holdout size")
    return tuple(ordered)


def _canonical_edge_set(
    edges: Iterable[Pair], nodes: frozenset[str], *, drop_loops: bool
) -> frozenset[Pair]:
    result: set[Pair] = set()
    for u, v in edges:
        if u not in nodes or v not in nodes:
            raise ValueError(f"edge endpoint outside operative train nodes: {(u, v)!r}")
        if drop_loops and u == v:
            continue
        result.add(canonical_pair(u, v))
    return frozenset(result)


def _cross_partition_counts(
    edges: Iterable[Pair], partitions: dict[str, frozenset[str]]
) -> dict[str, int]:
    owner = {node: name for name, nodes in partitions.items() for node in nodes}
    counts = {"fit__qual": 0, "fit__select": 0, "qual__select": 0}
    for u, v in edges:
        left, right = owner[u], owner[v]
        if left == right:
            continue
        key = "__".join(sorted((left, right)))
        counts[key] += 1
    return counts


def _pairwise_overlap_counts(parts: Mapping[str, AbstractSet[object]]) -> dict[str, int]:
    names = tuple(sorted(parts))
    return {
        f"{left}__{right}": len(parts[left] & parts[right])
        for left, right in itertools.combinations(names, 2)
    }


def _sha256_rows(rows: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "InternalHoldoutPartition",
    "OverlapProof",
    "PairLabelManifest",
    "QuarantineCounts",
    "build_pair_label_manifest",
    "canonical_pair_label_sha256",
    "derive_internal_holdout",
]
