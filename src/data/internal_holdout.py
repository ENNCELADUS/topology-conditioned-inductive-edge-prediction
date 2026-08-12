"""Current E2E internal topology holdout (spec §§9.3 and 13.19).

The holdout is derived from the full loopless training-interaction graph. ``V_hold`` is the
union of the two historical BFS draws and is the single validation universe for
the formal training plan.  This module also materializes the
complete, non-self pair/label universe over ``V_hold`` so its exact contents can
be pinned before binding.
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
    """Counts of training interactions crossing each unordered node-partition pair."""

    training_interactions: dict[str, int]


@dataclass(frozen=True)
class OverlapProof:
    """Explicit overlap counts published in the run access audit."""

    node: dict[str, int]
    label_edge: dict[str, int]

    @property
    def all_zero(self) -> bool:
        """Return whether every recorded node and label-edge overlap is zero."""
        return not any((*self.node.values(), *self.label_edge.values()))


@dataclass(frozen=True)
class InternalHoldoutPartition:
    """Training nodes and the single isolated internal topology holdout."""

    v_fit: frozenset[str]
    v_hold: frozenset[str]
    holdout_draws: tuple[frozenset[str], frozenset[str]]
    training_interactions_fit: frozenset[Pair]
    topology_fit: frozenset[Pair]
    topology_hold: frozenset[Pair]
    hold_manifest: PairLabelManifest
    quarantine_counts: QuarantineCounts
    overlap_proof: OverlapProof

    def build_g_fit(self) -> nx.Graph:
        """Return the loopless induced training topology, including isolates."""
        graph = nx.Graph()
        graph.add_nodes_from(self.v_fit)
        graph.add_edges_from(self.topology_fit)
        return graph

    def build_g_hold(self) -> nx.Graph:
        """Return the loopless induced holdout gold graph, including isolates."""
        graph = nx.Graph()
        graph.add_nodes_from(self.v_hold)
        graph.add_edges_from(self.topology_hold)
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
    training_interactions: Iterable[Pair],
    *,
    holdout_size: int = 256,
) -> InternalHoldoutPartition:
    """Derive the deterministic ``V_hold`` and ``V_fit`` partitions.

    ``V_hold`` is the union of two hashed-frontier BFS prefixes of the largest
    loopless training-topology component, drawn with the historical ``g5-v2-qual|`` and
    ``g5-v2-select|`` salts; the first draw is removed before the second is
    taken.  Both draws are kept verbatim so that ``V_fit`` — and therefore every
    feature-stats, grounding and pack digest keyed on it. ``holdout_size`` is the size of
    *each* draw, so ``V_hold`` holds twice that many nodes. The shared training
    interactions are then restricted to ``V_fit``; their loopless projection
    induced on ``V_hold`` — including the edges crossing between the two draws —
    become the evaluation-only topology labels.
    """
    if holdout_size <= 0:
        raise ValueError("holdout_size must be positive")

    node_set = frozenset(train_nodes)
    interactions = _canonical_edge_set(training_interactions, node_set, drop_loops=False)
    topology = frozenset((u, v) for u, v in interactions if u != v)
    graph = nx.Graph()
    graph.add_nodes_from(node_set)
    graph.add_edges_from(topology)

    first_draw = frozenset(_holdout_bfs(graph, holdout_size, "g5-v2-qual|"))
    remaining = graph.subgraph(node_set - first_draw).copy()
    second_draw = frozenset(_holdout_bfs(remaining, holdout_size, "g5-v2-select|"))
    v_hold = first_draw | second_draw
    # Asserted, not assumed: the union must not shrink, or V_fit would grow and
    # every digest keyed on it would move.
    if len(v_hold) != 2 * holdout_size:
        raise AssertionError(
            f"V_hold must union two disjoint {holdout_size}-node draws, got {len(v_hold)} nodes"
        )
    v_fit = node_set - v_hold
    if v_fit != node_set - first_draw - second_draw:
        raise AssertionError("V_fit drifted from the two-draw complement")

    training_interactions_fit = frozenset(
        (u, v) for u, v in interactions if u in v_fit and v in v_fit
    )
    # Topology is the loopless projection of the same interactions, by construction.
    topology_fit = frozenset((u, v) for u, v in training_interactions_fit if u != v)
    topology_hold = frozenset((u, v) for u, v in topology if u in v_hold and v in v_hold)

    partitions = {"fit": v_fit, "hold": v_hold}
    quarantine = QuarantineCounts(
        training_interactions=_cross_partition_counts(interactions, partitions),
    )
    proof = OverlapProof(
        node=_pairwise_overlap_counts(partitions),
        label_edge=_pairwise_overlap_counts({"fit": topology_fit, "hold": topology_hold}),
    )
    if not proof.all_zero:  # defensive invariant: disjoint node-induced labels
        raise AssertionError("internal holdout overlap proof is non-zero")

    return InternalHoldoutPartition(
        v_fit=v_fit,
        v_hold=v_hold,
        holdout_draws=(first_draw, second_draw),
        training_interactions_fit=training_interactions_fit,
        topology_fit=topology_fit,
        topology_hold=topology_hold,
        hold_manifest=build_pair_label_manifest(v_hold, topology_hold),
        quarantine_counts=quarantine,
        overlap_proof=proof,
    )


def _holdout_bfs(graph: nx.Graph, size: int, prefix: str) -> tuple[str, ...]:
    """Return a deterministic hashed-frontier BFS prefix of the largest component."""
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    if not components:
        raise ValueError("training topology has no nodes")
    component = min(components, key=lambda nodes: (-len(nodes), nodes))
    if len(component) < size:
        raise ValueError(
            f"largest remaining training-topology component has {len(component)} nodes; need {size}"
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
    edges: Iterable[Pair], partitions: Mapping[str, frozenset[str]]
) -> dict[str, int]:
    owner = {node: name for name, nodes in partitions.items() for node in nodes}
    counts = {
        f"{left}__{right}": 0 for left, right in itertools.combinations(sorted(partitions), 2)
    }
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
