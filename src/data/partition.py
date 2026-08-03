"""Shared train-side interactions for topology and classification (spec §9.3)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx

from src.data.artifacts import canonical_pair


@dataclass(frozen=True)
class TrainingInteractions:
    """The canonical positive interactions shared by every training objective.

    Attributes:
        positives: Canonical, de-duplicated train-side positive interactions.
            These are the classification (`L_edge`) positives verbatim — self-pairs
            retained (spec §9.3). Topology objectives take `topology_edges`.
    """

    positives: frozenset[tuple[str, str]]

    @property
    def topology_edges(self) -> frozenset[tuple[str, str]]:
        """Return the loopless projection used by topology objectives."""
        return frozenset((u, v) for u, v in self.positives if u != v)


def derive_training_interactions(
    train_positives: Sequence[tuple[str, str]],
) -> TrainingInteractions:
    """Canonicalize the positive interactions shared by topology and classification.

    Args:
        train_positives: Positive `(u, v)` pairs of `train_edges.txt` (any order,
            duplicates and mirrored pairs allowed — they are canonicalized and
            de-duplicated).

    Returns:
        The single `TrainingInteractions` source consumed by both objectives.
    """
    return TrainingInteractions(
        positives=frozenset(canonical_pair(u, v) for u, v in train_positives)
    )


def build_g_struct(
    train_nodes: Iterable[str],
    training_interactions: Iterable[tuple[str, str]],
) -> nx.Graph:
    """Build `G_struct` from the loopless projection of all training interactions.

    Args:
        train_nodes: All train-side node ids (isolated nodes are kept).
        training_interactions: Shared positive interactions; self-loops are dropped.

    Returns:
        A simple `networkx.Graph` with nodes = `train_nodes` and edges =
        `training_interactions` minus self-loops.
    """
    graph = nx.Graph()
    graph.add_nodes_from(train_nodes)
    graph.add_edges_from((u, v) for u, v in training_interactions if u != v)
    return graph


def strip_self_loops(g: nx.Graph) -> nx.Graph:
    """Return a copy of `g` with self-loops removed; `g` itself is untouched.

    Args:
        g: Input graph.

    Returns:
        A new `networkx.Graph` equal to `g` minus its self-loop edges.
    """
    result = g.copy()
    result.remove_edges_from(list(nx.selfloop_edges(result)))
    return result
