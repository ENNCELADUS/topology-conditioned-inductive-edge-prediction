"""Message/supervision partition of train-side positives (spec §9.3).

The shipped artifacts do not ship the message/supervision split described in
`docs/06-egostitch-spec.md` §9.3; it is derived at load time, per seed, from the
positives of `train_edges.txt`. `G_struct` is the simple (self-loop-free) graph
over all train nodes built from the message-edge subset only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.data.artifacts import canonical_pair


@dataclass(frozen=True)
class MessageSupervisionPartition:
    """A seeded 80/20 (default) split of train-side positives.

    Attributes:
        e_msg: Message-edge subset (structural context).
        e_sup: Supervision-edge subset (`L_edge` positive targets).
        seed: The seed used to derive this partition.
    """

    e_msg: frozenset[tuple[str, str]]
    e_sup: frozenset[tuple[str, str]]
    seed: int


def derive_partition(
    train_positives: Sequence[tuple[str, str]],
    seed: int,
    msg_fraction: float = 0.8,
) -> MessageSupervisionPartition:
    """Derive a deterministic message/supervision partition of train positives.

    Pinned determinism rule: canonicalize and de-duplicate `train_positives`,
    sort the result, shuffle with `numpy.random.default_rng(seed)`, and take the
    first `floor(msg_fraction * n)` (post-shuffle) as `e_msg`; the rest is
    `e_sup`. The same seed always yields the same partition.

    Args:
        train_positives: Positive `(u, v)` pairs of `train_edges.txt` (any order,
            duplicates and mirrored pairs allowed — they are canonicalized and
            de-duplicated before splitting).
        seed: Seed for `numpy.random.default_rng`.
        msg_fraction: Fraction routed to `e_msg`; defaults to 0.8.

    Returns:
        The `MessageSupervisionPartition` (`e_msg`, `e_sup` disjoint, union ==
        the canonicalized de-duplicated input set).
    """
    unique_sorted = sorted({canonical_pair(u, v) for u, v in train_positives})
    n = len(unique_sorted)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_msg = math.floor(msg_fraction * n)
    msg_indices = permutation[:n_msg]
    sup_indices = permutation[n_msg:]
    e_msg = frozenset(unique_sorted[i] for i in msg_indices)
    e_sup = frozenset(unique_sorted[i] for i in sup_indices)
    return MessageSupervisionPartition(e_msg=e_msg, e_sup=e_sup, seed=seed)


def build_g_struct(
    train_nodes: Iterable[str],
    e_msg: Iterable[tuple[str, str]],
) -> nx.Graph:
    """Build `G_struct`: the simple graph of message edges over all train nodes.

    Args:
        train_nodes: All train-side node ids (isolated nodes are kept).
        e_msg: Message-edge pairs; self-loops (`u == v`) are dropped.

    Returns:
        A simple `networkx.Graph` with nodes = `train_nodes` and edges =
        `e_msg` minus self-loops.
    """
    graph = nx.Graph()
    graph.add_nodes_from(train_nodes)
    graph.add_edges_from((u, v) for u, v in e_msg if u != v)
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
