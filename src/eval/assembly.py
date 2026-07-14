"""Assemble a predicted graph from scored pairs and sweep the assembly threshold.

No dependence on `src.data` or `torch` — operates on plain pair lists, numpy
probability arrays, and `networkx.Graph`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph, strip_self_loops


def assemble_graph(
    pairs: Sequence[tuple[str, str]],
    probs: np.ndarray,
    *,
    threshold: float,
    nodes: Iterable[str],
) -> nx.Graph:
    """Assemble a predicted graph from scored node pairs.

    Args:
        pairs: Node pairs `(u, v)` in the same order as `probs`.
        probs: Predicted edge probabilities, same length as `pairs`.
        threshold: A pair is realized as an edge iff its probability is `>= threshold`.
        nodes: The full node universe; every node is present in the returned graph
            (isolated nodes are fine) regardless of whether it appears in `pairs`.

    Returns:
        A `networkx.Graph` with all of `nodes` present. A pair `(u, u)` that clears
        the threshold becomes a self-loop.
    """
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for (u, v), p in zip(pairs, probs, strict=True):
        if p >= threshold:
            g.add_edge(u, v)
    return g


def density_matched_threshold(probs: np.ndarray, target_edges: int) -> float:
    """Find the smallest threshold `t` such that `#(probs >= t) <= target_edges`.

    Args:
        probs: 1-D array of predicted probabilities.
        target_edges: The maximum number of pairs allowed to clear the threshold.

    Returns:
        The threshold value. Since `probs >= t` only changes count at the observed
        distinct values, ties are resolved atomically: a whole tie-group is either
        fully included or fully excluded. The returned threshold is the smallest
        observed probability value whose cumulative count (from the top) is still
        `<= target_edges`.

        Edge cases:
        - `target_edges <= 0`: no data value can satisfy the constraint (since even
          one edge would exceed a limit of 0); the returned threshold is strictly
          above `max(probs)` (via `np.nextafter`), which yields 0 realized edges.
        - `target_edges >= len(probs)`: the returned threshold is `min(probs)`,
          which includes everyone.
        - If even the top tie-group's count exceeds `target_edges` (e.g. more ties
          at the maximum value than `target_edges` allows), the same
          just-above-max fallback is used, yielding 0 realized edges even though
          `target_edges > 0` — this is an unavoidable consequence of atomic ties.

    Raises:
        ValueError: If `probs` is empty.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError("probs must be non-empty")
    if target_edges <= 0:
        return float(np.nextafter(np.max(probs), np.inf))
    n = probs.size
    if target_edges >= n:
        return float(np.min(probs))

    vals, counts = np.unique(probs, return_counts=True)
    vals_desc = vals[::-1]
    counts_desc = counts[::-1]

    cum = 0
    chosen: float | None = None
    for v, c in zip(vals_desc, counts_desc, strict=True):
        if cum + int(c) <= target_edges:
            cum += int(c)
            chosen = float(v)
        else:
            break

    if chosen is None:
        return float(np.nextafter(np.max(probs), np.inf))
    return chosen


@dataclass(frozen=True)
class SweepPoint:
    """One point on a threshold sweep of the assembled-graph evaluation.

    Attributes:
        threshold: The assembly threshold used.
        recall: Fraction of `g_ref`'s (self-loop-stripped) canonical edges that are
            also present in the assembled graph at this threshold.
        graph_similarity: Official per-subgraph GS macro mean.
        relative_density: Official per-subgraph RD macro mean.
        mmd_ratio: Statistic -> reference-normalized MMD ratio at this threshold.
    """

    threshold: float
    recall: float
    graph_similarity: float
    relative_density: float
    mmd_ratio: dict[str, float]


def threshold_sweep(
    pairs: Sequence[tuple[str, str]],
    probs: np.ndarray,
    *,
    thresholds: Sequence[float],
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> list[SweepPoint]:
    """Sweep the assembly threshold and report recall/density/MMD at each point.

    The node universe for assembly is taken to be `g_ref.nodes()` (the reference
    graph defines the full candidate node set that `pairs` is drawn from).

    Args:
        pairs: Candidate node pairs to score.
        probs: Predicted probabilities for `pairs`.
        thresholds: Threshold values to sweep over.
        g_ref: The held-out reference graph.
        buckets: Bucket size -> list of node sets, passed through to
            `evaluate_assembled_graph`.
        config: Shared MMD/descriptor configuration.

    Returns:
        One `SweepPoint` per threshold, in the same order as `thresholds`.
    """
    ref_simple = strip_self_loops(g_ref)
    ref_edges = {frozenset(e) for e in ref_simple.edges()}
    n_ref_edges = len(ref_edges)
    nodes = list(g_ref.nodes())

    points: list[SweepPoint] = []
    for t in thresholds:
        g_pred = assemble_graph(pairs, probs, threshold=t, nodes=nodes)
        report = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
        pred_simple = strip_self_loops(g_pred)
        pred_edges = {frozenset(e) for e in pred_simple.edges()}
        n_overlap = len(ref_edges & pred_edges)
        recall = (n_overlap / n_ref_edges) if n_ref_edges > 0 else 0.0
        points.append(
            SweepPoint(
                threshold=float(t),
                recall=recall,
                graph_similarity=report.graph_similarity,
                relative_density=report.relative_density,
                mmd_ratio=dict(report.mmd_ratio),
            )
        )
    return points
