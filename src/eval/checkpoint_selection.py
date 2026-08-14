"""Checkpoint selection over pairwise AUPRC plus all five topology metrics.

Protocol rule (docs/03-experiment-protocol.md §7.1): pairwise and the five
topology numbers are reported together and never substituted by a single
favorable criterion. Selection follows the same discipline: every candidate
epoch is ranked jointly on AUPRC↑, GS↑, RD→1, and the degree / clustering /
spectral MMDs↓, and the best mean rank wins. A lexicographic rule with an
absolute tolerance (the pre-2026-08-14 `select_e2e_checkpoint`) collapses to
one criterion whenever an arm's AUPRC spread is smaller than the tolerance —
which is exactly how kd_d2 published its untrained epoch-1 snapshot.

Validation-side metric computation mirrors the house V_hold convention
(`train_egostitch._validation_clustering_mmd`, now superseded): assemble the
predicted graph at the exact gold edge count with the deterministic
``(-logit, pair)`` ranking, keep self-pairs as-is, and compare raw
single-graph descriptors. RD is the one metric that cannot come from a
density-matched assembly (it would be identically 1), so it is the predicted
positive count at probability 0.5 (logit 0) against the gold edge count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from src.eval.graph_metrics import (
    MMDConfig,
    clustering_histogram,
    compute_graph_similarity,
    degree_histogram,
    laplacian_spectrum_histogram,
    mmd_squared,
)


@dataclass(frozen=True)
class TopologyValidationMetrics:
    """The five topology metrics of one epoch's validation assembly.

    Attributes:
        gs: Edge-set Dice similarity at the gold-edge-count assembly.
        rd: Predicted positives at logit 0 over the gold edge count.
        degree_mmd: Raw single-graph degree-histogram MMD^2.
        clustering_mmd: Raw single-graph clustering-histogram MMD^2.
        spectral_mmd: Raw single-graph Laplacian-spectrum MMD^2.
    """

    gs: float
    rd: float
    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float


@dataclass(frozen=True)
class CheckpointCandidate:
    """One selectable epoch: its AUPRC and topology validation metrics."""

    epoch: int
    auprc: float
    topology: TopologyValidationMetrics


def validation_topology_metrics(
    *,
    pairs: Sequence[tuple[str, str]],
    logits: NDArray[np.floating],
    positive_edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
) -> TopologyValidationMetrics:
    """Compute the five topology metrics from one epoch's validation scores.

    Args:
        pairs: The fixed validation pair list, aligned with `logits`.
        logits: One logit per validation pair (probability 0.5 at logit 0).
        positive_edges: Gold edges over `nodes` (the validation positives).
        nodes: The validation node universe both graphs are built over.

    Returns:
        The `TopologyValidationMetrics`.

    Raises:
        ValueError: On length mismatch, non-finite logits, or an edgeless
            gold graph (all fail-closed states).
    """
    if len(pairs) != len(logits):
        raise ValueError(f"pairs/logits length mismatch: {len(pairs)} != {len(logits)}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("non-finite validation logits")

    gold = nx.Graph()
    gold.add_nodes_from(nodes)
    gold.add_edges_from(positive_edges)
    target_edges = gold.number_of_edges()
    if target_edges == 0:
        raise ValueError("validation universe has no positive edges")

    ranked = sorted(range(len(pairs)), key=lambda index: (-float(logits[index]), pairs[index]))
    predicted = nx.Graph()
    predicted.add_nodes_from(nodes)
    predicted.add_edges_from(pairs[index] for index in ranked[:target_edges])

    config = MMDConfig()
    return TopologyValidationMetrics(
        gs=compute_graph_similarity(predicted, gold),
        rd=float(np.count_nonzero(np.asarray(logits) > 0.0)) / float(target_edges),
        degree_mmd=mmd_squared([degree_histogram(predicted)], [degree_histogram(gold)], config),
        clustering_mmd=mmd_squared(
            [clustering_histogram(predicted)], [clustering_histogram(gold)], config
        ),
        spectral_mmd=mmd_squared(
            [laplacian_spectrum_histogram(predicted)],
            [laplacian_spectrum_histogram(gold)],
            config,
        ),
    )


def select_checkpoint(
    candidates: Sequence[CheckpointCandidate],
) -> CheckpointCandidate | None:
    """Select the checkpoint with the best mean rank over all six criteria.

    Criteria (all ranked with average ranks, then averaged): AUPRC higher,
    GS higher, |RD - 1| lower, and each of the degree / clustering / spectral
    MMDs lower. Ties break on higher AUPRC, then the later epoch (an
    untrained early snapshot never wins a full tie).

    Args:
        candidates: One entry per eligible epoch; empty selects nothing.

    Returns:
        The winning candidate, or `None` for an empty sequence.

    Raises:
        ValueError: If any candidate carries a non-finite metric.
    """
    if not candidates:
        return None
    columns = np.array(
        [
            [
                -candidate.auprc,
                -candidate.topology.gs,
                abs(candidate.topology.rd - 1.0),
                candidate.topology.degree_mmd,
                candidate.topology.clustering_mmd,
                candidate.topology.spectral_mmd,
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(columns)):
        raise ValueError("non-finite checkpoint-selection metric")
    ranks = np.stack(
        [rankdata(columns[:, criterion], method="average") for criterion in range(columns.shape[1])]
    )
    mean_rank = ranks.mean(axis=0)
    best = min(
        range(len(candidates)),
        key=lambda index: (
            float(mean_rank[index]),
            -candidates[index].auprc,
            -candidates[index].epoch,
        ),
    )
    return candidates[best]


__all__ = [
    "CheckpointCandidate",
    "TopologyValidationMetrics",
    "select_checkpoint",
    "validation_topology_metrics",
]
