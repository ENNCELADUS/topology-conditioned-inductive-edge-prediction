"""Checkpoint selection over pairwise AUPRC plus all five topology metrics.

Protocol rule (docs/03-experiment-protocol.md §7.1): pairwise and the five
topology numbers are reported together and never substituted by a single
favorable criterion. Selection follows the same discipline: every candidate
epoch is ranked jointly on AUPRC↑, GS↑, RD→1, and the degree / clustering /
spectral MMD ratios↓, and the best mean rank wins. A lexicographic rule with an
absolute tolerance (the pre-2026-08-14 `select_e2e_checkpoint`) collapses to
one criterion whenever an arm's AUPRC spread is smaller than the tolerance —
which is exactly how kd_d2 published its untrained epoch-1 snapshot.

The five numbers come from the V_val bucket evaluation
(`src.eval.val_topology.val_region_topology_metrics`): density-matched global
assembly over the complete V_val-internal pair universe, then the test
evaluator over the pinned 500-ball bucket bank — BFS-macro GS, BFS-macro RD,
and the three odd/even-floor-normalized MMD ratios.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class TopologyValidationMetrics:
    """The five topology metrics of one epoch's V_val bucket evaluation.

    Attributes:
        gs: BFS-macro edge-set Dice similarity across the bucket bank.
        rd: BFS-macro relative density at the density-matched assembly.
        degree_mmd: Degree-histogram MMD ratio over the odd/even floor.
        clustering_mmd: Clustering-histogram MMD ratio over the odd/even floor.
        spectral_mmd: Laplacian-spectrum MMD ratio over the odd/even floor.
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
]
