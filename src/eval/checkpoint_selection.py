"""Checkpoint selection by equal mean rank over five V_val quality metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

SELECTION_RULE = "geometric_rd_five_rank_v1"


@dataclass(frozen=True)
class TopologyValidationMetrics:
    """The five topology metrics of one epoch's V_val bucket evaluation.

    Attributes:
        gs: BFS-macro edge-set Dice similarity across the bucket bank.
        rd: BFS-macro relative density at the sampled-only fixed threshold.
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
    """Rank AUPRC, GS and three MMD ratios equally; RD is reporting-only.

    Exact metric ties receive average ranks. Mean-rank ties prefer higher GS,
    lower geo-MMD, then earlier epoch. Only an empty input returns None;
    invalid numerical inputs still fail closed.
    """
    if not candidates:
        return None
    raw = np.array(
        [
            [
                candidate.auprc,
                candidate.topology.gs,
                candidate.topology.rd,
                candidate.topology.degree_mmd,
                candidate.topology.clustering_mmd,
                candidate.topology.spectral_mmd,
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("non-finite checkpoint-selection metric")
    if np.any(raw[:, 3:] < 0):
        raise ValueError("checkpoint-selection MMD ratios must be non-negative")
    if np.any(raw[:, 2] < 0):
        raise ValueError("checkpoint-selection RD must be non-negative")
    columns = np.column_stack([-raw[:, 0], -raw[:, 1], raw[:, 3], raw[:, 4], raw[:, 5]])
    ranks = np.stack(
        [rankdata(columns[:, criterion], method="average") for criterion in range(columns.shape[1])]
    )
    mean_rank = ranks.mean(axis=0)
    best = min(
        range(len(candidates)),
        key=lambda index: (
            float(mean_rank[index]),
            -candidates[index].topology.gs,
            float(np.prod(raw[index, 3:]) ** (1.0 / 3.0)),
            candidates[index].epoch,
        ),
    )
    return candidates[best]


__all__ = [
    "CheckpointCandidate",
    "TopologyValidationMetrics",
    "select_checkpoint",
]
