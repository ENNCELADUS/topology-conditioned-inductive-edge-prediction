"""Selection over pairwise + five-topology validation metrics (B1 fix).

The kd_d2 postmortem (2026-08-14): `select_e2e_checkpoint`'s absolute AUPRC
tolerance went vacuous (whole-range spread < 0.02), fell through to clustering
MMD alone, and published the untrained epoch-1 phase-A snapshot. The corrected
rule ranks every candidate on AUPRC plus all five topology metrics together,
so no single degenerate criterion can select an untrained checkpoint.
"""

from __future__ import annotations

import pytest
from src.eval.checkpoint_selection import (
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)


def _topo(
    gs: float,
    rd: float,
    degree: float,
    clustering: float,
    spectral: float,
) -> TopologyValidationMetrics:
    return TopologyValidationMetrics(
        gs=gs,
        rd=rd,
        degree_mmd=degree,
        clustering_mmd=clustering,
        spectral_mmd=spectral,
    )


class TestSelectCheckpoint:
    def test_empty_returns_none(self) -> None:
        assert select_checkpoint([]) is None

    def test_dominating_candidate_wins(self) -> None:
        worse = CheckpointCandidate(epoch=1, auprc=0.01, topology=_topo(0.10, 0.5, 0.3, 0.3, 0.3))
        better = CheckpointCandidate(epoch=2, auprc=0.20, topology=_topo(0.40, 1.0, 0.1, 0.1, 0.1))
        selected = select_checkpoint([worse, better])
        assert selected is not None and selected.epoch == 2

    def test_kd_d2_untrained_epoch_trap(self) -> None:
        """An untrained snapshot winning clustering/spectral MMD must not win overall."""
        untrained = CheckpointCandidate(
            epoch=1,
            auprc=0.0084,
            topology=_topo(0.05, 0.0, 0.20, 0.1399, 0.20),
        )
        trained = CheckpointCandidate(
            epoch=25,
            auprc=0.0213,
            topology=_topo(0.30, 0.85, 0.10, 0.1860, 0.25),
        )
        selected = select_checkpoint([untrained, trained])
        assert selected is not None and selected.epoch == 25

    def test_three_three_criteria_split_breaks_on_auprc(self) -> None:
        """3-vs-3 criteria tie -> equal mean rank -> higher AUPRC wins."""
        low = CheckpointCandidate(
            epoch=1, auprc=0.0084, topology=_topo(0.30, 1.0, 0.10, 0.10, 0.20)
        )
        high = CheckpointCandidate(
            epoch=25, auprc=0.0213, topology=_topo(0.45, 0.9, 0.08, 0.19, 0.25)
        )
        selected = select_checkpoint([low, high])
        assert selected is not None and selected.epoch == 25

    def test_full_tie_prefers_later_epoch(self) -> None:
        topo = _topo(0.30, 1.0, 0.10, 0.10, 0.10)
        a = CheckpointCandidate(epoch=3, auprc=0.10, topology=topo)
        b = CheckpointCandidate(epoch=7, auprc=0.10, topology=topo)
        selected = select_checkpoint([a, b])
        assert selected is not None and selected.epoch == 7

    def test_non_finite_metric_fails_closed(self) -> None:
        bad = CheckpointCandidate(
            epoch=1, auprc=float("nan"), topology=_topo(0.3, 1.0, 0.1, 0.1, 0.1)
        )
        with pytest.raises(ValueError, match="non-finite"):
            select_checkpoint([bad])

    def test_negative_rd_fails_closed(self) -> None:
        bad = CheckpointCandidate(epoch=1, auprc=0.1, topology=_topo(0.3, -0.1, 0.1, 0.1, 0.1))
        with pytest.raises(ValueError, match="RD must be non-negative"):
            select_checkpoint([bad])

    def test_rd_ranked_by_abs_log_distance_to_one(self) -> None:
        """RD 1.05 beats RD 0.5: |log RD| closeness to 0, not magnitude, is ranked."""
        near_one = CheckpointCandidate(
            epoch=1, auprc=0.10, topology=_topo(0.30, 1.05, 0.10, 0.10, 0.10)
        )
        sparse = CheckpointCandidate(
            epoch=2, auprc=0.10, topology=_topo(0.30, 0.50, 0.10, 0.10, 0.10)
        )
        selected = select_checkpoint([near_one, sparse])
        assert selected is not None and selected.epoch == 1

    def test_log_rd_orders_ratios_symmetrically_unlike_abs_rd_minus_one(self) -> None:
        """RD 1.6 beats RD 0.5: |log 1.6| < |log 0.5| although |RD-1| says otherwise."""
        halved = CheckpointCandidate(
            epoch=1, auprc=0.10, topology=_topo(0.30, 0.50, 0.10, 0.10, 0.10)
        )
        dense = CheckpointCandidate(
            epoch=2, auprc=0.10, topology=_topo(0.30, 1.60, 0.10, 0.10, 0.10)
        )
        selected = select_checkpoint([halved, dense])
        assert selected is not None and selected.epoch == 2

    def test_rd_zero_is_a_legitimate_worst_rank_not_an_error(self) -> None:
        """A finite RD of 0 derives |log RD| = +inf and simply ranks last."""
        empty = CheckpointCandidate(
            epoch=9, auprc=0.10, topology=_topo(0.30, 0.0, 0.10, 0.10, 0.10)
        )
        sparse = CheckpointCandidate(
            epoch=2, auprc=0.10, topology=_topo(0.30, 0.10, 0.10, 0.10, 0.10)
        )
        selected = select_checkpoint([empty, sparse])
        assert selected is not None and selected.epoch == 2
