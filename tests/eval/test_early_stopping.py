"""The monitored total validation loss (`src.eval.early_stopping`).

Patience counts on this quantity in every training worker; checkpoint selection
never does (`src.eval.checkpoint_selection`). These tests pin the arm -> term
table, because a silently dropped KD term would make the monitor jump mid-run
and fire patience on a composition change rather than on stagnation.
"""

from __future__ import annotations

import pytest
from src.distill.config import DistillConfig
from src.eval.early_stopping import compose_val_total, val_total_terms


class TestComposeValTotal:
    def test_no_distill_section_is_the_task_loss(self) -> None:
        assert compose_val_total(0.25, None, None) == 0.25

    def test_all_zero_weights_are_the_task_loss(self) -> None:
        assert compose_val_total(0.25, {"val_kd_logit_loss": 9.0}, DistillConfig()) == 0.25

    def test_kd_logit_adds_the_weighted_logit_term(self) -> None:
        distill = DistillConfig(targets_path="t", w_logit=2.0)
        total = compose_val_total(0.25, {"val_kd_logit_loss": 0.5}, distill)
        assert total == pytest.approx(0.25 + 2.0 * 0.5)

    def test_kd_rank_adds_both_context_terms(self) -> None:
        distill = DistillConfig(targets_path="t", context_targets_path="c", w_rank=3.0, w_dist=0.5)
        kd = {"val_kd_rank_loss": 0.2, "val_kd_dist_loss": 0.4, "val_kd_rank_live_pairs": 128.0}
        total = compose_val_total(1.0, kd, distill)
        assert total == pytest.approx(1.0 + 3.0 * 0.2 + 0.5 * 0.4)

    def test_kd_rep_uses_the_one_minus_cosine_term_not_the_cosine(self) -> None:
        distill = DistillConfig(targets_path="t", w_rep=1.0)
        kd = {"val_kd_rep_loss": 0.3, "val_kd_rep_cos": 0.7}
        assert compose_val_total(0.1, kd, distill) == pytest.approx(0.4)

    def test_kd_gram_uses_the_block_loss(self) -> None:
        distill = DistillConfig(targets_path="t", w_gram=4.0)
        assert compose_val_total(0.1, {"val_kd_gram_block_loss": 0.25}, distill) == pytest.approx(
            1.1
        )

    def test_kd_gen_has_no_validation_counterpart_and_degrades_to_the_task_term(self) -> None:
        distill = DistillConfig(targets_path="t", w_gen=5.0)
        assert compose_val_total(0.6, {"val_kd_latent_cos": 0.9}, distill) == 0.6

    def test_missing_counterpart_for_an_active_arm_raises(self) -> None:
        distill = DistillConfig(targets_path="t", w_logit=1.0)
        with pytest.raises(RuntimeError, match="val_kd_logit_loss"):
            compose_val_total(0.25, {"val_kd_prob_mae": 0.1}, distill)

    def test_active_arm_with_no_kd_diagnostics_at_all_raises(self) -> None:
        distill = DistillConfig(targets_path="t", w_rep=1.0)
        with pytest.raises(RuntimeError, match="kd_rep"):
            compose_val_total(0.25, None, distill)

    def test_kd_rank_rep_adds_rank_dist_and_rep_terms(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=3.0, w_dist=0.5, w_rep=2.0
        )
        kd = {"val_kd_rank_loss": 0.2, "val_kd_dist_loss": 0.4, "val_kd_rep_loss": 0.1}
        assert compose_val_total(0.25, kd, distill) == pytest.approx(
            0.25 + 3.0 * 0.2 + 0.5 * 0.4 + 2.0 * 0.1
        )

    def test_kd_rank_rep_missing_rep_counterpart_raises(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=1.0, w_dist=1.0, w_rep=1.0
        )
        with pytest.raises(RuntimeError, match="val_kd_rep_loss"):
            compose_val_total(0.25, {"val_kd_rank_loss": 0.2, "val_kd_dist_loss": 0.4}, distill)


class TestValTotalTerms:
    def test_undistilled_names_only_the_task_term(self) -> None:
        assert val_total_terms(None) == ["val_task_loss"]

    def test_kd_rank_names_both_weighted_terms_in_order(self) -> None:
        distill = DistillConfig(targets_path="t", context_targets_path="c", w_rank=1.0, w_dist=1.0)
        assert val_total_terms(distill) == [
            "val_task_loss",
            "w_rank * val_kd_rank_loss",
            "w_dist * val_kd_dist_loss",
        ]

    def test_kd_rank_rep_names_three_weighted_terms(self) -> None:
        distill = DistillConfig(
            targets_path="t", context_targets_path="c", w_rank=0.1, w_dist=10.0, w_rep=1.0
        )
        assert val_total_terms(distill) == [
            "val_task_loss",
            "w_rank * val_kd_rank_loss",
            "w_dist * val_kd_dist_loss",
            "w_rep * val_kd_rep_loss",
        ]
