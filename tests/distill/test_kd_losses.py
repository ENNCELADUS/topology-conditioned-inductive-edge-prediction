"""Contracts for the pure KD loss functions in `src.distill.losses`.

Covers `kd_control`/`kd_d1`/`kd_d2`/`kd_d3`/`kd_d4`/`kd_d5`; `kd_d6` and
`kd_d7a` reuse `kd_rank_loss`/`kd_dist_loss`/`kd_gram_loss` unchanged and add
no new loss functions to test here.
"""

from __future__ import annotations

import pytest
import torch
from src.distill.losses import (
    kd_align_loss,
    kd_dist_loss,
    kd_gram_loss,
    kd_label_loss,
    kd_logit_loss,
    kd_rank_loss,
    kd_residual_loss,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- kd_label_loss


def test_label_loss_matches_plain_bce_on_a_fully_live_batch() -> None:
    logit = torch.tensor([0.5, -1.2, 2.0, -0.3])
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
    torch.testing.assert_close(kd_label_loss(logit, label, mask), expected)


def test_label_loss_ignores_padded_rows() -> None:
    logit = torch.tensor([0.5, -1.2, 999.0])
    label = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    padded = kd_label_loss(logit, label, mask)
    unpadded = kd_label_loss(logit[:2], label[:2], torch.ones(2))
    torch.testing.assert_close(padded, unpadded)


def test_label_loss_is_exact_zero_for_an_empty_mask() -> None:
    logit = torch.tensor([0.5, -1.2])
    label = torch.tensor([1.0, 0.0])
    mask = torch.zeros(2)

    result = kd_label_loss(logit, label, mask)
    assert float(result) == 0.0
    assert torch.isfinite(result)


# --------------------------------------------------------------------------- kd_logit_loss


def test_logit_loss_matches_bce_against_teacher_sigmoid() -> None:
    student = torch.tensor([1.0, -0.5, 0.2])
    teacher = torch.tensor([2.0, -1.0, 0.0])
    mask = torch.ones(3)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher))
    torch.testing.assert_close(kd_logit_loss(student, teacher, mask), expected)


def test_logit_loss_is_zero_when_student_matches_teacher_exactly() -> None:
    teacher = torch.tensor([1.5, -2.0, 0.3])
    mask = torch.ones(3)
    result = kd_logit_loss(teacher.clone(), teacher, mask)
    # BCE(x, sigmoid(x)) is not identically zero -- assert it is the
    # deterministic, finite entropy value, not an unmasked artifact.
    expected = torch.nn.functional.binary_cross_entropy_with_logits(teacher, torch.sigmoid(teacher))
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result)


# --------------------------------------------------------------------------- kd_rank_loss


def test_rank_loss_is_zero_when_student_order_matches_teacher_beyond_margin() -> None:
    teacher = torch.tensor([1.0, 0.0, -1.0])
    # Student agrees with teacher ranking by a comfortable margin.
    student = torch.tensor([5.0, 0.0, -5.0])
    group_idx = torch.zeros(3, dtype=torch.long)
    mask = torch.ones(3)

    result = kd_rank_loss(student, teacher, group_idx, mask, margin=0.1)
    assert float(result) == pytest.approx(0.0, abs=1e-6)


def test_rank_loss_is_positive_when_student_order_is_inverted() -> None:
    teacher = torch.tensor([1.0, 0.0, -1.0])
    student = torch.tensor([-5.0, 0.0, 5.0])  # exactly inverted
    group_idx = torch.zeros(3, dtype=torch.long)
    mask = torch.ones(3)

    result = kd_rank_loss(student, teacher, group_idx, mask, margin=0.1)
    assert float(result) > 0.0


def test_rank_loss_only_compares_within_group() -> None:
    teacher = torch.tensor([1.0, -1.0, 1.0, -1.0])
    student = torch.tensor([5.0, -5.0, 5.0, -5.0])
    group_idx = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4)

    same_group_loss = kd_rank_loss(student, teacher, group_idx, mask, margin=0.1)
    # Perfectly consistent ranking within each group -> exact zero.
    assert float(same_group_loss) == pytest.approx(0.0, abs=1e-6)


def test_rank_loss_single_live_row_group_is_exact_zero_no_nan() -> None:
    teacher = torch.tensor([1.0, 0.0, -1.0])
    student = torch.tensor([0.3, 0.0, 0.0])
    group_idx = torch.tensor([0, 1, 2])  # every row its own group
    mask = torch.ones(3)

    result = kd_rank_loss(student, teacher, group_idx, mask, margin=0.1)
    assert float(result) == 0.0
    assert torch.isfinite(result)


def test_rank_loss_padded_rows_never_affect_the_value() -> None:
    teacher = torch.tensor([1.0, -1.0, 999.0])
    student = torch.tensor([-5.0, 5.0, -999.0])  # inverted padded row too
    group_idx = torch.tensor([0, 0, 0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    padded = kd_rank_loss(student, teacher, group_idx, mask, margin=0.1)
    unpadded = kd_rank_loss(student[:2], teacher[:2], group_idx[:2], torch.ones(2), margin=0.1)
    torch.testing.assert_close(padded, unpadded)


# --------------------------------------------------------------------------- kd_dist_loss


def test_dist_loss_is_zero_at_identical_distributions() -> None:
    logit = torch.tensor([1.0, 0.5, -0.3, 2.0])
    group_idx = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4)

    result = kd_dist_loss(logit.clone(), logit, group_idx, mask, temperature=1.0)
    assert float(result) == pytest.approx(0.0, abs=1e-5)


def test_dist_loss_is_positive_when_distributions_differ() -> None:
    teacher = torch.tensor([2.0, -2.0])
    student = torch.tensor([-2.0, 2.0])
    group_idx = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(2)

    result = kd_dist_loss(student, teacher, group_idx, mask, temperature=1.0)
    assert float(result) > 0.0


def test_dist_loss_single_live_row_group_is_exact_zero_no_nan() -> None:
    teacher = torch.tensor([3.0, -1.0, 0.5])
    student = torch.tensor([-1.0, 4.0, 0.1])
    group_idx = torch.tensor([0, 1, 2])
    mask = torch.ones(3)

    result = kd_dist_loss(student, teacher, group_idx, mask, temperature=1.0)
    assert float(result) == 0.0
    assert torch.isfinite(result)


def test_dist_loss_padded_rows_never_affect_the_value() -> None:
    teacher = torch.tensor([1.0, -1.0, 50.0])
    student = torch.tensor([0.2, -0.6, -50.0])
    group_idx = torch.tensor([0, 0, 0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    padded = kd_dist_loss(student, teacher, group_idx, mask, temperature=1.0)
    unpadded = kd_dist_loss(student[:2], teacher[:2], group_idx[:2], torch.ones(2), temperature=1.0)
    torch.testing.assert_close(padded, unpadded, atol=1e-6, rtol=1e-5)


def test_dist_loss_is_finite_with_an_entirely_dead_group() -> None:
    teacher = torch.tensor([1.0, -1.0, 0.3, 0.3])
    student = torch.tensor([0.2, -0.5, 0.1, 0.1])
    group_idx = torch.tensor([0, 0, 1, 1])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])  # group 1 fully masked out

    result = kd_dist_loss(student, teacher, group_idx, mask, temperature=1.0)
    assert torch.isfinite(result)


def test_dist_loss_gradient_is_finite_through_dead_groups() -> None:
    teacher = torch.tensor([1.0, -1.0, 0.3, 0.3])
    student = torch.tensor([0.2, -0.5, 0.1, 0.1], requires_grad=True)
    group_idx = torch.tensor([0, 0, 1, 1])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    result = kd_dist_loss(student, teacher, group_idx, mask, temperature=1.0)
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


# --------------------------------------------------------------------------- kd_gram_loss


def _pair_endpoints(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long)


def test_gram_loss_is_zero_for_identical_feature_sets() -> None:
    torch.manual_seed(0)
    feat = torch.randn(6, 8)
    endpoints = _pair_endpoints([(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0)])
    mask = torch.ones(6)

    result = kd_gram_loss(feat, feat.clone(), endpoints, mask)
    assert float(result) == pytest.approx(0.0, abs=1e-5)


def test_gram_loss_is_positive_for_permuted_rows() -> None:
    torch.manual_seed(1)
    student_feat = torch.randn(6, 8)
    teacher_feat = student_feat[torch.tensor([1, 0, 3, 2, 5, 4])]
    endpoints = _pair_endpoints([(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0)])
    mask = torch.ones(6)

    result = kd_gram_loss(student_feat, teacher_feat, endpoints, mask)
    assert float(result) > 0.0


def test_gram_loss_excludes_pairs_sharing_an_endpoint() -> None:
    torch.manual_seed(2)
    feat = torch.randn(4, 8)
    # Rows 0 and 1 share endpoint node "1" (0,1) & (1,2); row 3 is unrelated to both.
    endpoints = _pair_endpoints([(0, 1), (1, 2), (3, 4), (5, 6)])
    mask = torch.ones(4)
    teacher_feat = feat + torch.randn(4, 8) * 0.5

    # Manually zero out the excluded entries and confirm the loss matches a
    # hand-computed masked Frobenius norm over the surviving entries.
    student_norm = torch.nn.functional.normalize(feat, dim=-1)
    teacher_norm = torch.nn.functional.normalize(teacher_feat, dim=-1)
    gram_s = student_norm @ student_norm.T
    gram_t = teacher_norm @ teacher_norm.T
    a0, a1 = endpoints[:, 0], endpoints[:, 1]
    shares = (
        (a0.unsqueeze(1) == a0.unsqueeze(0))
        | (a1.unsqueeze(1) == a1.unsqueeze(0))
        | (a0.unsqueeze(1) == a1.unsqueeze(0))
        | (a1.unsqueeze(1) == a0.unsqueeze(0))
    )
    not_self = ~torch.eye(4, dtype=torch.bool)
    valid = not_self & ~shares
    expected = ((gram_s - gram_t) ** 2 * valid).sum() / valid.sum()

    result = kd_gram_loss(feat, teacher_feat, endpoints, mask)
    torch.testing.assert_close(result, expected)


def test_gram_loss_padded_rows_never_affect_the_value() -> None:
    torch.manual_seed(3)
    feat = torch.randn(4, 8)
    teacher_feat = torch.randn(4, 8)
    endpoints = _pair_endpoints([(0, 1), (2, 3), (4, 5), (6, 7)])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    padded = kd_gram_loss(feat, teacher_feat, endpoints, mask)
    unpadded = kd_gram_loss(feat[:3], teacher_feat[:3], endpoints[:3], torch.ones(3))
    torch.testing.assert_close(padded, unpadded)


def test_gram_loss_is_batch_size_invariant_given_the_same_valid_entries() -> None:
    """Appending more masked-out (padded) rows must not move the loss.

    The reduction normalizes by the live off-diagonal count, not by batch size.
    """
    torch.manual_seed(4)
    feat = torch.randn(3, 8)
    teacher_feat = torch.randn(3, 8)
    endpoints = _pair_endpoints([(0, 1), (2, 3), (4, 5)])
    base_loss = kd_gram_loss(feat, teacher_feat, endpoints, torch.ones(3))

    for n_padding in (1, 4, 9):
        padded_feat = torch.cat([feat, torch.randn(n_padding, 8)], dim=0)
        padded_teacher = torch.cat([teacher_feat, torch.randn(n_padding, 8)], dim=0)
        padded_endpoints = torch.cat(
            [endpoints, _pair_endpoints([(100 + i, 200 + i) for i in range(n_padding)])], dim=0
        )
        padded_mask = torch.cat([torch.ones(3), torch.zeros(n_padding)], dim=0)
        padded_loss = kd_gram_loss(padded_feat, padded_teacher, padded_endpoints, padded_mask)
        torch.testing.assert_close(padded_loss, base_loss)


# --------------------------------------------------------------------------- kd_align_loss


def test_align_loss_is_zero_for_identical_directions() -> None:
    torch.manual_seed(5)
    student = torch.randn(4, 6)
    mask = torch.ones(4)

    result = kd_align_loss(student, student.clone(), mask)
    assert float(result) == pytest.approx(0.0, abs=1e-5)


def test_align_loss_is_two_for_exactly_opposite_directions() -> None:
    torch.manual_seed(6)
    student = torch.randn(3, 5)
    teacher = -student
    mask = torch.ones(3)

    result = kd_align_loss(student, teacher, mask)
    assert float(result) == pytest.approx(2.0, abs=1e-5)


def test_align_loss_ignores_padded_rows() -> None:
    torch.manual_seed(7)
    student = torch.randn(3, 4)
    teacher = torch.randn(3, 4)
    mask = torch.tensor([1.0, 1.0, 0.0])

    padded = kd_align_loss(student, teacher, mask)
    unpadded = kd_align_loss(student[:2], teacher[:2], torch.ones(2))
    torch.testing.assert_close(padded, unpadded)


def test_align_loss_is_exact_zero_for_an_empty_mask() -> None:
    torch.manual_seed(8)
    student = torch.randn(3, 4)
    teacher = torch.randn(3, 4)
    mask = torch.zeros(3)

    result = kd_align_loss(student, teacher, mask)
    assert float(result) == 0.0
    assert torch.isfinite(result)


def test_align_loss_gradient_is_finite_for_an_empty_mask() -> None:
    torch.manual_seed(8)
    student = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4)
    mask = torch.zeros(3)

    result = kd_align_loss(student, teacher, mask)
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_align_loss_is_batch_size_invariant_given_the_same_valid_rows() -> None:
    torch.manual_seed(9)
    student = torch.randn(3, 4)
    teacher = torch.randn(3, 4)
    base_loss = kd_align_loss(student, teacher, torch.ones(3))

    for n_padding in (1, 5):
        padded_student = torch.cat([student, torch.randn(n_padding, 4)], dim=0)
        padded_teacher = torch.cat([teacher, torch.randn(n_padding, 4)], dim=0)
        padded_mask = torch.cat([torch.ones(3), torch.zeros(n_padding)], dim=0)
        padded_loss = kd_align_loss(padded_student, padded_teacher, padded_mask)
        torch.testing.assert_close(padded_loss, base_loss)


def test_align_loss_handles_bf16_teacher_input() -> None:
    torch.manual_seed(10)
    student = torch.randn(4, 6)
    teacher_bf16 = torch.randn(4, 6).to(dtype=torch.bfloat16)
    mask = torch.ones(4)

    result = kd_align_loss(student, teacher_bf16, mask)
    assert result.dtype == student.dtype
    assert torch.isfinite(result)


# --------------------------------------------------------------------------- kd_residual_loss


def test_residual_loss_matches_manual_smooth_l1_on_a_small_example() -> None:
    student_residual = torch.tensor([0.5, -1.5, 3.0])
    delta_target = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.ones(3)

    expected = torch.nn.functional.smooth_l1_loss(student_residual, delta_target, beta=1.0)
    result = kd_residual_loss(student_residual, delta_target, mask, beta=1.0)
    torch.testing.assert_close(result, expected)


def test_residual_loss_ignores_padded_rows() -> None:
    student_residual = torch.tensor([0.5, -1.5, 999.0])
    delta_target = torch.tensor([0.2, -1.0, -999.0])
    mask = torch.tensor([1.0, 1.0, 0.0])

    padded = kd_residual_loss(student_residual, delta_target, mask)
    unpadded = kd_residual_loss(student_residual[:2], delta_target[:2], torch.ones(2))
    torch.testing.assert_close(padded, unpadded)


def test_residual_loss_is_exact_zero_for_an_empty_mask() -> None:
    student_residual = torch.tensor([0.5, -1.5])
    delta_target = torch.tensor([3.0, -3.0])
    mask = torch.zeros(2)

    result = kd_residual_loss(student_residual, delta_target, mask)
    assert float(result) == 0.0
    assert torch.isfinite(result)


def test_residual_loss_gradient_is_finite_for_an_empty_mask() -> None:
    student_residual = torch.tensor([0.5, -1.5], requires_grad=True)
    delta_target = torch.tensor([3.0, -3.0])
    mask = torch.zeros(2)

    result = kd_residual_loss(student_residual, delta_target, mask)
    result.backward()  # type: ignore[no-untyped-call]
    assert student_residual.grad is not None
    assert torch.isfinite(student_residual.grad).all()


def test_residual_loss_is_batch_size_invariant_given_the_same_valid_rows() -> None:
    torch.manual_seed(11)
    student_residual = torch.randn(3)
    delta_target = torch.randn(3)
    base_loss = kd_residual_loss(student_residual, delta_target, torch.ones(3))

    for n_padding in (1, 6):
        padded_student = torch.cat([student_residual, torch.randn(n_padding)], dim=0)
        padded_delta = torch.cat([delta_target, torch.randn(n_padding)], dim=0)
        padded_mask = torch.cat([torch.ones(3), torch.zeros(n_padding)], dim=0)
        padded_loss = kd_residual_loss(padded_student, padded_delta, padded_mask)
        torch.testing.assert_close(padded_loss, base_loss)


def test_residual_loss_handles_bf16_delta_target_input() -> None:
    student_residual = torch.tensor([0.5, -1.5, 2.0])
    delta_target_bf16 = torch.tensor([0.3, -1.0, 1.8]).to(dtype=torch.bfloat16)
    mask = torch.ones(3)

    result = kd_residual_loss(student_residual, delta_target_bf16, mask)
    assert result.dtype == student_residual.dtype
    assert torch.isfinite(result)
