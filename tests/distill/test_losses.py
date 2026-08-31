"""Reference-parity tests for the LLP context-pair losses."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from src.distill.losses import kd_dist_loss, kd_rank_loss

pytestmark = pytest.mark.unit


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-values))
    return result


def _numpy_rank_loss(
    student_logit: np.ndarray,
    teacher_logit: np.ndarray,
    group_idx: np.ndarray,
    delta: float,
) -> float:
    student_prob = _sigmoid(student_logit)
    teacher_prob = _sigmoid(teacher_logit)
    losses: list[float] = []
    for group in np.unique(group_idx):
        rows = np.flatnonzero(group_idx == group)
        for left_position, left in enumerate(rows):
            for right in rows[left_position + 1 :]:
                teacher_diff = teacher_prob[left] - teacher_prob[right]
                rank = 1.0 if teacher_diff > delta else -1.0 if teacher_diff < -delta else 0.0
                student_diff = student_prob[left] - student_prob[right]
                losses.append(max(0.0, -rank * student_diff + delta))
    return float(np.mean(losses)) if losses else 0.0


def _numpy_dist_loss(
    student_logit: np.ndarray,
    teacher_logit: np.ndarray,
    group_idx: np.ndarray,
) -> float:
    student_prob = _sigmoid(student_logit)
    teacher_prob = _sigmoid(teacher_logit)
    group_losses: list[float] = []
    for group in np.unique(group_idx):
        rows = group_idx == group
        student_exp = np.exp(student_prob[rows] - np.max(student_prob[rows]))
        teacher_exp = np.exp(teacher_prob[rows] - np.max(teacher_prob[rows]))
        student_dist = student_exp / student_exp.sum()
        teacher_dist = teacher_exp / teacher_exp.sum()
        group_losses.append(float(np.sum(teacher_dist * np.log(teacher_dist / student_dist))))
    return float(np.mean(group_losses)) if group_losses else 0.0


def test_llp_losses_match_direct_numpy_transcription_on_random_inputs() -> None:
    rng = np.random.default_rng(17)
    student = rng.standard_normal(17)
    teacher = rng.standard_normal(17)
    groups = np.repeat(np.arange(5), [4, 1, 5, 3, 4])
    permutation = rng.permutation(groups.size)
    student = student[permutation]
    teacher = teacher[permutation]
    groups = groups[permutation]
    delta = 0.1

    student_tensor = torch.from_numpy(student)
    teacher_tensor = torch.from_numpy(teacher)
    group_tensor = torch.from_numpy(groups)
    torch.testing.assert_close(
        kd_rank_loss(student_tensor, teacher_tensor, group_tensor, margin=delta),
        torch.tensor(_numpy_rank_loss(student, teacher, groups, delta), dtype=student_tensor.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        kd_dist_loss(student_tensor, teacher_tensor, group_tensor),
        torch.tensor(_numpy_dist_loss(student, teacher, groups), dtype=student_tensor.dtype),
        atol=1e-12,
        rtol=1e-12,
    )


def test_rank_keeps_delta_boundary_ties_and_jointly_means_all_pairs() -> None:
    teacher_prob = torch.tensor([0.75, 0.5, 0.9, 0.1], dtype=torch.float64)
    student_prob = torch.tensor([0.9, 0.1, 0.6, 0.5], dtype=torch.float64)
    teacher = torch.logit(teacher_prob)
    student = torch.logit(student_prob).requires_grad_()
    groups = torch.tensor([0, 0, 1, 1])
    delta = float((torch.sigmoid(teacher[0]) - torch.sigmoid(teacher[1])).item())

    loss = kd_rank_loss(student, teacher, groups, margin=delta)

    # Group 0 is tied exactly at delta and contributes delta. Group 1 is
    # strictly ordered and contributes delta - (0.6 - 0.5). Both pairs share
    # one joint mean: (0.25 + 0.15) / 2 = 0.20.
    assert loss.item() == pytest.approx(0.2)
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    torch.testing.assert_close(student.grad[:2], torch.zeros(2, dtype=torch.float64))
    assert torch.count_nonzero(student.grad[2:]).item() == 2


def test_dist_sums_context_kl_then_means_over_all_anchors() -> None:
    teacher_prob = torch.tensor([0.8, 0.2, 0.4], dtype=torch.float64)
    student_prob = torch.tensor([0.3, 0.7, 0.9], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1])

    loss = kd_dist_loss(torch.logit(student_prob), torch.logit(teacher_prob), groups)

    teacher_first = math.exp(0.8) / (math.exp(0.8) + math.exp(0.2))
    student_first = math.exp(0.3) / (math.exp(0.3) + math.exp(0.7))
    two_context_kl = teacher_first * math.log(teacher_first / student_first) + (
        1.0 - teacher_first
    ) * math.log((1.0 - teacher_first) / (1.0 - student_first))
    # The singleton anchor has KL=0 but remains in the anchor-mean denominator.
    assert loss.item() == pytest.approx(two_context_kl / 2.0)
