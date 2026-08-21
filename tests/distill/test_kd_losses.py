"""Contracts for full-row B1 KD and Gate A set-student losses."""

from __future__ import annotations

import pytest
import torch
from src.distill.losses import (
    kd_dist_loss,
    kd_gram_loss,
    kd_kl_loss,
    kd_logit_loss,
    kd_rank_loss,
    kd_rep_loss,
    kd_seed_gram_loss,
    kd_seed_loss,
    kd_set_gram_loss,
    kd_set_seed_loss,
)

pytestmark = pytest.mark.unit


def test_kd_logit_loss_matches_teacher_soft_target_and_backpropagates() -> None:
    student = torch.tensor([1.0, -0.5, 0.2], requires_grad=True)
    teacher = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float16)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        student, torch.sigmoid(teacher.float())
    )
    loss = kd_logit_loss(student, teacher)
    torch.testing.assert_close(loss, expected)
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_kd_rep_loss_is_scale_invariant_and_backpropagates() -> None:
    torch.manual_seed(1)
    student = torch.randn(3, 5, requires_grad=True)
    teacher = torch.randn(3, 5)
    loss = kd_rep_loss(student, teacher)
    torch.testing.assert_close(
        loss, kd_rep_loss(5.0 * student, 0.1 * teacher), atol=1e-5, rtol=1e-5
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_kd_rank_loss_uses_only_within_anchor_ordering() -> None:
    teacher = torch.tensor([1.0, -1.0, -1.0, 1.0])
    group = torch.tensor([0, 0, 1, 1])
    matching = torch.tensor([5.0, -5.0, -5.0, 5.0])
    inverted = -matching
    assert kd_rank_loss(matching, teacher, group).item() == pytest.approx(0.0)
    assert kd_rank_loss(inverted, teacher, group).item() > 0.0


def test_kd_rank_loss_no_comparable_group_is_differentiable_zero() -> None:
    student = torch.tensor([0.2, -0.4, 0.1], requires_grad=True)
    teacher = torch.tensor([1.0, -1.0, 0.0])
    group = torch.tensor([0, 1, 2])
    loss = kd_rank_loss(student, teacher, group)
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_kd_dist_loss_matches_groups_and_handles_only_singletons() -> None:
    teacher = torch.tensor([2.0, -2.0, 1.0])
    group = torch.tensor([0, 0, 1])
    identical = kd_dist_loss(teacher.clone(), teacher, group)
    assert identical.item() == pytest.approx(0.0, abs=1e-6)
    student = torch.tensor([-2.0, 2.0, 0.0], requires_grad=True)
    loss = kd_dist_loss(student, teacher, group)
    assert loss.item() > 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()

    singleton_student = torch.randn(3, requires_grad=True)
    zero = kd_dist_loss(singleton_student, teacher, torch.arange(3))
    assert zero.item() == 0.0 and torch.isfinite(zero)
    zero.backward()  # type: ignore[no-untyped-call]
    assert singleton_student.grad is not None
    torch.testing.assert_close(singleton_student.grad, torch.zeros_like(singleton_student))


def test_kd_gram_loss_includes_rows_that_share_an_endpoint() -> None:
    # Rows 0 and 1 are the only two rows and represent endpoint-sharing pairs.
    # D3 receives no endpoint mask, so their changed cosine must remain live.
    student = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = kd_gram_loss(student, teacher)
    assert loss.item() > 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_kd_gram_loss_single_row_is_differentiable_zero() -> None:
    student = torch.randn(1, 4, requires_grad=True)
    loss = kd_gram_loss(student, torch.randn(1, 7))
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_kd_gram_loss_is_zero_under_independent_feature_width_rotation() -> None:
    torch.manual_seed(3)
    teacher = torch.randn(5, 4)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    torch.testing.assert_close(
        kd_gram_loss(teacher @ rotation, teacher), torch.tensor(0.0), atol=1e-5, rtol=0.0
    )


def test_d9_seed_losses_and_kl_keep_original_unmasked_api() -> None:
    torch.manual_seed(4)
    seeds = torch.randn(3, 4, 8)
    torch.testing.assert_close(
        kd_seed_loss(seeds, 2.0 * seeds), torch.tensor(0.0), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        kd_seed_gram_loss(seeds, seeds), torch.tensor(0.0), atol=1e-6, rtol=0.0
    )
    kl = torch.tensor([[0.01, 0.2], [0.03, 0.04]])
    torch.testing.assert_close(kd_kl_loss(kl, free_bits=0.05), torch.tensor(0.175))


def test_gate_set_seed_loss_masks_dead_rows_and_backpropagates() -> None:
    torch.manual_seed(5)
    student = torch.randn(3, 2, 4, requires_grad=True)
    teacher = torch.randn(3, 2, 4)
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss = kd_set_seed_loss(student, teacher, mask)
    torch.testing.assert_close(loss, kd_set_seed_loss(student[:2], teacher[:2], torch.ones(2)))
    loss.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()
    torch.testing.assert_close(student.grad[2], torch.zeros_like(student.grad[2]))
    empty = kd_set_seed_loss(student.detach(), teacher, torch.zeros(3))
    assert empty.item() == 0.0 and torch.isfinite(empty)


def test_gate_set_gram_ignores_padded_nodes_and_rows_without_two_nodes() -> None:
    torch.manual_seed(6)
    student = torch.randn(2, 3, 6)
    teacher = torch.randn(2, 3, 6)
    node_mask = torch.tensor([[True, True, True], [True, False, False]])
    expected = kd_set_gram_loss(student[:1], teacher[:1], node_mask[:1], torch.ones(1))
    result = kd_set_gram_loss(student, teacher, node_mask, torch.ones(2))
    torch.testing.assert_close(result, expected)
