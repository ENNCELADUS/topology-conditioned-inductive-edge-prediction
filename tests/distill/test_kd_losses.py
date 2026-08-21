"""Contracts for the pure KD loss functions in `src.distill.losses`.

Covers `kd_logit` (`kd_logit_loss`), `kd_rep` (`kd_rep_loss`), and `kd_d9`
(`kd_seed_loss`, `kd_seed_gram_loss`, `kd_kl_loss`). All five are plain batch
means with no mask argument -- task and KD supervision always share the same
row set (see the module docstring), so there is no padding to ignore.
"""

from __future__ import annotations

import pytest
import torch
from src.distill.losses import (
    kd_kl_loss,
    kd_logit_loss,
    kd_rep_loss,
    kd_seed_gram_loss,
    kd_seed_loss,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- kd_logit_loss


def test_kd_logit_loss_matches_manual_bce_against_teacher_sigmoid() -> None:
    student = torch.tensor([1.0, -0.5, 0.2])
    teacher = torch.tensor([2.0, -1.0, 0.0])

    expected = torch.nn.functional.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher))
    torch.testing.assert_close(kd_logit_loss(student, teacher), expected)


def test_kd_logit_loss_is_the_deterministic_entropy_value_when_student_equals_teacher() -> None:
    teacher = torch.tensor([1.5, -2.0, 0.3])
    result = kd_logit_loss(teacher.clone(), teacher)
    # BCE(x, sigmoid(x)) is not identically zero -- pin the exact finite value.
    expected = torch.nn.functional.binary_cross_entropy_with_logits(teacher, torch.sigmoid(teacher))
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result)


def test_kd_logit_loss_gradient_flows_to_student_only_with_a_detached_fp16_teacher() -> None:
    student = torch.tensor([1.0, -0.5, 0.2], requires_grad=True)
    teacher = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float16)

    result = kd_logit_loss(student, teacher)
    assert result.dtype == student.dtype
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


# --------------------------------------------------------------------------- kd_rep_loss


def test_kd_rep_loss_is_zero_for_identical_directions() -> None:
    student = torch.tensor([[3.0, 4.0]])
    teacher = torch.tensor([[3.0, 4.0]])
    result = kd_rep_loss(student, teacher)
    torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_kd_rep_loss_is_two_for_exactly_opposite_directions() -> None:
    student = torch.tensor([[3.0, 4.0]])
    teacher = torch.tensor([[-3.0, -4.0]])
    result = kd_rep_loss(student, teacher)
    torch.testing.assert_close(result, torch.tensor(2.0), atol=1e-6, rtol=0.0)


def test_kd_rep_loss_is_scale_invariant() -> None:
    torch.manual_seed(0)
    student = torch.randn(4, 6)
    teacher = torch.randn(4, 6)
    base = kd_rep_loss(student, teacher)
    scaled = kd_rep_loss(5.0 * student, 0.1 * teacher)
    torch.testing.assert_close(base, scaled, atol=1e-5, rtol=1e-5)


def test_kd_rep_loss_gradient_flows_to_student_only_with_a_detached_fp16_teacher() -> None:
    torch.manual_seed(1)
    student = torch.randn(3, 5, requires_grad=True)
    teacher = torch.randn(3, 5).to(dtype=torch.float16)

    result = kd_rep_loss(student, teacher)
    assert result.dtype == student.dtype
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


# --------------------------------------------------------------------------- kd_seed_loss


def test_kd_seed_loss_is_zero_for_identical_directions_and_two_for_opposite() -> None:
    torch.manual_seed(2)
    seeds = torch.randn(3, 4, 8)
    torch.testing.assert_close(
        kd_seed_loss(seeds, 2.0 * seeds), torch.tensor(0.0), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(kd_seed_loss(seeds, -seeds), torch.tensor(2.0), atol=1e-6, rtol=0.0)


def test_kd_seed_loss_matches_slot_k_to_slot_k_with_no_cross_slot_projection() -> None:
    """Slot k must match slot k -- a slot-permuted teacher must NOT still read as zero."""
    torch.manual_seed(3)
    seeds = torch.randn(2, 4, 8)
    student = seeds.clone()

    exact = kd_seed_loss(student, seeds)
    permuted_teacher = seeds[:, [1, 2, 3, 0], :]
    permuted = kd_seed_loss(student, permuted_teacher)

    assert float(exact) == pytest.approx(0.0, abs=1e-6)
    # Random vectors are near-orthogonal in expectation; a real cross-slot
    # projection could still recover zero loss, but slotwise cosine cannot.
    assert float(permuted) > 0.5


def test_kd_seed_loss_gradient_flows_to_student_only_with_a_detached_fp16_teacher() -> None:
    torch.manual_seed(4)
    student = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn(2, 4, 8).to(dtype=torch.float16)

    result = kd_seed_loss(student, teacher)
    assert result.dtype == student.dtype
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


# --------------------------------------------------------------------------- kd_seed_gram_loss


def test_kd_seed_gram_loss_is_zero_for_identical_seed_sets() -> None:
    torch.manual_seed(5)
    seeds = torch.randn(3, 4, 8)
    torch.testing.assert_close(
        kd_seed_gram_loss(seeds, seeds), torch.tensor(0.0), atol=1e-6, rtol=0.0
    )


def test_kd_seed_gram_loss_is_nonzero_under_a_slot_norm_change() -> None:
    # The Gram is raw (unnormalized): its diagonal carries slot norms, so a
    # uniform rescale must not be free the way a cosine-only loss would allow.
    torch.manual_seed(6)
    seeds = torch.randn(3, 4, 8)
    assert kd_seed_gram_loss(2.0 * seeds, seeds).item() > 0.0


def test_kd_seed_gram_loss_is_invariant_to_a_shared_rotation_of_both_sets() -> None:
    torch.manual_seed(7)
    student = torch.randn(2, 4, 8)
    teacher = torch.randn(2, 4, 8)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8))

    base = kd_seed_gram_loss(student, teacher)
    rotated = kd_seed_gram_loss(student @ rotation, teacher @ rotation)
    torch.testing.assert_close(base, rotated, atol=1e-5, rtol=1e-5)


def test_kd_seed_gram_loss_gradient_flows_to_student_only_with_a_detached_fp16_teacher() -> None:
    torch.manual_seed(8)
    student = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn(2, 4, 8).to(dtype=torch.float16)

    result = kd_seed_gram_loss(student, teacher)
    assert result.dtype == student.dtype
    result.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


# --------------------------------------------------------------------------- kd_kl_loss


def test_kd_kl_loss_all_below_floor_equals_z_dim_times_free_bits() -> None:
    z_dim = 5
    free_bits = 0.1
    kl = torch.full((3, z_dim), 0.01)
    result = kd_kl_loss(kl, free_bits=free_bits)
    torch.testing.assert_close(result, torch.tensor(z_dim * free_bits), atol=1e-6, rtol=0.0)


def test_kd_kl_loss_applies_the_free_bits_floor_per_dimension() -> None:
    kl = torch.tensor([[0.01, 0.2], [0.03, 0.04]])
    # Per row: sum(max(kl_dim, 0.05)) -> [0.05 + 0.2, 0.05 + 0.05]
    expected = torch.tensor((0.25 + 0.10) / 2.0)
    torch.testing.assert_close(kd_kl_loss(kl, free_bits=0.05), expected)


def test_kd_kl_loss_zero_free_bits_is_a_plain_mean_of_the_kl_sum() -> None:
    kl = torch.tensor([[0.5, 0.5], [100.0, 100.0]])
    torch.testing.assert_close(kd_kl_loss(kl, free_bits=0.0), torch.tensor(100.5))


def test_kd_kl_loss_gradient_is_zero_below_the_floor_and_nonzero_above() -> None:
    kl = torch.tensor([[0.01, 0.2]], requires_grad=True)
    result = kd_kl_loss(kl, free_bits=0.1)
    result.backward()  # type: ignore[no-untyped-call]
    assert kl.grad is not None
    assert torch.isfinite(kl.grad).all()
    # Below-floor dim 0 is in clamp_min's flat region: no gradient reaches it
    # (the posterior-collapse guard -- see the function docstring).
    assert kl.grad[0, 0].item() == 0.0
    assert kl.grad[0, 1].item() != 0.0
