"""Pure tensor KD loss functions for the B1 full-row arms (`kd_logit`/`kd_rep`/`kd_d9`).

Every function is a stateless tensor->tensor reduction over one training batch
(no model imports, no config reads): the trainer (`src.train_b0`) supplies
student outputs from the single task forward, and the row-aligned
`KDRowTargets` artifact (`src.distill.artifacts`) supplies teacher values for
exactly the same ``_row_id`` set -- task and KD supervision always share rows,
so none of these reductions needs a mask.

- `kd_logit_loss` is `kd_logit` (GLNN-style pointwise binary soft-target KD).
- `kd_rep_loss` is `kd_rep` (normalized per-row cosine alignment of the
  student pair representation -- through a train-time linear projection only
  when student and teacher widths differ -- to the symmetrized teacher pooled
  embedding).
- `kd_seed_loss` + `kd_seed_gram_loss` + `kd_kl_loss` are `kd_d9`
  (pair-conditioned oracle latent generation): posterior-path reconstruction
  of the teacher's symmetrized PMA seed tokens (per-slot cosine + raw
  within-pair Gram) plus the free-bits CVAE KL tying prior to posterior.

``BCE(student_logit, sigmoid(teacher_logit))`` is the Bernoulli KL form:
writing ``p_t = sigmoid(teacher_logit)``, ``KL(p_t || p_s) = BCE(student_logit,
p_t) - H(p_t)``, and ``H(p_t)`` does not depend on the student, so minimizing
the BCE minimizes the binary KL up to a student-independent constant.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def kd_logit_loss(student_logit: torch.Tensor, teacher_logit: torch.Tensor) -> torch.Tensor:
    """Mean pointwise soft-target BCE (`kd_logit`; see the module docstring)."""
    teacher_prob = torch.sigmoid(teacher_logit.to(dtype=student_logit.dtype))
    return F.binary_cross_entropy_with_logits(student_logit, teacher_prob)


def kd_rep_loss(
    student_rep: torch.Tensor,
    teacher_rep: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-row cosine distance ``1 - cos(student_rep, teacher_rep)`` (`kd_rep`).

    Both are row-normalized (eps-guarded `F.normalize`) before the rowwise dot
    product, so the result is 0 for identical directions and 2 for exactly
    opposite ones, regardless of either vector's scale. `teacher_rep` is the
    symmetrized fp32 teacher pooled embedding ``0.5 * (pooled_ab + pooled_ba)``,
    cast to `student_rep`'s dtype before normalizing.
    """
    student_norm = F.normalize(student_rep, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_rep.to(dtype=student_rep.dtype), p=2, dim=-1, eps=eps)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()


def kd_seed_loss(
    student_seeds: torch.Tensor,
    teacher_seeds: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-slot cosine distance to the teacher's seed tokens (`kd_d9`).

    ``student_seeds``/``teacher_seeds`` are shape ``(B, K, d)``: the generated
    posterior-path seeds and the symmetrized teacher PMA seed tokens. Slot
    ``k`` matches slot ``k`` with no learned projection -- PMA seed queries
    are fixed learned parameters, so slots correspond across pairs and the
    student decodes directly in teacher coordinates. Per row: mean over slots
    of ``1 - cos``; reduced by the batch mean.
    """
    student_norm = F.normalize(student_seeds, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_seeds.to(dtype=student_seeds.dtype), p=2, dim=-1, eps=eps)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()


def kd_seed_gram_loss(
    student_seeds: torch.Tensor,
    teacher_seeds: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative Frobenius match of the raw within-pair ``K x K`` seed Gram (`kd_d9`).

    Per row: ``||S_s S_s^T - S_t S_t^T||_F^2 / (||S_t S_t^T||_F^2 + eps)``.
    The Gram is *raw* (unnormalized) on purpose: its diagonal carries slot
    norms and its off-diagonal the inter-slot geometry, complementing
    `kd_seed_loss`'s per-slot directions -- together they determine the seed
    set up to a shared feature-space rotation, which the raw-coordinate
    cosine term then pins. This matches slot-slot geometry *within* one
    pair's seed set, not across the batch.
    """
    teacher = teacher_seeds.to(dtype=student_seeds.dtype)
    gram_student = student_seeds @ student_seeds.transpose(1, 2)
    gram_teacher = teacher @ teacher.transpose(1, 2)
    diff2 = (gram_student - gram_teacher).square().sum(dim=(1, 2))
    scale = gram_teacher.square().sum(dim=(1, 2))
    return (diff2 / (scale + eps)).mean()


def kd_kl_loss(kl_per_dim: torch.Tensor, *, free_bits: float) -> torch.Tensor:
    """Mean free-bits KL reduction (`kd_d9`).

    ``kl_per_dim`` is the model-computed analytic per-dimension
    ``KL(q(z|c, S*) || p(z|c))``, shape ``(B, z_dim)``. Per row: sum over
    dimensions of ``max(kl_dim, free_bits)`` -- the floor keeps gradient off
    dimensions already below budget (posterior-collapse guard) without ever
    rewarding collapse. The caller owns the warmup schedule; this reduction
    is schedule-free.
    """
    return kl_per_dim.clamp_min(free_bits).sum(dim=-1).mean()


__all__ = [
    "kd_kl_loss",
    "kd_logit_loss",
    "kd_rep_loss",
    "kd_seed_gram_loss",
    "kd_seed_loss",
]
