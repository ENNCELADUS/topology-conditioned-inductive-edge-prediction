"""Pure tensor KD loss functions for the B1 arms (`kd_control`/`kd_d1`/.../`kd_d5`).

Every function is a stateless tensor->tensor reduction with an explicit `mask`
(no model imports, no config reads): the trainer (`src.train_egostitch`,
a later wave) supplies student logits/features, the loaded `KDTargets`
(`src.distill.artifacts`) supplies teacher values, and every masked-out row
contributes exactly zero -- never NaN, regardless of group size.

- `kd_label_loss` is the `kd_control` arm's term (BCE against `pair_label`).
- `kd_logit_loss` is `kd_d1` (GLNN-style pointwise soft-label KD).
- `kd_rank_loss` + `kd_dist_loss` are `kd_d2` (LLP_R + LLP_D, Guo et al.,
  "Linkless Link Prediction via Relational Distillation", ICML 2023).
- `kd_gram_loss` is `kd_d3` (pair-space cosine-Gram matching, Graph2Feat/CAZI
  family).
- `kd_align_loss` is `kd_d4` (SA-MLP/SALE-MLP-style projected representation
  alignment: a learnable head answers the rotation/non-identifiability
  objection to raw coordinate MSE).
- `kd_residual_loss` is `kd_d5` (topology-residual distillation: the
  student's node-factor residual against the teacher's beyond-content
  residual, `teacher_logit - content_logit`).
- `kd_d6` is `kd_rank_loss` + `kd_dist_loss` + `kd_gram_loss` run together
  (the D2+D3 interaction test) -- no new loss code.
- `kd_d7a` is `kd_d2`'s exact losses (`kd_rank_loss` + `kd_dist_loss`) run
  over a heuristic-teacher targets artifact instead of the learned oracle --
  also no new loss code, only a different `teacher_logit` source.
- `kd_seed_loss` + `kd_seed_gram_loss` + `kd_kl_loss` are `kd_d9`
  (pair-conditioned oracle latent generation): posterior-path reconstruction
  of the teacher's symmetrized PMA seed tokens (per-slot cosine + raw
  within-pair Gram) plus the free-bits CVAE KL tying prior to posterior.

LLP constants, verified against the paper's released code
(https://github.com/snap-research/linkless-link-prediction/blob/main/src/main.py,
2026-08-13): `margin` defaults to ``0.1`` (`argparse` default, line 263,
confirmed by the `configurations/*.yaml` sweeps' `margin` grid centering on
it); the distribution-matching temperature is not a tunable at all -- both
`kl_loss` call sites hardcode ``T=1`` (lines 27-29, 107, 188) -- so
`DEFAULT_TEMPERATURE` here is ``1.0``, not a guessed fallback. `kd_rank_loss`
deviates from the reference in one respect: the reference gives near-tied
teacher pairs (within `margin`) a constant-``margin`` penalty via a `target=0`
row fed through `nn.MarginRankingLoss`; this module instead excludes only
exact teacher ties and ranks every other pair by `sign`, per this package's
own spec (a cleaner, dead-zone-free formulation, not a reproduction bug).
Context-size defaults (`k_near`/`k_rand`) are owned by `DistillConfig`
(a parallel wave), out of scope here.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_MARGIN = 0.1
DEFAULT_TEMPERATURE = 1.0

_EPS = 1e-12
# Finite (not -inf) masked-softmax sentinel: subtracting two `-inf` sentinels
# in the same group (`kd_dist_loss`'s all-dead-group case) would produce NaN
# that `torch.where` cannot un-poison in backward (0 * NaN = NaN), even though
# every affected row is discarded in the forward value. A large finite
# sentinel keeps every intermediate finite while still underflowing to an
# exp() of 0 for any group with at least one live row.
_NEG_SENTINEL = -1.0e9


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over the boolean/float `valid` mask; exact 0 for an empty mask."""
    weight = valid.to(dtype=values.dtype)
    denom = weight.sum()
    live = (denom > 0).to(values.dtype)
    return (values * weight).sum() / denom.clamp_min(1.0) * live


def kd_label_loss(
    student_logit: torch.Tensor, pair_label: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked-mean BCE-with-logits against the pair's `g_fit` edge label.

    The `kd_control` arm's auxiliary term: every arm consumes the identical
    KD pair stream (B1 plan's supervision-budget-confound fix), and control
    supervises it with ground truth rather than a teacher soft target.
    """
    per_row = F.binary_cross_entropy_with_logits(
        student_logit, pair_label.to(dtype=student_logit.dtype), reduction="none"
    )
    return _masked_mean(per_row, mask)


def kd_logit_loss(
    student_logit: torch.Tensor, teacher_logit: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked-mean pointwise soft-target BCE (`kd_d1`, GLNN-style).

    ``BCE(student_logit, sigmoid(teacher_logit))`` is the Bernoulli KL form:
    writing ``p_t = sigmoid(teacher_logit)`` and ``p_s = sigmoid(student_logit)``,
    ``KL(p_t || p_s) = BCE(student_logit, p_t) - H(p_t)``, and ``H(p_t)`` does
    not depend on the student, so minimizing this BCE is exactly minimizing
    the binary KL divergence up to a student-independent constant.
    """
    teacher_prob = torch.sigmoid(teacher_logit.to(dtype=student_logit.dtype))
    per_row = F.binary_cross_entropy_with_logits(student_logit, teacher_prob, reduction="none")
    return _masked_mean(per_row, mask)


def kd_rank_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    group_idx: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = DEFAULT_MARGIN,
) -> torch.Tensor:
    """LLP_R: margin-ranking loss over every within-group ordered pair (`kd_d2`).

    For live rows ``i, j`` in the same `group_idx` (an anchor id) whose
    teacher logits differ, the target is ``sign(teacher_i - teacher_j)`` and
    the per-pair loss is ``relu(margin - target * (student_i - student_j))``,
    averaged over every such ordered pair (`(i, j)` and `(j, i)` both count).
    A group with fewer than two live, teacher-distinct rows contributes
    exactly 0, not NaN.
    """
    live = mask.to(dtype=torch.bool)
    n = student_logit.shape[0]
    same_group = group_idx.unsqueeze(1) == group_idx.unsqueeze(0)
    live_pair = live.unsqueeze(1) & live.unsqueeze(0)
    not_self = ~torch.eye(n, dtype=torch.bool, device=student_logit.device)
    teacher_diff = teacher_logit.unsqueeze(1) - teacher_logit.unsqueeze(0)
    differ = teacher_diff != 0
    valid = same_group & live_pair & not_self & differ

    target = torch.sign(teacher_diff)
    student_diff = student_logit.unsqueeze(1) - student_logit.unsqueeze(0)
    per_pair = F.relu(margin - target * student_diff)
    return _masked_mean(per_pair, valid)


def kd_dist_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    group_idx: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """LLP_D: per-group ``KL(softmax(teacher/T) || softmax(student/T))`` (`kd_d2`).

    The softmax in each group (an anchor's context rows) is computed only
    over that group's live rows (masked softmax via a finite sentinel, not
    `-inf` -- see `_NEG_SENTINEL`). A group with fewer than two live rows
    contributes exactly 0: a single live row's masked softmax is a point mass
    at that row for both teacher and student, giving an exact-0 KL on its
    own, and this is additionally forced for safety.
    """
    device = student_logit.device
    dtype = student_logit.dtype
    live = mask.to(dtype=torch.bool)
    groups, inverse = torch.unique(group_idx, return_inverse=True)
    n_groups = int(groups.numel())

    t_over_tau = teacher_logit.to(dtype=dtype) / temperature
    s_over_tau = student_logit / temperature
    t_masked = torch.where(live, t_over_tau, torch.full_like(t_over_tau, _NEG_SENTINEL))
    s_masked = torch.where(live, s_over_tau, torch.full_like(s_over_tau, _NEG_SENTINEL))

    group_max_t = torch.full((n_groups,), _NEG_SENTINEL, dtype=dtype, device=device)
    group_max_t.scatter_reduce_(0, inverse, t_masked, reduce="amax", include_self=True)
    group_max_s = torch.full((n_groups,), _NEG_SENTINEL, dtype=dtype, device=device)
    group_max_s.scatter_reduce_(0, inverse, s_masked, reduce="amax", include_self=True)

    live_count = torch.zeros(n_groups, dtype=dtype, device=device)
    live_count.scatter_add_(0, inverse, live.to(dtype=dtype))
    valid_group = live_count >= 2

    exp_t = torch.where(
        live, torch.exp(t_masked - group_max_t[inverse]), torch.zeros_like(t_masked)
    )
    exp_s = torch.where(
        live, torch.exp(s_masked - group_max_s[inverse]), torch.zeros_like(s_masked)
    )
    sum_t = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(0, inverse, exp_t)
    sum_s = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(0, inverse, exp_s)
    safe_sum_t = torch.where(valid_group, sum_t, torch.ones_like(sum_t))
    safe_sum_s = torch.where(valid_group, sum_s, torch.ones_like(sum_s))

    p = exp_t / safe_sum_t[inverse]
    q = exp_s / safe_sum_s[inverse]
    row_kl = torch.where(
        live,
        p * (torch.log(p.clamp_min(_EPS)) - torch.log(q.clamp_min(_EPS))),
        torch.zeros_like(p),
    )
    group_kl = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(0, inverse, row_kl)
    return _masked_mean(group_kl, valid_group)


def kd_gram_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    pair_endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pair-space cosine-Gram matching loss (`kd_d3`).

    ``student_feat`` is the caller-supplied ``z_a (*) z_b`` elementwise
    product (the w-free pair feature -- `NodeFactorBottleneck.diag_w` is
    zero-initialized, so this loss's own gradient is what first makes it
    live). Both feature sets are row-normalized (eps-guarded), their cosine
    Gram matrices computed over the batch, and the diagonal plus every
    pair-pair sharing an endpoint (including exact/reversed duplicates,
    identified via `pair_endpoints`, shape ``(B, 2)``) excluded before the
    squared-Frobenius reduction -- otherwise two rows scoring the same or an
    adjacent pair trivially agree regardless of what either model learned.
    The reduction divides by the live off-diagonal count, so this loss's
    scale is batch-size invariant.
    """
    live = mask.to(dtype=torch.bool)
    student_norm = F.normalize(student_feat, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_feat.to(dtype=student_feat.dtype), p=2, dim=-1, eps=eps)
    gram_student = student_norm @ student_norm.transpose(0, 1)
    gram_teacher = teacher_norm @ teacher_norm.transpose(0, 1)

    n = student_feat.shape[0]
    not_self = ~torch.eye(n, dtype=torch.bool, device=student_feat.device)
    endpoint_a = pair_endpoints[:, 0]
    endpoint_b = pair_endpoints[:, 1]
    shares_endpoint = (
        (endpoint_a.unsqueeze(1) == endpoint_a.unsqueeze(0))
        | (endpoint_b.unsqueeze(1) == endpoint_b.unsqueeze(0))
        | (endpoint_a.unsqueeze(1) == endpoint_b.unsqueeze(0))
        | (endpoint_b.unsqueeze(1) == endpoint_a.unsqueeze(0))
    )
    live_pair = live.unsqueeze(1) & live.unsqueeze(0)
    valid = not_self & live_pair & ~shares_endpoint

    diff2 = (gram_student - gram_teacher) ** 2
    return _masked_mean(diff2, valid)


def kd_align_loss(
    student_align: torch.Tensor,
    teacher_feat: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked-mean per-row cosine distance, ``1 - cos(student_align, teacher_feat)`` (`kd_d4`).

    ``student_align`` is the model's `align_head`-projected pair feature
    (shape ``(B, D)``); ``teacher_feat`` is the symmetrized fp32 teacher
    pooled embedding, ``½(pooled_ab + pooled_ba)``, cast to `student_align`'s
    dtype before normalizing. Both are row-normalized (eps-guarded
    `F.normalize`) before the rowwise dot product, so the result is 0 for
    identical directions and 2 for exactly opposite ones, regardless of
    either vector's scale.
    """
    student_norm = F.normalize(student_align, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_feat.to(dtype=student_align.dtype), p=2, dim=-1, eps=eps)
    cosine = (student_norm * teacher_norm).sum(dim=-1)
    per_row = 1.0 - cosine
    return _masked_mean(per_row, mask)


def kd_residual_loss(
    student_residual: torch.Tensor,
    delta_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Masked-mean Huber loss between the student's residual and `delta_target` (`kd_d5`).

    ``student_residual`` is the model's node-factor residual
    (``output["node_factor_residual"]``, shape ``(B,)``); ``delta_target`` is
    the caller-computed ``teacher_logit - content_logit`` -- the teacher's
    beyond-content signal, cast to `student_residual`'s dtype. Smooth L1
    (Huber, half-width `beta`) rather than plain MSE bounds the gradient
    against the occasional large-magnitude residual outlier.
    """
    per_row = F.smooth_l1_loss(
        student_residual,
        delta_target.to(dtype=student_residual.dtype),
        beta=beta,
        reduction="none",
    )
    return _masked_mean(per_row, mask)


def kd_seed_loss(
    student_seeds: torch.Tensor,
    teacher_seeds: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked-mean per-slot cosine distance to the teacher's seed tokens (`kd_d9`).

    ``student_seeds``/``teacher_seeds`` are shape ``(B, K, d)``: the generated
    posterior-path seeds and the symmetrized teacher PMA seed tokens. Slot
    ``k`` matches slot ``k`` with no learned projection -- PMA seed queries
    are fixed learned parameters, so slots correspond across pairs and the
    student decodes directly in teacher coordinates. Per row: mean over slots
    of ``1 - cos``; reduced by masked mean over live rows.
    """
    student_norm = F.normalize(student_seeds, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_seeds.to(dtype=student_seeds.dtype), p=2, dim=-1, eps=eps)
    per_row = (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean(dim=-1)
    return _masked_mean(per_row, mask)


def kd_seed_gram_loss(
    student_seeds: torch.Tensor,
    teacher_seeds: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative Frobenius match of the raw within-pair ``K x K`` seed Gram (`kd_d9`).

    Per row: ``||S_s S_s^T - S_t S_t^T||_F^2 / (||S_t S_t^T||_F^2 + eps)``.
    The Gram is *raw* (unnormalized) on purpose: its diagonal carries slot
    norms and its off-diagonal the inter-slot geometry, complementing
    `kd_seed_loss`'s per-slot directions -- together they determine the seed
    set up to a shared feature-space rotation, which the raw-coordinate
    cosine term then pins. Unlike `kd_gram_loss` (pair-space, across the
    batch), this matches slot-slot geometry *within* one pair's seed set.
    """
    teacher = teacher_seeds.to(dtype=student_seeds.dtype)
    gram_student = student_seeds @ student_seeds.transpose(1, 2)
    gram_teacher = teacher @ teacher.transpose(1, 2)
    diff2 = (gram_student - gram_teacher).square().sum(dim=(1, 2))
    scale = gram_teacher.square().sum(dim=(1, 2))
    per_row = diff2 / (scale + eps)
    return _masked_mean(per_row, mask)


def kd_kl_loss(
    kl_per_dim: torch.Tensor,
    mask: torch.Tensor,
    *,
    free_bits: float,
) -> torch.Tensor:
    """Masked-mean free-bits KL reduction (`kd_d9`).

    ``kl_per_dim`` is the model-computed analytic per-dimension
    ``KL(q(z|c, S*) || p(z|c))``, shape ``(B, z_dim)``. Per row: sum over
    dimensions of ``max(kl_dim, free_bits)`` -- the floor keeps gradient off
    dimensions already below budget (posterior-collapse guard) without ever
    rewarding collapse. The caller owns the warmup schedule; this reduction
    is schedule-free.
    """
    floored = kl_per_dim.clamp_min(free_bits)
    return _masked_mean(floored.sum(dim=-1), mask)


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_TEMPERATURE",
    "kd_align_loss",
    "kd_dist_loss",
    "kd_gram_loss",
    "kd_kl_loss",
    "kd_label_loss",
    "kd_logit_loss",
    "kd_rank_loss",
    "kd_residual_loss",
    "kd_seed_gram_loss",
    "kd_seed_loss",
]
