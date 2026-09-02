"""Pure tensor losses for B1 KD and Gate A set-student training.

The D2 losses operate on explicit per-anchor context pairs; D3 matches
pair-representation geometry across rows from the normal task batch.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_MARGIN = 0.1
_EPS = 1e-12
_NEG_SENTINEL = -1.0e9


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over ``valid`` entries, with a differentiable zero if none are valid."""
    weight = valid.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def kd_logit_loss(student_logit: torch.Tensor, teacher_logit: torch.Tensor) -> torch.Tensor:
    """Mean pointwise binary soft-target KD."""
    teacher_prob = torch.sigmoid(teacher_logit.to(dtype=student_logit.dtype))
    return F.binary_cross_entropy_with_logits(student_logit, teacher_prob)


def kd_rep_loss(
    student_rep: torch.Tensor,
    teacher_rep: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-row cosine distance between student and teacher pair representations."""
    student_norm = F.normalize(student_rep, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_rep.to(dtype=student_rep.dtype), p=2, dim=-1, eps=eps)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()


def kd_struct_loss(student_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the auxiliary head against z-scored structural descriptors."""
    return F.mse_loss(student_pred, target.to(dtype=student_pred.dtype))


def kd_rank_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    group_idx: torch.Tensor,
    *,
    margin: float = DEFAULT_MARGIN,
) -> torch.Tensor:
    """LLP margin ranking over unordered context pairs within each anchor group.

    Teacher and student logits are converted to probabilities. ``margin`` is
    both the strict teacher-probability tie band and the hinge margin. Tied
    pairs retain their constant-margin loss, as in the LLP reference.
    """
    order = torch.argsort(group_idx)
    sorted_groups = group_idx[order]
    _, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
    pair_counts = counts.square()
    pair_group = torch.repeat_interleave(
        torch.arange(counts.numel(), device=group_idx.device), pair_counts
    )
    group_offsets = torch.cumsum(counts, dim=0) - counts
    pair_offsets = torch.cumsum(pair_counts, dim=0) - pair_counts
    relative = torch.arange(pair_counts.sum(), device=group_idx.device) - torch.repeat_interleave(
        pair_offsets, pair_counts
    )
    group_width = counts[pair_group]
    left = order[
        group_offsets[pair_group] + torch.div(relative, group_width, rounding_mode="floor")
    ]
    right = order[group_offsets[pair_group] + torch.remainder(relative, group_width)]
    teacher_prob = torch.sigmoid(teacher_logit.to(dtype=student_logit.dtype))
    student_prob = torch.sigmoid(student_logit)
    teacher_diff = teacher_prob[left] - teacher_prob[right]
    target = torch.where(
        teacher_diff > margin,
        torch.ones_like(teacher_diff),
        torch.where(teacher_diff < -margin, -torch.ones_like(teacher_diff), 0.0),
    )
    student_diff = student_prob[left] - student_prob[right]
    return _masked_mean(F.relu(-target * student_diff + margin), left < right)


def kd_dist_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    group_idx: torch.Tensor,
) -> torch.Tensor:
    """LLP mean per-anchor KL between context-probability distributions.

    Each anchor's student and teacher logits are first converted to
    probabilities, then softmaxed across contexts at the fixed reference
    temperature of one. KL is summed over contexts and averaged over anchors.
    """
    if student_logit.numel() == 0:
        return student_logit.sum() * 0.0
    dtype = student_logit.dtype
    device = student_logit.device
    _, inverse = torch.unique(group_idx, return_inverse=True)
    n_groups = int(inverse.max().item()) + 1
    student_prob = torch.sigmoid(student_logit)
    teacher_prob = torch.sigmoid(teacher_logit.to(dtype=dtype))

    max_student = torch.full((n_groups,), _NEG_SENTINEL, dtype=dtype, device=device)
    max_student.scatter_reduce_(0, inverse, student_prob, reduce="amax", include_self=True)
    max_teacher = torch.full((n_groups,), _NEG_SENTINEL, dtype=dtype, device=device)
    max_teacher.scatter_reduce_(0, inverse, teacher_prob, reduce="amax", include_self=True)

    exp_student = torch.exp(student_prob - max_student[inverse])
    exp_teacher = torch.exp(teacher_prob - max_teacher[inverse])
    sum_student = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(
        0, inverse, exp_student
    )
    sum_teacher = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(
        0, inverse, exp_teacher
    )
    student_prob = exp_student / sum_student[inverse]
    teacher_prob = exp_teacher / sum_teacher[inverse]
    row_kl = teacher_prob * (
        torch.log(teacher_prob.clamp_min(_EPS)) - torch.log(student_prob.clamp_min(_EPS))
    )
    group_kl = torch.zeros(n_groups, dtype=dtype, device=device).scatter_add_(0, inverse, row_kl)
    return group_kl.mean()


def kd_gram_loss(
    student_rep: torch.Tensor,
    teacher_rep: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SPKD-style (Tung & Mori 2019) cosine-Gram matching across every distinct pair of task rows.

    Rows sharing an endpoint remain eligible. A batch with fewer than two
    rows returns a differentiable zero.
    """
    student_norm = F.normalize(student_rep, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_rep.to(dtype=student_rep.dtype), p=2, dim=-1, eps=eps)
    gram_student = student_norm @ student_norm.transpose(0, 1)
    gram_teacher = teacher_norm @ teacher_norm.transpose(0, 1)
    n_rows = student_rep.shape[0]
    off_diagonal = ~torch.eye(n_rows, dtype=torch.bool, device=student_rep.device)
    return _masked_mean((gram_student - gram_teacher).square(), off_diagonal)


def kd_set_seed_loss(
    student_seeds: torch.Tensor,
    teacher_seeds: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gate A masked per-seed cosine distance for full-ego set students."""
    student_norm = F.normalize(student_seeds, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_seeds.to(dtype=student_seeds.dtype), p=2, dim=-1, eps=eps)
    per_row = (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean(dim=-1)
    return _masked_mean(per_row, mask.to(dtype=torch.bool))


def kd_set_gram_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    node_mask: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gate A masked within-set node-token cosine-Gram matching."""
    valid_nodes = node_mask.to(dtype=torch.bool)
    node_weight = valid_nodes.to(dtype=student_tokens.dtype).unsqueeze(-1)
    student_norm = F.normalize(student_tokens, p=2, dim=-1, eps=eps) * node_weight
    teacher_norm = (
        F.normalize(teacher_tokens.to(dtype=student_tokens.dtype), p=2, dim=-1, eps=eps)
        * node_weight
    )
    gram_student = student_norm @ student_norm.transpose(1, 2)
    gram_teacher = teacher_norm @ teacher_norm.transpose(1, 2)
    n_nodes = student_tokens.shape[1]
    off_diagonal = ~torch.eye(n_nodes, dtype=torch.bool, device=student_tokens.device)
    valid_entries = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1) & off_diagonal
    entry_weight = valid_entries.to(dtype=student_tokens.dtype)
    squared = (gram_student - gram_teacher).square() * entry_weight
    entry_counts = entry_weight.sum(dim=(1, 2))
    per_row = squared.sum(dim=(1, 2)) / entry_counts.clamp_min(1.0)
    live = mask.to(dtype=torch.bool) & (entry_counts > 0)
    return _masked_mean(per_row, live)


__all__ = [
    "DEFAULT_MARGIN",
    "kd_dist_loss",
    "kd_gram_loss",
    "kd_logit_loss",
    "kd_rank_loss",
    "kd_rep_loss",
    "kd_set_gram_loss",
    "kd_set_seed_loss",
    "kd_struct_loss",
]
