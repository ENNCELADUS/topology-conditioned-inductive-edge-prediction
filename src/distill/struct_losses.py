"""Structural-stream topology terms over one sampled subgraph's logit matrix.

Every function takes symmetric ``n x n`` float32 tensors ``logits``, ``target``
(0/1 adjacency) and ``mask`` (0/1 legal-pair mask, zero diagonal) and reduces
only where ``mask == 1``. Each returns a differentiable zero when its
reduction set is empty so DDP sees every parameter on every step.
Spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md`` section 5.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.distill.struct_config import StructConfig

EPSILON = 1.0e-8


def _upper(mask: torch.Tensor) -> torch.Tensor:
    return torch.triu(mask, diagonal=1) > 0


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def _motif_counts(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    two_hop = adjacency @ adjacency
    return adjacency * two_hop, (1.0 - adjacency) * two_hop


def struct_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Positive-weighted BCE over legal upper-triangle pairs (the task loss's form)."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    z = logits[sel]
    y = target[sel]
    smoothed = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    per_pair = F.binary_cross_entropy_with_logits(z, smoothed, reduction="none")
    weights = 1.0 + (positive_weight - 1.0) * y
    return (weights * per_pair).sum() / weights.sum()


def struct_gs(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GRAND's soft graph-similarity loss: ``sum|p - a| / (sum p + sum a + eps)``."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits)[sel]
    a = target[sel]
    return (p - a).abs().sum() / (p.sum() + a.sum() + EPSILON)


def struct_rd(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Two-sided relative-density loss: SmoothL1 of ``log(sum p / sum a)``."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits)[sel]
    a = target[sel]
    log_ratio = torch.log((p.sum() + EPSILON) / (a.sum() + EPSILON))
    return F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=huber_delta)


def _soft_histogram(values: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    scaled = (values.unsqueeze(-1) - centers.unsqueeze(0)) / sigma
    histogram = torch.exp(-0.5 * scaled.square()).sum(dim=0)
    return histogram / (histogram.sum() + EPSILON)


def struct_deg_mmd(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, sigma: float, bins: int
) -> torch.Tensor:
    """GRAND's Gaussian-TV degree-histogram MMD between soft and true in-subgraph degrees."""
    if not bool(mask.any()):
        return _zero_like(logits)
    n = logits.size(0)
    soft_degrees = (torch.sigmoid(logits) * mask).sum(dim=1)
    true_degrees = (target * mask).sum(dim=1)
    centers = torch.linspace(0.0, float(max(1, n - 1)), steps=max(2, bins), device=logits.device)
    tv = (
        0.5
        * (
            _soft_histogram(soft_degrees, centers, sigma)
            - _soft_histogram(true_degrees, centers, sigma)
        )
        .abs()
        .sum()
    )
    return 2.0 - 2.0 * torch.exp(-tv.square() / (2.0 * sigma * sigma))


def struct_degree(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Node-wise SmoothL1 of ``log1p`` soft vs true legal in-subgraph degree."""
    valid = mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return _zero_like(logits)
    soft_degrees = (torch.sigmoid(logits) * mask).sum(dim=1)[valid]
    true_degrees = (target * mask).sum(dim=1)[valid]
    return F.smooth_l1_loss(torch.log1p(soft_degrees), torch.log1p(true_degrees), beta=huber_delta)


def struct_rank(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Per-anchor neighbour ranking: softplus((z_ik - z_ij + m)/T) over j in P_i, k in N_i."""
    positive = (target > 0) & (mask > 0)
    negative = (target == 0) & (mask > 0)
    valid = positive.unsqueeze(2) & negative.unsqueeze(1)  # [i, j, k]
    counts = valid.sum(dim=(1, 2))
    anchors = counts > 0
    if not bool(anchors.any()):
        return _zero_like(logits)
    diff = logits.unsqueeze(1) - logits.unsqueeze(2) + margin  # z_ik - z_ij + m
    per_pair = F.softplus(diff / temperature) * valid
    per_anchor = per_pair.sum(dim=(1, 2))[anchors] / counts[anchors]
    return per_anchor.mean()


def struct_motif(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, huber_delta: float
) -> torch.Tensor:
    """Closed/open two-hop count matching with a balanced mean over target classes."""
    sel = _upper(mask)
    if not bool(sel.any()):
        return _zero_like(logits)
    p = torch.sigmoid(logits) * mask
    a = target * mask
    closed_p, open_p = _motif_counts(p)
    closed_a, open_a = _motif_counts(a)
    errors = F.smooth_l1_loss(
        torch.log1p(closed_p[sel]), torch.log1p(closed_a[sel]), reduction="none", beta=huber_delta
    ) + F.smooth_l1_loss(
        torch.log1p(open_p[sel]), torch.log1p(open_a[sel]), reduction="none", beta=huber_delta
    )
    ca = closed_a[sel]
    oa = open_a[sel]
    classes = (ca > 0, (ca == 0) & (oa > 0), (ca == 0) & (oa == 0))
    means = [errors[cls].mean() for cls in classes if bool(cls.any())]
    return torch.stack(means).mean()


def struct_total(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: StructConfig,
    *,
    positive_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the active terms; returns the total and each raw term."""
    terms: dict[str, torch.Tensor] = {}
    for key in config.active_weights:
        if key == "bce":
            terms[key] = struct_bce(
                logits,
                target,
                mask,
                positive_weight=positive_weight,
                label_smoothing=label_smoothing,
            )
        elif key == "gs":
            terms[key] = struct_gs(logits, target, mask)
        elif key == "rd":
            terms[key] = struct_rd(logits, target, mask, huber_delta=config.huber_delta)
        elif key == "deg_mmd":
            terms[key] = struct_deg_mmd(
                logits, target, mask, sigma=config.mmd_sigma, bins=config.mmd_bins
            )
        elif key == "rank":
            terms[key] = struct_rank(
                logits,
                target,
                mask,
                margin=config.rank_margin,
                temperature=config.rank_temperature,
            )
        elif key == "degree":
            terms[key] = struct_degree(logits, target, mask, huber_delta=config.huber_delta)
        elif key == "motif":
            terms[key] = struct_motif(logits, target, mask, huber_delta=config.huber_delta)
        else:  # pragma: no cover - StructConfig validates the key set
            raise KeyError(key)
    total = _zero_like(logits)
    for key, term in terms.items():
        total = total + config.active_weights[key] * term
    return total, terms


@torch.no_grad()
def hard_struct_errors(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float
) -> dict[str, float]:
    """Mean absolute per-node degree / triangle / open-wedge errors at a hard threshold."""
    predicted = (logits > threshold).float() * mask
    truth = target * mask

    def _node_counts(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degrees = adjacency.sum(dim=1)
        triangles = torch.diagonal(adjacency @ adjacency @ adjacency) / 2.0
        wedges = degrees * (degrees - 1.0) / 2.0 - triangles
        return degrees, triangles, wedges

    p_deg, p_tri, p_wedge = _node_counts(predicted)
    t_deg, t_tri, t_wedge = _node_counts(truth)
    return {
        "hard_degree_mae": float((p_deg - t_deg).abs().mean().item()),
        "hard_triangle_mae": float((p_tri - t_tri).abs().mean().item()),
        "hard_wedge_mae": float((p_wedge - t_wedge).abs().mean().item()),
    }


__all__ = [
    "EPSILON",
    "hard_struct_errors",
    "struct_bce",
    "struct_deg_mmd",
    "struct_degree",
    "struct_gs",
    "struct_motif",
    "struct_rank",
    "struct_rd",
    "struct_total",
]
