"""Module 3a: Stitch — unbalanced Sinkhorn slot alignment (spec Sec 3, Sec 13.3).

Retained in Stage 1 because the closure channel ``s2`` consumes the alignment
plan (spec Sec 13.1). Stage-1 cost drops the code term (weights x5/4):

``C = (5/4)·||h_i - h_j||^2 + (5/16)·|pi_i - pi_j|``

Numerics: log-domain, fixed iteration count (determinism), computed in an
fp32 island under autocast (bf16-safe), marginals ``a ∝ pi·m`` clamped at
``1e-8`` (degenerate all-``pi``-zero inputs stay finite). Unrolled and
differentiable.
"""

from __future__ import annotations

import torch

from src.model.egostitch.layers import stable_log

_W_FEAT = 5.0 / 4.0
_W_PI = 5.0 / 16.0


def stitch_cost(
    h_i: torch.Tensor, h_j: torch.Tensor, pi_i: torch.Tensor, pi_j: torch.Tensor
) -> torch.Tensor:
    """Stage-1 OT cost matrix (spec Sec 13.3).

    Args:
        h_i: Shape ``(B, K, d_p)`` side-i slot embeddings.
        h_j: Shape ``(B, K, d_p)`` side-j slot embeddings.
        pi_i: Shape ``(B, K)`` side-i existence probabilities.
        pi_j: Shape ``(B, K)`` side-j existence probabilities.

    Returns:
        Shape ``(B, K, K)`` cost tensor.
    """
    diff = h_i[:, :, None, :] - h_j[:, None, :, :]
    feat = (diff**2).sum(dim=-1)
    pi_gap = (pi_i[:, :, None] - pi_j[:, None, :]).abs()
    return _W_FEAT * feat + _W_PI * pi_gap


def sinkhorn_plan(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    pi_i: torch.Tensor,
    pi_j: torch.Tensor,
    m_i: torch.Tensor,
    m_j: torch.Tensor,
    *,
    eps: float = 0.1,
    iters: int = 20,
    tau: float = 1.0,
) -> torch.Tensor:
    """Compute the unbalanced-OT alignment plan ``Pi`` (spec Sec 3).

    Marginals ``a ∝ pi_i·m_i`` and ``b ∝ pi_j·m_j`` are normalized to unit
    mass and clamped at ``1e-8``. KL relaxation with strength `tau` gives the
    standard damped log-domain updates (damping ``phi = tau / (tau + eps)``);
    the plan is ``exp((f + g - C) / eps + log a + log b)`` — slots may be
    unmatched. Fixed `iters` iterations; fully differentiable (unrolled).

    Args:
        h_i: Shape ``(B, K, d_p)`` side-i slot embeddings.
        h_j: Shape ``(B, K, d_p)`` side-j slot embeddings.
        pi_i: Shape ``(B, K)`` side-i existence probabilities.
        pi_j: Shape ``(B, K)`` side-j existence probabilities.
        m_i: Shape ``(B, K)`` side-i multiplicities.
        m_j: Shape ``(B, K)`` side-j multiplicities.
        eps: Entropic regularizer.
        iters: Fixed iteration count.
        tau: Unbalanced KL relaxation strength.

    Returns:
        Shape ``(B, K, K)`` non-negative alignment plan.
    """
    # fp32 island: inputs must be promoted before either the cost or marginal
    # products are formed; casting their bf16 results afterward is too late.
    with torch.autocast(device_type=h_i.device.type, enabled=False):
        h_i32, h_j32 = h_i.float(), h_j.float()
        pi_i32, pi_j32 = pi_i.float(), pi_j.float()
        m_i32, m_j32 = m_i.float(), m_j.float()
        cost = stitch_cost(h_i32, h_j32, pi_i32, pi_j32)
        a = pi_i32 * m_i32
        b = pi_j32 * m_j32
        a = a / torch.clamp(a.sum(dim=-1, keepdim=True), min=1e-8)
        b = b / torch.clamp(b.sum(dim=-1, keepdim=True), min=1e-8)
        log_a = stable_log(a)
        log_b = stable_log(b)

        phi = tau / (tau + eps)
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
        for _ in range(iters):
            f = -phi * eps * torch.logsumexp(
                (g[:, None, :] - cost) / eps + log_b[:, None, :], dim=2
            )
            g = -phi * eps * torch.logsumexp(
                (f[:, :, None] - cost) / eps + log_a[:, :, None], dim=1
            )
        return torch.exp(
            (f[:, :, None] + g[:, None, :] - cost) / eps
            + log_a[:, :, None]
            + log_b[:, None, :]
        )
