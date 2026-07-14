"""Shared building blocks for the EgoStitch modules.

Implements the spec Sec 0 convention: ``MLP_k(a -> b)`` = k-layer MLP from dim
``a`` to dim ``b`` with GELU activations and LayerNorm. The hidden width (left
free by the spec's convention) is pinned to ``d_h`` by the callers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, *, layers: int = 2) -> nn.Sequential:
    """Build the spec's ``MLP_k(in_dim -> out_dim)`` (GELU, LayerNorm).

    Args:
        in_dim: Input dimension ``a``.
        hidden_dim: Width of every intermediate layer.
        out_dim: Output dimension ``b`` (no activation/norm on the output).
        layers: ``k`` — the number of Linear layers (>= 1).

    Returns:
        The `nn.Sequential` module.

    Raises:
        ValueError: If ``layers < 1``.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    if layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers - 1):
        blocks += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


def stable_log(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """``log(clamp(x, eps))`` — the pinned degenerate-marginal guard (spec Sec 13)."""
    return torch.log(torch.clamp(x, min=eps))
