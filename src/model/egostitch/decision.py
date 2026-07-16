"""Module 4 (Stage-1 form): decision head over (s0, s1, s2) (spec Sec 5, Sec 13.1).

``p_ij = sigmoid(s0 + g_theta(s1, s2) · w)`` with ``s0`` a frozen cached B0
logit (spec Sec 13.10 — never trained through), ``g_theta = MLP_2``, and ``w``
a learned scalar initialized at 0.1. ``s3``/``s4`` join in Stages 2/3.

Pinned readings recorded here (left symbolic in Sec 5):

- ``tau_kappa``: learned softplus-parameterized scalar, init 1.0 (spec Sec 13.7).
- The ``s2`` AA variant divides by ``0.5·(log1p(d_hat_i) + log1p(d_hat_j))``
  (clamped away from 0) — an Adamic-Adar-style damping by the endpoint degree
  budgets; it is one of ``g_theta``'s learned inputs, so the exact
  normalization is disclosure, not a load-bearing constant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.layers import build_mlp, stable_log

_SOFTPLUS_UNIT_INIT = math.log(math.expm1(1.0))


class DecisionHead(nn.Module):
    """Fused Stage-1 edge-decision head.

    The s-channel gate consumes ``[s1, s2, s2_aa]`` (3 features); its output is
    scaled by the learned scalar ``w`` and added to the frozen ``s0`` logit.
    """

    def __init__(self, config: EgoStitchConfig) -> None:
        """Build the gate MLP and the learned scalars.

        Args:
            config: The pinned Stage-1 configuration.
        """
        super().__init__()
        self.gate = build_mlp(3, config.d_h, 1)
        self.w = nn.Parameter(torch.tensor(0.1))
        self.tau_kappa_raw = nn.Parameter(torch.tensor(_SOFTPLUS_UNIT_INIT))

    @property
    def tau_kappa(self) -> torch.Tensor:
        """The positive similarity temperature (softplus-parameterized, init 1)."""
        return F.softplus(self.tau_kappa_raw)

    def _membership(self, slots: SlotSet, other_proj: torch.Tensor) -> torch.Tensor:
        """One side of ``s1`` using the scale-safe cosine-distance kernel."""
        slot_direction = F.normalize(slots.h, p=2.0, dim=-1)
        other_direction = F.normalize(other_proj, p=2.0, dim=-1)
        diff = slot_direction - other_direction[:, None, :]
        kappa = -(diff**2).sum(dim=-1) / self.tau_kappa
        return torch.logsumexp(kappa + stable_log(slots.pi * slots.mult), dim=-1)

    def channels(
        self,
        slots_i: SlotSet,
        slots_j: SlotSet,
        plan: torch.Tensor,
        proj_i: torch.Tensor,
        proj_j: torch.Tensor,
        d_hat_i: torch.Tensor,
        d_hat_j: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the Stage-1 s-channels for a pair batch.

        Args:
            slots_i: Side-i slot set.
            slots_j: Side-j slot set.
            plan: Shape ``(B, K, K)`` Sinkhorn alignment plan.
            proj_i: Shape ``(B, d_p)`` projected side-i features (stop-grad
                applied by the caller per spec Sec 13.7).
            proj_j: Shape ``(B, d_p)`` projected side-j features.
            d_hat_i: Shape ``(B,)`` side-i density-normalized degree budgets.
            d_hat_j: Shape ``(B,)`` side-j degree budgets.

        Returns:
            ``{"s1", "s2", "s2_aa"}`` channel tensors, each ``(B,)``.
        """
        s1 = 0.5 * (self._membership(slots_i, proj_j) + self._membership(slots_j, proj_i))
        weighted = plan * slots_i.pi[:, :, None] * slots_j.pi[:, None, :]
        s2 = weighted.sum(dim=(1, 2))
        damping = torch.clamp(0.5 * (torch.log1p(d_hat_i) + torch.log1p(d_hat_j)), min=1e-3)
        return {"s1": s1, "s2": s2, "s2_aa": s2 / damping}

    def fuse(self, s0: torch.Tensor, channels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse the frozen ``s0`` with the gated s-channels into edge logits."""
        stacked = torch.stack([channels["s1"], channels["s2"], channels["s2_aa"]], dim=-1)
        gated: torch.Tensor = self.gate(stacked)
        return s0 + gated.squeeze(-1) * self.w

    def forward(
        self,
        s0: torch.Tensor,
        slots_i: SlotSet,
        slots_j: SlotSet,
        plan: torch.Tensor,
        proj_i: torch.Tensor,
        proj_j: torch.Tensor,
        d_hat_i: torch.Tensor,
        d_hat_j: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused edge logits for a non-self pair batch.

        Args:
            s0: Shape ``(B,)`` frozen cached B0 logits.
            slots_i: Side-i slot set.
            slots_j: Side-j slot set.
            plan: Shape ``(B, K, K)`` alignment plan.
            proj_i: Shape ``(B, d_p)`` projected side-i features.
            proj_j: Shape ``(B, d_p)`` projected side-j features.
            d_hat_i: Shape ``(B,)`` degree budgets.
            d_hat_j: Shape ``(B,)`` degree budgets.

        Returns:
            Shape ``(B,)`` edge logits.
        """
        channels = self.channels(slots_i, slots_j, plan, proj_i, proj_j, d_hat_i, d_hat_j)
        return self.fuse(s0, channels)

    def forward_self(
        self,
        s0: torch.Tensor,
        slots: SlotSet,
        proj: torch.Tensor,
        d_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Single-ego path for ``(u, u)`` queries (spec Sec 9.4.2, Sec 13.9).

        ``Pi = I`` on all slots; ``s1`` is the self-membership term;
        ``s2(u, u) = sum_k (pi^k)^2 · adj[k, k]``.

        Args:
            s0: Shape ``(B,)`` frozen cached B0 self-pair logits.
            slots: The node's slot set.
            proj: Shape ``(B, d_p)`` the node's own projected features.
            d_hat: Shape ``(B,)`` degree budgets.

        Returns:
            Shape ``(B,)`` self-loop logits.
        """
        channels = self.self_channels(slots, proj, d_hat)
        return self.fuse(s0, channels)

    def self_channels(
        self,
        slots: SlotSet,
        proj: torch.Tensor,
        d_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the three decision channels for the registered self path."""
        s1 = self._membership(slots, proj)
        diag = torch.diagonal(slots.adj, dim1=-2, dim2=-1)
        s2 = (slots.pi**2 * diag).sum(dim=-1)
        damping = torch.clamp(torch.log1p(d_hat), min=1e-3)
        return {"s1": s1, "s2": s2, "s2_aa": s2 / damping}
