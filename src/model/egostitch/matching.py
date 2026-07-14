"""Hungarian slot-target matching, Stage-1 cost (spec Sec 2, Sec 13.3).

Two-pass compound cost without the code-agreement term (weights uniformly
rescaled x7/6; ``linear_sum_assignment`` is invariant to the rescale — recorded
for exactness only):

``C = (7/6)·||h - proj(x_v)||^2 + (7/24)·|deg_bucket(k) - deg_bucket(v)|
+ (7/24)·overlap_penalty``  (overlap only on the second pass).

``deg_bucket`` is pinned (spec Sec 13.3) to log2 multiplicity buckets
``{[1,2), [2,4), [4,8), [8,16), [16,32]}``. Costs are built on GPU; the
assignment runs on CPU under ``no_grad`` and is **constant in the backward
pass** (standard DETR practice).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.model.egostitch.imagine import SlotSet

_W_FEAT = 7.0 / 6.0
_W_BUCKET = 7.0 / 24.0
_W_OVERLAP = 7.0 / 24.0
_N_BUCKETS = 5


class Assignment:
    """One batch of Hungarian slot-target assignments (constant in backward).

    Attributes:
        slot_idx: Per-node int64 arrays of assigned slot indices.
        target_idx: Per-node int64 arrays of assigned target indices, aligned
            with `slot_idx`.
    """

    def __init__(
        self, slot_idx: list[NDArray[np.int64]], target_idx: list[NDArray[np.int64]]
    ) -> None:
        if len(slot_idx) != len(target_idx):
            raise ValueError("slot_idx and target_idx must have equal batch length")
        self.slot_idx = slot_idx
        self.target_idx = target_idx

    def __len__(self) -> int:
        """Return the batch size."""
        return len(self.slot_idx)

    def __eq__(self, other: object) -> bool:
        """Elementwise equality of both index lists (for determinism tests)."""
        if not isinstance(other, Assignment):
            return NotImplemented
        return len(self) == len(other) and all(
            np.array_equal(a, b) and np.array_equal(c, d)
            for a, b, c, d in zip(
                self.slot_idx, other.slot_idx, self.target_idx, other.target_idx, strict=True
            )
        )

    def __hash__(self) -> int:  # pragma: no cover - identity hashing suffices
        """Identity hash (assignments are never dict keys in hot paths)."""
        return id(self)


def deg_bucket_index(m: torch.Tensor) -> torch.Tensor:
    """Log2 multiplicity bucket index (spec Sec 13.3), in ``0..4``."""
    clamped = torch.clamp(m, min=1.0)
    return torch.clamp(torch.floor(torch.log2(clamped)), max=float(_N_BUCKETS - 1))


def base_match_cost(
    slots: SlotSet,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
) -> torch.Tensor:
    """First-pass matching cost (feature + degree-bucket terms).

    Args:
        slots: The generated slot set.
        target_proj: Shape ``(B, T, d_p)`` **stop-gradient** projected target
            features (spec Sec 13.7).
        target_mult: Shape ``(B, T)`` target multiplicity labels.

    Returns:
        Shape ``(B, K, T)`` cost tensor (differentiable through the slot side;
        used only to *derive* the constant assignment).
    """
    diff = slots.h[:, :, None, :] - target_proj[:, None, :, :]
    feat = (diff**2).sum(dim=-1)
    bucket_slot = deg_bucket_index(slots.mult)
    bucket_target = deg_bucket_index(target_mult)
    bucket = (bucket_slot[:, :, None] - bucket_target[:, None, :]).abs()
    return _W_FEAT * feat + _W_BUCKET * bucket


def overlap_penalty(
    slots: SlotSet, target_adj: torch.Tensor, assignment: Assignment
) -> torch.Tensor:
    """Second-pass overlap term (spec Sec 2): slot-adjacency vs target-adjacency.

    For candidate pairing ``(k, v)``: mean over the first-pass matched pairs
    ``(k', v')`` (excluding ``k`` itself) of ``|adj[k, k'] - A[v, v']|``.

    Args:
        slots: The generated slot set.
        target_adj: Shape ``(B, T, T)`` binary adjacency among targets.
        assignment: The first-pass assignment.

    Returns:
        Shape ``(B, K, T)`` penalty tensor (zeros for nodes with < 2 matches).
    """
    batch, k = slots.adj.shape[0], slots.adj.shape[1]
    t = target_adj.shape[1]
    penalty = slots.adj.new_zeros((batch, k, t))
    for b in range(batch):
        s_idx = torch.from_numpy(assignment.slot_idx[b]).to(slots.adj.device)
        t_idx = torch.from_numpy(assignment.target_idx[b]).to(slots.adj.device)
        if s_idx.numel() < 2:
            continue
        # (K, M): each candidate slot k vs matched slots k'; (T, M): each
        # candidate target v vs matched targets v'.
        adj_rows = slots.adj[b][:, s_idx]
        target_rows = target_adj[b][:, t_idx]
        disagreement = (adj_rows[:, None, :] - target_rows[None, :, :]).abs()
        # Exclude the (k == k') self column per candidate slot.
        self_mask = s_idx[None, :] == torch.arange(k, device=s_idx.device)[:, None]  # (K, M)
        weights = (~self_mask).float()[:, None, :]
        denom = torch.clamp(weights.sum(dim=-1), min=1.0)
        penalty[b] = (disagreement * weights).sum(dim=-1) / denom
    return penalty


def hungarian_assign(cost: torch.Tensor, target_mask: torch.Tensor) -> Assignment:
    """Solve the per-node assignment on CPU (one transfer, ``no_grad``).

    Args:
        cost: Shape ``(B, K, T)`` matching costs.
        target_mask: Shape ``(B, T)`` booleans; ``False`` targets are padding.

    Returns:
        The `Assignment` (constant in the backward pass).
    """
    with torch.no_grad():
        cost_np = cost.detach().float().cpu().numpy()
        mask_np = target_mask.detach().cpu().numpy()
    slot_idx: list[NDArray[np.int64]] = []
    target_idx: list[NDArray[np.int64]] = []
    for b in range(cost_np.shape[0]):
        valid = np.flatnonzero(mask_np[b])
        if valid.size == 0:
            slot_idx.append(np.empty(0, dtype=np.int64))
            target_idx.append(np.empty(0, dtype=np.int64))
            continue
        rows, cols = linear_sum_assignment(cost_np[b][:, valid])
        slot_idx.append(rows.astype(np.int64))
        target_idx.append(valid[cols].astype(np.int64))
    return Assignment(slot_idx, target_idx)


def match_slots(
    slots: SlotSet,
    *,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
    target_adj: torch.Tensor,
    target_mask: torch.Tensor,
) -> Assignment:
    """Run the pinned two-pass Hungarian matching (spec Sec 2, Sec 13.3).

    Pass 1 uses the feature + degree-bucket terms; pass 2 adds the overlap
    penalty computed against pass 1's matched pairs and re-solves.

    Args:
        slots: The generated slot set.
        target_proj: Shape ``(B, T, d_p)`` stop-gradient projected targets.
        target_mult: Shape ``(B, T)`` target multiplicity labels.
        target_adj: Shape ``(B, T, T)`` adjacency among targets.
        target_mask: Shape ``(B, T)`` target validity mask.

    Returns:
        The final `Assignment`.
    """
    base = base_match_cost(slots, target_proj, target_mult)
    first = hungarian_assign(base, target_mask)
    penalty = overlap_penalty(slots, target_adj, first)
    return hungarian_assign(base + _W_OVERLAP * penalty, target_mask)
