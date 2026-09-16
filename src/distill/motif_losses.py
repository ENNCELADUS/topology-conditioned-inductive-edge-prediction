"""``L_slot``, the motif-prompt per-family order-statistics loss.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``
section 7.3. Four multisets are supervised per row -- the 8 wedge products, the
64 bridge path products, the 16 raw attachment weights and the 64 raw interior
weights -- each through a descending sort and a Huber term. Sibling of
`src.distill.struct_losses`, where the other training losses live.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.data.motif_template import slot_profiles

_TERMS: tuple[tuple[str, str], ...] = (
    ("p", "beta_p"),
    ("q", "beta_q"),
    ("a", "beta_a"),
    ("b", "beta_i"),
)


def slot_loss_rows(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Per-row ``L_slot``: descending-sorted Huber over four per-family multisets.

    Path products, not raw closure weights, are supervised: the 16 closure edge
    weights as one multiset are invariant to moving weight between the two sides
    of a wedge, so the products are what pins the wedge mass. Because
    ``wedge_mass`` and ``bridge_mass`` are sums over the supervised multisets and
    a sum is permutation-invariant, matching sorted products pins both count-head
    quantities exactly. The two product terms route gradient to both (or all
    three) edges of each motif; the two raw terms keep gradient flowing to an
    edge whose path product vanishes because a *different* edge of that path is
    zero. The raw interior term is not decoration: for ``q = a b c`` the interior
    gradient is attenuated quadratically by small attachments, and the attachment
    term contributes nothing to it (spec section 7.3).

    Sorting is a permutation, so the loss is invariant to within-role slot
    permutation and to the endpoint swap, consistent with the slot randomisation
    of spec section 3. Rows are independent, so a caller masks self rows out.

    Args:
        predicted: ``(B, 96)`` predicted edge weights.
        target: ``(B, 96)`` compiled edge weights; detached here.
        beta_p: Weight of the wedge-product term.
        beta_q: Weight of the bridge-path-product term.
        beta_a: Weight of the raw attachment term.
        beta_i: Weight of the raw interior term.
        huber_delta: Huber transition point.

    Returns:
        ``(B,)`` per-row losses, in float32.

    Raises:
        ValueError: If the two tables disagree in shape.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and target {tuple(target.shape)} must match"
        )
    parts = slot_profiles(predicted)
    truth = slot_profiles(target.detach())
    betas = {"beta_p": beta_p, "beta_q": beta_q, "beta_a": beta_a, "beta_i": beta_i}
    total = predicted.new_zeros(predicted.size(0), dtype=torch.float32)
    for key, name in _TERMS:
        scale = betas[name]
        if scale == 0.0:
            continue
        ours = parts[key].sort(dim=1, descending=True).values
        theirs = truth[key].sort(dim=1, descending=True).values
        term = F.huber_loss(ours, theirs, delta=huber_delta, reduction="none").mean(dim=1)
        total = total + scale * term
    return total


def slot_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_p: float,
    beta_q: float,
    beta_a: float,
    beta_i: float,
    huber_delta: float,
) -> torch.Tensor:
    """Batch-mean ``L_slot``.

    Args:
        predicted: ``(B, 96)`` predicted edge weights.
        target: ``(B, 96)`` compiled edge weights.
        beta_p: Weight of the wedge-product term.
        beta_q: Weight of the bridge-path-product term.
        beta_a: Weight of the raw attachment term.
        beta_i: Weight of the raw interior term.
        huber_delta: Huber transition point.

    Returns:
        The scalar mean of `slot_loss_rows`.
    """
    return slot_loss_rows(
        predicted,
        target,
        beta_p=beta_p,
        beta_q=beta_q,
        beta_a=beta_a,
        beta_i=beta_i,
        huber_delta=huber_delta,
    ).mean()
