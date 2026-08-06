"""Stage-1 loss tree (spec Sec 7 subset, Sec 13.5-13.6 and Sec 14.4.1).

``L = L_edge + 0.5·L_real + 0.1·L_ssl + 1.0·L_recon`` with

``L_recon(egostitch) = 1.0·L_feat + 0.5·L_exist + 0.25·L_mult
+ 0.5·L_deg + 0.5·L_slotadj + 0.25·L_gate``
``L_recon(egostitch_e2e) = L_recon(egostitch) + 0.25·L_ptr
+ 0.5·L_align + 0.1·L_div + 0.25·L_rel``
``L_real  = (2/3)·ED(ego stats) + (1/3)·ED(random-GIN embeddings)``
``L_ssl   = 0.5·feature-noise consistency + 0.5·pool-resample consistency``

Dropped with their mechanisms: KL, L_VQ, L_codestats, L_entropy, L_BP, L_joint.
Pinned instantiation notes: the multiplicity NLL is the unit-variance lognormal
NLL (squared log error); the target sides of ``L_feat`` and the matching cost
are stop-gradient (spec Sec 13.7).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.assemble import Assignment
from src.model.egostitch.layers import stable_log

if TYPE_CHECKING:
    # Deferred to break the losses.py <-> imagine.py import cycle:
    # `EgoStitchStage1` (imagine.py) imports this module at runtime (it
    # instantiates `RandomGIN` and calls `degree_nll`/`recon_losses`/etc.),
    # while this module's own functions only ever use `SlotSet`/`DenoiseSlots`
    # in type annotations -- never as a runtime value -- so, with
    # `from __future__ import annotations`, importing them is safe to defer to
    # type-checking only.
    from src.model.egostitch.generator.imagine import DenoiseSlots, SlotSet

_SIGMA_MIN = 1e-3
LossFamily = Literal["egostitch", "egostitch_e2e"]


def _masked_global_mean(
    per_row: torch.Tensor,
    real_row_mask: torch.Tensor,
    *,
    world_size: int = 1,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return a padding-aware global mean with DDP-gradient scaling.

    The all-reduced denominator excludes `_edge_tensors` filler rows. Multiplying
    the local numerator by ``world_size`` compensates for DDP's subsequent
    gradient average, matching the §13.19.1 reduction class.
    """
    if per_row.ndim != 1 or per_row.shape != real_row_mask.shape:
        raise ValueError("per_row and real_row_mask must be identical rank-1 tensors")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    with torch.autocast(device_type=per_row.device.type, enabled=False):
        values = per_row.float()
        mask = real_row_mask.float()
        local_denominator = mask.sum().detach()
        if all_reduce_sum is not None:
            denominator = all_reduce_sum(local_denominator)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            denominator = local_denominator.clone()
            torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        else:
            denominator = local_denominator
        if not bool(torch.isfinite(denominator)):
            raise RuntimeError("global masked-mean denominator must be finite")
        if float(denominator) <= 0.0:
            return values.sum() * 0.0
        return world_size * (values * mask).sum() / denominator


def alignment_teacher_cells(
    assignment_a: Assignment,
    assignment_b: Assignment,
    *,
    target_ids_a: torch.Tensor,
    target_ids_b: torch.Tensor,
    slots: int,
) -> torch.Tensor:
    """Build teacher cells from endpoint Hungarian matches (spec §14.4.1).

    A cell is true exactly when its two matched slots were generated from the
    same real target-neighbor identity. Grounding pools are deliberately absent.
    """
    if target_ids_a.ndim != 2 or target_ids_b.ndim != 2:
        raise ValueError("target id tensors must have shape (B, T)")
    batch = target_ids_a.shape[0]
    if target_ids_b.shape[0] != batch or len(assignment_a) != batch or len(assignment_b) != batch:
        raise ValueError("assignments and target id tensors must share a batch dimension")
    if slots <= 0:
        raise ValueError("slots must be positive")
    cells = torch.zeros(batch, slots, slots, dtype=torch.bool, device=target_ids_a.device)
    for row in range(batch):
        slot_a = torch.from_numpy(assignment_a.slot_idx[row]).to(target_ids_a.device)
        slot_b = torch.from_numpy(assignment_b.slot_idx[row]).to(target_ids_a.device)
        target_a = torch.from_numpy(assignment_a.target_idx[row]).to(target_ids_a.device)
        target_b = torch.from_numpy(assignment_b.target_idx[row]).to(target_ids_a.device)
        if slot_a.numel() == 0 or slot_b.numel() == 0:
            continue
        ids_a = target_ids_a[row, target_a]
        ids_b = target_ids_b[row, target_b]
        shared = (ids_a[:, None] == ids_b[None, :]) & (ids_a[:, None] >= 0)
        cells[row][slot_a[:, None], slot_b[None, :]] = shared
    return cells


def alignment_loss(
    log_plan: torch.Tensor,
    teacher_cells: torch.Tensor,
    *,
    positive_real_mask: torch.Tensor,
    world_size: int = 1,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Row/column-conditional `L_align` over teacher-bearing positive rows."""
    if log_plan.ndim != 3 or log_plan.shape[-1] != log_plan.shape[-2]:
        raise ValueError("log_plan must have shape (B, K, K)")
    if teacher_cells.shape != log_plan.shape or teacher_cells.dtype != torch.bool:
        raise ValueError("teacher_cells must be a boolean tensor matching log_plan")
    if positive_real_mask.shape != log_plan.shape[:1]:
        raise ValueError("positive_real_mask must have shape (B,)")
    with torch.autocast(device_type=log_plan.device.type, enabled=False):
        log_plan_fp32 = log_plan.float()
        log_row_mass = torch.logsumexp(log_plan_fp32, dim=2, keepdim=True)
        log_col_mass = torch.logsumexp(log_plan_fp32, dim=1, keepdim=True)
        conditional = 0.5 * (log_plan_fp32 - log_row_mass + log_plan_fp32 - log_col_mass)
        teacher_fp32 = teacher_cells.float()
        teacher_count = teacher_fp32.sum(dim=(1, 2))
        per_row = -(
            torch.where(teacher_cells, conditional, torch.zeros_like(conditional)).sum(dim=(1, 2))
            / teacher_count.clamp_min(1.0)
        )
        effective_mask = positive_real_mask.float() * (teacher_count > 0).float()
        return _masked_global_mean(
            per_row,
            effective_mask,
            world_size=world_size,
            all_reduce_sum=all_reduce_sum,
        )


def relational_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    real_row_mask: torch.Tensor,
    *,
    world_size: int = 1,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Huber `L_rel` for both relational targets on every real edge-stream row."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must both have shape (B, 2)")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        per_row = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(dim=1)
        return _masked_global_mean(
            per_row,
            real_row_mask,
            world_size=world_size,
            all_reduce_sum=all_reduce_sum,
        )


def _probability_bce(
    input: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    """Evaluate probability-space BCE in fp32 outside mixed-precision autocast."""
    with torch.autocast(device_type=input.device.type, enabled=False):
        return F.binary_cross_entropy(input.float(), target.float(), reduction=reduction)


def _positive_weighted_probability_bce(
    input: torch.Tensor, target: torch.Tensor, *, pos_weight: float
) -> torch.Tensor:
    """Evaluate positive-class-weighted probability BCE in an fp32 island."""
    with torch.autocast(device_type=input.device.type, enabled=False):
        probability = input.float().clamp(1e-6, 1.0 - 1e-6)
        labels = target.float()
        return -(
            pos_weight * labels * torch.log(probability)
            + (1.0 - labels) * torch.log1p(-probability)
        ).mean()


# --------------------------------------------------------------------------- reconstruction


def recon_losses(
    slots: SlotSet,
    assignment: Assignment,
    *,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
    target_adj: torch.Tensor,
    target_in_pool: torch.Tensor,
    target_pool_index: torch.Tensor,
    config: EgoStitchConfig,
    family: LossFamily = "egostitch",
) -> dict[str, torch.Tensor]:
    """Matched-slot reconstruction losses (spec Sec 13.5 and Sec 14.4.1).

    Args:
        slots: The generated slot set.
        assignment: The constant Hungarian assignment.
        target_proj: Shape ``(B, T, d_p)`` **stop-gradient** projected targets.
        target_mult: Shape ``(B, T)`` multiplicity labels.
        target_adj: Shape ``(B, T, T)`` adjacency among targets.
        target_in_pool: Shape ``(B, T)`` booleans — target is in the node's
            grounding pool ``G(u)`` (Stage-1 ``L_gate`` supervision).
        target_pool_index: Shape ``(B, T)`` ordered indices into ``G(u)``, or
            ``-1`` for out-of-pool targets/padding (``L_ptr`` supervision).
        config: Supplies the registered temperatures and gate positive weight.
        family: Selects the frozen Stage-1 or rev-3.1 E2E decomposition.

    Returns:
        Seven scalar component tensors through Task 5 of the rev-3.1 repair.
    """
    device = slots.h.device
    feat_terms: list[torch.Tensor] = []
    mult_terms: list[torch.Tensor] = []
    slotadj_terms: list[torch.Tensor] = []
    gate_logit_terms: list[torch.Tensor] = []
    gate_label_terms: list[torch.Tensor] = []
    pointer_terms: list[torch.Tensor] = []
    pointer_label_terms: list[torch.Tensor] = []
    matched_mask = torch.zeros_like(slots.pi, dtype=torch.bool)

    for b in range(len(assignment)):
        s_idx = torch.from_numpy(assignment.slot_idx[b]).to(device)
        t_idx = torch.from_numpy(assignment.target_idx[b]).to(device)
        if s_idx.numel() == 0:
            continue
        matched_mask[b, s_idx] = True
        feat_terms.append(
            F.smooth_l1_loss(slots.h[b, s_idx], target_proj[b, t_idx].detach(), reduction="mean")
        )
        log_gap = torch.log(slots.mult[b, s_idx]) - stable_log(target_mult[b, t_idx])
        mult_terms.append(0.5 * (log_gap**2).mean())
        if s_idx.numel() >= 2:
            adj_true = target_adj[b][t_idx][:, t_idx]
            off_diag = ~torch.eye(s_idx.numel(), dtype=torch.bool, device=device)
            if family == "egostitch_e2e":
                adj_pred = slots.adj_logits[b][s_idx][:, s_idx]
                with torch.autocast(device_type=adj_pred.device.type, enabled=False):
                    slotadj = F.binary_cross_entropy_with_logits(
                        adj_pred[off_diag].float() / config.tau_adj,
                        adj_true[off_diag].float(),
                    )
            else:
                adj_pred = slots.adj[b][s_idx][:, s_idx]
                slotadj = _probability_bce(
                    torch.clamp(adj_pred[off_diag], 1e-6, 1.0 - 1e-6),
                    adj_true[off_diag],
                )
            slotadj_terms.append(slotadj)
        gate_logit_terms.append(slots.gate[b, s_idx])
        gate_label_terms.append(target_in_pool[b, t_idx].float())
        if family == "egostitch_e2e":
            pointer_valid = target_in_pool[b, t_idx]
            if bool(pointer_valid.any()):
                pointer_terms.append(slots.pointer[b, s_idx[pointer_valid]])
                pointer_label_terms.append(target_pool_index[b, t_idx[pointer_valid]])

    zero = slots.h.sum() * 0.0
    feat = torch.stack(feat_terms).mean() if feat_terms else zero
    mult = torch.stack(mult_terms).mean() if mult_terms else zero
    slotadj = torch.stack(slotadj_terms).mean() if slotadj_terms else zero
    if gate_logit_terms and family == "egostitch_e2e":
        gate = _positive_weighted_probability_bce(
            torch.cat(gate_logit_terms),
            torch.cat(gate_label_terms),
            pos_weight=config.l_gate_pos_weight,
        )
    elif gate_logit_terms:
        gate = _probability_bce(
            torch.clamp(torch.cat(gate_logit_terms), 1e-6, 1.0 - 1e-6),
            torch.cat(gate_label_terms),
        )
    else:
        gate = zero

    if family == "egostitch_e2e" and pointer_terms:
        pointer = torch.cat(pointer_terms)
        pointer_labels = torch.cat(pointer_label_terms).long()
        if bool(((pointer_labels < 0) | (pointer_labels >= pointer.shape[-1])).any()):
            raise ValueError("in-pool target has invalid grounding-pool index")
        with torch.autocast(device_type=pointer.device.type, enabled=False):
            ptr = F.nll_loss(
                torch.log(pointer.float().clamp(min=1e-8)), pointer_labels, reduction="mean"
            )
    else:
        ptr = slots.pointer.sum() * 0.0

    # Existence BCE, ∅-balanced: matched and unmatched slots contribute with
    # equal class weight regardless of the match rate.
    pi = torch.clamp(slots.pi, 1e-6, 1.0 - 1e-6)
    labels = matched_mask.float()
    per_slot = _probability_bce(pi, labels, reduction="none")
    n_pos = labels.sum()
    n_neg = (1.0 - labels).sum()
    pos_term = (per_slot * labels).sum() / torch.clamp(n_pos, min=1.0)
    neg_term = (per_slot * (1.0 - labels)).sum() / torch.clamp(n_neg, min=1.0)
    exist = 0.5 * pos_term + 0.5 * neg_term

    if family == "egostitch_e2e":
        with torch.autocast(device_type=slots.h.device.type, enabled=False):
            normalized = F.normalize(slots.h.float(), dim=-1)
            cosine = torch.bmm(normalized, normalized.transpose(1, 2))
            distinct = ~torch.eye(
                slots.h.shape[1], dtype=torch.bool, device=slots.h.device
            ).unsqueeze(0)
            eligible = distinct & ~(matched_mask[:, :, None] & matched_mask[:, None, :])
            if bool(eligible.any()):
                div = torch.relu(cosine[eligible] - config.tau_div).square().mean()
            else:
                div = slots.h.float().sum() * 0.0
    else:
        div = slots.h.sum() * 0.0

    return {
        "feat": feat,
        "exist": exist,
        "mult": mult,
        "slotadj": slotadj,
        "gate": gate,
        "ptr": ptr,
        "div": div,
    }


def denoise_losses(
    denoise: DenoiseSlots, *, target_proj: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Fixed-assignment losses of the denoising queries (spec Sec 2).

    Each denoise slot is supervised against the target it was initialized
    from (identity assignment): Huber on ``h`` and an existence BCE toward 1.

    Args:
        denoise: The denoise-slot outputs.
        target_proj: Shape ``(B, K_d, d_p)`` stop-gradient projections of the
            noised-source targets.
        mask: Shape ``(B, K_d)`` validity mask.

    Returns:
        ``{"feat", "exist"}`` scalar loss tensors.
    """
    zero = denoise.h.sum() * 0.0
    valid = mask.bool()
    if not bool(valid.any()):
        return {"feat": zero, "exist": zero}
    feat = F.smooth_l1_loss(denoise.h[valid], target_proj.detach()[valid], reduction="mean")
    exist = _probability_bce(
        torch.clamp(denoise.pi[valid], 1e-6, 1.0 - 1e-6),
        torch.ones_like(denoise.pi[valid]),
    )
    return {"feat": feat, "exist": exist}


def degree_nll(
    deg_mu: torch.Tensor, deg_log_sigma: torch.Tensor, true_degree: torch.Tensor
) -> torch.Tensor:
    """Lognormal degree NLL (spec Sec 1): ``log sigma + 0.5·((log d - mu)/sigma)^2``.

    Degrees are clamped at 1 (isolated training nodes contribute the ``d = 1``
    point mass; the constant ``log d`` term is dropped).

    Args:
        deg_mu: Shape ``(B,)`` location parameters.
        deg_log_sigma: Shape ``(B,)`` log-scale parameters.
        true_degree: Shape ``(B,)`` true simple degrees ``|N(u)|`` on G_struct.

    Returns:
        The scalar mean NLL.
    """
    sigma = torch.clamp(torch.exp(deg_log_sigma), min=_SIGMA_MIN)
    log_d = torch.log(torch.clamp(true_degree, min=1.0))
    return (torch.log(sigma) + 0.5 * ((log_d - deg_mu) / sigma) ** 2).mean()


# --------------------------------------------------------------------------- realism (ED)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch energy distance ``2·E||x-y|| - E||x-x'|| - E||y-y'||``.

    Args:
        x: Shape ``(n, d)`` sample batch.
        y: Shape ``(m, d)`` sample batch.

    Returns:
        The scalar (non-negative up to estimation noise) energy distance.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return x.sum() * 0.0 + y.sum() * 0.0
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return 2.0 * d_xy - d_xx - d_yy


def generated_ego_stats(slots: SlotSet) -> torch.Tensor:
    """Soft ego-stat 4-vectors from the Imagine heads (spec Sec 13.6).

    ``d = sum pi·m``; ``E_nn = sum_{k<k'} adj·pi·pi·m·m``; the vector is
    ``[d, E_nn / C(d, 2), d + E_nn, (d + E_nn) / C(d+1, 2)]``.

    Args:
        slots: The generated slot set.

    Returns:
        Shape ``(B, 4)`` stat vectors.
    """
    weight = slots.pi * slots.mult
    d_soft = weight.sum(dim=-1)
    pair_w = weight[:, :, None] * weight[:, None, :]
    upper = torch.triu(slots.adj * pair_w, diagonal=1)
    e_nn = upper.sum(dim=(1, 2))
    choose2 = torch.clamp(d_soft * (d_soft - 1.0) / 2.0, min=1.0)
    choose2_ego = torch.clamp((d_soft + 1.0) * d_soft / 2.0, min=1.0)
    return torch.stack(
        [d_soft, e_nn / choose2, d_soft + e_nn, (d_soft + e_nn) / choose2_ego], dim=-1
    )


def standardized_energy_distance(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """ED after per-coordinate standardization by the real-side mean/std.

    Args:
        generated: Shape ``(n, d)`` generated stat vectors.
        real: Shape ``(m, d)`` real stat vectors (standardization reference;
            detached — targets never receive gradients).

    Returns:
        The scalar energy distance.
    """
    real = real.detach()
    mean = real.mean(dim=0, keepdim=True)
    std = torch.clamp(real.std(dim=0, keepdim=True), min=1e-6)
    return energy_distance((generated - mean) / std, (real - mean) / std)


class RandomGIN(nn.Module):
    """Frozen randomly-initialized GIN over weighted ego-nets (spec Sec 13.6).

    3 layers, hidden 64, sum-pool; weights drawn once at seed 0 and never
    trained (``requires_grad = False``): it is a fixed random feature map whose
    embedding distributions the ED term compares.
    """

    def __init__(self, config: EgoStitchConfig, *, feature_dim: int = 4, seed: int = 0) -> None:
        """Build and freeze the random GIN.

        Args:
            config: Supplies ``gin_hidden`` / ``gin_layers``.
            feature_dim: Node feature dim (``[is_anchor, pi, m, g]``).
            seed: Weight-init seed (pinned to 0).
        """
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        dims = [feature_dim] + [config.gin_hidden] * config.gin_layers
        weights: list[nn.Linear] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layer = nn.Linear(d_in, d_out)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.2)
                layer.bias.zero_()
            weights.append(layer)
        self.layers = nn.ModuleList(weights)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Embed a batch of weighted graphs.

        Args:
            features: Shape ``(B, N, feature_dim)`` node features.
            adjacency: Shape ``(B, N, N)`` (weighted) adjacency.

        Returns:
            Shape ``(B, gin_hidden)`` sum-pooled embeddings.
        """
        h = features
        for layer in self.layers:
            h = F.gelu(layer(h + torch.bmm(adjacency, h)))
        return h.sum(dim=1)


def generated_ego_graph(slots: SlotSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft ego-net tensors for the GIN term (spec Sec 13.6).

    Node 0 is the anchor ``u``; nodes ``1..K`` are the slots. Star edges are
    weighted ``pi·m``, slot-slot edges ``adj·pi·pi``. Features are
    ``[is_anchor, pi, m, g]``.

    Args:
        slots: The generated slot set.

    Returns:
        ``(features (B, K+1, 4), adjacency (B, K+1, K+1))``.
    """
    batch, k = slots.pi.shape
    features = slots.h.new_zeros((batch, k + 1, 4))
    features[:, 0, 0] = 1.0
    features[:, 1:, 1] = slots.pi
    features[:, 1:, 2] = slots.mult
    features[:, 1:, 3] = slots.gate
    adjacency = slots.h.new_zeros((batch, k + 1, k + 1))
    star = slots.pi * slots.mult
    adjacency[:, 0, 1:] = star
    adjacency[:, 1:, 0] = star
    pair_pi = slots.pi[:, :, None] * slots.pi[:, None, :]
    slot_block = slots.adj * pair_pi
    slot_block = slot_block - torch.diag_embed(torch.diagonal(slot_block, dim1=-2, dim2=-1))
    adjacency[:, 1:, 1:] = slot_block
    return features, adjacency


def real_ego_graph(
    target_mult: torch.Tensor,
    target_adj: torch.Tensor,
    target_mask: torch.Tensor,
    target_in_pool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-side ego-net tensors aligned with `generated_ego_graph`.

    Star edges carry the multiplicity labels; target-target edges are the
    binary ``G_struct`` adjacency; features are ``[is_anchor, 1, mult, in_pool]``.

    Args:
        target_mult: Shape ``(B, T)`` multiplicity labels.
        target_adj: Shape ``(B, T, T)`` adjacency among targets.
        target_mask: Shape ``(B, T)`` validity mask.
        target_in_pool: Shape ``(B, T)`` grounding-pool membership.

    Returns:
        ``(features (B, T+1, 4), adjacency (B, T+1, T+1))``.
    """
    batch, t = target_mult.shape
    valid = target_mask.float()
    features = target_mult.new_zeros((batch, t + 1, 4))
    features[:, 0, 0] = 1.0
    features[:, 1:, 1] = valid
    features[:, 1:, 2] = target_mult * valid
    features[:, 1:, 3] = target_in_pool.float() * valid
    adjacency = target_mult.new_zeros((batch, t + 1, t + 1))
    star = target_mult * valid
    adjacency[:, 0, 1:] = star
    adjacency[:, 1:, 0] = star
    pair_valid = valid[:, :, None] * valid[:, None, :]
    adjacency[:, 1:, 1:] = target_adj * pair_valid
    return features, adjacency


# --------------------------------------------------------------------------- SSL


def ssl_consistency(
    slots_a: SlotSet, slots_b: SlotSet, *, ungrounded: torch.Tensor
) -> torch.Tensor:
    """Slot-embedding consistency between two conditioned passes (spec Sec 7).

    Applied to **ungrounded** slots only (``gate < 0.5`` on the clean pass);
    slot correspondence is by index (same queries, perturbed conditioning).

    Args:
        slots_a: Clean-pass slot set.
        slots_b: Perturbed-pass slot set.
        ungrounded: Shape ``(B, K)`` boolean mask of ungrounded slots.

    Returns:
        The scalar mean squared consistency distance.
    """
    if not bool(ungrounded.any()):
        return slots_a.h.sum() * 0.0
    diff = (slots_a.h - slots_b.h)[ungrounded]
    return (diff**2).mean()


# --------------------------------------------------------------------------- totals


_RECON_COMPONENT_NAMES = (
    "feat",
    "exist",
    "mult",
    "deg",
    "slotadj",
    "gate",
    "ptr",
    "align",
    "div",
    "rel",
)


def _recon_total(
    config: EgoStitchConfig,
    *,
    family: LossFamily,
    recon: Mapping[str, torch.Tensor],
    deg: torch.Tensor,
    zero: torch.Tensor,
    recon_factors: Mapping[str, float] | None,
) -> torch.Tensor:
    """Apply rev-3.1 schedule factors inside the reconstruction decomposition."""
    factors = dict.fromkeys(_RECON_COMPONENT_NAMES, 1.0)
    if family == "egostitch_e2e" and recon_factors is not None:
        if set(recon_factors) != set(_RECON_COMPONENT_NAMES):
            raise ValueError("recon_factors must name exactly the ten L_recon components")
        factors.update({name: float(value) for name, value in recon_factors.items()})
    l_recon = (
        factors["feat"] * config.w_feat * recon["feat"]
        + factors["exist"] * config.w_exist * recon["exist"]
        + factors["mult"] * config.w_mult * recon["mult"]
        + factors["deg"] * config.w_deg * deg
        + factors["slotadj"] * config.w_slotadj * recon["slotadj"]
        + factors["gate"] * config.w_gate * recon["gate"]
    )
    if family == "egostitch_e2e":
        l_recon = (
            l_recon
            + factors["ptr"] * config.w_ptr * recon["ptr"]
            + factors["align"] * config.w_align * recon.get("align", zero)
            + factors["div"] * config.w_div * recon["div"]
            + factors["rel"] * config.w_rel * recon.get("rel", zero)
        )
    return l_recon


def stage1_family_tensors(
    config: EgoStitchConfig,
    *,
    family: LossFamily = "egostitch",
    edge: torch.Tensor,
    recon: dict[str, torch.Tensor],
    deg: torch.Tensor,
    real_egostat: torch.Tensor,
    real_gin: torch.Tensor,
    ssl_noise: torch.Tensor,
    ssl_pool: torch.Tensor,
    recon_factors: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return the four *weighted* loss-family tensors used by the optimizer."""
    zero = edge * 0.0
    l_recon = _recon_total(
        config,
        family=family,
        recon=recon,
        deg=deg,
        zero=zero,
        recon_factors=recon_factors,
    )
    l_real = config.w_egostat * real_egostat + config.w_gin * real_gin
    l_ssl = 0.5 * ssl_noise + 0.5 * ssl_pool
    return {
        "edge": edge,
        "recon": config.lambda_recon * l_recon,
        "real": config.lambda_real * l_real,
        "ssl": config.lambda_ssl * l_ssl,
    }


def stage1_total(
    config: EgoStitchConfig,
    *,
    family: LossFamily = "egostitch",
    edge: torch.Tensor,
    recon: dict[str, torch.Tensor],
    deg: torch.Tensor,
    real_egostat: torch.Tensor,
    real_gin: torch.Tensor,
    ssl_noise: torch.Tensor,
    ssl_pool: torch.Tensor,
    recon_factors: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine loss families with the pinned weights (spec Sec 13.5 / Sec 14.4.1).

    Args:
        config: Supplies every master and interior weight.
        family: Selects the frozen Stage-1 or rev-3.1 E2E decomposition.
        edge: ``L_edge`` (BCE over the edge stream).
        recon: The `recon_losses` dict (denoise terms folded in by the caller).
        deg: The lognormal degree NLL.
        real_egostat: Ego-stat energy distance.
        real_gin: Random-GIN energy distance.
        ssl_noise: Feature-noise consistency.
        ssl_pool: Pool-resample consistency.
        recon_factors: Optional named rev-3.1 schedule factors applied inside
            ``L_recon``; ignored for the standalone ``egostitch`` family.

    Returns:
        ``(total, per-family floats)`` — the float dict is for logging
        (gradient-norm-per-family monitoring is the trainer's job).
    """
    zero = edge * 0.0
    align = recon.get("align", zero)
    rel = recon.get("rel", zero)
    l_recon = _recon_total(
        config,
        family=family,
        recon=recon,
        deg=deg,
        zero=zero,
        recon_factors=recon_factors,
    )
    l_real = config.w_egostat * real_egostat + config.w_gin * real_gin
    l_ssl = 0.5 * ssl_noise + 0.5 * ssl_pool
    families = stage1_family_tensors(
        config,
        family=family,
        edge=edge,
        recon=recon,
        deg=deg,
        real_egostat=real_egostat,
        real_gin=real_gin,
        ssl_noise=ssl_noise,
        ssl_pool=ssl_pool,
        recon_factors=recon_factors,
    )
    total = torch.stack(tuple(families.values())).sum()
    parts = {
        "edge": float(edge.detach()),
        "recon": float(l_recon.detach()),
        "recon_feat": float(recon["feat"].detach()),
        "recon_exist": float(recon["exist"].detach()),
        "recon_mult": float(recon["mult"].detach()),
        "recon_deg": float(deg.detach()),
        "recon_slotadj": float(recon["slotadj"].detach()),
        "recon_gate": float(recon["gate"].detach()),
        "recon_ptr": float(recon["ptr"].detach()),
        "recon_align": float(align.detach()),
        "recon_div": float(recon["div"].detach()),
        "recon_rel": float(rel.detach()),
        "real": float(l_real.detach()),
        "real_egostat": float(real_egostat.detach()),
        "real_gin": float(real_gin.detach()),
        "ssl": float(l_ssl.detach()),
        "total": float(total.detach()),
    }
    return total, parts
