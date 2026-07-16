"""Stage-1 loss tree (spec Sec 7 subset, pinned in Sec 13.5-13.6).

``L = L_edge + 0.5·L_real + 0.1·L_ssl + 1.0·L_recon`` with

``L_recon = 1.0·L_feat + 0.5·L_exist + 0.25·L_mult + 0.5·L_deg
+ 0.5·L_slotadj + 0.25·L_gate``
``L_real  = (2/3)·ED(ego stats) + (1/3)·ED(random-GIN embeddings)``
``L_ssl   = 0.5·feature-noise consistency + 0.5·pool-resample consistency``

Dropped with their mechanisms: KL, L_VQ, L_codestats, L_entropy, L_BP, L_joint.
Pinned instantiation notes: the multiplicity NLL is the unit-variance lognormal
NLL (squared log error); the target sides of ``L_feat`` and the matching cost
are stop-gradient (spec Sec 13.7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.imagine import DenoiseSlots, SlotSet
from src.model.egostitch.layers import stable_log
from src.model.egostitch.matching import Assignment

_SIGMA_MIN = 1e-3


def _probability_bce(
    input: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    """Evaluate probability-space BCE in fp32 outside mixed-precision autocast."""
    with torch.autocast(device_type=input.device.type, enabled=False):
        return F.binary_cross_entropy(input.float(), target.float(), reduction=reduction)


# --------------------------------------------------------------------------- reconstruction


def recon_losses(
    slots: SlotSet,
    assignment: Assignment,
    *,
    target_proj: torch.Tensor,
    target_mult: torch.Tensor,
    target_adj: torch.Tensor,
    target_in_pool: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Matched-slot reconstruction losses (spec Sec 13.5).

    Args:
        slots: The generated slot set.
        assignment: The constant Hungarian assignment.
        target_proj: Shape ``(B, T, d_p)`` **stop-gradient** projected targets.
        target_mult: Shape ``(B, T)`` multiplicity labels.
        target_adj: Shape ``(B, T, T)`` adjacency among targets.
        target_in_pool: Shape ``(B, T)`` booleans — target is in the node's
            grounding pool ``G(u)`` (Stage-1 ``L_gate`` supervision).

    Returns:
        ``{"feat", "exist", "mult", "slotadj", "gate"}`` scalar loss tensors.
    """
    device = slots.h.device
    feat_terms: list[torch.Tensor] = []
    mult_terms: list[torch.Tensor] = []
    slotadj_terms: list[torch.Tensor] = []
    gate_logit_terms: list[torch.Tensor] = []
    gate_label_terms: list[torch.Tensor] = []
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
            adj_pred = slots.adj[b][s_idx][:, s_idx]
            adj_true = target_adj[b][t_idx][:, t_idx]
            off_diag = ~torch.eye(s_idx.numel(), dtype=torch.bool, device=device)
            slotadj_terms.append(
                _probability_bce(
                    torch.clamp(adj_pred[off_diag], 1e-6, 1.0 - 1e-6), adj_true[off_diag]
                )
            )
        gate_logit_terms.append(slots.gate[b, s_idx])
        gate_label_terms.append(target_in_pool[b, t_idx].float())

    zero = slots.h.sum() * 0.0
    feat = torch.stack(feat_terms).mean() if feat_terms else zero
    mult = torch.stack(mult_terms).mean() if mult_terms else zero
    slotadj = torch.stack(slotadj_terms).mean() if slotadj_terms else zero
    if gate_logit_terms:
        gate = _probability_bce(
            torch.clamp(torch.cat(gate_logit_terms), 1e-6, 1.0 - 1e-6),
            torch.cat(gate_label_terms),
        )
    else:
        gate = zero

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

    return {"feat": feat, "exist": exist, "mult": mult, "slotadj": slotadj, "gate": gate}


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


def stage1_family_tensors(
    config: EgoStitchConfig,
    *,
    edge: torch.Tensor,
    recon: dict[str, torch.Tensor],
    deg: torch.Tensor,
    real_egostat: torch.Tensor,
    real_gin: torch.Tensor,
    ssl_noise: torch.Tensor,
    ssl_pool: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the four *weighted* loss-family tensors used by the optimizer."""
    l_recon = (
        config.w_feat * recon["feat"]
        + config.w_exist * recon["exist"]
        + config.w_mult * recon["mult"]
        + config.w_deg * deg
        + config.w_slotadj * recon["slotadj"]
        + config.w_gate * recon["gate"]
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
    edge: torch.Tensor,
    recon: dict[str, torch.Tensor],
    deg: torch.Tensor,
    real_egostat: torch.Tensor,
    real_gin: torch.Tensor,
    ssl_noise: torch.Tensor,
    ssl_pool: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the Stage-1 loss families with the pinned weights (spec Sec 13.5).

    Args:
        config: Supplies every master and interior weight.
        edge: ``L_edge`` (BCE over the edge stream).
        recon: The `recon_losses` dict (denoise terms folded in by the caller).
        deg: The lognormal degree NLL.
        real_egostat: Ego-stat energy distance.
        real_gin: Random-GIN energy distance.
        ssl_noise: Feature-noise consistency.
        ssl_pool: Pool-resample consistency.

    Returns:
        ``(total, per-family floats)`` — the float dict is for logging
        (gradient-norm-per-family monitoring is the trainer's job).
    """
    l_recon = (
        config.w_feat * recon["feat"]
        + config.w_exist * recon["exist"]
        + config.w_mult * recon["mult"]
        + config.w_deg * deg
        + config.w_slotadj * recon["slotadj"]
        + config.w_gate * recon["gate"]
    )
    l_real = config.w_egostat * real_egostat + config.w_gin * real_gin
    l_ssl = 0.5 * ssl_noise + 0.5 * ssl_pool
    families = stage1_family_tensors(
        config,
        edge=edge,
        recon=recon,
        deg=deg,
        real_egostat=real_egostat,
        real_gin=real_gin,
        ssl_noise=ssl_noise,
        ssl_pool=ssl_pool,
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
        "real": float(l_real.detach()),
        "real_egostat": float(real_egostat.detach()),
        "real_gin": float(real_gin.detach()),
        "ssl": float(l_ssl.detach()),
        "total": float(total.detach()),
    }
    return total, parts
