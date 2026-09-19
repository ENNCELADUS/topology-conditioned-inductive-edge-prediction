"""Where the Stage II motif generator loses its slot distinction.

Pilot B reads the generator's *output*: wave 1 and wave 2 both found a
near-constant, slot-symmetric 96-edge graph whose ``L_G`` does not beat a fitted
constant. This probe reads the *inside* of the same generator, layer by layer, so
the collapse can be attributed to a stage rather than to the arm.

The measurement is a replay, not a second model. One `src.score_universe` packed
pass caches the generator's own inputs -- the frozen trunk's residue states of
both endpoints and their true lengths -- through a forward pre-hook on
``model.generator`` while the scorer builds its own predicted-graph bank. Every
number below is then produced by calling that same generator's own submodules
step by step (`decompose`) on the cached inputs, twice: once under bf16 autocast,
which is the regime the *trainer* runs the pair pass in, and once in full fp32.
`decompose` is checked against ``MotifGenerator.forward`` on every universe and
the max absolute disagreement is reported, so the decomposition is known to be
the model rather than a paraphrase of it.

Note that the formal scoring path runs the generator with autocast *off* over an
fp32 encoding cache (`src.score_universe._score_v3_1_packed`, the motif branch),
so the cached inputs arrive in fp32 and the bf16 replay is the training regime,
not the scoring one. The header of ``probe.md`` records the dtype as delivered.

What is measured, per checkpoint:

1. The per-role slot spread ``D(S) = sqrt(mean_k ||s_k - s_bar||^2 / d)`` at
   every stage from the raw attention reads to the emitted weights, in absolute
   and in units of the stage's own scale, with the fraction of rows whose
   within-role states are identical to 1e-6, 1e-4 and 1e-2.
2. The input side: the centred variance of the residue states, of their 96-d
   projection and of the attention's own K and V, the projected query norms, the
   pre-softmax attention logit spread and the attention entropy over
   ``log(valid length)``.
3. The heads: the last-layer gain against its ``1e-3`` initialisation and its
   bias, and the gradient of ``L_G`` into every generator group, as a ratio to
   the gradient into the bias the wave-1 verdict found the head never left.
4. The two loss families on the shared parameters: the closure gradient
   (``beta_c L_c + beta_p L_p``) against the bridge gradient
   (``beta_q L_q + beta_a L_a + beta_i L_i``), their norms and their cosine,
   under the checkpoint's own betas and under ``beta_c = 1``.
5. Fit per true-closure-count stratum against the re-fitted asymmetric constant
   of pilot B, with the predicted and true wedge and bridge masses.

Rows are drawn stratified by the true closure count from two universes: V_val's
``val_cls`` list (a labelled diagnostic, as in pilot B) and node-disjoint
held-out training rows.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

import src.score_universe as su
from src.data.motif_template import (
    C_SLOTS,
    CLOSURE_U,
    CLOSURE_V,
    EDGE_ENDPOINTS,
    EDGE_TYPES,
    L_SLOTS,
    N_EDGE_TYPES,
    N_EDGES,
    R_SLOTS,
    MotifTemplateTable,
    count_statistics,
)
from src.data.packed_features import load_packed_manifest
from src.experiments.motif_pilot_b import (
    GraphLossWeights,
    fit_constant_template,
    predicted_bank,
    rows_inside,
    split_training_nodes,
)
from src.model.egostitch.classifier.layers import (
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)
from src.model.egostitch.classifier.motif_prompt import (
    EDGE_TYPE_NAMES,
    MotifGenerator,
    V3_1MotifPrompt,
    _ResidualReadBlock,
)

logger = logging.getLogger("motif_generator_probe")

Pair = tuple[str, str]

#: The closure-count strata, as ``(name, low, high)`` inclusive bounds.
STRATA: tuple[tuple[str, int, int], ...] = (("0", 0, 0), ("1-2", 1, 2), ("3-7", 3, 7), ("8", 8, 8))
#: Identity thresholds the within-role max pairwise distance is read against.
IDENTITY_THRESHOLDS: tuple[float, ...] = (1e-6, 1e-4, 1e-2)
#: The gate width every slot state lives in (``MotifGenerator``'s ``_GATE_DIM``).
GATE_DIM = 96
#: Generator parameter groups the gradient of ``L_G`` is attributed to.
UPSTREAM_GROUPS: dict[str, tuple[str, ...]] = {
    "residue_proj": ("residue_proj.",),
    "attention_in_proj": ("attention.in_proj_weight",),
    "attention_all": ("attention.",),
    "bridge_queries": ("bridge_queries",),
    "witness_queries": ("witness_queries",),
    "endpoint_proj": ("endpoint_proj.",),
    "witness_mix": ("witness_mix.",),
    "message_layers": ("message_layers.",),
}
#: Parameters both loss families share: everything that is not a gate head.
SHARED_PREFIXES: tuple[str, ...] = (
    "residue_proj.",
    "attention.",
    "bridge_queries",
    "witness_queries",
    "endpoint_proj.",
    "witness_mix.",
    "message_layers.",
)

_ROLE_SLOTS: dict[str, tuple[int, ...]] = {"C": C_SLOTS, "L": L_SLOTS, "R": R_SLOTS}
_EDGE_ROW_INDEX = torch.as_tensor([i for i, _ in EDGE_ENDPOINTS], dtype=torch.long)
_EDGE_COL_INDEX = torch.as_tensor([j for _, j in EDGE_ENDPOINTS], dtype=torch.long)
_TYPE_MASK = tuple(
    torch.as_tensor([kind == edge_type for kind in EDGE_TYPES]) for edge_type in range(N_EDGE_TYPES)
)


# ---------------------------------------------------------------------------
# Spread measurements
# ---------------------------------------------------------------------------


def slot_spread(states: torch.Tensor) -> torch.Tensor:
    """Return the per-row slot spread ``D(S)`` of one role's states.

    ``D(S) = sqrt(mean_k ||s_k - s_bar||^2 / d)``, the root-mean-square entry of
    the mean-centred state block: zero exactly when every slot of the role
    carries the same vector, and in the same units as the states themselves.

    Args:
        states: ``(B, K, d)`` states of one role, one stage.

    Returns:
        ``(B,)`` spreads in float32.

    Raises:
        ValueError: If ``states`` is not a 3-D block with at least one slot.
    """
    if states.dim() != 3 or states.size(1) < 1:
        raise ValueError(f"expected a (B, K, d) state block, got {tuple(states.shape)}")
    value = states.detach().float()
    centred = value - value.mean(dim=1, keepdim=True)
    return centred.pow(2).mean(dim=(1, 2)).sqrt()


def slot_scale(states: torch.Tensor) -> torch.Tensor:
    """Return the per-row RMS entry of one role's states, the spread's own unit.

    Args:
        states: ``(B, K, d)`` states of one role, one stage.

    Returns:
        ``(B,)`` scales in float32.
    """
    return states.detach().float().pow(2).mean(dim=(1, 2)).sqrt()


def max_pairwise_distance(states: torch.Tensor) -> torch.Tensor:
    """Return the per-row largest distance between two slots of one role.

    Args:
        states: ``(B, K, d)`` states of one role, one stage.

    Returns:
        ``(B,)`` distances in float32; zero for a single-slot role.
    """
    value = states.detach().float()
    if value.size(1) < 2:
        return value.new_zeros(value.size(0))
    return torch.cdist(value, value).amax(dim=(1, 2))


def spread_summary(
    spread: NDArray[np.float64], scale: NDArray[np.float64], distance: NDArray[np.float64]
) -> dict[str, float]:
    """Summarise one ``(stage, role)`` cell over a set of rows.

    Args:
        spread: Per-row ``D(S)``.
        scale: Per-row RMS entry of the same states.
        distance: Per-row within-role max pairwise distance.

    Returns:
        The absolute and relative spread and the identity fractions.
    """
    scale_mean = float(np.mean(scale)) if scale.size else 0.0
    out: dict[str, float] = {
        "rows": float(spread.size),
        "D": float(np.mean(spread)) if spread.size else float("nan"),
        "D_median": float(np.median(spread)) if spread.size else float("nan"),
        "scale": scale_mean,
        "D_relative": (
            float(np.mean(spread / np.maximum(scale, 1e-30))) if spread.size else float("nan")
        ),
    }
    for threshold in IDENTITY_THRESHOLDS:
        key = f"frac_identical_below_{threshold:.0e}"
        out[key] = float(np.mean(distance < threshold)) if distance.size else float("nan")
    return out


# ---------------------------------------------------------------------------
# Step-by-step replay of MotifGenerator.forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorStages:
    """Every intermediate of one batch's pass through `MotifGenerator`."""

    bridge_u: torch.Tensor
    bridge_v: torch.Tensor
    witness_u: torch.Tensor
    witness_v: torch.Tensor
    closure: torch.Tensor
    slots: tuple[torch.Tensor, ...]
    features: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    state_u: torch.Tensor
    state_v: torch.Tensor
    pad_u: torch.Tensor | None
    pad_v: torch.Tensor | None


def read_slots(
    generator: MotifGenerator,
    queries: torch.Tensor,
    states: torch.Tensor,
    pad: torch.Tensor | None,
    block_name: str,
) -> torch.Tensor:
    """Call the generator's own slot read, whatever read variant it carries.

    ``MotifGenerator._read`` gained a read-block argument when ``slot_read``
    arrived, and a checkpoint trained under either variant must decompose
    faithfully, so the call is built from the method's own signature rather than
    pinned to one of them.

    Args:
        generator: The Stage II generator.
        queries: ``(K, 96)`` shared slot queries.
        states: ``(B, L, 96)`` projected residue states of one endpoint.
        pad: ``(B, L)`` key padding mask, or ``None``.
        block_name: The attribute holding this family's read block, when the
            signature takes one.

    Returns:
        ``(B, K, 96)`` slot states.
    """
    read = cast(Callable[..., torch.Tensor], generator._read)
    if "block" in inspect.signature(generator._read).parameters:
        return read(queries, states, pad, getattr(generator, block_name, None))
    return read(queries, states, pad)


def decompose(
    generator: MotifGenerator,
    encoded_u: torch.Tensor,
    encoded_v: torch.Tensor,
    lengths_u: torch.Tensor,
    lengths_v: torch.Tensor,
) -> GeneratorStages:
    """Run one batch through the generator's own submodules, keeping every stage.

    Every learned step is the module's own -- `MotifGenerator._read`, its
    ``witness_mix``, ``endpoint_proj``, ``message_layers`` and gate heads are
    called here, not reimplemented -- so the decomposition follows the
    checkpoint's architecture. Only the slot assembly and the typed head loop are
    written out, and `check_decomposition` verifies the result against
    ``MotifGenerator.forward`` itself.

    Args:
        generator: The Stage II generator, in ``eval()`` mode.
        encoded_u: ``(B, L_u, d_model)`` frozen residue states of ``u``.
        encoded_v: ``(B, L_v, d_model)`` frozen residue states of ``v``.
        lengths_u: ``(B,)`` true residue lengths of ``u``.
        lengths_v: ``(B,)`` true residue lengths of ``v``.

    Returns:
        The stage-by-stage intermediates, ending in the emitted weights.
    """
    pad_u = _build_padding_mask(lengths_u, encoded_u.size(1))
    pad_v = _build_padding_mask(lengths_v, encoded_v.size(1))
    state_u = generator.residue_proj(encoded_u)
    state_v = generator.residue_proj(encoded_v)
    left = read_slots(generator, generator.bridge_queries, state_u, pad_u, "bridge_read")
    right = read_slots(generator, generator.bridge_queries, state_v, pad_v, "bridge_read")
    witness_u = read_slots(generator, generator.witness_queries, state_u, pad_u, "witness_read")
    witness_v = read_slots(generator, generator.witness_queries, state_v, pad_v, "witness_read")
    closure = generator.witness_mix(
        torch.cat([witness_u + witness_v, (witness_u - witness_v).abs()], dim=-1)
    )
    pooled_u = generator.endpoint_proj(
        masked_mean(state_u, inner_token_mask(x=state_u, padding_mask=pad_u))
    )
    pooled_v = generator.endpoint_proj(
        masked_mean(state_v, inner_token_mask(x=state_v, padding_mask=pad_v))
    )
    head = torch.cat([pooled_u.unsqueeze(1), pooled_v.unsqueeze(1), closure, left, right], dim=1)
    slots = [head]
    for layer in generator.message_layers:
        slots.append(layer(slots[-1], generator.incidence))

    final = slots[-1]
    rows = _EDGE_ROW_INDEX.to(final.device)
    cols = _EDGE_COL_INDEX.to(final.device)
    with torch.autocast(device_type=final.device.type, enabled=False):
        # `MotifGenerator.forward` promotes both operands before forming the sum
        # and the difference, so a replay that adds in bf16 and promotes after
        # measures a rounding grid the generator no longer uses.
        h_i, h_j = final[:, rows].float(), final[:, cols].float()
        features = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
        logits = features.new_zeros(final.size(0), N_EDGES)
        for edge_type, gate in enumerate(generator.heads):
            mask = _TYPE_MASK[edge_type].to(final.device)
            block = features[:, mask]
            if generator.cfg.gate_mode == "per_type":
                logits[:, mask] = gate(block.mean(dim=1)).expand(-1, int(block.size(1)))
            else:
                logits[:, mask] = gate(block).squeeze(-1)
        weights = torch.sigmoid(logits)
    return GeneratorStages(
        bridge_u=left,
        bridge_v=right,
        witness_u=witness_u,
        witness_v=witness_v,
        closure=closure,
        slots=tuple(slots),
        features=features,
        logits=logits,
        weights=weights,
        state_u=state_u,
        state_v=state_v,
        pad_u=pad_u,
        pad_v=pad_v,
    )


def check_decomposition(
    generator: MotifGenerator,
    encoded_u: torch.Tensor,
    encoded_v: torch.Tensor,
    lengths_u: torch.Tensor,
    lengths_v: torch.Tensor,
) -> float:
    """Return the max absolute disagreement of `decompose` with the module itself.

    Args:
        generator: The Stage II generator, in ``eval()`` mode.
        encoded_u: ``(B, L_u, d_model)`` residue states of ``u``.
        encoded_v: ``(B, L_v, d_model)`` residue states of ``v``.
        lengths_u: ``(B,)`` true lengths of ``u``.
        lengths_v: ``(B,)`` true lengths of ``v``.

    Returns:
        The largest absolute difference of the two emitted weight tables.
    """
    with torch.no_grad(), torch.autocast(device_type=encoded_u.device.type, enabled=False):
        stages = decompose(generator, encoded_u, encoded_v, lengths_u, lengths_v)
        reference = generator(encoded_u, encoded_v, lengths_u, lengths_v)
    return float((stages.weights - reference).abs().max())


def stage_groups(stages: GeneratorStages) -> dict[str, dict[str, torch.Tensor]]:
    """Return the ``(stage, role)`` state blocks whose spread is measured.

    Args:
        stages: One batch's intermediates.

    Returns:
        ``stage -> role -> (B, K, d)`` blocks, from the raw reads to the weights.
    """
    groups: dict[str, dict[str, torch.Tensor]] = {
        "reads": {
            "bridge_u": stages.bridge_u,
            "bridge_v": stages.bridge_v,
            "witness_u": stages.witness_u,
            "witness_v": stages.witness_v,
        },
        "mix": {"C": stages.closure},
    }
    for index, slots in enumerate(stages.slots[1:], start=1):
        groups[f"mp{index}"] = {
            role: slots[:, list(indices)] for role, indices in _ROLE_SLOTS.items()
        }
    groups["head_features"] = {
        name: stages.features[:, _TYPE_MASK[edge_type].to(stages.features.device)]
        for edge_type, name in enumerate(EDGE_TYPE_NAMES)
    }
    groups["logits"] = {
        name: stages.logits[:, _TYPE_MASK[edge_type].to(stages.logits.device)].unsqueeze(-1)
        for edge_type, name in enumerate(EDGE_TYPE_NAMES)
    }
    groups["weights"] = {
        name: stages.weights[:, _TYPE_MASK[edge_type].to(stages.weights.device)].unsqueeze(-1)
        for edge_type, name in enumerate(EDGE_TYPE_NAMES)
    }
    return groups


# ---------------------------------------------------------------------------
# Input-side and attention measurements
# ---------------------------------------------------------------------------


def _centred_variance(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Per-row mean squared deviation per dimension over the kept tokens.

    Args:
        values: ``(B, L, d)`` token states.
        keep: ``(B, L)`` boolean keep mask.

    Returns:
        ``(B,)`` variances in float32.
    """
    value = values.detach().float()
    mask = keep.float().unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (value * mask).sum(dim=1) / count
    centred = (value - mean.unsqueeze(1)) * mask
    return centred.pow(2).sum(dim=(1, 2)) / (count.squeeze(-1) * value.size(-1))


def _attention_side(
    generator: MotifGenerator,
    raw: torch.Tensor,
    state: torch.Tensor,
    pad: torch.Tensor | None,
    queries: torch.Tensor,
    block: _ResidualReadBlock | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Measure one query family reading one endpoint.

    Args:
        generator: The Stage II generator.
        raw: ``(B, L, d_model)`` trunk residue states.
        state: ``(B, L, 96)`` their projection, the attention's own input.
        pad: ``(B, L)`` padding mask, or ``None``.
        queries: ``(8, 96)`` the query family.
        block: The family's residual read block, or ``None`` for the bare read.

    Returns:
        Per-row arrays: the four centred variances, the pre-softmax logit spread
        and the attention entropy over ``log(valid length)``.
    """
    keep = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
    if pad is not None:
        keep = ~pad
    weight = generator.attention.in_proj_weight.detach().float()
    bias = generator.attention.in_proj_bias.detach().float()
    query_weight, query_bias = weight[:GATE_DIM], bias[:GATE_DIM]
    key_weight, key_bias = weight[GATE_DIM : 2 * GATE_DIM], bias[GATE_DIM : 2 * GATE_DIM]
    value_weight, value_bias = weight[2 * GATE_DIM :], bias[2 * GATE_DIM :]
    projected = state.detach().float()
    # Under ``slot_read='residual_block'`` the attention is handed ``LN_q(Q)`` and
    # ``LN_h(S)``, and its values are ``LN_h(S)`` or ``S`` by ``value_norm``.
    # Measuring the bare projection instead would report the entropy and the K/V
    # variance of a computation the generator does not run -- and ``V/proj`` is
    # exactly the number the value-norm comparison turns on.
    key_source = projected if block is None else block.norm_h(projected)
    value_source = key_source if block is None or block.value_norm else projected
    # `nn.LayerNorm` is per position over the last dimension, so normalising the
    # ``(K, 96)`` queries is the same as normalising their batch expansion.
    bare_queries = queries.detach().float()
    query_source = bare_queries if block is None else block.norm_q(bare_queries)
    keys = key_source @ key_weight.T + key_bias
    values = value_source @ value_weight.T + value_bias
    projected_queries = query_source @ query_weight.T + query_bias

    heads = generator.attention.num_heads
    head_dim = GATE_DIM // heads
    batch, length, _ = projected.shape
    key_heads = keys.view(batch, length, heads, head_dim).permute(0, 2, 1, 3)
    query_heads = projected_queries.view(-1, heads, head_dim).permute(1, 0, 2)
    scores = torch.einsum("bhld,hqd->bhql", key_heads, query_heads) / math.sqrt(head_dim)
    valid = keep.view(batch, 1, 1, length)
    counts = keep.sum(dim=1).clamp_min(1).float()
    mean = (scores * valid).sum(dim=-1) / counts.view(batch, 1, 1)
    spread = (((scores - mean.unsqueeze(-1)) * valid) ** 2).sum(dim=-1) / counts.view(batch, 1, 1)
    logit_std = spread.sqrt().mean(dim=(1, 2))

    expanded = query_source.unsqueeze(0).expand(state.size(0), -1, -1)
    attention, weights = generator.attention(
        expanded,
        key_source,
        value_source,
        key_padding_mask=pad,
        need_weights=True,
        average_attn_weights=True,
    )
    probability = weights.detach().float().clamp_min(0.0)
    entropy = -(probability * torch.log(probability.clamp_min(1e-30))).sum(dim=-1)
    normaliser = torch.log(counts.clamp_min(2.0)).view(batch, 1)
    return {
        "H": _centred_variance(raw, keep).cpu().numpy().astype(np.float64),
        "proj": _centred_variance(projected, keep).cpu().numpy().astype(np.float64),
        "K": _centred_variance(keys, keep).cpu().numpy().astype(np.float64),
        "V": _centred_variance(values, keep).cpu().numpy().astype(np.float64),
        "logit_std": logit_std.detach().cpu().numpy().astype(np.float64),
        "entropy_ratio": (entropy / normaliser)
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "read_cosine": _read_cosine(attention).cpu().numpy().astype(np.float64),
    }


def attention_side(
    generator: MotifGenerator,
    raw: torch.Tensor,
    state: torch.Tensor,
    pad: torch.Tensor | None,
    queries: torch.Tensor,
    block: _ResidualReadBlock | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Measure one query family reading one endpoint (public wrapper).

    `src.experiments.motif_generator_fit` reports the same attention entropy and
    K/V variance ratios as this probe, and must measure them with this module's
    code rather than a second copy of it.

    Args:
        generator: The Stage II generator.
        raw: ``(B, L, d_model)`` trunk residue states.
        state: ``(B, L, 96)`` their projection, the attention's own input.
        pad: ``(B, L)`` padding mask, or ``None``.
        queries: ``(8, 96)`` the query family.
        block: The family's residual read block, or ``None`` for the bare read.

    Returns:
        The per-row arrays of `_attention_side`.
    """
    return _attention_side(generator, raw, state, pad, queries, block)


def _read_cosine(reads: torch.Tensor) -> torch.Tensor:
    """Mean cosine between two different slots' reads, per row.

    Args:
        reads: ``(B, K, d)`` one query family's reads.

    Returns:
        ``(B,)`` mean off-diagonal cosine similarity in float32.
    """
    value = reads.detach().float()
    normalised = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    gram = normalised @ normalised.transpose(1, 2)
    slots = gram.size(1)
    if slots < 2:
        return cast(torch.Tensor, gram.new_zeros(gram.size(0)))
    off = gram.sum(dim=(1, 2)) - torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    return cast(torch.Tensor, off / (slots * (slots - 1)))


def query_norms(generator: MotifGenerator) -> dict[str, float]:
    """Return the mean norm of each query family before and after the Q projection.

    Args:
        generator: The Stage II generator.

    Returns:
        Raw and projected mean query norms per family.
    """
    weight = generator.attention.in_proj_weight.detach().float()[:GATE_DIM]
    bias = generator.attention.in_proj_bias.detach().float()[:GATE_DIM]
    out: dict[str, float] = {}
    for name, queries in (
        ("bridge", generator.bridge_queries),
        ("witness", generator.witness_queries),
    ):
        raw = queries.detach().float()
        out[f"{name}_raw_norm"] = float(raw.norm(dim=-1).mean())
        out[f"{name}_projected_norm"] = float((raw @ weight.T + bias).norm(dim=-1).mean())
    return out


# ---------------------------------------------------------------------------
# Cached generator inputs
# ---------------------------------------------------------------------------


@dataclass
class CachedInputs:
    """One universe's generator inputs, as the scoring pass delivered them."""

    states_u: list[torch.Tensor]
    states_v: list[torch.Tensor]
    lengths_u: torch.Tensor
    lengths_v: torch.Tensor
    dtype: str

    def rows(self) -> int:
        """Return the number of cached rows."""
        return len(self.states_u)

    def megabytes(self) -> float:
        """Return the host memory the cache occupies, in MiB."""
        total = sum(t.numel() * t.element_size() for t in (*self.states_u, *self.states_v))
        return total / 1024.0**2

    def batch(
        self, indices: Sequence[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-pad one chunk of rows onto ``device``.

        Padded positions are masked out of both the attention and the pooling, so
        the chunk's own padding width does not change any measured quantity.

        Args:
            indices: Row positions to gather.
            device: Destination device.

        Returns:
            ``(encoded_u, encoded_v, lengths_u, lengths_v)``.
        """
        rows = list(indices)
        width_u = max(int(self.lengths_u[row]) for row in rows)
        width_v = max(int(self.lengths_v[row]) for row in rows)
        dim = self.states_u[rows[0]].size(-1)
        encoded_u = torch.zeros((len(rows), width_u, dim), dtype=torch.float32)
        encoded_v = torch.zeros((len(rows), width_v, dim), dtype=torch.float32)
        for position, row in enumerate(rows):
            state_u, state_v = self.states_u[row], self.states_v[row]
            encoded_u[position, : state_u.size(0)] = state_u
            encoded_v[position, : state_v.size(0)] = state_v
        index = torch.as_tensor(rows, dtype=torch.int64)
        return (
            encoded_u.to(device),
            encoded_v.to(device),
            self.lengths_u.index_select(0, index).to(device),
            self.lengths_v.index_select(0, index).to(device),
        )


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Copy one tensor out of inference mode onto the host as a normal tensor.

    The scoring pass runs under ``torch.inference_mode``; a tensor born there
    stays an inference tensor through ``clone`` and can never be used in an
    autograd graph, which the gradient measurements need. A numpy round trip is
    the one copy that produces an ordinary tensor.

    Args:
        tensor: Any tensor captured inside the scoring pass.

    Returns:
        An ordinary CPU tensor, float32 for floating inputs.
    """
    source = tensor.detach()
    if source.is_floating_point():
        source = source.float()
    return torch.from_numpy(source.cpu().numpy().copy())


@contextmanager
def capture_generator_inputs(model: V3_1MotifPrompt) -> Iterator[dict[str, object]]:
    """Capture the generator's own inputs during one predicted-graph scoring pass.

    `src.score_universe` hands the generator its batches inside a private
    two-pass routine, so the capture wraps the one builder that drives pass 1 --
    which also tells the hook which row each batch position belongs to -- and
    reads the inputs off a forward pre-hook rather than reimplementing the packed
    encode path.

    Args:
        model: The Stage II checkpoint whose generator is read.

    Yields:
        A mapping filled on exit with ``batches`` (row index lists, in call
        order) and ``calls`` (one host-side per-row input list per call). Each
        batch is trimmed to its true residue lengths inside the hook, so the
        capture never holds a padded batch of the scorer's token budget.

    Raises:
        RuntimeError: If the model carries no generator.
    """
    if model.generator is None:
        raise RuntimeError("a Stage II checkpoint is required; this model has no generator")
    captured: dict[str, object] = {}
    batches: list[list[int]] = []
    calls: list[list[tuple[torch.Tensor, torch.Tensor, int, int]]] = []
    active = {"pass_one": False}
    dtypes: set[str] = set()
    original = su._motif_source_weights

    def spy(
        source_batches: Sequence[Sequence[int]],
        predict_batch: Callable[[Sequence[int]], torch.Tensor],
        *,
        num_rows: int,
    ) -> torch.Tensor:
        """Record the pass-1 batch order and run the real builder.

        Args:
            source_batches: The scorer's batch index lists over the source pairs.
            predict_batch: The scorer's own per-batch generator call.
            num_rows: Rows the scoring process covers.

        Returns:
            The real builder's row-aligned weight bank.
        """
        batches.extend([list(batch) for batch in source_batches])
        active["pass_one"] = True
        try:
            return original(source_batches, predict_batch, num_rows=num_rows)
        finally:
            active["pass_one"] = False

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Copy one pass-1 batch of generator inputs onto the host.

        Args:
            module: The generator the hook is attached to.
            args: Its positional inputs.
        """
        if not active["pass_one"]:
            return
        encoded_u, encoded_v, lengths_u, lengths_v = args
        rows: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
        for position in range(encoded_u.size(0)):
            length_u, length_v = int(lengths_u[position]), int(lengths_v[position])
            rows.append(
                (
                    _to_host(encoded_u[position, :length_u]),
                    _to_host(encoded_v[position, :length_v]),
                    length_u,
                    length_v,
                )
            )
        calls.append(rows)
        dtypes.add(str(encoded_u.dtype))

    handle = model.generator.register_forward_pre_hook(hook)
    su._motif_source_weights = spy
    try:
        yield captured
    finally:
        su._motif_source_weights = original
        handle.remove()
        captured["batches"] = batches
        captured["calls"] = calls
        captured["dtypes"] = sorted(dtypes)


def cache_inputs(
    model: V3_1MotifPrompt,
    pairs: Sequence[Pair],
    pack_dir: Path,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
) -> tuple[CachedInputs, torch.Tensor]:
    """Run one formal scoring pass and keep the generator's inputs and its bank.

    Args:
        model: The Stage II checkpoint, in ``eval()`` mode on ``device``.
        pairs: The rows to read.
        pack_dir: The packed feature directory.
        device: Compute device.
        amp: Encoder autocast mode.
        token_budget: Scoring token budget.

    Returns:
        ``(cached inputs, (n, 96) predicted bank)``, both row-aligned with ``pairs``.

    Raises:
        RuntimeError: If the capture did not cover every row exactly once.
    """
    with capture_generator_inputs(model) as captured:
        bank, _ = predicted_bank(
            model, pairs, pack_dir, device=device, amp=amp, token_budget=token_budget
        )
    batches = cast(list[list[int]], captured["batches"])
    calls = cast(list[list[tuple[torch.Tensor, torch.Tensor, int, int]]], captured["calls"])
    if len(batches) != len(calls):
        raise RuntimeError(f"captured {len(calls)} generator calls for {len(batches)} batches")
    states_u: list[torch.Tensor | None] = [None] * len(pairs)
    states_v: list[torch.Tensor | None] = [None] * len(pairs)
    lengths_u = torch.zeros(len(pairs), dtype=torch.int64)
    lengths_v = torch.zeros(len(pairs), dtype=torch.int64)
    for indices, call in zip(batches, calls, strict=True):
        for row, (state_u, state_v, length_u, length_v) in zip(indices, call, strict=True):
            states_u[row] = state_u
            states_v[row] = state_v
            lengths_u[row] = length_u
            lengths_v[row] = length_v
    if any(state is None for state in (*states_u, *states_v)):
        raise RuntimeError("the generator capture missed at least one row")
    dtype = ",".join(cast(list[str], captured["dtypes"])) or "unknown"
    return (
        CachedInputs(
            states_u=cast(list[torch.Tensor], states_u),
            states_v=cast(list[torch.Tensor], states_v),
            lengths_u=lengths_u,
            lengths_v=lengths_v,
            dtype=dtype,
        ),
        bank,
    )


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


def closure_counts(targets: torch.Tensor) -> NDArray[np.int64]:
    """Return each row's number of witnesses carrying a non-zero closure weight.

    Args:
        targets: ``(n, 96)`` compiled templates.

    Returns:
        ``(n,)`` counts in ``[0, 8]``.
    """
    value = targets.float()
    present = (value[:, CLOSURE_U] > 0.0) | (value[:, CLOSURE_V] > 0.0)
    return present.sum(dim=1).numpy().astype(np.int64)


def stratum_of(counts: NDArray[np.int64]) -> list[str]:
    """Return each row's stratum name.

    Args:
        counts: Per-row closure counts.

    Returns:
        One stratum name per row.
    """
    names: list[str] = []
    for count in counts.tolist():
        for name, low, high in STRATA:
            if low <= count <= high:
                names.append(name)
                break
        else:  # pragma: no cover - the strata cover [0, 8] by construction
            raise ValueError(f"closure count {count} falls outside the strata")
    return names


def stratified_sample(counts: NDArray[np.int64], *, rows: int, seed: int) -> NDArray[np.int64]:
    """Draw as even a sample across the closure-count strata as the data allows.

    A stratum with fewer rows than its share gives the remainder back, which is
    redistributed over the strata that still have spare rows -- so a universe
    whose eight-witness rows are rare still contributes every one of them rather
    than failing the draw.

    Args:
        counts: Per-row closure counts of the candidate pool.
        rows: Total rows wanted; ``0`` or more than available takes everything.
        seed: Sampling seed.

    Returns:
        Sorted row positions into the pool.

    Raises:
        ValueError: If ``rows`` is negative.
    """
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    pools = {name: np.flatnonzero((counts >= low) & (counts <= high)) for name, low, high in STRATA}
    total = int(sum(pool.size for pool in pools.values()))
    wanted = total if rows == 0 else min(rows, total)
    quota = dict.fromkeys(pools, 0)
    remaining = wanted
    live = [name for name, pool in pools.items() if pool.size > 0]
    while remaining > 0 and live:
        share = max(remaining // len(live), 1)
        for name in list(live):
            if remaining <= 0:
                break
            spare = pools[name].size - quota[name]
            take = min(share, spare, remaining)
            quota[name] += take
            remaining -= take
            if quota[name] >= pools[name].size:
                live.remove(name)
    generator = np.random.default_rng(seed)
    picked: list[int] = []
    for name, pool in pools.items():
        take = quota[name]
        if take <= 0:
            continue
        chosen = pool if take >= pool.size else generator.choice(pool, size=take, replace=False)
        picked.extend(int(value) for value in chosen)
    return np.sort(np.asarray(picked, dtype=np.int64))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    """Return the replay's autocast context (``fp32`` disables it).

    Args:
        device: The device the replay runs on.
        precision: ``bf16`` or ``fp32``.

    Returns:
        The autocast context, enabled only for ``bf16``.
    """
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


def replay(
    generator: MotifGenerator,
    cached: CachedInputs,
    *,
    device: torch.device,
    precision: str,
    chunk: int,
) -> dict[str, object]:
    """Replay every cached row and collect the per-row stage measurements.

    Args:
        generator: The Stage II generator, in ``eval()`` mode on ``device``.
        cached: The universe's cached generator inputs.
        device: Compute device.
        precision: ``bf16`` (the training regime) or ``fp32``.
        chunk: Rows per replayed batch.

    Returns:
        ``spread`` (per stage, role and measurement), ``attention`` (per query
        family and side) and ``weights`` -- the replayed ``(n, 96)`` table.
    """
    spread: dict[str, dict[str, dict[str, list[float]]]] = {}
    attention: dict[str, dict[str, list[float]]] = {}
    emitted: list[torch.Tensor] = []
    for start in range(0, cached.rows(), chunk):
        indices = range(start, min(start + chunk, cached.rows()))
        encoded_u, encoded_v, lengths_u, lengths_v = cached.batch(list(indices), device)
        with torch.no_grad(), _autocast(device, precision):
            stages = decompose(generator, encoded_u, encoded_v, lengths_u, lengths_v)
            for stage, groups in stage_groups(stages).items():
                bucket = spread.setdefault(stage, {})
                for role, states in groups.items():
                    role_cell = bucket.setdefault(role, {"D": [], "scale": [], "maxd": []})
                    role_cell["D"].extend(slot_spread(states).cpu().tolist())
                    role_cell["scale"].extend(slot_scale(states).cpu().tolist())
                    role_cell["maxd"].extend(max_pairwise_distance(states).cpu().tolist())
            for family, queries, block in (
                ("bridge", generator.bridge_queries, generator.bridge_read),
                ("witness", generator.witness_queries, generator.witness_read),
            ):
                for side, raw, state, pad in (
                    ("u", encoded_u, stages.state_u, stages.pad_u),
                    ("v", encoded_v, stages.state_v, stages.pad_v),
                ):
                    measured = _attention_side(generator, raw, state, pad, queries, block)
                    family_cell = attention.setdefault(f"{family}_{side}", {})
                    for key, series in measured.items():
                        family_cell.setdefault(key, []).extend(series.tolist())
            emitted.append(stages.weights.detach().float().cpu())
    return {
        "spread": spread,
        "attention": attention,
        "weights": torch.cat(emitted, dim=0) if emitted else torch.zeros((0, N_EDGES)),
    }


def summarise_replay(
    measured: Mapping[str, object], strata: Sequence[str]
) -> dict[str, dict[str, object]]:
    """Summarise one replay per stratum and overall.

    Args:
        measured: A `replay` payload.
        strata: Per-row stratum names, row-aligned with the replay.

    Returns:
        ``stratum -> stage -> role -> summary`` plus the attention block.
    """
    names = np.asarray(strata)
    keys = ["all", *[name for name, _, _ in STRATA]]
    spread = cast(Mapping[str, Mapping[str, Mapping[str, list[float]]]], measured["spread"])
    attention = cast(Mapping[str, Mapping[str, list[float]]], measured["attention"])
    out: dict[str, dict[str, object]] = {}
    for key in keys:
        mask = np.ones(names.size, dtype=bool) if key == "all" else names == key
        if not mask.any():
            continue
        stages: dict[str, object] = {}
        for stage, roles in spread.items():
            cell: dict[str, dict[str, float]] = {}
            for role, values in roles.items():
                cell[role] = spread_summary(
                    np.asarray(values["D"], dtype=np.float64)[mask],
                    np.asarray(values["scale"], dtype=np.float64)[mask],
                    np.asarray(values["maxd"], dtype=np.float64)[mask],
                )
            stages[stage] = cell
        families: dict[str, dict[str, float]] = {}
        for family, values in attention.items():
            families[family] = {
                name: float(np.mean(np.asarray(series, dtype=np.float64)[mask]))
                for name, series in values.items()
            }
        stages["attention"] = families
        out[key] = stages
    return out


def variance_ratios(summary: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Add the ``proj/H``, ``K/proj`` and ``V/proj`` ratios to each family's block.

    Args:
        summary: The per-family attention block of `summarise_replay`.

    Returns:
        The same block with the three ratios appended.
    """
    out: dict[str, dict[str, float]] = {}
    for family, values in summary.items():
        block = dict(values)
        block["proj_over_H"] = values["proj"] / values["H"] if values["H"] > 0.0 else float("nan")
        scale = values["proj"]
        block["K_over_proj"] = values["K"] / scale if scale > 0.0 else float("nan")
        block["V_over_proj"] = values["V"] / scale if scale > 0.0 else float("nan")
        out[family] = block
    return out


# ---------------------------------------------------------------------------
# Head gain and gradients
# ---------------------------------------------------------------------------


def head_gain(generator: MotifGenerator) -> dict[str, dict[str, float]]:
    """Return each gate head's output gain against its initialisation and its bias.

    Args:
        generator: The Stage II generator.

    Returns:
        Per edge type the last-layer weight norm, the ``1e-3`` init reference,
        their ratio and the bias.
    """
    reference = float(generator.cfg.head_output_init_std) * math.sqrt(GATE_DIM)
    out: dict[str, dict[str, float]] = {}
    for edge_type, name in enumerate(EDGE_TYPE_NAMES):
        gate = cast(nn.Sequential, generator.heads[edge_type])
        output = cast(nn.Linear, gate[-1])
        norm = float(output.weight.detach().float().norm())
        out[name] = {
            "output_weight_norm": norm,
            "init_reference": reference,
            "gain_over_init": norm / reference if reference > 0.0 else float("nan"),
            "bias": float(output.bias.detach().float().reshape(-1)[0]),
            "first_linear_norm": float(cast(nn.Linear, gate[1]).weight.detach().float().norm()),
        }
    return out


def _group_norm(grads: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> float:
    """Return the Euclidean norm of one parameter group's gradient.

    Args:
        grads: Per-parameter gradients, by parameter name.
        prefixes: Name prefixes the group covers.

    Returns:
        The group's gradient norm.
    """
    total = 0.0
    for name, grad in grads.items():
        if name.startswith(tuple(prefixes)):
            total += float(grad.double().pow(2).sum())
    return math.sqrt(total)


def _flatten(grads: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    """Concatenate one ordered set of gradients into a single vector."""
    if not names:
        return torch.zeros(1, dtype=torch.float64)
    return torch.cat([grads[name].double().reshape(-1) for name in names])


def gradient_report(
    generator: MotifGenerator,
    cached: CachedInputs,
    target: torch.Tensor,
    weights: GraphLossWeights,
    *,
    device: torch.device,
    rows: int,
) -> dict[str, object]:
    """Measure where ``L_G`` puts its gradient inside the generator.

    One fp32 forward over a batch of held-out rows supports every backward, so
    the four gradient fields are strictly comparable: the full ``L_G``, the
    closure family ``beta_c L_c + beta_p L_p``, the bridge family
    ``beta_q L_q + beta_a L_a + beta_i L_i``, and the closure family again with
    ``beta_c = 1``.

    Args:
        generator: The Stage II generator, in ``eval()`` mode on ``device``.
        cached: The universe's cached inputs.
        target: ``(n, 96)`` compiled templates, row-aligned with ``cached``.
        weights: The checkpoint's own ``L_G``.
        device: Compute device.
        rows: Rows in the gradient batch.

    Returns:
        Per-group gradient norms, the ratios to the head-bias gradient, and the
        two loss families' norms and cosine under both beta settings.
    """
    count = min(rows, cached.rows())
    encoded_u, encoded_v, lengths_u, lengths_v = cached.batch(list(range(count)), device)
    truth = target[:count].to(device=device, dtype=torch.float32)
    generator.requires_grad_(True)
    named = [(name, param) for name, param in generator.named_parameters() if param.requires_grad]
    parameters = [param for _, param in named]
    with torch.autocast(device_type=device.type, enabled=False):
        predicted = generator(encoded_u, encoded_v, lengths_u, lengths_v)

    def grads_of(loss_weights: GraphLossWeights) -> dict[str, torch.Tensor]:
        """Return the gradient of one beta setting's loss, by parameter name.

        Args:
            loss_weights: The betas the loss is formed with.

        Returns:
            One CPU gradient tensor per trainable parameter; an unused parameter
            gets zeros so every group's norm stays comparable across settings.
        """
        loss = loss_weights.rows(predicted, truth).mean()
        computed = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        out: dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named, computed, strict=True):
            out[name] = (
                torch.zeros_like(param, device="cpu") if grad is None else grad.detach().cpu()
            )
        return out

    closure_weights = replace(
        weights, beta_p=weights.beta_p, beta_q=0.0, beta_a=0.0, beta_i=0.0, beta_c=weights.beta_c
    )
    closure_one = replace(
        weights, beta_p=weights.beta_p, beta_q=0.0, beta_a=0.0, beta_i=0.0, beta_c=1.0
    )
    bridge_weights = replace(
        weights,
        beta_p=0.0,
        beta_q=weights.beta_q,
        beta_a=weights.beta_a,
        beta_i=weights.beta_i,
        beta_c=0.0,
    )
    full = grads_of(weights)
    closure = grads_of(closure_weights)
    closure_forced = grads_of(closure_one)
    bridge = grads_of(bridge_weights)

    bias_names = [name for name in full if name.startswith("heads.") and name.endswith(".3.bias")]
    bias_norm = _group_norm(full, bias_names) if bias_names else float("nan")
    groups: dict[str, float] = {
        name: _group_norm(full, prefixes) for name, prefixes in UPSTREAM_GROUPS.items()
    }
    heads: dict[str, dict[str, float]] = {}
    for edge_type, name in enumerate(EDGE_TYPE_NAMES):
        heads[name] = {
            "output_weight": _group_norm(full, (f"heads.{edge_type}.3.weight",)),
            "output_bias": _group_norm(full, (f"heads.{edge_type}.3.bias",)),
            "first_linear": _group_norm(full, (f"heads.{edge_type}.1.",)),
        }
    shared = sorted(name for name in full if name.startswith(SHARED_PREFIXES))
    families: dict[str, dict[str, float]] = {}
    for label, closure_grads in (("checkpoint_betas", closure), ("beta_c_1", closure_forced)):
        left = _flatten(closure_grads, shared)
        right = _flatten(bridge, shared)
        left_norm, right_norm = float(left.norm()), float(right.norm())
        denominator = left_norm * right_norm
        families[label] = {
            "closure_norm": left_norm,
            "bridge_norm": right_norm,
            "cosine": float(left.dot(right)) / denominator if denominator > 0.0 else float("nan"),
            "closure_over_bridge": (left_norm / right_norm if right_norm > 0.0 else float("nan")),
        }
    return {
        "rows": count,
        "head_bias_grad_norm": bias_norm,
        "heads": heads,
        "upstream": groups,
        "upstream_over_bias": {
            name: value / bias_norm if bias_norm > 0.0 else float("nan")
            for name, value in groups.items()
        },
        "families_on_shared_parameters": families,
        "shared_parameters": len(shared),
    }


# ---------------------------------------------------------------------------
# Per-stratum fit
# ---------------------------------------------------------------------------


def stratum_fit(
    predicted: torch.Tensor,
    target: torch.Tensor,
    strata: Sequence[str],
    weights: GraphLossWeights,
    *,
    fitted_constant: torch.Tensor,
    mean_template: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compare the generator with the fitted constant and the mean, per stratum.

    Args:
        predicted: ``(n, 96)`` predicted weights.
        target: ``(n, 96)`` compiled templates.
        strata: Per-row stratum names.
        weights: The checkpoint's own ``L_G``.
        fitted_constant: The pilot-B constant re-fitted on node-disjoint rows.
        mean_template: The corpus mean adjacency the checkpoint carries.

    Returns:
        One entry per stratum plus ``all``.
    """
    names = np.asarray(strata)
    generator_rows = weights.rows(predicted, target).numpy().astype(np.float64)
    constant_rows = (
        weights.rows(fitted_constant.unsqueeze(0).expand_as(predicted), target)
        .numpy()
        .astype(np.float64)
    )
    mean_rows = (
        weights.rows(mean_template.unsqueeze(0).expand_as(predicted), target)
        .numpy()
        .astype(np.float64)
    )
    ours = count_statistics(predicted)
    theirs = count_statistics(target)
    out: dict[str, dict[str, float]] = {}
    for key in ["all", *[name for name, _, _ in STRATA]]:
        mask = np.ones(names.size, dtype=bool) if key == "all" else names == key
        if not mask.any():
            continue
        entry = {
            "rows": float(mask.sum()),
            "L_G_generator": float(np.mean(generator_rows[mask])),
            "L_G_fitted_constant": float(np.mean(constant_rows[mask])),
            "L_G_mean_template": float(np.mean(mean_rows[mask])),
        }
        for mass in ("wedge_mass", "bridge_mass"):
            mine = ours[mass].numpy().astype(np.float64)[mask]
            truth = theirs[mass].numpy().astype(np.float64)[mask]
            entry[f"predicted_{mass}"] = float(np.mean(mine))
            entry[f"true_{mass}"] = float(np.mean(truth))
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Universe:
    """One universe's probe rows, targets and provenance."""

    name: str
    pairs: list[Pair]
    target: torch.Tensor
    strata: list[str]
    pool_rows: int
    eligible_rows: int


def packed_node_lengths(pack_dir: Path) -> dict[str, int]:
    """Return each packed node's true residue length.

    Args:
        pack_dir: The packed feature directory.

    Returns:
        ``node id -> length`` over the pack's manifest.
    """
    manifest = load_packed_manifest(pack_dir)
    return {record.node_id: int(record.length) for record in manifest.nodes}


def _short_enough(pairs: Sequence[Pair], lengths: Mapping[str, int], cap: int) -> NDArray[np.bool_]:
    """Return the mask of pairs whose two endpoints both fit under ``cap``.

    The scorer sizes one fp32 encoding cache over the *whole* pack at the length
    bucket of the longest node it is asked for, so a single 1002-residue row
    costs 20 GiB of device memory while a training chain holds the rest of the
    card. Capping the probe's rows at a lower bucket is what keeps the
    measurement runnable beside a running job; the kept fraction is reported.

    Args:
        pairs: The candidate pairs.
        lengths: The pack's node lengths.
        cap: The longest packed node a probe row may touch; ``0`` keeps all.

    Returns:
        The boolean keep mask.
    """
    if cap <= 0:
        return np.ones(len(pairs), dtype=bool)
    return np.asarray(
        [lengths.get(u, cap + 1) <= cap and lengths.get(v, cap + 1) <= cap for u, v in pairs],
        dtype=bool,
    )


def _draw_universe(
    name: str,
    pairs: Sequence[Pair],
    target: torch.Tensor,
    *,
    rows: int,
    seed: int,
    keep: NDArray[np.bool_],
) -> Universe:
    """Draw one universe's stratified probe rows from the eligible candidates.

    Args:
        name: Universe name.
        pairs: The candidate pool, row-aligned with ``target``.
        target: ``(n, 96)`` compiled templates of the pool.
        rows: Rows to keep.
        seed: Sampling seed.
        keep: Which candidates the length cap leaves eligible.

    Returns:
        The drawn universe.
    """
    eligible = np.flatnonzero(keep)
    counts = closure_counts(target.index_select(0, torch.from_numpy(eligible)))
    picked = eligible[stratified_sample(counts, rows=rows, seed=seed)]
    kept = target.index_select(0, torch.from_numpy(picked))
    return Universe(
        name=name,
        pairs=[pairs[int(index)] for index in picked],
        target=kept,
        strata=stratum_of(closure_counts(kept)),
        pool_rows=int(len(pairs)),
        eligible_rows=int(eligible.size),
    )


def _select_universes(
    *,
    data_root: Path,
    strategy: str,
    rows: int,
    seed: int,
    pool_rows: int,
    fit_rows: int,
    node_lengths: Mapping[str, int],
    max_node_length: int,
) -> tuple[list[Universe], torch.Tensor]:
    """Draw the stratified rows of both universes and the constant's fit targets.

    Args:
        data_root: Benchmark data root.
        strategy: Split strategy.
        rows: Probe rows per universe.
        seed: Master seed.
        pool_rows: Candidate held-out training rows the strata are drawn from.
        fit_rows: Training rows the asymmetric constant is fitted on.
        node_lengths: The pack's node lengths.
        max_node_length: The longest packed node a probe row may touch.

    Returns:
        ``([held-out training, val_cls], fit targets)``.
    """
    split = su._load_val_region_split(data_root, strategy)
    fit_nodes, heldout_nodes = split_training_nodes(sorted(split.train_nodes), seed=seed)
    fit_pairs, _ = rows_inside(
        sorted(split.training_positives),
        list(split.training_negatives),
        fit_nodes,
        limit=fit_rows,
        seed=seed + 1,
    )
    pool_pairs, _ = rows_inside(
        sorted(split.training_positives),
        list(split.training_negatives),
        heldout_nodes,
        limit=pool_rows,
        seed=seed + 2,
    )
    training_table = MotifTemplateTable(split.build_training_graph())
    fit_target = torch.from_numpy(training_table.weights(fit_pairs))
    pool_target = torch.from_numpy(training_table.weights(pool_pairs))
    heldout = _draw_universe(
        "heldout_train",
        pool_pairs,
        pool_target,
        rows=rows,
        seed=seed + 3,
        keep=_short_enough(pool_pairs, node_lengths, max_node_length),
    )

    val_pairs, _ = su._resolve_pairs("val_cls", data_root, strategy)
    val_table = MotifTemplateTable(
        su._oracle_truth_graph_for_scoring("val_cls", data_root, strategy)
    )
    nonself = [(u, v) for u, v in val_pairs if u != v]
    val_target = torch.from_numpy(val_table.weights(nonself))
    val = _draw_universe(
        "val_cls",
        nonself,
        val_target,
        rows=rows,
        seed=seed + 4,
        keep=_short_enough(nonself, node_lengths, max_node_length),
    )
    logger.info(
        "rows: held-out training %d of %d candidates, val_cls %d of %d nonself",
        len(heldout.pairs),
        pool_target.size(0),
        len(val.pairs),
        val_target.size(0),
    )
    return [heldout, val], fit_target


def run_probe(
    *,
    checkpoint: Path,
    pack_dir: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    rows: int,
    pool_rows: int,
    fit_rows: int,
    chunk: int,
    grad_rows: int,
    max_node_length: int,
    device: torch.device,
    amp: str,
    token_budget: int,
    seed: int,
    fit_steps: int,
) -> dict[str, object]:
    """Probe one Stage II checkpoint's generator and return the full report.

    Args:
        checkpoint: The Stage II checkpoint.
        pack_dir: The packed feature directory.
        data_root: Benchmark data root.
        strategy: Split strategy.
        output_dir: Destination directory.
        rows: Probe rows per universe.
        pool_rows: Candidate held-out training rows.
        fit_rows: Rows the asymmetric constant is fitted on.
        chunk: Rows per replayed batch.
        grad_rows: Rows in the gradient batch.
        max_node_length: The longest packed node a probe row may touch.
        device: Compute device.
        amp: Encoder autocast mode of the capturing scoring pass.
        token_budget: Scoring token budget.
        seed: Master seed.
        fit_steps: Adam steps for the constant template.

    Returns:
        The ``probe.json`` payload.

    Raises:
        ValueError: If the checkpoint is not a Stage II motif prompt.
    """
    torch.manual_seed(seed)
    loaded, family, checkpoint_id = su._load_checkpoint(checkpoint)
    if family != su.MOTIF_PROMPT_FAMILY:
        raise ValueError(f"{checkpoint}: model_family {family!r}, expected a motif prompt")
    model = cast(V3_1MotifPrompt, loaded).to(device).eval()
    generator = model.generator
    if model.cfg.stage != "two" or generator is None:
        raise ValueError(f"{checkpoint}: the generator probe reads a Stage II checkpoint")
    weights = GraphLossWeights.from_config(model.cfg)
    mean_template = model.mean_template.detach().float().cpu()

    node_lengths = packed_node_lengths(pack_dir)
    universes, fit_target = _select_universes(
        data_root=data_root,
        strategy=strategy,
        rows=rows,
        seed=seed,
        pool_rows=pool_rows,
        fit_rows=fit_rows,
        node_lengths=node_lengths,
        max_node_length=max_node_length,
    )
    logger.info("fitting the asymmetric constant template on %d rows", fit_target.size(0))
    fitted = fit_constant_template(fit_target, weights, steps=fit_steps, device=device)

    precisions = ["fp32"] + (["bf16"] if device.type == "cuda" else [])
    report: dict[str, object] = {
        "probe": "motif_generator_probe",
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "run_name": checkpoint.parent.name,
        "strategy": strategy,
        "seed": seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "gate_mode": model.cfg.gate_mode,
        "families": list(model.cfg.families),
        "loss_weights": {
            "beta_p": weights.beta_p,
            "beta_q": weights.beta_q,
            "beta_a": weights.beta_a,
            "beta_i": weights.beta_i,
            "beta_c": weights.beta_c,
            "huber_delta": weights.huber_delta,
        },
        "precisions": precisions,
        "max_node_length": max_node_length,
        "scoring_note": (
            "the formal packed path runs the generator with autocast off over an fp32 "
            "encoding cache, so the bf16 replay is the trainer's regime, not the scorer's"
        ),
        "head_gain": head_gain(generator),
        "query_norms": query_norms(generator),
    }
    spread_block: dict[str, dict[str, dict[str, object]]] = {key: {} for key in precisions}
    attention_block: dict[str, dict[str, dict[str, object]]] = {key: {} for key in precisions}
    fidelity: dict[str, float] = {}
    row_block: dict[str, dict[str, object]] = {}
    fit_block: dict[str, object] = {}
    gradients: dict[str, object] = {}

    for universe in universes:
        logger.info("caching generator inputs for %s (%d rows)", universe.name, len(universe.pairs))
        cached, bank = cache_inputs(
            model,
            universe.pairs,
            pack_dir,
            device=device,
            amp=amp,
            token_budget=token_budget,
        )
        counts = {name: universe.strata.count(name) for name, _, _ in STRATA}
        row_block[universe.name] = {
            "rows": len(universe.pairs),
            "strata": counts,
            "candidate_rows": universe.pool_rows,
            "eligible_under_length_cap": universe.eligible_rows,
            "cached_input_dtype": cached.dtype,
            "cached_input_mib": cached.megabytes(),
        }
        head_rows = list(range(min(chunk, cached.rows())))
        fidelity[universe.name] = check_decomposition(generator, *cached.batch(head_rows, device))
        for precision in precisions:
            measured = replay(generator, cached, device=device, precision=precision, chunk=chunk)
            summary = summarise_replay(measured, universe.strata)
            for stratum, stages in summary.items():
                attention = cast(dict[str, dict[str, float]], stages.pop("attention"))
                spread_block[precision].setdefault(universe.name, {})[stratum] = stages
                attention_block[precision].setdefault(universe.name, {})[stratum] = variance_ratios(
                    attention
                )
            replayed = cast(torch.Tensor, measured["weights"])
            key = f"replay_{precision}_vs_scoring_bank_max_abs_diff"
            row_block[universe.name][key] = float((replayed - bank).abs().max())
        fit_block[universe.name] = stratum_fit(
            bank.float(),
            universe.target.float(),
            universe.strata,
            weights,
            fitted_constant=fitted,
            mean_template=mean_template,
        )
        if universe.name == "heldout_train":
            gradients = gradient_report(
                generator,
                cached,
                universe.target,
                weights,
                device=device,
                rows=grad_rows,
            )
        del cached

    report["universes"] = row_block
    report["decomposition_max_abs_diff"] = fidelity
    report["slot_spread"] = spread_block
    report["input_and_attention"] = attention_block
    report["gradients"] = gradients
    report["stratum_fit"] = fit_block
    report["fitted_constant_template"] = {
        "steps": fit_steps,
        "rows": int(fit_target.size(0)),
        "weights": [float(value) for value in fitted],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CHAIN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reads", ("bridge_u", "bridge_v", "witness_u", "witness_v")),
    ("mix", ("C",)),
    ("mp1", ("C", "L", "R")),
    ("mp2", ("C", "L", "R")),
    ("head_features", EDGE_TYPE_NAMES),
    ("logits", EDGE_TYPE_NAMES),
    ("weights", EDGE_TYPE_NAMES),
)


def _number(value: object, digits: int = 4) -> str:
    """Render one JSON scalar for a markdown cell."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number != number:
            return "nan"
        if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e5):
            return f"{number:.3e}"
        return f"{number:.{digits}f}"
    return str(value)


def markdown_summary(report: Mapping[str, object]) -> str:
    """Render the probe's tables.

    Args:
        report: A `run_probe` payload.

    Returns:
        The markdown document.
    """
    universes = cast(Mapping[str, Mapping[str, object]], report["universes"])
    lines = [
        "# Motif Stage II generator probe",
        "",
        f"Checkpoint `{report['run_name']}` (`{report['checkpoint_id']}`), "
        f"gate mode `{report['gate_mode']}`, betas "
        f"`{json.dumps(report['loss_weights'], sort_keys=True)}`.",
        "",
        f"{report['scoring_note']}.",
        "",
        f"Rows touch only packed nodes of at most {report['max_node_length']} residues "
        "(0 means no cap).",
        "",
        "| universe | rows | strata 0 / 1-2 / 3-7 / 8 | eligible / candidates | cached dtype | "
        "cache MiB | decomposition max abs diff |",
        "|---|---|---|---|---|---|---|",
    ]
    fidelity = cast(Mapping[str, float], report["decomposition_max_abs_diff"])
    for name, block in universes.items():
        strata = cast(Mapping[str, int], block["strata"])
        counts = " / ".join(str(strata.get(key, 0)) for key, _, _ in STRATA)
        lines.append(
            f"| {name} | {block['rows']} | {counts} | "
            f"{block['eligible_under_length_cap']} / {block['candidate_rows']} | "
            f"`{block['cached_input_dtype']}` | "
            f"{_number(block['cached_input_mib'], 1)} | {_number(fidelity.get(name), 8)} |"
        )

    spread = cast(Mapping[str, Mapping[str, Mapping[str, object]]], report["slot_spread"])
    for precision, per_universe in spread.items():
        for universe, strata_block in per_universe.items():
            stages = cast(Mapping[str, Mapping[str, Mapping[str, float]]], strata_block["all"])
            lines += [
                "",
                f"## Slot spread chain -- {universe}, {precision}",
                "",
                "| stage | role | D | D/scale | <1e-6 | <1e-4 | <1e-2 |",
                "|---|---|---|---|---|---|---|",
            ]
            for stage, roles in _CHAIN:
                block = stages.get(stage, {})
                for role in roles:
                    cell = block.get(role)
                    if cell is None:
                        continue
                    lines.append(
                        f"| {stage} | {role} | {_number(cell['D'], 6)} | "
                        f"{_number(cell['D_relative'], 6)} | "
                        f"{_number(cell['frac_identical_below_1e-06'], 3)} | "
                        f"{_number(cell['frac_identical_below_1e-04'], 3)} | "
                        f"{_number(cell['frac_identical_below_1e-02'], 3)} |"
                    )

    for precision, per_universe in spread.items():
        lines += [
            "",
            f"## Closure-slot spread by stratum -- {precision}",
            "",
            "| universe | stratum | mix D | mp2 C D | weights closure D | "
            "weights closure D/scale |",
            "|---|---|---|---|---|---|",
        ]
        for universe, strata_block in per_universe.items():
            for stratum, stages_object in strata_block.items():
                stages = cast(Mapping[str, Mapping[str, Mapping[str, float]]], stages_object)
                mix = stages.get("mix", {}).get("C", {})
                mp2 = stages.get("mp2", {}).get("C", {})
                emitted = stages.get("weights", {}).get("closure", {})
                lines.append(
                    f"| {universe} | {stratum} | {_number(mix.get('D'), 6)} | "
                    f"{_number(mp2.get('D'), 6)} | {_number(emitted.get('D'), 6)} | "
                    f"{_number(emitted.get('D_relative'), 6)} |"
                )

    attention = cast(
        Mapping[str, Mapping[str, Mapping[str, object]]], report["input_and_attention"]
    )
    lines += [
        "",
        "## Input variance and attention",
        "",
        "| precision | universe | family | H | proj/H | K/proj | V/proj | logit std | "
        "entropy/log L | read cosine |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for precision, per_universe in attention.items():
        for universe, strata_block in per_universe.items():
            families = cast(Mapping[str, Mapping[str, float]], strata_block["all"])
            for name, values in families.items():
                lines.append(
                    f"| {precision} | {universe} | {name} | {_number(values['H'], 5)} | "
                    f"{_number(values['proj_over_H'], 4)} | {_number(values['K_over_proj'], 4)} | "
                    f"{_number(values['V_over_proj'], 4)} | {_number(values['logit_std'], 4)} | "
                    f"{_number(values['entropy_ratio'], 4)} | {_number(values['read_cosine'], 5)} |"
                )

    gains = cast(Mapping[str, Mapping[str, float]], report["head_gain"])
    lines += [
        "",
        "## Head gain and bias",
        "",
        "| edge type | ||v_t|| | init ref | gain/init | bias | first linear |",
        "|---|---|---|---|---|---|",
    ]
    for name, values in gains.items():
        lines.append(
            f"| {name} | {_number(values['output_weight_norm'], 6)} | "
            f"{_number(values['init_reference'], 6)} | {_number(values['gain_over_init'], 3)} | "
            f"{_number(values['bias'], 4)} | {_number(values['first_linear_norm'], 4)} |"
        )

    gradients = cast(Mapping[str, object], report["gradients"])
    if gradients:
        heads = cast(Mapping[str, Mapping[str, float]], gradients["heads"])
        lines += [
            "",
            f"## Gradient of L_G ({gradients['rows']} held-out rows, fp32)",
            "",
            "| head | d/d output weight | d/d output bias | d/d first linear |",
            "|---|---|---|---|",
        ]
        for name, values in heads.items():
            lines.append(
                f"| {name} | {_number(values['output_weight'], 6)} | "
                f"{_number(values['output_bias'], 6)} | {_number(values['first_linear'], 6)} |"
            )
        upstream = cast(Mapping[str, float], gradients["upstream"])
        ratios = cast(Mapping[str, float], gradients["upstream_over_bias"])
        lines += [
            "",
            f"Head-bias gradient norm: {_number(gradients['head_bias_grad_norm'], 6)}.",
            "",
            "| upstream group | grad norm | over bias |",
            "|---|---|---|",
        ]
        for name, value in upstream.items():
            lines.append(f"| {name} | {_number(value, 6)} | {_number(ratios[name], 4)} |")
        families = cast(
            Mapping[str, Mapping[str, float]], gradients["families_on_shared_parameters"]
        )
        lines += [
            "",
            "| betas | ||g_C|| | ||g_B|| | cos(g_C, g_B) | ||g_C||/||g_B|| |",
            "|---|---|---|---|---|",
        ]
        for label, values in families.items():
            lines.append(
                f"| {label} | {_number(values['closure_norm'], 6)} | "
                f"{_number(values['bridge_norm'], 6)} | {_number(values['cosine'], 4)} | "
                f"{_number(values['closure_over_bridge'], 4)} |"
            )

    fit = cast(Mapping[str, Mapping[str, Mapping[str, float]]], report["stratum_fit"])
    lines += [
        "",
        "## Fit per closure-count stratum (formal predicted bank)",
        "",
        "| universe | stratum | rows | L_G gen | L_G constant | L_G mean | pred wedge | "
        "true wedge | pred bridge | true bridge |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for universe, strata_block in fit.items():
        for stratum, values in strata_block.items():
            lines.append(
                f"| {universe} | {stratum} | {int(values['rows'])} | "
                f"{_number(values['L_G_generator'], 5)} | "
                f"{_number(values['L_G_fitted_constant'], 5)} | "
                f"{_number(values['L_G_mean_template'], 5)} | "
                f"{_number(values['predicted_wedge_mass'], 5)} | "
                f"{_number(values['true_wedge_mass'], 5)} | "
                f"{_number(values['predicted_bridge_mass'], 5)} | "
                f"{_number(values['true_bridge_mass'], 5)} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure, layer by layer, where a Stage II motif generator loses the "
            "distinction between its slots."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=600, help="probe rows per universe")
    parser.add_argument("--pool-rows", type=int, default=60_000)
    parser.add_argument("--fit-rows", type=int, default=20_000)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--grad-rows", type=int, default=64)
    parser.add_argument(
        "--max-node-length",
        type=int,
        default=512,
        help=(
            "skip rows touching a longer packed node; the scorer's fp32 encoding cache "
            "is sized by the longest node it is asked for (0 disables the cap)"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="bf16", choices=["off", "bf16"])
    parser.add_argument("--token-budget", type=int, default=131_072)
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the probe and write ``probe.json`` and ``probe.md``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    report = run_probe(
        checkpoint=args.checkpoint,
        pack_dir=args.pack_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        rows=args.rows,
        pool_rows=args.pool_rows,
        fit_rows=args.fit_rows,
        chunk=args.chunk,
        grad_rows=args.grad_rows,
        max_node_length=args.max_node_length,
        device=su._resolve_device(args.device),
        amp=args.amp,
        token_budget=args.token_budget,
        seed=args.seed,
        fit_steps=args.fit_steps,
    )
    (args.output_dir / "probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "probe.md").write_text(markdown_summary(report), encoding="utf-8")
    logger.info("wrote %s", args.output_dir / "probe.json")


if __name__ == "__main__":
    main()


__all__ = [
    "CachedInputs",
    "GeneratorStages",
    "Universe",
    "attention_side",
    "build_parser",
    "cache_inputs",
    "capture_generator_inputs",
    "check_decomposition",
    "closure_counts",
    "decompose",
    "gradient_report",
    "head_gain",
    "main",
    "markdown_summary",
    "max_pairwise_distance",
    "packed_node_lengths",
    "query_norms",
    "replay",
    "run_probe",
    "slot_scale",
    "slot_spread",
    "spread_summary",
    "stage_groups",
    "stratified_sample",
    "stratum_fit",
    "stratum_of",
    "summarise_replay",
    "variance_ratios",
]
