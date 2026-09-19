"""Motif-graph prompts read by GRIT on a frozen V3.1 trunk.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
Stage I trains the reader, the count head, the token projections, the role
embeddings and the prefix adapter on compiled true training templates; Stage II
trains a residue-conditioned generator that predicts the same 96 edge weights
from ``(x_u, x_v)`` alone, read through the identical interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.motif_template import (
    EDGE_ENDPOINTS,
    EDGE_TYPES,
    N_EDGE_TYPES,
    N_EDGES,
    N_ROLES,
    N_SLOTS,
    SLOT_ROLES,
    SLOT_U,
    SLOT_V,
    TYPE_ATTACH,
    TYPE_CLOSURE,
    TYPE_INTERIOR,
    MotifTemplateStatistics,
    count_statistics,
)
from src.distill.motif_losses import closure_nonempty, slot_loss_rows, topo_loss_rows
from src.model.egostitch.classifier.b0_v31 import V3_1, unpack_pair_batch
from src.model.egostitch.classifier.layers import (
    CrossAttentionLayer,
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)
from src.model.egostitch.classifier.prefix import SITES, prefix_branch
from src.model.egostitch.encoder.grit_gmt import _grit_layer_cfg, _GritBatch, dense_rrwp
from src.vendor.grit_official import GritTransformerLayer

FIELD_ORDER = ("topo_self", "topo_partner", "topo_rel", "topo_cnt")
TEMPLATE_KEY = "motif_weights"
TEMPLATE_MASK_KEY = "motif_mask"
INTERVENTIONS = (
    "none",
    "gates_off",
    "mean",
    "shuffle_graph",
    "permute_closure",
    "rewire_bridge",
)
TRAINING_POLICIES = ("default", "head_only")
STAGES = ("one", "two")
CLOSURE_BIAS_INITS = ("density", "nonzero_mean")
#: Which losses reach the generator while the interface is held. ``joint`` is the
#: wave-1 composite; ``graph_only`` cuts the task, structural and topo losses off
#: from G for the warm-up epochs only; ``graph_only_always`` keeps that cut for
#: the whole run, so the interface opens on schedule onto a generator that is
#: only ever supervised by ``L_G`` (spec section 7.5).
WARMUP_LOSSES = ("joint", "graph_only", "graph_only_always")
#: How the rows of the graph loss are weighted. ``uniform`` is the row
#: distribution of the stream itself; ``closure_balanced`` gives the rows whose
#: closure family is non-empty a fixed share of L_G's mass (spec section 7.5).
GRAPH_ROW_WEIGHTINGS = ("uniform", "closure_balanced")
#: How a slot query reads one endpoint's residue states. ``bare`` is the wave-1
#: and wave-2 read -- the raw `nn.MultiheadAttention` output, whose slot identity
#: rides on the attention weights alone. ``residual_block`` keeps the query on
#: the path (spec section 4, wave-3 fix G1).
SLOT_READS = ("bare", "residual_block")
#: ``w_slot`` may be this string instead of a number: the weight is then fixed
#: once, by gradient-norm balancing at the first interface-open step.
BALANCED_W_SLOT = "balanced"
#: Edge-type names, in `src.data.motif_template`'s type order.
EDGE_TYPE_NAMES: tuple[str, ...] = ("closure", "attach", "interior")
#: Generator parameter groups the per-epoch gradient probe attributes terms to.
GENERATOR_GRAD_GROUPS: tuple[str, ...] = (
    "slot_queries",
    "attention",
    "mpnn",
    "head_closure",
    "head_attach",
    "head_interior",
    "other",
)
TOKEN_SOURCES = ("graph", "direct")
COUNT_FEATURES = ("all", "degree")
GATE_MODES = ("learned", "per_type", "mean_graph")
FAMILIES = ("closure", "bridge")
LOSS_TERM_NAMES = ("task", "slot", "topo")
_GRAPH_INTERVENTIONS = ("shuffle_graph", "permute_closure", "rewire_bridge")
_READER_FIELDS = frozenset({"topo_self", "topo_partner", "topo_rel"})
#: Reader and prefix parameters train at this fraction of the generator's LR
#: once the Stage II warm-up opens them (spec section 7.5).
STAGE_TWO_INTERFACE_LR_SCALE = 0.1


@dataclass(frozen=True)
class ReaderConfig:
    """The GRIT reader block (spec section 5.2)."""

    layers: int = 3
    dim: int = 96
    heads: int = 4
    rrwp_k: int = 4

    def __post_init__(self) -> None:
        """Validate the reader shape.

        Raises:
            ValueError: On a non-positive size or an indivisible width.
        """
        if self.layers <= 0 or self.dim <= 0 or self.heads <= 0:
            raise ValueError("reader.layers, reader.dim and reader.heads must be positive")
        if self.rrwp_k < 1:
            raise ValueError("reader.rrwp_k must be at least 1")
        if self.dim % self.heads:
            raise ValueError(f"reader.dim ({self.dim}) must be divisible by reader.heads")


@dataclass(frozen=True)
class CorruptionConfig:
    """Stationary Stage I adjacency corruption (spec section 3)."""

    prob: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    def __post_init__(self) -> None:
        """Validate the distribution.

        Raises:
            ValueError: On an out-of-range probability or an empty lambda range.
        """
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("motif_prompt.corruption.prob must lie in [0, 1]")
        if not 0.0 <= self.lambda_min <= self.lambda_max <= 1.0:
            raise ValueError("motif_prompt.corruption lambda range must lie in [0, 1]")


@dataclass(frozen=True)
class MotifPromptConfig:
    """The ``model.config.motif_prompt`` block."""

    stage: str = "one"
    base_checkpoint: str = ""
    base_checkpoint_sha256: str | None = None
    bundle_checkpoint: str = ""
    bundle_checkpoint_sha256: str | None = None
    width: int = 128
    slots_per_field: int = 2
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    fields: tuple[str, ...] = FIELD_ORDER
    families: tuple[str, ...] = FAMILIES
    token_source: str = "graph"
    gate_mode: str = "learned"
    count_features: str = "all"
    interface_warmup_epochs: int | None = 2
    warmup_losses: str = "joint"
    w_slot: float | str = 1.0
    w_slot_multiplier: float = 1.0
    balance_probe_rows: int = 256
    w_topo: float = 0.1
    beta_p: float = 1.0
    beta_q: float = 1.0
    beta_a: float = 1.0
    beta_i: float = 1.0
    beta_c: float = 0.0
    closure_bias_init: str = "density"
    graph_row_weighting: str = "uniform"
    graph_row_positive_share: float = 0.5
    huber_delta: float = 1.0
    slot_read: str = "bare"
    slot_read_value_norm: bool = True
    slot_query_init_std: float | None = None
    head_output_init_std: float = 1e-3
    training_policy: str = "default"
    head_prompt_enabled: bool = True
    init_checkpoint: str = ""

    def __post_init__(self) -> None:
        """Validate the block.

        Raises:
            ValueError: On an unknown enum value, a missing required checkpoint,
                a duplicated or empty field/family list, or a negative weight.
        """
        if self.stage not in STAGES:
            raise ValueError(f"motif_prompt.stage must be one of {list(STAGES)}")
        if not self.base_checkpoint:
            raise ValueError("motif_prompt requires motif_prompt.base_checkpoint")
        if self.stage == "two" and not self.bundle_checkpoint:
            raise ValueError("stage 'two' requires motif_prompt.bundle_checkpoint")
        if self.width <= 0 or self.slots_per_field <= 0:
            raise ValueError("motif_prompt.width and slots_per_field must be positive")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("motif_prompt.fields must be a non-empty set of field names")
        if any(name not in FIELD_ORDER for name in self.fields):
            raise ValueError(f"motif_prompt.fields must be drawn from {list(FIELD_ORDER)}")
        if not self.families or any(name not in FAMILIES for name in self.families):
            raise ValueError(
                f"motif_prompt.families must be a non-empty subset of {list(FAMILIES)}"
            )
        if self.token_source not in TOKEN_SOURCES:
            raise ValueError(f"motif_prompt.token_source must be one of {list(TOKEN_SOURCES)}")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"motif_prompt.gate_mode must be one of {list(GATE_MODES)}")
        if self.count_features not in COUNT_FEATURES:
            raise ValueError(f"motif_prompt.count_features must be one of {list(COUNT_FEATURES)}")
        if self.interface_warmup_epochs is not None and self.interface_warmup_epochs < 0:
            raise ValueError("motif_prompt.interface_warmup_epochs must be non-negative or null")
        if self.warmup_losses not in WARMUP_LOSSES:
            raise ValueError(f"motif_prompt.warmup_losses must be one of {list(WARMUP_LOSSES)}")
        if self.closure_bias_init not in CLOSURE_BIAS_INITS:
            raise ValueError(
                f"motif_prompt.closure_bias_init must be one of {list(CLOSURE_BIAS_INITS)}"
            )
        if self.graph_row_weighting not in GRAPH_ROW_WEIGHTINGS:
            raise ValueError(
                f"motif_prompt.graph_row_weighting must be one of {list(GRAPH_ROW_WEIGHTINGS)}"
            )
        if not 0.0 < self.graph_row_positive_share < 1.0:
            raise ValueError("motif_prompt.graph_row_positive_share must lie in (0, 1)")
        if isinstance(self.w_slot, str):
            if self.w_slot != BALANCED_W_SLOT:
                raise ValueError(
                    f"motif_prompt.w_slot must be a non-negative number or {BALANCED_W_SLOT!r}"
                )
        elif float(self.w_slot) < 0.0:
            raise ValueError("motif_prompt.w_slot must be non-negative")
        if self.w_slot_multiplier <= 0.0:
            raise ValueError("motif_prompt.w_slot_multiplier must be positive")
        if self.balance_probe_rows < 1:
            raise ValueError("motif_prompt.balance_probe_rows must be at least one row")
        for name in ("w_topo", "beta_p", "beta_q", "beta_a", "beta_i", "beta_c"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"motif_prompt.{name} must be non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("motif_prompt.huber_delta must be positive")
        if self.slot_read not in SLOT_READS:
            raise ValueError(f"motif_prompt.slot_read must be one of {list(SLOT_READS)}")
        if self.slot_query_init_std is not None and float(self.slot_query_init_std) <= 0.0:
            raise ValueError("motif_prompt.slot_query_init_std must be positive or null")
        if self.head_output_init_std <= 0.0:
            raise ValueError("motif_prompt.head_output_init_std must be positive")
        if self.training_policy not in TRAINING_POLICIES:
            raise ValueError(
                f"motif_prompt.training_policy must be one of {list(TRAINING_POLICIES)}"
            )
        if self.training_policy == "head_only" and not self.init_checkpoint:
            raise ValueError("head-only motif_prompt requires motif_prompt.init_checkpoint")

    @property
    def w_slot_is_balanced(self) -> bool:
        """Whether ``w_slot`` is resolved by gradient-norm balancing at run time."""
        return isinstance(self.w_slot, str)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MotifPromptConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.motif_prompt``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown motif_prompt keys: {unknown}")
        values = dict(raw)
        reader = ReaderConfig(**cast(Mapping[str, int], values.pop("reader", {})))
        corruption = CorruptionConfig(**cast(Mapping[str, float], values.pop("corruption", {})))
        fields_raw = values.pop("fields", FIELD_ORDER)
        families_raw = values.pop("families", FAMILIES)
        warmup = values.pop("interface_warmup_epochs", 2)
        return cls(
            reader=reader,
            corruption=corruption,
            fields=tuple(str(name) for name in cast(Sequence[object], fields_raw)),
            families=tuple(str(name) for name in cast(Sequence[object], families_raw)),
            interface_warmup_epochs=None if warmup is None else int(cast(int, warmup)),
            # The remaining keys are heterogeneous scalars validated at
            # runtime by __post_init__; ** unpacking cannot be typed here.
            **values,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))


class MotifCountHead(nn.Module):
    """The retained closed-form count pathway (spec section 5.1).

    Reads the same predicted adjacency as the reader and emits one token from
    four swap-invariant statistics: ``log1p`` wedge mass, ``log1p`` bridge mass,
    the sum and the absolute difference of the two ``log1p`` degrees. The degree
    pair never enters as ``[deg_u, deg_v]``, which would not be invariant under
    ``u<->v``.

    ``degree_only`` zeroes the two mass entries, leaving the symmetric degree
    pair as the only content of ``topo_cnt``. It is the section 8 degree-only
    control: a separately trained arm, identical in every other respect, whose
    prompt carries only degree -- never a degree-marginal re-thresholding.
    """

    def __init__(self, width: int, *, degree_only: bool = False) -> None:
        """Build the single linear map.

        Args:
            width: Token width.
            degree_only: Zero ``wedge_mass`` and ``bridge_mass`` in the token.
        """
        super().__init__()
        self.proj = nn.Linear(4, width)
        self.degree_only = degree_only

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, width)`` count token, always in float32.

        The arithmetic runs with autocast disabled rather than merely on fp32
        inputs: inside an active autocast context the matmul of the projection
        would return bf16 from fp32 operands (spec section 5.1).

        Args:
            weights: ``(B, 96)`` predicted edge weights.

        Returns:
            The ``(B, width)`` count token in float32.
        """
        with torch.autocast(device_type=weights.device.type, enabled=False):
            stats = count_statistics(weights.float())
            log_u = torch.log1p(stats["deg_u"])
            log_v = torch.log1p(stats["deg_v"])
            zero = torch.zeros_like(log_u)
            features = torch.stack(
                [
                    zero if self.degree_only else torch.log1p(stats["wedge_mass"]),
                    zero if self.degree_only else torch.log1p(stats["bridge_mass"]),
                    log_u + log_v,
                    (log_u - log_v).abs(),
                ],
                dim=-1,
            )
            token: torch.Tensor = self.proj(features.float())
            return token.float()


_EDGE_ROWS = torch.as_tensor([i for i, _ in EDGE_ENDPOINTS], dtype=torch.long).view(1, -1)
_EDGE_COLS = torch.as_tensor([j for _, j in EDGE_ENDPOINTS], dtype=torch.long).view(1, -1)
_EDGE_TYPE_IDS = torch.as_tensor(EDGE_TYPES, dtype=torch.long).view(1, -1)


def dense_adjacency(weights: torch.Tensor) -> torch.Tensor:
    """Scatter the 96 weights into a symmetric ``(B, 26, 26)`` adjacency.

    Every unlisted entry, the diagonal and the queried ``u-v`` entry stay exactly
    zero (spec section 2).

    Args:
        weights: ``(B, 96)`` edge weights.

    Returns:
        ``(B, 26, 26)`` symmetric adjacency in ``weights``' dtype.
    """
    batch = weights.size(0)
    index = torch.arange(batch, device=weights.device).view(-1, 1)
    rows = _EDGE_ROWS.to(weights.device)
    cols = _EDGE_COLS.to(weights.device)
    adj = weights.new_zeros(batch, N_SLOTS, N_SLOTS).index_put((index, rows, cols), weights)
    return adj + adj.transpose(1, 2)


def typed_adjacency(weights: torch.Tensor) -> torch.Tensor:
    """Scatter the weights into a symmetric ``(B, 26, 26, 3)`` typed adjacency.

    Args:
        weights: ``(B, 96)`` edge weights.

    Returns:
        ``(B, 26, 26, 3)`` typed adjacency summing to `dense_adjacency`.
    """
    batch = weights.size(0)
    index = torch.arange(batch, device=weights.device).view(-1, 1)
    rows = _EDGE_ROWS.to(weights.device)
    cols = _EDGE_COLS.to(weights.device)
    types = _EDGE_TYPE_IDS.to(weights.device)
    typed = weights.new_zeros(batch, N_SLOTS, N_SLOTS, N_EDGE_TYPES).index_put(
        (index, rows, cols, types), weights
    )
    return typed + typed.transpose(1, 2)


def motif_rrwp(adj: torch.Tensor, k: int) -> torch.Tensor:
    """Compute the RRWP stack in fp32 with autocast explicitly disabled.

    Verified on torch 2.10: ``torch.bmm`` and ``@`` return bf16 from *fp32*
    inputs inside an active autocast context, so feeding `dense_rrwp` an fp32
    tensor is not sufficient -- the whole call must run with autocast disabled
    (spec section 5.2). `dense_rrwp` clamps the degree at ``1e-6`` and keeps the
    autograd chain alive for a learned generator's exactly-zero adjacency; the
    ``mask`` argument is deliberately not passed, because all 26 slots exist in
    both stages (spec section 2).

    Args:
        adj: ``(B, 26, 26)`` untyped adjacency in any dtype.
        k: Number of stacked walk orders, including the identity term.

    Returns:
        ``(B, 26, 26, k)`` float32 RRWP stack.
    """
    with torch.autocast(device_type=adj.device.type, enabled=False):
        return dense_rrwp(adj.float(), k).float()


_ROLE_IDS = torch.as_tensor(SLOT_ROLES, dtype=torch.long)
_ROLE_WIDTH = 16


class MotifGritReader(nn.Module):
    """Vendored GRIT over the 26-slot motif graph, exposing node and pair states.

    The node channel sees role embeddings and the RRWP diagonal only; the edge
    channel sees the RRWP stack and the typed raw weights; the degree input is
    the weighted degree, since walk normalisation alone discards edge scale.
    Neither residue states nor generator hidden states nor slot-index embeddings
    reach it, so an unrestricted pair-feature vector cannot enter disguised as a
    node feature (spec sections 2 and 5.2).

    Every layer is constructed with ``O_e=True`` and ``norm_e=True`` so the final
    layer's edge state is projected and normalised before it is read; the
    vendored layer already writes it to ``batch.edge_attr`` because
    ``cfg["update_e"]`` is set, so ``src/vendor/`` is not edited. Upstream's
    `get_log_deg` is decorated ``@torch.no_grad``, so the degree input carries
    scale but no gradient; the generator is reached through the edge embedding
    and the RRWP stack instead.

    The shared ``node_proj`` and the per-role embedding are what make the
    endpoints exchange under ``u<->v, L<->R``; ``topo_rel`` symmetrises the two
    directed edge states, and the per-token `LayerNorm` keeps every token
    batch-independent.
    """

    def __init__(self, cfg: ReaderConfig, width: int) -> None:
        """Build the RRWP embeddings, the GRIT stack and the token projections.

        Args:
            cfg: The reader block.
            width: Output token width.
        """
        super().__init__()
        self.cfg = cfg
        self.role_embed = nn.Embedding(N_ROLES, _ROLE_WIDTH)
        self.node_embed = nn.Linear(_ROLE_WIDTH + cfg.rrwp_k, cfg.dim)
        self.edge_embed = nn.Linear(cfg.rrwp_k + N_EDGE_TYPES, cfg.dim)
        layer_cfg = _grit_layer_cfg()
        self.layers = nn.ModuleList(
            GritTransformerLayer(  # type: ignore[no-untyped-call]
                in_dim=cfg.dim,
                out_dim=cfg.dim,
                num_heads=cfg.heads,
                dropout=0.0,
                attn_dropout=0.0,
                layer_norm=True,
                batch_norm=False,
                residual=True,
                act="relu",
                norm_e=True,
                O_e=True,
                cfg=layer_cfg,
            )
            for _ in range(cfg.layers)
        )
        self.node_proj = nn.Linear(cfg.dim, width)
        self.pair_proj = nn.Linear(cfg.dim, width)
        self.token_norm = nn.LayerNorm(width)

    def adaptable_modules(self) -> list[nn.Module]:
        """The final block and the output projections: all Stage II may adapt.

        Returns:
            The modules spec section 7.5 opens after the warm-up.
        """
        return [self.layers[-1], self.node_proj, self.pair_proj, self.token_norm]

    def frozen_stage_two_modules(self) -> list[nn.Module]:
        """The role/input embeddings and every earlier block.

        Returns:
            The modules spec section 7.5 keeps frozen for all of Stage II.
        """
        return [self.role_embed, self.node_embed, self.edge_embed, *list(self.layers)[:-1]]

    def _flat_batch(self, weights: torch.Tensor, adj: torch.Tensor) -> _GritBatch:
        """Flatten one batch of motif graphs into the layout GRIT's layer reads."""
        batch = weights.size(0)
        device = weights.device
        stack = motif_rrwp(adj, self.cfg.rrwp_k)
        diagonal = torch.diagonal(stack, dim1=1, dim2=2).transpose(1, 2)
        roles = self.role_embed(_ROLE_IDS.to(device)).unsqueeze(0).expand(batch, -1, -1)
        node_input = torch.cat((roles, diagonal.to(roles.dtype)), dim=-1)
        pair_input = torch.cat((stack, typed_adjacency(weights)), dim=-1)

        grid = torch.arange(N_SLOTS, device=device)
        src = grid.repeat_interleave(N_SLOTS).repeat(batch)
        dst = grid.repeat(N_SLOTS).repeat(batch)
        offsets = (torch.arange(batch, device=device) * N_SLOTS).repeat_interleave(N_SLOTS**2)
        edge_index = torch.stack((src + offsets, dst + offsets))

        flat_x = self.node_embed(node_input.reshape(batch * N_SLOTS, -1))
        edge_attr = self.edge_embed(pair_input.reshape(batch * N_SLOTS**2, -1).to(flat_x.dtype))
        weighted_degree = adj.sum(dim=-1).reshape(batch * N_SLOTS, 1)
        log_deg = torch.log(weighted_degree.float() + 1.0).to(flat_x.dtype)
        return _GritBatch(flat_x, edge_index, edge_attr, log_deg)

    def forward(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read one batch of motif graphs.

        Args:
            weights: ``(B, 96)`` edge weights.

        Returns:
            ``topo_u``, ``topo_v`` and the swap-invariant ``topo_rel``.
        """
        batch = weights.size(0)
        adj = dense_adjacency(weights)
        flat = self._flat_batch(weights, adj)
        for layer in self.layers:
            flat = layer(flat)
        nodes = flat.x.view(batch, N_SLOTS, -1)
        edges = flat.edge_attr.view(batch, N_SLOTS, N_SLOTS, -1)
        relation = 0.5 * (edges[:, SLOT_U, SLOT_V] + edges[:, SLOT_V, SLOT_U])
        return {
            "topo_u": self.token_norm(self.node_proj(nodes[:, SLOT_U])),
            "topo_v": self.token_norm(self.node_proj(nodes[:, SLOT_V])),
            "topo_rel": self.token_norm(self.pair_proj(relation)),
        }


_GATE_DIM = 96
_BIAS_CLIP = (0.01, 0.99)
_TYPE_MASKS = tuple(
    torch.as_tensor([kind == edge_type for kind in EDGE_TYPES]) for edge_type in range(N_EDGE_TYPES)
)


def _family_mask(families: Sequence[str]) -> torch.Tensor:
    """Return the ``(96,)`` 0/1 mask keeping only the active motif families."""
    mask = torch.ones(N_EDGES)
    if "closure" not in families:
        mask[:16] = 0.0
    if "bridge" not in families:
        mask[16:] = 0.0
    return mask


def _candidate_incidence() -> torch.Tensor:
    """Return the row-normalised ``(3, 26, 26)`` incidence of the fixed candidate template."""
    incidence = torch.zeros(N_EDGE_TYPES, N_SLOTS, N_SLOTS)
    for edge, (i, j) in enumerate(EDGE_ENDPOINTS):
        incidence[EDGE_TYPES[edge], i, j] = 1.0
        incidence[EDGE_TYPES[edge], j, i] = 1.0
    return incidence / incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)


class _MessageLayer(nn.Module):
    """One shared residual message-passing layer over the fixed candidate template.

    Three relation transforms, one per *edge type* -- closure, attachment and
    interior. Both sides of a family share a transform, which is what makes the
    whole generator equivariant under ``u<->v, L<->R`` (spec section 4).
    """

    def __init__(self, dim: int) -> None:
        """Build the relation transforms and the update MLP.

        Args:
            dim: Slot-state width.
        """
        super().__init__()
        self.relations = nn.ModuleList(nn.Linear(dim, dim) for _ in range(N_EDGE_TYPES))
        self.update = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, h: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """Aggregate typed mean messages and apply the residual update.

        Args:
            h: ``(B, 26, dim)`` slot states.
            incidence: ``(3, 26, 26)`` row-normalised candidate incidence per type.

        Returns:
            Updated ``(B, 26, dim)`` states.
        """
        messages = torch.zeros_like(h)
        for index, relation in enumerate(self.relations):
            messages = messages + incidence[index].to(h.dtype) @ relation(h)
        update: torch.Tensor = self.update(messages)
        return h + update


class _ResidualReadBlock(nn.Module):
    """A pre-norm, query-preserving read of one endpoint's residue states.

    The wave-1/wave-2 read was the bare attention output ``r_k = MHA(q_k, H, H)``:
    the query never reaches the result, so a slot's identity rides entirely on
    its attention weights and a near-uniform attention returns eight copies of
    the same mean of ``V``. That is exactly what the wave-2 prefixes emitted
    (100% of rows with identical closure slots, witness entropy 0.996 of
    ``log L``). This block puts the query back on the path,
    ``Z = Q + MHA(LN_q(Q), LN_h(S), LN_h(S))`` followed by ``Z + FFN(LN_z(Z))``,
    so distinct queries give distinct slots whatever the attention does, and the
    queries carry a gradient that does not have to pass through the attention
    weights first.

    One instance is shared by both endpoints of a read, which is what keeps the
    ``u<->v, L<->R`` equivariance of spec section 4 intact.

    ``value_norm`` controls whether the value path is normalised too. With it on
    -- the first form of this block -- ``LN_h`` rescales every residue state to
    unit RMS before the values are formed, which discards exactly the per-residue
    magnitude the bare read carried through as its pair signal, and the wave-3
    prefix duly lost its pair dependence (transplant rise +71% -> +15%, ``V``
    over projection variance 4.87 -> 0.54). With it off the keys are still
    normalised, so the attention logits keep their scale, but the values are the
    raw projected states ``S``.
    """

    def __init__(self, dim: int, *, value_norm: bool = True) -> None:
        """Build the three norms and the position-wise FFN.

        Args:
            dim: Slot-state width.
            value_norm: Read ``LN_h(S)`` as the values; ``False`` reads ``S``.
        """
        super().__init__()
        self.value_norm = value_norm
        self.norm_q = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(
        self,
        queries: torch.Tensor,
        states: torch.Tensor,
        pad: torch.Tensor | None,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        """Read ``states`` with ``queries`` kept on the residual path.

        The attention module is passed in rather than owned so the generator's
        one `nn.MultiheadAttention` stays a single registered parameter set
        whatever mix of blocks reads through it.

        Args:
            queries: ``(B, K, dim)`` expanded slot queries.
            states: ``(B, L, dim)`` projected residue states.
            pad: ``(B, L)`` key padding mask, or ``None``.
            attention: The shared multi-head attention.

        Returns:
            ``(B, K, dim)`` slot states.
        """
        normed = self.norm_h(states)
        values = normed if self.value_norm else states
        read, _ = attention(
            self.norm_q(queries), normed, values, key_padding_mask=pad, need_weights=False
        )
        mixed = queries + read
        out: torch.Tensor = mixed + self.ffn(self.norm_z(mixed))
        return out


class MotifGenerator(nn.Module):
    """Residue-conditioned edge gates over the fixed template (spec section 4).

    Shared slot queries read ``H_u`` and ``H_v`` separately under residue-length
    masks: eight bridge queries used on both sides and eight witness queries read
    on both proteins, combined through the symmetric pair ``[a_u+a_v, |a_u-a_v|]``.
    Gate-head inputs are LayerNormed and the heads are MLPs rather than free edge
    logits, because the predecessor arm's gate died when its input norm grew.

    Slot states are assembled in the canonical slot order ``[u, v, C, L, R]``, so
    swapping the endpoints exchanges the left and right reads with the identity
    index map and leaves the closure states fixed. Together with the three shared
    typed heads that is exactly the `SWAP_PERM` equivariance of spec section 4.

    ``gate_mode`` carries two of the section 8 controls: ``per_type`` emits one
    common weight per pair and edge type, and ``mean_graph`` ignores the
    endpoints entirely and returns the installed training-mean adjacency -- in
    that mode no generator parameter is on the autograd path, which a trainer
    must account for before wrapping the model in DDP.

    ``slot_read`` selects the read. ``bare`` is the wave-1/wave-2 behaviour.
    ``residual_block`` reads through `_ResidualReadBlock` and, because the query
    is then a term of the result rather than only a selector, initialises the
    slot queries at ``std = 1.0`` instead of ``0.02``: the residual has to be
    comparable in norm to the attention output it is added to, or the block
    reduces to the bare read again. At ``std = 1.0`` the residual (norm ~10)
    instead dominates an attention output of order one, so the slot state is
    mostly a learned constant; ``slot_query_init_std`` overrides the rule with an
    explicit scale, and ``slot_read_value_norm`` decides whether the residual
    block's values keep their per-residue magnitude.

    ``head_output_init_std`` scales the last linear of each gate head. At the
    wave-1 value of ``1e-3`` the head's bias -- whose gradient does not pass
    through the slot states -- absorbs the target mean in the first steps while
    every upstream gradient is attenuated by the tiny output weight, so the
    queries and the attention barely train.
    """

    incidence: torch.Tensor
    fixed_weights: torch.Tensor

    def __init__(self, d_model: int, cfg: MotifPromptConfig) -> None:
        """Build the attention, the gate MPNN and the three typed heads.

        Args:
            d_model: Frozen trunk width the residue states arrive in.
            cfg: The motif-prompt block.
        """
        super().__init__()
        self.cfg = cfg
        self.residue_proj = nn.Linear(d_model, _GATE_DIM)
        self.attention = nn.MultiheadAttention(_GATE_DIM, 4, batch_first=True)
        residual = cfg.slot_read == "residual_block"
        default_std = 1.0 if residual else 0.02
        query_std = (
            default_std if cfg.slot_query_init_std is None else float(cfg.slot_query_init_std)
        )
        self.bridge_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * query_std)
        self.witness_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * query_std)
        value_norm = cfg.slot_read_value_norm
        self.bridge_read = (
            _ResidualReadBlock(_GATE_DIM, value_norm=value_norm) if residual else None
        )
        self.witness_read = (
            _ResidualReadBlock(_GATE_DIM, value_norm=value_norm) if residual else None
        )
        self.endpoint_proj = nn.Linear(_GATE_DIM, _GATE_DIM)
        self.witness_mix = nn.Sequential(
            nn.Linear(2 * _GATE_DIM, _GATE_DIM), nn.GELU(), nn.Linear(_GATE_DIM, _GATE_DIM)
        )
        self.message_layers = nn.ModuleList(_MessageLayer(_GATE_DIM) for _ in range(2))
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * _GATE_DIM),
                nn.Linear(2 * _GATE_DIM, _GATE_DIM),
                nn.GELU(),
                nn.Linear(_GATE_DIM, 1),
            )
            for _ in range(N_EDGE_TYPES)
        )
        for head in self.heads:
            output = cast(nn.Linear, cast(nn.Sequential, head)[-1])
            nn.init.normal_(output.weight, std=float(cfg.head_output_init_std))
            nn.init.zeros_(output.bias)
        self.register_buffer("fixed_weights", torch.zeros(N_EDGES))
        self.register_buffer("incidence", _candidate_incidence())
        #: What `init_biases` did, per edge type, for ``profile.json``.
        self.bias_init_record: dict[str, dict[str, float | str]] = {}

    def _closure_init_weight(
        self, stats: MotifTemplateStatistics, *, closure_bias_init: str
    ) -> tuple[str, float]:
        """Return the rule name and the closure magnitude the bias is set from.

        ``density`` is the wave-1 rule: the per-edge mean of the closure block.
        58% of training rows carry no closure edge, so that mean is a density and
        ``logit`` of it starts the gates at ``z = -3.97``, where the sigmoid
        derivative is 0.015 and they stay for the whole run.

        ``nonzero_mean`` uses the mean weight of a closure edge that *exists*
        instead, then mass-checks it: eight wedges at magnitude ``w`` carry
        ``8 w^2``, and if that exceeds twice the mean wedge mass of a row that has
        one, the magnitude is lowered to ``sqrt(m_C^+ / 8)`` so the graph does not
        start far denser than the corpus it is fitting. Both land in the
        unsaturated regime (spec section 4, wave-2 fix F1).

        Args:
            stats: The training-corpus statistics.
            closure_bias_init: ``density`` or ``nonzero_mean``.

        Returns:
            ``(rule, weight)`` before the ``[0.01, 0.99]`` clip.
        """
        if closure_bias_init == "density":
            return "density", float(self.fixed_weights[_TYPE_MASKS[TYPE_CLOSURE]].mean())
        weight = float(stats.nonzero_mean_by_type[TYPE_CLOSURE])
        mass = float(stats.positive_row_wedge_mass)
        # The mass check needs a positive-row mass to check against: with none the
        # fallback would be sqrt(0) = 0, clipped to 0.01 -- a deeper saturation than
        # the density rule this initialisation replaces.
        if mass > 0.0 and 8.0 * weight**2 > 2.0 * mass:
            return "nonzero_mean_mass_checked", math.sqrt(mass / 8.0)
        return "nonzero_mean", weight

    @torch.no_grad()
    def init_biases(self, stats: MotifTemplateStatistics, *, closure_bias_init: str) -> None:
        """Publish the training mean and set each head's output bias from it.

        The attachment and interior heads keep the wave-1 rule: the logit of the
        edge type's mean training weight. The non-zero interior mean is 1, which
        the clip would turn into ``z = 4.6``, a new saturation on the other side,
        and attachment already starts unsaturated at ``z ~ -1.2``. The closure
        head follows ``closure_bias_init`` (`_closure_init_weight`).

        Every magnitude is clipped to ``[0.01, 0.99]`` so no trainable sigmoid
        head starts in a saturated region (spec section 4); the clip belongs to
        the bias alone. The ``mean_graph`` control has no sigmoid to saturate and
        must return the adjacency that was actually installed -- clipping it would
        turn zero and rare edges positive, move both motif masses, and make the
        control disagree with the ``mean`` intervention over the same graph.

        Args:
            stats: The training-corpus statistics of the compiled templates.
            closure_bias_init: ``density`` or ``nonzero_mean``.

        Raises:
            ValueError: On a shape mismatch or an unknown rule.
        """
        if tuple(stats.mean.shape) != (N_EDGES,):
            raise ValueError(f"mean weights must be a ({N_EDGES},) vector")
        if closure_bias_init not in CLOSURE_BIAS_INITS:
            raise ValueError(f"closure_bias_init must be one of {list(CLOSURE_BIAS_INITS)}")
        low, high = _BIAS_CLIP
        self.fixed_weights.copy_(torch.as_tensor(stats.mean, dtype=torch.float32))
        record: dict[str, dict[str, float | str]] = {}
        for edge_type, head in enumerate(self.heads):
            if edge_type == TYPE_CLOSURE:
                rule, raw = self._closure_init_weight(stats, closure_bias_init=closure_bias_init)
            else:
                rule = "density"
                raw = float(self.fixed_weights[_TYPE_MASKS[edge_type]].mean())
            weight = min(max(raw, low), high)
            bias = float(torch.logit(torch.tensor(weight, dtype=torch.float32)))
            output = cast(nn.Linear, cast(nn.Sequential, head)[-1])
            output.bias.fill_(bias)
            record[EDGE_TYPE_NAMES[edge_type]] = {
                "rule": rule,
                "weight": weight,
                "bias_logit": bias,
            }
        self.bias_init_record = record

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """This generator's trainable parameters, grouped for the gradient probe.

        Every trainable parameter lands in exactly one group, the remainder in
        ``other``, so a group's norm is never silently dropped when the module
        tree changes (spec section 7.5 telemetry, wave-2 fix F3).

        Returns:
            One list per `GENERATOR_GRAD_GROUPS` entry; a group may be empty.
        """
        named: dict[str, list[nn.Parameter]] = {name: [] for name in GENERATOR_GRAD_GROUPS}
        head_groups = {
            TYPE_CLOSURE: "head_closure",
            TYPE_ATTACH: "head_attach",
            TYPE_INTERIOR: "head_interior",
        }
        heads = {f"heads.{index}": head_groups[index] for index in range(N_EDGE_TYPES)}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            group = "other"
            if name in ("bridge_queries", "witness_queries"):
                group = "slot_queries"
            elif name.startswith(
                (
                    "residue_proj.",
                    "attention.",
                    "endpoint_proj.",
                    "witness_mix.",
                    "bridge_read.",
                    "witness_read.",
                )
            ):
                group = "attention"
            elif name.startswith("message_layers."):
                group = "mpnn"
            else:
                for prefix, head_group in heads.items():
                    if name.startswith(f"{prefix}."):
                        group = head_group
                        break
            named[group].append(param)
        return named

    @staticmethod
    def gate_logit_statistics(weights: torch.Tensor) -> dict[str, float]:
        """Pre-activation statistics per edge type, inverted from the emitted weights.

        The heads emit ``sigmoid(z)``, so ``logit`` recovers ``z`` exactly; the
        clamp only guards the fp32 endpoints. ``frac_abs_gt_3`` is the saturated
        fraction that made wave 1's closure family immovable.

        Args:
            weights: ``(B, 96)`` emitted weights.

        Returns:
            ``gate_logit_{type}_{mean,frac_abs_gt_3}`` for the three edge types.
        """
        logits = torch.logit(weights.detach().float().clamp(1e-6, 1.0 - 1e-6))
        out: dict[str, float] = {}
        for edge_type, name in enumerate(EDGE_TYPE_NAMES):
            block = logits[:, _TYPE_MASKS[edge_type]]
            out[f"gate_logit_{name}_mean"] = float(block.mean())
            out[f"gate_logit_{name}_frac_abs_gt_3"] = float((block.abs() > 3.0).float().mean())
        return out

    def _read(
        self,
        queries: torch.Tensor,
        states: torch.Tensor,
        pad: torch.Tensor | None,
        block: _ResidualReadBlock | None,
    ) -> torch.Tensor:
        """Let one shared query block attend over one endpoint's residue states.

        Args:
            queries: ``(K, 96)`` shared slot queries.
            states: ``(B, L, 96)`` projected residue states of one endpoint.
            pad: ``(B, L)`` key padding mask.
            block: The residual read block, or ``None`` for the bare read.

        Returns:
            ``(B, K, 96)`` slot states.
        """
        expanded = queries.unsqueeze(0).expand(states.size(0), -1, -1)
        if block is not None:
            return cast(torch.Tensor, block(expanded, states, pad, self.attention))
        out, _ = self.attention(expanded, states, states, key_padding_mask=pad, need_weights=False)
        return cast(torch.Tensor, out)

    def _slot_states(
        self,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ``(B, 26, 96)`` slot states in canonical slot order."""
        pad_u = _build_padding_mask(lengths_u, encoded_u.size(1))
        pad_v = _build_padding_mask(lengths_v, encoded_v.size(1))
        state_u = self.residue_proj(encoded_u)
        state_v = self.residue_proj(encoded_v)
        left = self._read(self.bridge_queries, state_u, pad_u, self.bridge_read)
        right = self._read(self.bridge_queries, state_v, pad_v, self.bridge_read)
        witness_u = self._read(self.witness_queries, state_u, pad_u, self.witness_read)
        witness_v = self._read(self.witness_queries, state_v, pad_v, self.witness_read)
        closure = self.witness_mix(
            torch.cat([witness_u + witness_v, (witness_u - witness_v).abs()], dim=-1)
        )
        pooled_u = self.endpoint_proj(
            masked_mean(state_u, inner_token_mask(x=state_u, padding_mask=pad_u))
        )
        pooled_v = self.endpoint_proj(
            masked_mean(state_v, inner_token_mask(x=state_v, padding_mask=pad_v))
        )
        # `SLOT_U`, `SLOT_V`, `C_SLOTS`, `L_SLOTS` and `R_SLOTS` are contiguous
        # and in this order (`src.data.motif_template`), so the assembly is one
        # concatenation and no in-place scatter into a zero tensor is needed.
        return torch.cat(
            [pooled_u.unsqueeze(1), pooled_v.unsqueeze(1), closure, left, right], dim=1
        )

    def forward(
        self,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the 96 edge weights of one batch of pairs.

        Args:
            encoded_u: Frozen residue states of ``u`` ``(B, L_u, d_model)``.
            encoded_v: Frozen residue states of ``v`` ``(B, L_v, d_model)``.
            lengths_u: True residue lengths of ``u``.
            lengths_v: True residue lengths of ``v``.

        Returns:
            ``(B, 96)`` weights in ``[0, 1]``, always in float32.
        """
        batch = encoded_u.size(0)
        if self.cfg.gate_mode == "mean_graph":
            return self.fixed_weights.unsqueeze(0).expand(batch, -1)
        h = self._slot_states(encoded_u, encoded_v, lengths_u, lengths_v)
        for layer in self.message_layers:
            h = layer(h, self.incidence)

        rows = _EDGE_ROWS.to(h.device).reshape(-1)
        cols = _EDGE_COLS.to(h.device).reshape(-1)
        # The gate heads run in fp32 with autocast disabled: the weights they
        # emit are the inputs of the fp32 count and RRWP arithmetic, and a
        # bf16 sigmoid would quantise them before that arithmetic ever sees
        # them (the `assemble.py` trap, spec section 5). Under autocast the
        # slot states arrive in bf16 and are promoted *before* the sum and the
        # difference are formed: the wave-1/wave-2 order promoted afterwards,
        # so a slot difference below the bf16 ulp of the sum was rounded away
        # and the fp32 head saw eight identical rows. It changes the numerics
        # of an older checkpoint only by that rounding.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h_i = h[:, rows].float()
            h_j = h[:, cols].float()
            features = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
            logits = features.new_zeros(batch, N_EDGES)
            for edge_type, head in enumerate(self.heads):
                mask = _TYPE_MASKS[edge_type].to(h.device)
                block = features[:, mask]
                if self.cfg.gate_mode == "per_type":
                    logits[:, mask] = head(block.mean(dim=1)).expand(-1, int(block.size(1)))
                else:
                    logits[:, mask] = head(block).squeeze(-1)
            return torch.sigmoid(logits)


def pool_residues(states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool one endpoint's inner residues under its true length.

    The encoder's padding mask keeps padded positions out of attention but does
    not zero the states it writes at them, so an unmasked mean makes the same
    pair's prompt depend on the padding length of whatever batch it was collated
    with -- and packed scoring gathers different padding than training does. The
    inner-token convention (BOS/EOS excluded) is the one the generator and
    `V3_1Prefix` already pool under.

    Args:
        states: ``(B, L, d_model)`` encoder output.
        lengths: ``(B,)`` true sequence lengths.

    Returns:
        ``(B, d_model)`` pooled states.
    """
    padding = _build_padding_mask(lengths, states.size(1))
    return masked_mean(states, inner_token_mask(x=states, padding_mask=padding))


class MotifDirectTokens(nn.Module):
    """The section 8 direct-prefix control's three topology tokens, read off the residues.

    It answers "does a graph bottleneck help beyond a conditional adapter?", so it
    replaces the reader and never sees a motif graph -- but it has to keep the
    interface's symmetries or it is not comparing like with like. The endpoint map
    is shared and reads its own endpoint beside the swap-invariant context
    ``[p_u + p_v, |p_u - p_v|]``, so swapping the endpoints exchanges ``topo_u``
    and ``topo_v`` exactly as `MotifGritReader`'s shared ``node_proj`` does, and
    the relation token is a function of the context alone and is therefore
    swap-invariant like ``topo_rel``. The AB/BA aggregation of spec section 6
    cannot repair an order-dependent token, because both orientations reuse the
    tokens computed from the original ordering.
    """

    def __init__(self, d_model: int, width: int) -> None:
        """Build the shared endpoint map and the relation map.

        Args:
            d_model: Frozen trunk width the pooled residue states arrive in.
            width: Token width.
        """
        super().__init__()
        self.node = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.relation = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, pooled_u: torch.Tensor, pooled_v: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the three tokens of one batch of pairs.

        Args:
            pooled_u: ``(B, d_model)`` masked pooled residues of ``u``.
            pooled_v: ``(B, d_model)`` masked pooled residues of ``v``.

        Returns:
            ``topo_u``, ``topo_v`` and the swap-invariant ``topo_rel``.
        """
        context = torch.cat([pooled_u + pooled_v, (pooled_u - pooled_v).abs()], dim=-1)
        return {
            "topo_u": self.node(torch.cat([pooled_u, context], dim=-1)),
            "topo_v": self.node(torch.cat([pooled_v, context], dim=-1)),
            "topo_rel": self.relation(context),
        }


_FIELD_TO_TOKEN = {
    "topo_self": ("topo_u", "topo_v"),
    "topo_partner": ("topo_v", "topo_u"),
    "topo_rel": ("topo_rel", "topo_rel"),
    "topo_cnt": ("topo_cnt", "topo_cnt"),
}


class MotifPromptAdapter(nn.Module):
    """The four-token gated KV prefix at all nine cross-attention sites (spec section 6).

    Fields are laid out in the canonical `FIELD_ORDER`; ``cfg.fields`` selects
    which of them survive, and an unselected field's rows are *removed* from the
    prefix so the separately-softmaxed branch renormalises over the remainder.
    Zeroing a field's values would leave its keys in the denominator, which is
    not the same model -- the count-only and GRIT-only arms of section 8 are
    therefore separately trained, not inference ablations.

    Gates are shaped ``(n_layers, SITES, n_heads)``: per head, never per field,
    because one softmax covers all prefix rows jointly. They start at exactly
    zero, so the composed model reproduces the frozen base bit for bit.

    The role embedding marks the prefix *slot* -- self, partner, relation, count
    -- and not the endpoint identity. Prefix rows carry no positional signal
    inside the branch softmax, so a role tied to ``u``/``v`` would leave the two
    stream views holding the same set of rows and erase the self/partner
    distinction that the AB/BA symmetrisation exists to carry.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, cfg: MotifPromptConfig) -> None:
        """Build the role embeddings, per-layer expansions, offsets and gates.

        Args:
            d_model: Trunk width.
            n_layers: Number of cross-attention layers.
            n_heads: Heads per attention site.
            cfg: The motif-prompt block.
        """
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_layers = n_layers
        self.slots = len(FIELD_ORDER) * cfg.slots_per_field
        self.active_rows: tuple[int, ...] = tuple(
            field_index * cfg.slots_per_field + offset
            for field_index, name in enumerate(FIELD_ORDER)
            if name in cfg.fields
            for offset in range(cfg.slots_per_field)
        )
        self.role_embed = nn.Parameter(torch.randn(len(FIELD_ORDER), cfg.width) * 0.02)
        self.token_norm = nn.LayerNorm(cfg.width)
        self.layer_proj = nn.ModuleList(
            nn.Linear(cfg.width, cfg.slots_per_field * d_model) for _ in range(n_layers)
        )
        self.p0 = nn.ParameterList(
            nn.Parameter(torch.randn(self.slots, d_model) * 0.02) for _ in range(n_layers)
        )
        self.gates = nn.Parameter(torch.zeros(n_layers, SITES, n_heads))

    def views(self, tokens: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(B, 4, width)`` field stacks of the two stream views.

        Args:
            tokens: ``topo_u``, ``topo_v``, ``topo_rel`` and ``topo_cnt``.

        Returns:
            ``(view_u, view_v)``; ``view_u`` names ``u`` as *self*. Exchanging
            ``topo_u`` and ``topo_v`` exchanges the two views, and ``topo_rel``
            and ``topo_cnt`` are identical in both because both are
            swap-invariant.
        """
        stacks: list[list[torch.Tensor]] = [[], []]
        for index, name in enumerate(FIELD_ORDER):
            first, second = _FIELD_TO_TOKEN[name]
            stacks[0].append(tokens[first] + self.role_embed[index])
            stacks[1].append(tokens[second] + self.role_embed[index])
        return (
            self.token_norm(torch.stack(stacks[0], dim=1)),
            self.token_norm(torch.stack(stacks[1], dim=1)),
        )

    def prefix(self, layer_index: int, view: torch.Tensor) -> torch.Tensor:
        """Expand a field stack to this layer's active prefix rows.

        Args:
            layer_index: Which frozen layer.
            view: ``(B, 4, width)`` field stack.

        Returns:
            ``(B, len(active_rows), d_model)`` prefix rows.
        """
        proj = cast(nn.Linear, self.layer_proj[layer_index])
        rows: torch.Tensor = proj(view).view(view.size(0), self.slots, self.d_model)
        p0: torch.Tensor = self.p0[layer_index]
        return (rows + p0.unsqueeze(0))[:, list(self.active_rows)]

    def gate(self, layer_index: int, site: int) -> torch.Tensor:
        """Return the raw (pre-tanh) per-head gate for one site.

        Args:
            layer_index: Which frozen layer.
            site: ``0`` for ``A<-B``, ``1`` for ``B<-A``, ``2`` for the CLS site.

        Returns:
            The ``(n_heads,)`` raw gate.
        """
        return self.gates[layer_index, site]


class MotifPromptCrossAttentionLayer(nn.Module):
    """Re-drive one frozen `CrossAttentionLayer` with the role-aware motif prefix.

    Site ``A<-B`` and the CLS site read the view whose *self* endpoint is the A
    stream's node; site ``B<-A`` reads the other view. The wrapped layer and the
    adapter are plain references, exactly as the prefix and topology-prompt arms
    do, so their state-dict keys are not duplicated.
    """

    layer: CrossAttentionLayer
    adapter: MotifPromptAdapter

    def __init__(
        self, layer: CrossAttentionLayer, layer_index: int, adapter: MotifPromptAdapter
    ) -> None:
        """Wrap a frozen layer.

        Args:
            layer: The trunk's `CrossAttentionLayer` (registered under ``base``).
            layer_index: Its index in the trunk.
            adapter: The shared prompt parameters (registered under ``adapter``).
        """
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "adapter", adapter)
        self.layer_index = layer_index

    def _attend(
        self,
        site: int,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prefix: torch.Tensor,
        gate_scale: float,
    ) -> torch.Tensor:
        """Run one site's frozen attention and add the gated prefix branch."""
        layer = self.layer
        query_norm = layer.norm_attn(query)
        attn_out, _ = layer.attn(
            query_norm, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn, query_norm, prefix, self.adapter.gate(self.layer_index, site), gate_scale
        )
        return query + cast(torch.Tensor, layer.drop_attn(attn_out)) + branch

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
        prefix_a: torch.Tensor,
        prefix_b: torch.Tensor,
        gate_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One bidirectional block plus the gated prefix branch at all three sites.

        Args:
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            cls_token: CLS state ``(B, 1, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.
            prefix_a: This layer's prefix in the A stream's view.
            prefix_b: This layer's prefix in the B stream's view.
            gate_scale: ``0.0`` realises the gates-off intervention.

        Returns:
            Updated ``(h_a, h_b, cls_token)``.
        """
        layer = self.layer
        h_a = self._attend(0, h_a, h_b, mask_b, prefix_a, gate_scale)
        h_a = layer._ffn(h_a)  # noqa: SLF001
        h_b = self._attend(1, h_b, h_a, mask_a, prefix_b, gate_scale)
        h_b = layer._ffn(h_b)  # noqa: SLF001

        combined = torch.cat([h_a, h_b], dim=1)
        combined_mask = (
            torch.cat([mask_a, mask_b], dim=1)
            if mask_a is not None and mask_b is not None
            else None
        )
        cls_norm = layer.norm_cls_attn(cls_token)
        attn_cls, _ = layer.attn_cls(
            cls_norm, combined, combined, key_padding_mask=combined_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn_cls, cls_norm, prefix_a, self.adapter.gate(self.layer_index, 2), gate_scale
        )
        cls_token = cls_token + layer.drop_cls_attn(attn_cls) + branch
        cls_token = cls_token + layer.drop_cls_ffn(layer.ff_cls(layer.norm_cls_ffn(cls_token)))
        return h_a, h_b, cls_token


class V3_1MotifPrompt(nn.Module):
    """A frozen bidirectional-cross `V3_1` reading a motif graph through gated prefixes.

    ``reader`` and ``adapter`` are registered before ``base``, so the structural
    stream's first-trainable-parameter anchor lands on the motif path rather than
    on the frozen trunk. Stage I reads a compiled template from
    ``batch['motif_weights']`` and fails closed without one; Stage II predicts the
    template from the endpoints and never reads a truth graph at inference, where
    ``batch['motif_weights']`` is a supervision *target* only.

    With every gate at zero, and under ``intervention='gates_off'``, the frozen
    trunk and head reproduce the published base bit for bit: `prefix_branch`
    multiplies by ``tanh(0)`` before the output projection, so the added branch is
    an exact zero tensor.
    """

    name: str = "v3_1_motif_prompt"
    mean_template: torch.Tensor
    family_mask: torch.Tensor
    w_slot_resolved: torch.Tensor

    def __init__(self, *, base: Mapping[str, object], motif_prompt: Mapping[str, object]) -> None:
        """Build the frozen base and the motif path on top.

        Args:
            base: The base `V3_1` constructor kwargs (a checkpoint's ``model_config``).
            motif_prompt: The ``model.config.motif_prompt`` block.

        Raises:
            ValueError: If the base trunk has no bidirectional cross-attention layers.
        """
        super().__init__()
        self.cfg = MotifPromptConfig.from_mapping(motif_prompt)
        self.base_config: dict[str, object] = dict(base)
        base_model = V3_1(**self.base_config)
        trunk = base_model.cross_attention
        if trunk.mixing_mode != "bidirectional_cross" or len(trunk.layers) == 0:
            raise ValueError(
                "v3_1_motif_prompt needs a base with model.config.mixing.mode == "
                "'bidirectional_cross' and at least one cross-attention layer"
            )
        self.d_model = int(base_model.d_model)
        self.input_dim = int(base_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.reader = MotifGritReader(self.cfg.reader, self.cfg.width)
        self.count_head = MotifCountHead(
            self.cfg.width, degree_only=self.cfg.count_features == "degree"
        )
        self.direct_head = (
            MotifDirectTokens(self.d_model, self.cfg.width)
            if self.cfg.token_source == "direct"
            else None
        )
        self.adapter = MotifPromptAdapter(
            self.d_model, len(trunk.layers), int(base_model.n_heads), self.cfg
        )
        self.generator = MotifGenerator(self.d_model, self.cfg) if self.cfg.stage == "two" else None
        self.base = base_model
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base.eval()
        self.prompt_layers = nn.ModuleList(
            MotifPromptCrossAttentionLayer(cast(CrossAttentionLayer, layer), index, self.adapter)
            for index, layer in enumerate(trunk.layers)
        )
        # Which student-side graph readers the prompt actually uses: a field
        # removed from the prefix by `MotifPromptAdapter` never reads its
        # module, so that module is frozen and skipped rather than run for a
        # token that is discarded (spec section 8's count-only / GRIT-only /
        # degree-only arms).
        self.student_reads_graph = self.cfg.token_source == "graph" and bool(
            _READER_FIELDS & set(self.cfg.fields)
        )
        self.student_counts = "topo_cnt" in self.cfg.fields
        self._freeze_permanently_frozen()
        if self.cfg.training_policy == "head_only":
            self.requires_grad_(False)
            self.base.output_head.requires_grad_(True)
        self.register_buffer("mean_template", torch.zeros(N_EDGES))
        # The balanced ``w_slot``, once measured; -1 means "not yet". It is a
        # buffer so it rides in ``model_state``: a resumed or scored run reads
        # the weight the run was actually trained under and never re-balances.
        self.register_buffer("w_slot_resolved", torch.full((), -1.0))
        # Derived from `cfg.families` on every construction, so a checkpoint can
        # never restore a stale family gate over a changed config.
        self.register_buffer("family_mask", _family_mask(self.cfg.families), persistent=False)
        self.teacher: nn.Module | None = None
        self.intervention: str = "none"
        self.corruption_seed = 42
        self.interface_open = self.interface_open_at(1)

    @property
    def encoder(self) -> nn.Module:
        """The frozen base's per-node encoder (packed scoring caches its output)."""
        return self.base.encoder

    @property
    def generator_trainable(self) -> bool:
        """Whether any generator parameter trains in this configuration."""
        return self.generator is not None and any(
            param.requires_grad for param in self.generator.parameters()
        )

    def interface_open_at(self, epoch: int) -> bool:
        """Whether the interface group trains in the 1-based ``epoch``.

        Stage I trains the interface throughout. Stage II holds it for the
        ``interface_warmup_epochs`` in which the generator trains alone (spec
        section 7.5); ``null`` never opens it (the freeze-forever control). A
        configuration whose generator never trains -- ``gate_mode='mean_graph'``
        -- has nothing to warm up: holding its only optimiser group would waste
        the epochs and open it on the one-cycle's annealing tail, so the
        interface trains from epoch 1 there, on the full cycle.

        Args:
            epoch: The 1-based epoch about to run.

        Returns:
            ``True`` when the interface group's learning rate follows the schedule.
        """
        if self.cfg.stage == "one" or not self.generator_trainable:
            return True
        warmup = self.cfg.interface_warmup_epochs
        return warmup is not None and epoch > warmup

    @property
    def interface_lr_scale(self) -> float:
        """The interface group's peak LR as a fraction of the generator's.

        Stage I trains the interface at the run's LR; Stage II opens it at 0.1x
        the generator's instantaneous LR (spec sections 7.2 and 7.5).
        """
        return 1.0 if self.cfg.stage == "one" else STAGE_TWO_INTERFACE_LR_SCALE

    @property
    def graph_only_warmup(self) -> bool:
        """Whether this step's task, structural and topo losses are cut off from G.

        Wave-2 fix F3: with ``warmup_losses='graph_only'`` the warm-up epochs are
        the spec 0.2 pilot B run as the first two epochs. The predicted weights
        the trunk and the immutable teacher read are detached there, so every
        loss is still computed and logged at its unchanged value while ``L_slot``
        is the only term that reaches the generator. The task gradient into G
        outweighed the graph gradient 170-860x at wave-1 initialisation, which is
        what the warm start removes.

        Wave-3: ``graph_only_always`` never lifts the cut. The eight-epoch warm-up
        run showed the closure gates re-saturating within one epoch of the losses
        reaching G (the task gradient into the closure head was 1400x the graph
        gradient by then), so this variant opens the interface on the same
        schedule while leaving G supervised by ``L_G`` alone for the whole run.
        """
        if self.cfg.stage != "two":
            return False
        if self.cfg.warmup_losses == "graph_only_always":
            return True
        return self.cfg.warmup_losses == "graph_only" and not self.interface_open

    @property
    def w_slot_value(self) -> float:
        """The ``L_slot`` weight in force, balanced or configured.

        A balanced weight that has not been measured yet -- the warm-up epochs,
        which run before the first interface-open step -- is 1.0, the numeric
        default, so the warm start is ``L_G`` at its own scale. Under
        ``warmup_losses='graph_only_always'`` it is never measured (the balance is
        a task/graph gradient ratio and no task gradient reaches G), so ``L_G``
        keeps that scale for the whole run.
        """
        resolved = float(self.w_slot_resolved)
        if resolved >= 0.0:
            return resolved
        return 1.0 if self.cfg.w_slot_is_balanced else float(cast(float, self.cfg.w_slot))

    @torch.no_grad()
    def resolve_w_slot(self, value: float) -> None:
        """Fix the balanced ``L_slot`` weight for the rest of the run.

        Args:
            value: The measured weight.

        Raises:
            ValueError: On a negative weight, or on a model whose ``w_slot`` is
                a number and therefore never balanced.
        """
        if not self.cfg.w_slot_is_balanced:
            raise ValueError("motif_prompt.w_slot is a number; it is not balanced at run time")
        if not value >= 0.0:
            raise ValueError(f"the balanced w_slot must be non-negative, got {value}")
        self.w_slot_resolved.fill_(float(value))

    def train(self, mode: bool = True) -> V3_1MotifPrompt:
        """Switch the wrapper's mode while the frozen base and teacher stay in eval.

        Args:
            mode: Training mode for the motif path.

        Returns:
            ``self``.
        """
        super().train(mode)
        self.base.eval()
        if self.teacher is not None:
            self.teacher.eval()
        if self.cfg.training_policy == "head_only":
            # Calling ``train()`` on the wrapper must not reactivate dropout in
            # any frozen feature producer.  The output head is the sole module
            # whose mode and parameters follow the optimisation phase.
            self.reader.eval()
            self.count_head.eval()
            self.adapter.eval()
            self.prompt_layers.eval()
            if self.direct_head is not None:
                self.direct_head.eval()
            if self.generator is not None:
                self.generator.eval()
            self.base.output_head.train(mode)
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates."""
        return [param for param in self.parameters() if param.requires_grad]

    def _freeze_permanently_frozen(self) -> None:
        """Drop every parameter this configuration can never train.

        Several section 8 controls bypass a whole module: ``token_source='direct'``
        never calls the student reader (the immutable teacher reads the graph
        through its own frozen copy), a field list without the three GRIT fields
        never reads the student reader either, one without ``topo_cnt`` never
        reads the count head, and ``gate_mode='mean_graph'`` returns the
        installed mean adjacency without touching a generator parameter. Stage II
        additionally freezes the reader's role/input embeddings and every block
        but the last for the whole run (spec section 7.5). Production DDP wraps
        this model with ``find_unused_parameters=False``, so a trainable parameter
        that never receives a gradient aborts the second iteration: the freeze
        happens here, at construction, before any wrapping.
        """
        if not self.student_reads_graph:
            self.reader.requires_grad_(False)
        elif self.cfg.stage == "two":
            for module in self.reader.frozen_stage_two_modules():
                module.requires_grad_(False)
        if not self.student_counts:
            self.count_head.requires_grad_(False)
        if self.cfg.gate_mode == "mean_graph" and self.generator is not None:
            self.generator.requires_grad_(False)

    def _interface_modules(self) -> list[nn.Module]:
        """The warm-up-controlled bundle: reader interface, count head, prefix adapter.

        Stage I trains the whole reader (spec section 7.2). Stage II may adapt
        only its final block and the output projections once the warm-up opens
        the group (spec section 7.5), so the earlier blocks and the role/input
        embeddings -- frozen outright in `_freeze_permanently_frozen` -- are not
        members of it and opening the group cannot reach them.

        The direct-prefix control's token head is deliberately *not* a member:
        it is the student's token producer, the counterpart of the generator
        rather than of the Stage I-trained reader it replaces, and no bundle
        carries a trained one. Holding it at its random initialisation behind
        gates trained open for two epochs, then opening it at a tenth of the
        generator's rate, would handicap the control against the main arm; it
        trains in the generator group instead.
        """
        reader_modules: list[nn.Module] = (
            [self.reader] if self.cfg.stage == "one" else self.reader.adaptable_modules()
        )
        return [*reader_modules, self.count_head, self.adapter]

    def optimizer_parameter_groups(
        self, generator_lr: float, interface_lr: float, weight_decay: float
    ) -> list[dict[str, object]]:
        """Two named groups: ``generator`` and the 0.1x ``interface`` (spec section 7.5).

        Every parameter that is ever trainable keeps ``requires_grad`` from
        construction, so DDP registers and all-reduces it. The epochs 1-2 warm-up
        is realised by zeroing the ``interface`` group's LR, which is an exact
        freeze under AdamW's decoupled weight decay.

        Args:
            generator_lr: Peak LR of the generator group.
            interface_lr: Peak LR of the reader/count-head/adapter group.
            weight_decay: Decoupled weight decay for both groups.

        Returns:
            The non-empty groups, generator first.
        """
        interface = [
            param
            for module in self._interface_modules()
            for param in module.parameters()
            if param.requires_grad
        ]
        interface_ids = {id(param) for param in interface}
        generator = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in interface_ids
        ]
        groups: list[dict[str, object]] = []
        for name, params, lr in (
            ("generator", generator, generator_lr),
            ("interface", interface, interface_lr),
        ):
            if params:
                groups.append(
                    {
                        "name": name,
                        "params": params,
                        "lr": lr,
                        "max_lr": lr,
                        "weight_decay": weight_decay,
                    }
                )
        return groups

    @torch.no_grad()
    def install_mean_template(self, stats: MotifTemplateStatistics) -> None:
        """Publish the training-corpus mean adjacency ``Abar`` (spec section 3).

        The buffer keeps its meaning -- the corpus mean, which the ``mean``
        intervention and the ``mean_graph`` control read -- and the rest of the
        statistics reach the generator's bias initialisation alone.

        Args:
            stats: Statistics of the randomised compiled training templates.

        Raises:
            ValueError: On a shape mismatch or a non-finite entry.
        """
        mean = torch.as_tensor(stats.mean, dtype=torch.float32)
        if tuple(mean.shape) != (N_EDGES,) or not torch.isfinite(mean).all():
            raise ValueError(f"mean template must be a finite ({N_EDGES},) vector")
        self.mean_template.copy_(mean.to(self.mean_template))
        if self.generator is not None:
            self.generator.init_biases(stats, closure_bias_init=self.cfg.closure_bias_init)

    def initialize_teacher(self) -> None:
        """Snapshot the loaded Stage I bundle as the immutable teacher ``R_T``.

        The bundle is the reader, the count head, the token projections carried
        inside them and the prefix adapter with its gates (spec section 7.4), so
        the student never starts from a different Stage I checkpoint than its
        teacher.

        The direct-token head is deliberately not a member. It is a student-side
        substitution for the *prompt*, no Stage I checkpoint carries a trained
        one, and a teacher routed through it would read the endpoints alone: both
        ``_supervision_rows`` calls would then return identical ``topo_u``,
        ``topo_v`` and ``topo_rel``, zeroing three of the four terms of
        ``L_topo`` whatever the predicted and true adjacency, while spec section
        8 advertises the ``token_source='direct'`` control as carrying the same
        ``L_topo`` as the main arm.
        """
        self.teacher = (
            nn.ModuleDict(
                {
                    "reader": deepcopy(self.reader),
                    "count_head": deepcopy(self.count_head),
                    "adapter": deepcopy(self.adapter),
                }
            )
            .requires_grad_(False)
            .eval()
        )

    def set_corruption_step(self, step: int, seed: int = 42) -> None:
        """Set reproducible Stage I corruption for this global training step.

        Args:
            step: Global optimiser step.
            seed: Run seed.
        """
        self.corruption_seed = seed + step * 104729

    def _corrupt(self, weights: torch.Tensor, nonself: torch.Tensor | None) -> torch.Tensor:
        """Mix half the nonself Stage I rows towards ``Abar`` (spec section 3).

        Self rows keep the explicit empty template the compiler gives them and are
        never corrupted: mixing one towards the training mean would hand it
        nonzero topology in training while evaluation still reads it empty.

        Args:
            weights: ``(B, 96)`` compiled templates.
            nonself: ``(B,)`` 1/0 nonself mask, or ``None`` when the batch carries
                none, in which case every row is treated as nonself.

        Returns:
            The possibly corrupted templates.
        """
        cfg = self.cfg.corruption
        if not self.training or self.cfg.stage != "one" or cfg.prob == 0.0:
            return weights
        rng = torch.Generator(device=weights.device).manual_seed(self.corruption_seed)
        selected = torch.rand((weights.size(0), 1), device=weights.device, generator=rng) < cfg.prob
        if nonself is not None:
            selected = selected & (nonself.reshape(-1, 1).to(weights) > 0.0)
        span = cfg.lambda_max - cfg.lambda_min
        lam = cfg.lambda_min + span * torch.rand(
            (weights.size(0), 1), device=weights.device, generator=rng
        )
        mixed = (1.0 - lam) * weights + lam * self.mean_template.to(weights).unsqueeze(0)
        return torch.where(selected, mixed, weights)

    def _tokens(
        self,
        bundle: nn.ModuleDict | None,
        weights: torch.Tensor,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Read one graph through the student path or through the immutable teacher.

        The teacher always reads every field from the graph: `initialize_teacher`
        never puts a direct head in the bundle, so `L_topo` stays sensitive to
        the adjacency even for the direct-prefix control. The student reads only
        the fields its prefix keeps; a masked field's token is an exact zero that
        the adapter removes before the branch softmax.
        """
        if bundle is not None:
            reader = cast(MotifGritReader, bundle["reader"])
            tokens: dict[str, torch.Tensor] = reader(weights)
            tokens["topo_cnt"] = cast(MotifCountHead, bundle["count_head"])(weights)
            return tokens
        if self.direct_head is not None:
            tokens = self.direct_head(
                pool_residues(encoded_u, lengths_u), pool_residues(encoded_v, lengths_v)
            )
        elif self.student_reads_graph:
            tokens = self.reader(weights)
        else:
            zero = weights.new_zeros(weights.size(0), self.cfg.width, dtype=torch.float32)
            tokens = {"topo_u": zero, "topo_v": zero, "topo_rel": zero}
        tokens["topo_cnt"] = (
            self.count_head(weights)
            if self.student_counts
            else weights.new_zeros(weights.size(0), self.cfg.width, dtype=torch.float32)
        )
        return tokens

    def tokens_from_weights(
        self,
        weights: torch.Tensor,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the four token fields for one batch of motif graphs.

        Args:
            weights: ``(B, 96)`` edge weights.
            encoded_u: Residue states of ``u`` (used only by ``token_source='direct'``).
            encoded_v: Residue states of ``v``.
            lengths_u: True residue lengths of ``u``, so the direct control pools
                under the mask rather than over the batch's padding.
            lengths_v: True residue lengths of ``v``.

        Returns:
            ``topo_u``, ``topo_v``, ``topo_rel`` and ``topo_cnt``.
        """
        return self._tokens(None, weights, encoded_u, encoded_v, lengths_u, lengths_v)

    def predict_weights(
        self,
        encoded_u: torch.Tensor,
        encoded_v: torch.Tensor,
        lengths_u: torch.Tensor,
        lengths_v: torch.Tensor,
    ) -> torch.Tensor:
        """Stage II: predict the 96 edge weights from the endpoints alone.

        Args:
            encoded_u: Frozen residue states of ``u``.
            encoded_v: Frozen residue states of ``v``.
            lengths_u: True residue lengths of ``u``.
            lengths_v: True residue lengths of ``v``.

        Returns:
            ``(B, 96)`` predicted weights, family-gated.

        Raises:
            RuntimeError: If called on a Stage I model, which has no generator.
        """
        if self.generator is None:
            raise RuntimeError("stage 'one' has no generator; supply batch['motif_weights']")
        return self._gate_families(self.generator(encoded_u, encoded_v, lengths_u, lengths_v))

    def _gate_families(self, weights: torch.Tensor) -> torch.Tensor:
        """Zero every edge of an inactive family, in both stages (spec section 8)."""
        if len(self.cfg.families) == len(FAMILIES):
            return weights
        return weights * self.family_mask.to(weights)

    def closure_nonempty_rows(self, target: torch.Tensor) -> torch.Tensor:
        """``(B,)`` float indicator of a family-gated non-empty closure target.

        The trainer sums these rows across ranks beside the per-stream row counts
        and re-weights the graph loss with them when
        ``motif_prompt.graph_row_weighting`` asks for it (spec section 7.5). The
        family gating is `_supervision_rows`' own, so a bridge-only arm reports no
        non-empty row and the weighting falls back to uniform.

        Args:
            target: ``(B, 96)`` compiled edge weights, as the batch carries them.

        Returns:
            ``(B,)`` float32, one where the row's closure family is non-empty.
        """
        gated = self._gate_families(target.to(dtype=torch.float32))
        return closure_nonempty(gated).to(dtype=torch.float32)

    def _apply_intervention(self, weights: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Return the (possibly substituted) graph and the gate scale.

        Fails closed rather than silently no-opping. The three whole-graph
        substitutions of spec section 8 are scorer-level: they re-pair rows across
        the *whole scored universe*, which only the scorer can see, and reach this
        class as an explicit ``weights`` argument to `logits_from_encoded`, exactly
        as `V3_1Prefix` handles ``shuffle``.

        Raises:
            ValueError: On an unknown intervention name or a scorer-level one.
        """
        if self.intervention not in INTERVENTIONS:
            raise ValueError(f"unknown motif_prompt intervention {self.intervention!r}")
        if self.intervention == "none":
            return weights, 1.0
        if self.intervention == "gates_off":
            return weights, 0.0
        if self.intervention == "mean":
            mean = self._gate_families(self.mean_template.to(weights))
            return mean.unsqueeze(0).expand_as(weights), 1.0
        raise ValueError(
            f"motif_prompt intervention {self.intervention!r} is a scoring-time substitution "
            "over the whole universe; the scorer passes the substituted graph as an explicit "
            "weights argument and leaves the model on 'none'"
        )

    def _trunk(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        prefixes_a: list[torch.Tensor],
        prefixes_b: list[torch.Tensor],
        gate_scale: float,
    ) -> torch.Tensor:
        """Drive the frozen trunk once, in the A-as-self orientation."""
        trunk = self.base.cross_attention
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))
        cls_token = trunk.cls_token.repeat(h_a.size(0), 1, 1)
        for index, layer in enumerate(self.prompt_layers):
            h_a, h_b, cls_token = layer(
                h_a,
                h_b,
                cls_token,
                mask_a,
                mask_b,
                prefixes_a[index],
                prefixes_b[index],
                gate_scale,
            )
        cls_vec = cls_token.squeeze(1)
        if trunk.pair_readout_mode == "pair_context_gated":
            return cast(torch.Tensor, trunk.pair_context_readout(h_a, h_b, cls_vec, mask_a, mask_b))
        base_repr = trunk._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)  # noqa: SLF001
        if trunk.pair_readout_mode == "grid_sketch_fusion":
            return cast(
                torch.Tensor, trunk.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b)
            )
        return base_repr

    def logits_from_encoded(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        weights: torch.Tensor,
        return_pair_repr: bool = False,
    ) -> torch.Tensor:
        """Compute logits from encoded token states and one batch of motif graphs.

        Shared by `forward` and packed scoring, which caches the frozen encoder's
        output across pairs, so the trunk drive exists once. A scorer's
        whole-universe graph substitution arrives here as ``weights``.

        Args:
            encoded_a: Frozen encoder output for item A ``(B, L_a, d_model)``.
            encoded_b: Frozen encoder output for item B ``(B, L_b, d_model)``.
            lengths_a: True sequence lengths for A.
            lengths_b: True sequence lengths for B.
            weights: ``(B, 96)`` motif graph of each pair, with A as endpoint ``u``.
            return_pair_repr: Return the representation before the output head.

        Returns:
            The pair logits, or the pair representation.

        Raises:
            ValueError: If an intervention is set while `self.training`
                (interventions are scoring-time only), or via
                `_apply_intervention` on an unknown or scorer-level name.
        """
        if self.intervention != "none" and self.training:
            raise ValueError("motif_prompt interventions are scoring-time only; call eval() first")
        weights, gate_scale = self._apply_intervention(weights)
        tokens = self.tokens_from_weights(weights, encoded_a, encoded_b, lengths_a, lengths_b)
        return self.logits_from_tokens(
            encoded_a,
            encoded_b,
            lengths_a,
            lengths_b,
            tokens=tokens,
            gate_scale=gate_scale,
            return_pair_repr=return_pair_repr,
        )

    def logits_from_tokens(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        tokens: Mapping[str, torch.Tensor],
        gate_scale: float = 1.0,
        return_pair_repr: bool = False,
    ) -> torch.Tensor:
        """Compute pair logits from the existing four prompt-token fields.

        This is the graph-independent prompt interface shared by the legacy
        motif model and dictionary mixtures.  Factoring it here leaves the
        graph-to-token path and its checkpoint keys unchanged.
        """
        if self.cfg.training_policy == "head_only" and not self.cfg.head_prompt_enabled:
            gate_scale = 0.0
        view_a, view_b = self.adapter.views(tokens)
        prefixes_a = [self.adapter.prefix(i, view_a) for i in range(len(self.prompt_layers))]
        prefixes_b = [self.adapter.prefix(i, view_b) for i in range(len(self.prompt_layers))]
        feature_ab = self._trunk(
            encoded_a, encoded_b, lengths_a, lengths_b, prefixes_a, prefixes_b, gate_scale
        )
        if self.base.order_aggregation == "single":
            pair_repr = feature_ab
        else:
            feature_ba = self._trunk(
                encoded_b, encoded_a, lengths_b, lengths_a, prefixes_b, prefixes_a, gate_scale
            )
            pair_repr = torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values
        if return_pair_repr:
            return pair_repr
        return cast(torch.Tensor, self.base.output_head(pair_repr))

    def resolve_weights(
        self,
        merged: Mapping[str, torch.Tensor],
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return the graph the trunk reads for this batch.

        Stage I reads the compiled template and corrupts it; Stage II predicts it
        from the endpoints and treats any attached template as a target only.

        Args:
            merged: The merged batch.
            encoded_a: Frozen encoder output for item A.
            encoded_b: Frozen encoder output for item B.
            lengths_a: True sequence lengths for A.
            lengths_b: True sequence lengths for B.

        Returns:
            ``(B, 96)`` edge weights.

        Raises:
            ValueError: If Stage I is called without ``batch['motif_weights']``.
        """
        if self.cfg.stage == "two":
            return self.predict_weights(encoded_a, encoded_b, lengths_a, lengths_b)
        target = merged.get(TEMPLATE_KEY)
        if target is None:
            raise ValueError(
                "v3_1_motif_prompt stage 'one' requires batch['motif_weights']; this stage "
                "never scores a pair without its compiled template"
            )
        # The compiled weights stay fp32 whatever precision the trunk runs in:
        # they feed the fp32 count and RRWP arithmetic, and casting them to the
        # encoder's bf16 first would quantise them on the way (spec section 5).
        # The corruption mixes towards the ungated corpus mean, so the family
        # gate is applied last and holds in both stages (spec section 8).
        nonself = merged.get(TEMPLATE_MASK_KEY)
        template = target.to(device=encoded_a.device, dtype=torch.float32)
        return self._gate_families(self._corrupt(template, nonself))

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score pairs and, when targets ride along, return the composite loss.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            ``logits`` always; ``predicted_weights`` in Stage II; in Stage II with
            ``motif_weights`` also ``slot_loss_rows`` and, once a teacher is
            snapshotted, ``topo_loss_rows`` -- undetached, unscaled and self-row
            masked; with ``label`` the weighted task-BCE ``loss``, its
            ``loss_weight_sum``, the detached ``task_loss`` and the undetached
            ``loss_term_*`` shares of `LOSS_TERM_NAMES`. The trainer folds the two
            graph terms in, because only it sees both streams (spec section 7.5).

        Raises:
            ValueError: If Stage I is called without ``batch['motif_weights']``,
                or an intervention is set while training.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)
        emb_a, emb_b, lengths_a, lengths_b = unpack_pair_batch(merged, self.input_dim)
        with torch.no_grad():
            encoded_a = self.base.encoder(emb_a, lengths_a)
            encoded_b = self.base.encoder(emb_b, lengths_b)
        with self._eval_pair_precision(encoded_a.device):
            if not self.training:
                encoded_a, encoded_b = encoded_a.float(), encoded_b.float()
            weights = self.resolve_weights(merged, encoded_a, encoded_b, lengths_a, lengths_b)
            # The single point the graph-only warm-up acts at: everything that
            # reads the graph -- the token/trunk path of both streams and the
            # immutable teacher -- takes the detached weights, ``L_slot`` takes
            # the live ones. The values are identical, so every logged loss is
            # unchanged and only G's gradient path differs.
            read_weights = weights.detach() if self.graph_only_warmup else weights
            logits = self.logits_from_encoded(
                encoded_a, encoded_b, lengths_a, lengths_b, weights=read_weights
            )
            output: dict[str, torch.Tensor] = {"logits": logits}
            if self.cfg.stage == "two":
                # `_apply_intervention` may have substituted the graph the trunk read;
                # report what was actually read, not what the generator emitted.
                output["predicted_weights"] = self._apply_intervention(read_weights)[0]
            slot_row, topo_row = self._supervision_rows(
                merged, weights, read_weights, encoded_a, encoded_b, lengths_a, lengths_b
            )
            if slot_row is not None:
                output["slot_loss_rows"] = slot_row
            if topo_row is not None:
                output["topo_loss_rows"] = topo_row
            if "label" not in merged:
                return output
            self._add_composite_loss(output, logits, merged["label"], slot_row, topo_row)
            if self.graph_only_warmup:
                # Keep G on the synced task backward's autograd graph with an
                # exact zero, so a global batch without a valid slot row leaves
                # no generator parameter unreduced under DDP.
                output["loss"] = output["loss"] + 0.0 * weights.sum()
            return output

    def _eval_pair_precision(self, device: torch.device) -> AbstractContextManager[None]:
        """Return the pair pass's precision context for this mode.

        Training keeps the run's mixed precision. Evaluation must instead
        reproduce the published scorer exactly: `src.score_universe` promotes the
        cached encoder states to fp32 and drives the generator, the trunk and the
        output head with autocast disabled, and pins that as ``pair_autocast:
        False`` in the artifact's precision contract (spec section 9), because the
        reader's RRWP arithmetic needs fp32.

        Accelerate wraps this whole forward -- output head included -- in the
        run's autocast, so without this context validation would freeze the
        checkpoint and the ONE V_val-selected topology threshold on bf16-quantized
        logits that `test_protocol` then replays on the scorer's fp32 logits, and
        the same row can land on the other side of the threshold.

        Args:
            device: The device the encoder states live on.

        Returns:
            A no-op during training, otherwise an autocast-disabling context.
        """
        if self.training:
            return nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

    def supervises(self, batch: Mapping[str, torch.Tensor]) -> tuple[bool, bool]:
        """Whether a forward over ``batch`` emits ``slot_loss_rows`` and ``topo_loss_rows``.

        The trainer reads this before the task forward: the structural pass
        backpropagates first, and its share of the per-stream composite needs
        the task stream's global row count, which exists only if the task rows
        will carry the term at all (spec section 7.5).

        Args:
            batch: The batch as the forward will see it.

        Returns:
            ``(slot, topo)`` flags.
        """
        if self.cfg.training_policy == "head_only":
            return False, False
        slot = self.cfg.stage == "two" and TEMPLATE_KEY in batch
        topo = slot and self.teacher is not None and self.cfg.w_topo != 0.0
        return slot, topo

    def _supervision_rows(
        self,
        merged: Mapping[str, torch.Tensor],
        weights: torch.Tensor,
        read_weights: torch.Tensor,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the masked per-row ``L_slot`` and ``L_topo`` of this batch.

        ``weights`` is what ``L_slot`` is measured on and ``read_weights`` what
        ``L_topo``'s student side reads; they are the same tensor except in the
        graph-only warm-up, where the second is detached (`graph_only_warmup`).

        Only Stage II is supervised on the graph: in Stage I the compiled template
        is the model's *input*, so comparing it against itself would add a constant
        with no gradient path (spec sections 7.2 and 7.5). Self rows carry task BCE
        but are excluded from both terms through ``batch['motif_mask']``.

        The target is family-gated exactly as the prediction is: a closure-only or
        bridge-only arm may not emit an inactive family's edge, so leaving it in
        the target would charge ``L_slot`` for unavoidable error and let the
        teacher read topology the generator is forbidden to produce -- and that the
        matching Stage I reader never received either.
        """
        emits_slot, emits_topo = self.supervises(merged)
        if not emits_slot:
            return None, None
        target = self._gate_families(
            merged[TEMPLATE_KEY].to(device=weights.device, dtype=torch.float32)
        )
        mask = merged.get(TEMPLATE_MASK_KEY)
        row_mask = torch.ones_like(weights[:, 0]) if mask is None else mask.reshape(-1).to(weights)
        slot_row = (
            slot_loss_rows(
                weights,
                target,
                beta_p=self.cfg.beta_p,
                beta_q=self.cfg.beta_q,
                beta_a=self.cfg.beta_a,
                beta_i=self.cfg.beta_i,
                beta_c=self.cfg.beta_c,
                huber_delta=self.cfg.huber_delta,
            )
            * row_mask
        )
        if not emits_topo:
            return slot_row, None
        bundle = cast(nn.ModuleDict, self.teacher)
        # Both sides run through the immutable teacher: `R_T(Ahat)` keeps autograd
        # so the term reaches the generator, `R_T(A*)` is detached (spec 7.5).
        student_tokens = self._tokens(
            bundle, read_weights, encoded_a, encoded_b, lengths_a, lengths_b
        )
        with torch.no_grad():
            teacher_tokens = self._tokens(
                bundle, target, encoded_a, encoded_b, lengths_a, lengths_b
            )
        return slot_row, topo_loss_rows(student_tokens, teacher_tokens) * row_mask

    def _add_composite_loss(
        self,
        output: dict[str, torch.Tensor],
        logits: torch.Tensor,
        label: torch.Tensor,
        slot_row: torch.Tensor | None,
        topo_row: torch.Tensor | None,
    ) -> None:
        """Return the weighted task BCE and each term's share of the objective.

        ``loss`` is the task BCE alone. ``L_slot`` and ``L_topo`` are averaged
        over valid nonself rows *per stream* and then across streams (spec
        section 7.5), and only the trainer sees both streams, so it owns the
        composite and this class hands it the undetached, unscaled per-row terms.
        """
        flat = logits.reshape(-1).float()
        labels = label.reshape(-1).float()
        smoothing = float(self.base.label_smoothing)
        targets = labels * (1.0 - smoothing) + 0.5 * smoothing if smoothing > 0.0 else labels
        bce_row = F.binary_cross_entropy_with_logits(flat, targets, reduction="none")
        row_weights = 1.0 + (float(self.base.positive_weight) - 1.0) * labels
        weight_sum = row_weights.sum().detach()
        output["loss"] = (row_weights * bce_row).sum() / weight_sum
        output["loss_weight_sum"] = weight_sum
        output["task_loss"] = output["loss"].detach()
        zero = torch.zeros((), dtype=logits.dtype, device=logits.device)

        def stream_share(row: torch.Tensor | None, scale: float) -> torch.Tensor:
            """This stream's share of one graph term, undetached and unreduced."""
            if row is None or scale == 0.0:
                return zero
            return scale * row.mean()

        # The per-epoch gradient probe attributes each term to a parameter group.
        # ``task`` is the exact `loss`; ``slot``/``topo`` are this stream's share
        # of the composite the trainer forms. None is detached and every rank
        # emits all three.
        output["loss_term_task"] = output["loss"]
        output["loss_term_slot"] = stream_share(slot_row, self.w_slot_value)
        output["loss_term_topo"] = stream_share(topo_row, float(self.cfg.w_topo))


__all__ = [
    "BALANCED_W_SLOT",
    "CLOSURE_BIAS_INITS",
    "COUNT_FEATURES",
    "EDGE_TYPE_NAMES",
    "GENERATOR_GRAD_GROUPS",
    "WARMUP_LOSSES",
    "STAGE_TWO_INTERFACE_LR_SCALE",
    "FAMILIES",
    "FIELD_ORDER",
    "GATE_MODES",
    "GRAPH_ROW_WEIGHTINGS",
    "INTERVENTIONS",
    "LOSS_TERM_NAMES",
    "STAGES",
    "TEMPLATE_KEY",
    "TEMPLATE_MASK_KEY",
    "TOKEN_SOURCES",
    "TRAINING_POLICIES",
    "CorruptionConfig",
    "MotifCountHead",
    "MotifDirectTokens",
    "MotifGenerator",
    "MotifGritReader",
    "MotifPromptAdapter",
    "MotifPromptConfig",
    "MotifPromptCrossAttentionLayer",
    "ReaderConfig",
    "V3_1MotifPrompt",
    "dense_adjacency",
    "pool_residues",
    "motif_rrwp",
    "typed_adjacency",
]
