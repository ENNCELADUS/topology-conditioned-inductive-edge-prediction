"""Motif-graph prompts read by GRIT on a frozen V3.1 trunk.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
Stage I trains the reader, the count head, the token projections, the role
embeddings and the prefix adapter on compiled true training templates; Stage II
trains a residue-conditioned generator that predicts the same 96 edge weights
from ``(x_u, x_v)`` alone, read through the identical interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    count_statistics,
)
from src.distill.motif_losses import slot_loss_rows, topo_loss_rows
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
STAGES = ("one", "two")
TOKEN_SOURCES = ("graph", "direct")
COUNT_FEATURES = ("all", "degree")
GATE_MODES = ("learned", "per_type", "mean_graph")
FAMILIES = ("closure", "bridge")
LOSS_TERM_NAMES = ("task", "slot", "topo")
_GRAPH_INTERVENTIONS = ("shuffle_graph", "permute_closure", "rewire_bridge")


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
    w_slot: float = 1.0
    w_topo: float = 0.1
    beta_p: float = 1.0
    beta_q: float = 1.0
    beta_a: float = 1.0
    beta_i: float = 1.0
    huber_delta: float = 1.0
    cache_templates: bool = False

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
        for name in ("w_slot", "w_topo", "beta_p", "beta_q", "beta_a", "beta_i"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"motif_prompt.{name} must be non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("motif_prompt.huber_delta must be positive")

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
    """

    incidence: torch.Tensor
    fixed_logits: torch.Tensor

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
        self.bridge_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * 0.02)
        self.witness_queries = nn.Parameter(torch.randn(8, _GATE_DIM) * 0.02)
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
            nn.init.normal_(output.weight, std=1e-3)
            nn.init.zeros_(output.bias)
        self.register_buffer("fixed_logits", torch.zeros(N_EDGES))
        self.register_buffer("incidence", _candidate_incidence())

    @torch.no_grad()
    def init_biases(self, mean_weights: torch.Tensor) -> None:
        """Set each head's output bias from the training mean weight of its type.

        The mean is clipped to ``[0.01, 0.99]`` before the logit so no head
        starts in a saturated region of the sigmoid (spec section 4).

        Args:
            mean_weights: ``(96,)`` training-corpus mean edge weights.

        Raises:
            ValueError: On a shape mismatch.
        """
        if tuple(mean_weights.shape) != (N_EDGES,):
            raise ValueError(f"mean weights must be a ({N_EDGES},) vector")
        low, high = _BIAS_CLIP
        clipped = mean_weights.float().clamp(low, high)
        self.fixed_logits.copy_(torch.logit(clipped))
        for edge_type, head in enumerate(self.heads):
            mask = _TYPE_MASKS[edge_type]
            output = cast(nn.Linear, cast(nn.Sequential, head)[-1])
            output.bias.fill_(float(torch.logit(clipped[mask].mean())))

    def _read(
        self, queries: torch.Tensor, states: torch.Tensor, pad: torch.Tensor | None
    ) -> torch.Tensor:
        """Let one shared query block attend over one endpoint's residue states."""
        expanded = queries.unsqueeze(0).expand(states.size(0), -1, -1)
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
        left = self._read(self.bridge_queries, state_u, pad_u)
        right = self._read(self.bridge_queries, state_v, pad_v)
        witness_u = self._read(self.witness_queries, state_u, pad_u)
        witness_v = self._read(self.witness_queries, state_v, pad_v)
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
            ``(B, 96)`` weights in ``[0, 1]``.
        """
        batch = encoded_u.size(0)
        if self.cfg.gate_mode == "mean_graph":
            return torch.sigmoid(self.fixed_logits).unsqueeze(0).expand(batch, -1)
        h = self._slot_states(encoded_u, encoded_v, lengths_u, lengths_v)
        for layer in self.message_layers:
            h = layer(h, self.incidence)

        rows = _EDGE_ROWS.to(h.device).reshape(-1)
        cols = _EDGE_COLS.to(h.device).reshape(-1)
        h_i, h_j = h[:, rows], h[:, cols]
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
        self._freeze_permanently_frozen()
        self.register_buffer("mean_template", torch.zeros(N_EDGES))
        # Derived from `cfg.families` on every construction, so a checkpoint can
        # never restore a stale family gate over a changed config.
        self.register_buffer("family_mask", _family_mask(self.cfg.families), persistent=False)
        self.teacher: nn.Module | None = None
        self.intervention: str = "none"
        self.corruption_seed = 42
        self.interface_open = self.cfg.stage == "one" or self.cfg.interface_warmup_epochs == 0

    @property
    def encoder(self) -> nn.Module:
        """The frozen base's per-node encoder (packed scoring caches its output)."""
        return self.base.encoder

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
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates."""
        return [param for param in self.parameters() if param.requires_grad]

    def _freeze_permanently_frozen(self) -> None:
        """Drop every parameter this configuration can never train.

        Two section 8 controls bypass a whole module: ``token_source='direct'``
        never calls the reader, and ``gate_mode='mean_graph'`` returns the
        installed mean adjacency without touching a generator parameter. Stage II
        additionally freezes the reader's role/input embeddings and every block
        but the last for the whole run (spec section 7.5). Production DDP wraps
        this model with ``find_unused_parameters=False``, so a trainable parameter
        that never receives a gradient aborts the second iteration: the freeze
        happens here, at construction, before any wrapping.
        """
        if self.cfg.token_source == "direct":
            self.reader.requires_grad_(False)
        elif self.cfg.stage == "two":
            for module in self.reader.frozen_stage_two_modules():
                module.requires_grad_(False)
        if self.cfg.gate_mode == "mean_graph" and self.generator is not None:
            self.generator.requires_grad_(False)

    def _interface_modules(self) -> list[nn.Module]:
        """The warm-up-controlled bundle: reader interface, count head, prefix adapter.

        Stage I trains the whole reader (spec section 7.2). Stage II may adapt
        only its final block and the output projections once the warm-up opens
        the group (spec section 7.5), so the earlier blocks and the role/input
        embeddings -- frozen outright in `_freeze_permanently_frozen` -- are not
        members of it and opening the group cannot reach them.
        """
        reader_modules: list[nn.Module] = (
            [self.reader] if self.cfg.stage == "one" else self.reader.adaptable_modules()
        )
        modules: list[nn.Module] = [*reader_modules, self.count_head, self.adapter]
        if self.direct_head is not None:
            modules.append(self.direct_head)
        return modules

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
    def install_mean_template(self, mean: torch.Tensor) -> None:
        """Publish the training-corpus mean adjacency ``Abar`` (spec section 3).

        Args:
            mean: ``(96,)`` mean of the randomised compiled training templates.

        Raises:
            ValueError: On a shape mismatch or a non-finite entry.
        """
        if tuple(mean.shape) != (N_EDGES,) or not torch.isfinite(mean).all():
            raise ValueError(f"mean template must be a finite ({N_EDGES},) vector")
        self.mean_template.copy_(mean.to(self.mean_template))
        if self.generator is not None:
            self.generator.init_biases(self.mean_template)

    def initialize_teacher(self) -> None:
        """Snapshot the loaded Stage I bundle as the immutable teacher ``R_T``.

        The bundle is the reader, the count head, the token projections carried
        inside them and the prefix adapter with its gates (spec section 7.4), so
        the student never starts from a different Stage I checkpoint than its
        teacher.
        """
        members: dict[str, nn.Module] = {
            "reader": deepcopy(self.reader),
            "count_head": deepcopy(self.count_head),
            "adapter": deepcopy(self.adapter),
        }
        if self.direct_head is not None:
            members["direct_head"] = deepcopy(self.direct_head)
        self.teacher = nn.ModuleDict(members).requires_grad_(False).eval()

    def set_corruption_step(self, step: int, seed: int = 42) -> None:
        """Set reproducible Stage I corruption for this global training step.

        Args:
            step: Global optimiser step.
            seed: Run seed.
        """
        self.corruption_seed = seed + step * 104729

    def _corrupt(self, weights: torch.Tensor) -> torch.Tensor:
        """Mix half the nonself Stage I rows towards ``Abar`` (spec section 3)."""
        cfg = self.cfg.corruption
        if not self.training or self.cfg.stage != "one" or cfg.prob == 0.0:
            return weights
        rng = torch.Generator(device=weights.device).manual_seed(self.corruption_seed)
        selected = torch.rand((weights.size(0), 1), device=weights.device, generator=rng) < cfg.prob
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
        """Read one graph through the student path or through the immutable teacher."""
        if bundle is None:
            reader, count_head = self.reader, self.count_head
            direct_head: nn.Module | None = self.direct_head
        else:
            reader = cast(MotifGritReader, bundle["reader"])
            count_head = cast(MotifCountHead, bundle["count_head"])
            # `nn.ModuleDict` is not a `Mapping`, so membership is the only read.
            has_direct = "direct_head" in bundle
            direct_head = bundle["direct_head"] if has_direct else None
        if direct_head is None:
            tokens: dict[str, torch.Tensor] = reader(weights)
        else:
            tokens = cast(MotifDirectTokens, direct_head)(
                pool_residues(encoded_u, lengths_u), pool_residues(encoded_v, lengths_v)
            )
        tokens["topo_cnt"] = count_head(weights)
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
        # The corruption mixes towards the ungated corpus mean, so the family
        # gate is applied last and holds in both stages (spec section 8).
        return self._gate_families(self._corrupt(target.to(encoded_a)))

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
        weights = self.resolve_weights(merged, encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.logits_from_encoded(
            encoded_a, encoded_b, lengths_a, lengths_b, weights=weights
        )
        output: dict[str, torch.Tensor] = {"logits": logits}
        if self.cfg.stage == "two":
            # `_apply_intervention` may have substituted the graph the trunk read;
            # report what was actually read, not what the generator emitted.
            output["predicted_weights"] = self._apply_intervention(weights)[0]
        slot_row, topo_row = self._supervision_rows(
            merged, weights, encoded_a, encoded_b, lengths_a, lengths_b
        )
        if slot_row is not None:
            output["slot_loss_rows"] = slot_row
        if topo_row is not None:
            output["topo_loss_rows"] = topo_row
        if "label" not in merged:
            return output
        self._add_composite_loss(output, logits, merged["label"], slot_row, topo_row)
        return output

    def _supervision_rows(
        self,
        merged: Mapping[str, torch.Tensor],
        weights: torch.Tensor,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the masked per-row ``L_slot`` and ``L_topo`` of this batch.

        Only Stage II is supervised on the graph: in Stage I the compiled template
        is the model's *input*, so comparing it against itself would add a constant
        with no gradient path (spec sections 7.2 and 7.5). Self rows carry task BCE
        but are excluded from both terms through ``batch['motif_mask']``.
        """
        target = merged.get(TEMPLATE_KEY)
        if self.cfg.stage != "two" or target is None:
            return None, None
        target = target.to(weights)
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
                huber_delta=self.cfg.huber_delta,
            )
            * row_mask
        )
        if self.teacher is None or self.cfg.w_topo == 0.0:
            return slot_row, None
        bundle = cast(nn.ModuleDict, self.teacher)
        # Both sides run through the immutable teacher: `R_T(Ahat)` keeps autograd
        # so the term reaches the generator, `R_T(A*)` is detached (spec 7.5).
        student_tokens = self._tokens(bundle, weights, encoded_a, encoded_b, lengths_a, lengths_b)
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
        output["loss_term_slot"] = stream_share(slot_row, float(self.cfg.w_slot))
        output["loss_term_topo"] = stream_share(topo_row, float(self.cfg.w_topo))


__all__ = [
    "COUNT_FEATURES",
    "FAMILIES",
    "FIELD_ORDER",
    "GATE_MODES",
    "INTERVENTIONS",
    "LOSS_TERM_NAMES",
    "STAGES",
    "TEMPLATE_KEY",
    "TEMPLATE_MASK_KEY",
    "TOKEN_SOURCES",
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
