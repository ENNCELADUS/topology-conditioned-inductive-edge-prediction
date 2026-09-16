"""Motif-graph prompts read by GRIT on a frozen V3.1 trunk.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
Stage I trains the reader, the count head, the token projections, the role
embeddings and the prefix adapter on compiled true training templates; Stage II
trains a residue-conditioned generator that predicts the same 96 edge weights
from ``(x_u, x_v)`` alone, read through the identical interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import cast

import torch
from torch import nn

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
from src.model.egostitch.classifier.layers import (
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)
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
GATE_MODES = ("learned", "per_type", "mean_graph")
FAMILIES = ("closure", "bridge")


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
    """

    def __init__(self, width: int) -> None:
        """Build the single linear map.

        Args:
            width: Token width.
        """
        super().__init__()
        self.proj = nn.Linear(4, width)

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
            features = torch.stack(
                [
                    torch.log1p(stats["wedge_mass"]),
                    torch.log1p(stats["bridge_mass"]),
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
