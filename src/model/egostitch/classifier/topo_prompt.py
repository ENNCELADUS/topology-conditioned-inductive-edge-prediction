"""Topology-prompted V3.1: structural coordinates read through gated KV prefixes.

Stage I of the topology-representation-transfer pipeline
(``docs/superpowers/specs/2026-09-12-topology-prompt-stage1-design.md``). The
pair trunk of a bidirectional-cross `V3_1` reads, at every cross-attention
site of every layer, a short prefix built from the queried pair's fixed-
semantics structural coordinates (`src.data.struct_coords`): two endpoint
tokens (self / partner roles relative to the attending stream), one relation
token and one context token, each expanded to ``slots_per_field`` key/value
rows. The prefix branch is the separately-softmaxed, zero-init tanh-gated
attention of `src.model.egostitch.classifier.prefix.prefix_branch`, so at
initialisation the model computes exactly the base.

Two trainability modes: ``all`` (Stage I proper -- the whole trunk learns how a
given structure should alter the pair decision) and ``prompt`` (the frozen
`prefix_base` trunk with only the prompt path trainable -- the frozen-trunk
control that asks whether a fixed reader can exploit true structure at all).

The coordinates are measured on true structure, so every run of this family
is a ceiling diagnostic: the trainer refuses it outside ``--run-kind
diagnostic`` and the scorer marks its artifacts ``formal: False``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.struct_coords import (
    CONTEXT_DIM,
    COORD_DIM,
    COORD_SPEC,
    ENDPOINT_DIM,
    FIELD_SLICES,
    RELATION_DIM,
)
from src.model.egostitch.classifier.b0_v31 import V3_1, unpack_pair_batch, weighted_pair_bce
from src.model.egostitch.classifier.layers import CrossAttentionLayer, _build_padding_mask
from src.model.egostitch.classifier.prefix import SITES, prefix_branch

TRAINABLE_MODES = ("all", "prompt")
INTERVENTIONS = ("none", "gates_off", "mean", "mean_endpoint", "mean_relation", "mean_context")
FIELD_ORDER = ("endpoint_u", "endpoint_v", "relation", "context")
ROLE_SELF, ROLE_PARTNER, ROLE_RELATION, ROLE_CONTEXT = range(4)
COORDS_KEY = "struct_coords"


@dataclass(frozen=True)
class TopoPromptConfig:
    """The ``model.config.topo_prompt`` block.

    Attributes:
        trainable: ``"all"`` (trunk and prompt train) or ``"prompt"`` (frozen trunk).
        width: Token width of the prompt encoder.
        slots_per_field: Prefix rows each of the four fields expands to, per layer.
        field_mask_prob: Training-time probability of replacing one field of one
            row by its training mean (standardised zero); ``0`` disables it.
        coord_spec: Coordinate specification the checkpoint was trained on.
        base_checkpoint: Path of the base ``v3_1`` checkpoint (required for
            ``prompt``; an optional warm start for ``all``). Provenance only.
        base_checkpoint_sha256: SHA-256 of that file (provenance only, never verified).
    """

    trainable: str = "all"
    width: int = 128
    slots_per_field: int = 2
    field_mask_prob: float = 0.1
    coord_spec: str = COORD_SPEC
    base_checkpoint: str = ""
    base_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: On an unknown mode, a non-positive size, an out-of-range
                mask probability, an unsupported coordinate spec, or a
                ``prompt`` mode without a base checkpoint.
        """
        if self.trainable not in TRAINABLE_MODES:
            raise ValueError(
                f"topo_prompt.trainable must be one of {TRAINABLE_MODES}, got {self.trainable!r}"
            )
        if self.width <= 0 or self.slots_per_field <= 0:
            raise ValueError("topo_prompt.width and topo_prompt.slots_per_field must be positive")
        if not 0.0 <= self.field_mask_prob < 1.0:
            raise ValueError("topo_prompt.field_mask_prob must lie in [0, 1)")
        if self.coord_spec != COORD_SPEC:
            raise ValueError(
                f"topo_prompt.coord_spec {self.coord_spec!r} is not the supported {COORD_SPEC!r}"
            )
        if self.trainable == "prompt" and not self.base_checkpoint:
            raise ValueError("topo_prompt.trainable 'prompt' requires topo_prompt.base_checkpoint")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> TopoPromptConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.topo_prompt``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown topo_prompt keys: {unknown}")
        sha = raw.get("base_checkpoint_sha256")
        return cls(
            trainable=str(raw.get("trainable", "all")),
            width=int(cast(int, raw.get("width", 128))),
            slots_per_field=int(cast(int, raw.get("slots_per_field", 2))),
            field_mask_prob=float(cast(float, raw.get("field_mask_prob", 0.1))),
            coord_spec=str(raw.get("coord_spec", COORD_SPEC)),
            base_checkpoint=str(raw.get("base_checkpoint", "")),
            base_checkpoint_sha256=None if sha is None else str(sha),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))


class TopoPromptGenerator(nn.Module):
    """The prompt path: coordinate standardisation, field tokens, per-layer prefixes, gates.

    ``coord_mean`` / ``coord_std`` / ``coord_count`` are buffers published with
    the checkpoint; ``coord_count == 0`` means no statistics were ever set and
    `standardize` fails closed. Each field token is
    ``LayerNorm(GELU(Linear(field)) + role)``; the two endpoint tokens carry
    *self* and *partner* roles relative to the attending stream, so the
    prefix a stream reads names its own endpoint. Per layer, one linear map
    expands the four tokens to ``4 * slots_per_field`` prefix rows plus a
    learned static offset ``p0``. Gates ``(n_layers, SITES, n_heads)`` start at
    zero: the only zero factor.
    """

    coord_mean: torch.Tensor
    coord_std: torch.Tensor
    coord_count: torch.Tensor

    def __init__(self, d_model: int, n_layers: int, n_heads: int, cfg: TopoPromptConfig) -> None:
        """Build the parameters.

        Args:
            d_model: Trunk width.
            n_layers: Number of cross-attention layers.
            n_heads: Heads per attention site.
            cfg: The ``topo_prompt`` block.
        """
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.slots = len(FIELD_ORDER) * cfg.slots_per_field
        self.register_buffer("coord_mean", torch.zeros(COORD_DIM))
        self.register_buffer("coord_std", torch.ones(COORD_DIM))
        self.register_buffer("coord_count", torch.zeros(()))
        self.endpoint_proj = nn.Linear(ENDPOINT_DIM, cfg.width)
        self.relation_proj = nn.Linear(RELATION_DIM, cfg.width)
        self.context_proj = nn.Linear(CONTEXT_DIM, cfg.width)
        self.role_embed = nn.Parameter(torch.randn(len(FIELD_ORDER), cfg.width) * 0.02)
        self.token_norm = nn.LayerNorm(cfg.width)
        self.layer_proj = nn.ModuleList(
            nn.Linear(cfg.width, cfg.slots_per_field * d_model) for _ in range(n_layers)
        )
        self.p0 = nn.ParameterList(
            nn.Parameter(torch.randn(self.slots, d_model) * 0.02) for _ in range(n_layers)
        )
        self.gates = nn.Parameter(torch.zeros(n_layers, SITES, n_heads))

    @torch.no_grad()
    def set_coord_stats(self, mean: torch.Tensor, std: torch.Tensor, count: int) -> None:
        """Install the training-row standardisation statistics.

        Args:
            mean: ``(COORD_DIM,)`` per-coordinate mean.
            std: ``(COORD_DIM,)`` per-coordinate scale (already floored).
            count: Number of rows the statistics were taken over.

        Raises:
            ValueError: On a shape mismatch, a non-positive count, or a non-finite
                or non-positive scale.
        """
        if tuple(mean.shape) != (COORD_DIM,) or tuple(std.shape) != (COORD_DIM,):
            raise ValueError(f"coordinate statistics must have shape ({COORD_DIM},)")
        if count <= 0:
            raise ValueError("coordinate statistics need a positive row count")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("coordinate statistics must be finite with positive scale")
        self.coord_mean.copy_(mean.to(self.coord_mean))
        self.coord_std.copy_(std.to(self.coord_std))
        self.coord_count.fill_(float(count))

    def standardize(self, coords: torch.Tensor) -> torch.Tensor:
        """Return ``(coords - mean) / std`` in float32.

        Raises:
            ValueError: If the statistics were never set, or ``coords`` is not
                ``(B, COORD_DIM)``.
        """
        if float(self.coord_count) <= 0.0:
            raise ValueError(
                "topo_prompt coordinate statistics were never set (coord_count == 0); "
                "the trainer installs them at startup and the checkpoint carries them"
            )
        if coords.dim() != 2 or coords.size(-1) != COORD_DIM:
            raise ValueError(
                f"struct_coords must have shape (B, {COORD_DIM}), got {tuple(coords.shape)}"
            )
        return (coords.float() - self.coord_mean) / self.coord_std

    def mask_fields(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero (= training mean) every field whose ``(B, 4)`` mask entry is True."""
        out = z.clone()
        for field_index, field in enumerate(FIELD_ORDER):
            out[:, FIELD_SLICES[field]] = torch.where(
                mask[:, field_index : field_index + 1],
                torch.zeros_like(out[:, FIELD_SLICES[field]]),
                out[:, FIELD_SLICES[field]],
            )
        return out

    def training_field_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the Bernoulli field masking regulariser (training mode only)."""
        if not self.training or self.cfg.field_mask_prob <= 0.0:
            return z
        mask = torch.rand(z.size(0), len(FIELD_ORDER), device=z.device) < self.cfg.field_mask_prob
        return self.mask_fields(z, mask)

    def tokens(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(B, 4, width)`` token stacks for the two stream views.

        The first stack names endpoint ``u`` as *self* (the view of the stream
        encoding ``u``); the second names ``v`` as *self*. Relation and context
        tokens are shared.
        """
        e_u = F.gelu(self.endpoint_proj(z[:, FIELD_SLICES["endpoint_u"]]))
        e_v = F.gelu(self.endpoint_proj(z[:, FIELD_SLICES["endpoint_v"]]))
        rel = (
            F.gelu(self.relation_proj(z[:, FIELD_SLICES["relation"]]))
            + self.role_embed[ROLE_RELATION]
        )
        ctx = (
            F.gelu(self.context_proj(z[:, FIELD_SLICES["context"]])) + self.role_embed[ROLE_CONTEXT]
        )
        view_u = torch.stack(
            [e_u + self.role_embed[ROLE_SELF], e_v + self.role_embed[ROLE_PARTNER], rel, ctx], dim=1
        )
        view_v = torch.stack(
            [e_v + self.role_embed[ROLE_SELF], e_u + self.role_embed[ROLE_PARTNER], rel, ctx], dim=1
        )
        return self.token_norm(view_u), self.token_norm(view_v)

    def prefix(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        """Expand ``(B, 4, width)`` tokens to this layer's ``(B, slots, d_model)`` prefix."""
        proj = cast(nn.Linear, self.layer_proj[layer_index])
        rows = proj(tokens).view(tokens.size(0), self.slots, self.d_model)
        p0: torch.Tensor = self.p0[layer_index]
        return cast(torch.Tensor, rows + p0.unsqueeze(0))

    def gate(self, layer_index: int, site: int) -> torch.Tensor:
        """Return the raw (pre-tanh) per-head gate for one site."""
        return self.gates[layer_index, site]


class TopoPromptCrossAttentionLayer(nn.Module):
    """Re-drive one `CrossAttentionLayer` with a role-aware gated prefix at each site.

    Site ``A<-B`` and the CLS site read the prefix whose *self* endpoint is the
    A stream's node; site ``B<-A`` reads the other view. The layer and generator
    are plain references (not re-registered), as in the prefix arm.
    """

    layer: CrossAttentionLayer
    generator: TopoPromptGenerator

    def __init__(
        self, layer: CrossAttentionLayer, layer_index: int, generator: TopoPromptGenerator
    ) -> None:
        """Wrap a layer.

        Args:
            layer: The trunk's `CrossAttentionLayer` (registered under ``base``).
            layer_index: Its index in the trunk.
            generator: The shared prompt parameters (registered under ``generator``).
        """
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "generator", generator)
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
        layer = self.layer
        query_norm = layer.norm_attn(query)
        attn_out, _ = layer.attn(
            query_norm, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False
        )
        branch = prefix_branch(
            layer.attn, query_norm, prefix, self.generator.gate(self.layer_index, site), gate_scale
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
        """One bidirectional block plus the prefix branch at all three sites.

        Args:
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            cls_token: CLS state ``(B, 1, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.
            prefix_a: This layer's prefix in the A stream's view ``(B, slots, d_model)``.
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
            layer.attn_cls, cls_norm, prefix_a, self.generator.gate(self.layer_index, 2), gate_scale
        )
        cls_token = cls_token + layer.drop_cls_attn(attn_cls) + branch
        cls_token = cls_token + layer.drop_cls_ffn(layer.ff_cls(layer.norm_cls_ffn(cls_token)))
        return h_a, h_b, cls_token


class V3_1TopoPrompt(nn.Module):
    """A bidirectional-cross `V3_1` whose trunk reads structural-coordinate prefixes.

    ``generator`` is registered before ``base`` so ``next(model.parameters())``
    is a prompt parameter in both modes. Every forward needs
    ``batch["struct_coords"]`` of shape ``(B, COORD_DIM)``; a missing tensor
    raises rather than defaulting, so no code path can silently score without
    structure.
    """

    name: str = "v3_1_topo_prompt"

    def __init__(self, *, base: Mapping[str, object], topo_prompt: Mapping[str, object]) -> None:
        """Build the base from its config and the prompt path on top.

        Args:
            base: The base `V3_1` constructor kwargs.
            topo_prompt: The ``model.config.topo_prompt`` block.

        Raises:
            ValueError: If the base trunk has no bidirectional cross-attention layers.
        """
        super().__init__()
        self.cfg = TopoPromptConfig.from_mapping(topo_prompt)
        self.base_config: dict[str, object] = dict(base)
        base_model = V3_1(**self.base_config)
        trunk = base_model.cross_attention
        if trunk.mixing_mode != "bidirectional_cross" or len(trunk.layers) == 0:
            raise ValueError(
                "v3_1_topo_prompt needs a base with model.config.mixing.mode == "
                "'bidirectional_cross' and at least one cross-attention layer"
            )
        self.d_model = int(base_model.d_model)
        self.input_dim = int(base_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.generator = TopoPromptGenerator(
            self.d_model, len(trunk.layers), int(base_model.n_heads), self.cfg
        )
        self.base = base_model
        if self.cfg.trainable == "prompt":
            for param in self.base.parameters():
                param.requires_grad_(False)
            self.base.eval()
        self.prompt_layers = nn.ModuleList(
            TopoPromptCrossAttentionLayer(cast(CrossAttentionLayer, layer), index, self.generator)
            for index, layer in enumerate(trunk.layers)
        )
        self.intervention: str = "none"

    @property
    def encoder(self) -> nn.Module:
        """The base's per-node encoder (packed scoring caches its output)."""
        return self.base.encoder

    @property
    def frozen_base(self) -> bool:
        """Whether the base trunk is frozen (``trainable == "prompt"``)."""
        return self.cfg.trainable == "prompt"

    def train(self, mode: bool = True) -> V3_1TopoPrompt:
        """Switch mode; a frozen base stays in eval mode regardless.

        Args:
            mode: Training mode for the trainable parameters.

        Returns:
            ``self``.
        """
        super().train(mode)
        if self.frozen_base:
            self.base.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates: the prompt path, or everything."""
        if self.frozen_base:
            return list(self.generator.parameters())
        return list(self.parameters())

    def _apply_intervention(self, z: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Return the (possibly substituted) standardised coordinates and the gate scale.

        Raises:
            ValueError: On an unknown intervention name.
        """
        if self.intervention not in INTERVENTIONS:
            raise ValueError(f"unknown topo_prompt intervention {self.intervention!r}")
        if self.intervention == "none":
            return z, 1.0
        if self.intervention == "gates_off":
            return z, 0.0
        fields = {
            "mean": FIELD_ORDER,
            "mean_endpoint": ("endpoint_u", "endpoint_v"),
            "mean_relation": ("relation",),
            "mean_context": ("context",),
        }[self.intervention]
        mask = torch.zeros(z.size(0), len(FIELD_ORDER), dtype=torch.bool, device=z.device)
        for field in fields:
            mask[:, FIELD_ORDER.index(field)] = True
        return self.generator.mask_fields(z, mask), 1.0

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
        base_repr = trunk._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
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
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits from encoded token states and the pair's coordinates.

        Args:
            encoded_a: Encoder output for item A ``(B, L_a, d_model)``.
            encoded_b: Encoder output for item B ``(B, L_b, d_model)``.
            lengths_a: True sequence lengths for A.
            lengths_b: True sequence lengths for B.
            coords: Raw structural coordinates ``(B, COORD_DIM)`` of the pairs
                ``(a, b)``, with ``a`` as endpoint ``u``.

        Returns:
            The pair logits.

        Raises:
            ValueError: If an intervention is set while training, or the
                coordinates are mis-shaped or unstandardisable.
        """
        if self.intervention != "none" and self.training:
            raise ValueError("topo_prompt interventions are scoring-time only; call eval() first")
        z = self.generator.standardize(coords.to(encoded_a.device))
        z = self.generator.training_field_mask(z)
        z, gate_scale = self._apply_intervention(z)
        view_a, view_b = self.generator.tokens(z)
        prefixes_a = [self.generator.prefix(i, view_a) for i in range(len(self.prompt_layers))]
        prefixes_b = [self.generator.prefix(i, view_b) for i in range(len(self.prompt_layers))]
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
        return cast(torch.Tensor, self.base.output_head(pair_repr))

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score pairs whose structural coordinates ride in ``batch["struct_coords"]``.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            ``logits`` and, with ``label``, the weighted BCE ``loss`` and
            ``loss_weight_sum`` exactly as `V3_1` computes them.

        Raises:
            ValueError: If ``struct_coords`` is missing (this family never
                scores without structure).
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)
        coords = merged.get(COORDS_KEY)
        if coords is None:
            raise ValueError(
                "v3_1_topo_prompt requires batch['struct_coords']; this family never scores "
                "a pair without its structural coordinates"
            )
        emb_a, emb_b, lengths_a, lengths_b = unpack_pair_batch(merged, self.input_dim)
        if self.frozen_base:
            with torch.no_grad():
                encoded_a = self.base.encoder(emb_a, lengths_a)
                encoded_b = self.base.encoder(emb_b, lengths_b)
        else:
            encoded_a = self.base.encoder(emb_a, lengths_a)
            encoded_b = self.base.encoder(emb_b, lengths_b)
        logits = self.logits_from_encoded(encoded_a, encoded_b, lengths_a, lengths_b, coords=coords)
        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            output["loss"], output["loss_weight_sum"] = weighted_pair_bce(
                logits,
                merged["label"],
                label_smoothing=float(self.base.label_smoothing),
                positive_weight=float(self.base.positive_weight),
            )
        return output


__all__ = [
    "COORDS_KEY",
    "INTERVENTIONS",
    "TRAINABLE_MODES",
    "TopoPromptConfig",
    "TopoPromptCrossAttentionLayer",
    "TopoPromptGenerator",
    "V3_1TopoPrompt",
]
