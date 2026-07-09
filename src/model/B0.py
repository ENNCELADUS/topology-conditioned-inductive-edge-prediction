"""Standalone V3.1 paired-item model definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class DropPath(nn.Module):
    """Stochastic depth per-sample residual drop-path layer."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(max(0.0, min(1.0, drop_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic path dropping.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after stochastic depth.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def _build_padding_mask(lengths: torch.Tensor | None, max_len: int) -> torch.Tensor | None:
    """Create a padding mask from sequence lengths.

    Args:
        lengths: Batch sequence lengths.
        max_len: Padded sequence length.

    Returns:
        Boolean padding mask of shape ``(batch, max_len)``.
    """
    if lengths is None:
        return None
    if lengths.dim() != 1:
        raise ValueError("lengths must be a 1D tensor of shape (batch_size,)")
    return torch.arange(max_len, device=lengths.device).expand(
        lengths.size(0), max_len
    ) >= lengths.unsqueeze(1)


def _to_int(value: object, field_name: str) -> int:
    """Convert config value to integer."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be int-compatible") from error
    raise ValueError(f"{field_name} must be int-compatible")


def _to_float(value: object, field_name: str) -> float:
    """Convert config value to float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be float-compatible") from error
    raise ValueError(f"{field_name} must be float-compatible")


def _to_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Validate config value as key-value mapping."""
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be a mapping")


class SiameseEncoder(nn.Module):
    """Shared transformer encoder for each item sequence."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        token_dropout: float,
        stochastic_depth: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.token_dropout = nn.Dropout(token_dropout) if token_dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        )
        # Stochastic depth: linear schedule across depth, identity when rate == 0
        sd_max = float(max(0.0, min(1.0, stochastic_depth)))
        if sd_max > 0.0 and n_layers > 0:
            rates = [sd_max * float(i + 1) / float(n_layers) for i in range(n_layers)]
            self.drop_paths = nn.ModuleList([DropPath(r) for r in rates])
        else:
            self.drop_paths = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode token embeddings.

        Args:
            embeddings: Input embedding tensor ``(batch, seq_len, input_dim)``.
            lengths: Sequence lengths tensor ``(batch,)``.

        Returns:
            Encoded embedding tensor ``(batch, seq_len, d_model)``.
        """
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be of shape (batch_size, seq_len, embedding_dim)")
        projected = self.input_projection(embeddings)
        projected = self.token_dropout(projected)
        max_len = projected.size(1)
        padding_mask = _build_padding_mask(lengths, max_len)
        for idx, layer in enumerate(self.layers):
            residual_in = projected
            y = layer(projected, src_key_padding_mask=padding_mask)
            # Apply layer-level stochastic depth to the residual delta f(x) = y - x
            delta = y - residual_in
            delta = self.drop_paths[idx](delta)  # Identity in eval or when rate==0
            projected = residual_in + delta
        return cast(torch.Tensor, self.output_norm(projected))


class MLPHead(nn.Module):
    """Configurable MLP head for binary pair logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        activation: str,
        norm: str,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        activation_map: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}' for MLPHead")

        def build_norm(dim: int) -> nn.Module:
            if norm == "layernorm":
                return nn.LayerNorm(dim)
            if norm == "batchnorm":
                return nn.BatchNorm1d(dim)
            if norm == "none":
                return nn.Identity()
            raise ValueError(f"Unsupported norm '{norm}' for MLPHead")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(build_norm(hidden_dim))
            layers.append(activation_map[activation])
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers with Xavier uniform weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output logits.

        Args:
            x: Input feature tensor.

        Returns:
            Output logit tensor.
        """
        return cast(torch.Tensor, self.layers(x))


def inner_token_mask(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Return mask for non-special inner_token tokens, excluding BOS/EOS/padding."""
    batch_size, sequence_length, _ = x.shape
    if padding_mask is not None:
        valid_mask = ~padding_mask
    else:
        valid_mask = torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        )

    positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
    valid_lengths = valid_mask.sum(dim=1, keepdim=True)
    mask = valid_mask & (positions > 0) & (positions < valid_lengths - 1)
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("Pair readout requires at least one inner_token token between BOS and EOS")
    return mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a token sequence with a boolean keep mask."""
    mask_3d = mask.float().unsqueeze(-1)
    return (x * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pool a token sequence with a boolean keep mask."""
    masked_x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    return masked_x.max(dim=1).values


class TokenCompressor(nn.Module):
    """Mask-aware inner_token compressor for fixed-size grid sketches."""

    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("sketch_tokens must be >= 1")
        self.num_tokens = int(num_tokens)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Compress inner_token states to ``num_tokens`` by adaptive average pooling."""
        mask = inner_token_mask(x=x, padding_mask=padding_mask)
        pooled_rows: list[torch.Tensor] = []
        for sample, sample_mask in zip(x, mask, strict=True):
            inner_tokens = sample[sample_mask].transpose(0, 1).unsqueeze(0)
            pooled = F.adaptive_avg_pool1d(inner_tokens, self.num_tokens).squeeze(0).transpose(0, 1)
            pooled_rows.append(pooled)
        return torch.stack(pooled_rows, dim=0)


def _linear(in_features: int, out_features: int, *, use_spectral_norm: bool = False) -> nn.Linear:
    """Build an optionally spectral-normalized linear layer."""
    layer = nn.Linear(in_features, out_features)
    return spectral_norm(layer) if use_spectral_norm else layer


class PairContextGatedReadout(nn.Module):
    """InnerToken-only mean/max/pair-conditioned-attention readout."""

    def __init__(self, d_model: int, dropout: float, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            _linear(d_model * 2, d_model, use_spectral_norm=use_spectral_norm),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_scorer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            _linear(d_model * 2, d_model, use_spectral_norm=use_spectral_norm),
            nn.GELU(),
            nn.Dropout(dropout),
            _linear(d_model, 1, use_spectral_norm=use_spectral_norm),
        )
        self.branch_gate = _linear(d_model * 12, 3, use_spectral_norm=use_spectral_norm)
        self.branch_proj = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            _linear(d_model * 4, d_model, use_spectral_norm=use_spectral_norm),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            _linear(d_model * 2, d_model, use_spectral_norm=use_spectral_norm),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _attention_pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        other_context: torch.Tensor,
    ) -> torch.Tensor:
        context = other_context.unsqueeze(1).expand(-1, x.size(1), -1)
        scores = self.attn_scorer(torch.cat([x, context], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

    @staticmethod
    def _pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_vec: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a pair representation from inner_token-only branch features."""
        inner_token_a = inner_token_mask(x=h_a, padding_mask=mask_a)
        inner_token_b = inner_token_mask(x=h_b, padding_mask=mask_b)

        mean_a = masked_mean(h_a, inner_token_a)
        mean_b = masked_mean(h_b, inner_token_b)
        max_a = masked_max(h_a, inner_token_a)
        max_b = masked_max(h_b, inner_token_b)
        context_a = self.context_proj(torch.cat([mean_a, max_a], dim=-1))
        context_b = self.context_proj(torch.cat([mean_b, max_b], dim=-1))
        attn_a = self._attention_pool(h_a, inner_token_a, context_b)
        attn_b = self._attention_pool(h_b, inner_token_b, context_a)

        branches = torch.stack(
            [
                self._pair_features(mean_a, mean_b),
                self._pair_features(max_a, max_b),
                self._pair_features(attn_a, attn_b),
            ],
            dim=1,
        )
        gate_input = branches.flatten(start_dim=1)
        gate_weights = torch.softmax(self.branch_gate(gate_input), dim=1).unsqueeze(-1)
        branch_repr = (branches * gate_weights).sum(dim=1)
        projected = self.branch_proj(branch_repr)
        return cast(torch.Tensor, self.final_proj(torch.cat([cls_vec, projected], dim=-1)))


class GridSketchFusionReadout(nn.Module):
    """Fuse no-CLS rich-pooling representation with a latent grid sketch."""

    def __init__(
        self,
        d_model: int,
        sketch_tokens: int,
        pair_dim: int,
        cnn_dim: int,
        cnn_blocks: int,
        cnn_dropout: float,
        dropout: float,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be >= 1")
        if cnn_dim <= 0:
            raise ValueError("cnn_dim must be >= 1")
        if cnn_blocks <= 0:
            raise ValueError("cnn_blocks must be >= 1")
        self.eps = float(eps)
        self.compressor = TokenCompressor(num_tokens=sketch_tokens)
        self.proj_a = nn.Sequential(nn.Linear(d_model, pair_dim), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(d_model, pair_dim), nn.GELU())
        in_channels = 4 * pair_dim + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, cnn_dim),
            nn.GELU(),
            *[
                _GridResidualBlock(channels=cnn_dim, dropout=cnn_dropout)
                for _ in range(cnn_blocks)
            ],
        )
        self.grid_proj = nn.Sequential(
            nn.LayerNorm(cnn_dim * 2),
            nn.Linear(cnn_dim * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_grid(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_a(h_a)
        z_b = self.proj_b(h_b)
        len_a = z_a.size(1)
        len_b = z_b.size(1)
        z_a_exp = z_a.unsqueeze(2).expand(-1, -1, len_b, -1)
        z_b_exp = z_b.unsqueeze(1).expand(-1, len_a, -1, -1)
        cosine = F.cosine_similarity(z_a_exp, z_b_exp, dim=-1, eps=self.eps).unsqueeze(-1)
        grid = torch.cat(
            [
                z_a_exp,
                z_b_exp,
                torch.abs(z_a_exp - z_b_exp),
                z_a_exp * z_b_exp,
                cosine,
            ],
            dim=-1,
        )
        return grid.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        base_repr: torch.Tensor,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return fused rich-pooling plus grid-sketch pair representation."""
        sketch_a = self.compressor(h_a, mask_a)
        sketch_b = self.compressor(h_b, mask_b)
        features = self.cnn(self._build_grid(sketch_a, sketch_b))
        pooled_max = F.adaptive_max_pool2d(features, (1, 1)).flatten(1)
        pooled_mean = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        grid_repr = self.grid_proj(torch.cat([pooled_max, pooled_mean], dim=1))
        return cast(torch.Tensor, self.fusion(torch.cat([base_repr, grid_repr], dim=1)))


class _GridResidualBlock(nn.Module):
    """Small residual CNN block for grid sketches."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(x + self.block(x)))


POOLING_BASE_COMPONENTS = ("first_token", "mean", "attn", "max")
POOLING_GATED_COMPONENT = "gated"
DEFAULT_RICH_POOLING_COMPONENTS = (*POOLING_BASE_COMPONENTS, POOLING_GATED_COMPONENT)
PAIR_READOUT_MODES = ("rich_pooling", "pair_context_gated", "grid_sketch_fusion")
MIXING_MODES = ("bidirectional_cross", "none", "block_self")
ORDER_AGGREGATION_MODES = ("single", "abba_max")


def _to_bool(value: object, field_name: str) -> bool:
    """Convert a config value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be bool-compatible")


def _parse_rich_pooling_components(value: object) -> tuple[str, ...]:
    """Parse enabled rich-pooling components in canonical order."""
    if value is None:
        return DEFAULT_RICH_POOLING_COMPONENTS
    if not isinstance(value, list) or not value:
        raise ValueError("model_config.rich_pooling.components must be a non-empty list")

    raw_components: set[str] = set()
    valid_components = set(DEFAULT_RICH_POOLING_COMPONENTS)
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("model_config.rich_pooling.components must contain strings")
        component = item.strip().lower()
        if component not in valid_components:
            raise ValueError(
                "model_config.rich_pooling.components contains unsupported component: "
                f"{component}"
            )
        if component in raw_components:
            raise ValueError(
                f"model_config.rich_pooling.components contains duplicate: {component}"
            )
        raw_components.add(component)

    base_components = tuple(
        component for component in POOLING_BASE_COMPONENTS if component in raw_components
    )
    if not base_components:
        raise ValueError("model_config.rich_pooling.components must enable a base component")
    if POOLING_GATED_COMPONENT in raw_components and not base_components:
        raise ValueError("model_config.rich_pooling.components cannot enable gated alone")

    components = list(base_components)
    if POOLING_GATED_COMPONENT in raw_components:
        components.append(POOLING_GATED_COMPONENT)
    return tuple(components)

class CrossAttentionLayer(nn.Module):
    """Shared-weight bidirectional cross-attention with FFN and CLS pooling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_cls_attn = nn.LayerNorm(d_model)
        self.norm_cls_ffn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.attn_cls = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff_cls = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.drop_cls_attn = nn.Dropout(dropout)
        self.drop_cls_ffn = nn.Dropout(dropout)

    def _attend(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query_norm = self.norm_attn(query)
        attn_out, _ = self.attn(query_norm, key_value, key_value, key_padding_mask=key_padding_mask)
        return query + cast(torch.Tensor, self.drop_attn(attn_out))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward sub-layer with residual connection."""
        return x + cast(torch.Tensor, self.drop_ffn(self.ffn(self.norm_ffn(x))))

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one bidirectional cross-attention block.

        Args:
            h_a: Item A hidden states.
            h_b: Item B hidden states.
            cls_token: CLS token state.
            mask_a: Padding mask for sequence A.
            mask_b: Padding mask for sequence B.

        Returns:
            Updated ``(h_a, h_b, cls_token)`` tuple.
        """
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)
        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        combined = torch.cat([h_a, h_b], dim=1)
        combined_mask = (
            torch.cat([mask_a, mask_b], dim=1)
            if mask_a is not None and mask_b is not None
            else None
        )

        cls_norm = self.norm_cls_attn(cls_token)
        attn_cls, _ = self.attn_cls(cls_norm, combined, combined, key_padding_mask=combined_mask)
        cls_token = cls_token + self.drop_cls_attn(attn_cls)
        cls_token = cls_token + self.drop_cls_ffn(self.ff_cls(self.norm_cls_ffn(cls_token)))

        return h_a, h_b, cls_token


class BlockSelfMixingLayer(nn.Module):
    """Self-only A/B mixing layer that avoids cross-stream token mixing."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)

    def _self_attend(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=padding_mask)
        x = x + cast(torch.Tensor, self.drop_attn(attn_out))
        return x + cast(torch.Tensor, self.drop_ffn(self.ffn(self.norm_ffn(x))))

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one self-only layer and keep the pair CLS token unchanged."""
        return self._self_attend(h_a, mask_a), self._self_attend(h_b, mask_b), cls_token


# ---------------------------------------------------------------------------
# Rich pooling
# ---------------------------------------------------------------------------


class RichPooling(nn.Module):
    """Configurable first-token and inner_token pooling with optional gated fusion.

    ``first_token`` uses the first source special-token embedding. InnerToken pooling
    components use positions between the first BOS token and final EOS token.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        components: tuple[str, ...] = DEFAULT_RICH_POOLING_COMPONENTS,
    ) -> None:
        super().__init__()
        self.components = _parse_rich_pooling_components(list(components))
        self.base_components = tuple(
            component for component in POOLING_BASE_COMPONENTS if component in self.components
        )
        self.use_gated = POOLING_GATED_COMPONENT in self.components
        self.attn_scorer = nn.Linear(d_model, 1) if "attn" in self.base_components else None
        self.pool_gate = (
            nn.Linear(d_model * len(self.base_components), len(self.base_components))
            if self.use_gated
            else None
        )
        projection_components = len(self.base_components) + int(self.use_gated)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * projection_components),
            nn.Linear(d_model * projection_components, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _inner_token_mask(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Return a mask for non-special inner_token tokens."""
        batch_size, sequence_length, _ = x.shape
        if padding_mask is not None:
            valid_mask = ~padding_mask
        else:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )

        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)
        inner_token_mask = valid_mask & (positions > 0) & (positions < valid_lengths - 1)
        if bool((inner_token_mask.sum(dim=1) == 0).any()):
            raise ValueError(
                "RichPooling requires at least one inner_token token between BOS and EOS"
            )
        return inner_token_mask

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Pool token sequence to a single ``d_model`` vector.

        Args:
            x: Token hidden states ``(batch, seq_len, d_model)``.
            padding_mask: Boolean mask ``(batch, seq_len)`` — ``True`` = pad.

        Returns:
            Pooled vector ``(batch, d_model)``.
        """
        pooled_by_component: dict[str, torch.Tensor] = {}
        if "first_token" in self.base_components:
            pooled_by_component["first_token"] = x[:, 0]

        if any(component in self.base_components for component in ("mean", "attn", "max")):
            inner_token_mask = self._inner_token_mask(x=x, padding_mask=padding_mask)
            inner_token_float_mask = inner_token_mask.float()
            inner_token_3d = inner_token_float_mask.unsqueeze(-1)

            if "mean" in self.base_components:
                pooled_by_component["mean"] = (x * inner_token_3d).sum(dim=1) / inner_token_3d.sum(
                    dim=1
                ).clamp_min(1.0)

            if "attn" in self.base_components:
                if self.attn_scorer is None:
                    raise RuntimeError(
                        "attn_scorer must be initialized when attention pooling is enabled"
                    )
                scores = self.attn_scorer(x).squeeze(-1)
                scores = scores.masked_fill(~inner_token_mask, torch.finfo(scores.dtype).min)
                attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled_by_component["attn"] = (x * attn_weights).sum(dim=1)

            if "max" in self.base_components:
                masked_x = x.masked_fill(~inner_token_mask.unsqueeze(-1), torch.finfo(x.dtype).min)
                pooled_by_component["max"] = masked_x.max(dim=1).values

        base_vectors = [pooled_by_component[component] for component in self.base_components]
        output_vectors = list(base_vectors)
        if self.use_gated:
            if self.pool_gate is None:
                raise RuntimeError("pool_gate must be initialized when gated pooling is enabled")
            gate_input = torch.cat(base_vectors, dim=1)
            gate_weights = torch.softmax(self.pool_gate(gate_input), dim=1).unsqueeze(-1)
            pooled_stack = torch.stack(base_vectors, dim=1)
            output_vectors.append((pooled_stack * gate_weights).sum(dim=1))

        combined = torch.cat(output_vectors, dim=1)
        return cast(torch.Tensor, self.proj(combined))


# ---------------------------------------------------------------------------
# Pair cross-attention with rich pooling
# ---------------------------------------------------------------------------


class PairCrossAttention(nn.Module):
    """Stacked cross-attention encoder with configurable pair readout."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        pooling_components: tuple[str, ...] = DEFAULT_RICH_POOLING_COMPONENTS,
        pair_readout_mode: str = "rich_pooling",
        sketch_tokens: int = 64,
        pair_dim: int = 32,
        cnn_dim: int = 64,
        cnn_blocks: int = 2,
        cnn_dropout: float = 0.1,
        mixing_mode: str = "bidirectional_cross",
        pair_readout_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if pair_readout_mode not in PAIR_READOUT_MODES:
            raise ValueError(
                "model_config.pair_readout.mode must be one of: "
                f"{', '.join(PAIR_READOUT_MODES)}"
            )
        if mixing_mode not in MIXING_MODES:
            raise ValueError(
                "model_config.mixing.mode must be one of: "
                f"{', '.join(MIXING_MODES)}"
            )
        self.pair_readout_mode = pair_readout_mode
        self.mixing_mode = mixing_mode
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if mixing_mode == "bidirectional_cross":
            self.layers = nn.ModuleList(
                CrossAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            )
        elif mixing_mode == "block_self":
            self.layers = nn.ModuleList(
                BlockSelfMixingLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            )
        else:
            self.layers = nn.ModuleList()
        if pair_readout_mode in {"rich_pooling", "grid_sketch_fusion"}:
            shared_pool = RichPooling(
                d_model=d_model, dropout=dropout, components=pooling_components
            )
            self.pool_a = shared_pool
            self.pool_b = shared_pool
            self.fusion = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if pair_readout_mode == "pair_context_gated":
            self.pair_context_readout = PairContextGatedReadout(
                d_model=d_model,
                dropout=dropout,
                use_spectral_norm=pair_readout_spectral_norm,
            )
        if pair_readout_mode == "grid_sketch_fusion":
            self.grid_sketch_readout = GridSketchFusionReadout(
                d_model=d_model,
                sketch_tokens=sketch_tokens,
                pair_dim=pair_dim,
                cnn_dim=cnn_dim,
                cnn_blocks=cnn_blocks,
                cnn_dropout=cnn_dropout,
                dropout=dropout,
            )

    def _rich_pooling_readout(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_vec: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        if not hasattr(self, "pool_a") or not hasattr(self, "pool_b"):
            raise RuntimeError("rich pooling readout modules are not initialized")
        pooled_a = self.pool_a(h_a, mask_a)
        pooled_b = self.pool_b(h_b, mask_b)
        fused = torch.cat([cls_vec, pooled_a, pooled_b], dim=1)
        return cast(torch.Tensor, self.fusion(fused))

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Pool pair representation from encoded items.

        Args:
            h_a: Item A hidden states ``(batch, seq_len_a, d_model)``.
            h_b: Item B hidden states ``(batch, seq_len_b, d_model)``.
            lengths_a: Sequence lengths for A ``(batch,)``.
            lengths_b: Sequence lengths for B ``(batch,)``.

        Returns:
            Fused pair representation ``(batch, d_model)``.
        """
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")

        batch_size = h_a.size(0)
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))

        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        for layer in self.layers:
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)

        cls_vec = cls_token.squeeze(1)
        if self.pair_readout_mode == "pair_context_gated":
            return cast(
                torch.Tensor,
                self.pair_context_readout(h_a, h_b, cls_vec, mask_a, mask_b),
            )

        base_repr = self._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
        if self.pair_readout_mode == "grid_sketch_fusion":
            return cast(
                torch.Tensor,
                self.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b),
            )
        return base_repr


# ---------------------------------------------------------------------------
# V3_1 top-level model
# ---------------------------------------------------------------------------


class V3_1(nn.Module):
    """V3.1 model — V3 with rich per-item pooling (CLS+mean+attn+max+gate).

    Config keys are identical to V3; no new required fields.
    """

    name: str = "v3.1"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [f for f in required_fields if f not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model = _to_int(model_config["d_model"], "model_config.d_model")
        self.encoder_layers = _to_int(model_config["encoder_layers"], "model_config.encoder_layers")
        self.cross_attn_layers = _to_int(
            model_config["cross_attn_layers"], "model_config.cross_attn_layers"
        )
        self.n_heads = _to_int(model_config["n_heads"], "model_config.n_heads")

        mlp_cfg_raw = model_config.get("mlp_head")
        if not isinstance(mlp_cfg_raw, dict) or not mlp_cfg_raw:
            raise ValueError("mlp_head configuration is required for V3_1")
        mlp_cfg = _to_mapping(mlp_cfg_raw, "model_config.mlp_head")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError("mlp_head.hidden_dims and mlp_head.dropout must be provided")
        hidden_dims_raw = mlp_cfg["hidden_dims"]
        if not isinstance(hidden_dims_raw, list) or not hidden_dims_raw:
            raise ValueError("mlp_head.hidden_dims must be a non-empty list")
        self.mlp_hidden_dims = [
            _to_int(v, "model_config.mlp_head.hidden_dims") for v in hidden_dims_raw
        ]
        self.mlp_dropout = _to_float(mlp_cfg["dropout"], "model_config.mlp_head.dropout")
        self.mlp_activation = str(mlp_cfg.get("activation", "gelu"))
        self.mlp_norm = str(mlp_cfg.get("norm", "layernorm"))
        self.mlp_spectral_norm = _to_bool(
            mlp_cfg.get("spectral_norm", False),
            "model_config.mlp_head.spectral_norm",
        )

        reg_cfg_raw = model_config.get("regularization")
        if not isinstance(reg_cfg_raw, dict) or "dropout" not in reg_cfg_raw:
            raise ValueError("regularization.dropout must be provided for V3_1")
        reg_cfg = _to_mapping(reg_cfg_raw, "model_config.regularization")
        self.encoder_dropout = _to_float(reg_cfg["dropout"], "model_config.regularization.dropout")
        self.cross_attention_dropout = _to_float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout),
            "model_config.regularization.cross_attention_dropout",
        )
        self.token_dropout = _to_float(
            reg_cfg.get("token_dropout", 0.0), "model_config.regularization.token_dropout"
        )
        self.stochastic_depth = _to_float(
            reg_cfg.get("stochastic_depth", 0.0), "model_config.regularization.stochastic_depth"
        )
        mixing_cfg_raw = model_config.get("mixing", {})
        if mixing_cfg_raw is None:
            mixing_cfg_raw = {}
        if not isinstance(mixing_cfg_raw, dict):
            raise ValueError("model_config.mixing must be a mapping")
        mixing_cfg = _to_mapping(mixing_cfg_raw, "model_config.mixing")
        self.mixing_mode = str(
            mixing_cfg.get("mode", "bidirectional_cross")
        ).lower()
        if self.mixing_mode not in MIXING_MODES:
            raise ValueError(
                "model_config.mixing.mode must be one of: "
                f"{', '.join(MIXING_MODES)}"
            )
        rich_pooling_cfg_raw = model_config.get("rich_pooling", {})
        if rich_pooling_cfg_raw is None:
            rich_pooling_cfg_raw = {}
        if not isinstance(rich_pooling_cfg_raw, dict):
            raise ValueError("model_config.rich_pooling must be a mapping")
        rich_pooling_cfg = _to_mapping(rich_pooling_cfg_raw, "model_config.rich_pooling")
        self.rich_pooling_components = _parse_rich_pooling_components(
            rich_pooling_cfg.get("components")
        )
        pair_readout_cfg_raw = model_config.get("pair_readout", {})
        if pair_readout_cfg_raw is None:
            pair_readout_cfg_raw = {}
        if not isinstance(pair_readout_cfg_raw, dict):
            raise ValueError("model_config.pair_readout must be a mapping")
        pair_readout_cfg = _to_mapping(pair_readout_cfg_raw, "model_config.pair_readout")
        self.pair_readout_mode = str(pair_readout_cfg.get("mode", "rich_pooling")).lower()
        if self.pair_readout_mode not in PAIR_READOUT_MODES:
            raise ValueError(
                "model_config.pair_readout.mode must be one of: "
                f"{', '.join(PAIR_READOUT_MODES)}"
            )
        self.order_aggregation = str(pair_readout_cfg.get("order_aggregation", "single")).lower()
        if self.order_aggregation not in ORDER_AGGREGATION_MODES:
            raise ValueError(
                "model_config.pair_readout.order_aggregation must be one of: "
                f"{', '.join(ORDER_AGGREGATION_MODES)}"
            )
        self.pair_readout_spectral_norm = _to_bool(
            pair_readout_cfg.get("spectral_norm", False),
            "model_config.pair_readout.spectral_norm",
        )
        sketch_tokens = _to_int(
            pair_readout_cfg.get("sketch_tokens", 64),
            "model_config.pair_readout.sketch_tokens",
        )
        pair_dim = _to_int(
            pair_readout_cfg.get("pair_dim", 32),
            "model_config.pair_readout.pair_dim",
        )
        cnn_dim = _to_int(
            pair_readout_cfg.get("cnn_dim", 64),
            "model_config.pair_readout.cnn_dim",
        )
        cnn_blocks = _to_int(
            pair_readout_cfg.get("cnn_blocks", 2),
            "model_config.pair_readout.cnn_blocks",
        )
        cnn_dropout = _to_float(
            pair_readout_cfg.get("cnn_dropout", 0.1),
            "model_config.pair_readout.cnn_dropout",
        )

        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
        )
        self.cross_attention = PairCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
            pooling_components=self.rich_pooling_components,
            pair_readout_mode=self.pair_readout_mode,
            sketch_tokens=sketch_tokens,
            pair_dim=pair_dim,
            cnn_dim=cnn_dim,
            cnn_blocks=cnn_blocks,
            cnn_dropout=cnn_dropout,
            mixing_mode=self.mixing_mode,
            pair_readout_spectral_norm=self.pair_readout_spectral_norm,
        )
        self.output_head = MLPHead(
            input_dim=self.d_model,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=1,
            dropout=self.mlp_dropout,
            activation=self.mlp_activation,
            norm=self.mlp_norm,
        )
        if self.mlp_spectral_norm:
            self._apply_output_head_spectral_norm()

    def _apply_output_head_spectral_norm(self) -> None:
        """Apply spectral norm to output-head linear layers."""
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                spectral_norm(module)

    def _pair_representation(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return a pair representation with optional AB/BA max aggregation."""
        feature_ab = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)
        if self.order_aggregation == "single":
            return cast(torch.Tensor, feature_ab)

        feature_ba = self.cross_attention(encoded_b, encoded_a, lengths_b, lengths_a)
        return torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a model forward pass.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            Output dictionary containing ``logits`` and optional ``loss``.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)

        if "emb_a" not in merged or "emb_b" not in merged:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = merged["emb_a"]
        emb_b = merged["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = merged.get("len_a")
        lengths_b = merged.get("len_b")
        if lengths_a is None:
            lengths_a = torch.full((emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long)
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full((emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long)
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)
        pair_repr = self._pair_representation(encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.output_head(pair_repr)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, labels_for_loss
            )

        return output



BEST_V3_1_CONFIG: dict[str, object] = {
    "input_dim": 1536,
    "d_model": 256,
    "encoder_layers": 3,
    "cross_attn_layers": 3,
    "n_heads": 8,
    "mlp_head": {
        "hidden_dims": [256, 128, 64],
        "dropout": 0.2,
        "activation": "gelu",
        "norm": "layernorm",
        "spectral_norm": True,
    },
    "regularization": {
        "dropout": 0.1,
        "token_dropout": 0.1,
        "cross_attention_dropout": 0.1,
        "stochastic_depth": 0.1,
    },
    "rich_pooling": {
        "components": ["mean", "attn", "max", "gated"],
    },
    "pair_readout": {
        "mode": "pair_context_gated",
        "order_aggregation": "abba_max",
        "spectral_norm": True,
    },
    "mixing": {
        "mode": "block_self",
    },
}

BEST_V3_1_SELECTION: dict[str, object] = {
    "run_id": "pair_context_gated_abba_block_sn_d256_s13",
    "seed": 13,
    "selection_metric": "test_auprc",
    "selection_value": 0.7227626490230867,
}

def build_best_v3_1() -> V3_1:
    """Build the selected V3.1 model variant."""
    return V3_1(**BEST_V3_1_CONFIG)


__all__ = ["V3_1", "BEST_V3_1_CONFIG", "BEST_V3_1_SELECTION", "build_best_v3_1"]
