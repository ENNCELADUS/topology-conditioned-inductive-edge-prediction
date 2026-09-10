"""Topology-supervised prefix tuning on a frozen V3.1 trunk.

Design: docs/superpowers/specs/2026-09-10-prefix-tuning-frozen-trunk-design.md.
`PrefixGenerator` owns every trainable parameter (static prefix tokens, the
pair generator, the gates); `PrefixCrossAttentionLayer` re-drives one frozen
`CrossAttentionLayer` with a separately-softmaxed, tanh-gated prefix branch;
`V3_1Prefix` wraps a frozen `V3_1` whose pair trunk has bidirectional
cross-attention layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.model.egostitch.classifier.layers import inner_token_mask, masked_mean

CONDITIONING_MODES = ("static", "pair")
SITES = 3  # A<-B, B<-A, CLS


@dataclass(frozen=True)
class PrefixConfig:
    """The ``model.config.prefix`` block.

    Attributes:
        tokens: Prefix length per layer.
        rank: Rank of the slot-specific pair shift.
        conditioning: ``"static"`` or ``"pair"``.
        bottleneck: Width of the pair condition ``z``.
        base_checkpoint: Path of the frozen base checkpoint (provenance only).
        base_checkpoint_sha256: SHA-256 of that file (provenance only, never verified).
    """

    tokens: int = 16
    rank: int = 8
    conditioning: str = "static"
    bottleneck: int = 128
    base_checkpoint: str = ""
    base_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: On a non-positive size or an unknown conditioning mode.
        """
        if self.tokens <= 0 or self.rank <= 0 or self.bottleneck <= 0:
            raise ValueError("prefix.tokens, prefix.rank, and prefix.bottleneck must be positive")
        if self.conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"prefix.conditioning must be one of {CONDITIONING_MODES}, "
                f"got {self.conditioning!r}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PrefixConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.prefix``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown prefix keys: {unknown}")
        sha = raw.get("base_checkpoint_sha256")
        return cls(
            tokens=int(cast(int, raw.get("tokens", 16))),
            rank=int(cast(int, raw.get("rank", 8))),
            conditioning=str(raw.get("conditioning", "static")),
            bottleneck=int(cast(int, raw.get("bottleneck", 128))),
            base_checkpoint=str(raw.get("base_checkpoint", "")),
            base_checkpoint_sha256=None if sha is None else str(sha),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))


class PrefixGenerator(nn.Module):
    """Every trainable parameter of the prefix arm.

    Static prefix ``p0[l]`` of shape ``(tokens, d_model)`` per layer; for
    ``conditioning == "pair"`` a shared condition ``z = GELU(Linear(LN(c)))``
    from the symmetric pair features ``c = [e_u + e_v; |e_u - e_v|; e_u * e_v]``
    and, per layer, a slot-specific low-rank shift
    ``reshape(shift_in[l](z), tokens, rank) @ shift_out[l]``. A shift shared
    across slots would cancel in the separate prefix softmax (spec §5.1), so
    the shift is per slot by construction.

    Gates ``(n_layers, SITES, n_heads)`` start at zero: the only zero factor.
    ``shift_out`` is small but nonzero so the generator receives gradient as
    soon as a gate opens.
    """

    z_sum: torch.Tensor
    z_count: torch.Tensor

    def __init__(self, d_model: int, n_layers: int, n_heads: int, cfg: PrefixConfig) -> None:
        """Build the parameters.

        Args:
            d_model: Trunk width.
            n_layers: Number of frozen cross-attention layers.
            n_heads: Heads per attention site.
            cfg: The prefix block.
        """
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p0 = nn.ParameterList(
            nn.Parameter(torch.randn(cfg.tokens, d_model) * 0.02) for _ in range(n_layers)
        )
        self.gates = nn.Parameter(torch.zeros(n_layers, SITES, n_heads))
        if cfg.conditioning == "pair":
            self.cond_norm = nn.LayerNorm(3 * d_model)
            self.cond_proj = nn.Linear(3 * d_model, cfg.bottleneck)
            self.shift_in = nn.ModuleList(
                nn.Linear(cfg.bottleneck, cfg.tokens * cfg.rank) for _ in range(n_layers)
            )
            self.shift_out = nn.ParameterList(
                nn.Parameter(torch.randn(cfg.rank, d_model) * 0.02) for _ in range(n_layers)
            )
        self.register_buffer("z_sum", torch.zeros(cfg.bottleneck))
        self.register_buffer("z_count", torch.zeros(()))

    @property
    def z_mean(self) -> torch.Tensor:
        """Running mean of ``z`` over training forwards (zeros before any)."""
        return self.z_sum / self.z_count.clamp_min(1.0)

    def condition(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return ``z`` of shape ``(B, bottleneck)``, or ``None`` when static.

        Args:
            encoded_a: Frozen encoder output for item A ``(B, L_a, d_model)``.
            encoded_b: Frozen encoder output for item B ``(B, L_b, d_model)``.
            mask_a: Padding mask for A (True = PAD) or ``None``.
            mask_b: Padding mask for B (True = PAD) or ``None``.

        Returns:
            The symmetric pair condition, or ``None`` for the static prefix.
        """
        if self.cfg.conditioning != "pair":
            return None
        e_a = masked_mean(encoded_a, inner_token_mask(x=encoded_a, padding_mask=mask_a))
        e_b = masked_mean(encoded_b, inner_token_mask(x=encoded_b, padding_mask=mask_b))
        c = torch.cat([e_a + e_b, (e_a - e_b).abs(), e_a * e_b], dim=-1)
        z = F.gelu(self.cond_proj(self.cond_norm(c)))
        if self.training:
            with torch.no_grad():
                self.z_sum += z.detach().float().sum(dim=0)
                self.z_count += float(z.size(0))
        return z

    def prefix(self, layer_index: int, z: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """Return the prefix tokens for one layer, ``(B, tokens, d_model)``.

        Args:
            layer_index: Which frozen layer.
            z: The pair condition from `condition`, or ``None``.
            batch_size: Rows to expand a static prefix to.

        Returns:
            The (possibly pair-shifted) prefix tokens.
        """
        p0: torch.Tensor = self.p0[layer_index]
        if z is None:
            return p0.unsqueeze(0).expand(batch_size, -1, -1)
        shift_in_layer = cast(nn.Linear, self.shift_in[layer_index])
        coeff = shift_in_layer(z).view(z.size(0), self.cfg.tokens, self.cfg.rank)
        shift_out: torch.Tensor = self.shift_out[layer_index]
        delta: torch.Tensor = coeff @ shift_out
        return p0.unsqueeze(0) + delta

    def gate(self, layer_index: int, site: int) -> torch.Tensor:
        """Return the raw (pre-tanh) per-head gate for one site."""
        return self.gates[layer_index, site]

    @torch.no_grad()
    def init_static_from_tokens(self, tokens: torch.Tensor, seed: int) -> None:
        """Initialise every ``p0[l]`` from sampled real token states.

        Args:
            tokens: Candidate rows ``(N, d_model)``, ``N >= cfg.tokens``.
            seed: Draw seed (identical across ranks; DDP's construction-time
                broadcast makes rank 0's draw authoritative anyway).

        Raises:
            ValueError: If fewer than ``cfg.tokens`` rows are supplied.
        """
        if tokens.size(0) < self.cfg.tokens:
            raise ValueError("init_static_from_tokens needs at least cfg.tokens rows")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for layer_index in range(self.n_layers):
            idx = torch.randperm(tokens.size(0), generator=gen)[: self.cfg.tokens]
            self.p0[layer_index].copy_(tokens[idx].to(self.p0[layer_index].dtype))
