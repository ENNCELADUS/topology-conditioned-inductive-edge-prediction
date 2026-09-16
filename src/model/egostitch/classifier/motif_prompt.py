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
    N_SLOTS,
    count_statistics,
)
from src.model.egostitch.encoder.grit_gmt import dense_rrwp

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
