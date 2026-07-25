"""Frozen configuration for the EgoStitch Stage-1 model (spec Sec 0 / Sec 13).

Defaults are the spec-pinned values; every field is overridable through
``model.config`` in the training YAML (validated in :func:`EgoStitchConfig.from_mapping`)
so unit tests can run tiny instances. The G5 stage gate runs at these defaults
(protocol Sec 5.0.5 instantiation).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, TypeVar, cast

_ConfigT = TypeVar("_ConfigT")


def _from_mapping(
    cls: type[_ConfigT],
    mapping: Mapping[str, object],
    *,
    label: str,
    string_fields: frozenset[str] = frozenset(),
) -> _ConfigT:
    """Build one strict numeric dataclass config from a YAML mapping."""
    config_fields = {field.name: field for field in fields(cast(Any, cls))}
    unknown = sorted(set(mapping) - config_fields.keys())
    if unknown:
        raise ValueError(f"unknown {label} config keys: {unknown}")

    kwargs: dict[str, object] = {}
    for name, raw in mapping.items():
        field = config_fields[name]
        if name in string_fields:
            if not isinstance(raw, str):
                raise ValueError(f"{name} must be a string, got {raw!r}")
            kwargs[name] = raw
        elif field.type == "int":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{name} must be an int, got {raw!r}")
            kwargs[name] = raw
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be a number, got {raw!r}")
            kwargs[name] = float(raw)
    return cls(**kwargs)


@dataclass(frozen=True)
class EgoStitchConfig:
    """Pinned Stage-1 hyperparameters.

    Attributes:
        input_dim: Frozen feature dim ``d`` (F0 mean-pool, spec Sec 9.2).
        d_p: Projected feature space (set-decoder target space).
        d_z: Encoder embedding dim (``e_u``; the codebook dim it replaces in
            Stage 1, spec Sec 13.2).
        d_h: Decoder hidden dim.
        slots: Neighbor slots per node ``K``.
        m_max: Max multiplicity per slot.
        n_ground: Grounding candidates per node ``n_g``.
        decoder_layers: Imagine decoder depth.
        n_heads: Decoder attention heads (spec Sec 13.7).
        sinkhorn_eps: Entropic regularizer ``eps`` (spec Sec 3).
        sinkhorn_iters: Fixed Sinkhorn iteration count (determinism).
        sinkhorn_tau: Unbalanced-OT KL relaxation strength ``tau_OT``.
        denoise_fraction: Fraction of training nodes receiving denoising
            queries (spec Sec 2).
        denoise_sigma: Noise scale of denoising query initializations.
        null_dropout: Probability of each conditioning-dropout null
            (``∅_content`` / ``∅_all``, disjoint draws).
        lambda_recon: Master loss weight of ``L_recon``.
        lambda_real: Master loss weight of ``L_real``.
        lambda_ssl: Master loss weight of ``L_ssl``.
        w_feat: Interior ``L_recon`` weight of the Hungarian Huber feature loss.
        w_exist: Interior weight of the existence BCE.
        w_mult: Interior weight of the multiplicity NLL.
        w_deg: Interior weight of the lognormal degree NLL.
        w_slotadj: Interior weight of the slot-adjacency group BCE.
        w_gate: Interior weight of the grounding-gate BCE.
        w_egostat: Interior ``L_real`` weight of the ego-stat energy distance
            (Stage-1 renormalized, spec Sec 13.5).
        w_gin: Interior ``L_real`` weight of the random-GIN energy distance.
        gin_hidden: Hidden width of the frozen random GIN (spec Sec 13.6).
        gin_layers: Depth of the frozen random GIN.
        ssl_noise_sigma: Feature-noise scale of the SSL consistency term.
    """

    input_dim: int = 1536
    d_p: int = 256
    d_z: int = 64
    d_h: int = 256
    slots: int = 16
    m_max: int = 32
    n_ground: int = 20
    decoder_layers: int = 3
    n_heads: int = 8
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 20
    sinkhorn_tau: float = 1.0
    denoise_fraction: float = 0.25
    denoise_sigma: float = 0.1
    null_dropout: float = 0.1
    lambda_recon: float = 1.0
    lambda_real: float = 0.5
    lambda_ssl: float = 0.1
    w_feat: float = 1.0
    w_exist: float = 0.5
    w_mult: float = 0.25
    w_deg: float = 0.5
    w_slotadj: float = 0.5
    w_gate: float = 0.25
    w_egostat: float = 2.0 / 3.0
    w_gin: float = 1.0 / 3.0
    gin_hidden: int = 64
    gin_layers: int = 3
    ssl_noise_sigma: float = 0.05

    def __post_init__(self) -> None:
        """Validate cross-field invariants.

        Raises:
            ValueError: On any non-positive dimension or out-of-range rate.
        """
        for name in (
            "input_dim",
            "d_p",
            "d_z",
            "d_h",
            "slots",
            "m_max",
            "n_ground",
            "decoder_layers",
            "n_heads",
            "sinkhorn_iters",
            "gin_hidden",
            "gin_layers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_h % self.n_heads != 0:
            raise ValueError(f"d_h ({self.d_h}) must be divisible by n_heads ({self.n_heads})")
        for name in ("sinkhorn_eps", "sinkhorn_tau", "denoise_sigma", "ssl_noise_sigma"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("denoise_fraction", "null_dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> EgoStitchConfig:
        """Build a config from a YAML ``model.config`` mapping.

        Args:
            mapping: Field-name -> value overrides; unknown keys are rejected
                (the train_b0 strict-config convention).

        Returns:
            The validated `EgoStitchConfig`.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(cls, mapping, label="EgoStitch")


@dataclass(frozen=True)
class E2EConfig:
    """Rev-3.0 end-to-end conditioned-encoder hyperparameters (design rev 3).

    The internal Stage-1 generator keeps its own pinned `EgoStitchConfig`
    defaults (spec Sec 13); these fields size only the pair-encoder trunk and
    its topo/content conditioning pathways (design rev 3 Sec 3.4-3.5) -- with
    one exception: `n_ground` supersedes the generator's own pinned value for
    this family (spec Sec 14.4.4), since grounding-pool size is a per-arm
    dial (`full` etc. use the rev-3.1 default 50; the `cosine_pool` ablation
    arm pins 20).

    Attributes:
        d_model: Pair-trunk hidden width.
        encoder_layers: `SiameseEncoder` depth (per-item token encoder).
        cross_attn_layers: `ConditionedPairCrossAttention` depth.
        n_heads: Attention heads for the item encoder and pair trunk.
        n_inj: Trailing trunk layers receiving gated topo/content injection.
        ste_dim: Stitched-topology encoder hidden width.
        ste_layers: Stitched-topology encoder depth.
        xattn_heads: Gated cross-attention heads (topo/content pathways).
        p_topo: Training-time branch-dropout rate for the topo pathway.
        p_cont: Training-time branch-dropout rate for the content pathway.
        n_ground: Grounding candidates per node `n_g` (spec Sec 14.4.4;
            supersedes the internal generator's own pinned `EgoStitchConfig`
            default for this family).
    """

    d_model: int = 512
    encoder_layers: int = 3
    cross_attn_layers: int = 3
    n_heads: int = 8
    n_inj: int = 1
    ste_dim: int = 128
    ste_layers: int = 3
    xattn_heads: int = 8
    p_topo: float = 0.15
    p_cont: float = 0.15
    permanent_null: str = "none"
    n_ground: int = 50

    def __post_init__(self) -> None:
        """Validate cross-field invariants.

        Raises:
            ValueError: On any non-positive dimension or out-of-range rate.
        """
        for name in (
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
            "n_inj",
            "ste_dim",
            "ste_layers",
            "xattn_heads",
            "n_ground",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 1 <= self.n_inj <= self.cross_attn_layers:
            raise ValueError(f"n_inj must be in [1, cross_attn_layers], got {self.n_inj}")
        for name in ("p_topo", "p_cont"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.permanent_null not in ("none", "all_head", "content_head"):
            raise ValueError(
                "permanent_null must be one of 'none', 'all_head', or 'content_head', "
                f"got {self.permanent_null!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> E2EConfig:
        """Build a config from a YAML ``model.config`` mapping.

        Args:
            mapping: Field-name -> value overrides; unknown keys are rejected
                (the train_b0 strict-config convention).

        Returns:
            The validated `E2EConfig`.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(cls, mapping, label="E2E", string_fields=frozenset({"permanent_null"}))
