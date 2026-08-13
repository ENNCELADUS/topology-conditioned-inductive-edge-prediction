"""Frozen configuration for the EgoStitch Stage-1 model (spec Sec 0 / Sec 13).

Defaults are the spec-pinned values; every field is overridable through
``model.config`` in the training YAML (validated in :func:`EgoStitchConfig.from_mapping`)
so unit tests can run tiny instances. The G5 stage gate runs at these defaults
(protocol Sec 5.0.5 instantiation).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar, cast

_ConfigT = TypeVar("_ConfigT")

_FEATURE_STANDARDIZATION_MODES = frozenset({"row_layernorm", "zscore_vfit_v1"})

CONDITIONING_MODES = frozenset({"film_logit", "pooled_adapter", "xattn_cls", "xattn_tokens"})
"""The conditioning-ladder rungs `ClassifierConfig.conditioning_mode` may name.

Public (unlike `_FEATURE_STANDARDIZATION_MODES`) because
`classifier/b0_v31.py` validates against the same set when constructed
directly, and a second literal in a second file is exactly how the two drift
apart."""


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
        w_ptr: Interior weight of the grounding-pointer CE.
        w_align: Interior weight of the Task-7 Sinkhorn alignment loss.
        w_div: Interior weight of the slot-diversity loss.
        w_rel: Interior weight of the Task-7 relational prediction loss.
        tau_adj: Temperature for logit-space slot-adjacency BCE; must be in
            ``(0, 1)`` (spec Sec 14.4.2).
        tau_div: Maximum unpenalized cosine similarity between eligible slots.
        l_gate_pos_weight: Positive-class weight for ``L_gate``. The default
            ``6.17`` is ``(1 - 0.1395) / 0.1395``, derived from the P0.2
            measured top-50 slot-recall ceiling (spec Sec 14.4.1).
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
    w_ptr: float = 0.25
    w_align: float = 0.5
    w_div: float = 0.1
    w_rel: float = 0.25
    tau_adj: float = 0.5
    tau_div: float = 0.5
    l_gate_pos_weight: float = 6.17
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
        if not 0.0 < self.tau_adj < 1.0:
            raise ValueError(f"tau_adj must be in (0, 1), got {self.tau_adj}")
        if not -1.0 <= self.tau_div <= 1.0:
            raise ValueError(f"tau_div must be in [-1, 1], got {self.tau_div}")
        if self.l_gate_pos_weight <= 0.0:
            raise ValueError(f"l_gate_pos_weight must be positive, got {self.l_gate_pos_weight}")
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
class GeneratorConfig:
    """Neighborhood-generator selection and calibration (design 2026-08-02 §8).

    Attributes:
        name: Registry key selecting the generator implementation
            (`registry.GENERATOR_REGISTRY`); ``"egostitch_imagine"`` for
            today's Tokenize-lite + Imagine + Stitch pipeline, ``"null"`` for
            the parameter-free pairwise-baseline generator,
            ``"oracle_struct"`` for the fixed-slot ground-truth-scaffold
            upper bound (`generator/oracle.py`), or ``"full_ego_oracle"``
            for the uncapped ground-truth local-graph diagnostic
            (`generator/full_oracle.py`).
        oracle_seed: Table-wide seed for ``"oracle_struct"``'s per-node hub
            subsample. Folded into every node's
            ``blake2b(f"{node_id}|{seed}|oracle")`` rng, so it selects *which*
            K neighbors a hub contributes without making the table depend on
            batching or rank. Ignored by every other generator, but part of
            `config_hash` unconditionally -- the same field must change the
            recorded hash whether or not the selected arm reads it, or two
            oracle runs with different subsamples would be indistinguishable
            in the run record.
        oracle_truth_source: Graph an oracle generator reads its topology from.
            ``"g_fit"`` keeps the inductive input boundary, so every held-out
            node gets an empty scaffold. ``"g_fit_plus_v_hold"`` attaches the
            internal validation positives as a disjoint component and is
            therefore a diagnostic-only true-oracle ceiling, never a formal
            experiment input.
        n_ground: Grounding candidates per node `n_g` (spec Sec 14.4.4;
            supersedes the internal generator's own pinned `EgoStitchConfig`
            default for this family).
        feature_standardization: Registered feature-preprocessing mode passed
            to the internal generator (``"row_layernorm"`` or
            ``"zscore_vfit_v1"``, spec Sec 13.16 D0). Part of ``config_hash``
            so a run's recorded hash changes when the transform changes.
        feature_stats_sha256: Digest pinning the exact V_fit statistics used
            by ``"zscore_vfit_v1"``. Empty until the statistics are measured
            at execution time; when non-empty must be a 64-hex lowercase
            sha256.
        tau_adj: Registered slot-adjacency temperature (spec Sec 14.4.2).
        tau_div: Registered slot-diversity threshold (spec Sec 14.4.1).
        l_gate_pos_weight: Registered positive-class weight for ``L_gate``
            (spec Sec 14.4.1).
    """

    name: str = "egostitch_imagine"
    n_ground: int = 20
    feature_standardization: str = "zscore_vfit_v1"
    feature_stats_sha256: str = ""
    tau_adj: float = 0.5
    tau_div: float = 0.5
    l_gate_pos_weight: float = 6.17
    oracle_seed: int = 0
    oracle_truth_source: str = "g_fit"

    def __post_init__(self) -> None:
        """Validate cross-field invariants.

        Raises:
            ValueError: On any non-positive dimension or out-of-range rate.
        """
        if self.n_ground <= 0:
            raise ValueError(f"n_ground must be positive, got {self.n_ground}")
        if self.oracle_seed < 0:
            raise ValueError(f"oracle_seed must be non-negative, got {self.oracle_seed}")
        if self.oracle_truth_source not in ("g_fit", "g_fit_plus_v_hold"):
            raise ValueError("oracle_truth_source must be one of ['g_fit', 'g_fit_plus_v_hold']")
        if self.oracle_truth_source != "g_fit" and self.name not in (
            "oracle_struct",
            "full_ego_oracle",
        ):
            raise ValueError(
                "oracle_truth_source='g_fit_plus_v_hold' requires name='oracle_struct' "
                "or name='full_ego_oracle'"
            )
        if not 0.0 < self.tau_adj < 1.0:
            raise ValueError(f"tau_adj must be in (0, 1), got {self.tau_adj}")
        if not -1.0 <= self.tau_div <= 1.0:
            raise ValueError(f"tau_div must be in [-1, 1], got {self.tau_div}")
        if self.l_gate_pos_weight <= 0.0:
            raise ValueError(f"l_gate_pos_weight must be positive, got {self.l_gate_pos_weight}")
        if self.feature_standardization not in _FEATURE_STANDARDIZATION_MODES:
            raise ValueError(
                f"feature_standardization must be one of {sorted(_FEATURE_STANDARDIZATION_MODES)}"
            )
        if self.feature_stats_sha256 and (
            len(self.feature_stats_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.feature_stats_sha256)
        ):
            raise ValueError("feature_stats_sha256 must be empty or a 64-hex lowercase sha256")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> GeneratorConfig:
        """Build a `generator:` config section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(
            cls,
            mapping,
            label="generator",
            string_fields=frozenset(
                {
                    "name",
                    "feature_standardization",
                    "feature_stats_sha256",
                    "oracle_truth_source",
                }
            ),
        )


@dataclass(frozen=True)
class EncoderConfig:
    """Graph-encoder selection and sizing (design 2026-08-02 §8).

    Deliberately carries no ``d_model``: the encoder's output width must
    match what the classifier attends over, and `ClassifierConfig.d_model` is
    the single source of truth for that width (design task pin) -- the
    composite passes it into the encoder's constructor at build time.

    Attributes:
        name: Registry key selecting the encoder implementation
            (`registry.ENCODER_REGISTRY`); ``"ste_typed"`` for the typed
            message-passing port of today's `STEncoder`, ``"grit_gmt"`` for
            the official-GRIT + official-PMA encoder
            (`encoder/grit_gmt.py`).
        dim: Hidden width of the encoder's internal stack (was ``ste_dim``).
            Shared across encoders on purpose: it is the same knob, and
            ``grit_gmt`` is simply run at a narrower default (``96``) than
            ``ste_typed``'s ``128`` because its per-layer cost is higher.
        layers: Stack depth (was ``ste_layers``).
        w_rel: Interior ``L_recon`` weight for the discarded-at-inference
            relational head; ``0`` defines the formal ``no_l_rel`` arm and
            omits the relational head entirely.
        rrwp_k: ``grit_gmt`` only -- relative random-walk positional-encoding
            length ``K``. The edge channel carries ``[I, P, ..., P^(K-1)]``,
            so ``K`` bounds the walk radius the attention can distinguish.
        n_heads: ``grit_gmt`` only -- GRIT attention heads. Must divide `dim`.
        seeds: ``grit_gmt`` only -- PMA seed count; the encoder appends this
            many learned graph-level summary tokens to its node tokens.
    """

    name: str = "ste_typed"
    dim: int = 128
    layers: int = 3
    w_rel: float = 0.25
    rrwp_k: int = 8
    n_heads: int = 8
    seeds: int = 4

    def __post_init__(self) -> None:
        """Validate cross-field invariants.

        Raises:
            ValueError: On any non-positive dimension or a negative weight.
        """
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.layers <= 0:
            raise ValueError(f"layers must be positive, got {self.layers}")
        if self.w_rel < 0.0:
            raise ValueError(f"w_rel must be non-negative, got {self.w_rel}")
        for name in ("rrwp_k", "n_heads", "seeds"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> EncoderConfig:
        """Build an `encoder:` config section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(cls, mapping, label="encoder", string_fields=frozenset({"name"}))


@dataclass(frozen=True)
class ClassifierConfig:
    """Pairwise-classifier selection and pair-trunk sizing (design 2026-08-02 §8).

    Attributes:
        name: Registry key selecting the classifier implementation
            (`registry.CLASSIFIER_REGISTRY`); ``"b0_v31"`` for the mature
            B0-V3.1 baseline plus its conditioning socket.
        d_model: Pair-trunk hidden width. Also the width the composite passes
            into the encoder's constructor (see `EncoderConfig`).
        encoder_layers: `SiameseEncoder` depth (per-item token encoder).
        cross_attn_layers: `ConditionedPairCrossAttention` depth.
        n_heads: Attention heads for the item encoder and pair trunk.
        n_inj: Trailing trunk layers receiving gated topo injection.
        xattn_heads: Gated cross-attention heads (topo pathway).
        p_topo: Training-time branch-dropout rate for the topo pathway.
        permanent_null: Permanently-forced head-null condition (``"none"`` or
            ``"all_head"``).
        conditioning_ema_decay: Decay for the synchronized conditioning-center
            EMA stored in rev-3.1 checkpoints (spec Sec 14.4.2).
        conditioning_mode: Which rung of the conditioning ladder the classifier
            builds -- exactly one, never several
            (`classifier/b0_v31.py`). In increasing capacity:
            ``"film_logit"`` (a scalar scale/shift on the fused pair logit,
            from the pooled graph summary), ``"pooled_adapter"`` (a low-rank
            pooled vector added into the trunk's token streams),
            ``"xattn_cls"`` (today's gated cross-attention from the trunk's
            cls token onto the graph tokens -- the default, and the only mode
            existing checkpoints carry), ``"xattn_tokens"`` (the same gated
            cross-attention, with every valid non-cls pair token as a query).
            Every mode is zero-initialized, so at initialization all four are
            bit-identical to the unconditioned path.
        node_factor_dim: Width `dim` of the `NodeFactorBottleneck` low-rank
            node-factor path (`classifier/node_factor.py`, B1 KD plan). ``0``
            (default) builds no node factor at all -- DDP unused-parameter
            safety, the same conditional-build convention as `film_logit`.
            Independent of `conditioning_mode`: the node factor is a
            content-path residual, never gated by topo conditioning.
    """

    name: str = "b0_v31"
    d_model: int = 512
    encoder_layers: int = 3
    cross_attn_layers: int = 3
    n_heads: int = 8
    n_inj: int = 1
    xattn_heads: int = 8
    p_topo: float = 0.15
    permanent_null: str = "none"
    conditioning_ema_decay: float = 0.99
    conditioning_mode: str = "xattn_cls"
    node_factor_dim: int = 0

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
            "xattn_heads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 1 <= self.n_inj <= self.cross_attn_layers:
            raise ValueError(f"n_inj must be in [1, cross_attn_layers], got {self.n_inj}")
        if not 0.0 <= self.p_topo <= 1.0:
            raise ValueError(f"p_topo must be in [0, 1], got {self.p_topo}")
        if self.permanent_null not in ("none", "all_head"):
            raise ValueError(
                f"permanent_null must be one of 'none' or 'all_head', got {self.permanent_null!r}"
            )
        if not 0.0 < self.conditioning_ema_decay < 1.0:
            raise ValueError(
                f"conditioning_ema_decay must be in (0, 1), got {self.conditioning_ema_decay}"
            )
        if self.conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {sorted(CONDITIONING_MODES)}, "
                f"got {self.conditioning_mode!r}"
            )
        if self.node_factor_dim < 0:
            raise ValueError(f"node_factor_dim must be non-negative, got {self.node_factor_dim}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ClassifierConfig:
        """Build a `classifier:` config section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(
            cls,
            mapping,
            label="classifier",
            string_fields=frozenset({"name", "permanent_null", "conditioning_mode"}),
        )


@dataclass(frozen=True)
class DistillConfig:
    """B1 training-time topology-distillation knobs (fourth optional `E2EConfig` section).

    All-zero-weight defaults keep every existing config valid with no
    ``distill:`` section at all (`E2EConfig.from_mapping`'s absent-section ->
    defaults rule) and stay out of the pinned ``training:`` gate
    (`train_egostitch.py`). Exactly one arm group's weight(s) may be nonzero
    at a time -- ``kd_control`` (`w_label`), ``kd_d1`` (`w_logit`), ``kd_d2``
    (`w_rank` and `w_dist` together), ``kd_d3`` (`w_gram`) -- so a config can
    never straddle two KD mechanisms;
    `train_egostitch._e2e_arm_name_from_config` reads this same pattern to
    name the run.

    Attributes:
        targets_path: Path to the dumped teacher-target artifact
            (`src/distill/teacher_targets.py`, Wave 2). Required whenever any
            weight below is nonzero.
        targets_sha256: Digest pinning the exact artifact `targets_path`
            must resolve to.
        w_label: ``kd_control`` weight -- BCE against the KD stream's own
            ``pair_label`` (the matched control: same stream as every other
            arm, label supervision instead of a teacher signal).
        w_logit: ``kd_d1`` weight -- pointwise logit KD against the
            teacher's soft score (GLNN).
        w_rank: ``kd_d2`` weight -- margin-rank loss over per-anchor teacher
            rows (LLP_R, LLP ICML'23). Paired with `w_dist`.
        w_dist: ``kd_d2`` weight -- temperature-KL over per-anchor teacher
            rows (LLP_D). Paired with `w_rank`.
        w_gram: ``kd_d3`` weight -- pair-space cosine-Gram matching against
            the teacher's pooled embeddings (Graph2Feat/CAZI family).
        temperature: Softmax/KL temperature for `w_dist` (LLP reference pins 1.0).
        margin: Margin for the `w_rank` pairwise ranking loss.
        anchors_per_step: KD anchor groups drawn per optimizer microstep.
    """

    targets_path: str = ""
    targets_sha256: str = ""
    w_label: float = 0.0
    w_logit: float = 0.0
    w_rank: float = 0.0
    w_dist: float = 0.0
    w_gram: float = 0.0
    temperature: float = 1.0
    margin: float = 0.1
    anchors_per_step: int = 2

    def __post_init__(self) -> None:
        """Validate weight signs/ranges and the single-arm-group pattern.

        Raises:
            ValueError: On a negative weight, a non-positive
                `temperature`/`margin`, a non-positive `anchors_per_step`, or
                a nonzero-weight pattern outside the five legal arm groups.
        """
        for name in ("w_label", "w_logit", "w_rank", "w_dist", "w_gram"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.anchors_per_step < 1:
            raise ValueError(f"anchors_per_step must be >= 1, got {self.anchors_per_step}")
        nonzero = frozenset(
            name
            for name in ("w_label", "w_logit", "w_rank", "w_dist", "w_gram")
            if float(getattr(self, name)) > 0.0
        )
        legal_patterns: tuple[frozenset[str], ...] = (
            frozenset(),
            frozenset({"w_label"}),
            frozenset({"w_logit"}),
            frozenset({"w_rank", "w_dist"}),
            frozenset({"w_gram"}),
        )
        if nonzero not in legal_patterns:
            raise ValueError(
                "distill weights must follow exactly one arm group -- all zero, only "
                "w_label (kd_control), only w_logit (kd_d1), w_rank and w_dist together "
                f"(kd_d2), or only w_gram (kd_d3); got nonzero weights {sorted(nonzero)}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> DistillConfig:
        """Build a `distill:` config section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        return _from_mapping(
            cls,
            mapping,
            label="distill",
            string_fields=frozenset({"targets_path", "targets_sha256"}),
        )


_E2E_TOP_LEVEL_KEYS = frozenset({"generator", "encoder", "classifier", "distill"})

# Every field name that lived directly on the pre-P3 flat `E2EConfig`
# (design 2026-08-02 §8/§12 P3). A mapping presenting any of these at the top
# level is the dead flat schema, not an unrecognized future key -- callers get
# a message pointing at the nested shape instead of a generic "unknown key"
# error that would leave them guessing where the field went.
_E2E_FLAT_LEGACY_KEYS = frozenset(
    {
        "d_model",
        "encoder_layers",
        "cross_attn_layers",
        "n_heads",
        "n_inj",
        "ste_dim",
        "ste_layers",
        "xattn_heads",
        "p_topo",
        "permanent_null",
        "n_ground",
        "tau_adj",
        "tau_div",
        "l_gate_pos_weight",
        "conditioning_ema_decay",
        "w_rel",
        "feature_standardization",
        "feature_stats_sha256",
    }
)


def _as_section_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Extract and type-check one nested config section."""
    raw = mapping.get(key, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"model.config.{key} must be a mapping, got {raw!r}")
    return raw


@dataclass(frozen=True)
class E2EConfig:
    """Rev-4 end-to-end three-component hyperparameters (design 2026-08-02 §8).

    Nests the pair-trunk/classifier, graph-encoder and neighborhood-generator
    hyperparameters under their own sub-configs -- ``generator:``,
    ``encoder:``, ``classifier:`` -- each independently strict about unknown
    keys (`GeneratorConfig.from_mapping`, `EncoderConfig.from_mapping`,
    `ClassifierConfig.from_mapping`). There is no backward-compatible flat
    path: `from_mapping` rejects the pre-P3 flat schema loudly rather than
    silently reinterpreting it (design §12 P3).

    Attributes:
        generator: Neighborhood-generator selection and calibration.
        encoder: Graph-encoder selection and sizing.
        classifier: Pairwise-classifier selection and pair-trunk sizing.
        distill: B1 training-time topology-distillation knobs (optional;
            absent ``distill:`` section -> all-zero-weight defaults, so every
            pre-B1 config stays valid unchanged).
    """

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    def __post_init__(self) -> None:
        """Validate the cross-section KD-architecture requirement.

        Raises:
            ValueError: If any `distill` weight is nonzero while
                `distill.targets_path` is empty or `classifier.node_factor_dim`
                is `0` -- a KD loss with no teacher artifact or no node-factor
                path to train would silently be a no-op.
        """
        distill_active = any(
            getattr(self.distill, name) > 0.0
            for name in ("w_label", "w_logit", "w_rank", "w_dist", "w_gram")
        )
        node_factor_configured = self.classifier.node_factor_dim > 0
        if distill_active and not (self.distill.targets_path and node_factor_configured):
            raise ValueError(
                "a nonzero distill weight requires both distill.targets_path and "
                "classifier.node_factor_dim > 0"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> E2EConfig:
        """Build a config from a YAML ``model.config`` mapping.

        Args:
            mapping: Must carry only ``generator``/``encoder``/``classifier``/
                ``distill`` keys, each itself a mapping validated by that
                component's own `from_mapping` (unknown keys rejected per
                component, the train_b0 strict-config convention).

        Returns:
            The validated `E2EConfig`.

        Raises:
            ValueError: If `mapping` carries the pre-P3 flat schema (any
                legacy top-level field name), any other unknown top-level
                key, or an invalid value inside any section.
        """
        unknown = sorted(set(mapping) - _E2E_TOP_LEVEL_KEYS)
        if unknown:
            flat_hit = sorted(set(unknown) & _E2E_FLAT_LEGACY_KEYS)
            if flat_hit:
                raise ValueError(
                    "E2EConfig no longer accepts the flat model.config schema "
                    f"(found top-level {flat_hit}); nest fields under "
                    "'generator:'/'encoder:'/'classifier:' instead "
                    "(design 2026-08-02 §8)"
                )
            raise ValueError(f"unknown E2E config keys: {unknown}")
        return cls(
            generator=GeneratorConfig.from_mapping(_as_section_mapping(mapping, "generator")),
            encoder=EncoderConfig.from_mapping(_as_section_mapping(mapping, "encoder")),
            classifier=ClassifierConfig.from_mapping(_as_section_mapping(mapping, "classifier")),
            distill=DistillConfig.from_mapping(_as_section_mapping(mapping, "distill")),
        )
