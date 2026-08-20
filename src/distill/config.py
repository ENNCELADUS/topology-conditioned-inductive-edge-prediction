"""`DistillConfig`: the B1 training-time topology-distillation knobs.

Consumed by the simple B0-protocol trainer (`src.train_b0`) as the optional
top-level ``distill:`` config section. Exactly one arm group's weight(s) may
be nonzero at a time -- ``kd_control`` (`w_label`), ``kd_d1`` (`w_logit`),
``kd_d2`` (`w_rank` and `w_dist` together), ``kd_d3`` (`w_gram`), ``kd_d4``
(`w_align`), ``kd_d5`` (`w_residual`) -- with two deliberate exceptions,
``kd_d6`` (`w_rank`, `w_dist`, and `w_gram` together, the interaction test of
`kd_d2` + `kd_d3`) and ``kd_d9`` (`w_seed`, `w_geom`, and `w_kl` together,
the pair-latent-generation arm's inseparable reconstruction + KL objective)
-- so a config can never straddle two KD mechanisms outside those named
combinations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

_WEIGHT_NAMES = (
    "w_label",
    "w_logit",
    "w_rank",
    "w_dist",
    "w_gram",
    "w_align",
    "w_residual",
    "w_seed",
    "w_geom",
    "w_kl",
)
_INT_FIELDS = ("anchors_per_step", "kl_warmup_steps", "joint_start_epoch")


@dataclass(frozen=True)
class DistillConfig:
    """B1 training-time topology-distillation knobs.

    Attributes:
        targets_path: Path to the dumped teacher-target artifact
            (`src/distill/teacher_targets.py`). Required whenever any weight
            below is nonzero.
        w_label: ``kd_control`` weight -- BCE against the KD stream's own
            ``pair_label`` (the matched control: same stream as every other
            arm, label supervision instead of a teacher signal).
        w_logit: ``kd_d1`` weight -- pointwise logit KD against the teacher's
            soft score (GLNN).
        w_rank: ``kd_d2`` weight -- margin-rank loss over per-anchor teacher
            rows (LLP_R, LLP ICML'23). Paired with `w_dist`.
        w_dist: ``kd_d2`` weight -- temperature-KL over per-anchor teacher
            rows (LLP_D). Paired with `w_rank`.
        w_gram: ``kd_d3`` weight -- pair-space cosine-Gram matching against
            the teacher's pooled embeddings (Graph2Feat/CAZI family).
        w_align: ``kd_d4`` weight -- cosine alignment of the model's
            projected pair feature to the symmetrized teacher pooled
            embedding (SA-MLP/SALE-MLP representation-alignment family).
        w_residual: ``kd_d5`` weight -- Huber loss between the student's
            node-factor residual and the teacher's beyond-content residual
            (`teacher_logit - content_logit`).
        w_seed: ``kd_d9`` weight -- per-slot cosine reconstruction of the
            teacher's symmetrized PMA seed tokens by the posterior-path
            generated seeds (pair-latent generation, D9 design).
        w_geom: ``kd_d9`` weight -- relative Frobenius match of the raw
            within-pair seed Gram (slot norms + inter-slot geometry).
        w_kl: ``kd_d9`` weight -- free-bits KL(q || p) tying the recognition
            posterior to the deployed prior; warmed up over `kl_warmup_steps`.
        temperature: Softmax/KL temperature for `w_dist` (LLP reference pins 1.0).
        margin: Margin for the `w_rank` pairwise ranking loss.
        anchors_per_step: KD anchor groups drawn per optimizer step per rank.
        kl_warmup_steps: ``kd_d9`` -- optimizer steps over which the `w_kl`
            weight ramps linearly 0 -> 1 (0 disables the ramp).
        joint_start_epoch: ``kd_d9`` -- first epoch of stage-2 joint
            fine-tuning: the fusion-boundary stop-gradient drops and the
            generator param group switches to `gen_lr_scale` x base LR
            (0 = joint from the start).
        gen_lr_scale: ``kd_d9`` -- stage-2 LR multiplier for the generator
            param group.
        arm_label: Explicit arm-name override, empty by default (the weight
            pattern names the arm). `kd_d7a` reuses `kd_d2`'s exact pattern
            (`w_rank` + `w_dist`) against a heuristic-teacher targets
            artifact, so the weight pattern alone cannot tell the two apart.
    """

    targets_path: str = ""
    w_label: float = 0.0
    w_logit: float = 0.0
    w_rank: float = 0.0
    w_dist: float = 0.0
    w_gram: float = 0.0
    w_align: float = 0.0
    w_residual: float = 0.0
    w_seed: float = 0.0
    w_geom: float = 0.0
    w_kl: float = 0.0
    temperature: float = 1.0
    margin: float = 0.1
    anchors_per_step: int = 2
    kl_warmup_steps: int = 2000
    joint_start_epoch: int = 0
    gen_lr_scale: float = 0.1
    arm_label: str = ""

    def __post_init__(self) -> None:
        """Validate weight signs/ranges and the single-arm-group pattern.

        Raises:
            ValueError: On a negative weight, a non-positive
                `temperature`/`margin`, a non-positive `anchors_per_step`, a
                nonzero-weight pattern outside the legal arm groups, or a
                nonzero weight without a `targets_path`.
        """
        for name in _WEIGHT_NAMES:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.anchors_per_step < 1:
            raise ValueError(f"anchors_per_step must be >= 1, got {self.anchors_per_step}")
        if self.kl_warmup_steps < 0:
            raise ValueError(f"kl_warmup_steps must be >= 0, got {self.kl_warmup_steps}")
        if self.joint_start_epoch < 0:
            raise ValueError(f"joint_start_epoch must be >= 0, got {self.joint_start_epoch}")
        if self.gen_lr_scale <= 0.0:
            raise ValueError(f"gen_lr_scale must be positive, got {self.gen_lr_scale}")
        nonzero = frozenset(name for name in _WEIGHT_NAMES if float(getattr(self, name)) > 0.0)
        legal_patterns: tuple[frozenset[str], ...] = (
            frozenset(),
            frozenset({"w_label"}),
            frozenset({"w_logit"}),
            frozenset({"w_rank", "w_dist"}),
            frozenset({"w_gram"}),
            frozenset({"w_align"}),
            frozenset({"w_residual"}),
            frozenset({"w_rank", "w_dist", "w_gram"}),
            frozenset({"w_seed", "w_geom", "w_kl"}),
        )
        if nonzero not in legal_patterns:
            raise ValueError(
                "distill weights must follow exactly one arm group -- all zero, only "
                "w_label (kd_control), only w_logit (kd_d1), w_rank and w_dist together "
                "(kd_d2), only w_gram (kd_d3), only w_align (kd_d4), only w_residual "
                "(kd_d5), w_rank, w_dist, and w_gram together (kd_d6), or w_seed, "
                "w_geom, and w_kl together (kd_d9); got nonzero "
                f"weights {sorted(nonzero)}"
            )
        if nonzero and not self.targets_path:
            raise ValueError("distill.targets_path is required when any weight is nonzero")
        if self.arm_label and nonzero != frozenset({"w_rank", "w_dist"}):
            raise ValueError(
                "distill.arm_label exists only to disambiguate the kd_d2 weight pattern "
                "(w_rank and w_dist) by teacher provenance; any other pattern would let a "
                f"config publish under a false arm name (got nonzero weights {sorted(nonzero)})"
            )

    @property
    def active(self) -> bool:
        """Whether any distillation weight is nonzero."""
        return any(float(getattr(self, name)) > 0.0 for name in _WEIGHT_NAMES)

    @property
    def arm(self) -> str:
        """The KD arm this weight pattern names (``none`` when inactive).

        `arm_label`, when set on an active config, overrides the mapped name
        -- the only way to distinguish `kd_d7a` from `kd_d2`, whose weight
        patterns are identical.
        """
        nonzero = frozenset(name for name in _WEIGHT_NAMES if float(getattr(self, name)) > 0.0)
        mapped = {
            frozenset(): "none",
            frozenset({"w_label"}): "kd_control",
            frozenset({"w_logit"}): "kd_d1",
            frozenset({"w_rank", "w_dist"}): "kd_d2",
            frozenset({"w_gram"}): "kd_d3",
            frozenset({"w_align"}): "kd_d4",
            frozenset({"w_residual"}): "kd_d5",
            frozenset({"w_rank", "w_dist", "w_gram"}): "kd_d6",
            frozenset({"w_seed", "w_geom", "w_kl"}): "kd_d9",
        }[nonzero]
        return self.arm_label if (self.arm_label and self.active) else mapped

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> DistillConfig:
        """Build a ``distill:`` config section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or invalid values.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown distill config keys: {unknown}")
        kwargs: dict[str, object] = {}
        for field_spec in fields(cls):
            if field_spec.name not in mapping:
                continue
            raw = mapping[field_spec.name]
            if field_spec.name in ("targets_path", "arm_label"):
                if not isinstance(raw, str):
                    raise ValueError(f"distill.{field_spec.name} must be a string")
                kwargs[field_spec.name] = raw
            elif field_spec.name in _INT_FIELDS:
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise ValueError(f"distill.{field_spec.name} must be an integer")
                kwargs[field_spec.name] = raw
            else:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"distill.{field_spec.name} must be a number")
                kwargs[field_spec.name] = float(raw)
        return cls(**kwargs)  # type: ignore[arg-type]


__all__ = ["DistillConfig"]
