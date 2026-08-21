"""`DistillConfig`: the B1 training-time knowledge-distillation knobs.

Consumed by the simple B0-protocol trainer (`src.train_b0`) as the optional
top-level ``distill:`` config section. Teacher targets are dumped once for
every official training row (`src/distill/teacher_targets.py`, format
``kd_row_targets_v1``); task and KD losses then share one student forward
pass over exactly those same rows -- there is no separate KD-only stream or
second forward. Exactly one arm group's weight(s) may be nonzero at a time:
``kd_logit`` (`w_logit`, pointwise soft-target logit KD), ``kd_rank``
(`w_rank` + `w_dist`, anchor ranking/distribution KD), ``kd_gram`` (`w_gram`,
batch-relational Gram KD), ``kd_rep`` (`w_rep`, per-row representation
alignment), or ``kd_d9`` (`w_seed`, `w_geom`, and `w_kl` together). The
matched control is a config with no
``distill:`` section at all, or one with every weight left at zero; either
must reproduce the undistilled baseline exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

_WEIGHT_NAMES = (
    "w_logit",
    "w_rank",
    "w_dist",
    "w_gram",
    "w_rep",
    "w_seed",
    "w_geom",
    "w_kl",
)
_INT_FIELDS = ("kl_warmup_steps", "joint_start_epoch")


@dataclass(frozen=True)
class DistillConfig:
    """B1 training-time knowledge-distillation knobs.

    Attributes:
        targets_path: Path to the dumped full-row teacher-target artifact
            (`src/distill/teacher_targets.py`, format ``kd_row_targets_v1``).
            Required whenever any weight below is nonzero.
        w_logit: ``kd_logit`` weight -- pointwise binary soft-target KD,
            BCE(student_logit, sigmoid(teacher_logit)), on the training
            batch's own rows (GLNN family).
        w_rank: ``kd_rank`` margin-ranking weight over official batch rows
            sharing either endpoint as an anchor.
        w_dist: ``kd_rank`` per-anchor distribution-KL weight over the same
            official batch rows.
        w_gram: ``kd_gram`` cosine-Gram weight over pair representations from
            all official rows in the task batch.
        w_rep: ``kd_rep`` weight -- per-row cosine alignment of the student
            pair representation (through a train-time linear projection only
            when the student and teacher widths differ) to the symmetrized
            teacher pooled embedding.
        w_seed: ``kd_d9`` weight -- per-slot cosine reconstruction of the
            teacher's symmetrized PMA seed tokens by the posterior-path
            generated seeds (pair-latent generation, D9 design).
        w_geom: ``kd_d9`` weight -- relative Frobenius match of the raw
            within-pair seed Gram (slot norms + inter-slot geometry).
        w_kl: ``kd_d9`` weight -- free-bits KL(q || p) tying the recognition
            posterior to the deployed prior; warmed up over `kl_warmup_steps`.
        kl_warmup_steps: ``kd_d9`` -- optimizer steps over which the `w_kl`
            weight ramps linearly 0 -> 1 (0 disables the ramp).
        joint_start_epoch: ``kd_d9`` -- first epoch of stage-2 joint
            fine-tuning: the fusion-boundary stop-gradient drops and the
            generator param group switches to `gen_lr_scale` x base LR
            (0 = joint from the start).
        gen_lr_scale: ``kd_d9`` -- stage-2 LR multiplier for the generator
            param group.
        margin: ``kd_rank`` ranking margin.
        temperature: ``kd_rank`` distribution-softmax temperature.
    """

    targets_path: str = ""
    w_logit: float = 0.0
    w_rank: float = 0.0
    w_dist: float = 0.0
    w_gram: float = 0.0
    w_rep: float = 0.0
    w_seed: float = 0.0
    w_geom: float = 0.0
    w_kl: float = 0.0
    kl_warmup_steps: int = 2000
    joint_start_epoch: int = 0
    gen_lr_scale: float = 0.1
    margin: float = 0.1
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate weight signs/ranges and the single-arm-group pattern.

        Raises:
            ValueError: On a negative weight, a negative `kl_warmup_steps` or
                `joint_start_epoch`, a non-positive `gen_lr_scale`, a
                nonzero-weight pattern outside the legal arm groups, or a
                nonzero weight without a `targets_path`.
        """
        for name in _WEIGHT_NAMES:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.kl_warmup_steps < 0:
            raise ValueError(f"kl_warmup_steps must be >= 0, got {self.kl_warmup_steps}")
        if self.joint_start_epoch < 0:
            raise ValueError(f"joint_start_epoch must be >= 0, got {self.joint_start_epoch}")
        if self.gen_lr_scale <= 0.0:
            raise ValueError(f"gen_lr_scale must be positive, got {self.gen_lr_scale}")
        if self.margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        nonzero = frozenset(name for name in _WEIGHT_NAMES if float(getattr(self, name)) > 0.0)
        legal_patterns: tuple[frozenset[str], ...] = (
            frozenset(),
            frozenset({"w_logit"}),
            frozenset({"w_rank", "w_dist"}),
            frozenset({"w_gram"}),
            frozenset({"w_rep"}),
            frozenset({"w_seed", "w_geom", "w_kl"}),
        )
        if nonzero not in legal_patterns:
            raise ValueError(
                "distill weights must follow exactly one arm group -- all zero, only "
                "w_logit (kd_logit), w_rank and w_dist (kd_rank), only w_gram "
                "(kd_gram), only w_rep (kd_rep), or w_seed, w_geom, and w_kl together "
                f"(kd_d9); got nonzero weights {sorted(nonzero)}"
            )
        if nonzero and not self.targets_path:
            raise ValueError("distill.targets_path is required when any weight is nonzero")

    @property
    def active(self) -> bool:
        """Whether any distillation weight is nonzero."""
        return any(float(getattr(self, name)) > 0.0 for name in _WEIGHT_NAMES)

    @property
    def arm(self) -> str:
        """The KD arm this weight pattern names (``none`` when inactive)."""
        nonzero = frozenset(name for name in _WEIGHT_NAMES if float(getattr(self, name)) > 0.0)
        mapped = {
            frozenset(): "none",
            frozenset({"w_logit"}): "kd_logit",
            frozenset({"w_rank", "w_dist"}): "kd_rank",
            frozenset({"w_gram"}): "kd_gram",
            frozenset({"w_rep"}): "kd_rep",
            frozenset({"w_seed", "w_geom", "w_kl"}): "kd_d9",
        }
        return mapped[nonzero]

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
            if field_spec.name == "targets_path":
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
