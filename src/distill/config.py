"""`DistillConfig`: the B1 training-time knowledge-distillation knobs.

Consumed by the simple B0-protocol trainer (`src.train_b0`) as the optional
top-level ``distill:`` config section. Teacher row targets are dumped once
for every official training row (`src/distill/teacher_targets.py`, format
``kd_row_targets_v1``). ``kd_rank`` also consumes a separately dumped context
bank (``kd_ctx_targets_v1``); other arms share the task-row forward. Exactly
one arm group's weight(s) may be nonzero at a time:
``kd_logit`` (`w_logit`, pointwise soft-target logit KD), ``kd_rank``
(`w_rank` + `w_dist`, anchor ranking/distribution KD), ``kd_gram`` (`w_gram`,
batch-relational Gram KD), ``kd_rep`` (`w_rep`, per-row representation
alignment), ``kd_gen`` (`w_gen`, generator distillation), or ``kd_struct``
(`w_struct`, descriptor-level structural auxiliary head with in-process
targets and no teacher artifact). The matched control is a config with no
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
    "w_gen",
    "w_struct",
)


@dataclass(frozen=True)
class DistillConfig:
    """B1 training-time knowledge-distillation knobs.

    Attributes:
        targets_path: Path to the dumped full-row teacher-target artifact
            (`src/distill/teacher_targets.py`, format ``kd_row_targets_v1``).
            Required whenever any weight below is nonzero.
        context_targets_path: Path to the dumped ``kd_ctx_targets_v1`` artifact.
            Required exactly when the ``kd_rank`` arm is active.
        w_logit: ``kd_logit`` weight -- pointwise binary soft-target KD,
            BCE(student_logit, sigmoid(teacher_logit)), on the training
            batch's own rows (GLNN family).
        w_rank: ``kd_rank`` margin-ranking weight over the separately sampled
            per-epoch anchor/context groups.
        w_dist: ``kd_rank`` per-anchor distribution-KL weight over those same
            context groups.
        w_gram: ``kd_gram`` cosine-Gram weight over pair representations from
            all official rows in the task batch.
        w_rep: ``kd_rep`` weight -- per-row cosine alignment of the student
            pair representation (through a train-time linear projection only
            when the student and teacher widths differ) to the symmetrized
            teacher pooled embedding.
        w_gen: ``kd_gen`` generator-loss weight.
        w_struct: ``kd_struct`` weight -- MSE of ``model.config.kd_struct_dim``
            auxiliary-head outputs against z-scored ego-graph descriptors of
            the truth graph (`src/distill/struct_targets.py`); needs no
            ``targets_path``.
        joint_warmup_frac: ``kd_gen`` fraction of training spent with ``sg``
            at the adapter before joint optimization.
        gen_lr_scale: ``kd_gen`` joint-phase generator LR multiplier.
        margin: ``kd_rank`` shared teacher tie-band and hinge margin delta.
    """

    targets_path: str = ""
    context_targets_path: str = ""
    w_logit: float = 0.0
    w_rank: float = 0.0
    w_dist: float = 0.0
    w_gram: float = 0.0
    w_rep: float = 0.0
    w_gen: float = 0.0
    w_struct: float = 0.0
    joint_warmup_frac: float = 0.1
    gen_lr_scale: float = 0.1
    margin: float = 0.1

    def __post_init__(self) -> None:
        """Validate weight signs/ranges and the single-arm-group pattern.

        Raises:
            ValueError: On a negative weight, an invalid `joint_warmup_frac`,
                a non-positive `gen_lr_scale`, a nonzero-weight pattern outside
                the legal arm groups, or a nonzero weight without a
                `targets_path`.
        """
        for name in _WEIGHT_NAMES:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not 0.0 <= self.joint_warmup_frac < 1.0:
            raise ValueError(f"joint_warmup_frac must be in [0, 1), got {self.joint_warmup_frac}")
        if self.gen_lr_scale <= 0.0:
            raise ValueError(f"gen_lr_scale must be positive, got {self.gen_lr_scale}")
        if self.margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        nonzero = frozenset(name for name in _WEIGHT_NAMES if float(getattr(self, name)) > 0.0)
        legal_patterns: tuple[frozenset[str], ...] = (
            frozenset(),
            frozenset({"w_logit"}),
            frozenset({"w_rank", "w_dist"}),
            frozenset({"w_gram"}),
            frozenset({"w_rep"}),
            frozenset({"w_gen"}),
            frozenset({"w_struct"}),
        )
        if nonzero not in legal_patterns:
            raise ValueError(
                "distill weights must follow exactly one arm group -- all zero, only "
                "w_logit (kd_logit), w_rank and w_dist (kd_rank), only w_gram "
                "(kd_gram), only w_rep (kd_rep), only w_gen (kd_gen), or only w_struct "
                f"(kd_struct); got nonzero weights {sorted(nonzero)}"
            )
        kd_struct_active = nonzero == frozenset({"w_struct"})
        if nonzero and not kd_struct_active and not self.targets_path:
            raise ValueError("distill.targets_path is required when a teacher arm is active")
        if kd_struct_active and self.targets_path:
            raise ValueError("kd_struct derives its targets in-process; drop distill.targets_path")
        kd_rank_active = nonzero == frozenset({"w_rank", "w_dist"})
        if kd_rank_active and not self.context_targets_path:
            raise ValueError("distill.context_targets_path is required when kd_rank is active")
        if not kd_rank_active and self.context_targets_path:
            raise ValueError("distill.context_targets_path is only valid when kd_rank is active")

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
            frozenset({"w_gen"}): "kd_gen",
            frozenset({"w_struct"}): "kd_struct",
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
            if field_spec.name in {"targets_path", "context_targets_path"}:
                if not isinstance(raw, str):
                    raise ValueError(f"distill.{field_spec.name} must be a string")
                kwargs[field_spec.name] = raw
            else:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"distill.{field_spec.name} must be a number")
                kwargs[field_spec.name] = float(raw)
        return cls(**kwargs)  # type: ignore[arg-type]


__all__ = ["DistillConfig"]
