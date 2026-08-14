"""`DistillConfig`: the B1 training-time topology-distillation knobs.

Consumed by the simple B0-protocol trainer (`src.train_b0`) as the optional
top-level ``distill:`` config section. Exactly one arm group's weight(s) may
be nonzero at a time -- ``kd_control`` (`w_label`), ``kd_d1`` (`w_logit`),
``kd_d2`` (`w_rank` and `w_dist` together), ``kd_d3`` (`w_gram`) -- so a
config can never straddle two KD mechanisms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


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
        temperature: Softmax/KL temperature for `w_dist` (LLP reference pins 1.0).
        margin: Margin for the `w_rank` pairwise ranking loss.
        anchors_per_step: KD anchor groups drawn per optimizer step per rank.
    """

    targets_path: str = ""
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
                `temperature`/`margin`, a non-positive `anchors_per_step`, a
                nonzero-weight pattern outside the four legal arm groups, or
                a nonzero weight without a `targets_path`.
        """
        weight_names = ("w_label", "w_logit", "w_rank", "w_dist", "w_gram")
        for name in weight_names:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.anchors_per_step < 1:
            raise ValueError(f"anchors_per_step must be >= 1, got {self.anchors_per_step}")
        nonzero = frozenset(name for name in weight_names if float(getattr(self, name)) > 0.0)
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
        if nonzero and not self.targets_path:
            raise ValueError("distill.targets_path is required when any weight is nonzero")

    @property
    def active(self) -> bool:
        """Whether any distillation weight is nonzero."""
        return any(
            float(getattr(self, name)) > 0.0
            for name in ("w_label", "w_logit", "w_rank", "w_dist", "w_gram")
        )

    @property
    def arm(self) -> str:
        """The KD arm this weight pattern names (``none`` when inactive)."""
        nonzero = frozenset(
            name
            for name in ("w_label", "w_logit", "w_rank", "w_dist", "w_gram")
            if float(getattr(self, name)) > 0.0
        )
        return {
            frozenset(): "none",
            frozenset({"w_label"}): "kd_control",
            frozenset({"w_logit"}): "kd_d1",
            frozenset({"w_rank", "w_dist"}): "kd_d2",
            frozenset({"w_gram"}): "kd_d3",
        }[nonzero]

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
                    raise ValueError("distill.targets_path must be a string")
                kwargs[field_spec.name] = raw
            elif field_spec.name == "anchors_per_step":
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise ValueError("distill.anchors_per_step must be an integer")
                kwargs[field_spec.name] = raw
            else:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"distill.{field_spec.name} must be a number")
                kwargs[field_spec.name] = float(raw)
        return cls(**kwargs)  # type: ignore[arg-type]


__all__ = ["DistillConfig"]
