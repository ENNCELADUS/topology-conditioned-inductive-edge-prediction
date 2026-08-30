"""Frozen keep/revert verdict over the five topology metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.autoresearch.metrics_io import RunMetrics
from src.eval.checkpoint_selection import TopologyValidationMetrics

METRIC_NAMES = ("gs", "log_rd", "degree_mmd", "clustering_mmd", "spectral_mmd")


@dataclass(frozen=True)
class Verdict:
    """Outcome of judging one trial against the incumbent."""

    decision: str
    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    deltas: dict[str, float]
    auprc_delta: float


def oriented(topology: TopologyValidationMetrics) -> dict[str, float]:
    """Orient the five metrics so that lower is always better."""
    raise NotImplementedError("scaffold: plan Task 3")


def judge_runs(
    incumbent: RunMetrics,
    trial: RunMetrics,
    bands: Mapping[str, float] | None = None,
) -> Verdict:
    """Keep iff >=1 metric improves beyond its band and none regresses beyond its band.

    Bands default to zero width (strict no-regression). AUPRC is telemetry
    only and never enters the decision.

    Raises:
        ValueError: If any band width is negative.
    """
    raise NotImplementedError("scaffold: plan Task 3")


def undominated(runs: Sequence[RunMetrics]) -> list[RunMetrics]:
    """Return runs not Pareto-dominated on the five oriented topology metrics."""
    raise NotImplementedError("scaffold: plan Task 3")
