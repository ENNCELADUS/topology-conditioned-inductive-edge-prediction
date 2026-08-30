"""Frozen keep/revert verdict over the five topology metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.autoresearch.metrics_io import RunMetrics
from src.eval.checkpoint_selection import TopologyValidationMetrics

METRIC_NAMES: tuple[str, ...] = (
    "gs",
    "log_rd",
    "degree_mmd",
    "clustering_mmd",
    "spectral_mmd",
)


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
    return {
        "gs": -topology.gs,
        "log_rd": abs(math.log(topology.rd)),
        "degree_mmd": topology.degree_mmd,
        "clustering_mmd": topology.clustering_mmd,
        "spectral_mmd": topology.spectral_mmd,
    }


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
    widths = {name: float((bands or {}).get(name, 0.0)) for name in METRIC_NAMES}
    if any(width < 0.0 for width in widths.values()):
        raise ValueError("tolerance bands must be non-negative")

    incumbent_values = oriented(incumbent.topology)
    trial_values = oriented(trial.topology)
    deltas = {name: trial_values[name] - incumbent_values[name] for name in METRIC_NAMES}
    improved = tuple(name for name in METRIC_NAMES if deltas[name] < -widths[name])
    regressed = tuple(name for name in METRIC_NAMES if deltas[name] > widths[name])
    return Verdict(
        decision="keep" if improved and not regressed else "revert",
        improved=improved,
        regressed=regressed,
        deltas=deltas,
        auprc_delta=trial.auprc - incumbent.auprc,
    )


def undominated(runs: Sequence[RunMetrics]) -> list[RunMetrics]:
    """Return runs not Pareto-dominated on the five oriented topology metrics."""
    surfaces = [oriented(run.topology) for run in runs]

    def dominates(first: dict[str, float], second: dict[str, float]) -> bool:
        return all(first[name] <= second[name] for name in METRIC_NAMES) and any(
            first[name] < second[name] for name in METRIC_NAMES
        )

    return [
        run
        for index, run in enumerate(runs)
        if not any(
            dominates(other, surfaces[index])
            for other_index, other in enumerate(surfaces)
            if other_index != index
        )
    ]
