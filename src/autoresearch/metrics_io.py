"""Read the frozen objective surface of one published training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.checkpoint_selection import TopologyValidationMetrics


class RunFailure(RuntimeError):
    """The run directory carries a ``failure.json`` marker."""


@dataclass(frozen=True)
class RunMetrics:
    """Judge-facing summary of one published run at its selected epoch."""

    run_dir: Path
    selected_epoch: int
    auprc: float
    topology: TopologyValidationMetrics
    threshold: float
    total_seconds: float


def read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """Parse every ``metrics.jsonl`` row strictly (published runs are validated).

    Raises:
        ValueError: If a line is not a JSON object.
    """
    raise NotImplementedError("scaffold: plan Task 2")


def read_run(run_dir: Path) -> RunMetrics:
    """Load the six-metric surface at the selected epoch of ``run_dir``.

    Raises:
        RunFailure: If ``failure.json`` is present (the run must count as a crash).
        ValueError: On missing rows/keys, non-finite metrics, or RD <= 0.
    """
    raise NotImplementedError("scaffold: plan Task 2")
