"""Read the frozen objective surface of one published training run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
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
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{metrics_path}:{number}: row is not an object")
        rows.append(parsed)
    return rows


def read_run(run_dir: Path) -> RunMetrics:
    """Load the six-metric surface at the selected epoch of ``run_dir``.

    Raises:
        RunFailure: If ``failure.json`` is present (the run must count as a crash).
        ValueError: On missing rows/keys, non-finite metrics, or RD <= 0.
    """
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        detail = failure_path.read_text(encoding="utf-8").strip()
        raise RunFailure(f"{run_dir} failed: {detail}")
    metadata = _load_json(run_dir / "run_metadata.json")
    selected_epoch = metadata.get("selected_epoch")
    if not isinstance(selected_epoch, int) or isinstance(selected_epoch, bool):
        raise ValueError(f"{run_dir}: selected_epoch must be an int")
    row = _selected_row(run_dir / "metrics.jsonl", selected_epoch)
    auprc = _finite(row, "val_auprc", run_dir)
    gs = _finite(row, "val_gs_bfs", run_dir)
    rd = _finite(row, "val_rd_bfs", run_dir)
    degree_mmd = _finite(row, "val_degree_mmd_ratio", run_dir)
    clustering_mmd = _finite(row, "val_clustering_mmd_ratio", run_dir)
    spectral_mmd = _finite(row, "val_spectral_mmd_ratio", run_dir)
    threshold = _finite(row, "val_threshold", run_dir)
    if rd <= 0.0:
        raise ValueError(f"{run_dir}: val_rd_bfs must be positive, got {rd}")
    complete = _load_json(run_dir / "complete.json")
    total_seconds = _finite(complete, "total_seconds", run_dir)
    return RunMetrics(
        run_dir=run_dir,
        selected_epoch=selected_epoch,
        auprc=auprc,
        topology=TopologyValidationMetrics(gs, rd, degree_mmd, clustering_mmd, spectral_mmd),
        threshold=threshold,
        total_seconds=total_seconds,
    )


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return parsed


def _selected_row(metrics_path: Path, selected_epoch: int) -> dict[str, Any]:
    for row in read_metric_rows(metrics_path):
        if row.get("epoch") == selected_epoch:
            return row
    raise ValueError(f"{metrics_path}: no row for selected epoch {selected_epoch}")


def _finite(row: Mapping[str, Any], key: str, run_dir: Path) -> float:
    if key not in row:
        raise ValueError(f"{run_dir}: metrics row missing {key!r}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_dir}: {key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{run_dir}: non-finite {key}={value}")
    return value
