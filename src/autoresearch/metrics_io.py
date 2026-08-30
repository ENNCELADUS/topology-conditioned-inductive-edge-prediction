"""Read the frozen objective surface of one published training run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from src.eval.checkpoint_selection import (
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)


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


def read_run(run_dir: Path, topology_every: int | None = None) -> RunMetrics:
    """Load the six-metric surface at the selected epoch of ``run_dir``.

    ``topology_every`` reselects the epoch with the frozen checkpoint selector
    restricted to that cadence's due epochs (divisible by it, plus the final
    epoch), so a run measured at a denser cadence yields the surface a
    campaign at that cadence compares against; ``None`` trusts
    ``run_metadata.json``.

    Raises:
        RunFailure: If ``failure.json`` is present (the run must count as a crash).
        ValueError: On missing rows/keys, non-finite metrics, or RD <= 0.
    """
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        detail = failure_path.read_text(encoding="utf-8").strip()
        raise RunFailure(f"{run_dir} failed: {detail}")
    if topology_every is None:
        metadata = _load_json(run_dir / "run_metadata.json")
        selected_epoch = metadata.get("selected_epoch")
        if not _positive_int(selected_epoch):
            raise ValueError(f"{run_dir}: selected_epoch must be a positive int")
        row = _selected_row(run_dir / "metrics.jsonl", selected_epoch)
    else:
        selected_epoch, row = _reselect(run_dir, topology_every)
    auprc = _finite(row, "val_auprc", run_dir)
    topology = _topology(row, run_dir)
    threshold = _finite(row, "val_threshold", run_dir)
    complete = _load_json(run_dir / "complete.json")
    if complete.get("status") != "complete":
        raise ValueError(f"{run_dir}: complete.json status must be 'complete'")
    total_seconds = _finite(complete, "total_seconds", run_dir)
    if total_seconds < 0.0:
        raise ValueError(f"{run_dir}: total_seconds must be nonnegative, got {total_seconds}")
    return RunMetrics(
        run_dir=run_dir,
        selected_epoch=selected_epoch,
        auprc=auprc,
        topology=topology,
        threshold=threshold,
        total_seconds=total_seconds,
    )


def surface(run: RunMetrics) -> dict[str, Any]:
    """Flatten one run's judge-facing surface for a JSON payload."""
    return {
        "run_dir": str(run.run_dir),
        "selected_epoch": run.selected_epoch,
        "auprc": run.auprc,
        "gs": run.topology.gs,
        "rd": run.topology.rd,
        "degree_mmd": run.topology.degree_mmd,
        "clustering_mmd": run.topology.clustering_mmd,
        "spectral_mmd": run.topology.spectral_mmd,
        "threshold": run.threshold,
        "total_seconds": run.total_seconds,
    }


def _reselect(run_dir: Path, topology_every: int) -> tuple[int, dict[str, Any]]:
    """Re-run frozen selection over the epochs due at ``topology_every``."""
    if topology_every < 1:
        raise ValueError(f"{run_dir}: topology_every must be >= 1, got {topology_every}")
    rows_by_epoch: dict[int, dict[str, Any]] = {}
    for row in read_metric_rows(run_dir / "metrics.jsonl"):
        epoch = row.get("epoch")
        if not _positive_int(epoch):
            raise ValueError(f"{run_dir}: metrics row epoch must be a positive int")
        if epoch in rows_by_epoch:
            raise ValueError(f"{run_dir}: duplicate metrics row for epoch {epoch}")
        rows_by_epoch[epoch] = row
    if not rows_by_epoch:
        raise ValueError(f"{run_dir}: metrics.jsonl has no rows")
    final = max(rows_by_epoch)
    candidates = [
        CheckpointCandidate(
            epoch=epoch,
            auprc=_finite(row, "val_auprc", run_dir),
            topology=_topology(row, run_dir),
        )
        for epoch, row in sorted(rows_by_epoch.items())
        if epoch % topology_every == 0 or epoch == final
    ]
    selected = select_checkpoint(candidates)
    assert selected is not None  # the final epoch is always due, so never empty
    return selected.epoch, rows_by_epoch[selected.epoch]


def _topology(row: Mapping[str, Any], run_dir: Path) -> TopologyValidationMetrics:
    """Validate and extract the five topology metrics of one row."""
    gs = _finite(row, "val_gs_bfs", run_dir)
    rd = _finite(row, "val_rd_bfs", run_dir)
    degree_mmd = _finite(row, "val_degree_mmd_ratio", run_dir)
    clustering_mmd = _finite(row, "val_clustering_mmd_ratio", run_dir)
    spectral_mmd = _finite(row, "val_spectral_mmd_ratio", run_dir)
    if rd <= 0.0:
        raise ValueError(f"{run_dir}: val_rd_bfs must be positive, got {rd}")
    return TopologyValidationMetrics(gs, rd, degree_mmd, clustering_mmd, spectral_mmd)


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return parsed


def _selected_row(metrics_path: Path, selected_epoch: int) -> dict[str, Any]:
    for row in read_metric_rows(metrics_path):
        epoch = row.get("epoch")
        if epoch == selected_epoch:
            if not _positive_int(epoch):
                raise ValueError(f"{metrics_path}: selected row epoch must be a positive int")
            return row
    raise ValueError(f"{metrics_path}: no row for selected epoch {selected_epoch}")


def _finite(row: Mapping[str, Any], key: str, run_dir: Path) -> float:
    if key not in row:
        raise ValueError(f"{run_dir}: metrics row missing {key!r}")
    raw = row[key]
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        raise ValueError(f"{run_dir}: {key} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{run_dir}: non-finite {key}={value}")
    return value


def _positive_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
