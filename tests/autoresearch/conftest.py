"""Fixture helpers that synthesize minimal published run directories."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

RunDirFactory = Callable[..., Path]


def make_metric_row(epoch: int, **overrides: object) -> dict[str, Any]:
    """One synthetic metrics.jsonl row in the B1 KD schema."""
    row: dict[str, Any] = {
        "epoch": epoch,
        "attempt_id": "fixture",
        "global_step": epoch * 10,
        "timestamp": f"2026-08-30T00:{epoch:02d}:00+00:00",
        "learning_rate": 1e-4,
        "train_loss": 1.0 / epoch,
        "train_kd_loss": 0.5 / epoch,
        "val_task_loss": 1.2 / epoch,
        "val_auroc": 0.9,
        "val_auprc": 0.80 + 0.01 * epoch,
        "val_ece": 0.05,
        "val_brier": 0.10,
        "val_gs_bfs": 0.50 + 0.01 * epoch,
        "val_rd_bfs": 1.10,
        "val_degree_mmd_ratio": 0.90,
        "val_clustering_mmd_ratio": 0.85,
        "val_spectral_mmd_ratio": 0.80,
        "val_threshold": 2.5,
        "grad_norm_task": 1.0,
        "grad_norm_kd": 0.3,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def make_run_dir(tmp_path: Path) -> RunDirFactory:
    """Factory building a published-run directory under ``tmp_path``."""

    def _make(
        name: str = "run",
        epochs: int = 3,
        selected_epoch: int = 2,
        rows: list[dict[str, Any]] | None = None,
        total_seconds: float = 123.0,
        failure: dict[str, Any] | None = None,
    ) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        actual = rows if rows is not None else [make_metric_row(e) for e in range(1, epochs + 1)]
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in actual), encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "selected_epoch": selected_epoch,
                    "arm": "kd_logit",
                    "config_hash": "deadbeef",
                    "checkpoint_id": "cafe0000",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "complete.json").write_text(
            json.dumps(
                {"status": "complete", "attempt_id": "fixture", "total_seconds": total_seconds}
            ),
            encoding="utf-8",
        )
        if failure is not None:
            (run_dir / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
        return run_dir

    return _make
