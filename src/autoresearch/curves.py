"""Distill one run's metrics.jsonl into a learning-curve CSV and plot.

Usage: ``python -m src.autoresearch.curves RUN_DIR OUT_DIR``. Writes
``learning_curves.csv`` (columns below) and ``learning_curves.png`` (four
panels — losses, GS/RD, MMD ratios, AUPRC — with the selected epoch marked).
"""

from __future__ import annotations

from pathlib import Path

CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "train_kd_loss",
    "val_task_loss",
    "val_auprc",
    "val_gs_bfs",
    "val_rd_bfs",
    "val_degree_mmd_ratio",
    "val_clustering_mmd_ratio",
    "val_spectral_mmd_ratio",
    "learning_rate",
    "grad_norm_task",
    "grad_norm_kd",
)


def write_curves(run_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write ``learning_curves.csv`` and ``learning_curves.png`` for one run."""
    raise NotImplementedError("scaffold: plan Task 4")


def main(argv: list[str] | None = None) -> int:
    """CLI: distill RUN_DIR's metrics.jsonl into OUT_DIR's CSV + PNG."""
    raise NotImplementedError("scaffold: plan Task 4")


if __name__ == "__main__":
    raise SystemExit(main())
