"""Distill one run's metrics.jsonl into a learning-curve CSV and plot.

Usage: ``python -m src.autoresearch.curves RUN_DIR OUT_DIR``. Writes
``learning_curves.csv`` (columns below) and ``learning_curves.png`` (four
panels — losses, GS/RD, MMD ratios, AUPRC — with the selected epoch marked).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.autoresearch.metrics_io import read_metric_rows  # noqa: E402

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

_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("train_loss", "train_kd_loss", "val_task_loss")),
    ("topology (GS, RD)", ("val_gs_bfs", "val_rd_bfs")),
    (
        "MMD ratios",
        ("val_degree_mmd_ratio", "val_clustering_mmd_ratio", "val_spectral_mmd_ratio"),
    ),
    ("val AUPRC", ("val_auprc",)),
)


def write_curves(run_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write ``learning_curves.csv`` and ``learning_curves.png`` for one run."""
    rows = read_metric_rows(run_dir / "metrics.jsonl")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    selected_epoch = int(metadata["selected_epoch"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "learning_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    png_path = out_dir / "learning_curves.png"
    _plot(rows, selected_epoch, png_path)
    return csv_path, png_path


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """Collect (epoch, value) pairs for rows that carry ``key``."""
    epochs = [int(row["epoch"]) for row in rows if key in row]
    values = [float(row[key]) for row in rows if key in row]
    return epochs, values


def _plot(rows: list[dict[str, Any]], selected_epoch: int, png_path: Path) -> None:
    """Render the four-panel learning-curve figure with the selected epoch marked."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    flat_axes = (axes[0][0], axes[0][1], axes[1][0], axes[1][1])
    for ax, (title, keys) in zip(flat_axes, _PANELS, strict=True):
        for key in keys:
            epochs, values = _series(rows, key)
            if epochs:
                ax.plot(epochs, values, marker="o", markersize=2.5, linewidth=1.4, label=key)
        ax.axvline(selected_epoch, color="#666666", linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.4, linewidth=0.5)
        ax.legend(fontsize=7)
    for ax in (axes[1][0], axes[1][1]):
        ax.set_xlabel("epoch")
    fig.suptitle(f"selected epoch {selected_epoch}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: distill RUN_DIR's metrics.jsonl into OUT_DIR's CSV + PNG."""
    parser = argparse.ArgumentParser(prog="autoresearch-curves")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    csv_path, png_path = write_curves(args.run_dir, args.out_dir)
    sys.stdout.write(f"{csv_path}\n{png_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
