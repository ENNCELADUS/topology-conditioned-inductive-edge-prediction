"""Publication learning curves for KD1 (kd_logit), run kd_logit_w100.

Reads learning_curves.csv next to this file and renders the loss and
validation-topology PNGs next to it. Raw (unweighted) loss terms are
plotted; KD weights appear in the panel captions only. The total
panel is task + sum(w * raw KD term), composed identically for
train and validation from the per-row-mean telemetry.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

ARM = "KD1 (kd_logit)"
RUN = "kd_logit_w100"
KD_WEIGHTS = {"w_logit": 100.0}
SELECTED_EPOCH = 25
# Panels: (caption, ylabel, train_column, val_column).
PANELS: tuple[tuple[str, str, str, str], ...] = (
    ("(a) Task BCE", "BCE", "train_loss", "val_task_loss"),
    ("(b) KD logit BCE (w = 100)", "Soft-target BCE", "kd_logit_loss", "val_kd_logit_loss"),
    ("(c) Total (task + w·KD)", "Total loss", "train_total", "val_total"),
)

TOPOLOGY_PANELS: tuple[tuple[str, str, str, float | None], ...] = (
    ("(a) BFS-macro GS (higher)", "GS", "val_gs_bfs", None),
    ("(b) BFS-macro RD (target 1)", "RD", "val_rd_bfs", 1.0),
    ("(c) Degree MMD ratio (lower)", "MMD ratio", "val_degree_mmd_ratio", None),
    ("(d) Clustering MMD ratio (lower)", "MMD ratio", "val_clustering_mmd_ratio", None),
    ("(e) Spectral MMD ratio (lower)", "MMD ratio", "val_spectral_mmd_ratio", None),
)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "learning_curves.csv"
OUTPUT_PATH = HERE / "learning_curves.png"
TOPOLOGY_OUTPUT_PATH = HERE / "validation_topology_curves.png"
ORACLE_REFERENCE_PATH = HERE.parent / "oracle_vval_reference.csv"

TRAIN_COLOR = "#0072B2"
VAL_COLOR = "#D55E00"
ORACLE_COLOR = "#009E73"
GRAY = "#666666"


def load_curves() -> dict[str, list[float]]:
    """Load the CSV and validate the fixed epoch-1..25 curve contract."""
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 26)):
        raise ValueError("learning_curves.csv must contain epochs 1..25 exactly once")
    columns = (
        ["epoch"]
        + [key for panel in PANELS for key in panel[2:]]
        + [panel[2] for panel in TOPOLOGY_PANELS]
    )
    curves = {name: [float(row[name]) for row in rows] for name in columns}
    for name, values in curves.items():
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite value in column {name!r}")
    return curves


def load_oracle_reference() -> dict[str, float]:
    """Load the shared selected-epoch Oracle V_val metrics."""
    with ORACLE_REFERENCE_PATH.open(newline="", encoding="utf-8") as handle:
        values = {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}
    required = {panel[2] for panel in TOPOLOGY_PANELS}
    if missing := required - values.keys():
        raise ValueError(f"missing Oracle V_val metrics: {sorted(missing)}")
    return values


def configure_style() -> None:
    """Apply a compact, print-safe sans-serif style on the Agg backend."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def plot_panel(
    ax: Axes,
    epochs: list[float],
    train: list[float],
    validation: list[float],
    caption: str,
    ylabel: str,
    label_selected: bool,
) -> None:
    """Draw one train/validation panel with the selected epoch marked."""
    ax.plot(
        epochs,
        train,
        color=TRAIN_COLOR,
        linewidth=1.6,
        marker="o",
        markersize=2.8,
        markevery=2,
    )
    ax.plot(
        epochs,
        validation,
        color=VAL_COLOR,
        linewidth=1.6,
        linestyle="--",
        marker="s",
        markersize=2.8,
        markevery=2,
    )
    ax.axvline(SELECTED_EPOCH, color=GRAY, linewidth=1.0, linestyle=":", zorder=0)
    if label_selected:
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            SELECTED_EPOCH - 0.5,
            0.97,
            "selected",
            transform=blend,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=GRAY,
        )
    ax.set_title(caption, loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 25.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def plot_topology_panel(
    ax: Axes,
    epochs: list[float],
    values: list[float],
    caption: str,
    ylabel: str,
    target: float | None,
    oracle: float,
    label_selected: bool,
) -> None:
    """Draw one validation-topology metric across epochs."""
    ax.plot(
        epochs,
        values,
        color=VAL_COLOR,
        linewidth=1.6,
        marker="s",
        markersize=2.8,
        markevery=2,
    )
    if target is not None:
        ax.axhline(target, color="#999999", linewidth=0.9, linestyle="--", zorder=0)
    ax.axhline(oracle, color=ORACLE_COLOR, linewidth=1.2, linestyle="-.", zorder=0)
    ax.axvline(SELECTED_EPOCH, color=GRAY, linewidth=1.0, linestyle=":", zorder=0)
    if label_selected:
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            SELECTED_EPOCH - 0.5,
            0.97,
            "selected",
            transform=blend,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=GRAY,
        )
    ax.set_title(caption, loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 25.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def render_topology_curves(
    curves: dict[str, list[float]], oracle_reference: dict[str, float]
) -> None:
    """Render the five validation-topology metrics as a 300-DPI PNG."""
    epochs = curves["epoch"]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.6, 5.6),
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for index, (caption, ylabel, key, target) in enumerate(TOPOLOGY_PANELS):
        plot_topology_panel(
            flat_axes[index],
            epochs,
            curves[key],
            caption,
            ylabel,
            target,
            oracle_reference[key],
            index == 0,
        )
    legend_ax = flat_axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=VAL_COLOR, marker="s", linewidth=1.6, label="KD V_val"),
            Line2D(
                [0],
                [0],
                color=ORACLE_COLOR,
                linewidth=1.2,
                linestyle="-.",
                label="Oracle V_val",
            ),
            Line2D([0], [0], color=GRAY, linewidth=1.0, linestyle=":", label="Selected epoch"),
            Line2D([0], [0], color="#999999", linewidth=0.9, linestyle="--", label="RD target 1"),
        ],
        loc="center",
        frameon=False,
    )
    fig.savefig(TOPOLOGY_OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    """Render the learning-curve figure for this arm as a 300-DPI PNG."""
    configure_style()
    curves = load_curves()
    oracle_reference = load_oracle_reference()
    epochs = curves["epoch"]
    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(3.2 * len(PANELS), 2.9),
        constrained_layout=True,
    )
    for index, (caption, ylabel, train_key, val_key) in enumerate(PANELS):
        plot_panel(
            axes[index],
            epochs,
            curves[train_key],
            curves[val_key],
            caption,
            ylabel,
            index == 0,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=TRAIN_COLOR,
            marker="o",
            linewidth=1.6,
            markersize=3.2,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            color=VAL_COLOR,
            marker="s",
            linewidth=1.6,
            linestyle="--",
            markersize=3.2,
            label="Validation",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    render_topology_curves(curves, oracle_reference)


if __name__ == "__main__":
    main()
