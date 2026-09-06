"""Render Trial 5 epoch telemetry; missing topology epochs remain unfilled."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with (HERE / "learning_curves.csv").open() as handle:
    rows = list(csv.DictReader(handle))
epochs = [int(row["epoch"]) for row in rows]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def values(key):
    return [float(row[key]) if row[key] else float("nan") for row in rows]


def decorate(ax, title):
    ax.set_title(title, loc="left")
    ax.axvline(20, color="#666666", linestyle=":", label="Selected epoch 20")
    ax.set(xlabel="Epoch", xlim=(0.5, 25.5), xticks=[1, 5, 10, 15, 20, 25])
    ax.grid(axis="y", alpha=0.2)


fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), constrained_layout=True)
panels = [
    ("Task BCE", "train_loss", "val_task_loss"),
    ("Rank loss (w = 0.0141)", "kd_rank_loss", "val_kd_rank_loss"),
    ("Distribution KL (w = 5.781)", "kd_dist_loss", "val_kd_dist_loss"),
    ("Representation: 1 − cosine (w = 0.0961)", "kd_rep_loss", "val_kd_rep_loss"),
    ("Total loss", None, "val_total_loss"),
]
for ax, (title, train, val) in zip(axes.flat, panels):
    train_values = values(train) if train else [
        float(r["train_loss"]) + float(r["train_kd_loss"]) for r in rows
    ]
    ax.plot(epochs, train_values, color="#0072B2", label="Train")
    ax.plot(epochs, values(val), color="#D55E00", linestyle="--", label="V_val")
    decorate(ax, title)
axes.flat[-1].axis("off")
axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False)
fig.savefig(HERE / "learning_curves.png", dpi=300)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), constrained_layout=True)
for ax, (title, key) in zip(axes.flat, [
    ("BFS GS ↑", "val_gs_bfs"), ("BFS RD → 1", "val_rd_bfs"),
    ("Degree MMD ratio ↓", "val_degree_mmd_ratio"),
    ("Clustering MMD ratio ↓", "val_clustering_mmd_ratio"),
    ("Spectral MMD ratio ↓", "val_spectral_mmd_ratio"),
]):
    measured = [(e, float(r[key])) for e, r in zip(epochs, rows) if r[key]]
    ax.plot(*zip(*measured), color="#D55E00", marker="s", markersize=3, label="V_val")
    decorate(ax, title)
    if key == "val_rd_bfs":
        ax.axhline(1, color="#999999", linestyle="--")
axes.flat[-1].axis("off")
axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False)
fig.savefig(HERE / "validation_topology_curves.png", dpi=300)
plt.close(fig)
