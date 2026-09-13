"""Render the Stage I reader (true coordinates) against its base recipe; logged validation only."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/topo_stage1_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
RUNS = {"prefix_base (no structure)": ("prefix_base_metrics.jsonl", 10, "#555555"),
        "topo_prompt_full (true coordinates)": ("topo_prompt_full_metrics.jsonl", 15, "#0072B2")}
PANELS = [("val_auprc", "Validation AUPRC ↑"), ("val_task_loss", "Validation task BCE ↓"),
          ("val_gs_bfs", "BFS-macro GS ↑"), ("val_rd_bfs", "Arithmetic RD → 1"),
          ("val_degree_mmd_ratio", "Degree MMD ratio ↓"), ("val_spectral_mmd_ratio", "Spectral MMD ratio ↓")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (key, title) in zip(axes.ravel(), PANELS):
    for label, (path, selected, color) in RUNS.items():
        rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
        xs = [r["epoch"] for r in rows if key in r]; ys = [r[key] for r in rows if key in r]
        ax.plot(xs, ys, marker="o", ms=3, color=color, label=label)
        sel = next((r for r in rows if r["epoch"] == selected), None)
        if sel is not None and key in sel:
            ax.plot([selected], [sel[key]], marker="*", ms=13, color=color, mec="black", zorder=5)
    if key == "val_rd_bfs":
        ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(axis="y", alpha=0.18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("Stage I: the reader on true structural coordinates against the same recipe without them", y=1.0)
fig.text(0.5, -0.01, "Stars: V_val-selected checkpoints (prefix_base epoch 10, topo_prompt_full epoch 15). "
         "Every epoch uses its own validation-selected topology threshold.", ha="center", fontsize=9)
fig.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(ROOT / f"stage1_curves.{ext}", dpi=200, bbox_inches="tight")
print("written")
