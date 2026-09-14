"""Stage I learning curves: the reader on true coordinates against its prompt-free recipe.

Logged V_val metrics only; no checkpoint is loaded or rescored. Every epoch's topology
numbers use that epoch's own V_val-selected threshold.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from paper_plot_style import BASE, FULL, plt, save  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

RUNS = {
    "prefix_base (no structure)": ("prefix_base_metrics.jsonl", 10, BASE, "s"),
    "topo_prompt_full (true coordinates)": ("topo_prompt_full_metrics.jsonl", 15, FULL, "o"),
}
PANELS = [
    ("val_auprc", "V_val AUPRC ↑", None),
    ("val_task_loss", "V_val task BCE ↓", None),
    ("val_gs_bfs", "GS ↑", None),
    ("val_rd_bfs", "RD → 1", 1.0),
    ("val_degree_mmd_ratio", "Degree MMD ratio ↓", None),
    ("val_clustering_mmd_ratio", "Clustering MMD ratio ↓", None),
    ("val_spectral_mmd_ratio", "Spectral MMD ratio ↓", None),
]

fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), layout="constrained")
for ax, (key, title, reference) in zip(axes.ravel(), PANELS):
    for label, (path, selected, color, marker) in RUNS.items():
        rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
        xs = [r["epoch"] for r in rows if key in r]
        ys = [r[key] for r in rows if key in r]
        ax.plot(xs, ys, marker=marker, color=color, label=label)
        sel = next((r for r in rows if r["epoch"] == selected), None)
        if sel is not None and key in sel:
            ax.plot([selected], [sel[key]], marker="*", ms=11, color=color, mec="black", mew=0.6, zorder=5)
    if reference is not None:
        ax.axhline(reference, color="black", ls=":", lw=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
for ax in axes[1]:
    ax.set_xlabel("Epoch")
legend_ax = axes[1, 3]
legend_ax.axis("off")
handles, labels = axes[0, 0].get_legend_handles_labels()
star = plt.Line2D([], [], marker="*", ms=11, color="white", mec="black", mew=0.6, ls="")
legend_ax.legend(
    [*handles, star],
    [*labels, "V_val-selected checkpoint"],
    loc="center",
    frameon=False,
)
save(fig, ROOT, "stage1_curves")
print("written", ROOT / "stage1_curves.png")
