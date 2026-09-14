"""Stage II learning curves of the coord_gen_full lane (the v1 deployable row) only.

Top row: the logged composite training loss and the three validation loss terms (KD before
its 0.1 weight). Bottom row: V_val edge and topology metrics at each epoch's own selected
threshold, against the prompt-free recipe's selected checkpoint (dotted). Logged values
only; no rescoring.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from paper_plot_style import BASE, FULL, GREEN, ORANGE, VERMILION, plt, save  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ARM = "coord_gen_full"
rows = [json.loads(s) for s in (ROOT / ARM / "metrics.jsonl").read_text().splitlines() if s.strip()]
selected = json.loads((ROOT / ARM / "test_report.json").read_text())["arm"]["selected_epoch"]
base_rows = [
    json.loads(s)
    for s in (ROOT.parent / "topo_prompt_stage1_curves" / "prefix_base_metrics.jsonl").read_text().splitlines()
    if s.strip()
]
base = next(r for r in base_rows if r["epoch"] == 10)  # prefix_base's V_val-selected checkpoint
x = [r["epoch"] for r in rows]


def series(key):
    return [r[key] for r in rows]


fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), layout="constrained")
losses = [
    ("train_loss", "Training composite loss"),
    ("val_task_loss", r"V_val $\mathcal{L}_{\mathrm{BCE}}$"),
    ("val_coord_loss", r"V_val $\mathcal{L}_{\mathrm{coord}}$"),
    ("val_kd_loss", r"V_val $\mathcal{L}_{\mathrm{KD}}$ (unweighted)"),
]
for ax, (key, title) in zip(axes[0], losses):
    y = series(key)
    ax.plot(x, y, marker="o", color=FULL)
    k = min(range(len(y)), key=y.__getitem__)
    ax.plot([x[k]], [y[k]], marker="v", ms=6, color="black", ls="", zorder=5)
    ax.plot([selected], [y[selected - 1]], marker="*", ms=11, color=FULL, mec="black", mew=0.6, ls="", zorder=6)
    ax.set_title(title)
    ax.margins(y=0.15)

metrics = [("val_auprc", "V_val AUPRC ↑"), ("val_gs_bfs", "GS ↑"), ("val_rd_bfs", "RD → 1")]
for ax, (key, title) in zip(axes[1], metrics):
    y = series(key)
    ax.plot(x, y, marker="o", color=FULL, label=ARM)
    ax.plot([selected], [y[selected - 1]], marker="*", ms=11, color=FULL, mec="black", mew=0.6, ls="", zorder=6)
    ax.axhline(base[key], color=BASE, ls=":", lw=1.0, label="prefix_base (selected)")
    if key == "val_rd_bfs":
        ax.axhline(1.0, color="black", ls="-", lw=0.6, alpha=0.5)
    ax.set_title(title)
ax = axes[1, 3]
for key, label, color, marker in [
    ("val_degree_mmd_ratio", "degree", ORANGE, "o"),
    ("val_clustering_mmd_ratio", "clustering", GREEN, "s"),
    ("val_spectral_mmd_ratio", "spectral", VERMILION, "^"),
]:
    y = series(key)
    ax.plot(x, y, marker=marker, color=color, label=label)
    ax.plot([selected], [y[selected - 1]], marker="*", ms=11, color=color, mec="black", mew=0.6, ls="", zorder=6)
    ax.axhline(base[key], color=color, ls=":", lw=1.0)
ax.set_title("MMD ratios ↓ (dotted: prefix_base)")
ax.set_ylim(top=max(series("val_degree_mmd_ratio")) * 1.3)
ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
for ax in axes.ravel():
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
for ax in axes[1]:
    ax.set_xlabel("Epoch")
star = plt.Line2D([], [], marker="*", ms=11, color="white", mec="black", mew=0.6, ls="")
tri = plt.Line2D([], [], marker="v", ms=6, color="black", ls="")
handles, labels = axes[1, 0].get_legend_handles_labels()
axes[0, 0].legend(
    [*handles, star, tri],
    [*labels, f"selected epoch ({selected})", "loss minimum"],
    frameon=False,
    loc="upper right",
    handlelength=1.6,
)
save(fig, ROOT, "stage2_full_curves")
print("written", ROOT / "stage2_full_curves.png")
