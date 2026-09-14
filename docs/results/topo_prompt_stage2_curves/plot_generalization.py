"""Predicted vs true structural statistics of the published coord_gen_full generator.

Reads coord_gen_full/coord_fit_universes.npz (written on H20 by coord_fit_universes.py: the
generator's own predictions and the true coordinates on label-balanced samples of training,
V_val and test rows). Top strip: Pearson ρ per statistic on the three universes plus the
distance-class accuracy. Grid: predicted against true value on the unseen test rows for every
continuous statistic (L3PPI Fig. 5(B) style), ρ annotated for train / V_val / test; the last
panels show the distance-class confusion on V_val and test rows.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parents[2]))
from matplotlib import gridspec  # noqa: E402
from paper_plot_style import UNIVERSE_COLORS, plt, save  # noqa: E402
from src.data.struct_coords import CONTEXT_NAMES, ENDPOINT_NAMES, FIELD_SLICES, RELATION_NAMES  # noqa: E402

data = np.load(ROOT / "coord_gen_full" / "coord_fit_universes.npz")
ORDER = ["train", "val", "test"]
LABELS = {"train": "training rows", "val": "V_val rows (unseen)", "test": "test rows (unseen)"}
MARKERS = {"train": "o", "val": "s", "test": "^"}
DIST_CLASSES = ["2", "3", "≥4", "∞", "self"]
RNG = np.random.default_rng(0)
N_SCATTER = 2000

eu, ev = FIELD_SLICES["endpoint_u"], FIELD_SLICES["endpoint_v"]
n_rel_cont = len(RELATION_NAMES) - 4  # the four distance one-hots are categorical
rel_cols = list(range(FIELD_SLICES["relation"].start, FIELD_SLICES["relation"].start + n_rel_cont))
ctx_cols = list(range(FIELD_SLICES["context"].start, FIELD_SLICES["context"].stop))


SHORT = {
    "log1p_degree": "degree", "clustering": "clustering", "log1p_triangles": "triangles",
    "log1p_open_wedges": "open wedges", "log1p_two_hop": "two-hop reach",
    "log1p_mean_neighbor_degree": "mean nbr degree", "return_2": "return 2", "return_3": "return 3",
    "return_4": "return 4", "log1p_common": "common nbrs", "jaccard": "Jaccard", "log1p_l3": "L3 paths",
    "walk_2": "walk 2", "walk_3": "walk 3", "walk_4": "walk 4", "walk_5": "walk 5",
    "log1p_hop1_union": "hop-1 union", "log1p_shell": "shell", "shell_shared_fraction": "shell shared frac.",
    "log1p_cross_ring": "cross ring", "l3_density": "L3 density",
}


def pretty(name: str) -> str:
    return SHORT.get(name, name.replace("_", " "))


# (field, label, column getter) for every continuous statistic; endpoints pool u and v.
STATS = []
for j, name in enumerate(ENDPOINT_NAMES):
    STATS.append(("endpoint", pretty(name), (eu.start + j, ev.start + j)))
for j, name in zip(rel_cols, RELATION_NAMES[:n_rel_cont]):
    STATS.append(("relation", pretty(name), (j,)))
for j, name in zip(ctx_cols, CONTEXT_NAMES):
    STATS.append(("context", pretty(name), (j,)))


def values(universe: str, cols: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.concatenate([data[f"{universe}_pred_raw"][:, c] for c in cols])
    true = np.concatenate([data[f"{universe}_true_raw"][:, c] for c in cols])
    return pred, true


def rho(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


rows_by_field = {"endpoint": 9, "relation": n_rel_cont, "context": len(CONTEXT_NAMES)}
n_cols = 9
fig = plt.figure(figsize=(9.6, 8.4))
gs = gridspec.GridSpec(4, n_cols, figure=fig, height_ratios=[1.9, 1, 1, 1], hspace=0.65, wspace=0.5,
                       left=0.06, right=0.995, top=0.985, bottom=0.05)

# --- Top strip: Pearson ρ per statistic, three universes.
top = fig.add_subplot(gs[0, :])
xs = np.arange(len(STATS) + 1)
for name in ORDER:
    rhos = [rho(*values(name, cols)) for _, _, cols in STATS]
    acc = float((data[f"{name}_dist_pred"] == data[f"{name}_dist_true"]).mean())
    top.plot(xs[:-1], rhos, marker=MARKERS[name], ms=4.5, ls="-", lw=0.8, color=UNIVERSE_COLORS[name], label=LABELS[name])
    top.plot([xs[-1]], [acc], marker=MARKERS[name], ms=4.5, ls="", color=UNIVERSE_COLORS[name])
    majority = float(np.bincount(data[f"{name}_dist_true"]).max() / len(data[f"{name}_dist_true"]))
    top.plot([xs[-1] - 0.25, xs[-1] + 0.25], [majority] * 2, color=UNIVERSE_COLORS[name], lw=1.0, ls=":")
top.axhline(0, color="black", lw=0.7)
bounds = np.cumsum([rows_by_field["endpoint"], rows_by_field["relation"]])
for b in bounds:
    top.axvline(b - 0.5, color="black", lw=0.6, alpha=0.6)
top.axvline(len(STATS) - 0.5, color="black", lw=0.6, alpha=0.6)
top.set_xticks(xs, [label for _, label, _ in STATS] + ["distance class (acc.)"], rotation=40, ha="right", fontsize=6.5)
top.set_ylabel("Pearson ρ (pred, true)")
top.set_ylim(-0.25, 1.0)
top.set_xlim(-0.7, len(STATS) + 0.7)
for x, text in [(4, "endpoint (u, v pooled)"), (bounds[0] + n_rel_cont / 2 - 0.5, "relation"), (bounds[1] + 2, "context")]:
    top.text(x, 0.96, text, ha="center", va="top", fontsize=7.5, style="italic")
top.legend(frameon=False, loc="lower left", ncol=3, fontsize=7, handlelength=1.4, columnspacing=1.0)

# --- Scatter grid on the unseen test rows.
positions = [(1, j) for j in range(9)] + [(2, j) for j in range(n_rel_cont)] + [(3, j) for j in range(len(CONTEXT_NAMES))]
for (row, col), (field, label, cols) in zip(positions, STATS):
    ax = fig.add_subplot(gs[row, col])
    pred, true = values("test", cols)
    take = RNG.choice(len(pred), size=min(N_SCATTER, len(pred)), replace=False)
    ax.scatter(true[take], pred[take], s=2.5, alpha=0.35, color=UNIVERSE_COLORS["test"], linewidths=0, rasterized=True)
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.6, ls="--")
    rhos = " | ".join(f"{rho(*values(n, cols)):.2f}" for n in ORDER)
    ax.set_title(f"{label}\nρ {rhos}", fontsize=6.3, pad=2, linespacing=1.1)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.grid(False)
    if col == 0:
        ax.set_ylabel("predicted", fontsize=6)
    if row == 3 or (row == 2 and col >= len(CONTEXT_NAMES)):
        ax.set_xlabel("true", fontsize=6, labelpad=1)

# Distance-class confusion (row-normalised) on the unseen universes.
for k, name in enumerate(["val", "test"]):
    ax = fig.add_subplot(gs[2, n_rel_cont + k])
    true = data[f"{name}_dist_true"].astype(int)
    pred = data[f"{name}_dist_pred"].astype(int)
    matrix = np.zeros((5, 5))
    for t, p in zip(true, pred):
        matrix[t, p] += 1
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            if matrix[i].sum() > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=4.6, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(5), DIST_CLASSES, fontsize=5)
    ax.set_yticks(range(5), DIST_CLASSES, fontsize=5)
    ax.tick_params(length=0, pad=1)
    ax.grid(False)
    acc = float((true == pred).mean())
    ax.set_title(f"distance class, {name.replace('val', 'V_val')}\nacc. {acc:.2f}", fontsize=6.3, pad=2, linespacing=1.1)
    ax.set_xlabel("predicted", fontsize=6, labelpad=1)
    ax.set_ylabel("true", fontsize=6)

save(fig, ROOT, "generator_generalization")
print("written", ROOT / "generator_generalization.png")
for name in ORDER:
    rhos = {label: round(rho(*values(name, cols)), 3) for _, label, cols in STATS}
    acc = float((data[f"{name}_dist_pred"] == data[f"{name}_dist_true"]).mean())
    print(name, "distance acc", round(acc, 3), rhos)
