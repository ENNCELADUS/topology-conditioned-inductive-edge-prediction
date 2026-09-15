"""Per-statistic generalisation audit of a `coord_fit_universes.npz` dump.

Recomputes, on CPU from the saved predictions, the tables of
`docs/results/topo_prompt_stage2_verdict/README.md` (Part A): within-label and per-label
Pearson correlation, the recalibration ceiling, the spread/offset decomposition,
the label AUROC of predicted against true coordinates, the variance spectrum of
the predicted coordinate vector, and the distance head's per-class recall.

Usage:
    PYTHONPATH=. .venv/bin/python docs/results/topo_prompt_stage2_curves/coord_fit_analysis.py \
        --npz docs/results/topo_prompt_stage2_curves/coord_gen_full/coord_fit_universes.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.struct_coords import CONTEXT_NAMES, ENDPOINT_NAMES, FIELD_SLICES, RELATION_NAMES
from src.model.egostitch.classifier.coord_gen import FIELD_CONTINUOUS_INDEX

UNIVERSES = ("train", "val", "test")
NAMES = (
    [f"endpoint.{n}" for n in ENDPOINT_NAMES]
    + [f"relation.{n}" for n in RELATION_NAMES[: len(FIELD_CONTINUOUS_INDEX["relation"])]]
    + [f"context.{n}" for n in CONTEXT_NAMES]
)
DISTANCE_CLASSES = ("d=2", "d=3", "d>=4", "d=inf", "self")


def stack(dump: np.lib.npyio.NpzFile, universe: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardised (prediction, truth, label) columns for one statistic.

    Endpoint statistics pool the u and v columns, as the probe's own per-coordinate
    numbers do; relation and context statistics are single columns.

    Args:
        dump: The loaded `coord_fit_universes.npz`.
        universe: One of `train`, `val`, `test`.
        name: A dotted statistic name from `NAMES`.

    Returns:
        Prediction, truth and binary label arrays of equal length.
    """
    pred, true = dump[f"{universe}_pred_std"], dump[f"{universe}_true_std"]
    labels = dump[f"{universe}_labels"].astype(int)
    field, stat = name.split(".", 1)
    if field == "endpoint":
        j = ENDPOINT_NAMES.index(stat)
        cu, cv = FIELD_SLICES["endpoint_u"].start + j, FIELD_SLICES["endpoint_v"].start + j
        return (
            np.concatenate([pred[:, cu], pred[:, cv]]).astype(float),
            np.concatenate([true[:, cu], true[:, cv]]).astype(float),
            np.concatenate([labels, labels]),
        )
    if field == "relation":
        col = FIELD_CONTINUOUS_INDEX["relation"][RELATION_NAMES.index(stat)]
    else:
        col = FIELD_CONTINUOUS_INDEX["context"][CONTEXT_NAMES.index(stat)]
    return pred[:, col].astype(float), true[:, col].astype(float), labels


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN when either side is constant."""
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def label_centred(v: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return `v` with each label class's mean removed."""
    out = v.astype(float).copy()
    for k in (0, 1):
        out[labels == k] -= v[labels == k].mean()
    return out


def eta_squared(v: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of the variance of `v` explained by the binary label."""
    if v.std() < 1e-12:
        return float("nan")
    total = ((v - v.mean()) ** 2).sum()
    between = sum(len(v[labels == k]) * (v[labels == k].mean() - v.mean()) ** 2 for k in (0, 1))
    return float(between / total)


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    """Return the coefficient of determination of `pred` against `true`."""
    sst = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ((pred - true) ** 2).sum() / sst) if sst > 0 else float("nan")


def effective_dimension(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the explained-variance spectrum and its perplexity for a centred matrix."""
    centred = matrix - matrix.mean(0)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    spectrum = spectrum / spectrum.sum()
    return spectrum, float(np.exp(-(spectrum * np.log(spectrum + 1e-12)).sum()))


def main() -> None:
    """Print every table of the generalisation audit as markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("docs/results/topo_prompt_stage2_curves/coord_gen_full/coord_fit_universes.npz"),
    )
    args = parser.parse_args()
    dump = np.load(args.npz)

    print("## Correlation (within-label unless stated)\n")
    print("| statistic | rho train | rho val | rho test | within train | within val | within test "
          "| pos test | neg test | rho^2 test |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        cells = []
        within = {}
        for universe in UNIVERSES:
            pred, true, labels = stack(dump, universe, name)
            cells.append(pearson(pred, true))
            within[universe] = pearson(label_centred(pred, labels), label_centred(true, labels))
        pred, true, labels = stack(dump, "test", name)
        pos = pearson(pred[labels == 1], true[labels == 1])
        neg = pearson(pred[labels == 0], true[labels == 0])
        row = cells + [within[u] for u in UNIVERSES] + [pos, neg, within["test"] ** 2]
        print(f"| `{name}` | " + " | ".join(f"{v:.2f}" for v in row) + " |")

    print("\n## Spread, offset and R^2\n")
    print("| statistic | sd(pred) tr/val/test | sd(true) tr/val/test | offset val | offset test "
          "| R^2 val | R^2 test |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name in NAMES:
        sd_pred, sd_true, offset, r2 = [], [], {}, {}
        for universe in UNIVERSES:
            pred, true, _ = stack(dump, universe, name)
            sd_pred.append(pred.std())
            sd_true.append(true.std())
            offset[universe] = pred.mean() - true.mean()
            r2[universe] = r_squared(pred, true)
        print(
            f"| `{name}` | " + " / ".join(f"{v:.2f}" for v in sd_pred)
            + " | " + " / ".join(f"{v:.2f}" for v in sd_true)
            + f" | {offset['val']:+.2f} | {offset['test']:+.2f} | {r2['val']:.2f} | {r2['test']:.2f} |"
        )

    print("\n## AUROC against the edge label\n")
    print("| statistic | val true | val predicted | test true | test predicted |")
    print("|---|---:|---:|---:|---:|")
    for name in NAMES:
        cells = []
        for universe in ("val", "test"):
            pred, true, labels = stack(dump, universe, name)
            cells += [
                roc_auc_score(labels, true) if true.std() > 1e-12 else float("nan"),
                roc_auc_score(labels, pred) if pred.std() > 1e-12 else float("nan"),
            ]
        print(f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in cells) + " |")

    print("\n## Variance spectrum of the 30 continuous coordinates\n")
    print("| universe | source | PC1 | PC1-3 | effective dim | eta^2(PC1, label) |")
    print("|---|---|---:|---:|---:|---:|")
    columns = (
        FIELD_CONTINUOUS_INDEX["endpoint"] + FIELD_CONTINUOUS_INDEX["relation"] + FIELD_CONTINUOUS_INDEX["context"]
    )
    for universe in UNIVERSES:
        labels = dump[f"{universe}_labels"].astype(int)
        for source in ("pred", "true"):
            matrix = dump[f"{universe}_{source}_std"][:, columns].astype(float)
            spectrum, effective = effective_dimension(matrix)
            centred = matrix - matrix.mean(0)
            pc1 = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
            print(
                f"| {universe} | {'predicted' if source == 'pred' else 'true'} | {spectrum[0]:.3f} "
                f"| {spectrum[:3].sum():.3f} | {effective:.2f} | {eta_squared(pc1, labels):.3f} |"
            )

    print("\n## Distance head\n")
    print("| universe | accuracy | majority | " + " | ".join(f"recall {c}" for c in DISTANCE_CLASSES)
          + " | rows per true class |")
    print("|---|---:|---:|" + "---:|" * (len(DISTANCE_CLASSES) + 1))
    for universe in UNIVERSES:
        pred = dump[f"{universe}_dist_pred"].astype(int)
        true = dump[f"{universe}_dist_true"].astype(int)
        counts = np.bincount(true, minlength=len(DISTANCE_CLASSES))
        recalls = [
            f"{float((pred[true == c] == c).mean()):.3f}" if counts[c] else "-"
            for c in range(len(DISTANCE_CLASSES))
        ]
        print(
            f"| {universe} | {float((pred == true).mean()):.3f} | {counts.max() / len(true):.3f} | "
            + " | ".join(recalls)
            + " | " + "/".join(str(int(c)) for c in counts) + " |"
        )


if __name__ == "__main__":
    main()
