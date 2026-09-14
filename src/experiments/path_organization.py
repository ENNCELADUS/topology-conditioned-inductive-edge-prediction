"""Train/V_val oracle exploration of L3 organization; never opens test data.

Run with ``python -m src.experiments.path_organization --output <directory>``.
All statistics exclude the query edge. These probes consume truth structure,
so their results are diagnostics, not endpoint-only prediction performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.artifacts import canonical_pair
from src.data.val_region import derive_val_region_split

BASE = ["degree_min", "degree_max", "common", "l3", "distance_category"]
LOCAL = ["triangles_min", "triangles_max", "neighbor_degree_min", "neighbor_degree_max"]
ORG = [
    "internal_nodes",
    "path_edges",
    "cycle_rank",
    "max_edge_share",
    "max_node_share",
    "disjoint_path_pair_fraction",
    "used_neighbors_min",
    "used_neighbors_max",
    "unused_neighbors_min",
    "unused_neighbors_max",
]


def pair_features(adj: dict[str, set[str]], u: str, v: str) -> dict[str, float]:
    """Measure symmetric organization of all simple L3 paths after edge deletion.

    Bottlenecks refer only to these paths, not global graph connectivity.
    Distance category is 2, 3, or 4 (meaning >=4 or disconnected).
    """
    if u == v:
        raise ValueError("Self-pairs are outside this distinct-endpoint exploration")
    nu, nv = adj[u] - {v}, adj[v] - {u}
    paths = [(a, b) for a in sorted(nu) for b in sorted(adj[a] & nv) if a != b]
    internal: Counter[str] = Counter()
    edges: Counter[tuple[str, str]] = Counter()
    internal_pairs: Counter[tuple[str, str]] = Counter()
    for a, b in paths:
        internal.update((a, b))
        internal_pairs[canonical_pair(a, b)] += 1
        edges.update(canonical_pair(*e) for e in ((u, a), (a, b), (b, v)))
    n = len(paths)
    total_pairs = n * (n - 1) // 2
    shared_pairs = sum(c * (c - 1) // 2 for c in internal.values())
    shared_pairs -= sum(c * (c - 1) // 2 for c in internal_pairs.values())
    used = sorted((len({a for a, _ in paths}), len({b for _, b in paths})))
    unused = sorted((len(nu) - len({a for a, _ in paths}), len(nv) - len({b for _, b in paths})))
    degrees = sorted((len(nu), len(nv)))
    triangles = sorted(sum(len(adj[a] & ns) for a in ns) / 2 for ns in (nu, nv))
    # Removing u-v decrements both endpoint degrees even when one is a neighbor
    # of the other endpoint's neighbors (e.g. triangles).
    removed = v in adj[u]
    means = sorted(
        sum(len(adj[a]) - int(removed and a in (u, v)) for a in ns) / max(1, len(ns))
        for ns in (nu, nv)
    )
    return dict(
        zip(
            BASE + LOCAL + ORG,
            [
                *degrees,
                len(nu & nv),
                n,
                2 if nu & nv else 3 if n else 4,
                *triangles,
                *means,
                len(internal),
                len(edges),
                len(edges) - len(internal) - 1 if n else 0,
                max(edges.values(), default=0) / max(1, n),
                max(internal.values(), default=0) / max(1, n),
                (total_pairs - shared_pairs) / total_pairs if total_pairs else 0,
                *used,
                *unused,
            ],
            strict=True,
        )
    )


def extract(
    graph: nx.Graph, positives: list[tuple[str, str]], negatives: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Extract every distinct non-self pair and retain row identities."""
    adj = {str(u): set(graph[u]) - {u} for u in graph}
    rows = []
    for label, pairs in ((1, positives), (0, negatives)):
        for u, v in sorted(set(pairs)):
            if u == v:
                continue
            if graph.has_edge(u, v) != bool(label):
                raise ValueError(f"Label contradicts graph for {(u, v)}")
            rows.append({"u": u, "v": v, "label": label, **pair_features(adj, u, v)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact inspectable row table."""
    if not rows:
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def organization_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tabulate label rates by exact L3 count and shared-edge status."""
    groups: dict[tuple[float, bool], list[int]] = defaultdict(list)
    for row in rows:
        if row["l3"] >= 2:
            groups[(row["l3"], row["max_edge_share"] == 1)].append(row["label"])
    return [
        {
            "l3": k[0],
            "shared_edge": k[1],
            "n": len(y),
            "positives": sum(y),
            "positive_rate": float(np.mean(y)),
        }
        for k, y in sorted(groups.items())
    ]


def matched_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe exact degree/common/L3-matched overlap without causal claims."""
    strata: dict[tuple[float, ...], dict[bool, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["l3"] >= 2:
            key = tuple(row[k] for k in BASE)
            strata[key][row["max_edge_share"] == 1].append(row["label"])
    supported = [g for g in strata.values() if min(len(g[False]), len(g[True])) >= 5]
    weights = [min(len(g[False]), len(g[True])) for g in supported]
    return {
        "minimum_per_organization": 5,
        "supported_strata": len(supported),
        "rows_in_supported_strata": sum(len(g[False]) + len(g[True]) for g in supported),
        "weighted_rate_difference_no_shared_minus_shared": float(
            np.average([np.mean(g[False]) - np.mean(g[True]) for g in supported], weights=weights)
        )
        if supported
        else None,
    }


def fit_probes(
    train: list[dict[str, Any]], val: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit fixed probes on training only; no validation tuning or early stopping."""
    y = np.array([r["label"] for r in train])
    yv = np.array([r["label"] for r in val])
    predictions = [{k: r[k] for k in ("u", "v", "label")} for r in val]
    metrics = []
    for family in ("logistic", "hist_gb"):
        for name, fields in (
            ("base", BASE),
            ("base_local", BASE + LOCAL),
            ("base_org", BASE + ORG),
            ("base_local_org", BASE + LOCAL + ORG),
        ):
            x = np.log1p([[r[k] for k in fields] for r in train])
            xv = np.log1p([[r[k] for k in fields] for r in val])
            model = (
                make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
                if family == "logistic"
                else HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=15,
                    min_samples_leaf=50,
                    l2_regularization=1,
                    early_stopping=False,
                    random_state=42,
                )
            )
            model.fit(x, y)
            p = model.predict_proba(xv)[:, 1]
            for row, probability in zip(predictions, p, strict=True):
                row[f"{family}_{name}"] = float(probability)
            for subset, mask in (
                ("all", np.ones(len(val), dtype=bool)),
                ("l3_ge_2", np.array([r["l3"] >= 2 for r in val])),
            ):
                metrics.append(
                    {
                        "family": family,
                        "features": name,
                        "subset": subset,
                        "n": int(mask.sum()),
                        "auroc": float(roc_auc_score(yv[mask], p[mask])),
                        "auprc": float(average_precision_score(yv[mask], p[mask])),
                        "log_loss": float(log_loss(yv[mask], p[mask])),
                    }
                )
    return metrics, predictions


def plot_results(output: Path, metrics: list[dict[str, Any]]) -> None:
    """Plot validation probes and unadjusted equal-L3 label rates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    names = ["base", "base_local", "base_org", "base_local_org"]
    labels = ["Counts", "+ Local", "+ Organization", "+ Both"]
    for family, color in (("logistic", "#267a95"), ("hist_gb", "#c4772e")):
        selected = {
            m["features"]: m for m in metrics if m["family"] == family and m["subset"] == "all"
        }
        axes[0].plot(labels, [selected[n]["auprc"] for n in names], "o-", label=family, color=color)
    axes[0].set(title="Train-fitted probes on V_val", ylabel="Validation AUPRC")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(frameon=False)
    for ax, split in zip(axes[1:], ("train", "val"), strict=True):
        with (output / f"{split}_l3_groups.csv").open() as handle:
            groups = list(csv.DictReader(handle))
        for shared, color, label in (
            ("False", "#267a95", "No universal L3 edge"),
            ("True", "#c4772e", "Universal L3 edge"),
        ):
            chosen = [r for r in groups if r["shared_edge"] == shared and int(r["l3"]) <= 6]
            ax.plot(
                [int(r["l3"]) for r in chosen],
                [float(r["positive_rate"]) for r in chosen],
                "o-",
                color=color,
                label=label,
            )
        ax.set(
            title=f"{split}: equal L3 count, unadjusted",
            xlabel="Exact L3 count",
            ylabel="Positive fraction",
            ylim=(0, 1),
            xticks=range(2, 7),
        )
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(output / "exploration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Rebuild the current split using train-side files and write diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / "benchmark_2025_neurips" / "breadth_first"
    with (root / "split.pkl").open("rb") as handle:
        nodes = pickle.load(handle)["train"]
    with (root / "train_graph.pkl").open("rb") as handle:
        substrate = pickle.load(handle)
    negatives = []
    for name in ("train_edges.txt", "val_edges.txt"):
        for line in (root / name).read_text().splitlines():
            u, v, label = line.split()
            if int(label) == 0:
                negatives.append(tuple(sorted((u, v))))
    split = derive_val_region_split(nodes, substrate.edges, negatives, frozenset())
    manifest = json.loads((args.data_root / "val_region/breadth_first.json").read_text())
    if set(manifest["v_val"]) != split.v_val or set(manifest["train_nodes"]) != split.train_nodes:
        raise ValueError("Derived split does not match the current manifest")
    if set(map(tuple, manifest["val_negatives"])) != set(split.val_negatives):
        raise ValueError("Validation negatives differ from manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    train = extract(
        split.build_training_graph(), list(split.training_positives), list(split.training_negatives)
    )
    val = extract(split.build_g_val_simple(), list(split.val_positives), list(split.val_negatives))
    for name, rows in (("train", train), ("val", val)):
        write_csv(args.output / f"{name}_features.csv", rows)
        write_csv(args.output / f"{name}_l3_groups.csv", organization_groups(rows))
    with threadpool_limits(limits=2):
        metrics, predictions = fit_probes(train, val)
    write_csv(args.output / "probe_metrics.csv", metrics)
    write_csv(args.output / "val_predictions.csv", predictions)
    plot_results(args.output, metrics)
    summary = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": 42,
        "train_nodes": len(split.train_nodes),
        "val_nodes": len(split.v_val),
        "region_seeds": split.region_seeds,
        "features": {"base": BASE, "local": LOCAL, "org": ORG},
        "protocol": "Query-edge-deleted truth graphs; fixed benchmark training negatives; "
        "official val_cls negatives; self-pairs excluded; no test files read; "
        "fixed train-fitted probes, no validation tuning; oracle diagnostic only.",
        "splits": {
            name: {
                "rows": len(rows),
                "positives": sum(r["label"] for r in rows),
                "l3_ge_2": sum(r["l3"] >= 2 for r in rows),
                "self_positives_excluded": sum(u == v for u, v in positives),
                "matched": matched_summary(rows),
            }
            for name, rows, positives in (
                ("train", train, split.training_positives),
                ("val", val, split.val_positives),
            )
        },
        "metrics": metrics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
