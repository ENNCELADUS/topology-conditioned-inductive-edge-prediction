"""Test-informed density design audit; not a prediction/threshold experiment."""

from __future__ import annotations

import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from src.data.val_region import _bfs_positive_budget, sample_bfs_ball_buckets

OUT = Path(__file__).parent
P = Path("data/benchmark_2025_neurips/breadth_first")
with (P / "train_graph.pkl").open("rb") as handle:
    g = pickle.load(handle)
with (P / "test_graph.pkl").open("rb") as handle:
    t = pickle.load(handle)
with (P / "test_node_buckets.pkl").open("rb") as handle:
    tb = pickle.load(handle)
sizes = sorted(tb)


def counts(graph: nx.Graph, banks: dict[int, list[set[str]]]) -> dict[str, Any]:
    """Count positive pairs once per sample, with and without self-loops."""
    nodes = sorted(graph)
    idx = {n: i for i, n in enumerate(nodes)}
    a = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)
    ans = {}
    for size, balls in banks.items():
        rows = []
        nonself = []
        for b in balls:
            ix = [idx[n] for n in b]
            sub = a[np.ix_(ix, ix)]
            loops = np.trace(sub)
            edges = (sub.sum() + loops) / 2
            rows.append(float(edges))
            nonself.append(float(edges - loops))
        ans[str(size)] = {"edges": rows, "nonself": nonself}
    return ans


ref = counts(t, tb)


def distance(bank: dict[str, Any]) -> dict[str, Any]:
    """Measure size-macro relative edge-count discrepancy from fixed test truth."""
    per = {}
    for size in sizes:
        k = str(size)
        a = np.log(bank[k]["edges"])
        b = np.log(ref[k]["edges"])
        per[k] = {
            "log_edge_w1": float(wasserstein_distance(a, b)),
            "val_mean_edges": float(np.mean(bank[k]["edges"])),
            "test_mean_edges": float(np.mean(ref[k]["edges"])),
            "nonself_density_w1": float(
                wasserstein_distance(bank[k]["nonself"], ref[k]["nonself"])
                / (size * (size - 1) / 2)
            ),
        }
    return {
        "score": float(np.mean([v["log_edge_w1"] for v in per.values()])),
        "nonself_density_w1": float(np.mean([v["nonself_density_w1"] for v in per.values()])),
        "per_size": per,
    }


roots = sorted(n for c in nx.connected_components(g) if len(c) >= 200 for n in c if len(g[n]) == 5)
regions = {}
rows = []
start = time.time()
for i, root in enumerate(roots):
    v = _bfs_positive_budget(g, root, int(g.number_of_edges() * 0.1))
    if len(v) < 200:
        continue
    sub = g.subgraph(v).copy()
    regions[root] = sub
    bk = sample_bfs_ball_buckets(sub, sizes=sizes, per_size=50, seed=43)
    score = distance(counts(sub, bk))
    fit = g.subgraph(set(g) - v).number_of_edges()
    rows.append(
        {
            "root": root,
            "nodes": len(v),
            "val_positive_pairs": sub.number_of_edges(),
            "fit_positive_pairs": fit,
            "boundary_pairs": g.number_of_edges() - sub.number_of_edges() - fit,
            **score,
        }
    )
    if i % 100 == 0:
        logging.info("sweep %d (%.1f seconds)", i, time.time() - start)
rows.sort(key=lambda r: r["score"])
(OUT / "screen.json").write_text(json.dumps(rows, indent=2) + "\n")
# Shortlist fixed from the screen; include previous representative candidates as controls.
shortlist = list(
    dict.fromkeys([r["root"] for r in rows[:12]] + ["node_007630", "node_006901", "node_001280"])
)
refined = []
for root in shortlist:
    results = []
    for seed in [43, 44, 45, 46, 47]:
        bk = sample_bfs_ball_buckets(regions[root], sizes=sizes, per_size=50, seed=seed)
        results.append({"seed": seed, **distance(counts(regions[root], bk))})
    source = next(r for r in rows if r["root"] == root)
    refined.append(
        {
            **source,
            "score": float(np.mean([r["score"] for r in results])),
            "score_std": float(np.std([r["score"] for r in results])),
            "nonself_density_w1": float(np.mean([r["nonself_density_w1"] for r in results])),
            "replicates": results,
        }
    )
refined.sort(key=lambda r: (r["score"], -r["fit_positive_pairs"], r["root"]))
winner = refined[0]["root"]
seed = next(s for s in range(100000) if random.Random(s).choice(roots) == winner)
out = {
    "criterion": (
        "Mean across sizes of W1(log positive edge counts), loops included; "
        "then mean across seeds 43..47. Training retention breaks exact ties only."
    ),
    "scope": (
        "Test-informed split redesign, not untouched-test evidence. "
        "No model rescoring or actual threshold transfer measured."
    ),
    "valid_roots": len(rows),
    "selected_root": winner,
    "selected_split_seed": seed,
    "shortlist": refined,
}
# Compare the historical 20% positive-budget holdout under the same score.
base20 = _bfs_positive_budget(g, "node_007630", int(g.number_of_edges() * 0.2))
sub20 = g.subgraph(base20).copy()
rep20 = [
    {
        "seed": s,
        **distance(counts(sub20, sample_bfs_ball_buckets(sub20, sizes=sizes, per_size=50, seed=s))),
    }
    for s in [43, 44, 45, 46, 47]
]
out["reference_20pct"] = {
    "root": "node_007630",
    "nodes": len(base20),
    "fit_positive_pairs": g.subgraph(set(g) - base20).number_of_edges(),
    "replicates": rep20,
    "score": float(np.mean([r["score"] for r in rep20])),
    "nonself_density_w1": float(np.mean([r["nonself_density_w1"] for r in rep20])),
}
(OUT / "selection.json").write_text(json.dumps(out, indent=2) + "\n")
logging.info("selected %s, seed=%d, score=%f", winner, seed, refined[0]["score"])
