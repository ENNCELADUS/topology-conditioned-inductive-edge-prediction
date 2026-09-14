"""Sampler-aligned endpoint-context probes on an existing frozen content score.

The fixed design is in docs/results/path_organization_train_val/content_probe_protocol.md.
Prepare emits scoring pairs; analyze requires their production scorer artifact.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.artifacts import canonical_pair
from src.data.pairs import NegativeSampler
from src.data.training_sampler import enumerate_edge_stream

MISSING = {"node_004764", "node_007050"}
ARMS = {
    "content": [0],
    "content_degree": [0, 1, 2],
    "content_degree_triangles": [0, 1, 2, 3, 4],
    "content_degree_neighbor": [0, 1, 2, 5, 6],
    "content_degree_both": list(range(7)),
    "content_degree_both_roles": list(range(11)),
    "context_only": list(range(1, 11)),
}


def context_features(graph: nx.Graph, pairs: list[tuple[str, str]]) -> NDArray[np.float64]:
    """Return query-edge-deleted margins plus endpoint-assignment-preserving products."""
    g = graph.copy()
    g.remove_edges_from(nx.selfloop_edges(g))
    adj = {u: set(g[u]) for u in g}
    degree = dict(g.degree())
    triangles = nx.triangles(g)
    neighbor_sum = {u: sum(degree[v] for v in g[u]) for u in g}
    result = np.empty((len(pairs), 10), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        if u == v:
            raise ValueError("Self-pairs are outside this probe")
        edge = int(v in adj[u])
        common = len(adj[u] & adj[v]) if edge else 0
        d = [degree[u] - edge, degree[v] - edge]
        t = [triangles[u] - common, triangles[v] - common]
        n = [
            (neighbor_sum[u] - edge * degree[v]) / max(1, d[0]),
            (neighbor_sum[v] - edge * degree[u]) / max(1, d[1]),
        ]
        ld, lt, ln = np.log1p([d, t, n])
        result[i] = [
            *sorted(ld),
            *sorted(lt),
            *sorted(ln),
            np.dot(ld, lt),
            np.dot(ld, ln),
            np.dot(lt, ln),
            np.sum(ld * lt * ln),
        ]
    return result


def sampled_rows(graph: nx.Graph, seed: int) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Use the production sampler, then restrict to distinct featureful endpoints."""
    positives = sorted(canonical_pair(u, v) for u, v in graph.edges if not ({u, v} & MISSING))
    simple = graph.copy()
    simple.remove_edges_from(nx.selfloop_edges(simple))
    sampler = NegativeSampler(
        sorted(set(graph) - MISSING), dict(simple.degree()), frozenset(positives)
    )
    rows = enumerate_edge_stream(
        positives, sampler, negative_ratio=5, seed=seed, epoch=1, rank=0, world_size=1
    )
    rows = sorted((u, v, y) for u, v, y in rows if u != v)
    return [(u, v) for u, v, _ in rows], np.array([y for _, _, y in rows], dtype=np.int8)


def prepare(output: Path) -> None:
    """Persist aligned datasets and a unique scoring union without fitting probes."""
    manifest = json.loads(Path("data/val_region/breadth_first.json").read_text())
    holdouts = json.loads((output / "holdouts.json").read_text())
    with Path("data/benchmark_2025_neurips/breadth_first/train_graph.pkl").open("rb") as f:
        substrate = pickle.load(f)
    train_nodes, val_nodes = set(manifest["train_nodes"]), set(manifest["v_val"])
    if train_nodes & val_nodes:
        raise ValueError("Overlapping main universes")
    specifications = [("main_fit", train_nodes, True), ("main_eval", val_nodes, False)]
    for seed, nodes in holdouts.items():
        held = set(nodes)
        if not held <= train_nodes or len(held) != len(train_nodes) // 5:
            raise ValueError("Holdout membership mismatch")
        specifications += [
            (f"h{seed}_fit", train_nodes - held, True),
            (f"h{seed}_eval", held, False),
        ]
    all_rows: dict[tuple[str, str], int] = {}
    datasets: dict[str, Any] = {}
    for name, nodes, fit in specifications + [("official_eval", val_nodes, False)]:
        graph = substrate.subgraph(nodes).copy()
        if name == "official_eval":
            rows = [
                (canonical_pair(u, v), y)
                for key, y in (("val_positives", 1), ("val_negatives", 0))
                for u, v in manifest[key]
                if u != v and not ({u, v} & MISSING)
            ]
            pairs = [p for p, _ in rows]
            labels = np.array([y for _, y in rows], dtype=np.int8)
        else:
            pairs, labels = sampled_rows(graph, 20260914 if fit else 20260915)
        for pair, label in zip(pairs, labels.tolist(), strict=True):
            if pair in all_rows and all_rows[pair] != label:
                raise ValueError("Inconsistent labels in score union")
            all_rows[pair] = label
        features = context_features(graph, pairs)
        simple = graph.copy()
        simple.remove_edges_from(nx.selfloop_edges(simple))
        communities = nx.community.louvain_communities(simple, seed=42, resolution=1)
        community = {u: i for i, group in enumerate(communities) for u in group}
        np.savez_compressed(
            output / f"{name}.npz",
            pairs=np.array(pairs),
            label=labels,
            context=features,
            community=np.array([[community[u], community[v]] for u, v in pairs]),
        )
        datasets[name] = {
            "nodes": len(nodes),
            "rows": len(pairs),
            "positives": int(labels.sum()),
            "communities": len(communities),
        }
    with (output / "score_pairs.tsv").open("w") as f:
        for (u, v), label in sorted(all_rows.items()):
            f.write(f"{u}\t{v}\t{label}\n")
    (output / "datasets.json").write_text(json.dumps(datasets, indent=2) + "\n")


def balanced_loss(y: NDArray[Any], z: NDArray[Any]) -> float:
    """Balanced BCE from logits without probability clipping."""
    losses = np.logaddexp(0, z) - y * z
    return float(0.5 * (losses[y == 1].mean() + losses[y == 0].mean()))


def paired_intervals(
    y: NDArray[Any], differences: NDArray[Any], groups: NDArray[Any]
) -> list[list[float]]:
    """Paired node/group multiplicity sensitivity intervals; never resample rows."""
    _, inverse = np.unique(groups, return_inverse=True)
    indices = inverse.reshape(groups.shape)
    count = int(indices.max()) + 1
    rng = np.random.default_rng(771)
    draws = []
    same = indices[:, 0] == indices[:, 1]
    for _ in range(1000):
        multiplicity = rng.multinomial(count, np.full(count, 1 / count))
        w = multiplicity[indices[:, 0]] * np.where(same, 1, multiplicity[indices[:, 1]])
        wp, wn = w * (y == 1), w * (y == 0)
        if wp.sum() == 0 or wn.sum() == 0:
            continue
        draws.append(0.5 * (wp @ differences / wp.sum() + wn @ differences / wn.sum()))
    if len(draws) < 900:
        raise ValueError("Too few nondegenerate group resamples")
    return [[float(v) for v in row] for row in np.quantile(draws, [0.025, 0.975], axis=0).T]


def analyze(output: Path, scores: Path) -> None:
    """Fit fixed paired probes and save predictions and dependence-aware summaries."""
    with np.load(scores, allow_pickle=False) as artifact:
        node_ids = artifact["node_ids"].tolist()
        score_map = {
            canonical_pair(node_ids[u], node_ids[v]): (float(z), int(y))
            for u, v, z, y in zip(
                artifact["u_idx"],
                artifact["v_idx"],
                artifact["logit"],
                artifact["label"],
                strict=True,
            )
        }
        metadata = json.loads(str(artifact["meta"].item()))
    if not all(np.isfinite(z) for z, _ in score_map.values()):
        raise ValueError("Nonfinite content logits")
    if metadata.get("checkpoint_id") != "614003b9ba3a8d53":
        raise ValueError(f"Unexpected checkpoint: {metadata.get('checkpoint_id')}")

    def load(name: str) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        with np.load(output / f"{name}.npz") as a:
            pairs, labels = a["pairs"], a["label"]
            values = [score_map[tuple(pair)] for pair in pairs.tolist()]
            if not np.array_equal([v[1] for v in values], labels):
                raise ValueError("Score label join mismatch")
            return (
                np.column_stack(([v[0] for v in values], a["context"])),
                labels,
                pairs,
                a["community"],
            )

    result: dict[str, Any] = {"content_metadata": metadata, "fits": {}, "evaluations": {}}
    for fit_name, eval_names in (
        ("main_fit", ["main_eval", "official_eval"]),
        *((f"h{s}_fit", [f"h{s}_eval"]) for s in (1101, 1102, 1103)),
    ):
        x, y, _, _ = load(fit_name)
        models = {}
        fit_details = {}
        for name, columns in ARMS.items():
            scaler = StandardScaler().fit(x[:, columns])
            model = LogisticRegression(C=1, max_iter=1000)
            model.fit(scaler.transform(x[:, columns]), y, sample_weight=np.where(y == 1, 5, 1))
            models[name] = scaler, model
            fit_details[name] = {
                "iterations": model.n_iter_.tolist(),
                "coefficient": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
            }
        result["fits"][fit_name] = fit_details
        for eval_name in eval_names:
            xv, yv, pairs, community = load(eval_name)
            logits = {"raw_content": xv[:, 0]}
            for name, (scaler, model) in models.items():
                logits[name] = model.decision_function(scaler.transform(xv[:, ARMS[name]]))
            losses = {name: np.logaddexp(0, z) - yv * z for name, z in logits.items()}
            references = {
                "content": "raw_content",
                "content_degree": "content",
                "content_degree_triangles": "content_degree",
                "content_degree_neighbor": "content_degree",
                "content_degree_both": "content_degree",
                "content_degree_both_roles": "content_degree_both",
                "context_only": "content",
                "both_vs_content": "content",
                "roles_vs_content": "content",
            }
            targets = {
                "both_vs_content": "content_degree_both",
                "roles_vs_content": "content_degree_both_roles",
            }
            names = list(references)
            differences = np.column_stack(
                [losses[references[n]] - losses[targets.get(n, n)] for n in names]
            )
            ci_node = paired_intervals(yv, differences, pairs)
            ci_group = paired_intervals(yv, differences, community)
            metrics = {}
            for name, z in logits.items():
                metrics[name] = {
                    "balanced_bce": balanced_loss(yv, z),
                    "auroc": float(roc_auc_score(yv, z)),
                    "auprc": float(average_precision_score(yv, z)),
                }
            contrasts = {}
            for i, name in enumerate(names):
                ref = references[name]
                target = targets.get(name, name)
                contrasts[name] = {
                    "target": target,
                    "reference": ref,
                    "balanced_bce_gain": metrics[ref]["balanced_bce"]
                    - metrics[target]["balanced_bce"],
                    "auroc_gain": metrics[target]["auroc"] - metrics[ref]["auroc"],
                    "auprc_gain": metrics[target]["auprc"] - metrics[ref]["auprc"],
                    "node_interval": ci_node[i],
                    "community_interval": ci_group[i],
                }
            result["evaluations"][eval_name] = {"metrics": metrics, "contrasts": contrasts}
            np.savez_compressed(
                output / f"{eval_name}_predictions.npz",
                pairs=pairs,
                label=yv,
                arm_names=np.array(list(logits)),
                logits=np.column_stack(list(logits.values())),
            )
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    """Prepare or analyze the predeclared experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "prepare":
            prepare(args.output)
        else:
            if args.scores is None:
                parser.error("analyze requires --scores")
            analyze(args.output, args.scores)
            plot_summary(args.output)


def plot_summary(output: Path) -> None:
    """Plot the predeclared marginal gains and community sensitivity intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((output / "results.json").read_text())["evaluations"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), layout="constrained")
    arms = [
        "content_degree_triangles",
        "content_degree_neighbor",
        "content_degree_both",
        "content_degree_both_roles",
    ]
    labels = ["Triangles", "Neighbor degree", "Both", "Role products*"]
    for ax, name, title in zip(
        axes,
        ["main_eval", "h1101_eval", "h1102_eval", "h1103_eval"],
        ["V_val", "Holdout 1101", "Holdout 1102", "Holdout 1103"],
        strict=True,
    ):
        for i, arm in enumerate(arms):
            c = results[name]["contrasts"][arm]
            lo, hi = c["community_interval"]
            value = c["balanced_bce_gain"]
            ax.plot([lo, hi], [i, i], color="#267a95", linewidth=2)
            ax.plot(value, i, "o", color="#267a95")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(yticks=range(4), yticklabels=labels, title=title, xlabel="Balanced BCE reduction")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.suptitle(
        "Paired gains; 95% community-resampling sensitivity intervals\n"
        "Against content + degree; *role products against content + degree + both",
        fontsize=11,
    )
    fig.savefig(output / "paired_gains.png", dpi=180)
    fig.savefig(output / "paired_gains.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
