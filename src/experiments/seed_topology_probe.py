"""Probe whether teacher and generated latents linearly encode local topology.

Generated validation latents are produced operator-side. If a future
``--dump-latents PATH`` surface is needed, it should reuse ``score_universe``'s
``--topo-gen-control`` scoring machinery; this post-hoc probe only consumes the
resulting NPZ artifact.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.artifacts import load_benchmark
from src.data.val_region import derive_val_region_split
from src.distill.artifacts import load_kd_targets
from src.distill.teacher_targets import truth_graph_for_kd

_RIDGE = 1e-3
_FOLDS = 5
_STAT_NAMES = (
    "deg_u",
    "deg_v",
    "common_neighbors",
    "clustering_u",
    "clustering_v",
)


def _as_finite_matrix(name: str, values: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def ridge_r2(latents: NDArray[np.floating], target: NDArray[np.floating]) -> float:
    """Return deterministic five-fold out-of-fold ridge R2."""
    features = _as_finite_matrix("latents", latents)
    response = np.asarray(target, dtype=np.float64)
    if response.ndim != 1:
        raise ValueError("target must be a rank-1 array")
    if len(features) != len(response):
        raise ValueError("latents and target must share the same row count")
    if len(features) < _FOLDS:
        raise ValueError(f"ridge probe requires at least {_FOLDS} rows")
    if not np.isfinite(response).all():
        raise ValueError("target must contain only finite values")

    predictions = np.empty_like(response)
    shuffled_rows = np.random.default_rng(0).permutation(len(features))
    folds = np.array_split(shuffled_rows, _FOLDS)
    for test_rows in folds:
        train_rows = np.setdiff1d(shuffled_rows, test_rows, assume_unique=True)
        train_x = features[train_rows]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale == 0.0] = 1.0
        train_design = np.column_stack(((train_x - mean) / scale, np.ones(len(train_rows))))
        test_design = np.column_stack(
            ((features[test_rows] - mean) / scale, np.ones(len(test_rows)))
        )
        penalty = np.eye(train_design.shape[1], dtype=np.float64) * _RIDGE
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            train_design.T @ train_design + penalty,
            train_design.T @ response[train_rows],
        )
        predictions[test_rows] = test_design @ coefficients

    residual = float(np.square(response - predictions).sum())
    total = float(np.square(response - response.mean()).sum())
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


def _topology_targets(
    graph: nx.Graph, pairs: Sequence[tuple[str, str]]
) -> dict[str, NDArray[np.float64]]:
    values = {name: np.empty(len(pairs), dtype=np.float64) for name in _STAT_NAMES}
    for row, (u, v) in enumerate(pairs):
        has_query_edge = graph.has_edge(u, v)
        degree_decrement = 2 if u == v else 1
        if not has_query_edge:
            degree_decrement = 0
        values["deg_u"][row] = float(graph.degree[u] - degree_decrement)
        values["deg_v"][row] = float(graph.degree[v] - degree_decrement)

        neighbors_u = set(graph[u]) - {u}
        neighbors_v = set(graph[v]) - {v}
        if has_query_edge and u != v:
            neighbors_u.remove(v)
            neighbors_v.remove(u)
        values["common_neighbors"][row] = float(len(neighbors_u & neighbors_v))
        for name, neighbors in (
            ("clustering_u", neighbors_u),
            ("clustering_v", neighbors_v),
        ):
            degree = len(neighbors)
            links = sum(graph.has_edge(a, b) for a, b in combinations(neighbors, 2))
            values[name][row] = 0.0 if degree < 2 else 2.0 * links / (degree * (degree - 1))
    return values


def probe_latents(
    graph: nx.Graph,
    pairs: Sequence[tuple[str, str]],
    *,
    teacher: NDArray[np.floating],
    generated: NDArray[np.floating],
) -> dict[str, dict[str, float]]:
    """Probe five query-edge-masked topology statistics from two latent arrays."""
    teacher_matrix = _as_finite_matrix("teacher", teacher)
    generated_matrix = _as_finite_matrix("generated", generated)
    if len(teacher_matrix) != len(pairs):
        raise ValueError("teacher row count must match pairs")
    if len(generated_matrix) != len(pairs):
        raise ValueError("generated row count must match pairs")
    if generated_matrix.shape[1] != teacher_matrix.shape[1]:
        raise ValueError("teacher and generated latent dimensions must match")

    targets = _topology_targets(graph, pairs)
    shuffled = generated_matrix[np.random.default_rng(0).permutation(len(generated_matrix))]
    return {
        name: {
            "teacher_r2": ridge_r2(teacher_matrix, target),
            "generated_r2": ridge_r2(generated_matrix, target),
            "shuffled_r2": ridge_r2(shuffled, target),
        }
        for name, target in targets.items()
    }


def _load_training_graph(data_root: Path, strategy: str) -> nx.Graph:
    benchmark_root = data_root / "benchmark_2025_neurips"
    benchmark = load_benchmark(benchmark_root, strategy, verify=True)
    raw_benchmark = load_benchmark(benchmark_root, strategy, verify=False)
    raw_negatives = [
        pair
        for pair, label in zip(
            raw_benchmark.split.train_pairs.pairs,
            raw_benchmark.split.train_pairs.labels,
            strict=True,
        )
        if label == 0
    ] + [
        pair
        for pair, label in zip(
            raw_benchmark.split.val_pairs.pairs,
            raw_benchmark.split.val_pairs.labels,
            strict=True,
        )
        if label == 0
    ]
    split = derive_val_region_split(
        benchmark.split.train_nodes,
        benchmark.split.train_graph.edges(),
        raw_negatives,
        benchmark.positive_edges,
    )
    return truth_graph_for_kd(split)


def build_parser() -> argparse.ArgumentParser:
    """Build the topology-probe CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the post-hoc topology probe and write its JSON report."""
    args = build_parser().parse_args(argv)
    with np.load(args.generated, allow_pickle=False) as generated_archive:
        if "latents" not in generated_archive:
            raise ValueError("generated NPZ must contain a 'latents' array")
        generated = np.asarray(generated_archive["latents"], dtype=np.float64)

    targets = load_kd_targets(args.targets)
    pairs = [
        (targets.node_ids[int(a)], targets.node_ids[int(b)])
        for a, b in zip(targets.val_pair_a_idx, targets.val_pair_b_idx, strict=True)
    ]
    teacher = targets.val_teacher_rep.astype(np.float64)
    graph = _load_training_graph(args.graph, args.strategy)
    report = probe_latents(graph, pairs, teacher=teacher, generated=generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
