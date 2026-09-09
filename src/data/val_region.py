"""PRING-style node-held-out V_val inside the original train-side substrate.

A single seeded FIFO BFS admits at most 10% of substrate positive pairs, including
self-loops. V_val nodes are removed from the training universe: every pair touching
V_val (internal and cross-boundary, positive and negative) is held out, mirroring
the official train/test boundary. Fixed BFS buckets and endpoint-frequency
classification negatives come from the validation induced graph.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair, load_benchmark
from src.eval.graph_metrics import (
    MMDConfig,
    clustering_histogram,
    degree_histogram,
    laplacian_spectrum_histogram,
    mmd_squared,
)

Pair = tuple[str, str]


@dataclass(frozen=True)
class ValRegionParams:
    """Fixed production defaults; explicit overrides support small test graphs."""

    positive_edge_fraction: float = 0.10
    root_neighbors: int = 5
    # Pre-registered random root draw (the project's original default). The
    # test-informed seed 273 (root node_002696) is retired to a labeled
    # secondary result; see docs/results/validation_density_selection/README.md.
    split_seed: int = 42
    bucket_seed: int = 43
    bucket_sizes: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
    buckets_per_size: int = 50
    negative_seed: int = 0


@dataclass(frozen=True)
class ValRegionSplit:
    """Validation membership and disjoint training/validation pair partitions.

    ``train_nodes`` is the substrate minus V_val. Every pair touching V_val is
    held out from training, including cross-boundary positives and negatives;
    ``substrate_nodes`` restores the complete train-side node set.
    """

    train_nodes: frozenset[str]
    v_val: frozenset[str]
    region_seeds: tuple[str, ...]
    training_positives: frozenset[Pair]
    training_negatives: tuple[Pair, ...]
    val_positives: tuple[Pair, ...]
    val_negatives: tuple[Pair, ...]
    buckets: dict[int, list[set[str]]]
    params: ValRegionParams

    @property
    def substrate_nodes(self) -> frozenset[str]:
        """Original train-side membership before the node holdout."""
        return self.train_nodes | self.v_val

    def build_training_graph(self) -> nx.Graph:
        """Return the loopless training-side graph over all substrate nodes."""
        graph = nx.Graph()
        graph.add_nodes_from(self.train_nodes)
        graph.add_edges_from((u, v) for u, v in self.training_positives if u != v)
        return graph

    def build_g_val(self) -> nx.Graph:
        """Return the induced V_val gold graph, self-loops kept."""
        graph = nx.Graph()
        graph.add_nodes_from(self.v_val)
        graph.add_edges_from(self.val_positives)
        return graph

    def build_g_val_simple(self) -> nx.Graph:
        """Return the induced V_val gold graph, self-loops stripped."""
        graph = nx.Graph()
        graph.add_nodes_from(self.v_val)
        graph.add_edges_from((u, v) for u, v in self.val_positives if u != v)
        return graph

    @property
    def val_cls_pairs(self) -> list[Pair]:
        """Return the classification-validation pairs: positives then negatives."""
        return [*self.val_positives, *self.val_negatives]

    @property
    def val_cls_labels(self) -> list[int]:
        """Return labels aligned with `val_cls_pairs`."""
        return [1] * len(self.val_positives) + [0] * len(self.val_negatives)


def _canonicalize_truth_edges(
    truth_edges: Iterable[Pair], node_set: frozenset[str]
) -> frozenset[Pair]:
    """Canonicalize truth edges, rejecting endpoints outside `node_set`."""
    result: set[Pair] = set()
    for u, v in truth_edges:
        if u not in node_set or v not in node_set:
            raise ValueError(f"truth edge endpoint outside train nodes: {(u, v)!r}")
        result.add(canonical_pair(u, v))
    return frozenset(result)


def _giant_component_nodes(graph: nx.Graph) -> frozenset[str]:
    """Return the largest connected component, tie-broken by sorted node tuple."""
    components: list[tuple[str, ...]] = [tuple(sorted(c)) for c in nx.connected_components(graph)]
    if not components:
        raise ValueError("graph has no nodes to derive a giant component from")
    largest = min(components, key=lambda nodes: (-len(nodes), nodes))
    return frozenset(largest)


def _bfs_ball(graph: nx.Graph, seed: str, size: int) -> set[str]:
    """Take exactly ``size`` nodes using PRING's sorted-neighbor FIFO BFS."""
    if size < 1 or seed not in graph:
        raise ValueError("BFS requires a positive size and a root in the graph")
    visited: set[str] = set()
    queue = deque([seed])
    while queue and len(visited) < size:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(sorted(set(graph.neighbors(node)) - visited))
    if len(visited) != size:
        raise ValueError(f"component containing seed {seed!r} has fewer than {size} nodes")
    return visited


def _bfs_positive_budget(graph: nx.Graph, root: str, budget: int) -> frozenset[str]:
    """Longest FIFO prefix within the induced positive-pair cap, loops counted once.

    Stop before the first node that would exceed the cap; never skip a frontier
    node to search for a cheaper or better-matching region.
    """
    visited: set[str] = set()
    queue = deque([root])
    edges = 0
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        added = sum(neighbor in visited or neighbor == node for neighbor in graph[node])
        if edges + added > budget:
            break
        edges += added
        visited.add(node)
        queue.extend(sorted(set(graph.neighbors(node)) - visited))
    return frozenset(visited)


def _sample_val_negatives(val_positives: Sequence[Pair], *, seed: int) -> tuple[Pair, ...]:
    """Fresh PRING endpoint-frequency negatives, canonical and without replacement."""
    positives = frozenset(val_positives)
    endpoints = [node for pair in sorted(positives) for node in pair]
    nodes = set(endpoints)
    target = len(positives)
    capacity = len(nodes) * (len(nodes) - 1) // 2 - sum(u != v for u, v in positives)
    if capacity < target:
        raise ValueError(
            f"validation has only {capacity} distinct negatives for {target} positives"
        )
    rng = random.Random(seed)
    chosen: list[Pair] = []
    seen: set[Pair] = set()
    for _ in range(max(target * 100, 10_000)):
        if len(chosen) == target:
            return tuple(chosen)
        u, v = rng.choice(endpoints), rng.choice(endpoints)
        pair = canonical_pair(u, v)
        if u == v or pair in positives or pair in seen:
            continue
        seen.add(pair)
        chosen.append(pair)
    raise RuntimeError(f"validation negative sampling could not reach {target} rows")


def derive_val_region_split(
    train_nodes: Iterable[str],
    truth_edges: Iterable[Pair],
    benchmark_negatives: Iterable[Pair],
    global_positive_edges: frozenset[Pair],
    *,
    params: ValRegionParams | None = None,
) -> ValRegionSplit:
    """Derive a node-held-out split solely from the complete train-side substrate.

    ``global_positive_edges`` is accepted by shared callers, but validation
    negative rejection needs only induced validation truth, never test labels.
    """
    params = params if params is not None else ValRegionParams()
    if not 0.0 < params.positive_edge_fraction < 1.0:
        raise ValueError("positive_edge_fraction must be in (0, 1)")
    if params.root_neighbors < 0:
        raise ValueError("root_neighbors must be non-negative")
    node_set = frozenset(train_nodes)
    truth = _canonicalize_truth_edges(truth_edges, node_set)
    budget = int(len(truth) * params.positive_edge_fraction)
    if budget < 1:
        raise ValueError("positive edge budget must be non-empty")
    if not params.bucket_sizes or min(params.bucket_sizes) < 1:
        raise ValueError("bucket sizes must be positive")
    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_set))
    graph.add_edges_from(sorted(truth))
    eligible = {
        node
        for component in nx.connected_components(graph)
        if len(component) >= max(params.bucket_sizes)
        for node in component
    }
    roots = sorted(node for node in eligible if len(graph[node]) == params.root_neighbors)
    if not roots:
        raise ValueError("no eligible root with the required neighbor count and component size")
    root = random.Random(params.split_seed).choice(roots)
    v_val = _bfs_positive_budget(graph, root, budget)
    if len(v_val) < max(params.bucket_sizes):
        raise ValueError("positive edge budget cannot support the requested bucket sizes")
    val_positives = tuple(sorted((u, v) for u, v in truth if u in v_val and v in v_val))
    negatives = [canonical_pair(u, v) for u, v in benchmark_negatives]
    if any(u not in node_set or v not in node_set for u, v in negatives):
        raise ValueError("benchmark negative endpoint outside train-side substrate")
    return ValRegionSplit(
        train_nodes=node_set - v_val,
        v_val=v_val,
        region_seeds=(root,),
        training_positives=frozenset((u, v) for u, v in truth if u not in v_val and v not in v_val),
        training_negatives=tuple((u, v) for u, v in negatives if u not in v_val and v not in v_val),
        val_positives=val_positives,
        val_negatives=_sample_val_negatives(val_positives, seed=params.negative_seed),
        buckets=sample_bfs_ball_buckets(
            graph.subgraph(v_val),
            sizes=params.bucket_sizes,
            per_size=params.buckets_per_size,
            seed=params.bucket_seed,
        ),
        params=params,
    )


@dataclass(frozen=True)
class ValBallUnionUniverse:
    """The deduplicated within-ball pair union used by topology validation.

    Attributes:
        u_idx: Row endpoint one of every deduplicated within-ball pair
            (self-pairs included), canonical `u_idx <= v_idx`, sorted
            lexicographically by `(u_idx, v_idx)`.
        v_idx: Row endpoint two, aligned with `u_idx`.
    """

    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]


def val_ball_union_universe(split: ValRegionSplit) -> ValBallUnionUniverse:
    """Return the exact within-ball pair union.

    The union `U` covers every unordered pair (self-pairs included) whose
    endpoints share at least one common ball of `split.buckets`, deduplicated
    globally across balls. Out-of-ball pairs are intentionally absent: the
    deployable threshold is selected against the sampled-subgraph evaluation
    distribution rather than a full-region graph-reconstruction objective.

    Indices are int32 positions into `sorted(split.v_val)`.

    Args:
        split: The derived V_val region split; `split.buckets` must be
            non-empty.

    Returns:
        The `ValBallUnionUniverse`.

    Raises:
        ValueError: If `split.buckets` is empty.
    """
    if not split.buckets:
        raise ValueError("split.buckets must be non-empty")

    nodes = sorted(split.v_val)
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    encoded_chunks: list[NDArray[np.int64]] = []
    for balls in split.buckets.values():
        for ball in balls:
            ball_idx = np.array(sorted(index[node] for node in ball), dtype=np.int64)
            if ball_idx.size == 0:
                continue
            tri_i, tri_j = np.triu_indices(ball_idx.size, k=0)
            encoded_chunks.append(ball_idx[tri_i] * n + ball_idx[tri_j])

    encoded_union = (
        np.unique(np.concatenate(encoded_chunks)) if encoded_chunks else np.empty(0, dtype=np.int64)
    )
    u_idx = (encoded_union // n).astype(np.int32)
    v_idx = (encoded_union % n).astype(np.int32)

    return ValBallUnionUniverse(u_idx=u_idx, v_idx=v_idx)


def sample_bfs_ball_buckets(
    graph: nx.Graph,
    *,
    sizes: Sequence[int],
    per_size: int,
    seed: int | str,
) -> dict[int, list[set[str]]]:
    """Uniform roots with replacement and FIFO BFS, retaining sample overlaps.

    Require all nodes to reach each size; never silently condition the root
    distribution on connected-component size.
    """
    if not sizes or per_size < 1 or any(size < 1 for size in sizes):
        raise ValueError("bucket sizes and per_size must be positive")
    if not graph or min(map(len, nx.connected_components(graph))) < max(sizes):
        raise ValueError("every root must be able to reach every requested bucket size")
    nodes = sorted(graph.nodes())
    rng = random.Random(seed)
    return {
        size: [_bfs_ball(graph, rng.choice(nodes), size) for _ in range(per_size)]
        for size in sorted(sizes)
    }


def region_fidelity_stats(
    split: ValRegionSplit,
    full_loopless_truth_graph: nx.Graph,
    *,
    config: MMDConfig | None = None,
) -> dict[str, float | int]:
    """Report V_val's sample fidelity against the full training giant component.

    Builder-only, non-gating provenance: samples a same-sampler bank of BFS
    balls on the full loopless truth giant component (the same
    `sample_bfs_ball_buckets` primitive `split.buckets` was drawn with) and
    compares it, per descriptor, against `split.buckets`. Never imported by a
    worker code path.

    Args:
        split: The derived V_val region split.
        full_loopless_truth_graph: The loopless truth graph over ALL train
            nodes (self-loops already stripped by the caller).
        config: MMD kernel configuration; defaults to `MMDConfig()`.

    Returns:
        A flat dict of floats/ints: V_val and full-giant density/clustering/
        size stats, plus per-descriptor `mmd2_vs_full` (V_val bank vs full-giant
        bank) and `full_bank_floor` (the full-giant bank's own odd/even floor).
    """
    config = config if config is not None else MMDConfig()
    giant_nodes = _giant_component_nodes(full_loopless_truth_graph)
    giant = full_loopless_truth_graph.subgraph(giant_nodes).copy()
    g_val_simple = split.build_g_val_simple()

    full_bank = sample_bfs_ball_buckets(
        giant,
        sizes=split.params.bucket_sizes,
        per_size=split.params.buckets_per_size,
        seed=split.params.bucket_seed + 1,
    )

    stats: dict[str, float | int] = {
        "val_region_nodes": g_val_simple.number_of_nodes(),
        "val_region_edges": g_val_simple.number_of_edges(),
        "val_region_density": float(nx.density(g_val_simple)),
        "val_region_avg_clustering": float(nx.average_clustering(g_val_simple)),
        "val_region_components": nx.number_connected_components(g_val_simple),
        "full_giant_nodes": giant.number_of_nodes(),
        "full_giant_edges": giant.number_of_edges(),
        "full_giant_density": float(nx.density(giant)),
        "full_giant_avg_clustering": float(nx.average_clustering(giant)),
    }

    descriptors: dict[str, Callable[[nx.Graph], NDArray[np.float64]]] = {
        "degree": degree_histogram,
        "clustering": clustering_histogram,
        "spectral": laplacian_spectrum_histogram,
    }
    for stat_name, descriptor_fn in descriptors.items():
        per_size_mmd2: list[float] = []
        per_size_floor: list[float] = []
        for size in split.params.bucket_sizes:
            val_descs = [
                descriptor_fn(g_val_simple.subgraph(nodes)) for nodes in split.buckets[size]
            ]
            full_descs = [descriptor_fn(giant.subgraph(nodes)) for nodes in full_bank[size]]
            per_size_mmd2.append(mmd_squared(val_descs, full_descs, config))
            per_size_floor.append(mmd_squared(full_descs[::2], full_descs[1::2], config))
        stats[f"{stat_name}_mmd2_vs_full"] = float(np.mean(per_size_mmd2))
        stats[f"{stat_name}_full_bank_floor"] = float(np.mean(per_size_floor))

    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = ValRegionParams()
    parser = argparse.ArgumentParser(
        description="Derive and write the V_val region-split manifest."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positive-edge-fraction", type=float, default=defaults.positive_edge_fraction
    )
    parser.add_argument("--root-neighbors", type=int, default=defaults.root_neighbors)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--bucket-seed", type=int, default=defaults.bucket_seed)
    parser.add_argument("--bucket-sizes", type=int, nargs="+", default=list(defaults.bucket_sizes))
    parser.add_argument("--buckets-per-size", type=int, default=defaults.buckets_per_size)
    parser.add_argument("--negative-seed", type=int, default=defaults.negative_seed)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Derive V_val for one benchmark package and write a provenance JSON manifest."""
    args = _build_arg_parser().parse_args(argv)
    params = ValRegionParams(
        positive_edge_fraction=args.positive_edge_fraction,
        root_neighbors=args.root_neighbors,
        split_seed=args.split_seed,
        bucket_seed=args.bucket_seed,
        bucket_sizes=tuple(args.bucket_sizes),
        buckets_per_size=args.buckets_per_size,
        negative_seed=args.negative_seed,
    )

    benchmark = load_benchmark(args.data_root, args.strategy)
    train_nodes = benchmark.split.train_nodes
    truth_edges = list(benchmark.split.train_graph.edges())
    benchmark_negatives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs, benchmark.split.train_pairs.labels, strict=True
        )
        if label == 0
    ] + [
        pair
        for pair, label in zip(
            benchmark.split.val_pairs.pairs, benchmark.split.val_pairs.labels, strict=True
        )
        if label == 0
    ]

    split = derive_val_region_split(
        train_nodes, truth_edges, benchmark_negatives, benchmark.positive_edges, params=params
    )

    full_loopless = nx.Graph()
    full_loopless.add_nodes_from(train_nodes)
    full_loopless.add_edges_from((u, v) for u, v in truth_edges if u != v)
    fidelity = region_fidelity_stats(split, full_loopless)

    manifest = {
        "schema_version": "pring_bfs_positive_budget_val_v1",
        "strategy": args.strategy,
        "substrate_nodes": sorted(split.substrate_nodes),
        "train_nodes": sorted(split.train_nodes),
        "params": asdict(params),
        "region_seeds": list(split.region_seeds),
        "v_val": sorted(split.v_val),
        "val_positives": [list(p) for p in split.val_positives],
        "val_negatives": [list(p) for p in split.val_negatives],
        "training_positives_count": len(split.training_positives),
        "training_negatives_count": len(split.training_negatives),
        "buckets": {
            str(size): [sorted(nodes) for nodes in balls] for size, balls in split.buckets.items()
        },
        "fidelity": fidelity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "Pair",
    "ValBallUnionUniverse",
    "ValRegionParams",
    "ValRegionSplit",
    "derive_val_region_split",
    "region_fidelity_stats",
    "sample_bfs_ball_buckets",
    "val_ball_union_universe",
]
