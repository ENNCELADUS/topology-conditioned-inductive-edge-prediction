"""V_val: a full-density validation node region carved out of `train_graph.pkl`.

Replaces the old `V_hold` protocol. `V_val` is a hashed-frontier, round-robin
grown union of `n_regions` dispersed BFS regions inside the loopless giant
component of the training truth graph (train⁺ ∪ val⁺), stopped at the node-count
prefix whose induced edge count is nearest `edge_fraction` of the component's
total loopless edges. Unlike `V_hold`, cross-boundary edges are **not**
quarantined: only pairs with *both* endpoints inside `V_val` are pulled out of
training, so nothing is wasted, and `V_val` gets a fully induced (not
uniform-thinned) gold topology.

Every derivation is a pure function of its inputs plus `ValRegionParams`, so
callers (workers, the builder CLI) re-derive it deterministically rather than
reading a cached split; `outputs/val_region/*.json` (this module's `__main__`)
is provenance only and is never read back by a worker.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
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
    """Tunable knobs of the V_val derivation; the seam small toy tests use.

    Attributes:
        edge_fraction: Target share of the giant component's loopless edges
            that `V_val`'s induced subgraph should hold.
        n_regions: Number of dispersed seed regions grown into `V_val`.
        salt: Hash-ordering salt shared by seed selection, region growth, and
            bucket-ball sampling.
        bucket_sizes: Node-count sizes of the BFS-ball evaluation buckets.
        buckets_per_size: Number of balls drawn per size.
        negative_seed: RNG seed for the deterministic validation-negative
            top-up sampler.
    """

    edge_fraction: float = 0.20
    n_regions: int = 5
    salt: str = "val-region-v1|"
    bucket_sizes: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
    buckets_per_size: int = 50
    negative_seed: int = 0


@dataclass(frozen=True)
class ValRegionSplit:
    """The derived V_val region plus the training/validation pair partition.

    Attributes:
        train_nodes: All train-side node ids (unchanged by the split).
        v_val: The derived validation node region, a subset of `train_nodes`.
        region_seeds: The `n_regions` seed nodes, in selection order.
        training_positives: Truth edges (self-pairs included) with at least
            one endpoint outside `v_val`.
        training_negatives: Benchmark negative rows with at least one endpoint
            outside `v_val`, in input order.
        val_positives: Truth edges with both endpoints inside `v_val`
            (self-pairs included), sorted.
        val_negatives: `v_val`-internal benchmark negatives topped up to
            `len(val_positives)` by deterministic sampling.
        buckets: Bucket size -> BFS-ball node sets, sampled inside `v_val`.
        params: The parameters this split was derived under.
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

    def build_training_graph(self) -> nx.Graph:
        """Return the loopless training-side graph over ALL train nodes."""
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


def _select_seeds(
    component: nx.Graph, n_regions: int, order_key: Callable[[str], tuple[bytes, str]]
) -> tuple[str, ...]:
    """Select `n_regions` dispersed seeds: hashed-min, then iterative farthest-point."""
    component_nodes = frozenset(component.nodes())
    seed1 = min(component_nodes, key=order_key)
    seeds = [seed1]
    dist_maps = {seed1: dict(nx.single_source_shortest_path_length(component, seed1))}
    for _ in range(n_regions - 1):
        remaining = component_nodes - set(seeds)
        dist_values = {
            node: min(dist_maps[s].get(node, len(component_nodes)) for s in seeds)
            for node in remaining
        }
        max_dist = max(dist_values.values())
        candidates = [node for node in remaining if dist_values[node] == max_dist]
        best = min(candidates, key=order_key)
        seeds.append(best)
        dist_maps[best] = dict(nx.single_source_shortest_path_length(component, best))
    return tuple(seeds)


def _grow_regions(
    component: nx.Graph,
    seeds: Sequence[str],
    edge_fraction: float,
    order_key: Callable[[str], tuple[bytes, str]],
) -> tuple[frozenset[str], tuple[tuple[str, ...], ...]]:
    """Round-robin grow `seeds` to the prefix nearest `edge_fraction` of the component's edges.

    Each region keeps its own hashed-order frontier (a min-heap over
    `order_key`); one turn adds one unclaimed node to one region, cycling
    through regions. The induced edge count over the union of all claimed
    nodes is tracked incrementally and checked after every addition; growth
    stops at whichever of the first crossing prefix or its immediate
    predecessor is numerically closer to the target (ties keep the crossing
    prefix).

    Returns:
        `(v_val, region_membership)`: the union of claimed nodes, and each
        region's claimed nodes in claim order (seed first).

    Raises:
        ValueError: If every region's frontier is exhausted before the target
            is ever reached.
    """
    k = len(seeds)
    claimed: dict[str, int] = {seed: i for i, seed in enumerate(seeds)}
    region_nodes: list[list[str]] = [[seed] for seed in seeds]
    frontiers: list[list[tuple[bytes, str]]] = []
    for seed in seeds:
        heap = [order_key(n) for n in component.neighbors(seed) if n not in claimed]
        heapq.heapify(heap)
        frontiers.append(heap)

    target = edge_fraction * component.number_of_edges()
    edge_count = 0
    for i, seed in enumerate(seeds):
        prior = set(seeds[:i])
        edge_count += sum(1 for nb in component.neighbors(seed) if nb in prior)

    if edge_count >= target:
        return frozenset(claimed), tuple(tuple(nodes) for nodes in region_nodes)

    previous_edge_count = edge_count
    stalled: set[int] = set()
    turn = 0
    while True:
        i = turn % k
        turn += 1
        if i in stalled:
            if len(stalled) == k:
                raise ValueError(
                    "all region frontiers exhausted before reaching the edge_fraction target"
                )
            continue

        heap = frontiers[i]
        while heap and heap[0][1] in claimed:
            heapq.heappop(heap)
        if not heap:
            stalled.add(i)
            if len(stalled) == k:
                raise ValueError(
                    "all region frontiers exhausted before reaching the edge_fraction target"
                )
            continue

        _, node = heapq.heappop(heap)
        claimed[node] = i
        region_nodes[i].append(node)
        # `node` was just added to `claimed`, but the component is loopless so
        # `node` is never its own neighbor: this only counts edges to others.
        edge_count += sum(1 for nb in component.neighbors(node) if nb in claimed)
        for neighbor in component.neighbors(node):
            if neighbor not in claimed:
                heapq.heappush(heap, order_key(neighbor))

        if edge_count >= target:
            current_diff = abs(edge_count - target)
            previous_diff = abs(previous_edge_count - target)
            if previous_diff < current_diff:
                del claimed[node]
                region_nodes[i].pop()
            return frozenset(claimed), tuple(tuple(nodes) for nodes in region_nodes)
        previous_edge_count = edge_count


def _derive_region_membership(
    component: nx.Graph, params: ValRegionParams
) -> tuple[tuple[str, ...], frozenset[str], tuple[tuple[str, ...], ...]]:
    """Select seeds and grow V_val over one connected `component`.

    Args:
        component: A connected graph (callers pass the truth graph's giant
            component); farthest-point seed selection assumes connectivity.
        params: Region-derivation parameters.

    Returns:
        `(seeds, v_val, region_membership)`.

    Raises:
        ValueError: If `component` is empty, or `params.n_regions` exceeds its
            node count.
    """
    nodes = tuple(component.nodes())
    if not nodes:
        raise ValueError("component has no nodes to derive a V_val region from")
    if params.n_regions > len(nodes):
        raise ValueError(f"n_regions={params.n_regions} exceeds the component's {len(nodes)} nodes")

    def order_key(node: str) -> tuple[bytes, str]:
        return hashlib.sha256(f"{params.salt}{node}".encode()).digest(), node

    seeds = _select_seeds(component, params.n_regions, order_key)
    v_val, region_membership = _grow_regions(component, seeds, params.edge_fraction, order_key)
    return seeds, v_val, region_membership


def _fill_val_negatives(
    *,
    v_val: frozenset[str],
    val_positives: Sequence[Pair],
    existing: Sequence[Pair],
    global_positive_edges: frozenset[Pair],
    seed: int,
) -> tuple[Pair, ...]:
    """Top up `existing` V_val-internal negatives to `len(val_positives)` rows.

    Truncates `existing` (in its given order) if it already meets the target;
    otherwise draws additional pairs with endpoints proportional to loopless
    truth degree within `v_val`, rejecting self-pairs, global positives, and
    duplicates against `existing` plus rows already drawn this call.

    Raises:
        RuntimeError: If the target cannot be reached within a generous
            attempt budget.
    """
    target = len(val_positives)
    if len(existing) >= target:
        return tuple(existing[:target])

    nodes = sorted(v_val)
    n = len(nodes)
    degree: dict[str, int] = dict.fromkeys(nodes, 0)
    for u, v in val_positives:
        if u != v:
            degree[u] += 1
            degree[v] += 1
    weights = np.array([float(degree[node]) for node in nodes], dtype=np.float64)
    total = weights.sum()
    probs = weights / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    rng = np.random.default_rng((seed,))
    chosen: list[Pair] = []
    seen: set[Pair] = set(existing)
    needed = target - len(existing)
    max_attempts = max(needed * 50, 10_000)
    attempts = 0
    while len(chosen) < needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"val-negative top-up could not reach target count {needed} "
                f"after {attempts - 1} attempts"
            )
        i = int(rng.choice(n, p=probs))
        j = int(rng.choice(n, p=probs))
        if i == j:
            continue
        pair = canonical_pair(nodes[i], nodes[j])
        if pair in global_positive_edges or pair in seen:
            continue
        seen.add(pair)
        chosen.append(pair)
    return tuple(existing) + tuple(chosen)


def derive_val_region_split(
    train_nodes: Iterable[str],
    truth_edges: Iterable[Pair],
    benchmark_negatives: Iterable[Pair],
    global_positive_edges: frozenset[Pair],
    *,
    params: ValRegionParams | None = None,
) -> ValRegionSplit:
    """Derive the deterministic V_val region and the training/validation partition.

    Args:
        train_nodes: All train-side node ids.
        truth_edges: `train_graph.pkl`'s edge set (self-pairs included).
        benchmark_negatives: Label-0 rows from `train_edges.txt` + `val_edges.txt`.
        global_positive_edges: The full `positive_edges.txt` set (rejection set
            for sampled validation negatives).
        params: Region-derivation parameters; defaults to `ValRegionParams()`.

    Returns:
        The `ValRegionSplit`.

    Raises:
        ValueError: If `params.edge_fraction` is not in `(0, 1)`,
            `params.n_regions < 1`, `train_nodes` is empty, a truth edge has an
            endpoint outside `train_nodes`, or region growth exhausts every
            frontier before reaching the target.
    """
    params = params if params is not None else ValRegionParams()
    if not 0.0 < params.edge_fraction < 1.0:
        raise ValueError(f"edge_fraction must be in (0, 1), got {params.edge_fraction}")
    if params.n_regions < 1:
        raise ValueError(f"n_regions must be >= 1, got {params.n_regions}")

    node_set = frozenset(train_nodes)
    if not node_set:
        raise ValueError("train_nodes must be non-empty")

    canonical_truth = _canonicalize_truth_edges(truth_edges, node_set)
    loopless_edges = frozenset((u, v) for u, v in canonical_truth if u != v)
    full_graph = nx.Graph()
    full_graph.add_nodes_from(node_set)
    full_graph.add_edges_from(loopless_edges)

    giant_nodes = _giant_component_nodes(full_graph)
    component = full_graph.subgraph(giant_nodes).copy()
    seeds, v_val, _region_membership = _derive_region_membership(component, params)

    training_positives = frozenset(
        (u, v) for u, v in canonical_truth if not (u in v_val and v in v_val)
    )
    val_positives = tuple(sorted((u, v) for u, v in canonical_truth if u in v_val and v in v_val))

    canonical_negatives = [canonical_pair(u, v) for u, v in benchmark_negatives]
    training_negatives = tuple(
        pair for pair in canonical_negatives if not (pair[0] in v_val and pair[1] in v_val)
    )
    val_negatives_from_benchmark = [
        pair for pair in canonical_negatives if pair[0] in v_val and pair[1] in v_val
    ]
    val_negatives = _fill_val_negatives(
        v_val=v_val,
        val_positives=val_positives,
        existing=val_negatives_from_benchmark,
        global_positive_edges=global_positive_edges,
        seed=params.negative_seed,
    )

    g_val_simple = nx.Graph()
    g_val_simple.add_nodes_from(v_val)
    g_val_simple.add_edges_from((u, v) for u, v in val_positives if u != v)
    buckets = sample_bfs_ball_buckets(
        g_val_simple, sizes=params.bucket_sizes, per_size=params.buckets_per_size, salt=params.salt
    )

    return ValRegionSplit(
        train_nodes=node_set,
        v_val=v_val,
        region_seeds=seeds,
        training_positives=training_positives,
        training_negatives=training_negatives,
        val_positives=val_positives,
        val_negatives=val_negatives,
        buckets=buckets,
        params=params,
    )


def val_universe_arrays(v_val: Iterable[str]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Return the complete C(n,2)+n row universe over `sorted(v_val)`, as int32 index arrays.

    Args:
        v_val: The validation node region.

    Returns:
        `(u_idx, v_idx)`: non-self rows in `itertools.combinations` order
        (`np.triu_indices(n, k=1)`), followed by the n self rows `(i, i)`.
    """
    nodes = sorted(v_val)
    n = len(nodes)
    tri_u, tri_v = np.triu_indices(n, k=1)
    diag = np.arange(n)
    u_idx = np.concatenate([tri_u, diag]).astype(np.int32)
    v_idx = np.concatenate([tri_v, diag]).astype(np.int32)
    return u_idx, v_idx


_COMPLEMENT_SAMPLE_SEED: int = 424242
BALL_UNION_COMPLEMENT_SAMPLE_SIZE: int = 200_000


@dataclass(frozen=True)
class ValBallUnionUniverse:
    """The deduplicated within-ball pair union plus a pinned complement sample.

    Attributes:
        u_idx: Row endpoint one of every deduplicated within-ball pair
            (self-pairs included), canonical `u_idx <= v_idx`, sorted
            lexicographically by `(u_idx, v_idx)`.
        v_idx: Row endpoint two, aligned with `u_idx`.
        sample_u_idx: Row endpoint one of the pinned uniform complement
            sample (non-self pairs outside the within-ball union), sorted
            lexicographically.
        sample_v_idx: Row endpoint two, aligned with `sample_u_idx`.
        complement_total: Total count of non-self pairs outside the
            within-ball union (before sampling).
    """

    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]
    sample_u_idx: NDArray[np.int32]
    sample_v_idx: NDArray[np.int32]
    complement_total: int


def val_ball_union_universe(
    split: ValRegionSplit, *, sample_size: int = BALL_UNION_COMPLEMENT_SAMPLE_SIZE
) -> ValBallUnionUniverse:
    """Return the within-ball pair union plus a pinned uniform complement sample.

    The union `U` covers every unordered pair (self-pairs included) whose
    endpoints share at least one common ball of `split.buckets`, deduplicated
    globally across balls. The complement is every non-self pair over
    `sorted(split.v_val)` NOT in `U`; it feeds threshold estimation only, so a
    fixed-size pinned uniform sample (without replacement) stands in for it.

    Indices are int32 positions into `sorted(split.v_val)`, the same index
    space `val_universe_arrays` uses.

    Args:
        split: The derived V_val region split; `split.buckets` must be
            non-empty.
        sample_size: Complement sample size; capped at `complement_total`.

    Returns:
        The `ValBallUnionUniverse`.

    Raises:
        ValueError: If `sample_size < 0` or `split.buckets` is empty.
    """
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}")
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

    nonself_mask = encoded_union % n != encoded_union // n
    union_nonself_encoded = encoded_union[nonself_mask]

    tri_u, tri_v = np.triu_indices(n, k=1)
    full_nonself_encoded = tri_u.astype(np.int64) * n + tri_v.astype(np.int64)
    complement_encoded = np.setdiff1d(
        full_nonself_encoded, union_nonself_encoded, assume_unique=True
    )

    complement_total = math.comb(n, 2) - int(union_nonself_encoded.size)

    sample_count = min(sample_size, complement_total)
    if sample_count > 0:
        rng = np.random.default_rng(_COMPLEMENT_SAMPLE_SEED)
        sampled_encoded = np.sort(rng.choice(complement_encoded, size=sample_count, replace=False))
        sample_u_idx = (sampled_encoded // n).astype(np.int32)
        sample_v_idx = (sampled_encoded % n).astype(np.int32)
    else:
        sample_u_idx = np.empty(0, dtype=np.int32)
        sample_v_idx = np.empty(0, dtype=np.int32)

    return ValBallUnionUniverse(
        u_idx=u_idx,
        v_idx=v_idx,
        sample_u_idx=sample_u_idx,
        sample_v_idx=sample_v_idx,
        complement_total=complement_total,
    )


def _bfs_ball(
    graph: nx.Graph, seed: str, size: int, order_key: Callable[[str], tuple[bytes, str]]
) -> set[str]:
    """Grow a `size`-node hashed-frontier BFS ball from `seed`."""
    visited = {seed}
    ordered = [seed]
    frontier = [seed]
    while frontier and len(ordered) < size:
        next_frontier = {
            neighbor
            for node in frontier
            for neighbor in graph.neighbors(node)
            if neighbor not in visited
        }
        frontier = sorted(next_frontier, key=order_key)
        visited.update(frontier)
        ordered.extend(frontier[: size - len(ordered)])
    if len(ordered) != size:
        raise ValueError(
            f"component containing seed {seed!r} has fewer than {size} reachable nodes"
        )
    return set(ordered)


def sample_bfs_ball_buckets(
    graph: nx.Graph,
    *,
    sizes: Sequence[int],
    per_size: int,
    salt: str,
) -> dict[int, list[set[str]]]:
    """Sample deterministic hashed-frontier BFS-ball buckets, mirroring the test protocol.

    For each size, seeds are drawn in hashed order from nodes whose component
    has at least `size` nodes (a node not yet used as a seed for this size is
    preferred; seeds are reused once every eligible node has served once), and
    each ball grows via the same hashed-frontier idiom as region growth.
    Overlapping balls (within or across draws) are expected.

    Args:
        graph: Graph to sample balls from.
        sizes: Ball sizes to sample.
        per_size: Number of balls to draw per size.
        salt: Hash-ordering salt; combined with size and draw index so
            different draws (and different sizes) get independent orderings.

    Returns:
        Size -> list of `per_size` node sets, each of exactly `size` nodes.

    Raises:
        ValueError: If no component has at least `size` nodes, for some size.
    """
    components = list(nx.connected_components(graph))
    result: dict[int, list[set[str]]] = {}
    for size in sizes:
        eligible_nodes = sorted(
            {node for component in components if len(component) >= size for node in component}
        )
        if not eligible_nodes:
            raise ValueError(f"no component with at least {size} nodes for bucket sampling")

        balls: list[set[str]] = []
        used_seeds: set[str] = set()
        for draw_index in range(per_size):

            def order_key(
                node: str, _draw_index: int = draw_index, _size: int = size
            ) -> tuple[bytes, str]:
                return hashlib.sha256(
                    f"{salt}bucket|{_size}|{_draw_index}|{node}".encode()
                ).digest(), node

            candidates = [n for n in eligible_nodes if n not in used_seeds] or eligible_nodes
            seed = min(candidates, key=order_key)
            used_seeds.add(seed)
            balls.append(_bfs_ball(graph, seed, size, order_key))
        result[size] = balls
    return result


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
        salt=f"{split.params.salt}fidelity|",
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
    parser.add_argument("--edge-fraction", type=float, default=defaults.edge_fraction)
    parser.add_argument("--n-regions", type=int, default=defaults.n_regions)
    parser.add_argument("--salt", type=str, default=defaults.salt)
    parser.add_argument("--bucket-sizes", type=int, nargs="+", default=list(defaults.bucket_sizes))
    parser.add_argument("--buckets-per-size", type=int, default=defaults.buckets_per_size)
    parser.add_argument("--negative-seed", type=int, default=defaults.negative_seed)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Derive V_val for one benchmark package and write a provenance JSON manifest."""
    args = _build_arg_parser().parse_args(argv)
    params = ValRegionParams(
        edge_fraction=args.edge_fraction,
        n_regions=args.n_regions,
        salt=args.salt,
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
        "strategy": args.strategy,
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
    "BALL_UNION_COMPLEMENT_SAMPLE_SIZE",
    "Pair",
    "ValBallUnionUniverse",
    "ValRegionParams",
    "ValRegionSplit",
    "derive_val_region_split",
    "region_fidelity_stats",
    "sample_bfs_ball_buckets",
    "val_ball_union_universe",
    "val_universe_arrays",
]
