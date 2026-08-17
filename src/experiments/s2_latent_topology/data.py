"""S2 region corpus: BFS-ball node regions with frozen features, batched for a GAE.

Draws deterministic, salt-keyed BFS-ball node regions from the (self-loop-stripped)
training graph via `sample_bfs_ball_buckets` — the exact test-bucket construction
rule `src/data/val_region.py` uses for V_val's own evaluation buckets — and pairs
each region with the frozen 1536-d F0 mean-pooled node feature matrix. Regions
touching a known featureless node are dropped rather than fed to the generator with
a zero-feature row standing in for real input. The resulting `RegionCorpus` is
train/val split per size, cached to disk, and batched (`collate_regions`,
`iter_size_batches`) for a graph autoencoder.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch

from src.data.features import FeatureStore, build_f0_matrix
from src.data.val_region import sample_bfs_ball_buckets
from src.eval.graph_metrics import strip_self_loops

S2_SIZES: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
FEATURELESS_NODES: frozenset[str] = frozenset({"node_004764", "node_007050"})


@dataclass(frozen=True)
class RegionCorpus:
    """Sampled BFS-ball node regions plus frozen features, ready for batching.

    Attributes:
        node_ids: Sorted node ids of the source graph; a node's row in
            `features` is its position in this list.
        features: `(N, D)` fp32 F0 mean-pooled feature matrix in `node_ids`
            order; nodes absent from the feature store get an all-zero row.
        regions: Per region, its member nodes as sorted global indices into
            `node_ids`.
        edges: Per region, its induced loopless edges as a `(E_r, 2)` long
            tensor of LOCAL indices into that region's `regions[i]` tuple,
            `i < j`.
        train_idx: Indices into `regions`/`edges` assigned to training.
        val_idx: Indices into `regions`/`edges` held out for validation.
        dropped_featureless_regions: Count of sampled regions discarded for
            containing a node in `FEATURELESS_NODES`.
    """

    node_ids: list[str]
    features: torch.Tensor
    regions: list[tuple[int, ...]]
    edges: list[torch.Tensor]
    train_idx: list[int]
    val_idx: list[int]
    dropped_featureless_regions: int


def build_features(
    store: FeatureStore, node_ids: Sequence[str], *, cache_path: Path
) -> tuple[torch.Tensor, list[str]]:
    """Build a full `(N, D)` fp32 feature matrix in `node_ids` order.

    Ids absent from `store` get an all-zero row. `build_f0_matrix` is called
    only on the ids present in `store.node_ids`, cached at `cache_path` (a
    fresh path — never `allow_cache_subset`), and the result is scattered into
    the full-width matrix.

    Args:
        store: Frozen per-node feature store.
        node_ids: Ordered node ids to build rows for.
        cache_path: Fresh F0 cache path for the present-id subset.

    Returns:
        `(matrix, missing_ids)`: the `(N, D)` matrix and the sorted list of
        `node_ids` absent from `store`.
    """
    present = [node_id for node_id in node_ids if node_id in store.node_ids]
    missing = sorted(node_id for node_id in node_ids if node_id not in store.node_ids)
    present_matrix, present_index = build_f0_matrix(store, present, cache_path=cache_path)

    matrix = torch.zeros(len(node_ids), store.input_dim, dtype=torch.float32)
    for row, node_id in enumerate(node_ids):
        if node_id in present_index:
            matrix[row] = present_matrix[present_index[node_id]]
    return matrix, missing


def build_region_corpus(
    train_graph: nx.Graph,
    store: FeatureStore,
    *,
    sizes: Sequence[int],
    per_size: int,
    salt: str,
    holdout_frac: float,
    cache_dir: Path,
) -> RegionCorpus:
    """Sample BFS-ball regions from `train_graph` and attach frozen features.

    Strips self-loops, draws `sample_bfs_ball_buckets(sizes=sizes,
    per_size=per_size, salt=salt)`, then flattens the result in ascending size
    order, preserving draw order within a size. Regions containing a
    `FEATURELESS_NODES` member are dropped and counted rather than kept. Each
    kept region's induced loopless edges are precomputed as local `(E_r, 2)`
    `i < j` index pairs.

    Split: for each size independently, the last `ceil(holdout_frac * kept)`
    kept regions (by draw order) go to `val_idx`, the rest to `train_idx` — a
    deterministic split with no shuffling.

    Args:
        train_graph: Training truth graph (self-loops allowed; stripped here).
        store: Frozen per-node feature store to mean-pool F0 features from.
        sizes: Region node-count sizes to sample.
        per_size: Number of regions to draw per size.
        salt: Hash-ordering salt passed through to `sample_bfs_ball_buckets`.
        holdout_frac: Per-size fraction of kept regions held out for `val_idx`.
        cache_dir: Directory the F0 feature cache (`f0_train.pt`) is written to.

    Returns:
        The `RegionCorpus`.
    """
    simple_graph = strip_self_loops(train_graph)
    buckets = sample_bfs_ball_buckets(simple_graph, sizes=sizes, per_size=per_size, salt=salt)

    node_ids = sorted(simple_graph.nodes())
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    features, _missing = build_features(store, node_ids, cache_path=cache_dir / "f0_train.pt")

    regions: list[tuple[int, ...]] = []
    edges: list[torch.Tensor] = []
    train_idx: list[int] = []
    val_idx: list[int] = []
    dropped = 0

    for size in sizes:
        kept: list[int] = []
        for node_set in buckets[size]:
            if node_set & FEATURELESS_NODES:
                dropped += 1
                continue
            ordered_node_ids = sorted(node_set, key=node_index.__getitem__)
            region = tuple(node_index[node_id] for node_id in ordered_node_ids)
            local_idx = {node_id: pos for pos, node_id in enumerate(ordered_node_ids)}
            local_edges = sorted(
                (local_idx[u], local_idx[v])
                if local_idx[u] < local_idx[v]
                else (local_idx[v], local_idx[u])
                for u, v in simple_graph.subgraph(node_set).edges()
            )
            kept.append(len(regions))
            regions.append(region)
            edges.append(torch.tensor(local_edges, dtype=torch.long).reshape(-1, 2))

        n_val = math.ceil(holdout_frac * len(kept))
        if n_val > 0:
            val_idx.extend(kept[-n_val:])
            train_idx.extend(kept[:-n_val])
        else:
            train_idx.extend(kept)

    return RegionCorpus(
        node_ids=node_ids,
        features=features,
        regions=regions,
        edges=edges,
        train_idx=train_idx,
        val_idx=val_idx,
        dropped_featureless_regions=dropped,
    )


def save_corpus(corpus: RegionCorpus, path: Path) -> None:
    """Save a `RegionCorpus` as a plain-dict torch checkpoint at `path`."""
    torch.save(
        {
            "node_ids": corpus.node_ids,
            "features": corpus.features,
            "regions": corpus.regions,
            "edges": corpus.edges,
            "train_idx": corpus.train_idx,
            "val_idx": corpus.val_idx,
            "dropped_featureless_regions": corpus.dropped_featureless_regions,
        },
        path,
    )


def load_corpus(path: Path) -> RegionCorpus:
    """Load a `RegionCorpus` previously written by `save_corpus`."""
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    return RegionCorpus(
        node_ids=cast(list[str], payload["node_ids"]),
        features=cast(torch.Tensor, payload["features"]),
        regions=cast(list[tuple[int, ...]], payload["regions"]),
        edges=cast(list[torch.Tensor], payload["edges"]),
        train_idx=cast(list[int], payload["train_idx"]),
        val_idx=cast(list[int], payload["val_idx"]),
        dropped_featureless_regions=cast(int, payload["dropped_featureless_regions"]),
    )


@dataclass(frozen=True)
class RegionBatch:
    """A padded, dense batch of regions ready for a graph autoencoder.

    Attributes:
        x: `(B, N, D)` fp32 node features, zero-padded past each region's size.
        adj: `(B, N, N)` fp32 symmetric 0/1 adjacency, zero diagonal and
            zero-padded past each region's size.
        mask: `(B, N)` fp32 0/1 node validity mask.
        n: `(B,)` long true region sizes.
    """

    x: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor
    n: torch.Tensor


def collate_regions(
    corpus: RegionCorpus, indices: Sequence[int], device: torch.device
) -> RegionBatch:
    """Pad `indices`' regions to the batch's max size and build dense adjacency.

    Args:
        corpus: Region corpus to draw regions/edges/features from.
        indices: Region indices (into `corpus.regions`/`corpus.edges`) to batch.
        device: Device the returned tensors are placed on.

    Returns:
        A `RegionBatch` with `x`/`adj`/`mask` zero-padded to the batch's max
        region size.

    Raises:
        ValueError: If `indices` is empty.
    """
    if not indices:
        raise ValueError("collate_regions requires at least one region index")
    region_sizes = [len(corpus.regions[i]) for i in indices]
    max_n = max(region_sizes)
    dim = corpus.features.size(1)

    x = torch.zeros(len(indices), max_n, dim, dtype=torch.float32, device=device)
    adj = torch.zeros(len(indices), max_n, max_n, dtype=torch.float32, device=device)
    mask = torch.zeros(len(indices), max_n, dtype=torch.float32, device=device)
    n = torch.zeros(len(indices), dtype=torch.long, device=device)

    for row, (idx, size) in enumerate(zip(indices, region_sizes, strict=True)):
        global_ids = torch.tensor(corpus.regions[idx], dtype=torch.long)
        x[row, :size] = corpus.features[global_ids].to(device)
        mask[row, :size] = 1.0
        n[row] = size
        region_edges = corpus.edges[idx]
        if region_edges.numel() > 0:
            src, dst = region_edges[:, 0].to(device), region_edges[:, 1].to(device)
            adj[row, src, dst] = 1.0
            adj[row, dst, src] = 1.0

    return RegionBatch(x=x, adj=adj, mask=mask, n=n)


def iter_size_batches(
    corpus: RegionCorpus,
    indices: Sequence[int],
    *,
    batch_sets: int,
    rng: np.random.Generator,
) -> Iterator[list[int]]:
    """Yield shuffled, size-pure batches of region indices.

    Shuffles `indices` with `rng`, groups the shuffled order by region size,
    then chunks each size's group (visited in ascending size order) into
    pieces of at most `batch_sets` region indices.

    Args:
        corpus: Region corpus `indices` refer into.
        indices: Region indices to batch.
        batch_sets: Maximum region count per yielded batch.
        rng: Seeded generator controlling the shuffle; a fixed seed makes the
            yielded sequence deterministic.

    Yields:
        Lists of region indices, all sharing one region size, of length at
        most `batch_sets`.
    """
    shuffled = list(indices)
    rng.shuffle(shuffled)
    by_size: dict[int, list[int]] = {}
    for idx in shuffled:
        by_size.setdefault(len(corpus.regions[idx]), []).append(idx)
    for size in sorted(by_size):
        group = by_size[size]
        for start in range(0, len(group), batch_sets):
            yield group[start : start + batch_sets]


__all__ = [
    "FEATURELESS_NODES",
    "S2_SIZES",
    "RegionBatch",
    "RegionCorpus",
    "build_features",
    "build_region_corpus",
    "collate_regions",
    "iter_size_batches",
    "load_corpus",
    "save_corpus",
]
