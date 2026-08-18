"""S3 data layer: region corpus with V_val masking, V_val buckets, fp32 base-logit cache.

Spec: `docs/tmp/s3_set_residual_plan.md`. Reuses S2's BFS-ball region sampler
(`src.experiments.s2_latent_topology.data.build_region_corpus`) for the raw
training regions and `src.data.val_region`'s canonical `V_val` derivation for
the node set that must never appear on both sides of a training pair. Three
pieces are built here:

- `S3Corpus`: per region, induced-positive plus sampled-negative `i < j` local
  pairs (V_val-internal pairs dropped), their labels, and the frozen
  checkpoint's base logit for each -- everything `train.py`'s loss needs.
- `VvalEval`: V_val's own BFS-ball selection buckets, each with a dense
  fp32 base-logit matrix, the true loopless adjacency, and the precomputed
  GS/RD/MMD reference state (self-loops kept, mirroring the official
  evaluator) -- everything `train.py`'s per-epoch selection metric needs.
- A fp32 base-logit cache (`BaseLogitCache`): one scoring pass per corpus
  (training pairs) or per V_val-eval build (bucket pairs) with the frozen
  `v3_1` checkpoint via `src.score_universe._load_checkpoint`/`_score_v3_1`.
  Training pairs and V_val bucket pairs are disjoint by construction (a
  training pair always has at least one endpoint outside V_val; a bucket pair
  always has both endpoints inside it), so no pair is ever scored by both
  passes. A cache file is content-addressed: `_ensure_base_logit_cache` reuses
  an on-disk cache only when its `checkpoint_id` matches and it already covers
  every needed pair, so a stale or differently-scoped file is never silently
  trusted -- it is just rescored and overwritten.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair
from src.data.features import FeatureStore
from src.data.val_region import ValRegionParams, derive_val_region_split, sample_bfs_ball_buckets
from src.eval.graph_metrics import (
    BucketReference,
    MMDConfig,
    precompute_bucket_reference,
    strip_self_loops,
)
from src.experiments.s2_latent_topology.data import (
    S2_SIZES,
    RegionBatch,
    RegionCorpus,
    build_region_corpus,
    collate_regions,
)
from src.score_universe import _load_checkpoint, _score_v3_1

logger = logging.getLogger(__name__)

#: Approximate per-batch token budget for `_score_v3_1`'s bucketed sampler.
DEFAULT_TOKEN_BUDGET = 16_384

__all__ = [
    "DEFAULT_TOKEN_BUDGET",
    "BaseLogitCache",
    "S3Corpus",
    "VvalEval",
    "build_base_logit_cache",
    "build_s3_corpus",
    "build_vval_eval",
    "collate_pair_batch",
    "derive_vval_nodes",
    "load_base_logit_cache",
    "load_s3_corpus",
    "load_vval_eval",
    "save_base_logit_cache",
    "save_s3_corpus",
    "save_vval_eval",
]


# --------------------------------------------------------------------------- V_val node set


def derive_vval_nodes(
    train_graph: nx.Graph, *, params: ValRegionParams | None = None
) -> frozenset[str]:
    """Derive V_val's node set via the repo's canonical `derive_val_region_split` construction.

    Mirrors `score_universe._load_val_region_split`'s call (`ValRegionParams()`
    defaults): same seed selection and hashed-frontier region growth over
    `train_graph`'s truth edges. Benchmark negatives and the global positive
    set only shape `ValRegionSplit`'s pair partitions, never `v_val` itself, so
    they are passed empty here -- this function's only output is the node set.

    Args:
        train_graph: Training truth graph (self-pairs allowed; the split
            derivation strips them internally before growing the region).
        params: Region-derivation parameters; defaults to `ValRegionParams()`,
            the repo's canonical default.

    Returns:
        The `V_val` node set.
    """
    truth_edges = list(train_graph.edges())
    split = derive_val_region_split(
        train_graph.nodes(), truth_edges, [], frozenset(), params=params or ValRegionParams()
    )
    return split.v_val


# --------------------------------------------------------------------------- base-logit cache


@dataclass(frozen=True)
class BaseLogitCache:
    """A deduplicated global node-id-pair -> fp32 base-logit table.

    Attributes:
        node_ids: Unique node ids referenced by `u_idx`/`v_idx`, sorted ascending.
        u_idx: `(P,)` int32 indices into `node_ids`; each row is a canonical
            `(u_idx, v_idx)` pair (`node_ids[u_idx] <= node_ids[v_idx]`),
            sorted lexicographically by node id.
        v_idx: `(P,)` int32 indices into `node_ids`, aligned with `u_idx`.
        logit: `(P,)` fp32 raw frozen-checkpoint logits, aligned with `u_idx`/`v_idx`.
        checkpoint_id: The frozen scorer's state-dict digest
            (`score_universe._checkpoint_id`), for content-addressed reuse.
    """

    node_ids: list[str]
    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]
    logit: NDArray[np.float32]
    checkpoint_id: str


def build_base_logit_cache(
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    checkpoint: Path,
    device: torch.device,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    chunk_size: int = 200_000,
) -> BaseLogitCache:
    """Score the deduplicated, canonicalized pair union once with the frozen checkpoint.

    Args:
        pairs: Node-id pairs to score; duplicates and input order are
            irrelevant (deduplicated and canonicalized here).
        store: Feature store providing per-node token sequences.
        checkpoint: Frozen `v3_1`-family scorer checkpoint (`best.pt`/`last.pt`),
            loaded via `score_universe._load_checkpoint`.
        device: Compute device.
        token_budget: Per-batch token budget passed to `_score_v3_1`.
        chunk_size: Outer pair-count chunk size; scoring and progress logging
            proceed chunk by chunk so a multi-million-pair union never builds
            one oversized dataset pass at once.

    Returns:
        The `BaseLogitCache`, fp32, rows sorted lexicographically by
        `(u_idx, v_idx)` node id order.

    Raises:
        ValueError: If `pairs` is empty, the checkpoint is not `v3_1`-family,
            or any produced logit is non-finite.
    """
    if not pairs:
        raise ValueError("build_base_logit_cache requires at least one pair")
    deduped = sorted({canonical_pair(u, v) for u, v in pairs})
    node_ids = sorted({node for pair in deduped for node in pair})
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    model, model_family, checkpoint_id = _load_checkpoint(checkpoint)
    if model_family != "v3_1":
        raise ValueError(
            f"S3 base-logit scoring requires a 'v3_1' checkpoint, got {model_family!r}"
        )
    model = model.to(device)
    model.eval()

    total = len(deduped)
    logit = np.empty(total, dtype=np.float32)
    for start in range(0, total, chunk_size):
        chunk = deduped[start : start + chunk_size]
        logit[start : start + len(chunk)] = _score_v3_1(
            model, chunk, store, device=device, amp="off", token_budget=token_budget
        )
        logger.info(
            "S3 base-logit cache: scored %d/%d unique pairs",
            min(start + chunk_size, total),
            total,
        )
    if not np.isfinite(logit.astype(np.float64)).all():
        raise ValueError("S3 base-logit cache: scoring produced a non-finite logit")

    u_idx = np.array([node_index[u] for u, _ in deduped], dtype=np.int32)
    v_idx = np.array([node_index[v] for _, v in deduped], dtype=np.int32)
    return BaseLogitCache(
        node_ids=node_ids, u_idx=u_idx, v_idx=v_idx, logit=logit, checkpoint_id=checkpoint_id
    )


def save_base_logit_cache(cache: BaseLogitCache, path: Path) -> None:
    """Persist a `BaseLogitCache` as a self-contained `.npz` (fp32 `logit`, no pickling)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        node_ids=np.array(cache.node_ids),
        u_idx=cache.u_idx,
        v_idx=cache.v_idx,
        logit=cache.logit,
        checkpoint_id=np.array(cache.checkpoint_id),
    )


def load_base_logit_cache(path: Path) -> BaseLogitCache:
    """Load a `BaseLogitCache` previously written by `save_base_logit_cache`.

    Raises:
        ValueError: If the stored `logit` array is not fp32.
    """
    with np.load(path) as data:
        logit = data["logit"]
        if logit.dtype != np.float32:
            raise ValueError(f"{path}: base-logit cache must be fp32, got {logit.dtype}")
        return BaseLogitCache(
            node_ids=data["node_ids"].tolist(),
            u_idx=data["u_idx"].astype(np.int32),
            v_idx=data["v_idx"].astype(np.int32),
            logit=logit,
            checkpoint_id=str(data["checkpoint_id"]),
        )


def _base_logit_lookup(cache: BaseLogitCache) -> dict[tuple[str, str], float]:
    """Return a `(u, v) -> logit` dict from a `BaseLogitCache` (`u <= v` keys)."""
    return {
        (cache.node_ids[u], cache.node_ids[v]): float(value)
        for u, v, value in zip(
            cache.u_idx.tolist(), cache.v_idx.tolist(), cache.logit.tolist(), strict=True
        )
    }


def _ensure_base_logit_cache(
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    checkpoint: Path,
    device: torch.device,
    token_budget: int,
    cache_path: Path,
) -> BaseLogitCache:
    """Load a matching on-disk base-logit cache, or build and persist a fresh one.

    A cached artifact is reused only when its `checkpoint_id` matches the
    checkpoint's own state-dict digest AND it covers every pair in `pairs`;
    otherwise scoring reruns over `pairs` and overwrites `cache_path`. Reuse
    is content-addressed (by pair coverage and checkpoint identity), never by
    trusting the path alone.
    """
    deduped = sorted({canonical_pair(u, v) for u, v in pairs})
    if cache_path.exists():
        cached = load_base_logit_cache(cache_path)
        _, _, checkpoint_id = _load_checkpoint(checkpoint)
        if cached.checkpoint_id == checkpoint_id:
            cached_lookup = _base_logit_lookup(cached)
            if set(deduped) <= cached_lookup.keys():
                return cached
    fresh = build_base_logit_cache(
        deduped, store, checkpoint=checkpoint, device=device, token_budget=token_budget
    )
    save_base_logit_cache(fresh, cache_path)
    return fresh


# --------------------------------------------------------------------------- S3Corpus


@dataclass(frozen=True)
class S3Corpus:
    """S3's training corpus: S2 region corpus plus V_val-masked pair/label/base-logit tensors.

    Attributes:
        base: The underlying `RegionCorpus` (S2 type): node ids, F0 features,
            sampled regions, and their induced edges. `base.train_idx` covers
            every region (S3 selects on `VvalEval`, not an in-graph holdout).
        vval_nodes: The `V_val` node set; a pair with both endpoints here is
            excluded from every region's training pairs.
        train_pairs: Per region (aligned index-for-index with `base.regions`),
            `(P_r, 2)` long LOCAL `i < j` indices: induced positives plus
            sampled in-region negatives, V_val-internal pairs already dropped.
        train_labels: Per region, `(P_r,)` fp32 labels aligned with `train_pairs`.
        train_base_logits: Per region, `(P_r,)` fp32 frozen-checkpoint base
            logits aligned with `train_pairs`.
    """

    base: RegionCorpus
    vval_nodes: frozenset[str]
    train_pairs: list[torch.Tensor]
    train_labels: list[torch.Tensor]
    train_base_logits: list[torch.Tensor]


def _stable_seed(*parts: str) -> int:
    """Derive a deterministic 64-bit seed from `parts` via SHA-256 (never Python's `hash`)."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_negative_pairs(
    n: int, positives: set[tuple[int, int]], count: int, seed: int
) -> list[tuple[int, int]]:
    """Deterministically sample up to `count` in-region non-edge `i < j` local pairs.

    Enumerates every `i < j` pair not in `positives`; if `count` meets or
    exceeds that pool, the whole pool is returned. Otherwise `count` pairs are
    drawn without replacement via a `seed`-keyed `np.random.Generator`.

    Returns:
        The sampled pairs in ascending `(i, j)` order.
    """
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in positives]
    if count >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[i] for i in sorted(chosen.tolist())]


def build_s3_corpus(
    train_graph: nx.Graph,
    store: FeatureStore,
    *,
    checkpoint: Path,
    sizes: Sequence[int] = S2_SIZES,
    per_size: int = 150,
    neg_ratio: int = 5,
    salt: str = "s3_train_v1",
    cache_dir: Path,
    device: torch.device,
    vval_nodes: frozenset[str] | None = None,
    val_region_params: ValRegionParams | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> S3Corpus:
    """Sample S3's training corpus: BFS-ball regions, V_val-masked pairs, frozen base logits.

    Reuses `build_region_corpus` (`holdout_frac=0.0` -- S3 selects on
    `VvalEval`, not an in-graph holdout). Per region, positives are every
    induced loopless local pair; negatives are `neg_ratio * len(positives)`
    in-region non-edge `i < j` pairs sampled without replacement,
    deterministically seeded per region from `salt`. Any pair (positive or
    negative) with both endpoints in V_val is dropped. The frozen checkpoint
    then scores the deduplicated union of every surviving pair exactly once
    (`_ensure_base_logit_cache`, persisted under `cache_dir`).

    Args:
        train_graph: Training truth graph (self-pairs allowed; `build_region_corpus`
            strips them before sampling regions).
        store: Frozen per-node feature store.
        checkpoint: Frozen `v3_1`-family scorer checkpoint.
        sizes: Region node-count sizes to sample.
        per_size: Number of regions to draw per size.
        neg_ratio: Negative-to-positive sampling ratio per region.
        salt: Hash-ordering salt for both region sampling and negative sampling.
        cache_dir: Directory for the F0 feature cache and the base-logit cache.
        device: Compute device for scoring.
        vval_nodes: Explicit V_val node set (for tests on graphs too small for
            the canonical derivation); defaults to `derive_vval_nodes`.
        val_region_params: Parameters for the canonical V_val derivation when
            `vval_nodes` is not given; defaults to `ValRegionParams()`.
        token_budget: Approximate per-batch token budget for scoring.

    Returns:
        The `S3Corpus`.
    """
    resolved_vval = (
        vval_nodes
        if vval_nodes is not None
        else derive_vval_nodes(train_graph, params=val_region_params)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = build_region_corpus(
        train_graph,
        store,
        sizes=sizes,
        per_size=per_size,
        salt=salt,
        holdout_frac=0.0,
        cache_dir=cache_dir,
    )

    positives_by_region: list[list[tuple[int, int]]] = []
    negatives_by_region: list[list[tuple[int, int]]] = []
    for r, region in enumerate(base.regions):
        region_node_ids = [base.node_ids[i] for i in region]
        n = len(region)
        positive_set = {(int(i), int(j)) for i, j in base.edges[r].tolist()}
        seed = _stable_seed(salt, "neg", str(r))
        negatives = _sample_negative_pairs(n, positive_set, neg_ratio * len(positive_set), seed)

        def _kept(pair: tuple[int, int], _ids: list[str] = region_node_ids) -> bool:
            u, v = _ids[pair[0]], _ids[pair[1]]
            return not (u in resolved_vval and v in resolved_vval)

        positives_by_region.append([p for p in sorted(positive_set) if _kept(p)])
        negatives_by_region.append([p for p in negatives if _kept(p)])

    needed_pairs: set[tuple[str, str]] = set()
    for r, region in enumerate(base.regions):
        region_node_ids = [base.node_ids[i] for i in region]
        for i, j in positives_by_region[r] + negatives_by_region[r]:
            needed_pairs.add(canonical_pair(region_node_ids[i], region_node_ids[j]))

    if needed_pairs:
        cache = _ensure_base_logit_cache(
            sorted(needed_pairs),
            store,
            checkpoint=checkpoint,
            device=device,
            token_budget=token_budget,
            cache_path=cache_dir / "base_logits_train.npz",
        )
        lookup = _base_logit_lookup(cache)
    else:
        lookup = {}

    train_pairs: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    train_base_logits: list[torch.Tensor] = []
    for r, region in enumerate(base.regions):
        region_node_ids = [base.node_ids[i] for i in region]
        pairs = positives_by_region[r] + negatives_by_region[r]
        labels = [1.0] * len(positives_by_region[r]) + [0.0] * len(negatives_by_region[r])
        logits = [lookup[canonical_pair(region_node_ids[i], region_node_ids[j])] for i, j in pairs]
        train_pairs.append(torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))
        train_labels.append(torch.tensor(labels, dtype=torch.float32))
        train_base_logits.append(torch.tensor(logits, dtype=torch.float32))

    return S3Corpus(
        base=base,
        vval_nodes=resolved_vval,
        train_pairs=train_pairs,
        train_labels=train_labels,
        train_base_logits=train_base_logits,
    )


def save_s3_corpus(corpus: S3Corpus, path: Path) -> None:
    """Save an `S3Corpus` as a plain-dict torch checkpoint at `path`."""
    torch.save(
        {
            "node_ids": corpus.base.node_ids,
            "features": corpus.base.features,
            "regions": corpus.base.regions,
            "edges": corpus.base.edges,
            "train_idx": corpus.base.train_idx,
            "val_idx": corpus.base.val_idx,
            "dropped_featureless_regions": corpus.base.dropped_featureless_regions,
            "vval_nodes": sorted(corpus.vval_nodes),
            "train_pairs": corpus.train_pairs,
            "train_labels": corpus.train_labels,
            "train_base_logits": corpus.train_base_logits,
        },
        path,
    )


def load_s3_corpus(path: Path) -> S3Corpus:
    """Load an `S3Corpus` previously written by `save_s3_corpus`."""
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    base = RegionCorpus(
        node_ids=cast(list[str], payload["node_ids"]),
        features=cast(torch.Tensor, payload["features"]),
        regions=cast(list[tuple[int, ...]], payload["regions"]),
        edges=cast(list[torch.Tensor], payload["edges"]),
        train_idx=cast(list[int], payload["train_idx"]),
        val_idx=cast(list[int], payload["val_idx"]),
        dropped_featureless_regions=cast(int, payload["dropped_featureless_regions"]),
    )
    return S3Corpus(
        base=base,
        vval_nodes=frozenset(cast(list[str], payload["vval_nodes"])),
        train_pairs=cast(list[torch.Tensor], payload["train_pairs"]),
        train_labels=cast(list[torch.Tensor], payload["train_labels"]),
        train_base_logits=cast(list[torch.Tensor], payload["train_base_logits"]),
    )


def collate_pair_batch(
    corpus: S3Corpus, region_indices: Sequence[int], device: torch.device
) -> tuple[RegionBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch a set of regions' features plus their pooled training pairs/labels/base-logits.

    Args:
        corpus: The `S3Corpus` to batch from.
        region_indices: Region indices (into `corpus.base.regions`, and
            equally into `corpus.train_pairs`/`train_labels`/`train_base_logits`)
            to include, in output-row order.
        device: Device the returned tensors are placed on.

    Returns:
        `(region_batch, pairs, labels, base_logits)`: `region_batch` is
        `collate_regions`'s padded `(x, adj, mask, n)` bundle (`adj` unused by
        S3 training); `pairs` is `(P, 3)` long `[set_row, i, j]` -- `set_row`
        indexes `region_indices`' output order, `i`/`j` are that region's
        local indices; `labels`/`base_logits` are `(P,)` fp32, row-aligned
        with `pairs`.
    """
    region_batch = collate_regions(corpus.base, region_indices, device)

    pairs_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    logits_rows: list[torch.Tensor] = []
    for row, region_idx in enumerate(region_indices):
        region_pairs = corpus.train_pairs[region_idx]
        if region_pairs.numel() == 0:
            continue
        set_row_col = torch.full((region_pairs.size(0), 1), row, dtype=torch.long)
        pairs_rows.append(torch.cat([set_row_col, region_pairs], dim=1))
        labels_rows.append(corpus.train_labels[region_idx])
        logits_rows.append(corpus.train_base_logits[region_idx])

    if pairs_rows:
        pairs = torch.cat(pairs_rows, dim=0).to(device)
        labels = torch.cat(labels_rows, dim=0).to(device)
        base_logits = torch.cat(logits_rows, dim=0).to(device)
    else:
        pairs = torch.zeros((0, 3), dtype=torch.long, device=device)
        labels = torch.zeros((0,), dtype=torch.float32, device=device)
        base_logits = torch.zeros((0,), dtype=torch.float32, device=device)
    return region_batch, pairs, labels, base_logits


# --------------------------------------------------------------------------- VvalEval


@dataclass(frozen=True)
class VvalEval:
    """V_val selection buckets: dense fp32 base-logit + true-adjacency state per BFS-ball set.

    Attributes:
        sizes: Ascending bucket sizes sampled (`bucket_ref.node_sets` keys).
        node_ids: size -> per-draw sorted node-id lists; a set's `k`-th row in
            `base_logits[size][k]`/`true_adj[size][k]` follows this list's
            local index order.
        base_logits: size -> per-draw `(n, n)` fp32 base-logit matrices from
            the frozen checkpoint, symmetric, zero diagonal (`i < j` entries
            filled on both sides).
        true_adj: size -> per-draw `(n, n)` fp32 loopless 0/1 true-adjacency
            matrices (`train_graph`'s self-loop-stripped edges), symmetric,
            zero diagonal. V_val-internal edges are legal here -- this is
            evaluation, not training.
        bucket_ref: Precomputed reference-side state (self-loops kept) for
            GS/RD/MMD, built by `precompute_bucket_reference` over these same
            node sets from the raw V_val-induced training subgraph.
        checkpoint_id: The frozen scorer's checkpoint id the base logits came from.
    """

    sizes: tuple[int, ...]
    node_ids: dict[int, list[list[str]]]
    base_logits: dict[int, list[torch.Tensor]]
    true_adj: dict[int, list[torch.Tensor]]
    bucket_ref: BucketReference
    checkpoint_id: str


def _giant_component(g: nx.Graph) -> nx.Graph:
    """Return `g`'s largest connected component, tie-broken by sorted node tuple.

    Mirrors `src.data.val_region`'s private giant-component tie-break so
    V_val's own bucket-sampling substrate matches the repo's convention.

    Raises:
        ValueError: If `g` has no nodes.
    """
    components = [tuple(sorted(c)) for c in nx.connected_components(g)]
    if not components:
        raise ValueError("graph has no nodes to derive a giant component from")
    largest = min(components, key=lambda nodes: (-len(nodes), nodes))
    return cast(nx.Graph, g.subgraph(largest).copy())


def _dense_symmetric_matrix(
    node_ids: Sequence[str], get_value: Callable[[str, str], float]
) -> torch.Tensor:
    """Build an `(n, n)` fp32 symmetric zero-diagonal matrix from a pairwise `i < j` value fn."""
    n = len(node_ids)
    matrix = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = get_value(node_ids[i], node_ids[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def build_vval_eval(
    train_graph: nx.Graph,
    store: FeatureStore,
    *,
    checkpoint: Path,
    sizes: Sequence[int] = S2_SIZES,
    per_size: int = 20,
    salt: str = "s3_vval_v1",
    cache_dir: Path,
    device: torch.device,
    vval_nodes: frozenset[str] | None = None,
    val_region_params: ValRegionParams | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> VvalEval:
    """Sample V_val's BFS-ball selection buckets and score them with the frozen checkpoint.

    Buckets are drawn via `sample_bfs_ball_buckets` on the V_val-induced
    graph's loopless giant component. Every bucket's full `i < j` pair list is
    scored once (`_ensure_base_logit_cache`, persisted under `cache_dir`); the
    reference-side GS/RD/MMD state is precomputed from the raw (self-loops
    kept) V_val-induced training subgraph over the same node sets.

    Args:
        train_graph: Training truth graph (self-pairs allowed).
        store: Frozen per-node feature store.
        checkpoint: Frozen `v3_1`-family scorer checkpoint.
        sizes: Bucket node-count sizes to sample.
        per_size: Number of buckets to draw per size.
        salt: Hash-ordering salt for bucket sampling.
        cache_dir: Directory for the base-logit cache.
        device: Compute device for scoring.
        vval_nodes: Explicit V_val node set (for tests on graphs too small for
            the canonical derivation); defaults to `derive_vval_nodes`.
        val_region_params: Parameters for the canonical V_val derivation when
            `vval_nodes` is not given; defaults to `ValRegionParams()`.
        token_budget: Approximate per-batch token budget for scoring.

    Returns:
        The `VvalEval`.
    """
    resolved_vval = (
        vval_nodes
        if vval_nodes is not None
        else derive_vval_nodes(train_graph, params=val_region_params)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    g_val_raw = cast(nx.Graph, train_graph.subgraph(resolved_vval).copy())
    giant = _giant_component(strip_self_loops(g_val_raw))
    buckets = sample_bfs_ball_buckets(giant, sizes=sizes, per_size=per_size, salt=salt)

    needed_pairs = {
        canonical_pair(u, v)
        for node_sets in buckets.values()
        for node_set in node_sets
        for u, v in itertools.combinations(sorted(node_set), 2)
    }
    cache = _ensure_base_logit_cache(
        sorted(needed_pairs),
        store,
        checkpoint=checkpoint,
        device=device,
        token_budget=token_budget,
        cache_path=cache_dir / "base_logits_vval.npz",
    )
    lookup = _base_logit_lookup(cache)
    edge_set = {canonical_pair(u, v) for u, v in train_graph.edges() if u != v}

    node_ids: dict[int, list[list[str]]] = {}
    base_logits: dict[int, list[torch.Tensor]] = {}
    true_adj: dict[int, list[torch.Tensor]] = {}
    for size in sorted(buckets):
        ordered_sets = [sorted(node_set) for node_set in buckets[size]]
        node_ids[size] = ordered_sets
        base_logits[size] = [
            _dense_symmetric_matrix(ordered, lambda u, v: lookup[canonical_pair(u, v)])
            for ordered in ordered_sets
        ]
        true_adj[size] = [
            _dense_symmetric_matrix(
                ordered, lambda u, v: 1.0 if canonical_pair(u, v) in edge_set else 0.0
            )
            for ordered in ordered_sets
        ]

    bucket_ref = precompute_bucket_reference(g_val_raw, buckets, MMDConfig())

    return VvalEval(
        sizes=tuple(sorted(buckets)),
        node_ids=node_ids,
        base_logits=base_logits,
        true_adj=true_adj,
        bucket_ref=bucket_ref,
        checkpoint_id=cache.checkpoint_id,
    )


def save_vval_eval(vval: VvalEval, path: Path) -> None:
    """Pickle a `VvalEval` at `path` (holds `networkx.Graph`s, so not a torch checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(vval, handle)


def load_vval_eval(path: Path) -> VvalEval:
    """Load a `VvalEval` previously written by `save_vval_eval`."""
    with path.open("rb") as handle:
        return cast(VvalEval, pickle.load(handle))  # noqa: S301 - internally generated S3 cache
