"""STPD pre-gate probe: degree-matched partner discrimination on corrupted training regions.

evidence_class=diagnostic, formal=false; fp32, single GPU, no DDP. (Later tasks extend it; keep
the docstring short.)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn.utils import clip_grad_norm_

from src.data.features import FeatureStore
from src.experiments.s2_latent_topology.data import RegionCorpus, build_region_corpus


def _stable_seed(*parts: str) -> int:
    """Derive a deterministic 64-bit seed from `parts` via SHA-256 (never Python's `hash`)."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SwapProvenance:
    """Provenance for one (region, severity) degree-preserving-swap corruption cell.

    All arrays hold `numpy` `int64` LOCAL region node indices. `deleted`, `inserted`,
    and `kept` are each normalized `i < j`; `quads` instead preserves the randomized
    orientation actually swapped, so `deleted` and `inserted` can be reconstructed
    from it exactly by sorting each pair.

    Attributes:
        quads: `(S, 4)` -- one row `(i, j, k, l)` per completed swap: deleted true
            edges `(i, j)` and `(k, l)`, inserted false edges `(i, l)` and `(k, j)`.
        deleted: `(2S, 2)` -- row `2s` is quad `s`'s `(i, j)`, row `2s+1` is its
            `(k, l)`, each normalized `i < j`.
        inserted: `(2S, 2)` -- row `2s` is quad `s`'s `sorted(i, l)`, row `2s+1` is
            its `sorted(k, j)`.
        kept: `(E-2S, 2)` -- original edges never deleted, normalized, original order.
    """

    quads: NDArray[np.int64]
    deleted: NDArray[np.int64]
    inserted: NDArray[np.int64]
    kept: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate array dtypes, shapes, and the `i < j` normalization invariant.

        Raises:
            ValueError: If any array has the wrong dtype/rank/column count, if
                `deleted`/`inserted` row counts disagree with `2 * len(quads)`, or if
                a normalized-pair array (`deleted`, `inserted`, `kept`) violates `i < j`.
        """
        for name, arr, ncols in (
            ("quads", self.quads, 4),
            ("deleted", self.deleted, 2),
            ("inserted", self.inserted, 2),
            ("kept", self.kept, 2),
        ):
            if arr.dtype != np.int64:
                raise ValueError(f"{name} must be int64, got {arr.dtype}")
            if arr.ndim != 2 or arr.shape[1] != ncols:
                raise ValueError(f"{name} must be shaped (*, {ncols}), got {arr.shape}")

        n_swaps = self.quads.shape[0]
        if self.deleted.shape[0] != 2 * n_swaps:
            raise ValueError(
                f"deleted must have {2 * n_swaps} rows (2 * quads), got {self.deleted.shape[0]}"
            )
        if self.inserted.shape[0] != 2 * n_swaps:
            raise ValueError(
                f"inserted must have {2 * n_swaps} rows (2 * quads), got {self.inserted.shape[0]}"
            )

        normalized = (("deleted", self.deleted), ("inserted", self.inserted), ("kept", self.kept))
        for name, arr in normalized:
            if arr.shape[0] and bool(np.any(arr[:, 0] >= arr[:, 1])):
                raise ValueError(f"{name} rows must satisfy i < j")


def provenance_swaps(
    edges: list[tuple[int, int]],
    fraction: float,
    rng: np.random.Generator,
) -> SwapProvenance | None:
    """Provenance-recording degree-preserving double-edge swaps on a region's edges.

    Repeatedly draws two distinct never-yet-deleted edges, randomizes each edge's
    endpoint orientation, and proposes the cross-swap insertion; accepts only when
    the four endpoints are distinct and neither proposed edge is already live or was
    previously deleted (so `kept`, `deleted`, and `inserted` stay pairwise disjoint).
    Complete-or-drop: either exactly `round(fraction * len(edges))` swaps are found
    within the try budget, or the whole cell is dropped.

    Args:
        edges: The region's true induced edges, local `i < j`, caller-sorted.
        fraction: Target swap fraction of `len(edges)`.
        rng: Source of randomness; the same `rng` state always yields the same result.

    Returns:
        A `SwapProvenance` recording exactly `n_target = round(fraction * len(edges))`
        completed swaps, or `None` if `n_target == 0` or the try budget
        (`50 * n_target` proposal attempts) is exhausted before that many swaps complete.
    """
    n_target = round(fraction * len(edges))
    if n_target == 0:
        return None

    untouched = list(edges)
    current: set[tuple[int, int]] = set(edges)
    deleted_set: set[tuple[int, int]] = set()

    quad_rows: list[tuple[int, int, int, int]] = []
    deleted_rows: list[tuple[int, int]] = []
    inserted_rows: list[tuple[int, int]] = []

    max_tries = 50 * n_target
    tries = 0
    while len(quad_rows) < n_target and tries < max_tries and len(untouched) >= 2:
        tries += 1
        idx = rng.choice(len(untouched), size=2, replace=False)
        e1 = untouched[int(idx[0])]
        e2 = untouched[int(idx[1])]

        a, b = e1
        if rng.random() < 0.5:
            a, b = b, a
        c, d = e2
        if rng.random() < 0.5:
            c, d = d, c

        if len({a, b, c, d}) != 4:
            continue

        p1 = (a, d) if a < d else (d, a)
        p2 = (b, c) if b < c else (c, b)
        if p1 in current or p2 in current:
            continue
        if p1 in deleted_set or p2 in deleted_set:
            continue

        untouched.remove(e1)
        untouched.remove(e2)
        current.discard(e1)
        current.discard(e2)
        deleted_set.add(e1)
        deleted_set.add(e2)
        current.add(p1)
        current.add(p2)

        quad_rows.append((a, b, c, d))
        deleted_rows.append(e1)
        deleted_rows.append(e2)
        inserted_rows.append(p1)
        inserted_rows.append(p2)

    if len(quad_rows) < n_target:
        return None

    kept_rows = [e for e in edges if e not in deleted_set]

    return SwapProvenance(
        quads=np.array(quad_rows, dtype=np.int64).reshape(-1, 4),
        deleted=np.array(deleted_rows, dtype=np.int64).reshape(-1, 2),
        inserted=np.array(inserted_rows, dtype=np.int64).reshape(-1, 2),
        kept=np.array(kept_rows, dtype=np.int64).reshape(-1, 2),
    )


@dataclass(frozen=True)
class PregateConfig:
    """Experiment constants for the STPD pre-gate probe (paths/device stay CLI-level).

    Attributes:
        sizes: Region node-count sizes to sample.
        per_size: Regions drawn per size.
        holdout_frac: Per-size fraction of kept regions held out for validation.
        severities: `(name, swap_fraction)` corruption severities, in build order.
        seeds: Training seeds a later task iterates over.
        salt: Hash-ordering salt for both region sampling and swap-seed derivation.
        lr: Optimizer learning rate (consumed by a later task).
        weight_decay: Optimizer weight decay (consumed by a later task).
        grad_clip: Gradient-norm clip (consumed by a later task).
        epochs: Training epoch budget (consumed by a later task).
        regions_per_step: Regions per training step (consumed by a later task).
    """

    sizes: tuple[int, ...] = (64, 96, 128, 160, 200)
    per_size: int = 120
    holdout_frac: float = 0.2
    severities: tuple[tuple[str, float], ...] = (
        ("light", 0.05),
        ("moderate", 0.15),
        ("heavy", 0.30),
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    salt: str = "stpd_pregate_v1"
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 30
    regions_per_step: int = 16


@dataclass(frozen=True)
class PregateCorpus:
    """The S2 region corpus plus per-(region, severity) swap-corruption provenance.

    Attributes:
        regions: The S2 `RegionCorpus` (unmodified region sampling and features).
        provenance: `(region index into regions.regions, severity name)` -> the
            completed `SwapProvenance` for that cell. Only non-dropped cells present.
        dropped_cells: `(region index, severity name)` cells where `provenance_swaps`
            returned `None`, in deterministic build order.
    """

    regions: RegionCorpus
    provenance: dict[tuple[int, str], SwapProvenance]
    dropped_cells: tuple[tuple[int, str], ...]


def build_pregate_corpus(
    train_graph: nx.Graph,
    store: FeatureStore,
    cfg: PregateConfig,
    *,
    cache_dir: Path,
) -> PregateCorpus:
    """Sample the S2 region corpus and apply degree-preserving swaps per (region, severity).

    For every region (train and holdout alike) and every configured severity, derives a
    salted seed from `(cfg.salt, "swap", region index, severity name)` and calls
    `provenance_swaps` on that region's induced edges (converted to the caller-sorted
    `list[tuple[int, int]]` form, preserving the S2 corpus's stored local order).

    Args:
        train_graph: Training truth graph, passed through to `build_region_corpus`.
        store: Frozen per-node feature store, passed through to `build_region_corpus`.
        cfg: Experiment constants; only the region-sampling and salt fields are read.
        cache_dir: F0 feature cache directory, passed through to `build_region_corpus`.

    Returns:
        The `PregateCorpus`.
    """
    regions = build_region_corpus(
        train_graph,
        store,
        sizes=cfg.sizes,
        per_size=cfg.per_size,
        salt=cfg.salt,
        holdout_frac=cfg.holdout_frac,
        cache_dir=cache_dir,
    )

    provenance: dict[tuple[int, str], SwapProvenance] = {}
    dropped_cells: list[tuple[int, str]] = []
    for region_idx, edges_tensor in enumerate(regions.edges):
        region_edges = [(int(i), int(j)) for i, j in edges_tensor.tolist()]
        for name, fraction in cfg.severities:
            rng = np.random.default_rng(_stable_seed(cfg.salt, "swap", str(region_idx), name))
            prov = provenance_swaps(region_edges, fraction, rng)
            if prov is None:
                dropped_cells.append((region_idx, name))
            else:
                provenance[(region_idx, name)] = prov

    return PregateCorpus(regions=regions, provenance=provenance, dropped_cells=tuple(dropped_cells))


def corrupted_edges_of(prov: SwapProvenance) -> NDArray[np.int64]:
    """Rebuild a cell's corrupted edge set: `kept` union `inserted`, local `i < j`."""
    return cast(NDArray[np.int64], np.concatenate([prov.kept, prov.inserted]))


@dataclass(frozen=True)
class CellContext:
    """Precomputed unmasked dense structural context for one corrupted (region, severity) cell.

    All tensors are LOCAL-index, torch fp32, dense; regions have `n <= 200` so this is
    cheap. Removing the queried pair `(u, v)` never changes which nodes are common
    neighbors of `u` and `v`, nor any node's degree other than `u`'s and `v`'s -- so
    pair-level `cn`/`ra` read directly off these unmasked matrices already equal their
    masked (queried-edge-removed) values; only the endpoint degrees and endpoint
    neighbor-feature aggregates need the masking correction done in
    `pair_context_features`.

    Attributes:
        features: `(n, d)` fp32 region node features, held for masked neighbor-mean
            queries.
        adj: `(n, n)` symmetric 0/1 fp32 adjacency, zero diagonal.
        deg: `(n,)` unmasked degree, `adj.sum(1)`.
        nbr_sum: `(n, d)` unmasked neighbor feature sum, `adj @ features`.
        cn: `(n, n)` unmasked common-neighbor count, `adj @ adj`.
        ra: `(n, n)` unmasked resource-allocation sum, `adj @ diag(r) @ adj` with
            `r[w] = 1 / deg[w]` when `deg[w] > 1`, else `0` (`_ra_term`'s guard).
    """

    features: torch.Tensor
    adj: torch.Tensor
    deg: torch.Tensor
    nbr_sum: torch.Tensor
    cn: torch.Tensor
    ra: torch.Tensor


def build_cell_context(
    n: int,
    corrupted_edges: NDArray[np.int64],
    features: torch.Tensor,
) -> CellContext:
    """Precompute one cell's dense unmasked structural context.

    Args:
        n: Region node count.
        corrupted_edges: `(E, 2)` local `i < j` corrupted edges (kept union inserted
            from a `SwapProvenance`, see `corrupted_edges_of`).
        features: `(n, d)` fp32 region node features.

    Returns:
        The `CellContext`.
    """
    adj = torch.zeros(n, n, dtype=torch.float32)
    if corrupted_edges.shape[0] > 0:
        src = torch.from_numpy(corrupted_edges[:, 0]).long()
        dst = torch.from_numpy(corrupted_edges[:, 1]).long()
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0

    deg = adj.sum(1)
    nbr_sum = adj @ features
    cn = adj @ adj
    r = torch.where(deg > 1, 1.0 / deg, torch.zeros_like(deg))
    ra = adj @ torch.diag(r) @ adj

    return CellContext(features=features, adj=adj, deg=deg, nbr_sum=nbr_sum, cn=cn, ra=ra)


def pair_context_features(ctx: CellContext, pairs: torch.Tensor) -> torch.Tensor:
    """Per-pair masked structural context features, the queried edge removed first.

    Args:
        ctx: A cell's precomputed `CellContext`.
        pairs: `(P, 2)` int64 local `(u, v)` indices.

    Returns:
        `(P, 4 + 2d)` fp32: `log1p(masked_deg_u)`, `log1p(masked_deg_v)`,
        `log1p(cn[u,v])`, `ra[u,v]`, masked neighbor-feature mean of `u`, then of `v`.
    """
    u, v = pairs[:, 0], pairs[:, 1]
    edge_uv = ctx.adj[u, v]

    masked_deg_u = ctx.deg[u] - edge_uv
    masked_deg_v = ctx.deg[v] - edge_uv
    denom_u = masked_deg_u.clamp(min=1.0).unsqueeze(1)
    denom_v = masked_deg_v.clamp(min=1.0).unsqueeze(1)
    mean_u = (ctx.nbr_sum[u] - edge_uv.unsqueeze(1) * ctx.features[v]) / denom_u
    mean_v = (ctx.nbr_sum[v] - edge_uv.unsqueeze(1) * ctx.features[u]) / denom_v

    return torch.cat(
        [
            torch.log1p(masked_deg_u).unsqueeze(1),
            torch.log1p(masked_deg_v).unsqueeze(1),
            torch.log1p(ctx.cn[u, v]).unsqueeze(1),
            ctx.ra[u, v].unsqueeze(1),
            mean_u,
            mean_v,
        ],
        dim=1,
    )


def structure_scores(ctx: CellContext, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Structure-only control arm scores: `(ra[u,v], cn[u,v])` per pair, unmasked-equal.

    Args:
        ctx: A cell's precomputed `CellContext`.
        pairs: `(P, 2)` int64 local `(u, v)` indices.

    Returns:
        `(ra_scores, cn_scores)`, each `(P,)` fp32.
    """
    u, v = pairs[:, 0], pairs[:, 1]
    return ctx.ra[u, v], ctx.cn[u, v]


def save_pregate_corpus(corpus: PregateCorpus, path: Path) -> None:
    """Save a `PregateCorpus` as a plain-dict `torch` checkpoint at `path`."""
    torch.save(
        {
            "regions": {
                "node_ids": corpus.regions.node_ids,
                "features": corpus.regions.features,
                "regions": corpus.regions.regions,
                "edges": corpus.regions.edges,
                "train_idx": corpus.regions.train_idx,
                "val_idx": corpus.regions.val_idx,
                "dropped_featureless_regions": corpus.regions.dropped_featureless_regions,
            },
            "provenance": corpus.provenance,
            "dropped_cells": corpus.dropped_cells,
        },
        path,
    )


def load_pregate_corpus(path: Path) -> PregateCorpus:
    """Load a `PregateCorpus` previously written by `save_pregate_corpus`."""
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=False))
    regions_payload = cast(dict[str, object], payload["regions"])
    regions = RegionCorpus(
        node_ids=cast(list[str], regions_payload["node_ids"]),
        features=cast(torch.Tensor, regions_payload["features"]),
        regions=cast(list[tuple[int, ...]], regions_payload["regions"]),
        edges=cast(list[torch.Tensor], regions_payload["edges"]),
        train_idx=cast(list[int], regions_payload["train_idx"]),
        val_idx=cast(list[int], regions_payload["val_idx"]),
        dropped_featureless_regions=cast(int, regions_payload["dropped_featureless_regions"]),
    )
    return PregateCorpus(
        regions=regions,
        provenance=cast(dict[tuple[int, str], SwapProvenance], payload["provenance"]),
        dropped_cells=cast(tuple[tuple[int, str], ...], payload["dropped_cells"]),
    )


# --------------------------------------------------------------------------- probe model


class PregateProbe(nn.Module):
    """Binary edge-decision probe: `Linear(in_dim, 256) -> ReLU -> Linear(256, 1)`, fp32.

    `in_dim` is a plain constructor argument -- no variant ("p" vs "s") logic lives
    inside the module; the caller picks `in_dim` to match `probe_pair_inputs`'s output.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(P, in_dim)` fp32 input -> `(P,)` fp32 logits."""
        return cast(torch.Tensor, self.net(x).squeeze(-1))


def probe_pair_inputs(
    features: torch.Tensor, pairs: torch.Tensor, ctx: CellContext | None
) -> torch.Tensor:
    """Assemble one cell's `(P, in_dim)` probe input rows for `pairs`.

    Args:
        features: `(n, d)` fp32 region node features, LOCAL index.
        pairs: `(P, 2)` int64 local `(u, v)` indices.
        ctx: The cell's `CellContext` for variant "s" (masked structural context
            concatenated on), or `None` for variant "p" (base block only).

    Returns:
        `(P, 3d)` fp32 for variant "p" (`ctx=None`), `(P, 3d + 4 + 2d)` for variant
        "s": `[x_u + x_v, x_u * x_v, |x_u - x_v|]`, then `pair_context_features` when
        `ctx` is given.
    """
    u, v = pairs[:, 0], pairs[:, 1]
    x_u, x_v = features[u], features[v]
    base = torch.cat([x_u + x_v, x_u * x_v, (x_u - x_v).abs()], dim=1)
    if ctx is None:
        return base
    return torch.cat([base, pair_context_features(ctx, pairs)], dim=1)


# --------------------------------------------------------------------------- trusted rows / quads


def trusted_rows(prov: SwapProvenance) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """One cell's trusted-label training rows: every deleted, inserted, and kept pair.

    Unknown non-edges (the vast majority of node pairs) never appear -- only pairs
    with a swap-corruption-derived ground-truth label do.

    Returns:
        `(pairs, labels)`: `pairs` is `concat(deleted, inserted, kept)`, `(R, 2)`
        int64 local indices; `labels` is `(R,)` fp32, `1.0` for deleted and kept
        (both true edges of the original graph), `0.0` for inserted (false edges).
    """
    pairs = cast(
        NDArray[np.int64], np.concatenate([prov.deleted, prov.inserted, prov.kept], axis=0)
    )
    labels = np.concatenate(
        [
            np.ones(prov.deleted.shape[0], dtype=np.float32),
            np.zeros(prov.inserted.shape[0], dtype=np.float32),
            np.ones(prov.kept.shape[0], dtype=np.float32),
        ]
    )
    return pairs, labels


def quad_comparison_pairs(prov: SwapProvenance) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Paired true-vs-false comparisons built directly from `prov.quads`.

    Built from the quads themselves (not by indexing into `deleted`/`inserted`) to
    avoid row-alignment mistakes: `SwapProvenance`'s documented convention is that
    quad row `(i, j, k, l)` deleted true edges `(i, j)` and `(k, l)`, and inserted
    false edges `(i, l)` and `(k, j)`. For each quad:

    - comparison A: true `sorted(i, j)` vs false `sorted(k, j)`
    - comparison B: true `sorted(k, l)` vs false `sorted(i, l)`

    Returns:
        `(true_pairs, false_pairs)`, each `(2S, 2)` int64 local indices, `S =
        len(prov.quads)`, rows `[A_0, B_0, A_1, B_1, ...]` aligned index-for-index
        between the two arrays.
    """
    s = prov.quads.shape[0]
    i, j, k, m = (prov.quads[:, col] for col in range(4))

    true_pairs = np.empty((2 * s, 2), dtype=np.int64)
    false_pairs = np.empty((2 * s, 2), dtype=np.int64)

    true_pairs[0::2, 0], true_pairs[0::2, 1] = np.minimum(i, j), np.maximum(i, j)
    false_pairs[0::2, 0], false_pairs[0::2, 1] = np.minimum(k, j), np.maximum(k, j)
    true_pairs[1::2, 0], true_pairs[1::2, 1] = np.minimum(k, m), np.maximum(k, m)
    false_pairs[1::2, 0], false_pairs[1::2, 1] = np.minimum(i, m), np.maximum(i, m)

    return true_pairs, false_pairs


def paired_accuracy(scores_true: torch.Tensor, scores_false: torch.Tensor) -> float:
    """Mean paired-comparison accuracy: `1.0` if true > false, `0.5` if equal, else `0.0`."""
    wins = (scores_true > scores_false).to(torch.float32)
    ties = (scores_true == scores_false).to(torch.float32) * 0.5
    return float((wins + ties).mean().item())


def bucketed_macro(
    per_cell: dict[tuple[int, str], float], cell_size: dict[int, int]
) -> dict[str, float]:
    """Bucket per-cell values by `(severity, region size)`, then macro-average buckets.

    Args:
        per_cell: `(region index, severity)` -> that cell's scalar metric.
        cell_size: Region index -> that region's node-count size.

    Returns:
        `{"<severity>|<size>": bucket mean, ..., "macro": unweighted mean over the
        (severity, size) bucket values}`.
    """
    buckets: dict[str, list[float]] = {}
    for (region_idx, severity), value in per_cell.items():
        key = f"{severity}|{cell_size[region_idx]}"
        buckets.setdefault(key, []).append(value)

    result = {key: float(np.mean(values)) for key, values in buckets.items()}
    result["macro"] = float(np.mean(list(result.values()))) if result else float("nan")
    return result


# --------------------------------------------------------------------------- training loop


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append one JSON-encoded record as a line to `path` (the S3 house pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _context_to_device(ctx: CellContext, device: torch.device) -> CellContext:
    """Move every `CellContext` tensor field to `device` (a no-op when already there)."""
    return CellContext(
        features=ctx.features.to(device),
        adj=ctx.adj.to(device),
        deg=ctx.deg.to(device),
        nbr_sum=ctx.nbr_sum.to(device),
        cn=ctx.cn.to(device),
        ra=ctx.ra.to(device),
    )


def train_probe(
    corpus: PregateCorpus,
    cfg: PregateConfig,
    *,
    variant: str,
    seed: int,
    device: torch.device,
    out_path: Path,
    metrics_path: Path,
) -> dict[str, float]:
    """Train and select a `PregateProbe` on `corpus`'s trusted rows and comparison pairs.

    Every (region, severity) provenance cell whose region is in `corpus.regions.train_idx`
    is a training cell; cells in `val_idx` are validation cells. All trusted-row inputs
    and labels (both splits) and all validation comparison-pair inputs are precomputed
    once, fp32 on `device`, before the epoch loop -- `CellContext` (variant "s" only) is
    built on CPU (its internals are unconditionally CPU-constructed) then moved to
    `device` as a whole.

    An epoch is one shuffled pass over training cells, one `AdamW` step per chunk of
    `cfg.regions_per_step` cells (loss = `BCEWithLogitsLoss` over the chunk's
    concatenated trusted rows, `pos_weight` fixed for the whole run from every training
    cell's trusted-label class balance). After each epoch, validation trusted-row loss
    (one pass over every validation cell's concatenated rows) and validation macro
    paired accuracy (`bucketed_macro` over every validation cell's `paired_accuracy`)
    are logged to `metrics_path`; the state dict with the best validation macro paired
    accuracy is kept (ties -> earliest epoch) and saved to `out_path`.

    Args:
        corpus: The swap-corrupted region corpus.
        cfg: Experiment constants (seeds/salt for RNG derivation; `lr`, `weight_decay`,
            `grad_clip`, `epochs`, `regions_per_step` for the loop).
        variant: `"p"` (base block only) or `"s"` (base block + masked context).
        seed: This run's seed, folded into the derived torch/shuffle RNGs.
        device: Single training device; no DDP, no autocast.
        out_path: Checkpoint output path.
        metrics_path: `metrics.jsonl` output path.

    Returns:
        `{"best_epoch": ..., "best_val_macro_paired_acc": ...}`.

    Raises:
        RuntimeError: If an epoch's aggregate training or validation trusted-row loss
            is non-finite.
    """
    torch.manual_seed(_stable_seed(cfg.salt, "train", variant, str(seed)))
    shuffle_rng = np.random.default_rng(_stable_seed(cfg.salt, "shuffle", variant, str(seed)))

    train_idx_set = set(corpus.regions.train_idx)
    val_idx_set = set(corpus.regions.val_idx)
    train_cells = sorted(key for key in corpus.provenance if key[0] in train_idx_set)
    val_cells = sorted(key for key in corpus.provenance if key[0] in val_idx_set)
    cell_size = {idx: len(region) for idx, region in enumerate(corpus.regions.regions)}

    trusted_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    comparison_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    val_cell_set = set(val_cells)

    for key in (*train_cells, *val_cells):
        region_idx, _severity = key
        prov = corpus.provenance[key]
        region = corpus.regions.regions[region_idx]
        n = len(region)
        local_features = corpus.regions.features[list(region)]

        ctx: CellContext | None = None
        if variant == "s":
            ctx_cpu = build_cell_context(n, corrupted_edges_of(prov), local_features)
            ctx = _context_to_device(ctx_cpu, device)

        features_device = local_features.to(device)

        pairs_np, labels_np = trusted_rows(prov)
        pairs_t = torch.from_numpy(pairs_np).to(device)
        labels_t = torch.from_numpy(labels_np).to(device)
        inputs = probe_pair_inputs(features_device, pairs_t, ctx)
        trusted_cache[key] = (inputs, labels_t)

        if key in val_cell_set:
            true_np, false_np = quad_comparison_pairs(prov)
            true_t = torch.from_numpy(true_np).to(device)
            false_t = torch.from_numpy(false_np).to(device)
            comparison_cache[key] = (
                probe_pair_inputs(features_device, true_t, ctx),
                probe_pair_inputs(features_device, false_t, ctx),
            )

    in_dim = next(iter(trusted_cache.values()))[0].shape[1]
    model = PregateProbe(in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_pos = sum(float((trusted_cache[key][1] == 1.0).sum()) for key in train_cells)
    total_neg = sum(float((trusted_cache[key][1] == 0.0).sum()) for key in train_cells)
    pos_weight = torch.tensor(total_neg / total_pos, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_macro = -math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        shuffled = list(train_cells)
        shuffle_rng.shuffle(shuffled)

        chunk_losses: list[tuple[float, int]] = []
        for start in range(0, len(shuffled), cfg.regions_per_step):
            chunk = shuffled[start : start + cfg.regions_per_step]
            inputs = torch.cat([trusted_cache[key][0] for key in chunk], dim=0)
            labels = torch.cat([trusted_cache[key][1] for key in chunk], dim=0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            chunk_losses.append((float(loss.detach()), labels.shape[0]))

        total_rows = sum(weight for _, weight in chunk_losses)
        train_loss = (
            sum(value * weight for value, weight in chunk_losses) / total_rows
            if total_rows
            else float("nan")
        )
        if not math.isfinite(train_loss):
            raise RuntimeError(
                f"non-finite train loss: variant={variant} epoch={epoch} loss={train_loss}"
            )

        model.eval()
        with torch.no_grad():
            val_inputs = torch.cat([trusted_cache[key][0] for key in val_cells], dim=0)
            val_labels = torch.cat([trusted_cache[key][1] for key in val_cells], dim=0)
            val_loss = float(criterion(model(val_inputs), val_labels))

            per_cell: dict[tuple[int, str], float] = {}
            for key in val_cells:
                true_inputs, false_inputs = comparison_cache[key]
                per_cell[key] = paired_accuracy(model(true_inputs), model(false_inputs))

        if not math.isfinite(val_loss):
            raise RuntimeError(f"non-finite val loss: variant={variant} epoch={epoch}")

        val_macro = bucketed_macro(per_cell, cell_size)["macro"]

        _append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_paired_acc": val_macro,
            },
        )

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    torch.save(
        {
            "model": best_state,
            "config": dataclasses.asdict(cfg),
            "variant": variant,
            "seed": seed,
            "in_dim": in_dim,
            "best_epoch": best_epoch,
            "best_val_macro_paired_acc": best_val_macro,
        },
        out_path,
    )

    return {"best_epoch": float(best_epoch), "best_val_macro_paired_acc": best_val_macro}
