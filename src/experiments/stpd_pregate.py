"""STPD pre-gate probe: degree-matched partner discrimination on corrupted training regions.

evidence_class=diagnostic, formal=false; fp32, single GPU, no DDP.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn.utils import clip_grad_norm_

from src.data.features import FeatureStore
from src.distill.teacher_targets import _load_val_region_split, assert_training_side_only
from src.experiments.s2_latent_topology.data import RegionCorpus, build_region_corpus
from src.score_universe import _load_checkpoint, _score_v3_1

logger = logging.getLogger(__name__)

_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_DEFAULT_B0_CHECKPOINT = Path("outputs/deliverables/b0_v31_breadth_first_20260711/model/best.pt")


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


# --------------------------------------------------------------------------- frozen-B0 baseline arm

_B0_TOKEN_BUDGET = 32_768  # mirrors src.distill.content_logit._DEFAULT_TOKEN_BUDGET


def collect_eval_pairs(corpus: PregateCorpus) -> list[tuple[str, str]]:
    """Deduplicated, sorted node-id pairs judged across every held-out comparison cell.

    A held-out cell is a `(region index, severity)` key of `corpus.provenance` whose
    region index is in `corpus.regions.val_idx` (train-only and dropped cells never
    contribute). Every pair from `quad_comparison_pairs(prov)` -- both the true and
    false side of every quad -- is mapped from the cell's local indices to global
    node-id strings via `corpus.regions.regions[region_idx]` and
    `corpus.regions.node_ids`, then normalized as `tuple(sorted((u_id, v_id)))`.

    Args:
        corpus: The swap-corrupted region corpus (`build_pregate_corpus`'s output).

    Returns:
        The deduplicated union of sorted node-id pairs across every held-out cell, in
        sorted (deterministic) order.
    """
    val_idx_set = set(corpus.regions.val_idx)
    pairs: set[tuple[str, str]] = set()
    for (region_idx, _severity), prov in corpus.provenance.items():
        if region_idx not in val_idx_set:
            continue
        region = corpus.regions.regions[region_idx]
        true_pairs, false_pairs = quad_comparison_pairs(prov)
        for local_pairs in (true_pairs, false_pairs):
            for i, j in local_pairs.tolist():
                u_id = corpus.regions.node_ids[region[int(i)]]
                v_id = corpus.regions.node_ids[region[int(j)]]
                a, b = sorted((u_id, v_id))
                pairs.add((a, b))
    return sorted(pairs)


@dataclass(frozen=True)
class B0Scores:
    """Frozen-B0 baseline scoring output: per-pair fp32 logits and the checkpoint's id.

    Attributes:
        logits: `(len(pairs),)` fp32 logits, row-aligned with the input pairs.
        checkpoint_id: The scoring checkpoint's `_load_checkpoint`-derived id, for
            report metadata.
    """

    logits: NDArray[np.float32]
    checkpoint_id: str


def score_b0_pairs(
    checkpoint_path: Path,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
) -> B0Scores:
    """Score `pairs` once with a published `v3_1` checkpoint, fp32, no autocast.

    Args:
        checkpoint_path: Path to a published B0 v3.1 checkpoint.
        pairs: Node-id pairs in input row order (orientation irrelevant to the model).
        store: Feature store providing per-node token sequences.
        device: Compute device.

    Returns:
        The `B0Scores`.

    Raises:
        ValueError: If the checkpoint's model family is not `v3_1`, or any produced
            logit is non-finite.
    """
    model, model_family, checkpoint_id = _load_checkpoint(checkpoint_path)
    if model_family != "v3_1":
        raise ValueError(
            f"score_b0_pairs requires a 'v3_1' checkpoint, got model_family={model_family!r}"
        )
    model.to(device)
    model.eval()

    logits = _score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=_B0_TOKEN_BUDGET
    )
    if not np.isfinite(logits).all():
        raise ValueError(
            f"score_b0_pairs produced a non-finite logit: checkpoint={checkpoint_path}"
        )
    return B0Scores(logits=logits.astype(np.float32), checkpoint_id=checkpoint_id)


def save_b0_scores(
    path: Path,
    pairs: Sequence[tuple[str, str]],
    logits: NDArray[np.float32],
    *,
    checkpoint_id: str,
) -> None:
    """Save `score_b0_pairs` output as an `.npz` artifact.

    Args:
        path: Output `.npz` path (`np.savez` appends the suffix if absent).
        pairs: Node-id pairs, row-aligned with `logits`.
        logits: `(len(pairs),)` fp32 logits.
        checkpoint_id: The scoring checkpoint's id.
    """
    np.savez(
        path,
        u_ids=np.array([u for u, _v in pairs]),
        v_ids=np.array([v for _u, v in pairs]),
        logit=logits.astype(np.float32),
        checkpoint_id=np.array(checkpoint_id),
    )


def load_b0_scores(path: Path) -> tuple[dict[tuple[str, str], float], str]:
    """Load a `save_b0_scores` artifact.

    Lookup semantics: keys are the SORTED node-id pair `tuple(sorted((u, v)))` --
    `collect_eval_pairs` already normalizes every scored pair this way, so a caller
    holding an unordered pair must sort it before looking it up; either original
    orientation resolves to the same entry.

    Args:
        path: Path previously written by `save_b0_scores`.

    Returns:
        `(scores, checkpoint_id)`: `scores` maps each sorted node-id pair to its fp32
        logit (as a Python `float`); `checkpoint_id` is the scoring checkpoint's id.
    """
    with np.load(path) as npz:
        u_ids = npz["u_ids"]
        v_ids = npz["v_ids"]
        logit = npz["logit"]
        checkpoint_id = str(npz["checkpoint_id"])

    scores: dict[tuple[str, str], float] = {}
    for u, v, value in zip(u_ids, v_ids, logit, strict=True):
        a, b = sorted((str(u), str(v)))
        scores[a, b] = float(value)
    return scores, checkpoint_id


# --------------------------------------------------------------------------- stage: cache


def run_cache(
    cfg: PregateConfig,
    *,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    b0_checkpoint: Path,
    device: torch.device,
) -> None:
    """Stage `cache`: sample the corpus, enforce the quarantine, score the frozen-B0 arm.

    Loads the training-side `ValRegionSplit` exactly as training does, samples
    `PregateCorpus`, hard-refuses any test-split/foreign/V_val-internal leakage over
    every node id appearing in a sampled region, then writes `corpus.pt`,
    `b0_scores.npz`, and a small `cache_meta.json` (the `b0_checkpoint` path -- the only
    piece of `run_cache`'s provenance `run_eval` cannot otherwise recover from disk).

    Args:
        cfg: Experiment constants.
        data_root: Benchmark/feature-store root.
        strategy: Split strategy name.
        output_dir: This run's output directory (created if absent).
        b0_checkpoint: Path to a published `v3_1` checkpoint for the frozen-B0 arm.
        device: Scoring device for the frozen-B0 pass.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    split, test_nodes = _load_val_region_split(data_root, strategy)
    train_graph = split.build_training_graph()
    store = FeatureStore(data_root / _FEATURES_SUBDIR)

    corpus = build_pregate_corpus(train_graph, store, cfg, cache_dir=output_dir / "cache")

    node_ids = sorted(
        {corpus.regions.node_ids[i] for region in corpus.regions.regions for i in region}
    )
    assert_training_side_only(node_ids, train_graph, split, test_nodes)

    save_pregate_corpus(corpus, output_dir / "corpus.pt")
    logger.info(
        "cached %d regions (%d dropped cells)",
        len(corpus.regions.regions),
        len(corpus.dropped_cells),
    )

    pairs = collect_eval_pairs(corpus)
    scores = score_b0_pairs(b0_checkpoint, pairs, store, device=device)
    save_b0_scores(
        output_dir / "b0_scores.npz", pairs, scores.logits, checkpoint_id=scores.checkpoint_id
    )
    (output_dir / "cache_meta.json").write_text(
        json.dumps({"b0_checkpoint": str(b0_checkpoint)}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- stage: train


def run_train(cfg: PregateConfig, *, output_dir: Path, device: torch.device) -> None:
    """Stage `train`: train both variants at every configured seed.

    Deletes a pre-existing `metrics_<variant>_s<seed>.jsonl` before each run so a rerun
    never appends to stale lines (`train_probe` itself only ever appends).

    Args:
        cfg: Experiment constants (`cfg.seeds` iterated, `("p", "s")` variants).
        output_dir: Directory holding `corpus.pt` (written by `run_cache`); checkpoints
            and metrics are written here too.
        device: Training device.
    """
    corpus = load_pregate_corpus(output_dir / "corpus.pt")
    for variant in ("p", "s"):
        for seed in cfg.seeds:
            metrics_path = output_dir / f"metrics_{variant}_s{seed}.jsonl"
            metrics_path.unlink(missing_ok=True)
            result = train_probe(
                corpus,
                cfg,
                variant=variant,
                seed=seed,
                device=device,
                out_path=output_dir / f"probe_{variant}_s{seed}.pt",
                metrics_path=metrics_path,
            )
            logger.info("trained variant=%s seed=%d: %s", variant, seed, result)


# --------------------------------------------------------------------------- stage: eval


def _auroc(scores: NDArray[Any], labels: NDArray[Any]) -> float:
    """Rank-based AUROC (Mann-Whitney U form), ties resolved via average rank.

    Args:
        scores: `(N,)` scores; higher means more likely positive.
        labels: `(N,)` binary labels (`1` positive, `0` negative; any numeric dtype).

    Returns:
        AUROC in `[0, 1]`, or `NaN` if either class is empty.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    sum_ranks_pos = float(ranks[labels_arr == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pair_ids(
    region: tuple[int, ...], node_ids: list[str], pairs: NDArray[np.int64]
) -> list[tuple[str, str]]:
    """Map a cell's local `(i, j)` pairs to sorted global node-id pairs."""
    result: list[tuple[str, str]] = []
    for i, j in pairs:
        a, b = sorted((node_ids[region[int(i)]], node_ids[region[int(j)]]))
        result.append((a, b))
    return result


def _lookup_b0(
    b0_scores: dict[tuple[str, str], float], pair_ids: list[tuple[str, str]]
) -> torch.Tensor:
    """Look up `pair_ids` in `b0_scores`; a missing pair is a hard error (cannot happen)."""
    missing = [pid for pid in pair_ids if pid not in b0_scores]
    if missing:
        raise KeyError(
            f"b0 score missing for {len(missing)} held-out comparison pair(s) "
            f"(first few: {missing[:5]}) -- collect_eval_pairs should have covered these"
        )
    return torch.tensor([b0_scores[pid] for pid in pair_ids], dtype=torch.float32)


def _split_buckets(raw: dict[str, float]) -> tuple[dict[str, float], float]:
    """Split a `bucketed_macro` result into `(buckets_without_macro, macro)`."""
    buckets = {key: value for key, value in raw.items() if key != "macro"}
    return buckets, raw["macro"]


def _per_severity_means(buckets: dict[str, float], severities: Sequence[str]) -> dict[str, float]:
    """Unweighted mean over each severity's size buckets."""
    result: dict[str, float] = {}
    for severity in severities:
        values = [value for key, value in buckets.items() if key.startswith(f"{severity}|")]
        result[severity] = float(np.mean(values)) if values else float("nan")
    return result


def _arm_block(
    per_cell: dict[tuple[int, str], float],
    cell_size: dict[int, int],
    severities: Sequence[str],
    auroc_trusted: float | None,
) -> dict[str, Any]:
    """Assemble one non-probe arm's `{buckets, per_severity, macro, auroc_trusted}` block."""
    raw = bucketed_macro(per_cell, cell_size)
    buckets, macro = _split_buckets(raw)
    return {
        "buckets": buckets,
        "per_severity": _per_severity_means(buckets, severities),
        "macro": macro,
        "auroc_trusted": auroc_trusted,
    }


def _mean_sd_over_seeds(
    per_seed: dict[str, dict[str, Any]], severities: Sequence[str]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Seed-mean `{per_severity, macro, auroc_trusted}` and seed-sd `{macro}` for a probe arm."""
    blocks = list(per_seed.values())
    macros = [cast(float, block["macro"]) for block in blocks]
    aurocs = [cast(float, block["auroc_trusted"]) for block in blocks]
    mean_per_severity = {
        severity: float(
            np.mean([cast(dict[str, float], b["per_severity"])[severity] for b in blocks])
        )
        for severity in severities
    }
    mean = {
        "per_severity": mean_per_severity,
        "macro": float(np.mean(macros)),
        "auroc_trusted": float(np.mean(aurocs)),
    }
    sd = {"macro": float(np.std(macros))}
    return mean, sd


def run_eval(cfg: PregateConfig, *, output_dir: Path) -> dict[str, object]:
    """Stage `eval`: score every held-out cell under every arm, aggregate, decide, write.

    Pure given `output_dir`'s contents (`corpus.pt`, `b0_scores.npz`, `cache_meta.json`,
    the six `probe_<variant>_s<seed>.pt` checkpoints written by `run_cache`/`run_train`):
    CPU-only, deterministic, byte-identical on rerun. Also writes `report.json` and
    `tables.md` next to those inputs.

    Args:
        cfg: Experiment constants (`cfg.severities`, `cfg.seeds` drive aggregation).
        output_dir: Directory holding the cache/train stage outputs.

    Returns:
        The report payload (see the module's report-schema documentation).
    """
    corpus = load_pregate_corpus(output_dir / "corpus.pt")
    b0_scores, b0_checkpoint_id = load_b0_scores(output_dir / "b0_scores.npz")
    b0_checkpoint = json.loads((output_dir / "cache_meta.json").read_text())["b0_checkpoint"]

    probes: dict[tuple[str, int], PregateProbe] = {}
    for variant in ("p", "s"):
        for seed in cfg.seeds:
            checkpoint = torch.load(
                output_dir / f"probe_{variant}_s{seed}.pt", map_location="cpu", weights_only=False
            )
            probe = PregateProbe(in_dim=checkpoint["in_dim"])
            probe.load_state_dict(checkpoint["model"])
            probe.eval()
            probes[variant, seed] = probe

    val_idx_set = set(corpus.regions.val_idx)
    held_out_cells = sorted(key for key in corpus.provenance if key[0] in val_idx_set)
    cell_size = {idx: len(region) for idx, region in enumerate(corpus.regions.regions)}
    severities = [name for name, _fraction in cfg.severities]

    per_cell_b0: dict[tuple[int, str], float] = {}
    per_cell_structure_ra: dict[tuple[int, str], float] = {}
    per_cell_structure_cn: dict[tuple[int, str], float] = {}
    per_cell_probe: dict[tuple[str, int], dict[tuple[int, str], float]] = {
        (variant, seed): {} for variant in ("p", "s") for seed in cfg.seeds
    }

    trusted_labels_parts: list[NDArray[np.float32]] = []
    trusted_ra_parts: list[NDArray[np.float32]] = []
    trusted_cn_parts: list[NDArray[np.float32]] = []
    trusted_probe_parts: dict[tuple[str, int], list[NDArray[np.float32]]] = {
        (variant, seed): [] for variant in ("p", "s") for seed in cfg.seeds
    }

    with torch.no_grad():
        for key in held_out_cells:
            region_idx, _severity = key
            prov = corpus.provenance[key]
            region = corpus.regions.regions[region_idx]
            n = len(region)
            features = corpus.regions.features[list(region)]
            ctx = build_cell_context(n, corrupted_edges_of(prov), features)

            true_pairs_np, false_pairs_np = quad_comparison_pairs(prov)
            true_pairs_t = torch.from_numpy(true_pairs_np)
            false_pairs_t = torch.from_numpy(false_pairs_np)

            true_ids = _pair_ids(region, corpus.regions.node_ids, true_pairs_np)
            false_ids = _pair_ids(region, corpus.regions.node_ids, false_pairs_np)
            per_cell_b0[key] = paired_accuracy(
                _lookup_b0(b0_scores, true_ids), _lookup_b0(b0_scores, false_ids)
            )

            ra_true, cn_true = structure_scores(ctx, true_pairs_t)
            ra_false, cn_false = structure_scores(ctx, false_pairs_t)
            per_cell_structure_ra[key] = paired_accuracy(ra_true, ra_false)
            per_cell_structure_cn[key] = paired_accuracy(cn_true, cn_false)

            inputs_p_true = probe_pair_inputs(features, true_pairs_t, None)
            inputs_p_false = probe_pair_inputs(features, false_pairs_t, None)
            inputs_s_true = probe_pair_inputs(features, true_pairs_t, ctx)
            inputs_s_false = probe_pair_inputs(features, false_pairs_t, ctx)
            for seed in cfg.seeds:
                per_cell_probe["p", seed][key] = paired_accuracy(
                    probes["p", seed](inputs_p_true), probes["p", seed](inputs_p_false)
                )
                per_cell_probe["s", seed][key] = paired_accuracy(
                    probes["s", seed](inputs_s_true), probes["s", seed](inputs_s_false)
                )

            trusted_pairs_np, trusted_labels_np = trusted_rows(prov)
            trusted_pairs_t = torch.from_numpy(trusted_pairs_np)
            trusted_labels_parts.append(trusted_labels_np)

            ra_trusted, cn_trusted = structure_scores(ctx, trusted_pairs_t)
            trusted_ra_parts.append(ra_trusted.numpy())
            trusted_cn_parts.append(cn_trusted.numpy())

            inputs_p_trusted = probe_pair_inputs(features, trusted_pairs_t, None)
            inputs_s_trusted = probe_pair_inputs(features, trusted_pairs_t, ctx)
            for seed in cfg.seeds:
                trusted_probe_parts["p", seed].append(probes["p", seed](inputs_p_trusted).numpy())
                trusted_probe_parts["s", seed].append(probes["s", seed](inputs_s_trusted).numpy())

    trusted_labels = np.concatenate(trusted_labels_parts)
    auroc_structure_ra = _auroc(np.concatenate(trusted_ra_parts), trusted_labels)
    auroc_structure_cn = _auroc(np.concatenate(trusted_cn_parts), trusted_labels)
    auroc_probe = {
        key: _auroc(np.concatenate(parts), trusted_labels)
        for key, parts in trusted_probe_parts.items()
    }

    arms: dict[str, Any] = {
        "b0_frozen": _arm_block(per_cell_b0, cell_size, severities, None),
        "structure_ra": _arm_block(
            per_cell_structure_ra, cell_size, severities, auroc_structure_ra
        ),
        "structure_cn": _arm_block(
            per_cell_structure_cn, cell_size, severities, auroc_structure_cn
        ),
    }
    for variant in ("p", "s"):
        per_seed = {
            str(seed): _arm_block(
                per_cell_probe[variant, seed], cell_size, severities, auroc_probe[variant, seed]
            )
            for seed in cfg.seeds
        }
        mean, sd = _mean_sd_over_seeds(per_seed, severities)
        arms[f"probe_{variant}"] = {"per_seed": per_seed, "mean": mean, "sd": sd}

    probe_s_moderate = cast(dict[str, float], arms["probe_s"]["mean"]["per_severity"])["moderate"]
    b0_moderate = cast(dict[str, float], arms["b0_frozen"]["per_severity"])["moderate"]
    structure_ra_moderate = cast(dict[str, float], arms["structure_ra"]["per_severity"])["moderate"]
    margin_vs_b0 = probe_s_moderate - b0_moderate
    margin_vs_structure = probe_s_moderate - structure_ra_moderate
    verdict = (
        "edge_identity_supported"
        if margin_vs_b0 >= 0.03 and margin_vs_structure >= 0.03
        else "edge_identity_killed"
    )
    decision = {
        "rule": (
            "probe_s (seed-mean, moderate severity, size-macro over held-out regions) must "
            "exceed BOTH b0_frozen and structure_ra by >= 0.03 paired accuracy"
        ),
        "margin": 0.03,
        "probe_s_moderate": probe_s_moderate,
        "b0_moderate": b0_moderate,
        "structure_ra_moderate": structure_ra_moderate,
        "margin_vs_b0": margin_vs_b0,
        "margin_vs_structure": margin_vs_structure,
        "verdict": verdict,
    }

    meta = {
        "experiment": "stpd_pregate",
        "format": "stpd_pregate/report_v1",
        "evidence_class": "diagnostic",
        "formal": False,
        "config": dataclasses.asdict(cfg),
        "n_regions": len(corpus.regions.regions),
        "n_train_regions": len(corpus.regions.train_idx),
        "n_val_regions": len(corpus.regions.val_idx),
        "dropped_cells": [[region, severity] for region, severity in corpus.dropped_cells],
        "b0_checkpoint": b0_checkpoint,
        "b0_checkpoint_id": b0_checkpoint_id,
        "n_unique_b0_pairs": len(b0_scores),
        "selection_note": (
            "probe selection on the same held-out regions as this report; optimism favors "
            "the probes, so a kill verdict is conservative"
        ),
        "auroc_note": "b0_frozen has no trusted-row AUROC (scored on comparison pairs only)",
        "vval_note": (
            "synthetic inserted pairs may coincide with quarantined V_val-internal pairs; "
            "their labels derive from insertion provenance, never from V_val truth"
        ),
    }

    payload: dict[str, object] = {"meta": meta, "arms": arms, "decision": decision}

    (output_dir / "report.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "tables.md").write_text(render_tables_markdown(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- report tables


def _fmt(value: float | None) -> str:
    """`""` for `None`/`NaN`, else `f"{value:.6g}"` (the S2 style)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6g}"


_ARM_NAMES: tuple[str, ...] = ("b0_frozen", "structure_ra", "structure_cn", "probe_p", "probe_s")


def _arm_per_severity(entry: dict[str, Any]) -> dict[str, float]:
    """An arm's `per_severity` dict, seed-mean already applied for probe arms."""
    if "per_seed" in entry:
        return cast(dict[str, float], entry["mean"]["per_severity"])
    return cast(dict[str, float], entry["per_severity"])


def _arm_macro(entry: dict[str, Any]) -> float:
    if "per_seed" in entry:
        return cast(float, entry["mean"]["macro"])
    return cast(float, entry["macro"])


def _arm_auroc(entry: dict[str, Any]) -> float | None:
    if "per_seed" in entry:
        return cast(float, entry["mean"]["auroc_trusted"])
    return cast(float | None, entry["auroc_trusted"])


def _arm_bucket_values(entry: dict[str, Any], columns: list[str]) -> dict[str, float]:
    """An arm's bucket values by column, seed-averaged at render time for probe arms."""
    if "per_seed" not in entry:
        return cast(dict[str, float], entry["buckets"])
    per_seed = cast(dict[str, dict[str, Any]], entry["per_seed"])
    result: dict[str, float] = {}
    for column in columns:
        values = [
            value
            for block in per_seed.values()
            if (value := cast(dict[str, float], block["buckets"]).get(column)) is not None
        ]
        result[column] = float(np.mean(values)) if values else float("nan")
    return result


def render_tables_markdown(payload: dict[str, object]) -> str:
    """Render the paired-accuracy, per-severity, AUROC, and decision tables.

    Args:
        payload: A `run_eval` result payload (or an equivalent dict, e.g. loaded back
            from `report.json`).

    Returns:
        Markdown text.
    """
    data = cast(dict[str, Any], payload)
    meta = cast(dict[str, Any], data["meta"])
    config = cast(dict[str, Any], meta["config"])
    arms = cast(dict[str, Any], data["arms"])
    decision = cast(dict[str, Any], data["decision"])

    severities = [str(name) for name, _fraction in config["severities"]]
    sizes = [int(size) for size in config["sizes"]]
    columns = [f"{severity}|{size}" for severity in severities for size in sizes]

    lines: list[str] = ["# STPD pre-gate evaluation tables", ""]

    lines.append("## Paired accuracy: arm x bucket")
    lines.append("")
    lines.append("| arm | " + " | ".join(columns) + " | macro |")
    lines.append("|" + "---|" * (len(columns) + 2))
    for arm_name in _ARM_NAMES:
        entry = arms[arm_name]
        bucket_values = _arm_bucket_values(entry, columns)
        row = " | ".join(_fmt(bucket_values.get(column)) for column in columns)
        lines.append(f"| {arm_name} | {row} | {_fmt(_arm_macro(entry))} |")
    lines.append("")

    lines.append("## Per-severity summary")
    lines.append("")
    lines.append("| arm | " + " | ".join(severities) + " |")
    lines.append("|" + "---|" * (len(severities) + 1))
    for arm_name in _ARM_NAMES:
        per_severity = _arm_per_severity(arms[arm_name])
        row = " | ".join(_fmt(per_severity.get(severity)) for severity in severities)
        lines.append(f"| {arm_name} | {row} |")
    lines.append("")

    lines.append("## AUROC (trusted rows, held-out)")
    lines.append("")
    lines.append("| arm | auroc_trusted |")
    lines.append("|---|---|")
    for arm_name in _ARM_NAMES:
        lines.append(f"| {arm_name} | {_fmt(_arm_auroc(arms[arm_name]))} |")
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"Rule: {decision['rule']}")
    lines.append("")
    lines.append(f"- margin threshold: {_fmt(decision['margin'])}")
    lines.append(f"- probe_s_moderate: {_fmt(decision['probe_s_moderate'])}")
    lines.append(f"- b0_moderate: {_fmt(decision['b0_moderate'])}")
    lines.append(f"- structure_ra_moderate: {_fmt(decision['structure_ra_moderate'])}")
    lines.append(f"- margin_vs_b0: {_fmt(decision['margin_vs_b0'])}")
    lines.append(f"- margin_vs_structure: {_fmt(decision['margin_vs_structure'])}")
    lines.append(f"- verdict: {decision['verdict']}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.stpd_pregate` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.stpd_pregate",
        description="STPD pre-gate diagnostic: cache, train the probes, and evaluate.",
    )
    parser.add_argument("--stage", required=True, choices=("cache", "train", "eval", "all"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--b0-checkpoint", type=Path, default=_DEFAULT_B0_CHECKPOINT)
    return parser


def _require(path: Path, parser: argparse.ArgumentParser, *, prereq_stage: str) -> None:
    """`parser.error` with the missing-prerequisite convention if `path` is absent."""
    if not path.exists():
        parser.error(f"{path} not found (run --stage {prereq_stage} first)")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: dispatch `--stage {cache,train,eval,all}` against `--output-dir`.

    Args:
        argv: Argument list (defaults to `sys.argv[1:]`).

    Returns:
        `0` on success.

    Raises:
        SystemExit: On argument errors or a missing stage prerequisite (`parser.error`).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = PregateConfig()

    stages = ("cache", "train", "eval") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "cache":
            run_cache(
                cfg,
                data_root=args.data_root,
                strategy=args.strategy,
                output_dir=output_dir,
                b0_checkpoint=args.b0_checkpoint,
                device=device,
            )
        elif stage == "train":
            _require(output_dir / "corpus.pt", parser, prereq_stage="cache")
            run_train(cfg, output_dir=output_dir, device=device)
        else:
            _require(output_dir / "corpus.pt", parser, prereq_stage="cache")
            _require(output_dir / "b0_scores.npz", parser, prereq_stage="cache")
            for variant in ("p", "s"):
                for seed in cfg.seeds:
                    _require(
                        output_dir / f"probe_{variant}_s{seed}.pt", parser, prereq_stage="train"
                    )
            run_eval(cfg, output_dir=output_dir)

    return 0


__all__ = ["build_parser", "main", "run_cache", "run_eval", "run_train"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    raise SystemExit(main())
