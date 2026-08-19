"""STPD pre-gate probe: degree-matched partner discrimination on corrupted training regions.

evidence_class=diagnostic, formal=false; fp32, single GPU, no DDP. (Later tasks extend it; keep
the docstring short.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

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
