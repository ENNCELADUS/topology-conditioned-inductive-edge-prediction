"""Cross-fitted motif-generator folds and predicted-template cache.

Every cached prediction is made by a generator trained only on the opposite
node fold.  Canonical pair keys make lookup independent of endpoint order; the
motif template itself is swapped back to the requested orientation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.data.motif_template import N_EDGES, SWAP_PERM
from src.data.struct_sampler import StructSampler

Pair = tuple[str, str]
CACHE_VERSION = 1


def node_fold(node: str, *, seed: int = 42, folds: int = 2) -> int:
    """Return a stable seeded-hash fold for ``node``."""
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    digest = hashlib.blake2b(f"{seed}:{node}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % folds


def seeded_hash_node_folds(
    nodes: Iterable[str], *, seed: int = 42, folds: int = 2
) -> dict[str, int]:
    """Partition unique nodes without depending on input order or Python hash state."""
    return {node: node_fold(node, seed=seed, folds=folds) for node in sorted(set(nodes))}


def canonical_pair(u: str, v: str) -> Pair:
    """Return the lexical canonical orientation of a pair."""
    return (u, v) if u <= v else (v, u)


def filter_rows_both_endpoints(
    rows: Sequence[tuple[str, str, int]], fold_by_node: Mapping[str, int], fold: int
) -> list[tuple[str, str, int]]:
    """Keep rows whose two endpoints are in ``fold``; missing nodes fail closed."""
    return [row for row in rows if fold_by_node.get(row[0]) == fold_by_node.get(row[1]) == fold]


def eligible_target_fold(pair: Pair, fold_by_node: Mapping[str, int]) -> int | None:
    """Return the common endpoint fold, or ``None`` for cross/self/missing rows."""
    u, v = pair
    if u == v:
        return None
    left, right = fold_by_node.get(u), fold_by_node.get(v)
    return left if left is not None and left == right else None


def enumerate_struct_pairs(
    sampler: StructSampler,
    epoch_steps: Mapping[int, int],
    *,
    seed: int,
) -> list[Pair]:
    """Enumerate the exact deterministic structural-stream pair union.

    ``epoch_steps`` must be the task loader's optimizer-step count for every
    epoch.  This is also the default number of sampled subgraphs per epoch.
    """
    pairs: set[Pair] = set()
    for epoch, steps in sorted(epoch_steps.items()):
        plan = sampler.plan(seed=seed, epoch=epoch, count=steps)
        for subgraph in plan.subgraphs:
            legal = sampler.legal_mask(subgraph)
            for i, u in enumerate(subgraph.nodes):
                for j in range(i + 1, len(subgraph.nodes)):
                    if legal[i, j]:
                        pairs.add(canonical_pair(u, subgraph.nodes[j]))
    return sorted(pairs)


@dataclass(frozen=True)
class CrossfitLookup:
    """Row-aligned tensors supplied to a motif-prompt batch."""

    weights: torch.Tensor
    presence: torch.Tensor | None
    mask: torch.Tensor
    source_folds: torch.Tensor


@dataclass(frozen=True)
class MotifCrossfitCache:
    """Validated, canonical cross-fitted predicted motif rows."""

    pairs: tuple[Pair, ...]
    weights: torch.Tensor
    target_folds: torch.Tensor
    source_folds: torch.Tensor
    seed: int = 42
    presence: torch.Tensor | None = None
    fold_stats: dict[str, int] = field(default_factory=dict)
    _position: dict[Pair, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate tensor shapes, precision, keys, provenance and finiteness."""
        count = len(self.pairs)
        if self.weights.shape != (count, N_EDGES) or self.weights.dtype != torch.float32:
            raise ValueError(f"weights must be fp32 ({count}, {N_EDGES})")
        if self.target_folds.shape != (count,) or self.source_folds.shape != (count,):
            raise ValueError("fold provenance must have one value per cache row")
        if self.target_folds.dtype != torch.int64 or self.source_folds.dtype != torch.int64:
            raise ValueError("fold provenance must be int64")
        if self.presence is not None and (
            self.presence.shape != (count, 3) or self.presence.dtype != torch.float32
        ):
            raise ValueError("presence must be optional fp32 (n, 3)")
        canonical = all(pair == canonical_pair(*pair) for pair in self.pairs)
        if len(set(self.pairs)) != count or not canonical:
            raise ValueError("cache pairs must be unique and canonical")
        if not bool(((self.target_folds >= 0) & (self.target_folds <= 1)).all()):
            raise ValueError("target folds must be 0 or 1")
        if not torch.equal(self.source_folds, 1 - self.target_folds):
            raise ValueError("source fold must be exactly opposite every target fold")
        if not bool(torch.isfinite(self.weights).all()) or (
            self.presence is not None and not bool(torch.isfinite(self.presence).all())
        ):
            raise ValueError("crossfit cache contains non-finite predictions")
        if not bool(((self.weights >= 0.0) & (self.weights <= 1.0)).all()):
            raise ValueError("crossfit weights must lie in [0, 1]")
        if self.presence is not None and not bool(
            ((self.presence >= 0.0) & (self.presence <= 1.0)).all()
        ):
            raise ValueError("crossfit presence probabilities must lie in [0, 1]")
        object.__setattr__(self, "_position", {pair: i for i, pair in enumerate(self.pairs)})
        expected_stats = {
            "rows": count,
            "target_fold_0": int((self.target_folds == 0).sum()),
            "target_fold_1": int((self.target_folds == 1).sum()),
            "source_fold_0": int((self.source_folds == 0).sum()),
            "source_fold_1": int((self.source_folds == 1).sum()),
        }
        if self.fold_stats and self.fold_stats != expected_stats:
            raise ValueError("stored crossfit fold statistics do not match cache rows")
        object.__setattr__(self, "fold_stats", expected_stats)

    @classmethod
    def build(
        cls,
        pairs: Sequence[Pair],
        weights: torch.Tensor,
        *,
        fold_by_node: Mapping[str, int],
        source_folds: Sequence[int],
        seed: int = 42,
        presence: torch.Tensor | None = None,
        forbidden_nodes: frozenset[str] = frozenset(),
    ) -> MotifCrossfitCache:
        """Canonicalise rows and verify no-endpoint-seen provenance."""
        if len(pairs) != len(source_folds) or weights.shape[0] != len(pairs):
            raise ValueError("pairs, predictions and source provenance must align")
        if presence is not None and presence.shape[0] != len(pairs):
            raise ValueError("presence rows must align with pairs")
        order: list[Pair] = []
        rows: list[torch.Tensor] = []
        pres: list[torch.Tensor] = []
        targets: list[int] = []
        sources: list[int] = []
        seen: set[Pair] = set()
        swap = torch.as_tensor(SWAP_PERM, dtype=torch.int64, device=weights.device)
        for index, pair in enumerate(pairs):
            u, v = pair
            if u in forbidden_nodes or v in forbidden_nodes:
                raise ValueError(f"cache row touches a forbidden validation/excluded node: {pair}")
            target = eligible_target_fold(pair, fold_by_node)
            if target is None:
                raise ValueError(f"cache row is not an eligible within-fold nonself pair: {pair}")
            source = int(source_folds[index])
            if source == target:
                raise ValueError(f"source fold {source} saw endpoints of target row {pair}")
            key = canonical_pair(u, v)
            if key in seen:
                raise ValueError(f"duplicate canonical cache pair: {key}")
            seen.add(key)
            row = weights[index].detach().to(dtype=torch.float32, device="cpu")
            if pair != key:
                row = row.index_select(0, swap.cpu())
            order.append(key)
            rows.append(row)
            if presence is not None:
                pres.append(presence[index].detach().to(dtype=torch.float32, device="cpu"))
            targets.append(target)
            sources.append(source)
        permutation = sorted(range(len(order)), key=order.__getitem__)
        order_index = torch.as_tensor(permutation, dtype=torch.int64)
        return cls(
            pairs=tuple(order[i] for i in permutation),
            weights=torch.stack(rows).index_select(0, order_index),
            target_folds=torch.tensor(targets, dtype=torch.int64).index_select(0, order_index),
            source_folds=torch.tensor(sources, dtype=torch.int64).index_select(0, order_index),
            seed=seed,
            presence=(None if presence is None else torch.stack(pres).index_select(0, order_index)),
        )

    def require_pairs(self, pairs: Iterable[Pair], fold_by_node: Mapping[str, int]) -> None:
        """Reject any missing eligible row in a declared task/struct corpus."""
        present = set(self.pairs)
        missing = sorted(
            canonical_pair(*pair)
            for pair in set(pairs)
            if eligible_target_fold(pair, fold_by_node) is not None
            and canonical_pair(*pair) not in present
        )
        if missing:
            raise ValueError(f"crossfit cache misses {len(missing)} eligible pairs: {missing[:5]}")

    def lookup(self, pairs: Sequence[Pair]) -> CrossfitLookup:
        """Look up rows, swapping weights when the requested orientation is reversed."""
        weights = torch.zeros((len(pairs), N_EDGES), dtype=torch.float32)
        presence = (
            None if self.presence is None else torch.zeros((len(pairs), 3), dtype=torch.float32)
        )
        mask = torch.zeros(len(pairs), dtype=torch.bool)
        sources = torch.full((len(pairs),), -1, dtype=torch.int64)
        swap = torch.as_tensor(SWAP_PERM, dtype=torch.int64)
        for row, pair in enumerate(pairs):
            index = self._position.get(canonical_pair(*pair))
            if index is None:
                continue
            value = self.weights[index]
            weights[row] = value if pair == canonical_pair(*pair) else value.index_select(0, swap)
            if presence is not None and self.presence is not None:
                presence[row] = self.presence[index]
            mask[row] = True
            sources[row] = self.source_folds[index]
        return CrossfitLookup(weights, presence, mask, sources)

    def save(self, path: Path) -> None:
        """Write the portable tensor cache atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "seed": self.seed,
            "pairs": list(self.pairs),
            "weights": self.weights,
            "target_folds": self.target_folds,
            "source_folds": self.source_folds,
            "presence": self.presence,
            "fold_stats": self.fold_stats,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> MotifCrossfitCache:
        """Load and validate a cache without accepting arbitrary Python objects."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            raise ValueError(f"unsupported motif crossfit cache at {path}")
        pairs = payload.get("pairs")
        if not isinstance(pairs, list) or not all(
            isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in pairs
        ):
            raise ValueError("crossfit cache pairs are malformed")
        return cls(
            pairs=tuple((str(pair[0]), str(pair[1])) for pair in pairs),
            weights=payload["weights"],
            target_folds=payload["target_folds"],
            source_folds=payload["source_folds"],
            seed=int(payload["seed"]),
            presence=payload.get("presence"),
            fold_stats=payload.get("fold_stats", {}),
        )
