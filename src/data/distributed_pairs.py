"""Deterministic distributed batch planning over compact pair metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.pairs import BUCKET_BOUNDARIES


@dataclass(frozen=True)
class PairBatchSpec:
    """Indices and shared metadata for one rank's part of a global batch."""

    indices: Sequence[int]
    bucket_boundary: int
    global_pair_count: int


@dataclass(frozen=True)
class CompactPairBatch:
    """Pair metadata gathered for one rank without loading feature tensors."""

    row_ids: torch.Tensor
    node_a: torch.Tensor
    node_b: torch.Tensor
    labels: torch.Tensor
    bucket_boundary: int
    global_pair_count: int


def identity_compact_batch(batch: CompactPairBatch) -> CompactPairBatch:
    """Return an already assembled compact batch unchanged."""
    return batch


def build_distributed_epoch_plan(
    lengths: Sequence[tuple[int, int]],
    *,
    token_budget_per_rank: int,
    max_pairs_per_rank: int,
    world_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[list[PairBatchSpec]]:
    """Build equal-step rank plans that cover every pair exactly once.

    Rows are grouped by the existing token-length boundaries. Each bucket is
    optionally shuffled, divided into global chunks, then split across ranks.
    Global chunks are rebalanced so every rank receives a non-empty local batch
    without exceeding either per-rank cap.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if token_budget_per_rank <= 0:
        raise ValueError("token_budget_per_rank must be positive")
    if max_pairs_per_rank <= 0:
        raise ValueError("max_pairs_per_rank must be positive")

    buckets: dict[int, list[int]] = {boundary: [] for boundary in BUCKET_BOUNDARIES}
    for index, (len_a, len_b) in enumerate(lengths):
        boundary = _bucket_boundary(max(len_a, len_b))
        buckets[boundary].append(index)

    rng = np.random.default_rng((seed, epoch))
    plan: list[list[PairBatchSpec]] = [[] for _ in range(world_size)]
    for boundary in BUCKET_BOUNDARIES:
        indices = buckets[boundary]
        if not indices:
            continue
        if len(indices) < world_size:
            raise ValueError(
                f"bucket {boundary} has {len(indices)} rows, fewer than world_size {world_size}"
            )
        if shuffle:
            indices = [indices[position] for position in rng.permutation(len(indices))]

        token_cap = token_budget_per_rank // (2 * boundary)
        local_cap = min(max_pairs_per_rank, token_cap)
        if local_cap < 1:
            raise ValueError(
                f"token_budget_per_rank {token_budget_per_rank} cannot fit bucket {boundary}"
            )
        global_cap = local_cap * world_size
        step_count = (len(indices) + global_cap - 1) // global_cap
        if len(indices) < step_count * world_size:
            raise ValueError(
                f"bucket {boundary} with {len(indices)} rows cannot form non-empty "
                f"synchronized steps under local cap {local_cap}"
            )
        base_chunk_size, extra_rows = divmod(len(indices), step_count)
        chunks: list[list[int]] = []
        start = 0
        for step in range(step_count):
            chunk_size = base_chunk_size + (1 if step < extra_rows else 0)
            chunks.append(indices[start : start + chunk_size])
            start += chunk_size

        for chunk in chunks:
            global_pair_count = len(chunk)
            rank_indices = np.array_split(np.asarray(chunk, dtype=np.int64), world_size)
            for rank, local_indices in enumerate(rank_indices):
                plan[rank].append(
                    PairBatchSpec(
                        indices=tuple(int(index) for index in local_indices),
                        bucket_boundary=boundary,
                        global_pair_count=global_pair_count,
                    )
                )

    return plan


class CompactPairBatchDataset(Dataset[CompactPairBatch]):
    """Gather planned batches from prebuilt compact pair tensors."""

    def __init__(
        self,
        row_ids: torch.Tensor,
        node_a: torch.Tensor,
        node_b: torch.Tensor,
        labels: torch.Tensor,
        batch_specs: Sequence[PairBatchSpec],
    ) -> None:
        """Store compact pair columns and the batch plan for one rank."""
        lengths = {tensor.shape[0] for tensor in (row_ids, node_a, node_b, labels)}
        if len(lengths) != 1:
            raise ValueError("compact pair tensors must have the same first-dimension length")
        self._row_ids = row_ids
        self._node_a = node_a
        self._node_b = node_b
        self._labels = labels
        self._batch_specs = list(batch_specs)

    def __len__(self) -> int:
        """Return the number of planned optimizer steps for this rank."""
        return len(self._batch_specs)

    def __getitem__(self, index: int) -> CompactPairBatch:
        """Gather one planned local batch from the compact pair columns."""
        spec = self._batch_specs[index]
        selected = torch.tensor(spec.indices, dtype=torch.int64)
        return CompactPairBatch(
            row_ids=self._row_ids.index_select(0, selected),
            node_a=self._node_a.index_select(0, selected),
            node_b=self._node_b.index_select(0, selected),
            labels=self._labels.index_select(0, selected),
            bucket_boundary=spec.bucket_boundary,
            global_pair_count=spec.global_pair_count,
        )


def _bucket_boundary(max_len: int) -> int:
    """Return the smallest existing bucket boundary that fits ``max_len``."""
    for boundary in BUCKET_BOUNDARIES:
        if max_len <= boundary:
            return boundary
    raise ValueError(
        f"length {max_len} exceeds the largest bucket boundary {BUCKET_BOUNDARIES[-1]}"
    )


__all__ = [
    "CompactPairBatch",
    "CompactPairBatchDataset",
    "PairBatchSpec",
    "build_distributed_epoch_plan",
    "identity_compact_batch",
]
