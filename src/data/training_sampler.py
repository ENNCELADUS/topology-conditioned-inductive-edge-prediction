"""Shared teacher/student dynamic edge sampling and offline KD row coverage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.data.pairs import NegativeSampler

Pair = tuple[str, str]


def enumerate_edge_stream(
    training_positives: Sequence[tuple[str, str]],
    sampler: NegativeSampler,
    *,
    negative_ratio: int,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> list[tuple[str, str, int]]:
    """Enumerate one (epoch, rank) edge-stream pair list, deterministically.

    The single source of truth for the training loader
    (`_BatchFactory.epoch_batches`): positives are epoch-shuffled and
    rank-strided; negatives come from the pinned Sec 10.2 sampler seeded by
    ``(seed, epoch, rank)``; the combined list is shuffled with the same stream.

    Args:
        training_positives: Canonical shared training positives (self-pairs included).
        sampler: The pinned negative sampler.
        negative_ratio: Negatives per positive.
        seed: Base seed.
        epoch: 1-based epoch.
        rank: Rank index.
        world_size: Rank count.

    Returns:
        Row list ``(u, v, label)`` for this epoch/rank.
    """
    positives = sorted(training_positives)
    rng = np.random.default_rng((seed, epoch, rank, 0xE5))
    order = np.random.default_rng((seed, epoch, 0xE5)).permutation(len(positives))
    shard = [positives[i] for i in order.tolist()][rank::world_size]
    negatives = sampler.sample(shard, ratio=negative_ratio, seed=seed, epoch=epoch, rank=rank)
    rows = [(u, v, 1) for u, v in shard] + [(u, v, 0) for u, v in negatives]
    perm = rng.permutation(len(rows))
    return [rows[i] for i in perm.tolist()]


@dataclass(frozen=True)
class TrainingCorpus:
    """Unique scoring rows and exact per-epoch indices into those rows."""

    pairs: list[Pair]
    labels: list[int]
    epoch_rows: dict[int, np.ndarray]


def build_training_corpus(
    positives: Sequence[Pair],
    sampler: NegativeSampler,
    *,
    negative_ratio: int,
    seed: int,
    epochs: int,
) -> TrainingCorpus:
    """Precompute dynamic epoch membership without fixing negatives across epochs.

    Use one global sampling rank, independent of scoring shards or training GPU
    count. Distributed student loaders partition these rows after sampling.
    Epoch zero reuses epoch one's membership for warmup/probe planning only.
    """
    if epochs < 1 or negative_ratio < 1:
        raise ValueError("epochs and negative_ratio must be positive")
    pairs: list[Pair] = []
    labels: list[int] = []
    position: dict[Pair, int] = {}
    epoch_rows: dict[int, np.ndarray] = {}
    for epoch in range(1, epochs + 1):
        rows = enumerate_edge_stream(
            positives,
            sampler,
            negative_ratio=negative_ratio,
            seed=seed,
            epoch=epoch,
            rank=0,
            world_size=1,
        )
        indices = np.empty(len(rows), dtype=np.int64)
        for i, (u, v, label) in enumerate(rows):
            pair = (u, v)
            index = position.get(pair)
            if index is None:
                index = len(pairs)
                position[pair] = index
                pairs.append(pair)
                labels.append(label)
            elif labels[index] != label:
                raise ValueError("conflicting labels in dynamic training corpus")
            indices[i] = index
        epoch_rows[epoch] = indices
    epoch_rows[0] = epoch_rows[1]
    return TrainingCorpus(pairs, labels, epoch_rows)
