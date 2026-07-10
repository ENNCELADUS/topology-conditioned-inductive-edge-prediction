"""Token-pair datasets, length-bucketed batching, and the negative pair sampler.

Implements the pair-facing half of the ``V3_1`` batch contract (spec §9/§10):
``TokenPairDataset`` + ``collate_token_pairs`` build the
``emb_a``/``emb_b``/``len_a``/``len_b``/``label`` batch dict, ``LengthBucketedBatchSampler``
groups examples by padded-length bucket under a token budget, and ``NegativeSampler``
implements the degree-corrected negative sampler pinned in spec §10.2.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from ctypes import c_float, c_int64
from multiprocessing import RawArray
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.features import FeatureStore

logger = logging.getLogger(__name__)

BUCKET_BOUNDARIES: tuple[int, ...] = (128, 256, 384, 512, 768, 1024)


class TokenPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Paired node token-sequence dataset feeding the ``V3_1`` batch contract."""

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        labels: Sequence[int] | None,
        store: FeatureStore,
        lengths: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Build a dataset over node-id pairs backed by a frozen feature store.

        Args:
            pairs: Sequence of ``(node_a, node_b)`` id pairs.
            labels: Optional per-pair binary labels; when ``None``, ``__getitem__``
                omits the ``"label"`` key.
            store: Feature store used to load per-node token sequences lazily.
            lengths: Optional precomputed ``(L_a, L_b)`` per example (e.g. from
                :func:`probe_lengths`). Computing lengths from ``index.json`` alone is
                impossible (shapes aren't known until a tensor is loaded), so when this
                is omitted, ``.lengths`` is left as ``None``: callers that need lengths
                for :class:`LengthBucketedBatchSampler` must call :func:`probe_lengths`
                themselves and pass the result in here (or straight to the sampler).

        Raises:
            ValueError: If ``labels`` or ``lengths`` are provided with a length that
                does not match ``pairs``.
        """
        if labels is not None and len(labels) != len(pairs):
            raise ValueError("labels must have the same length as pairs")
        if lengths is not None and len(lengths) != len(pairs):
            raise ValueError("lengths must have the same length as pairs")
        self._pairs = list(pairs)
        self._labels = list(labels) if labels is not None else None
        self._store = store
        self.lengths: list[tuple[int, int]] | None = list(lengths) if lengths is not None else None

    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load token sequences (and label, if present) for one pair.

        Args:
            index: Position in the pair list.

        Returns:
            Dict with ``emb_a``, ``emb_b``, and ``label`` (if labels were given).
        """
        node_a, node_b = self._pairs[index]
        item: dict[str, torch.Tensor] = {
            "emb_a": self._store.load_tokens(node_a),
            "emb_b": self._store.load_tokens(node_b),
        }
        if self._labels is not None:
            item["label"] = torch.tensor(float(self._labels[index]), dtype=torch.float32)
        return item


class SharedEpochTokenPairDataset(Dataset[int]):
    """Fixed-capacity pair dataset whose epoch rows are visible to persistent workers.

    Endpoint indices and labels live in manager-free shared C arrays. The parent replaces
    their contents only between fully exhausted DataLoader iterations. Workers return only
    integer row descriptors; the parent reads the shared rows and materializes padded token
    tensors from its preloaded feature cache, avoiding large tensor IPC through ``/dev/shm``.
    """

    def __init__(self, node_ids: Sequence[str], capacity: int, store: FeatureStore) -> None:
        """Allocate shared epoch storage over a fixed node-id vocabulary.

        Args:
            node_ids: Stable node-id vocabulary used by shared endpoint indices.
            capacity: Exact number of pair rows in every epoch.
            store: Preloaded feature store read by the parent during materialization.

        Raises:
            ValueError: If ``node_ids`` contains duplicates or ``capacity`` is negative.
        """
        self._node_ids = tuple(node_ids)
        self._node_index = {node_id: index for index, node_id in enumerate(self._node_ids)}
        if len(self._node_index) != len(self._node_ids):
            raise ValueError("node_ids must be unique")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self._capacity = capacity
        allocated = max(1, capacity)
        self._a_indices: Any = RawArray(c_int64, allocated)
        self._b_indices: Any = RawArray(c_int64, allocated)
        self._labels: Any = RawArray(c_float, allocated)
        self._store = store

    def __len__(self) -> int:
        """Return the fixed number of rows in every epoch."""
        return self._capacity

    def replace_epoch(self, pairs: Sequence[tuple[str, str]], labels: Sequence[int]) -> None:
        """Replace all endpoint-index and label rows for the next epoch.

        Args:
            pairs: Epoch pair rows in their existing pre-sampler order.
            labels: Binary labels aligned with ``pairs``.

        Raises:
            KeyError: If a pair endpoint is outside the fixed node vocabulary.
            ValueError: If the epoch row count differs from the fixed capacity.
        """
        if len(pairs) != self._capacity or len(labels) != self._capacity:
            raise ValueError("epoch pairs and labels must match the fixed dataset capacity")

        a_indices = np.fromiter(
            (self._node_index[node_a] for node_a, _ in pairs),
            dtype=np.int64,
            count=self._capacity,
        )
        b_indices = np.fromiter(
            (self._node_index[node_b] for _, node_b in pairs),
            dtype=np.int64,
            count=self._capacity,
        )
        label_values = np.asarray(labels, dtype=np.float32)
        np.frombuffer(self._a_indices, dtype=np.int64, count=self._capacity)[:] = a_indices
        np.frombuffer(self._b_indices, dtype=np.int64, count=self._capacity)[:] = b_indices
        np.frombuffer(self._labels, dtype=np.float32, count=self._capacity)[:] = label_values

    def __getitem__(self, index: int) -> int:
        """Return a small row descriptor for worker prefetch and IPC."""
        return index

    def materialize(self, indices: Sequence[int], *, pin_memory: bool) -> dict[str, torch.Tensor]:
        """Build one padded batch in the parent process from preloaded token tensors.

        Args:
            indices: Shared row indices returned by DataLoader workers.
            pin_memory: Whether the newly allocated batch tensors use pinned host memory.

        Returns:
            The unchanged ``V3_1`` batch contract.
        """
        use_pinned_memory = pin_memory and torch.cuda.is_available()
        items: list[dict[str, torch.Tensor]] = []
        for index in indices:
            node_a = self._node_ids[int(self._a_indices[index])]
            node_b = self._node_ids[int(self._b_indices[index])]
            items.append(
                {
                    "emb_a": self._store.load_tokens(node_a),
                    "emb_b": self._store.load_tokens(node_b),
                    "label": torch.tensor(float(self._labels[index]), dtype=torch.float32),
                }
            )
        return collate_token_pairs(items, pin_memory=use_pinned_memory)


def probe_lengths(store: FeatureStore, pairs: Sequence[tuple[str, str]]) -> list[tuple[int, int]]:
    """Compute ``(L_a, L_b)`` per pair by lazily loading each tensor's shape.

    Each unique node id is loaded at most once: a ``node_id -> length`` cache is
    populated as nodes are first seen, so a pair list with heavy node reuse (the
    common case — every node appears in many pairs) costs one tensor load per
    unique node rather than one load per pair-endpoint.

    Args:
        store: Feature store to load raw token sequences from.
        pairs: Sequence of ``(node_a, node_b)`` id pairs.

    Returns:
        A list of ``(L_a, L_b)`` lengths, one per pair, in the same order as ``pairs``.
    """
    unique_node_count = len({node for pair in pairs for node in pair})
    node_lengths: dict[str, int] = {}

    def length_of(node_id: str) -> int:
        cached = node_lengths.get(node_id)
        if cached is not None:
            return cached
        value = int(store.load_tokens(node_id).size(0))
        node_lengths[node_id] = value
        if len(node_lengths) % 1000 == 0:
            logger.info(
                "probe_lengths: loaded %d/%d unique nodes", len(node_lengths), unique_node_count
            )
        return value

    return [(length_of(node_a), length_of(node_b)) for node_a, node_b in pairs]


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups examples by padded-length bucket under a token budget.

    Example ``i`` is assigned to the smallest boundary in ``BUCKET_BOUNDARIES`` that is
    ``>= max(L_a, L_b)``. Batches never mix buckets (so every batch's padded sequence
    length is bounded by its bucket boundary), and the per-batch example count is capped
    so that ``2 * bucket_boundary * batch_size <= token_budget`` (both sequences in a
    pair are padded to the bucket boundary in the worst case).
    """

    def __init__(
        self,
        lengths: Sequence[tuple[int, int]],
        *,
        token_budget: int = 131_072,
        shuffle: bool = True,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        """Build a bucketed batch sampler.

        Args:
            lengths: Per-example ``(L_a, L_b)`` true sequence lengths.
            token_budget: Approximate token budget per batch.
            shuffle: Whether to shuffle within buckets and shuffle batch order.
            seed: Base RNG seed.
            epoch: Current epoch, combined with ``seed`` to derive the RNG stream.
        """
        self._lengths = list(lengths)
        self._token_budget = token_budget
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch used to derive the RNG stream on the next iteration.

        Args:
            epoch: New epoch number.
        """
        self._epoch = epoch

    def replace_epoch(self, lengths: Sequence[tuple[int, int]], *, epoch: int) -> None:
        """Replace per-row lengths and RNG epoch before the next iteration.

        Args:
            lengths: Per-example lengths aligned with the epoch dataset rows.
            epoch: Epoch used with the fixed base seed for bucket shuffling.
        """
        self._lengths = list(lengths)
        self._epoch = epoch

    @staticmethod
    def _bucket_boundary(max_len: int) -> int:
        """Return the smallest bucket boundary ``>= max_len``.

        Args:
            max_len: The example's ``max(L_a, L_b)``.

        Returns:
            The smallest value in ``BUCKET_BOUNDARIES`` that is ``>= max_len``.

        Raises:
            ValueError: If ``max_len`` exceeds the largest bucket boundary.
        """
        for boundary in BUCKET_BOUNDARIES:
            if max_len <= boundary:
                return boundary
        raise ValueError(
            f"length {max_len} exceeds the largest bucket boundary {BUCKET_BOUNDARIES[-1]}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield index batches, grouped by length bucket, for the current epoch."""
        rng = np.random.default_rng((self._seed, self._epoch))
        buckets: dict[int, list[int]] = {boundary: [] for boundary in BUCKET_BOUNDARIES}
        for index, (len_a, len_b) in enumerate(self._lengths):
            boundary = self._bucket_boundary(max(len_a, len_b))
            buckets[boundary].append(index)

        batches: list[list[int]] = []
        for boundary in BUCKET_BOUNDARIES:
            indices = buckets[boundary]
            if not indices:
                continue
            if self._shuffle:
                permutation = rng.permutation(len(indices))
                indices = [indices[i] for i in permutation]
            cap = max(1, self._token_budget // (2 * boundary))
            for start in range(0, len(indices), cap):
                batches.append(indices[start : start + cap])

        if self._shuffle:
            permutation = rng.permutation(len(batches))
            batches = [batches[i] for i in permutation]

        yield from batches


def collate_pair_indices(items: list[int]) -> list[int]:
    """Return worker-fetched row descriptors without creating tensor storage."""
    return items


def collate_token_pairs(
    items: list[dict[str, torch.Tensor]], *, pin_memory: bool = False
) -> dict[str, torch.Tensor]:
    """Collate a list of pair items into the ``V3_1`` batch contract.

    Args:
        items: List of per-example dicts, each with ``emb_a``/``emb_b`` (and
            optionally ``label``), as produced by :class:`TokenPairDataset`.
        pin_memory: Allocate the collated batch in pinned host memory. This is used by
            the V3.1 training loader only after descriptor IPC returns to the parent.

    Returns:
        Dict with keys exactly ``emb_a``, ``emb_b``, ``len_a``, ``len_b``, and (only if
        every item carries a label) ``label``.

    Raises:
        ValueError: If ``items`` is empty, or only some items carry a ``label``.
    """
    if not items:
        raise ValueError("collate_token_pairs requires at least one item")

    has_label = ["label" in item for item in items]
    if any(has_label) and not all(has_label):
        raise ValueError("collate_token_pairs requires all items or no items to have a label")

    len_a = torch.tensor(
        [item["emb_a"].size(0) for item in items],
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    len_b = torch.tensor(
        [item["emb_b"].size(0) for item in items],
        dtype=torch.int64,
        pin_memory=pin_memory,
    )

    batch: dict[str, torch.Tensor] = {
        "emb_a": _pad_stack([item["emb_a"] for item in items], pin_memory=pin_memory),
        "emb_b": _pad_stack([item["emb_b"] for item in items], pin_memory=pin_memory),
        "len_a": len_a,
        "len_b": len_b,
    }
    if all(has_label):
        labels = torch.empty(len(items), dtype=torch.float32, pin_memory=pin_memory)
        for index, item in enumerate(items):
            labels[index] = item["label"]
        batch["label"] = labels
    return batch


def _pad_stack(tensors: list[torch.Tensor], *, pin_memory: bool = False) -> torch.Tensor:
    """Zero-pad a list of ``(L_i, D)`` tensors to a common ``(B, max_L, D)`` tensor.

    Args:
        tensors: List of per-example ``(L_i, D)`` tensors sharing feature dim ``D``.
        pin_memory: Allocate the padded output in pinned host memory.

    Returns:
        A zero-padded ``(len(tensors), max_L, D)`` float32 tensor.
    """
    max_len = max(tensor.size(0) for tensor in tensors)
    dim = tensors[0].size(1)
    out = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32, pin_memory=pin_memory)
    for i, tensor in enumerate(tensors):
        out[i, : tensor.size(0)] = tensor
    return out


class NegativeSampler:
    """Degree-corrected negative pair sampler (implementation spec §10.2, pinned).

    Half of proposals are drawn uniformly over ``train_nodes^2`` (with a
    universe-rate self-pair boost); the other half pick a random positive and replace
    one endpoint with a node drawn with probability proportional to degree. Proposals
    are rejected if they canonicalize to a global positive or a pair already sampled
    in this call.
    """

    def __init__(
        self,
        train_nodes: Sequence[str],
        degrees: Mapping[str, int],
        global_positives: frozenset[tuple[str, str]],
    ) -> None:
        """Build a negative sampler over a fixed train-node universe.

        Args:
            train_nodes: Node ids eligible for negative pair endpoints.
            degrees: Simple-graph degree per node (from ``G_struct``); nodes absent
                from the mapping are treated as degree 0. If every node has degree 0,
                the degree-corrected replacement draw falls back to uniform.
            global_positives: Canonicalized global positive pairs; never sampled.

        Raises:
            ValueError: If ``train_nodes`` is empty.
        """
        if not train_nodes:
            raise ValueError("train_nodes must be non-empty")
        self._train_nodes = list(train_nodes)
        self._n = len(self._train_nodes)
        self._global_positives = global_positives

        weights = np.array(
            [float(degrees.get(node, 0)) for node in self._train_nodes], dtype=np.float64
        )
        total = weights.sum()
        self._degree_weights = (
            weights / total if total > 0 else np.full(self._n, 1.0 / self._n, dtype=np.float64)
        )
        self._p_self = self._n / (math.comb(self._n, 2) + self._n)

    @staticmethod
    def _canonicalize(u: str, v: str) -> tuple[str, str]:
        """Return the canonical ``(min(u, v), max(u, v))`` ordering of a pair."""
        return (u, v) if u <= v else (v, u)

    def sample(
        self,
        positives: Sequence[tuple[str, str]],
        *,
        ratio: int = 5,
        seed: int = 0,
        epoch: int = 0,
        rank: int = 0,
    ) -> list[tuple[str, str]]:
        """Sample a deterministic set of negative pairs for one epoch/rank.

        Args:
            positives: Positive pairs sizing the target negative count, and (for the
                degree-corrected branch) providing endpoint-replacement proposals.
            ratio: Target negative count = ``ratio * len(positives)``.
            seed: Base RNG seed.
            epoch: Current epoch.
            rank: Distributed rank; combined with ``seed``/``epoch`` to derive an
                independent RNG stream per rank.

        Returns:
            A list of canonicalized negative pairs of length ``ratio * len(positives)``,
            in draw order. Identical arguments always produce an identical list.

        Raises:
            RuntimeError: If the target count cannot be reached within a generous
                attempt budget (e.g. the eligible universe is exhausted).
        """
        target = ratio * len(positives)
        if target <= 0:
            return []

        rng = np.random.default_rng((seed, epoch, rank))
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        max_attempts = max(target * 50, 10_000)
        attempts = 0

        while len(result) < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"NegativeSampler could not reach target count {target} after "
                    f"{attempts - 1} attempts"
                )
            if rng.random() < 0.5:
                pair = self._propose_uniform(rng)
            else:
                proposal = self._propose_degree_corrected(rng, positives)
                if proposal is None:
                    continue
                pair = proposal

            canonical = self._canonicalize(*pair)
            if canonical in self._global_positives or canonical in seen:
                continue
            seen.add(canonical)
            result.append(canonical)

        return result

    def _propose_uniform(self, rng: np.random.Generator) -> tuple[str, str]:
        """Draw a uniform proposal, applying the universe-rate self-pair boost."""
        if rng.random() < self._p_self:
            index = int(rng.integers(self._n))
            return self._train_nodes[index], self._train_nodes[index]
        index_u = int(rng.integers(self._n))
        index_v = int(rng.integers(self._n))
        return self._train_nodes[index_u], self._train_nodes[index_v]

    def _propose_degree_corrected(
        self, rng: np.random.Generator, positives: Sequence[tuple[str, str]]
    ) -> tuple[str, str] | None:
        """Draw a degree-corrected proposal by replacing one endpoint of a positive."""
        if not positives:
            return None
        pos_index = int(rng.integers(len(positives)))
        node_u, node_v = positives[pos_index]
        replacement_index = int(rng.choice(self._n, p=self._degree_weights))
        replacement = self._train_nodes[replacement_index]
        if rng.random() < 0.5:
            return replacement, node_v
        return node_u, replacement


__all__ = [
    "BUCKET_BOUNDARIES",
    "LengthBucketedBatchSampler",
    "NegativeSampler",
    "SharedEpochTokenPairDataset",
    "TokenPairDataset",
    "collate_pair_indices",
    "collate_token_pairs",
    "probe_lengths",
]
