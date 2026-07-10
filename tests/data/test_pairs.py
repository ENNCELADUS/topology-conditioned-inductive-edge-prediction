"""Tests for token-pair datasets, length-bucketed batching, and the negative sampler."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest
import torch
from src.data.features import FeatureStore
from src.data.pairs import (
    BUCKET_BOUNDARIES,
    LengthBucketedBatchSampler,
    NegativeSampler,
    SharedEpochTokenPairDataset,
    TokenPairDataset,
    collate_pair_indices,
    collate_token_pairs,
    probe_lengths,
)
from src.model.B0 import V3_1

pytestmark = pytest.mark.unit


def _write_feature_root(
    tmp_path: Path,
    node_shapes: dict[str, tuple[int, int]],
    *,
    input_dim: int,
) -> Path:
    """Build a tiny synthetic feature root (metadata.json + index.json + .pt files)."""
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id, (length, dim) in node_shapes.items():
        tensor = torch.randn(length, dim, dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": input_dim, "max_sequence_length": 1024}
        )
    )
    return root


def _canonical(pair: tuple[str, str]) -> tuple[str, str]:
    u, v = pair
    return (u, v) if u <= v else (v, u)


class _CountingFeatureStore(FeatureStore):
    """``FeatureStore`` spy that counts ``load_tokens`` calls, for memoization tests."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.load_tokens_calls = 0

    def load_tokens(self, node_id: str) -> torch.Tensor:
        self.load_tokens_calls += 1
        return super().load_tokens(node_id)


class TestTokenPairDataset:
    def test_getitem_returns_expected_keys_and_shapes(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        pairs = [("node_000001", "node_000002")]
        dataset = TokenPairDataset(pairs, [1], store)

        item = dataset[0]

        assert item["emb_a"].shape == (5, 4)
        assert item["emb_b"].shape == (3, 4)
        assert item["label"].shape == ()
        assert item["label"].dtype == torch.float32
        assert item["label"].item() == 1.0

    def test_getitem_without_labels_omits_label_key(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        dataset = TokenPairDataset([("node_000001", "node_000002")], None, store)

        item = dataset[0]

        assert "label" not in item

    def test_len_matches_number_of_pairs(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        pairs = [("node_000001", "node_000002"), ("node_000002", "node_000001")]
        dataset = TokenPairDataset(pairs, None, store)

        assert len(dataset) == 2

    def test_labels_length_mismatch_raises(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)

        with pytest.raises(ValueError, match="labels"):
            TokenPairDataset([("node_000001", "node_000002")], [1, 0], store)

    def test_lengths_attribute_uses_precomputed_when_given(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        dataset = TokenPairDataset([("node_000001", "node_000002")], None, store, lengths=[(5, 3)])

        assert dataset.lengths == [(5, 3)]

    def test_lengths_attribute_none_when_not_given(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        dataset = TokenPairDataset([("node_000001", "node_000002")], None, store)

        assert dataset.lengths is None


class TestProbeLengths:
    def test_computes_true_lengths_lazily(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        shapes = {
            "node_000001": (5, 4),
            "node_000002": (3, 4),
            "node_000003": (7, 4),
        }
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        pairs = [
            ("node_000001", "node_000002"),
            ("node_000002", "node_000003"),
        ]

        with caplog.at_level(logging.INFO):
            lengths = probe_lengths(store, pairs)

        assert lengths == [(5, 3), (3, 7)]

    def test_memoizes_loads_per_unique_node(self, tmp_path: Path) -> None:
        shapes = {
            "node_000001": (5, 4),
            "node_000002": (3, 4),
            "node_000003": (7, 4),
        }
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = _CountingFeatureStore(root)
        pairs = [
            ("node_000001", "node_000002"),
            ("node_000001", "node_000003"),
            ("node_000002", "node_000003"),
        ]

        lengths = probe_lengths(store, pairs)

        assert lengths == [(5, 3), (5, 7), (3, 7)]
        assert store.load_tokens_calls == 3


class TestCollateTokenPairs:
    def test_pads_to_max_length_and_reports_true_lengths(self) -> None:
        items = [
            {"emb_a": torch.ones(3, 4), "emb_b": torch.ones(2, 4) * 2},
            {"emb_a": torch.ones(5, 4) * 3, "emb_b": torch.ones(1, 4) * 4},
        ]

        batch = collate_token_pairs(items)

        assert set(batch.keys()) == {"emb_a", "emb_b", "len_a", "len_b"}
        assert batch["emb_a"].shape == (2, 5, 4)
        assert batch["emb_b"].shape == (2, 2, 4)
        assert torch.equal(batch["len_a"], torch.tensor([3, 5], dtype=torch.int64))
        assert torch.equal(batch["len_b"], torch.tensor([2, 1], dtype=torch.int64))
        assert torch.all(batch["emb_a"][0, 3:] == 0)
        assert torch.all(batch["emb_b"][1, 1:] == 0)
        assert torch.equal(batch["emb_a"][0, :3], items[0]["emb_a"])
        assert torch.equal(batch["emb_b"][1, :1], items[1]["emb_b"])

    def test_includes_label_key_when_present(self) -> None:
        items = [
            {"emb_a": torch.ones(2, 4), "emb_b": torch.ones(2, 4), "label": torch.tensor(1.0)},
            {"emb_a": torch.ones(2, 4), "emb_b": torch.ones(2, 4), "label": torch.tensor(0.0)},
        ]

        batch = collate_token_pairs(items)

        assert batch["label"].shape == (2,)
        assert batch["label"].dtype == torch.float32
        assert torch.equal(batch["label"], torch.tensor([1.0, 0.0]))

    def test_omits_label_key_when_absent(self) -> None:
        items = [{"emb_a": torch.ones(2, 4), "emb_b": torch.ones(2, 4)}]

        batch = collate_token_pairs(items)

        assert "label" not in batch

    def test_mixed_label_presence_raises(self) -> None:
        items = [
            {"emb_a": torch.ones(2, 4), "emb_b": torch.ones(2, 4), "label": torch.tensor(1.0)},
            {"emb_a": torch.ones(2, 4), "emb_b": torch.ones(2, 4)},
        ]

        with pytest.raises(ValueError, match="label"):
            collate_token_pairs(items)


class TestDescriptorOnlyWorkerPayload:
    def test_shared_epoch_dataset_workers_return_only_row_indices(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        store.preload()
        dataset = SharedEpochTokenPairDataset(
            ["node_000001", "node_000002"],
            capacity=1,
            store=store,
        )
        dataset.replace_epoch([("node_000001", "node_000002")], [1])

        worker_payload = collate_pair_indices([dataset[0]])

        assert worker_payload == [0]
        assert not any(isinstance(value, torch.Tensor) for value in worker_payload)

        batch = dataset.materialize(worker_payload, pin_memory=False)
        assert set(batch) == {"emb_a", "emb_b", "len_a", "len_b", "label"}
        assert batch["emb_a"].shape == (1, 5, 4)
        assert batch["emb_b"].shape == (1, 3, 4)
        assert batch["label"].item() == 1.0


class TestV3_1IntegrationSmoke:
    def test_forward_on_collated_batch_of_four(self, tmp_path: Path) -> None:
        shapes = {
            "node_000001": (4, 1536),
            "node_000002": (6, 1536),
            "node_000003": (8, 1536),
            "node_000004": (5, 1536),
        }
        root = _write_feature_root(tmp_path, shapes, input_dim=1536)
        store = FeatureStore(root)
        pairs = [
            ("node_000001", "node_000002"),
            ("node_000002", "node_000003"),
            ("node_000003", "node_000004"),
            ("node_000004", "node_000001"),
        ]
        labels = [1, 0, 1, 0]
        dataset = TokenPairDataset(pairs, labels, store)
        items = [dataset[i] for i in range(len(dataset))]
        batch = collate_token_pairs(items)

        model = V3_1(
            input_dim=1536,
            d_model=32,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={"hidden_dims": [16], "dropout": 0.0},
            regularization={"dropout": 0.0},
        )
        model.eval()
        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (4, 1)
        assert torch.isfinite(output["loss"]).item()


class TestLengthBucketedBatchSampler:
    def test_every_index_appears_exactly_once_per_epoch(self) -> None:
        lengths = [(10, 20), (300, 100), (500, 100), (1000, 50), (50, 50)] * 4
        sampler = LengthBucketedBatchSampler(lengths, token_budget=1024, seed=0, epoch=0)

        seen: list[int] = []
        for batch in sampler:
            seen.extend(batch)

        assert sorted(seen) == list(range(len(lengths)))

    def test_batches_never_mix_assigned_buckets(self) -> None:
        # Spans 4 distinct buckets (128, 256, 512, 1024) so that a sampler which
        # ignored bucket assignment (e.g. one big pool shuffled and sliced by cap)
        # would produce batches mixing items from different buckets and fail below.
        lengths = [(10, 20), (200, 50), (500, 50), (900, 50)] * 5
        sampler = LengthBucketedBatchSampler(lengths, token_budget=131_072, seed=0, epoch=0)

        def assigned_bucket(index: int) -> int:
            max_len = max(lengths[index])
            return min(b for b in BUCKET_BOUNDARIES if b >= max_len)

        for batch in sampler:
            assigned = {assigned_bucket(i) for i in batch}
            assert len(assigned) == 1, f"batch mixes buckets: {assigned}"
            shared_boundary = next(iter(assigned))
            for i in batch:
                assert max(lengths[i]) <= shared_boundary

    def test_batch_size_respects_token_budget_cap(self) -> None:
        lengths = [(100, 100)] * 50
        token_budget = 2000
        sampler = LengthBucketedBatchSampler(lengths, token_budget=token_budget, seed=0, epoch=0)
        boundary = 128  # smallest boundary >= 100
        cap = max(1, token_budget // (2 * boundary))

        for batch in sampler:
            assert len(batch) <= cap

    def test_reproducible_for_same_seed_and_epoch(self) -> None:
        lengths = [(i * 10 % 900 + 10, (i * 7) % 900 + 10) for i in range(40)]
        batches1 = list(LengthBucketedBatchSampler(lengths, seed=42, epoch=1))
        batches2 = list(LengthBucketedBatchSampler(lengths, seed=42, epoch=1))

        assert batches1 == batches2

    def test_different_epoch_gives_different_batch_order(self) -> None:
        lengths = [(i * 10 % 900 + 10, (i * 7) % 900 + 10) for i in range(40)]
        sampler = LengthBucketedBatchSampler(lengths, seed=42, epoch=0)

        batches_epoch0 = list(sampler)
        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)

        assert batches_epoch0 != batches_epoch1

    def test_replace_epoch_matches_fresh_sampler_for_new_lengths_and_epoch(self) -> None:
        initial_lengths = [(20, 30), (40, 50), (60, 70), (80, 90)]
        next_lengths = [(500, 20), (40, 50), (700, 30), (80, 90)]
        sampler = LengthBucketedBatchSampler(initial_lengths, token_budget=1024, seed=42, epoch=1)

        initial_batches = list(sampler)
        sampler.replace_epoch(next_lengths, epoch=2)
        mutated_batches = list(sampler)
        expected_batches = list(
            LengthBucketedBatchSampler(next_lengths, token_budget=1024, seed=42, epoch=2)
        )

        assert mutated_batches == expected_batches
        assert mutated_batches != initial_batches

    def test_length_exceeding_max_boundary_raises(self) -> None:
        sampler = LengthBucketedBatchSampler([(2000, 10)], seed=0, epoch=0)

        with pytest.raises(ValueError, match="exceeds"):
            list(sampler)


class TestNegativeSampler:
    def test_exact_count_no_global_positives_no_duplicates_canonical(self) -> None:
        train_nodes = [f"node_{i:06d}" for i in range(1, 11)]
        degrees = dict.fromkeys(train_nodes, 1)
        positives = [
            ("node_000001", "node_000002"),
            ("node_000003", "node_000004"),
        ]
        global_positives = frozenset(_canonical(p) for p in positives)
        sampler = NegativeSampler(train_nodes, degrees, global_positives)

        negatives = sampler.sample(positives, ratio=5, seed=0, epoch=0, rank=0)

        assert len(negatives) == 5 * len(positives)
        assert len(set(negatives)) == len(negatives)
        assert global_positives.isdisjoint(negatives)
        for u, v in negatives:
            assert u <= v

    def test_deterministic_for_same_arguments(self) -> None:
        train_nodes = [f"node_{i:06d}" for i in range(1, 21)]
        degrees = {n: i for i, n in enumerate(train_nodes)}
        positives = [("node_000001", "node_000002"), ("node_000003", "node_000004")]
        global_positives = frozenset(_canonical(p) for p in positives)
        sampler = NegativeSampler(train_nodes, degrees, global_positives)

        neg1 = sampler.sample(positives, ratio=3, seed=7, epoch=2, rank=0)
        neg2 = sampler.sample(positives, ratio=3, seed=7, epoch=2, rank=0)

        assert neg1 == neg2

    def test_different_seed_changes_output(self) -> None:
        train_nodes = [f"node_{i:06d}" for i in range(1, 21)]
        degrees = {n: i for i, n in enumerate(train_nodes)}
        positives = [("node_000001", "node_000002"), ("node_000003", "node_000004")]
        global_positives = frozenset(_canonical(p) for p in positives)
        sampler = NegativeSampler(train_nodes, degrees, global_positives)

        neg1 = sampler.sample(positives, ratio=3, seed=1, epoch=0, rank=0)
        neg2 = sampler.sample(positives, ratio=3, seed=2, epoch=0, rank=0)

        assert neg1 != neg2

    def test_degree_corrected_favors_high_degree_node(self) -> None:
        # A large-enough node universe is required so that ~2,000 unique canonical
        # negative pairs (minus the star's own positive edges) actually exist to draw.
        n_leaves = 200
        center = "node_000000"
        leaves = [f"node_{i:06d}" for i in range(1, n_leaves + 1)]
        train_nodes = [center, *leaves]
        n = len(train_nodes)
        degrees = {center: 100_000}
        degrees.update(dict.fromkeys(leaves, 1))
        positives = [(leaves[i], leaves[i + 1]) for i in range(len(leaves) - 1)]
        global_positives = frozenset(_canonical(p) for p in positives)
        sampler = NegativeSampler(train_nodes, degrees, global_positives)

        negatives = sampler.sample(positives, ratio=11, seed=0, epoch=0, rank=0)

        assert len(negatives) >= 2000
        center_count = sum(1 for u, v in negatives if u == center or v == center)
        observed_fraction = center_count / len(negatives)
        uniform_expected_fraction = n / (math.comb(n, 2) + n)
        assert observed_fraction >= 2 * uniform_expected_fraction

    def test_empty_train_nodes_raises(self) -> None:
        with pytest.raises(ValueError, match="train_nodes"):
            NegativeSampler([], {}, frozenset())

    def test_zero_positives_yields_empty_list(self) -> None:
        train_nodes = [f"node_{i:06d}" for i in range(1, 6)]
        degrees = dict.fromkeys(train_nodes, 1)
        sampler = NegativeSampler(train_nodes, degrees, frozenset())

        assert sampler.sample([], ratio=5, seed=0, epoch=0, rank=0) == []
