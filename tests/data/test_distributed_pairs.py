from __future__ import annotations

import pytest
import torch
from src.data.distributed_pairs import (
    CompactPairBatchDataset,
    PairBatchSpec,
    build_distributed_epoch_plan,
    identity_compact_batch,
)


def test_plan_has_exact_coverage_and_equal_step_counts() -> None:
    lengths = [(100, 100)] * 19 + [(300, 200)] * 17
    plan = build_distributed_epoch_plan(
        lengths,
        token_budget_per_rank=2048,
        max_pairs_per_rank=8,
        world_size=4,
        seed=47,
        epoch=2,
        shuffle=True,
    )

    assert len({len(rank_plan) for rank_plan in plan}) == 1
    seen = [index for rank_plan in plan for spec in rank_plan for index in spec.indices]
    assert sorted(seen) == list(range(len(lengths)))
    assert len(seen) == len(set(seen))


def test_plan_is_reproducible() -> None:
    def build_plan() -> list[list[PairBatchSpec]]:
        return build_distributed_epoch_plan(
            [(100, 100)] * 32,
            token_budget_per_rank=2048,
            max_pairs_per_rank=8,
            world_size=4,
            seed=7,
            epoch=3,
            shuffle=True,
        )

    assert build_plan() == build_plan()


def test_tail_batch_records_global_pair_count() -> None:
    plan = build_distributed_epoch_plan(
        [(100, 100)] * 21,
        token_budget_per_rank=1024,
        max_pairs_per_rank=8,
        world_size=4,
        seed=0,
        epoch=0,
        shuffle=False,
    )
    final_specs = [rank_plan[-1] for rank_plan in plan]
    assert sum(len(spec.indices) for spec in final_specs) == final_specs[0].global_pair_count


def test_plan_merges_a_too_small_tail_into_the_previous_step() -> None:
    plan = build_distributed_epoch_plan(
        [(100, 100)] * 17,
        token_budget_per_rank=1024,
        max_pairs_per_rank=8,
        world_size=4,
        seed=0,
        epoch=0,
        shuffle=False,
    )

    assert [len(rank_plan) for rank_plan in plan] == [2, 2, 2, 2]
    assert all(len(spec.indices) <= 4 for rank_plan in plan for spec in rank_plan)
    assert [sum(len(rank_plan[step].indices) for rank_plan in plan) for step in range(2)] == [9, 8]


@pytest.mark.parametrize("row_count", [4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33])
def test_plan_never_exceeds_pair_or_token_caps(row_count: int) -> None:
    lengths = [(100, 100)] * row_count
    plan = build_distributed_epoch_plan(
        lengths,
        token_budget_per_rank=1024,
        max_pairs_per_rank=4,
        world_size=4,
        seed=11,
        epoch=3,
        shuffle=True,
    )

    assert len({len(rank_plan) for rank_plan in plan}) == 1
    seen = [index for rank_plan in plan for spec in rank_plan for index in spec.indices]
    assert sorted(seen) == list(range(row_count))
    assert len(seen) == len(set(seen))
    for rank_plan in plan:
        for spec in rank_plan:
            assert 1 <= len(spec.indices) <= 4
            assert len(spec.indices) * 2 * spec.bucket_boundary <= 1024


def test_plan_rejects_tail_that_cannot_fill_a_synchronized_step() -> None:
    with pytest.raises(ValueError, match="cannot form non-empty synchronized steps"):
        build_distributed_epoch_plan(
            [(100, 100)] * 5,
            token_budget_per_rank=256,
            max_pairs_per_rank=1,
            world_size=4,
            seed=0,
            epoch=0,
            shuffle=False,
        )


def test_plan_rejects_a_bucket_smaller_than_world_size() -> None:
    with pytest.raises(ValueError, match="bucket 128 has 3 rows, fewer than world_size 4"):
        build_distributed_epoch_plan(
            [(100, 100)] * 3,
            token_budget_per_rank=1024,
            max_pairs_per_rank=8,
            world_size=4,
            seed=0,
            epoch=0,
            shuffle=False,
        )


def test_compact_dataset_gathers_only_prebuilt_pair_tensors() -> None:
    specs = [PairBatchSpec(indices=(2, 0), bucket_boundary=256, global_pair_count=3)]
    dataset = CompactPairBatchDataset(
        row_ids=torch.tensor([10, 11, 12]),
        node_a=torch.tensor([20, 21, 22]),
        node_b=torch.tensor([30, 31, 32]),
        labels=torch.tensor([0.0, 1.0, 1.0]),
        batch_specs=specs,
    )

    assert len(dataset) == 1
    batch = dataset[0]
    assert torch.equal(batch.row_ids, torch.tensor([12, 10]))
    assert torch.equal(batch.node_a, torch.tensor([22, 20]))
    assert torch.equal(batch.node_b, torch.tensor([32, 30]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.bucket_boundary == 256
    assert batch.global_pair_count == 3
    assert identity_compact_batch(batch) is batch


def test_compact_dataset_rejects_mismatched_tensor_lengths() -> None:
    with pytest.raises(ValueError, match="same first-dimension length"):
        CompactPairBatchDataset(
            row_ids=torch.tensor([10, 11]),
            node_a=torch.tensor([20]),
            node_b=torch.tensor([30, 31]),
            labels=torch.tensor([0.0, 1.0]),
            batch_specs=[],
        )
