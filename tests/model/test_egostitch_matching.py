"""Tests for src.model.egostitch.matching: the two-pass Hungarian assignment."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.matching import (
    Assignment,
    base_match_cost,
    deg_bucket_index,
    hungarian_assign,
    match_slots,
    overlap_penalty,
)

pytestmark = pytest.mark.unit


def _slot_set(
    h: torch.Tensor, *, mult: torch.Tensor | None = None, adj: torch.Tensor | None = None
) -> SlotSet:
    b, k, _ = h.shape
    adjacency = torch.zeros(b, k, k) if adj is None else adj
    return SlotSet(
        h=h,
        pi=torch.full((b, k), 0.5),
        mult=torch.ones(b, k) if mult is None else mult,
        gate=torch.zeros(b, k),
        pointer=torch.full((b, k, 1), 1.0),
        adj=adjacency,
        adj_logits=torch.logit(adjacency.clamp(1e-6, 1.0 - 1e-6)),
    )


class TestDegBucketIndex:
    def test_pinned_buckets(self) -> None:
        m = torch.tensor([1.0, 1.9, 2.0, 3.9, 4.0, 7.9, 8.0, 15.9, 16.0, 32.0])
        out = deg_bucket_index(m)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        torch.testing.assert_close(out, expected)

    def test_below_one_clamps_to_first_bucket(self) -> None:
        assert float(deg_bucket_index(torch.tensor([0.1]))) == 0.0


class TestBaseMatchCost:
    def test_hand_case(self) -> None:
        # One slot at origin; two targets at distance^2 = 1 and 4.
        h = torch.zeros(1, 1, 2)
        slots = _slot_set(h)
        target_proj = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        target_mult = torch.tensor([[1.0, 4.0]])
        cost = base_match_cost(slots, target_proj, target_mult)
        # feat terms: (7/6)*1, (7/6)*4; bucket terms: |0-0|=0, |0-2|=2 -> (7/24)*2.
        assert cost[0, 0, 0] == pytest.approx(7.0 / 6.0)
        assert cost[0, 0, 1] == pytest.approx(7.0 / 6.0 * 4.0 + 7.0 / 24.0 * 2.0)


class TestHungarianAssign:
    def test_hand_computable_toy(self) -> None:
        # Costs pinned so the unique optimum is slot0->t1, slot1->t0.
        cost = torch.tensor([[[10.0, 1.0], [2.0, 10.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        assignment = hungarian_assign(cost, mask)
        np.testing.assert_array_equal(np.sort(assignment.slot_idx[0]), [0, 1])
        pairing = dict(zip(assignment.slot_idx[0], assignment.target_idx[0], strict=True))
        assert pairing[0] == 1
        assert pairing[1] == 0

    def test_masked_targets_excluded(self) -> None:
        cost = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[True, False]])
        assignment = hungarian_assign(cost, mask)
        assert assignment.target_idx[0].tolist() == [0]

    def test_empty_targets(self) -> None:
        cost = torch.zeros(1, 2, 2)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        assignment = hungarian_assign(cost, mask)
        assert assignment.slot_idx[0].size == 0

    def test_no_gradient_through_assignment(self) -> None:
        cost = torch.rand(1, 3, 3, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        hungarian_assign(cost, mask)
        # The assignment is numpy-side; the cost tensor accumulates no grad_fn use.
        assert cost.grad is None


class TestOverlapPenalty:
    def test_agreeing_adjacency_zero_penalty(self) -> None:
        adj = torch.zeros(1, 2, 2)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        slots = _slot_set(torch.zeros(1, 2, 2), adj=adj)
        target_adj = torch.zeros(1, 2, 2)
        target_adj[0, 0, 1] = target_adj[0, 1, 0] = 1.0
        assignment = Assignment(
            [np.array([0, 1], dtype=np.int64)], [np.array([0, 1], dtype=np.int64)]
        )
        penalty = overlap_penalty(slots, target_adj, assignment)
        # Candidate (0, 0): matched pair (1, 1) gives |adj[0,1] - A[0,1]| = 0.
        assert float(penalty[0, 0, 0]) == pytest.approx(0.0)
        assert float(penalty[0, 1, 1]) == pytest.approx(0.0)

    def test_disagreeing_adjacency_positive_penalty(self) -> None:
        adj = torch.zeros(1, 2, 2)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        slots = _slot_set(torch.zeros(1, 2, 2), adj=adj)
        target_adj = torch.zeros(1, 2, 2)  # targets NOT adjacent
        assignment = Assignment(
            [np.array([0, 1], dtype=np.int64)], [np.array([0, 1], dtype=np.int64)]
        )
        penalty = overlap_penalty(slots, target_adj, assignment)
        assert float(penalty[0, 0, 0]) == pytest.approx(1.0)

    def test_single_match_no_penalty(self) -> None:
        slots = _slot_set(torch.zeros(1, 2, 2))
        assignment = Assignment([np.array([0], dtype=np.int64)], [np.array([0], dtype=np.int64)])
        penalty = overlap_penalty(slots, torch.zeros(1, 2, 2), assignment)
        assert float(penalty.abs().sum()) == 0.0


class TestMatchSlots:
    def _inputs(self) -> tuple[SlotSet, dict[str, torch.Tensor]]:
        gen = torch.Generator().manual_seed(0)
        h = torch.randn(2, 4, 3, generator=gen)
        slots = _slot_set(h, adj=torch.rand(2, 4, 4, generator=gen))
        targets = {
            "target_proj": torch.randn(2, 4, 3, generator=gen),
            "target_mult": torch.ones(2, 4),
            "target_adj": (torch.rand(2, 4, 4, generator=gen) > 0.5).float(),
            "target_mask": torch.tensor([[True, True, True, False], [True] * 4]),
        }
        return slots, targets

    def test_two_pass_deterministic(self) -> None:
        slots, targets = self._inputs()
        first = match_slots(slots, **targets)
        second = match_slots(slots, **targets)
        assert first == second

    def test_respects_target_mask(self) -> None:
        slots, targets = self._inputs()
        assignment = match_slots(slots, **targets)
        assert 3 not in assignment.target_idx[0].tolist()
        assert len(assignment.target_idx[0]) == 3
        assert len(assignment.target_idx[1]) == 4

    def test_nearest_targets_win_with_identical_buckets(self) -> None:
        h = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
        slots = _slot_set(h)
        targets = {
            "target_proj": torch.tensor([[[9.9, 10.0], [0.1, 0.0]]]),
            "target_mult": torch.ones(1, 2),
            "target_adj": torch.zeros(1, 2, 2),
            "target_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        assignment = match_slots(slots, **targets)
        pairing = dict(zip(assignment.slot_idx[0], assignment.target_idx[0], strict=True))
        assert pairing[0] == 1
        assert pairing[1] == 0
