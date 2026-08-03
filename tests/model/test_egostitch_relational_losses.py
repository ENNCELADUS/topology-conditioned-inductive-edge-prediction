"""Task-7 relational-loss contracts from EgoStitch spec Sec 14.4.1."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
import torch
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.assemble import Assignment, match_slots, sinkhorn_log_plan
from src.model.egostitch.generator.imagine import EgoStitchStage1, SlotSet
from src.model.egostitch.generator.losses import (
    alignment_loss,
    alignment_teacher_cells,
    relational_loss,
)
from src.train_egostitch import relational_pair_targets
from torch import nn

pytestmark = pytest.mark.unit


def _slots(h: torch.Tensor) -> SlotSet:
    batch, slots, _ = h.shape
    adj = torch.zeros(batch, slots, slots)
    return SlotSet(
        h=h,
        pi=torch.full((batch, slots), 0.8),
        mult=torch.ones(batch, slots),
        gate=torch.zeros(batch, slots),
        pointer=torch.ones(batch, slots, 1),
        adj=adj,
        adj_logits=torch.zeros_like(adj),
    )


def _match(slots: SlotSet, target_proj: torch.Tensor, target_mask: torch.Tensor) -> Assignment:
    batch, targets, _ = target_proj.shape
    return match_slots(
        slots,
        target_proj=target_proj,
        target_mult=torch.ones(batch, targets),
        target_adj=torch.zeros(batch, targets, targets),
        target_mask=target_mask,
    )


def test_teacher_cells_follow_same_real_shared_neighbor_on_hand_built_graph() -> None:
    graph = nx.Graph([("u", "w"), ("v", "w"), ("u", "x")])
    node_index = {node: index for index, node in enumerate(sorted(graph.nodes))}
    feature = {
        node: torch.tensor([float(index), float(index**2)])
        for node, index in node_index.items()
    }

    target_a = torch.stack([feature[node] for node in sorted(graph.neighbors("u"))])[None]
    target_b = torch.stack([feature[node] for node in sorted(graph.neighbors("v"))])[None]
    # Endpoint-a slots deliberately reverse the target order; Hungarian must recover it.
    slots_a = _slots(target_a.flip(1).clone().requires_grad_())
    slots_b = _slots(
        torch.cat([target_b, torch.tensor([[[99.0, 99.0]]])], dim=1).requires_grad_()
    )
    assignment_a = _match(slots_a, target_a, torch.ones(1, 2, dtype=torch.bool))
    assignment_b = _match(slots_b, target_b, torch.ones(1, 1, dtype=torch.bool))
    target_ids_a = torch.tensor(
        [[node_index[node] for node in sorted(graph.neighbors("u"))]], dtype=torch.long
    )
    target_ids_b = torch.tensor(
        [[node_index[node] for node in sorted(graph.neighbors("v"))]], dtype=torch.long
    )

    teacher = alignment_teacher_cells(
        assignment_a,
        assignment_b,
        target_ids_a=target_ids_a,
        target_ids_b=target_ids_b,
        slots=2,
    )
    expected = torch.zeros(1, 2, 2, dtype=torch.bool)
    expected[0, 1, 0] = True  # both slots were matched to real node "w"
    assert torch.equal(teacher, expected)

    no_shared_a = Assignment([np.array([0], dtype=np.int64)], [np.array([0], dtype=np.int64)])
    no_shared_b = Assignment([np.array([0], dtype=np.int64)], [np.array([0], dtype=np.int64)])
    no_shared = alignment_teacher_cells(
        no_shared_a,
        no_shared_b,
        target_ids_a=torch.tensor([[node_index["w"]]]),
        target_ids_b=torch.tensor([[node_index["u"]]]),
        slots=2,
    )
    assert not no_shared.any()

    plan = torch.full((1, 2, 2), 0.25, requires_grad=True)
    skipped = alignment_loss(
        torch.log(plan),
        no_shared,
        positive_real_mask=torch.ones(1),
    )
    assert float(skipped.detach()) == 0.0
    assert torch.isfinite(skipped)
    skipped.backward()
    assert plan.grad is not None
    assert torch.equal(plan.grad, torch.zeros_like(plan.grad))


def test_alignment_loss_and_gradients_are_ab_ba_invariant() -> None:
    plan_ab = torch.tensor(
        [[[0.31, 0.07, 0.02], [0.05, 0.22, 0.03], [0.04, 0.09, 0.17]]],
        requires_grad=True,
    )
    teacher_ab = torch.zeros(1, 3, 3, dtype=torch.bool)
    teacher_ab[0, 0, 1] = True
    teacher_ab[0, 2, 2] = True
    loss_ab = alignment_loss(torch.log(plan_ab), teacher_ab, positive_real_mask=torch.ones(1))
    (grad_ab,) = torch.autograd.grad(loss_ab, plan_ab)

    plan_ba = plan_ab.detach().transpose(1, 2).clone().requires_grad_()
    loss_ba = alignment_loss(
        torch.log(plan_ba),
        teacher_ab.transpose(1, 2),
        positive_real_mask=torch.ones(1),
    )
    (grad_ba,) = torch.autograd.grad(loss_ba, plan_ba)

    torch.testing.assert_close(loss_ab, loss_ba, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(grad_ab, grad_ba.transpose(1, 2), rtol=0.0, atol=1e-6)


def test_default_fresh_slots_have_live_alignment_loss_and_gradients() -> None:
    torch.manual_seed(20260726)
    config = EgoStitchConfig()
    model = EgoStitchStage1(config)
    state_a = model.encode_nodes(
        torch.randn(1, config.input_dim),
        torch.randn(1, config.n_ground, config.input_dim),
    )
    state_b = model.encode_nodes(
        torch.randn(1, config.input_dim),
        torch.randn(1, config.n_ground, config.input_dim),
    )
    state_a.slots.h.retain_grad()
    state_b.slots.h.retain_grad()
    log_plan = sinkhorn_log_plan(
        state_a.slots.h,
        state_b.slots.h,
        state_a.slots.pi,
        state_b.slots.pi,
        state_a.slots.mult,
        state_b.slots.mult,
        eps=config.sinkhorn_eps,
        iters=config.sinkhorn_iters,
        tau=config.sinkhorn_tau,
    )
    shared_assignment = Assignment(
        [np.array([0], dtype=np.int64)],
        [np.array([0], dtype=np.int64)],
    )
    teacher = alignment_teacher_cells(
        shared_assignment,
        shared_assignment,
        target_ids_a=torch.tensor([[7]]),
        target_ids_b=torch.tensor([[7]]),
        slots=config.slots,
    )
    assert bool(teacher.any())

    loss = alignment_loss(log_plan, teacher, positive_real_mask=torch.ones(1))
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert state_a.slots.h.grad is not None
    assert state_b.slots.h.grad is not None
    assert bool((state_a.slots.h.grad != 0).any())
    assert bool((state_b.slots.h.grad != 0).any())


def test_masked_alignment_reduction_ignores_duplicated_filler_rows() -> None:
    plan = torch.tensor([[[0.6, 0.1], [0.2, 0.1]]])
    teacher = torch.tensor([[[True, False], [False, False]]])
    single = alignment_loss(torch.log(plan), teacher, positive_real_mask=torch.ones(1))
    padded = alignment_loss(
        torch.log(torch.cat([plan, plan * 0.01], dim=0)),
        torch.cat([teacher, teacher], dim=0),
        positive_real_mask=torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(padded, single)


class _ToyRelationalParameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.align_h_a = nn.Parameter(
            torch.tensor([[0.2, -0.1], [0.7, 0.3]], dtype=torch.float32)
        )
        self.align_h_b = nn.Parameter(
            torch.tensor([[0.1, 0.4], [0.6, -0.2]], dtype=torch.float32)
        )
        self.rel_head = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))

    def plans(self, pair_ids: torch.Tensor, *, ba_mask: torch.Tensor) -> torch.Tensor:
        offsets = pair_ids.float().view(-1, 1, 1) * torch.tensor(
            [[[0.03, -0.02], [0.01, 0.04]]], dtype=torch.float32
        )
        h_a = self.align_h_a.unsqueeze(0) + offsets
        h_b = self.align_h_b.unsqueeze(0) - offsets
        pi_a = torch.tensor([[0.8, 0.6]], dtype=torch.float32).expand(len(pair_ids), -1)
        pi_b = torch.tensor([[0.7, 0.9]], dtype=torch.float32).expand(len(pair_ids), -1)
        mult = torch.ones(len(pair_ids), 2)
        plans_ab = sinkhorn_log_plan(h_a, h_b, pi_a, pi_b, mult, mult, iters=8)
        plans_ba = sinkhorn_log_plan(h_b, h_a, pi_b, pi_a, mult, mult, iters=8)
        return torch.where(ba_mask.view(-1, 1, 1), plans_ba, plans_ab)


def _world_size_teacher_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    assignments = Assignment(
        [np.array([0, 1], dtype=np.int64) for _ in range(3)],
        [np.array([0, 1], dtype=np.int64) for _ in range(3)],
    )
    teacher = alignment_teacher_cells(
        assignments,
        assignments,
        target_ids_a=torch.tensor([[10, 11], [20, 21], [30, 31]]),
        target_ids_b=torch.tensor([[10, 12], [22, 21], [31, 33]]),
        slots=2,
    )
    graph = nx.Graph(
        [
            ("a", "x"),
            ("b", "x"),
            ("c", "x"),
            ("c", "y"),
            ("d", "y"),
            ("d", "z"),
            ("e", "x"),
            ("e", "y"),
            ("e", "z"),
            ("f", "y"),
            ("f", "z"),
            ("f", "q"),
        ]
    )
    targets = relational_pair_targets(
        graph,
        [("a", "b", 1), ("c", "d", 0), ("e", "f", 1)],
    )
    return teacher, targets


def test_relational_targets_remove_the_queried_edge_before_jaccard() -> None:
    graph = nx.Graph([("u", "v"), ("u", "x"), ("v", "x")])

    target = relational_pair_targets(graph, [("u", "v", 1)])

    torch.testing.assert_close(target[0], torch.tensor([math.log(2.0), 1.0]))


def _combined_loss(
    model: _ToyRelationalParameters,
    pair_ids: torch.Tensor,
    *,
    ba_mask: torch.Tensor,
    labels: torch.Tensor,
    edge_mask: torch.Tensor,
    world_size: int,
    align_denominator: torch.Tensor,
    rel_denominator: torch.Tensor,
) -> torch.Tensor:
    teacher_ab, full_targets = _world_size_teacher_and_targets()
    teacher = teacher_ab.index_select(0, pair_ids)
    teacher = torch.where(ba_mask.view(-1, 1, 1), teacher.transpose(1, 2), teacher)
    pair_states = torch.tensor(
        [[0.2, -0.4, 0.7], [0.5, 0.1, -0.2], [-0.3, 0.8, 0.4]]
    ).index_select(0, pair_ids)
    targets = full_targets.index_select(0, pair_ids)
    align = alignment_loss(
        model.plans(pair_ids, ba_mask=ba_mask),
        teacher,
        positive_real_mask=edge_mask * (labels == 1).float(),
        world_size=world_size,
        all_reduce_sum=lambda _: align_denominator,
    )
    rel = relational_loss(
        model.rel_head(pair_states),
        targets,
        edge_mask,
        world_size=world_size,
        all_reduce_sum=lambda _: rel_denominator,
    )
    return 0.5 * align + 0.25 * rel


def test_per_pair_world_size_invariance_targets_losses_and_per_parameter_gradients() -> None:
    torch.manual_seed(19)
    full_model = _ToyRelationalParameters()
    initial = full_model.state_dict()
    pair_ids = torch.tensor([0, 1, 2])
    full_loss = _combined_loss(
        full_model,
        pair_ids,
        ba_mask=torch.tensor([False, False, True]),
        labels=torch.tensor([1, 0, 1]),
        edge_mask=torch.ones(3),
        world_size=1,
        align_denominator=torch.tensor(2.0),
        rel_denominator=torch.tensor(3.0),
    )
    full_gradients = dict(
        zip(
            (name for name, _ in full_model.named_parameters()),
            torch.autograd.grad(full_loss, tuple(full_model.parameters())),
            strict=True,
        )
    )

    rank_specs = (
        (
            torch.tensor([2, 0]),
            torch.tensor([True, False]),
            torch.tensor([1, 1]),
            torch.tensor([1.0, 1.0]),
        ),
        # Rank 1 has an unequal tail, represents pair 1 as BA, and pads by duplicating it.
        (
            torch.tensor([1, 1]),
            torch.tensor([True, True]),
            torch.tensor([0, 0]),
            torch.tensor([1.0, 0.0]),
        ),
    )
    rank_losses: list[torch.Tensor] = []
    rank_gradients: list[dict[str, torch.Tensor]] = []
    reconstructed_targets = torch.zeros(3, 2)
    _, full_targets = _world_size_teacher_and_targets()
    for ids, ba_mask, labels, mask in rank_specs:
        rank_model = _ToyRelationalParameters()
        rank_model.load_state_dict(initial)
        rank_loss = _combined_loss(
            rank_model,
            ids,
            ba_mask=ba_mask,
            labels=labels,
            edge_mask=mask,
            world_size=2,
            align_denominator=torch.tensor(2.0),
            rel_denominator=torch.tensor(3.0),
        )
        rank_losses.append(rank_loss)
        rank_gradients.append(
            dict(
                zip(
                    (name for name, _ in rank_model.named_parameters()),
                    torch.autograd.grad(rank_loss, tuple(rank_model.parameters())),
                    strict=True,
                )
            )
        )
        for pair_id, real in zip(ids, mask, strict=True):
            if bool(real):
                reconstructed_targets[pair_id] = full_targets[pair_id]

    torch.testing.assert_close(reconstructed_targets, full_targets)
    torch.testing.assert_close(torch.stack(rank_losses).mean(), full_loss)
    for name, full_gradient in full_gradients.items():
        distributed_gradient = torch.stack([gradient[name] for gradient in rank_gradients]).mean(0)
        torch.testing.assert_close(
            distributed_gradient,
            full_gradient,
            rtol=5e-5,
            atol=5e-5,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
