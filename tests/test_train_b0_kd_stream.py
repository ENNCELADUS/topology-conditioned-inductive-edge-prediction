"""`train_b0.KDStream`: deterministic anchors, rank disjointness, finite loss."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from src.data.packed_features import PackedFeatureTable
from src.distill.artifacts import KDTargets
from src.distill.config import DistillConfig
from src.train_b0 import KDStream
from torch import nn

pytestmark = pytest.mark.unit

_NODE_IDS = ["node_a", "node_b", "node_c", "node_d"]


class _FakeTable:
    """Duck-typed stand-in for the packed table (tokens + manifest only)."""

    def __init__(self, dim: int = 8) -> None:
        self.tokens = torch.zeros(1, dim)
        self._dim = dim
        records = [
            SimpleNamespace(node_id=node_id, length=3 + index)
            for index, node_id in enumerate(_NODE_IDS)
        ]
        self.manifest = SimpleNamespace(
            nodes=records,
            node_index=lambda: {record.node_id: row for row, record in enumerate(records)},
        )

    def gather_nodes(
        self, node_indices: torch.Tensor, boundary: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.randn(node_indices.numel(), boundary, self._dim)
        lengths = torch.full((node_indices.numel(),), min(boundary, 3), dtype=torch.long)
        return emb, lengths


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.scale * batch["emb_a"].mean(dim=(1, 2))
        pair = torch.stack([logits, logits], dim=-1)
        return {"logits": logits.unsqueeze(-1), "node_factor_pair": pair}


def _targets() -> KDTargets:
    # Two context rows per anchor over 4 anchors: partner = (anchor + 1/2) % 4.
    pair_anchor = np.repeat(np.arange(4, dtype=np.int32), 2)
    pair_partner = np.array(
        [(anchor + shift) % 4 for anchor in range(4) for shift in (1, 2)], dtype=np.int32
    )
    return KDTargets(
        node_ids=list(_NODE_IDS),
        pair_anchor_idx=pair_anchor,
        pair_partner_idx=pair_partner,
        anchor_offsets=np.array([0, 2, 4, 6, 8], dtype=np.int64),
        teacher_logit=np.linspace(-2.0, 2.0, 8, dtype=np.float32),
        teacher_pooled_ab=np.ones((8, 2), dtype=np.float16),
        teacher_pooled_ba=np.ones((8, 2), dtype=np.float16),
        is_near=np.ones(8, dtype=np.uint8),
        pair_label=np.array([1, 0] * 4, dtype=np.int8),
        manifest={"k_near": 1, "k_rand": 1},
    )


def _stream(
    distill: DistillConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    allowed_nodes: frozenset[str] | None = None,
) -> KDStream:
    return KDStream(
        distill,
        _targets(),
        cast(PackedFeatureTable, _FakeTable()),
        allowed_nodes=allowed_nodes if allowed_nodes is not None else frozenset(_NODE_IDS),
        seed=0,
        rank=rank,
        world_size=world_size,
    )


def test_targets_outside_training_universe_fail_closed() -> None:
    distill = DistillConfig(targets_path="t", w_label=1.0)
    with pytest.raises(ValueError, match="outside the V_fit training universe"):
        _stream(distill, allowed_nodes=frozenset({"node_a", "node_b"}))


def test_anchor_positions_are_deterministic_and_rank_disjoint() -> None:
    distill = DistillConfig(targets_path="t", w_label=1.0, anchors_per_step=2)
    rank0 = _stream(distill, rank=0, world_size=2)
    rank1 = _stream(distill, rank=1, world_size=2)
    a0 = rank0._anchor_positions(epoch=1, step=5)
    assert a0 == rank0._anchor_positions(epoch=1, step=5)
    assert set(a0) <= {0, 2}
    assert set(rank1._anchor_positions(epoch=1, step=5)) <= {1, 3}
    assert (
        rank0._anchor_positions(epoch=1, step=6) != a0
        or rank0._anchor_positions(epoch=2, step=5) != a0
    )


@pytest.mark.parametrize(
    "weights",
    [
        {"w_label": 1.0},
        {"w_logit": 1.0},
        {"w_rank": 1.0, "w_dist": 1.0},
        {"w_gram": 1.0},
    ],
)
def test_loss_is_finite_and_differentiable_for_every_arm(weights: dict[str, float]) -> None:
    distill = DistillConfig(targets_path="t", anchors_per_step=2, **weights)
    stream = _stream(distill)
    model = _StubModel()
    loss = stream.loss(model, epoch=1, step=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scale.grad is not None


def test_gram_arm_requires_node_factor_pair() -> None:
    distill = DistillConfig(targets_path="t", w_gram=1.0, anchors_per_step=1)
    stream = _stream(distill)

    class _NoFactor(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"logits": batch["emb_a"].mean(dim=(1, 2))}

    with pytest.raises(RuntimeError, match="node_factor_dim"):
        stream.loss(_NoFactor(), epoch=1, step=1)


def test_world_size_larger_than_universe_fails_closed() -> None:
    distill = DistillConfig(targets_path="t", w_label=1.0)
    with pytest.raises(RuntimeError, match="smaller than the DDP world size"):
        _stream(distill, rank=4, world_size=5)
