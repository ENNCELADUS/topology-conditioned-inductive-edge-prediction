"""`train_b0.KDStream`: deterministic anchors, rank disjointness, finite loss."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import src.train_b0 as train_b0
import torch
from src.data.packed_features import PackedFeatureTable
from src.distill.artifacts import KDTargets
from src.distill.config import DistillConfig
from src.distill.losses import kd_align_loss, kd_residual_loss
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
        return {
            "logits": logits.unsqueeze(-1),
            "node_factor_pair": pair,
            "node_factor_align": pair,
            "node_factor_residual": logits,
        }


def _targets(
    *, content_logit: np.ndarray | None = None, teacher_logit: np.ndarray | None = None
) -> KDTargets:
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
        teacher_logit=(
            teacher_logit
            if teacher_logit is not None
            else np.linspace(-2.0, 2.0, 8, dtype=np.float32)
        ),
        teacher_pooled_ab=np.ones((8, 2), dtype=np.float16),
        teacher_pooled_ba=np.ones((8, 2), dtype=np.float16),
        is_near=np.ones(8, dtype=np.uint8),
        pair_label=np.array([1, 0] * 4, dtype=np.int8),
        content_logit=content_logit,
        manifest={"k_near": 1, "k_rand": 1},
    )


def _stream(
    distill: DistillConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    allowed_nodes: frozenset[str] | None = None,
    forbidden_internal_nodes: frozenset[str] = frozenset(),
    content_logit: np.ndarray | None = None,
    teacher_logit: np.ndarray | None = None,
) -> KDStream:
    return KDStream(
        distill,
        _targets(content_logit=content_logit, teacher_logit=teacher_logit),
        cast(PackedFeatureTable, _FakeTable()),
        allowed_nodes=allowed_nodes if allowed_nodes is not None else frozenset(_NODE_IDS),
        forbidden_internal_nodes=forbidden_internal_nodes,
        seed=0,
        rank=rank,
        world_size=world_size,
    )


def test_targets_outside_training_universe_fail_closed() -> None:
    distill = DistillConfig(targets_path="t", w_label=1.0)
    with pytest.raises(ValueError, match="outside the training universe"):
        _stream(distill, allowed_nodes=frozenset({"node_a", "node_b"}))


def test_target_pair_inside_v_val_fails_closed() -> None:
    # _targets()'s anchor 0 (node_a) has partner rows to node_b and node_c
    # (partner = (anchor + 1/2) % 4); a V_val region covering node_a and
    # node_b must be refused at the pair level even though every individual
    # node is inside `allowed_nodes`.
    distill = DistillConfig(targets_path="t", w_label=1.0)
    with pytest.raises(ValueError, match="KD target pair inside the V_val validation region"):
        _stream(distill, forbidden_internal_nodes=frozenset({"node_a", "node_b"}))


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
        {"w_align": 1.0},
        {"w_rowmass": 1.0},
    ],
)
def test_loss_is_finite_and_differentiable_for_every_arm(weights: dict[str, float]) -> None:
    distill = DistillConfig.from_mapping({"targets_path": "t", "anchors_per_step": 2, **weights})
    stream = _stream(distill)
    model = _StubModel()
    loss = stream.loss(model, epoch=1, step=1)
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
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


def test_align_arm_requires_node_factor_align() -> None:
    distill = DistillConfig(targets_path="t", w_align=1.0, anchors_per_step=1)
    stream = _stream(distill)

    class _NoAlign(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"logits": batch["emb_a"].mean(dim=(1, 2))}

    with pytest.raises(RuntimeError, match="node_factor_align_dim"):
        stream.loss(_NoAlign(), epoch=1, step=1)


def test_align_arm_calls_kd_align_loss_with_hoisted_pooled_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # teacher_pooled_ab/ba are all-ones in the fixture, so the symmetrized
    # pooled tensor is exactly 1.0 on every live (mask=1) row and 0.0 on
    # every padding row -- a value independent of the (random) student
    # embeddings, so it can be checked without reproducing row assembly.
    distill = DistillConfig(targets_path="t", w_align=1.0, anchors_per_step=2)
    stream = _stream(distill)
    model = _StubModel()

    captured: dict[str, torch.Tensor] = {}

    def _spy(
        student_align: torch.Tensor,
        teacher_feat: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: float,
    ) -> torch.Tensor:
        captured["teacher_feat"] = teacher_feat
        captured["mask"] = mask
        return kd_align_loss(student_align, teacher_feat, mask, **kwargs)

    monkeypatch.setattr(train_b0, "kd_align_loss", _spy)
    loss = stream.loss(model, epoch=1, step=1)
    assert torch.isfinite(loss)

    pooled = captured["teacher_feat"]
    mask = captured["mask"]
    expected = mask.unsqueeze(-1).expand(-1, pooled.shape[-1])
    assert torch.allclose(pooled, expected)


def test_residual_arm_requires_node_factor_residual() -> None:
    distill = DistillConfig(targets_path="t", w_residual=1.0, anchors_per_step=1)
    stream = _stream(distill, content_logit=np.zeros(8, dtype=np.float32))

    class _NoResidual(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"logits": batch["emb_a"].mean(dim=(1, 2))}

    with pytest.raises(RuntimeError, match="node_factor_dim"):
        stream.loss(_NoResidual(), epoch=1, step=1)


def test_residual_arm_requires_content_logit_in_targets() -> None:
    distill = DistillConfig(targets_path="t", w_residual=1.0, anchors_per_step=1)
    with pytest.raises(RuntimeError, match="content_logit"):
        _stream(distill)


def test_residual_arm_computes_teacher_minus_content_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # content_logit is teacher_logit offset by a constant, so the delta
    # target is exactly 0.5 on every live row regardless of which context
    # rows were sampled, and exactly 0.0 on every padding row (both the
    # padded teacher_logit and the padded content_logit are appended as
    # 0.0) -- both checkable without reproducing row assembly.
    teacher_logit = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    content_logit = (teacher_logit - 0.5).astype(np.float32)
    distill = DistillConfig(targets_path="t", w_residual=1.0, anchors_per_step=2)
    stream = _stream(distill, content_logit=content_logit)
    model = _StubModel()

    captured: dict[str, torch.Tensor] = {}

    def _spy(
        student_residual: torch.Tensor,
        delta_target: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: float,
    ) -> torch.Tensor:
        captured["delta_target"] = delta_target
        captured["mask"] = mask
        return kd_residual_loss(student_residual, delta_target, mask, **kwargs)

    monkeypatch.setattr(train_b0, "kd_residual_loss", _spy)
    loss = stream.loss(model, epoch=1, step=1)
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    assert model.scale.grad is not None

    delta = captured["delta_target"]
    mask = captured["mask"]
    live_delta = delta[mask == 1.0]
    pad_delta = delta[mask == 0.0]
    assert torch.allclose(live_delta, torch.full_like(live_delta, 0.5))
    assert torch.allclose(pad_delta, torch.zeros_like(pad_delta))


def test_rank_dist_gram_pattern_sums_the_three_terms() -> None:
    # w_rank + w_dist + w_gram compose additively already (each is an
    # independent `if` branch adding into `total`), so this exercises the
    # combined pattern rather than any new trainer wiring.
    distill_combo = DistillConfig(
        targets_path="t", w_rank=1.0, w_dist=1.0, w_gram=1.0, anchors_per_step=2
    )
    model = _StubModel()

    torch.manual_seed(0)
    combined = _stream(distill_combo).loss(model, epoch=1, step=1)

    total_individual = torch.zeros(())
    for weights in ({"w_rank": 1.0, "w_dist": 1.0}, {"w_gram": 1.0}):
        distill_legal = DistillConfig.from_mapping(
            {"targets_path": "t", "anchors_per_step": 2, **weights}
        )
        torch.manual_seed(0)
        total_individual = total_individual + _stream(distill_legal).loss(model, epoch=1, step=1)

    assert torch.allclose(combined, total_individual, atol=1e-5)


def test_rowmass_arm_responds_to_teacher_mass_changes() -> None:
    # Same student model, same anchors, same (seeded) student embeddings --
    # only the teacher's per-row logits change (raising every row's
    # sigmoid(logit) mass). kd_d8 (absolute row-mass distillation) must move
    # in response; a stream wired to ignore the teacher would not.
    distill = DistillConfig(targets_path="t", w_rowmass=1.0, anchors_per_step=2)
    model = _StubModel()

    low_teacher = np.full(8, -5.0, dtype=np.float32)
    torch.manual_seed(0)
    loss_low = _stream(distill, teacher_logit=low_teacher).loss(model, epoch=1, step=1)

    high_teacher = np.full(8, 5.0, dtype=np.float32)
    torch.manual_seed(0)
    loss_high = _stream(distill, teacher_logit=high_teacher).loss(model, epoch=1, step=1)

    assert torch.isfinite(loss_low)
    assert torch.isfinite(loss_high)
    assert not torch.allclose(loss_low, loss_high)
