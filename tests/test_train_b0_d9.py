"""Focused Wave-5 trainer wiring contracts for the D9 pair latent arm."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import src.train_b0 as train_b0
import torch
from src.data.packed_features import PackedFeatureTable
from src.distill.artifacts import KD_TARGETS_FORMAT_V4, KDTargets
from src.distill.config import DistillConfig
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import Config, KDStream, SchedulerConfig
from torch import nn

pytestmark = pytest.mark.unit


class _FakeTable:
    def __init__(self) -> None:
        self.tokens = torch.zeros(1, 8)
        records = [
            SimpleNamespace(node_id="node_a", length=2),
            SimpleNamespace(node_id="node_b", length=2),
        ]
        self.manifest = SimpleNamespace(
            nodes=records,
            node_index=lambda: {record.node_id: row for row, record in enumerate(records)},
        )

    def gather_nodes(
        self, node_indices: torch.Tensor, boundary: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = node_indices.float().view(-1, 1, 1).expand(-1, boundary, 8)
        return values, torch.full((len(node_indices),), boundary, dtype=torch.long)


class _PairLatentStub(nn.Module):
    def __init__(self, seed_count: int = 2, seed_dim: int = 3) -> None:
        super().__init__()
        self.seed_count = seed_count
        self.seed_dim = seed_dim
        self.kl_free_bits = 0.05
        self.alpha = nn.Parameter(torch.tensor(0.25))


class _D9StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.pair_latent_gen = _PairLatentStub()
        self.seen_teacher: torch.Tensor | None = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        teacher = batch["kd_teacher_seeds"]
        self.seen_teacher = teacher.detach().clone()
        batch_size = teacher.shape[0]
        seeds_q = teacher + self.scale
        return {
            "logits": self.scale.expand(batch_size, 1),
            "gen_seeds_q": seeds_q,
            "gen_seeds_prior_mean": teacher + 0.5 * self.scale,
            "gen_kl": self.scale.expand(batch_size, 4).square(),
            "gen_delta_std": self.scale.expand(batch_size) * 0.2,
            "gen_prior_dispersion": self.scale.expand(batch_size) * 0.3,
        }


def _targets(teacher_seeds: np.ndarray | None) -> KDTargets:
    manifest: dict[str, object] = {
        "format": KD_TARGETS_FORMAT_V4,
        "k_near": 1,
        "k_rand": 1,
        "seed_count": 2,
        "seed_dim": 3,
    }
    return KDTargets(
        node_ids=["node_a", "node_b"],
        pair_anchor_idx=np.array([0, 1], dtype=np.int32),
        pair_partner_idx=np.array([1, 0], dtype=np.int32),
        anchor_offsets=np.array([0, 1, 2], dtype=np.int64),
        teacher_logit=np.zeros(2, dtype=np.float32),
        teacher_pooled_ab=np.ones((2, 3), dtype=np.float16),
        teacher_pooled_ba=np.ones((2, 3), dtype=np.float16),
        is_near=np.ones(2, dtype=np.uint8),
        pair_label=np.ones(2, dtype=np.int8),
        teacher_seeds=teacher_seeds,
        manifest=manifest,
    )


def _distill(**overrides: object) -> DistillConfig:
    values: dict[str, object] = {
        "targets_path": "targets",
        "w_seed": 1.0,
        "w_geom": 1.0,
        "w_kl": 1.0,
        "anchors_per_step": 1,
        "kl_warmup_steps": 10,
        "joint_start_epoch": 3,
        "gen_lr_scale": 0.1,
    }
    values.update(overrides)
    return DistillConfig.from_mapping(values)


def _stream(model: nn.Module, targets: KDTargets) -> KDStream:
    return KDStream(
        _distill(),
        targets,
        cast(PackedFeatureTable, _FakeTable()),
        allowed_nodes=frozenset({"node_a", "node_b"}),
        forbidden_internal_nodes=frozenset(),
        seed=0,
        rank=0,
        world_size=1,
        model=model,
    )


def test_d9_stream_requires_v4_teacher_seeds_matching_model_shape() -> None:
    model = _D9StubModel()
    with pytest.raises(RuntimeError, match="kd_targets_v4"):
        _stream(model, _targets(None))
    with pytest.raises(RuntimeError, match="teacher_seeds shape"):
        _stream(model, _targets(np.ones((2, 3, 3), dtype=np.float16)))


def test_d9_stream_gathers_zero_filler_applies_kl_warmup_and_records_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _D9StubModel()
    targets = _targets(np.ones((2, 2, 3), dtype=np.float16))
    stream = _stream(model, targets)

    monkeypatch.setattr(train_b0, "kd_seed_loss", lambda student, teacher, mask: model.scale * 1.0)
    monkeypatch.setattr(
        train_b0, "kd_seed_gram_loss", lambda student, teacher, mask: model.scale * 2.0
    )
    monkeypatch.setattr(
        train_b0,
        "kd_kl_loss",
        lambda kl, mask, *, free_bits: model.scale * 4.0,
    )

    loss = stream.loss(model, epoch=1, step=5)
    assert loss.detach().item() == pytest.approx(5.0)
    assert model.seen_teacher is not None
    assert torch.all(model.seen_teacher[0] == 1.0)
    assert torch.all(model.seen_teacher[1] == 0.0)
    assert stream.last_telemetry["kl_warmup_scale"] == pytest.approx(0.5)
    assert {
        "kd_prior_cos",
        "kd_seed_cos",
        "kd_recon_delta",
        "kd_geom",
        "kd_kl_per_dim",
        "kl_active_units",
        "gen_alpha",
        "mc_logit_std",
        "gen_prior_dispersion",
    } <= stream.last_telemetry.keys()


def _tiny_model() -> V3_1:
    return V3_1(
        input_dim=8,
        d_model=16,
        encoder_layers=1,
        cross_attn_layers=1,
        n_heads=4,
        mlp_head={"hidden_dims": [8], "dropout": 0.0},
        regularization={"dropout": 0.0},
        pair_latent_gen={
            "z_dim": 4,
            "cond_dim": 8,
            "hidden": 12,
            "seed_count": 2,
            "seed_dim": 3,
            "mc_samples": 2,
            "kl_free_bits": 0.05,
        },
    )


def _cfg() -> Config:
    optim = SimpleNamespace(
        lr=1.0e-3,
        weight_decay=0.0,
        epochs=4,
        scheduler=SchedulerConfig(
            type="onecycle",
            max_lr=1.0e-3,
            pct_start=0.25,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        ),
    )
    return cast(Config, SimpleNamespace(optim=optim, distill=_distill()))


def test_generator_optimizer_group_excludes_fusion_and_tracks_base_schedule() -> None:
    model = _tiny_model()
    cfg = _cfg()
    optimizer = train_b0._build_optimizer(model, cfg)
    scheduler = train_b0._build_scheduler(optimizer, cfg, warmup_steps=1, total_steps=8)
    assert len(optimizer.param_groups) == 2
    generator_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert model.pair_latent_gen is not None
    assert id(model.pair_latent_gen.alpha) not in generator_ids
    assert generator_ids.isdisjoint(id(p) for p in model.pair_latent_gen.delta_head.parameters())

    train_b0._set_pair_latent_training_stage(model, optimizer, cfg.distill, epoch=2)
    optimizer.step()
    train_b0._step_scheduler(scheduler)
    train_b0._sync_pair_latent_generator_lr(optimizer, cfg.distill, epoch=2)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    assert model.pair_latent_gen.joint_stage is False

    train_b0._set_pair_latent_training_stage(model, optimizer, cfg.distill, epoch=3)
    optimizer.step()
    train_b0._step_scheduler(scheduler)
    train_b0._sync_pair_latent_generator_lr(optimizer, cfg.distill, epoch=3)
    assert cfg.distill is not None
    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"] * cfg.distill.gen_lr_scale
    )
    assert model.pair_latent_gen.joint_stage is True

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    resumed_model = _tiny_model()
    resumed_optimizer = train_b0._build_optimizer(resumed_model, cfg)
    resumed_scheduler = train_b0._build_scheduler(
        resumed_optimizer, cfg, warmup_steps=1, total_steps=8
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    train_b0._set_pair_latent_training_stage(resumed_model, resumed_optimizer, cfg.distill, epoch=3)
    resumed_optimizer.step()
    train_b0._step_scheduler(resumed_scheduler)
    train_b0._sync_pair_latent_generator_lr(resumed_optimizer, cfg.distill, epoch=3)
    assert resumed_optimizer.param_groups[1]["lr"] == pytest.approx(
        resumed_optimizer.param_groups[0]["lr"] * cfg.distill.gen_lr_scale
    )
