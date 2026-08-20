"""Focused Wave-5 trainer wiring contracts for the D9 pair latent arm."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import src.train_b0 as train_b0
import torch
from accelerate import Accelerator
from src.data.packed_features import PackedFeatureTable
from src.distill.artifacts import KD_TARGETS_FORMAT_V4, KDTargets
from src.distill.config import DistillConfig
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import (
    Config,
    KDStream,
    ModelConfig,
    SchedulerConfig,
    ValidationOutcome,
)
from torch import nn

from tests.test_train_b0 import _constant_metrics, _tiny_config

pytestmark = pytest.mark.unit


class _FakeTable:
    def __init__(self) -> None:
        self.tokens = torch.zeros(1, 8)
        records = [
            SimpleNamespace(node_id="node_a", length=3),
            SimpleNamespace(node_id="node_b", length=3),
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


class _GlobalLiveRows:
    """Two-rank collective simulation returning a pinned global live count."""

    def __init__(self, global_live_rows: float) -> None:
        self.global_live_rows = global_live_rows
        self.seen_local_rows: list[float] = []

    def reduce(self, value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        self.seen_local_rows.append(float(value.item()))
        return value.new_tensor(self.global_live_rows)


class _TelemetryCollectives:
    device = torch.device("cpu")

    def reduce(self, value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        if value.numel() == 1:
            return value.new_tensor(4.0)
        if value.numel() == len(train_b0.D9_ROW_TELEMETRY_KEYS):
            return torch.arange(1, value.numel() + 1, dtype=value.dtype) * 4.0
        assert value.numel() == 2
        return value.new_tensor([0.04, 0.12])


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
    assert stream.last_live_rows == 1
    assert stream.last_kl_dim_sum is not None
    torch.testing.assert_close(stream.last_kl_dim_sum, torch.ones(4))
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


def test_two_rank_live_row_scaling_recovers_the_global_masked_mean() -> None:
    # Rank 0 has one live row with mean loss 2; rank 1 has three with mean 6.
    # The true global masked mean is (1*2 + 3*6) / 4 = 5.
    rank0 = _GlobalLiveRows(global_live_rows=4.0)
    rank1 = _GlobalLiveRows(global_live_rows=4.0)
    parameter = torch.tensor(1.0, requires_grad=True)
    scaled0 = train_b0._scale_d9_kd_loss_for_ddp(
        2.0 * parameter,
        local_live_rows=1,
        world_size=2,
        accelerator=cast(Accelerator, rank0),
    )
    scaled1 = train_b0._scale_d9_kd_loss_for_ddp(
        6.0 * parameter,
        local_live_rows=3,
        world_size=2,
        accelerator=cast(Accelerator, rank1),
    )
    ddp_objective = (scaled0 + scaled1) / 2.0
    assert ddp_objective.item() == pytest.approx(5.0)
    ddp_objective.backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(5.0)
    assert rank0.seen_local_rows == [1.0]
    assert rank1.seen_local_rows == [3.0]


def test_d9_live_row_scaling_fails_closed_when_every_rank_is_padding() -> None:
    accelerator = _GlobalLiveRows(global_live_rows=0.0)
    with pytest.raises(RuntimeError, match="zero live teacher rows"):
        train_b0._scale_d9_kd_loss_for_ddp(
            torch.tensor(1.0),
            local_live_rows=0,
            world_size=2,
            accelerator=cast(Accelerator, accelerator),
        )


def test_d9_rank_with_only_padding_contributes_zero_when_other_rank_is_live() -> None:
    accelerator = _GlobalLiveRows(global_live_rows=3.0)
    scaled = train_b0._scale_d9_kd_loss_for_ddp(
        torch.tensor(7.0, requires_grad=True),
        local_live_rows=0,
        world_size=2,
        accelerator=cast(Accelerator, accelerator),
    )
    assert scaled.item() == 0.0


def test_epoch_telemetry_reduces_row_sums_and_kl_dimensions_globally() -> None:
    telemetry = train_b0._reduce_d9_epoch_telemetry(
        cast(Accelerator, _TelemetryCollectives()),
        row_weighted_sums=dict.fromkeys(train_b0.D9_ROW_TELEMETRY_KEYS, 0.0),
        local_live_rows=1,
        kl_dim_sum=torch.zeros(2),
        gen_alpha=0.3,
        kl_warmup_scale=0.7,
    )
    for index, key in enumerate(train_b0.D9_ROW_TELEMETRY_KEYS, start=1):
        assert telemetry[key] == pytest.approx(float(index))
    assert telemetry["kd_kl_per_dim"] == pytest.approx(0.02)
    assert telemetry["kl_active_units"] == 1.0
    assert telemetry["gen_alpha"] == pytest.approx(0.3)
    assert telemetry["kl_warmup_scale"] == pytest.approx(0.7)


def _tiny_model_kwargs() -> dict[str, object]:
    return {
        "input_dim": 8,
        "d_model": 16,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 4,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0},
        "regularization": {"dropout": 0.0},
        "pair_latent_gen": {
            "z_dim": 4,
            "cond_dim": 8,
            "hidden": 12,
            "seed_count": 2,
            "seed_dim": 3,
            "mc_samples": 2,
            "kl_free_bits": 0.05,
        },
    }


def _tiny_model() -> V3_1:
    return V3_1(**_tiny_model_kwargs())


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


def _training_cfg(tmp_path: Path) -> Config:
    base = _tiny_config(epochs=4, patience=8)
    return replace(
        base,
        model=ModelConfig(family="v3_1", config=_tiny_model_kwargs()),
        optim=replace(
            base.optim,
            lr=1.0e-3,
            scheduler=SchedulerConfig(
                type="onecycle",
                max_lr=1.0e-3,
                pct_start=0.5,
                div_factor=10.0,
                final_div_factor=100.0,
                anneal_strategy="cos",
            ),
        ),
        seed=17,
        output_dir=tmp_path / "unused-output",
        distill=_distill(),
    )


def _training_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(44)
    return {
        "emb_a": torch.randn(2, 3, 8, generator=generator),
        "emb_b": torch.randn(2, 3, 8, generator=generator),
        "len_a": torch.tensor([3, 3]),
        "len_b": torch.tensor([3, 3]),
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
        "_local_pair_count": torch.tensor(2),
        "_global_pair_count": torch.tensor(2),
    }


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


def test_d9_resume_matches_uninterrupted_across_joint_stage(tmp_path: Path) -> None:
    cfg = _training_cfg(tmp_path)
    batch = _training_batch()
    targets = _targets(np.ones((2, 2, 3), dtype=np.float16))

    def evaluate(
        model: nn.Module,
        loader: Iterable[dict[str, torch.Tensor]],
        accelerator: Accelerator,
    ) -> ValidationOutcome:
        return ValidationOutcome(_constant_metrics(), None)

    torch.manual_seed(123)
    uninterrupted_model = _tiny_model()
    uninterrupted_stream = _stream(uninterrupted_model, targets)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = train_b0.train_ddp_loop(
        uninterrupted_model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=uninterrupted_dir,
        schedule_total_steps=4,
        evaluate_fn=evaluate,
        kd_loss_fn=uninterrupted_stream.loss,
    )

    evaluations = 0

    def interrupt_after_two(
        model: nn.Module,
        loader: Iterable[dict[str, torch.Tensor]],
        accelerator: Accelerator,
    ) -> ValidationOutcome:
        nonlocal evaluations
        evaluations += 1
        if evaluations == 3:
            raise RuntimeError("interrupted after joint-stage step")
        return ValidationOutcome(_constant_metrics(), None)

    torch.manual_seed(123)
    interrupted_model = _tiny_model()
    interrupted_stream = _stream(interrupted_model, targets)
    prior_dir = tmp_path / "prior"
    with pytest.raises(RuntimeError, match="interrupted after joint-stage step"):
        train_b0.train_ddp_loop(
            interrupted_model,
            lambda epoch: [batch],
            [batch],
            cfg,
            Accelerator(cpu=True),
            warmup_steps=1,
            artifact_dir=prior_dir,
            schedule_total_steps=4,
            evaluate_fn=interrupt_after_two,
            kd_loss_fn=interrupted_stream.loss,
        )

    prior_state = torch.load(
        prior_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    assert prior_state["epoch"] == 2
    prior_groups = {group["name"]: group for group in prior_state["optimizer"]["param_groups"]}
    assert prior_groups["pair_latent_gen"]["lr"] == pytest.approx(prior_groups["base"]["lr"])

    resumed_dir = tmp_path / "resumed"
    (resumed_dir / "checkpoints").mkdir(parents=True)
    shutil.copy2(prior_dir / "metrics.jsonl", resumed_dir / "metrics.jsonl")
    for candidate in (prior_dir / "checkpoints").glob("epoch-*.pt"):
        shutil.copy2(candidate, resumed_dir / "checkpoints" / candidate.name)

    resumed_model = _tiny_model()
    resumed_stream = _stream(resumed_model, targets)
    resumed = train_b0.train_ddp_loop(
        resumed_model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=resumed_dir,
        resume_attempt=prior_dir,
        schedule_total_steps=4,
        evaluate_fn=evaluate,
        kd_loss_fn=resumed_stream.loss,
    )

    assert [row["epoch"] for row in resumed.history] == [1, 2, 3, 4]
    assert uninterrupted_model.pair_latent_gen is not None
    assert uninterrupted_model.pair_latent_gen.joint_stage is True
    assert resumed_model.pair_latent_gen is not None
    assert resumed_model.pair_latent_gen.joint_stage is True
    assert uninterrupted.last_state_dict
    assert resumed.last_state_dict.keys() == uninterrupted.last_state_dict.keys()
    for name, expected in uninterrupted.last_state_dict.items():
        torch.testing.assert_close(resumed.last_state_dict[name], expected, rtol=0.0, atol=0.0)

    uninterrupted_state = torch.load(
        uninterrupted_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    resumed_state = torch.load(
        resumed_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    for actual_group, expected_group in zip(
        resumed_state["optimizer"]["param_groups"],
        uninterrupted_state["optimizer"]["param_groups"],
        strict=True,
    ):
        assert actual_group["name"] == expected_group["name"]
        assert actual_group["lr"] == pytest.approx(expected_group["lr"], rel=0.0, abs=0.0)
    resumed_groups = {group["name"]: group for group in resumed_state["optimizer"]["param_groups"]}
    assert resumed_groups["pair_latent_gen"]["lr"] == pytest.approx(
        resumed_groups["base"]["lr"] * 0.1
    )
