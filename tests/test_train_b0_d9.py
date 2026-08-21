"""Focused trainer wiring contracts for the D9 pair-latent-generation KD arm.

The D9 KD term now rides the main student forward via `KDRowBank`/`attach()`
(no separate sampled anchor-context stream, no DDP-live-row scaling -- every
training row carries a teacher seed target, so there is no padding to weight
around). What remains worth testing here, on top of the generic `KDRowBank`
coverage in `tests/test_train_b0_kd.py`, is D9-specific: the staged joint
fine-tune (`joint_start_epoch` gating), the generator's separate LR schedule,
KL warmup scaling, D9 telemetry keys, that `attach()` injects
`kd_teacher_seeds` before the model forward, and that a teacher-seed shape
mismatch against `model.config.pair_latent_gen` fails closed.
"""

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
from src.data.val_region import Pair
from src.distill.artifacts import KDRowTargets, load_kd_targets, write_kd_targets
from src.distill.config import DistillConfig
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.train_b0 import (
    Config,
    KDRowBank,
    ModelConfig,
    SchedulerConfig,
    ValidationOutcome,
)
from torch import nn

from tests.test_train_b0 import _constant_metrics, _tiny_config

pytestmark = pytest.mark.unit

_NODE_IDS = ["node_a", "node_b"]
_TRAIN_PAIRS: list[Pair] = [("node_a", "node_b"), ("node_b", "node_a")]
_TRAIN_LABELS = [1, 0]


# --------------------------------------------------------------------------- artifact helper


def _write_seed_targets(
    out_dir: Path,
    *,
    teacher_seeds: np.ndarray,
    train_pairs: list[Pair] = _TRAIN_PAIRS,
    train_labels: list[int] = _TRAIN_LABELS,
    n_val: int = 1,
) -> None:
    """Write one minimal KD row-targets artifact carrying `teacher_seeds`."""
    index = {node: position for position, node in enumerate(_NODE_IDS)}
    a_idx = np.array([index[a] for a, _ in train_pairs], dtype=np.int32)
    b_idx = np.array([index[b] for _, b in train_pairs], dtype=np.int32)
    n = len(train_pairs)
    val_teacher_seeds = teacher_seeds[:n_val]
    write_kd_targets(
        out_dir,
        node_ids=_NODE_IDS,
        pair_a_idx=a_idx,
        pair_b_idx=b_idx,
        pair_label=np.asarray(train_labels, dtype=np.int8),
        teacher_logit=np.zeros(n, dtype=np.float32),
        teacher_rep=np.zeros((n, 4), dtype=np.float32),
        val_pair_a_idx=a_idx[:n_val],
        val_pair_b_idx=b_idx[:n_val],
        val_pair_label=np.asarray(train_labels[:n_val], dtype=np.int8),
        val_teacher_logit=np.zeros(n_val, dtype=np.float32),
        val_teacher_rep=np.zeros((n_val, 4), dtype=np.float32),
        truth_graph_sha256="0" * 64,
        checkpoint_path=out_dir / "ckpt.pt",
        checkpoint_sha256="1" * 64,
        checkpoint_id=None,
        teacher_seeds=teacher_seeds,
        val_teacher_seeds=val_teacher_seeds,
    )


def _load_seed_targets(tmp_path: Path, *, seed_dim: int = 3, n_val: int = 1) -> KDRowTargets:
    teacher_seeds = np.ones((2, 2, seed_dim), dtype=np.float16)
    out_dir = tmp_path / "targets"
    _write_seed_targets(out_dir, teacher_seeds=teacher_seeds, n_val=n_val)
    return load_kd_targets(out_dir, load_seeds=True)


def _distill(**overrides: object) -> DistillConfig:
    values: dict[str, object] = {
        "targets_path": "targets",
        "w_seed": 1.0,
        "w_geom": 1.0,
        "w_kl": 1.0,
        "kl_warmup_steps": 10,
        "joint_start_epoch": 3,
        "gen_lr_scale": 0.1,
    }
    values.update(overrides)
    return DistillConfig.from_mapping(values)


def _kd_bank(
    model: nn.Module,
    targets: KDRowTargets,
    *,
    distill: DistillConfig | None = None,
    n_val: int = 1,
) -> KDRowBank:
    return KDRowBank(
        distill if distill is not None else _distill(),
        targets,
        train_pairs=_TRAIN_PAIRS,
        train_labels=_TRAIN_LABELS,
        val_pairs=_TRAIN_PAIRS[:n_val],
        val_labels=_TRAIN_LABELS[:n_val],
        model=model,
        device=torch.device("cpu"),
    )


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


def _training_cfg(tmp_path: Path, *, epochs: int = 4) -> Config:
    base = _tiny_config(epochs=epochs, patience=8)
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


# --------------------------------------------------------------------------- join + arch checks


def test_seed_shape_mismatch_against_model_raises(tmp_path: Path) -> None:
    # model.pair_latent_gen expects (seed_count=2, seed_dim=3); the artifact
    # carries seed_dim=4 instead.
    targets = _load_seed_targets(tmp_path, seed_dim=4)
    model = _tiny_model()
    with pytest.raises(RuntimeError, match="teacher_seeds shape"):
        _kd_bank(model, targets)


def test_attach_injects_teacher_seeds_before_forward(tmp_path: Path) -> None:
    targets = _load_seed_targets(tmp_path, seed_dim=3)
    model = _tiny_model()
    bank = _kd_bank(model, targets)

    batch: dict[str, torch.Tensor] = {"_row_id": torch.tensor([1, 0])}
    assert "kd_teacher_seeds" not in batch
    bank.attach(batch)
    assert "kd_teacher_seeds" in batch
    expected = torch.as_tensor(np.ones((2, 2, 3), dtype=np.float16)[[1, 0]], dtype=torch.float32)
    torch.testing.assert_close(batch["kd_teacher_seeds"], expected)


def test_d9_validation_diagnostics_preserve_training_rng(tmp_path: Path) -> None:
    """Injected val seeds must never advance the training RNG stream.

    `forward_kd` draws `randn_like` even under eval; the fork_rng guard keeps
    the diagnostic-only validation pass from altering the trajectory.
    """
    targets = _load_seed_targets(tmp_path, seed_dim=3, n_val=2)
    model = _tiny_model()
    bank = _kd_bank(model, targets, n_val=2)
    kd_val = bank.val_diagnostics()
    assert kd_val.teacher_seeds is not None
    generator = torch.Generator().manual_seed(7)
    batch = {
        "emb_a": torch.randn(2, 3, 8, generator=generator),
        "emb_b": torch.randn(2, 3, 8, generator=generator),
        "len_a": torch.tensor([3, 3]),
        "len_b": torch.tensor([3, 3]),
        "label": torch.tensor([1.0, 0.0]),
        "_row_id": torch.tensor([0, 1]),
    }
    accelerator = Accelerator(cpu=True)
    state_before = torch.get_rng_state()
    outcome = train_b0._evaluate_distributed(
        model, [batch], accelerator, expected_row_ids=np.arange(2), kd_val=kd_val
    )
    assert outcome.kd is not None
    assert "val_kd_prior_cos" in outcome.kd
    assert torch.equal(torch.get_rng_state(), state_before)


# --------------------------------------------------------------------------- optimizer staging


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


# --------------------------------------------------------------------------- telemetry + KL warmup


def test_ddp_loop_kd_d9_telemetry_and_kl_warmup(tmp_path: Path) -> None:
    targets = _load_seed_targets(tmp_path, seed_dim=3)
    model = _tiny_model()
    distill = _distill(kl_warmup_steps=10, joint_start_epoch=0)
    bank = _kd_bank(model, targets, distill=distill)
    batch = _training_batch()
    cfg = _training_cfg(tmp_path, epochs=1)
    cfg = replace(cfg, distill=distill)

    result = train_b0.train_ddp_loop(
        model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=tmp_path / "attempt",
        schedule_total_steps=1,
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        kd_bank=bank,
    )

    entry = result.history[0]
    for key in (
        "kd_prior_cos",
        "kd_seed_cos",
        "kd_recon_delta",
        "kd_geom",
        "mc_logit_std",
        "gen_prior_dispersion",
        "kd_kl_per_dim",
        "kl_active_units",
        "gen_alpha",
        "kl_warmup_scale",
    ):
        assert key in entry, f"missing D9 telemetry key {key!r} in {sorted(entry)}"
    # One optimizer step out of a 10-step warmup: scale == 1 / 10.
    assert entry["kl_warmup_scale"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- staged resume


def test_d9_resume_matches_uninterrupted_across_joint_stage(tmp_path: Path) -> None:
    cfg = _training_cfg(tmp_path)
    batch = _training_batch()
    targets = _load_seed_targets(tmp_path, seed_dim=3)

    def evaluate(
        model: nn.Module,
        loader: Iterable[dict[str, torch.Tensor]],
        accelerator: Accelerator,
    ) -> ValidationOutcome:
        return ValidationOutcome(_constant_metrics(), None)

    torch.manual_seed(123)
    uninterrupted_model = _tiny_model()
    uninterrupted_bank = _kd_bank(uninterrupted_model, targets)
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
        kd_bank=uninterrupted_bank,
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
    interrupted_bank = _kd_bank(interrupted_model, targets)
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
            kd_bank=interrupted_bank,
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
    resumed_bank = _kd_bank(resumed_model, targets)
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
        kd_bank=resumed_bank,
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
