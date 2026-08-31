"""KDRowBank kd_gen arm: join, guards, loss wiring, and diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import src.train_b0 as train_b0_module
import torch
import yaml
from accelerate import Accelerator
from src.distill.artifacts import KDRowTargets
from src.distill.config import DistillConfig
from src.model.egostitch.classifier.b0_v31 import BEST_V3_1_CONFIG, V3_1
from src.model.egostitch.classifier.topo_gen import build_topo_gen
from src.train_b0 import (
    Config,
    KDRowBank,
    ModelConfig,
    RuntimeConfig,
    _build_optimizer,
    _evaluate_distributed,
    _run_probe_mode,
    _set_topo_gen_training_stage,
    _term_grad_norms,
    build_model,
    train_loop,
)
from torch import nn

from tests.test_train_b0 import _tiny_config

LATENT_DIM = 8


@pytest.mark.parametrize(
    ("path", "family", "mc_samples", "sampler_steps"),
    [
        ("configs/b1_kd_gen_edm_breadth_first.yaml", "edm", 4, 4),
        ("configs/b1_kd_gen_det_breadth_first.yaml", "det_mse", 1, None),
    ],
)
def test_kd_gen_configs_parse(
    path: str, family: str, mc_samples: int, sampler_steps: int | None
) -> None:
    raw = yaml.safe_load(Path(path).read_text())
    distill = DistillConfig.from_mapping(raw["distill"])
    topo_config = raw["model"]["config"]["topo_gen"]
    module = build_topo_gen(topo_config, raw["model"]["config"]["d_model"])

    assert distill.arm == "kd_gen"
    assert module.family == family
    assert module.latent_dim == 512
    assert module.mc_samples == mc_samples
    assert topo_config.get("sampler_steps") == sampler_steps


def _model_config(topo: bool = True, latent_dim: int = LATENT_DIM) -> dict[str, object]:
    config = {**BEST_V3_1_CONFIG, "input_dim": 24, "d_model": 16, "n_heads": 2}
    if topo:
        config["topo_gen"] = {
            "name": "edm",
            "latent_dim": latent_dim,
            "cond_dim": 8,
            "blocks": 1,
            "adapter_dim": 4,
            "mc_samples": 2,
            "sampler_steps": 2,
        }
    return config


def _model(topo: bool = True, latent_dim: int = LATENT_DIM) -> V3_1:
    torch.manual_seed(0)
    config = _model_config(topo=topo, latent_dim=latent_dim)
    return V3_1(**config)


def _production_config(*, topo: bool, w_gen: bool) -> Config:
    distill = DistillConfig(targets_path="x", w_gen=1.0) if w_gen else None
    return replace(
        _tiny_config(),
        model=ModelConfig(family="v3_1", config=_model_config(topo=topo)),
        runtime=RuntimeConfig(
            world_size=1,
            pack_dir=Path("pack"),
            pack_workers=1,
            loader_workers_per_rank=1,
            prefetch_factor=2,
            token_budget=1024,
            max_pairs_per_rank=8,
            memory_limit_gib=1.0,
            probe_warmup_steps=1,
            probe_timed_steps=1,
        ),
        distill=distill,
    )


def _targets(
    rep_dim: int = LATENT_DIM, *, zeros: bool = False, two_val_classes: bool = False
) -> tuple[
    KDRowTargets,
    list[tuple[str, str]],
    list[int],
    list[tuple[str, str]],
    list[int],
]:
    node_ids = ["n0", "n1", "n2", "n3"]
    pairs = [("n0", "n1"), ("n2", "n3")]
    val_pairs = [("n0", "n2"), ("n1", "n3")] if two_val_classes else [("n0", "n2")]
    val_labels = [1, 0] if two_val_classes else [1]
    rng = np.random.default_rng(0)
    train_rep = rng.normal(size=(2, rep_dim)).astype(np.float16)
    val_rep = rng.normal(size=(len(val_pairs), rep_dim)).astype(np.float16)
    if zeros:
        train_rep.fill(0)
        val_rep.fill(0)
    targets = KDRowTargets(
        node_ids=node_ids,
        pair_a_idx=np.array([0, 2], dtype=np.int32),
        pair_b_idx=np.array([1, 3], dtype=np.int32),
        pair_label=np.array([1, 0], dtype=np.int8),
        teacher_logit=np.array([1.0, -1.0], dtype=np.float32),
        teacher_rep=train_rep,
        val_pair_a_idx=np.array([0, 1] if two_val_classes else [0], dtype=np.int32),
        val_pair_b_idx=np.array([2, 3] if two_val_classes else [2], dtype=np.int32),
        val_pair_label=np.array(val_labels, dtype=np.int8),
        val_teacher_logit=np.array([0.5, -0.5] if two_val_classes else [0.5], dtype=np.float32),
        val_teacher_rep=val_rep,
        manifest={},
    )
    return targets, pairs, [1, 0], val_pairs, val_labels


def _bank(
    model: nn.Module,
    *,
    targets: KDRowTargets | None = None,
    w_gen: float = 1.0,
) -> KDRowBank:
    default_targets, pairs, labels, val_pairs, val_labels = _targets()
    if targets is not None:
        node_ids = np.asarray(targets.node_ids, dtype=object)
        pairs = list(
            zip(
                node_ids[targets.pair_a_idx].tolist(),
                node_ids[targets.pair_b_idx].tolist(),
                strict=True,
            )
        )
        labels = targets.pair_label.astype(np.int64).tolist()
        val_pairs = list(
            zip(
                node_ids[targets.val_pair_a_idx].tolist(),
                node_ids[targets.val_pair_b_idx].tolist(),
                strict=True,
            )
        )
        val_labels = targets.val_pair_label.astype(np.int64).tolist()
    return KDRowBank(
        DistillConfig(targets_path="x", w_gen=w_gen),
        default_targets if targets is None else targets,
        train_pairs=pairs,
        train_labels=labels,
        val_pairs=val_pairs,
        val_labels=val_labels,
        model=model,
        device=torch.device("cpu"),
    )


def _train_batch() -> dict[str, torch.Tensor]:
    return {
        "_row_id": torch.tensor([0, 1]),
        "emb_a": torch.randn(2, 4, 24),
        "emb_b": torch.randn(2, 4, 24),
        "len_a": torch.full((2,), 4, dtype=torch.long),
        "len_b": torch.full((2,), 4, dtype=torch.long),
        "label": torch.tensor([1.0, 0.0]),
    }


def _kd_gen_config(*, joint_warmup_frac: float = 0.1, gen_lr_scale: float = 0.1) -> Config:
    cfg = _production_config(topo=True, w_gen=True)
    return replace(
        cfg,
        optim=replace(cfg.optim, lr=1.0e-3, weight_decay=0.0),
        distill=DistillConfig(
            targets_path="x",
            w_gen=1.0,
            joint_warmup_frac=joint_warmup_frac,
            gen_lr_scale=gen_lr_scale,
        ),
    )


def test_kd_gen_optimizer_groups_and_epoch_boundary() -> None:
    model = _model()
    cfg = _kd_gen_config(joint_warmup_frac=0.1, gen_lr_scale=0.1)
    optimizer = _build_optimizer(model, cfg)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"base", "topo_gen"}
    assert {id(param) for param in groups["topo_gen"]["params"]} == {
        id(param) for param in model.topo_gen_parameters()
    }
    assert model.topo_gen is not None
    assert id(model.topo_gen.gate) in {id(param) for param in groups["base"]["params"]}

    _set_topo_gen_training_stage(model, optimizer, cfg.distill, epoch=3, total_epochs=25)
    assert model.topo_gen.joint_stage is False
    assert groups["base"]["lr"] == pytest.approx(1.0e-3)
    assert groups["topo_gen"]["lr"] == pytest.approx(1.0e-3)

    _set_topo_gen_training_stage(model, optimizer, cfg.distill, epoch=4, total_epochs=25)
    assert model.topo_gen.joint_stage is True
    assert groups["base"]["lr"] == pytest.approx(1.0e-3)
    assert groups["topo_gen"]["lr"] == pytest.approx(1.0e-4)


def test_kd_gen_train_loop_resyncs_lr_ratio_after_lambda_scheduler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    cfg = _kd_gen_config(joint_warmup_frac=0.5, gen_lr_scale=0.1)
    cfg = replace(cfg, optim=replace(cfg.optim, epochs=2, warmup_steps=4))
    accelerator = Accelerator(cpu=True)
    batch = _train_batch()
    scheduler_ratios: list[tuple[int, float]] = []
    synced_ratios: list[tuple[int, float]] = []
    current_epoch = 0
    real_set_stage = train_b0_module._set_topo_gen_training_stage
    real_step_scheduler = train_b0_module._step_scheduler

    def track_stage(
        tracked_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        distill: DistillConfig | None,
        *,
        epoch: int,
        total_epochs: int,
    ) -> None:
        nonlocal current_epoch
        current_epoch = epoch
        real_set_stage(
            tracked_model,
            optimizer,
            distill,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        synced_ratios.append((epoch, float(groups["topo_gen"]["lr"]) / groups["base"]["lr"]))

    def track_scheduler_step(scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        real_step_scheduler(scheduler)
        groups = {group["name"]: group for group in scheduler.optimizer.param_groups}
        scheduler_ratios.append(
            (current_epoch, float(groups["topo_gen"]["lr"]) / groups["base"]["lr"])
        )

    monkeypatch.setattr(train_b0_module, "_set_topo_gen_training_stage", track_stage)
    monkeypatch.setattr(train_b0_module, "_step_scheduler", track_scheduler_step)

    train_loop(model, lambda epoch: [batch], [batch], cfg, accelerator)

    assert scheduler_ratios == pytest.approx([(1, 1.0), (2, 1.0)])
    assert synced_ratios == pytest.approx([(1, 1.0), (1, 1.0), (2, 0.1), (2, 0.1)])


def test_kd_gen_warmup_stops_task_gradient_then_joint_enables_it() -> None:
    model = _model().train()
    cfg = _kd_gen_config()
    optimizer = _build_optimizer(model, cfg)
    assert model.topo_gen is not None
    torch.nn.init.normal_(model.topo_gen.adapter_up.weight, std=0.1)
    generator_params = model.topo_gen_parameters()

    _set_topo_gen_training_stage(model, optimizer, cfg.distill, epoch=1, total_epochs=10)
    model(_train_batch())["loss"].backward()
    assert all(param.grad is None for param in generator_params)

    model.zero_grad(set_to_none=True)
    _set_topo_gen_training_stage(model, optimizer, cfg.distill, epoch=2, total_epochs=10)
    model(_train_batch())["loss"].backward()
    assert (
        sum(float(param.grad.abs().sum()) for param in generator_params if param.grad is not None)
        > 0
    )


@pytest.mark.parametrize("epoch", [1, 2])
def test_kd_gen_task_plus_generator_loss_reaches_all_parameters(epoch: int) -> None:
    model = _model().train()
    cfg = _kd_gen_config()
    optimizer = _build_optimizer(model, cfg)
    bank = _bank(model)
    _set_topo_gen_training_stage(model, optimizer, cfg.distill, epoch=epoch, total_epochs=10)
    batch = _train_batch()
    bank.attach(batch)
    output = model(batch)
    kd_loss, _ = bank.loss(batch, output)

    (output["loss"] + kd_loss).backward()

    missing = [name for name, param in model.named_parameters() if param.grad is None]
    assert missing == []


def test_kd_gen_requires_topo_gen_both_directions() -> None:
    with pytest.raises(RuntimeError, match="topo_gen"):
        _bank(_model(topo=False))

    targets, pairs, labels, val_pairs, val_labels = _targets()
    with pytest.raises(RuntimeError, match="w_gen"):
        KDRowBank(
            DistillConfig(targets_path="x", w_logit=1.0),
            targets,
            train_pairs=pairs,
            train_labels=labels,
            val_pairs=val_pairs,
            val_labels=val_labels,
            model=_model(topo=True),
            device=torch.device("cpu"),
        )


def test_production_model_build_rejects_topo_gen_without_w_gen() -> None:
    with pytest.raises(RuntimeError, match="w_gen"):
        build_model(_production_config(topo=True, w_gen=False))


def test_production_model_build_rejects_w_gen_without_topo_gen() -> None:
    with pytest.raises(RuntimeError, match="topo_gen"):
        build_model(_production_config(topo=False, w_gen=True))


def test_kd_gen_ddp_probe_rejects_before_accelerator_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _production_config(topo=True, w_gen=True)
    model = build_model(cfg)
    accelerator = Accelerator(cpu=True)
    prepare_called = False

    def fail_prepare(*args: object) -> None:
        nonlocal prepare_called
        prepare_called = True
        raise AssertionError("accelerator.prepare must not run for kd_gen probe mode")

    monkeypatch.setattr(accelerator, "prepare", fail_prepare)
    with pytest.raises(RuntimeError, match="probe.*kd_gen|kd_gen.*probe"):
        _run_probe_mode(
            model,
            lambda epoch: [],
            cfg,
            accelerator,
            token_budget_per_rank=1024,
            profile_output=Path("probe.json"),
        )
    assert prepare_called is False


def test_kd_gen_accepts_prepared_model_wrapper() -> None:
    model = _model()
    bank = _bank(nn.DataParallel(model))
    assert bank._topo_gen is model.topo_gen


def test_latent_dim_mismatch_raises() -> None:
    with pytest.raises(RuntimeError, match="latent_dim"):
        _bank(_model(latent_dim=4))


def test_rms_scale_is_all_elements_fp64_and_stamped_on_model() -> None:
    model = _model()
    targets, *_ = _targets()
    _bank(model, targets=targets)
    assert model.topo_gen is not None
    expected = float(np.sqrt(np.mean(np.square(targets.teacher_rep.astype(np.float64)))))
    state = model.state_dict()
    assert float(state["topo_gen.latent_rms_scale"].item()) == pytest.approx(expected)


def test_invalid_rms_scale_fails_closed() -> None:
    targets, *_ = _targets(zeros=True)
    with pytest.raises((RuntimeError, ValueError), match="latent_rms_scale|RMS"):
        _bank(_model(), targets=targets)


def test_attach_injects_normalized_latent_and_loss_produces_telemetry() -> None:
    model = _model().train()
    bank = _bank(model, w_gen=0.4)
    batch = _train_batch()
    bank.attach(batch)
    latent = batch["kd_teacher_latent"]
    assert latent.shape == (2, LATENT_DIM) and latent.dtype == torch.float32
    assert float(latent.square().mean().sqrt()) == pytest.approx(1.0, rel=1e-5)

    output = model(batch)
    total, stats = bank.loss(batch, output)
    assert total.requires_grad
    assert float(total.detach().item()) == pytest.approx(
        0.4 * float(output["gen_loss"].detach().item())
    )
    for key in (
        "sum_latent_cos",
        "sum_prob_std",
        "sum_dispersion",
        "sum_branch_ratio",
        "sum_gen_loss",
    ):
        assert key in stats
    for quartile in range(1, 5):
        assert f"sum_sigma_q{quartile}" in stats
        assert f"count_sigma_q{quartile}" in stats


def test_epoch_telemetry_aggregates_generator_and_sigma_sums() -> None:
    model = _model().train()
    bank = _bank(model)
    batch = _train_batch()
    bank.attach(batch)
    _, stats = bank.loss(batch, model(batch))

    telemetry = bank.epoch_telemetry(Accelerator(cpu=True), stats)
    assert {
        "kd_gen_loss",
        "kd_latent_cos",
        "mc_prob_std",
        "gen_sample_dispersion",
        "gen_branch_ratio",
        "gen_gate",
    } <= telemetry.keys()
    for quartile in range(1, 5):
        count = stats[f"count_sigma_q{quartile}"]
        key = f"kd_gen_sigma_q{quartile}"
        if count > 0:
            assert telemetry[key] == pytest.approx(stats[f"sum_sigma_q{quartile}"] / count)
        else:
            assert key not in telemetry


def test_first_step_gradient_probe_sees_generator_loss() -> None:
    model = _model().train()
    bank = _bank(model)
    batch = _train_batch()
    bank.attach(batch)
    output = model(batch)
    kd_loss, _ = bank.loss(batch, output)

    task_norm, kd_norm = _term_grad_norms(output["loss"], kd_loss, model)
    assert task_norm > 0.0
    assert kd_norm > 0.0


def test_validation_injects_normalized_latent_preserves_rng_and_reports_cosine() -> None:
    model = _model().train()
    targets, *_ = _targets(two_val_classes=True)
    bank = _bank(model, targets=targets)
    batch = {
        "_row_id": torch.tensor([0, 1]),
        "emb_a": torch.randn(2, 4, 24),
        "emb_b": torch.randn(2, 4, 24),
        "len_a": torch.full((2,), 4, dtype=torch.long),
        "len_b": torch.full((2,), 4, dtype=torch.long),
        "label": torch.tensor([1.0, 0.0]),
    }
    rng_before = torch.random.get_rng_state().clone()
    outcome = _evaluate_distributed(
        model,
        [batch],
        Accelerator(cpu=True),
        expected_row_ids=np.array([0, 1], dtype=np.int64),
        kd_val=bank.val_diagnostics(),
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert outcome.kd is not None
    assert "val_kd_latent_cos" in outcome.kd
    assert "val_kd_rep_cos" not in outcome.kd
    assert "val_kd_rep_loss" not in outcome.kd
