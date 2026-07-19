"""Unit contract for the prospective EgoStitch §13.19 v2 trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from src import train_egostitch as te
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E

pytestmark = pytest.mark.unit


def _v2_config(tmp_path: Path) -> Path:
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text(json.dumps({"status": "DRAFT"}), encoding="utf-8")
    mapping = {
        "model": {"family": "egostitch_e2e", "config": {}},
        "data": {
            "root": str(tmp_path / "data"),
            "strategy": "breadth_first",
            "train_positives": "e_sup",
            "negative_ratio": 5,
            "partition_seed": 0,
            "msg_fraction": 0.8,
            "node_batch": 2,
            "edge_batch": 6,
            "f0_cache": str(tmp_path / "f0.pt"),
            "grounding_cache": str(tmp_path / "grounding.npz"),
            "expected_missing_features": [],
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "epochs": 30,
            "warmup_steps": 500,
            "grad_clip": 1.0,
            "warmstart_fraction": 0.2,
        },
        "diagnostics": {
            "gradient_probe_interval": 50,
            "gradient_imbalance_ratio": 50.0,
            "gradient_imbalance_steps": 200,
            "probe_s1_abs_mean_max": 1000.0,
            "selection_auprc_tolerance": 0.02,
            "topk_fraction": 0.01,
        },
        "eval": {"patience": 30, "eval_every": 1},
        "seed": 0,
        "output_dir": str(tmp_path / "out"),
        "mixed_precision": "bf16",
        "preregistration": str(preregistration),
        "stability_v2": {},
    }
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def test_v2_config_schema_is_strict_and_preserves_run_kind(tmp_path: Path) -> None:
    cfg = te.load_config(_v2_config(tmp_path))
    assert cfg.stability_v2 == te.EgoStitchV2Config()

    raw = yaml.safe_load(_v2_config(tmp_path).read_text(encoding="utf-8"))
    raw["stability_v2"]["positive_weight"] = 4.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        te.load_config(bad)


def test_v2_run_kinds_fail_closed(tmp_path: Path) -> None:
    loaded = te.load_config(_v2_config(tmp_path))
    overfit = te.apply_overrides(
        loaded,
        te.EgoCliArgs(
            config=tmp_path / "v2.yaml", seed=None, output_dir=None, v2_run_kind="overfit"
        ),
    )
    with pytest.raises(RuntimeError, match="fixed 510-row"):
        te.prepare_ddp_run_config(overfit, max_steps=2000)

    rehearsal = te.apply_overrides(
        loaded,
        te.EgoCliArgs(
            config=tmp_path / "v2.yaml", seed=None, output_dir=None, v2_run_kind="rehearsal"
        ),
    )
    with pytest.raises(ValueError, match="complete schedule"):
        te.prepare_ddp_run_config(rehearsal, max_steps=1)

    formal = loaded
    with pytest.raises(te.PreregistrationNotBinding):
        te.prepare_ddp_run_config(formal, max_steps=None)


def test_v2_three_phase_boundaries_and_first_eligibility() -> None:
    assert te.v2_phase_boundaries(2000) == (400, 600)
    assert te.v2_phase_state(399, 2000) == te.V2PhaseState("A", 0.0, True, 0.0)
    assert te.v2_phase_state(400, 2000).alpha == pytest.approx(1 / 200)
    assert te.v2_phase_state(599, 2000) == te.V2PhaseState("B", 1.0, False, 1.0)
    assert te.v2_phase_state(600, 2000) == te.V2PhaseState("C", 1.0, False, 1.0)
    assert te.v2_first_eligible_epoch(3000, 100) == 10


def test_v2_weighted_bce_matches_one_and_two_rank_gradients_with_padding() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    logits = torch.linspace(-0.3, 0.3, len(labels), requires_grad=True)
    mask = torch.ones_like(labels)
    full_loss = te.v2_weighted_bce_with_logits(logits, labels, mask)
    (full_grad,) = torch.autograd.grad(full_loss, logits)

    global_denominator = torch.tensor(15.0)
    rank_grads = []
    for indices, rank_mask in (
        ([0, 2, 4, 6], [1.0, 1.0, 1.0, 1.0]),
        ([1, 3, 5, 5], [1.0, 1.0, 1.0, 0.0]),
    ):
        rank_logits = logits.detach()[indices].clone().requires_grad_()
        rank_labels = labels[indices]
        rank_loss = te.v2_weighted_bce_with_logits(
            rank_logits,
            rank_labels,
            torch.tensor(rank_mask),
            world_size=2,
            all_reduce_sum=lambda _: global_denominator,
        )
        (rank_grad,) = torch.autograd.grad(rank_loss, rank_logits)
        rank_grads.append((indices, rank_mask, rank_grad))
    reconstructed = torch.zeros_like(full_grad)
    for indices, rank_mask, rank_grad in rank_grads:
        for index, real, value in zip(indices, rank_mask, rank_grad, strict=True):
            if real:
                reconstructed[index] += value / 2
    torch.testing.assert_close(reconstructed, full_grad)

    zero_logits = torch.zeros(6, requires_grad=True)
    balanced = te.v2_weighted_bce_with_logits(
        zero_logits, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), torch.ones(6)
    )
    (zero_grad,) = torch.autograd.grad(balanced, zero_logits)
    assert float(zero_grad.sum()) == pytest.approx(0.0, abs=1e-8)

    features = torch.linspace(-1.0, 1.0, len(labels))
    full_weight = torch.tensor(0.2, requires_grad=True)
    full_parameter_loss = te.v2_weighted_bce_with_logits(full_weight * features, labels, mask)
    (full_parameter_grad,) = torch.autograd.grad(full_parameter_loss, full_weight)
    rank_parameter_grads = []
    for indices, rank_mask in (
        ([0, 2, 4, 6], [1.0, 1.0, 1.0, 1.0]),
        ([1, 3, 5, 5], [1.0, 1.0, 1.0, 0.0]),
    ):
        rank_weight = full_weight.detach().clone().requires_grad_()
        rank_loss = te.v2_weighted_bce_with_logits(
            rank_weight * features[indices],
            labels[indices],
            torch.tensor(rank_mask),
            world_size=2,
            all_reduce_sum=lambda _: global_denominator,
        )
        rank_parameter_grads.append(torch.autograd.grad(rank_loss, rank_weight)[0])
    torch.testing.assert_close(torch.stack(rank_parameter_grads).mean(), full_parameter_grad)


def test_v2_parameter_groups_are_disjoint_exhaustive_and_exclude_kendall() -> None:
    model = EgoStitchE2E(
        E2EConfig(
            d_model=16,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=2,
            n_inj=1,
            ste_dim=8,
            ste_layers=1,
            xattn_heads=2,
        )
    )
    composite = te._CompositeStep(model, world_size=1)
    manifest = te.build_v2_parameter_groups(model, composite)
    ids = [id(parameter) for group in manifest.groups.values() for parameter in group]
    assert len(ids) == len(set(ids)) == len(te._e2e_trainable_parameters(model))
    assert set(manifest.groups) == {
        "pair_encoder_head",
        "generator",
        "topology_content_conditioning",
    }
    assert all(not parameter.requires_grad for parameter in composite.kendall_log_vars.values())
    assert all(len(digest) == 64 for digest in manifest.sha256.values())


def test_v2_per_group_gradient_guards_clip_and_fail_closed() -> None:
    first = torch.nn.Parameter(torch.tensor([3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    records = te.v2_check_and_clip_gradients({"active": (first, second)}, {"active"})
    assert records["active"].norm == pytest.approx(5.0)
    assert records["active"].clip_coefficient == pytest.approx(0.2)
    assert torch.linalg.vector_norm(torch.stack([first.grad[0], second.grad[0]])) == pytest.approx(
        1.0
    )

    first.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        te.v2_check_and_clip_gradients({"active": (first,)}, {"active"})

    guard = te.V2ClipGuard(persistent_steps=2)
    clipped = te.V2GradientGroupRecord(True, 20.0, 0.05, 0)
    guard.update({"active": clipped})
    with pytest.raises(RuntimeError, match="persistent clipping"):
        guard.update({"active": clipped})

    te.v2_assert_replicated_squared_norms({"active": torch.tensor([4.0, 4.0])})
    with pytest.raises(RuntimeError, match="differ across ranks"):
        te.v2_assert_replicated_squared_norms({"active": torch.tensor([4.0, 5.0])})


def _record(epoch: int, *, mmd: float, brier: float, auprc: float = 0.6) -> te.V2CheckpointRecord:
    return te.V2CheckpointRecord(
        epoch=epoch,
        phase="C",
        full_joint_epochs_completed=epoch,
        guards_passed=True,
        auprc=auprc,
        prevalence=0.2,
        active_logit_std=0.2,
        clustering_mmd=mmd,
        brier=brier,
        warm_reference_std=0.4,
        warm_reference_auprc=0.61,
        residual_ratio=1e-2,
    )


def test_v2_eligibility_and_topology_aware_selection_are_fail_closed() -> None:
    warm = te.V2CheckpointRecord(**{**_record(1, mmd=0.1, brier=0.1).__dict__, "phase": "A"})
    assert not te.v2_checkpoint_eligible(warm, "full")
    assert te.select_v2_checkpoint([warm], "full") is None

    selected = te.select_v2_checkpoint(
        [
            _record(2, mmd=0.4, brier=0.1, auprc=0.62),
            _record(3, mmd=0.2, brier=0.3, auprc=0.60),
            _record(4, mmd=0.2 + 5e-7, brier=0.2, auprc=0.60),
        ],
        "full",
    )
    assert selected is not None and selected.epoch == 4
