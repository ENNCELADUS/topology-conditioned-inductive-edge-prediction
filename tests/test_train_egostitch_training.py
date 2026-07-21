"""Unit contract for the EgoStitch §13.19 training protocol."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from src import train_egostitch as te
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E

pytestmark = pytest.mark.unit


def _training_config(tmp_path: Path) -> Path:
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
        "training": {},
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def test_e2e_config_schema_is_strict_and_preserves_run_kind(tmp_path: Path) -> None:
    cfg = te.load_config(_training_config(tmp_path))
    assert cfg.training == te.EgoStitchTrainingConfig()

    raw = yaml.safe_load(_training_config(tmp_path).read_text(encoding="utf-8"))
    raw["training"]["positive_weight"] = 4.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        te.load_config(bad)


def test_formal_binding_preflight_validates_live_config_and_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/egostitch_e2e_breadth_first.yaml"
    cfg = te.load_config(config_path)
    arm_paths = {
        "full": "configs/egostitch_e2e_breadth_first.yaml",
        "b0_e2e_f_only": "configs/egostitch_e2e_f_only_breadth_first.yaml",
        "pair_topology": "configs/egostitch_e2e_pair_topology_breadth_first.yaml",
        "p0": "configs/egostitch_e2e_p0_breadth_first.yaml",
    }
    artifact = tmp_path / "binding-artifact.json"
    artifact.write_text('{"status":"pass"}\n')
    artifact_record = {"path": str(artifact), "sha256": te._sha256_file(artifact)}
    evidence: dict[str, object] = {
        "schema_version": "egostitch_e2e_binding_evidence_v1",
        "implementation": {"commit": "a" * 40},
        "configs": {
            arm: {"path": path, "sha256": te._sha256_file(root / path)}
            for arm, path in arm_paths.items()
        },
        "parameter_group_manifests": dict(artifact_record),
        "packs_and_validation_manifests": dict(artifact_record),
        "qualification_attempts": dict(artifact_record),
        "boundary_access_audit": dict(artifact_record),
        "runtime_and_peak_memory": dict(artifact_record),
        "checkpoint_policy_version": "v1",
    }
    snapshot = te.PreregistrationSnapshot(
        {
            "arms": {arm: {"training": path} for arm, path in arm_paths.items()},
            "binding_evidence": evidence,
        },
        "f" * 64,
    )

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=("a" * 40 + "\n") if "rev-parse" in command else "")

    monkeypatch.setattr(te.subprocess, "run", fake_run)
    binding = te._validate_e2e_formal_binding(cfg, snapshot, config_path)

    assert binding["arm"] == "full"
    assert binding["config_sha256"] == te._sha256_file(config_path)
    artifact.unlink()
    with pytest.raises(te.PreregistrationNotBinding, match="missing or hash-mismatched"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    artifact.write_text('{"status":"pass"}\n')
    configs = evidence["configs"]
    assert isinstance(configs, dict)
    full = configs["full"]
    assert isinstance(full, dict)
    full["sha256"] = "0" * 64
    with pytest.raises(te.PreregistrationNotBinding, match="live config digest"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)


def test_formal_output_metadata_matches_scorer_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/egostitch_e2e_breadth_first.yaml"
    cfg = replace(
        te.load_config(config_path),
        output_dir=tmp_path / "formal",
        run_kind="formal",
    )
    data = SimpleNamespace(
        rho_train=0.1,
        validation_role="V_select",
        access_audit={"observed_training_access": []},
    )
    metrics = te.EdgeMetrics(
        auroc=0.5,
        auprc=0.5,
        accuracy=0.5,
        sensitivity=0.5,
        specificity=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        mcc=0.0,
        ece=0.1,
        brier=0.25,
        threshold=0.5,
        n_pos=1,
        n_neg=1,
    )
    result = te.EgoTrainResult(
        best_state_dict={"weight": torch.tensor([1.0])},
        best_epoch=10,
        best_val_metrics=metrics,
        last_state_dict={"weight": torch.tensor([1.0])},
        last_epoch=30,
        last_val_metrics=metrics,
        history=[{"epoch": 10.0, "fidelity": {"topology_delta_ratio": 0.01}}],
        counterfactual_stop_epoch=None,
        runtime_profile={
            "selected_epoch": 10,
            "gradient_norm_series": [],
            "optimizer_step_gradients": [],
            "kendall_fallback": {},
        },
        kendall_state={},
    )
    te.write_run_start_metadata(
        cfg,
        data,
        world_size=4,
        config_path=config_path,
        formal_binding={"implementation_commit": "a" * 40},
    )
    te.write_outputs(result, cfg, data)

    metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
    assert metadata["selected_checkpoint_eligible"] is True
    assert metadata["arm"] == "full"
    assert metadata["config_sha256"] == te._sha256_file(config_path)
    assert metadata["implementation_commit"] == "a" * 40
    assert metadata["checkpoint_sha256"] == te._sha256_file(cfg.output_dir / "best.pt")


def test_run_kinds_enforce_registered_boundaries(tmp_path: Path) -> None:
    loaded = te.load_config(_training_config(tmp_path))
    overfit = te.apply_overrides(
        loaded,
        te.EgoCliArgs(
            config=tmp_path / "training.yaml", seed=None, output_dir=None, run_kind="overfit"
        ),
    )
    prepared, is_debug, _ = te.prepare_ddp_run_config(overfit, max_steps=None)
    assert prepared.run_kind == "overfit"
    assert is_debug is False
    with pytest.raises(ValueError, match="2,000 registered steps"):
        te.prepare_ddp_run_config(overfit, max_steps=2000)

    rehearsal = te.apply_overrides(
        loaded,
        te.EgoCliArgs(
            config=tmp_path / "training.yaml", seed=None, output_dir=None, run_kind="rehearsal"
        ),
    )
    with pytest.raises(ValueError, match="complete schedule"):
        te.prepare_ddp_run_config(rehearsal, max_steps=1)

    formal = loaded
    with pytest.raises(te.PreregistrationNotBinding):
        te.prepare_ddp_run_config(formal, max_steps=None)


def test_e2e_three_phase_boundaries_and_first_eligibility() -> None:
    epoch_steps = te.e2e_overfit_epoch_step_counts(30)
    assert len(epoch_steps) == 30
    assert sum(epoch_steps) == 2000
    assert epoch_steps[:20] == (67,) * 20
    assert epoch_steps[20:] == (66,) * 10
    assert te.e2e_overfit_epoch_step_counts(30, profile_only=True) == (67,)
    assert te.e2e_phase_boundaries(2000) == (400, 600)
    assert te.e2e_phase_state(399, 2000) == te.E2EPhaseState("A", 0.0, True, 0.0)
    assert te.e2e_phase_state(400, 2000).alpha == pytest.approx(1 / 200)
    assert te.e2e_phase_state(599, 2000) == te.E2EPhaseState("B", 1.0, False, 1.0)
    assert te.e2e_phase_state(600, 2000) == te.E2EPhaseState("C", 1.0, False, 1.0)
    assert te.e2e_first_eligible_epoch(3000, 100) == 10


def test_e2e_lr_and_active_groups_follow_registered_phase_contract() -> None:
    config = te.EgoStitchTrainingConfig()
    assert te._e2e_base_lr(0, 2000, config) == pytest.approx(2e-7)
    assert te._e2e_base_lr(499, 2000, config) == pytest.approx(1e-4)
    assert te._e2e_base_lr(1999, 2000, config) == pytest.approx(1e-5)
    phase_a = te.E2EPhaseState("A", 0.0, True, 0.0)
    phase_c = te.E2EPhaseState("C", 1.0, False, 1.0)
    assert te._e2e_active_groups(phase_a, "full") == {
        "pair_encoder_head",
        "generator",
    }
    assert te._e2e_active_groups(phase_c, "full") == {
        "pair_encoder_head",
        "generator",
        "topology_content_conditioning",
    }
    assert te._e2e_active_groups(phase_c, "b0_e2e_f_only") == {
        "pair_encoder_head",
        "generator",
    }


def test_qualification_profile_requires_registered_guard_margins(tmp_path: Path) -> None:
    profile = {
        "total_optimizer_steps": 4,
        "optimizer_step_gradients": [
            {
                "optimizer_group_gradients": {
                    "pair_encoder_head": {"active": True, "clip_coefficient": 0.5},
                    "generator": {"active": True, "clip_coefficient": 0.6},
                }
            }
            for _ in range(4)
        ],
        "gradient_norm_series": [
            {
                "alpha": 1.0,
                "family_group_ratios": {"generator": 2.0},
                "submodule_gradient_rms": {
                    "grad_rms_trunk": 0.1,
                    "grad_rms_ste": 0.01,
                    "grad_rms_content": 0.02,
                },
            }
        ],
    }
    path = tmp_path / "profile.json"
    output = tmp_path / "margins.json"
    path.write_text(json.dumps(profile))

    summary = te.validate_e2e_qualification_profile(path, output_path=output)

    assert summary["status"] == "pass"
    assert json.loads(output.read_text())["family_ratio_p99"] == pytest.approx(2.0)
    profile["optimizer_step_gradients"][0]["optimizer_group_gradients"]["generator"][  # type: ignore[index]
        "clip_coefficient"
    ] = 0.001
    path.write_text(json.dumps(profile))
    with pytest.raises(RuntimeError, match="clip margins"):
        te.validate_e2e_qualification_profile(path)


def test_e2e_weighted_bce_matches_one_and_two_rank_gradients_with_padding() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    logits = torch.linspace(-0.3, 0.3, len(labels), requires_grad=True)
    mask = torch.ones_like(labels)
    full_loss = te.e2e_weighted_bce_with_logits(logits, labels, mask)
    (full_grad,) = torch.autograd.grad(full_loss, logits)

    global_denominator = torch.tensor(15.0)
    rank_grads = []
    for indices, rank_mask in (
        ([0, 2, 4, 6], [1.0, 1.0, 1.0, 1.0]),
        ([1, 3, 5, 5], [1.0, 1.0, 1.0, 0.0]),
    ):
        rank_logits = logits.detach()[indices].clone().requires_grad_()
        rank_labels = labels[indices]
        rank_loss = te.e2e_weighted_bce_with_logits(
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
    balanced = te.e2e_weighted_bce_with_logits(
        zero_logits, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), torch.ones(6)
    )
    (zero_grad,) = torch.autograd.grad(balanced, zero_logits)
    assert float(zero_grad.sum()) == pytest.approx(0.0, abs=1e-8)

    features = torch.linspace(-1.0, 1.0, len(labels))
    full_weight = torch.tensor(0.2, requires_grad=True)
    full_parameter_loss = te.e2e_weighted_bce_with_logits(full_weight * features, labels, mask)
    (full_parameter_grad,) = torch.autograd.grad(full_parameter_loss, full_weight)
    rank_parameter_grads = []
    for indices, rank_mask in (
        ([0, 2, 4, 6], [1.0, 1.0, 1.0, 1.0]),
        ([1, 3, 5, 5], [1.0, 1.0, 1.0, 0.0]),
    ):
        rank_weight = full_weight.detach().clone().requires_grad_()
        rank_loss = te.e2e_weighted_bce_with_logits(
            rank_weight * features[indices],
            labels[indices],
            torch.tensor(rank_mask),
            world_size=2,
            all_reduce_sum=lambda _: global_denominator,
        )
        rank_parameter_grads.append(torch.autograd.grad(rank_loss, rank_weight)[0])
    torch.testing.assert_close(torch.stack(rank_parameter_grads).mean(), full_parameter_grad)


def test_e2e_parameter_groups_are_disjoint_exhaustive_and_exclude_kendall() -> None:
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
    manifest = te.build_e2e_parameter_groups(model)
    ids = [id(parameter) for group in manifest.groups.values() for parameter in group]
    assert len(ids) == len(set(ids)) == len(te._e2e_trainable_parameters(model))
    assert set(manifest.groups) == {
        "pair_encoder_head",
        "generator",
        "topology_content_conditioning",
    }
    assert all(len(digest) == 64 for digest in manifest.sha256.values())


def test_e2e_per_group_gradient_guards_clip_and_fail_closed() -> None:
    first = torch.nn.Parameter(torch.tensor([3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    records = te.e2e_check_and_clip_gradients({"active": (first, second)}, {"active"})
    assert records["active"].norm == pytest.approx(5.0)
    assert records["active"].clip_coefficient == pytest.approx(0.2)
    assert torch.linalg.vector_norm(torch.stack([first.grad[0], second.grad[0]])) == pytest.approx(
        1.0
    )

    first.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        te.e2e_check_and_clip_gradients({"active": (first,)}, {"active"})

    guard = te.E2EClipGuard(persistent_steps=2)
    clipped = te.E2EGradientGroupRecord(True, 20.0, 0.05, 0)
    guard.update({"active": clipped})
    with pytest.raises(RuntimeError, match="persistent clipping"):
        guard.update({"active": clipped})

    te.e2e_assert_replicated_squared_norms({"active": torch.tensor([4.0, 4.0])})
    with pytest.raises(RuntimeError, match="differ across ranks"):
        te.e2e_assert_replicated_squared_norms({"active": torch.tensor([4.0, 5.0])})


def _record(epoch: int, *, mmd: float, brier: float, auprc: float = 0.6) -> te.E2ECheckpointRecord:
    return te.E2ECheckpointRecord(
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


def test_e2e_eligibility_and_topology_aware_selection_are_fail_closed() -> None:
    warm = te.E2ECheckpointRecord(**{**_record(1, mmd=0.1, brier=0.1).__dict__, "phase": "A"})
    assert not te.e2e_checkpoint_eligible(warm, "full")
    assert te.select_e2e_checkpoint([warm], "full") is None

    selected = te.select_e2e_checkpoint(
        [
            _record(2, mmd=0.4, brier=0.1, auprc=0.62),
            _record(3, mmd=0.2, brier=0.3, auprc=0.60),
            _record(4, mmd=0.2 + 5e-7, brier=0.2, auprc=0.60),
        ],
        "full",
    )
    assert selected is not None and selected.epoch == 4
