"""Unit contract for the EgoStitch §13.19 training protocol."""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator
from src import train_egostitch as te
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import (
    ClassifierConfig,
    E2EConfig,
    EncoderConfig,
    GeneratorConfig,
)
from src.model.egostitch.generator.oracle import OracleStructGenerator

pytestmark = pytest.mark.unit


def test_quality_miss_uses_completed_final_epoch_instead_of_aborting() -> None:
    source = inspect.getsource(te._train_e2e_stability_loop)
    assert 'selection_status = "telemetry_miss_last_epoch"' in source
    assert "best_state = last_state" in source
    assert "produced no eligible checkpoint" not in source
    assert "enforce_quality = False" in source


def test_selection_returns_the_best_record_with_no_eligibility_predicate() -> None:
    """A degenerate record is still selected: eligibility is an owner judgement.

    There is deliberately no `e2e_checkpoint_eligible` anywhere in the tree —
    nothing in code decides whether a checkpoint is scientifically usable.
    """
    assert not hasattr(te, "e2e_checkpoint_eligible")
    record = te.E2ECheckpointRecord(
        epoch=30,
        phase="C",
        full_joint_epochs_completed=20,
        guards_passed=False,
        auprc=0.0,
        prevalence=0.1,
        active_logit_std=0.0,
        clustering_mmd=1.0,
        brier=1.0,
    )
    assert te.select_e2e_checkpoint([record], "full") == record


def _training_config(tmp_path: Path) -> Path:
    mapping = {
        "model": {"family": "egostitch_e2e", "config": {}},
        "data": {
            "root": str(tmp_path / "data"),
            "strategy": "breadth_first",
            "negative_ratio": 5,
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


def test_formal_output_metadata_matches_scorer_contract(tmp_path: Path) -> None:
    """The only end-to-end check that ``run_metadata.json`` is well-formed.

    Retargeted at the surviving key set after the registration excision
    (design 2026-08-02 Sec 10): no more preregistration binding, config hash,
    or git-derived implementation provenance -- just the plain run record the
    trainer and scorer still rely on.
    """
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/egostitch_e2e_v3_full_breadth_first.yaml"
    cfg = replace(
        te.load_config(config_path),
        output_dir=tmp_path / "formal",
        run_kind="formal",
    )
    # The plan pins a single `V_hold` manifest, so `V_hold` is the only role
    # production can emit
    # (`_E2E_VALIDATION_ROLE`, typed `Literal["V_hold"] | None`). The stub is
    # pinned to that constant rather than to a hand-written string, so a role
    # rename cannot leave this fixture asserting a value the worker never
    # writes.
    data = SimpleNamespace(
        rho_train=0.1,
        validation_role=te._E2E_VALIDATION_ROLE,
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
            "epochs_completed": 30,
            "optimizer_steps": 2340,
            "peak_memory_gib_per_rank": [1.0, 1.0, 1.0, 1.0],
            "parameter_groups": {"names": ["generator"], "sha256": {"generator": "a" * 64}},
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
    )
    te.write_outputs(result, cfg, data)

    metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
    # No eligibility verdict is recorded at all: the raw per-epoch metrics in
    # metrics.jsonl are the only basis, and reading them is the owner's call.
    assert "checkpoint_eligible" not in metadata
    assert "selected_checkpoint_eligible" not in metadata
    assert "quality_fields_policy" not in metadata
    assert metadata["arm"] == "full"
    assert metadata["config_sha256"] == te._sha256_file(config_path)
    assert metadata["checkpoint_sha256"] == te._sha256_file(cfg.output_dir / "best.pt")
    assert te._E2E_VALIDATION_ROLE == "V_hold"
    assert metadata["validation_role"] == "V_hold"
    assert metadata["run_kind"] == "formal"
    assert metadata["world_size"] == 4
    assert metadata["rho_train"] == data.rho_train
    assert metadata["access_audit"] == data.access_audit
    assert metadata["training_diagnostics"]["fidelity_series"] == [
        entry["fidelity"] for entry in result.history
    ]


def test_e2e_three_phase_boundaries_and_first_eligibility() -> None:
    assert te.e2e_phase_boundaries(2000) == (400, 600)
    warm = te.e2e_phase_state(399, 2000)
    first_edge = te.e2e_phase_state(400, 2000)
    assert warm == te.E2EPhaseState("A", 0.0, False, 0.0)
    assert not warm.edge_active
    assert first_edge.alpha == pytest.approx(1 / 200)
    assert first_edge.edge_active
    assert te.e2e_phase_state(599, 2000) == te.E2EPhaseState("B", 1.0, True, 1.0)
    assert te.e2e_phase_state(600, 2000) == te.E2EPhaseState("C", 1.0, True, 1.0)
    assert "pair_only" not in te.E2EPhaseState.__dataclass_fields__
    assert te.e2e_first_eligible_epoch(3000, 100) == 10


def test_e2e_recon_anneal_factors_are_component_specific_by_name() -> None:
    factors = te.e2e_recon_component_factors(1999, 2000)
    assert factors == {
        "feat": 0.25,
        "exist": 0.25,
        "mult": 0.25,
        "deg": 0.25,
        "slotadj": 1.0,
        "gate": 1.0,
        "ptr": 1.0,
        "align": 1.0,
        "div": 1.0,
        "rel": 1.0,
    }


def test_slot_collapse_guard_telemeters_two_consecutive_quality_misses() -> None:
    healthy = {
        "h_pairwise_cosine_mean": 0.2,
        "plan_rank1_marginal_residual": 0.4,
    }
    cosine_collapse = {**healthy, "h_pairwise_cosine_mean": 0.96}
    plan_collapse = {**healthy, "plan_rank1_marginal_residual": 0.04}

    for collapsed in (cosine_collapse, plan_collapse):
        guard = te.E2ESlotCollapseGuard()
        guard.update(collapsed, conditioning_active=False)
        guard.update(collapsed, conditioning_active=False)
        guard.update(collapsed, conditioning_active=True, enforce_quality=False)
        assert guard.update(collapsed, conditioning_active=True, enforce_quality=False)

    guard = te.E2ESlotCollapseGuard()
    for _ in range(4):
        guard.update(healthy, conditioning_active=True)

    telemetry_guard = te.E2ESlotCollapseGuard()
    telemetry_guard.update(cosine_collapse, conditioning_active=True)
    assert telemetry_guard.update(cosine_collapse, conditioning_active=True, enforce_quality=False)
    with pytest.raises(RuntimeError, match="non-finite E2E slot-collapse telemetry"):
        telemetry_guard.update(
            {**healthy, "h_pairwise_cosine_mean": float("nan")},
            conditioning_active=True,
            enforce_quality=False,
        )


def test_family_ratio_guard_telemeters_finite_quality_misses_but_not_nonfinite() -> None:
    zero = {"generator": {"edge": 0.0, "recon": 0.0}}
    telemetry_guard = te._E2EFamilyRatioGuard(threshold=2.0, required_probes=2)
    assert telemetry_guard.update(zero, enabled=True, enforce_quality=False) == {"generator": 0.0}
    assert telemetry_guard.ratio_defined == {"generator": False}

    imbalanced = {"generator": {"edge": 10.0, "recon": 1.0, "ssl": 1.0}}
    telemetry_guard.update(imbalanced, enabled=True, enforce_quality=False)
    telemetry_guard.update(imbalanced, enabled=True, enforce_quality=False)
    assert telemetry_guard.streaks == {"generator": 2}
    with pytest.raises(RuntimeError, match="non-finite E2E family-gradient norm"):
        telemetry_guard.update(
            {"generator": {"edge": float("nan"), "recon": 1.0}},
            enabled=True,
            enforce_quality=False,
        )


def test_rank1_plan_abort_is_not_blinded_by_concentrated_row_entropy() -> None:
    row_marginal = torch.tensor([0.74, 0.08666667, 0.08666667, 0.08666667])
    column_marginal = row_marginal.clone()
    plan = torch.outer(row_marginal, column_marginal).unsqueeze(0)
    pi = row_marginal.unsqueeze(0)
    h = torch.eye(4).unsqueeze(0)
    adj = torch.eye(4).unsqueeze(0)
    telemetry = te.e2e_dispersion_statistics(pi, h, adj, plan)

    assert telemetry["plan_row_entropy"] == pytest.approx(0.624, abs=0.01)
    assert telemetry["plan_rank1_marginal_residual"] < 0.05
    guard = te.E2ESlotCollapseGuard()
    guard.update(telemetry, conditioning_active=True)
    with pytest.raises(RuntimeError, match=r"training_invalid\(slot_collapse\)"):
        guard.update(telemetry, conditioning_active=True)


def test_degree_decorrelation_telemetry_uses_expected_validation_key() -> None:
    record = te.e2e_degree_decorrelation_telemetry(
        np.array([3.0, 1.0, 4.0, 2.0]),
        np.array([0.3, 0.1, 0.4, 0.2]),
    )
    assert record == {"topology_delta_degree_correlation": pytest.approx(1.0)}


def test_degree_telemetry_uses_validation_role_topology() -> None:
    data = cast(
        te.EgoStitchData,
        SimpleNamespace(
            validation_nodes=("a", "b", "c"),
            validation_positive_edges=(("a", "b"), ("a", "c")),
            val_pairs=[("a", "b"), ("b", "c")],
            target_builder=SimpleNamespace(graph=nx.Graph()),
        ),
    )
    np.testing.assert_array_equal(te._e2e_validation_endpoint_degrees(data), [3.0, 2.0])


class TestScaleTelemetry:
    def test_scale_rows_report_plan_scale_and_slot_geometry(self) -> None:
        torch.manual_seed(0)
        h = torch.randn(3, 4, 6)
        plan = torch.rand(3, 4, 4)
        rows = te._e2e_scale_rows(h, plan)

        assert set(rows) == {
            "plan_total_mass",
            "plan_max_cell_fraction",
            "h_norm_mean",
            "h_pairwise_sqdist_mean",
        }
        torch.testing.assert_close(rows["plan_total_mass"], plan.sum(dim=(1, 2)))
        torch.testing.assert_close(
            rows["h_norm_mean"], torch.linalg.vector_norm(h, dim=-1).mean(dim=-1)
        )
        assert bool((rows["plan_max_cell_fraction"] <= 1.0).all())

    def test_pairwise_squared_distance_matches_the_direct_computation(self) -> None:
        h = torch.randn(2, 5, 3)
        direct = ((h[:, :, None, :] - h[:, None, :, :]) ** 2).sum(-1)
        upper = torch.triu_indices(5, 5, offset=1)
        expected = direct[:, upper[0], upper[1]].mean(dim=-1)
        torch.testing.assert_close(
            te._e2e_scale_rows(h, torch.rand(2, 5, 5))["h_pairwise_sqdist_mean"],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_dispersion_keys_are_unchanged(self) -> None:
        """The probe ABI and fidelity_series bind these five names exactly."""
        rows = te._e2e_dispersion_rows(
            torch.rand(2, 4), torch.randn(2, 4, 6), torch.rand(2, 4, 4), torch.rand(2, 4, 4)
        )
        assert set(rows) == {
            "pi_slot_std",
            "h_pairwise_cosine_mean",
            "adj_offdiag_std",
            "plan_row_entropy",
            "plan_rank1_marginal_residual",
        }

    def test_plan_scale_metrics_match_hand_computed_values(self) -> None:
        """Assert `plan_total_mass`/`plan_max_cell_fraction` against hand arithmetic.

        `torch.rand` fixtures keep every cell below 1, so a `<= 1.0` bound
        check on `plan_max_cell_fraction` would pass even with the `/ mass`
        division missing entirely. This plan is hand-built instead: row 0's
        dominant cell (2.0) is off the diagonal and strictly less than the
        row total (4.0), so a missing division (which would leave the raw
        cell value 2.0 instead of 0.5) or a wrong-axis reduction (which would
        not find the true max at position (0, 1)) both fail the assertion.
        Row 1's dominant cell equals its total, exercising the fraction's
        other boundary.
        """
        plan = torch.tensor(
            [
                [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        h = torch.randn(2, 3, 5)
        rows = te._e2e_scale_rows(h, plan)

        torch.testing.assert_close(rows["plan_total_mass"], torch.tensor([4.0, 3.0]))
        torch.testing.assert_close(rows["plan_max_cell_fraction"], torch.tensor([0.5, 1.0]))

    def test_scale_rows_rejects_shape_mismatch_between_h_and_plan(self) -> None:
        with pytest.raises(ValueError, match="plan shape disagrees"):
            te._e2e_scale_rows(torch.randn(2, 4, 6), torch.rand(2, 3, 3))
        with pytest.raises(ValueError, match="scale inputs must have shapes"):
            te._e2e_scale_rows(torch.randn(2, 4, 6), torch.rand(2, 4))


def test_prefetch_batches_builds_ahead_without_reordering() -> None:
    second_started = threading.Event()
    release_second = threading.Event()

    def source() -> Iterator[int]:
        yield 1
        second_started.set()
        assert release_second.wait(timeout=1.0)
        yield 2

    batches = te._prefetch_batches(source(), depth=2)
    try:
        assert next(batches) == 1
        assert second_started.wait(timeout=1.0)
        release_second.set()
        assert next(batches) == 2
        with pytest.raises(StopIteration):
            next(batches)
    finally:
        release_second.set()
        batches.close()


def test_prefetch_batches_shutdowns_after_consumer_abort() -> None:
    second_started = threading.Event()
    release_second = threading.Event()

    def source() -> Iterator[int]:
        yield 1
        second_started.set()
        assert release_second.wait(timeout=1.0)
        yield 2

    batches = te._prefetch_batches(source(), depth=2)
    try:
        assert next(batches) == 1
        assert second_started.wait(timeout=1.0)
    finally:
        release_second.set()
        batches.close()

    assert not any(thread.name.startswith("egostitch-batch") for thread in threading.enumerate())


def test_prefetch_batches_propagates_producer_error_and_shutdowns() -> None:
    def source() -> Iterator[int]:
        yield 1
        raise ValueError("batch construction failed")

    batches = te._prefetch_batches(source(), depth=2)
    assert next(batches) == 1
    with pytest.raises(ValueError, match="batch construction failed"):
        next(batches)
    batches.close()

    assert not any(thread.name.startswith("egostitch-batch") for thread in threading.enumerate())


def test_e2e_lr_and_active_groups_follow_registered_phase_contract() -> None:
    config = te.EgoStitchTrainingConfig()
    full_model = EgoStitchModel(E2EConfig(encoder=EncoderConfig(w_rel=0.25)))
    assert te._e2e_base_lr(0, 2000, config) == pytest.approx(2e-7)
    assert te._e2e_base_lr(499, 2000, config) == pytest.approx(1e-4)
    assert te._e2e_base_lr(1999, 2000, config) == pytest.approx(1e-5)
    phase_a = te.E2EPhaseState("A", 0.0, False, 0.0)
    first_edge = te.e2e_phase_state(400, 2000)
    phase_c = te.E2EPhaseState("C", 1.0, True, 1.0)
    assert te._e2e_active_groups(phase_a, full_model) == {
        "generator",
        "encoder",
    }
    assert te._e2e_active_groups(first_edge, full_model) == {
        "classifier",
        "generator",
        "encoder",
    }
    assert te._e2e_active_groups(phase_c, full_model) == {
        "classifier",
        "generator",
        "encoder",
    }
    full_model.generator = OracleStructGenerator(slots=16)
    assert te._e2e_active_groups(phase_c, full_model) == {"classifier", "encoder"}
    full_phase_a_groups = te._e2e_active_groups(phase_a, full_model)
    full_edge_groups = te._e2e_active_groups(first_edge, full_model)
    assert te._e2e_optimizer_group_lr(1e-4, phase_a, "encoder", full_phase_a_groups) == 1e-4
    assert te._e2e_optimizer_group_lr(
        1e-4, first_edge, "encoder", full_edge_groups
    ) == pytest.approx(1e-4 * first_edge.alpha)
    assert te._e2e_optimizer_group_lr(1e-4, phase_a, "classifier", full_phase_a_groups) == 0.0
    assert te._e2e_optimizer_group_lr(1e-4, first_edge, "classifier", full_edge_groups) == 1e-4


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
    model = EgoStitchModel(
        E2EConfig(
            encoder=EncoderConfig(dim=8, layers=1),
            classifier=ClassifierConfig(
                d_model=16,
                encoder_layers=1,
                cross_attn_layers=1,
                n_heads=2,
                n_inj=1,
                xattn_heads=2,
            ),
        )
    )
    manifest = te.build_e2e_parameter_groups(model)
    ids = [id(parameter) for group in manifest.groups.values() for parameter in group]
    assert len(ids) == len(set(ids)) == len(te._e2e_trainable_parameters(model))
    assert set(manifest.groups) == {
        "generator",
        "encoder",
        "classifier",
    }
    assert all(len(digest) == 64 for digest in manifest.sha256.values())


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


def test_null_generator_has_empty_component_parameter_groups() -> None:
    """Null components stay absent without dropping classifier parameters."""
    cfg = E2EConfig(
        generator=GeneratorConfig(name="null"),
        encoder=EncoderConfig(dim=16, layers=1),
        classifier=ClassifierConfig(
            d_model=32, encoder_layers=1, cross_attn_layers=2, n_heads=2, xattn_heads=2
        ),
    )
    model = EgoStitchModel(cfg)
    assert model.encoder is None
    groups = te.build_e2e_parameter_groups(model)
    assert groups.groups["generator"] == ()
    assert groups.groups["encoder"] == ()
    grouped = {id(p) for params in groups.groups.values() for p in params}
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
    assert len(grouped) > 0

    phase_c = te.E2EPhaseState("C", 1.0, True, 1.0)
    assert te._e2e_active_groups(phase_c, model) == {"classifier"}


def test_group_gradient_statistics_gather_once_and_reuse_for_clipping() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    groups = {"active": (parameter,), "empty": ()}

    class _FakeAccelerator:
        calls = 0

        def gather(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.cat((value, value), dim=0)

    accelerator = _FakeAccelerator()
    local, gathered_squared, gathered_nonfinite = te._gather_e2e_group_gradient_statistics(
        groups, cast(Accelerator, accelerator)
    )

    assert accelerator.calls == 1
    assert gathered_squared["active"].tolist() == [25.0, 25.0]
    assert gathered_nonfinite["active"].tolist() == [0, 0]
    assert gathered_squared["empty"].tolist() == [0.0, 0.0]

    records = te.e2e_check_and_clip_gradients(
        groups,
        {"active"},
        max_norm={"active": 1.0, "empty": 1.0},
        statistics=local,
    )
    assert records["active"].norm == pytest.approx(5.0)
    assert parameter.grad.tolist() == pytest.approx([0.6, 0.8])


def test_optimizer_state_finite_check_reports_corrupt_group() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    groups = {"classifier": (parameter,)}

    te.e2e_assert_finite_optimizer_state(groups, optimizer)
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(RuntimeError, match="optimizer state.*classifier"):
        te.e2e_assert_finite_optimizer_state(groups, optimizer)
