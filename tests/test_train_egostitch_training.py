"""Unit contract for the EgoStitch §13.19 training protocol."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import networkx as nx
import numpy as np
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
    config_path = root / "configs/egostitch_e2e_v3_full_breadth_first.yaml"
    cfg = te.load_config(config_path)
    arm_paths = {
        "full": "configs/egostitch_e2e_v3_full_breadth_first.yaml",
        "b0_e2e_f_only": "configs/egostitch_e2e_v3_f_only_breadth_first.yaml",
        "pair_topology": "configs/egostitch_e2e_v3_pair_topology_breadth_first.yaml",
        "p0": "configs/egostitch_e2e_v3_p0_breadth_first.yaml",
        "cosine_pool": "configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml",
        "no_l_rel": "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
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
            "arms": {
                **{
                    arm: {"kind": "trained_checkpoint", "training": path}
                    for arm, path in arm_paths.items()
                },
                "structure_control_6a_v3": {
                    "kind": "scoring_time_control",
                    "training": None,
                    "checkpoint_arm": "full",
                },
                "structure_control_6e_v1": {
                    "kind": "scoring_time_control",
                    "training": None,
                    "checkpoint_arm": "full",
                },
            },
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

    v2_configs = cast(dict[str, object], evidence["configs"])
    evidence["configs"] = {
        name: entry for name, entry in v2_configs.items() if name not in {"cosine_pool", "no_l_rel"}
    }
    with pytest.raises(te.PreregistrationNotBinding, match="six trained"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    evidence["configs"] = {**v2_configs, "unknown": v2_configs["full"]}
    with pytest.raises(te.PreregistrationNotBinding, match="six trained"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    evidence["configs"] = v2_configs

    registered_arms = cast(dict[str, object], snapshot.payload["arms"])
    snapshot.payload["arms"] = {
        **{
            name: registered_arms[name]
            for name in ("full", "b0_e2e_f_only", "pair_topology", "p0")
        },
        "structure_control_6a": registered_arms["structure_control_6a_v3"],
    }
    with pytest.raises(te.PreregistrationNotBinding, match="six-trained-plus-two-control"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    snapshot.payload["arms"] = {**registered_arms, "unknown": registered_arms["full"]}
    with pytest.raises(te.PreregistrationNotBinding, match="six-trained-plus-two-control"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    snapshot.payload["arms"] = registered_arms

    def fake_run_descent(diff_paths: str, ancestor_rc: int) -> Callable[..., SimpleNamespace]:
        def runner(command: list[str], **_: object) -> SimpleNamespace:
            if "rev-parse" in command:
                return SimpleNamespace(stdout="b" * 40 + "\n", returncode=0)
            if "merge-base" in command:
                return SimpleNamespace(stdout="", returncode=ancestor_rc)
            if "diff" in command:
                return SimpleNamespace(stdout=diff_paths, returncode=0)
            return SimpleNamespace(stdout="", returncode=0)

        return runner

    monkeypatch.setattr(
        "src.train_egostitch.subprocess.run",
        fake_run_descent("docs/registrations/g5_e2e_stage1_preregistration_v2.json\n", 0),
    )
    assert te._validate_e2e_formal_binding(cfg, snapshot, config_path)["arm"] == "full"
    monkeypatch.setattr(
        "src.train_egostitch.subprocess.run",
        fake_run_descent("docs/registrations/x.json\nsrc/train_b0.py\n", 0),
    )
    with pytest.raises(te.PreregistrationNotBinding, match="clean live checkout"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    monkeypatch.setattr("src.train_egostitch.subprocess.run", fake_run_descent("", 1))
    with pytest.raises(te.PreregistrationNotBinding, match="clean live checkout"):
        te._validate_e2e_formal_binding(cfg, snapshot, config_path)
    monkeypatch.setattr("src.train_egostitch.subprocess.run", fake_run)
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
    config_path = root / "configs/egostitch_e2e_v3_full_breadth_first.yaml"
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
    assert metadata["arm_kind"] == "trained_checkpoint"
    assert metadata["checkpoint_arm"] == "full"
    assert metadata["scoring_semantics"] == {
        "scaffold_control": "none",
        "permanent_null": "none",
        "primary_logit": "full",
    }
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


def test_e2e_eligibility_reference_starts_at_first_edge_active_validation() -> None:
    phase_a = te.e2e_phase_state(399, 2000)
    first_edge_active = te.e2e_phase_state(400, 2000)

    assert not te._e2e_should_capture_eligibility_reference(
        phase_a,
        warm_reference_auprc=None,
    )
    assert te._e2e_should_capture_eligibility_reference(
        first_edge_active,
        warm_reference_auprc=None,
    )
    assert not te._e2e_should_capture_eligibility_reference(
        first_edge_active,
        warm_reference_auprc=0.4,
    )


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


def test_slot_collapse_guard_uses_two_active_consecutive_validations_per_arm() -> None:
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
        guard.update(collapsed, conditioning_active=True)
        with pytest.raises(RuntimeError, match=r"training_invalid\(slot_collapse\)"):
            guard.update(collapsed, conditioning_active=True)

    guard = te.E2ESlotCollapseGuard()
    for _ in range(4):
        guard.update(healthy, conditioning_active=True)


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


def test_overfit_profile_is_the_first_production_epoch_not_a_compressed_schedule() -> None:
    production = te.e2e_overfit_epoch_step_counts(30)
    sampled = te.e2e_overfit_epoch_step_counts(30, profile_only=True)

    assert sampled == production[:1]
    assert all(
        te.e2e_phase_state(step, sum(production)).phase == "A" for step in range(sum(sampled))
    )


def test_overfit_rows_and_target_seed_are_world_size_invariant() -> None:
    rows = tuple((f"n{i}", f"n{i + 1}", i % 2) for i in range(10))
    manifest = te.OverfitManifest(rows=rows, sha256="a" * 64)
    expected = te.e2e_overfit_step_rows(manifest, step=1, batch_size=6)
    assert expected == rows[6:] + rows[:2]

    for world_size in (1, 2, 4):
        shards = [
            te.e2e_overfit_rank_step_rows(
                manifest,
                step=1,
                global_batch_size=6,
                rank=rank,
                world_size=world_size,
            )
            for rank in range(world_size)
        ]
        recovered = tuple(
            shards[index % world_size][index // world_size] for index in range(len(expected))
        )
        assert recovered == expected

    assert te.e2e_overfit_target_seed(seed=7, step=11, rank=3) == (7, 0, 11, 3, 0x7A)


def test_fixed_overfit_batches_preserve_global_rows_and_four_rank_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = te.load_config(_training_config(tmp_path))
    rows = tuple((f"n{i}", f"n{i + 1}", i % 2) for i in range(10))
    manifest = te.OverfitManifest(rows=rows, sha256="a" * 64)
    seen_rows: dict[int, tuple[tuple[str, str, int], ...]] = {}
    seen_node_epochs: dict[int, int] = {}
    seen_rng_draws: dict[int, int] = {}

    class _TargetBuilder:
        def __init__(self, rank: int) -> None:
            self.rank = rank

        def build(self, nodes: list[str], rng: np.random.Generator) -> None:
            del nodes
            seen_rng_draws[self.rank] = int(rng.integers(0, 2**31))

    def _next_nodes(factory: te._BatchFactory) -> list[str]:
        del factory
        return ["n0"]

    def _node_tensors(
        factory: te._BatchFactory,
        nodes: list[str],
        targets: None,
        *,
        epoch: int,
        step: int,
    ) -> dict[str, torch.Tensor]:
        del nodes, targets, step
        seen_node_epochs[factory._rank] = epoch
        return {
            "x": torch.zeros(1, 1),
            "ground_x": torch.zeros(1, 1, 1),
            "target_features": torch.zeros(1, 1, 1),
        }

    def _edge_tensors(
        factory: te._BatchFactory,
        local_rows: tuple[tuple[str, str, int], ...],
        *,
        pad_to: int,
        epoch: int,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], int]:
        del epoch, step
        seen_rows[factory._rank] = local_rows
        labels = [label for _, _, label in local_rows]
        return {
            "x_i": torch.zeros(pad_to, 1),
            "label": torch.tensor(labels + [0] * (pad_to - len(labels))),
            "edge_mask": torch.tensor([1] * len(labels) + [0] * (pad_to - len(labels))),
        }, len(local_rows)

    monkeypatch.setattr(te._BatchFactory, "_next_nodes", _next_nodes)
    monkeypatch.setattr(te._BatchFactory, "_node_tensors", _node_tensors)
    monkeypatch.setattr(te._BatchFactory, "_edge_tensors", _edge_tensors)

    batches: list[te._CompositeBatch] = []
    for rank in range(4):
        factory = object.__new__(te._BatchFactory)
        factory._cfg = cfg
        factory._rank = rank
        factory._world = 4
        factory._data = cast(te.EgoStitchData, SimpleNamespace(target_builder=_TargetBuilder(rank)))
        batches.append(
            next(
                factory.fixed_row_batches(
                    manifest=manifest,
                    epoch=1,
                    steps=1,
                    step_offset=1,
                )
            )
        )

    expected = te.e2e_overfit_step_rows(manifest, step=1, batch_size=cfg.data.edge_batch)
    recovered = tuple(seen_rows[index % 4][index // 4] for index in range(cfg.data.edge_batch))
    assert recovered == expected
    assert all(batch.edge_rows_global == cfg.data.edge_batch for batch in batches)
    assert [batch.edge_rows_true for batch in batches] == [2, 2, 1, 1]
    assert [int(batch.edge["edge_mask"].sum()) for batch in batches] == [2, 2, 1, 1]
    assert seen_node_epochs == dict.fromkeys(range(4), 0)
    assert seen_rng_draws == {
        rank: int(np.random.default_rng((cfg.seed, 0, 1, rank, 0x7A)).integers(0, 2**31))
        for rank in range(4)
    }


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
    full_model = EgoStitchE2E(E2EConfig(w_rel=0.25))
    assert te._e2e_base_lr(0, 2000, config) == pytest.approx(2e-7)
    assert te._e2e_base_lr(499, 2000, config) == pytest.approx(1e-4)
    assert te._e2e_base_lr(1999, 2000, config) == pytest.approx(1e-5)
    phase_a = te.E2EPhaseState("A", 0.0, False, 0.0)
    first_edge = te.e2e_phase_state(400, 2000)
    phase_c = te.E2EPhaseState("C", 1.0, True, 1.0)
    assert te._e2e_active_groups(phase_a, full_model) == {
        "generator",
        "topology_content_conditioning",
    }
    assert te._e2e_active_groups(first_edge, full_model) == {
        "pair_encoder_head",
        "generator",
        "topology_content_conditioning",
    }
    assert te._e2e_active_groups(phase_c, full_model) == {
        "pair_encoder_head",
        "generator",
        "topology_content_conditioning",
    }
    full_phase_a_groups = te._e2e_active_groups(phase_a, full_model)
    full_edge_groups = te._e2e_active_groups(first_edge, full_model)
    assert (
        te._e2e_optimizer_group_lr(
            1e-4, phase_a, "topology_content_conditioning", full_phase_a_groups
        )
        == 1e-4
    )
    assert te._e2e_optimizer_group_lr(
        1e-4, first_edge, "topology_content_conditioning", full_edge_groups
    ) == pytest.approx(1e-4 * first_edge.alpha)
    assert (
        te._e2e_optimizer_group_lr(1e-4, phase_a, "pair_encoder_head", full_phase_a_groups)
        == 0.0
    )
    assert (
        te._e2e_optimizer_group_lr(1e-4, first_edge, "pair_encoder_head", full_edge_groups)
        == 1e-4
    )


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


def test_qualification_clip_floors_are_per_group_and_fail_closed(tmp_path: Path) -> None:
    """The calibrated per-group p1 floors admit the retained rehearsal pattern.

    Regression for the 2026-07-22 attempt-3 margins failure: measured p1
    0.1100/0.0281/0.5187 (pair/generator/topology) must pass the calibrated
    floors 0.04/0.01/0.15, while an unlisted group falls back to the
    scaffold-era 0.12 default.
    """

    def _profile(groups: dict[str, float]) -> dict[str, object]:
        return {
            "total_optimizer_steps": 200,
            "optimizer_step_gradients": [
                {
                    "optimizer_group_gradients": {
                        # The first 4 of 200 steps (2%) carry each group's
                        # p1-scale low value so np.percentile(values, 1) lands
                        # on it; the rest sit high so streak/minimum stay clean.
                        name: {
                            "active": True,
                            "clip_coefficient": low if step < 4 else max(0.5, low * 5),
                        }
                        for name, low in groups.items()
                    }
                }
                for step in range(200)
            ],
            "gradient_norm_series": [
                {
                    "alpha": 1.0,
                    "family_group_ratios": {"generator": 13.7},
                    "submodule_gradient_rms": {
                        "grad_rms_trunk": 0.1,
                        "grad_rms_ste": 0.01,
                        "grad_rms_content": 0.02,
                    },
                }
            ],
        }

    rehearsal_like = _profile(
        {
            "pair_encoder_head": 0.110,
            "generator": 0.0281,
            "topology_content_conditioning": 0.5187,
        }
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(rehearsal_like))
    summary = te.validate_e2e_qualification_profile(path)
    assert summary["status"] == "pass"
    clip_groups = cast(dict[str, dict[str, float]], summary["clip_groups"])
    assert clip_groups["generator"]["p1_floor"] == pytest.approx(0.01)
    assert clip_groups["topology_content_conditioning"]["p1_floor"] == pytest.approx(0.15)

    topology_below_floor = _profile(
        {
            "pair_encoder_head": 0.110,
            "generator": 0.0281,
            "topology_content_conditioning": 0.10,
        }
    )
    path.write_text(json.dumps(topology_below_floor))
    with pytest.raises(RuntimeError, match="clip margins failed for topology"):
        te.validate_e2e_qualification_profile(path)

    unknown_group_uses_default = _profile({"mystery_group": 0.05})
    path.write_text(json.dumps(unknown_group_uses_default))
    with pytest.raises(RuntimeError, match="clip margins failed for mystery_group"):
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

    generator = torch.nn.Parameter(torch.tensor([0.0]))
    generator.grad = torch.tensor([1545.4539013796064])
    calibrated = te.e2e_check_and_clip_gradients(
        {"generator": (generator,)},
        {"generator"},
        max_norm={"generator": 3.0},
    )
    assert calibrated["generator"].clip_coefficient == pytest.approx(3.0 / 1545.4539013796064)
    te.E2EClipGuard().update(calibrated)
    with pytest.raises(ValueError, match="max_norm mapping"):
        te.e2e_check_and_clip_gradients({"generator": (generator,)}, {"generator"}, max_norm={})
    with pytest.raises(ValueError, match="max_norm mapping"):
        te.e2e_check_and_clip_gradients(
            {"generator": (generator,)}, {"generator"}, max_norm={"generator": 0.0}
        )

    first.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        te.e2e_check_and_clip_gradients({"active": (first,)}, {"active"})

    guard = te.E2EClipGuard(persistent_steps=2)
    clipped = te.E2EGradientGroupRecord(True, 20.0, 0.05, 0)
    guard.update({"active": clipped}, step=11, phase="A")
    with pytest.raises(RuntimeError, match="persistent clipping") as error:
        guard.update({"active": clipped}, step=12, phase="A")
    message = str(error.value)
    assert "step=12" in message
    assert "phase=A" in message
    assert "norm=20.0" in message
    assert "coefficient=0.05" in message
    assert '"step": 11' in message
    assert '"step": 12' in message

    probe_guard = te.E2EClipGuard(persistent_steps=2)
    probe_guard.update({"active": clipped}, enforce_persistent=False)
    probe_guard.update({"active": clipped}, enforce_persistent=False)
    with pytest.raises(RuntimeError, match="persistent clipping"):
        probe_guard.update({"active": clipped})
    extreme = te.E2EGradientGroupRecord(True, 4000.0, 0.00075, 0)
    with pytest.raises(RuntimeError, match="extreme clipping"):
        te.E2EClipGuard().update({"active": extreme}, enforce_persistent=False)

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


def test_e2e_eligibility_reference_keeps_pair_head_learning_guard_strict() -> None:
    learned = _record(1, mmd=0.1, brier=0.1, auprc=0.23)
    learned = te.E2ECheckpointRecord(
        **{
            **learned.__dict__,
            "prevalence": 0.2,
            "warm_reference_auprc": 0.23,
        }
    )
    assert te.e2e_checkpoint_eligible(learned, "full")

    at_threshold = te.E2ECheckpointRecord(
        **{
            **learned.__dict__,
            "warm_reference_auprc": 0.22,
        }
    )
    below_threshold = te.E2ECheckpointRecord(
        **{
            **learned.__dict__,
            "warm_reference_auprc": 0.21,
        }
    )
    assert te.e2e_checkpoint_eligible(at_threshold, "full")
    assert not te.e2e_checkpoint_eligible(below_threshold, "full")
