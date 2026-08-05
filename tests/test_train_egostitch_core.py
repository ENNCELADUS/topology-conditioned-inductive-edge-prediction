"""Live-core tests for the single-stage EgoStitch E2E worker."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]
from src import train_egostitch as te
from src.data.feature_stats import compute_feature_stats
from src.eval.edge_metrics import EdgeMetrics
from src.model.egostitch import EgoStitchConfig
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig

from tests.test_train_egostitch import (
    _E2E_TINY_MODEL,
    _config_mapping,
    _e2e_model_config,
    _toy_bundle,
    _toy_cfg,
)

pytestmark = pytest.mark.unit

_Mutator = Callable[[te.EgoConfig], te.EgoConfig]


# --------------------------------------------------------------------------- helpers


def _edge_metrics() -> EdgeMetrics:
    return EdgeMetrics(
        auroc=0.5,
        auprc=0.5,
        accuracy=0.5,
        sensitivity=0.5,
        specificity=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        mcc=0.0,
        ece=0.0,
        brier=0.25,
        threshold=0.5,
        n_pos=1,
        n_neg=1,
    )


def _stub_result(*, selected_epoch: int | None = 1) -> te.EgoTrainResult:
    """A minimal finished-run bundle satisfying `write_outputs`' contract.

    The artifact writer is what the run-kind gating lives in, so these tests
    drive it directly rather than through a training loop -- the only
    executable loop left is the §13.19 E2E schedule, which needs a token pack.
    """
    return te.EgoTrainResult(
        best_state_dict={"w": torch.zeros(1)},
        best_epoch=1,
        best_val_metrics=_edge_metrics(),
        last_state_dict={"w": torch.zeros(1)},
        last_epoch=1,
        last_val_metrics=_edge_metrics(),
        history=[
            {
                "epoch": 1.0,
                "auprc": 0.5,
                "loss_total": 1.0,
                "fidelity": {"topology_delta_ratio": 1.0},
                "gradient_norm_probes": [],
            }
        ],
        counterfactual_stop_epoch=None,
        runtime_profile={
            "gradient_norm_series": [],
            "optimizer_step_gradients": [],
            "kendall_fallback": {},
            "selected_epoch": selected_epoch,
        },
        kendall_state={},
    )


def _e2e_training_cfg(tmp_path: Path, run_kind: te.E2ERunKind | None = None) -> te.EgoConfig:
    """A config carrying the registered ``training`` section, without a pack.

    `load_config` refuses an ``egostitch_e2e`` family without it, and every
    run-kind decision in the worker is gated on ``cfg.training is not None``.
    """
    cfg = _toy_cfg(tmp_path)
    return replace(
        cfg,
        model=replace(cfg.model, family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
        training=te.EgoStitchTrainingConfig(),
        run_kind=run_kind,
    )


def _write_historical_pass_artifact(
    cfg: te.EgoConfig, *, feature_stats_sha256: str
) -> Path:
    """Create a legacy approved artifact without using the current producer."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / te.QUALIFICATION_FILENAME
    path.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "feature_stats_sha256": feature_stats_sha256,
                "model_config_sha256": te.model_config_hash(cfg),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- DDP contract


def test_ddp_accelerator_detects_conditionally_unused_parameters() -> None:
    handler = te._egostitch_ddp_kwargs()
    assert handler.broadcast_buffers is False
    assert handler.find_unused_parameters is True
    assert handler.gradient_as_bucket_view is True
    assert te._egostitch_ddp_kwargs(find_unused_parameters=False).find_unused_parameters is False


@pytest.mark.parametrize(
    ("n_val", "expected"),
    [(0, ()), (1, (0,)), (99, (0,)), (100, (0,)), (101, (0, 1)), (250, (0, 1, 2))],
)
def test_e2e_validation_slice_uses_fixed_global_manifest_rows(
    n_val: int, expected: tuple[int, ...]
) -> None:
    assert te._e2e_validation_slice_rows(n_val) == expected


# --------------------------------------------------------------------------- e2e config


class TestE2EConfigRejection:
    def test_rejects_e2e_without_training_contract(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        del mapping["training"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="legacy egostitch_e2e v1 execution"):
            te.load_config(path)

    def test_egostitch_e2e_family_rejects_stage1_only_keys(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["model"] = {"family": "egostitch_e2e", "config": {"slots": 4}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="unknown E2E config keys"):
            te.load_config(path)

    def test_e2e_permanent_null_accepts_only_registered_values(self) -> None:
        for value in ("none", "all_head"):
            cfg = E2EConfig.from_mapping({"classifier": {"permanent_null": value}})
            assert cfg.classifier.permanent_null == value
        with pytest.raises(ValueError, match="permanent_null"):
            E2EConfig.from_mapping({"classifier": {"permanent_null": "content_head"}})
        with pytest.raises(ValueError, match="permanent_null"):
            E2EConfig.from_mapping({"classifier": {"permanent_null": "topo_head"}})


# --------------------------------------------------------------------------- ddp run config


class TestDdpRunConfigPreparation:
    def test_bounded_e2e_worker_run_is_forbidden(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--max-steps forbidden"):
            te.prepare_ddp_run_config(_toy_cfg(tmp_path), max_steps=1)


# --------------------------------------------------------------------------- run kinds


class TestRunKindDomain:
    def test_run_kind_domain_includes_explicit_diagnostic_and_derived_debug(self) -> None:
        assert get_args(te.E2ERunKind) == ("formal", "diagnostic", "debug")
        assert te.parse_args(["--config", "c.yaml", "--run-kind", "formal"]).run_kind == (
            "formal"
        )
        assert te.parse_args(["--config", "c.yaml", "--run-kind", "diagnostic"]).run_kind == (
            "diagnostic"
        )
        with pytest.raises(SystemExit):
            te.parse_args(["--config", "c.yaml", "--run-kind", "qualification"])

    @pytest.mark.parametrize("flag", ["--qualification-artifact", "--epochs"])
    def test_worker_cli_rejects_retired_qualification_flags(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            te.parse_args(["--config", "c.yaml", flag, "value"])

    def test_feature_binding_has_no_qualification_artifact_input(self) -> None:
        assert "qualification_artifact" not in inspect.signature(
            te._bind_feature_standardization
        ).parameters

    def test_qualification_artifact_producers_are_removed(self) -> None:
        assert not hasattr(te, "write_qualification_artifact")
        assert not hasattr(te, "validate_qualification_artifact")
        assert not hasattr(te, "qualification_verdict")






    def test_run_kind_is_not_part_of_the_scientific_config(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        assert "run_kind" not in te.config_to_dict(cfg)
        assert te.config_to_dict(replace(cfg, run_kind="debug")) == te.config_to_dict(
            replace(cfg, run_kind="formal")
        )


class TestFeatureDigestPinByRunKind:
    """Feature-stat identity stays pinned independently of quality telemetry."""

    def _model_and_data(self, tmp_path: Path) -> tuple[EgoStitchModel, te.EgoStitchData]:
        config = _e2e_model_config(generator={"feature_standardization": "zscore_vfit_v1"})
        model = EgoStitchModel(E2EConfig.from_mapping(config))
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        rows = np.arange(4 * model.generator_cfg.input_dim, dtype=np.float32).reshape(4, -1)
        data.feature_stats = compute_feature_stats(rows, ["a", "b", "c", "d"])
        return model, data

    def _cfg(self, tmp_path: Path, run_kind: te.E2ERunKind | None) -> te.EgoConfig:
        cfg = _e2e_training_cfg(tmp_path, run_kind=run_kind)
        return replace(
            cfg,
            model=replace(
                cfg.model,
                config=_e2e_model_config(generator={"feature_standardization": "zscore_vfit_v1"}),
            ),
        )






    def test_debug_is_exempt_from_the_stage_contract(self, tmp_path: Path) -> None:
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None
        digest = te._bind_feature_standardization(model, self._cfg(tmp_path, "debug"), data)
        assert digest == data.feature_stats.digest

    def test_a_config_pin_is_still_equality_checked(self, tmp_path: Path) -> None:
        """A config that still carries the field must agree, in every kind."""
        model, data = self._model_and_data(tmp_path)
        cfg = self._cfg(tmp_path, "formal")
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                config=_e2e_model_config(
                    cfg.model.config, generator={"feature_stats_sha256": "ab" * 32}
                ),
            ),
        )
        with pytest.raises(RuntimeError, match="feature_stats_sha256 mismatch"):
            te._bind_feature_standardization(model, cfg, data)


class TestModelConfigHash:
    """Digest is invariant to execution mode but sensitive to model definition."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda cfg: replace(cfg, output_dir=Path("/elsewhere")), id="output_dir"),
            pytest.param(
                lambda cfg: replace(cfg, data=replace(cfg.data, root=Path("/elsewhere"))),
                id="data_root",
            ),
            pytest.param(
                lambda cfg: replace(cfg, optim=replace(cfg.optim, epochs=30)), id="epochs"
            ),
            pytest.param(lambda cfg: replace(cfg, seed=7), id="seed"),
            pytest.param(
                lambda cfg: replace(
                    cfg,
                    model=replace(
                        cfg.model,
                        config=_e2e_model_config(
                            cfg.model.config, generator={"feature_stats_sha256": "ab" * 32}
                        ),
                    ),
                ),
                id="feature_stats_sha256",
            ),
        ],
    )
    def test_invariant_keys(self, tmp_path: Path, mutate: _Mutator) -> None:
        cfg = _toy_cfg(tmp_path)
        assert te.model_config_hash(mutate(cfg)) == te.model_config_hash(cfg)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda cfg: replace(
                    cfg, model=replace(cfg.model, config={**cfg.model.config, "slots": 8})
                ),
                id="model_config",
            ),
            pytest.param(
                lambda cfg: replace(cfg, data=replace(cfg.data, negative_ratio=7)),
                id="negative_ratio",
            ),
            pytest.param(lambda cfg: replace(cfg, optim=replace(cfg.optim, lr=0.5)), id="lr"),
            pytest.param(lambda cfg: replace(cfg, mixed_precision="bf16"), id="mixed_precision"),
            pytest.param(
                lambda cfg: replace(cfg, training=None), id="training"
            ),
        ],
    )
    def test_model_defining_keys_change_the_digest(self, tmp_path: Path, mutate: _Mutator) -> None:
        cfg = _toy_cfg(tmp_path)
        assert te.model_config_hash(mutate(cfg)) != te.model_config_hash(cfg)

    def test_effective_runtime_node_batch_changes_the_digest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = te.load_config(root / "configs/egostitch_e2e_v3_full_breadth_first.yaml")
        assert cfg.runtime is not None
        changed = replace(cfg, runtime=replace(cfg.runtime, token_budget=64))
        assert te.model_config_hash(changed) != te.model_config_hash(cfg)

    def test_oracle_truth_source_changes_the_digest(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        fit_only = replace(
            cfg,
            model=replace(
                cfg.model,
                config=_e2e_model_config(
                    cfg.model.config,
                    generator={"name": "oracle_struct", "oracle_truth_source": "g_fit"},
                ),
            ),
        )
        with_hold_truth = replace(
            fit_only,
            model=replace(
                fit_only.model,
                config=_e2e_model_config(
                    fit_only.model.config,
                    generator={"oracle_truth_source": "g_fit_plus_v_hold"},
                ),
            ),
        )
        assert te.model_config_hash(with_hold_truth) != te.model_config_hash(fit_only)

    def test_every_diagnostic_field_changes_the_digest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = te.load_config(root / "configs/egostitch_e2e_v3_full_breadth_first.yaml")
        for field_name in te.EgoDiagnosticsConfig.__dataclass_fields__:
            value = getattr(cfg.diagnostics, field_name)
            changed_value = value + 1 if isinstance(value, int) else value + 0.001
            changed = replace(
                cfg,
                diagnostics=replace(cfg.diagnostics, **{field_name: changed_value}),
            )
            assert te.model_config_hash(changed) != te.model_config_hash(cfg), field_name


# --------------------------------------------------------------------------- step plan


class TestEpochStepPlan:
    def test_rows_and_steps(self) -> None:
        rows, steps = te._epoch_step_plan(10, negative_ratio=2, edge_batch=6, world_size=4)
        assert rows == [9, 9, 6, 6]  # ceil splits of 10 positives x (1 + 2)
        assert steps == 2

    def test_global_count_tail(self) -> None:
        rows = [9, 9, 6, 6]
        assert te._step_global_count(rows, 0, 6) == 24
        assert te._step_global_count(rows, 1, 6) == 6
        assert te._step_global_count(rows, 2, 6) == 0


class TestRegisteredDiagnostics:
    def test_gradient_imbalance_activates_only_after_persistence(self) -> None:
        monitor = te._GradientImbalanceMonitor(ratio=10.0, required_steps=100, interval=50)
        norms = {"edge": 100.0, "recon": 1.0, "real": 1.0, "ssl": 1.0}
        assert monitor.update(50, norms) is False
        assert monitor.update(100, norms) is True
        assert monitor.activated_step == 100

    def test_probe_scale_guard_fails_on_synthetic_broken_kernel(self) -> None:
        with pytest.raises(RuntimeError, match="pathological Stage-1 membership scale"):
            te._enforce_probe_s1_scale(1.2e7, 1.0e3)


# --------------------------------------------------------------------------- run metadata


class TestRunStartMetadata:
    def test_records_the_bound_feature_stats_digest(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        digest = "ab" * 32
        te.write_run_start_metadata(cfg, data, world_size=1, feature_stats_sha256=digest)
        started = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert started["feature_stats_sha256"] == digest

    def test_defaults_the_feature_stats_digest_to_empty(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        started = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert started["feature_stats_sha256"] == ""
        assert started["status"] == "started"
        assert started["seed"] == cfg.seed
        assert started["world_size"] == 1
        assert started["rho_train"] == data.rho_train

    def test_diagnostic_completion_cannot_masquerade_as_formal(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind="diagnostic")
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        te.write_outputs(_stub_result(), cfg, data)
        metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert metadata["run_kind"] == "diagnostic"
        assert metadata["checkpoint_role"] == "diagnostic_only"
        assert metadata["formal_artifacts_published"] is False


# --------------------------------------------------------------------------- oracle truth boundary


class TestOracleTruthGraph:
    @staticmethod
    def _data(tmp_path: Path) -> te.EgoStitchData:
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        data.target_builder.graph.remove_nodes_from(("n6", "n7"))
        data.validation_nodes = ("n6", "n7")
        data.validation_positive_edges = (("n6", "n7"),)
        return data

    def test_true_oracle_adds_nonempty_v_hold_and_records_provenance(self, tmp_path: Path) -> None:
        data = self._data(tmp_path)
        graph = te._oracle_truth_graph(
            data,
            truth_source="g_fit_plus_v_hold",
            run_kind="diagnostic",
        )
        assert graph.has_edge("n6", "n7")
        assert data.target_builder.graph.has_edge("n6", "n7") is False
        assert data.access_audit is not None
        audit = data.access_audit["oracle_truth"]
        assert audit["diagnostic_only"] is True
        assert audit["v_hold_node_count"] == 2
        assert audit["v_hold_positive_edge_count"] == 1
        assert len(audit["v_hold_positive_edges_sha256"]) == 64

    def test_g_fit_source_is_the_untouched_training_graph(self, tmp_path: Path) -> None:
        data = self._data(tmp_path)
        graph = te._oracle_truth_graph(data, truth_source="g_fit", run_kind="formal")
        assert graph is data.target_builder.graph
        assert data.access_audit is None or "oracle_truth" not in data.access_audit

    def test_installed_true_oracle_table_has_a_nonempty_v_hold_row(self, tmp_path: Path) -> None:
        data = self._data(tmp_path)
        model = EgoStitchModel(
            E2EConfig.from_mapping(
                {
                    "generator": {
                        "name": "oracle_struct",
                        "oracle_truth_source": "g_fit_plus_v_hold",
                    }
                }
            )
        )
        te._install_oracle_context(model, data, run_kind="diagnostic")
        table = model.generator._table
        assert table is not None
        hold_row = data.node_index["n6"]
        assert int((table.neighbor_row[hold_row] >= 0).sum().item()) == 1

        # ...and the queried partner is still masked out of the stitched scaffold.
        other_row = data.node_index["n7"]
        ground = torch.zeros(1, 1, data.f0.shape[1])
        state_a = model.generator.encode_node(
            data.f0[hold_row : hold_row + 1],
            ground,
            node_rows=torch.tensor([hold_row]),
        )
        state_b = model.generator.encode_node(
            data.f0[other_row : other_row + 1],
            ground,
            node_rows=torch.tensor([other_row]),
        )
        stitched = model.generator.stitch(state_a, state_b, torch.zeros(1, dtype=torch.bool))
        slots = model.generator_cfg.slots
        assert torch.count_nonzero(stitched.x[0, 2 : 2 + slots, 4]).item() == 0
        assert torch.count_nonzero(stitched.x[0, 2 + slots : 2 + 2 * slots, 4]).item() == 0

    def test_true_oracle_refuses_formal_execution(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="requires --run-kind diagnostic"):
            te._oracle_truth_graph(
                self._data(tmp_path),
                truth_source="g_fit_plus_v_hold",
                run_kind="formal",
            )

    def test_true_oracle_refuses_fit_hold_node_overlap(self, tmp_path: Path) -> None:
        data = self._data(tmp_path)
        data.target_builder.graph.add_node("n6")
        with pytest.raises(RuntimeError, match="node-disjoint"):
            te._oracle_truth_graph(
                data,
                truth_source="g_fit_plus_v_hold",
                run_kind="diagnostic",
            )


# --------------------------------------------------------------------------- batch factory
