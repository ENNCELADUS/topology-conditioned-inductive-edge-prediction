"""Live-core tests for `src.train_egostitch` (the two-stage e2e worker).

Split out of `tests/test_train_egostitch.py` by the 2026-07-29 two-stage
cleanup: that module's frozen-s0 ``egostitch`` coverage was retired with the
code, while everything here exercises surface the qualification and the formal
stage both run. The shared toy fixtures stay in `tests/test_train_egostitch.py`
because three test modules import them.

Covered: the DDP handler contract, the fixed validation slice, the e2e config
rejection paths, the BINDING gate, the ``{qualification, formal, debug}`` run
kind domain, the ``qualification.json`` writer/validator, `model_config_hash`,
the epoch step plan, the registered gradient diagnostics, the run-start
metadata contract, and `_BatchFactory` determinism.
"""

from __future__ import annotations

import hashlib
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
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E

from tests.test_train_egostitch import (
    _E2E_TINY_MODEL,
    _config_mapping,
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
        for value in ("none", "all_head", "content_head"):
            assert E2EConfig.from_mapping({"permanent_null": value}).permanent_null == value
        with pytest.raises(ValueError, match="permanent_null"):
            E2EConfig.from_mapping({"permanent_null": "topo_head"})


# --------------------------------------------------------------------------- BINDING gate


class TestRegistrationRunMode:
    def test_formal_e2e_worker_refuses_registration_without_binding_status(
        self, tmp_path: Path
    ) -> None:
        cfg = _toy_cfg(tmp_path)
        e2e_cfg = replace(cfg, model=replace(cfg.model, family="egostitch_e2e"))
        with pytest.raises(te.PreregistrationNotBinding, match="status == 'BINDING'"):
            te.prepare_ddp_run_config(e2e_cfg, max_steps=None)

    def test_formal_e2e_worker_refuses_unresolved_binding_marker(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        cfg.preregistration.write_text(
            json.dumps(
                {
                    "status": "BINDING",
                    "frozen_inputs": {"b0cal_results": {"sha256": "REQUIRED-BEFORE-BINDING"}},
                }
            )
        )
        e2e_cfg = replace(cfg, model=replace(cfg.model, family="egostitch_e2e"))

        with pytest.raises(te.PreregistrationNotBinding, match="marker to be resolved"):
            te.prepare_ddp_run_config(e2e_cfg, max_steps=None)

    def test_formal_e2e_worker_accepts_resolved_binding_registration(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        cfg.preregistration.write_text(
            json.dumps(
                {
                    "status": "BINDING",
                    "frozen_inputs": {"b0cal_results": {"sha256": "a" * 64}},
                    "cost_report": {"measured": {"profile_sha256": "b" * 64}},
                }
            )
        )
        e2e_cfg = replace(cfg, model=replace(cfg.model, family="egostitch_e2e"))

        prepared, is_debug, _ = te.prepare_ddp_run_config(e2e_cfg, max_steps=None)
        assert prepared == e2e_cfg
        assert is_debug is False

    def test_qualification_stage_does_not_require_a_binding_registration(
        self, tmp_path: Path
    ) -> None:
        # Design 2026-07-29 Sec 2.3: the qualification stage is guards-only and
        # publishes no formal artifact, so it is exempt from the BINDING gate
        # the same registration would fail for the formal stage below.
        qualification = _e2e_training_cfg(tmp_path, run_kind="qualification")
        prepared, is_debug, _ = te.prepare_ddp_run_config(qualification, max_steps=None)
        assert prepared == qualification
        assert is_debug is False

        with pytest.raises(te.PreregistrationNotBinding, match="status == 'BINDING'"):
            te.prepare_ddp_run_config(replace(qualification, run_kind="formal"), max_steps=None)

    def test_direct_worker_refuses_post_binding_qualification(self, tmp_path: Path) -> None:
        qualification = _e2e_training_cfg(tmp_path, run_kind="qualification")
        qualification.preregistration.write_text(json.dumps({"status": "BINDING"}))
        args = te.parse_args(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--ddp-mode",
                "train",
                "--run-kind",
                "qualification",
                "--pack-dir",
                str(tmp_path / "pack"),
                "--output-dir",
                str(tmp_path / "output"),
                "--token-budget-per-rank",
                "4",
                "--profile-output",
                str(tmp_path / "profile.json"),
            ]
        )

        with pytest.raises(te.PreregistrationNotBinding, match="registered K disclosure"):
            te._run_ddp_worker(qualification, args)

    @pytest.mark.parametrize("run_kind", ["qualification", "formal", None])
    def test_both_e2e_stages_forbid_max_steps(
        self, tmp_path: Path, run_kind: te.E2ERunKind | None
    ) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind=run_kind)
        with pytest.raises(ValueError, match="must execute the complete schedule"):
            te.prepare_ddp_run_config(cfg, max_steps=5)

    def test_active_e2e_config_uses_current_training_contract(self) -> None:
        """The active arm is v3; the v2 configs are byte-frozen BINDING evidence.

        `optimizer_groups` is registered once, in the v2 registration the v3
        draft names as its predecessor, so the group clip norms are resolved
        through that link — and the link's own digest is checked, because an
        unverified pointer would let the contract be re-pinned silently.
        """
        root = Path(__file__).resolve().parents[1]
        cfg = te.load_config(root / "configs/egostitch_e2e_v3_full_breadth_first.yaml")
        assert cfg.model.family == "egostitch_e2e"
        assert cfg.training == te.EgoStitchTrainingConfig()
        assert cfg.training.pair_encoder_clip_norm == 3.0
        assert cfg.training.generator_clip_norm == 3.0
        assert cfg.training.clip_norm == 1.0
        registration = json.loads(cfg.preregistration.read_text(encoding="utf-8"))
        predecessor = registration["predecessor"]
        assert predecessor["status"] == "BINDING"
        predecessor_path = root / predecessor["path"]
        assert hashlib.sha256(predecessor_path.read_bytes()).hexdigest() == predecessor["sha256"]
        bound = json.loads(predecessor_path.read_text(encoding="utf-8"))
        registered_groups = bound["training_contract"]["optimizer_groups"]
        assert registered_groups["pair_encoder_head"]["grad_clip_l2"] == 3.0
        assert registered_groups["generator"]["grad_clip_l2"] == 3.0
        assert registered_groups["topology_content_conditioning"]["grad_clip_l2"] == 1.0

    def test_bounded_e2e_worker_run_is_forbidden(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--max-steps forbidden"):
            te.prepare_ddp_run_config(_toy_cfg(tmp_path), max_steps=1)


# --------------------------------------------------------------------------- run kinds


class TestRunKindDomain:
    def test_literal_domain_is_exactly_the_three_registered_kinds(self) -> None:
        # `debug` is a real published value (`run_metadata.json`, read back by
        # `src.experiments.probes`) and must stay in the Literal; the retired
        # `overfit`/`rehearsal`/`calibration` kinds must not.
        assert get_args(te.E2ERunKind) == ("qualification", "formal", "debug")

    @pytest.mark.parametrize("run_kind", ["qualification", "formal"])
    def test_cli_offers_the_two_selectable_stages(self, run_kind: str) -> None:
        args = te.parse_args(["--config", "c.yaml", "--run-kind", run_kind])
        assert args.run_kind == run_kind

    @pytest.mark.parametrize("run_kind", ["overfit", "rehearsal", "calibration", "debug"])
    def test_cli_refuses_unselectable_run_kinds(self, run_kind: str) -> None:
        # `debug` is derived from `--max-steps`, never selected: offering it
        # would hand out the `_bind_feature_standardization` digest-pin
        # exemption without running a bounded, non-publishing schedule.
        with pytest.raises(SystemExit):
            te.parse_args(["--config", "c.yaml", "--run-kind", run_kind])

    def test_run_kind_override_requires_a_training_section(self, tmp_path: Path) -> None:
        args = te.parse_args(["--config", "c.yaml", "--run-kind", "qualification"])
        with pytest.raises(ValueError, match="requires a training config section"):
            te.apply_overrides(replace(_toy_cfg(tmp_path), training=None), args)

    def test_run_kind_override_binds_and_defaults_to_unset(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        bound = te.apply_overrides(
            cfg, te.parse_args(["--config", "c.yaml", "--run-kind", "qualification"])
        )
        assert bound.run_kind == "qualification"
        assert te.apply_overrides(cfg, te.parse_args(["--config", "c.yaml"])).run_kind is None

    def test_run_kind_is_not_part_of_the_scientific_config(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        assert "run_kind" not in te.config_to_dict(cfg)
        assert te._config_hash(replace(cfg, run_kind="qualification")) == te._config_hash(
            replace(cfg, run_kind="formal")
        )


class TestQualificationEpochOverride:
    """``--epochs``: the one value the two stages may differ in (design Sec 2).

    It has to land in `cfg.optim.epochs` rather than only in the training
    loop's schedule, because everything downstream keys off that single field
    -- the `metrics.jsonl` row count, the staged-artifact epoch validation,
    and the ``epochs`` recorded in ``qualification.json``.
    """

    def _args(self, *extra: str) -> te.EgoCliArgs:
        return te.parse_args(["--config", "c.yaml", *extra])

    def test_the_override_lands_in_the_config_epoch_count(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        assert cfg.optim.epochs != 3  # precondition: the override really moves it

        bound = te.apply_overrides(cfg, self._args("--run-kind", "qualification", "--epochs", "3"))

        assert bound.optim.epochs == 3
        assert bound.run_kind == "qualification"

    def test_the_recorded_verdict_carries_the_overridden_epoch_count(self, tmp_path: Path) -> None:
        cfg = te.apply_overrides(
            _e2e_training_cfg(tmp_path), self._args("--run-kind", "qualification", "--epochs", "3")
        )
        path = te.write_qualification_artifact(
            cfg, verdict="pending_manual_review", feature_stats_sha256="ab" * 32
        )
        assert json.loads(path.read_text())["epochs"] == 3

    def test_the_formal_stage_refuses_the_override(self, tmp_path: Path) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        for extra in (("--run-kind", "formal", "--epochs", "3"), ("--epochs", "3")):
            with pytest.raises(ValueError, match="qualification-stage override"):
                te.apply_overrides(cfg, self._args(*extra))

    def test_the_override_requires_a_training_section(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires a training config section"):
            te.apply_overrides(
                replace(_toy_cfg(tmp_path), training=None), self._args("--epochs", "3")
            )

    @pytest.mark.parametrize("epochs", ["0", "-1"])
    def test_a_non_positive_override_is_refused(self, tmp_path: Path, epochs: str) -> None:
        cfg = _e2e_training_cfg(tmp_path)
        with pytest.raises(ValueError, match="--epochs must be positive"):
            te.apply_overrides(cfg, self._args("--run-kind", "qualification", "--epochs", epochs))

    def test_the_override_does_not_move_the_model_config_digest(self, tmp_path: Path) -> None:
        """`model_config_hash` excludes `optim.epochs` -- it is what the stages differ in."""
        cfg = _e2e_training_cfg(tmp_path, run_kind="qualification")
        short = te.apply_overrides(cfg, self._args("--epochs", "3"))
        assert te.model_config_hash(short) == te.model_config_hash(cfg)

    def test_the_override_defaults_to_unset(self) -> None:
        assert self._args().epochs is None

    def test_qualification_artifacts_are_never_marked_eligible(self, tmp_path: Path) -> None:
        # The pre-cleanup form was `expected_run_kind != "overfit"`, which
        # becomes unconditionally true once `overfit` leaves the domain and
        # would publish a short-schedule checkpoint as selectable (design
        # 2026-07-29 Sec 6.1). Gating must be a positive `== "formal"`.
        cfg = replace(_toy_cfg(tmp_path), output_dir=tmp_path / "qual", run_kind="qualification")
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        te.write_outputs(_stub_result(), cfg, data)
        metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert metadata["run_kind"] == "qualification"
        assert metadata["checkpoint_eligible"] is False
        assert metadata["selected_checkpoint_eligible"] is False
        assert metadata["formal_artifacts_published"] is False

    def test_formal_artifacts_stay_eligible(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        te.write_outputs(_stub_result(), cfg, data)
        metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert metadata["run_kind"] == "formal"
        assert metadata["checkpoint_eligible"] is True
        assert metadata["selected_checkpoint_eligible"] is True
        assert metadata["formal_artifacts_published"] is True

    def test_formal_checkpoint_without_a_selected_epoch_is_not_eligible(
        self, tmp_path: Path
    ) -> None:
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        te.write_outputs(_stub_result(selected_epoch=None), cfg, data)
        metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert metadata["checkpoint_eligible"] is False

    def test_run_kind_may_not_change_between_start_and_finalize(self, tmp_path: Path) -> None:
        cfg = replace(_toy_cfg(tmp_path), run_kind="qualification")
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        te.write_run_start_metadata(cfg, data, world_size=1)
        with pytest.raises(RuntimeError, match="run kind changed after run start"):
            te.write_outputs(_stub_result(), replace(cfg, run_kind="formal"), data)


class TestFeatureDigestPinByRunKind:
    """`_bind_feature_standardization`'s two-stage digest contract (design Sec 3).

    The qualification stage *computes* the digest -- it is not a config input,
    because the six v3 configs no longer carry `feature_stats_sha256` and the
    preflight stage that used to supply it is gone. The formal stage *compares*
    its own digest against the one the qualification stage recorded. Neither
    branch may fail open, so both directions are asserted here: the
    qualification stage must not demand an input, and the formal stage must
    refuse without one.

    The pre-cleanup form was `effective_run_kind in ("rehearsal", "formal")`,
    which would have left the added `qualification` kind silently unguarded;
    the replacement is written as an exhaustive match so an unrecognized kind
    lands on the strict (formal) branch.
    """

    def _model_and_data(self, tmp_path: Path) -> tuple[EgoStitchE2E, te.EgoStitchData]:
        config = {**_E2E_TINY_MODEL, "feature_standardization": "zscore_vfit_v1"}
        model = EgoStitchE2E(E2EConfig.from_mapping(config))
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
                config={**_E2E_TINY_MODEL, "feature_standardization": "zscore_vfit_v1"},
            ),
        )

    def test_the_qualification_stage_computes_the_digest_without_an_input(
        self, tmp_path: Path
    ) -> None:
        """Requiring a pin here is the circularity the two-stage design removed."""
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None

        digest = te._bind_feature_standardization(model, self._cfg(tmp_path, "qualification"), data)

        assert digest == data.feature_stats.digest
        assert model.feature_stats_digest_hex == digest

    @pytest.mark.parametrize("run_kind", ["formal", None])
    def test_the_formal_stage_refuses_without_a_qualification_artifact(
        self, tmp_path: Path, run_kind: te.E2ERunKind | None
    ) -> None:
        """An unset run kind normalizes to formal, so it must refuse too."""
        model, data = self._model_and_data(tmp_path)
        with pytest.raises(RuntimeError, match="feature_stats_sha256 is unpinned"):
            te._bind_feature_standardization(model, self._cfg(tmp_path, run_kind), data)
        assert model.feature_stats_digest_hex == ""

    def test_the_formal_stage_equality_checks_the_recorded_digest(self, tmp_path: Path) -> None:
        model, data = self._model_and_data(tmp_path)
        qualification = self._cfg(tmp_path, "qualification")
        artifact = _write_historical_pass_artifact(
            qualification, feature_stats_sha256="ab" * 32
        )

        with pytest.raises(RuntimeError, match="feature_stats_sha256 does not match"):
            te._bind_feature_standardization(
                model,
                self._cfg(tmp_path, "formal"),
                data,
                qualification_artifact=artifact,
            )
        assert model.feature_stats_digest_hex == ""

    def test_the_formal_stage_accepts_the_recorded_digest(self, tmp_path: Path) -> None:
        """Both stages train on the identical V_fit, so this is a real equality."""
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None
        qualification = self._cfg(tmp_path, "qualification")
        artifact = _write_historical_pass_artifact(
            qualification, feature_stats_sha256=data.feature_stats.digest
        )

        digest = te._bind_feature_standardization(
            model,
            self._cfg(tmp_path, "formal"),
            data,
            qualification_artifact=artifact,
        )

        assert digest == data.feature_stats.digest
        assert model.feature_stats_digest_hex == digest

    def test_the_formal_stage_refuses_a_failed_qualification(self, tmp_path: Path) -> None:
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None
        qualification = self._cfg(tmp_path, "qualification")
        artifact = te.write_qualification_artifact(
            qualification,
            verdict="fail(slot_collapse)",
            feature_stats_sha256=data.feature_stats.digest,
        )

        with pytest.raises(RuntimeError, match="verdict is not 'pass'"):
            te._bind_feature_standardization(
                model,
                self._cfg(tmp_path, "formal"),
                data,
                qualification_artifact=artifact,
            )
        assert model.feature_stats_digest_hex == ""

    def test_debug_is_exempt_from_the_stage_contract(self, tmp_path: Path) -> None:
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None
        digest = te._bind_feature_standardization(model, self._cfg(tmp_path, "debug"), data)
        assert digest == data.feature_stats.digest

    @pytest.mark.parametrize("ddp_mode", te._PROBE_DISPATCH_MODES)
    def test_measurement_dispatch_is_exempt_from_the_formal_digest_input(
        self, tmp_path: Path, ddp_mode: str
    ) -> None:
        model, data = self._model_and_data(tmp_path)
        assert data.feature_stats is not None
        digest = te._bind_feature_standardization(
            model,
            self._cfg(tmp_path, "formal"),
            data,
            ddp_mode=ddp_mode,
        )
        assert digest == data.feature_stats.digest

    def test_a_config_pin_is_still_equality_checked(self, tmp_path: Path) -> None:
        """A config that still carries the field must agree, in every kind."""
        model, data = self._model_and_data(tmp_path)
        cfg = self._cfg(tmp_path, "qualification")
        cfg = replace(
            cfg,
            model=replace(
                cfg.model, config={**cfg.model.config, "feature_stats_sha256": "ab" * 32}
            ),
        )
        with pytest.raises(RuntimeError, match="feature_stats_sha256 mismatch"):
            te._bind_feature_standardization(model, cfg, data)


# --------------------------------------------------------------------------- qualification


class TestQualificationVerdict:
    def test_a_completed_run_requires_manual_review(self) -> None:
        assert te.qualification_verdict(None) == "pending_manual_review"

    @pytest.mark.parametrize(
        ("message", "verdict"),
        [
            (
                "no eligible checkpoint; fallback is forbidden",
                "fail(no_eligible_checkpoint)",
            ),
            ("training_invalid(slot_collapse) at epoch 3", "training_invalid(slot_collapse)"),
            (
                "training_invalid(initial_slot_collapse): h-cos 0.99",
                "training_invalid(initial_slot_collapse)",
            ),
            (
                "E2E training opened a held-out path: /d/test_edges.txt",
                "fail(held_out_path)",
            ),
            ("non-finite E2E loss at step 7", "fail(nonfinite_loss)"),
        ],
    )
    def test_registered_guards_are_named(self, message: str, verdict: str) -> None:
        assert te.qualification_verdict(RuntimeError(message)) == verdict

    def test_an_unregistered_failure_is_still_named(self) -> None:
        # A hang or a bare traceback is not a valid Stage-1 result (design
        # 2026-07-29 Sec 8, acceptance item 1).
        assert te.qualification_verdict(ValueError("kaboom")) == "fail(unclassified_guard)"


class TestQualificationArtifact:
    def _cfg(self, tmp_path: Path) -> te.EgoConfig:
        return replace(_toy_cfg(tmp_path), output_dir=tmp_path / "qual", run_kind="qualification")

    def test_payload_contract(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = te.write_qualification_artifact(
            cfg, verdict="pending_manual_review", feature_stats_sha256="ab" * 32
        )
        assert path.name == te.QUALIFICATION_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {
            "verdict",
            "epochs",
            "hparams",
            "feature_stats_sha256",
            "model_config_sha256",
        }
        assert payload["verdict"] == "pending_manual_review"
        assert payload["epochs"] == cfg.optim.epochs
        assert payload["feature_stats_sha256"] == "ab" * 32
        assert payload["model_config_sha256"] == te.model_config_hash(cfg)
        assert payload["hparams"]["seed"] == cfg.seed
        assert payload["hparams"]["lr"] == cfg.optim.lr

    def test_named_failures_are_writable(self, tmp_path: Path) -> None:
        path = te.write_qualification_artifact(
            self._cfg(tmp_path),
            verdict="fail(no_eligible_checkpoint)",
            feature_stats_sha256="ab" * 32,
        )
        assert json.loads(path.read_text())["verdict"] == "fail(no_eligible_checkpoint)"

    def test_the_registered_pending_verdict_is_writable(self, tmp_path: Path) -> None:
        path = te.write_qualification_artifact(
            self._cfg(tmp_path),
            verdict="pending_manual_review",
            feature_stats_sha256="ab" * 32,
        )
        assert json.loads(path.read_text())["verdict"] == "pending_manual_review"

    def test_writer_refuses_to_produce_a_pass_verdict(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be 'pending_manual_review'"):
            te.write_qualification_artifact(
                self._cfg(tmp_path), verdict="pass", feature_stats_sha256="ab" * 32
            )

    @pytest.mark.parametrize(
        "verdict",
        ["fail", "boom", "failed(x)", "fail(x", "pending", "pending_review"],
    )
    def test_unnamed_verdicts_are_refused(self, tmp_path: Path, verdict: str) -> None:
        with pytest.raises(ValueError, match="must be 'pending_manual_review'"):
            te.write_qualification_artifact(
                self._cfg(tmp_path), verdict=verdict, feature_stats_sha256="ab" * 32
            )

    # --- the formal stage's three preflight assertions (acceptance item 2)

    def test_formal_stage_refuses_a_missing_artifact(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        with pytest.raises(RuntimeError, match="requires a qualification artifact"):
            te.validate_qualification_artifact(
                cfg.output_dir / te.QUALIFICATION_FILENAME,
                cfg,
                feature_stats_sha256="ab" * 32,
            )

    def test_formal_stage_refuses_a_failed_qualification(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = te.write_qualification_artifact(
            cfg, verdict="training_invalid(slot_collapse)", feature_stats_sha256="ab" * 32
        )
        with pytest.raises(RuntimeError, match="verdict is not 'pass'"):
            te.validate_qualification_artifact(path, cfg, feature_stats_sha256="ab" * 32)

    def test_formal_stage_refuses_a_pending_manual_review(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = te.write_qualification_artifact(
            cfg, verdict="pending_manual_review", feature_stats_sha256="ab" * 32
        )
        with pytest.raises(RuntimeError, match="verdict is not 'pass'"):
            te.validate_qualification_artifact(path, cfg, feature_stats_sha256="ab" * 32)

    def test_formal_stage_refuses_a_model_digest_mismatch(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = _write_historical_pass_artifact(cfg, feature_stats_sha256="ab" * 32)
        drifted = replace(cfg, model=replace(cfg.model, config={**cfg.model.config, "slots": 8}))
        with pytest.raises(RuntimeError, match="model_config_sha256 does not match"):
            te.validate_qualification_artifact(path, drifted, feature_stats_sha256="ab" * 32)

    def test_formal_stage_refuses_a_feature_digest_mismatch(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = _write_historical_pass_artifact(cfg, feature_stats_sha256="ab" * 32)
        with pytest.raises(RuntimeError, match="feature_stats_sha256 does not match"):
            te.validate_qualification_artifact(path, cfg, feature_stats_sha256="cd" * 32)

    def test_formal_stage_accepts_a_matching_artifact(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        path = _write_historical_pass_artifact(cfg, feature_stats_sha256="ab" * 32)
        # The two stages differ only in `optim.epochs` and `output_dir`, and
        # neither enters `model_config_hash` -- so this is a genuine equality
        # rather than a propagated digest (design 2026-07-29 Sec 3).
        formal = replace(
            cfg,
            output_dir=tmp_path / "formal",
            optim=replace(cfg.optim, epochs=30),
            run_kind="formal",
        )
        payload = te.validate_qualification_artifact(path, formal, feature_stats_sha256="ab" * 32)
        assert payload["verdict"] == "pass"


class _StubAccelerator:
    """The rank-0 surface `_run_ddp_worker` touches before the training loop."""

    process_index = 0
    num_processes = 1
    device = "cpu"
    is_main_process = True

    def wait_for_everyone(self) -> None:
        return None


class TestMeasurementDispatch:
    @pytest.mark.parametrize("ddp_mode", te._PROBE_DISPATCH_MODES)
    def test_writes_only_the_feature_statistics_measurement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ddp_mode: str
    ) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind="formal")
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        model = EgoStitchE2E(E2EConfig.from_mapping(_E2E_TINY_MODEL))
        profile = tmp_path / f"{ddp_mode}.json"
        args = te.EgoCliArgs(
            config=tmp_path / "config.yaml",
            seed=None,
            output_dir=None,
            ddp_mode=ddp_mode,
            pack_dir=tmp_path / "pack",
            token_budget_per_rank=4,
            profile_output=profile,
        )
        monkeypatch.setattr(
            te,
            "write_run_start_metadata",
            lambda *args, **kwargs: pytest.fail("measurement wrote run metadata"),
        )
        monkeypatch.setattr(
            te,
            "train_egostitch_ddp_loop",
            lambda *args, **kwargs: pytest.fail("measurement entered training"),
        )

        te._run_ddp_dispatch(
            cfg,
            args,
            model,
            data,
            accelerator=_StubAccelerator(),
            preregistration=te.PreregistrationSnapshot(
                payload={"status": "DRAFT"}, sha256="0" * 64
            ),
            registered_config_hash=None,
            formal_binding=None,
            run_kind="formal",
            feature_stats_sha256="ab" * 32,
            node_batch=4,
            profile_output=profile,
        )

        assert json.loads(profile.read_text(encoding="utf-8")) == {
            "mode": ddp_mode,
            "feature_stats_sha256": "ab" * 32,
            "feature_stats_rows": 0,
        }
        assert not (cfg.output_dir / "run_metadata.json").exists()

    def test_completed_qualification_writes_pending_manual_review(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind="qualification")
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        model = EgoStitchE2E(E2EConfig.from_mapping(_E2E_TINY_MODEL))
        args = te.EgoCliArgs(
            config=tmp_path / "config.yaml",
            seed=None,
            output_dir=None,
            ddp_mode="train",
            pack_dir=tmp_path / "pack",
            token_budget_per_rank=4,
            profile_output=tmp_path / "profile.json",
        )
        monkeypatch.setattr(te, "e2e_degree_prior_init", lambda *args: 0.0)
        monkeypatch.setattr(te, "write_run_start_metadata", lambda *args, **kwargs: None)
        monkeypatch.setattr(te, "train_egostitch_ddp_loop", lambda *args, **kwargs: _stub_result())
        monkeypatch.setattr(te, "write_outputs", lambda *args, **kwargs: None)

        te._run_ddp_dispatch(
            cfg,
            args,
            model,
            data,
            accelerator=_StubAccelerator(),
            preregistration=te.PreregistrationSnapshot(
                payload={"status": "DRAFT"}, sha256="0" * 64
            ),
            registered_config_hash=None,
            formal_binding=None,
            run_kind="qualification",
            feature_stats_sha256="ab" * 32,
            node_batch=4,
            profile_output=args.profile_output,
        )

        payload = json.loads(
            (cfg.output_dir / te.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["verdict"] == "pending_manual_review"


class TestQualificationVerdictCoversPreTrainingGuards:
    """A qualification run ends pending manual review or with a named failure.

    The guards that fire before the first optimizer step -- the held-out path
    boundary in data assembly and the digest contract in
    `_bind_feature_standardization` -- are the ones a fast development loop
    hits most, and both map onto registered verdicts. Wrapping only the
    training loop left them writing no `qualification.json` at all, which is
    indistinguishable to the formal stage's preflight from a run that never
    started.
    """

    def _args(self, tmp_path: Path) -> te.EgoCliArgs:
        return te.EgoCliArgs(
            config=tmp_path / "config.yaml",
            seed=None,
            output_dir=None,
            ddp_mode="train",
            pack_dir=tmp_path / "pack",
            token_budget_per_rank=4,
            profile_output=tmp_path / "profile.json",
            run_kind="qualification",
        )

    def _install_stubs(self, monkeypatch: pytest.MonkeyPatch, *, error: Exception) -> None:
        monkeypatch.setattr(
            te, "build_egostitch_ddp_accelerator", lambda *a, **k: _StubAccelerator()
        )

        def refuse(cfg: te.EgoConfig, *, pack_dir: Path | None = None) -> te.EgoStitchData:
            raise error

        monkeypatch.setattr(te, "assemble_egostitch_data", refuse)

    def test_a_held_out_trespass_before_training_is_recorded_and_re_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind="qualification")
        self._install_stubs(
            monkeypatch,
            error=RuntimeError("E2E training opened a held-out path: /d/test_edges.txt"),
        )

        with pytest.raises(RuntimeError, match="opened a held-out path"):
            te._run_ddp_worker(cfg, self._args(tmp_path))

        payload = json.loads(
            (cfg.output_dir / te.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["verdict"] == "fail(held_out_path)"
        # The run died before binding, so the digest field is empty rather
        # than absent -- still a readable verdict.
        assert payload["feature_stats_sha256"] == ""

    def test_an_unrecognized_pre_training_failure_is_still_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _e2e_training_cfg(tmp_path, run_kind="qualification")
        self._install_stubs(monkeypatch, error=RuntimeError("something nobody registered"))

        with pytest.raises(RuntimeError, match="nobody registered"):
            te._run_ddp_worker(cfg, self._args(tmp_path))

        payload = json.loads(
            (cfg.output_dir / te.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["verdict"] == "fail(unclassified_guard)"

    def test_the_formal_stage_writes_no_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`qualification.json` is a Stage-1 product; a formal run must not forge one.

        The registration gates are stubbed past here on purpose: a formal run
        in this fixture would otherwise refuse at `prepare_ddp_run_config`
        (status is not BINDING) and never reach the verdict writer at all,
        which is the branch under test.
        """
        cfg = _e2e_training_cfg(tmp_path, run_kind="formal")
        snapshot = te.PreregistrationSnapshot(payload={"status": "BINDING"}, sha256="0" * 64)
        monkeypatch.setattr(
            te, "prepare_ddp_run_config", lambda cfg, *, max_steps: (cfg, False, snapshot)
        )
        monkeypatch.setattr(
            te, "_validate_e2e_formal_binding", lambda cfg, snapshot, path: {"arm": "full"}
        )
        self._install_stubs(
            monkeypatch,
            error=RuntimeError("E2E training opened a held-out path: /d/test_edges.txt"),
        )
        args = replace(self._args(tmp_path), run_kind="formal")

        with pytest.raises(RuntimeError, match="opened a held-out path"):
            te._run_ddp_worker(cfg, args)

        assert not (cfg.output_dir / te.QUALIFICATION_FILENAME).exists()


class TestModelConfigHash:
    """Digest must be invariant to everything the two stages legitimately differ in."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda cfg: replace(cfg, output_dir=Path("/elsewhere")), id="output_dir"),
            pytest.param(
                lambda cfg: replace(cfg, data=replace(cfg.data, root=Path("/elsewhere"))),
                id="data_root",
            ),
            pytest.param(
                lambda cfg: replace(cfg, preregistration=Path("/elsewhere.json")), id="prereg"
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
                        config={**cfg.model.config, "feature_stats_sha256": "ab" * 32},
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
    def test_preregistration_snapshot_survives_later_file_replacement(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        snapshot = te._preregistration_snapshot(cfg.preregistration)
        cfg.preregistration.write_text('{"registration_id": "changed"}\n')
        te.write_run_start_metadata(cfg, data, world_size=1, preregistration_sha256=snapshot.sha256)
        te.write_outputs(_stub_result(), cfg, data)
        completed = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert completed["status"] == "complete"
        assert completed["preregistration_sha256"] == snapshot.sha256
        assert completed["preregistration_sha256"] != te._sha256_file(cfg.preregistration)

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


# --------------------------------------------------------------------------- batch factory
