"""Contracts for the executable pre-binding G3 gates (spec Sec 14.4.7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from src.experiments import prebinding_gates as pg

_MARKER = "REQUIRED-BEFORE-BINDING: freeze after GPU calibration"


def _registration(status: str = "DRAFT") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "grounding": {"n_ground": 50, "cosine_pool_arm_n_ground": 20},
        "prebinding_qualification": {
            "gates": [
                {
                    "id": "G3.1",
                    "name": "slot_recall_at_n_ground",
                    "operator": ">=",
                    "threshold": 0.0698,
                    "n_ground": 50,
                },
                {
                    "id": "G3.2",
                    "name": "Pi_consistency_v2",
                    "operator": ">",
                    "threshold": 0.05,
                },
                {
                    "id": "G3.3",
                    "name": "degree_partialled_clustering_probe_r2",
                    "operator": ">=",
                    "threshold": 0.1,
                },
                {
                    "id": "G3.4",
                    "name": "structure_control_6a_v3_clustering_mmd_movement",
                    "operator": ">",
                    "threshold": _MARKER,
                },
                {
                    "id": "G3.5",
                    "name": "matched_edge_auprc_guard",
                    "operator": "passes frozen guard",
                    "threshold": _MARKER,
                },
            ]
        },
    }
    return payload


def _evaluation(
    *, slot_recall: float, pi_v2: float, clustering_r2: float, n_ground: int = 50
) -> dict[str, Any]:
    return {
        "metadata": {
            "checkpoint_id": "abc123",
            "format": "egostitch_e2e_probe_v2",
            "n_ground": n_ground,
            "scope": "calibration_fit",
        },
        "slot_recall_at_n_ground": {"mean": slot_recall, "n_ground": n_ground},
        "pi_consistency_v2": {"mean": pi_v2},
        "degree_partialled_r2": {"clustering": clustering_r2, "ego_density": 0.0},
    }


def _report(*, slot_recall: float, pi_v2: float, clustering_r2: float) -> dict[str, Any]:
    return pg.build_gate_report(
        _registration(),
        _evaluation(slot_recall=slot_recall, pi_v2=pi_v2, clustering_r2=clustering_r2),
        registration_sha="f" * 64,
        scope="calibration_fit",
    )


def _row(report: dict[str, Any], gate_id: str) -> dict[str, Any]:
    rows = cast(list[dict[str, Any]], report["gates"])
    return next(row for row in rows if row["id"] == gate_id)


class TestProbeComputableGates:
    def test_evaluates_the_three_probe_gates_against_registered_thresholds(self) -> None:
        report = _report(slot_recall=0.12, pi_v2=0.20, clustering_r2=0.30)
        for gate_id, value in (("G3.1", 0.12), ("G3.2", 0.20), ("G3.3", 0.30)):
            row = _row(report, gate_id)
            assert row["status"] == "evaluated"
            assert row["value"] == pytest.approx(value)
            assert row["passed"] is True
        assert report["n_evaluated"] == 3

    def test_marks_each_gate_failed_below_its_threshold(self) -> None:
        report = _report(slot_recall=0.05, pi_v2=0.01, clustering_r2=0.02)
        for gate_id in ("G3.1", "G3.2", "G3.3"):
            assert _row(report, gate_id)["passed"] is False

    def test_strict_and_inclusive_operators_differ_at_the_boundary(self) -> None:
        # G3.2 is '>' 0.05 and G3.3 is '>=' 0.10: exactly-at-threshold must
        # fail the former and pass the latter.
        report = _report(slot_recall=0.0698, pi_v2=0.05, clustering_r2=0.1)
        assert _row(report, "G3.1")["passed"] is True
        assert _row(report, "G3.2")["passed"] is False
        assert _row(report, "G3.3")["passed"] is True

    def test_records_the_pinned_reducer_for_array_valued_gates(self) -> None:
        report = _report(slot_recall=0.12, pi_v2=0.20, clustering_r2=0.30)
        assert _row(report, "G3.1")["reducer"] == pg.GATE_REDUCER == "mean"
        assert _row(report, "G3.2")["reducer"] == "mean"
        # The clustering R2 is a bare float, not a reduced array.
        assert _row(report, "G3.3")["reducer"] is None


class TestUnreachableGates:
    def test_scoring_gates_are_never_reported_as_passed(self) -> None:
        report = _report(slot_recall=0.12, pi_v2=0.20, clustering_r2=0.30)
        for gate_id in ("G3.4", "G3.5"):
            row = _row(report, gate_id)
            assert row["status"] == "not_evaluable_prebinding"
            assert row["passed"] is None
            assert "DRAFT" in cast(str, row["reason"])

    def test_overall_result_is_incomplete_and_never_passes(self) -> None:
        report = _report(slot_recall=0.99, pi_v2=0.99, clustering_r2=0.99)
        assert report["complete"] is False
        assert report["passed"] is False
        assert report["n_incomplete"] == 2

    def test_unresolved_threshold_on_a_probe_gate_is_not_a_pass(self) -> None:
        registration = _registration()
        gates = registration["prebinding_qualification"]["gates"]
        gates[1]["threshold"] = _MARKER
        report = pg.build_gate_report(
            registration,
            _evaluation(slot_recall=0.12, pi_v2=0.20, clustering_r2=0.30),
            registration_sha="f" * 64,
            scope="calibration_fit",
        )
        row = _row(report, "G3.2")
        assert row["status"] == "unresolved_threshold"
        assert row["passed"] is None


class TestFailClosed:
    def test_unknown_gate_name_raises_rather_than_being_skipped(self) -> None:
        registration = _registration()
        registration["prebinding_qualification"]["gates"].append(
            {"id": "G3.6", "name": "some_future_gate", "operator": ">", "threshold": 1.0}
        )
        with pytest.raises(pg.GateEvaluationError, match="no evaluator"):
            pg.build_gate_report(
                registration,
                _evaluation(slot_recall=0.12, pi_v2=0.2, clustering_r2=0.3),
                registration_sha="f" * 64,
                scope="calibration_fit",
            )

    def test_unsupported_operator_raises(self) -> None:
        registration = _registration()
        registration["prebinding_qualification"]["gates"][0]["operator"] = "~="
        with pytest.raises(pg.GateEvaluationError, match="unsupported gate operator"):
            pg.build_gate_report(
                registration,
                _evaluation(slot_recall=0.12, pi_v2=0.2, clustering_r2=0.3),
                registration_sha="f" * 64,
                scope="calibration_fit",
            )

    def test_missing_probe_quantity_raises(self) -> None:
        evaluation = _evaluation(slot_recall=0.12, pi_v2=0.2, clustering_r2=0.3)
        del evaluation["degree_partialled_r2"]["clustering"]
        with pytest.raises(pg.GateEvaluationError, match="missing"):
            pg.build_gate_report(
                _registration(),
                evaluation,
                registration_sha="f" * 64,
                scope="calibration_fit",
            )

    def test_malformed_gate_block_raises(self) -> None:
        with pytest.raises(pg.GateEvaluationError, match="non-empty list"):
            pg.load_prebinding_gates({"prebinding_qualification": {"gates": []}})
        with pytest.raises(pg.GateEvaluationError, match="prebinding_qualification"):
            pg.load_prebinding_gates({})


class TestRegisteredNGround:
    def test_grounding_value_governs(self) -> None:
        registration = _registration()
        gates = pg.load_prebinding_gates(registration)
        assert pg._registered_n_ground(registration, gates) == 50

    def test_self_inconsistent_registration_is_rejected(self) -> None:
        registration = _registration()
        registration["prebinding_qualification"]["gates"][0]["n_ground"] = 20
        gates = pg.load_prebinding_gates(registration)
        with pytest.raises(pg.GateEvaluationError, match="self-inconsistent"):
            pg._registered_n_ground(registration, gates)

    def test_cosine_pool_value_never_leaks_into_full_arm_qualification(self) -> None:
        # grounding.cosine_pool_arm_n_ground is 20; qualification is full-arm
        # only, so the resolved pool size must stay 50.
        registration = _registration()
        gates = pg.load_prebinding_gates(registration)
        assert pg._registered_n_ground(registration, gates) != 20

    def test_missing_or_invalid_grounding_raises(self) -> None:
        with pytest.raises(pg.GateEvaluationError, match="grounding object"):
            pg._registered_n_ground({}, [])
        with pytest.raises(pg.GateEvaluationError, match="positive integer"):
            pg._registered_n_ground({"grounding": {"n_ground": 0}}, [])


class TestEntryPointGuards:
    def test_formal_train_scope_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(pg.GateEvaluationError, match="post-binding artifact"):
            pg.evaluate_prebinding_gates(
                probe_artifact_path=tmp_path / "probe.npz",
                preregistration_path=tmp_path / "reg.json",
                data_root=tmp_path,
                strategy="toy",
                scope="formal_train",
                partition_seed=0,
                msg_fraction=0.8,
                expected_missing_features=[],
            )

    def test_non_draft_registration_is_refused(self, tmp_path: Path) -> None:
        registration_path = tmp_path / "reg.json"
        registration_path.write_text(json.dumps(_registration(status="BINDING")))
        with pytest.raises(pg.GateEvaluationError, match="DRAFT"):
            pg.evaluate_prebinding_gates(
                probe_artifact_path=tmp_path / "probe.npz",
                preregistration_path=registration_path,
                data_root=tmp_path,
                strategy="toy",
                scope="calibration_fit",
                partition_seed=0,
                msg_fraction=0.8,
                expected_missing_features=[],
            )


class TestVerdictEnforcement:
    def test_failing_rehearsal_report_stops_the_process(self, tmp_path: Path) -> None:
        report = _report(slot_recall=0.0, pi_v2=0.0, clustering_r2=0.0)
        with pytest.raises(SystemExit, match="did not pass"):
            pg.enforce_verdict(
                report, scope="qualification_qual", output_path=tmp_path / "gates.json"
            )

    def test_incomplete_rehearsal_report_stops_the_process(self, tmp_path: Path) -> None:
        # Every evaluated gate passes, but G3.4/G3.5 are incomplete: a rehearsal
        # that could not judge all five gates is not a qualification.
        report = _report(slot_recall=0.99, pi_v2=0.99, clustering_r2=0.99)
        assert report["n_incomplete"] == 2
        with pytest.raises(SystemExit, match="did not pass"):
            pg.enforce_verdict(
                report, scope="qualification_qual", output_path=tmp_path / "gates.json"
            )

    def test_calibration_never_fails_on_an_incomplete_report(self, tmp_path: Path) -> None:
        # Calibration exists to run while thresholds are unresolved.
        report = _report(slot_recall=0.0, pi_v2=0.0, clustering_r2=0.0)
        pg.enforce_verdict(
            report, scope="calibration_fit", output_path=tmp_path / "gates.json"
        )


class TestUnimplementedGates:
    def test_reports_the_two_scoring_gates(self) -> None:
        assert pg.unimplemented_gates(_registration()) == ["G3.4", "G3.5"]

    def test_empty_when_every_registered_gate_has_an_evaluator(self) -> None:
        registration = _registration()
        gates = registration["prebinding_qualification"]["gates"]
        registration["prebinding_qualification"]["gates"] = gates[:3]
        assert pg.unimplemented_gates(registration) == []

    def test_frozen_thresholds_do_not_make_a_gate_implementable(self) -> None:
        # The review's P1: freezing G3.4/G3.5 must not be mistaken for
        # implementing them, or the rehearsal spends V_qual and only then
        # discovers it cannot judge those gates.
        registration = _registration()
        for gate in registration["prebinding_qualification"]["gates"]:
            if gate["id"] in {"G3.4", "G3.5"}:
                gate["threshold"] = 0.01
        assert pg.unimplemented_gates(registration) == ["G3.4", "G3.5"]

    def test_live_v3_registration_is_not_yet_implementable(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registration = json.loads(
            (
                repo_root
                / "docs"
                / "registrations"
                / "g5_e2e_stage1_preregistration_v3.json"
            ).read_text()
        )
        assert pg.unimplemented_gates(registration) == ["G3.4", "G3.5"]


def test_live_v3_registration_yields_three_evaluable_and_two_blocked() -> None:
    """The shipped v3 draft must parse, and must not certify anything today."""
    repo_root = Path(__file__).resolve().parents[2]
    registration = json.loads(
        (
            repo_root
            / "docs"
            / "registrations"
            / "g5_e2e_stage1_preregistration_v3.json"
        ).read_text()
    )
    report = pg.build_gate_report(
        registration,
        _evaluation(slot_recall=0.99, pi_v2=0.99, clustering_r2=0.99),
        registration_sha="f" * 64,
        scope="calibration_fit",
    )
    assert report["n_evaluated"] == 3
    assert report["n_incomplete"] == 2
    assert report["passed"] is False
