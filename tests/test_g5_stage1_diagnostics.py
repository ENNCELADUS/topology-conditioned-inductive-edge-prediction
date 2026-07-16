"""Lightweight tests for the Stage-1 diagnostic report generator."""

import numpy as np
from src.experiments import g5_stage1_diagnostics as diagnostics


def test_channel_correlation_reports_all_registered_channels() -> None:
    x = np.arange(12, dtype=np.float64)
    report = diagnostics._correlation_report(np.column_stack((x, x * 2, -x, x + 3)))

    assert report["labels"] == ["s0", "s1", "s2", "s2_aa"]
    assert report["n"] == 12
    matrix = np.asarray(report["pearson"])
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert matrix[0, 2] == -1.0


def test_nonfinite_diagnostic_values_are_written_as_json_null() -> None:
    payload = {"pearson": float("nan"), "nested": [float("inf"), 1.0]}

    assert diagnostics._finite_json(payload) == {
        "pearson": None,
        "nested": [None, 1.0],
    }


def test_diagnostics_cli_requires_both_report_outputs() -> None:
    parser = diagnostics.build_parser()
    option_names = {option for action in parser._actions for option in action.option_strings}

    assert "--fidelity-output" in option_names
    assert "--cost-output" in option_names
    assert "--s0-universe" in option_names
