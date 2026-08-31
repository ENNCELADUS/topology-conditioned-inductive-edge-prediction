import os
from pathlib import Path
from typing import Any

import pytest
from src.autoresearch.ledger import append_row, read_rows

pytestmark = pytest.mark.unit

DELTA_KEYS = ("gs", "log_rd", "degree_mmd", "clustering_mmd", "spectral_mmd")


def valid_verdict(decision: str = "keep") -> dict[str, Any]:
    improved = ["gs"] if decision == "keep" else []
    return {
        "decision": decision,
        "improved": improved,
        "regressed": [],
        "deltas": {name: 0.5 - 0.6 if name == "gs" and improved else 0.0 for name in DELTA_KEYS},
        "auprc_delta": 0.0,
        "reasons": (
            ["improved without regression: gs"]
            if decision == "keep"
            else ["no topology metric improved beyond tolerance"]
        ),
    }


def valid_row(trial: int = 1, **overrides: object) -> dict[str, Any]:
    status = overrides.get("status", "baseline" if trial == 1 else "keep")
    metrics = dict.fromkeys(
        ("auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"), 0.5
    )
    if status == "keep":
        metrics["gs"] = 0.6
    row: dict[str, Any] = {
        "trial": trial,
        "campaign": "kd_logit",
        "commit": f"c{trial:07d}",
        "config_hash": "deadbeef",
        "output_dir": f"outputs/trial_{trial:03d}",
        "hypothesis": "baseline" if trial == 1 else f"hypothesis {trial}",
        "status": status,
        "metrics": metrics,
        "selected_epoch": 2,
        "total_seconds": 123.0,
        "verdict": None if trial == 1 else valid_verdict(),
        "asi": "healthy fit",
        "timestamp": "2026-08-30T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    append_row(path, valid_row(2))
    rows = read_rows(path)
    assert [row["trial"] for row in rows] == [1, 2]
    assert rows[1]["status"] == "keep"


def test_missing_key_rejected(tmp_path: Path) -> None:
    row = valid_row(1)
    del row["hypothesis"]
    with pytest.raises(ValueError, match="hypothesis"):
        append_row(tmp_path / "ledger.jsonl", row)


def test_unexpected_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpected"):
        append_row(tmp_path / "ledger.jsonl", valid_row(extra="nope"))


def test_scalar_types_rejected(tmp_path: Path) -> None:
    for key, value in (("trial", True), ("campaign", 3), ("selected_epoch", 2.5)):
        row = valid_row()
        row[key] = value
        with pytest.raises(ValueError, match=key):
            append_row(tmp_path / f"{key}.jsonl", row)


def test_non_monotonic_trial_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="trial must be 2"):
        append_row(path, valid_row(3))


def test_duplicate_output_dir_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="duplicate output_dir"):
        append_row(path, valid_row(2, output_dir=valid_row(1)["output_dir"]))


def test_duplicate_commit_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="duplicate commit"):
        append_row(path, valid_row(2, commit="c0000001"))


def test_keep_row_contradicting_verdict_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="decision"):
        append_row(path, valid_row(2, verdict=valid_verdict("revert")))


@pytest.mark.parametrize("campaign", ["kd_unknown", "KD_LOGIT", "", True])
def test_unknown_campaign_rejected(tmp_path: Path, campaign: object) -> None:
    with pytest.raises(ValueError, match="campaign"):
        append_row(tmp_path / "ledger.jsonl", valid_row(campaign=campaign))


@pytest.mark.parametrize("campaign", ["kd_logit", "kd_rank", "kd_gram", "kd_rep", "kd_gen"])
def test_each_campaign_name_is_accepted(tmp_path: Path, campaign: str) -> None:
    append_row(tmp_path / "ledger.jsonl", valid_row(campaign=campaign))


def test_each_campaign_starts_with_exactly_one_baseline(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="must start"):
        append_row(path, valid_row(1, status="keep", verdict=valid_verdict()))
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="already has"):
        append_row(path, valid_row(2, status="baseline", verdict=None))
    append_row(
        path,
        valid_row(
            2,
            campaign="kd_rank",
            status="baseline",
            verdict=None,
            output_dir="outputs/kd_rank/baseline",
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("improved", ["degree_mmd", "gs"], "frozen order"),
        ("improved", ["unknown"], "frozen order"),
        ("deltas", dict.fromkeys(DELTA_KEYS[:-1], 0.0), "deltas"),
        (
            "deltas",
            {name: (True if name == "gs" else 0.0) for name in DELTA_KEYS},
            "finite number",
        ),
        (
            "deltas",
            {name: (float("inf") if name == "gs" else 0.0) for name in DELTA_KEYS},
            "finite number",
        ),
        ("auprc_delta", True, "auprc_delta"),
        ("auprc_delta", float("nan"), "auprc_delta"),
        ("reasons", ["made up"], "reasons"),
    ],
)
def test_incomplete_or_invalid_verdict_evidence_rejected(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    verdict = valid_verdict()
    verdict[field] = value
    with pytest.raises(ValueError, match=match):
        append_row(path, valid_row(2, verdict=verdict))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("deltas", dict.fromkeys(DELTA_KEYS, 0.0), "exact oriented delta"),
        ("auprc_delta", 0.1, "must equal"),
    ],
)
def test_finite_but_inexact_verdict_deltas_rejected(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    verdict = valid_verdict()
    verdict[field] = value
    with pytest.raises(ValueError, match=match):
        append_row(path, valid_row(2, verdict=verdict))


def test_verdict_keys_and_keep_consistency_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    incomplete = valid_verdict()
    del incomplete["reasons"]
    with pytest.raises(ValueError, match="exactly"):
        append_row(path, valid_row(2, verdict=incomplete))

    contradictory = valid_verdict()
    contradictory["improved"] = []
    contradictory["deltas"]["gs"] = 0.0
    contradictory["reasons"] = ["no topology metric improved beyond tolerance"]
    with pytest.raises(ValueError, match="contradicts"):
        append_row(path, valid_row(2, verdict=contradictory))


def test_complete_regression_verdict_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    metrics = dict(valid_row(1)["metrics"])
    metrics["degree_mmd"] = 0.6
    verdict = valid_verdict("revert")
    verdict["regressed"] = ["degree_mmd"]
    verdict["deltas"]["degree_mmd"] = 0.6 - 0.5
    verdict["reasons"] = ["regressed beyond tolerance: degree_mmd"]
    append_row(path, valid_row(2, status="revert", metrics=metrics, verdict=verdict))


def test_append_flushes_before_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.jsonl"
    observed: list[str] = []

    def inspect_flushed_file(_fd: int) -> None:
        observed.append(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(os, "fsync", inspect_flushed_file)
    append_row(path, valid_row(1))
    assert observed and '"trial": 1' in observed[0]


def test_crash_row_requires_null_fields(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    append_row(
        path,
        valid_row(
            2,
            status="crash",
            verdict=None,
            metrics=None,
            selected_epoch=None,
            total_seconds=None,
        ),
    )
    with pytest.raises(ValueError, match="crash rows"):
        append_row(path, valid_row(3, status="crash", verdict=None))


def test_replay_skips_torn_line_and_append_continues(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"trial": 2, "camp')
    assert [row["trial"] for row in read_rows(path)] == [1]
    append_row(path, valid_row(2))
    assert [row["trial"] for row in read_rows(path)] == [1, 2]


def test_non_finite_metric_and_timing_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        append_row(
            tmp_path / "metric.jsonl",
            valid_row(
                metrics={
                    "auprc": float("inf"),
                    "gs": 0.5,
                    "rd": 0.5,
                    "degree_mmd": 0.5,
                    "clustering_mmd": 0.5,
                    "spectral_mmd": 0.5,
                }
            ),
        )
    with pytest.raises(ValueError, match="total_seconds"):
        append_row(tmp_path / "timing.jsonl", valid_row(total_seconds=float("nan")))


@pytest.mark.parametrize("selected_epoch", [None, 0, -1, True])
def test_non_crash_requires_positive_selected_epoch(tmp_path: Path, selected_epoch: object) -> None:
    row = valid_row(selected_epoch=selected_epoch)
    with pytest.raises(ValueError, match="selected_epoch"):
        append_row(tmp_path / "epoch.jsonl", row)


@pytest.mark.parametrize("total_seconds", [None, -1, True])
def test_non_crash_requires_nonnegative_total_seconds(
    tmp_path: Path, total_seconds: object
) -> None:
    row = valid_row(total_seconds=total_seconds)
    with pytest.raises(ValueError, match="total_seconds"):
        append_row(tmp_path / "timing.jsonl", row)
