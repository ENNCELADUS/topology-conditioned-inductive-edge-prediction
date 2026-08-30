from pathlib import Path
from typing import Any

import pytest
from src.autoresearch.ledger import append_row, read_rows

pytestmark = pytest.mark.unit


def valid_row(trial: int = 1, **overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial": trial,
        "campaign": "kd_logit",
        "commit": f"c{trial:07d}",
        "config_hash": "deadbeef",
        "output_dir": f"outputs/trial_{trial:03d}",
        "hypothesis": "baseline" if trial == 1 else f"hypothesis {trial}",
        "status": "baseline" if trial == 1 else "keep",
        "metrics": dict.fromkeys(
            ("auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"), 0.5
        ),
        "selected_epoch": 2,
        "total_seconds": 123.0,
        "verdict": None if trial == 1 else {"decision": "keep"},
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
        append_row(path, valid_row(2, verdict={"decision": "revert"}))


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
