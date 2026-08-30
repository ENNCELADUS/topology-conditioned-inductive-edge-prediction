from pathlib import Path
from typing import Any

import pytest
from src.autoresearch.ledger import append_row
from src.autoresearch.summary import render_summary

pytestmark = pytest.mark.unit


def row(trial: int, status: str, campaign: str = "kd_logit", **overrides: object) -> dict[str, Any]:
    verdict: dict[str, Any] | None = None
    if status in {"keep", "revert"}:
        verdict = {"decision": status, "improved": ["gs"] if status == "keep" else []}
    base: dict[str, Any] = {
        "trial": trial,
        "campaign": campaign,
        "commit": f"c{trial:07d}",
        "config_hash": "deadbeef",
        "output_dir": f"outputs/b1_row_kd_ar/{campaign}/trial_{trial:03d}",
        "hypothesis": f"hypothesis {trial}",
        "status": status,
        "metrics": {
            "auprc": 0.82,
            "gs": 0.52,
            "rd": 1.10,
            "degree_mmd": 0.90,
            "clustering_mmd": 0.85,
            "spectral_mmd": 0.80,
        },
        "selected_epoch": 2,
        "total_seconds": 123.0,
        "verdict": verdict,
        "asi": "healthy fit",
        "timestamp": "2026-08-30T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_summary_reports_standings_and_recent_trials(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, row(1, "baseline"))
    append_row(path, row(2, "keep", metrics=dict(row(1, "baseline")["metrics"], gs=0.60)))
    append_row(path, row(3, "revert"))
    text = render_summary(path)
    lines = text.splitlines()
    assert lines[0] == "# autoresearch summary"
    assert lines[1].startswith("campaign kd_logit: trials=3 keeps=1 incumbent=")
    assert "trial_002" in lines[1]
    assert "gs=0.6" in lines[1]
    assert lines[2] == "last 3 trials:"
    assert lines[3].startswith("#1 [kd_logit] baseline | hypothesis 1")
    assert "improved:gs" in lines[4]
    assert lines[5].startswith("#3 [kd_logit] revert")


def test_summary_honors_last_limit(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, row(1, "baseline"))
    append_row(path, row(2, "revert"))
    append_row(path, row(3, "revert"))
    text = render_summary(path, last=1)
    assert "last 1 trials:" in text
    assert "#3 " in text
    assert "#2 " not in text


def test_summary_of_empty_ledger(tmp_path: Path) -> None:
    assert render_summary(tmp_path / "missing.jsonl") == "ledger empty; no trials recorded\n"


def test_summary_includes_default_open_ideas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_dir = tmp_path / "autoresearch"
    ideas_dir.mkdir()
    (ideas_dir / "ideas.md").write_text(
        "# Ideas backlog\n\n- test idea\n- second idea\n", encoding="utf-8"
    )
    path = tmp_path / "ledger.jsonl"
    append_row(path, row(1, "baseline"))
    monkeypatch.chdir(tmp_path)
    text = render_summary(path)
    assert "open ideas:" in text
    assert "- test idea" in text
    assert "- second idea" in text
