"""Terminal report comparisons and partial learning-curve rendering."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from src.experiments.l3ppi_report import (
    comparison_rows,
    generate_report,
    read_history,
    runtime_row,
)


def _report() -> dict[str, Any]:
    return {
        "schema_version": "test_protocol_v8",
        "arm": {"checkpoint_id": "checkpoint", "selected_epoch": 3},
        "edge": {
            "strategy": "breadth_first",
            "pairs_source": "test",
            "num_rows": 20,
            "self_rows": 0,
            "self_loops_included": True,
            "metrics": {
                "n_pos": 10,
                "n_neg": 10,
                "auroc": 0.7,
                "auprc": 0.8,
                "accuracy": 0.6,
                "f1": 0.65,
                "mcc": 0.2,
                "ece": 0.1,
                "brier": 0.2,
            },
        },
        "graph": {
            "fixed_threshold": {
                "test": {
                    "graph_similarity": {"bfs_macro": 0.4},
                    "relative_density": {"bfs_macro": 0.8},
                    "mmd_ratio": {"degree": 2.0, "clustering": 3.0, "spectral": 4.0},
                    "per_size": {"20": {"sample_count": 50}},
                }
            }
        },
        "provenance": {
            "test": {"path": "model-specific", "sha256": "model-specific"},
            "test_topology": {"num_rows": 100},
        },
    }


def test_extract_actual_v8_metrics_and_ignore_model_specific_score_paths() -> None:
    base = _report()
    other = copy.deepcopy(base)
    other["provenance"]["test"].update(path="different", sha256="different")
    rows = comparison_rows(base, other)
    assert [row["arm"] for row in rows] == ["B0", "L3-PPI"]
    assert rows[0]["gs"] == 0.4
    assert rows[0]["rd"] == 0.8
    assert rows[0]["spectral_mmd"] == 4.0
    assert rows[0]["auprc"] == 0.8


@pytest.mark.parametrize("field", ["rows", "class", "strategy", "schema", "universe"])
def test_mismatched_benchmark_metadata_fails(field: str) -> None:
    base = _report()
    other = copy.deepcopy(base)
    if field == "rows":
        other["edge"]["num_rows"] = 21
    elif field == "class":
        other["edge"]["metrics"]["n_pos"] = 11
    elif field == "strategy":
        other["edge"]["strategy"] = "depth_first"
    elif field == "schema":
        other["schema_version"] = "test_protocol_v7"
    else:
        other["graph"]["fixed_threshold"]["test"]["per_size"]["20"]["sample_count"] = 49
    with pytest.raises(ValueError):
        comparison_rows(base, other)


def test_partial_history_plot_and_terminal_comparison(tmp_path: Path) -> None:
    study = tmp_path / "study"
    trial = study / "trial_005"
    trial.mkdir(parents=True)
    row = {
        "phase": "joint",
        "epoch": 3,
        "train_task_loss": 0.4,
        "train_path_loss": 0.1,
        "val_task_loss": 0.5,
        "val_auprc": 0.8,
        "val_auroc": 0.7,
        "train_soft_paths": 3,
        "train_hard_paths": 2,
        "val_soft_paths": 3,
        "val_hard_paths": 2,
        "train_seconds": 2,
        "pairs_per_second": 20,
        "peak_gpu_gib": 1,
        "topology": {
            "metrics": {
                "gs": 0.4,
                "rd": 0.8,
                "degree_mmd": 2,
                "clustering_mmd": 3,
                "spectral_mmd": 4,
            }
        },
    }
    (trial / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    (trial / "selection.json").write_text('{"selected_epoch": 3}')
    baseline = tmp_path / "b0.json"
    baseline.write_text(json.dumps(_report()))
    out = tmp_path / "report"
    assert generate_report(study, baseline, out)["status"] == "curves_only"
    assert (out / "trial_005_curves.png").stat().st_size > 1000
    assert (out / "trial_005_curves.pdf").stat().st_size > 1000
    assert not (out / "README.md").exists()
    assert not (out / "comparison.csv").exists()
    (study / "winner.json").write_text('{"trial_dir": "/remote/path/trial_005"}')
    (trial / "test_report.json").write_text(json.dumps(_report()))
    assert generate_report(study, baseline, out)["status"] == "compared"
    assert (out / "comparison.csv").exists()
    assert "No inference-speed comparison" in (out / "README.md").read_text()


def test_runtime_uses_observed_time_weighting_and_partial_jsonl(tmp_path: Path) -> None:
    rows = [
        {"train_seconds": 2, "pairs_per_second": 10, "peak_gpu_gib": 1},
        {"train_seconds": 8, "pairs_per_second": 20, "peak_gpu_gib": 2},
    ]
    result = runtime_row("trial", rows)
    assert result["train_seconds"] == 10
    assert result["pairs_per_second"] == 18
    assert result["peak_gpu_gib"] == 2
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(rows[0]) + '\n{"partial":')
    assert read_history(path) == rows[:1]
