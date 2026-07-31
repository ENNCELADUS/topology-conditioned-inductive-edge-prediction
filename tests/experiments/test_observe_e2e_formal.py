"""Unit tests for the read-only formal-run observation reports (v5 observation_plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.experiments import observe_e2e_formal

pytestmark = pytest.mark.unit


def _write_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    profile = {
        "arm": "full",
        "run_kind": "formal",
        "quality_guards_passed": True,
        "optimizer_step_gradients": [
            {
                "step": 1,
                "optimizer_group_gradients": {
                    "generator": {"norm": 2835.5, "clip_coefficient": 0.05, "active": True},
                    "pair_encoder": {"norm": 2.0, "clip_coefficient": 1.0, "active": True},
                },
            },
            {
                "step": 2,
                "optimizer_group_gradients": {
                    "generator": {"norm": 40.0, "clip_coefficient": 0.075, "active": True},
                    "pair_encoder": {"norm": 1.5, "clip_coefficient": 1.0, "active": True},
                },
            },
        ],
        "gradient_norm_series": [
            {"step": 2, "family_group_ratios": {"generator": {"pair_encoder": 12.5}}}
        ],
        "quality_guard_events": [{"kind": "optimizer_gradient"}],
        "initial_slot_health": {
            "h_pairwise_cosine_mean": 0.62,
            "quality_threshold_missed": False,
        },
        "precision_differential": {
            "end_ramp": {
                "quality_pass": True,
                "full_relative_l2": 0.01,
                "f_logit_relative_l2": 0.012,
                "residual_relative_l2": 0.02,
                "residual_correlation": 0.9995,
            }
        },
    }
    (run_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    history = [
        {
            "epoch": 1,
            "fidelity": {
                "h_pairwise_cosine_mean": 0.70,
                "plan_rank1_marginal_residual": 0.20,
            },
            "quality_thresholds": {"slot_collapse": False},
        },
        {
            "epoch": 2,
            "fidelity": {
                "h_pairwise_cosine_mean": 0.96,
                "plan_rank1_marginal_residual": 0.04,
            },
            "quality_thresholds": {"slot_collapse": True},
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "complete", "checkpoint_id": "ckpt_full"}), encoding="utf-8"
    )
    return run_dir


def test_build_report_covers_all_three_registered_observation_sections(
    tmp_path: Path,
) -> None:
    report = observe_e2e_formal.build_report(_write_run_dir(tmp_path))

    assert "== 1. Generator clipping ==" in report
    assert "== 2. Slot collapse ==" in report
    assert "== 3. End-ramp precision differential ==" in report
    assert "arm=full run_kind=formal status=complete" in report

    # Section 1: per-group maxima, streaks vs the registered persistent budget,
    # family ratios, and quality-event counts.
    assert "group generator: steps=2 max_pre_clip_norm=2835.5 (step 1)" in report
    assert "max consecutive streak 2 vs registered 10" in report
    assert "worst family ratio 12.5" in report
    assert "optimizer_gradient=1" in report

    # Section 2: step-0 health, peak trajectory vs the 0.95 trip, and the epoch
    # where the collapse predicate fired.
    assert "step-0: h_pairwise_cosine_mean=0.62" in report
    assert "peak=0.96 (epoch 2)" in report
    assert "epochs with slot_collapse predicate true: [2]" in report

    # Section 3: recorded end-ramp payload; selected checkpoint not recorded.
    assert "end_ramp: quality_pass=True full_relative_l2=0.01" in report
    assert "residual_correlation=0.9995" in report
    assert "selected: not recorded" in report


def test_build_report_requires_profile_json(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="profile.json"):
        observe_e2e_formal.build_report(empty)
