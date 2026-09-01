"""CPU-only tests for the strict-LLP kd_rank Optuna sweep driver."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from src.autoresearch.metrics_io import RunFailure
from src.experiments import kd_rank_strict_hpo as hpo

from tests.autoresearch.conftest import make_cadence_rows

BASE_CONFIG = Path("configs/autoresearch/kd_rank.yaml")


def test_bank_registry_matches_spec() -> None:
    assert set(hpo.BANKS) == {"h2ns1", "h2ns3", "h2ns5", "h3ns3"}
    assert hpo.BANKS["h2ns1"].path == "outputs/distill/kd_ctx_targets_breadth_first"
    h3ns3 = hpo.BANKS["h3ns3"]
    assert (h3ns3.rw_step, h3ns3.hops, h3ns3.ns_rate) == (3, 3, 3)


def test_enqueued_priors_match_spec() -> None:
    assert len(hpo.ENQUEUED_PRIORS) == 6
    assert hpo.ENQUEUED_PRIORS[0] == {"w_rank": 1.0, "w_dist": 1.0, "bank": "h2ns1", "margin": 0.1}
    assert hpo.ENQUEUED_PRIORS[4] == {
        "w_rank": 0.1,
        "w_dist": 100.0,
        "bank": "h2ns3",
        "margin": 0.1,
    }


def test_materialize_changes_only_whitelisted_keys(tmp_path: Path) -> None:
    params = {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns3", "margin": 0.05}
    config_path = hpo.materialize_trial_config(BASE_CONFIG, params, 7, tmp_path)
    assert config_path == tmp_path / "configs" / "trial_007.yaml"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    trial = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert trial["output_dir"] == str(tmp_path / "trial_007")
    assert trial["distill"]["w_rank"] == 0.1
    assert trial["distill"]["w_dist"] == 10.0
    assert trial["distill"]["margin"] == 0.05
    assert trial["distill"]["context_targets_path"] == hpo.BANKS["h2ns3"].path
    trial["output_dir"] = base["output_dir"]
    trial["distill"] = base["distill"]
    assert trial == base


def test_materialize_rejects_unknown_bank(tmp_path: Path) -> None:
    params = {"w_rank": 0.1, "w_dist": 10.0, "bank": "h9ns9", "margin": 0.1}
    with pytest.raises(KeyError):
        hpo.materialize_trial_config(BASE_CONFIG, params, 1, tmp_path)


def test_materialize_rejects_illegal_weight_pattern(tmp_path: Path) -> None:
    params = {"w_rank": 0.0, "w_dist": 0.0, "bank": "h2ns1", "margin": 0.1}
    with pytest.raises(ValueError):
        hpo.materialize_trial_config(BASE_CONFIG, params, 1, tmp_path)


def _publish_run(run_dir: Path, rows: list[dict[str, object]], *, failure: bool = False) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {"selected_epoch": 2, "arm": "kd_rank", "config_hash": "d", "checkpoint_id": "c"}
        ),
        encoding="utf-8",
    )
    (run_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "fixture", "total_seconds": 60.0}),
        encoding="utf-8",
    )
    if failure:
        (run_dir / "failure.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    return run_dir


def test_trial_outcome_reads_cadence2_surface(tmp_path: Path) -> None:
    run_dir = _publish_run(tmp_path / "trial_000", make_cadence_rows())
    outcome = hpo.trial_outcome(run_dir, rd_band=0.05)
    # make_cadence_rows: epoch 4 dominates the cadence-2 due set (gs .80, rd 1.02, mmds .60).
    assert outcome.gs == pytest.approx(0.80)
    assert outcome.geo_mmd == pytest.approx(0.60)
    assert outcome.constraint == pytest.approx(abs(math.log(1.02)) - 0.05)
    assert outcome.constraint < 0.0
    assert outcome.surface["selected_epoch"] == 4.0


def test_trial_outcome_flags_rd_outside_band(tmp_path: Path) -> None:
    rows = make_cadence_rows()
    rows[3]["val_rd_bfs"] = 1.20
    run_dir = _publish_run(tmp_path / "trial_001", rows)
    assert hpo.trial_outcome(run_dir, rd_band=0.05).constraint > 0.0


def test_trial_outcome_propagates_run_failure(tmp_path: Path) -> None:
    run_dir = _publish_run(tmp_path / "trial_002", make_cadence_rows(), failure=True)
    with pytest.raises(RunFailure):
        hpo.trial_outcome(run_dir, rd_band=0.05)


def test_trial_outcome_rejects_nonpositive_mmd_ratio(tmp_path: Path) -> None:
    rows = make_cadence_rows()
    rows[3]["val_degree_mmd_ratio"] = 0.0
    run_dir = _publish_run(tmp_path / "trial_003", rows)
    with pytest.raises(ValueError):
        hpo.trial_outcome(run_dir, rd_band=0.05)
