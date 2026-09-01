"""CPU-only tests for the strict-LLP kd_rank Optuna sweep driver."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.experiments import kd_rank_strict_hpo as hpo

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
