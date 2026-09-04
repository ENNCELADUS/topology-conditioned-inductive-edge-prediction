"""CPU-only tests for the kd_rank_rep strict Optuna sweep driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
from src.distill.config import DistillConfig
from src.experiments import kd_rank_rep_hpo as hpo
from src.experiments import kd_rank_strict_hpo as shared

from tests.autoresearch.conftest import make_cadence_rows

pytestmark = pytest.mark.unit

BASE_CONFIG = Path("configs/autoresearch/kd_rank_rep.yaml")


def test_base_config_is_a_legal_kd_rank_rep_arm() -> None:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    assert DistillConfig.from_mapping(cfg["distill"]).arm == "kd_rank_rep"
    assert cfg["eval"]["topology_every"] == 2
    rank_cfg = yaml.safe_load(Path("configs/autoresearch/kd_rank.yaml").read_text(encoding="utf-8"))
    for key in ("output_dir", "distill"):
        cfg.pop(key)
        rank_cfg.pop(key)
    assert cfg == rank_cfg


def test_enqueued_priors_match_spec() -> None:
    assert hpo.ENQUEUED_PRIORS == (
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 0.1},
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 1.0},
        {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 10.0},
        {"w_rank": 1.0, "w_dist": 1.0, "w_rep": 1.0},
    )
    assert len(hpo.ENQUEUED_PRIORS) == hpo.N_STARTUP_TRIALS


def test_suggest_params_covers_the_three_log_boxes(tmp_path: Path) -> None:
    study = shared.build_study(
        tmp_path / "optuna.db", study_name=hpo.STUDY_NAME, n_startup_trials=1
    )
    trial = study.ask()
    params = hpo.suggest_params(trial)
    assert set(params) == {"w_rank", "w_dist", "w_rep"}
    assert 0.01 <= float(params["w_rank"]) <= 1.0  # type: ignore[arg-type]
    assert 0.1 <= float(params["w_dist"]) <= 100.0  # type: ignore[arg-type]
    assert 0.01 <= float(params["w_rep"]) <= 100.0  # type: ignore[arg-type]
    assert trial.distributions == {
        "w_rank": FloatDistribution(0.01, 1.0, log=True),
        "w_dist": FloatDistribution(0.1, 100.0, log=True),
        "w_rep": FloatDistribution(0.01, 100.0, log=True),
    }


def test_materialize_changes_only_whitelisted_keys(tmp_path: Path) -> None:
    params = {"w_rank": 0.3, "w_dist": 5.0, "w_rep": 2.0}
    config_path = hpo.materialize_trial_config(
        BASE_CONFIG, params, 3, tmp_path, bank="h2ns5", margin=0.2
    )
    assert config_path == tmp_path / "configs" / "trial_003.yaml"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    trial = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert trial["output_dir"] == str(tmp_path / "trial_003")
    assert trial["distill"]["w_rank"] == 0.3
    assert trial["distill"]["w_dist"] == 5.0
    assert trial["distill"]["w_rep"] == 2.0
    assert trial["distill"]["margin"] == 0.2
    assert trial["distill"]["context_targets_path"] == shared.BANKS["h2ns5"].path
    assert DistillConfig.from_mapping(trial["distill"]).arm == "kd_rank_rep"
    trial["output_dir"] = base["output_dir"]
    for key in ("w_rank", "w_dist", "w_rep", "margin", "context_targets_path"):
        trial["distill"][key] = base["distill"][key]
    assert trial == base


def test_materialize_rejects_unknown_bank_and_zero_weight(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        hpo.materialize_trial_config(
            BASE_CONFIG,
            {"w_rank": 0.1, "w_dist": 1.0, "w_rep": 1.0},
            1,
            tmp_path,
            bank="h9",
            margin=0.1,
        )
    with pytest.raises(ValueError):
        hpo.materialize_trial_config(
            BASE_CONFIG,
            {"w_rank": 0.1, "w_dist": 0.0, "w_rep": 1.0},
            1,
            tmp_path,
            bank="h2ns3",
            margin=0.1,
        )


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    argv = ["--sweep-dir", str(tmp_path), "--n-trials", "2"]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return hpo.build_parser().parse_args(argv)


def test_parser_defaults_match_spec() -> None:
    args = hpo.build_parser().parse_args([])
    assert args.base_config == BASE_CONFIG
    assert args.sweep_dir == Path("outputs/b1_kd_rank_rep_hpo")
    assert (args.n_trials, args.rd_band, args.bank, args.margin) == (12, 0.05, "h2ns3", 0.1)
    with pytest.raises(SystemExit):
        hpo.build_parser().parse_args(["--bank", "h9"])


def test_require_bank_fails_closed_without_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = shared.BANKS["h2ns3"]
    monkeypatch.setitem(
        shared.BANKS,
        "h2ns3",
        shared.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(tmp_path / "bank")),
    )
    with pytest.raises(RuntimeError, match="h2ns3"):
        hpo.require_bank(_args(tmp_path))
    (tmp_path / "bank").mkdir()
    (tmp_path / "bank" / "manifest.json").write_text("{}", encoding="utf-8")
    hpo.require_bank(_args(tmp_path))


def _publish_run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["output_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in make_cadence_rows()), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {"selected_epoch": 2, "arm": "kd_rank_rep", "config_hash": "d", "checkpoint_id": "c"}
        ),
        encoding="utf-8",
    )
    (run_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "fixture", "total_seconds": 60.0}),
        encoding="utf-8",
    )


def test_main_runs_priors_first_with_the_frozen_bank_and_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = shared.BANKS["h2ns5"]
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        shared.BANKS, "h2ns5", shared.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank_dir))
    )
    launched: list[Path] = []

    def fake_run(cmd: list[str]) -> int:
        assert cmd[:3] == ["bash", "hpc/run.sh", "train"] and cmd[-1] == "--skip-test"
        launched.append(Path(cmd[3]))
        _publish_run(Path(cmd[3]))
        return 0

    monkeypatch.setattr(shared, "run_command", fake_run)
    argv = ["--sweep-dir", str(tmp_path), "--bank", "h2ns5", "--margin", "0.2"]
    hpo.main(argv)
    study = shared.build_study(tmp_path / "optuna.db", study_name=hpo.STUDY_NAME)
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    assert len(complete) == 12
    assert [t.params for t in complete[:4]] == list(hpo.ENQUEUED_PRIORS)
    assert all(set(t.params) == {"w_rank", "w_dist", "w_rep"} for t in complete)
    for config_path in launched:
        distill = yaml.safe_load(config_path.read_text(encoding="utf-8"))["distill"]
        assert distill["context_targets_path"] == str(bank_dir)
        assert distill["margin"] == 0.2
    hpo.main(argv)  # Restart at the completed budget must not enqueue or launch duplicates.
    assert len(launched) == len(study.get_trials(deepcopy=False)) == 12
    assert "number state w_rank w_dist w_rep auprc" in capsys.readouterr().out
