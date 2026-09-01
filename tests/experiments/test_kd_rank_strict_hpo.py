"""CPU-only tests for the strict-LLP kd_rank Optuna sweep driver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import optuna
import pytest
import yaml
from optuna.trial import TrialState
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


def test_build_study_directions_and_priors(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    assert study.study_name == "kd_rank_strict_llp"
    assert [d.name.lower() for d in study.directions] == ["maximize", "minimize"]
    assert len(study.get_trials(deepcopy=False)) == 6


def test_build_study_enqueue_is_idempotent(tmp_path: Path) -> None:
    hpo.build_study(tmp_path / "optuna.db")
    study = hpo.build_study(tmp_path / "optuna.db")
    assert len(study.get_trials(deepcopy=False)) == 6


def test_suggest_params_consumes_priors_in_order(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    first = hpo.suggest_params(study.ask())
    second = hpo.suggest_params(study.ask())
    assert first == hpo.ENQUEUED_PRIORS[0]
    assert second == hpo.ENQUEUED_PRIORS[1]


def test_constraints_default_to_infeasible() -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.add_trial(
        optuna.trial.create_trial(params={}, distributions={}, values=[0.5, 1.0], user_attrs={})
    )
    assert hpo._constraints(study.get_trials(deepcopy=False)[0]) == (float("inf"),)


def _ask_running_trial(study: optuna.Study) -> optuna.Trial:
    trial = study.ask()
    hpo.suggest_params(trial)
    return trial


def test_reconcile_completed_run_is_retold_with_values(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    trial = _ask_running_trial(study)
    _publish_run(tmp_path / f"trial_{trial.number:03d}", make_cadence_rows())
    hpo.reconcile_running(study, tmp_path, rd_band=0.05)
    trials = study.get_trials(deepcopy=False)
    assert [t.state for t in trials if t.number == trial.number] == [TrialState.FAIL]
    twins = [t for t in trials if t.state == TrialState.COMPLETE]
    assert len(twins) == 1
    assert twins[0].params == dict(hpo.ENQUEUED_PRIORS[0])
    assert twins[0].values == pytest.approx([0.80, 0.60])
    assert twins[0].user_attrs["constraint"][0] < 0.0
    assert twins[0].user_attrs["surface"]["selected_epoch"] == 4.0
    assert [t.number for t in study.best_trials] == [twins[0].number]


def test_reconcile_failed_and_vanished_runs_are_failed(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    failed = _ask_running_trial(study)
    _publish_run(tmp_path / f"trial_{failed.number:03d}", make_cadence_rows(), failure=True)
    vanished = _ask_running_trial(study)
    hpo.reconcile_running(study, tmp_path, rd_band=0.05)
    states = {t.number: t.state for t in study.get_trials(deepcopy=False)}
    assert states[failed.number] == TrialState.FAIL
    assert states[vanished.number] == TrialState.FAIL
    assert TrialState.COMPLETE not in states.values()


def _fabricate_run(config_path: Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _publish_run(Path(cfg["output_dir"]), make_cadence_rows())


def _fabricate_bank(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    return path


def _sweep_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    argv = [
        "--teacher-checkpoint",
        str(tmp_path / "teacher.pt"),
        "--sweep-dir",
        str(tmp_path),
        "--n-trials",
        "2",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return hpo.build_parser().parse_args(argv)


def test_run_sweep_completes_n_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, spec in list(hpo.BANKS.items()):  # pre-existing banks: no dumps expected
        bank = _fabricate_bank(tmp_path / Path(spec.path).name)
        monkeypatch.setitem(
            hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank))
        )
    launched: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        launched.append(cmd)
        assert cmd[:3] == ["bash", "hpc/run.sh", "train"]
        assert cmd[-1] == "--skip-test"
        _fabricate_run(Path(cmd[3]))
        return 0

    monkeypatch.setattr(hpo, "run_command", fake_run)
    monkeypatch.setattr(
        hpo, "run_commands_parallel", lambda commands: pytest.fail("no dumps expected")
    )
    hpo.run_sweep(_sweep_args(tmp_path))
    study = hpo.build_study(tmp_path / "optuna.db")
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    assert len(complete) == 2
    assert [t.params for t in complete] == [dict(p) for p in hpo.ENQUEUED_PRIORS[:2]]
    assert all(t.user_attrs["surface"]["gs"] == pytest.approx(0.80) for t in complete)
    assert len(launched) == 2


def test_run_sweep_marks_failed_run_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, spec in list(hpo.BANKS.items()):
        bank = _fabricate_bank(tmp_path / Path(spec.path).name)
        monkeypatch.setitem(
            hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank))
        )
    calls = {"n": 0}

    def fake_run(cmd: list[str]) -> int:
        calls["n"] += 1
        cfg = yaml.safe_load(Path(cmd[3]).read_text(encoding="utf-8"))
        _publish_run(Path(cfg["output_dir"]), make_cadence_rows(), failure=calls["n"] == 1)
        return 0

    monkeypatch.setattr(hpo, "run_command", fake_run)
    monkeypatch.setattr(hpo, "run_commands_parallel", lambda commands: [])
    hpo.run_sweep(_sweep_args(tmp_path, n_trials=1))
    states = [t.state for t in hpo.build_study(tmp_path / "optuna.db").get_trials(deepcopy=False)]
    assert states.count(TrialState.FAIL) == 1
    assert states.count(TrialState.COMPLETE) == 1


def test_dump_missing_banks_shards_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _fabricate_bank(tmp_path / "kd_ctx_targets_breadth_first")
    monkeypatch.setitem(hpo.BANKS, "h2ns1", hpo.BankSpec(3, 2, 1, str(existing)))
    partial = tmp_path / "b_h2ns3"  # dir exists but no manifest: dump must run
    partial.mkdir()
    monkeypatch.setitem(hpo.BANKS, "h2ns3", hpo.BankSpec(3, 2, 3, str(partial)))
    monkeypatch.setitem(hpo.BANKS, "h2ns5", hpo.BankSpec(3, 2, 5, str(existing)))
    monkeypatch.setitem(hpo.BANKS, "h3ns3", hpo.BankSpec(3, 3, 3, str(existing)))
    parallel_calls: list[list[tuple[list[str], dict[str, str]]]] = []
    merges: list[list[str]] = []

    def fake_run_commands_parallel(
        commands: list[tuple[list[str], dict[str, str]]],
    ) -> list[int]:
        parallel_calls.append(commands)
        return [0] * len(commands)

    def fake_run_command(cmd: list[str]) -> int:
        merges.append(cmd)
        return 0

    monkeypatch.setattr(hpo, "run_commands_parallel", fake_run_commands_parallel)
    monkeypatch.setattr(hpo, "run_command", fake_run_command)
    hpo.dump_missing_banks(_sweep_args(tmp_path, dump_shards=2))
    assert len(parallel_calls) == 1 and len(parallel_calls[0]) == 2
    shard_cmd, shard_env = parallel_calls[0][0]
    assert shard_cmd[:3] == ["bash", "hpc/run.sh", "kd-targets"]
    assert "--contexts" in shard_cmd and "--row-shard" in shard_cmd
    for flag, value in (("--rw-step", "3"), ("--hops", "2"), ("--ns-rate", "3")):
        assert shard_cmd[shard_cmd.index(flag) + 1] == value
    assert shard_env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert len(merges) == 1 and "--merge" in merges[0]


def test_dump_missing_banks_raises_on_shard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(hpo.BANKS, "h2ns3", hpo.BankSpec(3, 2, 3, str(tmp_path / "missing")))
    for name in ("h2ns1", "h2ns5", "h3ns3"):
        spec = hpo.BANKS[name]
        existing = _fabricate_bank(tmp_path / f"bank_{name}")
        monkeypatch.setitem(
            hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(existing))
        )
    monkeypatch.setattr(hpo, "run_commands_parallel", lambda commands: [0, 1])
    with pytest.raises(RuntimeError):
        hpo.dump_missing_banks(_sweep_args(tmp_path, dump_shards=2))
