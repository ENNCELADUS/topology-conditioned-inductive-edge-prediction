from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pytest
import yaml
from src.experiments import struct_hpo
from src.experiments.kd_rank_strict_hpo import SweepSpec


def test_arm_table_matches_spec() -> None:
    grand = struct_hpo.ARMS["grand"]
    new = struct_hpo.ARMS["new"]
    assert grand.study_name == "struct_grand" and new.study_name == "struct_new"
    assert grand.base_config == Path("configs/struct_grand_breadth_first.yaml")
    assert new.base_config == Path("configs/struct_new_breadth_first.yaml")
    assert grand.priors == ({"gs": 0.70, "rd": 0.90}, {"gs": 0.35, "rd": 0.45})
    assert new.priors == (
        {"rank": 1.0, "degree": 0.1, "motif": 0.1},
        {"rank": 1.0, "degree": 0.03, "motif": 0.03},
    )
    assert grand.param_names == ("gs", "rd")
    assert new.param_names == ("rank", "degree", "motif")
    assert struct_hpo.N_STARTUP_TRIALS == 3
    assert struct_hpo.DEFAULT_TRIALS == 10


@pytest.mark.parametrize("arm", ["grand", "new"])
def test_suggest_stays_inside_the_boxes(arm: str) -> None:
    spec = struct_hpo.ARMS[arm]
    study = optuna.create_study(
        directions=["maximize", "minimize"], sampler=optuna.samplers.RandomSampler(seed=0)
    )
    boxes = struct_hpo.SEARCH_BOXES[arm]
    for _ in range(50):
        params = spec.suggest(study.ask())
        assert set(params) == set(spec.param_names)
        for name, value in params.items():
            low, high = boxes[name]
            assert low <= float(value) <= high  # type: ignore[arg-type]


@pytest.mark.parametrize("arm", ["grand", "new"])
def test_materialize_overrides_only_struct_weights_and_output_dir(arm: str, tmp_path: Path) -> None:
    spec = struct_hpo.ARMS[arm]
    params = dict(spec.priors[0])
    path = struct_hpo.materialize_trial_config(spec.base_config, params, 7, tmp_path / "sweep")
    trial = yaml.safe_load(path.read_text())
    base = yaml.safe_load(spec.base_config.read_text())
    assert trial["output_dir"] == str(tmp_path / "sweep" / "trial_007")
    assert trial["struct"]["weights"]["bce"] == 1.0
    for name, value in params.items():
        assert trial["struct"]["weights"][name] == value
    for key, value in trial["struct"]["weights"].items():
        if key not in params and key != "bce":
            assert value == 0.0
    trial["output_dir"] = base["output_dir"]
    trial["struct"]["weights"] = base["struct"]["weights"]
    assert trial == base


def test_materialize_rejects_illegal_weights(tmp_path: Path) -> None:
    spec = struct_hpo.ARMS["new"]
    with pytest.raises(ValueError, match="non-negative"):
        struct_hpo.materialize_trial_config(
            spec.base_config, {"rank": -1.0, "degree": 0.1, "motif": 0.1}, 0, tmp_path
        )


def test_build_spec_binds_the_arm_and_fills_defaults() -> None:
    args = argparse.Namespace(
        arm="new", base_config=None, sweep_dir=None, n_trials=10, rd_band=0.05
    )
    spec = struct_hpo.build_spec(args)
    assert isinstance(spec, SweepSpec)
    assert spec.study_name == "struct_new"
    assert spec.n_startup_trials == 3
    assert spec.materialize is struct_hpo.materialize_trial_config
    assert args.base_config == Path("configs/struct_new_breadth_first.yaml")
    assert args.sweep_dir == Path("outputs/struct_hpo/new")


def test_parser_defaults() -> None:
    args = struct_hpo.build_parser().parse_args(["--arm", "grand"])
    assert args.n_trials == 10
    assert not hasattr(args, "rd_band")
    assert args.base_config is None and args.sweep_dir is None
    with pytest.raises(SystemExit):
        struct_hpo.build_parser().parse_args([])


def test_driver_never_touches_frozen_paths() -> None:
    source = Path("src/experiments/struct_hpo.py").read_text()
    assert "autoresearch/" not in source
    assert "configs/sweep" not in source
