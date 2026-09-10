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
    for arm in ("prefix_static", "prefix_pair"):
        spec = struct_hpo.ARMS[arm]
        assert spec.study_name == arm
        assert spec.base_config == Path(f"configs/split_seed42/{arm}.yaml")
        assert spec.param_names == ("rank", "degree", "motif", "lr")
        assert spec.priors == (
            {"rank": 1.0, "degree": 0.1, "motif": 0.1, "lr": 1e-3},
            {"rank": 3.0, "degree": 0.5, "motif": 0.5, "lr": 3e-3},
        )
    bce = struct_hpo.ARMS["prefix_pair_bce"]
    assert bce.param_names == ("lr",)
    assert bce.base_config == Path("configs/split_seed42/prefix_pair_bce.yaml")
    assert bce.priors == ()


@pytest.mark.parametrize("arm", ["grand", "new", "prefix_static", "prefix_pair", "prefix_pair_bce"])
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


@pytest.mark.parametrize("arm", ["grand", "new", "prefix_static", "prefix_pair", "prefix_pair_bce"])
def test_materialize_overrides_only_struct_weights_and_output_dir(arm: str, tmp_path: Path) -> None:
    spec = struct_hpo.ARMS[arm]
    params = dict(spec.priors[0]) if spec.priors else {"lr": 2.5e-3}
    if "lr" in spec.param_names:
        # A prior's lr (1e-3) equals the base config's optim.lr, which would leave the lr-write
        # assertions below satisfied even if materialize_trial_config never touched optim. Use a
        # value that appears in no base file so the lr routing is actually exercised.
        params["lr"] = 2.5e-3
    path = struct_hpo.materialize_trial_config(spec.base_config, params, 7, tmp_path / "sweep")
    trial = yaml.safe_load(path.read_text())
    base = yaml.safe_load(spec.base_config.read_text())
    assert trial["output_dir"] == str(tmp_path / "sweep" / "trial_007")
    assert trial["struct"]["weights"]["bce"] == 1.0
    for name, value in params.items():
        if name == "lr":
            assert trial["optim"]["lr"] == value
            assert trial["optim"]["scheduler"]["max_lr"] == value
            assert "lr" not in trial["struct"]["weights"]
        else:
            assert trial["struct"]["weights"][name] == value
    for key, value in trial["struct"]["weights"].items():
        if key not in params and key != "bce":
            assert value == 0.0
    if "lr" in spec.param_names:
        assert trial["optim"]["lr"] != base["optim"]["lr"]
    trial["output_dir"] = base["output_dir"]
    trial["struct"]["weights"] = base["struct"]["weights"]
    trial["optim"] = base["optim"]
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


def test_bce_arm_priors_come_from_lr_center() -> None:
    args = argparse.Namespace(
        arm="prefix_pair_bce", base_config=None, sweep_dir=None, lr_center=3e-3
    )
    spec = struct_hpo.build_spec(args)
    assert len(spec.priors) == 3
    assert spec.priors[0] == pytest.approx({"lr": 1e-3})
    assert spec.priors[1] == pytest.approx({"lr": 3e-3})
    assert spec.priors[2] == pytest.approx({"lr": 9e-3})
    assert args.sweep_dir == Path("outputs/struct_hpo/prefix_pair_bce")


def test_bce_arm_requires_lr_center() -> None:
    args = argparse.Namespace(
        arm="prefix_pair_bce", base_config=None, sweep_dir=None, lr_center=None
    )
    with pytest.raises(ValueError, match="lr-center"):
        struct_hpo.build_spec(args)


def test_parser_defaults() -> None:
    args = struct_hpo.build_parser().parse_args(["--arm", "grand"])
    assert args.n_trials == 10
    assert not hasattr(args, "rd_band")
    assert args.base_config is None and args.sweep_dir is None
    assert args.lr_center is None
    with pytest.raises(SystemExit):
        struct_hpo.build_parser().parse_args([])


def test_driver_never_touches_frozen_paths() -> None:
    source = Path("src/experiments/struct_hpo.py").read_text()
    assert "autoresearch/" not in source
    assert "configs/sweep" not in source
