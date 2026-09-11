"""Twelve-trial V_val-only search over joint logit and representation KD weights."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import optuna

from src.experiments.kd_rank_strict_hpo import (
    SweepSpec,
    _write_trial_config,
    row_bank_path,
    run_sweep,
)


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Search the existing single-arm weight ranges on a log scale."""
    return {name: trial.suggest_float(name, 0.01, 100.0, log=True) for name in ("w_logit", "w_rep")}


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Keep the split, bank, optimizer and model fixed; vary the two KD weights."""
    return _write_trial_config(base_config, params, trial_number, sweep_dir)


def require_rows(args: argparse.Namespace) -> None:
    """Require the existing training row bank; never dump validation targets."""
    path = row_bank_path(args.base_config) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)


SPEC = SweepSpec(
    study_name="kd_logit_rep",
    n_startup_trials=4,
    priors=(
        {"w_logit": 10.0, "w_rep": 0.01},
        {"w_logit": 10.0, "w_rep": 0.1},
        {"w_logit": 10.0, "w_rep": 1.0},
        {"w_logit": 1.0, "w_rep": 0.1},
    ),
    param_names=("w_logit", "w_rep"),
    suggest=suggest_params,
    materialize=materialize_trial_config,
    prepare=require_rows,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared Optuna search and publish its V_val-selected winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/split_seed42/sweep/kd_logit_w10.yaml")
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=12)
    run_sweep(parser.parse_args(argv), SPEC)


if __name__ == "__main__":
    main()
