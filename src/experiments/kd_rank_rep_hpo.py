"""Unattended Optuna sweep for the ``kd_rank_rep`` double-KD arm.

Same ask-and-tell constrained MO-TPE loop, objectives (GS max, geometric-mean
MMD ratio min), and ``|log RD|`` soft constraint as the strict-LLP kd_rank
sweep (`src.experiments.kd_rank_strict_hpo`); this study searches only the
three loss weights and inherits the kd_rank winner's context bank and margin
through ``--bank``/``--margin``. Winner selection stays the frozen
five-metric undominated verdict plus the human pick.
Spec: ``docs/superpowers/specs/2026-09-04-kd-rank-rep-double-kd-design.md``.
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import optuna

from src.experiments.kd_rank_strict_hpo import BANKS, SweepSpec, _write_trial_config, run_sweep

STUDY_NAME = "kd_rank_rep_strict"
N_STARTUP_TRIALS = 4

ENQUEUED_PRIORS: tuple[dict[str, object], ...] = (
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 1.0},
    {"w_rank": 0.1, "w_dist": 10.0, "w_rep": 10.0},
    {"w_rank": 1.0, "w_dist": 1.0, "w_rep": 1.0},
)


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Draw one point of the three-weight log box (enqueued values pass through)."""
    return {
        "w_rank": float(trial.suggest_float("w_rank", 0.01, 1.0, log=True)),
        "w_dist": float(trial.suggest_float("w_dist", 0.1, 100.0, log=True)),
        "w_rep": float(trial.suggest_float("w_rep", 0.01, 100.0, log=True)),
    }


def materialize_trial_config(
    base_config: Path,
    params: Mapping[str, object],
    trial_number: int,
    sweep_dir: Path,
    *,
    bank: str,
    margin: float,
) -> Path:
    """Write trial ``trial_number``'s config; only the whitelisted distill keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    overrides = {
        "w_rank": float(params["w_rank"]),  # type: ignore[arg-type]
        "w_dist": float(params["w_dist"]),  # type: ignore[arg-type]
        "w_rep": float(params["w_rep"]),  # type: ignore[arg-type]
        "margin": float(margin),
        "context_targets_path": BANKS[bank].path,
    }
    return _write_trial_config(base_config, overrides, trial_number, sweep_dir)


def require_bank(args: argparse.Namespace) -> None:
    """Fail closed before any training budget if the frozen bank is not on disk.

    Raises:
        RuntimeError: If the bank's manifest is missing (dump it with the
            kd_rank sweep driver first).
    """
    path = Path(BANKS[args.bank].path)
    if not (path / "manifest.json").exists():
        raise RuntimeError(f"context bank {args.bank} has no manifest at {path}")


def build_spec(args: argparse.Namespace) -> SweepSpec:
    """Bind the frozen bank and margin into the sweep specification."""
    return SweepSpec(
        study_name=STUDY_NAME,
        n_startup_trials=N_STARTUP_TRIALS,
        priors=ENQUEUED_PRIORS,
        param_names=("w_rank", "w_dist", "w_rep"),
        suggest=suggest_params,
        materialize=functools.partial(materialize_trial_config, bank=args.bank, margin=args.margin),
        prepare=require_bank,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.kd_rank_rep_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/autoresearch/kd_rank_rep.yaml")
    )
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/b1_kd_rank_rep_hpo"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--rd-band", type=float, default=0.05)
    parser.add_argument("--bank", choices=sorted(BANKS), default="h2ns3")
    parser.add_argument("--margin", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args, build_spec(args))


if __name__ == "__main__":
    main()
