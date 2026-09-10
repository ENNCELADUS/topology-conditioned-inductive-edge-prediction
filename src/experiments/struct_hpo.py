"""Structural-arm HPO using the shared three-objective search and five-metric rank protocol."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna
import yaml

from src.distill.struct_config import StructConfig
from src.experiments.kd_rank_strict_hpo import SweepSpec, run_sweep

N_STARTUP_TRIALS = 3
DEFAULT_TRIALS = 10

SEARCH_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "grand": {"gs": (0.1, 2.0), "rd": (0.1, 2.0)},
    "new": {"rank": (0.1, 3.0), "degree": (0.01, 1.0), "motif": (0.01, 1.0)},
}


@dataclass(frozen=True)
class ArmSpec:
    """One structural arm: study identity, base config, priors, and search space."""

    study_name: str
    base_config: Path
    priors: tuple[dict[str, object], ...]
    param_names: tuple[str, ...]
    suggest: Callable[[optuna.Trial], dict[str, object]]


def _suggest_for(arm: str) -> Callable[[optuna.Trial], dict[str, object]]:
    boxes = SEARCH_BOXES[arm]

    def suggest(trial: optuna.Trial) -> dict[str, object]:
        return {
            name: float(trial.suggest_float(name, low, high, log=True))
            for name, (low, high) in boxes.items()
        }

    return suggest


ARMS: dict[str, ArmSpec] = {
    "grand": ArmSpec(
        study_name="struct_grand",
        base_config=Path("configs/struct_grand_breadth_first.yaml"),
        priors=({"gs": 0.70, "rd": 0.90}, {"gs": 0.35, "rd": 0.45}),
        param_names=("gs", "rd"),
        suggest=_suggest_for("grand"),
    ),
    "new": ArmSpec(
        study_name="struct_new",
        base_config=Path("configs/struct_new_breadth_first.yaml"),
        priors=(
            {"rank": 1.0, "degree": 0.1, "motif": 0.1},
            {"rank": 1.0, "degree": 0.03, "motif": 0.03},
        ),
        param_names=("rank", "degree", "motif"),
        suggest=_suggest_for("new"),
    ),
}


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config: base + searched ``struct.weights`` + ``output_dir``.

    Every searched weight replaces its base value; ``bce`` and unsearched keys
    keep the base file's values.

    Raises:
        ValueError: If the resulting ``struct`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    weights = {**cfg["struct"]["weights"], **{k: float(v) for k, v in params.items()}}  # type: ignore[arg-type]
    cfg["struct"] = {**cfg["struct"], "weights": weights}
    StructConfig.from_mapping(cfg["struct"])
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _no_prepare(args: argparse.Namespace) -> None:
    """Structural arms need no bank; nothing to prepare."""


def build_spec(args: argparse.Namespace) -> SweepSpec:
    """Bind the chosen arm into a ``SweepSpec`` and fill the arm's default paths."""
    arm = ARMS[str(args.arm)]
    if args.base_config is None:
        args.base_config = arm.base_config
    if args.sweep_dir is None:
        args.sweep_dir = Path("outputs/struct_hpo") / str(args.arm)
    return SweepSpec(
        study_name=arm.study_name,
        n_startup_trials=N_STARTUP_TRIALS,
        priors=arm.priors,
        param_names=arm.param_names,
        suggest=arm.suggest,
        materialize=materialize_trial_config,
        prepare=_no_prepare,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.struct_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args, build_spec(args))


if __name__ == "__main__":
    main()
