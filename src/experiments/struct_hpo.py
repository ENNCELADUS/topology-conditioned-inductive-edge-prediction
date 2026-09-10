"""Structural-arm HPO using the shared three-objective search and five-metric rank protocol.

Six arms: the frozen-trunk structural baselines ``grand`` and ``new``, and three prefix-tuning
arms (frozen ``prefix_base`` trunk, trainable gated KV prefix) that additionally search a
learning rate. ``prefix_static`` and ``prefix_pair`` search ``rank``/``degree``/``motif`` (the
``struct_new`` weights) together with ``lr``; a searched ``lr`` is written to both ``optim.lr``
and ``optim.scheduler.max_lr``, never into ``struct.weights``. ``prefix_pair_bce`` is the BCE-only
control for the ``prefix_pair`` conditioning: it searches only ``lr``, and its three priors are
derived from ``--lr-center X`` (the ``prefix_pair`` winner's lr) as ``(X/3, X, 3X)``. Launch it
with ``--arm prefix_pair_bce --n-trials 3 --lr-center <prefix_pair winner lr>``: Optuna's
``Study.ask()`` drains the three enqueued (WAITING) priors before it ever consults the sampler, so
with ``--n-trials 3`` all three trials are exactly those priors.
"""

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

LR_BOX: tuple[float, float] = (1e-4, 1e-2)
PREFIX_BOX: dict[str, tuple[float, float]] = {
    "rank": (0.1, 3.0),
    "degree": (0.01, 1.0),
    "motif": (0.01, 1.0),
    "lr": LR_BOX,
}
SEARCH_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "grand": {"gs": (0.1, 2.0), "rd": (0.1, 2.0)},
    "new": {"rank": (0.1, 3.0), "degree": (0.01, 1.0), "motif": (0.01, 1.0)},
    "prefix_static": dict(PREFIX_BOX),
    "prefix_pair": dict(PREFIX_BOX),
    "prefix_pair_bce": {"lr": LR_BOX},
}
PREFIX_PRIORS: tuple[dict[str, object], ...] = (
    {"rank": 1.0, "degree": 0.1, "motif": 0.1, "lr": 1e-3},
    {"rank": 3.0, "degree": 0.5, "motif": 0.5, "lr": 3e-3},
)


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
    "prefix_static": ArmSpec(
        study_name="prefix_static",
        base_config=Path("configs/split_seed42/prefix_static.yaml"),
        priors=PREFIX_PRIORS,
        param_names=("rank", "degree", "motif", "lr"),
        suggest=_suggest_for("prefix_static"),
    ),
    "prefix_pair": ArmSpec(
        study_name="prefix_pair",
        base_config=Path("configs/split_seed42/prefix_pair.yaml"),
        priors=PREFIX_PRIORS,
        param_names=("rank", "degree", "motif", "lr"),
        suggest=_suggest_for("prefix_pair"),
    ),
    "prefix_pair_bce": ArmSpec(
        study_name="prefix_pair_bce",
        base_config=Path("configs/split_seed42/prefix_pair_bce.yaml"),
        priors=(),
        param_names=("lr",),
        suggest=_suggest_for("prefix_pair_bce"),
    ),
}


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config: base + searched params + ``output_dir``.

    Every searched weight replaces its base ``struct.weights`` value; ``bce`` and unsearched
    keys keep the base file's values. A searched ``lr`` is written to ``optim.lr`` and, when the
    base config has an ``optim.scheduler.max_lr``, to that too; it never enters ``struct.weights``.

    Raises:
        ValueError: If the resulting ``struct`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    lr = params.get("lr")
    weight_params = {k: float(v) for k, v in params.items() if k != "lr"}  # type: ignore[arg-type]
    weights = {**cfg["struct"]["weights"], **weight_params}
    cfg["struct"] = {**cfg["struct"], "weights": weights}
    if lr is not None:
        cfg["optim"]["lr"] = float(lr)  # type: ignore[arg-type]
        scheduler = cfg["optim"].get("scheduler")
        if isinstance(scheduler, dict) and "max_lr" in scheduler:
            scheduler["max_lr"] = float(lr)  # type: ignore[arg-type]
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
    priors = arm.priors
    if str(args.arm) == "prefix_pair_bce":
        center = getattr(args, "lr_center", None)
        if center is None:
            raise ValueError(
                "--lr-center (the prefix_pair winner's lr) is required for prefix_pair_bce"
            )
        priors = ({"lr": float(center) / 3.0}, {"lr": float(center)}, {"lr": float(center) * 3.0})
    return SweepSpec(
        study_name=arm.study_name,
        n_startup_trials=N_STARTUP_TRIALS,
        priors=priors,
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
    parser.add_argument(
        "--lr-center",
        type=float,
        default=None,
        help=(
            "prefix_pair_bce only: the prefix_pair winner's lr; the three priors are "
            "(X/3, X, 3X), and with --n-trials 3 Study.ask() drains them as WAITING "
            "trials before sampling, so they are the arm's only trials."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args, build_spec(args))


if __name__ == "__main__":
    main()
