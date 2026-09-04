"""Unattended Optuna sweep for the strict-LLP ``kd_rank`` arm.

Runs on the H20 container: an ask-and-tell TPE loop proposes
``(w_rank, w_dist, context bank, margin)``, launches one grid-protocol
training per trial through ``hpc/run.sh train --skip-test``, and scores the
cadence-2 V_val surface as (GS max, geometric-mean MMD ratio min) with an
``|log RD|`` soft constraint. The feasible Pareto front is advisory: the
recorded winner comes from the frozen five-metric undominated verdict.
Spec: ``docs/superpowers/specs/2026-09-01-kd-rank-strict-llp-optuna-hpo-design.md``.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna
import yaml
from optuna.trial import TrialState

from src.autoresearch.metrics_io import RunFailure, RunMetrics, read_run
from src.distill.config import DistillConfig


@dataclass(frozen=True)
class BankSpec:
    """One frozen context bank: sampler composition and artifact path."""

    rw_step: int
    hops: int
    ns_rate: int
    path: str


BANKS: dict[str, BankSpec] = {
    "h2ns1": BankSpec(3, 2, 1, "outputs/distill/kd_ctx_targets_breadth_first"),
    "h2ns3": BankSpec(3, 2, 3, "outputs/distill/kd_ctx_targets_breadth_first_h2ns3"),
    "h2ns5": BankSpec(3, 2, 5, "outputs/distill/kd_ctx_targets_breadth_first_h2ns5"),
    "h3ns3": BankSpec(3, 3, 3, "outputs/distill/kd_ctx_targets_breadth_first_h3ns3"),
}

ENQUEUED_PRIORS: tuple[dict[str, object], ...] = (
    {"w_rank": 1.0, "w_dist": 1.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.01, "w_dist": 10.0, "bank": "h2ns5", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 100.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h3ns3", "margin": 0.1},
)


def _write_trial_config(
    base_config: Path, distill_overrides: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config: base + ``distill_overrides`` + ``output_dir``.

    Raises:
        ValueError: If the resulting ``distill`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    distill = {**cfg["distill"], **distill_overrides}
    cfg["distill"] = distill
    DistillConfig.from_mapping(distill)
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s kd_rank config; only the five whitelisted keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    overrides = {
        "w_rank": float(params["w_rank"]),  # type: ignore[arg-type]
        "w_dist": float(params["w_dist"]),  # type: ignore[arg-type]
        "margin": float(params["margin"]),  # type: ignore[arg-type]
        "context_targets_path": BANKS[str(params["bank"])].path,
    }
    return _write_trial_config(base_config, overrides, trial_number, sweep_dir)


@dataclass(frozen=True)
class TrialOutcome:
    """Objectives, constraint, and telemetry surface of one completed trial."""

    gs: float
    geo_mmd: float
    constraint: float
    surface: dict[str, float]


@dataclass(frozen=True)
class SweepSpec:
    """One arm's sweep: study identity, priors, search space, config writer, and prep step."""

    study_name: str
    n_startup_trials: int
    priors: tuple[dict[str, object], ...]
    param_names: tuple[str, ...]
    suggest: Callable[[optuna.Trial], dict[str, object]]
    materialize: Callable[[Path, Mapping[str, object], int, Path], Path]
    prepare: Callable[[argparse.Namespace], None]


def trial_outcome(run_dir: Path, rd_band: float) -> TrialOutcome:
    """Score one run at its cadence-2 selected epoch.

    Raises:
        RunFailure: If the run wrote ``failure.json``.
        ValueError: On missing/non-finite metrics or a non-positive MMD ratio.
    """
    run: RunMetrics = read_run(run_dir, topology_every=2)
    topo = run.topology
    ratios = (topo.degree_mmd, topo.clustering_mmd, topo.spectral_mmd)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"{run_dir}: MMD ratios must be positive, got {ratios}")
    geo_mmd = math.exp(sum(math.log(ratio) for ratio in ratios) / 3.0)
    surface = {
        "auprc": run.auprc,
        "gs": topo.gs,
        "rd": topo.rd,
        "degree_mmd": topo.degree_mmd,
        "clustering_mmd": topo.clustering_mmd,
        "spectral_mmd": topo.spectral_mmd,
        "selected_epoch": float(run.selected_epoch),
    }
    return TrialOutcome(topo.gs, geo_mmd, abs(math.log(topo.rd)) - rd_band, surface)


STUDY_NAME = "kd_rank_strict_llp"
N_STARTUP_TRIALS = 6
MAX_CONSECUTIVE_FAILURES = 3


def _constraints(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
    constraint = trial.user_attrs.get("constraint")
    if not isinstance(constraint, list) or len(constraint) != 1:
        return (float("inf"),)
    return (float(constraint[0]),)


def build_study(
    db_path: Path, *, study_name: str = STUDY_NAME, n_startup_trials: int = N_STARTUP_TRIALS
) -> optuna.Study:
    """Create-or-load a sweep study (kd_rank's by default)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=0,
        multivariate=True,
        n_startup_trials=n_startup_trials,
        constraints_func=_constraints,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_priors(
    study: optuna.Study, priors: Sequence[Mapping[str, object]] = ENQUEUED_PRIORS
) -> None:
    """Enqueue every prior that has no waiting, running, or completed trial.

    A prior whose trial failed is re-enqueued, so a fixed environment gets
    the full prior set on restart (``skip_if_exists`` would treat the failed
    twin as done).
    """
    live = (TrialState.WAITING, TrialState.RUNNING, TrialState.COMPLETE)
    seen = [
        t.system_attrs.get("fixed_params", t.params)
        for t in study.get_trials(deepcopy=False, states=live)
    ]
    for params in priors:
        if dict(params) not in seen:
            study.enqueue_trial(dict(params))


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Draw one point of the spec's search space (enqueued values pass through)."""
    return {
        "w_rank": float(trial.suggest_float("w_rank", 0.01, 1.0, log=True)),
        "w_dist": float(trial.suggest_float("w_dist", 0.1, 100.0, log=True)),
        "bank": str(trial.suggest_categorical("bank", sorted(BANKS))),
        "margin": float(trial.suggest_categorical("margin", [0.05, 0.1, 0.2])),
    }


def reconcile_running(study: optuna.Study, sweep_dir: Path, rd_band: float) -> None:
    """Resolve trials left RUNNING by an interrupted driver.

    The stale trial is always failed; a run that actually completed is
    re-added as a COMPLETE twin with its real objectives and constraint.
    """
    for stale in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        run_dir = sweep_dir / f"trial_{stale.number:03d}"
        twin: optuna.trial.FrozenTrial | None = None
        if (run_dir / "complete.json").exists():
            try:
                outcome = trial_outcome(run_dir, rd_band)
            except RunFailure:
                outcome = None
            if outcome is not None:
                twin = optuna.trial.create_trial(
                    params=dict(stale.params),
                    distributions=dict(stale.distributions),
                    values=[outcome.gs, outcome.geo_mmd],
                    user_attrs={"constraint": [outcome.constraint], "surface": outcome.surface},
                    # "constraints" is where samplers materialize constraints_func
                    # results; add_trial bypasses after_trial, so without it the
                    # constrained TPE and best_trials would treat the twin as
                    # infeasible.
                    system_attrs={"constraints": [outcome.constraint]},
                )
        study.tell(stale.number, state=TrialState.FAIL)
        if twin is not None:
            study.add_trial(twin)


_THREAD_CAPS = {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}


def run_command(cmd: list[str]) -> int:
    """Run one foreground container command with the H20 thread caps."""
    return subprocess.run(cmd, env={**os.environ, **_THREAD_CAPS}, check=False).returncode


def run_commands_parallel(commands: list[tuple[list[str], dict[str, str]]]) -> list[int]:
    """Run commands concurrently; each tuple is (argv, extra env)."""
    procs = [
        subprocess.Popen(cmd, env={**os.environ, **_THREAD_CAPS, **extra})
        for cmd, extra in commands
    ]
    return [proc.wait() for proc in procs]


def _dump_cmd(args: argparse.Namespace, spec: BankSpec) -> list[str]:
    return [
        "bash",
        "hpc/run.sh",
        "kd-targets",
        "--contexts",
        "--config",
        str(args.base_config),
        "--checkpoint",
        str(args.teacher_checkpoint),
        "--output",
        spec.path,
        "--rw-step",
        str(spec.rw_step),
        "--hops",
        str(spec.hops),
        "--ns-rate",
        str(spec.ns_rate),
    ]


def dump_missing_banks(args: argparse.Namespace) -> None:
    """Dump every context bank whose artifact is absent (sharded, then merged).

    Raises:
        RuntimeError: If any shard or merge exits nonzero (fail-closed
            before any training budget is spent).
    """
    for name in sorted(BANKS):
        spec = BANKS[name]
        # A partial dump leaves the directory (shards, f0_cache.pt) without the
        # manifest, which the artifact writer emits last.
        if (Path(spec.path) / "manifest.json").exists():
            continue
        shards = [
            (
                _dump_cmd(args, spec)
                + ["--device", "cuda", "--row-shard", f"{index}/{args.dump_shards}"],
                {"CUDA_VISIBLE_DEVICES": str(index)},
            )
            for index in range(args.dump_shards)
        ]
        codes = run_commands_parallel(shards)
        if any(code != 0 for code in codes):
            raise RuntimeError(f"bank {name}: shard exit codes {codes}")
        merge_code = run_command(
            _dump_cmd(args, spec) + ["--merge", "--row-shard", f"0/{args.dump_shards}"]
        )
        if merge_code != 0:
            raise RuntimeError(f"bank {name}: merge exited {merge_code}")


def _n_complete(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


KD_RANK_SPEC = SweepSpec(
    study_name=STUDY_NAME,
    n_startup_trials=N_STARTUP_TRIALS,
    priors=ENQUEUED_PRIORS,
    param_names=("w_rank", "w_dist", "bank", "margin"),
    suggest=suggest_params,
    materialize=materialize_trial_config,
    prepare=dump_missing_banks,
)


def run_sweep(args: argparse.Namespace, spec: SweepSpec = KD_RANK_SPEC) -> None:
    """Drive the whole sweep: reconcile, prepare banks, ask/tell until budget."""
    study = build_study(
        args.sweep_dir / "optuna.db",
        study_name=spec.study_name,
        n_startup_trials=spec.n_startup_trials,
    )
    reconcile_running(study, args.sweep_dir, args.rd_band)
    enqueue_priors(study, spec.priors)
    spec.prepare(args)
    failures = 0
    while _n_complete(study) < args.n_trials:
        trial = study.ask()
        params = spec.suggest(trial)
        config_path = spec.materialize(args.base_config, params, trial.number, args.sweep_dir)
        run_command(["bash", "hpc/run.sh", "train", str(config_path), "--skip-test"])
        run_dir = args.sweep_dir / f"trial_{trial.number:03d}"
        try:
            outcome = trial_outcome(run_dir, args.rd_band)
        except RunFailure:
            study.tell(trial, state=TrialState.FAIL)
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{failures} consecutive failed trials, last {run_dir}: fix the "
                    "environment and restart the sweep"
                ) from None
            continue
        failures = 0
        trial.set_user_attr("constraint", [outcome.constraint])
        trial.set_user_attr("surface", outcome.surface)
        study.tell(trial, values=[outcome.gs, outcome.geo_mmd])
    print_report(study, spec.param_names)


def print_report(
    study: optuna.Study, param_names: Sequence[str] = ("w_rank", "w_dist", "bank", "margin")
) -> None:
    """Print the full trial table, then the feasible Pareto front."""
    columns = [
        "auprc",
        "gs",
        "rd",
        "degree_mmd",
        "clustering_mmd",
        "spectral_mmd",
        "selected_epoch",
    ]
    print("number state " + " ".join(param_names) + " " + " ".join(columns))  # noqa: T201 -- CLI report goes to stdout
    for t in study.get_trials(deepcopy=False):
        surface = t.user_attrs.get("surface", {})
        params = " ".join(str(t.params.get(name, "-")) for name in param_names)
        values = " ".join(f"{surface[c]:.4f}" if c in surface else "-" for c in columns)
        print(f"{t.number} {t.state.name} {params} {values}")  # noqa: T201 -- CLI report goes to stdout
    front = ", ".join(str(t.number) for t in study.best_trials)
    print(f"feasible Pareto front (advisory): trials [{front}]")  # noqa: T201 -- CLI report goes to stdout


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.kd_rank_strict_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/autoresearch/kd_rank.yaml")
    )
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/b1_kd_rank_strict_hpo"))
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--rd-band", type=float, default=0.05)
    parser.add_argument("--dump-shards", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
