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

import math
from collections.abc import Mapping, Sequence
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


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config; only the five whitelisted keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    distill = dict(cfg["distill"])
    distill["w_rank"] = float(params["w_rank"])  # type: ignore[arg-type]
    distill["w_dist"] = float(params["w_dist"])  # type: ignore[arg-type]
    distill["margin"] = float(params["margin"])  # type: ignore[arg-type]
    distill["context_targets_path"] = BANKS[str(params["bank"])].path
    cfg["distill"] = distill
    DistillConfig.from_mapping(distill)
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


@dataclass(frozen=True)
class TrialOutcome:
    """Objectives, constraint, and telemetry surface of one completed trial."""

    gs: float
    geo_mmd: float
    constraint: float
    surface: dict[str, float]


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


def _constraints(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
    constraint = trial.user_attrs.get("constraint")
    if not isinstance(constraint, list) or len(constraint) != 1:
        return (float("inf"),)
    return (float(constraint[0]),)


def build_study(db_path: Path) -> optuna.Study:
    """Create-or-load the sweep study and (re-)enqueue the prior trials."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=0,
        multivariate=True,
        n_startup_trials=N_STARTUP_TRIALS,
        constraints_func=_constraints,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db_path}",
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )
    for params in ENQUEUED_PRIORS:
        study.enqueue_trial(dict(params), skip_if_exists=True)
    return study


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
                )
        study.tell(stale.number, state=TrialState.FAIL)
        if twin is not None:
            study.add_trial(twin)
