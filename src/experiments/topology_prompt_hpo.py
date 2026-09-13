"""Two-stage prompt studies declared in docs/03-experiments.md §6.

Every invocation takes an explicit inherited base config. Trials publish without
test; replicas, teacher choice and the final test remain separate operations.
This study-specific constrained two-objective search does not change the shared
three-objective HPO protocol of the other research arms.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna
import yaml
from optuna.trial import FrozenTrial, TrialState

from src.autoresearch.metrics_io import read_metric_rows
from src.eval.checkpoint_selection import (
    SELECTION_RULE,
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)
from src.experiments.kd_rank_strict_hpo import enqueue_priors, run_command

# Paths are the public configuration names. Float intervals are logarithmic
# except alpha, whose declared box includes zero.
SPACES: dict[str, dict[str, tuple[float, float] | list[float | int | None]]] = {
    "T1": {
        "model.config.topo_prompt.corruption.prob": [0.25, 0.5, 0.75],
        "model.config.topo_prompt.corruption.shrink_min": (0.1, 0.7),
        "model.config.topo_prompt.corruption.sigma_max": (0.1, 1.0),
    },
    "T2": {
        "struct.nodes": [20, 40, 60],
        "struct.weights.rank": (0.1, 3.0),
        "struct.weights.degree": (0.01, 1.0),
        "struct.weights.motif": (0.01, 1.0),
    },
    "S1": {
        "optim.groups.generator.max_lr": (1e-4, 1e-3),
        "optim.groups.interface.max_lr": (3e-5, 3e-4),
    },
    "S2": {
        "model.config.coord_gen.w_coord": (0.3, 3.0),
        "model.config.coord_gen.kd_alpha": (0.0, 0.7),
        "model.config.coord_gen.w_anchor": (0.1, 3.0),
        "model.config.coord_gen.anchor_temperature": [0.5, 1.0, 2.0],
        "struct.scale": (0.3, 3.0),
    },
    "S3_size": {"struct.nodes": [20, 40, 60]},
    "S3_frequency": {"struct.subgraphs_per_epoch": [None, 0.5]},
}
BUDGETS = {"T1": 6, "T2": 6, "S1": 4, "S2": 10, "S3_size": 3, "S3_frequency": 2}
PRIOR_VALUES: dict[str, tuple[tuple[float | int | None, ...], ...]] = {
    "T1": ((0.5, 0.3, 0.5), (0.75, 0.1, 1.0), (0.25, 0.5, 0.25)),
    "T2": ((40, 1.0, 0.1, 0.1), (20, 1.0, 0.1, 0.1), (60, 1.0, 0.1, 0.1)),
    "S1": ((3e-4, 1e-4), (1e-3, 1e-4), (1e-4, 3e-5), (3e-4, 3e-4)),
    "S2": ((1.0, 0.5, 1.0, 1.0, 1.0), (0.3, 0.7, 3.0, 1.0, 1.0), (1.0, 0.5, 1.0, 1.0, 3.0)),
    "S3_size": ((20,), (40,), (60,)),
    "S3_frequency": ((None,), (0.5,)),
}


def suggest(study: str, trial: optuna.Trial) -> dict[str, object]:
    """Draw only the knobs declared for this study."""
    return {
        key: trial.suggest_categorical(key, box)
        if isinstance(box, list)
        else trial.suggest_float(key, *box, log=box[0] > 0)
        for key, box in SPACES[study].items()
    }


def materialize(base: Path, params: Mapping[str, object], output: Path) -> Path:
    """Inherit the supplied winner and replace the current study's knobs."""
    cfg = yaml.safe_load(base.read_text())
    if any(key.startswith("struct.") for key in params) and "struct" not in cfg:
        cfg["struct"] = {
            "nodes": 40,
            "background_nodes": 8,
            "mix": {"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
            "subgraphs_per_epoch": None,
            "weights": {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1},
        }
    for key_path, value in params.items():
        node = cfg
        keys = key_path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    cfg["output_dir"] = str(output)
    if cfg["model"]["family"] == "v3_1_coord_gen":
        cfg["optim"]["epochs"] = 15
        cfg["eval"]["patience"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / "configs" / f"{output.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _finite(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def outcome(run: Path, study: str, baseline_auprc: float | None) -> dict[str, Any]:
    """Read clean V_val only, plus the explicitly declared diagnostic constraints."""
    if (run / "failure.json").exists():
        raise ValueError(f"failed trial: {run}")
    diagnostic = study.startswith("T")
    sentinel = "diagnostic_complete" if diagnostic else "complete"
    if json.loads((run / f"{sentinel}.json").read_text())["status"] != sentinel:
        raise ValueError(f"incomplete publication: {run}")
    epoch = int(json.loads((run / "run_metadata.json").read_text())["selected_epoch"])
    rows = read_metric_rows(run / "metrics.jsonl")
    if not diagnostic and sorted(r["epoch"] for r in rows) != list(range(1, 16)):
        raise ValueError("Stage II trials require the full 15-epoch budget")
    row = next(row for row in rows if row["epoch"] == epoch)
    surface = {
        name: _finite(row, key)
        for name, key in {
            "auprc": "val_auprc",
            "gs": "val_gs_bfs",
            "rd": "val_rd_bfs",
            "degree_mmd": "val_degree_mmd_ratio",
            "clustering_mmd": "val_clustering_mmd_ratio",
            "spectral_mmd": "val_spectral_mmd_ratio",
            "threshold": "val_threshold",
        }.items()
    }
    ratios = [surface[f"{field}_mmd"] for field in ("degree", "clustering", "spectral")]
    if surface["rd"] <= 0 or min(ratios) < 0:
        raise ValueError("RD must be positive and MMD ratios nonnegative")
    constraints = [abs(math.log(surface["rd"])) - 0.05]
    selectable = True
    diagnostics: dict[str, float] = {}
    if study in {"T1", "T2"}:
        p95 = _finite(row, "sensitivity_noise_logit_change_p95")
        diagnostics["sensitivity_noise_logit_change_p95"] = p95
        selectable = p95 <= 2.0
    if study == "S1":
        overlaps = [_finite(r, "stability_edge_jaccard") for r in rows if r["epoch"] > 1]
        if not overlaps:
            raise ValueError("S1 needs consecutive-epoch stability diagnostics")
        diagnostics["minimum_edge_jaccard"] = min(overlaps)
        selectable = min(overlaps) >= 0.7
    if study == "S2":
        if baseline_auprc is None or not math.isfinite(baseline_auprc):
            raise ValueError("S2 requires the prefix_base V_val classification AUPRC")
        constraints.append(baseline_auprc - 0.01 - surface["auprc"])
        selectable = constraints[-1] <= 0
    return {
        "surface": {**surface, "selected_epoch": epoch},
        "constraints": constraints,
        "selectable": selectable,
        "diagnostics": diagnostics,
        "values": [surface["gs"], math.prod(ratios) ** (1 / 3)],
        "mmd_spread": {
            field: {
                "min": min(_finite(r, f"val_{field}_mmd_ratio") for r in rows),
                "max": max(_finite(r, f"val_{field}_mmd_ratio") for r in rows),
            }
            for field in ("degree", "clustering", "spectral")
        },
    }


def select_winner(trials: Sequence[FrozenTrial]) -> FrozenTrial | None:
    """Five-rank selection after study-specific filters; RD remains a soft constraint."""
    eligible = [t for t in trials if t.state == TrialState.COMPLETE and t.user_attrs["selectable"]]
    candidates = []
    for trial in eligible:
        s = trial.user_attrs["surface"]
        candidates.append(
            CheckpointCandidate(
                epoch=trial.number + 1,
                auprc=s["auprc"],
                topology=TopologyValidationMetrics(
                    s["gs"], s["rd"], s["degree_mmd"], s["clustering_mmd"], s["spectral_mmd"]
                ),
            )
        )
    winner = select_checkpoint(candidates)
    return next((t for t in eligible if winner and t.number + 1 == winner.epoch), None)


def _tell(study: optuna.Study, trial: optuna.Trial, result: Mapping[str, Any]) -> None:
    for key, value in result.items():
        if key != "values":
            trial.set_user_attr(key, value)
    study.tell(trial, result["values"])


def build_parser() -> argparse.ArgumentParser:
    """Build the study launcher CLI; nothing runs when importing this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=SPACES, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="explicit config inheriting preceding winners",
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--prefix-base-auprc", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the declared trial budget; select and report, never launch test or replicas."""
    args = build_parser().parse_args(argv)
    if args.study == "S2" and args.prefix_base_auprc is None:
        raise ValueError("S2 requires --prefix-base-auprc")
    base = yaml.safe_load(args.base_config.read_text())
    expected_family = "v3_1_topo_prompt" if args.study.startswith("T") else "v3_1_coord_gen"
    if base["model"]["family"] != expected_family:
        raise ValueError(f"{args.study} needs a {expected_family} base config")
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"topology_prompt_v2_{args.study}",
        storage=f"sqlite:///{args.sweep_dir / 'optuna.db'}",
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=len(PRIOR_VALUES[args.study]),
            constraints_func=lambda t: t.user_attrs["constraints"],
        ),
    )
    # A live/unfinished run is not silently marked failed and duplicated on restart.
    for frozen in study.get_trials(states=(TrialState.RUNNING,)):
        result = outcome(
            args.sweep_dir / f"trial_{frozen.number:03d}", args.study, args.prefix_base_auprc
        )
        _tell(study, optuna.Trial(study, frozen._trial_id), result)
    enqueue_priors(
        study,
        [dict(zip(SPACES[args.study], values, strict=True)) for values in PRIOR_VALUES[args.study]],
    )
    while len(study.get_trials(states=(TrialState.COMPLETE,))) < BUDGETS[args.study]:
        trial = study.ask()
        output = args.sweep_dir / f"trial_{trial.number:03d}"
        path = materialize(args.base_config, suggest(args.study, trial), output)
        command = ["bash", "hpc/run.sh", "train", str(path), "--skip-test"]
        if args.study.startswith("T"):
            command += ["--run-kind", "diagnostic"]
        if run_command(command) != 0:
            study.tell(trial, state=TrialState.FAIL)
            raise RuntimeError(f"trial {trial.number} failed; inspect {output}")
        _tell(study, trial, outcome(output, args.study, args.prefix_base_auprc))
    winner = select_winner(study.get_trials())
    report = {
        "selection_rule": SELECTION_RULE,
        "status": "selected" if winner else "no_selectable_trial",
        "trial_number": winner.number if winner else None,
        "params": winner.params if winner else None,
        "result": winner.user_attrs if winner else None,
    }
    (args.sweep_dir / "best_trial.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
