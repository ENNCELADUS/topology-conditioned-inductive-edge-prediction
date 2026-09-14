"""Six-trial L3-PPI reproduction; select on V_val before testing one winner."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.eval.checkpoint_selection import (
    SELECTION_RULE,
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)

TRIALS = tuple((k, lr) for k in (4, 16, 64) for lr in (1e-4, 1e-3))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _run(command: list[str], log_path: Path) -> None:
    """Keep worker output streaming to disk without buffering it in memory."""
    with log_path.open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def _published(directory: Path) -> bool:
    complete = (directory / "complete.json").exists()
    failure = (directory / "failure.json").exists()
    if complete and failure:
        raise RuntimeError(f"Conflicting completion and failure artifacts: {directory}")
    return complete and (directory / "best.pt").exists() and not failure


def select_trial(directories: list[Path]) -> dict[str, Any]:
    """Rank selected trial checkpoints by the shared five-metric selector."""
    selections = [
        json.loads((directory / "selection.json").read_text()) for directory in directories
    ]
    candidates = [
        CheckpointCandidate(
            epoch=index,
            auprc=float(selection["val_auprc"]),
            topology=TopologyValidationMetrics(**selection["topology"]["metrics"]),
        )
        for index, selection in enumerate(selections)
    ]
    selected = select_checkpoint(candidates)
    if selected is None:
        raise ValueError("No L3-PPI trials to select")
    index = selected.epoch
    return {
        "trial_index": index,
        "trial_dir": str(directories[index]),
        "checkpoint": str(directories[index] / "best.pt"),
        "model_epoch": selections[index]["epoch"],
        "selection_rule": SELECTION_RULE,
        "val_auprc": selections[index]["val_auprc"],
        "topology": selections[index]["topology"],
    }


def run_study(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    skip_test: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Run on the H20 checkout; all GPU work goes through hpc/run.sh."""
    base = yaml.safe_load(config_path.read_text())
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Study directory is nonempty; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    shared = copy.deepcopy(base)
    shared["surrogate_checkpoint"] = str(output_dir / "surrogate" / "best.pt")
    shared["feature_cache"] = str(output_dir / "endpoint_features.pt")
    stages: list[tuple[str, Path, Path]] = []
    for index in range(-1, len(TRIALS)):
        name = "surrogate" if index == -1 else f"trial_{index:03d}"
        stage = "surrogate" if index == -1 else "trial"
        config = copy.deepcopy(shared)
        config["output_dir"] = str(output_dir / name)
        if index >= 0:
            config["model"]["config"]["k"], config["optim"]["lr"] = TRIALS[index]
        generated = config_dir / f"{name}.yaml"
        if generated.exists() and yaml.safe_load(generated.read_text()) != config:
            raise ValueError(f"Resume configuration differs from existing study: {generated}")
        generated.write_text(yaml.safe_dump(config, sort_keys=False))
        stages.append((stage, generated, output_dir / name))
    if prepare_only:
        return {"status": "prepared", "trial_count": len(TRIALS)}
    status_path = output_dir / "status.json"
    _write_json(status_path, {"status": "running"})
    try:
        for stage, generated, directory in stages:
            if _published(directory):
                continue
            command = [
                "hpc/run.sh",
                "train",
                str(generated),
                "--worker-module",
                "src.train_l3ppi",
                "--stage",
                stage,
                "--skip-test",
            ]
            if resume and directory.exists():
                command.append("--resume")
            _write_json(status_path, {"status": "running", "stage": directory.name})
            _run(command, output_dir / f"{directory.name}.log")
            if not _published(directory):
                raise RuntimeError(f"Worker exited without successful publication: {directory}")
        winner = select_trial([directory for _, _, directory in stages[1:]])
        winner_path = output_dir / "winner.json"
        if winner_path.exists() and json.loads(winner_path.read_text()) != winner:
            raise ValueError("Published trial selection changed after winner was locked")
        _write_json(winner_path, winner)
        report = Path(winner["trial_dir"]) / "test_report.json"
        if not skip_test and not report.exists():
            _write_json(status_path, {"status": "running", "stage": "winner_test"})
            _run(
                [
                    "hpc/run.sh",
                    "test",
                    "--checkpoint",
                    winner["checkpoint"],
                    "--output-dir",
                    winner["trial_dir"],
                    "--pack-dir",
                    str(base["pack_dir"]),
                    "--data-root",
                    str(base["data_root"]),
                    "--strategy",
                    "breadth_first",
                    "--arm",
                    "l3ppi",
                    "--seed",
                    str(base["seed"]),
                ],
                output_dir / "winner_test.log",
            )
            if not report.exists():
                raise RuntimeError("Winner evaluation exited without test_report.json")
        result = {"status": "tested" if report.exists() else "published", "winner": winner}
        _write_json(status_path, result)
        _write_json(output_dir / "complete.json", result)
        (output_dir / "failure.json").unlink(missing_ok=True)
        return result
    except Exception as error:
        failure = {"status": "failed", "error": str(error), "type": type(error).__name__}
        _write_json(status_path, failure)
        _write_json(output_dir / "failure.json", failure)
        (output_dir / "complete.json").unlink(missing_ok=True)
        raise


def main() -> None:
    """Parse the study command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/split_seed42/l3ppi.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/l3ppi"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(
            run_study(
                args.config,
                args.output_dir,
                resume=args.resume,
                skip_test=args.skip_test,
                prepare_only=args.prepare_only,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
