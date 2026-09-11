"""Continue a completed KD lane: test arm winners, Gram grid/test, Logit+Rep HPO.

All winners are selected from published V_val metrics before their held-out test.
Run on the same container as --wait-pid. The existing lane must finish successfully.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.autoresearch.metrics_io import read_run
from src.eval.checkpoint_selection import CheckpointCandidate, select_checkpoint


def select_run(runs: list[Path]) -> Path:
    """Apply production five-metric ranking to completed runs only."""
    runs = sorted(runs)
    candidates = []
    for index, path in enumerate(runs, 1):
        metrics = read_run(path)
        candidates.append(CheckpointCandidate(index, metrics.auprc, metrics.topology))
    winner = select_checkpoint(candidates)
    if winner is None:
        raise ValueError("No completed runs to select")
    return runs[winner.epoch - 1]


def completed_runs(root: Path, pattern: str) -> list[Path]:
    """Find publications, excluding attempt copies and incomplete trials."""
    return [p.parent for p in root.glob(pattern + "/complete.json")]


def run_command(argv: list[str]) -> None:
    """Run one foreground stage, aborting the queue on failure."""
    print("RUN", argv, flush=True)  # noqa: T201 -- operator log
    subprocess.run(
        argv, check=True, env={**os.environ, "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}
    )


def test_winner(run: Path, arm: str) -> None:
    """Test the frozen published checkpoint without retraining."""
    run_command(
        [
            "bash",
            "hpc/run.sh",
            "test",
            "--checkpoint",
            str(run / "best.pt"),
            "--output-dir",
            str(run),
            "--data-root",
            "data",
            "--strategy",
            "breadth_first",
            "--arm",
            arm,
            "--seed",
            "0",
        ]
    )
    if not (run / "test_report.json").exists():
        raise RuntimeError(f"test exited without report: {run}")


def main() -> None:
    """Own the serialized follow-up and persist progress for operators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    status_path = root / "followup_queue.json"
    lock_path = root / ".followup_queue.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        state.update(pid=os.getpid())
        state.setdefault("completed_tests", {})

        def status(stage: str, **extra: object) -> None:
            state.update(stage=stage, **extra)
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(status_path)

        def evaluate(arm: str, runs: list[Path]) -> None:
            winner = select_run(runs)
            done = state["completed_tests"]
            metadata = json.loads((winner / "run_metadata.json").read_text())
            identity = {"run_dir": str(winner), "checkpoint_id": metadata["checkpoint_id"]}
            if done.get(arm) == identity and (winner / "test_report.json").exists():
                return
            status("testing", arm=arm, run_dir=str(winner))
            test_winner(winner, arm)
            done[arm] = identity
            status("tested")

        try:
            status("waiting", wait_pid=args.wait_pid)
            proc = Path(f"/proc/{args.wait_pid}/cmdline")
            while proc.exists():
                try:
                    command = proc.read_bytes()
                except FileNotFoundError:
                    break
                if b"src.experiments.reselect_campaign" not in command:
                    break
                time.sleep(30)
            lane = json.loads((root / "lane_kd.json").read_text())
            if lane["status"] != "complete":
                raise RuntimeError(f"predecessor did not complete: {lane}")
            for arm, pattern in [
                ("kd_rank", "kd_hpo/rank/trial_*"),
                ("kd_rank_rep", "kd_hpo/rank_rep/trial_*"),
                ("kd_rep", "kd_hpo/grid/kd_rep_*"),
                ("kd_logit", "kd_hpo/grid/kd_logit_*"),
            ]:
                evaluate(arm, completed_runs(root, pattern))
            status("gram_grid")
            configs = sorted(Path("configs/split_seed42/sweep").glob("kd_gram_*.yaml"))
            if len(configs) != 5:
                raise ValueError(f"expected five Gram grid configs, got {len(configs)}")
            for source in configs:
                target = root / "kd_hpo/grid" / source.stem
                if (target / "complete.json").exists():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                config = yaml.safe_load(source.read_text())
                config["output_dir"] = str(target)
                path = target / "config.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                status("gram_grid", run_dir=str(target))
                run_command(["bash", "hpc/run.sh", "train", str(path), "--skip-test"])
                read_run(target)
            evaluate("kd_gram", completed_runs(root, "kd_hpo/grid/kd_gram_*"))
            status("logit_rep_hpo")
            run_command(
                [
                    sys.executable,
                    "-m",
                    "src.experiments.kd_logit_rep_hpo",
                    "--sweep-dir",
                    str(root / "kd_hpo/logit_rep"),
                    "--n-trials",
                    "12",
                ]
            )
            status("complete")
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
