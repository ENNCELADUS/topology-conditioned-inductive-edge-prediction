"""One-shot TUnA then PPITrans pipeline after the current container job exits."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml


def process_start(pid: int) -> str | None:
    """Identify a live Linux process without confusing a reused PID or zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def main() -> None:
    """Wait for the specified running job, then run both existing HPC commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"
    signature = process_start(args.wait_pid)

    def status(value: dict[str, object]) -> None:
        status_path.write_text(json.dumps(value, indent=2) + "\n")

    status({"status": "waiting", "wait_pid": args.wait_pid, "order": ["tuna", "ppitrans"]})
    while True:
        predecessor = signature is not None and process_start(args.wait_pid) == signature
        gpu_pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if not predecessor and not gpu_pids:
            break
        time.sleep(30)
    results: dict[str, int] = {}
    for method in ("tuna", "ppitrans"):
        config = Path(f"configs/split_seed42/{method}_official.yaml")
        run_output = Path(yaml.safe_load(config.read_text())["output_dir"])
        if run_output.exists():
            raise FileExistsError(f"Refusing to overwrite {run_output}")
        status({"status": "running", "method": method, "results": results})
        with (args.output / f"{method}.log").open("x") as log:
            result = subprocess.run(
                ["hpc/run.sh", "train", str(config), "--worker-module", "src.train_official_ppi"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results[method] = result.returncode
        if result.returncode:
            run_output.mkdir(parents=True, exist_ok=True)
            (run_output / "failure.json").write_text(
                json.dumps({"exit_code": result.returncode}) + "\n"
            )
    status(
        {
            "status": "complete" if all(code == 0 for code in results.values()) else "failed",
            "results": results,
        }
    )


if __name__ == "__main__":
    main()
