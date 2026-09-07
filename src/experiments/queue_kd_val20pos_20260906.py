"""One-shot H20 chain: finish the new teacher, dump banks, train/test five KDs.

Launch from the checkout with its .venv Python and nohup. The teacher's pipeline
lock covers its test stage too. Any failed dependency stops this chain.
"""

import fcntl
import json
import logging
import os
import subprocess
import time
from pathlib import Path

TEACHER = Path("outputs/egostitch_e2e_stage1_v3/full_ego_teacher_pma1_val20pos_20260906")
TEACHER_ATTEMPT = "181013a087964e9e93081d24a7fc6198"
ROOT = Path("outputs/kd_val20pos_20260906")
CONFIGS = Path("configs/kd_val20pos_20260906")
ARMS = ("kd_logit", "kd_rank", "kd_gram", "kd_rep", "kd_rank_rep")
BANKS = Path("outputs/distill/val20pos_20260906")
F0 = "outputs/feature_packs/egostitch_e2e_val20pos_20260906_ng50/f0_matrix.pt"


def status(stage: str, **extra: object) -> None:
    """Write the current queue stage atomically."""
    record = {"stage": stage, "pid": os.getpid(), "time": time.time(), **extra}
    temporary = ROOT / "queue_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(ROOT / "queue_status.json")
    logging.info("%s", json.dumps(record))


def require_complete(output: Path, *, diagnostic: bool = False) -> None:
    """Require publication and successful held-out testing before advancing."""
    prefix = "diagnostic_" if diagnostic else ""
    if (output / "failure.json").exists():
        raise RuntimeError(f"Failed dependency: {output}/failure.json")
    for name in ("complete.json", "test_complete.json", "test_report.json"):
        if not (output / (prefix + name)).is_file():
            raise RuntimeError(f"Missing completion artifact: {output}/{prefix}{name}")
    if not (output / "best.pt").is_file():
        raise RuntimeError(f"Missing checkpoint: {output}/best.pt")


def wait_teacher() -> None:
    """Wait for the exact teacher pipeline to finish training and testing."""
    status("waiting_for_teacher", teacher_attempt=TEACHER_ATTEMPT)
    # Blocking flock is released only when the entire teacher pipeline exits.
    with (TEACHER / ".pipeline.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        attempt = json.loads((TEACHER / "current_attempt.json").read_text())
        if attempt["attempt_id"] != TEACHER_ATTEMPT:
            raise RuntimeError("Teacher attempt changed while queued")
        require_complete(TEACHER, diagnostic=True)


def run(command: list[str], log_name: str) -> None:
    """Run one pipeline stage to completion with a dedicated log."""
    logging.info("RUN %s", command)
    with (ROOT / "logs" / log_name).open("ab") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def dump_bank(*, contexts: bool, devices: list[str]) -> None:
    """Score a fresh bank across available GPUs and merge after shard exits."""
    name = "contexts_h2ns3" if contexts else "rows"
    output = BANKS / name
    if output.exists():
        raise RuntimeError(f"Fresh queue refuses an existing bank: {output}")
    status("dumping_bank", bank=name)
    command = [
        "bash", "hpc/run.sh", "kd-targets",
        "--config", str(CONFIGS / "kd_logit.yaml"),
        "--checkpoint", str(TEACHER / "best.pt"),
        "--output", str(output), "--f0-cache", F0,
    ]
    if contexts:
        command += ["--contexts", "--rw-step", "3", "--hops", "2", "--ns-rate", "3"]
    processes = []
    logs = []
    try:
        for index, device in enumerate(devices):
            log = (ROOT / "logs" / f"{name}_{index}.log").open("ab")
            logs.append(log)
            processes.append(subprocess.Popen(
                command + ["--device", "cuda", "--row-shard", f"{index}/{len(devices)}"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                stdout=log, stderr=subprocess.STDOUT,
            ))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"{name} shard exits: {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    run(command + ["--merge", "--row-shard", f"0/{len(devices)}"], f"{name}_merge.log")
    if not (output / "manifest.json").is_file():
        raise RuntimeError(f"Missing bank manifest: {output}")


def main() -> None:
    """Execute this one-shot dependency chain under an exclusive queue lock."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="16", MKL_NUM_THREADS="16")
    with (ROOT / ".queue.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_teacher()
            devices = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
            ).split()
            if not devices:
                raise RuntimeError("No GPUs available")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            dump_bank(contexts=False, devices=devices)
            dump_bank(contexts=True, devices=devices)
            for arm in ARMS:
                if (ROOT / arm).exists():
                    raise RuntimeError(f"Fresh queue refuses an existing run: {arm}")
                status("training_and_testing", arm=arm)
                run([
                    "bash", "hpc/run.sh", "train", str(CONFIGS / f"{arm}.yaml"),
                ], f"{arm}.log")
                require_complete(ROOT / arm)
            status("complete", arms=ARMS)
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
