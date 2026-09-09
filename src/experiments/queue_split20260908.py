"""One-shot H20 chain for one split campaign: teacher, banks, five KD students.

Launch from the checkout with its ``.venv`` Python and ``nohup`` after the teacher
pipeline has started (``configs/<campaign>/teacher_pma1.yaml``), passing
``--campaign <name>`` (``split_seed42`` for the random headline split,
``split20260908`` for the retired test-informed one). The chain waits for the
teacher's pipeline lock, requires its diagnostic completion artifacts, dumps the
shared row bank and the ``h2ns3`` context bank from ``best.pt``, then trains,
publishes and tests ``kd_logit``, ``kd_rank``, ``kd_gram``, ``kd_rep`` and
``kd_rank_rep`` in sequence on every visible GPU. Any failed dependency stops it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from src.data.val_region import ValRegionParams

# The split is pinned in code (ValRegionParams.split_seed); a campaign's configs
# only carry paths. Each campaign is therefore bound to the seed its teacher and
# banks were built under, and the chain refuses to run one against another split.
CAMPAIGN_SPLIT_SEED = {"split_seed42": 42, "split20260908": 273}
CAMPAIGN = "split20260908"
TEACHER = Path(f"outputs/{CAMPAIGN}/teacher_pma1")
ROOT = Path(f"outputs/{CAMPAIGN}")
CONFIGS = Path(f"configs/{CAMPAIGN}")
ARMS = ("kd_logit", "kd_rank", "kd_gram", "kd_rep", "kd_rank_rep")
BANKS = Path(f"outputs/distill/{CAMPAIGN}")
POLL_SECONDS = 60.0


def configure(campaign: str) -> None:
    """Point every campaign path at ``configs/<campaign>`` and its output trees.

    Raises:
        ValueError: If the campaign is unknown or was built under a different
            split seed than the checkout's ``ValRegionParams`` default, which
            would silently mix a retired split's teacher with new-split targets.
    """
    global CAMPAIGN, TEACHER, ROOT, CONFIGS, BANKS  # noqa: PLW0603
    if campaign not in CAMPAIGN_SPLIT_SEED:
        raise ValueError(f"unknown campaign {campaign!r}; known: {sorted(CAMPAIGN_SPLIT_SEED)}")
    expected, actual = CAMPAIGN_SPLIT_SEED[campaign], ValRegionParams().split_seed
    if expected != actual:
        raise ValueError(
            f"campaign {campaign!r} was built under split_seed={expected}, but this checkout "
            f"pins split_seed={actual}; run it only from its historical checkout"
        )
    CAMPAIGN = campaign
    ROOT = Path(f"outputs/{campaign}")
    TEACHER = ROOT / "teacher_pma1"
    CONFIGS = Path(f"configs/{campaign}")
    BANKS = Path(f"outputs/distill/{campaign}")


def status(stage: str, **extra: object) -> None:
    """Write the current queue stage atomically."""
    record = {"stage": stage, "pid": os.getpid(), "time": time.time(), **extra}
    temporary = ROOT / "queue_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(ROOT / "queue_status.json")
    logging.info("%s", json.dumps(record))


def is_complete(output: Path, *, diagnostic: bool = False) -> bool:
    """Return whether publication and held-out testing artifacts all exist."""
    prefix = "diagnostic_" if diagnostic else ""
    names = ("complete.json", "test_complete.json", "test_report.json")
    return (
        all((output / (prefix + name)).is_file() for name in names)
        and (output / "best.pt").is_file()
    )


def require_complete(output: Path, *, diagnostic: bool = False) -> None:
    """Require publication and successful held-out testing before advancing.

    Raises:
        RuntimeError: If the run failed or any completion artifact is missing.
    """
    prefix = "diagnostic_" if diagnostic else ""
    if (output / "failure.json").exists():
        raise RuntimeError(f"Failed dependency: {output}/failure.json")
    for name in ("complete.json", "test_complete.json", "test_report.json"):
        if not (output / (prefix + name)).is_file():
            raise RuntimeError(f"Missing completion artifact: {output}/{prefix}{name}")
    if not (output / "best.pt").is_file():
        raise RuntimeError(f"Missing checkpoint: {output}/best.pt")


def wait_teacher() -> None:
    """Block until the teacher pipeline has trained and tested, or failed.

    The pipeline holds ``.pipeline.lock`` for its whole pack/train/publish/test
    run, so a blocking flock returns only once it exits. Acquiring the lock
    before the pipeline does (or between attempts) is harmless: the loop
    releases it and polls again until the completion artifacts exist.

    Raises:
        RuntimeError: If the teacher wrote ``failure.json``.
    """
    status("waiting_for_teacher", teacher=str(TEACHER))
    lock_path = TEACHER / ".pipeline.lock"
    while True:
        if (TEACHER / "failure.json").exists():
            raise RuntimeError(f"Failed dependency: {TEACHER}/failure.json")
        if lock_path.exists():
            with lock_path.open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                if is_complete(TEACHER, diagnostic=True):
                    require_complete(TEACHER, diagnostic=True)
                    return
        time.sleep(POLL_SECONDS)


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
        "bash",
        "hpc/run.sh",
        "kd-targets",
        "--config",
        str(CONFIGS / "kd_logit.yaml"),
        "--checkpoint",
        str(TEACHER / "best.pt"),
        "--output",
        str(output),
    ]
    if contexts:
        command += ["--contexts", "--rw-step", "3", "--hops", "2", "--ns-rate", "3"]
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for index, device in enumerate(devices):
            log = (ROOT / "logs" / f"{name}_{index}.log").open("ab")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command + ["--device", "cuda", "--row-shard", f"{index}/{len(devices)}"],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
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


def main(campaign: str | None = None) -> None:
    """Execute this one-shot dependency chain under an exclusive queue lock."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if campaign is not None:
        configure(campaign)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="16", MKL_NUM_THREADS="16")
    with (ROOT / ".queue.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_teacher()
            devices = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
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
                run(["bash", "hpc/run.sh", "train", str(CONFIGS / f"{arm}.yaml")], f"{arm}.log")
                require_complete(ROOT / arm)
            status("complete", arms=ARMS)
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--campaign", required=True, help="configs/<campaign> directory name")
    main(parser.parse_args().campaign)
