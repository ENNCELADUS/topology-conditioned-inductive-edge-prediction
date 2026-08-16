"""E2 pipeline worker payloads, failure artifacts, and orchestration.

Provides pure dataclasses and functions for:
- ProbeResult: one worker-reported throughput/memory measurement
- write_failure(): atomic failure JSON artifact writer

And the production orchestrator (``python -m src.e2_pipeline``) that drives the
cold multi-H20 run end to end in four sub-stages — ``pack -> train -> publish
-> test``: pack-or-validate the BF16 feature cache, launch one clean
``accelerate launch`` ``train`` process group at the configured
``runtime.token_budget``, merge the worker's runtime profile with pipeline-level
fields, publish the validated attempt artifact tree atomically, then run the published
checkpoint through the held-out test protocol. See :func:`run_pipeline` for
the full contract.

EgoStitch E2E uses this same orchestrator, selected via ``--worker-module
src.train_egostitch``.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from io import BufferedReader
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single worker throughput/memory measurement.

    The orchestrator no longer sweeps token budgets, but the workers still emit
    this payload from their own ``--ddp-mode probe`` measurement path.

    Attributes:
        token_budget: Number of tokens per global batch (pairs).
        valid: True if the measurement completed without OOM or other hard failure.
        global_pairs_per_second: Throughput achieved in this measurement.
        peak_memory_gib: Peak GPU memory used (GiB).
        failure: None if valid=True; error message if valid=False.
    """

    token_budget: int
    valid: bool
    global_pairs_per_second: float
    peak_memory_gib: float
    failure: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-serializable dict without external dependencies."""
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ProbeResult":
        """Reconstruct and strictly validate an untrusted worker payload."""
        token_budget = data["token_budget"]
        valid = data["valid"]
        throughput = data["global_pairs_per_second"]
        peak_memory = data["peak_memory_gib"]
        failure = data["failure"]
        if isinstance(token_budget, bool) or not isinstance(token_budget, int):
            raise TypeError("token_budget must be an integer")
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not isinstance(valid, bool):
            raise TypeError("valid must be a bool")
        if isinstance(throughput, bool) or not isinstance(throughput, int | float):
            raise TypeError("global_pairs_per_second must be numeric")
        if isinstance(peak_memory, bool) or not isinstance(peak_memory, int | float):
            raise TypeError("peak_memory_gib must be numeric")
        throughput_float = float(throughput)
        peak_memory_float = float(peak_memory)
        if not math.isfinite(throughput_float) or throughput_float < 0:
            raise ValueError("global_pairs_per_second must be finite and nonnegative")
        if not math.isfinite(peak_memory_float) or peak_memory_float < 0:
            raise ValueError("peak_memory_gib must be finite and nonnegative")
        if valid and throughput_float <= 0:
            raise ValueError("a valid probe must report positive throughput")
        if valid and failure is not None:
            raise ValueError("a valid probe cannot report a failure")
        if not valid and (not isinstance(failure, str) or not failure.strip()):
            raise ValueError("an invalid probe must report a non-empty failure")
        return cls(
            token_budget=token_budget,
            valid=valid,
            global_pairs_per_second=throughput_float,
            peak_memory_gib=peak_memory_float,
            failure=cast(str | None, failure),
        )


def write_failure(
    output_dir: Path,
    *,
    stage: str,
    message: str,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write atomic failure artifact to output directory.

    Writes to a temporary file first, then atomically renames to final path to prevent
    partial/corrupted reads.

    Args:
        output_dir: Directory to write failure.json to.
        stage: Pipeline stage at which failure occurred.
        message: Human-readable failure message.
        extra: Optional dict of additional fields to include in payload.

    Returns:
        Path to final failure.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": stage, "message": message}
    if extra is not None:
        payload.update(extra)
    temp_path = output_dir / "failure.json.tmp"
    final_path = output_dir / "failure.json"
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, final_path)
    return final_path


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write ``payload`` as pretty JSON via a temp-file-then-rename (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


# --------------------------------------------------------------------------- pipeline orchestration


@dataclass(frozen=True)
class PipelineArgs:
    """Parsed ``python -m src.e2_pipeline`` command-line arguments.

    Attributes:
        config: Path to the worker training YAML config; must include a
            ``runtime:`` section (see ``src.train_b0.RuntimeConfig``).
        pack_dir: Optional override for the packed feature directory;
            defaults to ``cfg.runtime.pack_dir`` when omitted.
        output_dir: Optional override for the run output directory; defaults to
            ``cfg.output_dir`` when omitted.
        seed: Optional worker seed override for pre-registered multi-seed runs.
        max_steps: Optional DEBUG-ONLY bounded worker-step limit.
        run_kind: Optional EgoStitch-E2E execution context forwarded to the
            worker; the only public value is ``formal``. ``debug`` is not
            selectable here — it is derived by the worker from ``--max-steps``.
        worker_module: Dotted module implementing the worker contract
            (``load_config``, ``prepare_pack``, and the ``--ddp-mode train``
            CLI). Defaults to the formal E2 B0 worker; ``src.train_egostitch``
            selects the EgoStitch worker.
        rescore_reason: Forwarded unchanged to the test stage's
            ``run_test_protocol`` call. Required only when this ``(arm,
            seed)`` has already opened a held-out scoring epoch; absent here
            means absent there — this pipeline never synthesizes one.
    """

    config: Path
    pack_dir: Path | None
    output_dir: Path | None
    worker_module: str = "src.train_b0"
    seed: int | None = None
    max_steps: int | None = None
    run_kind: str | None = None
    rescore_reason: str | None = None
    resume_attempt: Path | None = None


def build_accelerate_command(
    *,
    accelerate_bin: Path,
    config_path: Path,
    mode: str,
    pack_dir: Path,
    output_dir: Path,
    token_budget: int,
    profile_output: Path,
    world_size: int,
    worker_module: str = "src.train_b0",
    seed: int | None = None,
    max_steps: int | None = None,
    run_kind: str | None = None,
    resume_attempt: Path | None = None,
) -> list[str]:
    """Build the pinned ``accelerate launch -m <worker>`` worker command.

    Args:
        accelerate_bin: Path to the ``accelerate`` executable (same venv as the
            orchestrator's own interpreter).
        config_path: Path to the worker training YAML config.
        mode: The worker's DDP mode; the orchestrator only launches ``train``.
        pack_dir: Packed feature directory the worker loads onto its device.
        output_dir: Run output directory (checkpoints/metrics for ``train``;
            passed through unconditionally so every mode sees the same override).
        token_budget: Per-rank token budget for this worker invocation
            (``runtime.token_budget``; the EgoStitch worker reinterprets it as
            the node-batch ``B_n``).
        profile_output: Path the rank-zero worker writes its JSON profile to.
        world_size: Number of visible H20 ranks to launch.
        worker_module: Dotted worker module (default: the formal E2 B0 worker).
        seed: Optional worker seed override.
        max_steps: Optional DEBUG-ONLY bounded worker-step limit.
        run_kind: Optional EgoStitch-E2E execution context forwarded unchanged
            to the worker.
        resume_attempt: Optional prior attempt directory whose completed
            epoch-boundary state the worker resumes from.

    Returns:
        The exact ``accelerate launch --num_processes <world_size> --mixed_precision bf16
        -m <worker_module> ...`` argv list.
    """
    command = [
        str(accelerate_bin),
        "launch",
        "--num_processes",
        str(world_size),
        "--mixed_precision",
        "bf16",
        "-m",
        worker_module,
        "--config",
        str(config_path),
        "--ddp-mode",
        mode,
        "--pack-dir",
        str(pack_dir),
        "--output-dir",
        str(output_dir),
        "--token-budget-per-rank",
        str(token_budget),
        "--profile-output",
        str(profile_output),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if max_steps is not None:
        command.extend(("--max-steps", str(max_steps)))
    if run_kind is not None:
        command.extend(("--run-kind", run_kind))
    if resume_attempt is not None:
        command.extend(("--resume-attempt", str(resume_attempt)))
    return command


def detect_visible_gpu_count() -> int:
    """Return the number of GPUs visible to the current process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if devices and devices != ["-1"]:
            return len(devices)
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    count = len([line for line in completed.stdout.splitlines() if line.strip()])
    if count < 1:
        raise RuntimeError("no visible GPUs found for distributed training")
    return count


def parse_pipeline_args(argv: Sequence[str] | None = None) -> PipelineArgs:
    """Parse the public ``python -m src.e2_pipeline`` CLI.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The parsed ``PipelineArgs``.
    """
    parser = argparse.ArgumentParser(prog="python -m src.e2_pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-attempt", type=Path, default=None)
    parser.add_argument(
        "--run-kind",
        choices=("formal", "diagnostic"),
        default=None,
        help=(
            "EgoStitch E2E execution context; forwarded unchanged to the worker. "
            "'diagnostic' never publishes formal artifacts; 'debug' is not "
            "selectable: the worker derives it from --max-steps"
        ),
    )
    parser.add_argument(
        "--worker-module",
        default="src.train_b0",
        help=(
            "worker module implementing load_config/prepare_pack and the "
            "--ddp-mode CLI (default: src.train_b0; src.train_egostitch for "
            "the EgoStitch worker)"
        ),
    )
    parser.add_argument(
        "--rescore-reason",
        default=None,
        help=(
            "required to re-open held-out scoring for an (arm, seed) that "
            "already has a scoring epoch; forwarded unchanged to the test "
            "stage. Never synthesized: omit it and the test stage fails "
            "loudly instead of silently reusing a prior epoch"
        ),
    )
    namespace = parser.parse_args(argv)
    return PipelineArgs(
        config=namespace.config,
        pack_dir=namespace.pack_dir,
        output_dir=namespace.output_dir,
        worker_module=namespace.worker_module,
        seed=namespace.seed,
        max_steps=namespace.max_steps,
        run_kind=namespace.run_kind,
        rescore_reason=namespace.rescore_reason,
        resume_attempt=namespace.resume_attempt,
    )


def _pipeline_rerun_command(args: PipelineArgs) -> str:
    """Render the exact CLI invocation to re-open held-out scoring for this run.

    Used only to make the test-access ledger's repeat-scoring refusal
    actionable: the operator gets a copy-pasteable command instead of having
    to reconstruct the original invocation by hand. The reason itself is
    always a placeholder — a synthetic reason would hollow out the ledger.
    """
    parts = ["python", "-m", "src.e2_pipeline", "--config", str(args.config)]
    if args.pack_dir is not None:
        parts += ["--pack-dir", str(args.pack_dir)]
    if args.output_dir is not None:
        parts += ["--output-dir", str(args.output_dir)]
    if args.seed is not None:
        parts += ["--seed", str(args.seed)]
    if args.max_steps is not None:
        parts += ["--max-steps", str(args.max_steps)]
    if args.run_kind is not None:
        parts += ["--run-kind", args.run_kind]
    if args.resume_attempt is not None:
        parts += ["--resume-attempt", str(args.resume_attempt)]
    if args.worker_module != "src.train_b0":
        parts += ["--worker-module", args.worker_module]
    parts += ["--rescore-reason", "<reason for the repeat held-out scoring epoch>"]
    return " ".join(parts)


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
TrainingCommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
StageRunner = Callable[[Callable[[], None]], None]


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``command`` to completion on the fixed POSIX/Linux target.

    Args:
        command: The argv list to execute.

    Returns:
        The ``CompletedProcess`` (``check=False``: callers inspect ``.returncode``).
    """
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def run_logged_command(command: Sequence[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a process group while the parent durably streams merged output to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.Popen(
            list(command),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
    except BaseException:
        log_handle.close()
        raise
    output_pipe = cast(BufferedReader, process.stdout)
    reader_error: list[BaseException] = []

    def read_output() -> None:
        try:
            while chunk := output_pipe.read1(65536):
                log_handle.write(chunk)
        except BaseException as error:
            reader_error.append(error)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while process.poll() is None:
            if reader_error:
                break
            time.sleep(0.05)
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    except BaseException:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        reader.join(timeout=0.2)
        if reader.is_alive():
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            reader.join(timeout=2.0)
        if reader.is_alive():
            with suppress(OSError):
                os.close(output_pipe.fileno())
            reader.join(timeout=2.0)
        if reader.is_alive():
            reader_error.append(TimeoutError("training output pipe did not close"))
        log_handle.close()
    if reader_error:
        raise OSError(f"failed to persist training output: {reader_error[0]}")
    return subprocess.CompletedProcess(list(command), process.returncode, "", "")


def _read_log_tail(path: Path, *, max_bytes: int = 16384, max_lines: int = 50) -> str:
    """Read a bounded UTF-8 tail without loading an arbitrarily large log."""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read(max_bytes)
    text = data.decode("utf-8", errors="replace")
    if start > 0 and "\n" in text:
        text = text.split("\n", 1)[1]
    return "\n".join(text.splitlines()[-max_lines:])


def _run_stage_child(operation: Callable[[], None], sender: Connection) -> None:
    """Execute a Python stage and report its exception."""
    try:
        operation()
    except BaseException as error:
        sender.send((False, f"{type(error).__name__}: {error}"))
    else:
        sender.send((True, ""))
    finally:
        sender.close()


def run_stage(operation: Callable[[], None]) -> None:
    """Run a Python stage in a child process and propagate its result."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_stage_child, args=(operation, sender))
    process.start()
    sender.close()
    result: tuple[bool, str] | None = None
    try:
        with suppress(EOFError):
            result = cast(tuple[bool, str], receiver.recv())
    finally:
        receiver.close()
        process.join()
    if result is None:
        raise RuntimeError(f"supervised stage exited with code {process.exitcode} without a result")
    succeeded, message = result
    if not succeeded:
        raise RuntimeError(message)


def _resolve_accelerate_bin() -> Path:
    """Return the ``accelerate`` executable next to the running interpreter."""
    return Path(sys.executable).with_name("accelerate")


def _finite_number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _validate_worker_profile(
    data: object,
    *,
    epochs: int,
    world_size: int,
    memory_limit_gib: float,
    allow_partial: bool = False,
    enforce_engineering_limits: bool = True,
) -> dict[str, object]:
    """Validate worker completeness, integrity, and optional engineering limits."""
    if not isinstance(data, dict):
        raise ValueError("worker profile must be a JSON object")
    required = {
        "epochs_completed",
        "validations_completed",
        "peak_memory_gib_per_rank",
        "steady_state_data_wait_fraction",
        "training_coverage_exact",
        "validation_coverage_exact",
        "feature_cache_hit_rate",
        "per_epoch",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"worker profile missing keys: {', '.join(missing)}")
    for key in ("epochs_completed", "validations_completed"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        if value != epochs and not (allow_partial and 1 <= value <= epochs):
            raise ValueError(f"{key} must equal configured epochs ({epochs})")
    for key in ("training_coverage_exact", "validation_coverage_exact"):
        if data[key] is not True:
            raise ValueError(f"{key} must be exactly true")
    hit_rate = _finite_number(data["feature_cache_hit_rate"], field="feature_cache_hit_rate")
    if enforce_engineering_limits and hit_rate != 1.0:
        raise ValueError("feature_cache_hit_rate must be exactly 1.0")
    memories = data["peak_memory_gib_per_rank"]
    if not isinstance(memories, list) or len(memories) != world_size:
        raise ValueError(f"peak_memory_gib_per_rank must contain {world_size} values")
    for rank, value in enumerate(memories):
        memory = _finite_number(value, field=f"peak_memory_gib_per_rank[{rank}]")
        if enforce_engineering_limits and memory > memory_limit_gib:
            raise ValueError(f"rank {rank} peak memory exceeds {memory_limit_gib} GiB")
    wait_fraction = _finite_number(
        data["steady_state_data_wait_fraction"], field="steady_state_data_wait_fraction"
    )
    if enforce_engineering_limits and wait_fraction > 0.05:
        raise ValueError(f"steady_state_data_wait_fraction {wait_fraction:.6f} exceeds 0.05")
    per_epoch = data["per_epoch"]
    expected_epochs = cast(int, data["epochs_completed"])
    if not isinstance(per_epoch, list) or len(per_epoch) != expected_epochs:
        raise ValueError(f"per_epoch must contain exactly {expected_epochs} entries")
    duration_fields = (
        "wall_seconds",
        "data_wait_seconds",
        "compute_seconds",
        "validation_seconds",
    )
    count_fields = ("steps", "global_pairs", "local_pairs", "local_tokens")
    for expected_epoch, entry in enumerate(per_epoch, start=1):
        if not isinstance(entry, dict):
            raise TypeError("per_epoch entries must be objects")
        reported_epoch = entry.get("epoch")
        if (
            isinstance(reported_epoch, bool)
            or not isinstance(reported_epoch, int)
            or reported_epoch != expected_epoch
        ):
            raise ValueError("per_epoch epoch sequence is invalid")
        for field in count_fields:
            _positive_int(entry.get(field), field=f"per_epoch.{field}")
        for field in duration_fields:
            _finite_number(entry.get(field), field=f"per_epoch.{field}")
    counterfactual = data.get("counterfactual_stop_epoch")
    if counterfactual is not None:
        stop_epoch = _positive_int(counterfactual, field="counterfactual_stop_epoch")
        if stop_epoch > expected_epochs:
            raise ValueError("counterfactual_stop_epoch exceeds configured epochs")
    per_rank = data.get("per_rank")
    if per_rank is not None:
        if not isinstance(per_rank, list) or len(per_rank) != world_size:
            raise ValueError(f"per_rank must contain {world_size} entries")
        for expected_rank, entry in enumerate(per_rank):
            if not isinstance(entry, dict) or entry.get("rank") != expected_rank:
                raise ValueError("per_rank rank sequence is invalid")
            for field in ("pairs", "batches", "steps", "tokens"):
                _positive_int(entry.get(field), field=f"per_rank.{field}")
            for field in (
                "train_wall_seconds",
                "data_wait_seconds",
                "pairs_per_second",
                "tokens_per_second",
            ):
                _finite_number(entry.get(field), field=f"per_rank.{field}")
    return cast(dict[str, object], data)


_CHECKPOINT_KEYS = {
    "model_state",
    "model_family",
    "model_config",
    "epoch",
    "val_metrics",
    "seed",
    "config",
}
_PUBLISHED_FILENAMES = (
    "best.pt",
    "last.pt",
    "metrics.jsonl",
    "run_metadata.json",
    "profile.json",
    "artifact_manifest.json",
)
VAL_REGION_VALIDATION_EVENTS_FILENAME = "val_region_validation_events.jsonl"
_OPTIONAL_PUBLISHED_FILENAMES = (VAL_REGION_VALIDATION_EVENTS_FILENAME,)
# One terminal sentinel per run kind, so a diagnostic run can never leave a
# `complete.json` a reader would take for a formal result (`debug_complete.json`
# is written by the debug branch, which never publishes through staging).
_COMPLETION_FILENAMES = ("complete.json", "diagnostic_complete.json")
# Mirrors `_COMPLETION_FILENAMES` one stage later: the test stage's own
# terminal sentinel, split the same way so a diagnostic test run can never
# leave a `test_complete.json` a reader would take for a formal result.
_TEST_COMPLETION_FILENAMES = ("test_complete.json", "diagnostic_test_complete.json")


def _validate_staged_artifacts(
    staging_dir: Path,
    *,
    epochs: int,
    model_family: str,
    allow_partial: bool = False,
    require_val_region_validation_events: bool = False,
    expected_run_kind: str | None = None,
) -> None:
    """Load and validate every formal worker artifact before hashing it."""
    import torch

    for filename in ("best.pt", "last.pt", "metrics.jsonl", "run_metadata.json"):
        path = staging_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{filename} is missing or empty")
    for filename, exact_epoch in (("best.pt", None), ("last.pt", epochs)):
        payload = torch.load(staging_dir / filename, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
            raise ValueError(f"{filename} has an invalid checkpoint payload")
        if payload["model_family"] != model_family:
            raise ValueError(f"{filename} model_family does not match config")
        epoch = payload["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= epochs:
            raise ValueError(f"{filename} epoch is invalid")
        if exact_epoch is not None and epoch != exact_epoch:
            raise ValueError(f"{filename} epoch must equal {exact_epoch}")
    metric_rows: list[object] = []
    with (staging_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("metrics.jsonl contains a blank row")
            metric_rows.append(json.loads(line))
    if len(metric_rows) != epochs:
        raise ValueError(f"metrics.jsonl must contain {epochs} evaluations")
    for expected_epoch, row in enumerate(metric_rows, start=1):
        if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
            raise ValueError("metrics.jsonl epoch sequence is invalid")
    metadata = json.loads((staging_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata.json must contain an object")
    if expected_run_kind is not None:
        expected_role = {
            "formal": "formal_plan_selected",
            "diagnostic": "diagnostic_only",
            "debug": "debug_only",
        }[expected_run_kind]
        if metadata.get("run_kind") != expected_run_kind:
            raise ValueError("run_metadata.json run_kind does not match pipeline execution")
        if metadata.get("checkpoint_role") != expected_role:
            raise ValueError("run_metadata.json checkpoint_role does not match run_kind")
        if metadata.get("formal_artifacts_published") is not (expected_run_kind == "formal"):
            raise ValueError("run_metadata.json formal_artifacts_published does not match run_kind")
    if require_val_region_validation_events:
        evidence = metadata.get("val_region_validation_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("run_metadata.json is missing V_val validation evidence")
        if evidence.get("schema") != "egostitch_e2e_val_region_validation_events_v1":
            raise ValueError("V_val validation-event evidence has an invalid schema")
        if evidence.get("path") != VAL_REGION_VALIDATION_EVENTS_FILENAME:
            raise ValueError("V_val validation-event evidence has an invalid path")
        count = evidence.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("V_val validation-event evidence has an invalid count")
        ledger_path = staging_dir / VAL_REGION_VALIDATION_EVENTS_FILENAME
        if not ledger_path.is_file() or ledger_path.stat().st_size <= 0:
            raise ValueError(f"{VAL_REGION_VALIDATION_EVENTS_FILENAME} is missing or empty")
        ledger_rows: list[object] = []
        with ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("V_val validation-event ledger contains a blank row")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("V_val validation-event ledger row must be an object")
                ledger_rows.append(row)
        if len(ledger_rows) != count:
            raise ValueError("V_val validation-event ledger count does not match metadata")
        digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        if evidence.get("sha256") != digest:
            raise ValueError("V_val validation-event ledger digest does not match metadata")


def _publish_staged(
    staging_dir: Path,
    output_dir: Path,
    *,
    optional_filenames: Sequence[str] = _OPTIONAL_PUBLISHED_FILENAMES,
) -> tuple[Path, list[str]]:
    """Publish validated files with rollback-capable per-file atomic replaces."""
    optional_names = tuple(optional_filenames)
    backup_dir = Path(tempfile.mkdtemp(prefix=".e2-backup-", dir=output_dir))
    published: list[str] = []
    completion_backed_up = False
    staged_optional = tuple(
        filename for filename in optional_names if (staging_dir / filename).is_file()
    )
    try:
        for completion_name in _COMPLETION_FILENAMES:
            completion = output_dir / completion_name
            if completion.exists():
                os.replace(completion, backup_dir / completion.name)
                completion_backed_up = True
        # Every optional name is backed up whether or not this run staged one, so
        # a prior run's verdict is removed rather than left to masquerade as this
        # run's; rollback restores it from the backup.
        for filename in _PUBLISHED_FILENAMES + optional_names:
            canonical = output_dir / filename
            if canonical.exists():
                os.replace(canonical, backup_dir / filename)
        for filename in _PUBLISHED_FILENAMES + staged_optional:
            os.replace(staging_dir / filename, output_dir / filename)
            published.append(filename)
    except BaseException:
        _rollback_publication(
            output_dir,
            backup_dir,
            published,
            remove_completion=completion_backed_up,
            restore_dir=staging_dir,
        )
        raise
    return backup_dir, published


def _rollback_publication(
    output_dir: Path,
    backup_dir: Path,
    published: Sequence[str],
    *,
    remove_completion: bool = True,
    restore_dir: Path | None = None,
) -> None:
    """Remove newly published files and atomically restore the prior canonical run."""
    for filename in published:
        published_path = output_dir / filename
        if restore_dir is None:
            published_path.unlink(missing_ok=True)
        elif published_path.exists():
            os.replace(published_path, restore_dir / filename)
    if remove_completion:
        for completion_name in _COMPLETION_FILENAMES:
            (output_dir / completion_name).unlink(missing_ok=True)
    for backup in backup_dir.iterdir():
        os.replace(backup, output_dir / backup.name)
    shutil.rmtree(backup_dir, ignore_errors=True)


def _assert_no_cross_kind_completion(output_dir: Path, *, run_kind: str) -> None:
    """Refuse to overwrite a run completed under the other run kind."""
    other = "complete.json" if run_kind == "diagnostic" else "diagnostic_complete.json"
    if (output_dir / other).exists():
        raise RuntimeError(
            "refusing to replace an output directory completed under a different run kind"
        )


def _assert_no_cross_kind_test_completion(output_dir: Path, *, run_kind: str) -> None:
    """Refuse to overwrite a test-stage result completed under the other run kind."""
    other = "test_complete.json" if run_kind == "diagnostic" else "diagnostic_test_complete.json"
    if (output_dir / other).exists():
        raise RuntimeError(
            "refusing to replace a test-stage result completed under a different run kind"
        )


def _test_stage_filenames(run_kind: str | None) -> tuple[str, str]:
    """Return ``(report_filename, test_complete_filename)`` for this run kind.

    Mirrors the formal/diagnostic split of ``_COMPLETION_FILENAMES``: a
    diagnostic run must never write the formal test-stage names.
    """
    if run_kind == "diagnostic":
        return "diagnostic_test_report.json", "diagnostic_test_complete.json"
    return "test_report.json", "test_complete.json"


def _clear_stale_test_artifacts(
    output_dir: Path, *, report_filename: str, test_complete_name: str
) -> None:
    """Remove a prior run's same-kind test sentinel before the test stage starts.

    Republishing a checkpoint (pack -> train -> publish onto an existing
    output directory) never touches ``test_report.json``/``test_complete.json``
    -- they live outside `_PUBLISHED_FILENAMES` and outside publish/rollback
    entirely. Left alone, a scoring failure (or a process death) on the *new*
    checkpoint would leave the *previous* checkpoint's completion sentinel
    sitting beside it, falsely certifying the new one as tested.

    The sentinel is unlinked first and the report second: a crash between the
    two unlinks then leaves, at worst, a stale report with no sentinel next to
    it -- never a sentinel a reader could mistake for certifying the
    checkpoint currently on disk.
    """
    (output_dir / test_complete_name).unlink(missing_ok=True)
    (output_dir / report_filename).unlink(missing_ok=True)


def _run_pipeline_unlocked(
    args: PipelineArgs,
    *,
    command_runner: CommandRunner = run_command,
    training_command_runner: TrainingCommandRunner = run_logged_command,
    stage_runner: StageRunner = run_stage,
) -> int:
    """Execute the cold E2 pipeline and return 0 on success or 2 on failure.

    Sub-stages, ``pack -> train -> publish -> test``: (1) load and validate the
    config/runtime; (2) record whether the pack path was initially absent;
    (3) build or strictly validate the pack; (4) launch
    one clean ``train`` process group at ``runtime.token_budget``; (5) validate
    the worker's runtime profile and every staged artifact against
    ``cfg.optim.epochs``; (6) merge stage/profile data into ``profile.json`` and write
    ``artifact_manifest.json`` with SHA-256 and byte size for ``best.pt``,
    ``last.pt``, ``metrics.jsonl``, ``run_metadata.json``, ``profile.json``, plus
    the E2E ``val_region_validation_events.jsonl`` when applicable; (7) publish the
    validated attempt artifact tree into the canonical output directory with per-file
    atomic replaces and full rollback; and (8), only once publication has
    committed, run the published ``best.pt`` through the held-out test
    protocol, recording its duration into
    ``profile.json``'s ``stage_seconds["test"]`` and writing ``test_report.json``
    then ``test_complete.json`` last (``diagnostic_*`` names under
    ``run_kind == "diagnostic"``). Immediately before the test stage starts, any
    same-kind ``test_report.json``/``test_complete.json`` left by a prior run on
    this output directory is removed first, so a republish's new checkpoint can
    never be found sitting beside a completion sentinel that certified the
    checkpoint it replaced.

    The test stage never rolls back publication: a held-out test-access ledger
    record must not be spent on a checkpoint that is later rolled back (so
    testing only ever runs after a committed publish), and a test-stage
    failure must not discard an otherwise valid trained model (so its failure
    path leaves ``complete.json`` and every published artifact untouched and
    reports through ``failure.json`` alone). A bounded debug run
    (``--max-steps``) returns before publication is even attempted, so it
    never spends a held-out scoring epoch.

    Every scoring subprocess call goes through ``command_runner`` with
    ``check=False``. Training goes through ``training_command_runner`` so worker
    output is durably streamed into the attempt's ``train.log``. Every failure
    after training starts stops immediately;
    nothing is published until the whole attempt artifact tree has been
    validated.

    Args:
        args: Parsed pipeline CLI arguments.
        command_runner: Injectable subprocess seam (real subprocesses in
            production; a fake in tests) for scoring fan-out.
        training_command_runner: Injectable training subprocess seam; production
            uses a durable merged-output ``train.log`` writer.
        stage_runner: Injectable supervised Python-stage seam for pack,
            artifact, and test execution.

    Returns:
        0 on success, 2 on failure (see the stage in ``failure.json``).
    """
    # Deferred import: workers import ProbeResult from *this* module at their
    # own top level, so importing them back at this module's top level would
    # create an import cycle. A call-time import breaks it in both directions.
    import importlib

    from src.data.packed_features import sha256_file

    worker = importlib.import_module(args.worker_module)

    pipeline_started = time.monotonic()
    cfg = worker.load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.run_kind is not None:
        if not hasattr(cfg, "run_kind"):
            raise ValueError("--run-kind is only supported by an E2E-aware worker")
        cfg = replace(cfg, run_kind=args.run_kind)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    diagnostic_run = args.run_kind == "diagnostic"
    if cfg.runtime is None:
        raise ValueError("the E2 pipeline requires a config with a 'runtime:' section")
    runtime = cfg.runtime
    if runtime.world_size == 0:
        runtime = replace(runtime, world_size=detect_visible_gpu_count())
        cfg = replace(cfg, runtime=runtime)
    debug_run = args.max_steps is not None
    formal_output_dir = cfg.output_dir
    output_dir = (
        formal_output_dir.with_name(f"{formal_output_dir.name}_debug")
        if debug_run
        else formal_output_dir
    )
    cfg = replace(cfg, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    attempt_dir = output_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    train_log_path = attempt_dir / "train.log"
    train_log_path.touch()
    attempt_status_path = attempt_dir / "status.json"
    current_attempt_path = output_dir / "current_attempt.json"
    attempt_status: dict[str, object] = {
        "attempt_id": attempt_id,
        "status": "running",
        "started_at_unix_seconds": time.time(),
    }
    resolved_resume_attempt: Path | None = None
    if args.resume_attempt is not None:
        attempt_status["resume_attempt"] = str(args.resume_attempt)
    _write_json_atomic(attempt_status_path, attempt_status)
    _write_json_atomic(current_attempt_path, attempt_status)
    (output_dir / "failure.json").unlink(missing_ok=True)
    pack_dir = args.pack_dir if args.pack_dir is not None else runtime.pack_dir

    def attempt_failure_extra(extra: Mapping[str, object] | None) -> dict[str, object]:
        progress: object | None = None
        progress_path = attempt_dir / "progress.json"
        if progress_path.is_file():
            with suppress(OSError, json.JSONDecodeError):
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
        log_tail = ""
        with suppress(OSError):
            log_tail = _read_log_tail(train_log_path)
        attempt_relative = Path("attempts") / attempt_id
        failure_extra: dict[str, object] = dict(extra or {})
        failure_extra.update(
            {
                "attempt_id": attempt_id,
                "attempt_path": str(attempt_relative),
                "log_path": str(attempt_relative / "train.log"),
                "log_tail": log_tail,
                "last_progress": progress,
            }
        )
        return failure_extra

    def fail(*, stage: str, message: str, extra: Mapping[str, object] | None = None) -> int:
        failure_extra = attempt_failure_extra(extra)
        terminal_status = {**attempt_status, "status": "failed", "stage": stage}
        _write_json_atomic(attempt_status_path, terminal_status)
        _write_json_atomic(current_attempt_path, terminal_status)
        write_failure(attempt_dir, stage=stage, message=message, extra=failure_extra)
        write_failure(output_dir, stage=stage, message=message, extra=failure_extra)
        return 2

    def fail_after_publication(
        *, stage: str, message: str, extra: Mapping[str, object] | None = None
    ) -> int:
        """Record a post-publication stage failure without touching publication.

        ``fail()`` is written for pre-publication failures: at that point
        ``complete.json`` does not exist yet, so retaining the attempt directory
        is the complete failure evidence. By the time the test stage
        runs, `_publish_staged` has already committed a valid, real
        ``complete.json`` and checkpoint; a scoring failure is a fact about the
        *test* stage only; and a held-out test-access ledger record for this
        ``(arm, seed)`` cannot be un-spent by rolling training back. So this
        path never calls `_rollback_publication` and never removes anything
        the publish stage wrote — it only records `failure.json` alongside the
        still-valid `complete.json`.
        """
        failure_extra = attempt_failure_extra(extra)
        terminal_status = {**attempt_status, "status": "failed", "stage": stage}
        _write_json_atomic(attempt_status_path, terminal_status)
        _write_json_atomic(current_attempt_path, terminal_status)
        write_failure(attempt_dir, stage=stage, message=message, extra=failure_extra)
        write_failure(output_dir, stage=stage, message=message, extra=failure_extra)
        return 2

    if args.resume_attempt is not None:
        try:
            import torch

            resume_attempt = args.resume_attempt.resolve(strict=True)
            resolved_resume_attempt = resume_attempt
            attempts_root = (output_dir / "attempts").resolve()
            if not resume_attempt.is_dir() or resume_attempt.parent != attempts_root:
                raise ValueError("--resume-attempt must be a direct child of output_dir/attempts")
            if resume_attempt == attempt_dir.resolve():
                raise ValueError("--resume-attempt must identify a prior attempt")
            source_status = json.loads((resume_attempt / "status.json").read_text(encoding="utf-8"))
            if not isinstance(source_status, dict):
                raise ValueError("resume attempt must have failed or abandoned terminal status")
            if source_status.get("status") == "running":
                source_status.update({"status": "abandoned", "abandoned_by_attempt_id": attempt_id})
                _write_json_atomic(resume_attempt / "status.json", source_status)
            elif source_status.get("status") not in {"failed", "abandoned"}:
                raise ValueError("resume attempt must be failed, abandoned, or orphaned running")
            state_path = resume_attempt / "training_state.pt"
            metrics_path = resume_attempt / "metrics.jsonl"
            checkpoints_dir = resume_attempt / "checkpoints"
            if (
                not state_path.is_file()
                or not metrics_path.is_file()
                or not checkpoints_dir.is_dir()
            ):
                raise ValueError(
                    "resume attempt is missing training_state.pt, metrics.jsonl, or checkpoints"
                )
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or state.get("resume_supported") is not True:
                raise ValueError("training_state.pt does not contain resumable state")
            required_state = {
                "config",
                "model_state",
                "optimizer",
                "scheduler",
                "world_size",
                "warmup_steps",
                "schedule_total_steps",
                "epoch",
                "global_step",
                "rng_by_rank",
                "kd_by_rank",
                "runtime_by_rank",
                "per_epoch_profiles",
                "evals_without_improvement",
                "counterfactual_stop_epoch",
            }
            if not required_state.issubset(state):
                raise ValueError("training_state.pt is incomplete")
            if state["world_size"] != runtime.world_size:
                raise ValueError("training_state.pt does not match the current world size")
            completed_epoch = state["epoch"]
            if (
                isinstance(completed_epoch, bool)
                or not isinstance(completed_epoch, int)
                or not 1 <= completed_epoch <= cfg.optim.epochs
            ):
                raise ValueError("training_state.pt epoch is not resumable for this config")
            if not isinstance(state["global_step"], int) or state["global_step"] <= 0:
                raise ValueError("training_state.pt global_step is invalid")
            for field in ("rng_by_rank", "kd_by_rank", "runtime_by_rank"):
                value = state[field]
                if not isinstance(value, list) or len(value) != runtime.world_size:
                    raise ValueError(f"training_state.pt {field} does not match world size")
            metric_rows = metrics_path.read_text(encoding="utf-8").splitlines()
            if len(metric_rows) < completed_epoch:
                raise ValueError("resume metrics.jsonl does not match the completed epoch")
            metric_rows = metric_rows[:completed_epoch]
            parsed_metric_rows: list[dict[str, object]] = []
            for expected_epoch, line in enumerate(metric_rows, start=1):
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
                    raise ValueError("resume metrics.jsonl epoch sequence is invalid")
                parsed_metric_rows.append(row)
            expected_checkpoints = [
                checkpoints_dir / f"epoch-{epoch:04d}.pt" for epoch in range(1, completed_epoch + 1)
            ]
            if any(not path.is_file() for path in expected_checkpoints):
                raise ValueError("resume attempt has an incomplete checkpoint sequence")
            for expected_epoch, checkpoint in enumerate(expected_checkpoints, start=1):
                candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (
                    not isinstance(candidate, dict)
                    or candidate.get("epoch") != expected_epoch
                    or candidate.get("selection_metrics") != parsed_metric_rows[expected_epoch - 1]
                ):
                    raise ValueError("resume checkpoint candidate does not match metrics prefix")
            (attempt_dir / "metrics.jsonl").write_text(
                "\n".join(metric_rows) + "\n", encoding="utf-8"
            )
            seeded_checkpoints = attempt_dir / "checkpoints"
            seeded_checkpoints.mkdir()
            for checkpoint in expected_checkpoints:
                os.link(checkpoint, seeded_checkpoints / checkpoint.name)
            attempt_status["resumed_from_attempt_id"] = resume_attempt.name
            _write_json_atomic(attempt_status_path, attempt_status)
            _write_json_atomic(current_attempt_path, attempt_status)
        except Exception as error:
            return fail(stage="train", message=f"resume attempt validation failed: {error}")

    stage_seconds: dict[str, float] = {}

    # --- pack: build (cold) or strictly validate (warm) ---
    required_pack_paths = getattr(worker, "required_pack_paths", None)
    try:
        pack_paths = (
            tuple(required_pack_paths(cfg, pack_dir))
            if callable(required_pack_paths)
            else (pack_dir,)
        )
    except Exception as error:
        return fail(stage="pack", message=f"required pack path resolution failed: {error}")
    if not pack_paths or any(not isinstance(path, Path) for path in pack_paths):
        return fail(stage="pack", message="worker returned invalid required pack paths")
    cold_cache = any(not path.exists() for path in pack_paths)
    pack_started = time.monotonic()
    pack_validation_path = attempt_dir / "pack_validation.json"
    pack_temp_prefix = f".{pack_dir.name}.{attempt_id}.pack-"

    def cleanup_owned_pack_temps() -> None:
        for parent in {path.parent for path in pack_paths}:
            if not parent.exists():
                continue
            for path in parent.glob(f"{pack_temp_prefix}*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    def pack_operation() -> None:
        payload = worker.prepare_pack(
            cfg, pack_dir, cold_cache=cold_cache, temp_prefix=pack_temp_prefix
        )
        _write_json_atomic(pack_validation_path, cast(dict[str, object], payload))

    try:
        stage_runner(pack_operation)
    except Exception as error:
        cleanup_owned_pack_temps()
        return fail(stage="pack", message=f"feature pack stage failed: {error}")
    stage_seconds["pack"] = time.monotonic() - pack_started
    try:
        pack_validation = json.loads(pack_validation_path.read_text(encoding="utf-8"))
        if not isinstance(pack_validation, dict):
            raise ValueError("pack validation result must be an object")
        pack_manifest_payload = cast(dict[str, object], pack_validation["pack_manifest"])
        pack_identity_sha256 = cast(str, pack_validation["pack_identity_sha256"])
        if not isinstance(pack_manifest_payload, dict) or not isinstance(pack_identity_sha256, str):
            raise TypeError("pack validation result has invalid fields")
    except Exception as error:
        return fail(stage="pack", message=f"pack validation result failed: {error}")
    pack_validation_path.unlink(missing_ok=True)
    pack_evidence = cast(dict[str, object], pack_validation.get("packs", {}))

    profile_path = attempt_dir / "profile.json"
    evidence_profile: dict[str, object] = {
        "cold_cache": cold_cache,
        "stage_seconds": dict(stage_seconds),
        "token_budget": runtime.token_budget,
        "pack_manifest": pack_manifest_payload,
        "pack_identity_sha256": pack_identity_sha256,
        "pack_evidence": pack_evidence,
        "world_size": runtime.world_size,
    }
    _write_json_atomic(profile_path, evidence_profile)

    # --- train: one clean process group at the configured token budget ---
    train_started = time.monotonic()
    worker_profile_path = attempt_dir / "worker_profile.json"
    worker_profile_path.unlink(missing_ok=True)
    try:
        command = build_accelerate_command(
            accelerate_bin=_resolve_accelerate_bin(),
            config_path=args.config,
            mode="train",
            pack_dir=pack_dir,
            output_dir=attempt_dir,
            token_budget=runtime.token_budget,
            profile_output=worker_profile_path,
            world_size=runtime.world_size,
            worker_module=args.worker_module,
            seed=args.seed,
            max_steps=args.max_steps,
            run_kind=args.run_kind,
            resume_attempt=resolved_resume_attempt,
        )
    except Exception as error:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(stage="train", message=f"training worker launch setup failed: {error}")
    try:
        completed = training_command_runner(command, train_log_path)
    except Exception as error:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(stage="train", message=f"training worker launch failed: {error}")
    if completed.returncode != 0:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(
            stage="train",
            message=f"training subprocess exited with code {completed.returncode}",
            extra={"returncode": completed.returncode},
        )
    stage_seconds["train"] = time.monotonic() - train_started

    # --- merge the worker runtime profile with pipeline-level fields ---
    worker_data: object | None = None
    try:
        worker_data = json.loads(worker_profile_path.read_text(encoding="utf-8"))
        worker_runtime_profile = _validate_worker_profile(
            worker_data,
            epochs=cfg.optim.epochs,
            world_size=runtime.world_size,
            memory_limit_gib=runtime.memory_limit_gib,
            allow_partial=debug_run,
            enforce_engineering_limits=not diagnostic_run,
        )
        completed_epochs = cast(int, worker_runtime_profile["epochs_completed"])
        _validate_staged_artifacts(
            attempt_dir,
            epochs=completed_epochs if debug_run else cfg.optim.epochs,
            model_family=cfg.model.family,
            allow_partial=debug_run,
            require_val_region_validation_events=(
                not debug_run and args.worker_module == "src.train_egostitch"
            ),
            expected_run_kind=(
                ("debug" if debug_run else (args.run_kind or "formal"))
                if args.worker_module == "src.train_egostitch"
                else None
            ),
        )
    except Exception as error:
        rejected_profile = {**evidence_profile}
        if isinstance(worker_data, dict):
            rejected_profile["rejected_worker_profile"] = worker_data
        _write_json_atomic(profile_path, rejected_profile)
        return fail(
            stage="artifacts",
            message=f"worker profile is missing or malformed: {error}",
        )
    final_profile: dict[str, object] = {
        **worker_runtime_profile,
        **evidence_profile,
    }
    if debug_run:
        final_profile["run_kind"] = "debug"
        _write_json_atomic(output_dir / "profile.json", final_profile)
        for filename in ("best.pt", "last.pt", "metrics.jsonl", "run_metadata.json"):
            shutil.copy2(attempt_dir / filename, output_dir / filename)
        _write_json_atomic(output_dir / "debug_complete.json", {"status": "debug_complete"})
        terminal_status = {**attempt_status, "status": "debug_complete"}
        _write_json_atomic(attempt_status_path, terminal_status)
        _write_json_atomic(current_attempt_path, terminal_status)
        return 0
    _write_json_atomic(profile_path, final_profile)
    artifacts_started = time.monotonic()
    optional_published_filenames = _OPTIONAL_PUBLISHED_FILENAMES

    def artifact_operation() -> None:
        """Publish immutable profile+manifest at the documented late timing cutoff."""
        manifest: dict[str, object] = {}
        non_profile_filenames = (
            "best.pt",
            "last.pt",
            "metrics.jsonl",
            "run_metadata.json",
            # Optional plan-required artifacts remain inside the integrity record.
            *(
                filename
                for filename in optional_published_filenames
                if (attempt_dir / filename).is_file()
            ),
        )
        for filename in non_profile_filenames:
            artifact_path = attempt_dir / filename
            manifest[filename] = {
                "sha256": sha256_file(artifact_path),
                "byte_size": artifact_path.stat().st_size,
            }
        # The recorded cutoff is immediately before the final profile write. The
        # profile is never rewritten after its digest is taken.
        cutoff = time.monotonic()
        final_profile["timing_cutoff"] = "before_final_profile_write"
        final_profile["stage_seconds"] = {
            **stage_seconds,
            "artifacts": cutoff - artifacts_started,
        }
        final_profile["total_seconds"] = cutoff - pipeline_started
        _write_json_atomic(profile_path, final_profile)
        manifest["profile.json"] = {
            "sha256": sha256_file(profile_path),
            "byte_size": profile_path.stat().st_size,
        }
        _write_json_atomic(attempt_dir / "artifact_manifest.json", manifest)

    try:
        stage_runner(artifact_operation)
    except Exception as error:
        _write_json_atomic(profile_path, final_profile)
        return fail(stage="artifacts", message=f"artifact merge/manifest failed: {error}")

    # Publish hardlinks from a temporary tree, leaving the complete attempt
    # evidence untouched. Publication remains reversible until its sentinel lands.
    publication_dir = Path(tempfile.mkdtemp(prefix=".e2-publish-", dir=output_dir))
    filenames_to_publish = _PUBLISHED_FILENAMES + tuple(
        filename for filename in optional_published_filenames if (attempt_dir / filename).is_file()
    )
    try:
        for filename in filenames_to_publish:
            os.link(attempt_dir / filename, publication_dir / filename)
    except Exception as error:
        shutil.rmtree(publication_dir, ignore_errors=True)
        return fail(stage="publication", message=f"publication staging failed: {error}")
    try:
        _assert_no_cross_kind_completion(output_dir, run_kind=args.run_kind or "formal")
        backup_dir, published = _publish_staged(
            publication_dir,
            output_dir,
            optional_filenames=optional_published_filenames,
        )
    except Exception as error:
        shutil.rmtree(publication_dir, ignore_errors=True)
        return fail(stage="publication", message=f"canonical publication failed: {error}")
    (output_dir / "failure.json").unlink(missing_ok=True)
    total_elapsed = time.monotonic() - pipeline_started
    try:
        completion_name = (
            "diagnostic_complete.json" if args.run_kind == "diagnostic" else "complete.json"
        )
        _write_json_atomic(
            output_dir / completion_name,
            {
                "status": completion_name.removesuffix(".json"),
                "attempt_id": attempt_id,
                "total_seconds": total_elapsed,
            },
        )
    except Exception as error:
        _rollback_publication(output_dir, backup_dir, published, restore_dir=publication_dir)
        shutil.rmtree(publication_dir, ignore_errors=True)
        return fail(stage="publication", message=f"completion sentinel failed: {error}")
    shutil.rmtree(backup_dir, ignore_errors=True)
    shutil.rmtree(publication_dir, ignore_errors=True)
    logger.info("E2 pipeline published: %.1fs elapsed", total_elapsed)

    # --- test: run the published checkpoint through the held-out test protocol ---
    # Deferred imports, matching the import-cycle precedent at the top of this
    # function: a bounded debug run never reaches this branch (it already
    # returned above), so its heavier scoring/eval import graph never loads on
    # that path, and the two modules stay free to import this one back.
    from src.eval.test_protocol import run_test_protocol
    from src.score_fanout import score_sharded

    report_filename, test_complete_name = _test_stage_filenames(args.run_kind)
    test_started = time.monotonic()

    def score_runner(score_args: Sequence[str]) -> Path:
        return score_sharded(
            score_args,
            gpu_count=runtime.world_size,
            python_bin=Path(sys.executable),
            command_runner=command_runner,
        )

    def test_operation() -> None:
        """Score `best.pt`, then fold the stage duration into the published profile."""
        published_metadata = json.loads(
            (output_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        metadata_arm = (
            published_metadata.get("arm") if isinstance(published_metadata, dict) else None
        )
        # Only egostitch_e2e run_metadata carries a real `arm`; every other
        # worker has no experimental-arm axis, so its model family stands in
        # as the test-report's arm identity.
        arm = metadata_arm if isinstance(metadata_arm, str) and metadata_arm else cfg.model.family
        run_test_protocol(
            checkpoint=output_dir / "best.pt",
            output_dir=output_dir,
            data_root=cfg.data.root,
            strategy=cfg.data.strategy,
            arm=arm,
            seed=cfg.seed,
            score_runner=score_runner,
            pack_dir=pack_dir,
            rescore_reason=args.rescore_reason,
            report_filename=report_filename,
            # A diagnostic run is the true-Oracle path, whose generator consumes
            # held-out positive structure; `score_universe` refuses this flag for
            # any other generator, so a non-oracle diagnostic fails closed here
            # rather than being silently scored as one.
            allow_oracle_diagnostic=args.run_kind == "diagnostic",
        )
        test_duration = time.monotonic() - test_started
        published_profile_path = output_dir / "profile.json"
        published_profile = cast(
            dict[str, object], json.loads(published_profile_path.read_text(encoding="utf-8"))
        )
        stage_seconds_value = published_profile.get("stage_seconds")
        published_stage_seconds: dict[str, object] = (
            dict(stage_seconds_value) if isinstance(stage_seconds_value, dict) else {}
        )
        published_stage_seconds["test"] = test_duration
        published_profile["stage_seconds"] = published_stage_seconds
        # Recomputed from the pipeline's own clock origin, not the published
        # figure plus `test_duration`. The published figure is measured at the
        # artifact cutoff, *before* publication, so accumulating onto it would
        # leave `total_seconds` excluding publication entirely -- and smaller
        # than `complete.json`'s own total whenever publication outlasts the
        # test stage.
        _finite_number(published_profile.get("total_seconds"), field="profile.total_seconds")
        published_profile["total_seconds"] = time.monotonic() - pipeline_started
        _write_json_atomic(published_profile_path, published_profile)
        # The manifest records profile.json's own digest, so touching the
        # profile after publication means re-hashing it here too -- otherwise
        # the manifest would silently stop matching the bytes on disk.
        manifest_path = output_dir / "artifact_manifest.json"
        manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["profile.json"] = {
            "sha256": sha256_file(published_profile_path),
            "byte_size": published_profile_path.stat().st_size,
        }
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(
            output_dir / test_complete_name,
            {"status": test_complete_name.removesuffix(".json")},
        )

    try:
        _assert_no_cross_kind_test_completion(output_dir, run_kind=args.run_kind or "formal")
        _clear_stale_test_artifacts(
            output_dir, report_filename=report_filename, test_complete_name=test_complete_name
        )
        stage_runner(test_operation)
    except Exception as error:
        message = f"held-out test stage failed: {error}"
        if "requires --rescore-reason" in str(error):
            message += (
                ". This (arm, seed) already has a held-out scoring epoch; re-issue "
                f"with an explicit reason: {_pipeline_rerun_command(args)}"
            )
        return fail_after_publication(stage="test", message=message)

    terminal_status = {**attempt_status, "status": "complete"}
    _write_json_atomic(attempt_dir / "complete.json", terminal_status)
    _write_json_atomic(attempt_status_path, terminal_status)
    _write_json_atomic(current_attempt_path, terminal_status)
    logger.info("E2 pipeline complete: %.1fs elapsed", time.monotonic() - pipeline_started)
    return 0


def run_pipeline(
    args: PipelineArgs,
    *,
    command_runner: CommandRunner = run_command,
    training_command_runner: TrainingCommandRunner = run_logged_command,
    stage_runner: StageRunner = run_stage,
) -> int:
    """Run one exclusive pipeline attempt for the resolved output directory."""
    import importlib

    worker = importlib.import_module(args.worker_module)
    cfg = worker.load_config(args.config)
    formal_output_dir = args.output_dir if args.output_dir is not None else cfg.output_dir
    output_dir = (
        formal_output_dir.with_name(f"{formal_output_dir.name}_debug")
        if args.max_steps is not None
        else formal_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".pipeline.lock").open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error("another pipeline is active for %s", output_dir)
            return 2
        try:
            return _run_pipeline_unlocked(
                args,
                command_runner=command_runner,
                training_command_runner=training_command_runner,
                stage_runner=stage_runner,
            )
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m src.e2_pipeline``.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The ``run_pipeline`` exit code (0 success, 2 gated failure).
    """
    args = parse_pipeline_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    raise SystemExit(main())
