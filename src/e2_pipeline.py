"""E2 pipeline worker payloads, failure artifacts, and orchestration.

Provides pure dataclasses and functions for:
- ProbeResult: one worker-reported throughput/memory measurement
- write_failure(): atomic failure JSON artifact writer

And the production orchestrator (``python -m src.e2_pipeline``) that drives the
cold multi-H20 run end to end in three sub-stages — ``pack -> train ->
publish``: pack-or-validate the BF16 feature cache, launch one clean
``accelerate launch`` ``train`` process group at the configured
``runtime.token_budget``, merge the worker's runtime profile with pipeline-level
fields, and publish the validated staging tree atomically. See
:func:`run_pipeline` for the full contract.

EgoStitch E2E uses this same orchestrator, selected via ``--worker-module
src.train_egostitch``.
"""

import argparse
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
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
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
    """

    config: Path
    pack_dir: Path | None
    output_dir: Path | None
    worker_module: str = "src.train_b0"
    seed: int | None = None
    max_steps: int | None = None
    run_kind: str | None = None


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
    parser.add_argument(
        "--run-kind",
        choices=("formal",),
        default=None,
        help=(
            "EgoStitch E2E execution context; forwarded unchanged to the worker. "
            "'debug' is not selectable: the worker derives it from --max-steps"
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
    namespace = parser.parse_args(argv)
    return PipelineArgs(
        config=namespace.config,
        pack_dir=namespace.pack_dir,
        output_dir=namespace.output_dir,
        worker_module=namespace.worker_module,
        seed=namespace.seed,
        max_steps=namespace.max_steps,
        run_kind=namespace.run_kind,
    )


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
StageRunner = Callable[[Callable[[], None], float], None]


def run_command(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in a killable process group on the fixed POSIX/Linux target.

    Args:
        command: The argv list to execute.
        timeout_seconds: Hard wall-clock timeout; raises
            ``subprocess.TimeoutExpired`` if exceeded.

    Returns:
        The ``CompletedProcess`` (``check=False``: callers inspect ``.returncode``).
    """
    process = subprocess.Popen(
        list(command),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            error.cmd, error.timeout, output=stdout, stderr=stderr
        ) from None
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


def _run_stage_child(operation: Callable[[], None], sender: Connection) -> None:
    """Execute a Python stage in its own session and report its exception."""
    os.setsid()
    try:
        operation()
    except BaseException as error:
        sender.send((False, f"{type(error).__name__}: {error}"))
    else:
        sender.send((True, ""))
    finally:
        sender.close()


def run_stage(operation: Callable[[], None], timeout_seconds: float) -> None:
    """Run a Python stage under a killable POSIX/Linux process-group deadline."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_stage_child, args=(operation, sender))
    process.start()
    sender.close()
    process.join(timeout_seconds)
    if process.is_alive():
        if process.pid is None:  # pragma: no cover - start() above guarantees a PID
            raise RuntimeError("supervised stage has no process ID")
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.join()
        receiver.close()
        raise subprocess.TimeoutExpired(cmd="python stage", timeout=timeout_seconds)
    if not receiver.poll():
        receiver.close()
        raise RuntimeError(f"supervised stage exited with code {process.exitcode} without a result")
    succeeded, message = cast(tuple[bool, str], receiver.recv())
    receiver.close()
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
) -> dict[str, object]:
    """Validate the formal worker's complete runtime-profile contract."""
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
    if hit_rate != 1.0:
        raise ValueError("feature_cache_hit_rate must be exactly 1.0")
    memories = data["peak_memory_gib_per_rank"]
    if not isinstance(memories, list) or len(memories) != world_size:
        raise ValueError(f"peak_memory_gib_per_rank must contain {world_size} values")
    for rank, value in enumerate(memories):
        memory = _finite_number(value, field=f"peak_memory_gib_per_rank[{rank}]")
        if memory > memory_limit_gib:
            raise ValueError(f"rank {rank} peak memory exceeds {memory_limit_gib} GiB")
    wait_fraction = _finite_number(
        data["steady_state_data_wait_fraction"], field="steady_state_data_wait_fraction"
    )
    if wait_fraction > 0.05:
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
V_HOLD_VALIDATION_EVENTS_FILENAME = "v_hold_validation_events.jsonl"
_OPTIONAL_PUBLISHED_FILENAMES = (
    V_HOLD_VALIDATION_EVENTS_FILENAME,
)


def _validate_staged_artifacts(
    staging_dir: Path,
    *,
    epochs: int,
    model_family: str,
    allow_partial: bool = False,
    require_v_hold_validation_events: bool = False,
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
    if require_v_hold_validation_events:
        evidence = metadata.get("v_hold_validation_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("run_metadata.json is missing V_hold validation evidence")
        if evidence.get("schema") != "egostitch_e2e_v_hold_validation_events_v1":
            raise ValueError("V_hold validation-event evidence has an invalid schema")
        if evidence.get("path") != V_HOLD_VALIDATION_EVENTS_FILENAME:
            raise ValueError("V_hold validation-event evidence has an invalid path")
        count = evidence.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("V_hold validation-event evidence has an invalid count")
        ledger_path = staging_dir / V_HOLD_VALIDATION_EVENTS_FILENAME
        if not ledger_path.is_file() or ledger_path.stat().st_size <= 0:
            raise ValueError(f"{V_HOLD_VALIDATION_EVENTS_FILENAME} is missing or empty")
        ledger_rows: list[object] = []
        with ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("V_hold validation-event ledger contains a blank row")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("V_hold validation-event ledger row must be an object")
                ledger_rows.append(row)
        if len(ledger_rows) != count:
            raise ValueError("V_hold validation-event ledger count does not match metadata")
        digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        if evidence.get("sha256") != digest:
            raise ValueError("V_hold validation-event ledger digest does not match metadata")
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
        completion = output_dir / "complete.json"
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
        )
        raise
    return backup_dir, published


def _rollback_publication(
    output_dir: Path,
    backup_dir: Path,
    published: Sequence[str],
    *,
    remove_completion: bool = True,
) -> None:
    """Remove newly published files and atomically restore the prior canonical run."""
    for filename in published:
        (output_dir / filename).unlink(missing_ok=True)
    if remove_completion:
        (output_dir / "complete.json").unlink(missing_ok=True)
    for backup in backup_dir.iterdir():
        os.replace(backup, output_dir / backup.name)
    shutil.rmtree(backup_dir, ignore_errors=True)


def run_pipeline(
    args: PipelineArgs,
    *,
    command_runner: CommandRunner = run_command,
    stage_runner: StageRunner = run_stage,
) -> int:
    """Execute the cold E2 pipeline and return 0 on success or 2 on failure.

    Sub-stages, ``pack -> train -> publish``: (1) load and validate the
    config/runtime; (2) record whether the pack path was initially absent;
    (3) build or strictly validate the pack within the pack budget; (4) launch
    one clean ``train`` process group at ``runtime.token_budget``; (5) validate
    the worker's runtime profile and every staged artifact against
    ``cfg.optim.epochs``; (6) merge stage/profile data into ``profile.json`` and write
    ``artifact_manifest.json`` with SHA-256 and byte size for ``best.pt``,
    ``last.pt``, ``metrics.jsonl``, ``run_metadata.json``, ``profile.json``, plus
    the E2E ``v_hold_validation_events.jsonl`` when applicable; and (7)
    publish the validated staging tree into the canonical output directory with
    per-file atomic replaces and full rollback.

    Every subprocess call goes through ``command_runner`` with ``check=False``
    and an explicit timeout (``runtime.train_eval_budget_seconds`` for
    training). Every failure after training starts stops immediately; nothing is
    published until the whole staging tree has been validated.

    Args:
        args: Parsed pipeline CLI arguments.
        command_runner: Injectable subprocess seam (real subprocesses in
            production; a fake in tests).
        stage_runner: Injectable supervised Python-stage seam for pack and
            artifact deadlines.

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
    staging_dir = Path(tempfile.mkdtemp(prefix=".e2-run-", dir=output_dir))
    pack_dir = args.pack_dir if args.pack_dir is not None else runtime.pack_dir

    def fail(*, stage: str, message: str, extra: Mapping[str, object] | None = None) -> int:
        staged_profile = staging_dir / "profile.json"
        if staged_profile.is_file():
            with suppress(OSError):
                os.replace(staged_profile, output_dir / "failed_run_profile.json")
        staged_history = staging_dir / "failed_run_history.json"
        if staged_history.is_file():
            with suppress(OSError):
                os.replace(staged_history, output_dir / "failed_run_history.json")
        else:
            # A failure before training produces no history; a prior run's
            # retained history must not masquerade as this failure's evidence.
            (output_dir / "failed_run_history.json").unlink(missing_ok=True)
        write_failure(output_dir, stage=stage, message=message, extra=extra)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 2

    stage_seconds: dict[str, float] = {}

    # --- pack: build (cold) or strictly validate (warm) within the pack budget ---
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
    pack_validation_path = staging_dir / "pack_validation.json"
    pack_temp_prefix = f".{pack_dir.name}.{staging_dir.name}.pack-"

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
        stage_runner(pack_operation, float(runtime.pack_budget_seconds))
    except subprocess.TimeoutExpired:
        cleanup_owned_pack_temps()
        return fail(
            stage="pack",
            message=f"feature pack stage timed out after {runtime.pack_budget_seconds}s",
            extra={"timeout_seconds": runtime.pack_budget_seconds},
        )
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

    profile_path = staging_dir / "profile.json"
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
    worker_profile_path = staging_dir / "train_worker_profile.json"
    worker_profile_path.unlink(missing_ok=True)
    try:
        command = build_accelerate_command(
            accelerate_bin=_resolve_accelerate_bin(),
            config_path=args.config,
            mode="train",
            pack_dir=pack_dir,
            output_dir=output_dir if debug_run else staging_dir,
            token_budget=runtime.token_budget,
            profile_output=worker_profile_path,
            world_size=runtime.world_size,
            worker_module=args.worker_module,
            seed=args.seed,
            max_steps=args.max_steps,
            run_kind=args.run_kind,
        )
    except Exception as error:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(stage="train", message=f"training worker launch setup failed: {error}")
    try:
        completed = command_runner(command, float(runtime.train_eval_budget_seconds))
    except subprocess.TimeoutExpired:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(
            stage="train",
            message=f"training timed out after {runtime.train_eval_budget_seconds}s",
            extra={"timeout_seconds": runtime.train_eval_budget_seconds},
        )
    except Exception as error:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(stage="train", message=f"training worker launch failed: {error}")
    if completed.returncode != 0:
        _write_json_atomic(profile_path, evidence_profile)
        return fail(
            stage="train",
            message=f"training subprocess exited with code {completed.returncode}",
            extra={"returncode": completed.returncode, "stderr": completed.stderr},
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
        )
        completed_epochs = cast(int, worker_runtime_profile["epochs_completed"])
        _validate_staged_artifacts(
            output_dir if debug_run else staging_dir,
            epochs=completed_epochs if debug_run else cfg.optim.epochs,
            model_family=cfg.model.family,
            allow_partial=debug_run,
            require_v_hold_validation_events=(
                not debug_run and args.run_kind == "formal"
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
    worker_profile_path.unlink(missing_ok=True)

    final_profile: dict[str, object] = {
        **worker_runtime_profile,
        **evidence_profile,
    }
    if debug_run:
        final_profile["run_kind"] = "debug"
        _write_json_atomic(output_dir / "profile.json", final_profile)
        _write_json_atomic(output_dir / "debug_complete.json", {"status": "debug_complete"})
        shutil.rmtree(staging_dir, ignore_errors=True)
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
                if (staging_dir / filename).is_file()
            ),
        )
        for filename in non_profile_filenames:
            artifact_path = staging_dir / filename
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
        _write_json_atomic(staging_dir / "artifact_manifest.json", manifest)

    try:
        stage_runner(artifact_operation, float(runtime.artifact_budget_seconds))
    except subprocess.TimeoutExpired:
        _write_json_atomic(profile_path, final_profile)
        return fail(
            stage="artifacts",
            message=f"artifact stage timed out after {runtime.artifact_budget_seconds}s",
            extra={"timeout_seconds": runtime.artifact_budget_seconds},
        )
    except Exception as error:
        _write_json_atomic(profile_path, final_profile)
        return fail(stage="artifacts", message=f"artifact merge/manifest failed: {error}")

    # The staging tree is now complete and validated. Publication is reversible
    # until the success sentinel lands.
    try:
        backup_dir, published = _publish_staged(
            staging_dir,
            output_dir,
            optional_filenames=optional_published_filenames,
        )
    except Exception as error:
        return fail(stage="publication", message=f"canonical publication failed: {error}")
    (output_dir / "failure.json").unlink(missing_ok=True)
    (output_dir / "failed_run_profile.json").unlink(missing_ok=True)
    (output_dir / "failed_run_history.json").unlink(missing_ok=True)
    total_elapsed = time.monotonic() - pipeline_started
    b0_formal_cold = (
        not debug_run
        and cold_cache
        and cfg.model.family in ("v3_1", "f0_mlp")
        and args.run_kind is None
    )
    if b0_formal_cold and total_elapsed > runtime.total_budget_seconds:
        _rollback_publication(output_dir, backup_dir, published)
        return fail(
            stage="total_budget",
            message=(
                f"cold formal pipeline elapsed {total_elapsed:.1f}s exceeds "
                f"{runtime.total_budget_seconds}s"
            ),
            extra={
                "elapsed_seconds": total_elapsed,
                "limit_seconds": runtime.total_budget_seconds,
            },
        )
    try:
        _write_json_atomic(
            output_dir / "complete.json",
            {"status": "complete", "total_seconds": total_elapsed},
        )
    except Exception as error:
        _rollback_publication(output_dir, backup_dir, published)
        return fail(stage="publication", message=f"completion sentinel failed: {error}")
    shutil.rmtree(backup_dir, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info("E2 pipeline complete: %.1fs elapsed", total_elapsed)
    return 0


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
