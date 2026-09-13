"""Isolated, debug-only benchmark for the production task + structural train step.

Run this module from each independently checked-out implementation (legacy,
coordinate reuse, and distributed structural scoring).  It restores one
epoch-boundary training snapshot, performs real optimizer updates through the
production ``src.train_b0`` DDP worker, discards those updates, and writes a
single JSON report.  It never publishes a checkpoint or validation result.

Example::

    python -m src.experiments.struct_pipeline_benchmark \
      --config configs/split_seed42/topo_prompt_full_struct.yaml \
      --resume-attempt outputs/.../attempts/<id> \
      --output /tmp/full-round-1.json --variant full --round-index 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch
from accelerate import Accelerator
from numpy.typing import NDArray

from src import train_b0
from src.data.struct_coords import StructCoordinateTable
from src.e2_pipeline import detect_visible_gpu_count


@dataclass
class StepTiming:
    """Rank-local measurements for one real optimizer update."""

    step: int
    started_at_unix_seconds: float
    finished_at_unix_seconds: float
    local_task_pairs: int
    global_task_pairs: int
    struct_pairs_contribution: float
    step_wall_seconds: float
    data_wait_seconds: float
    coordinate_prepare_seconds: float
    coordinate_prepare_forward_seconds: float
    coordinate_prepare_backward_seconds: float
    coordinate_prepare_forward_calls: int
    coordinate_prepare_backward_calls: int
    struct_forward_seconds: float
    shared_backward_seconds: float
    memory_allocated_gib: float
    memory_reserved_gib: float
    memory_current_allocated_gib: float
    memory_current_reserved_gib: float


class _BenchmarkComplete(BaseException):
    """Stop the production loop after the requested update count."""


class _WorkerComplete(BaseException):
    """Return successfully from ``train_b0.main`` without finalization."""


class _Collector:
    def __init__(self, *, warmup_steps: int, timed_steps: int, device: torch.device) -> None:
        self.warmup_steps = warmup_steps
        self.timed_steps = timed_steps
        self.total_steps = warmup_steps + timed_steps
        self.device = device
        self.records: list[StepTiming] = []
        self.update_count = 0
        self._step_started: float | None = None
        self._step_started_unix: float | None = None
        self._data_wait = 0.0
        self._coordinate_prepare = 0.0
        self._coordinate_prepare_forward = 0.0
        self._coordinate_prepare_backward = 0.0
        self._coordinate_prepare_forward_calls = 0
        self._coordinate_prepare_backward_calls = 0
        self.in_backward = False
        self._struct_forward = 0.0
        self._backward = 0.0
        self._local_pairs = 0
        self._global_pairs = 0
        self._struct_pairs = 0.0
        self._update_finished = False

    @property
    def use_cuda(self) -> bool:
        return self.device.type == "cuda"

    def synchronize(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

    def begin_batch(self) -> float:
        if self._update_finished:
            self._finish_previous_update()
        if self._step_started is not None:
            raise RuntimeError("benchmark observed a new batch before the previous update finished")
        self.synchronize()
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._step_started = time.monotonic()
        self._step_started_unix = time.time()
        return self._step_started

    def set_batch(self, batch: train_b0.Batch, *, waited: float, world_size: int) -> None:
        self._data_wait = waited
        self._local_pairs, self._global_pairs = train_b0._batch_pair_counts(batch, world_size)

    def add_coordinate_prepare(self, seconds: float) -> None:
        self._coordinate_prepare += seconds
        if self.in_backward:
            self._coordinate_prepare_backward += seconds
            self._coordinate_prepare_backward_calls += 1
        else:
            self._coordinate_prepare_forward += seconds
            self._coordinate_prepare_forward_calls += 1

    def add_struct_forward(self, seconds: float, struct_pairs: float) -> None:
        self._struct_forward += seconds
        self._struct_pairs += struct_pairs

    def add_backward(self, seconds: float) -> None:
        self._backward += seconds

    def mark_update_finished(self) -> None:
        if self._step_started is None or self._update_finished:
            raise RuntimeError("benchmark optimizer update had no matching live batch")
        self.update_count += 1
        self._update_finished = True

    def _finish_previous_update(self) -> None:
        if self._step_started is None or self._step_started_unix is None:
            raise RuntimeError("benchmark optimizer update had no matching batch")
        self.synchronize()
        wall = time.monotonic() - self._step_started
        finished_at = time.time()
        allocated = 0.0
        reserved = 0.0
        current_allocated = 0.0
        current_reserved = 0.0
        if self.use_cuda:
            allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
            current_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            current_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        if self.update_count > self.warmup_steps:
            self.records.append(
                StepTiming(
                    step=self.update_count - self.warmup_steps,
                    started_at_unix_seconds=self._step_started_unix,
                    finished_at_unix_seconds=finished_at,
                    local_task_pairs=self._local_pairs,
                    global_task_pairs=self._global_pairs,
                    struct_pairs_contribution=self._struct_pairs,
                    step_wall_seconds=wall,
                    data_wait_seconds=self._data_wait,
                    coordinate_prepare_seconds=self._coordinate_prepare,
                    coordinate_prepare_forward_seconds=self._coordinate_prepare_forward,
                    coordinate_prepare_backward_seconds=self._coordinate_prepare_backward,
                    coordinate_prepare_forward_calls=self._coordinate_prepare_forward_calls,
                    coordinate_prepare_backward_calls=self._coordinate_prepare_backward_calls,
                    struct_forward_seconds=self._struct_forward,
                    shared_backward_seconds=self._backward,
                    memory_allocated_gib=allocated,
                    memory_reserved_gib=reserved,
                    memory_current_allocated_gib=current_allocated,
                    memory_current_reserved_gib=current_reserved,
                )
            )
        self._step_started = None
        self._step_started_unix = None
        self._data_wait = 0.0
        self._coordinate_prepare = 0.0
        self._coordinate_prepare_forward = 0.0
        self._coordinate_prepare_backward = 0.0
        self._coordinate_prepare_forward_calls = 0
        self._coordinate_prepare_backward_calls = 0
        self._struct_forward = 0.0
        self._backward = 0.0
        self._local_pairs = 0
        self._global_pairs = 0
        self._struct_pairs = 0.0
        self._update_finished = False
        if self.update_count >= self.total_steps:
            raise _BenchmarkComplete


class _TimedIterable(Iterable[train_b0.Batch]):
    def __init__(
        self,
        source: Iterable[train_b0.Batch],
        collector: _Collector,
        *,
        world_size: int,
    ) -> None:
        self._source = source
        self._collector = collector
        self._world_size = world_size

    def __len__(self) -> int:
        if not hasattr(self._source, "__len__"):
            raise TypeError("wrapped training loader has no length")
        return len(cast(Any, self._source))

    def __iter__(self) -> Iterator[train_b0.Batch]:
        iterator = iter(self._source)
        while True:
            started = self._collector.begin_batch()
            try:
                batch = next(iterator)
            except StopIteration:
                self._collector._step_started = None
                self._collector._step_started_unix = None
                return
            waited = time.monotonic() - started
            self._collector.set_batch(batch, waited=waited, world_size=self._world_size)
            yield batch


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _install_worker_hooks(
    *,
    warmup_steps: int,
    timed_steps: int,
    rank_output_dir: Path,
) -> Callable[[], None]:
    """Patch timing around the production loop inside one Accelerate worker."""
    original_loop = train_b0.train_ddp_loop
    original_backward = Accelerator.backward
    original_scheduler_step = train_b0._step_scheduler
    original_struct_loss = train_b0.StructStream.loss
    original_coords = StructCoordinateTable.coords_for_pairs

    collector_box: list[_Collector | None] = [None]

    def collector() -> _Collector:
        value = collector_box[0]
        if value is None:
            raise RuntimeError("benchmark timing hook ran before train_ddp_loop setup")
        return value

    def timed_backward(
        accelerator: Accelerator, loss: torch.Tensor, **kwargs: object
    ) -> None:
        active = collector()
        started = time.monotonic()
        active.in_backward = True
        try:
            original_backward(accelerator, loss, **cast(dict[str, Any], kwargs))
        finally:
            active.in_backward = False
            active.add_backward(time.monotonic() - started)

    def timed_scheduler_step(scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        original_scheduler_step(scheduler)
        collector().mark_update_finished()

    def timed_struct_loss(
        stream: train_b0.StructStream,
        model: torch.nn.Module,
        *,
        epoch: int,
        step: int,
        steps: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        active = collector()
        started = time.monotonic()
        result = original_struct_loss(stream, model, epoch=epoch, step=step, steps=steps)
        elapsed = time.monotonic() - started
        active.add_struct_forward(elapsed, float(result[1].get("struct_pairs", 0.0)))
        return result

    def timed_coords(
        table: StructCoordinateTable,
        u_idx: NDArray[np.int64],
        v_idx: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        started = time.monotonic()
        result = original_coords(table, u_idx, v_idx)
        if collector_box[0] is not None:
            collector().add_coordinate_prepare(time.monotonic() - started)
        return result

    def benchmark_loop(
        *args: object, **kwargs: object
    ) -> train_b0.TrainResult:
        accelerator = cast(Accelerator, args[4])
        active = _Collector(
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            device=accelerator.device,
        )
        collector_box[0] = active
        factory = cast(train_b0.PackedLoaderFactory, args[1])

        def timed_factory(epoch: int) -> Iterable[train_b0.Batch]:
            return _TimedIterable(factory(epoch), active, world_size=accelerator.num_processes)

        patched_args = list(args)
        patched_args[1] = timed_factory
        try:
            cast(Callable[..., train_b0.TrainResult], original_loop)(*patched_args, **kwargs)
        except _BenchmarkComplete:
            accelerator.wait_for_everyone()
            _write_json_atomic(
                rank_output_dir / f"rank-{accelerator.process_index}.json",
                {
                    "rank": accelerator.process_index,
                    "world_size": accelerator.num_processes,
                    "warmup_steps": warmup_steps,
                    "timed_steps": timed_steps,
                    "steps": [asdict(record) for record in active.records],
                },
            )
            accelerator.wait_for_everyone()
            raise _WorkerComplete from None
        raise RuntimeError("production training ended before the benchmark step budget")

    Accelerator.backward = timed_backward
    train_b0._step_scheduler = timed_scheduler_step
    train_b0.StructStream.loss = timed_struct_loss  # type: ignore[assignment]
    StructCoordinateTable.coords_for_pairs = timed_coords  # type: ignore[assignment]
    train_b0.train_ddp_loop = benchmark_loop

    def restore() -> None:
        Accelerator.backward = original_backward
        train_b0._step_scheduler = original_scheduler_step
        train_b0.StructStream.loss = original_struct_loss  # type: ignore[method-assign]
        StructCoordinateTable.coords_for_pairs = original_coords  # type: ignore[method-assign]
        train_b0.train_ddp_loop = original_loop

    return restore


def _aggregate_rank_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("benchmark produced no rank reports")
    world_size = int(reports[0]["world_size"])
    if len(reports) != world_size:
        raise ValueError(f"expected {world_size} rank reports, found {len(reports)}")
    rows_by_rank = [cast(list[dict[str, Any]], report["steps"]) for report in reports]
    timed_steps = int(reports[0]["timed_steps"])
    if any(len(rows) != timed_steps for rows in rows_by_rank):
        raise ValueError("rank reports do not contain the requested timed-step count")

    steps: list[dict[str, Any]] = []
    for index in range(timed_steps):
        rows = [rank_rows[index] for rank_rows in rows_by_rank]
        global_counts = {int(row["global_task_pairs"]) for row in rows}
        if len(global_counts) != 1:
            raise ValueError(f"global task-pair count diverged at timed step {index + 1}")
        walls = [float(row["step_wall_seconds"]) for row in rows]
        wall = max(walls)
        global_task_pairs = global_counts.pop()
        global_struct_pairs = sum(float(row["struct_pairs_contribution"]) for row in rows)
        steps.append(
            {
                "step": index + 1,
                "slowest_rank": walls.index(wall),
                "slowest_rank_step_seconds": wall,
                "global_task_pairs": global_task_pairs,
                "global_struct_pairs": global_struct_pairs,
                "global_task_pairs_per_second": global_task_pairs / wall if wall else 0.0,
                "global_struct_pairs_per_second": global_struct_pairs / wall if wall else 0.0,
                "data_wait_seconds": max(float(row["data_wait_seconds"]) for row in rows),
                "coordinate_prepare_seconds": max(
                    float(row["coordinate_prepare_seconds"]) for row in rows
                ),
                "coordinate_prepare_forward_seconds": max(
                    float(row["coordinate_prepare_forward_seconds"]) for row in rows
                ),
                "coordinate_prepare_backward_seconds": max(
                    float(row["coordinate_prepare_backward_seconds"]) for row in rows
                ),
                "coordinate_prepare_forward_calls": sum(
                    int(row["coordinate_prepare_forward_calls"]) for row in rows
                ),
                "coordinate_prepare_backward_calls": sum(
                    int(row["coordinate_prepare_backward_calls"]) for row in rows
                ),
                "struct_forward_seconds": max(
                    float(row["struct_forward_seconds"]) for row in rows
                ),
                "shared_backward_seconds": max(
                    float(row["shared_backward_seconds"]) for row in rows
                ),
                "memory_allocated_gib": max(
                    float(row["memory_allocated_gib"]) for row in rows
                ),
                "memory_reserved_gib": max(float(row["memory_reserved_gib"]) for row in rows),
                "memory_current_allocated_gib": max(
                    float(row["memory_current_allocated_gib"]) for row in rows
                ),
                "memory_current_reserved_gib": max(
                    float(row["memory_current_reserved_gib"]) for row in rows
                ),
            }
        )
    total_wall = sum(float(row["slowest_rank_step_seconds"]) for row in steps)
    total_task_pairs = sum(int(row["global_task_pairs"]) for row in steps)
    total_struct_pairs = sum(float(row["global_struct_pairs"]) for row in steps)
    sorted_walls = sorted(float(row["slowest_rank_step_seconds"]) for row in steps)
    midpoint = len(sorted_walls) // 2
    median_wall = (
        sorted_walls[midpoint]
        if len(sorted_walls) % 2
        else (sorted_walls[midpoint - 1] + sorted_walls[midpoint]) / 2.0
    )
    return {
        "world_size": world_size,
        "steps": steps,
        "timed_window": {
            "started_at_unix_seconds": min(
                float(row["started_at_unix_seconds"])
                for rank_rows in rows_by_rank
                for row in rank_rows
            ),
            "finished_at_unix_seconds": max(
                float(row["finished_at_unix_seconds"])
                for rank_rows in rows_by_rank
                for row in rank_rows
            ),
        },
        "summary": {
            "timed_steps": timed_steps,
            "total_slowest_rank_seconds": total_wall,
            "median_slowest_rank_step_seconds": median_wall,
            "global_task_pairs_per_second": total_task_pairs / total_wall if total_wall else 0.0,
            "global_struct_pairs_per_second": (
                total_struct_pairs / total_wall if total_wall else 0.0
            ),
            "max_memory_allocated_gib": max(float(row["memory_allocated_gib"]) for row in steps),
            "max_memory_reserved_gib": max(float(row["memory_reserved_gib"]) for row in steps),
        },
    }


_GPU_FIELDS = (
    "timestamp,index,name,temperature.gpu,clocks.sm,memory.used,pstate,power.draw,"
    "clocks_throttle_reasons.sw_thermal_slowdown"
)


def _gpu_snapshot() -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    names = _GPU_FIELDS.split(",")
    devices = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        devices.append(dict(zip(names, values, strict=False)))
    return {"available": True, "devices": devices}


def _start_gpu_sampler(output_path: Path) -> tuple[subprocess.Popen[str], TextIO] | None:
    handle = output_path.open("w", encoding="utf-8")
    handle.write(_GPU_FIELDS + "\n")
    handle.flush()
    try:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_FIELDS}",
                "--format=csv,noheader,nounits",
                "--loop-ms=1000",
            ],
            stdout=handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        handle.close()
        return None
    return process, handle


def _finish_gpu_sampler(
    sampler: tuple[subprocess.Popen[str], TextIO] | None, *, output_path: Path
) -> dict[str, object]:
    if sampler is None:
        return {"available": False, "error": "nvidia-smi sampler could not start"}
    process, handle = sampler
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    finally:
        handle.close()
    samples = max(0, len(output_path.read_text(encoding="utf-8").splitlines()) - 1)
    return {
        "available": samples > 0,
        "interval_seconds": 1,
        "sample_rows": samples,
        "sidecar": str(output_path.resolve()),
        "returncode": process.returncode,
    }


def _git_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    return {"available": True, "head": revision, "dirty": dirty}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.experiments.struct_pipeline_benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=40)
    parser.add_argument("--variant", choices=("legacy", "coordinate-only", "full"), required=True)
    parser.add_argument("--round-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank-output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup_steps < 0 or args.timed_steps < 1:
        parser.error("--warmup-steps must be non-negative and --timed-steps must be positive")
    if args.worker and (args.artifact_dir is None or args.rank_output_dir is None):
        parser.error("internal --worker requires --artifact-dir and --rank-output-dir")
    return args


def _run_worker(args: argparse.Namespace) -> None:
    _install_worker_hooks(
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        rank_output_dir=args.rank_output_dir,
    )
    cfg = train_b0.load_config(args.config)
    if cfg.runtime is None:
        raise ValueError("benchmark config requires a runtime section")
    pack_dir = args.pack_dir or cfg.runtime.pack_dir
    profile_output = args.artifact_dir / "unused-profile.json"
    try:
        train_b0.main(
            [
                "--config",
                str(args.config),
                "--ddp-mode",
                "train",
                "--pack-dir",
                str(pack_dir),
                "--output-dir",
                str(args.artifact_dir),
                "--token-budget-per-rank",
                str(cfg.runtime.token_budget),
                "--profile-output",
                str(profile_output),
                "--resume-attempt",
                str(args.resume_attempt),
                "--run-kind",
                "diagnostic",
            ]
        )
    except _WorkerComplete:
        return


def _run_coordinator(args: argparse.Namespace) -> None:
    cfg = train_b0.load_config(args.config)
    if cfg.runtime is None:
        raise ValueError("benchmark config requires a runtime section")
    if cfg.struct is None:
        raise ValueError("benchmark config must enable the structural stream")
    state = args.resume_attempt / "training_state.pt"
    if not state.is_file():
        raise FileNotFoundError(f"resume attempt is missing {state.name}: {args.resume_attempt}")
    visible_world_size = detect_visible_gpu_count()
    world_size = visible_world_size if cfg.runtime.world_size == 0 else cfg.runtime.world_size
    if world_size != visible_world_size:
        raise ValueError(
            f"configured world size {world_size} does not match {visible_world_size} visible GPUs"
        )
    accelerate = Path(sys.executable).with_name("accelerate")
    if not accelerate.is_file():
        raise FileNotFoundError(f"accelerate executable not found next to Python: {accelerate}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = _gpu_snapshot()
    started = time.time()
    gpu_sidecar = args.output.with_suffix(args.output.suffix + ".gpu.csv")
    gpu_sampler = _start_gpu_sampler(gpu_sidecar)
    with tempfile.TemporaryDirectory(
        prefix="struct-pipeline-benchmark-", dir=args.output.parent
    ) as tmp:
        temporary = Path(tmp)
        artifact_dir = temporary / "attempt"
        rank_output_dir = temporary / "ranks"
        command = [
            str(accelerate),
            "launch",
            "--num_processes",
            str(world_size),
            "--mixed_precision",
            "bf16",
            "-m",
            "src.experiments.struct_pipeline_benchmark",
            "--worker",
            "--config",
            str(args.config.resolve()),
            "--resume-attempt",
            str(args.resume_attempt.resolve()),
            "--output",
            str(args.output.resolve()),
            "--warmup-steps",
            str(args.warmup_steps),
            "--timed-steps",
            str(args.timed_steps),
            "--variant",
            args.variant,
            "--round-index",
            str(args.round_index),
            "--artifact-dir",
            str(artifact_dir),
            "--rank-output-dir",
            str(rank_output_dir),
        ]
        if args.pack_dir is not None:
            command.extend(("--pack-dir", str(args.pack_dir.resolve())))
        try:
            subprocess.run(command, check=True)
        finally:
            gpu_samples = _finish_gpu_sampler(
                gpu_sampler, output_path=gpu_sidecar
            )
        reports = [
            json.loads((rank_output_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
            for rank in range(world_size)
        ]
        aggregate = _aggregate_rank_reports(reports)
    after = _gpu_snapshot()
    payload: dict[str, object] = {
        "schema": "struct_pipeline_benchmark_v1",
        "debug_only": True,
        "variant": args.variant,
        "round_index": args.round_index,
        "config": str(args.config.resolve()),
        "resume_attempt": str(args.resume_attempt.resolve()),
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "started_at_unix_seconds": started,
        "finished_at_unix_seconds": time.time(),
        "git": _git_revision(),
        "gpu_before": before,
        "gpu_after": after,
        "gpu_samples": gpu_samples,
        "timing_semantics": {
            "step_wall": "CUDA-synchronized slowest-rank wall time",
            "phase_seconds": "host-call elapsed time; phases may overlap and are not additive",
            "memory_allocated_reserved": "per-step CUDA peaks across ranks",
            "coordinate_calls": "sum over ranks, split by forward and checkpoint backward",
        },
        **aggregate,
    }
    _write_json_atomic(args.output, payload)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark coordinator, or one internal Accelerate worker."""
    args = _parse_args(argv)
    if args.worker:
        _run_worker(args)
    else:
        _run_coordinator(args)


if __name__ == "__main__":
    main()
