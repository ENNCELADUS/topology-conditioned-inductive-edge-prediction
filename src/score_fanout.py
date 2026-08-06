"""Contiguous score-shard fan-out across visible GPUs, with a strict merge.

Single owner of score sharding. Both the thin ``hpc/run.sh score`` passthrough
and the pipeline's ``test`` stage call :func:`score_sharded`; the shell no
longer orchestrates shards.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.e2_pipeline import CommandRunner, detect_visible_gpu_count, run_command

logger = logging.getLogger(__name__)

__all__ = ["main", "score_sharded", "detect_visible_gpu_count"]

_OWNED_SHARD_FLAGS = ("--shard", "--num-shards")


def _resolve_output(score_args: Sequence[str]) -> Path:
    """Validate `score_args` and return the declared ``--output`` path.

    Mirrors the exact ``hpc/run.sh:parallel_score`` scan: it rejects the shard
    flags this function owns wherever they appear, and (like the bash) only
    recognizes the space-separated ``--output <path>`` form, taking the last
    occurrence if repeated.

    Args:
        score_args: Raw arguments after the ``score`` subcommand.

    Returns:
        The declared output path.

    Raises:
        ValueError: If a shard flag is present, ``--output`` is missing, or
            the output does not end in ``.npz``.
    """
    for flag in _OWNED_SHARD_FLAGS:
        if flag in score_args:
            raise ValueError(f"score_sharded owns sharding; do not pass {flag}")
    output: str | None = None
    for index, token in enumerate(score_args):
        if token == "--output":
            if index + 1 >= len(score_args):
                raise ValueError("--output requires a path")
            output = score_args[index + 1]
    if output is None:
        raise ValueError("score requires --output")
    if not output.endswith(".npz"):
        raise ValueError("score output must end in .npz")
    return Path(output)


def _visible_device_ids(gpu_count: int) -> list[str]:
    """Return the device ids each shard must pin, honoring an inherited mask.

    ``CUDA_VISIBLE_DEVICES`` names *physical* device ids, not ids relative to an
    existing mask. So under an inherited ``CUDA_VISIBLE_DEVICES=1,3`` a shard
    that set itself to ``0`` would target a device outside the caller's
    allocation. When a mask is present its own tokens are used verbatim.

    Args:
        gpu_count: Number of shards to pin.

    Returns:
        One device id per shard, in shard order.

    Raises:
        ValueError: If an inherited mask names fewer devices than `gpu_count`.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(shard) for shard in range(gpu_count)]
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices or devices == ["-1"]:
        return [str(shard) for shard in range(gpu_count)]
    if len(devices) < gpu_count:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES names {len(devices)} devices but {gpu_count} shards "
            "were requested"
        )
    return devices[:gpu_count]


def _shard_output_path(output: Path, shard: int) -> Path:
    """Return the per-shard output path ``<output stem>.shard-K<suffix>``.

    Matches ``src.score_universe._shard_output_path`` exactly: that CLI
    derives each shard's own output file from the unmodified ``--output``
    plus ``--shard``/``--num-shards`` it is given, so this function must
    compute the same shard file names to pass to ``merge``.
    """
    return output.with_name(f"{output.stem}.shard-{shard}{output.suffix}")


def score_sharded(
    score_args: Sequence[str],
    *,
    gpu_count: int,
    python_bin: Path,
    timeout_seconds: float | None = None,
    command_runner: CommandRunner = run_command,
) -> Path:
    """Score one pairs universe across `gpu_count` GPUs and merge the shards.

    Launches one contiguous shard per GPU with ``CUDA_VISIBLE_DEVICES`` pinned,
    waits for all of them, kills the survivors on the first failure, then merges
    strictly via ``score_universe merge``. With ``gpu_count == 1`` it runs a
    single unsharded pass and no merge.

    Args:
        score_args: Arguments after the ``score`` subcommand. Must carry
            ``--output <path>.npz`` and must not carry ``--shard`` or
            ``--num-shards``; this function owns sharding.
        gpu_count: Number of GPUs to fan out across.
        python_bin: Interpreter used to launch each shard.
        timeout_seconds: Optional hard wall-clock deadline for each shard.
            Defaults to ``None`` (wait indefinitely): a scoring pass's cost is
            set by its pairs universe, so a fixed budget is a guess that throws
            away hours of finished work whenever it is too small.
        command_runner: Injectable subprocess seam.

    Returns:
        The merged output path.

    Raises:
        ValueError: If `score_args` omits ``--output``, names a non-``.npz``
            output, or passes shard flags this function owns.
        RuntimeError: If any shard or the merge fails.
    """
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
    output = _resolve_output(score_args)

    base_command = [
        str(python_bin),
        "-m",
        "src.score_universe",
        "score",
        "--device",
        "cuda",
        "--amp",
        "bf16",
        *score_args,
    ]

    if gpu_count == 1:
        result = command_runner(base_command, timeout_seconds)
        if result.returncode != 0:
            raise RuntimeError(
                f"score failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
        return output

    device_ids = _visible_device_ids(gpu_count)
    shard_outputs = [_shard_output_path(output, shard) for shard in range(gpu_count)]
    shard_commands = [
        [
            "env",
            f"CUDA_VISIBLE_DEVICES={device_ids[shard]}",
            *base_command,
            "--shard",
            str(shard),
            "--num-shards",
            str(gpu_count),
        ]
        for shard in range(gpu_count)
    ]

    # command_runner is a blocking (command, timeout) -> CompletedProcess seam
    # with no process handle exposed, so a running shard cannot be forcibly
    # killed from here the way the bash `kill -TERM "${pids[@]}"` could; each
    # survivor still runs to its own timeout_seconds deadline before this
    # function's exception can propagate. Shards are inspected in launch
    # order (matching the bash `wait` loop, which also reports the first
    # index-order failure rather than the first-to-finish failure).
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        futures = [
            executor.submit(command_runner, command, timeout_seconds) for command in shard_commands
        ]
        for shard, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(f"score shard {shard}/{gpu_count} failed: {error}") from error
            if result.returncode != 0:
                raise RuntimeError(
                    f"score shard {shard}/{gpu_count} failed with exit code "
                    f"{result.returncode}: {result.stderr.strip()}"
                )

    merge_command = [
        str(python_bin),
        "-m",
        "src.score_universe",
        "merge",
        "--inputs",
        *[str(shard_output) for shard_output in shard_outputs],
        "--output",
        str(output),
    ]
    merge_result = command_runner(merge_command, timeout_seconds)
    if merge_result.returncode != 0:
        raise RuntimeError(
            f"score merge failed with exit code "
            f"{merge_result.returncode}: {merge_result.stderr.strip()}"
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m src.score_fanout``.

    Auto-detects the visible GPU count and forwards every argument it does not
    recognize verbatim to :func:`score_sharded` as ``score_args``. This is
    what ``hpc/run.sh score`` calls in place of its former bash fan-out.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(prog="python -m src.score_fanout")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="optional per-shard wall-clock deadline; omit to wait indefinitely",
    )
    args, score_args = parser.parse_known_args(argv)
    output = score_sharded(
        score_args,
        gpu_count=detect_visible_gpu_count(),
        python_bin=Path(sys.executable),
        timeout_seconds=args.timeout_seconds,
    )
    logger.info("wrote merged scores artifact: %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    raise SystemExit(main())
