import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from src.score_fanout import score_sharded

pytestmark = pytest.mark.unit

_PYTHON_BIN = Path("/fake/python")
_TIMEOUT = 123.0


def _arg_value(command: Sequence[str], flag: str) -> str:
    command_list = list(command)
    return command_list[command_list.index(flag) + 1]


class _RecordingRunner:
    """Thread-safe fake `CommandRunner` that records every call it receives."""

    def __init__(self, *, fail_shard: int | None = None, fail_merge: bool = False) -> None:
        self._fail_shard = fail_shard
        self._fail_merge = fail_merge
        self._lock = threading.Lock()
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        command_list = list(command)
        with self._lock:
            self.calls.append((command_list, timeout))

        if "score" in command_list and "--shard" in command_list:
            shard = int(_arg_value(command_list, "--shard"))
            if self._fail_shard == shard:
                return subprocess.CompletedProcess(command_list, 1, "", "shard boom")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if "merge" in command_list:
            if self._fail_merge:
                return subprocess.CompletedProcess(command_list, 1, "", "merge boom")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    def merge_calls(self) -> list[tuple[list[str], float]]:
        return [call for call in self.calls if "merge" in call[0]]

    def shard_calls(self) -> list[tuple[list[str], float]]:
        return [call for call in self.calls if "--shard" in call[0]]


def test_single_gpu_issues_one_unsharded_pass_and_no_merge() -> None:
    runner = _RecordingRunner()

    output = score_sharded(
        ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
        gpu_count=1,
        python_bin=_PYTHON_BIN,
        timeout_seconds=_TIMEOUT,
        command_runner=runner,
    )

    assert output == Path("scores/out.npz")
    assert len(runner.calls) == 1
    command, timeout = runner.calls[0]
    assert timeout == _TIMEOUT
    assert command[0] == str(_PYTHON_BIN)
    assert command[1:4] == ["-m", "src.score_universe", "score"]
    assert "--device" in command and _arg_value(command, "--device") == "cuda"
    assert "--amp" in command and _arg_value(command, "--amp") == "bf16"
    assert "--checkpoint" in command and _arg_value(command, "--checkpoint") == "ckpt.pt"
    assert "--output" in command and _arg_value(command, "--output") == "scores/out.npz"
    assert "--shard" not in command
    assert "--num-shards" not in command
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(command)
    assert runner.merge_calls() == []


def test_single_gpu_failure_raises_runtime_error() -> None:
    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 1, "", "score failed hard")

    with pytest.raises(RuntimeError, match="score failed hard"):
        score_sharded(
            ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
            gpu_count=1,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )


def test_multi_gpu_launches_one_shard_per_gpu_and_merges_in_order() -> None:
    runner = _RecordingRunner()

    output = score_sharded(
        ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
        gpu_count=3,
        python_bin=_PYTHON_BIN,
        timeout_seconds=_TIMEOUT,
        command_runner=runner,
    )

    assert output == Path("scores/out.npz")
    shard_calls = runner.shard_calls()
    assert len(shard_calls) == 3

    seen_shards = set()
    for command, timeout in shard_calls:
        assert timeout == _TIMEOUT
        shard = int(_arg_value(command, "--shard"))
        seen_shards.add(shard)
        assert _arg_value(command, "--num-shards") == "3"
        assert _arg_value(command, "--output") == "scores/out.npz"
        assert _arg_value(command, "--device") == "cuda"
        assert _arg_value(command, "--amp") == "bf16"
        # CUDA_VISIBLE_DEVICES must be pinned to this shard's own child env,
        # via an `env VAR=value` prefix since CommandRunner exposes no env hook.
        assert command[0] == "env"
        assert command[1] == f"CUDA_VISIBLE_DEVICES={shard}"
        assert str(_PYTHON_BIN) in command

    assert seen_shards == {0, 1, 2}

    merge_calls = runner.merge_calls()
    assert len(merge_calls) == 1
    merge_command, merge_timeout = merge_calls[0]
    assert merge_timeout == _TIMEOUT
    assert merge_command[0] == str(_PYTHON_BIN)
    assert merge_command[1:4] == ["-m", "src.score_universe", "merge"]
    inputs_index = merge_command.index("--inputs")
    output_index = merge_command.index("--output")
    inputs = merge_command[inputs_index + 1 : output_index]
    assert inputs == [
        "scores/out.shard-0.npz",
        "scores/out.shard-1.npz",
        "scores/out.shard-2.npz",
    ]
    assert merge_command[output_index + 1] == "scores/out.npz"


def test_shards_pin_the_inherited_visible_device_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited mask names physical ids, so shards must reuse its own tokens.

    Synthesizing ``0..N-1`` from the count would send shards to devices outside
    the caller's allocation whenever the mask is not already ``0,1,...``.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    runner = _RecordingRunner()

    score_sharded(
        ["--output", "scores/out.npz"],
        gpu_count=2,
        python_bin=_PYTHON_BIN,
        timeout_seconds=_TIMEOUT,
        command_runner=runner,
    )

    pinned = {
        int(_arg_value(command, "--shard")): command[1] for command, _ in runner.shard_calls()
    }
    assert pinned == {0: "CUDA_VISIBLE_DEVICES=1", 1: "CUDA_VISIBLE_DEVICES=3"}


def test_rejects_a_mask_smaller_than_the_requested_shard_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed rather than oversubscribe a device outside the allocation."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    runner = _RecordingRunner()

    with pytest.raises(ValueError, match="names 1 devices but 3 shards"):
        score_sharded(
            ["--output", "scores/out.npz"],
            gpu_count=3,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )
    assert runner.calls == []


def test_failing_shard_raises_and_issues_no_merge() -> None:
    runner = _RecordingRunner(fail_shard=1)

    with pytest.raises(RuntimeError, match=r"shard 1/3 failed"):
        score_sharded(
            ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
            gpu_count=3,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )

    assert runner.merge_calls() == []


def test_failing_merge_raises() -> None:
    runner = _RecordingRunner(fail_merge=True)

    with pytest.raises(RuntimeError, match="merge boom"):
        score_sharded(
            ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
            gpu_count=2,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )

    assert len(runner.merge_calls()) == 1


def test_shard_flag_in_score_args_is_rejected() -> None:
    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command_runner must not be called")

    with pytest.raises(ValueError, match="--shard"):
        score_sharded(
            ["--output", "scores/out.npz", "--shard", "0"],
            gpu_count=2,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )


def test_num_shards_flag_in_score_args_is_rejected() -> None:
    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command_runner must not be called")

    with pytest.raises(ValueError, match="--num-shards"):
        score_sharded(
            ["--output", "scores/out.npz", "--num-shards", "2"],
            gpu_count=2,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )


def test_missing_output_is_rejected() -> None:
    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command_runner must not be called")

    with pytest.raises(ValueError, match="--output"):
        score_sharded(
            ["--checkpoint", "ckpt.pt"],
            gpu_count=1,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )


def test_non_npz_output_is_rejected() -> None:
    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command_runner must not be called")

    with pytest.raises(ValueError, match=r"\.npz"):
        score_sharded(
            ["--checkpoint", "ckpt.pt", "--output", "scores/out.txt"],
            gpu_count=1,
            python_bin=_PYTHON_BIN,
            timeout_seconds=_TIMEOUT,
            command_runner=runner,
        )


def test_timeout_is_forwarded_to_every_call() -> None:
    runner = _RecordingRunner()

    score_sharded(
        ["--checkpoint", "ckpt.pt", "--output", "scores/out.npz"],
        gpu_count=4,
        python_bin=_PYTHON_BIN,
        timeout_seconds=987.5,
        command_runner=runner,
    )

    assert len(runner.calls) == 5  # 4 shards + 1 merge
    assert all(timeout == 987.5 for _, timeout in runner.calls)
