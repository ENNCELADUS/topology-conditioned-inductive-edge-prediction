"""Dependency and failure behavior of the one-shot 2026-09-08 H20 KD chain."""

import fcntl
import json
import subprocess
from pathlib import Path
from threading import Event, Thread

import pytest
from src.experiments import queue_split20260908 as queue


def _wait_then_set(done: Event) -> None:
    queue.wait_teacher()
    done.set()


def _completed(path: Path, prefix: str = "") -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("complete.json", "test_complete.json", "test_report.json"):
        (path / (prefix + name)).write_text("{}")
    (path / "best.pt").touch()


def test_teacher_lock_prevents_early_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher"
    _completed(teacher, "diagnostic_")
    monkeypatch.setattr(queue, "TEACHER", teacher)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "POLL_SECONDS", 0.05)
    entered, finished = Event(), Event()
    monkeypatch.setattr(queue, "status", lambda *args, **kwargs: entered.set())
    with (teacher / ".pipeline.lock").open("wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        thread = Thread(target=_wait_then_set, args=(finished,), daemon=True)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.2)
        fcntl.flock(handle, fcntl.LOCK_UN)
        thread.join(2)
    assert finished.is_set()


def test_wait_polls_until_the_teacher_finishes_and_stops_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    monkeypatch.setattr(queue, "TEACHER", teacher)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(queue, "status", lambda *args, **kwargs: None)
    finished = Event()
    thread = Thread(target=_wait_then_set, args=(finished,), daemon=True)
    thread.start()
    # No lock file yet, then a lock file with an unfinished teacher: keep waiting.
    assert not finished.wait(0.1)
    (teacher / ".pipeline.lock").touch()
    assert not finished.wait(0.1)
    _completed(teacher, "diagnostic_")
    thread.join(2)
    assert finished.is_set()

    (teacher / "failure.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Failed dependency"):
        queue.wait_teacher()


def test_publication_without_test_does_not_advance(tmp_path: Path) -> None:
    _completed(tmp_path)
    (tmp_path / "test_complete.json").unlink()
    with pytest.raises(RuntimeError, match="test_complete"):
        queue.require_complete(tmp_path)
    _completed(tmp_path)
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Failed dependency"):
        queue.require_complete(tmp_path)


@pytest.mark.parametrize("failed_test", [False, True])
def test_chain_runs_each_test_before_next_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_test: bool
) -> None:
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "wait_teacher", lambda: None)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "0\n1\n")
    events: list[object] = []
    monkeypatch.setattr(queue, "dump_bank", lambda **kw: events.append(kw["contexts"]))

    def run(command: list[str], log_name: str) -> None:
        assert "--skip-test" not in command
        # V3.1 has no EgoStitch test-access ledger; its scorer rejects this flag.
        assert "--rescore-reason" not in command
        arm = Path(command[3]).stem
        events.append(arm)
        _completed(tmp_path / arm)
        if failed_test:
            (tmp_path / arm / "test_complete.json").unlink()

    monkeypatch.setattr(queue, "run", run)
    if failed_test:
        with pytest.raises(RuntimeError, match="test_complete"):
            queue.main()
        assert events == [False, True, "kd_logit"]
    else:
        queue.main()
        assert events == [False, True, *queue.ARMS]
    state = json.loads((tmp_path / "queue_status.json").read_text())
    assert state["stage"] == ("failed" if failed_test else "complete")
