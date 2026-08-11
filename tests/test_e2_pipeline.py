import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

import pytest
import src.data.packed_features as packed_features
import src.eval.test_protocol as test_protocol
import torch
from src.data.packed_features import build_packed_features
from src.e2_pipeline import (
    _PUBLISHED_FILENAMES,
    V_HOLD_VALIDATION_EVENTS_FILENAME,
    PipelineArgs,
    ProbeResult,
    _assert_no_cross_kind_completion,
    _assert_no_cross_kind_test_completion,
    _clear_stale_test_artifacts,
    _kill_registered_process_groups,
    _pipeline_rerun_command,
    _publish_staged,
    _register_nested_process_group,
    _rollback_publication,
    _test_stage_filenames,
    _unregister_nested_process_group,
    _validate_staged_artifacts,
    build_accelerate_command,
    detect_visible_gpu_count,
    main,
    parse_pipeline_args,
    run_command,
    run_pipeline,
    run_stage,
    write_failure,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _avoid_pack_process_pool_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline unit tests need pack semantics, not OS process startup."""
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)


@pytest.fixture(autouse=True)
def _fake_test_protocol_succeeds(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Default the test stage to a fast, deterministic success.

    ``run_test_protocol``'s real body is ``NotImplementedError`` (a concurrent
    agent owns it) and even once implemented it does real GPU scoring, so
    every pipeline test needs a fake here regardless. This autouse default
    keeps every pre-existing pack/train/publish test working unchanged; tests
    that exercise test-stage behavior directly override
    ``src.eval.test_protocol.run_test_protocol`` (or inspect ``calls`` below)
    themselves.
    """
    calls: dict[str, object] = {}

    def fake_run_test_protocol(
        *, output_dir: Path, report_filename: str, **kwargs: object
    ) -> test_protocol.TestProtocolResult:
        calls["output_dir"] = output_dir
        calls["report_filename"] = report_filename
        calls.update(kwargs)
        report_path = output_dir / report_filename
        report: dict[str, object] = {"status": "ok"}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return test_protocol.TestProtocolResult(report_path=report_path, report=report)

    monkeypatch.setattr(test_protocol, "run_test_protocol", fake_run_test_protocol)
    return calls


def test_publication_revokes_and_restores_the_old_completion_sentinel_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    staging_dir = tmp_path / "staging"
    output_dir.mkdir()
    staging_dir.mkdir()
    for filename in _PUBLISHED_FILENAMES:
        (output_dir / filename).write_text(f"old-{filename}")
        (staging_dir / filename).write_text(f"new-{filename}")
    (output_dir / "complete.json").write_text("old-complete")
    real_replace = __import__("os").replace
    moves: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        moves.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr("src.e2_pipeline.os.replace", recording_replace)
    backup_dir, published = _publish_staged(staging_dir, output_dir)

    assert moves[0][0] == output_dir / "complete.json"
    _rollback_publication(output_dir, backup_dir, published)
    assert (output_dir / "complete.json").read_text() == "old-complete"


def test_failed_completion_sentinel_backup_does_not_delete_the_old_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    staging_dir = tmp_path / "staging"
    output_dir.mkdir()
    staging_dir.mkdir()
    (output_dir / "complete.json").write_text("old-complete")
    for filename in _PUBLISHED_FILENAMES:
        (staging_dir / filename).write_text(f"new-{filename}")

    def failing_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("sentinel backup failed")

    monkeypatch.setattr("src.e2_pipeline.os.replace", failing_replace)
    with pytest.raises(OSError, match="sentinel backup failed"):
        _publish_staged(staging_dir, output_dir)

    assert (output_dir / "complete.json").read_text() == "old-complete"


def test_diagnostic_cannot_replace_a_formal_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="different run kind"):
        _assert_no_cross_kind_completion(output_dir, run_kind="diagnostic")
    _assert_no_cross_kind_completion(output_dir, run_kind="formal")
    (output_dir / "diagnostic_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="different run kind"):
        _assert_no_cross_kind_completion(output_dir, run_kind="formal")


def test_staged_metadata_role_must_match_diagnostic_execution(tmp_path: Path) -> None:
    _write_train_outputs(tmp_path)
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_kind": "formal",
                "checkpoint_role": "formal_plan_selected",
                "formal_artifacts_published": True,
            }
        )
    )
    with pytest.raises(ValueError, match="run_kind"):
        _validate_staged_artifacts(
            tmp_path,
            epochs=2,
            model_family="v3_1",
            expected_run_kind="diagnostic",
        )


# --------------------------------------------------------------------------- failure artifacts


def test_failure_json_is_atomic_and_structured(tmp_path: Path) -> None:
    path = write_failure(tmp_path, stage="pack", message="feature pack stage failed")
    payload = json.loads(path.read_text())
    assert payload["stage"] == "pack"
    assert payload["message"] == "feature pack stage failed"
    assert not path.with_suffix(".json.tmp").exists()


# --------------------------------------------------------------------------- command building


def test_accelerate_command_uses_resolved_world_size(tmp_path: Path) -> None:
    command = build_accelerate_command(
        accelerate_bin=Path("/venv/bin/accelerate"),
        config_path=Path("configs/b0_v31_breadth_first.yaml"),
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "outputs",
        token_budget=524288,
        profile_output=tmp_path / "probe.json",
        world_size=2,
    )
    assert command[:4] == ["/venv/bin/accelerate", "launch", "--num_processes", "2"]
    assert command[-2:] == ["--profile-output", str(tmp_path / "probe.json")]


def test_accelerate_command_plumbs_mode_pack_dir_output_dir_and_token_budget(
    tmp_path: Path,
) -> None:
    command = build_accelerate_command(
        accelerate_bin=Path("/venv/bin/accelerate"),
        config_path=Path("configs/b0_v31_breadth_first.yaml"),
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "outputs",
        token_budget=1048576,
        profile_output=tmp_path / "profile.json",
        world_size=3,
    )
    assert command[command.index("--mixed_precision") + 1] == "bf16"
    assert command[command.index("-m") + 1] == "src.train_b0"
    assert command[command.index("--config") + 1] == "configs/b0_v31_breadth_first.yaml"
    assert command[command.index("--ddp-mode") + 1] == "train"
    assert command[command.index("--pack-dir") + 1] == str(tmp_path / "pack")
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "outputs")
    assert command[command.index("--token-budget-per-rank") + 1] == "1048576"


def test_detect_visible_gpu_count_uses_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert detect_visible_gpu_count() == 2


@pytest.mark.integration
@pytest.mark.slow
def test_run_command_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    grandchild_pid_path = tmp_path / "grandchild.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    # The budget only has to be short relative to the child's 60 s sleep -- it is
    # never a measurement of how fast the kill is. It does have to be long enough
    # for the child to reach `write_text`, which costs an interpreter start plus a
    # `Popen`. At 0.5 s that raced: under `hpc/run.sh check`'s xdist workers on the
    # H20 container the child died before writing the pid file, and the test failed
    # with `FileNotFoundError` on `grandchild.pid` without ever reaching its
    # assertion. Widening the budget removes the race and weakens nothing -- the
    # grandchild still has ~55 s of sleep left when the group is killed, so a
    # surviving grandchild is caught exactly as before.
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, "-c", script], 5.0)

    grandchild_pid = int(grandchild_pid_path.read_text())
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(grandchild_pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z"), (
        f"grandchild process {grandchild_pid} remained live in state {state!r}"
    )


# --------------------------------------------------------------- nested process-group registry
# `run_command` always launches with `start_new_session=True` so its OWN
# timeout can kill just its own group -- but that means the group is a
# *different* session from a supervising `run_stage` call's forked child, so
# it would otherwise survive that stage's own deadline. These tests cover the
# registry `run_command`/`run_stage` share to close that gap.


def test_register_nested_process_group_writes_a_pgid_file(tmp_path: Path) -> None:
    entry = _register_nested_process_group(tmp_path, 4321)
    assert entry == tmp_path / "4321.pgid"
    assert entry is not None
    assert entry.read_text() == "4321"


def test_register_nested_process_group_is_a_noop_without_an_active_registry() -> None:
    assert _register_nested_process_group(None, 4321) is None


def test_unregister_nested_process_group_removes_the_file(tmp_path: Path) -> None:
    entry = tmp_path / "4321.pgid"
    entry.write_text("4321")
    _unregister_nested_process_group(entry)
    assert not entry.exists()


def test_unregister_nested_process_group_tolerates_none() -> None:
    _unregister_nested_process_group(None)  # must not raise


def test_kill_registered_process_groups_kills_every_recorded_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "111.pgid").write_text("111")
    (tmp_path / "222.pgid").write_text("222")
    killed: list[int] = []
    monkeypatch.setattr("src.e2_pipeline.os.killpg", lambda pgid, _sig: killed.append(pgid))

    _kill_registered_process_groups(tmp_path)

    assert sorted(killed) == [111, 222]


def test_kill_registered_process_groups_tolerates_an_already_dead_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "999.pgid").write_text("999")

    def raising_killpg(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError()

    monkeypatch.setattr("src.e2_pipeline.os.killpg", raising_killpg)

    _kill_registered_process_groups(tmp_path)  # must not raise


def test_kill_registered_process_groups_tolerates_an_empty_registry(tmp_path: Path) -> None:
    _kill_registered_process_groups(tmp_path)  # must not raise


def test_run_stage_cleans_up_its_nested_process_group_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every `run_stage` call, timeout or not, must leave no registry directory behind."""
    created: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(prefix: str | None = None) -> str:
        path = original_mkdtemp(prefix=prefix)
        created.append(Path(path))
        return path

    monkeypatch.setattr("src.e2_pipeline.tempfile.mkdtemp", recording_mkdtemp)

    run_stage(lambda: None, 5.0)

    assert created
    assert not created[-1].exists()


@pytest.mark.integration
@pytest.mark.slow
def test_run_stage_timeout_reaps_a_nested_run_command_process_group(tmp_path: Path) -> None:
    """A stage-level deadline must also kill a nested `run_command` subprocess.

    This mirrors the real test stage's shape: `score_sharded` launches scoring
    subprocesses from inside `run_stage`'s forked child, several layers of
    `command_runner` indirection away, each with its own
    `start_new_session=True` group. A stage deadline must reap that nested
    group too, even though the nested command's OWN (much longer) timeout
    never fires -- otherwise a stage timeout leaves GPU-occupying subprocesses
    running past the reported failure.
    """
    grandchild_pid_path = tmp_path / "nested_grandchild.pid"
    nested_script = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    def operation() -> None:
        # 30 s comfortably outlasts the stage's own 10 s deadline below, so
        # only the STAGE timeout -- not this nested command's own -- can be
        # what kills it in this test.
        run_command([sys.executable, "-c", nested_script], 30.0)

    # See afb73ab: this only has to outlast fork + interpreter start + two
    # nested levels of Popen, never a measurement of kill latency -- the
    # grandchild still has ~50 s of sleep left when the stage deadline fires.
    with pytest.raises(subprocess.TimeoutExpired):
        run_stage(operation, 10.0)

    grandchild_pid = int(grandchild_pid_path.read_text())
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(grandchild_pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z"), (
        f"nested grandchild process {grandchild_pid} remained live in state {state!r}"
    )


# --------------------------------------------------------------------------- CLI parsing


def test_parse_pipeline_args_requires_config() -> None:
    with pytest.raises(SystemExit):
        parse_pipeline_args([])


def test_parse_pipeline_args_parses_all_fields(tmp_path: Path) -> None:
    args = parse_pipeline_args(
        [
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--pack-dir",
            str(tmp_path / "pack"),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "2",
        ]
    )
    assert args == PipelineArgs(
        config=tmp_path / "cfg.yaml",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "out",
        seed=2,
    )


def test_parse_pipeline_args_defaults_pack_and_output_dir_to_none(tmp_path: Path) -> None:
    args = parse_pipeline_args(["--config", str(tmp_path / "cfg.yaml")])
    assert args.pack_dir is None
    assert args.output_dir is None
    assert args.worker_module == "src.train_b0"
    assert args.seed is None


def test_parse_pipeline_args_accepts_worker_module_override(tmp_path: Path) -> None:
    args = parse_pipeline_args(
        ["--config", str(tmp_path / "cfg.yaml"), "--worker-module", "src.train_egostitch"]
    )
    assert args.worker_module == "src.train_egostitch"


def test_pipeline_forwards_debug_max_steps_to_worker(tmp_path: Path) -> None:
    args = parse_pipeline_args(["--config", str(tmp_path / "cfg.yaml"), "--max-steps", "5"])
    assert args.max_steps == 5

    command = build_accelerate_command(
        accelerate_bin=tmp_path / "accelerate",
        config_path=args.config,
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "out",
        token_budget=256,
        profile_output=tmp_path / "profile.json",
        world_size=2,
        max_steps=args.max_steps,
    )

    assert command[command.index("--max-steps") + 1] == "5"


@pytest.mark.parametrize("run_kind", ["formal", "diagnostic"])
def test_pipeline_forwards_run_kind_to_worker(tmp_path: Path, run_kind: str) -> None:
    args = parse_pipeline_args(
        [
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--run-kind",
            run_kind,
        ]
    )
    assert args.run_kind == run_kind

    command = build_accelerate_command(
        accelerate_bin=tmp_path / "accelerate",
        config_path=args.config,
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "out",
        token_budget=256,
        profile_output=tmp_path / "profile.json",
        world_size=2,
        worker_module="src.train_egostitch",
        run_kind=args.run_kind,
    )

    assert command[command.index("--run-kind") + 1] == run_kind


@pytest.mark.parametrize(
    "run_kind", ["qualification", "overfit", "rehearsal", "calibration", "debug"]
)
def test_pipeline_refuses_run_kinds_outside_the_single_stage_contract(
    tmp_path: Path, run_kind: str
) -> None:
    """The retired kinds are rejected, and `debug` is never selectable.

    `debug` is a real `run_kind` the worker records, but it is *derived* from
    ``--max-steps``; offering it here would hand out the feature-digest-pin
    exemption to any caller that asked for it.
    """
    with pytest.raises(SystemExit):
        parse_pipeline_args(["--config", str(tmp_path / "cfg.yaml"), "--run-kind", run_kind])


def test_pipeline_omits_run_kind_when_unset(tmp_path: Path) -> None:
    args = parse_pipeline_args(["--config", str(tmp_path / "cfg.yaml")])
    assert args.run_kind is None

    command = build_accelerate_command(
        accelerate_bin=tmp_path / "accelerate",
        config_path=args.config,
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "out",
        token_budget=256,
        profile_output=tmp_path / "profile.json",
        world_size=2,
        worker_module="src.train_egostitch",
        run_kind=args.run_kind,
    )

    assert "--run-kind" not in command


def test_build_accelerate_command_worker_module(tmp_path: Path) -> None:
    from src.e2_pipeline import build_accelerate_command

    kwargs: dict[str, object] = {
        "accelerate_bin": tmp_path / "accelerate",
        "config_path": tmp_path / "cfg.yaml",
        "mode": "train",
        "pack_dir": tmp_path / "pack",
        "output_dir": tmp_path / "out",
        "token_budget": 256,
        "profile_output": tmp_path / "profile.json",
        "world_size": 2,
    }
    default = build_accelerate_command(**kwargs)  # type: ignore[arg-type]
    assert default[default.index("-m") + 1] == "src.train_b0"
    custom = build_accelerate_command(**kwargs, worker_module="src.train_egostitch")  # type: ignore[arg-type]
    assert custom[custom.index("-m") + 1] == "src.train_egostitch"
    # Everything else in the pinned argv is unchanged by the override.
    assert [x for x in custom if x != "src.train_egostitch"] == [
        x for x in default if x != "src.train_b0"
    ]


def test_build_accelerate_command_seed_override(tmp_path: Path) -> None:
    command = build_accelerate_command(
        accelerate_bin=tmp_path / "accelerate",
        config_path=tmp_path / "cfg.yaml",
        mode="train",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "out",
        token_budget=256,
        profile_output=tmp_path / "profile.json",
        world_size=2,
        worker_module="src.train_egostitch",
        seed=2,
    )
    assert command[command.index("--seed") + 1] == "2"


# ------------------------------------------------------------------- ProbeResult payload contract
# The orchestrator no longer sweeps token budgets, but the workers' own
# ``--ddp-mode probe`` path still emits this payload, so its strict reader has
# to keep rejecting malformed worker output.


def test_probe_result_round_trips_through_json() -> None:
    probe = ProbeResult(262144, True, 1200.0, 40.0, None)

    assert ProbeResult.from_dict(json.loads(json.dumps(probe.to_dict()))) == probe


@pytest.mark.parametrize(
    "payload",
    [
        {
            "token_budget": True,
            "valid": True,
            "global_pairs_per_second": 1.0,
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": 1,
            "global_pairs_per_second": 1.0,
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": True,
            "global_pairs_per_second": float("nan"),
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": True,
            "global_pairs_per_second": float("inf"),
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": True,
            "global_pairs_per_second": -1.0,
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": False,
            "global_pairs_per_second": 0.0,
            "peak_memory_gib": 1.0,
            "failure": None,
        },
        {
            "token_budget": 1,
            "valid": True,
            "global_pairs_per_second": 1.0,
            "peak_memory_gib": 1.0,
            "failure": "oom",
        },
    ],
)
def test_probe_result_rejects_malformed_values(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProbeResult.from_dict(payload)


# --------------------------------------------------------------------------- run_pipeline fixtures


def _write_feature_root(root: Path, node_shapes: dict[str, tuple[int, int]]) -> Path:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id, (length, dim) in node_shapes.items():
        relative_path = f"embeddings/{node_id}.pt"
        torch.save(torch.zeros(length, dim, dtype=torch.float32), root / relative_path)
        index[node_id] = relative_path
    (root / "metadata.json").write_text(json.dumps({"format": "torch_pt_per_node", "input_dim": 4}))
    (root / "index.json").write_text(json.dumps(index))
    return root


# `runtime.token_budget` is a scalar the orchestrator forwards verbatim; the
# candidate sweep and the projection budget it fed were deleted with the probe
# stage (design 2026-07-29 Sec 4).
_RUNTIME_TOKEN_BUDGET = 524288


def _pipeline_config_dict(
    *,
    data_root: Path,
    pack_dir: Path,
    output_dir: Path,
    epochs: int = 2,
    runtime_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    runtime: dict[str, object] = {
        "world_size": 4,
        "pack_dir": str(pack_dir),
        "pack_workers": 1,
        "loader_workers_per_rank": 0,
        "prefetch_factor": 2,
        "token_budget": _RUNTIME_TOKEN_BUDGET,
        "max_pairs_per_rank": 4096,
        "memory_limit_gib": 85.0,
        "total_budget_seconds": 40,
        "pack_budget_seconds": 20,
        "setup_probe_budget_seconds": 10,
        "train_eval_budget_seconds": 5,
        "artifact_budget_seconds": 3,
        "reserve_seconds": 2,
        "probe_warmup_steps": 1,
        "probe_timed_steps": 1,
    }
    if runtime_overrides:
        runtime.update(runtime_overrides)
        # `train_b0.load_config` asserts the five stage budgets sum to
        # `total_budget_seconds`, so an override of any of the five has to be
        # re-balanced or the config is rejected before the pipeline ever runs.
        # The test stage has no budget at all: its cost is set by the pairs
        # universe, not boundable up front.
        if "total_budget_seconds" not in runtime_overrides:
            runtime["total_budget_seconds"] = sum(
                cast(int, runtime[key])
                for key in (
                    "pack_budget_seconds",
                    "setup_probe_budget_seconds",
                    "train_eval_budget_seconds",
                    "artifact_budget_seconds",
                    "reserve_seconds",
                )
            )
    return {
        "model": {"family": "v3_1", "config": {}},
        "data": {
            "root": str(data_root),
            "strategy": "breadth_first",
            "negative_ratio": 1,
            "token_budget": 131072,
            "batch_pairs": 1024,
            "num_workers": 0,
            "f0_cache": str(output_dir / "f0_cache" / "f0_matrix.pt"),
            "expected_missing_features": [],
        },
        "optim": {
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "epochs": epochs,
            "warmup_steps": 1,
            "grad_clip": 1.0,
        },
        "eval": {"patience": 8, "eval_every": 1},
        "seed": 47,
        "output_dir": str(output_dir),
        "mixed_precision": "no",
        "runtime": runtime,
    }


def _write_pipeline_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config))


def _arg_value(command: Sequence[str], flag: str) -> str:
    command_list = list(command)
    return command_list[command_list.index(flag) + 1]


def _write_train_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {},
        "model_family": "v3_1",
        "model_config": {},
        "epoch": 1,
        "val_metrics": {},
        "seed": 47,
        "config": {},
    }
    torch.save(checkpoint, output_dir / "best.pt")
    torch.save({**checkpoint, "epoch": 2}, output_dir / "last.pt")
    (output_dir / "metrics.jsonl").write_text(
        '{"epoch": 1, "val_auroc": 0.7}\n{"epoch": 2, "val_auroc": 0.8}\n'
    )
    (output_dir / "run_metadata.json").write_text(json.dumps({"config_hash": "abc123"}))


def _valid_worker_profile() -> dict[str, object]:
    return {
        "epochs_completed": 2,
        "validations_completed": 2,
        "peak_memory_gib_per_rank": [40.0, 41.0, 40.5, 39.5],
        "steady_state_data_wait_fraction": 0.01,
        "training_coverage_exact": True,
        "validation_coverage_exact": True,
        "feature_cache_hit_rate": 1.0,
        "per_epoch": [
            {
                "epoch": epoch,
                "steps": 1,
                "global_pairs": 4,
                "local_pairs": 1,
                "local_tokens": 8,
                "wall_seconds": 1.0,
                "data_wait_seconds": 0.01,
                "compute_seconds": 0.9,
                "validation_seconds": 0.1,
            }
            for epoch in (1, 2)
        ],
    }


def _make_fake_runner(
    *,
    train_runtime_profile: dict[str, object] | None = None,
    fail_mode: str | None = None,
    timeout_mode: str | None = None,
    write_v_hold_validation_ledger: bool = False,
) -> Callable[[Sequence[str], float | None], subprocess.CompletedProcess[str]]:
    """Stand in for the single ``train`` process group the orchestrator launches."""
    resolved_train_profile: dict[str, object] = train_runtime_profile or _valid_worker_profile()

    def runner(command: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        command_list = list(command)
        mode = _arg_value(command_list, "--ddp-mode")
        profile_output = Path(_arg_value(command_list, "--profile-output"))
        out_dir = Path(_arg_value(command_list, "--output-dir"))

        if timeout_mode == mode:
            raise subprocess.TimeoutExpired(cmd=command_list, timeout=timeout or 0.0)
        if fail_mode == mode:
            return subprocess.CompletedProcess(command_list, returncode=1, stdout="", stderr="boom")

        if mode == "train":
            _write_train_outputs(out_dir)
            if write_v_hold_validation_ledger:
                ledger_path = out_dir / V_HOLD_VALIDATION_EVENTS_FILENAME
                ledger_path.write_text(
                    json.dumps(
                        {
                            "ordinal": 1,
                            "kind": "epoch_end",
                            "epoch": 1,
                            "optimizer_step": 1,
                            "run_kind": "formal",
                            "arm": "full",
                            "validation_role": "V_hold",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metadata_path = out_dir / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["v_hold_validation_evidence"] = {
                    "schema": "egostitch_e2e_v_hold_validation_events_v1",
                    "count": 1,
                    "path": V_HOLD_VALIDATION_EVENTS_FILENAME,
                    "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                }
                metadata_path.write_text(json.dumps(metadata))
            profile_output.write_text(json.dumps(resolved_train_profile))
        return subprocess.CompletedProcess(command_list, returncode=0, stdout="", stderr="")

    return runner


def test_orchestrator_launches_only_the_train_stage(tmp_path: Path) -> None:
    """`pack -> train -> publish`: no probe, epoch-probe or projection launch."""
    args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
    base_runner = _make_fake_runner()
    launched: list[tuple[str, str]] = []

    def runner(command: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        launched.append(
            (_arg_value(command, "--ddp-mode"), _arg_value(command, "--token-budget-per-rank"))
        )
        return base_runner(command, timeout)

    assert run_pipeline(args, command_runner=runner) == 0
    assert launched == [("train", str(_RUNTIME_TOKEN_BUDGET))]
    profile = json.loads((output_dir / "profile.json").read_text())
    assert profile["token_budget"] == _RUNTIME_TOKEN_BUDGET
    assert "probe_results" not in profile
    assert "selected_token_budget" not in profile
    assert "projected_total_seconds" not in profile
    assert "epoch_probe" not in profile


# ---------------------------------------------------------------------- run_pipeline: success paths


class TestRunPipelineSuccess:
    def test_generic_pack_stage_builds_and_reuses_all_worker_pack_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dual-pack worker gets one supervised cold/warm orchestration boundary."""
        from src import train_b0

        data_root = tmp_path / "data"
        source_root = _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        primary_pack = tmp_path / "f0-pack"
        raw_pack = data_root / "raw-token-pack"
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=primary_pack, output_dir=output_dir
            ),
        )

        def required_pack_paths(_cfg: object, pack_dir: Path) -> tuple[Path, Path]:
            return pack_dir, raw_pack

        def prepare_pack(
            cfg: object, pack_dir: Path, *, cold_cache: bool, temp_prefix: str
        ) -> dict[str, object]:
            primary_cold = not pack_dir.exists()
            primary = train_b0.prepare_pack(
                cast(train_b0.Config, cfg),
                pack_dir,
                cold_cache=primary_cold,
                temp_prefix=temp_prefix,
            )
            raw_cold = not raw_pack.exists()
            if raw_cold:
                raw_manifest = packed_features.build_packed_features(
                    source_root, raw_pack, workers=1, temp_prefix=temp_prefix
                )
            else:
                raw_manifest = packed_features.validate_packed_manifest(raw_pack, source_root)
            return {
                **primary,
                "packs": {
                    "primary": {"cold": primary_cold},
                    "raw_tokens": {
                        "cold": raw_cold,
                        "manifest": asdict(raw_manifest),
                        "identity_sha256": packed_features.sha256_file(raw_pack / "manifest.json"),
                    },
                },
            }

        class DualWorker:
            pass

        DualWorker.load_config = staticmethod(train_b0.load_config)  # type: ignore[attr-defined]
        DualWorker.prepare_pack = staticmethod(prepare_pack)  # type: ignore[attr-defined]
        DualWorker.required_pack_paths = staticmethod(  # type: ignore[attr-defined]
            required_pack_paths
        )

        original_import = importlib.import_module
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: DualWorker if name == "fake.dual_worker" else original_import(name),
        )
        args = PipelineArgs(
            config=config_path,
            pack_dir=None,
            output_dir=None,
            worker_module="fake.dual_worker",
        )

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
            )
            == 0
        )
        cold_profile = json.loads((output_dir / "profile.json").read_text())
        assert cold_profile["cold_cache"] is True
        assert cold_profile["pack_evidence"]["raw_tokens"]["cold"] is True
        assert (raw_pack / "manifest.json").is_file()

        assert run_pipeline(args, command_runner=_make_fake_runner()) == 0
        warm_profile = json.loads((output_dir / "profile.json").read_text())
        assert warm_profile["cold_cache"] is False
        assert warm_profile["pack_evidence"]["raw_tokens"]["cold"] is False

    def test_bounded_debug_pipeline_completes_in_debug_root_only(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        formal_output = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=tmp_path / "pack", output_dir=formal_output
            ),
        )
        profile = _valid_worker_profile()
        profile["epochs_completed"] = 1
        profile["validations_completed"] = 1
        profile["per_epoch"] = cast(list[object], profile["per_epoch"])[:1]

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            result = _make_fake_runner(train_runtime_profile=profile)(command, timeout)
            if _arg_value(command, "--ddp-mode") == "train":
                out = Path(_arg_value(command, "--output-dir"))
                best = torch.load(out / "best.pt", weights_only=False)
                torch.save(best, out / "last.pt")
                (out / "metrics.jsonl").write_text('{"epoch": 1, "val_auroc": 0.7}\n')
            return result

        assert (
            run_pipeline(PipelineArgs(config_path, None, None, max_steps=1), command_runner=runner)
            == 0
        )
        debug_output = tmp_path / "out_debug"
        assert (debug_output / "debug_complete.json").is_file()
        assert json.loads((debug_output / "profile.json").read_text())["run_kind"] == "debug"
        assert not formal_output.exists()

    def test_cold_cache_full_run_writes_merged_profile_and_manifest(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        pack_dir = tmp_path / "pack"
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(data_root=data_root, pack_dir=pack_dir, output_dir=output_dir),
        )
        args = PipelineArgs(config=config_path, pack_dir=None, output_dir=None)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 0
        assert not (output_dir / "failure.json").exists()
        profile = json.loads((output_dir / "profile.json").read_text())
        assert profile["cold_cache"] is True
        assert profile["token_budget"] == _RUNTIME_TOKEN_BUDGET
        assert profile["epochs_completed"] == 2
        # "test" lands only after publish, so profile.json is rewritten once
        # more post-publication to fold its duration in alongside the rest.
        assert set(profile["stage_seconds"]) == {"pack", "train", "artifacts", "test"}
        assert profile["total_seconds"] > 0
        assert profile["stage_seconds"]["artifacts"] >= 0
        assert profile["stage_seconds"]["test"] >= 0
        assert profile["total_seconds"] >= sum(profile["stage_seconds"].values())
        assert profile["pack_manifest"]["source_metadata_sha256"]
        assert profile["pack_manifest"]["source_index_sha256"]
        assert profile["pack_identity_sha256"]

        manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
        assert set(manifest) == {
            "best.pt",
            "last.pt",
            "metrics.jsonl",
            "run_metadata.json",
            "profile.json",
        }
        for filename, entry in manifest.items():
            path = output_dir / filename
            assert entry["byte_size"] == path.stat().st_size
            assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        complete = json.loads((output_dir / "complete.json").read_text())
        assert complete["status"] == "complete"
        # complete.json's total_seconds is frozen at publish time; the test
        # stage runs after that and recomputes profile.json's figure from the
        # same clock origin, so it always covers strictly more of the run.
        # (It must be *recomputed*, not accumulated onto the published value:
        # that value stops at the artifact cutoff, before publication, so
        # `published + test_duration` loses the publication interval and can
        # land below complete.json's total whenever publication is slower than
        # the test stage -- an intermittent failure under parallel load.)
        assert profile["total_seconds"] >= complete["total_seconds"]
        # Recomputed, not accumulated: the published value stops at the artifact
        # cutoff, so the post-test total must exceed the test stage's own
        # duration by the whole preceding run.
        assert profile["total_seconds"] > profile["stage_seconds"]["test"]
        assert (output_dir / "test_report.json").is_file()
        assert (output_dir / "test_complete.json").is_file()
        assert not (output_dir / "diagnostic_test_report.json").exists()
        assert not (output_dir / "diagnostic_test_complete.json").exists()

    def test_successful_rerun_replaces_prior_canonical_only_after_validation(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        before = TestRunPipelineFailures._seed_canonical(output_dir)

        assert run_pipeline(args, command_runner=_make_fake_runner()) == 0

        for filename, old_bytes in before.items():
            assert (output_dir / filename).read_bytes() != old_bytes
        manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
        for filename, entry in manifest.items():
            actual_sha256 = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            assert entry["sha256"] == actual_sha256

    def test_pack_validation_and_manifest_checksum_are_inside_supervised_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        import src.data.packed_features as packed_features

        original_validate = packed_features.validate_packed_manifest
        inside_stage = False
        validation_calls = 0

        def checked_validate(
            pack_root: Path,
            source_root: Path | None,
            *,
            verify_shard_sha256: bool = True,
        ) -> object:
            nonlocal validation_calls
            validation_calls += 1
            assert inside_stage, "full pack validation escaped the supervised pack stage"
            return original_validate(
                pack_root,
                source_root,
                verify_shard_sha256=verify_shard_sha256,
            )

        def supervised(operation: Callable[[], None], _timeout: float | None) -> None:
            nonlocal inside_stage
            inside_stage = True
            try:
                operation()
            finally:
                inside_stage = False

        monkeypatch.setattr(packed_features, "validate_packed_manifest", checked_validate)
        assert run_pipeline(args, command_runner=_make_fake_runner(), stage_runner=supervised) == 0
        assert validation_calls == 1  # the cold builder's publication validation only
        assert (output_dir / "profile.json").exists()

    def test_oom_during_training_is_a_train_stage_failure(self, tmp_path: Path) -> None:
        """With no candidate sweep left, an OOM is simply a failed train stage."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        def runner(
            command: Sequence[str], _timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, "", "CUDA out of memory")

        assert run_pipeline(args, command_runner=runner) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "train"
        assert "CUDA out of memory" in failure["stderr"]

    def test_successful_rerun_cannot_reuse_stale_worker_outputs(self, tmp_path: Path) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        output_dir.mkdir()
        for filename in (
            "best.pt",
            "last.pt",
            "metrics.jsonl",
            "run_metadata.json",
            "profile.json",
            "train_worker_profile.json",
            "artifact_manifest.json",
            "complete.json",
        ):
            (output_dir / filename).write_text("stale")

        def no_write_train(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "train":
                return subprocess.CompletedProcess(command, 0, "", "")
            return _make_fake_runner()(command, timeout)

        assert run_pipeline(args, command_runner=no_write_train) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "artifacts"
        assert (output_dir / "best.pt").read_text() == "stale"
        assert (output_dir / "complete.json").read_text() == "stale"

    def test_failed_train_rescues_worker_history_evidence(self, tmp_path: Path) -> None:
        """A worker-side failed_run_history.json survives staging-dir teardown."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        base_runner = _make_fake_runner()

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            command_list = list(command)
            if _arg_value(command_list, "--ddp-mode") == "train":
                staging = Path(_arg_value(command_list, "--output-dir"))
                (staging / "failed_run_history.json").write_text('{"history": []}')
                return subprocess.CompletedProcess(command_list, 1, "", "no eligible checkpoint")
            return base_runner(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 2
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "train"
        assert json.loads((output_dir / "failed_run_history.json").read_text()) == {"history": []}

    def test_failed_rerun_without_new_history_removes_stale_history(self, tmp_path: Path) -> None:
        """A pre-train failure must not leave a prior run's history as evidence."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(
            tmp_path, pack_budget_seconds=0
        )
        output_dir.mkdir()
        (output_dir / "failed_run_history.json").write_text('{"history": ["stale"]}')

        assert run_pipeline(args, command_runner=_make_fake_runner()) == 2
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "pack"
        assert not (output_dir / "failed_run_history.json").exists()

    def test_successful_rerun_removes_stale_history_evidence(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        pack_dir = tmp_path / "pack"
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "failed_run_history.json").write_text('{"history": []}')
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(data_root=data_root, pack_dir=pack_dir, output_dir=output_dir),
        )

        assert (
            run_pipeline(PipelineArgs(config_path, None, None), command_runner=_make_fake_runner())
            == 0
        )
        assert not (output_dir / "failed_run_history.json").exists()

    def test_successful_rerun_removes_stale_failure_marker(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        pack_dir = tmp_path / "pack"
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "failure.json").write_text('{"stage":"old"}')
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(data_root=data_root, pack_dir=pack_dir, output_dir=output_dir),
        )

        assert (
            run_pipeline(PipelineArgs(config_path, None, None), command_runner=_make_fake_runner())
            == 0
        )
        assert not (output_dir / "failure.json").exists()

    def test_warm_cache_validates_instead_of_rebuilding(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        source_root = _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        pack_dir = tmp_path / "pack"
        build_packed_features(source_root, pack_dir, workers=1)
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(data_root=data_root, pack_dir=pack_dir, output_dir=output_dir),
        )
        args = PipelineArgs(config=config_path, pack_dir=None, output_dir=None)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 0
        profile = json.loads((output_dir / "profile.json").read_text())
        assert profile["cold_cache"] is False

    def test_cli_pack_dir_and_output_dir_override_the_config(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        config_pack_dir = tmp_path / "config-pack"
        config_output_dir = tmp_path / "config-out"
        override_pack_dir = tmp_path / "cli-pack"
        override_output_dir = tmp_path / "cli-out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=config_pack_dir, output_dir=config_output_dir
            ),
        )
        args = PipelineArgs(
            config=config_path, pack_dir=override_pack_dir, output_dir=override_output_dir
        )

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 0
        assert override_pack_dir.exists()
        assert not config_pack_dir.exists()
        assert (override_output_dir / "profile.json").exists()
        assert not config_output_dir.exists()


# ---------------------------------------------------------------------- run_pipeline: failure paths


class TestRunPipelineFailures:
    def _base_args_and_config(
        self, tmp_path: Path, **runtime_overrides: object
    ) -> tuple[PipelineArgs, Path]:
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        pack_dir = tmp_path / "pack"
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root,
                pack_dir=pack_dir,
                output_dir=output_dir,
                runtime_overrides=runtime_overrides or None,
            ),
        )
        return PipelineArgs(config=config_path, pack_dir=None, output_dir=None), output_dir

    @staticmethod
    def _seed_canonical(output_dir: Path) -> dict[str, bytes]:
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = (
            "best.pt",
            "last.pt",
            "metrics.jsonl",
            "run_metadata.json",
            "profile.json",
            "artifact_manifest.json",
            "complete.json",
        )
        for filename in filenames:
            (output_dir / filename).write_bytes(f"old-{filename}".encode())
        return {filename: (output_dir / filename).read_bytes() for filename in filenames}

    @staticmethod
    def _assert_canonical_unchanged(output_dir: Path, before: dict[str, bytes]) -> None:
        assert {name: (output_dir / name).read_bytes() for name in before} == before

    def test_prior_canonical_survives_subprocess_failure(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        before = self._seed_canonical(output_dir)

        assert run_pipeline(args, command_runner=_make_fake_runner(fail_mode="train")) == 2

        self._assert_canonical_unchanged(output_dir, before)
        assert (output_dir / "failure.json").exists()

    def test_prior_canonical_survives_pack_failure(self, tmp_path: Path) -> None:
        data_root = tmp_path / "missing"
        output_dir = tmp_path / "out"
        before = self._seed_canonical(output_dir)
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=tmp_path / "pack", output_dir=output_dir
            ),
        )

        assert run_pipeline(PipelineArgs(config_path, None, None)) == 2

        self._assert_canonical_unchanged(output_dir, before)

    def test_pack_timeout_removes_the_owned_pack_temporary_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        import src.data.packed_features as packed_features

        created: list[Path] = []

        def leaking_build(
            source_root: Path,
            pack_root: Path,
            workers: int,
            *,
            temp_prefix: str | None = None,
        ) -> object:
            assert temp_prefix is not None
            leaked = pack_root.parent / f"{temp_prefix}leaked"
            leaked.mkdir(parents=True)
            created.append(leaked)
            raise subprocess.TimeoutExpired(cmd="pack", timeout=1)

        monkeypatch.setattr(packed_features, "build_packed_features", leaking_build)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 2
        )
        assert created
        assert not created[0].exists()
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "pack"

    @pytest.mark.parametrize(
        "profile_update",
        [
            {"training_coverage_exact": False},
            {"validation_coverage_exact": 1},
            {"feature_cache_hit_rate": 0.999},
            {"peak_memory_gib_per_rank": [40.0, 41.0, 86.0, 39.5]},
            {"steady_state_data_wait_fraction": 0.051},
            {"per_epoch": []},
        ],
    )
    def test_strict_worker_profile_rejects_invalid_runtime_evidence(
        self, tmp_path: Path, profile_update: dict[str, object]
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        profile = {**_valid_worker_profile(), **profile_update}

        assert (
            run_pipeline(args, command_runner=_make_fake_runner(train_runtime_profile=profile)) == 2
        )

        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
        failed_profile = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert failed_profile["rejected_worker_profile"] == profile
        assert not (output_dir / "artifact_manifest.json").exists()
        assert not (output_dir / "complete.json").exists()

    def test_diagnostic_runtime_telemetry_and_engineering_deadlines_never_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Diagnostic truth/integrity checks remain strict; utilization is telemetry."""
        from src import train_b0

        @dataclass(frozen=True)
        class _DiagnosticConfig(train_b0.Config):
            run_kind: str | None = None

        def fake_load_config(path: Path) -> _DiagnosticConfig:
            base = train_b0.load_config(path)
            return _DiagnosticConfig(**{f.name: getattr(base, f.name) for f in fields(base)})

        class DiagnosticWorker:
            pass

        DiagnosticWorker.load_config = staticmethod(fake_load_config)  # type: ignore[attr-defined]
        DiagnosticWorker.prepare_pack = staticmethod(train_b0.prepare_pack)  # type: ignore[attr-defined]

        base_args, output_dir = self._base_args_and_config(tmp_path)
        args = PipelineArgs(
            config=base_args.config,
            pack_dir=base_args.pack_dir,
            output_dir=base_args.output_dir,
            worker_module="fake.diagnostic_worker",
            run_kind="diagnostic",
        )
        original_import = importlib.import_module
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: (
                DiagnosticWorker if name == "fake.diagnostic_worker" else original_import(name)
            ),
        )
        profile = {
            **_valid_worker_profile(),
            "peak_memory_gib_per_rank": [90.0, 91.0, 92.0, 93.0],
            "steady_state_data_wait_fraction": 0.25,
            "feature_cache_hit_rate": 0.9,
        }
        command_timeouts: list[float | None] = []
        base_runner = _make_fake_runner(train_runtime_profile=profile)

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            command_timeouts.append(timeout)
            return base_runner(command, timeout)

        stage_timeouts: list[float | None] = []

        def stage_runner(operation: Callable[[], None], timeout: float | None) -> None:
            stage_timeouts.append(timeout)
            operation()

        assert run_pipeline(args, command_runner=runner, stage_runner=stage_runner) == 0
        assert command_timeouts and all(timeout is None for timeout in command_timeouts)
        assert stage_timeouts and all(timeout is None for timeout in stage_timeouts)
        assert (output_dir / "diagnostic_complete.json").is_file()
        published = json.loads((output_dir / "profile.json").read_text())
        assert published["steady_state_data_wait_fraction"] == 0.25
        assert published["feature_cache_hit_rate"] == 0.9

    @pytest.mark.parametrize("corrupt", ["checkpoint", "metrics", "metadata"])
    def test_corrupt_staged_artifact_is_rejected_before_hashing(
        self, tmp_path: Path, corrupt: str
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            completed = base(command, timeout)
            if _arg_value(command, "--ddp-mode") == "train":
                staging = Path(_arg_value(command, "--output-dir"))
                if corrupt == "checkpoint":
                    (staging / "best.pt").write_bytes(b"not a checkpoint")
                elif corrupt == "metrics":
                    (staging / "metrics.jsonl").write_text("not-json\n")
                else:
                    (staging / "run_metadata.json").write_text("[]")
            return completed

        assert run_pipeline(args, command_runner=runner) == 2
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
        assert not (output_dir / "artifact_manifest.json").exists()

    def test_pack_build_error_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        # data_root has no features/ directory at all -> build_packed_features raises.
        data_root = tmp_path / "data-missing-features"
        pack_dir = tmp_path / "pack"
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(data_root=data_root, pack_dir=pack_dir, output_dir=output_dir),
        )
        args = PipelineArgs(config=config_path, pack_dir=None, output_dir=None)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "pack"

    def test_pack_stage_exceeding_its_budget_writes_failure_and_returns_2(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path, pack_budget_seconds=0)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "pack"

    def test_pack_timeout_is_supervised_and_structured(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path, pack_budget_seconds=1)
        observed: dict[str, float | None] = {}

        def timeout_stage(_operation: Callable[[], None], timeout: float | None) -> None:
            observed["timeout"] = timeout
            raise subprocess.TimeoutExpired(cmd="pack", timeout=timeout or 0.0)

        exit_code = run_pipeline(
            args, command_runner=_make_fake_runner(), stage_runner=timeout_stage
        )

        assert exit_code == 2
        assert observed["timeout"] == 1
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "pack"

    @pytest.mark.parametrize("payload", [None, "not-json", "{}"])
    def test_missing_or_malformed_train_profile_is_a_structured_artifact_failure(
        self, tmp_path: Path, payload: str | None
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            completed = base(command, timeout)
            profile_output = Path(_arg_value(command, "--profile-output"))
            profile_output.unlink(missing_ok=True)
            if payload is not None:
                profile_output.write_text(payload)
            return completed

        assert run_pipeline(args, command_runner=runner) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "artifacts"
        assert "profile" in failure["message"]
        evidence = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert evidence["token_budget"] == _RUNTIME_TOKEN_BUDGET
        assert evidence["stage_seconds"]["pack"] >= 0

    def test_train_subprocess_failure_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(fail_mode="train"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "train"

    def test_train_failure_restores_completed_pipeline_evidence(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base_runner = _make_fake_runner()

        def corrupting_runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "train":
                Path(_arg_value(command, "--profile-output")).write_text("partial")
                return subprocess.CompletedProcess(command, 1, "", "boom")
            return base_runner(command, timeout)

        assert run_pipeline(args, command_runner=corrupting_runner) == 2
        profile = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert profile["token_budget"] == _RUNTIME_TOKEN_BUDGET
        assert profile["stage_seconds"]["pack"] >= 0
        assert profile["cold_cache"] is True

    def test_train_subprocess_timeout_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(timeout_mode="train"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "train"

    @pytest.mark.parametrize(
        "worker_profile",
        [
            {"epochs_completed": 2},
            {"epochs_completed": 1, "validations_completed": 2},
            {"epochs_completed": 2, "validations_completed": True},
        ],
    )
    def test_train_worker_profile_requires_exact_epoch_and_validation_completion(
        self, tmp_path: Path, worker_profile: dict[str, object]
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(train_runtime_profile=worker_profile),
            )
            == 2
        )
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"

    def test_artifact_timeout_preserves_worker_and_pipeline_evidence(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        calls = 0

        # Two supervised stages remain: pack, then artifacts.
        def stage_runner(operation: Callable[[], None], timeout: float | None) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise subprocess.TimeoutExpired(cmd="artifacts", timeout=timeout or 0.0)
            operation()

        worker_profile = {
            **_valid_worker_profile(),
            "counterfactual_stop_epoch": 1,
            "per_rank": [
                {
                    "rank": rank,
                    "pairs": 2,
                    "batches": 1,
                    "steps": 1,
                    "tokens": 8,
                    "train_wall_seconds": 1.0,
                    "data_wait_seconds": 0.01,
                    "pairs_per_second": 42.0,
                    "tokens_per_second": 8.0,
                }
                for rank in range(4)
            ],
        }
        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(train_runtime_profile=worker_profile),
                stage_runner=stage_runner,
            )
            == 2
        )
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "artifacts"
        profile = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert profile["counterfactual_stop_epoch"] == 1
        assert profile["per_rank"][0]["pairs_per_second"] == 42.0
        assert profile["token_budget"] == _RUNTIME_TOKEN_BUDGET

    def test_cold_b0_total_budget_includes_publication_and_rolls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        before = self._seed_canonical(output_dir)
        ticks = iter((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 45.0))
        monkeypatch.setattr(
            "src.e2_pipeline.time",
            types.SimpleNamespace(monotonic=lambda: next(ticks)),
        )

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 2
        )

        self._assert_canonical_unchanged(output_dir, before)
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "total_budget"
        assert failure["elapsed_seconds"] == 45.0
        assert failure["limit_seconds"] == 40


# ---------------------------------------------------------------------- run_pipeline: test stage


def test_test_stage_filenames_follow_the_formal_diagnostic_split() -> None:
    assert _test_stage_filenames(None) == ("test_report.json", "test_complete.json")
    assert _test_stage_filenames("formal") == ("test_report.json", "test_complete.json")
    assert _test_stage_filenames("diagnostic") == (
        "diagnostic_test_report.json",
        "diagnostic_test_complete.json",
    )


def test_cross_kind_test_completion_guard_mirrors_the_publish_guard(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "test_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="different run kind"):
        _assert_no_cross_kind_test_completion(output_dir, run_kind="diagnostic")
    _assert_no_cross_kind_test_completion(output_dir, run_kind="formal")
    (output_dir / "diagnostic_test_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="different run kind"):
        _assert_no_cross_kind_test_completion(output_dir, run_kind="formal")


def test_clear_stale_test_artifacts_removes_both_same_kind_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "test_complete.json").write_text("{}")
    (output_dir / "test_report.json").write_text("{}")
    # A different run kind's names must never be touched by this call.
    (output_dir / "diagnostic_test_complete.json").write_text("{}")

    _clear_stale_test_artifacts(
        output_dir, report_filename="test_report.json", test_complete_name="test_complete.json"
    )

    assert not (output_dir / "test_complete.json").exists()
    assert not (output_dir / "test_report.json").exists()
    assert (output_dir / "diagnostic_test_complete.json").exists()


def test_clear_stale_test_artifacts_removes_the_sentinel_before_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-clear must never leave a sentinel without its checked report."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "test_complete.json").write_text("{}")
    (output_dir / "test_report.json").write_text("{}")
    order: list[str] = []
    original_unlink = Path.unlink

    def recording_unlink(self: Path, missing_ok: bool = False) -> None:
        order.append(self.name)
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", recording_unlink)

    _clear_stale_test_artifacts(
        output_dir, report_filename="test_report.json", test_complete_name="test_complete.json"
    )

    assert order == ["test_complete.json", "test_report.json"]


def test_clear_stale_test_artifacts_tolerates_missing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _clear_stale_test_artifacts(
        output_dir, report_filename="test_report.json", test_complete_name="test_complete.json"
    )  # must not raise


def test_pipeline_rerun_command_names_config_and_appends_rescore_reason(tmp_path: Path) -> None:
    args = PipelineArgs(
        config=tmp_path / "cfg.yaml",
        pack_dir=None,
        output_dir=None,
        seed=3,
        run_kind="diagnostic",
        worker_module="src.train_egostitch",
    )
    command = _pipeline_rerun_command(args)
    assert command.startswith(f"python -m src.e2_pipeline --config {tmp_path / 'cfg.yaml'}")
    assert "--seed 3" in command
    assert "--run-kind diagnostic" in command
    assert "--worker-module src.train_egostitch" in command
    assert command.endswith("--rescore-reason <reason for the repeat held-out scoring epoch>")


def test_parse_pipeline_args_rescore_reason_defaults_to_none_and_is_never_synthesized(
    tmp_path: Path,
) -> None:
    args = parse_pipeline_args(["--config", str(tmp_path / "cfg.yaml")])
    assert args.rescore_reason is None

    with_reason = parse_pipeline_args(
        ["--config", str(tmp_path / "cfg.yaml"), "--rescore-reason", "replace corrupt artifact"]
    )
    assert with_reason.rescore_reason == "replace corrupt artifact"


class TestRunPipelineTestStage:
    def test_test_stage_runs_after_publish_and_sees_the_completed_publication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The test stage must observe a real, already-committed `complete.json`."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        seen: dict[str, object] = {}

        def fake_run_test_protocol(
            *, output_dir: Path, report_filename: str, **_kwargs: object
        ) -> test_protocol.TestProtocolResult:
            seen["complete_json_is_file"] = (output_dir / "complete.json").is_file()
            seen["best_pt_is_file"] = (output_dir / "best.pt").is_file()
            report_path = output_dir / report_filename
            report: dict[str, object] = {"status": "ok"}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            return test_protocol.TestProtocolResult(report_path=report_path, report=report)

        monkeypatch.setattr(test_protocol, "run_test_protocol", fake_run_test_protocol)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(),
            stage_runner=lambda operation, _timeout: operation(),
        )

        assert exit_code == 0
        assert seen["complete_json_is_file"] is True
        assert seen["best_pt_is_file"] is True
        assert (output_dir / "test_report.json").is_file()
        assert (output_dir / "test_complete.json").is_file()

    def test_test_stage_forwards_the_resolved_config_arm_seed_and_rescore_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        args = PipelineArgs(**{**vars(args), "rescore_reason": "replace corrupt score artifact"})
        captured: dict[str, object] = {}

        def fake_run_test_protocol(**kwargs: object) -> test_protocol.TestProtocolResult:
            captured.update(kwargs)
            report_path = cast(Path, kwargs["output_dir"]) / cast(str, kwargs["report_filename"])
            report: dict[str, object] = {"status": "ok"}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            return test_protocol.TestProtocolResult(report_path=report_path, report=report)

        monkeypatch.setattr(test_protocol, "run_test_protocol", fake_run_test_protocol)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )
        assert captured["checkpoint"] == output_dir / "best.pt"
        assert captured["seed"] == 47
        # train_b0 run_metadata.json carries no "arm"; the model family stands
        # in as the test-report's arm identity for non-E2E workers.
        assert captured["arm"] == "v3_1"
        assert captured["rescore_reason"] == "replace corrupt score artifact"
        assert captured["report_filename"] == "test_report.json"

    def test_absent_rescore_reason_is_never_synthesized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        assert args.rescore_reason is None
        captured: dict[str, object] = {}

        def fake_run_test_protocol(**kwargs: object) -> test_protocol.TestProtocolResult:
            captured.update(kwargs)
            report_path = cast(Path, kwargs["output_dir"]) / cast(str, kwargs["report_filename"])
            report: dict[str, object] = {"status": "ok"}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            return test_protocol.TestProtocolResult(report_path=report_path, report=report)

        monkeypatch.setattr(test_protocol, "run_test_protocol", fake_run_test_protocol)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )
        assert captured["rescore_reason"] is None

    def test_test_stage_failure_preserves_publication_and_returns_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scoring failure must not roll back a committed publication."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        def failing_run_test_protocol(**_kwargs: object) -> test_protocol.TestProtocolResult:
            raise ValueError("scoring blew up")

        monkeypatch.setattr(test_protocol, "run_test_protocol", failing_run_test_protocol)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(),
            stage_runner=lambda operation, _timeout: operation(),
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "test"
        assert "scoring blew up" in failure["message"]
        # The publication this failure must not touch:
        assert (output_dir / "complete.json").is_file()
        assert (output_dir / "best.pt").is_file()
        assert (output_dir / "last.pt").is_file()
        assert (output_dir / "profile.json").is_file()
        assert (output_dir / "artifact_manifest.json").is_file()
        assert not (output_dir / "test_complete.json").exists()

    def test_repeat_scoring_ledger_failure_names_the_rescore_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        def failing_run_test_protocol(**_kwargs: object) -> test_protocol.TestProtocolResult:
            raise ValueError(
                "held-out scoring already has an epoch for arm='v3_1', seed=47; "
                "repeat scoring requires --rescore-reason"
            )

        monkeypatch.setattr(test_protocol, "run_test_protocol", failing_run_test_protocol)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(),
            stage_runner=lambda operation, _timeout: operation(),
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "test"
        assert "--rescore-reason" in failure["message"]
        assert f"python -m src.e2_pipeline --config {args.config}" in failure["message"]
        assert (output_dir / "complete.json").is_file()

    def test_test_stage_runs_without_a_deadline(self, tmp_path: Path) -> None:
        """The test stage is supervised but unbounded.

        Scoring cost is set by the pairs universe -- the candidate universe is
        ~2.0M pairs, ~32x the test universe -- so no fixed budget can be chosen
        up front. A wrong one silently discards hours of finished scoring, which
        is exactly how the first real run died. Both the supervising stage and
        the per-shard fan-out must therefore receive `None`.
        """
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        timeouts: list[float | None] = []

        def stage_runner(operation: Callable[[], None], timeout: float | None) -> None:
            timeouts.append(timeout)
            operation()

        exit_code = run_pipeline(
            args, command_runner=_make_fake_runner(), stage_runner=stage_runner
        )

        assert exit_code == 0
        # pack, artifacts, test -- the test stage is the last and is unbounded.
        assert timeouts[-1] is None
        assert (output_dir / "test_complete.json").is_file()

    def test_max_steps_debug_run_never_reaches_the_test_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bounded debug smoke run must not spend a held-out scoring epoch."""
        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        formal_output = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=tmp_path / "pack", output_dir=formal_output
            ),
        )
        profile = _valid_worker_profile()
        profile["epochs_completed"] = 1
        profile["validations_completed"] = 1
        profile["per_epoch"] = cast(list[object], profile["per_epoch"])[:1]

        def never_called(**_kwargs: object) -> test_protocol.TestProtocolResult:
            raise AssertionError("the test stage must not run for a --max-steps debug run")

        monkeypatch.setattr(test_protocol, "run_test_protocol", never_called)

        def runner(
            command: Sequence[str], timeout: float | None
        ) -> subprocess.CompletedProcess[str]:
            result = _make_fake_runner(train_runtime_profile=profile)(command, timeout)
            if _arg_value(command, "--ddp-mode") == "train":
                out = Path(_arg_value(command, "--output-dir"))
                best = torch.load(out / "best.pt", weights_only=False)
                torch.save(best, out / "last.pt")
                (out / "metrics.jsonl").write_text('{"epoch": 1, "val_auroc": 0.7}\n')
            return result

        assert (
            run_pipeline(PipelineArgs(config_path, None, None, max_steps=1), command_runner=runner)
            == 0
        )
        debug_output = tmp_path / "out_debug"
        assert (debug_output / "debug_complete.json").is_file()
        assert not (debug_output / "test_report.json").exists()
        assert not (debug_output / "test_complete.json").exists()
        assert not (debug_output / "diagnostic_test_report.json").exists()

    def test_diagnostic_run_writes_diagnostic_named_test_artifacts_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diagnostic run's test stage writes `diagnostic_*`, never the formal names."""
        from src import train_b0

        @dataclass(frozen=True)
        class _FakeE2EAwareConfig(train_b0.Config):
            run_kind: str | None = None

        def fake_load_config(path: Path) -> _FakeE2EAwareConfig:
            base = train_b0.load_config(path)
            return _FakeE2EAwareConfig(**{f.name: getattr(base, f.name) for f in fields(base)})

        class DiagnosticAwareWorker:
            pass

        DiagnosticAwareWorker.load_config = staticmethod(fake_load_config)  # type: ignore[attr-defined]
        DiagnosticAwareWorker.prepare_pack = staticmethod(train_b0.prepare_pack)  # type: ignore[attr-defined]

        data_root = tmp_path / "data"
        _write_feature_root(
            data_root / "features" / "frozen_node_features_1024", {"node_a": (3, 4)}
        )
        output_dir = tmp_path / "out"
        config_path = tmp_path / "cfg.yaml"
        _write_pipeline_config(
            config_path,
            _pipeline_config_dict(
                data_root=data_root, pack_dir=tmp_path / "pack", output_dir=output_dir
            ),
        )

        original_import = importlib.import_module
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: (
                DiagnosticAwareWorker if name == "fake.diagnostic_worker" else original_import(name)
            ),
        )

        args = PipelineArgs(
            config=config_path,
            pack_dir=None,
            output_dir=None,
            worker_module="fake.diagnostic_worker",
            run_kind="diagnostic",
        )

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )
        assert (output_dir / "diagnostic_complete.json").is_file()
        assert (output_dir / "diagnostic_test_report.json").is_file()
        assert (output_dir / "diagnostic_test_complete.json").is_file()
        assert not (output_dir / "complete.json").exists()
        assert not (output_dir / "test_report.json").exists()
        assert not (output_dir / "test_complete.json").exists()

    def test_stage_seconds_test_reaches_profile_json_and_manifest_stays_consistent(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )

        profile_path = output_dir / "profile.json"
        profile = json.loads(profile_path.read_text())
        assert profile["stage_seconds"]["test"] >= 0

        manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
        assert (
            manifest["profile.json"]["sha256"]
            == hashlib.sha256(profile_path.read_bytes()).hexdigest()
        )
        assert manifest["profile.json"]["byte_size"] == profile_path.stat().st_size

    # ------------------------------------------------------- republish: stale test sentinel
    # A republish (pack -> train -> publish onto an existing, already-tested
    # output directory) never touches test_report.json/test_complete.json --
    # they sit outside publish/rollback entirely. Without an explicit clear,
    # a failure (or death) in the NEW test stage would leave the OLD
    # checkpoint's sentinel falsely certifying the new one.

    def test_republish_removes_stale_test_completion_when_the_new_test_stage_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )
        assert (output_dir / "test_complete.json").is_file()
        assert (output_dir / "test_report.json").is_file()

        def failing_run_test_protocol(**_kwargs: object) -> test_protocol.TestProtocolResult:
            raise ValueError("scoring blew up on the republished checkpoint")

        monkeypatch.setattr(test_protocol, "run_test_protocol", failing_run_test_protocol)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(),
            stage_runner=lambda operation, _timeout: operation(),
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "test"
        # The republished checkpoint and its publication must survive...
        assert (output_dir / "best.pt").is_file()
        assert (output_dir / "complete.json").is_file()
        # ...but the sentinel certifying the checkpoint it REPLACED must not.
        assert not (output_dir / "test_complete.json").exists()
        assert not (output_dir / "test_report.json").exists()

    def test_republish_removes_stale_test_completion_when_the_new_test_stage_times_out(
        self, tmp_path: Path
    ) -> None:
        """Covers the process-dies-mid-test case: no new sentinel is ever written."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(),
                stage_runner=lambda operation, _timeout: operation(),
            )
            == 0
        )
        assert (output_dir / "test_complete.json").is_file()

        calls = 0

        # `calls` only counts this SECOND run_pipeline invocation (the first
        # ran under its own stage_runner above). Three supervised stages per
        # run: pack, artifacts, then test.
        def stage_runner(operation: Callable[[], None], timeout: float | None) -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise subprocess.TimeoutExpired(cmd="test", timeout=timeout or 0.0)
            operation()

        exit_code = run_pipeline(
            args, command_runner=_make_fake_runner(), stage_runner=stage_runner
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "test"
        assert (output_dir / "complete.json").is_file()
        assert (output_dir / "best.pt").is_file()
        assert not (output_dir / "test_complete.json").exists()
        assert not (output_dir / "test_report.json").exists()

    def test_republish_then_successful_test_leaves_exactly_one_fresh_sentinel(
        self, tmp_path: Path
    ) -> None:
        """The ordinary republish-and-retest path must still end in one clean sentinel."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        for _ in range(2):
            assert (
                run_pipeline(
                    args,
                    command_runner=_make_fake_runner(),
                    stage_runner=lambda operation, _timeout: operation(),
                )
                == 0
            )

        assert (output_dir / "test_complete.json").is_file()
        assert (output_dir / "test_report.json").is_file()
        assert not (output_dir / "diagnostic_test_complete.json").exists()


# --------------------------------------------------------------------------- CLI entry point


def test_main_returns_run_pipelines_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(args: PipelineArgs, **_kwargs: object) -> int:
        captured["args"] = args
        return 2

    monkeypatch.setattr("src.e2_pipeline.run_pipeline", fake_run_pipeline)

    exit_code = main(["--config", str(tmp_path / "cfg.yaml")])

    assert exit_code == 2
    assert cast(PipelineArgs, captured["args"]).config == tmp_path / "cfg.yaml"
