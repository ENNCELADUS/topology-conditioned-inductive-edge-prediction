import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
import src.data.packed_features as packed_features
import torch
from src.data.packed_features import build_packed_features
from src.e2_pipeline import (
    _PUBLISHED_FILENAMES,
    BudgetExceeded,
    PipelineArgs,
    PipelineProfile,
    ProbeResult,
    _publish_staged,
    _rollback_publication,
    build_accelerate_command,
    detect_visible_gpu_count,
    enforce_projection,
    main,
    parse_pipeline_args,
    project_total_seconds,
    run_command,
    run_pipeline,
    select_probe_result,
    write_failure,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _avoid_pack_process_pool_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline unit tests need pack semantics, not OS process startup."""
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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


# --------------------------------------------------------------------------- Task 7 (unchanged)


def test_selects_fastest_valid_probe_below_memory_limit() -> None:
    results = [
        ProbeResult(262144, True, 1200.0, 40.0, None),
        ProbeResult(524288, True, 1900.0, 70.0, None),
        ProbeResult(1048576, True, 1800.0, 82.0, None),
        ProbeResult(1572864, False, 0.0, 90.0, "memory limit"),
    ]
    assert select_probe_result(results, memory_limit_gib=85.0).token_budget == 524288


def test_projection_includes_all_thirty_epochs() -> None:
    projected = project_total_seconds(
        pack_seconds=240.0,
        setup_probe_seconds=240.0,
        epoch_seconds=90.0,
        epochs=30,
        artifact_seconds=60.0,
    )
    assert projected == pytest.approx(3240.0)


def test_failure_json_is_atomic_and_structured(tmp_path: Path) -> None:
    path = write_failure(tmp_path, stage="probe", message="projected budget miss")
    payload = json.loads(path.read_text())
    assert payload["stage"] == "probe"
    assert payload["message"] == "projected budget miss"
    assert not path.with_suffix(".json.tmp").exists()


# --------------------------------------------------------------------------- command building


def test_accelerate_command_uses_resolved_world_size(tmp_path: Path) -> None:
    command = build_accelerate_command(
        accelerate_bin=Path("/venv/bin/accelerate"),
        config_path=Path("configs/b0_v31_breadth_first.yaml"),
        mode="probe",
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


def test_run_command_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    grandchild_pid_path = tmp_path / "grandchild.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, "-c", script], 0.5)

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


# --------------------------------------------------------------------------- projection gate


def test_projection_budget_failure_writes_artifact(tmp_path: Path) -> None:
    with pytest.raises(BudgetExceeded, match="3600"):
        enforce_projection(projected_seconds=6200.0, limit_seconds=3600, output_dir=tmp_path)
    payload = json.loads((tmp_path / "failure.json").read_text())
    assert payload["stage"] == "projection"
    assert payload["projected_seconds"] == 6200.0


def test_projection_within_budget_does_not_raise_or_write_failure(tmp_path: Path) -> None:
    enforce_projection(projected_seconds=3000.0, limit_seconds=3600, output_dir=tmp_path)
    assert not (tmp_path / "failure.json").exists()


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


# ------------------------------------------------------------------- PipelineProfile round trip
# (closes the Task 7 review gap: from_dict had no round-trip coverage)


def test_pipeline_profile_round_trips_through_json() -> None:
    probes = [
        ProbeResult(262144, True, 1200.0, 40.0, None),
        ProbeResult(524288, False, 0.0, 90.0, "oom"),
    ]
    profile = PipelineProfile(
        cold_cache=True,
        stage_seconds={"pack": 12.0, "setup_probe": 34.0},
        probe_results=probes,
        selected_token_budget=262144,
        projected_total_seconds=3120.5,
    )

    restored = PipelineProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert restored == profile


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


_RUNTIME_TOKEN_BUDGETS = (262144, 524288, 1048576, 1572864)


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
        "token_budget_candidates": list(_RUNTIME_TOKEN_BUDGETS),
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
    return {
        "model": {"family": "v3_1", "config": {}},
        "data": {
            "root": str(data_root),
            "strategy": "breadth_first",
            "train_positives": "train_plus",
            "negative_ratio": 1,
            "partition_seed": 0,
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


def _default_probe_results() -> dict[int, ProbeResult]:
    return {
        262144: ProbeResult(262144, True, 1000.0, 40.0, None),
        524288: ProbeResult(524288, True, 1900.0, 70.0, None),
        1048576: ProbeResult(1048576, True, 1800.0, 82.0, None),
        1572864: ProbeResult(1572864, False, 0.0, 90.0, "oom"),
    }


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
    probe_by_budget: dict[int, ProbeResult] | None = None,
    epoch_seconds: object = 5.0,
    train_runtime_profile: dict[str, object] | None = None,
    fail_mode: str | None = None,
    timeout_mode: str | None = None,
    advance_seconds_by_mode: dict[str, float] | None = None,
    clock: _FakeClock | None = None,
) -> Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]:
    resolved_probes = probe_by_budget or _default_probe_results()
    resolved_train_profile: dict[str, object] = train_runtime_profile or _valid_worker_profile()
    resolved_advances = advance_seconds_by_mode or {}

    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        command_list = list(command)
        mode = _arg_value(command_list, "--ddp-mode")
        profile_output = Path(_arg_value(command_list, "--profile-output"))
        out_dir = Path(_arg_value(command_list, "--output-dir"))
        token_budget = int(_arg_value(command_list, "--token-budget-per-rank"))

        if timeout_mode == mode:
            raise subprocess.TimeoutExpired(cmd=command_list, timeout=timeout)
        advance_by = resolved_advances.get(mode)
        if advance_by:
            assert clock is not None
            clock.advance(advance_by)
        if fail_mode == mode:
            return subprocess.CompletedProcess(command_list, returncode=1, stdout="", stderr="boom")

        if mode == "probe":
            profile_output.write_text(json.dumps(resolved_probes[token_budget].to_dict()))
        elif mode == "epoch-probe":
            profile_output.write_text(
                json.dumps({"epoch_seconds": epoch_seconds, "runtime_profile": {}})
            )
        elif mode == "train":
            _write_train_outputs(out_dir)
            profile_output.write_text(json.dumps(resolved_train_profile))
        return subprocess.CompletedProcess(command_list, returncode=0, stdout="", stderr="")

    return runner


# ---------------------------------------------------------------------- run_pipeline: success paths


class TestRunPipelineSuccess:
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

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
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
        assert profile["selected_token_budget"] == 524288  # fastest valid at/below 85 GiB
        assert profile["epochs_completed"] == 2
        assert set(profile["stage_seconds"]) == {"pack", "setup_probe", "train", "artifacts"}
        assert len(profile["probe_results"]) == 4
        assert profile["total_seconds"] > 0
        assert profile["projected_total_seconds"] > 0
        assert profile["stage_seconds"]["artifacts"] >= 0
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
        assert complete["total_seconds"] >= profile["total_seconds"]

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

        def checked_validate(pack_root: Path, source_root: Path | None) -> object:
            nonlocal validation_calls
            validation_calls += 1
            assert inside_stage, "full pack validation escaped the supervised pack stage"
            return original_validate(pack_root, source_root)

        def supervised(operation: Callable[[], None], _timeout: float) -> None:
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

    def test_unstructured_oom_is_a_systemic_probe_failure(self, tmp_path: Path) -> None:
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        base_runner = _make_fake_runner()
        attempted: list[int] = []

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "probe":
                budget = int(_arg_value(command, "--token-budget-per-rank"))
                attempted.append(budget)
                if budget == _RUNTIME_TOKEN_BUDGETS[0]:
                    return subprocess.CompletedProcess(command, 1, "", "CUDA out of memory")
            return base_runner(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 2
        assert attempted == [_RUNTIME_TOKEN_BUDGETS[0]]
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "probe"

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
            command: Sequence[str], timeout: float
        ) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "train":
                return subprocess.CompletedProcess(command, 0, "", "")
            return _make_fake_runner()(command, timeout)

        assert run_pipeline(args, command_runner=no_write_train) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "artifacts"
        assert (output_dir / "best.pt").read_text() == "stale"
        assert (output_dir / "complete.json").read_text() == "stale"

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

    @pytest.mark.parametrize("fail_mode", ["probe", "epoch-probe", "train"])
    def test_prior_canonical_survives_subprocess_failure(
        self, tmp_path: Path, fail_mode: str
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        before = self._seed_canonical(output_dir)

        assert run_pipeline(args, command_runner=_make_fake_runner(fail_mode=fail_mode)) == 2

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

    def test_prior_canonical_survives_total_budget_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = self._base_args_and_config(
            tmp_path,
            pack_budget_seconds=1,
            setup_probe_budget_seconds=1,
            train_eval_budget_seconds=0,
            artifact_budget_seconds=1,
            reserve_seconds=0,
            total_budget_seconds=3,
        )
        before = self._seed_canonical(output_dir)
        clock = _FakeClock()
        monkeypatch.setattr("src.e2_pipeline.time.monotonic", clock.monotonic)

        assert (
            run_pipeline(
                args,
                command_runner=_make_fake_runner(
                    epoch_seconds=0.001,
                    advance_seconds_by_mode={"train": 3.2},
                    clock=clock,
                ),
            )
            == 2
        )

        self._assert_canonical_unchanged(output_dir, before)

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
        assert not (output_dir / "artifact_manifest.json").exists()
        assert not (output_dir / "complete.json").exists()

    @pytest.mark.parametrize("corrupt", ["checkpoint", "metrics", "metadata"])
    def test_corrupt_staged_artifact_is_rejected_before_hashing(
        self, tmp_path: Path, corrupt: str
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
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

    def test_only_structured_oom_marker_is_candidate_local_and_skips_larger_budgets(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()
        attempted: list[int] = []

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "probe":
                budget = int(_arg_value(command, "--token-budget-per-rank"))
                attempted.append(budget)
                if budget == 524288:
                    marker = (
                        "E2_PROBE_CANDIDATE_FAILURE:"
                        '{"kind":"oom","message":"rank-symmetric CUDA OOM"}'
                    )
                    return subprocess.CompletedProcess(command, 1, "", marker)
            return base(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 0
        profile = json.loads((output_dir / "profile.json").read_text())
        assert attempted == [262144, 524288]
        assert [result["failure"] for result in profile["probe_results"][1:]] == [
            "rank-symmetric CUDA OOM",
            "skipped after smaller token budget OOM",
            "skipped after smaller token budget OOM",
        ]

    def test_generic_oom_text_without_marker_is_systemic(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "probe":
                return subprocess.CompletedProcess(command, 1, "", "launcher CUDA out of memory")
            return base(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 2
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "probe"

    def test_structured_nonfinite_marker_invalidates_only_that_candidate(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()
        attempted: list[int] = []

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "probe":
                budget = int(_arg_value(command, "--token-budget-per-rank"))
                attempted.append(budget)
                if budget == 524288:
                    marker = (
                        "E2_PROBE_CANDIDATE_FAILURE:"
                        '{"kind":"nonfinite","message":"rank-symmetric non-finite loss"}'
                    )
                    return subprocess.CompletedProcess(command, 1, "", marker)
            return base(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 0
        profile = json.loads((output_dir / "profile.json").read_text())
        assert attempted == list(_RUNTIME_TOKEN_BUDGETS)
        assert profile["probe_results"][1]["valid"] is False
        assert profile["probe_results"][1]["failure"] == "rank-symmetric non-finite loss"

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
        args, output_dir = self._base_args_and_config(
            tmp_path, pack_budget_seconds=0, total_budget_seconds=20
        )

        exit_code = run_pipeline(args, command_runner=_make_fake_runner())

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "pack"

    def test_pack_timeout_is_supervised_and_structured(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(
            tmp_path, pack_budget_seconds=1, total_budget_seconds=21
        )
        observed: dict[str, float] = {}

        def timeout_stage(_operation: Callable[[], None], timeout: float) -> None:
            observed["timeout"] = timeout
            raise subprocess.TimeoutExpired(cmd="pack", timeout=timeout)

        exit_code = run_pipeline(
            args, command_runner=_make_fake_runner(), stage_runner=timeout_stage
        )

        assert exit_code == 2
        assert observed["timeout"] == 1
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "pack"

    def test_probe_subprocess_failure_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(fail_mode="probe"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "probe"
        assert failure["returncode"] == 1

    def test_probe_subprocess_timeout_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(timeout_mode="probe"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "probe"
        assert "timeout_seconds" in failure

    def test_unstructured_oom_timeout_is_systemic(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base_runner = _make_fake_runner()
        first = True

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            nonlocal first
            if _arg_value(command, "--ddp-mode") == "probe" and first:
                first = False
                raise subprocess.TimeoutExpired(
                    cmd=command, timeout=timeout, stderr="CUDA out of memory"
                )
            return base_runner(command, timeout)

        assert run_pipeline(args, command_runner=runner) == 2
        assert json.loads((output_dir / "failure.json").read_text())["stage"] == "probe"

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "token_budget": 999,
                "valid": True,
                "global_pairs_per_second": 1.0,
                "peak_memory_gib": 1.0,
                "failure": None,
            },
            {
                "token_budget": 262144,
                "valid": True,
                "global_pairs_per_second": float("nan"),
                "peak_memory_gib": 1.0,
                "failure": None,
            },
        ],
    )
    def test_malformed_probe_profile_persists_partial_evidence(
        self, tmp_path: Path, payload: dict[str, object]
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        def runner(command: Sequence[str], _timeout: float) -> subprocess.CompletedProcess[str]:
            Path(_arg_value(command, "--profile-output")).write_text(json.dumps(payload))
            return subprocess.CompletedProcess(command, 0, "", "")

        assert run_pipeline(args, command_runner=runner) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        evidence = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert failure["stage"] == "probe"
        assert evidence["probe_results"] == []
        assert evidence["stage_seconds"]["pack"] >= 0

    @pytest.mark.parametrize("payload", [None, "not-json", "{}"])
    def test_missing_or_malformed_probe_profile_is_structured_failure(
        self, tmp_path: Path, payload: str | None
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        def runner(command: Sequence[str], _timeout: float) -> subprocess.CompletedProcess[str]:
            profile_output = Path(_arg_value(command, "--profile-output"))
            if payload is not None:
                profile_output.write_text(payload)
            return subprocess.CompletedProcess(command, 0, "", "")

        assert run_pipeline(args, command_runner=runner) == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "probe"
        assert "profile" in failure["message"]

    def test_no_valid_probe_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        all_invalid = {
            budget: ProbeResult(budget, False, 0.0, 99.0, "oom")
            for budget in _RUNTIME_TOKEN_BUDGETS
        }

        exit_code = run_pipeline(
            args, command_runner=_make_fake_runner(probe_by_budget=all_invalid)
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "probe"

    def test_setup_probe_budget_exhausted_before_first_probe_returns_2(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(
            tmp_path, setup_probe_budget_seconds=0, total_budget_seconds=30
        )

        def never_called(
            command: Sequence[str], timeout: float
        ) -> subprocess.CompletedProcess[str]:
            raise AssertionError("command_runner must not be invoked once the budget is exhausted")

        exit_code = run_pipeline(args, command_runner=never_called)

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "probe"

    def test_setup_probe_budget_exhausted_between_probes_and_epoch_probe_returns_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The deadline is shared across the whole setup/probe stage: each of the 4
        # candidate probes eats into it, so it can run out strictly *after* the
        # last probe succeeds but *before* the epoch-probe is ever launched.
        args, output_dir = self._base_args_and_config(
            tmp_path, setup_probe_budget_seconds=2, total_budget_seconds=32
        )
        clock = _FakeClock()
        monkeypatch.setattr("src.e2_pipeline.time.monotonic", clock.monotonic)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(advance_seconds_by_mode={"probe": 0.55}, clock=clock),
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "epoch_probe"
        assert "epoch probe" in failure["message"]

    def test_epoch_probe_subprocess_failure_writes_failure_and_returns_2(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(fail_mode="epoch-probe"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "epoch_probe"

    def test_epoch_probe_subprocess_timeout_writes_failure_and_returns_2(
        self, tmp_path: Path
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(timeout_mode="epoch-probe"))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "epoch_probe"

    @pytest.mark.parametrize("epoch_seconds", [True, 0, -1, float("nan"), float("inf")])
    def test_epoch_probe_rejects_nonpositive_nonfinite_or_bool_seconds(
        self, tmp_path: Path, epoch_seconds: object
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        assert (
            run_pipeline(args, command_runner=_make_fake_runner(epoch_seconds=epoch_seconds)) == 2
        )
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "epoch_probe"
        assert "malformed" in failure["message"]

    def test_projection_over_budget_writes_failure_and_returns_2(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)

        exit_code = run_pipeline(args, command_runner=_make_fake_runner(epoch_seconds=1000.0))

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "projection"
        profile = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert len(profile["probe_results"]) == 4
        assert profile["stage_seconds"]["pack"] >= 0
        assert profile["epoch_probe"]["epoch_seconds"] == 1000.0

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
            command: Sequence[str], timeout: float
        ) -> subprocess.CompletedProcess[str]:
            if _arg_value(command, "--ddp-mode") == "train":
                Path(_arg_value(command, "--profile-output")).write_text("partial")
                return subprocess.CompletedProcess(command, 1, "", "boom")
            return base_runner(command, timeout)

        assert run_pipeline(args, command_runner=corrupting_runner) == 2
        profile = json.loads((output_dir / "failed_run_profile.json").read_text())
        assert len(profile["probe_results"]) == 4
        assert profile["epoch_probe"]["epoch_seconds"] == 5.0

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

    def test_total_elapsed_over_budget_after_artifacts_writes_failure_and_returns_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args, output_dir = self._base_args_and_config(
            tmp_path,
            pack_budget_seconds=1,
            setup_probe_budget_seconds=1,
            train_eval_budget_seconds=0,
            artifact_budget_seconds=1,
            reserve_seconds=0,
            total_budget_seconds=3,
        )
        clock = _FakeClock()
        monkeypatch.setattr("src.e2_pipeline.time.monotonic", clock.monotonic)

        exit_code = run_pipeline(
            args,
            command_runner=_make_fake_runner(
                epoch_seconds=0.001,
                advance_seconds_by_mode={"train": 3.2},
                clock=clock,
            ),
        )

        assert exit_code == 2
        failure = json.loads((output_dir / "failure.json").read_text())
        assert failure["stage"] == "total_budget"
        assert not (output_dir / "profile.json").exists()
        assert not (output_dir / "artifact_manifest.json").exists()

    def test_artifact_timeout_preserves_worker_and_pipeline_evidence(self, tmp_path: Path) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        calls = 0

        def stage_runner(operation: Callable[[], None], timeout: float) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise subprocess.TimeoutExpired(cmd="artifacts", timeout=timeout)
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
        assert len(profile["probe_results"]) == 4


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
