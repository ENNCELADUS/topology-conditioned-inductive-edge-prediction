import hashlib
import importlib
import json
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import src.data.packed_features as packed_features
import torch
from src.data.packed_features import build_packed_features
from src.e2_pipeline import (
    _PUBLISHED_FILENAMES,
    V_HOLD_VALIDATION_EVENTS_FILENAME,
    PipelineArgs,
    ProbeResult,
    _publish_staged,
    _rollback_publication,
    build_accelerate_command,
    detect_visible_gpu_count,
    main,
    parse_pipeline_args,
    run_command,
    run_pipeline,
    write_failure,
)

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def _avoid_pack_process_pool_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline unit tests need pack semantics, not OS process startup."""
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)


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


@pytest.mark.parametrize("run_kind", ["formal"])
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
        # `total_budget_seconds`, so an override of any one of them has to be
        # re-balanced or the config is rejected before the pipeline ever runs.
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
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {"config_hash": "abc123"}
        )
    )


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
) -> Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]:
    """Stand in for the single ``train`` process group the orchestrator launches."""
    resolved_train_profile: dict[str, object] = train_runtime_profile or _valid_worker_profile()

    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        command_list = list(command)
        mode = _arg_value(command_list, "--ddp-mode")
        profile_output = Path(_arg_value(command_list, "--profile-output"))
        out_dir = Path(_arg_value(command_list, "--output-dir"))

        if timeout_mode == mode:
            raise subprocess.TimeoutExpired(cmd=command_list, timeout=timeout)
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

    def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
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
        assert profile["token_budget"] == _RUNTIME_TOKEN_BUDGET
        assert profile["epochs_completed"] == 2
        assert set(profile["stage_seconds"]) == {"pack", "train", "artifacts"}
        assert profile["total_seconds"] > 0
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

    def test_oom_during_training_is_a_train_stage_failure(self, tmp_path: Path) -> None:
        """With no candidate sweep left, an OOM is simply a failed train stage."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)

        def runner(command: Sequence[str], _timeout: float) -> subprocess.CompletedProcess[str]:
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

    def test_failed_train_rescues_worker_history_evidence(self, tmp_path: Path) -> None:
        """A worker-side failed_run_history.json survives staging-dir teardown."""
        args, output_dir = TestRunPipelineFailures()._base_args_and_config(tmp_path)
        base_runner = _make_fake_runner()

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
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

    @pytest.mark.parametrize("payload", [None, "not-json", "{}"])
    def test_missing_or_malformed_train_profile_is_a_structured_artifact_failure(
        self, tmp_path: Path, payload: str | None
    ) -> None:
        args, output_dir = self._base_args_and_config(tmp_path)
        base = _make_fake_runner()

        def runner(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
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
            command: Sequence[str], timeout: float
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
