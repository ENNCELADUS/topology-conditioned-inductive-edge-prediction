"""Qualification verdict coverage for pipeline-owned failure paths."""

import json
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from src.e2_pipeline import PipelineArgs, run_pipeline

from tests import test_e2_pipeline as pipeline_tests

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _ModelConfig:
    family: str = "v3_1"
    config: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            object.__setattr__(self, "config", {})


@dataclass(frozen=True)
class _DataConfig:
    strategy: str = "breadth_first"
    train_positives: str = "train_plus"
    negative_ratio: int = 5
    partition_seed: int = 0
    msg_fraction: float = 0.5
    node_batch: int = 1
    edge_batch: int = 1
    expected_missing_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 2
    warmup_steps: int = 0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class _RuntimeConfig:
    pack_dir: Path
    world_size: int = 4
    pack_budget_seconds: int = 3
    train_eval_budget_seconds: int = 3
    artifact_budget_seconds: int = 3
    token_budget: int = 8
    memory_limit_gib: float = 85.0


@dataclass(frozen=True)
class _QualificationConfig:
    """Minimal config accepted by both the B0 pack seam and verdict writer."""

    model: _ModelConfig
    data: _DataConfig
    optim: _OptimConfig
    runtime: _RuntimeConfig
    output_dir: Path
    seed: int = 0
    mixed_precision: str = "bf16"
    run_kind: str | None = None
    training: object | None = None


@pytest.fixture
def qualification_worker(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a lightweight worker that uses the production verdict schema."""
    module_name = "tests._qualification_failure_worker"
    module = types.ModuleType(module_name)

    def load_config(path: Path) -> _QualificationConfig:
        return _QualificationConfig(
            model=_ModelConfig(),
            data=_DataConfig(),
            optim=_OptimConfig(),
            runtime=_RuntimeConfig(pack_dir=path.parent / "pack"),
            output_dir=path.parent / "out",
        )

    def prepare_pack(
        _cfg: _QualificationConfig,
        pack_dir: Path,
        *,
        cold_cache: bool,
        temp_prefix: str,
    ) -> dict[str, object]:
        del temp_prefix
        pack_dir.mkdir(parents=True, exist_ok=True)
        return {
            "pack_manifest": {"cold": cold_cache},
            "pack_identity_sha256": "ab" * 32,
            "packs": {},
        }

    module.load_config = load_config  # type: ignore[attr-defined]
    module.prepare_pack = prepare_pack  # type: ignore[attr-defined]

    def write_qualification_artifact(
        cfg: _QualificationConfig,
        *,
        verdict: str,
        feature_stats_sha256: str,
        output_dir: Path | None = None,
    ) -> Path:
        destination = cfg.output_dir if output_dir is None else output_dir
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "qualification.json"
        path.write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "epochs": cfg.optim.epochs,
                    "hparams": {
                        "lr": cfg.optim.lr,
                        "weight_decay": cfg.optim.weight_decay,
                        "warmup_steps": cfg.optim.warmup_steps,
                        "grad_clip": cfg.optim.grad_clip,
                        "seed": cfg.seed,
                        "negative_ratio": cfg.data.negative_ratio,
                        "node_batch": cfg.data.node_batch,
                        "edge_batch": cfg.data.edge_batch,
                        "mixed_precision": cfg.mixed_precision,
                        "training": None,
                    },
                    "feature_stats_sha256": feature_stats_sha256,
                    "model_config_sha256": "cd" * 32,
                }
            )
        )
        return path

    module.write_qualification_artifact = write_qualification_artifact  # type: ignore[attr-defined]
    def validate_qualification_artifact(
        path: Path,
        _cfg: _QualificationConfig,
        *,
        feature_stats_sha256: str,
    ) -> dict[str, object]:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError("qualification artifact must contain an object")
        if payload.get("verdict") != "pass":
            raise RuntimeError("qualification verdict is not pass")
        if payload.get("model_config_sha256") != "cd" * 32:
            raise RuntimeError("qualification model identity mismatch")
        if payload.get("feature_stats_sha256") != feature_stats_sha256:
            raise RuntimeError("qualification feature identity mismatch")
        return payload

    module.validate_qualification_artifact = (  # type: ignore[attr-defined]
        validate_qualification_artifact
    )
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def _qualification_args(tmp_path: Path, worker_module: str) -> tuple[PipelineArgs, Path]:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("unused: true\n")
    output_dir = tmp_path / "out"
    args = PipelineArgs(config_path, None, None)
    return (
        replace(args, worker_module=worker_module, run_kind="qualification", epochs=2),
        output_dir,
    )


def test_pre_pack_failure_writes_named_qualification_verdict(
    tmp_path: Path, qualification_worker: str
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)
    worker = sys.modules[qualification_worker]

    def fail_before_pack(_cfg: object, _pack_dir: Path) -> tuple[Path, ...]:
        raise RuntimeError("pack paths unavailable")

    worker.required_pack_paths = fail_before_pack  # type: ignore[attr-defined]

    assert run_pipeline(args) == 2
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "pack"
    qualification = json.loads((output_dir / "qualification.json").read_text())
    assert qualification["verdict"] == "fail(pack_stage)"
    assert qualification["feature_stats_sha256"] == ""


def test_worker_launch_failure_writes_named_qualification_verdict(
    tmp_path: Path, qualification_worker: str
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)

    def launch_failure(
        _command: Sequence[str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        raise OSError("accelerate executable unavailable")

    assert run_pipeline(args, command_runner=launch_failure) == 2
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "train"
    qualification = json.loads((output_dir / "qualification.json").read_text())
    assert qualification["verdict"] == "fail(training_worker)"
    assert qualification["feature_stats_sha256"] == ""


def test_staged_artifact_failure_replaces_worker_pass_with_failure(
    tmp_path: Path, qualification_worker: str
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)
    base = pipeline_tests._qualification_runner(verdict="pass")

    def corrupt_checkpoint(
        command: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        completed = base(command, timeout)
        staging_dir = Path(pipeline_tests._arg_value(command, "--output-dir"))
        (staging_dir / "best.pt").write_bytes(b"invalid checkpoint")
        return completed

    assert run_pipeline(args, command_runner=corrupt_checkpoint) == 2
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
    qualification = json.loads((output_dir / "qualification.json").read_text())
    assert qualification["verdict"] == "fail(staged_artifacts)"
    assert qualification["feature_stats_sha256"] == "ab" * 32
    assert (output_dir / "run_metadata.json").is_file()
    assert (output_dir / "v_hold_validation_events.jsonl").is_file()


def test_success_preserves_worker_pass_verdict(
    tmp_path: Path, qualification_worker: str
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)
    runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] = (
        pipeline_tests._qualification_runner(verdict="pass")
    )

    assert run_pipeline(args, command_runner=runner) == 0
    assert json.loads((output_dir / "qualification.json").read_text())["verdict"] == "pass"
    assert not (output_dir / "failure.json").exists()


@pytest.mark.parametrize("failure_mode", ["missing", "malformed", "schema", "named_schema"])
def test_zero_exit_refuses_missing_or_malformed_qualification_artifact(
    tmp_path: Path,
    qualification_worker: str,
    failure_mode: str,
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)
    base = pipeline_tests._qualification_runner(verdict="pass")

    def invalid_artifact_runner(
        command: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        completed = base(command, timeout)
        artifact = Path(pipeline_tests._arg_value(command, "--output-dir")) / "qualification.json"
        if failure_mode == "missing":
            artifact.unlink()
        elif failure_mode == "malformed":
            artifact.write_text("not-json")
        else:
            payload = json.loads(artifact.read_text())
            payload.pop("hparams")
            if failure_mode == "named_schema":
                payload["verdict"] = "fail(slot_collapse)"
            artifact.write_text(json.dumps(payload))
        return completed

    assert run_pipeline(args, command_runner=invalid_artifact_runner) == 2
    assert not (output_dir / "complete.json").exists()
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
    assert json.loads((output_dir / "qualification.json").read_text())["verdict"] == (
        "fail(staged_artifacts)"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 3),
        ("model_config_sha256", "ef" * 32),
        ("feature_stats_sha256", "ef" * 32),
    ],
)
def test_zero_exit_refuses_identity_mismatched_qualification_artifact(
    tmp_path: Path,
    qualification_worker: str,
    field: str,
    value: object,
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)
    base = pipeline_tests._qualification_runner(verdict="pass")

    def mismatched_artifact_runner(
        command: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        completed = base(command, timeout)
        artifact = Path(pipeline_tests._arg_value(command, "--output-dir")) / "qualification.json"
        payload = json.loads(artifact.read_text())
        payload[field] = value
        artifact.write_text(json.dumps(payload))
        return completed

    assert run_pipeline(args, command_runner=mismatched_artifact_runner) == 2
    assert not (output_dir / "complete.json").exists()
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
    assert json.loads((output_dir / "qualification.json").read_text())["verdict"] == (
        "fail(staged_artifacts)"
    )


def test_zero_exit_preserves_worker_named_failure_instead_of_completing(
    tmp_path: Path, qualification_worker: str
) -> None:
    args, output_dir = _qualification_args(tmp_path, qualification_worker)

    assert (
        run_pipeline(
            args,
            command_runner=pipeline_tests._qualification_runner(
                verdict="training_invalid(slot_collapse)"
            ),
        )
        == 2
    )
    assert not (output_dir / "complete.json").exists()
    assert json.loads((output_dir / "failure.json").read_text())["stage"] == "artifacts"
    assert json.loads((output_dir / "qualification.json").read_text())["verdict"] == (
        "training_invalid(slot_collapse)"
    )
