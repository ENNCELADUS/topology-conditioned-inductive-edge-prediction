"""Study orchestration uses validation only and tests one locked winner."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from src.experiments import l3ppi_study as study


def _selection(index: int) -> dict[str, Any]:
    return {
        "epoch": 10 + index,
        "val_auprc": 0.5 + index * 0.01,
        "topology": {
            "metrics": {
                "gs": 0.5 + index * 0.01,
                "rd": 10.0 if index == 5 else 1.0,
                "degree_mmd": 1.0,
                "clustering_mmd": 1.0,
                "spectral_mmd": 1.0,
            }
        },
    }


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"config": {"k": 16}},
                "optim": {"lr": 0.001},
                "pack_dir": "pack",
                "data_root": "data",
                "seed": 0,
            }
        )
    )
    return path


def _fake_run(calls: list[list[str]]) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if command[1] == "test":
            directory = Path(command[command.index("--output-dir") + 1])
            (directory / "test_report.json").write_text("{}")
        else:
            config = yaml.safe_load(Path(command[2]).read_text())
            directory = Path(config["output_dir"])
            directory.mkdir(exist_ok=True)
            (directory / "best.pt").touch()
            (directory / "complete.json").write_text("{}")
            if directory.name.startswith("trial"):
                index = int(directory.name.split("_")[1])
                (directory / "selection.json").write_text(json.dumps(_selection(index)))
        return subprocess.CompletedProcess(command, 0)

    return run


def test_order_shared_surrogate_validation_ranking_and_only_one_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    result = study.run_study(_config(tmp_path), tmp_path / "study")
    assert result["status"] == "tested"
    assert len(calls) == 8
    assert calls[0][calls[0].index("--stage") + 1] == "surrogate"
    configs = [yaml.safe_load(Path(command[2]).read_text()) for command in calls[1:7]]
    assert [(c["model"]["config"]["k"], c["optim"]["lr"]) for c in configs] == list(study.TRIALS)
    assert len({c["surrogate_checkpoint"] for c in configs}) == 1
    assert len({c["feature_cache"] for c in configs}) == 1
    assert all("--skip-test" in command for command in calls[:7])
    assert calls[-1][1] == "test"
    assert result["winner"]["trial_index"] == 5  # RD=10 does not gate selection.
    assert result["winner"]["model_epoch"] == 15
    study.run_study(tmp_path / "base.yaml", tmp_path / "study", resume=True)
    assert len(calls) == 8


def test_interrupted_resume_reuses_published_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    successful = _fake_run(calls)

    def interrupted(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if "trial_001.yaml" in command[2]:
            config = yaml.safe_load(Path(command[2]).read_text())
            Path(config["output_dir"]).mkdir()
            raise subprocess.CalledProcessError(1, command)
        return successful(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", interrupted)
    config = _config(tmp_path)
    output = tmp_path / "study"
    with pytest.raises(subprocess.CalledProcessError):
        study.run_study(config, output)
    assert (output / "failure.json").exists()
    monkeypatch.setattr(subprocess, "run", successful)
    result = study.run_study(config, output, resume=True, skip_test=True)
    assert result["status"] == "published"
    assert len(calls) == 7
    assert "--resume" in calls[2]
    assert not (output / "failure.json").exists()
    assert not any(command[1] == "test" for command in calls)


def test_prepare_only_conflicting_artifacts_and_changed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    config = _config(tmp_path)
    output = tmp_path / "study"
    assert study.run_study(config, output, prepare_only=True)["status"] == "prepared"
    assert not calls
    with pytest.raises(FileExistsError):
        study.run_study(config, output)
    directory = output / "surrogate"
    directory.mkdir()
    (directory / "complete.json").touch()
    (directory / "failure.json").touch()
    with pytest.raises(RuntimeError, match="Conflicting"):
        study.run_study(config, output, resume=True)
    base = yaml.safe_load(config.read_text())
    base["seed"] = 2
    config.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="configuration differs"):
        study.run_study(config, output, resume=True)
