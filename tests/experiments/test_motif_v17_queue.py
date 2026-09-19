from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from src.experiments import motif_v17_queue as queue


def _report(*, arm: float, gates: float, arm_gs: float, base_gs: float) -> dict[str, object]:
    def row(gs: float) -> dict[str, object]:
        return {"panel": {"graph_similarity": {"bfs_macro": gs}}}

    return {
        "cases": {"none": {"auprc": arm}, "gates_off": {"auprc": gates}},
        "matched_output_density": {"rows": {"none": row(arm_gs), "prefix_base": row(base_gs)}},
    }


def test_adoption_requires_strict_gain_and_noninferior_matched_gs() -> None:
    assert queue.adoption(_report(arm=0.8181, gates=0.814, arm_gs=0.41, base_gs=0.40))["adopt"]
    assert not queue.adoption(_report(arm=0.818, gates=0.814, arm_gs=0.41, base_gs=0.40))["adopt"]
    assert not queue.adoption(_report(arm=0.819, gates=0.814, arm_gs=0.39, base_gs=0.40))["adopt"]


def test_wait_requires_exact_predecessor_exit_and_empty_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = {"pid": 7, "start_ticks": "11", "command": "prior queue"}
    states = iter([True, False, False])
    gpu_states = iter([[90], [90], []])
    monkeypatch.setattr(queue, "same_process", lambda identity: next(states))
    monkeypatch.setattr(queue, "gpu_compute_pids", lambda: next(gpu_states))
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    driver = queue.Queue(
        root=tmp_path,
        predecessor=predecessor,
        data_root=Path("data"),
        pack_dir=Path("pack"),
        run_command=lambda command, log: None,
    )
    driver.wait()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "waiting"
    assert status["predecessor_alive"] is False
    assert status["gpu_compute_pids"] == []


def _driver(tmp_path: Path) -> queue.Queue:
    return queue.Queue(
        root=tmp_path,
        predecessor={"pid": 1, "start_ticks": "1", "command": "parent"},
        data_root=Path("data"),
        pack_dir=Path("pack"),
        run_command=lambda command, log: None,
    )


def test_step0_negative_stops_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path)
    monkeypatch.setattr(driver, "wait", lambda: None)
    monkeypatch.setattr(
        driver,
        "step0",
        lambda: {"decisions": {"run_shift": False, "run_confidence": False, "stop": True}},
    )
    monkeypatch.setattr(driver, "train", lambda name: pytest.fail(f"unexpected train {name}"))
    driver.run()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "stopped_negative"
    assert status["stage"] == "step0"


def test_presence_negative_skips_presence_work_and_confidence_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path)
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(driver, "wait", lambda: None)
    monkeypatch.setattr(
        driver,
        "step0",
        lambda: {"decisions": {"run_shift": True, "run_confidence": False, "stop": False}},
    )
    monkeypatch.setattr(
        driver, "prepare_crossfit", lambda kind, names: events.append(("crossfit", kind))
    )
    monkeypatch.setattr(
        driver,
        "train",
        lambda name: events.append(("train", name)) or Path("outputs") / name,
    )
    monkeypatch.setattr(
        driver,
        "train_lane",
        lambda lane: (
            events.append(("train", queue.LANES[lane])) or Path("outputs") / queue.LANES[lane]
        ),
    )
    monkeypatch.setattr(
        driver,
        "validate_lane",
        lambda lane, run: _report(arm=0.81, gates=0.81, arm_gs=0.4, base_gs=0.4),
    )
    monkeypatch.setattr(queue, "select_lane", lambda runs: "shift")
    driver.run()
    assert ("crossfit", "flat") in events
    assert ("crossfit", "presence") not in events
    trained = [name for kind, name in events if kind == "train"]
    assert queue.LANES["confidence"] not in trained
    assert trained == [queue.LANES["shift"], queue.LANES["control"]]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "stopped_negative"


def test_no_passing_generator_pilot_is_scientific_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path)
    monkeypatch.setattr(driver, "wait", lambda: None)
    monkeypatch.setattr(
        driver,
        "step0",
        lambda: {"decisions": {"run_shift": True, "run_confidence": False, "stop": False}},
    )
    reports = [tmp_path / "epoch-0001/pilot_b.json"]
    monkeypatch.setattr(
        driver,
        "prepare_crossfit",
        lambda kind, names: (_ for _ in ()).throw(queue.NoPassingPilotError(reports)),
    )
    driver.run()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "stopped_negative"
    assert status["stage"] == "generator_pilot_b"
    assert status["reports"] == [str(reports[0])]


def test_adoption_replicates_before_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _driver(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(driver, "wait", lambda: None)
    monkeypatch.setattr(
        driver,
        "step0",
        lambda: {"decisions": {"run_shift": True, "run_confidence": True, "stop": False}},
    )
    monkeypatch.setattr(driver, "prepare_crossfit", lambda kind, names: events.append(kind))
    monkeypatch.setattr(driver, "train", lambda name: events.append(name) or Path("outputs") / name)
    monkeypatch.setattr(
        driver,
        "train_lane",
        lambda lane: events.append(queue.LANES[lane]) or Path("outputs") / queue.LANES[lane],
    )
    monkeypatch.setattr(
        driver,
        "validate_lane",
        lambda lane, run: _report(arm=0.819, gates=0.814, arm_gs=0.40, base_gs=0.40),
    )
    monkeypatch.setattr(queue, "select_lane", lambda runs: "confidence")
    driver.generator_runs["presence"] = (Path("fold0"), Path("fold1"))
    monkeypatch.setattr(
        driver,
        "prepare_cache",
        lambda kind, checkpoints, run_seed: (
            events.append(f"cache:{kind}:{run_seed}") or Path(f"cache{run_seed}.pt")
        ),
    )
    monkeypatch.setattr(
        driver,
        "replicate",
        lambda name, seed, cache: events.append(f"{name}_seed{seed}") or Path("outputs") / name,
    )
    monkeypatch.setattr(
        driver,
        "test_and_controls",
        lambda lane, run: events.append(f"test:{lane}"),
    )
    driver.run()
    winner = queue.LANES["confidence"]
    assert events.index("cache:presence:1") < events.index(f"{winner}_seed1")
    assert events.index("cache:presence:2") < events.index(f"{winner}_seed2")
    assert events.index(f"{winner}_seed2") < events.index(f"test:{winner}")
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"


def test_train_refuses_existing_publication_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "arm.yaml"
    publication = tmp_path / "published"
    publication.mkdir()
    config.write_text(f"output_dir: {publication}\n")
    monkeypatch.setattr(queue, "CONFIG_ROOT", tmp_path)
    called = False

    def execute(command: list[str], log: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _driver(tmp_path).train("arm")
    assert called is False


def test_replicate_updates_real_nested_yaml_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs_source"
    output_root = tmp_path / "publications"
    config_root.mkdir()
    source = config_root / "lane.yaml"
    source.write_text(
        "model:\n"
        "  family: v3_1_motif_prompt\n"
        "  config:\n"
        "    motif_prompt:\n"
        "      corruption:\n"
        "        source: predicted\n"
        "seed: 0\n"
        f"output_dir: {output_root / 'lane'}\n"
    )
    monkeypatch.setattr(queue, "CONFIG_ROOT", config_root)
    monkeypatch.setattr(queue, "OUTPUT_ROOT", output_root)

    def publish(command: list[str], log: Path) -> None:
        generated = Path(command[3])
        destination = queue.config_output(generated)
        destination.mkdir(parents=True)
        (destination / "complete.json").write_text('{"status":"complete"}\n')
        (destination / "best.pt").touch()

    driver = queue.Queue(
        root=tmp_path / "campaign",
        predecessor={"pid": 1, "start_ticks": "1", "command": "parent"},
        data_root=Path("data"),
        pack_dir=Path("pack"),
        run_command=publish,
    )
    driver.root.mkdir()
    result = driver.replicate("lane", 1, Path("seed1-cache.pt"))
    generated = yaml.safe_load((driver.root / "configs/lane_seed1.yaml").read_text())
    assert generated["seed"] == 1
    assert (
        generated["model"]["config"]["motif_prompt"]["corruption"]["predicted_cache_path"]
        == "seed1-cache.pt"
    )
    assert result == output_root / "lane_seed1"


def test_default_executor_uses_distinct_exclusive_stage_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def succeed(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(queue.subprocess, "run", succeed)
    driver = queue.Queue(
        root=tmp_path,
        predecessor={"pid": 1, "start_ticks": "1", "command": "parent"},
        data_root=Path("data"),
        pack_dir=Path("pack"),
    )
    driver.command("cache_flat_seed1", ["one"])
    driver.command("cache_flat_seed2", ["two"])
    assert calls == [["one"], ["two"]]
    assert (tmp_path / "logs/cache_flat_seed1.log").exists()
    assert (tmp_path / "logs/cache_flat_seed2.log").exists()


def test_prepare_cache_forwards_nondefault_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    seed_root = output_root / "motif_crossfit/seed42_k2/run_seed0"
    manifest = seed_root / "manifest"
    manifest.mkdir(parents=True)
    for name in ("pairs.json", "nodes.json", "forbidden_nodes.json", "manifest.json"):
        (manifest / name).write_text("[]\n")
    monkeypatch.setattr(queue, "OUTPUT_ROOT", output_root)
    commands: list[list[str]] = []

    def execute(command: list[str], log: Path) -> None:
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True)
        output.touch()
        output.with_suffix(output.suffix + ".report.json").write_text("{}\n")

    driver = queue.Queue(
        root=tmp_path / "campaign",
        predecessor={"pid": 1, "start_ticks": "1", "command": "parent"},
        data_root=tmp_path / "alternate-data",
        pack_dir=Path("pack"),
        run_command=execute,
    )
    driver.root.mkdir()
    driver.deployed_generators["flat"] = Path("deployed.pt")
    driver.prepare_cache("flat", (Path("fold0.pt"), Path("fold1.pt")), run_seed=0)
    command = commands[0]
    assert command[command.index("--data-root") + 1] == str(tmp_path / "alternate-data")
