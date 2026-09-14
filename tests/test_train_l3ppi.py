"""Exercise the actual phase lifecycle, loss contract, and interrupted recovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch
import yaml
from src import train_l3ppi as training
from src.baselines.l3ppi import build_l3ppi
from src.data.l3_patterns import L3PatternCache
from src.data.val_region import ValRegionParams, ValRegionSplit
from src.eval.checkpoint_selection import SELECTION_RULE, TopologyValidationMetrics
from src.eval.val_topology import ValTopologyResult

from tests.test_l3ppi_model import config as model_config


@pytest.fixture
def tiny_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    torch.set_num_threads(1)
    cfg = model_config()
    cfg["dropout"] = 0.1
    base = build_l3ppi(cfg)
    checkpoint = tmp_path / "b0.pt"
    torch.save(
        {
            "model_state": base.state_dict(),
            "model_family": "v3_1",
            "model_config": cfg["backbone_config"],
            "epoch": 8,
        },
        checkpoint,
    )
    train_nodes = frozenset("abcdefghij")
    val_nodes = frozenset("wxyz")
    positives = frozenset({("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")})
    split = ValRegionSplit(
        train_nodes,
        val_nodes,
        ("w",),
        positives,
        (),
        (("w", "x"), ("y", "z")),
        (("w", "y"), ("x", "z")),
        {4: [set(val_nodes)]},
        ValRegionParams(),
    )
    generator = torch.Generator().manual_seed(11)
    features = {n: torch.randn(16, generator=generator) for n in sorted(train_nodes | val_nodes)}
    monkeypatch.setattr(training, "_load_val_region_split", lambda *_: split)
    monkeypatch.setattr(
        training,
        "load_packed_manifest",
        lambda *_: SimpleNamespace(node_index=lambda: dict.fromkeys(features)),
    )
    monkeypatch.setattr(training, "load_features", lambda *_: features)
    monkeypatch.setattr(
        training,
        "val_ball_union_universe",
        lambda *_: SimpleNamespace(u_idx=np.array([0, 0, 1, 2]), v_idx=np.array([1, 2, 3, 3])),
    )
    monkeypatch.setattr(
        training,
        "val_region_topology_metrics",
        lambda **_: ValTopologyResult(TopologyValidationMetrics(0.5, 1.0, 1.0, 1.0, 1.0), 0.25),
    )
    return {
        "model": {"family": "l3ppi", "config": cfg},
        "backbone_checkpoint": str(checkpoint),
        "surrogate_checkpoint": str(tmp_path / "surrogate" / "best.pt"),
        "feature_cache": str(tmp_path / "features.pt"),
        "pack_dir": "unused",
        "data_root": "unused",
        "encode_batch_size": 2,
        "batch_size": 7,
        "output_dir": str(tmp_path / "surrogate"),
        "seed": 0,
        "optim": {
            "lr": 1e-3,
            "surrogate_lr": 1e-3,
            "surrogate_epochs": 2,
            "prompt_epochs": 2,
            "joint_epochs": 2,
            "patience": 10,
        },
        "gamma": 3,
        "path_weight": 0.3,
    }


def test_real_three_stage_lifecycle(tiny_training: dict[str, Any], tmp_path: Path) -> None:
    cfg = tiny_training
    device = torch.device("cpu")
    training.run(cfg, "surrogate", False, 0, 1, device)
    surrogate = torch.load(cfg["surrogate_checkpoint"], weights_only=True)
    assert surrogate["stage"] == "surrogate"
    assert not surrogate["converged"]  # Two epochs are a test cap, not scientific convergence.
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(tmp_path / "trial")
    training.run(cfg, "trial", False, 0, 1, device)
    output = Path(cfg["output_dir"])
    best = torch.load(output / "best.pt", weights_only=True)
    assert best["selection_rule"] == SELECTION_RULE
    assert best["val_threshold_transfer"]["threshold"] == 0.25
    assert all(
        torch.equal(value, best["model_state"][key])
        for key, value in surrogate["model_state"].items()
        if key.startswith(("encoder.", "surrogate."))
    )
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [(row["phase"], row["epoch"]) for row in rows] == [
        ("prompt", 1),
        ("prompt", 2),
        ("joint", 1),
        ("joint", 2),
    ]
    assert all(row["train_path_loss"] == 0 for row in rows[:2])
    assert not (output / "test_report.json").exists()
    training.run(cfg, "trial", True, 0, 1, device)  # Completed resume is a no-op.


@pytest.mark.parametrize("interrupted_phase", ["prompt", "joint"])
def test_resume_matches_uninterrupted_weights(
    tiny_training: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_phase: str,
) -> None:
    cfg = tiny_training
    device = torch.device("cpu")
    training.run(cfg, "surrogate", False, 0, 1, device)
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = str(tmp_path / "reference")
    training.run(cfg, "trial", False, 0, 1, device)
    reference = torch.load(Path(cfg["output_dir"]) / "last.pt", weights_only=True)
    cfg["output_dir"] = str(tmp_path / "interrupted")
    original = training.train_epoch
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, float]:  # noqa: ANN401
        nonlocal calls
        if args[6] == interrupted_phase:
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption after committed epoch")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.run(cfg, "trial", False, 0, 1, device)
    monkeypatch.setattr(training, "train_epoch", original)
    training.run(cfg, "trial", True, 0, 1, device)
    resumed = torch.load(Path(cfg["output_dir"]) / "last.pt", weights_only=True)
    for key, value in reference["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][key], rtol=0, atol=0)
    assert len(resumed["history"]) == len(reference["history"]) == 4


def test_weighted_bce_denominator_matches_b0() -> None:
    model = build_l3ppi(model_config())
    model.surrogate.requires_grad_(True)
    # Zero parameters gives constant zero logits, regardless of labels/graph size.
    with torch.no_grad():
        for parameter in model.surrogate.parameters():
            parameter.zero_()
    rows = [("a", "b", 1), ("a", "a", 0)]
    features = {"a": torch.zeros(16), "b": torch.zeros(16)}
    patterns = L3PatternCache(nx.empty_graph(["a", "b"]), {"a", "b"})
    optimizer = torch.optim.SGD(model.surrogate.parameters(), lr=0)
    stats = training.train_epoch(
        model,
        model.surrogate,
        optimizer,
        rows,
        features,
        patterns,
        "surrogate",
        {"batch_size": 2, "path_weight": 0.3},
        1.0,
        0,
        1,
        torch.device("cpu"),
    )
    assert stats["train_task_loss"] == pytest.approx(np.log(2))


def test_convergence_and_temperature() -> None:
    assert not training.converged([0.1] * 9)
    assert training.converged([1.0, *([0.1] * 10)])
    assert not training.converged([0.1] * 9 + [0.2])
    assert training.temperature(1, 25) == 1
    assert training.temperature(25, 25) == pytest.approx(0.1)


def test_duplicate_cli_does_not_mark_published_run_failed(
    tiny_training: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tiny_training
    training.run(cfg, "surrogate", False, 0, 1, torch.device("cpu"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("sys.argv", ["train_l3ppi", str(path), "--stage", "surrogate"])
    with pytest.raises(FileExistsError, match="completed output"):
        training.main()
    assert not (Path(cfg["output_dir"]) / "failure.json").exists()
