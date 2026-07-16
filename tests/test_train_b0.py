"""Tests for src.train_b0: config schema, model builder, train loop, output writer."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
import src.data.packed_features as packed_features
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from src.data.artifacts import ArtifactVerificationError
from src.data.distributed_pairs import CompactPairBatchDataset, PairBatchSpec
from src.data.packed_features import PackedFeatureTable, build_packed_features
from src.data.pairs import TokenPairDataset
from src.eval.edge_metrics import EdgeMetrics
from src.model.B0 import BEST_V3_1_CONFIG, V3_1
from src.model.b0_alt import F0PairMLP
from src.train_b0 import (
    AssembledData,
    Config,
    GpuBatchIterable,
    TrainResult,
    _all_ranks_loss_finite,
    _build_packed_v3_1_loaders,
    _build_v3_1_loaders,
    _cycle_assembled_batches,
    _EpochGpuBatchIterable,
    _evaluate_distributed,
    _interleave_bucket_specs,
    _run_probe_mode,
    _run_timed_epoch_probe,
    _validate_distributed_plan,
    apply_overrides,
    assemble_data,
    build_ddp_accelerator,
    build_model,
    compute_sample_warmup_steps,
    config_to_dict,
    load_config,
    parse_args,
    resolve_model_kwargs,
    scale_ddp_mean_loss,
    train_ddp_loop,
    train_loop,
    validate_gathered_validation,
    write_outputs,
)
from torch.utils.data import DataLoader

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- helpers


def _write_yaml_config(path: Path, overrides: dict[str, object] | None = None) -> dict[str, object]:
    base: dict[str, object] = {
        "model": {"family": "f0_mlp", "config": {}},
        "data": {
            "root": "data",
            "strategy": "breadth_first",
            "train_positives": "train_plus",
            "negative_ratio": 5,
            "partition_seed": 0,
            "token_budget": 131072,
            "batch_pairs": 1024,
            "num_workers": 0,
            "f0_cache": "outputs/f0_cache/f0_matrix.pt",
            "expected_missing_features": ["node_004764", "node_007050"],
        },
        "optim": {
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "epochs": 30,
            "warmup_steps": 500,
            "grad_clip": 1.0,
        },
        "eval": {"patience": 8, "eval_every": 1},
        "seed": 47,
        "output_dir": "outputs/b0_alt",
        "mixed_precision": "no",
    }
    if overrides:
        for dotted_key, value in overrides.items():
            section, _, key = dotted_key.partition(".")
            if key:
                cast_section = base[section]
                assert isinstance(cast_section, dict)
                cast_section[key] = value
            else:
                base[dotted_key] = value
    path.write_text(json.dumps(base))
    return base


def _runtime_dict() -> dict[str, object]:
    return {
        "world_size": 4,
        "pack_dir": "outputs/feature_packs/b0_v31_bf16",
        "pack_workers": 16,
        "loader_workers_per_rank": 4,
        "prefetch_factor": 4,
        "token_budget_candidates": [262144, 524288, 1048576, 1572864],
        "max_pairs_per_rank": 4096,
        "memory_limit_gib": 85.0,
        "total_budget_seconds": 3600,
        "pack_budget_seconds": 300,
        "setup_probe_budget_seconds": 300,
        "train_eval_budget_seconds": 2820,
        "artifact_budget_seconds": 60,
        "reserve_seconds": 120,
        "probe_warmup_steps": 10,
        "probe_timed_steps": 30,
    }


def _make_synthetic_pair_dataset(
    n: int, input_dim: int = 4, *, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    """Build a batch list of linearly separable pair examples for f0_mlp."""
    rng = np.random.default_rng(seed)
    items: list[dict[str, torch.Tensor]] = []
    for i in range(n):
        label = i % 2
        base = rng.normal(loc=3.0 if label == 1 else -3.0, scale=0.1, size=input_dim)
        x_a = torch.tensor(base + rng.normal(scale=0.05, size=input_dim), dtype=torch.float32)
        x_b = torch.tensor(base + rng.normal(scale=0.05, size=input_dim), dtype=torch.float32)
        items.append(
            {
                "x_a": x_a,
                "x_b": x_b,
                "label": torch.tensor(float(label)),
            }
        )
    return items


def _batch_of(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in ("x_a", "x_b", "label")}


# --------------------------------------------------------------------------- load_config


class TestLoadConfig:
    def test_valid_config_loads_all_fields(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)

        cfg = load_config(config_path)

        assert cfg.model.family == "f0_mlp"
        assert cfg.model.config == {}
        assert cfg.data.root == Path("data")
        assert cfg.data.strategy == "breadth_first"
        assert cfg.data.train_positives == "train_plus"
        assert cfg.data.negative_ratio == 5
        assert cfg.data.partition_seed == 0
        assert cfg.data.token_budget == 131072
        assert cfg.data.batch_pairs == 1024
        assert cfg.data.num_workers == 0
        assert cfg.data.f0_cache == Path("outputs/f0_cache/f0_matrix.pt")
        assert cfg.data.expected_missing_features == ["node_004764", "node_007050"]
        assert cfg.optim.lr == pytest.approx(1.0e-4)
        assert cfg.optim.epochs == 30
        assert cfg.eval.patience == 8
        assert cfg.eval.eval_every == 1
        assert cfg.seed == 47
        assert cfg.output_dir == Path("outputs/b0_alt")
        assert cfg.mixed_precision == "no"

    def test_loads_four_h20_runtime_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"runtime": _runtime_dict()})

        cfg = load_config(config_path)

        assert cfg.runtime is not None
        assert cfg.runtime.world_size == 4

    def test_loads_auto_world_size_runtime_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        runtime = _runtime_dict()
        runtime["world_size"] = "auto"
        _write_yaml_config(config_path, {"runtime": runtime})

        cfg = load_config(config_path)

        assert cfg.runtime is not None
        assert cfg.runtime.world_size == 0
        assert cfg.runtime.token_budget_candidates == [262144, 524288, 1048576, 1572864]
        assert cfg.runtime.total_budget_seconds == 3600

    def test_runtime_budget_must_sum_to_total(self, tmp_path: Path) -> None:
        runtime = _runtime_dict()
        runtime["reserve_seconds"] = 119
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"runtime": runtime})

        with pytest.raises(ValueError, match="runtime stage budgets must sum to 3600"):
            load_config(config_path)

    def test_unknown_family_raises_clear_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"model": {"family": "bogus_family"}})

        with pytest.raises(ValueError, match="bogus_family"):
            load_config(config_path)

    def test_unknown_train_positives_raises_clear_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        base = _write_yaml_config(config_path)
        data_section = base["data"]
        assert isinstance(data_section, dict)
        data_section["train_positives"] = "bogus_mode"
        config_path.write_text(json.dumps(base))

        with pytest.raises(ValueError, match="train_positives"):
            load_config(config_path)

    def test_missing_top_level_section_raises_clear_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        base = _write_yaml_config(config_path)
        del base["optim"]
        config_path.write_text(json.dumps(base))

        with pytest.raises(ValueError, match="optim"):
            load_config(config_path)

    def test_missing_nested_key_raises_clear_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        base = _write_yaml_config(config_path)
        optim_section = base["optim"]
        assert isinstance(optim_section, dict)
        del optim_section["lr"]
        config_path.write_text(json.dumps(base))

        with pytest.raises(ValueError, match="lr"):
            load_config(config_path)

    def test_model_config_defaults_to_empty_dict_when_absent(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        base = _write_yaml_config(config_path)
        model_section = base["model"]
        assert isinstance(model_section, dict)
        del model_section["config"]
        config_path.write_text(json.dumps(base))

        cfg = load_config(config_path)

        assert cfg.model.config == {}


# --------------------------------------------------------------------------- CLI overrides


class TestCliOverrides:
    def test_seed_and_output_dir_overrides_win(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)
        cfg = load_config(config_path)

        args = parse_args(
            ["--config", str(config_path), "--seed", "999", "--output-dir", "outputs/custom"]
        )
        overridden = apply_overrides(cfg, args)

        assert overridden.seed == 999
        assert overridden.output_dir == Path("outputs/custom")

    def test_no_overrides_leaves_config_unchanged(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)
        cfg = load_config(config_path)

        args = parse_args(["--config", str(config_path)])
        unchanged = apply_overrides(cfg, args)

        assert unchanged == cfg

    def test_max_steps_defaults_to_none(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)

        args = parse_args(["--config", str(config_path)])

        assert args.max_steps is None

    def test_max_steps_parses_as_int(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)

        args = parse_args(["--config", str(config_path), "--max-steps", "20"])

        assert args.max_steps == 20


# --------------------------------------------------------------------------- build_model


class TestBuildModel:
    def test_v3_1_with_empty_config_uses_best_v3_1_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"model": {"family": "v3_1", "config": {}}})
        cfg = load_config(config_path)

        model = build_model(cfg)

        assert isinstance(model, V3_1)
        assert model.d_model == BEST_V3_1_CONFIG["d_model"]

    def test_v3_1_with_explicit_tiny_config(self, tmp_path: Path) -> None:
        tiny_config = {
            "input_dim": 8,
            "d_model": 32,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 4,
            "mlp_head": {"hidden_dims": [16], "dropout": 0.0},
            "regularization": {"dropout": 0.0},
        }
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"model": {"family": "v3_1", "config": tiny_config}})
        cfg = load_config(config_path)

        model = build_model(cfg)

        assert isinstance(model, V3_1)
        assert model.d_model == 32

    def test_f0_mlp_with_empty_config_uses_defaults(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path, {"model": {"family": "f0_mlp", "config": {}}})
        cfg = load_config(config_path)

        model = build_model(cfg)

        assert isinstance(model, F0PairMLP)
        assert model.input_dim == 1536

    def test_f0_mlp_with_explicit_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(
            config_path,
            {"model": {"family": "f0_mlp", "config": {"input_dim": 8, "hidden_dims": [16]}}},
        )
        cfg = load_config(config_path)

        model = build_model(cfg)

        assert isinstance(model, F0PairMLP)
        assert model.input_dim == 8

    def test_unknown_family_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        base = _write_yaml_config(config_path)
        cfg = load_config(config_path)
        bogus_cfg = Config(
            model=type(cfg.model)(family="not_a_family", config={}),
            data=cfg.data,
            optim=cfg.optim,
            eval=cfg.eval,
            seed=cfg.seed,
            output_dir=cfg.output_dir,
            mixed_precision=cfg.mixed_precision,
        )
        del base

        with pytest.raises(ValueError, match="not_a_family"):
            build_model(bogus_cfg)


# ------------------------------------------------- resolve_model_kwargs / config_to_dict


class TestResolveModelKwargs:
    def test_v3_1_empty_resolves_to_best_config(self) -> None:
        from src.train_b0 import ModelConfig

        resolved = resolve_model_kwargs(ModelConfig(family="v3_1", config={}))

        assert resolved == BEST_V3_1_CONFIG

    def test_f0_mlp_empty_resolves_to_empty_dict(self) -> None:
        from src.train_b0 import ModelConfig

        resolved = resolve_model_kwargs(ModelConfig(family="f0_mlp", config={}))

        assert resolved == {}


class TestConfigToDict:
    def test_round_trips_to_json_serializable_dict(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.yaml"
        _write_yaml_config(config_path)
        cfg = load_config(config_path)

        resolved = config_to_dict(cfg)
        # Must be JSON-serializable (Paths -> str) for the config hash.
        serialized = json.dumps(resolved, sort_keys=True)

        assert isinstance(serialized, str)
        assert resolved["seed"] == 47
        assert resolved["data"]["root"] == "data"


# ----------------------------------------------------- train_loop (synthetic, in-memory)


def _tiny_v3_1_model() -> V3_1:
    return V3_1(
        input_dim=8,
        d_model=16,
        encoder_layers=1,
        cross_attn_layers=1,
        n_heads=4,
        mlp_head={"hidden_dims": [8], "dropout": 0.0},
        regularization={"dropout": 0.0},
    )


def _tiny_config(epochs: int = 5, patience: int = 8, eval_every: int = 1) -> Config:
    from src.train_b0 import DataConfig, EvalConfig, ModelConfig, OptimConfig

    return Config(
        model=ModelConfig(family="f0_mlp", config={}),
        data=DataConfig(
            root=Path("data"),
            strategy="breadth_first",
            train_positives="train_plus",
            negative_ratio=5,
            partition_seed=0,
            token_budget=131072,
            batch_pairs=1024,
            num_workers=0,
            f0_cache=Path("outputs/f0_cache/f0_matrix.pt"),
            expected_missing_features=["node_004764", "node_007050"],
        ),
        optim=OptimConfig(
            lr=1.0e-2, weight_decay=0.0, epochs=epochs, warmup_steps=1, grad_clip=1.0
        ),
        eval=EvalConfig(patience=patience, eval_every=eval_every),
        seed=0,
        output_dir=Path("outputs/test"),
        mixed_precision="no",
    )


class TestTrainLoopSyntheticF0Mlp:
    def test_loss_decreases_over_epochs(self) -> None:
        torch.manual_seed(0)
        model = F0PairMLP(input_dim=4, hidden_dims=(16,), dropout=0.0)
        train_items = _make_synthetic_pair_dataset(64, input_dim=4, seed=1)
        val_items = _make_synthetic_pair_dataset(32, input_dim=4, seed=2)
        train_batch = _batch_of(train_items)
        val_batch = _batch_of(val_items)
        cfg = _tiny_config(epochs=10)
        accelerator = Accelerator(cpu=True)

        def train_loader_factory(epoch: int) -> list[dict[str, torch.Tensor]]:
            return [train_batch]

        result = train_loop(model, train_loader_factory, [val_batch], cfg, accelerator)

        first_loss = result.history[0]["train_loss"]
        last_loss = result.history[-1]["train_loss"]
        assert isinstance(first_loss, float)
        assert isinstance(last_loss, float)
        assert last_loss < first_loss

    def test_metrics_history_entries_have_expected_keys(self) -> None:
        torch.manual_seed(0)
        model = F0PairMLP(input_dim=4, hidden_dims=(16,), dropout=0.0)
        train_batch = _batch_of(_make_synthetic_pair_dataset(32, input_dim=4, seed=1))
        val_batch = _batch_of(_make_synthetic_pair_dataset(16, input_dim=4, seed=2))
        cfg = _tiny_config(epochs=3)
        accelerator = Accelerator(cpu=True)

        result = train_loop(model, lambda epoch: [train_batch], [val_batch], cfg, accelerator)

        for entry in result.history:
            assert set(entry.keys()) == {"epoch", "train_loss", "val_auroc", "val_auprc"}

    def test_checkpoint_save_reload_reproduces_logits_exactly(self) -> None:
        torch.manual_seed(0)
        model = F0PairMLP(input_dim=4, hidden_dims=(16,), dropout=0.0)
        train_batch = _batch_of(_make_synthetic_pair_dataset(32, input_dim=4, seed=1))
        val_batch = _batch_of(_make_synthetic_pair_dataset(16, input_dim=4, seed=2))
        cfg = _tiny_config(epochs=3)
        accelerator = Accelerator(cpu=True)

        result = train_loop(model, lambda epoch: [train_batch], [val_batch], cfg, accelerator)

        reloaded = F0PairMLP(input_dim=4, hidden_dims=(16,), dropout=0.0)
        reloaded.load_state_dict(result.best_state_dict)
        reloaded.eval()
        original = F0PairMLP(input_dim=4, hidden_dims=(16,), dropout=0.0)
        original.load_state_dict(result.best_state_dict)
        original.eval()

        probe = {"x_a": torch.randn(3, 4), "x_b": torch.randn(3, 4)}
        with torch.no_grad():
            logits_reloaded = reloaded(probe)["logits"]
            logits_original = original(probe)["logits"]

        assert torch.equal(logits_reloaded, logits_original)


class _ScriptedEvalModel(nn.Module):
    """Test double: real training dynamics but fully scripted eval-mode logits.

    Lets checkpoint-selection / early-stopping bookkeeping in `train_loop` be
    tested against a controlled sequence of val AUPRC values, independent of
    whatever the toy training data/optimizer actually do.
    """

    def __init__(self, eval_logits_by_call: list[torch.Tensor]) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self._eval_logits = eval_logits_by_call
        self._eval_call_idx = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.linear(batch["x_a"])
        if not self.training:
            logits = self._eval_logits[self._eval_call_idx].clone()
            self._eval_call_idx += 1
        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in batch:
            output["loss"] = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), batch["label"].float()
            )
        return output


class TestTrainLoopCheckpointSelectionAndEarlyStopping:
    def test_best_checkpoint_picks_best_auprc_epoch_and_stops_early(self) -> None:
        # Fixed 4-example val set (2 pos, 2 neg). Epoch 1 logits perfectly separate
        # (best AUPRC = 1.0); epochs 2-4 are actively wrong (AUPRC drops each time).
        val_labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        val_batch = {
            "x_a": torch.randn(4, 4),
            "x_b": torch.randn(4, 4),
            "label": val_labels,
        }
        eval_logits_by_call = [
            torch.tensor([[5.0], [5.0], [-5.0], [-5.0]]),  # epoch1: perfect -> auprc=1.0
            torch.tensor([[-1.0], [-2.0], [1.0], [2.0]]),  # epoch2: reversed -> worse
            torch.tensor([[-2.0], [-3.0], [2.0], [3.0]]),  # epoch3: still bad
            torch.tensor([[-3.0], [-4.0], [3.0], [4.0]]),  # epoch4: still bad -> stop
            torch.tensor([[-4.0], [-5.0], [4.0], [5.0]]),  # would never be reached
        ]
        model = _ScriptedEvalModel(eval_logits_by_call)
        train_batch = _batch_of(_make_synthetic_pair_dataset(16, input_dim=4, seed=3))
        cfg = _tiny_config(epochs=10, patience=2, eval_every=1)
        accelerator = Accelerator(cpu=True)

        result = train_loop(model, lambda epoch: [train_batch], [val_batch], cfg, accelerator)

        assert result.best_epoch == 1
        assert result.best_val_metrics.auprc == pytest.approx(1.0)
        assert result.stopped_early is True
        assert result.last_epoch == 3  # epoch1 best, epoch2 (1 bad), epoch3 (2 bad) -> stop


# ----------------------------------------------------- train_loop (tiny real v3_1 model)


class TestTrainLoopTinyV3_1:
    def test_one_optimizer_step_runs_with_finite_loss(self) -> None:
        torch.manual_seed(0)
        model = _tiny_v3_1_model()
        emb_a = torch.randn(4, 6, 8)
        emb_b = torch.randn(4, 5, 8)
        batch = {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": torch.tensor([6, 6, 6, 6], dtype=torch.int64),
            "len_b": torch.tensor([5, 5, 5, 5], dtype=torch.int64),
            "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        cfg = _tiny_config(epochs=1)
        accelerator = Accelerator(cpu=True)

        result = train_loop(model, lambda epoch: [batch], [batch], cfg, accelerator)

        assert result.history
        train_loss = result.history[0]["train_loss"]
        assert isinstance(train_loss, float)
        assert train_loss == train_loss  # not NaN
        assert train_loss >= 0.0


# --------------------------------------------------------------------------- write_outputs


class TestWriteOutputs:
    def test_writes_expected_artifacts(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "run"
        cfg = _tiny_config(epochs=1)
        cfg = Config(
            model=cfg.model,
            data=cfg.data,
            optim=cfg.optim,
            eval=cfg.eval,
            seed=cfg.seed,
            output_dir=output_dir,
            mixed_precision=cfg.mixed_precision,
        )
        model = F0PairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        metrics = EdgeMetrics(
            auroc=0.9,
            auprc=0.85,
            accuracy=0.8,
            sensitivity=0.7,
            specificity=0.9,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            mcc=0.5,
            ece=0.05,
            brier=0.1,
            threshold=0.5,
            n_pos=10,
            n_neg=10,
        )
        result = TrainResult(
            best_state_dict=state_dict,
            best_epoch=2,
            best_val_metrics=metrics,
            last_state_dict=state_dict,
            last_epoch=3,
            last_val_metrics=metrics,
            history=[
                {"epoch": 1, "train_loss": 0.5, "val_auroc": 0.8, "val_auprc": 0.75},
                {"epoch": 2, "train_loss": 0.4, "val_auroc": 0.9, "val_auprc": 0.85},
            ],
            stopped_early=False,
        )
        assembled_dropped_counts = {"train_edges.txt": 0, "val_edges.txt": 0, "test_edges.txt": 0}

        write_outputs(
            result,
            cfg,
            {"input_dim": 4, "hidden_dims": [8], "dropout": 0.0},
            assembled_dropped_counts,
        )

        assert (output_dir / "best.pt").exists()
        assert (output_dir / "last.pt").exists()
        assert (output_dir / "metrics.jsonl").exists()
        assert (output_dir / "run_metadata.json").exists()

        best_payload = torch.load(output_dir / "best.pt", weights_only=False)
        assert set(best_payload.keys()) == {
            "model_state",
            "model_family",
            "model_config",
            "epoch",
            "val_metrics",
            "seed",
            "config",
        }
        assert best_payload["model_family"] == "f0_mlp"
        assert best_payload["epoch"] == 2

        with (output_dir / "metrics.jsonl").open() as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 2
        assert lines[0]["epoch"] == 1

        run_metadata = json.loads((output_dir / "run_metadata.json").read_text())
        assert "config_hash" in run_metadata
        assert "checkpoint_id" in run_metadata
        assert len(run_metadata["checkpoint_id"]) == 16
        assert "torch_version" in run_metadata
        assert "timestamp" in run_metadata
        assert run_metadata["dropped_pair_counts"] == assembled_dropped_counts
        assert run_metadata["positives_mode"] == "train_plus"


# ------------------------------------------------- assemble_data (feature-coverage gate)


def _write_pairs_file(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v, label in rows:
            f.write(f"{u}\t{v}\t{label}\n")


def _write_edge_list_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v in pairs:
            f.write(f"{u}\t{v}\n")


def _build_synthetic_benchmark(root: Path, strategy: str) -> None:
    """Write a tiny, internally-consistent benchmark package (no verify_benchmark)."""
    graph = nx.Graph()
    graph.add_nodes_from([f"node_{i:06d}" for i in range(1, 8)])
    graph.add_edges_from(
        [
            ("node_000001", "node_000002"),
            ("node_000003", "node_000004"),
            ("node_000005", "node_000006"),
        ]
    )
    with (root / "graph.pkl").open("wb") as f:
        pickle.dump(graph, f)
    _write_edge_list_file(
        root / "positive_edges.txt",
        [("node_000001", "node_000002"), ("node_000003", "node_000004")],
    )

    strategy_dir = root / strategy
    strategy_dir.mkdir(parents=True)
    train_nodes = {f"node_{i:06d}" for i in range(1, 6)}
    test_nodes = {"node_000006", "node_000007"}
    with (strategy_dir / "split.pkl").open("wb") as f:
        pickle.dump({"train": train_nodes, "test": test_nodes}, f)

    _write_pairs_file(
        strategy_dir / "train_edges.txt",
        [
            ("node_000001", "node_000002", 1),
            ("node_000003", "node_000004", 1),
            ("node_000002", "node_000005", 0),
            ("node_000001", "node_000004", 0),
        ],
    )
    _write_pairs_file(
        strategy_dir / "val_edges.txt",
        [("node_000001", "node_000003", 0), ("node_000004", "node_000005", 0)],
    )
    _write_pairs_file(strategy_dir / "test_edges.txt", [("node_000006", "node_000007", 0)])

    train_graph = nx.Graph()
    train_graph.add_nodes_from(train_nodes)
    train_graph.add_edges_from([("node_000001", "node_000002"), ("node_000003", "node_000004")])
    with (strategy_dir / "train_graph.pkl").open("wb") as f:
        pickle.dump(train_graph, f)

    test_graph = nx.Graph()
    test_graph.add_nodes_from(test_nodes)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(test_graph, f)

    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump({}, f)

    with (strategy_dir / "candidate_test_edges.txt").open("w", encoding="utf-8") as f:
        pass  # not read when verify=False


def _write_feature_store(root: Path, node_ids: list[str], *, input_dim: int = 4) -> None:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.randn(3, input_dim, dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": input_dim, "max_sequence_length": 1024}
        )
    )


def _synthetic_data_config(data_root: Path, *, expected_missing_features: list[str]) -> Config:
    from src.train_b0 import DataConfig, EvalConfig, ModelConfig, OptimConfig

    return Config(
        model=ModelConfig(family="f0_mlp", config={}),
        data=DataConfig(
            root=data_root,
            strategy="synthetic",
            train_positives="train_plus",
            negative_ratio=2,
            partition_seed=0,
            token_budget=131072,
            batch_pairs=8,
            num_workers=0,
            f0_cache=data_root / "f0_cache" / "f0_matrix.pt",
            expected_missing_features=expected_missing_features,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, epochs=1, warmup_steps=1, grad_clip=1.0),
        eval=EvalConfig(patience=1, eval_every=1),
        seed=0,
        output_dir=data_root / "outputs",
        mixed_precision="no",
    )


class TestAssembleDataFeatureCoverageGate:
    def test_raises_when_exclude_set_does_not_match_expected(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        benchmark_root.mkdir(parents=True)
        _build_synthetic_benchmark(benchmark_root, "synthetic")
        features_root = data_root / "features" / "frozen_node_features_1024"
        # All graph nodes have features EXCEPT node_000007 -> exclude == {node_000007}
        all_nodes = [f"node_{i:06d}" for i in range(1, 7)]
        _write_feature_store(features_root, all_nodes)

        cfg = _synthetic_data_config(data_root, expected_missing_features=[])

        with pytest.raises(ArtifactVerificationError, match="node_000007"):
            assemble_data(cfg, verify=False)

    def test_passes_when_exclude_set_matches_expected(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        benchmark_root.mkdir(parents=True)
        _build_synthetic_benchmark(benchmark_root, "synthetic")
        features_root = data_root / "features" / "frozen_node_features_1024"
        all_nodes = [f"node_{i:06d}" for i in range(1, 7)]
        _write_feature_store(features_root, all_nodes)

        cfg = _synthetic_data_config(data_root, expected_missing_features=["node_000007"])

        assembled = assemble_data(cfg, verify=False)

        assert isinstance(assembled, AssembledData)
        assert assembled.operative_node_count == 6
        # node_000007 is test-side only and touches zero train/val pairs here.
        assert assembled.dropped_pair_counts["train_edges.txt"] == 0
        assert assembled.dropped_pair_counts["val_edges.txt"] == 0
        assert len(assembled.training_positives) == 2  # the two train+ positives
        assert assembled.degrees  # built from G_struct, non-empty

    def test_e_sup_mode_uses_supervision_split_only(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        benchmark_root.mkdir(parents=True)
        _build_synthetic_benchmark(benchmark_root, "synthetic")
        features_root = data_root / "features" / "frozen_node_features_1024"
        all_nodes = [f"node_{i:06d}" for i in range(1, 7)]
        _write_feature_store(features_root, all_nodes)

        cfg = _synthetic_data_config(data_root, expected_missing_features=["node_000007"])
        from dataclasses import replace

        cfg = replace(cfg, data=replace(cfg.data, train_positives="e_sup"))

        assembled = assemble_data(cfg, verify=False)

        # e_sup is a strict subset of the 2 train+ positives (80/20 split of n=2 -> 1/1).
        assert len(assembled.training_positives) <= 2


class TestV31TrainingLoader:
    def test_each_epoch_only_reorders_fixed_train_file_pairs(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        benchmark_root.mkdir(parents=True)
        _build_synthetic_benchmark(benchmark_root, "synthetic")
        features_root = data_root / "features" / "frozen_node_features_1024"
        all_nodes = [f"node_{i:06d}" for i in range(1, 7)]
        _write_feature_store(features_root, all_nodes)
        cfg = _synthetic_data_config(data_root, expected_missing_features=["node_000007"])
        cfg = replace(cfg, model=replace(cfg.model, family="v3_1"))
        assembled = assemble_data(cfg, verify=False)

        factory, _ = _build_v3_1_loaders(cfg, assembled)
        expected = Counter(
            zip(
                assembled.benchmark.split.train_pairs.pairs,
                assembled.benchmark.split.train_pairs.labels.tolist(),
                strict=True,
            )
        )

        for epoch in (0, 1):
            loader = factory(epoch)
            assert isinstance(loader, DataLoader)
            dataset = loader.dataset
            assert isinstance(dataset, TokenPairDataset)
            assert dataset._labels is not None
            observed = Counter(zip(dataset._pairs, dataset._labels, strict=True))
            assert observed == expected
            assert Counter(dataset._labels) == {0: 2, 1: 2}


def _synthetic_v31_pack_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Config, AssembledData, Path]:
    """Build a synthetic benchmark + feature store + packed feature table."""
    data_root = tmp_path / "data"
    benchmark_root = data_root / "benchmark_2025_neurips"
    benchmark_root.mkdir(parents=True)
    _build_synthetic_benchmark(benchmark_root, "synthetic")
    source_root = data_root / "features" / "frozen_node_features_1024"
    _write_feature_store(source_root, [f"node_{index:06d}" for index in range(1, 7)])
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            root=data_root,
            strategy="synthetic",
            expected_missing_features=["node_000007"],
        ),
        runtime=replace(
            cfg.runtime,
            pack_workers=1,
            loader_workers_per_rank=1,
        ),
        output_dir=tmp_path / "outputs",
    )
    assembled = assemble_data(cfg, verify=False)
    pack_root = tmp_path / "pack"
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)
    build_packed_features(source_root, pack_root, workers=1)
    return cfg, assembled, pack_root


def test_sample_warmup_preserves_pair_exposure() -> None:
    assert compute_sample_warmup_steps([10, 10], [25], baseline_steps=5) == 2


def test_packed_loader_does_not_call_feature_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, assembled, pack_root = _synthetic_v31_pack_fixture(tmp_path, monkeypatch)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    monkeypatch.setattr(
        assembled.store,
        "load_tokens",
        Mock(side_effect=AssertionError("unexpected feature I/O")),
    )

    factory, val_loader, warmup_steps = _build_packed_v3_1_loaders(
        cfg,
        assembled,
        table,
        token_budget_per_rank=1024,
        process_index=0,
        world_size=1,
    )

    batch = next(iter(factory(1)))
    assert set(batch) >= {"emb_a", "emb_b", "len_a", "len_b", "label"}
    assert next(iter(val_loader))["_row_id"].numel() > 0
    assert warmup_steps >= 1


def test_packed_loader_multi_worker_ranks_cover_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """world_size=2 takes the production multi-worker persistent DataLoader branch."""
    cfg, assembled, pack_root = _synthetic_v31_pack_fixture(tmp_path, monkeypatch)
    assert cfg.runtime is not None
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    # All synthetic features are length 3 -> every pair lands in bucket 128, and the
    # train (4 rows) and val (2 rows) buckets both satisfy the >= world_size planner gate.
    expected_row_ids = list(range(len(assembled.benchmark.split.train_pairs.pairs)))
    seen_row_ids: list[int] = []
    for process_index in (0, 1):
        multi_worker_factory, _, _ = _build_packed_v3_1_loaders(
            cfg,
            assembled,
            table,
            token_budget_per_rank=1024,
            process_index=process_index,
            world_size=2,
        )
        multi_worker_iterable = multi_worker_factory(1)
        assert isinstance(multi_worker_iterable, _EpochGpuBatchIterable)
        loader = multi_worker_iterable._source
        assert isinstance(loader, DataLoader)
        # Guard against silently regressing into the single-process unit-test branch.
        assert loader.num_workers == cfg.runtime.loader_workers_per_rank
        assert loader.num_workers > 0
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == cfg.runtime.prefetch_factor
        assert loader.batch_size is None

        dataset = loader.dataset
        assert isinstance(dataset, CompactPairBatchDataset)
        multi_worker_iterable._sampler.set_epoch(1)
        for batch_index in loader.sampler:
            compact_batch = dataset[int(batch_index)]
            seen_row_ids.extend(int(row_id) for row_id in compact_batch.row_ids)

    # Exact coverage across the two rank partitions: no missing rows, no duplicates.
    assert sorted(seen_row_ids) == expected_row_ids


def test_packed_loader_reuses_one_dataloader_across_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, assembled, pack_root = _synthetic_v31_pack_fixture(tmp_path, monkeypatch)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    factory, _, _ = _build_packed_v3_1_loaders(
        cfg,
        assembled,
        table,
        token_budget_per_rank=1024,
        process_index=0,
        world_size=1,
    )

    epoch_one = factory(1)
    epoch_two = factory(2)
    assert isinstance(epoch_one, GpuBatchIterable)
    assert isinstance(epoch_two, GpuBatchIterable)
    assert epoch_one._source is epoch_two._source


def test_packed_loader_enables_persistent_workers_and_reuses_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, assembled, pack_root = _synthetic_v31_pack_fixture(tmp_path, monkeypatch)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    factory, _, _ = _build_packed_v3_1_loaders(
        cfg,
        assembled,
        table,
        token_budget_per_rank=1024,
        process_index=0,
        world_size=2,
    )

    epoch_one = factory(1)
    assert isinstance(epoch_one, GpuBatchIterable)
    loader = epoch_one._source
    assert isinstance(loader, DataLoader)
    epoch_two = factory(2)
    assert isinstance(epoch_two, GpuBatchIterable)
    assert epoch_two._source is loader
    assert loader.num_workers == 1
    assert loader.persistent_workers is True


def test_distributed_plan_rejects_duplicate_and_missing_rows() -> None:
    plan = [
        [PairBatchSpec(indices=(0, 0), bucket_boundary=128, global_pair_count=4)],
        [PairBatchSpec(indices=(1, 2), bucket_boundary=128, global_pair_count=4)],
    ]
    with pytest.raises(ValueError, match="duplicate training plan row IDs"):
        _validate_distributed_plan(plan, expected_row_count=4)


def test_probe_batch_cycle_is_lazy_and_traverses_all_batches() -> None:
    yielded: list[tuple[int, str]] = []
    calls: list[int] = []

    def factory(epoch: int) -> Iterable[dict[str, torch.Tensor]]:
        calls.append(epoch)

        def batches() -> Iterator[dict[str, torch.Tensor]]:
            for bucket in ("short", "medium", "long"):
                yielded.append((epoch, bucket))
                yield {"bucket": torch.tensor(len(yielded))}

        return batches()

    iterator = _cycle_assembled_batches(factory)
    next(iterator)
    assert yielded == [(1, "short")]
    next(iterator)
    next(iterator)
    next(iterator)
    assert yielded == [
        (1, "short"),
        (1, "medium"),
        (1, "long"),
        (1, "short"),
    ]
    assert calls == [1, 1]


def test_probe_plan_interleaves_all_nonempty_buckets_in_first_window() -> None:
    first_bucket = [
        PairBatchSpec(indices=(index,), bucket_boundary=128, global_pair_count=1)
        for index in range(41)
    ]
    later_buckets = [
        PairBatchSpec(indices=(41,), bucket_boundary=256, global_pair_count=1),
        PairBatchSpec(indices=(42,), bucket_boundary=512, global_pair_count=1),
    ]

    ordered = _interleave_bucket_specs(first_bucket + later_buckets)

    assert [spec.bucket_boundary for spec in ordered[:3]] == [128, 256, 512]


class _ProbeLossModel(nn.Module):
    def __init__(self, *, finite: bool, clock_reads: list[int] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.finite = finite
        self.clock_reads = clock_reads
        self.forward_clock_values: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del batch
        if self.clock_reads is not None:
            self.forward_clock_values.append(len(self.clock_reads))
        multiplier = torch.tensor(1.0 if self.finite else float("nan"))
        return {"loss": self.weight * multiplier}


def _probe_batch(global_count: int) -> dict[str, torch.Tensor]:
    return {
        "label": torch.zeros(global_count),
        "_local_pair_count": torch.tensor(global_count),
        "_global_pair_count": torch.tensor(global_count),
    }


def test_probe_starts_timer_before_first_timed_forward_and_counts_exact_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, probe_warmup_steps=1, probe_timed_steps=2),
    )
    clock_reads: list[int] = []

    def monotonic() -> float:
        clock_reads.append(len(clock_reads))
        return 10.0 + 2.0 * (len(clock_reads) - 1)

    monkeypatch.setattr("src.train_b0.time.monotonic", monotonic)
    model = _ProbeLossModel(finite=True, clock_reads=clock_reads)
    batches = [_probe_batch(count) for count in (5, 7, 11)]
    output = tmp_path / "probe.json"
    _run_probe_mode(
        model,
        lambda epoch: iter(batches),
        cfg,
        Accelerator(),
        token_budget_per_rank=1024,
        profile_output=output,
    )

    payload = json.loads(output.read_text())
    assert model.forward_clock_values == [0, 1, 1]
    assert payload["global_pairs_per_second"] == pytest.approx((7 + 11) / 2.0)


def test_probe_rejects_nonfinite_loss(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, probe_warmup_steps=0, probe_timed_steps=1),
    )
    output = tmp_path / "probe.json"
    with pytest.raises(RuntimeError, match="non-finite probe loss"):
        _run_probe_mode(
            _ProbeLossModel(finite=False),
            lambda epoch: iter([_probe_batch(2)]),
            cfg,
            Accelerator(),
            token_budget_per_rank=1024,
            profile_output=output,
        )

    assert not output.exists()


def test_probe_emits_candidate_failure_marker_for_nonfinite_loss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, probe_warmup_steps=0, probe_timed_steps=1),
    )

    with pytest.raises(RuntimeError, match="non-finite probe loss"):
        _run_probe_mode(
            _ProbeLossModel(finite=False),
            lambda epoch: iter([_probe_batch(2)]),
            cfg,
            Accelerator(),
            token_budget_per_rank=1024,
            profile_output=tmp_path / "probe.json",
        )

    marker = capsys.readouterr().err.strip()
    prefix = "E2_PROBE_CANDIDATE_FAILURE:"
    assert marker.startswith(prefix)
    assert json.loads(marker.removeprefix(prefix)) == {
        "kind": "nonfinite",
        "message": "non-finite probe loss",
    }


def test_probe_reraises_backward_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, probe_warmup_steps=0, probe_timed_steps=1),
    )
    output = tmp_path / "probe.json"
    accelerator = Accelerator()
    monkeypatch.setattr(
        accelerator,
        "backward",
        Mock(side_effect=RuntimeError("backward collective failed")),
    )

    with pytest.raises(RuntimeError, match="backward collective failed"):
        _run_probe_mode(
            _ProbeLossModel(finite=True),
            lambda epoch: iter([_probe_batch(2)]),
            cfg,
            accelerator,
            token_budget_per_rank=1024,
            profile_output=output,
        )

    assert not output.exists()
    assert "E2_PROBE_CANDIDATE_FAILURE:" not in capsys.readouterr().err


def test_probe_marks_backward_oom_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    assert cfg.runtime is not None
    cfg = replace(
        cfg,
        runtime=replace(cfg.runtime, probe_warmup_steps=0, probe_timed_steps=1),
    )
    accelerator = Accelerator()
    monkeypatch.setattr(
        accelerator,
        "backward",
        Mock(side_effect=torch.OutOfMemoryError("CUDA out of memory in backward")),
    )

    with pytest.raises(torch.OutOfMemoryError, match="CUDA out of memory"):
        _run_probe_mode(
            _ProbeLossModel(finite=True),
            lambda epoch: iter([_probe_batch(2)]),
            cfg,
            accelerator,
            token_budget_per_rank=1024,
            profile_output=tmp_path / "probe.json",
        )

    marker = capsys.readouterr().err.strip()
    prefix = "E2_PROBE_CANDIDATE_FAILURE:"
    assert marker.startswith(prefix)
    payload = json.loads(marker.removeprefix(prefix))
    assert payload["kind"] == "oom"
    assert "CUDA out of memory" in payload["message"]


class _EpochProbeClockAccelerator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.barriers = 0

    def wait_for_everyone(self) -> None:
        self.barriers += 1

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat((tensor, torch.tensor([7.5], dtype=tensor.dtype)))


def test_epoch_probe_barriers_before_training_and_reports_slowest_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _EpochProbeClockAccelerator()
    events: list[str] = []
    clock = iter((10.0, 14.0))
    monkeypatch.setattr("src.train_b0.time.monotonic", lambda: next(clock))

    def train_once() -> str:
        events.append(f"train-after-{accelerator.barriers}")
        return "result"

    result, elapsed = _run_timed_epoch_probe(
        cast(Accelerator, accelerator),
        train_once,
    )

    assert result == "result"
    assert events == ["train-after-1"]
    assert elapsed == pytest.approx(7.5)


# ------------------------------------------------------- fixed-epoch DDP train/eval semantics


def test_ddp_loss_scaling_matches_global_sample_mean() -> None:
    local_mean = torch.tensor(2.0)
    scaled = scale_ddp_mean_loss(local_mean, local_count=3, global_count=10, world_size=4)
    assert scaled.item() == pytest.approx(2.4)


def test_ddp_loss_scaling_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="invalid DDP loss-scaling counts"):
        scale_ddp_mean_loss(torch.tensor(1.0), local_count=0, global_count=4, world_size=2)
    with pytest.raises(ValueError, match="invalid DDP loss-scaling counts"):
        scale_ddp_mean_loss(torch.tensor(1.0), local_count=5, global_count=4, world_size=2)


def test_build_ddp_accelerator_pins_ddp_kwargs() -> None:
    accelerator = build_ddp_accelerator("no")
    handler = accelerator.ddp_handler
    assert handler is not None
    assert handler.broadcast_buffers is False
    assert handler.find_unused_parameters is False
    assert handler.gradient_as_bucket_view is True


def test_validation_rejects_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="duplicate validation row IDs"):
        validate_gathered_validation(
            row_ids=np.array([0, 0, 1]),
            labels=np.array([0, 0, 1]),
            logits=np.array([-1.0, -1.0, 1.0]),
            expected_row_ids=np.array([0, 1]),
        )


def test_validation_rejects_missing_rows() -> None:
    with pytest.raises(ValueError, match="do not cover the fixed validation set"):
        validate_gathered_validation(
            row_ids=np.array([0, 2]),
            labels=np.array([1, 0]),
            logits=np.array([1.0, -1.0]),
            expected_row_ids=np.array([0, 1, 2]),
        )


def test_validation_returns_row_sorted_labels_and_logits() -> None:
    labels, logits = validate_gathered_validation(
        row_ids=np.array([2, 0, 1]),
        labels=np.array([0, 1, 1]),
        logits=np.array([0.5, -0.5, 2.0]),
        expected_row_ids=np.array([0, 1, 2]),
    )
    assert labels.tolist() == [1, 1, 0]
    assert logits.tolist() == pytest.approx([-0.5, 2.0, 0.5])


class _StubSumReduceAccelerator:
    """Simulate accelerate's SUM-only cross-process reduce for a 4-rank world.

    ``accelerate.utils.operations._reduce_across_processes`` unconditionally
    all-reduces with ``ReduceOp.SUM`` and only special-cases ``reduction ==
    "mean"`` — any other reduction string silently returns the sum. This stub
    reproduces exactly that: it adds the simulated other ranks' flag values.
    """

    device = torch.device("cpu")

    def __init__(self, other_rank_flags: list[float]) -> None:
        self._other_rank_flags = other_rank_flags

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        # The guard must rely only on SUM semantics; anything else degrades to
        # SUM inside accelerate and must not be requested.
        assert reduction == "sum"
        return tensor + sum(self._other_rank_flags)


def test_finite_guard_fails_when_any_rank_nonfinite() -> None:
    # Local loss is finite, but one simulated peer rank reports non-finite.
    # Under the broken reduction="min" guard (which accelerate silently turns
    # into a SUM of finite-flags), this scenario passed and backward spread the
    # corrupt gradient to every rank.
    accelerator = cast(Accelerator, _StubSumReduceAccelerator([1.0, 0.0, 0.0]))
    assert _all_ranks_loss_finite(torch.tensor(0.5), accelerator) is False


def test_finite_guard_passes_when_all_ranks_finite() -> None:
    accelerator = cast(Accelerator, _StubSumReduceAccelerator([0.0, 0.0, 0.0]))
    assert _all_ranks_loss_finite(torch.tensor(0.5), accelerator) is True


def test_finite_guard_fails_on_local_nonfinite_single_process() -> None:
    accelerator = Accelerator()
    assert _all_ranks_loss_finite(torch.tensor(float("nan")), accelerator) is False
    assert _all_ranks_loss_finite(torch.tensor(1.0), accelerator) is True


class _NonMainGatherAccelerator:
    """Single-process stand-in for a non-zero rank in distributed validation.

    ``gather``/``pad_across_processes`` are identities (gather already returns
    the full tensors on every rank in real DDP); ``is_main_process`` is False so
    any rank-zero-only code path is skipped.
    """

    device = torch.device("cpu")
    is_main_process = False
    num_processes = 2
    process_index = 1

    def pad_across_processes(
        self, tensor: torch.Tensor, dim: int = 0, pad_index: int = 0
    ) -> torch.Tensor:
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_evaluate_distributed_coverage_failure_raises_on_non_main_rank() -> None:
    # A duplicate validation row must raise the coverage ValueError on EVERY
    # rank symmetrically. If only rank zero validated, the other ranks would
    # block in broadcast_object_list until the NCCL watchdog killed the job.
    model = F0PairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
    batch = _batch_of(_make_synthetic_pair_dataset(4))
    batch["_row_id"] = torch.tensor([0, 0, 1, 2])  # row 0 gathered twice

    accelerator = cast(Accelerator, _NonMainGatherAccelerator())
    with pytest.raises(ValueError, match="duplicate validation row IDs"):
        _evaluate_distributed(
            model,
            [batch],
            accelerator,
            expected_row_ids=np.arange(3, dtype=np.int64),
        )


def test_ddp_loop_records_counterfactual_stop_but_runs_all_epochs(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"optim.epochs": 4, "eval.patience": 1})
    cfg = load_config(config_path)
    model = F0PairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
    batch = _batch_of(_make_synthetic_pair_dataset(8))
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    batch["_row_id"] = torch.arange(8)

    def factory(epoch: int) -> list[dict[str, torch.Tensor]]:
        assert 1 <= epoch <= 4
        return [batch]

    metrics = EdgeMetrics(
        auroc=0.5,
        auprc=0.5,
        accuracy=0.5,
        sensitivity=0.5,
        specificity=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        mcc=0.0,
        ece=0.0,
        brier=0.25,
        threshold=0.5,
        n_pos=4,
        n_neg=4,
    )
    result = train_ddp_loop(
        model,
        factory,
        [batch],
        cfg,
        Accelerator(),
        warmup_steps=1,
        evaluate_fn=lambda model, loader, accelerator: metrics,
    )
    assert result.last_epoch == 4
    assert result.stopped_early is False
    assert result.counterfactual_stop_epoch == 2
    # Count epochs that actually executed (one history entry and one per-epoch
    # profile per epoch), not the returned last_epoch constant: a hypothetical
    # early break under fired patience must fail these.
    assert len(result.history) == 4
    assert [entry["epoch"] for entry in result.history] == [1, 2, 3, 4]
    assert len(cast(list[object], result.runtime_profile["per_epoch"])) == 4


class _BatchValueLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"loss": self.weight * 0.0 + batch["loss_value"].mean()}


def _constant_metrics() -> EdgeMetrics:
    return EdgeMetrics(
        auroc=0.5,
        auprc=0.5,
        accuracy=0.5,
        sensitivity=0.5,
        specificity=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        mcc=0.0,
        ece=0.0,
        brier=0.25,
        threshold=0.5,
        n_pos=2,
        n_neg=2,
    )


def _loss_batch(value: float, row_ids: list[int]) -> dict[str, torch.Tensor]:
    count = len(row_ids)
    return {
        "loss_value": torch.full((count,), value),
        "label": torch.zeros(count),
        "_row_id": torch.tensor(row_ids),
        "_local_pair_count": torch.tensor(count),
        "_global_pair_count": torch.tensor(count),
    }


def test_ddp_loop_reports_global_sample_weighted_train_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"optim.epochs": 1})
    cfg = load_config(config_path)
    batches = [_loss_batch(1.0, [0]), _loss_batch(3.0, [1, 2, 3])]

    result = train_ddp_loop(
        _BatchValueLossModel(),
        lambda epoch: batches,
        batches,
        cfg,
        Accelerator(),
        warmup_steps=1,
        evaluate_fn=lambda model, loader, accelerator: _constant_metrics(),
    )

    assert result.history[0]["train_loss"] == pytest.approx(2.5)


def test_ddp_loop_rejects_duplicate_and_missing_training_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"optim.epochs": 1})
    cfg = load_config(config_path)
    batch = _loss_batch(1.0, [0, 0, 2, 3])

    with pytest.raises(ValueError, match="duplicate training row IDs"):
        train_ddp_loop(
            _BatchValueLossModel(),
            lambda epoch: [batch],
            [batch],
            cfg,
            Accelerator(),
            warmup_steps=1,
            evaluate_fn=lambda model, loader, accelerator: _constant_metrics(),
        )


def test_ddp_loop_runtime_profile_has_task12_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"optim.epochs": 2, "eval.patience": 8})
    cfg = load_config(config_path)
    model = F0PairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
    batch = _batch_of(_make_synthetic_pair_dataset(8))
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    batch["_row_id"] = torch.arange(8)

    metrics = EdgeMetrics(
        auroc=0.5,
        auprc=0.5,
        accuracy=0.5,
        sensitivity=0.5,
        specificity=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        mcc=0.0,
        ece=0.0,
        brier=0.25,
        threshold=0.5,
        n_pos=4,
        n_neg=4,
    )
    result = train_ddp_loop(
        model,
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(),
        warmup_steps=1,
        evaluate_fn=lambda model, loader, accelerator: metrics,
    )
    profile = result.runtime_profile
    assert set(profile) >= {
        "epochs_completed",
        "validations_completed",
        "peak_memory_gib_per_rank",
        "steady_state_data_wait_fraction",
        "training_coverage_exact",
        "validation_coverage_exact",
        "feature_cache_hit_rate",
        "per_epoch",
        "counterfactual_stop_epoch",
        "per_rank",
        "global_pairs_per_second",
        "global_tokens",
        "global_tokens_per_second",
        "validation_seconds",
    }
    assert profile["epochs_completed"] == 2
    assert profile["validations_completed"] == 2
    assert len(cast(list[object], profile["per_epoch"])) == 2


# --------------------------------------------------------------------------- real-data integration


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataV31Assembly:
    def test_assemble_breadth_first_and_forward_one_batch(
        self, benchmark_root: Path, features_root: Path
    ) -> None:
        from src.data.pairs import TokenPairDataset, collate_token_pairs, probe_lengths
        from src.train_b0 import DataConfig, EvalConfig, ModelConfig, OptimConfig

        data_root = benchmark_root.parent
        cfg = Config(
            model=ModelConfig(family="v3_1", config={}),
            data=DataConfig(
                root=data_root,
                strategy="breadth_first",
                train_positives="train_plus",
                negative_ratio=5,
                partition_seed=0,
                token_budget=131072,
                batch_pairs=1024,
                num_workers=0,
                f0_cache=data_root / "features" / "f0_cache_unused.pt",
                expected_missing_features=["node_004764", "node_007050"],
            ),
            optim=OptimConfig(
                lr=1.0e-4, weight_decay=0.01, epochs=30, warmup_steps=500, grad_clip=1.0
            ),
            eval=EvalConfig(patience=8, eval_every=1),
            seed=47,
            output_dir=data_root / "outputs_unused",
            mixed_precision="no",
        )

        assembled = assemble_data(cfg)

        # Feature-coverage gate numbers pinned in the brief (measured on the real package).
        assert assembled.exclude_nodes == frozenset({"node_004764", "node_007050"})
        assert assembled.operative_node_count == 10_088
        # Known fact (task-1 review): these two nodes touch ZERO breadth_first pairs.
        assert assembled.dropped_pair_counts["train_edges.txt"] == 0
        assert assembled.dropped_pair_counts["val_edges.txt"] == 0
        assert len(assembled.training_positives) == 42_880
        assert assembled.degrees  # G_struct degrees over all train nodes

        # ONE batch (a handful of nodes only) through BEST_V3_1_CONFIG on CPU. No training.
        pairs = assembled.training_positives[:2]
        lengths = probe_lengths(assembled.store, pairs)
        dataset = TokenPairDataset(pairs, [1, 1], assembled.store, lengths=lengths)
        batch = collate_token_pairs([dataset[0], dataset[1]])

        assert set(batch.keys()) == {"emb_a", "emb_b", "len_a", "len_b", "label"}

        model = V3_1(**BEST_V3_1_CONFIG)
        model.eval()
        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (2, 1)
        assert torch.isfinite(output["logits"]).all()
