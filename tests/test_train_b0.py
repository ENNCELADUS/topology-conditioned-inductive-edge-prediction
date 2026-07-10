"""Tests for src.train_b0: config schema, model builder, train loop, output writer."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
import src.train_b0 as train_b0_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from src.data.artifacts import ArtifactVerificationError
from src.eval.edge_metrics import EdgeMetrics
from src.model.B0 import BEST_V3_1_CONFIG, V3_1
from src.model.b0_alt import F0PairMLP
from src.train_b0 import (
    AssembledData,
    Config,
    TrainResult,
    _to_device,
    _v3_loader_options,
    apply_overrides,
    assemble_data,
    build_model,
    config_to_dict,
    load_config,
    parse_args,
    resolve_model_kwargs,
    train_loop,
    write_outputs,
)

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

    def test_shipped_v3_1_config_uses_four_workers(self) -> None:
        cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))

        assert cfg.data.num_workers == 4

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


class TestV31LoaderContract:
    def test_worker_options_enable_pinned_prefetched_loading(self) -> None:
        assert _v3_loader_options(4) == {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        assert _v3_loader_options(0) == {
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
        }

    def test_to_device_requests_non_blocking_copy(self) -> None:
        tensor = Mock(spec=torch.Tensor)
        device = torch.device("cpu")

        moved = _to_device({"value": cast(torch.Tensor, tensor)}, device)

        tensor.to.assert_called_once_with(device, non_blocking=True)
        assert moved["value"] is tensor.to.return_value


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


class TestV31LoaderConstruction:
    def test_preloads_before_both_loaders_and_wires_worker_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        data_root = tmp_path / "data"
        benchmark_root = data_root / "benchmark_2025_neurips"
        benchmark_root.mkdir(parents=True)
        _build_synthetic_benchmark(benchmark_root, "synthetic")
        features_root = data_root / "features" / "frozen_node_features_1024"
        _write_feature_store(features_root, [f"node_{i:06d}" for i in range(1, 7)])
        cfg = _synthetic_data_config(data_root, expected_missing_features=["node_000007"])
        cfg = replace(cfg, data=replace(cfg.data, num_workers=4))
        assembled = assemble_data(cfg, verify=False)
        events: list[tuple[object, ...]] = []
        loader_kwargs: list[dict[str, object]] = []
        real_preload = assembled.store.preload

        def recording_preload(node_ids: Iterable[str] | None = None) -> int:
            assert node_ids is not None
            events.append(("preload", tuple(node_ids)))
            return real_preload(node_ids)

        def recording_data_loader(*args: object, **kwargs: object) -> list[object]:
            events.append(("loader",))
            loader_kwargs.append(kwargs)
            return []

        monkeypatch.setattr(assembled.store, "preload", recording_preload)
        monkeypatch.setattr(train_b0_module, "DataLoader", recording_data_loader)

        factory, _ = train_b0_module._build_v3_1_loaders(cfg, assembled)
        factory(1)

        assert events[0] == ("preload", tuple(assembled.operative_node_ids))
        assert events[1:] == [("loader",), ("loader",)]
        expected_options = {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        assert [{key: kwargs[key] for key in expected_options} for kwargs in loader_kwargs] == [
            expected_options,
            expected_options,
        ]


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
