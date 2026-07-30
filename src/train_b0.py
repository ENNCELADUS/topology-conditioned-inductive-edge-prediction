"""Training CLI for the frozen B0 pairwise scorers (families ``v3_1`` and ``f0_mlp``).

Usage::

    python -m src.train_b0 --config configs/b0_v31_breadth_first.yaml \
        [--seed N] [--output-dir DIR] [--max-steps N]

``--max-steps`` is a DEBUG-ONLY flag: it stops training after N optimizer steps so a
bounded local smoke run terminates cleanly. Never use it for real training runs.

The pure pieces stay importable for tests — :func:`load_config`, :func:`build_model`,
:func:`assemble_data`, :func:`train_loop`, :func:`write_outputs` — and :func:`main`
wires the real benchmark/feature data into them.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import pickle
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
import yaml  # type: ignore[import-untyped,unused-ignore]
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader, Sampler

from src.data.artifacts import ArtifactVerificationError, Benchmark, load_benchmark
from src.data.distributed_pairs import (
    CompactPairBatch,
    CompactPairBatchDataset,
    PairBatchSpec,
    build_distributed_epoch_plan,
    identity_compact_batch,
)
from src.data.features import FeatureStore, build_f0_matrix
from src.data.packed_features import PackedFeatureTable
from src.data.pairs import (
    LengthBucketedBatchSampler,
    NegativeSampler,
    TokenPairDataset,
    collate_token_pairs,
)
from src.data.partition import build_g_struct, derive_partition
from src.e2_pipeline import ProbeResult
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.model.B0 import BEST_V3_1_CONFIG, V3_1
from src.model.b0_alt import F0PairMLP

logger = logging.getLogger(__name__)

MODEL_FAMILIES = ("v3_1", "f0_mlp")
TRAIN_POSITIVES_MODES = ("train_plus", "e_sup")
MIXED_PRECISION_MODES = ("no", "bf16")

Batch = dict[str, torch.Tensor]
LoaderFactory = Callable[[int], Iterable[Batch]]
PackedLoaderFactory = Callable[[int], Iterable[Batch]]
OnEval = Callable[[dict[str, float], bool, EdgeMetrics], None]
EvaluateFn = Callable[[nn.Module, Iterable[Batch], Accelerator], EdgeMetrics]
DDP_MODES = ("probe", "epoch-probe", "train")
T = TypeVar("T")


# --------------------------------------------------------------------------- config schema


@dataclass(frozen=True)
class ModelConfig:
    """The ``model:`` config section.

    Attributes:
        family: Scorer family, one of ``v3_1`` or ``f0_mlp``.
        config: Model constructor kwargs. Empty for ``v3_1`` means
            :data:`~src.model.B0.BEST_V3_1_CONFIG`; empty for ``f0_mlp`` means the
            :class:`~src.model.b0_alt.F0PairMLP` defaults.
    """

    family: str
    config: dict[str, object]


@dataclass(frozen=True)
class DataConfig:
    """The ``data:`` config section.

    Attributes:
        root: Data root containing ``benchmark_2025_neurips/`` and
            ``features/frozen_node_features_1024/``.
        strategy: Split strategy name (e.g. ``breadth_first``).
        train_positives: ``train_plus`` (all train-side positives) or ``e_sup``
            (supervision split of the seeded message/supervision partition).
        negative_ratio: Negatives sampled per positive, per epoch on the F0-MLP path.
        partition_seed: Seed for the message/supervision partition.
        token_budget: Per-batch token budget on the ``v3_1`` path.
        batch_pairs: Per-batch pair count on the ``f0_mlp`` path.
        num_workers: DataLoader workers on the ``v3_1`` path.
        f0_cache: Cache path for the F0 mean-pooled feature matrix.
        expected_missing_features: Exact set of graph nodes expected to lack
            features (the feature-coverage gate fails on any drift).
    """

    root: Path
    strategy: str
    train_positives: str
    negative_ratio: int
    partition_seed: int
    token_budget: int
    batch_pairs: int
    num_workers: int
    f0_cache: Path
    expected_missing_features: list[str]


@dataclass(frozen=True)
class OptimConfig:
    """The ``optim:`` config section.

    Attributes:
        lr: AdamW learning rate (post-warmup constant).
        weight_decay: AdamW weight decay.
        epochs: Maximum number of epochs.
        warmup_steps: Linear LR warmup steps (then constant).
        grad_clip: Gradient-norm clip value; 0 disables clipping.
    """

    lr: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    grad_clip: float


@dataclass(frozen=True)
class EvalConfig:
    """The ``eval:`` config section.

    Attributes:
        patience: Early stop after this many evals without val-AUPRC improvement.
        eval_every: Evaluate every N epochs.
    """

    patience: int
    eval_every: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and wall-clock budget contract for formal distributed training.

    ``world_size == 0`` means that the orchestrator must use every visible H20.
    Positive values remain supported for explicit reproducibility checks.
    """

    world_size: int
    pack_dir: Path
    pack_workers: int
    loader_workers_per_rank: int
    prefetch_factor: int
    token_budget: int
    max_pairs_per_rank: int
    memory_limit_gib: float
    total_budget_seconds: int
    pack_budget_seconds: int
    setup_probe_budget_seconds: int
    train_eval_budget_seconds: int
    artifact_budget_seconds: int
    reserve_seconds: int
    probe_warmup_steps: int
    probe_timed_steps: int


@dataclass(frozen=True)
class Config:
    """The full validated training configuration.

    Attributes:
        model: Model family and constructor kwargs.
        data: Data assembly settings.
        optim: Optimizer settings.
        eval: Evaluation/early-stopping settings.
        seed: Global seed.
        output_dir: Directory for checkpoints, metrics, and run metadata.
        mixed_precision: ``"no"`` or ``"bf16"``.
    """

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    eval: EvalConfig
    seed: int
    output_dir: Path
    mixed_precision: str
    runtime: RuntimeConfig | None = None


@dataclass(frozen=True)
class CliArgs:
    """Parsed command-line arguments.

    Attributes:
        config: Path to the YAML config file.
        seed: Optional seed override (wins over the config).
        output_dir: Optional output-dir override (wins over the config).
        max_steps: DEBUG ONLY — stop after this many optimizer steps.
        ddp_mode: Internal multi-H20 worker mode (``probe``/``epoch-probe``/``train``)
            launched by ``accelerate launch``; ``None`` selects the legacy
            single-process path. Requires ``pack_dir``, ``token_budget_per_rank``,
            and ``profile_output`` when set.
        pack_dir: Packed-feature directory the DDP worker loads onto its device.
        token_budget_per_rank: Per-rank token budget for the distributed batch plan.
        profile_output: Path the rank-zero worker writes its JSON profile artifact to.
    """

    config: Path
    seed: int | None
    output_dir: Path | None
    max_steps: int | None
    ddp_mode: str | None = None
    pack_dir: Path | None = None
    token_budget_per_rank: int | None = None
    profile_output: Path | None = None


def _require(mapping: dict[str, object], key: str, context: str) -> object:
    """Return ``mapping[key]`` or raise a ValueError naming the missing key."""
    if key not in mapping:
        raise ValueError(f"config is missing required key '{context}{key}'")
    return mapping[key]


def _as_mapping(value: object, name: str) -> dict[str, object]:
    """Validate that a config value is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"config key '{name}' must be a mapping, got {type(value).__name__}")
    return cast(dict[str, object], value)


def _as_int(value: object, name: str) -> int:
    """Validate that a config value is an integer (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config key '{name}' must be an int, got {value!r}")
    return value


def _as_float(value: object, name: str) -> float:
    """Validate that a config value is a float (ints accepted, bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config key '{name}' must be a float, got {value!r}")
    return float(value)


def _as_str(value: object, name: str) -> str:
    """Validate that a config value is a string."""
    if not isinstance(value, str):
        raise ValueError(f"config key '{name}' must be a string, got {value!r}")
    return value


def _as_str_list(value: object, name: str) -> list[str]:
    """Validate that a config value is a list of strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"config key '{name}' must be a list of strings, got {value!r}")
    return cast(list[str], value)


def _check_no_unknown_keys(
    mapping: dict[str, object], allowed: Sequence[str], context: str
) -> None:
    """Reject config keys outside the pinned schema (catches typos loudly)."""
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"unknown config keys in '{context}': {unknown}")


def load_config(path: Path) -> Config:
    """Load and validate a training config from a YAML file.

    The schema is pinned (see the module docstring of this file and the shipped
    configs under ``configs/``). Unknown keys, missing keys, unknown model
    families, and out-of-range enum values all fail with a message naming the
    offending key.

    Args:
        path: Path to the YAML config file.

    Returns:
        The validated `Config`.

    Raises:
        ValueError: If the config violates the pinned schema.
    """
    raw_obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = _as_mapping(raw_obj, "<top level>")
    _check_no_unknown_keys(
        raw,
        ("model", "data", "optim", "eval", "seed", "output_dir", "mixed_precision", "runtime"),
        "<top level>",
    )

    model_raw = _as_mapping(_require(raw, "model", ""), "model")
    _check_no_unknown_keys(model_raw, ("family", "config"), "model")
    family = _as_str(_require(model_raw, "family", "model."), "model.family")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"model.family must be one of {list(MODEL_FAMILIES)}, got '{family}'")
    model_kwargs_raw = model_raw.get("config") or {}
    model_kwargs = _as_mapping(model_kwargs_raw, "model.config")
    model = ModelConfig(family=family, config=dict(model_kwargs))

    data_raw = _as_mapping(_require(raw, "data", ""), "data")
    data_keys = (
        "root",
        "strategy",
        "train_positives",
        "negative_ratio",
        "partition_seed",
        "token_budget",
        "batch_pairs",
        "num_workers",
        "f0_cache",
        "expected_missing_features",
    )
    _check_no_unknown_keys(data_raw, data_keys, "data")
    train_positives = _as_str(
        _require(data_raw, "train_positives", "data."), "data.train_positives"
    )
    if train_positives not in TRAIN_POSITIVES_MODES:
        raise ValueError(
            f"data.train_positives must be one of {list(TRAIN_POSITIVES_MODES)}, "
            f"got '{train_positives}'"
        )
    data = DataConfig(
        root=Path(_as_str(_require(data_raw, "root", "data."), "data.root")),
        strategy=_as_str(_require(data_raw, "strategy", "data."), "data.strategy"),
        train_positives=train_positives,
        negative_ratio=_as_int(
            _require(data_raw, "negative_ratio", "data."), "data.negative_ratio"
        ),
        partition_seed=_as_int(
            _require(data_raw, "partition_seed", "data."), "data.partition_seed"
        ),
        token_budget=_as_int(_require(data_raw, "token_budget", "data."), "data.token_budget"),
        batch_pairs=_as_int(_require(data_raw, "batch_pairs", "data."), "data.batch_pairs"),
        num_workers=_as_int(_require(data_raw, "num_workers", "data."), "data.num_workers"),
        f0_cache=Path(_as_str(_require(data_raw, "f0_cache", "data."), "data.f0_cache")),
        expected_missing_features=_as_str_list(
            _require(data_raw, "expected_missing_features", "data."),
            "data.expected_missing_features",
        ),
    )

    optim_raw = _as_mapping(_require(raw, "optim", ""), "optim")
    _check_no_unknown_keys(
        optim_raw, ("lr", "weight_decay", "epochs", "warmup_steps", "grad_clip"), "optim"
    )
    optim = OptimConfig(
        lr=_as_float(_require(optim_raw, "lr", "optim."), "optim.lr"),
        weight_decay=_as_float(_require(optim_raw, "weight_decay", "optim."), "optim.weight_decay"),
        epochs=_as_int(_require(optim_raw, "epochs", "optim."), "optim.epochs"),
        warmup_steps=_as_int(_require(optim_raw, "warmup_steps", "optim."), "optim.warmup_steps"),
        grad_clip=_as_float(_require(optim_raw, "grad_clip", "optim."), "optim.grad_clip"),
    )
    if optim.epochs < 1:
        raise ValueError(f"optim.epochs must be >= 1, got {optim.epochs}")

    eval_raw = _as_mapping(_require(raw, "eval", ""), "eval")
    _check_no_unknown_keys(eval_raw, ("patience", "eval_every"), "eval")
    eval_cfg = EvalConfig(
        patience=_as_int(_require(eval_raw, "patience", "eval."), "eval.patience"),
        eval_every=_as_int(_require(eval_raw, "eval_every", "eval."), "eval.eval_every"),
    )

    mixed_precision_raw = _require(raw, "mixed_precision", "")
    # YAML 1.1 parses an unquoted `no` as boolean False; map it back.
    if mixed_precision_raw is False:
        mixed_precision = "no"
    else:
        mixed_precision = _as_str(mixed_precision_raw, "mixed_precision")
    if mixed_precision not in MIXED_PRECISION_MODES:
        raise ValueError(
            f"mixed_precision must be one of {list(MIXED_PRECISION_MODES)}, got '{mixed_precision}'"
        )

    runtime = None
    if "runtime" in raw:
        runtime_raw = _as_mapping(raw["runtime"], "runtime")
        runtime_keys = (
            "world_size",
            "pack_dir",
            "pack_workers",
            "loader_workers_per_rank",
            "prefetch_factor",
            "token_budget",
            "max_pairs_per_rank",
            "memory_limit_gib",
            "total_budget_seconds",
            "pack_budget_seconds",
            "setup_probe_budget_seconds",
            "train_eval_budget_seconds",
            "artifact_budget_seconds",
            "reserve_seconds",
            "probe_warmup_steps",
            "probe_timed_steps",
        )
        _check_no_unknown_keys(runtime_raw, runtime_keys, "runtime")
        world_size_raw = _require(runtime_raw, "world_size", "runtime.")
        runtime = RuntimeConfig(
            world_size=(
                0 if world_size_raw == "auto" else _as_int(world_size_raw, "runtime.world_size")
            ),
            pack_dir=Path(
                _as_str(_require(runtime_raw, "pack_dir", "runtime."), "runtime.pack_dir")
            ),
            pack_workers=_as_int(
                _require(runtime_raw, "pack_workers", "runtime."), "runtime.pack_workers"
            ),
            loader_workers_per_rank=_as_int(
                _require(runtime_raw, "loader_workers_per_rank", "runtime."),
                "runtime.loader_workers_per_rank",
            ),
            prefetch_factor=_as_int(
                _require(runtime_raw, "prefetch_factor", "runtime."),
                "runtime.prefetch_factor",
            ),
            token_budget=_as_int(
                _require(runtime_raw, "token_budget", "runtime."),
                "runtime.token_budget",
            ),
            max_pairs_per_rank=_as_int(
                _require(runtime_raw, "max_pairs_per_rank", "runtime."),
                "runtime.max_pairs_per_rank",
            ),
            memory_limit_gib=_as_float(
                _require(runtime_raw, "memory_limit_gib", "runtime."),
                "runtime.memory_limit_gib",
            ),
            total_budget_seconds=_as_int(
                _require(runtime_raw, "total_budget_seconds", "runtime."),
                "runtime.total_budget_seconds",
            ),
            pack_budget_seconds=_as_int(
                _require(runtime_raw, "pack_budget_seconds", "runtime."),
                "runtime.pack_budget_seconds",
            ),
            setup_probe_budget_seconds=_as_int(
                _require(runtime_raw, "setup_probe_budget_seconds", "runtime."),
                "runtime.setup_probe_budget_seconds",
            ),
            train_eval_budget_seconds=_as_int(
                _require(runtime_raw, "train_eval_budget_seconds", "runtime."),
                "runtime.train_eval_budget_seconds",
            ),
            artifact_budget_seconds=_as_int(
                _require(runtime_raw, "artifact_budget_seconds", "runtime."),
                "runtime.artifact_budget_seconds",
            ),
            reserve_seconds=_as_int(
                _require(runtime_raw, "reserve_seconds", "runtime."),
                "runtime.reserve_seconds",
            ),
            probe_warmup_steps=_as_int(
                _require(runtime_raw, "probe_warmup_steps", "runtime."),
                "runtime.probe_warmup_steps",
            ),
            probe_timed_steps=_as_int(
                _require(runtime_raw, "probe_timed_steps", "runtime."),
                "runtime.probe_timed_steps",
            ),
        )

        stage_total = (
            runtime.pack_budget_seconds
            + runtime.setup_probe_budget_seconds
            + runtime.train_eval_budget_seconds
            + runtime.artifact_budget_seconds
            + runtime.reserve_seconds
        )
        if runtime.world_size < 0:
            raise ValueError("runtime.world_size must be 'auto' or a positive integer")
        if stage_total != runtime.total_budget_seconds:
            raise ValueError(
                f"runtime stage budgets must sum to {runtime.total_budget_seconds}; "
                f"got {stage_total}"
            )
        if runtime.token_budget <= 0:
            raise ValueError("runtime.token_budget must be positive")

    return Config(
        model=model,
        data=data,
        optim=optim,
        eval=eval_cfg,
        seed=_as_int(_require(raw, "seed", ""), "seed"),
        output_dir=Path(_as_str(_require(raw, "output_dir", ""), "output_dir")),
        mixed_precision=mixed_precision,
        runtime=runtime,
    )


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    """Parse command-line arguments for the training CLI.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The parsed `CliArgs`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.train_b0",
        description="Train a frozen B0 pairwise scorer (v3_1 or f0_mlp).",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="DEBUG ONLY: stop after N optimizer steps (bounded smoke runs).",
    )
    parser.add_argument(
        "--ddp-mode",
        choices=DDP_MODES,
        default=None,
        help="internal multi-H20 worker mode (launched by accelerate launch).",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help="packed-feature directory the DDP worker loads onto its device.",
    )
    parser.add_argument(
        "--token-budget-per-rank",
        type=int,
        default=None,
        help="per-rank token budget for the distributed batch plan.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="path the rank-zero DDP worker writes its JSON profile artifact to.",
    )
    namespace = parser.parse_args(argv)
    if namespace.ddp_mode is not None:
        missing = [
            flag
            for flag, value in (
                ("--pack-dir", namespace.pack_dir),
                ("--output-dir", namespace.output_dir),
                ("--token-budget-per-rank", namespace.token_budget_per_rank),
                ("--profile-output", namespace.profile_output),
            )
            if value is None
        ]
        if missing:
            parser.error(f"--ddp-mode requires {', '.join(missing)}")
    return CliArgs(
        config=namespace.config,
        seed=namespace.seed,
        output_dir=namespace.output_dir,
        max_steps=namespace.max_steps,
        ddp_mode=namespace.ddp_mode,
        pack_dir=namespace.pack_dir,
        token_budget_per_rank=namespace.token_budget_per_rank,
        profile_output=namespace.profile_output,
    )


def apply_overrides(cfg: Config, args: CliArgs) -> Config:
    """Apply CLI overrides (``--seed``, ``--output-dir``) on top of the config.

    Args:
        cfg: Loaded config.
        args: Parsed CLI arguments.

    Returns:
        The config with overrides applied (unchanged when no overrides given).
    """
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a `Config` to a JSON-serializable nested dict (Paths become str).

    Args:
        cfg: The config to convert.

    Returns:
        A nested plain dict safe for ``json.dumps`` (used for the config hash and
        the checkpoint ``config`` payload).
    """

    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return cast(dict[str, Any], convert(asdict(cfg)))


# --------------------------------------------------------------------------- model building


def resolve_model_kwargs(model_cfg: ModelConfig) -> dict[str, object]:
    """Resolve the effective model constructor kwargs for a model config.

    Args:
        model_cfg: The ``model:`` config section.

    Returns:
        For ``v3_1``: the explicit kwargs, or a copy of
        :data:`~src.model.B0.BEST_V3_1_CONFIG` when empty. For ``f0_mlp``: the
        explicit kwargs (empty means :class:`~src.model.b0_alt.F0PairMLP` defaults).

    Raises:
        ValueError: If the family is unknown.
    """
    if model_cfg.family == "v3_1":
        return dict(model_cfg.config) if model_cfg.config else dict(BEST_V3_1_CONFIG)
    if model_cfg.family == "f0_mlp":
        return dict(model_cfg.config)
    raise ValueError(f"unknown model family '{model_cfg.family}' (expected v3_1 or f0_mlp)")


def build_model(cfg: Config) -> nn.Module:
    """Build the scorer model named by ``cfg.model``.

    Args:
        cfg: The full training config.

    Returns:
        A `V3_1` or `F0PairMLP` instance.

    Raises:
        ValueError: If the model family is unknown.
    """
    kwargs = resolve_model_kwargs(cfg.model)
    if cfg.model.family == "v3_1":
        return V3_1(**kwargs)
    return F0PairMLP(**cast(dict[str, Any], kwargs))


# --------------------------------------------------------------------------- data assembly


@dataclass(frozen=True)
class AssembledData:
    """Everything the training loop needs from the benchmark + feature packages.

    Attributes:
        benchmark: Loaded benchmark (featureless-node pairs already dropped).
        store: Frozen per-node token-sequence feature store.
        training_positives: Positive training pairs per ``data.train_positives``.
        degrees: Simple-graph degrees from ``G_struct`` (message partition).
        dropped_pair_counts: Rows dropped per labeled file due to missing features.
        operative_node_ids: Sorted node ids in both the graph and the feature index.
        operative_node_count: ``len(operative_node_ids)``.
        exclude_nodes: Graph nodes without features (the verified missing set).
    """

    benchmark: Benchmark
    store: FeatureStore
    training_positives: list[tuple[str, str]]
    degrees: dict[str, int]
    dropped_pair_counts: dict[str, int]
    operative_node_ids: list[str]
    operative_node_count: int
    exclude_nodes: frozenset[str]


def _count_rows(path: Path) -> int:
    """Count non-empty lines of a TSV file (raw row count before filtering)."""
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def assemble_data(cfg: Config, *, verify: bool = True) -> AssembledData:
    """Assemble benchmark + features for training, enforcing the coverage gate.

    Feature-coverage gate: after computing ``exclude = graph_nodes -
    store.node_ids``, this raises unless ``exclude`` equals
    ``data.expected_missing_features`` exactly, and logs the operative node count
    ``|graph_nodes ∩ store.node_ids|`` (10,088 on the real package).

    Args:
        cfg: The full training config.
        verify: Forwarded to :func:`~src.data.artifacts.load_benchmark`; when True
            the raw artifacts are verified against the pinned constants first.

    Returns:
        The `AssembledData` bundle.

    Raises:
        ArtifactVerificationError: If the set of graph nodes lacking features
            drifts from ``data.expected_missing_features`` (or, with
            ``verify=True``, if the artifact package itself fails verification).
    """
    benchmark_root = cfg.data.root / "benchmark_2025_neurips"
    features_root = cfg.data.root / "features" / "frozen_node_features_1024"

    store = FeatureStore(features_root)
    with (benchmark_root / "graph.pkl").open("rb") as f:
        graph = pickle.load(f)
    graph_nodes: set[str] = set(graph.nodes())

    exclude = frozenset(graph_nodes - store.node_ids)
    expected_missing = set(cfg.data.expected_missing_features)
    if set(exclude) != expected_missing:
        raise ArtifactVerificationError(
            "feature-coverage gate failed: graph nodes missing features "
            f"{sorted(exclude)} != expected {sorted(expected_missing)}"
        )
    operative_node_ids = sorted(graph_nodes & store.node_ids)
    logger.info(
        "feature coverage: %d operative nodes (graph ∩ feature index); %d excluded: %s",
        len(operative_node_ids),
        len(exclude),
        sorted(exclude),
    )

    bench = load_benchmark(benchmark_root, cfg.data.strategy, verify=verify, exclude_nodes=exclude)

    strategy_dir = benchmark_root / cfg.data.strategy
    dropped_pair_counts = {
        "train_edges.txt": _count_rows(strategy_dir / "train_edges.txt")
        - len(bench.split.train_pairs.pairs),
        "val_edges.txt": _count_rows(strategy_dir / "val_edges.txt")
        - len(bench.split.val_pairs.pairs),
        "test_edges.txt": _count_rows(strategy_dir / "test_edges.txt")
        - len(bench.split.test_pairs.pairs),
    }

    train_pairs = bench.split.train_pairs
    train_plus = [
        pair
        for pair, label in zip(train_pairs.pairs, train_pairs.labels, strict=True)
        if label == 1
    ]
    partition = derive_partition(train_plus, cfg.data.partition_seed)
    if cfg.data.train_positives == "train_plus":
        training_positives = train_plus
    else:
        training_positives = sorted(partition.e_sup)

    g_struct = build_g_struct(bench.split.train_nodes, partition.e_msg)
    degrees = {str(node): int(degree) for node, degree in g_struct.degree()}

    return AssembledData(
        benchmark=bench,
        store=store,
        training_positives=training_positives,
        degrees=degrees,
        dropped_pair_counts=dropped_pair_counts,
        operative_node_ids=operative_node_ids,
        operative_node_count=len(operative_node_ids),
        exclude_nodes=exclude,
    )


# --------------------------------------------------------------------------- training loop


@dataclass
class TrainResult:
    """Outcome of a full training run.

    Attributes:
        best_state_dict: CPU state dict of the best-val-AUPRC checkpoint.
        best_epoch: Epoch (1-based) of the best checkpoint.
        best_val_metrics: Full val metrics of the best checkpoint.
        last_state_dict: CPU state dict at the end of training.
        last_epoch: Last epoch (1-based) trained.
        last_val_metrics: Full val metrics of the final evaluation.
        history: One entry per evaluation: epoch, train_loss, val_auroc, val_auprc.
        stopped_early: Whether early stopping fired before ``optim.epochs``.
        counterfactual_stop_epoch: Epoch at which patience *would* have stopped the
            run, recorded without ever stopping it (``None`` on the legacy
            single-GPU :func:`train_loop`, which really does stop early). The
            fixed-epoch DDP loop always trains ``optim.epochs`` epochs.
        runtime_profile: Rank-zero timing/coverage payload for the fixed-epoch DDP
            run (empty on the legacy :func:`train_loop`). Keys are pinned by the
            Task-12 acceptance test.
    """

    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_metrics: EdgeMetrics
    last_state_dict: dict[str, torch.Tensor]
    last_epoch: int
    last_val_metrics: EdgeMetrics
    history: list[dict[str, float]]
    stopped_early: bool
    counterfactual_stop_epoch: int | None = None
    runtime_profile: dict[str, object] = field(default_factory=dict)


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor of a batch dict to `device`."""
    return {key: value.to(device) for key, value in batch.items()}


def _cpu_state_dict(accelerator: Accelerator, model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the unwrapped model state dict onto the CPU (cloned)."""
    unwrapped = accelerator.unwrap_model(model)
    return {key: value.detach().to("cpu").clone() for key, value in unwrapped.state_dict().items()}


def _evaluate(
    model: nn.Module, val_loader: Iterable[Batch], accelerator: Accelerator
) -> EdgeMetrics:
    """Compute full edge metrics over the validation loader (model left in train mode)."""
    model.eval()
    labels_parts: list[np.ndarray] = []
    probs_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, accelerator.device)
            output = model(batch)
            logits = output["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            probs_parts.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            labels_parts.append(batch["label"].detach().float().cpu().numpy())
    model.train()
    labels = np.concatenate(labels_parts)
    probs = np.concatenate(probs_parts)
    return compute_edge_metrics(labels, probs)


def train_loop(
    model: nn.Module,
    train_loader_factory: LoaderFactory,
    val_loader: Iterable[Batch],
    cfg: Config,
    accelerator: Accelerator,
    *,
    max_steps: int | None = None,
    on_eval: OnEval | None = None,
) -> TrainResult:
    """Run the full training loop with eval-driven checkpoint selection.

    AdamW + linear warmup then constant LR + gradient clipping; BCE-with-logits is
    computed by the models themselves when ``label`` is present in the batch.
    Every ``eval_every`` epochs the val set is scored with
    :func:`~src.eval.edge_metrics.compute_edge_metrics` on ``sigmoid(logits)``;
    the best checkpoint is kept by val AUPRC and training stops early after
    ``patience`` evals without improvement. A final-epoch eval always runs so at
    least one checkpoint exists.

    Args:
        model: The scorer to train (not yet prepared by the accelerator).
        train_loader_factory: ``epoch -> iterable of batch dicts`` (epoch shuffling
            happens inside the factory).
        val_loader: Re-iterable batches of the fixed validation set.
        cfg: The full training config.
        accelerator: HF Accelerator (single-process semantics).
        max_steps: DEBUG ONLY — stop after this many optimizer steps.
        on_eval: Optional callback ``(history_entry, improved, metrics)`` invoked
            after every evaluation (used by the CLI for incremental artifacts).

    Returns:
        The `TrainResult`.

    Raises:
        RuntimeError: If training ends without a single evaluation.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    warmup = max(1, cfg.optim.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup))
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: EdgeMetrics | None = None
    best_epoch = 0
    last_metrics: EdgeMetrics | None = None
    evals_without_improvement = 0
    stopped_early = False
    reached_max_steps = False
    global_step = 0
    last_epoch = 0

    for epoch in range(1, cfg.optim.epochs + 1):
        last_epoch = epoch
        model.train()
        losses: list[float] = []
        for batch in train_loader_factory(epoch):
            batch = _to_device(batch, accelerator.device)
            output = model(batch)
            loss = output["loss"]
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().float().item()))
            if global_step % 50 == 0:
                logger.info(
                    "epoch %d step %d loss %.4f lr %.3e",
                    epoch,
                    global_step,
                    losses[-1],
                    scheduler.get_last_lr()[0],
                )
            if max_steps is not None and global_step >= max_steps:
                reached_max_steps = True
                logger.warning("--max-steps %d reached (debug); stopping training", max_steps)
                break

        train_loss = float(np.mean(losses)) if losses else float("nan")
        is_final = epoch == cfg.optim.epochs or reached_max_steps
        if epoch % cfg.eval.eval_every == 0 or is_final:
            metrics = _evaluate(model, val_loader, accelerator)
            last_metrics = metrics
            entry: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auroc": metrics.auroc,
                "val_auprc": metrics.auprc,
            }
            history.append(entry)
            improved = best_metrics is None or metrics.auprc > best_metrics.auprc
            if improved:
                best_metrics = metrics
                best_epoch = epoch
                best_state = _cpu_state_dict(accelerator, model)
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            logger.info(
                "eval epoch %d: train_loss %.4f val_auroc %.4f val_auprc %.4f%s",
                epoch,
                train_loss,
                metrics.auroc,
                metrics.auprc,
                " (new best)" if improved else "",
            )
            if on_eval is not None:
                on_eval(entry, improved, metrics)
            if evals_without_improvement >= cfg.eval.patience:
                stopped_early = True
                logger.info(
                    "early stopping at epoch %d (%d evals without val-AUPRC improvement)",
                    epoch,
                    evals_without_improvement,
                )
        if stopped_early or reached_max_steps:
            break

    if best_state is None or best_metrics is None or last_metrics is None:
        raise RuntimeError("training ended without a single evaluation")

    return TrainResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        last_state_dict=_cpu_state_dict(accelerator, model),
        last_epoch=last_epoch,
        last_val_metrics=last_metrics,
        history=history,
        stopped_early=stopped_early,
    )


# --------------------------------------------------------------------------- output artifacts


def _state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Deterministic sha256 hex digest over a state dict's tensors (sorted keys)."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy(force=True).tobytes())
    return hasher.hexdigest()


def _checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    cfg: Config,
    model_kwargs: dict[str, object],
    epoch: int,
    metrics: EdgeMetrics,
    config_dict: dict[str, Any],
) -> dict[str, object]:
    """Build the pinned checkpoint payload (contract consumed by score_universe)."""
    return {
        "model_state": state_dict,
        "model_family": cfg.model.family,
        "model_config": model_kwargs,
        "epoch": epoch,
        "val_metrics": asdict(metrics),
        "seed": cfg.seed,
        "config": config_dict,
    }


def write_outputs(
    result: TrainResult,
    cfg: Config,
    model_kwargs: dict[str, object],
    dropped_pair_counts: dict[str, int],
) -> None:
    """Write the pinned run artifacts into ``cfg.output_dir``.

    Writes ``best.pt`` / ``last.pt`` (payload keys exactly ``model_state``,
    ``model_family``, ``model_config``, ``epoch``, ``val_metrics``, ``seed``,
    ``config``), ``metrics.jsonl`` (one line per eval), and ``run_metadata.json``
    (config hash, checkpoint id = first 16 hex of the sha256 over the best
    checkpoint's model_state tensor bytes, torch version, timestamp, dropped-pair
    counts, positives mode).

    Args:
        result: The finished training result.
        cfg: The full training config.
        model_kwargs: Resolved model constructor kwargs (stored as ``model_config``).
        dropped_pair_counts: Per-file dropped-row counts from data assembly.
    """
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config_to_dict(cfg)

    torch.save(
        _checkpoint_payload(
            result.best_state_dict,
            cfg,
            model_kwargs,
            result.best_epoch,
            result.best_val_metrics,
            config_dict,
        ),
        output_dir / "best.pt",
    )
    torch.save(
        _checkpoint_payload(
            result.last_state_dict,
            cfg,
            model_kwargs,
            result.last_epoch,
            result.last_val_metrics,
            config_dict,
        ),
        output_dir / "last.pt",
    )

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for entry in result.history:
            f.write(json.dumps(entry) + "\n")

    run_metadata = {
        "config_hash": hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "checkpoint_id": _state_digest(result.best_state_dict)[:16],
        "torch_version": str(torch.__version__),
        "timestamp": datetime.now(UTC).isoformat(),
        "dropped_pair_counts": dropped_pair_counts,
        "positives_mode": cfg.data.train_positives,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "wrote artifacts to %s (checkpoint_id %s)", output_dir, run_metadata["checkpoint_id"]
    )


def prepare_pack(
    cfg: Config, pack_dir: Path, *, cold_cache: bool, temp_prefix: str = ""
) -> dict[str, object]:
    """Build (cold) or strictly validate (warm) the V3.1 BF16 feature pack.

    The orchestrator's per-worker pack seam (`src.e2_pipeline.run_pipeline`
    dispatches to ``<worker>.prepare_pack``); this implementation preserves the
    pipeline's original inline semantics exactly.

    Args:
        cfg: The validated training config (``runtime`` must be present).
        pack_dir: The packed-feature directory.
        cold_cache: ``True`` builds from scratch; ``False`` validates.
        temp_prefix: Temp-directory prefix for the cold build.

    Returns:
        ``{"pack_manifest": {...}, "pack_identity_sha256": <sha of manifest.json>}``.

    Raises:
        ValueError: If ``cfg.runtime`` is missing, or on validation drift.
    """
    # Call-time import: keeps the module-attribute patch seam used by the
    # pipeline tests (src.data.packed_features.<fn>) working unchanged.
    from src.data import packed_features

    if cfg.runtime is None:
        raise ValueError("prepare_pack requires a configured cfg.runtime")
    source_root = cfg.data.root / "features" / "frozen_node_features_1024"
    if cold_cache:
        logger.info("pack cache cold: building %s from %s", pack_dir, source_root)
        manifest = packed_features.build_packed_features(
            source_root,
            pack_dir,
            workers=cfg.runtime.pack_workers,
            temp_prefix=temp_prefix,
        )
    else:
        logger.info("pack cache warm: validating %s against %s", pack_dir, source_root)
        manifest = packed_features.validate_packed_manifest(pack_dir, source_root)
    return {
        "pack_manifest": cast(dict[str, object], asdict(manifest)),
        "pack_identity_sha256": packed_features.sha256_file(pack_dir / "manifest.json"),
    }


# --------------------------------------------------------------------------- real-data loaders


class _F0PairBatches:
    """Re-iterable batches of F0-matrix row pairs (used for both train and val)."""

    def __init__(
        self,
        matrix: torch.Tensor,
        a_indices: torch.Tensor,
        b_indices: torch.Tensor,
        labels: torch.Tensor,
        batch_pairs: int,
    ) -> None:
        """Build a batch iterable over aligned row-index/label tensors.

        Args:
            matrix: The ``(N, input_dim)`` F0 feature matrix.
            a_indices: ``(R,)`` int64 rows for the first endpoint of each pair.
            b_indices: ``(R,)`` int64 rows for the second endpoint of each pair.
            labels: ``(R,)`` float32 labels aligned with the index tensors.
            batch_pairs: Number of pairs per yielded batch.
        """
        self._matrix = matrix
        self._a_indices = a_indices
        self._b_indices = b_indices
        self._labels = labels
        self._batch_pairs = batch_pairs

    def __iter__(self) -> Iterator[Batch]:
        """Yield ``{"x_a", "x_b", "label"}`` batches in index order."""
        total = self._a_indices.size(0)
        for start in range(0, total, self._batch_pairs):
            stop = start + self._batch_pairs
            yield {
                "x_a": self._matrix[self._a_indices[start:stop]],
                "x_b": self._matrix[self._b_indices[start:stop]],
                "label": self._labels[start:stop],
            }


def _shuffled_pairs_and_labels(
    positives: Sequence[tuple[str, str]],
    negatives: Sequence[tuple[str, str]],
    seed: int,
    epoch: int,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Combine positives+negatives and shuffle them with a per-epoch seeded RNG."""
    pairs = list(positives) + list(negatives)
    labels = [1] * len(positives) + [0] * len(negatives)
    rng = np.random.default_rng((seed, epoch))
    permutation = rng.permutation(len(pairs))
    return [pairs[i] for i in permutation], [labels[i] for i in permutation]


def _build_negative_sampler(assembled: AssembledData) -> NegativeSampler:
    """Build the degree-corrected negative sampler over featureful train nodes."""
    train_universe = sorted(set(assembled.benchmark.split.train_nodes) - assembled.exclude_nodes)
    return NegativeSampler(train_universe, assembled.degrees, assembled.benchmark.positive_edges)


def _build_v3_1_loaders(
    cfg: Config, assembled: AssembledData
) -> tuple[LoaderFactory, Iterable[Batch]]:
    """Build loaders using the fixed labeled training pairs from ``train_edges.txt``."""
    store = assembled.store
    train_pairs = assembled.benchmark.split.train_pairs
    positives = [
        pair
        for pair, label in zip(train_pairs.pairs, train_pairs.labels, strict=True)
        if label == 1
    ]
    negatives = [
        pair
        for pair, label in zip(train_pairs.pairs, train_pairs.labels, strict=True)
        if label == 0
    ]
    length_cache: dict[str, int] = {}

    def length_of(node_id: str) -> int:
        cached = length_cache.get(node_id)
        if cached is None:
            cached = int(store.load_tokens(node_id).size(0))
            length_cache[node_id] = cached
            if len(length_cache) % 1000 == 0:
                logger.info("length probe: %d unique nodes cached", len(length_cache))
        return cached

    def lengths_for(pairs: Sequence[tuple[str, str]]) -> list[tuple[int, int]]:
        return [(length_of(u), length_of(v)) for u, v in pairs]

    val_pairs = assembled.benchmark.split.val_pairs
    val_labels = [int(label) for label in val_pairs.labels]
    val_lengths = lengths_for(val_pairs.pairs)
    val_dataset = TokenPairDataset(val_pairs.pairs, val_labels, store, lengths=val_lengths)
    val_loader: DataLoader[Batch] = DataLoader(
        val_dataset,
        batch_sampler=LengthBucketedBatchSampler(
            val_lengths, token_budget=cfg.data.token_budget, shuffle=False
        ),
        collate_fn=collate_token_pairs,
        num_workers=cfg.data.num_workers,
    )

    def factory(epoch: int) -> DataLoader[Batch]:
        pairs, labels = _shuffled_pairs_and_labels(positives, negatives, cfg.seed, epoch)
        lengths = lengths_for(pairs)
        dataset = TokenPairDataset(pairs, labels, store, lengths=lengths)
        batch_sampler = LengthBucketedBatchSampler(
            lengths,
            token_budget=cfg.data.token_budget,
            shuffle=True,
            seed=cfg.seed,
            epoch=epoch,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_token_pairs,
            num_workers=cfg.data.num_workers,
        )

    return factory, val_loader


def _build_f0_loaders(
    cfg: Config, assembled: AssembledData
) -> tuple[LoaderFactory, Iterable[Batch]]:
    """Build the F0-matrix train-loader factory and val loader (f0_mlp path)."""
    cfg.data.f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, row_index = build_f0_matrix(
        assembled.store, assembled.operative_node_ids, cache_path=cfg.data.f0_cache
    )
    sampler = _build_negative_sampler(assembled)
    positives = assembled.training_positives

    def batches_for(pairs: Sequence[tuple[str, str]], labels: Sequence[int]) -> _F0PairBatches:
        a_indices = torch.tensor([row_index[u] for u, _ in pairs], dtype=torch.int64)
        b_indices = torch.tensor([row_index[v] for _, v in pairs], dtype=torch.int64)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return _F0PairBatches(matrix, a_indices, b_indices, label_tensor, cfg.data.batch_pairs)

    val_pairs = assembled.benchmark.split.val_pairs
    val_loader = batches_for(val_pairs.pairs, [int(label) for label in val_pairs.labels])

    def factory(epoch: int) -> _F0PairBatches:
        negatives = sampler.sample(
            positives, ratio=cfg.data.negative_ratio, seed=cfg.seed, epoch=epoch
        )
        pairs, labels = _shuffled_pairs_and_labels(positives, negatives, cfg.seed, epoch)
        return batches_for(pairs, labels)

    return factory, val_loader


class GpuBatchIterable:
    """Adapt a compact-batch source into assembled model batches.

    Each worker process yields :class:`~src.data.distributed_pairs.CompactPairBatch`
    values (integer endpoint ids, labels, and lengths only); this iterable does the
    actual feature gather — :meth:`~src.data.packed_features.PackedFeatureTable.assemble`
    — in the main process, from the packed feature table already resident on the
    training device. No feature tensor ever crosses the DataLoader worker boundary.
    """

    def __init__(self, source: Iterable[CompactPairBatch], table: PackedFeatureTable) -> None:
        """Store the compact-batch source and the packed feature table to assemble from."""
        self._source = source
        self._table = table

    def __iter__(self) -> Iterator[Batch]:
        """Yield one assembled model batch per compact batch from the source."""
        for compact in self._source:
            yield self._table.assemble(compact)


def compute_sample_warmup_steps(
    baseline_batch_sizes: Sequence[int],
    new_global_batch_sizes: Sequence[int],
    *,
    baseline_steps: int,
) -> int:
    """Return the new-schedule step count needed to preserve warmup pair exposure.

    ``baseline_batch_sizes`` is the legacy single-GPU per-step batch-size sequence;
    cycling it for ``baseline_steps`` steps defines the target number of pair
    samples the original warmup schedule saw. This returns how many steps of the
    new (larger, distributed) global-batch-size sequence — also cycled — are
    needed to see at least as many total pair samples.

    Args:
        baseline_batch_sizes: Legacy per-step batch sizes (one epoch's worth).
        new_global_batch_sizes: New per-step *global* (summed across ranks) batch
            sizes (one epoch's worth).
        baseline_steps: Number of legacy warmup steps to reproduce the exposure of.

    Returns:
        The number of new-schedule steps whose cumulative pair count first
        reaches the baseline target.

    Raises:
        ValueError: If either batch-size sequence is empty.
    """
    if not baseline_batch_sizes or not new_global_batch_sizes:
        raise ValueError("warmup batch-size sequences must be non-empty")
    target = sum(islice(cycle(baseline_batch_sizes), baseline_steps))
    seen = 0
    for steps, batch_size in enumerate(cycle(new_global_batch_sizes), start=1):
        seen += batch_size
        if seen >= target:
            return steps
    raise AssertionError("cycle over non-empty batch sizes must terminate")


def _pair_lengths_from_manifest(
    pairs: Sequence[tuple[str, str]], lengths_by_node: dict[str, int]
) -> list[tuple[int, int]]:
    """Look up each pair's ``(L_a, L_b)`` from the packed manifest (no feature I/O)."""
    return [(lengths_by_node[node_a], lengths_by_node[node_b]) for node_a, node_b in pairs]


def _compact_pair_columns(
    pairs: Sequence[tuple[str, str]],
    labels: Sequence[int],
    node_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the ``(row_ids, node_a, node_b, labels)`` compact CPU columns for a pair list."""
    row_ids = torch.arange(len(pairs), dtype=torch.int64)
    node_a = torch.tensor([node_index[u] for u, _ in pairs], dtype=torch.int64)
    node_b = torch.tensor([node_index[v] for _, v in pairs], dtype=torch.int64)
    label_tensor = torch.tensor([float(label) for label in labels], dtype=torch.float32)
    return row_ids, node_a, node_b, label_tensor


def _validate_distributed_plan(
    plan: Sequence[Sequence[PairBatchSpec]], *, expected_row_count: int
) -> None:
    """Validate equal-step metadata and exact global row coverage before loading."""
    if not plan:
        raise ValueError("distributed training plan has no ranks")
    step_counts = {len(rank_plan) for rank_plan in plan}
    if len(step_counts) != 1:
        raise ValueError(f"distributed training plan has unequal step counts: {step_counts}")
    all_indices: list[int] = []
    for step in range(len(plan[0])):
        specs = [rank_plan[step] for rank_plan in plan]
        global_counts = {spec.global_pair_count for spec in specs}
        actual_count = sum(len(spec.indices) for spec in specs)
        if len(global_counts) != 1 or actual_count != specs[0].global_pair_count:
            raise ValueError(f"distributed training plan step {step} has inconsistent counts")
        all_indices.extend(index for spec in specs for index in spec.indices)
    unique_ids, counts = np.unique(np.asarray(all_indices, dtype=np.int64), return_counts=True)
    if np.any(counts > 1):
        duplicates = unique_ids[counts > 1].tolist()
        raise ValueError(f"duplicate training plan row IDs: {duplicates}")
    expected = np.arange(expected_row_count, dtype=np.int64)
    if not np.array_equal(unique_ids, expected):
        raise ValueError(
            "distributed training plan does not cover the fixed row set "
            f"({unique_ids.shape[0]} unique of {expected_row_count} expected)"
        )


def _interleave_bucket_specs(specs: Sequence[PairBatchSpec]) -> list[PairBatchSpec]:
    """Round-robin compact batch specs by length bucket, preserving bucket order."""
    buckets: dict[int, list[PairBatchSpec]] = {}
    for spec in specs:
        buckets.setdefault(spec.bucket_boundary, []).append(spec)
    ordered: list[PairBatchSpec] = []
    for offset in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for bucket in buckets.values():
            if offset < len(bucket):
                ordered.append(bucket[offset])
    return ordered


class _EpochIndexSampler(Sampler[int]):
    """Select one epoch's contiguous slice from a flattened plan dataset."""

    def __init__(self, offsets: dict[int, tuple[int, int]]) -> None:
        self._offsets = offsets
        self._epoch = min(offsets)

    def set_epoch(self, epoch: int) -> None:
        if epoch not in self._offsets:
            raise ValueError(f"no packed training plan for epoch {epoch}")
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        start, stop = self._offsets[self._epoch]
        return iter(range(start, stop))

    def __len__(self) -> int:
        start, stop = self._offsets[self._epoch]
        return stop - start


class _EpochGpuBatchIterable(GpuBatchIterable):
    """Reuse one DataLoader while selecting the requested epoch at iteration time."""

    def __init__(
        self,
        source: DataLoader[CompactPairBatch],
        table: PackedFeatureTable,
        sampler: _EpochIndexSampler,
        epoch: int,
    ) -> None:
        super().__init__(source, table)
        self._sampler = sampler
        self._epoch = epoch

    def __iter__(self) -> Iterator[Batch]:
        self._sampler.set_epoch(self._epoch)
        yield from super().__iter__()


def _wrap_compact_loader(
    dataset: CompactPairBatchDataset,
    cfg: Config,
    world_size: int,
    *,
    sampler: Sampler[int] | None = None,
) -> DataLoader[CompactPairBatch]:
    """Wrap a per-rank compact-batch dataset; workers only ever move index tensors.

    ``world_size == 1`` is the unit-test regime: a non-persistent single-process
    loader avoids spawning worker processes for a handful of synthetic batches.
    Any other world size uses the fixed production configuration: persistent
    multi-process workers pinned by ``cfg.runtime``.
    """
    # torch's DataLoader stub types collate_fn as Callable[[list[T]], Any] even when
    # batch_size=None (no list wrapping occurs); identity_compact_batch's precise
    # Callable[[CompactPairBatch], CompactPairBatch] signature is correct at runtime.
    if world_size == 1:
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            collate_fn=identity_compact_batch,  # type: ignore[arg-type]
        )
    runtime = cfg.runtime
    if runtime is None:
        raise ValueError("packed v3_1 loaders require a configured cfg.runtime for world_size > 1")
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=runtime.loader_workers_per_rank,
        persistent_workers=True,
        prefetch_factor=runtime.prefetch_factor,
        collate_fn=identity_compact_batch,  # type: ignore[arg-type]
    )


def _build_packed_v3_1_loaders(
    cfg: Config,
    assembled: AssembledData,
    table: PackedFeatureTable,
    *,
    token_budget_per_rank: int,
    process_index: int,
    world_size: int,
) -> tuple[PackedLoaderFactory, Iterable[Batch], int]:
    """Build packed, multi-worker ``v3_1`` train/val loaders for one DDP rank.

    Endpoint integer ids and true token lengths come from ``table.manifest`` only —
    no feature tensors are read to plan batches. The fixed labeled
    ``train_edges.txt`` rows are replanned into per-rank compact batches every
    epoch via :func:`~src.data.distributed_pairs.build_distributed_epoch_plan`; the
    fixed validation rows are planned once, unshuffled. DataLoader workers hand
    back :class:`~src.data.distributed_pairs.CompactPairBatch` values only — the
    returned iterables assemble feature tensors from ``table`` in the main process.

    Args:
        cfg: The full training config; ``cfg.runtime`` must be set.
        assembled: Assembled benchmark data (only ``.benchmark.split`` is used).
        table: Packed feature table resident on the training device.
        token_budget_per_rank: Per-rank token budget for the distributed batch plan.
        process_index: This rank's index into the ``world_size`` per-rank plans.
        world_size: Total rank count (``1`` selects the unit-test single-process path).

    Returns:
        ``(train_loader_factory, val_loader, warmup_steps)``.

    Raises:
        ValueError: If ``cfg.runtime`` is unset.
    """
    runtime = cfg.runtime
    if runtime is None:
        raise ValueError("packed v3_1 loaders require a configured cfg.runtime")

    node_index = table.manifest.node_index()
    lengths_by_node = {record.node_id: record.length for record in table.manifest.nodes}

    train_pairs_bundle = assembled.benchmark.split.train_pairs
    train_pairs = list(train_pairs_bundle.pairs)
    train_labels = [int(label) for label in train_pairs_bundle.labels]
    train_lengths = _pair_lengths_from_manifest(train_pairs, lengths_by_node)
    train_row_ids, train_node_a, train_node_b, train_label_tensor = _compact_pair_columns(
        train_pairs, train_labels, node_index
    )

    val_pairs_bundle = assembled.benchmark.split.val_pairs
    val_pairs = list(val_pairs_bundle.pairs)
    val_labels = [int(label) for label in val_pairs_bundle.labels]
    val_lengths = _pair_lengths_from_manifest(val_pairs, lengths_by_node)
    val_row_ids, val_node_a, val_node_b, val_label_tensor = _compact_pair_columns(
        val_pairs, val_labels, node_index
    )

    val_plan = build_distributed_epoch_plan(
        val_lengths,
        token_budget_per_rank=token_budget_per_rank,
        max_pairs_per_rank=runtime.max_pairs_per_rank,
        world_size=world_size,
        seed=cfg.seed,
        epoch=0,
        shuffle=False,
    )
    _validate_distributed_plan(val_plan, expected_row_count=len(val_pairs))
    val_dataset = CompactPairBatchDataset(
        val_row_ids, val_node_a, val_node_b, val_label_tensor, val_plan[process_index]
    )
    val_loader = GpuBatchIterable(_wrap_compact_loader(val_dataset, cfg, world_size), table)

    flat_rank_specs: list[PairBatchSpec] = []
    epoch_offsets: dict[int, tuple[int, int]] = {}
    plans_by_epoch: dict[int, list[list[PairBatchSpec]]] = {}
    for epoch in range(cfg.optim.epochs + 1):
        plan = build_distributed_epoch_plan(
            train_lengths,
            token_budget_per_rank=token_budget_per_rank,
            max_pairs_per_rank=runtime.max_pairs_per_rank,
            world_size=world_size,
            seed=cfg.seed,
            epoch=epoch,
            shuffle=True,
        )
        plan = [_interleave_bucket_specs(rank_plan) for rank_plan in plan]
        _validate_distributed_plan(plan, expected_row_count=len(train_pairs))
        plans_by_epoch[epoch] = plan
        start = len(flat_rank_specs)
        flat_rank_specs.extend(plan[process_index])
        epoch_offsets[epoch] = (start, len(flat_rank_specs))

    train_dataset = CompactPairBatchDataset(
        train_row_ids,
        train_node_a,
        train_node_b,
        train_label_tensor,
        flat_rank_specs,
    )
    epoch_sampler = _EpochIndexSampler(epoch_offsets)
    train_loader = _wrap_compact_loader(train_dataset, cfg, world_size, sampler=epoch_sampler)

    def factory(epoch: int) -> GpuBatchIterable:
        return _EpochGpuBatchIterable(train_loader, table, epoch_sampler, epoch)

    baseline_batch_sizes = [
        len(batch)
        for batch in LengthBucketedBatchSampler(
            train_lengths,
            token_budget=cfg.data.token_budget,
            shuffle=True,
            seed=cfg.seed,
            epoch=0,
        )
    ]
    reference_plan = plans_by_epoch[0]
    new_global_batch_sizes = [spec.global_pair_count for spec in reference_plan[process_index]]
    warmup_steps = compute_sample_warmup_steps(
        baseline_batch_sizes, new_global_batch_sizes, baseline_steps=cfg.optim.warmup_steps
    )

    return factory, val_loader, warmup_steps


# --------------------------------------------------------------------------- DDP training


def build_ddp_accelerator(mixed_precision: str) -> Accelerator:
    """Build the multi-H20 DDP accelerator with the pinned communication settings.

    Args:
        mixed_precision: ``"no"`` or ``"bf16"`` (the formal E2 run pins ``"bf16"``).

    Returns:
        An `Accelerator` whose DDP wrapping disables buffer broadcast and
        unused-parameter search and reuses gradient bucket views.
    """
    kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[kwargs])


def scale_ddp_mean_loss(
    loss: torch.Tensor, *, local_count: int, global_count: int, world_size: int
) -> torch.Tensor:
    """Rescale a per-rank mean loss so its gradient matches the global sample mean.

    DDP averages gradients across ranks, but a tail batch gives ranks unequal
    pair counts, so each rank's local mean must be reweighted by
    ``local_count * world_size / global_count`` before ``backward`` for the
    averaged gradient to equal the true global-batch mean gradient.

    Args:
        loss: This rank's mean loss over its ``local_count`` pairs.
        local_count: Pairs this rank contributed to the global batch (``>= 1``).
        global_count: Pairs across all ranks in the global batch (``>= local_count``).
        world_size: Number of ranks averaged by DDP (``>= 1``).

    Returns:
        The rescaled loss.

    Raises:
        ValueError: If the counts are non-positive or inconsistent.
    """
    if local_count < 1 or global_count < local_count or world_size < 1:
        raise ValueError("invalid DDP loss-scaling counts")
    return loss * (float(local_count * world_size) / float(global_count))


def _all_ranks_loss_finite(loss: torch.Tensor, accelerator: Accelerator) -> bool:
    """Return True iff every rank's loss is finite (checked before ``backward``).

    Accelerate's ``reduce`` only implements SUM across processes:
    ``accelerate.utils.operations._reduce_across_processes`` unconditionally
    all-reduces with ``ReduceOp.SUM`` and special-cases only ``reduction ==
    "mean"`` — any other string (including ``"min"``) silently returns the sum.
    A min-reduction over *finite* flags therefore degrades to a sum that only
    trips when every rank is non-finite. Summing *non-finite* flags instead
    gives exact any-rank detection under SUM semantics: the total is zero iff
    all ranks are finite. This runs on every rank, so all ranks agree and fail
    together before the corrupt gradient enters the DDP all-reduce.

    Args:
        loss: This rank's (already tail-scaled) training loss.
        accelerator: The DDP accelerator.

    Returns:
        True when the summed non-finite flag across ranks is zero.
    """
    nonfinite_flag = torch.tensor(
        0.0 if bool(torch.isfinite(loss).all()) else 1.0, device=accelerator.device
    )
    total_nonfinite = accelerator.reduce(nonfinite_flag, reduction="sum")
    return float(total_nonfinite.item()) == 0.0


def validate_gathered_validation(
    *,
    row_ids: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    expected_row_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify a gathered validation set covers every fixed row exactly once.

    Sorts the gathered rows by row ID and checks that they are unique and cover
    exactly ``expected_row_ids``. Never compute metrics from a partial gather:
    any duplicate or missing row is a hard error.

    Args:
        row_ids: Gathered (padding already masked out) validation row IDs.
        labels: Labels aligned row-for-row with ``row_ids``.
        logits: Logits aligned row-for-row with ``row_ids``.
        expected_row_ids: The complete set of fixed validation row IDs.

    Returns:
        ``(labels, logits)`` reordered by ascending row ID.

    Raises:
        ValueError: If any row appears twice, or the covered set differs from
            ``expected_row_ids``.
    """
    order = np.argsort(row_ids, kind="stable")
    row_ids = row_ids[order]
    labels = labels[order]
    logits = logits[order]

    unique_ids, counts = np.unique(row_ids, return_counts=True)
    if np.any(counts > 1):
        duplicates = unique_ids[counts > 1].tolist()
        raise ValueError(f"duplicate validation row IDs in gathered set: {duplicates}")
    expected_unique = np.unique(expected_row_ids)
    if unique_ids.shape[0] != expected_unique.shape[0] or not np.array_equal(
        unique_ids, expected_unique
    ):
        raise ValueError(
            "gathered validation rows do not cover the fixed validation set "
            f"({unique_ids.shape[0]} unique of {expected_unique.shape[0]} expected)"
        )
    return labels, logits


def validate_training_coverage(*, row_ids: np.ndarray, expected_row_ids: np.ndarray) -> None:
    """Require every fixed training row exactly once in a completed epoch."""
    unique_ids, counts = np.unique(row_ids, return_counts=True)
    if np.any(counts > 1):
        duplicates = unique_ids[counts > 1].tolist()
        raise ValueError(f"duplicate training row IDs in gathered set: {duplicates}")
    expected_unique = np.unique(expected_row_ids)
    if unique_ids.shape != expected_unique.shape or not np.array_equal(unique_ids, expected_unique):
        raise ValueError(
            "gathered training rows do not cover the fixed training set "
            f"({unique_ids.shape[0]} unique of {expected_unique.shape[0]} expected)"
        )


def _evaluate_distributed(
    model: nn.Module,
    val_loader: Iterable[Batch],
    accelerator: Accelerator,
    *,
    expected_row_ids: np.ndarray | None = None,
) -> EdgeMetrics:
    """Score the fixed validation set across all ranks and agree on the metrics.

    Each rank scores its slice; the slices are padded to a common length and
    gathered onto every rank. Every rank then masks the padding and verifies
    exact once-per-row coverage via :func:`validate_gathered_validation`, so a
    coverage violation raises identically on all ranks (never a rank-zero-only
    death that leaves the others deadlocked in the broadcast). Rank zero
    computes the metrics and the dict is broadcast so every rank makes the same
    best-epoch decision.

    Args:
        model: The scorer (restored to train mode before returning).
        val_loader: This rank's assembled validation batches (carry ``_row_id``).
        accelerator: The DDP accelerator.
        expected_row_ids: Complete fixed validation row-ID set. When ``None`` it is
            derived as ``arange`` over the gathered row count — a self-contained
            fallback that still catches duplicates and interior gaps; production
            binds the true set so truncation is caught too.

    Returns:
        The `EdgeMetrics`, identical on every rank.

    Raises:
        ValueError: On any duplicate or missing validation row (all ranks).
    """
    model.eval()
    row_id_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, accelerator.device)
            output = model(batch)
            logits = output["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            row_id_parts.append(batch["_row_id"].detach().to(torch.int64))
            label_parts.append(batch["label"].detach().to(torch.float32))
            logit_parts.append(logits.detach().to(torch.float32))
    model.train()

    device = accelerator.device
    local_row_ids = (
        torch.cat(row_id_parts)
        if row_id_parts
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    local_labels = (
        torch.cat(label_parts)
        if label_parts
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    local_logits = (
        torch.cat(logit_parts)
        if logit_parts
        else torch.empty(0, dtype=torch.float32, device=device)
    )

    padded_row_ids = accelerator.pad_across_processes(local_row_ids, dim=0, pad_index=-1)
    padded_labels = accelerator.pad_across_processes(local_labels, dim=0, pad_index=-1)
    padded_logits = accelerator.pad_across_processes(local_logits, dim=0, pad_index=0)

    gathered_row_ids = accelerator.gather(padded_row_ids)
    gathered_labels = accelerator.gather(padded_labels)
    gathered_logits = accelerator.gather(padded_logits)

    # Coverage must be validated symmetrically: accelerator.gather returns the
    # full gathered tensors on EVERY rank, so every rank masks the padding and
    # runs the same coverage check. A duplicate/missing row then raises
    # identically everywhere; if only rank zero validated, it would die before
    # the broadcast below and every other rank would block in it until the
    # NCCL watchdog killed the job, masking the real error.
    row_ids_np = gathered_row_ids.cpu().numpy()
    labels_np = gathered_labels.cpu().numpy()
    logits_np = gathered_logits.cpu().numpy()
    keep = row_ids_np >= 0
    row_ids_np = row_ids_np[keep]
    labels_np = labels_np[keep]
    logits_np = logits_np[keep]
    expected = (
        expected_row_ids
        if expected_row_ids is not None
        else np.arange(row_ids_np.shape[0], dtype=np.int64)
    )
    labels_sorted, logits_sorted = validate_gathered_validation(
        row_ids=row_ids_np,
        labels=labels_np,
        logits=logits_np,
        expected_row_ids=expected,
    )

    metrics_payload: list[dict[str, object] | None] = [None]
    if accelerator.is_main_process:
        probs = _stable_sigmoid(logits_sorted.astype(np.float64))
        metrics = compute_edge_metrics(labels_sorted.astype(np.float64), probs)
        metrics_payload[0] = asdict(metrics)

    broadcast_object_list(metrics_payload, from_process=0)
    payload = metrics_payload[0]
    if payload is None:  # pragma: no cover - broadcast always populates rank>0
        raise RuntimeError("distributed validation failed to broadcast metrics")
    return EdgeMetrics(**cast(dict[str, Any], payload))


def _stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid on a float64 array."""
    out = np.empty_like(logit, dtype=np.float64)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_l = np.exp(logit[~positive])
    out[~positive] = exp_l / (1.0 + exp_l)
    return out


def _batch_pair_counts(batch: Batch, world_size: int) -> tuple[int, int]:
    """Return ``(local_count, global_count)`` for a batch, tolerating plain batches."""
    if "_local_pair_count" in batch:
        local_count = int(batch["_local_pair_count"].item())
    else:
        local_count = int(batch["label"].shape[0])
    if "_global_pair_count" in batch:
        global_count = int(batch["_global_pair_count"].item())
    else:
        global_count = local_count * world_size
    return local_count, global_count


def train_ddp_loop(
    model: nn.Module,
    train_loader_factory: PackedLoaderFactory,
    val_loader: Iterable[Batch],
    cfg: Config,
    accelerator: Accelerator,
    *,
    warmup_steps: int,
    evaluate_fn: EvaluateFn = _evaluate_distributed,
    on_eval: OnEval | None = None,
) -> TrainResult:
    """Run fixed-epoch E2 DDP training and return rank-consistent metrics.

    Trains exactly ``cfg.optim.epochs`` epochs with a validation after every
    epoch. Patience is evaluated counterfactually only: the first epoch at which
    it would have stopped is recorded in ``TrainResult.counterfactual_stop_epoch``
    but the run never stops early. Tail batches are loss-scaled with
    :func:`scale_ddp_mean_loss`; a non-finite loss on any rank aborts all ranks;
    and every epoch boundary asserts all ranks ran the same number of steps.
    Only the main rank snapshots checkpoint state or runs ``on_eval``.

    Args:
        model: The scorer (not yet prepared by the accelerator).
        train_loader_factory: ``epoch -> iterable`` of this rank's assembled batches.
        val_loader: This rank's re-iterable assembled validation batches.
        cfg: The full training config.
        accelerator: DDP accelerator (see :func:`build_ddp_accelerator`).
        warmup_steps: Linear-warmup step count (then constant LR).
        evaluate_fn: Validation function; defaults to distributed validation.
            Unit tests inject a deterministic metric source.
        on_eval: Optional main-rank-only callback after each evaluation.

    Returns:
        The `TrainResult`, identical across ranks except for the main-rank-only
        checkpoint snapshots and rank-zero ``runtime_profile``.

    Raises:
        RuntimeError: On a non-finite loss or a per-rank step-count divergence.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    warmup = max(1, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup))
    )
    world_size = accelerator.num_processes
    use_cuda = accelerator.device.type == "cuda"

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] = {}
    best_metrics: EdgeMetrics | None = None
    best_epoch = 0
    last_metrics: EdgeMetrics | None = None
    evals_without_improvement = 0
    counterfactual_stop_epoch: int | None = None
    global_step = 0

    per_epoch_profiles: list[dict[str, object]] = []
    total_data_wait_seconds = 0.0
    total_train_wall_seconds = 0.0
    total_rank_local_pairs = 0
    total_global_pairs = 0
    total_local_tokens = 0
    total_validation_seconds = 0.0
    total_steps = 0
    local_peak_memory_gib = 0.0

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        local_loss_sum = 0.0
        epoch_steps = 0
        epoch_local_pairs = 0
        epoch_global_pairs = 0
        epoch_local_tokens = 0
        epoch_row_ids: list[torch.Tensor] = []
        epoch_data_wait_seconds = 0.0
        epoch_compute_seconds = 0.0
        cuda_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(accelerator.device)

        epoch_wall_start = time.monotonic()
        iterator = iter(train_loader_factory(epoch))
        while True:
            wait_start = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            epoch_data_wait_seconds += time.monotonic() - wait_start

            batch = _to_device(batch, accelerator.device)
            local_count, global_count = _batch_pair_counts(batch, world_size)

            start_event, end_event = _maybe_cuda_events(use_cuda)
            output = model(batch)
            local_mean_loss = output["loss"]
            loss = scale_ddp_mean_loss(
                local_mean_loss,
                local_count=local_count,
                global_count=global_count,
                world_size=world_size,
            )

            if not _all_ranks_loss_finite(loss, accelerator):
                raise RuntimeError(f"non-finite training loss on at least one rank (epoch {epoch})")

            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            if start_event is not None and end_event is not None:
                end_event.record()  # type: ignore[no-untyped-call]
                cuda_event_pairs.append((start_event, end_event))

            global_step += 1
            epoch_steps += 1
            epoch_local_pairs += local_count
            epoch_global_pairs += global_count
            local_loss_sum += float(local_mean_loss.detach().float().item()) * local_count
            if "_row_id" not in batch:
                raise ValueError("training batch is missing required _row_id coverage metadata")
            epoch_row_ids.append(batch["_row_id"].detach().to(torch.int64))
            if "len_a" in batch and "len_b" in batch:
                epoch_local_tokens += int(
                    (batch["len_a"].to(torch.int64) + batch["len_b"].to(torch.int64)).sum().item()
                )

        if use_cuda:
            torch.cuda.synchronize(accelerator.device)
            epoch_compute_seconds = sum(
                start.elapsed_time(end) / 1000.0  # type: ignore[no-untyped-call]
                for start, end in cuda_event_pairs
            )

        # Epoch boundary: every rank must have run the same number of steps.
        step_counts = accelerator.gather(torch.tensor([epoch_steps], device=accelerator.device))
        if int(step_counts.min().item()) != int(step_counts.max().item()):
            raise RuntimeError(
                f"rank step-count divergence at epoch {epoch}: "
                f"{step_counts.tolist()} (a rank would hang on the next collective)"
            )

        epoch_wall_seconds = time.monotonic() - epoch_wall_start
        total_train_wall_seconds += epoch_wall_seconds
        total_data_wait_seconds += epoch_data_wait_seconds

        # Cross-rank pair coverage is a hard, symmetric gate: ranks must agree on
        # the planned total, and the gathered row IDs must be the exact fixed set.
        planned_totals = accelerator.gather(
            torch.tensor([epoch_global_pairs], device=accelerator.device, dtype=torch.int64)
        )
        if int(planned_totals.min().item()) != int(planned_totals.max().item()):
            raise RuntimeError(
                f"rank global-pair plan divergence at epoch {epoch}: {planned_totals.tolist()}"
            )
        expected_global_pairs = int(planned_totals[0].item())
        gathered_local_pairs = int(
            accelerator.reduce(
                torch.tensor(epoch_local_pairs, device=accelerator.device), reduction="sum"
            ).item()
        )
        if gathered_local_pairs != expected_global_pairs:
            raise RuntimeError(
                f"cross-rank training pair-count mismatch at epoch {epoch}: "
                f"gathered={gathered_local_pairs}, planned={expected_global_pairs}"
            )
        local_rows = (
            torch.cat(epoch_row_ids)
            if epoch_row_ids
            else torch.empty(0, dtype=torch.int64, device=accelerator.device)
        )
        padded_rows = accelerator.pad_across_processes(local_rows, dim=0, pad_index=-1)
        gathered_rows = accelerator.gather(padded_rows)
        row_ids_np = gathered_rows[gathered_rows >= 0].cpu().numpy()
        validate_training_coverage(
            row_ids=row_ids_np,
            expected_row_ids=np.arange(expected_global_pairs, dtype=np.int64),
        )
        total_rank_local_pairs += epoch_local_pairs
        total_global_pairs += expected_global_pairs
        total_local_tokens += epoch_local_tokens
        total_steps += epoch_steps

        if use_cuda:
            epoch_peak_gib = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
            local_peak_memory_gib = max(local_peak_memory_gib, epoch_peak_gib)

        global_loss_stats = accelerator.reduce(
            torch.tensor(
                [local_loss_sum, float(epoch_local_pairs)],
                device=accelerator.device,
                dtype=torch.float64,
            ),
            reduction="sum",
        )
        global_sample_count = float(global_loss_stats[1].item())
        train_loss = (
            float(global_loss_stats[0].item()) / global_sample_count
            if global_sample_count > 0
            else float("nan")
        )
        validation_start = time.monotonic()
        metrics = evaluate_fn(model, val_loader, accelerator)
        if use_cuda:
            torch.cuda.synchronize(accelerator.device)
        local_validation_seconds = time.monotonic() - validation_start
        validation_times = accelerator.gather(
            torch.tensor([local_validation_seconds], device=accelerator.device, dtype=torch.float64)
        )
        validation_seconds = float(validation_times.max().item())
        total_validation_seconds += validation_seconds
        last_metrics = metrics
        entry: dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": metrics.auroc,
            "val_auprc": metrics.auprc,
        }
        history.append(entry)

        improved = best_metrics is None or metrics.auprc > best_metrics.auprc
        if improved:
            best_metrics = metrics
            best_epoch = epoch
            evals_without_improvement = 0
            if accelerator.is_main_process:
                best_state = _cpu_state_dict(accelerator, model)
        else:
            evals_without_improvement += 1
            if evals_without_improvement >= cfg.eval.patience and counterfactual_stop_epoch is None:
                counterfactual_stop_epoch = epoch

        logger.info(
            "ddp eval epoch %d/%d: train_loss %.4f val_auroc %.4f val_auprc %.4f%s",
            epoch,
            cfg.optim.epochs,
            train_loss,
            metrics.auroc,
            metrics.auprc,
            " (new best)" if improved else "",
        )
        if accelerator.is_main_process and on_eval is not None:
            on_eval(entry, improved, metrics)

        per_epoch_profiles.append(
            {
                "epoch": epoch,
                "steps": epoch_steps,
                "global_pairs": epoch_global_pairs,
                "local_pairs": epoch_local_pairs,
                "local_tokens": epoch_local_tokens,
                "wall_seconds": epoch_wall_seconds,
                "data_wait_seconds": epoch_data_wait_seconds,
                "compute_seconds": epoch_compute_seconds,
                "validation_seconds": validation_seconds,
            }
        )

    if best_metrics is None or last_metrics is None:
        raise RuntimeError("DDP training ended without a single evaluation")

    peak_memory_tensor = accelerator.gather(
        torch.tensor([local_peak_memory_gib], device=accelerator.device, dtype=torch.float32)
    )
    peak_memory_gib_per_rank = [float(value) for value in peak_memory_tensor.tolist()]
    rank_totals = accelerator.gather(
        torch.tensor(
            [
                float(total_rank_local_pairs),
                float(total_steps),
                float(total_local_tokens),
                total_train_wall_seconds,
                total_data_wait_seconds,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
    ).reshape(world_size, 5)
    per_rank: list[dict[str, object]] = []
    data_wait_fractions: list[float] = []
    for rank, values in enumerate(rank_totals.tolist()):
        pairs, steps, tokens, wall_seconds, data_wait_seconds = values
        data_wait_fraction = data_wait_seconds / wall_seconds if wall_seconds > 0 else 0.0
        data_wait_fractions.append(data_wait_fraction)
        per_rank.append(
            {
                "rank": rank,
                "pairs": int(pairs),
                "batches": int(steps),
                "steps": int(steps),
                "tokens": int(tokens),
                "train_wall_seconds": wall_seconds,
                "data_wait_seconds": data_wait_seconds,
                "pairs_per_second": pairs / wall_seconds if wall_seconds > 0 else 0.0,
                "tokens_per_second": tokens / wall_seconds if wall_seconds > 0 else 0.0,
            }
        )
    slowest_train_seconds = max(float(values[3]) for values in rank_totals.tolist())
    global_tokens = int(rank_totals[:, 2].sum().item())
    steady_state_data_wait_fraction = max(data_wait_fractions, default=0.0)
    runtime_profile: dict[str, object] = {
        "epochs_completed": cfg.optim.epochs,
        "validations_completed": cfg.optim.epochs,
        "peak_memory_gib_per_rank": peak_memory_gib_per_rank,
        "steady_state_data_wait_fraction": steady_state_data_wait_fraction,
        "training_coverage_exact": True,
        "validation_coverage_exact": True,
        "feature_cache_hit_rate": 1.0,
        "counterfactual_stop_epoch": counterfactual_stop_epoch,
        "per_rank": per_rank,
        "global_pairs": total_global_pairs,
        "global_pairs_per_second": (
            total_global_pairs / slowest_train_seconds if slowest_train_seconds > 0 else 0.0
        ),
        "global_tokens": global_tokens,
        "global_tokens_per_second": (
            global_tokens / slowest_train_seconds if slowest_train_seconds > 0 else 0.0
        ),
        "validation_seconds": total_validation_seconds,
        "per_epoch": per_epoch_profiles,
    }

    if accelerator.is_main_process:
        last_state = _cpu_state_dict(accelerator, model)
        if not best_state:
            best_state = last_state
    else:
        last_state = {}

    return TrainResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        last_state_dict=last_state,
        last_epoch=cfg.optim.epochs,
        last_val_metrics=last_metrics,
        history=history,
        stopped_early=False,
        counterfactual_stop_epoch=counterfactual_stop_epoch,
        runtime_profile=runtime_profile,
    )


def _maybe_cuda_events(
    use_cuda: bool,
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    """Return a started/pending CUDA event pair, or ``(None, None)`` off CUDA."""
    if not use_cuda:
        return None, None
    start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start_event.record()  # type: ignore[no-untyped-call]
    return start_event, end_event


# --------------------------------------------------------------------------- DDP worker modes


def _write_json_rank_zero(accelerator: Accelerator, path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as pretty JSON from the main rank only (atomically)."""
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _cycle_assembled_batches(factory: PackedLoaderFactory) -> Iterator[Batch]:
    """Repeatedly traverse epoch 1 lazily without retaining assembled GPU batches."""
    while True:
        yielded = False
        for batch in factory(1):
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("packed loader factory produced no batches for the probe")


def _is_oom_error(error: RuntimeError) -> bool:
    """Return whether ``error`` is a candidate-local accelerator OOM."""
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _emit_probe_candidate_failure(kind: str, message: str) -> None:
    """Emit the subprocess contract for a failed probe candidate."""
    payload = json.dumps({"kind": kind, "message": message}, separators=(",", ":"))
    sys.stderr.write(f"E2_PROBE_CANDIDATE_FAILURE:{payload}\n")
    sys.stderr.flush()


def _run_timed_epoch_probe(accelerator: Accelerator, operation: Callable[[], T]) -> tuple[T, float]:
    """Run one epoch probe and return the slowest rank's full train+validation time."""
    accelerator.wait_for_everyone()
    use_cuda = accelerator.device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(accelerator.device)
    started = time.monotonic()
    result = operation()
    if use_cuda:
        torch.cuda.synchronize(accelerator.device)
    local_elapsed = time.monotonic() - started
    elapsed_by_rank = accelerator.gather(
        torch.tensor([local_elapsed], device=accelerator.device, dtype=torch.float64)
    )
    return result, float(elapsed_by_rank.max().item())


def _run_probe_mode(
    model: nn.Module,
    factory: PackedLoaderFactory,
    cfg: Config,
    accelerator: Accelerator,
    *,
    token_budget_per_rank: int,
    profile_output: Path,
) -> None:
    """Run warm-up + timed steps and write one rank-zero ``ProbeResult`` JSON."""
    runtime = cfg.runtime
    if runtime is None:
        raise ValueError("probe mode requires a configured cfg.runtime")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    world_size = accelerator.num_processes
    use_cuda = accelerator.device.type == "cuda"
    warmup = runtime.probe_warmup_steps
    timed = runtime.probe_timed_steps

    model.train()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    timed_global_pairs = 0
    timed_start: float | None = None
    failure: str | None = None
    iterator = _cycle_assembled_batches(factory)
    for step in range(warmup + timed):
        if step == warmup:
            accelerator.wait_for_everyone()
            if use_cuda:
                torch.cuda.synchronize(accelerator.device)
            timed_start = time.monotonic()
        batch = _to_device(next(iterator), accelerator.device)
        local_count, global_count = _batch_pair_counts(batch, world_size)
        loss: torch.Tensor | None = None
        local_failure: tuple[str, str] | None = None
        try:
            loss = scale_ddp_mean_loss(
                model(batch)["loss"],
                local_count=local_count,
                global_count=global_count,
                world_size=world_size,
            )
            if not bool(torch.isfinite(loss).all()):
                local_failure = ("nonfinite", "non-finite probe loss")
        except RuntimeError as error:
            if not _is_oom_error(error):
                raise
            local_failure = ("oom", str(error))

        failed_ranks = accelerator.reduce(
            torch.tensor(
                1 if local_failure is not None else 0,
                device=accelerator.device,
                dtype=torch.int64,
            ),
            reduction="sum",
        )
        if int(failed_ranks.item()) > 0:
            if use_cuda and local_failure is not None:
                torch.cuda.empty_cache()
            kind, message = local_failure or (
                "oom",
                "probe candidate failed on another rank",
            )
            _emit_probe_candidate_failure(kind, message)
            raise RuntimeError(message)
        if loss is None:  # pragma: no cover - collective failure breaks above
            raise RuntimeError("probe forward produced no loss")
        optimizer.zero_grad()
        # From this point onward DDP collectives may already be in flight. Any
        # exception must escape the worker immediately; serializing it and entering
        # later gathers on just one rank can deadlock the remaining ranks.
        try:
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
        except RuntimeError as error:
            if _is_oom_error(error):
                _emit_probe_candidate_failure("oom", str(error))
            raise
        if step >= warmup:
            timed_global_pairs += global_count
    if use_cuda:
        torch.cuda.synchronize(accelerator.device)

    local_elapsed = time.monotonic() - timed_start if timed_start is not None else 0.0
    elapsed_tensor = accelerator.gather(
        torch.tensor([local_elapsed], device=accelerator.device, dtype=torch.float64)
    )
    elapsed = float(elapsed_tensor.max().item())
    pair_counts = accelerator.gather(
        torch.tensor([timed_global_pairs], device=accelerator.device, dtype=torch.int64)
    )
    if int(pair_counts.min().item()) != int(pair_counts.max().item()):
        failure = f"probe timed global-pair count diverged across ranks: {pair_counts.tolist()}"
    timed_global_pairs = int(pair_counts.min().item())
    local_peak_gib = (
        torch.cuda.max_memory_allocated(accelerator.device) / (1024**3) if use_cuda else 0.0
    )
    peak_tensor = accelerator.gather(
        torch.tensor([local_peak_gib], device=accelerator.device, dtype=torch.float32)
    )
    peak_memory_gib = float(peak_tensor.max().item())
    # timed_global_pairs already counts each global batch once (it is the cross-rank
    # total), so throughput needs no further reduction.
    throughput = timed_global_pairs / elapsed if elapsed > 0 and failure is None else 0.0
    probe = ProbeResult(
        token_budget=token_budget_per_rank,
        valid=failure is None,
        global_pairs_per_second=throughput,
        peak_memory_gib=peak_memory_gib,
        failure=failure,
    )
    _write_json_rank_zero(accelerator, profile_output, probe.to_dict())
    logger.info("probe complete: %s", probe.to_dict())


def _run_ddp_worker(cfg: Config, args: CliArgs) -> None:
    """Dispatch an ``accelerate launch`` worker to the requested DDP mode.

    Shared setup (seeded accelerator, verified data, packed feature table, per-rank
    packed loaders) is done once, then ``probe`` / ``epoch-probe`` / ``train`` run
    their pinned work and rank zero writes the profile/artifact contract.
    """
    if args.pack_dir is None or args.token_budget_per_rank is None or args.profile_output is None:
        raise ValueError(
            "DDP worker modes require --pack-dir, --token-budget-per-rank, and --profile-output"
        )
    if cfg.model.family != "v3_1":
        raise ValueError(f"DDP worker modes only support the v3_1 family, got {cfg.model.family!r}")

    accelerator = build_ddp_accelerator(cfg.mixed_precision)
    set_seed(cfg.seed)
    logger.info(
        "ddp worker mode=%s rank=%d/%d device=%s",
        args.ddp_mode,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )

    assembled = assemble_data(cfg)
    table = PackedFeatureTable.from_pack(args.pack_dir, accelerator.device)
    factory, val_loader, warmup_steps = _build_packed_v3_1_loaders(
        cfg,
        assembled,
        table,
        token_budget_per_rank=args.token_budget_per_rank,
        process_index=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    model = build_model(cfg)

    if args.ddp_mode == "probe":
        _run_probe_mode(
            model,
            factory,
            cfg,
            accelerator,
            token_budget_per_rank=args.token_budget_per_rank,
            profile_output=args.profile_output,
        )
        return

    num_val_rows = len(assembled.benchmark.split.val_pairs.pairs)
    evaluate_fn = cast(
        EvaluateFn,
        functools.partial(
            _evaluate_distributed,
            expected_row_ids=np.arange(num_val_rows, dtype=np.int64),
        ),
    )

    if args.ddp_mode == "epoch-probe":
        one_epoch_cfg = replace(cfg, optim=replace(cfg.optim, epochs=1))
        result, elapsed = _run_timed_epoch_probe(
            accelerator,
            lambda: train_ddp_loop(
                model,
                factory,
                val_loader,
                one_epoch_cfg,
                accelerator,
                warmup_steps=warmup_steps,
                evaluate_fn=evaluate_fn,
            ),
        )
        _write_json_rank_zero(
            accelerator,
            args.profile_output,
            {"epoch_seconds": elapsed, "runtime_profile": result.runtime_profile},
        )
        logger.info("epoch-probe complete: %.2fs", elapsed)
        return

    # args.ddp_mode == "train": full fixed-epoch run + formal artifacts.
    model_kwargs = resolve_model_kwargs(cfg.model)
    result = train_ddp_loop(
        model,
        factory,
        val_loader,
        cfg,
        accelerator,
        warmup_steps=warmup_steps,
        evaluate_fn=evaluate_fn,
    )
    if accelerator.is_main_process:
        write_outputs(result, cfg, model_kwargs, assembled.dropped_pair_counts)
    _write_json_rank_zero(accelerator, args.profile_output, result.runtime_profile)
    logger.info(
        "ddp train complete: best epoch %d val AUPRC %.4f (counterfactual_stop_epoch=%s)",
        result.best_epoch,
        result.best_val_metrics.auprc,
        result.counterfactual_stop_epoch,
    )


# --------------------------------------------------------------------------- CLI entry point


def main(argv: Sequence[str] | None = None) -> None:
    """Run the training CLI end to end.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    if args.ddp_mode is not None:
        _run_ddp_worker(cfg, args)
        return
    if args.max_steps is not None:
        logger.warning(
            "--max-steps %d is a DEBUG-ONLY flag for bounded smoke runs; "
            "never use it for real training",
            args.max_steps,
        )

    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    logger.info(
        "training %s on %s (mixed_precision=%s, seed=%d)",
        cfg.model.family,
        accelerator.device,
        cfg.mixed_precision,
        cfg.seed,
    )

    assembled = assemble_data(cfg)
    logger.info(
        "assembled data: %d training positives (%s mode), dropped pairs %s",
        len(assembled.training_positives),
        cfg.data.train_positives,
        assembled.dropped_pair_counts,
    )
    model = build_model(cfg)
    model_kwargs = resolve_model_kwargs(cfg.model)

    if cfg.model.family == "v3_1":
        factory, val_loader = _build_v3_1_loaders(cfg, assembled)
    else:
        factory, val_loader = _build_f0_loaders(cfg, assembled)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    config_dict = config_to_dict(cfg)

    def on_eval(entry: dict[str, float], improved: bool, metrics: EdgeMetrics) -> None:
        """Append a metrics line and persist checkpoints incrementally."""
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        state = _cpu_state_dict(accelerator, model)
        epoch = int(entry["epoch"])
        torch.save(
            _checkpoint_payload(state, cfg, model_kwargs, epoch, metrics, config_dict),
            cfg.output_dir / "last.pt",
        )
        if improved:
            torch.save(
                _checkpoint_payload(state, cfg, model_kwargs, epoch, metrics, config_dict),
                cfg.output_dir / "best.pt",
            )

    result = train_loop(
        model,
        factory,
        val_loader,
        cfg,
        accelerator,
        max_steps=args.max_steps,
        on_eval=on_eval,
    )
    write_outputs(result, cfg, model_kwargs, assembled.dropped_pair_counts)
    logger.info(
        "training complete: best epoch %d val AUPRC %.4f (stopped_early=%s)",
        result.best_epoch,
        result.best_val_metrics.auprc,
        result.stopped_early,
    )


if __name__ == "__main__":
    main()
