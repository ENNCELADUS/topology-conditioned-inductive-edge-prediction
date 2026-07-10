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
import hashlib
import json
import logging
import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from src.data.artifacts import ArtifactVerificationError, Benchmark, load_benchmark
from src.data.features import FeatureStore, build_f0_matrix
from src.data.pairs import (
    LengthBucketedBatchSampler,
    NegativeSampler,
    SharedEpochTokenPairDataset,
    TokenPairDataset,
    collate_token_pairs,
)
from src.data.partition import build_g_struct, derive_partition
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.model.B0 import BEST_V3_1_CONFIG, V3_1
from src.model.b0_alt import F0PairMLP

logger = logging.getLogger(__name__)

MODEL_FAMILIES = ("v3_1", "f0_mlp")
TRAIN_POSITIVES_MODES = ("train_plus", "e_sup")
MIXED_PRECISION_MODES = ("no", "bf16")

Batch = dict[str, torch.Tensor]
LoaderFactory = Callable[[int], Iterable[Batch]]
OnEval = Callable[[dict[str, float], bool, EdgeMetrics], None]


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
        negative_ratio: Negatives sampled per positive, per epoch.
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


@dataclass(frozen=True)
class CliArgs:
    """Parsed command-line arguments.

    Attributes:
        config: Path to the YAML config file.
        seed: Optional seed override (wins over the config).
        output_dir: Optional output-dir override (wins over the config).
        max_steps: DEBUG ONLY — stop after this many optimizer steps.
    """

    config: Path
    seed: int | None
    output_dir: Path | None
    max_steps: int | None


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
        ("model", "data", "optim", "eval", "seed", "output_dir", "mixed_precision"),
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

    return Config(
        model=model,
        data=data,
        optim=optim,
        eval=eval_cfg,
        seed=_as_int(_require(raw, "seed", ""), "seed"),
        output_dir=Path(_as_str(_require(raw, "output_dir", ""), "output_dir")),
        mixed_precision=mixed_precision,
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
    namespace = parser.parse_args(argv)
    return CliArgs(
        config=namespace.config,
        seed=namespace.seed,
        output_dir=namespace.output_dir,
        max_steps=namespace.max_steps,
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
    """

    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_metrics: EdgeMetrics
    last_state_dict: dict[str, torch.Tensor]
    last_epoch: int
    last_val_metrics: EdgeMetrics
    history: list[dict[str, float]]
    stopped_early: bool


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor of a batch dict to `device`."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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
        train_loader_factory: ``epoch -> iterable of batch dicts`` (fresh negatives
            and shuffling per epoch happen inside the factory).
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


def _v3_loader_options(num_workers: int) -> dict[str, object]:
    """Return the pinned DataLoader options for raw-token V3.1 batches."""
    options: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 4
    return options


class _ReusableEpochDataLoader(DataLoader[Batch]):
    """One DataLoader whose shared dataset is replaced only between exhausted epochs."""

    def __init__(
        self,
        dataset: SharedEpochTokenPairDataset,
        batch_sampler: LengthBucketedBatchSampler,
        loader_options: dict[str, object],
    ) -> None:
        """Bind shared worker data and the parent-process mutable batch sampler."""
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_token_pairs,
            **cast(Any, loader_options),
        )
        self._epoch_dataset = dataset
        self._epoch_batch_sampler = batch_sampler
        self._epoch_exhausted = True
        self._iteration_active = False

    def ensure_epoch_replaceable(self) -> None:
        """Raise unless the preceding epoch iterator reached true exhaustion."""
        if not self._epoch_exhausted or self._iteration_active:
            raise RuntimeError(
                "cannot replace epoch data before the previous iterator is exhausted"
            )

    def prepare_epoch(
        self,
        pairs: Sequence[tuple[str, str]],
        labels: Sequence[int],
        lengths: Sequence[tuple[int, int]],
        *,
        epoch: int,
    ) -> None:
        """Publish one epoch after the preceding loader iterator is exhausted."""
        self.ensure_epoch_replaceable()
        if len(lengths) != len(self._epoch_dataset):
            raise ValueError("epoch lengths must match the fixed dataset capacity")
        self._epoch_dataset.replace_epoch(pairs, labels)
        self._epoch_batch_sampler.replace_epoch(lengths, epoch=epoch)
        self._epoch_exhausted = False

    def __iter__(self) -> Iterator[Batch]:  # type: ignore[override]
        """Yield one prepared epoch and mark it replaceable only at true exhaustion."""
        if self._epoch_exhausted:
            raise RuntimeError("prepare_epoch must be called before iterating the training loader")
        if self._iteration_active:
            raise RuntimeError("the training loader already has an active epoch iterator")
        self._iteration_active = True
        iterator = super().__iter__()
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                self._iteration_active = False
                self._epoch_exhausted = True
                return


def _build_v3_1_loaders(
    cfg: Config, assembled: AssembledData
) -> tuple[LoaderFactory, Iterable[Batch]]:
    """Build the token-sequence train-loader factory and val loader (v3_1 path)."""
    cached_nodes = assembled.store.preload(assembled.operative_node_ids)
    logger.info(
        "preloaded %d operative node tensors (%.2f GiB) into host memory",
        cached_nodes,
        assembled.store.cached_bytes / float(1024**3),
    )
    store = assembled.store
    sampler = _build_negative_sampler(assembled)
    positives = assembled.training_positives
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
        **cast(Any, _v3_loader_options(cfg.data.num_workers)),
    )

    epoch_size = len(positives) * (1 + cfg.data.negative_ratio)
    train_dataset = SharedEpochTokenPairDataset(
        assembled.operative_node_ids,
        epoch_size,
        store,
    )
    train_batch_sampler = LengthBucketedBatchSampler(
        [(1, 1)] * epoch_size,
        token_budget=cfg.data.token_budget,
        shuffle=True,
        seed=cfg.seed,
        epoch=0,
    )
    train_loader = _ReusableEpochDataLoader(
        train_dataset,
        train_batch_sampler,
        _v3_loader_options(cfg.data.num_workers),
    )

    def factory(epoch: int) -> DataLoader[Batch]:
        train_loader.ensure_epoch_replaceable()
        negatives = sampler.sample(
            positives, ratio=cfg.data.negative_ratio, seed=cfg.seed, epoch=epoch
        )
        pairs, labels = _shuffled_pairs_and_labels(positives, negatives, cfg.seed, epoch)
        lengths = lengths_for(pairs)
        train_loader.prepare_epoch(
            pairs,
            labels,
            lengths,
            epoch=epoch,
        )
        return train_loader

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


# --------------------------------------------------------------------------- CLI entry point


def main(argv: Sequence[str] | None = None) -> None:
    """Run the training CLI end to end.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
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
