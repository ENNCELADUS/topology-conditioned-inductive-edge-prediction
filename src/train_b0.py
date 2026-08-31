"""Training CLI for the frozen B0 pairwise scorer (family ``v3_1``).

``model.family: f0_mlp`` remains a schema-valid ``data:``/``model:`` value (kept for
config schema symmetry) but has no buildable model: ``src/model/b0_alt.py``
(``F0PairMLP``, the B0-alt baseline) was removed 2026-08-03 by owner decision;
:func:`build_model` raises for this family. See
``docs/results/E2-pair-to-topology-gap.md`` for the closed result it produced.

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
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list, gather_object, set_seed
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
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
    BUCKET_BOUNDARIES,
    LengthBucketedBatchSampler,
    NegativeSampler,
    TokenPairDataset,
    collate_token_pairs,
)
from src.data.partition import build_g_struct
from src.data.val_region import (
    Pair,
    ValRegionParams,
    ValRegionSplit,
    derive_val_region_split,
    val_ball_union_universe,
)
from src.distill.artifacts import (
    KDContextBank,
    KDContextTargets,
    KDRowTargets,
    load_kd_context_targets,
    load_kd_targets,
)
from src.distill.config import DistillConfig
from src.distill.losses import (
    kd_dist_loss,
    kd_gram_loss,
    kd_logit_loss,
    kd_rank_loss,
    kd_rep_loss,
)
from src.e2_pipeline import ProbeResult
from src.eval.checkpoint_selection import (
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.eval.val_topology import (
    ValTopologyReference,
    ValTopologyResult,
    build_val_topology_reference,
    val_region_topology_metrics,
)
from src.model.egostitch.classifier.b0_v31 import BEST_V3_1_CONFIG, V3_1
from src.model.egostitch.classifier.topo_gen import TopoGenBase

logger = logging.getLogger(__name__)

MODEL_FAMILIES = ("v3_1", "f0_mlp")
MIXED_PRECISION_MODES = ("no", "bf16")

Batch = dict[str, torch.Tensor]
LoaderFactory = Callable[[int], Iterable[Batch]]
PackedLoaderFactory = Callable[[int], Iterable[Batch]]
OnEval = Callable[[dict[str, object], bool, EdgeMetrics], None]


class ValidationOutcome(NamedTuple):
    """One validation pass: edge metrics, optional topology metrics, optional KD diagnostics."""

    metrics: EdgeMetrics
    topology: ValTopologyResult | None
    kd: dict[str, float] | None = None
    task_loss: float | None = None


EvaluateFn = Callable[[nn.Module, Iterable[Batch], Accelerator], ValidationOutcome]
DDP_MODES = ("probe", "epoch-probe", "train")
T = TypeVar("T")


# --------------------------------------------------------------------------- config schema


@dataclass(frozen=True)
class ModelConfig:
    """The ``model:`` config section.

    Attributes:
        family: Scorer family, one of ``v3_1`` or ``f0_mlp``. ``f0_mlp`` (the B0-alt
            baseline) has no buildable model: :func:`build_model` raises for it.
        config: Model constructor kwargs. Empty for ``v3_1`` means
            :data:`~src.model.egostitch.classifier.b0_v31.BEST_V3_1_CONFIG`; for
            ``f0_mlp`` this is stored but never consumed (see above).
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
        negative_ratio: Negatives sampled per positive, per epoch on the F0-MLP path.
        token_budget: Per-batch token budget on the ``v3_1`` path.
        batch_pairs: Per-batch pair count on the ``f0_mlp`` path.
        num_workers: DataLoader workers on the ``v3_1`` path.
        f0_cache: Cache path for the F0 mean-pooled feature matrix.
        expected_missing_features: Exact set of graph nodes expected to lack
            features (the feature-coverage gate fails on any drift).
    """

    root: Path
    strategy: str
    negative_ratio: int
    token_budget: int
    batch_pairs: int
    num_workers: int
    f0_cache: Path
    expected_missing_features: list[str]


@dataclass(frozen=True)
class SchedulerConfig:
    """The optional ``optim.scheduler:`` block.

    Only ``onecycle`` is supported. When this block is absent the trainer keeps
    its historical schedule: linear warmup over ``optim.warmup_steps`` then a
    constant LR.

    Attributes:
        type: Scheduler name; must be ``"onecycle"``.
        max_lr: Peak LR at the end of the warmup phase.
        pct_start: Fraction of total steps spent ramping up to `max_lr`.
        div_factor: Initial LR is ``max_lr / div_factor``.
        final_div_factor: Final LR is ``max_lr / div_factor / final_div_factor``.
        anneal_strategy: ``"cos"`` or ``"linear"``.
    """

    type: str
    max_lr: float
    pct_start: float
    div_factor: float
    final_div_factor: float
    anneal_strategy: str


@dataclass(frozen=True)
class OptimConfig:
    """The ``optim:`` config section.

    Attributes:
        lr: AdamW learning rate (post-warmup constant). With a ``onecycle``
            scheduler the schedule is driven by ``scheduler.max_lr`` instead.
        weight_decay: AdamW weight decay.
        epochs: Maximum number of epochs.
        warmup_steps: Linear LR warmup steps (then constant). Ignored when a
            ``onecycle`` scheduler is configured — OneCycle owns its own warmup
            via ``scheduler.pct_start``.
        grad_clip: Gradient-norm clip value; 0 disables clipping.
        scheduler: Optional LR-schedule override; `None` keeps warmup+constant.
    """

    lr: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    grad_clip: float
    scheduler: SchedulerConfig | None = None


@dataclass(frozen=True)
class EvalConfig:
    """The ``eval:`` config section.

    Attributes:
        patience: Early stop after this many evals without val-AUPRC improvement.
        eval_every: Evaluate every N epochs.
        topology_every: Run the V_val topology pass only on epochs divisible by
            N (the final epoch always runs it); classification metrics keep the
            ``eval_every`` cadence.
        classification_only: Skip topology validation, select the checkpoint by
            validation AUPRC, and stop when ``patience`` is exhausted.
    """

    patience: int
    eval_every: int
    topology_every: int = 1
    classification_only: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime contract for formal distributed training.

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
        runtime: Optional DDP runtime contract.
        distill: Optional B1 KD section; ``None`` or all-zero weights keep the
            plain supervised protocol.
    """

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    eval: EvalConfig
    seed: int
    output_dir: Path
    mixed_precision: str
    runtime: RuntimeConfig | None = None
    distill: DistillConfig | None = None


@dataclass(frozen=True)
class CliArgs:
    """Parsed command-line arguments.

    Attributes:
        config: Path to the YAML config file.
        seed: Optional seed override (wins over the config).
        output_dir: Optional output-dir override (wins over the config).
        max_steps: DEBUG ONLY — stop after this many optimizer steps.
        ddp_mode: Internal multi-H20 worker mode (``probe``/``epoch-probe``/``train``)
            launched by ``accelerate launch``; ``None`` selects the direct debug
            single-process path. Requires ``pack_dir``, ``token_budget_per_rank``,
            and ``profile_output`` when set.
        pack_dir: Packed-feature directory the DDP worker loads onto its device.
        token_budget_per_rank: Per-rank token budget for the distributed batch plan.
    profile_output: Path the rank-zero worker writes its JSON profile artifact to.
        resume_attempt: Prior attempt directory to resume at its completed epoch.
    """

    config: Path
    seed: int | None
    output_dir: Path | None
    max_steps: int | None
    ddp_mode: str | None = None
    pack_dir: Path | None = None
    token_budget_per_rank: int | None = None
    profile_output: Path | None = None
    resume_attempt: Path | None = None


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


def _as_bool(value: object, name: str) -> bool:
    """Validate that a config value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"config key '{name}' must be a bool, got {value!r}")
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


SCHEDULER_TYPES = ("onecycle",)
ANNEAL_STRATEGIES = ("cos", "linear")


def _parse_scheduler(value: object) -> SchedulerConfig | None:
    """Parse the optional ``optim.scheduler`` block.

    Args:
        value: The raw ``optim.scheduler`` value; `None`/absent selects the
            historical warmup-then-constant schedule.

    Returns:
        The validated `SchedulerConfig`, or `None` when no block is present.

    Raises:
        ValueError: On an unknown scheduler type, an unknown anneal strategy, an
            unknown key, or an out-of-range numeric field.
    """
    if value is None:
        return None

    raw = _as_mapping(value, "optim.scheduler")
    _check_no_unknown_keys(
        raw,
        ("type", "max_lr", "pct_start", "div_factor", "final_div_factor", "anneal_strategy"),
        "optim.scheduler",
    )

    scheduler_type = _as_str(_require(raw, "type", "optim.scheduler."), "optim.scheduler.type")
    if scheduler_type not in SCHEDULER_TYPES:
        raise ValueError(
            f"optim.scheduler.type must be one of {list(SCHEDULER_TYPES)}, got {scheduler_type!r}"
        )

    anneal_strategy = _as_str(
        _require(raw, "anneal_strategy", "optim.scheduler."), "optim.scheduler.anneal_strategy"
    )
    if anneal_strategy not in ANNEAL_STRATEGIES:
        raise ValueError(
            f"optim.scheduler.anneal_strategy must be one of {list(ANNEAL_STRATEGIES)}, "
            f"got {anneal_strategy!r}"
        )

    max_lr = _as_float(_require(raw, "max_lr", "optim.scheduler."), "optim.scheduler.max_lr")
    pct_start = _as_float(
        _require(raw, "pct_start", "optim.scheduler."), "optim.scheduler.pct_start"
    )
    div_factor = _as_float(
        _require(raw, "div_factor", "optim.scheduler."), "optim.scheduler.div_factor"
    )
    final_div_factor = _as_float(
        _require(raw, "final_div_factor", "optim.scheduler."), "optim.scheduler.final_div_factor"
    )

    if max_lr <= 0.0:
        raise ValueError(f"optim.scheduler.max_lr must be > 0, got {max_lr}")
    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"optim.scheduler.pct_start must be in (0.0, 1.0), got {pct_start}")
    if div_factor <= 0.0:
        raise ValueError(f"optim.scheduler.div_factor must be > 0, got {div_factor}")
    if final_div_factor <= 0.0:
        raise ValueError(f"optim.scheduler.final_div_factor must be > 0, got {final_div_factor}")

    return SchedulerConfig(
        type=scheduler_type,
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy=anneal_strategy,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    *,
    warmup_steps: int,
    total_steps: int | None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the per-step LR scheduler for a training loop.

    With no ``optim.scheduler`` block this reproduces the historical schedule
    exactly: linear warmup over `warmup_steps` then a constant LR. With a
    ``onecycle`` block it builds `torch.optim.lr_scheduler.OneCycleLR` sized to
    `total_steps`; `warmup_steps` is then unused because OneCycle owns its own
    ramp via ``pct_start``.

    `total_steps` must be the *exact* optimizer-step count for the full run. The
    length-bucketed batch plan yields a different batch count per epoch, so
    ``steps_per_epoch * epochs`` is not a valid substitute — callers derive the
    exact sum from the precomputed per-epoch plans.

    Args:
        optimizer: The prepared optimizer to schedule.
        cfg: The full training config.
        warmup_steps: Linear-warmup step count for the default schedule.
        total_steps: Exact total optimizer steps; required for OneCycle.

    Returns:
        The scheduler, to be stepped once per optimizer step via `_step_scheduler`.

    Raises:
        ValueError: If OneCycle is configured but `total_steps` is unknown or
            non-positive, which would otherwise yield a silently wrong schedule.
    """
    scheduler_cfg = cfg.optim.scheduler
    if scheduler_cfg is None:
        warmup = max(1, warmup_steps)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup))
        )

    if total_steps is None or total_steps < 1:
        raise ValueError(
            "optim.scheduler.type='onecycle' requires a known positive total_steps to size "
            f"its schedule, got {total_steps!r}"
        )

    logger.info(
        "onecycle scheduler: max_lr=%.3e total_steps=%d over %d epochs "
        "pct_start=%.3f div_factor=%.1f final_div_factor=%.1f anneal=%s",
        scheduler_cfg.max_lr,
        total_steps,
        cfg.optim.epochs,
        scheduler_cfg.pct_start,
        scheduler_cfg.div_factor,
        scheduler_cfg.final_div_factor,
        scheduler_cfg.anneal_strategy,
    )
    max_lr: float | list[float] = scheduler_cfg.max_lr
    if len(optimizer.param_groups) > 1:
        max_lr = [scheduler_cfg.max_lr] * len(optimizer.param_groups)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=scheduler_cfg.pct_start,
        div_factor=scheduler_cfg.div_factor,
        final_div_factor=scheduler_cfg.final_div_factor,
        anneal_strategy=cast(Literal["cos", "linear"], scheduler_cfg.anneal_strategy),
        cycle_momentum=False,
    )


def _unwrapped_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when Accelerate/DDP wrapped it."""
    current = model
    while isinstance(getattr(current, "module", None), nn.Module):
        current = cast(nn.Module, current.module)
    return current


def _validate_topo_gen_distill_contract(
    model: nn.Module, distill: DistillConfig | None
) -> TopoGenBase | None:
    """Require ``model.topo_gen`` exactly when distillation has positive ``w_gen``."""
    raw_model = _unwrapped_model(model)
    topo_gen = cast(TopoGenBase | None, getattr(raw_model, "topo_gen", None))
    has_w_gen = distill is not None and distill.w_gen > 0.0
    if topo_gen is None and has_w_gen:
        raise RuntimeError("distill.w_gen > 0 requires model.config.topo_gen")
    if topo_gen is not None and not has_w_gen:
        raise RuntimeError("model.config.topo_gen requires distill.w_gen > 0")
    return topo_gen


def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.AdamW:
    """Build AdamW, separating the kd_gen core from base/fusion parameters."""
    raw_model = _unwrapped_model(model)
    generator_getter = getattr(raw_model, "topo_gen_parameters", None)
    generator_params = (
        list(cast(Callable[[], list[nn.Parameter]], generator_getter)())
        if cfg.distill is not None and cfg.distill.arm == "kd_gen" and callable(generator_getter)
        else []
    )
    if not generator_params:
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )

    generator_ids = {id(param) for param in generator_params}
    base_params = [param for param in model.parameters() if id(param) not in generator_ids]
    if not base_params:
        raise RuntimeError("kd_gen optimizer group leaves no base parameters")
    return torch.optim.AdamW(
        [
            {"name": "base", "params": base_params, "lr": cfg.optim.lr},
            {"name": "topo_gen", "params": generator_params, "lr": cfg.optim.lr},
        ],
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )


def _set_topo_gen_training_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    distill: DistillConfig | None,
    *,
    epoch: int,
    total_epochs: int,
) -> None:
    """Set the kd_gen stop-gradient phase and its epoch-relative learning rate."""
    if distill is None or distill.arm != "kd_gen":
        return
    topo_gen = _validate_topo_gen_distill_contract(model, distill)
    if topo_gen is None:
        raise RuntimeError("kd_gen training stage requires model.config.topo_gen")

    base_group = next(
        (group for group in optimizer.param_groups if group.get("name") == "base"), None
    )
    generator_group = next(
        (group for group in optimizer.param_groups if group.get("name") == "topo_gen"), None
    )
    if base_group is None or generator_group is None:
        raise RuntimeError("kd_gen optimizer requires named base and topo_gen parameter groups")

    warmup_epochs = math.ceil(distill.joint_warmup_frac * total_epochs)
    topo_gen.joint_stage = epoch > warmup_epochs
    base_lr = float(base_group["lr"])
    generator_group["lr"] = base_lr * (distill.gen_lr_scale if topo_gen.joint_stage else 1.0)


def _count_single_process_steps(factory: LoaderFactory, cfg: Config) -> int:
    """Count exact optimizer steps across all epochs for the single-process loop.

    The length-bucketed sampler reshuffles per epoch, so each epoch has its own
    batch count and they must be summed rather than extrapolated from one epoch.
    Each epoch's loader is sized without touching feature data (the node-length
    probe is cached), so this only walks the batch plan.

    Args:
        factory: The per-epoch loader factory.
        cfg: The full training config.

    Returns:
        Total optimizer steps over epochs ``1..cfg.optim.epochs``.

    Raises:
        ValueError: If a per-epoch loader has no length, which would leave a
            OneCycle schedule mis-sized.
    """
    total = 0
    for epoch in range(1, cfg.optim.epochs + 1):
        loader = factory(epoch)
        try:
            total += len(cast(Sized, loader))
        except TypeError as exc:
            raise ValueError(
                "optim.scheduler requires a sized per-epoch loader to count total steps; "
                f"epoch {epoch} loader {type(loader).__name__} has no __len__"
            ) from exc
    return total


def _step_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    """Advance the LR scheduler by one optimizer step, tolerating overshoot.

    `OneCycleLR` raises once it is stepped past its ``total_steps``. Callers size
    it from the exact per-epoch plans, but a resumed or re-planned run could
    still take one extra step; holding the final LR is the correct degradation
    there, whereas crashing would discard a nearly finished training run.
    """
    try:
        scheduler.step()
    except ValueError:
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            raise
        logger.warning("LR scheduler exhausted its total_steps; holding the final LR")


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
        (
            "model",
            "data",
            "optim",
            "eval",
            "seed",
            "output_dir",
            "mixed_precision",
            "runtime",
            "distill",
        ),
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
        "negative_ratio",
        "token_budget",
        "batch_pairs",
        "num_workers",
        "f0_cache",
        "expected_missing_features",
    )
    _check_no_unknown_keys(data_raw, data_keys, "data")
    data = DataConfig(
        root=Path(_as_str(_require(data_raw, "root", "data."), "data.root")),
        strategy=_as_str(_require(data_raw, "strategy", "data."), "data.strategy"),
        negative_ratio=_as_int(
            _require(data_raw, "negative_ratio", "data."), "data.negative_ratio"
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
        optim_raw,
        ("lr", "weight_decay", "epochs", "warmup_steps", "grad_clip", "scheduler"),
        "optim",
    )
    optim = OptimConfig(
        lr=_as_float(_require(optim_raw, "lr", "optim."), "optim.lr"),
        weight_decay=_as_float(_require(optim_raw, "weight_decay", "optim."), "optim.weight_decay"),
        epochs=_as_int(_require(optim_raw, "epochs", "optim."), "optim.epochs"),
        warmup_steps=_as_int(_require(optim_raw, "warmup_steps", "optim."), "optim.warmup_steps"),
        grad_clip=_as_float(_require(optim_raw, "grad_clip", "optim."), "optim.grad_clip"),
        scheduler=_parse_scheduler(optim_raw.get("scheduler")),
    )
    if optim.epochs < 1:
        raise ValueError(f"optim.epochs must be >= 1, got {optim.epochs}")

    eval_raw = _as_mapping(_require(raw, "eval", ""), "eval")
    _check_no_unknown_keys(
        eval_raw, ("patience", "eval_every", "topology_every", "classification_only"), "eval"
    )
    eval_cfg = EvalConfig(
        patience=_as_int(_require(eval_raw, "patience", "eval."), "eval.patience"),
        eval_every=_as_int(_require(eval_raw, "eval_every", "eval."), "eval.eval_every"),
        topology_every=_as_int(eval_raw.get("topology_every", 1), "eval.topology_every"),
        classification_only=_as_bool(
            eval_raw.get("classification_only", False), "eval.classification_only"
        ),
    )
    if eval_cfg.topology_every < 1:
        raise ValueError(f"eval.topology_every must be >= 1, got {eval_cfg.topology_every}")

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
            probe_warmup_steps=_as_int(
                _require(runtime_raw, "probe_warmup_steps", "runtime."),
                "runtime.probe_warmup_steps",
            ),
            probe_timed_steps=_as_int(
                _require(runtime_raw, "probe_timed_steps", "runtime."),
                "runtime.probe_timed_steps",
            ),
        )

        if runtime.world_size < 0:
            raise ValueError("runtime.world_size must be 'auto' or a positive integer")
        if runtime.token_budget <= 0:
            raise ValueError("runtime.token_budget must be positive")

    distill: DistillConfig | None = None
    if "distill" in raw:
        distill = DistillConfig.from_mapping(_as_mapping(raw["distill"], "distill"))

    return Config(
        model=model,
        data=data,
        optim=optim,
        eval=eval_cfg,
        seed=_as_int(_require(raw, "seed", ""), "seed"),
        output_dir=Path(_as_str(_require(raw, "output_dir", ""), "output_dir")),
        mixed_precision=mixed_precision,
        runtime=runtime,
        distill=distill,
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
        description="Train a frozen B0 pairwise scorer (v3_1); f0_mlp/B0-alt is retired.",
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
    parser.add_argument(
        "--resume-attempt",
        type=Path,
        default=None,
        help="prior attempt directory containing an epoch-boundary training snapshot",
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
        resume_attempt=namespace.resume_attempt,
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
        :data:`~src.model.egostitch.classifier.b0_v31.BEST_V3_1_CONFIG` when empty.
        For ``f0_mlp``: the explicit kwargs verbatim, unused (see :func:`build_model`).

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
        A `V3_1` instance.

    Raises:
        RuntimeError: If ``model.config.topo_gen`` and positive
            ``cfg.distill.w_gen`` are not configured together.
        ValueError: If the model family is unknown, or is ``f0_mlp`` (the B0-alt
            baseline): ``src/model/b0_alt.py`` (``F0PairMLP``) was removed 2026-08-03
            by owner decision and has no replacement. See
            ``docs/results/E2-pair-to-topology-gap.md`` for its closed result.
    """
    kwargs = resolve_model_kwargs(cfg.model)
    if cfg.model.family == "v3_1":
        model = V3_1(**kwargs)
        _validate_topo_gen_distill_contract(model, cfg.distill)
        return model
    raise ValueError(
        f"model family '{cfg.model.family}' has no buildable model: "
        "src/model/b0_alt.py (F0PairMLP) was removed 2026-08-03 by owner decision"
    )


# --------------------------------------------------------------------------- data assembly


@dataclass(frozen=True)
class AssembledData:
    """Everything the training loop needs from the benchmark + feature packages.

    Attributes:
        benchmark: Loaded benchmark (featureless-node pairs already dropped).
        store: Frozen per-node token-sequence feature store.
        val_split: The derived V_val region and training/validation partition.
        training_positives: V_val-partitioned training positives, minus rows
            touching `exclude_nodes` (self-pairs included, sorted).
        degrees: Simple-graph degrees over the loopless training positives.
        dropped_pair_counts: Rows dropped per labeled file due to missing features.
        operative_node_ids: Sorted node ids in both the graph and the feature index.
        operative_node_count: ``len(operative_node_ids)``.
        exclude_nodes: Graph nodes without features (the verified missing set).
    """

    benchmark: Benchmark
    store: FeatureStore
    val_split: ValRegionSplit
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


def _drop_touching(pairs: Iterable[Pair], exclude_nodes: frozenset[str]) -> list[Pair]:
    """Drop pairs with either endpoint in `exclude_nodes`."""
    return [pair for pair in pairs if pair[0] not in exclude_nodes and pair[1] not in exclude_nodes]


def _training_rows(
    val_split: ValRegionSplit, exclude_nodes: frozenset[str]
) -> tuple[list[Pair], list[Pair]]:
    """Return `(positives, negatives)`: the V_val-partitioned training rows.

    Rows touching `exclude_nodes` are dropped. Positives are sorted for
    determinism (the split stores them as a frozenset); negatives keep the
    split's file order.
    """
    positives = _drop_touching(sorted(val_split.training_positives), exclude_nodes)
    negatives = _drop_touching(val_split.training_negatives, exclude_nodes)
    return positives, negatives


def _val_cls_rows(
    val_split: ValRegionSplit, exclude_nodes: frozenset[str]
) -> tuple[list[Pair], list[int]]:
    """Return `(pairs, labels)`: the V_val classification-validation rows.

    Rows touching `exclude_nodes` are dropped.
    """
    kept = [
        (pair, label)
        for pair, label in zip(val_split.val_cls_pairs, val_split.val_cls_labels, strict=True)
        if pair[0] not in exclude_nodes and pair[1] not in exclude_nodes
    ]
    return [pair for pair, _ in kept], [label for _, label in kept]


def assemble_data(
    cfg: Config, *, verify: bool = True, val_region_params: ValRegionParams | None = None
) -> AssembledData:
    """Assemble benchmark + features for training, enforcing the coverage gate.

    Feature-coverage gate: after computing ``exclude = graph_nodes -
    store.node_ids``, this raises unless ``exclude`` equals
    ``data.expected_missing_features`` exactly, and logs the operative node count
    ``|graph_nodes ∩ store.node_ids|`` (10,088 on the real package).

    The V_val region split is derived from the raw, unfiltered artifacts (its
    node universe and truth graph are exclude-node-independent); `exclude_nodes`
    row filtering is applied afterward wherever a caller consumes split rows.

    Args:
        cfg: The full training config.
        verify: Forwarded to :func:`~src.data.artifacts.load_benchmark`; when True
            the raw artifacts are verified against the pinned constants first.
        val_region_params: V_val derivation parameters; `None` uses the pinned
            production defaults. Tests pass tiny values instead of monkeypatching.

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

    # V_val derivation needs the raw label-0 rows; the `bench` above already
    # dropped exclude_nodes-touching rows, so re-read train/val_edges.txt
    # unfiltered here (train_graph.pkl and train_nodes are never exclude-node
    # filtered, so `bench.split` already carries those unfiltered).
    raw_bench = load_benchmark(benchmark_root, cfg.data.strategy, verify=False)
    raw_negatives = [
        pair
        for pair, label in zip(
            raw_bench.split.train_pairs.pairs, raw_bench.split.train_pairs.labels, strict=True
        )
        if label == 0
    ] + [
        pair
        for pair, label in zip(
            raw_bench.split.val_pairs.pairs, raw_bench.split.val_pairs.labels, strict=True
        )
        if label == 0
    ]
    val_split = derive_val_region_split(
        bench.split.train_nodes,
        bench.split.train_graph.edges(),
        raw_negatives,
        bench.positive_edges,
        params=val_region_params,
    )

    training_positives, _ = _training_rows(val_split, exclude)
    g_struct = build_g_struct(bench.split.train_nodes, training_positives)
    degrees = {str(node): int(degree) for node, degree in g_struct.degree()}

    return AssembledData(
        benchmark=bench,
        store=store,
        val_split=val_split,
        training_positives=training_positives,
        degrees=degrees,
        dropped_pair_counts=dropped_pair_counts,
        operative_node_ids=operative_node_ids,
        operative_node_count=len(operative_node_ids),
        exclude_nodes=exclude,
    )


# --------------------------------------------------------------------------- training loop


@dataclass(frozen=True)
class ValThresholdTransfer:
    """Sampled-only V_val threshold metadata for the selected checkpoint.

    Attributes:
        n_val: `len(reference.nodes)` — the V_val node-region size.
        threshold: The selected epoch's deployable sampled-subgraph threshold.
    """

    n_val: int
    threshold: float


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
            run, recorded without ever stopping it (``None`` on the direct debug
            single-GPU :func:`train_loop`, which really does stop early). The
            fixed-epoch DDP loop always trains ``optim.epochs`` epochs.
        runtime_profile: Rank-zero timing/coverage payload for the fixed-epoch DDP
            run (empty on the direct debug :func:`train_loop`). Keys are pinned by the
            Task-12 acceptance test.
        val_threshold_transfer: The selected epoch's V_val threshold-transfer
            metadata, or ``None`` when training ran without topology
            evaluation (unit-test stubs, or the best-AUPRC fallback).
    """

    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_metrics: EdgeMetrics
    last_state_dict: dict[str, torch.Tensor]
    last_epoch: int
    last_val_metrics: EdgeMetrics
    history: list[dict[str, object]]
    stopped_early: bool
    counterfactual_stop_epoch: int | None = None
    runtime_profile: dict[str, object] = field(default_factory=dict)
    val_threshold_transfer: ValThresholdTransfer | None = None


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
    schedule_total_steps: int | None = None,
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
        schedule_total_steps: Exact optimizer-step count over all epochs, used to
            size a ``optim.scheduler`` OneCycle schedule; unused otherwise.
        on_eval: Optional callback ``(history_entry, improved, metrics)`` invoked
            after every evaluation (used by the CLI for incremental artifacts).

    Returns:
        The `TrainResult`.

    Raises:
        RuntimeError: If training ends without a single evaluation.
    """
    optimizer = _build_optimizer(model, cfg)
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = _build_scheduler(
        optimizer,
        cfg,
        warmup_steps=cfg.optim.warmup_steps,
        total_steps=schedule_total_steps,
    )

    history: list[dict[str, object]] = []
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
        _set_topo_gen_training_stage(
            model,
            optimizer,
            cfg.distill,
            epoch=epoch,
            total_epochs=cfg.optim.epochs,
        )
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
            _step_scheduler(scheduler)
            _set_topo_gen_training_stage(
                model,
                optimizer,
                cfg.distill,
                epoch=epoch,
                total_epochs=cfg.optim.epochs,
            )
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
            entry: dict[str, object] = {
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

    Finalizes ``best.pt`` / ``last.pt`` (payload keys exactly ``model_state``,
    ``model_family``, ``model_config``, ``epoch``, ``val_metrics``, ``seed``,
    ``config``) and ``run_metadata.json`` (config hash, checkpoint id = first
    16 hex of the sha256 over the best
    checkpoint's model_state tensor bytes, torch version, timestamp, dropped-pair
    counts, positives mode, and — when ``result.val_threshold_transfer`` is set —
    a ``val_threshold_transfer`` block). The training loop owns incremental
    ``metrics.jsonl``; finalization never rewrites it.

    Args:
        result: The finished training result.
        cfg: The full training config.
        model_kwargs: Resolved model constructor kwargs (stored as ``model_config``).
        dropped_pair_counts: Per-file dropped-row counts from data assembly.
    """
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config_to_dict(cfg)

    _torch_save_atomic(
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
    _torch_save_atomic(
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

    run_metadata = {
        "config_hash": hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "checkpoint_id": _state_digest(result.best_state_dict)[:16],
        "torch_version": str(torch.__version__),
        "timestamp": datetime.now(UTC).isoformat(),
        "dropped_pair_counts": dropped_pair_counts,
        "training_interactions": "all_train_positives",
        "arm": (
            cfg.distill.arm if cfg.distill is not None and cfg.distill.active else cfg.model.family
        ),
        "selected_epoch": result.best_epoch,
    }
    if result.val_threshold_transfer is not None:
        # Lets a future test-time operating point be derived by degree-density
        # transfer without re-scoring V_val.
        run_metadata["val_threshold_transfer"] = asdict(result.val_threshold_transfer)
    _write_json_atomic(output_dir / "run_metadata.json", run_metadata)
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
    """Build the degree-corrected negative sampler over featureful train nodes.

    Rejects V_val-internal pairs like a global positive.
    """
    train_universe = sorted(set(assembled.benchmark.split.train_nodes) - assembled.exclude_nodes)
    return NegativeSampler(
        train_universe,
        assembled.degrees,
        assembled.benchmark.positive_edges,
        forbidden_internal_nodes=assembled.val_split.v_val,
    )


def _build_v3_1_loaders(
    cfg: Config, assembled: AssembledData
) -> tuple[LoaderFactory, Iterable[Batch]]:
    """Build loaders over the V_val training/validation partition."""
    store = assembled.store
    val_split = assembled.val_split
    positives, negatives = _training_rows(val_split, assembled.exclude_nodes)
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

    val_pairs, val_labels = _val_cls_rows(val_split, assembled.exclude_nodes)
    val_lengths = lengths_for(val_pairs)
    val_dataset = TokenPairDataset(val_pairs, val_labels, store, lengths=val_lengths)
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

    val_pairs, val_labels = _val_cls_rows(assembled.val_split, assembled.exclude_nodes)
    val_loader = batches_for(val_pairs, val_labels)

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

    def __len__(self) -> int:
        """Return the compact source length used by deterministic KD schedules."""
        if not isinstance(self._source, Sized):
            raise TypeError("compact batch source has no known length")
        return len(self._source)


def compute_sample_warmup_steps(
    baseline_batch_sizes: Sequence[int],
    new_global_batch_sizes: Sequence[int],
    *,
    baseline_steps: int,
) -> int:
    """Return the new-schedule step count needed to preserve warmup pair exposure.

    ``baseline_batch_sizes`` is the direct-debug per-step batch-size sequence;
    cycling it for ``baseline_steps`` steps defines the target number of pair
    samples the original warmup schedule saw. This returns how many steps of the
    new (larger, distributed) global-batch-size sequence — also cycled — are
    needed to see at least as many total pair samples.

    Args:
        baseline_batch_sizes: Legacy per-step batch sizes (one epoch's worth).
        new_global_batch_sizes: New per-step *global* (summed across ranks) batch
            sizes (one epoch's worth).
        baseline_steps: Number of direct-debug warmup steps to reproduce the exposure of.

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
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)
    if world_size == 1:
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            collate_fn=identity_compact_batch,  # type: ignore[arg-type]
            generator=loader_generator,
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
        generator=loader_generator,
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
) -> tuple[PackedLoaderFactory, Iterable[Batch], int, int]:
    """Build packed, multi-worker ``v3_1`` train/val loaders for one DDP rank.

    Endpoint integer ids and true token lengths come from ``table.manifest`` only —
    no feature tensors are read to plan batches. The V_val-partitioned training
    rows are replanned into per-rank compact batches every epoch via
    :func:`~src.data.distributed_pairs.build_distributed_epoch_plan`; the fixed
    V_val classification-validation rows are planned once, unshuffled. DataLoader workers hand
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
        ``(train_loader_factory, val_loader, warmup_steps, schedule_total_steps)``,
        where the last element is the exact optimizer-step count across epochs
        ``1..cfg.optim.epochs`` for this rank (used to size a OneCycle schedule).

    Raises:
        ValueError: If ``cfg.runtime`` is unset.
    """
    runtime = cfg.runtime
    if runtime is None:
        raise ValueError("packed v3_1 loaders require a configured cfg.runtime")

    node_index = table.manifest.node_index()
    lengths_by_node = {record.node_id: record.length for record in table.manifest.nodes}

    positives, negatives = _training_rows(assembled.val_split, assembled.exclude_nodes)
    train_pairs = positives + negatives
    train_labels = [1] * len(positives) + [0] * len(negatives)
    train_lengths = _pair_lengths_from_manifest(train_pairs, lengths_by_node)
    train_row_ids, train_node_a, train_node_b, train_label_tensor = _compact_pair_columns(
        train_pairs, train_labels, node_index
    )

    val_pairs, val_labels = _val_cls_rows(assembled.val_split, assembled.exclude_nodes)
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

    # Exact optimizer-step count for the whole run. The training loop iterates
    # epochs 1..epochs (epoch 0 is only the warmup-scaling reference plan), and
    # each epoch's length-bucketed plan has its own batch count, so this sum --
    # not steps_per_epoch * epochs -- is what sizes a OneCycle schedule.
    schedule_total_steps = sum(
        len(plans_by_epoch[epoch][process_index]) for epoch in range(1, cfg.optim.epochs + 1)
    )

    return factory, val_loader, warmup_steps, schedule_total_steps


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
    kd_val: KDValDiagnostics | None = None,
    label_smoothing: float = 0.0,
) -> ValidationOutcome:
    """Score the fixed cls validation set across all ranks and agree on the metrics.

    Each rank scores its slice; the slices are padded to a common length and
    gathered onto every rank. Every rank then masks the padding and verifies
    exact once-per-row coverage via :func:`validate_gathered_validation`, so a
    coverage violation raises identically on all ranks (never a rank-zero-only
    death that leaves the others deadlocked in the broadcast). Rank zero
    computes the metrics and the dict is broadcast so every rank makes the same
    best-epoch decision. This is the cls-only pass; `_evaluate_two_pass` merges
    it with the separate V_val topology pass.

    Args:
        model: The scorer (restored to train mode before returning).
        val_loader: This rank's assembled validation batches (carry ``_row_id``).
        accelerator: The DDP accelerator.
        expected_row_ids: Complete fixed validation row-ID set. When ``None`` it is
            derived as ``arange`` over the gathered row count — a self-contained
            fallback that still catches duplicates and interior gaps; production
            binds the true set so truncation is caught too.
        kd_val: Validation-row teacher targets for the KD diagnostics; ``None``
            leaves `ValidationOutcome.kd` unset. For ``kd_gen``, injects the
            normalized teacher latent before the diagnostic forward.
        label_smoothing: Symmetric binary smoothing ε applied to the labels for
            `ValidationOutcome.task_loss`, matching the training objective.

    Returns:
        The `ValidationOutcome` with `topology=None`, identical on every rank.

    Raises:
        ValueError: On any duplicate or missing validation row (all ranks).
    """
    model.eval()
    row_id_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    diag_parts: list[torch.Tensor] = []
    relational_block = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    inject_latent = kd_val is not None and kd_val.teacher_latent is not None
    collect_diag = kd_val is not None and (kd_val.arm == "kd_rep" or inject_latent)
    rng_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=inject_latent), torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, accelerator.device)
            if inject_latent:
                assert kd_val is not None and kd_val.teacher_latent is not None
                rows = batch["_row_id"]
                batch["kd_teacher_latent"] = (
                    kd_val.teacher_latent[rows].float() / kd_val.latent_scale
                )
            output = model(batch)
            logits = output["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            row_id_parts.append(batch["_row_id"].detach().to(torch.int64))
            label_parts.append(batch["label"].detach().to(torch.float32))
            logit_parts.append(logits.detach().to(torch.float32))
            if kd_val is not None and kd_val.arm == "kd_gram":
                rows = batch["_row_id"]
                student_rep = output.get("pair_repr")
                if student_rep is None or kd_val.teacher_rep is None:
                    raise RuntimeError(
                        "kd_gram validation block diagnostics require pair_repr and teacher_rep"
                    )
                if rows.numel() >= 2:
                    relational_block[0] += kd_gram_loss(
                        student_rep.float(), kd_val.teacher_rep[rows].float()
                    ).double()
                    relational_block[2] += 1.0
            if collect_diag:
                assert kd_val is not None
                rows = batch["_row_id"]
                if kd_val.arm == "kd_rep":
                    student_rep = output.get("kd_rep")
                    if student_rep is None:
                        student_rep = output.get("pair_repr")
                    if student_rep is None:
                        raise RuntimeError(
                            "kd_rep validation diagnostics require kd_rep or pair_repr in "
                            "the model forward output"
                        )
                    assert kd_val.teacher_rep is not None
                    teacher_rep = kd_val.teacher_rep[rows].float()
                    diag = nn.functional.cosine_similarity(
                        student_rep.float(), teacher_rep, dim=-1, eps=1e-8
                    )
                else:  # kd_gen
                    latent_sample = output.get("gen_latent_sample")
                    if latent_sample is None:
                        raise RuntimeError(
                            "kd_gen validation diagnostics require gen_latent_sample"
                        )
                    teacher_latent = batch["kd_teacher_latent"]
                    diag = nn.functional.cosine_similarity(
                        latent_sample.float(), teacher_latent.float(), dim=-1, eps=1e-8
                    )
                diag_parts.append(diag.detach().to(torch.float32))
        context_metrics = (
            kd_val.context_stream.validation_diagnostics(model)
            if kd_val is not None and kd_val.context_stream is not None
            else None
        )
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
    local_diag = (
        torch.cat(diag_parts) if diag_parts else torch.empty(0, dtype=torch.float32, device=device)
    )

    padded_row_ids = accelerator.pad_across_processes(local_row_ids, dim=0, pad_index=-1)
    padded_labels = accelerator.pad_across_processes(local_labels, dim=0, pad_index=-1)
    padded_logits = accelerator.pad_across_processes(local_logits, dim=0, pad_index=0)

    gathered_row_ids = accelerator.gather(padded_row_ids)
    gathered_labels = accelerator.gather(padded_labels)
    gathered_logits = accelerator.gather(padded_logits)
    gathered_diag: torch.Tensor | None = None
    if collect_diag:
        padded_diag = accelerator.pad_across_processes(local_diag, dim=0, pad_index=0)
        gathered_diag = accelerator.gather(padded_diag)
    reduced_relational_block = relational_block
    if kd_val is not None and kd_val.arm == "kd_gram":
        reduced_relational_block = accelerator.reduce(relational_block, reduction="sum")

    # Coverage must be validated symmetrically: accelerator.gather returns the
    # full gathered tensors on EVERY rank, so every rank masks the padding and
    # runs the same coverage check. A duplicate/missing row then raises
    # identically everywhere; if only rank zero validated, it would die before
    # the broadcast below and every other rank would block in it until the
    # NCCL watchdog killed the job, masking the real error.
    row_ids_np = gathered_row_ids.cpu().numpy()
    labels_np = gathered_labels.cpu().numpy()
    logits_np = gathered_logits.cpu().numpy()
    diag_np = gathered_diag.cpu().numpy() if gathered_diag is not None else None
    keep = row_ids_np >= 0
    row_ids_np = row_ids_np[keep]
    labels_np = labels_np[keep]
    logits_np = logits_np[keep]
    if diag_np is not None:
        diag_np = diag_np[keep]
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

    outcome_payload: list[dict[str, object] | None] = [None]
    if accelerator.is_main_process:
        probs = _stable_sigmoid(logits_sorted.astype(np.float64))
        metrics = compute_edge_metrics(labels_sorted.astype(np.float64), probs)
        outcome_dict: dict[str, object] = {"metrics": asdict(metrics)}
        if labels_sorted.shape[0] > 0:
            smoothed = (
                labels_sorted.astype(np.float64) * (1.0 - label_smoothing) + 0.5 * label_smoothing
            )
            outcome_dict["task_loss"] = _stable_bce_with_logits(
                logits_sorted.astype(np.float64), smoothed
            )
        if kd_val is not None:
            kd_metrics: dict[str, float] = {}
            if diag_np is not None and diag_np.size > 0:
                key = "val_kd_rep_cos" if kd_val.arm == "kd_rep" else "val_kd_latent_cos"
                kd_metrics[key] = float(diag_np.mean())
                if kd_val.arm == "kd_rep":
                    kd_metrics["val_kd_rep_loss"] = 1.0 - kd_metrics[key]
            logits64 = logits_sorted.astype(np.float64)
            teacher64 = kd_val.teacher_logit_np
            if logits64.shape[0] > 0:
                kd_metrics["val_kd_logit_corr"] = _pearson_from_moments(
                    float(logits64.sum()),
                    float(teacher64.sum()),
                    float((logits64 * logits64).sum()),
                    float((teacher64 * teacher64).sum()),
                    float((logits64 * teacher64).sum()),
                    float(logits64.shape[0]),
                )
                prob_err = np.abs(_stable_sigmoid(logits64) - _stable_sigmoid(teacher64))
                kd_metrics["val_kd_prob_mae"] = float(prob_err.mean())
                kd_metrics["val_kd_logit_loss"] = _stable_bce_with_logits(
                    logits64, _stable_sigmoid(teacher64)
                )
            block_count = float(reduced_relational_block[2].item())
            if context_metrics is not None:
                kd_metrics.update(context_metrics)
            if block_count > 0.0 and kd_val.arm == "kd_gram":
                kd_metrics["val_kd_gram_block_loss"] = (
                    float(reduced_relational_block[0].item()) / block_count
                )
            outcome_dict["kd"] = kd_metrics
        outcome_payload[0] = outcome_dict

    broadcast_object_list(outcome_payload, from_process=0)
    payload = outcome_payload[0]
    if payload is None:  # pragma: no cover - broadcast always populates rank>0
        raise RuntimeError("distributed validation failed to broadcast metrics")
    kd_result = cast(dict[str, float] | None, payload.get("kd")) if kd_val is not None else None
    return ValidationOutcome(
        metrics=EdgeMetrics(**cast(dict[str, Any], payload["metrics"])),
        topology=None,
        kd=kd_result,
        task_loss=cast(float | None, payload.get("task_loss")),
    )


def _stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid on a float64 array."""
    out = np.empty_like(logit, dtype=np.float64)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_l = np.exp(logit[~positive])
    out[~positive] = exp_l / (1.0 + exp_l)
    return out


def _stable_bce_with_logits(logit: np.ndarray, target_prob: np.ndarray) -> float:
    """Mean float64 BCE of logits against probability targets via softplus(x) - y*x."""
    return float(np.mean(np.logaddexp(0.0, logit) - target_prob * logit))


TopologyEvalFn = Callable[[nn.Module, Accelerator], ValTopologyResult]


def _evaluate_val_universe(
    model: nn.Module,
    accelerator: Accelerator,
    *,
    table: PackedFeatureTable,
    node_a_all: torch.Tensor,
    node_b_all: torch.Tensor,
    boundary: int,
    batch_pairs: int,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    reference: ValTopologyReference,
) -> ValTopologyResult:
    """Score the exact ball-union rows and select sampled-only topology threshold.

    Mirrors `_evaluate_distributed`'s DDP fail-closed gather discipline: every
    rank scores its `range(rank, n, world)` shard of the ball-union
    `node_a_all`/`node_b_all` rows
    through the same packed-table forward path as the cls pairs, the padded
    logits are gathered onto every rank, and `validate_gathered_validation`
    enforces exact once-per-row coverage on every rank. Rank zero computes
    `val_region_topology_metrics` and broadcasts the result so every rank agrees.

    Args:
        model: The scorer (restored to train mode before returning).
        accelerator: The DDP accelerator.
        table: Packed feature table resident on the training device.
        node_a_all: Packed row index of the exact sampled-pair union.
        node_b_all: Packed row index aligned with `node_a_all`.
        boundary: Token-sequence padding boundary shared by every V_val node.
        batch_pairs: Rows scored per no-grad forward call.
        u_idx: `(n_u,)` `reference.nodes` index for the U rows only, row-ID
            order (from `val_ball_union_universe`).
        v_idx: `(n_u,)` `reference.nodes` index for the U rows only, aligned
            with `u_idx`.
        reference: The once-per-run `ValTopologyReference`.

    Returns:
        The `ValTopologyResult`, identical on every rank.

    Raises:
        ValueError: On any duplicate or missing universe row (all ranks).
    """
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_rows = int(node_a_all.shape[0])
    device = accelerator.device

    row_ids = torch.arange(rank, n_rows, world_size, dtype=torch.int64)
    model.eval()
    logit_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, row_ids.shape[0], batch_pairs):
            rows = row_ids[start : start + batch_pairs]
            emb_a, len_a = table.gather_nodes(node_a_all.index_select(0, rows), boundary)
            emb_b, len_b = table.gather_nodes(node_b_all.index_select(0, rows), boundary)
            output = model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b})
            logits = output["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            logit_parts.append(logits.detach().to(torch.float32))
    model.train()

    local_row_ids = row_ids.to(device)
    local_logits = (
        torch.cat(logit_parts)
        if logit_parts
        else torch.empty(0, dtype=torch.float32, device=device)
    )

    padded_row_ids = accelerator.pad_across_processes(local_row_ids, dim=0, pad_index=-1)
    padded_logits = accelerator.pad_across_processes(local_logits, dim=0, pad_index=0)
    gathered_row_ids = accelerator.gather(padded_row_ids)
    gathered_logits = accelerator.gather(padded_logits)

    # Symmetric on every rank, exactly like `_evaluate_distributed`: a
    # duplicate/missing row raises identically everywhere rather than a
    # rank-zero-only death that deadlocks the others in the broadcast below.
    row_ids_np = gathered_row_ids.cpu().numpy()
    logits_np = gathered_logits.cpu().numpy()
    keep = row_ids_np >= 0
    row_ids_np = row_ids_np[keep]
    logits_np = logits_np[keep]
    # `validate_gathered_validation` also reorders a `labels` array; the
    # universe pass has no per-row label, so a zero placeholder rides along
    # and is discarded once the coverage check passes.
    _, logits_sorted = validate_gathered_validation(
        row_ids=row_ids_np,
        labels=np.zeros_like(row_ids_np, dtype=np.float64),
        logits=logits_np,
        expected_row_ids=np.arange(n_rows, dtype=np.int64),
    )

    payload: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        result = val_region_topology_metrics(
            u_idx=u_idx,
            v_idx=v_idx,
            logits=logits_sorted.astype(np.float64),
            reference=reference,
        )
        payload[0] = asdict(result)

    broadcast_object_list(payload, from_process=0)
    result_payload = payload[0]
    if result_payload is None:  # pragma: no cover - broadcast always populates rank>0
        raise RuntimeError("distributed V_val topology evaluation failed to broadcast metrics")
    metrics_payload = cast(dict[str, Any], result_payload["metrics"])
    return ValTopologyResult(
        metrics=TopologyValidationMetrics(**metrics_payload),
        threshold=cast(float, result_payload["threshold"]),
    )


def _evaluate_two_pass(
    model: nn.Module,
    val_loader: Iterable[Batch],
    accelerator: Accelerator,
    *,
    expected_row_ids: np.ndarray,
    topology_eval_fn: TopologyEvalFn,
    kd_val: KDValDiagnostics | None = None,
    label_smoothing: float = 0.0,
) -> ValidationOutcome:
    """Run the cls and V_val-topology validation passes and merge their outcomes."""
    cls_outcome = _evaluate_distributed(
        model,
        val_loader,
        accelerator,
        expected_row_ids=expected_row_ids,
        kd_val=kd_val,
        label_smoothing=label_smoothing,
    )
    topology = topology_eval_fn(model, accelerator)
    return ValidationOutcome(
        metrics=cls_outcome.metrics,
        topology=topology,
        kd=cls_outcome.kd,
        task_loss=cls_outcome.task_loss,
    )


def _topology_from_metrics_row(row: dict[str, object]) -> ValTopologyResult | None:
    """Reconstruct the selector's V_val topology result from one persisted epoch row.

    Returns `None` for a row missing any of the five topology metrics or the
    sampled-only `val_threshold` field. A row from an older validation protocol
    then falls back to
    `require_topology`'s fail-closed behavior ("resume across the V_val
    protocol change unsupported") rather than silently mixing metric
    semantics.
    """
    keys = (
        "val_gs_bfs",
        "val_rd_bfs",
        "val_degree_mmd_ratio",
        "val_clustering_mmd_ratio",
        "val_spectral_mmd_ratio",
        "val_threshold",
    )
    if not all(key in row for key in keys) or "val_admitted_non_self_fraction" in row:
        return None
    return ValTopologyResult(
        metrics=TopologyValidationMetrics(
            gs=float(cast(float, row["val_gs_bfs"])),
            rd=float(cast(float, row["val_rd_bfs"])),
            degree_mmd=float(cast(float, row["val_degree_mmd_ratio"])),
            clustering_mmd=float(cast(float, row["val_clustering_mmd_ratio"])),
            spectral_mmd=float(cast(float, row["val_spectral_mmd_ratio"])),
        ),
        threshold=float(cast(float, row["val_threshold"])),
    )


@dataclass(frozen=True)
class KDValDiagnostics:
    """Validation-row teacher targets for the validation-only KD diagnostics.

    Attributes:
        arm: The active KD arm.
        teacher_logit: ``(n_val,)`` fp32 teacher logits, on the training device.
        teacher_logit_np: ``(n_val,)`` fp64 CPU copy, aligned with V_val
            classification row ids ``0..n_val-1`` -- the row order
            `validate_gathered_validation` sorts scored logits into.
        teacher_rep: ``(n_val, rep_dim)`` fp16 teacher pooled embeddings, on
            the training device; populated for ``kd_rep`` and ``kd_gram``.
        teacher_latent: ``(n_val, latent_dim)`` fp16 teacher latents, on the
            training device; populated only for ``kd_gen`` and normalized at use.
        latent_scale: Artifact-wide teacher latent RMS used for normalization.
        context_stream: Fixed V_val context bank used only for ``kd_rank``
            rank/distribution/tie diagnostics.
    """

    arm: str
    teacher_logit: torch.Tensor
    teacher_logit_np: np.ndarray
    teacher_rep: torch.Tensor | None
    teacher_latent: torch.Tensor | None
    latent_scale: float = 1.0
    context_stream: KDContextStream | None = None


def _pearson_from_moments(
    sum_s: float, sum_t: float, sum_s2: float, sum_t2: float, sum_st: float, n: float
) -> float:
    """Pearson correlation from batch moment sums; 0.0 under a degenerate variance."""
    mean_s = sum_s / n
    mean_t = sum_t / n
    var_s = max(sum_s2 / n - mean_s * mean_s, 0.0)
    var_t = max(sum_t2 / n - mean_t * mean_t, 0.0)
    denom = (var_s * var_t) ** 0.5
    if denom <= 1e-12:
        return 0.0
    cov = sum_st / n - mean_s * mean_t
    return float(cov / denom)


class GlobalRelationalRows(NamedTuple):
    """One global optimizer step's row-aligned inputs for Gram KD."""

    student: torch.Tensor
    teacher: torch.Tensor
    endpoint_a: torch.Tensor
    endpoint_b: torch.Tensor
    row_ids: torch.Tensor


def _pad_first_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pad ``tensor`` with zero rows to ``size`` without detaching live rows."""
    if tensor.shape[0] == size:
        return tensor
    padding = tensor.new_zeros((size - tensor.shape[0], *tensor.shape[1:]))
    return torch.cat([tensor, padding], dim=0)


def _gather_global_relational_rows(
    student: torch.Tensor,
    teacher: torch.Tensor,
    endpoint_a: torch.Tensor,
    endpoint_b: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    world_size: int,
) -> GlobalRelationalRows:
    """Differentiably gather a variable-length official-row batch across ranks.

    Student values use ``torch.distributed.nn.functional.all_gather`` so the
    global D2/D3 loss propagates to every rank's one local forward. Teacher
    values, endpoints, and row IDs are detached metadata. Every rank computes
    the same global scalar; differentiable all-gather's SUM reduce-scatter in
    backward followed by DDP's parameter-gradient mean gives the gradient of
    that scalar exactly, so callers must not apply row-count scaling again.
    """
    if world_size == 1:
        return GlobalRelationalRows(student, teacher, endpoint_a, endpoint_b, row_ids)
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError("global relational KD requires the initialized DDP process group")

    local_size = torch.tensor([student.shape[0]], dtype=torch.int64, device=student.device)
    size_parts = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_parts, local_size)
    sizes = [int(part.item()) for part in size_parts]
    max_size = max(sizes)

    student_parts = differentiable_all_gather(  # type: ignore[no-untyped-call]
        _pad_first_dim(student, max_size)
    )
    global_student = torch.cat(
        [part[:size] for part, size in zip(student_parts, sizes, strict=True)], dim=0
    )

    def gather_metadata(tensor: torch.Tensor) -> torch.Tensor:
        padded = _pad_first_dim(tensor.detach(), max_size)
        parts = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(parts, padded)
        return torch.cat([part[:size] for part, size in zip(parts, sizes, strict=True)], dim=0)

    global_teacher = gather_metadata(teacher)
    global_a = gather_metadata(endpoint_a)
    global_b = gather_metadata(endpoint_b)
    global_row_ids = gather_metadata(row_ids)
    if torch.unique(global_row_ids).numel() != global_row_ids.numel():
        raise ValueError("duplicate official training row IDs in one global relational KD step")
    return GlobalRelationalRows(global_student, global_teacher, global_a, global_b, global_row_ids)


class KDContextStream:
    """Deterministic epoch-banked anchor/context stream for reference-faithful KD2."""

    def __init__(
        self,
        distill: DistillConfig,
        targets: KDContextTargets,
        table: PackedFeatureTable,
        *,
        allowed_nodes: frozenset[str],
        forbidden_internal_nodes: frozenset[str],
        epochs: int,
        rank: int,
        world_size: int,
    ) -> None:
        if distill.arm != "kd_rank":
            raise ValueError("KDContextStream requires the active kd_rank arm")
        if epochs > len(targets.banks):
            raise ValueError(
                f"training epochs ({epochs}) exceed context banks ({len(targets.banks)})"
            )
        if rank < 0 or rank >= world_size or world_size < 1:
            raise ValueError(f"invalid KD context rank/world size: {rank}/{world_size}")
        outside = sorted(set(targets.node_ids) - allowed_nodes)
        if outside:
            raise ValueError(
                f"KD context target nodes outside the training universe: {outside[:5]}"
            )
        node_index = table.manifest.node_index()
        missing = [node for node in targets.node_ids if node not in node_index]
        if missing:
            raise ValueError(
                f"KD context target nodes missing from the feature pack: {missing[:5]}"
            )
        nodes = np.asarray(targets.node_ids, dtype=object)
        forbidden = list(forbidden_internal_nodes)
        for bank_index, bank in enumerate(targets.banks):
            anchors = np.repeat(bank.anchor_idx, np.diff(bank.anchor_offsets))
            if bool(
                (
                    np.isin(nodes[anchors], forbidden) & np.isin(nodes[bank.partner_idx], forbidden)
                ).any()
            ):
                raise ValueError(f"KD context bank {bank_index} contains a V_val-internal pair")
        expected_val = np.asarray(
            [i for i, node in enumerate(targets.node_ids) if node in forbidden_internal_nodes],
            dtype=np.int32,
        )
        if not np.array_equal(targets.val_bank.anchor_idx, expected_val):
            raise ValueError("KD context validation anchors do not match the V_val identity")

        self._distill = distill
        self._targets = targets
        self._table = table
        self._rank = rank
        self._world_size = world_size
        self._packed_rows = np.asarray(
            [node_index[node] for node in targets.node_ids], dtype=np.int64
        )
        self.last_anchor_idx: tuple[int, ...] = ()
        self.last_bank_index = -1

    @staticmethod
    def _step_slice(size: int, steps: int, step: int) -> tuple[int, int]:
        """Partition ``size`` rows exactly once, placing the remainder at the end."""
        if steps < 1 or step < 0 or step >= steps:
            raise ValueError(f"invalid KD context step {step} for {steps} steps")
        base, remainder = divmod(size, steps)
        extra_start = steps - remainder
        start = step * base + max(0, step - extra_start)
        count = base + int(step >= extra_start)
        return start, start + count

    def _anchor_positions(
        self, bank: KDContextBank, *, rank: int, steps: int, step: int
    ) -> list[int]:
        shard = list(range(rank, len(bank.anchor_idx), self._world_size))
        start, stop = self._step_slice(len(shard), steps, step)
        return shard[start:stop]

    @staticmethod
    def _live_counts(bank: KDContextBank, positions: Sequence[int]) -> tuple[int, int]:
        lengths = np.diff(bank.anchor_offsets)
        live_lengths = lengths[np.asarray(positions, dtype=np.int64)] if positions else np.empty(0)
        pairs = int(sum(int(length) * (int(length) - 1) // 2 for length in live_lengths))
        anchors = int((live_lengths > 0).sum())
        return pairs, anchors

    def _global_step_counts(self, bank: KDContextBank, *, steps: int, step: int) -> tuple[int, int]:
        pairs = 0
        anchors = 0
        for rank in range(self._world_size):
            rank_pairs, rank_anchors = self._live_counts(
                bank, self._anchor_positions(bank, rank=rank, steps=steps, step=step)
            )
            pairs += rank_pairs
            anchors += rank_anchors
        return pairs, anchors

    def _score(
        self, model: nn.Module, bank: KDContextBank, positions: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_rows: list[int] = []
        partner_rows: list[int] = []
        teacher_rows: list[int] = []
        groups: list[int] = []
        for group, position in enumerate(positions):
            start = int(bank.anchor_offsets[position])
            stop = int(bank.anchor_offsets[position + 1])
            anchor_idx = int(bank.anchor_idx[position])
            for context_row in range(start, stop):
                anchor_rows.append(int(self._packed_rows[anchor_idx]))
                partner_rows.append(int(self._packed_rows[int(bank.partner_idx[context_row])]))
                teacher_rows.append(int(bank.score_idx[context_row]))
                groups.append(group)
        device = self._table.tokens.device
        if not anchor_rows:
            empty_float = torch.empty(0, dtype=torch.float32, device=device)
            empty_long = torch.empty(0, dtype=torch.int64, device=device)
            return empty_float, empty_float, empty_long

        lengths = self._table.manifest.nodes
        buckets: dict[int, list[int]] = {boundary: [] for boundary in BUCKET_BOUNDARIES}
        for row, (anchor_row, partner_row) in enumerate(
            zip(anchor_rows, partner_rows, strict=True)
        ):
            max_length = max(lengths[anchor_row].length, lengths[partner_row].length)
            boundary = next((value for value in BUCKET_BOUNDARIES if max_length <= value), None)
            if boundary is None:
                raise ValueError(
                    f"KD context packed length {max_length} exceeds {BUCKET_BOUNDARIES[-1]}"
                )
            buckets[boundary].append(row)

        student_parts: list[torch.Tensor] = []
        teacher_parts: list[torch.Tensor] = []
        group_parts: list[torch.Tensor] = []
        for boundary, rows in buckets.items():
            if not rows:
                continue
            anchor = torch.as_tensor(
                [anchor_rows[row] for row in rows], dtype=torch.int64, device=device
            )
            partner = torch.as_tensor(
                [partner_rows[row] for row in rows], dtype=torch.int64, device=device
            )
            emb_a, len_a = self._table.gather_nodes(anchor, boundary)
            emb_b, len_b = self._table.gather_nodes(partner, boundary)
            output = cast(
                dict[str, torch.Tensor],
                model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b}),
            )
            logits = output["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            student_parts.append(logits.float())
            score_rows = np.asarray([teacher_rows[row] for row in rows], dtype=np.int64)
            teacher_parts.append(
                torch.as_tensor(
                    self._targets.teacher_logit[score_rows], dtype=torch.float32, device=device
                )
            )
            group_parts.append(
                torch.as_tensor([groups[row] for row in rows], dtype=torch.int64, device=device)
            )
        return torch.cat(student_parts), torch.cat(teacher_parts), torch.cat(group_parts)

    @staticmethod
    def _tie_counts(teacher: torch.Tensor, groups: torch.Tensor, margin: float) -> tuple[int, int]:
        ties = 0
        pairs = 0
        teacher_prob = torch.sigmoid(teacher.detach())
        for group in torch.unique(groups):
            values = teacher_prob[groups == group]
            if values.numel() < 2:
                continue
            left, right = torch.triu_indices(values.numel(), values.numel(), offset=1)
            diff = (values[left] - values[right]).abs()
            ties += int((diff <= margin).sum().item())
            pairs += int(diff.numel())
        return ties, pairs

    def loss(
        self, model: nn.Module, *, epoch: int, step: int, steps: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return separately count-scaled rank/dist losses for one optimizer step."""
        bank_index = epoch - 1
        if bank_index < 0 or bank_index >= len(self._targets.banks):
            raise ValueError(f"KD context epoch {epoch} has no bank")
        bank = self._targets.banks[bank_index]
        positions = self._anchor_positions(bank, rank=self._rank, steps=steps, step=step)
        self.last_anchor_idx = tuple(int(bank.anchor_idx[position]) for position in positions)
        self.last_bank_index = bank_index
        local_pairs, local_anchors = self._live_counts(bank, positions)
        global_pairs, global_anchors = self._global_step_counts(bank, steps=steps, step=step)
        student, teacher, groups = self._score(model, bank, positions)
        zero = next(model.parameters()).sum() * 0.0
        rank_term = (
            kd_rank_loss(student, teacher, groups, margin=self._distill.margin)
            if local_pairs
            else zero
        )
        dist_term = kd_dist_loss(student, teacher, groups) if local_anchors else zero
        rank_scale = self._world_size * local_pairs / global_pairs if global_pairs else 0.0
        dist_scale = self._world_size * local_anchors / global_anchors if global_anchors else 0.0
        total = (
            self._distill.w_rank * rank_term * rank_scale
            + self._distill.w_dist * dist_term * dist_scale
        )
        ties, counted_pairs = self._tie_counts(teacher, groups, self._distill.margin)
        return total, {
            "context_pairs": float(local_pairs),
            "context_anchors": float(local_anchors),
            "context_ties": float(ties),
            "context_tie_pairs": float(counted_pairs),
            "sum_rank": float(rank_term.detach().item()) * local_pairs,
            "sum_dist": float(dist_term.detach().item()) * local_anchors,
        }

    def epoch_telemetry(self, accelerator: Accelerator, sums: dict[str, float]) -> dict[str, float]:
        """Reduce rank-local context counts and loss numerators for one epoch."""
        keys = sorted(sums)
        reduced = accelerator.reduce(
            torch.tensor(
                [sums[key] for key in keys], device=accelerator.device, dtype=torch.float64
            ),
            reduction="sum",
        )
        values = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        pairs = values.get("context_pairs", 0.0)
        anchors = values.get("context_anchors", 0.0)
        tie_pairs = values.get("context_tie_pairs", 0.0)
        return {
            "kd_rank_loss": values.get("sum_rank", 0.0) / max(pairs, 1.0),
            "kd_dist_loss": values.get("sum_dist", 0.0) / max(anchors, 1.0),
            "kd_rank_tie_fraction": values.get("context_ties", 0.0) / max(tie_pairs, 1.0),
            "kd_rank_live_pairs": pairs,
            "kd_dist_live_anchors": anchors,
        }

    def validation_diagnostics(self, model: nn.Module) -> dict[str, float]:
        """Score the fixed V_val context bank without gradients."""
        bank = self._targets.val_bank
        with torch.no_grad():
            student, teacher, groups = self._score(model, bank, range(len(bank.anchor_idx)))
            pair_count, anchor_count = self._live_counts(bank, range(len(bank.anchor_idx)))
            rank = kd_rank_loss(student, teacher, groups, margin=self._distill.margin)
            distribution = kd_dist_loss(student, teacher, groups)
            ties, tie_pairs = self._tie_counts(teacher, groups, self._distill.margin)
        return {
            "val_kd_rank_loss": float(rank.item()),
            "val_kd_dist_loss": float(distribution.item()),
            "val_kd_rank_tie_fraction": ties / max(tie_pairs, 1),
            "val_kd_rank_live_pairs": float(pair_count),
            "val_kd_dist_live_anchors": float(anchor_count),
        }


class KDRowBank:
    """Official-row targets for telemetry and non-rank same-batch KD terms.

    Replaces the old `KDStream` sampled anchor-context second forward: a
    teacher target exists for every official training row
    (`src/distill/teacher_targets.py`, format ``kd_row_targets_v1``), joined
    to the trainer's own rows by ``batch["_row_id"]``. Non-rank KD terms use
    the task forward; ``kd_rank`` keeps this bank only for official-row
    logit-correlation and probability-error telemetry while its relational
    losses come from :class:`KDContextStream`.

    The matched-control invariant: with ``cfg.distill`` absent or every
    weight zero, the caller never constructs a `KDRowBank`, and the training
    loop is bit-identical to the undistilled baseline -- no extra RNG draw,
    no extra forward, and no batch mutation.
    """

    def __init__(
        self,
        distill: DistillConfig,
        targets: KDRowTargets,
        *,
        train_pairs: Sequence[Pair],
        train_labels: Sequence[int],
        val_pairs: Sequence[Pair],
        val_labels: Sequence[int],
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """Verify the row-exact join and the model architecture, then stage tensors.

        Args:
            distill: The active `DistillConfig` (`.arm` selects the KD arm).
            targets: The loaded `KDRowTargets` artifact.
            train_pairs: The trainer's official training rows, in exact
                row-id order (row_id == position) -- ``positives + negatives``
                from `_training_rows`.
            train_labels: Labels aligned with `train_pairs`.
            val_pairs: The trainer's V_val classification rows, in exact
                row-id order -- `_val_cls_rows`'s pairs.
            val_labels: Labels aligned with `val_pairs`.
            model: The scorer, unwrapped or not (unwrapped internally), used
                only to validate the architecture.
            device: The training device the row tensors are staged onto.

        Raises:
            ValueError: If either row block does not equal the trainer's own
                rows, in order, exactly once. This subsumes the old
                allowed-nodes/V_val-boundary checks entirely: the joined rows
                ARE the trainer's own quarantined rows, so a row-exact join
                can never smuggle in a foreign or cross-boundary row.
            RuntimeError: On a `kd_rep_head`/`d_model` width mismatch between
                the artifact and the model.
        """
        self.arm = distill.arm
        self._w_logit = distill.w_logit
        self._w_gram = distill.w_gram
        self._w_rep = distill.w_rep
        self._w_gen = distill.w_gen

        node_ids_arr = np.asarray(targets.node_ids, dtype=object)
        self._verify_join(
            block_name="training",
            a_idx=targets.pair_a_idx,
            b_idx=targets.pair_b_idx,
            label=targets.pair_label,
            node_ids_arr=node_ids_arr,
            pairs=train_pairs,
            labels=train_labels,
        )
        self._verify_join(
            block_name="validation",
            a_idx=targets.val_pair_a_idx,
            b_idx=targets.val_pair_b_idx,
            label=targets.val_pair_label,
            node_ids_arr=node_ids_arr,
            pairs=val_pairs,
            labels=val_labels,
        )

        raw_model = _unwrapped_model(model)
        kd_rep_head = getattr(raw_model, "kd_rep_head", None)

        if distill.w_rep > 0.0:
            rep_dim = int(targets.teacher_rep.shape[1])
            if kd_rep_head is not None:
                if int(kd_rep_head.out_features) != rep_dim:
                    raise RuntimeError(
                        f"model.config.kd_rep_dim ({kd_rep_head.out_features}) does not "
                        f"match the KD targets rep_dim ({rep_dim})"
                    )
            elif int(cast(int, raw_model.d_model)) != rep_dim:
                raise RuntimeError(
                    f"model d_model ({raw_model.d_model}) does not match the KD targets "
                    f"rep_dim ({rep_dim}); set model.config.kd_rep_dim: {rep_dim}"
                )
        elif kd_rep_head is not None:
            raise RuntimeError(
                "model.config.kd_rep_dim > 0 requires distill.w_rep > 0 -- DDP would see "
                "never-grad'ed kd_rep_head parameters otherwise"
            )

        self._topo_gen: TopoGenBase | None = None
        self._latent_scale = 1.0
        topo_gen = _validate_topo_gen_distill_contract(model, distill)
        if topo_gen is not None:
            rep_dim = int(targets.teacher_rep.shape[1])
            if topo_gen.latent_dim != rep_dim:
                raise RuntimeError(
                    f"topo_gen.latent_dim ({topo_gen.latent_dim}) does not match the "
                    f"KD targets rep_dim ({rep_dim})"
                )
            scale = float(np.sqrt(np.mean(np.square(targets.teacher_rep.astype(np.float64)))))
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError(
                    f"KD teacher latent RMS must be finite and positive, got {scale}"
                )
            topo_gen.set_rms_scale(scale)
            self._topo_gen = topo_gen
            self._latent_scale = scale

        self.train_logit = torch.as_tensor(
            targets.teacher_logit, dtype=torch.float32, device=device
        )
        self.train_a_idx: torch.Tensor | None = None
        self.train_b_idx: torch.Tensor | None = None
        if self.arm == "kd_gram":
            self.train_a_idx = torch.as_tensor(targets.pair_a_idx, dtype=torch.int64, device=device)
            self.train_b_idx = torch.as_tensor(targets.pair_b_idx, dtype=torch.int64, device=device)
        self.train_rep: torch.Tensor | None = None
        if distill.w_rep > 0.0 or distill.w_gram > 0.0 or distill.w_gen > 0.0:
            self.train_rep = torch.as_tensor(
                targets.teacher_rep, dtype=torch.float16, device=device
            )
        val_teacher_rep: torch.Tensor | None = None
        if self.arm in {"kd_rep", "kd_gram"}:
            val_teacher_rep = torch.as_tensor(
                targets.val_teacher_rep, dtype=torch.float16, device=device
            )
        val_teacher_latent: torch.Tensor | None = None
        if self.arm == "kd_gen":
            val_teacher_latent = torch.as_tensor(
                targets.val_teacher_rep, dtype=torch.float16, device=device
            )
        self._val = KDValDiagnostics(
            arm=self.arm,
            teacher_logit=torch.as_tensor(
                targets.val_teacher_logit, dtype=torch.float32, device=device
            ),
            teacher_logit_np=np.asarray(targets.val_teacher_logit, dtype=np.float64),
            teacher_rep=val_teacher_rep,
            teacher_latent=val_teacher_latent,
            latent_scale=self._latent_scale,
        )

    @staticmethod
    def _verify_join(
        *,
        block_name: str,
        a_idx: np.ndarray,
        b_idx: np.ndarray,
        label: np.ndarray,
        node_ids_arr: np.ndarray,
        pairs: Sequence[Pair],
        labels: Sequence[int],
    ) -> None:
        """Verify one KD target block equals the trainer's own rows, in order, exactly once."""
        if len(a_idx) != len(pairs):
            raise ValueError(
                f"KD {block_name} block has {len(a_idx)} rows, trainer has {len(pairs)}"
            )
        artifact_a = node_ids_arr[np.asarray(a_idx)]
        artifact_b = node_ids_arr[np.asarray(b_idx)]
        trainer_a = np.asarray([pair[0] for pair in pairs], dtype=object)
        trainer_b = np.asarray([pair[1] for pair in pairs], dtype=object)
        if not np.array_equal(artifact_a, trainer_a) or not np.array_equal(artifact_b, trainer_b):
            raise ValueError(
                f"KD {block_name} block endpoints do not match the trainer's {block_name} "
                "rows in row order -- re-dump the targets against the current split"
            )
        if not np.array_equal(
            np.asarray(label, dtype=np.int64), np.asarray(list(labels), dtype=np.int64)
        ):
            raise ValueError(
                f"KD {block_name} block labels do not match the trainer's {block_name} rows"
            )

    def attach(self, batch: Batch) -> None:
        """Inject this step's normalized teacher latent for ``kd_gen``."""
        if self.arm != "kd_gen":
            return
        assert self.train_rep is not None
        rows = batch["_row_id"]
        batch["kd_teacher_latent"] = (
            self.train_rep.index_select(0, rows).float() / self._latent_scale
        )

    def loss(
        self,
        batch: Batch,
        output: dict[str, torch.Tensor],
        *,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute this step's KD loss and per-batch telemetry sums from the shared forward."""
        rows = batch["_row_id"]
        student_logit = output["logits"]
        if student_logit.dim() > 1 and student_logit.size(-1) == 1:
            student_logit = student_logit.squeeze(-1)
        student_logit = student_logit.float()
        teacher_logit = self.train_logit.index_select(0, rows)

        total = student_logit.new_zeros(())
        logit_term: torch.Tensor | None = None
        if self.arm == "kd_logit":
            logit_term = kd_logit_loss(student_logit, teacher_logit)
            total = total + self._w_logit * logit_term

        with torch.no_grad():
            student_d = student_logit.detach().double()
            teacher_d = teacher_logit.detach().double()
            stats: dict[str, float] = {
                "rows": float(student_d.numel()),
                "sum_s": float(student_d.sum().item()),
                "sum_t": float(teacher_d.sum().item()),
                "sum_s2": float((student_d * student_d).sum().item()),
                "sum_t2": float((teacher_d * teacher_d).sum().item()),
                "sum_st": float((student_d * teacher_d).sum().item()),
                "sum_prob_err": float(
                    (torch.sigmoid(student_d) - torch.sigmoid(teacher_d)).abs().sum().item()
                ),
            }
            if logit_term is not None:
                stats["sum_logit_bce"] = float(logit_term.detach().item()) * stats["rows"]

        if self.arm == "kd_rep":
            student_rep = output.get("kd_rep")
            if student_rep is None:
                student_rep = output.get("pair_repr")
            if student_rep is None:
                raise RuntimeError("kd_rep requires the model forward to emit kd_rep or pair_repr")
            assert self.train_rep is not None
            teacher_rep = self.train_rep.index_select(0, rows).float()
            total = total + self._w_rep * kd_rep_loss(student_rep.float(), teacher_rep)
            with torch.no_grad():
                cos = nn.functional.cosine_similarity(
                    student_rep.float(), teacher_rep, dim=-1, eps=1e-8
                )
                stats["sum_rep_cos"] = float(cos.sum().item())

        if self.arm == "kd_gram":
            student_rep = output.get("pair_repr")
            if student_rep is None:
                raise RuntimeError("kd_gram requires the model forward to emit pair_repr")
            assert (
                self.train_rep is not None
                and self.train_a_idx is not None
                and self.train_b_idx is not None
            )
            teacher_rep = self.train_rep.index_select(0, rows).float()
            global_rows = _gather_global_relational_rows(
                student_rep.float(),
                teacher_rep,
                self.train_a_idx.index_select(0, rows),
                self.train_b_idx.index_select(0, rows),
                rows,
                world_size=world_size,
            )
            gram_loss = kd_gram_loss(global_rows.student, global_rows.teacher)
            total = total + self._w_gram * gram_loss
            stats["sum_gram"] = float(gram_loss.detach().item()) * stats["rows"]

        if self.arm == "kd_gen":
            gen_loss = output.get("gen_loss")
            latent_sample = output.get("gen_latent_sample")
            prob_std = output.get("gen_prob_std")
            dispersion = output.get("gen_sample_dispersion")
            branch_ratio = output.get("gen_branch_ratio")
            teacher_latent = batch.get("kd_teacher_latent")
            if gen_loss is None or any(
                value is None
                for value in (
                    latent_sample,
                    prob_std,
                    dispersion,
                    branch_ratio,
                    teacher_latent,
                )
            ):
                raise RuntimeError("kd_gen model forward is missing generator outputs")
            assert latent_sample is not None
            assert prob_std is not None
            assert dispersion is not None
            assert branch_ratio is not None
            assert teacher_latent is not None
            total = total + self._w_gen * gen_loss
            with torch.no_grad():
                cos = nn.functional.cosine_similarity(
                    latent_sample.float(), teacher_latent.float(), dim=-1, eps=1e-8
                )
                stats["sum_gen_loss"] = float(gen_loss.detach().item()) * stats["rows"]
                stats["sum_latent_cos"] = float(cos.sum().item())
                stats["sum_prob_std"] = float(prob_std.float().sum().item())
                stats["sum_dispersion"] = float(dispersion.float().sum().item())
                stats["sum_branch_ratio"] = float(branch_ratio.float().sum().item())
                bin_loss = output.get("gen_sigma_bin_loss")
                bin_count = output.get("gen_sigma_bin_count")
                if bin_loss is not None and bin_count is not None:
                    for quartile in range(4):
                        stats[f"sum_sigma_q{quartile + 1}"] = float(bin_loss[quartile].item())
                        stats[f"count_sigma_q{quartile + 1}"] = float(bin_count[quartile].item())

        return total, stats

    def epoch_telemetry(
        self,
        accelerator: Accelerator,
        sums: dict[str, float],
    ) -> dict[str, float]:
        """Reduce this epoch's accumulated per-batch sums into epoch-level telemetry."""
        keys = sorted(sums)
        values = torch.tensor(
            [sums[key] for key in keys], device=accelerator.device, dtype=torch.float64
        )
        reduced = accelerator.reduce(values, reduction="sum")
        reduced_sums = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        n = reduced_sums.get("rows", 0.0)
        if n <= 0.0:
            raise RuntimeError("KD epoch telemetry has zero rows across all ranks")

        telemetry: dict[str, float] = {
            "kd_logit_corr": _pearson_from_moments(
                reduced_sums["sum_s"],
                reduced_sums["sum_t"],
                reduced_sums["sum_s2"],
                reduced_sums["sum_t2"],
                reduced_sums["sum_st"],
                n,
            ),
            "kd_prob_mae": reduced_sums["sum_prob_err"] / n,
        }
        if self.arm == "kd_logit":
            telemetry["kd_logit_loss"] = reduced_sums["sum_logit_bce"] / n
        if self.arm == "kd_rep":
            telemetry["kd_rep_cos"] = reduced_sums["sum_rep_cos"] / n
            telemetry["kd_rep_loss"] = 1.0 - telemetry["kd_rep_cos"]
        if self.arm == "kd_gram":
            telemetry["kd_gram"] = reduced_sums["sum_gram"] / n
        if self.arm == "kd_gen":
            if self._topo_gen is None:
                raise RuntimeError("kd_gen requires model.config.topo_gen")
            telemetry.update(
                {
                    "kd_gen_loss": reduced_sums["sum_gen_loss"] / n,
                    "kd_latent_cos": reduced_sums["sum_latent_cos"] / n,
                    "mc_prob_std": reduced_sums["sum_prob_std"] / n,
                    "gen_sample_dispersion": reduced_sums["sum_dispersion"] / n,
                    "gen_branch_ratio": reduced_sums["sum_branch_ratio"] / n,
                    "gen_gate": float(torch.tanh(self._topo_gen.gate).item()),
                }
            )
            for quartile in range(4):
                count = reduced_sums.get(f"count_sigma_q{quartile + 1}", 0.0)
                if count > 0.0:
                    telemetry[f"kd_gen_sigma_q{quartile + 1}"] = (
                        reduced_sums[f"sum_sigma_q{quartile + 1}"] / count
                    )
        return telemetry

    def val_diagnostics(self) -> KDValDiagnostics:
        """Return the staged validation-row teacher targets for `_evaluate_distributed`."""
        return self._val

    @property
    def global_relational(self) -> bool:
        """Whether the arm's KD scalar already covers the global DDP step."""
        return self.arm == "kd_gram"


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


def _scale_kd_loss(
    kd_bank: KDRowBank,
    kd_loss: torch.Tensor,
    *,
    local_count: int,
    global_count: int,
    world_size: int,
) -> torch.Tensor:
    """Scale row-local KD means, leaving the global Gram objective unchanged."""
    if kd_bank.global_relational:
        return kd_loss
    return scale_ddp_mean_loss(
        kd_loss,
        local_count=local_count,
        global_count=global_count,
        world_size=world_size,
    )


def _term_grad_norms(
    task_loss: torch.Tensor, kd_loss: torch.Tensor, model: nn.Module
) -> tuple[float, float]:
    """Per-epoch diagnostic: task-loss and KD-loss gradient L2 norms over shared parameters.

    `torch.autograd.grad` never executes the `AccumulateGrad` nodes DDP's
    reducer hooks are attached to (those only fire inside `Tensor.backward`),
    so this probe leaves the subsequent shared `accelerator.backward(loss)`
    call's graph and DDP's gradient-sync hooks untouched -- `retain_graph=True`
    keeps both terms' graphs alive for that later call. D2/D3 KD graphs do
    execute differentiable-all-gather backward collectives during this probe,
    so every rank must enter `_term_grad_norms` in the same order; the training
    loop guarantees that rank-symmetric call. A term with no grad_fn (e.g. a
    constant KD loss from a test double) reports a 0.0 norm rather than raising.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def _term_norm(term: torch.Tensor) -> float:
        if not term.requires_grad:
            return 0.0
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        squares = [g.float().pow(2).sum() for g in grads if g is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt().item())

    return _term_norm(task_loss), _term_norm(kd_loss)


def _topology_due(
    epoch: int, *, epochs: int, topology_every: int, classification_only: bool
) -> bool:
    """Whether the V_val topology pass runs at ``epoch``; the final epoch is always due."""
    if classification_only:
        return False
    return epoch % topology_every == 0 or epoch == epochs


def train_ddp_loop(
    model: nn.Module,
    train_loader_factory: PackedLoaderFactory,
    val_loader: Iterable[Batch],
    cfg: Config,
    accelerator: Accelerator,
    *,
    warmup_steps: int,
    artifact_dir: Path,
    profile_output: Path | None = None,
    resume_attempt: Path | None = None,
    schedule_total_steps: int | None = None,
    evaluate_fn: EvaluateFn = _evaluate_distributed,
    evaluate_cls_fn: EvaluateFn | None = None,
    kd_bank: KDRowBank | None = None,
    kd_context_stream: KDContextStream | None = None,
    require_topology: bool = False,
    val_topology_reference: ValTopologyReference | None = None,
) -> TrainResult:
    """Run fixed-epoch E2 DDP training and return rank-consistent metrics.

    Normally trains exactly ``cfg.optim.epochs`` epochs with a validation after
    every epoch (the V_val topology pass on the ``eval.topology_every`` cadence)
    and records patience counterfactually. When
    ``eval.classification_only`` is enabled, patience performs real early stopping
    and checkpoint selection uses validation AUPRC alone. Tail batches are loss-scaled with
    :func:`scale_ddp_mean_loss`; a non-finite loss on any rank aborts all ranks;
    and every epoch boundary asserts all ranks ran the same number of steps.
    Rank zero persists the completed epoch before the next epoch begins.

    Args:
        model: The scorer (not yet prepared by the accelerator).
        train_loader_factory: ``epoch -> iterable`` of this rank's assembled batches.
        val_loader: This rank's re-iterable assembled validation batches.
        cfg: The full training config.
        accelerator: DDP accelerator (see :func:`build_ddp_accelerator`).
        warmup_steps: Linear-warmup step count (then constant LR).
        artifact_dir: Persistent active-attempt directory owned by rank zero.
        profile_output: Atomic worker-profile snapshot path, when requested.
        resume_attempt: Prior attempt supplying the exact epoch-boundary state.
        schedule_total_steps: Exact optimizer-step count over all epochs, used to
            size a ``optim.scheduler`` OneCycle schedule; unused otherwise.
        evaluate_fn: Validation function; defaults to distributed validation.
            Unit tests inject a deterministic metric source.
        evaluate_cls_fn: Classification-only validation used on epochs where
            ``eval.topology_every`` skips the V_val topology pass; ``None``
            evaluates with ``evaluate_fn`` on every epoch.
        kd_bank: Optional row-aligned KD teacher targets (`KDRowBank`); its
            per-step loss is added to the scaled supervised loss before the
            shared backward, computed from the SAME student forward pass.
        kd_context_stream: Optional rank-sharded context targets for `kd_rank`.
        require_topology: When True, selection raises instead of falling back
            to max-AUPRC if any topology-due epoch has no topology metrics
            (production, e.g. a resume across the V_val protocol change).
            Unit-test stubs that inject an `evaluate_fn` without topology keep
            `False`.
        val_topology_reference: The once-per-run `ValTopologyReference`, used
            only to attach `TrainResult.val_threshold_transfer` (node count
            and sampled-only threshold). `None` leaves that field unset.

    Returns:
        The `TrainResult`, identical across ranks except for the main-rank-only
        checkpoint snapshots and rank-zero ``runtime_profile``.

    Raises:
        RuntimeError: On a non-finite loss or a per-rank step-count divergence.
    """
    optimizer = _build_optimizer(model, cfg)
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = _build_scheduler(
        optimizer,
        cfg,
        warmup_steps=warmup_steps,
        total_steps=schedule_total_steps,
    )
    world_size = accelerator.num_processes
    use_cuda = accelerator.device.type == "cuda"

    history: list[dict[str, object]] = []
    metrics_by_epoch: dict[int, EdgeMetrics] = {}
    topology_by_epoch: dict[int, ValTopologyResult | None] = {}
    best_auprc: float | None = None
    last_metrics: EdgeMetrics | None = None
    evals_without_improvement = 0
    counterfactual_stop_epoch: int | None = None
    stopped_early = False
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

    setup_error: list[str | None] = [None]
    if accelerator.is_main_process:
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            if resume_attempt is None:
                (artifact_dir / "metrics.jsonl").unlink(missing_ok=True)
        except Exception as error:
            setup_error[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(setup_error, from_process=0)
    if setup_error[0] is not None:
        raise RuntimeError(f"rank-zero attempt setup failed: {setup_error[0]}")

    start_epoch = 1
    if resume_attempt is not None:
        snapshot = _run_rank_symmetric(
            accelerator,
            "resume snapshot load",
            lambda: cast(
                dict[str, object],
                torch.load(
                    resume_attempt / "training_state.pt",
                    map_location="cpu",
                    weights_only=False,
                ),
            ),
        )
        completed_epoch = cast(int, snapshot["epoch"])
        rng_by_rank = cast(list[dict[str, object]], snapshot["rng_by_rank"])
        runtime_by_rank = cast(list[dict[str, object]], snapshot["runtime_by_rank"])

        def restore_rank_state() -> None:
            if snapshot.get("resume_supported") is not True:
                raise RuntimeError("attempt does not contain a supported resume snapshot")
            saved_config = snapshot.get("config")
            if not isinstance(saved_config, dict):
                raise RuntimeError("resume snapshot is missing its training configuration")
            saved_resume_config = dict(saved_config)
            current_resume_config = config_to_dict(cfg)
            saved_resume_config.pop("output_dir", None)
            current_resume_config.pop("output_dir", None)
            if saved_resume_config != current_resume_config:
                raise ValueError("resume configuration does not match the current training run")
            if snapshot.get("world_size") != world_size:
                raise ValueError("resume world size does not match the current training run")
            if snapshot.get("warmup_steps") != warmup_steps:
                raise ValueError("resume warmup schedule does not match the current training run")
            if snapshot.get("schedule_total_steps") != schedule_total_steps:
                raise ValueError("resume step schedule does not match the current training run")
            if completed_epoch < 1 or completed_epoch > cfg.optim.epochs:
                raise ValueError(
                    "resume snapshot epoch must be within the configured run: "
                    f"got {completed_epoch}, target epochs {cfg.optim.epochs}"
                )
            if len(rng_by_rank) != world_size or len(runtime_by_rank) != world_size:
                raise RuntimeError(
                    "resume snapshot world-size mismatch: "
                    f"snapshot={len(rng_by_rank)}, current={world_size}"
                )
            accelerator.unwrap_model(model).load_state_dict(snapshot["model_state"])
            optimizer.load_state_dict(snapshot["optimizer"])
            scheduler.load_state_dict(cast(dict[str, Any], snapshot["scheduler"]))
            scaler_state = snapshot.get("scaler")
            if scaler_state is not None:
                if accelerator.scaler is None:
                    raise RuntimeError(
                        "resume snapshot has scaler state but this worker has no scaler"
                    )
                accelerator.scaler.load_state_dict(cast(dict[str, Any], scaler_state))
            _restore_local_rng_state(rng_by_rank[accelerator.process_index], accelerator.device)

        _run_rank_symmetric(accelerator, "resume state restore", restore_rank_state)

        def load_rank_runtime() -> dict[str, int | float]:
            rank_runtime = runtime_by_rank[accelerator.process_index]
            return {
                "total_rank_local_pairs": int(rank_runtime["total_rank_local_pairs"]),
                "total_global_pairs": int(rank_runtime["total_global_pairs"]),
                "total_local_tokens": int(rank_runtime["total_local_tokens"]),
                "total_steps": int(rank_runtime["total_steps"]),
                "total_train_wall_seconds": float(rank_runtime["total_train_wall_seconds"]),
                "total_data_wait_seconds": float(rank_runtime["total_data_wait_seconds"]),
                "total_validation_seconds": float(rank_runtime["total_validation_seconds"]),
                "local_peak_memory_gib": float(rank_runtime["local_peak_memory_gib"]),
            }

        resumed_runtime = _run_rank_symmetric(
            accelerator, "resume runtime restore", load_rank_runtime
        )
        total_rank_local_pairs = int(resumed_runtime["total_rank_local_pairs"])
        total_global_pairs = int(resumed_runtime["total_global_pairs"])
        total_local_tokens = int(resumed_runtime["total_local_tokens"])
        total_steps = int(resumed_runtime["total_steps"])
        total_train_wall_seconds = float(resumed_runtime["total_train_wall_seconds"])
        total_data_wait_seconds = float(resumed_runtime["total_data_wait_seconds"])
        total_validation_seconds = float(resumed_runtime["total_validation_seconds"])
        local_peak_memory_gib = float(resumed_runtime["local_peak_memory_gib"])
        global_step = cast(int, snapshot["global_step"])
        per_epoch_profiles = cast(list[dict[str, object]], snapshot["per_epoch_profiles"])
        evals_without_improvement = cast(int, snapshot["evals_without_improvement"])
        counterfactual_stop_epoch = cast(int | None, snapshot["counterfactual_stop_epoch"])
        metrics_rows = _run_rank_symmetric(
            accelerator,
            "resume metrics load",
            lambda: [
                json.loads(line)
                for line in (artifact_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ],
        )
        if len(metrics_rows) < completed_epoch:
            raise RuntimeError(
                "resume metrics/checkpoint prefix is incomplete: "
                f"expected at least {completed_epoch} rows, got {len(metrics_rows)}"
            )
        metrics_rows = metrics_rows[:completed_epoch]
        resume_cleanup_error: list[str | None] = [None]
        if accelerator.is_main_process:
            try:
                _write_jsonl_atomic(
                    artifact_dir / "metrics.jsonl",
                    [cast(dict[str, object], row) for row in metrics_rows],
                )
                for orphan in (artifact_dir / "checkpoints").glob("epoch-*.pt"):
                    if int(orphan.stem.removeprefix("epoch-")) > completed_epoch:
                        orphan.unlink()
            except Exception as error:
                resume_cleanup_error[0] = f"{type(error).__name__}: {error}"
        broadcast_object_list(resume_cleanup_error, from_process=0)
        if resume_cleanup_error[0] is not None:
            raise RuntimeError(f"rank-zero resume-prefix cleanup failed: {resume_cleanup_error[0]}")
        for prior_epoch, prior_entry in enumerate(metrics_rows, start=1):

            def load_candidate(
                epoch: int = prior_epoch, entry: object = prior_entry
            ) -> dict[str, object]:
                candidate = cast(
                    dict[str, object],
                    torch.load(
                        artifact_dir / "checkpoints" / f"epoch-{epoch:04d}.pt",
                        map_location="cpu",
                        weights_only=False,
                    ),
                )
                if cast(int, candidate["epoch"]) != epoch:
                    raise RuntimeError(f"resume candidate epoch mismatch at epoch {epoch}")
                if candidate.get("selection_metrics") != entry:
                    raise RuntimeError(f"resume metric/candidate mismatch at epoch {epoch}")
                return candidate

            candidate = _run_rank_symmetric(
                accelerator, f"resume candidate {prior_epoch} load", load_candidate
            )
            prior_metrics = EdgeMetrics(**cast(dict[str, Any], candidate["val_metrics"]))
            history.append(cast(dict[str, object], prior_entry))
            metrics_by_epoch[prior_epoch] = prior_metrics
            topology_by_epoch[prior_epoch] = _topology_from_metrics_row(prior_entry)
        last_metrics = metrics_by_epoch[completed_epoch]
        best_auprc = max(metrics.auprc for metrics in metrics_by_epoch.values())
        start_epoch = completed_epoch + 1
        if cfg.eval.classification_only and counterfactual_stop_epoch is not None:
            stopped_early = True
            start_epoch = cfg.optim.epochs + 1

    last_heartbeat = time.monotonic()
    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        _set_topo_gen_training_stage(
            model,
            optimizer,
            cfg.distill,
            epoch=epoch,
            total_epochs=cfg.optim.epochs,
        )
        model.train()
        local_loss_sum = 0.0
        epoch_kd_loss_sum = 0.0
        epoch_kd_sums: dict[str, float] = {}
        grad_norm_task = 0.0
        grad_norm_kd = 0.0
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
        epoch_loader = train_loader_factory(epoch)
        if kd_context_stream is not None and not isinstance(epoch_loader, Sized):
            raise TypeError("kd_rank requires a training loader with a known step count")
        epoch_step_count = len(epoch_loader) if isinstance(epoch_loader, Sized) else 0
        iterator = iter(epoch_loader)
        while True:
            wait_start = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            epoch_data_wait_seconds += time.monotonic() - wait_start

            batch = _to_device(batch, accelerator.device)
            local_count, global_count = _batch_pair_counts(batch, world_size)
            if kd_bank is not None:
                kd_bank.attach(batch)

            start_event, end_event = _maybe_cuda_events(use_cuda)
            output = model(batch)
            local_mean_loss = output["loss"]
            loss = scale_ddp_mean_loss(
                local_mean_loss,
                local_count=local_count,
                global_count=global_count,
                world_size=world_size,
            )
            kd_loss: torch.Tensor | None = None
            if kd_bank is not None:
                kd_local, kd_stats = kd_bank.loss(
                    batch,
                    output,
                    world_size=world_size,
                )
                kd_loss = _scale_kd_loss(
                    kd_bank,
                    kd_local,
                    local_count=local_count,
                    global_count=global_count,
                    world_size=world_size,
                )
                for key, value in kd_stats.items():
                    epoch_kd_sums[key] = epoch_kd_sums.get(key, 0.0) + value
            if kd_context_stream is not None:
                context_loss, context_stats = kd_context_stream.loss(
                    model,
                    epoch=epoch,
                    step=epoch_steps,
                    steps=epoch_step_count,
                )
                kd_loss = context_loss if kd_loss is None else kd_loss + context_loss
                for key, value in context_stats.items():
                    epoch_kd_sums[key] = epoch_kd_sums.get(key, 0.0) + value
            if kd_loss is not None:
                if epoch_steps == 0:
                    # Per-epoch gradient-norm probe, before the shared backward:
                    # `torch.autograd.grad` never executes the `AccumulateGrad`
                    # nodes DDP's reducer hooks are attached to (those only fire
                    # inside `Tensor.backward`), so this leaves the subsequent
                    # `accelerator.backward(loss)` call's graph and DDP's
                    # gradient-sync hooks untouched.
                    grad_norm_task, grad_norm_kd = _term_grad_norms(loss, kd_loss, model)
                loss = loss + kd_loss
                epoch_kd_loss_sum += float(kd_loss.detach().float().item())

            if not _all_ranks_loss_finite(loss, accelerator):
                raise RuntimeError(f"non-finite training loss on at least one rank (epoch {epoch})")

            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            _step_scheduler(scheduler)
            _set_topo_gen_training_stage(
                model,
                optimizer,
                cfg.distill,
                epoch=epoch,
                total_epochs=cfg.optim.epochs,
            )
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
            heartbeat_now = time.monotonic()
            if accelerator.is_main_process and heartbeat_now - last_heartbeat >= 60.0:
                logger.info(
                    "ddp heartbeat epoch=%d/%d global_step=%d loss=%.4f lr=%.3e",
                    epoch,
                    cfg.optim.epochs,
                    global_step,
                    float(local_mean_loss.detach().float().item()),
                    float(scheduler.get_last_lr()[0]),
                )
                last_heartbeat = heartbeat_now

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
        train_kd_loss: float | None = None
        if (kd_bank is not None or kd_context_stream is not None) and epoch_steps > 0:
            global_kd_loss_sum = accelerator.reduce(
                torch.tensor(
                    epoch_kd_loss_sum,
                    device=accelerator.device,
                    dtype=torch.float64,
                ),
                reduction="sum",
            )
            train_kd_loss = float(global_kd_loss_sum.item()) / float(epoch_steps * world_size)
        epoch_kd_telemetry: dict[str, float] = {}
        if kd_bank is not None and epoch_steps > 0:
            epoch_kd_telemetry = kd_bank.epoch_telemetry(accelerator, epoch_kd_sums)
        if kd_context_stream is not None and epoch_steps > 0:
            epoch_kd_telemetry.update(kd_context_stream.epoch_telemetry(accelerator, epoch_kd_sums))
        validation_start = time.monotonic()
        run_topology = _topology_due(
            epoch,
            epochs=cfg.optim.epochs,
            topology_every=cfg.eval.topology_every,
            classification_only=cfg.eval.classification_only,
        )
        epoch_evaluate_fn = (
            evaluate_fn if run_topology or evaluate_cls_fn is None else evaluate_cls_fn
        )
        outcome = epoch_evaluate_fn(model, val_loader, accelerator)
        metrics = outcome.metrics
        if use_cuda:
            torch.cuda.synchronize(accelerator.device)
        local_validation_seconds = time.monotonic() - validation_start
        validation_times = accelerator.gather(
            torch.tensor([local_validation_seconds], device=accelerator.device, dtype=torch.float64)
        )
        validation_seconds = float(validation_times.max().item())
        total_validation_seconds += validation_seconds
        last_metrics = metrics
        entry: dict[str, object] = {
            "epoch": epoch,
            "attempt_id": artifact_dir.name,
            "global_step": global_step,
            "timestamp": datetime.now(UTC).isoformat(),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss,
            "val_auroc": metrics.auroc,
            "val_auprc": metrics.auprc,
            "val_ece": metrics.ece,
            "val_brier": metrics.brier,
        }
        if outcome.task_loss is not None:
            entry["val_task_loss"] = outcome.task_loss
        if train_kd_loss is not None:
            entry["train_kd_loss"] = train_kd_loss
        entry.update(epoch_kd_telemetry)
        if kd_bank is not None or kd_context_stream is not None:
            grad_norm_tensor = accelerator.reduce(
                torch.tensor(
                    [grad_norm_task, grad_norm_kd], device=accelerator.device, dtype=torch.float64
                ),
                reduction="mean",
            )
            entry["grad_norm_task"] = float(grad_norm_tensor[0].item())
            entry["grad_norm_kd"] = float(grad_norm_tensor[1].item())
        if outcome.kd is not None:
            entry.update(outcome.kd)
        if outcome.topology is not None:
            entry.update(
                {
                    "val_gs_bfs": outcome.topology.metrics.gs,
                    "val_rd_bfs": outcome.topology.metrics.rd,
                    "val_degree_mmd_ratio": outcome.topology.metrics.degree_mmd,
                    "val_clustering_mmd_ratio": outcome.topology.metrics.clustering_mmd,
                    "val_spectral_mmd_ratio": outcome.topology.metrics.spectral_mmd,
                    "val_threshold": outcome.topology.threshold,
                }
            )
        history.append(entry)
        metrics_by_epoch[epoch] = metrics
        topology_by_epoch[epoch] = outcome.topology
        improved = best_auprc is None or metrics.auprc > best_auprc
        if improved:
            best_auprc = metrics.auprc
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1
            if evals_without_improvement >= cfg.eval.patience and counterfactual_stop_epoch is None:
                counterfactual_stop_epoch = epoch

        if accelerator.is_main_process:
            logger.info(
                "ddp eval epoch %d/%d: train_loss %.4f val_auroc %.4f val_auprc %.4f%s",
                epoch,
                cfg.optim.epochs,
                train_loss,
                metrics.auroc,
                metrics.auprc,
                " (new AUPRC high)" if improved else "",
            )
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
        entry.update(
            {
                "epoch_steps": epoch_steps,
                "epoch_global_pairs": epoch_global_pairs,
                "epoch_local_pairs": epoch_local_pairs,
                "epoch_local_tokens": epoch_local_tokens,
                "epoch_wall_seconds": epoch_wall_seconds,
                "epoch_data_wait_seconds": epoch_data_wait_seconds,
                "epoch_compute_seconds": epoch_compute_seconds,
                "validation_seconds": validation_seconds,
                "global_pairs_per_second": (
                    epoch_global_pairs / epoch_wall_seconds if epoch_wall_seconds > 0 else 0.0
                ),
            }
        )

        rng_by_rank = _gather_rank_objects(accelerator, _local_rng_state(accelerator.device))
        runtime_by_rank = _gather_rank_objects(
            accelerator,
            {
                "total_rank_local_pairs": total_rank_local_pairs,
                "total_global_pairs": total_global_pairs,
                "total_local_tokens": total_local_tokens,
                "total_steps": total_steps,
                "total_train_wall_seconds": total_train_wall_seconds,
                "total_data_wait_seconds": total_data_wait_seconds,
                "total_validation_seconds": total_validation_seconds,
                "local_peak_memory_gib": local_peak_memory_gib,
            },
        )
        io_error: list[str | None] = [None]
        if accelerator.is_main_process:
            try:
                checkpoint_path = artifact_dir / "checkpoints" / f"epoch-{epoch:04d}.pt"
                model_state = _cpu_state_dict(accelerator, model)
                checkpoint = _checkpoint_payload(
                    model_state,
                    cfg,
                    resolve_model_kwargs(cfg.model),
                    epoch,
                    metrics,
                    config_to_dict(cfg),
                )
                checkpoint["selection_metrics"] = entry
                _torch_save_atomic(checkpoint, checkpoint_path)
                _append_jsonl_durable(artifact_dir / "metrics.jsonl", entry)
                if profile_output is not None:
                    _write_json_atomic(
                        profile_output,
                        {
                            "status": "running",
                            "epochs_completed": epoch,
                            "validations_completed": epoch,
                            "global_step": global_step,
                            "counterfactual_stop_epoch": counterfactual_stop_epoch,
                            "per_epoch": per_epoch_profiles,
                        },
                    )
                _write_json_atomic(
                    artifact_dir / "progress.json",
                    {
                        "status": "running",
                        "last_completed_epoch": epoch,
                        "epochs_total": cfg.optim.epochs,
                        "global_step": global_step,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
                # This atomic replacement is the epoch commit record and must be
                # last: any earlier orphan candidate/metric can be ignored using
                # the completed epoch stored here.
                _torch_save_atomic(
                    {
                        "resume_supported": True,
                        "config": config_to_dict(cfg),
                        "world_size": world_size,
                        "warmup_steps": warmup_steps,
                        "schedule_total_steps": schedule_total_steps,
                        "model_state": model_state,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": (
                            accelerator.scaler.state_dict()
                            if accelerator.scaler is not None
                            else None
                        ),
                        "epoch": epoch,
                        "global_step": global_step,
                        "rng_by_rank": rng_by_rank,
                        "runtime_by_rank": runtime_by_rank,
                        "per_epoch_profiles": per_epoch_profiles,
                        "evals_without_improvement": evals_without_improvement,
                        "counterfactual_stop_epoch": counterfactual_stop_epoch,
                    },
                    artifact_dir / "training_state.pt",
                )
            except Exception as error:
                io_error[0] = f"{type(error).__name__}: {error}"
        broadcast_object_list(io_error, from_process=0)
        if io_error[0] is not None:
            raise RuntimeError(f"rank-zero epoch artifact write failed: {io_error[0]}")
        if cfg.eval.classification_only and counterfactual_stop_epoch == epoch:
            stopped_early = True
            if accelerator.is_main_process:
                logger.info(
                    "early stopping at epoch %d (%d evals without val-AUPRC improvement)",
                    epoch,
                    cfg.eval.patience,
                )
            break
    if not metrics_by_epoch or last_metrics is None:
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
    last_epoch = max(metrics_by_epoch)
    runtime_profile: dict[str, object] = {
        "status": "trained",
        "global_step": global_step,
        "epochs_completed": last_epoch,
        "validations_completed": last_epoch,
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
        "stopped_early": stopped_early,
        "per_epoch": per_epoch_profiles,
    }

    # Selection over the whole run: mean rank on AUPRC plus all five topology
    # metrics over the epochs whose due V_val pass ran (production; the
    # eval.topology_every cadence skips the rest), best AUPRC otherwise
    # (unit tests inject evaluate_fn stubs without a topology context).
    if require_topology and any(
        topology_by_epoch.get(epoch) is None
        for epoch in metrics_by_epoch
        if _topology_due(
            epoch,
            epochs=cfg.optim.epochs,
            topology_every=cfg.eval.topology_every,
            classification_only=cfg.eval.classification_only,
        )
    ):
        raise RuntimeError(
            "resume across the V_val protocol change unsupported; start a fresh attempt dir"
        )
    candidates = [
        CheckpointCandidate(
            epoch=epoch, auprc=metrics_by_epoch[epoch].auprc, topology=topology.metrics
        )
        for epoch in sorted(metrics_by_epoch)
        if (topology := topology_by_epoch.get(epoch)) is not None
    ]
    if candidates:
        selected = select_checkpoint(candidates)
        assert selected is not None
        best_epoch = selected.epoch
    elif require_topology:
        raise RuntimeError(
            "resume across the V_val protocol change unsupported; start a fresh attempt dir"
        )
    else:
        if cfg.eval.classification_only:
            best_epoch = max(
                sorted(metrics_by_epoch),
                key=lambda epoch: metrics_by_epoch[epoch].auprc,
            )
        else:
            best_epoch = max(
                sorted(metrics_by_epoch),
                key=lambda epoch: (metrics_by_epoch[epoch].auprc, epoch),
            )
    best_metrics = metrics_by_epoch[best_epoch]

    val_threshold_transfer: ValThresholdTransfer | None = None
    if val_topology_reference is not None:
        selected_topology = topology_by_epoch.get(best_epoch)
        if selected_topology is not None:
            val_threshold_transfer = ValThresholdTransfer(
                n_val=len(val_topology_reference.nodes),
                threshold=selected_topology.threshold,
            )

    last_state: dict[str, torch.Tensor] = {}
    best_state: dict[str, torch.Tensor] = {}
    final_io_error: list[str | None] = [None]
    if accelerator.is_main_process:
        try:
            last_state = _cpu_state_dict(accelerator, model)
            selected_checkpoint = torch.load(
                artifact_dir / "checkpoints" / f"epoch-{best_epoch:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            best_state = cast(dict[str, torch.Tensor], selected_checkpoint["model_state"])
            _write_json_atomic(
                artifact_dir / "progress.json",
                {
                    "status": "trained",
                    "last_completed_epoch": last_epoch,
                    "epochs_total": cfg.optim.epochs,
                    "global_step": global_step,
                    "selected_epoch": best_epoch,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            if profile_output is not None:
                _write_json_atomic(profile_output, runtime_profile)
        except Exception as error:
            final_io_error[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(final_io_error, from_process=0)
    if final_io_error[0] is not None:
        raise RuntimeError(f"rank-zero training finalization failed: {final_io_error[0]}")

    return TrainResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        last_state_dict=last_state,
        last_epoch=last_epoch,
        last_val_metrics=last_metrics,
        history=history,
        stopped_early=stopped_early,
        counterfactual_stop_epoch=counterfactual_stop_epoch,
        runtime_profile=runtime_profile,
        val_threshold_transfer=val_threshold_transfer,
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


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
    _fsync_directory(path.parent)


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Atomically replace a JSONL prefix used to seed a resumed attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
    _fsync_directory(path.parent)


def _write_json_rank_zero(accelerator: Accelerator, path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as pretty JSON from the main rank only (atomically)."""
    if accelerator.is_main_process:
        _write_json_atomic(path, payload)


def _append_jsonl_durable(path: Path, payload: dict[str, object]) -> None:
    """Append one complete JSONL record and force it to stable storage."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _torch_save_atomic(payload: dict[str, object], path: Path) -> None:
    """Write a torch payload without exposing a partially serialized checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    with temp_path.open("rb") as handle:
        os.fsync(handle.fileno())
    temp_path.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes after an atomic artifact replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _local_rng_state(device: torch.device) -> dict[str, object]:
    """Capture this rank's stochastic state for an exact future resume."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_local_rng_state(state: dict[str, object], device: torch.device) -> None:
    """Restore the stochastic state captured for this DDP rank."""
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(tuple[Any, ...], state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch_cpu"]))
    cuda_state = cast(torch.Tensor | None, state["torch_cuda"])
    if cuda_state is not None:
        if device.type != "cuda":
            raise RuntimeError("CUDA RNG state cannot be restored on a CPU worker")
        torch.cuda.set_rng_state(cuda_state, device)


def _gather_rank_objects(accelerator: Accelerator, payload: T) -> list[T]:
    """Return one gathered object per rank, normalizing Accelerate's local case."""
    gathered = gather_object([payload])
    result = cast(list[T], gathered)
    if len(result) != accelerator.num_processes:
        raise RuntimeError(
            "rank recovery-state gather mismatch: "
            f"expected {accelerator.num_processes}, got {len(result)}"
        )
    return result


def _run_rank_symmetric(accelerator: Accelerator, context: str, operation: Callable[[], T]) -> T:
    """Run rank-local resume work and turn any rank's failure into an all-rank error."""
    result: T | None = None
    local_error: str | None = None
    try:
        result = operation()
    except Exception as error:
        local_error = f"rank {accelerator.process_index}: {type(error).__name__}: {error}"
    errors = _gather_rank_objects(accelerator, local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"{context} failed on at least one rank: {'; '.join(failures)}")
    return cast(T, result)


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
    if _validate_topo_gen_distill_contract(model, cfg.distill) is not None:
        raise RuntimeError(
            "ddp-mode probe does not support kd_gen because it has no KDRowBank "
            "teacher-latent/gen-loss path; use epoch-probe or train"
        )
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
    if cfg.runtime is None:
        raise ValueError("DDP worker modes require a configured cfg.runtime")
    runtime = cfg.runtime

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
    factory, val_loader, warmup_steps, schedule_total_steps = _build_packed_v3_1_loaders(
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

    val_split = assembled.val_split

    val_cls_pairs, val_cls_labels = _val_cls_rows(val_split, assembled.exclude_nodes)
    num_val_rows = len(val_cls_pairs)

    kd_bank: KDRowBank | None = None
    kd_context_stream: KDContextStream | None = None
    kd_val: KDValDiagnostics | None = None
    if cfg.distill is not None and cfg.distill.active:
        positives, negatives = _training_rows(val_split, assembled.exclude_nodes)
        targets = load_kd_targets(Path(cfg.distill.targets_path))
        kd_bank = KDRowBank(
            cfg.distill,
            targets,
            train_pairs=positives + negatives,
            train_labels=[1] * len(positives) + [0] * len(negatives),
            val_pairs=val_cls_pairs,
            val_labels=val_cls_labels,
            model=model,
            device=accelerator.device,
        )
        kd_val = kd_bank.val_diagnostics()
        if cfg.distill.arm == "kd_rank":
            context_targets = load_kd_context_targets(Path(cfg.distill.context_targets_path))
            kd_context_stream = KDContextStream(
                cfg.distill,
                context_targets,
                table,
                allowed_nodes=frozenset(val_split.train_nodes - assembled.exclude_nodes),
                forbidden_internal_nodes=val_split.v_val,
                epochs=cfg.optim.epochs,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
            )
            kd_val = replace(kd_val, context_stream=kd_context_stream)
        if accelerator.is_main_process:
            logger.info(
                "KD active: arm=%s targets=%s context_targets=%s",
                cfg.distill.arm,
                cfg.distill.targets_path,
                cfg.distill.context_targets_path or None,
            )

    val_label_smoothing = float(
        cast(float, resolve_model_kwargs(cfg.model).get("label_smoothing", 0.0))
    )
    reference: ValTopologyReference | None = None
    cls_evaluate_fn = cast(
        EvaluateFn,
        functools.partial(
            _evaluate_distributed,
            expected_row_ids=np.arange(num_val_rows, dtype=np.int64),
            kd_val=kd_val,
            label_smoothing=val_label_smoothing,
        ),
    )
    evaluate_cls_fn: EvaluateFn | None = None
    if cfg.eval.classification_only:
        evaluate_fn = cls_evaluate_fn
    else:
        evaluate_cls_fn = cls_evaluate_fn
        reference = build_val_topology_reference(val_split)
        universe = val_ball_union_universe(val_split)
        u_idx, v_idx = universe.u_idx, universe.v_idx
        node_index = table.manifest.node_index()
        lengths_by_node = {record.node_id: record.length for record in table.manifest.nodes}
        node_positions = np.array([node_index[node] for node in reference.nodes], dtype=np.int64)
        evaluate_fn = cast(
            EvaluateFn,
            functools.partial(
                _evaluate_two_pass,
                expected_row_ids=np.arange(num_val_rows, dtype=np.int64),
                topology_eval_fn=cast(
                    TopologyEvalFn,
                    functools.partial(
                        _evaluate_val_universe,
                        table=table,
                        node_a_all=torch.from_numpy(node_positions[universe.u_idx]).to(torch.int64),
                        node_b_all=torch.from_numpy(node_positions[universe.v_idx]).to(torch.int64),
                        boundary=max(lengths_by_node[node] for node in reference.nodes),
                        batch_pairs=runtime.max_pairs_per_rank,
                        u_idx=u_idx,
                        v_idx=v_idx,
                        reference=reference,
                    ),
                ),
                kd_val=kd_val,
                label_smoothing=val_label_smoothing,
            ),
        )

    if args.ddp_mode == "epoch-probe":
        one_epoch_cfg = replace(cfg, optim=replace(cfg.optim, epochs=1))
        probe_setup: list[dict[str, str | None]] = [{"path": None, "error": None}]
        if accelerator.is_main_process:
            try:
                args.profile_output.parent.mkdir(parents=True, exist_ok=True)
                probe_setup[0]["path"] = tempfile.mkdtemp(
                    prefix=".epoch-probe-", dir=args.profile_output.parent
                )
            except Exception as error:
                probe_setup[0]["error"] = f"{type(error).__name__}: {error}"
        broadcast_object_list(probe_setup, from_process=0)
        if probe_setup[0]["error"] is not None:
            raise RuntimeError(f"rank-zero epoch-probe setup failed: {probe_setup[0]['error']}")
        probe_path = probe_setup[0]["path"]
        if probe_path is None:
            raise RuntimeError("rank-zero epoch-probe setup returned no artifact path")
        probe_artifact_dir = Path(probe_path)
        try:
            result, elapsed = _run_timed_epoch_probe(
                accelerator,
                lambda: train_ddp_loop(
                    model,
                    factory,
                    val_loader,
                    one_epoch_cfg,
                    accelerator,
                    warmup_steps=warmup_steps,
                    artifact_dir=probe_artifact_dir,
                    schedule_total_steps=schedule_total_steps,
                    evaluate_fn=evaluate_fn,
                    evaluate_cls_fn=evaluate_cls_fn,
                    kd_bank=kd_bank,
                    kd_context_stream=kd_context_stream,
                    require_topology=not cfg.eval.classification_only,
                ),
            )
        except BaseException:
            if accelerator.is_main_process:
                shutil.rmtree(probe_artifact_dir, ignore_errors=True)
            raise
        cleanup_error: list[str | None] = [None]
        if accelerator.is_main_process:
            try:
                shutil.rmtree(probe_artifact_dir)
            except Exception as error:
                cleanup_error[0] = f"{type(error).__name__}: {error}"
        broadcast_object_list(cleanup_error, from_process=0)
        if cleanup_error[0] is not None:
            raise RuntimeError(f"rank-zero epoch-probe cleanup failed: {cleanup_error[0]}")
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
        artifact_dir=cfg.output_dir,
        profile_output=args.profile_output,
        resume_attempt=args.resume_attempt,
        schedule_total_steps=schedule_total_steps,
        evaluate_fn=evaluate_fn,
        evaluate_cls_fn=evaluate_cls_fn,
        kd_bank=kd_bank,
        kd_context_stream=kd_context_stream,
        require_topology=not cfg.eval.classification_only,
        val_topology_reference=reference,
    )
    finalization_error: list[str | None] = [None]
    if accelerator.is_main_process:
        try:
            write_outputs(result, cfg, model_kwargs, assembled.dropped_pair_counts)
            result.runtime_profile["status"] = "complete"
            _write_json_atomic(args.profile_output, result.runtime_profile)
            _write_json_atomic(
                cfg.output_dir / "progress.json",
                {
                    "status": "complete",
                    "last_completed_epoch": result.last_epoch,
                    "epochs_total": cfg.optim.epochs,
                    "global_step": result.runtime_profile["global_step"],
                    "selected_epoch": result.best_epoch,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as error:
            finalization_error[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(finalization_error, from_process=0)
    if finalization_error[0] is not None:
        raise RuntimeError(f"rank-zero final artifact write failed: {finalization_error[0]}")
    if accelerator.is_main_process:
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
    if cfg.distill is not None and cfg.distill.active:
        raise ValueError(
            "distill training runs only through the DDP pipeline path "
            "(hpc/run.sh train <config>), not the direct single-process debug CLI"
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
        "assembled data: %d shared training positives, dropped pairs %s",
        len(assembled.training_positives),
        assembled.dropped_pair_counts,
    )
    model = build_model(cfg)
    model_kwargs = resolve_model_kwargs(cfg.model)

    if cfg.model.family == "v3_1":
        factory, val_loader = _build_v3_1_loaders(cfg, assembled)
    else:
        factory, val_loader = _build_f0_loaders(cfg, assembled)

    schedule_total_steps = (
        _count_single_process_steps(factory, cfg) if cfg.optim.scheduler is not None else None
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    config_dict = config_to_dict(cfg)

    def on_eval(entry: dict[str, object], improved: bool, metrics: EdgeMetrics) -> None:
        """Append a metrics line and persist checkpoints incrementally."""
        _append_jsonl_durable(metrics_path, entry)
        state = _cpu_state_dict(accelerator, model)
        epoch = int(cast(int, entry["epoch"]))
        _torch_save_atomic(
            _checkpoint_payload(state, cfg, model_kwargs, epoch, metrics, config_dict),
            cfg.output_dir / "last.pt",
        )
        if improved:
            _torch_save_atomic(
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
        schedule_total_steps=schedule_total_steps,
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
