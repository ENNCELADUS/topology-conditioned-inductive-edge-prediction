r"""EgoStitch Stage-1 training worker (spec Sec 13; auto-sized H20 DDP).

Drop-in worker for the `src.e2_pipeline` orchestrator: implements the same
``--ddp-mode train`` CLI contract, runtime-profile schema,
and Task-4 checkpoint payload as ``src.train_b0``, over the EgoStitch two-stream
composite step (node stream -> L_recon/L_ssl/L_real; edge stream -> L_edge; the
joint-pair stream joins in Stage 3). ``--token-budget-per-rank`` is
reinterpreted for this family as the per-rank node-stream batch size ``B_n``
(spec Sec 13.13); the runtime budget is config-driven, not the E2 60-minute pin.

The worker executes the plan-bound formal schedule on ``V_fit`` and validates
on ``V_hold``. Model-quality diagnostics are recorded as telemetry; they do not
authorize or block checkpoint publication, scoring, or evaluation.

Launch (formal):

    accelerate launch --num_processes <visible-H20-count> --mixed_precision bf16 \
        -m src.train_egostitch --config configs/egostitch_e2e_v3_full_breadth_first.yaml \
        --ddp-mode train --run-kind formal --pack-dir <pack> --output-dir <out> \
        --token-budget-per-rank 256 --profile-output <profile.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair
from src.data.ego_targets import EgoTargetBuilder, EgoTargets
from src.data.feature_stats import FeatureStats, feature_stats_for_universe
from src.data.features import FeatureStore, build_f0_matrix
from src.data.grounding import build_grounding_pool
from src.data.internal_holdout import InternalHoldoutPartition, derive_internal_holdout
from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable
from src.data.pairs import NegativeSampler
from src.data.partition import derive_partition
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.eval.graph_metrics import MMDConfig, clustering_histogram, mmd_squared
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1
from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    GatedCrossAttention,
    masks_for_null,
    sample_branch_masks,
)
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.generator import StitchedGraph
from src.model.egostitch.graph import GraphEmbedding
from src.model.egostitch.imagine import NULL_MODE_ALL, NULL_MODE_CONTENT, NULL_MODE_FULL, SlotSet
from src.model.egostitch.losses import stage1_family_tensors, stage1_total
from src.train_b0 import (
    EvalConfig,
    ModelConfig,
    _as_float,
    _as_int,
    _as_mapping,
    _as_str,
    _as_str_list,
    _check_no_unknown_keys,
    _require,
    _state_digest,
    _write_json_rank_zero,
)

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = "benchmark_2025_neurips"
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_PACK_F0_FILENAME = "f0_matrix.pt"
_PACK_GROUNDING_FILENAME = "grounding.npz"
_PACK_VALIDATION_GROUNDING_FILENAME = "grounding_validation.npz"
_PACK_MANIFEST_FILENAME = "manifest.json"
_PACK_FEATURE_STATS_FILENAME = "feature_stats.npz"
def _egostitch_ddp_kwargs(
    *, find_unused_parameters: bool = True
) -> DistributedDataParallelKwargs:
    """Return DDP settings for EgoStitch's conditionally unused heads."""
    return DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


def build_egostitch_ddp_accelerator(
    mixed_precision: str,
    *,
    find_unused_parameters: bool = True,
) -> Accelerator:
    """Build the EgoStitch distributed accelerator."""
    return Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[
            _egostitch_ddp_kwargs(
                find_unused_parameters=find_unused_parameters
            )
        ],
    )


# The complete execution-context domain: ``formal`` produces registered results,
# and ``debug`` is *derived* from ``--max-steps``
# by `write_run_start_metadata`/`write_outputs` rather than selected on the
# CLI. It is named here because `run_metadata.json` publishes it and
# `src.experiments.probes` reads it back.
E2ERunKind = Literal["formal", "debug"]

# The single validation universe (`V_hold = V_qual union V_select`). It is also
# the grounding-pool ``role_universe`` identity.
_E2E_VALIDATION_ROLE: Literal["V_hold"] = "V_hold"


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class EgoDataConfig:
    """The ``data:`` config section for the EgoStitch worker.

    Attributes:
        root: Data root containing the benchmark and feature packages.
        strategy: Split strategy name (Benchmark-A = ``breadth_first``).
        train_positives: Pinned to ``e_sup`` (spec Sec 9.3).
        negative_ratio: Negatives per positive in the edge stream (spec 1:5).
        partition_seed: Seed of the message/supervision partition.
        msg_fraction: Message share of train positives (spec 0.8).
        node_batch: Per-rank node-stream batch size ``B_n``.
        edge_batch: Per-rank edge-stream batch size ``B_e``.
        f0_cache: F0 matrix cache path used outside packed execution; DDP modes
            use the pack directory instead.
        grounding_cache: Grounding-pool cache path (same convention).
        expected_missing_features: Exact graph nodes expected to lack features.
        pack_dir: Raw-token pack directory (spec Sec 13.18 family only; the
            same packed-feature store the B0 V3.1 loader consumes). ``None``
            when no packed execution is requested.
    """

    root: Path
    strategy: str
    train_positives: str
    negative_ratio: int
    partition_seed: int
    msg_fraction: float
    node_batch: int
    edge_batch: int
    f0_cache: Path
    grounding_cache: Path
    expected_missing_features: list[str]
    pack_dir: Path | None = None


@dataclass(frozen=True)
class EgoOptimConfig:
    """The ``optim:`` config section.

    Attributes:
        lr: AdamW learning rate (post-warmup constant).
        weight_decay: AdamW weight decay.
        epochs: Fixed epoch count (counterfactual early stop is bookkeeping).
        warmup_steps: Linear LR warmup steps.
        grad_clip: Gradient-norm clip; 0 disables.
    """

    lr: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    grad_clip: float


@dataclass(frozen=True)
class EgoRuntimeConfig:
    """Runtime contract for the active E2E ``pack -> train -> publish`` path."""

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
    train_eval_budget_seconds: int
    artifact_budget_seconds: int


@dataclass(frozen=True)
class EgoDiagnosticsConfig:
    """Registered training-fidelity and loss-balance diagnostics."""

    gradient_probe_interval: int
    gradient_imbalance_ratio: float
    gradient_imbalance_steps: int
    probe_s1_abs_mean_max: float
    selection_auprc_tolerance: float
    topk_fraction: float


@dataclass(frozen=True)
class EgoStitchTrainingConfig:
    """Strict §13.19 stability-screen training controls.

    Every executable ``egostitch_e2e`` configuration on the active branch must
    include this section. Historical v1 configurations and their executable
    behavior are archived on ``archive/egostitch-e2e-v1``.
    """

    positive_weight: float = 5.0
    phase_a_fraction: float = 0.2
    phase_b_fraction: float = 0.1
    lr_peak: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    clip_norm: float = 1.0
    pair_encoder_clip_norm: float = 3.0
    generator_clip_norm: float = 3.0
    clip_immediate_abort: float = 1e-3
    clip_persistent_threshold: float = 0.1
    clip_persistent_steps: int = 10
    family_ratio_abort: float = 50.0
    family_ratio_probes: int = 4
    collapse_fraction: float = 0.25
    collapse_floor: float = 1e-4
    collapse_validations: int = 2
    selection_auprc_tolerance: float = 0.02
    selection_mmd_tolerance: float = 1e-6
    residual_ratio_min: float = 1e-3


@dataclass(frozen=True)
class EgoConfig:
    """The full validated EgoStitch worker configuration.

    Attributes:
        model: The active ``family: egostitch_e2e`` model configuration.
        data: Data assembly settings.
        optim: Optimizer settings.
        eval: VAL-CRITERION bookkeeping settings.
        seed: Global seed.
        output_dir: Artifact directory.
        mixed_precision: ``"no"`` or ``"bf16"``.
        runtime: Orchestrator runtime contract (``token_budget`` reinterpreted
            as the node batch ``B_n``; no frozen E2 probe list).
    """

    model: ModelConfig
    data: EgoDataConfig
    optim: EgoOptimConfig
    diagnostics: EgoDiagnosticsConfig
    eval: EvalConfig
    seed: int
    output_dir: Path
    mixed_precision: str
    runtime: EgoRuntimeConfig | None = None
    training: EgoStitchTrainingConfig | None = None
    run_kind: E2ERunKind | None = None


@dataclass(frozen=True)
class EgoCliArgs:
    """Parsed CLI arguments (the shared ``src.train_b0`` worker contract)."""

    config: Path
    seed: int | None
    output_dir: Path | None
    max_steps: int | None = None
    ddp_mode: str | None = None
    pack_dir: Path | None = None
    token_budget_per_rank: int | None = None
    profile_output: Path | None = None
    run_kind: E2ERunKind | None = None


def load_config(path: Path) -> EgoConfig:
    """Load and strictly validate an EgoStitch worker YAML config.

    Args:
        path: The YAML config path.

    Returns:
        The validated `EgoConfig`.

    Raises:
        ValueError: On any schema violation.
    """
    raw = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "<root>")
    _check_no_unknown_keys(
        raw,
        (
            "model",
            "data",
            "optim",
            "diagnostics",
            "eval",
            "seed",
            "output_dir",
            "mixed_precision",
            "runtime",
            "training",
        ),
        "",
    )

    model_raw = _as_mapping(_require(raw, "model", ""), "model")
    _check_no_unknown_keys(model_raw, ("family", "config"), "model")
    family = _as_str(_require(model_raw, "family", "model."), "model.family")
    if family != _EGOSTITCH_E2E_FAMILY:
        raise ValueError(f"model.family must be {_EGOSTITCH_E2E_FAMILY!r}, got {family!r}")
    model_kwargs = _as_mapping(model_raw.get("config") or {}, "model.config")
    E2EConfig.from_mapping(model_kwargs)  # validate eagerly, fail loudly
    model = ModelConfig(family=family, config=dict(model_kwargs))

    data_raw = _as_mapping(_require(raw, "data", ""), "data")
    data_keys = (
        "root",
        "strategy",
        "train_positives",
        "negative_ratio",
        "partition_seed",
        "msg_fraction",
        "node_batch",
        "edge_batch",
        "f0_cache",
        "grounding_cache",
        "expected_missing_features",
        "pack_dir",
    )
    _check_no_unknown_keys(data_raw, data_keys, "data")
    train_positives = _as_str(
        _require(data_raw, "train_positives", "data."), "data.train_positives"
    )
    if train_positives != "e_sup":
        raise ValueError(f"data.train_positives is pinned to 'e_sup', got {train_positives!r}")
    msg_fraction = _as_float(data_raw.get("msg_fraction", 0.8), "data.msg_fraction")
    if not 0.0 < msg_fraction < 1.0:
        raise ValueError(f"data.msg_fraction must be in (0, 1), got {msg_fraction}")
    data = EgoDataConfig(
        root=Path(_as_str(_require(data_raw, "root", "data."), "data.root")),
        strategy=_as_str(_require(data_raw, "strategy", "data."), "data.strategy"),
        train_positives=train_positives,
        negative_ratio=_as_int(
            _require(data_raw, "negative_ratio", "data."), "data.negative_ratio"
        ),
        partition_seed=_as_int(
            _require(data_raw, "partition_seed", "data."), "data.partition_seed"
        ),
        msg_fraction=msg_fraction,
        node_batch=_as_int(_require(data_raw, "node_batch", "data."), "data.node_batch"),
        edge_batch=_as_int(_require(data_raw, "edge_batch", "data."), "data.edge_batch"),
        f0_cache=Path(_as_str(_require(data_raw, "f0_cache", "data."), "data.f0_cache")),
        grounding_cache=Path(
            _as_str(_require(data_raw, "grounding_cache", "data."), "data.grounding_cache")
        ),
        expected_missing_features=_as_str_list(
            data_raw.get("expected_missing_features", []), "data.expected_missing_features"
        ),
        pack_dir=(
            Path(_as_str(data_raw["pack_dir"], "data.pack_dir")) if "pack_dir" in data_raw else None
        ),
    )
    if data.node_batch <= 0 or data.edge_batch <= 0:
        raise ValueError("data.node_batch and data.edge_batch must be positive")
    if data.negative_ratio <= 0:
        raise ValueError("data.negative_ratio must be positive")

    optim_raw = _as_mapping(_require(raw, "optim", ""), "optim")
    optim_keys: tuple[str, ...] = ("lr", "weight_decay", "epochs", "warmup_steps", "grad_clip")
    _check_no_unknown_keys(optim_raw, optim_keys, "optim")
    optim = EgoOptimConfig(
        lr=_as_float(_require(optim_raw, "lr", "optim."), "optim.lr"),
        weight_decay=_as_float(_require(optim_raw, "weight_decay", "optim."), "optim.weight_decay"),
        epochs=_as_int(_require(optim_raw, "epochs", "optim."), "optim.epochs"),
        warmup_steps=_as_int(_require(optim_raw, "warmup_steps", "optim."), "optim.warmup_steps"),
        grad_clip=_as_float(_require(optim_raw, "grad_clip", "optim."), "optim.grad_clip"),
    )
    if optim.epochs <= 0:
        raise ValueError("optim.epochs must be positive")

    diagnostics_raw = _as_mapping(_require(raw, "diagnostics", ""), "diagnostics")
    diagnostic_keys = (
        "gradient_probe_interval",
        "gradient_imbalance_ratio",
        "gradient_imbalance_steps",
        "probe_s1_abs_mean_max",
        "selection_auprc_tolerance",
        "topk_fraction",
    )
    _check_no_unknown_keys(diagnostics_raw, diagnostic_keys, "diagnostics")
    diagnostics = EgoDiagnosticsConfig(
        gradient_probe_interval=_as_int(
            _require(diagnostics_raw, "gradient_probe_interval", "diagnostics."),
            "diagnostics.gradient_probe_interval",
        ),
        gradient_imbalance_ratio=_as_float(
            _require(diagnostics_raw, "gradient_imbalance_ratio", "diagnostics."),
            "diagnostics.gradient_imbalance_ratio",
        ),
        gradient_imbalance_steps=_as_int(
            _require(diagnostics_raw, "gradient_imbalance_steps", "diagnostics."),
            "diagnostics.gradient_imbalance_steps",
        ),
        probe_s1_abs_mean_max=_as_float(
            _require(diagnostics_raw, "probe_s1_abs_mean_max", "diagnostics."),
            "diagnostics.probe_s1_abs_mean_max",
        ),
        selection_auprc_tolerance=_as_float(
            _require(diagnostics_raw, "selection_auprc_tolerance", "diagnostics."),
            "diagnostics.selection_auprc_tolerance",
        ),
        topk_fraction=_as_float(
            _require(diagnostics_raw, "topk_fraction", "diagnostics."),
            "diagnostics.topk_fraction",
        ),
    )
    if diagnostics.gradient_probe_interval <= 0 or diagnostics.gradient_imbalance_steps <= 0:
        raise ValueError("diagnostic gradient intervals must be positive")
    if diagnostics.gradient_imbalance_ratio <= 1.0:
        raise ValueError("diagnostics.gradient_imbalance_ratio must exceed 1")
    if diagnostics.probe_s1_abs_mean_max <= 0:
        raise ValueError("diagnostics.probe_s1_abs_mean_max must be positive")
    if not math.isfinite(diagnostics.selection_auprc_tolerance) or not (
        0.0 <= diagnostics.selection_auprc_tolerance <= 1.0
    ):
        raise ValueError("diagnostics.selection_auprc_tolerance must be finite and in [0, 1]")
    if not 0.0 < diagnostics.topk_fraction <= 1.0:
        raise ValueError("diagnostics.topk_fraction must be in (0, 1]")

    eval_raw = _as_mapping(_require(raw, "eval", ""), "eval")
    _check_no_unknown_keys(eval_raw, ("patience", "eval_every"), "eval")
    eval_cfg = EvalConfig(
        patience=_as_int(_require(eval_raw, "patience", "eval."), "eval.patience"),
        eval_every=_as_int(_require(eval_raw, "eval_every", "eval."), "eval.eval_every"),
    )

    mixed_precision = _as_str(_require(raw, "mixed_precision", ""), "mixed_precision")
    if mixed_precision not in ("no", "bf16"):
        raise ValueError(f"mixed_precision must be 'no' or 'bf16', got {mixed_precision!r}")

    runtime: EgoRuntimeConfig | None = None
    if raw.get("runtime") is not None:
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
            "train_eval_budget_seconds",
            "artifact_budget_seconds",
        )
        _check_no_unknown_keys(runtime_raw, runtime_keys, "runtime")
        world_size_raw = _require(runtime_raw, "world_size", "runtime.")
        if world_size_raw != "auto":
            raise ValueError("runtime.world_size must be 'auto' for EgoStitch Stage-1")
        runtime = EgoRuntimeConfig(
            world_size=0,
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
                _require(runtime_raw, "prefetch_factor", "runtime."), "runtime.prefetch_factor"
            ),
            token_budget=_as_int(
                _require(runtime_raw, "token_budget", "runtime."), "runtime.token_budget"
            ),
            max_pairs_per_rank=_as_int(
                _require(runtime_raw, "max_pairs_per_rank", "runtime."),
                "runtime.max_pairs_per_rank",
            ),
            memory_limit_gib=_as_float(
                _require(runtime_raw, "memory_limit_gib", "runtime."), "runtime.memory_limit_gib"
            ),
            total_budget_seconds=_as_int(
                _require(runtime_raw, "total_budget_seconds", "runtime."),
                "runtime.total_budget_seconds",
            ),
            pack_budget_seconds=_as_int(
                _require(runtime_raw, "pack_budget_seconds", "runtime."),
                "runtime.pack_budget_seconds",
            ),
            train_eval_budget_seconds=_as_int(
                _require(runtime_raw, "train_eval_budget_seconds", "runtime."),
                "runtime.train_eval_budget_seconds",
            ),
            artifact_budget_seconds=_as_int(
                _require(runtime_raw, "artifact_budget_seconds", "runtime."),
                "runtime.artifact_budget_seconds",
            ),
        )
        if runtime.token_budget <= 0:
            raise ValueError(
                "runtime.token_budget must be a positive node-batch (B_n) size for this family"
            )
        stage_total = (
            runtime.pack_budget_seconds
            + runtime.train_eval_budget_seconds
            + runtime.artifact_budget_seconds
        )
        if stage_total != runtime.total_budget_seconds:
            raise ValueError(
                f"runtime stage budgets must sum to {runtime.total_budget_seconds}; "
                f"got {stage_total}"
            )

    training: EgoStitchTrainingConfig | None = None
    if raw.get("training") is not None:
        if family != _EGOSTITCH_E2E_FAMILY:
            raise ValueError("training is only valid for model.family='egostitch_e2e'")
        training_raw = _as_mapping(raw["training"], "training")
        training_keys = tuple(EgoStitchTrainingConfig.__dataclass_fields__)
        _check_no_unknown_keys(training_raw, training_keys, "training")
        betas_raw = training_raw.get("betas", [0.9, 0.999])
        if not isinstance(betas_raw, list) or len(betas_raw) != 2:
            raise ValueError("training.betas must be a two-element list")
        training = EgoStitchTrainingConfig(
            positive_weight=_as_float(
                training_raw.get("positive_weight", 5.0), "training.positive_weight"
            ),
            phase_a_fraction=_as_float(
                training_raw.get("phase_a_fraction", 0.2), "training.phase_a_fraction"
            ),
            phase_b_fraction=_as_float(
                training_raw.get("phase_b_fraction", 0.1), "training.phase_b_fraction"
            ),
            lr_peak=_as_float(training_raw.get("lr_peak", 1e-4), "training.lr_peak"),
            min_lr=_as_float(training_raw.get("min_lr", 1e-5), "training.min_lr"),
            warmup_steps=_as_int(training_raw.get("warmup_steps", 500), "training.warmup_steps"),
            betas=(
                _as_float(betas_raw[0], "training.betas[0]"),
                _as_float(betas_raw[1], "training.betas[1]"),
            ),
            eps=_as_float(training_raw.get("eps", 1e-8), "training.eps"),
            clip_norm=_as_float(training_raw.get("clip_norm", 1.0), "training.clip_norm"),
            pair_encoder_clip_norm=_as_float(
                training_raw.get("pair_encoder_clip_norm", 3.0),
                "training.pair_encoder_clip_norm",
            ),
            generator_clip_norm=_as_float(
                training_raw.get("generator_clip_norm", 3.0),
                "training.generator_clip_norm",
            ),
            clip_immediate_abort=_as_float(
                training_raw.get("clip_immediate_abort", 1e-3), "training.clip_immediate_abort"
            ),
            clip_persistent_threshold=_as_float(
                training_raw.get("clip_persistent_threshold", 0.1),
                "training.clip_persistent_threshold",
            ),
            clip_persistent_steps=_as_int(
                training_raw.get("clip_persistent_steps", 10), "training.clip_persistent_steps"
            ),
            family_ratio_abort=_as_float(
                training_raw.get("family_ratio_abort", 50.0), "training.family_ratio_abort"
            ),
            family_ratio_probes=_as_int(
                training_raw.get("family_ratio_probes", 4), "training.family_ratio_probes"
            ),
            collapse_fraction=_as_float(
                training_raw.get("collapse_fraction", 0.25), "training.collapse_fraction"
            ),
            collapse_floor=_as_float(
                training_raw.get("collapse_floor", 1e-4), "training.collapse_floor"
            ),
            collapse_validations=_as_int(
                training_raw.get("collapse_validations", 2), "training.collapse_validations"
            ),
            selection_auprc_tolerance=_as_float(
                training_raw.get("selection_auprc_tolerance", 0.02),
                "training.selection_auprc_tolerance",
            ),
            selection_mmd_tolerance=_as_float(
                training_raw.get("selection_mmd_tolerance", 1e-6),
                "training.selection_mmd_tolerance",
            ),
            residual_ratio_min=_as_float(
                training_raw.get("residual_ratio_min", 1e-3), "training.residual_ratio_min"
            ),
        )
        if not math.isfinite(training.selection_auprc_tolerance) or not (
            0.0 <= training.selection_auprc_tolerance <= 1.0
        ):
            raise ValueError("training.selection_auprc_tolerance must be finite and in [0, 1]")
        if diagnostics.selection_auprc_tolerance != training.selection_auprc_tolerance:
            raise ValueError(
                "diagnostics.selection_auprc_tolerance must exactly equal "
                "training.selection_auprc_tolerance"
            )
        pinned_training = EgoStitchTrainingConfig()
        if replace(
            training,
            selection_auprc_tolerance=pinned_training.selection_auprc_tolerance,
        ) != pinned_training:
            raise ValueError(
                f"training values must exactly match the pinned defaults; got {training!r}"
            )
        if data.negative_ratio != 5:
            raise ValueError("training requires data.negative_ratio=5")
        # `optim.epochs` belongs to the experiment plan while the three-phase
        # curriculum scales with `schedule_total_steps`.
        if (
            optim.lr != 1e-4
            or optim.weight_decay != 0.01
            or optim.warmup_steps != 500
            or optim.grad_clip != 1.0
        ):
            raise ValueError("training requires the pinned optimizer settings")
        resolved_e2e = E2EConfig.from_mapping(model_kwargs)
        if resolved_e2e.p_topo not in (0.15, 0.0):
            raise ValueError("training requires p_topo=0.15, or 0.0 for the p0 arm")
        if mixed_precision != "bf16" or eval_cfg.eval_every != 1:
            raise ValueError("training requires mixed_precision=bf16 and eval.eval_every=1")
        if (
            diagnostics.gradient_probe_interval != 50
            or diagnostics.gradient_imbalance_ratio != 50.0
            or diagnostics.gradient_imbalance_steps != 200
        ):
            raise ValueError("training diagnostics do not match the pinned guard cadence")

    if family == _EGOSTITCH_E2E_FAMILY and training is None:
        raise ValueError(
            "legacy egostitch_e2e v1 execution is not available on this branch; "
            "use a training config or checkout archive/egostitch-e2e-v1"
        )

    return EgoConfig(
        model=model,
        data=data,
        optim=optim,
        diagnostics=diagnostics,
        eval=eval_cfg,
        seed=_as_int(_require(raw, "seed", ""), "seed"),
        output_dir=Path(_as_str(_require(raw, "output_dir", ""), "output_dir")),
        mixed_precision=mixed_precision,
        runtime=runtime,
        training=training,
        run_kind=None,
    )


def config_to_dict(cfg: EgoConfig) -> dict[str, object]:
    """Return a JSON-serializable dict of the full config (checkpoint payload)."""
    payload = asdict(cfg)
    payload.pop("run_kind", None)  # execution context is not scientific config

    def _stringify(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _stringify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_stringify(v) for v in value]
        return value

    return cast(dict[str, object], _stringify(payload))


# --------------------------------------------------------- E2E training primitives


E2EPhaseName = Literal["A", "B", "C"]
E2EArmName = Literal[
    "full",
    "b0_e2e_f_only",
    "p0",
    "no_l_rel",
    "row_layernorm",
]


@dataclass(frozen=True)
class E2EPhaseState:
    """Zero-based optimizer-step state for the rev-3.1 §14.4.3 curriculum."""

    phase: E2EPhaseName
    alpha: float
    edge_active: bool
    real_ssl_scale: float


def e2e_phase_boundaries(total_steps: int) -> tuple[int, int]:
    """Return the exclusive Phase-A and Phase-B end steps."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    phase_a_end = math.ceil(0.2 * total_steps)
    return phase_a_end, phase_a_end + math.ceil(0.1 * total_steps)


def e2e_phase_state(step: int, total_steps: int) -> E2EPhaseState:
    """Resolve the exact A/B/C behavior for one zero-based optimizer step."""
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    phase_a_end, phase_b_end = e2e_phase_boundaries(total_steps)
    if step < phase_a_end:
        return E2EPhaseState("A", 0.0, False, 0.0)
    if step < phase_b_end:
        ramp_steps = phase_b_end - phase_a_end
        alpha = min(1.0, max(0.0, (step - phase_a_end + 1) / ramp_steps))
        return E2EPhaseState("B", alpha, True, alpha)
    return E2EPhaseState("C", 1.0, True, 1.0)


def _e2e_should_capture_eligibility_reference(
    phase: E2EPhaseState,
    *,
    warm_reference_auprc: float | None,
) -> bool:
    """Take the AUPRC reference at the first validation after edge training starts."""
    return phase.edge_active and warm_reference_auprc is None


_E2E_RECON_COMPONENTS = (
    "feat",
    "exist",
    "mult",
    "deg",
    "slotadj",
    "gate",
    "ptr",
    "align",
    "div",
    "rel",
)
_E2E_FIDELITY_COMPONENTS = frozenset({"feat", "exist", "mult", "deg"})


def e2e_recon_component_factors(step: int, total_steps: int) -> dict[str, float]:
    """Return the §14.4.1 per-component reconstruction anneal factors."""
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    edge_start, _ = e2e_phase_boundaries(total_steps)
    edge_steps = total_steps - edge_start
    if step < edge_start or edge_steps <= 1:
        fidelity_factor = 1.0
    else:
        progress = (step - edge_start) / (edge_steps - 1)
        fidelity_factor = 1.0 - 0.75 * progress
    return {
        name: fidelity_factor if name in _E2E_FIDELITY_COMPONENTS else 1.0
        for name in _E2E_RECON_COMPONENTS
    }


def _e2e_dispersion_rows(
    pi: torch.Tensor,
    h: torch.Tensor,
    adj: torch.Tensor,
    plan: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-row rev-3.1 collapse telemetry in fp32."""
    if pi.ndim != 2 or h.ndim != 3 or adj.ndim != 3 or plan.ndim != 3:
        raise ValueError("dispersion inputs must have shapes (B,K), (B,K,D), (B,K,K), (B,K,K)")
    batch, slots = pi.shape
    if h.shape[:2] != (batch, slots) or adj.shape != (batch, slots, slots):
        raise ValueError("slot dispersion tensor shapes disagree")
    if plan.shape[1:] != (slots, slots):
        raise ValueError("plan slot dimensions disagree with node slots")
    with torch.autocast(device_type=pi.device.type, enabled=False):
        pi32 = pi.float()
        h32 = torch.nn.functional.normalize(h.float(), dim=-1)
        adj32 = adj.float()
        plan32 = plan.float()
        upper = torch.triu_indices(slots, slots, offset=1, device=pi.device)
        cosine = torch.bmm(h32, h32.transpose(1, 2))[:, upper[0], upper[1]].mean(dim=-1)
        adj_std = adj32[:, upper[0], upper[1]].std(dim=-1)
        mass = plan32.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        row_mass = plan32.sum(dim=-1)
        column_mass = plan32.sum(dim=-2)
        rank1 = row_mass[:, :, None] * column_mass[:, None, :] / mass
        residual = torch.linalg.vector_norm(plan32 - rank1, dim=(1, 2)) / torch.linalg.vector_norm(
            plan32, dim=(1, 2)
        ).clamp_min(1e-30)
        row_probability = plan32 / row_mass[:, :, None].clamp_min(1e-30)
        row_entropy = -(
            row_probability * row_probability.clamp_min(1e-30).log()
        ).sum(dim=-1).mean(dim=-1) / math.log(slots)
    return {
        "pi_slot_std": pi32.std(dim=-1),
        "h_pairwise_cosine_mean": cosine,
        "adj_offdiag_std": adj_std,
        "plan_row_entropy": row_entropy,
        "plan_rank1_marginal_residual": residual,
    }


def _e2e_scale_rows(h: torch.Tensor, plan: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-row OT-scale diagnostics (training log only, never an artifact).

    The Sinkhorn stage is numerically dead when squared slot distances greatly
    exceed `eps` and diffuse-rank-1 when they are far below it (design record
    2026-07-27 R3). These four numbers say which regime a run is in.

    Args:
        h: Shape ``(B, K, D)`` slot embeddings.
        plan: Shape ``(B, K, K)`` transport plan.

    Returns:
        Per-row ``plan_total_mass``, ``plan_max_cell_fraction``, ``h_norm_mean``,
        ``h_pairwise_sqdist_mean``.

    Raises:
        ValueError: If ``h``/``plan`` are not rank 3, or ``plan``'s batch and
            slot dimensions disagree with ``h``'s.
    """
    if h.ndim != 3 or plan.ndim != 3:
        raise ValueError("scale inputs must have shapes (B,K,D) and (B,K,K)")
    batch, slots = h.shape[:2]
    if plan.shape != (batch, slots, slots):
        raise ValueError("plan shape disagrees with slot embedding shape")
    with torch.autocast(device_type=h.device.type, enabled=False):
        h32 = h.float()
        plan32 = plan.float()
        mass = plan32.sum(dim=(1, 2))
        max_cell = plan32.amax(dim=(1, 2)) / mass.clamp_min(1e-30)
        square = h32.square().sum(dim=-1)
        distance = (
            square[:, :, None] + square[:, None, :] - 2.0 * torch.bmm(h32, h32.transpose(1, 2))
        ).clamp_min(0.0)
        upper = torch.triu_indices(slots, slots, offset=1, device=h.device)
        return {
            "plan_total_mass": mass,
            "plan_max_cell_fraction": max_cell,
            "h_norm_mean": torch.linalg.vector_norm(h32, dim=-1).mean(dim=-1),
            "h_pairwise_sqdist_mean": distance[:, upper[0], upper[1]].mean(dim=-1),
        }


def e2e_dispersion_statistics(
    pi: torch.Tensor,
    h: torch.Tensor,
    adj: torch.Tensor,
    plan: torch.Tensor,
) -> dict[str, float]:
    """Aggregate rev-3.1 validation dispersion telemetry."""
    rows = _e2e_dispersion_rows(pi, h, adj, plan)
    return {name: float(value.mean()) for name, value in rows.items()}


@dataclass
class E2ESlotCollapseGuard:
    """Two-validation §14.4.8 slot-collapse death guard."""

    streak: int = 0

    def update(
        self,
        telemetry: Mapping[str, float],
        *,
        conditioning_active: bool,
        enforce_quality: bool = True,
    ) -> bool:
        """Track collapse and optionally enforce the finite quality threshold."""
        if not conditioning_active:
            self.streak = 0
            return False
        values = (
            telemetry["h_pairwise_cosine_mean"],
            telemetry["plan_rank1_marginal_residual"],
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("non-finite E2E slot-collapse telemetry")
        collapsed = (
            telemetry["h_pairwise_cosine_mean"] > 0.95
            or telemetry["plan_rank1_marginal_residual"] < 0.05
        )
        self.streak = self.streak + 1 if collapsed else 0
        tripped = self.streak >= 2
        if tripped and enforce_quality:
            raise RuntimeError("training_invalid(slot_collapse)")
        return tripped


def e2e_degree_decorrelation_telemetry(
    endpoint_degree: NDArray[np.float64],
    topology_delta: NDArray[np.float64],
) -> dict[str, float]:
    """Correlation watchdog for the full-minus-f_logit validation residual."""
    if endpoint_degree.shape != topology_delta.shape:
        raise ValueError("endpoint degree and topology delta shapes must match")
    if endpoint_degree.size < 2 or np.std(endpoint_degree) == 0 or np.std(topology_delta) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(endpoint_degree, topology_delta)[0, 1])
    return {"topology_delta_degree_correlation": correlation}


def _e2e_validation_endpoint_degrees(data: EgoStitchData) -> NDArray[np.float64]:
    """Return degree sums from the validation role's own structural graph."""
    if data.validation_nodes:
        graph = nx.Graph()
        graph.add_nodes_from(data.validation_nodes)
        graph.add_edges_from(data.validation_positive_edges)
    else:
        graph = data.target_builder.graph
    return np.asarray(
        [
            float(graph.degree(u) if u in graph else 0)
            + float(graph.degree(v) if v in graph else 0)
            for u, v in data.val_pairs
        ],
        dtype=np.float64,
    )


def e2e_degree_prior_init(
    model: EgoStitchStage1 | EgoStitchModel, data: EgoStitchData
) -> float:
    """Center the lognormal degree head on the ``G_fit`` degree prior.

    ``deg_mu`` is a raw linear output (`tokenize.py:71-75`) born near 0, while
    ``mean(log d)`` on ``G_fit`` sits several nats above it. `degree_nll`'s
    ``1/sigma**2`` factor turns that standing residual into a generator gradient
    above the Sec 13.19 clip threshold on *every* step from step 1 -- the
    2026-07-28 `persistent clipping` abort, whose streak needs a term that is
    live on all ten steps. Setting the output bias removes the residual at
    initialization. ``log sigma`` is deliberately left alone: matching it
    instead of ``mu`` measures worse, because it shrinks the denominator while
    the numerator is what is wrong.

    ``sorted`` is load-bearing, not cosmetic: ``build_g_fit`` adds nodes from a
    ``frozenset[str]``, whose iteration order depends on ``PYTHONHASHSEED``
    (pinned nowhere in this repo), and ``np.log(...).mean()`` uses pairwise
    summation, so an unsorted traversal makes the last ulp order-dependent. Each
    rank computes this independently, so without the sort the replicas can
    disagree in the final bit from step 0.
    """
    generator = model.generator.stage1 if isinstance(model, EgoStitchModel) else model
    graph = data.target_builder.graph
    degrees = np.asarray(
        [max(int(graph.degree(node)), 1) for node in sorted(graph.nodes())],
        dtype=np.float64,
    )
    if degrees.size == 0:
        raise RuntimeError("G_fit carries no nodes for the degree prior")
    mu0 = float(np.log(degrees).mean())
    if not math.isfinite(mu0):
        raise RuntimeError(f"degree prior mean(log d) is not finite: {mu0}")
    # `nn.Sequential.__getitem__` is typed `-> Module`, and `Module.__getattr__`
    # widens `.bias` to `Tensor | Module`; the isinstance narrowing below is what
    # makes `head.bias` a `Tensor` for the type checker. It is also a real
    # assertion: `build_mlp` (`layers.py:14-32`) ends every head in a bias-
    # carrying `nn.Linear`, and silently skipping the write if that ever stopped
    # being true would reintroduce the standing residual this function removes.
    head = generator.tokenize.degree_dist_head[-1]
    if not isinstance(head, torch.nn.Linear):
        raise RuntimeError(f"degree head does not end in nn.Linear: {type(head).__name__}")
    bias: torch.Tensor | None = head.bias
    if bias is None:
        raise RuntimeError("degree head's output layer carries no bias to center")
    with torch.no_grad():
        bias[0] = mu0
    return mu0


def e2e_first_eligible_epoch(total_steps: int, steps_per_epoch: int) -> int:
    """First 1-based epoch ending after one complete Phase-C epoch."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    _, phase_b_end = e2e_phase_boundaries(total_steps)
    return math.ceil((phase_b_end + steps_per_epoch) / steps_per_epoch)


def e2e_weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    real_row_mask: torch.Tensor,
    *,
    world_size: int = 1,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
    positive_weight: float = 5.0,
) -> torch.Tensor:
    """Exact padding-aware global weighted BCE under DDP gradient averaging.

    ``all_reduce_sum`` is injectable so equivalence tests need no process
    group.  It receives the detached local effective-weight denominator.
    """
    if logits.shape != labels.shape or logits.shape != real_row_mask.shape:
        raise ValueError("logits, labels, and real_row_mask must have identical shapes")
    if world_size <= 0 or positive_weight <= 0:
        raise ValueError("world_size and positive_weight must be positive")
    logits_fp32 = logits.float()
    labels_fp32 = labels.float()
    mask_fp32 = real_row_mask.float()
    weights = 5.0 * labels_fp32 + (1.0 - labels_fp32)
    if positive_weight != 5.0:
        weights = positive_weight * labels_fp32 + (1.0 - labels_fp32)
    local_denominator = (mask_fp32 * weights).sum().detach()
    if all_reduce_sum is not None:
        denominator = all_reduce_sum(local_denominator)
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        denominator = local_denominator.clone()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
    else:
        denominator = local_denominator
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise RuntimeError("global weighted-BCE denominator must be finite and positive")
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_fp32, labels_fp32, reduction="none"
    )
    return world_size * (mask_fp32 * weights * per_row).sum() / denominator


@dataclass(frozen=True)
class E2EParameterGroups:
    """Disjoint/exhaustive neural optimizer groups and their stable manifest."""

    groups: dict[str, tuple[torch.nn.Parameter, ...]]
    names: dict[str, tuple[str, ...]]
    sha256: dict[str, str]


def build_e2e_parameter_groups(model: EgoStitchModel) -> E2EParameterGroups:
    """Build one optimizer group per top-level component (design 2026-08-02 §7).

    Every parameter's group is simply its qualified name's first path segment
    -- ``generator``, ``encoder``, or ``classifier`` -- since those are
    exactly `EgoStitchModel`'s three submodule names. This replaces the old
    name-prefix scheme (a bespoke ``conditioning_prefixes`` allowlist for the
    STE encoder, the relational head, and the trunk's gated cross-attention)
    with the component boundary itself: ``rel_head`` now lives inside
    ``encoder`` and needs no special case, and ``trunk`` (including its
    ``topo_xattn`` gates) now lives inside ``classifier``.
    """
    live_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "generator": [],
        "encoder": [],
        "classifier": [],
    }
    for name, parameter in model.named_parameters():
        if id(parameter) not in live_ids:
            continue
        component = name.split(".", 1)[0]
        if component not in grouped:
            raise RuntimeError(
                f"E2E parameter {name!r} does not belong to a known component group"
            )
        grouped[component].append((name, parameter))

    all_ids = [id(parameter) for rows in grouped.values() for _, parameter in rows]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != live_ids:
        raise RuntimeError("E2E optimizer groups must be disjoint and exhaustive")
    names = {group: tuple(sorted(name for name, _ in rows)) for group, rows in grouped.items()}
    parameters = {
        group: tuple(parameter for _, parameter in sorted(rows, key=lambda row: row[0]))
        for group, rows in grouped.items()
    }
    if any(not rows for rows in parameters.values()):
        raise RuntimeError("every E2E optimizer group must contain trainable parameters")
    hashes = {
        group: hashlib.sha256(("\n".join(group_names) + "\n").encode()).hexdigest()
        for group, group_names in names.items()
    }
    return E2EParameterGroups(parameters, names, hashes)


@dataclass(frozen=True)
class E2EGradientGroupRecord:
    """One pre-clip optimizer-group guard measurement."""

    active: bool
    norm: float | None
    clip_coefficient: float | None
    nonfinite_elements: int


def e2e_check_and_clip_gradients(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    active_groups: set[str],
    *,
    max_norm: float | Mapping[str, float] = 1.0,
    enforce_nonzero: bool = True,
) -> dict[str, E2EGradientGroupRecord]:
    """Fail closed on group gradients and independently clip active groups."""
    if isinstance(max_norm, Mapping):
        if set(max_norm) != set(groups) or any(value <= 0 for value in max_norm.values()):
            raise ValueError("max_norm mapping must cover every group with positive values")
        max_norms = dict(max_norm)
    else:
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        max_norms = dict.fromkeys(groups, max_norm)
    unknown = active_groups - set(groups)
    if unknown:
        raise ValueError(f"unknown active optimizer groups: {sorted(unknown)}")
    records: dict[str, E2EGradientGroupRecord] = {}
    for name, parameters in groups.items():
        all_grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        all_nonfinite = sum(int((~torch.isfinite(grad)).sum().item()) for grad in all_grads)
        if name not in active_groups:
            if all_nonfinite:
                raise RuntimeError(f"non-finite gradient in inactive E2E group {name!r}")
            records[name] = E2EGradientGroupRecord(False, None, None, all_nonfinite)
            continue
        grads = all_grads
        nonfinite = all_nonfinite
        squared_terms = [grad.detach().double().square().sum() for grad in grads]
        squared = (
            torch.stack(squared_terms).sum()
            if squared_terms
            else torch.zeros((), dtype=torch.float64)
        )
        norm = float(torch.sqrt(squared).item())
        if nonfinite or not math.isfinite(norm):
            raise RuntimeError(f"non-finite gradient in active E2E group {name!r}")
        if norm == 0.0 and enforce_nonzero:
            raise RuntimeError(f"zero gradient norm in active E2E group {name!r}")
        coefficient = min(1.0, max_norms[name] / (norm + 1e-12))
        for grad in grads:
            grad.mul_(coefficient)
        records[name] = E2EGradientGroupRecord(True, norm, coefficient, nonfinite)
    return records


def e2e_assert_replicated_squared_norms(
    gathered_by_group: Mapping[str, torch.Tensor], *, rtol: float = 1e-7, atol: float = 1e-12
) -> None:
    """Assert each group's fp64 squared norm is identical across DDP ranks."""
    for name, gathered in gathered_by_group.items():
        values = gathered.detach().double().reshape(-1)
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"invalid gathered squared norms for E2E group {name!r}")
        reference = values[0].expand_as(values)
        if not bool(torch.allclose(values, reference, rtol=rtol, atol=atol)):
            raise RuntimeError(f"DDP gradient norms differ across ranks for E2E group {name!r}")


@dataclass
class E2EClipGuard:
    """Stateful §13.19 immediate and persistent clip-coefficient guard."""

    immediate_threshold: float = 1e-3
    persistent_threshold: float = 0.1
    persistent_steps: int = 10
    streaks: dict[str, int] | None = None
    recent: dict[str, list[dict[str, object]]] | None = None

    def update(
        self,
        records: Mapping[str, E2EGradientGroupRecord],
        *,
        step: int | None = None,
        phase: E2EPhaseName | None = None,
        enforce_immediate: bool = True,
        enforce_persistent: bool = True,
    ) -> None:
        """Advance one optimizer step and raise immediately on a violation."""
        if self.streaks is None:
            self.streaks = {}
        if self.recent is None:
            self.recent = {}
        for name, record in records.items():
            if not record.active:
                self.streaks[name] = 0
                self.recent[name] = []
                continue
            coefficient = record.clip_coefficient
            if coefficient is None or not math.isfinite(coefficient):
                raise RuntimeError(f"invalid clip coefficient for active E2E group {name!r}")
            trail = self.recent.setdefault(name, [])
            trail.append(
                {
                    "step": step,
                    "phase": phase,
                    "norm": record.norm,
                    "coefficient": coefficient,
                    "nonfinite_elements": record.nonfinite_elements,
                }
            )
            del trail[: -self.persistent_steps]
            context = f"step={step} phase={phase} norm={record.norm} coefficient={coefficient}"
            recent = json.dumps(trail, sort_keys=True)
            self.streaks[name] = (
                self.streaks.get(name, 0) + 1 if coefficient < self.persistent_threshold else 0
            )
            if coefficient < self.immediate_threshold and enforce_immediate:
                raise RuntimeError(
                    f"extreme clipping in active E2E group {name!r}; "
                    f"{context} streak={self.streaks[name]} recent={recent}"
                )
            if enforce_persistent and self.streaks[name] >= self.persistent_steps:
                raise RuntimeError(
                    f"persistent clipping in active E2E group {name!r}; "
                    f"{context} streak={self.streaks[name]} recent={recent}"
                )


def e2e_assert_finite_optimizer_state(
    groups: Mapping[str, Sequence[torch.nn.Parameter]], optimizer: torch.optim.Optimizer
) -> None:
    """Reject any non-finite post-step parameter or optimizer-state tensor."""
    for name, parameters in groups.items():
        for parameter in parameters:
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(f"non-finite parameter in E2E group {name!r}")
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"non-finite optimizer state in E2E group {name!r}")


@dataclass(frozen=True)
class E2ECheckpointRecord:
    """Train-side evidence used by the fail-closed E2E selector."""

    epoch: int
    phase: E2EPhaseName
    full_joint_epochs_completed: int
    guards_passed: bool
    auprc: float
    prevalence: float
    active_logit_std: float
    clustering_mmd: float
    brier: float
    warm_reference_std: float | None = None
    warm_reference_auprc: float | None = None
    residual_ratio: float | None = None
    topology_gradient_norm: float | None = None


def select_e2e_checkpoint(
    records: Sequence[E2ECheckpointRecord],
    arm: E2EArmName,
    *,
    auprc_tolerance: float = 0.02,
    mmd_tolerance: float = 1e-6,
) -> E2ECheckpointRecord | None:
    """Select the best-ranked checkpoint: AUPRC, then clustering MMD, then Brier.

    There is no eligibility predicate. Whether a selected checkpoint is
    scientifically usable is an owner-side judgement made from the raw
    per-epoch metrics in ``metrics.jsonl``, not something this function or any
    other code path decides. ``arm`` is accepted only to keep one call
    signature across arms.
    """
    del arm
    if not records:
        return None
    best_auprc = max(record.auprc for record in records)
    candidates = [record for record in records if record.auprc >= best_auprc - auprc_tolerance]
    best_mmd = min(record.clustering_mmd for record in candidates)
    candidates = [
        record for record in candidates if record.clustering_mmd <= best_mmd + mmd_tolerance
    ]
    return min(candidates, key=lambda record: (record.brier, record.epoch))


_PROBE_DISPATCH_MODES = ("probe", "epoch-probe")
_TRAIN_EGOSTITCH_DDP_MODES = ("train", *_PROBE_DISPATCH_MODES)


def parse_args(argv: Sequence[str] | None = None) -> EgoCliArgs:
    """Parse the worker CLI (the shared ``src.train_b0`` contract).

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The parsed `EgoCliArgs`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.train_egostitch",
        description="Train the EgoStitch Stage-1 model (spec Sec 13).",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="DEBUG ONLY: stop after N optimizer steps and publish only debug artifacts.",
    )
    parser.add_argument(
        "--ddp-mode",
        choices=_TRAIN_EGOSTITCH_DDP_MODES,
        default=None,
        help="internal multi-H20 worker mode (launched by accelerate launch).",
    )
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument(
        "--token-budget-per-rank",
        type=int,
        default=None,
        help="per-rank node-stream batch size B_n for this family (spec Sec 13.13).",
    )
    parser.add_argument("--profile-output", type=Path, default=None)
    # `debug` is deliberately absent: it is derived from `--max-steps`, never
    # selected. Offering it here would let a caller claim the digest-pin
    # exemption `_bind_feature_standardization` grants debug runs without
    # actually running a bounded, non-publishing schedule.
    parser.add_argument(
        "--run-kind",
        choices=("formal",),
        default=None,
        help="E2E execution context; defaults to formal and does not alter the config hash",
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
    return EgoCliArgs(
        config=namespace.config,
        seed=namespace.seed,
        output_dir=namespace.output_dir,
        max_steps=namespace.max_steps,
        ddp_mode=namespace.ddp_mode,
        pack_dir=namespace.pack_dir,
        token_budget_per_rank=namespace.token_budget_per_rank,
        profile_output=namespace.profile_output,
        run_kind=namespace.run_kind,
    )


def apply_overrides(cfg: EgoConfig, args: EgoCliArgs) -> EgoConfig:
    """Apply the ``--seed`` / ``--output-dir`` / ``--run-kind`` overrides.

    Raises:
        ValueError: When an override requires a missing training section.
    """
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.run_kind is not None:
        if cfg.training is None:
            raise ValueError("--run-kind requires a training config section")
        cfg = replace(cfg, run_kind=args.run_kind)
    return cfg


def prepare_ddp_run_config(cfg: EgoConfig, *, max_steps: int | None) -> tuple[EgoConfig, bool]:
    """Enforce the formal/debug output-directory boundary before DDP work starts.

    A bounded run is never allowed to use the configured formal output directory.
    Its checkpoints may support local smoke checks, but its metadata explicitly
    marks them non-formal so the gate cannot publish held-out results from them.
    """
    if cfg.training is not None:
        run_kind = cfg.run_kind or "formal"
        if max_steps is not None:
            raise ValueError(
                f"E2E {run_kind} runs must execute the complete schedule; --max-steps forbidden"
            )
    if max_steps is None:
        return cfg, False
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    debug_dir = (
        cfg.output_dir
        if cfg.output_dir.name.endswith("_debug")
        else cfg.output_dir.with_name(f"{cfg.output_dir.name}_debug")
    )
    return replace(cfg, output_dir=debug_dir), True


# --------------------------------------------------------------------------- pack stage


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# The held-out artifacts no training-side stage may read (design 2026-07-29
# Sec 3.1). Named once so every stage -- the two-stage e2e assembly, its
# `cfg.training is None` sibling, and the pack builder that precedes both --
# shares one definition rather than one carrying the boundary and the rest
# carrying nothing.
_HELD_OUT_FILENAMES = ("candidate_test_edges.txt", "test_edges.txt", "test_graph.pkl")

# The complete set of benchmark inputs each stage opens, enumerated so the
# boundary can be applied *before* the first open rather than after. Keep these
# in step with the reads themselves: `_assemble_e2e_data` / `prepare_pack`'s
# training branch open the train-side three; the `cfg.training is None` sibling
# assembles through `load_benchmark` + `verify_benchmark`
# (`src/data/artifacts.py:205-311, 350-384`), which open the rest.
_E2E_TRAIN_SIDE_INPUTS = ("split.pkl", "train_edges.txt", "val_edges.txt")
_BENCHMARK_PACKAGE_ROOT_INPUTS = ("graph.pkl", "positive_edges.txt")
_BENCHMARK_PACKAGE_STRATEGY_INPUTS = (
    "split.pkl",
    "train_edges.txt",
    "val_edges.txt",
    "test_edges.txt",
    "candidate_test_edges.txt",
    "train_graph.pkl",
    "test_graph.pkl",
    "test_node_buckets.pkl",
)


def _stat_or_none(path: Path) -> os.stat_result | None:
    """Return ``path.stat()`` (symlinks followed), or ``None`` if it is unreadable."""
    try:
        return path.stat()
    except OSError:
        return None


def _assert_no_held_out_access(strategy_dir: Path, opened: Iterable[Path]) -> None:
    """Raise when a training-side stage would read a held-out artifact.

    The check is path-scoped and run-kind independent: it is *opening* a
    held-out path that is forbidden, in every run kind, which is strictly
    stronger than the pre-cleanup form (which raised on the mere presence of
    those files and exempted `run_kind == "formal"`, so it made every other
    kind impossible in the repository data root while never once guarding the
    run that matters).

    Aliasing is checked two ways, because name resolution alone does not cover
    it. `Path.resolve` collapses symlinks and relative aliases such as
    ``<strategy>/../<strategy>/test_edges.txt``, and it is applied to the
    forbidden set too so a symlinked `strategy_dir` cannot put the two sides in
    different namespaces. `(st_dev, st_ino)` then catches the case resolution
    cannot see at all: a *hard* link, which is a second name for the same inode
    with no link to follow. Candidate paths that do not exist are dropped: a
    read of a missing file raises before any held-out row can reach the model,
    and keeping them would make the sibling branch unconditionally unusable
    rather than boundary-checked.

    Args:
        strategy_dir: The split-strategy directory the stage reads from.
        opened: Every path this stage reads.

    Raises:
        RuntimeError: When any opened path resolves or aliases onto a held-out
            artifact.
    """
    forbidden = {(strategy_dir / name).resolve() for name in _HELD_OUT_FILENAMES}
    forbidden_inodes: set[tuple[int, int]] = set()
    for path in forbidden:
        info = _stat_or_none(path)
        if info is not None:
            forbidden_inodes.add((info.st_dev, info.st_ino))
    trespass: set[str] = set()
    for path in opened:
        info = _stat_or_none(path)
        if info is None:
            continue
        if path.resolve() in forbidden or (info.st_dev, info.st_ino) in forbidden_inodes:
            trespass.add(str(path.resolve()))
    if trespass:
        raise RuntimeError("E2E training opened a held-out path: " + ", ".join(sorted(trespass)))


def _assert_input_boundary(cfg: EgoConfig) -> Path:
    """Reject every input path this config would open, before the first open.

    A guard that runs after the reads is not a guard: the pre-Wave-3 form fired
    only once `split.pkl`, `train_edges.txt` and `val_edges.txt` had been read
    and the F0, feature-statistics and grounding caches had been written, so a
    trespassing root still got its held-out bytes into memory and left derived
    caches on disk keyed to them. `prepare_pack` -- which opens the same paths
    one stage earlier -- had no boundary at all. This resolves the whole input
    set up front so both stages refuse before touching the filesystem.

    Args:
        cfg: The validated worker config.

    Returns:
        The strategy directory, so callers need not re-derive it.

    Raises:
        RuntimeError: When any path this config would open is held out.
    """
    strategy_dir = cfg.data.root / _BENCHMARK_SUBDIR / cfg.data.strategy
    opened = [strategy_dir / name for name in _E2E_TRAIN_SIDE_INPUTS]
    _assert_no_held_out_access(strategy_dir, opened)
    return strategy_dir


def required_pack_paths(cfg: EgoConfig, pack_dir: Path) -> tuple[Path, ...]:
    """Return every cache directory the generic pack stage must supervise."""
    if cfg.data.pack_dir is None:
        raise ValueError("data.pack_dir is required when model.family == 'egostitch_e2e'")
    return (pack_dir, cfg.data.pack_dir)


def prepare_pack(
    cfg: EgoConfig, pack_dir: Path, *, cold_cache: bool, temp_prefix: str = ""
) -> dict[str, object]:
    """Build (cold) or strictly validate (warm) this family's feature pack.

    The EgoStitch "pack" (spec Sec 13.13) is the F0 pooled matrix over every
    operative node plus the train-side grounding-pool cache, with a sha256
    manifest. The return payload matches the orchestrator's pack-evidence
    contract (``pack_manifest`` + ``pack_identity_sha256``).

    Args:
        cfg: The validated worker config.
        pack_dir: The pack directory.
        cold_cache: ``True`` builds from scratch; ``False`` validates.
        temp_prefix: Unused for this family (single-directory build); accepted
            for orchestrator-seam parity.

    Returns:
        ``{"pack_manifest": {...}, "pack_identity_sha256": <sha of manifest.json>}``.

    Raises:
        ValueError: On warm-cache validation drift.
        RuntimeError: When any benchmark input this pack would open is held out.
    """
    # [H] Path-scoped held-out boundary (design 2026-07-29 Sec 3.1), first
    # statement of the function: the pack stage runs *before* assembly and opens
    # the same `split.pkl`/`train_edges.txt` (or the whole benchmark package on
    # the `cfg.training is None` sibling), so a boundary that lived only in
    # assembly let the pack consume held-out rows and bake them into the F0,
    # grounding and feature-statistics caches the assembly then reuses.
    strategy_dir = _assert_input_boundary(cfg)
    # Call-time import preserves the pack builder/validator monkeypatch seam.
    from src.data import packed_features

    e2e_model_cfg = E2EConfig.from_mapping(cfg.model.config)
    n_ground = e2e_model_cfg.n_ground
    manifest_path = pack_dir / _PACK_MANIFEST_FILENAME
    # The pack carries data-universe identity rather than execution-stage
    # identity. Grounding caches keep their own `role_universe` internally.
    role: Literal["V_hold"] = _E2E_VALIDATION_ROLE
    f0_cold = not pack_dir.exists()
    raw_manifest: PackedFeatureManifest | None = None
    raw_pack_dir = cfg.data.pack_dir
    raw_cold = bool(raw_pack_dir is not None and not raw_pack_dir.exists())
    if raw_pack_dir is None:
        raise ValueError("data.pack_dir is required when model.family == 'egostitch_e2e'")
    if not cold_cache and raw_cold:
        raise ValueError(f"warm raw-token pack is missing: {raw_pack_dir}")
    if not cold_cache and f0_cold:
        raise ValueError(f"warm F0/grounding pack is missing: {pack_dir}")
    if f0_cold:
        pack_dir.mkdir(parents=True, exist_ok=True)
        store = FeatureStore(cfg.data.root / _FEATURES_SUBDIR)
        with (strategy_dir / "split.pkl").open("rb") as handle:
            split_payload = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
        if not isinstance(split_payload, dict) or "train" not in split_payload:
            raise ValueError("split.pkl must contain a train node collection")
        train_nodes_all = sorted(
            set(cast(Sequence[str], split_payload["train"]))
            - set(cfg.data.expected_missing_features)
        )
        train_pairs, train_labels = _read_labeled_pairs(strategy_dir / "train_edges.txt")
        positives = [
            pair
            for pair, label in zip(train_pairs, train_labels, strict=True)
            if int(label) == 1
        ]
        partition = derive_partition(
            positives, seed=cfg.data.partition_seed, msg_fraction=cfg.data.msg_fraction
        )
        holdout = derive_internal_holdout(train_nodes_all, partition.e_msg, partition.e_sup)
        validation_nodes = holdout.hold_manifest.nodes
        train_nodes = sorted(holdout.v_fit)
        operative = sorted(set(train_nodes) | set(validation_nodes))
        if raw_cold:
            assert raw_pack_dir is not None
            raw_manifest = packed_features.build_packed_features(
                cfg.data.root / _FEATURES_SUBDIR,
                raw_pack_dir,
                workers=cfg.runtime.pack_workers if cfg.runtime is not None else 1,
                temp_prefix=temp_prefix or None,
                f0_node_ids=operative,
                f0_cache_path=pack_dir / _PACK_F0_FILENAME,
            )
        matrix, index = build_f0_matrix(store, operative, cache_path=pack_dir / _PACK_F0_FILENAME)
        train_rows = np.asarray(
            matrix.numpy()[[index[node] for node in train_nodes]], dtype=np.float32
        )
        build_grounding_pool(
            train_rows,
            train_nodes,
            n_ground=n_ground,
            role_universe="V_fit",
            cache_path=pack_dir / _PACK_GROUNDING_FILENAME,
        )
        feature_stats_for_universe(
            np.asarray(matrix.numpy(), dtype=np.float32),
            index,
            train_nodes,
            cache_path=pack_dir / _PACK_FEATURE_STATS_FILENAME,
        )
        files = [
            _PACK_F0_FILENAME,
            _PACK_GROUNDING_FILENAME,
            _PACK_FEATURE_STATS_FILENAME,
        ]
        if validation_nodes:
            validation_rows = np.asarray(
                matrix.numpy()[[index[node] for node in validation_nodes]], dtype=np.float32
            )
            build_grounding_pool(
                validation_rows,
                validation_nodes,
                n_ground=n_ground,
                role_universe=role,
                cache_path=pack_dir / _PACK_VALIDATION_GROUNDING_FILENAME,
            )
            files.append(_PACK_VALIDATION_GROUNDING_FILENAME)
        manifest = {
            "family": cfg.model.family,
            "strategy": cfg.data.strategy,
            "validation_role": role,
            "n_operative_nodes": len(operative),
            "n_train_nodes": len(train_nodes),
            "n_validation_nodes": len(validation_nodes),
            "n_ground": n_ground,
            "files": {name: _sha256_file(pack_dir / name) for name in files},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_hashes = cast(dict[str, str], manifest["files"])
        # [P1-A] Packs built before rev-3.2 have no `feature_stats.npz` entry
        # in their manifest at all, so the generic per-file drift loop below
        # would simply never look at it -- leaving the preprocessing
        # statistics outside the recorded pack identity. Require the entry
        # explicitly instead of silently accepting (and later rebuilding) it.
        if _PACK_FEATURE_STATS_FILENAME not in file_hashes:
            raise ValueError(
                f"pack manifest at {manifest_path} has no "
                f"{_PACK_FEATURE_STATS_FILENAME} entry (a pre-rev-3.2 pack); "
                "rebuild the pack with cold_cache=True to register the "
                "feature-standardization statistics"
            )
        for name, expected in file_hashes.items():
            actual = _sha256_file(pack_dir / name)
            if actual != expected:
                raise ValueError(f"pack file {name} drifted: {actual} != {expected}")
        if manifest.get("strategy") != cfg.data.strategy:
            raise ValueError("pack manifest strategy does not match the config")
        if manifest.get("n_ground") != n_ground:
            raise ValueError("pack manifest n_ground does not match the model config")
        if manifest.get("validation_role") != role:
            raise ValueError("pack manifest validation role does not match the execution context")
    f0_identity = _sha256_file(manifest_path)
    packs: dict[str, object] = {
        "f0_grounding": {
            "path": str(pack_dir),
            "manifest": manifest,
            "identity_sha256": f0_identity,
            "cold": f0_cold,
        }
    }
    assert raw_pack_dir is not None
    source_root = cfg.data.root / _FEATURES_SUBDIR
    if raw_manifest is None:
        if raw_cold:
            raw_manifest = packed_features.build_packed_features(
                source_root,
                raw_pack_dir,
                workers=cfg.runtime.pack_workers if cfg.runtime is not None else 1,
                temp_prefix=temp_prefix or None,
            )
        else:
            raw_manifest = packed_features.validate_packed_manifest(raw_pack_dir, source_root)
    raw_identity = packed_features.sha256_file(raw_pack_dir / "manifest.json")
    assert raw_manifest is not None
    packs["raw_tokens"] = {
        "path": str(raw_pack_dir),
        "manifest": asdict(raw_manifest),
        "identity_sha256": raw_identity,
        "cold": raw_cold,
    }
    return {
        "pack_manifest": manifest,
        "pack_identity_sha256": f0_identity,
        "packs": packs,
    }


def enumerate_edge_stream(
    e_sup_positives: Sequence[tuple[str, str]],
    sampler: NegativeSampler,
    *,
    negative_ratio: int,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> list[tuple[str, str, int]]:
    """Enumerate one (epoch, rank) edge-stream pair list, deterministically.

    The single source of truth for the training loader
    (`_BatchFactory.epoch_batches`): positives are epoch-shuffled and
    rank-strided; negatives come from the pinned Sec 10.2 sampler seeded by
    ``(seed, epoch, rank)``; the combined list is shuffled with the same stream.

    Args:
        e_sup_positives: Canonical supervision positives (self-pairs included).
        sampler: The pinned negative sampler.
        negative_ratio: Negatives per positive.
        seed: Base seed.
        epoch: 1-based epoch.
        rank: Rank index.
        world_size: Rank count.

    Returns:
        Row list ``(u, v, label)`` for this epoch/rank.
    """
    positives = sorted(e_sup_positives)
    rng = np.random.default_rng((seed, epoch, rank, 0xE5))
    order = np.random.default_rng((seed, epoch, 0xE5)).permutation(len(positives))
    shard = [positives[i] for i in order.tolist()][rank::world_size]
    negatives = sampler.sample(shard, ratio=negative_ratio, seed=seed, epoch=epoch, rank=rank)
    rows = [(u, v, 1) for u, v in shard] + [(u, v, 0) for u, v in negatives]
    perm = rng.permutation(len(rows))
    return [rows[i] for i in perm.tolist()]


# --------------------------------------------------------------------------- data assembly


@dataclass
class EgoStitchData:
    """Everything the training loop consumes.

    Attributes:
        train_nodes: Sorted train-side node ids with F0 rows.
        e_sup_positives: Canonical supervision positives (self-pairs included).
        val_pairs: Validation pairs in canonical V_hold non-self manifest order.
        val_labels: Aligned validation labels.
        f0: Shape ``(N, d)`` float32 CPU matrix.
        node_index: Node id -> `f0` row.
        grounding_index: Shape ``(n_train, n_g)`` int64 rows into `f0` for each
            train node's pool, aligned with `train_nodes`.
        train_pos: Node id -> position in `train_nodes`.
        target_builder: The `EgoTargetBuilder` over ``G_struct``.
        sampler: The pinned negative sampler.
        rho_train: Message-partition edge density (spec Sec 9.3).
        feature_stats: Registered V_fit-only standardization constants.
    """

    train_nodes: list[str]
    e_sup_positives: list[tuple[str, str]]
    val_pairs: list[tuple[str, str]]
    val_labels: NDArray[np.int8]
    f0: torch.Tensor
    node_index: dict[str, int]
    grounding_index: NDArray[np.int64]
    train_pos: dict[str, int]
    target_builder: EgoTargetBuilder
    sampler: NegativeSampler
    rho_train: float
    internal_holdout: InternalHoldoutPartition | None = None
    validation_role: Literal["V_hold"] | None = None
    access_audit: dict[str, object] | None = None
    validation_nodes: tuple[str, ...] = ()
    validation_positive_edges: tuple[tuple[str, str], ...] = ()
    validation_grounding_index: NDArray[np.int64] | None = None
    validation_pos: dict[str, int] | None = None
    feature_stats: FeatureStats | None = None


def _read_labeled_pairs(path: Path) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Read only one explicitly allowed train-side labeled-pair file."""
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            u, v, raw_label = line.rstrip("\n").split("\t")
            label = int(raw_label)
            if label not in (0, 1):
                raise ValueError(f"non-binary label in {path}: {label}")
            pairs.append(canonical_pair(u, v))
            labels.append(label)
    return pairs, np.asarray(labels, dtype=np.int8)


def _sha256_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256("".join(f"{u}\t{v}\n" for u, v in sorted(pairs)).encode()).hexdigest()


def _assemble_e2e_data(
    cfg: EgoConfig,
    generator_cfg: EgoStitchConfig,
    *,
    pack_dir: Path | None,
) -> EgoStitchData:
    """Assemble E2E training data from train-side files only, with role-isolated holdouts.

    Raises:
        RuntimeError: When any benchmark input this assembly would open is held
            out -- checked before the first open, so nothing held out is read
            and no derived cache is written on the way to the raise.
    """
    # [H] Path-scoped held-out boundary (design 2026-07-29 Sec 3.1), first
    # statement of the function. It sat at the *end* of the assembly until
    # Wave 3, which meant `split.pkl`, `train_edges.txt` and `val_edges.txt`
    # were already read and the F0 / feature-statistics / grounding caches were
    # already written by the time it fired -- a post-mortem, not a guard.
    strategy_dir = _assert_input_boundary(cfg)
    split_path = strategy_dir / "split.pkl"
    with split_path.open("rb") as handle:
        split_payload = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    if not isinstance(split_payload, dict) or "train" not in split_payload:
        raise ValueError("split.pkl must contain a train node collection")
    train_nodes_all = sorted(
        set(cast(Sequence[str], split_payload["train"])) - set(cfg.data.expected_missing_features)
    )
    train_pairs, train_labels = _read_labeled_pairs(strategy_dir / "train_edges.txt")
    positives = [
        pair for pair, label in zip(train_pairs, train_labels, strict=True) if int(label) == 1
    ]
    partition = derive_partition(
        positives, seed=cfg.data.partition_seed, msg_fraction=cfg.data.msg_fraction
    )
    holdout = derive_internal_holdout(
        train_nodes_all,
        partition.e_msg,
        partition.e_sup,
    )
    # One universe pair for both stages (design 2026-07-29 Sec 2): training is
    # always the full V_fit, validation is always V_hold. Nothing here reads
    # `run_kind` any more, which is exactly what makes the two stages share a
    # pack, a grounding cache and a `feature_stats_sha256`.
    run_kind = cfg.run_kind or "formal"
    role: Literal["V_hold"] = _E2E_VALIDATION_ROLE
    validation = holdout.hold_manifest
    allowed_nodes = sorted(set(holdout.v_fit) | set(validation.nodes))

    store = FeatureStore(cfg.data.root / _FEATURES_SUBDIR)
    f0_cache = (pack_dir / _PACK_F0_FILENAME) if pack_dir is not None else cfg.data.f0_cache
    f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, node_index = build_f0_matrix(
        store,
        allowed_nodes,
        cache_path=f0_cache,
        allow_cache_subset=False,
    )
    fit_nodes = sorted(holdout.v_fit)
    fit_rows = np.asarray(
        matrix.numpy()[[node_index[node] for node in fit_nodes]], dtype=np.float32
    )
    feature_stats_cache = (
        (pack_dir / _PACK_FEATURE_STATS_FILENAME)
        if pack_dir is not None
        else cfg.data.f0_cache.with_name(_PACK_FEATURE_STATS_FILENAME)
    )
    feature_stats = feature_stats_for_universe(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        fit_nodes,
        cache_path=feature_stats_cache,
    )
    grounding_cache = (
        (pack_dir / _PACK_GROUNDING_FILENAME) if pack_dir is not None else cfg.data.grounding_cache
    )
    pool = build_grounding_pool(
        fit_rows,
        fit_nodes,
        n_ground=generator_cfg.n_ground,
        role_universe="V_fit",
        cache_path=grounding_cache,
    )
    grounding_index = np.asarray(
        [[node_index[neighbor] for neighbor in pool[node]] for node in fit_nodes], dtype=np.int64
    )
    validation_nodes: tuple[str, ...] = validation.nodes
    validation_positive_edges: tuple[tuple[str, str], ...] = validation.positive_edges
    validation_rows = np.asarray(
        matrix.numpy()[[node_index[node] for node in validation_nodes]], dtype=np.float32
    )
    validation_cache = (
        pack_dir / _PACK_VALIDATION_GROUNDING_FILENAME
        if pack_dir is not None
        else cfg.output_dir / "cache" / role / _PACK_GROUNDING_FILENAME
    )
    validation_pool = build_grounding_pool(
        validation_rows,
        validation_nodes,
        n_ground=generator_cfg.n_ground,
        role_universe=role,
        cache_path=validation_cache,
    )
    validation_grounding_index: NDArray[np.int64] = np.asarray(
        [[node_index[neighbor] for neighbor in validation_pool[node]] for node in validation_nodes],
        dtype=np.int64,
    )
    validation_pos: dict[str, int] = {node: index for index, node in enumerate(validation_nodes)}
    g_fit = holdout.build_g_fit()
    target_builder = EgoTargetBuilder(
        g_fit,
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        pool,
        slots=generator_cfg.slots,
    )
    degrees = {node: int(g_fit.degree(node)) for node in fit_nodes}
    rejection_positives = set(partition.e_msg) | set(partition.e_sup)
    val_path = strategy_dir / "val_edges.txt"
    validation_positive_membership_used = val_path.is_file()
    if validation_positive_membership_used:
        val_pairs, val_labels = _read_labeled_pairs(val_path)
        rejection_positives.update(
            pair for pair, label in zip(val_pairs, val_labels, strict=True) if int(label) == 1
        )
    sampler = NegativeSampler(fit_nodes, degrees, frozenset(rejection_positives))
    n_fit = len(fit_nodes)
    rho_train = g_fit.number_of_edges() / (math.comb(n_fit, 2) + n_fit)
    forbidden_files_absent = {
        name: not (strategy_dir / name).exists() for name in _HELD_OUT_FILENAMES
    }
    audit: dict[str, object] = {
        "run_kind": run_kind,
        "validation_role": role,
        "training_feature_nodes_sha256": hashlib.sha256(
            "".join(f"{node}\n" for node in fit_nodes).encode()
        ).hexdigest(),
        "validation_feature_nodes_sha256": validation.nodes_sha256,
        "training_structural_target_sha256": _sha256_pairs(sorted(holdout.e_msg_fit)),
        "training_supervision_sha256": _sha256_pairs(sorted(holdout.e_sup_fit)),
        "training_endpoints_within_v_fit": True,
        "structural_target_equals_e_msg_fit": {canonical_pair(u, v) for u, v in g_fit.edges()}
        == set(holdout.e_msg_fit),
        "validation_positive_membership_used_for_negative_rejection_only": (
            validation_positive_membership_used
        ),
        "forbidden_files_absent": forbidden_files_absent,
        "quarantine_counts": asdict(holdout.quarantine_counts),
        "overlap_proof": asdict(holdout.overlap_proof),
        "training_feature_stats_sha256": feature_stats.digest,
        "training_feature_stats_universe_sha256": feature_stats.node_ids_sha256,
        "training_feature_stats_rows": feature_stats.n_rows,
    }
    if audit["training_feature_stats_universe_sha256"] != audit["training_feature_nodes_sha256"]:
        raise RuntimeError(
            "feature standardization statistics were computed over a universe other than V_fit"
        )
    data = EgoStitchData(
        train_nodes=fit_nodes,
        e_sup_positives=sorted(holdout.e_sup_fit),
        val_pairs=list(validation.pairs),
        val_labels=np.asarray(validation.labels, dtype=np.int8),
        f0=matrix,
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(fit_nodes)},
        target_builder=target_builder,
        sampler=sampler,
        rho_train=rho_train,
        feature_stats=feature_stats,
        internal_holdout=holdout,
        validation_role=role,
        access_audit=audit,
        validation_nodes=validation_nodes,
        validation_positive_edges=validation_positive_edges,
        validation_grounding_index=validation_grounding_index,
        validation_pos=validation_pos,
    )
    return data


def assemble_egostitch_data(
    cfg: EgoConfig,
    *,
    pack_dir: Path | None = None,
) -> EgoStitchData:
    """Assemble the full training data bundle from the frozen artifacts.

    Args:
        cfg: The validated worker config.
        pack_dir: DDP pack directory (its F0/grounding caches win); ``None``
            uses ``cfg.data.f0_cache`` / ``cfg.data.grounding_cache``.
            The node stream uses the internal trainable generator and its
            registered rev-3.1 grounding/loss-calibration fields.

    Returns:
        The `EgoStitchData` bundle.

    Raises:
        RuntimeError: When any benchmark input this config would open is held
            out -- checked before the first open on both branches.
    """
    # [H] Path-scoped held-out boundary (design 2026-07-29 Sec 3.1), first
    # statement of the function so it precedes the first open.
    _assert_input_boundary(cfg)
    e2e_model_cfg = E2EConfig.from_mapping(cfg.model.config)
    generator_cfg = replace(
        EgoStitchConfig(),
        n_ground=e2e_model_cfg.n_ground,
        tau_adj=e2e_model_cfg.tau_adj,
        tau_div=e2e_model_cfg.tau_div,
        l_gate_pos_weight=e2e_model_cfg.l_gate_pos_weight,
    )
    return _assemble_e2e_data(cfg, generator_cfg, pack_dir=pack_dir)


# --------------------------------------------------------------------------- batches


def _epoch_step_plan(
    n_positives: int, *, negative_ratio: int, edge_batch: int, world_size: int
) -> tuple[list[int], int]:
    """Per-rank edge-row counts and the shared per-epoch step count.

    Every rank derives the identical plan from the same scalars (no
    communication): rank ``r`` owns ``ceil((n_pos - r) / world)`` positives and
    ``ratio`` negatives each; the epoch runs ``ceil(max_rows / B_e)`` steps on
    every rank (short ranks pad, spec-exact coverage is preserved by masking).
    """
    rows = [
        (n_positives - rank + world_size - 1) // world_size * (1 + negative_ratio)
        for rank in range(world_size)
    ]
    steps = max((max(rows) + edge_batch - 1) // edge_batch, 1) if max(rows) > 0 else 1
    return rows, steps


def _step_global_count(rows_per_rank: Sequence[int], step: int, edge_batch: int) -> int:
    """True (unpadded) edge rows across all ranks in one step."""
    return sum(max(0, min(rows - step * edge_batch, edge_batch)) for rows in rows_per_rank)


def _seeded_generator(*parts: int) -> torch.Generator:
    """A CPU torch generator seeded from a tuple (stable across runs/ranks)."""
    seed = int.from_bytes(
        hashlib.sha256(",".join(str(p) for p in parts).encode()).digest()[:8], "little"
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _edge_target_rng(node_id: str, *, seed: int, epoch: int) -> np.random.Generator:
    """Return the rev-3.1 node/epoch-keyed ego-target RNG (spec Sec 14.4.1).

    The 128-bit BLAKE2b node-identity key is XORed with the ordered
    ``(seed, epoch)`` pair packed as two unsigned 64-bit integers. Rank, step,
    pair identity, and endpoint direction are deliberately absent.
    """
    if not 0 <= seed < 2**64 or not 0 <= epoch < 2**64:
        raise ValueError("edge-target seed and epoch must fit unsigned 64-bit integers")
    node_key = int.from_bytes(
        hashlib.blake2b(node_id.encode("utf-8"), digest_size=16).digest(),
        "little",
    )
    seed_epoch_key = (seed << 64) | epoch
    return np.random.default_rng(node_key ^ seed_epoch_key)


def relational_pair_targets(
    graph: nx.Graph,
    rows: Sequence[tuple[str, str, int]],
) -> torch.Tensor:
    """Compute `L_rel` targets from `G_fit` for every positive or negative pair."""
    targets = torch.zeros(len(rows), 2, dtype=torch.float32)
    for row, (node_u, node_v, _) in enumerate(rows):
        if node_u not in graph or node_v not in graph:
            raise ValueError("relational target pair contains a node outside G_fit")
        neighbors_u = set(graph.neighbors(node_u))
        neighbors_v = set(graph.neighbors(node_v))
        common = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        targets[row, 0] = math.log1p(common)
        targets[row, 1] = common / union if union > 0 else 0.0
    return targets


@dataclass
class _CompositeBatch:
    """One composite optimizer-step batch (CPU tensors; moved by the caller).

    Edge rows are padded to ``B_e`` with ``edge_mask = 0`` rows so every rank
    executes the same step count.
    """

    node: dict[str, torch.Tensor]
    edge: dict[str, torch.Tensor]
    edge_rows_true: int
    edge_rows_global: int
    f0_rows_gathered: int


_BatchT = TypeVar("_BatchT")


def _prefetch_batches(batches: Iterator[_BatchT], *, depth: int) -> Generator[_BatchT, None, None]:
    """Build bounded deterministic CPU batches ahead of GPU consumption."""
    if depth <= 0:
        yield from batches
        return

    iterator = iter(batches)

    def read_next() -> _BatchT:
        return next(iterator)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="egostitch-batch")
    futures: deque[Future[_BatchT]] = deque(executor.submit(read_next) for _ in range(depth))
    try:
        while futures:
            future = futures.popleft()
            try:
                batch = future.result()
            except StopIteration:
                return
            futures.append(executor.submit(read_next))
            yield batch
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


_EGOSTITCH_E2E_FAMILY = "egostitch_e2e"


def _gather_token_streams(
    table: PackedFeatureTable,
    index: Mapping[str, int],
    endpoints_a: Sequence[str],
    endpoints_b: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Gather raw token streams for both pair endpoints (spec Sec 13.18).

    Both endpoints are gathered to a shared padded length ``T`` (the max true
    token length across every node in this batch), matching the packed-
    feature reader the B0 V3.1 loader consumes
    (:class:`~src.data.packed_features.PackedFeatureTable`). Shared by
    `_BatchFactory` (training edge batches) and `_validate_epoch` (family
    `egostitch_e2e` validation batches) so both build the identical contract.

    Args:
        table: The loaded packed-token store.
        index: Node id -> row in `table`.
        endpoints_a: The ``u`` node of each padded row.
        endpoints_b: The ``v`` node of each padded row.

    Returns:
        ``{"emb_a", "emb_b", "len_a", "len_b"}`` CPU tensors.
    """
    idx_a = torch.tensor([index[node] for node in endpoints_a], dtype=torch.long)
    idx_b = torch.tensor([index[node] for node in endpoints_b], dtype=torch.long)
    boundary = max(
        max(table.manifest.nodes[i].length for i in idx_a.tolist()),
        max(table.manifest.nodes[i].length for i in idx_b.tolist()),
    )
    emb_a, len_a = table.gather_nodes(idx_a, boundary)
    emb_b, len_b = table.gather_nodes(idx_b, boundary)
    return {"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b}


class _BatchFactory:
    """Deterministic composite-batch construction for one rank."""

    def __init__(
        self,
        cfg: EgoConfig,
        model_cfg: EgoStitchConfig,
        data: EgoStitchData,
        *,
        node_batch: int,
        rank: int,
        world_size: int,
    ) -> None:
        self._cfg = cfg
        self._model_cfg = model_cfg
        self._data = data
        self._node_batch = node_batch
        self._rank = rank
        self._world = world_size
        self._node_cursor = 0
        self._node_cycle = 0
        self._node_order = self._shuffled_nodes(0)
        self._f0_train_rows = torch.tensor(
            [data.node_index[node] for node in data.train_nodes], dtype=torch.long
        )
        self._allowed_training_rows = frozenset(self._f0_train_rows.tolist())
        self._allowed_training_nodes = frozenset(data.train_nodes)
        self._training_node_by_row = {data.node_index[node]: node for node in data.train_nodes}
        self.training_nodes_read: set[str] = set()
        self.training_f0_rows_read: set[int] = set()
        self._data.target_builder.set_feature_read_observer(self._record_training_nodes)
        self._token_table: PackedFeatureTable | None = None
        self._token_node_index: dict[str, int] | None = None
        if cfg.model.family == _EGOSTITCH_E2E_FAMILY:
            if cfg.data.pack_dir is None:
                raise ValueError("data.pack_dir is required when model.family == 'egostitch_e2e'")
            self._token_table = PackedFeatureTable.from_pack(cfg.data.pack_dir, torch.device("cpu"))
            self._token_node_index = self._token_table.manifest.node_index()

    # ---- node stream (cycles independently, spec Sec 10.1)

    def _shuffled_nodes(self, cycle: int) -> list[str]:
        rng = np.random.default_rng((self._cfg.seed, cycle, self._rank, 0x17))
        shard = self._data.train_nodes[self._rank :: self._world]
        order = rng.permutation(len(shard))
        return [shard[i] for i in order.tolist()]

    def _next_nodes(self) -> list[str]:
        nodes: list[str] = []
        while len(nodes) < self._node_batch:
            if self._node_cursor >= len(self._node_order):
                self._node_cycle += 1
                self._node_order = self._shuffled_nodes(self._node_cycle)
                self._node_cursor = 0
            take = min(self._node_batch - len(nodes), len(self._node_order) - self._node_cursor)
            nodes.extend(self._node_order[self._node_cursor : self._node_cursor + take])
            self._node_cursor += take
        return nodes

    def _ground_pool_rows(self, nodes: Sequence[str]) -> torch.Tensor:
        """Grounding-pool candidate rows into `self._data.f0` for `nodes`.

        These are the "global node id" values `egostitch_e2e`'s
        `ground_id_a`/`ground_id_b` batch keys carry (spec Sec 13.18): the
        same index space `x_i`/`x_j` are drawn from (both are rows into
        `self._data.f0`), so a grounding-pool candidate shared by both pair
        endpoints compares equal across sides, matching
        `src.score_universe._score_egostitch_e2e`'s `pool_rows` convention.
        """
        rows = self._data.grounding_index[[self._data.train_pos[n] for n in nodes]]
        self._record_training_nodes(nodes)
        self._record_training_rows(rows.reshape(-1).tolist())
        return torch.from_numpy(rows)

    def _record_training_nodes(self, nodes: Sequence[str]) -> None:
        invalid = set(nodes) - self._allowed_training_nodes
        if invalid:
            raise RuntimeError(f"training feature/token read escaped V_fit: {sorted(invalid)[:3]}")
        self.training_nodes_read.update(nodes)
        self._record_training_rows([self._data.node_index[node] for node in nodes])

    def _record_training_rows(self, rows: Sequence[int]) -> None:
        invalid = set(rows) - self._allowed_training_rows
        if invalid:
            raise RuntimeError(f"training F0 read escaped V_fit: {sorted(invalid)[:3]}")
        self.training_f0_rows_read.update(rows)
        self.training_nodes_read.update(self._training_node_by_row[row] for row in rows)

    def _ground_rows(self, nodes: Sequence[str]) -> torch.Tensor:
        return self._data.f0[self._ground_pool_rows(nodes)]

    def _node_tensors(
        self, nodes: Sequence[str], targets: EgoTargets, *, epoch: int, step: int
    ) -> dict[str, torch.Tensor]:
        batch = len(nodes)
        k_d = max(1, self._model_cfg.slots // 2)
        gen = _seeded_generator(self._cfg.seed, epoch, step, self._rank, 0x0D)
        self._record_training_nodes(nodes)
        x = self._data.f0[torch.tensor([self._data.node_index[n] for n in nodes], dtype=torch.long)]
        ground = self._ground_rows(nodes)

        # Conditioning dropout: two disjoint p = 0.1 nulls (spec Sec 2).
        draw = torch.rand(batch, generator=gen)
        null_mode = torch.full((batch,), NULL_MODE_FULL, dtype=torch.long)
        null_mode[draw < self._model_cfg.null_dropout] = NULL_MODE_CONTENT
        null_mode[
            (draw >= self._model_cfg.null_dropout) & (draw < 2 * self._model_cfg.null_dropout)
        ] = NULL_MODE_ALL

        # Denoising queries on 25% of nodes (spec Sec 2), K/2 noised targets.
        denoise_selected = torch.rand(batch, generator=gen) < self._model_cfg.denoise_fraction
        denoise_features = targets.features[:, :k_d]
        denoise_mask = targets.mask[:, :k_d] & denoise_selected[:, None]
        denoise_noise = self._model_cfg.denoise_sigma * torch.randn(
            batch, k_d, self._model_cfg.d_p, generator=gen
        )

        # SSL inputs (spec Sec 7): feature noise + pool resample.
        ssl_noise = self._model_cfg.ssl_noise_sigma * torch.randn(
            batch, self._model_cfg.input_dim, generator=gen
        )
        resample_rows = torch.randint(
            0,
            len(self._data.train_nodes),
            (batch, self._model_cfg.n_ground),
            generator=gen,
        )
        resampled_rows = self._f0_train_rows[resample_rows]
        self._record_training_rows(resampled_rows.reshape(-1).tolist())
        ground_resampled = self._data.f0[resampled_rows]

        return {
            "x": x,
            "ground_x": ground,
            "target_features": targets.features,
            "target_mult": targets.mult,
            "target_adj": targets.adj,
            "target_mask": targets.mask,
            "target_in_pool": targets.in_pool,
            "target_pool_index": targets.pool_index,
            "true_degree": targets.degree,
            "real_ego_stats": targets.ego_stats,
            "null_mode": null_mode,
            "denoise_features": denoise_features,
            "denoise_mask": denoise_mask,
            "denoise_noise": denoise_noise,
            "ssl_noise": ssl_noise,
            "ground_resampled": ground_resampled,
        }

    def _token_streams(
        self, endpoints_a: Sequence[str], endpoints_b: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Gather raw token streams for both pair endpoints (spec Sec 13.18)."""
        table = self._token_table
        index = self._token_node_index
        assert table is not None and index is not None  # family-gated by the caller
        self._record_training_nodes((*endpoints_a, *endpoints_b))
        return _gather_token_streams(table, index, endpoints_a, endpoints_b)

    def _edge_target_tensors(
        self,
        rows: Sequence[tuple[str, str, int]],
        *,
        true_rows: int,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Build positive-real-row endpoint targets keyed only by node and epoch."""
        batch = len(rows)
        slots = self._model_cfg.slots
        input_dim = self._model_cfg.input_dim
        targets: dict[str, torch.Tensor] = {}
        for side in ("i", "j"):
            targets[f"target_features_{side}"] = torch.zeros(
                batch, slots, input_dim, dtype=torch.float32
            )
            targets[f"target_mult_{side}"] = torch.zeros(batch, slots, dtype=torch.float32)
            targets[f"target_adj_{side}"] = torch.zeros(
                batch, slots, slots, dtype=torch.float32
            )
            targets[f"target_mask_{side}"] = torch.zeros(batch, slots, dtype=torch.bool)
            targets[f"target_node_index_{side}"] = torch.full(
                (batch, slots), -1, dtype=torch.long
            )

        cached: dict[str, EgoTargets] = {}
        for row_index, (node_i, node_j, label) in enumerate(rows):
            if row_index >= true_rows or label != 1:
                continue
            for side, node_id in (("i", node_i), ("j", node_j)):
                node_targets = cached.get(node_id)
                if node_targets is None:
                    node_targets = self._data.target_builder.build(
                        [node_id],
                        _edge_target_rng(node_id, seed=self._cfg.seed, epoch=epoch),
                    )
                    cached[node_id] = node_targets
                targets[f"target_features_{side}"][row_index] = node_targets.features[0]
                targets[f"target_mult_{side}"][row_index] = node_targets.mult[0]
                targets[f"target_adj_{side}"][row_index] = node_targets.adj[0]
                targets[f"target_mask_{side}"][row_index] = node_targets.mask[0]
                targets[f"target_node_index_{side}"][row_index] = node_targets.node_index[0]
        return targets

    def _edge_tensors(
        self,
        rows: Sequence[tuple[str, str, int]],
        *,
        pad_to: int,
        epoch: int,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], int]:
        del step  # Rev-3.1 target sampling is explicitly step-independent.
        true_rows = len(rows)
        padded: list[tuple[str, str, int]] = list(rows)
        if true_rows == 0:
            filler = (self._data.train_nodes[0], self._data.train_nodes[0], 0)
            padded = [filler]
        while len(padded) < pad_to:
            padded.append(padded[0])
        self._record_training_nodes([node for u, v, _ in padded for node in (u, v)])
        idx_i = torch.tensor([self._data.node_index[u] for u, _, _ in padded], dtype=torch.long)
        idx_j = torch.tensor([self._data.node_index[v] for _, v, _ in padded], dtype=torch.long)
        edge: dict[str, torch.Tensor] = {
            "x_i": self._data.f0[idx_i],
            "x_j": self._data.f0[idx_j],
            "ground_i": self._ground_rows([u for u, _, _ in padded]),
            "ground_j": self._ground_rows([v for _, v, _ in padded]),
            "label": torch.tensor([lab for _, _, lab in padded], dtype=torch.float32),
            "is_self": torch.tensor([u == v for u, v, _ in padded], dtype=torch.bool),
            "edge_mask": torch.tensor(
                [1.0 if i < true_rows else 0.0 for i in range(len(padded))],
                dtype=torch.float32,
            ),
        }
        if self._token_table is None:
            raise RuntimeError("egostitch_e2e batches require a packed token table")
        endpoints_u = [u for u, _, _ in padded]
        endpoints_v = [v for _, v, _ in padded]
        edge.update(self._token_streams(endpoints_u, endpoints_v))
        # Same-index-space grounding ids for both endpoints (spec Sec 13.18).
        edge["ground_id_i"] = self._ground_pool_rows(endpoints_u)
        edge["ground_id_j"] = self._ground_pool_rows(endpoints_v)
        edge.update(self._edge_target_tensors(padded, true_rows=true_rows, epoch=epoch))
        edge["rel_target"] = relational_pair_targets(self._data.target_builder.graph, padded)
        return edge, true_rows

    def epoch_batches(
        self, epoch: int, *, rows_per_rank: Sequence[int], steps: int
    ) -> Iterator[_CompositeBatch]:
        """Yield the epoch's composite batches for this rank."""
        edge_rows = enumerate_edge_stream(
            self._data.e_sup_positives,
            self._data.sampler,
            negative_ratio=self._cfg.data.negative_ratio,
            seed=self._cfg.seed,
            epoch=epoch,
            rank=self._rank,
            world_size=self._world,
        )
        edge_batch = self._cfg.data.edge_batch
        for step in range(steps):
            nodes = self._next_nodes()
            targets = self._data.target_builder.build(
                nodes, np.random.default_rng((self._cfg.seed, epoch, step, self._rank, 0x7A))
            )
            node = self._node_tensors(nodes, targets, epoch=epoch, step=step)
            chunk = edge_rows[step * edge_batch : (step + 1) * edge_batch]
            edge, true_rows = self._edge_tensors(
                chunk,
                pad_to=edge_batch,
                epoch=epoch,
                step=step,
            )
            global_count = _step_global_count(rows_per_rank, step, edge_batch)
            f0_rows = (
                node["x"].shape[0]
                + node["ground_x"].shape[0] * node["ground_x"].shape[1]
                + node["target_features"].shape[0] * node["target_features"].shape[1]
                + 4 * edge["x_i"].shape[0]  # x_i, x_j and both grounding gathers
                + sum(
                    edge[name].shape[0] * edge[name].shape[1]
                    for name in ("target_features_i", "target_features_j")
                    if name in edge
                )
            )
            yield _CompositeBatch(
                node=node,
                edge=edge,
                edge_rows_true=true_rows,
                edge_rows_global=max(global_count, 1),
                f0_rows_gathered=f0_rows,
            )


# --------------------------------------------------------------------------- composite module


class _CompositeStep(torch.nn.Module):
    """One-forward composite step so DDP sees a single forward per backward.

    The warm-start curriculum keeps every stream running with a constant
    autograd shape but suppresses ``L_edge``, ``L_real``, and ``L_ssl``.
    ``L_recon`` remains fully active, including the rev-3.1 ``L_align`` and
    ``L_rel`` edge-stream components.

    Family `egostitch_e2e` (design rev 3, spec Sec 14; three-component
    refactor design 2026-08-02): ``model`` is an `EgoStitchModel`. The node
    stream (``L_recon``/``L_real``/``L_ssl``) and the edge stream's
    ``L_align`` are delegated to `EgoStitchModel.generator.auxiliary_losses`;
    ``L_rel`` to `EgoStitchModel.encoder.auxiliary_losses` (design §6: each
    component owns the losses that supervise it, so swapping a component
    swaps its auxiliary losses with it). This module never reaches into
    `generator.stage1` directly -- that reach-through was the exact coupling
    the three-component split exists to remove. Both calls reuse the
    `ImaginedGraph` / AB `GraphEmbedding` that `model.forward` already built
    to score ``logits`` (its ``"graph"``/``"embedding_ab"`` output), so no
    second stitch+encode pass runs. This module keeps applying
    `real_ssl_scale` (the Phase A/B warm-start ramp) and
    `losses.stage1_total`/`.stage1_family_tensors` itself -- that is trainer
    policy, not model policy (design §6/§12 P2 retargeting note;
    `composite.py`'s "loss aggregation" section explains why no
    `EgoStitchModel.aggregate_losses` exists to do this instead).
    """

    def __init__(self, model: EgoStitchModel, world_size: int) -> None:
        super().__init__()
        self.model = model
        self.world_size = world_size

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        node = cast(dict[str, torch.Tensor], batch["node"])
        edge = cast(dict[str, torch.Tensor], batch["edge"])
        collect_diagnostics = bool(batch.get("collect_diagnostics", False))

        extra: dict[str, object] = {}
        edge_active = bool(batch.get("edge_active", True))
        real_ssl_scale = cast(torch.Tensor, batch["real_ssl_scale"])
        seed = cast(int, batch["seed"])
        epoch = cast(int, batch["epoch"])
        step = cast(int, batch["step"])
        if self.model.cfg.permanent_null == "none":
            branch_masks = sample_branch_masks(
                edge["label"].shape[0],
                self.model.cfg.p_topo,
                generator=_seeded_generator(seed, epoch, step),
                device=edge["label"].device,
            )
        else:
            branch_masks = masks_for_null(
                self.model.cfg.permanent_null,
                edge["label"].shape[0],
                edge["label"].device,
            )
        edge_view = _e2e_edge_view(edge)
        edge_view.update(
                {
                    "target_features_a": edge["target_features_i"],
                    "target_features_b": edge["target_features_j"],
                    "target_mult_a": edge["target_mult_i"],
                    "target_mult_b": edge["target_mult_j"],
                    "target_adj_a": edge["target_adj_i"],
                    "target_adj_b": edge["target_adj_j"],
                    "target_mask_a": edge["target_mask_i"],
                    "target_mask_b": edge["target_mask_j"],
                    "target_node_index_a": edge["target_node_index_i"],
                    "target_node_index_b": edge["target_node_index_j"],
                    "rel_target": edge["rel_target"],
                    "edge_mask": edge["edge_mask"],
                    "label": edge["label"],
                    "loss_world_size": torch.tensor(
                        self.world_size,
                        device=edge["label"].device,
                    ),
                }
            )
        edge_output = self.model(edge_view, masks=branch_masks)
        logits = cast(torch.Tensor, edge_output["logits"])
        # `self.model.generator` is concretely `EgoStitchImagineGenerator` for
        # P2 (registry-driven swapping is P3), whose `stitch` only ever
        # produces a `StitchedGraph` -- match its own `auxiliary_losses`
        # contract exactly rather than the wider base `ImaginedGraph`.
        graph = cast(StitchedGraph, edge_output["graph"])
        embedding_ab = cast(GraphEmbedding, edge_output["embedding_ab"])

        # Every key `NeighborhoodGenerator.auxiliary_losses`/
        # `GraphEncoder.auxiliary_losses` read is either a `node`-stream key
        # (unsuffixed) or an `edge_view`-stream key (`_a`/`_b`-suffixed or
        # distinctly named), so the merge below is a disjoint union -- see
        # `generator/egostitch.py:auxiliary_losses`'s docstring for the exact
        # key inventory.
        auxiliary_batch = {**node, **edge_view}
        generator_losses = self.model.generator.auxiliary_losses(graph, auxiliary_batch)
        encoder_losses = self.model.encoder.auxiliary_losses(embedding_ab, auxiliary_batch)
        recon = {
            "feat": generator_losses["feat"],
            "exist": generator_losses["exist"],
            "mult": generator_losses["mult"],
            "slotadj": generator_losses["slotadj"],
            "gate": generator_losses["gate"],
            "ptr": generator_losses["ptr"],
            "div": generator_losses["div"],
            "align": generator_losses["align"],
            "rel": encoder_losses["rel_loss"],
        }

        if collect_diagnostics:
            extra.update(_e2e_gate_tanh(self.model))
        edge_loss = (
            e2e_weighted_bce_with_logits(
                logits, edge["label"], edge["edge_mask"], world_size=self.world_size
            )
            if edge_active
            else logits.sum() * 0.0
        )

        total, parts = stage1_total(
            self.model.generator_cfg,
            family="egostitch_e2e",
            edge=edge_loss,
            recon=recon,
            deg=generator_losses["deg"],
            real_egostat=generator_losses["real_egostat"] * real_ssl_scale,
            real_gin=generator_losses["real_gin"] * real_ssl_scale,
            ssl_noise=generator_losses["ssl_noise"] * real_ssl_scale,
            ssl_pool=generator_losses["ssl_pool"] * real_ssl_scale,
            recon_factors=cast(
                Mapping[str, float] | None,
                batch.get("recon_factors"),
            ),
        )
        families = stage1_family_tensors(
            self.model.generator_cfg,
            family="egostitch_e2e",
            edge=edge_loss,
            recon=recon,
            deg=generator_losses["deg"],
            real_egostat=generator_losses["real_egostat"] * real_ssl_scale,
            real_gin=generator_losses["real_gin"] * real_ssl_scale,
            ssl_noise=generator_losses["ssl_noise"] * real_ssl_scale,
            ssl_pool=generator_losses["ssl_pool"] * real_ssl_scale,
            recon_factors=cast(
                Mapping[str, float] | None,
                batch.get("recon_factors"),
            ),
        )
        parameter_anchor = 0.0 * torch.stack(
            tuple(
                parameter.sum()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        ).sum()
        total = total + parameter_anchor
        families = {name: family + parameter_anchor for name, family in families.items()}
        result: dict[str, object] = {
            "loss": total,
            "parts": parts,
        }
        if collect_diagnostics:
            result["families"] = families
            result.update(extra)
        return result


def _to_device(value: object, device: torch.device) -> object:
    """Recursively move tensors in a nested batch structure to `device`."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _detached_clone(value: object) -> object:
    """Clone a nested payload so the gradient probe is fixed across steps."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _detached_clone(item) for key, item in value.items()}
    return value


def _e2e_edge_view(edge: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate the worker's edge-tensor keys to `EgoStitchModel`'s batch contract.

    Maps ``x_i``/``x_j`` to ``x_a``/``x_b`` and ``ground_i``/``ground_j``/
    ``ground_id_i``/``ground_id_j`` to ``ground_a``/``ground_b``/
    ``ground_id_a``/``ground_id_b`` (spec Sec 13.18: the real grounding-
    candidate features and their same-index-space global ids, required for
    `EgoStitchModel`'s grounded-identity-match flag to engage instead of its
    degenerate placeholder path); ``emb_a``/``emb_b``/``len_a``/``len_b``
    already match and pass through unchanged. `_BatchFactory._edge_tensors`
    always populates ``ground_i``/``ground_j``/``ground_id_i``/``ground_id_j``
    for this family (Task 12/this task), so every key below is always present
    at the one call site (`_CompositeStep.forward`'s `egostitch_e2e` branch).
    """
    return {
        "emb_a": edge["emb_a"],
        "emb_b": edge["emb_b"],
        "len_a": edge["len_a"],
        "len_b": edge["len_b"],
        "x_a": edge["x_i"],
        "x_b": edge["x_j"],
        "ground_a": edge["ground_i"],
        "ground_b": edge["ground_j"],
        "ground_id_a": edge["ground_id_i"],
        "ground_id_b": edge["ground_id_j"],
        "is_self": edge["is_self"],
    }


def _e2e_gate_tanh(model: EgoStitchModel) -> dict[str, list[float]]:
    """Per-injected-block ``tanh(gate)`` readout for the topology conditioning pathway.

    Registered name (spec Sec 13.17): ``gate_topo_tanh``. A pure parameter
    readout -- no forward/backward pass required, safe to call every step.
    """

    def _values(modules: torch.nn.ModuleList) -> list[float]:
        out: list[float] = []
        for module in modules:
            assert isinstance(module, GatedCrossAttention)
            out.append(float(torch.tanh(module.gate).detach()))
        return out

    return {
        "gate_topo_tanh": _values(model.classifier.trunk.topo_xattn),
    }


def _e2e_submodule_gradient_rms(model: EgoStitchModel, loss: torch.Tensor) -> dict[str, float]:
    """Per-submodule gradient RMS from one isolated retained-graph backward.

    Registered names (spec Sec 13.17): ``grad_rms_trunk``, ``grad_rms_ste``.
    Measures how much of ``loss``'s gradient reaches the pair trunk
    (`model.classifier.trunk`) and the stitched-topology encoder
    (`model.encoder`, née `model.ste`) -- the zero-init gated pathways the
    warm-start curriculum is designed to keep dead until ``L_edge``
    activates. Mirrors `_family_gradient_norms`'s isolated-backward pattern:
    intended to be called on a dedicated probe forward's output (spec Sec
    13.17's fixed replay batch), not on the tensor the caller is about to
    call its own `.backward()` on -- leaves every parameter's ``.grad`` at
    ``None`` afterward either way.
    """
    groups: dict[str, list[torch.nn.Parameter]] = {
        "grad_rms_trunk": list(model.classifier.trunk.parameters()),
        "grad_rms_ste": list(model.encoder.parameters()),
    }
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
    rms: dict[str, float] = {}
    for name, parameters in groups.items():
        squared = loss.new_zeros((), dtype=torch.float32)
        count = 0
        for parameter in parameters:
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().float().square().sum()
                count += parameter.grad.numel()
        rms[name] = float(torch.sqrt(squared / max(count, 1)))
    model.zero_grad(set_to_none=True)
    return rms


def _e2e_topology_delta_std(
    model: EgoStitchModel, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Std of ``full - f_logit`` over one batch (design Sec 14, spec Sec 13.17).

    Registered name: ``topology_delta_std``, intended as a per-epoch signal on
    a fixed validation slice; this function measures the quantity for
    whatever batch it is given, so the caller controls the "fixed validation
    slice, once per epoch" cadence. Returns a keyed dict (rather than a bare
    float) to match the shape of its sibling telemetry helpers
    (`_e2e_gate_tanh`, `_e2e_submodule_gradient_rms`), so a caller can
    `dict.update(...)` it directly into a metrics row.
    """
    return {"topology_delta_std": _e2e_topology_fidelity(model, batch)["topology_delta_std"]}


def _e2e_topology_fidelity(
    model: EgoStitchModel, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Compute the validation topology tie-break with one shared pair context."""
    with torch.no_grad():
        context = model.build_pair_context(batch)
        full = model.score_pair_context(context)
        f_logit = model.score_pair_context(
            context,
            masks=masks_for_null(NULL_ALL_HEAD, full.shape[0], full.device),
        )
    delta = (full - f_logit).detach()
    topology_delta_std = 0.0 if delta.numel() < 2 else float(torch.std(delta))
    f_logit_std = float(np.std(f_logit.detach().float().cpu().numpy()))
    return {
        "topology_delta_std": topology_delta_std,
        "f_logit_std": f_logit_std,
        "topology_delta_ratio": topology_delta_std / max(f_logit_std, 1e-30),
    }


def _e2e_null_arm_tiebreak(logits: np.ndarray) -> dict[str, float]:
    """Return null-arm validation diagnostics without excluded-pathway selection.

    Permanent-null arms choose checkpoints by their active validation AUPRC;
    their tie-break is pinned to zero so an excluded pathway cannot affect the
    selected checkpoint.
    """
    return {"active_logit_std": float(np.std(logits)), "selection_tiebreak": 0.0}


def _e2e_trainable_parameters(model: EgoStitchModel) -> list[torch.nn.Parameter]:
    """Trainable parameters for family `egostitch_e2e`, excluding dead ones.

    `DecisionHead` (the frozen-s0 family's ``(s0, s1, s2)`` fusion head, spec
    Sec 13.1) is deleted entirely along with the content path -- there is no
    longer a `generator.stage1.decision` submodule to special-case.
    `generator.stage1.random_gin` *is* exercised (via `node_losses`'s
    ``L_real`` energy-distance term) but is already frozen at construction
    (`requires_grad=False`, spec Sec 13.6), so the plain `requires_grad`
    filter below excludes it without any name-based special case.
    """
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _e2e_optimizer_parameters(
    model: EgoStitchModel, composite: _CompositeStep
) -> list[torch.nn.Parameter]:
    """Return only live E2E model parameters; Kendall is frozen by §13.19."""
    del composite
    return _e2e_trainable_parameters(model)


@dataclass
class _GradientImbalanceMonitor:
    """Track persistence of a registered high-family gradient imbalance."""

    ratio: float
    required_steps: int
    interval: int
    streak_steps: int = 0
    activated_step: int | None = None

    def update(self, step: int, norms: dict[str, float]) -> bool:
        values = np.asarray(list(norms.values()), dtype=np.float64)
        median = float(np.median(values))
        imbalanced = bool(values.max(initial=0.0) > self.ratio * max(median, 1e-30))
        self.streak_steps = self.streak_steps + self.interval if imbalanced else 0
        if self.activated_step is None and self.streak_steps >= self.required_steps:
            self.activated_step = step
            return True
        return False


def _enforce_probe_s1_scale(s1_abs_mean: float, limit: float) -> None:
    """Reject a membership channel that has returned to the saturated scale."""
    if s1_abs_mean > limit:
        raise RuntimeError(
            "pathological Stage-1 membership scale: "
            f"|mean(s1)|={s1_abs_mean:.6g} exceeds registered limit {limit:.6g}"
        )


# --------------------------------------------------------------------------- validation


@dataclass(frozen=True)
class _ValidationResult:
    metrics: EdgeMetrics
    fidelity: dict[str, float]
    scale_telemetry: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    active_logits: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype="<f4")
    )


def _validation_clustering_mmd(data: EgoStitchData, logits: np.ndarray) -> float:
    """Raw single-graph clustering MMD at the exact held-out gold edge count."""
    if not data.validation_nodes:
        return 0.0
    target_edges = len(data.validation_positive_edges)
    ranked = sorted(
        range(len(data.val_pairs)),
        key=lambda index: (-float(logits[index]), data.val_pairs[index]),
    )
    predicted = nx.Graph()
    predicted.add_nodes_from(data.validation_nodes)
    predicted.add_edges_from(data.val_pairs[index] for index in ranked[:target_edges])
    gold = nx.Graph()
    gold.add_nodes_from(data.validation_nodes)
    gold.add_edges_from(data.validation_positive_edges)
    return mmd_squared(
        [clustering_histogram(predicted)],
        [clustering_histogram(gold)],
        MMDConfig(),
    )


def _e2e_validation_grounding_rows(
    data: EgoStitchData,
    nodes: Sequence[str],
) -> NDArray[np.int64]:
    """Resolve role-specific validation grounding to global F0 row ids."""
    validation_index = data.validation_grounding_index
    validation_pos = data.validation_pos or {}
    rows: list[NDArray[np.int64]] = []
    for node in nodes:
        if node in data.train_pos:
            rows.append(data.grounding_index[data.train_pos[node]])
        elif validation_index is not None and node in validation_pos:
            rows.append(validation_index[validation_pos[node]])
        else:
            raise RuntimeError(f"no role-specific grounding row for validation node {node!r}")
    return np.stack(rows).astype(np.int64, copy=False)


def _e2e_validation_node_batch(
    data: EgoStitchData,
    token_table: PackedFeatureTable,
    token_node_index: Mapping[str, int],
    nodes: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build one unique-node validation encode batch."""
    packed_rows = torch.tensor([token_node_index[node] for node in nodes], dtype=torch.long)
    boundary = max(token_table.manifest.nodes[row].length for row in packed_rows.tolist())
    emb, length = token_table.gather_nodes(packed_rows, boundary)
    node_rows = torch.tensor([data.node_index[node] for node in nodes], dtype=torch.long)
    grounding_rows = _e2e_validation_grounding_rows(data, nodes)
    batch = {
        "emb": emb,
        "length": length,
        "x": data.f0[node_rows],
        "ground": data.f0[torch.from_numpy(grounding_rows)],
        "ground_ids": torch.from_numpy(grounding_rows),
    }
    return cast(dict[str, torch.Tensor], _to_device(batch, device))


def _fp32_cached_node_state(state: E2ENodeState, row: int) -> E2ENodeState:
    """Detach one encoded node while preserving integer identity fields exactly."""
    length = state.length[row : row + 1].clone()
    true_length = int(length.item())
    slots = SlotSet(
        *(
            value[row : row + 1].float().clone()
            if value.is_floating_point()
            else value[row : row + 1].clone()
            for value in state.slots
        )
    )
    ground_ids = None
    if state.ground_ids is not None:
        ground_ids = state.ground_ids[row : row + 1].clone()
    return E2ENodeState(
        encoded=state.encoded[row : row + 1, :true_length].float().clone(),
        length=length,
        slots=slots,
        projected_x=state.projected_x[row : row + 1].float().clone(),
        ground_ids=ground_ids,
    )


def _stack_cached_node_states(
    cache: Mapping[str, E2ENodeState], nodes: Sequence[str]
) -> E2ENodeState:
    """Reconstruct one pair-endpoint state batch from the per-rank cache."""
    states = [cache[node] for node in nodes]
    width = max(state.encoded.size(1) for state in states)
    encoded = torch.cat(
        [F.pad(state.encoded, (0, 0, 0, width - state.encoded.size(1))) for state in states]
    )
    ground_ids: torch.Tensor | None
    if any(state.ground_ids is None for state in states):
        if not all(state.ground_ids is None for state in states):
            raise RuntimeError("validation node cache has inconsistent grounding ids")
        ground_ids = None
    else:
        ground_ids = torch.cat([cast(torch.Tensor, state.ground_ids) for state in states])
    return E2ENodeState(
        encoded=encoded,
        length=torch.cat([state.length for state in states]),
        slots=SlotSet(
            *(torch.cat(values) for values in zip(*(state.slots for state in states), strict=True))
        ),
        projected_x=torch.cat([state.projected_x for state in states]),
        ground_ids=ground_ids,
    )


def _e2e_validation_slice_rows(n_val: int) -> tuple[int, ...]:
    """Frozen global rows used by the E2E checkpoint-selection tie-break."""
    if n_val <= 0:
        return ()
    return tuple(range(max(1, math.ceil(0.01 * n_val))))


def _validate_epoch(
    model: EgoStitchModel,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    edge_batch: int,
    topk_fraction: float,
    token_table: PackedFeatureTable | None = None,
    token_node_index: Mapping[str, int] | None = None,
) -> _ValidationResult | None:
    """Score the validation pairs with exact distributed coverage.

    Rows are rank-strided and padded to a common shard length; the main rank
    deduplicates gathered row ids and computes the metrics (``None`` on other
    ranks).

    Family `egostitch_e2e` (spec Sec 13.17 re-registration): every validation
    pair is scored through the configured permanent-null arm in eval mode (no
    branch dropout); the ``none`` arm is the true full decomposition. There is
    no `s0` fusion and no self/non-self split (a self pair is simply
    ``x_a == x_b``/``emb_a == emb_b``, handled internally by
    `EgoStitchModel.forward`). `token_table`/`token_node_index` (the packed
    raw-token store `_BatchFactory` already loaded) are required for this
    family to build the ``emb_a``/``emb_b`` batch keys. The per-epoch
    `topology_delta_std` telemetry and checkpoint-selection tie-break apply
    only to the full ``none`` arm; permanent-null arms select by active-arm
    AUPRC without that topology tie-break.
    """
    was_training = model.training
    model.eval()
    n_val = len(data.val_pairs)
    rank, world = accelerator.process_index, accelerator.num_processes
    shard_rows = list(range(rank, n_val, world))
    shard_len = (n_val + world - 1) // world
    while len(shard_rows) < shard_len:
        shard_rows.append(shard_rows[0] if shard_rows else 0)

    assert token_table is not None and token_node_index is not None, (
        "family egostitch_e2e requires token_table/token_node_index"
    )
    unique_nodes = list(
        dict.fromkeys(node for row in shard_rows for node in data.val_pairs[row])
    )

    def synchronize_device() -> None:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)

    synchronize_device()
    node_cache_started = time.monotonic()
    node_cache: dict[str, E2ENodeState] = {}
    values_out: list[torch.Tensor] = []
    # Validation may run inside an outer autocast context (for example the
    # CPU bf16 contract test).  `inference_mode` can then seed autocast's
    # weight cache with inference tensors that a later training forward tries
    # to save for backward.  `no_grad` preserves eval semantics without
    # contaminating the following optimizer step.
    with torch.no_grad():
        length_buckets: dict[int, list[str]] = {}
        for node in unique_nodes:
            length = token_table.manifest.nodes[token_node_index[node]].length
            bucket = next(
                (boundary for boundary in (128, 256, 384, 512, 768, 1024) if length <= boundary),
                length,
            )
            length_buckets.setdefault(bucket, []).append(node)
        for bucket_nodes in length_buckets.values():
            for start in range(0, len(bucket_nodes), edge_batch):
                nodes = bucket_nodes[start : start + edge_batch]
                node_batch = _e2e_validation_node_batch(
                    data, token_table, token_node_index, nodes, accelerator.device
                )
                with torch.autocast(
                    device_type=accelerator.device.type,
                    dtype=torch.bfloat16,
                ):
                    encoded = model.encode_node_state(
                        node_batch["emb"],
                        node_batch["length"],
                        node_batch["x"],
                        node_batch["ground"],
                        node_batch["ground_ids"],
                    )
                for offset, node in enumerate(nodes):
                    node_cache[node] = _fp32_cached_node_state(encoded, offset)
        synchronize_device()
        node_cache_seconds = time.monotonic() - node_cache_started
        pair_scoring_started = time.monotonic()
        for start in range(0, shard_len, edge_batch):
            chunk = shard_rows[start : start + edge_batch]
            rows = [(*data.val_pairs[i], int(data.val_labels[i])) for i in chunk]
            endpoints_u = [u for u, _, _ in rows]
            endpoints_v = [v for _, v, _ in rows]
            state_a = _stack_cached_node_states(node_cache, endpoints_u)
            state_b = _stack_cached_node_states(node_cache, endpoints_v)
            is_self = torch.tensor(
                [u == v for u, v, _ in rows],
                dtype=torch.bool,
                device=accelerator.device,
            )
            masks = (
                None
                if model.cfg.permanent_null == "none"
                else masks_for_null(
                    model.cfg.permanent_null,
                    len(rows),
                    accelerator.device,
                )
            )
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                context = model.build_pair_context_from_states(state_a, state_b, is_self)
                full_logits = model.score_pair_context(context)
                f_logits = model.score_pair_context(
                    context,
                    masks=masks_for_null(
                        NULL_ALL_HEAD,
                        len(rows),
                        accelerator.device,
                    ),
                )
                active_logits = (
                    full_logits
                    if masks is None
                    else model.score_pair_context(context, masks=masks)
                )
            assert context.plan is not None
            dispersion_a = _e2e_dispersion_rows(
                state_a.slots.pi,
                state_a.slots.h,
                state_a.slots.adj,
                context.plan,
            )
            dispersion_b = _e2e_dispersion_rows(
                state_b.slots.pi,
                state_b.slots.h,
                state_b.slots.adj,
                context.plan,
            )
            dispersion_rows = {
                name: (
                    0.5 * (dispersion_a[name] + dispersion_b[name])
                    if name
                    in {"pi_slot_std", "h_pairwise_cosine_mean", "adj_offdiag_std"}
                    else dispersion_a[name]
                )
                for name in dispersion_a
            }
            for name in ("plan_row_entropy", "plan_rank1_marginal_residual"):
                dispersion_rows[name] = dispersion_rows[name].masked_fill(is_self, torch.nan)
            scale_a = _e2e_scale_rows(state_a.slots.h, context.plan)
            scale_b = _e2e_scale_rows(state_b.slots.h, context.plan)
            scale_rows = {
                name: (
                    0.5 * (scale_a[name] + scale_b[name])
                    if name in {"h_norm_mean", "h_pairwise_sqdist_mean"}
                    else scale_a[name]
                )
                for name in scale_a
            }
            for name in ("plan_total_mass", "plan_max_cell_fraction"):
                scale_rows[name] = scale_rows[name].masked_fill(is_self, torch.nan)
            values_out.append(
                torch.stack(
                    [
                        active_logits.float(),
                        full_logits.float(),
                        f_logits.float(),
                        dispersion_rows["pi_slot_std"],
                        dispersion_rows["h_pairwise_cosine_mean"],
                        dispersion_rows["adj_offdiag_std"],
                        dispersion_rows["plan_row_entropy"],
                        dispersion_rows["plan_rank1_marginal_residual"],
                        scale_rows["plan_total_mass"],
                        scale_rows["plan_max_cell_fraction"],
                        scale_rows["h_norm_mean"],
                        scale_rows["h_pairwise_sqdist_mean"],
                    ],
                    dim=-1,
                )
            )
        synchronize_device()
        pair_scoring_seconds = time.monotonic() - pair_scoring_started
    n_cols = 12
    local_values = (
        torch.cat(values_out) if values_out else torch.zeros((0, n_cols), device=accelerator.device)
    )
    local_rows = torch.tensor(shard_rows, dtype=torch.long, device=accelerator.device)
    gather_metrics_started = time.monotonic()
    phase_timing_rows = accelerator.gather(
        torch.tensor(
            [node_cache_seconds, pair_scoring_seconds],
            dtype=torch.float64,
            device=accelerator.device,
        ).unsqueeze(0)
    ).reshape(world, 2)
    node_cache_seconds = float(phase_timing_rows[:, 0].max().item())
    pair_scoring_seconds = float(phase_timing_rows[:, 1].max().item())
    gathered_values = accelerator.gather(local_values)
    gathered_rows = accelerator.gather(local_rows)
    if was_training:
        model.train()
    if not accelerator.is_main_process:
        return None
    rows_np = gathered_rows.cpu().numpy()
    values_np = gathered_values.cpu().numpy()
    seen: dict[int, list[float]] = {}
    for row, values in zip(rows_np.tolist(), values_np.tolist(), strict=True):
        seen.setdefault(int(row), [float(value) for value in values])
    if len(seen) != n_val:
        raise RuntimeError(f"validation coverage broken: {len(seen)} of {n_val} rows scored")
    ordered = np.asarray([seen[i] for i in range(n_val)], dtype=np.float64)
    logits_np = ordered[:, 0]
    probs = 1.0 / (1.0 + np.exp(-logits_np))

    full_np = ordered[:, 1]
    f_np = ordered[:, 2]
    residual_std = float(np.std(full_np - f_np))
    f_std = float(np.std(f_np))
    f_probs = 1.0 / (1.0 + np.exp(-f_np))
    f_metrics = compute_edge_metrics(data.val_labels.astype(np.int64), f_probs)
    active_std = float(np.std(logits_np))
    dispersion_names = (
        "pi_slot_std",
        "h_pairwise_cosine_mean",
        "adj_offdiag_std",
        "plan_row_entropy",
        "plan_rank1_marginal_residual",
    )
    dispersion_summary = {
        name: (
            float(np.mean(values[np.isfinite(values)]))
            if bool(np.isfinite(values).any())
            else 0.0
        )
        for name, values in zip(dispersion_names, ordered[:, 3:8].T, strict=True)
    }
    scale_names = (
        "plan_total_mass",
        "plan_max_cell_fraction",
        "h_norm_mean",
        "h_pairwise_sqdist_mean",
    )
    scale_telemetry = {
        name: (
            float(np.mean(values[np.isfinite(values)]))
            if bool(np.isfinite(values).any())
            else float("nan")
        )
        for name, values in zip(scale_names, ordered[:, 8:12].T, strict=True)
    }
    endpoint_degree = _e2e_validation_endpoint_degrees(data)
    fidelity = {
        "active_logit_std": active_std,
        "f_logit_std": f_std,
        "f_logit_auprc": f_metrics.auprc,
        "topology_delta_std": residual_std,
        "topology_delta_ratio": residual_std / max(f_std, 1e-12),
        "selection_tiebreak": 0.0,
        "clustering_mmd": _validation_clustering_mmd(data, logits_np),
        "prevalence": float(np.mean(data.val_labels)),
        **dispersion_summary,
        **e2e_degree_decorrelation_telemetry(endpoint_degree, full_np - f_np),
    }
    active_metrics = compute_edge_metrics(data.val_labels.astype(np.int64), probs)
    gather_metrics_seconds = time.monotonic() - gather_metrics_started
    return _ValidationResult(
        metrics=active_metrics,
        fidelity=fidelity,
        scale_telemetry=scale_telemetry,
        timing={
            "node_cache_encode_seconds": node_cache_seconds,
            "pair_scoring_seconds": pair_scoring_seconds,
            "gather_metrics_seconds": gather_metrics_seconds,
        },
        active_logits=np.asarray(logits_np, dtype="<f4"),
    )


# --------------------------------------------------------------------------- training loop


@dataclass
class EgoTrainResult:
    """Finished-run bundle mirroring the train_b0 result contract."""

    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_metrics: EdgeMetrics
    last_state_dict: dict[str, torch.Tensor]
    last_epoch: int
    last_val_metrics: EdgeMetrics
    history: list[dict[str, object]]
    counterfactual_stop_epoch: int | None
    runtime_profile: dict[str, object]
    kendall_state: dict[str, object]


def _cpu_state_dict(accelerator: Accelerator, wrapped: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the *inner* EgoStitch model's state dict."""
    inner = accelerator.unwrap_model(wrapped)
    model = inner.model if isinstance(inner, _CompositeStep) else inner
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _write_failed_run_history(
    output_dir: Path,
    *,
    run_kind: str,
    arm: str,
    history: Sequence[Mapping[str, object]],
) -> None:
    """Retain per-epoch validation evidence when checkpoint selection fails.

    Best-effort by design: evidence retention must never mask the primary
    no-eligible-checkpoint failure, so I/O and serialization errors are
    logged and swallowed.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"run_kind": run_kind, "arm": arm, "history": list(history)}
        (output_dir / "failed_run_history.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as error:
        logger.error("failed to retain per-epoch failure history: %s", error)


def _e2e_arm_name(model: EgoStitchModel) -> E2EArmName:
    return _e2e_arm_name_from_config(model.cfg)


def _e2e_arm_name_from_config(config: E2EConfig) -> E2EArmName:
    if config.permanent_null == "all_head":
        return "b0_e2e_f_only"
    if config.p_topo == 0.0:
        return "p0"
    if config.w_rel == 0.0:
        return "no_l_rel"
    if config.feature_standardization == "row_layernorm":
        return "row_layernorm"
    return "full"


def _e2e_base_lr(step: int, total_steps: int, config: EgoStitchTrainingConfig) -> float:
    """Registered 500-step warm-up followed by cosine decay to ``min_lr``."""
    if not 0 <= step < total_steps:
        raise ValueError("E2E LR step is outside the registered schedule")
    if step < config.warmup_steps:
        return config.lr_peak * (step + 1) / config.warmup_steps
    decay_steps = max(total_steps - config.warmup_steps, 1)
    progress = min(1.0, (step + 1 - config.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.lr_peak - config.min_lr) * cosine


def _e2e_active_groups(phase: E2EPhaseState, model: EgoStitchModel) -> set[str]:
    groups = {"generator"}
    if model.encoder.rel_head is not None or phase.edge_active:
        groups.add("encoder")
    if phase.edge_active:
        groups.add("classifier")
    return groups


def _e2e_optimizer_group_lr(
    base_lr: float,
    phase: E2EPhaseState,
    group_name: object,
    active_groups: set[str],
) -> float:
    """Keep repair modules live in Phase A, then preserve the joint-entry ramp."""
    if group_name not in active_groups:
        return 0.0
    if group_name == "encoder" and phase.edge_active:
        return base_lr * phase.alpha
    return base_lr


def _e2e_training_payload(
    batch: _CompositeBatch,
    cfg: EgoConfig,
    phase: E2EPhaseState,
    *,
    epoch: int,
    step: int,
    total_steps: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        "node": _to_device(batch.node, device),
        "edge": _to_device(batch.edge, device),
        "edge_rows_global": batch.edge_rows_global,
        "edge_active": phase.edge_active,
        "recon_factors": e2e_recon_component_factors(step, total_steps),
        "real_ssl_scale": torch.tensor(phase.real_ssl_scale, device=device),
        "seed": cfg.seed,
        "epoch": epoch,
        "step": step,
    }


def _e2e_group_squared_norms(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    accelerator: Accelerator,
) -> dict[str, torch.Tensor]:
    gathered: dict[str, torch.Tensor] = {}
    for name, parameters in groups.items():
        terms = [
            parameter.grad.detach().double().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        local = torch.stack(terms).sum() if terms else torch.zeros((), device=accelerator.device)
        gathered[name] = accelerator.gather(local.reshape(1))
    return gathered


@dataclass
class _E2EFamilyRatioGuard:
    threshold: float
    required_probes: int
    streaks: dict[str, int] | None = None
    ratio_defined: dict[str, bool] | None = None

    def update(
        self,
        norms: Mapping[str, Mapping[str, float]],
        *,
        enabled: bool,
        enforce_quality: bool = True,
    ) -> dict[str, float]:
        if self.streaks is None:
            self.streaks = {}
        if self.ratio_defined is None:
            self.ratio_defined = {}
        ratios: dict[str, float] = {}
        for group, family_norms in norms.items():
            values = np.asarray(list(family_norms.values()), dtype=np.float64)
            if values.size < 2:
                self.streaks[group] = 0
                self.ratio_defined[group] = False
                continue
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite E2E family-gradient norm for group {group!r}")
            median = float(np.median(values))
            if median <= 0.0:
                self.ratio_defined[group] = False
                ratios[group] = 0.0
                self.streaks[group] = 0
                if not enforce_quality:
                    continue
                raise RuntimeError(f"invalid E2E family-gradient median for group {group!r}")
            self.ratio_defined[group] = True
            ratio = float(values.max() / median)
            ratios[group] = ratio
            if not enabled:
                self.streaks[group] = 0
                continue
            self.streaks[group] = self.streaks.get(group, 0) + 1 if ratio > self.threshold else 0
            if self.streaks[group] >= self.required_probes and enforce_quality:
                raise RuntimeError(f"persistent E2E family-gradient imbalance in group {group!r}")
        return ratios


def _e2e_family_probe(
    wrapped: torch.nn.Module,
    payload: dict[str, object],
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    phase: E2EPhaseState,
    arm: E2EArmName,
    accelerator: Accelerator,
    *,
    require_live_gradients: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Isolated synchronized family backwards on one immutable replay batch.

    `require_live_gradients` may only be cleared by a caller that measures
    allocation rather than reading the norms -- see `_probe_family_peak`. It
    relaxes the "every expected group received gradient" check and nothing else;
    finiteness stays enforced. A zero norm is a real dead-path signal during
    training, but at budget-probe time it is an artifact of step-0 init.
    """
    inner = cast(_CompositeStep, accelerator.unwrap_model(wrapped)).model
    assert isinstance(inner, EgoStitchModel)
    families = ["recon"]
    if phase.edge_active:
        families.insert(0, "edge")
    if phase.real_ssl_scale > 0.0:
        families.extend(("real", "ssl"))
    expected: dict[str, set[str]] = {
        "classifier": {"edge"} if phase.edge_active else set(),
        "generator": {"recon"} | ({"real", "ssl"} if phase.real_ssl_scale > 0.0 else set()),
        "encoder": {"recon"} if inner.encoder.rel_head is not None else set(),
    }
    if phase.edge_active and arm != "b0_e2e_f_only":
        expected["generator"].add("edge")
        expected["encoder"].add("edge")
    result: dict[str, dict[str, float]] = {group: {} for group in groups}
    submodule_rms: dict[str, float] = {}
    probe_payload = {**payload, "collect_diagnostics": True}
    ema_snapshot = [
        (buffer, buffer.detach().clone())
        for name, buffer in inner.named_buffers()
        if name.endswith(("ema_mu", "ema_updates"))
    ]
    try:
        for family in families:
            wrapped.zero_grad(set_to_none=True)
            probe_out = cast(dict[str, object], wrapped(probe_payload))
            family_loss = cast(dict[str, torch.Tensor], probe_out["families"])[family]
            # Release the forward's other outputs before the backward. The
            # `families` dict holds one loss tensor per family, and each pins the
            # autograd graph reachable from it: keeping the dict alive keeps the
            # three *unselected* families' subgraphs (for an "edge" backward,
            # that is the whole generator node stream) resident for the rest of
            # the loop, and keeps this entire forward resident across the *next*
            # family's forward. That double-buffering is the probe's real memory
            # cost, and it is pure waste -- nothing here reads the other
            # families, and each family already gets its own fresh forward, so no
            # gradient value depends on when these references are dropped.
            del probe_out
            accelerator.backward(family_loss)
            del family_loss
            gathered = _e2e_group_squared_norms(groups, accelerator)
            e2e_assert_replicated_squared_norms(gathered)
            if family == "edge":
                submodule_rms = _e2e_current_submodule_gradient_rms(inner, accelerator)
            for group, family_names in expected.items():
                if family not in family_names:
                    continue
                norm = float(torch.sqrt(gathered[group].double().mean()).item())
                if not math.isfinite(norm) or (require_live_gradients and norm <= 0.0):
                    raise RuntimeError(
                        f"invalid E2E fixed-replay norm for family {family!r}, group {group!r}"
                    )
                result[group][family] = norm
    finally:
        with torch.no_grad():
            for buffer, value in ema_snapshot:
                buffer.copy_(value)
        wrapped.zero_grad(set_to_none=True)
    return result, submodule_rms


def _e2e_current_submodule_gradient_rms(
    model: EgoStitchModel, accelerator: Accelerator
) -> dict[str, float]:
    """RMS telemetry from the current synchronized fixed-replay edge backward."""
    submodules: dict[str, Sequence[torch.nn.Parameter]] = {
        "grad_rms_trunk": tuple(model.classifier.trunk.parameters()),
        "grad_rms_ste": tuple(model.encoder.parameters()),
    }
    result: dict[str, float] = {}
    for name, parameters in submodules.items():
        terms = [
            parameter.grad.detach().double().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        count = sum(
            parameter.grad.numel() for parameter in parameters if parameter.grad is not None
        )
        local = torch.stack(terms).sum() if terms else torch.zeros((), device=accelerator.device)
        gathered = accelerator.gather(local.reshape(1)).double()
        if not bool(torch.isfinite(gathered).all()):
            raise RuntimeError(f"non-finite E2E fixed-replay submodule RMS for {name}")
        reference = gathered[0].expand_as(gathered)
        if not bool(torch.allclose(gathered, reference, rtol=1e-7, atol=1e-12)):
            raise RuntimeError(f"DDP fixed-replay submodule gradients differ for {name}")
        value = float(torch.sqrt(gathered.mean() / max(count, 1)).item())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite E2E fixed-replay submodule RMS for {name}")
        result[name] = value
    return result


class _E2EPrecisionNumericalError(RuntimeError):
    """Non-finite precision input or derived numerical state."""


class _E2EPrecisionQualityError(RuntimeError):
    """Finite precision-differential quality-threshold miss."""


def _validate_e2e_precision_outputs(
    mixed_full: torch.Tensor,
    mixed_f: torch.Tensor,
    fp32_full: torch.Tensor,
    fp32_f: torch.Tensor,
    *,
    enforce_quality: bool = True,
) -> dict[str, object]:
    """Measure precision and optionally enforce finite quality thresholds."""
    tensors = {
        "mixed_full": mixed_full,
        "mixed_f": mixed_f,
        "fp32_full": fp32_full,
        "fp32_f": fp32_f,
    }
    nonfinite_inputs = [
        name for name, value in tensors.items() if not bool(torch.isfinite(value).all())
    ]
    if nonfinite_inputs:
        raise _E2EPrecisionNumericalError(
            f"non-finite E2E precision input tensors: {', '.join(nonfinite_inputs)}"
        )
    mixed_residual = mixed_full - mixed_f
    fp32_residual = fp32_full - fp32_f
    residual_relative_l2 = float(
        torch.linalg.vector_norm(mixed_residual - fp32_residual)
        / torch.clamp(torch.linalg.vector_norm(fp32_residual), min=1e-12)
    )
    mixed_residual_np = mixed_residual.detach().double().cpu().numpy()
    fp32_residual_np = fp32_residual.detach().double().cpu().numpy()
    mixed_std = float(np.std(mixed_residual_np))
    fp32_std = float(np.std(fp32_residual_np))
    if not all(math.isfinite(value) for value in (mixed_std, fp32_std)):
        raise _E2EPrecisionNumericalError("non-finite E2E precision residual standard deviation")
    residual_correlation_defined = (
        mixed_residual_np.size >= 2 and mixed_std > 0.0 and fp32_std > 0.0
    )
    residual_correlation = (
        float(np.corrcoef(mixed_residual_np, fp32_residual_np)[0, 1])
        if residual_correlation_defined
        else 0.0
    )
    full_relative_l2 = float(
        torch.linalg.vector_norm(mixed_full - fp32_full)
        / torch.clamp(torch.linalg.vector_norm(fp32_full), min=1e-12)
    )
    f_logit_relative_l2 = float(
        torch.linalg.vector_norm(mixed_f - fp32_f)
        / torch.clamp(torch.linalg.vector_norm(fp32_f), min=1e-12)
    )
    numerical_metrics = {
        "full_max_abs_error": float(torch.max(torch.abs(mixed_full - fp32_full))),
        "f_logit_max_abs_error": float(torch.max(torch.abs(mixed_f - fp32_f))),
        "full_relative_l2": full_relative_l2,
        "f_logit_relative_l2": f_logit_relative_l2,
        "residual_relative_l2": residual_relative_l2,
        "residual_correlation": residual_correlation,
    }
    nonfinite_derived = [
        name for name, value in numerical_metrics.items() if not math.isfinite(value)
    ]
    if nonfinite_derived:
        raise _E2EPrecisionNumericalError(
            f"non-finite E2E precision derived metrics: {', '.join(nonfinite_derived)}"
        )
    failures: list[str] = []
    # Vector relative-L2 bounds (registration vector_tolerance_amendment
    # 2026-07-22): per-element max-abs is an extreme-value statistic of
    # BF16-trunk noise and stays a logged diagnostic only.
    if full_relative_l2 > 5e-2:
        failures.append("full relative L2 <= 0.05")
    if f_logit_relative_l2 > 5e-2:
        failures.append("f_logit relative L2 <= 0.05")
    residual_nonzero = bool((mixed_residual != 0).any()) and bool((fp32_residual != 0).any())
    if not residual_nonzero:
        failures.append("non-zero residual")
    if residual_relative_l2 > 5e-2:
        failures.append("residual relative L2 <= 0.05")
    if not residual_correlation_defined:
        failures.append("residual correlation defined")
    elif residual_correlation < 0.999:
        failures.append("residual correlation >= 0.999")
    metrics: dict[str, object] = {
        **numerical_metrics,
        "residual_nonzero": residual_nonzero,
        "residual_correlation_defined": residual_correlation_defined,
        "quality_pass": not failures,
        "quality_failures": failures,
    }
    if failures and enforce_quality:
        raise _E2EPrecisionQualityError(
            "E2E precision differential failed: "
            f"{', '.join(failures)}; metrics={json.dumps(metrics, sort_keys=True)}"
        )
    return metrics


def _e2e_precision_differential(
    model: EgoStitchModel,
    edge: dict[str, torch.Tensor],
    accelerator: Accelerator,
    *,
    enforce_quality: bool = True,
) -> dict[str, object]:
    """Compare BF16+islands with pure fp32 on the same fixed replay identities."""
    batch = _e2e_edge_view(edge)
    float_batch = {
        name: value.float() if value.is_floating_point() else value for name, value in batch.items()
    }
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), accelerator.autocast():
            mixed_context = model.build_pair_context(batch)
            mixed_full = model.score_pair_context(mixed_context).float()
            mixed_f = model.score_pair_context(
                mixed_context,
                masks=masks_for_null(NULL_ALL_HEAD, mixed_full.shape[0], mixed_full.device),
            ).float()
        with torch.no_grad(), torch.autocast(device_type=accelerator.device.type, enabled=False):
            fp32_context = model.build_pair_context(float_batch)
            fp32_full = model.score_pair_context(fp32_context).float()
            fp32_f = model.score_pair_context(
                fp32_context,
                masks=masks_for_null(NULL_ALL_HEAD, fp32_full.shape[0], fp32_full.device),
            ).float()
        return _validate_e2e_precision_outputs(
            mixed_full,
            mixed_f,
            fp32_full,
            fp32_f,
            enforce_quality=enforce_quality,
        )
    finally:
        model.train(was_training)


def _raise_synchronized_precision_failure(
    accelerator: Accelerator,
    *,
    local_error: Exception | None,
    context: str,
) -> None:
    """Synchronize any rank-local precision exception before raising everywhere."""
    failed = accelerator.reduce(
        torch.tensor(int(local_error is not None), device=accelerator.device), reduction="sum"
    )
    if int(failed.item()) <= 0:
        return
    failure = RuntimeError(f"{context} E2E precision differential failed")
    if local_error is not None:
        raise failure from local_error
    raise failure


def _enforce_e2e_initial_slot_health(
    model: EgoStitchModel,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    edge_batch: int,
    topk_fraction: float,
    token_table: PackedFeatureTable | None,
    token_node_index: Mapping[str, int] | None,
    validation_event_callback: Callable[[str, int | None, int], None] | None = None,
    enforce_quality: bool = True,
) -> dict[str, float]:
    """Refuse to start a run that is already above the cosine trip line.

    Spec Sec 13.19.1 landed a preprocessing change whose whole claim is that a
    random model is born healthy. This is where that claim is *enforced*, at
    step 0, before any GPU time is spent on a schedule that cannot survive it:
    the retired ``--ddp-mode init-probe`` measured exactly this and then only
    logged, which is how the 2026-07-27 run trained for fifteen minutes before
    dying of a condition present at initialization.

    Only the ``h_pairwise_cosine_mean`` half of `E2ESlotCollapseGuard` is
    checkable here. That guard also needs two consecutive validations and
    short-circuits on ``conditioning_active``, which is ``False`` throughout
    Phase A (`e2e_phase_state(0, N)`), so it cannot be moved to step 0 and
    stays a during-training guard.

    Args:
        model: The prepared, freshly initialized E2E model (unwrapped, per the
            ``[P2]`` convention every other validation call in this module
            follows).
        data: The assembled training data.
        accelerator: The live accelerator.
        edge_batch: Validation pair batch size.
        topk_fraction: The registered top-k fidelity fraction.
        token_table: The raw-token store the validation batches need.
        token_node_index: Its node index.
        validation_event_callback: Records the completed step-0 validation before
            any resulting guard failure is raised.
        enforce_quality: Whether a finite threshold miss aborts the run.

    Returns:
        The guard telemetry plus scale diagnostics on the main process (an
        empty dict on the others).

    Raises:
        RuntimeError: When there are no validation pairs to measure, or the
            initial ``h_pairwise_cosine_mean`` is above the Sec 14.4.8 trip
            line. Raised on *every* rank -- `_validate_epoch` returns ``None``
            off the main process, so a rank-0-only raise is a DDP hang.
    """
    if not data.val_pairs:
        raise RuntimeError(
            "step-0 slot guard has an empty population: this config gives "
            "_validate_epoch no validation pairs, so neither this guard nor "
            "the during-training slot-collapse guard would ever evaluate"
        )
    validation = _validate_epoch(
        model,
        data,
        accelerator,
        edge_batch=edge_batch,
        topk_fraction=topk_fraction,
        token_table=token_table,
        token_node_index=token_node_index,
    )
    if validation_event_callback is not None:
        validation_event_callback("step_0", None, 0)
    report: dict[str, float] = {}
    born_collapsed = 0
    nonfinite_telemetry = 0
    if accelerator.is_main_process:
        assert validation is not None
        report = {
            "h_pairwise_cosine_mean": validation.fidelity["h_pairwise_cosine_mean"],
            "plan_rank1_marginal_residual": validation.fidelity["plan_rank1_marginal_residual"],
            "pi_slot_std": validation.fidelity["pi_slot_std"],
            "adj_offdiag_std": validation.fidelity["adj_offdiag_std"],
            **validation.scale_telemetry,
        }
        if not all(math.isfinite(value) for value in report.values()):
            nonfinite_telemetry = 1
        report["quality_threshold_missed"] = float(
            report["h_pairwise_cosine_mean"] > 0.95
        )
        logger.info(
            "e2e step-0 slot health rows=%d feature_stats_sha256=%s %s",
            len(data.val_pairs),
            model.feature_stats_digest_hex,
            " ".join(f"{name}={value:.6g}" for name, value in sorted(report.items())),
        )
        if report["h_pairwise_cosine_mean"] > 0.95:
            logger.error(
                "step-0 slot health is above the Sec 14.4.8 cosine trip line "
                "(%.6f > 0.95): the run is born collapsed",
                report["h_pairwise_cosine_mean"],
            )
            born_collapsed = 1
    nonfinite = accelerator.reduce(
        torch.tensor(nonfinite_telemetry, device=accelerator.device), reduction="sum"
    )
    if int(nonfinite.item()) > 0:
        raise RuntimeError("non-finite E2E step-0 slot-health telemetry")
    failed = accelerator.reduce(
        torch.tensor(born_collapsed, device=accelerator.device), reduction="sum"
    )
    if int(failed.item()) > 0 and enforce_quality:
        raise RuntimeError("training_invalid(initial_slot_collapse)")
    return report


def _train_e2e_stability_loop(
    model: EgoStitchModel,
    cfg: EgoConfig,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    node_batch: int,
    profile_only: bool = False,
) -> EgoTrainResult:
    """Execute the rev-3.1 §14.4 curriculum, death guards, and selector."""
    training = cfg.training
    if training is None:
        raise ValueError("E2E stability training requires cfg.training")
    run_kind = cfg.run_kind or "formal"
    # Model-quality thresholds are telemetry-only. Truthfulness failures such
    # as non-finite tensors, DDP disagreement, coverage, and boundary errors
    # remain enforced at their source.
    enforce_quality = False
    arm = _e2e_arm_name(model)
    world = accelerator.num_processes
    rank = accelerator.process_index
    use_cuda = accelerator.device.type == "cuda"

    parameter_groups = build_e2e_parameter_groups(model)
    composite = _CompositeStep(model, world)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameter_groups.groups[name], "lr": training.lr_peak, "name": name}
            for name in (
                "generator",
                "encoder",
                "classifier",
            )
        ],
        betas=training.betas,
        eps=training.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    wrapped, optimizer = accelerator.prepare(composite, optimizer)
    factory = _BatchFactory(
        cfg,
        model.generator_cfg,
        data,
        node_batch=node_batch,
        rank=rank,
        world_size=world,
    )

    validation_events_path = cfg.output_dir / V_HOLD_VALIDATION_EVENTS_FILENAME
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if validation_events_path.exists():
            raise FileExistsError(
                f"V_hold validation-event ledger already exists: {validation_events_path}"
            )
        validation_events_path.touch()
    accelerator.wait_for_everyone()
    validation_events: list[dict[str, object]] = []

    def record_validation_event(kind: str, epoch: int | None, optimizer_step: int) -> None:
        event: dict[str, object] = {
            "ordinal": len(validation_events) + 1,
            "kind": kind,
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "run_kind": run_kind,
            "arm": arm,
            "validation_role": data.validation_role,
        }
        validation_events.append(event)
        if accelerator.is_main_process:
            with validation_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            metadata_path = cfg.output_dir / "run_metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["v_hold_validation_evidence"] = {
                    "schema": "egostitch_e2e_v_hold_validation_events_v1",
                    "count": len(validation_events),
                    "path": V_HOLD_VALIDATION_EVENTS_FILENAME,
                    "sha256": _sha256_file(validation_events_path),
                }
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # [C] Step-0 death guard: the run must not be *born* above the Sec 14.4.8
    # cosine trip line. Placed here so the model is on `accelerator.device`
    # (`accelerator.prepare` moves it in place) and the factory's token table
    # exists, and run before the first optimizer step so a collapsed
    # initialization costs one validation pass instead of a whole schedule.
    #
    # It is deliberately not gated on `profile_only`: a guard a flag can skip
    # is a guard that fails open. The cost is one extra full validation pass
    # over C(|V_hold|, 2) rows per run, charged before the peak-memory counter
    # is reset below and therefore outside the measured training peak.
    initial_slot_health = _enforce_e2e_initial_slot_health(
        model,
        data,
        accelerator,
        edge_batch=cfg.data.edge_batch,
        topk_fraction=cfg.diagnostics.topk_fraction,
        token_table=factory._token_table,
        token_node_index=factory._token_node_index,
        validation_event_callback=record_validation_event,
        enforce_quality=enforce_quality,
    )

    rows_per_rank, steps_per_epoch = _epoch_step_plan(
        len(data.e_sup_positives),
        negative_ratio=cfg.data.negative_ratio,
        edge_batch=cfg.data.edge_batch,
        world_size=world,
    )
    production_epoch_step_counts = [steps_per_epoch] * cfg.optim.epochs
    epoch_step_counts = (
        production_epoch_step_counts[:1] if profile_only else production_epoch_step_counts
    )
    schedule_total_steps = steps_per_epoch * cfg.optim.epochs
    executed_steps = sum(epoch_step_counts)
    phase_a_end, phase_b_end = e2e_phase_boundaries(schedule_total_steps)
    first_eligible_epoch = e2e_first_eligible_epoch(schedule_total_steps, steps_per_epoch)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    clip_guard = E2EClipGuard(
        immediate_threshold=training.clip_immediate_abort,
        persistent_threshold=training.clip_persistent_threshold,
        persistent_steps=training.clip_persistent_steps,
    )
    ratio_guard = _E2EFamilyRatioGuard(
        threshold=training.family_ratio_abort,
        required_probes=training.family_ratio_probes,
    )
    history: list[dict[str, object]] = []
    records: list[E2ECheckpointRecord] = []
    metrics_by_epoch: dict[int, EdgeMetrics] = {}
    state_paths: dict[int, Path] = {}
    checkpoint_dir = cfg.output_dir / ".eligible_checkpoints"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    warm_reference_std: float | None = None
    warm_reference_auprc: float | None = None
    collapse_streak = 0
    slot_collapse_guard = E2ESlotCollapseGuard()
    last_metrics: EdgeMetrics | None = None
    last_fidelity: dict[str, float] | None = None
    fixed_replay: dict[str, object] | None = None
    end_ramp_precision: dict[str, object] | None = None
    selected_precision: dict[str, object] | None = None
    latest_topology_norm: float | None = None
    gradient_norm_series: list[dict[str, object]] = []
    optimizer_step_gradients: list[dict[str, object]] = []
    quality_guard_events: list[dict[str, object]] = []
    quality_guards_passed = not bool(initial_slot_health.get("quality_threshold_missed", 0.0))
    if accelerator.is_main_process and not quality_guards_passed:
        quality_guard_events.append(
            {"kind": "initial_slot_collapse", "optimizer_step": 0, **initial_slot_health}
        )
    warm_reference_quality_pass: bool | None = None
    per_epoch_profiles: list[dict[str, object]] = []
    total_local_pairs = 0
    total_local_tokens = 0
    total_wall = 0.0
    total_data_wait = 0.0
    total_validation_seconds = 0.0
    total_validation_timing = {
        "node_cache_encode_seconds": 0.0,
        "pair_scoring_seconds": 0.0,
        "gather_metrics_seconds": 0.0,
    }
    global_step = 0
    prefetch_depth = cfg.runtime.prefetch_factor if cfg.runtime is not None else 1

    for epoch, epoch_steps in enumerate(epoch_step_counts, start=1):
        epoch_started = time.monotonic()
        epoch_data_wait = 0.0
        epoch_local_pairs = 0
        epoch_local_tokens = 0
        epoch_global_pairs = 0
        epoch_parts: dict[str, float] = {}
        epoch_probes: list[dict[str, object]] = []
        epoch_validation_seconds = 0.0
        epoch_validation_timing = dict.fromkeys(total_validation_timing, 0.0)
        batch_source = iter(
            factory.epoch_batches(epoch, rows_per_rank=rows_per_rank, steps=epoch_steps)
        )
        batches = _prefetch_batches(batch_source, depth=prefetch_depth)
        try:
            for _step_in_epoch in range(epoch_steps):
                fetch_started = time.monotonic()
                batch = next(batches)
                epoch_data_wait += time.monotonic() - fetch_started
                phase = e2e_phase_state(global_step, schedule_total_steps)
                active_groups = _e2e_active_groups(phase, model)
                base_lr = _e2e_base_lr(global_step, schedule_total_steps, training)
                for group in optimizer.param_groups:
                    group["lr"] = _e2e_optimizer_group_lr(
                        base_lr,
                        phase,
                        group.get("name"),
                        active_groups,
                    )
                payload = _e2e_training_payload(
                    batch,
                    cfg,
                    phase,
                    epoch=epoch,
                    step=global_step,
                    total_steps=schedule_total_steps,
                    device=accelerator.device,
                )
                if fixed_replay is None:
                    fixed_replay = cast(dict[str, object], _detached_clone(payload))
                optimizer.zero_grad(set_to_none=True)
                out = cast(dict[str, object], wrapped(payload))
                loss = cast(torch.Tensor, out["loss"])
                local_bad = not bool(torch.isfinite(loss).all())
                bad_ranks = accelerator.reduce(
                    torch.tensor(int(local_bad), device=accelerator.device), reduction="sum"
                )
                if int(bad_ranks.item()) > 0:
                    raise RuntimeError(f"non-finite E2E loss at optimizer step {global_step}")
                accelerator.backward(loss)
                gathered_squared = _e2e_group_squared_norms(parameter_groups.groups, accelerator)
                e2e_assert_replicated_squared_norms(gathered_squared)
                gradient_records = e2e_check_and_clip_gradients(
                    parameter_groups.groups,
                    active_groups,
                    max_norm={
                        "classifier": training.pair_encoder_clip_norm,
                        "generator": training.generator_clip_norm,
                        "encoder": training.clip_norm,
                    },
                    enforce_nonzero=enforce_quality,
                )
                gradient_row: dict[str, object] = {
                    "step": global_step + 1,
                    "phase": phase.phase,
                    "alpha": phase.alpha,
                    "optimizer_group_gradients": {
                        name: asdict(record) for name, record in gradient_records.items()
                    },
                }
                if accelerator.is_main_process:
                    optimizer_step_gradients.append(gradient_row)
                clip_guard.update(
                    gradient_records,
                    step=global_step + 1,
                    phase=phase.phase,
                    enforce_immediate=enforce_quality,
                    enforce_persistent=(
                        not profile_only
                        and enforce_quality
                    ),
                )
                gradient_quality: dict[str, dict[str, bool]] = {}
                for name, gradient_record in gradient_records.items():
                    if not gradient_record.active:
                        gradient_quality[name] = {
                            "finite_zero_norm": False,
                            "immediate_clip_threshold_missed": False,
                            "persistent_clip_threshold_missed": False,
                        }
                        continue
                    coefficient = cast(float, gradient_record.clip_coefficient)
                    flags = {
                        "finite_zero_norm": gradient_record.norm == 0.0,
                        "immediate_clip_threshold_missed": (
                            coefficient < training.clip_immediate_abort
                        ),
                        "persistent_clip_threshold_missed": (
                            clip_guard.streaks is not None
                            and clip_guard.streaks.get(name, 0) >= training.clip_persistent_steps
                        ),
                    }
                    gradient_quality[name] = flags
                    if any(flags.values()):
                        quality_guards_passed = False
                        if accelerator.is_main_process:
                            quality_guard_events.append(
                                {
                                    "kind": "optimizer_gradient",
                                    "step": global_step + 1,
                                    "phase": phase.phase,
                                    "group": name,
                                    **flags,
                                }
                            )
                gradient_row["quality_thresholds"] = gradient_quality
                optimizer.step()
                post_step_failure = 0
                try:
                    e2e_assert_finite_optimizer_state(parameter_groups.groups, optimizer)
                except RuntimeError:
                    post_step_failure = 1
                failed = accelerator.reduce(
                    torch.tensor(post_step_failure, device=accelerator.device), reduction="sum"
                )
                if int(failed.item()) > 0:
                    raise RuntimeError("non-finite E2E parameter or optimizer state after step")
                epoch_parts = cast(dict[str, float], out["parts"])
                global_step += 1

                # `parts` are plain floats (`stage1_total`), so nothing below
                # reads this step's forward output, its loss, or its device
                # payload. Dropping them here matters because the family probe
                # runs up to four more full forward/backward passes before the
                # next step rebinds these names -- holding a spare training batch
                # and the step's output handles across those passes is memory the
                # probe cannot use.
                del out, loss, gathered_squared, payload

                if not profile_only and global_step % cfg.diagnostics.gradient_probe_interval == 0:
                    assert fixed_replay is not None
                    probe_payload = cast(dict[str, object], _detached_clone(fixed_replay))
                    probe_payload["edge_active"] = phase.edge_active
                    probe_payload["recon_factors"] = e2e_recon_component_factors(
                        global_step - 1, schedule_total_steps
                    )
                    probe_payload["real_ssl_scale"] = torch.tensor(
                        phase.real_ssl_scale, device=accelerator.device
                    )
                    family_norms, submodule_rms = _e2e_family_probe(
                        wrapped,
                        probe_payload,
                        parameter_groups.groups,
                        phase,
                        arm,
                        accelerator,
                        require_live_gradients=enforce_quality,
                    )
                    ratios = ratio_guard.update(
                        family_norms,
                        enabled=phase.alpha == 1.0,
                        enforce_quality=enforce_quality,
                    )
                    ratio_defined = dict(ratio_guard.ratio_defined or {})
                    family_quality_misses = {
                        group: {
                            "finite_zero_norm": any(value == 0.0 for value in values.values()),
                            "ratio_defined": ratio_defined.get(group, False),
                            "ratio_threshold_missed": (
                                ratio_defined.get(group, False)
                                and ratios.get(group, 0.0) > training.family_ratio_abort
                            ),
                            "persistent_ratio_threshold_missed": (
                                ratio_guard.streaks is not None
                                and ratio_guard.streaks.get(group, 0)
                                >= training.family_ratio_probes
                            ),
                        }
                        for group, values in family_norms.items()
                    }
                    if any(
                        flags["finite_zero_norm"]
                        or (len(family_norms[group]) >= 2 and not flags["ratio_defined"])
                        or flags["ratio_threshold_missed"]
                        or flags["persistent_ratio_threshold_missed"]
                        for group, flags in family_quality_misses.items()
                    ):
                        quality_guards_passed = False
                    latest_topology_norm = family_norms["encoder"].get("edge")
                    probe_record: dict[str, object] = {
                        "step": global_step,
                        "phase": phase.phase,
                        "alpha": phase.alpha,
                        "optimizer_group_gradients": {
                            name: asdict(record) for name, record in gradient_records.items()
                        },
                        "family_group_norms": family_norms,
                        "family_group_ratios": ratios,
                        "family_group_ratio_defined": ratio_defined,
                        "family_quality_thresholds": family_quality_misses,
                        "submodule_gradient_rms": submodule_rms,
                        **_e2e_gate_tanh(
                            cast(EgoStitchModel, accelerator.unwrap_model(wrapped).model)
                        ),
                    }
                    epoch_probes.append(probe_record)
                    gradient_norm_series.append(probe_record)
                    if accelerator.is_main_process:
                        for group, flags in family_quality_misses.items():
                            if (
                                flags["finite_zero_norm"]
                                or (
                                    len(family_norms[group]) >= 2
                                    and not flags["ratio_defined"]
                                )
                                or flags["ratio_threshold_missed"]
                                or flags["persistent_ratio_threshold_missed"]
                            ):
                                quality_guard_events.append(
                                    {
                                        "kind": "family_gradient",
                                        "step": global_step,
                                        "phase": phase.phase,
                                        "group": group,
                                        **flags,
                                    }
                                )

                if not profile_only and global_step == phase_a_end:
                    phase_a_validation_started = time.monotonic()
                    warm = _validate_epoch(
                        model,
                        data,
                        accelerator,
                        edge_batch=cfg.data.edge_batch,
                        topk_fraction=cfg.diagnostics.topk_fraction,
                        token_table=factory._token_table,
                        token_node_index=factory._token_node_index,
                    )
                    epoch_validation_seconds += time.monotonic() - phase_a_validation_started
                    if warm is not None:
                        for name in epoch_validation_timing:
                            epoch_validation_timing[name] += warm.timing.get(name, 0.0)
                    record_validation_event("phase_a_end", epoch, global_step)
                    warm_failure = 0
                    warm_floor_failure = 0
                    if accelerator.is_main_process:
                        assert warm is not None
                        warm_reference_std = warm.fidelity["f_logit_std"]
                        if not math.isfinite(warm_reference_std):
                            warm_failure = 1
                        else:
                            warm_reference_quality_pass = warm_reference_std >= 1e-4
                            if not warm_reference_quality_pass:
                                warm_floor_failure = 1
                                quality_guards_passed = False
                                quality_guard_events.append(
                                    {
                                        "kind": "warm_reference_std",
                                        "step": global_step,
                                        "value": warm_reference_std,
                                        "floor": 1e-4,
                                    }
                                )
                    failed = accelerator.reduce(
                        torch.tensor(warm_failure, device=accelerator.device), reduction="sum"
                    )
                    if int(failed.item()) > 0:
                        raise RuntimeError("invalid E2E warm-reference logit standard deviation")
                    failed = accelerator.reduce(
                        torch.tensor(warm_floor_failure, device=accelerator.device),
                        reduction="sum",
                    )
                    if int(failed.item()) > 0 and enforce_quality:
                        raise RuntimeError("invalid E2E warm-reference logit standard deviation")
                if not profile_only and global_step == phase_b_end and arm == "full":
                    assert fixed_replay is not None
                    precision_error: Exception | None = None
                    if accelerator.is_main_process:
                        try:
                            inner_model = cast(
                                _CompositeStep, accelerator.unwrap_model(wrapped)
                            ).model
                            assert isinstance(inner_model, EgoStitchModel)
                            end_ramp_precision = _e2e_precision_differential(
                                inner_model,
                                cast(dict[str, torch.Tensor], fixed_replay["edge"]),
                                accelerator,
                                enforce_quality=enforce_quality,
                            )
                            if not bool(end_ramp_precision["quality_pass"]):
                                quality_guards_passed = False
                                quality_guard_events.append(
                                    {
                                        "kind": "end_ramp_precision",
                                        "step": global_step,
                                        "quality_failures": end_ramp_precision[
                                            "quality_failures"
                                        ],
                                    }
                                )
                        except Exception as error:
                            precision_error = error
                            logger.error(
                                "end-ramp precision differential: %s: %s",
                                type(error).__name__,
                                error,
                            )
                    _raise_synchronized_precision_failure(
                        accelerator,
                        local_error=precision_error,
                        context="end-ramp",
                    )

                epoch_local_pairs += batch.edge_rows_true
                epoch_local_tokens += batch.f0_rows_gathered
                epoch_global_pairs += batch.edge_rows_global
        finally:
            batches.close()

        validation_started = time.monotonic()
        validation = _validate_epoch(
            model,
            data,
            accelerator,
            edge_batch=cfg.data.edge_batch,
            topk_fraction=cfg.diagnostics.topk_fraction,
            token_table=factory._token_table,
            token_node_index=factory._token_node_index,
        )
        epoch_validation_seconds += time.monotonic() - validation_started
        if validation is not None:
            for name in epoch_validation_timing:
                epoch_validation_timing[name] += validation.timing.get(name, 0.0)
        record_validation_event("epoch_end", epoch, global_step)
        validation_seconds = epoch_validation_seconds
        epoch_wall = time.monotonic() - epoch_started
        phase = e2e_phase_state(global_step - 1, schedule_total_steps)
        collapse_failure = 0
        slot_collapse_failure = 0
        validation_nonfinite_failure = 0
        if accelerator.is_main_process:
            assert validation is not None
            metrics = validation.metrics
            fidelity = validation.fidelity
            last_metrics = metrics
            last_fidelity = fidelity
            full_joint_epochs = max(0, epoch - first_eligible_epoch + 1)
            validation_quality_values = {
                "auprc": metrics.auprc,
                "brier": metrics.brier,
                "prevalence": fidelity["prevalence"],
                "active_logit_std": fidelity["active_logit_std"],
                "clustering_mmd": fidelity["clustering_mmd"],
                "f_logit_std": fidelity["f_logit_std"],
                "f_logit_auprc": fidelity["f_logit_auprc"],
                "h_pairwise_cosine_mean": fidelity["h_pairwise_cosine_mean"],
                "plan_rank1_marginal_residual": fidelity[
                    "plan_rank1_marginal_residual"
                ],
            }
            validation_nonfinite_failure = int(
                not all(math.isfinite(value) for value in validation_quality_values.values())
            )
            if (
                not profile_only
                and _e2e_should_capture_eligibility_reference(
                    phase,
                    warm_reference_auprc=warm_reference_auprc,
                )
            ):
                warm_reference_auprc = fidelity["f_logit_auprc"]
            if warm_reference_std is not None and not validation_nonfinite_failure:
                threshold = max(
                    training.collapse_fraction * warm_reference_std,
                    training.collapse_floor,
                )
                collapse_streak = collapse_streak + 1 if fidelity["f_logit_std"] < threshold else 0
                if collapse_streak >= training.collapse_validations:
                    collapse_failure = 1
            if not validation_nonfinite_failure:
                slot_collapse_failure = int(
                    slot_collapse_guard.update(
                        fidelity,
                        conditioning_active=phase.edge_active and arm != "b0_e2e_f_only",
                        enforce_quality=False,
                    )
                )
            if collapse_failure:
                quality_guards_passed = False
                quality_guard_events.append(
                    {
                        "kind": "validation_logit_collapse",
                        "epoch": epoch,
                        "step": global_step,
                        "streak": collapse_streak,
                        "value": fidelity["f_logit_std"],
                        "threshold": threshold,
                    }
                )
            if slot_collapse_failure:
                quality_guards_passed = False
                quality_guard_events.append(
                    {
                        "kind": "slot_collapse",
                        "epoch": epoch,
                        "step": global_step,
                        "streak": slot_collapse_guard.streak,
                        "h_pairwise_cosine_mean": fidelity["h_pairwise_cosine_mean"],
                        "plan_rank1_marginal_residual": fidelity[
                            "plan_rank1_marginal_residual"
                        ],
                    }
                )
            # `history` is written only when the run completes, and the
            # registered `training_invalid(slot_collapse)` label carries no
            # numbers -- so a collapse otherwise leaves no record of which
            # condition fired. Degenerate slots (`h_pairwise_cosine_mean`) and a
            # rank-1 transport plan (`plan_rank1_marginal_residual`) are
            # different failures with different fixes, and the guard needs two
            # consecutive validations to trip, so the trajectory matters as much
            # as the final value.
            logger.log(
                logging.ERROR if slot_collapse_failure else logging.INFO,
                "e2e slot telemetry epoch=%d h_pairwise_cosine_mean=%.6f "
                "plan_rank1_marginal_residual=%.6f streak=%d plan_total_mass=%.6g "
                "plan_max_cell_fraction=%.6f h_norm_mean=%.4f h_pairwise_sqdist_mean=%.4f",
                epoch,
                fidelity.get("h_pairwise_cosine_mean", float("nan")),
                fidelity.get("plan_rank1_marginal_residual", float("nan")),
                slot_collapse_guard.streak,
                validation.scale_telemetry.get("plan_total_mass", float("nan")),
                validation.scale_telemetry.get("plan_max_cell_fraction", float("nan")),
                validation.scale_telemetry.get("h_norm_mean", float("nan")),
                validation.scale_telemetry.get("h_pairwise_sqdist_mean", float("nan")),
            )
            record = E2ECheckpointRecord(
                epoch=epoch,
                phase=phase.phase,
                full_joint_epochs_completed=full_joint_epochs,
                guards_passed=quality_guards_passed,
                auprc=metrics.auprc,
                prevalence=fidelity["prevalence"],
                active_logit_std=fidelity["active_logit_std"],
                clustering_mmd=fidelity["clustering_mmd"],
                brier=metrics.brier,
                warm_reference_std=warm_reference_std,
                warm_reference_auprc=warm_reference_auprc,
                residual_ratio=fidelity["topology_delta_ratio"],
                topology_gradient_norm=latest_topology_norm,
            )
            records.append(record)
            metrics_by_epoch[epoch] = metrics
            # Snapshot every completed epoch. Quality predicates remain in the
            # history as telemetry but do not control checkpoint availability.
            if not profile_only:
                path = checkpoint_dir / f"epoch-{epoch:03d}.pt"
                torch.save(_cpu_state_dict(accelerator, wrapped), path)
                state_paths[epoch] = path
            history.append(
                {
                    "epoch": float(epoch),
                    "phase": phase.phase,
                    "auroc": metrics.auroc,
                    "auprc": metrics.auprc,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "fidelity": fidelity,
                    "quality_thresholds": {
                        "validation_values_finite": not bool(validation_nonfinite_failure),
                        "warm_reference_floor_pass": warm_reference_quality_pass,
                        "validation_logit_collapse": bool(collapse_failure),
                        "slot_collapse": bool(slot_collapse_failure),
                        "cumulative_quality_guards_passed": quality_guards_passed,
                    },
                    "gradient_norm_probes": epoch_probes,
                    **{f"loss_{name}": value for name, value in epoch_parts.items()},
                }
            )
        failed = accelerator.reduce(
            torch.tensor(validation_nonfinite_failure, device=accelerator.device),
            reduction="sum",
        )
        if int(failed.item()) > 0:
            raise RuntimeError("non-finite E2E validation quality telemetry")
        failed = accelerator.reduce(
            torch.tensor(slot_collapse_failure, device=accelerator.device), reduction="sum"
        )
        if int(failed.item()) > 0 and enforce_quality:
            raise RuntimeError("training_invalid(slot_collapse)")
        failed = accelerator.reduce(
            torch.tensor(collapse_failure, device=accelerator.device), reduction="sum"
        )
        if int(failed.item()) > 0 and enforce_quality:
            raise RuntimeError("persistent E2E validation-logit collapse")
        per_epoch_profiles.append(
            {
                "epoch": epoch,
                "steps": epoch_steps,
                "global_pairs": epoch_global_pairs,
                "local_pairs": epoch_local_pairs,
                "local_tokens": epoch_local_tokens,
                "wall_seconds": epoch_wall,
                "data_wait_seconds": epoch_data_wait,
                "compute_seconds": max(epoch_wall - epoch_data_wait - validation_seconds, 0.0),
                "validation_seconds": validation_seconds,
                **epoch_validation_timing,
            }
        )
        total_local_pairs += epoch_local_pairs
        total_local_tokens += epoch_local_tokens
        total_wall += epoch_wall
        total_data_wait += epoch_data_wait
        total_validation_seconds += validation_seconds
        for name in total_validation_timing:
            total_validation_timing[name] += epoch_validation_timing[name]

    if global_step != executed_steps:
        raise RuntimeError(f"E2E execution coverage broken: {global_step} != {executed_steps}")
    last_state = _cpu_state_dict(accelerator, wrapped) if accelerator.is_main_process else {}
    selected_epoch_local = 0
    best_state: dict[str, torch.Tensor] = {}
    best_metrics: EdgeMetrics | None = None
    if accelerator.is_main_process:
        assert last_metrics is not None and last_fidelity is not None
        if profile_only:
            selected_epoch_local = len(epoch_step_counts)
            best_state = last_state
            best_metrics = last_metrics
        else:
            selected = select_e2e_checkpoint(
                records,
                arm,
                auprc_tolerance=training.selection_auprc_tolerance,
                mmd_tolerance=training.selection_mmd_tolerance,
            )
            if selected is not None:
                selected_epoch_local = selected.epoch
                best_state = cast(
                    dict[str, torch.Tensor],
                    torch.load(state_paths[selected.epoch], map_location="cpu", weights_only=True),
                )
                best_metrics = metrics_by_epoch[selected.epoch]
    selected_epoch_tensor = accelerator.reduce(
        torch.tensor(selected_epoch_local, device=accelerator.device, dtype=torch.int64),
        reduction="sum",
    )
    selected_epoch = int(selected_epoch_tensor.item())
    selection_status = "selected"
    diagnostic_epoch: int | None = None
    if selected_epoch <= 0:
        selection_status = "telemetry_miss_last_epoch"
        diagnostic_epoch = len(epoch_step_counts)
        if accelerator.is_main_process:
            best_state = last_state
            best_metrics = last_metrics
    result_epoch = selected_epoch if selected_epoch > 0 else cast(int, diagnostic_epoch)

    if not profile_only and arm == "full":
        precision_error = None
        if accelerator.is_main_process:
            assert fixed_replay is not None
            inner_model = cast(_CompositeStep, accelerator.unwrap_model(wrapped)).model
            assert isinstance(inner_model, EgoStitchModel)
            inner_model.load_state_dict(best_state)
            try:
                selected_precision = _e2e_precision_differential(
                    inner_model,
                    cast(dict[str, torch.Tensor], fixed_replay["edge"]),
                    accelerator,
                    enforce_quality=enforce_quality,
                )
                if not bool(selected_precision["quality_pass"]):
                    quality_guards_passed = False
                    quality_guard_events.append(
                        {
                            "kind": "selected_precision",
                            "epoch": result_epoch,
                            "quality_failures": selected_precision["quality_failures"],
                        }
                    )
            except Exception as error:
                precision_error = error
                logger.error(
                    "selected-checkpoint precision differential: %s: %s",
                    type(error).__name__,
                    error,
                )
        _raise_synchronized_precision_failure(
            accelerator,
            local_error=precision_error,
            context="selected-checkpoint",
        )

    local_peak_gib = (
        torch.cuda.max_memory_allocated(accelerator.device) / (1024**3) if use_cuda else 0.0
    )
    stats = torch.tensor(
        [total_local_pairs, total_local_tokens, total_wall, total_data_wait, local_peak_gib],
        device=accelerator.device,
        dtype=torch.float64,
    )
    rank_stats = accelerator.gather(stats.unsqueeze(0)).cpu().numpy()
    access_digest = hashlib.sha256(
        "".join(f"{node}\n" for node in sorted(factory.training_nodes_read)).encode()
    ).digest()
    row_access_digest = hashlib.sha256(
        "".join(f"{row}\n" for row in sorted(factory.training_f0_rows_read)).encode()
    ).digest()
    access_digest_rows = (
        accelerator.gather(
            torch.tensor(list(access_digest), device=accelerator.device, dtype=torch.uint8)
        )
        .reshape(world, 32)
        .cpu()
        .numpy()
    )
    access_counts = (
        accelerator.gather(
            torch.tensor(
                [len(factory.training_nodes_read)], device=accelerator.device, dtype=torch.int64
            )
        )
        .cpu()
        .tolist()
    )
    row_access_digest_rows = (
        accelerator.gather(
            torch.tensor(list(row_access_digest), device=accelerator.device, dtype=torch.uint8)
        )
        .reshape(world, 32)
        .cpu()
        .numpy()
    )
    row_access_counts = (
        accelerator.gather(
            torch.tensor(
                [len(factory.training_f0_rows_read)],
                device=accelerator.device,
                dtype=torch.int64,
            )
        )
        .cpu()
        .tolist()
    )
    local_access_valid = (
        factory.training_nodes_read <= factory._allowed_training_nodes
        and factory.training_f0_rows_read <= factory._allowed_training_rows
    )
    access_valid = (
        accelerator.gather(
            torch.tensor([int(local_access_valid)], device=accelerator.device, dtype=torch.int64)
        )
        .cpu()
        .tolist()
    )
    observed_access = [
        {
            "rank": index,
            "node_count": int(access_counts[index]),
            "nodes_sha256": bytes(access_digest_rows[index].tolist()).hex(),
            "f0_row_count": int(row_access_counts[index]),
            "f0_rows_sha256": bytes(row_access_digest_rows[index].tolist()).hex(),
            "all_nodes_within_v_fit": bool(access_valid[index]),
        }
        for index in range(world)
    ]
    slowest_wall = max(float(row[2]) for row in rank_stats)
    global_pairs = int(sum(int(cast(int, entry["global_pairs"])) for entry in per_epoch_profiles))
    global_tokens = int(sum(float(row[1]) for row in rank_stats))
    runtime_profile: dict[str, object] = {
        "epochs_completed": len(epoch_step_counts),
        "validations_completed": len(epoch_step_counts),
        "v_hold_validation_event_count": len(validation_events),
        "v_hold_validation_events": validation_events,
        "peak_memory_gib_per_rank": [float(row[4]) for row in rank_stats],
        "steady_state_data_wait_fraction": max(
            float(row[3] / row[2]) if row[2] > 0 else 0.0 for row in rank_stats
        ),
        "training_coverage_exact": True,
        "validation_coverage_exact": True,
        "feature_cache_hit_rate": 1.0,
        "counterfactual_stop_epoch": None,
        "per_rank": [
            {
                "rank": index,
                "pairs": int(row[0]),
                "batches": executed_steps,
                "steps": executed_steps,
                "tokens": int(row[1]),
                "train_wall_seconds": float(row[2]),
                "data_wait_seconds": float(row[3]),
                "pairs_per_second": float(row[0] / row[2]) if row[2] > 0 else 0.0,
                "tokens_per_second": float(row[1] / row[2]) if row[2] > 0 else 0.0,
            }
            for index, row in enumerate(rank_stats)
        ],
        "global_pairs": global_pairs,
        "global_pairs_per_second": global_pairs / slowest_wall if slowest_wall > 0 else 0.0,
        "global_tokens": global_tokens,
        "global_tokens_per_second": global_tokens / slowest_wall if slowest_wall > 0 else 0.0,
        "validation_seconds": total_validation_seconds,
        "validation_timing": total_validation_timing,
        "per_epoch": per_epoch_profiles,
        "gradient_norm_series": gradient_norm_series,
        "optimizer_step_gradients": optimizer_step_gradients,
        "observed_training_access": observed_access,
        "kendall_fallback": {"active": False, "activated_step": None, "imbalance_streak_steps": 0},
        "run_kind": run_kind,
        "arm": arm,
        "total_optimizer_steps": executed_steps,
        "schedule_total_optimizer_steps": schedule_total_steps,
        "phase_boundaries": {"phase_a_end": phase_a_end, "phase_b_end": phase_b_end},
        "parameter_groups": {
            "names": parameter_groups.names,
            "sha256": parameter_groups.sha256,
        },
        "validation_role": data.validation_role,
        "access_audit": data.access_audit,
        "precision_differential": {
            "end_ramp": end_ramp_precision,
            "selected": selected_precision,
        },
        "initial_slot_health": initial_slot_health,
        "quality_guard_events": quality_guard_events,
        "quality_guards_passed": quality_guards_passed,
        "selected_epoch": selected_epoch if selected_epoch > 0 else None,
        "selection_status": selection_status,
        "diagnostic_epoch": diagnostic_epoch,
        "profile_only": profile_only,
    }
    if accelerator.is_main_process and data.access_audit is not None:
        data.access_audit["observed_training_access"] = observed_access
    if accelerator.is_main_process:
        assert best_metrics is not None and last_metrics is not None
    else:
        placeholder = EdgeMetrics(
            auroc=0.0,
            auprc=0.0,
            accuracy=0.0,
            sensitivity=0.0,
            specificity=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mcc=0.0,
            ece=0.0,
            brier=0.0,
            threshold=0.5,
            n_pos=0,
            n_neg=0,
        )
        best_metrics = placeholder
        last_metrics = placeholder
    for path in state_paths.values():
        path.unlink(missing_ok=True)
    if accelerator.is_main_process:
        checkpoint_dir.rmdir()
    return EgoTrainResult(
        best_state_dict=best_state,
        best_epoch=result_epoch,
        best_val_metrics=best_metrics,
        last_state_dict=last_state,
        last_epoch=len(epoch_step_counts),
        last_val_metrics=last_metrics,
        history=history,
        counterfactual_stop_epoch=None,
        runtime_profile=runtime_profile,
        kendall_state={"active": False, "activated_step": None, "log_variances": {}},
    )


def train_egostitch_ddp_loop(
    model: EgoStitchStage1 | EgoStitchModel,
    cfg: EgoConfig,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    node_batch: int,
    max_steps: int | None = None,
) -> EgoTrainResult:
    """Dispatch the §13.19 E2E stability schedule (any world size >= 1).

    Emits the exact runtime-profile schema the orchestrator validates and the
    Task-4 checkpoint state via `EgoTrainResult`.

    The legacy frozen-s0 loop this function used to carry was removed with the
    rest of the `egostitch` family (design 2026-07-29 Sec 6.2); the only
    executable path left is `_train_e2e_stability_loop`.

    Args:
        model: The e2e model.
        cfg: The validated worker config.
        data: The assembled data bundle.
        accelerator: The (DDP or single-process) accelerator.
        node_batch: Per-rank ``B_n`` (the orchestrator-selected candidate).
        max_steps: Bounded optimizer-step limit; forbidden for this family.

    Returns:
        The `EgoTrainResult`.

    Raises:
        RuntimeError: When the config carries no ``training`` section, or the
            model is not an `EgoStitchModel`.
        ValueError: When ``max_steps`` is set.
    """
    if cfg.training is None:
        raise RuntimeError(
            "legacy frozen-s0 egostitch training was removed; every executable "
            "configuration must carry a training section"
        )
    if not isinstance(model, EgoStitchModel):
        raise RuntimeError("§13.19 training requires model.family='egostitch_e2e'")
    if max_steps is not None:
        raise ValueError("§13.19 execution forbids --max-steps")
    if cfg.runtime is not None and node_batch != cfg.runtime.token_budget:
        raise ValueError(
            "effective node_batch must equal the model-config-bound "
            f"runtime.token_budget ({node_batch} != {cfg.runtime.token_budget})"
        )
    return _train_e2e_stability_loop(
        model,
        cfg,
        data,
        accelerator,
        node_batch=node_batch,
    )


# --------------------------------------------------------------------------- artifacts


V_HOLD_VALIDATION_EVENTS_FILENAME = "v_hold_validation_events.jsonl"

_MODEL_CONFIG_HASH_SCHEMA = "egostitch_e2e_model_config_v2"

def model_config_hash(cfg: EgoConfig) -> str:
    """Hash the model-defining configuration for run provenance.

    `config_to_dict` is path-sensitive: it carries the whole config, including
    ``output_dir`` and ``data.root`` (documented CLAUDE.md trap). This digest
    covers what defines the model and what it is trained on, and deliberately
    excludes:

    - ``output_dir`` / ``data.root`` / every other path;
    - ``optim.epochs`` -- recorded by the plan/config identity instead;
    - ``model.config['feature_stats_sha256']`` -- recorded alongside this digest;
    - ``seed`` -- an execution parameter. The formal stage may sweep
      ``--seeds`` without changing the model definition.

    Args:
        cfg: The validated worker config.

    Returns:
        The 64-character hex digest.
    """
    model_config = {
        key: value for key, value in cfg.model.config.items() if key != "feature_stats_sha256"
    }
    payload: dict[str, object] = {
        "schema": _MODEL_CONFIG_HASH_SCHEMA,
        "model": {"family": cfg.model.family, "config": model_config},
        "data": {
            "strategy": cfg.data.strategy,
            "train_positives": cfg.data.train_positives,
            "negative_ratio": cfg.data.negative_ratio,
            "partition_seed": cfg.data.partition_seed,
            "msg_fraction": cfg.data.msg_fraction,
            "node_batch": cfg.data.node_batch,
            "edge_batch": cfg.data.edge_batch,
            "expected_missing_features": list(cfg.data.expected_missing_features),
        },
        "optim": {
            "lr": cfg.optim.lr,
            "weight_decay": cfg.optim.weight_decay,
            "warmup_steps": cfg.optim.warmup_steps,
            "grad_clip": cfg.optim.grad_clip,
        },
        "training": asdict(cfg.training) if cfg.training is not None else None,
        "diagnostics": asdict(cfg.diagnostics),
        "runtime": {
            "token_budget": cfg.runtime.token_budget if cfg.runtime is not None else None,
        },
        "mixed_precision": cfg.mixed_precision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_run_start_metadata(
    cfg: EgoConfig,
    data: EgoStitchData,
    *,
    world_size: int,
    debug: bool = False,
    config_path: Path | None = None,
    feature_stats_sha256: str | None = None,
) -> None:
    """Bind the run to its config and s0 statistics before optimization."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "run_metadata.json"
    if path.exists():
        raise FileExistsError(f"run-start metadata already exists: {path}")
    run_kind = "debug" if debug else (cfg.run_kind or "formal")
    e2e_config = (
        E2EConfig.from_mapping(cfg.model.config)
        if cfg.training is not None and cfg.model.family == _EGOSTITCH_E2E_FAMILY
        else None
    )
    arm = _e2e_arm_name_from_config(e2e_config) if e2e_config is not None else None
    metadata = {
        "status": "started",
        "run_kind": run_kind,
        "formal_artifacts_published": False,
        "started_at": datetime.now(UTC).isoformat(),
        "seed": cfg.seed,
        "world_size": world_size,
        "partition_seed": cfg.data.partition_seed,
        "strategy": cfg.data.strategy,
        "rho_train": data.rho_train,
        "positives_mode": cfg.data.train_positives,
        "permanent_null": cfg.model.config.get("permanent_null", "none"),
        "model_family": cfg.model.family,
        "p_topo": cfg.model.config.get("p_topo", 0.0),
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "config_sha256": _sha256_file(config_path) if config_path is not None else None,
        "arm": arm,
        "feature_stats_sha256": feature_stats_sha256 or "",
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_outputs(
    result: EgoTrainResult, cfg: EgoConfig, data: EgoStitchData, *, debug: bool = False
) -> None:
    """Write the pinned Task-4 artifacts and finalize the run's metadata record.

    ``best.pt``/``last.pt`` carry exactly the seven pinned payload keys;
    ``run_metadata.json`` additionally records the s0 checkpoint identity, the
    partition seed, and the measured ``rho_train``.
    """
    output_dir = cfg.output_dir
    config_dict = config_to_dict(cfg)
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("run-start metadata is missing; refuse post-hoc artifact finalization")
    run_metadata = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
    expected_run_kind = "debug" if debug else (cfg.run_kind or "formal")
    if run_metadata.get("run_kind") != expected_run_kind:
        raise RuntimeError("run kind changed after run start; refusing to finalize artifacts")
    validation_events = result.runtime_profile.get("v_hold_validation_events")
    validation_event_count = result.runtime_profile.get("v_hold_validation_event_count")
    validation_evidence: dict[str, object] | None = None
    if validation_events is not None or validation_event_count is not None:
        if (
            not isinstance(validation_events, list)
            or isinstance(validation_event_count, bool)
            or not isinstance(validation_event_count, int)
            or validation_event_count != len(validation_events)
        ):
            raise RuntimeError("invalid V_hold validation-event count in runtime profile")
        validation_events_path = output_dir / V_HOLD_VALIDATION_EVENTS_FILENAME
        if not validation_events_path.is_file():
            raise RuntimeError("V_hold validation-event ledger is missing")
        persisted_events = [
            json.loads(line)
            for line in validation_events_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if persisted_events != validation_events:
            raise RuntimeError("V_hold validation-event ledger disagrees with runtime profile")
        validation_evidence = {
            "schema": "egostitch_e2e_v_hold_validation_events_v1",
            "count": validation_event_count,
            "path": V_HOLD_VALIDATION_EVENTS_FILENAME,
            "sha256": _sha256_file(validation_events_path),
        }

    def payload(
        state: dict[str, torch.Tensor], epoch: int, metrics: EdgeMetrics
    ) -> dict[str, object]:
        return {
            "model_state": state,
            "model_family": cfg.model.family,
            "model_config": dict(cfg.model.config),
            "epoch": epoch,
            "val_metrics": asdict(metrics),
            "seed": cfg.seed,
            "config": config_dict,
        }

    best_path = output_dir / "best.pt"
    torch.save(
        payload(result.best_state_dict, result.best_epoch, result.best_val_metrics), best_path
    )
    torch.save(
        payload(result.last_state_dict, result.last_epoch, result.last_val_metrics),
        output_dir / "last.pt",
    )
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.history:
            handle.write(json.dumps({**entry, "epoch": int(cast(float, entry["epoch"]))}) + "\n")

    checkpoint_sha256 = _sha256_file(best_path)
    checkpoint_role = "debug_only" if debug else "formal_plan_selected"
    validation_liveness_observed = (
        result.best_epoch > 0
        and (
            cfg.model.config.get("permanent_null") != "none"
            or any(
                cast(dict[str, float], entry["fidelity"]).get("topology_delta_ratio", 0.0)
                >= 1e-3
                for entry in result.history
                if int(cast(float, entry["epoch"])) == result.best_epoch
            )
        )
    )
    run_metadata.update(
        {
            "status": "debug_complete" if debug else "complete",
            "formal_artifacts_published": not debug and expected_run_kind == "formal",
            "checkpoint_id": _state_digest(result.best_state_dict)[:16],
            "checkpoint_role": checkpoint_role,
            "checkpoint_sha256": checkpoint_sha256,
            "diagnostic_checkpoint_sha256": None,
            "diagnostic_checkpoint_epoch": None,
            "validation_liveness_pass": validation_liveness_observed,
            "validation_role": data.validation_role,
            "v_hold_validation_evidence": validation_evidence,
            "access_audit": data.access_audit,
            "kendall_fallback": result.kendall_state,
            "training_diagnostics": {
                "fidelity_series": [entry["fidelity"] for entry in result.history],
                "gradient_norm_series": result.runtime_profile["gradient_norm_series"],
                "optimizer_step_gradients": result.runtime_profile.get(
                    "optimizer_step_gradients", []
                ),
                "kendall_fallback": result.runtime_profile["kendall_fallback"],
            },
            "torch_version": str(torch.__version__),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote artifacts to %s (checkpoint_id %s)", output_dir, run_metadata["checkpoint_id"]
    )


# --------------------------------------------------------------------------- DDP worker modes


def _bind_feature_standardization(
    model: EgoStitchModel,
    cfg: EgoConfig,
    data: EgoStitchData,
) -> str:
    """Bind and record the plan-scoped F0 statistics before the first step.

    A config that still pins `feature_stats_sha256` is honoured in every kind
    -- a disagreement with the assembled statistics is always a refusal.

    Args:
        model: The freshly constructed E2E model.
        cfg: The loaded run configuration.
        data: The assembled training data, carrying the V_fit statistics.

    Returns:
        The bound `feature_stats_sha256`, or ``""`` for the `row_layernorm`
        mode (the registered D0-ablation arm and replay of pre-D0
        checkpoints), which binds no statistics.

    Raises:
        RuntimeError: When the registered `zscore_vfit_v1` mode has no
            statistics available (`data.feature_stats is None`).
        RuntimeError: When the config pins a non-empty `feature_stats_sha256`
            that disagrees with the assembled statistics' digest.
    """
    mode = str(cfg.model.config.get("feature_standardization", "zscore_vfit_v1"))
    if mode != "zscore_vfit_v1":
        return ""
    stats = data.feature_stats
    if stats is None:
        raise RuntimeError(
            "feature standardization statistics are unavailable; "
            "rebuild the feature pack before training"
        )
    pinned = str(cfg.model.config.get("feature_stats_sha256", ""))
    if pinned and pinned != stats.digest:
        raise RuntimeError(
            "feature_stats_sha256 mismatch: config pins "
            f"{pinned}, assembled statistics are {stats.digest}"
        )
    effective_run_kind = cfg.run_kind or "formal"
    model.set_feature_stats(stats)
    logger.info(
        "registered feature standardization mode=%s rows=%d feature_stats_sha256=%s "
        "run_kind=%s config_pinned=%s",
        mode,
        stats.n_rows,
        stats.digest,
        effective_run_kind,
        "yes" if pinned else "no",
    )
    return stats.digest


def _run_ddp_worker(cfg: EgoConfig, args: EgoCliArgs) -> None:
    """Dispatch an ``accelerate launch`` worker to the requested DDP mode."""
    measurement_only = args.ddp_mode in _PROBE_DISPATCH_MODES
    if measurement_only:
        if args.max_steps is not None:
            raise ValueError("measurement-only probe modes do not accept --max-steps")
    else:
        cfg, _is_debug = prepare_ddp_run_config(cfg, max_steps=args.max_steps)
    if args.pack_dir is None or args.token_budget_per_rank is None or args.profile_output is None:
        raise ValueError(
            "DDP worker modes require --pack-dir, --token-budget-per-rank, and --profile-output"
        )

    effective_run_kind = "debug" if args.max_steps is not None else (cfg.run_kind or "formal")
    accelerator = build_egostitch_ddp_accelerator(
        cfg.mixed_precision,
        find_unused_parameters=False,
    )
    set_seed(cfg.seed)
    logger.info(
        "egostitch ddp worker mode=%s rank=%d/%d device=%s",
        args.ddp_mode,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    run_kind = effective_run_kind
    feature_stats_sha256 = ""

    data = assemble_egostitch_data(cfg, pack_dir=args.pack_dir)
    model = EgoStitchModel(E2EConfig.from_mapping(cfg.model.config))
    feature_stats_sha256 = _bind_feature_standardization(model, cfg, data)
    _run_ddp_dispatch(
        cfg,
        args,
        model,
        data,
        accelerator=accelerator,
        run_kind=run_kind,
        feature_stats_sha256=feature_stats_sha256,
        node_batch=args.token_budget_per_rank,
        profile_output=args.profile_output,
    )


def _run_ddp_dispatch(
    cfg: EgoConfig,
    args: EgoCliArgs,
    model: EgoStitchModel,
    data: EgoStitchData,
    *,
    accelerator: Accelerator,
    run_kind: str,
    feature_stats_sha256: str,
    node_batch: int,
    profile_output: Path,
) -> None:
    """Run one dispatch mode on an assembled, bound model (see `_run_ddp_worker`)."""
    if args.ddp_mode in _PROBE_DISPATCH_MODES:
        accelerator.wait_for_everyone()
        _write_json_rank_zero(
            accelerator,
            profile_output,
            {
                "mode": args.ddp_mode,
                "feature_stats_sha256": feature_stats_sha256,
                "feature_stats_rows": data.feature_stats.n_rows if data.feature_stats else 0,
            },
        )
        logger.info(
            "egostitch %s measurement complete: feature_stats_sha256=%s",
            args.ddp_mode,
            feature_stats_sha256,
        )
        return
    if args.ddp_mode != "train":
        raise ValueError(f"unsupported DDP mode: {args.ddp_mode!r}")

    degree_prior = e2e_degree_prior_init(model, data)
    logger.info("degree head centered on G_fit prior mean(log d)=%.6f", degree_prior)

    if accelerator.is_main_process:
        write_run_start_metadata(
            cfg,
            data,
            world_size=accelerator.num_processes,
            debug=args.max_steps is not None,
            config_path=args.config,
            feature_stats_sha256=feature_stats_sha256,
        )
    accelerator.wait_for_everyone()
    # The failure half of this contract lives in `_run_ddp_worker`, which wraps
    # this call together with data assembly and digest binding -- guards that
    # also raise before the first step.
    result = train_egostitch_ddp_loop(
        model, cfg, data, accelerator, node_batch=node_batch, max_steps=args.max_steps
    )
    if accelerator.is_main_process:
        write_outputs(result, cfg, data, debug=args.max_steps is not None)
    _write_json_rank_zero(accelerator, profile_output, result.runtime_profile)
    logger.info(
        "egostitch ddp train complete: best epoch %d val AUPRC %.4f (counterfactual_stop_epoch=%s)",
        result.best_epoch,
        result.best_val_metrics.auprc,
        result.counterfactual_stop_epoch,
    )


# --------------------------------------------------------------------------- CLI entry point


def main(argv: Sequence[str] | None = None) -> None:
    """Run the EgoStitch worker CLI end to end.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    loaded_cfg = load_config(args.config)
    cfg = apply_overrides(loaded_cfg, args)

    if args.ddp_mode is not None:
        _run_ddp_worker(cfg, args)
        return

    raise ValueError(
        "EgoStitch Stage-1 training must run through src.e2_pipeline so the visible "
        "GPU count is auto-detected and workers are launched with Accelerate DDP"
    )


if __name__ == "__main__":
    main()
