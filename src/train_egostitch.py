r"""EgoStitch Stage-1 training worker (spec Sec 13; auto-sized H20 DDP).

Drop-in worker for the `src.e2_pipeline` orchestrator: implements the same
``--ddp-mode train`` CLI contract, runtime-profile schema,
and Task-4 checkpoint payload as ``src.train_b0``, over the EgoStitch two-stream
composite step (node stream -> L_recon/L_ssl/L_real; edge stream -> L_edge; the
joint-pair stream joins in Stage 3). ``--token-budget-per-rank`` is
reinterpreted for this family as the per-rank node-stream batch size ``B_n``
(spec Sec 13.13).

The worker executes the plan-bound formal schedule on the training universe and
validates on ``V_val``, a full-density node region carved out of
`train_graph.pkl` (`src/data/val_region.py`). Model-quality diagnostics are
recorded as telemetry; they do not authorize or block checkpoint publication,
scoring, or evaluation.

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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair
from src.data.ego_targets import EgoTargetBuilder, EgoTargets
from src.data.feature_stats import FeatureStats, feature_stats_for_universe
from src.data.features import FeatureStore, build_f0_matrix
from src.data.grounding import build_grounding_pool
from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable
from src.data.pairs import NegativeSampler
from src.data.prefetch import _prefetch_batches
from src.data.val_region import (
    ValBallUnionUniverse,
    ValRegionParams,
    ValRegionSplit,
    derive_val_region_split,
    val_ball_union_universe,
)
from src.distill.losses import kd_set_gram_loss, kd_set_seed_loss
from src.eval.checkpoint_selection import (
    CheckpointCandidate,
    TopologyValidationMetrics,
    select_checkpoint,
)
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.eval.val_topology import (
    ValTopologyReference,
    build_val_topology_reference,
    val_region_topology_metrics,
)
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1
from src.model.egostitch.classifier.b0_v31 import (
    NULL_ALL_HEAD,
    B0V31PairClassifier,
    GatedCrossAttention,
    masks_for_null,
    sample_branch_masks,
)
from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.generator import EgoStitchImagineGenerator, StitchedGraph
from src.model.egostitch.generator.full_oracle import (
    FullEgoFeaturesGenerator,
    FullEgoGraph,
    FullOracleGenerator,
)
from src.model.egostitch.generator.imagine import (
    NULL_MODE_ALL,
    NULL_MODE_CONTENT,
    NULL_MODE_FULL,
    SlotSet,
)
from src.model.egostitch.generator.losses import stage1_family_tensors, stage1_total
from src.model.egostitch.generator.oracle import OracleStructGenerator, build_oracle_table
from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph
from src.model.egostitch.registry import build_encoder
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
    _stable_bce_with_logits,
    _state_digest,
    _write_json_rank_zero,
    validate_gathered_validation,
)

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = "benchmark_2025_neurips"
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_PACK_F0_FILENAME = "f0_matrix.pt"
_PACK_GROUNDING_FILENAME = "grounding.npz"
_PACK_VALIDATION_GROUNDING_FILENAME = "grounding_validation.npz"
_PACK_MANIFEST_FILENAME = "manifest.json"
_PACK_FEATURE_STATS_FILENAME = "feature_stats.npz"


def _egostitch_ddp_kwargs(*, find_unused_parameters: bool = True) -> DistributedDataParallelKwargs:
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
        kwargs_handlers=[_egostitch_ddp_kwargs(find_unused_parameters=find_unused_parameters)],
    )


# The complete execution-context domain: ``formal`` produces registered results,
# ``diagnostic`` may consume explicitly declared held-out truth and publishes
# nothing formal, and ``debug`` is *derived* from ``--max-steps``
# by `write_run_start_metadata`/`write_outputs` rather than selected on the
# CLI. It is named here because `run_metadata.json` publishes it and
# `src.experiments.probes` reads it back.
E2ERunKind = Literal["formal", "diagnostic", "debug"]

# The single validation universe: the V_val node region carved out of
# `train_graph.pkl` (`src/data/val_region.py`). It is also the grounding-pool
# ``role_universe`` identity.
_E2E_VALIDATION_ROLE: Literal["V_val"] = "V_val"


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class EgoDataConfig:
    """The ``data:`` config section for the EgoStitch worker.

    Attributes:
        root: Data root containing the benchmark and feature packages.
        strategy: Split strategy name (Benchmark-A = ``breadth_first``).
        negative_ratio: Negatives per positive in the edge stream (spec 1:5).
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
    negative_ratio: int
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
        epochs: Maximum epoch count; patience on the total validation loss may
            stop the run earlier.
        warmup_steps: Linear LR warmup steps.
        grad_clip: Gradient-norm clip; 0 disables.
        gradient_accumulation_steps: Physical microbatches per optimizer step.
    """

    lr: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    grad_clip: float
    gradient_accumulation_steps: int = 1


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


@dataclass(frozen=True)
class EgoDiagnosticsConfig:
    """Registered training-fidelity and loss-balance diagnostics."""

    gradient_probe_interval: int
    gradient_imbalance_ratio: float
    gradient_imbalance_steps: int
    probe_s1_abs_mean_max: float
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
    residual_ratio_min: float = 1e-3


@dataclass(frozen=True)
class EgoDistillConfig:
    """Gate A set-student distillation controls (``full_ego_features`` arm only).

    Attributes:
        teacher_checkpoint: A published ``full_ego_oracle`` + ``grit_gmt``
            ``best.pt`` whose frozen encoder supplies the per-batch KD targets
            (node tokens and PMA seed tokens over the identical node layout).
        lambda_seed: Weight of `kd_set_seed_loss` (per-seed PMA latent cosine).
        lambda_gram: Weight of `kd_set_gram_loss` (within-set node Gram).
        warm_start_readout: Copy the teacher encoder's ``project``/``readout``
            weights into the student encoder at startup so seed ``k`` starts
            with the teacher's seed-``k`` semantics (identical shapes by the
            d_model/seeds equality this config enforces at teacher load).
    """

    teacher_checkpoint: Path
    lambda_seed: float = 1.0
    lambda_gram: float = 1.0
    warm_start_readout: bool = True


@dataclass(frozen=True)
class EgoTopologyValidationConfig:
    """Two-resolution V_val topology-validation cadence.

    ``full_every_epochs=1`` preserves the existing behavior. Larger intervals
    use a fixed, smaller bucket panel on intervening epochs while retaining the
    full 500-ball sampled bank on the configured cadence. Only full-resolution
    epochs participate in checkpoint selection.
    """

    full_every_epochs: int = 1
    cascade_buckets_per_size: int = 5


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
    distill: EgoDistillConfig | None = None
    topology_validation: EgoTopologyValidationConfig = field(
        default_factory=EgoTopologyValidationConfig
    )


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
            "distill",
            "topology_validation",
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
        "negative_ratio",
        "node_batch",
        "edge_batch",
        "f0_cache",
        "grounding_cache",
        "expected_missing_features",
        "pack_dir",
    )
    _check_no_unknown_keys(data_raw, data_keys, "data")
    data = EgoDataConfig(
        root=Path(_as_str(_require(data_raw, "root", "data."), "data.root")),
        strategy=_as_str(_require(data_raw, "strategy", "data."), "data.strategy"),
        negative_ratio=_as_int(
            _require(data_raw, "negative_ratio", "data."), "data.negative_ratio"
        ),
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
    optim_keys: tuple[str, ...] = (
        "lr",
        "weight_decay",
        "epochs",
        "warmup_steps",
        "grad_clip",
        "gradient_accumulation_steps",
    )
    _check_no_unknown_keys(optim_raw, optim_keys, "optim")
    optim = EgoOptimConfig(
        lr=_as_float(_require(optim_raw, "lr", "optim."), "optim.lr"),
        weight_decay=_as_float(_require(optim_raw, "weight_decay", "optim."), "optim.weight_decay"),
        epochs=_as_int(_require(optim_raw, "epochs", "optim."), "optim.epochs"),
        warmup_steps=_as_int(_require(optim_raw, "warmup_steps", "optim."), "optim.warmup_steps"),
        grad_clip=_as_float(_require(optim_raw, "grad_clip", "optim."), "optim.grad_clip"),
        gradient_accumulation_steps=_as_int(
            optim_raw.get("gradient_accumulation_steps", 1),
            "optim.gradient_accumulation_steps",
        ),
    )
    if optim.epochs <= 0:
        raise ValueError("optim.epochs must be positive")
    if optim.gradient_accumulation_steps <= 0:
        raise ValueError("optim.gradient_accumulation_steps must be positive")
    if optim.gradient_accumulation_steps > 1 and E2EConfig.from_mapping(
        model_kwargs
    ).generator.name not in ("full_ego_oracle", "full_ego_features"):
        raise ValueError(
            "optim.gradient_accumulation_steps > 1 is only supported for "
            "model.config.generator.name='full_ego_oracle' or 'full_ego_features'"
        )

    diagnostics_raw = _as_mapping(_require(raw, "diagnostics", ""), "diagnostics")
    diagnostic_keys = (
        "gradient_probe_interval",
        "gradient_imbalance_ratio",
        "gradient_imbalance_steps",
        "probe_s1_abs_mean_max",
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
        )
        if runtime.token_budget <= 0:
            raise ValueError(
                "runtime.token_budget must be a positive node-batch (B_n) size for this family"
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
            residual_ratio_min=_as_float(
                training_raw.get("residual_ratio_min", 1e-3), "training.residual_ratio_min"
            ),
        )
        # `phase_a_fraction`/`phase_b_fraction` are the Wave-1 oracle-scaffold
        # experiment's carve-out from the pinned-defaults check below: every R1
        # config sets `phase_a_fraction: 0.0` (a zero-parameter generator has
        # nothing for Phase A to pretrain, design doc 2026-08-04 §8), which the
        # pinned 0.2/0.1 split this gate otherwise enforces would reject.
        if not math.isfinite(training.phase_a_fraction) or not (
            0.0 <= training.phase_a_fraction <= 1.0
        ):
            raise ValueError("training.phase_a_fraction must be finite and in [0, 1]")
        if not math.isfinite(training.phase_b_fraction) or not (
            0.0 <= training.phase_b_fraction <= 1.0
        ):
            raise ValueError("training.phase_b_fraction must be finite and in [0, 1]")
        if training.phase_a_fraction + training.phase_b_fraction > 1.0:
            raise ValueError(
                "training.phase_a_fraction + training.phase_b_fraction must not exceed 1"
            )
        pinned_training = EgoStitchTrainingConfig()
        if (
            replace(
                training,
                phase_a_fraction=pinned_training.phase_a_fraction,
                phase_b_fraction=pinned_training.phase_b_fraction,
            )
            != pinned_training
        ):
            raise ValueError(
                "training values must exactly match the pinned defaults (except "
                f"phase_a_fraction and phase_b_fraction); got {training!r}"
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
        if resolved_e2e.classifier.p_topo not in (0.15, 0.0):
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

    distill: EgoDistillConfig | None = None
    if raw.get("distill") is not None:
        distill_raw = _as_mapping(raw["distill"], "distill")
        _check_no_unknown_keys(
            distill_raw,
            ("teacher_checkpoint", "lambda_seed", "lambda_gram", "warm_start_readout"),
            "distill",
        )
        warm_start_raw = distill_raw.get("warm_start_readout", True)
        if not isinstance(warm_start_raw, bool):
            raise ValueError("distill.warm_start_readout must be a boolean")
        distill = EgoDistillConfig(
            teacher_checkpoint=Path(
                _as_str(
                    _require(distill_raw, "teacher_checkpoint", "distill."),
                    "distill.teacher_checkpoint",
                )
            ),
            lambda_seed=_as_float(distill_raw.get("lambda_seed", 1.0), "distill.lambda_seed"),
            lambda_gram=_as_float(distill_raw.get("lambda_gram", 1.0), "distill.lambda_gram"),
            warm_start_readout=warm_start_raw,
        )
        if distill.lambda_seed < 0.0 or distill.lambda_gram < 0.0:
            raise ValueError("distill.lambda_seed and distill.lambda_gram must be non-negative")
        if E2EConfig.from_mapping(model_kwargs).generator.name != "full_ego_features":
            raise ValueError(
                "distill requires model.config.generator.name='full_ego_features': the KD "
                "teacher matches the features generator's stashed structural view"
            )

    topology_validation = EgoTopologyValidationConfig()
    if raw.get("topology_validation") is not None:
        topology_raw = _as_mapping(raw["topology_validation"], "topology_validation")
        _check_no_unknown_keys(
            topology_raw,
            tuple(EgoTopologyValidationConfig.__dataclass_fields__),
            "topology_validation",
        )
        topology_validation = EgoTopologyValidationConfig(
            full_every_epochs=_as_int(
                topology_raw.get("full_every_epochs", 1),
                "topology_validation.full_every_epochs",
            ),
            cascade_buckets_per_size=_as_int(
                topology_raw.get("cascade_buckets_per_size", 5),
                "topology_validation.cascade_buckets_per_size",
            ),
        )
        if topology_validation.full_every_epochs <= 0:
            raise ValueError("topology_validation.full_every_epochs must be positive")
        if topology_validation.cascade_buckets_per_size < 2:
            raise ValueError("topology_validation.cascade_buckets_per_size must be at least 2")

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
        distill=distill,
        topology_validation=topology_validation,
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
    "null_generator",
    "oracle",
    "full_ego_oracle",
    "full_ego_features",
]


@dataclass(frozen=True)
class E2EPhaseState:
    """Zero-based optimizer-step state for the rev-3.1 §14.4.3 curriculum."""

    phase: E2EPhaseName
    alpha: float
    edge_active: bool
    real_ssl_scale: float


def e2e_phase_boundaries(
    total_steps: int,
    *,
    phase_a_fraction: float = 0.2,
    phase_b_fraction: float = 0.1,
) -> tuple[int, int]:
    """Return the exclusive Phase-A and Phase-B end steps.

    ``phase_a_fraction``/``phase_b_fraction`` default to the registered
    0.2/0.1 split (every pre-oracle config pins exactly these values, so the
    defaults are behavior-preserving). The oracle-scaffold arm's
    ``phase_a_fraction: 0.0`` collapses Phase A to zero length: ``phase_a_end
    == 0`` then makes ``step < phase_a_end`` false for every non-negative
    step, so Phase A is simply skipped -- no generator pretraining steps run
    -- and the Phase-B ramp below never divides by zero, since
    ``phase_b_end > phase_a_end`` whenever ``phase_b_fraction > 0``.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= phase_a_fraction <= 1.0 or not 0.0 <= phase_b_fraction <= 1.0:
        raise ValueError("phase fractions must be within [0, 1]")
    if phase_a_fraction + phase_b_fraction > 1.0:
        raise ValueError("phase_a_fraction + phase_b_fraction must not exceed 1")
    phase_a_end = math.ceil(phase_a_fraction * total_steps)
    return phase_a_end, phase_a_end + math.ceil(phase_b_fraction * total_steps)


def e2e_phase_state(
    step: int,
    total_steps: int,
    *,
    phase_a_fraction: float = 0.2,
    phase_b_fraction: float = 0.1,
) -> E2EPhaseState:
    """Resolve the exact A/B/C behavior for one zero-based optimizer step."""
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    phase_a_end, phase_b_end = e2e_phase_boundaries(
        total_steps, phase_a_fraction=phase_a_fraction, phase_b_fraction=phase_b_fraction
    )
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


def e2e_recon_component_factors(
    step: int,
    total_steps: int,
    *,
    phase_a_fraction: float = 0.2,
    phase_b_fraction: float = 0.1,
) -> dict[str, float]:
    """Return the §14.4.1 per-component reconstruction anneal factors."""
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    edge_start, _ = e2e_phase_boundaries(
        total_steps, phase_a_fraction=phase_a_fraction, phase_b_fraction=phase_b_fraction
    )
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
        row_entropy = -(row_probability * row_probability.clamp_min(1e-30).log()).sum(dim=-1).mean(
            dim=-1
        ) / math.log(slots)
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


def e2e_degree_prior_init(model: EgoStitchStage1 | EgoStitchModel, data: EgoStitchData) -> float:
    """Center the lognormal degree head on the ``G_train`` degree prior.

    ``deg_mu`` is a raw linear output (`generator/imagine.py`'s
    `TokenizeLite.degree_dist_head`) born near 0, while
    ``mean(log d)`` on ``G_train`` sits several nats above it. `degree_nll`'s
    ``1/sigma**2`` factor turns that standing residual into a generator gradient
    above the Sec 13.19 clip threshold on *every* step from step 1 -- the
    2026-07-28 `persistent clipping` abort, whose streak needs a term that is
    live on all ten steps. Setting the output bias removes the residual at
    initialization. ``log sigma`` is deliberately left alone: matching it
    instead of ``mu`` measures worse, because it shrinks the denominator while
    the numerator is what is wrong.

    ``sorted`` is load-bearing, not cosmetic: ``build_training_graph`` adds
    nodes from a ``frozenset[str]``, whose iteration order depends on
    ``PYTHONHASHSEED`` (pinned nowhere in this repo), and ``np.log(...).mean()``
    uses pairwise summation, so an unsorted traversal makes the last ulp
    order-dependent. Each rank computes this independently, so without the
    sort the replicas can disagree in the final bit from step 0.
    """
    if isinstance(model, EgoStitchModel):
        if not isinstance(model.generator, EgoStitchImagineGenerator):
            raise RuntimeError(
                "the degree prior centers a real generator's degree head; "
                "a null-generator arm has none"
            )
        generator = model.generator.stage1
    else:
        generator = model
    graph = data.target_builder.graph
    degrees = np.asarray(
        [max(int(graph.degree(node)), 1) for node in sorted(graph.nodes())],
        dtype=np.float64,
    )
    if degrees.size == 0:
        raise RuntimeError("G_train carries no nodes for the degree prior")
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


def e2e_first_eligible_epoch(
    total_steps: int,
    steps_per_epoch: int,
    *,
    phase_a_fraction: float = 0.2,
    phase_b_fraction: float = 0.1,
) -> int:
    """First 1-based epoch ending after one complete Phase-C epoch."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    _, phase_b_end = e2e_phase_boundaries(
        total_steps, phase_a_fraction=phase_a_fraction, phase_b_fraction=phase_b_fraction
    )
    return math.ceil((phase_b_end + steps_per_epoch) / steps_per_epoch)


def e2e_weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    real_row_mask: torch.Tensor,
    *,
    world_size: int = 1,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
    positive_weight: float = 5.0,
    global_denominator: torch.Tensor | None = None,
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
    if global_denominator is not None:
        denominator = global_denominator
    elif all_reduce_sum is not None:
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


def e2e_accumulation_window_sizes(microsteps: int, accumulation_steps: int) -> list[int]:
    """Return complete optimizer-window sizes, including an exact tail window."""
    if microsteps < 0:
        raise ValueError("microsteps must be non-negative")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    full, tail = divmod(microsteps, accumulation_steps)
    return [accumulation_steps] * full + ([tail] if tail else [])


def e2e_global_live_row_mean(
    local_sum: torch.Tensor,
    *,
    live_rows: torch.Tensor,
    world_size: int,
    global_denominator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize one local KD sum for accumulation plus DDP gradient averaging."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    denominator = live_rows if global_denominator is None else global_denominator
    if not bool(torch.isfinite(denominator)) or float(denominator) < 0.0:
        raise RuntimeError("global KD live-row denominator must be finite and non-negative")
    if float(denominator) == 0.0:
        if float(live_rows) != 0.0 or float(local_sum.detach()) != 0.0:
            raise RuntimeError("empty global KD population has a non-zero local contribution")
        return local_sum * 0.0
    return world_size * local_sum / denominator


def e2e_window_effective_weight_denominator(
    batches: Sequence[_CompositeBatch], *, positive_weight: float
) -> torch.Tensor:
    """Compute one local effective-weight denominator for a microbatch window."""
    denominator = torch.zeros((), dtype=torch.float32)
    for batch in batches:
        labels = batch.edge["label"].float()
        mask = batch.edge["edge_mask"].float()
        weights = positive_weight * labels + (1.0 - labels)
        denominator += (mask * weights).sum()
    return denominator


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
            raise RuntimeError(f"E2E parameter {name!r} does not belong to a known component group")
        grouped[component].append((name, parameter))

    all_ids = [id(parameter) for rows in grouped.values() for _, parameter in rows]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != live_ids:
        raise RuntimeError("E2E optimizer groups must be disjoint and exhaustive")
    names = {group: tuple(sorted(name for name, _ in rows)) for group, rows in grouped.items()}
    parameters = {
        group: tuple(parameter for _, parameter in sorted(rows, key=lambda row: row[0]))
        for group, rows in grouped.items()
    }
    empty = sorted(group for group, rows in parameters.items() if not rows)
    if empty:
        # An empty group is legitimate in exactly two cases (Wave-1 oracle-
        # scaffold design): the component is *absent* (``model.encoder is
        # None`` for a null generator -- nothing was ever imagined, so there
        # is nothing to encode) or it is *parameter-free by construction*
        # (``generator.name == "oracle_struct"``: a deterministic lookup
        # table has no weights to learn, and the null generator itself is
        # also parameter-free). Anything else empty is still the original
        # bug this assertion exists to catch: a parameter that failed to
        # route into any component group.
        illegitimate = [
            group
            for group in empty
            if not (
                (group == "generator" and not any(model.generator.parameters()))
                or (
                    group == "encoder"
                    and (model.encoder is None or not any(model.encoder.parameters()))
                )
            )
        ]
        if illegitimate:
            raise RuntimeError(
                "every E2E optimizer group must contain trainable parameters unless its "
                f"component is absent or parameter-free by construction; illegitimately "
                f"empty: {illegitimate}"
            )
        # The null-generator arm used to refuse to build optimizer groups at
        # all here (with a pointer to `src.train_b0`) because nothing else in
        # this module was ready for a null/parameter-free generator either.
        # Wave 1 (oracle scaffold) makes both the null-generator and the
        # oracle-scaffold arms trainable through this same pipeline -- the
        # phase curriculum, family probe, active-group schedule, and
        # validation dispersion telemetry now all tolerate an absent or
        # parameter-free generator/encoder (see `_e2e_active_groups`,
        # `_e2e_family_probe`, `_validate_epoch`, `_enforce_e2e_initial_slot_health`).
        # An empty group here is therefore just a fact about this arm's
        # architecture, logged for visibility rather than refused.
        logger.info(
            "E2E optimizer group(s) with no trainable parameters (component absent or "
            "parameter-free for arm=%s): %s",
            _e2e_arm_name_from_config(model.cfg),
            empty,
        )
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
    statistics: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, E2EGradientGroupRecord]:
    """Fail closed on group gradients and independently clip active groups.

    Precomputed ``(squared_norm, nonfinite_count)`` tensors let the DDP loop
    reuse the exact statistics it already gathered for replicated-gradient
    verification instead of scanning every gradient a second time.
    """
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
    if statistics is None:
        statistics = _e2e_group_gradient_statistics(groups)
    if set(statistics) != set(groups):
        raise ValueError("gradient statistics must cover every optimizer group")
    records: dict[str, E2EGradientGroupRecord] = {}
    for name, parameters in groups.items():
        all_grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        squared_tensor, nonfinite_tensor = statistics[name]
        summary = torch.stack((squared_tensor.double(), nonfinite_tensor.double())).detach().cpu()
        squared_value = float(summary[0])
        all_nonfinite = int(summary[1])
        if name not in active_groups:
            if all_nonfinite:
                raise RuntimeError(f"non-finite gradient in inactive E2E group {name!r}")
            records[name] = E2EGradientGroupRecord(False, None, None, all_nonfinite)
            continue
        grads = all_grads
        nonfinite = all_nonfinite
        norm = math.sqrt(squared_value)
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


def e2e_assert_no_nonfinite_gradients(
    gathered_by_group: Mapping[str, torch.Tensor],
) -> None:
    """Raise with deterministic per-rank element counts before norm validation."""
    counts = {
        name: [int(value) for value in values.tolist()]
        for name, values in gathered_by_group.items()
        if bool(values.ne(0).any().item())
    }
    if counts:
        raise RuntimeError(f"non-finite E2E gradient counts by rank: {counts}")


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
    """Reject any non-finite post-step parameter or optimizer-state tensor.

    The normal path performs one device-to-host synchronization for the whole
    optimizer. A slow diagnostic pass runs only after that aggregate check
    fails, retaining the original group-specific error message.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, parameters in groups.items():
        for parameter in parameters:
            tensors.append((f"parameter in E2E group {name!r}", parameter))
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor):
                    tensors.append((f"optimizer state in E2E group {name!r}", value))
    if not tensors:
        return
    checks_by_device: dict[torch.device, list[torch.Tensor]] = {}
    for _, value in tensors:
        checks_by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    if all(
        bool(torch.stack(device_checks).all().item()) for device_checks in checks_by_device.values()
    ):
        return
    for label, value in tensors:
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite {label}")
    raise RuntimeError("non-finite E2E optimizer state")


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
    gs: float
    rd: float
    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float
    brier: float
    warm_reference_std: float | None = None
    warm_reference_auprc: float | None = None
    residual_ratio: float | None = None
    topology_gradient_norm: float | None = None


def select_e2e_checkpoint(
    records: Sequence[E2ECheckpointRecord],
    arm: E2EArmName,
) -> E2ECheckpointRecord | None:
    """Select by mean rank over AUPRC and all five topology metrics.

    Delegates to :func:`src.eval.checkpoint_selection.select_checkpoint`
    (AUPRC↑, GS↑, RD→1, degree/clustering/spectral MMD↓, ties on higher
    AUPRC then later epoch). There is still no eligibility predicate:
    whether the selected checkpoint is scientifically usable remains an
    owner-side judgement made from ``metrics.jsonl``. ``arm`` is accepted
    only to keep one call signature across arms.
    """
    del arm
    by_epoch = {record.epoch: record for record in records}
    selected = select_checkpoint(
        [
            CheckpointCandidate(
                epoch=record.epoch,
                auprc=record.auprc,
                topology=TopologyValidationMetrics(
                    gs=record.gs,
                    rd=record.rd,
                    degree_mmd=record.degree_mmd,
                    clustering_mmd=record.clustering_mmd,
                    spectral_mmd=record.spectral_mmd,
                ),
            )
            for record in records
        ]
    )
    return None if selected is None else by_epoch[selected.epoch]


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
        choices=("formal", "diagnostic"),
        default=None,
        help=(
            "E2E execution context; defaults to formal and does not alter the config hash. "
            "'diagnostic' permits explicitly configured held-out-truth inputs and never "
            "publishes formal artifacts"
        ),
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
_E2E_TRAIN_SIDE_INPUTS = (
    "split.pkl",
    "train_edges.txt",
    "val_edges.txt",
    "train_graph.pkl",
    "positive_edges.txt",
)
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
    cfg: EgoConfig,
    pack_dir: Path,
    *,
    cold_cache: bool,
    temp_prefix: str = "",
    val_region_params: ValRegionParams | None = None,
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
        val_region_params: V_val derivation parameters; defaults to
            `ValRegionParams()`. The test seam small toy fixtures override.

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
    # Derive the split before any cache is written: this reads every
    # train-side input `derive_val_region_split` needs and so fails before the
    # pack directory (or raw-token pack) exists if any of them is unusable.
    split = _derive_e2e_val_region_split(
        cfg, strategy_dir, val_region_params=val_region_params or ValRegionParams()
    )
    # Call-time import preserves the pack builder/validator monkeypatch seam.
    from src.data import packed_features

    e2e_model_cfg = E2EConfig.from_mapping(cfg.model.config)
    n_ground = e2e_model_cfg.generator.n_ground
    manifest_path = pack_dir / _PACK_MANIFEST_FILENAME
    # The pack carries data-universe identity rather than execution-stage
    # identity. Grounding caches keep their own `role_universe` internally.
    role: Literal["V_val"] = _E2E_VALIDATION_ROLE
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
        train_nodes = sorted(split.train_nodes)
        validation_nodes = sorted(split.v_val)
        # V_val is a subset of the training universe by construction (cross-
        # boundary edges stay legal training-side signal), so the operative F0
        # set is exactly the training universe -- no separate union is needed
        # the way the retired V_fit/V_hold disjoint split required.
        operative = train_nodes
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
            role_universe="train",
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
    training_positives: Sequence[tuple[str, str]],
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
        training_positives: Canonical shared training positives (self-pairs included).
        sampler: The pinned negative sampler.
        negative_ratio: Negatives per positive.
        seed: Base seed.
        epoch: 1-based epoch.
        rank: Rank index.
        world_size: Rank count.

    Returns:
        Row list ``(u, v, label)`` for this epoch/rank.
    """
    positives = sorted(training_positives)
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
        train_nodes: Sorted train-side node ids with F0 rows (the complete
            training universe; V_val is a subset of it, not disjoint).
        training_positives: Canonical shared training positives (self-pairs included).
        val_pairs: Classification-validation pairs (`ValRegionSplit.val_cls_pairs`).
        val_labels: Aligned validation labels.
        f0: Shape ``(N, d)`` float32 CPU matrix.
        node_index: Node id -> `f0` row.
        grounding_index: Shape ``(n_train, n_g)`` int64 rows into `f0` for each
            train node's pool, aligned with `train_nodes`.
        train_pos: Node id -> position in `train_nodes`.
        target_builder: The `EgoTargetBuilder` over ``G_struct``.
        sampler: The pinned negative sampler.
        rho_train: Shared training-topology edge density (spec Sec 9.3).
        val_split: The derived V_val region split, or `None` for toy fixtures
            that build validation pairs by hand.
        val_topology_reference: The once-per-run topology-validation reference
            built from `val_split`, or `None` when `val_split` is absent.
        val_ball_union: The exact once-per-run ball-union pair universe, built
            from `val_split`, or `None` when `val_split` is absent.
        val_cascade_topology_reference: Reference restricted to the fixed
            cascade bucket panel, or the full reference when cascading is off.
        val_cascade_ball_union: Pair universe for the fixed cascade panel, or
            the full universe when cascading is off.
        feature_stats: Registered training-universe standardization constants.
    """

    train_nodes: list[str]
    training_positives: list[tuple[str, str]]
    val_pairs: list[tuple[str, str]]
    val_labels: NDArray[np.int8]
    f0: torch.Tensor
    node_index: dict[str, int]
    grounding_index: NDArray[np.int64]
    train_pos: dict[str, int]
    target_builder: EgoTargetBuilder
    sampler: NegativeSampler
    rho_train: float
    val_split: ValRegionSplit | None = None
    val_topology_reference: ValTopologyReference | None = None
    val_ball_union: ValBallUnionUniverse | None = None
    val_cascade_topology_reference: ValTopologyReference | None = None
    val_cascade_ball_union: ValBallUnionUniverse | None = None
    validation_role: Literal["V_val"] | None = None
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


def _read_pair_rows(path: Path) -> list[tuple[str, str]]:
    r"""Read a plain tab-separated ``u\tv`` file (e.g. `positive_edges.txt`)."""
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            u, v = stripped.split("\t")
            pairs.append((u, v))
    return pairs


def _read_val_region_inputs(
    strategy_dir: Path, benchmark_root: Path, expected_missing_features: Sequence[str]
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]], frozenset[tuple[str, str]]]:
    """Read the four train-side inputs `derive_val_region_split` needs.

    Returns:
        ``(train_nodes_all, truth_edges, benchmark_negatives, global_positive_edges)``.

    Raises:
        FileNotFoundError: If ``val_edges.txt`` (the negative-rejection source)
            is absent, which would otherwise fail with a bare `open()` traceback
            instead of an actionable message.
        ValueError: If ``split.pkl`` carries no train node collection.
    """
    val_path = strategy_dir / "val_edges.txt"
    if not val_path.is_file():
        raise FileNotFoundError(
            f"required validation-positive rejection source is missing: {val_path}"
        )
    with (strategy_dir / "split.pkl").open("rb") as handle:
        split_payload = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    if not isinstance(split_payload, dict) or "train" not in split_payload:
        raise ValueError("split.pkl must contain a train node collection")
    train_nodes_all = sorted(
        set(cast(Sequence[str], split_payload["train"])) - set(expected_missing_features)
    )
    with (strategy_dir / "train_graph.pkl").open("rb") as handle:
        train_graph = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    truth_edges = list(train_graph.edges())
    train_pairs, train_labels = _read_labeled_pairs(strategy_dir / "train_edges.txt")
    val_pairs, val_labels = _read_labeled_pairs(val_path)
    benchmark_negatives = [
        pair for pair, label in zip(train_pairs, train_labels, strict=True) if int(label) == 0
    ] + [pair for pair, label in zip(val_pairs, val_labels, strict=True) if int(label) == 0]
    global_positive_edges = frozenset(
        canonical_pair(u, v) for u, v in _read_pair_rows(benchmark_root / "positive_edges.txt")
    )
    return train_nodes_all, truth_edges, benchmark_negatives, global_positive_edges


def _derive_e2e_val_region_split(
    cfg: EgoConfig, strategy_dir: Path, *, val_region_params: ValRegionParams
) -> ValRegionSplit:
    """Derive the deterministic V_val split from this config's train-side inputs."""
    train_nodes_all, truth_edges, benchmark_negatives, global_positive_edges = (
        _read_val_region_inputs(
            strategy_dir, cfg.data.root / _BENCHMARK_SUBDIR, cfg.data.expected_missing_features
        )
    )
    return derive_val_region_split(
        train_nodes_all,
        truth_edges,
        benchmark_negatives,
        global_positive_edges,
        params=val_region_params,
    )


def _assemble_e2e_data(
    cfg: EgoConfig,
    generator_cfg: EgoStitchConfig,
    *,
    pack_dir: Path | None,
    val_region_params: ValRegionParams | None = None,
) -> EgoStitchData:
    """Assemble E2E training data from train-side files only, with the V_val boundary held.

    Raises:
        RuntimeError: When any benchmark input this assembly would open is held
            out -- checked before the first open, so nothing held out is read
            and no derived cache is written on the way to the raise. Also
            raised when a V_val-internal pair reaches the training positives
            (data-boundary fail-closed).
    """
    # [H] Path-scoped held-out boundary (design 2026-07-29 Sec 3.1), first
    # statement of the function. It sat at the *end* of the assembly until
    # Wave 3, which meant `split.pkl`, `train_edges.txt` and `val_edges.txt`
    # were already read and the F0 / feature-statistics / grounding caches were
    # already written by the time it fired -- a post-mortem, not a guard.
    strategy_dir = _assert_input_boundary(cfg)
    split = _derive_e2e_val_region_split(
        cfg, strategy_dir, val_region_params=val_region_params or ValRegionParams()
    )
    # One universe pair for both stages (design 2026-07-29 Sec 2): training is
    # always the full training universe, validation is always V_val. Nothing
    # here reads `run_kind` any more, which is exactly what makes the two
    # stages share a pack, a grounding cache and a `feature_stats_sha256`.
    run_kind = cfg.run_kind or "formal"
    role: Literal["V_val"] = _E2E_VALIDATION_ROLE
    train_nodes = sorted(split.train_nodes)
    validation_nodes: tuple[str, ...] = tuple(sorted(split.v_val))
    validation_positive_edges: tuple[tuple[str, str], ...] = split.val_positives
    # V_val is a subset of the training universe by construction (cross-
    # boundary edges stay legal training-side signal), so the F0 universe this
    # assembly needs is exactly the training universe -- no separate union of
    # a disjoint fit/hold split.
    allowed_nodes = train_nodes

    store = FeatureStore(cfg.data.root / _FEATURES_SUBDIR)
    f0_cache = (pack_dir / _PACK_F0_FILENAME) if pack_dir is not None else cfg.data.f0_cache
    f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, node_index = build_f0_matrix(
        store,
        allowed_nodes,
        cache_path=f0_cache,
        allow_cache_subset=False,
    )
    train_rows = np.asarray(
        matrix.numpy()[[node_index[node] for node in train_nodes]], dtype=np.float32
    )
    feature_stats_cache = (
        (pack_dir / _PACK_FEATURE_STATS_FILENAME)
        if pack_dir is not None
        else cfg.data.f0_cache.with_name(_PACK_FEATURE_STATS_FILENAME)
    )
    feature_stats = feature_stats_for_universe(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        train_nodes,
        cache_path=feature_stats_cache,
    )
    grounding_cache = (
        (pack_dir / _PACK_GROUNDING_FILENAME) if pack_dir is not None else cfg.data.grounding_cache
    )
    pool = build_grounding_pool(
        train_rows,
        train_nodes,
        n_ground=generator_cfg.n_ground,
        role_universe="train",
        cache_path=grounding_cache,
    )
    grounding_index = np.asarray(
        [[node_index[neighbor] for neighbor in pool[node]] for node in train_nodes], dtype=np.int64
    )
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
    g_train = split.build_training_graph()
    target_builder = EgoTargetBuilder(
        g_train,
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        pool,
        slots=generator_cfg.slots,
    )
    degrees = {node: int(g_train.degree(node)) for node in train_nodes}
    # The truth-edge set is train_graph.pkl's edge set, expressed once: the
    # split partitions it disjointly into `training_positives`/`val_positives`,
    # so their union reconstructs it exactly. Rejecting every truth edge --
    # not only the training-side ones -- keeps a sampled negative from
    # secretly being a V_val-internal positive.
    truth_edges = frozenset(split.training_positives) | frozenset(split.val_positives)
    sampler = NegativeSampler(
        train_nodes, degrees, truth_edges, forbidden_internal_nodes=split.v_val
    )
    n_train = len(train_nodes)
    rho_train = g_train.number_of_edges() / (math.comb(n_train, 2) + n_train)
    forbidden_files_absent = {
        name: not (strategy_dir / name).exists() for name in _HELD_OUT_FILENAMES
    }
    v_val_set = split.v_val
    n_val_internal_rows_in_training = sum(
        1 for u, v in split.training_positives if u in v_val_set and v in v_val_set
    )
    cross_boundary_edge_count = sum(
        1 for u, v in split.training_positives if (u in v_val_set) != (v in v_val_set)
    )
    audit: dict[str, object] = {
        "run_kind": run_kind,
        "validation_role": role,
        "training_feature_nodes_sha256": hashlib.sha256(
            "".join(f"{node}\n" for node in train_nodes).encode()
        ).hexdigest(),
        "validation_feature_nodes_sha256": hashlib.sha256(
            "".join(f"{node}\n" for node in validation_nodes).encode()
        ).hexdigest(),
        "forbidden_files_absent": forbidden_files_absent,
        "n_val_internal_rows_in_training": n_val_internal_rows_in_training,
        "cross_boundary_edge_count": cross_boundary_edge_count,
        "training_feature_stats_sha256": feature_stats.digest,
        "training_feature_stats_universe_sha256": feature_stats.node_ids_sha256,
        "training_feature_stats_rows": feature_stats.n_rows,
    }
    if n_val_internal_rows_in_training != 0:
        # [H] Data-boundary fail-closed: `ValRegionSplit` guarantees this is
        # zero by construction, so tripping this is a real corruption, not a
        # reachable steady state.
        raise RuntimeError(
            f"{n_val_internal_rows_in_training} V_val-internal pair(s) reached "
            "the training positives (data-boundary violation)"
        )
    if audit["training_feature_stats_universe_sha256"] != audit["training_feature_nodes_sha256"]:
        raise RuntimeError(
            "feature standardization statistics were computed over a universe "
            "other than the training universe"
        )
    val_topology_reference = build_val_topology_reference(split)
    val_ball_union = val_ball_union_universe(split)
    topology_cfg = cfg.topology_validation
    if topology_cfg.full_every_epochs == 1:
        val_cascade_topology_reference = val_topology_reference
        val_cascade_ball_union = val_ball_union
    else:
        cascade_buckets: dict[int, list[set[str]]] = {}
        for size, balls in split.buckets.items():
            if len(balls) < topology_cfg.cascade_buckets_per_size:
                raise ValueError(
                    "topology_validation.cascade_buckets_per_size exceeds the "
                    f"available bucket count for size {size}: "
                    f"{topology_cfg.cascade_buckets_per_size} > {len(balls)}"
                )
            cascade_buckets[size] = balls[: topology_cfg.cascade_buckets_per_size]
        cascade_split = replace(split, buckets=cascade_buckets)
        val_cascade_topology_reference = build_val_topology_reference(cascade_split)
        val_cascade_ball_union = val_ball_union_universe(cascade_split)
    data = EgoStitchData(
        train_nodes=train_nodes,
        training_positives=sorted(split.training_positives),
        val_pairs=split.val_cls_pairs,
        val_labels=np.asarray(split.val_cls_labels, dtype=np.int8),
        f0=matrix,
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(train_nodes)},
        target_builder=target_builder,
        sampler=sampler,
        rho_train=rho_train,
        feature_stats=feature_stats,
        val_split=split,
        val_topology_reference=val_topology_reference,
        val_ball_union=val_ball_union,
        val_cascade_topology_reference=val_cascade_topology_reference,
        val_cascade_ball_union=val_cascade_ball_union,
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
    val_region_params: ValRegionParams | None = None,
) -> EgoStitchData:
    """Assemble the full training data bundle from the frozen artifacts.

    Args:
        cfg: The validated worker config.
        pack_dir: DDP pack directory (its F0/grounding caches win); ``None``
            uses ``cfg.data.f0_cache`` / ``cfg.data.grounding_cache``.
            The node stream uses the internal trainable generator and its
            registered rev-3.1 grounding/loss-calibration fields.
        val_region_params: V_val derivation parameters; defaults to
            `ValRegionParams()`. The test seam small toy fixtures override.

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
        n_ground=e2e_model_cfg.generator.n_ground,
        tau_adj=e2e_model_cfg.generator.tau_adj,
        tau_div=e2e_model_cfg.generator.tau_div,
        l_gate_pos_weight=e2e_model_cfg.generator.l_gate_pos_weight,
    )
    return _assemble_e2e_data(
        cfg, generator_cfg, pack_dir=pack_dir, val_region_params=val_region_params
    )


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
    """Compute pair-edge-masked `L_rel` targets for every positive or negative pair."""
    targets = torch.zeros(len(rows), 2, dtype=torch.float32)
    for row, (node_u, node_v, _) in enumerate(rows):
        if node_u not in graph or node_v not in graph:
            raise ValueError("relational target pair contains a node outside G_train")
        neighbors_u = set(graph.neighbors(node_u))
        neighbors_v = set(graph.neighbors(node_v))
        neighbors_u.discard(node_v)
        neighbors_v.discard(node_u)
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
        generator_supervision: bool = True,
        relational_supervision: bool = True,
    ) -> None:
        self._cfg = cfg
        self._model_cfg = model_cfg
        self._data = data
        self._node_batch = node_batch
        self._rank = rank
        self._world = world_size
        self._generator_supervision = generator_supervision
        self._relational_supervision = relational_supervision
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
            raise RuntimeError(
                f"training feature/token read escaped the training universe: {sorted(invalid)[:3]}"
            )
        self.training_nodes_read.update(nodes)
        self._record_training_rows([self._data.node_index[node] for node in nodes])

    def _record_training_rows(self, rows: Sequence[int]) -> None:
        invalid = set(rows) - self._allowed_training_rows
        if invalid:
            raise RuntimeError(
                f"training F0 read escaped the training universe: {sorted(invalid)[:3]}"
            )
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
        """Build positive-real-row targets with explicit queried-partner leave-one-out."""
        batch = len(rows)
        slots = self._model_cfg.slots
        input_dim = self._model_cfg.input_dim
        targets: dict[str, torch.Tensor] = {}
        for side in ("i", "j"):
            targets[f"target_features_{side}"] = torch.zeros(
                batch, slots, input_dim, dtype=torch.float32
            )
            targets[f"target_mult_{side}"] = torch.zeros(batch, slots, dtype=torch.float32)
            targets[f"target_adj_{side}"] = torch.zeros(batch, slots, slots, dtype=torch.float32)
            targets[f"target_mask_{side}"] = torch.zeros(batch, slots, dtype=torch.bool)
            targets[f"target_node_index_{side}"] = torch.full((batch, slots), -1, dtype=torch.long)

        cached: dict[tuple[str, str], EgoTargets] = {}
        for row_index, (node_i, node_j, label) in enumerate(rows):
            if row_index >= true_rows or label != 1:
                continue
            for side, node_id, excluded_id in (
                ("i", node_i, node_j),
                ("j", node_j, node_i),
            ):
                cache_key = (node_id, excluded_id)
                node_targets = cached.get(cache_key)
                if node_targets is None:
                    node_targets = self._data.target_builder.build(
                        [node_id],
                        _edge_target_rng(node_id, seed=self._cfg.seed, epoch=epoch),
                        exclude_neighbors=[excluded_id],
                    )
                    cached[cache_key] = node_targets
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
        filler = (self._data.train_nodes[0], self._data.train_nodes[0], 0)
        if true_rows == 0:
            padded = [filler]
        while len(padded) < pad_to:
            padded.append(filler)
        self._record_training_nodes([node for u, v, _ in padded for node in (u, v)])
        idx_i = torch.tensor([self._data.node_index[u] for u, _, _ in padded], dtype=torch.long)
        idx_j = torch.tensor([self._data.node_index[v] for _, v, _ in padded], dtype=torch.long)
        # F0 row identities for each endpoint (Wave-1 oracle-scaffold
        # addendum): `OracleStructGenerator` needs to know *which* real node
        # it is encoding to look up its scaffold, an identity `x_i`/`x_j`
        # alone do not carry. Filler rows use one authorized training-universe
        # self-pair, keeping features, row identities, and `is_self` mutually
        # consistent; `edge_mask` still excludes them from every loss downstream.
        node_row_i = idx_i.clone()
        node_row_j = idx_j.clone()
        edge: dict[str, torch.Tensor] = {
            "x_i": self._data.f0[idx_i],
            "x_j": self._data.f0[idx_j],
            "node_row_i": node_row_i,
            "node_row_j": node_row_j,
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
        if self._generator_supervision:
            edge["ground_i"] = self._ground_rows(endpoints_u)
            edge["ground_j"] = self._ground_rows(endpoints_v)
            # Same-index-space grounding ids for both endpoints (spec Sec 13.18).
            edge["ground_id_i"] = self._ground_pool_rows(endpoints_u)
            edge["ground_id_j"] = self._ground_pool_rows(endpoints_v)
            edge.update(self._edge_target_tensors(padded, true_rows=true_rows, epoch=epoch))
        else:
            # Null and oracle generators ignore grounding features. Preserve the
            # shared generator call signature without gathering or transferring
            # B x n_ground x d feature tensors that no component consumes.
            empty_shape = (len(padded), 0, self._model_cfg.input_dim)
            edge["ground_i"] = torch.empty(empty_shape, dtype=self._data.f0.dtype)
            edge["ground_j"] = torch.empty(empty_shape, dtype=self._data.f0.dtype)
        if self._relational_supervision:
            edge["rel_target"] = relational_pair_targets(self._data.target_builder.graph, padded)
        return edge, true_rows

    def epoch_batches(
        self, epoch: int, *, rows_per_rank: Sequence[int], steps: int
    ) -> Iterator[_CompositeBatch]:
        """Yield the epoch's composite batches for this rank."""
        edge_rows = enumerate_edge_stream(
            self._data.training_positives,
            self._data.sampler,
            negative_ratio=self._cfg.data.negative_ratio,
            seed=self._cfg.seed,
            epoch=epoch,
            rank=self._rank,
            world_size=self._world,
        )
        edge_batch = self._cfg.data.edge_batch
        for step in range(steps):
            node: dict[str, torch.Tensor] = {}
            if self._generator_supervision:
                nodes = self._next_nodes()
                targets = self._data.target_builder.build(
                    nodes,
                    np.random.default_rng((self._cfg.seed, epoch, step, self._rank, 0x7A)),
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
            f0_rows = 2 * edge["x_i"].shape[0]
            if self._generator_supervision:
                f0_rows += (
                    node["x"].shape[0]
                    + node["ground_x"].shape[0] * node["ground_x"].shape[1]
                    + node["target_features"].shape[0] * node["target_features"].shape[1]
                    + 2 * edge["x_i"].shape[0]  # both edge grounding gathers
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


@dataclass
class _KdRuntime:
    """Frozen Gate A teacher encoder plus KD loss weights.

    Deliberately not an ``nn.Module`` field of `_CompositeStep`: a dataclass
    attribute stays out of the wrapped module tree, so DDP never syncs the
    teacher, `_cpu_state_dict` never persists it, and `accelerator.prepare`
    never touches it -- every rank builds it identically from the same
    checkpoint file and `_build_kd_runtime` moves it to the device itself.
    """

    teacher_encoder: GraphEncoder
    lambda_seed: float
    lambda_gram: float


def _build_kd_runtime(
    distill: EgoDistillConfig, model: EgoStitchModel, device: torch.device
) -> _KdRuntime:
    """Load the frozen full-ego-oracle teacher encoder and prime the student.

    The teacher is the published ``full_ego_oracle`` + ``grit_gmt``
    checkpoint's encoder, rebuilt at its structural input width (5 role
    channels, 1 relation) and kept fp32; its KD targets are computed with
    autocast disabled (`_CompositeStep._kd_terms`), matching the fp32 teacher
    convention of `src.distill.teacher_targets`. Seed KD is projection-free,
    so the student's encoder section and ``d_model`` must equal the
    teacher's exactly; with `EgoDistillConfig.warm_start_readout` the
    teacher's ``project``/``readout`` weights are copied into the student so
    seed ``k`` starts with the teacher's seed-``k`` semantics.

    Also flips the features generator's teacher-view stash on: training is
    the only consumer of ``aux["teacher_x"]``/``aux["teacher_adj"]``, and
    scoring must not pay for tensors nobody reads.
    """
    generator = model.generator
    if not isinstance(generator, FullEgoFeaturesGenerator):
        raise ValueError("distill requires the full_ego_features generator")
    if model.cfg.encoder.name != "grit_gmt":
        raise ValueError("Gate A seed KD requires a grit_gmt student encoder")
    checkpoint = cast(
        Mapping[str, object],
        torch.load(distill.teacher_checkpoint, map_location="cpu", weights_only=True),
    )
    missing_keys = [
        key for key in ("model_state", "model_family", "model_config") if key not in checkpoint
    ]
    if missing_keys:
        raise ValueError(
            f"teacher checkpoint {distill.teacher_checkpoint} is missing keys {missing_keys}"
        )
    if checkpoint["model_family"] != _EGOSTITCH_E2E_FAMILY:
        raise ValueError("teacher checkpoint must be an egostitch_e2e model")
    teacher_cfg = E2EConfig.from_mapping(cast(Mapping[str, object], checkpoint["model_config"]))
    if teacher_cfg.generator.name != "full_ego_oracle":
        raise ValueError("teacher checkpoint generator must be full_ego_oracle")
    if teacher_cfg.encoder.name != "grit_gmt":
        raise ValueError("Gate A seed KD requires a grit_gmt teacher encoder")
    if (
        teacher_cfg.encoder != model.cfg.encoder
        or teacher_cfg.classifier.d_model != model.cfg.classifier.d_model
    ):
        raise ValueError(
            "teacher and student encoder sections must match exactly (projection-free "
            f"seed KD): teacher {teacher_cfg.encoder!r} at d_model "
            f"{teacher_cfg.classifier.d_model}, student {model.cfg.encoder!r} at d_model "
            f"{model.cfg.classifier.d_model}"
        )
    state = cast(dict[str, torch.Tensor], checkpoint["model_state"])
    encoder_state = {
        key[len("encoder.") :]: value for key, value in state.items() if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("teacher checkpoint carries no encoder parameters")
    teacher_in_dim, teacher_relations = FullOracleGenerator().graph_dims()
    teacher_encoder = build_encoder(
        teacher_cfg.encoder,
        in_dim=teacher_in_dim,
        num_relations=teacher_relations,
        d_model=teacher_cfg.classifier.d_model,
    )
    teacher_encoder.load_state_dict(encoder_state, strict=True)
    if distill.warm_start_readout:
        student_encoder = model.encoder
        assert student_encoder is not None, "full_ego_features always constructs an encoder"
        readout_state = {
            key: value
            for key, value in encoder_state.items()
            if key.startswith(("project.", "readout."))
        }
        if not readout_state:
            raise ValueError("teacher encoder carries no project/readout weights to warm-start")
        load_result = student_encoder.load_state_dict(readout_state, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"warm-start keys absent on the student encoder: {load_result.unexpected_keys[:5]}"
            )
    teacher_encoder.to(device=device, dtype=torch.float32)
    teacher_encoder.eval()
    teacher_encoder.requires_grad_(False)
    generator.set_stash_teacher_view(True)
    return _KdRuntime(
        teacher_encoder=teacher_encoder,
        lambda_seed=distill.lambda_seed,
        lambda_gram=distill.lambda_gram,
    )


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

    def __init__(
        self, model: EgoStitchModel, world_size: int, *, kd: _KdRuntime | None = None
    ) -> None:
        super().__init__()
        self.model = model
        self.world_size = world_size
        # Plain attribute, not a submodule: see `_KdRuntime`'s docstring.
        self._kd = kd

    def _kd_terms(
        self,
        graph: ImaginedGraph | None,
        embedding_ab: GraphEmbedding | None,
        embedding_ba: GraphEmbedding | None,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local Gate A KD sums and their effective row counts.

        The teacher graph is rebuilt from the features generator's stashed
        structural view over the *same* padded node layout the student just
        encoded, so node tokens align positionally and the PMA seed tokens
        align seed-for-seed. Both orientations are matched (AB student to AB
        teacher, BA to BA) and averaged. The whole computation runs with
        autocast disabled: the frozen teacher forwards in fp32 under
        ``no_grad``, and the student tokens are upcast so the cosine/Gram
        reductions never quantize to the bf16 ulp grid.
        """
        kd = self._kd
        assert kd is not None
        if graph is None or embedding_ab is None or embedding_ba is None:
            raise RuntimeError(
                "KD requires the topology pathway: the features generator must emit a "
                "graph and both orientation embeddings on every training step"
            )
        aux = graph.aux
        if "teacher_x" not in aux or "teacher_adj" not in aux:
            raise RuntimeError(
                "KD requires the features generator's stashed teacher view "
                "(set_stash_teacher_view(True) at startup)"
            )
        teacher_graph = FullEgoGraph(
            x=aux["teacher_x"],
            adj=aux["teacher_adj"],
            mask=graph.mask,
            aux={"plan": aux["plan"], "log_plan": aux["log_plan"]},
            directed=True,
        )
        nodes = graph.num_nodes
        node_mask = graph.mask > 0.0
        live = edge_mask.to(dtype=torch.bool)
        with torch.autocast(device_type=teacher_graph.x.device.type, enabled=False):
            with torch.no_grad():
                teacher_ab = kd.teacher_encoder(teacher_graph)
                teacher_ba = kd.teacher_encoder(teacher_graph.swapped())
            seed = 0.5 * (
                kd_set_seed_loss(
                    embedding_ab.tokens[:, nodes:].float(), teacher_ab.tokens[:, nodes:], live
                )
                + kd_set_seed_loss(
                    embedding_ba.tokens[:, nodes:].float(), teacher_ba.tokens[:, nodes:], live
                )
            )
            gram = 0.5 * (
                kd_set_gram_loss(
                    embedding_ab.tokens[:, :nodes].float(),
                    teacher_ab.tokens[:, :nodes],
                    node_mask,
                    live,
                )
                + kd_set_gram_loss(
                    embedding_ba.tokens[:, :nodes].float(),
                    teacher_ba.tokens[:, :nodes],
                    node_mask,
                    live,
                )
            )
        live_rows = live.to(dtype=seed.dtype).sum()
        gram_live_rows = (
            (live & (node_mask.to(dtype=torch.bool).sum(dim=1) >= 2)).to(dtype=gram.dtype).sum()
        )
        return seed * live_rows, gram * gram_live_rows, live_rows, gram_live_rows

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
        if self.model.cfg.classifier.permanent_null == "none":
            branch_masks = sample_branch_masks(
                edge["label"].shape[0],
                self.model.cfg.classifier.p_topo,
                generator=_seeded_generator(seed, epoch, step),
                device=edge["label"].device,
            )
        else:
            branch_masks = masks_for_null(
                self.model.cfg.classifier.permanent_null,
                edge["label"].shape[0],
                edge["label"].device,
            )
        edge_view = _e2e_edge_view(edge)
        optional_auxiliary_keys = {
            "target_features_i": "target_features_a",
            "target_features_j": "target_features_b",
            "target_mult_i": "target_mult_a",
            "target_mult_j": "target_mult_b",
            "target_adj_i": "target_adj_a",
            "target_adj_j": "target_adj_b",
            "target_mask_i": "target_mask_a",
            "target_mask_j": "target_mask_b",
            "target_node_index_i": "target_node_index_a",
            "target_node_index_j": "target_node_index_b",
            "rel_target": "rel_target",
        }
        edge_view.update(
            {
                destination: edge[source]
                for source, destination in optional_auxiliary_keys.items()
                if source in edge
            }
        )
        edge_view.update(
            {
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
        # the `full`/`p0`/`no_l_rel`/`row_layernorm` arms (registry-driven
        # swapping is P3), whose `stitch` only ever produces a
        # `StitchedGraph`. The oracle-scaffold and null generators are the
        # registry's other members (design 2026-08-02 §3, §8, §12 P3;
        # Wave-1 oracle-scaffold addendum): `NullGenerator.stitch` always
        # returns `None`, so `EgoStitchModel.forward` omits `"graph"`/
        # `"embedding_ab"` from its output entirely rather than emitting a
        # `None` value (`composite.py`'s `if graph is not None and
        # embedding_ab is not None`) -- `.get(...)` (not `[...]`) is what
        # makes that omission legal here instead of a `KeyError`.
        graph = cast(StitchedGraph | None, edge_output.get("graph"))
        embedding_ab = cast(GraphEmbedding | None, edge_output.get("embedding_ab"))

        # Every key `NeighborhoodGenerator.auxiliary_losses`/
        # `GraphEncoder.auxiliary_losses` read is either a `node`-stream key
        # (unsuffixed) or an `edge_view`-stream key (`_a`/`_b`-suffixed or
        # distinctly named), so the merge below is a disjoint union -- see
        # `generator/egostitch.py:auxiliary_losses`'s docstring for the exact
        # key inventory. `NullGenerator.auxiliary_losses`/a parameter-free
        # oracle generator's `auxiliary_losses` both accept `graph is None`
        # and return `{}` -- nothing was imagined, so nothing needs
        # supervision -- and the encoder is skipped outright when absent
        # (`self.model.encoder is None`, the null-generator arm's design
        # §3.4 invariant) rather than reached through `_require_encoder`,
        # which exists precisely to raise on that condition for callers that
        # are not prepared to tolerate it.
        auxiliary_batch = {**node, **edge_view}
        generator_losses = self.model.generator.auxiliary_losses(graph, auxiliary_batch)
        encoder = self.model.encoder
        encoder_losses = (
            encoder.auxiliary_losses(embedding_ab, auxiliary_batch)
            if encoder is not None and embedding_ab is not None
            else {}
        )
        # A component that returns no loss for a given term (oracle: every
        # term, by construction; null: every term) contributes an honest zero
        # rather than a missing key -- `stage1_total`/`_recon_total`
        # (`generator/losses.py`) index several of these unconditionally, so
        # the dict below must always be complete regardless of arm.
        zero = logits.sum() * 0.0
        recon = {
            "feat": generator_losses.get("feat", zero),
            "exist": generator_losses.get("exist", zero),
            "mult": generator_losses.get("mult", zero),
            "slotadj": generator_losses.get("slotadj", zero),
            "gate": generator_losses.get("gate", zero),
            "ptr": generator_losses.get("ptr", zero),
            "div": generator_losses.get("div", zero),
            "align": generator_losses.get("align", zero),
            "rel": encoder_losses.get("rel_loss", zero),
        }

        # Gate telemetry belongs to the conditioning pathway, not the generator
        # family: oracle arms need it to show whether conditioning opens. The
        # readout is mode-safe -- `trunk.topo_xattn` is an empty ModuleList for
        # the film_logit/pooled_adapter rungs, yielding an empty list.
        if collect_diagnostics and self.model.encoder is not None:
            extra.update(_e2e_gate_tanh(self.model))
        edge_loss = (
            e2e_weighted_bce_with_logits(
                logits,
                edge["label"],
                edge["edge_mask"],
                world_size=self.world_size,
                global_denominator=cast(torch.Tensor | None, batch.get("edge_loss_denominator")),
                positive_weight=cast(float, batch.get("positive_weight", 5.0)),
            )
            if edge_active
            else logits.sum() * 0.0
        )

        total, parts = stage1_total(
            self.model.generator_cfg,
            family="egostitch_e2e",
            edge=edge_loss,
            recon=recon,
            deg=generator_losses.get("deg", zero),
            real_egostat=generator_losses.get("real_egostat", zero) * real_ssl_scale,
            real_gin=generator_losses.get("real_gin", zero) * real_ssl_scale,
            ssl_noise=generator_losses.get("ssl_noise", zero) * real_ssl_scale,
            ssl_pool=generator_losses.get("ssl_pool", zero) * real_ssl_scale,
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
            deg=generator_losses.get("deg", zero),
            real_egostat=generator_losses.get("real_egostat", zero) * real_ssl_scale,
            real_gin=generator_losses.get("real_gin", zero) * real_ssl_scale,
            ssl_noise=generator_losses.get("ssl_noise", zero) * real_ssl_scale,
            ssl_pool=generator_losses.get("ssl_pool", zero) * real_ssl_scale,
            recon_factors=cast(
                Mapping[str, float] | None,
                batch.get("recon_factors"),
            ),
        )
        kd_stats: dict[str, float] | None = None
        if self._kd is not None:
            kd_seed_sum, kd_gram_sum, kd_live_rows, kd_gram_live_rows = self._kd_terms(
                graph,
                embedding_ab,
                cast(GraphEmbedding | None, edge_output.get("embedding_ba")),
                edge["edge_mask"],
            )
            kd_seed = e2e_global_live_row_mean(
                kd_seed_sum,
                live_rows=kd_live_rows,
                world_size=self.world_size,
                global_denominator=cast(torch.Tensor | None, batch.get("kd_seed_denominator")),
            )
            kd_gram = e2e_global_live_row_mean(
                kd_gram_sum,
                live_rows=kd_gram_live_rows,
                world_size=self.world_size,
                global_denominator=cast(torch.Tensor | None, batch.get("kd_gram_denominator")),
            )
            kd_total = self._kd.lambda_seed * kd_seed + self._kd.lambda_gram * kd_gram
            total = total + kd_total
            # KD is representation supervision, not an edge-family member:
            # folded into `total` (and logged via `parts`) rather than into
            # the frozen stage1 families the gradient probe enumerates.
            parts["kd_seed"] = float(kd_seed.detach())
            parts["kd_gram"] = float(kd_gram.detach())
            parts["total"] = float(total.detach())
            kd_stats = {
                "seed_sum": float(kd_seed_sum.detach()),
                "gram_sum": float(kd_gram_sum.detach()),
                "live_rows": float(kd_live_rows.detach()),
                "gram_live_rows": float(kd_gram_live_rows.detach()),
            }
        parameter_anchor = (
            0.0
            * torch.stack(
                tuple(
                    parameter.sum()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                )
            ).sum()
        )
        total = total + parameter_anchor
        families = {name: family + parameter_anchor for name, family in families.items()}
        result: dict[str, object] = {
            "loss": total,
            "parts": parts,
        }
        if kd_stats is not None:
            result["kd_stats"] = kd_stats
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
    already match and pass through unchanged. Learned generators receive the
    full grounding features and ids. Zero-parameter oracle generators receive
    zero-width grounding tensors and omit the ids because they do not consume
    either input; `_CompositeStep.forward` adds the optional ids only when the
    batch contains them.

    ``node_row_i``/``node_row_j`` (Wave-1 oracle-scaffold addendum) pass
    through **unrenamed**, unlike every ``_i``/``_j`` -> ``_a``/``_b`` rename
    above: `EgoStitchModel._pair_node_states` (`composite.py`) reads them
    under exactly these names (``batch.get("node_row_i", ...)``/
    ``batch.get("node_row_j", ...)``) to resolve which real node
    `OracleStructGenerator.encode_node` is encoding -- an identity
    `NullGenerator`/`EgoStitchImagineGenerator` never need because they read
    only ``x``/``ground``.
    """
    view = {
        "emb_a": edge["emb_a"],
        "emb_b": edge["emb_b"],
        "len_a": edge["len_a"],
        "len_b": edge["len_b"],
        "x_a": edge["x_i"],
        "x_b": edge["x_j"],
        "ground_a": edge["ground_i"],
        "ground_b": edge["ground_j"],
        "node_row_i": edge["node_row_i"],
        "node_row_j": edge["node_row_j"],
        "is_self": edge["is_self"],
    }
    if "ground_id_i" in edge:
        view["ground_id_a"] = edge["ground_id_i"]
        view["ground_id_b"] = edge["ground_id_j"]
    return view


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
        "gate_topo_tanh": _values(_require_b0v31_classifier(model).trunk.topo_xattn),
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
        "grad_rms_trunk": list(_require_b0v31_classifier(model).trunk.parameters()),
        "grad_rms_ste": list(_require_encoder(model).parameters()),
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
    task_loss: float = float("nan")
    scale_telemetry: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    topology_scope: Literal["cascade", "full"] = "full"
    active_logits: NDArray[np.float32] = field(default_factory=lambda: np.empty(0, dtype="<f4"))


def _e2e_validation_grounding_rows(
    data: EgoStitchData,
    nodes: Sequence[str],
) -> NDArray[np.int64]:
    """Resolve role-specific validation grounding to global F0 row ids.

    The V_val pool is checked first, not `train_pos`: V_val is a subset of
    the training universe (unlike the retired disjoint V_fit/V_hold split),
    so every production validation node is *also* in `train_pos` and would
    otherwise always take the wrong branch, silently reading the train-role
    pool instead of the role-isolated V_val one the pack built for it.
    """
    validation_index = data.validation_grounding_index
    validation_pos = data.validation_pos or {}
    rows: list[NDArray[np.int64]] = []
    for node in nodes:
        if validation_index is not None and node in validation_pos:
            rows.append(validation_index[validation_pos[node]])
        elif node in data.train_pos:
            rows.append(data.grounding_index[data.train_pos[node]])
        else:
            raise RuntimeError(f"no role-specific grounding row for validation node {node!r}")
    return np.stack(rows).astype(np.int64, copy=False)


def _e2e_validation_node_batch(
    data: EgoStitchData,
    token_table: PackedFeatureTable,
    token_node_index: Mapping[str, int],
    nodes: Sequence[str],
    device: torch.device,
    *,
    generator_supervision: bool = True,
) -> dict[str, torch.Tensor]:
    """Build one unique-node validation encode batch."""
    packed_rows = torch.tensor([token_node_index[node] for node in nodes], dtype=torch.long)
    boundary = max(token_table.manifest.nodes[row].length for row in packed_rows.tolist())
    emb, length = token_table.gather_nodes(packed_rows, boundary)
    node_rows = torch.tensor([data.node_index[node] for node in nodes], dtype=torch.long)
    batch = {
        "emb": emb,
        "length": length,
        "x": data.f0[node_rows],
        # F0 row identity per node (Wave-1 oracle-scaffold addendum, mirrors
        # the edge-batch `node_row_i`/`node_row_j` addition): lets
        # `OracleStructGenerator` resolve which real node it is encoding
        # during validation, the same way `node_row_i`/`node_row_j` do for
        # the training-time edge stream.
        "node_rows": node_rows,
    }
    if generator_supervision:
        grounding_rows = _e2e_validation_grounding_rows(data, nodes)
        batch["ground"] = data.f0[torch.from_numpy(grounding_rows)]
        batch["ground_ids"] = torch.from_numpy(grounding_rows)
    else:
        batch["ground"] = torch.empty(len(nodes), 0, data.f0.shape[1], dtype=data.f0.dtype)
    return cast(dict[str, torch.Tensor], _to_device(batch, device))


def _fp32_cached_node_state(state: E2ENodeState, row: int) -> E2ENodeState:
    """Detach one encoded node while preserving integer identity fields exactly.

    ``slots``/``projected_x`` are ``None`` only for a generator whose
    ``encode_node`` allocates no ``GeneratorNodeState`` (``NullGenerator``,
    design §3.3) -- preserved as ``None`` rather than defaulted, since a
    null-generator arm's cache must stay legible as carrying no slot
    geometry, not a fabricated one.
    """
    length = state.length[row : row + 1].clone()
    true_length = int(length.item())
    slots = (
        SlotSet(
            *(
                value[row : row + 1].float().clone()
                if value.is_floating_point()
                else value[row : row + 1].clone()
                for value in state.slots
            )
        )
        if state.slots is not None
        else None
    )
    ground_ids = None
    if state.ground_ids is not None:
        ground_ids = state.ground_ids[row : row + 1].clone()
    projected_x = (
        state.projected_x[row : row + 1].float().clone() if state.projected_x is not None else None
    )
    return E2ENodeState(
        encoded=state.encoded[row : row + 1, :true_length].float().clone(),
        length=length,
        slots=slots,
        projected_x=projected_x,
        ground_ids=ground_ids,
    )


def _stack_cached_node_states(
    cache: Mapping[str, E2ENodeState], nodes: Sequence[str]
) -> E2ENodeState:
    """Reconstruct one pair-endpoint state batch from the per-rank cache.

    ``slots``/``projected_x`` are carried through as ``None`` when every
    state in the batch carries ``None`` (a null-generator arm), matching
    ``ground_ids``'s existing all-or-nothing check -- a cache mixing real and
    absent slot geometry within one batch is a real inconsistency, not a
    valid null-generator batch.
    """
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
    slots: SlotSet | None
    if any(state.slots is None for state in states):
        if not all(state.slots is None for state in states):
            raise RuntimeError("validation node cache has inconsistent slot state")
        slots = None
    else:
        slots = SlotSet(
            *(
                torch.cat(values)
                for values in zip(*(cast(SlotSet, state.slots) for state in states), strict=True)
            )
        )
    projected_x: torch.Tensor | None
    if any(state.projected_x is None for state in states):
        if not all(state.projected_x is None for state in states):
            raise RuntimeError("validation node cache has inconsistent projected features")
        projected_x = None
    else:
        projected_x = torch.cat([cast(torch.Tensor, state.projected_x) for state in states])
    return E2ENodeState(
        encoded=encoded,
        length=torch.cat([state.length for state in states]),
        slots=slots,
        projected_x=projected_x,
        ground_ids=ground_ids,
    )


def _e2e_validation_slice_rows(n_val: int) -> tuple[int, ...]:
    """Frozen global rows used by the E2E checkpoint-selection tie-break."""
    if n_val <= 0:
        return ()
    return tuple(range(max(1, math.ceil(0.01 * n_val))))


def _e2e_encode_node_cache(
    model: EgoStitchModel,
    data: EgoStitchData,
    token_table: PackedFeatureTable,
    token_node_index: Mapping[str, int],
    nodes: Sequence[str],
    accelerator: Accelerator,
    *,
    edge_batch: int,
    generator_supervision: bool,
) -> dict[str, E2ENodeState]:
    """Encode `nodes` once into a length-bucketed fp32 cache, no_grad.

    Shared by the classification and topology-universe validation passes in
    `_validate_epoch`, so every node is encoded exactly once per epoch rather
    than once per pass.
    """
    node_cache: dict[str, E2ENodeState] = {}
    with torch.no_grad():
        length_buckets: dict[int, list[str]] = {}
        for node in nodes:
            length = token_table.manifest.nodes[token_node_index[node]].length
            bucket = next(
                (boundary for boundary in (128, 256, 384, 512, 768, 1024) if length <= boundary),
                length,
            )
            length_buckets.setdefault(bucket, []).append(node)
        for bucket_nodes in length_buckets.values():
            for start in range(0, len(bucket_nodes), edge_batch):
                chunk_nodes = bucket_nodes[start : start + edge_batch]
                node_batch = _e2e_validation_node_batch(
                    data,
                    token_table,
                    token_node_index,
                    chunk_nodes,
                    accelerator.device,
                    generator_supervision=generator_supervision,
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
                        node_batch.get("ground_ids"),
                        node_batch["node_rows"],
                    )
                for offset, node in enumerate(chunk_nodes):
                    node_cache[node] = _fp32_cached_node_state(encoded, offset)
    return node_cache


def _score_val_universe_logits(
    model: EgoStitchModel,
    node_cache: Mapping[str, E2ENodeState],
    reference: ValTopologyReference,
    universe: ValBallUnionUniverse,
    accelerator: Accelerator,
    *,
    edge_batch: int,
) -> NDArray[np.float64] | None:
    """Score the exact ball-union active-arm logits with DDP coverage checks.

    Rank-strided over `universe`'s sampled-pair rows; every rank gathers and
    checks coverage identically (DDP fail-closed -- a partial gather must never
    silently assemble the topology graph from a subset of the universe). Active-arm logit only: no
    full/f_logit/dispersion columns, since only the assembled graph is read
    from this pass.

    Returns:
        The row-ordered ball-union logits on the main process; ``None`` elsewhere.

    Raises:
        ValueError: If the gathered rows do not cover the complete row set
            exactly once, on every rank.
    """
    u_idx = universe.u_idx
    v_idx = universe.v_idx
    n_rows = u_idx.shape[0]
    rank, world = accelerator.process_index, accelerator.num_processes
    row_ids = np.arange(rank, n_rows, world, dtype=np.int64)

    logits_out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, row_ids.shape[0], edge_batch):
            chunk_rows = row_ids[start : start + edge_batch]
            chunk_u = u_idx[chunk_rows]
            chunk_v = v_idx[chunk_rows]
            state_a = _stack_cached_node_states(node_cache, [reference.nodes[i] for i in chunk_u])
            state_b = _stack_cached_node_states(node_cache, [reference.nodes[i] for i in chunk_v])
            is_self = torch.from_numpy(chunk_u == chunk_v).to(accelerator.device)
            masks = (
                None
                if model.cfg.classifier.permanent_null == "none"
                else masks_for_null(
                    model.cfg.classifier.permanent_null, len(chunk_rows), accelerator.device
                )
            )
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                context = model.build_pair_context_from_states(state_a, state_b, is_self)
                active_logits = (
                    model.score_pair_context(context)
                    if masks is None
                    else model.score_pair_context(context, masks=masks)
                )
            logits_out.append(active_logits.float())

    local_logits = (
        torch.cat(logits_out) if logits_out else torch.zeros(0, device=accelerator.device)
    )
    local_row_ids = torch.from_numpy(row_ids).to(accelerator.device)
    padded_row_ids = accelerator.pad_across_processes(local_row_ids, dim=0, pad_index=-1)
    padded_logits = accelerator.pad_across_processes(local_logits, dim=0, pad_index=0.0)
    gathered_row_ids = accelerator.gather(padded_row_ids)
    gathered_logits = accelerator.gather(padded_logits)

    row_ids_np = gathered_row_ids.cpu().numpy()
    logits_np = gathered_logits.cpu().numpy().astype(np.float64)
    keep = row_ids_np >= 0
    row_ids_np = row_ids_np[keep]
    logits_np = logits_np[keep]
    # `validate_gathered_validation` is the generic train_b0 DDP coverage
    # check; it is labels-shaped by contract, so a zero dummy array stands in
    # for the label return this scoring pass has no use for.
    _, logits_sorted = validate_gathered_validation(
        row_ids=row_ids_np,
        labels=np.zeros_like(logits_np),
        logits=logits_np,
        expected_row_ids=np.arange(n_rows, dtype=np.int64),
    )
    if not accelerator.is_main_process:
        return None
    return logits_sorted


def _validate_epoch(
    model: EgoStitchModel,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    edge_batch: int,
    topk_fraction: float,
    token_table: PackedFeatureTable | None = None,
    token_node_index: Mapping[str, int] | None = None,
    topology_scope: Literal["cascade", "full"] = "full",
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

    A second, lean pass follows the classification one when
    `data.val_topology_reference`/`data.val_ball_union` are set (absent only
    for toy fixtures that hand-build `val_pairs` with no derived V_val
    region): it scores the exact deduplicated ball-union pairs U
    (`_score_val_universe_logits`) and feeds `val_region_topology_metrics` for
    sampled-only threshold selection and the five topology metrics. Both
    passes share one node-encoding cache built once over the complete V_val
    region.
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
    # The complete V_val region when one was derived; the toy-fixture fallback
    # (no `val_split`) keeps the old shard-local touched-node set, since those
    # fixtures build `val_pairs` by hand with no V_val region behind them.
    if topology_scope == "full":
        reference = data.val_topology_reference
        ball_union = data.val_ball_union
    else:
        reference = data.val_cascade_topology_reference
        ball_union = data.val_cascade_ball_union
    encode_nodes = (
        list(reference.nodes)
        if reference is not None
        else list(dict.fromkeys(node for row in shard_rows for node in data.val_pairs[row]))
    )

    def synchronize_device() -> None:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)

    synchronize_device()
    node_cache_started = time.monotonic()
    generator_supervision = isinstance(model.generator, EgoStitchImagineGenerator)
    # Shared by the classification pass below and the topology-universe pass
    # after it, so every V_val node is encoded exactly once per epoch.
    node_cache = _e2e_encode_node_cache(
        model,
        data,
        token_table,
        token_node_index,
        encode_nodes,
        accelerator,
        edge_batch=edge_batch,
        generator_supervision=generator_supervision,
    )
    synchronize_device()
    node_cache_seconds = time.monotonic() - node_cache_started

    values_out: list[torch.Tensor] = []
    # Validation may run inside an outer autocast context (for example the
    # CPU bf16 contract test).  `inference_mode` can then seed autocast's
    # weight cache with inference tensors that a later training forward tries
    # to save for backward.  `no_grad` preserves eval semantics without
    # contaminating the following optimizer step.
    with torch.no_grad():
        pair_scoring_started = time.monotonic()
        shard_rows.sort(
            key=lambda row: (
                max(
                    node_cache[data.val_pairs[row][0]].encoded.size(1),
                    node_cache[data.val_pairs[row][1]].encoded.size(1),
                ),
                row,
            )
        )
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
                if model.cfg.classifier.permanent_null == "none"
                else masks_for_null(
                    model.cfg.classifier.permanent_null,
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
                    full_logits if masks is None else model.score_pair_context(context, masks=masks)
                )
            # Slot-geometry telemetry (dispersion/scale) only exists for
            # `EgoStitchImagineGenerator`: the oracle-scaffold and null
            # generators carry no `SlotSet`/Sinkhorn plan at all (`context.plan
            # is None`, `state.slots is None`, design §3.4), so they emit NaN
            # columns here instead -- the aggregation below already treats an
            # all-NaN column as "no signal" for the five dispersion names
            # (falls back to 0.0) and reports NaN for the four scale names,
            # exactly the "explicit NaN columns" convention every other
            # generator-internals block in this module follows for these arms.
            if isinstance(model.generator, EgoStitchImagineGenerator):
                assert context.plan is not None
                slots_a = _require_slots(state_a)
                slots_b = _require_slots(state_b)
                dispersion_a = _e2e_dispersion_rows(
                    slots_a.pi,
                    slots_a.h,
                    slots_a.adj,
                    context.plan,
                )
                dispersion_b = _e2e_dispersion_rows(
                    slots_b.pi,
                    slots_b.h,
                    slots_b.adj,
                    context.plan,
                )
                dispersion_rows = {
                    name: (
                        0.5 * (dispersion_a[name] + dispersion_b[name])
                        if name in {"pi_slot_std", "h_pairwise_cosine_mean", "adj_offdiag_std"}
                        else dispersion_a[name]
                    )
                    for name in dispersion_a
                }
                for name in ("plan_row_entropy", "plan_rank1_marginal_residual"):
                    dispersion_rows[name] = dispersion_rows[name].masked_fill(is_self, torch.nan)
                scale_a = _e2e_scale_rows(slots_a.h, context.plan)
                scale_b = _e2e_scale_rows(slots_b.h, context.plan)
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
            else:
                nan_row = torch.full(
                    (len(rows),), float("nan"), dtype=torch.float32, device=accelerator.device
                )
                dispersion_rows = dict.fromkeys(
                    (
                        "pi_slot_std",
                        "h_pairwise_cosine_mean",
                        "adj_offdiag_std",
                        "plan_row_entropy",
                        "plan_rank1_marginal_residual",
                    ),
                    nan_row,
                )
                scale_rows = dict.fromkeys(
                    (
                        "plan_total_mass",
                        "plan_max_cell_fraction",
                        "h_norm_mean",
                        "h_pairwise_sqdist_mean",
                    ),
                    nan_row,
                )
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
    gather_seconds = time.monotonic() - gather_metrics_started

    # Lean topology-universe pass: active-arm logit only, over the exact
    # deduplicated ball-union pairs U. Runs on
    # every rank (its own rank-strided gather and exact-coverage check),
    # mirroring the classification pass's DDP-fail-closed discipline -- a
    # partial gather here would silently assemble the topology graph from a
    # subset of the row set.
    universe_started = time.monotonic()
    universe_logits = (
        _score_val_universe_logits(
            model, node_cache, reference, ball_union, accelerator, edge_batch=edge_batch
        )
        if reference is not None and ball_union is not None
        else None
    )
    universe_seconds = time.monotonic() - universe_started

    if was_training:
        model.train()
    if not accelerator.is_main_process:
        return None
    metrics_started = time.monotonic()
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
            float(np.mean(values[np.isfinite(values)])) if bool(np.isfinite(values).any()) else 0.0
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
    val_threshold = 0.0
    if reference is not None and ball_union is not None and universe_logits is not None:
        topology_result = val_region_topology_metrics(
            u_idx=ball_union.u_idx,
            v_idx=ball_union.v_idx,
            logits=universe_logits,
            reference=reference,
        )
        validation_topology = topology_result.metrics
        val_threshold = topology_result.threshold
    else:
        validation_topology = TopologyValidationMetrics(
            gs=0.0, rd=0.0, degree_mmd=0.0, clustering_mmd=0.0, spectral_mmd=0.0
        )
    fidelity = {
        "active_logit_std": active_std,
        "f_logit_std": f_std,
        "f_logit_auprc": f_metrics.auprc,
        "topology_delta_std": residual_std,
        "topology_delta_ratio": residual_std / max(f_std, 1e-12),
        "selection_tiebreak": 0.0,
        "gs_bfs": validation_topology.gs,
        "rd_bfs": validation_topology.rd,
        "degree_mmd_ratio": validation_topology.degree_mmd,
        "clustering_mmd_ratio": validation_topology.clustering_mmd,
        "spectral_mmd_ratio": validation_topology.spectral_mmd,
        "val_threshold": val_threshold,
        "topology_validation_full": float(topology_scope == "full"),
        "topology_scored_rows": float(0 if ball_union is None else ball_union.u_idx.size),
        "prevalence": float(np.mean(data.val_labels)),
        **dispersion_summary,
        **e2e_degree_decorrelation_telemetry(endpoint_degree, full_np - f_np),
    }
    active_metrics = compute_edge_metrics(data.val_labels.astype(np.int64), probs)
    gather_metrics_seconds = gather_seconds + (time.monotonic() - metrics_started)
    return _ValidationResult(
        metrics=active_metrics,
        fidelity=fidelity,
        task_loss=_stable_bce_with_logits(logits_np, data.val_labels.astype(np.float64)),
        scale_telemetry=scale_telemetry,
        timing={
            "node_cache_encode_seconds": node_cache_seconds,
            "pair_scoring_seconds": pair_scoring_seconds,
            "gather_metrics_seconds": gather_metrics_seconds,
            "val_universe_scoring_seconds": universe_seconds,
        },
        topology_scope=topology_scope,
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
    stop_epoch: int | None
    runtime_profile: dict[str, object]
    kendall_state: dict[str, object]


def _e2e_topology_validation_scope(
    epoch: int,
    total_epochs: int,
    config: EgoTopologyValidationConfig,
) -> Literal["cascade", "full"]:
    """Return the fixed two-resolution validation scope for one epoch."""
    if epoch <= 0 or total_epochs <= 0 or epoch > total_epochs:
        raise ValueError(f"invalid epoch position {epoch}/{total_epochs}")
    if config.full_every_epochs <= 0:
        raise ValueError("topology full-validation interval must be positive")
    if epoch % config.full_every_epochs == 0 or epoch == total_epochs:
        return "full"
    return "cascade"


def _cpu_state_dict(accelerator: Accelerator, wrapped: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the *inner* EgoStitch model's state dict."""
    inner = accelerator.unwrap_model(wrapped)
    model = inner.model if isinstance(inner, _CompositeStep) else inner
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _e2e_arm_name(model: EgoStitchModel) -> E2EArmName:
    return _e2e_arm_name_from_config(model.cfg)


def _e2e_arm_name_from_config(config: E2EConfig) -> E2EArmName:
    if config.generator.name == "null":
        return "null_generator"
    if config.generator.name == "oracle_struct":
        return "oracle"
    if config.generator.name == "full_ego_oracle":
        return "full_ego_oracle"
    if config.generator.name == "full_ego_features":
        return "full_ego_features"
    if config.classifier.permanent_null == "all_head":
        return "b0_e2e_f_only"
    if config.classifier.p_topo == 0.0:
        return "p0"
    if config.encoder.w_rel == 0.0:
        return "no_l_rel"
    if config.generator.feature_standardization == "row_layernorm":
        return "row_layernorm"
    return "full"


# --------------------------------------------------------- component narrowing
#
# The registry (design 2026-08-02 §3, §8, §12 P3) widened `EgoStitchModel`'s
# component attributes so a null generator is a legal composite: `.generator`
# is the abstract `NeighborhoodGenerator`, `.encoder` is `GraphEncoder | None`
# (`None` for `generator.name == "null"`, since nothing is ever imagined and
# there is nothing to encode -- composite.py), and `E2ENodeState.slots` /
# `.projected_x` are `SlotSet | None` / `Tensor | None` to match
# (`NullGenerator.encode_node` echoes its input back rather than allocating a
# `GeneratorNodeState`). Every function below this point that reads a real
# generator/encoder's internals or a node's slot geometry (family probes,
# gradient telemetry, the gate-tanh readout, the fidelity/dispersion
# validation summary, the degree-prior center) is reachable *only* under the
# current real-generator curriculum: `build_e2e_parameter_groups` already
# refuses to build optimizer groups for a null-generator arm (both its
# "generator" and "encoder" groups are empty -- P3 report), so nothing here
# runs for that arm today. These helpers narrow the type and raise the same
# clear message the empty-group refusal already gives, rather than let a
# silent `cast` paper over an invariant that is not actually universal.
def _require_encoder(model: EgoStitchModel) -> GraphEncoder:
    """Narrow `model.encoder` to a concrete `GraphEncoder`."""
    if model.encoder is None:
        raise RuntimeError(
            "this code path requires a real generator/encoder pair; "
            "a null-generator arm has no encoder to read"
        )
    return model.encoder


def _require_b0v31_classifier(model: EgoStitchModel) -> B0V31PairClassifier:
    """Narrow `model.classifier` to the concrete `B0V31PairClassifier`."""
    if not isinstance(model.classifier, B0V31PairClassifier):
        raise RuntimeError(
            f"this code path requires the b0_v31 classifier, got {type(model.classifier).__name__}"
        )
    return model.classifier


def _require_slots(state: E2ENodeState) -> SlotSet:
    """Narrow one cached node state's `slots` to a concrete `SlotSet`."""
    if state.slots is None:
        raise RuntimeError(
            "slot-geometry telemetry requires a real generator's SlotSet; "
            "a null-generator arm carries none"
        )
    return state.slots


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
    """Optimizer groups whose learning rate should be nonzero this step.

    Null-safe on ``model.encoder`` (``None`` for a null-generator arm, per
    design §3.4): an absent encoder is simply never "active" outside the
    edge-active phase, the same as one whose relational head is disabled.
    Parameter-free groups are excluded: treating an empty oracle-generator
    group as active would emit a false zero-gradient quality failure every
    step even though there is nothing to optimize.
    """
    groups: set[str] = set()
    if any(parameter.requires_grad for parameter in model.generator.parameters()):
        groups.add("generator")
    if (
        model.encoder is not None
        and any(parameter.requires_grad for parameter in model.encoder.parameters())
        and (model.encoder.rel_head is not None or phase.edge_active)
    ):
        groups.add("encoder")
    if phase.edge_active and any(
        parameter.requires_grad for parameter in model.classifier.parameters()
    ):
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
    micro_step: int,
    total_steps: int,
    device: torch.device,
    edge_loss_denominator: torch.Tensor | None = None,
    kd_seed_denominator: torch.Tensor | None = None,
    kd_gram_denominator: torch.Tensor | None = None,
) -> dict[str, object]:
    return {
        "node": _to_device(batch.node, device),
        "edge": _to_device(batch.edge, device),
        "edge_rows_global": batch.edge_rows_global,
        "edge_active": phase.edge_active,
        "recon_factors": e2e_recon_component_factors(
            step,
            total_steps,
            phase_a_fraction=cfg.training.phase_a_fraction if cfg.training is not None else 0.2,
            phase_b_fraction=cfg.training.phase_b_fraction if cfg.training is not None else 0.1,
        ),
        "real_ssl_scale": torch.tensor(phase.real_ssl_scale, device=device),
        "seed": cfg.seed,
        "epoch": epoch,
        "step": micro_step,
        "edge_loss_denominator": edge_loss_denominator,
        "kd_seed_denominator": kd_seed_denominator,
        "kd_gram_denominator": kd_gram_denominator,
        "positive_weight": (cfg.training.positive_weight if cfg.training is not None else 5.0),
    }


def _e2e_group_gradient_statistics(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute each group's fp64 squared norm and non-finite element count once."""
    device = next(
        (parameter.device for parameters in groups.values() for parameter in parameters),
        torch.device("cpu"),
    )
    statistics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, parameters in groups.items():
        grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
        if grads:
            squared = torch.stack([grad.double().square().sum() for grad in grads]).sum()
            nonfinite = torch.stack(
                [(~torch.isfinite(grad)).sum().to(dtype=torch.float64) for grad in grads]
            ).sum()
        else:
            squared = torch.zeros((), dtype=torch.float64, device=device)
            nonfinite = torch.zeros((), dtype=torch.float64, device=device)
        statistics[name] = (squared, nonfinite)
    return statistics


def _gather_e2e_group_gradient_statistics(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    accelerator: Accelerator,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Gather every group statistic in one collective."""
    local = _e2e_group_gradient_statistics(groups)
    names = tuple(groups)
    packed = torch.stack(
        [torch.stack((local[name][0].double(), local[name][1].double())) for name in names]
    )
    gathered = accelerator.gather(packed).reshape(-1, len(names), 2)
    squared = {name: gathered[:, index, 0] for index, name in enumerate(names)}
    nonfinite = {name: gathered[:, index, 1] for index, name in enumerate(names)}
    return local, squared, nonfinite


def _e2e_group_squared_norms(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
    accelerator: Accelerator,
) -> dict[str, torch.Tensor]:
    """Compatibility wrapper for diagnostic family probes."""
    _, squared, _ = _gather_e2e_group_gradient_statistics(groups, accelerator)
    return squared


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
    # A "recon"/"real"/"ssl" node-stream family is only a meaningful concept
    # for `EgoStitchImagineGenerator`: the oracle-scaffold and null generators
    # are both parameter-free/absent and return `{}` from `auxiliary_losses`,
    # so `_CompositeStep.forward` zero-fills every `recon` term for them
    # (spec: no generator-internals telemetry for those arms). Gating the
    # expected membership here, not just relaxing the finiteness check below,
    # keeps the probe from recording a spurious always-zero "generator recon"
    # entry for an arm that never actually trains a generator.
    is_imagine_generator = isinstance(inner.generator, EgoStitchImagineGenerator)
    expected: dict[str, set[str]] = {
        "classifier": {"edge"} if phase.edge_active else set(),
        "generator": (
            {"recon"} | ({"real", "ssl"} if phase.real_ssl_scale > 0.0 else set())
            if is_imagine_generator
            else set()
        ),
        "encoder": (
            {"recon"} if inner.encoder is not None and inner.encoder.rel_head is not None else set()
        ),
    }
    if phase.edge_active and arm != "b0_e2e_f_only" and is_imagine_generator:
        expected["generator"].add("edge")
    if phase.edge_active and arm != "b0_e2e_f_only" and inner.encoder is not None:
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
    """RMS telemetry from the current synchronized fixed-replay edge backward.

    ``grad_rms_ste`` reads 0.0 for a null-generator arm's absent encoder
    (design §3.4) rather than raising: there is no submodule to measure, and
    an empty parameter tuple is the honest representation of that fact.
    """
    submodules: dict[str, Sequence[torch.nn.Parameter]] = {
        "grad_rms_trunk": tuple(_require_b0v31_classifier(model).trunk.parameters()),
        "grad_rms_ste": () if model.encoder is None else tuple(model.encoder.parameters()),
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
    # This guard measures learned SlotSet geometry. FullOracleGenerator has no
    # slots, plan, or trainable generator state, so running a V_val sampled-union
    # scoring pass here cannot evaluate the guard. Return before touching the
    # validation population or model state; ordinary epoch validation remains
    # unchanged and still supplies the diagnostic's selection metrics.
    if not data.val_pairs:
        raise RuntimeError(
            "step-0 slot guard has an empty population: this config gives "
            "_validate_epoch no validation pairs, so neither this guard nor "
            "the during-training slot-collapse guard would ever evaluate"
        )
    if isinstance(model.generator, FullOracleGenerator):
        return {}
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
    # The Sec 14.4.8 cosine trip line is a claim about `EgoStitchImagineGenerator`'s
    # `SlotSet` geometry at initialization; the remaining oracle-scaffold and
    # null generators carry no such geometry (`_validate_epoch` reports NaN
    # scale telemetry for them, design §3.4), so `report`'s finiteness check
    # below would spuriously fail either arm. There is nothing this guard can
    # check for them -- skip identically on every rank after the legacy step-0
    # validation event above. FullOracleGenerator returned before validation.
    if not isinstance(model.generator, EgoStitchImagineGenerator):
        return {}
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
        report["quality_threshold_missed"] = float(report["h_pairwise_cosine_mean"] > 0.95)
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
    kd_runtime = (
        _build_kd_runtime(cfg.distill, model, device=accelerator.device)
        if cfg.distill is not None
        else None
    )
    kd_generator = (
        cast(FullEgoFeaturesGenerator, model.generator) if kd_runtime is not None else None
    )
    composite = _CompositeStep(model, world, kd=kd_runtime)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameter_groups.groups[name], "lr": training.lr_peak, "name": name}
            for name in (
                "generator",
                "encoder",
                "classifier",
            )
            # Skip a group with no trainable parameters (the oracle-scaffold
            # generator, or the null generator's absent generator/encoder)
            # cleanly rather than registering an empty `torch.optim` group.
            if parameter_groups.groups[name]
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
        generator_supervision=isinstance(model.generator, EgoStitchImagineGenerator),
        relational_supervision=(model.encoder is not None and model.encoder.rel_head is not None),
    )

    validation_events_path = cfg.output_dir / VAL_REGION_VALIDATION_EVENTS_FILENAME
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if validation_events_path.exists():
            raise FileExistsError(
                f"V_val validation-event ledger already exists: {validation_events_path}"
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
                metadata["val_region_validation_evidence"] = {
                    "schema": "egostitch_e2e_val_region_validation_events_v1",
                    "count": len(validation_events),
                    "path": VAL_REGION_VALIDATION_EVENTS_FILENAME,
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
    # over the V_val sampled-pair union per run, charged before the
    # peak-memory counter is reset below and therefore outside the measured
    # training peak.
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

    rows_per_rank, microsteps_per_epoch = _epoch_step_plan(
        len(data.training_positives),
        negative_ratio=cfg.data.negative_ratio,
        edge_batch=cfg.data.edge_batch,
        world_size=world,
    )
    accumulation_steps = cfg.optim.gradient_accumulation_steps
    window_sizes = e2e_accumulation_window_sizes(microsteps_per_epoch, accumulation_steps)
    steps_per_epoch = len(window_sizes)
    production_epoch_step_counts = [steps_per_epoch] * cfg.optim.epochs
    epoch_step_counts = (
        production_epoch_step_counts[:1] if profile_only else production_epoch_step_counts
    )
    schedule_total_steps = steps_per_epoch * cfg.optim.epochs
    phase_a_end, phase_b_end = e2e_phase_boundaries(
        schedule_total_steps,
        phase_a_fraction=training.phase_a_fraction,
        phase_b_fraction=training.phase_b_fraction,
    )
    first_eligible_epoch = e2e_first_eligible_epoch(
        schedule_total_steps,
        steps_per_epoch,
        phase_a_fraction=training.phase_a_fraction,
        phase_b_fraction=training.phase_b_fraction,
    )

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
    # EgoStitch scores no validation counterpart for its recon/real/ssl terms,
    # so the monitored total validation loss is the edge BCE alone.
    best_val_total = float("inf")
    evals_without_improvement = 0
    stop_epoch: int | None = None
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
        "val_universe_scoring_seconds": 0.0,
    }
    global_step = 0
    prefetch_depth = cfg.runtime.prefetch_factor if cfg.runtime is not None else 1
    full_topology_validations = 0
    cascade_topology_validations = 0
    completed_epochs = len(epoch_step_counts)

    for epoch, epoch_steps in enumerate(epoch_step_counts, start=1):
        epoch_started = time.monotonic()
        epoch_data_wait = 0.0
        epoch_local_pairs = 0
        epoch_local_tokens = 0
        epoch_global_pairs = 0
        epoch_parts: dict[str, float] = {}
        epoch_kd_local = [0.0, 0.0, 0.0, 0.0]
        epoch_probes: list[dict[str, object]] = []
        epoch_validation_seconds = 0.0
        epoch_validation_timing = dict.fromkeys(total_validation_timing, 0.0)
        epoch_window_sizes = window_sizes[:epoch_steps]
        epoch_microsteps = sum(epoch_window_sizes)
        batch_source = iter(
            factory.epoch_batches(epoch, rows_per_rank=rows_per_rank, steps=epoch_microsteps)
        )
        batches = _prefetch_batches(batch_source, depth=prefetch_depth)
        micro_step_in_epoch = 0
        try:
            for window_size in epoch_window_sizes:
                fetch_started = time.monotonic()
                window = [next(batches) for _ in range(window_size)]
                epoch_data_wait += time.monotonic() - fetch_started
                phase = e2e_phase_state(
                    global_step,
                    schedule_total_steps,
                    phase_a_fraction=training.phase_a_fraction,
                    phase_b_fraction=training.phase_b_fraction,
                )
                active_groups = _e2e_active_groups(phase, model)
                base_lr = _e2e_base_lr(global_step, schedule_total_steps, training)
                for group in optimizer.param_groups:
                    group["lr"] = _e2e_optimizer_group_lr(
                        base_lr,
                        phase,
                        group.get("name"),
                        active_groups,
                    )
                local_denominator = e2e_window_effective_weight_denominator(
                    window, positive_weight=training.positive_weight
                ).to(accelerator.device)
                edge_loss_denominator = accelerator.reduce(local_denominator, reduction="sum")
                if (
                    not bool(torch.isfinite(edge_loss_denominator))
                    or float(edge_loss_denominator) <= 0.0
                ):
                    raise RuntimeError(
                        "global accumulated weighted-BCE denominator must be finite and positive"
                    )
                kd_seed_denominator: torch.Tensor | None = None
                kd_gram_denominator: torch.Tensor | None = None
                if kd_runtime is not None:
                    assert kd_generator is not None
                    local_kd_rows = torch.zeros(2, dtype=torch.float32)
                    for batch in window:
                        live = batch.edge["edge_mask"].to(dtype=torch.bool)
                        candidate_counts = kd_generator.candidate_node_counts(
                            batch.edge["node_row_i"], batch.edge["node_row_j"]
                        )
                        local_kd_rows[0] += live.sum()
                        local_kd_rows[1] += (live & (candidate_counts >= 2)).sum()
                    kd_denominators = accelerator.reduce(
                        local_kd_rows.to(accelerator.device), reduction="sum"
                    )
                    kd_seed_denominator = kd_denominators[0]
                    kd_gram_denominator = kd_denominators[1]
                    if (
                        not bool(torch.isfinite(kd_denominators).all())
                        or float(kd_seed_denominator) <= 0.0
                        or float(kd_gram_denominator) < 0.0
                    ):
                        raise RuntimeError(
                            "global accumulated KD denominators must be finite with seed rows"
                        )
                optimizer.zero_grad(set_to_none=True)
                out: dict[str, object] | None = None
                loss: torch.Tensor | None = None
                window_parts: dict[str, float] = {}
                for micro_index, batch in enumerate(window):
                    payload = _e2e_training_payload(
                        batch,
                        cfg,
                        phase,
                        epoch=epoch,
                        step=global_step,
                        micro_step=micro_step_in_epoch,
                        total_steps=schedule_total_steps,
                        device=accelerator.device,
                        edge_loss_denominator=edge_loss_denominator,
                        kd_seed_denominator=kd_seed_denominator,
                        kd_gram_denominator=kd_gram_denominator,
                    )
                    if fixed_replay is None:
                        fixed_replay = cast(dict[str, object], _detached_clone(payload))
                        fixed_replay.pop("edge_loss_denominator", None)
                        fixed_replay.pop("kd_seed_denominator", None)
                        fixed_replay.pop("kd_gram_denominator", None)
                    synchronization = (
                        accelerator.no_sync(wrapped)
                        if micro_index + 1 < len(window)
                        else nullcontext()
                    )
                    with synchronization:
                        out = cast(dict[str, object], wrapped(payload))
                        loss = cast(torch.Tensor, out["loss"])
                        local_bad = not bool(torch.isfinite(loss).all())
                        bad_ranks = accelerator.reduce(
                            torch.tensor(int(local_bad), device=accelerator.device),
                            reduction="sum",
                        )
                        if int(bad_ranks.item()) > 0:
                            raise RuntimeError(
                                f"non-finite E2E loss at optimizer step {global_step}"
                            )
                        accelerator.backward(loss)
                    for name, value in cast(dict[str, float], out["parts"]).items():
                        window_parts[name] = window_parts.get(name, 0.0) + value
                    if kd_runtime is not None:
                        kd_stats = cast(dict[str, float], out["kd_stats"])
                        epoch_kd_local[0] += kd_stats["seed_sum"]
                        epoch_kd_local[1] += kd_stats["gram_sum"]
                        epoch_kd_local[2] += kd_stats["live_rows"]
                        epoch_kd_local[3] += kd_stats["gram_live_rows"]
                    micro_step_in_epoch += 1
                    epoch_local_pairs += batch.edge_rows_true
                    epoch_local_tokens += batch.f0_rows_gathered
                    epoch_global_pairs += batch.edge_rows_global
                assert out is not None and loss is not None
                local_gradient_statistics, gathered_squared, gathered_nonfinite = (
                    _gather_e2e_group_gradient_statistics(parameter_groups.groups, accelerator)
                )
                e2e_assert_no_nonfinite_gradients(gathered_nonfinite)
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
                    statistics=local_gradient_statistics,
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
                    enforce_persistent=(not profile_only and enforce_quality),
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
                epoch_parts = window_parts
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
                        global_step - 1,
                        schedule_total_steps,
                        phase_a_fraction=training.phase_a_fraction,
                        phase_b_fraction=training.phase_b_fraction,
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
                                or (len(family_norms[group]) >= 2 and not flags["ratio_defined"])
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
                        topology_scope=_e2e_topology_validation_scope(
                            epoch,
                            cfg.optim.epochs,
                            cfg.topology_validation,
                        ),
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
                                        "quality_failures": end_ramp_precision["quality_failures"],
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

        finally:
            batches.close()

        if kd_runtime is not None:
            epoch_kd_global = accelerator.reduce(
                torch.tensor(epoch_kd_local, device=accelerator.device, dtype=torch.float64),
                reduction="sum",
            )
            if (
                not bool(torch.isfinite(epoch_kd_global).all())
                or float(epoch_kd_global[2]) <= 0.0
                or float(epoch_kd_global[3]) < 0.0
            ):
                raise RuntimeError("global epoch KD telemetry must be finite with live rows")
            epoch_parts["kd_seed"] = float(epoch_kd_global[0] / epoch_kd_global[2])
            epoch_parts["kd_gram"] = (
                float(epoch_kd_global[1] / epoch_kd_global[3])
                if float(epoch_kd_global[3]) > 0.0
                else 0.0
            )

        topology_scope = _e2e_topology_validation_scope(
            epoch,
            cfg.optim.epochs,
            cfg.topology_validation,
        )
        validation_started = time.monotonic()
        validation = _validate_epoch(
            model,
            data,
            accelerator,
            edge_batch=cfg.data.edge_batch,
            topk_fraction=cfg.diagnostics.topk_fraction,
            token_table=factory._token_table,
            token_node_index=factory._token_node_index,
            topology_scope=topology_scope,
        )
        epoch_validation_seconds += time.monotonic() - validation_started
        if validation is not None:
            for name in epoch_validation_timing:
                epoch_validation_timing[name] += validation.timing.get(name, 0.0)
        if topology_scope == "full":
            full_topology_validations += 1
        else:
            cascade_topology_validations += 1
        record_validation_event("epoch_end", epoch, global_step)
        validation_seconds = epoch_validation_seconds
        epoch_wall = time.monotonic() - epoch_started
        phase = e2e_phase_state(
            global_step - 1,
            schedule_total_steps,
            phase_a_fraction=training.phase_a_fraction,
            phase_b_fraction=training.phase_b_fraction,
        )
        collapse_failure = 0
        slot_collapse_failure = 0
        validation_nonfinite_failure = 0
        if accelerator.is_main_process:
            assert validation is not None
            metrics = validation.metrics
            fidelity = validation.fidelity
            last_metrics = metrics
            last_fidelity = fidelity
            if validation.task_loss < best_val_total:
                best_val_total = validation.task_loss
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            full_joint_epochs = max(0, epoch - first_eligible_epoch + 1)
            validation_quality_values = {
                "auprc": metrics.auprc,
                "brier": metrics.brier,
                "prevalence": fidelity["prevalence"],
                "active_logit_std": fidelity["active_logit_std"],
                "gs_bfs": fidelity["gs_bfs"],
                "rd_bfs": fidelity["rd_bfs"],
                "degree_mmd_ratio": fidelity["degree_mmd_ratio"],
                "clustering_mmd_ratio": fidelity["clustering_mmd_ratio"],
                "spectral_mmd_ratio": fidelity["spectral_mmd_ratio"],
                "f_logit_std": fidelity["f_logit_std"],
                "f_logit_auprc": fidelity["f_logit_auprc"],
                "h_pairwise_cosine_mean": fidelity["h_pairwise_cosine_mean"],
                "plan_rank1_marginal_residual": fidelity["plan_rank1_marginal_residual"],
            }
            validation_nonfinite_failure = int(
                not all(math.isfinite(value) for value in validation_quality_values.values())
            )
            if not profile_only and _e2e_should_capture_eligibility_reference(
                phase,
                warm_reference_auprc=warm_reference_auprc,
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
            # `h_pairwise_cosine_mean`/`plan_rank1_marginal_residual` are the
            # NaN-defaulted-to-0.0/0.0 slot-geometry telemetry for the
            # oracle-scaffold and null generators (`_validate_epoch`); at
            # (0.0, 0.0) the guard's `residual < 0.05` half would spuriously
            # read "collapsed" every epoch once conditioning is active, for a
            # generator that has no slot geometry to collapse. Skip the guard
            # entirely for those arms rather than accept that false signal.
            if not validation_nonfinite_failure and isinstance(
                model.generator, EgoStitchImagineGenerator
            ):
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
                        "plan_rank1_marginal_residual": fidelity["plan_rank1_marginal_residual"],
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
            if topology_scope == "full":
                record = E2ECheckpointRecord(
                    epoch=epoch,
                    phase=phase.phase,
                    full_joint_epochs_completed=full_joint_epochs,
                    guards_passed=quality_guards_passed,
                    auprc=metrics.auprc,
                    prevalence=fidelity["prevalence"],
                    active_logit_std=fidelity["active_logit_std"],
                    gs=fidelity["gs_bfs"],
                    rd=fidelity["rd_bfs"],
                    degree_mmd=fidelity["degree_mmd_ratio"],
                    clustering_mmd=fidelity["clustering_mmd_ratio"],
                    spectral_mmd=fidelity["spectral_mmd_ratio"],
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
                    "val_task_loss": validation.task_loss,
                    "val_total_loss": validation.task_loss,
                    "topology_validation_scope": topology_scope,
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
                "topology_validation_scope": topology_scope,
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
        # Patience lives on the main rank (only it holds the gathered metrics),
        # so the decision is reduced before any rank leaves the loop -- a
        # rank-local break would deadlock the survivors in the next collective.
        # The stop is deferred to the next full-topology epoch (`topology_scope`
        # is a pure function of the epoch, identical on every rank): only those
        # epochs produce `E2ECheckpointRecord`s, and stopping on a cascade epoch
        # would drop selection into the `telemetry_miss_last_epoch` fallback
        # instead of the six-criterion mean rank.
        stop_now = accelerator.reduce(
            torch.tensor(
                int(
                    topology_scope == "full"
                    and accelerator.is_main_process
                    and evals_without_improvement >= cfg.eval.patience
                ),
                device=accelerator.device,
            ),
            reduction="sum",
        )
        if int(stop_now.item()) > 0:
            stop_epoch = epoch
            completed_epochs = epoch
            logger.info(
                "egostitch early stopping at epoch %d (%d evals without "
                "val-total-loss improvement)",
                epoch,
                cfg.eval.patience,
            )
            break

    executed_steps = sum(epoch_step_counts[:completed_epochs])
    executed_microbatches = microsteps_per_epoch * completed_epochs
    if global_step != executed_steps:
        raise RuntimeError(f"E2E execution coverage broken: {global_step} != {executed_steps}")
    last_state = _cpu_state_dict(accelerator, wrapped) if accelerator.is_main_process else {}
    selected_epoch_local = 0
    best_state: dict[str, torch.Tensor] = {}
    best_metrics: EdgeMetrics | None = None
    if accelerator.is_main_process:
        assert last_metrics is not None and last_fidelity is not None
        if profile_only:
            selected_epoch_local = completed_epochs
            best_state = last_state
            best_metrics = last_metrics
        else:
            selected = select_e2e_checkpoint(records, arm)
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
        diagnostic_epoch = completed_epochs
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
            "all_nodes_within_training_universe": bool(access_valid[index]),
        }
        for index in range(world)
    ]
    slowest_wall = max(float(row[2]) for row in rank_stats)
    global_pairs = int(sum(int(cast(int, entry["global_pairs"])) for entry in per_epoch_profiles))
    global_tokens = int(sum(float(row[1]) for row in rank_stats))
    runtime_profile: dict[str, object] = {
        "epochs_completed": completed_epochs,
        "validations_completed": completed_epochs,
        "full_topology_validations_completed": full_topology_validations,
        "cascade_topology_validations_completed": cascade_topology_validations,
        "val_region_validation_event_count": len(validation_events),
        "val_region_validation_events": validation_events,
        "peak_memory_gib_per_rank": [float(row[4]) for row in rank_stats],
        "steady_state_data_wait_fraction": max(
            float(row[3] / row[2]) if row[2] > 0 else 0.0 for row in rank_stats
        ),
        "training_coverage_exact": True,
        "validation_coverage_exact": True,
        "feature_cache_hit_rate": 1.0,
        "stop_epoch": stop_epoch,
        "per_rank": [
            {
                "rank": index,
                "pairs": int(row[0]),
                "batches": executed_microbatches,
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
        "gradient_accumulation_steps": accumulation_steps,
        "total_microbatches": executed_microbatches,
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
        last_epoch=completed_epochs,
        last_val_metrics=last_metrics,
        history=history,
        stop_epoch=stop_epoch,
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


VAL_REGION_VALIDATION_EVENTS_FILENAME = "val_region_validation_events.jsonl"

_MODEL_CONFIG_HASH_SCHEMA = "egostitch_e2e_model_config_v3"


def model_config_hash(cfg: EgoConfig) -> str:
    """Hash the model-defining configuration for run provenance.

    `config_to_dict` is path-sensitive: it carries the whole config, including
    ``output_dir`` and ``data.root`` (documented CLAUDE.md trap). This digest
    covers what defines the model and what it is trained on, and deliberately
    excludes:

    - ``output_dir`` / ``data.root`` / every other path;
    - ``optim.epochs`` -- recorded by the plan/config identity instead;
    - ``model.config['generator']['feature_stats_sha256']`` -- recorded
      alongside this digest, at its nested location (design 2026-08-02 §8);
    - ``seed`` -- an execution parameter. The formal stage may sweep
      ``--seeds`` without changing the model definition.

    Args:
        cfg: The validated worker config.

    Returns:
        The 64-character hex digest.
    """
    model_config = {
        key: (
            {k: v for k, v in value.items() if k != "feature_stats_sha256"}
            if key == "generator" and isinstance(value, Mapping)
            else value
        )
        for key, value in cfg.model.config.items()
    }
    payload: dict[str, object] = {
        "schema": _MODEL_CONFIG_HASH_SCHEMA,
        "model": {"family": cfg.model.family, "config": model_config},
        "data": {
            "strategy": cfg.data.strategy,
            "training_interactions": "all_train_positives",
            "negative_ratio": cfg.data.negative_ratio,
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
        "strategy": cfg.data.strategy,
        "rho_train": data.rho_train,
        "training_interactions": "all_train_positives",
        "permanent_null": (
            e2e_config.classifier.permanent_null if e2e_config is not None else "none"
        ),
        "model_family": cfg.model.family,
        "p_topo": e2e_config.classifier.p_topo if e2e_config is not None else 0.0,
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "config_sha256": _sha256_file(config_path) if config_path is not None else None,
        "arm": arm,
        "feature_stats_sha256": feature_stats_sha256 or "",
        "oracle_truth_source": (
            e2e_config.generator.oracle_truth_source if e2e_config is not None else None
        ),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_outputs(
    result: EgoTrainResult, cfg: EgoConfig, data: EgoStitchData, *, debug: bool = False
) -> None:
    """Write the pinned Task-4 artifacts and finalize the run's metadata record.

    ``best.pt``/``last.pt`` carry exactly the seven pinned payload keys;
    ``run_metadata.json`` additionally records the s0 checkpoint identity and the
    measured ``rho_train``.
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
    validation_events = result.runtime_profile.get("val_region_validation_events")
    validation_event_count = result.runtime_profile.get("val_region_validation_event_count")
    validation_evidence: dict[str, object] | None = None
    if validation_events is not None or validation_event_count is not None:
        if (
            not isinstance(validation_events, list)
            or isinstance(validation_event_count, bool)
            or not isinstance(validation_event_count, int)
            or validation_event_count != len(validation_events)
        ):
            raise RuntimeError("invalid V_val validation-event count in runtime profile")
        validation_events_path = output_dir / VAL_REGION_VALIDATION_EVENTS_FILENAME
        if not validation_events_path.is_file():
            raise RuntimeError("V_val validation-event ledger is missing")
        persisted_events = [
            json.loads(line)
            for line in validation_events_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if persisted_events != validation_events:
            raise RuntimeError("V_val validation-event ledger disagrees with runtime profile")
        validation_evidence = {
            "schema": "egostitch_e2e_val_region_validation_events_v1",
            "count": validation_event_count,
            "path": VAL_REGION_VALIDATION_EVENTS_FILENAME,
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
    checkpoint_role = (
        "debug_only"
        if debug
        else "diagnostic_only"
        if expected_run_kind == "diagnostic"
        else "formal_plan_selected"
    )
    classifier_config = cast(Mapping[str, object], cfg.model.config.get("classifier", {}))
    validation_liveness_observed = result.best_epoch > 0 and (
        classifier_config.get("permanent_null") != "none"
        or any(
            cast(dict[str, float], entry["fidelity"]).get("topology_delta_ratio", 0.0) >= 1e-3
            for entry in result.history
            if int(cast(float, entry["epoch"])) == result.best_epoch
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
            "val_region_validation_evidence": validation_evidence,
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
        data: The assembled training data, carrying the training-universe statistics.

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
    generator_config = cast(Mapping[str, object], cfg.model.config.get("generator", {}))
    mode = str(generator_config.get("feature_standardization", "zscore_vfit_v1"))
    if mode != "zscore_vfit_v1":
        return ""
    stats = data.feature_stats
    if stats is None:
        raise RuntimeError(
            "feature standardization statistics are unavailable; "
            "rebuild the feature pack before training"
        )
    pinned = str(generator_config.get("feature_stats_sha256", ""))
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


def _oracle_truth_graph(data: EgoStitchData, *, truth_source: str, run_kind: str) -> nx.Graph:
    """Return the oracle source graph named by ``generator.oracle_truth_source``.

    ``training_structure`` is the protocol-clean training structural graph
    (spec Sec 9.3/9.4); it already carries every V_val node via its legal
    cross-boundary edges, so a V_val node's cross-boundary scaffold is visible
    under this source alone. ``training_structure_plus_g_val`` additionally
    overlays the V_val-internal positives the training graph excludes by
    construction, so a V_val node's within-region neighbors also appear. That
    overlay is a deliberate held-out-truth leak: it is accepted only under
    ``--run-kind diagnostic`` and is stamped into `EgoStitchData.access_audit`.
    Stitch-time leave-one-out still masks the queried partner, so the queried
    edge itself is never handed to the classifier.

    Args:
        data: The assembled training data (`EgoTargetBuilder` + V_val split).
        truth_source: The configured `GeneratorConfig.oracle_truth_source`.
        run_kind: Effective execution context, used to keep held-out truth out of
            any run that publishes formal artifacts.

    Returns:
        `EgoTargetBuilder.graph` itself for ``training_structure``, or a copy
        extended with the loopless V_val positive overlay.

    Raises:
        RuntimeError: If held-out truth is requested outside a diagnostic run,
            or if V_val carries no positive edges.
    """
    if truth_source == "training_structure":
        return data.target_builder.graph
    if run_kind != "diagnostic":
        raise RuntimeError(
            "oracle_truth_source='training_structure_plus_g_val' requires --run-kind diagnostic"
        )
    val_edges = data.validation_positive_edges
    if not val_edges:
        raise RuntimeError("true-oracle diagnostic requires non-empty V_val positive edges")
    graph = nx.Graph(data.target_builder.graph)
    graph.add_nodes_from(data.validation_nodes)
    # Loopless: the structural overlay carries no self-loops, matching
    # `ValRegionSplit.build_training_graph`'s own convention.
    graph.add_edges_from((u, v) for u, v in val_edges if u != v)
    audit = data.access_audit if data.access_audit is not None else {}
    audit["oracle_truth"] = {
        "source": truth_source,
        "diagnostic_only": True,
        "val_region_node_count": len(data.validation_nodes),
        "val_region_positive_edge_count": len(val_edges),
        "val_region_positive_edges_sha256": hashlib.sha256(
            json.dumps(sorted(val_edges), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "queried_partner_masked_at_stitch": True,
    }
    data.access_audit = audit
    return graph


def _install_oracle_context(model: EgoStitchModel, data: EgoStitchData, *, run_kind: str) -> None:
    """Install truth context for either registered oracle generator arm.

    Runs once at startup, after the model is built and `_bind_feature_standardization`
    has called `EgoStitchModel.set_feature_stats` (Wave-1 oracle-scaffold
    design). One table row is built per node in the F0 universe (``data.f0``
    / ``data.node_index``, the training universe -- which already contains
    every V_val node), in F0-row order, so the F0-row -> table-row lookup
    `OracleStructGenerator.set_oracle_context` needs is simply the identity
    permutation (``arange``) -- table row ``r`` describes exactly the node at
    F0 row ``r``.

    The source graph is whichever one `_oracle_truth_graph` returns for the
    configured ``generator.oracle_truth_source``.

    Args:
        model: The freshly constructed E2E model, already feature-stats-bound.
        data: The assembled training data (F0 universe + `EgoTargetBuilder`).
        run_kind: Effective execution context, forwarded to `_oracle_truth_graph`
            to keep held-out structural truth inside diagnostic runs.

    Raises:
        RuntimeError: If `model.generator` is not a registered oracle
            generator, if a full-ego oracle is attempted outside a diagnostic
            run, or if `data.node_index` does not densely cover every F0 row.
    """
    if not isinstance(model.generator, (OracleStructGenerator, FullOracleGenerator)):
        raise RuntimeError(
            "oracle context install requires a registered oracle generator, got "
            f"{type(model.generator).__name__}"
        )
    if isinstance(model.generator, FullOracleGenerator) and run_kind != "diagnostic":
        raise RuntimeError("full_ego_oracle requires --run-kind diagnostic")
    n_rows = int(data.f0.shape[0])
    node_by_row: list[str | None] = [None] * n_rows
    for node, row in data.node_index.items():
        node_by_row[row] = node
    missing_rows = [row for row, node in enumerate(node_by_row) if node is None]
    if missing_rows:
        raise RuntimeError(
            f"F0 universe has {len(missing_rows)} row(s) with no node_index entry; "
            f"first few: {missing_rows[:5]}"
        )
    ordered_node_ids = cast(list[str], node_by_row)
    truth_source = model.cfg.generator.oracle_truth_source
    truth_graph = _oracle_truth_graph(data, truth_source=truth_source, run_kind=run_kind)
    if isinstance(model.generator, OracleStructGenerator):
        table = build_oracle_table(
            truth_graph,
            ordered_node_ids,
            slots=model.generator_cfg.slots,
            seed=model.cfg.generator.oracle_seed,
        )
        lookup = torch.arange(n_rows, dtype=torch.long)
        model.generator.set_oracle_context(table, lookup)
    else:
        full_truth_graph = truth_graph.copy()
        # F0 rows with zero structural degree are legitimate explicit
        # isolates in the full-oracle truth context.
        full_truth_graph.add_nodes_from(ordered_node_ids)
        model.generator.set_oracle_context(full_truth_graph, ordered_node_ids)
        if isinstance(model.generator, FullEgoFeaturesGenerator):
            # Same rows, same order: `data.f0` is the training-universe F0
            # matrix `ordered_node_ids` was derived from above. Truth-graph
            # nodes outside it (featureless survivors) gather zeros with an
            # explicit has_f0=0 indicator inside the generator.
            model.generator.set_node_features(data.f0, ordered_node_ids)
    logger.info(
        "installed oracle truth context generator=%s rows=%d truth_source=%s",
        model.cfg.generator.name,
        n_rows,
        truth_source,
    )


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
    if model.cfg.generator.name in ("oracle_struct", "full_ego_oracle", "full_ego_features"):
        _install_oracle_context(model, data, run_kind=effective_run_kind)
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

    # The degree prior centers `EgoStitchImagineGenerator`'s lognormal degree
    # head bias; the oracle-scaffold and null generators have no such head
    # (design §3.4), so there is nothing to center for either arm.
    if isinstance(model.generator, EgoStitchImagineGenerator):
        degree_prior = e2e_degree_prior_init(model, data)
        logger.info("degree head centered on G_train prior mean(log d)=%.6f", degree_prior)
    else:
        logger.info(
            "skipping degree-prior centering: %s generator has no degree head",
            type(model.generator).__name__,
        )

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
        "egostitch ddp train complete: best epoch %d val AUPRC %.4f (stop_epoch=%s)",
        result.best_epoch,
        result.best_val_metrics.auprc,
        result.stop_epoch,
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
