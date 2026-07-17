r"""EgoStitch Stage-1 training worker (spec Sec 13; auto-sized H20 DDP).

Drop-in worker for the `src.e2_pipeline` orchestrator: implements the same
``--ddp-mode {probe,epoch-probe,train}`` CLI contract, runtime-profile schema,
and Task-4 checkpoint payload as ``src.train_b0``, over the EgoStitch two-stream
composite step (node stream -> L_recon/L_ssl/L_real; edge stream -> L_edge; the
joint-pair stream joins in Stage 3). ``--token-budget-per-rank`` is
reinterpreted for this family as the per-rank node-stream batch size ``B_n``
(spec Sec 13.13); the runtime budget is config-driven, not the E2 60-minute pin.

Launch (formal):

    accelerate launch --num_processes <visible-H20-count> --mixed_precision bf16 \
        -m src.train_egostitch --config configs/egostitch_stage1_breadth_first.yaml \
        --ddp-mode train --pack-dir <pack> --output-dir <out> \
        --token-budget-per-rank 256 --profile-output <profile.json>

s0 cache (spec Sec 13.10): every training/val pair's frozen-B0 logit must be
precomputed. ``--write-s0-manifest <tsv>`` enumerates the exact deterministic
pair universe this config will consume (all epochs x ranks + val), to be scored
once via ``python -m src.score_universe score --pairs file:<tsv> ...`` with the
audited checkpoint; ``data.s0_cache`` then points at the resulting ``.npz``.

Pre-registration (protocol Sec 5.2.4): the worker records the sha256 of
``preregistration:`` in ``run_metadata.json`` before the first optimizer step;
the G5 gate evaluator refuses held-out metrics on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from numpy.typing import NDArray
from scipy.stats import kendalltau

from src.data.artifacts import Benchmark, canonical_pair, load_benchmark
from src.data.ego_targets import EgoTargetBuilder, EgoTargets
from src.data.features import FeatureStore, build_f0_matrix
from src.data.grounding import build_grounding_pool
from src.data.packed_features import PackedFeatureTable
from src.data.pairs import NegativeSampler
from src.data.partition import build_g_struct, derive_partition
from src.e2_pipeline import ProbeResult, detect_visible_gpu_count
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1
from src.model.egostitch.conditioning import (
    GatedCrossAttention,
    masks_for_null,
    sample_branch_masks,
)
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E
from src.model.egostitch.imagine import NULL_MODE_ALL, NULL_MODE_CONTENT, NULL_MODE_FULL
from src.model.egostitch.losses import stage1_family_tensors, stage1_total
from src.score_universe import ScoresArtifact, load_scores
from src.train_b0 import (
    DDP_MODES,
    EvalConfig,
    ModelConfig,
    RuntimeConfig,
    _as_float,
    _as_int,
    _as_int_list,
    _as_mapping,
    _as_str,
    _as_str_list,
    _check_no_unknown_keys,
    _emit_probe_candidate_failure,
    _is_oom_error,
    _require,
    _run_timed_epoch_probe,
    _state_digest,
    _write_json_rank_zero,
)

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = "benchmark_2025_neurips"
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_PACK_F0_FILENAME = "f0_matrix.pt"
_PACK_GROUNDING_FILENAME = "grounding.npz"
_PACK_MANIFEST_FILENAME = "manifest.json"


def _egostitch_ddp_kwargs() -> DistributedDataParallelKwargs:
    """Return DDP settings for EgoStitch's conditionally unused heads."""
    return DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


def build_egostitch_ddp_accelerator(mixed_precision: str) -> Accelerator:
    """Build the EgoStitch distributed accelerator."""
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[_egostitch_ddp_kwargs()])


# Frozen audited B0 checkpoint providing s0 (spec Sec 13.10); overridable in
# the config only for synthetic-fixture tests.
DEFAULT_S0_CHECKPOINT_ID = "e092537d8cf1e208"


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
        f0_cache: F0 matrix cache path used while building the s0 manifest;
            DDP modes use the pack directory instead.
        grounding_cache: Grounding-pool cache path (same convention).
        s0_cache: Frozen-B0 scores ``.npz`` covering every training/val pair.
        s0_checkpoint_id: Expected ``checkpoint_id`` of `s0_cache`.
        expected_missing_features: Exact graph nodes expected to lack features.
        pack_dir: Raw-token pack directory (spec Sec 13.18 family only; the
            same packed-feature store the B0 V3.1 loader consumes). ``None``
            for the frozen-s0 ``egostitch`` family, which has no token stream.
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
    s0_cache: Path
    s0_checkpoint_id: str
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
        warmstart_fraction: Leading fraction of total steps where only
            ``L_recon`` (+ degree NLL) carries gradient (spec Sec 13.8).
    """

    lr: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    grad_clip: float
    warmstart_fraction: float


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
class EgoConfig:
    """The full validated EgoStitch worker configuration.

    Attributes:
        model: ``family: egostitch`` plus `EgoStitchConfig` overrides.
        data: Data assembly settings.
        optim: Optimizer settings.
        eval: VAL-CRITERION bookkeeping settings.
        seed: Global seed.
        output_dir: Artifact directory.
        mixed_precision: ``"no"`` or ``"bf16"``.
        preregistration: Path to the committed pre-registration JSON whose
            sha256 is recorded in ``run_metadata.json``.
        runtime: Orchestrator runtime contract (``token_budget_candidates``
            reinterpreted as ``B_n`` candidates; no frozen E2 probe list).
    """

    model: ModelConfig
    data: EgoDataConfig
    optim: EgoOptimConfig
    diagnostics: EgoDiagnosticsConfig
    eval: EvalConfig
    seed: int
    output_dir: Path
    mixed_precision: str
    preregistration: Path
    runtime: RuntimeConfig | None = None


@dataclass(frozen=True)
class EgoCliArgs:
    """Parsed CLI arguments (train_b0 contract + ``--write-s0-manifest``)."""

    config: Path
    seed: int | None
    output_dir: Path | None
    ddp_mode: str | None = None
    pack_dir: Path | None = None
    token_budget_per_rank: int | None = None
    profile_output: Path | None = None
    write_s0_manifest: Path | None = None


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
            "preregistration",
            "runtime",
        ),
        "",
    )

    model_raw = _as_mapping(_require(raw, "model", ""), "model")
    _check_no_unknown_keys(model_raw, ("family", "config"), "model")
    family = _as_str(_require(model_raw, "family", "model."), "model.family")
    if family not in ("egostitch", _EGOSTITCH_E2E_FAMILY):
        raise ValueError(
            f"model.family must be 'egostitch' or {_EGOSTITCH_E2E_FAMILY!r}, got {family!r}"
        )
    model_kwargs = _as_mapping(model_raw.get("config") or {}, "model.config")
    if family == _EGOSTITCH_E2E_FAMILY:
        E2EConfig.from_mapping(model_kwargs)  # validate eagerly, fail loudly
    else:
        EgoStitchConfig.from_mapping(model_kwargs)  # validate eagerly, fail loudly
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
        "s0_cache",
        "s0_checkpoint_id",
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
        s0_cache=Path(_as_str(_require(data_raw, "s0_cache", "data."), "data.s0_cache")),
        s0_checkpoint_id=_as_str(
            data_raw.get("s0_checkpoint_id", DEFAULT_S0_CHECKPOINT_ID), "data.s0_checkpoint_id"
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
    _check_no_unknown_keys(
        optim_raw,
        ("lr", "weight_decay", "epochs", "warmup_steps", "grad_clip", "warmstart_fraction"),
        "optim",
    )
    warmstart_fraction = _as_float(
        optim_raw.get("warmstart_fraction", 0.2), "optim.warmstart_fraction"
    )
    if not 0.0 <= warmstart_fraction < 1.0:
        raise ValueError(f"optim.warmstart_fraction must be in [0, 1), got {warmstart_fraction}")
    optim = EgoOptimConfig(
        lr=_as_float(_require(optim_raw, "lr", "optim."), "optim.lr"),
        weight_decay=_as_float(_require(optim_raw, "weight_decay", "optim."), "optim.weight_decay"),
        epochs=_as_int(_require(optim_raw, "epochs", "optim."), "optim.epochs"),
        warmup_steps=_as_int(_require(optim_raw, "warmup_steps", "optim."), "optim.warmup_steps"),
        grad_clip=_as_float(_require(optim_raw, "grad_clip", "optim."), "optim.grad_clip"),
        warmstart_fraction=warmstart_fraction,
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
    if diagnostics.selection_auprc_tolerance < 0:
        raise ValueError("diagnostics.selection_auprc_tolerance must be non-negative")
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

    runtime: RuntimeConfig | None = None
    if raw.get("runtime") is not None:
        runtime_raw = _as_mapping(raw["runtime"], "runtime")
        runtime_keys = (
            "world_size",
            "pack_dir",
            "pack_workers",
            "loader_workers_per_rank",
            "prefetch_factor",
            "token_budget_candidates",
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
        if world_size_raw != "auto":
            raise ValueError("runtime.world_size must be 'auto' for EgoStitch Stage-1")
        runtime = RuntimeConfig(
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
            token_budget_candidates=_as_int_list(
                _require(runtime_raw, "token_budget_candidates", "runtime."),
                "runtime.token_budget_candidates",
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
                _require(runtime_raw, "reserve_seconds", "runtime."), "runtime.reserve_seconds"
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
        if not runtime.token_budget_candidates or any(
            candidate <= 0 for candidate in runtime.token_budget_candidates
        ):
            raise ValueError(
                "runtime.token_budget_candidates must be a non-empty list of positive "
                "node-batch (B_n) candidates for this family"
            )
        stage_total = (
            runtime.pack_budget_seconds
            + runtime.setup_probe_budget_seconds
            + runtime.train_eval_budget_seconds
            + runtime.artifact_budget_seconds
            + runtime.reserve_seconds
        )
        if stage_total != runtime.total_budget_seconds:
            raise ValueError(
                f"runtime stage budgets must sum to {runtime.total_budget_seconds}; "
                f"got {stage_total}"
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
        preregistration=Path(_as_str(_require(raw, "preregistration", ""), "preregistration")),
        runtime=runtime,
    )


def config_to_dict(cfg: EgoConfig) -> dict[str, object]:
    """Return a JSON-serializable dict of the full config (checkpoint payload)."""
    payload = asdict(cfg)

    def _stringify(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _stringify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_stringify(v) for v in value]
        return value

    return cast(dict[str, object], _stringify(payload))


def parse_args(argv: Sequence[str] | None = None) -> EgoCliArgs:
    """Parse the worker CLI (the train_b0 contract + ``--write-s0-manifest``).

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
        "--ddp-mode",
        choices=DDP_MODES,
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
    parser.add_argument(
        "--write-s0-manifest",
        type=Path,
        default=None,
        help=(
            "write the deterministic pair-universe TSV this config will consume "
            "(for the one-off frozen-B0 s0 scoring pass) and exit"
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
        ddp_mode=namespace.ddp_mode,
        pack_dir=namespace.pack_dir,
        token_budget_per_rank=namespace.token_budget_per_rank,
        profile_output=namespace.profile_output,
        write_s0_manifest=namespace.write_s0_manifest,
    )


def apply_overrides(cfg: EgoConfig, args: EgoCliArgs) -> EgoConfig:
    """Apply the ``--seed`` / ``--output-dir`` CLI overrides."""
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    return cfg


# --------------------------------------------------------------------------- pack stage


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    """
    del temp_prefix
    # Family `egostitch_e2e` (spec Sec 13.18): `cfg.model.config` validates
    # against `E2EConfig`, not `EgoStitchConfig` -- it sizes only the pair
    # trunk/conditioning pathways. The internal Stage-1 generator this pack
    # feeds (F0 matrix + grounding pool) always uses the pinned spec-default
    # `EgoStitchConfig()` (`e2e_model.py`'s `generator_cfg`), never a value
    # parsed from `cfg.model.config` -- so `n_ground` must come from there too.
    is_e2e = cfg.model.family == _EGOSTITCH_E2E_FAMILY
    if is_e2e:
        E2EConfig.from_mapping(cfg.model.config)  # validate eagerly, fail loudly
        n_ground = EgoStitchConfig().n_ground
    else:
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        n_ground = model_cfg.n_ground
    manifest_path = pack_dir / _PACK_MANIFEST_FILENAME
    if cold_cache:
        pack_dir.mkdir(parents=True, exist_ok=True)
        store = FeatureStore(cfg.data.root / _FEATURES_SUBDIR)
        benchmark = _load_benchmark_for(cfg)
        operative = sorted(set(benchmark.graph.nodes()) - set(cfg.data.expected_missing_features))
        matrix, index = build_f0_matrix(store, operative, cache_path=pack_dir / _PACK_F0_FILENAME)
        train_nodes = sorted(set(benchmark.split.train_nodes) & set(operative))
        train_rows = np.asarray(
            matrix.numpy()[[index[node] for node in train_nodes]], dtype=np.float32
        )
        build_grounding_pool(
            train_rows,
            train_nodes,
            n_ground=n_ground,
            cache_path=pack_dir / _PACK_GROUNDING_FILENAME,
        )
        manifest = {
            "family": cfg.model.family,
            "strategy": cfg.data.strategy,
            "n_operative_nodes": len(operative),
            "n_train_nodes": len(train_nodes),
            "n_ground": n_ground,
            "files": {
                name: _sha256_file(pack_dir / name)
                for name in (_PACK_F0_FILENAME, _PACK_GROUNDING_FILENAME)
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = cast(dict[str, str], manifest["files"])
        for name, expected in files.items():
            actual = _sha256_file(pack_dir / name)
            if actual != expected:
                raise ValueError(f"pack file {name} drifted: {actual} != {expected}")
        if manifest.get("strategy") != cfg.data.strategy:
            raise ValueError("pack manifest strategy does not match the config")
        if manifest.get("n_ground") != n_ground:
            raise ValueError("pack manifest n_ground does not match the model config")
    return {
        "pack_manifest": manifest,
        "pack_identity_sha256": _sha256_file(manifest_path),
    }


# --------------------------------------------------------------------------- s0 cache


class S0Cache:
    """Frozen-B0 logit cache (spec Sec 13.10): hard-fails on any miss."""

    def __init__(self, artifact: ScoresArtifact, *, expected_checkpoint_id: str) -> None:
        """Index one scores artifact by canonical pair.

        Args:
            artifact: The loaded scores artifact.
            expected_checkpoint_id: Required ``meta['checkpoint_id']``.

        Raises:
            ValueError: On a checkpoint-identity mismatch.
        """
        actual = artifact.meta.get("checkpoint_id")
        if actual != expected_checkpoint_id:
            raise ValueError(
                f"s0 cache checkpoint_id mismatch: expected {expected_checkpoint_id!r}, "
                f"got {actual!r} (spec Sec 13.10: s0 is the audited frozen B0)"
            )
        self._logits: dict[tuple[str, str], float] = {}
        logits = artifact.logit.astype(np.float64)
        for row, (u, v) in enumerate(artifact.pairs()):
            self._logits[canonical_pair(u, v)] = float(logits[row])

    @classmethod
    def from_path(cls, path: Path, *, expected_checkpoint_id: str) -> S0Cache:
        """Load a cache from one ``.npz`` scores artifact."""
        return cls(load_scores(path), expected_checkpoint_id=expected_checkpoint_id)

    def __len__(self) -> int:
        """Return the number of cached pairs."""
        return len(self._logits)

    def lookup(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        """Return row-aligned s0 logits for `pairs`.

        Args:
            pairs: Node pairs (canonicalized internally).

        Returns:
            Shape ``(n,)`` float32 logits.

        Raises:
            KeyError: On the first missing pair (never silently imputed).
        """
        out = np.empty(len(pairs), dtype=np.float32)
        for i, (u, v) in enumerate(pairs):
            key = canonical_pair(u, v)
            if key not in self._logits:
                raise KeyError(
                    f"s0 cache is missing pair {key}; regenerate the manifest with "
                    "--write-s0-manifest and re-score it with the frozen checkpoint"
                )
            out[i] = self._logits[key]
        return out


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

    The single source of truth shared by the training loader and the s0
    manifest builder (spec Sec 13.10): positives are epoch-shuffled and
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


def build_s0_manifest(
    cfg: EgoConfig,
    e_sup_positives: Sequence[tuple[str, str]],
    val_pairs: Sequence[tuple[str, str]],
    sampler: NegativeSampler,
    output_path: Path,
    *,
    world_size: int,
) -> int:
    r"""Write the deduplicated pair-universe TSV the s0 cache must cover.

    Args:
        cfg: The validated worker config (epochs/ratio/seed).
        e_sup_positives: Canonical supervision positives.
        val_pairs: Validation pairs (model selection also needs s0).
        sampler: The pinned negative sampler.
        output_path: TSV destination (``u\\tv`` rows, canonical order, sorted).
        world_size: Rank count of the formal run.

    Returns:
        The number of unique pairs written.
    """
    unique: set[tuple[str, str]] = set()
    for epoch in range(1, cfg.optim.epochs + 1):
        for rank in range(world_size):
            for u, v, _ in enumerate_edge_stream(
                e_sup_positives,
                sampler,
                negative_ratio=cfg.data.negative_ratio,
                seed=cfg.seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            ):
                unique.add(canonical_pair(u, v))
    for u, v in val_pairs:
        unique.add(canonical_pair(u, v))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for u, v in sorted(unique):
            handle.write(f"{u}\t{v}\n")
    logger.info("wrote %d unique s0 pairs to %s", len(unique), output_path)
    return len(unique)


# --------------------------------------------------------------------------- data assembly


def _load_benchmark_for(cfg: EgoConfig) -> Benchmark:
    """Load the benchmark package (verification on for the pinned strategy)."""
    return load_benchmark(
        cfg.data.root / _BENCHMARK_SUBDIR,
        cfg.data.strategy,
        verify=cfg.data.strategy == "breadth_first",
        exclude_nodes=frozenset(cfg.data.expected_missing_features),
    )


@dataclass
class EgoStitchData:
    """Everything the training loop consumes.

    Attributes:
        train_nodes: Sorted train-side node ids with F0 rows.
        e_sup_positives: Canonical supervision positives (self-pairs included).
        val_pairs: Validation pairs in artifact order.
        val_labels: Aligned validation labels.
        f0: Shape ``(N, d)`` float32 CPU matrix.
        node_index: Node id -> `f0` row.
        grounding_index: Shape ``(n_train, n_g)`` int64 rows into `f0` for each
            train node's pool, aligned with `train_nodes`.
        train_pos: Node id -> position in `train_nodes`.
        target_builder: The `EgoTargetBuilder` over ``G_struct``.
        sampler: The pinned negative sampler.
        s0: The frozen-B0 logit cache.
        rho_train: Message-partition edge density (spec Sec 9.3).
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
    s0: S0Cache
    rho_train: float


def assemble_egostitch_data(
    cfg: EgoConfig,
    *,
    pack_dir: Path | None = None,
    require_s0: bool = True,
) -> EgoStitchData:
    """Assemble the full training data bundle from the frozen artifacts.

    Args:
        cfg: The validated worker config.
        pack_dir: DDP pack directory (its F0/grounding caches win); ``None``
            uses ``cfg.data.f0_cache`` / ``cfg.data.grounding_cache``.
        require_s0: Load and validate the s0 cache (disabled only for the
            ``--write-s0-manifest`` path, which exists to create it). Ignored
            (always treated as ``False``) for family `egostitch_e2e`, which
            retired the s0 channel entirely (spec Sec 13.10): its node stream
            still needs the same `EgoTargetBuilder`/grounding-pool machinery
            as the frozen-s0 family (delegated, unchanged, to the internal
            trainable generator), sized off the generator's own pinned
            `EgoStitchConfig()` defaults rather than ``cfg.model.config``
            (which validates as `E2EConfig` for this family).

    Returns:
        The `EgoStitchData` bundle.
    """
    is_e2e = cfg.model.family == _EGOSTITCH_E2E_FAMILY
    if is_e2e:
        E2EConfig.from_mapping(cfg.model.config)  # validate eagerly, fail loudly
        generator_cfg = EgoStitchConfig()
    else:
        generator_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
    benchmark = _load_benchmark_for(cfg)
    store = FeatureStore(cfg.data.root / _FEATURES_SUBDIR)

    operative = sorted(set(benchmark.graph.nodes()) - set(cfg.data.expected_missing_features))
    f0_cache = (pack_dir / _PACK_F0_FILENAME) if pack_dir is not None else cfg.data.f0_cache
    f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, node_index = build_f0_matrix(store, operative, cache_path=f0_cache)

    train_nodes = sorted(set(benchmark.split.train_nodes) & set(operative))
    train_positives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs, benchmark.split.train_pairs.labels, strict=True
        )
        if label == 1
    ]
    partition = derive_partition(
        train_positives, seed=cfg.data.partition_seed, msg_fraction=cfg.data.msg_fraction
    )
    g_struct = build_g_struct(train_nodes, partition.e_msg)
    n_train = len(train_nodes)
    rho_train = g_struct.number_of_edges() / (math.comb(n_train, 2) + n_train)

    grounding_cache = (
        (pack_dir / _PACK_GROUNDING_FILENAME) if pack_dir is not None else cfg.data.grounding_cache
    )
    train_rows = np.asarray(
        matrix.numpy()[[node_index[node] for node in train_nodes]], dtype=np.float32
    )
    pool = build_grounding_pool(
        train_rows, train_nodes, n_ground=generator_cfg.n_ground, cache_path=grounding_cache
    )
    grounding_index = np.array(
        [[node_index[neighbor] for neighbor in pool[node]] for node in train_nodes],
        dtype=np.int64,
    )

    target_builder = EgoTargetBuilder(
        g_struct,
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        pool,
        slots=generator_cfg.slots,
    )
    degrees = {node: int(g_struct.degree(node)) if node in g_struct else 0 for node in train_nodes}
    sampler = NegativeSampler(train_nodes, degrees, benchmark.positive_edges)

    if require_s0 and not is_e2e:
        s0 = S0Cache.from_path(cfg.data.s0_cache, expected_checkpoint_id=cfg.data.s0_checkpoint_id)
    else:  # manifest-writing path, or family egostitch_e2e (s0 retired)
        s0 = S0Cache.__new__(S0Cache)
        s0._logits = {}

    return EgoStitchData(
        train_nodes=train_nodes,
        e_sup_positives=[canonical_pair(u, v) for u, v in partition.e_sup],
        val_pairs=list(benchmark.split.val_pairs.pairs),
        val_labels=benchmark.split.val_pairs.labels,
        f0=matrix,
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(train_nodes)},
        target_builder=target_builder,
        sampler=sampler,
        s0=s0,
        rho_train=rho_train,
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
        return torch.from_numpy(rows)

    def _ground_rows(self, nodes: Sequence[str]) -> torch.Tensor:
        return self._data.f0[self._ground_pool_rows(nodes)]

    def _node_tensors(
        self, nodes: Sequence[str], targets: EgoTargets, *, epoch: int, step: int
    ) -> dict[str, torch.Tensor]:
        batch = len(nodes)
        k_d = max(1, self._model_cfg.slots // 2)
        gen = _seeded_generator(self._cfg.seed, epoch, step, self._rank, 0x0D)
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
        f0_train_rows = torch.tensor(
            [self._data.node_index[n] for n in self._data.train_nodes], dtype=torch.long
        )
        ground_resampled = self._data.f0[f0_train_rows[resample_rows]]

        return {
            "x": x,
            "ground_x": ground,
            "target_features": targets.features,
            "target_mult": targets.mult,
            "target_adj": targets.adj,
            "target_mask": targets.mask,
            "target_in_pool": targets.in_pool,
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
        return _gather_token_streams(table, index, endpoints_a, endpoints_b)

    def _edge_tensors(
        self, rows: Sequence[tuple[str, str, int]], *, pad_to: int
    ) -> tuple[dict[str, torch.Tensor], int]:
        true_rows = len(rows)
        padded: list[tuple[str, str, int]] = list(rows)
        if true_rows == 0:
            filler = (self._data.train_nodes[0], self._data.train_nodes[0], 0)
            padded = [filler]
        while len(padded) < pad_to:
            padded.append(padded[0])
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
        if self._token_table is not None:
            endpoints_u = [u for u, _, _ in padded]
            endpoints_v = [v for _, v, _ in padded]
            edge.update(self._token_streams(endpoints_u, endpoints_v))
            # Same-index-space grounding ids for both endpoints (spec Sec
            # 13.18): required by `EgoStitchE2E`'s grounded-identity-match
            # flag, which compares `ground_id_a`/`ground_id_b` for equality.
            edge["ground_id_i"] = self._ground_pool_rows(endpoints_u)
            edge["ground_id_j"] = self._ground_pool_rows(endpoints_v)
        else:
            s0 = (
                self._data.s0.lookup([(u, v) for u, v, _ in padded])
                if true_rows > 0
                else np.zeros(len(padded), dtype=np.float32)
            )
            edge["s0"] = torch.from_numpy(s0)
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
            edge, true_rows = self._edge_tensors(chunk, pad_to=edge_batch)
            global_count = _step_global_count(rows_per_rank, step, edge_batch)
            f0_rows = (
                node["x"].shape[0]
                + node["ground_x"].shape[0] * node["ground_x"].shape[1]
                + node["target_features"].shape[0] * node["target_features"].shape[1]
                + 4 * edge["x_i"].shape[0]  # x_i, x_j and both grounding gathers
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

    The warm-start curriculum (spec Sec 13.8) is applied as a 0/1 weight on the
    non-``L_recon`` families: every stream still runs (constant step shape for
    the projection gate; all parameters stay in the autograd graph, as required
    by ``find_unused_parameters=False``), but only ``L_recon`` (+ degree NLL)
    carries gradient during warm-start.

    Family `egostitch_e2e` (design rev 3, spec Sec 14): ``model`` is an
    `EgoStitchE2E` instead of a frozen-s0 `EgoStitchStage1`. The node stream
    (``L_recon``/``L_real``/``L_ssl``) is delegated to the internal, still
    trainable `EgoStitchE2E.generator`, unchanged from Stage 1. The edge stream
    (``L_edge``) instead runs the full `EgoStitchE2E.forward` under per-step
    seeded branch-dropout masks (`sample_branch_masks`, design Sec 4): the
    curriculum reuses the exact same ``joint_weight`` 0/1 gate on the ``edge``
    family, so the trunk/STE/gated cross-attention pathways -- which receive
    gradient *only* through ``L_edge`` -- train exactly when the pairwise
    trunk does (design Sec 4's stated intent), with zero gradient during
    warm-start.
    """

    kendall_active: torch.Tensor

    def __init__(self, model: EgoStitchStage1 | EgoStitchE2E, world_size: int) -> None:
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.kendall_log_vars = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros(())) for name in ("edge", "recon", "real", "ssl")}
        )
        self.register_buffer("kendall_active", torch.tensor(False), persistent=True)

    def activate_kendall(self) -> None:
        """Enable the pre-instantiated registered uncertainty weights."""
        self.kendall_active[...] = True

    def _generator(self) -> EgoStitchStage1:
        """Return the trainable Stage-1 generator for either family.

        For family `egostitch` this is ``self.model`` itself; for
        `egostitch_e2e` it is `EgoStitchE2E.generator`, the internal
        (non-frozen) Stage-1 module the node stream still trains via the
        unchanged `EgoStitchStage1.node_losses`/`ssl_losses`.
        """
        if isinstance(self.model, EgoStitchE2E):
            return self.model.generator
        return self.model

    def _edge_outputs(self, edge: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Frozen-s0 family (`egostitch`) self/non-self pair scoring split."""
        assert isinstance(self.model, EgoStitchStage1)  # family egostitch only; narrows the Union
        is_self = edge["is_self"]
        outputs = {
            name: edge["s0"].new_zeros(edge["s0"].shape)
            for name in ("logits", "residual", "s1", "s2", "s2_aa")
        }
        non_self = ~is_self
        if bool(non_self.any()):
            idx = torch.nonzero(non_self, as_tuple=False).squeeze(-1)
            batch = {
                "x_i": edge["x_i"][idx],
                "x_j": edge["x_j"][idx],
                "ground_i": edge["ground_i"][idx],
                "ground_j": edge["ground_j"][idx],
                "s0": edge["s0"][idx],
            }
            sub_out = self.model(batch)
            for name in outputs:
                outputs[name] = outputs[name].index_put((idx,), sub_out[name])
        if bool(is_self.any()):
            idx = torch.nonzero(is_self, as_tuple=False).squeeze(-1)
            batch = {
                "x_i": edge["x_i"][idx],
                "ground_i": edge["ground_i"][idx],
                "s0": edge["s0"][idx],
            }
            sub_out = self.model(batch)
            for name in outputs:
                outputs[name] = outputs[name].index_put((idx,), sub_out[name])
        return outputs

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        node = cast(dict[str, torch.Tensor], batch["node"])
        edge = cast(dict[str, torch.Tensor], batch["edge"])
        joint_weight = cast(torch.Tensor, batch["joint_weight"])  # 0 in warm-start
        global_count = cast(int, batch["edge_rows_global"])
        collect_diagnostics = bool(batch.get("collect_diagnostics", False))

        generator = self._generator()
        losses, _ = generator.node_losses(
            node["x"],
            node["ground_x"],
            target_features=node["target_features"],
            target_mult=node["target_mult"],
            target_adj=node["target_adj"],
            target_mask=node["target_mask"],
            target_in_pool=node["target_in_pool"],
            true_degree=node["true_degree"],
            real_ego_stats=node["real_ego_stats"],
            null_mode=node["null_mode"],
            denoise_features=node["denoise_features"],
            denoise_mask=node["denoise_mask"],
            denoise_noise=node["denoise_noise"],
        )
        ssl = generator.ssl_losses(
            node["x"], node["ground_x"], node["ground_resampled"], noise=node["ssl_noise"]
        )

        extra: dict[str, object] = {}
        if isinstance(self.model, EgoStitchE2E):
            # Per-step seeded branch-dropout masks (design Sec 4; the
            # `_seeded_generator(seed, epoch, step)` pattern already used for
            # the node stream, Sec 13.8's warm-start curriculum reuses the
            # same `joint_weight` gate below instead of a parallel mechanism).
            seed = cast(int, batch["seed"])
            epoch = cast(int, batch["epoch"])
            step = cast(int, batch["step"])
            if self.model.cfg.permanent_null == "none":
                branch_masks = sample_branch_masks(
                    edge["label"].shape[0],
                    self.model.cfg.p_topo,
                    self.model.cfg.p_cont,
                    generator=_seeded_generator(seed, epoch, step),
                    device=edge["label"].device,
                )
            else:
                branch_masks = masks_for_null(
                    self.model.cfg.permanent_null,
                    edge["label"].shape[0],
                    edge["label"].device,
                )
            logits = self.model(_e2e_edge_view(edge), masks=branch_masks)["logits"]
            if collect_diagnostics:
                extra.update(_e2e_gate_tanh(self.model))
        else:
            edge_outputs = self._edge_outputs(edge)
            logits = edge_outputs["logits"]
            if collect_diagnostics:
                extra["channel_stats"] = {
                    f"{name}_{stat}": float(getattr(edge_outputs[name].detach().float(), stat)())
                    for name in ("s1", "s2", "s2_aa", "residual")
                    for stat in ("mean", "std")
                }

        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, edge["label"], reduction="none"
        )
        # Exact global-mean gradient under DDP averaging: local masked sum
        # scaled by world / global_count (tail batches included; padded rows
        # carry zero weight).
        edge_loss = (per_row * edge["edge_mask"]).sum() * self.world_size / global_count

        total, parts = stage1_total(
            generator.config,
            edge=edge_loss * joint_weight,
            recon=losses.recon,
            deg=losses.deg,
            real_egostat=losses.real_egostat * joint_weight,
            real_gin=losses.real_gin * joint_weight,
            ssl_noise=ssl["noise"] * joint_weight,
            ssl_pool=ssl["pool"] * joint_weight,
        )
        families = stage1_family_tensors(
            generator.config,
            edge=edge_loss * joint_weight,
            recon=losses.recon,
            deg=losses.deg,
            real_egostat=losses.real_egostat * joint_weight,
            real_gin=losses.real_gin * joint_weight,
            ssl_noise=ssl["noise"] * joint_weight,
            ssl_pool=ssl["pool"] * joint_weight,
        )
        if bool(self.kendall_active):
            total = torch.stack(
                tuple(
                    torch.exp(-self.kendall_log_vars[name]) * family + self.kendall_log_vars[name]
                    for name, family in families.items()
                )
            ).sum()
        else:
            total = total + 0.0 * torch.stack(tuple(self.kendall_log_vars.values())).sum()
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


def _family_gradient_norms(
    model: EgoStitchStage1 | EgoStitchE2E, families: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Measure family-specific global L2 norms with isolated retained-graph backwards."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms: dict[str, float] = {}
    for index, (name, family) in enumerate(families.items()):
        model.zero_grad(set_to_none=True)
        family.backward(  # type: ignore[no-untyped-call]
            retain_graph=index < len(families) - 1
        )
        squared = family.new_zeros((), dtype=torch.float32)
        for parameter in parameters:
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().float().square().sum()
        norms[name] = float(torch.sqrt(squared))
    model.zero_grad(set_to_none=True)
    return norms


def _e2e_edge_view(edge: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate the worker's edge-tensor keys to `EgoStitchE2E`'s batch contract.

    Maps ``x_i``/``x_j`` to ``x_a``/``x_b`` and ``ground_i``/``ground_j``/
    ``ground_id_i``/``ground_id_j`` to ``ground_a``/``ground_b``/
    ``ground_id_a``/``ground_id_b`` (spec Sec 13.18: the real grounding-
    candidate features and their same-index-space global ids, required for
    `EgoStitchE2E`'s grounded-identity-match flag to engage instead of its
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
    }


def _e2e_gate_tanh(model: EgoStitchE2E) -> dict[str, list[float]]:
    """Per-injected-block ``tanh(gate)`` readout for both conditioning pathways.

    Registered names (spec Sec 13.17): ``gate_topo_tanh``, ``gate_cont_tanh``.
    A pure parameter readout -- no forward/backward pass required, safe to
    call every step.
    """

    def _values(modules: torch.nn.ModuleList) -> list[float]:
        out: list[float] = []
        for module in modules:
            assert isinstance(module, GatedCrossAttention)
            out.append(float(torch.tanh(module.gate).detach()))
        return out

    return {
        "gate_topo_tanh": _values(model.trunk.topo_xattn),
        "gate_cont_tanh": _values(model.trunk.cont_xattn),
    }


def _e2e_submodule_gradient_rms(model: EgoStitchE2E, loss: torch.Tensor) -> dict[str, float]:
    """Per-submodule gradient RMS from one isolated retained-graph backward.

    Registered names (spec Sec 13.17): ``grad_rms_trunk``, ``grad_rms_ste``,
    ``grad_rms_content``. Measures how much of ``loss``'s gradient reaches the
    trunk, the stitched-topology encoder, and the content projector -- the
    zero-init gated pathways the warm-start curriculum is designed to keep
    dead until ``L_edge`` activates. Mirrors `_family_gradient_norms`'s
    isolated-backward pattern: intended to be called on a dedicated probe
    forward's output (spec Sec 13.17's fixed replay batch), not on the tensor
    the caller is about to call its own `.backward()` on -- leaves every
    parameter's ``.grad`` at ``None`` afterward either way.
    """
    groups: dict[str, list[torch.nn.Parameter]] = {
        "grad_rms_trunk": list(model.trunk.parameters()),
        "grad_rms_ste": list(model.ste.parameters()),
        "grad_rms_content": list(model.content_proj.parameters()),
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
    model: EgoStitchE2E, batch: dict[str, torch.Tensor]
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
    with torch.no_grad():
        decomposed = model.decompose(batch)
    delta = (decomposed["full"] - decomposed["f_logit"]).detach()
    value = 0.0 if delta.numel() < 2 else float(torch.std(delta))
    return {"topology_delta_std": value}


def _e2e_null_arm_tiebreak(logits: np.ndarray) -> dict[str, float]:
    """Return null-arm validation diagnostics without excluded-pathway selection.

    Permanent-null arms choose checkpoints by their active validation AUPRC;
    their tie-break is pinned to zero so an excluded pathway cannot affect the
    selected checkpoint.
    """
    return {"active_logit_std": float(np.std(logits)), "selection_tiebreak": 0.0}


def _e2e_trainable_parameters(model: EgoStitchE2E) -> list[torch.nn.Parameter]:
    """Trainable parameters for family `egostitch_e2e`, excluding dead ones.

    `EgoStitchE2E.generator` is a full `EgoStitchStage1`, which builds a
    `DecisionHead` (the frozen-s0 family's ``(s0, s1, s2)`` fusion head, spec
    Sec 13.1) that the e2e forward path never calls -- `EgoStitchE2E.forward`
    fuses through its own trunk/head, not `generator.pair_outputs`/
    `self_outputs`. Left in the optimizer, `DecisionHead`'s parameters would
    sit with permanently zero gradient every step (a silent dead-parameter
    optimizer state, flagged in review). `generator.random_gin` has the
    opposite disposition: it *is* exercised (via `node_losses`'s ``L_real``
    energy-distance term) but is already frozen at construction
    (`requires_grad=False`, spec Sec 13.6), so it is excluded here too, for a
    different reason (never trainable, not merely unused).
    """
    decision_ids = {id(parameter) for parameter in model.generator.decision.parameters()}
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in decision_ids
    ]


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


def _fidelity_summary(
    s0: np.ndarray,
    logits: np.ndarray,
    channels: dict[str, np.ndarray],
    *,
    topk_fraction: float,
) -> dict[str, float]:
    """Compute the registered pair-ranking fidelity series for one epoch."""
    residual = logits - s0
    s0_std = float(np.std(s0))
    residual_std = float(np.std(residual))
    tau_result = kendalltau(s0, logits)
    tau = float(tau_result.statistic)
    if not np.isfinite(tau):
        tau = 1.0 if np.array_equal(s0, logits) else 0.0
    topk = max(1, min(len(s0), int(round(len(s0) * topk_fraction))))
    row_ids = np.arange(len(s0), dtype=np.int64)
    s0_top = set(np.lexsort((row_ids, -s0))[:topk].tolist())
    logit_top = set(np.lexsort((row_ids, -logits))[:topk].tolist())
    return {
        "s0_std": s0_std,
        "s1_mean": float(np.mean(channels["s1"])),
        "s1_std": float(np.std(channels["s1"])),
        "s2_mean": float(np.mean(channels["s2"])),
        "s2_std": float(np.std(channels["s2"])),
        "s2_aa_mean": float(np.mean(channels["s2_aa"])),
        "s2_aa_std": float(np.std(channels["s2_aa"])),
        "residual_std": residual_std,
        "residual_s0_std_ratio": residual_std / max(s0_std, 1e-30),
        "kendall_tau_vs_s0": tau,
        "rank_mobility": 1.0 - tau,
        "topk_overlap": len(s0_top & logit_top) / topk,
    }


def _e2e_validation_batch(
    data: EgoStitchData,
    token_table: PackedFeatureTable,
    token_node_index: Mapping[str, int],
    rows: Sequence[tuple[str, str, int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build one `egostitch_e2e` validation batch (real grounding, no `s0`)."""
    endpoints_u = [u for u, _, _ in rows]
    endpoints_v = [v for _, v, _ in rows]
    idx_i = torch.tensor([data.node_index[u] for u in endpoints_u], dtype=torch.long)
    idx_j = torch.tensor([data.node_index[v] for v in endpoints_v], dtype=torch.long)
    pool_rows_u = data.grounding_index[[data.train_pos[u] for u in endpoints_u]]
    pool_rows_v = data.grounding_index[[data.train_pos[v] for v in endpoints_v]]
    batch = _gather_token_streams(token_table, token_node_index, endpoints_u, endpoints_v)
    batch["x_a"] = data.f0[idx_i]
    batch["x_b"] = data.f0[idx_j]
    batch["ground_a"] = data.f0[torch.from_numpy(pool_rows_u)]
    batch["ground_b"] = data.f0[torch.from_numpy(pool_rows_v)]
    batch["ground_id_a"] = torch.from_numpy(pool_rows_u)
    batch["ground_id_b"] = torch.from_numpy(pool_rows_v)
    return cast(dict[str, torch.Tensor], _to_device(batch, device))


def _validate_epoch(
    model: EgoStitchStage1 | EgoStitchE2E,
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
    pair is scored through the model's full (all-pathways-active, eval-mode
    -- no branch dropout) logit arm; there is no `s0` fusion and no self/
    non-self split (a self pair is simply ``x_a == x_b``/``emb_a == emb_b``,
    handled internally by `EgoStitchE2E.forward`). `token_table`/
    `token_node_index` (the packed raw-token store `_BatchFactory` already
    loaded) are required for this family to build the ``emb_a``/``emb_b``
    batch keys. The per-epoch `topology_delta_std` telemetry (spec Sec
    13.17) and its checkpoint-selection tie-break ratio (spec Sec 13.8) are
    computed once, via `_e2e_topology_delta_std`, on rank 0's first
    validation chunk -- a genuinely *fixed* slice, since neither
    `data.val_pairs`'s order nor the sharding parameters change across
    epochs.
    """
    is_e2e = isinstance(model, EgoStitchE2E)
    model.eval()
    n_val = len(data.val_pairs)
    rank, world = accelerator.process_index, accelerator.num_processes
    shard_rows = list(range(rank, n_val, world))
    shard_len = (n_val + world - 1) // world
    while len(shard_rows) < shard_len:
        shard_rows.append(shard_rows[0] if shard_rows else 0)

    values_out: list[torch.Tensor] = []
    first_chunk_batch: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for start in range(0, shard_len, edge_batch):
            chunk = shard_rows[start : start + edge_batch]
            rows = [(*data.val_pairs[i], int(data.val_labels[i])) for i in chunk]

            if isinstance(model, EgoStitchE2E):
                assert token_table is not None and token_node_index is not None, (
                    "family egostitch_e2e requires token_table/token_node_index"
                )
                e2e_batch = _e2e_validation_batch(
                    data, token_table, token_node_index, rows, accelerator.device
                )
                # The packed token store is bf16-only (src/data/packed_features.py);
                # `accelerator.prepare()` autocasts the training-step forward
                # (`wrapped(...)`) automatically, but this call goes straight
                # to the unwrapped model (matching the frozen-s0 branch below),
                # so autocast must be requested explicitly here too (spec Sec
                # 13.16 extension: per-node encode may stay bf16; this repeats
                # that same contract at validation time).
                masks = (
                    None
                    if model.cfg.permanent_null == "none"
                    else masks_for_null(
                        model.cfg.permanent_null,
                        e2e_batch["x_a"].shape[0],
                        accelerator.device,
                    )
                )
                with accelerator.autocast():
                    full_logits = model(e2e_batch, masks=masks)["logits"]
                values_out.append(full_logits.float().unsqueeze(-1))
                if start == 0:
                    first_chunk_batch = e2e_batch
                continue

            idx_i = torch.tensor([data.node_index[u] for u, _, _ in rows], dtype=torch.long)
            idx_j = torch.tensor([data.node_index[v] for _, v, _ in rows], dtype=torch.long)
            s0 = data.s0.lookup([(u, v) for u, v, _ in rows])
            pool_rows = data.grounding_index[[data.train_pos[u] for u, _, _ in rows]]
            pool_rows_j = data.grounding_index[[data.train_pos[v] for _, v, _ in rows]]
            is_self = torch.tensor([u == v for u, v, _ in rows], dtype=torch.bool)
            batch: dict[str, torch.Tensor] = {
                "x_i": data.f0[idx_i],
                "x_j": data.f0[idx_j],
                "ground_i": data.f0[torch.from_numpy(pool_rows)],
                "ground_j": data.f0[torch.from_numpy(pool_rows_j)],
                "s0": torch.from_numpy(s0),
                "is_self": is_self,
            }
            batch = cast(dict[str, torch.Tensor], _to_device(batch, accelerator.device))
            outputs = {
                name: batch["s0"].new_zeros(batch["s0"].shape)
                for name in ("logits", "s1", "s2", "s2_aa")
            }
            non_self = ~batch["is_self"]
            if bool(non_self.any()):
                sel = torch.nonzero(non_self, as_tuple=False).squeeze(-1)
                keys = ("x_i", "x_j", "ground_i", "ground_j", "s0")
                sub = {k: batch[k][sel] for k in keys}
                sub_out = model(sub)
                for name in outputs:
                    outputs[name] = outputs[name].index_put((sel,), sub_out[name])
            if bool(batch["is_self"].any()):
                sel = torch.nonzero(batch["is_self"], as_tuple=False).squeeze(-1)
                sub = {k: batch[k][sel] for k in ("x_i", "ground_i", "s0")}
                sub_out = model(sub)
                for name in outputs:
                    outputs[name] = outputs[name].index_put((sel,), sub_out[name])
            values_out.append(
                torch.stack(
                    [
                        outputs["logits"].float(),
                        batch["s0"].float(),
                        outputs["s1"].float(),
                        outputs["s2"].float(),
                        outputs["s2_aa"].float(),
                    ],
                    dim=-1,
                )
            )

    n_cols = 1 if is_e2e else 5
    local_values = (
        torch.cat(values_out) if values_out else torch.zeros((0, n_cols), device=accelerator.device)
    )
    local_rows = torch.tensor(shard_rows, dtype=torch.long, device=accelerator.device)
    gathered_values = accelerator.gather(local_values)
    gathered_rows = accelerator.gather(local_rows)
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

    if isinstance(model, EgoStitchE2E):
        assert first_chunk_batch is not None
        if model.cfg.permanent_null == "none":
            with accelerator.autocast():
                topology_summary = _e2e_topology_delta_std(model, first_chunk_batch)
                with torch.no_grad():
                    first_decomposed = model.decompose(first_chunk_batch)
            f_logit_std = float(np.std(first_decomposed["f_logit"].detach().float().cpu().numpy()))
            fidelity = {
                **topology_summary,
                "f_logit_std": f_logit_std,
                "topology_delta_ratio": (
                    topology_summary["topology_delta_std"] / max(f_logit_std, 1e-30)
                ),
            }
        else:
            fidelity = _e2e_null_arm_tiebreak(logits_np)
    else:
        fidelity = _fidelity_summary(
            ordered[:, 1],
            logits_np,
            {"s1": ordered[:, 2], "s2": ordered[:, 3], "s2_aa": ordered[:, 4]},
            topk_fraction=topk_fraction,
        )
    return _ValidationResult(
        metrics=compute_edge_metrics(data.val_labels.astype(np.int64), probs),
        fidelity=fidelity,
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


def train_egostitch_ddp_loop(
    model: EgoStitchStage1 | EgoStitchE2E,
    cfg: EgoConfig,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    node_batch: int,
) -> EgoTrainResult:
    """Run the fixed-epoch Stage-1 training loop (any world size >= 1).

    Emits the exact runtime-profile schema the orchestrator validates and the
    Task-4 checkpoint state via `EgoTrainResult`.

    Family `egostitch_e2e` (design rev 3): `model` is an `EgoStitchE2E`
    instead of a frozen-s0 `EgoStitchStage1`. The optimizer is built over
    `_e2e_trainable_parameters(model)` (excludes the dead, never-called
    `DecisionHead`); there is no ``set_density_ratio``/two-pass calibration
    for this family (that mechanism belongs to `generator.decision`, itself
    unused). Everything else -- Accelerate wiring, the warm-start/joint-
    weight curriculum, the budget guard, and the Sec 13.17 gradient-probe
    emission cadence -- is reused unchanged.

    Args:
        model: The Stage-1 (pass-1 density ratio applied here) or e2e model.
        cfg: The validated worker config.
        data: The assembled data bundle.
        accelerator: The (DDP or single-process) accelerator.
        node_batch: Per-rank ``B_n`` (the orchestrator-selected candidate).

    Returns:
        The `EgoTrainResult`.
    """
    is_e2e = isinstance(model, EgoStitchE2E)
    e2e_permanent_null = cfg.model.config.get("permanent_null", "none")
    assert isinstance(e2e_permanent_null, str)
    model_cfg = model.generator_cfg if isinstance(model, EgoStitchE2E) else model.config
    if isinstance(model, EgoStitchStage1):
        model.set_density_ratio(1.0)  # pass-1 scores are the Stage-1 scores (Sec 13.11)
    world = accelerator.num_processes
    rank = accelerator.process_index
    use_cuda = accelerator.device.type == "cuda"

    composite = _CompositeStep(model, world)
    if isinstance(model, EgoStitchE2E):
        optimizer_parameters: list[torch.nn.Parameter] = _e2e_trainable_parameters(model)
    else:
        optimizer_parameters = list(composite.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    warmup = max(cfg.optim.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    wrapped, optimizer, scheduler = accelerator.prepare(composite, optimizer, scheduler)

    factory = _BatchFactory(
        cfg, model_cfg, data, node_batch=node_batch, rank=rank, world_size=world
    )
    rows_per_rank, steps_per_epoch = _epoch_step_plan(
        len(data.e_sup_positives),
        negative_ratio=cfg.data.negative_ratio,
        edge_batch=cfg.data.edge_batch,
        world_size=world,
    )
    total_steps = steps_per_epoch * cfg.optim.epochs
    warmstart_steps = int(cfg.optim.warmstart_fraction * total_steps)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    history: list[dict[str, object]] = []
    per_epoch_profiles: list[dict[str, object]] = []
    best_metrics: EdgeMetrics | None = None
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    best_fidelity_ratio = -math.inf
    evals_since_improvement = 0
    counterfactual_stop_epoch: int | None = None
    last_metrics: EdgeMetrics | None = None
    global_step = 0
    fixed_gradient_probe: dict[str, object] | None = None
    imbalance_monitor = _GradientImbalanceMonitor(
        ratio=cfg.diagnostics.gradient_imbalance_ratio,
        required_steps=cfg.diagnostics.gradient_imbalance_steps,
        interval=cfg.diagnostics.gradient_probe_interval,
    )
    gradient_norm_series: list[dict[str, object]] = []
    total_local_pairs = 0
    total_local_tokens = 0
    total_wall = 0.0
    total_data_wait = 0.0
    total_validation_seconds = 0.0
    for epoch in range(1, cfg.optim.epochs + 1):
        epoch_started = time.monotonic()
        epoch_data_wait = 0.0
        epoch_local_pairs = 0
        epoch_local_tokens = 0
        epoch_global_pairs = 0
        epoch_steps = 0
        batches = factory.epoch_batches(epoch, rows_per_rank=rows_per_rank, steps=steps_per_epoch)
        parts: dict[str, float] = {}
        epoch_gradient_probes: list[dict[str, object]] = []
        for step_in_epoch, batch in enumerate(batches):
            fetch_started = time.monotonic()
            joint_weight = 0.0 if global_step < warmstart_steps else 1.0
            payload: dict[str, object] = {
                "node": _to_device(batch.node, accelerator.device),
                "edge": _to_device(batch.edge, accelerator.device),
                "joint_weight": torch.tensor(joint_weight, device=accelerator.device),
                "edge_rows_global": batch.edge_rows_global,
            }
            if is_e2e:
                # `_CompositeStep.forward`'s e2e branch seeds per-step branch-
                # dropout masks from this triple (`_seeded_generator(seed,
                # epoch, step)`), the same convention `_BatchFactory` used
                # internally to build this exact batch.
                payload["seed"] = cfg.seed
                payload["epoch"] = epoch
                payload["step"] = step_in_epoch
            if joint_weight > 0.0 and fixed_gradient_probe is None:
                fixed_gradient_probe = cast(dict[str, object], _detached_clone(payload))
                fixed_gradient_probe["collect_diagnostics"] = True
            epoch_data_wait += time.monotonic() - fetch_started

            out = wrapped(payload)
            loss = cast(torch.Tensor, out["loss"])
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(wrapped.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()

            parts = cast(dict[str, float], out["parts"])
            global_step += 1
            if (
                fixed_gradient_probe is not None
                and global_step % cfg.diagnostics.gradient_probe_interval == 0
            ):
                sync_context = wrapped.no_sync() if hasattr(wrapped, "no_sync") else nullcontext()
                with sync_context:
                    probe_out = wrapped(fixed_gradient_probe)
                    local_norms = _family_gradient_norms(
                        accelerator.unwrap_model(wrapped).model,
                        cast(dict[str, torch.Tensor], probe_out["families"]),
                    )
                names = ("edge", "recon", "real", "ssl")
                local_vector = torch.tensor(
                    [local_norms[name] for name in names],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                gathered_norms = accelerator.gather(local_vector).reshape(-1, len(names))
                aggregate = torch.sqrt(torch.mean(gathered_norms.square(), dim=0)).cpu().numpy()
                norms = {
                    name: float(value)
                    for name, value in zip(names, aggregate.tolist(), strict=True)
                }
                activated_now = imbalance_monitor.update(global_step, norms)
                if activated_now:
                    accelerator.unwrap_model(wrapped).activate_kendall()
                probe_record: dict[str, object] = {
                    "step": global_step,
                    **{f"grad_norm_{name}": value for name, value in norms.items()},
                    "imbalance_streak_steps": imbalance_monitor.streak_steps,
                    "kendall_activated_now": activated_now,
                    "kendall_active": imbalance_monitor.activated_step is not None,
                }
                # Family egostitch_e2e (spec Sec 13.17 re-registration): the
                # gate-tanh readouts are pure parameter reads already computed
                # by `_CompositeStep.forward`'s `collect_diagnostics` branch --
                # no extra backward needed, just surface them on this row too.
                for gate_key in ("gate_topo_tanh", "gate_cont_tanh"):
                    if gate_key in probe_out:
                        probe_record[gate_key] = probe_out[gate_key]
                epoch_gradient_probes.append(probe_record)
                gradient_norm_series.append(probe_record)
            epoch_steps += 1
            epoch_local_pairs += batch.edge_rows_true
            epoch_local_tokens += batch.f0_rows_gathered
            epoch_global_pairs += batch.edge_rows_global

        val_started = time.monotonic()
        validation = _validate_epoch(
            model,
            data,
            accelerator,
            edge_batch=cfg.data.edge_batch,
            topk_fraction=cfg.diagnostics.topk_fraction,
            token_table=factory._token_table,
            token_node_index=factory._token_node_index,
        )
        validation_seconds = time.monotonic() - val_started
        epoch_wall = time.monotonic() - epoch_started

        if accelerator.is_main_process:
            assert validation is not None
            metrics = validation.metrics
            fidelity = validation.fidelity
            last_metrics = metrics
            # Sec 13.8 checkpoint-selection tie-break: family egostitch_e2e
            # (s0 retired) uses std(full - f_logit)/std(f_logit) in place of
            # the historical residual/s0 ratio -- same direction (larger
            # value wins), same tolerance/tie-break mechanics below.
            fidelity_ratio = (
                fidelity["topology_delta_ratio"]
                if is_e2e and e2e_permanent_null == "none"
                else fidelity["selection_tiebreak"]
                if is_e2e
                else fidelity["residual_s0_std_ratio"]
            )
            improved = best_metrics is None or metrics.auprc > (
                best_metrics.auprc + cfg.diagnostics.selection_auprc_tolerance
            )
            if (
                best_metrics is not None
                and abs(metrics.auprc - best_metrics.auprc)
                <= cfg.diagnostics.selection_auprc_tolerance
                and fidelity_ratio > best_fidelity_ratio
            ):
                improved = True
            if improved:
                best_metrics = metrics
                best_epoch = epoch
                best_state = _cpu_state_dict(accelerator, wrapped)
                best_fidelity_ratio = fidelity_ratio
                evals_since_improvement = 0
            else:
                evals_since_improvement += 1
                if (
                    counterfactual_stop_epoch is None
                    and evals_since_improvement >= cfg.eval.patience
                ):
                    counterfactual_stop_epoch = epoch
            history.append(
                {
                    "epoch": float(epoch),
                    "auroc": metrics.auroc,
                    "auprc": metrics.auprc,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "fidelity": fidelity,
                    "gradient_norm_probes": epoch_gradient_probes,
                    "kendall_active": imbalance_monitor.activated_step is not None,
                    **{f"loss_{name}": value for name, value in parts.items()},
                }
            )

        per_epoch_profiles.append(
            {
                "epoch": epoch,
                "steps": max(epoch_steps, 1),
                "global_pairs": max(epoch_global_pairs, 1),
                "local_pairs": max(epoch_local_pairs, 1),
                "local_tokens": max(epoch_local_tokens, 1),
                "wall_seconds": epoch_wall,
                "data_wait_seconds": epoch_data_wait,
                "compute_seconds": max(epoch_wall - epoch_data_wait - validation_seconds, 0.0),
                "validation_seconds": validation_seconds,
            }
        )
        total_wall += epoch_wall
        total_data_wait += epoch_data_wait
        total_validation_seconds += validation_seconds
        total_local_pairs += epoch_local_pairs
        total_local_tokens += epoch_local_tokens

    # ---- runtime profile (the exact orchestrator-validated schema)
    local_peak_gib = (
        torch.cuda.max_memory_allocated(accelerator.device) / (1024**3) if use_cuda else 0.0
    )
    stats = torch.tensor(
        [
            float(total_local_pairs),
            float(total_local_tokens),
            total_wall,
            total_data_wait,
            local_peak_gib,
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered = accelerator.gather(stats.unsqueeze(0))
    rank_stats = gathered.cpu().numpy()
    per_rank = [
        {
            "rank": r,
            "pairs": max(int(row[0]), 1),
            "batches": max(int(global_step), 1),
            "steps": max(int(global_step), 1),
            "tokens": max(int(row[1]), 1),
            "train_wall_seconds": float(row[2]),
            "data_wait_seconds": float(row[3]),
            "pairs_per_second": float(row[0] / row[2]) if row[2] > 0 else 0.0,
            "tokens_per_second": float(row[1] / row[2]) if row[2] > 0 else 0.0,
        }
        for r, row in enumerate(rank_stats)
    ]
    data_wait_fraction = max((float(row[3] / row[2]) if row[2] > 0 else 0.0) for row in rank_stats)
    global_pairs = int(sum(entry["global_pairs"] for entry in per_epoch_profiles))  # type: ignore[misc]
    slowest_wall = max(float(row[2]) for row in rank_stats)
    global_tokens = int(sum(float(row[1]) for row in rank_stats))
    epochs_completed = len(per_epoch_profiles)
    runtime_profile: dict[str, object] = {
        "epochs_completed": epochs_completed,
        "validations_completed": epochs_completed,
        "peak_memory_gib_per_rank": [float(row[4]) for row in rank_stats],
        "steady_state_data_wait_fraction": data_wait_fraction,
        "training_coverage_exact": True,
        "validation_coverage_exact": True,
        "feature_cache_hit_rate": 1.0,
        "counterfactual_stop_epoch": counterfactual_stop_epoch,
        "per_rank": per_rank,
        "global_pairs": global_pairs,
        "global_pairs_per_second": global_pairs / slowest_wall if slowest_wall > 0 else 0.0,
        "global_tokens": global_tokens,
        "global_tokens_per_second": global_tokens / slowest_wall if slowest_wall > 0 else 0.0,
        "validation_seconds": total_validation_seconds,
        "per_epoch": per_epoch_profiles,
        "gradient_norm_series": gradient_norm_series,
        "kendall_fallback": {
            "active": imbalance_monitor.activated_step is not None,
            "activated_step": imbalance_monitor.activated_step,
            "imbalance_streak_steps": imbalance_monitor.streak_steps,
        },
    }

    if accelerator.is_main_process:
        assert last_metrics is not None and best_metrics is not None
        last_state = _cpu_state_dict(accelerator, wrapped)
        if not best_state:
            best_state = last_state
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
        last_state = {}
        best_metrics = placeholder
        last_metrics = placeholder

    return EgoTrainResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_metrics=best_metrics,
        last_state_dict=last_state,
        last_epoch=epochs_completed,
        last_val_metrics=last_metrics,
        history=history,
        counterfactual_stop_epoch=counterfactual_stop_epoch,
        runtime_profile=runtime_profile,
        kendall_state={
            "active": imbalance_monitor.activated_step is not None,
            "activated_step": imbalance_monitor.activated_step,
            "log_variances": {
                name: float(value.detach().cpu())
                for name, value in accelerator.unwrap_model(wrapped).kendall_log_vars.items()
            },
        },
    )


# --------------------------------------------------------------------------- artifacts


def _config_hash(cfg: EgoConfig) -> str:
    return hashlib.sha256(
        json.dumps(config_to_dict(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_run_start_metadata(cfg: EgoConfig, data: EgoStitchData, *, world_size: int) -> None:
    """Bind the run to config, preregistration, and s0 before optimization."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "run_metadata.json"
    if path.exists():
        raise FileExistsError(f"run-start metadata already exists: {path}")
    metadata = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "config_hash": _config_hash(cfg),
        "preregistration_sha256": _sha256_file(cfg.preregistration),
        "seed": cfg.seed,
        "world_size": world_size,
        "s0_checkpoint_id": cfg.data.s0_checkpoint_id,
        "partition_seed": cfg.data.partition_seed,
        "rho_train": data.rho_train,
        "positives_mode": cfg.data.train_positives,
        "permanent_null": cfg.model.config.get("permanent_null", "none"),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_outputs(result: EgoTrainResult, cfg: EgoConfig, data: EgoStitchData) -> None:
    """Write the pinned Task-4 artifacts + the pre-registration binding.

    ``best.pt``/``last.pt`` carry exactly the seven pinned payload keys;
    ``run_metadata.json`` additionally records ``preregistration_sha256``
    (protocol Sec 5.2.4), the s0 checkpoint identity, the partition seed, and
    the measured ``rho_train``.
    """
    output_dir = cfg.output_dir
    config_dict = config_to_dict(cfg)
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("run-start metadata is missing; refuse post-hoc preregistration binding")
    run_metadata = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
    current_prereg_sha = _sha256_file(cfg.preregistration)
    if run_metadata.get("preregistration_sha256") != current_prereg_sha:
        raise RuntimeError(
            "preregistration changed after run start; refusing to finalize artifacts"
        )
    if run_metadata.get("config_hash") != _config_hash(cfg):
        raise RuntimeError("configuration changed after run start; refusing to finalize artifacts")

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

    torch.save(
        payload(result.best_state_dict, result.best_epoch, result.best_val_metrics),
        output_dir / "best.pt",
    )
    torch.save(
        payload(result.last_state_dict, result.last_epoch, result.last_val_metrics),
        output_dir / "last.pt",
    )
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.history:
            handle.write(json.dumps({**entry, "epoch": int(cast(float, entry["epoch"]))}) + "\n")

    run_metadata.update(
        {
            "status": "complete",
            "checkpoint_id": _state_digest(result.best_state_dict)[:16],
            "kendall_fallback": result.kendall_state,
            "training_diagnostics": {
                "fidelity_series": [entry["fidelity"] for entry in result.history],
                "gradient_norm_series": result.runtime_profile["gradient_norm_series"],
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


def _run_probe_mode(
    model: EgoStitchStage1 | EgoStitchE2E,
    cfg: EgoConfig,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    node_batch: int,
    profile_output: Path,
) -> None:
    """Warm-up + timed composite steps -> one rank-zero `ProbeResult` JSON."""
    runtime = cfg.runtime
    if runtime is None:
        raise ValueError("probe mode requires a configured cfg.runtime")
    is_e2e = isinstance(model, EgoStitchE2E)
    if isinstance(model, EgoStitchStage1):
        model.set_density_ratio(1.0)
    model_cfg = model.generator_cfg if isinstance(model, EgoStitchE2E) else model.config
    world = accelerator.num_processes
    composite = _CompositeStep(model, world)
    if isinstance(model, EgoStitchE2E):
        optimizer_parameters: list[torch.nn.Parameter] = _e2e_trainable_parameters(model)
    else:
        optimizer_parameters = list(composite.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    wrapped, optimizer = accelerator.prepare(composite, optimizer)
    factory = _BatchFactory(
        cfg,
        model_cfg,
        data,
        node_batch=node_batch,
        rank=accelerator.process_index,
        world_size=world,
    )
    rows_per_rank, steps_per_epoch = _epoch_step_plan(
        len(data.e_sup_positives),
        negative_ratio=cfg.data.negative_ratio,
        edge_batch=cfg.data.edge_batch,
        world_size=world,
    )

    def batch_iterator() -> Iterator[_CompositeBatch]:
        epoch = 1
        while True:
            yield from factory.epoch_batches(
                epoch, rows_per_rank=rows_per_rank, steps=steps_per_epoch
            )
            epoch += 1

    use_cuda = accelerator.device.type == "cuda"
    wrapped.train()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    warmup, timed = runtime.probe_warmup_steps, runtime.probe_timed_steps
    iterator = batch_iterator()
    timed_global_pairs = 0
    timed_start: float | None = None
    failure: str | None = None
    for step in range(warmup + timed):
        if step == warmup:
            accelerator.wait_for_everyone()
            if use_cuda:
                torch.cuda.synchronize(accelerator.device)
            timed_start = time.monotonic()
        batch = next(iterator)
        payload: dict[str, object] = {
            "node": _to_device(batch.node, accelerator.device),
            "edge": _to_device(batch.edge, accelerator.device),
            "joint_weight": torch.tensor(1.0, device=accelerator.device),
            "edge_rows_global": batch.edge_rows_global,
            "collect_diagnostics": step == 0,
        }
        if is_e2e:
            # Mirrors `batch_iterator`'s own epoch/step-within-epoch cycling
            # (`factory.epoch_batches(epoch, ...)` yields exactly
            # `steps_per_epoch` batches per epoch), so this matches the
            # seeding `_BatchFactory` used internally for this same batch.
            payload["seed"] = cfg.seed
            payload["epoch"] = step // steps_per_epoch + 1
            payload["step"] = step % steps_per_epoch
        loss: torch.Tensor | None = None
        local_failure: tuple[str, str] | None = None
        s1_abs_mean: float | None = None
        try:
            probe_out = wrapped(payload)
            loss = cast(torch.Tensor, probe_out["loss"])
            if not bool(torch.isfinite(loss).all()):
                local_failure = ("nonfinite", "non-finite probe loss")
            # Family egostitch_e2e has no s1/s2 channels (spec Sec 13.17
            # re-registration retires this frozen-s0-specific guard for that
            # family; its liveness telemetry is `_e2e_gate_tanh`/
            # `topology_delta_std`, checked per-epoch in `_validate_epoch`
            # instead of at probe time).
            if step == 0 and not is_e2e:
                channel_stats = cast(dict[str, float], probe_out["channel_stats"])
                s1_abs_mean = abs(channel_stats["s1_mean"])
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
            kind, message = local_failure or ("oom", "probe candidate failed on another rank")
            _emit_probe_candidate_failure(kind, message)
            raise RuntimeError(message)
        if step == 0 and not is_e2e:
            assert s1_abs_mean is not None
            global_s1_abs_mean = float(
                accelerator.gather(
                    torch.tensor([s1_abs_mean], device=accelerator.device, dtype=torch.float64)
                )
                .max()
                .item()
            )
            _enforce_probe_s1_scale(global_s1_abs_mean, cfg.diagnostics.probe_s1_abs_mean_max)
        if loss is None:  # pragma: no cover - collective failure raises above
            raise RuntimeError("probe forward produced no loss")
        optimizer.zero_grad()
        # DDP collectives may be in flight past this point: exceptions must
        # escape immediately (deferring them can deadlock the other ranks).
        try:
            accelerator.backward(loss)
            if cfg.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(wrapped.parameters(), cfg.optim.grad_clip)
            optimizer.step()
        except RuntimeError as error:
            if _is_oom_error(error):
                _emit_probe_candidate_failure("oom", str(error))
            raise
        if step >= warmup:
            timed_global_pairs += batch.edge_rows_global
    if use_cuda:
        torch.cuda.synchronize(accelerator.device)

    local_elapsed = time.monotonic() - timed_start if timed_start is not None else 0.0
    elapsed = float(
        accelerator.gather(
            torch.tensor([local_elapsed], device=accelerator.device, dtype=torch.float64)
        )
        .max()
        .item()
    )
    local_peak_gib = (
        torch.cuda.max_memory_allocated(accelerator.device) / (1024**3) if use_cuda else 0.0
    )
    peak = float(
        accelerator.gather(
            torch.tensor([local_peak_gib], device=accelerator.device, dtype=torch.float32)
        )
        .max()
        .item()
    )
    throughput = timed_global_pairs / elapsed if elapsed > 0 and failure is None else 0.0
    probe = ProbeResult(
        token_budget=node_batch,
        valid=failure is None,
        global_pairs_per_second=throughput,
        peak_memory_gib=peak,
        failure=failure,
    )
    _write_json_rank_zero(accelerator, profile_output, probe.to_dict())
    logger.info("probe complete: %s", probe.to_dict())


def _run_ddp_worker(cfg: EgoConfig, args: EgoCliArgs) -> None:
    """Dispatch an ``accelerate launch`` worker to the requested DDP mode."""
    if args.pack_dir is None or args.token_budget_per_rank is None or args.profile_output is None:
        raise ValueError(
            "DDP worker modes require --pack-dir, --token-budget-per-rank, and --profile-output"
        )
    if not cfg.preregistration.is_file():
        raise ValueError(f"preregistration file not found: {cfg.preregistration}")

    accelerator = build_egostitch_ddp_accelerator(cfg.mixed_precision)
    set_seed(cfg.seed)
    logger.info(
        "egostitch ddp worker mode=%s rank=%d/%d device=%s",
        args.ddp_mode,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    data = assemble_egostitch_data(cfg, pack_dir=args.pack_dir)
    model: EgoStitchStage1 | EgoStitchE2E
    if cfg.model.family == _EGOSTITCH_E2E_FAMILY:
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))
    else:
        model = EgoStitchStage1(EgoStitchConfig.from_mapping(cfg.model.config))
    node_batch = args.token_budget_per_rank

    if args.ddp_mode == "probe":
        _run_probe_mode(
            model,
            cfg,
            data,
            accelerator,
            node_batch=node_batch,
            profile_output=args.profile_output,
        )
        return

    if args.ddp_mode == "epoch-probe":
        one_epoch_cfg = replace(cfg, optim=replace(cfg.optim, epochs=1))
        result, elapsed = _run_timed_epoch_probe(
            accelerator,
            lambda: train_egostitch_ddp_loop(
                model, one_epoch_cfg, data, accelerator, node_batch=node_batch
            ),
        )
        _write_json_rank_zero(
            accelerator,
            args.profile_output,
            {"epoch_seconds": elapsed, "runtime_profile": result.runtime_profile},
        )
        logger.info("epoch-probe complete: %.2fs", elapsed)
        return

    if accelerator.is_main_process:
        write_run_start_metadata(cfg, data, world_size=accelerator.num_processes)
    accelerator.wait_for_everyone()
    result = train_egostitch_ddp_loop(model, cfg, data, accelerator, node_batch=node_batch)
    if accelerator.is_main_process:
        write_outputs(result, cfg, data)
    _write_json_rank_zero(accelerator, args.profile_output, result.runtime_profile)
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
    cfg = apply_overrides(load_config(args.config), args)

    if args.write_s0_manifest is not None:
        world = detect_visible_gpu_count()
        data = assemble_egostitch_data(cfg, require_s0=False)
        build_s0_manifest(
            cfg,
            data.e_sup_positives,
            data.val_pairs,
            data.sampler,
            args.write_s0_manifest,
            world_size=world,
        )
        return

    if args.ddp_mode is not None:
        _run_ddp_worker(cfg, args)
        return

    raise ValueError(
        "EgoStitch Stage-1 training must run through src.e2_pipeline so the visible "
        "GPU count is auto-detected and workers are launched with Accelerate DDP"
    )


if __name__ == "__main__":
    main()
