r"""Universe/pairs scoring CLI with sharding and the pinned scores artifact.

Scores node-pair lists with a frozen B0-family checkpoint (Task-4 format:
``{"model_state", "model_family", "model_config", ...}``, or a bare legacy state
dict with explicit ``--model-family`` / ``--model-config`` metadata) and writes the single
self-contained ``.npz`` artifact the G1/G2 analyses consume. Pairs come from the
candidate universe, the val/test edge files, or an arbitrary TSV
(``u\tv[\tlabel]``). Supports optional contiguous row-range sharding
plus a ``merge`` subcommand that validates and concatenates shard outputs.

CLI::

    python -m src.score_universe score \
        --checkpoint outputs/b0_v31/best.pt \
        --pairs candidate|test|val|file:<path.tsv> \
        --data-root data --strategy breadth_first \
        --output scores/b0_v31_candidate.npz \
        [--batch-pairs 8192] [--token-budget 131072] [--device auto|cpu|cuda|mps] \
        [--amp off|bf16] [--shard K --num-shards N]

    python -m src.score_universe merge \
        --inputs scores/b0_v31_candidate.shard-*.npz \
        --output scores/b0_v31_candidate.npz

Artifact format (pinned — do not drift):

- ``node_ids``: unique node-id strings, sorted ascending.
- ``u_idx``/``v_idx``: int32, row-aligned with the input pair order (pairs are
  stored canonicalized: ``(min(u, v), max(u, v))``), indices into ``node_ids``.
- ``logit``: float32 raw model logits (NOT sigmoid).
- ``label``: int8, ``-1`` when unlabeled.
- ``row_start``: int64 scalar (0 for unsharded/merged; shard offset for shards).
- ``meta``: 0-d JSON string array with keys ``checkpoint_id``, ``model_family``,
  ``pairs_source``, ``strategy``, ``num_rows``, ``created_utc``, ``torch_version``;
  EgoStitch artifacts additionally pin pair-pass precision provenance and
  descriptive score-resolution diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from torch import nn

from src.data.artifacts import canonical_pair, load_candidate_pairs
from src.data.features import FeatureStore, build_f0_matrix
from src.data.packed_features import PackedFeatureTable
from src.data.pairs import (
    BUCKET_BOUNDARIES,
    LengthBucketedBatchSampler,
    TokenPairDataset,
    collate_token_pairs,
    probe_lengths,
)
from src.model.B0 import V3_1

logger = logging.getLogger(__name__)

_LOG_EVERY_ROWS = 50_000
_FEATURES_SUBDIR = Path("features") / "frozen_node_features_1024"
_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_META_KEYS = (
    "checkpoint_id",
    "model_family",
    "pairs_source",
    "strategy",
    "num_rows",
    "created_utc",
    "torch_version",
)
_NAMED_PAIR_SOURCES = ("candidate", "test", "val")
_EGOSTITCH_PAIR_PRECISION_CONTRACT = "egostitch_pair_fp32_v1"


# ---------------------------------------------------------------------------
# Scores artifact (pinned format) — save / load / merge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoresArtifact:
    """In-memory view of one pinned scores ``.npz`` artifact.

    Attributes:
        node_ids: Unique node-id strings, sorted ascending.
        u_idx: Shape ``(n,)`` int32 indices into `node_ids`, in input row order.
        v_idx: Shape ``(n,)`` int32 indices into `node_ids`, aligned with `u_idx`.
        logit: Shape ``(n,)`` float32 raw model logits (not sigmoid).
        label: Shape ``(n,)`` int8 labels; ``-1`` where the input row was unlabeled.
        meta: Parsed artifact metadata (the pinned seven keys).
    """

    node_ids: list[str]
    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]
    logit: NDArray[np.float32]
    label: NDArray[np.int8]
    meta: dict[str, object]

    def pairs(self) -> Iterator[tuple[str, str]]:
        """Yield the ``(u, v)`` node-id pair of each row, in artifact row order."""
        for u, v in zip(self.u_idx.tolist(), self.v_idx.tolist(), strict=True):
            yield self.node_ids[u], self.node_ids[v]

    def probs(self) -> NDArray[np.float64]:
        """Return ``sigmoid(logit)`` as float64 probabilities."""
        logit64 = self.logit.astype(np.float64)
        out = np.empty_like(logit64)
        positive = logit64 >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-logit64[positive]))
        exp_l = np.exp(logit64[~positive])
        out[~positive] = exp_l / (1.0 + exp_l)
        return out


class _Shard(NamedTuple):
    """One raw scores file (including its shard offset), as read from disk."""

    path: Path
    node_ids: list[str]
    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]
    logit: NDArray[np.float32]
    label: NDArray[np.int8]
    row_start: int
    meta: dict[str, object]


def score_resolution_diagnostics(logit: NDArray[np.float32]) -> dict[str, int | float]:
    """Return descriptive resolution diagnostics without judging model ties."""
    values = np.asarray(logit, dtype=np.float32)
    n = len(values)
    if n == 0:
        return {
            "n_rows": 0,
            "n_unique": 0,
            "unique_fraction": 0.0,
            "bf16_grid_fraction": 0.0,
            "fp16_grid_fraction": 0.0,
        }
    bits = values.view(np.uint32)
    n_unique = int(np.unique(values).shape[0])
    with np.errstate(over="ignore", invalid="ignore"):
        fp16_roundtrip = values.astype(np.float16).astype(np.float32)
    return {
        "n_rows": n,
        "n_unique": n_unique,
        "unique_fraction": float(n_unique / n),
        "bf16_grid_fraction": float(np.mean((bits & np.uint32(0xFFFF)) == 0)),
        "fp16_grid_fraction": float(np.mean(fp16_roundtrip == values)),
    }


def validate_score_precision(
    logit: NDArray[np.float32],
    *,
    meta: Mapping[str, object],
    label: str,
    require_diagnostics: bool = True,
) -> None:
    """Validate EgoStitch pair-pass fp32 provenance and stored diagnostics.

    Args:
        logit: The scores artifact's logit column.
        meta: Parsed artifact metadata.
        label: Human-readable artifact label used in errors.
        require_diagnostics: Require the persisted descriptive diagnostics.

    Raises:
        ValueError: If an EgoStitch artifact lacks or contradicts the pinned
            pair-pass fp32 contract, or its stored diagnostics are inconsistent.
    """
    if meta.get("model_family") != "egostitch":
        return
    precision = meta.get("score_precision")
    if not isinstance(precision, dict):
        raise ValueError(
            f"{label}: EgoStitch artifact is missing score_precision provenance; "
            "rescore with the pair-pass fp32 contract"
        )
    expected = {
        "contract": _EGOSTITCH_PAIR_PRECISION_CONTRACT,
        "pair_compute_dtype": "float32",
        "pair_autocast": False,
        "logit_storage_dtype": "float32",
    }
    mismatches = {
        key: (precision.get(key), value)
        for key, value in expected.items()
        if precision.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label}: invalid EgoStitch score_precision provenance: {mismatches}")
    if precision.get("encode_autocast") not in {"off", "bf16"}:
        raise ValueError(f"{label}: score_precision.encode_autocast must be 'off' or 'bf16'")
    if np.asarray(logit).dtype != np.float32:
        raise ValueError(f"{label}: EgoStitch logits must be stored as float32")

    if not require_diagnostics:
        return
    recorded = meta.get("score_resolution")
    actual = score_resolution_diagnostics(logit)
    if recorded != actual:
        raise ValueError(
            f"{label}: score_resolution diagnostics are missing or inconsistent; "
            f"recorded={recorded!r}, actual={actual!r}"
        )


def save_scores(
    path: Path,
    *,
    node_ids: Sequence[str],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    logit: NDArray[np.float32],
    label: NDArray[np.int8],
    row_start: int,
    meta: dict[str, object],
) -> None:
    """Write a scores artifact in the pinned ``.npz`` format.

    Args:
        path: Destination ``.npz`` path (parent directories are created).
        node_ids: Unique node-id strings, sorted ascending.
        u_idx: Shape ``(n,)`` int32 indices into `node_ids`.
        v_idx: Shape ``(n,)`` int32 indices into `node_ids`.
        logit: Shape ``(n,)`` float32 raw model logits.
        label: Shape ``(n,)`` int8 labels (``-1`` for unlabeled rows).
        row_start: Row offset of this artifact in the full input (0 unless sharded).
        meta: Metadata dict; must contain the pinned keys and be JSON-serializable.

    Raises:
        ValueError: If array lengths disagree, indices fall outside `node_ids`,
            `row_start` is negative, `meta` is missing pinned keys, or an
            EgoStitch artifact violates the pair-pass fp32 provenance contract.
    """
    n = len(logit)
    if not (len(u_idx) == len(v_idx) == len(label) == n):
        raise ValueError(
            "u_idx, v_idx, logit, and label must have identical lengths; got "
            f"{len(u_idx)}, {len(v_idx)}, {n}, {len(label)}"
        )
    if n > 0 and (
        int(u_idx.min()) < 0
        or int(v_idx.min()) < 0
        or int(u_idx.max()) >= len(node_ids)
        or int(v_idx.max()) >= len(node_ids)
    ):
        raise ValueError("u_idx/v_idx contain indices outside [0, len(node_ids))")
    if row_start < 0:
        raise ValueError(f"row_start must be >= 0, got {row_start}")
    missing = [key for key in _META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"meta is missing required keys: {missing}")
    stored_meta = dict(meta)
    if stored_meta.get("model_family") == "egostitch":
        validate_score_precision(
            logit, meta=stored_meta, label=str(path), require_diagnostics=False
        )
        stored_meta["score_resolution"] = score_resolution_diagnostics(logit)
        validate_score_precision(logit, meta=stored_meta, label=str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_ids=np.array(list(node_ids), dtype=np.str_),
        u_idx=u_idx.astype(np.int32, copy=False),
        v_idx=v_idx.astype(np.int32, copy=False),
        logit=logit.astype(np.float32, copy=False),
        label=label.astype(np.int8, copy=False),
        row_start=np.int64(row_start),
        meta=np.array(json.dumps(stored_meta, sort_keys=True)),
    )


def _load_shard(path: Path) -> _Shard:
    """Read one scores ``.npz`` file raw, including its ``row_start`` offset.

    Args:
        path: Path to a scores artifact written by :func:`save_scores`.

    Returns:
        The parsed `_Shard`.
    """
    with np.load(path, allow_pickle=False) as data:
        node_ids = [str(node_id) for node_id in data["node_ids"].tolist()]
        u_idx: NDArray[np.int32] = data["u_idx"].astype(np.int32, copy=False)
        v_idx: NDArray[np.int32] = data["v_idx"].astype(np.int32, copy=False)
        logit: NDArray[np.float32] = data["logit"].astype(np.float32, copy=False)
        label: NDArray[np.int8] = data["label"].astype(np.int8, copy=False)
        row_start = int(data["row_start"][()])
        meta = cast(dict[str, object], json.loads(str(data["meta"][()])))
    return _Shard(
        path=path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logit,
        label=label,
        row_start=row_start,
        meta=meta,
    )


def load_scores(path: Path) -> ScoresArtifact:
    """Load a scores artifact from disk.

    Args:
        path: Path to a scores ``.npz`` file written by :func:`save_scores`.

    Returns:
        The loaded `ScoresArtifact` (the ``row_start`` scalar is not exposed here;
        it only matters to :func:`merge_scores`).
    """
    shard = _load_shard(path)
    return ScoresArtifact(
        node_ids=shard.node_ids,
        u_idx=shard.u_idx,
        v_idx=shard.v_idx,
        logit=shard.logit,
        label=shard.label,
        meta=shard.meta,
    )


def merge_scores(inputs: Sequence[Path]) -> ScoresArtifact:
    """Validate and concatenate shard scores files into one full artifact.

    Shards must share ``checkpoint_id``, ``model_family``, ``pairs_source``,
    ``strategy``, and ``num_rows``, and their ``[row_start, row_start + n)``
    ranges must be non-overlapping, contiguous, and cover exactly
    ``[0, num_rows)``. Node-id indices are remapped onto the sorted union of the
    shards' ``node_ids`` vocabularies.

    Args:
        inputs: Paths to the shard ``.npz`` files (any order).

    Returns:
        The merged `ScoresArtifact`, rows ordered by ``row_start``.

    Raises:
        ValueError: If `inputs` is empty, shard metadata disagrees, or the row
            ranges overlap, leave a gap, or do not cover ``[0, num_rows)``.
    """
    if not inputs:
        raise ValueError("merge requires at least one input scores file")
    shards = [_load_shard(path) for path in inputs]

    reference = shards[0]
    for key in (
        "checkpoint_id",
        "model_family",
        "pairs_source",
        "strategy",
        "num_rows",
        "score_precision",
    ):
        values = {str(shard.meta.get(key)) for shard in shards}
        if len(values) > 1:
            raise ValueError(
                f"merge inputs disagree on meta {key!r}: {sorted(values)} "
                f"(files: {[str(shard.path) for shard in shards]})"
            )

    ordered = sorted(shards, key=lambda shard: shard.row_start)
    expected_start = 0
    for shard in ordered:
        if shard.row_start < expected_start:
            raise ValueError(
                f"shard row ranges overlap: {shard.path} starts at row {shard.row_start} "
                f"but a previous shard already covers up to row {expected_start}"
            )
        if shard.row_start > expected_start:
            raise ValueError(
                f"shard row ranges leave a gap: rows [{expected_start}, {shard.row_start}) "
                f"are missing before {shard.path}"
            )
        expected_start += len(shard.logit)
    num_rows = int(cast(int, reference.meta["num_rows"]))
    if expected_start != num_rows:
        raise ValueError(
            f"shards cover rows [0, {expected_start}) but meta num_rows is {num_rows} "
            "(gap at the end or extra rows)"
        )

    node_set: set[str] = set()
    for shard in ordered:
        node_set.update(shard.node_ids)
    node_ids = sorted(node_set)
    position = {node_id: i for i, node_id in enumerate(node_ids)}

    u_parts: list[NDArray[np.int32]] = []
    v_parts: list[NDArray[np.int32]] = []
    for shard in ordered:
        mapping = np.array([position[node_id] for node_id in shard.node_ids], dtype=np.int32)
        u_parts.append(mapping[shard.u_idx] if len(shard.u_idx) else shard.u_idx)
        v_parts.append(mapping[shard.v_idx] if len(shard.v_idx) else shard.v_idx)

    merged_meta = dict(reference.meta)
    profiles = [shard.meta.get("score_profile") for shard in ordered]
    if all(isinstance(profile, dict) for profile in profiles):
        typed_profiles = cast(list[dict[str, object]], profiles)
        merged_meta["score_profile"] = {
            "wall_seconds": max(
                float(cast(float, profile["wall_seconds"])) for profile in typed_profiles
            ),
            "rows": sum(int(cast(int, profile["rows"])) for profile in typed_profiles),
            "unique_nodes": len(node_ids),
            "measurement": "max_concurrent_shard_compute_wall_seconds",
        }

    return ScoresArtifact(
        node_ids=node_ids,
        u_idx=np.concatenate(u_parts) if u_parts else np.empty(0, dtype=np.int32),
        v_idx=np.concatenate(v_parts) if v_parts else np.empty(0, dtype=np.int32),
        logit=np.concatenate([shard.logit for shard in ordered]),
        label=np.concatenate([shard.label for shard in ordered]),
        meta=merged_meta,
    )


# ---------------------------------------------------------------------------
# Model rebuild from the Task-4 checkpoint format
# ---------------------------------------------------------------------------


def _build_v3_1(model_config: dict[str, object]) -> nn.Module:
    """Build a `V3_1` model from its checkpointed config."""
    return V3_1(**model_config)


def _build_f0_mlp(model_config: dict[str, object]) -> nn.Module:
    """Build an `F0PairMLP` model from its checkpointed config.

    The import is deliberately lazy: ``src/model/b0_alt.py`` is delivered by a
    parallel task, and this module must import cleanly regardless of landing
    order. The checkpoint format and forward contract are pinned in
    ``.superpowers/sdd/briefs/task-4-brief.md``.
    """
    from src.model.b0_alt import F0PairMLP

    # Checkpointed configs are inherently dynamic (parsed from a .pt payload),
    # so the kwargs unpack goes through Any deliberately.
    return cast(nn.Module, F0PairMLP(**cast(dict[str, Any], model_config)))


def _build_egostitch(model_config: dict[str, object]) -> nn.Module:
    """Build an `EgoStitchStage1` model from its checkpointed config (spec Sec 13)."""
    from src.model.egostitch import EgoStitchConfig, EgoStitchStage1

    return EgoStitchStage1(EgoStitchConfig.from_mapping(model_config))


MODEL_BUILDERS: dict[str, Callable[[dict[str, object]], nn.Module]] = {
    "v3_1": _build_v3_1,
    "f0_mlp": _build_f0_mlp,
    "egostitch": _build_egostitch,
}


def build_model(model_family: str, model_config: dict[str, object]) -> nn.Module:
    """Build an untrained scorer of the given family from a checkpointed config.

    Args:
        model_family: One of the keys of :data:`MODEL_BUILDERS` (``v3_1``/``f0_mlp``).
        model_config: The checkpoint's ``model_config`` dict.

    Returns:
        The constructed (randomly initialized) model.

    Raises:
        ValueError: If `model_family` is unknown.
    """
    try:
        builder = MODEL_BUILDERS[model_family]
    except KeyError:
        raise ValueError(
            f"unknown model_family {model_family!r}; expected one of {sorted(MODEL_BUILDERS)}"
        ) from None
    return builder(model_config)


def _checkpoint_id(model_state: Mapping[str, torch.Tensor]) -> str:
    """Derive a deterministic 16-hex id from a model state dict.

    Mirrors ``src.train_b0._state_digest`` exactly (sorted keys; per tensor the
    key, dtype string, and raw bytes), truncated to 16 hex characters, so the id
    written into the scores-artifact meta equals the ``checkpoint_id`` that
    training records in ``run_metadata.json`` for the same checkpoint.

    Args:
        model_state: The checkpoint's ``model_state`` mapping.

    Returns:
        The first 16 hex characters of the SHA-256 digest.
    """
    hasher = hashlib.sha256()
    for key in sorted(model_state):
        tensor = model_state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy(force=True).tobytes())
    return hasher.hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of an input artifact used for score provenance."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    model_family: str | None = None,
    model_config: dict[str, object] | None = None,
) -> tuple[nn.Module, str, str]:
    """Rebuild the frozen model from a Task-4 or bare legacy checkpoint.

    Args:
        path: Path to a ``best.pt``/``last.pt`` checkpoint.
        model_family: Explicit family for a bare legacy state dict.
        model_config: Explicit constructor config for a bare legacy state dict.

    Returns:
        ``(model, model_family, checkpoint_id)``; the model has its weights
        loaded and is left on CPU in ``eval()`` mode.

    Raises:
        ValueError: If the checkpoint is missing pinned keys or has unexpected
            key types.
    """
    checkpoint = cast(Mapping[str, object], torch.load(path, map_location="cpu", weights_only=True))
    if all(key in checkpoint for key in ("model_state", "model_family", "model_config")):
        embedded_family = checkpoint["model_family"]
        if not isinstance(embedded_family, str):
            raise ValueError(f"checkpoint {path}: model_family must be a string")
        embedded_config = checkpoint["model_config"]
        if not isinstance(embedded_config, dict):
            raise ValueError(f"checkpoint {path}: model_config must be a dict")
        model_family = embedded_family
        model_config = cast(dict[str, object], embedded_config)
        model_state = cast(dict[str, torch.Tensor], checkpoint["model_state"])
    else:
        if model_family is None or model_config is None:
            raise ValueError(
                f"bare legacy checkpoint {path} requires explicit model_family and model_config"
            )
        model_state = cast(dict[str, torch.Tensor], checkpoint)

    model = build_model(model_family, model_config)
    model.load_state_dict(model_state)
    model.eval()
    return model, model_family, _checkpoint_id(model_state)


def _load_model_config(path: Path, model_family: str) -> dict[str, object]:
    """Read ``model.config`` from a project training-config YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"model config {path} must contain a model mapping")
    model = cast(dict[str, object], raw["model"])
    configured_family = model.get("family")
    if configured_family != model_family:
        raise ValueError(
            f"model config {path} family {configured_family!r} does not match {model_family!r}"
        )
    config = model.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"model config {path}: model.config must be a mapping")
    return cast(dict[str, object], config)


# ---------------------------------------------------------------------------
# Pair sources
# ---------------------------------------------------------------------------


def _read_pairs_tsv(path: Path) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    r"""Read a ``u\tv[\tlabel]`` TSV into canonical pairs plus an int8 label array.

    Rows without a label column get label ``-1``. Pairs are canonicalized to
    ``(min(u, v), max(u, v))``; row order is preserved.

    Args:
        path: TSV file path.

    Returns:
        ``(pairs, labels)``, aligned index-for-index.

    Raises:
        ValueError: If a row does not have exactly 2 or 3 tab-separated fields.
    """
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) == 2:
                u, v = parts
                label = -1
            elif len(parts) == 3:
                u, v, label_str = parts
                label = int(label_str)
            else:
                raise ValueError(
                    f"{path}:{line_number}: expected 2 or 3 tab-separated fields, got {len(parts)}"
                )
            pairs.append(canonical_pair(u, v))
            labels.append(label)
    return pairs, np.array(labels, dtype=np.int8)


def _resolve_pairs(
    pairs_source: str, data_root: Path, strategy: str
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Resolve a ``--pairs`` spec to canonical pairs plus labels, in file row order.

    Args:
        pairs_source: ``candidate``, ``test``, ``val``, or ``file:<path.tsv>``.
        data_root: Directory containing ``benchmark_2025_neurips/`` and
            ``features/frozen_node_features_1024/``.
        strategy: Split strategy name (e.g. ``breadth_first``).

    Returns:
        ``(pairs, labels)``, aligned index-for-index.

    Raises:
        ValueError: If `pairs_source` is not one of the supported forms.
    """
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    if pairs_source == "candidate":
        labeled = load_candidate_pairs(benchmark_root, strategy)
        return labeled.pairs, labeled.labels
    if pairs_source in ("test", "val"):
        return _read_pairs_tsv(benchmark_root / strategy / f"{pairs_source}_edges.txt")
    if pairs_source.startswith("file:"):
        return _read_pairs_tsv(Path(pairs_source[len("file:") :]))
    raise ValueError(
        f"unsupported --pairs value {pairs_source!r}; expected "
        f"{'|'.join(_NAMED_PAIR_SOURCES)} or file:<path.tsv>"
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    """Resolve a ``--device`` spec (``auto`` picks cuda, then mps, then cpu)."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _autocast_context(device: torch.device, amp: str) -> torch.autocast | nullcontext[None]:
    """Return the autocast context for the ``--amp`` mode (``off`` is a no-op)."""
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _log_progress(processed: int, total: int, batch_rows: int) -> None:
    """Emit a progress log line whenever a 50k-row boundary is crossed."""
    if processed // _LOG_EVERY_ROWS > (processed - batch_rows) // _LOG_EVERY_ROWS:
        logger.info("scored %d/%d rows", processed, total)


def _score_v3_1(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
) -> NDArray[np.float32]:
    """Score pairs with a `V3_1` model via the length-bucketed batching machinery.

    Examples are processed in length-bucketed order for padding efficiency
    (``shuffle=False``, so the order is deterministic), and each batch's logits
    are scattered back to the original row positions, restoring input order.

    Args:
        model: Frozen `V3_1` model, already on `device` and in ``eval()`` mode.
        pairs: Node-id pairs in input row order.
        store: Feature store providing per-node token sequences.
        device: Compute device.
        amp: ``off`` or ``bf16``.
        token_budget: Approximate per-batch token budget for the bucketed sampler.

    Returns:
        Shape ``(len(pairs),)`` float32 logits in input row order.
    """
    lengths = probe_lengths(store, pairs)
    dataset = TokenPairDataset(pairs, None, store, lengths=lengths)
    sampler = LengthBucketedBatchSampler(lengths, token_budget=token_budget, shuffle=False)

    out: NDArray[np.float32] = np.empty(len(pairs), dtype=np.float32)
    processed = 0
    for batch_indices in sampler:
        batch = collate_token_pairs([dataset[i] for i in batch_indices])
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.inference_mode(), _autocast_context(device, amp):
            logits = cast(torch.Tensor, model(batch)["logits"])
        out[np.asarray(batch_indices, dtype=np.int64)] = (
            logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
        )
        processed += len(batch_indices)
        _log_progress(processed, len(pairs), len(batch_indices))
    return out


def _score_v3_1_packed(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    pack_dir: Path,
    *,
    device: torch.device,
    amp: str,
    token_budget: int,
) -> NDArray[np.float32]:
    """Score V3.1 pairs with packed features and cached per-node encodings."""
    if not isinstance(model, V3_1):
        raise TypeError(f"packed V3.1 scoring requires V3_1, got {type(model).__name__}")
    load_started = perf_counter()
    table = PackedFeatureTable.from_pack(pack_dir, device)
    logger.info(
        "loaded packed feature table from %s in %.3f seconds",
        pack_dir,
        perf_counter() - load_started,
    )
    node_index = table.manifest.node_index()
    missing = sorted({node for pair in pairs for node in pair if node not in node_index})
    if missing:
        raise ValueError(f"packed feature table is missing {len(missing)} nodes: {missing[:5]}")

    lengths_by_node = {record.node_id: record.length for record in table.manifest.nodes}
    lengths = [(lengths_by_node[u], lengths_by_node[v]) for u, v in pairs]
    sampler = LengthBucketedBatchSampler(lengths, token_budget=token_budget, shuffle=False)
    node_a = torch.tensor([node_index[u] for u, _ in pairs], dtype=torch.int64)
    node_b = torch.tensor([node_index[v] for _, v in pairs], dtype=torch.int64)

    used_node_indices = sorted({node_index[node] for pair in pairs for node in pair})
    max_boundary = max(
        (
            next(
                value for value in BUCKET_BOUNDARIES if value >= table.manifest.nodes[index].length
            )
            for index in used_node_indices
        ),
        default=0,
    )
    # V3.1's final encoder normalization returns FP32 even under BF16 autocast.
    # Preserve that dtype: narrowing this cache changes frozen-B0 logits.
    cache_dtype = next(model.parameters()).dtype
    encoded = torch.zeros(
        (len(table.manifest.nodes), max_boundary, model.d_model),
        dtype=cache_dtype,
        device=device,
    )
    encode_started = perf_counter()
    encoded_nodes = 0
    previous_boundary = 0
    for boundary in BUCKET_BOUNDARIES:
        bucket_nodes = [
            index
            for index in used_node_indices
            if previous_boundary < table.manifest.nodes[index].length <= boundary
        ]
        previous_boundary = boundary
        batch_nodes = max(token_budget // boundary, 1)
        for start in range(0, len(bucket_nodes), batch_nodes):
            indices = torch.tensor(bucket_nodes[start : start + batch_nodes], dtype=torch.int64)
            raw_tokens, node_lengths = table.gather_nodes(indices, boundary)
            if amp == "off":
                raw_tokens = raw_tokens.to(cache_dtype)
            with torch.inference_mode(), _autocast_context(device, amp):
                node_encoded = model.encoder(raw_tokens, node_lengths)
            device_indices = indices.to(device)
            encoded[device_indices, :boundary] = node_encoded.to(cache_dtype)
            encoded_nodes += len(indices)
    encode_seconds = perf_counter() - encode_started
    logger.info(
        "cached %d unique node encodings in %.3f seconds (%.1f nodes/s)",
        encoded_nodes,
        encode_seconds,
        encoded_nodes / encode_seconds if encode_seconds else float("inf"),
    )
    packed_lengths = table.lengths
    del table
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out: NDArray[np.float32] = np.empty(len(pairs), dtype=np.float32)
    processed = 0
    batch_count = 0
    score_started = perf_counter()
    for batch_indices in sampler:
        row_ids = torch.tensor(batch_indices, dtype=torch.int64)
        max_length = max(max(lengths[index]) for index in batch_indices)
        boundary = next(value for value in BUCKET_BOUNDARIES if value >= max_length)
        pair_a = node_a.index_select(0, row_ids).to(device)
        pair_b = node_b.index_select(0, row_ids).to(device)
        len_a = packed_lengths.index_select(0, pair_a)
        len_b = packed_lengths.index_select(0, pair_b)
        with torch.inference_mode(), _autocast_context(device, amp):
            pair_repr = model._pair_representation(
                encoded.index_select(0, pair_a)[:, :boundary],
                encoded.index_select(0, pair_b)[:, :boundary],
                len_a,
                len_b,
            )
            logits = model.output_head(pair_repr)
        out[np.asarray(batch_indices, dtype=np.int64)] = (
            logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
        )
        processed += len(batch_indices)
        batch_count += 1
        _log_progress(processed, len(pairs), len(batch_indices))
    score_seconds = perf_counter() - score_started
    logger.info(
        "packed scoring completed %d rows in %d batches over %.3f seconds (%.1f rows/s)",
        len(pairs),
        batch_count,
        score_seconds,
        len(pairs) / score_seconds if score_seconds else float("inf"),
    )
    return out


def _score_f0_mlp(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
    amp: str,
    batch_pairs: int,
    f0_cache: Path,
) -> NDArray[np.float32]:
    """Score pairs with an `F0PairMLP` over mean-pooled F0 feature rows.

    Args:
        model: Frozen `F0PairMLP`-contract model, on `device`, in ``eval()`` mode.
        pairs: Node-id pairs in input row order.
        store: Feature store providing per-node token sequences.
        device: Compute device.
        amp: ``off`` or ``bf16``.
        batch_pairs: Number of pair rows per forward pass.
        f0_cache: F0 matrix cache path (see :func:`src.data.features.build_f0_matrix`).
            If an existing cache covers a different node set, it is left untouched
            and the matrix is recomputed in memory.

    Returns:
        Shape ``(len(pairs),)`` float32 logits in input row order.
    """
    node_ids = sorted({node_id for pair in pairs for node_id in pair})
    try:
        f0_cache.parent.mkdir(parents=True, exist_ok=True)
        matrix, index = build_f0_matrix(
            store, node_ids, cache_path=f0_cache, allow_cache_subset=True
        )
    except ValueError:
        logger.warning(
            "F0 cache at %s does not match the requested node set; recomputing without cache",
            f0_cache,
        )
        matrix, index = build_f0_matrix(store, node_ids, cache_path=None)

    u_rows = torch.tensor([index[u] for u, _ in pairs], dtype=torch.int64)
    v_rows = torch.tensor([index[v] for _, v in pairs], dtype=torch.int64)

    out: NDArray[np.float32] = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), batch_pairs):
        end = min(start + batch_pairs, len(pairs))
        batch = {
            "x_a": matrix[u_rows[start:end]].to(device),
            "x_b": matrix[v_rows[start:end]].to(device),
        }
        with torch.inference_mode(), _autocast_context(device, amp):
            logits = cast(torch.Tensor, model(batch)["logits"])
        out[start:end] = logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
        _log_progress(end, len(pairs), end - start)
    return out


def _align_s0_logits(
    artifact: ScoresArtifact, pairs: Sequence[tuple[str, str]]
) -> NDArray[np.float32]:
    """Row-align frozen-B0 logits to `pairs` by canonical pair (hard-fails on a miss).

    Alignment is by pair identity, not row order, so a candidate/val/test
    artifact covers any subset or reordering of its pairs. A missing pair is a
    contract violation (spec Sec 13.10) and never silently imputed.
    """
    lookup: dict[tuple[str, str], float] = {}
    logits = artifact.logit.astype(np.float64)
    for row, (u, v) in enumerate(artifact.pairs()):
        lookup[(u, v) if u <= v else (v, u)] = float(logits[row])
    out = np.empty(len(pairs), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        key = (u, v) if u <= v else (v, u)
        if key not in lookup:
            raise ValueError(
                f"--b0-scores is missing pair {key}; it must cover every scored row "
                "(spec Sec 13.10)"
            )
        out[i] = lookup[key]
    return out


def _score_egostitch(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
    amp: str,
    batch_pairs: int,
    f0_cache: Path,
    grounding_cache: Path | None,
    s0_logits: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Score pairs with an `EgoStitchStage1` model (spec Sec 10.3 / Sec 13).

    Runs the cacheable per-node pass (Tokenize-lite + Imagine) once over the
    unique endpoints, then scores pair batches from the cache: Sinkhorn stitch
    + decision head fused with the cached frozen-B0 ``s0`` logits. Self-pairs
    route through the single-ego path (spec Sec 13.9). Stage-1 scores are
    pass-1 (ratio 1) scores (spec Sec 13.11). The per-pair pass (Sinkhorn
    stitch, decision head, self-pair single-ego branch) always runs in fp32,
    regardless of `amp` (spec Sec 13.16 score-precision pin); only the
    per-node encode pass honors `amp`.

    Args:
        model: Frozen `EgoStitchStage1`, on `device`, in ``eval()`` mode.
        pairs: Node-id pairs in input row order.
        store: Feature store providing per-node token sequences.
        device: Compute device.
        amp: ``off`` or ``bf16``; applies only to the per-node encode pass.
            The per-pair scoring pass always runs in fp32 (see above).
        batch_pairs: Pair rows per forward pass.
        f0_cache: F0 matrix cache path (f0_mlp semantics).
        grounding_cache: Grounding-pool cache path (derived from `f0_cache`
            when ``None``).
        s0_logits: Row-aligned frozen-B0 logits.

    Returns:
        Shape ``(len(pairs),)`` float32 logits in input row order.
    """
    from src.data.grounding import build_grounding_pool
    from src.model.egostitch.model import EgoStitchStage1, NodeEncoding
    from src.model.egostitch.stitch import sinkhorn_plan

    assert isinstance(model, EgoStitchStage1)
    model.set_density_ratio(1.0)  # pass-1 scores (spec Sec 13.11)
    node_ids = sorted({node_id for pair in pairs for node_id in pair})
    try:
        f0_cache.parent.mkdir(parents=True, exist_ok=True)
        matrix, index = build_f0_matrix(
            store, node_ids, cache_path=f0_cache, allow_cache_subset=True
        )
    except ValueError:
        logger.warning(
            "F0 cache at %s does not match the requested node set; recomputing without cache",
            f0_cache,
        )
        matrix, index = build_f0_matrix(store, node_ids, cache_path=None)
    if grounding_cache is None:
        grounding_cache = f0_cache.with_name(f"{f0_cache.stem}_grounding.npz")
    pool = build_grounding_pool(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_ids,
        n_ground=model.config.n_ground,
        cache_path=grounding_cache,
    )
    pool_rows = torch.tensor(
        [[index[neighbor] for neighbor in pool[node]] for node in node_ids],
        dtype=torch.int64,
    )

    # Cacheable per-node pass (spec Sec 10.3): one batched encode per node.
    encode_batch = max(batch_pairs // max(model.config.slots, 1), 1)
    h_cache: list[torch.Tensor] = []
    pi_cache: list[torch.Tensor] = []
    mult_cache: list[torch.Tensor] = []
    adj_cache: list[torch.Tensor] = []
    d_hat_cache: list[torch.Tensor] = []
    proj_cache: list[torch.Tensor] = []
    for start in range(0, len(node_ids), encode_batch):
        rows = torch.arange(start, min(start + encode_batch, len(node_ids)))
        x = matrix[rows].to(device)
        ground_x = matrix[pool_rows[rows]].to(device)
        with torch.inference_mode(), _autocast_context(device, amp):
            enc: NodeEncoding = model.encode_nodes(x, ground_x)
            proj = model.proj(x)
        h_cache.append(enc.slots.h.float())
        pi_cache.append(enc.slots.pi.float())
        mult_cache.append(enc.slots.mult.float())
        adj_cache.append(enc.slots.adj.float())
        d_hat_cache.append(model.d_hat(enc.tok).float())
        proj_cache.append(proj.float())
    h_all = torch.cat(h_cache)
    pi_all = torch.cat(pi_cache)
    mult_all = torch.cat(mult_cache)
    adj_all = torch.cat(adj_cache)
    d_hat_all = torch.cat(d_hat_cache)
    proj_all = torch.cat(proj_cache)

    from src.model.egostitch.imagine import SlotSet

    u_rows = torch.tensor([index[u] for u, _ in pairs], dtype=torch.int64)
    v_rows = torch.tensor([index[v] for _, v in pairs], dtype=torch.int64)
    is_self = u_rows == v_rows
    out: NDArray[np.float32] = np.empty(len(pairs), dtype=np.float32)
    filler = torch.zeros(1, dtype=torch.float32, device=device)

    def _slots(rows: torch.Tensor) -> SlotSet:
        return SlotSet(
            h=h_all[rows],
            pi=pi_all[rows],
            mult=mult_all[rows],
            gate=filler.expand(rows.shape[0], pi_all.shape[1]),
            pointer=filler.expand(rows.shape[0], pi_all.shape[1], 1),
            adj=adj_all[rows],
        )

    for start in range(0, len(pairs), batch_pairs):
        end = min(start + batch_pairs, len(pairs))
        batch_u = u_rows[start:end].to(device)
        batch_v = v_rows[start:end].to(device)
        batch_self = is_self[start:end].to(device)
        s0 = torch.from_numpy(s0_logits[start:end].astype(np.float32)).to(device)
        # Pair pass always runs fp32 regardless of `amp` (spec Sec 13.16
        # score-precision pin): under bf16 autocast the Sinkhorn stitch and
        # decision head (both branches below) emit bfloat16 values, and
        # assigning them into `logits` silently preserves the bf16 ulp grid
        # (spacing 0.03125 at |logit| in [4,8)) even though `logits` itself is
        # fp32. Inputs here are already fp32 (`h_all`/`pi_all`/`mult_all`/
        # `adj_all`/`proj_all`/`d_hat_all` are `.float()`-cast above; `s0` is
        # fp32), so disabling autocast is behavior-correcting, not a refactor.
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            logits = torch.zeros(end - start, dtype=torch.float32, device=device)
            sel = torch.nonzero(~batch_self, as_tuple=False).squeeze(-1)
            if sel.numel() > 0:
                slots_i = _slots(batch_u[sel])
                slots_j = _slots(batch_v[sel])
                plan = sinkhorn_plan(
                    slots_i.h,
                    slots_j.h,
                    slots_i.pi,
                    slots_j.pi,
                    slots_i.mult,
                    slots_j.mult,
                    eps=model.config.sinkhorn_eps,
                    iters=model.config.sinkhorn_iters,
                    tau=model.config.sinkhorn_tau,
                )
                logits[sel] = model.decision(
                    s0[sel],
                    slots_i,
                    slots_j,
                    plan,
                    proj_all[batch_u[sel]],
                    proj_all[batch_v[sel]],
                    d_hat_all[batch_u[sel]],
                    d_hat_all[batch_v[sel]],
                ).float()
            sel = torch.nonzero(batch_self, as_tuple=False).squeeze(-1)
            if sel.numel() > 0:
                logits[sel] = model.decision.forward_self(
                    s0[sel],
                    _slots(batch_u[sel]),
                    proj_all[batch_u[sel]],
                    d_hat_all[batch_u[sel]],
                ).float()
        out[start:end] = logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
        _log_progress(end, len(pairs), end - start)
    return out


def _shard_range(num_rows: int, shard: int, num_shards: int) -> tuple[int, int]:
    """Return shard `shard`-of-`num_shards`'s contiguous ``[start, end)`` row range.

    Uses ``chunk = ceil(num_rows / num_shards)`` slicing, so all shards but the
    last have exactly ``chunk`` rows and trailing shards may be empty.

    Args:
        num_rows: Total input rows.
        shard: Shard index in ``[0, num_shards)``.
        num_shards: Total shard count.

    Returns:
        The ``(start, end)`` row range (clamped to ``num_rows``).
    """
    chunk = math.ceil(num_rows / num_shards) if num_rows else 0
    start = min(shard * chunk, num_rows)
    end = min(start + chunk, num_rows)
    return start, end


def _shard_output_path(output: Path, shard: int) -> Path:
    """Return the per-shard output path ``<output stem>.shard-K<suffix>``."""
    return output.with_name(f"{output.stem}.shard-{shard}{output.suffix}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``score``/``merge`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.score_universe",
        description="Score node-pair lists with a frozen checkpoint; merge shard outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score a pair source with a frozen checkpoint")
    score.add_argument("--checkpoint", type=Path, required=True, help="checkpoint .pt/.pth file")
    score.add_argument(
        "--model-family",
        choices=sorted(MODEL_BUILDERS),
        help="model family for a bare legacy state_dict checkpoint",
    )
    score.add_argument(
        "--model-config",
        type=Path,
        help="training YAML containing model.config for a bare legacy checkpoint",
    )
    score.add_argument(
        "--pairs",
        required=True,
        help="candidate | test | val | file:<path.tsv> (TSV rows: u\\tv[\\tlabel])",
    )
    score.add_argument("--data-root", type=Path, default=Path("data"))
    score.add_argument("--strategy", default="breadth_first")
    score.add_argument("--output", type=Path, required=True, help="output .npz path")
    score.add_argument("--batch-pairs", type=int, default=8192, help="f0_mlp rows per batch")
    score.add_argument("--token-budget", type=int, default=131_072, help="v3_1 tokens per batch")
    score.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help="GPU-resident packed BF16 feature directory for v3_1 scoring",
    )
    score.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    score.add_argument("--amp", choices=["off", "bf16"], default="off")
    score.add_argument(
        "--b0-scores",
        type=Path,
        default=None,
        help="frozen-B0 scores .npz supplying s0 (required for egostitch; spec Sec 13.10)",
    )
    score.add_argument(
        "--s0-checkpoint-id",
        default="e092537d8cf1e208",
        help="required checkpoint_id of --b0-scores (the audited frozen B0)",
    )
    score.add_argument(
        "--grounding-cache",
        type=Path,
        default=None,
        help="grounding-pool cache path (egostitch only; built when absent)",
    )
    score.add_argument("--shard", type=int, default=None, help="shard index K (with --num-shards)")
    score.add_argument("--num-shards", type=int, default=None, help="total shard count N")
    score.add_argument(
        "--f0-cache",
        type=Path,
        default=Path("outputs/f0_cache/f0_matrix.pt"),
        help="F0 matrix cache path (f0_mlp only)",
    )

    merge = subparsers.add_parser("merge", help="merge shard outputs into one artifact")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def _validate_score_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate ``score`` arguments, exiting via ``parser.error`` on bad input."""
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if (args.model_family is None) != (args.model_config is None):
        parser.error("--model-family and --model-config must be given together")
    if args.model_config is not None and not args.model_config.exists():
        parser.error(f"model config not found: {args.model_config}")
    pairs_source: str = args.pairs
    if pairs_source not in _NAMED_PAIR_SOURCES and not pairs_source.startswith("file:"):
        parser.error(
            f"--pairs must be one of {'|'.join(_NAMED_PAIR_SOURCES)} or file:<path.tsv>; "
            f"got {pairs_source!r}"
        )
    if pairs_source.startswith("file:") and not Path(pairs_source[len("file:") :]).exists():
        parser.error(f"--pairs file not found: {pairs_source[len('file:') :]}")
    if (args.shard is None) != (args.num_shards is None):
        parser.error("--shard and --num-shards must be given together")
    if args.num_shards is not None:
        if args.num_shards < 1:
            parser.error(f"--num-shards must be >= 1, got {args.num_shards}")
        if not 0 <= args.shard < args.num_shards:
            parser.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")


def _run_score(args: argparse.Namespace) -> None:
    """Execute the ``score`` subcommand."""
    device = _resolve_device(args.device)
    logger.info("loading checkpoint %s (device %s)", args.checkpoint, device)
    model_config = (
        _load_model_config(args.model_config, args.model_family)
        if args.model_config is not None
        else None
    )
    model, model_family, checkpoint_id = _load_checkpoint(
        args.checkpoint,
        model_family=args.model_family,
        model_config=model_config,
    )
    model.to(device)

    pairs, labels = _resolve_pairs(args.pairs, args.data_root, args.strategy)
    total_rows = len(pairs)
    logger.info("resolved %d pairs from %s", total_rows, args.pairs)

    if args.shard is not None:
        start, end = _shard_range(total_rows, args.shard, args.num_shards)
        output = _shard_output_path(args.output, args.shard)
        logger.info(
            "shard %d/%d: rows [%d, %d) -> %s", args.shard, args.num_shards, start, end, output
        )
    else:
        start, end = 0, total_rows
        output = args.output
    row_pairs = pairs[start:end]
    row_labels = labels[start:end]

    store = FeatureStore(args.data_root / _FEATURES_SUBDIR)
    meta_extra: dict[str, object] = {
        "score_precision": {
            "encode_autocast": args.amp,
            "logit_storage_dtype": "float32",
        }
    }
    score_started = perf_counter()
    if model_family == "v3_1":
        if args.pack_dir is None:
            logits = _score_v3_1(
                model,
                row_pairs,
                store,
                device=device,
                amp=args.amp,
                token_budget=args.token_budget,
            )
        else:
            logits = _score_v3_1_packed(
                model,
                row_pairs,
                args.pack_dir,
                device=device,
                amp=args.amp,
                token_budget=args.token_budget,
            )
    elif model_family == "f0_mlp":
        logits = _score_f0_mlp(
            model,
            row_pairs,
            store,
            device=device,
            amp=args.amp,
            batch_pairs=args.batch_pairs,
            f0_cache=args.f0_cache,
        )
    elif model_family == "egostitch":
        if args.b0_scores is None:
            raise ValueError(
                "egostitch scoring requires --b0-scores (the frozen-B0 s0 cache, spec Sec 13.10)"
            )
        s0_artifact = load_scores(args.b0_scores)
        actual_s0_id = s0_artifact.meta.get("checkpoint_id")
        if actual_s0_id != args.s0_checkpoint_id:
            raise ValueError(
                f"--b0-scores checkpoint_id mismatch: expected {args.s0_checkpoint_id!r}, "
                f"got {actual_s0_id!r} (spec Sec 13.10: s0 is the audited frozen B0)"
            )
        s0_rows = _align_s0_logits(s0_artifact, row_pairs)
        logits = _score_egostitch(
            model,
            row_pairs,
            store,
            device=device,
            amp=args.amp,
            batch_pairs=args.batch_pairs,
            f0_cache=args.f0_cache,
            grounding_cache=args.grounding_cache,
            s0_logits=s0_rows,
        )
        logits64 = logits.astype(np.float64)
        probs = np.where(
            logits64 >= 0,
            1.0 / (1.0 + np.exp(-np.abs(logits64))),
            np.exp(-np.abs(logits64)) / (1.0 + np.exp(-np.abs(logits64))),
        )
        meta_extra = {
            "s0_checkpoint_id": args.s0_checkpoint_id,
            "s0_artifact_sha256": _file_sha256(args.b0_scores),
            "score_precision": {
                "contract": _EGOSTITCH_PAIR_PRECISION_CONTRACT,
                "encode_autocast": args.amp,
                "pair_compute_dtype": "float32",
                "pair_autocast": False,
                "logit_storage_dtype": "float32",
            },
            "density_calibration": {
                "pass": 1,
                "rho_hat_eval": float(np.mean(probs)) if len(probs) else 0.0,
                "note": "Stage-1 scores are pass-1 scores (spec Sec 13.11)",
            },
        }
    else:  # pragma: no cover - build_model already rejects unknown families
        raise ValueError(f"no scoring path for model_family {model_family!r}")

    score_wall_seconds = perf_counter() - score_started
    node_ids = sorted({node_id for pair in row_pairs for node_id in pair})
    meta_extra["score_profile"] = {
        "wall_seconds": score_wall_seconds,
        "rows": len(row_pairs),
        "unique_nodes": len(node_ids),
        "measurement": "single_process_compute_wall_seconds",
    }
    node_position = {node_id: i for i, node_id in enumerate(node_ids)}
    meta: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "model_family": model_family,
        "pairs_source": args.pairs,
        "strategy": args.strategy,
        "num_rows": total_rows,
        "created_utc": datetime.now(UTC).isoformat(),
        "torch_version": str(torch.__version__),
        **meta_extra,
    }
    save_scores(
        output,
        node_ids=node_ids,
        u_idx=np.array([node_position[u] for u, _ in row_pairs], dtype=np.int32),
        v_idx=np.array([node_position[v] for _, v in row_pairs], dtype=np.int32),
        logit=logits,
        label=row_labels,
        row_start=start,
        meta=meta,
    )
    logger.info("wrote %d scored rows to %s", len(row_pairs), output)


def _run_merge(args: argparse.Namespace) -> None:
    """Execute the ``merge`` subcommand."""
    merged = merge_scores(args.inputs)
    save_scores(
        args.output,
        node_ids=merged.node_ids,
        u_idx=merged.u_idx,
        v_idx=merged.v_idx,
        logit=merged.logit,
        label=merged.label,
        row_start=0,
        meta=merged.meta,
    )
    logger.info(
        "merged %d shard files (%d rows) into %s", len(args.inputs), len(merged.logit), args.output
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the ``score`` and ``merge`` subcommands.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors (via ``argparse``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        _validate_score_args(parser, args)
        _run_score(args)
    else:
        _run_merge(args)


__all__ = [
    "MODEL_BUILDERS",
    "ScoresArtifact",
    "build_model",
    "build_parser",
    "load_scores",
    "main",
    "merge_scores",
    "save_scores",
    "score_resolution_diagnostics",
    "validate_score_precision",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
