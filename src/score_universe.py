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
        --pairs candidate|test|test_topology|val_topology|val_cls|file:<path.tsv> \
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
  descriptive score-resolution diagnostics. A checkpoint whose ``egostitch_e2e``
  generator is ``oracle_struct`` additionally carries ``oracle_diagnostic``
  (``{"generator": "oracle_struct", "truth_source": ..., "diagnostic_only":
  true}``) and is forced ``heldout: false`` regardless of ``pairs_source`` --
  it consumes ground-truth topology by construction and is a ceiling
  diagnostic, never a formal held-out result (``--allow-oracle-diagnostic``).
- ``full``/``f_logit``: float32 arrays present only for family
  ``egostitch_e2e`` (design rev 4 two-logit decomposition, content path
  removed; ``meta.scores_meta_version == "egostitch_e2e_scores_v4"``).
  ``logit`` holds the active primary arm; legacy v1 artifacts may omit
  ``full`` when ``logit`` is already the true full arm, or use
  ``full_logit`` for a permanent-null primary. Legacy v3 artifacts
  (``scores_meta_version == "egostitch_e2e_scores_v3"``) additionally carry
  ``pair_content``/``pair_topology``; those two arrays are a read-only
  backward-compatibility surface -- new runs never write them.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import pickle
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import networkx as nx
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch import nn

from src.data.artifacts import canonical_pair, load_candidate_pairs, load_test_topology_pairs
from src.data.feature_stats import FeatureStats, load_feature_stats
from src.data.features import FeatureStore, build_f0_matrix
from src.data.grounding import POOL_METHOD_ID
from src.data.packed_features import PackedFeatureTable
from src.data.pairs import (
    BUCKET_BOUNDARIES,
    LengthBucketedBatchSampler,
    TokenPairDataset,
    collate_token_pairs,
    probe_lengths,
)
from src.data.val_region import (
    ValRegionParams,
    ValRegionSplit,
    derive_val_region_split,
)
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.topo_gen import CONTROLS, TopoGenBase
from src.model.egostitch.generator.assemble import make_scaffold_input_perturbation

if TYPE_CHECKING:
    # `src.train_cazi_mbn` transitively imports `torch_geometric` (via
    # `src.baselines.cazi_mbn`); every *runtime* use stays a function-local
    # import (matching this file's existing lazy-import style for e2e-specific
    # modules) so merely importing `score_universe` never pays that cost. This
    # import is erased at runtime (`from __future__ import annotations` makes
    # every annotation a string), so it exists for mypy only.
    from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
    from src.model.egostitch.generator.full_oracle import FullEgoGraph
    from src.train_cazi_mbn import CAZIConfig

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
_NAMED_PAIR_SOURCES = ("candidate", "test", "test_topology", "val_topology", "val_cls")
#: Non-held-out V_val role-universe sources: the sampled topology union and the
#: balanced classification rows.
_VAL_PAIR_SOURCES = frozenset({"val_topology", "val_cls"})
#: `derive_val_region_split`'s parameters for `_load_val_region_split`'s
#: production re-derivation; the test seam a small monkeypatched value lets
#: synthetic fixtures satisfy (`ValRegionParams`'s own defaults assume a
#: real-package-scale train universe).
_VAL_REGION_PARAMS = ValRegionParams()
_EGOSTITCH_E2E_PAIR_PRECISION_CONTRACT = "egostitch_e2e_pair_fp32_v1"
_EGOSTITCH_E2E_ARRAY_KEYS = ("full", "f_logit")
_SCORES_META_VERSION = "egostitch_e2e_scores_v4"
#: The one prior scores-meta version whose artifacts remain readable. v3
#: artifacts additionally carry ``pair_content``/``pair_topology`` (the
#: now-deleted content-path decomposition arms); v4 never writes them. Both
#: versions are accepted on read so existing v3 artifacts keep loading; only
#: v4 is ever written by :func:`save_scores`.
_LEGACY_E2E_SCORES_META_VERSION = "egostitch_e2e_scores_v3"
_SUPPORTED_E2E_SCORES_META_VERSIONS = (_SCORES_META_VERSION, _LEGACY_E2E_SCORES_META_VERSION)
_SCAFFOLD_CONTROL_NONE = "none"
_SCAFFOLD_CONTROL_SHUFFLE_V3 = "shuffle_within_pair_v3"
_SCAFFOLD_CONTROL_REWIRE_V1 = "rewire_checkerboard_v1"
_SCAFFOLD_CONTROL_SHUFFLE_V2 = "shuffle_within_pair"
_SCAFFOLD_CONTROL_SEED = 0
#: Distinct test-access-ledger arm identity for each active scaffold control,
#: matching the arm names `src/experiments/g5_stage1.py` already reports
#: them under (`_CONTROL_ARMS`: `structure_control_6a_v3` is the
#: shuffle-within-pair control, `structure_control_6e_v1` is the
#: degree-preserving (checkerboard) rewiring control). Both controls reuse
#: the `full` checkpoint and its run metadata, so without this remap they
#: would ledger under arm `full` and collide with the ordinary full-arm
#: score -- rejecting normal first-time scoring of either control unless the
#: operator supplies a misleading `--rescore-reason`.
_SCAFFOLD_CONTROL_ARM_NAMES: dict[str, str] = {
    _SCAFFOLD_CONTROL_SHUFFLE_V3: "structure_control_6a_v3",
    _SCAFFOLD_CONTROL_REWIRE_V1: "structure_control_6e_v1",
}
_TEST_ACCESS_LEDGER_FILENAME = "test_access_ledger.jsonl"
_TEST_ACCESS_LEDGER_SCHEMA = "egostitch_test_access_v1"
_FULL_ORACLE_CELL_BUDGET = 12_000_000
_FULL_ORACLE_MAX_BATCH_PAIRS = 1024
_FULL_ORACLE_PREFETCH_DEPTH = 4
_FULL_ORACLE_SHARD_COST_FLOOR = 4096
_FULL_ORACLE_CPU_THREADS = 2


def _e2e_primary_logit_key(permanent_null: str) -> str:
    """Return the published primary-logit array for an e2e permanent-null arm.

    Raises:
        ValueError: On ``"content_head"``. Legacy v3 artifacts from the retired
            ``pair_topology`` arm carry that null and a ``pair_topology``
            primary logit; the arm went away with the content path (design
            2026-08-02 §9) and is not supported for validation or merge. This
            says so, rather than letting the lookup raise a bare ``KeyError``
            from inside precision validation.
    """
    if permanent_null == "content_head":
        raise ValueError(
            "permanent_null 'content_head' belongs to the retired pair_topology arm, "
            "removed with the content path (design 2026-08-02 §9). Legacy v3 artifacts "
            "from that arm cannot be validated or merged; re-score from a current "
            "checkpoint. Other v3 arms still load."
        )
    try:
        return {
            "none": "full",
            "all_head": "f_logit",
        }[permanent_null]
    except KeyError:
        raise ValueError(f"unknown egostitch_e2e permanent_null: {permanent_null!r}") from None


def _reject_superseded_scaffold_control(scaffold_control: str) -> None:
    """Reject the v2 adj-only control with the governing rev-3.1 contract."""
    if scaffold_control == _SCAFFOLD_CONTROL_SHUFFLE_V2:
        raise ValueError(
            "'shuffle_within_pair' is superseded by the rebuild-form control; "
            "see docs/05-egostitch-spec.md §14.4.5"
        )


def _default_grounding_cache_path(
    f0_cache: Path,
    *,
    n_ground: int,
    node_ids: Iterable[str],
    method_id: str = POOL_METHOD_ID,
    role_universe: str = "test",
) -> Path:
    """Namespace a derived grounding cache by pool configuration and universe."""
    universe_payload = json.dumps(
        {"role_universe": role_universe, "node_ids": sorted(set(node_ids))},
        separators=(",", ":"),
        sort_keys=True,
    )
    universe_id = hashlib.sha256(universe_payload.encode("utf-8")).hexdigest()
    return f0_cache.with_name(
        f"{f0_cache.stem}_grounding_{method_id}_n{n_ground}_u{universe_id}.npz"
    )


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
        logit: Shape ``(n,)`` float32 raw primary model logits (not sigmoid).
            For permanent-null ``egostitch_e2e`` artifacts this is the active
            arm selected for downstream metrics.
        label: Shape ``(n,)`` int8 labels; ``-1`` where the input row was unlabeled.
        meta: Parsed artifact metadata (the pinned seven keys).
        f_logit: Shape ``(n,)`` float32 ``egostitch_e2e`` frozen-topology arm;
            ``None`` for every other family or artifact vintage.
        pair_content: Shape ``(n,)`` float32 ``egostitch_e2e`` content-only arm.
            Populated only when loading a legacy v3 artifact (content path
            removed rev 4); ``None`` for every v4 artifact, and for every
            other family.
        pair_topology: Shape ``(n,)`` float32 ``egostitch_e2e`` topology-only
            arm. Populated only when loading a legacy v3 artifact; ``None``
            for every v4 artifact, and for every other family.
        full_logit: Optional shape ``(n,)`` float32 true full e2e arm. Present
            when ``logit`` stores a permanent-null primary instead.
    """

    node_ids: list[str]
    u_idx: NDArray[np.int32]
    v_idx: NDArray[np.int32]
    logit: NDArray[np.float32]
    label: NDArray[np.int8]
    meta: dict[str, object]
    f_logit: NDArray[np.float32] | None = None
    pair_content: NDArray[np.float32] | None = None
    pair_topology: NDArray[np.float32] | None = None
    full_logit: NDArray[np.float32] | None = None

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


@dataclass
class _TestAccessContext:
    """Identity needed to ledger one held-out scoring epoch."""

    ledger_path: Path
    scoring_arm: str
    seed: int
    output: Path
    shard: int
    num_shards: int
    rescore_reason: str | None
    scoring_run_id: str | None = None
    ledger_binding: dict[str, object] | None = None


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
    f_logit: NDArray[np.float32] | None = None
    pair_content: NDArray[np.float32] | None = None
    pair_topology: NDArray[np.float32] | None = None
    full_logit: NDArray[np.float32] | None = None


@dataclass
class _FullOracleScoreTelemetry:
    """Mutable full-ego scorer measurements returned to `_run_score`."""

    ego_sizes: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    batch_max_ego_sizes: list[int] = field(default_factory=list)
    producer_wait_seconds: float = 0.0


@dataclass(frozen=True)
class _PreparedFullOracleBatch:
    """One CPU-stitched batch ready for asynchronous device transfer."""

    output_rows: list[tuple[int, int]]
    node_rows_a: torch.Tensor
    node_rows_b: torch.Tensor
    encoded_width: int
    state_a: E2ENodeState | None
    state_b: E2ENodeState | None
    is_self: torch.Tensor
    graph: FullEgoGraph


@contextmanager
def _torch_intraop_threads(count: int) -> Iterator[None]:
    """Temporarily cap Torch CPU workers for a scoped mixed CPU/GPU pipeline."""
    if count <= 0:
        raise ValueError("Torch intra-op thread count must be positive")
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _gather_padded_node_rows(
    encoded_table: torch.Tensor,
    length_table: torch.Tensor,
    node_rows: torch.Tensor,
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather padded node states without widening beyond the batch-local maximum."""
    if width <= 0 or width > encoded_table.shape[1]:
        raise ValueError("gather width must be within the padded node-state table")
    rows = node_rows.to(device=encoded_table.device, non_blocking=True)
    lengths = length_table.index_select(0, rows)
    if lengths.device.type == "cpu" and bool((lengths > width).any()):
        raise ValueError("gather width is shorter than a selected node state")
    return encoded_table[:, :width].index_select(0, rows), lengths


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
    extra_arrays: Mapping[str, NDArray[np.float32]] | None = None,
) -> None:
    """Validate EgoStitch-E2E pair-pass fp32 provenance and stored diagnostics.

    Args:
        logit: The scores artifact's primary logit column.
        meta: Parsed artifact metadata.
        label: Human-readable artifact label used in errors.
        require_diagnostics: Require the persisted descriptive diagnostics.
        extra_arrays: For family ``egostitch_e2e`` only: decomposition
            diagnostics keyed by name, including ``full_logit`` when the
            primary logit is a permanent-null arm.
            Ignored for every other family.

    Raises:
        ValueError: If an EgoStitch-E2E artifact lacks or contradicts its
            pinned pair-pass fp32 contract, is missing its ``f_logit``
            decomposition array, or its stored diagnostics are inconsistent;
            or if the artifact belongs to the retired frozen-s0 ``egostitch``
            family, whose validator was removed with its scorer.
    """
    family = meta.get("model_family")
    if family == "egostitch":
        # Fail closed rather than silently passing: the frozen-s0 family's
        # pair-pass fp32 validator was excised with its scorer, so a surviving
        # legacy artifact has no contract left to check. Its evidence is kept
        # under docs/results/ and outputs/egostitch_stage1/; score it from a
        # pre-excision commit if it must be re-analysed.
        raise ValueError(
            f"{label}: the frozen-s0 'egostitch' family is retired; its scorer and "
            "precision contract were removed. Re-analyse legacy artifacts from a "
            "pre-excision commit."
        )
    if family != "egostitch_e2e":
        return
    _validate_egostitch_e2e_precision(
        logit,
        meta=meta,
        label=label,
        require_diagnostics=require_diagnostics,
        extra_arrays=extra_arrays or {},
    )


def validate_artifact_precision(artifact: ScoresArtifact, *, label: str = "artifact") -> None:
    """Validate a loaded `ScoresArtifact`'s EgoStitch-E2E pair-pass fp32 provenance.

    Thin, family-agnostic wrapper over :func:`validate_score_precision`: it
    derives ``extra_arrays`` from the artifact's own diagnostic fields, so a caller
    that only has a `ScoresArtifact` (and
    does not know in advance whether its family is ``egostitch_e2e`` or a plain
    B0-family scorer) can validate it directly, the
    same way :func:`load_scores` returns it. Non-``egostitch_e2e`` artifacts
    simply pass an empty ``extra_arrays`` mapping, matching
    :func:`validate_score_precision`'s existing behavior for those families.

    This is the only correct entry point for an ``egostitch_e2e`` artifact:
    calling :func:`validate_score_precision` directly with just
    ``artifact.logit`` omits the decomposition arrays the contract requires and
    raises a spurious "missing arrays" error.

    Args:
        artifact: The loaded scores artifact (from :func:`load_scores` or
            :func:`merge_scores`).
        label: Human-readable artifact label used in errors.

    Raises:
        ValueError: If an EgoStitch-E2E artifact lacks or contradicts its
            pinned pair-pass fp32 contract, is missing its ``f_logit``
            decomposition array, or its stored diagnostics are inconsistent.
    """
    validate_test_access_ledger_binding(artifact.meta, label=label)
    extra_arrays: dict[str, NDArray[np.float32]] = {}
    if artifact.f_logit is not None:
        extra_arrays["f_logit"] = artifact.f_logit
    if artifact.pair_content is not None:
        extra_arrays["pair_content"] = artifact.pair_content
    if artifact.pair_topology is not None:
        extra_arrays["pair_topology"] = artifact.pair_topology
    if artifact.full_logit is not None:
        extra_arrays["full_logit"] = artifact.full_logit
    validate_score_precision(
        artifact.logit,
        meta=artifact.meta,
        label=label,
        extra_arrays=extra_arrays,
    )


def _is_heldout_universe(meta: Mapping[str, object]) -> bool:
    """Return whether `meta` describes an artifact over the held-out E2E universe.

    Held-out access is a data-boundary property of the family and pairs
    source, not of registration: family ``egostitch_e2e`` scoring the
    ``candidate``/``test`` manifests or an arbitrary ``file:`` source reads
    from the held-out universe. Everything else (``val_topology``/``val_cls``, other
    families) is not a held-out claim.

    An ``oracle_diagnostic`` artifact is excluded unconditionally, even when
    its ``pairs_source`` is ``candidate``/``test``/``file:*``: the
    ``oracle_struct`` generator consumes ground-truth topology by
    construction (it is fed the true test graph or the V_val validation-region
    truth at scoring time, never an imagined one), so it is always a ceiling
    diagnostic and never the formal held-out claim this predicate gates. This
    is what keeps such an artifact's ``heldout`` field ``False`` and its
    ``test_access_ledger`` binding unvalidated by
    :func:`validate_test_access_ledger_binding` -- the artifact cannot pass
    for a formal held-out result even if fed into that validation path.
    """
    if meta.get("model_family") != "egostitch_e2e":
        return False
    if meta.get("oracle_diagnostic") is not None:
        return False
    pairs_source = meta.get("pairs_source")
    return isinstance(pairs_source, str) and (
        pairs_source in {"candidate", "test", "test_topology"} or pairs_source.startswith("file:")
    )


def validate_test_access_ledger_binding(
    meta: Mapping[str, object],
    *,
    artifact_path: Path | None = None,
    label: str = "artifact",
) -> None:
    """Fail closed when held-out E2E score metadata is not bound to its ledger.

    Each shard binds the hash of its own append-only ledger record. A merged
    artifact binds the complete shard-record set. The ledger itself is a hash
    chain, so deletion or mutation of any historical record invalidates every
    later artifact without making valid artifacts fail when new records append.
    """
    if not _is_heldout_universe(meta):
        return
    heldout = meta.get("heldout")
    if heldout is None:
        # Fail closed: an e2e artifact over a held-out-shaped pairs_source with
        # no explicit heldout marker at all cannot be trusted to have skipped
        # this validation legitimately.
        raise ValueError(f"{label}: E2E artifact is missing the heldout marker")
    if heldout is not True:
        # `save_scores` always derives `heldout` from `_is_heldout_universe`
        # for a held-out-shaped artifact (family `egostitch_e2e` scoring
        # `candidate`/`test`/`file:*`), so there is no legitimate way such an
        # artifact carries anything but `True` here. `False` (or any non-bool)
        # is either tampering or a forged artifact -- fail closed rather than
        # treating it as a synthetic container that legitimately skipped this
        # validation.
        raise ValueError(
            f"{label}: held-out-shaped E2E artifact has heldout={heldout!r}, expected True"
        )
    raw_binding = meta.get("test_access_ledger")
    if not isinstance(raw_binding, dict):
        raise ValueError(f"{label}: held-out E2E artifact is missing test_access_ledger")
    binding = cast(dict[str, object], raw_binding)
    required = {
        "schema_version": str,
        "path": str,
        "record_sha256": str,
        "scoring_arm": str,
        "seed": int,
        "scoring_epoch": int,
        "num_shards": int,
        "output": str,
    }
    for key, expected_type in required.items():
        value = binding.get(key)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ValueError(f"{label}: invalid test_access_ledger.{key}")
    if binding["schema_version"] != _TEST_ACCESS_LEDGER_SCHEMA:
        raise ValueError(f"{label}: unsupported test-access ledger schema")
    ledger_path = Path(cast(str, binding["path"]))
    if not ledger_path.is_file():
        raise ValueError(f"{label}: test-access ledger is missing: {ledger_path}")
    with ledger_path.open(encoding="utf-8") as ledger:
        fcntl.flock(ledger.fileno(), fcntl.LOCK_SH)
        records = _validate_test_access_records(ledger, label=f"{label}: test-access ledger")

    raw_members = binding.get("records")
    if raw_members is None:
        raw_shard = binding.get("shard")
        if isinstance(raw_shard, bool) or not isinstance(raw_shard, int):
            raise ValueError(f"{label}: invalid test_access_ledger.shard")
        members = [binding]
    else:
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError(f"{label}: invalid test_access_ledger.records")
        if not all(isinstance(member, dict) for member in raw_members):
            raise ValueError(f"{label}: invalid test_access_ledger.records")
        members = cast(list[dict[str, object]], raw_members)
        member_digests = [member.get("record_sha256") for member in members]
        aggregate = hashlib.sha256(
            json.dumps(member_digests, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if binding["record_sha256"] != aggregate:
            raise ValueError(f"{label}: merged test-access ledger digest mismatch")

    by_digest = {record.get("record_sha256"): record for record in records}
    expected_common = {
        "scoring_arm": binding["scoring_arm"],
        "seed": binding["seed"],
        "scoring_epoch": binding["scoring_epoch"],
        "num_shards": binding["num_shards"],
        "output": binding["output"],
    }
    seen_shards: set[int] = set()
    for member in members:
        digest = member.get("record_sha256")
        record = by_digest.get(digest)
        if record is None:
            raise ValueError(f"{label}: bound test-access ledger record is missing")
        for key, expected in expected_common.items():
            if member.get(key, expected) != expected or record.get(key) != expected:
                raise ValueError(f"{label}: test-access ledger {key} mismatch")
        shard = member.get("shard")
        if isinstance(shard, bool) or not isinstance(shard, int) or record.get("shard") != shard:
            raise ValueError(f"{label}: test-access ledger shard mismatch")
        if shard in seen_shards:
            raise ValueError(f"{label}: duplicate test-access ledger shard")
        seen_shards.add(shard)
    num_shards = cast(int, binding["num_shards"])
    if raw_members is not None and seen_shards != set(range(num_shards)):
        raise ValueError(f"{label}: merged test-access ledger shard set is incomplete")

    if artifact_path is not None:
        output = Path(cast(str, binding["output"]))
        expected_path = (
            output
            if raw_members is not None or num_shards == 1
            else _shard_output_path(output, next(iter(seen_shards)))
        )
        if artifact_path.resolve() != expected_path.resolve():
            raise ValueError(f"{label}: artifact path contradicts test-access ledger output")


def _validate_egostitch_e2e_precision(
    logit: NDArray[np.float32],
    *,
    meta: Mapping[str, object],
    label: str,
    require_diagnostics: bool,
    extra_arrays: Mapping[str, NDArray[np.float32]],
) -> None:
    """Validate the ``egostitch_e2e`` two-array pair-pass fp32 contract.

    Args:
        logit: The published primary arm array.
        meta: Parsed artifact metadata.
        label: Human-readable artifact label used in errors.
        require_diagnostics: Require the persisted per-array descriptive diagnostics.
        extra_arrays: The non-primary diagnostic arrays, including
            ``full_logit`` when the primary arm is permanently nulled. May
            also carry the legacy ``pair_content``/``pair_topology`` arrays
            from a v3 artifact; those are ignored here (rev 4 has no content
            path), but their presence does not raise.

    Raises:
        ValueError: If the artifact lacks or contradicts the pinned
            ``egostitch_e2e_pair_fp32_v1`` contract, is missing its
            ``f_logit`` decomposition array, or its stored per-array
            diagnostics are inconsistent.
    """
    precision = meta.get("score_precision")
    if not isinstance(precision, dict):
        raise ValueError(
            f"{label}: EgoStitch-E2E artifact is missing score_precision provenance; "
            "rescore with the pair-pass fp32 contract"
        )
    expected = {
        "contract": _EGOSTITCH_E2E_PAIR_PRECISION_CONTRACT,
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
        raise ValueError(f"{label}: invalid EgoStitch-E2E score_precision provenance: {mismatches}")

    permanent_null = meta.get("permanent_null", "none")
    primary_logit = meta.get("primary_logit", "full")
    expected_primary = _e2e_primary_logit_key(str(permanent_null))
    if primary_logit != expected_primary:
        raise ValueError(
            f"{label}: primary_logit {primary_logit!r} contradicts permanent_null "
            f"{permanent_null!r}"
        )
    full = extra_arrays.get("full_logit", logit) if primary_logit != "full" else logit
    arrays: dict[str, NDArray[np.float32]] = {
        "full": full,
        "f_logit": extra_arrays.get("f_logit", logit),
    }
    if primary_logit != "full":
        if "full_logit" not in extra_arrays:
            raise ValueError(f"{label}: permanent-null e2e artifact requires full_logit")
        if not np.array_equal(logit, arrays[str(primary_logit)]):
            raise ValueError(f"{label}: logit does not equal its declared primary_logit arm")
    missing = [key for key in ("f_logit",) if key not in extra_arrays]
    if missing:
        raise ValueError(
            f"{label}: EgoStitch-E2E artifact is missing arrays: {missing}. "
            "This array may already be present on the loaded ScoresArtifact "
            "(as f_logit) even though this call did "
            "not pass it as extra_arrays; callers that only have a "
            "ScoresArtifact should use validate_artifact_precision(artifact, "
            "label=...) instead of calling validate_score_precision(logit, ...) "
            "directly."
        )
    non_float32 = [
        key for key in _EGOSTITCH_E2E_ARRAY_KEYS if np.asarray(arrays[key]).dtype != np.float32
    ]
    if non_float32:
        raise ValueError(f"{label}: EgoStitch-E2E arrays must be stored as float32: {non_float32}")

    if not require_diagnostics:
        return
    recorded = meta.get("score_resolution")
    if not isinstance(recorded, dict):
        raise ValueError(
            f"{label}: score_resolution diagnostics are missing or not a per-array mapping"
        )
    for key in _EGOSTITCH_E2E_ARRAY_KEYS:
        actual = score_resolution_diagnostics(arrays[key])
        if recorded.get(key) != actual:
            raise ValueError(
                f"{label}: score_resolution[{key!r}] diagnostics are missing or inconsistent; "
                f"recorded={recorded.get(key)!r}, actual={actual!r}"
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
    f_logit: NDArray[np.float32] | None = None,
    pair_content: NDArray[np.float32] | None = None,
    pair_topology: NDArray[np.float32] | None = None,
    full_logit: NDArray[np.float32] | None = None,
) -> None:
    """Write a scores artifact in the pinned ``.npz`` format.

    Args:
        path: Destination ``.npz`` path (parent directories are created).
        node_ids: Unique node-id strings, sorted ascending.
        u_idx: Shape ``(n,)`` int32 indices into `node_ids`.
        v_idx: Shape ``(n,)`` int32 indices into `node_ids`.
        logit: Shape ``(n,)`` float32 raw primary model logits.
        label: Shape ``(n,)`` int8 labels (``-1`` for unlabeled rows).
        row_start: Row offset of this artifact in the full input (0 unless sharded).
        meta: Metadata dict; must contain the pinned keys and be JSON-serializable.
            The stored copy always gets an explicit ``heldout`` boolean set from
            `meta`'s own ``model_family``/``pairs_source`` (see
            :func:`_is_heldout_universe`); any caller-supplied ``heldout`` value
            is overwritten.
        f_logit: For family ``egostitch_e2e`` only: the frozen-topology arm,
            shape ``(n,)`` float32. Required when `meta`'s ``model_family`` is
            ``egostitch_e2e``; ignored otherwise.
        pair_content: Deleted-content-path compatibility parameter. Content
            path removed rev 4 (design doc §9) -- new artifacts never write
            it. Must be ``None``; a non-``None`` value raises.
        pair_topology: Deleted-content-path compatibility parameter, same
            contract as `pair_content`. With content gone, ``pair_topology``
            would be numerically identical to `full`, so it is retired
            rather than kept as a redundant write.
        full_logit: For a permanent-null ``egostitch_e2e`` primary only: the
            true full-arm diagnostic. New artifacts always physically publish
            it as ``full``; this argument remains optional when `logit` carries
            the full arm.

    Raises:
        ValueError: If array lengths disagree, indices fall outside `node_ids`,
            `row_start` is negative, `meta` is missing pinned keys, `pair_content`
            or `pair_topology` is supplied (non-``None``), an ``egostitch_e2e``
            artifact violates the pair-pass fp32 provenance contract, or an
            ``egostitch_e2e`` artifact is missing its `f_logit` array.
    """
    if pair_content is not None or pair_topology is not None:
        raise ValueError(
            f"{path}: pair_content/pair_topology are retired with the content path "
            "(design doc §9) and must not be supplied to save_scores; new "
            "egostitch_e2e artifacts publish only full/f_logit. Legacy v3 "
            "artifacts that already carry these two arrays remain readable "
            "via load_scores/merge_scores."
        )
    n = len(logit)
    if not (len(u_idx) == len(v_idx) == len(label) == n):
        raise ValueError(
            "u_idx, v_idx, logit, and label must have identical lengths; got "
            f"{len(u_idx)}, {len(v_idx)}, {n}, {len(label)}"
        )
    for name, arr in (
        ("f_logit", f_logit),
        ("full_logit", full_logit),
    ):
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} must have length {n}, got {len(arr)}")
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
    stored_meta["heldout"] = _is_heldout_universe(stored_meta)
    if stored_meta.get("model_family") == "egostitch_e2e":
        stored_meta["scores_meta_version"] = _SCORES_META_VERSION
        if f_logit is None:
            raise ValueError(f"{path}: egostitch_e2e artifacts require an f_logit array")
        extra_arrays = {"f_logit": f_logit}
        if full_logit is not None:
            extra_arrays["full_logit"] = full_logit
        validate_score_precision(
            logit,
            meta=stored_meta,
            label=str(path),
            require_diagnostics=False,
            extra_arrays=extra_arrays,
        )
        stored_meta["score_resolution"] = {
            key: score_resolution_diagnostics(arr)
            for key, arr in {
                "full": full_logit if full_logit is not None else logit,
                "f_logit": f_logit,
            }.items()
        }
        validate_score_precision(
            logit, meta=stored_meta, label=str(path), extra_arrays=extra_arrays
        )
        validate_test_access_ledger_binding(stored_meta, artifact_path=path, label=str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    if f_logit is None and full_logit is None:
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
    else:
        # egostitch_e2e (the only family reaching here): f_logit is required;
        # the true full arm is either `logit` or the explicit `full_logit`
        # array when a permanent null is primary.
        assert f_logit is not None
        arrays: dict[str, Any] = {
            "node_ids": np.array(list(node_ids), dtype=np.str_),
            "u_idx": u_idx.astype(np.int32, copy=False),
            "v_idx": v_idx.astype(np.int32, copy=False),
            "logit": logit.astype(np.float32, copy=False),
            "label": label.astype(np.int8, copy=False),
            "row_start": np.int64(row_start),
            "meta": np.array(json.dumps(stored_meta, sort_keys=True)),
            "full": (full_logit if full_logit is not None else logit).astype(
                np.float32, copy=False
            ),
            "f_logit": f_logit.astype(np.float32, copy=False),
        }
        if full_logit is not None:
            arrays["full_logit"] = full_logit.astype(np.float32, copy=False)
        np.savez_compressed(path, **arrays)


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
        current_e2e = meta.get("model_family") == "egostitch_e2e"
        scores_meta_version_ok = (
            meta.get("scores_meta_version") in _SUPPORTED_E2E_SCORES_META_VERSIONS
        )
        if current_e2e and not scores_meta_version_ok:
            raise ValueError(
                f"{path}: scores_meta_version must be one of "
                f"{_SUPPORTED_E2E_SCORES_META_VERSIONS!r}; got {meta.get('scores_meta_version')!r}"
            )
        if current_e2e and "full" not in data:
            raise ValueError(f"{path}: formal egostitch_e2e artifact is missing full")
        f_logit: NDArray[np.float32] | None = (
            data["f_logit"].astype(np.float32, copy=False) if "f_logit" in data else None
        )
        pair_content: NDArray[np.float32] | None = (
            data["pair_content"].astype(np.float32, copy=False) if "pair_content" in data else None
        )
        pair_topology: NDArray[np.float32] | None = (
            data["pair_topology"].astype(np.float32, copy=False)
            if "pair_topology" in data
            else None
        )
        primary_logit = meta.get("primary_logit", "full")
        if current_e2e and primary_logit == "full" and not np.array_equal(data["full"], logit):
            raise ValueError(f"{path}: formal E2E full array contradicts primary logit")
        if "full" in data and primary_logit != "full":
            full_logit = data["full"].astype(np.float32, copy=False)
        elif "full_logit" in data:
            # Backward compatibility with v1 permanent-null artifacts.
            full_logit = data["full_logit"].astype(np.float32, copy=False)
        else:
            full_logit = None
    validate_test_access_ledger_binding(meta, artifact_path=path, label=str(path))
    return _Shard(
        path=path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=logit,
        label=label,
        row_start=row_start,
        meta=meta,
        f_logit=f_logit,
        pair_content=pair_content,
        pair_topology=pair_topology,
        full_logit=full_logit,
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
        f_logit=shard.f_logit,
        pair_content=shard.pair_content,
        pair_topology=shard.pair_topology,
        full_logit=shard.full_logit,
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
    for shard in shards:
        if "topo_gen_control" not in shard.meta:
            raise ValueError(
                f"merge input {shard.path} is missing required meta 'topo_gen_control'"
            )
    reference_topo_gen_control = reference.meta["topo_gen_control"]
    if any(shard.meta["topo_gen_control"] != reference_topo_gen_control for shard in shards[1:]):
        raise ValueError(
            "merge inputs disagree on meta 'topo_gen_control': "
            f"{[shard.meta['topo_gen_control'] for shard in shards]} "
            f"(files: {[str(shard.path) for shard in shards]})"
        )
    for key in (
        "checkpoint_id",
        "model_family",
        "pairs_source",
        "strategy",
        "num_rows",
        "score_precision",
        "scaffold_control",
        "permanent_null",
        "primary_logit",
        "scores_meta_version",
        "heldout",
        "oracle_diagnostic",
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
    if reference.meta.get("model_family") == "egostitch_e2e" and isinstance(
        reference.meta.get("test_access_ledger"), dict
    ):
        bindings = [cast(dict[str, object], shard.meta["test_access_ledger"]) for shard in ordered]
        common_keys = (
            "schema_version",
            "path",
            "scoring_arm",
            "seed",
            "scoring_epoch",
            "scoring_run_id",
            "num_shards",
            "output",
        )
        for key in common_keys:
            if any(binding.get(key) != bindings[0].get(key) for binding in bindings[1:]):
                raise ValueError(f"merge inputs disagree on test-access ledger {key!r}")
        member_digests = [binding["record_sha256"] for binding in bindings]
        merged_meta["test_access_ledger"] = {
            **{key: bindings[0][key] for key in common_keys},
            "record_sha256": hashlib.sha256(
                json.dumps(member_digests, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "records": bindings,
        }
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

    extra: dict[str, NDArray[np.float32]] = {}
    if reference.meta.get("model_family") == "egostitch_e2e":
        f_logit_parts = [shard.f_logit for shard in ordered]
        if any(part is None for part in f_logit_parts):
            raise ValueError(
                "egostitch_e2e merge requires 'f_logit' in every shard "
                f"(files: {[str(shard.path) for shard in ordered]})"
            )
        extra["f_logit"] = np.concatenate(cast(list[NDArray[np.float32]], f_logit_parts))
        # pair_content/pair_topology are the retired content-path arrays
        # (design doc §9): a v4 shard never carries them (all None here), a
        # legacy v3 shard always does (save_scores required all three
        # together at v3 write time). Mixing the two vintages in one merge
        # is already rejected above by the scores_meta_version equality
        # check, so "all present" vs. "all absent" are the only two shapes
        # that can reach this point; the mixed branch is a defensive
        # fail-closed guard, not an expected path.
        for name in ("pair_content", "pair_topology"):
            legacy_parts = [getattr(shard, name) for shard in ordered]
            if all(part is not None for part in legacy_parts):
                extra[name] = np.concatenate(cast(list[NDArray[np.float32]], legacy_parts))
            elif any(part is not None for part in legacy_parts):
                raise ValueError(
                    f"egostitch_e2e merge has inconsistent {name!r} presence across shards "
                    f"(files: {[str(shard.path) for shard in ordered]})"
                )
        full_parts = [shard.full_logit for shard in ordered]
        if any(part is not None for part in full_parts):
            if any(part is None for part in full_parts):
                raise ValueError(
                    "egostitch_e2e merge requires full_logit in every shard when present"
                )
            extra["full_logit"] = np.concatenate(cast(list[NDArray[np.float32]], full_parts))

    return ScoresArtifact(
        node_ids=node_ids,
        u_idx=np.concatenate(u_parts) if u_parts else np.empty(0, dtype=np.int32),
        v_idx=np.concatenate(v_parts) if v_parts else np.empty(0, dtype=np.int32),
        logit=np.concatenate([shard.logit for shard in ordered]),
        label=np.concatenate([shard.label for shard in ordered]),
        meta=merged_meta,
        f_logit=extra.get("f_logit"),
        pair_content=extra.get("pair_content"),
        pair_topology=extra.get("pair_topology"),
        full_logit=extra.get("full_logit"),
    )


# ---------------------------------------------------------------------------
# Model rebuild from the Task-4 checkpoint format
# ---------------------------------------------------------------------------


def _build_v3_1(model_config: dict[str, object]) -> nn.Module:
    """Build a `V3_1` model from its checkpointed config."""
    return V3_1(**model_config)


def _build_egostitch_e2e(model_config: dict[str, object]) -> nn.Module:
    """Build an `EgoStitchModel` from its checkpointed config (design rev 3).

    Three-component refactor design (2026-08-02) Sec 8/P3: `E2EConfig` is now
    three nested sub-configs -- `generator`/`encoder`/`classifier`, each its
    own strict `from_mapping` (`GeneratorConfig`/`EncoderConfig`/
    `ClassifierConfig`) -- in place of the old flat field list. The internal
    Stage-1 generator still keeps its own pinned spec defaults
    (``EgoStitchConfig()``, spec Sec 13) except for the fields the nested
    `generator` sub-config supersedes (`n_ground`, `tau_adj`, `tau_div`,
    `l_gate_pos_weight`, spec Sec 14.4.1-14.4.4). `EgoStitchModel` replaces
    `EgoStitchE2E`; the family name `egostitch_e2e` and `E2EConfig` itself
    are unchanged (design Sec 8).

    No compatibility shim is written for the pre-refactor flat schema
    (design Sec 11): a checkpoint whose `model_config` still carries the old
    top-level fields (`d_model`/`ste_dim`/`n_ground`/...) instead of the three
    nested sub-configs predates this refactor and cannot be loaded.
    `E2EConfig.from_mapping` itself detects that shape (it recognizes every
    pre-P3 flat field name) and raises naming the schema change -- "E2EConfig
    no longer accepts the flat model.config schema ... nest fields under
    'generator:'/'encoder:'/'classifier:' instead" -- rather than the opaque
    "unknown E2E config keys" a plain unknown-key rejection would give for a
    mapping that never mentions `generator`/`encoder`/`classifier` at all, so
    no separate guard is needed here.
    """
    from src.model.egostitch.composite import EgoStitchModel
    from src.model.egostitch.config import E2EConfig

    # Checkpointed configs are inherently dynamic (parsed from a .pt payload),
    # so the mapping goes through Any deliberately.
    return EgoStitchModel(E2EConfig.from_mapping(cast(dict[str, Any], model_config)))


def _cazi_model_config_from_state(model_state: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Infer `CAZIStudent`'s constructor args from its own checkpoint tensor shapes.

    CAZI-MBN's own training YAML (``configs/cazi_mbn_breadth_first.yaml``) has
    no ``model.family``/``model.config`` nesting -- it is a flat
    ``data``/``model``/``optim``/``runtime``/``loss`` schema entirely unlike
    every other family's checkpointed config -- so this bypasses
    `_load_model_config` and instead reads the shapes `CAZIStudent.__init__`
    (`src/baselines/cazi_mbn.py`) itself allocated back out of the checkpoint:
    self-describing, and unable to drift from the checkpoint that actually
    produced it. `--model-config` still points at the training YAML for
    `_run_score`'s cazi_mbn scoring branch (feature standardization and the
    release's known feature gaps), just not for this.

    Args:
        model_state: The unwrapped ``state_dict`` (already extracted from the
            released ``{"state_dict": ..., "best_val_auroc": ...}`` checkpoint
            wrapper by :func:`_load_checkpoint`).

    Returns:
        ``{"sequence_dim": ..., "latent_dim": ..., "network_layers": ...}``.

    Raises:
        ValueError: If the expected `CAZIStudent` parameter tensors are absent
            or malformed.
    """
    weight = model_state.get("latent_projection.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("cazi_mbn checkpoint is missing a 2-D latent_projection.weight tensor")
    latent_dim, sequence_dim = (int(dim) for dim in weight.shape)
    gate_weight = model_state.get("classifier.classifier.gating_network.weight")
    if not isinstance(gate_weight, torch.Tensor) or gate_weight.ndim != 2:
        raise ValueError(
            "cazi_mbn checkpoint is missing a 2-D "
            "classifier.classifier.gating_network.weight tensor"
        )
    network_layers = int(gate_weight.shape[0])
    return {
        "sequence_dim": sequence_dim,
        "latent_dim": latent_dim,
        "network_layers": network_layers,
    }


def _build_cazi_mbn(model_config: dict[str, object]) -> nn.Module:
    """Build a `CAZIStudent` from the config `_cazi_model_config_from_state` inferred."""
    from src.baselines.cazi_mbn import CAZIStudent

    return CAZIStudent(**cast(dict[str, Any], model_config))


MODEL_BUILDERS: dict[str, Callable[[dict[str, object]], nn.Module]] = {
    "v3_1": _build_v3_1,
    "egostitch_e2e": _build_egostitch_e2e,
    "cazi_mbn": _build_cazi_mbn,
}


def build_model(model_family: str, model_config: dict[str, object]) -> nn.Module:
    """Build an untrained scorer of the given family from a checkpointed config.

    ``f0_mlp`` (the B0-alt baseline, ``src/model/b0_alt.py``'s ``F0PairMLP``) has no
    entry here: it was removed 2026-08-03 by owner decision. Its scoring path
    (:func:`_score_f0_mlp`) and CLI plumbing remain, since neither imports the deleted
    module; a caller may still register a ``"f0_mlp"`` builder into
    :data:`MODEL_BUILDERS` itself (as the tests do) to score against a different model
    under that family name.

    Args:
        model_family: One of the keys of :data:`MODEL_BUILDERS` (``v3_1``/
            ``egostitch_e2e`` by default).
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


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Load a JSON object with a provenance-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} {path} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return cast(dict[str, object], value)


def _load_checkpoint(
    path: Path,
    *,
    model_family: str | None = None,
    model_config: dict[str, object] | None = None,
) -> tuple[nn.Module, str, str]:
    """Rebuild the frozen model from a current or supported legacy checkpoint.

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
    elif model_family == "cazi_mbn":
        # The CAZI-MBN release's own `train_student`/`train_teacher`
        # (`src/train_cazi_mbn.py`) write `{"state_dict": ..., "best_val_auroc":
        # ...}`, never this file's Task-4 keys: unwrap the released wrapper
        # explicitly instead of trying to `load_state_dict` it as-is (which
        # would treat "state_dict"/"best_val_auroc" themselves as parameter
        # names and fail confusingly).
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"cazi_mbn checkpoint {path} must be the released "
                "{'state_dict': ..., 'best_val_auroc': ...} format written by "
                "src.train_cazi_mbn.train_student/train_teacher"
            )
        model_state = cast(dict[str, torch.Tensor], state_dict)
        model_config = _cazi_model_config_from_state(model_state)
    else:
        if model_family is None or model_config is None:
            raise ValueError(
                f"bare legacy checkpoint {path} requires explicit model_family and model_config"
            )
        model_state = cast(dict[str, torch.Tensor], checkpoint)

    if model_family == "egostitch_e2e":
        # Three-component refactor design (2026-08-02) §5: `model.ste` (the
        # `STEncoder`) is now `model.encoder` (a `TypedMessagePassingEncoder`),
        # so the stitched-topology-encoder's input projection now serializes
        # under the `encoder.` prefix, not `ste.`. Keying this check on the
        # old `ste.embed.weight` name would silently stop finding the key on
        # any current-format checkpoint, so the shape guard below would never
        # fire again -- vacuously passing on a genuinely incompatible
        # checkpoint instead of rejecting it.
        scaffold_embed = model_state.get("encoder.embed.weight")
        if (
            isinstance(scaffold_embed, torch.Tensor)
            and scaffold_embed.ndim == 2
            and scaffold_embed.shape[1] == 9
        ):
            raise ValueError(
                f"checkpoint {path}: the rev-3.1 scaffold change expanded FEAT_DIM 9 to 11 "
                "and EDGE_TYPES 3 to 4 (spec section 14.4.2); pre-rev-3.1 e2e checkpoints "
                "are not loadable under rev-3.1 code and must be scored from a "
                "pre-rev-3.1 commit"
            )
        # No checkpoint-config normalization: the rev-3.1 backfills were deleted
        # with the content path (design 2026-08-02 §11), so a checkpoint must
        # carry its own complete model config. One that does not fails loudly at
        # the strict `load_state_dict` below rather than being reconstructed
        # under a config it was never trained with.
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


def _ledger_record_sha256(record: Mapping[str, object]) -> str:
    """Return the canonical digest of one ledger record, excluding its digest field."""
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_test_access_records(
    raw_lines: Iterable[str], *, label: str
) -> list[dict[str, object]]:
    """Parse and validate the append-only ledger hash chain."""
    records: list[dict[str, object]] = []
    previous: str | None = None
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is malformed at line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict) or record.get("schema_version") != (
            _TEST_ACCESS_LEDGER_SCHEMA
        ):
            raise ValueError(f"{label} has an invalid record at line {line_number}")
        typed = cast(dict[str, object], record)
        recorded_digest = typed.get("record_sha256")
        if not isinstance(recorded_digest, str) or recorded_digest != _ledger_record_sha256(typed):
            raise ValueError(f"{label} has a digest mismatch at line {line_number}")
        if typed.get("previous_record_sha256") != previous:
            raise ValueError(f"{label} has a broken hash chain at line {line_number}")
        records.append(typed)
        previous = recorded_digest
    return records


def _record_test_access(context: _TestAccessContext, *, pairs_source: str) -> dict[str, object]:
    """Append one held-out manifest read while enforcing scoring-epoch uniqueness.

    An epoch's identity is its ``--scoring-run-id`` when the caller supplies
    one, so passes that share a run id belong to the SAME scoring epoch even
    when they score different pairs sources (e.g. ``test`` then
    ``candidate`` for the same arm/seed, one full test-protocol invocation).
    Without a run id, an output's own resolved path stands in for it instead
    -- the original behavior, preserved exactly: a call scoring a different
    output with no run id is a different, unshared run.

    Within one epoch, shard bookkeeping (duplicate-shard rejection, the
    ``shard < num_shards`` cap) is scoped to each output separately, since one
    epoch may now span several outputs, each with its own shard count. A call
    that does not join the latest epoch -- a mismatched run id/output, a
    shard already recorded for its output, or that output's shard budget
    already exhausted -- starts a new epoch and requires ``--rescore-reason``.
    """
    context.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    output_key = str(context.output.resolve())
    with context.ledger_path.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
        ledger.seek(0)
        records = _validate_test_access_records(ledger, label="test-access ledger")

        matching = [
            record
            for record in records
            if record.get("event") == "resolve_pairs"
            and record.get("scoring_arm") == context.scoring_arm
            and record.get("seed") == context.seed
        ]
        epoch = 1
        if matching:
            epoch_values = [record.get("scoring_epoch") for record in matching]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in epoch_values):
                raise ValueError("test-access ledger contains an invalid scoring_epoch")
            latest_epoch = max(cast(list[int], epoch_values))
            latest = [record for record in matching if record["scoring_epoch"] == latest_epoch]
            latest_run_key = latest[0].get("scoring_run_id") or latest[0].get("output")
            current_run_key: object = context.scoring_run_id or output_key
            same_output = [record for record in latest if record.get("output") == output_key]
            used_shards = {record.get("shard") for record in same_output}
            joins_latest_epoch = (
                current_run_key == latest_run_key
                and all(record.get("rescore_reason") == context.rescore_reason for record in latest)
                and all(record.get("num_shards") == context.num_shards for record in same_output)
                and context.shard not in used_shards
                and len(used_shards) < context.num_shards
            )
            if joins_latest_epoch:
                epoch = latest_epoch
            else:
                if context.rescore_reason is None:
                    raise ValueError(
                        "held-out scoring already has an epoch for "
                        f"arm={context.scoring_arm!r}, seed={context.seed}; "
                        "repeat scoring requires --rescore-reason"
                    )
                epoch = latest_epoch + 1

        record = {
            "schema_version": _TEST_ACCESS_LEDGER_SCHEMA,
            "event": "resolve_pairs",
            "accessed_utc": datetime.now(UTC).isoformat(),
            "scoring_epoch": epoch,
            "scoring_arm": context.scoring_arm,
            "seed": context.seed,
            "scoring_run_id": context.scoring_run_id,
            "pairs_source": pairs_source,
            "output": output_key,
            "shard": context.shard,
            "num_shards": context.num_shards,
            "rescore_reason": context.rescore_reason,
            "previous_record_sha256": (records[-1]["record_sha256"] if records else None),
        }
        record["record_sha256"] = _ledger_record_sha256(record)
        ledger.seek(0, 2)
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
        context.ledger_binding = {
            "schema_version": _TEST_ACCESS_LEDGER_SCHEMA,
            "path": str(context.ledger_path.resolve()),
            "record_sha256": record["record_sha256"],
            "scoring_arm": context.scoring_arm,
            "seed": context.seed,
            "scoring_epoch": epoch,
            "scoring_run_id": context.scoring_run_id,
            "shard": context.shard,
            "num_shards": context.num_shards,
            "output": output_key,
        }
        return record


def _load_val_region_split(data_root: Path, strategy: str) -> ValRegionSplit:
    """Rebuild the deterministic ``V_val`` region split bit-for-bit from disk.

    Mirrors the training-side derivation exactly: the same ``split.pkl``
    train-node collection, ``train_graph.pkl`` edge set, ``train_edges.txt``
    plus ``val_edges.txt`` label-0 rows (in file order), and
    ``positive_edges.txt`` global positive set feed the same
    :func:`~src.data.val_region.derive_val_region_split` call training itself
    runs, so the returned split -- and therefore its ``v_val`` region and
    validation topology -- is exactly what training validates against and
    never a re-derivation that could quietly drift from it.

    Shared by the sampled-union ``val_topology`` source and its V_val-truth
    oracle diagnostic graph.

    Args:
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Split strategy name (e.g. ``breadth_first``).

    Returns:
        The deterministic `ValRegionSplit`.

    Raises:
        ValueError: If ``split.pkl`` does not contain a ``train`` node collection.
    """
    strategy_dir = data_root / _BENCHMARK_SUBDIR / strategy
    with (strategy_dir / "split.pkl").open("rb") as handle:
        split_payload = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    if not isinstance(split_payload, dict) or "train" not in split_payload:
        raise ValueError(f"{strategy_dir / 'split.pkl'} must contain a train node collection")
    train_nodes = cast(Iterable[str], split_payload["train"])

    with (strategy_dir / "train_graph.pkl").open("rb") as handle:
        train_graph = pickle.load(handle)  # noqa: S301 - repository benchmark artifact
    truth_edges = list(train_graph.edges())

    benchmark_negatives: list[tuple[str, str]] = []
    for filename in ("train_edges.txt", "val_edges.txt"):
        pairs, labels = _read_pairs_tsv(strategy_dir / filename)
        benchmark_negatives.extend(
            pair for pair, label in zip(pairs, labels.tolist(), strict=True) if label == 0
        )

    positive_pairs, _ = _read_pairs_tsv(data_root / _BENCHMARK_SUBDIR / "positive_edges.txt")

    return derive_val_region_split(
        train_nodes,
        truth_edges,
        benchmark_negatives,
        frozenset(positive_pairs),
        params=_VAL_REGION_PARAMS,
    )


def _resolve_val_topology_pairs(
    data_root: Path, strategy: str
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Materialize only the pair union used by sampled V_val topology balls.

    A support-only self row is retained for every V_val node so EgoStitch's F0
    and grounding pool remain identical to the complete V_val role universe.
    """
    split = _load_val_region_split(data_root, strategy)
    pair_set = {
        canonical_pair(u, v)
        for node_sets in split.buckets.values()
        for nodes in node_sets
        for u, v in combinations_with_replacement(sorted(nodes), 2)
    }
    pair_set.update((node, node) for node in split.v_val)
    pairs = sorted(pair_set)
    positives = frozenset(split.val_positives)
    labels = np.fromiter(
        (int(pair in positives) for pair in pairs), dtype=np.int8, count=len(pairs)
    )
    return pairs, labels


def _resolve_val_cls_pairs(
    data_root: Path, strategy: str
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Materialize the balanced V_val classification rows, positives then negatives.

    These are exactly the rows training validates classification on
    (`ValRegionSplit.val_cls_pairs`), so a threshold selected here is selected
    on the model-selection set, never on held-out data.
    """
    split = _load_val_region_split(data_root, strategy)
    return list(split.val_cls_pairs), np.asarray(split.val_cls_labels, dtype=np.int8)


def _oracle_truth_graph_for_scoring(pairs_source: str, data_root: Path, strategy: str) -> nx.Graph:
    """Return the ground-truth topology an ``oracle_struct`` diagnostic scores against.

    Only two truth sources are supported, matching
    ``docs/superpowers/specs/2026-08-04-oracle-scaffold-experiment-design.md``
    §3's Row R2 (the wave-2 diagnostic this module owns) and the V_val
    validation-region truth the same document's Row R1-true-oracle training
    diagnostic reads:

    - ``test``/``candidate``/``file:*``: the labeled benchmark test graph
      (self-loop-stripped, matching ``EgoTargetBuilder``'s self-loop-free
      requirement) -- the R2 diagnostic ceiling.
    - ``val_topology``/``val_cls``: the V_val validation-region truth topology
      (:func:`_load_val_region_split`'s ``build_g_val_simple()``), already
      self-loop-free by construction.

    Every other pairs source has no defined truth graph here and is refused
    rather than guessed at.

    Args:
        pairs_source: The same ``--pairs`` spec passed to :func:`_resolve_pairs`.
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Split strategy name (e.g. ``breadth_first``).

    Returns:
        The self-loop-free ground-truth `networkx.Graph`.

    Raises:
        ValueError: If `pairs_source` has no defined oracle truth graph.
    """
    if pairs_source in ("test", "candidate", "test_topology") or pairs_source.startswith("file:"):
        # Local imports: `src.experiments.g1_hardened_e2` imports back from this
        # module (`ScoresArtifact`/`load_scores`/`validate_score_precision`), so
        # a module-level import here would be circular.
        from src.eval.graph_metrics import strip_self_loops
        from src.experiments.g1_hardened_e2 import load_test_graph

        return strip_self_loops(load_test_graph(data_root / _BENCHMARK_SUBDIR, strategy))
    if pairs_source in _VAL_PAIR_SOURCES:
        return _load_val_region_split(data_root, strategy).build_g_val_simple()
    raise ValueError(
        f"oracle_struct scoring has no diagnostic truth graph for --pairs {pairs_source!r}; "
        "supported sources are test/candidate/test_topology/file:<path> (the labeled test graph, "
        "oracle-scaffold design doc Row R2) and val_topology/val_cls "
        "(V_val validation-region truth)"
    )


def _oracle_truth_graph_sha256(graph: nx.Graph) -> str:
    """Return a stable digest of a complete oracle truth graph, including isolates."""
    nodes = sorted(str(node) for node in graph.nodes)
    edges = sorted(canonical_pair(str(node_u), str(node_v)) for node_u, node_v in graph.edges)
    payload = json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _install_oracle_context(
    model: EgoStitchModel, node_ids: Sequence[str], *, truth_graph: nx.Graph
) -> None:
    """Install the role-specific truth context required by an oracle generator.

    Mirrors ``src.train_egostitch._install_oracle_context``'s F0-row =
    table-row identity lookup, specialized to this scoring call's own node
    universe (`node_ids`, already the exact row order :func:`build_f0_matrix`
    assigned) rather than training's all-train-nodes universe.

    Args:
        model: The `EgoStitchModel`, whose ``generator`` must already be an
            `OracleStructGenerator`.
        node_ids: This call's F0-row-ordered node universe.
        truth_graph: The self-loop-free ground-truth graph to read scaffolds from.
    """
    from src.model.egostitch.generator.full_oracle import FullOracleGenerator
    from src.model.egostitch.generator.oracle import OracleStructGenerator, build_oracle_table

    if isinstance(model.generator, OracleStructGenerator):
        table = build_oracle_table(
            truth_graph,
            list(node_ids),
            slots=model.generator_cfg.slots,
            seed=model.cfg.generator.oracle_seed,
        )
        lookup = torch.arange(len(node_ids), dtype=torch.long)
        model.generator.set_oracle_context(table, lookup)
        return
    if isinstance(model.generator, FullOracleGenerator):
        full_truth_graph = truth_graph.copy()
        # A queryable node with no positive incident edge is a legitimate
        # isolate. Make that node membership explicit before the generator's
        # strict context validation rather than weakening its graph contract.
        full_truth_graph.add_nodes_from(node_ids)
        model.generator.set_oracle_context(full_truth_graph, node_ids)
        return
    raise TypeError(
        "oracle context install requires a registered oracle generator, got "
        f"{type(model.generator).__name__}"
    )


def _resolve_pairs(
    pairs_source: str,
    data_root: Path,
    strategy: str,
    *,
    test_access: _TestAccessContext | None = None,
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Resolve a ``--pairs`` spec to canonical pairs plus labels, in file row order.

    Args:
        pairs_source: ``candidate``, ``test``, ``test_topology``, ``val_topology``,
            ``val_cls``, or ``file:<path.tsv>``.
        data_root: Directory containing ``benchmark_2025_neurips/`` and
            ``features/frozen_node_features_1024/``.
        strategy: Split strategy name (e.g. ``breadth_first``).
        test_access: Held-out scoring identity to append before the pair manifest
            is read; ``None`` for the non-held-out ``val_topology``/``val_cls`` sources.

    Returns:
        ``(pairs, labels)``, aligned index-for-index.

    Raises:
        ValueError: If `pairs_source` is not one of the supported forms.
    """
    if test_access is not None:
        _record_test_access(test_access, pairs_source=pairs_source)
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    if pairs_source == "candidate":
        labeled = load_candidate_pairs(benchmark_root, strategy)
        return labeled.pairs, labeled.labels
    if pairs_source == "test_topology":
        labeled = load_test_topology_pairs(benchmark_root, strategy)
        return labeled.pairs, labeled.labels
    if pairs_source == "test":
        return _read_pairs_tsv(benchmark_root / strategy / f"{pairs_source}_edges.txt")
    if pairs_source == "val_topology":
        return _resolve_val_topology_pairs(data_root, strategy)
    if pairs_source == "val_cls":
        return _resolve_val_cls_pairs(data_root, strategy)
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
    pair_amp: str | None = None,
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
        with torch.inference_mode(), _autocast_context(device, pair_amp or amp):
            encoded_a = encoded.index_select(0, pair_a)[:, :boundary]
            encoded_b = encoded.index_select(0, pair_b)[:, :boundary]
            pair_repr = model._pair_representation(
                encoded_a,
                encoded_b,
                len_a,
                len_b,
            )
            if model.topo_gen is None:
                logits = model.output_head(pair_repr)
            else:
                logits = model.topo_gen.marginal_forward(
                    encoded_a, encoded_b, len_a, len_b, pair_repr, model.output_head
                )["logits"]
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


def _score_cazi_mbn(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
    batch_size: int,
    feature_stats: FeatureStats,
    missing_features: frozenset[str],
) -> NDArray[np.float32]:
    """Score pairs with a `CAZIStudent`, matching `src.train_cazi_mbn`'s own path exactly.

    Reuses `src.train_cazi_mbn._standardize_f0` verbatim so standardization is
    bit-for-bit identical to the released training/scoring code, rather than a
    parallel reimplementation that could silently drift. Only the final
    sigmoid is skipped -- `train_cazi_mbn.score_pairs` returns probabilities,
    but this file's artifact contract pins ``logit`` to the raw pre-sigmoid
    value for every family, so this calls `CAZIStudent.pair_logits` directly.

    `missing_features` (the release's two known feature-store gaps,
    `CAZIConfig.expected_missing_features`) get an all-zero raw F0 row before
    standardization, exactly as `src.train_cazi_mbn._test_features` does for
    the benchmark's own held-out test nodes: a node this scorer has never seen
    raw features for still needs a row to standardize and classify, not a
    `KeyError`.

    Args:
        model: Frozen `CAZIStudent`, in `eval()` mode (moved onto `device` here).
        pairs: Node-id pairs in input row order.
        store: Feature store providing per-node raw F0 rows.
        device: Compute device.
        batch_size: Pair rows per forward pass.
        feature_stats: The training run's registered standardization constants
            (its ``feature_stats.npz``, next to its checkpoint's ``output_dir``).
        missing_features: Node ids the release's feature store never covers.

    Returns:
        Shape ``(len(pairs),)`` float32 raw logits in input row order.
    """
    from src.baselines.cazi_mbn import CAZIStudent
    from src.train_cazi_mbn import _standardize_f0

    assert isinstance(model, CAZIStudent)
    model.to(device)
    model.eval()
    node_ids = sorted({node_id for pair in pairs for node_id in pair})
    present = [node for node in node_ids if node not in missing_features]
    # No F0 cache here (unlike v3_1/egostitch_e2e's `--f0-cache`): caching this
    # matrix would need the same `allow_cache_subset` gather CLAUDE.md flags as
    # a silent-mismatch trap, and this scorer's own universe is typically small
    # enough that recomputing it is cheap.
    present_f0, present_position = build_f0_matrix(store, present, cache_path=None)
    raw = torch.zeros((len(node_ids), present_f0.shape[1]), dtype=torch.float32)
    position = {node: i for i, node in enumerate(node_ids)}
    for node in present:
        raw[position[node]] = present_f0[present_position[node]]
    sequence = _standardize_f0(raw, feature_stats).to(device)

    u_idx = torch.tensor([position[u] for u, _ in pairs], dtype=torch.long)
    v_idx = torch.tensor([position[v] for _, v in pairs], dtype=torch.long)
    out: NDArray[np.float32] = np.empty(len(pairs), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            logits = model.pair_logits(
                sequence,
                u_idx[start:end].to(device),
                v_idx[start:end].to(device),
            )
            out[start:end] = logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
            _log_progress(end, len(pairs), end - start)
    return out


def _map_full_ego_graph_tensors(
    graph: FullEgoGraph, transform: Callable[[torch.Tensor], torch.Tensor]
) -> FullEgoGraph:
    """Transform graph tensors without discarding a concrete graph subtype."""
    return replace(
        graph,
        x=transform(graph.x),
        adj=transform(graph.adj),
        mask=transform(graph.mask),
        aux={key: transform(value) for key, value in graph.aux.items()},
    )


def _score_egostitch_e2e(
    model: nn.Module,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    *,
    device: torch.device,
    token_budget: int,
    f0_cache: Path,
    grounding_cache: Path | None = None,
    role_universe: str = "test",
    scaffold_control: str = _SCAFFOLD_CONTROL_NONE,
    universe_pairs: Sequence[tuple[str, str]] | None = None,
    row_start: int = 0,
    oracle_truth_graph: nx.Graph | None = None,
    full_oracle_ego_sizes: Sequence[int] | None = None,
    full_oracle_telemetry: _FullOracleScoreTelemetry | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Score pairs with an `EgoStitchModel`'s two-logit decomposition.

    Encodes each unique node exactly once, caches its raw-token/generator state
    on CPU, then builds one shared pair context per pair batch and applies the
    two hard-bypass heads to that context (content path removed rev 4, design
    doc §9). The complete cache and pair passes are pinned to fp32 by
    ``egostitch_e2e_pair_fp32_v1``; autocast is always disabled for this
    family, regardless of any future `--amp` wiring.

    Supports three registered generators (design 2026-08-02 §12 P3 and the
    2026-08-04 oracle-scaffold-experiment design): the real `egostitch_imagine`
    generator, whose per-node cache carries a `SlotSet`/projected features; the
    null generator (`generator.name: "null"`), whose per-node cache carries
    `slots=None`/`projected_x=None` because there is nothing to imagine; and
    `OracleStructGenerator` (`generator.name: "oracle_struct"`), which -- like
    the real generator -- caches a populated `SlotSet`/projected features, but
    reads them from `oracle_truth_graph` (a ground-truth table, installed here
    via `_install_oracle_context`) rather than imagining them. Every oracle
    call is therefore a truth-consuming ceiling diagnostic, never a formal
    result: its caller (`_run_score`) gates it behind an explicit operator
    opt-in and this function requires `oracle_truth_graph` whenever the
    generator is an `OracleStructGenerator`. For the null generator the
    returned ``full`` and ``f_logit`` arrays are identical -- the composite's
    `self.encoder is None` clamps topology off for both heads, so this is the
    correct null result (the topology pathway contributed nothing) rather than
    a special-cased single array. Any other generator raises `ValueError`.

    Each batch is assembled with real grounding-pool candidates (spec Sec
    13.12) via the `build_f0_matrix`/`build_grounding_pool` loader:
    ``ground_a``/``ground_b`` carry
    the pool's F0 features and ``ground_id_a``/``ground_id_b`` carry the
    pool's global node ids (matrix-row indices into the shared, run-scoped
    `node_ids` vocabulary — consistent across both endpoints and batches, so
    `EgoStitchModel`'s grounded-identity-match flag can compare them for
    equality). This always exercises the non-degenerate grounding path in
    `EgoStitchModel`'s pair-context construction; the placeholder
    `_ground`/zero-matched-flags path is reserved for tiny unit fixtures that
    omit these batch keys.

    Args:
        model: Frozen `EgoStitchModel`, on `device`, in `eval()` mode.
        pairs: Node-id pairs in input row order.
        store: Feature store providing per-node token sequences.
        device: Compute device.
        token_budget: Approximate per-batch token budget for the bucketed sampler.
        f0_cache: F0 matrix cache path providing the mean-pooled `x_a`/`x_b`
            generator inputs (f0_mlp semantics).
        grounding_cache: Grounding-pool cache path (derived from `f0_cache`
            when ``None``).
        role_universe: The scored universe's role identity, namespacing the
            grounding-pool cache so it matches the training-side pool: `"test"`
            for test/candidate/test_topology/file sources, `"V_val"` for
            val_topology/val_cls.
        scaffold_control: Optional registered within-pair scaffold perturbation.
        universe_pairs: Full input universe used to keep grounding pools stable
            when this scorer receives a contiguous shard.
        row_start: Global start row of `pairs` within `universe_pairs`.
        oracle_truth_graph: The self-loop-free ground-truth graph an
            `OracleStructGenerator` reads its scaffolds from
            (:func:`_oracle_truth_graph_for_scoring`). Required exactly when
            `model.generator` is an `OracleStructGenerator`; ignored otherwise.
        full_oracle_ego_sizes: Optional precomputed exact ego sizes aligned to
            `pairs`; used by cost-balanced full-oracle shards.
        full_oracle_telemetry: Optional mutable receiver for full-oracle batch
            composition and producer-wait measurements.

    Returns:
        Dict with keys ``full`` and ``f_logit``, each a shape ``(len(pairs),)``
        float32 array in input row order.

    Raises:
        ValueError: If `model.generator` is not a registered supported
            generator, or if it is an `OracleStructGenerator` and
            `oracle_truth_graph` is ``None``.
    """
    from src.data.grounding import build_grounding_pool
    from src.data.prefetch import _prefetch_batches
    from src.model.egostitch.composite import E2ENodeState, EgoStitchModel
    from src.model.egostitch.generator.egostitch import (
        EgoStitchImagineGenerator,
        GeneratorNodeState,
    )
    from src.model.egostitch.generator.full_oracle import (
        FullEgoFeaturesGenerator,
        FullEgoGraph,
        FullOracleGenerator,
    )
    from src.model.egostitch.generator.imagine import SlotSet
    from src.model.egostitch.generator.null import NullGenerator
    from src.model.egostitch.generator.oracle import OracleStructGenerator

    assert isinstance(model, EgoStitchModel)
    # `EgoStitchModel.decompose_pair_context` (composite.py) is well-defined
    # for every registered generator. For the real `egostitch_imagine`
    # generator, `full` and `f_logit` diverge once trained -- the ordinary
    # topology-conditioned decomposition. For `generator.name: "null"`
    # (design 2026-08-02 §12 P3), `self.encoder is None` unconditionally
    # clamps `need_topo` off inside `score_pair_context`, so *both* logits
    # run the classifier fully unconditioned and come out identical -- the
    # correct null result (the topology pathway contributed nothing, i.e.
    # this arm reproduces the pairwise baseline), not a vacuous one. Its
    # node state correspondingly has no `SlotSet`/`projected_x` to cache
    # (`encode_node_state` returns `slots=None`, `projected_x=None`); this
    # scorer carries that `None` through its per-node cache explicitly
    # (`is_slot_generator` below) instead of asserting it away.
    # `OracleStructGenerator` (2026-08-04 oracle-scaffold-experiment design)
    # produces a populated `SlotSet`/`projected_x` exactly like the real
    # generator -- it just reads them from a ground-truth table instead of
    # imagining them -- so it shares the real generator's caching branch.
    # Any other generator is genuinely unsupported here and must fail loudly
    # at this point, not with an opaque `NoneType` crash deep in the
    # node-state cache below.
    is_real_generator = isinstance(model.generator, EgoStitchImagineGenerator)
    is_null_generator = isinstance(model.generator, NullGenerator)
    is_oracle_generator = isinstance(model.generator, OracleStructGenerator)
    is_full_oracle_generator = isinstance(model.generator, FullOracleGenerator)
    is_truth_generator = is_oracle_generator or is_full_oracle_generator
    is_slot_generator = is_real_generator or is_truth_generator
    if not (is_slot_generator or is_null_generator):
        raise ValueError(
            "egostitch_e2e scoring supports only the real egostitch_imagine "
            "generator (topology-conditioned), the null generator "
            "(topology-free pairwise baseline), or an oracle diagnostic "
            f"generator; got {type(model.generator).__name__}"
        )
    if is_truth_generator and oracle_truth_graph is None:
        raise ValueError(
            "oracle scoring requires oracle_truth_graph (the ground-truth "
            "topology to install via _install_oracle_context); this is a caller bug, "
            "not a legal degraded mode"
        )
    _reject_superseded_scaffold_control(scaffold_control)
    active_controls = (
        _SCAFFOLD_CONTROL_SHUFFLE_V3,
        _SCAFFOLD_CONTROL_REWIRE_V1,
    )
    if scaffold_control not in (_SCAFFOLD_CONTROL_NONE, *active_controls):
        raise ValueError(f"unknown scaffold control: {scaffold_control!r}")
    if is_full_oracle_generator and scaffold_control != _SCAFFOLD_CONTROL_NONE:
        raise ValueError("full_ego_oracle does not support scoring-time scaffold controls")
    # Grounding candidates must come from the full scored universe, not this
    # process's shard, so control (and ordinary e2e) logits are shard-invariant.
    node_universe = universe_pairs if universe_pairs is not None else pairs
    if not 0 <= row_start <= row_start + len(pairs) <= len(node_universe):
        raise ValueError("e2e scoring rows are outside the declared pair universe")
    node_ids = sorted({node_id for pair in node_universe for node_id in pair})
    if is_truth_generator:
        assert oracle_truth_graph is not None  # checked above
        _install_oracle_context(model, node_ids, truth_graph=oracle_truth_graph)
    f0_cache.parent.mkdir(parents=True, exist_ok=True)
    matrix, index = build_f0_matrix(store, node_ids, cache_path=f0_cache)
    if isinstance(model.generator, FullEgoFeaturesGenerator):
        # Candidate egos routinely reach truth-graph nodes outside the scored
        # pair universe, so the student's feature table must cover the whole
        # truth graph's featured nodes, not just `node_ids`. Featureless
        # truth-graph nodes stay out and gather zeros with has_f0=0 inside
        # the generator. Cached beside the pair-universe F0 cache under its
        # own name -- `build_f0_matrix`'s exact-order contract would reject
        # sharing one file between the two node orderings.
        assert oracle_truth_graph is not None
        featured_ids = sorted((set(oracle_truth_graph.nodes) | set(node_ids)) & store.node_ids)
        ego_matrix, _ego_index = build_f0_matrix(
            store,
            featured_ids,
            cache_path=f0_cache.with_name(f"{f0_cache.stem}_ego_universe.pt"),
        )
        model.generator.set_node_features(ego_matrix, featured_ids)

    registered_n_ground = model.generator_cfg.n_ground
    if grounding_cache is None:
        grounding_cache = _default_grounding_cache_path(
            f0_cache,
            n_ground=registered_n_ground,
            node_ids=node_ids,
            role_universe=role_universe,
        )
    # The e2e generator's own-split n_ground default (spec Sec 13) assumes a
    # real candidate-universe scale (thousands of nodes); clamp to what this
    # call's node set can actually support so tiny fixtures (few endpoints)
    # remain valid `build_grounding_pool` calls without changing production
    # behavior, where len(node_ids) - 1 always exceeds the spec default.
    n_ground = min(registered_n_ground, len(node_ids) - 1)
    if n_ground < registered_n_ground:
        logger.warning(
            "n_ground clamped from registered %d to %d for this scoring call's "
            "universe of %d node(s); effective grounding-pool size is smaller "
            "than the registered default (expected only for small universes, "
            "not production scoring runs)",
            registered_n_ground,
            n_ground,
            len(node_ids),
        )
    pool = build_grounding_pool(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_ids,
        n_ground=n_ground,
        role_universe=role_universe,
        cache_path=grounding_cache,
    )
    pool_rows = torch.tensor(
        [[index[neighbor] for neighbor in pool[node]] for node in node_ids],
        dtype=torch.int64,
    )

    # Probe lengths once, then run the expensive raw-token encoder and
    # imagination generator exactly once per unique node. Cache rows on CPU so
    # the production candidate universe does not retain every token state on
    # device; pair batches transfer only their endpoint rows back to `device`.
    node_lengths = {
        node_id: length[0]
        for node_id, length in zip(
            node_ids,
            probe_lengths(store, [(node_id, node_id) for node_id in node_ids]),
            strict=True,
        )
    }
    buckets: dict[int, list[str]] = {}
    for node_id in node_ids:
        length = node_lengths[node_id]
        boundary = next((value for value in BUCKET_BOUNDARIES if length <= value), length)
        buckets.setdefault(boundary, []).append(node_id)

    node_cache: dict[str, E2ENodeState] = {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for boundary in sorted(buckets):
            batch_size = max(1, token_budget // max(1, boundary))
            bucket_nodes = buckets[boundary]
            for start in range(0, len(bucket_nodes), batch_size):
                batch_nodes = bucket_nodes[start : start + batch_size]
                tokens = [store.load_tokens(node_id) for node_id in batch_nodes]
                lengths = torch.tensor(
                    [token.size(0) for token in tokens], dtype=torch.int64, device=device
                )
                embeddings = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(device)
                rows = torch.tensor([index[node_id] for node_id in batch_nodes], dtype=torch.long)
                state = model.encode_node_state(
                    embeddings,
                    lengths,
                    matrix.index_select(0, rows).to(device),
                    matrix[pool_rows[rows]].to(device),
                    pool_rows[rows].to(device),
                    rows.to(device),
                )
                if is_slot_generator:
                    # A real `egostitch_imagine` or `oracle_struct` generator's
                    # `encode_node` always returns a `GeneratorNodeState`
                    # (`EgoStitchModel.encode_node_state`), so `slots`/
                    # `projected_x` are always populated here. `None` would
                    # mean a broken generator/checkpoint pairing, not the
                    # (separately handled) null-generator arm -- this must
                    # still fail loudly rather than caching a silent `None`.
                    assert state.slots is not None and state.projected_x is not None, (
                        "slot-producing generator produced no slot state for "
                        "this node batch -- this is a bug, not the supported "
                        "null-generator arm"
                    )
                else:
                    assert state.slots is None and state.projected_x is None, (
                        "null generator unexpectedly produced slot state -- "
                        "encode_node_state should never populate it for NullGenerator"
                    )
                for position, node_id in enumerate(batch_nodes):
                    true_length = int(lengths[position].item())
                    ground_ids = (
                        None
                        if state.ground_ids is None
                        else state.ground_ids[position : position + 1].detach().cpu()
                    )
                    slots = (
                        SlotSet(
                            *(
                                value[position : position + 1].detach().float().cpu()
                                for value in state.slots
                            )
                        )
                        if state.slots is not None
                        else None
                    )
                    projected_x = (
                        state.projected_x[position : position + 1].detach().float().cpu()
                        if state.projected_x is not None
                        else None
                    )
                    node_cache[node_id] = E2ENodeState(
                        encoded=state.encoded[position : position + 1, :true_length]
                        .detach()
                        .float()
                        .cpu(),
                        length=state.length[position : position + 1].detach().cpu(),
                        slots=slots,
                        projected_x=projected_x,
                        ground_ids=ground_ids,
                    )

    full_oracle_encoded_table: torch.Tensor | None = None
    full_oracle_length_table: torch.Tensor | None = None
    if is_full_oracle_generator and device.type == "cuda" and node_ids:
        max_length = max(int(node_cache[node_id].encoded.shape[1]) for node_id in node_ids)
        encoded_dim = int(node_cache[node_ids[0]].encoded.shape[2])
        encoded_host = torch.zeros(
            (len(node_ids), max_length, encoded_dim),
            dtype=torch.float32,
            pin_memory=True,
        )
        for row, node_id in enumerate(node_ids):
            encoded = node_cache[node_id].encoded.squeeze(0)
            encoded_host[row, : encoded.shape[0]].copy_(encoded)
        length_host = torch.cat(
            [node_cache[node_id].length for node_id in node_ids], dim=0
        ).pin_memory()
        full_oracle_encoded_table = encoded_host.to(device=device, non_blocking=True)
        full_oracle_length_table = length_host.to(device=device, non_blocking=True)
        torch.cuda.synchronize(device)

    def _stack_cached_cpu(nodes: Sequence[str]) -> E2ENodeState:
        states = [node_cache[node_id] for node_id in nodes]
        encoded = torch.nn.utils.rnn.pad_sequence(
            [state.encoded.squeeze(0) for state in states], batch_first=True
        )
        ground_ids: torch.Tensor | None
        if all(state.ground_ids is not None for state in states):
            ground_ids = torch.cat(
                [cast(torch.Tensor, state.ground_ids) for state in states], dim=0
            )
        else:
            ground_ids = None
        # `node_cache` is populated homogeneously by this call's own
        # `is_slot_generator` branch above: every cached state's `slots`/
        # `projected_x` is populated together (real or oracle generator) or
        # `None` together (null generator), never mixed within one scoring call.
        slots: SlotSet | None
        projected_x: torch.Tensor | None
        if is_slot_generator:
            assert all(state.slots is not None for state in states)
            slots = SlotSet(
                *(
                    torch.cat(values, dim=0)
                    for values in zip(
                        *(cast(SlotSet, state.slots) for state in states), strict=True
                    )
                )
            )
            assert all(state.projected_x is not None for state in states)
            projected_x = torch.cat(
                [cast(torch.Tensor, state.projected_x) for state in states], dim=0
            )
        else:
            assert all(state.slots is None for state in states)
            assert all(state.projected_x is None for state in states)
            slots = None
            projected_x = None
        return E2ENodeState(
            encoded=encoded,
            length=torch.cat([state.length for state in states], dim=0),
            slots=slots,
            projected_x=projected_x,
            ground_ids=ground_ids,
        )

    def _stack_full_oracle_generator_state(nodes: Sequence[str]) -> GeneratorNodeState:
        """Stack only the tiny row-identity state consumed by the oracle stitcher."""
        states = [node_cache[node_id] for node_id in nodes]
        assert all(state.slots is not None and state.projected_x is not None for state in states)
        return GeneratorNodeState(
            slots=SlotSet(
                *(
                    torch.cat(values, dim=0)
                    for values in zip(
                        *(cast(SlotSet, state.slots) for state in states), strict=True
                    )
                )
            ),
            projected_x=torch.cat(
                [cast(torch.Tensor, state.projected_x) for state in states], dim=0
            ),
            ground_ids=None,
        )

    def _gather_full_oracle_device_state(
        node_rows: torch.Tensor, *, encoded_width: int
    ) -> E2ENodeState:
        """Gather endpoint states from the one-time GPU-resident padded table."""
        assert full_oracle_encoded_table is not None
        assert full_oracle_length_table is not None
        encoded, lengths = _gather_padded_node_rows(
            full_oracle_encoded_table,
            full_oracle_length_table,
            node_rows,
            width=encoded_width,
        )
        return E2ENodeState(
            encoded=encoded,
            length=lengths,
            slots=None,
            projected_x=None,
            ground_ids=None,
        )

    def _move_cached(state: E2ENodeState, *, non_blocking: bool = False) -> E2ENodeState:
        slots = (
            SlotSet(*(value.to(device=device, non_blocking=non_blocking) for value in state.slots))
            if state.slots is not None
            else None
        )
        return E2ENodeState(
            encoded=state.encoded.to(device=device, non_blocking=non_blocking),
            length=state.length.to(device=device, non_blocking=non_blocking),
            slots=slots,
            projected_x=(
                state.projected_x.to(device=device, non_blocking=non_blocking)
                if state.projected_x is not None
                else None
            ),
            ground_ids=(
                state.ground_ids.to(device=device, non_blocking=non_blocking)
                if state.ground_ids is not None
                else None
            ),
        )

    def _pin_graph(graph: FullEgoGraph) -> FullEgoGraph:
        return _map_full_ego_graph_tensors(graph, torch.Tensor.pin_memory)

    def _move_graph(graph: FullEgoGraph, *, non_blocking: bool) -> FullEgoGraph:
        return _map_full_ego_graph_tensors(
            graph,
            lambda value: value.to(device=device, non_blocking=non_blocking),
        )

    out: dict[str, NDArray[np.float32]] = {
        key: np.empty(len(pairs), dtype=np.float32) for key in _EGOSTITCH_E2E_ARRAY_KEYS
    }
    processed = 0
    if scaffold_control in active_controls:
        # Every shard reconstructs identical global, fixed-size padded blocks
        # and retains only its own rows. This keeps bit-exact shard invariance
        # without one pair-pass per row; a shard recomputes only boundary blocks.
        all_lengths = probe_lengths(store, node_universe)
        max_length = max((max(length) for length in all_lengths), default=1)
        block_size = min(len(node_universe), max(1, token_budget // (2 * max_length)))
        row_end = row_start + len(pairs)
        batch_specs: list[tuple[list[int], list[tuple[int, int]]]] = []
        control_order = sorted(
            range(len(node_universe)),
            key=lambda row: canonical_pair(*node_universe[row]),
        )
        for block_start in range(0, len(node_universe), block_size):
            indices = control_order[block_start : block_start + block_size]
            block_original_rows = set(indices)
            if not any(row_start <= row < row_end for row in block_original_rows):
                continue
            output_rows = [
                (global_row - row_start, position)
                for position, global_row in enumerate(indices)
                if row_start <= global_row < row_end
            ]
            indices.extend([indices[0]] * (block_size - len(indices)))
            batch_specs.append((indices, output_rows))
        batch_pair_source = node_universe
    else:
        if is_full_oracle_generator:
            assert oracle_truth_graph is not None
            ego_sizes = (
                list(full_oracle_ego_sizes)
                if full_oracle_ego_sizes is not None
                else _full_oracle_ego_sizes(oracle_truth_graph, pairs)
            )
            if len(ego_sizes) != len(pairs):
                raise ValueError("full_oracle_ego_sizes must align with scored pairs")
            pair_token_lengths = [
                2 * max(node_lengths[node_u], node_lengths[node_v]) for node_u, node_v in pairs
            ]
            batch_specs = _full_oracle_score_batch_specs(
                ego_sizes,
                pair_token_lengths,
                token_budget=token_budget,
                cell_budget=_FULL_ORACLE_CELL_BUDGET,
                max_batch_pairs=_FULL_ORACLE_MAX_BATCH_PAIRS,
            )
            if full_oracle_telemetry is not None:
                full_oracle_telemetry.ego_sizes = ego_sizes
                full_oracle_telemetry.batch_sizes = [len(indices) for indices, _ in batch_specs]
                full_oracle_telemetry.batch_max_ego_sizes = [
                    max(ego_sizes[index] for index in indices) for indices, _ in batch_specs
                ]
        else:
            pair_lengths = probe_lengths(store, pairs)
            sampler = LengthBucketedBatchSampler(
                pair_lengths, token_budget=token_budget, shuffle=False
            )
            batch_specs = [
                (indices, [(index, position) for position, index in enumerate(indices)])
                for indices in sampler
            ]
        batch_pair_source = pairs

    def _store_decomposed(
        decomposed: Mapping[str, torch.Tensor], output_rows: Sequence[tuple[int, int]]
    ) -> None:
        output_indices = np.fromiter((output_row for output_row, _ in output_rows), dtype=np.int64)
        batch_positions = np.fromiter(
            (batch_position for _, batch_position in output_rows), dtype=np.int64
        )
        for key in _EGOSTITCH_E2E_ARRAY_KEYS:
            values = decomposed[key].detach().to(dtype=torch.float32, device="cpu").numpy()
            out[key][output_indices] = values[batch_positions]

    if is_full_oracle_generator:

        def _prepare_full_oracle_batches() -> Iterator[_PreparedFullOracleBatch]:
            for batch_indices, output_rows in batch_specs:
                batch_pairs = [batch_pair_source[row] for row in batch_indices]
                nodes_a = [pair[0] for pair in batch_pairs]
                nodes_b = [pair[1] for pair in batch_pairs]
                encoded_width = max(
                    max(node_lengths[node_id] for node_id in nodes_a),
                    max(node_lengths[node_id] for node_id in nodes_b),
                )
                node_rows_a = torch.tensor([index[node_id] for node_id in nodes_a])
                node_rows_b = torch.tensor([index[node_id] for node_id in nodes_b])
                is_self = torch.tensor(
                    [node_u == node_v for node_u, node_v in batch_pairs], dtype=torch.bool
                )
                graph = model.generator.stitch(
                    _stack_full_oracle_generator_state(nodes_a),
                    _stack_full_oracle_generator_state(nodes_b),
                    is_self,
                )
                if not isinstance(graph, FullEgoGraph):
                    raise TypeError("full oracle generator returned a non-full-ego graph")
                state_a: E2ENodeState | None = None
                state_b: E2ENodeState | None = None
                if device.type == "cuda":
                    node_rows_a = node_rows_a.pin_memory()
                    node_rows_b = node_rows_b.pin_memory()
                    is_self = is_self.pin_memory()
                    graph = _pin_graph(graph)
                else:
                    state_a = _stack_cached_cpu(nodes_a)
                    state_b = _stack_cached_cpu(nodes_b)
                yield _PreparedFullOracleBatch(
                    output_rows=output_rows,
                    node_rows_a=node_rows_a,
                    node_rows_b=node_rows_b,
                    encoded_width=encoded_width,
                    state_a=state_a,
                    state_b=state_b,
                    is_self=is_self,
                    graph=graph,
                )

        prepared_batches = _prefetch_batches(
            iter(_prepare_full_oracle_batches()), depth=_FULL_ORACLE_PREFETCH_DEPTH
        )
        try:
            while True:
                wait_started = perf_counter()
                try:
                    prepared = next(prepared_batches)
                except StopIteration:
                    break
                if full_oracle_telemetry is not None:
                    full_oracle_telemetry.producer_wait_seconds += perf_counter() - wait_started
                non_blocking = device.type == "cuda"
                if device.type == "cuda":
                    state_a = _gather_full_oracle_device_state(
                        prepared.node_rows_a, encoded_width=prepared.encoded_width
                    )
                    state_b = _gather_full_oracle_device_state(
                        prepared.node_rows_b, encoded_width=prepared.encoded_width
                    )
                else:
                    assert prepared.state_a is not None and prepared.state_b is not None
                    state_a = _move_cached(prepared.state_a)
                    state_b = _move_cached(prepared.state_b)
                is_self = prepared.is_self.to(device=device, non_blocking=non_blocking)
                graph = _move_graph(prepared.graph, non_blocking=non_blocking)
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    context = model.build_pair_context_from_states(
                        state_a,
                        state_b,
                        is_self,
                        precomputed_graph=graph,
                    )
                    decomposed = model.decompose_pair_context(context)
                _store_decomposed(decomposed, prepared.output_rows)
                processed += len(prepared.output_rows)
                _log_progress(processed, len(pairs), len(prepared.output_rows))
        finally:
            prepared_batches.close()
    else:
        for batch_indices, output_rows in batch_specs:
            batch_pairs = [batch_pair_source[row] for row in batch_indices]
            state_a = _move_cached(_stack_cached_cpu([pair[0] for pair in batch_pairs]))
            state_b = _move_cached(_stack_cached_cpu([pair[1] for pair in batch_pairs]))
            is_self = torch.tensor(
                [node_u == node_v for node_u, node_v in batch_pairs],
                dtype=torch.bool,
                device=device,
            )
            perturbation = (
                None
                if scaffold_control == _SCAFFOLD_CONTROL_NONE
                else make_scaffold_input_perturbation(scaffold_control, batch_pairs)
            )
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                context = model.build_pair_context_from_states(
                    state_a,
                    state_b,
                    is_self,
                    scaffold_input_perturbation=perturbation,
                )
                decomposed = model.decompose_pair_context(context)
            _store_decomposed(decomposed, output_rows)
            processed += len(output_rows)
            _log_progress(processed, len(pairs), len(output_rows))
    return out


def _full_oracle_ego_sizes(
    truth_graph: nx.Graph[str], pairs: Sequence[tuple[str, str]]
) -> list[int]:
    """Return exact query-graph node counts via packed neighbor intersections."""
    graph_nodes = tuple(sorted(truth_graph.nodes))
    graph_row = {node_id: row for row, node_id in enumerate(graph_nodes)}
    try:
        src_rows = np.fromiter(
            (graph_row[node_u] for node_u, _ in pairs), dtype=np.int64, count=len(pairs)
        )
        dst_rows = np.fromiter(
            (graph_row[node_v] for _, node_v in pairs), dtype=np.int64, count=len(pairs)
        )
    except KeyError as exc:
        raise ValueError(
            f"full-oracle pair endpoint is absent from the truth graph: {exc.args[0]!r}"
        ) from exc

    bytes_per_row = (len(graph_nodes) + 7) // 8
    packed_neighbors = np.zeros((len(graph_nodes), bytes_per_row), dtype=np.uint8)
    edges = np.asarray(
        [(graph_row[node_u], graph_row[node_v]) for node_u, node_v in truth_graph.edges],
        dtype=np.int64,
    ).reshape(-1, 2)
    if len(edges):
        for left, right in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            bit_values = np.left_shift(np.uint8(1), (right & 7).astype(np.uint8, copy=False))
            np.bitwise_or.at(packed_neighbors, (left, right >> 3), bit_values)

    degrees = np.bitwise_count(packed_neighbors).sum(axis=1, dtype=np.int64)
    sizes = np.empty(len(pairs), dtype=np.int64)
    target_chunk_bytes = 32 * 1024 * 1024
    chunk_rows = max(1, target_chunk_bytes // max(bytes_per_row, 1))
    for start in range(0, len(pairs), chunk_rows):
        end = min(len(pairs), start + chunk_rows)
        src = src_rows[start:end]
        dst = dst_rows[start:end]
        common = np.bitwise_count(packed_neighbors[src] & packed_neighbors[dst]).sum(
            axis=1, dtype=np.int64
        )
        adjacent = (
            packed_neighbors[src, dst >> 3]
            & np.left_shift(np.uint8(1), (dst & 7).astype(np.uint8, copy=False))
        ) != 0
        self_pair = src == dst
        union_size = degrees[src] + degrees[dst] - common
        sizes[start:end] = union_size + self_pair.astype(np.int64) + 2 * (~self_pair & ~adjacent)
    return sizes.tolist()


def _full_oracle_score_batch_specs(
    ego_sizes: Sequence[int],
    token_lengths: Sequence[int],
    *,
    token_budget: int,
    cell_budget: int = _FULL_ORACLE_CELL_BUDGET,
    max_batch_pairs: int = _FULL_ORACLE_MAX_BATCH_PAIRS,
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Greedily bucket full-ego pairs under dense-cell and token budgets."""
    if len(ego_sizes) != len(token_lengths):
        raise ValueError("ego_sizes and token_lengths must have equal length")
    if token_budget <= 0 or cell_budget <= 0 or max_batch_pairs <= 0:
        raise ValueError("full-oracle batch budgets must be positive")
    if any(size <= 0 for size in ego_sizes):
        raise ValueError("full-oracle ego sizes must be positive")
    if any(length <= 0 for length in token_lengths):
        raise ValueError("full-oracle token lengths must be positive")

    order = sorted(range(len(ego_sizes)), key=lambda index: (-ego_sizes[index], index))
    batches: list[tuple[list[int], list[tuple[int, int]]]] = []
    current: list[int] = []
    current_n_max = 0
    current_t_max = 0

    def finish_current() -> None:
        if current:
            indices = list(current)
            batches.append(
                (indices, [(output_row, position) for position, output_row in enumerate(indices)])
            )

    for index in order:
        if not current:
            current = [index]
            current_n_max = ego_sizes[index]
            current_t_max = token_lengths[index]
            continue
        candidate_count = len(current) + 1
        candidate_t_max = max(current_t_max, token_lengths[index])
        if (
            candidate_count > max_batch_pairs
            or candidate_count * current_n_max * current_n_max > cell_budget
            or candidate_count * candidate_t_max > token_budget
        ):
            finish_current()
            current = [index]
            current_n_max = ego_sizes[index]
            current_t_max = token_lengths[index]
        else:
            current.append(index)
            current_t_max = candidate_t_max
    finish_current()
    return batches


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


def _balanced_shard_range(costs: Sequence[int], shard: int, num_shards: int) -> tuple[int, int]:
    """Split ordered rows at deterministic cumulative-cost quantiles."""
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid shard index/count")
    values = np.asarray(costs, dtype=np.int64)
    if values.ndim != 1 or bool((values < 0).any()):
        raise ValueError("shard costs must be a one-dimensional non-negative sequence")
    if not len(values) or not int(values.sum()):
        return _shard_range(len(values), shard, num_shards)
    cumulative = np.cumsum(values, dtype=np.int64)
    total = int(cumulative[-1])

    def boundary(part: int) -> int:
        if part == 0:
            return 0
        if part == num_shards:
            return len(values)
        target = total * part / num_shards
        return int(np.searchsorted(cumulative, target, side="right"))

    return boundary(shard), boundary(shard + 1)


def _full_oracle_high_degree_summary(
    truth_graph: nx.Graph[str],
    pairs: Sequence[tuple[str, str]],
    ego_sizes: Sequence[int],
) -> dict[str, object]:
    """Summarize rows whose query-removed endpoint degree exceeds 16."""
    selected_sizes: list[int] = []
    for (node_u, node_v), ego_size in zip(pairs, ego_sizes, strict=True):
        query_present = node_u != node_v and truth_graph.has_edge(node_u, node_v)
        degree_u = int(truth_graph.degree(node_u)) - int(query_present)
        degree_v = int(truth_graph.degree(node_v)) - int(query_present)
        if max(degree_u, degree_v) > 16:
            selected_sizes.append(ego_size)
    return {
        "count": len(selected_sizes),
        "fraction": len(selected_sizes) / len(pairs) if pairs else 0.0,
        "ego_size": (_full_oracle_distribution(selected_sizes) if selected_sizes else None),
    }


def _full_oracle_distribution(values: Sequence[int]) -> dict[str, float | int]:
    """Reuse the diagnostic's pinned |U_q| distribution implementation."""
    from src.experiments.full_ego_oracle.telemetry import _distribution

    if not values:
        raise ValueError("full-oracle distribution requires at least one value")
    return _distribution(values)


def _histogram(values: Sequence[int]) -> dict[str, int]:
    """Return a stable JSON histogram with numeric keys rendered as strings."""
    counts = Counter(values)
    return {str(value): counts[value] for value in sorted(counts)}


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
        "--run-metadata",
        type=Path,
        default=None,
        help=(
            "run metadata (for its arm/seed) required before egostitch_e2e "
            "candidate/test/file held-out scoring; identifies the test-access ledger"
        ),
    )
    score.add_argument(
        "--model-family",
        choices=sorted(MODEL_BUILDERS),
        help="model family for a bare legacy state_dict checkpoint",
    )
    score.add_argument(
        "--model-config",
        type=Path,
        help=(
            "training YAML containing model.config for a bare legacy checkpoint "
            "(for cazi_mbn: its own training YAML, used only for feature "
            "standardization and its known feature gaps -- architecture is "
            "inferred from the checkpoint itself)"
        ),
    )
    score.add_argument(
        "--pairs",
        required=True,
        help=(
            "candidate | test | test_topology | val_topology | val_cls | "
            "file:<path.tsv> (TSV rows: u\\tv[\\tlabel])"
        ),
    )
    score.add_argument("--data-root", type=Path, default=Path("data"))
    score.add_argument("--strategy", default="breadth_first")
    score.add_argument("--output", type=Path, required=True, help="output .npz path")
    score.add_argument(
        "--batch-pairs", type=int, default=8192, help="f0_mlp/cazi_mbn rows per batch"
    )
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
        "--pair-amp",
        choices=["off", "bf16"],
        default=None,
        help="optional pair-head autocast override for packed v3_1 scoring",
    )
    score.add_argument(
        "--grounding-cache",
        type=Path,
        default=None,
        help="grounding-pool cache path (egostitch_e2e only; built when absent)",
    )
    score.add_argument(
        "--scaffold-control",
        choices=[
            _SCAFFOLD_CONTROL_NONE,
            _SCAFFOLD_CONTROL_SHUFFLE_V3,
            _SCAFFOLD_CONTROL_REWIRE_V1,
            _SCAFFOLD_CONTROL_SHUFFLE_V2,
        ],
        default=_SCAFFOLD_CONTROL_NONE,
        help="egostitch_e2e scoring-time scaffold control",
    )
    score.add_argument(
        "--topo-gen-control",
        choices=CONTROLS,
        default=None,
        help="v3_1 topology-generator scoring-time control",
    )
    score.add_argument("--shard", type=int, default=None, help="shard index K (with --num-shards)")
    score.add_argument("--num-shards", type=int, default=None, help="total shard count N")
    score.add_argument(
        "--rescore-reason",
        default=None,
        help="required reason for a repeated egostitch_e2e held-out scoring epoch",
    )
    score.add_argument(
        "--scoring-run-id",
        default=None,
        help=(
            "shares one held-out scoring epoch across multiple --pairs sources "
            "scored for the same arm/seed (e.g. test then candidate, one "
            "test-protocol invocation); egostitch_e2e held-out scoring only"
        ),
    )
    score.add_argument(
        "--allow-oracle-diagnostic",
        action="store_true",
        help=(
            "required to score a checkpoint whose egostitch_e2e generator is "
            "oracle_struct or full_ego_oracle: it consumes ground-truth topology "
            "by construction "
            "(the 2026-08-04 oracle-scaffold-experiment design), so this is a "
            "ceiling diagnostic only, never a formal result, and is refused "
            "without this explicit acknowledgement"
        ),
    )
    score.add_argument(
        "--f0-cache",
        type=Path,
        default=Path("outputs/f0_cache/f0_matrix.pt"),
        help="F0 matrix cache path (f0_mlp/egostitch_e2e only; cazi_mbn never caches)",
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
    if args.run_metadata is not None and not args.run_metadata.is_file():
        parser.error(f"run metadata not found: {args.run_metadata}")
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
    if args.rescore_reason is not None and not args.rescore_reason.strip():
        parser.error("--rescore-reason must contain non-whitespace text")
    if args.scoring_run_id is not None and not args.scoring_run_id.strip():
        parser.error("--scoring-run-id must contain non-whitespace text")


def _resolve_cazi_context(args: argparse.Namespace) -> tuple[CAZIConfig, FeatureStats]:
    """Load the CAZI training config and the statistics of the run being scored.

    The statistics come from the checkpoint's own directory, not from the YAML's
    ``output_dir``. ``src.train_cazi_mbn`` supports an ``--output-dir`` override
    that writes ``student.pt`` and ``feature_stats.npz`` together into the
    override, so reading the YAML path would either miss the file or silently
    standardize with a different run's statistics — wrong scores that raise
    nothing.

    Args:
        args: Parsed ``score`` arguments carrying ``checkpoint`` and ``model_config``.

    Returns:
        The CAZI config and the feature statistics beside the checkpoint.

    Raises:
        ValueError: If ``--model-config`` is absent, or no ``feature_stats.npz``
            sits beside the checkpoint.
    """
    if args.model_config is None:
        raise ValueError("cazi_mbn scoring requires --model-config (its own training YAML)")
    from src.train_cazi_mbn import load_config as load_cazi_config

    cazi_cfg = load_cazi_config(args.model_config)
    stats_path = args.checkpoint.parent / "feature_stats.npz"
    if not stats_path.is_file():
        raise ValueError(
            f"cazi_mbn scoring requires {stats_path} beside the checkpoint; "
            "score the run that produced this checkpoint"
        )
    return cazi_cfg, load_feature_stats(stats_path)


def _run_score(args: argparse.Namespace) -> None:
    """Execute the ``score`` subcommand."""
    _reject_superseded_scaffold_control(args.scaffold_control)
    device = _resolve_device(args.device)
    logger.info("loading checkpoint %s (device %s)", args.checkpoint, device)
    model_config = (
        _load_model_config(args.model_config, args.model_family)
        if args.model_config is not None and args.model_family != "cazi_mbn"
        else None
    )
    model, model_family, checkpoint_id = _load_checkpoint(
        args.checkpoint,
        model_family=args.model_family,
        model_config=model_config,
    )
    if args.topo_gen_control is not None:
        topo_gen = getattr(model, "topo_gen", None)
        if topo_gen is None:
            raise SystemExit("--topo-gen-control requires a checkpoint with model.config.topo_gen")
        cast(TopoGenBase, topo_gen).control = args.topo_gen_control

    cazi_context = _resolve_cazi_context(args) if model_family == "cazi_mbn" else None

    # Import guarded by family, not unconditional: `src.model.egostitch.composite`
    # transitively pulls in the `grit_gmt` encoder (and, through it,
    # torch_geometric), a real cost `v3_1`/`f0_mlp`/`cazi_mbn` scoring must
    # never pay. For `egostitch_e2e` this is free -- `_load_checkpoint` already
    # imported the same module building the model above.
    is_oracle_generator = False
    is_full_ego_family = False
    oracle_generator_name: str | None = None
    if model_family == "egostitch_e2e":
        from src.model.egostitch.composite import EgoStitchModel
        from src.model.egostitch.generator.full_oracle import FullOracleGenerator
        from src.model.egostitch.generator.oracle import OracleStructGenerator

        if isinstance(model, EgoStitchModel) and isinstance(
            model.generator, (OracleStructGenerator, FullOracleGenerator)
        ):
            is_oracle_generator = True
            # Covers `full_ego_oracle` and its `full_ego_features` subclass:
            # both emit unbounded per-pair ego node sets, so both need the
            # ego-size-aware shard balancing, batching, and telemetry below.
            is_full_ego_family = isinstance(model.generator, FullOracleGenerator)
            oracle_generator_name = model.cfg.generator.name
    if is_full_ego_family and args.scaffold_control != _SCAFFOLD_CONTROL_NONE:
        raise ValueError(f"{oracle_generator_name} does not support scoring-time scaffold controls")
    if args.allow_oracle_diagnostic and not is_oracle_generator:
        raise ValueError(
            "--allow-oracle-diagnostic is valid only when the checkpoint's "
            "egostitch_e2e generator is oracle_struct or full_ego_oracle"
        )
    oracle_truth_graph: nx.Graph | None = None
    if is_oracle_generator:
        # Airtight by construction, not by trusting the checkpoint's own
        # training-time run_metadata: the 2026-08-04 oracle-scaffold-experiment
        # design's own §0 correction records a `run_kind: formal` checkpoint
        # whose generator was already `oracle_struct` (the pre-correction R1
        # run), so a checkpoint's training provenance cannot be relied on to
        # gate this. What is always true is that *scoring* test/candidate/
        # V_val pairs through an oracle generator means feeding it the true
        # topology for the test/candidate/test_topology/V_val query -- a
        # truth-consuming
        # ceiling diagnostic every time, regardless of how the checkpoint was
        # trained. Requiring an explicit, scoring-time operator flag (rather
        # than an automatic run_kind check) is what keeps this unreachable
        # from any automated formal test-protocol invocation that does not
        # itself pass it.
        if not args.allow_oracle_diagnostic:
            raise ValueError(
                f"checkpoint generator is {oracle_generator_name}, which consumes ground-truth "
                "topology by construction (2026-08-04 oracle-scaffold-experiment design); "
                "pass --allow-oracle-diagnostic to acknowledge this is a ceiling "
                "diagnostic, never a formal result"
            )
        oracle_truth_graph = _oracle_truth_graph_for_scoring(
            args.pairs, args.data_root, args.strategy
        )

    heldout_e2e = model_family == "egostitch_e2e" and (
        args.pairs in {"candidate", "test", "test_topology"} or args.pairs.startswith("file:")
    )
    # Every E2E `file:` source is conservatively held out. Content-equality
    # detection would itself read the path before ledger recording and would
    # miss test subsets/supersets.
    scoring_arm: str | None = None
    scoring_seed: int | None = None
    if heldout_e2e:
        if args.run_metadata is None:
            raise ValueError(
                "egostitch_e2e candidate/test or file scoring requires --run-metadata "
                "before pair access"
            )
        run_metadata_for_ledger = _load_json_object(args.run_metadata, label="run metadata")
        arm = run_metadata_for_ledger.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError("run metadata is missing arm")
        scoring_arm = arm
        if args.scaffold_control != _SCAFFOLD_CONTROL_NONE:
            # Both mandatory scaffold-structure controls score against the
            # `full` checkpoint and pass `full`'s run metadata, so without
            # this remap they would ledger under arm `full` too and collide
            # with the ordinary full-arm score.
            control_arm = _SCAFFOLD_CONTROL_ARM_NAMES.get(args.scaffold_control)
            if control_arm is None:
                raise ValueError(f"unknown scaffold control: {args.scaffold_control!r}")
            scoring_arm = control_arm
        seed = run_metadata_for_ledger.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("run metadata seed must be an integer")
        scoring_seed = seed
    if args.rescore_reason is not None and not heldout_e2e:
        raise ValueError("--rescore-reason is valid only for egostitch_e2e held-out scoring")
    if args.scoring_run_id is not None and not heldout_e2e:
        raise ValueError("--scoring-run-id is valid only for egostitch_e2e held-out scoring")
    if args.scaffold_control != _SCAFFOLD_CONTROL_NONE and model_family != "egostitch_e2e":
        raise ValueError("--scaffold-control is supported only for model_family 'egostitch_e2e'")
    model.to(device)

    test_access = None
    if heldout_e2e:
        assert scoring_arm is not None
        assert scoring_seed is not None
        test_access = _TestAccessContext(
            ledger_path=args.run_metadata.parent / _TEST_ACCESS_LEDGER_FILENAME,
            scoring_arm=scoring_arm,
            seed=scoring_seed,
            output=args.output,
            shard=args.shard if args.shard is not None else 0,
            num_shards=args.num_shards if args.num_shards is not None else 1,
            rescore_reason=args.rescore_reason,
            scoring_run_id=args.scoring_run_id,
        )
    pairs, labels = _resolve_pairs(
        args.pairs, args.data_root, args.strategy, test_access=test_access
    )
    total_rows = len(pairs)
    logger.info("resolved %d pairs from %s", total_rows, args.pairs)
    all_full_oracle_ego_sizes: list[int] | None = None
    if is_full_ego_family:
        assert oracle_truth_graph is not None
        all_full_oracle_ego_sizes = _full_oracle_ego_sizes(oracle_truth_graph, pairs)

    if args.shard is not None:
        if all_full_oracle_ego_sizes is not None:
            costs = [
                ego_size * ego_size + _FULL_ORACLE_SHARD_COST_FLOOR
                for ego_size in all_full_oracle_ego_sizes
            ]
            start, end = _balanced_shard_range(costs, args.shard, args.num_shards)
        else:
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
            "pair_autocast": args.pair_amp or args.amp,
            "logit_storage_dtype": "float32",
        },
        "topo_gen_control": args.topo_gen_control,
    }
    f_logit: NDArray[np.float32] | None = None
    full_logit: NDArray[np.float32] | None = None
    full_oracle_telemetry = _FullOracleScoreTelemetry() if is_full_ego_family else None
    if full_oracle_telemetry is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
                pair_amp=args.pair_amp or args.amp,
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
    elif model_family == "cazi_mbn":
        assert cazi_context is not None
        cazi_cfg, feature_stats = cazi_context
        logits = _score_cazi_mbn(
            model,
            row_pairs,
            store,
            device=device,
            batch_size=args.batch_pairs,
            feature_stats=feature_stats,
            missing_features=frozenset(cazi_cfg.expected_missing_features),
        )
    elif model_family == "egostitch_e2e":
        thread_count = _FULL_ORACLE_CPU_THREADS if is_full_ego_family else torch.get_num_threads()
        with _torch_intraop_threads(thread_count):
            decomposed = _score_egostitch_e2e(
                model,
                row_pairs,
                store,
                device=device,
                token_budget=args.token_budget,
                f0_cache=args.f0_cache,
                grounding_cache=args.grounding_cache,
                role_universe="V_val" if args.pairs in _VAL_PAIR_SOURCES else "test",
                scaffold_control=args.scaffold_control,
                universe_pairs=pairs,
                row_start=start,
                oracle_truth_graph=oracle_truth_graph,
                full_oracle_ego_sizes=(
                    all_full_oracle_ego_sizes[start:end]
                    if all_full_oracle_ego_sizes is not None
                    else None
                ),
                full_oracle_telemetry=full_oracle_telemetry,
            )
        from src.model.egostitch.composite import EgoStitchModel

        assert isinstance(model, EgoStitchModel)
        # Three-component refactor (design 2026-08-02 Sec 8): `permanent_null`
        # moved from flat `model.cfg.permanent_null` to nested
        # `model.cfg.classifier.permanent_null`.
        permanent_null = model.cfg.classifier.permanent_null
        primary_logit = _e2e_primary_logit_key(permanent_null)
        logits = decomposed[primary_logit]
        if primary_logit != "full":
            full_logit = decomposed["full"]
        f_logit = decomposed["f_logit"]
        meta_extra = {
            "score_precision": {
                "contract": _EGOSTITCH_E2E_PAIR_PRECISION_CONTRACT,
                "pair_compute_dtype": "float32",
                "pair_autocast": False,
                "logit_storage_dtype": "float32",
            },
            "topo_gen_control": args.topo_gen_control,
            "scaffold_control": {
                "mode": args.scaffold_control,
                "seed": _SCAFFOLD_CONTROL_SEED,
                "keying": "canonical_pair_v1",
            },
            "permanent_null": permanent_null,
            "primary_logit": primary_logit,
        }
        if is_oracle_generator:
            # `_is_heldout_universe` refuses this artifact `heldout: True` no
            # matter how truth-shaped its `pairs_source` looks (module-level
            # docstring), so it cannot pass `validate_test_access_ledger_binding`
            # as a formal held-out result even if fed into it downstream.
            assert oracle_truth_graph is not None
            meta_extra["formal"] = False
            meta_extra["oracle_diagnostic"] = {
                "generator": oracle_generator_name,
                "truth_source": (
                    f"{args.pairs}_g_val" if args.pairs in _VAL_PAIR_SOURCES else "test_graph"
                ),
                "diagnostic_only": True,
                "formal": False,
                "truth_graph_sha256": _oracle_truth_graph_sha256(oracle_truth_graph),
                "truth_graph_node_count": oracle_truth_graph.number_of_nodes(),
                "truth_graph_edge_count": oracle_truth_graph.number_of_edges(),
            }
    else:  # pragma: no cover - build_model already rejects unknown families
        raise ValueError(f"no scoring path for model_family {model_family!r}")

    if test_access is not None:
        if test_access.ledger_binding is None:  # pragma: no cover - invariant guard
            raise RuntimeError("held-out pair access completed without a durable ledger binding")
        meta_extra["test_access_ledger"] = test_access.ledger_binding

    score_wall_seconds = perf_counter() - score_started
    node_ids = sorted({node_id for pair in row_pairs for node_id in pair})
    meta_extra["score_profile"] = {
        "wall_seconds": score_wall_seconds,
        "rows": len(row_pairs),
        "unique_nodes": len(node_ids),
        "measurement": "single_process_compute_wall_seconds",
    }
    full_oracle_sidecar: dict[str, object] | None = None
    if full_oracle_telemetry is not None:
        assert oracle_truth_graph is not None
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        device_memory = (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else 0
        )
        score_profile = cast(dict[str, object], meta_extra["score_profile"])
        score_profile["peak_memory_allocated_bytes"] = peak_allocated
        full_oracle_sidecar = {
            "rows": len(row_pairs),
            "wall_seconds": score_wall_seconds,
            "pairs_per_second": (
                len(row_pairs) / score_wall_seconds if score_wall_seconds > 0.0 else 0.0
            ),
            "ego_size": (
                _full_oracle_distribution(full_oracle_telemetry.ego_sizes)
                if full_oracle_telemetry.ego_sizes
                else None
            ),
            "high_degree_gt16": _full_oracle_high_degree_summary(
                oracle_truth_graph,
                row_pairs,
                full_oracle_telemetry.ego_sizes,
            ),
            "batch_size_histogram": _histogram(full_oracle_telemetry.batch_sizes),
            "batch_n_max_histogram": _histogram(full_oracle_telemetry.batch_max_ego_sizes),
            "budgets": {
                "cell_budget": _FULL_ORACLE_CELL_BUDGET,
                "token_budget": args.token_budget,
                "max_batch_pairs": _FULL_ORACLE_MAX_BATCH_PAIRS,
                "prefetch_depth": _FULL_ORACLE_PREFETCH_DEPTH,
                "cpu_threads": _FULL_ORACLE_CPU_THREADS,
            },
            "max_memory_allocated_bytes": peak_allocated,
            "max_memory_reserved_bytes": peak_reserved,
            "device_total_memory_bytes": device_memory,
            "peak_allocated_fraction": (peak_allocated / device_memory if device_memory else 0.0),
            "producer_wait_seconds": full_oracle_telemetry.producer_wait_seconds,
            "producer_wait_fraction": (
                full_oracle_telemetry.producer_wait_seconds / score_wall_seconds
                if score_wall_seconds > 0.0
                else 0.0
            ),
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
        f_logit=f_logit,
        full_logit=full_logit,
    )
    if full_oracle_sidecar is not None:
        sidecar = output.with_name(f"{output.stem}.telemetry.json")
        sidecar.write_text(
            json.dumps(full_oracle_sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("wrote full-oracle score telemetry to %s", sidecar)
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
        f_logit=merged.f_logit,
        full_logit=merged.full_logit,
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
    "validate_artifact_precision",
    "validate_score_precision",
    "validate_test_access_ledger_binding",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
