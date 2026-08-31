"""KD teacher-target artifacts: `targets.npz` + `manifest.json` + `node_ids.json`.

Written by `src.distill.teacher_targets` and read by the KD trainer
(`src/train_b0.py`). Format tag ``"kd_row_targets_v1"``. `truth_source` is
always ``"training_structure"``: the V_val-quarantined training graph over
all train nodes (cross-boundary edges included, V_val-internal pairs
excluded). Teacher inference applies query-edge masking -- a positive
training edge is never visible in its own structural context (structural in
the full-ego oracle generator's stitch).

`targets.npz` holds two row blocks (dtype pinned per array; see
`_ARRAY_DTYPES`):

- Training block: one row per official training row, in the trainer's exact
  row order (row_id == array position), directly joinable against it.
  ``pair_a_idx``/``pair_b_idx`` int32 node indices, ``pair_label`` int8,
  ``teacher_logit`` fp32, ``teacher_rep`` fp16 ``(n_rows, rep_dim)`` -- the
  dump-side symmetrized teacher pooled pair embedding
  ``0.5 * (pooled_ab + pooled_ba)``.
- Validation block: the same five arrays, ``val_``-prefixed, one row per
  official V_val classification row in that fixed order, backing
validation-only KD diagnostics.

The sibling ``"kd_ctx_targets_v1"`` format keeps a deduplicated fp32
``(anchor, partner, teacher_logit)`` score table. Each epoch bank and the
fixed validation diagnostic bank store CSR rows plus indices into that table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

KD_ROW_TARGETS_FORMAT = "kd_row_targets_v1"
KD_CONTEXT_TARGETS_FORMAT = "kd_ctx_targets_v1"
TRUTH_SOURCE = "training_structure"

_NPZ_NAME = "targets.npz"
_MANIFEST_NAME = "manifest.json"
_NODE_IDS_NAME = "node_ids.json"

# dtype pinned per array: int32 pair indices, int8 binary label, fp32
# teacher logit, fp16 pooled teacher-rep embedding.
_ARRAY_DTYPES: dict[str, type] = {
    "pair_a_idx": np.int32,
    "pair_b_idx": np.int32,
    "pair_label": np.int8,
    "teacher_logit": np.float32,
    "teacher_rep": np.float16,
    "val_pair_a_idx": np.int32,
    "val_pair_b_idx": np.int32,
    "val_pair_label": np.int8,
    "val_teacher_logit": np.float32,
    "val_teacher_rep": np.float16,
}


@dataclass(frozen=True)
class KDRowTargets:
    """One loaded KD row-targets artifact."""

    node_ids: list[str]
    pair_a_idx: NDArray[np.int32]
    pair_b_idx: NDArray[np.int32]
    pair_label: NDArray[np.int8]
    teacher_logit: NDArray[np.float32]
    teacher_rep: NDArray[np.float16]
    val_pair_a_idx: NDArray[np.int32]
    val_pair_b_idx: NDArray[np.int32]
    val_pair_label: NDArray[np.int8]
    val_teacher_logit: NDArray[np.float32]
    val_teacher_rep: NDArray[np.float16]
    manifest: dict[str, object]


@dataclass(frozen=True)
class KDContextBank:
    """One CSR context bank joined to the artifact's unique score table."""

    anchor_idx: NDArray[np.int32]
    anchor_offsets: NDArray[np.int64]
    partner_idx: NDArray[np.int32]
    score_idx: NDArray[np.int32]
    is_near: NDArray[np.bool_]


@dataclass(frozen=True)
class KDContextTargets:
    """Loaded context targets for the reference-faithful KD rank arm."""

    node_ids: list[str]
    pair_a_idx: NDArray[np.int32]
    pair_b_idx: NDArray[np.int32]
    teacher_logit: NDArray[np.float32]
    banks: tuple[KDContextBank, ...]
    val_bank: KDContextBank
    manifest: dict[str, object]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_block(
    *,
    block_name: str,
    n_nodes: int,
    a_idx: NDArray[np.integer],
    b_idx: NDArray[np.integer],
    label: NDArray[np.integer],
    logit: NDArray[np.floating],
    rep: NDArray[np.floating],
) -> int:
    """Validate one row block's shared row count, indices, labels, and finiteness.

    Returns:
        The block's row count.

    Raises:
        ValueError: If the block's arrays disagree on row count, an index is
            out of range, `logit` contains a non-finite value, `label` is
            not strictly binary, or `rep` is not a finite rank-2 array.
    """
    n_rows = len(a_idx)
    if not (
        len(b_idx) == n_rows
        and len(label) == n_rows
        and len(logit) == n_rows
        and len(rep) == n_rows
    ):
        raise ValueError(f"{block_name} arrays must share the same row count")
    if n_rows == 0:
        raise ValueError(f"artifact must cover the official {block_name} rows")
    a_arr = np.asarray(a_idx)
    b_arr = np.asarray(b_idx)
    if bool(((a_arr < 0) | (a_arr >= n_nodes)).any()) or bool(
        ((b_arr < 0) | (b_arr >= n_nodes)).any()
    ):
        raise ValueError(f"{block_name} pair_a_idx/pair_b_idx contain an out-of-range node index")
    if not np.isfinite(np.asarray(logit, dtype=np.float64)).all():
        raise ValueError(f"{block_name} teacher_logit contains non-finite values")
    label_arr = np.asarray(label)
    if bool(((label_arr != 0) & (label_arr != 1)).any()):
        raise ValueError(f"{block_name} pair_label must be binary")
    rep_arr = np.asarray(rep)
    if rep_arr.ndim != 2:
        raise ValueError(f"{block_name} teacher_rep must be rank-2 (n_rows, rep_dim)")
    if not np.isfinite(rep_arr.astype(np.float64)).all():
        raise ValueError(f"{block_name} teacher_rep contains non-finite values")
    return n_rows


def write_kd_targets(
    output_dir: Path,
    *,
    node_ids: Sequence[str],
    pair_a_idx: NDArray[np.integer],
    pair_b_idx: NDArray[np.integer],
    pair_label: NDArray[np.integer],
    teacher_logit: NDArray[np.floating],
    teacher_rep: NDArray[np.floating],
    val_pair_a_idx: NDArray[np.integer],
    val_pair_b_idx: NDArray[np.integer],
    val_pair_label: NDArray[np.integer],
    val_teacher_logit: NDArray[np.floating],
    val_teacher_rep: NDArray[np.floating],
    truth_graph_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_id: str | None,
) -> None:
    """Write one validated KD row-targets artifact directory.

    The training block must cover the official training rows in the
    trainer's exact row order (row_id == array position); the validation
    block covers the official V_val classification rows in their fixed
    order.

    Raises:
        ValueError: If either block's arrays disagree on row count, is
            empty, contains an out-of-range pair index, a non-finite
            `teacher_logit`/`teacher_rep` value, a non-binary `pair_label`,
            or a `teacher_rep`/`val_teacher_rep` rank other than 2; if the
            two blocks' `rep_dim` disagree.
    """
    n_nodes = len(node_ids)
    n_rows = _validate_block(
        block_name="training",
        n_nodes=n_nodes,
        a_idx=pair_a_idx,
        b_idx=pair_b_idx,
        label=pair_label,
        logit=teacher_logit,
        rep=teacher_rep,
    )
    n_val_rows = _validate_block(
        block_name="validation",
        n_nodes=n_nodes,
        a_idx=val_pair_a_idx,
        b_idx=val_pair_b_idx,
        label=val_pair_label,
        logit=val_teacher_logit,
        rep=val_teacher_rep,
    )
    rep_dim = np.asarray(teacher_rep).shape[1]
    if np.asarray(val_teacher_rep).shape[1] != rep_dim:
        raise ValueError("teacher_rep and val_teacher_rep must share the same rep_dim")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / _NPZ_NAME
    arrays: dict[str, NDArray[np.generic]] = {
        "pair_a_idx": np.asarray(pair_a_idx, dtype=np.int32),
        "pair_b_idx": np.asarray(pair_b_idx, dtype=np.int32),
        "pair_label": np.asarray(pair_label, dtype=np.int8),
        "teacher_logit": np.asarray(teacher_logit, dtype=np.float32),
        "teacher_rep": np.asarray(teacher_rep, dtype=np.float16),
        "val_pair_a_idx": np.asarray(val_pair_a_idx, dtype=np.int32),
        "val_pair_b_idx": np.asarray(val_pair_b_idx, dtype=np.int32),
        "val_pair_label": np.asarray(val_pair_label, dtype=np.int8),
        "val_teacher_logit": np.asarray(val_teacher_logit, dtype=np.float32),
        "val_teacher_rep": np.asarray(val_teacher_rep, dtype=np.float16),
    }
    np.savez(npz_path, **cast(dict[str, Any], arrays))
    npz_sha256 = _sha256_file(npz_path)

    node_ids_path = output_dir / _NODE_IDS_NAME
    node_ids_bytes = json.dumps(list(node_ids)).encode("utf-8")
    node_ids_path.write_bytes(node_ids_bytes)

    label_arr = np.asarray(pair_label)
    manifest: dict[str, object] = {
        "format": KD_ROW_TARGETS_FORMAT,
        "truth_source": TRUTH_SOURCE,
        "truth_graph_sha256": truth_graph_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "n_nodes": n_nodes,
        "n_rows": n_rows,
        "n_val_rows": n_val_rows,
        "n_positive_rows": int(np.sum(label_arr)),
        "rep_dim": int(rep_dim),
        "teacher_rep_dtype": "float16",
        "npz_sha256": npz_sha256,
        "node_ids_sha256": _sha256_bytes(node_ids_bytes),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_kd_targets(path: Path) -> KDRowTargets:
    """Load one KD row-targets artifact directory.

    Digest/format verification was deliberately removed (user decision,
    2026-08-13): the manifest is provenance metadata, not a gate.

    Args:
        path: Artifact directory.

    Raises:
        ValueError: If a file is missing, the npz array set is not exactly
            the ten required names.
    """
    manifest_path = path / _MANIFEST_NAME
    node_ids_path = path / _NODE_IDS_NAME
    npz_path = path / _NPZ_NAME
    if not manifest_path.is_file():
        raise ValueError(f"KD target artifact {path} is missing {_MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not npz_path.is_file():
        raise ValueError(f"KD target artifact {path} is missing {_NPZ_NAME}")
    if not node_ids_path.is_file():
        raise ValueError(f"KD target artifact {path} is missing {_NODE_IDS_NAME}")
    node_ids = json.loads(node_ids_path.read_text(encoding="utf-8"))

    with np.load(npz_path) as archive:
        required = set(_ARRAY_DTYPES)
        present = set(archive.files)
        if present != required:
            raise ValueError(
                f"KD target artifact {path} arrays must be exactly {sorted(required)} "
                f"got {sorted(archive.files)}"
            )
        pair_a_idx = np.asarray(archive["pair_a_idx"], dtype=np.int32)
        pair_b_idx = np.asarray(archive["pair_b_idx"], dtype=np.int32)
        pair_label = np.asarray(archive["pair_label"], dtype=np.int8)
        teacher_logit = np.asarray(archive["teacher_logit"], dtype=np.float32)
        teacher_rep = np.asarray(archive["teacher_rep"], dtype=np.float16)
        val_pair_a_idx = np.asarray(archive["val_pair_a_idx"], dtype=np.int32)
        val_pair_b_idx = np.asarray(archive["val_pair_b_idx"], dtype=np.int32)
        val_pair_label = np.asarray(archive["val_pair_label"], dtype=np.int8)
        val_teacher_logit = np.asarray(archive["val_teacher_logit"], dtype=np.float32)
        val_teacher_rep = np.asarray(archive["val_teacher_rep"], dtype=np.float16)

    return KDRowTargets(
        node_ids=node_ids,
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        pair_label=pair_label,
        teacher_logit=teacher_logit,
        teacher_rep=teacher_rep,
        val_pair_a_idx=val_pair_a_idx,
        val_pair_b_idx=val_pair_b_idx,
        val_pair_label=val_pair_label,
        val_teacher_logit=val_teacher_logit,
        val_teacher_rep=val_teacher_rep,
        manifest=manifest,
    )


def _context_array_names(n_banks: int) -> set[str]:
    suffixes = {"anchor_idx", "anchor_offsets", "partner_idx", "score_idx", "is_near"}
    names = {"pair_a_idx", "pair_b_idx", "teacher_logit"}
    names.update(
        f"bank_{bank_idx:03d}_{suffix}" for bank_idx in range(n_banks) for suffix in suffixes
    )
    names.update(f"val_{suffix}" for suffix in suffixes)
    return names


def _validate_context_score_table(
    *,
    n_nodes: int,
    pair_a_idx: NDArray[np.integer],
    pair_b_idx: NDArray[np.integer],
    teacher_logit: NDArray[np.floating],
) -> int:
    arrays = (np.asarray(pair_a_idx), np.asarray(pair_b_idx), np.asarray(teacher_logit))
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("context score arrays must be rank-1")
    n_scores = len(arrays[0])
    if n_scores == 0 or any(len(array) != n_scores for array in arrays[1:]):
        raise ValueError("context score arrays must share a non-zero row count")
    if bool(((arrays[0] < 0) | (arrays[0] >= n_nodes)).any()) or bool(
        ((arrays[1] < 0) | (arrays[1] >= n_nodes)).any()
    ):
        raise ValueError("context score pairs contain an out-of-range node index")
    pair_keys = arrays[0].astype(np.int64) * n_nodes + arrays[1].astype(np.int64)
    if len(np.unique(pair_keys)) != n_scores:
        raise ValueError("context score pairs must be unique")
    with np.errstate(over="ignore", invalid="ignore"):
        fp32_logit = arrays[2].astype(np.float32)
    if not np.isfinite(fp32_logit).all():
        raise ValueError("context teacher_logit must remain finite in float32")
    return n_scores


def _validate_context_bank(
    bank: KDContextBank,
    *,
    label: str,
    n_nodes: int,
    pair_a_idx: NDArray[np.integer],
    pair_b_idx: NDArray[np.integer],
    require_full_universe: bool,
) -> None:
    anchor_idx = np.asarray(bank.anchor_idx)
    anchor_offsets = np.asarray(bank.anchor_offsets)
    partner_idx = np.asarray(bank.partner_idx)
    score_idx = np.asarray(bank.score_idx)
    is_near = np.asarray(bank.is_near)
    if any(
        array.ndim != 1 for array in (anchor_idx, anchor_offsets, partner_idx, score_idx, is_near)
    ):
        raise ValueError(f"{label} arrays must be rank-1")
    if is_near.dtype != np.dtype(np.bool_):
        raise ValueError(f"{label} is_near must have dtype bool")
    if len(anchor_offsets) != len(anchor_idx) + 1:
        raise ValueError(f"{label} anchor_offsets must have one entry per anchor plus one")
    if len(anchor_offsets) == 0 or anchor_offsets[0] != 0:
        raise ValueError(f"{label} anchor_offsets must start at zero")
    n_contexts = len(partner_idx)
    if len(score_idx) != n_contexts or len(is_near) != n_contexts:
        raise ValueError(f"{label} context arrays must share the same row count")
    if bool((np.diff(anchor_offsets) < 0).any()) or anchor_offsets[-1] != n_contexts:
        raise ValueError(f"{label} anchor_offsets must be monotone and end at the context count")
    if bool(((anchor_idx < 0) | (anchor_idx >= n_nodes)).any()) or bool(
        ((partner_idx < 0) | (partner_idx >= n_nodes)).any()
    ):
        raise ValueError(f"{label} contains an out-of-range node index")
    if len(np.unique(anchor_idx)) != len(anchor_idx):
        raise ValueError(f"{label} anchor_idx must be unique")
    if require_full_universe and not np.array_equal(anchor_idx, np.arange(n_nodes)):
        raise ValueError(f"{label} anchor_idx must cover the node universe in order")
    if bool(((score_idx < 0) | (score_idx >= len(pair_a_idx))).any()):
        raise ValueError(f"{label} score_idx contains an out-of-range score index")
    repeated_anchor_idx = np.repeat(anchor_idx, np.diff(anchor_offsets))
    if not np.array_equal(
        np.asarray(pair_a_idx)[score_idx], repeated_anchor_idx
    ) or not np.array_equal(np.asarray(pair_b_idx)[score_idx], partner_idx):
        raise ValueError(f"{label} score_idx does not identify its CSR (anchor, partner) rows")


def _validate_context_quarantine(
    banks: Sequence[KDContextBank],
    *,
    n_nodes: int,
    val_bank: KDContextBank,
) -> None:
    forbidden = np.zeros(n_nodes, dtype=np.bool_)
    forbidden[val_bank.anchor_idx] = True
    for bank_idx, bank in enumerate(banks):
        anchors = np.repeat(bank.anchor_idx, np.diff(bank.anchor_offsets))
        if bool((forbidden[anchors] & forbidden[bank.partner_idx]).any()):
            raise ValueError(f"context bank {bank_idx} contains a V_val-internal pair")
    if bool(forbidden[val_bank.partner_idx].any()):
        raise ValueError("validation context bank contains a V_val-internal pair")


def _context_bank_arrays(prefix: str, bank: KDContextBank) -> dict[str, NDArray[np.generic]]:
    return {
        f"{prefix}anchor_idx": np.asarray(bank.anchor_idx, dtype=np.int32),
        f"{prefix}anchor_offsets": np.asarray(bank.anchor_offsets, dtype=np.int64),
        f"{prefix}partner_idx": np.asarray(bank.partner_idx, dtype=np.int32),
        f"{prefix}score_idx": np.asarray(bank.score_idx, dtype=np.int32),
        f"{prefix}is_near": np.asarray(bank.is_near, dtype=np.bool_),
    }


def write_kd_context_targets(
    output_dir: Path,
    *,
    node_ids: Sequence[str],
    pair_a_idx: NDArray[np.integer],
    pair_b_idx: NDArray[np.integer],
    teacher_logit: NDArray[np.floating],
    banks: Sequence[KDContextBank],
    val_bank: KDContextBank,
    sampler_params: Mapping[str, object],
    seed: int,
    truth_graph_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_id: str | None,
) -> None:
    """Write deduplicated context scores and their per-bank CSR joins."""
    if not node_ids or any(not isinstance(node_id, str) for node_id in node_ids):
        raise ValueError("context node_ids must be a non-empty sequence of strings")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("context node_ids must be unique")
    if not banks:
        raise ValueError("context targets must contain at least one training bank")
    n_nodes = len(node_ids)
    n_scores = _validate_context_score_table(
        n_nodes=n_nodes,
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        teacher_logit=teacher_logit,
    )
    for bank_idx, bank in enumerate(banks):
        _validate_context_bank(
            bank,
            label=f"context bank {bank_idx}",
            n_nodes=n_nodes,
            pair_a_idx=pair_a_idx,
            pair_b_idx=pair_b_idx,
            require_full_universe=True,
        )
    _validate_context_bank(
        val_bank,
        label="validation context bank",
        n_nodes=n_nodes,
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        require_full_universe=False,
    )
    if len(val_bank.anchor_idx) == 0:
        raise ValueError("validation context bank must contain its V_val anchors")
    _validate_context_quarantine(banks, n_nodes=n_nodes, val_bank=val_bank)
    referenced = np.concatenate([bank.score_idx for bank in (*banks, val_bank)])
    if not np.array_equal(np.unique(referenced), np.arange(n_scores)):
        raise ValueError("every unique context score must be referenced by a bank")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {
        "pair_a_idx": np.asarray(pair_a_idx, dtype=np.int32),
        "pair_b_idx": np.asarray(pair_b_idx, dtype=np.int32),
        "teacher_logit": np.asarray(teacher_logit, dtype=np.float32),
    }
    for bank_idx, bank in enumerate(banks):
        arrays.update(_context_bank_arrays(f"bank_{bank_idx:03d}_", bank))
    arrays.update(_context_bank_arrays("val_", val_bank))
    np.savez(output_dir / _NPZ_NAME, **cast(dict[str, Any], arrays))

    node_ids_bytes = json.dumps(list(node_ids)).encode("utf-8")
    (output_dir / _NODE_IDS_NAME).write_bytes(node_ids_bytes)
    manifest: dict[str, object] = {
        "format": KD_CONTEXT_TARGETS_FORMAT,
        "truth_source": TRUTH_SOURCE,
        "truth_graph_sha256": truth_graph_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "sampler_params": dict(sampler_params),
        "seed": int(seed),
        "n_banks": len(banks),
        "n_nodes": n_nodes,
        "n_scores": n_scores,
        "n_val_anchors": len(val_bank.anchor_idx),
        "teacher_logit_dtype": "float32",
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_context_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    dtype: type[np.generic],
) -> NDArray[np.generic]:
    array = np.asarray(archive[name])
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"context target array {name} must have dtype {np.dtype(dtype)}")
    return array


def _load_context_bank(archive: np.lib.npyio.NpzFile, prefix: str) -> KDContextBank:
    return KDContextBank(
        anchor_idx=cast(
            NDArray[np.int32], _load_context_array(archive, f"{prefix}anchor_idx", np.int32)
        ),
        anchor_offsets=cast(
            NDArray[np.int64], _load_context_array(archive, f"{prefix}anchor_offsets", np.int64)
        ),
        partner_idx=cast(
            NDArray[np.int32], _load_context_array(archive, f"{prefix}partner_idx", np.int32)
        ),
        score_idx=cast(
            NDArray[np.int32], _load_context_array(archive, f"{prefix}score_idx", np.int32)
        ),
        is_near=cast(NDArray[np.bool_], _load_context_array(archive, f"{prefix}is_near", np.bool_)),
    )


def load_kd_context_targets(
    path: Path,
    *,
    expected_node_ids: Sequence[str] | None = None,
    expected_val_anchor_idx: NDArray[np.integer] | None = None,
) -> KDContextTargets:
    """Load and validate one ``kd_ctx_targets_v1`` artifact.

    Digests and provenance strings are deliberately not gates. Passing
    ``expected_node_ids`` enables the trainer to fail closed on universe and
    ordering drift at the artifact boundary; ``expected_val_anchor_idx`` does
    the same for the V_val diagnostic/quarantine identity.
    """
    manifest_path = path / _MANIFEST_NAME
    node_ids_path = path / _NODE_IDS_NAME
    npz_path = path / _NPZ_NAME
    for artifact_path in (manifest_path, node_ids_path, npz_path):
        if not artifact_path.is_file():
            raise ValueError(f"KD context target artifact {path} is missing {artifact_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    node_ids = json.loads(node_ids_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("context target manifest must be a JSON object")
    if not isinstance(node_ids, list) or any(not isinstance(node_id, str) for node_id in node_ids):
        raise ValueError("context node_ids must be a sequence of strings")
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("context node_ids must be non-empty and unique")
    if expected_node_ids is not None and node_ids != list(expected_node_ids):
        raise ValueError("context target node universe/order does not match expected_node_ids")
    n_banks_value = manifest.get("n_banks")
    if not isinstance(n_banks_value, int) or isinstance(n_banks_value, bool) or n_banks_value < 1:
        raise ValueError("context target manifest n_banks must be a positive integer")
    n_banks = n_banks_value

    with np.load(npz_path) as archive:
        required = _context_array_names(n_banks)
        if set(archive.files) != required:
            raise ValueError(
                f"KD context target artifact {path} arrays must be exactly {sorted(required)} "
                f"got {sorted(archive.files)}"
            )
        pair_a_idx = cast(NDArray[np.int32], _load_context_array(archive, "pair_a_idx", np.int32))
        pair_b_idx = cast(NDArray[np.int32], _load_context_array(archive, "pair_b_idx", np.int32))
        teacher_logit = cast(
            NDArray[np.float32], _load_context_array(archive, "teacher_logit", np.float32)
        )
        banks = tuple(
            _load_context_bank(archive, f"bank_{bank_idx:03d}_") for bank_idx in range(n_banks)
        )
        val_bank = _load_context_bank(archive, "val_")

    n_scores = _validate_context_score_table(
        n_nodes=len(node_ids),
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        teacher_logit=teacher_logit,
    )
    for bank_idx, bank in enumerate(banks):
        _validate_context_bank(
            bank,
            label=f"context bank {bank_idx}",
            n_nodes=len(node_ids),
            pair_a_idx=pair_a_idx,
            pair_b_idx=pair_b_idx,
            require_full_universe=True,
        )
    _validate_context_bank(
        val_bank,
        label="validation context bank",
        n_nodes=len(node_ids),
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        require_full_universe=False,
    )
    if len(val_bank.anchor_idx) == 0:
        raise ValueError("validation context bank must contain its V_val anchors")
    if expected_val_anchor_idx is not None and not np.array_equal(
        val_bank.anchor_idx, np.asarray(expected_val_anchor_idx)
    ):
        raise ValueError("validation context anchors do not match expected_val_anchor_idx")
    _validate_context_quarantine(banks, n_nodes=len(node_ids), val_bank=val_bank)
    referenced = np.concatenate([bank.score_idx for bank in (*banks, val_bank)])
    if not np.array_equal(np.unique(referenced), np.arange(n_scores)):
        raise ValueError("every unique context score must be referenced by a bank")
    n_nodes_value = manifest.get("n_nodes")
    n_scores_value = manifest.get("n_scores")
    n_val_anchors_value = manifest.get("n_val_anchors")
    if (
        not isinstance(n_nodes_value, int)
        or isinstance(n_nodes_value, bool)
        or n_nodes_value != len(node_ids)
        or not isinstance(n_scores_value, int)
        or isinstance(n_scores_value, bool)
        or n_scores_value != n_scores
        or not isinstance(n_val_anchors_value, int)
        or isinstance(n_val_anchors_value, bool)
        or n_val_anchors_value != len(val_bank.anchor_idx)
    ):
        raise ValueError("context target manifest row/universe counts do not match its arrays")
    return KDContextTargets(
        node_ids=node_ids,
        pair_a_idx=pair_a_idx,
        pair_b_idx=pair_b_idx,
        teacher_logit=teacher_logit,
        banks=banks,
        val_bank=val_bank,
        manifest=manifest,
    )


__all__ = [
    "KD_CONTEXT_TARGETS_FORMAT",
    "KD_ROW_TARGETS_FORMAT",
    "TRUTH_SOURCE",
    "KDContextBank",
    "KDContextTargets",
    "KDRowTargets",
    "load_kd_context_targets",
    "load_kd_targets",
    "write_kd_context_targets",
    "write_kd_targets",
]
