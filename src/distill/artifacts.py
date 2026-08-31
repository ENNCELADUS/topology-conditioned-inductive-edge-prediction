"""KD teacher-target artifact: `targets.npz` + `manifest.json` + `node_ids.json`.

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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

KD_ROW_TARGETS_FORMAT = "kd_row_targets_v1"
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


__all__ = [
    "KD_ROW_TARGETS_FORMAT",
    "TRUTH_SOURCE",
    "KDRowTargets",
    "load_kd_targets",
    "write_kd_targets",
]
