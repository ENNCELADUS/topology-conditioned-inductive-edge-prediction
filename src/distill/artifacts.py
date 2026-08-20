"""KD teacher-target artifact: `targets.npz` + `manifest.json` + `node_ids.json`.

Written by `src.distill.teacher_targets` (and its `src.distill.content_logit`/
`src.distill.heuristic_targets` derivatives) and read by the KD trainer
(`src/train_b0.py`'s `KDStream`). Format tag ``"kd_targets_v2"``, or
``"kd_targets_v3"`` when `content_logit` is written (write-side stamp only;
`load_kd_targets` performs no format check). `truth_source` is always
``"training_structure"``: the V_val-quarantined training graph over all train
nodes (cross-boundary edges included, V_val-internal pairs excluded);
pre-V_val ``"g_fit_only"`` artifacts carry a different stamp so the two
teacher semantics stay distinguishable.

`targets.npz` arrays, one row per context pair (dtype pinned per array; see
`_ARRAY_DTYPES`): ``pair_anchor_idx``/``pair_partner_idx`` int32 node
indices, ``anchor_offsets`` int64 CSR row starts, ``teacher_logit`` fp32,
``teacher_pooled_ab``/``teacher_pooled_ba`` fp16 (~1 GB budget), ``is_near``
uint8, ``pair_label`` int8. ``content_logit`` fp32 is optional (kd_targets_v3
only): the same rows scored by a content-only baseline checkpoint, consumed
by the D5 residual-distillation arm as ``teacher_logit - content_logit``.
``teacher_seeds`` fp16 ``(n_pairs, K, d_t)`` is optional (kd_targets_v4): the
symmetrized per-slot PMA seed tokens ``½(S_ab + S_ba)``, consumed by the D9
pair-latent-generation arm; its presence adds ``seed_count``/``seed_dim`` and
the dump-side AB/BA ``seed_symmetry`` audit statistics to the manifest.
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

KD_TARGETS_FORMAT = "kd_targets_v2"
KD_TARGETS_FORMAT_V3 = "kd_targets_v3"
KD_TARGETS_FORMAT_V4 = "kd_targets_v4"
TRUTH_SOURCE = "training_structure"

_NPZ_NAME = "targets.npz"
_MANIFEST_NAME = "manifest.json"
_NODE_IDS_NAME = "node_ids.json"
_CONTENT_LOGIT_KEY = "content_logit"
_TEACHER_SEEDS_KEY = "teacher_seeds"

# dtype pinned per array (B1 plan "KD sampler + teacher-target artifact"):
# int32 pair indices, int64 CSR offsets, fp32 teacher logits, fp16 pooled
# embeddings (~1 GB budget), uint8 near/random flag, int8 binary label.
_ARRAY_DTYPES: dict[str, type] = {
    "pair_anchor_idx": np.int32,
    "pair_partner_idx": np.int32,
    "anchor_offsets": np.int64,
    "teacher_logit": np.float32,
    "teacher_pooled_ab": np.float16,
    "teacher_pooled_ba": np.float16,
    "is_near": np.uint8,
    "pair_label": np.int8,
}


@dataclass(frozen=True)
class KDTargets:
    """One loaded, digest-validated KD teacher-target artifact."""

    node_ids: list[str]
    pair_anchor_idx: NDArray[np.int32]
    pair_partner_idx: NDArray[np.int32]
    anchor_offsets: NDArray[np.int64]
    teacher_logit: NDArray[np.float32]
    teacher_pooled_ab: NDArray[np.float16]
    teacher_pooled_ba: NDArray[np.float16]
    is_near: NDArray[np.uint8]
    pair_label: NDArray[np.int8]
    manifest: dict[str, object]
    content_logit: NDArray[np.float32] | None = None
    teacher_seeds: NDArray[np.float16] | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_kd_targets(
    output_dir: Path,
    *,
    node_ids: Sequence[str],
    pair_anchor_idx: NDArray[np.integer],
    pair_partner_idx: NDArray[np.integer],
    anchor_offsets: NDArray[np.integer],
    teacher_logit: NDArray[np.floating],
    teacher_pooled_ab: NDArray[np.floating],
    teacher_pooled_ba: NDArray[np.floating],
    is_near: NDArray[np.integer],
    pair_label: NDArray[np.integer],
    truth_graph_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_id: str | None,
    k_near: int,
    k_rand: int,
    seed: int,
    content_logit: NDArray[np.floating] | None = None,
    teacher_seeds: NDArray[np.floating] | None = None,
    seed_symmetry: dict[str, object] | None = None,
) -> None:
    """Write one validated KD teacher-target artifact directory.

    `content_logit`, when given, is an optional fp32 array (same pair count
    as the other rows) written alongside `teacher_logit`; its presence bumps
    the written manifest ``"format"`` from ``KD_TARGETS_FORMAT`` to
    ``KD_TARGETS_FORMAT_V3``. `teacher_seeds`, when given, is an optional
    fp16 ``(n_pairs, K, d_t)`` array of symmetrized PMA seed tokens; its
    presence stamps ``KD_TARGETS_FORMAT_V4`` and records `seed_symmetry`
    (the dump-side AB/BA order-asymmetry audit) in the manifest. All stamps
    are write-side only -- `load_kd_targets` performs no format check.

    Raises:
        ValueError: If the pair arrays disagree on length, `anchor_offsets`
            is not a valid non-decreasing CSR row-start array terminating at
            the pair count, any index is out of range, `teacher_logit`
            contains a non-finite value, `content_logit`/`teacher_seeds` is
            given with a mismatched pair count, wrong rank, or a non-finite
            value, or `seed_symmetry` is given without `teacher_seeds`.
    """
    n_pairs = len(pair_anchor_idx)
    if not (
        len(pair_partner_idx) == n_pairs
        and len(teacher_logit) == n_pairs
        and len(teacher_pooled_ab) == n_pairs
        and len(teacher_pooled_ba) == n_pairs
        and len(is_near) == n_pairs
        and len(pair_label) == n_pairs
    ):
        raise ValueError("KD target arrays must share the same pair count")
    n_nodes = len(node_ids)
    if len(anchor_offsets) != n_nodes + 1:
        raise ValueError("anchor_offsets must have len(node_ids) + 1 entries")
    if int(anchor_offsets[0]) != 0 or int(anchor_offsets[-1]) != n_pairs:
        raise ValueError("anchor_offsets must run from 0 to the pair count")
    if np.any(np.diff(np.asarray(anchor_offsets)) < 0):
        raise ValueError("anchor_offsets must be non-decreasing")
    anchor_arr = np.asarray(pair_anchor_idx)
    partner_arr = np.asarray(pair_partner_idx)
    if n_pairs and (
        bool(((anchor_arr < 0) | (anchor_arr >= n_nodes)).any())
        or bool(((partner_arr < 0) | (partner_arr >= n_nodes)).any())
    ):
        raise ValueError("pair_anchor_idx/pair_partner_idx contain an out-of-range node index")
    if not np.isfinite(np.asarray(teacher_logit, dtype=np.float64)).all():
        raise ValueError("teacher_logit contains non-finite values")
    if content_logit is not None:
        if len(content_logit) != n_pairs:
            raise ValueError("content_logit must have the same pair count as the other arrays")
        if not np.isfinite(np.asarray(content_logit, dtype=np.float64)).all():
            raise ValueError("content_logit contains non-finite values")
    if teacher_seeds is not None:
        seeds_arr = np.asarray(teacher_seeds)
        if seeds_arr.ndim != 3:
            raise ValueError("teacher_seeds must be (n_pairs, seed_count, seed_dim)")
        if len(seeds_arr) != n_pairs:
            raise ValueError("teacher_seeds must have the same pair count as the other arrays")
        if not np.isfinite(seeds_arr.astype(np.float64)).all():
            raise ValueError("teacher_seeds contains non-finite values")
    elif seed_symmetry is not None:
        raise ValueError("seed_symmetry requires teacher_seeds")
    label_arr = np.asarray(pair_label)
    if n_pairs and bool(((label_arr != 0) & (label_arr != 1)).any()):
        raise ValueError("pair_label must be binary")
    near_arr = np.asarray(is_near)
    if n_pairs and bool(((near_arr != 0) & (near_arr != 1)).any()):
        raise ValueError("is_near must be binary")
    if k_near < 0 or k_rand < 0:
        raise ValueError("k_near and k_rand must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / _NPZ_NAME
    arrays: dict[str, NDArray[np.generic]] = {
        "pair_anchor_idx": np.asarray(pair_anchor_idx, dtype=np.int32),
        "pair_partner_idx": np.asarray(pair_partner_idx, dtype=np.int32),
        "anchor_offsets": np.asarray(anchor_offsets, dtype=np.int64),
        "teacher_logit": np.asarray(teacher_logit, dtype=np.float32),
        "teacher_pooled_ab": np.asarray(teacher_pooled_ab, dtype=np.float16),
        "teacher_pooled_ba": np.asarray(teacher_pooled_ba, dtype=np.float16),
        "is_near": np.asarray(is_near, dtype=np.uint8),
        "pair_label": np.asarray(pair_label, dtype=np.int8),
    }
    if content_logit is not None:
        arrays[_CONTENT_LOGIT_KEY] = np.asarray(content_logit, dtype=np.float32)
    if teacher_seeds is not None:
        arrays[_TEACHER_SEEDS_KEY] = np.asarray(teacher_seeds, dtype=np.float16)
    np.savez(npz_path, **cast(dict[str, Any], arrays))
    npz_sha256 = _sha256_file(npz_path)

    node_ids_path = output_dir / _NODE_IDS_NAME
    node_ids_bytes = json.dumps(list(node_ids)).encode("utf-8")
    node_ids_path.write_bytes(node_ids_bytes)

    if teacher_seeds is not None:
        format_stamp = KD_TARGETS_FORMAT_V4
    elif content_logit is not None:
        format_stamp = KD_TARGETS_FORMAT_V3
    else:
        format_stamp = KD_TARGETS_FORMAT
    manifest: dict[str, object] = {
        "format": format_stamp,
        "truth_source": TRUTH_SOURCE,
        "truth_graph_sha256": truth_graph_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "k_near": k_near,
        "k_rand": k_rand,
        "seed": seed,
        "n_nodes": n_nodes,
        "n_pairs": n_pairs,
        "n_positive_pairs": int(np.sum(label_arr)) if n_pairs else 0,
        "npz_sha256": npz_sha256,
        "node_ids_sha256": _sha256_bytes(node_ids_bytes),
        "pooled_embedding_dtype": "float16",
        "created_utc": datetime.now(UTC).isoformat(),
    }
    if teacher_seeds is not None:
        seeds_arr = np.asarray(teacher_seeds)
        manifest["seed_count"] = int(seeds_arr.shape[1])
        manifest["seed_dim"] = int(seeds_arr.shape[2])
        manifest["teacher_seeds_dtype"] = "float16"
        if seed_symmetry is not None:
            manifest["seed_symmetry"] = seed_symmetry
    (output_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_kd_targets(path: Path) -> KDTargets:
    """Load one KD teacher-target artifact directory.

    Digest/format verification was deliberately removed (user decision,
    2026-08-13): the manifest is provenance metadata, not a gate.

    Raises:
        ValueError: If a file is missing or the npz array set is wrong.
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
        optional = {_CONTENT_LOGIT_KEY, _TEACHER_SEEDS_KEY}
        present = set(archive.files)
        if not (required <= present and present - required <= optional):
            raise ValueError(
                f"KD target artifact {path} arrays must be exactly {sorted(required)} "
                f"(optionally plus any of {sorted(optional)}), got {sorted(archive.files)}"
            )
        pair_anchor_idx = np.asarray(archive["pair_anchor_idx"], dtype=np.int32)
        pair_partner_idx = np.asarray(archive["pair_partner_idx"], dtype=np.int32)
        anchor_offsets = np.asarray(archive["anchor_offsets"], dtype=np.int64)
        teacher_logit = np.asarray(archive["teacher_logit"], dtype=np.float32)
        teacher_pooled_ab = np.asarray(archive["teacher_pooled_ab"], dtype=np.float16)
        teacher_pooled_ba = np.asarray(archive["teacher_pooled_ba"], dtype=np.float16)
        is_near = np.asarray(archive["is_near"], dtype=np.uint8)
        pair_label = np.asarray(archive["pair_label"], dtype=np.int8)
        content_logit = (
            np.asarray(archive[_CONTENT_LOGIT_KEY], dtype=np.float32)
            if _CONTENT_LOGIT_KEY in archive.files
            else None
        )
        teacher_seeds = (
            np.asarray(archive[_TEACHER_SEEDS_KEY], dtype=np.float16)
            if _TEACHER_SEEDS_KEY in archive.files
            else None
        )

    return KDTargets(
        node_ids=node_ids,
        pair_anchor_idx=pair_anchor_idx,
        pair_partner_idx=pair_partner_idx,
        anchor_offsets=anchor_offsets,
        teacher_logit=teacher_logit,
        teacher_pooled_ab=teacher_pooled_ab,
        teacher_pooled_ba=teacher_pooled_ba,
        is_near=is_near,
        pair_label=pair_label,
        manifest=manifest,
        content_logit=content_logit,
        teacher_seeds=teacher_seeds,
    )


__all__ = [
    "KD_TARGETS_FORMAT",
    "KD_TARGETS_FORMAT_V3",
    "KD_TARGETS_FORMAT_V4",
    "TRUTH_SOURCE",
    "KDTargets",
    "load_kd_targets",
    "write_kd_targets",
]
