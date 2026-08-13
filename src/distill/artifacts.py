"""KD teacher-target artifact: `targets.npz` + `manifest.json` + `node_ids.json`.

Written by `src.distill.teacher_targets` and read by the KD trainer
(`src/train_egostitch.py`'s `_BatchFactory`, a later wave). Format tag
``"kd_targets_v1"``. `truth_source` is always ``"g_fit_only"``: the
teacher-scoring dumper refuses to bind any other truth graph, so an artifact
claiming a different source is not one this loader can trust and is rejected
rather than silently accepted (mirrors the validation style of
`src.experiments.probes.write_e2e_probe_artifact`/`evaluate_e2e_probe_artifact`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

KD_TARGETS_FORMAT = "kd_targets_v1"
TRUTH_SOURCE = "g_fit_only"

_NPZ_NAME = "targets.npz"
_MANIFEST_NAME = "manifest.json"
_NODE_IDS_NAME = "node_ids.json"

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
) -> None:
    """Write one validated KD teacher-target artifact directory.

    Raises:
        ValueError: If the pair arrays disagree on length, `anchor_offsets`
            is not a valid non-decreasing CSR row-start array terminating at
            the pair count, any index is out of range, or `teacher_logit`
            contains a non-finite value.
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
    np.savez(
        npz_path,
        pair_anchor_idx=np.asarray(pair_anchor_idx, dtype=np.int32),
        pair_partner_idx=np.asarray(pair_partner_idx, dtype=np.int32),
        anchor_offsets=np.asarray(anchor_offsets, dtype=np.int64),
        teacher_logit=np.asarray(teacher_logit, dtype=np.float32),
        teacher_pooled_ab=np.asarray(teacher_pooled_ab, dtype=np.float16),
        teacher_pooled_ba=np.asarray(teacher_pooled_ba, dtype=np.float16),
        is_near=np.asarray(is_near, dtype=np.uint8),
        pair_label=np.asarray(pair_label, dtype=np.int8),
    )
    npz_sha256 = _sha256_file(npz_path)

    node_ids_path = output_dir / _NODE_IDS_NAME
    node_ids_bytes = json.dumps(list(node_ids)).encode("utf-8")
    node_ids_path.write_bytes(node_ids_bytes)

    manifest = {
        "format": KD_TARGETS_FORMAT,
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
        if set(archive.files) != required:
            raise ValueError(
                f"KD target artifact {path} arrays must be exactly {sorted(required)}, "
                f"got {sorted(archive.files)}"
            )
        pair_anchor_idx = np.asarray(archive["pair_anchor_idx"], dtype=np.int32)
        pair_partner_idx = np.asarray(archive["pair_partner_idx"], dtype=np.int32)
        anchor_offsets = np.asarray(archive["anchor_offsets"], dtype=np.int64)
        teacher_logit = np.asarray(archive["teacher_logit"], dtype=np.float32)
        teacher_pooled_ab = np.asarray(archive["teacher_pooled_ab"], dtype=np.float16)
        teacher_pooled_ba = np.asarray(archive["teacher_pooled_ba"], dtype=np.float16)
        is_near = np.asarray(archive["is_near"], dtype=np.uint8)
        pair_label = np.asarray(archive["pair_label"], dtype=np.int8)

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
    )


__all__ = ["KD_TARGETS_FORMAT", "TRUTH_SOURCE", "KDTargets", "load_kd_targets", "write_kd_targets"]
