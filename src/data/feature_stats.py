"""Registered per-dimension F0 standardization statistics (spec Sec 13.19.1).

The constants are computed over the ordered V_fit universe only, in fp64, and
are canonically fp32 -- the exact values the model divides by, so a checkpoint's
buffers are directly verifiable against `feature_stats_sha256`.

Fail-closed like `src.data.grounding`, not warn-and-recompute like the F0 cache:
a cached payload that disagrees with the caller's universe or with its own digest
raises. A silently recomputed statistic would change the preprocessing identity
without changing the recorded digest.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FEATURE_STATS_METHOD_ID = "zscore_vfit_v1"
VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class FeatureStats:
    """The registered standardization constants and their identity.

    Attributes:
        mu: Shape ``(d,)`` fp32 per-dimension mean.
        sigma: Shape ``(d,)`` fp32 per-dimension standard deviation.
        method_id: The pinned estimator id.
        node_ids_sha256: Digest of the ordered universe node-id list.
        n_rows: Number of rows the statistics were computed from.
        digest: The `feature_stats_sha256` identity.
    """

    mu: NDArray[np.float32]
    sigma: NDArray[np.float32]
    method_id: str
    node_ids_sha256: str
    n_rows: int
    digest: str


def node_ids_sha256(node_ids: Sequence[str]) -> str:
    """Digest an ordered node-id list exactly as the training access audit does.

    Args:
        node_ids: The ordered universe node ids.

    Returns:
        The 64-hex SHA-256 digest.
    """
    return hashlib.sha256("".join(f"{node}\n" for node in node_ids).encode()).hexdigest()


def feature_stats_digest(
    mu: NDArray[np.float32],
    sigma: NDArray[np.float32],
    *,
    method_id: str,
    node_ids_sha256: str,
) -> str:
    """Compute the `feature_stats_sha256` identity over the fp32 canonical values.

    Args:
        mu: Shape ``(d,)`` fp32 mean.
        sigma: Shape ``(d,)`` fp32 standard deviation.
        method_id: The pinned estimator id.
        node_ids_sha256: Digest of the ordered universe node-id list.

    Returns:
        The 64-hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(f"method\t{method_id}\n".encode())
    digest.update(f"node_ids\t{node_ids_sha256}\n".encode())
    digest.update(f"dim\t{int(mu.shape[0])}\n".encode())
    digest.update(np.ascontiguousarray(mu, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(sigma, dtype=np.float32).tobytes())
    return digest.hexdigest()


def compute_feature_stats(
    rows: NDArray[np.float32], node_ids: Sequence[str]
) -> FeatureStats:
    """Compute the registered constants from one universe's rows.

    Args:
        rows: Shape ``(n, d)`` fp32 F0 rows, aligned with `node_ids`.
        node_ids: The ordered universe node ids.

    Returns:
        The `FeatureStats` bundle.

    Raises:
        ValueError: On a shape/alignment mismatch, fewer than two rows, a
            non-finite fp64 or fp32 mu/var/sigma, or a dimension whose fp64
            variance is exactly zero (degenerate -- constant across the
            universe).
    """
    if rows.ndim != 2:
        raise ValueError("feature stats require a (n, d) row matrix")
    if rows.shape[0] != len(node_ids):
        raise ValueError("feature stats rows and node ids disagree")
    if rows.shape[0] < 2:
        raise ValueError("feature stats require at least two rows")
    accumulator = np.asarray(rows, dtype=np.float64)
    mu64 = accumulator.mean(axis=0)
    if not bool(np.isfinite(mu64).all()):
        # Checked before the variance pass: a non-finite mu64 (from a NaN or
        # Inf input row) would make `accumulator - mu64` itself non-finite,
        # which both warns (invalid value encountered in subtract) and would
        # otherwise reach the degeneracy check below with an all-False mask
        # (NaN is neither > 0 nor <= 0), crashing on an empty flatnonzero
        # instead of raising the documented ValueError.
        bad = int(np.flatnonzero(~np.isfinite(mu64))[0])
        raise ValueError(f"feature stats are not finite in dimension {bad}")
    var64 = np.square(accumulator - mu64).mean(axis=0)
    if not bool(np.isfinite(var64).all()):
        bad = int(np.flatnonzero(~np.isfinite(var64))[0])
        raise ValueError(f"feature stats are not finite in dimension {bad}")
    if not bool((var64 > 0.0).all()):
        # Checked *before* the variance floor: the floor exists only to guard
        # fp64-to-fp32 rounding noise on a legitimately tiny-but-nonzero
        # variance. A dimension that is exactly constant across the universe
        # is a data-quality defect, not something the floor should paper over.
        degenerate = int(np.flatnonzero(var64 <= 0.0)[0])
        raise ValueError(f"degenerate feature dimension {degenerate}: variance is zero")
    sigma64 = np.sqrt(np.maximum(var64, VARIANCE_FLOOR))
    mu = mu64.astype(np.float32)
    sigma = sigma64.astype(np.float32)
    if not bool(np.isfinite(mu).all()) or not bool(np.isfinite(sigma).all()):
        # Reachable independently of the fp64 guards above: an enormous but
        # finite fp64 value can overflow to +/-inf on the fp32 downcast even
        # though mu64/var64 were finite.
        raise ValueError("feature stats are not finite after the fp32 cast")
    identity = node_ids_sha256(node_ids)
    return FeatureStats(
        mu=mu,
        sigma=sigma,
        method_id=FEATURE_STATS_METHOD_ID,
        node_ids_sha256=identity,
        n_rows=int(rows.shape[0]),
        digest=feature_stats_digest(
            mu, sigma, method_id=FEATURE_STATS_METHOD_ID, node_ids_sha256=identity
        ),
    )


def save_feature_stats(stats: FeatureStats, path: Path) -> None:
    """Atomically write the constants next to the feature pack.

    Args:
        stats: The computed constants.
        path: Destination ``.npz`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    np.savez(
        temp,
        mu=stats.mu,
        sigma=stats.sigma,
        method_id=np.asarray(stats.method_id),
        node_ids_sha256=np.asarray(stats.node_ids_sha256),
        n_rows=np.asarray(stats.n_rows, dtype=np.int64),
        digest=np.asarray(stats.digest),
    )
    temp.replace(path)


def load_feature_stats(
    path: Path, *, expected_node_ids_sha256: str | None = None
) -> FeatureStats:
    """Load and verify cached constants, failing closed on any disagreement.

    Args:
        path: The ``.npz`` path written by `save_feature_stats`.
        expected_node_ids_sha256: The caller's ordered-universe digest, when known.

    Returns:
        The verified `FeatureStats`.

    Raises:
        ValueError: On a method, digest, or universe mismatch.
    """
    payload = np.load(path, allow_pickle=False)
    method_id = str(payload["method_id"])
    if method_id != FEATURE_STATS_METHOD_ID:
        raise ValueError(
            f"cached feature stats at {path} use method {method_id!r}, "
            f"expected {FEATURE_STATS_METHOD_ID!r}"
        )
    mu = np.ascontiguousarray(payload["mu"], dtype=np.float32)
    sigma = np.ascontiguousarray(payload["sigma"], dtype=np.float32)
    identity = str(payload["node_ids_sha256"])
    stored = str(payload["digest"])
    recomputed = feature_stats_digest(
        mu, sigma, method_id=method_id, node_ids_sha256=identity
    )
    if recomputed != stored:
        raise ValueError(f"cached feature stats at {path} fail their own digest check")
    if expected_node_ids_sha256 is not None and identity != expected_node_ids_sha256:
        raise ValueError(
            f"cached feature stats at {path} were computed over a different universe"
        )
    return FeatureStats(
        mu=mu,
        sigma=sigma,
        method_id=method_id,
        node_ids_sha256=identity,
        n_rows=int(payload["n_rows"]),
        digest=stored,
    )


def feature_stats_for_universe(
    matrix: NDArray[np.float32],
    node_index: Mapping[str, int],
    node_ids: Sequence[str],
    *,
    cache_path: Path | None = None,
) -> FeatureStats:
    """Compute (or load) the constants for one ordered universe of a shared matrix.

    The rows are gathered before any accumulation, so the result is bit-identical
    whether or not sealed universes share the loaded matrix.

    Args:
        matrix: Shape ``(N, d)`` fp32 matrix that contains the universe's rows.
        node_index: Node id -> row of `matrix`.
        node_ids: The ordered universe node ids.
        cache_path: Optional ``.npz`` cache; verified on read, written on miss.

    Returns:
        The `FeatureStats` bundle.
    """
    identity = node_ids_sha256(node_ids)
    if cache_path is not None and cache_path.is_file():
        return load_feature_stats(cache_path, expected_node_ids_sha256=identity)
    rows = np.ascontiguousarray(
        np.asarray(matrix, dtype=np.float32)[[node_index[node] for node in node_ids]],
        dtype=np.float32,
    )
    stats = compute_feature_stats(rows, node_ids)
    if cache_path is not None:
        save_feature_stats(stats, cache_path)
    return stats
