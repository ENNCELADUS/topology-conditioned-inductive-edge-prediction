"""Grounding pools: exact top-``n_g`` cosine neighbors in F0 space (spec Sec 13.12).

``G(u)`` is computed within the node's **own split side** (the caller passes
that side's node list and its ``role_universe`` identity), self excluded,
from the frozen F0 mean-pool matrix. Exact blockwise matmul -- the operative
node set is ~10k, no ANN needed. Deterministic: cosine ties break by
ascending node index. Optionally cached to an ``.npz``.

Rev-3.1 cache binding (spec Sec 14.4.4): every cache carries a
``pool_method_hash`` -- a SHA-256 over the method id, ``n_ground``, the
optional shortlist ``M`` (omitted entirely when absent; there is no reranker
in rev-3.1), the ordered F0/feature-pack digest, and the caller-supplied
``role_universe`` identity. A cache is trusted **only** when this hash
matches exactly; any mismatch -- including a pre-rev-3.1 cache with no
``pool_method_hash`` field at all -- raises instead of silently recomputing
or overwriting. This closes the gap where a redefined pool (different
`n_ground`, mutated F0 features under unchanged node ids, or a pool built for
the wrong role universe) could be read back through a stale cache path with
no error.

This change invalidates every existing grounding cache on disk: all four
spec Sec 13.12 role universes (``V_fit``, ``V_qual``, ``V_select``, ``test``)
for every strategy this family is run under (currently ``breadth_first`` is
the only strategy wired into an `egostitch_e2e` config). Regenerating those
caches is an execution step, out of scope for this change -- the first warm
read after this lands will raise on the missing ``pool_method_hash`` field
and a cold rebuild is required.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_BLOCK = 1024

# Pinned method id (spec Sec 14.4.4): exact top-n_g cosine, no reranker. A
# future two-stage method gets its own id here rather than colliding with
# this one.
POOL_METHOD_ID = "cosine_topk_v1"


def _feature_pack_digest(f0_matrix: NDArray[np.float32], node_ids: Sequence[str]) -> str:
    """SHA-256 over the ordered feature bytes plus the ordered node-id rows.

    Catches "mutated features, same ids" (spec Sec 14.4.4): any change to the
    feature values behind a pool, under an unchanged ordered node-id list,
    changes this digest and therefore the cache's `pool_method_hash`.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(f0_matrix, dtype=np.float32).tobytes())
    for node in node_ids:
        digest.update(f"{node}\n".encode())
    return digest.hexdigest()


def _pool_method_hash(
    *,
    n_ground: int,
    role_universe: str,
    feature_digest: str,
    shortlist_m: int | None,
) -> str:
    """SHA-256 over the pinned, order-stable pool-method identity (spec Sec 14.4.4).

    Covers, in this fixed order: the method id, `n_ground`, the shortlist `M`
    (omitted entirely -- not even as an explicit "absent" marker -- when
    `None`, since there is no reranker in rev-3.1 and a future two-stage
    method must not collide with this hash by having its `M` coincide with
    an encoded absence), the ordered F0/feature-pack digest, and the
    role-universe identity.
    """
    rows = [f"method\t{POOL_METHOD_ID}\n", f"n_ground\t{n_ground}\n"]
    if shortlist_m is not None:
        rows.append(f"M\t{shortlist_m}\n")
    rows.append(f"feature_digest\t{feature_digest}\n")
    rows.append(f"role_universe\t{role_universe}\n")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode())
    return digest.hexdigest()


def build_grounding_pool(
    f0_matrix: NDArray[np.float32],
    node_ids: Sequence[str],
    *,
    n_ground: int,
    role_universe: str,
    cache_path: Path | None = None,
) -> dict[str, list[str]]:
    """Compute (or load) the per-node grounding pools.

    Args:
        f0_matrix: Shape ``(N, d)`` F0 rows for exactly the split side's nodes,
            row-aligned with `node_ids`.
        node_ids: The side's node ids (defines both queries and candidates).
        n_ground: Pool size ``n_g``; must satisfy ``n_ground <= N - 1``.
        role_universe: The caller's role-universe identity (spec Sec 13.12:
            ``V_fit``/``V_qual``/``V_select``/``test``). Folded into the
            cache's `pool_method_hash` so two roles can never share a cache
            path undetected.
        cache_path: Optional ``.npz`` cache location.

    Returns:
        Node id -> list of ``n_ground`` neighbor node ids (cosine-descending).

    Raises:
        ValueError: On shape mismatch, an unsatisfiable `n_ground`, or a
            cache whose `pool_method_hash` does not match this call's
            method id / `n_ground` / features / role universe (fail closed;
            never silently recomputed or overwritten -- spec Sec 14.4.4).
    """
    n = len(node_ids)
    if f0_matrix.shape[0] != n:
        raise ValueError(f"f0_matrix has {f0_matrix.shape[0]} rows for {n} node ids")
    if not 1 <= n_ground <= n - 1:
        raise ValueError(f"n_ground must be in [1, {n - 1}], got {n_ground}")

    expected_hash = _pool_method_hash(
        n_ground=n_ground,
        role_universe=role_universe,
        feature_digest=_feature_pack_digest(f0_matrix, node_ids),
        shortlist_m=None,
    )

    if cache_path is not None and cache_path.exists():
        return _load_cache(cache_path, expected_hash)

    norms = np.linalg.norm(f0_matrix, axis=1, keepdims=True)
    unit = f0_matrix / np.clip(norms, 1e-12, None)
    neighbor_idx = np.empty((n, n_ground), dtype=np.int64)
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        sims = unit[start:stop] @ unit.T
        for row, i in enumerate(range(start, stop)):
            sims[row, i] = -np.inf  # self excluded
        # Descending similarity, ties by ascending node index (deterministic).
        block = sims.shape[0]
        idx = np.tile(np.arange(n, dtype=np.int64), (block, 1))
        order = np.lexsort((idx, -sims), axis=1)
        neighbor_idx[start:stop] = order[:, :n_ground]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
        try:
            np.savez_compressed(
                tmp,
                node_ids=np.array(list(node_ids)),
                neighbor_idx=neighbor_idx,
                n_ground=np.int64(n_ground),
                pool_method_hash=np.array(expected_hash),
            )
            tmp.replace(cache_path)
        finally:
            tmp.unlink(missing_ok=True)
        logger.info("wrote grounding cache to %s", cache_path)

    return {
        node: [node_ids[j] for j in neighbor_idx[i].tolist()] for i, node in enumerate(node_ids)
    }


def _load_cache(cache_path: Path, expected_hash: str) -> dict[str, list[str]]:
    """Load a cache, failing closed on any `pool_method_hash` mismatch.

    Never falls back to a silent recompute and never overwrites the cache on
    disk (spec Sec 14.4.4) -- a mismatch is always the caller's problem to
    resolve (redefine the cache path, or regenerate deliberately).
    """
    with np.load(cache_path, allow_pickle=False) as data:
        if "pool_method_hash" not in data:
            raise ValueError(
                f"grounding cache {cache_path} has no pool_method_hash field "
                f"(pre-rev-3.1 cache format); expected {expected_hash}, found <missing>"
            )
        found_hash = str(data["pool_method_hash"])
        if found_hash != expected_hash:
            raise ValueError(
                f"grounding cache {cache_path} pool_method_hash mismatch: "
                f"expected {expected_hash}, found {found_hash}"
            )
        cached_nodes = [str(x) for x in data["node_ids"].tolist()]
        neighbor_idx = data["neighbor_idx"]
    return {
        node: [cached_nodes[j] for j in neighbor_idx[i].tolist()]
        for i, node in enumerate(cached_nodes)
    }
