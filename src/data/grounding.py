"""Grounding pools: exact top-``n_g`` cosine neighbors in F0 space (spec Sec 13.12).

``G(u)`` is computed within the node's **own split side** (the caller passes
that side's node list), self excluded, from the frozen F0 mean-pool matrix.
Exact blockwise matmul — the operative node set is ~10k, no ANN needed.
Deterministic: cosine ties break by ascending node index. Optionally cached to
an ``.npz`` validated on reload (node set + ``n_ground`` must match).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_BLOCK = 1024


def build_grounding_pool(
    f0_matrix: NDArray[np.float32],
    node_ids: Sequence[str],
    *,
    n_ground: int,
    cache_path: Path | None = None,
) -> dict[str, list[str]]:
    """Compute (or load) the per-node grounding pools.

    Args:
        f0_matrix: Shape ``(N, d)`` F0 rows for exactly the split side's nodes,
            row-aligned with `node_ids`.
        node_ids: The side's node ids (defines both queries and candidates).
        n_ground: Pool size ``n_g``; must satisfy ``n_ground <= N - 1``.
        cache_path: Optional ``.npz`` cache location.

    Returns:
        Node id -> list of ``n_ground`` neighbor node ids (cosine-descending).

    Raises:
        ValueError: On shape mismatch or an unsatisfiable `n_ground`.
    """
    n = len(node_ids)
    if f0_matrix.shape[0] != n:
        raise ValueError(f"f0_matrix has {f0_matrix.shape[0]} rows for {n} node ids")
    if not 1 <= n_ground <= n - 1:
        raise ValueError(f"n_ground must be in [1, {n - 1}], got {n_ground}")

    if cache_path is not None and cache_path.exists():
        cached = _load_cache(cache_path, node_ids, n_ground)
        if cached is not None:
            return cached
        logger.warning("grounding cache %s is stale; rebuilding", cache_path)

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
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            node_ids=np.array(list(node_ids)),
            neighbor_idx=neighbor_idx,
            n_ground=np.int64(n_ground),
        )
        tmp.replace(cache_path)
        logger.info("wrote grounding cache to %s", cache_path)

    return {
        node: [node_ids[j] for j in neighbor_idx[i].tolist()] for i, node in enumerate(node_ids)
    }


def _load_cache(
    cache_path: Path, node_ids: Sequence[str], n_ground: int
) -> dict[str, list[str]] | None:
    """Load a cache if it matches the requested node set and pool size."""
    with np.load(cache_path, allow_pickle=False) as data:
        cached_nodes = [str(x) for x in data["node_ids"].tolist()]
        if cached_nodes != list(node_ids) or int(data["n_ground"]) != n_ground:
            return None
        neighbor_idx = data["neighbor_idx"]
    return {
        node: [cached_nodes[j] for j in neighbor_idx[i].tolist()]
        for i, node in enumerate(cached_nodes)
    }
