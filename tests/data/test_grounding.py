"""Tests for src.data.grounding: exact cosine grounding pools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.grounding import build_grounding_pool

pytestmark = pytest.mark.unit

_NODES = ["a", "b", "c", "d"]
# Unit-direction design: a ~ b (cos 1 along x), c orthogonal, d anti-parallel.
_F0 = np.array(
    [
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=np.float32,
)


class TestBuildGroundingPool:
    def test_exact_top_k_with_self_excluded(self) -> None:
        pool = build_grounding_pool(_F0, _NODES, n_ground=2)
        assert pool["a"] == ["b", "c"]  # cos: b=1.0, c=0.0, d=-1.0
        assert pool["b"] == ["a", "c"]
        assert "c" not in pool["a"][:1]

    def test_cosine_ties_break_by_node_index(self) -> None:
        f0 = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pool = build_grounding_pool(f0, ["x", "y", "z"], n_ground=2)
        assert pool["x"] == ["y", "z"]
        assert pool["y"] == ["x", "z"]
        assert pool["z"] == ["x", "y"]

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        f0 = rng.normal(size=(30, 5)).astype(np.float32)
        nodes = [f"n{i}" for i in range(30)]
        assert build_grounding_pool(f0, nodes, n_ground=4) == build_grounding_pool(
            f0.copy(), list(nodes), n_ground=4
        )

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            build_grounding_pool(_F0, _NODES[:3], n_ground=2)
        with pytest.raises(ValueError, match="n_ground"):
            build_grounding_pool(_F0, _NODES, n_ground=4)

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        cache = tmp_path / "pool.npz"
        first = build_grounding_pool(_F0, _NODES, n_ground=2, cache_path=cache)
        assert cache.exists()
        second = build_grounding_pool(np.zeros_like(_F0), _NODES, n_ground=2, cache_path=cache)
        # Cache hit: the zeroed matrix is ignored, cached pools returned.
        assert first == second

    def test_stale_cache_rebuilt(self, tmp_path: Path) -> None:
        cache = tmp_path / "pool.npz"
        build_grounding_pool(_F0, _NODES, n_ground=2, cache_path=cache)
        rebuilt = build_grounding_pool(_F0, _NODES, n_ground=3, cache_path=cache)
        assert all(len(v) == 3 for v in rebuilt.values())

    def test_blockwise_matches_direct(self) -> None:
        rng = np.random.default_rng(1)
        n = 50
        f0 = rng.normal(size=(n, 8)).astype(np.float32)
        nodes = [f"n{i:02d}" for i in range(n)]
        pool = build_grounding_pool(f0, nodes, n_ground=5)
        unit = f0 / np.linalg.norm(f0, axis=1, keepdims=True)
        sims = unit @ unit.T
        np.fill_diagonal(sims, -np.inf)
        for i, node in enumerate(nodes):
            expected_idx = np.lexsort((np.arange(n), -sims[i]))[:5]
            assert pool[node] == [nodes[j] for j in expected_idx.tolist()]
