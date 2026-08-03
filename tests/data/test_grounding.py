"""Tests for src.data.grounding: cosine pools and pool_method_hash cache binding."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from src.data.grounding import build_grounding_pool
from src.model.egostitch.config import E2EConfig

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
        pool = build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit")
        assert pool["a"] == ["b", "c"]  # cos: b=1.0, c=0.0, d=-1.0
        assert pool["b"] == ["a", "c"]
        assert "c" not in pool["a"][:1]

    def test_cosine_ties_break_by_node_index(self) -> None:
        f0 = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pool = build_grounding_pool(f0, ["x", "y", "z"], n_ground=2, role_universe="V_fit")
        assert pool["x"] == ["y", "z"]
        assert pool["y"] == ["x", "z"]
        assert pool["z"] == ["x", "y"]

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        f0 = rng.normal(size=(30, 5)).astype(np.float32)
        nodes = [f"n{i}" for i in range(30)]
        assert build_grounding_pool(
            f0, nodes, n_ground=4, role_universe="V_fit"
        ) == build_grounding_pool(f0.copy(), list(nodes), n_ground=4, role_universe="V_fit")

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            build_grounding_pool(_F0, _NODES[:3], n_ground=2, role_universe="V_fit")
        with pytest.raises(ValueError, match="n_ground"):
            build_grounding_pool(_F0, _NODES, n_ground=4, role_universe="V_fit")

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        cache = tmp_path / "pool.npz"
        first = build_grounding_pool(
            _F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache
        )
        assert cache.exists()
        second = build_grounding_pool(
            _F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache
        )
        assert first == second

    def test_concurrent_cache_publication_uses_unique_temp_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "pool.npz"
        barrier = threading.Barrier(2)
        original_save = np.savez_compressed
        save = cast(Callable[..., Any], original_save)

        def synchronized_save(path: Path, **kwargs: object) -> None:
            save(path, **kwargs)
            barrier.wait(timeout=5)

        monkeypatch.setattr(np, "savez_compressed", synchronized_save)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    build_grounding_pool,
                    _F0,
                    _NODES,
                    n_ground=2,
                    role_universe="V_fit",
                    cache_path=cache,
                )
                for _ in range(2)
            ]
        results = [future.result() for future in futures]

        assert results[0] == results[1]
        assert (
            build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache)
            == results[0]
        )
        assert not list(tmp_path.glob("pool.tmp-*.npz"))

    def test_blockwise_matches_direct(self) -> None:
        rng = np.random.default_rng(1)
        n = 50
        f0 = rng.normal(size=(n, 8)).astype(np.float32)
        nodes = [f"n{i:02d}" for i in range(n)]
        pool = build_grounding_pool(f0, nodes, n_ground=5, role_universe="V_fit")
        unit = f0 / np.linalg.norm(f0, axis=1, keepdims=True)
        sims = unit @ unit.T
        np.fill_diagonal(sims, -np.inf)
        for i, node in enumerate(nodes):
            expected_idx = np.lexsort((np.arange(n), -sims[i]))[:5]
            assert pool[node] == [nodes[j] for j in expected_idx.tolist()]


class TestPoolMethodHashCacheBinding:
    """Spec Sec 14.4.4: pool_method_hash fail-closed cache validation."""

    def test_pool_method_hash_deterministic_across_builds(self, tmp_path: Path) -> None:
        """Two builds on identical inputs give byte-identical pools and the same hash."""
        cache_a = tmp_path / "a.npz"
        cache_b = tmp_path / "b.npz"
        build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache_a)
        build_grounding_pool(
            _F0.copy(), list(_NODES), n_ground=2, role_universe="V_fit", cache_path=cache_b
        )
        with np.load(cache_a) as data_a, np.load(cache_b) as data_b:
            assert str(data_a["pool_method_hash"]) == str(data_b["pool_method_hash"])
            np.testing.assert_array_equal(data_a["neighbor_idx"], data_b["neighbor_idx"])

    def test_stale_method_rejected_on_n_ground_change(self, tmp_path: Path) -> None:
        """A cache written with a different `n_ground` is rejected, not silently rebuilt."""
        cache = tmp_path / "pool.npz"
        build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache)
        before = cache.read_bytes()
        with pytest.raises(ValueError, match="pool_method_hash"):
            build_grounding_pool(_F0, _NODES, n_ground=3, role_universe="V_fit", cache_path=cache)
        # Not overwritten: the cache on disk is byte-identical to before the
        # rejected call.
        assert cache.read_bytes() == before

    def test_mutated_features_same_ids_rejected(self, tmp_path: Path) -> None:
        """Mutated F0 features under an unchanged node-id list are caught, not served stale."""
        cache = tmp_path / "pool.npz"
        build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache)
        before = cache.read_bytes()
        with pytest.raises(ValueError, match="pool_method_hash"):
            build_grounding_pool(
                np.zeros_like(_F0), _NODES, n_ground=2, role_universe="V_fit", cache_path=cache
            )
        assert cache.read_bytes() == before

    def test_role_isolation_rejects_cross_role_cache_reuse(self, tmp_path: Path) -> None:
        """Two different role universes cannot share the same cache path undetected."""
        cache = tmp_path / "pool.npz"
        build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache)
        with pytest.raises(ValueError, match="pool_method_hash"):
            build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_qual", cache_path=cache)

    def test_role_isolation_produces_different_hashes(self, tmp_path: Path) -> None:
        """The same node set + n_ground under two role identities hash differently."""
        cache_fit = tmp_path / "fit.npz"
        cache_qual = tmp_path / "qual.npz"
        build_grounding_pool(
            _F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache_fit
        )
        build_grounding_pool(
            _F0, _NODES, n_ground=2, role_universe="V_qual", cache_path=cache_qual
        )
        with np.load(cache_fit) as data_fit, np.load(cache_qual) as data_qual:
            assert str(data_fit["pool_method_hash"]) != str(data_qual["pool_method_hash"])

    def test_missing_pool_method_hash_field_rejected(self, tmp_path: Path) -> None:
        """A pre-rev-3.1 (v2) cache with no pool_method_hash field is rejected, not tolerated."""
        cache = tmp_path / "legacy.npz"
        np.savez_compressed(
            cache,
            node_ids=np.array(list(_NODES)),
            neighbor_idx=np.zeros((len(_NODES), 2), dtype=np.int64),
            n_ground=np.int64(2),
        )
        with pytest.raises(ValueError, match="pool_method_hash"):
            build_grounding_pool(_F0, _NODES, n_ground=2, role_universe="V_fit", cache_path=cache)

    def test_own_split_side_legality(self) -> None:
        """A node's pool contains only nodes from its own role universe, never self."""
        rng = np.random.default_rng(2)
        nodes = [f"m{i}" for i in range(10)]
        f0 = rng.normal(size=(len(nodes), 4)).astype(np.float32)
        pool = build_grounding_pool(f0, nodes, n_ground=3, role_universe="V_select")
        node_set = set(nodes)
        for node, neighbors in pool.items():
            assert node in node_set
            assert set(neighbors) <= node_set
            assert node not in neighbors


class TestE2EConfigNGroundReachesPoolBuilder:
    def test_legacy_model_config_defaults_n_ground_to_20(self) -> None:
        """An absent checkpoint/config key preserves the legacy n_ground=20 semantics."""
        assert E2EConfig.from_mapping({}).generator.n_ground == 20
        assert E2EConfig.from_mapping({"generator": {"n_ground": 50}}).generator.n_ground == 50

    def test_e2e_config_n_ground_round_trips_through_yaml_and_reaches_pool_builder(self) -> None:
        """`generator.n_ground: 20` parsed from a YAML-style mapping sizes real pools."""
        cfg = E2EConfig.from_mapping({"generator": {"n_ground": 20}})
        assert cfg.generator.n_ground == 20
        rng = np.random.default_rng(3)
        nodes = [f"p{i}" for i in range(25)]
        f0 = rng.normal(size=(len(nodes), 6)).astype(np.float32)
        pool = build_grounding_pool(
            f0, nodes, n_ground=cfg.generator.n_ground, role_universe="V_fit"
        )
        assert all(len(v) == 20 for v in pool.values())
