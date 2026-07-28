"""Tests for src.data.feature_stats: the registered V_fit standardization constants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.feature_stats import (
    FEATURE_STATS_METHOD_ID,
    compute_feature_stats,
    feature_stats_for_universe,
    load_feature_stats,
    node_ids_sha256,
    save_feature_stats,
)

pytestmark = pytest.mark.unit


def _rows(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    gen = np.random.default_rng(seed)
    base = 40.0 * gen.standard_normal(d)
    return (base + 3.0 * gen.standard_normal((n, d))).astype(np.float32)


class TestComputeFeatureStats:
    def test_matches_population_moments(self) -> None:
        rows = _rows(64, 12)
        stats = compute_feature_stats(rows, [f"n{i}" for i in range(64)])

        expected_mu = rows.astype(np.float64).mean(axis=0).astype(np.float32)
        expected_sigma = rows.astype(np.float64).std(axis=0, ddof=0).astype(np.float32)
        np.testing.assert_allclose(stats.mu, expected_mu, rtol=0, atol=1e-4)
        np.testing.assert_allclose(stats.sigma, expected_sigma, rtol=0, atol=1e-4)
        assert stats.method_id == FEATURE_STATS_METHOD_ID
        assert stats.n_rows == 64
        assert stats.node_ids_sha256 == node_ids_sha256([f"n{i}" for i in range(64)])
        assert len(stats.digest) == 64

    def test_canonical_values_are_float32(self) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        assert stats.mu.dtype == np.float32
        assert stats.sigma.dtype == np.float32

    def test_constant_dimension_is_rejected_not_silently_floored(self) -> None:
        rows = _rows(8, 4)
        rows[:, 2] = 7.0
        with pytest.raises(ValueError, match="degenerate"):
            compute_feature_stats(rows, [f"n{i}" for i in range(8)])

    def test_digest_is_sensitive_to_universe_identity(self) -> None:
        rows = _rows(8, 4)
        a = compute_feature_stats(rows, [f"n{i}" for i in range(8)])
        b = compute_feature_stats(rows, [f"m{i}" for i in range(8)])
        assert a.digest != b.digest

    def test_rejects_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="node ids"):
            compute_feature_stats(_rows(8, 4), ["n0", "n1"])


class TestUniverseIsolation:
    def test_sealed_rows_in_the_matrix_do_not_change_the_statistics(self) -> None:
        fit_ids = [f"fit{i}" for i in range(16)]
        fit_rows = _rows(16, 6, seed=1)
        sealed_rows = 500.0 + _rows(24, 6, seed=2)

        fit_only = {node: i for i, node in enumerate(fit_ids)}
        alone = feature_stats_for_universe(fit_rows, fit_only, fit_ids)

        interleaved = np.zeros((40, 6), dtype=np.float32)
        index: dict[str, int] = {}
        for i, node in enumerate(fit_ids):
            interleaved[2 * i] = fit_rows[i]
            index[node] = 2 * i
        sealed_slots = [r for r in range(40) if r not in set(index.values())]
        for slot, row in zip(sealed_slots, sealed_rows, strict=False):
            interleaved[slot] = row
        mixed = feature_stats_for_universe(interleaved, index, fit_ids)

        assert mixed.digest == alone.digest
        np.testing.assert_array_equal(mixed.mu, alone.mu)
        np.testing.assert_array_equal(mixed.sigma, alone.sigma)

    def test_row_order_of_the_universe_list_is_load_bearing_for_identity(self) -> None:
        ids = [f"fit{i}" for i in range(8)]
        rows = _rows(8, 4, seed=3)
        index = {node: i for i, node in enumerate(ids)}
        forward = feature_stats_for_universe(rows, index, ids)
        reversed_ = feature_stats_for_universe(rows, index, list(reversed(ids)))

        np.testing.assert_allclose(forward.mu, reversed_.mu, rtol=0, atol=1e-5)
        assert forward.digest != reversed_.digest


class TestCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        loaded = load_feature_stats(path, expected_node_ids_sha256=stats.node_ids_sha256)

        assert loaded.digest == stats.digest
        np.testing.assert_array_equal(loaded.mu, stats.mu)
        np.testing.assert_array_equal(loaded.sigma, stats.sigma)

    def test_load_fails_closed_on_a_foreign_universe(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        with pytest.raises(ValueError, match="universe"):
            load_feature_stats(path, expected_node_ids_sha256="0" * 64)

    def test_load_fails_closed_on_a_tampered_payload(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        payload = dict(np.load(path, allow_pickle=False))
        payload["mu"] = payload["mu"] + np.float32(1.0)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="digest"):
            load_feature_stats(path)

    def test_universe_helper_reuses_a_matching_cache(self, tmp_path: Path) -> None:
        ids = [f"n{i}" for i in range(8)]
        rows = _rows(8, 4)
        index = {node: i for i, node in enumerate(ids)}
        path = tmp_path / "feature_stats.npz"
        first = feature_stats_for_universe(rows, index, ids, cache_path=path)
        assert path.is_file()
        second = feature_stats_for_universe(rows, index, ids, cache_path=path)
        assert second.digest == first.digest

    def test_universe_helper_fails_closed_on_a_stale_cache(self, tmp_path: Path) -> None:
        ids = [f"n{i}" for i in range(8)]
        rows = _rows(8, 4)
        index = {node: i for i, node in enumerate(ids)}
        path = tmp_path / "feature_stats.npz"
        feature_stats_for_universe(rows, index, ids, cache_path=path)
        other = [f"z{i}" for i in range(8)]
        other_index = {n: i for i, n in enumerate(other)}
        with pytest.raises(ValueError, match="universe"):
            feature_stats_for_universe(rows, other_index, other, cache_path=path)
