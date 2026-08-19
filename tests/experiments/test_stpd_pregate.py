"""Tests for `src/experiments/stpd_pregate.py`'s corruption core."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.experiments.stpd_pregate import SwapProvenance, _stable_seed, provenance_swaps

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- fixtures / helpers


def _region_edges(n: int = 60, k: int = 6, p: float = 0.3, seed: int = 0) -> list[tuple[int, int]]:
    """A deterministic connected Watts-Strogatz graph's edges, relabeled and normalized."""
    graph = nx.connected_watts_strogatz_graph(n, k, p, seed=seed)
    graph = nx.convert_node_labels_to_integers(graph)
    edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    return sorted(edges)


def _degrees(edges: list[tuple[int, int]] | np.ndarray, n: int) -> np.ndarray:
    """Per-node degree of an edge list/array over `n` nodes."""
    deg = np.zeros(n, dtype=np.int64)
    for i, j in edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    return deg


def _as_tuple_set(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in arr}


# --------------------------------------------------------------------------- _stable_seed


def test_stable_seed_deterministic_and_part_sensitive() -> None:
    a = _stable_seed("region-0", "severity-0.15")
    b = _stable_seed("region-0", "severity-0.15")
    c = _stable_seed("region-0", "severity-0.30")
    assert a == b
    assert a != c
    assert isinstance(a, int)
    assert 0 <= a < 2**64


# --------------------------------------------------------------------------- provenance exactness


def test_provenance_exactness() -> None:
    edges = _region_edges()
    rng = np.random.default_rng(_stable_seed("exactness"))
    prov = provenance_swaps(edges, fraction=0.15, rng=rng)
    assert prov is not None

    original = set(edges)
    deleted = _as_tuple_set(prov.deleted)
    inserted = _as_tuple_set(prov.inserted)
    kept = _as_tuple_set(prov.kept)

    corrupted = (original - deleted) | inserted

    assert deleted <= original
    assert inserted.isdisjoint(original)
    assert inserted.isdisjoint(deleted)
    assert corrupted == (original - deleted) | inserted
    assert kept == original - deleted

    # kept preserves original order.
    kept_rows = [tuple(int(x) for x in row) for row in prov.kept]
    expected_kept_order = [e for e in edges if e not in deleted]
    assert kept_rows == expected_kept_order

    # Every quad has 4 distinct nodes, and quads reconstruct deleted/inserted exactly.
    for s, (i, j, k, m) in enumerate(prov.quads.tolist()):
        assert len({i, j, k, m}) == 4
        d0 = tuple(sorted((i, j)))
        d1 = tuple(sorted((k, m)))
        ins0 = tuple(sorted((i, m)))
        ins1 = tuple(sorted((k, j)))
        assert tuple(int(x) for x in prov.deleted[2 * s]) == d0
        assert tuple(int(x) for x in prov.deleted[2 * s + 1]) == d1
        assert tuple(int(x) for x in prov.inserted[2 * s]) == ins0
        assert tuple(int(x) for x in prov.inserted[2 * s + 1]) == ins1


# --------------------------------------------------------------------------- degree preservation


@pytest.mark.parametrize("fraction", [0.05, 0.15, 0.30])
def test_degree_preservation(fraction: float) -> None:
    edges = _region_edges()
    n = max(max(e) for e in edges) + 1
    rng = np.random.default_rng(_stable_seed("degree", str(fraction)))
    prov = provenance_swaps(edges, fraction=fraction, rng=rng)
    assert prov is not None

    corrupted = list(_as_tuple_set(prov.kept) | _as_tuple_set(prov.inserted))
    original_degree = _degrees(edges, n)
    corrupted_degree = _degrees(corrupted, n)
    np.testing.assert_array_equal(original_degree, corrupted_degree)


# --------------------------------------------------------------------------- determinism / drop


def test_determinism_same_seed_byte_identical() -> None:
    edges = _region_edges()
    seed = _stable_seed("determinism")
    prov_a = provenance_swaps(edges, fraction=0.2, rng=np.random.default_rng(seed))
    prov_b = provenance_swaps(edges, fraction=0.2, rng=np.random.default_rng(seed))
    assert prov_a is not None
    assert prov_b is not None
    np.testing.assert_array_equal(prov_a.quads, prov_b.quads)
    np.testing.assert_array_equal(prov_a.deleted, prov_b.deleted)
    np.testing.assert_array_equal(prov_a.inserted, prov_b.inserted)
    np.testing.assert_array_equal(prov_a.kept, prov_b.kept)


def test_fraction_zero_returns_none() -> None:
    edges = _region_edges()
    rng = np.random.default_rng(_stable_seed("zero-fraction"))
    assert provenance_swaps(edges, fraction=0.0, rng=rng) is None


def test_tiny_graph_n_target_zero_returns_none() -> None:
    # round(0.1 * 3) == 0.
    edges = [(0, 1), (1, 2), (0, 2)]
    rng = np.random.default_rng(_stable_seed("tiny-graph"))
    assert provenance_swaps(edges, fraction=0.1, rng=rng) is None


def test_infeasible_triangle_returns_none_never_partial() -> None:
    # A triangle: only 3 nodes total, so no swap can ever touch 4 distinct nodes.
    edges = [(0, 1), (0, 2), (1, 2)]
    rng = np.random.default_rng(_stable_seed("infeasible-triangle"))
    assert provenance_swaps(edges, fraction=1.0, rng=rng) is None


# --------------------------------------------------------------------------- SwapProvenance


def test_swap_provenance_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="int64"):
        SwapProvenance(
            quads=np.zeros((1, 4), dtype=np.int32),
            deleted=np.zeros((2, 2), dtype=np.int64),
            inserted=np.zeros((2, 2), dtype=np.int64),
            kept=np.zeros((0, 2), dtype=np.int64),
        )


def test_swap_provenance_rejects_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="rows"):
        SwapProvenance(
            quads=np.zeros((1, 4), dtype=np.int64),
            deleted=np.zeros((1, 2), dtype=np.int64),  # should be 2
            inserted=np.zeros((2, 2), dtype=np.int64),
            kept=np.zeros((0, 2), dtype=np.int64),
        )


def test_swap_provenance_rejects_i_ge_j() -> None:
    with pytest.raises(ValueError, match="i < j"):
        SwapProvenance(
            quads=np.zeros((0, 4), dtype=np.int64),
            deleted=np.zeros((0, 2), dtype=np.int64),
            inserted=np.zeros((0, 2), dtype=np.int64),
            kept=np.array([[2, 1]], dtype=np.int64),
        )
