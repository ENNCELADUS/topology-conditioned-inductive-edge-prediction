"""Tests for `src/experiments/stpd_pregate.py`'s corruption core and corpus layer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from src.data.features import FeatureStore
from src.distill.heuristic_targets import heuristic_score
from src.experiments.stpd_pregate import (
    PregateConfig,
    SwapProvenance,
    _stable_seed,
    build_cell_context,
    build_pregate_corpus,
    corrupted_edges_of,
    load_pregate_corpus,
    pair_context_features,
    provenance_swaps,
    save_pregate_corpus,
    structure_scores,
)

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


# --------------------------------------------------------------------------- corpus fixtures


def _pregate_train_graph(n: int = 60, k: int = 6, p: float = 0.3, seed: int = 0) -> nx.Graph:
    """A deterministic connected Watts-Strogatz graph with `node_%06d` ids."""
    base = nx.connected_watts_strogatz_graph(n, k, p, seed=seed)
    return nx.relabel_nodes(base, {i: f"node_{i:06d}" for i in base.nodes()})


def _write_feature_store(root: Path, node_ids: list[str], *, input_dim: int = 8) -> FeatureStore:
    """A tiny synthetic `FeatureStore` root covering every id in `node_ids`."""
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((5, input_dim)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": input_dim, "max_sequence_length": 1024}
        )
    )
    return FeatureStore(root)


@pytest.fixture
def _pregate_setup(tmp_path: Path) -> tuple[nx.Graph, FeatureStore, PregateConfig, Path]:
    graph = _pregate_train_graph()
    store = _write_feature_store(tmp_path / "features", sorted(graph.nodes()))
    cfg = PregateConfig(sizes=(12, 16), per_size=4, holdout_frac=0.25, salt="test_pregate_salt")
    cache_dir = tmp_path / "cache"
    return graph, store, cfg, cache_dir


# --------------------------------------------------------------------------- PregateCorpus build


def test_build_pregate_corpus_degree_preserved(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)

    assert len(corpus.regions.regions) > 0
    assert len(corpus.provenance) + len(corpus.dropped_cells) == len(corpus.regions.regions) * len(
        cfg.severities
    )

    for (region_idx, _severity), prov in corpus.provenance.items():
        n_nodes = len(corpus.regions.regions[region_idx])
        original_edges = [(int(i), int(j)) for i, j in corpus.regions.edges[region_idx].tolist()]
        original_degree = _degrees(original_edges, n_nodes)

        corrupted_edges = list(_as_tuple_set(prov.kept) | _as_tuple_set(prov.inserted))
        corrupted_degree = _degrees(corrupted_edges, n_nodes)
        np.testing.assert_array_equal(original_degree, corrupted_degree)


def test_holdout_disjointness(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)
    regions = corpus.regions

    train_set = set(regions.train_idx)
    val_set = set(regions.val_idx)
    assert train_set.isdisjoint(val_set)

    for size in cfg.sizes:
        size_indices = {i for i, region in enumerate(regions.regions) if len(region) == size}
        assert size_indices  # sanity: this fixture yields regions of every configured size
        assert (train_set | val_set) & size_indices == size_indices


def test_build_pregate_corpus_deterministic(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus_a = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir / "a")
    corpus_b = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir / "b")

    assert corpus_a.dropped_cells == corpus_b.dropped_cells
    assert set(corpus_a.provenance.keys()) == set(corpus_b.provenance.keys())
    for key, prov_a in corpus_a.provenance.items():
        prov_b = corpus_b.provenance[key]
        np.testing.assert_array_equal(prov_a.quads, prov_b.quads)
        np.testing.assert_array_equal(prov_a.deleted, prov_b.deleted)
        np.testing.assert_array_equal(prov_a.inserted, prov_b.inserted)
        np.testing.assert_array_equal(prov_a.kept, prov_b.kept)


def test_save_load_round_trip(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path], tmp_path: Path
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)

    path = tmp_path / "pregate_corpus.pt"
    save_pregate_corpus(corpus, path)
    loaded = load_pregate_corpus(path)

    assert loaded.dropped_cells == corpus.dropped_cells
    assert set(loaded.provenance.keys()) == set(corpus.provenance.keys())
    for key, prov in corpus.provenance.items():
        loaded_prov = loaded.provenance[key]
        np.testing.assert_array_equal(loaded_prov.quads, prov.quads)
        np.testing.assert_array_equal(loaded_prov.deleted, prov.deleted)
        np.testing.assert_array_equal(loaded_prov.inserted, prov.inserted)
        np.testing.assert_array_equal(loaded_prov.kept, prov.kept)

    assert loaded.regions.node_ids == corpus.regions.node_ids
    assert torch.equal(loaded.regions.features, corpus.regions.features)
    assert loaded.regions.regions == corpus.regions.regions
    for loaded_edges, orig_edges in zip(loaded.regions.edges, corpus.regions.edges, strict=True):
        assert torch.equal(loaded_edges, orig_edges)
    assert loaded.regions.train_idx == corpus.regions.train_idx
    assert loaded.regions.val_idx == corpus.regions.val_idx
    assert loaded.regions.dropped_featureless_regions == corpus.regions.dropped_featureless_regions


# --------------------------------------------------------------------------- cell context fixtures


def _corrupted_cell(
    seed_label: str, *, fraction: float = 0.15, d: int = 5
) -> tuple[int, NDArray[np.int64], SwapProvenance, torch.Tensor]:
    """A corrupted cell (n, corrupted edges, provenance, random fp32 features)."""
    edges = _region_edges()
    n = max(max(e) for e in edges) + 1
    rng = np.random.default_rng(_stable_seed(seed_label))
    prov = provenance_swaps(edges, fraction=fraction, rng=rng)
    assert prov is not None
    corrupted_edges = corrupted_edges_of(prov)
    feat_rng = np.random.default_rng(0)
    features = torch.tensor(feat_rng.standard_normal((n, d)), dtype=torch.float32)
    return n, corrupted_edges, prov, features


def _sample_pairs(
    prov: SwapProvenance, corrupted_edges: NDArray[np.int64]
) -> list[tuple[int, int]]:
    """Sample pairs including one PRESENT (inserted) and one ABSENT (deleted) pair."""
    inserted_pair = (int(prov.inserted[0, 0]), int(prov.inserted[0, 1]))
    deleted_pair = (int(prov.deleted[0, 0]), int(prov.deleted[0, 1]))
    extra = [(int(i), int(j)) for i, j in corrupted_edges[: min(5, len(corrupted_edges))]]

    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for pair in (inserted_pair, deleted_pair, *extra):
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


# --------------------------------------------------------------------------- corrupted_edges_of


def test_corrupted_edges_of_is_kept_union_inserted() -> None:
    _n, corrupted_edges, prov, _features = _corrupted_cell("corrupted-edges-of")
    expected = _as_tuple_set(prov.kept) | _as_tuple_set(prov.inserted)
    assert _as_tuple_set(corrupted_edges) == expected
    assert corrupted_edges.dtype == np.int64
    assert corrupted_edges.shape[1] == 2


# --------------------------------------------------------------------------- pair_context_features


def test_pair_context_features_matches_naive_recompute() -> None:
    n, corrupted_edges, prov, features = _corrupted_cell("context-equivalence", d=5)
    d = features.shape[1]
    ctx = build_cell_context(n, corrupted_edges, features)

    sample_pairs = _sample_pairs(prov, corrupted_edges)
    pairs_tensor = torch.tensor(sample_pairs, dtype=torch.long)
    actual = pair_context_features(ctx, pairs_tensor)

    corrupted_graph = nx.Graph()
    corrupted_graph.add_nodes_from(range(n))
    corrupted_graph.add_edges_from((int(i), int(j)) for i, j in corrupted_edges)

    expected_rows = []
    for u, v in sample_pairs:
        g = corrupted_graph.copy()
        if g.has_edge(u, v):
            g.remove_edge(u, v)
        deg_u = g.degree(u)
        deg_v = g.degree(v)
        common = set(nx.common_neighbors(g, u, v))
        cn = len(common)
        ra = sum(1.0 / g.degree(w) for w in common if g.degree(w) > 1)
        if deg_u > 0:
            mean_u = sum((features[w] for w in g.neighbors(u)), torch.zeros(d)) / deg_u
        else:
            mean_u = torch.zeros(d)
        if deg_v > 0:
            mean_v = sum((features[w] for w in g.neighbors(v)), torch.zeros(d)) / deg_v
        else:
            mean_v = torch.zeros(d)
        row = torch.cat(
            [
                torch.tensor(
                    [math.log1p(deg_u), math.log1p(deg_v), math.log1p(cn), ra],
                    dtype=torch.float32,
                ),
                mean_u,
                mean_v,
            ]
        )
        expected_rows.append(row)
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_structure_scores_ra_matches_heuristic_targets_reference() -> None:
    n, corrupted_edges, prov, features = _corrupted_cell("context-ra-reference", d=3)
    ctx = build_cell_context(n, corrupted_edges, features)

    node_names = [f"n{i}" for i in range(n)]
    graph = nx.Graph()
    graph.add_nodes_from(node_names)
    graph.add_edges_from((node_names[int(i)], node_names[int(j)]) for i, j in corrupted_edges)

    sample_pairs = _sample_pairs(prov, corrupted_edges)
    pairs_tensor = torch.tensor(sample_pairs, dtype=torch.long)
    ra_scores, _cn_scores = structure_scores(ctx, pairs_tensor)

    for row, (u, v) in enumerate(sample_pairs):
        expected_ra = heuristic_score(graph, node_names[u], node_names[v], "ra")
        assert ra_scores[row].item() == pytest.approx(expected_ra, abs=1e-5)


def test_pair_context_features_masked_degree_zero_isolates_endpoint() -> None:
    # Node 1's only edge is (0, 1): masking the queried pair isolates it.
    n = 4
    corrupted_edges = np.array([[0, 1], [0, 2], [0, 3], [2, 3]], dtype=np.int64)
    d = 3
    feat_rng = np.random.default_rng(2)
    features = torch.tensor(feat_rng.standard_normal((n, d)), dtype=torch.float32)
    ctx = build_cell_context(n, corrupted_edges, features)

    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    result = pair_context_features(ctx, pairs)

    masked_deg_v = result[0, 1]
    mean_v = result[0, 4 + d : 4 + 2 * d]
    assert masked_deg_v.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(mean_v, torch.zeros(d), atol=1e-6)
