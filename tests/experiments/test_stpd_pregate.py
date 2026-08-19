"""Tests for `src/experiments/stpd_pregate.py`'s corruption core and corpus layer."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from src import score_universe
from src.data.features import FeatureStore
from src.distill.heuristic_targets import heuristic_score
from src.experiments.stpd_pregate import (
    B0Scores,
    PregateConfig,
    PregateProbe,
    SwapProvenance,
    _stable_seed,
    bucketed_macro,
    build_cell_context,
    build_pregate_corpus,
    collect_eval_pairs,
    corrupted_edges_of,
    load_b0_scores,
    load_pregate_corpus,
    pair_context_features,
    paired_accuracy,
    probe_pair_inputs,
    provenance_swaps,
    quad_comparison_pairs,
    save_b0_scores,
    save_pregate_corpus,
    score_b0_pairs,
    structure_scores,
    train_probe,
    trusted_rows,
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


# --------------------------------------------------------------------------- PregateProbe


def test_pregate_probe_shape_and_dtype() -> None:
    probe = PregateProbe(in_dim=10)
    x = torch.randn(7, 10, dtype=torch.float32)
    out = probe(x)
    assert out.shape == (7,)
    assert out.dtype == torch.float32


# --------------------------------------------------------------------------- probe_pair_inputs


def test_probe_pair_inputs_variant_p_shape_and_values() -> None:
    d = 4
    features = torch.arange(3 * d, dtype=torch.float32).reshape(3, d)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    out = probe_pair_inputs(features, pairs, None)
    assert out.shape == (2, 3 * d)

    x_u, x_v = features[0], features[1]
    expected_row0 = torch.cat([x_u + x_v, x_u * x_v, (x_u - x_v).abs()])
    torch.testing.assert_close(out[0], expected_row0)


def test_probe_pair_inputs_variant_s_concatenates_context() -> None:
    n, corrupted_edges, prov, features = _corrupted_cell("probe-pair-inputs-s", d=4)
    d = features.shape[1]
    ctx = build_cell_context(n, corrupted_edges, features)
    sample_pairs = _sample_pairs(prov, corrupted_edges)
    pairs_tensor = torch.tensor(sample_pairs, dtype=torch.long)

    out = probe_pair_inputs(features, pairs_tensor, ctx)
    expected_ctx = pair_context_features(ctx, pairs_tensor)
    assert out.shape == (len(sample_pairs), 3 * d + 4 + 2 * d)
    torch.testing.assert_close(out[:, 3 * d :], expected_ctx)


# --------------------------------------------------------------------------- trusted_rows


def test_trusted_rows_labels_and_order() -> None:
    _n, _corrupted_edges, prov, _features = _corrupted_cell("trusted-rows")
    pairs, labels = trusted_rows(prov)

    n_deleted = prov.deleted.shape[0]
    n_inserted = prov.inserted.shape[0]
    n_kept = prov.kept.shape[0]
    assert pairs.shape == (n_deleted + n_inserted + n_kept, 2)
    assert pairs.dtype == np.int64
    assert labels.dtype == np.float32

    np.testing.assert_array_equal(pairs[:n_deleted], prov.deleted)
    np.testing.assert_array_equal(pairs[n_deleted : n_deleted + n_inserted], prov.inserted)
    np.testing.assert_array_equal(pairs[n_deleted + n_inserted :], prov.kept)

    assert np.all(labels[:n_deleted] == 1.0)
    assert np.all(labels[n_deleted : n_deleted + n_inserted] == 0.0)
    assert np.all(labels[n_deleted + n_inserted :] == 1.0)
    assert int(np.sum(labels == 1.0)) == n_deleted + n_kept
    assert int(np.sum(labels == 0.0)) == n_inserted


# --------------------------------------------------------------------------- quad_comparison_pairs


def test_quad_comparison_pairs_membership_and_shape() -> None:
    _n, _corrupted_edges, prov, _features = _corrupted_cell("quad-comparison")
    true_pairs, false_pairs = quad_comparison_pairs(prov)

    s = prov.quads.shape[0]
    assert true_pairs.shape == (2 * s, 2)
    assert false_pairs.shape == (2 * s, 2)
    assert true_pairs.dtype == np.int64
    assert false_pairs.dtype == np.int64

    deleted_set = _as_tuple_set(prov.deleted)
    inserted_set = _as_tuple_set(prov.inserted)
    for row in true_pairs:
        assert (int(row[0]), int(row[1])) in deleted_set
    for row in false_pairs:
        assert (int(row[0]), int(row[1])) in inserted_set

    # Exact per-quad reconstruction, aligned [A_0, B_0, A_1, B_1, ...].
    for s_idx, (i, j, k, m) in enumerate(prov.quads.tolist()):
        a_true = tuple(sorted((i, j)))
        a_false = tuple(sorted((k, j)))
        b_true = tuple(sorted((k, m)))
        b_false = tuple(sorted((i, m)))
        assert tuple(int(x) for x in true_pairs[2 * s_idx]) == a_true
        assert tuple(int(x) for x in false_pairs[2 * s_idx]) == a_false
        assert tuple(int(x) for x in true_pairs[2 * s_idx + 1]) == b_true
        assert tuple(int(x) for x in false_pairs[2 * s_idx + 1]) == b_false


# --------------------------------------------------------------------------- paired_accuracy


def test_paired_accuracy_gt_eq_lt_mean() -> None:
    scores_true = torch.tensor([1.0, 0.5, -1.0, 2.0], dtype=torch.float32)
    scores_false = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
    # rows: > (1.0), == (0.5), < (0.0), == (0.5) -> mean 0.5
    acc = paired_accuracy(scores_true, scores_false)
    assert acc == pytest.approx(0.5)


# --------------------------------------------------------------------------- bucketed_macro


def test_bucketed_macro_bucket_and_macro_means() -> None:
    per_cell = {
        (0, "light"): 0.8,
        (1, "light"): 0.6,
        (2, "light"): 1.0,  # size 64 for light: mean(0.8, 0.6) = 0.7; size 32: 1.0
        (3, "heavy"): 0.0,
        (4, "heavy"): 0.5,  # size 32 for heavy: mean(0.0, 0.5) = 0.25
    }
    cell_size = {0: 64, 1: 64, 2: 32, 3: 32, 4: 32}
    result = bucketed_macro(per_cell, cell_size)

    assert result["light|64"] == pytest.approx(0.7)
    assert result["light|32"] == pytest.approx(1.0)
    assert result["heavy|32"] == pytest.approx(0.25)
    # macro = unweighted mean over bucket values: (0.7 + 1.0 + 0.25) / 3
    assert result["macro"] == pytest.approx((0.7 + 1.0 + 0.25) / 3)


# --------------------------------------------------------------------------- train_probe


@pytest.mark.parametrize("variant", ["p", "s"])
def test_train_probe_smoke(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
    tmp_path: Path,
    variant: str,
) -> None:
    import dataclasses as dc

    graph, store, cfg, cache_dir = _pregate_setup
    cfg = dc.replace(cfg, epochs=2)
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)

    out_path = tmp_path / f"probe_{variant}.pt"
    metrics_path = tmp_path / f"metrics_{variant}.jsonl"
    result = train_probe(
        corpus,
        cfg,
        variant=variant,
        seed=0,
        device=torch.device("cpu"),
        out_path=out_path,
        metrics_path=metrics_path,
    )

    assert 0.0 <= result["best_val_macro_paired_acc"] <= 1.0
    assert result["best_epoch"] >= 1

    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line_num, line in enumerate(lines, start=1):
        record = json.loads(line)
        assert record["epoch"] == line_num
        assert math.isfinite(record["train_loss"])
        assert math.isfinite(record["val_loss"])
        assert 0.0 <= record["val_macro_paired_acc"] <= 1.0

    checkpoint = torch.load(out_path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "config",
        "variant",
        "seed",
        "in_dim",
        "best_epoch",
        "best_val_macro_paired_acc",
    ):
        assert key in checkpoint
    assert checkpoint["variant"] == variant
    assert checkpoint["seed"] == 0

    probe = PregateProbe(in_dim=checkpoint["in_dim"])
    probe.load_state_dict(checkpoint["model"])


def test_train_probe_deterministic_same_seed(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
    tmp_path: Path,
) -> None:
    import dataclasses as dc

    graph, store, cfg, cache_dir = _pregate_setup
    cfg = dc.replace(cfg, epochs=2)
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)

    metrics_a = tmp_path / "metrics_a.jsonl"
    metrics_b = tmp_path / "metrics_b.jsonl"
    train_probe(
        corpus,
        cfg,
        variant="p",
        seed=0,
        device=torch.device("cpu"),
        out_path=tmp_path / "probe_a.pt",
        metrics_path=metrics_a,
    )
    train_probe(
        corpus,
        cfg,
        variant="p",
        seed=0,
        device=torch.device("cpu"),
        out_path=tmp_path / "probe_b.pt",
        metrics_path=metrics_b,
    )

    assert metrics_a.read_text() == metrics_b.read_text()


# --------------------------------------------------------------------------- collect_eval_pairs


def test_collect_eval_pairs_matches_manual_union_sorted_unique(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)
    val_idx_set = set(corpus.regions.val_idx)
    assert val_idx_set  # sanity: the fixture actually produces held-out regions

    expected: set[tuple[str, str]] = set()
    for (region_idx, _severity), prov in corpus.provenance.items():
        if region_idx not in val_idx_set:
            continue
        region = corpus.regions.regions[region_idx]
        true_pairs, false_pairs = quad_comparison_pairs(prov)
        for arr in (true_pairs, false_pairs):
            for i, j in arr.tolist():
                u = corpus.regions.node_ids[region[int(i)]]
                v = corpus.regions.node_ids[region[int(j)]]
                lo, hi = sorted((u, v))
                expected.add((lo, hi))

    actual = collect_eval_pairs(corpus)

    # Reconstructing every held-out cell's expected pairs by hand reproduces the
    # function's output exactly.
    assert actual == sorted(expected)
    # Sorted and unique.
    assert actual == sorted(set(actual))
    assert all(u <= v for u, v in actual)


def test_collect_eval_pairs_excludes_train_only_regions(
    _pregate_setup: tuple[nx.Graph, FeatureStore, PregateConfig, Path],
) -> None:
    graph, store, cfg, cache_dir = _pregate_setup
    corpus = build_pregate_corpus(graph, store, cfg, cache_dir=cache_dir)
    val_idx_set = set(corpus.regions.val_idx)

    # A corpus whose provenance keeps only train-region cells (val cells and dropped
    # cells alike differ from the original) must yield zero eval pairs: train and val
    # regions genuinely differ here, and collect_eval_pairs must never draw from the
    # train-only side.
    train_only_provenance = {
        key: prov for key, prov in corpus.provenance.items() if key[0] not in val_idx_set
    }
    assert train_only_provenance  # sanity: the fixture has train cells too
    train_only_corpus = dataclasses.replace(corpus, provenance=train_only_provenance)
    assert collect_eval_pairs(train_only_corpus) == []

    # The full corpus (train + held-out cells) returns a non-empty result.
    assert collect_eval_pairs(corpus) != []


# --------------------------------------------------------------------------- frozen-B0 scoring


def _tiny_v3_1_config(input_dim: int = 8) -> dict[str, object]:
    return {
        "input_dim": input_dim,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0, "activation": "gelu", "norm": "layernorm"},
        "regularization": {"dropout": 0.0},
    }


def _write_v3_1_checkpoint(
    path: Path, *, model: torch.nn.Module, model_family: str, model_config: dict[str, object]
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_family": model_family,
            "model_config": model_config,
            "epoch": 0,
            "val_metrics": {},
            "seed": 0,
            "config": {},
        },
        path,
    )


def test_score_b0_pairs_finite_shape_and_checkpoint_id(tmp_path: Path) -> None:
    node_ids = ["node_000000", "node_000001", "node_000002", "node_000003"]
    store = _write_feature_store(tmp_path / "features", node_ids)
    torch.manual_seed(0)
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_v3_1_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )

    pairs = [("node_000000", "node_000001"), ("node_000002", "node_000003")]
    result = score_b0_pairs(checkpoint_path, pairs, store, device=torch.device("cpu"))

    assert isinstance(result, B0Scores)
    assert result.logits.shape == (2,)
    assert result.logits.dtype == np.float32
    assert np.isfinite(result.logits).all()
    assert isinstance(result.checkpoint_id, str)
    assert result.checkpoint_id


def test_score_b0_pairs_rejects_wrong_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _build_fake_family(_model_config: dict[str, object]) -> torch.nn.Module:
        return torch.nn.Linear(1, 1)

    monkeypatch.setitem(score_universe.MODEL_BUILDERS, "fake_family", _build_fake_family)

    node_ids = ["node_000000", "node_000001"]
    store = _write_feature_store(tmp_path / "features", node_ids)
    model = _build_fake_family({})
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_v3_1_checkpoint(
        checkpoint_path, model=model, model_family="fake_family", model_config={}
    )

    with pytest.raises(ValueError, match="v3_1"):
        score_b0_pairs(
            checkpoint_path,
            [("node_000000", "node_000001")],
            store,
            device=torch.device("cpu"),
        )


# --------------------------------------------------------------------------- b0 score round-trip


def test_save_load_b0_scores_round_trip(tmp_path: Path) -> None:
    pairs = [("node_a", "node_b"), ("node_c", "node_d")]
    logits = np.array([0.5, -1.25], dtype=np.float32)
    path = tmp_path / "b0_scores.npz"

    save_b0_scores(path, pairs, logits, checkpoint_id="deadbeef1234")
    loaded_scores, loaded_checkpoint_id = load_b0_scores(path)

    assert loaded_checkpoint_id == "deadbeef1234"
    assert loaded_scores == {
        ("node_a", "node_b"): 0.5,
        ("node_c", "node_d"): -1.25,
    }
    # Orientation independence: both original orderings sort to the same key.
    lo_ab, hi_ab = sorted(("node_b", "node_a"))
    lo_cd, hi_cd = sorted(("node_d", "node_c"))
    assert loaded_scores[lo_ab, hi_ab] == pytest.approx(0.5)
    assert loaded_scores[lo_cd, hi_cd] == pytest.approx(-1.25)
