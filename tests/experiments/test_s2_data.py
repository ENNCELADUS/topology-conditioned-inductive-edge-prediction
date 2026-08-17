"""Tests for the S2 region corpus builder, feature scatter, and batching helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.features import FeatureStore
from src.data.val_region import sample_bfs_ball_buckets
from src.eval.graph_metrics import strip_self_loops
from src.experiments.s2_latent_topology.data import (
    FEATURELESS_NODES,
    build_features,
    build_region_corpus,
    collate_regions,
    iter_size_batches,
    load_corpus,
    save_corpus,
)

pytestmark = pytest.mark.unit

INPUT_DIM = 8
TOKEN_LEN = 5


def _write_feature_root(tmp_path: Path, node_ids: list[str]) -> Path:
    """Build a tiny synthetic FeatureStore root: metadata.json + index.json + per-node .pt."""
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index: dict[str, str] = {}
    for node_id in node_ids:
        tensor = torch.tensor(rng.standard_normal((TOKEN_LEN, INPUT_DIM)), dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": INPUT_DIM, "max_sequence_length": 1024}
        )
    )
    return root


def _make_base_graph() -> nx.Graph:
    """A small deterministic connected graph with `node_%06d` string ids."""
    base = nx.connected_watts_strogatz_graph(30, k=4, p=0.3, seed=0)
    return nx.relabel_nodes(base, {i: f"node_{i:06d}" for i in base.nodes()})


def _make_graph_with_featureless_hub() -> nx.Graph:
    """`_make_base_graph` plus a featureless node hub-connected to every other node.

    Hub-connecting guarantees the featureless node is an immediate neighbor of
    every seed, so most sampled BFS balls actually include it — the dropping
    behavior under test is exercised for real rather than left to chance.
    """
    graph = _make_base_graph()
    hub = sorted(FEATURELESS_NODES)[0]
    graph.add_node(hub)
    for other in list(graph.nodes()):
        if other != hub:
            graph.add_edge(hub, other)
    return graph


def _expected_kept_and_dropped(
    graph: nx.Graph, *, sizes: tuple[int, ...], per_size: int, salt: str
) -> tuple[list[set[str]], int]:
    """Independently replay bucket sampling + the featureless-drop rule as an oracle."""
    buckets = sample_bfs_ball_buckets(
        strip_self_loops(graph), sizes=sizes, per_size=per_size, salt=salt
    )
    kept: list[set[str]] = []
    dropped = 0
    for size in sizes:
        for node_set in buckets[size]:
            if node_set & FEATURELESS_NODES:
                dropped += 1
            else:
                kept.append(node_set)
    return kept, dropped


class TestBuildFeatures:
    def test_scatters_present_rows_and_zeros_missing(self, tmp_path: Path) -> None:
        node_ids = ["node_000001", "node_000002", "node_000003"]
        store = FeatureStore(_write_feature_root(tmp_path, ["node_000001", "node_000003"]))

        matrix, missing = build_features(store, node_ids, cache_path=tmp_path / "f0.pt")

        assert missing == ["node_000002"]
        assert matrix.shape == (3, INPUT_DIM)
        assert matrix.dtype == torch.float32
        assert torch.all(matrix[1] == 0)
        for row, node_id in [(0, "node_000001"), (2, "node_000003")]:
            expected = store.load_tokens(node_id).mean(dim=0)
            assert torch.allclose(matrix[row], expected, atol=1e-6)


class TestBuildRegionCorpus:
    def test_deterministic_given_same_salt(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        sizes = (6, 8)

        corpus1 = build_region_corpus(
            graph,
            store,
            sizes=sizes,
            per_size=4,
            salt="s2-determinism|",
            holdout_frac=0.25,
            cache_dir=tmp_path / "run1",
        )
        corpus2 = build_region_corpus(
            graph,
            store,
            sizes=sizes,
            per_size=4,
            salt="s2-determinism|",
            holdout_frac=0.25,
            cache_dir=tmp_path / "run2",
        )

        assert corpus1.node_ids == corpus2.node_ids
        assert corpus1.regions == corpus2.regions
        assert corpus1.train_idx == corpus2.train_idx
        assert corpus1.val_idx == corpus2.val_idx
        assert corpus1.dropped_featureless_regions == corpus2.dropped_featureless_regions
        assert torch.equal(corpus1.features, corpus2.features)
        for edges1, edges2 in zip(corpus1.edges, corpus2.edges, strict=True):
            assert torch.equal(edges1, edges2)

    def test_flatten_order_matches_bucket_sampler_and_split_takes_last_by_draw_order(
        self, tmp_path: Path
    ) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        sizes = (6, 8)
        per_size = 4
        salt = "s2-corpus|"

        corpus = build_region_corpus(
            graph,
            store,
            sizes=sizes,
            per_size=per_size,
            salt=salt,
            holdout_frac=0.25,
            cache_dir=tmp_path / "cache",
        )

        expected_kept, expected_dropped = _expected_kept_and_dropped(
            graph, sizes=sizes, per_size=per_size, salt=salt
        )
        assert corpus.dropped_featureless_regions == expected_dropped == 0
        assert len(corpus.regions) == len(expected_kept)
        for region, node_set in zip(corpus.regions, expected_kept, strict=True):
            assert list(region) == sorted(region)
            assert {corpus.node_ids[i] for i in region} == node_set

        assert set(corpus.train_idx) | set(corpus.val_idx) == set(range(len(corpus.regions)))
        assert set(corpus.train_idx).isdisjoint(corpus.val_idx)
        for size in sizes:
            size_positions = [i for i, r in enumerate(corpus.regions) if len(r) == size]
            assert len(size_positions) == per_size
            n_val = math.ceil(0.25 * len(size_positions))
            expected_val = size_positions[-n_val:] if n_val else []
            expected_train = size_positions[: len(size_positions) - n_val]
            assert sorted(i for i in corpus.val_idx if len(corpus.regions[i]) == size) == sorted(
                expected_val
            )
            assert sorted(i for i in corpus.train_idx if len(corpus.regions[i]) == size) == sorted(
                expected_train
            )

    def test_local_edges_match_induced_subgraph_and_are_ordered_i_lt_j(
        self, tmp_path: Path
    ) -> None:
        graph = _make_base_graph()
        simple = strip_self_loops(graph)
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))

        corpus = build_region_corpus(
            graph,
            store,
            sizes=(6, 8),
            per_size=4,
            salt="s2-edges|",
            holdout_frac=0.0,
            cache_dir=tmp_path / "cache",
        )

        for region, edge_tensor in zip(corpus.regions, corpus.edges, strict=True):
            assert edge_tensor.dtype == torch.long
            assert edge_tensor.shape[1] == 2
            region_node_ids = [corpus.node_ids[i] for i in region]
            for u, v in edge_tensor.tolist():
                assert u < v
            got_edges = {
                frozenset((region_node_ids[u], region_node_ids[v])) for u, v in edge_tensor.tolist()
            }
            expected_edges = {
                frozenset((u, v)) for u, v in simple.subgraph(set(region_node_ids)).edges()
            }
            assert got_edges == expected_edges

    def test_region_containing_featureless_node_is_dropped_and_counted(
        self, tmp_path: Path
    ) -> None:
        graph = _make_graph_with_featureless_hub()
        node_ids = sorted(graph.nodes())
        store_ids = [n for n in node_ids if n not in FEATURELESS_NODES]
        store = FeatureStore(_write_feature_root(tmp_path, store_ids))
        sizes = (6, 8)
        per_size = 4
        salt = "s2-test|"

        corpus = build_region_corpus(
            graph,
            store,
            sizes=sizes,
            per_size=per_size,
            salt=salt,
            holdout_frac=0.0,
            cache_dir=tmp_path / "cache",
        )

        expected_kept, expected_dropped = _expected_kept_and_dropped(
            graph, sizes=sizes, per_size=per_size, salt=salt
        )
        assert expected_dropped > 0
        assert corpus.dropped_featureless_regions == expected_dropped
        assert len(corpus.regions) == len(expected_kept)

        hub = sorted(FEATURELESS_NODES)[0]
        hub_idx = corpus.node_ids.index(hub)
        for region in corpus.regions:
            assert hub_idx not in region
        assert torch.all(corpus.features[hub_idx] == 0)


class TestSaveLoadCorpus:
    def test_round_trip(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        corpus = build_region_corpus(
            graph,
            store,
            sizes=(6, 8),
            per_size=4,
            salt="s2-save|",
            holdout_frac=0.25,
            cache_dir=tmp_path / "cache",
        )

        path = tmp_path / "corpus.pt"
        save_corpus(corpus, path)
        loaded = load_corpus(path)

        assert loaded.node_ids == corpus.node_ids
        assert loaded.regions == corpus.regions
        assert loaded.train_idx == corpus.train_idx
        assert loaded.val_idx == corpus.val_idx
        assert loaded.dropped_featureless_regions == corpus.dropped_featureless_regions
        assert torch.equal(loaded.features, corpus.features)
        for edges1, edges2 in zip(loaded.edges, corpus.edges, strict=True):
            assert torch.equal(edges1, edges2)


class TestCollateRegions:
    def test_shapes_mask_x_and_adjacency(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        corpus = build_region_corpus(
            graph,
            store,
            sizes=(6, 8),
            per_size=4,
            salt="s2-collate|",
            holdout_frac=0.0,
            cache_dir=tmp_path / "cache",
        )
        idx6 = next(i for i, r in enumerate(corpus.regions) if len(r) == 6)
        idx8 = next(i for i, r in enumerate(corpus.regions) if len(r) == 8)

        batch = collate_regions(corpus, [idx6, idx8], torch.device("cpu"))

        assert batch.x.shape == (2, 8, corpus.features.shape[1])
        assert batch.adj.shape == (2, 8, 8)
        assert batch.mask.shape == (2, 8)
        assert batch.n.tolist() == [6, 8]
        assert batch.mask[0].tolist() == [1.0] * 6 + [0.0] * 2
        assert batch.mask[1].tolist() == [1.0] * 8

        for row, region_idx in enumerate([idx6, idx8]):
            region = corpus.regions[region_idx]
            for pos, global_id in enumerate(region):
                assert torch.allclose(batch.x[row, pos], corpus.features[global_id])
            for pos in range(len(region), 8):
                assert torch.all(batch.x[row, pos] == 0)

            size = len(region)
            adj_block = batch.adj[row]
            assert torch.equal(adj_block, adj_block.T)
            assert torch.all(torch.diagonal(adj_block) == 0)
            assert torch.all(adj_block[size:, :] == 0)
            assert torch.all(adj_block[:, size:] == 0)
            expected = torch.zeros(8, 8)
            for u, v in corpus.edges[region_idx].tolist():
                expected[u, v] = 1.0
                expected[v, u] = 1.0
            assert torch.equal(adj_block, expected)


class TestIterSizeBatches:
    def test_never_mixes_sizes_covers_all_indices_and_respects_cap(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        corpus = build_region_corpus(
            graph,
            store,
            sizes=(6, 8),
            per_size=4,
            salt="s2-iter|",
            holdout_frac=0.0,
            cache_dir=tmp_path / "cache",
        )
        indices = list(range(len(corpus.regions)))
        rng = np.random.default_rng(42)

        seen: list[int] = []
        for batch in iter_size_batches(corpus, indices, batch_sets=3, rng=rng):
            assert 1 <= len(batch) <= 3
            sizes_in_batch = {len(corpus.regions[i]) for i in batch}
            assert len(sizes_in_batch) == 1
            seen.extend(batch)
        assert sorted(seen) == sorted(indices)

    def test_deterministic_under_seeded_rng(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        store = FeatureStore(_write_feature_root(tmp_path, sorted(graph.nodes())))
        corpus = build_region_corpus(
            graph,
            store,
            sizes=(6, 8),
            per_size=4,
            salt="s2-iter2|",
            holdout_frac=0.0,
            cache_dir=tmp_path / "cache",
        )
        indices = list(range(len(corpus.regions)))

        batches1 = list(
            iter_size_batches(corpus, indices, batch_sets=3, rng=np.random.default_rng(7))
        )
        batches2 = list(
            iter_size_batches(corpus, indices, batch_sets=3, rng=np.random.default_rng(7))
        )

        assert batches1 == batches2
