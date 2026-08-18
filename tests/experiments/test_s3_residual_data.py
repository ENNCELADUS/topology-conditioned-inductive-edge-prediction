"""Tests for `src/experiments/s3_set_residual/data.py`'s S3 corpus, V_val masking, and cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
from src import score_universe
from src.data.features import FeatureStore
from src.data.val_region import ValRegionParams, derive_val_region_split
from src.experiments.s2_latent_topology.data import FEATURELESS_NODES
from src.experiments.s3_set_residual import data

pytestmark = pytest.mark.unit

INPUT_DIM = 8
TOKEN_LEN = 5
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- fixtures / helpers


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


def _tiny_v3_1_config() -> dict[str, object]:
    return {
        "input_dim": INPUT_DIM,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "mlp_head": {"hidden_dims": [8], "dropout": 0.0, "activation": "gelu", "norm": "layernorm"},
        "regularization": {"dropout": 0.0},
    }


def _write_checkpoint(
    path: Path, *, model: torch.nn.Module, model_family: str, model_config: dict[str, object]
) -> None:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "model_family": model_family,
        "model_config": model_config,
        "epoch": 0,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    torch.save(payload, path)


def _build_checkpoint_and_store(tmp_path: Path, node_ids: list[str]) -> tuple[Path, FeatureStore]:
    """Build a tiny real `v3_1` checkpoint plus a synthetic `FeatureStore` covering `node_ids`.

    Called at most once per test (each test gets its own `tmp_path`), so the
    feature root and checkpoint use fixed, non-colliding filenames.
    """
    store = FeatureStore(_write_feature_root(tmp_path, node_ids))
    torch.manual_seed(0)
    model = score_universe.build_model("v3_1", _tiny_v3_1_config())
    checkpoint_path = tmp_path / "ckpt.pt"
    _write_checkpoint(
        checkpoint_path, model=model, model_family="v3_1", model_config=_tiny_v3_1_config()
    )
    return checkpoint_path, store


def _make_base_graph() -> nx.Graph:
    """A small deterministic connected graph with `node_%06d` string ids."""
    base = nx.connected_watts_strogatz_graph(30, k=4, p=0.3, seed=0)
    return nx.relabel_nodes(base, {i: f"node_{i:06d}" for i in base.nodes()})


def _make_graph_with_featureless_hub() -> nx.Graph:
    """`_make_base_graph` plus a featureless node hub-connected to every other node."""
    graph = _make_base_graph()
    hub = sorted(FEATURELESS_NODES)[0]
    graph.add_node(hub)
    for other in list(graph.nodes()):
        if other != hub:
            graph.add_edge(hub, other)
    return graph


def _grid_nodes(rows: int, cols: int) -> list[str]:
    return [f"n{r:02d}_{c:02d}" for r in range(rows) for c in range(cols)]


def _grid_edges(rows: int, cols: int) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            node = f"n{r:02d}_{c:02d}"
            if c + 1 < cols:
                edges.append((node, f"n{r:02d}_{c + 1:02d}"))
            if r + 1 < rows:
                edges.append((node, f"n{r + 1:02d}_{c:02d}"))
    return edges


# --------------------------------------------------------------------------- derive_vval_nodes


class TestDeriveVvalNodes:
    def test_matches_canonical_split_v_val(self) -> None:
        nodes = _grid_nodes(6, 6)
        edges = _grid_edges(6, 6)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        params = ValRegionParams(
            edge_fraction=0.25,
            n_regions=3,
            salt="s3-vval-nodes|",
            bucket_sizes=(3, 5),
            buckets_per_size=3,
        )

        result = data.derive_vval_nodes(graph, params=params)

        expected = derive_val_region_split(nodes, edges, [], frozenset(), params=params).v_val
        assert result == expected


# --------------------------------------------------------------------------- base-logit cache


class TestBaseLogitCache:
    def test_build_dedupes_and_roundtrips_fp32(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())[:6]
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)
        pairs = [
            (node_ids[0], node_ids[1]),
            (node_ids[1], node_ids[2]),
            (node_ids[1], node_ids[0]),  # duplicate, reversed order
        ]

        cache = data.build_base_logit_cache(pairs, store, checkpoint=checkpoint_path, device=CPU)

        assert cache.logit.dtype == np.float32
        assert len(cache.u_idx) == 2

        path = tmp_path / "cache.npz"
        data.save_base_logit_cache(cache, path)
        loaded = data.load_base_logit_cache(path)

        assert loaded.node_ids == cache.node_ids
        np.testing.assert_array_equal(loaded.u_idx, cache.u_idx)
        np.testing.assert_array_equal(loaded.v_idx, cache.v_idx)
        np.testing.assert_array_equal(loaded.logit, cache.logit)
        assert loaded.checkpoint_id == cache.checkpoint_id

    def test_rejects_empty_pairs(self, tmp_path: Path) -> None:
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, ["node_a"])
        with pytest.raises(ValueError, match="at least one pair"):
            data.build_base_logit_cache([], store, checkpoint=checkpoint_path, device=CPU)

    def test_load_rejects_non_fp32(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.npz"
        np.savez(
            path,
            node_ids=np.array(["node_a", "node_b"]),
            u_idx=np.array([0], dtype=np.int32),
            v_idx=np.array([1], dtype=np.int32),
            logit=np.array([0.5], dtype=np.float64),
            checkpoint_id=np.array("deadbeef"),
        )
        with pytest.raises(ValueError, match="fp32"):
            data.load_base_logit_cache(path)


# --------------------------------------------------------------------------- build_s3_corpus


class TestBuildS3Corpus:
    def test_deterministic_given_same_salt(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        def _build(cache_dir: Path) -> data.S3Corpus:
            return data.build_s3_corpus(
                graph,
                store,
                checkpoint=checkpoint_path,
                sizes=(6, 8),
                per_size=3,
                neg_ratio=2,
                salt="s3-det|",
                cache_dir=cache_dir,
                device=CPU,
                vval_nodes=frozenset(),
            )

        corpus1 = _build(tmp_path / "run1")
        corpus2 = _build(tmp_path / "run2")

        assert corpus1.base.regions == corpus2.base.regions
        for p1, p2 in zip(corpus1.train_pairs, corpus2.train_pairs, strict=True):
            assert torch.equal(p1, p2)
        for l1, l2 in zip(corpus1.train_labels, corpus2.train_labels, strict=True):
            assert torch.equal(l1, l2)
        for g1, g2 in zip(corpus1.train_base_logits, corpus2.train_base_logits, strict=True):
            assert torch.equal(g1, g2)

    def test_vval_internal_pairs_dropped_everywhere_and_only_there(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)
        sizes = (6, 8)
        per_size = 3
        salt = "s3-vvalmask|"

        baseline = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=sizes,
            per_size=per_size,
            neg_ratio=2,
            salt=salt,
            cache_dir=tmp_path / "cache0",
            device=CPU,
            vval_nodes=frozenset(),
        )
        r = next(i for i, p in enumerate(baseline.train_pairs) if p.numel() > 0)
        region_node_ids = [baseline.base.node_ids[i] for i in baseline.base.regions[r]]
        pos_local = tuple(baseline.base.edges[r][0].tolist())
        u, v = region_node_ids[pos_local[0]], region_node_ids[pos_local[1]]
        vval_nodes = frozenset({u, v})

        masked = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=sizes,
            per_size=per_size,
            neg_ratio=2,
            salt=salt,
            cache_dir=tmp_path / "cache1",
            device=CPU,
            vval_nodes=vval_nodes,
        )

        assert masked.base.regions == baseline.base.regions
        for i in range(len(baseline.base.regions)):
            ids = [baseline.base.node_ids[k] for k in baseline.base.regions[i]]
            baseline_pairs = {tuple(p) for p in baseline.train_pairs[i].tolist()}
            masked_pairs = {tuple(p) for p in masked.train_pairs[i].tolist()}
            expected = {p for p in baseline_pairs if {ids[p[0]], ids[p[1]]} != {u, v}}
            assert masked_pairs == expected
        assert pos_local not in {tuple(p) for p in masked.train_pairs[r].tolist()}
        assert pos_local in {tuple(p) for p in baseline.train_pairs[r].tolist()}

    def test_neg_ratio_honored_when_pool_sufficient(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=3,
            neg_ratio=2,
            salt="s3-negratio|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(),
        )

        for r in range(len(corpus.base.regions)):
            n = len(corpus.base.regions[r])
            n_pos = int(corpus.base.edges[r].shape[0])
            max_neg = n * (n - 1) // 2 - n_pos
            expected_neg = min(2 * n_pos, max_neg)
            n_pos_kept = int(corpus.train_labels[r].sum().item())
            n_neg_kept = int(corpus.train_labels[r].numel() - n_pos_kept)
            assert n_pos_kept == n_pos
            assert n_neg_kept == expected_neg

    def test_no_self_pairs(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=3,
            neg_ratio=2,
            salt="s3-noself|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(),
        )

        for pairs in corpus.train_pairs:
            for i, j in pairs.tolist():
                assert i < j

    def test_featureless_region_dropping_inherited_from_s2(self, tmp_path: Path) -> None:
        graph = _make_graph_with_featureless_hub()
        node_ids = sorted(n for n in graph.nodes() if n not in FEATURELESS_NODES)
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=3,
            neg_ratio=1,
            salt="s3-featureless|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(),
        )

        assert corpus.base.dropped_featureless_regions > 0
        assert len(corpus.train_pairs) == len(corpus.base.regions)
        hub = sorted(FEATURELESS_NODES)[0]
        assert hub not in {
            node
            for region in corpus.base.regions
            for node in (corpus.base.node_ids[i] for i in region)
        }

    def test_second_build_reuses_cache_without_rescoring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)
        cache_dir = tmp_path / "cache"
        calls = {"n": 0}
        original = getattr(data, "_score_v3_1")  # noqa: B009 - avoids a private-reexport lint

        def counting(*args: object, **kwargs: object) -> np.ndarray:
            calls["n"] += 1
            return cast(np.ndarray, original(*args, **kwargs))

        monkeypatch.setattr(data, "_score_v3_1", counting)

        def _build() -> data.S3Corpus:
            return data.build_s3_corpus(
                graph,
                store,
                checkpoint=checkpoint_path,
                sizes=(6,),
                per_size=2,
                neg_ratio=1,
                salt="s3-reuse|",
                cache_dir=cache_dir,
                device=CPU,
                vval_nodes=frozenset(),
            )

        _build()
        first_calls = calls["n"]
        assert first_calls > 0

        _build()
        assert calls["n"] == first_calls

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=2,
            neg_ratio=1,
            salt="s3-roundtrip|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(),
        )

        path = tmp_path / "corpus.pt"
        data.save_s3_corpus(corpus, path)
        loaded = data.load_s3_corpus(path)

        assert loaded.base.regions == corpus.base.regions
        assert loaded.vval_nodes == corpus.vval_nodes
        for p1, p2 in zip(loaded.train_pairs, corpus.train_pairs, strict=True):
            assert torch.equal(p1, p2)
        for g1, g2 in zip(loaded.train_base_logits, corpus.train_base_logits, strict=True):
            assert torch.equal(g1, g2)


# --------------------------------------------------------------------------- collate_pair_batch


class TestCollatePairBatch:
    def test_shapes_and_set_row_alignment(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=2,
            neg_ratio=2,
            salt="s3-collate|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(),
        )
        idx6 = next(i for i, r in enumerate(corpus.base.regions) if len(r) == 6)
        idx8 = next(i for i, r in enumerate(corpus.base.regions) if len(r) == 8)

        region_batch, pairs, labels, base_logits = data.collate_pair_batch(
            corpus, [idx6, idx8], CPU
        )

        assert region_batch.x.shape == (2, 8, corpus.base.features.shape[1])
        expected_p = corpus.train_pairs[idx6].shape[0] + corpus.train_pairs[idx8].shape[0]
        assert pairs.shape == (expected_p, 3)
        assert labels.shape == (expected_p,)
        assert base_logits.shape == (expected_p,)

        n6 = corpus.train_pairs[idx6].shape[0]
        assert torch.all(pairs[:n6, 0] == 0)
        assert torch.all(pairs[n6:, 0] == 1)
        assert torch.equal(pairs[:n6, 1:], corpus.train_pairs[idx6])
        assert torch.equal(pairs[n6:, 1:], corpus.train_pairs[idx8])
        assert torch.equal(labels[:n6], corpus.train_labels[idx6])
        assert torch.equal(labels[n6:], corpus.train_labels[idx8])
        assert torch.equal(base_logits[:n6], corpus.train_base_logits[idx6])
        assert torch.equal(base_logits[n6:], corpus.train_base_logits[idx8])

    def test_empty_pairs_regions_yield_empty_batch_rows(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)
        # vval_nodes = every node: every region's pairs are fully masked out.
        corpus = data.build_s3_corpus(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6,),
            per_size=2,
            neg_ratio=1,
            salt="s3-empty|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(node_ids),
        )

        region_batch, pairs, labels, base_logits = data.collate_pair_batch(corpus, [0], CPU)

        assert region_batch.x.shape[0] == 1
        assert pairs.shape == (0, 3)
        assert labels.shape == (0,)
        assert base_logits.shape == (0,)


# --------------------------------------------------------------------------- build_vval_eval


class TestBuildVvalEval:
    def test_shapes_symmetry_and_true_adjacency(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        vval = data.build_vval_eval(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6, 8),
            per_size=2,
            salt="s3-vvaleval|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(node_ids),
        )

        assert vval.sizes == (6, 8)
        for size in vval.sizes:
            assert len(vval.node_ids[size]) == 2
            triples = zip(
                vval.node_ids[size], vval.base_logits[size], vval.true_adj[size], strict=True
            )
            for ordered, base_mat, adj_mat in triples:
                n = len(ordered)
                assert base_mat.shape == (n, n)
                assert adj_mat.shape == (n, n)
                assert base_mat.dtype == torch.float32
                assert adj_mat.dtype == torch.float32
                assert torch.equal(base_mat, base_mat.T)
                assert torch.equal(adj_mat, adj_mat.T)
                assert torch.all(torch.diagonal(base_mat) == 0)
                assert torch.all(torch.diagonal(adj_mat) == 0)
                for i in range(n):
                    for j in range(i + 1, n):
                        expected = 1.0 if graph.has_edge(ordered[i], ordered[j]) else 0.0
                        assert adj_mat[i, j].item() == expected

        assert vval.checkpoint_id
        assert set(vval.bucket_ref.node_sets) == set(vval.sizes)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        graph = _make_base_graph()
        node_ids = sorted(graph.nodes())
        checkpoint_path, store = _build_checkpoint_and_store(tmp_path, node_ids)

        vval = data.build_vval_eval(
            graph,
            store,
            checkpoint=checkpoint_path,
            sizes=(6,),
            per_size=2,
            salt="s3-vvalsave|",
            cache_dir=tmp_path / "cache",
            device=CPU,
            vval_nodes=frozenset(node_ids),
        )

        path = tmp_path / "vval.pkl"
        data.save_vval_eval(vval, path)
        loaded = data.load_vval_eval(path)

        assert loaded.sizes == vval.sizes
        assert loaded.checkpoint_id == vval.checkpoint_id
        for size in vval.sizes:
            assert loaded.node_ids[size] == vval.node_ids[size]
            for m1, m2 in zip(loaded.base_logits[size], vval.base_logits[size], strict=True):
                assert torch.equal(m1, m2)
