"""Tests for the frozen feature store and F0 mean-pool matrix builder."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from src.data.features import FeatureStore, build_f0_matrix

pytestmark = pytest.mark.unit


def _write_feature_root(
    tmp_path: Path,
    node_shapes: dict[str, tuple[int, int]],
    *,
    input_dim: int,
    fmt: str = "torch_pt_per_node",
) -> Path:
    """Build a tiny synthetic feature root (metadata.json + index.json + .pt files)."""
    root = tmp_path / "features"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id, (length, dim) in node_shapes.items():
        tensor = torch.randn(length, dim, dtype=torch.float32)
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tensor, root / rel_path)
        index[node_id] = rel_path
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps({"format": fmt, "input_dim": input_dim, "max_sequence_length": 1024})
    )
    return root


class TestFeatureStore:
    def test_node_ids_and_input_dim_from_metadata(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (7, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        assert store.node_ids == frozenset(shapes)
        assert store.input_dim == 4

    def test_load_tokens_returns_expected_tensor(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        expected = torch.load(
            root / "embeddings/node_000001.pt", map_location="cpu", weights_only=True
        )
        tokens = store.load_tokens("node_000001")
        assert tokens.shape == (5, 4)
        assert tokens.dtype == torch.float32
        assert torch.equal(tokens, expected)

    def test_load_tokens_unknown_node_raises_keyerror(self, tmp_path: Path) -> None:
        root = _write_feature_root(tmp_path, {"node_000001": (5, 4)}, input_dim=4)
        store = FeatureStore(root)
        with pytest.raises(KeyError, match="node_999999"):
            store.load_tokens("node_999999")

    def test_load_tokens_wrong_ndim_raises(self, tmp_path: Path) -> None:
        root = _write_feature_root(tmp_path, {"node_000001": (5, 4)}, input_dim=4)
        torch.save(torch.randn(5), root / "embeddings/node_000001.pt")
        store = FeatureStore(root)
        with pytest.raises(ValueError, match="ndim"):
            store.load_tokens("node_000001")

    def test_load_tokens_wrong_feature_dim_raises(self, tmp_path: Path) -> None:
        root = _write_feature_root(tmp_path, {"node_000001": (5, 4)}, input_dim=4)
        torch.save(torch.randn(5, 3), root / "embeddings/node_000001.pt")
        store = FeatureStore(root)
        with pytest.raises(ValueError, match="input_dim"):
            store.load_tokens("node_000001")

    def test_load_tokens_wrong_dtype_raises(self, tmp_path: Path) -> None:
        root = _write_feature_root(tmp_path, {"node_000001": (5, 4)}, input_dim=4)
        torch.save(torch.randn(5, 4).double(), root / "embeddings/node_000001.pt")
        store = FeatureStore(root)
        with pytest.raises(ValueError, match="dtype"):
            store.load_tokens("node_000001")

    def test_bad_metadata_format_raises(self, tmp_path: Path) -> None:
        root = _write_feature_root(
            tmp_path, {"node_000001": (5, 4)}, input_dim=4, fmt="something_else"
        )
        with pytest.raises(ValueError, match="format"):
            FeatureStore(root)


class TestBuildF0Matrix:
    def test_mean_matches_manual_computation(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4), "node_000003": (7, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        node_ids = list(shapes)
        matrix, index = build_f0_matrix(store, node_ids)
        assert matrix.shape == (3, 4)
        assert matrix.dtype == torch.float32
        assert index == {nid: i for i, nid in enumerate(node_ids)}
        for nid in node_ids:
            expected = store.load_tokens(nid).mean(dim=0)
            row = matrix[index[nid]]
            assert torch.allclose(row, expected, atol=1e-6)

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        node_ids = list(shapes)
        cache_path = tmp_path / "f0_cache.pt"

        matrix1, index1 = build_f0_matrix(store, node_ids, cache_path=cache_path)
        assert cache_path.exists()

        matrix2, index2 = build_f0_matrix(store, node_ids, cache_path=cache_path)
        assert torch.equal(matrix1, matrix2)
        assert index1 == index2

    def test_cache_wrong_order_raises(self, tmp_path: Path) -> None:
        shapes = {"node_000001": (5, 4), "node_000002": (3, 4)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        node_ids = list(shapes)
        cache_path = tmp_path / "f0_cache.pt"
        build_f0_matrix(store, node_ids, cache_path=cache_path)

        reordered = list(reversed(node_ids))
        with pytest.raises(ValueError, match="order"):
            build_f0_matrix(store, reordered, cache_path=cache_path)

    def test_logs_progress_every_1000_nodes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        shapes = {f"node_{i:06d}": (2, 4) for i in range(1, 1001)}
        root = _write_feature_root(tmp_path, shapes, input_dim=4)
        store = FeatureStore(root)
        node_ids = list(shapes)
        with caplog.at_level(logging.INFO):
            build_f0_matrix(store, node_ids)
        assert any("1000" in record.message for record in caplog.records)


@pytest.mark.integration
class TestFeatureStoreIntegration:
    def test_load_tokens_real_node(self, features_root: Path) -> None:
        store = FeatureStore(features_root)
        tokens = store.load_tokens("node_000001")
        assert tokens.shape == (123, 1536)
        assert tokens.dtype == torch.float32

    def test_build_f0_matrix_real_nodes_matches_manual_means(self, features_root: Path) -> None:
        store = FeatureStore(features_root)
        node_ids = [
            "node_000001",
            "node_000002",
            "node_000003",
            "node_000004",
            "node_000005",
        ]
        matrix, index = build_f0_matrix(store, node_ids)
        for nid in node_ids:
            expected = store.load_tokens(nid).mean(dim=0)
            row = matrix[index[nid]]
            assert torch.allclose(row, expected, atol=1e-6)
