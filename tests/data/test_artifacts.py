"""Tests for src.data.artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import src.data.artifacts as artifacts
from src.data.artifacts import (
    ArtifactVerificationError,
    canonical_pair,
    load_benchmark,
    load_candidate_pairs,
    load_test_topology_pairs,
    verify_benchmark,
)


class TestCanonicalPair:
    def test_orders_lexicographically(self) -> None:
        assert canonical_pair("node_000002", "node_000001") == (
            "node_000001",
            "node_000002",
        )

    def test_already_ordered_stays_ordered(self) -> None:
        assert canonical_pair("node_000001", "node_000002") == (
            "node_000001",
            "node_000002",
        )

    def test_self_pair_stays_self_pair(self) -> None:
        assert canonical_pair("node_000005", "node_000005") == (
            "node_000005",
            "node_000005",
        )


class TestVerifyBenchmarkUnpinnedStrategy:
    def test_raises_for_strategy_without_pinned_expectations(self, tmp_path: Path) -> None:
        with pytest.raises(
            ArtifactVerificationError, match="no pinned expectations for depth_first"
        ):
            verify_benchmark(tmp_path, "depth_first")


# --- Synthetic mini-package for verify_benchmark failure-mode tests --------------
#
# A tiny, fully self-consistent 7-node package (4 train nodes, 3 test nodes) that
# satisfies every check `verify_benchmark` performs. Registered under a "synthetic"
# strategy key (via monkeypatch on the module-private constants map) so the shipped
# production mapping keeps only "breadth_first".

SYNTHETIC_EXPECTED: dict[str, int] = {
    "graph_nodes": 7,
    "graph_edges": 6,
    "graph_self_loops": 2,
    "train_nodes": 4,
    "test_nodes": 3,
    "train_edges_rows": 2,
    "train_edges_positives": 1,
    "val_edges_rows": 2,
    "val_edges_positives": 1,
    "train_graph_nodes": 4,
    "train_graph_edges": 2,
    "test_graph_nodes": 3,
    "test_graph_edges": 3,
    "test_graph_self_loops": 1,
    "candidate_rows": 6,
}


def _write_pairs_file(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v, label in rows:
            f.write(f"{u}\t{v}\t{label}\n")


def _write_edge_list_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v in pairs:
            f.write(f"{u}\t{v}\n")


def _build_synthetic_package(root: Path) -> None:
    """Write a tiny, fully self-consistent synthetic benchmark package under root."""
    graph = nx.Graph()
    graph.add_nodes_from(["n1", "n2", "n3", "n4", "n5", "n6", "n7"])
    graph.add_edges_from(
        [("n1", "n2"), ("n1", "n1"), ("n3", "n4"), ("n5", "n6"), ("n5", "n5"), ("n6", "n7")]
    )
    with (root / "graph.pkl").open("wb") as f:
        pickle.dump(graph, f)

    _write_edge_list_file(
        root / "positive_edges.txt",
        [("n1", "n2"), ("n1", "n1"), ("n3", "n4"), ("n5", "n6"), ("n5", "n5"), ("n6", "n7")],
    )

    strategy_dir = root / "synthetic"
    strategy_dir.mkdir()

    with (strategy_dir / "split.pkl").open("wb") as f:
        pickle.dump({"train": {"n1", "n2", "n3", "n4"}, "test": {"n5", "n6", "n7"}}, f)

    _write_pairs_file(strategy_dir / "train_edges.txt", [("n1", "n2", 1), ("n2", "n3", 0)])
    _write_pairs_file(strategy_dir / "val_edges.txt", [("n3", "n4", 1), ("n1", "n4", 0)])
    _write_pairs_file(strategy_dir / "test_edges.txt", [("n5", "n6", 1), ("n5", "n7", 0)])

    train_graph = nx.Graph()
    train_graph.add_nodes_from(["n1", "n2", "n3", "n4"])
    train_graph.add_edges_from([("n1", "n2"), ("n3", "n4")])
    with (strategy_dir / "train_graph.pkl").open("wb") as f:
        pickle.dump(train_graph, f)

    test_graph = nx.Graph()
    test_graph.add_nodes_from(["n5", "n6", "n7"])
    test_graph.add_edges_from([("n5", "n6"), ("n6", "n7"), ("n5", "n5")])
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(test_graph, f)

    _write_pairs_file(
        strategy_dir / "candidate_test_edges.txt",
        [
            ("n5", "n6", 1),
            ("n5", "n7", 0),
            ("n6", "n7", 1),
            ("n5", "n5", 1),
            ("n6", "n6", 0),
            ("n7", "n7", 0),
        ],
    )

    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump({20: [{"n5", "n6", "n7"}]}, f)


@pytest.fixture
def synthetic_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp_path-built, fully consistent tiny benchmark package under "synthetic"."""
    _build_synthetic_package(tmp_path)
    monkeypatch.setitem(artifacts._VERIFICATION_CONSTANTS, "synthetic", dict(SYNTHETIC_EXPECTED))
    return tmp_path


class TestSyntheticBenchmarkPackage:
    def test_consistent_package_returns_measured_stats(self, synthetic_package: Path) -> None:
        stats = verify_benchmark(synthetic_package, "synthetic")
        assert stats == SYNTHETIC_EXPECTED

    def test_single_violation_names_the_check(
        self, synthetic_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(artifacts._VERIFICATION_CONSTANTS["synthetic"], "graph_edges", 999)
        with pytest.raises(ArtifactVerificationError) as exc_info:
            verify_benchmark(synthetic_package, "synthetic")
        message = str(exc_info.value)
        assert "graph_edges" in message
        assert "999" in message
        assert "6" in message

    def test_multiple_violations_are_all_listed(
        self, synthetic_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(artifacts._VERIFICATION_CONSTANTS["synthetic"], "graph_edges", 999)
        monkeypatch.setitem(
            artifacts._VERIFICATION_CONSTANTS["synthetic"], "test_graph_self_loops", 42
        )
        with pytest.raises(ArtifactVerificationError) as exc_info:
            verify_benchmark(synthetic_package, "synthetic")
        message = str(exc_info.value)
        assert "graph_edges" in message
        assert "test_graph_self_loops" in message

    def test_train_graph_edge_set_mismatch_is_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_synthetic_package(tmp_path)
        # Tamper: swap an edge so the SET differs from train+/union val+ while the
        # edge COUNT still matches the pinned expectation (isolates the set check).
        train_graph = nx.Graph()
        train_graph.add_nodes_from(["n1", "n2", "n3", "n4"])
        train_graph.add_edges_from([("n1", "n2"), ("n2", "n4")])
        with (tmp_path / "synthetic" / "train_graph.pkl").open("wb") as f:
            pickle.dump(train_graph, f)
        monkeypatch.setitem(
            artifacts._VERIFICATION_CONSTANTS, "synthetic", dict(SYNTHETIC_EXPECTED)
        )
        with pytest.raises(ArtifactVerificationError, match="train_graph edge set"):
            verify_benchmark(tmp_path, "synthetic")

    def test_negatives_intersecting_global_positives_is_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_synthetic_package(tmp_path)
        # Tamper: train_edges.txt negative row now collides with a global positive.
        _write_pairs_file(
            tmp_path / "synthetic" / "train_edges.txt", [("n1", "n2", 1), ("n3", "n4", 0)]
        )
        monkeypatch.setitem(
            artifacts._VERIFICATION_CONSTANTS, "synthetic", dict(SYNTHETIC_EXPECTED)
        )
        with pytest.raises(ArtifactVerificationError, match="negatives"):
            verify_benchmark(tmp_path, "synthetic")

    def test_load_benchmark_drops_rows_touching_excluded_nodes(
        self, synthetic_package: Path
    ) -> None:
        baseline = load_benchmark(synthetic_package, "synthetic")
        exclude = frozenset({"n1"})

        filtered = load_benchmark(synthetic_package, "synthetic", exclude_nodes=exclude)

        assert len(filtered.split.train_pairs.pairs) == len(baseline.split.train_pairs.pairs) - 1
        assert len(filtered.split.val_pairs.pairs) == len(baseline.split.val_pairs.pairs) - 1
        assert len(filtered.split.test_pairs.pairs) == len(baseline.split.test_pairs.pairs)
        assert all("n1" not in pair for pair in filtered.split.train_pairs.pairs)
        assert all("n1" not in pair for pair in filtered.split.val_pairs.pairs)
        assert filtered.split.train_pairs.labels.dtype == np.int8

    def test_load_candidate_pairs_reads_labeled_rows(self, synthetic_package: Path) -> None:
        candidate_pairs = load_candidate_pairs(synthetic_package, "synthetic")
        assert candidate_pairs.pairs == [
            ("n5", "n6"),
            ("n5", "n7"),
            ("n6", "n7"),
            ("n5", "n5"),
            ("n6", "n6"),
            ("n7", "n7"),
        ]
        np.testing.assert_array_equal(candidate_pairs.labels, np.array([1, 0, 1, 1, 0, 0]))
        assert candidate_pairs.labels.dtype == np.int8

    def test_load_test_topology_pairs_is_exact_sample_union(self, synthetic_package: Path) -> None:
        strategy_dir = synthetic_package / "synthetic"
        with (strategy_dir / "test_node_buckets.pkl").open("wb") as handle:
            pickle.dump({2: [{"n5", "n6"}]}, handle)

        topology_pairs = load_test_topology_pairs(synthetic_package, "synthetic")

        assert topology_pairs.pairs == [
            ("n5", "n5"),
            ("n5", "n6"),
            ("n6", "n6"),
            ("n7", "n7"),
        ]
        np.testing.assert_array_equal(topology_pairs.labels, np.array([1, 1, 0, 0]))
