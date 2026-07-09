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


class TestVerifyBenchmarkSyntheticPackage:
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


@pytest.mark.integration
class TestVerifyBenchmarkRealPackage:
    def test_breadth_first_passes_and_stats_match_pinned_constants(
        self, benchmark_root: Path
    ) -> None:
        stats = verify_benchmark(benchmark_root, "breadth_first")
        assert stats["graph_nodes"] == 10_090
        assert stats["graph_edges"] == 129_861
        assert stats["graph_self_loops"] == 7_769
        assert stats["train_nodes"] == 8_072
        assert stats["test_nodes"] == 2_018
        assert stats["train_graph_edges"] == 53_640
        assert stats["candidate_rows"] == 2_037_171


@pytest.mark.integration
class TestLoadBenchmarkExcludeNodes:
    def test_drops_rows_touching_a_real_train_side_node(self, benchmark_root: Path) -> None:
        # Self-verifying: pick a node that genuinely appears in train_edges.txt so the
        # test proves the drop mechanism fires, independent of which specific nodes are
        # excluded (see report for why the two spec-named featureless nodes don't
        # produce a nonzero delta under the breadth_first strategy).
        baseline = load_benchmark(benchmark_root, "breadth_first")
        sample_node = baseline.split.train_pairs.pairs[0][0]
        exclude = frozenset({sample_node})
        expected_train_dropped = sum(
            1 for u, v in baseline.split.train_pairs.pairs if u in exclude or v in exclude
        )
        expected_val_dropped = sum(
            1 for u, v in baseline.split.val_pairs.pairs if u in exclude or v in exclude
        )
        assert expected_train_dropped > 0

        filtered = load_benchmark(benchmark_root, "breadth_first", exclude_nodes=exclude)

        train_delta = len(baseline.split.train_pairs.pairs) - len(filtered.split.train_pairs.pairs)
        val_delta = len(baseline.split.val_pairs.pairs) - len(filtered.split.val_pairs.pairs)
        assert train_delta == expected_train_dropped
        assert val_delta == expected_val_dropped
        assert len(filtered.split.train_pairs.labels) == len(filtered.split.train_pairs.pairs)

    def test_spec_featureless_nodes_touch_zero_breadth_first_pairs(
        self, benchmark_root: Path
    ) -> None:
        # docs/06-egostitch-spec.md §9.2 states these two globally-featureless nodes
        # (node_004764, node_007050) touch "3 train pairs + 1 val pair" — but that
        # count is measured against the `random_walk` strategy, not `breadth_first`
        # (the strategy pinned for this round). Under `breadth_first` neither node
        # appears in train_edges.txt/val_edges.txt/test_edges.txt at all, so
        # exclude_nodes correctly drops zero rows here. See task-1 report for the
        # flagged discrepancy.
        baseline = load_benchmark(benchmark_root, "breadth_first")
        exclude = frozenset({"node_004764", "node_007050"})

        filtered = load_benchmark(benchmark_root, "breadth_first", exclude_nodes=exclude)

        assert len(filtered.split.train_pairs.pairs) == len(baseline.split.train_pairs.pairs)
        assert len(filtered.split.val_pairs.pairs) == len(baseline.split.val_pairs.pairs)
        assert len(filtered.split.test_pairs.pairs) == len(baseline.split.test_pairs.pairs)


@pytest.mark.integration
class TestLoadCandidatePairs:
    def test_loads_full_pinned_row_count(self, benchmark_root: Path) -> None:
        candidate_pairs = load_candidate_pairs(benchmark_root, "breadth_first")
        assert len(candidate_pairs.pairs) == 2_037_171
        assert candidate_pairs.labels.shape == (2_037_171,)
        assert candidate_pairs.labels.dtype == np.int8
