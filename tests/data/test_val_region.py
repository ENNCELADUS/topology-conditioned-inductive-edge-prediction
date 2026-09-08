"""Tests for src.data.val_region."""

from __future__ import annotations

import itertools
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import src.data.artifacts as artifacts
from src.data.artifacts import canonical_pair
from src.data.val_region import (
    ValBallUnionUniverse,
    ValRegionParams,
    ValRegionSplit,
    _bfs_ball,
    _bfs_positive_budget,
    _sample_val_negatives,
    derive_val_region_split,
    main,
    sample_bfs_ball_buckets,
    val_ball_union_universe,
)

pytestmark = pytest.mark.unit

Pair = tuple[str, str]


def _grid_nodes(rows: int, cols: int) -> list[str]:
    return [f"n{r:02d}_{c:02d}" for r in range(rows) for c in range(cols)]


def _grid_edges(rows: int, cols: int) -> list[Pair]:
    edges: list[Pair] = []
    for r in range(rows):
        for c in range(cols):
            node = f"n{r:02d}_{c:02d}"
            if c + 1 < cols:
                edges.append((node, f"n{r:02d}_{c + 1:02d}"))
            if r + 1 < rows:
                edges.append((node, f"n{r + 1:02d}_{c:02d}"))
    return edges


def _path_graph(n: int) -> nx.Graph:
    nodes = [f"n{i:02d}" for i in range(n)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(zip(nodes, nodes[1:], strict=False))
    return graph


class TestNodeHeldOutSampling:
    def test_order_independent_partition_drops_every_pair_touching_v_val(self) -> None:
        nodes = _grid_nodes(6, 6)
        edges = _grid_edges(6, 6)
        negatives = list(itertools.combinations(nodes, 2))
        params = ValRegionParams(
            positive_edge_fraction=0.4, root_neighbors=2, bucket_sizes=(3, 5), buckets_per_size=3
        )
        truth = frozenset(edges)
        negatives = [pair for pair in negatives if pair not in truth]
        split = derive_val_region_split(nodes, edges, negatives, truth, params=params)
        replay = derive_val_region_split(
            reversed(nodes), reversed(edges), negatives, truth, params=params
        )
        assert split == replay
        assert len(split.val_positives) <= int(0.4 * len(truth))
        assert split.substrate_nodes == frozenset(nodes)
        assert split.v_val and not split.v_val & split.train_nodes
        assert split.train_nodes | split.v_val == frozenset(nodes)
        assert set(split.build_training_graph()) == split.train_nodes
        assert nx.is_connected(split.build_g_val_simple())
        assert all(
            u in split.train_nodes and v in split.train_nodes
            for u, v in (*split.training_positives, *split.training_negatives)
        )
        for original, retained in (
            (edges, split.training_positives),
            (negatives, split.training_negatives),
        ):
            boundary = {(u, v) for u, v in original if (u in split.v_val) != (v in split.v_val)}
            assert boundary and not boundary & set(retained)
            assert set(retained) == {
                (u, v) for u, v in original if u not in split.v_val and v not in split.v_val
            }
        assert not set(split.training_positives) & set(split.val_positives)
        assert not set(split.training_negatives) & set(split.val_negatives)
        assert split.val_positives == tuple(
            sorted((u, v) for u, v in edges if u in split.v_val and v in split.v_val)
        )
        assert len(split.val_negatives) == len(split.val_positives)
        assert len(set(split.val_negatives)) == len(split.val_negatives)
        assert all(
            u != v and u in split.v_val and v in split.v_val and (u, v) not in truth
            for u, v in split.val_negatives
        )

    def test_edge_budget_counts_loops_and_stops_before_overshoot(self) -> None:
        graph = nx.Graph([("a", "a"), ("a", "b"), ("a", "c"), ("b", "c")])
        assert _bfs_positive_budget(graph, "a", 3) == frozenset({"a", "b"})
        assert _bfs_positive_budget(graph, "a", 4) == frozenset({"a", "b", "c"})

    def test_fifo_preserves_parent_queue_order(self) -> None:
        graph = nx.Graph([("r", "a"), ("r", "b"), ("a", "z"), ("b", "c")])
        # Whole-layer lexical sorting would choose c before z.
        assert _bfs_ball(graph, "r", 4) == {"r", "a", "b", "z"}

    def test_root_neighbor_count_includes_loop_once(self) -> None:
        graph = _path_graph(20)
        graph.add_edge("n00", "n00")
        params = ValRegionParams(
            positive_edge_fraction=0.4, root_neighbors=2, bucket_sizes=(3,), buckets_per_size=2
        )
        assert len(graph["n00"]) == 2 and graph.degree("n00") == 3
        # All eligible roots, including n00, participate in the same seeded choice.
        import random

        expected = random.Random(42).choice(sorted(n for n in graph if len(graph[n]) == 2))
        split = derive_val_region_split(graph.nodes, graph.edges, [], frozenset(), params=params)
        assert split.region_seeds == (expected,)

    def test_no_root_and_unreachable_bucket_fail_explicitly(self) -> None:
        graph = _path_graph(20)
        with pytest.raises(ValueError, match="eligible root"):
            derive_val_region_split(graph.nodes, graph.edges, [], frozenset())
        with pytest.raises(ValueError, match="every root"):
            sample_bfs_ball_buckets(graph, sizes=(21,), per_size=1, seed=43)

    def test_buckets_allow_repeat_roots_and_are_deterministic(self) -> None:
        graph = _path_graph(6)
        first = sample_bfs_ball_buckets(graph, sizes=(3, 6), per_size=20, seed=43)
        assert first == sample_bfs_ball_buckets(graph, sizes=(3, 6), per_size=20, seed=43)
        assert len(first[6]) == 20
        assert all(ball == set(graph) for ball in first[6])

    def test_negative_sampler_rejects_impossible_balance(self) -> None:
        with pytest.raises(ValueError, match="distinct negatives"):
            _sample_val_negatives([("a", "b"), ("a", "a")], seed=0)


def _make_split(v_val: set[str], buckets: dict[int, list[set[str]]]) -> ValRegionSplit:
    """Build a minimal ValRegionSplit for val_ball_union_universe tests only."""
    return ValRegionSplit(
        train_nodes=frozenset(),
        v_val=frozenset(v_val),
        region_seeds=(),
        training_positives=frozenset(),
        training_negatives=(),
        val_positives=(),
        val_negatives=(),
        buckets=buckets,
        params=ValRegionParams(),
    )


def _brute_force_union_pairs(
    v_val: set[str], buckets: dict[int, list[set[str]]]
) -> set[tuple[int, int]]:
    nodes = sorted(v_val)
    index = {node: i for i, node in enumerate(nodes)}
    pairs: set[tuple[int, int]] = set()
    for balls in buckets.values():
        for ball in balls:
            ball_indices = sorted(index[node] for node in ball)
            for u, v in itertools.combinations_with_replacement(ball_indices, 2):
                pairs.add((u, v))
    return pairs


class TestValBallUnionUniverse:
    def _tiny_split(self) -> tuple[ValRegionSplit, set[str]]:
        # n02 is shared between the two 3-balls (overlap to dedup); n05 only
        # appears in the 2-ball; n06..n09 never appear in any ball.
        v_val = {f"n{i:02d}" for i in range(10)}
        buckets = {
            3: [{"n00", "n01", "n02"}, {"n02", "n03", "n04"}],
            2: [{"n00", "n05"}],
        }
        return _make_split(v_val, buckets), v_val

    def test_union_matches_brute_force_dedup_across_balls(self) -> None:
        split, v_val = self._tiny_split()
        expected = _brute_force_union_pairs(v_val, split.buckets)

        result = val_ball_union_universe(split)

        assert isinstance(result, ValBallUnionUniverse)
        rows = set(zip(result.u_idx.tolist(), result.v_idx.tolist(), strict=True))
        assert rows == expected
        assert len(rows) == len(result.u_idx), "rows must be globally deduplicated"

    def test_self_pairs_exactly_cover_ball_nodes(self) -> None:
        split, v_val = self._tiny_split()
        nodes = sorted(v_val)
        index = {node: i for i, node in enumerate(nodes)}
        covered = {
            index[node] for balls in split.buckets.values() for ball in balls for node in ball
        }

        result = val_ball_union_universe(split)

        self_pair_indices = {
            u for u, v in zip(result.u_idx.tolist(), result.v_idx.tolist(), strict=True) if u == v
        }
        assert self_pair_indices == covered
        assert covered != set(range(len(nodes))), "fixture must leave some nodes uncovered"

    def test_determinism_across_calls(self) -> None:
        split, _ = self._tiny_split()

        result1 = val_ball_union_universe(split)
        result2 = val_ball_union_universe(split)

        assert np.array_equal(result1.u_idx, result2.u_idx)
        assert np.array_equal(result1.v_idx, result2.v_idx)

    def test_canonical_order_and_dtype(self) -> None:
        split, _ = self._tiny_split()

        result = val_ball_union_universe(split)

        assert result.u_idx.dtype == np.int32
        assert result.v_idx.dtype == np.int32
        assert np.all(result.u_idx <= result.v_idx)
        rows = list(zip(result.u_idx.tolist(), result.v_idx.tolist(), strict=True))
        assert rows == sorted(rows)

    def test_raises_on_empty_buckets(self) -> None:
        split = _make_split({"a", "b"}, {})

        with pytest.raises(ValueError, match="buckets"):
            val_ball_union_universe(split)


_SYNTHETIC_TRAIN_NODES = [f"t{i:02d}" for i in range(1, 13)]
_SYNTHETIC_TEST_NODES = [f"s{i:02d}" for i in range(1, 5)]
_SYNTHETIC_TRAIN_PATH_EDGES = list(
    zip(_SYNTHETIC_TRAIN_NODES, _SYNTHETIC_TRAIN_NODES[1:], strict=False)
)
_SYNTHETIC_TEST_PATH_EDGES = list(
    zip(_SYNTHETIC_TEST_NODES, _SYNTHETIC_TEST_NODES[1:], strict=False)
)

SYNTHETIC_VAL_REGION_EXPECTED: dict[str, int] = {
    "graph_nodes": 16,
    "graph_edges": 14,
    "graph_self_loops": 0,
    "train_nodes": 12,
    "test_nodes": 4,
    "train_edges_rows": 8,
    "train_edges_positives": 6,
    "val_edges_rows": 7,
    "val_edges_positives": 5,
    "train_graph_nodes": 12,
    "train_graph_edges": 11,
    "test_graph_nodes": 4,
    "test_graph_edges": 3,
    "test_graph_self_loops": 0,
    "candidate_rows": 10,
}


def _write_pairs_file(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v, label in rows:
            f.write(f"{u}\t{v}\t{label}\n")


def _write_edge_list_file(path: Path, pairs: list[Pair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v in pairs:
            f.write(f"{u}\t{v}\n")


def _build_synthetic_val_region_package(root: Path) -> None:
    """Build a tiny, `verify_benchmark`-consistent package with a real train path.

    A 12-train/4-test synthetic package: a train path (11 edges) large enough
    for a real (tiny-params) V_val derivation.
    """
    graph = nx.Graph()
    graph.add_nodes_from(_SYNTHETIC_TRAIN_NODES + _SYNTHETIC_TEST_NODES)
    graph.add_edges_from(_SYNTHETIC_TRAIN_PATH_EDGES + _SYNTHETIC_TEST_PATH_EDGES)
    with (root / "graph.pkl").open("wb") as f:
        pickle.dump(graph, f)

    _write_edge_list_file(
        root / "positive_edges.txt", _SYNTHETIC_TRAIN_PATH_EDGES + _SYNTHETIC_TEST_PATH_EDGES
    )

    strategy_dir = root / "synthetic_val_region"
    strategy_dir.mkdir()
    with (strategy_dir / "split.pkl").open("wb") as f:
        pickle.dump({"train": set(_SYNTHETIC_TRAIN_NODES), "test": set(_SYNTHETIC_TEST_NODES)}, f)

    train_positive_rows = [(u, v, 1) for u, v in _SYNTHETIC_TRAIN_PATH_EDGES[:6]]
    train_negative_rows = [("t01", "t12", 0), ("t02", "t11", 0)]
    _write_pairs_file(strategy_dir / "train_edges.txt", train_positive_rows + train_negative_rows)

    val_positive_rows = [(u, v, 1) for u, v in _SYNTHETIC_TRAIN_PATH_EDGES[6:]]
    val_negative_rows = [("t01", "t11", 0), ("t03", "t10", 0)]
    _write_pairs_file(strategy_dir / "val_edges.txt", val_positive_rows + val_negative_rows)

    _write_pairs_file(strategy_dir / "test_edges.txt", [("s01", "s02", 1), ("s01", "s03", 0)])

    train_graph = nx.Graph()
    train_graph.add_nodes_from(_SYNTHETIC_TRAIN_NODES)
    train_graph.add_edges_from(_SYNTHETIC_TRAIN_PATH_EDGES)
    with (strategy_dir / "train_graph.pkl").open("wb") as f:
        pickle.dump(train_graph, f)

    test_graph = nx.Graph()
    test_graph.add_nodes_from(_SYNTHETIC_TEST_NODES)
    test_graph.add_edges_from(_SYNTHETIC_TEST_PATH_EDGES)
    with (strategy_dir / "test_graph.pkl").open("wb") as f:
        pickle.dump(test_graph, f)

    test_edge_set = {canonical_pair(u, v) for u, v in _SYNTHETIC_TEST_PATH_EDGES}
    candidate_rows: list[tuple[str, str, int]] = []
    for i, u in enumerate(_SYNTHETIC_TEST_NODES):
        for v in _SYNTHETIC_TEST_NODES[i + 1 :]:
            candidate_rows.append((u, v, int((u, v) in test_edge_set)))
    for u in _SYNTHETIC_TEST_NODES:
        candidate_rows.append((u, u, 0))
    _write_pairs_file(strategy_dir / "candidate_test_edges.txt", candidate_rows)

    with (strategy_dir / "test_node_buckets.pkl").open("wb") as f:
        pickle.dump({2: [{"s01", "s02"}]}, f)


@pytest.fixture
def synthetic_val_region_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _build_synthetic_val_region_package(tmp_path)
    monkeypatch.setitem(
        artifacts._VERIFICATION_CONSTANTS,
        "synthetic_val_region",
        dict(SYNTHETIC_VAL_REGION_EXPECTED),
    )
    return tmp_path


class TestBuilderCli:
    def test_writes_manifest_with_expected_top_level_keys(
        self, synthetic_val_region_package: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "manifest.json"

        main(
            [
                "--data-root",
                str(synthetic_val_region_package),
                "--strategy",
                "synthetic_val_region",
                "--output",
                str(output_path),
                "--positive-edge-fraction",
                "0.5",
                "--root-neighbors",
                "1",
                "--bucket-sizes",
                "2",
                "3",
                "--buckets-per-size",
                "2",
            ]
        )

        manifest = json.loads(output_path.read_text())
        assert set(manifest) == {
            "schema_version",
            "substrate_nodes",
            "train_nodes",
            "strategy",
            "params",
            "region_seeds",
            "v_val",
            "val_positives",
            "val_negatives",
            "training_positives_count",
            "training_negatives_count",
            "buckets",
            "fidelity",
        }
        assert manifest["strategy"] == "synthetic_val_region"
        assert manifest["v_val"]


class TestBuilderCliImportIsCheap:
    def test_import_alone_does_not_run_the_builder(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        start = time.time()
        proc = subprocess.run(  # noqa: S603 - fixed interpreter + literal snippet
            [sys.executable, "-c", "import src.data.val_region"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=dict(os.environ, PYTHONPATH=str(repo_root)),
            check=True,
            timeout=30,
        )
        elapsed = time.time() - start

        assert proc.returncode == 0
        assert elapsed < 10.0
