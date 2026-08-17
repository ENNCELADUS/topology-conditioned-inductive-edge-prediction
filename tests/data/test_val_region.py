"""Tests for src.data.val_region."""

from __future__ import annotations

import itertools
import json
import math
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
from src.data.artifacts import canonical_pair, load_benchmark
from src.data.val_region import (
    ValBallUnionUniverse,
    ValRegionParams,
    ValRegionSplit,
    _derive_region_membership,
    _fill_val_negatives,
    derive_val_region_split,
    main,
    sample_bfs_ball_buckets,
    val_ball_union_universe,
    val_universe_arrays,
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


class TestDeriveValRegionSplitDeterminism:
    def test_identical_under_permuted_node_and_edge_order(self) -> None:
        nodes = _grid_nodes(6, 6)
        edges = _grid_edges(6, 6)
        negatives = [("n00_00", "n05_05"), ("n00_05", "n05_00")]
        global_positives = frozenset(canonical_pair(u, v) for u, v in edges)
        params = ValRegionParams(
            edge_fraction=0.25, n_regions=3, salt="det|", bucket_sizes=(3, 5), buckets_per_size=3
        )

        split1 = derive_val_region_split(nodes, edges, negatives, global_positives, params=params)
        split2 = derive_val_region_split(
            list(reversed(nodes)), list(reversed(edges)), negatives, global_positives, params=params
        )

        assert split1 == split2

    def test_repeated_calls_are_identical(self) -> None:
        nodes = _grid_nodes(6, 6)
        edges = _grid_edges(6, 6)
        negatives = [("n00_00", "n05_05")]
        global_positives = frozenset(canonical_pair(u, v) for u, v in edges)
        params = ValRegionParams(
            edge_fraction=0.25, n_regions=3, salt="det2|", bucket_sizes=(3, 5), buckets_per_size=3
        )

        split1 = derive_val_region_split(nodes, edges, negatives, global_positives, params=params)
        split2 = derive_val_region_split(nodes, edges, negatives, global_positives, params=params)

        assert split1 == split2


class TestRegionMembershipGrowth:
    def test_k_seeds_are_distinct_and_every_region_grows(self) -> None:
        component = nx.Graph()
        component.add_edges_from(_grid_edges(8, 8))
        params = ValRegionParams(edge_fraction=0.5, n_regions=4, salt="growth|")

        seeds, v_val, region_membership = _derive_region_membership(component, params)

        assert len(seeds) == 4
        assert len(set(seeds)) == 4
        assert len(region_membership) == 4
        assert v_val == frozenset(node for nodes in region_membership for node in nodes)
        for nodes in region_membership:
            assert len(nodes) > 1, "every region must grow beyond its seed"
            assert nodes[0] in seeds

    def test_raises_when_n_regions_exceeds_component_size(self) -> None:
        component = nx.Graph()
        component.add_edges_from([("a", "b"), ("b", "c")])
        params = ValRegionParams(edge_fraction=0.5, n_regions=10, salt="x|")

        with pytest.raises(ValueError, match="n_regions"):
            _derive_region_membership(component, params)


class TestStopRulePicksNearestPrefix:
    """Growth on a path graph gives predictable edge counts per step.

    A 10-node path has exactly one new edge per growth step, so the induced
    edge count after `m` growth additions is exactly `m` regardless of hashed
    growth direction -- letting the target land deliberately closer to one
    side or the other of a specific crossing.
    """

    def test_undershoot_prefix_wins_when_closer(self) -> None:
        component = _path_graph(10)  # 9 edges
        params = ValRegionParams(edge_fraction=2.3 / 9, n_regions=1, salt="stop-under|")

        _, v_val, region_membership = _derive_region_membership(component, params)

        assert len(v_val) == 3  # seed + 2 growth nodes (edge_count=2, diff 0.3 < 0.7)
        assert len(region_membership[0]) == 3

    def test_overshoot_prefix_wins_when_closer(self) -> None:
        component = _path_graph(10)  # 9 edges
        params = ValRegionParams(edge_fraction=2.8 / 9, n_regions=1, salt="stop-over|")

        _, v_val, region_membership = _derive_region_membership(component, params)

        assert len(v_val) == 4  # seed + 3 growth nodes (edge_count=3, diff 0.2 < 0.8)
        assert len(region_membership[0]) == 4


class TestPartitionSemantics:
    def _split(
        self, *, edge_fraction: float = 0.2, n_regions: int = 3
    ) -> tuple[ValRegionSplit, list[Pair], list[str]]:
        nodes = _grid_nodes(8, 8)
        grid_edges = _grid_edges(8, 8)
        self_loops = [(node, node) for node in nodes]
        edges = grid_edges + self_loops
        negatives = [
            ("n00_00", "n07_07"),
            ("n00_07", "n07_00"),
            ("n03_03", "n04_04"),
        ]
        global_positives = frozenset(canonical_pair(u, v) for u, v in edges)
        params = ValRegionParams(
            edge_fraction=edge_fraction,
            n_regions=n_regions,
            salt="partition|",
            bucket_sizes=(2, 3),
            buckets_per_size=2,
        )
        split = derive_val_region_split(nodes, edges, negatives, global_positives, params=params)
        return split, edges, nodes

    def test_no_training_row_has_both_endpoints_internal(self) -> None:
        split, _, _ = self._split()

        assert not any(u in split.v_val and v in split.v_val for u, v in split.training_positives)
        assert not any(u in split.v_val and v in split.v_val for u, v in split.training_negatives)

    def test_positives_partition_exactly(self) -> None:
        split, edges, _ = self._split()
        truth = frozenset(canonical_pair(u, v) for u, v in edges)

        assert frozenset(split.val_positives) | split.training_positives == truth
        assert frozenset(split.val_positives).isdisjoint(split.training_positives)

    def test_cross_boundary_edges_land_in_training_positives(self) -> None:
        split, edges, _ = self._split()
        grid_only = [(u, v) for u, v in edges if u != v]
        cross_edges = [
            canonical_pair(u, v) for u, v in grid_only if (u in split.v_val) != (v in split.v_val)
        ]

        assert cross_edges, "test graph/params must actually produce a boundary"
        assert all(edge in split.training_positives for edge in cross_edges)

    def test_self_pair_with_endpoint_in_v_val_is_a_val_positive(self) -> None:
        split, _, _ = self._split()

        assert split.v_val, "region must be non-empty for this assertion to be meaningful"
        for node in split.v_val:
            assert (node, node) in split.val_positives

    def test_training_negatives_preserve_input_order(self) -> None:
        split, _, nodes = self._split()
        negatives = [
            ("n00_00", "n07_07"),
            ("n00_07", "n07_00"),
            ("n03_03", "n04_04"),
        ]
        expected = tuple(
            canonical_pair(u, v)
            for u, v in negatives
            if not (
                canonical_pair(u, v)[0] in split.v_val and canonical_pair(u, v)[1] in split.v_val
            )
        )

        assert split.training_negatives == expected


class TestFillValNegatives:
    def test_deterministic_topup_rejects_invalid_pairs_and_hits_target(self) -> None:
        v_val = frozenset(f"n{i:02d}" for i in range(10))
        val_positives: list[Pair] = [("n00", "n01"), ("n02", "n03")]
        existing: list[Pair] = [("n04", "n05")]
        global_positive_edges = frozenset({("n06", "n07")})

        result1 = _fill_val_negatives(
            v_val=v_val,
            val_positives=val_positives,
            existing=existing,
            global_positive_edges=global_positive_edges,
            seed=0,
        )
        result2 = _fill_val_negatives(
            v_val=v_val,
            val_positives=val_positives,
            existing=existing,
            global_positive_edges=global_positive_edges,
            seed=0,
        )

        assert result1 == result2
        assert len(result1) == len(val_positives)
        assert result1[: len(existing)] == tuple(existing)
        assert ("n06", "n07") not in result1
        assert all(u != v for u, v in result1)
        assert len(set(result1)) == len(result1)

    def test_truncates_when_existing_already_meets_target(self) -> None:
        v_val = frozenset({"a", "b", "c", "d"})
        val_positives: list[Pair] = [("a", "b")]
        existing: list[Pair] = [("a", "c"), ("a", "d"), ("b", "c")]

        result = _fill_val_negatives(
            v_val=v_val,
            val_positives=val_positives,
            existing=existing,
            global_positive_edges=frozenset(),
            seed=0,
        )

        assert result == (("a", "c"),)


class TestSampleBfsBallBuckets:
    def test_deterministic_exact_size_and_connected(self) -> None:
        g = nx.Graph()
        g.add_edges_from(_grid_edges(6, 6))

        buckets1 = sample_bfs_ball_buckets(g, sizes=(5, 10), per_size=4, salt="bucket-test|")
        buckets2 = sample_bfs_ball_buckets(g, sizes=(5, 10), per_size=4, salt="bucket-test|")

        assert buckets1 == buckets2
        for size, balls in buckets1.items():
            assert len(balls) == 4
            for ball in balls:
                assert len(ball) == size
                assert nx.is_connected(g.subgraph(ball))

    def test_raises_when_no_component_reaches_size(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_node("c")

        with pytest.raises(ValueError, match="no component"):
            sample_bfs_ball_buckets(g, sizes=(5,), per_size=1, salt="x|")


class TestValUniverseArrays:
    def test_row_count_canonical_order_and_dtype(self) -> None:
        v_val = {"c", "a", "b"}

        u_idx, v_idx = val_universe_arrays(v_val)

        assert u_idx.dtype == np.int32
        assert v_idx.dtype == np.int32
        assert len(u_idx) == math.comb(3, 2) + 3
        rows = list(zip(u_idx.tolist(), v_idx.tolist(), strict=True))
        assert rows == [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]


def _make_split(v_val: set[str], buckets: dict[int, list[set[str]]]) -> ValRegionSplit:
    """Build a minimal ValRegionSplit for val_ball_union_universe tests only."""
    return ValRegionSplit(
        train_nodes=frozenset(v_val),
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

    def test_complement_correctness(self) -> None:
        split, v_val = self._tiny_split()
        n = len(v_val)
        result = val_ball_union_universe(split, sample_size=1000)

        union_rows = set(zip(result.u_idx.tolist(), result.v_idx.tolist(), strict=True))
        nonself_union_count = sum(1 for u, v in union_rows if u != v)
        assert result.complement_total == math.comb(n, 2) - nonself_union_count

        sample_rows = list(
            zip(result.sample_u_idx.tolist(), result.sample_v_idx.tolist(), strict=True)
        )
        assert len(sample_rows) == min(1000, result.complement_total)
        assert len(set(sample_rows)) == len(sample_rows), "sample must have no duplicates"
        for u, v in sample_rows:
            assert u != v
            assert (u, v) not in union_rows
            assert 0 <= u < n
            assert 0 <= v < n

    def test_determinism_across_calls(self) -> None:
        split, _ = self._tiny_split()

        result1 = val_ball_union_universe(split, sample_size=5)
        result2 = val_ball_union_universe(split, sample_size=5)

        assert np.array_equal(result1.u_idx, result2.u_idx)
        assert np.array_equal(result1.v_idx, result2.v_idx)
        assert np.array_equal(result1.sample_u_idx, result2.sample_u_idx)
        assert np.array_equal(result1.sample_v_idx, result2.sample_v_idx)

    def test_sample_size_at_least_complement_returns_entire_complement_sorted(self) -> None:
        split, v_val = self._tiny_split()
        n = len(v_val)
        result = val_ball_union_universe(split, sample_size=10_000)

        union_rows = set(zip(result.u_idx.tolist(), result.v_idx.tolist(), strict=True))
        full_nonself = {(u, v) for u in range(n) for v in range(u + 1, n)}
        expected_complement = sorted(full_nonself - union_rows)

        sample_rows = list(
            zip(result.sample_u_idx.tolist(), result.sample_v_idx.tolist(), strict=True)
        )
        assert sample_rows == expected_complement
        assert len(sample_rows) == result.complement_total

    def test_sample_size_zero_gives_empty_typed_arrays(self) -> None:
        split, _ = self._tiny_split()

        result = val_ball_union_universe(split, sample_size=0)

        assert result.sample_u_idx.shape == (0,)
        assert result.sample_v_idx.shape == (0,)
        assert result.sample_u_idx.dtype == np.int32
        assert result.sample_v_idx.dtype == np.int32

    def test_canonical_order_and_dtype(self) -> None:
        split, _ = self._tiny_split()

        result = val_ball_union_universe(split, sample_size=1000)

        for u_idx, v_idx in (
            (result.u_idx, result.v_idx),
            (result.sample_u_idx, result.sample_v_idx),
        ):
            assert u_idx.dtype == np.int32
            assert v_idx.dtype == np.int32
            assert np.all(u_idx <= v_idx)
            rows = list(zip(u_idx.tolist(), v_idx.tolist(), strict=True))
            assert rows == sorted(rows)

    def test_raises_on_negative_sample_size(self) -> None:
        split, _ = self._tiny_split()

        with pytest.raises(ValueError, match="sample_size"):
            val_ball_union_universe(split, sample_size=-1)

    def test_raises_on_empty_buckets(self) -> None:
        split = _make_split({"a", "b"}, {})

        with pytest.raises(ValueError, match="buckets"):
            val_ball_union_universe(split)


class TestSmallParamsSeam:
    def test_tiny_edge_fraction_n_regions_and_bucket_sizes_on_toy_graph(self) -> None:
        nodes = _grid_nodes(5, 5)
        edges = _grid_edges(5, 5)
        global_positives = frozenset(canonical_pair(u, v) for u, v in edges)
        params = ValRegionParams(
            edge_fraction=0.15,
            n_regions=1,
            salt="tiny|",
            bucket_sizes=(2, 3),
            buckets_per_size=2,
            negative_seed=0,
        )

        split = derive_val_region_split(nodes, edges, [], global_positives, params=params)

        assert split.v_val
        assert split.buckets
        for size, balls in split.buckets.items():
            assert len(balls) == 2
            for ball in balls:
                assert len(ball) == size


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
                "--edge-fraction",
                "0.3",
                "--n-regions",
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


@pytest.mark.integration
class TestDeriveValRegionSplitRealData:
    # Measured once against data/benchmark_2025_neurips/breadth_first with the
    # production ValRegionParams() defaults (verified stable across
    # PYTHONHASHSEED=0/1/2): |v_val|=2553, loopless g_val edges=9528, single
    # connected component, edge fraction 47643 * 0.2 = 9528.6 -> 9528/47643 =
    # 0.199987... (within the [0.18, 0.22] acceptance band).
    def test_production_params_on_real_package(self, benchmark_root: Path) -> None:
        benchmark = load_benchmark(benchmark_root, "breadth_first")
        train_nodes = benchmark.split.train_nodes
        truth_edges = list(benchmark.split.train_graph.edges())
        benchmark_negatives = [
            pair
            for pair, label in zip(
                benchmark.split.train_pairs.pairs, benchmark.split.train_pairs.labels, strict=True
            )
            if label == 0
        ] + [
            pair
            for pair, label in zip(
                benchmark.split.val_pairs.pairs, benchmark.split.val_pairs.labels, strict=True
            )
            if label == 0
        ]

        split = derive_val_region_split(
            train_nodes,
            truth_edges,
            benchmark_negatives,
            benchmark.positive_edges,
            params=ValRegionParams(),
        )
        g_val_simple = split.build_g_val_simple()

        assert nx.number_connected_components(g_val_simple) == 1
        assert len(split.v_val) == 2553
        assert g_val_simple.number_of_edges() == 9528
        edge_fraction_achieved = (
            g_val_simple.number_of_edges() / 47_643
        )  # giant component edge count
        assert 0.18 <= edge_fraction_achieved <= 0.22
