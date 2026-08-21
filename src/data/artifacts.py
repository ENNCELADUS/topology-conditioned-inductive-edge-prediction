"""Benchmark artifact loader and verification for `benchmark_2025_neurips`.

Loads the repository-local artifact package (see ``data/README.md`` and
``docs/05-egostitch-spec.md`` §9) and verifies it against the pinned constants
measured on 2026-07-09. This module is intentionally dependency-light: only
``networkx``, ``numpy``, and the standard library. It must not import
``src/data/features.py`` or ``torch``.

Quarantine (spec §9.3): ``*_ratio5_exclusive.txt`` files draw negatives from the
global node set and leak test-side node features across the split boundary. No
loader function in this module may read them — do not add one.
"""

from __future__ import annotations

import logging
import math
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ArtifactVerificationError(RuntimeError):
    """Raised when the on-disk artifacts drift from the pinned expectations."""


def canonical_pair(u: str, v: str) -> tuple[str, str]:
    """Return the canonical ``(min, max)`` ordering of a node pair.

    Args:
        u: First node id.
        v: Second node id.

    Returns:
        ``(u, v)`` if ``u <= v`` else ``(v, u)``. Self-pairs ``(u, u)`` are
        returned unchanged.
    """
    return (u, v) if u <= v else (v, u)


@dataclass(frozen=True)
class LabeledPairs:
    """Canonicalized node pairs with an aligned integer label array.

    Attributes:
        pairs: Canonical-order pairs, in file order.
        labels: Shape ``(n,)`` int8 array, aligned index-for-index with `pairs`.
    """

    pairs: list[tuple[str, str]]
    labels: NDArray[np.int8]


@dataclass(frozen=True)
class SplitArtifacts:
    """All artifacts for a single split strategy (e.g. `breadth_first`).

    Attributes:
        strategy: Split strategy name.
        train_nodes: Train-side node ids.
        test_nodes: Test-side node ids (disjoint from `train_nodes`).
        train_pairs: Labeled pairs from `train_edges.txt`.
        val_pairs: Labeled pairs from `val_edges.txt`.
        test_pairs: Labeled pairs from `test_edges.txt`.
        train_graph: Reference graph over train nodes (`train_graph.pkl`).
        test_graph: Held-out reference graph over test nodes (`test_graph.pkl`).
        buckets: Node-count-keyed sample buckets for assembled-graph evaluation.
    """

    strategy: str
    train_nodes: frozenset[str]
    test_nodes: frozenset[str]
    train_pairs: LabeledPairs
    val_pairs: LabeledPairs
    test_pairs: LabeledPairs
    train_graph: nx.Graph
    test_graph: nx.Graph
    buckets: dict[int, list[set[str]]]


@dataclass(frozen=True)
class Benchmark:
    """A fully loaded benchmark package for one split strategy.

    Attributes:
        root: Benchmark package root directory.
        graph: Full reference graph (`graph.pkl`).
        positive_edges: Canonicalized global positive edge set.
        split: Strategy-specific split artifacts.
    """

    root: Path
    graph: nx.Graph
    positive_edges: frozenset[tuple[str, str]]
    split: SplitArtifacts


# Expected constants measured directly from the shipped artifacts on 2026-07-09
# (global-constraints.md, "breadth_first verification constants"). Only
# "breadth_first" is pinned; verifying any other strategy fails loudly rather
# than silently passing against absent expectations.
_VERIFICATION_CONSTANTS: dict[str, dict[str, int]] = {
    "breadth_first": {
        "graph_nodes": 10_090,
        "graph_edges": 129_861,
        "graph_self_loops": 7_769,
        "train_nodes": 8_072,
        "test_nodes": 2_018,
        "train_edges_rows": 85_824,
        "train_edges_positives": 42_880,
        "val_edges_rows": 21_456,
        "val_edges_positives": 10_760,
        "train_graph_nodes": 8_072,
        "train_graph_edges": 53_640,
        "test_graph_nodes": 2_018,
        "test_graph_edges": 32_019,
        "test_graph_self_loops": 1_891,
        "candidate_rows": 2_037_171,
    }
}


def _load_pickle(path: Path) -> Any:  # noqa: ANN401 - pickle.load is genuinely untyped
    """Unpickle a single object from `path`."""
    with path.open("rb") as f:
        return pickle.load(f)


def _iter_pair_rows(path: Path) -> Iterator[tuple[str, str]]:
    r"""Stream tab-separated ``u v`` rows from a TSV file (e.g. `positive_edges.txt`)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            u, v = stripped.split("\t")
            yield u, v


def _iter_pair_label_rows(path: Path) -> Iterator[tuple[str, str, int]]:
    r"""Stream tab-separated ``u v label`` rows from a TSV file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            u, v, label = stripped.split("\t")
            yield u, v, int(label)


def _read_labeled_pairs(path: Path) -> LabeledPairs:
    r"""Read a full tab-separated ``u v label`` file into a `LabeledPairs`."""
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    for u, v, label in _iter_pair_label_rows(path):
        pairs.append(canonical_pair(u, v))
        labels.append(label)
    return LabeledPairs(pairs=pairs, labels=np.array(labels, dtype=np.int8))


def _endpoints_subset(pairs: Sequence[tuple[str, str]], nodes: set[str]) -> bool:
    """Return whether every endpoint of every pair is in `nodes`."""
    return all(u in nodes and v in nodes for u, v in pairs)


def _positive_pairs(labeled: LabeledPairs) -> set[tuple[str, str]]:
    """Return the set of canonical pairs whose label is 1."""
    return {pair for pair, label in zip(labeled.pairs, labeled.labels, strict=True) if label == 1}


def _negative_pairs(labeled: LabeledPairs) -> set[tuple[str, str]]:
    """Return the set of canonical pairs whose label is 0."""
    return {pair for pair, label in zip(labeled.pairs, labeled.labels, strict=True) if label == 0}


def verify_benchmark(root: Path, strategy: str) -> dict[str, int]:
    """Verify the on-disk artifact package against the pinned expectations.

    Runs every check in the "breadth_first verification constants" section of
    `.superpowers/sdd/global-constraints.md`: exact counts, split disjointness,
    `train_graph.pkl` edge set == train+ union val+ (canonical sets),
    candidate-universe distinctness/cardinality/positive-set identity, and
    balanced-file negatives disjoint from the global positive set. Always reads
    the raw, unfiltered files (there is no `exclude_nodes` parameter here).

    Args:
        root: Benchmark package root (e.g. `data/benchmark_2025_neurips`).
        strategy: Split strategy name (only `"breadth_first"` is pinned).

    Returns:
        The measured stats dict on success (one entry per pinned check).

    Raises:
        ArtifactVerificationError: If `strategy` has no pinned expectations, or
            if one or more checks fail (all failures are listed together).
    """
    if strategy not in _VERIFICATION_CONSTANTS:
        raise ArtifactVerificationError(f"no pinned expectations for {strategy}")
    expected = _VERIFICATION_CONSTANTS[strategy]
    strategy_dir = root / strategy
    errors: list[str] = []
    stats: dict[str, int] = {}

    def check(name: str, measured: int) -> None:
        stats[name] = measured
        if measured != expected[name]:
            errors.append(f"{name}: expected {expected[name]}, got {measured}")

    graph = _load_pickle(root / "graph.pkl")
    check("graph_nodes", graph.number_of_nodes())
    check("graph_edges", graph.number_of_edges())
    check("graph_self_loops", nx.number_of_selfloops(graph))
    graph_nodes = set(graph.nodes())

    split = _load_pickle(strategy_dir / "split.pkl")
    train_nodes: set[str] = set(split["train"])
    test_nodes: set[str] = set(split["test"])
    check("train_nodes", len(train_nodes))
    check("test_nodes", len(test_nodes))
    if train_nodes & test_nodes:
        errors.append("split disjointness: train_nodes and test_nodes overlap")
    if not train_nodes <= graph_nodes:
        errors.append("split disjointness: train_nodes not a subset of graph nodes")
    if not test_nodes <= graph_nodes:
        errors.append("split disjointness: test_nodes not a subset of graph nodes")

    train_pairs = _read_labeled_pairs(strategy_dir / "train_edges.txt")
    check("train_edges_rows", len(train_pairs.pairs))
    check("train_edges_positives", int((train_pairs.labels == 1).sum()))
    if not _endpoints_subset(train_pairs.pairs, train_nodes):
        errors.append("train_edges.txt: endpoints not a subset of train nodes")

    val_pairs = _read_labeled_pairs(strategy_dir / "val_edges.txt")
    check("val_edges_rows", len(val_pairs.pairs))
    check("val_edges_positives", int((val_pairs.labels == 1).sum()))
    if not _endpoints_subset(val_pairs.pairs, train_nodes):
        errors.append("val_edges.txt: endpoints not a subset of train nodes")

    test_pairs = _read_labeled_pairs(strategy_dir / "test_edges.txt")
    if not _endpoints_subset(test_pairs.pairs, test_nodes):
        errors.append("test_edges.txt: endpoints not a subset of test nodes")

    train_graph = _load_pickle(strategy_dir / "train_graph.pkl")
    check("train_graph_nodes", train_graph.number_of_nodes())
    check("train_graph_edges", train_graph.number_of_edges())
    train_plus = _positive_pairs(train_pairs)
    val_plus = _positive_pairs(val_pairs)
    train_graph_edge_set = {canonical_pair(u, v) for u, v in train_graph.edges()}
    if train_graph_edge_set != (train_plus | val_plus):
        errors.append("train_graph edge set != train+ union val+ (as canonical sets)")

    test_graph = _load_pickle(strategy_dir / "test_graph.pkl")
    check("test_graph_nodes", test_graph.number_of_nodes())
    check("test_graph_edges", test_graph.number_of_edges())
    check("test_graph_self_loops", nx.number_of_selfloops(test_graph))
    if set(test_graph.nodes()) != test_nodes:
        errors.append("test_graph.pkl nodes != test split nodes")
    test_graph_edge_set = {canonical_pair(u, v) for u, v in test_graph.edges()}
    test_plus = _positive_pairs(test_pairs)
    if not test_plus <= test_graph_edge_set:
        errors.append("test_edges.txt positives not a subset of test_graph edge set")

    # Candidate universe: streamed, no full list retained (a dedup set is fine).
    all_canonical_pairs: set[tuple[str, str]] = set()
    candidate_positive_pairs: set[tuple[str, str]] = set()
    row_count = 0
    for u, v, label in _iter_pair_label_rows(strategy_dir / "candidate_test_edges.txt"):
        pair = canonical_pair(u, v)
        all_canonical_pairs.add(pair)
        row_count += 1
        if label == 1:
            candidate_positive_pairs.add(pair)
    check("candidate_rows", row_count)
    if len(all_canonical_pairs) != row_count:
        errors.append(
            "candidate_test_edges.txt: rows are not distinct canonical pairs "
            f"({row_count} rows, {len(all_canonical_pairs)} distinct)"
        )
    expected_candidate_count = math.comb(len(test_nodes), 2) + len(test_nodes)
    if row_count != expected_candidate_count:
        errors.append(
            f"candidate_test_edges.txt: row count {row_count} != "
            f"C({len(test_nodes)},2)+{len(test_nodes)} = {expected_candidate_count}"
        )
    if candidate_positive_pairs != test_graph_edge_set:
        errors.append("candidate_test_edges.txt positives != test_graph edge set")

    # Balanced-file negatives must never coincide with the global positive set.
    global_positive_pairs = {
        canonical_pair(u, v) for u, v in _iter_pair_rows(root / "positive_edges.txt")
    }
    for file_name, labeled in (
        ("train_edges.txt", train_pairs),
        ("val_edges.txt", val_pairs),
        ("test_edges.txt", test_pairs),
    ):
        collisions = _negative_pairs(labeled) & global_positive_pairs
        if collisions:
            errors.append(f"{file_name}: negatives intersect global positives ({len(collisions)})")

    if errors:
        raise ArtifactVerificationError("; ".join(errors))
    return stats


def _drop_pairs_touching(
    labeled: LabeledPairs, exclude_nodes: frozenset[str]
) -> tuple[LabeledPairs, int]:
    """Drop rows of `labeled` touching any node in `exclude_nodes`.

    Returns:
        The filtered `LabeledPairs` and the number of dropped rows.
    """
    keep_mask = [u not in exclude_nodes and v not in exclude_nodes for u, v in labeled.pairs]
    kept_pairs = [pair for pair, keep in zip(labeled.pairs, keep_mask, strict=True) if keep]
    kept_labels = labeled.labels[np.array(keep_mask, dtype=np.bool_)]
    n_dropped = len(labeled.pairs) - len(kept_pairs)
    return LabeledPairs(pairs=kept_pairs, labels=kept_labels), n_dropped


def load_benchmark(
    root: Path,
    strategy: str,
    *,
    verify: bool = True,
    exclude_nodes: frozenset[str] = frozenset(),
) -> Benchmark:
    """Load the full benchmark package for one split strategy.

    Args:
        root: Benchmark package root (e.g. `data/benchmark_2025_neurips`).
        strategy: Split strategy name (e.g. `"breadth_first"`).
        verify: If True, run `verify_benchmark` against the raw, unfiltered
            artifacts before loading (fails loudly on drift). Verification
            always runs on unfiltered data, independent of `exclude_nodes`.
        exclude_nodes: Featureless nodes (spec §9.2) to drop from the labeled
            train/val/test pairs. Dropped row counts are logged per file.

    Returns:
        The loaded `Benchmark`.
    """
    if verify:
        verify_benchmark(root, strategy)

    strategy_dir = root / strategy
    graph = _load_pickle(root / "graph.pkl")
    positive_edges = frozenset(
        canonical_pair(u, v) for u, v in _iter_pair_rows(root / "positive_edges.txt")
    )

    split = _load_pickle(strategy_dir / "split.pkl")
    train_nodes = frozenset(split["train"])
    test_nodes = frozenset(split["test"])

    train_pairs = _read_labeled_pairs(strategy_dir / "train_edges.txt")
    val_pairs = _read_labeled_pairs(strategy_dir / "val_edges.txt")
    test_pairs = _read_labeled_pairs(strategy_dir / "test_edges.txt")

    if exclude_nodes:
        for file_name, labeled_pairs in (
            ("train_edges.txt", train_pairs),
            ("val_edges.txt", val_pairs),
            ("test_edges.txt", test_pairs),
        ):
            filtered, n_dropped = _drop_pairs_touching(labeled_pairs, exclude_nodes)
            logger.info("%s: dropped %d pairs touching excluded nodes", file_name, n_dropped)
            if file_name == "train_edges.txt":
                train_pairs = filtered
            elif file_name == "val_edges.txt":
                val_pairs = filtered
            else:
                test_pairs = filtered

    train_graph = _load_pickle(strategy_dir / "train_graph.pkl")
    test_graph = _load_pickle(strategy_dir / "test_graph.pkl")
    buckets = _load_pickle(strategy_dir / "test_node_buckets.pkl")

    split_artifacts = SplitArtifacts(
        strategy=strategy,
        train_nodes=train_nodes,
        test_nodes=test_nodes,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_graph=train_graph,
        test_graph=test_graph,
        buckets=buckets,
    )
    return Benchmark(root=root, graph=graph, positive_edges=positive_edges, split=split_artifacts)


def load_candidate_pairs(root: Path, strategy: str) -> LabeledPairs:
    """Load the full candidate pair universe for `strategy`.

    Loads all 2,037,171 rows of `candidate_test_edges.txt` (the `breadth_first`
    pinned count) into memory as a canonical-pair list plus an aligned int8
    label array. This costs roughly 0.5 GB of resident memory; callers that
    evaluate repeatedly in one session should cache the result rather than
    reloading it.

    Args:
        root: Benchmark package root (e.g. `data/benchmark_2025_neurips`).
        strategy: Split strategy name (e.g. `"breadth_first"`).

    Returns:
        The full candidate-pair `LabeledPairs`.
    """
    return _read_labeled_pairs(root / strategy / "candidate_test_edges.txt")


def load_test_topology_pairs(root: Path, strategy: str) -> LabeledPairs:
    """Load the deduplicated pair union needed by sampled topology evaluation.

    The benchmark evaluates topology only on the sampled node sets in
    ``test_node_buckets.pkl``.  Scoring every pair in the full test-node
    universe is therefore unnecessary: this loader returns the union of
    canonical pairs induced by those samples, plus a self-pair for any test
    node absent from every sample. Those support-only rows preserve the full
    test-node grounding universe without restoring quadratic scoring.
    """
    strategy_dir = root / strategy
    buckets = _load_pickle(strategy_dir / "test_node_buckets.pkl")
    test_graph = _load_pickle(strategy_dir / "test_graph.pkl")
    if not isinstance(buckets, dict):
        raise TypeError("test_node_buckets.pkl must contain a dict")
    if not isinstance(test_graph, nx.Graph):
        raise TypeError("test_graph.pkl must contain a networkx.Graph")

    pair_set = {
        canonical_pair(u, v)
        for node_sets in buckets.values()
        for nodes in node_sets
        for u, v in combinations_with_replacement(sorted(nodes), 2)
    }
    pair_set.update((str(node), str(node)) for node in test_graph.nodes())
    pairs = sorted(pair_set)
    labels = np.fromiter(
        (int(test_graph.has_edge(u, v)) for u, v in pairs),
        dtype=np.int8,
        count=len(pairs),
    )
    return LabeledPairs(pairs=pairs, labels=labels)
