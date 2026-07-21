"""Tests for the current internal topology holdout."""

from __future__ import annotations

import hashlib
import itertools

import networkx as nx
import pytest
from src.data.internal_holdout import (
    build_pair_label_manifest,
    canonical_pair_label_sha256,
    derive_internal_holdout,
)


def _path_edges(size: int) -> list[tuple[str, str]]:
    return [(f"n{i:03d}", f"n{i + 1:03d}") for i in range(size - 1)]


def test_partition_is_deterministic_disjoint_and_exhaustive() -> None:
    nodes = {f"n{i:03d}" for i in range(20)}
    edges = _path_edges(20)
    first = derive_internal_holdout(nodes, edges, [], holdout_size=4)
    second = derive_internal_holdout(
        reversed(tuple(sorted(nodes))), reversed(edges), [], holdout_size=4
    )

    assert first == second
    assert len(first.v_qual) == len(first.v_select) == 4
    assert first.v_fit | first.v_qual | first.v_select == nodes
    assert not (first.v_fit & first.v_qual)
    assert not (first.v_fit & first.v_select)
    assert not (first.v_qual & first.v_select)
    assert first.overlap_proof.all_zero


def test_bfs_uses_largest_component_and_hash_ordered_frontiers() -> None:
    large = {"root", "a", "b", "c", "d"}
    small = {"x", "y", "z"}
    edges = [("root", node) for node in large - {"root"}] + [("x", "y"), ("y", "z")]
    result = derive_internal_holdout(large | small, edges, [], holdout_size=2)
    qual_seed = min(
        large,
        key=lambda node: (hashlib.sha256(f"g5-v2-qual|{node}".encode()).digest(), node),
    )

    assert qual_seed in result.v_qual
    assert result.v_qual <= large


def test_loopless_induced_fit_edges_and_supervision_restriction() -> None:
    nodes = {f"n{i:03d}" for i in range(12)}
    message = _path_edges(12) + [("n000", "n000")]
    supervision = list(itertools.combinations(sorted(nodes), 2)) + [("n011", "n011")]
    result = derive_internal_holdout(nodes, message, supervision, holdout_size=2)

    assert all(u != v and u in result.v_fit and v in result.v_fit for u, v in result.e_msg_fit)
    assert all(u in result.v_fit and v in result.v_fit for u, v in result.e_sup_fit)
    assert nx.number_of_selfloops(result.build_g_fit()) == 0
    assert set(result.build_g_fit().nodes) == result.v_fit
    assert ("n011", "n011") in result.e_sup_fit if "n011" in result.v_fit else True


def test_cross_partition_quarantine_counts_match_direct_count() -> None:
    nodes = {f"n{i:03d}" for i in range(14)}
    message = _path_edges(14)
    supervision = list(itertools.combinations(sorted(nodes), 2))
    result = derive_internal_holdout(nodes, message, supervision, holdout_size=3)
    owner = {
        node: name
        for name, part in (
            ("fit", result.v_fit),
            ("qual", result.v_qual),
            ("select", result.v_select),
        )
        for node in part
    }
    expected = {"fit__qual": 0, "fit__select": 0, "qual__select": 0}
    for u, v in supervision:
        if owner[u] != owner[v]:
            expected["__".join(sorted((owner[u], owner[v])))] += 1

    assert result.quarantine_counts.supervision == expected


def test_complete_manifest_is_loopless_labeled_and_hashes_canonical_rows() -> None:
    nodes = ["c", "a", "b"]
    manifest = build_pair_label_manifest(nodes, [("b", "a"), ("c", "c")])

    assert manifest.nodes == ("a", "b", "c")
    assert manifest.pairs == (("a", "b"), ("a", "c"), ("b", "c"))
    assert manifest.labels == (1, 0, 0)
    assert manifest.positive_edges == (("a", "b"),)
    assert manifest.positive_count == 1
    assert manifest.prevalence == pytest.approx(1 / 3)
    expected = hashlib.sha256(b"a\tb\t1\na\tc\t0\nb\tc\t0\n").hexdigest()
    assert manifest.pair_labels_sha256 == expected
    reversed_digest = canonical_pair_label_sha256(
        reversed(manifest.pairs), reversed(manifest.labels)
    )
    assert reversed_digest == expected


def test_rejects_too_small_remaining_component_and_foreign_endpoints() -> None:
    with pytest.raises(ValueError, match="largest remaining"):
        derive_internal_holdout({"a", "b", "c"}, [("a", "b"), ("b", "c")], [], holdout_size=2)
    with pytest.raises(ValueError, match="outside operative"):
        derive_internal_holdout({"a", "b"}, [("a", "foreign")], [], holdout_size=1)
