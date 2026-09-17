"""The section 8 output-density control: equal selected count, official self-loops."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest


def test_the_target_is_this_arms_own_selected_pair_count_not_nx_density() -> None:
    from src.experiments.motif_density_control import target_edge_count

    logits = np.asarray([2.0, -1.0, 3.0, 0.5])
    count, n_union = target_edge_count(logits, threshold=0.4)
    assert (count, n_union) == (3, 4)
    # nx.density over the four nodes of those pairs would be 3/6, not 3/4: the
    # union holds only some node pairs, so the rate is selected_pairs / n_union.
    assert count / n_union == 0.75


def test_ties_are_admitted_atomically_so_the_realised_count_can_fall_short() -> None:
    from src.eval.assembly import density_matched_threshold

    probs = np.asarray([0.9, 0.9, 0.9, 0.1])
    threshold = density_matched_threshold(probs, target_edges=2)
    assert int((probs >= threshold).sum()) == 0


def test_self_pairs_assemble_as_self_loops_and_official_gs_keeps_them() -> None:
    from src.eval.assembly import assemble_graph
    from src.eval.graph_metrics import compute_graph_similarity

    pairs = [("a", "b"), ("a", "a")]
    graph = assemble_graph(pairs, np.asarray([0.9, 0.9]), threshold=0.5, nodes=["a", "b"])
    assert graph.number_of_edges() == 2
    reference = nx.Graph()
    reference.add_nodes_from(["a", "b"])
    reference.add_edge("a", "b")
    assert compute_graph_similarity(graph, reference) == pytest.approx(2.0 / 3.0)


def test_a_row_more_than_one_percent_short_is_marked_approximate() -> None:
    from src.experiments.motif_density_control import density_matched_report

    report = density_matched_report(
        rows={"arm": np.asarray([0.9, 0.9, 0.9, 0.1])},
        union_pairs=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
        target_edges=2,
        g_ref=nx.Graph([("a", "b"), ("b", "c")]),
        buckets={4: [{"a", "b", "c", "d"}]},
        config=None,
    )
    assert report["matching"] == "output_density_matched_union"
    assert report["n_union"] == 4
    rows = report["rows"]
    assert isinstance(rows, dict)
    assert rows["arm"]["target_edges"] == 2
    assert rows["arm"]["realised_edges"] == 0
    assert rows["arm"]["shape_comparison"] == "approximate"


def test_every_row_is_matched_on_its_own_scored_union_not_on_a_transferred_rate() -> None:
    from src.experiments.motif_density_control import density_matched_report

    # Two rows whose logits differ by a constant shift: the pure threshold-placement
    # case this control exists to expose. At a matched count they select the same
    # rows, so any surviving GS difference is shape, not placement.
    base = np.asarray([3.0, 2.0, 1.0, -1.0, -2.0])
    report = density_matched_report(
        rows={"arm": base, "prefix_base": base + 7.5},
        union_pairs=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "a")],
        target_edges=3,
        g_ref=nx.Graph([("a", "b"), ("b", "c"), ("c", "d")]),
        buckets={4: [{"a", "b", "c", "d"}]},
        config=None,
    )
    rows = report["rows"]
    assert isinstance(rows, dict)
    for name in ("arm", "prefix_base"):
        assert rows[name]["realised_edges"] == 3
        assert rows[name]["shape_comparison"] == "matched"
    assert rows["arm"]["logit_threshold"] == 1.0
    assert rows["prefix_base"]["logit_threshold"] == 8.5
