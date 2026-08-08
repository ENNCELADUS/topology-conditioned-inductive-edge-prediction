from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import pytest
from src.experiments.full_ego_oracle.telemetry import (
    exact_query_graph,
    main,
    summarize_telemetry,
    truth_graph_digest,
)
from src.model.egostitch.generator.full_oracle import FullOracleGenerator


def test_positive_and_negative_queries_use_exact_query_removed_induced_graph() -> None:
    truth = nx.Graph(
        [
            ("u", "v"),
            ("u", "a"),
            ("v", "b"),
            ("a", "b"),
            ("x", "c"),
            ("x", "d"),
            ("c", "d"),
        ]
    )

    positive_nodes, positive_edges = exact_query_graph(truth, "u", "v")
    negative_nodes, negative_edges = exact_query_graph(truth, "x", "u")

    assert positive_nodes == ("u", "v", "a", "b")
    assert positive_edges == frozenset({("a", "b"), ("a", "u"), ("b", "v")})
    assert negative_nodes == ("x", "u", "a", "c", "d", "v")
    assert negative_edges == frozenset({("c", "d"), ("c", "x"), ("d", "x"), ("a", "u"), ("u", "v")})


def test_exact_helper_matches_full_oracle_generator_query_graph() -> None:
    truth = nx.Graph([("u", "v"), ("u", "a"), ("v", "a"), ("a", "b")])
    generator = FullOracleGenerator()
    generator.set_oracle_context(truth, list(truth.nodes))

    nodes, edges = exact_query_graph(truth, "u", "v")
    generator_nodes, generator_edges = generator._query_graph("u", "v")

    assert nodes == tuple(generator_nodes)
    assert edges == frozenset(tuple(sorted((left, right))) for left, right in generator_edges)


def test_summary_flags_every_neighbor_above_sixteen_and_has_stable_quantiles() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(["hub", "peer", "isolated"])
    truth.add_edges_from(("hub", f"n{index:02d}") for index in range(17))
    queries = [("hub", "peer"), ("peer", "isolated")]

    report = summarize_telemetry(truth, queries)

    assert report["queries"] == [
        {
            "src": "hub",
            "dst": "peer",
            "node_count": 19,
            "edge_count": 17,
            "max_endpoint_degree": 17,
            "high_degree_gt16": True,
        },
        {
            "src": "peer",
            "dst": "isolated",
            "node_count": 2,
            "edge_count": 0,
            "max_endpoint_degree": 0,
            "high_degree_gt16": False,
        },
    ]
    assert report["summary"] == {
        "count": 2,
        "node_count": {"p50": 10.5, "p90": 17.3, "p95": 18.15, "p99": 18.83, "max": 19},
        "edge_count": {"p50": 8.5, "p90": 15.3, "p95": 16.15, "p99": 16.83, "max": 17},
        "high_degree_gt16": {"count": 1, "fraction": 0.5},
    }


def test_truth_digest_is_order_independent_and_covers_isolates_and_edges() -> None:
    first = nx.Graph()
    first.add_nodes_from(["z", "a", "isolate"])
    first.add_edge("z", "a")
    reordered = nx.Graph()
    reordered.add_nodes_from(["isolate", "a", "z"])
    reordered.add_edge("a", "z")

    digest = truth_graph_digest(first)

    assert digest == truth_graph_digest(reordered)
    with_extra_edge = reordered.copy()
    with_extra_edge.add_edge("a", "isolate")
    assert digest != truth_graph_digest(with_extra_edge)
    without_isolate = reordered.copy()
    without_isolate.remove_node("isolate")
    assert digest != truth_graph_digest(without_isolate)


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (nx.DiGraph([("u", "v")]), "undirected"),
        (nx.Graph([("u", "u")]), "loopless"),
    ],
)
def test_invalid_truth_graphs_fail_closed(truth: nx.Graph[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_telemetry(truth, [("u", "v")])


def test_missing_query_endpoint_and_empty_queries_fail_closed() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(["u", "v"])

    with pytest.raises(ValueError, match="absent from the truth graph"):
        summarize_telemetry(truth, [("u", "missing")])
    with pytest.raises(ValueError, match="at least one query"):
        summarize_telemetry(truth, [])


def test_cli_loads_pickle_and_two_column_queries_and_writes_json(tmp_path: Path) -> None:
    truth = nx.Graph([("u", "v"), ("u", "a")])
    graph_path = tmp_path / "truth.pkl"
    query_path = tmp_path / "queries.txt"
    output_path = tmp_path / "telemetry.json"
    with graph_path.open("wb") as handle:
        pickle.dump(truth, handle)
    query_path.write_text("u v\n\nu a\n", encoding="utf-8")

    assert (
        main(
            [
                "--truth-graph",
                str(graph_path),
                "--queries",
                str(query_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["count"] == 2
    assert report["queries"][0]["node_count"] == 3

    query_path.write_text("u v extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two columns"):
        main(["--truth-graph", str(graph_path), "--queries", str(query_path)])
