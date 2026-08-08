"""Deterministic graph-size telemetry for the exact full-ego oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np


def _validate_truth_graph(graph: nx.Graph[str]) -> None:
    """Validate the simple, loopless truth-graph contract."""
    if graph.is_directed():
        raise ValueError("full-ego truth graph must be undirected")
    if graph.is_multigraph():
        raise ValueError("full-ego truth graph must be simple")
    if nx.number_of_selfloops(graph) != 0:
        raise ValueError("full-ego truth graph must be loopless")
    if any(not isinstance(node, str) for node in graph.nodes):
        raise TypeError("full-ego truth graph node identifiers must be strings")


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def truth_graph_digest(graph: nx.Graph[str]) -> str:
    """Return an insertion-order-independent digest of all truth nodes and edges."""
    _validate_truth_graph(graph)
    payload = {
        "edges": sorted(_canonical_edge(left, right) for left, right in graph.edges),
        "nodes": sorted(graph.nodes),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def exact_query_graph(
    graph: nx.Graph[str], src: str, dst: str
) -> tuple[tuple[str, ...], frozenset[tuple[str, str]]]:
    """Construct ``(G - {src,dst})[src, dst, N(src), N(dst)]`` exactly.

    The query edge is removed before the endpoint neighborhoods are taken. There
    is no neighbor cap, sampling, or multiplicity approximation.
    """
    _validate_truth_graph(graph)
    missing = [endpoint for endpoint in (src, dst) if endpoint not in graph]
    if missing:
        raise ValueError(f"query endpoints are absent from the truth graph: {missing}")

    src_neighbors = set(graph.neighbors(src))
    dst_neighbors = set(graph.neighbors(dst))
    endpoints: tuple[str, ...]
    if src != dst:
        src_neighbors.discard(dst)
        dst_neighbors.discard(src)
        endpoints = (src, dst)
    else:
        endpoints = (src,)

    nodes = endpoints + tuple(sorted((src_neighbors | dst_neighbors) - set(endpoints)))
    node_set = set(nodes)
    query_edge = {src, dst}
    edges = frozenset(
        _canonical_edge(left, right)
        for left, right in graph.subgraph(node_set).edges
        if {left, right} != query_edge
    )
    return nodes, edges


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def summarize_telemetry(graph: nx.Graph[str], queries: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Compute per-query and aggregate full-ego preflight telemetry."""
    _validate_truth_graph(graph)
    if not queries:
        raise ValueError("full-ego telemetry requires at least one query pair")

    rows: list[dict[str, str | int | bool]] = []
    for src, dst in queries:
        nodes, edges = exact_query_graph(graph, src, dst)
        query_present = src != dst and graph.has_edge(src, dst)
        src_degree = int(graph.degree(src)) - int(query_present)
        dst_degree = int(graph.degree(dst)) - int(query_present)
        max_endpoint_degree = max(src_degree, dst_degree)
        rows.append(
            {
                "src": src,
                "dst": dst,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "max_endpoint_degree": max_endpoint_degree,
                "high_degree_gt16": max_endpoint_degree > 16,
            }
        )

    node_counts = [cast(int, row["node_count"]) for row in rows]
    edge_counts = [cast(int, row["edge_count"]) for row in rows]
    high_degree_count = sum(bool(row["high_degree_gt16"]) for row in rows)
    return {
        "truth_graph": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "sha256": truth_graph_digest(graph),
        },
        "queries": rows,
        "summary": {
            "count": len(rows),
            "node_count": _distribution(node_counts),
            "edge_count": _distribution(edge_counts),
            "high_degree_gt16": {
                "count": high_degree_count,
                "fraction": high_degree_count / len(rows),
            },
        },
    }


def _load_truth_graph(path: Path) -> nx.Graph[str]:
    with path.open("rb") as handle:
        graph = pickle.load(handle)  # noqa: S301 - explicit local experiment input
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"truth graph pickle must contain a networkx graph, got {type(graph)!r}")
    typed_graph = cast("nx.Graph[str]", graph)
    _validate_truth_graph(typed_graph)
    return typed_graph


def _load_queries(path: Path) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        columns = stripped.split()
        if len(columns) != 2:
            raise ValueError(f"query line {line_number} must contain exactly two columns")
        queries.append((columns[0], columns[1]))
    return queries


def main(argv: Sequence[str] | None = None) -> int:
    """Run the telemetry CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = summarize_telemetry(_load_truth_graph(args.truth_graph), _load_queries(args.queries))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
