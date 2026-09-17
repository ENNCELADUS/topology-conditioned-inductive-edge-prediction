"""The section 8 output-density control: read shape at a common assembled density.

This is a property of the assembled graph, not of the inputs, and it is a
different object from the coordinate calibration of spec section 0.2. It asks
whether the arm's assembled edge set is a better *shape* at equal density, or
whether the apparent gain was threshold placement -- a previous arm's entire
topology headline was traced to a per-universe logit shift. It is reported
**alongside**, never instead of, the protocol's V_val-selected threshold, since
re-selecting a threshold per row is outside ``test_protocol_v8``.

Self-loops keep the official convention throughout: rows are the pairs the
protocol scores, self-pairs included, a self-pair over the threshold assembles
as a self-loop and official GS and RD retain it. A loopless variant is permitted
only as a separate, labelled diagnostic.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.eval.assembly import density_matched_threshold
from src.eval.fixed_threshold import evaluate_fixed_threshold
from src.eval.graph_metrics import MMDConfig

MATCHING = "output_density_matched_union"


def target_edge_count(logits: NDArray[np.float64], *, threshold: float) -> tuple[int, int]:
    """Return ``(target_edges, n_union)`` from this arm's own selected count.

    ``target_edges`` is the number of union pairs **this arm** realises at its
    protocol V_val-selected threshold on the evaluation union -- its own
    assembled edge count, taken directly, with no rate transferred between
    universes. A rate may substitute only if it is ``selected_pairs / n_union``;
    it is **not** ``nx.density``, whose denominator counts node pairs over the
    whole node set while a sampled union holds only some of those pairs. Because
    every row scores the same union over the same nodes, an equal selected count
    is an equal assembled edge count and therefore an equal assembled density.

    Args:
        logits: The arm's raw logits over the union rows, self-pairs included.
        threshold: The arm's V_val-selected fixed logit threshold.

    Returns:
        The realised edge count and the union row count.
    """
    values = np.asarray(logits, dtype=np.float64)
    return int((values >= threshold).sum()), int(values.size)


def density_matched_report(
    *,
    rows: Mapping[str, NDArray[np.float64]],
    union_pairs: Sequence[tuple[str, str]],
    target_edges: int,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig | None,
) -> dict[str, object]:
    """Score every row at the common target density and report the residual spread.

    One threshold per row, applied unchanged to every subgraph, as
    `evaluate_fixed_threshold` does. `density_matched_threshold` admits tie
    groups atomically, so the realised count is ``<= target_edges`` and exact
    equality is not guaranteed; a row landing more than 1% below the target is
    marked ``approximate`` rather than having its target adjusted. Matching the
    *union* density does not equalise density inside each BFS subgraph, so
    per-subgraph RD still varies and a density-matched row's macro RD is not 1:
    read GS and the three MMD ratios here and treat RD as a diagnostic of that
    residual spread.

    The threshold is selected on the raw logits rather than on probabilities.
    ``expit`` is strictly monotone, so the two selections admit exactly the same
    tie groups and the same rows, but staying on the logit scale keeps the
    threshold `evaluate_fixed_threshold` consumes exact: a probability that
    saturates to 1.0 in float64 has no finite logit to invert back to.

    Args:
        rows: Raw logits per row name (this arm, ``prefix_base``, ``B0``).
        union_pairs: The pooled union of pairs the protocol scores, self-pairs included.
        target_edges: The common target from `target_edge_count`.
        g_ref: The reference graph.
        buckets: The protocol's BFS subgraph buckets.
        config: The MMD configuration, or ``None`` to skip the topology panel.

    Returns:
        A report with, per row, the chosen threshold, the realised edge count,
        the target, whether the shape comparison is approximate, and the five
        topology numbers when ``config`` is supplied.
    """
    entries: dict[str, object] = {}
    for name, row in rows.items():
        logits = np.asarray(row, dtype=np.float64)
        threshold = density_matched_threshold(logits, target_edges)
        realised = int((logits >= threshold).sum())
        entry: dict[str, object] = {
            "logit_threshold": float(threshold),
            "probability_threshold": float(expit(threshold)),
            "realised_edges": realised,
            "target_edges": int(target_edges),
            "shape_comparison": ("approximate" if realised < target_edges * 0.99 else "matched"),
        }
        if config is not None:
            _, panel = evaluate_fixed_threshold(
                pairs=union_pairs,
                logits=logits,
                g_ref=g_ref,
                buckets=buckets,
                threshold=float(threshold),
                config=config,
            )
            entry["panel"] = panel
        entries[name] = entry
    return {
        "matching": MATCHING,
        "target_edges": int(target_edges),
        "n_union": len(union_pairs),
        "reported": "alongside the V_val-selected threshold, never instead of it",
        "rows": entries,
    }


__all__ = ["MATCHING", "density_matched_report", "target_edge_count"]
