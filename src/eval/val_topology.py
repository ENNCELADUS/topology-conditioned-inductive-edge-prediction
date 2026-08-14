"""Per-epoch topology validation over the complete V_val pair universe.

Mirrors the house density-matched assembly convention
(`src.experiments.g1_hardened_e2.run_threshold_sweep`): the assembly threshold
is matched against the loopless gold edge count on non-self rows only, so
self-pairs never consume quota meant for real edges, but a self-pair that
still clears that threshold assembles as a self-loop like any other row.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.data.val_region import ValRegionSplit
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.checkpoint_selection import TopologyValidationMetrics
from src.eval.graph_metrics import (
    BucketReference,
    MMDConfig,
    evaluate_assembled_graph_with_reference,
    precompute_bucket_reference,
    strip_self_loops,
)


@dataclass(frozen=True)
class ValTopologyReference:
    """Fixed, once-per-run reference state for V_val topology validation.

    Attributes:
        nodes: `sorted(v_val)` — the index space `u_idx`/`v_idx` are drawn over.
        g_val: The induced V_val gold graph, self-loops kept.
        target_edges: The loopless gold edge count, the density-match target.
        bucket_ref: Precomputed reference-side bucket state.
    """

    nodes: tuple[str, ...]
    g_val: nx.Graph
    target_edges: int
    bucket_ref: BucketReference


def build_val_topology_reference(
    split: ValRegionSplit, config: MMDConfig | None = None
) -> ValTopologyReference:
    """Build the once-per-run `ValTopologyReference` from a derived `ValRegionSplit`.

    Args:
        split: The derived V_val region split.
        config: MMD/descriptor configuration for the precomputed bucket state;
            defaults to `MMDConfig()`.

    Returns:
        The `ValTopologyReference`.
    """
    config = config if config is not None else MMDConfig()
    nodes = tuple(sorted(split.v_val))
    g_val = split.build_g_val()
    target_edges = strip_self_loops(g_val).number_of_edges()
    bucket_ref = precompute_bucket_reference(g_val, split.buckets, config)
    return ValTopologyReference(
        nodes=nodes, g_val=g_val, target_edges=target_edges, bucket_ref=bucket_ref
    )


def val_region_topology_metrics(
    *,
    u_idx: NDArray[np.integer],
    v_idx: NDArray[np.integer],
    logits: NDArray[np.floating],
    reference: ValTopologyReference,
) -> TopologyValidationMetrics:
    """Compute the five V_val topology metrics from one epoch's full-universe logits.

    `sigmoid(logits)` -> `density_matched_threshold` on non-self rows against
    `reference.target_edges` -> `assemble_graph` (self-pairs above threshold
    become self-loops) -> `evaluate_assembled_graph_with_reference`.

    Args:
        u_idx: Row-aligned node index into `reference.nodes` (from
            `val_universe_arrays`).
        v_idx: Row-aligned node index into `reference.nodes`.
        logits: Row-aligned raw logits, aligned with `u_idx`/`v_idx`.
        reference: The once-per-run `ValTopologyReference`.

    Returns:
        The `TopologyValidationMetrics`.

    Raises:
        ValueError: On a length mismatch between `u_idx`/`v_idx`/`logits`, or
            on any non-finite logit.
    """
    u_idx = np.asarray(u_idx)
    v_idx = np.asarray(v_idx)
    logits_arr = np.asarray(logits, dtype=np.float64)
    if len(u_idx) != len(v_idx) or len(u_idx) != len(logits_arr):
        raise ValueError(
            f"u_idx/v_idx/logits length mismatch: {len(u_idx)}, {len(v_idx)}, {len(logits_arr)}"
        )
    if not np.all(np.isfinite(logits_arr)):
        raise ValueError("non-finite validation logits")

    probs = expit(logits_arr)
    non_self_mask = u_idx != v_idx
    threshold = density_matched_threshold(probs[non_self_mask], reference.target_edges)

    pairs = [(reference.nodes[u], reference.nodes[v]) for u, v in zip(u_idx, v_idx, strict=True)]
    g_pred = assemble_graph(pairs, probs, threshold=threshold, nodes=reference.nodes)

    report = evaluate_assembled_graph_with_reference(g_pred, reference.bucket_ref, MMDConfig())
    return TopologyValidationMetrics(
        gs=report.graph_similarity,
        rd=report.relative_density,
        degree_mmd=report.mmd_ratio["degree"],
        clustering_mmd=report.mmd_ratio["clustering"],
        spectral_mmd=report.mmd_ratio["spectral"],
    )


__all__ = [
    "ValTopologyReference",
    "build_val_topology_reference",
    "val_region_topology_metrics",
]
