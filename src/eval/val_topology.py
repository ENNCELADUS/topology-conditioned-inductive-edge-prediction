"""Sampled-subgraph topology validation and fixed-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.val_region import ValRegionSplit
from src.eval.checkpoint_selection import TopologyValidationMetrics
from src.eval.fixed_threshold import select_fixed_threshold
from src.eval.graph_metrics import MMDConfig


@dataclass(frozen=True)
class ValTopologyReference:
    """Fixed V_val sampled-subgraph reference state."""

    nodes: tuple[str, ...]
    g_val: nx.Graph
    buckets: dict[int, list[set[str]]]


def build_val_topology_reference(split: ValRegionSplit) -> ValTopologyReference:
    """Build the sampled-subgraph reference from the derived V_val split."""
    return ValTopologyReference(
        nodes=tuple(sorted(split.v_val)),
        g_val=split.build_g_val(),
        buckets=split.buckets,
    )


@dataclass(frozen=True)
class ValTopologyResult:
    """One epoch's sampled-only topology validation result."""

    metrics: TopologyValidationMetrics
    threshold: float


def val_region_topology_metrics(
    *,
    u_idx: NDArray[np.integer],
    v_idx: NDArray[np.integer],
    logits: NDArray[np.floating],
    reference: ValTopologyReference,
) -> ValTopologyResult:
    """Select one threshold using only the exact V_val sampled-pair union."""
    u_idx = np.asarray(u_idx)
    v_idx = np.asarray(v_idx)
    logits_arr = np.asarray(logits, dtype=np.float64)
    if len(u_idx) != len(v_idx) or len(u_idx) != len(logits_arr):
        raise ValueError(
            f"u_idx/v_idx/logits length mismatch: {len(u_idx)}, {len(v_idx)}, {len(logits_arr)}"
        )
    if not np.isfinite(logits_arr).all():
        raise ValueError("non-finite validation logits")
    node_count = len(reference.nodes)
    if (
        np.any(u_idx < 0)
        or np.any(v_idx < 0)
        or np.any(u_idx >= node_count)
        or np.any(v_idx >= node_count)
    ):
        raise ValueError("validation pair index is outside the V_val node universe")

    pairs = [
        (reference.nodes[int(u)], reference.nodes[int(v)])
        for u, v in zip(u_idx, v_idx, strict=True)
    ]
    selection = select_fixed_threshold(
        pairs=pairs,
        logits=logits_arr,
        g_ref=reference.g_val,
        buckets=reference.buckets,
        config=MMDConfig(),
    )
    report = selection.metrics
    return ValTopologyResult(
        metrics=TopologyValidationMetrics(
            gs=report.graph_similarity,
            rd=report.relative_density,
            degree_mmd=report.mmd_ratio["degree"],
            clustering_mmd=report.mmd_ratio["clustering"],
            spectral_mmd=report.mmd_ratio["spectral"],
        ),
        threshold=selection.logit_threshold,
    )


__all__ = [
    "ValTopologyReference",
    "ValTopologyResult",
    "build_val_topology_reference",
    "val_region_topology_metrics",
]
