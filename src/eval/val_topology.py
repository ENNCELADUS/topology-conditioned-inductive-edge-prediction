"""Per-epoch topology validation over the ball-union pairs U plus a sampled complement.

Only the ball-union pairs U (self-pairs included) are ever assembled into a
graph — the five metrics read only bucket-induced subgraphs of the assembled
graph, and every bucket is a subset of U by construction, so scoring the
out-of-ball complement exhaustively is unnecessary for the metrics
themselves. It is, however, needed to estimate the assembly threshold: mirrors
the house density-matched assembly convention
(`src.experiments.g1_hardened_e2.run_threshold_sweep`) in that the threshold
is matched against the loopless gold edge count on non-self rows only, so
self-pairs never consume quota meant for real edges, but a self-pair that
still clears that threshold assembles as a self-loop like any other row. The
threshold itself, though, is only *estimated*: the out-of-ball complement is
observed solely through a pinned uniform without-replacement sample, and
`estimated_density_matched_threshold` reweights that sample to approximate
the FULL-universe density-matched threshold that exhaustive scoring would
have produced. This means long-range (out-of-bucket) false positives still
consume threshold quota statistically, even though they are never assembled
or scored individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.data.val_region import ValRegionSplit
from src.eval.assembly import assemble_graph, estimated_density_matched_threshold
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


@dataclass(frozen=True)
class ValTopologyResult:
    """One epoch's V_val topology validation result.

    Attributes:
        metrics: The five topology metrics.
        threshold: The estimated full-universe density-matched assembly
            threshold actually used.
        admitted_non_self_fraction: Estimated fraction of the full non-self
            pair universe (`n*(n-1)/2`) that clears `threshold` — the U rows'
            exact contribution plus the reweighted complement-sample
            contribution.
    """

    metrics: TopologyValidationMetrics
    threshold: float
    admitted_non_self_fraction: float


def val_region_topology_metrics(
    *,
    u_idx: NDArray[np.integer],
    v_idx: NDArray[np.integer],
    logits: NDArray[np.floating],
    sample_logits: NDArray[np.floating],
    complement_total: int,
    reference: ValTopologyReference,
) -> ValTopologyResult:
    """Compute the five V_val topology metrics from U rows plus a complement sample.

    `sigmoid(logits)`/`sigmoid(sample_logits)` -> `estimated_density_matched_threshold`
    (non-self U rows exact, complement sample reweighted by
    `complement_total / len(sample_logits)`) against `reference.target_edges`
    -> `assemble_graph` on the U rows only (self-pairs above threshold become
    self-loops; the complement sample never assembles) ->
    `evaluate_assembled_graph_with_reference`.

    Args:
        u_idx: Row-aligned node index into `reference.nodes` for the
            ball-union pairs U (self-pairs included).
        v_idx: Row-aligned node index into `reference.nodes`, aligned with
            `u_idx`.
        logits: Row-aligned raw logits for U, aligned with `u_idx`/`v_idx`.
        sample_logits: Raw logits for the pinned uniform without-replacement
            sample of out-of-ball, non-self complement pairs. No node indices
            are needed — these rows never assemble.
        complement_total: True size of the out-of-ball non-self complement
            population `sample_logits` was drawn from. `0` means U already
            covers the full non-self universe.
        reference: The once-per-run `ValTopologyReference`.

    Returns:
        The `ValTopologyResult`.

    Raises:
        ValueError: On a length mismatch between `u_idx`/`v_idx`/`logits`, on
            any non-finite `logits` or `sample_logits` value, on a negative
            `complement_total`, or (via `estimated_density_matched_threshold`)
            on a `sample_logits` size inconsistent with `complement_total`.
    """
    u_idx = np.asarray(u_idx)
    v_idx = np.asarray(v_idx)
    logits_arr = np.asarray(logits, dtype=np.float64)
    sample_logits_arr = np.asarray(sample_logits, dtype=np.float64)
    if len(u_idx) != len(v_idx) or len(u_idx) != len(logits_arr):
        raise ValueError(
            f"u_idx/v_idx/logits length mismatch: {len(u_idx)}, {len(v_idx)}, {len(logits_arr)}"
        )
    if not np.all(np.isfinite(logits_arr)):
        raise ValueError("non-finite validation logits")
    if not np.all(np.isfinite(sample_logits_arr)):
        raise ValueError("non-finite validation sample logits")
    if complement_total < 0:
        raise ValueError(f"complement_total must be non-negative, got {complement_total}")

    probs = expit(logits_arr)
    sample_probs = expit(sample_logits_arr)
    non_self_mask = u_idx != v_idx
    threshold = estimated_density_matched_threshold(
        probs[non_self_mask], sample_probs, complement_total, reference.target_edges
    )

    pairs = [(reference.nodes[u], reference.nodes[v]) for u, v in zip(u_idx, v_idx, strict=True)]
    g_pred = assemble_graph(pairs, probs, threshold=threshold, nodes=reference.nodes)

    report = evaluate_assembled_graph_with_reference(g_pred, reference.bucket_ref, MMDConfig())
    metrics = TopologyValidationMetrics(
        gs=report.graph_similarity,
        rd=report.relative_density,
        degree_mmd=report.mmd_ratio["degree"],
        clustering_mmd=report.mmd_ratio["clustering"],
        spectral_mmd=report.mmd_ratio["spectral"],
    )

    n = len(reference.nodes)
    total_non_self = n * (n - 1) // 2
    admitted_exact = float(np.sum(probs[non_self_mask] >= threshold))
    admitted_sample = 0.0
    if complement_total > 0:
        admitted_sample = (complement_total / sample_probs.size) * float(
            np.sum(sample_probs >= threshold)
        )
    admitted_non_self_fraction = (
        (admitted_exact + admitted_sample) / total_non_self if total_non_self > 0 else 0.0
    )

    return ValTopologyResult(
        metrics=metrics,
        threshold=threshold,
        admitted_non_self_fraction=admitted_non_self_fraction,
    )


__all__ = [
    "ValTopologyReference",
    "ValTopologyResult",
    "build_val_topology_reference",
    "val_region_topology_metrics",
]
