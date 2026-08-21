"""Validation-selected fixed topology threshold over sampled subgraphs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.data.artifacts import canonical_pair
from src.eval.graph_metrics import (
    STATISTICS,
    BucketedMMDReport,
    MMDConfig,
    clustering_histogram,
    degree_histogram,
    evaluate_sampled_subgraphs,
    laplacian_spectrum_histogram,
    mmd_squared,
)


@dataclass(frozen=True)
class FixedThresholdSelection:
    """One validation-selected logit threshold and its audit report."""

    logit_threshold: float
    metrics: BucketedMMDReport
    report: dict[str, object]


@dataclass(frozen=True)
class _LocalSample:
    size: int
    nodes: set[str]
    pairs: tuple[tuple[str, str], ...]
    logits: NDArray[np.float64]
    truth: NDArray[np.bool_]


def _score_map(
    pairs: Sequence[tuple[str, str]], logits: NDArray[np.float64]
) -> dict[tuple[str, str], float]:
    if len(pairs) != logits.size:
        raise ValueError("pairs and logits must be aligned")
    if not np.isfinite(logits).all():
        raise ValueError("topology logits must be finite")
    out: dict[tuple[str, str], float] = {}
    for pair, logit in zip(pairs, logits, strict=True):
        canonical = canonical_pair(*pair)
        if canonical in out:
            raise ValueError(f"duplicate topology score for pair {canonical!r}")
        out[canonical] = float(logit)
    return out


def _local_samples(
    *,
    pairs: Sequence[tuple[str, str]],
    logits: NDArray[np.float64],
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
) -> tuple[_LocalSample, ...]:
    score_by_pair = _score_map(pairs, logits)
    samples: list[_LocalSample] = []
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        for nodes in node_sets:
            local_pairs = tuple(
                canonical_pair(u, v) for u, v in combinations_with_replacement(sorted(nodes), 2)
            )
            try:
                local_logits = np.array(
                    [score_by_pair[pair] for pair in local_pairs], dtype=np.float64
                )
            except KeyError as error:
                raise ValueError(f"missing topology score for pair {error.args[0]!r}") from error
            truth = np.fromiter(
                (g_ref.has_edge(*pair) for pair in local_pairs),
                dtype=np.bool_,
                count=len(local_pairs),
            )
            if not truth.any():
                raise ValueError("fixed-threshold RD calibration requires an edge in every sample")
            samples.append(
                _LocalSample(
                    size=size,
                    nodes=set(nodes),
                    pairs=local_pairs,
                    logits=local_logits,
                    truth=truth,
                )
            )
    if len(samples) < 2:
        raise ValueError("fixed-threshold selection requires at least two sampled subgraphs")
    return tuple(samples)


def _candidate_thresholds(samples: tuple[_LocalSample, ...]) -> NDArray[np.float64]:
    score_by_pair = {
        pair: float(logit)
        for sample in samples
        for pair, logit in zip(sample.pairs, sample.logits, strict=True)
    }
    event_logits = np.fromiter(score_by_pair.values(), dtype=np.float64, count=len(score_by_pair))
    empty_threshold = np.nextafter(float(np.max(event_logits)), np.inf)
    if not np.isfinite(empty_threshold):
        raise ValueError("cannot construct a finite empty-graph threshold")
    return np.array(
        sorted({empty_threshold, *event_logits.tolist()}, reverse=True), dtype=np.float64
    )


def _metric_state(
    samples: tuple[_LocalSample, ...], threshold: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    predicted = np.array(
        [np.count_nonzero(sample.logits >= threshold) for sample in samples], dtype=np.float64
    )
    target = np.array([np.count_nonzero(sample.truth) for sample in samples], dtype=np.float64)
    true_positive = np.array(
        [np.count_nonzero((sample.logits >= threshold) & sample.truth) for sample in samples],
        dtype=np.float64,
    )
    graph_similarity = 2.0 * true_positive / (predicted + target)
    rd_error = np.abs(predicted / target - 1.0)
    return graph_similarity, rd_error


def _stratified_paired_bootstrap_se(
    differences: NDArray[np.float64], strata: NDArray[np.int64]
) -> float:
    """Exact paired-bootstrap mean SE for resampling within fixed strata."""
    if differences.ndim != 1 or strata.ndim != 1 or differences.size != strata.size:
        raise ValueError("paired bootstrap differences and strata must be aligned vectors")
    if differences.size < 2 or not np.isfinite(differences).all():
        raise ValueError("paired bootstrap requires at least two finite differences")
    total = differences.size
    variance = 0.0
    for stratum in np.unique(strata):
        values = differences[strata == stratum]
        weight = values.size / total
        variance += weight * weight * float(np.var(values, ddof=0)) / values.size
    return float(np.sqrt(variance))


def _sample_metric_curve(
    sample: _LocalSample,
    thresholds: NDArray[np.float64],
    *,
    metric: str,
) -> NDArray[np.float64]:
    sorted_logits = np.sort(sample.logits)
    truth_logits = np.sort(sample.logits[sample.truth])
    predicted = sorted_logits.size - np.searchsorted(sorted_logits, thresholds, side="left")
    true_positive = truth_logits.size - np.searchsorted(truth_logits, thresholds, side="left")
    if metric == "gs":
        return 2.0 * true_positive / (predicted + truth_logits.size)
    if metric == "rd_error":
        return np.abs(predicted / truth_logits.size - 1.0)
    raise ValueError(f"unknown fixed-threshold metric {metric!r}")


def _mean_metric_curve(
    samples: tuple[_LocalSample, ...],
    thresholds: NDArray[np.float64],
    *,
    metric: str,
) -> NDArray[np.float64]:
    mean = np.zeros(thresholds.size, dtype=np.float64)
    for sample in samples:
        mean += _sample_metric_curve(sample, thresholds, metric=metric) / len(samples)
    return mean


def _stratified_paired_se_curve(
    samples: tuple[_LocalSample, ...],
    thresholds: NDArray[np.float64],
    reference_threshold: float,
    *,
    metric: str,
    candidate_minus_reference: bool,
) -> NDArray[np.float64]:
    """Exact infinite-bootstrap SE with paired resampling inside size strata."""
    total = len(samples)
    variance = np.zeros(thresholds.size, dtype=np.float64)
    for size in sorted({sample.size for sample in samples}):
        stratum = [sample for sample in samples if sample.size == size]
        sum_diff = np.zeros(thresholds.size, dtype=np.float64)
        sum_sq_diff = np.zeros(thresholds.size, dtype=np.float64)
        for sample in stratum:
            curve = _sample_metric_curve(sample, thresholds, metric=metric)
            reference = float(
                _sample_metric_curve(
                    sample, np.array([reference_threshold]), metric=metric
                )[0]
            )
            differences = curve - reference if candidate_minus_reference else reference - curve
            sum_diff += differences
            sum_sq_diff += differences * differences
        count = len(stratum)
        population_variance = np.maximum(sum_sq_diff / count - (sum_diff / count) ** 2, 0.0)
        weight = count / total
        variance += weight * weight * population_variance / count
    return np.sqrt(variance)


def _descriptor(graph: nx.Graph, statistic: str) -> NDArray[np.float64]:
    if statistic == "degree":
        return degree_histogram(graph)
    if statistic == "clustering":
        return clustering_histogram(graph)
    if statistic == "spectral":
        return laplacian_spectrum_histogram(graph)
    raise ValueError(f"unknown topology statistic {statistic!r}")


def _precompute_reference_descriptors(
    samples: tuple[_LocalSample, ...], config: MMDConfig
) -> tuple[
    dict[str, dict[int, list[NDArray[np.float64]]]],
    dict[str, float],
]:
    by_stat_size: dict[str, dict[int, list[NDArray[np.float64]]]] = {
        statistic: {} for statistic in STATISTICS
    }
    for sample in samples:
        graph = nx.Graph()
        graph.add_nodes_from(sample.nodes)
        graph.add_edges_from(
            pair for pair, is_true in zip(sample.pairs, sample.truth, strict=True) if is_true
        )
        for statistic in STATISTICS:
            by_stat_size[statistic].setdefault(sample.size, []).append(
                _descriptor(graph, statistic)
            )
    denominators = {
        statistic: float(
            np.mean(
                [
                    mmd_squared(values[::2], values[1::2], config)
                    for values in by_stat_size[statistic].values()
                ]
            )
        )
        for statistic in STATISTICS
    }
    return by_stat_size, denominators


def _mmd_ratio_for_statistic(
    samples: tuple[_LocalSample, ...],
    threshold: float,
    statistic: str,
    reference_by_stat_size: dict[str, dict[int, list[NDArray[np.float64]]]],
    reference_mmd2: dict[str, float],
    config: MMDConfig,
) -> float:
    predicted_by_size: dict[int, list[NDArray[np.float64]]] = {}
    for sample in samples:
        graph = nx.Graph()
        graph.add_nodes_from(sample.nodes)
        graph.add_edges_from(
            pair
            for pair, logit in zip(sample.pairs, sample.logits, strict=True)
            if logit >= threshold
        )
        predicted_by_size.setdefault(sample.size, []).append(_descriptor(graph, statistic))
    raw = float(
        np.mean(
            [
                mmd_squared(predicted, reference_by_stat_size[statistic][size], config)
                for size, predicted in predicted_by_size.items()
            ]
        )
    )
    return raw / max(reference_mmd2[statistic], config.reference_epsilon)


def _fixed_predictions(
    samples: tuple[_LocalSample, ...], threshold: float
) -> dict[int, list[nx.Graph]]:
    predictions: dict[int, list[nx.Graph]] = {}
    for sample in samples:
        graph = nx.Graph()
        graph.add_nodes_from(sample.nodes)
        graph.add_edges_from(
            pair
            for pair, logit in zip(sample.pairs, sample.logits, strict=True)
            if logit >= threshold
        )
        predictions.setdefault(sample.size, []).append(graph)
    return predictions


def evaluate_fixed_threshold(
    *,
    pairs: Sequence[tuple[str, str]],
    logits: NDArray[np.float64],
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    threshold: float,
    config: MMDConfig,
) -> tuple[BucketedMMDReport, dict[str, object]]:
    """Apply one raw-logit threshold unchanged to every sampled subgraph."""
    if not np.isfinite(threshold):
        raise ValueError("fixed topology threshold must be finite")
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=g_ref, buckets=buckets)
    predictions = _fixed_predictions(samples, threshold)
    metrics = evaluate_sampled_subgraphs(predictions, g_ref, buckets, config)
    per_size: dict[str, object] = {}
    offset = 0
    for size, node_sets in buckets.items():
        count = len(node_sets)
        per_size[str(size)] = {
            "graph_similarity": float(np.mean(metrics.per_size_graph_similarity[size])),
            "relative_density": float(np.mean(metrics.per_size_relative_density[size])),
            "sample_count": count,
        }
        offset += count
    assert offset == len(samples)
    report = {
        "matching": "fixed_threshold_selected_on_validation",
        "selection": "logit_greater_than_or_equal",
        "logit_threshold": float(threshold),
        "probability_threshold": float(expit(threshold)),
        "graph_similarity": {"bfs_macro": metrics.graph_similarity},
        "relative_density": {"bfs_macro": metrics.relative_density},
        "mmd_ratio": dict(metrics.mmd_ratio),
        "per_size": per_size,
        "self_loop_occurrences": {
            "aggregation": "sum_over_sample_occurrences",
            "predicted": metrics.self_loops_pred,
            "reference": metrics.self_loops_ref,
        },
    }
    return metrics, report


def select_fixed_threshold(
    *,
    pairs: Sequence[tuple[str, str]],
    logits: NDArray[np.float64],
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> FixedThresholdSelection:
    """Select one deployable threshold using GS, RD calibration, then MMD.

    Candidate thresholds are every atomic validation-logit tie-group boundary
    plus the empty-graph boundary immediately above the maximum logit. Stage 1
    keeps candidates whose paired GS loss from the best-GS threshold is within
    one size-stratified bootstrap SE. Stage 2 applies the same rule to mean
    ``|RD-1|``. Stage 3 lexicographically minimizes degree, clustering, then
    spectral MMD ratio without aggregating them. Exact ties prefer the larger
    logit threshold.
    """
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=g_ref, buckets=buckets)
    thresholds = _candidate_thresholds(samples)
    mean_gs = _mean_metric_curve(samples, thresholds, metric="gs")
    mean_rd_error = _mean_metric_curve(samples, thresholds, metric="rd_error")

    best_gs_index = int(np.argmax(mean_gs))
    gs_se = _stratified_paired_se_curve(
        samples,
        thresholds,
        float(thresholds[best_gs_index]),
        metric="gs",
        candidate_minus_reference=False,
    )
    near_gs = mean_gs[best_gs_index] - mean_gs <= gs_se + 1e-12

    gs_indices = np.flatnonzero(near_gs)
    best_rd_index = int(gs_indices[np.argmin(mean_rd_error[gs_indices])])
    rd_se = _stratified_paired_se_curve(
        samples,
        thresholds,
        float(thresholds[best_rd_index]),
        metric="rd_error",
        candidate_minus_reference=True,
    )
    near_rd = near_gs & (mean_rd_error - mean_rd_error[best_rd_index] <= rd_se + 1e-12)

    mmd_indices = np.flatnonzero(near_rd)
    if mmd_indices.size == 0:
        raise ValueError("fixed-threshold selection produced an empty near-optimal RD set")
    reference_by_stat_size, reference_mmd2 = _precompute_reference_descriptors(samples, config)
    surviving_indices = mmd_indices
    stage_counts: dict[str, int] = {}
    selected_objectives: dict[str, float] = {}
    for statistic in STATISTICS:
        objectives = np.array(
            [
                _mmd_ratio_for_statistic(
                    samples,
                    float(thresholds[index]),
                    statistic,
                    reference_by_stat_size,
                    reference_mmd2,
                    config,
                )
                for index in surviving_indices
            ],
            dtype=np.float64,
        )
        if not np.isfinite(objectives).all():
            raise ValueError("validation topology MMD ratios must be finite")
        best_objective = float(np.min(objectives))
        surviving_indices = surviving_indices[objectives == best_objective]
        stage_counts[statistic] = int(surviving_indices.size)
        selected_objectives[statistic] = best_objective
    best_index = int(surviving_indices[0])
    threshold = float(thresholds[best_index])
    predictions = _fixed_predictions(samples, threshold)
    selected_metrics = evaluate_sampled_subgraphs(predictions, g_ref, buckets, config)
    score_by_pair = {
        pair: float(logit)
        for sample in samples
        for pair, logit in zip(sample.pairs, sample.logits, strict=True)
    }
    selected_tie_count = sum(logit == threshold for logit in score_by_pair.values())
    bucket_identity = "\n".join(
        f"{size}:{index}:{','.join(sorted(nodes))}"
        for size, node_sets in buckets.items()
        for index, nodes in enumerate(node_sets)
    )
    report = {
        "rule": "sampled_subgraph_gs_rd_mmd_1se_v2",
        "threshold_candidates": "every_unique_validation_sample_union_logit_plus_empty",
        "candidate_threshold_sha256": hashlib.sha256(thresholds.tobytes()).hexdigest(),
        "candidate_count": int(thresholds.size),
        "sample_count": len(samples),
        "sample_count_by_size": {
            str(size): sum(sample.size == size for sample in samples)
            for size in sorted({sample.size for sample in samples})
        },
        "sampled_bucket_sha256": hashlib.sha256(bucket_identity.encode()).hexdigest(),
        "standard_error": "exact_paired_nonparametric_bootstrap_se_within_size_strata",
        "resampling_unit": "sampled_ball_paired_within_size; overlapping_balls_not_independent",
        "gs_stage": {
            "objective": "maximize_bfs_macro_graph_similarity",
            "best_logit_threshold": float(thresholds[best_gs_index]),
            "best_mean": float(mean_gs[best_gs_index]),
            "near_optimal_count": int(np.count_nonzero(near_gs)),
            "criterion": "best_mean_minus_candidate_mean_lte_paired_1se",
        },
        "rd_stage": {
            "objective": "minimize_mean_abs_relative_density_minus_1",
            "best_logit_threshold": float(thresholds[best_rd_index]),
            "best_mean_abs_error": float(mean_rd_error[best_rd_index]),
            "near_optimal_count": int(np.count_nonzero(near_rd)),
            "criterion": "candidate_mean_minus_best_mean_lte_paired_1se",
        },
        "mmd_stage": {
            "objective": "lexicographic_minimize_degree_then_clustering_then_spectral_mmd_ratio",
            "candidate_count": int(mmd_indices.size),
            "surviving_count_by_statistic": stage_counts,
            "selected_mmd_ratio_by_statistic": selected_objectives,
            "config": {
                "sigma": config.sigma,
                "reference_epsilon": config.reference_epsilon,
            },
        },
        "tie_break": "higher_logit_threshold",
        "selected": {
            "logit_threshold": threshold,
            "probability_threshold": float(expit(threshold)),
            "union_pair_count": len(score_by_pair),
            "union_admitted_pair_count": sum(
                logit >= threshold for logit in score_by_pair.values()
            ),
            "sample_occurrence_admitted_pair_count": sum(
                int(np.count_nonzero(sample.logits >= threshold)) for sample in samples
            ),
            "selected_boundary_tie_group_pair_count": selected_tie_count,
            "validation_graph_similarity": selected_metrics.graph_similarity,
            "validation_relative_density": selected_metrics.relative_density,
            "validation_mmd_ratio": dict(selected_metrics.mmd_ratio),
        },
    }
    return FixedThresholdSelection(
        logit_threshold=threshold,
        metrics=selected_metrics,
        report=report,
    )
