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
from src.eval.checkpoint_selection import SELECTION_RULE
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


@dataclass
class _MMDSizeState:
    predicted: list[NDArray[np.float64]]
    reference: list[NDArray[np.float64]]
    predicted_kernel: NDArray[np.float64]
    cross_kernel: NDArray[np.float64]
    predicted_sum: float
    reference_sum: float
    cross_sum: float


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


def _macro_log_rd_gs_curves(
    samples: tuple[_LocalSample, ...], thresholds: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equal-size mean log RD and GS; empty predictions have log RD = -inf."""
    sizes = sorted({sample.size for sample in samples})
    rd = np.zeros(thresholds.size, dtype=np.float64)
    gs = np.zeros_like(rd)
    for size in sizes:
        stratum = [sample for sample in samples if sample.size == size]
        weight = 1.0 / (len(sizes) * len(stratum))
        for sample in stratum:
            sorted_logits = np.sort(sample.logits)
            true_logits = np.sort(sample.logits[sample.truth])
            predicted = sorted_logits.size - np.searchsorted(sorted_logits, thresholds, side="left")
            tp = true_logits.size - np.searchsorted(true_logits, thresholds, side="left")
            with np.errstate(divide="ignore"):
                rd += weight * np.log(predicted / true_logits.size)
            gs += weight * 2.0 * tp / (predicted + true_logits.size)
    return rd, gs


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


def _normalized_descriptor(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match ``graph_metrics.mmd_squared`` histogram normalization exactly."""
    return values / (float(np.sum(values)) + 1e-6)


def _descriptor_kernel(
    left: NDArray[np.float64], right: NDArray[np.float64], sigma: float
) -> float:
    """Match the canonical Gaussian total-variation MMD kernel exactly."""
    support_size = max(len(left), len(right))
    left_values = np.pad(left, (0, support_size - len(left)))
    right_values = np.pad(right, (0, support_size - len(right)))
    distance = float(np.abs(left_values - right_values).sum() / 2.0)
    return float(np.exp(-(distance * distance) / (2.0 * sigma * sigma)))


def _descriptor_kernel_row(
    left: NDArray[np.float64],
    right: Sequence[NDArray[np.float64]],
    sigma: float,
) -> NDArray[np.float64]:
    """Vectorize one canonical Gaussian-TV kernel row over descriptor samples."""
    support_size = max(len(left), *(len(values) for values in right))
    right_values = np.zeros((len(right), support_size), dtype=np.float64)
    for index, values in enumerate(right):
        right_values[index, : len(values)] = values
    left_values = np.pad(left, (0, support_size - len(left)))
    distances = np.abs(right_values - left_values).sum(axis=1) / 2.0
    kernel_values: NDArray[np.float64] = np.exp(-(distances * distances) / (2.0 * sigma * sigma))
    return kernel_values


def _initialize_mmd_state(
    predicted: list[NDArray[np.float64]],
    reference: list[NDArray[np.float64]],
    config: MMDConfig,
) -> _MMDSizeState:
    predicted_kernel = np.array(
        [[_descriptor_kernel(x, y, config.sigma) for y in predicted] for x in predicted],
        dtype=np.float64,
    )
    cross_kernel = np.array(
        [[_descriptor_kernel(x, y, config.sigma) for y in reference] for x in predicted],
        dtype=np.float64,
    )
    reference_sum = sum(
        _descriptor_kernel(x, y, config.sigma) for x in reference for y in reference
    )
    return _MMDSizeState(
        predicted=predicted,
        reference=reference,
        predicted_kernel=predicted_kernel,
        cross_kernel=cross_kernel,
        predicted_sum=sum(float(value) for row in predicted_kernel for value in row),
        reference_sum=float(reference_sum),
        cross_sum=sum(float(value) for row in cross_kernel for value in row),
    )


def _update_mmd_state(
    state: _MMDSizeState,
    updates: dict[int, NDArray[np.float64]],
    config: MMDConfig,
) -> None:
    """Update biased MMD kernel sums after descriptors change in one size bucket."""
    for index in sorted(updates):
        descriptor = updates[index]
        state.predicted[index] = descriptor
        predicted_count = len(state.predicted)
        values = _descriptor_kernel_row(
            descriptor, [*state.predicted, *state.reference], config.sigma
        )
        predicted_values = values[:predicted_count]
        reference_values = values[predicted_count:]
        for other_index, value in enumerate(predicted_values):
            if other_index == index:
                continue
            previous = float(state.predicted_kernel[index, other_index])
            state.predicted_sum += 2.0 * (value - previous)
            state.predicted_kernel[index, other_index] = value
            state.predicted_kernel[other_index, index] = value
        for reference_index, value in enumerate(reference_values):
            previous = float(state.cross_kernel[index, reference_index])
            state.cross_sum += value - previous
            state.cross_kernel[index, reference_index] = value


def _mmd_ratio_from_states(
    states: dict[int, _MMDSizeState], denominator: float, config: MMDConfig
) -> float:
    raw_by_size = []
    for state in states.values():
        predicted_count = len(state.predicted)
        reference_count = len(state.reference)
        raw_by_size.append(
            state.predicted_sum / (predicted_count * predicted_count)
            + state.reference_sum / (reference_count * reference_count)
            - 2.0 * state.cross_sum / (predicted_count * reference_count)
        )
    return float(np.mean(np.maximum(raw_by_size, 0.0))) / max(denominator, config.reference_epsilon)


def _incremental_mmd_ratio_curve(
    samples: tuple[_LocalSample, ...],
    thresholds: NDArray[np.float64],
    feasible_indices: NDArray[np.intp],
    reference_by_stat_size: dict[str, dict[int, list[NDArray[np.float64]]]],
    reference_mmd2: dict[str, float],
    config: MMDConfig,
) -> NDArray[np.float64]:
    """Return exact MMD ratios while admitting each threshold tie group once."""
    sizes = list(dict.fromkeys(sample.size for sample in samples))
    sample_indices_by_size: dict[int, dict[int, int]] = {size: {} for size in sizes}
    graphs: list[nx.Graph] = []
    event_order: list[NDArray[np.intp]] = []
    for sample_index, sample in enumerate(samples):
        sample_indices_by_size[sample.size][sample_index] = len(sample_indices_by_size[sample.size])
        graph = nx.Graph()
        graph.add_nodes_from(sample.nodes)
        graphs.append(graph)
        event_order.append(np.argsort(sample.logits, kind="stable")[::-1])

    event_offsets = np.zeros(len(samples), dtype=np.intp)

    def admit_until(threshold: float) -> set[int]:
        touched: set[int] = set()
        for sample_index, sample in enumerate(samples):
            order = event_order[sample_index]
            offset = int(event_offsets[sample_index])
            while offset < order.size and sample.logits[order[offset]] >= threshold:
                graphs[sample_index].add_edge(*sample.pairs[int(order[offset])])
                offset += 1
                touched.add(sample_index)
            event_offsets[sample_index] = offset
        return touched

    first_feasible = int(feasible_indices[0])
    admit_until(float(thresholds[first_feasible]))

    states_by_stat: dict[str, dict[int, _MMDSizeState]] = {
        statistic: {} for statistic in STATISTICS
    }
    for statistic in STATISTICS:
        for size in sizes:
            sample_indices = sample_indices_by_size[size]
            predicted = [
                _normalized_descriptor(_descriptor(graphs[index], statistic))
                for index in sample_indices
            ]
            reference = [
                _normalized_descriptor(values) for values in reference_by_stat_size[statistic][size]
            ]
            states_by_stat[statistic][size] = _initialize_mmd_state(predicted, reference, config)

    ratios = np.empty((feasible_indices.size, len(STATISTICS)), dtype=np.float64)
    for statistic_index, statistic in enumerate(STATISTICS):
        ratios[0, statistic_index] = _mmd_ratio_from_states(
            states_by_stat[statistic], reference_mmd2[statistic], config
        )
    for position, feasible_index_value in enumerate(feasible_indices[1:], start=1):
        feasible_index = int(feasible_index_value)
        touched = admit_until(float(thresholds[feasible_index]))
        for statistic_index, statistic in enumerate(STATISTICS):
            updates_by_size: dict[int, dict[int, NDArray[np.float64]]] = {}
            for sample_index in sorted(touched):
                size = samples[sample_index].size
                local_index = sample_indices_by_size[size][sample_index]
                updates_by_size.setdefault(size, {})[local_index] = _normalized_descriptor(
                    _descriptor(graphs[sample_index], statistic)
                )
            for size, updates in updates_by_size.items():
                _update_mmd_state(states_by_stat[statistic][size], updates, config)
            ratios[position, statistic_index] = _mmd_ratio_from_states(
                states_by_stat[statistic], reference_mmd2[statistic], config
            )
    return ratios


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


def density_diagnostics(per_size_rd: dict[int, list[float]]) -> dict[str, object]:
    """Report both RD centers and log distortion without masking empty graphs.

    A zero RD gives geometric RD zero and infinite log distortion, encoded as
    null plus an explicit zero count so reports remain standard JSON.
    """
    arrays = [np.asarray(values, dtype=np.float64) for values in per_size_rd.values()]
    zeros = sum(int(np.count_nonzero(values == 0)) for values in arrays)
    with np.errstate(divide="ignore"):
        logs = [np.log(values) for values in arrays]
    return {
        "arithmetic_mean": float(np.mean([values.mean() for values in arrays])),
        "geometric_mean": 0.0
        if zeros
        else float(np.exp(np.mean([values.mean() for values in logs]))),
        "mean_abs_log": None
        if zeros
        else float(np.mean([np.abs(values).mean() for values in logs])),
        "zero_rd_samples": zeros,
    }


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
            "density_diagnostics": density_diagnostics(
                {size: metrics.per_size_relative_density[size]}
            ),
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
        "density_diagnostics": density_diagnostics(metrics.per_size_relative_density),
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
    """Minimize absolute macro mean log RD, then maximize GS and minimize geo-MMD.

    Each size and each subgraph within a size have equal weight. A threshold
    emptying any positive-reference subgraph has infinite density error. The
    all-pairs threshold is finite, so no density band or eligibility gate is needed.
    """
    samples = _local_samples(pairs=pairs, logits=logits, g_ref=g_ref, buckets=buckets)
    thresholds = _candidate_thresholds(samples)
    mean_log_rd, macro_gs = _macro_log_rd_gs_curves(samples, thresholds)
    rd_error = np.abs(mean_log_rd)
    candidates = np.flatnonzero(rd_error == rd_error.min())
    # Shape cannot compensate for a GS loss: evaluate it only at exact GS ties.
    gs_indices = candidates[macro_gs[candidates] == macro_gs[candidates].max()]
    reference_by_stat_size, reference_mmd2 = _precompute_reference_descriptors(samples, config)
    ratios = _incremental_mmd_ratio_curve(
        samples,
        thresholds,
        gs_indices,
        reference_by_stat_size,
        reference_mmd2,
        config,
    )
    if not np.isfinite(ratios).all() or np.any(ratios < 0):
        raise ValueError("validation topology MMD ratios must be finite and non-negative")
    geo_mmd = np.prod(ratios, axis=1) ** (1.0 / 3.0)
    best_shape_local = min(
        range(len(gs_indices)),
        key=lambda i: (
            float(geo_mmd[i]),
            -float(thresholds[gs_indices[i]]),
        ),
    )
    best_index = int(gs_indices[best_shape_local])
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
        "rule": SELECTION_RULE,
        "density_diagnostics": density_diagnostics(selected_metrics.per_size_relative_density),
        "threshold_candidates": "every_unique_validation_sample_union_logit_plus_empty",
        "candidate_threshold_sha256": hashlib.sha256(thresholds.tobytes()).hexdigest(),
        "candidate_count": int(thresholds.size),
        "sample_count": len(samples),
        "sample_count_by_size": {
            str(size): sum(sample.size == size for sample in samples)
            for size in sorted({sample.size for sample in samples})
        },
        "sampled_bucket_sha256": hashlib.sha256(bucket_identity.encode()).hexdigest(),
        "density_stage": {
            "objective": "minimize_abs_macro_mean_log_relative_density",
            "finite_candidate_count": int(np.isfinite(rd_error).sum()),
            "density_tie_count": int(candidates.size),
            "selected_abs_log_rd": float(rd_error[best_index]),
        },
        "gs_stage": {
            "objective": "maximize_macro_graph_similarity",
            "tie_count": int(gs_indices.size),
        },
        "shape_stage": {
            "objective": "minimize_geo_mmd_at_exact_gs_ties",
            "selected_geo_mmd": float(geo_mmd[best_shape_local]),
        },
        "tie_break": "higher_gs_then_lower_geo_mmd_then_higher_logit_threshold",
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
