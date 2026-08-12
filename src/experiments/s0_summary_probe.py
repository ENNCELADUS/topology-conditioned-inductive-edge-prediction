r"""S0 — summary-value diagnostic for the latent-topology route.

Evidence class: **diagnostic**. Measures, without any training run, both halves
of the route's value chain plus the achievable middle:

- **Arm A (predictability, ``V_fit``):** nested-CV ridge probes from frozen F0
  features to per-node structural summaries (``probe_targets``), raw and
  degree-partialled, with a shuffled-target control. The spec-pinned probe
  lambda (``1e-3``) is under-regularized at ``d = 1536``, so lambda is selected
  per outer fold on an inner fold.
- **Arm B (value, complete ``V_hold`` pair universe):** out-of-fold ridge heads
  over symmetrized pair features comparing feature-only scoring against oracle
  per-node ego summaries (constant per node, computed on the true ``g_hold``),
  feature-predicted summaries (the protocol-legal achievable row), shuffled
  summaries, and an oracle CN/AA relational ceiling. Reads ``V_hold`` structure
  by design (like ``g3_oracle``), so its numbers never feed a formal claim
  directly. Partner-*excluded* summaries are deliberately **not** used as
  conditioning input: their per-pair leave-one-out variation encodes the label
  itself once a head can memorize per-node baselines across node-sharing folds
  (measured: AUPRC 1.0). Partner exclusion remains the training-target
  convention only.

CLI::

    python -m src.experiments.s0_summary_probe \
        --data-root data --strategy breadth_first \
        --f0-cache outputs/f0_cache/all_operative_nodes.pt \
        --feature-root data/features/frozen_node_features_1024 \
        --output-dir outputs/s0

The pure pieces below stay importable for tests; :func:`main` wires the real
benchmark artifacts into them.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.features import FeatureStore, build_f0_matrix
from src.data.internal_holdout import InternalHoldoutPartition, derive_internal_holdout
from src.data.partition import derive_training_interactions
from src.experiments.g1_hardened_e2 import common_neighbor_and_adamic_adar
from src.experiments.probes import (
    _MIN_VARIANCE,
    _as_2d,
    _kfold_indices,
    _load_train_side_probe_inputs,
    _partial_out_train_test,
    _ridge_fit_predict,
    g_struct_sha256,
    probe_targets,
)

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_LAMBDA_GRID = (1e-3, 1e-1, 1e1, 1e3, 1e5, 1e6, 1e7)
_N_FOLDS = 5
_DEFAULT_CHUNK = 8192
_DEFAULT_N_BOOT = 1000
_TARGET_NAMES = ("degree", "clustering", "ego_edges", "ego_density")
#: Matches ``configs/b0_v31_breadth_first.yaml`` `expected_missing_features`.
_DEFAULT_MISSING = ("node_004764", "node_007050")

PairFeatureFn = Callable[[NDArray[np.int64]], NDArray[np.float64]]


@dataclass(frozen=True)
class NestedProbeResult:
    """Out-of-fold probe ``R^2`` with the per-outer-fold selected lambdas."""

    r2: float
    fold_lambdas: tuple[float, ...]


def _split_r2(predictions: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
    """Plain ``R^2`` of `predictions` against `targets` (0.0 on ~zero variance)."""
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    if ss_tot < _MIN_VARIANCE:
        return 0.0
    return 1.0 - float(np.sum((targets - predictions) ** 2)) / ss_tot


def nested_probe_r2(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    lambdas: Sequence[float] = _LAMBDA_GRID,
    degrees: NDArray[np.float64] | None = None,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
) -> NestedProbeResult:
    """Nested-CV ridge probe ``R^2`` with per-outer-fold lambda selection.

    Each outer fold's lambda is chosen on one inner validation fold drawn from
    the remaining folds, then the head is refit on all non-test rows before
    predicting the outer test fold. With `degrees` given, states and targets
    are first residualized on degree with train-fold-only fits (the
    ``degree_partialled_r2`` convention), so the score is signal beyond a pure
    degree confound.
    """
    states64 = _as_2d(np.asarray(states, dtype=np.float64))
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    degrees64 = None if degrees is None else np.asarray(degrees, dtype=np.float64).reshape(-1)
    n = states64.shape[0]
    if n < n_folds:
        raise ValueError(f"nested_probe_r2 requires at least {n_folds} samples, got {n}")
    folds = _kfold_indices(n, n_folds, seed)
    predictions = np.empty(n, dtype=np.float64)
    pooled_targets = targets64.copy()
    fold_lambdas: list[float] = []

    def fit_predict(
        train_idx: NDArray[np.int64], test_idx: NDArray[np.int64], lam: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fit on `train_idx`, predict `test_idx`; returns (predictions, test targets)."""
        if degrees64 is None:
            return (
                _ridge_fit_predict(
                    states64[train_idx], targets64[train_idx], states64[test_idx], lam
                ),
                targets64[test_idx],
            )
        train_states, test_states = _partial_out_train_test(
            states64[train_idx], states64[test_idx], degrees64[train_idx], degrees64[test_idx]
        )
        train_targets, test_targets = _partial_out_train_test(
            targets64[train_idx], targets64[test_idx], degrees64[train_idx], degrees64[test_idx]
        )
        return _ridge_fit_predict(train_states, train_targets, test_states, lam), test_targets

    for i, test_idx in enumerate(folds):
        inner = (i + 1) % n_folds
        inner_idx = folds[inner]
        select_idx = np.concatenate([folds[j] for j in range(n_folds) if j not in (i, inner)])
        inner_scores: list[tuple[float, float]] = []
        for lam in lambdas:
            inner_pred, inner_targets = fit_predict(select_idx, inner_idx, lam)
            inner_scores.append((_split_r2(inner_pred, inner_targets), float(lam)))
        best_lam = max(inner_scores, key=lambda pair: (pair[0], pair[1]))[1]
        fold_lambdas.append(best_lam)
        full_train = np.concatenate([select_idx, inner_idx])
        predictions[test_idx], pooled_targets[test_idx] = fit_predict(
            full_train, test_idx, best_lam
        )
    return NestedProbeResult(
        r2=_split_r2(predictions, pooled_targets), fold_lambdas=tuple(fold_lambdas)
    )


def symmetric_pair_features(
    node_matrix: NDArray[np.float64],
    u_idx: NDArray[np.int64],
    v_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Root-swap-invariant pair encoding ``[rows_u + rows_v, |rows_u - rows_v|]``."""
    rows_u = node_matrix[u_idx]
    rows_v = node_matrix[v_idx]
    return np.concatenate([rows_u + rows_v, np.abs(rows_u - rows_v)], axis=1)


def partner_excluded_pair_stats(
    graph: nx.Graph, pairs: Sequence[tuple[str, str]]
) -> NDArray[np.float64]:
    """Symmetrized oracle ego summaries with the queried partner excluded.

    Per pair ``(u, v)``: each endpoint's ``(degree, clustering, ego_edges,
    ego_density)`` is computed on `graph` after removing the partner from its
    neighborhood (degree decremented), following the exact
    ``EgoTargetBuilder._node_ego_stats`` leave-one-out convention (pinned by
    test), then symmetrized as ``[s_u + s_v, |s_u - s_v|]``.

    Warning: this is a *training-target* convention. Never feed these rows to a
    pair head whose CV folds share nodes — the per-pair leave-one-out variation
    encodes the queried label itself once per-node baselines are memorized
    (S0 measured AUPRC 1.0 through exactly this path).
    """
    cache: dict[tuple[str, str | None], NDArray[np.float64]] = {}

    def node_stats(node: str, partner: str) -> NDArray[np.float64]:
        excluded = partner if graph.has_edge(node, partner) else None
        key = (node, excluded)
        cached = cache.get(key)
        if cached is not None:
            return cached
        neighbors = [n for n in graph.neighbors(node) if n != excluded]
        ego = graph.subgraph([node, *neighbors])
        degree = len(neighbors)
        neighbor_edges = ego.number_of_edges() - degree
        clustering = 2 * neighbor_edges / (degree * (degree - 1)) if degree > 1 else 0.0
        stats = np.array(
            [float(degree), clustering, float(ego.number_of_edges()), float(nx.density(ego))],
            dtype=np.float64,
        )
        cache[key] = stats
        return stats

    rows = np.empty((len(pairs), 8), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        s_u = node_stats(u, v)
        s_v = node_stats(v, u)
        rows[i, :4] = s_u + s_v
        rows[i, 4:] = np.abs(s_u - s_v)
    return rows


def _chunks(indices: NDArray[np.int64], chunk: int) -> list[NDArray[np.int64]]:
    """Split `indices` into contiguous chunks of at most `chunk` rows."""
    n_chunks = max(1, -(-len(indices) // chunk))
    return [c.astype(np.int64) for c in np.array_split(indices, n_chunks)]


def chunked_ridge_fit(
    feature_fn: PairFeatureFn,
    n_rows: int,
    targets: NDArray[np.float64],
    *,
    lam: float,
    chunk: int = _DEFAULT_CHUNK,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Centered closed-form ridge over chunked features: ``(weights, x_mean, y_mean)``.

    Accumulates the gram and cross moments in float64 without materializing the
    design matrix, matching ``_ridge_fit_predict``'s centered solution.
    """
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    if targets64.shape[0] != n_rows:
        raise ValueError("targets must have one row per feature row")
    all_idx = np.arange(n_rows, dtype=np.int64)
    d = int(np.asarray(feature_fn(all_idx[:1]), dtype=np.float64).shape[1])
    gram = np.zeros((d, d), dtype=np.float64)
    moment = np.zeros(d, dtype=np.float64)
    feature_sum = np.zeros(d, dtype=np.float64)
    target_sum = 0.0
    for chunk_idx in _chunks(all_idx, chunk):
        block = np.asarray(feature_fn(chunk_idx), dtype=np.float64)
        y = targets64[chunk_idx]
        gram += block.T @ block
        moment += block.T @ y
        feature_sum += block.sum(axis=0)
        target_sum += float(y.sum())
    gram -= np.outer(feature_sum, feature_sum) / n_rows
    moment -= feature_sum * (target_sum / n_rows)
    weights = np.asarray(np.linalg.solve(gram + lam * np.eye(d), moment), dtype=np.float64)
    return weights, feature_sum / n_rows, target_sum / n_rows


def chunked_ridge_predict(
    feature_fn: PairFeatureFn,
    indices: NDArray[np.int64],
    weights: NDArray[np.float64],
    feature_mean: NDArray[np.float64],
    target_mean: float,
    *,
    chunk: int = _DEFAULT_CHUNK,
) -> NDArray[np.float64]:
    """Predict `indices` rows from a :func:`chunked_ridge_fit` solution, chunked."""
    parts = [
        (np.asarray(feature_fn(chunk_idx), dtype=np.float64) - feature_mean) @ weights + target_mean
        for chunk_idx in _chunks(indices, chunk)
    ]
    return np.concatenate(parts)


def cv_pair_scores(
    feature_fn: PairFeatureFn,
    n_pairs: int,
    labels: NDArray[np.int64] | NDArray[np.float64],
    *,
    lambdas: Sequence[float] = _LAMBDA_GRID,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
    chunk: int = _DEFAULT_CHUNK,
    regression: bool = False,
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    """Out-of-fold ridge scores over a pair universe too large to materialize.

    Accumulates per-fold second moments in float64 over chunks produced by
    `feature_fn`, so any train-fold-subset ridge solves from cached moments
    without holding the design matrix. Lambda is selected per outer fold on one
    inner fold by average precision (or mean-squared error with
    ``regression=True``, and as the fallback when the inner fold is
    single-class), then the head is refit on all non-test rows.
    """
    labels64 = np.asarray(labels, dtype=np.float64).reshape(-1)
    if labels64.shape[0] != n_pairs:
        raise ValueError("labels must have one row per pair")
    folds = _kfold_indices(n_pairs, n_folds, seed)
    d = int(np.asarray(feature_fn(folds[0][:1]), dtype=np.float64).shape[1])
    grams = np.zeros((n_folds, d, d), dtype=np.float64)
    moments = np.zeros((n_folds, d), dtype=np.float64)
    feature_sums = np.zeros((n_folds, d), dtype=np.float64)
    label_sums = np.zeros(n_folds, dtype=np.float64)
    counts = np.zeros(n_folds, dtype=np.float64)
    for f, idx in enumerate(folds):
        for chunk_idx in _chunks(idx, chunk):
            block = np.asarray(feature_fn(chunk_idx), dtype=np.float64)
            y = labels64[chunk_idx]
            grams[f] += block.T @ block
            moments[f] += block.T @ y
            feature_sums[f] += block.sum(axis=0)
            label_sums[f] += float(y.sum())
        counts[f] = len(idx)

    def solve(
        fold_subset: Sequence[int], lam: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return centered ridge ``(weights, feature_mean, label_mean)`` for the subset."""
        subset = list(fold_subset)
        n = float(counts[subset].sum())
        sx = feature_sums[subset].sum(axis=0)
        sy = float(label_sums[subset].sum())
        gram = grams[subset].sum(axis=0) - np.outer(sx, sx) / n
        moment = moments[subset].sum(axis=0) - sx * (sy / n)
        weights = np.linalg.solve(gram + lam * np.eye(d), moment)
        return weights, sx / n, sy / n

    scores = np.empty(n_pairs, dtype=np.float64)
    fold_lambdas: list[float] = []
    for i in range(n_folds):
        inner = (i + 1) % n_folds
        select_subset = [j for j in range(n_folds) if j not in (i, inner)]
        inner_idx = folds[inner]
        inner_labels = labels64[inner_idx]
        use_mse = regression or inner_labels.min() == inner_labels.max()
        inner_scores: list[tuple[float, float]] = []
        for lam in lambdas:
            weights, feature_mean, label_mean = solve(select_subset, lam)
            inner_pred = chunked_ridge_predict(
                feature_fn, inner_idx, weights, feature_mean, label_mean, chunk=chunk
            )
            if use_mse:
                quality = -float(np.mean((inner_pred - inner_labels) ** 2))
            else:
                quality = float(average_precision_score(inner_labels, inner_pred))
            inner_scores.append((quality, float(lam)))
        best_lam = max(inner_scores, key=lambda pair: (pair[0], pair[1]))[1]
        fold_lambdas.append(best_lam)
        weights, feature_mean, label_mean = solve([j for j in range(n_folds) if j != i], best_lam)
        scores[folds[i]] = chunked_ridge_predict(
            feature_fn, folds[i], weights, feature_mean, label_mean, chunk=chunk
        )
    return scores, tuple(fold_lambdas)


def paired_bootstrap_delta(
    labels: NDArray[np.int64],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    metric: str = "auprc",
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap ``(point, lo, hi)`` for ``metric(a) - metric(b)``.

    The same resampled index vector is applied to both score arms. Single-class
    resamples are skipped (they leave the metric undefined at ~1% prevalence).
    """
    if metric not in ("auprc", "auroc"):
        raise ValueError(f"unsupported paired bootstrap metric {metric!r}")
    metric_fn = average_precision_score if metric == "auprc" else roc_auc_score
    labels64 = np.asarray(labels).reshape(-1)
    point = float(metric_fn(labels64, scores_a)) - float(metric_fn(labels64, scores_b))
    rng = np.random.default_rng(seed)
    n = labels64.shape[0]
    deltas: list[float] = []
    while len(deltas) < n_boot:
        idx = rng.integers(0, n, size=n)
        resampled = labels64[idx]
        if resampled.min() == resampled.max():
            continue
        deltas.append(
            float(metric_fn(resampled, scores_a[idx])) - float(metric_fn(resampled, scores_b[idx]))
        )
    lo, hi = np.quantile(np.asarray(deltas), [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def to_json_payload(
    *,
    strategy: str,
    arm_a: Mapping[str, object],
    arm_b: Mapping[str, object],
    deltas: Mapping[str, object],
    manifest_digests: Mapping[str, str],
) -> dict[str, object]:
    """Assemble the diagnostic-labeled JSON payload written by :func:`main`."""
    return {
        "format": "s0_summary_probe_v1",
        "evidence_class": "diagnostic",
        "strategy": strategy,
        "arm_a": dict(arm_a),
        "arm_b": dict(arm_b),
        "deltas": dict(deltas),
        "manifest_digests": dict(manifest_digests),
    }


def _modal_lambda(fold_lambdas: Sequence[float]) -> float:
    """Most frequent selected lambda; ties break toward the larger penalty."""
    values = list(fold_lambdas)
    return max(set(values), key=lambda lam: (values.count(lam), lam))


def _load_holdout(
    data_root: Path, strategy: str, expected_missing: Sequence[str]
) -> InternalHoldoutPartition:
    """Derive the internal holdout from the train-side benchmark artifacts."""
    formal_nodes, positives = _load_train_side_probe_inputs(
        data_root / _BENCHMARK_SUBDIR / strategy,
        expected_missing_features=expected_missing,
    )
    interactions = derive_training_interactions(positives)
    return derive_internal_holdout(formal_nodes, interactions.positives)


def _load_features(
    feature_root: Path, cache_path: Path, node_ids: Sequence[str]
) -> NDArray[np.float64]:
    """Gather F0 rows for `node_ids` from the cache, then re-assert the order.

    ``allow_cache_subset=True`` gathers a superset cache with no content check
    (the silent-corruption trap in ``CLAUDE.md``), so coverage is verified
    before the call and the returned index is re-asserted after it.
    """
    cached = cast(dict[str, object], torch.load(cache_path, map_location="cpu", weights_only=True))
    cached_ids = set(cast(list[str], cached["node_ids"]))
    missing = [node for node in node_ids if node not in cached_ids]
    if missing:
        raise ValueError(f"F0 cache is missing requested nodes: {missing[:5]}")
    matrix, index = build_f0_matrix(
        FeatureStore(feature_root),
        list(node_ids),
        cache_path=cache_path,
        allow_cache_subset=True,
    )
    if list(index) != list(node_ids) or matrix.shape[0] != len(node_ids):
        raise AssertionError("F0 gather returned rows out of requested order")
    return matrix.double().numpy()


def build_parser() -> argparse.ArgumentParser:
    """Build the S0 summary-probe CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s0_summary_probe",
        description=(
            "S0 summary-value diagnostic: structural-summary predictability (Arm A) "
            "and oracle/predicted summary value for edge decisions (Arm B)."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--f0-cache", type=Path, required=True)
    parser.add_argument(
        "--feature-root", type=Path, default=Path("data/features/frozen_node_features_1024")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument("--chunk", type=int, default=_DEFAULT_CHUNK)
    parser.add_argument("--expected-missing-features", nargs="*", default=list(_DEFAULT_MISSING))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: run both arms and write ``s0_results.json``."""
    args = build_parser().parse_args(argv)
    holdout = _load_holdout(args.data_root, args.strategy, args.expected_missing_features)
    v_fit_nodes = sorted(holdout.v_fit)
    hold_nodes = list(holdout.hold_manifest.nodes)
    g_fit = holdout.build_g_fit()
    g_hold = holdout.build_g_hold()
    features = _load_features(args.feature_root, args.f0_cache, [*v_fit_nodes, *hold_nodes])
    x_fit = features[: len(v_fit_nodes)]
    x_hold = features[len(v_fit_nodes) :]

    # Arm A — predictability on V_fit only: the summary predictor is later
    # applied to V_hold, so its supervision must never see V_hold structure.
    logger.info("Arm A: probing %d V_fit nodes", len(v_fit_nodes))
    fit_targets = probe_targets(g_fit, v_fit_nodes)
    degrees = fit_targets["degree"]
    shuffle_rng = np.random.default_rng(args.seed)
    arm_a: dict[str, object] = {}
    predicted_node_stats = np.empty((len(hold_nodes), len(_TARGET_NAMES)), dtype=np.float64)
    for column, name in enumerate(_TARGET_NAMES):
        raw = nested_probe_r2(x_fit, fit_targets[name], seed=args.seed)
        partialled = (
            None
            if name == "degree"
            else nested_probe_r2(x_fit, fit_targets[name], degrees=degrees, seed=args.seed)
        )
        shuffled = nested_probe_r2(
            x_fit, shuffle_rng.permutation(fit_targets[name]), seed=args.seed
        )
        lam = _modal_lambda(raw.fold_lambdas)
        predicted_node_stats[:, column] = _ridge_fit_predict(x_fit, fit_targets[name], x_hold, lam)
        arm_a[name] = {
            "r2_raw": raw.r2,
            "r2_degree_partialled": None if partialled is None else partialled.r2,
            "r2_shuffled_control": shuffled.r2,
            "fold_lambdas": list(raw.fold_lambdas),
            "predictor_lambda": lam,
        }
        logger.info("Arm A %s: %s", name, arm_a[name])

    # Arm B — value on the complete V_hold pair universe.
    pairs = holdout.hold_manifest.pairs
    labels = np.asarray(holdout.hold_manifest.labels, dtype=np.int64)
    hold_index = {node: i for i, node in enumerate(hold_nodes)}
    u_idx = np.asarray([hold_index[u] for u, _ in pairs], dtype=np.int64)
    v_idx = np.asarray([hold_index[v] for _, v in pairs], dtype=np.int64)
    logger.info("Arm B: %d pairs, %d positives", len(pairs), int(labels.sum()))
    # Plain per-node oracle summaries (constant per node): partner-excluded
    # variants vary with the queried pair and leak the label to a head that can
    # memorize per-node baselines across node-sharing folds (see module docstring).
    hold_targets = probe_targets(g_hold, hold_nodes)
    oracle_node_stats = np.column_stack([hold_targets[name] for name in _TARGET_NAMES])
    oracle_pair = symmetric_pair_features(oracle_node_stats, u_idx, v_idx)
    predicted_pair = symmetric_pair_features(predicted_node_stats, u_idx, v_idx)
    shuffled_node_stats = oracle_node_stats[
        np.random.default_rng(args.seed).permutation(len(hold_nodes))
    ]
    shuffled_pair = symmetric_pair_features(shuffled_node_stats, u_idx, v_idx)
    cn_matrix, aa_matrix = common_neighbor_and_adamic_adar(g_hold, hold_nodes)
    relational_pair = np.column_stack([cn_matrix[u_idx, v_idx], aa_matrix[u_idx, v_idx]])

    def features_only(idx: NDArray[np.int64]) -> NDArray[np.float64]:
        return symmetric_pair_features(x_hold, u_idx[idx], v_idx[idx])

    def with_block(block: NDArray[np.float64]) -> PairFeatureFn:
        return lambda idx: np.concatenate([features_only(idx), block[idx]], axis=1)

    rows: dict[str, PairFeatureFn] = {
        "row1_features_only": features_only,
        "row2_oracle_summaries": with_block(oracle_pair),
        "row3_predicted_summaries": with_block(predicted_pair),
        "row4_shuffled_summaries": with_block(shuffled_pair),
        "row5_oracle_summaries_plus_cn_aa": with_block(
            np.concatenate([oracle_pair, relational_pair], axis=1)
        ),
    }
    row_scores: dict[str, NDArray[np.float64]] = {}
    arm_b: dict[str, object] = {}
    for name, feature_fn in rows.items():
        scores, fold_lambdas = cv_pair_scores(
            feature_fn, len(pairs), labels, seed=args.seed, chunk=args.chunk
        )
        row_scores[name] = scores
        arm_b[name] = {
            "auprc": float(average_precision_score(labels, scores)),
            "auroc": float(roc_auc_score(labels, scores)),
            "fold_lambdas": list(fold_lambdas),
        }
        logger.info("Arm B %s: %s", name, arm_b[name])

    deltas: dict[str, object] = {}
    for name, (row_a, row_b) in {
        "row2_minus_row1": ("row2_oracle_summaries", "row1_features_only"),
        "row3_minus_row1": ("row3_predicted_summaries", "row1_features_only"),
        "row4_minus_row1": ("row4_shuffled_summaries", "row1_features_only"),
        "row5_minus_row2": ("row5_oracle_summaries_plus_cn_aa", "row2_oracle_summaries"),
    }.items():
        point, lo, hi = paired_bootstrap_delta(
            labels, row_scores[row_a], row_scores[row_b], n_boot=args.n_boot, seed=args.seed
        )
        deltas[name] = {"delta_auprc": point, "lo": lo, "hi": hi}
        logger.info("delta %s: %s", name, deltas[name])

    def auprc(name: str) -> float:
        return cast(dict[str, float], arm_b[name])["auprc"]

    ceiling_gain = auprc("row2_oracle_summaries") - auprc("row1_features_only")
    achieved_gain = auprc("row3_predicted_summaries") - auprc("row1_features_only")
    deltas["decision"] = {
        "context_valuable": cast(dict[str, float], deltas["row2_minus_row1"])["lo"] > 0,
        "route_signal": cast(dict[str, float], deltas["row3_minus_row1"])["lo"] > 0,
        "relational_headroom": cast(dict[str, float], deltas["row5_minus_row2"])["lo"] > 0,
        "captured_fraction": achieved_gain / ceiling_gain if ceiling_gain > 0 else None,
        "prevalence": holdout.hold_manifest.prevalence,
    }

    payload = to_json_payload(
        strategy=args.strategy,
        arm_a=arm_a,
        arm_b=arm_b,
        deltas=deltas,
        manifest_digests={
            "nodes_sha256": holdout.hold_manifest.nodes_sha256,
            "positive_edges_sha256": holdout.hold_manifest.positive_edges_sha256,
            "pair_labels_sha256": holdout.hold_manifest.pair_labels_sha256,
            "g_fit_sha256": g_struct_sha256(g_fit),
            "g_hold_sha256": g_struct_sha256(g_hold),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "s0_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote S0 results to %s", output_path)


__all__ = [
    "NestedProbeResult",
    "build_parser",
    "chunked_ridge_fit",
    "chunked_ridge_predict",
    "cv_pair_scores",
    "main",
    "nested_probe_r2",
    "paired_bootstrap_delta",
    "partner_excluded_pair_stats",
    "symmetric_pair_features",
    "to_json_payload",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
