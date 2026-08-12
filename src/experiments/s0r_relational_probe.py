r"""S0-R — relational summary-value diagnostic for the latent-topology route.

Evidence class: **diagnostic**. Follow-up to ``s0_summary_probe``: per-node
summaries showed a low conditioning ceiling (+0.024 AUPRC) while pair-relational
CN/AA context carried +0.222. S0-R measures the ``(x_u, x_v)``-reachable share
of that relational value, gating the relational-T route:

- **Arm A-R (predictability, sampled ``V_fit`` pairs):** out-of-fold prediction
  of ``CN > 0`` (AUPRC), ``log1p(CN)`` and Adamic–Adar (``R^2``) from
  symmetrized endpoint features, against oracle/predicted degree-product and
  shuffled controls.
- **Arm B-R (transfer, complete ``V_hold`` pair universe):** pair heads
  comparing features-only, + oracle CN/AA, + feature-predicted CN/AA (the
  protocol-legal achievable row), and + shuffled CN/AA. Reads ``V_hold``
  structure by design (like ``g3_oracle``); numbers never feed a formal claim.

CN/AA never include the queried edge (they count length-2 paths only), so no
partner exclusion applies, and both blocks are constant per pair — none of the
leave-one-out leakage documented in ``s0_summary_probe`` can occur here.

CLI::

    python -m src.experiments.s0r_relational_probe \
        --data-root data --strategy breadth_first \
        --f0-cache outputs/f0_cache/all_operative_nodes.pt \
        --feature-root data/features/frozen_node_features_1024 \
        --output-dir outputs/s0r

The pure pieces below stay importable for tests; :func:`main` wires the real
benchmark artifacts into them.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiments.probes import _ridge_fit_predict, g_struct_sha256
from src.experiments.s0_summary_probe import (
    _DEFAULT_CHUNK,
    _DEFAULT_MISSING,
    _DEFAULT_N_BOOT,
    PairFeatureFn,
    _load_features,
    _load_holdout,
    _modal_lambda,
    _split_r2,
    chunked_ridge_fit,
    chunked_ridge_predict,
    cv_pair_scores,
    nested_probe_r2,
    paired_bootstrap_delta,
    symmetric_pair_features,
)

logger = logging.getLogger(__name__)

_DEFAULT_N_RANDOM = 500_000
_DEFAULT_N_POSITIVE = 5_000


def pair_cn_aa(graph: nx.Graph, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]:
    """Per-pair common-neighbor count and Adamic–Adar score, ``(P, 2)`` float64.

    Computed by neighbor-set intersection (no dense matrix), matching the
    ``common_neighbor_and_adamic_adar`` convention (pinned by test): the
    Adamic–Adar weight of a shared neighbor ``w`` is ``1/log(deg(w))`` for
    ``deg(w) > 1`` and 0 otherwise. The queried edge itself never contributes —
    both quantities count length-2 paths only — so no partner exclusion applies.
    """
    neighbor_sets: dict[str, frozenset[str]] = {}
    weights: dict[str, float] = {}

    def neighbors(node: str) -> frozenset[str]:
        cached = neighbor_sets.get(node)
        if cached is None:
            cached = frozenset(graph.neighbors(node))
            neighbor_sets[node] = cached
        return cached

    def weight(node: str) -> float:
        cached = weights.get(node)
        if cached is None:
            degree = graph.degree(node)
            cached = 1.0 / np.log(degree) if degree > 1 else 0.0
            weights[node] = cached
        return cached

    rows = np.zeros((len(pairs), 2), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        common = neighbors(u) & neighbors(v)
        rows[i, 0] = float(len(common))
        rows[i, 1] = sum(weight(w) for w in common)
    return rows


def sample_fit_pairs(
    graph: nx.Graph,
    nodes: Sequence[str],
    *,
    n_random: int = _DEFAULT_N_RANDOM,
    n_positive: int = _DEFAULT_N_POSITIVE,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Seeded, deduplicated pair sample: uniform non-self pairs plus positives.

    The positive subsample (~1% at the defaults) keeps the sample's positive
    prevalence in the same regime as the complete ``V_hold`` universe the fitted
    predictors are later applied to. All pairs are canonical (``u < v``).
    """
    rng = np.random.default_rng(seed)
    node_list = [str(node) for node in nodes]
    n = len(node_list)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    while len(out) < n_random:
        batch = rng.integers(0, n, size=(n_random, 2))
        for i, j in batch:
            if i == j:
                continue
            pair = (node_list[i], node_list[j]) if i < j else (node_list[j], node_list[i])
            u, v = pair
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                continue
            seen.add((u, v))
            out.append((u, v))
            if len(out) == n_random:
                break
    edges = sorted((str(min(u, v)), str(max(u, v))) for u, v in graph.edges())
    chosen = rng.choice(len(edges), size=min(n_positive, len(edges)), replace=False)
    for edge_index in sorted(chosen):
        pair = edges[edge_index]
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _block_fn(base: PairFeatureFn, block: NDArray[np.float64]) -> PairFeatureFn:
    """Append a precomputed per-pair block to a base pair-feature function."""
    return lambda idx: np.concatenate([base(idx), block[idx]], axis=1)


def _rank_metrics(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> dict[str, float]:
    """AUPRC and AUROC of `scores` against binary `labels`."""
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the S0-R relational-probe CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s0r_relational_probe",
        description=(
            "S0-R relational diagnostic: CN/AA predictability from endpoint features "
            "(Arm A-R) and oracle/predicted relational value for edge decisions (Arm B-R)."
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
    parser.add_argument("--n-random", type=int, default=_DEFAULT_N_RANDOM)
    parser.add_argument("--n-positive", type=int, default=_DEFAULT_N_POSITIVE)
    parser.add_argument("--expected-missing-features", nargs="*", default=list(_DEFAULT_MISSING))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: run both relational arms and write ``s0r_results.json``."""
    args = build_parser().parse_args(argv)
    holdout = _load_holdout(args.data_root, args.strategy, args.expected_missing_features)
    v_fit_nodes = sorted(holdout.v_fit)
    hold_nodes = list(holdout.hold_manifest.nodes)
    g_fit = holdout.build_g_fit()
    g_hold = holdout.build_g_hold()
    features = _load_features(args.feature_root, args.f0_cache, [*v_fit_nodes, *hold_nodes])
    x_fit = features[: len(v_fit_nodes)]
    x_hold = features[len(v_fit_nodes) :]

    # Arm A-R — relational predictability on V_fit pairs only (never V_hold).
    fit_pairs = sample_fit_pairs(
        g_fit, v_fit_nodes, n_random=args.n_random, n_positive=args.n_positive, seed=args.seed
    )
    fit_index = {node: i for i, node in enumerate(v_fit_nodes)}
    u_fit = np.asarray([fit_index[u] for u, _ in fit_pairs], dtype=np.int64)
    v_fit = np.asarray([fit_index[v] for _, v in fit_pairs], dtype=np.int64)
    n_fit = len(fit_pairs)
    logger.info("Arm A-R: %d sampled V_fit pairs", n_fit)
    fit_cn_aa = pair_cn_aa(g_fit, fit_pairs)
    cn_positive = (fit_cn_aa[:, 0] > 0).astype(np.int64)
    regression_targets = {
        "log1p_cn": np.log1p(fit_cn_aa[:, 0]),
        "adamic_adar": fit_cn_aa[:, 1],
    }

    def fit_features(idx: NDArray[np.int64]) -> NDArray[np.float64]:
        return symmetric_pair_features(x_fit, u_fit[idx], v_fit[idx])

    arm_a: dict[str, object] = {
        "n_pairs": n_fit,
        "cn_positive_prevalence": float(cn_positive.mean()),
    }
    scores, fold_lambdas = cv_pair_scores(
        fit_features, n_fit, cn_positive, seed=args.seed, chunk=args.chunk
    )
    arm_a["cn_positive_features"] = {
        **_rank_metrics(cn_positive, scores),
        "fold_lambdas": list(fold_lambdas),
    }
    node_degrees = np.asarray([float(g_fit.degree(node)) for node in v_fit_nodes])
    degree_product = node_degrees[u_fit] * node_degrees[v_fit]
    arm_a["cn_positive_oracle_degree_product"] = _rank_metrics(cn_positive, degree_product)
    degree_probe = nested_probe_r2(x_fit, node_degrees, seed=args.seed)
    predicted_degree = _ridge_fit_predict(
        x_fit, node_degrees, x_fit, _modal_lambda(degree_probe.fold_lambdas)
    )
    arm_a["cn_positive_predicted_degree_product"] = _rank_metrics(
        cn_positive, predicted_degree[u_fit] * predicted_degree[v_fit]
    )
    shuffled_scores, _ = cv_pair_scores(
        fit_features,
        n_fit,
        np.random.default_rng(args.seed).permutation(cn_positive),
        seed=args.seed,
        chunk=args.chunk,
    )
    arm_a["cn_positive_shuffled_control"] = _rank_metrics(cn_positive, shuffled_scores)
    predictor_lambdas: dict[str, float] = {}
    degree_product_block = degree_product.reshape(-1, 1)
    for name, target in regression_targets.items():
        reg_scores, reg_lambdas = cv_pair_scores(
            fit_features, n_fit, target, seed=args.seed, chunk=args.chunk, regression=True
        )
        dp_scores, _ = cv_pair_scores(
            _block_fn(lambda idx: np.empty((len(idx), 0)), degree_product_block),
            n_fit,
            target,
            seed=args.seed,
            chunk=args.chunk,
            regression=True,
        )
        predictor_lambdas[name] = _modal_lambda(reg_lambdas)
        arm_a[name] = {
            "r2_features": _split_r2(reg_scores, target),
            "r2_oracle_degree_product_only": _split_r2(dp_scores, target),
            "fold_lambdas": list(reg_lambdas),
        }
    for key in (
        "cn_positive_features",
        "cn_positive_oracle_degree_product",
        "cn_positive_predicted_degree_product",
        "cn_positive_shuffled_control",
        "log1p_cn",
        "adamic_adar",
    ):
        logger.info("Arm A-R %s: %s", key, arm_a[key])

    # Arm B-R — transfer on the complete V_hold pair universe.
    pairs = holdout.hold_manifest.pairs
    labels = np.asarray(holdout.hold_manifest.labels, dtype=np.int64)
    hold_index = {node: i for i, node in enumerate(hold_nodes)}
    u_hold = np.asarray([hold_index[u] for u, _ in pairs], dtype=np.int64)
    v_hold = np.asarray([hold_index[v] for _, v in pairs], dtype=np.int64)
    logger.info("Arm B-R: %d pairs, %d positives", len(pairs), int(labels.sum()))
    hold_cn_aa = pair_cn_aa(g_hold, pairs)
    oracle_block = np.column_stack([np.log1p(hold_cn_aa[:, 0]), hold_cn_aa[:, 1]])
    shuffled_block = oracle_block[np.random.default_rng(args.seed).permutation(len(pairs))]

    def hold_features(idx: NDArray[np.int64]) -> NDArray[np.float64]:
        return symmetric_pair_features(x_hold, u_hold[idx], v_hold[idx])

    all_hold = np.arange(len(pairs), dtype=np.int64)
    predicted_columns = []
    for name, target in regression_targets.items():
        weights, feature_mean, target_mean = chunked_ridge_fit(
            fit_features, n_fit, target, lam=predictor_lambdas[name], chunk=args.chunk
        )
        predicted_columns.append(
            chunked_ridge_predict(
                hold_features, all_hold, weights, feature_mean, target_mean, chunk=args.chunk
            )
        )
    predicted_block = np.column_stack(predicted_columns)

    rows: dict[str, PairFeatureFn] = {
        "row1_features_only": hold_features,
        "row2r_oracle_cn_aa": _block_fn(hold_features, oracle_block),
        "row3r_predicted_cn_aa": _block_fn(hold_features, predicted_block),
        "row4r_shuffled_cn_aa": _block_fn(hold_features, shuffled_block),
    }
    row_scores: dict[str, NDArray[np.float64]] = {}
    arm_b: dict[str, object] = {}
    for name, feature_fn in rows.items():
        scores, fold_lambdas = cv_pair_scores(
            feature_fn, len(pairs), labels, seed=args.seed, chunk=args.chunk
        )
        row_scores[name] = scores
        arm_b[name] = {**_rank_metrics(labels, scores), "fold_lambdas": list(fold_lambdas)}
        logger.info("Arm B-R %s: %s", name, arm_b[name])

    deltas: dict[str, object] = {}
    for name, (row_a, row_b) in {
        "row2r_minus_row1": ("row2r_oracle_cn_aa", "row1_features_only"),
        "row3r_minus_row1": ("row3r_predicted_cn_aa", "row1_features_only"),
        "row4r_minus_row1": ("row4r_shuffled_cn_aa", "row1_features_only"),
    }.items():
        point, lo, hi = paired_bootstrap_delta(
            labels, row_scores[row_a], row_scores[row_b], n_boot=args.n_boot, seed=args.seed
        )
        deltas[name] = {"delta_auprc": point, "lo": lo, "hi": hi}
        logger.info("delta %s: %s", name, deltas[name])

    def auprc(name: str) -> float:
        entry = arm_b[name]
        assert isinstance(entry, dict)
        return float(entry["auprc"])

    ceiling_gain = auprc("row2r_oracle_cn_aa") - auprc("row1_features_only")
    achieved_gain = auprc("row3r_predicted_cn_aa") - auprc("row1_features_only")
    lo_of = {name: float(entry["lo"]) for name, entry in deltas.items() if isinstance(entry, dict)}
    deltas["decision"] = {
        "relational_ceiling": lo_of["row2r_minus_row1"] > 0,
        "relational_route_signal": lo_of["row3r_minus_row1"] > 0,
        "captured_fraction": achieved_gain / ceiling_gain if ceiling_gain > 0 else None,
        "prevalence": holdout.hold_manifest.prevalence,
    }

    payload: dict[str, object] = {
        "format": "s0r_relational_probe_v1",
        "evidence_class": "diagnostic",
        "strategy": args.strategy,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "deltas": deltas,
        "manifest_digests": {
            "nodes_sha256": holdout.hold_manifest.nodes_sha256,
            "positive_edges_sha256": holdout.hold_manifest.positive_edges_sha256,
            "pair_labels_sha256": holdout.hold_manifest.pair_labels_sha256,
            "g_fit_sha256": g_struct_sha256(g_fit),
            "g_hold_sha256": g_struct_sha256(g_hold),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "s0r_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote S0-R results to %s", output_path)


__all__ = [
    "build_parser",
    "main",
    "pair_cn_aa",
    "sample_fit_pairs",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
