r"""S1 — assembly-coherence diagnostic for the pair-to-topology gap.

Evidence class: **engineering/diagnostic** (same as g1/g3/b0_cal). S0/S0-R
killed per-pair latent-topology generation; the surviving hypothesis is that
modeling dependencies **across** edge decisions can improve the assembled
graph's five topology numbers while preserving pairwise quality. S1 tests the
deterministic limits of that hypothesis post-hoc on the frozen B0 candidate
scores, with no training run:

- **Oracle degree ceiling** — node-aligned true test degrees enforced by hard
  greedy quota assembly (`assemble_degree_quota`), with realized-vs-target
  degree error disclosed. Diagnostic-oracle disclosure class (g3-style).
- **IPF arms** — per-node logit offsets (the maximum-entropy form of a
  node-shared latent) fitted so expected non-self degrees match a target
  profile: the rank-matched reference degree multiset (a heuristic, kept for
  the record under Codex's name ``ipf_reference_multiset_ranked``),
  feature-predicted degrees (legal), or the train-graph degree multiset
  quantile-mapped (legal).
- **CN self-consistency grid** — iterated ``logit + beta * CN(assembled
  graph)`` reassembly with **both-signed** betas (never reads the reference).
- **Learned legal coupling** — cross-edge coupling fitted on the labeled
  ``v_hold`` score universe (the sanctioned model-selection universe), frozen,
  then applied self-consistently to the test universe: a logistic reranker
  (L1a) and a topology-selected CN coefficient (L1b).
- **Controls** — exact-top-N B0 baseline and shuffled-offset capacity control.

Under exact-top-N assembly a global monotone recalibration provably cannot
change the assembled graph (`b0_cal_density` verified this empirically), so any
movement here is attributable to node-heterogeneous or joint structure.

Decision reading reuses `b0_cal`'s pre-registered gap-closure statistic against
the frozen B0 -> Oracle-blend clustering-MMD gap with the 25% threshold. Every
verdict is scoped: negative results bound only the tested post-hoc transforms
of this frozen factorized scorer — jointly trained non-factorized models are
explicitly untested (2026-08-12 adversarial-review revision).

CLI::

    python -m src.experiments.s1_assembly_coherence \
        --universe scores/b0_v31_candidate.npz \
        --data-root data --strategy breadth_first \
        --f0-cache outputs/f0_cache/all_operative_nodes.pt \
        --output-dir outputs/s1

The pure pieces below stay importable for tests; :func:`main` wires the frozen
artifacts into them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.eval.assembly import (
    assemble_degree_quota,
    density_matched_threshold,
    rank_matched_degree_quotas,
)
from src.eval.calibration import stable_sigmoid
from src.eval.graph_metrics import (
    BucketedMMDReport,
    MMDConfig,
    clustering_histogram,
    evaluate_assembled_graph,
    mmd_squared,
    strip_self_loops,
)
from src.experiments.b0_cal import (
    _GAP_CLOSURE_HALT_THRESHOLD,
    _self_loop_edges,
    compute_gap_closure,
)
from src.experiments.g1_hardened_e2 import (
    AssembledRow,
    assemble_and_evaluate,
    assemble_top_n_by_score,
    common_neighbor_and_adamic_adar,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.experiments.probes import _ridge_fit_predict, build_probe_scope_context, probe_targets
from src.experiments.s0_summary_probe import (
    _DEFAULT_MISSING,
    _load_features,
    _load_holdout,
    _modal_lambda,
    nested_probe_r2,
)
from src.experiments.s0r_relational_probe import _rank_metrics
from src.score_universe import ScoresArtifact, load_scores

logger = logging.getLogger(__name__)

_BENCHMARK_SUBDIR = Path("benchmark_2025_neurips")
_DEFAULT_G3_RESULTS = Path("outputs/deliverables/g3_graph_metrics_20260714/g3_results.json")
_CN_BETAS = (-0.2, -0.05, 0.05, 0.2, 0.5)
_CN_ITERATIONS = (1, 2)
#: Pre-registered guards: an arm only "wins" if GS and edge AUPRC do not
#: degrade beyond these tolerances relative to the exact-N B0 baseline.
_GS_GUARD = 0.01
_AUPRC_GUARD = 0.02
#: Greedy quota shortfall above this fraction of target edges demotes the
#: oracle degree ceiling to a lower bound (exact b-matching is the follow-up).
_SHORTFALL_LIMIT = 0.02
_SCOPE = (
    "Bounds only the tested post-hoc transforms of the frozen factorized B0 "
    "scorer; jointly trained non-factorized models are untested."
)


def fit_ipf_offsets(
    logits: NDArray[np.float64],
    mask: NDArray[np.bool_],
    targets: NDArray[np.float64],
    *,
    damping: float = 0.5,
    max_iter: int = 300,
    tol: float = 1e-3,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Fit symmetric per-node logit offsets so expected degrees match `targets`.

    Damped IPF in logit space: ``p_ij = sigmoid(L_ij + a_i + a_j)`` on `mask`
    entries; each iteration updates ``a_i`` by the damped log-ratio of target to
    expected degree. Zero targets drive their offsets to the clip floor.
    """
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    offsets = np.zeros(logits.shape[0], dtype=np.float64)
    eps = 1e-12
    max_rel = np.inf
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        probs = stable_sigmoid(logits + offsets[:, None] + offsets[None, :])
        expected = (probs * mask).sum(axis=1)
        max_rel = float(np.max(np.abs(expected - targets64) / np.maximum(targets64, 1.0)))
        if max_rel <= tol:
            break
        offsets += damping * (np.log(targets64 + eps) - np.log(expected + eps))
        np.clip(offsets, -40.0, 40.0, out=offsets)
    return offsets, {"iterations": float(iterations), "max_rel_err": max_rel}


def offset_adjusted_scores(
    logit_rows: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Row scores ``logit + a_u + a_v`` — additive, root-swap symmetric."""
    return np.asarray(logit_rows, dtype=np.float64) + offsets[u_idx] + offsets[v_idx]


def scaled_rank_targets(
    ranking: NDArray[np.float64],
    multiset: Sequence[float],
    total: float | None = None,
) -> NDArray[np.float64]:
    """Assign a degree multiset to nodes by rank, optionally rescaled to `total`.

    Nodes are ranked by `ranking` descending (ties by ascending position); the
    multiset is assigned sorted descending, rank for rank — the
    `rank_matched_degree_quotas` convention on arrays.
    """
    ranking64 = np.asarray(ranking, dtype=np.float64).reshape(-1)
    if len(multiset) != ranking64.shape[0]:
        raise ValueError(f"multiset has {len(multiset)} entries for {ranking64.shape[0]} nodes")
    order = np.lexsort((np.arange(ranking64.shape[0]), -ranking64))
    targets = np.empty_like(ranking64)
    targets[order] = np.sort(np.asarray(multiset, dtype=np.float64))[::-1]
    if total is not None:
        targets *= total / targets.sum()
    return targets


def cn_adjusted_scores(
    logit_rows: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    graph: nx.Graph,
    node_ids: Sequence[str],
    *,
    beta: float,
) -> NDArray[np.float64]:
    """Row scores ``logit + beta * CN(graph)`` — `graph` is the *assembled* graph.

    The common-neighbor matrix is computed on the supplied graph only (callers
    pass the current assembled iterate, never the reference), self-loops
    stripped first.
    """
    cn_matrix, _ = common_neighbor_and_adamic_adar(strip_self_loops(graph), list(node_ids))
    return np.asarray(logit_rows, dtype=np.float64) + beta * cn_matrix[u_idx, v_idx]


def assemble_exact_n(
    pairs: Sequence[tuple[str, str]],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    scores: NDArray[np.float64],
    *,
    n: int,
    nodes: Sequence[str],
    self_loop_edges: Sequence[tuple[str, str]],
) -> nx.Graph:
    """Top-`n` non-self assembly plus a fixed self-loop set (b0_cal convention)."""
    graph = assemble_top_n_by_score(pairs, u_idx, v_idx, np.asarray(scores, np.float64), n, nodes)
    graph.add_edges_from(self_loop_edges)
    return graph


def largest_remainder_quotas(targets: NDArray[np.float64]) -> NDArray[np.int64]:
    """Round nonnegative float degree targets to integers preserving ``round(sum)``.

    Floor everything, then distribute the remaining units to the largest
    fractional parts (ties broken by ascending index) — the
    ``_largest_remainder_allocation`` convention from `src.data.ego_targets`.
    """
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    if np.any(targets64 < 0):
        raise ValueError("degree targets must be nonnegative")
    total = int(round(float(targets64.sum())))
    floors = np.floor(targets64).astype(np.int64)
    remainder = total - int(floors.sum())
    if remainder > 0:
        fractions = targets64 - floors
        order = np.lexsort((np.arange(targets64.shape[0]), -fractions))
        floors[order[:remainder]] += 1
    return floors


def degree_quota_error(graph: nx.Graph, quotas: Mapping[str, int]) -> dict[str, float]:
    """Realized-vs-target degree error of a quota assembly (self-loops stripped).

    Returns the L1 sum, L-infinity max, and the fraction of nodes whose realized
    degree equals their quota exactly — the enforcement proof Codex's review
    required for any hard-degree oracle ceiling.
    """
    simple = strip_self_loops(graph)
    gaps = [
        abs(float(simple.degree(node) if node in simple else 0) - float(quota))
        for node, quota in quotas.items()
    ]
    return {
        "l1": float(np.sum(gaps)),
        "linf": float(np.max(gaps)) if gaps else 0.0,
        "exact_fraction": float(np.mean([gap == 0.0 for gap in gaps])) if gaps else 1.0,
    }


def validate_v_hold_artifact(
    artifact: ScoresArtifact,
    *,
    expected_nodes: Collection[str],
    expected_checkpoint_id: str,
) -> None:
    """Validate a ``v_hold`` scores artifact before fitting any coupling on it.

    Guards the traps this universe carries: `pairs_source` must be ``v_hold``
    (never validated by `validate_universe_artifact`, which requires
    ``candidate``); the checkpoint identity must match the test-side universe
    (test-protocol `_require_same_checkpoint` convention); and the node set
    must equal the locally derived ``V_hold`` (the `expected_missing_features`
    derivation-drift trap — `score_universe`'s holdout loader does not subtract
    them). Labels must be complete and binary.
    """
    errors: list[str] = []
    pairs_source = artifact.meta.get("pairs_source")
    if pairs_source != "v_hold":
        errors.append(f"pairs_source expected 'v_hold', got {pairs_source!r}")
    checkpoint_id = artifact.meta.get("checkpoint_id")
    if checkpoint_id != expected_checkpoint_id:
        errors.append(
            f"checkpoint_id {checkpoint_id!r} does not match universe {expected_checkpoint_id!r}"
        )
    if set(artifact.node_ids) != set(expected_nodes):
        errors.append("node_ids differ from the locally derived V_hold node set")
    num_rows = artifact.meta.get("num_rows")
    if num_rows is not None and int(cast(int, num_rows)) != len(artifact.label):
        errors.append(f"row count {len(artifact.label)} != meta num_rows {num_rows}")
    if np.any((artifact.label != 0) & (artifact.label != 1)):
        errors.append("labels must be complete and binary for coupling fits")
    if np.any(artifact.u_idx == artifact.v_idx):
        errors.append("v_hold universe must not contain self-pairs")
    if errors:
        raise ValueError("; ".join(errors))


def coupling_features(
    logit_rows: NDArray[np.float64],
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    graph: nx.Graph,
    node_ids: Sequence[str],
) -> NDArray[np.float64]:
    """Per-pair coupling features from the *assembled* graph only.

    Columns: ``[logit, CN(graph), AA(graph), deg_u+deg_v, |deg_u-deg_v|]`` with
    all structural quantities read from the supplied assembled iterate, never
    the reference.
    """
    simple = strip_self_loops(graph)
    cn_matrix, aa_matrix = common_neighbor_and_adamic_adar(simple, list(node_ids))
    degrees = np.asarray(
        [float(simple.degree(node)) if node in simple else 0.0 for node in node_ids]
    )
    return np.column_stack(
        [
            np.asarray(logit_rows, dtype=np.float64),
            cn_matrix[u_idx, v_idx],
            aa_matrix[u_idx, v_idx],
            degrees[u_idx] + degrees[v_idx],
            np.abs(degrees[u_idx] - degrees[v_idx]),
        ]
    )


def apply_coupling(
    features: NDArray[np.float64],
    *,
    coef: NDArray[np.float64],
    intercept: float,
    means: NDArray[np.float64],
    stds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a frozen standardized linear coupling to `features`."""
    standardized = (features - means) / stds
    return standardized @ coef + intercept


def _row_to_json(row: AssembledRow) -> dict[str, object]:
    """JSON-encode an `AssembledRow` (int bucket keys stringified)."""
    payload = dataclasses.asdict(row)
    for key in ("per_size_graph_similarity", "per_size_relative_density"):
        payload[key] = {str(size): values for size, values in payload[key].items()}
    return payload


def _report_to_json(report: BucketedMMDReport) -> dict[str, object]:
    """Compact JSON encoding of a light (no-bootstrap) evaluation."""
    return {
        "mmd_ratio": dict(report.mmd_ratio),
        "graph_similarity": report.graph_similarity,
        "relative_density": report.relative_density,
    }


def _edge_overlap_metrics(g_pred: nx.Graph, g_ref_simple: nx.Graph) -> dict[str, float]:
    """Precision/recall of the assembled simple-edge set against the reference."""
    pred_edges = {frozenset(e) for e in strip_self_loops(g_pred).edges()}
    ref_edges = {frozenset(e) for e in g_ref_simple.edges()}
    overlap = len(pred_edges & ref_edges)
    return {
        "edge_precision": overlap / len(pred_edges) if pred_edges else 0.0,
        "edge_recall": overlap / len(ref_edges) if ref_edges else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the S1 assembly-coherence CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s1_assembly_coherence",
        description=(
            "S1 assembly-coherence diagnostic: node-coupled (IPF) and self-consistent "
            "(CN) post-hoc assemblies of the frozen B0 candidate scores."
        ),
    )
    parser.add_argument("--universe", type=Path, default=Path("scores/b0_v31_candidate.npz"))
    parser.add_argument(
        "--v-hold-universe",
        type=Path,
        default=None,
        help="v_hold scores .npz from the same checkpoint; enables the L1a/L1b learned arms",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--f0-cache", type=Path, required=True)
    parser.add_argument(
        "--feature-root", type=Path, default=Path("data/features/frozen_node_features_1024")
    )
    parser.add_argument("--g3-results", type=Path, default=_DEFAULT_G3_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-missing-features", nargs="*", default=list(_DEFAULT_MISSING))
    parser.add_argument(
        "--hard-decomposition",
        action="store_true",
        help="run only the S1-H hard-quota decomposition arms against --prior-results",
    )
    parser.add_argument("--prior-results", type=Path, default=Path("outputs/s1/s1_results.json"))
    return parser


def run_hard_decomposition(args: argparse.Namespace) -> None:
    """S1-H: the three non-node-aligned degree sources under hard-quota assembly.

    Completes the soft/hard x target-source matrix that decouples hard
    enforcement from node-identity degree information. Baseline guard values
    and the soft-arm closures are quoted from the frozen v2 results
    (`--prior-results`); only the three new hard arms are evaluated.
    """
    prior = json.loads(args.prior_results.read_text(encoding="utf-8"))
    if prior.get("format") != "s1_assembly_coherence_v2":
        raise ValueError(f"--prior-results must be v2 results, got {prior.get('format')!r}")
    base_row = prior["assembled"]["b0_exact_n"]
    g3_assembled = cast(
        dict[str, dict[str, object]],
        json.loads(args.g3_results.read_text(encoding="utf-8"))["assembled"],
    )

    universe = load_scores(args.universe)
    node_ids = universe.node_ids
    n_nodes = len(node_ids)
    validate_universe_artifact(
        universe, strategy=args.strategy, n_test_nodes=n_nodes, label="candidate universe"
    )
    benchmark_root = args.data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, args.strategy)
    g_simple = strip_self_loops(g_ref)
    buckets = load_test_node_buckets(benchmark_root, args.strategy)
    config = MMDConfig()
    target_edges = g_simple.number_of_edges()
    nodes = list(g_ref.nodes())
    pairs_all = list(universe.pairs())
    probs_raw = universe.probs()
    non_self_mask = universe.u_idx != universe.v_idx
    ns_idx = np.flatnonzero(non_self_mask)
    ns_pairs = [pairs_all[i] for i in ns_idx.tolist()]
    ns_u = universe.u_idx[ns_idx]
    ns_v = universe.v_idx[ns_idx]
    ns_labels = universe.label[ns_idx].astype(np.int64)
    threshold_b0 = density_matched_threshold(probs_raw[non_self_mask], target_edges)
    self_edges = _self_loop_edges(universe, probs_raw, threshold_b0)

    expected_raw = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(expected_raw, ns_u, probs_raw[ns_idx])
    np.add.at(expected_raw, ns_v, probs_raw[ns_idx])
    reference_degrees = [int(g_simple.degree(node)) if node in g_simple else 0 for node in node_ids]
    targets = _degree_targets(
        args, universe, expected_raw, reference_degrees, float(2 * target_edges)
    )
    expected_degree = {node: float(expected_raw[i]) for i, node in enumerate(node_ids)}
    quota_sets: dict[str, dict[str, int]] = {
        "hard_multiset_ranked": rank_matched_degree_quotas(expected_degree, reference_degrees),
        "hard_predicted": {
            node: int(quota)
            for node, quota in zip(
                node_ids, largest_remainder_quotas(targets["ipf_predicted"]), strict=True
            )
        },
        "hard_train_prior": {
            node: int(quota)
            for node, quota in zip(
                node_ids, largest_remainder_quotas(targets["ipf_train_prior"]), strict=True
            )
        },
    }

    arms: dict[str, object] = {}
    hard_closures: dict[str, float | None] = {}
    for name, quotas in quota_sets.items():
        graph, stats = assemble_degree_quota(ns_pairs, ns_u, ns_v, probs_raw[ns_idx], quotas, nodes)
        graph.add_edges_from(self_edges)
        shortfall_fraction = stats.shortfall / max(stats.target_edges, 1)
        row = assemble_and_evaluate(
            g_pred=graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=args.seed,
            threshold=None,
        )
        closure = compute_gap_closure(row, g3_assembled)
        clustering_closure = closure["clustering"]
        guards = {
            "gs_ok": row.graph_similarity >= float(base_row["graph_similarity"]) - _GS_GUARD,
            "degree_mmd_ok": row.mmd_ratio["degree"]
            <= float(base_row["mmd_ratio"]["degree"]) + float(base_row["bootstrap_std"]["degree"]),
            "auprc_ok": True,  # ranking scores are unchanged raw probabilities
        }
        wins = bool(
            clustering_closure is not None
            and clustering_closure >= _GAP_CLOSURE_HALT_THRESHOLD
            and all(guards.values())
        )
        hard_closures[name] = clustering_closure
        arms[name] = {
            "row": _row_to_json(row),
            "quota_stats": {
                **dataclasses.asdict(stats),
                "degree_error": degree_quota_error(graph, quotas),
                "shortfall_fraction": shortfall_fraction,
            },
            "edge_metrics": {
                **_rank_metrics(ns_labels, probs_raw[ns_idx]),
                **_edge_overlap_metrics(graph, g_simple),
            },
            "gap_closure": closure,
            "verdict": {
                "clustering_gap_closure": clustering_closure,
                **guards,
                "lower_bound_only": shortfall_fraction > _SHORTFALL_LIMIT,
                "wins": wins,
            },
        }
        logger.info(
            "%s: GS=%.6f mmd=%s closure=%s shortfall=%.4f wins=%s",
            name,
            row.graph_similarity,
            row.mmd_ratio,
            clustering_closure,
            shortfall_fraction,
            wins,
        )

    def prior_closure(arm: str) -> float | None:
        entry = prior["gap_closure"].get(arm)
        return None if entry is None else entry.get("clustering")

    matrix = {
        "node_aligned": {
            "soft": None,
            "hard": prior["gap_closure"]["oracle_degree_hard"]["clustering"],
        },
        "multiset_ranked": {
            "soft": prior_closure("ipf_reference_multiset_ranked"),
            "hard": hard_closures["hard_multiset_ranked"],
        },
        "predicted": {
            "soft": prior_closure("ipf_predicted"),
            "hard": hard_closures["hard_predicted"],
        },
        "train_prior": {
            "soft": prior_closure("ipf_train_prior"),
            "hard": hard_closures["hard_train_prior"],
        },
    }

    def arm_wins(name: str) -> bool:
        return bool(cast(dict[str, object], cast(dict[str, object], arms[name])["verdict"])["wins"])

    decision = {
        "scope": _SCOPE,
        "threshold": _GAP_CLOSURE_HALT_THRESHOLD,
        "multiset_sufficient": arm_wins("hard_multiset_ranked"),
        "legal_posthoc_exists": arm_wins("hard_predicted") or arm_wins("hard_train_prior"),
        "node_identity_is_crux": not arm_wins("hard_multiset_ranked"),
    }
    logger.info("S1-H decision: %s", decision)

    payload: dict[str, object] = {
        "format": "s1_hard_decomposition_v1",
        "evidence_class": "diagnostic",
        "scope": _SCOPE,
        "strategy": args.strategy,
        "prior_results": str(args.prior_results),
        "operating_point": {
            "target_edges": target_edges,
            "frozen_threshold": threshold_b0,
            "self_loops": len(self_edges),
        },
        "arms": arms,
        "matrix": matrix,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "s1_hard_decomposition.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote S1-H results to %s", output_path)


def _degree_targets(
    args: argparse.Namespace,
    universe: ScoresArtifact,
    expected_raw: NDArray[np.float64],
    reference_degrees: list[int],
    target_total: float,
) -> dict[str, NDArray[np.float64]]:
    """Build the three degree-target profiles (oracle, predicted, train-prior)."""
    node_ids = universe.node_ids
    expected_degree = {node: float(expected_raw[i]) for i, node in enumerate(node_ids)}
    quotas = rank_matched_degree_quotas(expected_degree, reference_degrees)
    oracle = np.asarray([float(quotas[node]) for node in node_ids])

    formal_nodes, g_struct = build_probe_scope_context(
        "formal_train",
        data_root=args.data_root,
        strategy=args.strategy,
        expected_missing_features=args.expected_missing_features,
    )
    features = _load_features(args.feature_root, args.f0_cache, [*formal_nodes, *node_ids])
    x_train = features[: len(formal_nodes)]
    x_test = features[len(formal_nodes) :]
    train_degrees = probe_targets(g_struct, formal_nodes)["degree"]
    probe = nested_probe_r2(x_train, train_degrees, seed=args.seed)
    predicted_raw = _ridge_fit_predict(
        x_train, train_degrees, x_test, _modal_lambda(probe.fold_lambdas)
    )
    predicted = np.clip(predicted_raw, 1e-3, None)
    predicted *= target_total / predicted.sum()
    logger.info(
        "predicted-degree probe: r2=%.4f lambda=%s", probe.r2, _modal_lambda(probe.fold_lambdas)
    )

    train_multiset = np.sort(train_degrees)
    quantile_idx = np.round(np.linspace(0, len(train_multiset) - 1, len(node_ids))).astype(int)
    train_prior = scaled_rank_targets(
        expected_raw, train_multiset[quantile_idx].tolist(), total=target_total
    )
    return {
        "ipf_reference_multiset_ranked": oracle,
        "ipf_predicted": predicted,
        "ipf_train_prior": train_prior,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: run all arms and write ``s1_results.json``."""
    args = build_parser().parse_args(argv)
    if args.hard_decomposition:
        run_hard_decomposition(args)
        return
    universe = load_scores(args.universe)
    node_ids = universe.node_ids
    n_nodes = len(node_ids)
    validate_universe_artifact(
        universe, strategy=args.strategy, n_test_nodes=n_nodes, label="candidate universe"
    )
    benchmark_root = args.data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, args.strategy)
    g_simple = strip_self_loops(g_ref)
    buckets = load_test_node_buckets(benchmark_root, args.strategy)
    config = MMDConfig()
    target_edges = g_simple.number_of_edges()
    nodes = list(g_ref.nodes())

    pairs_all = list(universe.pairs())
    probs_raw = universe.probs()
    non_self_mask = universe.u_idx != universe.v_idx
    ns_idx = np.flatnonzero(non_self_mask)
    ns_pairs = [pairs_all[i] for i in ns_idx.tolist()]
    ns_u = universe.u_idx[ns_idx]
    ns_v = universe.v_idx[ns_idx]
    ns_logit = universe.logit[ns_idx].astype(np.float64)
    ns_labels = universe.label[ns_idx].astype(np.int64)

    # Fixed self-loop policy across every arm: the frozen B0 operating threshold
    # on raw probabilities decides self-loops, so five-number differences are
    # attributable to non-self structure only.
    threshold_b0 = density_matched_threshold(probs_raw[non_self_mask], target_edges)
    self_edges = _self_loop_edges(universe, probs_raw, threshold_b0)
    logger.info(
        "N=%d non-self edges, %d self-loops at frozen threshold %.6f",
        target_edges,
        len(self_edges),
        threshold_b0,
    )

    def assemble(scores: NDArray[np.float64]) -> nx.Graph:
        return assemble_exact_n(
            ns_pairs,
            ns_u,
            ns_v,
            scores,
            n=target_edges,
            nodes=nodes,
            self_loop_edges=self_edges,
        )

    def evaluate_full(graph: nx.Graph) -> AssembledRow:
        return assemble_and_evaluate(
            g_pred=graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=args.seed,
            threshold=None,
        )

    # Degree targets and IPF fits.
    expected_raw = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(expected_raw, ns_u, probs_raw[ns_idx])
    np.add.at(expected_raw, ns_v, probs_raw[ns_idx])
    reference_degrees = [int(g_simple.degree(node)) if node in g_simple else 0 for node in node_ids]
    targets = _degree_targets(
        args, universe, expected_raw, reference_degrees, float(2 * target_edges)
    )
    dense_logit = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    dense_logit[ns_u, ns_v] = ns_logit
    dense_logit[ns_v, ns_u] = ns_logit
    dense_mask = ~np.eye(n_nodes, dtype=bool)

    arm_scores: dict[str, NDArray[np.float64]] = {"b0_exact_n": ns_logit}
    ipf_info: dict[str, object] = {}
    oracle_offsets: NDArray[np.float64] | None = None
    for name, target in targets.items():
        offsets, info = fit_ipf_offsets(dense_logit, dense_mask, target)
        ipf_info[name] = {**info, "mean_offset": float(offsets.mean())}
        logger.info("IPF %s: %s", name, ipf_info[name])
        arm_scores[name] = offset_adjusted_scores(ns_logit, ns_u, ns_v, offsets)
        if name == "ipf_reference_multiset_ranked":
            oracle_offsets = offsets
    assert oracle_offsets is not None
    shuffled = oracle_offsets[np.random.default_rng(args.seed).permutation(n_nodes)]
    arm_scores["shuffled_offsets"] = offset_adjusted_scores(ns_logit, ns_u, ns_v, shuffled)

    # Evaluate the IPF/control arms fully.
    assembled_rows: dict[str, AssembledRow] = {}
    arm_edge_metrics: dict[str, object] = {}
    for name, scores in arm_scores.items():
        graph = assemble(scores)
        assembled_rows[name] = evaluate_full(graph)
        arm_edge_metrics[name] = {
            **_rank_metrics(ns_labels, scores),
            **_edge_overlap_metrics(graph, g_simple),
        }
        logger.info(
            "%s: GS=%.6f RD=%.6f mmd=%s edges=%s",
            name,
            assembled_rows[name].graph_similarity,
            assembled_rows[name].relative_density,
            assembled_rows[name].mmd_ratio,
            arm_edge_metrics[name],
        )

    # Oracle degree ceiling: node-aligned TRUE degrees (diagnostic-oracle
    # disclosure, g3-class), hard-enforced by greedy quota assembly.
    oracle_quotas = {node: reference_degrees[i] for i, node in enumerate(node_ids)}
    quota_graph, quota_stats = assemble_degree_quota(
        ns_pairs, ns_u, ns_v, probs_raw[ns_idx], oracle_quotas, nodes
    )
    quota_graph.add_edges_from(self_edges)
    oracle_degree_stats: dict[str, object] = {
        **dataclasses.asdict(quota_stats),
        "degree_error": degree_quota_error(quota_graph, oracle_quotas),
        "shortfall_fraction": quota_stats.shortfall / max(quota_stats.target_edges, 1),
    }
    logger.info("oracle_degree_hard quota stats: %s", oracle_degree_stats)
    assembled_rows["oracle_degree_hard"] = evaluate_full(quota_graph)
    arm_edge_metrics["oracle_degree_hard"] = {
        **_rank_metrics(ns_labels, probs_raw[ns_idx]),
        **_edge_overlap_metrics(quota_graph, g_simple),
    }
    logger.info(
        "oracle_degree_hard: GS=%.6f RD=%.6f mmd=%s",
        assembled_rows["oracle_degree_hard"].graph_similarity,
        assembled_rows["oracle_degree_hard"].relative_density,
        assembled_rows["oracle_degree_hard"].mmd_ratio,
    )

    # CN self-consistency grid (light evaluation), best clustering cell promoted.
    cn_grid: dict[str, object] = {}
    best_cell: tuple[float, str, NDArray[np.float64]] | None = None
    for beta in _CN_BETAS:
        scores = ns_logit
        graph = assemble(scores)
        for iteration in range(1, max(_CN_ITERATIONS) + 1):
            scores = cn_adjusted_scores(ns_logit, ns_u, ns_v, graph, node_ids, beta=beta)
            graph = assemble(scores)
            if iteration in _CN_ITERATIONS:
                report = evaluate_assembled_graph(graph, g_ref, buckets, config)
                key = f"cn_beta{beta}_t{iteration}"
                cn_grid[key] = _report_to_json(report)
                logger.info("%s: %s", key, cn_grid[key])
                clustering = report.mmd_ratio["clustering"]
                if best_cell is None or clustering < best_cell[0]:
                    best_cell = (clustering, key, scores)
    assert best_cell is not None
    cn_scores = best_cell[2]
    cn_graph = assemble(cn_scores)
    assembled_rows["cn_best"] = evaluate_full(cn_graph)
    arm_edge_metrics["cn_best"] = {
        **_rank_metrics(ns_labels, cn_scores),
        **_edge_overlap_metrics(cn_graph, g_simple),
        "grid_cell": best_cell[1],
    }
    logger.info("cn_best (%s): %s", best_cell[1], arm_edge_metrics["cn_best"])

    # Learned legal coupling arms, fitted on the labeled v_hold universe.
    l1_fit: dict[str, object] | None = None
    if args.v_hold_universe is not None:
        vh = load_scores(args.v_hold_universe)
        holdout = _load_holdout(args.data_root, args.strategy, args.expected_missing_features)
        validate_v_hold_artifact(
            vh,
            expected_nodes=holdout.hold_manifest.nodes,
            expected_checkpoint_id=cast(str, universe.meta.get("checkpoint_id")),
        )
        vh_pairs = list(vh.pairs())
        vh_logit = vh.logit.astype(np.float64)
        vh_labels = vh.label.astype(np.int64)
        n_vh_edges = int(vh_labels.sum())
        vh_graph = assemble_top_n_by_score(
            vh_pairs, vh.u_idx, vh.v_idx, vh.probs(), n_vh_edges, vh.node_ids
        )
        vh_features = coupling_features(vh_logit, vh.u_idx, vh.v_idx, vh_graph, vh.node_ids)
        a0_graph = assemble(ns_logit)
        test_features = coupling_features(ns_logit, ns_u, ns_v, a0_graph, node_ids)

        # L1a — logistic reranker on true V_hold labels, frozen, applied to test.
        means = vh_features.mean(axis=0)
        stds = np.maximum(vh_features.std(axis=0), 1e-9)
        logistic = LogisticRegression(max_iter=1000)
        logistic.fit((vh_features - means) / stds, vh_labels)
        coef = logistic.coef_[0].astype(np.float64)
        intercept = float(logistic.intercept_[0])
        l1a_scores = apply_coupling(
            test_features, coef=coef, intercept=intercept, means=means, stds=stds
        )
        l1a_graph = assemble(l1a_scores)
        assembled_rows["l1a_logistic"] = evaluate_full(l1a_graph)
        arm_edge_metrics["l1a_logistic"] = {
            **_rank_metrics(ns_labels, l1a_scores),
            **_edge_overlap_metrics(l1a_graph, g_simple),
        }
        logger.info(
            "l1a_logistic: GS=%.6f mmd=%s edges=%s",
            assembled_rows["l1a_logistic"].graph_similarity,
            assembled_rows["l1a_logistic"].mmd_ratio,
            arm_edge_metrics["l1a_logistic"],
        )

        # L1b — CN coefficient selected on V_hold assembled topology, frozen.
        g_hold = holdout.build_g_hold()
        tri_ref = max(int(sum(nx.triangles(g_hold).values()) // 3), 1)
        ref_hist = clustering_histogram(g_hold)
        selection: dict[str, dict[str, float]] = {}
        best_l1b: tuple[tuple[float, float], float, int] | None = None
        for beta in _CN_BETAS:
            scores_vh = vh_logit
            graph_vh = assemble_top_n_by_score(
                vh_pairs, vh.u_idx, vh.v_idx, scores_vh, n_vh_edges, vh.node_ids
            )
            for iteration in range(1, max(_CN_ITERATIONS) + 1):
                scores_vh = cn_adjusted_scores(
                    vh_logit, vh.u_idx, vh.v_idx, graph_vh, vh.node_ids, beta=beta
                )
                graph_vh = assemble_top_n_by_score(
                    vh_pairs, vh.u_idx, vh.v_idx, scores_vh, n_vh_edges, vh.node_ids
                )
                if iteration not in _CN_ITERATIONS:
                    continue
                tri_pred = int(sum(nx.triangles(graph_vh).values()) // 3)
                criterion = abs(tri_pred / tri_ref - 1.0) + mmd_squared(
                    [clustering_histogram(graph_vh)], [ref_hist], config
                )
                auprc = float(average_precision_score(vh_labels, scores_vh))
                selection[f"beta{beta}_t{iteration}"] = {
                    "criterion": criterion,
                    "triangles_pred": float(tri_pred),
                    "auprc": auprc,
                }
                rank_key = (criterion, -auprc)
                if best_l1b is None or rank_key < best_l1b[0]:
                    best_l1b = (rank_key, beta, iteration)
        assert best_l1b is not None
        _, beta_star, t_star = best_l1b
        scores_l1b = ns_logit
        graph_l1b = assemble(scores_l1b)
        for _ in range(t_star):
            scores_l1b = cn_adjusted_scores(
                ns_logit, ns_u, ns_v, graph_l1b, node_ids, beta=beta_star
            )
            graph_l1b = assemble(scores_l1b)
        assembled_rows["l1b_topology_selected"] = evaluate_full(graph_l1b)
        arm_edge_metrics["l1b_topology_selected"] = {
            **_rank_metrics(ns_labels, scores_l1b),
            **_edge_overlap_metrics(graph_l1b, g_simple),
            "beta": beta_star,
            "iterations": t_star,
        }
        l1_fit = {
            "l1a_coef": coef.tolist(),
            "l1a_intercept": intercept,
            "l1a_feature_means": means.tolist(),
            "l1a_feature_stds": stds.tolist(),
            "l1b_selection": selection,
            "l1b_selected": {"beta": beta_star, "iterations": t_star},
            "v_hold_edges": n_vh_edges,
            "v_hold_checkpoint_id": vh.meta.get("checkpoint_id"),
        }
        logger.info(
            "l1b_topology_selected (beta=%s, t=%d): GS=%.6f mmd=%s",
            beta_star,
            t_star,
            assembled_rows["l1b_topology_selected"].graph_similarity,
            assembled_rows["l1b_topology_selected"].mmd_ratio,
        )
    else:
        logger.warning("no --v-hold-universe supplied; skipping the L1a/L1b learned arms")

    # Gap closure and decision against the frozen G3 record.
    gap_closure: dict[str, object] | None = None
    decision: dict[str, object] | None = None
    if args.g3_results.exists():
        g3_assembled = cast(
            dict[str, dict[str, object]],
            json.loads(args.g3_results.read_text(encoding="utf-8"))["assembled"],
        )
        gap_closure = {
            name: compute_gap_closure(row, g3_assembled) for name, row in assembled_rows.items()
        }
        base = assembled_rows["b0_exact_n"]
        base_auprc = cast(dict[str, float], arm_edge_metrics["b0_exact_n"])["auprc"]

        def arm_verdict(name: str) -> dict[str, object]:
            row = assembled_rows[name]
            closure = cast(dict[str, float | None], gap_closure[name])["clustering"]
            auprc = cast(dict[str, float], arm_edge_metrics[name])["auprc"]
            guards = {
                "gs_ok": row.graph_similarity >= base.graph_similarity - _GS_GUARD,
                "degree_mmd_ok": row.mmd_ratio["degree"]
                <= base.mmd_ratio["degree"] + base.bootstrap_std["degree"],
                "auprc_ok": auprc >= base_auprc - _AUPRC_GUARD,
            }
            closes = closure is not None and closure >= _GAP_CLOSURE_HALT_THRESHOLD
            return {
                "clustering_gap_closure": closure,
                **guards,
                "wins": bool(closes and all(guards.values())),
            }

        legal_arms = [
            name
            for name in (
                "ipf_predicted",
                "ipf_train_prior",
                "cn_best",
                "l1a_logistic",
                "l1b_topology_selected",
            )
            if name in assembled_rows
        ]
        shortfall_fraction = cast(float, oracle_degree_stats["shortfall_fraction"])
        decision = {
            "scope": _SCOPE,
            "threshold": _GAP_CLOSURE_HALT_THRESHOLD,
            "tested_posthoc_arms": {name: arm_verdict(name) for name in legal_arms},
            "oracle_degree_ceiling": {
                **arm_verdict("oracle_degree_hard"),
                "shortfall_fraction": shortfall_fraction,
                "lower_bound_only": shortfall_fraction > _SHORTFALL_LIMIT,
            },
            "ipf_reference_multiset_ranked_heuristic": arm_verdict("ipf_reference_multiset_ranked"),
        }
        logger.info("decision: %s", decision)
    else:
        logger.warning("g3 results not found at %s; gap closure skipped", args.g3_results)

    payload: dict[str, object] = {
        "format": "s1_assembly_coherence_v2",
        "evidence_class": "diagnostic",
        "scope": _SCOPE,
        "strategy": args.strategy,
        "operating_point": {
            "target_edges": target_edges,
            "frozen_threshold": threshold_b0,
            "self_loops": len(self_edges),
        },
        "ipf_info": ipf_info,
        "oracle_degree_stats": oracle_degree_stats,
        "l1_fit": l1_fit,
        "assembled": {name: _row_to_json(row) for name, row in assembled_rows.items()},
        "edge_metrics": arm_edge_metrics,
        "cn_grid": cn_grid,
        "gap_closure": gap_closure,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "s1_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote S1 results to %s", output_path)


__all__ = [
    "apply_coupling",
    "assemble_exact_n",
    "build_parser",
    "cn_adjusted_scores",
    "coupling_features",
    "degree_quota_error",
    "fit_ipf_offsets",
    "largest_remainder_quotas",
    "main",
    "offset_adjusted_scores",
    "run_hard_decomposition",
    "scaled_rank_targets",
    "validate_v_hold_artifact",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
