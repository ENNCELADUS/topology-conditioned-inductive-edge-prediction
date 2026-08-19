r"""S4 budget-assembly diagnostic: hard degree-quota replay of frozen B0 scores.

Score-time replay, no training. Tests whether a per-node degree *budget* --
supplied either by a true test-graph oracle or by an external predicted-degree
file -- lets hard-quota assembly (`assemble_degree_quota`) close more of the
pair-to-topology gap than the frozen B0 scorer's own exact-top-N control, using
the identical set of candidate pairs and (non-quota) B0 scores to break every
greedy tie.

Arms (all read the same frozen candidate-universe artifact):

- ``b0_exact_n``      — control: exact top-``target_edges`` non-self pairs by
                        raw B0 score (`assemble_top_n_by_score`), plus the
                        shared self-loop set.
- ``oracle_hard``     — true test-graph loopless degrees (`g_simple.degree`,
                        0 for a node absent from `g_simple`) as hard quotas,
                        greedily assembled (`assemble_degree_quota`), plus the
                        shared self-loop set. Diagnostic-oracle disclosure
                        class (g3-style): illegal as a selected method.
- ``predicted_hard_<variant>`` — one arm per ``--degree-predictions`` file: an
                        externally supplied per-node expected-degree profile,
                        clipped to >= 1e-3, rescaled to sum exactly
                        ``2 * target_edges``, rounded to integer quotas via
                        `largest_remainder_quotas`, then assembled the same
                        way. ``<variant>`` is the file's stem; two files with
                        the same stem raise (no silent overwrite).

Self-pairs are never routed into `assemble_degree_quota` (it raises on
self-pair rows). Instead every arm shares one frozen self-loop set computed
once, at the B0 density-matched non-self operating point
(``threshold_b0 = density_matched_threshold(probs[non_self], target_edges)``),
via :func:`src.experiments.b0_cal._self_loop_edges`.

Per arm this reports the five topology numbers (BFS-macro GS/RD plus the
degree/clustering/spectral MMD ratios, never aggregated -- CLAUDE.md claim
rule), the global simple-edge GS/RD (named separately from the BFS-macro
pair), edge-set precision/recall, and -- for the two hard-quota arms --
`QuotaAssemblyStats`, `degree_quota_error`, the shortfall fraction, and a
``lower_bound_only`` flag (shortfall > 2% of the quota-implied edge count: no
back-fill pass exists, so greedy stranding can leave real quota unconsumed,
and the reported numbers are then only a lower bound on what that quota
profile could achieve). Every ``predicted_hard_<variant>`` arm additionally
reports its per-axis gap closure of this run's own ``b0_exact_n`` ->
``oracle_hard`` gap (the five topology numbers only; the global simple-edge
variants have no oracle-referenced closure).

CLI::

    python -m src.experiments.s4_budget_assembly \
        --universe outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz \
        --data-root data --strategy breadth_first --output-dir outputs/s4/... \
        --seed 0 [--degree-predictions predictions_a.json] [--degree-predictions predictions_b.json]

Determinism: identical inputs produce a byte-identical ``s4_results.json``
(stable sorts everywhere; ``json.dumps(..., sort_keys=True)``; no wall-clock
fields in the payload).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.eval.assembly import (
    QuotaAssemblyStats,
    assemble_degree_quota,
    degree_quota_error,
    density_matched_threshold,
    largest_remainder_quotas,
)
from src.eval.graph_metrics import (
    STATISTICS,
    BucketReference,
    MMDConfig,
    compute_graph_similarity,
    compute_relative_density,
    evaluate_assembled_graph_with_reference,
    precompute_bucket_reference,
    strip_self_loops,
)
from src.experiments.b0_cal import _self_loop_edges
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    _artifact_meta_summary,
    assemble_top_n_by_score,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)

_DEGREE_PREDICTIONS_FORMAT = "degree_predictions_v1"
_MIN_PREDICTED_DEGREE = 1e-3
_LOWER_BOUND_SHORTFALL_THRESHOLD = 0.02


# --------------------------------------------------------------------------- degree predictions I/O


def load_degree_predictions(path: Path, node_ids: Sequence[str]) -> NDArray[np.float64]:
    """Load and validate one ``degree_predictions_v1`` file, aligned to `node_ids` order.

    Args:
        path: Path to the predictions JSON file. Expected shape:
            ``{"format": "degree_predictions_v1", "degree_predictions":
            {<node_id>: <float>, ...}}``.
        node_ids: The candidate universe's node_ids; defines both the required
            node set and the output order.

    Returns:
        Shape ``(len(node_ids),)`` float64 raw predicted expected degrees, in
        `node_ids` order.

    Raises:
        ValueError: If ``format`` is not ``"degree_predictions_v1"``,
            ``degree_predictions`` is not an object, or its node set does not
            exactly equal ``set(node_ids)``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    fmt = payload.get("format")
    if fmt != _DEGREE_PREDICTIONS_FORMAT:
        raise ValueError(f"{path}: expected format {_DEGREE_PREDICTIONS_FORMAT!r}, got {fmt!r}")
    predictions = payload.get("degree_predictions")
    if not isinstance(predictions, dict):
        raise ValueError(
            f"{path}: 'degree_predictions' must be an object mapping node id -> degree"
        )
    predicted_nodes = set(predictions)
    expected_nodes = set(node_ids)
    if predicted_nodes != expected_nodes:
        missing = sorted(expected_nodes - predicted_nodes)
        extra = sorted(predicted_nodes - expected_nodes)
        raise ValueError(
            f"{path}: node set mismatch vs universe.node_ids "
            f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})"
        )
    return np.array([float(predictions[node]) for node in node_ids], dtype=np.float64)


def predicted_hard_quotas(raw_degrees: NDArray[np.float64], target_edges: int) -> NDArray[np.int64]:
    """Scale raw predicted degrees into hard integer quotas summing to ``2 * target_edges``.

    Args:
        raw_degrees: Row-aligned raw predicted expected degrees (any sign or
            magnitude).
        target_edges: The simple-edge reference edge count.

    Returns:
        Integer quotas, same shape as `raw_degrees`, summing to
        ``2 * target_edges``.
    """
    clipped = np.clip(np.asarray(raw_degrees, dtype=np.float64), _MIN_PREDICTED_DEGREE, None)
    scale = (2.0 * target_edges) / float(clipped.sum())
    return largest_remainder_quotas(clipped * scale)


def oracle_hard_quotas(g_simple: nx.Graph, node_ids: Sequence[str]) -> dict[str, int]:
    """True test-graph loopless degree quotas, aligned to `node_ids` (0 if absent)."""
    return {node: (int(g_simple.degree(node)) if node in g_simple else 0) for node in node_ids}


# --------------------------------------------------------------------------- arm topology


@dataclass(frozen=True)
class ArmTopology:
    """The five topology numbers plus edge-set and self-loop bookkeeping for one arm.

    Attributes:
        graph_similarity: ``{"global_simple_edge": ..., "bfs_macro": ...}`` --
            named separately per the CLAUDE.md claim rule; never aggregated
            with `mmd_ratio`.
        relative_density: Same two-key shape as `graph_similarity`.
        mmd_ratio: statistic -> BFS-macro reference-normalized MMD ratio
            (degree, clustering, spectral); no global-simple-edge analogue
            exists for these.
        edge_precision: ``|pred ∩ ref| / |pred|`` over simple (self-loop-
            stripped) edge sets; 0.0 when `pred` is empty.
        edge_recall: ``|pred ∩ ref| / |ref|`` over simple edge sets; 0.0 when
            `ref` is empty.
        self_loops_pred: Self-loop count in the assembled graph.
        self_loops_ref: Self-loop count in the reference graph.
    """

    graph_similarity: dict[str, float]
    relative_density: dict[str, float]
    mmd_ratio: dict[str, float]
    edge_precision: float
    edge_recall: float
    self_loops_pred: int
    self_loops_ref: int


def _edge_overlap_metrics(g_pred_simple: nx.Graph, g_ref_simple: nx.Graph) -> dict[str, float]:
    """Precision/recall of the assembled simple-edge set against the reference."""
    pred_edges = {frozenset(e) for e in g_pred_simple.edges()}
    ref_edges = {frozenset(e) for e in g_ref_simple.edges()}
    overlap = len(pred_edges & ref_edges)
    return {
        "edge_precision": overlap / len(pred_edges) if pred_edges else 0.0,
        "edge_recall": overlap / len(ref_edges) if ref_edges else 0.0,
    }


def evaluate_arm(
    graph: nx.Graph, reference: BucketReference, config: MMDConfig, g_ref_simple: nx.Graph
) -> ArmTopology:
    """Evaluate one assembled arm's graph into an `ArmTopology`.

    Args:
        graph: The assembled predicted graph (self-loops already applied).
        reference: Precomputed reference-side bucket state
            (`precompute_bucket_reference`), shared across every arm.
        config: Shared MMD/descriptor configuration.
        g_ref_simple: The self-loop-stripped reference graph.

    Returns:
        The `ArmTopology`.
    """
    bucketed = evaluate_assembled_graph_with_reference(graph, reference, config)
    g_pred_simple = strip_self_loops(graph)
    overlap = _edge_overlap_metrics(g_pred_simple, g_ref_simple)
    return ArmTopology(
        graph_similarity={
            "global_simple_edge": compute_graph_similarity(g_pred_simple, g_ref_simple),
            "bfs_macro": bucketed.graph_similarity,
        },
        relative_density={
            "global_simple_edge": compute_relative_density(g_pred_simple, g_ref_simple),
            "bfs_macro": bucketed.relative_density,
        },
        mmd_ratio=dict(bucketed.mmd_ratio),
        edge_precision=overlap["edge_precision"],
        edge_recall=overlap["edge_recall"],
        self_loops_pred=bucketed.self_loops_pred,
        self_loops_ref=bucketed.self_loops_ref,
    )


# --------------------------------------------------------------------------- quota bookkeeping


@dataclass(frozen=True)
class ArmQuotaInfo:
    """Hard-quota enforcement bookkeeping for one quota-based arm.

    Attributes:
        stats: `QuotaAssemblyStats` from the greedy quota-constrained pass.
        quota_error: `degree_quota_error(graph, quotas)` (l1, linf, exact_fraction).
        shortfall_fraction: ``stats.shortfall / stats.target_edges`` (0.0 when
            `target_edges` is 0).
        lower_bound_only: True when `shortfall_fraction` exceeds
            `_LOWER_BOUND_SHORTFALL_THRESHOLD` -- greedy stranding (no
            back-fill) left enough quota unconsumed that this arm's topology
            numbers are only a lower bound on what the quota profile could
            achieve, not its exact realization.
    """

    stats: QuotaAssemblyStats
    quota_error: dict[str, float]
    shortfall_fraction: float
    lower_bound_only: bool


def build_arm_quota_info(
    graph: nx.Graph, quotas: Mapping[str, int], stats: QuotaAssemblyStats
) -> ArmQuotaInfo:
    """Bundle one hard-quota arm's enforcement bookkeeping and lower-bound flag."""
    shortfall_fraction = stats.shortfall / stats.target_edges if stats.target_edges > 0 else 0.0
    return ArmQuotaInfo(
        stats=stats,
        quota_error=degree_quota_error(graph, quotas),
        shortfall_fraction=shortfall_fraction,
        lower_bound_only=shortfall_fraction > _LOWER_BOUND_SHORTFALL_THRESHOLD,
    )


# --------------------------------------------------------------------------- gap closure


def _closure(base: float, arm: float, target: float) -> float | None:
    """Fraction of the ``base -> target`` gap closed by ``arm``; ``None`` if undefined."""
    if not (np.isfinite(base) and np.isfinite(arm) and np.isfinite(target)):
        return None
    gap = base - target
    if gap == 0.0:
        return None
    return (base - arm) / gap


def compute_arm_gap_closure(
    control: ArmTopology, oracle: ArmTopology, arm: ArmTopology
) -> dict[str, float | None]:
    """Per-axis fraction of this run's own control -> oracle gap closed by `arm`.

    Five axes: the three BFS-macro MMD ratios plus BFS-macro GS and RD -- the
    claim-rule topology result, never the global-simple-edge variants. 0 = no
    closure, 1 = `arm` reaches `oracle` exactly, ``None`` when the control ->
    oracle gap on that axis is exactly zero or any input is non-finite.

    Args:
        control: The ``b0_exact_n`` row.
        oracle: The ``oracle_hard`` row.
        arm: One ``predicted_hard_<variant>`` row.

    Returns:
        ``{"degree" | "clustering" | "spectral" | "graph_similarity" |
        "relative_density": closure}``.
    """
    out: dict[str, float | None] = {}
    for stat in STATISTICS:
        out[stat] = _closure(control.mmd_ratio[stat], arm.mmd_ratio[stat], oracle.mmd_ratio[stat])
    out["graph_similarity"] = _closure(
        control.graph_similarity["bfs_macro"],
        arm.graph_similarity["bfs_macro"],
        oracle.graph_similarity["bfs_macro"],
    )
    out["relative_density"] = _closure(
        control.relative_density["bfs_macro"],
        arm.relative_density["bfs_macro"],
        oracle.relative_density["bfs_macro"],
    )
    return out


# --------------------------------------------------------------------------- pipeline


def run_s4_pipeline(
    *,
    universe_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
    degree_predictions_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Run the full S4 budget-assembly pipeline and write its outputs.

    Args:
        universe_path: Path to the frozen B0 candidate-universe scores artifact.
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Benchmark split strategy (e.g. ``"breadth_first"``).
        output_dir: Directory to write ``s4_results.json`` into.
        seed: Base seed; recorded for provenance (no randomized step in this
            pipeline consumes it).
        degree_predictions_paths: ``degree_predictions_v1`` JSON files, one
            ``predicted_hard_<variant>`` arm per file (variant = file stem).

    Returns:
        The JSON-ready results payload (also written to
        ``output_dir/s4_results.json``).

    Raises:
        ValueError: If the scores artifact fails validation, a predictions
            file fails its format/node-set contract, or two predictions files
            share the same stem (duplicate variant).
    """
    universe = load_scores(universe_path)

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    n_test_nodes = g_ref.number_of_nodes()

    validate_artifact_precision(universe, label="universe")
    validate_universe_artifact(
        universe, strategy=strategy, n_test_nodes=n_test_nodes, label="universe"
    )

    g_simple = strip_self_loops(g_ref)
    target_edges = g_simple.number_of_edges()
    buckets = load_test_node_buckets(benchmark_root, strategy)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = MMDConfig()

    nodes = list(g_ref.nodes())
    pairs = list(universe.pairs())
    probs = universe.probs()

    non_self_mask = universe.u_idx != universe.v_idx
    non_self_row_idx = np.flatnonzero(non_self_mask)
    non_self_pairs = [pairs[i] for i in non_self_row_idx.tolist()]
    ns_u = universe.u_idx[non_self_row_idx]
    ns_v = universe.v_idx[non_self_row_idx]
    ns_probs = probs[non_self_row_idx]

    # Frozen self-loop policy shared by every arm: self-pairs never enter
    # assemble_degree_quota (it raises on self-pair rows); they assemble as
    # self-loops at the B0 non-self density-matched operating point instead.
    threshold_b0 = density_matched_threshold(probs[non_self_mask], target_edges)
    self_edges = _self_loop_edges(universe, probs, threshold_b0)

    reference = precompute_bucket_reference(g_ref, buckets, config)

    logger.info("assembling the b0_exact_n control arm")
    control_graph = assemble_top_n_by_score(
        non_self_pairs, ns_u, ns_v, ns_probs, target_edges, nodes
    )
    control_graph.add_edges_from(self_edges)

    logger.info("assembling the oracle_hard arm")
    oracle_quotas = oracle_hard_quotas(g_simple, universe.node_ids)
    oracle_graph, oracle_stats = assemble_degree_quota(
        non_self_pairs, ns_u, ns_v, ns_probs, oracle_quotas, nodes
    )
    oracle_graph.add_edges_from(self_edges)

    arm_graphs: dict[str, nx.Graph] = {"b0_exact_n": control_graph, "oracle_hard": oracle_graph}
    arm_quota_info: dict[str, ArmQuotaInfo] = {
        "oracle_hard": build_arm_quota_info(oracle_graph, oracle_quotas, oracle_stats),
    }
    predicted_sources: dict[str, str] = {}

    for path in degree_predictions_paths:
        variant = path.stem
        arm_name = f"predicted_hard_{variant}"
        if arm_name in arm_graphs:
            raise ValueError(f"duplicate degree-predictions variant {variant!r} (from {path})")
        logger.info("assembling the %s arm", arm_name)
        raw_degrees = load_degree_predictions(path, universe.node_ids)
        quotas_arr = predicted_hard_quotas(raw_degrees, target_edges)
        quotas = {
            node: int(q) for node, q in zip(universe.node_ids, quotas_arr.tolist(), strict=True)
        }
        graph, stats = assemble_degree_quota(non_self_pairs, ns_u, ns_v, ns_probs, quotas, nodes)
        graph.add_edges_from(self_edges)
        arm_graphs[arm_name] = graph
        arm_quota_info[arm_name] = build_arm_quota_info(graph, quotas, stats)
        predicted_sources[variant] = str(path)

    logger.info("evaluating %d arms", len(arm_graphs))
    arm_topology: dict[str, ArmTopology] = {
        arm: evaluate_arm(graph, reference, config, g_simple) for arm, graph in arm_graphs.items()
    }

    gap_closure: dict[str, dict[str, float | None]] = {
        f"predicted_hard_{variant}": compute_arm_gap_closure(
            arm_topology["b0_exact_n"],
            arm_topology["oracle_hard"],
            arm_topology[f"predicted_hard_{variant}"],
        )
        for variant in predicted_sources
    }

    assembled: dict[str, object] = {
        arm: dataclasses.asdict(topo) for arm, topo in arm_topology.items()
    }
    quota_info: dict[str, object] = {
        arm: dataclasses.asdict(info) for arm, info in arm_quota_info.items()
    }

    metadata: dict[str, object] = {
        "artifacts": {"universe": _artifact_meta_summary(universe.meta)},
        "benchmark": {"benchmark_a": strategy},
        "mmd_config": dataclasses.asdict(config),
        "metric_normalization": "ratio_of_size_mean_mmd2",
        "reference_split": "artifact_order_even_vs_odd_within_each_node_size",
        "graph_similarity_definition": (
            "1 - L1(A_pred - A_ref) / (sum(A_pred) + sum(A_ref)) == edge-set Dice/F1; "
            "global_simple_edge is the whole assembled graph (self-loops stripped), "
            "bfs_macro is the official per-subgraph macro mean over benchmark buckets "
            "(self-loops retained), named separately per the claim rule"
        ),
        "self_loop_policy": (
            "self-pairs are never routed into assemble_degree_quota (it raises on "
            "self-pair rows); every arm instead shares one frozen self-loop set "
            "computed once at threshold_b0 = density_matched_threshold(probs[non_self], "
            "target_edges), via src.experiments.b0_cal._self_loop_edges"
        ),
        "quota_scaling_policy": (
            "predicted_hard_<variant>: raw predicted degrees clipped to >= "
            f"{_MIN_PREDICTED_DEGREE}, rescaled so their sum is exactly 2 * target_edges, "
            "then rounded via largest_remainder_quotas (preserves the integer sum)"
        ),
        "gap_closure_definition": (
            "per axis: (control - arm) / (control - oracle), where control = "
            "this run's own b0_exact_n row and oracle = this run's own oracle_hard "
            "row; null where that gap is exactly zero"
        ),
        "lower_bound_only_threshold": _LOWER_BOUND_SHORTFALL_THRESHOLD,
        "target_edges": target_edges,
        "seed": seed,
        "degree_predictions_sources": predicted_sources,
        "arms": sorted(arm_graphs),
    }

    payload: dict[str, object] = {
        "metadata": metadata,
        "assembled": assembled,
        "quota_info": quota_info,
        "gap_closure": gap_closure,
    }

    (output_dir / "s4_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote S4 budget-assembly results to %s", output_dir)
    return payload


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``s4_budget_assembly`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.s4_budget_assembly",
        description=(
            "S4 budget-assembly diagnostic: hard degree-quota replay of frozen B0 scores."
        ),
    )
    parser.add_argument("--universe", type=Path, required=True, help="B0 candidate-universe .npz")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--degree-predictions",
        dest="degree_predictions",
        type=Path,
        action="append",
        default=[],
        help=(
            "degree_predictions_v1 JSON file; repeatable, one predicted_hard_<variant> "
            "arm per file (variant = file stem)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors (via ``parser.error``, missing input files).
        ValueError: If the scores artifact or a predictions file fails validation.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.universe.exists():
        parser.error(f"--universe not found: {args.universe}")
    for path in args.degree_predictions:
        if not path.exists():
            parser.error(f"--degree-predictions not found: {path}")

    run_s4_pipeline(
        universe_path=args.universe,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
        degree_predictions_paths=args.degree_predictions,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
