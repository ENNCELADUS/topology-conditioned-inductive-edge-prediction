r"""Gate G3 (Oracle) analysis pipeline: cached scores + true neighborhoods -> headroom.

Compares the frozen B0 scorer's density-matched assembly against two
protocol-violating oracle reference arms built from TRUE held-out test-graph
neighborhoods (docs/03-experiment-protocol.md SS5.0 G3, pinned instantiation
2026-07-13): ``oracle_topo`` (common-neighbor count, ties by Adamic-Adar, then
canonical pair order) and ``oracle_blend`` (parameter-free rank fusion of the
B0 probabilities with ``oracle_topo``). Reports the stop-rule quantity: per
statistic, ``mmd_ratio(B0) / mmd_ratio(oracle arm)`` (headroom). No model
scoring happens here -- everything is a row-selection or graph computation
over the cached candidate-universe artifact and the benchmark package.

CLI::

    python -m src.experiments.g3_oracle \
        --universe scores/b0_v31_candidate.npz \
        --data-root data --strategy breadth_first \
        --output-dir outputs/g3 [--seed 0] [--skip-perturbation-check]

Determinism: identical inputs produce a byte-identical ``g3_results.json``
(stable sorts everywhere; ``json.dumps(..., sort_keys=True)``; no wall-clock
fields in the payload).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.features import FeatureStore
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.composite import perturbation_check
from src.eval.graph_metrics import STATISTICS, MMDConfig, noise_floor, strip_self_loops
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    _BREADTH_FIRST_ZERO_DROP_NOTE,
    _FEATURES_SUBDIR,
    _RATIOS,
    AssembledRow,
    _artifact_meta_summary,
    _assembled_row_to_dict,
    _edge_metrics_table_to_dict,
    _self_pair_edge_metrics_to_dict,
    assemble_and_evaluate,
    assemble_top_n_by_score,
    common_neighbor_and_adamic_adar,
    compute_self_pair_edge_metrics,
    degree_heterogeneity_sigma,
    evaluate_regime_table,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.score_universe import ScoresArtifact, load_scores

logger = logging.getLogger(__name__)

_ORACLE_ACCESS_NOTE = (
    "evaluator_side_oracle_reference (protocol-violating by design; G3 reference row)"
)

# --------------------------------------------------------------------------- rank scalarization


def _average_tie_rank01_from_order(
    order: NDArray[np.int64], new_run: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Average-tie normalized ranks in ``[0, 1]`` from a precomputed sort order.

    Args:
        order: Permutation sorting the values ascending (stable).
        new_run: Shape ``(n - 1,)`` booleans; ``True`` where the sorted value at
            position ``i + 1`` starts a new tie run.

    Returns:
        Shape ``(n,)`` float64 scores: rank / (n - 1), where tied values share
        the mean of the integer ranks they occupy (1 = largest key; all-tied
        input degenerates to a uniform 0.5, matching `normalize_min_max`'s
        uninformative-but-valid convention; a single row maps to 0.5).
    """
    n = order.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float64)
    run_id = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(new_run, dtype=np.int64)])
    counts = np.bincount(run_id)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)[:-1]])
    mean_ranks = starts.astype(np.float64) + (counts.astype(np.float64) - 1.0) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_ranks[run_id]
    return ranks / (n - 1.0)


def rank01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-tie normalized rank of `values` in ``[0, 1]`` (1 = largest).

    Deterministic (stable sort); idempotent on its own output. Rows with equal
    values share the mean of the integer ranks they occupy.

    Args:
        values: Shape ``(n,)`` float64 scores.

    Returns:
        Shape ``(n,)`` float64 normalized ranks.
    """
    order = np.argsort(values, kind="stable").astype(np.int64)
    sorted_values = values[order]
    new_run = cast(NDArray[np.bool_], sorted_values[1:] != sorted_values[:-1])
    return _average_tie_rank01_from_order(order, new_run)


def rank01_lex(primary: NDArray[np.float64], secondary: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-tie normalized rank of the ``(primary, secondary)`` lexicographic key.

    Rows are ranked ascending by `primary`, ties by `secondary`; rows equal on
    BOTH keys share the mean rank. This is the `oracle_topo` scalarization of
    the ``(CN, AA)`` key (docs/03 SS5.0 G3): a strictly monotone function of
    the pinned lexicographic ordering, so ranking by the returned scalar (with
    any downstream tie-break) reproduces that ordering exactly.

    Args:
        primary: Shape ``(n,)`` float64 primary key (common-neighbor counts).
        secondary: Shape ``(n,)`` float64 secondary key (Adamic-Adar), aligned.

    Returns:
        Shape ``(n,)`` float64 normalized ranks in ``[0, 1]`` (1 = most edge-like).
    """
    order = np.lexsort((secondary, primary)).astype(np.int64)
    p_sorted = primary[order]
    s_sorted = secondary[order]
    new_run = cast(
        NDArray[np.bool_],
        (p_sorted[1:] != p_sorted[:-1]) | (s_sorted[1:] != s_sorted[:-1]),
    )
    return _average_tie_rank01_from_order(order, new_run)


# --------------------------------------------------------------------------- headroom


@dataclass(frozen=True)
class HeadroomRow:
    """Stop-rule headroom of one oracle arm against the B0 assembled row.

    Attributes:
        mmd_ratio_headroom: Statistic -> ``b0.mmd_ratio / arm.mmd_ratio``
            (``None`` where the arm ratio is exactly 0 -- disclosed, never inf).
            Values >> 1 mean the oracle assembles a far more realistic graph
            than B0 (headroom exists); ~1 triggers the G3 stop discussion.
        graph_similarity_ratio: ``arm.graph_similarity / b0.graph_similarity``
            when B0's value is nonzero, else ``None``.
    """

    mmd_ratio_headroom: dict[str, float | None]
    graph_similarity_ratio: float | None


def compute_headroom(b0_row: AssembledRow, arm_row: AssembledRow) -> HeadroomRow:
    """Compute the G3 stop-rule headroom of one oracle arm against B0.

    Args:
        b0_row: The B0 density-matched assembled row.
        arm_row: One oracle arm's assembled row.

    Returns:
        The `HeadroomRow` (see its attribute docs for the exact definitions).
    """
    ratios: dict[str, float | None] = {}
    for stat in STATISTICS:
        arm_ratio = arm_row.mmd_ratio[stat]
        ratios[stat] = None if arm_ratio == 0.0 else b0_row.mmd_ratio[stat] / arm_ratio
    graph_similarity_ratio = (
        None
        if b0_row.graph_similarity == 0.0
        else arm_row.graph_similarity / b0_row.graph_similarity
    )
    return HeadroomRow(
        mmd_ratio_headroom=ratios,
        graph_similarity_ratio=graph_similarity_ratio,
    )


# --------------------------------------------------------------------------- oracle scores


def oracle_topo_scores(g_simple: nx.Graph, universe: ScoresArtifact) -> NDArray[np.float64]:
    """Row-aligned `oracle_topo` scalar scores for every universe row.

    Common-neighbor count on `g_simple` (primary), Adamic-Adar (secondary),
    scalarized as the average-tie normalized rank of the ``(CN, AA)`` key
    (docs/03 SS5.0 G3). Self-pair rows use the dense matrices' diagonals as-is
    (disclosed in metadata); they are excluded from top-N assembly pools by the
    caller, never here.

    Args:
        g_simple: The reference test graph with self-loops already stripped.
        universe: The validated candidate-universe scores artifact.

    Returns:
        Shape ``(n_rows,)`` float64 scores in ``[0, 1]`` (1 = most edge-like).
    """
    cn_matrix, aa_matrix = common_neighbor_and_adamic_adar(g_simple, universe.node_ids)
    cn_rows = cn_matrix[universe.u_idx, universe.v_idx]
    aa_rows = aa_matrix[universe.u_idx, universe.v_idx]
    return rank01_lex(cn_rows, aa_rows)


# --------------------------------------------------------------------------- results assembly


@dataclass(frozen=True)
class G3Result:
    """The full, JSON-ready G3 Oracle gate result payload.

    Attributes:
        metadata: Every disclosure required by the gate (artifact provenance,
            benchmark strategy, MMD and official GS/RD config, oracle formulas, threshold
            policy, regime construction, seed, notes).
        noise_floor: Bucket size -> statistic -> mean noise-floor MMD^2.
        regime_table: Scorer key (``"b0"``, ``"oracle_topo"``, ``"oracle_blend"``)
            -> regime edge-metric table.
        assembled: Scorer key -> `AssembledRow`, JSON-encoded.
        headroom: Oracle arm key -> `HeadroomRow`, JSON-encoded (the stop-rule
            view: ``mmd_ratio(B0) / mmd_ratio(arm)`` per statistic).
        self_pair_edge_metrics: Scorer key -> `SelfPairEdgeMetrics`, JSON-encoded.
        degree_heterogeneity_sigma: ``std(log(k))`` over ``k >= 1`` on the
            reference graph.
        positive_rate: Measured positive rate over the full candidate universe.
    """

    metadata: dict[str, object]
    noise_floor: dict[int, dict[str, float]]
    regime_table: dict[str, object]
    assembled: dict[str, object]
    headroom: dict[str, object]
    self_pair_edge_metrics: dict[str, object]
    degree_heterogeneity_sigma: float
    positive_rate: float

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict, `json.dumps`-ready representation."""
        return {
            "metadata": self.metadata,
            "noise_floor": {str(size): dict(stats) for size, stats in self.noise_floor.items()},
            "regime_table": self.regime_table,
            "assembled": self.assembled,
            "headroom": self.headroom,
            "self_pair_edge_metrics": self.self_pair_edge_metrics,
            "degree_heterogeneity_sigma": self.degree_heterogeneity_sigma,
            "positive_rate": self.positive_rate,
        }


# --------------------------------------------------------------------------- pipeline


def run_g3_pipeline(
    *,
    universe_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
    skip_perturbation_check: bool = False,
) -> dict[str, object]:
    """Run the full G3 Oracle gate pipeline and write its outputs.

    Args:
        universe_path: Path to the B0 candidate-universe scores artifact.
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Benchmark split strategy (e.g. ``"breadth_first"``).
        output_dir: Directory to write ``g3_results.json`` and ``g3_tables.md``
            into.
        seed: Base seed for every randomized step of the pipeline.
        skip_perturbation_check: Debug-only speed flag; identical semantics to
            G1's. Official GS/RD remain defined when the diagnostic is skipped.

    Returns:
        The JSON-ready results payload (also written to
        ``output_dir/g3_results.json``).

    Raises:
        ValueError: If the scores artifact fails validation.
    """
    universe = load_scores(universe_path)

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    n_test_nodes = g_ref.number_of_nodes()

    validate_universe_artifact(
        universe, strategy=strategy, n_test_nodes=n_test_nodes, label="universe"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = MMDConfig()

    logger.info("computing real-vs-real noise floor (seed=%d)", seed)
    nf = noise_floor(g_ref, buckets, config)
    perturbation_meta: dict[str, object]
    if skip_perturbation_check:
        perturbation_meta = {"skipped": True, "passed": None, "failures": []}
    else:
        logger.info("running the O'Bray perturbation check (seed=%d)", seed)
        perturbation_result = perturbation_check(g_ref, buckets, config, seed=seed)
        perturbation_meta = {
            "skipped": False,
            "passed": perturbation_result.passed,
            "failures": perturbation_result.failures,
            "similarities": perturbation_result.similarities,
            "fractions": list(perturbation_result.fractions),
        }

    store: FeatureStore | None = None
    features_root = data_root / _FEATURES_SUBDIR
    if (features_root / "index.json").exists():
        store = FeatureStore(features_root)
    f0_cache = output_dir / "f0_cache.pt"

    g_simple = strip_self_loops(g_ref)
    probs = universe.probs()
    s_topo = oracle_topo_scores(g_simple, universe)
    # s_topo is already an average-tie normalized rank, so rank01(s_topo) == s_topo;
    # only the B0 probabilities need ranking before fusing.
    s_blend = 0.5 * rank01(probs) + 0.5 * s_topo

    scorer_probs: dict[str, NDArray[np.float64]] = {
        "b0": probs,
        "oracle_topo": s_topo,
        "oracle_blend": s_blend,
    }

    regime_table: dict[str, object] = {}
    self_pair_metrics: dict[str, object] = {}
    for scorer, scorer_prob in scorer_probs.items():
        logger.info("building the %s regime table (seed=%d)", scorer, seed)
        regime_table[scorer] = _edge_metrics_table_to_dict(
            evaluate_regime_table(
                labels=universe.label,
                probs=scorer_prob,
                u_idx=universe.u_idx,
                v_idx=universe.v_idx,
                node_ids=universe.node_ids,
                g_ref=g_ref,
                store=store,
                f0_cache=f0_cache,
                seed=seed,
            )
        )
        self_pair_metrics[scorer] = _self_pair_edge_metrics_to_dict(
            compute_self_pair_edge_metrics(
                universe.label, scorer_prob, universe.u_idx, universe.v_idx
            )
        )

    nodes = list(g_ref.nodes())
    pairs = list(universe.pairs())
    target_edges = g_simple.number_of_edges()
    non_self_mask = universe.u_idx != universe.v_idx
    non_self_row_idx = np.flatnonzero(non_self_mask)
    non_self_pairs = [pairs[i] for i in non_self_row_idx.tolist()]

    # B0: density-matched threshold on non-self rows (G1's operating-point
    # convention); self-pairs assemble at the same threshold, outside the quota.
    b0_threshold = density_matched_threshold(probs[non_self_mask], target_edges)
    b0_graph = assemble_graph(pairs, probs, threshold=b0_threshold, nodes=nodes)
    b0_row = assemble_and_evaluate(
        g_pred=b0_graph,
        g_ref=g_ref,
        buckets=buckets,
        config=config,
        seed=seed,
        threshold=b0_threshold,
    )

    # Oracle arms: PA-null convention -- exact top-target_edges among non-self
    # pairs only, deterministic tie-break, no threshold.
    arm_rows: dict[str, AssembledRow] = {}
    for arm in ("oracle_topo", "oracle_blend"):
        logger.info("assembling and evaluating the %s row", arm)
        arm_graph = assemble_top_n_by_score(
            non_self_pairs,
            universe.u_idx[non_self_row_idx],
            universe.v_idx[non_self_row_idx],
            scorer_probs[arm][non_self_row_idx],
            target_edges,
            nodes,
        )
        arm_rows[arm] = assemble_and_evaluate(
            g_pred=arm_graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=seed,
            threshold=None,
        )

    assembled: dict[str, object] = {
        "b0": _assembled_row_to_dict(b0_row),
        "oracle_topo": _assembled_row_to_dict(arm_rows["oracle_topo"]),
        "oracle_blend": _assembled_row_to_dict(arm_rows["oracle_blend"]),
    }
    headroom: dict[str, object] = {
        arm: dataclasses.asdict(compute_headroom(b0_row, arm_rows[arm]))
        for arm in ("oracle_topo", "oracle_blend")
    }

    n_pos = int(np.sum(universe.label == 1))
    n_neg = int(np.sum(universe.label == 0))
    positive_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
    sigma = degree_heterogeneity_sigma(g_ref)

    metadata: dict[str, object] = {
        "artifacts": {"b0": _artifact_meta_summary(universe.meta)},
        "benchmark": {"benchmark_a": strategy},
        "mmd_config": dataclasses.asdict(config),
        "metric_normalization": "ratio_of_size_mean_mmd2",
        "reference_split": "artifact_order_even_vs_odd_within_each_node_size",
        "canonical_metric": "mmd_ratio",
        "component_disclosure": ["raw_mmd2", "reference_mmd2", "mmd_ratio"],
        "graph_similarity": {
            "formula": "1 - L1(A_pred - A_ref) / (sum(A_pred) + sum(A_ref))",
            "aggregation": "unweighted macro mean over every fixed sampled subgraph",
            "self_loops": "retained",
            "empty_empty": 1.0,
        },
        "relative_density": {
            "formula": "density(G_pred) / density(G_ref)",
            "aggregation": "unweighted macro mean over every fixed sampled subgraph",
            "self_loops": "retained",
            "empty_empty": 1.0,
            "nonempty_over_empty": "inf",
        },
        "perturbation_check": perturbation_meta,
        "threshold_policy": (
            "B0 row: operating point = density_matched_threshold(own probs, "
            "target_edges = |E(strip_self_loops(test_graph))|) on non-self-pair rows; "
            "self-pairs assemble at the same threshold and never count toward the "
            "quota (G1 convention). Oracle arms (oracle_topo, oracle_blend) instead "
            "take the exact top-target_edges pairs by score among non-self pairs "
            "only, deterministic tie-break by canonical pair order, no threshold "
            "(PA-null convention); self-pairs are excluded from oracle top-N "
            "candidate pools entirely."
        ),
        "seed": seed,
        "regime_construction": {
            "easy_uniform": "uniform random sample of label-0 rows, without replacement.",
            "degree_corrected": (
                "label-0 rows sampled without replacement with probability proportional to "
                "(deg_u + 1) * (deg_v + 1) (2405.14985), via the Efraimidis-Spirakis "
                "weighted-sampling-without-replacement algorithm; degrees from "
                "strip_self_loops(test_graph)."
            ),
            "hard_heuristic": (
                "top-K label-0 rows by common-neighbor count on strip_self_loops(test_graph), "
                "ties broken by Adamic-Adar, remaining ties by canonical pair order; "
                "K = ratio * n_positives."
            ),
            "hard_feature": (
                "top-K label-0 rows by cosine similarity of F0 mean-pooled features, ties "
                "broken by canonical pair order; K = ratio * n_positives."
            ),
            "full_universe": "all labeled rows, no sampling (the imbalance view).",
            "ratios": list(_RATIOS),
        },
        "oracle": {
            "access": _ORACLE_ACCESS_NOTE,
            "oracle_topo": {
                "ordering": (
                    "common-neighbor count desc on strip_self_loops(test_graph), ties by "
                    "Adamic-Adar desc, remaining ties by canonical pair order asc "
                    "(docs/03 SS5.0 G3 pinned instantiation)"
                ),
                "scalarization": (
                    "average-tie normalized rank of the (CN, AA) key over all universe "
                    "rows, in [0, 1]"
                ),
            },
            "oracle_blend": {
                "formula": (
                    "0.5 * rank01(p_B0) + 0.5 * rank01(s_topo); rank01 = average-tie "
                    "normalized rank in [0, 1] (s_topo is already such a rank, so "
                    "rank01(s_topo) == s_topo)"
                ),
            },
        },
        "headroom_definition": (
            "per statistic: mmd_ratio(B0) / mmd_ratio(oracle arm), null where the arm "
            "ratio is 0; graph_similarity_ratio = GS(arm) / GS(B0), null when GS(B0) is zero"
        ),
        "notes": {
            "breadth_first_zero_drop": _BREADTH_FIRST_ZERO_DROP_NOTE,
            "hard_heuristic_degeneracy": (
                "hard_heuristic negatives are CN/AA-selected, so oracle_topo's "
                "hard_heuristic rows are degenerate by construction (disclosed, not "
                "hidden)."
            ),
            "self_pair_scores": (
                "oracle scores for self-pairs use the dense CN/AA matrix diagonals "
                "as-is; self-pairs are excluded from oracle top-N assembly pools "
                "(PA-null convention)."
            ),
        },
    }

    result = G3Result(
        metadata=metadata,
        noise_floor=nf,
        regime_table=regime_table,
        assembled=assembled,
        headroom=headroom,
        self_pair_edge_metrics=self_pair_metrics,
        degree_heterogeneity_sigma=sigma,
        positive_rate=positive_rate,
    )
    payload = result.to_jsonable()

    (output_dir / "g3_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "g3_tables.md").write_text(render_tables_markdown(payload), encoding="utf-8")
    logger.info("wrote G3 results and tables to %s", output_dir)
    return payload


# --------------------------------------------------------------------------- markdown tables

_SCORER_ORDER: tuple[str, ...] = ("b0", "oracle_topo", "oracle_blend")
_ARM_ORDER: tuple[str, ...] = ("oracle_topo", "oracle_blend")


def _fmt(value: object) -> str:
    """Format one table cell: 6 significant digits for numbers, '' for None."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def render_tables_markdown(payload: dict[str, object]) -> str:
    """Render the human-readable ``g3_tables.md`` companion for one payload.

    Args:
        payload: A `G3Result.to_jsonable` payload.

    Returns:
        The complete markdown document text.
    """
    regime_table = cast(dict[str, dict[str, dict[str, dict[str, object]]]], payload["regime_table"])
    assembled = cast(dict[str, dict[str, object]], payload["assembled"])
    headroom = cast(dict[str, dict[str, object]], payload["headroom"])
    noise = cast(dict[str, dict[str, float]], payload["noise_floor"])

    lines: list[str] = ["# G3 Oracle gate tables", "", "## Regime table", ""]
    lines.append("| scorer | regime | key | n_pos | n_neg | AUROC | AUPRC | MCC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for scorer in _SCORER_ORDER:
        for regime, keys in regime_table[scorer].items():
            for key, m in keys.items():
                lines.append(
                    f"| {scorer} | {regime} | {key} | {m['n_pos']} | {m['n_neg']} "
                    f"| {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | {_fmt(m['mcc'])} |"
                )

    lines += ["", "## Assembled-graph rows", ""]
    lines.append(
        "| scorer | threshold | graph similarity | rel. density | degree MMD ratio | "
        "clustering MMD ratio | spectral MMD ratio | self-loops (pred/ref) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for scorer in _SCORER_ORDER:
        row = assembled[scorer]
        ratio = cast(dict[str, float], row["mmd_ratio"])
        lines.append(
            f"| {scorer} | {_fmt(row['threshold'])} | {_fmt(row['graph_similarity'])} "
            f"| {_fmt(row['relative_density'])} "
            f"| {_fmt(ratio['degree'])} | {_fmt(ratio['clustering'])} | {_fmt(ratio['spectral'])} "
            f"| {row['self_loops_pred']}/{row['self_loops_ref']} |"
        )

    lines += ["", "## Headroom (stop-rule view)", ""]
    lines.append(
        "| oracle arm | degree headroom | clustering headroom | spectral headroom "
        "| graph similarity ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for arm in _ARM_ORDER:
        ratios = cast(dict[str, object], headroom[arm]["mmd_ratio_headroom"])
        lines.append(
            f"| {arm} | {_fmt(ratios['degree'])} | {_fmt(ratios['clustering'])} "
            f"| {_fmt(ratios['spectral'])} | {_fmt(headroom[arm]['graph_similarity_ratio'])} |"
        )

    lines += ["", "## MMD ratio components", ""]
    lines.append(
        "| scorer | statistic | raw numerator | reference denominator | normalized ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for scorer in _SCORER_ORDER:
        row = assembled[scorer]
        raw = cast(dict[str, float], row["raw_mmd2"])
        ref = cast(dict[str, float], row["reference_mmd2"])
        ratio = cast(dict[str, float], row["mmd_ratio"])
        for stat in STATISTICS:
            lines.append(
                f"| {scorer} | {stat} | {_fmt(raw[stat])} | {_fmt(ref[stat])} "
                f"| {_fmt(ratio[stat])} |"
            )

    lines += ["", "## Noise floor", ""]
    lines.append(
        "| bucket size | degree reference MMD2 | clustering reference MMD2 "
        "| spectral reference MMD2 |"
    )
    lines.append("|---|---|---|---|")
    for size in sorted(noise, key=int):
        stats = noise[size]
        lines.append(
            f"| {size} | {_fmt(stats['degree'])} | {_fmt(stats['clustering'])} "
            f"| {_fmt(stats['spectral'])} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``g3_oracle`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.g3_oracle",
        description="Gate G3 (Oracle) analysis pipeline from cached scores artifacts.",
    )
    parser.add_argument("--universe", type=Path, required=True, help="B0 candidate-universe .npz")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-perturbation-check",
        action="store_true",
        help=(
            "DEBUG-ONLY speed flag: skip the perturbation diagnostic; "
            "official GS/RD remain defined."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors (via ``parser.error``, missing input files).
        ValueError: If the scores artifact fails validation -- surfaced as an
            uncaught exception with a descriptive message, matching
            `src.experiments.g1_hardened_e2.main`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.universe.exists():
        parser.error(f"--universe not found: {args.universe}")

    run_g3_pipeline(
        universe_path=args.universe,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_perturbation_check=args.skip_perturbation_check,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
