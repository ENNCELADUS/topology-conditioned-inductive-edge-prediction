r"""Baseline ``B0+cal`` kill-test: calibrated assembly of the cached B0 scores.

Implements the protocol Sec 2 ``B0+cal`` baseline (temperature/Platt calibration +
density- and degree-sequence-matched thresholding of the frozen B0 candidate scores)
as a pre-Stage-1 kill-test for pre-registered decision rule Sec 5.2.1: if trivial
assembly calibration alone closes most of the pair-to-topology gap, the generative
apparatus is at direct risk and EgoStitch Stage 1 must not start before a
locked-decision discussion. No model scoring happens here — everything is a
row-selection, calibration, or graph computation over cached artifacts.

Calibration is strictly monotone, so rank metrics (AUROC/AUPRC, regime tables) are
identical to the frozen G1 B0 rows and are not recomputed here; only assembly- and
threshold-derived quantities move.

Arms (all from the same candidate artifact):

- ``b0``               — raw probs, density-matched threshold (G1 operating point).
- ``b0_cal_density``   — calibrated probs, density-matched threshold. Monotone ⇒
                         must reproduce ``b0``'s edge set exactly (sanity row).
- ``b0_cal_selfdensity`` — calibrated probs, exact top-N with
                         ``N = round(sum p_cal)`` (calibration moves the operating
                         point through the predicted edge volume).
- ``b0_cal_degseq``    — primary: greedy assembly under per-node degree quotas
                         rank-matched to the reference degree multiset.

CLI::

    python -m src.experiments.b0_cal \
        --universe scores/b0_v31_candidate.npz --val-scores scores/b0_v31_val.npz \
        --data-root data --strategy breadth_first --output-dir outputs/b0_cal \
        [--g3-results outputs/g3/g3_results.json] [--seed 0] \
        [--skip-perturbation-check]

Determinism: identical inputs produce a byte-identical ``b0cal_results.json``
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

from src.eval.assembly import (
    assemble_degree_quota,
    assemble_graph,
    density_matched_threshold,
    rank_matched_degree_quotas,
)
from src.eval.calibration import fit_platt, fit_temperature
from src.eval.composite import perturbation_check
from src.eval.edge_metrics import compute_edge_metrics
from src.eval.graph_metrics import STATISTICS, MMDConfig, noise_floor, strip_self_loops
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    AssembledRow,
    _artifact_meta_summary,
    _assembled_row_to_dict,
    assemble_and_evaluate,
    assemble_top_n_by_score,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.experiments.g3_oracle import compute_headroom
from src.score_universe import ScoresArtifact, load_scores

logger = logging.getLogger(__name__)

_CAL_ARMS: tuple[str, ...] = ("b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq")
_ALL_ARMS: tuple[str, ...] = ("b0",) + _CAL_ARMS

# Pre-registered kill-test reading threshold (docs/registrations/
# g5_stage1_preregistration.json; docs/03 Sec 5.2.1 de-risk): if the best
# calibrated arm closes >= 25% of the B0 -> oracle_blend clustering-MMD gap,
# halt before GPU spend for the locked-decision discussion.
_GAP_CLOSURE_HALT_THRESHOLD = 0.25


# --------------------------------------------------------------------------- validation


def validate_val_artifact(
    artifact: ScoresArtifact, *, strategy: str, universe_meta: dict[str, object]
) -> None:
    """Validate the calibration-fit artifact against the candidate universe.

    Args:
        artifact: The val scores artifact (calibration fit set).
        strategy: Expected benchmark strategy.
        universe_meta: The candidate universe's parsed metadata (checkpoint
            identity must match — both sides must come from the same frozen
            scorer).

    Raises:
        ValueError: On any provenance or label problem.
    """
    meta = artifact.meta
    if meta.get("pairs_source") != "val":
        raise ValueError(
            f"val artifact: pairs_source must be 'val', got {meta.get('pairs_source')!r}"
        )
    if meta.get("strategy") != strategy:
        raise ValueError(
            f"val artifact: strategy mismatch: expected {strategy!r}, got {meta.get('strategy')!r}"
        )
    if meta.get("checkpoint_id") != universe_meta.get("checkpoint_id"):
        raise ValueError(
            "val artifact: checkpoint_id mismatch vs universe: "
            f"{meta.get('checkpoint_id')!r} != {universe_meta.get('checkpoint_id')!r}"
        )
    labels = np.unique(artifact.label)
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError(
            f"val artifact: labels must all be in {{0, 1}}; got values {labels.tolist()}"
        )


# --------------------------------------------------------------------------- gap closure


def _closure(base: float, arm: float, target: float) -> float | None:
    """Fraction of the ``base -> target`` gap closed by ``arm``.

    ``None`` (a disclosed null) when the frozen gap is zero or any input is
    non-finite (e.g. an infinite relative density on a degenerate subgraph).
    """
    if not (np.isfinite(base) and np.isfinite(arm) and np.isfinite(target)):
        return None
    gap = base - target
    if gap == 0.0:
        return None
    return (base - arm) / gap


def compute_gap_closure(
    arm_row: AssembledRow, g3_assembled: dict[str, dict[str, object]]
) -> dict[str, float | None]:
    """Per-axis fraction of the B0 -> Oracle gap closed by one calibrated arm.

    MMD axes close toward ``oracle_blend`` (the all-MMD-improving G3 arm); GS and
    RD close toward ``oracle_topo`` (the GS/local-RD-improving G3 arm). 0 = no
    closure, 1 = the arm reaches the oracle reference, ``None`` = the frozen gap
    is exactly zero on that axis.

    Args:
        arm_row: The calibrated arm's assembled row.
        g3_assembled: The ``assembled`` section of a frozen ``g3_results.json``.

    Returns:
        ``{"degree" | "clustering" | "spectral" | "graph_similarity" |
        "relative_density": closure}``.
    """
    b0 = g3_assembled["b0"]
    blend = g3_assembled["oracle_blend"]
    topo = g3_assembled["oracle_topo"]
    b0_mmd = cast(dict[str, float], b0["mmd_ratio"])
    blend_mmd = cast(dict[str, float], blend["mmd_ratio"])

    out: dict[str, float | None] = {}
    for stat in STATISTICS:
        out[stat] = _closure(b0_mmd[stat], arm_row.mmd_ratio[stat], blend_mmd[stat])
    out["graph_similarity"] = _closure(
        cast(float, b0["graph_similarity"]),
        arm_row.graph_similarity,
        cast(float, topo["graph_similarity"]),
    )
    out["relative_density"] = _closure(
        cast(float, b0["relative_density"]),
        arm_row.relative_density,
        cast(float, topo["relative_density"]),
    )
    return out


# --------------------------------------------------------------------------- results assembly


@dataclass(frozen=True)
class B0CalResult:
    """The full, JSON-ready ``B0+cal`` kill-test payload.

    Attributes:
        metadata: Provenance, calibration fit disclosure, threshold policy,
            quota convention, seed, notes.
        noise_floor: Bucket size -> statistic -> mean noise-floor MMD^2.
        calibration: Fitted calibrator parameters and val ECE/Brier before/after.
        assembled: Arm key -> `AssembledRow`, JSON-encoded.
        headroom: Calibrated arm key -> `HeadroomRow` vs the ``b0`` arm.
        quota_stats: `QuotaAssemblyStats` of the ``b0_cal_degseq`` arm.
        gap_closure: Arm key -> per-axis B0->Oracle gap closure (``None`` when no
            ``--g3-results`` was supplied).
        decision: The pre-registered kill-test reading (``None`` without
            ``--g3-results``).
    """

    metadata: dict[str, object]
    noise_floor: dict[int, dict[str, float]]
    calibration: dict[str, object]
    assembled: dict[str, object]
    headroom: dict[str, object]
    quota_stats: dict[str, object]
    gap_closure: dict[str, object] | None
    decision: dict[str, object] | None

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict, `json.dumps`-ready representation."""
        return {
            "metadata": self.metadata,
            "noise_floor": {str(size): dict(stats) for size, stats in self.noise_floor.items()},
            "calibration": self.calibration,
            "assembled": self.assembled,
            "headroom": self.headroom,
            "quota_stats": self.quota_stats,
            "gap_closure": self.gap_closure,
            "decision": self.decision,
        }


def _self_loop_edges(
    universe: ScoresArtifact, probs: NDArray[np.float64], threshold: float
) -> list[tuple[str, str]]:
    """Self-pair rows clearing `threshold`, as ``(u, u)`` edges."""
    self_mask = universe.u_idx == universe.v_idx
    edges: list[tuple[str, str]] = []
    for i in np.flatnonzero(self_mask).tolist():
        if probs[i] >= threshold:
            node = universe.node_ids[int(universe.u_idx[i])]
            edges.append((node, node))
    return edges


def _edge_sets_identical(g_a: nx.Graph, g_b: nx.Graph) -> bool:
    """True iff the two graphs have exactly the same edge set (incl. self-loops)."""
    edges_a = {frozenset(e) if e[0] != e[1] else (e[0],) for e in g_a.edges()}
    edges_b = {frozenset(e) if e[0] != e[1] else (e[0],) for e in g_b.edges()}
    return edges_a == edges_b


# --------------------------------------------------------------------------- pipeline


def run_b0_cal_pipeline(
    *,
    universe_path: Path,
    val_scores_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
    g3_results_path: Path | None = None,
    skip_perturbation_check: bool = False,
) -> dict[str, object]:
    """Run the full ``B0+cal`` kill-test pipeline and write its outputs.

    Args:
        universe_path: Path to the frozen B0 candidate-universe scores artifact.
        val_scores_path: Path to the B0 val scores artifact (calibration fit set;
            same frozen checkpoint).
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Benchmark split strategy (e.g. ``"breadth_first"``).
        output_dir: Directory for ``b0cal_results.json`` and ``b0cal_tables.md``.
        seed: Base seed for every randomized step (bootstrap resampling).
        g3_results_path: Optional frozen ``g3_results.json``; enables the
            B0->Oracle gap-closure table and the pre-registered decision reading.
        skip_perturbation_check: Debug-only speed flag (G1/G3 semantics).

    Returns:
        The JSON-ready results payload (also written to
        ``output_dir/b0cal_results.json``).

    Raises:
        ValueError: If any artifact fails validation.
    """
    universe = load_scores(universe_path)
    val = load_scores(val_scores_path)

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    n_test_nodes = g_ref.number_of_nodes()

    validate_universe_artifact(
        universe, strategy=strategy, n_test_nodes=n_test_nodes, label="universe"
    )
    validate_val_artifact(val, strategy=strategy, universe_meta=universe.meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = MMDConfig()

    # ---- calibration fit (val artifact; temperature primary, Platt disclosed)
    val_logits = val.logit.astype(np.float64)
    val_labels = val.label
    temperature = fit_temperature(val_logits, val_labels)
    platt = fit_platt(val_logits, val_labels)
    val_labels64 = val_labels.astype(np.int64)
    val_before = compute_edge_metrics(val_labels64, val.probs())
    val_after = compute_edge_metrics(val_labels64, temperature.apply(val_logits))
    calibration: dict[str, object] = {
        "applied": "temperature",
        "fit_set": {
            "pairs_source": "val",
            "n_rows": int(val.label.size),
            "n_pos": int(np.sum(val.label == 1)),
            "note": (
                "fit on the balanced (~1:1) val pairs; the balanced-prior offset is "
                "absorbed by density-matched thresholding and disclosed as "
                "prior-sensitive for sum(p)-derived quantities (the selfdensity arm)"
            ),
        },
        "temperature": temperature.to_jsonable(),
        "platt": platt.to_jsonable(),
        "val_ece_before": val_before.ece,
        "val_ece_after": val_after.ece,
        "val_brier_before": val_before.brier,
        "val_brier_after": val_after.brier,
        "rank_metrics_note": (
            "calibration is strictly monotone: AUROC/AUPRC and every regime-table "
            "row equal the frozen G1 B0 values and are not recomputed here"
        ),
    }

    # ---- shared assembly ingredients
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

    g_simple = strip_self_loops(g_ref)
    target_edges = g_simple.number_of_edges()
    nodes = list(g_ref.nodes())
    pairs = list(universe.pairs())
    logits64 = universe.logit.astype(np.float64)
    probs_raw = universe.probs()
    probs_cal = temperature.apply(logits64)

    non_self_mask = universe.u_idx != universe.v_idx
    non_self_row_idx = np.flatnonzero(non_self_mask)
    non_self_pairs = [pairs[i] for i in non_self_row_idx.tolist()]
    ns_u = universe.u_idx[non_self_row_idx]
    ns_v = universe.v_idx[non_self_row_idx]

    # ---- arm: b0 (G1 operating-point convention)
    logger.info("assembling and evaluating the b0 arm")
    threshold_b0 = density_matched_threshold(probs_raw[non_self_mask], target_edges)
    b0_graph = assemble_graph(pairs, probs_raw, threshold=threshold_b0, nodes=nodes)
    b0_row = assemble_and_evaluate(
        g_pred=b0_graph,
        g_ref=g_ref,
        buckets=buckets,
        config=config,
        seed=seed,
        threshold=threshold_b0,
    )

    # ---- arm: b0_cal_density (monotonicity sanity row)
    threshold_cal = density_matched_threshold(probs_cal[non_self_mask], target_edges)
    cal_density_graph = assemble_graph(pairs, probs_cal, threshold=threshold_cal, nodes=nodes)
    density_identical = _edge_sets_identical(cal_density_graph, b0_graph)
    if density_identical:
        cal_density_row = dataclasses.replace(b0_row, threshold=threshold_cal)
    else:
        # Unreachable for a monotone calibrator; evaluated (not hidden) if it happens.
        logger.warning("b0_cal_density edge set differs from b0 — monotonicity violated?")
        cal_density_row = assemble_and_evaluate(
            g_pred=cal_density_graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=seed,
            threshold=threshold_cal,
        )

    # ---- arm: b0_cal_selfdensity (calibrated predicted edge volume)
    n_selfdensity = int(round(float(np.sum(probs_cal[non_self_mask]))))
    n_selfdensity = max(0, min(n_selfdensity, len(non_self_pairs)))
    logger.info("assembling the b0_cal_selfdensity arm (N=%d)", n_selfdensity)
    selfdensity_graph = assemble_top_n_by_score(
        non_self_pairs, ns_u, ns_v, probs_cal[non_self_row_idx], n_selfdensity, nodes
    )
    selfdensity_graph.add_edges_from(_self_loop_edges(universe, probs_cal, threshold_cal))
    selfdensity_row = assemble_and_evaluate(
        g_pred=selfdensity_graph,
        g_ref=g_ref,
        buckets=buckets,
        config=config,
        seed=seed,
        threshold=None,
    )

    # ---- arm: b0_cal_degseq (primary; rank-matched degree quotas)
    logger.info("assembling the b0_cal_degseq arm")
    expected = np.zeros(len(universe.node_ids), dtype=np.float64)
    np.add.at(expected, ns_u, probs_cal[non_self_row_idx])
    np.add.at(expected, ns_v, probs_cal[non_self_row_idx])
    expected_degree = {node: float(expected[i]) for i, node in enumerate(universe.node_ids)}
    reference_degrees = [
        int(g_simple.degree(node)) if node in g_simple else 0 for node in universe.node_ids
    ]
    quotas = rank_matched_degree_quotas(expected_degree, reference_degrees)
    degseq_graph, degseq_stats = assemble_degree_quota(
        non_self_pairs, ns_u, ns_v, probs_cal[non_self_row_idx], quotas, nodes
    )
    degseq_graph.add_edges_from(_self_loop_edges(universe, probs_cal, threshold_cal))
    degseq_row = assemble_and_evaluate(
        g_pred=degseq_graph,
        g_ref=g_ref,
        buckets=buckets,
        config=config,
        seed=seed,
        threshold=None,
    )

    arm_rows: dict[str, AssembledRow] = {
        "b0": b0_row,
        "b0_cal_density": cal_density_row,
        "b0_cal_selfdensity": selfdensity_row,
        "b0_cal_degseq": degseq_row,
    }
    assembled: dict[str, object] = {arm: _assembled_row_to_dict(arm_rows[arm]) for arm in _ALL_ARMS}
    headroom: dict[str, object] = {
        arm: dataclasses.asdict(compute_headroom(b0_row, arm_rows[arm])) for arm in _CAL_ARMS
    }

    # ---- gap closure + pre-registered decision reading (needs frozen G3 rows)
    gap_closure: dict[str, object] | None = None
    decision: dict[str, object] | None = None
    g3_b0_check: dict[str, object] | None = None
    if g3_results_path is not None:
        g3_payload = json.loads(g3_results_path.read_text(encoding="utf-8"))
        g3_assembled = cast(dict[str, dict[str, object]], g3_payload["assembled"])
        gap_closure = {
            arm: cast(dict[str, object], compute_gap_closure(arm_rows[arm], g3_assembled))
            for arm in _CAL_ARMS
        }
        g3_b0_mmd = cast(dict[str, float], g3_assembled["b0"]["mmd_ratio"])
        g3_b0_check = {
            "g3_b0_threshold": g3_assembled["b0"]["threshold"],
            "recomputed_b0_threshold": threshold_b0,
            "matches": bool(
                np.isclose(cast(float, g3_assembled["b0"]["threshold"]), threshold_b0, atol=1e-12)
                and all(
                    np.isclose(g3_b0_mmd[stat], b0_row.mmd_ratio[stat], atol=1e-9)
                    for stat in STATISTICS
                )
            ),
        }
        closures = {
            arm: cast(dict[str, float | None], gap_closure[arm])["clustering"] for arm in _CAL_ARMS
        }
        numeric = {arm: c for arm, c in closures.items() if c is not None}
        best_arm = max(numeric, key=lambda arm: numeric[arm]) if numeric else None
        best_closure = numeric[best_arm] if best_arm is not None else None
        halt = best_closure is not None and best_closure >= _GAP_CLOSURE_HALT_THRESHOLD
        decision = {
            "axis": "clustering",
            "halt_threshold": _GAP_CLOSURE_HALT_THRESHOLD,
            "clustering_gap_closure_per_arm": closures,
            "best_arm": best_arm,
            "best_closure": best_closure,
            "reading": ("halt_for_locked_decision_discussion" if halt else "proceed_to_stage1"),
            "note": (
                "pre-registered kill-test reading (docs/registrations/"
                "g5_stage1_preregistration.json): >= 25% closure of the "
                "B0->oracle_blend clustering-MMD gap by trivial calibration puts "
                "decision rule Sec 5.2.1 at direct risk before any GPU spend. "
                "Either way the Stage-1 bar is the best comparator per axis."
            ),
        }

    metadata: dict[str, object] = {
        "artifacts": {
            "universe": _artifact_meta_summary(universe.meta),
            "val": _artifact_meta_summary(val.meta),
            "g3_results": str(g3_results_path) if g3_results_path is not None else None,
        },
        "benchmark": {"benchmark_a": strategy},
        "mmd_config": dataclasses.asdict(config),
        "metric_normalization": "ratio_of_size_mean_mmd2",
        "reference_split": "artifact_order_even_vs_odd_within_each_node_size",
        "perturbation_check": perturbation_meta,
        "seed": seed,
        "threshold_policy": (
            "b0: density_matched_threshold(raw probs, target_edges = "
            "|E(strip_self_loops(test_graph))|) on non-self rows; self-pairs at the "
            "same threshold (G1 convention). b0_cal_density: identical policy on "
            "temperature-calibrated probs (monotone => same edge set; sanity row). "
            "b0_cal_selfdensity: exact top-N non-self pairs by calibrated probs "
            "with N = round(sum p_cal over non-self rows). b0_cal_degseq: greedy "
            "descending-score pass under rank-matched per-node degree quotas "
            "(reference degree multiset only; no node-identity degrees), no "
            "back-fill, shortfall disclosed. Calibrated arms assemble self-pairs "
            "at the calibrated density-matched threshold."
        ),
        "quota_convention": (
            "rank_matched_degree_quotas: nodes ranked by calibrated expected "
            "degree sum_v p_cal(u, v) desc (ties by node id asc), assigned the "
            "reference simple-degree multiset sorted desc, rank for rank"
        ),
        "gap_closure_definition": (
            "per axis: (value(B0) - value(arm)) / (value(B0) - value(oracle)); "
            "MMD axes close toward oracle_blend, GS/RD close toward oracle_topo; "
            "null where the frozen gap is zero"
        ),
        "g3_b0_cross_check": g3_b0_check,
        "notes": {
            "b0_cal_density_identical_edge_set": density_identical,
            "kill_test_role": (
                "runs before EgoStitch Stage-1 training (G5 plan phase 1); "
                "de-risks pre-registered decision rule Sec 5.2.1"
            ),
        },
    }

    result = B0CalResult(
        metadata=metadata,
        noise_floor=nf,
        calibration=calibration,
        assembled=assembled,
        headroom=headroom,
        quota_stats=dataclasses.asdict(degseq_stats),
        gap_closure=gap_closure,
        decision=decision,
    )
    payload = result.to_jsonable()

    (output_dir / "b0cal_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "b0cal_tables.md").write_text(render_tables_markdown(payload), encoding="utf-8")
    logger.info("wrote B0+cal results and tables to %s", output_dir)
    return payload


# --------------------------------------------------------------------------- markdown tables


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
    """Render the human-readable ``b0cal_tables.md`` companion for one payload.

    Args:
        payload: A `B0CalResult.to_jsonable` payload.

    Returns:
        The complete markdown document text.
    """
    calibration = cast(dict[str, object], payload["calibration"])
    assembled = cast(dict[str, dict[str, object]], payload["assembled"])
    headroom = cast(dict[str, dict[str, object]], payload["headroom"])
    quota_stats = cast(dict[str, object], payload["quota_stats"])
    temp = cast(dict[str, object], calibration["temperature"])
    platt = cast(dict[str, object], calibration["platt"])

    lines: list[str] = ["# B0+cal kill-test tables", "", "## Calibration fit (val)", ""]
    lines.append("| method | params | val NLL before | val NLL after | applied |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| temperature | T={_fmt(temp['temperature'])} | {_fmt(temp['val_nll_before'])} "
        f"| {_fmt(temp['val_nll_after'])} | {calibration['applied'] == 'temperature'} |"
    )
    lines.append(
        f"| platt | a={_fmt(platt['scale'])}, b={_fmt(platt['bias'])} "
        f"| {_fmt(platt['val_nll_before'])} | {_fmt(platt['val_nll_after'])} "
        f"| {calibration['applied'] == 'platt'} |"
    )
    lines.append("")
    lines.append(
        f"val ECE {_fmt(calibration['val_ece_before'])} -> {_fmt(calibration['val_ece_after'])}; "
        f"val Brier {_fmt(calibration['val_brier_before'])} -> "
        f"{_fmt(calibration['val_brier_after'])}"
    )

    lines += ["", "## Assembled-graph rows", ""]
    lines.append(
        "| arm | threshold | graph similarity | rel. density | degree MMD ratio | "
        "clustering MMD ratio | spectral MMD ratio | self-loops (pred/ref) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in _ALL_ARMS:
        row = assembled[arm]
        ratio = cast(dict[str, float], row["mmd_ratio"])
        lines.append(
            f"| {arm} | {_fmt(row['threshold'])} | {_fmt(row['graph_similarity'])} "
            f"| {_fmt(row['relative_density'])} "
            f"| {_fmt(ratio['degree'])} | {_fmt(ratio['clustering'])} | {_fmt(ratio['spectral'])} "
            f"| {row['self_loops_pred']}/{row['self_loops_ref']} |"
        )

    lines += ["", "## Headroom vs b0", ""]
    lines.append(
        "| arm | degree headroom | clustering headroom | spectral headroom "
        "| graph similarity ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for arm in _CAL_ARMS:
        ratios = cast(dict[str, object], headroom[arm]["mmd_ratio_headroom"])
        lines.append(
            f"| {arm} | {_fmt(ratios['degree'])} | {_fmt(ratios['clustering'])} "
            f"| {_fmt(ratios['spectral'])} | {_fmt(headroom[arm]['graph_similarity_ratio'])} |"
        )

    lines += ["", "## Degree-quota assembly (b0_cal_degseq)", ""]
    lines.append("| realized edges | target edges | shortfall | residual quota |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {quota_stats['realized_edges']} | {quota_stats['target_edges']} "
        f"| {quota_stats['shortfall']} | {quota_stats['residual_quota']} |"
    )

    gap_closure = cast(dict[str, dict[str, object]] | None, payload["gap_closure"])
    decision = cast(dict[str, object] | None, payload["decision"])
    if gap_closure is not None and decision is not None:
        lines += ["", "## B0 -> Oracle gap closure", ""]
        lines.append(
            "| arm | degree (vs blend) | clustering (vs blend) | spectral (vs blend) "
            "| GS (vs topo) | RD (vs topo) |"
        )
        lines.append("|---|---|---|---|---|---|")
        for arm in _CAL_ARMS:
            c = gap_closure[arm]
            lines.append(
                f"| {arm} | {_fmt(c['degree'])} | {_fmt(c['clustering'])} "
                f"| {_fmt(c['spectral'])} | {_fmt(c['graph_similarity'])} "
                f"| {_fmt(c['relative_density'])} |"
            )
        lines += [
            "",
            "## Decision reading",
            "",
            f"- best clustering closure: `{_fmt(decision['best_closure'])}` "
            f"(arm `{decision['best_arm']}`), halt threshold "
            f"`{_fmt(decision['halt_threshold'])}`",
            f"- **reading: `{decision['reading']}`**",
        ]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``b0_cal`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.b0_cal",
        description="B0+cal calibrated-assembly kill-test from cached scores artifacts.",
    )
    parser.add_argument("--universe", type=Path, required=True, help="B0 candidate-universe .npz")
    parser.add_argument("--val-scores", type=Path, required=True, help="B0 val scores .npz")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--g3-results",
        type=Path,
        default=None,
        help="frozen g3_results.json enabling the gap-closure table and decision reading",
    )
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
        ValueError: If any scores artifact fails validation.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.universe.exists():
        parser.error(f"--universe not found: {args.universe}")
    if not args.val_scores.exists():
        parser.error(f"--val-scores not found: {args.val_scores}")
    if args.g3_results is not None and not args.g3_results.exists():
        parser.error(f"--g3-results not found: {args.g3_results}")

    run_b0_cal_pipeline(
        universe_path=args.universe,
        val_scores_path=args.val_scores,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
        g3_results_path=args.g3_results,
        skip_perturbation_check=args.skip_perturbation_check,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
