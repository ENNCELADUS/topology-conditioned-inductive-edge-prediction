r"""G5 Stage-1 gate evaluation: pre-registered EgoStitch-vs-comparator decision.

Evaluates either one seed as an explicitly non-binding topology diagnostic or
the three pre-registered EgoStitch Stage-1 seeds as the formal gate against the
frozen comparators (B0 recomputed from its cached artifact; the B0+cal arms
from the committed kill-test payload) under the pre-registered criteria
(docs/registrations/g5_stage1_preregistration.json; protocol Sec 5.0.5/5.2):

- **Enforcement first**: the pre-registration file's sha256 must equal the
  ``preregistration_sha256`` recorded in every training run's
  ``run_metadata.json`` — held-out metrics are never opened on a mismatch.
- **Primary family** (Holm at the pre-registered alpha, all must pass):
  clustering-MMD ratio at the canonical G1 operating point; BFS-macro GS and
  RD at matched global simple-edge RD (per-comparator deterministic exact-quota
  re-assembly).
- **Guards**: degree-MMD non-regression (<= 1.10x B0); matched edge AUPRC
  (degree-corrected ratio-1, within 0.02 of B0).
- **Verdict**: one seed always yields ``"diagnostic_only"``; exactly three
  seeds yield ``"pass"`` or ``"cut"`` and, on cut, the pre-registered failure
  reading is written verbatim — criteria are never edited post hoc.

CLI::

    python -m src.experiments.g5_stage1 \
        --egostitch-universe s0.npz s1.npz s2.npz \
        --run-metadata run0/run_metadata.json run1/... run2/... \
        --b0-universe scores/b0_v31_candidate.npz \
        --b0cal-results outputs/b0_cal/b0cal_results.json \
        --preregistration docs/registrations/g5_stage1_preregistration.json \
        --data-root data --strategy breadth_first --output-dir outputs/g5_stage1

Determinism: identical inputs produce byte-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.features import FeatureStore
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import MMDConfig, evaluate_assembled_graph, strip_self_loops
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    _FEATURES_SUBDIR,
    AssembledRow,
    _assembled_row_to_dict,
    _edge_metrics_table_to_dict,
    _self_pair_edge_metrics_to_dict,
    assemble_and_evaluate,
    compute_self_pair_edge_metrics,
    evaluate_regime_table,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.score_universe import load_scores

logger = logging.getLogger(__name__)

_COMPARATORS: tuple[str, ...] = ("b0", "b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq")
_PRIMARY_FAMILY: tuple[str, ...] = ("clustering_mmd_ratio", "bfs_macro_gs", "bfs_macro_rd")
_Z_95 = 1.959963984540054
_SE_FLOOR = 1e-12


class PreregistrationMismatch(RuntimeError):
    """Raised before any held-out metric is touched (prereg mechanics)."""


@dataclass(frozen=True)
class MatchedRdSelection:
    """Registered exact-quota selection for one matched-global-RD row."""

    selected_rows: NDArray[np.int64]
    boundary_score: float
    realized_edges: int
    rd_gap: float
    boundary_tie_size: int
    selected_from_boundary_tie: int

    @property
    def split_boundary_tie(self) -> bool:
        """Whether the quota includes only part of the boundary-score tie."""
        return 0 < self.selected_from_boundary_tie < self.boundary_tie_size


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def enforce_preregistration(preregistration_path: Path, run_metadata_paths: Sequence[Path]) -> str:
    """Refuse to open held-out metrics unless every run pinned this prereg file.

    Args:
        preregistration_path: The committed pre-registration JSON.
        run_metadata_paths: One ``run_metadata.json`` per training seed.

    Returns:
        The pre-registration file's sha256 (recorded in the gate payload).

    Raises:
        PreregistrationMismatch: On any absent or non-matching hash.
    """
    expected = _sha256_file(preregistration_path)
    for path in run_metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        recorded = metadata.get("preregistration_sha256")
        if recorded != expected:
            raise PreregistrationMismatch(
                f"{path}: preregistration_sha256 {recorded!r} does not match "
                f"{preregistration_path} ({expected}); held-out metrics stay closed "
                "(protocol Sec 5.2.4)"
            )
    return expected


def _registered_path(preregistration_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    try:
        repo_root = preregistration_path.resolve().parents[2]
    except IndexError:
        repo_root = Path.cwd()
    return repo_root / path


def enforce_frozen_inputs(
    prereg: dict[str, object], preregistration_path: Path, b0_universe_path: Path
) -> None:
    """Verify every registered frozen input before opening held-out scores."""
    frozen = cast(dict[str, object] | None, prereg.get("frozen_inputs"))
    if frozen is None:
        raise PreregistrationMismatch("preregistration has no frozen_inputs binding")
    required = ("b0_candidate_scores", "g1_results", "g3_results")
    for name in required:
        entry = cast(dict[str, object] | None, frozen.get(name))
        if entry is None or "path" not in entry or "sha256" not in entry:
            raise PreregistrationMismatch(f"frozen input {name!r} is incompletely registered")
        path = (
            b0_universe_path
            if name == "b0_candidate_scores"
            else _registered_path(preregistration_path, entry["path"])
        )
        if not path.is_file():
            raise PreregistrationMismatch(f"frozen input {name!r} not found: {path}")
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise PreregistrationMismatch(
                f"frozen input {name!r} sha256 mismatch: {actual} != {entry['sha256']}"
            )


def select_matched_global_rd_rows(
    probs: np.ndarray,
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    *,
    target_edges: int,
    reference_edges: int,
    tolerance: float = 0.005,
) -> MatchedRdSelection:
    """Select an exact non-self quota with the registered canonical tie-break."""
    if not (len(probs) == len(u_idx) == len(v_idx)):
        raise ValueError("probs, u_idx, and v_idx must have identical lengths")
    non_self_rows = np.flatnonzero(u_idx != v_idx).astype(np.int64, copy=False)
    if not 0 <= target_edges <= len(non_self_rows):
        raise ValueError(
            f"matched global simple-edge quota {target_edges} is outside [0, {len(non_self_rows)}]"
        )

    non_self_probs = probs[non_self_rows]
    canonical_u = np.minimum(u_idx[non_self_rows], v_idx[non_self_rows])
    canonical_v = np.maximum(u_idx[non_self_rows], v_idx[non_self_rows])
    order = np.lexsort((canonical_v, canonical_u, -non_self_probs))
    selected_rows = non_self_rows[order[:target_edges]]

    if target_edges == 0:
        boundary_score = math.inf
        boundary_tie_size = 0
        selected_from_boundary_tie = 0
    else:
        boundary_score = float(probs[selected_rows[-1]])
        boundary_tie_size = int(np.count_nonzero(non_self_probs == boundary_score))
        selected_from_boundary_tie = int(np.count_nonzero(probs[selected_rows] == boundary_score))

    realized = len(selected_rows)
    gap = abs(realized - target_edges) / reference_edges if reference_edges else 0.0
    if gap > tolerance:
        raise ValueError(
            "matched global simple-edge RD tolerance was violated after exact-quota "
            f"selection: gap={gap:.6g}, tolerance={tolerance:.6g}"
        )
    return MatchedRdSelection(
        selected_rows=selected_rows,
        boundary_score=boundary_score,
        realized_edges=realized,
        rd_gap=gap,
        boundary_tie_size=boundary_tie_size,
        selected_from_boundary_tie=selected_from_boundary_tie,
    )


def assemble_matched_global_rd_graph(
    pairs: Sequence[tuple[str, str]],
    probs: np.ndarray,
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    selection: MatchedRdSelection,
    nodes: Sequence[str],
) -> nx.Graph:
    """Assemble exact-quota non-self rows and self-pairs at the boundary score."""
    selected = np.zeros(len(pairs), dtype=np.bool_)
    selected[selection.selected_rows] = True
    selected |= (u_idx == v_idx) & (probs >= selection.boundary_score)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(pair for pair, keep in zip(pairs, selected.tolist(), strict=True) if keep)
    return graph


_FIDELITY_KEYS = {
    "degree_calibration_curve",
    "slot_recall_at_k_train",
    "slot_recall_at_k_test",
    "slot_adjacency_clustering_correlation",
    "s_channel_correlation",
    "self_nonself",
    "self_loop_rate",
    "proj_variance_trajectory",
}
_COST_KEYS = {"per_node_cached", "per_pair_marginal", "candidate_universe"}


def _load_required_diagnostics(
    fidelity_report_paths: Sequence[Path], cost_report_path: Path | None, n_runs: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(fidelity_report_paths) != n_runs:
        raise ValueError("one required fidelity report per egostitch universe is required")
    if cost_report_path is None:
        raise ValueError("the required cost report is missing")
    fidelity: list[dict[str, object]] = []
    for path in fidelity_report_paths:
        report = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        missing = sorted(_FIDELITY_KEYS - report.keys())
        if missing:
            raise ValueError(f"fidelity report {path} is missing keys: {missing}")
        fidelity.append(report)
    cost = cast(dict[str, object], json.loads(cost_report_path.read_text(encoding="utf-8")))
    missing_cost = sorted(_COST_KEYS - cost.keys())
    if missing_cost or cost.get("harmonization_rounds") != 0:
        raise ValueError(f"cost report must include {_COST_KEYS} and harmonization_rounds=0")
    for key in _COST_KEYS:
        row = cast(dict[str, object], cost[key])
        if "flops" not in row or "wall_seconds" not in row:
            raise ValueError(f"cost report {key!r} requires flops and wall_seconds")
    return fidelity, cost


def _validate_b0cal_lineage(
    payload: dict[str, object], b0_meta: dict[str, object], b0_row: AssembledRow
) -> None:
    metadata = cast(dict[str, object], payload.get("metadata"))
    artifacts = cast(dict[str, object], metadata.get("artifacts"))
    registered_meta = cast(dict[str, object], artifacts.get("universe"))
    for key in ("checkpoint_id", "model_family", "pairs_source", "strategy", "num_rows"):
        if registered_meta.get(key) != b0_meta.get(key):
            raise ValueError(
                f"b0cal lineage mismatch for {key}: "
                f"{registered_meta.get(key)!r} != {b0_meta.get(key)!r}"
            )
    b0cal_b0 = cast(dict[str, object], cast(dict[str, object], payload["assembled"])["b0"])
    recomputed = _assembled_row_to_dict(b0_row)
    for key in ("threshold", "graph_similarity", "relative_density"):
        if not np.isclose(float(cast(float, b0cal_b0[key])), float(cast(float, recomputed[key]))):
            raise ValueError(f"b0cal B0 row mismatch for {key}")
    registered_mmd = cast(dict[str, float], b0cal_b0["mmd_ratio"])
    recomputed_mmd = cast(dict[str, float], recomputed["mmd_ratio"])
    for stat in ("degree", "clustering", "spectral"):
        if not np.isclose(registered_mmd[stat], recomputed_mmd[stat]):
            raise ValueError(f"b0cal B0 row mismatch for mmd_ratio.{stat}")


# --------------------------------------------------------------------------- statistics


def _one_sided_p(mean: float, se: float) -> float:
    """One-sided normal p-value for ``H1: mean > 0`` (prereg procedure)."""
    z = mean / max(se, _SE_FLOOR)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def holm_step_down(p_values: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down over a family: member -> survives.

    Args:
        p_values: Family member -> nominal one-sided p-value.
        alpha: Family-wise error rate.

    Returns:
        Member -> ``True`` iff it survives (once a member fails, every larger
        p-value fails with it).
    """
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    m = len(ordered)
    survives: dict[str, bool] = {}
    failed = False
    for i, name in enumerate(ordered):
        threshold = alpha / (m - i)
        if failed or p_values[name] > threshold:
            failed = True
            survives[name] = False
        else:
            survives[name] = True
    return survives


@dataclass(frozen=True)
class CriterionResult:
    """One primary-family member's evaluation.

    Attributes:
        metric: The family member name.
        mean_diff: Mean per-seed improvement vs the best comparator
            (positive = improvement in the beneficial direction).
        se: The pre-registered standard error.
        p_value: One-sided nominal p-value.
        dominates_every_comparator: Strict mean dominance over every
            comparator (pre-registered requirement).
        ci_excludes_zero: ``mean - 1.96·se > 0`` (binding for the MMD member).
        best_comparator: The comparator setting the bar on this axis.
    """

    metric: str
    mean_diff: float
    se: float
    p_value: float
    dominates_every_comparator: bool
    ci_excludes_zero: bool
    best_comparator: str

    def to_jsonable(self) -> dict[str, object]:
        """Return a plain-dict representation."""
        return {
            "metric": self.metric,
            "mean_diff": self.mean_diff,
            "se": self.se,
            "p_value": self.p_value,
            "dominates_every_comparator": self.dominates_every_comparator,
            "ci_excludes_zero": self.ci_excludes_zero,
            "best_comparator": self.best_comparator,
        }


def _seed_variance(values: Sequence[float]) -> float:
    """Unbiased between-seed variance (0 for a single seed, disclosed)."""
    if len(values) < 2:
        return 0.0
    return float(np.var(np.asarray(values, dtype=np.float64), ddof=1))


def clustering_criterion(
    ego_rows: Sequence[AssembledRow], comparators: dict[str, dict[str, object]]
) -> CriterionResult:
    """The clustering-MMD member (canonical operating point, bootstrap-aware).

    ``d_s = mmd(best comparator) - mmd(ego, seed s)`` (positive = better);
    ``SE^2 = mean_s(var_boot(ego_s) + var_boot(best comp)) + var_s(d_s)/n``.

    Args:
        ego_rows: Canonical assembled rows, one per seed.
        comparators: Comparator name -> assembled-row JSON dict.

    Returns:
        The `CriterionResult`.
    """
    comp_values = {
        name: cast(dict[str, float], row["mmd_ratio"])["clustering"]
        for name, row in comparators.items()
    }
    best_name = min(comp_values, key=lambda name: (comp_values[name], name))
    best_value = comp_values[best_name]
    best_boot_var = (
        cast(dict[str, float], comparators[best_name]["bootstrap_std"])["clustering"] ** 2
    )

    diffs = [best_value - row.mmd_ratio["clustering"] for row in ego_rows]
    boot_vars = [row.bootstrap_std["clustering"] ** 2 + best_boot_var for row in ego_rows]
    n = len(ego_rows)
    se = math.sqrt(float(np.mean(boot_vars)) + _seed_variance(diffs) / n)
    mean_diff = float(np.mean(diffs))
    ego_mean = float(np.mean([row.mmd_ratio["clustering"] for row in ego_rows]))
    dominates = all(ego_mean < value for value in comp_values.values())
    return CriterionResult(
        metric="clustering_mmd_ratio",
        mean_diff=mean_diff,
        se=se,
        p_value=_one_sided_p(mean_diff, se),
        dominates_every_comparator=dominates,
        ci_excludes_zero=mean_diff - _Z_95 * se > 0.0,
        best_comparator=best_name,
    )


def matched_rd_criterion(
    metric: str,
    matched_values: dict[str, list[float]],
    comparator_values: dict[str, float],
) -> CriterionResult:
    """One matched-global-RD member (GS or BFS-macro RD; higher is better).

    Per comparator ``c``: ``d_s = value(ego matched to c, seed s) - value(c)``.
    The Holm p-value comes from the best (highest-valued) comparator; strict
    mean dominance is required against every comparator at its own matched
    operating point.

    Args:
        metric: ``"bfs_macro_gs"`` or ``"bfs_macro_rd"``.
        matched_values: Comparator name -> per-seed egostitch values at that
            comparator's matched operating point.
        comparator_values: Comparator name -> its own value.

    Returns:
        The `CriterionResult`.
    """
    best_name = max(comparator_values, key=lambda name: (comparator_values[name], name))
    diffs = [value - comparator_values[best_name] for value in matched_values[best_name]]
    n = len(diffs)
    se = math.sqrt(_seed_variance(diffs) / n) if n else _SE_FLOOR
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    dominates = all(
        float(np.mean(matched_values[name])) > comparator_values[name] for name in comparator_values
    )
    return CriterionResult(
        metric=metric,
        mean_diff=mean_diff,
        se=se,
        p_value=_one_sided_p(mean_diff, se),
        dominates_every_comparator=dominates,
        ci_excludes_zero=mean_diff - _Z_95 * se > 0.0,
        best_comparator=best_name,
    )


# --------------------------------------------------------------------------- pipeline


def run_g5_stage1_pipeline(
    *,
    egostitch_universe_paths: Sequence[Path],
    run_metadata_paths: Sequence[Path],
    b0_universe_path: Path,
    b0cal_results_path: Path,
    preregistration_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
    fidelity_report_paths: Sequence[Path] = (),
    cost_report_path: Path | None = None,
) -> dict[str, object]:
    """Run the pre-registered G5 Stage-1 gate and write its outputs.

    Args:
        egostitch_universe_paths: One candidate-scores ``.npz`` per training seed.
        run_metadata_paths: The matching ``run_metadata.json`` per seed
            (pre-registration binding), aligned with `egostitch_universe_paths`.
        b0_universe_path: The frozen B0 candidate-scores artifact.
        b0cal_results_path: The committed ``b0cal_results.json`` (comparator
            rows + realized edge counts).
        preregistration_path: The committed pre-registration JSON.
        data_root: Directory containing the benchmark package.
        strategy: Benchmark split strategy.
        output_dir: Directory for ``g5_stage1_results.json`` / ``_tables.md``.
        seed: Evaluation seed (bootstrap resampling only; training seeds live
            in the artifacts).
        fidelity_report_paths: Per-seed fidelity JSONs (`src.eval.ego_fidelity`
            outputs), embedded verbatim and required for the formal 3-seed gate.
        cost_report_path: FLOPs/wall-clock JSON (proposal Sec 4.7 R = 0
            commitment), embedded verbatim and required for the formal gate.

    Returns:
        The JSON-ready payload (also written to disk).

    Raises:
        PreregistrationMismatch: Before any metric is computed, on hash drift.
        ValueError: On artifact validation failures.
    """
    n_runs = len(egostitch_universe_paths)
    if n_runs != len(run_metadata_paths):
        raise ValueError("one --run-metadata per --egostitch-universe is required")
    if n_runs not in (1, 3):
        raise ValueError(
            "G5 Stage-1 evaluation requires either one seed for a non-binding "
            "diagnostic or exactly three seeds for the formal Holm verdict"
        )
    formal = n_runs == 3

    # (1) Pre-registration enforcement FIRST — no scores are opened before this.
    prereg_sha = enforce_preregistration(preregistration_path, run_metadata_paths)
    prereg = cast(dict[str, object], json.loads(preregistration_path.read_text(encoding="utf-8")))
    enforce_frozen_inputs(prereg, preregistration_path, b0_universe_path)
    if formal:
        fidelity, cost_report = _load_required_diagnostics(
            fidelity_report_paths, cost_report_path, n_runs
        )
    else:
        fidelity, cost_report = [], None
    run_metadata = [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in run_metadata_paths
    ]
    holm_alpha = float(
        cast(float, cast(dict[str, object], prereg["primary_criteria"]).get("holm_alpha", 0.05))
    )

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    g_simple = strip_self_loops(g_ref)
    target_edges = g_simple.number_of_edges()
    n_test_nodes = g_ref.number_of_nodes()
    nodes = list(g_ref.nodes())
    config = MMDConfig()

    store: FeatureStore | None = None
    features_root = data_root / _FEATURES_SUBDIR
    if (features_root / "index.json").exists():
        store = FeatureStore(features_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    f0_cache = output_dir / "f0_cache.pt"

    # (2) Comparators: B0 recomputed from its frozen artifact; cal arms from
    # the committed kill-test payload.
    b0_universe = load_scores(b0_universe_path)
    validate_universe_artifact(
        b0_universe, strategy=strategy, n_test_nodes=n_test_nodes, label="b0 universe"
    )
    frozen = cast(dict[str, object], prereg["frozen_inputs"])
    frozen_b0 = cast(dict[str, object], frozen["b0_candidate_scores"])
    if b0_universe.meta.get("checkpoint_id") != frozen_b0.get("checkpoint_id"):
        raise PreregistrationMismatch(
            "frozen input B0 checkpoint_id mismatch: "
            f"{b0_universe.meta.get('checkpoint_id')!r} != {frozen_b0.get('checkpoint_id')!r}"
        )
    b0_probs = b0_universe.probs()
    b0_pairs = list(b0_universe.pairs())
    b0_non_self = b0_universe.u_idx != b0_universe.v_idx
    b0_threshold = density_matched_threshold(b0_probs[b0_non_self], target_edges)
    b0_graph = assemble_graph(b0_pairs, b0_probs, threshold=b0_threshold, nodes=nodes)
    b0_row = assemble_and_evaluate(
        g_pred=b0_graph,
        g_ref=g_ref,
        buckets=buckets,
        config=config,
        seed=seed,
        threshold=b0_threshold,
    )
    b0_regimes = evaluate_regime_table(
        labels=b0_universe.label,
        probs=b0_probs,
        u_idx=b0_universe.u_idx,
        v_idx=b0_universe.v_idx,
        node_ids=b0_universe.node_ids,
        g_ref=g_ref,
        store=store,
        f0_cache=f0_cache,
        seed=seed,
    )
    b0_auprc = b0_regimes["degree_corrected"]["ratio_1"].auprc

    b0cal_payload = json.loads(b0cal_results_path.read_text(encoding="utf-8"))
    _validate_b0cal_lineage(cast(dict[str, object], b0cal_payload), b0_universe.meta, b0_row)
    b0cal_assembled = cast(dict[str, dict[str, object]], b0cal_payload["assembled"])
    realized = cast(
        dict[str, int],
        cast(dict[str, object], b0cal_payload["metadata"])["realized_non_self_edges"],
    )
    comparators: dict[str, dict[str, object]] = {
        name: dict(b0cal_assembled[name]) for name in _COMPARATORS if name != "b0"
    }
    comparators["b0"] = _assembled_row_to_dict(b0_row)
    realized = dict(realized)
    realized["b0"] = int(strip_self_loops(b0_graph).number_of_edges())

    # (3) Per-seed egostitch rows.
    ego_rows: list[AssembledRow] = []
    ego_regime_tables: list[dict[str, object]] = []
    ego_self_pair: list[dict[str, object]] = []
    ego_auprcs: list[float] = []
    s0_correlations: list[float] = []
    matched_gs: dict[str, list[float]] = {name: [] for name in _COMPARATORS}
    matched_rd: dict[str, list[float]] = {name: [] for name in _COMPARATORS}
    matched_rd_gap: dict[str, list[float]] = {name: [] for name in _COMPARATORS}
    matched_rd_boundary_score: dict[str, list[float]] = {name: [] for name in _COMPARATORS}
    matched_rd_boundary_tie_size: dict[str, list[int]] = {name: [] for name in _COMPARATORS}
    matched_rd_selected_from_tie: dict[str, list[int]] = {name: [] for name in _COMPARATORS}
    matched_rd_split_boundary_tie: dict[str, list[bool]] = {name: [] for name in _COMPARATORS}

    b0_logit_by_pair = {
        pair: float(logit) for pair, logit in zip(b0_pairs, b0_universe.logit.tolist(), strict=True)
    }

    for artifact_path, metadata in zip(egostitch_universe_paths, run_metadata, strict=True):
        universe = load_scores(artifact_path)
        validate_universe_artifact(
            universe, strategy=strategy, n_test_nodes=n_test_nodes, label=str(artifact_path)
        )
        if universe.meta.get("model_family") != "egostitch":
            raise ValueError(f"{artifact_path}: model_family must be 'egostitch'")
        if metadata.get("checkpoint_id") != universe.meta.get("checkpoint_id"):
            raise ValueError(f"{artifact_path}: run metadata checkpoint_id mismatch")
        if metadata.get("s0_checkpoint_id") != frozen_b0.get("checkpoint_id"):
            raise ValueError(f"{artifact_path}: run metadata s0_checkpoint_id mismatch")
        probs = universe.probs()
        pairs = list(universe.pairs())
        non_self = universe.u_idx != universe.v_idx

        threshold = density_matched_threshold(probs[non_self], target_edges)
        graph = assemble_graph(pairs, probs, threshold=threshold, nodes=nodes)
        ego_rows.append(
            assemble_and_evaluate(
                g_pred=graph,
                g_ref=g_ref,
                buckets=buckets,
                config=config,
                seed=seed,
                threshold=threshold,
            )
        )
        regimes = evaluate_regime_table(
            labels=universe.label,
            probs=probs,
            u_idx=universe.u_idx,
            v_idx=universe.v_idx,
            node_ids=universe.node_ids,
            g_ref=g_ref,
            store=store,
            f0_cache=f0_cache,
            seed=seed,
        )
        ego_regime_tables.append(cast(dict[str, object], _edge_metrics_table_to_dict(regimes)))
        ego_auprcs.append(regimes["degree_corrected"]["ratio_1"].auprc)
        ego_self_pair.append(
            _self_pair_edge_metrics_to_dict(
                compute_self_pair_edge_metrics(
                    universe.label, probs, universe.u_idx, universe.v_idx
                )
            )
        )
        aligned_b0 = np.array([b0_logit_by_pair[pair] for pair in pairs], dtype=np.float64)
        s0_correlations.append(
            float(np.corrcoef(universe.logit.astype(np.float64), aligned_b0)[0, 1])
        )

        # Matched-global-RD rows: realize each comparator's exact non-self
        # quota; only the boundary-score tie uses canonical pair order.
        for name in _COMPARATORS:
            quota = realized[name]
            selected = select_matched_global_rd_rows(
                probs,
                universe.u_idx,
                universe.v_idx,
                target_edges=quota,
                reference_edges=target_edges,
            )
            matched_graph = assemble_matched_global_rd_graph(
                pairs,
                probs,
                universe.u_idx,
                universe.v_idx,
                selected,
                nodes,
            )
            report = evaluate_assembled_graph(matched_graph, g_ref, buckets, config)
            matched_gs[name].append(report.graph_similarity)
            matched_rd[name].append(report.relative_density)
            realized_matched = strip_self_loops(matched_graph).number_of_edges()
            if realized_matched != selected.realized_edges:
                raise ValueError("matched global simple-edge RD count disagrees with assembly")
            matched_rd_gap[name].append(selected.rd_gap)
            matched_rd_boundary_score[name].append(selected.boundary_score)
            matched_rd_boundary_tie_size[name].append(selected.boundary_tie_size)
            matched_rd_selected_from_tie[name].append(selected.selected_from_boundary_tie)
            matched_rd_split_boundary_tie[name].append(selected.split_boundary_tie)

    # (4) Primary criteria + Holm.
    criteria = {
        "clustering_mmd_ratio": clustering_criterion(ego_rows, comparators),
        "bfs_macro_gs": matched_rd_criterion(
            "bfs_macro_gs",
            matched_gs,
            {name: cast(float, comparators[name]["graph_similarity"]) for name in _COMPARATORS},
        ),
        "bfs_macro_rd": matched_rd_criterion(
            "bfs_macro_rd",
            matched_rd,
            {name: cast(float, comparators[name]["relative_density"]) for name in _COMPARATORS},
        ),
    }
    holm: dict[str, bool | None]
    primary_pass: dict[str, bool | None]
    if formal:
        formal_holm = holm_step_down(
            {name: criteria[name].p_value for name in _PRIMARY_FAMILY}, holm_alpha
        )
        holm = dict(formal_holm)
        primary_pass = {
            name: bool(
                formal_holm[name]
                and criteria[name].dominates_every_comparator
                and (name != "clustering_mmd_ratio" or criteria[name].ci_excludes_zero)
            )
            for name in _PRIMARY_FAMILY
        }
    else:
        holm = dict.fromkeys(_PRIMARY_FAMILY)
        primary_pass = dict.fromkeys(_PRIMARY_FAMILY)

    # (5) Guards.
    ego_degree_mean = float(np.mean([row.mmd_ratio["degree"] for row in ego_rows]))
    degree_guard = ego_degree_mean <= 1.10 * b0_row.mmd_ratio["degree"]
    auprc_mean = float(np.mean(ego_auprcs))
    auprc_guard = auprc_mean >= b0_auprc - 0.02
    guards = {
        "degree_mmd_non_regression": {
            "passed": bool(degree_guard),
            "ego_mean": ego_degree_mean,
            "limit": 1.10 * b0_row.mmd_ratio["degree"],
        },
        "matched_edge_auprc": {
            "passed": bool(auprc_guard),
            "ego_mean": auprc_mean,
            "limit": b0_auprc - 0.02,
            "b0_auprc": b0_auprc,
        },
    }

    if formal:
        verdict = (
            "pass"
            if all(value is True for value in primary_pass.values())
            and degree_guard
            and auprc_guard
            else "cut"
        )
    else:
        verdict = "diagnostic_only"

    payload: dict[str, object] = {
        "metadata": {
            "preregistration_sha256": prereg_sha,
            "preregistration_path": str(preregistration_path),
            "benchmark": {"benchmark_a": strategy},
            "n_seeds": n_runs,
            "evaluation_mode": "formal_3seed" if formal else "single_seed_diagnostic",
            "binding_verdict": formal,
            "continuation_rule": (
                "This single-seed output is non-binding and is not a G5 pass/cut. "
                "Changing model or hyperparameters after inspecting it invalidates the "
                "registered three-seed experiment and requires a new experiment ID and "
                "pre-registration; with unchanged scientific configuration, seeds 1 and 2 "
                "may be completed for the formal verdict."
                if not formal
                else None
            ),
            "evaluation_seed": seed,
            "holm_alpha": holm_alpha,
            "matched_rd_rule": (
                "per comparator: egostitch realizes the comparator's exact non-self "
                "edge quota by descending pass-1 score; only the boundary-score tie "
                "is split by canonical pair order; self-pairs use the boundary score; "
                "every row enforces |RD_global(ego)-RD_global(comparator)| <= 0.005"
            ),
            "single_seed_caveat": (
                "Descriptive metric differences and p-values are shown only as an early "
                "signal. Holm decisions and primary pass flags are suppressed because the "
                "pre-registered criteria require three seeds."
                if not formal
                else None
            ),
            "s0_correlation_note": (
                "corr(egostitch logit, frozen-B0 logit) per seed; the full "
                "(s0, s1, s2) channel matrix is a training-side diagnostic "
                "reported from the worker's validation logs"
            ),
        },
        "comparators": comparators,
        "comparator_realized_non_self_edges": realized,
        "egostitch": {
            "assembled": [_assembled_row_to_dict(row) for row in ego_rows],
            "regime_tables": ego_regime_tables,
            "self_pair_edge_metrics": ego_self_pair,
            "degree_corrected_auprc": ego_auprcs,
            "s0_logit_correlation": s0_correlations,
            "matched_gs": matched_gs,
            "matched_rd": matched_rd,
            "matched_rd_quota_gap": matched_rd_gap,
            "matched_rd_boundary_score": matched_rd_boundary_score,
            "matched_rd_boundary_tie_size": matched_rd_boundary_tie_size,
            "matched_rd_selected_from_boundary_tie": matched_rd_selected_from_tie,
            "matched_rd_split_boundary_tie": matched_rd_split_boundary_tie,
        },
        "criteria": {name: criteria[name].to_jsonable() for name in _PRIMARY_FAMILY},
        "holm_survives": holm,
        "primary_pass": primary_pass,
        "guards": guards,
        "verdict": verdict,
        "failure_reading": (cast(str, prereg["failure_reading"]) if verdict == "cut" else None),
        "decision_rules_5_2_verbatim": prereg.get("decision_rules_5_2_verbatim"),
        "fidelity_reports": fidelity,
        "cost_report": cost_report,
    }

    (output_dir / "g5_stage1_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "g5_stage1_tables.md").write_text(
        render_tables_markdown(payload), encoding="utf-8"
    )
    logger.info("wrote G5 Stage-1 gate results to %s (verdict: %s)", output_dir, verdict)
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
    """Render the human-readable gate report for one payload.

    Args:
        payload: A `run_g5_stage1_pipeline` payload.

    Returns:
        The complete markdown document text.
    """
    comparators = cast(dict[str, dict[str, object]], payload["comparators"])
    ego = cast(dict[str, object], payload["egostitch"])
    criteria = cast(dict[str, dict[str, object]], payload["criteria"])
    holm = cast(dict[str, bool | None], payload["holm_survives"])
    primary_pass = cast(dict[str, bool | None], payload["primary_pass"])
    guards = cast(dict[str, dict[str, object]], payload["guards"])
    metadata = cast(dict[str, object], payload["metadata"])
    formal = bool(metadata["binding_verdict"])

    lines: list[str] = [
        "# G5 Stage-1 gate report" if formal else "# G5 Stage-1 single-seed topology diagnostic",
        "",
        f"**Verdict: `{payload['verdict']}`**",
        "",
    ]
    if not formal:
        lines += [
            "**NON-BINDING:** This is not a G5 pass/cut. The formal verdict requires ",
            "seeds 0, 1, and 2 plus the pre-registered Holm procedure.",
            "",
            str(metadata["continuation_rule"]),
            "",
        ]
    lines += [
        "## Assembled rows (canonical operating point)",
        "",
        "| arm | GS | BFS-macro RD | degree MMD | clustering MMD | spectral MMD |",
        "|---|---|---|---|---|---|",
    ]
    for name in _COMPARATORS:
        row = comparators[name]
        ratio = cast(dict[str, float], row["mmd_ratio"])
        lines.append(
            f"| {name} | {_fmt(row['graph_similarity'])} | {_fmt(row['relative_density'])} "
            f"| {_fmt(ratio['degree'])} | {_fmt(ratio['clustering'])} | {_fmt(ratio['spectral'])} |"
        )
    for i, row in enumerate(cast(list[dict[str, object]], ego["assembled"])):
        ratio = cast(dict[str, float], row["mmd_ratio"])
        lines.append(
            f"| egostitch (seed {i}) | {_fmt(row['graph_similarity'])} "
            f"| {_fmt(row['relative_density'])} "
            f"| {_fmt(ratio['degree'])} | {_fmt(ratio['clustering'])} | {_fmt(ratio['spectral'])} |"
        )

    lines += [
        "",
        "## Pre-registered decision table",
        "",
        "| criterion | mean diff | SE | p | Holm | dominance | CI excl. 0 | bar | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in _PRIMARY_FAMILY:
        c = criteria[name]
        lines.append(
            f"| {name} | {_fmt(c['mean_diff'])} | {_fmt(c['se'])} | {_fmt(c['p_value'])} "
            f"| {_fmt(holm[name])} | {c['dominates_every_comparator']} "
            f"| {c['ci_excludes_zero']} | {c['best_comparator']} "
            f"| **{_fmt(primary_pass[name])}** |"
        )

    lines += ["", "## Guards", "", "| guard | passed | ego mean | limit |", "|---|---|---|---|"]
    for name, guard in guards.items():
        lines.append(
            f"| {name} | {guard['passed']} | {_fmt(guard['ego_mean'])} | {_fmt(guard['limit'])} |"
        )

    if payload.get("failure_reading"):
        lines += ["", "## Pre-registered failure reading", "", str(payload["failure_reading"])]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``g5_stage1`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.g5_stage1",
        description="Pre-registered G5 Stage-1 gate evaluation.",
    )
    parser.add_argument(
        "--egostitch-universe",
        type=Path,
        nargs="+",
        required=True,
        help="one candidate-scores .npz per training seed",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        nargs="+",
        required=True,
        help="the matching run_metadata.json per seed (prereg binding)",
    )
    parser.add_argument("--b0-universe", type=Path, required=True)
    parser.add_argument("--b0cal-results", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fidelity-report",
        type=Path,
        nargs="+",
        default=(),
        help=(
            "required per-seed reports for the formal 3-seed gate; omitted for one-seed diagnostic"
        ),
    )
    parser.add_argument(
        "--cost-report",
        type=Path,
        default=None,
        help="required for the formal 3-seed gate; omitted for one-seed diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors.
        PreregistrationMismatch: Before any metric, on prereg hash drift.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in [
        *args.egostitch_universe,
        *args.run_metadata,
        args.b0_universe,
        args.b0cal_results,
        args.preregistration,
        *args.fidelity_report,
    ]:
        if not path.exists():
            parser.error(f"input not found: {path}")
    if args.cost_report is not None and not args.cost_report.exists():
        parser.error(f"input not found: {args.cost_report}")

    run_g5_stage1_pipeline(
        egostitch_universe_paths=args.egostitch_universe,
        run_metadata_paths=args.run_metadata,
        b0_universe_path=args.b0_universe,
        b0cal_results_path=args.b0cal_results,
        preregistration_path=args.preregistration,
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
        fidelity_report_paths=args.fidelity_report,
        cost_report_path=args.cost_report,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()


# Referenced for reuse by tests and companion tooling.
__all__ = [
    "CriterionResult",
    "PreregistrationMismatch",
    "clustering_criterion",
    "enforce_preregistration",
    "holm_step_down",
    "matched_rd_criterion",
    "render_tables_markdown",
    "run_g5_stage1_pipeline",
]
