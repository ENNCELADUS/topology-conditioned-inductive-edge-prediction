r"""G5 Stage-1 gate evaluation: registered EgoStitch-vs-comparator screen.

Evaluates one fixed EgoStitch Stage-1 seed as a binding engineering screening
gate against the frozen comparators (B0 recomputed from its cached artifact;
the B0+cal arms from the committed kill-test payload) under the registered criteria
(docs/registrations/g5_stage1_preregistration.json; protocol Sec 5.0.5/5.2):

- **Enforcement first**: the pre-registration file's sha256 must equal the
  ``preregistration_sha256`` recorded in every training run's
  ``run_metadata.json`` — held-out metrics are never opened on a mismatch.
- **Primary family** (single-seed point-estimate dominance, all must pass):
  clustering-MMD ratio at the canonical G1 operating point; BFS-macro GS and RD
  at matched global simple-edge RD (per-comparator deterministic exact-quota
  re-assembly). Inferential and Holm fields are not applicable at Stage 1.
- **Guards**: degree-MMD non-regression (<= 1.10x B0); matched edge AUPRC
  (degree-corrected ratio-1, within 0.02 of B0).
- **Verdict**: the fixed seed yields ``"pass"`` or ``"cut"`` and, on cut, the
  registered failure reading is written verbatim. This engineering screen does
  not establish statistical significance or cross-seed robustness.

CLI::

    python -m src.experiments.g5_stage1 \
        --egostitch-universe s0.npz \
        --s0-universe b0_fp32_candidate.npz \
        --run-metadata run0/run_metadata.json \
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
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.data.artifacts import load_candidate_pairs
from src.data.features import FeatureStore
from src.eval.assembly import assemble_graph, density_matched_threshold
from src.eval.graph_metrics import (
    MMDConfig,
    _descriptors,
    _induced_subgraph,
    evaluate_assembled_graph,
    mmd_squared,
    strip_self_loops,
)
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    _FEATURES_SUBDIR,
    AssembledRow,
    _assembled_row_to_dict,
    _edge_metrics_table_to_dict,
    _expected_candidate_rows,
    _self_pair_edge_metrics_to_dict,
    assemble_and_evaluate,
    compute_self_pair_edge_metrics,
    evaluate_regime_table,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.experiments.probes import evaluate_e2e_probe_artifact
from src.score_universe import ScoresArtifact, load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)

_COMPARATORS: tuple[str, ...] = ("b0", "b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq")
_PRIMARY_FAMILY: tuple[str, ...] = ("clustering_mmd_ratio", "bfs_macro_gs", "bfs_macro_rd")
_Z_95 = 1.959963984540054
_SE_FLOOR = 1e-12


class PreregistrationMismatch(RuntimeError):
    """Raised before any held-out metric is touched (prereg mechanics)."""


class PreregistrationNotBinding(PreregistrationMismatch):
    """Raised when a gate is offered a non-binding registration."""


class RegistrationShaMismatch(PreregistrationMismatch, ValueError):
    """Raised when formal metadata is not bound to the supplied registration."""


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


def _preregistration_snapshot(path: Path) -> tuple[dict[str, object], str]:
    """Parse and hash the same immutable registration bytes."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration must be a JSON object")
    return cast(dict[str, object], payload), hashlib.sha256(raw).hexdigest()


def _enforce_metadata_registration_hash(
    preregistration_path: Path, run_metadata_paths: Sequence[Path], expected: str
) -> str:
    """Validate metadata against an already-captured registration hash."""
    for path in run_metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        recorded = metadata.get("preregistration_sha256")
        if not isinstance(recorded, str) or not recorded:
            raise RegistrationShaMismatch(
                f"{path}: formal run metadata requires a non-empty preregistration_sha256"
            )
        if recorded != expected:
            raise RegistrationShaMismatch(
                f"{path}: preregistration_sha256 {recorded!r} does not match "
                f"{preregistration_path} ({expected}); held-out metrics stay closed "
                "(protocol Sec 5.2.4)"
            )
        if (
            metadata.get("run_kind") == "debug"
            or metadata.get("formal_artifacts_published") is False
        ):
            raise RegistrationShaMismatch(
                f"{path}: debug/non-formal run metadata cannot publish held-out metrics"
            )
    return expected


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
    _, expected = _preregistration_snapshot(preregistration_path)
    return _enforce_metadata_registration_hash(preregistration_path, run_metadata_paths, expected)


def enforce_e2e_preregistration(
    preregistration_path: Path,
    run_metadata_paths: Sequence[Path],
    *,
    snapshot: tuple[dict[str, object], str] | None = None,
) -> str:
    """Apply the rev-3 E2E-only BINDING status contract."""
    prereg, expected = snapshot or _preregistration_snapshot(preregistration_path)
    if prereg.get("status") != "BINDING":
        raise PreregistrationNotBinding(
            "held-out E2E G5 metrics require preregistration status == 'BINDING'"
        )
    frozen = cast(Mapping[str, object] | None, prereg.get("frozen_inputs"))
    b0cal = cast(Mapping[str, object] | None, frozen.get("b0cal_results") if frozen else None)
    digest = b0cal.get("sha256") if b0cal else None
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PreregistrationMismatch(
            "BINDING E2E registration requires a real b0cal_results sha256; "
            "REQUIRED-BEFORE-BINDING is not a digest"
        )
    return _enforce_metadata_registration_hash(preregistration_path, run_metadata_paths, expected)


def _enforce_e2e_evaluator_seed(prereg: Mapping[str, object], seed: int) -> None:
    """Reject any formal evaluator stream except the registered seed zero."""
    evaluator = cast(Mapping[str, object] | None, prereg.get("evaluator"))
    if evaluator is None or evaluator.get("seed") != 0 or seed != 0:
        raise RegistrationShaMismatch("formal E2E gate requires registered evaluator seed 0")


def paired_bootstrap_lower_bound(
    stat_fn: Callable[[object], float],
    samples_a: object,
    samples_b: object,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> float:
    """Return the paired two-sided lower bootstrap bound for ``stat(a) - stat(b)``.

    The same single RNG stream generates every index vector, and each vector is
    applied to both arms. Mapping inputs preserve the evaluator's size-bucket
    grouping while resampling each bucket independently.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if isinstance(samples_a, Mapping) != isinstance(samples_b, Mapping):
        raise ValueError("paired bootstrap inputs must have matching structures")
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    if isinstance(samples_a, Mapping):
        if not isinstance(samples_b, Mapping) or set(samples_a) != set(samples_b):
            raise ValueError("paired bootstrap bucket keys must match")
        bucketed_a = cast(Mapping[object, Sequence[object]], samples_a)
        bucketed_b = cast(Mapping[object, Sequence[object]], samples_b)
        for key in bucketed_a:
            if len(bucketed_a[key]) != len(bucketed_b[key]) or not bucketed_a[key]:
                raise ValueError("paired bootstrap buckets must be non-empty and aligned")
        for _ in range(n_boot):
            resampled_a: dict[object, list[object]] = {}
            resampled_b: dict[object, list[object]] = {}
            for key in bucketed_a:
                indices = rng.integers(0, len(bucketed_a[key]), size=len(bucketed_a[key]))
                resampled_a[key] = [bucketed_a[key][index] for index in indices]
                resampled_b[key] = [bucketed_b[key][index] for index in indices]
            differences.append(float(stat_fn(resampled_a) - stat_fn(resampled_b)))
    else:
        if isinstance(samples_b, Mapping):
            raise ValueError("paired bootstrap samples must be non-empty and aligned")
        sequence_a = cast(Sequence[object], samples_a)
        sequence_b = cast(Sequence[object], samples_b)
        if len(sequence_a) != len(sequence_b) or len(sequence_a) == 0:
            raise ValueError("paired bootstrap samples must be non-empty and aligned")
        for _ in range(n_boot):
            indices = rng.integers(0, len(sequence_a), size=len(sequence_a))
            resampled_sequence_a = [sequence_a[index] for index in indices]
            resampled_sequence_b = [sequence_b[index] for index in indices]
            differences.append(float(stat_fn(resampled_sequence_a) - stat_fn(resampled_sequence_b)))
    return float(np.quantile(np.asarray(differences, dtype=np.float64), alpha / 2.0))


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


def enforce_e2e_frozen_inputs(
    prereg: dict[str, object],
    preregistration_path: Path,
    b0_universe_path: Path,
    b0cal_results_path: Path,
) -> Path:
    """Verify the E2E comparator payload's exact path/digest in addition to legacy inputs."""
    enforce_frozen_inputs(prereg, preregistration_path, b0_universe_path)
    frozen = cast(dict[str, object], prereg.get("frozen_inputs"))
    entry = cast(dict[str, object] | None, frozen.get("b0cal_results"))
    if entry is None or "path" not in entry or "sha256" not in entry:
        raise PreregistrationMismatch("frozen input 'b0cal_results' is incompletely registered")
    resolved = _resolve_b0cal_results_path(b0cal_results_path).resolve()
    expected_path = _registered_path(preregistration_path, entry["path"]).resolve()
    if resolved != expected_path:
        raise PreregistrationMismatch(
            f"frozen b0cal_results path mismatch: {resolved} != {expected_path}"
        )
    actual = _sha256_file(resolved)
    if actual != entry["sha256"]:
        raise PreregistrationMismatch(
            f"frozen input 'b0cal_results' sha256 mismatch: {actual} != {entry['sha256']}"
        )
    return resolved


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


def validate_dead_residual(
    egostitch: ScoresArtifact,
    s0: ScoresArtifact,
    *,
    min_residual_std_ratio: float,
    max_spearman: float,
    max_topk_overlap: float,
    topk_fraction: float,
) -> dict[str, float]:
    """Fail closed when scale, rank, and top-k evidence all identify a dead residual."""
    s0_by_pair = {
        pair: float(logit) for pair, logit in zip(s0.pairs(), s0.logit.tolist(), strict=True)
    }
    pairs = list(egostitch.pairs())
    if len(s0_by_pair) != len(pairs) or any(pair not in s0_by_pair for pair in pairs):
        raise ValueError("dead-residual validation requires identical pair universes")
    aligned_s0 = np.asarray([s0_by_pair[pair] for pair in pairs], dtype=np.float64)
    logits = egostitch.logit.astype(np.float64)
    residual = logits - aligned_s0
    s0_std = float(np.std(aligned_s0))
    residual_std = float(np.std(residual))
    residual_ratio = residual_std / max(s0_std, 1e-30)
    correlation = float(spearmanr(aligned_s0, logits).statistic)
    if not np.isfinite(correlation):
        correlation = 1.0 if np.array_equal(aligned_s0, logits) else 0.0
    topk = max(1, min(len(pairs), int(round(len(pairs) * topk_fraction))))
    row_ids = np.arange(len(pairs), dtype=np.int64)
    s0_top = set(np.lexsort((row_ids, -aligned_s0))[:topk].tolist())
    ego_top = set(np.lexsort((row_ids, -logits))[:topk].tolist())
    overlap = len(s0_top & ego_top) / topk
    report = {
        "s0_std": s0_std,
        "residual_std": residual_std,
        "residual_s0_std_ratio": residual_ratio,
        "spearman_vs_s0": correlation,
        "topk_overlap_vs_s0": overlap,
        "topk_fraction": topk_fraction,
    }
    if (
        residual_ratio < min_residual_std_ratio
        and correlation > max_spearman
        and overlap > max_topk_overlap
    ):
        raise ValueError(f"dead residual fidelity gate failed: {report}")
    return report


def validate_dead_residual_within_checkpoint(
    artifact: ScoresArtifact,
    *,
    min_residual_std_ratio: float,
    max_spearman: float,
    max_topk_overlap: float,
    topk_fraction: float,
) -> dict[str, float]:
    """Fail closed when a checkpoint's own topology/content pathway carries no signal.

    Family ``egostitch_e2e`` liveness (spec Sec 13.17/13.8, re-registered
    2026-07-17) references the **within-checkpoint** ``f_logit`` arm of the
    SAME scored artifact instead of a fresh frozen-s0 comparator: ``full`` and
    ``f_logit`` already share row order by construction (one artifact, one
    scoring pass over the same pair universe), so no cross-artifact pair
    alignment step exists or is needed for this family — contrast
    :func:`validate_dead_residual`, which aligns two distinct artifacts for
    the historical frozen-s0 family.

    Args:
        artifact: A loaded ``egostitch_e2e`` scores artifact; its ``f_logit``
            array must be present.
        min_residual_std_ratio: Registered lower bound on
            ``std(full - f_logit) / std(f_logit)``.
        max_spearman: Registered upper bound on ``Spearman(full, f_logit)``.
        max_topk_overlap: Registered upper bound on the top-``topk_fraction``
            overlap between ``full`` and ``f_logit``.
        topk_fraction: Top fraction used for the overlap signal (spec pins
            ``0.01`` for this family).

    Returns:
        A diagnostics dict (std ratio, correlation, overlap, and their raw
        ingredients).

    Raises:
        ValueError: If `artifact.f_logit` is absent, or if all three death
            signals hold conjunctively (residual/`f_logit` standard-deviation
            ratio below `min_residual_std_ratio`, Spearman correlation with
            `f_logit` above `max_spearman`, and top-k overlap with `f_logit`
            above `max_topk_overlap`).
    """
    if artifact.f_logit is None:
        raise ValueError(
            "within-checkpoint liveness requires an artifact with an f_logit array "
            "(family egostitch_e2e)"
        )
    full = artifact.logit.astype(np.float64)
    f_logit = artifact.f_logit.astype(np.float64)
    residual = full - f_logit
    f_logit_std = float(np.std(f_logit))
    residual_std = float(np.std(residual))
    residual_ratio = residual_std / max(f_logit_std, 1e-30)
    correlation = float(spearmanr(f_logit, full).statistic)
    if not np.isfinite(correlation):
        correlation = 1.0 if np.array_equal(f_logit, full) else 0.0
    n = len(full)
    topk = max(1, min(n, int(round(n * topk_fraction))))
    row_ids = np.arange(n, dtype=np.int64)
    f_top = set(np.lexsort((row_ids, -f_logit))[:topk].tolist())
    full_top = set(np.lexsort((row_ids, -full))[:topk].tolist())
    overlap = len(f_top & full_top) / topk
    report = {
        "f_logit_std": f_logit_std,
        "residual_std": residual_std,
        "residual_f_logit_std_ratio": residual_ratio,
        "spearman_vs_f_logit": correlation,
        "topk_overlap_vs_f_logit": overlap,
        "topk_fraction": topk_fraction,
    }
    if (
        residual_ratio < min_residual_std_ratio
        and correlation > max_spearman
        and overlap > max_topk_overlap
    ):
        raise ValueError(f"within-checkpoint dead residual fidelity gate failed: {report}")
    return report


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


def _training_diagnostics(
    run_metadata: Sequence[dict[str, object]], *, require_e2e_submodule_rms: bool = False
) -> list[dict[str, object]]:
    """Validate the registered training-side series embedded in run metadata."""
    required = {"fidelity_series", "gradient_norm_series", "kendall_fallback"}
    reports: list[dict[str, object]] = []
    for index, metadata in enumerate(run_metadata):
        report = cast(dict[str, object] | None, metadata.get("training_diagnostics"))
        if report is None or required - report.keys():
            missing = sorted(required - (report.keys() if report is not None else set()))
            raise ValueError(f"run metadata {index} is missing training diagnostics: {missing}")
        if not cast(list[object], report["fidelity_series"]):
            raise ValueError(f"run metadata {index} has an empty fidelity series")
        if not cast(list[object], report["gradient_norm_series"]):
            raise ValueError(f"run metadata {index} has an empty gradient-norm series")
        if require_e2e_submodule_rms:
            rms_keys = {"grad_rms_trunk", "grad_rms_ste", "grad_rms_content"}
            for row_index, raw_row in enumerate(
                cast(list[dict[str, object]], report["gradient_norm_series"])
            ):
                # The worker publishes the registered spec §13.17 names nested
                # under `submodule_gradient_rms`; accept that shape alongside
                # flat rows.
                nested = raw_row.get("submodule_gradient_rms")
                source = cast(dict[str, object], nested) if isinstance(nested, dict) else raw_row
                missing_rms = rms_keys - source.keys()
                if missing_rms:
                    raise ValueError(
                        f"run metadata {index} gradient row {row_index} is missing "
                        f"submodule RMS telemetry: {sorted(missing_rms)}"
                    )
                if any(
                    not isinstance(source[key], (int, float))
                    or not math.isfinite(float(cast(float, source[key])))
                    or float(cast(float, source[key])) < 0.0
                    for key in rms_keys
                ):
                    raise ValueError(
                        f"run metadata {index} gradient row {row_index} has invalid "
                        "submodule RMS telemetry"
                    )
        reports.append(report)
    return reports


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


def _resolve_b0cal_results_path(path: Path) -> Path:
    """Accept the Task-11 deliverable directory or one explicit JSON file."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"b0cal results input does not exist: {path}")
    candidates = sorted(path.rglob("b0cal_results.json"))
    if len(candidates) != 1:
        raise ValueError(
            "b0cal results directory requires exactly one b0cal_results.json, "
            f"found {len(candidates)}"
        )
    return candidates[0]


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
    s0_universe_paths: Sequence[Path] = (),
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
        s0_universe_paths: Matching fp32 frozen-B0 candidate artifacts used to
            compute each scored residual. These are distinct from the canonical
            comparator artifact when that historical deliverable is quantized.
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
            outputs), embedded verbatim and required for the binding gate.
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

    # (1) Pre-registration enforcement FIRST — no scores are opened before this.
    prereg_sha = enforce_preregistration(preregistration_path, run_metadata_paths)
    prereg = cast(dict[str, object], json.loads(preregistration_path.read_text(encoding="utf-8")))
    enforce_frozen_inputs(prereg, preregistration_path, b0_universe_path)
    if len(s0_universe_paths) != n_runs:
        raise ValueError("one --s0-universe per --egostitch-universe is required")
    registered_seeds = cast(list[int], prereg.get("seeds"))
    if registered_seeds != [0]:
        raise ValueError("G5 Stage-1 registration must pin exactly seeds: [0]")
    if n_runs != len(registered_seeds):
        raise ValueError(
            f"G5 Stage-1 registration requires {len(registered_seeds)} seed artifact(s); "
            f"received {n_runs}"
        )
    primary_config = cast(dict[str, object], prereg["primary_criteria"])
    decision_procedure = primary_config.get("decision_procedure")
    if decision_procedure != "single_seed_point_estimate_dominance":
        raise ValueError("unsupported G5 Stage-1 decision_procedure")
    dead_residual_config = cast(dict[str, object] | None, prereg.get("fidelity_validity_gate"))
    if dead_residual_config is None:
        raise ValueError("preregistration is missing fidelity_validity_gate")
    fidelity, cost_report = _load_required_diagnostics(
        fidelity_report_paths, cost_report_path, n_runs
    )
    run_metadata = [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in run_metadata_paths
    ]
    training_diagnostics = _training_diagnostics(run_metadata)
    for expected_seed, metadata in zip(registered_seeds, run_metadata, strict=True):
        if metadata.get("seed") not in (None, expected_seed):
            raise ValueError(
                f"run metadata seed {metadata.get('seed')!r} does not match "
                f"registered seed {expected_seed}"
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

    b0cal_results_path = _resolve_b0cal_results_path(b0cal_results_path)
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
    dead_residual_reports: list[dict[str, float]] = []
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

    for artifact_path, s0_path, metadata in zip(
        egostitch_universe_paths, s0_universe_paths, run_metadata, strict=True
    ):
        universe = load_scores(artifact_path)
        s0_universe = load_scores(s0_path)
        validate_universe_artifact(
            universe, strategy=strategy, n_test_nodes=n_test_nodes, label=str(artifact_path)
        )
        validate_universe_artifact(
            s0_universe, strategy=strategy, n_test_nodes=n_test_nodes, label=str(s0_path)
        )
        if universe.meta.get("model_family") != "egostitch":
            raise ValueError(f"{artifact_path}: model_family must be 'egostitch'")
        if metadata.get("checkpoint_id") != universe.meta.get("checkpoint_id"):
            raise ValueError(f"{artifact_path}: run metadata checkpoint_id mismatch")
        if metadata.get("s0_checkpoint_id") != frozen_b0.get("checkpoint_id"):
            raise ValueError(f"{artifact_path}: run metadata s0_checkpoint_id mismatch")
        if s0_universe.meta.get("checkpoint_id") != frozen_b0.get("checkpoint_id"):
            raise ValueError(f"{s0_path}: s0 checkpoint_id mismatch")
        dead_residual_reports.append(
            validate_dead_residual(
                universe,
                s0_universe,
                min_residual_std_ratio=cast(float, dead_residual_config["min_residual_std_ratio"]),
                max_spearman=cast(float, dead_residual_config["max_spearman"]),
                max_topk_overlap=cast(float, dead_residual_config["max_topk_overlap"]),
                topk_fraction=cast(float, dead_residual_config["topk_fraction"]),
            )
        )
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

    # (4) Primary criteria. Stage 1 is a one-seed engineering screen, so
    # inferential fields and Holm decisions are deliberately not applicable.
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
    holm: dict[str, bool | None] = dict.fromkeys(_PRIMARY_FAMILY)
    primary_pass: dict[str, bool | None] = {
        name: criteria[name].dominates_every_comparator for name in _PRIMARY_FAMILY
    }

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

    verdict = (
        "pass"
        if all(value is True for value in primary_pass.values()) and degree_guard and auprc_guard
        else "cut"
    )
    criteria_payload = {name: criteria[name].to_jsonable() for name in _PRIMARY_FAMILY}
    for row in criteria_payload.values():
        row["se"] = None
        row["p_value"] = None
        row["ci_excludes_zero"] = None

    payload: dict[str, object] = {
        "metadata": {
            "preregistration_sha256": prereg_sha,
            "preregistration_path": str(preregistration_path),
            "benchmark": {"benchmark_a": strategy},
            "n_seeds": n_runs,
            "evaluation_mode": "single_seed_screening",
            "binding_verdict": True,
            "decision_procedure": decision_procedure,
            "evaluation_seed": seed,
            "holm_alpha": None,
            "matched_rd_rule": (
                "per comparator: egostitch realizes the comparator's exact non-self "
                "edge quota by descending pass-1 score; only the boundary-score tie "
                "is split by canonical pair order; self-pairs use the boundary score; "
                "every row enforces |RD_global(ego)-RD_global(comparator)| <= 0.005"
            ),
            "single_seed_caveat": (
                "This binding Stage-1 engineering screen uses one fixed training seed and "
                "deterministic point-estimate dominance. It does not establish statistical "
                "significance or cross-seed robustness and does not replace E1/E3's "
                "at-least-three-seed Holm procedure."
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
            "dead_residual_fidelity": dead_residual_reports,
            "matched_gs": matched_gs,
            "matched_rd": matched_rd,
            "matched_rd_quota_gap": matched_rd_gap,
            "matched_rd_boundary_score": matched_rd_boundary_score,
            "matched_rd_boundary_tie_size": matched_rd_boundary_tie_size,
            "matched_rd_selected_from_boundary_tie": matched_rd_selected_from_tie,
            "matched_rd_split_boundary_tie": matched_rd_split_boundary_tie,
        },
        "criteria": criteria_payload,
        "holm_survives": holm,
        "primary_pass": primary_pass,
        "guards": guards,
        "verdict": verdict,
        "failure_reading": (cast(str, prereg["failure_reading"]) if verdict == "cut" else None),
        "decision_rules_5_2_verbatim": prereg.get("decision_rules_5_2_verbatim"),
        "fidelity_reports": fidelity,
        "training_diagnostics": training_diagnostics,
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


# --------------------------------------------------------------------------- e2e five-arm summary

_E2E_ARMS: tuple[str, ...] = (
    "full",
    "b0_e2e_f_only",
    "pair_topology",
    "structure_control_6a",
    "p0",
)
_E2E_FORMAL_ARMS: tuple[str, ...] = (
    "full",
    "b0_e2e_f_only",
    "pair_topology",
    "p0",
)


def _e2e_registration_sha256(run_metadata: Mapping[str, Mapping[str, object]]) -> str:
    """Return the single registration hash shared by every provided run metadata.

    Args:
        run_metadata: Formal-run arm name -> its parsed ``run_metadata.json``.
            ``structure_control_6a`` never has an entry (it is a scoring-time
            control over the ``full`` checkpoint, not its own training run).

    Returns:
        The non-empty common ``preregistration_sha256`` value.

    Raises:
        ValueError: If the metadata records are not exactly the four formal
            arms, a registration hash is missing/empty, or hashes disagree.
    """
    expected = set(_E2E_FORMAL_ARMS)
    provided = set(run_metadata)
    if provided != expected:
        raise ValueError(
            "e2e summary requires exactly the four formal run metadata records "
            f"{sorted(expected)}, got {sorted(provided)}"
        )
    hashes = {
        name: metadata.get("preregistration_sha256") for name, metadata in run_metadata.items()
    }
    if any(not isinstance(value, str) or not value for value in hashes.values()):
        raise ValueError(f"e2e run metadata require a non-empty preregistration_sha256: {hashes}")
    distinct = {cast(str, value) for value in hashes.values()}
    if len(distinct) != 1:
        raise RegistrationShaMismatch(
            f"e2e arm run metadata disagree on preregistration_sha256: {hashes}"
        )
    return next(iter(distinct))


def _enforce_e2e_formal_metadata(
    run_metadata: Mapping[str, Mapping[str, object]],
    *,
    preregistration: Mapping[str, object],
    preregistration_path: Path,
) -> None:
    """Reject anything except completed, explicitly formal E2E arm metadata."""
    expected_semantics: dict[str, tuple[str, float, float]] = {
        "full": ("none", 0.15, 0.15),
        "b0_e2e_f_only": ("all_head", 0.15, 0.15),
        "pair_topology": ("content_head", 0.15, 0.15),
        "p0": ("none", 0.0, 0.0),
    }
    checkpoints: set[str] = set()
    arms = cast(Mapping[str, Mapping[str, object]], preregistration.get("arms"))
    if set(arms) != set(_E2E_ARMS):
        raise RegistrationShaMismatch("registration must bind exactly the five E2E arms")
    benchmark = cast(Mapping[str, object], preregistration.get("benchmark"))
    registered_strategy = benchmark.get("strategy")
    from src.train_egostitch import _config_hash, load_config

    for arm, metadata in run_metadata.items():
        if metadata.get("run_kind") != "formal":
            raise RegistrationShaMismatch(f"{arm}: run metadata must declare run_kind 'formal'")
        if metadata.get("formal_artifacts_published") is not True:
            raise RegistrationShaMismatch(f"{arm}: formal_artifacts_published must be exactly true")
        if metadata.get("status") != "complete":
            raise RegistrationShaMismatch(f"{arm}: formal run metadata status must be 'complete'")
        checkpoint = metadata.get("checkpoint_id")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise RegistrationShaMismatch(f"{arm}: run metadata needs a checkpoint_id")
        if checkpoint in checkpoints:
            raise RegistrationShaMismatch("formal E2E arms must use distinct checkpoint_id values")
        checkpoints.add(checkpoint)
        permanent_null, p_topo, p_cont = expected_semantics[arm]
        if metadata.get("model_family") != "egostitch_e2e":
            raise RegistrationShaMismatch(f"{arm}: model_family must be 'egostitch_e2e'")
        if metadata.get("permanent_null") != permanent_null:
            raise RegistrationShaMismatch(f"{arm}: permanent_null does not match registered arm")
        if metadata.get("p_topo") != p_topo or metadata.get("p_cont") != p_cont:
            raise RegistrationShaMismatch(f"{arm}: branch dropout does not match registered arm")
        if metadata.get("seed") != 0:
            raise RegistrationShaMismatch(f"{arm}: formal E2E seed must be 0")
        if metadata.get("strategy") != registered_strategy:
            raise RegistrationShaMismatch(f"{arm}: strategy does not match registration")
        if metadata.get("partition_seed") != 0:
            raise RegistrationShaMismatch(f"{arm}: formal E2E partition_seed must be 0")
        training = arms[arm].get("training")
        expected_config_path = _registered_path(preregistration_path, training).resolve()
        config_path = metadata.get("config_path")
        if not isinstance(config_path, str) or Path(config_path).resolve() != expected_config_path:
            raise RegistrationShaMismatch(
                f"{arm}: config_path does not match registered arm config"
            )
        expected_config_hash = _config_hash(load_config(expected_config_path))
        if metadata.get("config_hash") != expected_config_hash:
            raise RegistrationShaMismatch(
                f"{arm}: config_hash does not match registered arm config"
            )


def _validate_e2e_universe_shape(
    artifact: ScoresArtifact, *, strategy: str, n_test_nodes: int, label: str
) -> None:
    """Structural candidate-universe checks for family ``egostitch_e2e`` artifacts.

    Mirrors :func:`src.experiments.g1_hardened_e2.validate_universe_artifact`'s
    non-precision checks (``pairs_source``/``strategy``/row-count/label-domain)
    WITHOUT its internal ``validate_score_precision(artifact.logit, meta=...)``
    call: that raw idiom always reports the ``f_logit``/``pair_content``/
    ``pair_topology`` arrays as "missing" for this family, because it never
    forwards them as ``extra_arrays`` (the exact footgun
    :func:`validate_artifact_precision` exists to avoid — Task 14 review
    finding). Precision is validated separately, via
    :func:`validate_artifact_precision`, in :func:`build_e2e_arm_summary`.

    Args:
        artifact: The loaded scores artifact.
        strategy: Expected split strategy name.
        n_test_nodes: Number of test nodes the candidate universe is defined
            over (used to derive the expected row count ``C(n, 2) + n``).
        label: Human-readable artifact label used in error messages.

    Raises:
        ValueError: If ``meta["pairs_source"] != "candidate"``,
            ``meta["strategy"]`` does not match `strategy`, the row count does
            not equal the expected candidate-universe size, or any label is
            outside ``{0, 1}``.
    """
    errors: list[str] = []
    pairs_source = artifact.meta.get("pairs_source")
    if pairs_source != "candidate":
        errors.append(f"{label}: pairs_source expected 'candidate', got {pairs_source!r}")
    meta_strategy = artifact.meta.get("strategy")
    if meta_strategy != strategy:
        errors.append(f"{label}: strategy expected {strategy!r}, got {meta_strategy!r}")
    expected_rows = _expected_candidate_rows(n_test_nodes)
    n_rows = len(artifact.logit)
    if n_rows != expected_rows:
        errors.append(
            f"{label}: row count expected {expected_rows} "
            f"(C({n_test_nodes},2)+{n_test_nodes}), got {n_rows}"
        )
    labels_present = set(np.unique(artifact.label).tolist())
    bad_labels = labels_present - {0, 1}
    if bad_labels:
        errors.append(f"{label}: label values outside {{0,1}} found: {sorted(bad_labels)}")
    if errors:
        raise ValueError("; ".join(errors))


def _validate_e2e_scoring_provenance(
    name: str,
    artifact: ScoresArtifact,
    registered_arm: Mapping[str, object],
) -> None:
    """Bind one score artifact to the registered training/control semantics."""
    expected = cast(Mapping[str, object] | None, registered_arm.get("scoring_provenance"))
    if expected is None:
        raise RegistrationShaMismatch(f"{name}: registration has no scoring_provenance")
    expected_control = {
        "mode": expected.get("scaffold_control"),
        "seed": 0,
        "keying": "canonical_pair_v1",
    }
    actual_control = artifact.meta.get("scaffold_control")
    if actual_control != expected_control:
        raise RegistrationShaMismatch(
            f"{name}: scaffold_control provenance mismatch: "
            f"{actual_control!r} != {expected_control!r}"
        )
    for key in ("permanent_null", "primary_logit"):
        if artifact.meta.get(key) != expected.get(key):
            raise RegistrationShaMismatch(
                f"{name}: {key} provenance mismatch: "
                f"{artifact.meta.get(key)!r} != {expected.get(key)!r}"
            )
    if name == "structure_control_6a":
        if expected.get("seed") != 0 or expected.get("keying") != "canonical_pair_v1":
            raise RegistrationShaMismatch(
                "structure_control_6a registration must pin seed 0/canonical_pair_v1"
            )
        if expected.get("checkpoint_arm") != "full":
            raise RegistrationShaMismatch(
                "structure_control_6a registration must pin checkpoint_arm 'full'"
            )


def _registered_candidate_manifest(
    preregistration: Mapping[str, object],
    preregistration_path: Path,
    benchmark_root: Path,
    strategy: str,
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Load and hash-check the exact registered candidate pair/label manifest."""
    frozen = cast(Mapping[str, object], preregistration.get("frozen_inputs"))
    entry = cast(Mapping[str, object] | None, frozen.get("candidate_manifest"))
    if entry is None or "path" not in entry or "sha256" not in entry:
        raise PreregistrationMismatch("frozen candidate_manifest is incompletely registered")
    registered_path = _registered_path(preregistration_path, entry["path"]).resolve()
    benchmark_path = (benchmark_root / strategy / "candidate_test_edges.txt").resolve()
    if registered_path != benchmark_path:
        raise PreregistrationMismatch(
            f"candidate_manifest path mismatch: {registered_path} != {benchmark_path}"
        )
    if not registered_path.is_file():
        raise PreregistrationMismatch(f"candidate_manifest not found: {registered_path}")
    actual_sha = _sha256_file(registered_path)
    if actual_sha != entry["sha256"]:
        raise PreregistrationMismatch(
            f"candidate_manifest sha256 mismatch: {actual_sha} != {entry['sha256']}"
        )
    candidate = load_candidate_pairs(benchmark_root, strategy)
    return candidate.pairs, candidate.labels


def _clustering_mmd_samples(
    g_pred: nx.Graph, g_ref: nx.Graph, buckets: Mapping[int, Sequence[set[str]]]
) -> dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Capture one clustering descriptor pair per fixed evaluator subgraph."""
    return {
        size: [
            (
                _descriptors(_induced_subgraph(g_pred, nodes))["clustering"],
                _descriptors(_induced_subgraph(g_ref, nodes))["clustering"],
            )
            for nodes in node_sets
        ]
        for size, node_sets in buckets.items()
    }


def _clustering_mmd_ratio(
    samples: Sequence[object] | Mapping[object, Sequence[object]], config: MMDConfig
) -> float:
    """Recompute the canonical clustering-MMD ratio from fixed-subgraph descriptors."""
    if not isinstance(samples, Mapping):
        raise TypeError("clustering MMD bootstrap samples must remain bucketed")
    raw: list[float] = []
    reference: list[float] = []
    for values in samples.values():
        pairs = cast(Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]], values)
        pred = [pair[0] for pair in pairs]
        ref = [pair[1] for pair in pairs]
        raw.append(mmd_squared(pred, ref, config))
        reference.append(mmd_squared(ref[::2], ref[1::2], config))
    return float(np.mean(raw) / max(float(np.mean(reference)), config.reference_epsilon))


def _array_summary(values: NDArray[np.float32] | NDArray[np.float64]) -> dict[str, float]:
    """Deterministic descriptive row for one aligned logit/delta array."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _artifact_four_logits(artifact: ScoresArtifact) -> dict[str, NDArray[np.float32]]:
    """Return the true aligned four-logit arrays regardless of active primary arm."""
    primary = artifact.meta.get("primary_logit")
    full = artifact.logit if primary == "full" else artifact.full_logit
    if (
        full is None
        or artifact.f_logit is None
        or artifact.pair_content is None
        or artifact.pair_topology is None
    ):
        raise ValueError("E2E artifact is missing its complete four-logit decomposition")
    return {
        "full": full,
        "f_logit": artifact.f_logit,
        "pair_content": artifact.pair_content,
        "pair_topology": artifact.pair_topology,
    }


def _e2e_decomposition_summary(
    artifacts: Mapping[str, ScoresArtifact],
) -> dict[str, object]:
    """Complete per-arm four-logit and registered aligned-delta evidence."""
    arrays = {name: _artifact_four_logits(artifact) for name, artifact in artifacts.items()}
    arm_rows: dict[str, object] = {}
    for name, values in arrays.items():
        arm_rows[name] = {
            "four_logits": {key: _array_summary(array) for key, array in values.items()},
            "deltas": {
                "full_minus_f_logit": _array_summary(values["full"] - values["f_logit"]),
                "topology_delta_full_minus_pair_content": _array_summary(
                    values["full"] - values["pair_content"]
                ),
                "content_delta_full_minus_pair_topology": _array_summary(
                    values["full"] - values["pair_topology"]
                ),
            },
        }
    full = arrays["full"]
    p0 = arrays["p0"]
    return {
        "arms": arm_rows,
        "cross_arm": {
            "full_bypass_vs_trained_pair_topology": _array_summary(
                full["pair_topology"] - artifacts["pair_topology"].logit
            ),
            "p0_minus_full": {key: _array_summary(p0[key] - full[key]) for key in full},
        },
    }


def _evaluate_registered_e2e_probe(
    *,
    probe_artifact_path: Path,
    preregistration: Mapping[str, object],
    preregistration_path: Path,
    run_metadata_path: Path,
    data_root: Path,
    strategy: str,
) -> dict[str, object]:
    """Rebuild G_struct identities and consume the required probe artifact."""
    from src import train_egostitch as te
    from src.data.partition import build_g_struct, derive_partition

    registration = cast(Mapping[str, object] | None, preregistration.get("probe_artifact"))
    if registration is None or registration.get("format") != "egostitch_e2e_probe_v1":
        raise PreregistrationMismatch("registration is missing the E2E probe artifact contract")
    expected_path = _registered_path(
        preregistration_path, registration.get("expected_path")
    ).resolve()
    if probe_artifact_path.resolve() != expected_path:
        raise PreregistrationMismatch(
            f"E2E probe artifact path mismatch: {probe_artifact_path.resolve()} != {expected_path}"
        )
    metadata = cast(dict[str, object], json.loads(run_metadata_path.read_text(encoding="utf-8")))
    config_path = Path(str(metadata.get("config_path")))
    cfg = te.load_config(config_path)
    if cfg.data.root.resolve() != data_root.resolve() or cfg.data.strategy != strategy:
        raise RegistrationShaMismatch("full-arm config data identity does not match gate inputs")
    benchmark = te._load_benchmark_for(cfg)
    operative = sorted(set(benchmark.graph.nodes()) - set(cfg.data.expected_missing_features))
    train_nodes = sorted(set(benchmark.split.train_nodes) & set(operative))
    train_positives = [
        pair
        for pair, label in zip(
            benchmark.split.train_pairs.pairs,
            benchmark.split.train_pairs.labels,
            strict=True,
        )
        if label == 1
    ]
    partition = derive_partition(
        train_positives,
        seed=cfg.data.partition_seed,
        msg_fraction=cfg.data.msg_fraction,
    )
    graph = build_g_struct(train_nodes, partition.e_msg)
    return evaluate_e2e_probe_artifact(
        probe_artifact_path,
        graph=graph,
        train_nodes=train_nodes,
        expected_metadata={
            "checkpoint_id": metadata.get("checkpoint_id"),
            "registration_sha256": metadata.get("preregistration_sha256"),
            "config_hash": metadata.get("config_hash"),
            "seed": 0,
            "partition_seed": 0,
            "strategy": strategy,
        },
    )


def build_e2e_arm_summary(
    *,
    arm_universe_paths: Mapping[str, Path],
    run_metadata_paths: Mapping[str, Path],
    preregistration_path: Path,
    data_root: Path,
    strategy: str,
    liveness_config: Mapping[str, float],
    seed: int = 0,
    preregistration_snapshot: tuple[dict[str, object], str] | None = None,
) -> dict[str, object]:
    """Build the registered five-arm ``egostitch_e2e`` Stage-1 summary table.

    Loads and validates every arm's scores artifact through
    :func:`validate_artifact_precision` (the artifact-aware entry point — never
    the raw ``validate_score_precision(artifact.logit, meta=...)`` idiom, which
    silently drops the ``f_logit``/``pair_content``/``pair_topology`` arrays),
    then reports each arm's canonical-operating-point assembled metrics plus
    the ``full`` arm's within-checkpoint liveness report. This is a
    self-contained entry point distinct from :func:`run_g5_stage1_pipeline`
    (which remains the historical frozen-s0 family gate, unmodified): every
    arm artifact and every formal run's metadata is loaded and validated in
    one place here, so a later registration-``BINDING``-status enforcement
    hook (exp-Task 7's scope) slots in without touching the per-arm metric
    computation below. This function does not itself compute the registered
    pathway-attribution or structure-control decision rules (spec Sec 14,
    ``e2e_rules``) — those consume this table's per-arm ``clustering_mmd_ratio``
    values plus (for the structure-control condition) a paired-bootstrap
    procedure that is exp-Task 7's scope.

    Args:
        arm_universe_paths: Exact registered five-arm mapping (``full``,
            ``b0_e2e_f_only``, ``pair_topology``, ``structure_control_6a``,
            ``p0``) -> its scored ``.npz`` artifact.
        run_metadata_paths: Formal-run arm name -> its ``run_metadata.json``.
            ``structure_control_6a`` never has an entry.
        preregistration_path: Binding registration whose sha must be present
            in every one of the four formal metadata records.
        data_root: Directory containing the benchmark package.
        strategy: Benchmark split strategy.
        liveness_config: The registered ``min_residual_std_ratio`` /
            ``max_spearman`` / ``max_topk_overlap`` / ``topk_fraction``
            liveness thresholds (spec Sec 13.17, re-registered for family
            ``egostitch_e2e``).
        seed: Bootstrap/evaluation seed for the assembled-graph metrics.
        preregistration_snapshot: Optional already-captured immutable registration
            payload and SHA-256, used by the enclosing formal gate.

    Returns:
        A JSON-ready payload: ``registration_sha256`` (the non-empty single
        hash shared by the four formal run metadata records), a per-arm
        ``checkpoint_id`` / ``registration_sha256`` / ``assembled`` /
        ``degree_corrected_auprc`` row, and the ``full`` arm's
        within-checkpoint ``liveness`` report.

    Raises:
        ValueError: On an invalid five-arm composition, invalid formal-run
            metadata/hash provenance, mismatched scoring checkpoint identities,
            an artifact whose ``model_family`` is not ``egostitch_e2e``, or an
            artifact that fails its candidate-universe shape check or
            `validate_artifact_precision`.
    """
    unknown = set(arm_universe_paths) - set(_E2E_ARMS)
    if unknown:
        raise ValueError(f"unrecognized e2e arm(s): {sorted(unknown)}")
    if "full" not in arm_universe_paths:
        raise ValueError("the 'full' arm is required to build the e2e arm summary")
    if set(arm_universe_paths) != set(_E2E_ARMS):
        raise ValueError(
            "e2e summary requires exactly the five registered arms "
            f"{list(_E2E_ARMS)}, got {sorted(arm_universe_paths)}"
        )

    resolved_metadata_paths = [path.resolve() for path in run_metadata_paths.values()]
    if len(set(resolved_metadata_paths)) != len(resolved_metadata_paths):
        raise RegistrationShaMismatch("formal E2E arms require distinct run metadata files")
    run_metadata: dict[str, dict[str, object]] = {
        name: cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for name, path in run_metadata_paths.items()
    }
    snapshot = preregistration_snapshot or _preregistration_snapshot(preregistration_path)
    preregistration, _ = snapshot
    bound_registration_sha256 = enforce_e2e_preregistration(
        preregistration_path, list(run_metadata_paths.values()), snapshot=snapshot
    )
    registration_sha256 = _e2e_registration_sha256(run_metadata)
    _enforce_e2e_formal_metadata(
        run_metadata,
        preregistration=preregistration,
        preregistration_path=preregistration_path,
    )
    training_diagnostics = _training_diagnostics(
        [run_metadata[name] for name in _E2E_FORMAL_ARMS],
        require_e2e_submodule_rms=True,
    )
    if registration_sha256 != bound_registration_sha256:  # pragma: no cover - enforced above
        raise RegistrationShaMismatch("formal e2e metadata disagree with the binding registration")

    # Run validity is evaluated before any held-out assembled topology metric.
    full_artifact = load_scores(arm_universe_paths["full"])
    if full_artifact.meta.get("model_family") != "egostitch_e2e":
        raise ValueError("full artifact: model_family must be 'egostitch_e2e'")
    validate_artifact_precision(full_artifact, label="full artifact")
    liveness = validate_dead_residual_within_checkpoint(
        full_artifact,
        min_residual_std_ratio=liveness_config["min_residual_std_ratio"],
        max_spearman=liveness_config["max_spearman"],
        max_topk_overlap=liveness_config["max_topk_overlap"],
        topk_fraction=liveness_config["topk_fraction"],
    )

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    registered_strategy = cast(Mapping[str, object], preregistration["benchmark"]).get("strategy")
    if strategy != registered_strategy:
        raise RegistrationShaMismatch("E2E gate strategy must match registration")
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    if strategy == "breadth_first" and sum(len(node_sets) for node_sets in buckets.values()) != 500:
        raise ValueError("registered fixed evaluator must contain exactly 500 subgraphs")
    n_test_nodes = g_ref.number_of_nodes()
    target_edges = strip_self_loops(g_ref).number_of_edges()
    nodes = list(g_ref.nodes())
    config = MMDConfig()
    candidate_pairs, candidate_labels = _registered_candidate_manifest(
        preregistration, preregistration_path, benchmark_root, strategy
    )
    registered_arms = cast(Mapping[str, Mapping[str, object]], preregistration["arms"])

    store: FeatureStore | None = None
    features_root = data_root / _FEATURES_SUBDIR
    if (features_root / "index.json").exists():
        store = FeatureStore(features_root)
    f0_cache = arm_universe_paths["full"].parent / "e2e_arm_summary_f0_cache.pt"

    artifacts: dict[str, ScoresArtifact] = {}
    rows: dict[str, dict[str, object]] = {}
    clustering_samples: dict[
        str, dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64]]]]
    ] = {}
    for name, path in arm_universe_paths.items():
        label = f"{name} ({path})"
        artifact = load_scores(path)
        _validate_e2e_universe_shape(
            artifact, strategy=strategy, n_test_nodes=n_test_nodes, label=label
        )
        if artifact.meta.get("model_family") != "egostitch_e2e":
            raise ValueError(f"{label}: model_family must be 'egostitch_e2e'")
        validate_artifact_precision(artifact, label=label)
        _validate_e2e_scoring_provenance(name, artifact, registered_arms[name])
        if list(artifact.pairs()) != candidate_pairs:
            raise RegistrationShaMismatch(
                f"{label}: candidate pair identity/order does not match frozen manifest"
            )
        if not np.array_equal(artifact.label, candidate_labels):
            raise RegistrationShaMismatch(f"{label}: candidate labels do not match frozen manifest")
        if name in _E2E_FORMAL_ARMS:
            metadata_checkpoint = run_metadata[name].get("checkpoint_id")
            artifact_checkpoint = artifact.meta.get("checkpoint_id")
            if (
                not isinstance(metadata_checkpoint, str)
                or not metadata_checkpoint
                or artifact_checkpoint != metadata_checkpoint
            ):
                raise ValueError(
                    f"{label}: run metadata checkpoint_id mismatch: "
                    f"{metadata_checkpoint!r} != {artifact_checkpoint!r}"
                )
        artifacts[name] = artifact

        probs = artifact.probs()
        pairs = list(artifact.pairs())
        non_self = artifact.u_idx != artifact.v_idx
        threshold = density_matched_threshold(probs[non_self], target_edges)
        graph = assemble_graph(pairs, probs, threshold=threshold, nodes=nodes)
        clustering_samples[name] = _clustering_mmd_samples(graph, g_ref, buckets)
        assembled = assemble_and_evaluate(
            g_pred=graph,
            g_ref=g_ref,
            buckets=buckets,
            config=config,
            seed=seed,
            threshold=threshold,
        )
        regimes = evaluate_regime_table(
            labels=artifact.label,
            probs=probs,
            u_idx=artifact.u_idx,
            v_idx=artifact.v_idx,
            node_ids=artifact.node_ids,
            g_ref=g_ref,
            store=store,
            f0_cache=f0_cache,
            seed=seed,
        )
        rows[name] = {
            "checkpoint_id": artifact.meta.get("checkpoint_id"),
            "registration_sha256": registration_sha256,
            "assembled": _assembled_row_to_dict(assembled),
            "degree_corrected_auprc": regimes["degree_corrected"]["ratio_1"].auprc,
        }

    full_checkpoint = artifacts["full"].meta.get("checkpoint_id")
    control_checkpoint = artifacts["structure_control_6a"].meta.get("checkpoint_id")
    if control_checkpoint != full_checkpoint:
        raise ValueError(
            "structure_control_6a checkpoint_id must match full scoring checkpoint: "
            f"{control_checkpoint!r} != {full_checkpoint!r}"
        )

    lower_bound = paired_bootstrap_lower_bound(
        lambda samples: _clustering_mmd_ratio(
            cast(Sequence[object] | Mapping[object, Sequence[object]], samples), config
        ),
        clustering_samples["structure_control_6a"],
        clustering_samples["full"],
        n_boot=1000,
        seed=0,
        alpha=0.05,
    )
    decomposition = _e2e_decomposition_summary(artifacts)

    return {
        "registration_sha256": registration_sha256,
        "arms": rows,
        "training_diagnostics": dict(zip(_E2E_FORMAL_ARMS, training_diagnostics, strict=True)),
        "decomposition": decomposition,
        "liveness": liveness,
        "structure_control": {
            "metric": "clustering_mmd_ratio",
            "lower_bound": lower_bound,
            "passed": lower_bound > 0.0,
            "n_boot": 1000,
            "seed": 0,
            "alpha": 0.05,
            "scope": "fixed-seed evaluator-stability evidence; not cross-seed or inferential",
        },
    }


_E2E_LIVENESS_CONFIG: dict[str, float] = {
    "min_residual_std_ratio": 1e-5,
    "max_spearman": 0.9999,
    "max_topk_overlap": 0.9999,
    "topk_fraction": 0.01,
}


def render_e2e_tables_markdown(payload: Mapping[str, object]) -> str:
    """Render the complete five-arm, decomposition, and probe evidence tables."""
    arms = cast(Mapping[str, Mapping[str, object]], payload["arms"])
    lines = [
        "# G5 E2E Stage-1 gate",
        "",
        f"**Verdict: `{payload['verdict']}`**",
        "",
        "Fixed-seed evaluator-stability evidence only; not significance or cross-seed robustness.",
        "",
        "## Five-arm summary",
        "",
        "| arm | checkpoint | GS | BFS-macro RD | degree MMD | clustering MMD | "
        "spectral MMD | degree-corrected AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _E2E_ARMS:
        row = arms[name]
        assembled = cast(Mapping[str, object], row["assembled"])
        mmd = cast(Mapping[str, float], assembled["mmd_ratio"])
        lines.append(
            f"| {name} | {row['checkpoint_id']} | {_fmt(assembled['graph_similarity'])} "
            f"| {_fmt(assembled['relative_density'])} | {_fmt(mmd['degree'])} "
            f"| {_fmt(mmd['clustering'])} | {_fmt(mmd['spectral'])} "
            f"| {_fmt(row['degree_corrected_auprc'])} |"
        )
    decomposition = cast(Mapping[str, object], payload["decomposition"])
    decomposition_arms = cast(Mapping[str, Mapping[str, object]], decomposition["arms"])
    lines += [
        "",
        "## Four-logit decomposition deltas",
        "",
        "| arm | full-f mean | full-f std | topology delta mean | topology delta std | "
        "content delta mean | content delta std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _E2E_ARMS:
        deltas = cast(Mapping[str, Mapping[str, float]], decomposition_arms[name]["deltas"])
        f_delta = deltas["full_minus_f_logit"]
        topo = deltas["topology_delta_full_minus_pair_content"]
        content = deltas["content_delta_full_minus_pair_topology"]
        lines.append(
            f"| {name} | {_fmt(f_delta['mean'])} | {_fmt(f_delta['std'])} "
            f"| {_fmt(topo['mean'])} | {_fmt(topo['std'])} "
            f"| {_fmt(content['mean'])} | {_fmt(content['std'])} |"
        )
    probes = cast(Mapping[str, object], payload["probes"])
    probe_r2 = cast(Mapping[str, float], probes["linear_probe_r2"])
    partial = cast(Mapping[str, float], probes["degree_partialled_r2"])
    lines += [
        "",
        "## Representation probes (nonbinding)",
        "",
        "| target | ridge R2 | degree-partialled R2 |",
        "|---|---:|---:|",
    ]
    for target in ("degree", "ego_density", "clustering"):
        lines.append(f"| {target} | {_fmt(probe_r2[target])} | {_fmt(partial.get(target))} |")
    pi = cast(Mapping[str, object], probes["pi_shared_neighbor_consistency"])
    lines += [
        "",
        "| Pi/shared-neighbor consistency | mean | std | nonzero fraction | n pairs |",
        "|---|---:|---:|---:|---:|",
        f"| registered E_msg selection | {_fmt(pi['mean'])} | {_fmt(pi['std'])} "
        f"| {_fmt(pi['nonzero_fraction'])} | {_fmt(pi['n_pairs'])} |",
        "",
        "## Decision checks",
        "",
        f"- Primary pass: `{payload['primary_pass']}`",
        f"- Guards: `{payload['guards']}`",
        f"- Pathway attribution: `{payload['pathway_attribution']}`",
        f"- Structure control: `{payload['structure_control']}`",
        "",
    ]
    return "\n".join(lines)


def run_g5_e2e_stage1_pipeline(
    *,
    arm_universe_paths: Mapping[str, Path],
    run_metadata_paths: Mapping[str, Path],
    b0_universe_path: Path,
    b0cal_results_path: Path,
    probe_artifact_path: Path,
    preregistration_path: Path,
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
) -> dict[str, object]:
    """Run the binding E2E five-arm G5 gate and write its verdict artifacts."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # A rejected run must not leave an older verdict looking authoritative.
    for filename in ("g5_e2e_stage1_results.json", "g5_e2e_stage1_tables.md"):
        (output_dir / filename).unlink(missing_ok=True)
    preregistration_snapshot = _preregistration_snapshot(preregistration_path)
    prereg, _ = preregistration_snapshot
    if cast(Mapping[str, object], prereg["benchmark"]).get("strategy") != "breadth_first":
        raise RegistrationShaMismatch("formal E2E gate requires registered breadth_first strategy")
    _enforce_e2e_evaluator_seed(prereg, seed)
    b0cal_results_path = enforce_e2e_frozen_inputs(
        prereg,
        preregistration_path,
        b0_universe_path,
        b0cal_results_path,
    )
    probes = _evaluate_registered_e2e_probe(
        probe_artifact_path=probe_artifact_path,
        preregistration=prereg,
        preregistration_path=preregistration_path,
        run_metadata_path=run_metadata_paths["full"],
        data_root=data_root,
        strategy=strategy,
    )
    summary = build_e2e_arm_summary(
        arm_universe_paths=arm_universe_paths,
        run_metadata_paths=run_metadata_paths,
        preregistration_path=preregistration_path,
        data_root=data_root,
        strategy=strategy,
        liveness_config=_E2E_LIVENESS_CONFIG,
        seed=seed,
        preregistration_snapshot=preregistration_snapshot,
    )

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    nodes = list(g_ref.nodes())
    target_edges = strip_self_loops(g_ref).number_of_edges()
    n_test_nodes = g_ref.number_of_nodes()
    config = MMDConfig()
    b0_universe = load_scores(b0_universe_path)
    validate_universe_artifact(
        b0_universe, strategy=strategy, n_test_nodes=n_test_nodes, label="b0 universe"
    )
    frozen = cast(dict[str, object], prereg["frozen_inputs"])
    frozen_b0 = cast(dict[str, object], frozen["b0_candidate_scores"])
    if b0_universe.meta.get("checkpoint_id") != frozen_b0.get("checkpoint_id"):
        raise PreregistrationMismatch("frozen input B0 checkpoint_id mismatch")
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
    b0cal_payload = cast(
        dict[str, object], json.loads(b0cal_results_path.read_text(encoding="utf-8"))
    )
    _validate_b0cal_lineage(b0cal_payload, b0_universe.meta, b0_row)
    b0cal_assembled = cast(dict[str, dict[str, object]], b0cal_payload["assembled"])
    comparators = {name: dict(b0cal_assembled[name]) for name in _COMPARATORS if name != "b0"}
    comparators["b0"] = _assembled_row_to_dict(b0_row)
    realized = cast(
        dict[str, int],
        cast(dict[str, object], b0cal_payload["metadata"])["realized_non_self_edges"],
    )
    realized = dict(realized)
    realized["b0"] = int(strip_self_loops(b0_graph).number_of_edges())

    full_artifact = load_scores(arm_universe_paths["full"])
    full_probs = full_artifact.probs()
    full_pairs = list(full_artifact.pairs())
    full_matched: dict[str, dict[str, float]] = {}
    for name in _COMPARATORS:
        selected = select_matched_global_rd_rows(
            full_probs,
            full_artifact.u_idx,
            full_artifact.v_idx,
            target_edges=realized[name],
            reference_edges=target_edges,
        )
        graph = assemble_matched_global_rd_graph(
            full_pairs,
            full_probs,
            full_artifact.u_idx,
            full_artifact.v_idx,
            selected,
            nodes,
        )
        report = evaluate_assembled_graph(graph, g_ref, buckets, config)
        full_matched[name] = {
            "bfs_macro_gs": report.graph_similarity,
            "bfs_macro_rd": report.relative_density,
        }

    arms = cast(dict[str, dict[str, object]], summary["arms"])
    full = arms["full"]
    full_assembled = cast(dict[str, object], full["assembled"])
    full_mmd = cast(dict[str, float], full_assembled["mmd_ratio"])
    clustering_pass = all(
        full_mmd["clustering"] < cast(dict[str, float], row["mmd_ratio"])["clustering"]
        for row in comparators.values()
    )
    gs_pass = all(
        full_matched[name]["bfs_macro_gs"] > cast(float, comparators[name]["graph_similarity"])
        for name in _COMPARATORS
    )
    rd_pass = all(
        full_matched[name]["bfs_macro_rd"] > cast(float, comparators[name]["relative_density"])
        for name in _COMPARATORS
    )
    primary_pass = {
        "clustering_mmd_ratio": clustering_pass,
        "bfs_macro_gs": gs_pass,
        "bfs_macro_rd": rd_pass,
    }
    b0_regimes = evaluate_regime_table(
        labels=b0_universe.label,
        probs=b0_probs,
        u_idx=b0_universe.u_idx,
        v_idx=b0_universe.v_idx,
        node_ids=b0_universe.node_ids,
        g_ref=g_ref,
        store=None,
        f0_cache=output_dir.parent / f".{output_dir.name}.e2e_gate_f0_cache.pt",
        seed=seed,
    )
    b0_auprc = b0_regimes["degree_corrected"]["ratio_1"].auprc
    guards = {
        "degree_mmd_non_regression": full_mmd["degree"] <= 1.10 * b0_row.mmd_ratio["degree"],
        "matched_edge_auprc": cast(float, full["degree_corrected_auprc"]) >= b0_auprc - 0.02,
    }
    f_only = arms["b0_e2e_f_only"]
    pair_topology = arms["pair_topology"]
    f_only_clu = cast(dict[str, float], cast(dict[str, object], f_only["assembled"])["mmd_ratio"])[
        "clustering"
    ]
    pair_topology_clu = cast(
        dict[str, float], cast(dict[str, object], pair_topology["assembled"])["mmd_ratio"]
    )["clustering"]
    gain_full = f_only_clu - full_mmd["clustering"]
    gain_pair_topology = f_only_clu - pair_topology_clu
    primary_all = all(primary_pass.values())
    pathway_applies = primary_all and gain_full > 0.0
    pathway_pass = not pathway_applies or gain_pair_topology >= 0.25 * gain_full
    structure_control = cast(dict[str, object], summary["structure_control"])
    structure_pass = cast(bool, structure_control["passed"])
    verdict = (
        "pass"
        if primary_all and all(guards.values()) and pathway_pass and structure_pass
        else "cut"
    )
    payload: dict[str, object] = {
        "metadata": {
            "registration_sha256": summary["registration_sha256"],
            "evaluation_mode": "single_seed_e2e_screening",
            "single_seed_caveat": (
                "Fixed-seed evaluator-stability evidence only; not significance or cross-seed "
                "robustness."
            ),
        },
        "arms": arms,
        "comparators": comparators,
        "full_matched": full_matched,
        "primary_pass": primary_pass,
        "guards": guards,
        "liveness": summary["liveness"],
        "decomposition": summary["decomposition"],
        "probes": probes,
        "pathway_attribution": {
            "applies": pathway_applies,
            "gain_full": gain_full,
            "gain_pair_topology": gain_pair_topology,
            "passed": pathway_pass,
        },
        "structure_control": structure_control,
        "verdict": verdict,
        "failure_reading": prereg["failure_reading"] if verdict == "cut" else None,
    }
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    (staging_dir / "g5_e2e_stage1_results.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (staging_dir / "g5_e2e_stage1_tables.md").write_text(
        render_e2e_tables_markdown(payload),
        encoding="utf-8",
    )
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    shutil.rmtree(backup_dir, ignore_errors=True)
    try:
        if output_dir.exists():
            os.replace(output_dir, backup_dir)
        os.replace(staging_dir, output_dir)
    except BaseException:
        if not output_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
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
    lines: list[str] = [
        "# G5 Stage-1 single-seed screening gate report",
        "",
        f"**Verdict: `{payload['verdict']}`**",
        "",
        "**Scope:** Binding Stage-1 engineering screen for one fixed seed; this report ",
        "does not claim statistical significance or cross-seed robustness and does not ",
        "replace the E1/E3 multi-seed Holm evaluation.",
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
        "## Registered decision table",
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
    parser.add_argument("--mode", choices=("frozen_s0", "e2e"), default="frozen_s0")
    parser.add_argument(
        "--egostitch-universe",
        type=Path,
        nargs="+",
        default=(),
        help="one candidate-scores .npz per training seed",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        nargs="+",
        default=(),
        help="the matching run_metadata.json per seed (prereg binding)",
    )
    parser.add_argument(
        "--s0-universe",
        type=Path,
        nargs="+",
        default=(),
        help="matching fp32 frozen-B0 candidate artifact per EgoStitch seed",
    )
    parser.add_argument("--b0-universe", type=Path)
    parser.add_argument("--b0cal-results", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fidelity-report",
        type=Path,
        nargs="+",
        default=(),
        help=("required per-seed fidelity report for the binding Stage-1 screen"),
    )
    parser.add_argument("--full-universe", type=Path)
    parser.add_argument("--control-universe", type=Path)
    parser.add_argument("--fonly-universe", type=Path)
    parser.add_argument("--pt-universe", type=Path)
    parser.add_argument("--p0-universe", type=Path)
    parser.add_argument("--probe-artifact", type=Path)
    parser.add_argument(
        "--cost-report",
        type=Path,
        default=None,
        help="required cost report for the binding Stage-1 screen",
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
    if args.mode == "e2e":
        arm_paths = {
            "full": args.full_universe,
            "structure_control_6a": args.control_universe,
            "b0_e2e_f_only": args.fonly_universe,
            "pair_topology": args.pt_universe,
            "p0": args.p0_universe,
        }
        required = {
            **arm_paths,
            "b0_universe": args.b0_universe,
            "b0cal_results": args.b0cal_results,
            "probe_artifact": args.probe_artifact,
            "preregistration": args.preregistration,
            "output_dir": args.output_dir,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            parser.error(f"--mode e2e requires: {', '.join(missing)}")
        if len(args.run_metadata) != 4:
            parser.error("--mode e2e requires exactly four --run-metadata paths")
        for path in [
            *cast(dict[str, Path], arm_paths).values(),
            *args.run_metadata,
            cast(Path, args.b0_universe),
            cast(Path, args.b0cal_results),
            cast(Path, args.probe_artifact),
            cast(Path, args.preregistration),
        ]:
            if not path.exists():
                parser.error(f"input not found: {path}")
        run_g5_e2e_stage1_pipeline(
            arm_universe_paths=cast(dict[str, Path], arm_paths),
            run_metadata_paths=dict(zip(_E2E_FORMAL_ARMS, args.run_metadata, strict=True)),
            b0_universe_path=cast(Path, args.b0_universe),
            b0cal_results_path=cast(Path, args.b0cal_results),
            probe_artifact_path=cast(Path, args.probe_artifact),
            preregistration_path=cast(Path, args.preregistration),
            data_root=args.data_root,
            strategy=args.strategy,
            output_dir=cast(Path, args.output_dir),
            seed=args.seed,
        )
        return
    frozen_required = {
        "egostitch_universe": args.egostitch_universe,
        "run_metadata": args.run_metadata,
        "s0_universe": args.s0_universe,
        "b0_universe": args.b0_universe,
        "b0cal_results": args.b0cal_results,
        "preregistration": args.preregistration,
        "output_dir": args.output_dir,
    }
    missing_frozen = [name for name, value in frozen_required.items() if not value]
    if missing_frozen:
        parser.error(f"--mode frozen_s0 requires: {', '.join(missing_frozen)}")
    for path in [
        *args.egostitch_universe,
        *args.run_metadata,
        *args.s0_universe,
        cast(Path, args.b0_universe),
        cast(Path, args.b0cal_results),
        cast(Path, args.preregistration),
        *args.fidelity_report,
    ]:
        if not path.exists():
            parser.error(f"input not found: {path}")
    if args.cost_report is not None and not args.cost_report.exists():
        parser.error(f"input not found: {args.cost_report}")

    run_g5_stage1_pipeline(
        egostitch_universe_paths=args.egostitch_universe,
        s0_universe_paths=args.s0_universe,
        run_metadata_paths=args.run_metadata,
        b0_universe_path=cast(Path, args.b0_universe),
        b0cal_results_path=cast(Path, args.b0cal_results),
        preregistration_path=cast(Path, args.preregistration),
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=cast(Path, args.output_dir),
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
    "PreregistrationNotBinding",
    "PreregistrationMismatch",
    "RegistrationShaMismatch",
    "build_e2e_arm_summary",
    "clustering_criterion",
    "enforce_preregistration",
    "holm_step_down",
    "matched_rd_criterion",
    "paired_bootstrap_lower_bound",
    "render_tables_markdown",
    "run_g5_stage1_pipeline",
    "run_g5_e2e_stage1_pipeline",
    "validate_dead_residual_within_checkpoint",
]
