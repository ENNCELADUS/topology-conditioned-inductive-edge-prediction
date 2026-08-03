r"""G5 E2E Stage-1 gate evaluation: the seven-arm EgoStitch screen.

Evaluates one training seed of the ``egostitch_e2e`` family against the frozen
comparators (B0 recomputed from its cached artifact; the B0+cal arms from the
committed kill-test payload) under the Stage-1 decision rules (protocol
Sec 5.0.5/5.2):

- **Seven arms**: the five trained checkpoint arms plus the two mandatory
  scoring-time controls ``structure_control_6a_v3`` (shuffle-within-pair) and
  ``structure_control_6e_v1`` (degree-preserving rewiring). Neither control is
  optional; both reuse the ``full`` checkpoint. (The ``pair_topology`` arm was
  retired with the content-path removal: with no content branch, a
  topology-only arm is identical to ``full`` and pathway isolation between the
  two is no longer computable — see ``pathway_attribution`` below.)
- **Primary family** (point-estimate dominance, all must pass): clustering-MMD
  ratio at the canonical operating point; BFS-macro GS and RD at matched global
  simple-edge RD (per-comparator deterministic exact-quota re-assembly).
- **Guards**: degree-MMD non-regression (<= 1.10x B0); matched edge AUPRC
  (degree-corrected ratio-1, within 0.02 of B0).
- **Evidence class**: always ``"engineering"``, at every seed count. Only E1/E3
  carry inference (CLAUDE.md; protocol Sec 5.0.5), so ``p_value``/``ci``/
  ``holm`` are written null and the artifact is refused if they are not.
- **Verdict**: the seed yields ``"pass"`` or ``"cut"``.

CLI::

    python -m src.experiments.g5_stage1 \
        --full-universe full.npz --fonly-universe f_only.npz \
        --p0-universe p0.npz \
        --no-l-rel-universe no_l_rel.npz --row-layernorm-universe row_layernorm.npz \
        --control-6a-universe control_6a.npz --control-6e-universe control_6e.npz \
        --run-metadata run_full.json ... \
        --b0-universe scores/b0_v31_candidate.npz \
        --b0cal-results outputs/b0_cal/b0cal_results.json \
        --probe-artifact outputs/e2e_probe/full_probe.json \
        --data-root data --strategy breadth_first --output-dir outputs/g5_e2e_stage1

The five ``--run-metadata`` paths are positional in trained-arm order
(``full``, ``b0_e2e_f_only``, ``p0``, ``no_l_rel``, ``row_layernorm``).

Determinism: identical inputs produce byte-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
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
    _expected_candidate_rows,
    assemble_and_evaluate,
    evaluate_regime_table,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.experiments.probes import (
    _load_train_side_probe_inputs,
    evaluate_e2e_probe_artifact,
)
from src.score_universe import (
    _SCAFFOLD_CONTROL_NONE,
    _SCAFFOLD_CONTROL_REWIRE_V1,
    _SCAFFOLD_CONTROL_SHUFFLE_V3,
    ScoresArtifact,
    load_scores,
    validate_artifact_precision,
)

logger = logging.getLogger(__name__)

_COMPARATORS: tuple[str, ...] = ("b0", "b0_cal_density", "b0_cal_selfdensity", "b0_cal_degseq")

#: This ladder is engineering evidence at every seed count (CLAUDE.md; protocol
#: Sec 5.0.5 — only E1/E3 carry inference), so every inferential field must be
#: written null at every nesting depth of the emitted artifact.
_EVIDENCE_CLASS = "engineering"
_INFERENCE_FIELDS: frozenset[str] = frozenset(
    {
        "ci",
        "ci_excludes_zero",
        "holm",
        "holm_alpha",
        "holm_survives",
        "p_value",
        "p_values",
    }
)


class EvidenceClassViolation(RuntimeError):
    """Raised when a screen artifact would carry or claim inferential evidence."""


@dataclass(frozen=True)
class MatchedRdSelection:
    """Exact-quota selection for one matched-global-RD row."""

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


def select_matched_global_rd_rows(
    probs: np.ndarray,
    u_idx: NDArray[np.int32],
    v_idx: NDArray[np.int32],
    *,
    target_edges: int,
    reference_edges: int,
    tolerance: float = 0.005,
) -> MatchedRdSelection:
    """Select an exact non-self quota with the canonical tie-break."""
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


def validate_dead_residual_within_checkpoint(
    artifact: ScoresArtifact,
    *,
    min_residual_std_ratio: float,
    max_spearman: float,
    max_topk_overlap: float,
    topk_fraction: float,
) -> dict[str, float]:
    """Report whether a checkpoint's own topology/content pathway carries signal.

    Family ``egostitch_e2e`` liveness (spec Sec 13.17/13.8) references the
    **within-checkpoint** ``f_logit`` arm of the SAME scored artifact rather
    than a separate comparator artifact: ``full`` and ``f_logit`` already
    share row order by construction (one artifact, one scoring pass over the
    same pair universe), so no cross-artifact pair alignment step exists or is
    needed.

    Args:
        artifact: A loaded ``egostitch_e2e`` scores artifact; its ``f_logit``
            array must be present.
        min_residual_std_ratio: Lower bound on
            ``std(full - f_logit) / std(f_logit)``.
        max_spearman: Upper bound on ``Spearman(full, f_logit)``.
        max_topk_overlap: Upper bound on the top-``topk_fraction`` overlap
            between ``full`` and ``f_logit``.
        topk_fraction: Top fraction used for the overlap signal (spec pins
            ``0.01`` for this family).

    Returns:
        A diagnostics dict (std ratio, correlation, overlap, and their raw
        ingredients).

    Missing plan-required score arrays remain an artifact-integrity error. The
    finite liveness thresholds themselves are telemetry and never authorize or
    block evaluation.
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
    report["quality_pass"] = not (
        residual_ratio < min_residual_std_ratio
        and correlation > max_spearman
        and overlap > max_topk_overlap
    )
    return report


def _training_diagnostics(
    run_metadata: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return training-side quality telemetry without gating the evaluator."""
    reports: list[dict[str, object]] = []
    for metadata in run_metadata:
        report = cast(dict[str, object] | None, metadata.get("training_diagnostics"))
        reports.append(report if isinstance(report, dict) else {})
    return reports


def _vhold_validation_event_count(
    path: Path,
    *,
    label: str,
    arm: str,
) -> int:
    """Validate and count one worker's explicit ``V_hold`` event ledger."""
    rows: list[object] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{label}: validation-event row {line_number} is blank")
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: validation-event evidence is unreadable: {path}") from error
    if not rows:
        raise ValueError(f"{label}: validation-event evidence contains no V_hold evaluations")
    for expected_ordinal, row in enumerate(rows, start=1):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != expected_ordinal
            or row.get("validation_role") != "V_hold"
            or row.get("arm") != arm
            or row.get("kind") not in {"step_0", "phase_a_end", "epoch_end"}
            or isinstance(row.get("optimizer_step"), bool)
            or not isinstance(row.get("optimizer_step"), int)
            or cast(int, row["optimizer_step"]) < 0
        ):
            raise ValueError(
                f"{label}: validation-event row {expected_ordinal} is not a verified V_hold "
                "validation result"
            )
    first = cast(Mapping[str, object], rows[0])
    if first.get("kind") != "step_0" or first.get("optimizer_step") != 0:
        raise ValueError(f"{label}: first V_hold validation event must be step_0 at step 0")
    return len(rows)


def _validate_vhold_event_ledgers(
    run_metadata: Mapping[str, Mapping[str, object]],
    *,
    run_metadata_paths: Mapping[str, Path],
    trained_arms: Sequence[str],
) -> None:
    """Fail closed when a trained arm's V_hold validation-event ledger is missing or tampered.

    This is a data-boundary evidence check, not registration: each trained
    arm's ``run_metadata.json`` binds the hash and row count of its own
    append-only ``V_hold`` validation-event ledger
    (:func:`_vhold_validation_event_count`), and this walks every trained arm
    to enforce that binding before any held-out metric is reported.
    """
    for arm in trained_arms:
        metadata_path = run_metadata_paths[arm]
        metadata = run_metadata[arm]
        evidence = metadata.get("v_hold_validation_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{arm}: missing validation-event evidence")
        events_path = metadata_path.parent / str(evidence.get("path", ""))
        count = _vhold_validation_event_count(events_path, label=arm, arm=arm)
        if evidence.get("count") != count or evidence.get("sha256") != _sha256_file(events_path):
            raise ValueError(f"{arm}: run metadata does not bind its validation-event evidence")


def _assert_shared_training_seed(
    run_metadata: Mapping[str, Mapping[str, object]], trained_arms: Sequence[str]
) -> int:
    """Verify one gate invocation covers exactly one training seed.

    Each trained arm's own ``run_metadata.json`` records its ``seed``; a
    mismatch would silently blend checkpoints trained under different seeds
    into one screen.

    Returns:
        The one shared training seed, for callers to record rather than
        re-derive.
    """
    seeds = {arm: run_metadata[arm].get("seed") for arm in trained_arms}
    distinct = set(seeds.values())
    if len(distinct) != 1 or any(
        not isinstance(value, int) or isinstance(value, bool) for value in distinct
    ):
        raise ValueError(
            "one E2E gate invocation screens exactly one training seed; the supplied "
            f"trained-arm run metadata disagree: {seeds}"
        )
    return cast(int, next(iter(distinct)))


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


# --------------------------------------------------------------------------- e2e seven-arm summary

#: The five trained checkpoint arms this gate screens. ``pair_topology`` was
#: retired with the content-path removal (three-component refactor P1): with
#: no content branch, a topology-only checkpoint arm is identical to ``full``
#: and cannot isolate a topology-only pathway.
_TRAINED_ARMS: tuple[str, ...] = (
    "full",
    "b0_e2e_f_only",
    "p0",
    "no_l_rel",
    "row_layernorm",
)
#: The two scoring-time controls; both reuse the ``full`` checkpoint.
_CONTROL_ARMS: tuple[str, ...] = (
    "structure_control_6a_v3",
    "structure_control_6e_v1",
)
#: The complete seven-arm set this gate reports.
_ALL_ARMS: tuple[str, ...] = _TRAINED_ARMS + _CONTROL_ARMS
#: Each control arm's required `meta["scaffold_control"]["mode"]`, matching
#: `score_universe.py`'s own `_SCAFFOLD_CONTROL_ARM_NAMES` mapping in the
#: other direction: `structure_control_6a_v3` is the shuffle-within-pair
#: control, `structure_control_6e_v1` is the degree-preserving
#: (checkerboard) rewiring control.
_CONTROL_ARM_SCAFFOLD_MODE: dict[str, str] = {
    "structure_control_6a_v3": _SCAFFOLD_CONTROL_SHUFFLE_V3,
    "structure_control_6e_v1": _SCAFFOLD_CONTROL_REWIRE_V1,
}


def _artifact_scaffold_control_mode(artifact: ScoresArtifact, *, label: str) -> str:
    """Extract and validate an artifact's own recorded scaffold-control mode.

    Fails closed: a missing or malformed ``scaffold_control`` block cannot be
    trusted to mean "no perturbation" -- that would let a mislabeled or
    corrupted artifact silently pass as either arm kind.
    """
    scaffold_control = artifact.meta.get("scaffold_control")
    if not isinstance(scaffold_control, Mapping):
        raise ValueError(f"{label}: artifact is missing scaffold_control metadata")
    mode = scaffold_control.get("mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{label}: artifact scaffold_control.mode is missing or invalid")
    return mode


def _validate_scaffold_control_arm_semantics(
    artifact: ScoresArtifact, *, name: str, label: str
) -> None:
    """Assert an artifact's own recorded scaffold-control mode matches its arm slot.

    Both mandatory scaffold-structure controls intentionally reuse the
    ``full`` checkpoint, so the checkpoint_id check alone cannot distinguish
    them from `full` or from each other -- swapping the two control NPZ
    paths, or passing an unperturbed `full` artifact as a control, would
    otherwise pass validation and silently change the structure-control
    verdict. This check is independent of any registration: it compares only
    the artifact's own recorded ``scaffold_control`` metadata against the arm
    name `name` it was supplied under.
    """
    mode = _artifact_scaffold_control_mode(artifact, label=label)
    expected = _CONTROL_ARM_SCAFFOLD_MODE.get(name, _SCAFFOLD_CONTROL_NONE)
    if mode != expected:
        raise ValueError(
            f"{label}: arm {name!r} requires scaffold_control.mode {expected!r}, got {mode!r}"
        )


def _validate_e2e_universe_shape(
    artifact: ScoresArtifact, *, strategy: str, n_test_nodes: int, label: str
) -> None:
    """Structural candidate-universe checks for family ``egostitch_e2e`` artifacts.

    Mirrors :func:`src.experiments.g1_hardened_e2.validate_universe_artifact`'s
    non-precision checks (``pairs_source``/``strategy``/row-count/label-domain)
    WITHOUT its internal ``validate_score_precision(artifact.logit, meta=...)``
    call: that raw idiom always reports the ``f_logit`` array as "missing"
    for this family, because it never forwards it as an ``extra_arrays``
    entry (the exact footgun :func:`validate_artifact_precision` exists to
    avoid — Task 14 review finding). Precision is validated separately, via
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


def _load_candidate_manifest(
    benchmark_root: Path, strategy: str
) -> tuple[list[tuple[str, str]], NDArray[np.int8]]:
    """Load the candidate pair/label manifest every arm artifact must share.

    This is the pair-order and label identity every arm is checked against in
    :func:`build_e2e_arm_summary` — genuine cross-artifact correctness, not
    registration binding.
    """
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


def _artifact_two_logits(artifact: ScoresArtifact) -> dict[str, NDArray[np.float32]]:
    """Return the true aligned two-logit arrays regardless of active primary arm.

    Post-content-path (three-component refactor P1), ``decompose()`` returns
    exactly ``{full, f_logit}`` — the ``pair_content``/``pair_topology``
    arrays no longer exist anywhere in the pipeline.
    """
    primary = artifact.meta.get("primary_logit")
    full = artifact.logit if primary == "full" else artifact.full_logit
    if full is None or artifact.f_logit is None:
        raise ValueError("E2E artifact is missing its complete two-logit decomposition")
    return {
        "full": full,
        "f_logit": artifact.f_logit,
    }


def _e2e_decomposition_summary(
    artifacts: Mapping[str, ScoresArtifact],
) -> dict[str, object]:
    """Complete per-arm two-logit and aligned-delta evidence."""
    arrays = {name: _artifact_two_logits(artifact) for name, artifact in artifacts.items()}
    arm_rows: dict[str, object] = {}
    for name, values in arrays.items():
        arm_rows[name] = {
            "logits": {key: _array_summary(array) for key, array in values.items()},
            "deltas": {
                "full_minus_f_logit": _array_summary(values["full"] - values["f_logit"]),
            },
        }
    full = arrays["full"]
    p0 = arrays["p0"]
    return {
        "arms": arm_rows,
        "cross_arm": {
            "p0_minus_full": {key: _array_summary(p0[key] - full[key]) for key in full},
        },
    }


def _evaluate_e2e_probe(
    *,
    probe_artifact_path: Path,
    run_metadata_path: Path,
    data_root: Path,
    strategy: str,
) -> dict[str, object]:
    """Rebuild the full arm's ``G_struct`` identity and evaluate its probe artifact."""
    from src import train_egostitch as te
    from src.data.partition import build_g_struct, derive_partition
    from src.model.egostitch.config import E2EConfig

    metadata = cast(dict[str, object], json.loads(run_metadata_path.read_text(encoding="utf-8")))
    config_path = Path(str(metadata.get("config_path")))
    cfg = te.load_config(config_path)
    if cfg.data.root.resolve() != data_root.resolve() or cfg.data.strategy != strategy:
        raise ValueError("full-arm config data identity does not match gate inputs")
    train_nodes, train_positives = _load_train_side_probe_inputs(
        cfg.data.root / _BENCHMARK_SUBDIR / cfg.data.strategy,
        expected_missing_features=cfg.data.expected_missing_features,
    )
    partition = derive_partition(
        train_positives,
        seed=cfg.data.partition_seed,
        msg_fraction=cfg.data.msg_fraction,
    )
    graph = build_g_struct(train_nodes, partition.e_msg)
    # Three-component refactor (design 2026-08-02 Sec 8): `n_ground` moved
    # from the flat `E2EConfig` onto the nested `generator` sub-config.
    n_ground = E2EConfig.from_mapping(cfg.model.config).generator.n_ground
    return evaluate_e2e_probe_artifact(
        probe_artifact_path,
        graph=graph,
        train_nodes=train_nodes,
        expected_metadata={
            "seed": metadata.get("seed"),
            "partition_seed": cfg.data.partition_seed,
            "strategy": strategy,
            "n_ground": n_ground,
        },
    )


def build_e2e_arm_summary(
    *,
    arm_universe_paths: Mapping[str, Path],
    run_metadata_paths: Mapping[str, Path],
    data_root: Path,
    strategy: str,
    liveness_config: Mapping[str, float],
    seed: int = 0,
) -> dict[str, object]:
    """Build the seven-arm ``egostitch_e2e`` Stage-1 summary table.

    Loads and validates every arm's scores artifact through
    :func:`validate_artifact_precision` (the artifact-aware entry point — never
    the raw ``validate_score_precision(artifact.logit, meta=...)`` idiom, which
    silently drops the ``f_logit`` array), then reports each arm's
    canonical-operating-point assembled metrics plus the ``full`` arm's
    within-checkpoint liveness report. Every arm artifact and every trained
    arm's run metadata is loaded and validated in one place here. This
    function does not itself compute the structure-control decision rule
    (spec Sec 14, ``e2e_rules``) — that is applied by
    :func:`run_g5_e2e_stage1_pipeline`, which consumes this table's per-arm
    ``clustering_mmd_ratio`` values plus (for the structure-control
    condition) the paired-bootstrap lower bound computed below. Pathway
    isolation between the topology-only and content-only pathways is no
    longer computable at all (the content path, and its ``pair_topology``
    arm, were removed by the three-component refactor P1) — see
    :func:`run_g5_e2e_stage1_pipeline`'s ``pathway_attribution`` payload.

    Args:
        arm_universe_paths: Exact five-trained-plus-two-control mapping to
            scored ``.npz`` artifacts.
        run_metadata_paths: Trained-arm name -> its ``run_metadata.json``.
            Scoring-time controls never have entries.
        data_root: Directory containing the benchmark package.
        strategy: Benchmark split strategy.
        liveness_config: The ``min_residual_std_ratio`` / ``max_spearman`` /
            ``max_topk_overlap`` / ``topk_fraction`` liveness thresholds (spec
            Sec 13.17, family ``egostitch_e2e``).
        seed: Bootstrap/evaluation seed for the assembled-graph metrics.

    Returns:
        A JSON-ready payload: a per-arm ``checkpoint_id`` / ``assembled`` /
        ``degree_corrected_auprc`` row, the shared ``training_seed`` all five
        trained arms were trained under, and the ``full`` arm's
        within-checkpoint ``liveness`` report.

    Raises:
        ValueError: On an invalid seven-arm composition, invalid trained-arm
            run metadata, mismatched scoring checkpoint identities, an
            artifact whose ``model_family`` is not ``egostitch_e2e``, or an
            artifact that fails its candidate-universe shape check or
            `validate_artifact_precision`.
    """
    unknown = set(arm_universe_paths) - set(_ALL_ARMS)
    if unknown:
        raise ValueError(f"unrecognized e2e arm(s): {sorted(unknown)}")
    if "full" not in arm_universe_paths:
        raise ValueError("the 'full' arm is required to build the e2e arm summary")
    if set(arm_universe_paths) != set(_ALL_ARMS):
        raise ValueError(
            "e2e summary requires exactly the seven registered arms "
            f"{list(_ALL_ARMS)}, got {sorted(arm_universe_paths)}"
        )
    if set(run_metadata_paths) != set(_TRAINED_ARMS):
        raise ValueError(
            "e2e summary requires exactly the five trained run metadata records "
            f"{sorted(_TRAINED_ARMS)}, got {sorted(run_metadata_paths)}"
        )

    run_metadata: dict[str, dict[str, object]] = {
        name: cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for name, path in run_metadata_paths.items()
    }
    training_seed = _assert_shared_training_seed(run_metadata, _TRAINED_ARMS)
    training_diagnostics = _training_diagnostics([run_metadata[name] for name in _TRAINED_ARMS])
    _validate_vhold_event_ledgers(
        run_metadata,
        run_metadata_paths=run_metadata_paths,
        trained_arms=_TRAINED_ARMS,
    )

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
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    if strategy == "breadth_first" and sum(len(node_sets) for node_sets in buckets.values()) != 500:
        raise ValueError("the fixed evaluator must contain exactly 500 subgraphs")
    n_test_nodes = g_ref.number_of_nodes()
    target_edges = strip_self_loops(g_ref).number_of_edges()
    nodes = list(g_ref.nodes())
    config = MMDConfig()
    candidate_pairs, candidate_labels = _load_candidate_manifest(benchmark_root, strategy)

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
        expected_checkpoint_arm = "full" if name in _CONTROL_ARMS else name
        metadata_checkpoint = run_metadata[expected_checkpoint_arm].get("checkpoint_id")
        artifact_checkpoint = artifact.meta.get("checkpoint_id")
        if artifact_checkpoint != metadata_checkpoint:
            raise ValueError(
                f"{name} checkpoint_id mismatch: "
                f"{metadata_checkpoint!r} != {artifact_checkpoint!r}"
            )
        _validate_scaffold_control_arm_semantics(artifact, name=name, label=label)
        if list(artifact.pairs()) != candidate_pairs:
            raise ValueError(
                f"{label}: candidate pair identity/order does not match the benchmark manifest"
            )
        if not np.array_equal(artifact.label, candidate_labels):
            raise ValueError(f"{label}: candidate labels do not match the benchmark manifest")
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
            "assembled": _assembled_row_to_dict(assembled),
            "degree_corrected_auprc": regimes["degree_corrected"]["ratio_1"].auprc,
        }

    full_checkpoint = artifacts["full"].meta.get("checkpoint_id")
    for control_name in _CONTROL_ARMS:
        control_checkpoint = artifacts[control_name].meta.get("checkpoint_id")
        if control_checkpoint != full_checkpoint:
            raise ValueError(
                f"{control_name} checkpoint_id must match full scoring checkpoint: "
                f"{control_checkpoint!r} != {full_checkpoint!r}"
            )

    lower_bound = paired_bootstrap_lower_bound(
        lambda samples: _clustering_mmd_ratio(
            cast(Sequence[object] | Mapping[object, Sequence[object]], samples), config
        ),
        clustering_samples["structure_control_6a_v3"],
        clustering_samples["full"],
        n_boot=1000,
        seed=0,
        alpha=0.05,
    )
    decomposition = _e2e_decomposition_summary(artifacts)

    return {
        "arms": rows,
        "training_seed": training_seed,
        "training_diagnostics": dict(zip(_TRAINED_ARMS, training_diagnostics, strict=True)),
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

_EVIDENCE_CLASS_NOTE = (
    "Engineering evidence at every seed count. Only E1/E3 carry inference "
    "(CLAUDE.md; protocol Sec 5.0.5), so p_value/ci/holm are null by "
    "construction — never significance or cross-seed robustness."
)


def _enforce_engineering_evidence_class(payload: Mapping[str, object]) -> None:
    """Refuse any artifact that would carry inference out of this screen.

    Args:
        payload: The complete JSON-ready gate payload, before it is written.

    Raises:
        EvidenceClassViolation: If ``evidence_class`` is anything other than
            ``"engineering"``, or if any inferential field (`_INFERENCE_FIELDS`)
            is non-null at any nesting depth. The walk is recursive precisely so
            a nested inferential field cannot slip past a top-level check.
    """
    if payload.get("evidence_class") != _EVIDENCE_CLASS:
        raise EvidenceClassViolation(
            f"the G5 E2E ladder emits evidence_class {_EVIDENCE_CLASS!r} at every seed "
            f"count; got {payload.get('evidence_class')!r}"
        )

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key) in _INFERENCE_FIELDS and value is not None:
                    raise EvidenceClassViolation(
                        f"{child} must be null in an {_EVIDENCE_CLASS}-class artifact; "
                        f"got {value!r}"
                    )
                walk(value, child)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "")


def render_e2e_tables_markdown(payload: Mapping[str, object]) -> str:
    """Render the complete seven-arm, decomposition, and probe evidence tables."""
    arms = cast(Mapping[str, Mapping[str, object]], payload["arms"])
    lines = [
        "# G5 E2E Stage-1 gate",
        "",
        f"**Verdict: `{payload['verdict']}`**",
        "",
        f"**Evidence class: `{payload['evidence_class']}`**",
        "",
        "Fixed-seed evaluator-stability evidence only; not significance or cross-seed robustness.",
        "",
        "## Seven-arm summary",
        "",
        "| arm | checkpoint | GS | BFS-macro RD | degree MMD | clustering MMD | "
        "spectral MMD | degree-corrected AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _ALL_ARMS:
        row = arms[name]
        assembled = cast(Mapping[str, object], row["assembled"])
        mmd = cast(Mapping[str, float], assembled["mmd_ratio"])
        lines.append(
            f"| {name} | {row['checkpoint_id']} | {_fmt(assembled['graph_similarity'])} "
            f"| {_fmt(assembled['relative_density'])} | {_fmt(mmd['degree'])} "
            f"| {_fmt(mmd['clustering'])} | {_fmt(mmd['spectral'])} "
            f"| {_fmt(row['degree_corrected_auprc'])} |"
        )
    training_diagnostics = cast(
        Mapping[str, Mapping[str, object]], payload.get("training_diagnostics", {})
    )
    lines += [
        "",
        "## Degree-decorrelation telemetry",
        "",
        "| arm | validation | corr(full-f_logit, endpoint degree) |",
        "|---|---:|---:|",
    ]
    for name in _TRAINED_ARMS:
        report = training_diagnostics.get(name, {})
        fidelity_series = cast(
            Sequence[Mapping[str, object]], report.get("fidelity_series", ())
        )
        if not fidelity_series:
            lines.append(f"| {name} |  |  |")
            continue
        for validation, row in enumerate(fidelity_series, start=1):
            lines.append(
                f"| {name} | {validation} "
                f"| {_fmt(row.get('topology_delta_degree_correlation'))} |"
            )
    decomposition = cast(Mapping[str, object], payload["decomposition"])
    decomposition_arms = cast(Mapping[str, Mapping[str, object]], decomposition["arms"])
    lines += [
        "",
        "## Logit decomposition deltas",
        "",
        "| arm | full-f mean | full-f std |",
        "|---|---:|---:|",
    ]
    for name in _ALL_ARMS:
        deltas = cast(Mapping[str, Mapping[str, float]], decomposition_arms[name]["deltas"])
        f_delta = deltas["full_minus_f_logit"]
        lines.append(f"| {name} | {_fmt(f_delta['mean'])} | {_fmt(f_delta['std'])} |")
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
        f"| E_msg selection | {_fmt(pi['mean'])} | {_fmt(pi['std'])} "
        f"| {_fmt(pi['nonzero_fraction'])} | {_fmt(pi['n_pairs'])} |",
        "",
        "## Probe-v2 evidence",
        "",
        "| quantity | value | std | nonzero fraction | n | context |",
        "|---|---:|---:|---:|---:|---|",
    ]
    pi_v2 = cast(Mapping[str, object], probes["pi_consistency_v2"])
    lines.append(
        f"| pi_consistency_v2 | {_fmt(pi_v2['mean'])} | {_fmt(pi_v2['std'])} "
        f"| {_fmt(pi_v2['nonzero_fraction'])} | {_fmt(pi_v2['n'])} "
        f"| n_pairs={_fmt(pi_v2['n_pairs'])} |"
    )
    slot_recall = cast(Mapping[str, object], probes["slot_recall_at_n_ground"])
    lines.append(
        f"| slot_recall_at_n_ground | {_fmt(slot_recall['mean'])} "
        f"| {_fmt(slot_recall['std'])} | {_fmt(slot_recall['nonzero_fraction'])} "
        f"| {_fmt(slot_recall['n'])} | n_ground={_fmt(slot_recall['n_ground'])}; "
        f"n_nodes={_fmt(slot_recall['n_nodes'])} |"
    )
    lines.append(
        f"| shared_neighbor_count_r2 | {_fmt(probes['shared_neighbor_count_r2'])} "
        "|  |  |  | pair-state ridge R2 |"
    )
    dispersion = cast(Mapping[str, Mapping[str, object]], probes["dispersion"])
    for name in (
        "pi_slot_std",
        "h_pairwise_cosine_mean",
        "adj_offdiag_std",
        "plan_row_entropy",
    ):
        row = dispersion[name]
        lines.append(
            f"| {name} | {_fmt(row['mean'])} | {_fmt(row['std'])} "
            f"| {_fmt(row['nonzero_fraction'])} | {_fmt(row['n'])} | pair dispersion |"
        )
    lines += [
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
    data_root: Path,
    strategy: str,
    output_dir: Path,
    seed: int = 0,
) -> dict[str, object]:
    """Run the binding E2E seven-arm G5 gate and write its verdict artifacts.

    One invocation screens one training seed over all seven arms (five
    trained checkpoints plus the two mandatory scoring-time controls). The
    emitted artifact is always ``evidence_class: "engineering"`` and is
    refused outright if any inferential field survives
    (`_enforce_engineering_evidence_class`).
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # A rejected run must not leave an older verdict looking authoritative.
    for filename in ("g5_e2e_stage1_results.json", "g5_e2e_stage1_tables.md"):
        (output_dir / filename).unlink(missing_ok=True)
    b0cal_results_path = _resolve_b0cal_results_path(b0cal_results_path)
    probes_report = _evaluate_e2e_probe(
        probe_artifact_path=probe_artifact_path,
        run_metadata_path=run_metadata_paths["full"],
        data_root=data_root,
        strategy=strategy,
    )
    summary = build_e2e_arm_summary(
        arm_universe_paths=arm_universe_paths,
        run_metadata_paths=run_metadata_paths,
        data_root=data_root,
        strategy=strategy,
        liveness_config=_E2E_LIVENESS_CONFIG,
        seed=seed,
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
    training_seed = cast(int, summary["training_seed"])
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
    primary_all = all(primary_pass.values())
    structure_control = cast(dict[str, object], summary["structure_control"])
    structure_pass = cast(bool, structure_control["passed"])
    # Pathway isolation ("gain_pair_topology >= 0.25 * gain_full") is retired,
    # not silently dropped: it required the trained `pair_topology` arm, which
    # the three-component refactor's content-path removal (P1) eliminated —
    # with no content branch, a topology-only checkpoint is identical to
    # `full` and has no numerator to compute a gain against. The verdict rule
    # below no longer has a pathway-isolation term; `pathway_attribution`
    # below records that explicitly (`evaluated: False`) so a reader cannot
    # mistake its absence for a pass.
    verdict = "pass" if primary_all and all(guards.values()) and structure_pass else "cut"
    payload: dict[str, object] = {
        "metadata": {
            "evaluation_mode": "single_seed_e2e_screening",
            "arms_screened": list(_ALL_ARMS),
            "evidence_class_note": _EVIDENCE_CLASS_NOTE,
            "single_seed_caveat": (
                "Fixed-seed evaluator-stability evidence only; not significance or cross-seed "
                "robustness."
            ),
            # The one training seed all five trained arms share
            # (`_assert_shared_training_seed`) and this invocation's
            # `--seed`, which drives bootstrap/regime-table sampling here.
            # Neither identifies the other: a result artifact must be able
            # to name both its training run and its evaluation random
            # stream.
            "training_seed": training_seed,
            "seed": seed,
        },
        "evidence_class": _EVIDENCE_CLASS,
        "p_value": None,
        "ci": None,
        "holm": None,
        "arms": arms,
        "comparators": comparators,
        "full_matched": full_matched,
        "primary_pass": primary_pass,
        "guards": guards,
        "liveness": summary["liveness"],
        "training_diagnostics": summary["training_diagnostics"],
        "decomposition": summary["decomposition"],
        "probes": probes_report,
        # Pathway isolation between the topology-only and content-only
        # pathways is NOT evaluated in this schema (`evaluated: False`) —
        # not a pass, not a skip-as-vacuous-pass. The `pair_topology` arm
        # that supplied `gain_pair_topology`'s numerator no longer exists:
        # the three-component refactor's content-path removal (P1) means a
        # topology-only checkpoint would be identical to `full`, so there is
        # nothing left to isolate against. See `_TRAINED_ARMS`.
        "pathway_attribution": {
            "evaluated": False,
            "reason": (
                "the 'pair_topology' arm was removed with the content path "
                "(three-component refactor P1); with no content branch, a "
                "topology-only checkpoint is identical to 'full' and "
                "'gain_pair_topology' has no numerator to compute — pathway "
                "isolation between the topology-only and content-only "
                "pathways is not computable in this schema, and this field "
                "does not gate the verdict"
            ),
        },
        "structure_control": structure_control,
        "verdict": verdict,
    }
    # Refused before anything is written: an inference-bearing screen artifact
    # must not exist on disk even transiently.
    _enforce_engineering_evidence_class(payload)
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


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``g5_stage1`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.g5_stage1",
        description="G5 E2E Stage-1 seven-arm gate evaluation.",
    )
    # The legacy frozen-s0 ``egostitch`` family and its ``--mode frozen_s0``
    # pipeline are gone; the flag survives only so an explicit ``--mode e2e``
    # keeps working and any other value is rejected loudly rather than silently
    # falling through to a pipeline that no longer exists.
    parser.add_argument("--mode", choices=("e2e",), default="e2e")
    parser.add_argument(
        "--run-metadata",
        type=Path,
        nargs="+",
        default=(),
        help="one run_metadata.json per trained arm, in trained-arm order",
    )
    parser.add_argument("--b0-universe", type=Path)
    parser.add_argument("--b0cal-results", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-universe", type=Path)
    parser.add_argument("--control-6a-universe", type=Path)
    parser.add_argument("--control-6e-universe", type=Path)
    parser.add_argument("--fonly-universe", type=Path)
    parser.add_argument("--p0-universe", type=Path)
    parser.add_argument("--no-l-rel-universe", type=Path)
    parser.add_argument("--row-layernorm-universe", type=Path)
    parser.add_argument("--probe-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argument errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    arm_paths = {
        "full": args.full_universe,
        "b0_e2e_f_only": args.fonly_universe,
        "p0": args.p0_universe,
        "no_l_rel": args.no_l_rel_universe,
        "row_layernorm": args.row_layernorm_universe,
        "structure_control_6a_v3": args.control_6a_universe,
        "structure_control_6e_v1": args.control_6e_universe,
    }
    required = {
        **arm_paths,
        "b0_universe": args.b0_universe,
        "b0cal_results": args.b0cal_results,
        "probe_artifact": args.probe_artifact,
        "output_dir": args.output_dir,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        parser.error(f"--mode e2e requires: {', '.join(missing)}")
    if len(args.run_metadata) != len(_TRAINED_ARMS):
        parser.error(
            f"--mode e2e requires exactly {len(_TRAINED_ARMS)} --run-metadata paths, "
            f"one per trained arm in the order {list(_TRAINED_ARMS)}"
        )
    for path in [
        *cast(dict[str, Path], arm_paths).values(),
        *args.run_metadata,
        cast(Path, args.b0_universe),
        cast(Path, args.b0cal_results),
        cast(Path, args.probe_artifact),
    ]:
        if not path.exists():
            parser.error(f"input not found: {path}")
    run_g5_e2e_stage1_pipeline(
        arm_universe_paths=cast(dict[str, Path], arm_paths),
        run_metadata_paths=dict(zip(_TRAINED_ARMS, args.run_metadata, strict=True)),
        b0_universe_path=cast(Path, args.b0_universe),
        b0cal_results_path=cast(Path, args.b0cal_results),
        probe_artifact_path=cast(Path, args.probe_artifact),
        data_root=args.data_root,
        strategy=args.strategy,
        output_dir=cast(Path, args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()


# Referenced for reuse by tests and companion tooling.
__all__ = [
    "EvidenceClassViolation",
    "build_e2e_arm_summary",
    "paired_bootstrap_lower_bound",
    "render_e2e_tables_markdown",
    "run_g5_e2e_stage1_pipeline",
    "validate_dead_residual_within_checkpoint",
]
