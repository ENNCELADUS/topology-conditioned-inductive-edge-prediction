"""Executable pre-binding G3 qualification gates (spec Sec 14.4.7).

The v3 registration declares five gates under ``prebinding_qualification.gates``
but nothing read them: they existed only as JSON and as assertions *about* that
JSON in ``tests/test_g5_e2e_registration_v3.py``. This module is the evaluator.

Three gates (``slot_recall_at_n_ground``, ``Pi_consistency_v2``,
``degree_partialled_clustering_probe_r2``) are computable from a single probe
artifact with no scoring and no graph assembly, so they are evaluated here.

The remaining two (``structure_control_6a_v3_clustering_mmd_movement``,
``matched_edge_auprc_guard``) are **not reachable pre-binding at all**, and this
module reports them as unresolved rather than pretending otherwise:

* both need a scored pair universe, and
  ``src.score_universe._preflight_binding_artifacts_before_pair_access`` refuses
  every e2e pair source (including ``file:``) unless the registration is already
  ``BINDING`` with no markers — which is the state calibration exists to reach;
* their registered thresholds are still ``REQUIRED-BEFORE-BINDING`` strings;
* ``matched_edge_auprc_guard``'s reference is unregistered in v3 (there is no
  ``frozen_inputs`` key) and the v2 analogue read the sealed B0 candidate
  universe; and
* the clustering-MMD movement test needs reference-graph node buckets, which
  exist only for the test graph (``load_test_node_buckets``).

Resolving that circularity is a scientific change to how a gate is measured and
therefore needs a spec Sec 14.4.7 amendment with a Sec 12 change-log line before
any code lands. Until then an evaluation that includes them is *incomplete*, and
``passed`` is never true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from src.experiments.probes import (
    E2E_PROBE_SCOPES,
    E2EProbeScope,
    build_probe_scope_context,
    evaluate_e2e_probe_artifact,
)

_UNRESOLVED_MARKER = "REQUIRED-BEFORE-BINDING"

#: The reducer that turns a per-node/per-pair probe array into its gate scalar.
#: Pinned explicitly here because the registration names the gate but not its
#: reduction, and the published tables render the mean.
GATE_REDUCER = "mean"

#: Gates computable from a probe artifact alone, mapped to their extractor.
_PROBE_GATE_PATHS: Mapping[str, tuple[str, ...]] = {
    "slot_recall_at_n_ground": ("slot_recall_at_n_ground", GATE_REDUCER),
    "Pi_consistency_v2": ("pi_consistency_v2", GATE_REDUCER),
    "degree_partialled_clustering_probe_r2": ("degree_partialled_r2", "clustering"),
}

#: Gates whose inputs are gated behind BINDING; see the module docstring.
_SCORING_GATE_REASONS: Mapping[str, str] = {
    "structure_control_6a_v3_clustering_mmd_movement": (
        "needs clustering-MMD over two scored arms plus reference-graph buckets; "
        "e2e scoring is refused while the registration is DRAFT and no V_fit "
        "bucket machinery exists"
    ),
    "matched_edge_auprc_guard": (
        "needs a scored pair universe and a registered matched reference; e2e "
        "scoring is refused while the registration is DRAFT and v3 registers no "
        "frozen_inputs reference"
    ),
}


class GateEvaluationError(RuntimeError):
    """Raised when the gate inputs are inconsistent with the registration."""


def _is_unresolved(value: object) -> bool:
    return isinstance(value, str) and _UNRESOLVED_MARKER in value


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    raise GateEvaluationError(
        f"unsupported gate operator {operator!r}; expected '>=' or '>'"
    )


def _extract(evaluation: Mapping[str, object], path: Sequence[str]) -> float:
    cursor: object = evaluation
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise GateEvaluationError(
                f"probe evaluation is missing {'.'.join(path)!r}"
            )
        cursor = cursor[key]
    if not isinstance(cursor, (int, float)):
        raise GateEvaluationError(f"probe value at {'.'.join(path)!r} is not numeric")
    return float(cursor)


#: The exact registered gate set. Pinned so that a hand edit during the
#: threshold freeze cannot drop or duplicate a gate and leave a report that
#: reads `complete` because the missing rows were never counted.
REGISTERED_GATE_SCHEMA: tuple[tuple[str, str], ...] = (
    ("G3.1", "slot_recall_at_n_ground"),
    ("G3.2", "Pi_consistency_v2"),
    ("G3.3", "degree_partialled_clustering_probe_r2"),
    ("G3.4", "structure_control_6a_v3_clustering_mmd_movement"),
    ("G3.5", "matched_edge_auprc_guard"),
)


def load_prebinding_gates(registration: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return the registered gate list, failing closed on a malformed block."""
    qualification = registration.get("prebinding_qualification")
    if not isinstance(qualification, Mapping):
        raise GateEvaluationError("registration lacks a prebinding_qualification object")
    gates = qualification.get("gates")
    if not isinstance(gates, list) or not gates:
        raise GateEvaluationError("prebinding_qualification.gates must be a non-empty list")
    parsed: list[Mapping[str, object]] = []
    for entry in gates:
        if not isinstance(entry, Mapping):
            raise GateEvaluationError("each prebinding gate must be an object")
        if not isinstance(entry.get("id"), str) or not isinstance(entry.get("name"), str):
            raise GateEvaluationError("each prebinding gate needs a string id and name")
        parsed.append(entry)
    _assert_exact_gate_schema(parsed)
    return parsed


def _assert_exact_gate_schema(gates: Sequence[Mapping[str, object]]) -> None:
    """Reject any gate list that is not exactly the registered five.

    A dropped gate would otherwise shrink the tally rather than fail it, so
    ``complete`` and ``passed`` could both become true against four gates.
    Duplicates are rejected for the same reason: a repeated passing row must not
    be able to stand in for a missing one.
    """
    observed = [(cast(str, gate["id"]), cast(str, gate["name"])) for gate in gates]
    if len(set(observed)) != len(observed):
        duplicates = sorted({pair for pair in observed if observed.count(pair) > 1})
        raise GateEvaluationError(
            f"prebinding_qualification.gates contains duplicate entries: {duplicates}"
        )
    if set(observed) != set(REGISTERED_GATE_SCHEMA):
        missing = sorted(set(REGISTERED_GATE_SCHEMA) - set(observed))
        unexpected = sorted(set(observed) - set(REGISTERED_GATE_SCHEMA))
        raise GateEvaluationError(
            "prebinding_qualification.gates does not match the registered schema; "
            f"missing={missing} unexpected={unexpected}"
        )


#: Placeholder substituted for each gate threshold before digesting, so the
#: freeze comparison ignores exactly the fields the owner is permitted to edit.
_THRESHOLD_SENTINEL = "<threshold-frozen-between-stages>"


def unresolved_gate_ids(registration: Mapping[str, object]) -> list[str]:
    """Gate ids whose threshold is still a REQUIRED-BEFORE-BINDING marker.

    Evaluated against the *calibration-time* registration, this is exactly the
    set the owner is permitted to fill in before the rehearsal.
    """
    return [
        cast(str, gate["id"])
        for gate in load_prebinding_gates(registration)
        if _is_unresolved(gate.get("threshold"))
    ]


def canonical_registration_digest(
    registration: Mapping[str, object], *, freeze_eligible_ids: Sequence[str]
) -> str:
    """Digest the registration with only the freeze-eligible thresholds masked.

    Freezing the GPU-calibrated thresholds into the DRAFT is the one change
    allowed between calibration and rehearsal, so the raw file digest cannot be
    compared. Ignoring registration drift *entirely* is the opposite error: a
    gate operator flipped from ``>`` to ``>=``, or a rewritten protocol field,
    would pass unnoticed.

    Masking *every* threshold is a subtler version of the same mistake. G3.1-G3.3
    are already fixed numbers before calibration begins; if their thresholds were
    masked too, weakening G3.1 from 0.0698 after calibration would leave the
    digest unchanged and the rehearsal would run against a gate nobody
    registered. Only the ids that were genuinely unresolved at calibration time
    are masked — and that set is recorded in the freeze manifest, not inferred
    again at rehearsal, when those thresholds are numbers like any other.
    """
    eligible = set(freeze_eligible_ids)
    masked = json.loads(json.dumps(registration))
    qualification = masked.get("prebinding_qualification")
    if isinstance(qualification, dict):
        gates = qualification.get("gates")
        if isinstance(gates, list):
            for gate in gates:
                if isinstance(gate, dict) and gate.get("id") in eligible:
                    gate["threshold"] = _THRESHOLD_SENTINEL
    payload = json.dumps(masked, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unimplemented_gates(registration: Mapping[str, object]) -> list[str]:
    """Return registered gate ids this module cannot evaluate.

    A rehearsal that cannot evaluate every registered gate is not a
    qualification, and V_qual is spent the moment the rehearsal starts — so this
    must be checkable *before* the run, not discovered from the report after.
    Freezing G3.4/G3.5's thresholds is necessary but not sufficient: their inputs
    also have to become reachable.
    """
    return [
        cast(str, gate["id"])
        for gate in load_prebinding_gates(registration)
        if gate["name"] not in _PROBE_GATE_PATHS
    ]


def _registered_n_ground(
    registration: Mapping[str, object], gates: Sequence[Mapping[str, object]]
) -> int:
    """Resolve the registered pool size, failing closed on an internal contradiction.

    ``grounding.n_ground`` governs: pre-binding qualification is full-arm only
    (``probe_artifact.source_arm``), so the ``cosine_pool`` arm's 20 never
    applies here. The G3.1 gate restates the value, and a registration that
    disagrees with itself must not be evaluated against either number.
    """
    grounding = registration.get("grounding")
    if not isinstance(grounding, Mapping):
        raise GateEvaluationError("registration lacks a grounding object")
    n_ground = grounding.get("n_ground")
    if not isinstance(n_ground, int) or isinstance(n_ground, bool) or n_ground <= 0:
        raise GateEvaluationError("grounding.n_ground must be a positive integer")
    for gate in gates:
        if gate.get("name") != "slot_recall_at_n_ground":
            continue
        gate_n_ground = gate.get("n_ground")
        if gate_n_ground is not None and gate_n_ground != n_ground:
            raise GateEvaluationError(
                f"registration is self-inconsistent: grounding.n_ground={n_ground} but "
                f"gate {gate.get('id')!r} registers n_ground={gate_n_ground!r}"
            )
    return n_ground


def evaluate_prebinding_gates(
    *,
    probe_artifact_path: Path,
    preregistration_path: Path,
    data_root: Path,
    strategy: str,
    scope: E2EProbeScope,
    partition_seed: int,
    msg_fraction: float,
    expected_missing_features: Sequence[str],
) -> dict[str, object]:
    """Evaluate every registered pre-binding gate against one probe artifact."""
    if scope not in E2E_PROBE_SCOPES:
        raise GateEvaluationError(f"unsupported probe scope {scope!r}")
    if scope == "formal_train":
        raise GateEvaluationError(
            "pre-binding gates are scoped to calibration_fit or qualification_qual; "
            "formal_train is a post-binding artifact"
        )

    registration_bytes = preregistration_path.read_bytes()
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    registration = cast(
        dict[str, object], json.loads(registration_bytes.decode("utf-8"))
    )
    if registration.get("status") != "DRAFT":
        raise GateEvaluationError(
            "pre-binding gate evaluation requires the registration to be DRAFT"
        )
    gates = load_prebinding_gates(registration)
    registered_n_ground = _registered_n_ground(registration, gates)

    nodes, graph = build_probe_scope_context(
        scope,
        data_root=data_root,
        strategy=strategy,
        partition_seed=partition_seed,
        msg_fraction=msg_fraction,
        expected_missing_features=expected_missing_features,
    )
    evaluation = evaluate_e2e_probe_artifact(
        probe_artifact_path,
        graph=graph,
        train_nodes=nodes,
        expected_metadata={
            "scope": scope,
            "registration_sha256": registration_sha,
            "n_ground": registered_n_ground,
        },
    )
    return build_gate_report(
        registration, evaluation, registration_sha=registration_sha, scope=scope
    )


def build_gate_report(
    registration: Mapping[str, object],
    evaluation: Mapping[str, object],
    *,
    registration_sha: str,
    scope: E2EProbeScope,
) -> dict[str, object]:
    """Map a validated probe evaluation onto the registered gate list."""
    gates = load_prebinding_gates(registration)
    metadata = cast(Mapping[str, object], evaluation["metadata"])

    results: list[dict[str, object]] = []
    for gate in gates:
        gate_id = cast(str, gate["id"])
        name = cast(str, gate["name"])
        threshold = gate.get("threshold")
        row: dict[str, object] = {"id": gate_id, "name": name}

        if name in _SCORING_GATE_REASONS:
            row.update(
                status="not_evaluable_prebinding",
                passed=None,
                reason=_SCORING_GATE_REASONS[name],
            )
            results.append(row)
            continue
        if _is_unresolved(threshold):
            row.update(
                status="unresolved_threshold",
                passed=None,
                reason="threshold is still a REQUIRED-BEFORE-BINDING marker",
            )
            results.append(row)
            continue
        if name not in _PROBE_GATE_PATHS:
            raise GateEvaluationError(
                f"registered gate {gate_id} ({name!r}) has no evaluator; refusing to "
                "report an unknown gate as passing"
            )
        if not isinstance(threshold, (int, float)):
            raise GateEvaluationError(f"gate {gate_id} threshold must be numeric")
        operator = gate.get("operator")
        if not isinstance(operator, str):
            raise GateEvaluationError(f"gate {gate_id} needs a string operator")

        if name == "slot_recall_at_n_ground":
            # Equality against the artifact is already enforced by
            # evaluate_e2e_probe_artifact via expected_metadata["n_ground"], and
            # the gate/grounding agreement by _registered_n_ground; record it so
            # the report is self-describing.
            row["n_ground"] = metadata.get("n_ground")

        value = _extract(evaluation, _PROBE_GATE_PATHS[name])
        row.update(
            status="evaluated",
            value=value,
            operator=operator,
            threshold=float(threshold),
            reducer=GATE_REDUCER if _PROBE_GATE_PATHS[name][-1] == GATE_REDUCER else None,
            passed=_compare(value, operator, float(threshold)),
        )
        results.append(row)

    evaluated = [row for row in results if row["status"] == "evaluated"]
    incomplete = [row for row in results if row["status"] != "evaluated"]
    return {
        "format": "egostitch_e2e_prebinding_gates_v1",
        "scope": scope,
        "registration_sha256": registration_sha,
        "checkpoint_id": metadata.get("checkpoint_id"),
        "probe_format": metadata.get("format"),
        "gates": results,
        "n_evaluated": len(evaluated),
        "n_incomplete": len(incomplete),
        # Never true while any gate is unresolved: an incomplete gate set cannot
        # certify qualification, and the protocol blocks binding on any failure.
        "passed": bool(evaluated) and not incomplete and all(row["passed"] for row in evaluated),
        "complete": not incomplete,
    }


def enforce_verdict(
    report: Mapping[str, object], *, scope: E2EProbeScope, output_path: Path
) -> None:
    """Fail the process when a rehearsal report is not a clean pass.

    A rehearsal is a verdict, so a non-passing report must stop the caller.
    Writing the JSON and exiting 0 would let ``set -e`` sail past a failed gate,
    let the launcher announce a completed rehearsal, and leave a report a
    binding-evidence validator could later accept by hash without ever reading
    its verdict.

    Calibration is a *measurement* stage whose entire purpose is to run while
    thresholds are still unresolved, so it never fails here — its report is
    input to the owner's freeze decision, not a judgement.
    """
    if scope != "qualification_qual":
        return
    if not report["passed"]:
        raise SystemExit(
            f"pre-binding gates did not pass: {output_path} "
            f"({report['n_evaluated']} evaluated, {report['n_incomplete']} incomplete)"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the pre-binding gate evaluator CLI."""
    parser = argparse.ArgumentParser(prog="python -m src.experiments.prebinding_gates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--probe-artifact", type=Path, required=True)
    evaluate.add_argument("--preregistration", type=Path, required=True)
    evaluate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="registered arm config; the scope universe is derived from it",
    )
    evaluate.add_argument(
        "--scope", choices=("calibration_fit", "qualification_qual"), required=True
    )
    evaluate.add_argument("--output", type=Path, required=True)
    implementable = subparsers.add_parser(
        "check-implementable",
        help="exit nonzero unless every registered gate has an evaluator",
    )
    implementable.add_argument("--preregistration", type=Path, required=True)
    digest = subparsers.add_parser(
        "canonical-digest",
        help="print the registration digest with permitted threshold edits masked",
    )
    digest.add_argument("--preregistration", type=Path, required=True)
    digest.add_argument(
        "--freeze-eligible",
        action="append",
        default=None,
        metavar="GATE_ID",
        help=(
            "gate id whose threshold may change across the freeze; omit at "
            "calibration to derive the set from the unresolved markers"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    from src.train_egostitch import load_config

    args = build_parser().parse_args(argv)
    if args.command == "canonical-digest":
        registration = cast(
            dict[str, object], json.loads(args.preregistration.read_text(encoding="utf-8"))
        )
        # Derived at calibration (when the markers are still present) and echoed
        # so the caller can record it; supplied verbatim from that record at
        # rehearsal, when the same thresholds are ordinary numbers.
        eligible = (
            unresolved_gate_ids(registration)
            if args.freeze_eligible is None
            else list(args.freeze_eligible)
        )
        sys.stdout.write(
            json.dumps(
                {
                    "digest": canonical_registration_digest(
                        registration, freeze_eligible_ids=eligible
                    ),
                    "freeze_eligible_gate_ids": sorted(eligible),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return
    if args.command == "check-implementable":
        registration = cast(
            dict[str, object], json.loads(args.preregistration.read_text(encoding="utf-8"))
        )
        missing = unimplemented_gates(registration)
        if missing:
            raise SystemExit(
                "no executable evaluator for registered gate(s): "
                + ", ".join(missing)
            )
        return
    if args.command != "evaluate":
        return
    # Derived from the config rather than repeated on the command line:
    # data.partition_seed is not `seed`, and a shell-side copy that drifts would
    # silently rebuild a different G_struct and a different holdout.
    cfg = load_config(args.config)
    report = evaluate_prebinding_gates(
        probe_artifact_path=args.probe_artifact,
        preregistration_path=args.preregistration,
        data_root=cfg.data.root,
        strategy=cfg.data.strategy,
        scope=args.scope,
        partition_seed=cfg.data.partition_seed,
        msg_fraction=cfg.data.msg_fraction,
        expected_missing_features=cfg.data.expected_missing_features,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    enforce_verdict(report, scope=args.scope, output_path=args.output)


if __name__ == "__main__":
    main()
