"""The section 8 output-density control: read shape at a common assembled density.

This is a property of the assembled graph, not of the inputs, and it is a
different object from the coordinate calibration of spec section 0.2. It asks
whether the arm's assembled edge set is a better *shape* at equal density, or
whether the apparent gain was threshold placement -- a previous arm's entire
topology headline was traced to a per-universe logit shift. It is reported
**alongside**, never instead of, the protocol's V_val-selected threshold, since
re-selecting a threshold per row is outside ``test_protocol_v8``.

Self-loops keep the official convention throughout: rows are the pairs the
protocol scores, self-pairs included, a self-pair over the threshold assembles
as a self-loop and official GS and RD retain it. A loopless variant is permitted
only as a separate, labelled diagnostic.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src.eval.assembly import density_matched_threshold
from src.eval.fixed_threshold import evaluate_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    load_test_graph,
    load_test_node_buckets,
)
from src.score_universe import ScoresArtifact, load_scores, validate_artifact_precision

MATCHING = "output_density_matched_union"
#: The protocol's test-universe topology artifact inside a published run directory.
TOPOLOGY_ARTIFACT = Path("scores") / "test_topology.npz"


def target_edge_count(logits: NDArray[np.float64], *, threshold: float) -> tuple[int, int]:
    """Return ``(target_edges, n_union)`` from this arm's own selected count.

    ``target_edges`` is the number of union pairs **this arm** realises at its
    protocol V_val-selected threshold on the evaluation union -- its own
    assembled edge count, taken directly, with no rate transferred between
    universes. A rate may substitute only if it is ``selected_pairs / n_union``;
    it is **not** ``nx.density``, whose denominator counts node pairs over the
    whole node set while a sampled union holds only some of those pairs. Because
    every row scores the same union over the same nodes, an equal selected count
    is an equal assembled edge count and therefore an equal assembled density.

    Args:
        logits: The arm's raw logits over the union rows, self-pairs included.
        threshold: The arm's V_val-selected fixed logit threshold.

    Returns:
        The realised edge count and the union row count.
    """
    values = np.asarray(logits, dtype=np.float64)
    return int((values >= threshold).sum()), int(values.size)


def density_matched_report(
    *,
    rows: Mapping[str, NDArray[np.float64]],
    union_pairs: Sequence[tuple[str, str]],
    target_edges: int,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig | None,
) -> dict[str, object]:
    """Score every row at the common target density and report the residual spread.

    One threshold per row, applied unchanged to every subgraph, as
    `evaluate_fixed_threshold` does. `density_matched_threshold` admits tie
    groups atomically, so the realised count is ``<= target_edges`` and exact
    equality is not guaranteed; a row landing more than 1% below the target is
    marked ``approximate`` rather than having its target adjusted. Matching the
    *union* density does not equalise density inside each BFS subgraph, so
    per-subgraph RD still varies and a density-matched row's macro RD is not 1:
    read GS and the three MMD ratios here and treat RD as a diagnostic of that
    residual spread.

    The threshold is selected on the raw logits rather than on probabilities.
    ``expit`` is strictly monotone, so the two selections admit exactly the same
    tie groups and the same rows, but staying on the logit scale keeps the
    threshold `evaluate_fixed_threshold` consumes exact: a probability that
    saturates to 1.0 in float64 has no finite logit to invert back to.

    Args:
        rows: Raw logits per row name (this arm, ``prefix_base``, ``B0``).
        union_pairs: The pooled union of pairs the protocol scores, self-pairs included.
        target_edges: The common target from `target_edge_count`.
        g_ref: The reference graph.
        buckets: The protocol's BFS subgraph buckets.
        config: The MMD configuration, or ``None`` to skip the topology panel.

    Returns:
        A report with, per row, the chosen threshold, the realised edge count,
        the target, whether the shape comparison is approximate, and the five
        topology numbers when ``config`` is supplied.
    """
    entries: dict[str, object] = {}
    for name, row in rows.items():
        logits = np.asarray(row, dtype=np.float64)
        threshold = density_matched_threshold(logits, target_edges)
        realised = int((logits >= threshold).sum())
        entry: dict[str, object] = {
            "logit_threshold": float(threshold),
            "probability_threshold": float(expit(threshold)),
            "realised_edges": realised,
            "target_edges": int(target_edges),
            "shape_comparison": ("approximate" if realised < target_edges * 0.99 else "matched"),
        }
        if config is not None:
            _, panel = evaluate_fixed_threshold(
                pairs=union_pairs,
                logits=logits,
                g_ref=g_ref,
                buckets=buckets,
                threshold=float(threshold),
                config=config,
            )
            entry["panel"] = panel
        entries[name] = entry
    return {
        "matching": MATCHING,
        "target_edges": int(target_edges),
        "n_union": len(union_pairs),
        "reported": "alongside the V_val-selected threshold, never instead of it",
        "rows": entries,
    }


def _load_topology_artifact(run_dir: Path) -> ScoresArtifact:
    """Load and precision-check a run's test-topology artifact.

    Args:
        run_dir: A published run directory holding ``scores/test_topology.npz``.

    Returns:
        The artifact.

    Raises:
        ValueError: If the artifact is not the protocol's ``test_topology`` universe.
    """
    path = run_dir / TOPOLOGY_ARTIFACT
    artifact = load_scores(path)
    validate_artifact_precision(artifact, label=str(path))
    if artifact.meta.get("pairs_source") != "test_topology":
        raise ValueError(
            f"{path}: pairs_source {artifact.meta.get('pairs_source')!r}, expected 'test_topology'"
        )
    return artifact


def selected_logit_threshold(report_path: Path) -> float:
    """Read the arm's ONE V_val-selected topology threshold from its test report.

    ``test_protocol_v8`` writes `select_fixed_threshold`'s report under
    ``graph.fixed_threshold.validation_selection``, where the chosen threshold
    lives in the nested ``selected`` block beside its selection audit -- the
    flat sibling key belongs to the ``test`` replay block, not to the selection.

    Args:
        report_path: The run's ``test_report.json`` written by `test_protocol`.

    Returns:
        The frozen logit threshold the protocol replayed on every test subgraph.

    Raises:
        ValueError: If the report has no selected validation threshold.
    """
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selection = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], payload["graph"])["fixed_threshold"])[
            "validation_selection"
        ],
    )
    selected = selection.get("selected")
    if not isinstance(selected, Mapping) or "logit_threshold" not in selected:
        raise ValueError(
            f"{report_path}: graph.fixed_threshold.validation_selection.selected."
            "logit_threshold is missing; the report predates test_protocol_v8"
        )
    return float(cast(float, selected["logit_threshold"]))


def density_control_from_runs(
    *,
    arm_dir: Path,
    reference_dirs: Sequence[Path],
    data_root: Path,
    strategy: str,
    report_filename: str = "test_report.json",
    config: MMDConfig | None = None,
) -> dict[str, object]:
    """Run the control over published run directories (spec section 8, steps 1-4).

    Every row must have scored the protocol's ``test_topology`` universe -- the
    same pairs in the same order -- because an equal selected count is an equal
    assembled density only over one universe. The target is the arm's own
    realised count at its V_val-selected threshold, read from its test report.

    Args:
        arm_dir: The arm's published run directory.
        reference_dirs: The comparison rows' run directories (``prefix_base``, B0).
        data_root: Benchmark data root.
        strategy: Split strategy (for example ``breadth_first``).
        report_filename: The arm's protocol report inside ``arm_dir``.
        config: The MMD configuration; ``None`` skips the topology panel.

    Returns:
        The `density_matched_report` payload with the run provenance attached.

    Raises:
        ValueError: If a reference row scored a different universe than the arm.
    """
    arm = _load_topology_artifact(arm_dir)
    arm_pairs = list(arm.pairs())
    threshold = selected_logit_threshold(arm_dir / report_filename)
    target_edges, n_union = target_edge_count(arm.logit.astype(np.float64), threshold=threshold)
    rows: dict[str, NDArray[np.float64]] = {arm_dir.name: arm.logit.astype(np.float64)}
    provenance: dict[str, object] = {
        arm_dir.name: {
            "run_dir": str(arm_dir),
            "checkpoint_id": arm.meta.get("checkpoint_id"),
            "role": "arm",
            "selected_logit_threshold": threshold,
        }
    }
    for reference_dir in reference_dirs:
        reference = _load_topology_artifact(reference_dir)
        if list(reference.pairs()) != arm_pairs:
            raise ValueError(
                f"{reference_dir / TOPOLOGY_ARTIFACT}: scored a different universe than "
                f"{arm_dir / TOPOLOGY_ARTIFACT}; the control needs one union of pairs"
            )
        rows[reference_dir.name] = reference.logit.astype(np.float64)
        provenance[reference_dir.name] = {
            "run_dir": str(reference_dir),
            "checkpoint_id": reference.meta.get("checkpoint_id"),
            "role": "reference",
        }
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    report = density_matched_report(
        rows=rows,
        union_pairs=arm_pairs,
        target_edges=target_edges,
        g_ref=load_test_graph(benchmark_root, strategy),
        buckets=load_test_node_buckets(benchmark_root, strategy),
        config=config,
    )
    assert report["n_union"] == n_union
    report["strategy"] = strategy
    report["provenance"] = provenance
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Section 8 output-density control: re-threshold the arm and its reference rows "
            "to the arm's own realised edge count on the protocol's test-topology union and "
            "report GS and the three MMD ratios at that common density, alongside -- never "
            "instead of -- the V_val-selected threshold."
        )
    )
    parser.add_argument("--arm", type=Path, required=True, help="the arm's published run dir")
    parser.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        required=True,
        help="reference run dirs scored on the same universe (prefix_base, B0)",
    )
    parser.add_argument("--output", type=Path, required=True, help="report JSON path")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strategy", default="breadth_first")
    parser.add_argument("--report-filename", default="test_report.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the control and write its report.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    report = density_control_from_runs(
        arm_dir=args.arm,
        reference_dirs=list(args.reference),
        data_root=args.data_root,
        strategy=args.strategy,
        report_filename=args.report_filename,
        config=MMDConfig(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "MATCHING",
    "TOPOLOGY_ARTIFACT",
    "density_control_from_runs",
    "density_matched_report",
    "main",
    "selected_logit_threshold",
    "target_edge_count",
]
