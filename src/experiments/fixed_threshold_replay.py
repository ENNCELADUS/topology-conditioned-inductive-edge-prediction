r"""Fixed-threshold assembled-graph replay over cached candidate-universe scores.

Score-time replay, no training. Every published test report assembles the
candidate universe only at the density-matched operating point (RD forced
toward 1 by construction) plus percentile sweep points, so it never answers:
what graph does an arm produce at the raw decision threshold 0.5 -- the same
fixed threshold its edge-level classification metrics already use? This module
replays one or more cached candidate-universe artifacts (score-once contract:
no checkpoint is loaded, no pair is rescored) at a single fixed threshold and
reports, per arm, the five topology numbers (BFS-macro GS/RD plus the
degree/clustering/spectral MMD ratios, never aggregated -- CLAUDE.md claim
rule), the global simple-edge GS/RD named separately, edge-set
precision/recall, and self-loop counts. A self-pair `(u, u)` clearing the
threshold assembles as a self-loop, exactly as in the density-matched blocks
it is compared against.

CLI::

    python -m src.experiments.fixed_threshold_replay \
        --universe kd_control=outputs/b1_stage_v2/kd_control/scores/candidate.npz \
        --universe d1=outputs/b1_stage_v2/kd_d1/scores/candidate.npz \
        --data-root data --strategy breadth_first \
        --output outputs/t05_replay/fixed_threshold_results.json [--threshold 0.5]

Determinism: identical inputs produce a byte-identical output JSON (no
randomized step, ``json.dumps(..., sort_keys=True)``, no wall-clock fields).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from src.eval.assembly import assemble_graph
from src.eval.graph_metrics import (
    MMDConfig,
    precompute_bucket_reference,
    strip_self_loops,
)
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    _artifact_meta_summary,
    load_test_graph,
    load_test_node_buckets,
    validate_universe_artifact,
)
from src.experiments.s4_budget_assembly import evaluate_arm
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)


def parse_universe_spec(spec: str) -> tuple[str, Path]:
    """Split one ``--universe`` value into its ``(arm, path)`` pair.

    Args:
        spec: An ``arm=path`` string; the arm name may not be empty or
            contain ``=``, the path is everything after the first ``=``.

    Returns:
        The ``(arm_name, artifact_path)`` pair.

    Raises:
        ValueError: If `spec` has no ``=`` or an empty arm name / path.
    """
    arm, sep, path = spec.partition("=")
    if not sep or not arm or not path:
        raise ValueError(f"--universe expects arm=path, got {spec!r}")
    return arm, Path(path)


def run_fixed_threshold_replay(
    *,
    universes: Sequence[tuple[str, Path]],
    data_root: Path,
    strategy: str,
    threshold: float,
    output_path: Path,
) -> dict[str, object]:
    """Assemble every arm's cached candidate universe at one fixed threshold.

    Args:
        universes: ``(arm_name, candidate_artifact_path)`` pairs, evaluated in
            the given order.
        data_root: Directory containing ``benchmark_2025_neurips/``.
        strategy: Benchmark split strategy (e.g. ``"breadth_first"``).
        threshold: The fixed assembly threshold (a pair is an edge iff its
            probability is ``>= threshold``; self-pairs become self-loops).
        output_path: JSON file the payload is written to.

    Returns:
        The JSON-ready results payload (also written to `output_path`).

    Raises:
        ValueError: If `threshold` is non-finite (fail closed), two universes
            share an arm name, or an artifact fails the candidate-universe /
            precision validation.
    """
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    names = [arm for arm, _ in universes]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names in --universe: {names}")

    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    g_simple = strip_self_loops(g_ref)
    target_edges = g_simple.number_of_edges()
    buckets = load_test_node_buckets(benchmark_root, strategy)
    config = MMDConfig()
    reference = precompute_bucket_reference(g_ref, buckets, config)
    nodes = list(g_ref.nodes())

    arms: dict[str, object] = {}
    artifacts_meta: dict[str, object] = {}
    for arm, path in universes:
        logger.info("replaying arm %s at threshold %.6g from %s", arm, threshold, path)
        artifact = load_scores(path)
        validate_artifact_precision(artifact, label=arm)
        validate_universe_artifact(
            artifact, strategy=strategy, n_test_nodes=g_ref.number_of_nodes(), label=arm
        )
        g_pred = assemble_graph(
            list(artifact.pairs()), artifact.probs(), threshold=threshold, nodes=nodes
        )
        topology = evaluate_arm(g_pred, reference, config, g_simple)
        arms[arm] = {
            **dataclasses.asdict(topology),
            "predicted_edges_simple": strip_self_loops(g_pred).number_of_edges(),
        }
        artifacts_meta[arm] = {**_artifact_meta_summary(artifact.meta), "path": str(path)}

    payload: dict[str, object] = {
        "metadata": {
            "threshold": threshold,
            "strategy": strategy,
            "target_edges": target_edges,
            "artifacts": artifacts_meta,
            "arms": sorted(names),
            "self_loop_policy": (
                "a self-pair (u, u) with probability >= threshold assembles as a "
                "self-loop; no density-matched self-loop quota exists at a fixed threshold"
            ),
        },
        "arms": arms,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote fixed-threshold replay results to %s", output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fixed_threshold_replay`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experiments.fixed_threshold_replay",
        description=(
            "Assemble cached candidate-universe scores at one fixed threshold and "
            "report the five topology numbers per arm."
        ),
    )
    parser.add_argument(
        "--universe",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help="arm name and its cached candidate .npz (repeatable)",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    run_fixed_threshold_replay(
        universes=[parse_universe_spec(spec) for spec in args.universe],
        data_root=args.data_root,
        strategy=args.strategy,
        threshold=args.threshold,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
