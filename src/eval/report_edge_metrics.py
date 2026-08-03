r"""Compute edge-level classification metrics from a cached scores artifact.

This is the reporting half of the "score once, analyze many times" contract for
the *edge-level* task, mirroring what
:mod:`src.experiments.g1_hardened_e2` is for the assembled-graph task. It loads a
``.npz`` written by :mod:`src.score_universe`, validates it, and writes the
metric block; it never loads a checkpoint and never rescores a pair.

It exists because the balanced test view required by ``docs/05-egostitch-spec.md``
§10.3 ("Test edge metrics: ``test_edges.txt`` as-is; run once, after freeze") had
no committed implementation: the G1/G2/G3/G5 analyzers all hard-reject any
artifact whose ``pairs_source`` is not ``candidate``, so a ``--pairs test``
artifact could be produced but never reported reproducibly.

Typical use, reporting both metric families for one checkpoint (protocol §E5 —
edge-level and assembled-graph metrics are always reported together)::

    python -m src.eval.report_edge_metrics \\
        --scores scores/b0_v31_test.npz \\
        --expect-pairs-source test \\
        --output outputs/b0_v31/test_edge_metrics.json

    hpc/run.sh g1 --universe scores/b0_v31_candidate.npz \\
        --data-root data --strategy breadth_first --output-dir outputs/g1
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.eval.edge_metrics import compute_edge_metrics
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)

#: Row label written by `src.score_universe` for a pair it could not label.
_UNLABELED = -1


def report_edge_metrics(
    scores_path: Path,
    *,
    expect_pairs_source: str | None = None,
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> dict[str, object]:
    """Load a scores artifact and compute its edge-level classification metrics.

    Args:
        scores_path: Path to a ``.npz`` written by :mod:`src.score_universe`.
        expect_pairs_source: When set, require ``meta["pairs_source"]`` to equal
            it. Guards against reporting a candidate-universe artifact as the
            balanced test view, or vice versa.
        threshold: Decision threshold for the thresholded metrics.
        ece_bins: Bin count for expected calibration error.

    Returns:
        A JSON-serializable dict carrying the metric block plus the artifact
        provenance needed to trace it back to a checkpoint.

    Raises:
        ValueError: If the artifact fails its precision contract, its
            ``pairs_source`` contradicts `expect_pairs_source`, or it contains
            unlabeled rows (which cannot enter a classification metric).
    """
    artifact = load_scores(scores_path)
    # The only correct entry point: calling validate_score_precision directly on
    # an egostitch_e2e artifact spuriously raises "missing arrays".
    validate_artifact_precision(artifact, label=str(scores_path))

    pairs_source = artifact.meta.get("pairs_source")
    if expect_pairs_source is not None and pairs_source != expect_pairs_source:
        raise ValueError(
            f"{scores_path}: pairs_source expected {expect_pairs_source!r}, got {pairs_source!r}"
        )

    labels = artifact.label
    n_unlabeled = int((labels == _UNLABELED).sum())
    if n_unlabeled:
        raise ValueError(
            f"{scores_path}: {n_unlabeled} of {labels.size} rows are unlabeled (-1); "
            "edge-level classification metrics require a fully labeled pair set"
        )

    metrics = compute_edge_metrics(
        labels.astype(np.int64), artifact.probs(), threshold=threshold, ece_bins=ece_bins
    )
    logger.info(
        "%s: %d rows, AUROC %.6f AUPRC %.6f ECE %.6f",
        scores_path,
        labels.size,
        metrics.auroc,
        metrics.auprc,
        metrics.ece,
    )

    return {
        "scores_path": str(scores_path),
        "checkpoint_id": artifact.meta.get("checkpoint_id"),
        "model_family": artifact.meta.get("model_family"),
        "pairs_source": pairs_source,
        "strategy": artifact.meta.get("strategy"),
        "num_rows": int(labels.size),
        "threshold": threshold,
        "ece_bins": ece_bins,
        "metrics": asdict(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.report_edge_metrics",
        description="Edge-level classification metrics from a cached scores .npz",
    )
    parser.add_argument("--scores", type=Path, required=True, help="scores .npz to report on")
    parser.add_argument("--output", type=Path, required=True, help="output .json path")
    parser.add_argument(
        "--expect-pairs-source",
        choices=("candidate", "test", "val"),
        default=None,
        help="require this pairs_source; omit to accept whatever the artifact carries",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    parser.add_argument("--ece-bins", type=int, default=15, help="calibration bin count")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI end to end.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    report = report_edge_metrics(
        args.scores,
        expect_pairs_source=args.expect_pairs_source,
        threshold=args.threshold,
        ece_bins=args.ece_bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
