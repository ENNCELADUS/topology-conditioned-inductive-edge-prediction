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
from numpy.typing import NDArray

from src.eval.edge_metrics import compute_edge_metrics
from src.score_universe import load_scores, validate_artifact_precision

logger = logging.getLogger(__name__)

#: Row label written by `src.score_universe` for a pair it could not label.
_UNLABELED = -1

#: Pair views that additionally report the self / non-self split. Every view keeps
#: self-loops in its headline metric; only the test view is also split out, because
#: the split exists to expose self-loop behavior in the final reported result, not
#: to re-cut model-selection numbers.
_SPLIT_REPORTING_SOURCES = ("test",)


def _shifted_probs(logit: NDArray[np.floating], logit_shift: float) -> NDArray[np.float64]:
    """Return the stable sigmoid of ``logit + logit_shift`` in float64.

    Mirrors ``ScoresArtifact.probs()`` exactly, so a shift of ``0.0`` is
    numerically identical to the unshifted artifact probabilities.
    """
    shifted = logit.astype(np.float64) + logit_shift
    out = np.empty_like(shifted)
    positive = shifted >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-shifted[positive]))
    exp_l = np.exp(shifted[~positive])
    out[~positive] = exp_l / (1.0 + exp_l)
    return out


def _stratum_metrics(
    labels: NDArray[np.int64],
    probs: NDArray[np.float64],
    *,
    threshold: float,
    ece_bins: int,
    name: str,
) -> dict[str, object] | None:
    """Compute one stratum's metrics, or `None` when it cannot carry them.

    A stratum with no rows, or with only one label class, has undefined
    AUROC/AUPRC/MCC. Spec §9.4 still requires the stratum to appear in the
    report, so it is emitted as an explicit null with a reason rather than
    silently dropped or filled with NaN.
    """
    n_rows = int(labels.size)
    n_pos = int((labels == 1).sum())
    if n_rows == 0:
        logger.info("%s stratum: empty", name)
        return None
    if n_pos == 0 or n_pos == n_rows:
        logger.warning(
            "%s stratum: single-class (%d/%d positive); metrics undefined", name, n_pos, n_rows
        )
        return None
    return asdict(compute_edge_metrics(labels, probs, threshold=threshold, ece_bins=ece_bins))


def report_edge_metrics(
    scores_path: Path,
    *,
    expect_pairs_source: str | None = None,
    threshold: float = 0.5,
    ece_bins: int = 15,
    logit_shift: float = 0.0,
) -> dict[str, object]:
    """Load a scores artifact and compute its edge-level classification metrics.

    **Self-loops are retained in every pair view.** They are first-class labeled
    ``(u, u)`` queries in this benchmark (spec §9.4), so the headline ``metrics``
    block always includes them, for val and test alike — nothing is stripped.

    The **test** view additionally reports the self / non-self split and a
    self-loop-rate row, per spec §9.4 rule 3, so that self-loop behavior is
    visible in the final reported result. The split is supplementary: the
    self-loop-including ``metrics`` block comes first and remains the headline.
    Other views (val, candidate) report the headline block only.

    Args:
        scores_path: Path to a ``.npz`` written by :mod:`src.score_universe`.
        expect_pairs_source: When set, require ``meta["pairs_source"]`` to equal
            it. Guards against reporting a candidate-universe artifact as the
            balanced test view, or vice versa.
        threshold: Decision threshold for the thresholded metrics.
        ece_bins: Bin count for expected calibration error.
        logit_shift: Added to every raw logit before the sigmoid, so calibration
            (e.g. shifting a validation-selected threshold to probability 0.5)
            is applied here rather than by rescoring; ECE/Brier then describe
            the calibrated, deployed probabilities.

    Returns:
        A JSON-serializable dict carrying the self-loop-including ``metrics``
        block first, the artifact provenance needed to trace it back to a
        checkpoint, and — for the test view only — the ``metrics_self`` /
        ``metrics_non_self`` strata and the ``self_loop_rate`` row.

    Raises:
        ValueError: If the artifact fails its precision contract, its
            ``pairs_source`` contradicts `expect_pairs_source`, it contains
            unlabeled rows, or it is a partial shard rather than a merged
            artifact.
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
    n_rows = int(labels.size)

    # A single distributed shard loads cleanly but carries only its own rows while
    # its metadata still records the full universe, so metrics computed from it
    # would silently describe a subset. Merge shards first (score_universe merge).
    meta_rows = artifact.meta.get("num_rows")
    if isinstance(meta_rows, int) and meta_rows != n_rows:
        raise ValueError(
            f"{scores_path}: loaded {n_rows} rows but metadata declares {meta_rows}; "
            "this looks like a partial shard — merge shards before reporting"
        )

    n_unlabeled = int((labels == _UNLABELED).sum())
    if n_unlabeled:
        raise ValueError(
            f"{scores_path}: {n_unlabeled} of {n_rows} rows are unlabeled (-1); "
            "edge-level classification metrics require a fully labeled pair set"
        )

    labels64 = labels.astype(np.int64)
    probs = _shifted_probs(artifact.logit, logit_shift)
    is_self = artifact.u_idx == artifact.v_idx

    metrics = compute_edge_metrics(labels64, probs, threshold=threshold, ece_bins=ece_bins)
    logger.info(
        "%s: %d rows (%d self), AUROC %.6f AUPRC %.6f ECE %.6f",
        scores_path,
        n_rows,
        int(is_self.sum()),
        metrics.auroc,
        metrics.auprc,
        metrics.ece,
    )

    # Key order is meaningful and preserved on write: the self-loop-including
    # headline block comes first, any split second.
    report: dict[str, object] = {
        "scores_path": str(scores_path),
        "checkpoint_id": artifact.meta.get("checkpoint_id"),
        "model_family": artifact.meta.get("model_family"),
        "pairs_source": pairs_source,
        "strategy": artifact.meta.get("strategy"),
        "num_rows": n_rows,
        "self_rows": int(is_self.sum()),
        "threshold": threshold,
        "logit_shift": logit_shift,
        "ece_bins": ece_bins,
        "self_loops_included": True,
        "metrics": asdict(metrics),
    }

    if pairs_source not in _SPLIT_REPORTING_SOURCES:
        return report

    # Spec §9.4 rule 3, test view only: the split plus predicted-vs-reference
    # self-loop counts, reported as their own rows rather than folded into the
    # headline metrics above.
    report["metrics_self"] = _stratum_metrics(
        labels64[is_self],
        probs[is_self],
        threshold=threshold,
        ece_bins=ece_bins,
        name="self",
    )
    report["metrics_non_self"] = _stratum_metrics(
        labels64[~is_self],
        probs[~is_self],
        threshold=threshold,
        ece_bins=ece_bins,
        name="non_self",
    )
    report["self_loop_rate"] = {
        "self_rows": int(is_self.sum()),
        "reference_positive": int((labels64[is_self] == 1).sum()),
        "predicted_positive": int((probs[is_self] >= threshold).sum()),
    }
    return report


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
    parser.add_argument(
        "--logit-shift",
        type=float,
        default=0.0,
        help="added to every raw logit before the sigmoid (test-time calibration)",
    )
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
        logit_shift=args.logit_shift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Not sort_keys: the insertion order puts the self-loop-including headline
    # metrics ahead of any supplementary split, and that ordering is intentional.
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
