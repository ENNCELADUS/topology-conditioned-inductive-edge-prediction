"""Validation-only diagnostics for the two-stage topology prompt.

Inputs must share a canonical pair universe. Thresholds are checkpoint-specific;
an affine logit change alone is not evidence of a topology change.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def score_stability(
    pairs: Sequence[tuple[str, str]],
    previous_logits: np.ndarray,
    logits: np.ndarray,
    previous_threshold: float,
    threshold: float,
) -> dict[str, object]:
    """Compare aligned rows, including self-loops once in predicted degrees."""
    before = np.asarray(previous_logits, dtype=np.float64).reshape(-1)
    after = np.asarray(logits, dtype=np.float64).reshape(-1)
    if before.shape != after.shape or len(pairs) != len(before) or not len(before):
        raise ValueError("stability requires the same non-empty row universe")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("non-finite stability logits")
    canonical = [tuple(sorted(pair)) for pair in pairs]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate canonical pairs in stability universe")
    slope, intercept = np.linalg.lstsq(
        np.column_stack((before, np.ones_like(before))), after, rcond=None
    )[0]
    residual = after - (slope * before + intercept)
    old, new = before >= previous_threshold, after >= threshold
    union = int((old | new).sum())
    degrees: dict[str, int] = {}
    for (u, v), delta in zip(pairs, new.astype(int) - old.astype(int), strict=True):
        degrees[u] = degrees.get(u, 0) + int(delta)
        if v != u:
            degrees[v] = degrees.get(v, 0) + int(delta)
    return {
        "stability_edge_jaccard": float((old & new).sum() / union) if union else 1.0,
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
        "affine_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "degree_changes": degrees,
    }


def sensitivity_row(
    clean_logits: np.ndarray,
    perturbed_logits: np.ndarray,
    cls_labels: np.ndarray,
    cls_logits: np.ndarray,
    fixed_threshold: float,
    topology_evaluator: Callable[[np.ndarray, float | None], dict[str, object]],
) -> dict[str, object]:
    """Report a perturbation at clean and freshly selected topology thresholds.

    ``topology_evaluator(scores, None)`` must use the existing closest-geometric-RD
    selector; a numeric second argument replays that threshold without selection.
    """
    clean, perturbed = np.asarray(clean_logits), np.asarray(perturbed_logits)
    labels, scores = np.asarray(cls_labels), np.asarray(cls_logits)
    if clean.shape != perturbed.shape or labels.shape != scores.shape or not clean.size:
        raise ValueError("sensitivity rows must be aligned and non-empty")
    if not all(np.isfinite(x).all() for x in (clean, perturbed, labels, scores)):
        raise ValueError("non-finite sensitivity inputs")
    change = np.abs(perturbed - clean)
    probabilities = np.exp(-np.logaddexp(0.0, -scores))
    return {
        "cls_auprc": float(average_precision_score(labels, scores)),
        "cls_bce": float(np.mean(np.logaddexp(0.0, scores) - labels * scores)),
        "cls_brier": float(np.mean((probabilities - labels) ** 2)),
        "logit_change_mean": float(change.mean()),
        "logit_change_p95": float(np.quantile(change, 0.95)),
        "topology_fixed": topology_evaluator(perturbed, fixed_threshold),
        "topology_reselected": topology_evaluator(perturbed, None),
    }


def main() -> None:
    """Compare saved V_val score artifacts, joining canonical identities first."""
    from src.score_universe import load_scores

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--previous-threshold", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous, current = load_scores(args.previous), load_scores(args.current)
    old_pairs = [(min(u, v), max(u, v)) for u, v in previous.pairs()]
    new_pairs = [(min(u, v), max(u, v)) for u, v in current.pairs()]
    if len(set(new_pairs)) != len(new_pairs) or set(old_pairs) != set(new_pairs):
        raise ValueError("score artifacts do not have the same unique canonical universe")
    index = {p: i for i, p in enumerate(new_pairs)}
    row = score_stability(
        old_pairs,
        previous.logit,
        current.logit[np.array([index[p] for p in old_pairs])],
        args.previous_threshold,
        args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
