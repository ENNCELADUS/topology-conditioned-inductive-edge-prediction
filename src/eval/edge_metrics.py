"""Edge-level classification metrics for topology-conditioned edge prediction.

Operates on plain numpy arrays of binary labels and predicted probabilities. No
dependence on `src.data` or `torch` — this module is intentionally standalone so
it can be tested and reused without the training stack.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True)
class EdgeMetrics:
    """Bundle of edge-level classification metrics for one labeled prediction set.

    Attributes:
        auroc: Area under the ROC curve (ranking metric, threshold-free).
        auprc: Area under the precision-recall curve (ranking metric, threshold-free).
        accuracy: Fraction of correct predictions at `threshold`.
        sensitivity: True positive rate at `threshold` (a.k.a. recall).
        specificity: True negative rate at `threshold`.
        precision: Positive predictive value at `threshold`; 0.0 when there are zero
            predicted positives (never NaN).
        recall: Same as `sensitivity`, kept as a distinct field for reporting parity.
        f1: Harmonic mean of precision and recall at `threshold`; 0.0 when both are 0.
        mcc: Matthews correlation coefficient at `threshold`.
        ece: Expected calibration error over `ece_bins` equal-width bins on [0, 1].
        brier: Mean squared error between `probs` and `labels`.
        threshold: The decision threshold used for all threshold-derived fields.
        n_pos: Number of positive labels.
        n_neg: Number of negative labels.
    """

    auroc: float
    auprc: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    mcc: float
    ece: float
    brier: float
    threshold: float
    n_pos: int
    n_neg: int


@dataclass(frozen=True)
class F1ThresholdSelection:
    """The max-F1 decision threshold selected on one labeled score set.

    Attributes:
        logit_threshold: Decision rule ``score >= logit_threshold`` (a score value
            from the selection set; ties in F1 resolve toward the larger value).
        f1: F1 at that threshold on the selection set.
        precision: Precision at that threshold on the selection set.
        recall: Recall at that threshold on the selection set.
        n_rows: Number of rows the threshold was selected on.
    """

    logit_threshold: float
    f1: float
    precision: float
    recall: float
    n_rows: int


def select_max_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> F1ThresholdSelection:
    """Select the ``score >= t`` threshold maximizing F1 on a labeled score set.

    Every distinct score value is a candidate ``t``; the rule is the same
    ``>=`` comparison :func:`compute_edge_metrics` applies, so replaying the
    returned threshold on the selection set reproduces ``f1`` exactly.

    Args:
        labels: 1-D array of binary labels (0/1).
        scores: 1-D array of raw scores (logits), same length as `labels`.

    Returns:
        The selected threshold and its selection-set precision/recall/F1.

    Raises:
        ValueError: On a shape mismatch or single-class `labels`.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} must be 1-D and aligned")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"labels must contain both classes to select an F1 threshold; got n_pos={n_pos}, "
            f"n_neg={n_neg}"
        )
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    positive = (labels[order] == 1).astype(np.int64)
    # Predicting positive for every row with score >= v, for each distinct v:
    # the cumulative counts at the last row of each equal-score run.
    last_of_run = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    tp = np.cumsum(positive)[last_of_run]
    predicted = (np.arange(1, sorted_scores.size + 1))[last_of_run]
    thresholds = sorted_scores[last_of_run]
    f1 = 2.0 * tp / (predicted + n_pos)
    best = int(np.flatnonzero(f1 == f1.max())[0])  # thresholds descend, so first = largest
    return F1ThresholdSelection(
        logit_threshold=float(thresholds[best]),
        f1=float(f1[best]),
        precision=float(tp[best] / predicted[best]),
        recall=float(tp[best] / n_pos),
        n_rows=int(labels.size),
    )


def _expected_calibration_error(labels: np.ndarray, probs: np.ndarray, ece_bins: int) -> float:
    """Compute expected calibration error over equal-width bins on [0, 1].

    Per bin b: conf_b = mean predicted probability of samples in b, acc_b = mean
    label (empirical frequency of the positive class) of samples in b. Empty bins
    contribute nothing. ECE = sum_b (n_b / n) * |acc_b - conf_b|.
    """
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    # right=False bins as [edge_i, edge_i+1); force prob==1.0 into the last bin.
    bin_idx = np.digitize(probs, edges[1:-1], right=False)
    n = labels.shape[0]
    ece = 0.0
    for b in range(ece_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf_b = float(probs[mask].mean())
        acc_b = float(labels[mask].mean())
        ece += (n_b / n) * abs(acc_b - conf_b)
    return ece


def compute_edge_metrics(
    labels: np.ndarray, probs: np.ndarray, *, threshold: float = 0.5, ece_bins: int = 15
) -> EdgeMetrics:
    """Compute the full edge-metric bundle for one labeled prediction set.

    Args:
        labels: 1-D array of binary labels (0/1).
        probs: 1-D array of predicted probabilities in [0, 1], same length as `labels`.
        threshold: Decision threshold applied to `probs` for the confusion-matrix-derived
            fields (accuracy, sensitivity, specificity, precision, recall, f1, mcc).
        ece_bins: Number of equal-width bins on [0, 1] used for `ece`.

    Returns:
        An `EdgeMetrics` instance.

    Raises:
        ValueError: If `labels` contains only one class (AUROC/AUPRC/MCC are undefined);
            this is raised rather than silently emitting NaN.

    Conventions:
        - Zero predicted positives at `threshold` -> precision = 0.0 (never NaN).
        - Zero positives predicted and zero actual positives predicted correctly ->
          f1 = 0.0 (never NaN) whenever precision + recall == 0.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"labels must contain both classes to compute edge metrics; got n_pos={n_pos}, "
            f"n_neg={n_neg} (single-class labels are not supported)"
        )

    auroc = float(roc_auc_score(labels, probs))
    auprc = float(average_precision_score(labels, probs))

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    total = tn + fp + fn + tp
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mcc = float(matthews_corrcoef(labels, preds))
    brier = float(brier_score_loss(labels, probs))
    ece = _expected_calibration_error(labels, probs, ece_bins)

    return EdgeMetrics(
        auroc=auroc,
        auprc=auprc,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        ece=ece,
        brier=brier,
        threshold=float(threshold),
        n_pos=n_pos,
        n_neg=n_neg,
    )
