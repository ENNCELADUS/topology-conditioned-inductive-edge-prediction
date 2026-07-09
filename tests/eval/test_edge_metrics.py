"""Tests for src.eval.edge_metrics: threshold-based and ranking edge-level metrics."""

import numpy as np
import pytest
from src.eval.edge_metrics import EdgeMetrics, compute_edge_metrics


@pytest.mark.unit
class TestComputeEdgeMetricsPerfectSeparation:
    """A 10-element array with perfect rank separation and hand-computed values."""

    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])

    def test_returns_edge_metrics_instance(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs)
        assert isinstance(result, EdgeMetrics)

    def test_auroc_is_one_for_perfect_separation(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs)
        assert result.auroc == pytest.approx(1.0)

    def test_auprc_is_one_for_perfect_separation(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs)
        assert result.auprc == pytest.approx(1.0)

    def test_thresholded_metrics_at_default_threshold(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs, threshold=0.5)
        assert result.accuracy == pytest.approx(1.0)
        assert result.sensitivity == pytest.approx(1.0)
        assert result.specificity == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.mcc == pytest.approx(1.0)

    def test_brier_hand_computed(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs)
        # sum((p - y)^2) / n, hand-computed = 1.005 / 10
        assert result.brier == pytest.approx(0.1005)

    def test_n_pos_n_neg_recorded(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs)
        assert result.n_pos == 5
        assert result.n_neg == 5

    def test_threshold_recorded(self) -> None:
        result = compute_edge_metrics(self.labels, self.probs, threshold=0.5)
        assert result.threshold == pytest.approx(0.5)


@pytest.mark.unit
class TestEceKnownValue:
    """A 4-element array with 2 equal-width bins, hand-computed ECE."""

    def test_ece_hand_computed_two_bins(self) -> None:
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.9, 0.8, 0.3, 0.2])
        result = compute_edge_metrics(labels, probs, ece_bins=2)
        # bin [0, 0.5): probs {0.3, 0.2}, labels {1, 0} -> conf=0.25, acc=0.5, weight=0.5
        # bin [0.5, 1]: probs {0.9, 0.8}, labels {1, 0} -> conf=0.85, acc=0.5, weight=0.5
        # ECE = 0.5*|0.5-0.25| + 0.5*|0.5-0.85| = 0.125 + 0.175 = 0.30
        assert result.ece == pytest.approx(0.30)


@pytest.mark.unit
class TestEdgeCaseConventions:
    def test_single_class_labels_raises_value_error(self) -> None:
        labels = np.array([1, 1, 1, 1])
        probs = np.array([0.9, 0.8, 0.7, 0.6])
        with pytest.raises(ValueError, match="class"):
            compute_edge_metrics(labels, probs)

    def test_single_class_negative_labels_raises_value_error(self) -> None:
        labels = np.array([0, 0, 0, 0])
        probs = np.array([0.9, 0.8, 0.7, 0.6])
        with pytest.raises(ValueError, match="class"):
            compute_edge_metrics(labels, probs)

    def test_zero_predicted_positives_precision_convention(self) -> None:
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.2, 0.1, 0.3, 0.05])
        result = compute_edge_metrics(labels, probs, threshold=0.5)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.accuracy == pytest.approx(0.5)
        assert not np.isnan(result.precision)

    def test_frozen_dataclass_immutable(self) -> None:
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2])
        result = compute_edge_metrics(labels, probs)
        with pytest.raises(Exception):  # noqa: B017, PT011 - frozen dataclass raises FrozenInstanceError
            result.auroc = 0.0  # type: ignore[misc]
