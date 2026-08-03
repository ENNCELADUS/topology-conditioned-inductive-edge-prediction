"""Tests for the balanced-test edge-metrics reporting CLI.

This module is the committed replacement for an inline ``python -c`` recipe that
lived only in a legacy run script, so the tests pin the two things that recipe
never checked: that the artifact's ``pairs_source`` matches what the caller
claims to be reporting, and that unlabeled rows can never silently enter a
classification metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.eval.edge_metrics import compute_edge_metrics
from src.eval.report_edge_metrics import main, report_edge_metrics
from src.score_universe import save_scores

pytestmark = pytest.mark.unit


def _write_artifact(
    path: Path,
    *,
    pairs_source: str = "test",
    labels: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> np.ndarray:
    """Write a minimal v3_1 scores artifact and return its labels."""
    if labels is None:
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    if logits is None:
        logits = np.array([2.0, -1.5, 0.5, -0.25, 3.0, -2.0], dtype=np.float32)
    n = labels.size
    node_ids = [f"node_{i:03d}" for i in range(n + 1)]
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=np.arange(n, dtype=np.int32),
        v_idx=np.arange(1, n + 1, dtype=np.int32),
        logit=logits,
        label=labels,
        row_start=0,
        meta={
            "checkpoint_id": "deadbeefcafe0000",
            "model_family": "v3_1",
            "pairs_source": pairs_source,
            "strategy": "breadth_first",
            "num_rows": n,
            "created_utc": "2026-08-03T00:00:00Z",
            "torch_version": "2.10.0+cu128",
        },
    )
    return labels


class TestReportEdgeMetrics:
    """`report_edge_metrics` behavior."""

    def test_computes_metrics_and_provenance(self, tmp_path: Path) -> None:
        """The report carries the metric block plus checkpoint provenance."""
        path = tmp_path / "test.npz"
        _write_artifact(path)

        report = report_edge_metrics(path)

        assert report["pairs_source"] == "test"
        assert report["model_family"] == "v3_1"
        assert report["checkpoint_id"] == "deadbeefcafe0000"
        assert report["strategy"] == "breadth_first"
        assert report["num_rows"] == 6
        assert report["threshold"] == 0.5
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["n_pos"] == 3
        assert metrics["n_neg"] == 3
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_matches_compute_edge_metrics_directly(self, tmp_path: Path) -> None:
        """The CLI adds provenance but does not alter the metric math."""
        path = tmp_path / "test.npz"
        logits = np.array([2.0, -1.5, 0.5, -0.25, 3.0, -2.0], dtype=np.float32)
        labels = _write_artifact(path, logits=logits)

        report = report_edge_metrics(path)

        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        expected = compute_edge_metrics(labels.astype(np.int64), probs)
        assert report["metrics"]["auroc"] == pytest.approx(expected.auroc)
        assert report["metrics"]["auprc"] == pytest.approx(expected.auprc)
        assert report["metrics"]["brier"] == pytest.approx(expected.brier)

    def test_accepts_matching_pairs_source(self, tmp_path: Path) -> None:
        """An explicit matching expectation passes through."""
        path = tmp_path / "test.npz"
        _write_artifact(path, pairs_source="test")
        report = report_edge_metrics(path, expect_pairs_source="test")
        assert report["pairs_source"] == "test"

    def test_rejects_mismatched_pairs_source(self, tmp_path: Path) -> None:
        """Reporting a candidate artifact as the balanced test view is blocked."""
        path = tmp_path / "candidate.npz"
        _write_artifact(path, pairs_source="candidate")
        with pytest.raises(ValueError, match="pairs_source expected 'test'"):
            report_edge_metrics(path, expect_pairs_source="test")

    def test_rejects_unlabeled_rows(self, tmp_path: Path) -> None:
        """Rows the scorer could not label cannot enter a classification metric."""
        path = tmp_path / "test.npz"
        _write_artifact(path, labels=np.array([1, 0, -1, 0, 1, 1], dtype=np.int8))
        with pytest.raises(ValueError, match="unlabeled"):
            report_edge_metrics(path)

    def test_single_class_still_raises(self, tmp_path: Path) -> None:
        """A degenerate all-positive set raises rather than emitting NaN."""
        path = tmp_path / "test.npz"
        _write_artifact(path, labels=np.ones(6, dtype=np.int8))
        with pytest.raises(ValueError, match="both classes"):
            report_edge_metrics(path)

    def test_threshold_is_honored(self, tmp_path: Path) -> None:
        """A non-default threshold changes the thresholded fields."""
        path = tmp_path / "test.npz"
        _write_artifact(path)

        low = report_edge_metrics(path, threshold=0.1)
        high = report_edge_metrics(path, threshold=0.9)

        assert low["metrics"]["sensitivity"] >= high["metrics"]["sensitivity"]
        assert low["threshold"] == 0.1
        assert high["threshold"] == 0.9


class TestCli:
    """The ``python -m src.eval.report_edge_metrics`` entry point."""

    def test_writes_json_report(self, tmp_path: Path) -> None:
        """The CLI writes a sorted, indented JSON report to --output."""
        scores = tmp_path / "test.npz"
        _write_artifact(scores)
        output = tmp_path / "nested" / "test_edge_metrics.json"

        main(
            [
                "--scores",
                str(scores),
                "--output",
                str(output),
                "--expect-pairs-source",
                "test",
            ]
        )

        payload = json.loads(output.read_text())
        assert payload["pairs_source"] == "test"
        assert payload["metrics"]["n_pos"] == 3
        assert payload["scores_path"] == str(scores)

    def test_cli_propagates_pairs_source_mismatch(self, tmp_path: Path) -> None:
        """A guard failure surfaces instead of writing a misleading report."""
        scores = tmp_path / "candidate.npz"
        _write_artifact(scores, pairs_source="candidate")
        output = tmp_path / "out.json"

        with pytest.raises(ValueError, match="pairs_source"):
            main(
                [
                    "--scores",
                    str(scores),
                    "--output",
                    str(output),
                    "--expect-pairs-source",
                    "test",
                ]
            )
        assert not output.exists()
