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


class TestLogitShiftCalibration:
    """Test-time calibration: sigma(l + shift) >= 0.5 iff l >= -shift."""

    def test_thresholded_metrics_are_shift_invariant(self, tmp_path: Path) -> None:
        """Shifting to 0.5 equals thresholding raw probs at sigma(t*)."""
        path = tmp_path / "test.npz"
        _write_artifact(path)
        t_star = 0.4

        shifted = report_edge_metrics(path, logit_shift=-t_star)["metrics"]
        raw = report_edge_metrics(path, threshold=float(1.0 / (1.0 + np.exp(-t_star))))["metrics"]

        assert isinstance(shifted, dict) and isinstance(raw, dict)
        for name in ("accuracy", "f1", "mcc", "sensitivity", "specificity", "auroc"):
            assert shifted[name] == pytest.approx(raw[name])

    def test_ece_and_brier_describe_the_calibrated_probabilities(self, tmp_path: Path) -> None:
        path = tmp_path / "test.npz"
        _write_artifact(path)

        unshifted = report_edge_metrics(path)["metrics"]
        shifted = report_edge_metrics(path, logit_shift=-3.0)["metrics"]

        assert isinstance(shifted, dict) and isinstance(unshifted, dict)
        assert shifted["brier"] != pytest.approx(unshifted["brier"])
        assert shifted["ece"] != pytest.approx(unshifted["ece"])

    def test_ranking_metrics_survive_shifted_sigmoid_saturation(self, tmp_path: Path) -> None:
        """AUROC/AUPRC use raw logits even when shifted probabilities all round to one."""
        path = tmp_path / "candidate.npz"
        _write_artifact(
            path,
            pairs_source="candidate",
            labels=np.array([0, 1, 0, 1], dtype=np.int8),
            logits=np.array([35.0, 36.0, 37.0, 38.0], dtype=np.float32),
        )

        unshifted = report_edge_metrics(path)["metrics"]
        saturated = report_edge_metrics(path, logit_shift=10.0)["metrics"]

        assert isinstance(unshifted, dict) and isinstance(saturated, dict)
        assert saturated["auroc"] == pytest.approx(0.75)
        assert saturated["auprc"] == pytest.approx(5 / 6)
        assert saturated["auroc"] == pytest.approx(unshifted["auroc"])
        assert saturated["auprc"] == pytest.approx(unshifted["auprc"])

    def test_logit_shift_is_echoed_beside_threshold(self, tmp_path: Path) -> None:
        path = tmp_path / "test.npz"
        _write_artifact(path)

        report = report_edge_metrics(path, logit_shift=-0.4)

        assert report["threshold"] == 0.5
        assert report["logit_shift"] == -0.4
        assert report_edge_metrics(path)["logit_shift"] == 0.0

    def test_cli_accepts_logit_shift(self, tmp_path: Path) -> None:
        scores = tmp_path / "test.npz"
        _write_artifact(scores)
        output = tmp_path / "out.json"

        main(["--scores", str(scores), "--output", str(output), "--logit-shift", "-0.4"])

        payload = json.loads(output.read_text())
        assert payload["logit_shift"] == -0.4


class TestSelfNonSelfSplit:
    """Spec §9.4 rule 3: overall metrics *and* the self / non-self split."""

    def _write_with_self_pairs(self, path: Path) -> None:
        """Write an artifact containing both ``(u, u)`` and ``(u, v)`` rows."""
        labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        logits = np.array([2.0, -1.5, 0.5, -0.25, 3.0, -2.0], dtype=np.float32)
        # Rows 0 and 1 are self-pairs; the rest are distinct-endpoint pairs.
        u_idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        v_idx = np.array([0, 1, 3, 4, 5, 6], dtype=np.int32)
        save_scores(
            path,
            node_ids=[f"node_{i:03d}" for i in range(7)],
            u_idx=u_idx,
            v_idx=v_idx,
            logit=logits,
            label=labels,
            row_start=0,
            meta={
                "checkpoint_id": "deadbeefcafe0000",
                "model_family": "v3_1",
                "pairs_source": "test",
                "strategy": "breadth_first",
                "num_rows": 6,
                "created_utc": "2026-08-03T00:00:00Z",
                "torch_version": "2.10.0+cu128",
            },
        )

    def test_reports_both_strata_and_self_loop_rate(self, tmp_path: Path) -> None:
        """Both strata and the self-loop-rate row are present."""
        path = tmp_path / "test.npz"
        self._write_with_self_pairs(path)

        report = report_edge_metrics(path)

        assert report["metrics"]["n_pos"] + report["metrics"]["n_neg"] == 6
        # Rows 0,1 are self (one positive, one negative) -> two-class, computable.
        assert report["metrics_self"] is not None
        assert report["metrics_self"]["n_pos"] == 1
        assert report["metrics_self"]["n_neg"] == 1
        assert report["metrics_non_self"]["n_pos"] == 2
        assert report["metrics_non_self"]["n_neg"] == 2
        assert report["self_loop_rate"] == {
            "self_rows": 2,
            "reference_positive": 1,
            "predicted_positive": 1,
        }

    def test_single_class_self_stratum_is_null_not_a_crash(self, tmp_path: Path) -> None:
        """An all-positive self stratum is emitted as null with the row still counted.

        This is the real benchmark's shape: every ``(u, u)`` row in
        ``test_edges.txt`` is positive, so AUROC/AUPRC are undefined there.
        """
        path = tmp_path / "test.npz"
        labels = np.array([1, 1, 1, 0, 1, 0], dtype=np.int8)
        save_scores(
            path,
            node_ids=[f"node_{i:03d}" for i in range(7)],
            u_idx=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            v_idx=np.array([0, 1, 3, 4, 5, 6], dtype=np.int32),
            logit=np.array([2.0, 1.5, 0.5, -0.25, 3.0, -2.0], dtype=np.float32),
            label=labels,
            row_start=0,
            meta={
                "checkpoint_id": "deadbeefcafe0000",
                "model_family": "v3_1",
                "pairs_source": "test",
                "strategy": "breadth_first",
                "num_rows": 6,
                "created_utc": "2026-08-03T00:00:00Z",
                "torch_version": "2.10.0+cu128",
            },
        )

        report = report_edge_metrics(path)

        assert report["metrics_self"] is None
        assert report["self_loop_rate"]["self_rows"] == 2
        assert report["self_loop_rate"]["reference_positive"] == 2
        assert report["metrics_non_self"] is not None

    def test_non_self_metrics_differ_from_aggregate(self, tmp_path: Path) -> None:
        """The split is not cosmetic: removing self rows changes the numbers.

        Self rows are perfectly ranked while the non-self stratum carries one
        inversion, so the aggregate AUROC is optimistic relative to non-self --
        the same direction seen on the real benchmark, where every ``(u, u)``
        row is a positive the model gets right for free.
        """
        path = tmp_path / "test.npz"
        save_scores(
            path,
            node_ids=[f"node_{i:03d}" for i in range(7)],
            u_idx=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            v_idx=np.array([0, 1, 3, 4, 5, 6], dtype=np.int32),
            # self: +2.0 (pos), -1.5 (neg) -> perfectly ranked.
            # non-self: 0.5 (pos) ranked below 1.0 (neg) -> one inversion.
            logit=np.array([2.0, -1.5, 0.5, 1.0, 3.0, -2.0], dtype=np.float32),
            label=np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
            row_start=0,
            meta={
                "checkpoint_id": "deadbeefcafe0000",
                "model_family": "v3_1",
                "pairs_source": "test",
                "strategy": "breadth_first",
                "num_rows": 6,
                "created_utc": "2026-08-03T00:00:00Z",
                "torch_version": "2.10.0+cu128",
            },
        )

        report = report_edge_metrics(path)

        assert report["metrics_non_self"]["auroc"] == pytest.approx(0.75)
        assert report["metrics"]["auroc"] == pytest.approx(8 / 9)
        assert report["metrics_non_self"]["auroc"] < report["metrics"]["auroc"]


class TestSplitScope:
    """Self-loops are kept everywhere; only the test view is also split."""

    @pytest.mark.parametrize("pairs_source", ["val", "candidate"])
    def test_non_test_views_report_headline_only(self, tmp_path: Path, pairs_source: str) -> None:
        """Val and candidate keep self-loops but emit no split."""
        path = tmp_path / f"{pairs_source}.npz"
        _write_artifact(path, pairs_source=pairs_source)

        report = report_edge_metrics(path)

        assert report["self_loops_included"] is True
        assert "metrics" in report
        assert "metrics_self" not in report
        assert "metrics_non_self" not in report
        assert "self_loop_rate" not in report

    def test_test_view_reports_both(self, tmp_path: Path) -> None:
        """The test view carries the split alongside the headline block."""
        path = tmp_path / "test.npz"
        _write_artifact(path, pairs_source="test")

        report = report_edge_metrics(path)

        assert "metrics" in report
        assert "metrics_non_self" in report
        assert "self_loop_rate" in report

    def test_self_loop_including_metrics_come_first(self, tmp_path: Path) -> None:
        """Key order is load-bearing: the headline precedes the split."""
        path = tmp_path / "test.npz"
        _write_artifact(path, pairs_source="test")

        keys = list(report_edge_metrics(path))

        assert keys.index("metrics") < keys.index("metrics_self")
        assert keys.index("metrics") < keys.index("metrics_non_self")
        assert keys.index("metrics") < keys.index("self_loop_rate")

    def test_cli_preserves_headline_first_ordering(self, tmp_path: Path) -> None:
        """The written JSON is not alphabetized, so the ordering survives."""
        scores = tmp_path / "test.npz"
        _write_artifact(scores, pairs_source="test")
        output = tmp_path / "out.json"

        main(["--scores", str(scores), "--output", str(output)])

        keys = list(json.loads(output.read_text()))
        assert keys.index("metrics") < keys.index("metrics_non_self")


class TestShardGuard:
    """A partial shard must not be reported as the full pair view."""

    def test_rejects_row_count_below_declared_num_rows(self, tmp_path: Path) -> None:
        """Metadata declaring more rows than loaded means an unmerged shard."""
        path = tmp_path / "shard.npz"
        _write_artifact(path)
        # Simulate a shard: same rows, metadata claiming the full universe.
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(str(arrays["meta"]))
        meta["num_rows"] = 2_037_171
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez_compressed(path, **arrays)

        with pytest.raises(ValueError, match="partial shard"):
            report_edge_metrics(path)


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
