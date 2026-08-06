"""Tests for src.eval.test_protocol: the per-arm held-out test protocol.

Builds synthetic v_hold/test/candidate artifacts and a tiny benchmark fixture
(reusing the artifact/graph builders already committed for
`tests.test_g1_hardened_e2`), then drives `run_test_protocol` with a fake
`score_runner` so nothing here depends on `src.score_fanout.score_sharded`
(still a stub owned by a concurrent agent) or on any real GPU.

`_write_checkpoint` writes a real (if minimal) torch checkpoint -- not just
arbitrary bytes -- because `run_test_protocol` now reads a checkpoint's own
embedded ``model_family`` (the Task-4 self-describing contract) to decide
whether the test/candidate passes may carry ``--scoring-run-id``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from src.eval import test_protocol
from src.eval.edge_metrics import compute_edge_metrics
from src.eval.test_protocol import run_test_protocol
from src.score_universe import ScoresArtifact, validate_artifact_precision

from tests.test_g1_hardened_e2 import (
    _NODES,
    _POSITIVE_EDGES,
    _logit_for_prob,
    _make_reference_graph,
    _small_buckets,
    _universe_rows,
    _write_benchmark,
    _write_universe_npz,
)

pytestmark = pytest.mark.unit

_CHECKPOINT_ID = "deadbeefcafefeed"
_STRATEGY = "toy"


def _arg_value(args: Sequence[str], flag: str) -> str | None:
    """Return the value following `flag` in `args`, or ``None`` if absent."""
    for i, token in enumerate(args):
        if token == flag:
            return args[i + 1]
    return None


class _FakeScoreRunner:
    """Records every call (in order) and serves a pre-built artifact per pass.

    Structurally satisfies `src.eval.test_protocol.ScoreRunner`.
    """

    def __init__(self, artifacts: dict[str, Path], *, fail_on: str | None = None) -> None:
        self._artifacts = artifacts
        self._fail_on = fail_on
        self.calls: list[list[str]] = []
        self.pairs_order: list[str] = []

    def __call__(self, score_args: Sequence[str]) -> Path:
        args = list(score_args)
        self.calls.append(args)
        pairs = _arg_value(args, "--pairs")
        assert pairs is not None
        self.pairs_order.append(pairs)
        if pairs == self._fail_on:
            raise RuntimeError(f"synthetic failure scoring {pairs!r}")
        return self._artifacts[pairs]

    def call_for(self, pairs_source: str) -> list[str]:
        """Return the captured args for the first call scoring `pairs_source`."""
        for args in self.calls:
            if _arg_value(args, "--pairs") == pairs_source:
                return args
        raise AssertionError(f"no call recorded for pairs={pairs_source!r}")


@dataclass
class _Fixture:
    data_root: Path
    artifacts: dict[str, Path]
    expected_threshold: float


def _build_candidate_and_test_probs(pairs: list[tuple[str, str]]) -> NDArray[np.float64]:
    """High prob on the five true edges, moderate on self-pairs, low elsewhere.

    Chosen so the density-matched threshold (target_edges=5, the reference's
    non-self edge count) lands exactly on the clean 0.9/0.1 gap and reproduces
    the reference graph exactly -- a hand-checkable happy path.
    """
    positive_set = {frozenset(edge) for edge in _POSITIVE_EDGES}
    probs = [
        0.5 if u == v else (0.9 if frozenset((u, v)) in positive_set else 0.1) for u, v in pairs
    ]
    return np.array(probs, dtype=np.float64)


def _reference_max_f1_threshold(labels: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    """Independent reference for the max-F1 threshold: scan every unique prob.

    Ties broken toward the smallest threshold, matching
    `test_protocol._select_max_f1_threshold`'s documented tie-break.
    """
    best_threshold: float | None = None
    best_f1 = -1.0
    for t in sorted(set(probs.tolist())):
        metrics = compute_edge_metrics(labels, probs, threshold=t)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = t
    assert best_threshold is not None
    return best_threshold


def _build_fixture(tmp_path: Path) -> _Fixture:
    """Build a full v_hold/test/candidate + benchmark fixture sharing one node universe."""
    g_ref = _make_reference_graph()
    buckets = _small_buckets(_NODES, size=5, n_samples=4, seed=0)
    data_root = _write_benchmark(tmp_path, _STRATEGY, g_ref, buckets)

    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    probs = _build_candidate_and_test_probs(pairs)
    logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)

    scores_dir = tmp_path / "prebuilt_scores"
    candidate_path = scores_dir / "candidate.npz"
    _write_universe_npz(
        candidate_path,
        node_ids=_NODES,
        pairs=pairs,
        logits=logits,
        labels=labels,
        strategy=_STRATEGY,
        pairs_source="candidate",
        checkpoint_id=_CHECKPOINT_ID,
    )
    test_path = scores_dir / "test.npz"
    _write_universe_npz(
        test_path,
        node_ids=_NODES,
        pairs=pairs,
        logits=logits,
        labels=labels,
        strategy=_STRATEGY,
        pairs_source="test",
        checkpoint_id=_CHECKPOINT_ID,
    )

    # V_hold: a small, independently labeled set with a non-trivial max-F1
    # threshold (a false positive sits strictly between two true positives).
    v_hold_labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    v_hold_probs = np.array([0.9, 0.8, 0.6, 0.7, 0.4, 0.3, 0.1])
    v_hold_logits = np.array([_logit_for_prob(p) for p in v_hold_probs], dtype=np.float32)
    v_hold_pairs = [(f"v{i}", f"v{i}_partner") for i in range(len(v_hold_labels))]
    v_hold_node_ids = sorted({node for pair in v_hold_pairs for node in pair})
    v_hold_path = scores_dir / "v_hold.npz"
    _write_universe_npz(
        v_hold_path,
        node_ids=v_hold_node_ids,
        pairs=v_hold_pairs,
        logits=v_hold_logits,
        labels=v_hold_labels,
        strategy=_STRATEGY,
        pairs_source="v_hold",
        checkpoint_id=_CHECKPOINT_ID,
    )

    expected_threshold = _reference_max_f1_threshold(v_hold_labels.astype(np.int64), v_hold_probs)

    return _Fixture(
        data_root=data_root,
        artifacts={"v_hold": v_hold_path, "test": test_path, "candidate": candidate_path},
        expected_threshold=expected_threshold,
    )


def _write_checkpoint(tmp_path: Path, *, model_family: str = "v3_1") -> Path:
    """Write a minimal but real Task-4-format checkpoint (self-describing family).

    `run_test_protocol` reads this checkpoint's own embedded `model_family` (a
    plain torch.load, no model construction) to decide whether the
    test/candidate passes may carry `--scoring-run-id`; a bare byte blob is no
    longer a valid fixture checkpoint.
    """
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_family": model_family}, checkpoint)
    return checkpoint


# --------------------------------------------------------------------------- tests


class TestRunTestProtocol:
    def test_full_report_shape_ordering_and_leakage_guarantee(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        result = run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        # 1. Pass ordering: v_hold strictly before test and candidate, test before candidate.
        assert runner.pairs_order == ["v_hold", "test", "candidate"]

        # 2. The operating point is frozen to disk; this file's existence at all
        # is the leakage guarantee (it can only have been written by the code
        # that ran between the v_hold and test score_runner calls).
        op_point_path = output_dir / "operating_point.json"
        assert op_point_path.exists()
        on_disk_op_point = json.loads(op_point_path.read_text(encoding="utf-8"))
        assert on_disk_op_point["threshold"] == pytest.approx(fixture.expected_threshold)

        # 3. --data-root/--strategy/--checkpoint and --run-metadata forwarded to every
        # pass, pointed at this module's own scoring-identity file (never the
        # published run_metadata.json -- see test_never_overwrites_published_run_metadata).
        for pairs_source in ("v_hold", "test", "candidate"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--checkpoint") == str(checkpoint)
            assert _arg_value(call, "--data-root") == str(fixture.data_root)
            assert _arg_value(call, "--strategy") == _STRATEGY
            assert _arg_value(call, "--run-metadata") == str(
                output_dir / "test_protocol_run_metadata.json"
            )

        report = result.report
        assert result.report_path == output_dir / "test_report.json"
        on_disk = json.loads(result.report_path.read_text(encoding="utf-8"))
        assert on_disk == report

        # 4. Pinned top-level key order.
        assert list(report.keys()) == [
            "schema_version",
            "arm",
            "operating_point",
            "edge",
            "graph",
            "sweep",
            "provenance",
        ]
        assert report["schema_version"] == "test_protocol_v1"

        arm_block = report["arm"]
        assert isinstance(arm_block, dict)
        assert arm_block["arm"] == "full"
        assert arm_block["seed"] == 0
        assert arm_block["model_family"] == "v3_1"
        assert arm_block["checkpoint_id"] == _CHECKPOINT_ID
        assert arm_block["checkpoint_sha256"] == test_protocol._sha256_file(checkpoint)

        # 5. Operating point: rule, threshold, and the V_hold metrics behind it.
        op = report["operating_point"]
        assert isinstance(op, dict)
        assert op["rule"] == "max_f1_on_v_hold"
        assert op["threshold"] == pytest.approx(fixture.expected_threshold)
        op_metrics = op["metrics"]
        assert isinstance(op_metrics, dict)
        assert op_metrics["threshold"] == pytest.approx(fixture.expected_threshold)

        # 6. Edge block: report_edge_metrics's exact shape, self-loop headline first.
        edge = report["edge"]
        assert isinstance(edge, dict)
        assert edge["pairs_source"] == "test"
        edge_keys = list(edge.keys())
        assert edge_keys[0] == "scores_path"
        assert edge_keys.index("metrics") < edge_keys.index("metrics_self")
        assert edge_keys.index("metrics_self") < edge_keys.index("metrics_non_self")
        assert edge_keys.index("metrics_non_self") < edge_keys.index("self_loop_rate")
        # Self stratum is degenerate by construction (every self-pair is
        # label 0 in this fixture); report_edge_metrics discloses that as an
        # explicit null rather than dropping the key.
        assert edge["metrics_self"] is None
        assert edge["metrics_non_self"] is not None

        # 7. Graph block: both metric families, both thresholds, five numbers
        # each, GS/RD named separately at the global-simple-edge and
        # BFS-macro granularity, never aggregated into a summary score.
        graph = report["graph"]
        assert isinstance(graph, dict)
        assert set(graph.keys()) == {"v_hold_selected_threshold", "density_matched_threshold"}
        for block_name in ("v_hold_selected_threshold", "density_matched_threshold"):
            block = graph[block_name]
            assert isinstance(block, dict)
            assert set(block["graph_similarity"]) == {"global_simple_edge", "bfs_macro"}
            assert set(block["relative_density"]) == {"global_simple_edge", "bfs_macro"}
            assert set(block["mmd_ratio"]) == {"degree", "clustering", "spectral"}

        dm = graph["density_matched_threshold"]
        assert dm["target_edges"] == 5
        # A clean 0.9/0.1 gap at exactly 5 true edges: the density-matched
        # assembly reproduces the reference graph exactly.
        assert dm["threshold"] == pytest.approx(0.9)
        assert dm["graph_similarity"]["global_simple_edge"] == pytest.approx(1.0)
        assert dm["relative_density"]["global_simple_edge"] == pytest.approx(1.0)

        vs = graph["v_hold_selected_threshold"]
        assert vs["threshold"] == pytest.approx(fixture.expected_threshold)

        # 8. Sweep rows present.
        assert isinstance(report["sweep"], list)
        assert len(report["sweep"]) > 0
        for row in report["sweep"]:
            assert {"threshold", "recall", "graph_similarity", "relative_density", "mmd_ratio"} <= (
                set(row)
            )

        # 9. Provenance: three artifact paths, sha256, and ledger digests
        # (null here -- v3_1 never claims the held-out E2E universe).
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        for pass_name, path in fixture.artifacts.items():
            entry = provenance[pass_name]
            assert entry["path"] == str(path)
            assert entry["sha256"] == test_protocol._sha256_file(path)
            assert entry["test_access_ledger_record_sha256"] is None

    def test_failing_candidate_pass_leaves_no_report_written(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts, fail_on="candidate")

        with pytest.raises(RuntimeError, match="candidate"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )

        # v_hold and test both ran (and test's edge report was at least attempted
        # to be built up to the point candidate failed); the report is absent.
        assert runner.pairs_order == ["v_hold", "test", "candidate"]
        assert (output_dir / "operating_point.json").exists()
        assert not (output_dir / "test_report.json").exists()

    def test_failing_test_pass_leaves_no_report_written(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts, fail_on="test")

        with pytest.raises(RuntimeError, match="test"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )

        assert runner.pairs_order == ["v_hold", "test"]
        assert (output_dir / "operating_point.json").exists()
        assert not (output_dir / "test_report.json").exists()

    def test_forwards_optional_flags_with_rescore_reason_only_on_heldout_passes(
        self, tmp_path: Path
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "control_6a_seed3"
        runner = _FakeScoreRunner(fixture.artifacts)
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="structure_control_6a_v3",
            seed=3,
            score_runner=runner,
            pack_dir=pack_dir,
            scaffold_control="shuffle_within_pair_v3",
            rescore_reason="repeat scoring for gate re-run",
        )

        for pairs_source in ("v_hold", "test", "candidate"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--pack-dir") == str(pack_dir)
            assert _arg_value(call, "--scaffold-control") == "shuffle_within_pair_v3"

        v_hold_call = runner.call_for("v_hold")
        assert "--rescore-reason" not in v_hold_call
        for pairs_source in ("test", "candidate"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--rescore-reason") == "repeat scoring for gate re-run"

    def test_omitted_optional_flags_are_never_forwarded(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        for call in runner.calls:
            assert "--pack-dir" not in call
            assert "--scaffold-control" not in call
            assert "--rescore-reason" not in call
            assert "--scoring-run-id" not in call
            assert "--allow-oracle-diagnostic" not in call
            assert "--model-family" not in call
            assert "--model-config" not in call

    def test_module_never_references_validate_score_precision_directly(self) -> None:
        """CLAUDE.md trap: validate_score_precision spuriously raises "missing arrays".

        On an egostitch_e2e artifact only validate_artifact_precision is the
        correct entry point. Absence of the name from this module's namespace
        is the guarantee: nothing here imports it.
        """
        assert not hasattr(test_protocol, "validate_score_precision")

    def test_validate_artifact_precision_is_exercised_for_every_loaded_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        calls: list[str] = []

        def _spy(artifact: ScoresArtifact, *, label: str) -> None:
            calls.append(label)
            validate_artifact_precision(artifact, label=label)

        # String target (rather than `monkeypatch.setattr(test_protocol, ...)`)
        # so this spy binds to the same module-global name run_test_protocol
        # resolves at call time, without re-exporting the name from
        # src.eval.test_protocol's own public surface.
        monkeypatch.setattr("src.eval.test_protocol.validate_artifact_precision", _spy)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        # run_test_protocol itself validates v_hold, the reloaded test artifact,
        # and candidate -- report_edge_metrics validates test a second time
        # through its own bound import, which this spy (patched only on
        # test_protocol's namespace) does not observe.
        assert calls == [
            str(fixture.artifacts["v_hold"]),
            str(fixture.artifacts["test"]),
            str(fixture.artifacts["candidate"]),
        ]

    def test_rejects_mismatched_checkpoint_across_passes(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"

        # Re-write the candidate artifact under a different checkpoint_id.
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        probs = _build_candidate_and_test_probs(pairs)
        logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)
        mismatched_candidate = tmp_path / "prebuilt_scores" / "candidate_mismatched.npz"
        _write_universe_npz(
            mismatched_candidate,
            node_ids=_NODES,
            pairs=pairs,
            logits=logits,
            labels=labels,
            strategy=_STRATEGY,
            pairs_source="candidate",
            checkpoint_id="0000000000000000",
        )
        artifacts = dict(fixture.artifacts)
        artifacts["candidate"] = mismatched_candidate
        runner = _FakeScoreRunner(artifacts)

        with pytest.raises(ValueError, match="different checkpoints"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )
        assert not (output_dir / "test_report.json").exists()

    def test_rejects_v_hold_artifact_with_wrong_pairs_source(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"

        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        probs = _build_candidate_and_test_probs(pairs)
        logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)
        wrong_source_v_hold = tmp_path / "prebuilt_scores" / "v_hold_wrong_source.npz"
        _write_universe_npz(
            wrong_source_v_hold,
            node_ids=_NODES,
            pairs=pairs,
            logits=logits,
            labels=labels,
            strategy=_STRATEGY,
            pairs_source="candidate",  # wrong: run_test_protocol expects "v_hold"
            checkpoint_id=_CHECKPOINT_ID,
        )
        artifacts = dict(fixture.artifacts)
        artifacts["v_hold"] = wrong_source_v_hold
        runner = _FakeScoreRunner(artifacts)

        with pytest.raises(ValueError, match="pairs_source"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )
        assert not (output_dir / "test_report.json").exists()


class TestReuseExistingScores:
    """Resuming a run whose later pass failed must not redo the finished ones."""

    def test_reuses_written_artifacts_and_only_scores_what_is_missing(self, tmp_path: Path) -> None:
        """Mimics the real failure: v_hold and test finished, candidate did not."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "resume"
        scores_dir = output_dir / "scores"
        scores_dir.mkdir(parents=True)
        for pairs_source in ("v_hold", "test"):
            (scores_dir / f"{pairs_source}.npz").write_bytes(
                fixture.artifacts[pairs_source].read_bytes()
            )

        runner = _FakeScoreRunner(fixture.artifacts)
        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
            reuse_existing_scores=True,
        )

        assert runner.pairs_order == ["candidate"]

    def test_reuse_is_opt_in(self, tmp_path: Path) -> None:
        """Implicit reuse would silently serve stale scores after a code fix."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "no_reuse"
        scores_dir = output_dir / "scores"
        scores_dir.mkdir(parents=True)
        for pairs_source in ("v_hold", "test"):
            (scores_dir / f"{pairs_source}.npz").write_bytes(
                fixture.artifacts[pairs_source].read_bytes()
            )

        runner = _FakeScoreRunner(fixture.artifacts)
        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        assert runner.pairs_order == ["v_hold", "test", "candidate"]


class TestPublishedRunMetadataNeverClobbered:
    """P1 fix #2: this module must never overwrite a published run_metadata.json."""

    def test_never_overwrites_published_run_metadata(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        published = {
            "status": "published",
            "checkpoint_id": _CHECKPOINT_ID,
            "arm": "full",
            "seed": 0,
            "formal_artifacts_published": True,
        }
        published_path = output_dir / "run_metadata.json"
        published_path.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
        published_bytes_before = published_path.read_bytes()
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        # Byte-identical: run_test_protocol never touched this file.
        assert published_path.read_bytes() == published_bytes_before
        # score_universe was pointed at this module's OWN file instead.
        scoring_identity_path = output_dir / "test_protocol_run_metadata.json"
        assert scoring_identity_path.exists()
        assert scoring_identity_path != published_path
        for pairs_source in ("v_hold", "test", "candidate"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--run-metadata") == str(scoring_identity_path)
        assert json.loads(scoring_identity_path.read_text(encoding="utf-8")) == {
            "arm": "full",
            "seed": 0,
        }

    def test_run_kind_comes_from_the_published_metadata(self, tmp_path: Path) -> None:
        """`score_universe` never writes `run_kind` into score metadata.

        The published training metadata is the only surviving record of a run's
        formal/diagnostic classification, so reading it off the scored artifact
        would silently null out that provenance in every report.
        """
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps({"arm": "full", "seed": 0, "run_kind": "diagnostic"}) + "\n",
            encoding="utf-8",
        )

        result = run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=_FakeScoreRunner(fixture.artifacts),
        )

        arm_block = result.report["arm"]
        assert isinstance(arm_block, dict)
        assert arm_block["run_kind"] == "diagnostic"

    def test_run_kind_is_none_without_published_metadata(self, tmp_path: Path) -> None:
        """A standalone scoring run (no published training dir) has no run kind."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "control_6a"
        output_dir.mkdir(parents=True)

        result = run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=_FakeScoreRunner(fixture.artifacts),
        )

        arm_block = result.report["arm"]
        assert isinstance(arm_block, dict)
        assert arm_block["run_kind"] is None

    def test_mismatched_published_arm_raises_before_any_scoring(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        published_path = output_dir / "run_metadata.json"
        published_path.write_text(
            json.dumps({"arm": "some_other_arm", "seed": 0}) + "\n", encoding="utf-8"
        )
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="contradicts --arm"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )
        assert runner.calls == []
        assert not (output_dir / "test_report.json").exists()

    def test_mismatched_published_seed_raises_before_any_scoring(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        published_path = output_dir / "run_metadata.json"
        published_path.write_text(json.dumps({"arm": "full", "seed": 7}) + "\n", encoding="utf-8")
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="contradicts --seed"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )
        assert runner.calls == []

    def test_published_metadata_without_arm_or_seed_is_not_a_conflict(self, tmp_path: Path) -> None:
        """`src.train_b0`'s own run_metadata.json carries neither field."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        published_path = output_dir / "run_metadata.json"
        published_path.write_text(
            json.dumps({"config_hash": "abc123", "checkpoint_id": _CHECKPOINT_ID}) + "\n",
            encoding="utf-8",
        )
        published_bytes_before = published_path.read_bytes()
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        assert published_path.read_bytes() == published_bytes_before
        assert runner.pairs_order == ["v_hold", "test", "candidate"]


class TestUniverseScopedCaches:
    """P1 fix #3: v_hold/test/candidate must never share an F0/grounding cache."""

    def test_three_passes_get_three_distinct_cache_paths(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        f0_caches = {
            pairs_source: _arg_value(runner.call_for(pairs_source), "--f0-cache")
            for pairs_source in ("v_hold", "test", "candidate")
        }
        grounding_caches = {
            pairs_source: _arg_value(runner.call_for(pairs_source), "--grounding-cache")
            for pairs_source in ("v_hold", "test", "candidate")
        }
        assert None not in f0_caches.values()
        assert None not in grounding_caches.values()
        assert len(set(f0_caches.values())) == 3
        assert len(set(grounding_caches.values())) == 3
        # Namespaced by pairs source, not merely random/unique, so the mapping
        # from cache file to universe is auditable from the filename alone.
        for pairs_source, path in f0_caches.items():
            assert path is not None
            assert pairs_source in path
        for pairs_source, path in grounding_caches.items():
            assert path is not None
            assert pairs_source in path


class TestOneLedgerEpochPerProtocolRun:
    """P1 fix #4: test and candidate must share one scoring-run id; v_hold gets none."""

    def test_scoring_run_id_shared_by_test_and_candidate_only_for_egostitch_e2e(
        self, tmp_path: Path
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
            model_family="egostitch_e2e",
        )

        v_hold_call = runner.call_for("v_hold")
        test_call = runner.call_for("test")
        candidate_call = runner.call_for("candidate")
        assert "--scoring-run-id" not in v_hold_call
        test_run_id = _arg_value(test_call, "--scoring-run-id")
        candidate_run_id = _arg_value(candidate_call, "--scoring-run-id")
        assert test_run_id is not None
        assert test_run_id == candidate_run_id

    def test_scoring_run_id_never_forwarded_for_a_non_egostitch_e2e_checkpoint(
        self, tmp_path: Path
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path, model_family="v3_1")
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        for call in runner.calls:
            assert "--scoring-run-id" not in call

    def test_scoring_run_id_reads_self_described_checkpoint_family_by_default(
        self, tmp_path: Path
    ) -> None:
        """No explicit `model_family` override.

        A self-describing egostitch_e2e checkpoint still gets a shared
        scoring-run id on its own.
        """
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path, model_family="egostitch_e2e")
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
        )

        test_run_id = _arg_value(runner.call_for("test"), "--scoring-run-id")
        candidate_run_id = _arg_value(runner.call_for("candidate"), "--scoring-run-id")
        assert test_run_id is not None
        assert test_run_id == candidate_run_id

    def test_scoring_run_id_is_stable_across_two_identical_calls(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)

        run_ids: list[str] = []
        for i in range(2):
            output_dir = tmp_path / "outputs" / f"run{i}"
            runner = _FakeScoreRunner(fixture.artifacts)
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
                model_family="egostitch_e2e",
            )
            run_id = _arg_value(runner.call_for("test"), "--scoring-run-id")
            assert run_id is not None
            run_ids.append(run_id)

        assert run_ids[0] == run_ids[1]

    def test_derive_scoring_run_id_is_pure_and_deterministic(self) -> None:
        first = test_protocol._derive_scoring_run_id(arm="full", seed=0, checkpoint_sha256="a" * 64)
        second = test_protocol._derive_scoring_run_id(
            arm="full", seed=0, checkpoint_sha256="a" * 64
        )
        different_seed = test_protocol._derive_scoring_run_id(
            arm="full", seed=1, checkpoint_sha256="a" * 64
        )
        assert first == second
        assert first != different_seed
        assert first.strip() == first
        assert first != ""


class TestOracleAndCaziForwarding:
    """Additional CLI surface: --allow-oracle-diagnostic and cazi_mbn model-family/config."""

    def test_allow_oracle_diagnostic_forwarded_to_every_pass(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="full",
            seed=0,
            score_runner=runner,
            allow_oracle_diagnostic=True,
        )

        for pairs_source in ("v_hold", "test", "candidate"):
            assert "--allow-oracle-diagnostic" in runner.call_for(pairs_source)

    def test_model_family_and_model_config_forwarded_to_every_pass(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "cazi_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)
        model_config = tmp_path / "cazi_mbn_breadth_first.yaml"
        model_config.write_text("model:\n  family: cazi_mbn\n", encoding="utf-8")

        run_test_protocol(
            checkpoint=checkpoint,
            output_dir=output_dir,
            data_root=fixture.data_root,
            strategy=_STRATEGY,
            arm="cazi_mbn",
            seed=0,
            score_runner=runner,
            model_family="cazi_mbn",
            model_config=model_config,
        )

        for pairs_source in ("v_hold", "test", "candidate"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--model-family") == "cazi_mbn"
            assert _arg_value(call, "--model-config") == str(model_config)
            # cazi_mbn is never egostitch_e2e: no scoring-run id either.
            assert "--scoring-run-id" not in call

    def test_model_config_without_model_family_raises_before_any_scoring(
        self, tmp_path: Path
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="model_family"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
                model_config=tmp_path / "some.yaml",
            )
        assert runner.calls == []

    def test_cazi_mbn_without_model_config_raises_before_any_scoring(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="model_config"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="cazi_mbn",
                seed=0,
                score_runner=runner,
                model_family="cazi_mbn",
            )
        assert runner.calls == []
