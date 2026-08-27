"""Tests for src.eval.test_protocol: the per-arm held-out test protocol.

Builds synthetic test and test_topology artifacts and a tiny benchmark fixture
(reusing the artifact/graph builders already committed for
`tests.test_g1_hardened_e2`), then drives `run_test_protocol` with a fake
`score_runner` so nothing here depends on `src.score_fanout.score_sharded`
(still a stub owned by a concurrent agent) or on any real GPU.

`_write_checkpoint` writes a real (if minimal) torch checkpoint -- not just
arbitrary bytes -- because `run_test_protocol` now reads a checkpoint's own
embedded ``model_family`` (the Task-4 self-describing contract) to decide
whether the test and test_topology passes may carry ``--scoring-run-id``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.special import expit
from src.eval import test_protocol
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


@dataclass(frozen=True)
class _ToyValidationSplit:
    buckets: dict[int, list[set[str]]]

    def build_g_val(self) -> nx.Graph:
        return _make_reference_graph()


@pytest.fixture(autouse=True)
def _sampled_validation_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep protocol tests focused on orchestration, not V_val disk derivation."""
    buckets = _small_buckets(_NODES, size=5, n_samples=4, seed=0)
    monkeypatch.setattr(
        test_protocol,
        "_load_val_region_split",
        lambda _data_root, _strategy: _ToyValidationSplit(buckets=buckets),
    )


def _build_test_topology_and_test_probs(pairs: list[tuple[str, str]]) -> NDArray[np.float64]:
    """High prob on the five true edges, moderate on self-pairs, low elsewhere.

    Chosen so every sample's exact-edge-count ranking follows the reference
    graph -- a hand-checkable happy path.
    """
    positive_set = {frozenset(edge) for edge in _POSITIVE_EDGES}
    probs = [
        0.5 if u == v else (0.9 if frozenset((u, v)) in positive_set else 0.1) for u, v in pairs
    ]
    return np.array(probs, dtype=np.float64)


def _build_fixture(tmp_path: Path) -> _Fixture:
    """Build a full test and test_topology + benchmark fixture sharing one node universe."""
    g_ref = _make_reference_graph()
    buckets = _small_buckets(_NODES, size=5, n_samples=4, seed=0)
    data_root = _write_benchmark(tmp_path, _STRATEGY, g_ref, buckets)

    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    probs = _build_test_topology_and_test_probs(pairs)
    logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)

    scores_dir = tmp_path / "prebuilt_scores"
    test_topology_path = scores_dir / "test_topology.npz"
    _write_universe_npz(
        test_topology_path,
        node_ids=_NODES,
        pairs=pairs,
        logits=logits,
        labels=labels,
        strategy=_STRATEGY,
        pairs_source="test_topology",
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
    val_topology_path = scores_dir / "val_topology.npz"
    _write_universe_npz(
        val_topology_path,
        node_ids=_NODES,
        pairs=pairs,
        # Deliberately shift validation above test. The selected fixed threshold
        # therefore predicts an empty test graph; if test were recalibrated,
        # the happy-path reference ranking would instead recover GS=1.
        logits=logits + 1.0,
        labels=labels,
        strategy=_STRATEGY,
        pairs_source="val_topology",
        checkpoint_id=_CHECKPOINT_ID,
    )

    return _Fixture(
        data_root=data_root,
        artifacts={
            "val_topology": val_topology_path,
            "test": test_path,
            "test_topology": test_topology_path,
        },
    )


def _write_checkpoint(tmp_path: Path, *, model_family: str = "v3_1") -> Path:
    """Write a minimal but real Task-4-format checkpoint (self-describing family).

    `run_test_protocol` reads this checkpoint's own embedded `model_family` (a
    plain torch.load, no model construction) to decide whether the
    test and test_topology passes may carry `--scoring-run-id`; a bare byte blob is no
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

        # 1. Validation freezes the deployable topology threshold before either
        # held-out pass is allowed to run.
        assert runner.pairs_order == ["val_topology", "test", "test_topology"]
        assert not (output_dir / "operating_point.json").exists()

        # 3. --data-root/--strategy/--checkpoint and --run-metadata forwarded to every
        # pass, pointed at this module's own scoring-identity file (never the
        # published run_metadata.json -- see test_never_overwrites_published_run_metadata).
        for pairs_source in ("val_topology", "test", "test_topology"):
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
            "edge",
            "calibration",
            "graph",
            "provenance",
        ]
        assert report["schema_version"] == "test_protocol_v6"

        arm_block = report["arm"]
        assert isinstance(arm_block, dict)
        assert arm_block["arm"] == "full"
        assert arm_block["seed"] == 0
        assert arm_block["model_family"] == "v3_1"
        assert arm_block["checkpoint_id"] == _CHECKPOINT_ID
        assert arm_block["checkpoint_sha256"] == test_protocol._sha256_file(checkpoint)

        # 5. Edge block: calibrated to the selected threshold, headline first.
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
        assert edge["metrics"]["threshold"] == pytest.approx(0.5)

        # 6. The single operating point: the validation-selected threshold,
        # shifted to probability 0.5 on test by logit calibration.
        graph = report["graph"]
        assert isinstance(graph, dict)
        assert set(graph.keys()) == {"fixed_threshold"}
        fixed = graph["fixed_threshold"]
        assert fixed["validation_selection"]["rule"] == "sampled_subgraph_density_shape_1se_v3"
        assert fixed["test"]["matching"] == "fixed_threshold_selected_on_validation"
        selected_threshold = fixed["validation_selection"]["selected"]["logit_threshold"]
        assert fixed["test"]["logit_threshold"] == pytest.approx(selected_threshold)
        assert fixed["test"]["graph_similarity"]["bfs_macro"] == pytest.approx(0.0)
        calibration = report["calibration"]
        assert isinstance(calibration, dict)
        assert calibration == {
            "method": "logit_shift_to_validation_selected_threshold",
            "logit_shift": -selected_threshold,
            "selected_logit_threshold": selected_threshold,
            "selected_probability_threshold": pytest.approx(float(expit(selected_threshold))),
        }
        assert edge["logit_shift"] == pytest.approx(-selected_threshold)

        # 7. Provenance: two artifact paths, sha256, and ledger digests
        # (null here -- v3_1 never claims the held-out E2E universe).
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        for pass_name, path in fixture.artifacts.items():
            entry = provenance[pass_name]
            assert entry["path"] == str(path)
            assert entry["sha256"] == test_protocol._sha256_file(path)
            assert entry["test_access_ledger_record_sha256"] is None

    def test_failing_test_topology_pass_leaves_no_report_written(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        runner = _FakeScoreRunner(fixture.artifacts, fail_on="test_topology")

        with pytest.raises(RuntimeError, match="test_topology"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )

        assert runner.pairs_order == ["val_topology", "test", "test_topology"]
        assert not (output_dir / "operating_point.json").exists()
        assert not (output_dir / "test_report.json").exists()

    def test_failing_validation_pass_never_opens_test(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        output_dir = tmp_path / "outputs" / "validation_failure"
        runner = _FakeScoreRunner(fixture.artifacts, fail_on="val_topology")

        with pytest.raises(RuntimeError, match="val_topology"):
            run_test_protocol(
                checkpoint=_write_checkpoint(tmp_path),
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )

        assert runner.pairs_order == ["val_topology"]
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

        assert runner.pairs_order == ["val_topology", "test"]
        assert not (output_dir / "operating_point.json").exists()
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

        for pairs_source in ("val_topology", "test", "test_topology"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--pack-dir") == str(pack_dir)
            assert _arg_value(call, "--scaffold-control") == "shuffle_within_pair_v3"

        for pairs_source in ("test", "test_topology"):
            call = runner.call_for(pairs_source)
            assert _arg_value(call, "--rescore-reason") == "repeat scoring for gate re-run"
        assert "--rescore-reason" not in runner.call_for("val_topology")

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

        # run_test_protocol validates validation plus both held-out artifacts;
        # report_edge_metrics validates test a second time
        # through its own bound import, which this spy (patched only on
        # test_protocol's namespace) does not observe.
        assert calls == [
            str(fixture.artifacts["val_topology"]),
            str(fixture.artifacts["test"]),
            str(fixture.artifacts["test_topology"]),
        ]

    def test_rejects_mismatched_checkpoint_across_passes(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"

        # Re-write the test_topology artifact under a different checkpoint_id.
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        probs = _build_test_topology_and_test_probs(pairs)
        logits = np.array([_logit_for_prob(p) for p in probs], dtype=np.float32)
        mismatched_test_topology = tmp_path / "prebuilt_scores" / "test_topology_mismatched.npz"
        _write_universe_npz(
            mismatched_test_topology,
            node_ids=_NODES,
            pairs=pairs,
            logits=logits,
            labels=labels,
            strategy=_STRATEGY,
            pairs_source="test_topology",
            checkpoint_id="0000000000000000",
        )
        artifacts = dict(fixture.artifacts)
        artifacts["test_topology"] = mismatched_test_topology
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

    def test_rejects_mutually_consistent_scores_from_another_checkpoint(
        self, tmp_path: Path
    ) -> None:
        fixture = _build_fixture(tmp_path)
        checkpoint = tmp_path / "real_checkpoint.pt"
        torch.save(
            {"model_family": "v3_1", "model_config": {}, "model_state": {"w": torch.ones(1)}},
            checkpoint,
        )
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="requested checkpoint"):
            run_test_protocol(
                checkpoint=checkpoint,
                output_dir=tmp_path / "outputs" / "stale",
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
                reuse_existing_scores=True,
            )

    def test_rejects_legacy_max_f1_artifacts_before_scoring(self, tmp_path: Path) -> None:
        fixture = _build_fixture(tmp_path)
        output_dir = tmp_path / "outputs" / "legacy"
        output_dir.mkdir(parents=True)
        (output_dir / "operating_point.json").write_text("{}\n", encoding="utf-8")
        runner = _FakeScoreRunner(fixture.artifacts)

        with pytest.raises(ValueError, match="obsolete max-F1"):
            run_test_protocol(
                checkpoint=_write_checkpoint(tmp_path),
                output_dir=output_dir,
                data_root=fixture.data_root,
                strategy=_STRATEGY,
                arm="full",
                seed=0,
                score_runner=runner,
            )
        assert runner.calls == []


class TestReuseExistingScores:
    """Resuming a run whose later pass failed must not redo the finished ones."""

    def test_reuses_written_artifacts_and_only_scores_what_is_missing(self, tmp_path: Path) -> None:
        """Mimics the real failure: test finished, test_topology did not."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "resume"
        scores_dir = output_dir / "scores"
        scores_dir.mkdir(parents=True)
        (scores_dir / "test.npz").write_bytes(fixture.artifacts["test"].read_bytes())

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

        assert runner.pairs_order == ["val_topology", "test_topology"]

    def test_reuse_is_opt_in(self, tmp_path: Path) -> None:
        """Implicit reuse would silently serve stale scores after a code fix."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "no_reuse"
        scores_dir = output_dir / "scores"
        scores_dir.mkdir(parents=True)
        (scores_dir / "test.npz").write_bytes(fixture.artifacts["test"].read_bytes())

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

        assert runner.pairs_order == ["val_topology", "test", "test_topology"]


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
        for pairs_source in ("test", "test_topology"):
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

    def test_selected_epoch_comes_from_the_published_metadata(self, tmp_path: Path) -> None:
        """Closes the checkpoint-selection audit gap: which epoch published."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {"arm": "full", "seed": 0, "checkpoint_id": _CHECKPOINT_ID, "selected_epoch": 17}
            )
            + "\n",
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
        assert arm_block["selected_epoch"] == 17

    def test_selected_epoch_is_none_without_published_metadata(self, tmp_path: Path) -> None:
        """A standalone scoring run (no published training dir) has no selected epoch."""
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
        assert arm_block["selected_epoch"] is None

    def test_selected_epoch_is_none_when_key_missing_from_published_metadata(
        self, tmp_path: Path
    ) -> None:
        """Older published metadata predating this field must not raise or fabricate."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "arm": "full",
                    "seed": 0,
                    "checkpoint_id": _CHECKPOINT_ID,
                    "run_kind": "diagnostic",
                }
            )
            + "\n",
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
        assert arm_block["selected_epoch"] is None

    def test_selected_epoch_is_none_for_a_different_checkpoint(self, tmp_path: Path) -> None:
        """A checkpoint the metadata does not describe must not inherit its epoch."""
        fixture = _build_fixture(tmp_path)
        checkpoint = _write_checkpoint(tmp_path)
        output_dir = tmp_path / "outputs" / "full_seed0"
        output_dir.mkdir(parents=True)
        (output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "arm": "full",
                    "seed": 0,
                    "checkpoint_id": "0000000000000000",
                    "selected_epoch": 17,
                }
            )
            + "\n",
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
        assert arm_block["selected_epoch"] is None

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
        assert runner.pairs_order == ["val_topology", "test", "test_topology"]


class TestUniverseScopedCaches:
    """Both passes share the exact complete test-node support and its caches."""

    def test_two_passes_share_test_support_cache_paths(self, tmp_path: Path) -> None:
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
            for pairs_source in ("test", "test_topology")
        }
        grounding_caches = {
            pairs_source: _arg_value(runner.call_for(pairs_source), "--grounding-cache")
            for pairs_source in ("test", "test_topology")
        }
        assert None not in f0_caches.values()
        assert None not in grounding_caches.values()
        assert len(set(f0_caches.values())) == 1
        assert len(set(grounding_caches.values())) == 1
        f0_path = next(iter(f0_caches.values()))
        grounding_path = next(iter(grounding_caches.values()))
        assert f0_path is not None and f0_path.endswith("f0_cache_test_support.pt")
        assert grounding_path is not None and grounding_path.endswith(
            "grounding_cache_test_support.npz"
        )
        validation_call = runner.call_for("val_topology")
        validation_f0 = _arg_value(validation_call, "--f0-cache")
        validation_grounding = _arg_value(validation_call, "--grounding-cache")
        assert validation_f0 is not None
        assert validation_f0.endswith("f0_cache_vval_support.pt")
        assert validation_grounding is not None
        assert validation_grounding.endswith("grounding_cache_vval_support.npz")
        assert validation_f0 != f0_path
        assert validation_grounding != grounding_path


class TestOneLedgerEpochPerProtocolRun:
    """P1 fix #4: test and test_topology must share one scoring-run id."""

    def test_scoring_run_id_shared_by_test_and_test_topology_only_for_egostitch_e2e(
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

        test_call = runner.call_for("test")
        test_topology_call = runner.call_for("test_topology")
        test_run_id = _arg_value(test_call, "--scoring-run-id")
        test_topology_run_id = _arg_value(test_topology_call, "--scoring-run-id")
        assert test_run_id is not None
        assert test_run_id == test_topology_run_id

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
        test_topology_run_id = _arg_value(runner.call_for("test_topology"), "--scoring-run-id")
        assert test_run_id is not None
        assert test_run_id == test_topology_run_id

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

        for pairs_source in ("test", "test_topology"):
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

        for pairs_source in ("test", "test_topology"):
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
