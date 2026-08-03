"""E2E fail-closed test-access ledger and four-array publication tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from src import score_universe

from tests.test_score_universe import _egostitch_e2e_setup

_DATA_PROVENANCE = {
    "data_contract": "shared_train_positives_v1",
    "training_interactions_sha256": "a" * 64,
    "training_topology_sha256": "b" * 64,
}


def _fake_loaded_e2e_model() -> torch.nn.Module:
    model = torch.nn.Linear(1, 1)
    model._training_data_provenance = dict(_DATA_PROVENANCE)  # type: ignore[attr-defined]
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_access_context(
    tmp_path: Path,
    *,
    shard: int = 0,
    num_shards: int = 1,
    reason: str | None = None,
) -> score_universe._TestAccessContext:
    return score_universe._TestAccessContext(
        ledger_path=tmp_path / "scores" / "test_access_ledger.jsonl",
        scoring_arm="full",
        seed=0,
        output=tmp_path / "scores" / "full.npz",
        shard=shard,
        num_shards=num_shards,
        rescore_reason=reason,
    )


def _write_test_pairs(tmp_path: Path) -> Path:
    path = (
        tmp_path / "data" / "benchmark_2025_neurips" / "breadth_first" / "test_edges.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("a\tb\t1\n", encoding="utf-8")
    return path


def _ledger_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_bound_score_artifact(
    tmp_path: Path,
    *,
    shard: int = 0,
    num_shards: int = 1,
) -> tuple[Path, Path]:
    output = tmp_path / "bound.npz"
    path = output if num_shards == 1 else score_universe._shard_output_path(output, shard)
    ledger_path = tmp_path / "test_access_ledger.jsonl"
    access = score_universe._TestAccessContext(
        ledger_path=ledger_path,
        scoring_arm="full",
        seed=0,
        output=output,
        shard=shard,
        num_shards=num_shards,
        rescore_reason=None,
    )
    score_universe._record_test_access(access, pairs_source="test")
    assert access.ledger_binding is not None
    values = np.array([0.1 + shard], dtype=np.float32)
    score_universe.save_scores(
        path,
        node_ids=["a", "b"],
        u_idx=np.array([0], dtype=np.int32),
        v_idx=np.array([1], dtype=np.int32),
        logit=values,
        label=np.array([1], dtype=np.int8),
        row_start=shard,
        meta={
            "checkpoint_id": "checkpoint",
            "model_family": "egostitch_e2e",
            "pairs_source": "test",
            "strategy": "toy",
            "num_rows": num_shards,
            "created_utc": "2026-07-30T00:00:00Z",
            "torch_version": "test",
            **_DATA_PROVENANCE,
            "permanent_null": "none",
            "primary_logit": "full",
            "score_precision": {
                "contract": "egostitch_e2e_pair_fp32_v1",
                "pair_compute_dtype": "float32",
                "pair_autocast": False,
                "logit_storage_dtype": "float32",
            },
            "test_access_ledger": access.ledger_binding,
        },
        f_logit=values,
    )
    return path, ledger_path


def _rewrite_artifact_meta(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    meta = json.loads(str(arrays["meta"][()]))
    mutate(meta)
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **arrays)


def test_test_access_ledger_groups_multi_gpu_shards_into_one_epoch(tmp_path: Path) -> None:
    _write_test_pairs(tmp_path)
    def resolve_shard(shard: int) -> None:
        score_universe._resolve_pairs(
            "test",
            tmp_path / "data",
            "breadth_first",
            test_access=_test_access_context(tmp_path, shard=shard, num_shards=4),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(resolve_shard, (2, 0, 3, 1)))

    records = _ledger_records(tmp_path / "scores" / "test_access_ledger.jsonl")
    assert len(records) == 4
    assert {record["scoring_epoch"] for record in records} == {1}
    assert {record["shard"] for record in records} == {0, 1, 2, 3}


def test_ledger_bound_shards_load_and_merge_as_one_scoring_epoch(tmp_path: Path) -> None:
    shard_paths = [
        _write_bound_score_artifact(tmp_path, shard=shard, num_shards=2)[0]
        for shard in range(2)
    ]

    for path in shard_paths:
        score_universe.load_scores(path)
    merged = score_universe.merge_scores(shard_paths)
    score_universe.validate_test_access_ledger_binding(merged.meta, label="merged")
    binding = merged.meta["test_access_ledger"]
    assert isinstance(binding, dict)
    assert binding["scoring_epoch"] == 1
    assert {record["shard"] for record in binding["records"]} == {0, 1}
    merged_path = tmp_path / "bound.npz"
    score_universe.save_scores(
        merged_path,
        node_ids=merged.node_ids,
        u_idx=merged.u_idx,
        v_idx=merged.v_idx,
        logit=merged.logit,
        label=merged.label,
        row_start=0,
        meta=merged.meta,
        f_logit=merged.f_logit,
        full_logit=merged.full_logit,
    )
    loaded = score_universe.load_scores(merged_path)
    score_universe.validate_artifact_precision(loaded, label="merged")


def test_merge_rejects_shards_with_different_training_data_provenance(tmp_path: Path) -> None:
    shard_paths = [
        _write_bound_score_artifact(tmp_path, shard=shard, num_shards=2)[0]
        for shard in range(2)
    ]
    _rewrite_artifact_meta(
        shard_paths[1],
        lambda meta: meta.__setitem__("training_interactions_sha256", "c" * 64),
    )

    with pytest.raises(ValueError, match="training_interactions_sha256"):
        score_universe.merge_scores(shard_paths)


def test_ledger_bound_artifact_rejects_deleted_ledger(tmp_path: Path) -> None:
    path, ledger = _write_bound_score_artifact(tmp_path)
    ledger.unlink()

    with pytest.raises(ValueError, match="test-access ledger is missing"):
        score_universe.load_scores(path)


def test_ledger_bound_artifact_rejects_ledger_digest_tamper(tmp_path: Path) -> None:
    path, ledger = _write_bound_score_artifact(tmp_path)
    record = _ledger_records(ledger)[0]
    record["pairs_source"] = "candidate"
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        score_universe.load_scores(path)


def test_heldout_shaped_artifact_missing_heldout_marker_fails_closed(tmp_path: Path) -> None:
    """The `heldout` discriminator itself must fail closed, not just the ledger file.

    A genuine held-out artifact is produced with a valid ledger binding and
    `heldout: True` (stamped automatically by `save_scores`). Stripping just the
    `heldout` marker -- leaving the valid `test_access_ledger` binding intact --
    must still raise: an e2e artifact over a held-out-shaped `pairs_source`
    with no explicit marker at all can never be trusted to have legitimately
    skipped ledger enforcement (design doc §10 item 1).
    """
    path, _ledger = _write_bound_score_artifact(tmp_path)

    def strip_heldout(meta: dict[str, object]) -> None:
        del meta["heldout"]

    _rewrite_artifact_meta(path, strip_heldout)
    with pytest.raises(ValueError, match="missing the heldout marker"):
        score_universe.load_scores(path)

    # Direct-call form: validate_test_access_ledger_binding is the SURVIVING
    # entry point other agents' code paths call directly, not only through
    # load_scores.
    with np.load(path, allow_pickle=False) as artifact:
        meta = json.loads(str(artifact["meta"][()]))
    assert "heldout" not in meta
    with pytest.raises(ValueError, match="missing the heldout marker"):
        score_universe.validate_test_access_ledger_binding(meta, label="direct")


def test_heldout_shaped_artifact_false_heldout_marker_fails_closed(tmp_path: Path) -> None:
    """A `heldout: false` marker on a held-out-shaped artifact must raise, not skip.

    `save_scores` always derives `heldout` from `_is_heldout_universe` for a
    held-out-shaped artifact (family `egostitch_e2e` scoring
    `candidate`/`test`/`file:*`), so there is no legitimate way such an
    artifact carries `heldout: False`. Flipping the flag must not bypass
    ledger validation -- the ledger could otherwise be deleted or tampered
    with while `load_scores` still succeeds (design doc §10 item 1).
    """
    path, _ledger = _write_bound_score_artifact(tmp_path)

    def flip_heldout(meta: dict[str, object]) -> None:
        meta["heldout"] = False

    _rewrite_artifact_meta(path, flip_heldout)
    with pytest.raises(ValueError, match="expected True"):
        score_universe.load_scores(path)

    with np.load(path, allow_pickle=False) as artifact:
        meta = json.loads(str(artifact["meta"][()]))
    assert meta["heldout"] is False
    with pytest.raises(ValueError, match="expected True"):
        score_universe.validate_test_access_ledger_binding(meta, label="direct")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scoring_arm", "p0", "scoring_arm mismatch"),
        ("seed", 7, "seed mismatch"),
        ("scoring_epoch", 2, "scoring_epoch mismatch"),
    ],
)
def test_ledger_bound_artifact_rejects_identity_mismatch(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    path, _ledger = _write_bound_score_artifact(tmp_path)

    def mutate(meta: dict[str, object]) -> None:
        binding = meta["test_access_ledger"]
        assert isinstance(binding, dict)
        binding[field] = value

    _rewrite_artifact_meta(path, mutate)
    with pytest.raises(ValueError, match=error):
        score_universe.load_scores(path)


def test_scoring_full_then_both_scaffold_controls_needs_no_rescore_reason(
    tmp_path: Path,
) -> None:
    """The two mandatory scaffold-structure controls must ledger under distinct arms.

    Both controls reuse the `full` checkpoint and its run metadata. Before
    the fix, scoring either one recorded it into the test-access ledger
    under arm `full`, so scoring a control after the ordinary full score was
    rejected as repeat full-arm scoring -- blocking normal first-time
    scoring of the mandatory controls unless the operator supplied a
    misleading `--rescore-reason` (design doc §10 item 1 / finding 2).
    """
    data_root, checkpoint, pairs_path = _egostitch_e2e_setup(tmp_path)
    test_pairs_path = data_root / "benchmark_2025_neurips" / "breadth_first" / "test_edges.txt"
    test_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    test_pairs_path.write_bytes(pairs_path.read_bytes())

    run_metadata_path = tmp_path / "run_metadata.json"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run_metadata_path.write_text(
        json.dumps(
            {
                "arm": "full",
                "seed": 0,
                "checkpoint_id": score_universe._checkpoint_id(
                    checkpoint_payload["model_state"]
                ),
                **_DATA_PROVENANCE,
            }
        ),
        encoding="utf-8",
    )

    def score(output_name: str, scaffold_control: str | None) -> None:
        args = [
            "score",
            "--checkpoint",
            str(checkpoint),
            "--pairs",
            "test",
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / output_name),
            "--token-budget",
            "8192",
            "--f0-cache",
            str(tmp_path / "f0_cache.pt"),
            "--device",
            "cpu",
            "--run-metadata",
            str(run_metadata_path),
        ]
        if scaffold_control is not None:
            args += ["--scaffold-control", scaffold_control]
        score_universe.main(args)

    # No --rescore-reason anywhere: each of these must be accepted as a
    # first-time score under its own arm.
    score("full.npz", None)
    score("control_6a.npz", "shuffle_within_pair_v3")
    score("control_6e.npz", "rewire_checkerboard_v1")

    records = _ledger_records(tmp_path / "test_access_ledger.jsonl")
    arms = {record["scoring_arm"] for record in records}
    assert arms == {"full", "structure_control_6a_v3", "structure_control_6e_v1"}
    assert {record["scoring_epoch"] for record in records} == {1}

    full = score_universe.load_scores(tmp_path / "full.npz")
    control_6a = score_universe.load_scores(tmp_path / "control_6a.npz")
    control_6e = score_universe.load_scores(tmp_path / "control_6e.npz")
    assert full.meta["test_access_ledger"]["scoring_arm"] == "full"
    assert control_6a.meta["test_access_ledger"]["scoring_arm"] == "structure_control_6a_v3"
    assert control_6e.meta["test_access_ledger"]["scoring_arm"] == "structure_control_6e_v1"


def test_test_access_ledger_rejects_repeat_without_reason_and_records_reasoned_repeat(
    tmp_path: Path,
) -> None:
    _write_test_pairs(tmp_path)
    first = _test_access_context(tmp_path)
    score_universe._resolve_pairs(
        "test", tmp_path / "data", "breadth_first", test_access=first
    )

    with pytest.raises(ValueError, match="repeat scoring requires --rescore-reason"):
        score_universe._resolve_pairs(
            "test", tmp_path / "data", "breadth_first", test_access=first
        )

    score_universe._resolve_pairs(
        "test",
        tmp_path / "data",
        "breadth_first",
        test_access=_test_access_context(tmp_path, reason="replace corrupt score artifact"),
    )
    records = _ledger_records(tmp_path / "scores" / "test_access_ledger.jsonl")
    assert [record["scoring_epoch"] for record in records] == [1, 2]
    assert records[1]["rescore_reason"] == "replace corrupt score artifact"


def test_test_access_ledger_refuses_malformed_history_before_pair_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_test_pairs(tmp_path)
    context = _test_access_context(tmp_path)
    context.ledger_path.parent.mkdir(parents=True)
    context.ledger_path.write_text("not-json\n", encoding="utf-8")

    def forbidden_pair_read(_path: Path) -> tuple[object, object]:
        raise AssertionError("pairs were read before the ledger was validated")

    monkeypatch.setattr(score_universe, "_read_pairs_tsv", forbidden_pair_read)
    with pytest.raises(ValueError, match="test-access ledger is malformed at line 1"):
        score_universe._resolve_pairs(
            "test", tmp_path / "data", "breadth_first", test_access=context
        )


def test_arbitrary_file_is_ledgered_as_heldout_before_pair_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    checkpoint_id = "0123456789abcdef"
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "arm": "full",
                "seed": 0,
                "checkpoint_id": checkpoint_id,
                **_DATA_PROVENANCE,
            }
        ),
        encoding="utf-8",
    )
    copied = tmp_path / "copied-test.tsv"
    copied.write_text("a\tb\t1\n", encoding="utf-8")
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (
            _fake_loaded_e2e_model(),
            "egostitch_e2e",
            checkpoint_id,
        ),
    )

    def inspect_after_ledger(_path: Path) -> tuple[object, object]:
        ledger = run_metadata_path.parent / "test_access_ledger.jsonl"
        records = _ledger_records(ledger)
        assert records[-1]["event"] == "resolve_pairs"
        assert records[-1]["pairs_source"] == f"file:{copied}"
        assert records[-1]["scoring_arm"] == "full"
        assert records[-1]["seed"] == 0
        raise AssertionError("pair read reached after ledger record")

    monkeypatch.setattr(score_universe, "_read_pairs_tsv", inspect_after_ledger)
    cli = [
        "score",
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        f"file:{copied}",
        "--output",
        str(tmp_path / "scores" / "full.npz"),
        "--run-metadata",
        str(run_metadata_path),
    ]

    with pytest.raises(AssertionError, match="pair read reached after ledger record"):
        score_universe.main(cli)


def test_file_alias_of_candidate_manifest_still_requires_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    checkpoint_id = "0123456789abcdef"
    candidate = (
        tmp_path
        / "data"
        / "benchmark_2025_neurips"
        / "breadth_first"
        / "candidate_test_edges.txt"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_text("a\tb\n", encoding="utf-8")
    alias = tmp_path / "copied-candidate.tsv"
    alias.write_bytes(candidate.read_bytes() + b"\n")
    monkeypatch.setattr(
        score_universe,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (_fake_loaded_e2e_model(), "egostitch_e2e", checkpoint_id),
    )
    output = tmp_path / "forbidden.npz"

    with pytest.raises(ValueError, match="requires --run-metadata"):
        score_universe.main(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--pairs",
                f"file:{alias}",
                "--data-root",
                str(tmp_path / "data"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_e2e_artifact_physically_stores_exactly_two_decomposition_arrays(tmp_path: Path) -> None:
    """New (v4) artifacts publish only full/f_logit -- the content path is gone (design doc §9)."""
    output = tmp_path / "scores.npz"
    values = np.array([0.1, 0.2], dtype=np.float32)
    access = score_universe._TestAccessContext(
        ledger_path=tmp_path / "test_access_ledger.jsonl",
        scoring_arm="full",
        seed=0,
        output=output,
        shard=0,
        num_shards=1,
        rescore_reason=None,
    )
    score_universe._record_test_access(access, pairs_source="candidate")
    assert access.ledger_binding is not None
    meta: dict[str, object] = {
        "checkpoint_id": "checkpoint",
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": "toy",
        "num_rows": 2,
        "created_utc": "2026-07-19T00:00:00Z",
        "torch_version": "test",
        **_DATA_PROVENANCE,
        "permanent_null": "none",
        "primary_logit": "full",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "test_access_ledger": access.ledger_binding,
    }
    score_universe.save_scores(
        output,
        node_ids=["a", "b"],
        u_idx=np.array([0, 0], dtype=np.int32),
        v_idx=np.array([1, 1], dtype=np.int32),
        logit=values,
        label=np.array([-1, -1], dtype=np.int8),
        row_start=0,
        meta=meta,
        f_logit=values + 1,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert {"full", "f_logit"} <= set(artifact.files)
        assert "pair_content" not in artifact.files
        assert "pair_topology" not in artifact.files
        np.testing.assert_array_equal(artifact["full"], artifact["logit"])

    loaded = score_universe.load_scores(output)
    assert loaded.meta["scores_meta_version"] == score_universe._SCORES_META_VERSION
    assert loaded.pair_content is None
    assert loaded.pair_topology is None


def test_v3_legacy_e2e_artifact_with_four_decomposition_arrays_still_loads(
    tmp_path: Path,
) -> None:
    """A pre-content-path-removal v3 artifact (four arrays) still loads.

    `save_scores` only ever writes the two-array v4 schema now (content path
    removed, design doc §9), so this constructs a v3 artifact directly rather
    than through `save_scores` -- proving `load_scores` keeps reading
    artifacts scored before the removal, with `pair_content`/`pair_topology`
    populated from the file rather than always `None`.
    """
    output = tmp_path / "legacy-v3.npz"
    values = np.array([0.1, 0.2], dtype=np.float32)
    resolution = score_universe.score_resolution_diagnostics(values)
    meta: dict[str, object] = {
        "checkpoint_id": "checkpoint",
        "model_family": "egostitch_e2e",
        # "val" (not "candidate"/"test"/"file:*") deliberately: this test is
        # only about legacy four-array loading, and a held-out pairs_source
        # would require a genuine test-access ledger binding it has no
        # reason to fabricate.
        "pairs_source": "val",
        "strategy": "toy",
        "num_rows": 2,
        "created_utc": "2026-07-19T00:00:00Z",
        "torch_version": "test",
        "permanent_null": "none",
        "primary_logit": "full",
        "scores_meta_version": "egostitch_e2e_scores_v3",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "score_resolution": dict.fromkeys(
            ("full", "f_logit", "pair_content", "pair_topology"), resolution
        ),
    }
    np.savez_compressed(
        output,
        node_ids=np.array(["a", "b"]),
        u_idx=np.array([0, 0], dtype=np.int32),
        v_idx=np.array([1, 1], dtype=np.int32),
        logit=values,
        label=np.array([-1, -1], dtype=np.int8),
        row_start=np.int64(0),
        meta=np.array(json.dumps(meta, sort_keys=True)),
        full=values,
        f_logit=values + 1,
        pair_content=values + 2,
        pair_topology=values + 3,
    )

    artifact = score_universe.load_scores(output)
    assert artifact.meta["scores_meta_version"] == "egostitch_e2e_scores_v3"
    assert artifact.f_logit is not None
    assert artifact.pair_content is not None
    assert artifact.pair_topology is not None
    np.testing.assert_array_equal(artifact.f_logit, values + 1)
    np.testing.assert_array_equal(artifact.pair_content, values + 2)
    np.testing.assert_array_equal(artifact.pair_topology, values + 3)


def test_loader_rejects_v1_e2e_artifact_without_meta_version(tmp_path: Path) -> None:
    output = tmp_path / "legacy-v1.npz"
    values = np.array([0.1], dtype=np.float32)
    resolution = score_universe.score_resolution_diagnostics(values)
    meta = {
        "checkpoint_id": "checkpoint",
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": "toy",
        "num_rows": 1,
        "created_utc": "2026-07-19T00:00:00Z",
        "torch_version": "test",
        "permanent_null": "none",
        "primary_logit": "full",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "score_resolution": dict.fromkeys(
            ("full", "f_logit", "pair_content", "pair_topology"), resolution
        ),
    }
    np.savez_compressed(
        output,
        node_ids=np.array(["a", "b"]),
        u_idx=np.array([0], dtype=np.int32),
        v_idx=np.array([1], dtype=np.int32),
        logit=values,
        label=np.array([-1], dtype=np.int8),
        row_start=np.int64(0),
        meta=np.array(json.dumps(meta)),
        f_logit=values,
        pair_content=values,
        pair_topology=values,
    )

    with pytest.raises(ValueError, match="scores_meta_version"):
        score_universe.load_scores(output)


def test_formal_v2_loader_rejects_missing_or_contradictory_full_array(tmp_path: Path) -> None:
    output = tmp_path / "malformed-e2e.npz"
    values = np.array([0.1], dtype=np.float32)
    resolution = score_universe.score_resolution_diagnostics(values)
    meta = {
        "model_family": "egostitch_e2e",
        "scores_meta_version": score_universe._SCORES_META_VERSION,
        "primary_logit": "full",
        "formal_scoring_provenance": {"registration_sha256": "a" * 64},
        "score_resolution": dict.fromkeys(
            ("full", "f_logit", "pair_content", "pair_topology"), resolution
        ),
    }
    common = {
        "node_ids": np.array(["a", "b"]),
        "u_idx": np.array([0], dtype=np.int32),
        "v_idx": np.array([1], dtype=np.int32),
        "logit": values,
        "label": np.array([-1], dtype=np.int8),
        "row_start": np.int64(0),
        "meta": np.array(json.dumps(meta)),
        "f_logit": values,
        "pair_content": values,
        "pair_topology": values,
    }
    np.savez_compressed(output, **common)
    with pytest.raises(ValueError, match="missing full"):
        score_universe.load_scores(output)

    np.savez_compressed(output, **common, full=values + 1)
    with pytest.raises(ValueError, match="contradicts primary logit"):
        score_universe.load_scores(output)


def test_legacy_content_head_artifact_is_rejected_with_a_clear_reason() -> None:
    """A v3 pair_topology artifact fails with an explanation, not a bare KeyError.

    The arm was retired with the content path (design 2026-08-02 §9). Other v3
    arms still load; this one is explicitly unsupported and must say so.
    """
    with pytest.raises(ValueError, match="retired pair_topology arm"):
        score_universe._e2e_primary_logit_key("content_head")


def test_unknown_permanent_null_is_a_value_error_not_a_key_error() -> None:
    with pytest.raises(ValueError, match="unknown egostitch_e2e permanent_null"):
        score_universe._e2e_primary_logit_key("nonsense")
