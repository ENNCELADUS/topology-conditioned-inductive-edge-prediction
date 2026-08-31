"""Per-arm held-out test protocol: score, then report edge and graph metrics together.

Owns the whole post-train sequence for one arm and nothing else::

    score val_topology -> select one fixed threshold on validation samples
    score test          -> edge metrics, logits calibrated so that threshold
                           sits at probability 0.5
    score test_topology -> the validation-fixed threshold replayed unchanged
        -> test_report.json

One operating point serves both metric families: the validation-selected logit
threshold, shifted to probability 0.5 by test-time logit calibration.
Both metric families are always reported together; a partial report is never
written.

This module composes existing analysis primitives; it never rescores a pair or
reimplements a metric. It never calls ``validate_score_precision`` directly on
a loaded artifact -- only :func:`src.score_universe.validate_artifact_precision`,
the correct entry point for an artifact whose family is not known in advance
(calling ``validate_score_precision`` directly on an ``egostitch_e2e`` artifact
spuriously raises "missing arrays", per ``CLAUDE.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from scipy.special import expit

from src.eval.fixed_threshold import evaluate_fixed_threshold, select_fixed_threshold
from src.eval.graph_metrics import MMDConfig
from src.eval.report_edge_metrics import report_edge_metrics
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    load_test_graph,
    load_test_node_buckets,
)
from src.score_universe import (
    MODEL_BUILDERS,
    ScoresArtifact,
    _checkpoint_id,
    _load_val_region_split,
    load_scores,
    validate_artifact_precision,
)

logger = logging.getLogger(__name__)

__all__ = ["ScoreRunner", "TestProtocolResult", "build_parser", "main", "run_test_protocol"]

_SCHEMA_VERSION = "test_protocol_v6"
#: The filename `src.e2_pipeline` (and `src.train_egostitch`/`src.train_b0`)
#: publish training provenance into: checkpoint identity, publication status,
#: and access-audit fields no scoring call may destroy. This module never
#: writes it -- see `_SCORING_IDENTITY_FILENAME`.
_RUN_METADATA_FILENAME = "run_metadata.json"
#: This module's own, private scoring-identity file: just enough
#: (``arm``/``seed``) for `src.score_universe`'s ``--run-metadata`` to key the
#: egostitch_e2e test-access ledger. Named so it can never collide with
#: `_RUN_METADATA_FILENAME` or any other filename `src.e2_pipeline` publishes
#: (`best.pt`, `last.pt`, `metrics.jsonl`, `profile.json`,
#: `artifact_manifest.json`).
_SCORING_IDENTITY_FILENAME = "test_protocol_run_metadata.json"


class ScoreRunner(Protocol):
    """Seam that scores one pairs universe and returns the merged artifact."""

    def __call__(self, score_args: Sequence[str]) -> Path:
        """Run ``score_universe score`` with `score_args` and return the output path."""


@dataclass(frozen=True)
class TestProtocolResult:
    """Outcome of one arm's test protocol."""

    report_path: Path
    report: dict[str, object]


# --------------------------------------------------------------------------- small helpers


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_pairs_source(artifact: ScoresArtifact, expected: str, *, label: str) -> None:
    """Raise unless `artifact` declares `expected` as its ``pairs_source``."""
    actual = artifact.meta.get("pairs_source")
    if actual != expected:
        raise ValueError(f"{label}: pairs_source expected {expected!r}, got {actual!r}")


def _require_full_universe(artifact: ScoresArtifact, *, label: str) -> None:
    """Raise if `artifact` looks like an unmerged shard rather than a full universe.

    Mirrors :func:`src.eval.report_edge_metrics.report_edge_metrics`'s own guard: a
    single distributed shard loads cleanly but carries only its own rows while its
    metadata still records the full universe's row count.
    """
    n_rows = int(artifact.label.size)
    meta_rows = artifact.meta.get("num_rows")
    if isinstance(meta_rows, int) and meta_rows != n_rows:
        raise ValueError(
            f"{label}: loaded {n_rows} rows but metadata declares {meta_rows}; "
            "this looks like a partial shard -- merge shards before reporting"
        )


def _require_same_checkpoint(*artifacts: ScoresArtifact) -> None:
    """Raise unless all protocol passes scored the same checkpoint and family."""
    checkpoint_ids = {artifact.meta.get("checkpoint_id") for artifact in artifacts}
    if len(checkpoint_ids) != 1:
        raise ValueError(f"protocol scoring passes used different checkpoints: {checkpoint_ids}")
    model_families = {artifact.meta.get("model_family") for artifact in artifacts}
    if len(model_families) != 1:
        raise ValueError(f"protocol scoring passes disagree on model_family: {model_families}")


def _expected_checkpoint_id(checkpoint: Path) -> str | None:
    """Return the score-artifact id for a real checkpoint, if it carries state."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    state = payload.get("model_state", payload.get("state_dict"))
    if (
        state is None
        and payload
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    ):
        state = payload
    if not isinstance(state, dict) or not state:
        return None
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        return None
    return _checkpoint_id(cast(dict[str, torch.Tensor], state))


def _require_scoring_identity(
    *,
    artifact: ScoresArtifact,
    checkpoint_id: str | None,
    strategy: str,
    topo_gen_control: str | None,
    label: str,
) -> None:
    """Bind a scored or reused artifact to this invocation's checkpoint and split."""
    if checkpoint_id is not None and artifact.meta.get("checkpoint_id") != checkpoint_id:
        raise ValueError(
            f"{label}: checkpoint_id {artifact.meta.get('checkpoint_id')!r} does not match "
            f"the requested checkpoint {checkpoint_id!r}"
        )
    if artifact.meta.get("strategy") != strategy:
        raise ValueError(
            f"{label}: strategy {artifact.meta.get('strategy')!r} does not match {strategy!r}"
        )
    if artifact.meta.get("topo_gen_control") != topo_gen_control:
        raise ValueError(
            f"{label}: topo_gen_control {artifact.meta.get('topo_gen_control')!r} "
            f"does not match {topo_gen_control!r}"
        )


def _ledger_record_sha256(meta: Mapping[str, object]) -> str | None:
    """Return the bound test-access ledger record digest, or ``None`` if absent."""
    binding = meta.get("test_access_ledger")
    if isinstance(binding, dict):
        digest = binding.get("record_sha256")
        if isinstance(digest, str):
            return digest
    return None


def _self_described_model_family(checkpoint: Path) -> str | None:
    """Return a Task-4 checkpoint's own embedded ``model_family``, or ``None``.

    Reads only the checkpoint's top-level ``model_family`` string -- the same
    self-describing contract ``src.score_universe._load_checkpoint`` keys off
    of -- without building the model or applying its weights. A bare legacy
    checkpoint (currently only ``cazi_mbn``'s released
    ``{"state_dict": ..., "best_val_auroc": ...}`` format) has no such key and
    returns ``None``; the caller must then supply ``model_family`` explicitly
    (this module's own ``model_family`` parameter, forwarded as
    ``--model-family``).
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        family = payload.get("model_family")
        if isinstance(family, str) and family:
            return family
    return None


def _derive_scoring_run_id(*, arm: str, seed: int, checkpoint_sha256: str) -> str:
    """Deterministically derive one held-out scoring-epoch identity.

    Stable across repeated calls with the same ``(arm, seed, checkpoint)`` so a
    resumed or re-run protocol invocation joins the SAME test-access-ledger
    epoch across its test and test_topology passes (the P1 defect this fixes: those
    two passes used to open two separate epochs, so a first-time run without
    ``--rescore-reason`` failed on the second pass). Never derived from
    wall-clock time or randomness -- either would break reproducibility and the
    ledger's resume path.
    """
    payload = json.dumps(
        {"arm": arm, "seed": seed, "checkpoint_sha256": checkpoint_sha256},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_published_run_metadata(
    output_dir: Path, *, arm: str, seed: int
) -> dict[str, object] | None:
    """Cross-check a pre-existing published ``run_metadata.json``; never write it.

    P1 fix: this module used to write ``{"arm": arm, "seed": seed}`` straight
    into ``output_dir / "run_metadata.json"``, unconditionally overwriting
    whatever a training publish (`src.e2_pipeline`/`src.train_egostitch`/
    `src.train_b0`) left there -- checkpoint identity, publication status,
    every other provenance field -- while leaving ``artifact_manifest.json``
    attesting bytes that no longer existed. This module now never writes that
    filename at all (see `_SCORING_IDENTITY_FILENAME`); when it is present, it
    is read back and checked for consistency instead of replaced.

    Args:
        output_dir: The scoring run's output directory (frequently a published
            training directory).
        arm: This call's own arm identity.
        seed: This call's own seed.

    Raises:
        ValueError: If the file exists but is not a readable JSON object, or
            its own ``arm``/``seed`` fields (when present -- not every
            family's published metadata carries them, e.g. `src.train_b0`'s
            does not) contradict this call's.
    """
    path = output_dir / _RUN_METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        published = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"published {path} is not readable JSON: {error}") from error
    if not isinstance(published, dict):
        raise ValueError(f"published {path} must contain a JSON object")
    published_arm = published.get("arm")
    if isinstance(published_arm, str) and published_arm and published_arm != arm:
        raise ValueError(
            f"published {path} has arm {published_arm!r}, which contradicts --arm {arm!r}"
        )
    published_seed = published.get("seed")
    if (
        isinstance(published_seed, int)
        and not isinstance(published_seed, bool)
        and published_seed != seed
    ):
        raise ValueError(
            f"published {path} has seed {published_seed!r}, which contradicts --seed {seed!r}"
        )
    return cast(dict[str, object], published)


def _write_scoring_identity(output_dir: Path, *, arm: str, seed: int) -> Path:
    """Write this call's own arm/seed identity for ``score_universe --run-metadata``.

    Deliberately not `_RUN_METADATA_FILENAME`: see that constant and
    `_validate_published_run_metadata`. This file is owned exclusively by this
    module, reflects only this call's own arguments, and is safe to
    unconditionally rewrite on every invocation.
    """
    path = output_dir / _SCORING_IDENTITY_FILENAME
    path.write_text(json.dumps({"arm": arm, "seed": seed}, indent=2) + "\n", encoding="utf-8")
    return path


def _build_score_args(
    *,
    checkpoint: Path,
    pairs: str,
    output: Path,
    data_root: Path,
    strategy: str,
    run_metadata: Path,
    f0_cache: Path,
    grounding_cache: Path,
    pack_dir: Path | None,
    scaffold_control: str | None,
    topo_gen_control: str | None,
    rescore_reason: str | None,
    scoring_run_id: str | None,
    allow_oracle_diagnostic: bool,
    model_family: str | None,
    model_config: Path | None,
) -> list[str]:
    """Build the ``score_universe score`` argument list for one pairs source.

    ``--run-metadata`` is always forwarded for the held-out passes.
    ``--f0-cache``/``--grounding-cache`` are likewise always forwarded with
    this call's own universe-scoped paths (see
    `run_test_protocol`): they are read only for ``f0_mlp``/``egostitch_e2e``
    scoring, so passing them for ``v3_1``/``cazi_mbn`` is equally inert.
    ``--rescore-reason``/``--scoring-run-id`` are forwarded only for held-out
    EgoStitch scoring.
    """
    args = [
        "--checkpoint",
        str(checkpoint),
        "--pairs",
        pairs,
        "--data-root",
        str(data_root),
        "--strategy",
        strategy,
        "--output",
        str(output),
        "--run-metadata",
        str(run_metadata),
        "--f0-cache",
        str(f0_cache),
        "--grounding-cache",
        str(grounding_cache),
    ]
    if pack_dir is not None:
        args += ["--pack-dir", str(pack_dir)]
    if scaffold_control is not None:
        args += ["--scaffold-control", scaffold_control]
    if topo_gen_control is not None:
        args += ["--topo-gen-control", topo_gen_control]
    if rescore_reason is not None:
        args += ["--rescore-reason", rescore_reason]
    if scoring_run_id is not None:
        args += ["--scoring-run-id", scoring_run_id]
    if allow_oracle_diagnostic:
        args += ["--allow-oracle-diagnostic"]
    if model_config is not None:
        assert model_family is not None  # enforced by run_test_protocol
        args += ["--model-family", model_family, "--model-config", str(model_config)]
    return args


# --------------------------------------------------------------------------- orchestration


def run_test_protocol(
    *,
    checkpoint: Path,
    output_dir: Path,
    data_root: Path,
    strategy: str,
    arm: str,
    seed: int,
    score_runner: ScoreRunner,
    pack_dir: Path | None = None,
    scaffold_control: str | None = None,
    topo_gen_control: str | None = None,
    rescore_reason: str | None = None,
    model_family: str | None = None,
    model_config: Path | None = None,
    allow_oracle_diagnostic: bool = False,
    report_filename: str = "test_report.json",
    reuse_existing_scores: bool = False,
) -> TestProtocolResult:
    """Run the full test protocol for one published checkpoint.

    Args:
        checkpoint: Published checkpoint to score. Never re-ranked or gated here.
        output_dir: Destination for ``scores/`` and the report. Frequently a
            published training directory; this function never writes
            `_RUN_METADATA_FILENAME` there (see `_validate_published_run_metadata`).
        data_root: Benchmark data root.
        strategy: Split strategy (for example ``breadth_first``).
        arm: Arm identity recorded in the report.
        seed: Run seed recorded in the report.
        score_runner: Seam that performs one scoring pass.
        pack_dir: Optional GPU-resident packed BF16 feature directory.
        scaffold_control: Optional scoring-time structure control.
        topo_gen_control: Optional topology-generator scoring-time control.
        rescore_reason: Required by the test-access ledger when this
            ``(arm, seed)`` has already opened held-out data.
        model_family: Explicit model family for a bare legacy checkpoint (only
            ``cazi_mbn`` today, whose released
            ``{"state_dict": ..., "best_val_auroc": ...}`` checkpoint does not
            self-describe). Leave ``None`` for a self-describing Task-4
            checkpoint (``v3_1``/``egostitch_e2e``): this function reads the
            checkpoint's own embedded ``model_family`` in that case, both to
            decide whether the test and test_topology passes may carry
            ``--scoring-run-id`` (egostitch_e2e held-out scoring only) and,
            when both `model_family` and `model_config` are given, to forward
            ``--model-family``/``--model-config`` on every pass.
        model_config: Training YAML supplying ``model.config`` for a bare
            legacy checkpoint (required together with `model_family`; for
            ``cazi_mbn`` specifically this YAML's ``expected_missing_features``
            and its ``output_dir``-relative ``feature_stats.npz`` are what
            scoring needs -- architecture is inferred from the checkpoint's
            own tensors).
        allow_oracle_diagnostic: Forwarded verbatim as ``--allow-oracle-diagnostic``
            to every pass. Required (and refused otherwise) exactly when the
            checkpoint's ``egostitch_e2e`` generator is ``oracle_struct``; such
            a checkpoint consumes ground-truth topology by construction and is
            never a formal held-out result, regardless of pairs source.
        report_filename: Report name; diagnostic runs pass
            ``diagnostic_test_report.json``.
        reuse_existing_scores: Reuse an already-written ``scores/<pairs>.npz``
            instead of rescoring that universe. Opt-in, for resuming a run
            whose later pass failed after earlier ones succeeded; reused
            artifacts are still checkpoint-cross-checked like scored ones.

    Returns:
        The written report path and its parsed payload.

    Raises:
        ValueError: On a precision-contract, ledger-binding, pairs-source, or
            model-family/model-config-pairing violation in any artifact this
            function loads or argument it is given.
    """
    if model_config is not None and model_family is None:
        raise ValueError(
            "model_config requires model_family (score_universe requires both together)"
        )
    if model_family == "cazi_mbn" and model_config is None:
        raise ValueError("cazi_mbn scoring requires model_config (its own training YAML)")

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    legacy_artifacts = [output_dir / "operating_point.json"]
    stale = [str(path) for path in legacy_artifacts if path.exists()]
    if stale:
        raise ValueError(f"remove obsolete max-F1 test artifacts before scoring: {stale}")

    # Fix for the P1 defect where this module clobbered a published
    # run_metadata.json: validate it in place (if present) and write this
    # call's own scoring identity under a filename that can never collide
    # with it. `--run-metadata` for every pass below points at OUR file, never
    # `_RUN_METADATA_FILENAME`.
    published_run_metadata = _validate_published_run_metadata(output_dir, arm=arm, seed=seed)
    scoring_identity_path = _write_scoring_identity(output_dir, arm=arm, seed=seed)

    checkpoint_sha256 = _sha256_file(checkpoint)
    expected_checkpoint_id = _expected_checkpoint_id(checkpoint)

    # egostitch_e2e held-out scoring (test and test_topology) is the only case
    # `--scoring-run-id` is valid for; every other family/pairs-source
    # combination has `src.score_universe` reject it outright. A Task-4
    # checkpoint self-describes its family, so this reads that rather than
    # requiring every caller to pass `model_family` just to score a v3_1 or
    # egostitch_e2e checkpoint (which never needs `--model-family` on the
    # wire at all -- see `_build_score_args`).
    resolved_model_family = model_family or _self_described_model_family(checkpoint)
    is_egostitch_e2e_family = resolved_model_family == "egostitch_e2e"
    # One ledger epoch per protocol run (P1 fix): the test and test_topology
    # passes below share this single, deterministic identity instead of each
    # opening its own epoch (which made an initial run without
    # `--rescore-reason` fail on the second pass). Computed unconditionally --
    # it is cheap and only ever forwarded on the wire when
    # `is_egostitch_e2e_family` -- so it stays stable even if a later branch
    # changes when it is used.
    scoring_run_id = _derive_scoring_run_id(arm=arm, seed=seed, checkpoint_sha256=checkpoint_sha256)

    def _score(
        pairs_source: str,
        *,
        support_namespace: str,
        allow_rescore_reason: bool,
        include_scoring_run_id: bool,
    ) -> Path:
        # The two held-out passes expose identical complete test-node support;
        # validation has its own complete V_val support. Never cross caches
        # between those role universes.
        output = scores_dir / f"{pairs_source}.npz"
        if reuse_existing_scores and output.is_file():
            # Opt-in only. A scoring pass costs hours, so a rerun after a later
            # pass failed must not redo the finished ones -- but reuse is never
            # implicit: a silently reused artifact from a different checkpoint
            # would be exactly the kind of wrong-but-quiet result this module
            # exists to prevent. `_require_same_checkpoint` still cross-checks
            # every reused artifact against the others below.
            logger.info("reusing existing %s scores: %s", pairs_source, output)
            return output
        args = _build_score_args(
            checkpoint=checkpoint,
            pairs=pairs_source,
            output=output,
            data_root=data_root,
            strategy=strategy,
            run_metadata=scoring_identity_path,
            f0_cache=scores_dir / f"f0_cache_{support_namespace}_support.pt",
            grounding_cache=scores_dir / f"grounding_cache_{support_namespace}_support.npz",
            pack_dir=pack_dir,
            scaffold_control=scaffold_control,
            topo_gen_control=topo_gen_control,
            rescore_reason=rescore_reason if allow_rescore_reason else None,
            scoring_run_id=scoring_run_id if include_scoring_run_id else None,
            allow_oracle_diagnostic=allow_oracle_diagnostic,
            model_family=model_family,
            model_config=model_config,
        )
        return score_runner(args)

    # 1. Select the fixed operating point before any held-out pair is read.
    validation_path = _score(
        "val_topology",
        support_namespace="vval",
        allow_rescore_reason=False,
        include_scoring_run_id=False,
    )
    validation_artifact = load_scores(validation_path)
    validate_artifact_precision(validation_artifact, label=str(validation_path))
    _require_pairs_source(validation_artifact, "val_topology", label=str(validation_path))
    _require_full_universe(validation_artifact, label=str(validation_path))
    _require_scoring_identity(
        artifact=validation_artifact,
        checkpoint_id=expected_checkpoint_id,
        strategy=strategy,
        topo_gen_control=topo_gen_control,
        label=str(validation_path),
    )
    validation_split = _load_val_region_split(data_root, strategy)
    config = MMDConfig()
    fixed_selection = select_fixed_threshold(
        pairs=list(validation_artifact.pairs()),
        logits=validation_artifact.logit.astype(np.float64),
        g_ref=validation_split.build_g_val(),
        buckets=validation_split.buckets,
        config=config,
    )

    # 2. Only after threshold freeze, score the held-out classification and
    # sampled-topology unions. Both passes share one scoring run,
    # sharing one scoring-run id (when this checkpoint is egostitch_e2e) so
    # they join the SAME test-access-ledger epoch.
    test_path = _score(
        "test",
        support_namespace="test",
        allow_rescore_reason=True,
        include_scoring_run_id=is_egostitch_e2e_family,
    )
    test_artifact = load_scores(test_path)
    validate_artifact_precision(test_artifact, label=str(test_path))
    _require_full_universe(test_artifact, label=str(test_path))
    _require_scoring_identity(
        artifact=test_artifact,
        checkpoint_id=expected_checkpoint_id,
        strategy=strategy,
        topo_gen_control=topo_gen_control,
        label=str(test_path),
    )

    topology_path = _score(
        "test_topology",
        support_namespace="test",
        allow_rescore_reason=True,
        include_scoring_run_id=is_egostitch_e2e_family,
    )
    topology_artifact = load_scores(topology_path)
    validate_artifact_precision(topology_artifact, label=str(topology_path))
    _require_pairs_source(topology_artifact, "test_topology", label=str(topology_path))
    _require_full_universe(topology_artifact, label=str(topology_path))
    _require_scoring_identity(
        artifact=topology_artifact,
        checkpoint_id=expected_checkpoint_id,
        strategy=strategy,
        topo_gen_control=topo_gen_control,
        label=str(topology_path),
    )

    _require_same_checkpoint(validation_artifact, test_artifact, topology_artifact)

    # 3. Classification uses the same validation-selected operating point:
    # logits shift by -t* so the frozen threshold sits at probability 0.5
    # (sigma(l - t*) >= 0.5 iff l >= t*), and ECE/Brier describe the
    # calibrated, deployed probabilities.
    logit_shift = -fixed_selection.logit_threshold
    edge_report = report_edge_metrics(
        test_path, expect_pairs_source="test", threshold=0.5, logit_shift=logit_shift
    )
    calibration_report: dict[str, object] = {
        "method": "logit_shift_to_validation_selected_threshold",
        "logit_shift": logit_shift,
        "selected_logit_threshold": fixed_selection.logit_threshold,
        "selected_probability_threshold": float(expit(fixed_selection.logit_threshold)),
    }

    # 4. Replay the frozen validation threshold unchanged on every test sample
    # -- the single reported topology operating point. Self-loops participate.
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    _, fixed_test_report = evaluate_fixed_threshold(
        pairs=list(topology_artifact.pairs()),
        logits=topology_artifact.logit.astype(np.float64),
        g_ref=g_ref,
        buckets=buckets,
        threshold=fixed_selection.logit_threshold,
        config=config,
    )

    graph_report: dict[str, object] = {
        "fixed_threshold": {
            "validation_selection": fixed_selection.report,
            "test": fixed_test_report,
        },
    }

    # 5. Arm identity: config-independent, sourced from the topology artifact.
    meta = topology_artifact.meta
    arm_report: dict[str, object] = {
        "arm": arm,
        "seed": seed,
        "model_family": meta.get("model_family"),
        "topo_gen_control": meta.get("topo_gen_control"),
        # `score_universe` never writes `run_kind` into score metadata, so the
        # artifact's own value is always absent. The published training
        # metadata is the only place a run's formal/diagnostic classification
        # survives, and losing it here would strip that provenance from every
        # report.
        "run_kind": (
            published_run_metadata.get("run_kind") if published_run_metadata is not None else None
        ),
        "checkpoint_id": meta.get("checkpoint_id"),
        "checkpoint_sha256": checkpoint_sha256,
        # Closes the checkpoint-selection audit gap: reports previously did not
        # say which epoch published. Same provenance source as `run_kind`
        # above -- absent when there is no published training dir (the two
        # scoring-time controls), when that dir's metadata predates this field,
        # or when the metadata describes a different checkpoint than the one
        # scored (scoring last.pt or a copied checkpoint in a published dir
        # must not inherit the best checkpoint's epoch).
        "selected_epoch": (
            published_run_metadata.get("selected_epoch")
            if published_run_metadata is not None
            and published_run_metadata.get("checkpoint_id") == meta.get("checkpoint_id")
            else None
        ),
    }

    provenance: dict[str, object] = {
        "val_topology": {
            "path": str(validation_path),
            "sha256": _sha256_file(validation_path),
            "test_access_ledger_record_sha256": None,
        },
        "test": {
            "path": str(test_path),
            "sha256": _sha256_file(test_path),
            "test_access_ledger_record_sha256": _ledger_record_sha256(test_artifact.meta),
        },
        "test_topology": {
            "path": str(topology_path),
            "sha256": _sha256_file(topology_path),
            "test_access_ledger_record_sha256": _ledger_record_sha256(topology_artifact.meta),
        },
    }

    # 6. Assemble and write. Every block above must already exist in memory --
    # nothing is written incrementally -- so a failure anywhere above (a
    # topology-pass score_runner error, a graph-evaluation ValueError, ...)
    # leaves no report file at all. Edge and graph metrics are always reported
    # together (CLAUDE.md claim rule); there is no code path that writes one
    # without the other.
    report: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "arm": arm_report,
        "edge": edge_report,
        "calibration": calibration_report,
        "graph": graph_report,
        "provenance": provenance,
    }
    report_path = output_dir / report_filename
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", report_path)
    return TestProtocolResult(report_path=report_path, report=report)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m src.eval.test_protocol`` CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.test_protocol",
        description=(
            "Run the post-train test protocol for one arm: select a fixed topology "
            "threshold on sampled V_val subgraphs, then score test/test_topology and "
            "report edge metrics calibrated to it plus its replayed topology."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="published checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--arm", required=True, help="arm identity recorded in the report")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--pack-dir", type=Path, default=None, help="GPU-resident packed BF16 feature directory"
    )
    parser.add_argument(
        "--scaffold-control", default=None, help="egostitch_e2e scoring-time structure control"
    )
    parser.add_argument(
        "--topo-gen-control",
        choices=["branch_zero", "shuffle"],
        default=None,
        help="v3_1 topology-generator scoring-time control",
    )
    parser.add_argument(
        "--rescore-reason",
        default=None,
        help="required reason for a repeated egostitch_e2e held-out scoring epoch",
    )
    parser.add_argument(
        "--model-family",
        choices=sorted(MODEL_BUILDERS),
        default=None,
        help="model family for a bare legacy state_dict checkpoint (e.g. cazi_mbn)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=(
            "training YAML for a bare legacy checkpoint (required together with "
            "--model-family; for cazi_mbn this is its own training YAML)"
        ),
    )
    parser.add_argument(
        "--allow-oracle-diagnostic",
        action="store_true",
        help=(
            "required to score a checkpoint whose egostitch_e2e generator is "
            "oracle_struct: a ceiling diagnostic only, never a formal result"
        ),
    )
    parser.add_argument("--report-filename", default="test_report.json")
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="GPUs to fan the default score_runner across; default detect_visible_gpu_count()",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="interpreter used to launch each score shard",
    )
    parser.add_argument(
        "--reuse-existing-scores",
        action="store_true",
        help="reuse already-written scores/<pairs>.npz instead of rescoring that universe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI end to end.

    This is the entry point for arms with no pipeline run to hang off (CAZI-MBN
    and the two scoring-time controls): its default `score_runner` calls
    `src.score_fanout.score_sharded` directly.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    # Deferred imports: src.score_fanout / src.e2_pipeline are owned by a
    # concurrently edited pipeline; run_test_protocol itself must not depend on
    # either module importing cleanly, only main() does.
    from src.e2_pipeline import detect_visible_gpu_count
    from src.score_fanout import score_sharded

    gpu_count = args.gpu_count if args.gpu_count is not None else detect_visible_gpu_count()

    def _score_runner(score_args: Sequence[str]) -> Path:
        return score_sharded(
            score_args,
            gpu_count=gpu_count,
            python_bin=args.python_bin,
        )

    result = run_test_protocol(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        strategy=args.strategy,
        arm=args.arm,
        seed=args.seed,
        score_runner=_score_runner,
        pack_dir=args.pack_dir,
        scaffold_control=args.scaffold_control,
        topo_gen_control=args.topo_gen_control,
        rescore_reason=args.rescore_reason,
        model_family=args.model_family,
        model_config=args.model_config,
        allow_oracle_diagnostic=args.allow_oracle_diagnostic,
        report_filename=args.report_filename,
        reuse_existing_scores=args.reuse_existing_scores,
    )
    logger.info("wrote %s", result.report_path)


if __name__ == "__main__":
    main()
