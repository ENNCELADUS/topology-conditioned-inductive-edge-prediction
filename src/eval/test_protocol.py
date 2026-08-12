"""Per-arm held-out test protocol: score, then report edge and graph metrics together.

Owns the whole post-train sequence for one arm and nothing else::

    score test      -> edge metrics at fixed threshold 0.5
    score candidate -> assembled-graph metrics at global density control
        -> test_report.json

Classification and topology use separate operating points: the former is
fixed at 0.5, while the latter matches the global simple-edge reference density
over the candidate universe. Both metric families are always reported together;
a partial report is never written.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.eval.assembly import assemble_graph, density_matched_threshold, threshold_sweep
from src.eval.graph_metrics import (
    MMDConfig,
    compute_graph_similarity,
    compute_relative_density,
    evaluate_assembled_graph,
    strip_self_loops,
)
from src.eval.report_edge_metrics import report_edge_metrics
from src.experiments.g1_hardened_e2 import (
    _BENCHMARK_SUBDIR,
    build_threshold_grid,
    load_test_graph,
    load_test_node_buckets,
)
from src.score_universe import (
    MODEL_BUILDERS,
    ScoresArtifact,
    _checkpoint_id,
    load_scores,
    validate_artifact_precision,
)

logger = logging.getLogger(__name__)

__all__ = ["ScoreRunner", "TestProtocolResult", "build_parser", "main", "run_test_protocol"]

_SCHEMA_VERSION = "test_protocol_v2"
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


def _require_same_checkpoint(*, test: ScoresArtifact, candidate: ScoresArtifact) -> None:
    """Raise unless both held-out passes scored the same checkpoint and family."""
    checkpoint_ids = {artifact.meta.get("checkpoint_id") for artifact in (test, candidate)}
    if len(checkpoint_ids) != 1:
        raise ValueError(
            f"test/candidate scoring passes used different checkpoints: {checkpoint_ids}"
        )
    model_families = {artifact.meta.get("model_family") for artifact in (test, candidate)}
    if len(model_families) != 1:
        raise ValueError(
            f"test/candidate scoring passes disagree on model_family: {model_families}"
        )


def _expected_checkpoint_id(checkpoint: Path) -> str | None:
    """Return the score-artifact id for a real checkpoint, if it carries state."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    state = payload.get("model_state", payload.get("state_dict"))
    if state is None and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
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
    *, artifact: ScoresArtifact, checkpoint_id: str | None, strategy: str, label: str
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
    epoch across its test and candidate passes (the P1 defect this fixes: those
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


def _graph_block(
    *,
    threshold: float,
    pairs: Sequence[tuple[str, str]],
    probs: NDArray[np.float64],
    nodes: Sequence[str],
    g_ref: nx.Graph,
    g_ref_simple: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> dict[str, object]:
    """Evaluate the five topology numbers at one assembly threshold.

    Reports GS and RD both ways, named separately (CLAUDE.md claim rule):
    ``global_simple_edge`` (the whole assembled graph, self-loops stripped,
    :func:`compute_graph_similarity`/:func:`compute_relative_density`) and
    ``bfs_macro`` (the official per-subgraph macro mean over benchmark buckets,
    self-loops retained, :func:`evaluate_assembled_graph`). The three MMD
    ratios (degree/clustering/spectral) have no global analogue -- they are
    defined only as a bucketed reference-normalized statistic -- so only the
    ``bfs_macro`` evaluation produces them.
    """
    g_pred = assemble_graph(pairs, probs, threshold=threshold, nodes=nodes)
    bucketed = evaluate_assembled_graph(g_pred, g_ref, buckets, config)
    g_pred_simple = strip_self_loops(g_pred)
    return {
        "threshold": threshold,
        "graph_similarity": {
            "global_simple_edge": compute_graph_similarity(g_pred_simple, g_ref_simple),
            "bfs_macro": bucketed.graph_similarity,
        },
        "relative_density": {
            "global_simple_edge": compute_relative_density(g_pred_simple, g_ref_simple),
            "bfs_macro": bucketed.relative_density,
        },
        "mmd_ratio": dict(bucketed.mmd_ratio),
        "self_loops": {"predicted": bucketed.self_loops_pred, "reference": bucketed.self_loops_ref},
    }


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
        rescore_reason: Required by the test-access ledger when this
            ``(arm, seed)`` has already opened held-out data.
        model_family: Explicit model family for a bare legacy checkpoint (only
            ``cazi_mbn`` today, whose released
            ``{"state_dict": ..., "best_val_auroc": ...}`` checkpoint does not
            self-describe). Leave ``None`` for a self-describing Task-4
            checkpoint (``v3_1``/``egostitch_e2e``): this function reads the
            checkpoint's own embedded ``model_family`` in that case, both to
            decide whether the test/candidate passes may carry
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
    legacy_artifacts = [output_dir / "operating_point.json", scores_dir / "v_hold.npz"]
    stale = [str(path) for path in legacy_artifacts if path.exists()]
    if stale:
        raise ValueError(f"remove obsolete max-F1/V_hold test artifacts before scoring: {stale}")

    # Fix for the P1 defect where this module clobbered a published
    # run_metadata.json: validate it in place (if present) and write this
    # call's own scoring identity under a filename that can never collide
    # with it. `--run-metadata` for every pass below points at OUR file, never
    # `_RUN_METADATA_FILENAME`.
    published_run_metadata = _validate_published_run_metadata(output_dir, arm=arm, seed=seed)
    scoring_identity_path = _write_scoring_identity(output_dir, arm=arm, seed=seed)

    checkpoint_sha256 = _sha256_file(checkpoint)
    expected_checkpoint_id = _expected_checkpoint_id(checkpoint)

    # egostitch_e2e held-out scoring (test/candidate) is the only case
    # `--scoring-run-id` is valid for; every other family/pairs-source
    # combination has `src.score_universe` reject it outright. A Task-4
    # checkpoint self-describes its family, so this reads that rather than
    # requiring every caller to pass `model_family` just to score a v3_1 or
    # egostitch_e2e checkpoint (which never needs `--model-family` on the
    # wire at all -- see `_build_score_args`).
    resolved_model_family = model_family or _self_described_model_family(checkpoint)
    is_egostitch_e2e_family = resolved_model_family == "egostitch_e2e"
    # One ledger epoch per protocol run (P1 fix): the test and candidate
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
        allow_rescore_reason: bool,
        include_scoring_run_id: bool,
    ) -> Path:
        # Universe-scoped F0/grounding caches (P1 fix): each pairs source gets
        # its own cache file, so test and candidate never collide despite
        # using different benchmark-universe node sets --
        # the exact-match `build_f0_matrix(..., allow_cache_subset=False)`
        # collision this used to hit. Deliberately not relying on
        # `allow_cache_subset=True` (CLAUDE.md trap: silently gathers a
        # superset cache into a different node set with no content check).
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
            f0_cache=scores_dir / f"f0_cache_{pairs_source}.pt",
            grounding_cache=scores_dir / f"grounding_cache_{pairs_source}.npz",
            pack_dir=pack_dir,
            scaffold_control=scaffold_control,
            rescore_reason=rescore_reason if allow_rescore_reason else None,
            scoring_run_id=scoring_run_id if include_scoring_run_id else None,
            allow_oracle_diagnostic=allow_oracle_diagnostic,
            model_family=model_family,
            model_config=model_config,
        )
        return score_runner(args)

    # 1. Score test, then candidate. Both held-out passes share one scoring run,
    # sharing one scoring-run id (when this checkpoint is egostitch_e2e) so
    # they join the SAME test-access-ledger epoch.
    test_path = _score(
        "test", allow_rescore_reason=True, include_scoring_run_id=is_egostitch_e2e_family
    )
    candidate_path = _score(
        "candidate", allow_rescore_reason=True, include_scoring_run_id=is_egostitch_e2e_family
    )

    # 2. Classification uses the fixed 0.5 decision threshold for every arm.
    edge_report = report_edge_metrics(test_path, expect_pairs_source="test", threshold=0.5)

    # Reload test/candidate directly (report_edge_metrics only returns a
    # metrics summary, not the raw artifact) so their meta is available for the
    # arm/provenance blocks, and so graph assembly has candidate's probs/pairs.
    test_artifact = load_scores(test_path)
    validate_artifact_precision(test_artifact, label=str(test_path))
    _require_full_universe(test_artifact, label=str(test_path))
    _require_scoring_identity(
        artifact=test_artifact,
        checkpoint_id=expected_checkpoint_id,
        strategy=strategy,
        label=str(test_path),
    )

    candidate_artifact = load_scores(candidate_path)
    validate_artifact_precision(candidate_artifact, label=str(candidate_path))
    _require_pairs_source(candidate_artifact, "candidate", label=str(candidate_path))
    _require_full_universe(candidate_artifact, label=str(candidate_path))
    _require_scoring_identity(
        artifact=candidate_artifact,
        checkpoint_id=expected_checkpoint_id,
        strategy=strategy,
        label=str(candidate_path),
    )

    _require_same_checkpoint(test=test_artifact, candidate=candidate_artifact)

    # 3. Topology uses only the global density-matched assembly threshold.
    benchmark_root = data_root / _BENCHMARK_SUBDIR
    g_ref = load_test_graph(benchmark_root, strategy)
    buckets = load_test_node_buckets(benchmark_root, strategy)
    g_ref_simple = strip_self_loops(g_ref)
    config = MMDConfig()

    nodes = list(g_ref.nodes())
    pairs = list(candidate_artifact.pairs())
    probs = candidate_artifact.probs()

    # Density-matched reference: non-self candidate rows only, matched against
    # the SIMPLE (self-loop-stripped) reference edge count. Self-pairs never
    # consume this quota but still assemble as self-loops at whatever threshold
    # is chosen -- mirrors src.experiments.g1_hardened_e2.run_threshold_sweep
    # exactly (CLAUDE.md: "Self-loop handling ... mirror g1_hardened_e2.py").
    target_edges = g_ref_simple.number_of_edges()
    non_self_mask = candidate_artifact.u_idx != candidate_artifact.v_idx
    density_threshold = density_matched_threshold(probs[non_self_mask], target_edges)

    graph_report: dict[str, object] = {
        "density_matched_threshold": {
            **_graph_block(
                threshold=density_threshold,
                pairs=pairs,
                probs=probs,
                nodes=nodes,
                g_ref=g_ref,
                g_ref_simple=g_ref_simple,
                buckets=buckets,
                config=config,
            ),
            "target_edges": target_edges,
        },
    }

    # 4. Threshold-sweep secondary view contains the density-matched point and
    # pinned percentiles of the candidate probability distribution.
    grid = build_threshold_grid(probs, density_threshold)
    sweep_points = threshold_sweep(
        pairs, probs, thresholds=grid, g_ref=g_ref, buckets=buckets, config=config
    )
    sweep_report: list[dict[str, object]] = [asdict(point) for point in sweep_points]

    # 5. Arm identity: config-independent, sourced from the candidate artifact.
    meta = candidate_artifact.meta
    arm_report: dict[str, object] = {
        "arm": arm,
        "seed": seed,
        "model_family": meta.get("model_family"),
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
    }

    provenance: dict[str, object] = {
        "test": {
            "path": str(test_path),
            "sha256": _sha256_file(test_path),
            "test_access_ledger_record_sha256": _ledger_record_sha256(test_artifact.meta),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "test_access_ledger_record_sha256": _ledger_record_sha256(candidate_artifact.meta),
        },
    }

    # 6. Assemble and write. Every block above must already exist in memory --
    # nothing is written incrementally -- so a failure anywhere above (a
    # candidate-pass score_runner error, a graph-evaluation ValueError, ...)
    # leaves no report file at all. Edge and graph metrics are always reported
    # together (CLAUDE.md claim rule); there is no code path that writes one
    # without the other.
    report: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "arm": arm_report,
        "edge": edge_report,
        "graph": graph_report,
        "sweep": sweep_report,
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
            "Run the post-train test protocol for one arm: score test/candidate, "
            "report test classification at threshold 0.5 and candidate topology "
            "at the global density-matched assembly threshold."
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
        "--timeout-seconds",
        type=float,
        default=None,
        help="optional per-shard wall-clock deadline; omit to wait indefinitely",
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
            timeout_seconds=args.timeout_seconds,
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
