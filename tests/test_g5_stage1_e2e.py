"""E2E liveness and eight-arm gate contracts."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
from src import score_universe
from src import train_egostitch as te
from src.data import internal_holdout
from src.experiments import b0_cal, g5_stage1, probes
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E
from src.score_universe import ScoresArtifact, load_scores, save_scores
from src.train_b0 import ModelConfig, _state_digest

from tests.test_b0_cal import _toy_inputs as _b0cal_toy_inputs
from tests.test_g1_hardened_e2 import (
    _NODES,
    _POSITIVE_EDGES,
    _universe_rows,
    _write_universe_npz,
)
from tests.test_score_universe import _write_feature_store
from tests.test_train_egostitch import _E2E_TINY_MODEL, _toy_cfg
from tests.test_train_egostitch_e2e import (
    _E2E_PIPELINE_NODES,
    _e2e_pipeline_benchmark,
    _write_tiny_token_pack,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- shared fixtures
# Re-homed from `tests/test_g5_stage1.py`, which was deleted with the frozen-s0
# `egostitch` pipeline (design 2026-07-29 Sec 6.2). Only the e2e gate reads them.


def _d(x: object) -> dict[str, Any]:
    return cast(dict[str, Any], x)


_PREREG = {
    "registration_id": "toy-prereg",
    "status": "BINDING",
    "seeds": [0],
    "primary_criteria": {"decision_procedure": "single_seed_point_estimate_dominance"},
    "failure_reading": "pre-registered failure reading text (verbatim)",
    "decision_rules_5_2_verbatim": ["rule one", "rule two"],
    "fidelity_validity_gate": {
        "min_residual_std_ratio": 1e-5,
        "max_spearman": 0.9999,
        "max_topk_overlap": 0.9999,
        "topk_fraction": 0.25,
    },
}


def _write_prereg(tmp_path: Path, b0_path: Path | None = None) -> Path:
    path = tmp_path / "prereg.json"
    payload = dict(_PREREG)
    if b0_path is not None:
        g1_path = tmp_path / "g1_results.json"
        g3_path = tmp_path / "g3_results.json"
        g1_path.write_text("{}\n")
        g3_path.write_text("{}\n")
        payload["frozen_inputs"] = {
            "b0_candidate_scores": {
                "path": str(b0_path),
                "sha256": hashlib.sha256(b0_path.read_bytes()).hexdigest(),
                "checkpoint_id": "deadbeefcafefeed",
            },
            "g1_results": {
                "path": str(g1_path),
                "sha256": hashlib.sha256(g1_path.read_bytes()).hexdigest(),
            },
            "g3_results": {
                "path": str(g3_path),
                "sha256": hashlib.sha256(g3_path.read_bytes()).hexdigest(),
            },
        }
        candidate_path = (
            tmp_path / "data" / "benchmark_2025_neurips" / "toy" / "candidate_test_edges.txt"
        )
        if candidate_path.is_file():
            _d(payload["frozen_inputs"])["candidate_manifest"] = {
                "path": str(candidate_path),
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            }
        b0cal_path = tmp_path / "b0cal" / "b0cal_results.json"
        if not b0cal_path.is_file():
            b0cal_path.parent.mkdir(exist_ok=True)
            b0cal_path.write_text("{}\n")
        _d(payload["frozen_inputs"])["b0cal_results"] = {
            "path": str(b0cal_path),
            "sha256": hashlib.sha256(b0cal_path.read_bytes()).hexdigest(),
        }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- e2e family:
# within-checkpoint liveness + eight-arm summary (Task 11)


class TestWithinCheckpointLivenessGuard:
    @staticmethod
    def _artifact(full: np.ndarray, f_logit: np.ndarray) -> ScoresArtifact:
        n = len(full)
        return ScoresArtifact(
            node_ids=[f"n{i}" for i in range(n)],
            u_idx=np.arange(n, dtype=np.int32),
            v_idx=np.arange(n, dtype=np.int32),
            logit=full.astype(np.float32),
            label=np.zeros(n, dtype=np.int8),
            meta={"model_family": "egostitch_e2e"},
            f_logit=f_logit.astype(np.float32),
        )

    def test_reports_failure_when_full_equals_f_logit(self) -> None:
        f_logit = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        artifact = self._artifact(f_logit.copy(), f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=0.01,
        )
        assert report["quality_pass"] is False

    def test_reports_failure_on_pair_invariant_residual(self) -> None:
        # A tiny constant offset scales the residual so small that the
        # conjunctive rule still fires, mirroring the frozen-s0 dead-residual
        # test: the residual must genuinely vary with the pair, not merely be
        # small in magnitude.
        f_logit = np.linspace(-2.0, 2.0, 200)
        full = f_logit + 1e-9
        artifact = self._artifact(full, f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=0.01,
        )
        assert report["quality_pass"] is False

    def test_does_not_fire_on_decorrelated_full(self) -> None:
        rng = np.random.default_rng(0)
        f_logit = rng.normal(size=200)
        full = rng.normal(size=200)  # independent of f_logit -> genuinely alive residual
        artifact = self._artifact(full, f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=0.01,
        )
        assert report["residual_std"] > 0

    def test_accepts_alive_residual_even_when_small(self) -> None:
        f_logit = np.array([-1.0, 0.0, 1.0])
        full = np.array([-0.99, -0.02, 1.03])
        artifact = self._artifact(full, f_logit)
        report = g5_stage1.validate_dead_residual_within_checkpoint(
            artifact,
            min_residual_std_ratio=1e-5,
            max_spearman=0.9999,
            max_topk_overlap=0.9999,
            topk_fraction=1 / 3,
        )
        assert report["residual_std"] > 0

    def test_requires_f_logit_array(self) -> None:
        artifact = ScoresArtifact(
            node_ids=["a"],
            u_idx=np.array([0], dtype=np.int32),
            v_idx=np.array([0], dtype=np.int32),
            logit=np.array([0.0], dtype=np.float32),
            label=np.array([0], dtype=np.int8),
            meta={},
        )
        with pytest.raises(ValueError, match="f_logit"):
            g5_stage1.validate_dead_residual_within_checkpoint(
                artifact,
                min_residual_std_ratio=1e-5,
                max_spearman=0.9999,
                max_topk_overlap=0.9999,
                topk_fraction=0.01,
            )


def _write_e2e_universe_npz(
    path: Path,
    *,
    node_ids: list[str],
    pairs: list[tuple[str, str]],
    full: np.ndarray,
    f_logit: np.ndarray,
    pair_content: np.ndarray,
    pair_topology: np.ndarray,
    labels: np.ndarray,
    strategy: str = "toy",
    checkpoint_id: str = "e2e0",
    scaffold_control: str = "none",
    permanent_null: str = "none",
    primary_logit: str = "full",
    scoring_arm: str = "full",
    arm_kind: str = "trained_checkpoint",
    checkpoint_arm: str = "full",
    scoring_semantics: dict[str, object] | None = None,
) -> None:
    """Write a toy family-``egostitch_e2e`` four-array scores artifact."""
    access = score_universe._TestAccessContext(
        ledger_path=path.parent / "test_access_ledger.jsonl",
        scoring_arm=scoring_arm,
        seed=0,
        output=path,
        shard=0,
        num_shards=1,
        rescore_reason=("replace test fixture score artifact" if path.exists() else None),
    )
    score_universe._record_test_access(access, pairs_source="candidate")
    assert access.ledger_binding is not None
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.array([position[u] for u, _ in pairs], dtype=np.int32)
    v_idx = np.array([position[v] for _, v in pairs], dtype=np.int32)
    meta: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "model_family": "egostitch_e2e",
        "pairs_source": "candidate",
        "strategy": strategy,
        "num_rows": len(pairs),
        "created_utc": "2026-07-17T00:00:00+00:00",
        "torch_version": "2.10.0",
        "score_precision": {
            "contract": "egostitch_e2e_pair_fp32_v1",
            "encode_autocast": "off",
            "pair_compute_dtype": "float32",
            "pair_autocast": False,
            "logit_storage_dtype": "float32",
        },
        "scaffold_control": {
            "mode": scaffold_control,
            "seed": 0,
            "keying": "canonical_pair_v1",
        },
        "permanent_null": permanent_null,
        "primary_logit": primary_logit,
        "scoring_arm": scoring_arm,
        "arm_kind": arm_kind,
        "checkpoint_arm": checkpoint_arm,
        "scoring_semantics": scoring_semantics
        or {
            "scaffold_control": scaffold_control,
            "permanent_null": permanent_null,
            "primary_logit": primary_logit,
        },
        "test_access_ledger": access.ledger_binding,
    }
    arrays = {
        "full": full,
        "f_logit": f_logit,
        "pair_content": pair_content,
        "pair_topology": pair_topology,
    }
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=arrays[primary_logit].astype(np.float32),
        label=labels.astype(np.int8),
        row_start=0,
        meta=meta,
        f_logit=f_logit.astype(np.float32),
        pair_content=pair_content.astype(np.float32),
        pair_topology=pair_topology.astype(np.float32),
        full_logit=full.astype(np.float32) if primary_logit != "full" else None,
    )


def _rewrite_e2e_artifact(
    path: Path,
    *,
    meta: dict[str, object] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    labels: np.ndarray | None = None,
) -> None:
    artifact = load_scores(path)
    resolved_pairs = pairs if pairs is not None else list(artifact.pairs())
    node_ids = list(artifact.node_ids)
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    u_idx = np.asarray([position[node_u] for node_u, _ in resolved_pairs], dtype=np.int32)
    v_idx = np.asarray([position[node_v] for _, node_v in resolved_pairs], dtype=np.int32)
    save_scores(
        path,
        node_ids=node_ids,
        u_idx=u_idx,
        v_idx=v_idx,
        logit=artifact.logit,
        label=artifact.label if labels is None else labels,
        row_start=0,
        meta=artifact.meta,
        f_logit=artifact.f_logit,
        pair_content=artifact.pair_content,
        pair_topology=artifact.pair_topology,
        full_logit=artifact.full_logit,
    )
    if meta is not None:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez_compressed(path, **arrays)


def _refresh_formal_run_metadata_hashes(inputs: dict[str, Any]) -> None:
    """Model rescoring after a semantically valid run-metadata fixture rewrite."""
    arm_paths = _d(inputs["arm_universe_paths"])
    metadata_paths = _d(inputs["run_metadata_paths"])
    registration_sha256 = hashlib.sha256(
        cast(Path, inputs["preregistration_path"]).read_bytes()
    ).hexdigest()
    for metadata_path in metadata_paths.values():
        metadata = json.loads(metadata_path.read_text())
        metadata["preregistration_sha256"] = registration_sha256
        metadata_path.write_text(json.dumps(metadata))
    for name, artifact_path in arm_paths.items():
        source_arm = "full" if name in g5_stage1._E2E_CONTROL_ARMS else name
        artifact = load_scores(artifact_path)
        meta = dict(artifact.meta)
        provenance = dict(_d(meta["formal_scoring_provenance"]))
        provenance["registration_sha256"] = registration_sha256
        provenance["run_metadata_sha256"] = hashlib.sha256(
            metadata_paths[source_arm].read_bytes()
        ).hexdigest()
        all_formal_arms = dict(_d(provenance["all_formal_arms"]))
        for arm, metadata_path in metadata_paths.items():
            arm_provenance = dict(_d(all_formal_arms[arm]))
            arm_provenance["run_metadata_sha256"] = hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest()
            all_formal_arms[arm] = arm_provenance
        provenance["all_formal_arms"] = all_formal_arms
        meta["formal_scoring_provenance"] = provenance
        _rewrite_e2e_artifact(artifact_path, meta=meta)


_E2E_LIVENESS_CONFIG = {
    "min_residual_std_ratio": 1e-5,
    "max_spearman": 0.9999,
    "max_topk_overlap": 0.9999,
    "topk_fraction": 0.01,
}


def test_real_probe_producer_artifact_is_accepted_by_g5_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production probe writer and the gate consumer as one contract."""
    benchmark = _e2e_pipeline_benchmark()
    nodes = list(_E2E_PIPELINE_NODES)
    data_root = tmp_path / "data"
    strategy_dir = data_root / te._BENCHMARK_SUBDIR / "toy"
    strategy_dir.mkdir(parents=True)
    with (strategy_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": nodes, "test": []}, handle)
    train_pairs = list(benchmark.split.train_pairs.pairs)
    (strategy_dir / "train_edges.txt").write_text(
        "".join(f"{node_u}\t{node_v}\t1\n" for node_u, node_v in train_pairs),
        encoding="utf-8",
    )

    torch.manual_seed(0)
    node_tokens = {
        node: torch.randn(3 + (index % 3), 1536)
        for index, node in enumerate(nodes)
    }
    _write_feature_store(
        data_root / te._FEATURES_SUBDIR,
        node_tokens,
        input_dim=1536,
    )
    token_pack = tmp_path / "token-pack"
    _write_tiny_token_pack(token_pack, nodes, min_length=3)

    model_config = {**_E2E_TINY_MODEL, "n_ground": 3}
    base_cfg = _toy_cfg(tmp_path)
    cfg = replace(
        base_cfg,
        model=ModelConfig(family="egostitch_e2e", config=model_config),
        data=replace(
            base_cfg.data,
            root=data_root,
            strategy="toy",
            pack_dir=token_pack,
            edge_batch=8,
            expected_missing_features=(),
        ),
    )
    config_path = tmp_path / "config.yaml"
    probe_path = tmp_path / "probe.npz"
    registration_path = cfg.preregistration
    registration = {
        "status": "BINDING",
        "seeds": [0],
        "probe_artifact": {
            "format": "egostitch_e2e_probe_v2",
            "source_arm": "full",
            "expected_path": str(probe_path),
        },
        "arms": {
            "full": {
                "training": str(config_path),
                "n_ground": 3,
            }
        },
    }
    registration_path.write_text(
        json.dumps(registration, sort_keys=True),
        encoding="utf-8",
    )
    registration_sha = hashlib.sha256(registration_path.read_bytes()).hexdigest()

    model = EgoStitchE2E(E2EConfig.from_mapping(model_config))
    checkpoint_path = tmp_path / "best.pt"
    state = model.state_dict()
    torch.save(
        {
            "model_state": state,
            "model_family": "egostitch_e2e",
            "model_config": model_config,
            "epoch": 0,
            "val_metrics": {},
            "seed": 0,
            "config": {},
        },
        checkpoint_path,
    )
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "preregistration_sha256": registration_sha,
                "run_kind": "formal",
                "status": "complete",
                "formal_artifacts_published": True,
                "permanent_null": "none",
                "seed": 0,
                "partition_seed": 0,
                "config_path": str(config_path),
                "config_hash": te._config_hash(cfg),
                "checkpoint_id": _state_digest(state)[:16],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(te, "load_config", lambda _path: cfg)

    def tiny_internal_holdout(
        train_nodes: list[str],
        e_msg: frozenset[tuple[str, str]],
        _e_sup: frozenset[tuple[str, str]],
    ) -> SimpleNamespace:
        # One holdout, not two (design 2026-07-29 Sec 2.1): `V_hold` is the
        # union of the former `V_qual`/`V_select` draws, and `V_fit ∪ V_hold`
        # must cover every formal node exactly -- the producer builds one
        # grounding pool per role and every probe node must land in one.
        v_fit = frozenset(train_nodes[:9])
        v_hold = frozenset(train_nodes[9:])

        def induced(nodes_subset: frozenset[str]) -> frozenset[tuple[str, str]]:
            return frozenset(
                (node_u, node_v)
                for node_u, node_v in e_msg
                if node_u in nodes_subset and node_v in nodes_subset
            )

        return SimpleNamespace(
            v_fit=v_fit,
            v_hold=v_hold,
            e_msg_fit=induced(v_fit),
            hold_manifest=SimpleNamespace(positive_edges=induced(v_hold)),
        )

    monkeypatch.setattr(
        internal_holdout,
        "derive_internal_holdout",
        tiny_internal_holdout,
    )
    original_select_probe_pairs = probes.select_probe_pairs
    monkeypatch.setattr(
        probes,
        "select_probe_pairs",
        lambda graph, limit=1000: original_select_probe_pairs(graph, limit=8),
    )

    probes.produce_e2e_probe_artifact(
        checkpoint_path=checkpoint_path,
        run_metadata_path=run_metadata_path,
        preregistration_path=registration_path,
        data_root=data_root,
        strategy="toy",
        output_path=probe_path,
        scope="formal_train",
    )
    with np.load(probe_path, allow_pickle=False) as archive:
        produced_metadata = json.loads(str(archive["meta"].item()))
    assert produced_metadata["format"] == "egostitch_e2e_probe_v2"

    report = g5_stage1._evaluate_registered_e2e_probe(
        probe_artifact_path=probe_path,
        preregistration=registration,
        preregistration_path=registration_path,
        run_metadata_path=run_metadata_path,
        data_root=data_root,
        strategy="toy",
    )
    assert cast(dict[str, object], report["metadata"])["format"] == (
        "egostitch_e2e_probe_v2"
    )


def test_g5_evaluator_rejects_v1_probe_registration(tmp_path: Path) -> None:
    with pytest.raises(
        g5_stage1.PreregistrationMismatch,
        match="egostitch_e2e_probe_v2.*egostitch_e2e_probe_v1.*rejected",
    ):
        g5_stage1._evaluate_registered_e2e_probe(
            probe_artifact_path=tmp_path / "probe-v1.npz",
            preregistration={
                "probe_artifact": {
                    "format": "egostitch_e2e_probe_v1",
                    "expected_path": str(tmp_path / "probe-v1.npz"),
                }
            },
            preregistration_path=tmp_path / "registration.json",
            run_metadata_path=tmp_path / "run_metadata.json",
            data_root=tmp_path / "data",
            strategy="toy",
        )


def _eight_arm_inputs(tmp_path: Path) -> dict[str, Any]:
    """Toy benchmark + eight E2E arm universes + six trained-run metadata records."""
    b0_universe_path, _val_path, data_root = _b0cal_toy_inputs(tmp_path)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(42)
    full = rng.normal(size=len(pairs))
    f_logit = rng.normal(size=len(pairs))  # independent of `full` -> alive residual
    pair_content = rng.normal(size=len(pairs))
    pair_topology = rng.normal(size=len(pairs))
    candidate_path = data_root / "benchmark_2025_neurips" / "toy" / "candidate_test_edges.txt"
    candidate_path.write_text(
        "".join(
            f"{node_u}\t{node_v}\t{int(label)}\n"
            for (node_u, node_v), label in zip(pairs, labels, strict=True)
        )
    )

    arm_paths: dict[str, Path] = {}
    for name in g5_stage1._E2E_ARMS:
        provenance = {
            "full": ("none", "none", "full"),
            "b0_e2e_f_only": ("none", "all_head", "f_logit"),
            "pair_topology": ("none", "content_head", "pair_topology"),
            "p0": ("none", "none", "full"),
            "cosine_pool": ("none", "none", "full"),
            "no_l_rel": ("none", "none", "full"),
            "structure_control_6a_v3": ("shuffle_within_pair_v3", "none", "full"),
            "structure_control_6e_v1": ("rewire_checkerboard_v1", "none", "full"),
        }[name]
        kind = (
            "scoring_time_control"
            if name in g5_stage1._E2E_CONTROL_ARMS
            else "trained_checkpoint"
        )
        checkpoint_arm = "full" if kind == "scoring_time_control" else name
        scoring_semantics: dict[str, object] = {
            "scaffold_control": provenance[0],
            "permanent_null": provenance[1],
            "primary_logit": provenance[2],
        }
        if kind == "scoring_time_control":
            scoring_semantics.update(
                {"seed": 0, "keying": "canonical_pair_v1", "checkpoint_arm": "full"}
            )
        path = tmp_path / f"{name}.npz"
        _write_e2e_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            full=full,
            f_logit=f_logit,
            pair_content=pair_content,
            pair_topology=pair_topology,
            labels=labels,
            checkpoint_id=("ckpt_full" if kind == "scoring_time_control" else f"ckpt_{name}"),
            scaffold_control=provenance[0],
            permanent_null=provenance[1],
            primary_logit=provenance[2],
            scoring_arm=name,
            arm_kind=kind,
            checkpoint_arm=checkpoint_arm,
            scoring_semantics=scoring_semantics,
        )
        arm_paths[name] = path

    preregistration_path = _write_prereg(tmp_path, b0_universe_path)
    config_root = Path(__file__).resolve().parents[1] / "configs"
    arm_config_paths = {
        "full": config_root / "egostitch_e2e_v3_full_breadth_first.yaml",
        "b0_e2e_f_only": config_root / "egostitch_e2e_v3_f_only_breadth_first.yaml",
        "pair_topology": config_root / "egostitch_e2e_v3_pair_topology_breadth_first.yaml",
        "p0": config_root / "egostitch_e2e_v3_p0_breadth_first.yaml",
        "cosine_pool": config_root / "egostitch_e2e_v3_cosine_pool_breadth_first.yaml",
        "no_l_rel": config_root / "egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
    }
    preregistration = json.loads(preregistration_path.read_text())
    preregistration["benchmark"] = {"strategy": "toy"}
    preregistration["arms"] = {
        **{
            name: {
                "kind": "trained_checkpoint",
                "training": str(path),
                "scoring_provenance": {
                    "scaffold_control": {
                        "full": "none",
                        "b0_e2e_f_only": "none",
                        "pair_topology": "none",
                        "p0": "none",
                        "cosine_pool": "none",
                        "no_l_rel": "none",
                    }[name],
                    "permanent_null": {
                        "full": "none",
                        "b0_e2e_f_only": "all_head",
                        "pair_topology": "content_head",
                        "p0": "none",
                        "cosine_pool": "none",
                        "no_l_rel": "none",
                    }[name],
                    "primary_logit": {
                        "full": "full",
                        "b0_e2e_f_only": "f_logit",
                        "pair_topology": "pair_topology",
                        "p0": "full",
                        "cosine_pool": "full",
                        "no_l_rel": "full",
                    }[name],
                },
            }
            for name, path in arm_config_paths.items()
        },
        "structure_control_6a_v3": {
            "kind": "scoring_time_control",
            "training": None,
            "checkpoint_arm": "full",
            "scoring_provenance": {
                "scaffold_control": "shuffle_within_pair_v3",
                "seed": 0,
                "keying": "canonical_pair_v1",
                "permanent_null": "none",
                "primary_logit": "full",
                "checkpoint_arm": "full",
            },
        },
        "structure_control_6e_v1": {
            "kind": "scoring_time_control",
            "training": None,
            "checkpoint_arm": "full",
            "scoring_provenance": {
                "scaffold_control": "rewire_checkerboard_v1",
                "seed": 0,
                "keying": "canonical_pair_v1",
                "permanent_null": "none",
                "primary_logit": "full",
                "checkpoint_arm": "full",
            },
        },
    }
    evidence_path = tmp_path / "binding-evidence.json"
    evidence_path.write_text('{"verified": true}\n')
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence_artifact = {"path": str(evidence_path), "sha256": evidence_sha256}
    def validation_events(name: str, run_kind: str, epochs: int) -> list[dict[str, object]]:
        raw = [("step_0", None, 0), ("phase_a_end", 1, 1)]
        raw.extend(("epoch_end", epoch, epoch) for epoch in range(1, epochs + 1))
        return [
            {
                "ordinal": ordinal,
                "kind": kind,
                "epoch": epoch,
                "optimizer_step": step,
                "run_kind": run_kind,
                "arm": name,
                "validation_role": "V_hold",
            }
            for ordinal, (kind, epoch, step) in enumerate(raw, start=1)
        ]

    implementation_commit = "4280c4b"
    preregistration["binding_evidence"] = {
        "schema_version": "egostitch_e2e_binding_evidence_v2",
        "implementation": {"commit": implementation_commit},
        "configs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in arm_config_paths.items()
        },
        "parameter_group_manifests": {"fixture": evidence_artifact},
        "packs_and_validation_manifests": {"fixture": evidence_artifact},
        "boundary_access_audit": {"fixture": evidence_artifact},
        "runtime_and_peak_memory": {"fixture": evidence_artifact},
        "checkpoint_policy_version": "fixture-v1",
    }
    preregistration["evaluator"] = {"seed": 0}
    preregistration["registration_id"] = (
        "g5-e2e-stage1-20260729-two-stage-ladder-screen-v4-draft"
    )
    preregistration_path.write_text(json.dumps(preregistration, sort_keys=True, indent=2) + "\n")
    preregistration_sha256 = hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
    run_metadata_paths: dict[str, Path] = {}
    for name in g5_stage1._E2E_FORMAL_ARMS:
        permanent_null, p_topo, p_cont = {
            "full": ("none", 0.15, 0.15),
            "b0_e2e_f_only": ("all_head", 0.15, 0.15),
            "pair_topology": ("content_head", 0.15, 0.15),
            "p0": ("none", 0.0, 0.0),
            "cosine_pool": ("none", 0.15, 0.15),
            "no_l_rel": ("none", 0.15, 0.15),
        }[name]
        scoring_semantics = cast(
            dict[str, object], preregistration["arms"][name]["scoring_provenance"]
        )
        # One directory per arm, as `_publish_staged` produces: the margin
        # verdict is read from beside the run metadata it certifies.
        arm_run_dir = tmp_path / "runs" / name
        arm_run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = arm_run_dir / "run_metadata.json"
        formal_events = arm_run_dir / te.V_HOLD_VALIDATION_EVENTS_FILENAME
        formal_event_rows = validation_events(name, "formal", 3)
        formal_events.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in formal_event_rows)
        )
        meta_path.write_text(
            json.dumps(
                {
                    "preregistration_sha256": preregistration_sha256,
                    "checkpoint_id": f"ckpt_{name}",
                    "arm": name,
                    "arm_kind": "trained_checkpoint",
                    "checkpoint_arm": name,
                    "scoring_semantics": scoring_semantics,
                    "run_kind": "formal",
                    "formal_artifacts_published": True,
                    "status": "complete",
                    "validation_role": "V_hold",
                    "v_hold_validation_evidence": {
                        "schema": "egostitch_e2e_v_hold_validation_events_v1",
                        "count": len(formal_event_rows),
                        "path": formal_events.name,
                        "sha256": hashlib.sha256(formal_events.read_bytes()).hexdigest(),
                    },
                    "selected_checkpoint_eligible": True,
                    "model_family": "egostitch_e2e",
                    "permanent_null": permanent_null,
                    "p_topo": p_topo,
                    "p_cont": p_cont,
                    "seed": 0,
                    "strategy": "toy",
                    "partition_seed": 0,
                    "config_path": str(arm_config_paths[name].resolve()),
                    "config_hash": te._config_hash(te.load_config(arm_config_paths[name])),
                    "config_sha256": hashlib.sha256(
                        arm_config_paths[name].read_bytes()
                    ).hexdigest(),
                    "checkpoint_sha256": hashlib.sha256(
                        f"checkpoint:{name}".encode()
                    ).hexdigest(),
                    "implementation_commit": implementation_commit,
                    "training_diagnostics": {
                        "fidelity_series": [{"topology_delta_std": 0.1}],
                        "gradient_norm_series": [
                            {
                                "step": 1,
                                "grad_rms_trunk": 0.1,
                                "grad_rms_ste": 0.1,
                                "grad_rms_content": 0.1,
                            }
                        ],
                        "kendall_fallback": {"active": False},
                    },
                }
            )
        )
        (arm_run_dir / "metrics.jsonl").write_text(
            "".join(
                json.dumps({"epoch": epoch, "auprc": 0.2, "fidelity": {"prevalence": 0.01}})
                + "\n"
                for epoch in (1, 2, 3)
            )
        )
        run_metadata_paths[name] = meta_path

    for name, artifact_path in arm_paths.items():
        source_arm = "full" if name in g5_stage1._E2E_CONTROL_ARMS else name
        metadata_path = run_metadata_paths[source_arm]
        metadata = json.loads(metadata_path.read_text())
        scoring_registration = preregistration["arms"][name]
        artifact = load_scores(artifact_path)
        artifact_meta = dict(artifact.meta)
        artifact_meta["formal_scoring_provenance"] = {
            "arm": source_arm,
            "arm_kind": scoring_registration["kind"],
            "checkpoint_arm": source_arm,
            "scoring_semantics": scoring_registration["scoring_provenance"],
            "registration_sha256": preregistration_sha256,
            "run_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "config_path": str(arm_config_paths[source_arm].resolve()),
            "config_sha256": metadata["config_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "implementation_commit": implementation_commit,
            "selected_checkpoint_eligible": True,
            "scoring_arm": name,
            "all_formal_arms": {
                arm: {
                    "run_metadata_sha256": hashlib.sha256(
                        run_metadata_paths[arm].read_bytes()
                    ).hexdigest(),
                    "checkpoint_sha256": json.loads(
                        run_metadata_paths[arm].read_text()
                    )["checkpoint_sha256"],
                    "config_sha256": preregistration["binding_evidence"]["configs"][arm][
                        "sha256"
                    ],
                    "selected_checkpoint_eligible": True,
                }
                for arm in g5_stage1._E2E_FORMAL_ARMS
            },
        }
        _rewrite_e2e_artifact(artifact_path, meta=artifact_meta)

    return {
        "arm_universe_paths": arm_paths,
        "run_metadata_paths": run_metadata_paths,
        "preregistration_path": preregistration_path,
        "data_root": data_root,
        "strategy": "toy",
    }


def _probe_v2_report(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Write and evaluate a real probe-v2 artifact for gate-renderer tests."""
    graph = nx.cycle_graph(12)
    graph = nx.relabel_nodes(graph, {node: f"n{node:02d}" for node in graph})
    nodes = sorted(graph.nodes())
    targets = probes.probe_targets(graph, nodes)
    pairs = probes.select_probe_pairs(graph)
    n_pairs = len(pairs)
    rng = np.random.default_rng(17)
    metadata: dict[str, object] = {
        "checkpoint_id": "probe-checkpoint",
        "registration_sha256": "a" * 64,
        "config_hash": "b" * 64,
        "seed": 0,
        "partition_seed": 0,
        "strategy": "toy",
        "g_struct_sha256": probes.g_struct_sha256(graph),
        "scope": "formal_train",
        "n_ground": 50,
    }
    path = tmp_path / "probe-v2.npz"
    probes.write_e2e_probe_artifact(
        path,
        metadata=metadata,
        node_ids=nodes,
        states=rng.normal(size=(len(nodes), 4)).astype(np.float32),
        targets={name: targets[name] for name in ("degree", "ego_density", "clustering")},
        pair_ids=pairs,
        pair_states=rng.normal(size=(n_pairs, 4)).astype(np.float32),
        pi_consistency_v1=np.linspace(0.0, 0.5, n_pairs, dtype=np.float64),
        pi_consistency_v2=np.linspace(0.2, 0.8, n_pairs, dtype=np.float64),
        slot_recall=np.linspace(0.05, 0.25, len(nodes), dtype=np.float64),
        shared_neighbor_count=np.asarray(
            [
                len(set(graph.neighbors(node_u)) & set(graph.neighbors(node_v)))
                for node_u, node_v in pairs
            ],
            dtype=np.float64,
        ),
        dispersion={
            "pi_slot_std": np.full(n_pairs, 0.11),
            "h_pairwise_cosine_mean": np.full(n_pairs, 0.22),
            "adj_offdiag_std": np.full(n_pairs, 0.33),
            "plan_row_entropy": np.full(n_pairs, 0.44),
        },
    )
    report = probes.evaluate_e2e_probe_artifact(
        path,
        graph=graph,
        train_nodes=nodes,
        expected_metadata=metadata,
    )
    return path, report


def _markdown_table(markdown: str, heading: str) -> dict[str, dict[str, str]]:
    """Parse one rendered Markdown table, keyed by its first column."""
    lines = markdown.splitlines()
    heading_index = lines.index(heading)
    header_index = next(
        index for index in range(heading_index + 1, len(lines)) if lines[index].startswith("|")
    )
    headers = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = dict(zip(headers, cells, strict=True))
    return rows


class TestBuildE2EArmSummary:
    def test_active_v4_binding_evidence_v2_is_accepted(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        registration_path = cast(Path, inputs["preregistration_path"])
        registration = json.loads(registration_path.read_text())

        g5_stage1._validate_e2e_binding_evidence(registration, registration_path)

    def test_requires_complete_binding_evidence(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        registration_path = _d(inputs)["preregistration_path"]
        registration = json.loads(registration_path.read_text())
        registration.pop("binding_evidence", None)
        registration_path.write_text(json.dumps(registration, sort_keys=True, indent=2) + "\n")
        registration_sha256 = hashlib.sha256(registration_path.read_bytes()).hexdigest()
        for metadata_path in _d(inputs["run_metadata_paths"]).values():
            metadata = json.loads(metadata_path.read_text())
            metadata["preregistration_sha256"] = registration_sha256
            metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(g5_stage1.PreregistrationMismatch, match="binding_evidence"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_scorer_emitted_formal_provenance(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        full_path = _d(inputs["arm_universe_paths"])["full"]
        artifact = load_scores(full_path)
        meta = dict(artifact.meta)
        meta.pop("formal_scoring_provenance", None)
        _rewrite_e2e_artifact(full_path, meta=meta)

        with pytest.raises(ValueError, match="formal_scoring_provenance"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_formal_artifact_missing_test_access_ledger(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        full_path = _d(inputs["arm_universe_paths"])["full"]
        artifact = load_scores(full_path)
        meta = dict(artifact.meta)
        meta.pop("test_access_ledger")
        _rewrite_e2e_artifact(full_path, meta=meta)

        with pytest.raises(ValueError, match="missing test_access_ledger"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_tampered_test_access_ledger(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        ledger_path = tmp_path / "test_access_ledger.jsonl"
        lines = ledger_path.read_text().splitlines()
        first = json.loads(lines[0])
        first["seed"] = 7
        lines[0] = json.dumps(first, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")

        with pytest.raises(ValueError, match="digest mismatch"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_all_eight_arms_reported_and_rerun_is_deterministic(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        payload = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

        arms = _d(payload["arms"])
        assert set(arms) == set(g5_stage1._E2E_ARMS)
        expected_sha = hashlib.sha256(_d(inputs)["preregistration_path"].read_bytes()).hexdigest()
        assert payload["registration_sha256"] == expected_sha
        for name in g5_stage1._E2E_ARMS:
            row = _d(arms[name])
            expected_checkpoint = (
                "ckpt_full" if name in g5_stage1._E2E_CONTROL_ARMS else f"ckpt_{name}"
            )
            assert row["checkpoint_id"] == expected_checkpoint
            assert "graph_similarity" in _d(row["assembled"])
        # Scoring controls have no run metadata, but inherit the full checkpoint.
        assert _d(arms["structure_control_6a_v3"])["registration_sha256"] == expected_sha
        assert _d(arms["structure_control_6e_v1"])["registration_sha256"] == expected_sha
        assert _d(arms["full"])["registration_sha256"] == expected_sha
        disclosure = _d(payload["v_hold_evaluation_disclosure"])
        trained = _d(disclosure["trained_arms"])
        assert set(trained) == set(g5_stage1._E2E_FORMAL_ARMS)
        for name in g5_stage1._E2E_FORMAL_ARMS:
            assert _d(trained[name])["k_cumulative"] == 5
            assert "qualification" not in _d(trained[name])
            assert _d(_d(trained[name])["formal"])["count"] == 5
        controls = _d(disclosure["scoring_time_controls"])
        assert set(controls) == set(g5_stage1._E2E_CONTROL_ARMS)
        assert all(_d(row)["k_cumulative"] is None for row in controls.values())
        liveness = _d(payload["liveness"])
        assert liveness["residual_std"] > 0
        structure_control = _d(payload["structure_control"])
        assert structure_control["n_boot"] == 1000
        assert structure_control["seed"] == 0
        assert structure_control["passed"] is (structure_control["lower_bound"] > 0.0)
        decomposition = _d(payload["decomposition"])
        assert set(_d(decomposition["arms"])) == set(g5_stage1._E2E_ARMS)
        full_deltas = _d(_d(_d(decomposition["arms"])["full"])["deltas"])
        assert set(full_deltas) == {
            "full_minus_f_logit",
            "topology_delta_full_minus_pair_content",
            "content_delta_full_minus_pair_topology",
        }
        rerun = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
        assert json.dumps(payload, sort_keys=True) == json.dumps(rerun, sort_keys=True)

    def test_vhold_disclosure_requires_adjacent_formal_event_ledger(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["full"]
        metadata_path.with_name(te.V_HOLD_VALIDATION_EVENTS_FILENAME).unlink()

        with pytest.raises(ValueError, match="validation-event evidence is unreadable"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_unknown_arm(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"]["bogus"] = inputs["arm_universe_paths"]["full"]
        with pytest.raises(ValueError, match="unrecognized"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_v2_five_arm_package(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        arm_paths = _d(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"] = {
            "full": arm_paths["full"],
            "b0_e2e_f_only": arm_paths["b0_e2e_f_only"],
            "pair_topology": arm_paths["pair_topology"],
            "structure_control_6a": arm_paths["structure_control_6a_v3"],
            "p0": arm_paths["p0"],
        }
        with pytest.raises(ValueError, match="unrecognized"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_full_arm(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = {
            k: v for k, v in inputs["arm_universe_paths"].items() if k != "full"
        }
        with pytest.raises(ValueError, match="'full' arm is required"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_exact_eight_arm_set(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"].pop("p0")
        with pytest.raises(ValueError, match="exactly the eight registered arms"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_all_six_trained_run_metadata_records(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        inputs["run_metadata_paths"] = dict(inputs["run_metadata_paths"])
        inputs["run_metadata_paths"].pop("p0")
        with pytest.raises(ValueError, match="exactly the six trained"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    @pytest.mark.parametrize("registration", [None, ""])
    def test_requires_nonempty_shared_registration_hash(
        self, tmp_path: Path, registration: str | None
    ) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["p0"]
        payload = json.loads(metadata_path.read_text())
        payload.pop("preregistration_sha256")
        if registration is not None:
            payload["preregistration_sha256"] = registration
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="non-empty preregistration_sha256"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_formal_artifact_checkpoint_mismatch(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["pair_topology"]
        payload = json.loads(metadata_path.read_text())
        payload["checkpoint_id"] = "wrong"
        metadata_path.write_text(json.dumps(payload))
        _refresh_formal_run_metadata_hashes(inputs)
        with pytest.raises(ValueError, match="checkpoint_id mismatch"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_6a_to_use_full_scoring_checkpoint(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        rng = np.random.default_rng(123)
        _write_e2e_universe_npz(
            _d(inputs["arm_universe_paths"])["structure_control_6a_v3"],
            node_ids=_NODES,
            pairs=pairs,
            full=rng.normal(size=len(pairs)),
            f_logit=rng.normal(size=len(pairs)),
            pair_content=rng.normal(size=len(pairs)),
            pair_topology=rng.normal(size=len(pairs)),
            labels=labels,
            checkpoint_id="ckpt_not_full",
            scaffold_control="shuffle_within_pair_v3",
            scoring_arm="structure_control_6a_v3",
            arm_kind="scoring_time_control",
            checkpoint_arm="full",
            scoring_semantics={
                "scaffold_control": "shuffle_within_pair_v3",
                "seed": 0,
                "keying": "canonical_pair_v1",
                "permanent_null": "none",
                "primary_logit": "full",
                "checkpoint_arm": "full",
            },
        )
        with pytest.raises(ValueError, match="structure_control_6a_v3 checkpoint_id"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_registration_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        bad_meta = tmp_path / "bad_meta.json"
        payload = json.loads(_d(inputs)["run_metadata_paths"]["p0"].read_text())
        payload["preregistration_sha256"] = "b" * 64
        bad_meta.write_text(json.dumps(payload))
        inputs["run_metadata_paths"] = dict(inputs["run_metadata_paths"])
        inputs["run_metadata_paths"]["p0"] = bad_meta
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="does not match"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_binding_registration_status(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        preregistration_path.write_text(json.dumps({"status": "DRAFT"}))

        with pytest.raises(g5_stage1.PreregistrationNotBinding, match="status == 'BINDING'"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_debug_run_metadata(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(metadata_path.read_text())
        metadata.update({"run_kind": "debug", "formal_artifacts_published": False})
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="debug/non-formal"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_completed_formal_run_metadata(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("formal_artifacts_published")
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="exactly true"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_one_full_run_masquerading_as_all_formal_arms(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        full = _d(inputs)["run_metadata_paths"]["full"]
        inputs["run_metadata_paths"] = dict.fromkeys(g5_stage1._E2E_FORMAL_ARMS, full)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="distinct run metadata"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_wrong_registered_arm_semantics(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["p0"]
        metadata = json.loads(path.read_text())
        metadata["p_topo"] = 0.15
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="branch dropout"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            # `seed` is now bound to the registration's declared `seeds` list
            # rather than hard-pinned to 0 (multi-seed screens are permitted,
            # still at `evidence_class: engineering`); the toy registration
            # declares `[0]`, so 1 is still refused -- by the registered-list
            # membership test, not by a literal zero pin.
            ("seed", 1, r"training seed 1 is not one of the registered seeds"),
            ("strategy", "alternate", "strategy does not match"),
            # `partition_seed` is NOT relaxed: it selects G_struct and the
            # whole pair universe, so it stays pinned at 0.
            ("partition_seed", 1, "partition_seed must be 0"),
            ("config_hash", "0" * 64, "config_hash does not match"),
        ],
    )
    def test_rejects_wrong_registered_run_identity(
        self, tmp_path: Path, field: str, value: object, match: str
    ) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(path.read_text())
        metadata[field] = value
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match=match):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_wrong_registered_config_path(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs)["run_metadata_paths"]["full"]
        metadata = json.loads(path.read_text())
        metadata["config_path"] = json.loads(_d(inputs)["run_metadata_paths"]["p0"].read_text())[
            "config_path"
        ]
        path.write_text(json.dumps(metadata))
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="config_path"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_captured_registration_snapshot_is_not_reopened(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        snapshot = g5_stage1._preregistration_snapshot(preregistration_path)
        preregistration_path.write_text(json.dumps({"status": "DRAFT"}))
        payload = g5_stage1.build_e2e_arm_summary(
            liveness_config=_E2E_LIVENESS_CONFIG,
            preregistration_snapshot=snapshot,
            **inputs,
        )
        assert payload["registration_sha256"] == snapshot[1]

    @pytest.mark.parametrize(
        ("arm", "field", "value", "match"),
        [
            ("full", "permanent_null", "all_head", "permanent_null"),
            ("p0", "primary_logit", "pair_topology", "primary_logit"),
            ("structure_control_6a_v3", "control_seed", 1, "scaffold_control"),
            ("structure_control_6a_v3", "control_keying", "wrong", "scaffold_control"),
            ("full", "control_mode", "shuffle_within_pair_v3", "scaffold_control"),
        ],
    )
    def test_rejects_each_mutated_score_provenance_field(
        self,
        tmp_path: Path,
        arm: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])[arm]
        artifact = load_scores(path)
        meta = dict(artifact.meta)
        if field.startswith("control_"):
            control = dict(cast(dict[str, object], meta["scaffold_control"]))
            control[field.removeprefix("control_")] = value
            meta["scaffold_control"] = control
        else:
            meta[field] = value
        _rewrite_e2e_artifact(path, meta=meta)
        with pytest.raises((ValueError, g5_stage1.RegistrationShaMismatch), match=match):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_pair_order_swap(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        pairs = list(artifact.pairs())
        pairs[0], pairs[1] = pairs[1], pairs[0]
        labels = artifact.label.copy()
        labels[[0, 1]] = labels[[1, 0]]
        _rewrite_e2e_artifact(path, pairs=pairs, labels=labels)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="pair identity/order"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_label_mutation(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        labels = artifact.label.copy()
        labels[0] = 1 - labels[0]
        _rewrite_e2e_artifact(path, labels=labels)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="candidate labels"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_binding_registration_rejects_b0cal_marker(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        registration = json.loads(preregistration_path.read_text())
        registration["frozen_inputs"]["b0cal_results"]["sha256"] = "REQUIRED-BEFORE-BINDING"
        preregistration_path.write_text(json.dumps(registration))
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="real b0cal_results"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_missing_submodule_rms_telemetry_does_not_gate_evaluation(
        self, tmp_path: Path
    ) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["full"]
        metadata = json.loads(metadata_path.read_text())
        del metadata["training_diagnostics"]["gradient_norm_series"][0]["grad_rms_content"]
        metadata_path.write_text(json.dumps(metadata))
        _refresh_formal_run_metadata_hashes(inputs)
        g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_nested_worker_submodule_rms_shape_is_accepted(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        for metadata_path in _d(inputs["run_metadata_paths"]).values():
            metadata = json.loads(metadata_path.read_text())
            row = metadata["training_diagnostics"]["gradient_norm_series"][0]
            nested = {
                key: row.pop(key) for key in ("grad_rms_trunk", "grad_rms_ste", "grad_rms_content")
            }
            row["submodule_gradient_rms"] = nested
            metadata_path.write_text(json.dumps(metadata))
        _refresh_formal_run_metadata_hashes(inputs)
        g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_formal_evaluator_seed_must_be_zero(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        preregistration = json.loads(_d(inputs)["preregistration_path"].read_text())
        g5_stage1._enforce_e2e_evaluator_seed(preregistration, 0)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="seed 0"):
            g5_stage1._enforce_e2e_evaluator_seed(preregistration, 1)

    def test_b0cal_comparator_path_and_digest_are_exact(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        preregistration_path = _d(inputs)["preregistration_path"]
        preregistration = json.loads(preregistration_path.read_text())
        frozen = preregistration["frozen_inputs"]
        b0_path = Path(frozen["b0_candidate_scores"]["path"])
        b0cal_path = Path(frozen["b0cal_results"]["path"])
        assert (
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, b0cal_path
            )
            == b0cal_path.resolve()
        )

        copy = tmp_path / "copied_b0cal_results.json"
        copy.write_bytes(b0cal_path.read_bytes())
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="path mismatch"):
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, copy
            )
        b0cal_path.write_text('{"mutated": true}\n')
        with pytest.raises(g5_stage1.PreregistrationMismatch, match="sha256 mismatch"):
            g5_stage1.enforce_e2e_frozen_inputs(
                preregistration, preregistration_path, b0_path, b0cal_path
            )


def test_formal_gate_reports_rev31_telemetry_without_changing_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real gate writer must headline all nonbinding rev-3.1 telemetry."""
    inputs = _eight_arm_inputs(tmp_path)
    preregistration_path = cast(Path, inputs["preregistration_path"])
    registration = json.loads(preregistration_path.read_text())
    frozen = _d(registration["frozen_inputs"])
    b0_universe_path = Path(cast(str, _d(frozen["b0_candidate_scores"])["path"]))
    b0cal_output_dir = tmp_path / "b0cal"
    b0_cal.run_b0_cal_pipeline(
        universe_path=b0_universe_path,
        val_scores_path=tmp_path / "val.npz",
        data_root=cast(Path, inputs["data_root"]),
        strategy="toy",
        output_dir=b0cal_output_dir,
        seed=0,
        skip_perturbation_check=True,
    )
    b0cal_results_path = b0cal_output_dir / "b0cal_results.json"
    _d(frozen["b0cal_results"])["sha256"] = hashlib.sha256(
        b0cal_results_path.read_bytes()
    ).hexdigest()
    preregistration_path.write_text(json.dumps(registration, sort_keys=True, indent=2) + "\n")

    expected_correlations: dict[str, float] = {}
    for index, (arm, metadata_path) in enumerate(
        _d(inputs["run_metadata_paths"]).items(), start=1
    ):
        metadata = json.loads(metadata_path.read_text())
        correlation = index / 10
        metadata["training_diagnostics"]["fidelity_series"][0][
            "topology_delta_degree_correlation"
        ] = correlation
        metadata_path.write_text(json.dumps(metadata))
        expected_correlations[arm] = correlation
    _refresh_formal_run_metadata_hashes(inputs)

    summary = g5_stage1.build_e2e_arm_summary(
        liveness_config=_E2E_LIVENESS_CONFIG,
        **inputs,
    )
    probe_path, probe_report = _probe_v2_report(tmp_path)
    current_summary = summary

    def _summary(**_kwargs: object) -> dict[str, object]:
        return current_summary

    monkeypatch.setattr(g5_stage1, "build_e2e_arm_summary", _summary)
    monkeypatch.setattr(
        g5_stage1,
        "_evaluate_registered_e2e_probe",
        lambda **_kwargs: probe_report,
    )
    gate_registration = cast(dict[str, object], json.loads(json.dumps(registration)))
    gate_registration["benchmark"] = {"strategy": "breadth_first"}
    monkeypatch.setattr(
        g5_stage1,
        "_preregistration_snapshot",
        lambda _path: (
            gate_registration,
            hashlib.sha256(preregistration_path.read_bytes()).hexdigest(),
        ),
    )
    b0_row = g5_stage1.AssembledRow(
        threshold=0.5,
        mmd_ratio={"degree": 1.0, "clustering": 1.0, "spectral": 1.0},
        raw_mmd2={"degree": 1.0, "clustering": 1.0, "spectral": 1.0},
        reference_mmd2={"degree": 1.0, "clustering": 1.0, "spectral": 1.0},
        graph_similarity=0.5,
        relative_density=0.5,
        per_size_graph_similarity={},
        per_size_relative_density={},
        self_loops_pred=0,
        self_loops_ref=0,
        bootstrap_mean={"degree": 1.0, "clustering": 1.0, "spectral": 1.0},
        bootstrap_std={"degree": 0.0, "clustering": 0.0, "spectral": 0.0},
    )
    monkeypatch.setattr(g5_stage1, "assemble_and_evaluate", lambda **_kwargs: b0_row)
    monkeypatch.setattr(g5_stage1, "_validate_b0cal_lineage", lambda *_args: None)
    monkeypatch.setattr(
        g5_stage1,
        "evaluate_assembled_graph",
        lambda *_args, **_kwargs: SimpleNamespace(
            graph_similarity=0.5,
            relative_density=0.5,
        ),
    )
    monkeypatch.setattr(
        g5_stage1,
        "evaluate_regime_table",
        lambda **_kwargs: {
            "degree_corrected": {"ratio_1": SimpleNamespace(auprc=0.5)}
        },
    )

    def run_gate(output_dir: Path) -> dict[str, object]:
        return g5_stage1.run_g5_e2e_stage1_pipeline(
            arm_universe_paths=_d(inputs["arm_universe_paths"]),
            run_metadata_paths=_d(inputs["run_metadata_paths"]),
            b0_universe_path=b0_universe_path,
            b0cal_results_path=b0cal_results_path,
            probe_artifact_path=probe_path,
            preregistration_path=preregistration_path,
            data_root=cast(Path, inputs["data_root"]),
            strategy="toy",
            output_dir=output_dir,
        )

    reported_dir = tmp_path / "reported-gate"
    reported = run_gate(reported_dir)
    reported_json = json.loads(
        (reported_dir / "g5_e2e_stage1_results.json").read_text()
    )
    reported_markdown = (reported_dir / "g5_e2e_stage1_tables.md").read_text()

    diagnostics = _d(reported_json["training_diagnostics"])
    for arm, expected in expected_correlations.items():
        fidelity = cast(list[dict[str, object]], _d(diagnostics[arm])["fidelity_series"])
        assert fidelity[0]["topology_delta_degree_correlation"] == expected

    probe_json = _d(reported_json["probes"])
    assert isinstance(probe_json["shared_neighbor_count_r2"], float)
    assert "pi_consistency_v2" in probe_json
    assert "slot_recall_at_n_ground" in probe_json
    assert set(_d(probe_json["dispersion"])) == {
        "pi_slot_std",
        "h_pairwise_cosine_mean",
        "adj_offdiag_std",
        "plan_row_entropy",
    }

    degree_rows = _markdown_table(reported_markdown, "## Degree-decorrelation telemetry")
    for arm, expected in expected_correlations.items():
        assert degree_rows[arm]["corr(full-f_logit, endpoint degree)"] == g5_stage1._fmt(
            expected
        )

    probe_rows = _markdown_table(reported_markdown, "## Registered probe-v2 evidence")
    expected_probe_rows = {
        "pi_consistency_v2",
        "slot_recall_at_n_ground",
        "shared_neighbor_count_r2",
        "pi_slot_std",
        "h_pairwise_cosine_mean",
        "adj_offdiag_std",
        "plan_row_entropy",
    }
    assert expected_probe_rows <= probe_rows.keys()
    assert probe_rows["pi_consistency_v2"]["value"] == g5_stage1._fmt(
        _d(probe_json["pi_consistency_v2"])["mean"]
    )
    assert probe_rows["slot_recall_at_n_ground"]["value"] == g5_stage1._fmt(
        _d(probe_json["slot_recall_at_n_ground"])["mean"]
    )
    assert probe_rows["shared_neighbor_count_r2"]["value"] == g5_stage1._fmt(
        probe_json["shared_neighbor_count_r2"]
    )
    for name, values in _d(probe_json["dispersion"]).items():
        assert probe_rows[name]["value"] == g5_stage1._fmt(_d(values)["mean"])

    current_summary = cast(dict[str, object], json.loads(json.dumps(summary)))
    for diagnostics_report in _d(current_summary["training_diagnostics"]).values():
        for row in cast(
            list[dict[str, object]], _d(diagnostics_report)["fidelity_series"]
        ):
            row.pop("topology_delta_degree_correlation", None)
    without_telemetry = run_gate(tmp_path / "gate-without-telemetry")
    assert without_telemetry["verdict"] == reported["verdict"]


class TestPairedBootstrap:
    def test_b0cal_directory_resolves_one_committed_payload(self, tmp_path: Path) -> None:
        directory = tmp_path / "b0cal"
        directory.mkdir()
        expected = directory / "b0cal_results.json"
        expected.write_text("{}")
        assert g5_stage1._resolve_b0cal_results_path(directory) == expected
        (directory / "nested").mkdir()
        (directory / "nested" / "b0cal_results.json").write_text("{}")
        with pytest.raises(ValueError, match="exactly one"):
            g5_stage1._resolve_b0cal_results_path(directory)

    def test_lower_bound_is_paired_and_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, 500)
        clearly_higher = base + 1.0
        mean_stat = cast(Callable[[object], float], np.mean)

        lower_bound = g5_stage1.paired_bootstrap_lower_bound(mean_stat, clearly_higher, base)

        assert lower_bound > 0.0
        assert lower_bound == g5_stage1.paired_bootstrap_lower_bound(
            mean_stat, clearly_higher, base
        )
        null_lower_bound = g5_stage1.paired_bootstrap_lower_bound(mean_stat, base + 0.001, base)
        assert null_lower_bound < 0.0 or abs(null_lower_bound) < 0.1


class TestE2EGateCli:
    def test_mode_e2e_forwards_exact_eight_arm_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        captured: dict[str, object] = {}
        b0_path = tmp_path / "b0.npz"
        b0cal_path = tmp_path / "b0cal.json"
        probe_path = tmp_path / "probe.npz"
        for path in (b0_path, b0cal_path, probe_path):
            path.write_bytes(b"placeholder")

        def _fake_run(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(g5_stage1, "run_g5_e2e_stage1_pipeline", _fake_run)
        g5_stage1.main(
            [
                "--mode",
                "e2e",
                "--full-universe",
                str(_d(inputs)["arm_universe_paths"]["full"]),
                "--control-6a-universe",
                str(_d(inputs)["arm_universe_paths"]["structure_control_6a_v3"]),
                "--control-6e-universe",
                str(_d(inputs)["arm_universe_paths"]["structure_control_6e_v1"]),
                "--fonly-universe",
                str(_d(inputs)["arm_universe_paths"]["b0_e2e_f_only"]),
                "--pt-universe",
                str(_d(inputs)["arm_universe_paths"]["pair_topology"]),
                "--p0-universe",
                str(_d(inputs)["arm_universe_paths"]["p0"]),
                "--cosine-pool-universe",
                str(_d(inputs)["arm_universe_paths"]["cosine_pool"]),
                "--no-l-rel-universe",
                str(_d(inputs)["arm_universe_paths"]["no_l_rel"]),
                "--run-metadata",
                *[str(path) for path in _d(inputs)["run_metadata_paths"].values()],
                "--b0-universe",
                str(b0_path),
                "--b0cal-results",
                str(b0cal_path),
                "--probe-artifact",
                str(probe_path),
                "--preregistration",
                str(_d(inputs)["preregistration_path"]),
                "--output-dir",
                str(tmp_path / "gate"),
            ]
        )
        assert set(_d(captured["arm_universe_paths"])) == set(g5_stage1._E2E_ARMS)
        assert set(_d(captured["run_metadata_paths"])) == set(g5_stage1._E2E_FORMAL_ARMS)
        assert captured["probe_artifact_path"] == probe_path

    def test_rejects_wrong_model_family(self, tmp_path: Path) -> None:
        inputs = _eight_arm_inputs(tmp_path)
        wrong_path = tmp_path / "wrong_family.npz"
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        _write_universe_npz(
            wrong_path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.zeros(len(pairs), dtype=np.float32),
            labels=labels,
            strategy="toy",
            model_family="v3_1",
        )
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"]["p0"] = wrong_path
        with pytest.raises(ValueError, match="model_family"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)


class TestEngineeringEvidenceClass:
    """The G5 E2E ladder never emits inference, at any seed count.

    CLAUDE.md is explicit that only E1/E3 carry inference, and protocol
    Sec 5.0.5 adds the 30-config HPO-parity budget and Holm over the
    pre-registered held-out family, neither of which a Stage-1-descended
    screen has (design 2026-07-29 Sec 2.5). `_enforce_engineering_evidence_class`
    runs before anything is written, so an inference-bearing artifact must not
    exist on disk even transiently.
    """

    @staticmethod
    def _payload(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "evidence_class": "engineering",
            "p_value": None,
            "ci": None,
            "holm": None,
            "metadata": {"training_seed": 0, "registered_seeds": [0]},
            "arms": {"full": {"graph_similarity": 0.5}},
            "verdict": "cut",
        }
        payload.update(overrides)
        return payload

    def test_a_clean_engineering_payload_is_accepted(self) -> None:
        g5_stage1._enforce_engineering_evidence_class(self._payload())

    @pytest.mark.parametrize("seeds", [[0], [0, 1, 2]])
    def test_evidence_class_is_engineering_at_every_seed_count(self, seeds: list[int]) -> None:
        """Extra seeds add cross-seed variance reporting, never significance."""
        payload = self._payload(metadata={"training_seed": seeds[0], "registered_seeds": seeds})
        g5_stage1._enforce_engineering_evidence_class(payload)
        assert payload["evidence_class"] == "engineering"
        assert payload["p_value"] is None
        assert payload["ci"] is None
        assert payload["holm"] is None

    @pytest.mark.parametrize("claimed", ["inference", "Engineering", None, ""])
    def test_refuses_any_class_other_than_engineering(self, claimed: object) -> None:
        with pytest.raises(g5_stage1.EvidenceClassViolation, match="evidence_class 'engineering'"):
            g5_stage1._enforce_engineering_evidence_class(self._payload(evidence_class=claimed))

    @pytest.mark.parametrize(
        "field",
        ["p_value", "p_values", "ci", "ci_excludes_zero", "holm", "holm_alpha", "holm_survives"],
    )
    def test_refuses_a_non_null_inferential_field_at_the_top_level(self, field: str) -> None:
        with pytest.raises(g5_stage1.EvidenceClassViolation, match=f"{field} must be null"):
            g5_stage1._enforce_engineering_evidence_class(self._payload(**{field: 0.04}))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("p_value", 0.04),
            ("ci", [0.01, 0.2]),
            ("holm", {"alpha": 0.05}),
            ("holm_survives", False),
            ("ci_excludes_zero", False),
        ],
    )
    def test_refuses_a_non_null_inferential_field_nested_anywhere(
        self, field: str, value: object
    ) -> None:
        """A `False` or an empty-looking value is still non-null and still refused.

        The walk is recursive precisely so a nested field cannot slip past a
        top-level check, and it tests `is not None` rather than truthiness --
        `holm_survives: false` is a significance claim with a negative answer.
        """
        buried = self._payload(arms={"full": {"comparators": [{"vs_f_only": {field: value}}]}})
        with pytest.raises(
            g5_stage1.EvidenceClassViolation,
            match=rf"arms\.full\.comparators\[0\]\.vs_f_only\.{field} must be null",
        ):
            g5_stage1._enforce_engineering_evidence_class(buried)

    def test_the_structure_control_bootstrap_bound_is_not_an_inferential_field(self) -> None:
        """`lower_bound`/`alpha` are the paired-bootstrap readout, not a p-value.

        They are deliberately outside `_INFERENCE_FIELDS`: protocol Sec E4.16(e)
        makes the 6e-v1 bound decisive for the screen, and refusing it would
        strip a required control rather than a significance claim.
        """
        assert "lower_bound" not in g5_stage1._INFERENCE_FIELDS
        assert "alpha" not in g5_stage1._INFERENCE_FIELDS
        g5_stage1._enforce_engineering_evidence_class(
            self._payload(structure_control={"lower_bound": 0.0, "alpha": 0.05})
        )


class TestRegisteredSeedsAndRunKindWhitelist:
    """Multi-seed relaxation and the publish whitelist that guards it."""

    @staticmethod
    def _metadata(tmp_path: Path, name: str, **overrides: object) -> Path:
        payload: dict[str, object] = {
            "preregistration_sha256": "f" * 64,
            "run_kind": "formal",
            "formal_artifacts_published": True,
        }
        payload.update(overrides)
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @pytest.mark.parametrize("run_kind", ["qualification", "debug", None, "formal_"])
    def test_only_formal_metadata_may_publish_held_out_metrics(
        self, tmp_path: Path, run_kind: object
    ) -> None:
        """A whitelist, not a `!= "debug"` blacklist.

        The pre-cleanup form rejected only `run_kind == "debug"`, so the new
        `qualification` kind would have published held-out metrics without
        anyone editing the guard. `formal_artifacts_published` is set true here
        on purpose: the run kind alone must be decisive.
        """
        path = self._metadata(tmp_path, "meta.json", run_kind=run_kind)
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="debug/non-formal"):
            g5_stage1._enforce_metadata_registration_hash(
                tmp_path / "prereg.json", [path], "f" * 64
            )

    def test_formal_metadata_passes_the_same_guard(self, tmp_path: Path) -> None:
        path = self._metadata(tmp_path, "meta.json")
        assert (
            g5_stage1._enforce_metadata_registration_hash(
                tmp_path / "prereg.json", [path], "f" * 64
            )
            == "f" * 64
        )

    @pytest.mark.parametrize("seeds", [[0], [0, 1, 2], [7]])
    def test_registered_seeds_are_read_verbatim(self, seeds: list[int]) -> None:
        assert g5_stage1._registered_training_seeds({"seeds": seeds}) == tuple(seeds)

    @pytest.mark.parametrize(
        "seeds",
        [None, [], "0", [0, 0], [0, "1"], [0, 1.0], [True], {"0": 0}],
    )
    def test_a_missing_or_malformed_seed_list_fails_closed(self, seeds: object) -> None:
        """No silent fallback to `[0]`: the registration must declare the axis."""
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="non-empty list of distinct"):
            g5_stage1._registered_training_seeds({"seeds": seeds})

    def test_a_run_seed_must_be_one_of_the_registered_seeds(self) -> None:
        bound = g5_stage1._enforce_registered_training_seed({"seed": 2}, (0, 1, 2), label="full")
        assert bound == 2
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="not one of the registered"):
            g5_stage1._enforce_registered_training_seed({"seed": 3}, (0, 1, 2), label="full")
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="not one of the registered"):
            g5_stage1._enforce_registered_training_seed({}, (0,), label="full")
        # `True == 1` in Python; a bool is not a seed.
        with pytest.raises(g5_stage1.RegistrationShaMismatch, match="not one of the registered"):
            g5_stage1._enforce_registered_training_seed({"seed": True}, (0, 1), label="full")
