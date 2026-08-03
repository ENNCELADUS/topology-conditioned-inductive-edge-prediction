"""E2E liveness and seven-arm gate contracts."""

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
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
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


# --------------------------------------------------------------------------- e2e family:
# within-checkpoint liveness + seven-arm summary (Task 11)


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
    labels: np.ndarray,
    strategy: str = "toy",
    checkpoint_id: str = "e2e0",
    scaffold_control: str = "none",
    permanent_null: str = "none",
    primary_logit: str = "full",
    scoring_arm: str = "full",
) -> None:
    """Write a toy family-``egostitch_e2e`` two-array scores artifact."""
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
        "test_access_ledger": access.ledger_binding,
    }
    arrays = {
        "full": full,
        "f_logit": f_logit,
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
        full_logit=artifact.full_logit,
    )
    if meta is not None:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez_compressed(path, **arrays)


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

    # Three-component refactor (design 2026-08-02 Sec 8): `n_ground` moved
    # from the flat `_E2E_TINY_MODEL` onto its nested `generator` sub-config.
    model_config = {
        **_E2E_TINY_MODEL,
        "generator": {
            **cast(dict[str, object], _E2E_TINY_MODEL["generator"]),
            "n_ground": 3,
        },
    }
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

    model = EgoStitchModel(E2EConfig.from_mapping(model_config))
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
                "seed": 0,
                "partition_seed": 0,
                "config_path": str(config_path),
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
        data_root=data_root,
        strategy="toy",
        output_path=probe_path,
        scope="formal_train",
    )
    with np.load(probe_path, allow_pickle=False) as archive:
        produced_metadata = json.loads(str(archive["meta"].item()))
    assert produced_metadata["format"] == "egostitch_e2e_probe_v2"

    report = g5_stage1._evaluate_e2e_probe(
        probe_artifact_path=probe_path,
        run_metadata_path=run_metadata_path,
        data_root=data_root,
        strategy="toy",
    )
    assert cast(dict[str, object], report["metadata"])["format"] == (
        "egostitch_e2e_probe_v2"
    )


def _validation_events(name: str, epochs: int) -> list[dict[str, object]]:
    """One ``step_0`` row plus ``phase_a_end``/``epoch_end`` rows, in ordinal order."""
    raw = [("step_0", 0), ("phase_a_end", 1)]
    raw.extend(("epoch_end", epoch) for epoch in range(1, epochs + 1))
    return [
        {
            "ordinal": ordinal,
            "kind": kind,
            "optimizer_step": step,
            "arm": name,
            "validation_role": "V_hold",
        }
        for ordinal, (kind, step) in enumerate(raw, start=1)
    ]


def _write_trained_arm_metadata(
    tmp_path: Path, name: str, *, checkpoint_id: str
) -> Path:
    """Write one trained arm's minimal ``run_metadata.json`` plus its V_hold ledger."""
    arm_dir = tmp_path / "runs" / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    events_path = arm_dir / te.V_HOLD_VALIDATION_EVENTS_FILENAME
    rows = _validation_events(name, epochs=3)
    events_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    meta_path = arm_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "arm": name,
                "run_kind": "formal",
                "seed": 0,
                "strategy": "toy",
                "partition_seed": 0,
                "config_path": str(tmp_path / f"{name}.yaml"),
                "v_hold_validation_evidence": {
                    "count": len(rows),
                    "path": events_path.name,
                    "sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                },
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
                },
            },
            sort_keys=True,
        )
    )
    return meta_path


def _seven_arm_inputs(tmp_path: Path) -> dict[str, Any]:
    """Toy benchmark + seven E2E arm universes + five trained-run metadata records."""
    _b0_universe_path, _val_path, data_root = _b0cal_toy_inputs(tmp_path)
    pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
    rng = np.random.default_rng(42)
    full = rng.normal(size=len(pairs))
    f_logit = rng.normal(size=len(pairs))  # independent of `full` -> alive residual
    candidate_path = data_root / "benchmark_2025_neurips" / "toy" / "candidate_test_edges.txt"
    candidate_path.write_text(
        "".join(
            f"{node_u}\t{node_v}\t{int(label)}\n"
            for (node_u, node_v), label in zip(pairs, labels, strict=True)
        )
    )

    provenance_by_arm = {
        "full": ("none", "none", "full"),
        "b0_e2e_f_only": ("none", "all_head", "f_logit"),
        "p0": ("none", "none", "full"),
        "no_l_rel": ("none", "none", "full"),
        "row_layernorm": ("none", "none", "full"),
        "structure_control_6a_v3": ("shuffle_within_pair_v3", "none", "full"),
        "structure_control_6e_v1": ("rewire_checkerboard_v1", "none", "full"),
    }
    arm_paths: dict[str, Path] = {}
    for name in g5_stage1._ALL_ARMS:
        scaffold_control, permanent_null, primary_logit = provenance_by_arm[name]
        checkpoint_id = "ckpt_full" if name in g5_stage1._CONTROL_ARMS else f"ckpt_{name}"
        path = tmp_path / f"{name}.npz"
        _write_e2e_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            full=full,
            f_logit=f_logit,
            labels=labels,
            checkpoint_id=checkpoint_id,
            scaffold_control=scaffold_control,
            permanent_null=permanent_null,
            primary_logit=primary_logit,
            scoring_arm=name,
        )
        arm_paths[name] = path

    run_metadata_paths: dict[str, Path] = {
        name: _write_trained_arm_metadata(tmp_path, name, checkpoint_id=f"ckpt_{name}")
        for name in g5_stage1._TRAINED_ARMS
    }

    return {
        "arm_universe_paths": arm_paths,
        "run_metadata_paths": run_metadata_paths,
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
    def test_all_seven_arms_reported_and_rerun_is_deterministic(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        payload = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

        arms = _d(payload["arms"])
        assert set(arms) == set(g5_stage1._ALL_ARMS)
        for name in g5_stage1._ALL_ARMS:
            row = _d(arms[name])
            expected_checkpoint = (
                "ckpt_full" if name in g5_stage1._CONTROL_ARMS else f"ckpt_{name}"
            )
            assert row["checkpoint_id"] == expected_checkpoint
            assert "graph_similarity" in _d(row["assembled"])
        liveness = _d(payload["liveness"])
        assert liveness["residual_std"] > 0
        structure_control = _d(payload["structure_control"])
        assert structure_control["n_boot"] == 1000
        assert structure_control["seed"] == 0
        assert structure_control["passed"] is (structure_control["lower_bound"] > 0.0)
        decomposition = _d(payload["decomposition"])
        assert set(_d(decomposition["arms"])) == set(g5_stage1._ALL_ARMS)
        full_deltas = _d(_d(_d(decomposition["arms"])["full"])["deltas"])
        assert set(full_deltas) == {"full_minus_f_logit"}
        rerun = g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)
        assert json.dumps(payload, sort_keys=True) == json.dumps(rerun, sort_keys=True)

    def test_rejects_unknown_arm(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"]["bogus"] = inputs["arm_universe_paths"]["full"]
        with pytest.raises(ValueError, match="unrecognized"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_full_arm(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = {
            k: v for k, v in inputs["arm_universe_paths"].items() if k != "full"
        }
        with pytest.raises(ValueError, match="'full' arm is required"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_exact_seven_arm_set(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        inputs["arm_universe_paths"] = dict(inputs["arm_universe_paths"])
        inputs["arm_universe_paths"].pop("p0")
        with pytest.raises(ValueError, match="exactly the seven registered arms"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_pair_order_swap(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        pairs = list(artifact.pairs())
        pairs[0], pairs[1] = pairs[1], pairs[0]
        labels = artifact.label.copy()
        labels[[0, 1]] = labels[[1, 0]]
        _rewrite_e2e_artifact(path, pairs=pairs, labels=labels)
        with pytest.raises(ValueError, match="pair identity/order"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_candidate_label_mutation(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        path = _d(inputs["arm_universe_paths"])["p0"]
        artifact = load_scores(path)
        labels = artifact.label.copy()
        labels[0] = 1 - labels[0]
        _rewrite_e2e_artifact(path, labels=labels)
        with pytest.raises(ValueError, match="candidate labels"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_requires_6a_to_use_full_scoring_checkpoint(self, tmp_path: Path) -> None:
        """The surviving checkpoint_id check: a control must reuse full's checkpoint.

        Restores the regression test deleted in error during the P0
        registration excision (adversarial-review finding 5): the check
        itself (`build_e2e_arm_summary`'s ``{name} checkpoint_id mismatch``
        raise) survived the excision, but its only test did not. Rebuilt
        here against the current post-excision fixtures rather than the
        deleted registration-based ones.
        """
        inputs = _seven_arm_inputs(tmp_path)
        pairs, labels = _universe_rows(_NODES, _POSITIVE_EDGES)
        rng = np.random.default_rng(123)
        _write_e2e_universe_npz(
            _d(inputs["arm_universe_paths"])["structure_control_6a_v3"],
            node_ids=_NODES,
            pairs=pairs,
            full=rng.normal(size=len(pairs)),
            f_logit=rng.normal(size=len(pairs)),
            labels=labels,
            checkpoint_id="ckpt_not_full",
            scaffold_control="shuffle_within_pair_v3",
            scoring_arm="structure_control_6a_v3",
        )
        with pytest.raises(ValueError, match="structure_control_6a_v3 checkpoint_id"):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_swapped_scaffold_control_arms(self, tmp_path: Path) -> None:
        """Swapping the two control NPZ paths must be caught by arm-semantic validation.

        Both controls share `full`'s checkpoint_id, so the checkpoint_id
        check cannot tell them apart. `_validate_scaffold_control_arm_semantics`
        reads each artifact's own `meta["scaffold_control"]["mode"]` and
        requires it to match the arm slot it was supplied under
        (adversarial-review finding 3).
        """
        inputs = _seven_arm_inputs(tmp_path)
        paths = _d(inputs["arm_universe_paths"])
        paths["structure_control_6a_v3"], paths["structure_control_6e_v1"] = (
            paths["structure_control_6e_v1"],
            paths["structure_control_6a_v3"],
        )
        with pytest.raises(
            ValueError, match="structure_control_6a_v3.*requires scaffold_control.mode"
        ):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_rejects_unperturbed_full_artifact_passed_as_control(self, tmp_path: Path) -> None:
        """An unperturbed `full` artifact substituted for a control must be rejected.

        This artifact shares `full`'s checkpoint_id trivially (it *is*
        full's artifact), so only the scaffold-control arm-semantic check
        catches it (adversarial-review finding 3).
        """
        inputs = _seven_arm_inputs(tmp_path)
        paths = _d(inputs["arm_universe_paths"])
        paths["structure_control_6a_v3"] = paths["full"]
        with pytest.raises(
            ValueError, match="structure_control_6a_v3.*requires scaffold_control.mode"
        ):
            g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_missing_submodule_rms_telemetry_does_not_gate_evaluation(
        self, tmp_path: Path
    ) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        metadata_path = _d(inputs["run_metadata_paths"])["full"]
        metadata = json.loads(metadata_path.read_text())
        del metadata["training_diagnostics"]["gradient_norm_series"][0]["grad_rms_content"]
        metadata_path.write_text(json.dumps(metadata))
        g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)

    def test_nested_worker_submodule_rms_shape_is_accepted(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
        for metadata_path in _d(inputs["run_metadata_paths"]).values():
            metadata = json.loads(metadata_path.read_text())
            row = metadata["training_diagnostics"]["gradient_norm_series"][0]
            nested = {
                key: row.pop(key) for key in ("grad_rms_trunk", "grad_rms_ste", "grad_rms_content")
            }
            row["submodule_gradient_rms"] = nested
            metadata_path.write_text(json.dumps(metadata))
        g5_stage1.build_e2e_arm_summary(liveness_config=_E2E_LIVENESS_CONFIG, **inputs)


def test_formal_gate_reports_rev31_telemetry_without_changing_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real gate writer must headline all nonbinding rev-3.1 telemetry."""
    inputs = _seven_arm_inputs(tmp_path)
    data_root = cast(Path, inputs["data_root"])
    b0_universe_path = tmp_path / "universe.npz"
    b0cal_output_dir = tmp_path / "b0cal"
    b0_cal.run_b0_cal_pipeline(
        universe_path=b0_universe_path,
        val_scores_path=tmp_path / "val.npz",
        data_root=data_root,
        strategy="toy",
        output_dir=b0cal_output_dir,
        seed=0,
        skip_perturbation_check=True,
    )
    b0cal_results_path = b0cal_output_dir / "b0cal_results.json"

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
        "_evaluate_e2e_probe",
        lambda **_kwargs: probe_report,
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

    def run_gate(output_dir: Path, *, seed: int = 0) -> dict[str, object]:
        return g5_stage1.run_g5_e2e_stage1_pipeline(
            arm_universe_paths=_d(inputs["arm_universe_paths"]),
            run_metadata_paths=_d(inputs["run_metadata_paths"]),
            b0_universe_path=b0_universe_path,
            b0cal_results_path=b0cal_results_path,
            probe_artifact_path=probe_path,
            data_root=data_root,
            strategy="toy",
            output_dir=output_dir,
            seed=seed,
        )

    reported_dir = tmp_path / "reported-gate"
    # Every trained arm's run metadata (`_write_trained_arm_metadata`) records
    # `seed: 0`; passing a distinct evaluation `--seed` here proves the two
    # values in the emitted metadata are independently sourced, not one
    # field silently duplicated as the other (adversarial-review finding 4).
    reported = run_gate(reported_dir, seed=3)
    reported_json = json.loads(
        (reported_dir / "g5_e2e_stage1_results.json").read_text()
    )
    reported_markdown = (reported_dir / "g5_e2e_stage1_tables.md").read_text()

    assert _d(reported["metadata"])["training_seed"] == 0
    assert _d(reported["metadata"])["seed"] == 3
    assert _d(reported_json["metadata"])["training_seed"] == 0
    assert _d(reported_json["metadata"])["seed"] == 3

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

    probe_rows = _markdown_table(reported_markdown, "## Probe-v2 evidence")
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
    def test_mode_e2e_forwards_exact_seven_arm_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = _seven_arm_inputs(tmp_path)
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
                "--p0-universe",
                str(_d(inputs)["arm_universe_paths"]["p0"]),
                "--no-l-rel-universe",
                str(_d(inputs)["arm_universe_paths"]["no_l_rel"]),
                "--row-layernorm-universe",
                str(_d(inputs)["arm_universe_paths"]["row_layernorm"]),
                "--run-metadata",
                *[str(path) for path in _d(inputs)["run_metadata_paths"].values()],
                "--b0-universe",
                str(b0_path),
                "--b0cal-results",
                str(b0cal_path),
                "--probe-artifact",
                str(probe_path),
                "--output-dir",
                str(tmp_path / "gate"),
            ]
        )
        assert set(_d(captured["arm_universe_paths"])) == set(g5_stage1._ALL_ARMS)
        assert set(_d(captured["run_metadata_paths"])) == set(g5_stage1._TRAINED_ARMS)
        assert captured["probe_artifact_path"] == probe_path

    def test_rejects_wrong_model_family(self, tmp_path: Path) -> None:
        inputs = _seven_arm_inputs(tmp_path)
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
            "metadata": {"arms_screened": list(g5_stage1._ALL_ARMS)},
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
        payload = self._payload(metadata={"training_seed": seeds[0], "screened_seeds": seeds})
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
