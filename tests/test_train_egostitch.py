"""Tests for src.train_egostitch: the EgoStitch Stage-1 training worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator
from src import train_egostitch as te
from src.data.artifacts import canonical_pair
from src.data.ego_targets import EgoTargetBuilder
from src.data.pairs import NegativeSampler
from src.e2_pipeline import _validate_worker_profile
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1

pytestmark = pytest.mark.unit

_TINY_MODEL: dict[str, object] = {
    "input_dim": 8,
    "d_p": 4,
    "d_z": 4,
    "d_h": 8,
    "slots": 4,
    "m_max": 8,
    "n_ground": 3,
    "decoder_layers": 1,
    "n_heads": 2,
    "gin_hidden": 8,
    "gin_layers": 2,
    "sinkhorn_iters": 3,
}


def _config_mapping(tmp_path: Path, **overrides: object) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "model": {"family": "egostitch", "config": dict(_TINY_MODEL)},
        "data": {
            "root": str(tmp_path / "data"),
            "strategy": "toy",
            "train_positives": "e_sup",
            "negative_ratio": 2,
            "partition_seed": 0,
            "msg_fraction": 0.8,
            "node_batch": 4,
            "edge_batch": 6,
            "f0_cache": str(tmp_path / "f0.pt"),
            "grounding_cache": str(tmp_path / "grounding.npz"),
            "s0_cache": str(tmp_path / "s0.npz"),
            "s0_checkpoint_id": "deadbeefcafefeed",
            "expected_missing_features": [],
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "warmstart_fraction": 0.25,
        },
        "diagnostics": {
            "gradient_probe_interval": 1,
            "gradient_imbalance_ratio": 10.0,
            "gradient_imbalance_steps": 2,
            "probe_s1_abs_mean_max": 1000.0,
            "selection_auprc_tolerance": 1e-4,
            "topk_fraction": 0.25,
        },
        "eval": {"patience": 2, "eval_every": 1},
        "seed": 0,
        "output_dir": str(tmp_path / "out"),
        "mixed_precision": "no",
        "preregistration": str(tmp_path / "prereg.json"),
    }
    mapping.update(overrides)
    return mapping


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config_mapping(tmp_path, **overrides)))
    return path


_RUNTIME: dict[str, Any] = {
    "world_size": "auto",
    "pack_dir": "outputs/pack",
    "pack_workers": 1,
    "loader_workers_per_rank": 0,
    "prefetch_factor": 2,
    "token_budget_candidates": [4, 8],
    "max_pairs_per_rank": 1024,
    "memory_limit_gib": 85.0,
    "total_budget_seconds": 100,
    "pack_budget_seconds": 20,
    "setup_probe_budget_seconds": 20,
    "train_eval_budget_seconds": 40,
    "artifact_budget_seconds": 10,
    "reserve_seconds": 10,
    "probe_warmup_steps": 1,
    "probe_timed_steps": 2,
}


def test_ddp_accelerator_detects_conditionally_unused_parameters() -> None:
    handler = te._egostitch_ddp_kwargs()
    assert handler.broadcast_buffers is False
    assert handler.find_unused_parameters is True
    assert handler.gradient_as_bucket_view is True


class TestLoadConfig:
    def test_round_trip(self, tmp_path: Path) -> None:
        cfg = te.load_config(_write_config(tmp_path))
        assert cfg.model.family == "egostitch"
        assert cfg.data.train_positives == "e_sup"
        assert cfg.optim.warmstart_fraction == 0.25
        assert cfg.diagnostics.gradient_probe_interval == 1
        assert cfg.diagnostics.gradient_imbalance_steps == 2
        assert cfg.preregistration == tmp_path / "prereg.json"
        assert cfg.runtime is None

    def test_runtime_accepts_family_specific_candidates(self, tmp_path: Path) -> None:
        cfg = te.load_config(_write_config(tmp_path, runtime=dict(_RUNTIME)))
        assert cfg.runtime is not None
        # NOT the frozen E2 probe list — B_n candidates are config-driven.
        assert cfg.runtime.token_budget_candidates == [4, 8]

    def test_rejects_wrong_family(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["model"]["family"] = "v3_1"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="egostitch"):
            te.load_config(path)

    def test_rejects_unknown_keys(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown config keys"):
            te.load_config(_write_config(tmp_path, bogus=1))

    def test_rejects_non_e_sup_positives(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["data"]["train_positives"] = "train_plus"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="e_sup"):
            te.load_config(path)

    def test_rejects_explicit_world_size(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, world_size=2)
        with pytest.raises(ValueError, match="must be 'auto'"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    def test_accepts_auto_world_size(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, world_size="auto")
        cfg = te.load_config(_write_config(tmp_path, runtime=runtime))
        assert cfg.runtime is not None
        assert cfg.runtime.world_size == 0

    def test_rejects_empty_candidates(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, token_budget_candidates=[])
        with pytest.raises(ValueError, match="token_budget_candidates"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    def test_rejects_budget_mismatch(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, reserve_seconds=999)
        with pytest.raises(ValueError, match="stage budgets"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    def test_missing_preregistration_key_rejected(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        del mapping["preregistration"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="preregistration"):
            te.load_config(path)


class TestParseArgs:
    def test_ddp_mode_requires_worker_flags(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            te.parse_args(["--config", str(tmp_path / "c.yaml"), "--ddp-mode", "probe"])

    def test_write_s0_manifest_flag(self, tmp_path: Path) -> None:
        args = te.parse_args(
            ["--config", str(tmp_path / "c.yaml"), "--write-s0-manifest", str(tmp_path / "s0.tsv")]
        )
        assert args.write_s0_manifest == tmp_path / "s0.tsv"
        assert args.ddp_mode is None

    def test_direct_training_entry_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="e2_pipeline"):
            te.main(["--config", str(_write_config(tmp_path))])

    def test_s0_manifest_uses_detected_world_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detected_world_size = 3
        data = type(
            "ManifestData",
            (),
            {"e_sup_positives": [], "val_pairs": [], "sampler": object()},
        )()
        captured: dict[str, object] = {}

        monkeypatch.setattr(te, "detect_visible_gpu_count", lambda: detected_world_size)
        monkeypatch.setattr(te, "assemble_egostitch_data", lambda *_args, **_kwargs: data)

        def fake_build(*_args: object, **kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(te, "build_s0_manifest", fake_build)
        te.main(
            [
                "--config",
                str(_write_config(tmp_path)),
                "--write-s0-manifest",
                str(tmp_path / "s0.tsv"),
            ]
        )
        assert captured["world_size"] == detected_world_size


# --------------------------------------------------------------------------- toy data bundle

_NODES = [f"n{i}" for i in range(8)]
_POSITIVES = [
    ("n0", "n1"),
    ("n0", "n2"),
    ("n1", "n2"),
    ("n3", "n4"),
    ("n0", "n3"),
    ("n5", "n6"),
    ("n2", "n5"),
    ("n4", "n6"),
    ("n1", "n7"),
    ("n7", "n7"),
]


def _toy_sampler(g: nx.Graph) -> NegativeSampler:
    degrees = {node: int(g.degree(node)) if node in g else 0 for node in _NODES}
    global_positives = frozenset(canonical_pair(u, v) for u, v in _POSITIVES)
    return NegativeSampler(_NODES, degrees, global_positives)


def _toy_bundle(tmp_path: Path, model_cfg: EgoStitchConfig) -> te.EgoStitchData:
    rng = np.random.default_rng(0)
    f0 = rng.normal(size=(len(_NODES), model_cfg.input_dim)).astype(np.float32)
    node_index = {node: i for i, node in enumerate(_NODES)}
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    g.add_edges_from([(u, v) for u, v in _POSITIVES[:6] if u != v])

    pool = {node: [v for v in _NODES if v != node][: model_cfg.n_ground] for node in _NODES}
    grounding_index = np.array(
        [[node_index[v] for v in pool[node]] for node in _NODES], dtype=np.int64
    )
    builder = EgoTargetBuilder(g, f0, node_index, pool, slots=model_cfg.slots)
    sampler = _toy_sampler(g)

    e_sup = [canonical_pair(u, v) for u, v in _POSITIVES[6:]]
    val_pairs = [("n0", "n4"), ("n1", "n2"), ("n2", "n6"), ("n3", "n7")]
    val_labels = np.array([0, 1, 0, 0], dtype=np.int8)

    # s0 cache over the full toy universe (guaranteed coverage).
    s0 = te.S0Cache.__new__(te.S0Cache)
    s0._logits = {}
    for i, u in enumerate(_NODES):
        for v in _NODES[i:]:
            s0._logits[canonical_pair(u, v)] = 0.1 * (i + 1)

    return te.EgoStitchData(
        train_nodes=list(_NODES),
        e_sup_positives=e_sup,
        val_pairs=val_pairs,
        val_labels=val_labels,
        f0=torch.from_numpy(f0),
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(_NODES)},
        target_builder=builder,
        sampler=sampler,
        s0=s0,
        rho_train=float(g.number_of_edges() / 36.0),
    )


def _toy_cfg(tmp_path: Path) -> te.EgoConfig:
    (tmp_path / "prereg.json").write_text('{"registration_id": "toy"}\n')
    return te.load_config(_write_config(tmp_path))


# --------------------------------------------------------------------------- streams


class TestEnumerateEdgeStream:
    def _rows(self, epoch: int, rank: int, world: int) -> list[tuple[str, str, int]]:
        g = nx.Graph()
        g.add_nodes_from(_NODES)
        g.add_edges_from([(u, v) for u, v in _POSITIVES[:6] if u != v])
        sampler = _toy_sampler(g)
        positives = [canonical_pair(u, v) for u, v in _POSITIVES[6:]]
        return te.enumerate_edge_stream(
            positives, sampler, negative_ratio=2, seed=0, epoch=epoch, rank=rank, world_size=world
        )

    def test_deterministic(self) -> None:
        assert self._rows(1, 0, 2) == self._rows(1, 0, 2)

    def test_deterministic_across_input_order(self) -> None:
        g = nx.Graph()
        g.add_nodes_from(_NODES)
        g.add_edges_from([(u, v) for u, v in _POSITIVES[:6] if u != v])
        positives = [canonical_pair(u, v) for u, v in _POSITIVES[6:]]
        forward = te.enumerate_edge_stream(
            positives,
            _toy_sampler(g),
            negative_ratio=2,
            seed=0,
            epoch=1,
            rank=0,
            world_size=2,
        )
        reversed_rows = te.enumerate_edge_stream(
            list(reversed(positives)),
            _toy_sampler(g),
            negative_ratio=2,
            seed=0,
            epoch=1,
            rank=0,
            world_size=2,
        )
        assert forward == reversed_rows

    def test_positive_shards_partition_e_sup(self) -> None:
        positives = {canonical_pair(u, v) for u, v in _POSITIVES[6:]}
        seen: list[tuple[str, str]] = []
        for rank in range(2):
            seen.extend((u, v) for u, v, label in self._rows(1, rank, 2) if label == 1)
        assert sorted(seen) == sorted(positives)

    def test_negative_ratio_respected(self) -> None:
        rows = self._rows(1, 0, 2)
        n_pos = sum(1 for _, _, label in rows if label == 1)
        n_neg = sum(1 for _, _, label in rows if label == 0)
        assert n_neg == 2 * n_pos

    def test_epochs_differ(self) -> None:
        assert self._rows(1, 0, 2) != self._rows(2, 0, 2)


class TestS0Manifest:
    def test_manifest_covers_every_stream_pair(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        manifest_path = tmp_path / "s0_manifest.tsv"
        n_unique = te.build_s0_manifest(
            cfg, data.e_sup_positives, data.val_pairs, data.sampler, manifest_path, world_size=2
        )
        listed = {
            tuple(line.split("\t")) for line in manifest_path.read_text().strip().splitlines()
        }
        assert n_unique == len(listed)
        for epoch in range(1, cfg.optim.epochs + 1):
            for rank in range(2):
                for u, v, _ in te.enumerate_edge_stream(
                    data.e_sup_positives,
                    data.sampler,
                    negative_ratio=cfg.data.negative_ratio,
                    seed=cfg.seed,
                    epoch=epoch,
                    rank=rank,
                    world_size=2,
                ):
                    assert canonical_pair(u, v) in listed
        for u, v in data.val_pairs:
            assert canonical_pair(u, v) in listed


class TestS0Cache:
    def test_checkpoint_mismatch_raises(self, tmp_path: Path) -> None:
        from tests.test_g1_hardened_e2 import _write_universe_npz

        pairs = [("n0", "n1"), ("n1", "n2")]
        path = tmp_path / "s0.npz"
        _write_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.array([0.5, -0.5], dtype=np.float32),
            labels=np.array([1, 0], dtype=np.int8),
        )
        with pytest.raises(ValueError, match="checkpoint_id"):
            te.S0Cache.from_path(path, expected_checkpoint_id="0000000000000000")

    def test_lookup_and_hard_miss(self, tmp_path: Path) -> None:
        from tests.test_g1_hardened_e2 import _write_universe_npz

        pairs = [("n0", "n1"), ("n1", "n2")]
        path = tmp_path / "s0.npz"
        _write_universe_npz(
            path,
            node_ids=_NODES,
            pairs=pairs,
            logits=np.array([0.5, -0.5], dtype=np.float32),
            labels=np.array([1, 0], dtype=np.int8),
        )
        cache = te.S0Cache.from_path(path, expected_checkpoint_id="deadbeefcafefeed")
        np.testing.assert_allclose(cache.lookup([("n1", "n0")]), [0.5])
        with pytest.raises(KeyError, match="missing pair"):
            cache.lookup([("n0", "n7")])


class TestEpochStepPlan:
    def test_rows_and_steps(self) -> None:
        rows, steps = te._epoch_step_plan(10, negative_ratio=2, edge_batch=6, world_size=4)
        assert rows == [9, 9, 6, 6]  # ceil splits of 10 positives x (1 + 2)
        assert steps == 2

    def test_global_count_tail(self) -> None:
        rows = [9, 9, 6, 6]
        assert te._step_global_count(rows, 0, 6) == 24
        assert te._step_global_count(rows, 1, 6) == 6
        assert te._step_global_count(rows, 2, 6) == 0


class TestRegisteredDiagnostics:
    def test_gradient_imbalance_activates_only_after_persistence(self) -> None:
        monitor = te._GradientImbalanceMonitor(ratio=10.0, required_steps=100, interval=50)
        norms = {"edge": 100.0, "recon": 1.0, "real": 1.0, "ssl": 1.0}
        assert monitor.update(50, norms) is False
        assert monitor.update(100, norms) is True
        assert monitor.activated_step == 100

    def test_probe_scale_guard_fails_on_synthetic_broken_kernel(self) -> None:
        with pytest.raises(RuntimeError, match="pathological Stage-1 membership scale"):
            te._enforce_probe_s1_scale(1.2e7, 1.0e3)


# --------------------------------------------------------------------------- loop smoke


class TestTrainLoop:
    def _run(self, tmp_path: Path) -> tuple[te.EgoConfig, te.EgoStitchData, te.EgoTrainResult]:
        torch.manual_seed(0)
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        model = EgoStitchStage1(model_cfg)
        accelerator = Accelerator(mixed_precision="no", cpu=True)
        result = te.train_egostitch_ddp_loop(
            model, cfg, data, accelerator, node_batch=cfg.data.node_batch
        )
        return cfg, data, result

    def test_history_and_profile_contract(self, tmp_path: Path) -> None:
        cfg, _, result = self._run(tmp_path)
        assert [int(cast(float, row["epoch"])) for row in result.history] == [1, 2]
        assert all("auprc" in row and "loss_total" in row for row in result.history)
        assert all("fidelity" in row for row in result.history)
        assert any(row["gradient_norm_probes"] for row in result.history)
        assert "kendall_fallback" in result.runtime_profile
        # The orchestrator-validated schema accepts this profile verbatim.
        profile = _validate_worker_profile(
            result.runtime_profile, epochs=cfg.optim.epochs, world_size=1, memory_limit_gib=85.0
        )
        assert profile["feature_cache_hit_rate"] == 1.0

    def test_write_outputs_payload_contract(self, tmp_path: Path) -> None:
        cfg, data, result = self._run(tmp_path)
        te.write_run_start_metadata(cfg, data, world_size=1)
        started = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert started["status"] == "started"
        assert started["seed"] == cfg.seed
        assert started["world_size"] == 1
        te.write_outputs(result, cfg, data)
        payload = torch.load(cfg.output_dir / "best.pt", weights_only=False)
        assert set(payload) == {
            "model_state",
            "model_family",
            "model_config",
            "epoch",
            "val_metrics",
            "seed",
            "config",
        }
        assert payload["model_family"] == "egostitch"
        rows = [
            json.loads(line) for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["epoch"] for row in rows] == [1, 2]
        assert all("fidelity" in row and "gradient_norm_probes" in row for row in rows)
        metadata = json.loads((cfg.output_dir / "run_metadata.json").read_text())
        assert metadata["status"] == "complete"
        assert metadata["started_at"] == started["started_at"]
        assert metadata["preregistration_sha256"] == te._sha256_file(cfg.preregistration)
        assert metadata["s0_checkpoint_id"] == "deadbeefcafefeed"
        assert metadata["rho_train"] == data.rho_train

    def test_preregistration_drift_after_start_refuses_finalize(self, tmp_path: Path) -> None:
        cfg, data, result = self._run(tmp_path)
        te.write_run_start_metadata(cfg, data, world_size=1)
        cfg.preregistration.write_text('{"registration_id": "changed"}\n')
        with pytest.raises(RuntimeError, match="changed after run start"):
            te.write_outputs(result, cfg, data)

    def test_batch_factory_deterministic(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        rows, steps = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=cfg.data.negative_ratio,
            edge_batch=cfg.data.edge_batch,
            world_size=1,
        )

        def first_batch() -> te._CompositeBatch:
            factory = te._BatchFactory(cfg, model_cfg, data, node_batch=4, rank=0, world_size=1)
            return next(iter(factory.epoch_batches(1, rows_per_rank=rows, steps=steps)))

        a, b = first_batch(), first_batch()
        torch.testing.assert_close(a.node["x"], b.node["x"])
        torch.testing.assert_close(a.edge["s0"], b.edge["s0"])
        torch.testing.assert_close(a.node["null_mode"], b.node["null_mode"])
        assert a.edge_rows_global == b.edge_rows_global


class TestPreparePack:
    def test_warm_validation_detects_drift(self, tmp_path: Path) -> None:
        # Build a fake warm pack with a manifest, then corrupt a file.
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        (pack_dir / te._PACK_F0_FILENAME).write_bytes(b"f0")
        (pack_dir / te._PACK_GROUNDING_FILENAME).write_bytes(b"pool")
        manifest = {
            "family": "egostitch",
            "strategy": "toy",
            "n_operative_nodes": 8,
            "n_train_nodes": 8,
            "n_ground": 3,
            "files": {
                te._PACK_F0_FILENAME: te._sha256_file(pack_dir / te._PACK_F0_FILENAME),
                te._PACK_GROUNDING_FILENAME: te._sha256_file(
                    pack_dir / te._PACK_GROUNDING_FILENAME
                ),
            },
        }
        (pack_dir / te._PACK_MANIFEST_FILENAME).write_text(json.dumps(manifest))
        cfg = _toy_cfg(tmp_path)
        payload = te.prepare_pack(cfg, pack_dir, cold_cache=False)
        assert cast(dict[str, object], payload["pack_manifest"])["n_ground"] == 3
        (pack_dir / te._PACK_F0_FILENAME).write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="drifted"):
            te.prepare_pack(cfg, pack_dir, cold_cache=False)
