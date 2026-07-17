"""Tests for src.train_egostitch: the EgoStitch Stage-1 training worker."""

from __future__ import annotations

import json
import math
from dataclasses import replace
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
from src.data.packed_features import (
    PACK_FORMAT,
    PackedFeatureManifest,
    PackedFeatureTable,
    PackedNodeRecord,
    PackedShardRecord,
    sha256_file,
    write_packed_manifest,
)
from src.data.pairs import NegativeSampler
from src.e2_pipeline import _validate_worker_profile
from src.model.egostitch import EgoStitchConfig, EgoStitchStage1
from src.model.egostitch.conditioning import GatedCrossAttention
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E
from src.train_b0 import ModelConfig

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

_E2E_TINY_MODEL: dict[str, object] = {
    "d_model": 16,
    "encoder_layers": 1,
    "cross_attn_layers": 1,
    "n_heads": 2,
    "n_inj": 1,
    "ste_dim": 8,
    "ste_layers": 1,
    "xattn_heads": 2,
    "p_topo": 0.15,
    "p_cont": 0.15,
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

    def test_pack_dir_defaults_to_none(self, tmp_path: Path) -> None:
        cfg = te.load_config(_write_config(tmp_path))
        assert cfg.data.pack_dir is None

    def test_accepts_pack_dir(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["data"]["pack_dir"] = str(tmp_path / "token_pack")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        cfg = te.load_config(path)
        assert cfg.data.pack_dir == tmp_path / "token_pack"

    def test_accepts_egostitch_e2e_family(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["model"] = {"family": "egostitch_e2e", "config": dict(_E2E_TINY_MODEL)}
        mapping["data"]["pack_dir"] = str(tmp_path / "token_pack")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        cfg = te.load_config(path)
        assert cfg.model.family == "egostitch_e2e"
        assert cfg.model.config == _E2E_TINY_MODEL

    def test_egostitch_e2e_family_rejects_stage1_only_keys(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["model"] = {"family": "egostitch_e2e", "config": {"slots": 4}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="unknown E2E config keys"):
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


_TOKEN_DIM = 1536


def _write_tiny_token_pack(pack_dir: Path, nodes: list[str], *, min_length: int = 2) -> None:
    """Write a minimal raw-token pack (same reader `_BatchFactory` consumes).

    Every node gets a distinct token-sequence length so pair identity and
    order can be recovered unambiguously from the returned ``len_a``/``len_b``.
    ``min_length`` (default 2, matching the original fixture) is bumped to 3
    by callers that run the sequences through a real `PairCrossAttention`-
    family forward pass, which requires at least one inner token strictly
    between the BOS/EOS positions (`src/model/B0.py`'s `inner_token_mask`).
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    shard_path = pack_dir / "shard-000.bin"
    records: list[PackedNodeRecord] = []
    offset = 0
    with shard_path.open("wb") as handle:
        for i, node in enumerate(nodes):
            length = min_length + i
            tensor = torch.from_numpy(rng.normal(size=(length, _TOKEN_DIM)).astype(np.float32))
            raw = tensor.to(torch.bfloat16).contiguous().view(torch.uint16)
            handle.write(raw.numpy().tobytes())
            records.append(PackedNodeRecord(node, 0, offset, offset, length))
            offset += length
    manifest = PackedFeatureManifest(
        format=PACK_FORMAT,
        input_dim=_TOKEN_DIM,
        dtype="bfloat16",
        source_metadata_sha256="0" * 64,
        source_index_sha256="0" * 64,
        nodes=tuple(records),
        shards=(
            PackedShardRecord(
                filename="shard-000.bin",
                num_tokens=offset,
                byte_size=shard_path.stat().st_size,
                sha256=sha256_file(shard_path),
            ),
        ),
        pack_workers=1,
        build_seconds=0.0,
    )
    write_packed_manifest(pack_dir, manifest)


class TestBatchFactoryE2E:
    def test_edge_batch_carries_token_streams_in_stream_order(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        pack_dir = tmp_path / "token_pack"
        _write_tiny_token_pack(pack_dir, _NODES)
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config={}),
            data=replace(cfg.data, pack_dir=pack_dir),
        )
        rows, steps = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=e2e_cfg.data.negative_ratio,
            edge_batch=e2e_cfg.data.edge_batch,
            world_size=1,
        )

        factory = te._BatchFactory(e2e_cfg, model_cfg, data, node_batch=4, rank=0, world_size=1)
        batch = next(iter(factory.epoch_batches(1, rows_per_rank=rows, steps=steps)))

        edge_batch = e2e_cfg.data.edge_batch
        assert batch.edge["emb_a"].shape == (edge_batch, batch.edge["emb_a"].shape[1], _TOKEN_DIM)
        assert batch.edge["emb_b"].shape == batch.edge["emb_a"].shape
        assert batch.edge["len_a"].shape == (edge_batch,)
        assert batch.edge["len_b"].shape == (edge_batch,)
        assert "s0" not in batch.edge
        # F0/grounding tensors stay in the batch: imagination still needs them.
        assert batch.edge["x_i"].shape == (edge_batch, model_cfg.input_dim)
        assert batch.edge["ground_i"].shape[0] == edge_batch

        expected_rows = te.enumerate_edge_stream(
            data.e_sup_positives,
            data.sampler,
            negative_ratio=e2e_cfg.data.negative_ratio,
            seed=e2e_cfg.seed,
            epoch=1,
            rank=0,
            world_size=1,
        )
        expected_chunk = expected_rows[:edge_batch]
        table = PackedFeatureTable.from_pack(pack_dir, torch.device("cpu"))
        index = table.manifest.node_index()
        expected_len_a = torch.tensor(
            [table.manifest.nodes[index[u]].length for u, _, _ in expected_chunk],
            dtype=torch.long,
        )
        expected_len_b = torch.tensor(
            [table.manifest.nodes[index[v]].length for _, v, _ in expected_chunk],
            dtype=torch.long,
        )
        expected_boundary = max(int(expected_len_a.max()), int(expected_len_b.max()))
        torch.testing.assert_close(batch.edge["len_a"], expected_len_a)
        torch.testing.assert_close(batch.edge["len_b"], expected_len_b)
        assert batch.edge["emb_a"].shape[1] == expected_boundary

    def test_requires_pack_dir(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        e2e_cfg = replace(cfg, model=ModelConfig(family="egostitch_e2e", config={}))
        with pytest.raises(ValueError, match="pack_dir"):
            te._BatchFactory(e2e_cfg, model_cfg, data, node_batch=4, rank=0, world_size=1)


class TestCompositeStepE2E:
    """One CPU optimization step through `_CompositeStep`, family egostitch_e2e."""

    def _batch_and_model(self, tmp_path: Path) -> tuple[te._CompositeBatch, EgoStitchE2E]:
        # EgoStitchE2E's internal generator always uses the full spec-default
        # EgoStitchConfig() (input_dim=1536, slots=16, ...), never a value
        # parsed from E2EConfig -- the toy bundle must match that, not the
        # tiny per-field dict the frozen-s0 family tests use.
        model_cfg = EgoStitchConfig()
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, model_cfg)
        pack_dir = tmp_path / "token_pack"
        _write_tiny_token_pack(pack_dir, _NODES, min_length=3)
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
            data=replace(cfg.data, pack_dir=pack_dir),
        )
        rows, steps = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=e2e_cfg.data.negative_ratio,
            edge_batch=e2e_cfg.data.edge_batch,
            world_size=1,
        )
        factory = te._BatchFactory(e2e_cfg, model_cfg, data, node_batch=4, rank=0, world_size=1)
        batch = next(iter(factory.epoch_batches(1, rows_per_rank=rows, steps=steps)))
        model = EgoStitchE2E(E2EConfig.from_mapping(e2e_cfg.model.config))
        return batch, model

    def _payload(
        self,
        batch: te._CompositeBatch,
        *,
        joint_weight: float,
        collect_diagnostics: bool = False,
    ) -> dict[str, object]:
        return {
            "node": batch.node,
            "edge": batch.edge,
            "joint_weight": torch.tensor(joint_weight),
            "edge_rows_global": batch.edge_rows_global,
            "seed": 0,
            "epoch": 1,
            "step": 0,
            "collect_diagnostics": collect_diagnostics,
        }

    def _gates(self, model: EgoStitchE2E) -> tuple[GatedCrossAttention, GatedCrossAttention]:
        topo_gate = model.trunk.topo_xattn[0]
        cont_gate = model.trunk.cont_xattn[0]
        assert isinstance(topo_gate, GatedCrossAttention)
        assert isinstance(cont_gate, GatedCrossAttention)
        return topo_gate, cont_gate

    @staticmethod
    def _bf16_autocast() -> torch.autocast:
        # The packed-token store is bf16-only (src/data/packed_features.py);
        # a real run consumes it under Accelerate's bf16 autocast (the
        # `mixed_precision: bf16` worker config) -- reproduced explicitly here
        # since this test drives `_CompositeStep` directly, without an
        # `Accelerator` wrapping it.
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    def test_loss_finite_and_gates_dead_during_warmstart(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        topo_gate, cont_gate = self._gates(model)

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=0.0))
        loss = cast(torch.Tensor, out["loss"])
        assert bool(torch.isfinite(loss))
        loss.backward()  # type: ignore[no-untyped-call]

        # joint_weight=0 zeros L_edge (and therefore trunk/STE/gates) during
        # warm-start; the gate either never enters the graph (None) or enters
        # multiplied by an exact zero (design Sec 4, spec Sec 13.8's reused
        # curriculum).
        for gate in (topo_gate.gate, cont_gate.gate):
            assert gate.grad is None or float(gate.grad) == pytest.approx(0.0)

    def test_gates_receive_gradient_after_warmstart(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        topo_gate, cont_gate = self._gates(model)

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=1.0))
        loss = cast(torch.Tensor, out["loss"])
        assert bool(torch.isfinite(loss))
        loss.backward()  # type: ignore[no-untyped-call]

        assert topo_gate.gate.grad is not None
        assert cont_gate.gate.grad is not None
        total_abs_grad = float(topo_gate.gate.grad.abs()) + float(cont_gate.gate.grad.abs())
        assert total_abs_grad > 0.0

    def test_telemetry_keys_present_in_metrics_row(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=1.0, collect_diagnostics=True))
        assert bool(torch.isfinite(cast(torch.Tensor, out["loss"])))

        for key in ("gate_topo_tanh", "gate_cont_tanh"):
            assert key in out
            values = cast(list[float], out[key])
            assert len(values) == model.cfg.n_inj
            assert all(math.isfinite(value) for value in values)

        families = cast(dict[str, torch.Tensor], out["families"])
        with self._bf16_autocast():
            grad_rms = te._e2e_submodule_gradient_rms(model, families["edge"])
        metrics_row: dict[str, object] = {**out, **grad_rms}
        for key in ("grad_rms_trunk", "grad_rms_ste", "grad_rms_content"):
            assert key in metrics_row
            assert math.isfinite(cast(float, metrics_row[key]))

        with self._bf16_autocast():
            metrics_row["topology_delta_std"] = te._e2e_topology_delta_std(
                model, te._e2e_edge_view(batch.edge)
            )
        assert math.isfinite(cast(float, metrics_row["topology_delta_std"]))

    def test_dead_decision_head_excluded_from_trainable_parameters(self, tmp_path: Path) -> None:
        _, model = self._batch_and_model(tmp_path)
        trainable_ids = {id(p) for p in te._e2e_trainable_parameters(model)}
        decision_ids = {id(p) for p in model.generator.decision.parameters()}
        assert decision_ids.isdisjoint(trainable_ids)
        assert any(id(p) in trainable_ids for p in model.trunk.parameters())


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
