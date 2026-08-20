"""Config-schema and edge-stream tests for the `src.train_egostitch` worker.

This module is also the shared toy-fixture home for the EgoStitch worker test
suite: `tests/test_train_egostitch_core.py`, `tests/test_train_egostitch_e2e.py`
and `tests/test_g5_stage1_e2e.py` all import `_toy_cfg`/`_toy_bundle` and the
tiny model dicts from here, so the fixtures below stay put even though the
frozen-s0 ``egostitch`` execution path they were originally written for was
retired with the two-stage cleanup (design 2026-07-29 Sec 6.2).

The live worker-core behaviour (run kinds, the BINDING plan lock, the batch
factory, and registered diagnostics) lives in
`tests/test_train_egostitch_core.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]
from src import train_egostitch as te
from src.data.artifacts import canonical_pair
from src.data.ego_targets import EgoTargetBuilder
from src.data.pairs import NegativeSampler
from src.model.egostitch import EgoStitchConfig

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
    "generator": {
        "name": "egostitch_imagine",
        # Matches the `EgoStitchConfig()` base default used to build the
        # paired `_toy_bundle`/`_BatchFactory` fixtures in
        # test_train_egostitch_e2e.py, so `EgoStitchModel.generator.cfg.n_ground`
        # (`GeneratorConfig`-sourced, spec Sec 14.4.4) stays consistent with
        # those fixtures rather than silently picking up the unrelated
        # rev-3.1 default (50).
        "n_ground": 20,
        # `_toy_bundle` builds an in-memory `EgoStitchData` with `feature_stats
        # is None` -- it never goes through `assemble_egostitch_data`'s
        # training-universe statistics path (Task 6), so the `zscore_vfit_v1` default would raise
        # unless a test explicitly registers statistics first. The
        # `TestE2ECompositeStep`/`TestPrepareAndAssembleE2E`-adjacent suites that
        # build models from this dict exercise the composite optimizer step,
        # gates, family/budget probes, and telemetry -- not standardization --
        # so pin the stateless, byte-identical-to-what-these-tests-were-written-
        # against transform, exactly like `_tiny_model_and_batch` does in
        # tests/model/test_egostitch_e2e_model.py.
        "feature_standardization": "row_layernorm",
    },
    "encoder": {
        "name": "ste_typed",
        "dim": 8,
        "layers": 1,
    },
    "classifier": {
        "name": "b0_v31",
        "d_model": 16,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "n_inj": 1,
        "xattn_heads": 2,
        "p_topo": 0.15,
    },
}


def _e2e_model_config(
    base: Mapping[str, object] = _E2E_TINY_MODEL, **section_overrides: Mapping[str, object]
) -> dict[str, Any]:
    """`base` (default `_E2E_TINY_MODEL`) with each keyword merged into that section.

    The nested-schema analogue of the old flat-config idiom
    ``{**_E2E_TINY_MODEL, "permanent_null": "all_head"}``: since a field now
    lives inside its component's own sub-mapping, overriding it means merging
    into that sub-mapping rather than the top level. Example:
    ``_e2e_model_config(classifier={"permanent_null": "all_head"})`` returns
    ``base`` with only ``classifier.permanent_null`` overridden -- every other
    ``classifier`` field, and every other section, is left at ``base``'s value.
    """
    merged: dict[str, Any] = {
        key: dict(cast(Mapping[str, object], value)) for key, value in base.items()
    }
    for section, overrides in section_overrides.items():
        merged.setdefault(section, {}).update(overrides)
    return merged


def _config_mapping(tmp_path: Path, **overrides: object) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "model": {"family": "egostitch_e2e", "config": dict(_E2E_TINY_MODEL)},
        "data": {
            "root": str(tmp_path / "data"),
            "strategy": "toy",
            "negative_ratio": 5,
            "node_batch": 4,
            "edge_batch": 6,
            "f0_cache": str(tmp_path / "f0.pt"),
            "grounding_cache": str(tmp_path / "grounding.npz"),
            "expected_missing_features": [],
            "pack_dir": str(tmp_path / "token_pack"),
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "epochs": 30,
            "warmup_steps": 500,
            "grad_clip": 1.0,
        },
        "diagnostics": {
            "gradient_probe_interval": 50,
            "gradient_imbalance_ratio": 50.0,
            "gradient_imbalance_steps": 200,
            "probe_s1_abs_mean_max": 1000.0,
            "topk_fraction": 0.01,
        },
        "eval": {"patience": 2, "eval_every": 1},
        "seed": 0,
        "output_dir": str(tmp_path / "out"),
        "mixed_precision": "bf16",
        "training": {},
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
    "token_budget": 4,
    "max_pairs_per_rank": 1024,
    "memory_limit_gib": 85.0,
}


class TestLoadConfig:
    def test_round_trip(self, tmp_path: Path) -> None:
        cfg = te.load_config(_write_config(tmp_path))
        assert cfg.model.family == "egostitch_e2e"
        assert cfg.diagnostics.gradient_probe_interval == 50
        assert cfg.diagnostics.gradient_imbalance_steps == 200
        assert cfg.runtime is None
        assert cfg.topology_validation == te.EgoTopologyValidationConfig()

    def test_accepts_cascade_topology_validation(self, tmp_path: Path) -> None:
        cfg = te.load_config(
            _write_config(
                tmp_path,
                topology_validation={
                    "full_every_epochs": 3,
                    "cascade_buckets_per_size": 5,
                    "cascade_complement_sample_size": 20_000,
                },
            )
        )
        assert cfg.topology_validation.full_every_epochs == 3
        assert cfg.topology_validation.cascade_buckets_per_size == 5
        assert cfg.topology_validation.cascade_complement_sample_size == 20_000

    @pytest.mark.parametrize(
        ("topology_validation", "message"),
        [
            ({"full_every_epochs": 0}, "full_every_epochs"),
            ({"cascade_buckets_per_size": 1}, "cascade_buckets_per_size"),
            ({"cascade_complement_sample_size": -1}, "complement_sample_size"),
        ],
    )
    def test_rejects_invalid_cascade_topology_validation(
        self,
        tmp_path: Path,
        topology_validation: dict[str, int],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            te.load_config(_write_config(tmp_path, topology_validation=topology_validation))

    def test_topology_validation_scope_uses_full_cadence_and_final_epoch(self) -> None:
        config = te.EgoTopologyValidationConfig(full_every_epochs=3)
        assert [te._e2e_topology_validation_scope(epoch, 8, config) for epoch in range(1, 9)] == [
            "cascade",
            "cascade",
            "full",
            "cascade",
            "cascade",
            "full",
            "cascade",
            "full",
        ]

    def test_runtime_accepts_family_specific_token_budget(self, tmp_path: Path) -> None:
        cfg = te.load_config(_write_config(tmp_path, runtime=dict(_RUNTIME)))
        assert cfg.runtime is not None
        # NOT a frozen E2 probe value — B_n is config-driven for this family.
        assert cfg.runtime.token_budget == 4

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

    def test_rejects_removed_message_supervision_config(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["data"]["msg_fraction"] = 0.8
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        with pytest.raises(ValueError, match="unknown config keys"):
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

    def test_rejects_non_positive_token_budget(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, token_budget=0)
        with pytest.raises(ValueError, match="token_budget"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    @pytest.mark.parametrize(
        "retired_key",
        [
            "total" + "_budget_seconds",
            "pack" + "_budget_seconds",
            "setup_probe" + "_budget_seconds",
            "train_eval" + "_budget_seconds",
            "artifact" + "_budget_seconds",
            "reserve" + "_seconds",
        ],
    )
    def test_rejects_removed_runtime_time_keys(self, tmp_path: Path, retired_key: str) -> None:
        runtime = dict(_RUNTIME, **{retired_key: 1})
        with pytest.raises(ValueError, match="unknown config keys"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    def test_rejects_retired_runtime_probe_key(self, tmp_path: Path) -> None:
        runtime = dict(_RUNTIME, probe_warmup_steps=1)
        with pytest.raises(ValueError, match="unknown config keys"):
            te.load_config(_write_config(tmp_path, runtime=runtime))

    def test_accepts_pack_dir(self, tmp_path: Path) -> None:
        mapping = _config_mapping(tmp_path)
        mapping["data"]["pack_dir"] = str(tmp_path / "token_pack")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mapping))
        cfg = te.load_config(path)
        assert cfg.data.pack_dir == tmp_path / "token_pack"


class TestParseArgs:
    def test_ddp_mode_requires_worker_flags(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            te.parse_args(["--config", str(tmp_path / "c.yaml"), "--ddp-mode", "train"])

    @pytest.mark.parametrize("measurement_mode", ["probe", "epoch-probe"])
    def test_measurement_ddp_modes_are_accepted(
        self, tmp_path: Path, measurement_mode: str
    ) -> None:
        args = te.parse_args(
            [
                "--config",
                str(tmp_path / "c.yaml"),
                "--ddp-mode",
                measurement_mode,
                "--pack-dir",
                str(tmp_path / "pack"),
                "--output-dir",
                str(tmp_path / "out"),
                "--token-budget-per-rank",
                "4",
                "--profile-output",
                str(tmp_path / "profile.json"),
            ]
        )
        assert args.ddp_mode == measurement_mode

    def test_retired_init_probe_mode_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            te.parse_args(["--config", str(tmp_path / "c.yaml"), "--ddp-mode", "init-probe"])

    def test_direct_training_entry_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="e2_pipeline"):
            te.main(["--config", str(_write_config(tmp_path))])


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

    training_positives = [canonical_pair(u, v) for u, v in _POSITIVES[6:]]
    val_pairs = [("n0", "n4"), ("n1", "n2"), ("n2", "n6"), ("n3", "n7")]
    val_labels = np.array([0, 1, 0, 0], dtype=np.int8)

    return te.EgoStitchData(
        train_nodes=list(_NODES),
        training_positives=training_positives,
        val_pairs=val_pairs,
        val_labels=val_labels,
        f0=torch.from_numpy(f0),
        node_index=node_index,
        grounding_index=grounding_index,
        train_pos={node: i for i, node in enumerate(_NODES)},
        target_builder=builder,
        sampler=sampler,
        rho_train=float(g.number_of_edges() / 36.0),
    )


def _toy_cfg(tmp_path: Path) -> te.EgoConfig:
    return te.EgoConfig(
        model=te.ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
        data=te.EgoDataConfig(
            root=tmp_path / "data",
            strategy="toy",
            negative_ratio=5,
            node_batch=4,
            edge_batch=6,
            f0_cache=tmp_path / "f0.pt",
            grounding_cache=tmp_path / "grounding.npz",
            expected_missing_features=[],
            pack_dir=tmp_path / "token_pack",
        ),
        optim=te.EgoOptimConfig(
            lr=1e-4, weight_decay=0.01, epochs=2, warmup_steps=2, grad_clip=1.0
        ),
        diagnostics=te.EgoDiagnosticsConfig(
            gradient_probe_interval=1,
            gradient_imbalance_ratio=10.0,
            gradient_imbalance_steps=2,
            probe_s1_abs_mean_max=1000.0,
            topk_fraction=0.25,
        ),
        eval=te.EvalConfig(patience=2, eval_every=1),
        seed=0,
        output_dir=tmp_path / "out",
        mixed_precision="no",
        training=te.EgoStitchTrainingConfig(),
    )


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

    def test_positive_shards_partition_all_training_positives(self) -> None:
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
