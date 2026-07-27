"""E2E training contracts for the EgoStitch worker."""

from __future__ import annotations

import gc
import json
import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src import train_egostitch as te
from src.data.artifacts import Benchmark, LabeledPairs, SplitArtifacts, canonical_pair
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
from src.model.egostitch import EgoStitchConfig
from src.model.egostitch import e2e_model as e2e_module
from src.model.egostitch.conditioning import GatedCrossAttention, HeadNullMasks
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import E2EPairContext, EgoStitchE2E
from src.train_b0 import ModelConfig

from tests.test_train_egostitch import _E2E_TINY_MODEL, _NODES, _toy_bundle, _toy_cfg

pytestmark = pytest.mark.unit

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
    @staticmethod
    def _target_factory(
        tmp_path: Path, *, rank: int = 0, world_size: int = 1
    ) -> tuple[te._BatchFactory, EgoStitchConfig]:
        model_cfg = EgoStitchConfig()
        nodes = [f"target-node-{index:02d}" for index in range(34)]
        node_index = {node: index for index, node in enumerate(nodes)}
        rng = np.random.default_rng(71)
        f0 = rng.normal(size=(len(nodes), model_cfg.input_dim)).astype(np.float32)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((nodes[0], node) for node in nodes[1:])
        pool = {
            node: [candidate for candidate in nodes if candidate != node][
                : model_cfg.n_ground
            ]
            for node in nodes
        }
        grounding_index = np.array(
            [[node_index[candidate] for candidate in pool[node]] for node in nodes],
            dtype=np.int64,
        )
        degrees = {node: int(graph.degree(node)) for node in nodes}
        sampler = NegativeSampler(nodes, degrees, frozenset(graph.edges()))
        s0 = te.S0Cache.__new__(te.S0Cache)
        s0._logits = {}
        data = te.EgoStitchData(
            train_nodes=nodes,
            e_sup_positives=[(nodes[0], nodes[1]), (nodes[0], nodes[2])],
            val_pairs=[],
            val_labels=np.empty(0, dtype=np.int8),
            f0=torch.from_numpy(f0),
            node_index=node_index,
            grounding_index=grounding_index,
            train_pos={node: index for index, node in enumerate(nodes)},
            target_builder=EgoTargetBuilder(
                graph,
                f0,
                node_index,
                pool,
                slots=model_cfg.slots,
            ),
            sampler=sampler,
            s0=s0,
            rho_train=float(graph.number_of_edges() / (len(nodes) * (len(nodes) - 1) / 2)),
        )
        pack_dir = tmp_path / f"target-pack-{rank}"
        _write_tiny_token_pack(pack_dir, nodes)
        cfg = _toy_cfg(tmp_path)
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config={}),
            data=replace(cfg.data, pack_dir=pack_dir),
        )
        return (
            te._BatchFactory(
                e2e_cfg,
                model_cfg,
                data,
                node_batch=4,
                rank=rank,
                world_size=world_size,
            ),
            model_cfg,
        )

    def test_edge_targets_are_node_epoch_keyed_across_pairs_directions_and_ranks(
        self, tmp_path: Path
    ) -> None:
        factory, _ = self._target_factory(tmp_path)
        node = "target-node-00"
        first, _ = factory._edge_tensors(
            [(node, "target-node-01", 1), ("target-node-02", node, 1)],
            pad_to=2,
            epoch=3,
            step=1,
        )
        later, _ = factory._edge_tensors(
            [("target-node-03", node, 1), (node, "target-node-04", 1)],
            pad_to=2,
            epoch=3,
            step=99,
        )
        other_rank_factory, _ = self._target_factory(tmp_path, rank=3, world_size=4)
        other_rank, _ = other_rank_factory._edge_tensors(
            [(node, "target-node-05", 1)],
            pad_to=1,
            epoch=3,
            step=17,
        )

        expected = first["target_features_i"][0]
        torch.testing.assert_close(first["target_features_j"][1], expected)
        torch.testing.assert_close(later["target_features_j"][0], expected)
        torch.testing.assert_close(later["target_features_i"][1], expected)
        torch.testing.assert_close(other_rank["target_features_i"][0], expected)
        assert torch.equal(first["target_mask_i"][0], first["target_mask_j"][1])
        assert torch.equal(later["target_mask_j"][0], first["target_mask_i"][0])
        assert torch.equal(first["target_node_index_i"][0], first["target_node_index_j"][1])
        assert torch.equal(later["target_node_index_j"][0], first["target_node_index_i"][0])
        assert torch.equal(other_rank["target_node_index_i"][0], first["target_node_index_i"][0])

        next_epoch, _ = factory._edge_tensors(
            [(node, "target-node-01", 1)],
            pad_to=1,
            epoch=4,
            step=1,
        )
        assert not torch.equal(next_epoch["target_features_i"][0], expected)

    def test_edge_targets_cap_mask_and_exclude_negative_or_filler_rows(
        self, tmp_path: Path
    ) -> None:
        factory, model_cfg = self._target_factory(tmp_path)
        edge, true_rows = factory._edge_tensors(
            [
                ("target-node-00", "target-node-01", 1),
                ("target-node-02", "target-node-03", 0),
            ],
            pad_to=4,
            epoch=2,
            step=8,
        )

        assert true_rows == 2
        assert edge["target_mask_i"].shape == (4, 16)
        assert edge["target_features_i"].shape == (4, 16, model_cfg.input_dim)
        assert int(edge["target_mask_i"][0].sum()) == 16
        assert int(edge["target_mask_j"][0].sum()) == 1
        assert not edge["target_mask_j"][0, 1:].any()
        assert torch.count_nonzero(edge["target_features_j"][0, 1:]) == 0
        for name in (
            "target_mask_i",
            "target_mask_j",
            "target_mult_i",
            "target_mult_j",
            "target_adj_i",
            "target_adj_j",
            "target_features_i",
            "target_features_j",
        ):
            assert torch.count_nonzero(edge[name][1:]) == 0
        assert (edge["target_node_index_i"][0, edge["target_mask_i"][0]] >= 0).all()
        assert (edge["target_node_index_j"][0, edge["target_mask_j"][0]] >= 0).all()
        assert (edge["target_node_index_i"][0, ~edge["target_mask_i"][0]] == -1).all()
        assert (edge["target_node_index_j"][0, ~edge["target_mask_j"][0]] == -1).all()
        assert (edge["target_node_index_i"][1:] == -1).all()
        assert (edge["target_node_index_j"][1:] == -1).all()

    def test_nonedge_shared_neighbor_has_nonzero_relational_targets(
        self, tmp_path: Path
    ) -> None:
        factory, _ = self._target_factory(tmp_path)
        # Leaves 01 and 02 are a non-edge but share hub 00.
        edge, _ = factory._edge_tensors(
            [("target-node-01", "target-node-02", 0)],
            pad_to=2,
            epoch=1,
            step=1,
        )
        expected = torch.tensor([math.log1p(1.0), 1.0])
        torch.testing.assert_close(edge["rel_target"][0], expected)
        assert torch.count_nonzero(edge["rel_target"][0]) == 2
        assert torch.equal(edge["rel_target"][1], expected)
        assert edge["edge_mask"].tolist() == [1.0, 0.0]
        other_rank_factory, _ = self._target_factory(tmp_path, rank=1, world_size=2)
        reversed_edge, _ = other_rank_factory._edge_tensors(
            [("target-node-02", "target-node-01", 0)],
            pad_to=1,
            epoch=7,
            step=99,
        )
        assert torch.equal(reversed_edge["rel_target"][0], edge["rel_target"][0])

    def test_edge_target_memory_manifest_counts_both_dense_endpoint_gathers(
        self, tmp_path: Path
    ) -> None:
        factory, model_cfg = self._target_factory(tmp_path)
        rows_per_rank, steps = te._epoch_step_plan(
            len(factory._data.e_sup_positives),
            negative_ratio=factory._cfg.data.negative_ratio,
            edge_batch=factory._cfg.data.edge_batch,
            world_size=1,
        )
        batch = next(factory.epoch_batches(1, rows_per_rank=rows_per_rank, steps=steps))
        old_total = (
            batch.node["x"].shape[0]
            + batch.node["ground_x"].shape[0] * batch.node["ground_x"].shape[1]
            + batch.node["target_features"].shape[0] * batch.node["target_features"].shape[1]
            + 4 * batch.edge["x_i"].shape[0]
        )
        expected_delta = 2 * batch.edge["x_i"].shape[0] * model_cfg.slots
        assert batch.f0_rows_gathered == old_total + expected_delta

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

        # Task 13c: same-index-space grounding ids for both endpoints (spec
        # Sec 13.18) -- required by EgoStitchE2E's grounded-identity-match
        # flag, which compares ground_id_a/ground_id_b for equality.
        assert batch.edge["ground_id_i"].shape == (edge_batch, model_cfg.n_ground)
        assert batch.edge["ground_id_j"].shape == (edge_batch, model_cfg.n_ground)
        assert batch.edge["ground_id_i"].dtype == torch.int64
        expected_ids_i = data.grounding_index[[data.train_pos[u] for u, _, _ in expected_chunk]]
        expected_ids_j = data.grounding_index[[data.train_pos[v] for _, v, _ in expected_chunk]]
        np.testing.assert_array_equal(batch.edge["ground_id_i"].numpy(), expected_ids_i)
        np.testing.assert_array_equal(batch.edge["ground_id_j"].numpy(), expected_ids_j)
        # Same index space x_i/x_j are drawn from: gathering data.f0 at these
        # ids must reproduce the already-asserted ground_i/ground_j features.
        torch.testing.assert_close(data.f0[batch.edge["ground_id_i"]], batch.edge["ground_i"])

    def test_requires_pack_dir(self, tmp_path: Path) -> None:
        cfg = _toy_cfg(tmp_path)
        model_cfg = EgoStitchConfig.from_mapping(cfg.model.config)
        data = _toy_bundle(tmp_path, model_cfg)
        e2e_cfg = replace(cfg, model=ModelConfig(family="egostitch_e2e", config={}))
        with pytest.raises(ValueError, match="pack_dir"):
            te._BatchFactory(e2e_cfg, model_cfg, data, node_batch=4, rank=0, world_size=1)

    def test_prefetch_preserves_real_epoch_and_overfit_batches(self, tmp_path: Path) -> None:
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
        rows_per_rank, epoch_steps = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=e2e_cfg.data.negative_ratio,
            edge_batch=e2e_cfg.data.edge_batch,
            world_size=2,
        )
        manifest_rows = tuple(
            (
                _NODES[index % len(_NODES)],
                _NODES[(index + 1) % len(_NODES)],
                index % 2,
            )
            for index in range(10)
        )
        manifest = te.OverfitManifest(rows=manifest_rows, sha256="a" * 64)

        def assert_batches_equal(
            direct: list[te._CompositeBatch], prefetched: list[te._CompositeBatch]
        ) -> None:
            assert len(direct) == len(prefetched)
            for expected, actual in zip(direct, prefetched, strict=True):
                assert actual.edge_rows_true == expected.edge_rows_true
                assert actual.edge_rows_global == expected.edge_rows_global
                assert actual.f0_rows_gathered == expected.f0_rows_gathered
                assert actual.node.keys() == expected.node.keys()
                assert actual.edge.keys() == expected.edge.keys()
                for name in expected.node:
                    assert torch.equal(actual.node[name], expected.node[name])
                for name in expected.edge:
                    assert torch.equal(actual.edge[name], expected.edge[name])

        for rank in range(2):
            for mode in ("epoch", "overfit"):
                direct_factory = te._BatchFactory(
                    e2e_cfg, model_cfg, data, node_batch=4, rank=rank, world_size=2
                )
                if mode == "epoch":
                    direct_source = direct_factory.epoch_batches(
                        1, rows_per_rank=rows_per_rank, steps=epoch_steps
                    )
                else:
                    direct_source = direct_factory.fixed_row_batches(
                        manifest=manifest, epoch=1, steps=3, step_offset=5
                    )
                direct = list(direct_source)
                direct_state = (
                    direct_factory._node_cursor,
                    direct_factory._node_cycle,
                    direct_factory.training_nodes_read,
                    direct_factory.training_f0_rows_read,
                )

                prefetch_factory = te._BatchFactory(
                    e2e_cfg, model_cfg, data, node_batch=4, rank=rank, world_size=2
                )
                if mode == "epoch":
                    prefetch_source = prefetch_factory.epoch_batches(
                        1, rows_per_rank=rows_per_rank, steps=epoch_steps
                    )
                else:
                    prefetch_source = prefetch_factory.fixed_row_batches(
                        manifest=manifest, epoch=1, steps=3, step_offset=5
                    )
                prefetched = list(te._prefetch_batches(iter(prefetch_source), depth=2))
                prefetch_state = (
                    prefetch_factory._node_cursor,
                    prefetch_factory._node_cycle,
                    prefetch_factory.training_nodes_read,
                    prefetch_factory.training_f0_rows_read,
                )

                assert_batches_equal(direct, prefetched)
                assert prefetch_state == direct_state


class TestE2ECompositeStep:
    """One CPU optimizer forward through the active §13.19 composite."""

    def _batch_and_model(
        self, tmp_path: Path, *, w_rel: float | None = None
    ) -> tuple[te._CompositeBatch, EgoStitchE2E]:
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
        model_config = dict(e2e_cfg.model.config)
        if w_rel is not None:
            model_config["w_rel"] = w_rel
        model = EgoStitchE2E(E2EConfig.from_mapping(model_config))
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
            "edge_active": joint_weight != 0.0,
            "real_ssl_scale": torch.tensor(joint_weight),
            "edge_rows_global": batch.edge_rows_global,
            "seed": 0,
            "epoch": 1,
            "step": 0,
            "collect_diagnostics": collect_diagnostics,
        }

    @staticmethod
    def _float_the_packed_store(monkeypatch: pytest.MonkeyPatch) -> None:
        """Float the bf16-only packed store for CPU runs without autocast."""
        original_from_pack = PackedFeatureTable.from_pack.__func__

        def _float_cpu_pack(
            cls: type[PackedFeatureTable], root: Path, device: torch.device
        ) -> PackedFeatureTable:
            table = original_from_pack(cls, root, device)
            table.tokens = table.tokens.float()
            return table

        monkeypatch.setattr(PackedFeatureTable, "from_pack", classmethod(_float_cpu_pack))

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

    def test_warmstart_preserves_relational_losses_and_gradients(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        teacher_row = int(
            torch.nonzero(
                batch.edge["edge_mask"].bool() & ~batch.edge["is_self"],
                as_tuple=False,
            )[0].item()
        )
        batch.edge["label"][teacher_row] = 1.0
        composite = te._CompositeStep(model, world_size=1)
        captured_log_plans: list[torch.Tensor] = []
        captured_teacher_cells: list[torch.Tensor] = []
        original_sinkhorn = e2e_module.sinkhorn_log_plan
        original_teacher_cells = e2e_module.alignment_teacher_cells

        def _retaining_sinkhorn(*args: object, **kwargs: object) -> torch.Tensor:
            log_plan = original_sinkhorn(*args, **kwargs)
            log_plan.retain_grad()
            captured_log_plans.append(log_plan)
            return log_plan

        def _teacher_bearing_cells(*args: object, **kwargs: object) -> torch.Tensor:
            cells = original_teacher_cells(*args, **kwargs).clone()
            cells[teacher_row] = True
            captured_teacher_cells.append(cells)
            return cells

        monkeypatch.setattr(e2e_module, "sinkhorn_log_plan", _retaining_sinkhorn)
        monkeypatch.setattr(
            e2e_module, "alignment_teacher_cells", _teacher_bearing_cells
        )

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=0.0))
        loss = cast(torch.Tensor, out["loss"])
        parts = cast(dict[str, float], out["parts"])
        assert bool(torch.isfinite(loss))
        assert captured_teacher_cells
        assert bool(captured_teacher_cells[0][teacher_row, 0, 0])
        assert parts["edge"] == 0.0
        assert parts["recon_align"] > 0.0
        assert parts["recon_rel"] > 0.0
        loss.backward()  # type: ignore[no-untyped-call]

        ste_gradients = [
            parameter.grad
            for parameter in model.ste.parameters()
            if parameter.requires_grad
        ]
        assert ste_gradients
        assert all(gradient is not None for gradient in ste_gradients)
        assert any(
            bool(torch.count_nonzero(cast(torch.Tensor, gradient)))
            for gradient in ste_gradients
        )
        assert captured_log_plans
        assert all(log_plan.grad is not None for log_plan in captured_log_plans)
        assert any(
            bool(torch.count_nonzero(cast(torch.Tensor, log_plan.grad)))
            for log_plan in captured_log_plans
        )
        missing_from_graph = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing_from_graph == []
        parameter_groups = te.build_e2e_parameter_groups(model)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameter_groups.groups[name],
                    "lr": te._e2e_optimizer_group_lr(
                        1e-3,
                        te.E2EPhaseState("A", 0.0, False, 0.0),
                        name,
                        {"generator", "topology_content_conditioning"},
                    ),
                }
                for name in parameter_groups.groups
            ],
            weight_decay=0.01,
        )
        ste_before = [parameter.detach().clone() for parameter in model.ste.parameters()]
        assert model.rel_head is not None
        rel_before = [
            parameter.detach().clone() for parameter in model.rel_head.parameters()
        ]
        pair_before = [
            parameter.detach().clone()
            for parameter in parameter_groups.groups["pair_encoder_head"]
        ]
        optimizer.step()
        assert any(
            not torch.equal(before, after)
            for before, after in zip(ste_before, model.ste.parameters(), strict=True)
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(rel_before, model.rel_head.parameters(), strict=True)
        )
        assert all(
            torch.equal(before, after)
            for before, after in zip(
                pair_before,
                parameter_groups.groups["pair_encoder_head"],
                strict=True,
            )
        )

    def test_no_rel_head_survives_warmstart_step_and_family_probe(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path, w_rel=0.0)
        assert model.rel_head is None
        composite = te._CompositeStep(model, world_size=1)
        parameter_groups = te.build_e2e_parameter_groups(model)
        phase = te.E2EPhaseState("A", 0.0, False, 0.0)
        optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": 1e-3}
                for parameters in parameter_groups.groups.values()
            ]
        )

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=0.0))
        cast(torch.Tensor, out["loss"]).backward()  # type: ignore[no-untyped-call]
        active_groups = te._e2e_active_groups(phase, model)
        assert active_groups == {"generator"}
        records = te.e2e_check_and_clip_gradients(
            parameter_groups.groups,
            active_groups,
        )
        assert records["generator"].active
        assert not records["topology_content_conditioning"].active
        optimizer.step()

        accelerator = Accelerator(cpu=True)
        with self._bf16_autocast():
            family_norms, submodule_rms = te._e2e_family_probe(
                composite,
                self._payload(batch, joint_weight=0.0),
                parameter_groups.groups,
                phase,
                "no_l_rel",
                accelerator,
            )
        assert set(family_norms["generator"]) == {"recon"}
        assert family_norms["topology_content_conditioning"] == {}
        assert submodule_rms == {}

    @pytest.mark.parametrize(
        ("w_rel", "phase", "joint_weight", "expect_update"),
        [
            (0.0, te.E2EPhaseState("A", 0.0, False, 0.0), 0.0, False),
            (0.25, te.E2EPhaseState("A", 0.0, False, 0.0), 0.0, True),
            (0.0, te.E2EPhaseState("B", 0.1, True, 0.0), 1.0, True),
        ],
    )
    def test_topology_content_optimizer_updates_only_when_active(
        self,
        tmp_path: Path,
        w_rel: float,
        phase: te.E2EPhaseState,
        joint_weight: float,
        expect_update: bool,
    ) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path, w_rel=w_rel)
        composite = te._CompositeStep(model, world_size=1)
        parameter_groups = te.build_e2e_parameter_groups(model)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "lr": te._e2e_optimizer_group_lr(
                        1e-3,
                        phase,
                        name,
                        te._e2e_active_groups(phase, model),
                    ),
                }
                for name, parameters in parameter_groups.groups.items()
            ],
            weight_decay=0.01,
        )

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=joint_weight))
        cast(torch.Tensor, out["loss"]).backward()  # type: ignore[no-untyped-call]
        topology_parameters = parameter_groups.groups["topology_content_conditioning"]
        before = [parameter.detach().clone() for parameter in topology_parameters]
        optimizer.step()

        changed = [
            not torch.equal(previous, parameter)
            for previous, parameter in zip(before, topology_parameters, strict=True)
        ]
        assert any(changed) if expect_update else not any(changed)

    def test_no_rel_head_conditioning_liveness_resumes_when_edge_active(
        self, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path, w_rel=0.0)
        composite = te._CompositeStep(model, world_size=1)
        parameter_groups = te.build_e2e_parameter_groups(model)
        phase = te.E2EPhaseState("B", 0.1, True, 0.0)

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=1.0))
        cast(torch.Tensor, out["loss"]).backward()  # type: ignore[no-untyped-call]
        active_groups = te._e2e_active_groups(phase, model)
        assert "topology_content_conditioning" in active_groups
        records = te.e2e_check_and_clip_gradients(
            parameter_groups.groups,
            active_groups,
        )
        assert records["topology_content_conditioning"].active
        for parameter in parameter_groups.groups["topology_content_conditioning"]:
            parameter.grad = None
        with pytest.raises(
            RuntimeError,
            match="zero gradient norm in active E2E group 'topology_content_conditioning'",
        ):
            te.e2e_check_and_clip_gradients(parameter_groups.groups, active_groups)

    def test_rel_head_keeps_conditioning_liveness_guard_active_in_both_phases(
        self, tmp_path: Path
    ) -> None:
        _, model = self._batch_and_model(tmp_path, w_rel=0.25)
        assert model.rel_head is not None
        parameter_groups = te.build_e2e_parameter_groups(model)
        phases = (
            te.E2EPhaseState("A", 0.0, False, 0.0),
            te.E2EPhaseState("B", 0.1, True, 0.0),
        )
        for phase in phases:
            active_groups = te._e2e_active_groups(phase, model)
            assert "topology_content_conditioning" in active_groups
            with pytest.raises(
                RuntimeError,
                match="zero gradient norm in active E2E group 'topology_content_conditioning'",
            ):
                te.e2e_check_and_clip_gradients(
                    {
                        "topology_content_conditioning": parameter_groups.groups[
                            "topology_content_conditioning"
                        ]
                    },
                    {"topology_content_conditioning"},
                )

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

    def test_registered_precision_differential_thresholds(self) -> None:
        fp32_f = torch.full((3,), 100.0)
        fp32_residual = torch.tensor([0.1, 0.2, 0.3])
        fp32_full = fp32_f + fp32_residual

        metrics = te._validate_e2e_precision_outputs(
            fp32_f + fp32_residual * 1.04,
            fp32_f,
            fp32_full,
            fp32_f,
        )
        assert metrics["residual_relative_l2"] == pytest.approx(0.04, abs=1e-4)
        assert metrics["residual_correlation"] >= 0.999

        with pytest.raises(RuntimeError, match=r"residual relative L2 <= 0.05.*metrics="):
            te._validate_e2e_precision_outputs(
                fp32_f + fp32_residual * 1.06,
                fp32_f,
                fp32_full,
                fp32_f,
            )

    def test_vector_tolerance_admits_bf16_tail_noise(self) -> None:
        """Per-element BF16-trunk tail noise passes the vector bounds.

        Regression for the 2026-07-22 rehearsal failures: selected-checkpoint
        max abs logit error ~0.1 at ordinary magnitudes must pass while the
        residual contract is enforced unchanged. Per-element errors stay
        logged as diagnostics.
        """
        generator = torch.Generator().manual_seed(0)
        fp32_f = torch.randn(64, generator=generator) * 3.0
        fp32_residual = torch.randn(64, generator=generator) * 0.1
        fp32_full = fp32_f + fp32_residual
        noise = torch.zeros(64)
        noise[7] = 0.1045
        noise[23] = -0.0907

        metrics = te._validate_e2e_precision_outputs(
            fp32_full + noise,
            fp32_f + noise,
            fp32_full,
            fp32_f,
        )
        assert metrics["full_max_abs_error"] == pytest.approx(0.1045, abs=1e-6)
        assert metrics["full_relative_l2"] < 5e-2
        assert metrics["f_logit_relative_l2"] < 5e-2

    def test_vector_tolerance_rejects_common_mode_corruption(self) -> None:
        """A bias on both paths cancels in the residual but must still fail."""
        generator = torch.Generator().manual_seed(1)
        fp32_f = torch.randn(64, generator=generator) * 3.0
        fp32_residual = torch.randn(64, generator=generator) * 0.1
        fp32_full = fp32_f + fp32_residual
        bias = torch.full((64,), 1.0)

        with pytest.raises(RuntimeError, match=r"full relative L2 <= 0.05.*metrics="):
            te._validate_e2e_precision_outputs(
                fp32_full + bias,
                fp32_f + bias,
                fp32_full,
                fp32_f,
            )

    def test_vector_tolerance_rejects_gross_single_element_corruption(self) -> None:
        """One logit-scale corrupted element must still fail the vector bound."""
        generator = torch.Generator().manual_seed(2)
        fp32_f = torch.randn(64, generator=generator) * 3.0
        fp32_residual = torch.randn(64, generator=generator) * 0.1
        fp32_full = fp32_f + fp32_residual
        corruption = torch.zeros(64)
        corruption[11] = 5.0

        with pytest.raises(RuntimeError, match=r"full relative L2 <= 0.05.*metrics="):
            te._validate_e2e_precision_outputs(
                fp32_full + corruption,
                fp32_f,
                fp32_full,
                fp32_f,
            )

    def test_family_probe_requests_family_tensors_from_standard_payload(
        self, tmp_path: Path
    ) -> None:
        """The step-50 replay must work when the training payload disables diagnostics."""
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        groups = te.build_e2e_parameter_groups(model).groups
        payload = self._payload(batch, joint_weight=0.0, collect_diagnostics=False)
        accelerator = Accelerator(cpu=True)

        with self._bf16_autocast():
            family_norms, submodule_rms = te._e2e_family_probe(
                composite,
                payload,
                groups,
                te.e2e_phase_state(0, 100),
                "full",
                accelerator,
            )

        assert family_norms["pair_encoder_head"] == {}
        assert set(family_norms["generator"]) == {"recon"}
        assert set(family_norms["topology_content_conditioning"]) == {"recon"}
        assert family_norms["topology_content_conditioning"]["recon"] > 0.0
        assert submodule_rms == {}
        assert payload["collect_diagnostics"] is False

    def test_family_probe_does_not_mutate_conditioning_ema(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        groups = te.build_e2e_parameter_groups(model).groups
        payload = self._payload(batch, joint_weight=1.0, collect_diagnostics=False)
        accelerator = Accelerator(cpu=True)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, GatedCrossAttention):
                    module.gate.fill_(0.7)
        ema_before = {
            name: buffer.clone()
            for name, buffer in model.named_buffers()
            if name.endswith(("ema_mu", "ema_updates"))
        }

        with self._bf16_autocast():
            te._e2e_family_probe(
                composite,
                payload,
                groups,
                te.e2e_phase_state(50, 100),
                "full",
                accelerator,
            )

        ema_after = {
            name: buffer
            for name, buffer in model.named_buffers()
            if name.endswith(("ema_mu", "ema_updates"))
        }
        assert ema_after.keys() == ema_before.keys()
        assert all(torch.equal(ema_after[name], value) for name, value in ema_before.items())

    def test_family_probe_releases_each_forward_before_the_next(self, tmp_path: Path) -> None:
        """The probe's four passes must not double-buffer their autograd graphs.

        Every family gets its own full forward, and the returned ``families``
        dict pins the graph reachable from *each* family in it -- so holding one
        pass's dict across the next pass keeps two full forwards resident, plus
        the three subgraphs that pass never reads. That double-buffering, not the
        plain training step, is what made this probe the run's memory peak.
        """
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        groups = te.build_e2e_parameter_groups(model).groups
        payload = self._payload(batch, joint_weight=1.0, collect_diagnostics=False)
        accelerator = Accelerator(cpu=True)
        # A step-0 model has every `gate` at zero, which severs the
        # edge -> generator path and trips the probe's own liveness check before
        # it reaches a second family. Open the gates so all four families run,
        # which is the state the training-time probe actually observes.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, GatedCrossAttention):
                    module.gate.fill_(0.7)

        class _ReleaseSpy(torch.nn.Module):
            def __init__(self, inner: te._CompositeStep) -> None:
                super().__init__()
                self.inner = inner
                self.model = inner.model
                self.previous: dict[str, weakref.ReferenceType[torch.Tensor]] = {}
                self.survivors: list[str] = []
                self.forwards = 0

            def forward(self, values: dict[str, object]) -> dict[str, object]:
                gc.collect()
                self.survivors.extend(
                    name for name, ref in self.previous.items() if ref() is not None
                )
                out = cast(dict[str, object], self.inner(values))
                families = cast(dict[str, torch.Tensor], out.get("families", {}))
                self.previous = {
                    name: weakref.ref(tensor) for name, tensor in families.items()
                }
                self.forwards += 1
                return out

        spy = _ReleaseSpy(composite)
        with self._bf16_autocast():
            te._e2e_family_probe(
                spy,
                payload,
                groups,
                te.e2e_phase_state(50, 100),
                "full",
                accelerator,
            )

        # Phase C activates edge + recon + real + ssl, so there is more than one
        # forward for the previous one to have been held across.
        assert spy.forwards == 4
        assert spy.survivors == []

    def test_family_probe_still_fails_a_dead_group_by_default(self, tmp_path: Path) -> None:
        """The budget probe's relaxation must stay opt-in.

        A step-0 model has every `gate` at zero, so `tanh(gate) == 0` severs the
        edge -> generator path -- exactly the dead-path shape the training-time
        probe has to abort on. `require_live_gradients` defaults to True so that
        only `_probe_family_peak`, which reads no norms, ever sees it relaxed.
        """
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        groups = te.build_e2e_parameter_groups(model).groups
        payload = self._payload(batch, joint_weight=1.0, collect_diagnostics=False)
        accelerator = Accelerator(cpu=True)

        with (
            self._bf16_autocast(),
            pytest.raises(RuntimeError, match="invalid E2E fixed-replay norm"),
        ):
            te._e2e_family_probe(
                composite,
                payload,
                groups,
                te.e2e_phase_state(50, 100),
                "full",
                accelerator,
            )

    def _probe_mode_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[te.EgoConfig, te.EgoStitchData, EgoStitchE2E]:
        # `_run_probe_mode` drives the model without Accelerate's bf16
        # autocast on CPU, so the bf16-only packed store must be floated first.
        self._float_the_packed_store(monkeypatch)
        cfg = _toy_cfg(tmp_path)
        registered = te.load_config(
            Path(__file__).resolve().parents[1] / "configs/egostitch_e2e_breadth_first.yaml"
        )
        assert registered.training is not None
        assert registered.runtime is not None
        pack_dir = tmp_path / "token_pack"
        _write_tiny_token_pack(pack_dir, _NODES, min_length=3)
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
            data=replace(cfg.data, pack_dir=pack_dir),
            runtime=replace(registered.runtime, probe_warmup_steps=1, probe_timed_steps=1),
            training=registered.training,
        )
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        model = EgoStitchE2E(E2EConfig.from_mapping(dict(_E2E_TINY_MODEL)))
        return e2e_cfg, data, model

    def test_budget_probe_measures_the_family_probe_peak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`memory_limit_gib` must be compared against a peak that includes the probe.

        `_run_probe_mode`'s timed loop only executes plain optimizer steps, but
        training additionally runs `_e2e_family_probe` every
        `gradient_probe_interval` steps -- up to four more full forward/backward
        passes. A candidate admitted on the plain-step peak is admitted on a
        number the run never reaches, which is how a 67.1 GiB prediction cleared
        an 85.0 GiB limit and then allocated 92.7 GiB.
        """
        torch.manual_seed(0)
        e2e_cfg, data, model = self._probe_mode_inputs(tmp_path, monkeypatch)
        seen: list[tuple[te.E2EPhaseState, dict[str, object], bool]] = []
        original = te._e2e_family_probe

        def _record(
            wrapped: torch.nn.Module,
            payload: dict[str, object],
            groups: Mapping[str, Sequence[torch.nn.Parameter]],
            phase: te.E2EPhaseState,
            arm: te.E2EArmName,
            accelerator: Accelerator,
            *,
            require_live_gradients: bool = True,
        ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
            seen.append((phase, payload, require_live_gradients))
            return original(
                wrapped,
                payload,
                groups,
                phase,
                arm,
                accelerator,
                require_live_gradients=require_live_gradients,
            )

        monkeypatch.setattr(te, "_e2e_family_probe", _record)
        profile_output = tmp_path / "probe.json"

        te._run_probe_mode(
            model,
            e2e_cfg,
            data,
            Accelerator(cpu=True),
            node_batch=4,
            profile_output=profile_output,
        )

        assert len(seen) == 1, "the budget probe must fold in exactly one family probe"
        phase, payload, require_live_gradients = seen[0]
        # Phase C is the worst case: every family is live at once.
        assert (phase.alpha, phase.edge_active, phase.real_ssl_scale) == (1.0, True, 1.0)
        assert payload["edge_active"] is True
        # Step-0 gates are zero, so liveness is relaxed -- allocation is not.
        assert require_live_gradients is False
        assert json.loads(profile_output.read_text())["valid"] is True

    def test_budget_probe_reports_a_family_probe_oom_as_a_candidate_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An OOM in the folded-in probe must fail the candidate, not the whole sweep."""
        torch.manual_seed(0)
        e2e_cfg, data, model = self._probe_mode_inputs(tmp_path, monkeypatch)

        def _oom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("CUDA out of memory. Tried to allocate 952.00 MiB")

        monkeypatch.setattr(te, "_e2e_family_probe", _oom)

        with pytest.raises(RuntimeError, match="out of memory"):
            te._run_probe_mode(
                model,
                e2e_cfg,
                data,
                Accelerator(cpu=True),
                node_batch=4,
                profile_output=tmp_path / "probe.json",
            )

        assert "E2_PROBE_CANDIDATE_FAILURE" in capsys.readouterr().err

    def test_profile_loop_executes_real_optimizer_and_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the activated trainer, not only its loss helper."""
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        cfg = _toy_cfg(tmp_path)
        registered = te.load_config(
            Path(__file__).resolve().parents[1] / "configs/egostitch_e2e_breadth_first.yaml"
        )
        assert registered.training is not None
        pack_dir = tmp_path / "token_pack"
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
            data=replace(cfg.data, pack_dir=pack_dir),
            training=replace(
                registered.training,
                clip_immediate_abort=1e-12,
                clip_persistent_threshold=0.0,
            ),
            run_kind="rehearsal",
        )
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        self._float_the_packed_store(monkeypatch)
        guard_persistence: list[bool] = []
        original_guard_update = te.E2EClipGuard.update

        def _count_guard_calls(
            guard: te.E2EClipGuard,
            records: dict[str, te.E2EGradientGroupRecord],
            *,
            step: int | None = None,
            phase: te.E2EPhaseName | None = None,
            enforce_persistent: bool = True,
        ) -> None:
            guard_persistence.append(enforce_persistent)
            original_guard_update(
                guard,
                records,
                step=step,
                phase=phase,
                enforce_persistent=enforce_persistent,
            )

        monkeypatch.setattr(te.E2EClipGuard, "update", _count_guard_calls)
        accelerator = Accelerator(cpu=True)

        result = te._train_e2e_stability_loop(
            model,
            e2e_cfg,
            data,
            accelerator,
            node_batch=4,
            profile_only=True,
        )

        assert result.runtime_profile is not None
        assert result.runtime_profile["total_optimizer_steps"] > 0
        assert (
            len(result.runtime_profile["optimizer_step_gradients"])
            == result.runtime_profile["total_optimizer_steps"]
        )
        optimizer_steps = result.runtime_profile["optimizer_step_gradients"]
        phases = [step["phase"] for step in optimizer_steps]
        _, steps_per_epoch = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=e2e_cfg.data.negative_ratio,
            edge_batch=e2e_cfg.data.edge_batch,
            world_size=1,
        )
        schedule_steps = steps_per_epoch * e2e_cfg.optim.epochs
        assert phases == [
            te.e2e_phase_state(step, schedule_steps).phase for step in range(len(phases))
        ]
        assert len(phases) == steps_per_epoch
        assert result.runtime_profile["schedule_total_optimizer_steps"] == schedule_steps
        assert result.runtime_profile["phase_boundaries"] == {
            "phase_a_end": te.e2e_phase_boundaries(schedule_steps)[0],
            "phase_b_end": te.e2e_phase_boundaries(schedule_steps)[1],
        }
        for group, ceiling in (
            ("pair_encoder_head", 3.0),
            ("generator", 3.0),
            ("topology_content_conditioning", 1.0),
        ):
            record = next(
                step["optimizer_group_gradients"][group]
                for step in optimizer_steps
                if step["optimizer_group_gradients"][group]["active"]
            )
            assert record["clip_coefficient"] == pytest.approx(
                min(1.0, ceiling / (record["norm"] + 1e-12))
            )
        assert (
            result.runtime_profile["observed_training_access"][0]["all_nodes_within_v_fit"] is True
        )
        assert result.runtime_profile["validation_coverage_exact"] is True
        assert result.runtime_profile["training_coverage_exact"] is True
        validation_record = cast(dict[str, float], result.history[0]["fidelity"])
        for key in (
            "pi_slot_std",
            "h_pairwise_cosine_mean",
            "adj_offdiag_std",
            "plan_row_entropy",
            "plan_rank1_marginal_residual",
            "topology_delta_degree_correlation",
        ):
            assert key in validation_record
            assert math.isfinite(validation_record[key])
        assert result.best_epoch == 1
        assert guard_persistence == [False] * len(optimizer_steps)

    @pytest.mark.parametrize(
        ("reference_auprc", "expect_eligible"),
        [(0.23, True), (0.22, True), (0.21, False)],
    )
    def test_training_loop_captures_first_edge_active_eligibility_reference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        reference_auprc: float,
        expect_eligible: bool,
    ) -> None:
        """Phase-A validation is ignored; the first trained-head validation stays fail-closed."""
        torch.manual_seed(0)
        _, model = self._batch_and_model(tmp_path)
        cfg = _toy_cfg(tmp_path)
        registered = te.load_config(
            Path(__file__).resolve().parents[1] / "configs/egostitch_e2e_breadth_first.yaml"
        )
        assert registered.training is not None
        pack_dir = tmp_path / "token_pack"
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
            data=replace(cfg.data, pack_dir=pack_dir),
            optim=replace(cfg.optim, epochs=5),
            diagnostics=replace(cfg.diagnostics, gradient_probe_interval=10_000),
            training=registered.training,
            run_kind="rehearsal",
        )
        data = _toy_bundle(tmp_path, EgoStitchConfig())
        original_from_pack = PackedFeatureTable.from_pack.__func__

        def _float_cpu_pack(
            cls: type[PackedFeatureTable], root: Path, device: torch.device
        ) -> PackedFeatureTable:
            table = original_from_pack(cls, root, device)
            table.tokens = table.tokens.float()
            return table

        validation_calls = 0

        def _validation(
            *_args: object,
            **_kwargs: object,
        ) -> te._ValidationResult:
            nonlocal validation_calls
            call = validation_calls
            validation_calls += 1
            f_logit_auprc = reference_auprc if call == 2 else (0.1 if call < 2 else 0.9)
            metrics = te.EdgeMetrics(
                auroc=0.7,
                auprc=0.3,
                accuracy=0.7,
                sensitivity=0.7,
                specificity=0.7,
                precision=0.7,
                recall=0.7,
                f1=0.7,
                mcc=0.4,
                ece=0.1,
                brier=0.1,
                threshold=0.5,
                n_pos=2,
                n_neg=2,
            )
            return te._ValidationResult(
                metrics=metrics,
                fidelity={
                    "prevalence": 0.2,
                    "active_logit_std": 0.2,
                    "clustering_mmd": 0.1,
                    "topology_delta_ratio": 0.01,
                    "f_logit_std": 0.4 if call == 0 else 0.8,
                    "f_logit_auprc": f_logit_auprc,
                    "h_pairwise_cosine_mean": 0.2,
                    "plan_rank1_marginal_residual": 0.2,
                },
            )

        seen_records: list[te.E2ECheckpointRecord] = []
        original_eligible = te.e2e_checkpoint_eligible

        def _track_eligibility(record: te.E2ECheckpointRecord, arm: te.E2EArmName) -> bool:
            seen_records.append(record)
            return original_eligible(record, arm)

        monkeypatch.setattr(PackedFeatureTable, "from_pack", classmethod(_float_cpu_pack))
        monkeypatch.setattr(te, "_validate_epoch", _validation)
        monkeypatch.setattr(te, "e2e_checkpoint_eligible", _track_eligibility)
        monkeypatch.setattr(te.E2EClipGuard, "update", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(te, "_e2e_precision_differential", lambda *_args, **_kwargs: {})
        accelerator = Accelerator(cpu=True)

        if expect_eligible:
            result = te._train_e2e_stability_loop(
                model,
                e2e_cfg,
                data,
                accelerator,
                node_batch=4,
            )
            assert result.best_epoch > 0
        else:
            with pytest.raises(RuntimeError, match="no eligible checkpoint"):
                te._train_e2e_stability_loop(
                    model,
                    e2e_cfg,
                    data,
                    accelerator,
                    node_batch=4,
                )

        phase_a_records = [record for record in seen_records if record.phase == "A"]
        assert phase_a_records
        assert all(record.warm_reference_auprc is None for record in phase_a_records)
        assert all(record.warm_reference_std == pytest.approx(0.4) for record in phase_a_records)
        first_referenced = next(
            record for record in seen_records if record.warm_reference_auprc is not None
        )
        assert first_referenced.warm_reference_std == pytest.approx(0.4)
        assert first_referenced.warm_reference_auprc == pytest.approx(reference_auprc)
        assert first_referenced.full_joint_epochs_completed == 0
        assert not original_eligible(first_referenced, "full")
        first_full_joint = next(
            record for record in seen_records if record.full_joint_epochs_completed == 1
        )
        assert first_full_joint.warm_reference_auprc == pytest.approx(reference_auprc)
        assert original_eligible(first_full_joint, "full") is expect_eligible

    def test_permanent_null_matches_eval_bypass(self, tmp_path: Path) -> None:
        """The training mask is exactly the corresponding hard eval bypass."""
        batch, model = self._batch_and_model(tmp_path)
        model.eval()
        batch.edge["emb_a"] = batch.edge["emb_a"].float()
        batch.edge["emb_b"] = batch.edge["emb_b"].float()
        edge_view = te._e2e_edge_view(batch.edge)
        for null, key in (("all_head", "f_logit"), ("content_head", "pair_topology")):
            model.cfg = replace(model.cfg, permanent_null=null)
            expected = model.decompose(edge_view)[key]
            seen: list[torch.Tensor] = []
            original = model.forward

            def _capture(
                payload: dict[str, torch.Tensor],
                *,
                masks: HeadNullMasks | None = None,
                _original: Callable[..., dict[str, torch.Tensor]] = original,
                _seen: list[torch.Tensor] = seen,
            ) -> dict[str, torch.Tensor]:
                output = _original(payload, masks=masks)
                _seen.append(output["logits"])
                return output

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(model, "forward", _capture)
            try:
                te._CompositeStep(model, world_size=1)(self._payload(batch, joint_weight=1.0))
            finally:
                monkeypatch.undo()
            assert len(seen) == 1
            torch.testing.assert_close(seen[0], expected)

    def test_telemetry_keys_present_in_metrics_row(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        batch, model = self._batch_and_model(tmp_path)
        composite = te._CompositeStep(model, world_size=1)
        topo_gate, cont_gate = self._gates(model)
        with torch.no_grad():
            topo_gate.gate.fill_(0.25)
            cont_gate.gate.fill_(0.25)

        with self._bf16_autocast():
            out = composite(self._payload(batch, joint_weight=1.0, collect_diagnostics=True))
        assert bool(torch.isfinite(cast(torch.Tensor, out["loss"])))
        parts = cast(dict[str, float], out["parts"])
        assert "recon_align" in parts
        assert "recon_rel" in parts
        assert math.isfinite(parts["recon_align"])
        assert math.isfinite(parts["recon_rel"])

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
            assert cast(float, metrics_row[key]) > 0.0

        with self._bf16_autocast():
            metrics_row.update(te._e2e_topology_delta_std(model, te._e2e_edge_view(batch.edge)))
        assert math.isfinite(cast(float, metrics_row["topology_delta_std"]))

    def test_topology_fidelity_reuses_one_pair_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch, model = self._batch_and_model(tmp_path)
        edge = te._e2e_edge_view(batch.edge)
        model.eval()
        with self._bf16_autocast(), torch.no_grad():
            reference = model.decompose(edge)
        reference_delta = (reference["full"] - reference["f_logit"]).detach()
        expected_delta_std = (
            0.0 if reference_delta.numel() < 2 else float(torch.std(reference_delta))
        )
        expected_f_logit_std = float(np.std(reference["f_logit"].detach().float().cpu().numpy()))
        expected = {
            "topology_delta_std": expected_delta_std,
            "f_logit_std": expected_f_logit_std,
            "topology_delta_ratio": expected_delta_std / max(expected_f_logit_std, 1e-30),
        }

        original_build = model.build_pair_context
        original_score = model.score_pair_context
        context_calls = 0
        score_calls = 0

        def _spy_build(
            pair_batch: dict[str, torch.Tensor],
            *,
            need_topo: bool = True,
            need_cont: bool = True,
        ) -> E2EPairContext:
            nonlocal context_calls
            context_calls += 1
            return original_build(pair_batch, need_topo=need_topo, need_cont=need_cont)

        def _spy_score(
            context: E2EPairContext,
            *,
            masks: HeadNullMasks | None = None,
        ) -> torch.Tensor:
            nonlocal score_calls
            score_calls += 1
            return original_score(context, masks=masks)

        monkeypatch.setattr(model, "build_pair_context", _spy_build)
        monkeypatch.setattr(model, "score_pair_context", _spy_score)
        with self._bf16_autocast():
            fidelity = te._e2e_topology_fidelity(model, edge)

        assert context_calls == 1
        assert score_calls == 2
        assert fidelity == expected

    def test_dead_decision_head_excluded_from_trainable_parameters(self, tmp_path: Path) -> None:
        _, model = self._batch_and_model(tmp_path)
        trainable_ids = {id(p) for p in te._e2e_trainable_parameters(model)}
        decision_ids = {
            name: id(parameter) for name, parameter in model.generator.decision.named_parameters()
        }
        assert trainable_ids & set(decision_ids.values()) == {decision_ids["tau_kappa_raw"]}
        assert any(id(p) in trainable_ids for p in model.trunk.parameters())


class _ArchivedV1TrainLoopE2E:
    """Full `train_egostitch_ddp_loop` run, family egostitch_e2e (Task 13c).

    Uses ``mixed_precision="no"`` (matching the `AcceleratorState` process-
    global singleton `TestTrainLoop` already initializes in this same test
    process -- accelerate hard-rejects a later `Accelerator(...)` call that
    requests a different `mixed_precision`) and reproduces the packed token
    store's bf16 requirement with an explicit `torch.autocast` wrapped around
    the call instead, exactly the same substitution `TestCompositeStepE2E`'s
    `_bf16_autocast()` already uses to drive `_CompositeStep` directly.
    """

    @staticmethod
    def _bf16_autocast() -> torch.autocast:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    def _e2e_setup(
        self, tmp_path: Path
    ) -> tuple[te.EgoConfig, te.EgoStitchData, EgoStitchE2E, Accelerator]:
        torch.manual_seed(0)
        model_cfg = EgoStitchConfig()
        cfg = _toy_cfg(tmp_path)
        data = _toy_bundle(tmp_path, model_cfg)
        pack_dir = tmp_path / "token_pack"
        _write_tiny_token_pack(pack_dir, _NODES, min_length=3)
        e2e_cfg = replace(
            cfg,
            model=ModelConfig(family="egostitch_e2e", config=dict(_E2E_TINY_MODEL)),
            data=replace(cfg.data, pack_dir=pack_dir),
            optim=replace(cfg.optim, epochs=1),
        )
        model = EgoStitchE2E(E2EConfig.from_mapping(e2e_cfg.model.config))
        accelerator = Accelerator(mixed_precision="no", cpu=True)
        return e2e_cfg, data, model, accelerator

    def _run(
        self, tmp_path: Path
    ) -> tuple[te.EgoConfig, te.EgoStitchData, EgoStitchE2E, te.EgoTrainResult]:
        e2e_cfg, data, model, accelerator = self._e2e_setup(tmp_path)
        with self._bf16_autocast():
            result = te.train_egostitch_ddp_loop(
                model, e2e_cfg, data, accelerator, node_batch=e2e_cfg.data.node_batch
            )
        return e2e_cfg, data, model, result

    def test_full_loop_contracts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_param_ids: set[int] = set()
        original_adamw = torch.optim.AdamW

        def _spy_adamw(params: object, **kwargs: object) -> torch.optim.AdamW:
            materialized = list(cast(Any, params))
            captured_param_ids.update(id(parameter) for parameter in materialized)
            return original_adamw(materialized, **kwargs)  # type: ignore[arg-type]

        def _activate_once(
            monitor: te._GradientImbalanceMonitor, step: int, norms: dict[str, float]
        ) -> bool:
            del norms
            if monitor.activated_step is not None:
                return False
            monitor.activated_step = step
            return True

        monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)
        monkeypatch.setattr(te._GradientImbalanceMonitor, "update", _activate_once)
        cfg, data, model, result = self._run(tmp_path)
        assert [int(cast(float, row["epoch"])) for row in result.history] == list(
            range(1, cfg.optim.epochs + 1)
        )
        for row in result.history:
            assert "auprc" in row
            assert math.isfinite(cast(float, row["auprc"]))
            fidelity = cast(dict[str, float], row["fidelity"])
            assert math.isfinite(fidelity["topology_delta_std"])
            assert math.isfinite(fidelity["topology_delta_ratio"])

        expected_ids = {id(parameter) for parameter in te._e2e_trainable_parameters(model)}
        assert expected_ids <= captured_param_ids
        assert len(captured_param_ids - expected_ids) == 4
        decision_ids = {
            name: id(parameter) for name, parameter in model.generator.decision.named_parameters()
        }
        assert captured_param_ids & set(decision_ids.values()) == {decision_ids["tau_kappa_raw"]}
        assert result.kendall_state["active"] is True
        log_variances = cast(dict[str, float], result.kendall_state["log_variances"])
        assert set(log_variances) == {"edge", "recon", "real", "ssl"}
        assert any(abs(value) > 0.0 for value in log_variances.values())

        te.write_run_start_metadata(cfg, data, world_size=1)
        te.write_outputs(result, cfg, data)
        payload = torch.load(cfg.output_dir / "best.pt", weights_only=False)
        assert payload["model_family"] == "egostitch_e2e"
        restored = E2EConfig.from_mapping(cast(dict[str, object], payload["model_config"]))
        assert restored == E2EConfig.from_mapping(cfg.model.config)

    def test_validation_uses_permanent_null_active_arm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for permanent_null in ("all_head", "content_head"):
            arm_path = tmp_path / permanent_null
            arm_path.mkdir()
            e2e_cfg, data, model, accelerator = self._e2e_setup(arm_path)
            e2e_cfg = replace(
                e2e_cfg,
                model=ModelConfig(
                    family="egostitch_e2e",
                    config={**_E2E_TINY_MODEL, "permanent_null": permanent_null},
                ),
            )
            model = EgoStitchE2E(E2EConfig.from_mapping(e2e_cfg.model.config))
            factory = te._BatchFactory(
                e2e_cfg,
                model.generator_cfg,
                data,
                node_batch=e2e_cfg.data.node_batch,
                rank=0,
                world_size=1,
            )
            rows, steps = te._epoch_step_plan(
                len(data.e_sup_positives),
                negative_ratio=e2e_cfg.data.negative_ratio,
                edge_batch=e2e_cfg.data.edge_batch,
                world_size=1,
            )
            next(iter(factory.epoch_batches(1, rows_per_rank=rows, steps=steps)))
            seen: list[HeadNullMasks | None] = []
            original = EgoStitchE2E.forward

            def _spy(
                self: EgoStitchE2E,
                batch: dict[str, torch.Tensor],
                *,
                masks: HeadNullMasks | None = None,
                _seen: list[HeadNullMasks | None] = seen,
                _original: Callable[..., dict[str, torch.Tensor]] = original,
            ) -> dict[str, torch.Tensor]:
                _seen.append(masks)
                return _original(self, batch, masks=masks)

            monkeypatch.setattr(EgoStitchE2E, "forward", _spy)
            try:
                with self._bf16_autocast():
                    validation = te._validate_epoch(
                        model,
                        data,
                        accelerator,
                        edge_batch=e2e_cfg.data.edge_batch,
                        topk_fraction=e2e_cfg.diagnostics.topk_fraction,
                        token_table=factory._token_table,
                        token_node_index=factory._token_node_index,
                    )
            finally:
                monkeypatch.undo()
            assert validation is not None
            assert seen and seen[0] is not None
            assert bool((~seen[0].topo).all()) == (permanent_null == "all_head")
            assert bool((~seen[0].cont).all()) == (permanent_null in ("all_head", "content_head"))
            assert validation.fidelity["selection_tiebreak"] == 0.0


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


# --------------------------------------------------------------------------- e2e full pipeline
#
# Family `egostitch_e2e` sources `n_ground` from `E2EConfig` (spec Sec 14.4.4;
# it supersedes the internal generator's own pinned `EgoStitchConfig` default
# for this family). This fixture pins it explicitly to the pre-rev-3.1
# default (20) rather than the new rev-3.1 default (50), so `n_ground` stays
# comfortably below `build_grounding_pool`'s `n_ground <= len(train_nodes) -
# 1` bound without needing more than the 25-node pipeline universe.

_E2E_PIPELINE_NODES = [f"g{i}" for i in range(25)]
_E2E_PIPELINE_N_GROUND = 20


def _e2e_pipeline_benchmark() -> Benchmark:
    """Build a self-contained `Benchmark` for the e2e pipeline fixture.

    All 25 nodes sit on the train side (a cycle graph), built directly as
    dataclasses -- no disk I/O, so tests that only need
    `prepare_pack`/`assemble_egostitch_data`'s own logic (not
    `load_benchmark`'s file parsing) can monkeypatch `_load_benchmark_for`
    with it directly.
    """
    nodes = _E2E_PIPELINE_NODES
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
    graph.add_edges_from(edges)
    positive_edges = frozenset(canonical_pair(u, v) for u, v in edges)
    pairs_sorted = sorted(canonical_pair(u, v) for u, v in edges)
    train_pairs = LabeledPairs(pairs=pairs_sorted, labels=np.ones(len(pairs_sorted), dtype=np.int8))
    val_pairs = LabeledPairs(
        pairs=[canonical_pair(nodes[0], nodes[5]), canonical_pair(nodes[1], nodes[6])],
        labels=np.array([1, 0], dtype=np.int8),
    )
    split = SplitArtifacts(
        strategy="toy",
        train_nodes=frozenset(nodes),
        test_nodes=frozenset(),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=LabeledPairs(pairs=[], labels=np.array([], dtype=np.int8)),
        train_graph=graph.copy(),
        test_graph=nx.Graph(),
        buckets={},
    )
    return Benchmark(root=Path("unused"), graph=graph, positive_edges=positive_edges, split=split)


def _write_e2e_feature_root(tmp_path: Path, nodes: list[str], *, input_dim: int = 1536) -> None:
    """Write a real, minimal, disk-backed `FeatureStore` root for `nodes`.

    Written under ``tmp_path / "data" / "features" / "frozen_node_features_1024"``,
    the exact location `_FEATURES_SUBDIR` resolves against `cfg.data.root`.
    """
    root = tmp_path / "data" / "features" / "frozen_node_features_1024"
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    index: dict[str, str] = {}
    for node in nodes:
        tensor = torch.from_numpy(rng.normal(size=(3, input_dim)).astype(np.float32))
        rel = f"embeddings/{node}.pt"
        torch.save(tensor, root / rel)
        index[node] = rel
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": input_dim, "max_sequence_length": 1024}
        )
    )


class TestPrepareAndAssembleE2E:
    """config load -> prepare_pack -> assemble_egostitch_data, family egostitch_e2e."""

    def _e2e_cfg(self, tmp_path: Path) -> te.EgoConfig:
        cfg = _toy_cfg(tmp_path)
        return replace(
            cfg,
            model=ModelConfig(
                family="egostitch_e2e", config={"n_ground": _E2E_PIPELINE_N_GROUND}
            ),
            data=replace(cfg.data, pack_dir=tmp_path / "raw-token-pack"),
        )

    def test_prepare_pack_uses_e2e_config_n_ground(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.data.packed_features as packed_features

        monkeypatch.setattr(packed_features, "ProcessPoolExecutor", ThreadPoolExecutor)
        _write_e2e_feature_root(tmp_path, _E2E_PIPELINE_NODES)
        benchmark = _e2e_pipeline_benchmark()
        monkeypatch.setattr(te, "_load_benchmark_for", lambda cfg: benchmark)
        original_load = torch.load
        source_load_counts: dict[str, int] = {}

        def counting_load(path: Path, *, map_location: str, weights_only: bool) -> object:
            if path.suffix == ".pt" and path.parent.name == "embeddings":
                source_load_counts[path.name] = source_load_counts.get(path.name, 0) + 1
            return original_load(path, map_location=map_location, weights_only=weights_only)

        monkeypatch.setattr(torch, "load", counting_load)
        e2e_cfg = self._e2e_cfg(tmp_path)
        pack_dir = tmp_path / "pack"

        payload = te.prepare_pack(e2e_cfg, pack_dir, cold_cache=True)
        manifest = cast(dict[str, object], payload["pack_manifest"])
        assert manifest["family"] == "egostitch_e2e"
        assert manifest["n_ground"] == _E2E_PIPELINE_N_GROUND
        packs = cast(dict[str, dict[str, object]], payload["packs"])
        assert set(packs) == {"f0_grounding", "raw_tokens"}
        assert packs["f0_grounding"]["cold"] is True
        assert packs["raw_tokens"]["cold"] is True
        assert source_load_counts == {f"{node}.pt": 1 for node in _E2E_PIPELINE_NODES}
        assert (cast(Path, e2e_cfg.data.pack_dir) / "manifest.json").is_file()
        assert te.required_pack_paths(e2e_cfg, pack_dir) == (
            pack_dir,
            e2e_cfg.data.pack_dir,
        )

        # Warm-path re-validation must agree with the same configured n_ground.
        rebuilt = te.prepare_pack(e2e_cfg, pack_dir, cold_cache=False)
        assert cast(dict[str, object], rebuilt["pack_manifest"])["n_ground"] == (
            _E2E_PIPELINE_N_GROUND
        )
        rebuilt_packs = cast(dict[str, dict[str, object]], rebuilt["packs"])
        assert rebuilt_packs["f0_grounding"]["cold"] is False
        assert rebuilt_packs["raw_tokens"]["cold"] is False
        assert (
            rebuilt_packs["raw_tokens"]["identity_sha256"] == packs["raw_tokens"]["identity_sha256"]
        )

    def test_prepare_pack_rejects_stage1_only_config_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_e2e_feature_root(tmp_path, _E2E_PIPELINE_NODES)
        benchmark = _e2e_pipeline_benchmark()
        monkeypatch.setattr(te, "_load_benchmark_for", lambda cfg: benchmark)
        cfg = _toy_cfg(tmp_path)
        e2e_cfg = replace(cfg, model=ModelConfig(family="egostitch_e2e", config={"slots": 4}))
        with pytest.raises(ValueError, match="unknown E2E config keys"):
            te.prepare_pack(e2e_cfg, tmp_path / "pack", cold_cache=True)

    def test_assemble_skips_s0_and_produces_e2e_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_e2e_feature_root(tmp_path, _E2E_PIPELINE_NODES)
        benchmark = _e2e_pipeline_benchmark()
        monkeypatch.setattr(te, "_load_benchmark_for", lambda cfg: benchmark)
        e2e_cfg = self._e2e_cfg(tmp_path)
        pack_dir = tmp_path / "pack"

        # require_s0 defaults True; family egostitch_e2e must still skip it
        # (spec Sec 13.10: s0 retired for this family).
        data = te.assemble_egostitch_data(e2e_cfg, pack_dir=pack_dir)
        assert len(data.s0) == 0
        assert data.grounding_index.shape == (
            len(_E2E_PIPELINE_NODES),
            _E2E_PIPELINE_N_GROUND,
        )

        token_pack_dir = tmp_path / "token_pack"
        _write_tiny_token_pack(token_pack_dir, _E2E_PIPELINE_NODES, min_length=3)
        e2e_cfg_with_pack = replace(e2e_cfg, data=replace(e2e_cfg.data, pack_dir=token_pack_dir))
        model_cfg = EgoStitchConfig()
        rows, steps = te._epoch_step_plan(
            len(data.e_sup_positives),
            negative_ratio=e2e_cfg_with_pack.data.negative_ratio,
            edge_batch=e2e_cfg_with_pack.data.edge_batch,
            world_size=1,
        )
        factory = te._BatchFactory(
            e2e_cfg_with_pack, model_cfg, data, node_batch=4, rank=0, world_size=1
        )
        batch = next(iter(factory.epoch_batches(1, rows_per_rank=rows, steps=steps)))
        for key in ("emb_a", "emb_b", "len_a", "len_b", "ground_id_i", "ground_id_j"):
            assert key in batch.edge
        assert "s0" not in batch.edge


class TestOverfitAcceptance:
    """§13.19.4 item-1 post-ramp overfit acceptance (spec change-log 2026-07-22)."""

    # The retained passing attempt-005 V_fit trajectory (metrics.jsonl,
    # 2026-07-21, BF16-era readout): AUPRC saturates at 1.0 from epoch 7;
    # Phase-C residual ratios oscillate around 1e-3 (readout quantization
    # noise per spec §12 2026-07-22).
    _RETAINED = (
        (1, "A", 0.348471, 0.0),
        (2, "A", 0.559022, 0.0),
        (3, "A", 0.720202, 0.0),
        (4, "A", 0.908526, 0.0),
        (5, "A", 0.990429, 0.0),
        (6, "B", 0.999725, 0.000520600),
        (7, "B", 1.0, 0.000901622),
        (8, "B", 1.0, 0.000668134),
        (9, "C", 1.0, 0.001495493),
        (10, "C", 1.0, 0.001216008),
        (11, "C", 1.0, 0.001083565),
        (12, "C", 1.0, 0.001181883),
        (13, "C", 1.0, 0.000997061),
        (14, "C", 1.0, 0.001157461),
        (15, "C", 1.0, 0.000979123),
        (16, "C", 1.0, 0.001140527),
        (17, "C", 1.0, 0.001179883),
        (18, "C", 1.0, 0.001072686),
        (19, "C", 1.0, 0.000954866),
        (20, "C", 1.0, 0.001165148),
        (21, "C", 1.0, 0.001060615),
        (22, "C", 1.0, 0.000946267),
        (23, "C", 1.0, 0.001054033),
        (24, "C", 1.0, 0.001412919),
        (25, "C", 1.0, 0.001370386),
        (26, "C", 1.0, 0.001328335),
        (27, "C", 1.0, 0.001241188),
        (28, "C", 1.0, 0.001282460),
        (29, "C", 1.0, 0.000992638),
        (30, "C", 1.0, 0.001278644),
    )

    @staticmethod
    def _record(
        epoch: int, phase: te.E2EPhaseName, auprc: float, residual: float | None
    ) -> te.E2ECheckpointRecord:
        return te.E2ECheckpointRecord(
            epoch=epoch,
            phase=phase,
            full_joint_epochs_completed=max(0, epoch - 9),
            guards_passed=True,
            auprc=auprc,
            prevalence=1.0 / 6.0,
            active_logit_std=1.0,
            clustering_mmd=0.0,
            brier=0.0,
            residual_ratio=residual,
        )

    # The retained fp32-readout gatefix trajectory (failed_run_history.json,
    # 2026-07-22): identical seed/config/pack; the honest fp32 residual sits
    # in the 1e-5 decade and grows monotonically across Phase C.
    _FP32_RETAINED = (
        (1, "A", 0.346710, 0.0),
        (2, "A", 0.544728, 0.0),
        (3, "A", 0.743517, 0.0),
        (4, "A", 0.903183, 0.0),
        (5, "A", 0.997577, 0.0),
        (6, "B", 1.0, 0.000000131),
        (7, "B", 1.0, 0.000007328),
        (8, "B", 1.0, 0.000004242),
        (9, "C", 1.0, 0.000003622),
        (10, "C", 1.0, 0.000003718),
        (11, "C", 1.0, 0.000003936),
        (12, "C", 1.0, 0.000004352),
        (13, "C", 1.0, 0.000004948),
        (14, "C", 1.0, 0.000005518),
        (15, "C", 1.0, 0.000006066),
        (16, "C", 1.0, 0.000006694),
        (17, "C", 1.0, 0.000007403),
        (18, "C", 1.0, 0.000008042),
        (19, "C", 1.0, 0.000008641),
        (20, "C", 1.0, 0.000009245),
        (21, "C", 1.0, 0.000009815),
        (22, "C", 1.0, 0.000010245),
        (23, "C", 1.0, 0.000010664),
        (24, "C", 1.0, 0.000010996),
        (25, "C", 1.0, 0.000011267),
        (26, "C", 1.0, 0.000011505),
        (27, "C", 1.0, 0.000011689),
        (28, "C", 1.0, 0.000011868),
        (29, "C", 1.0, 0.000012046),
        (30, "C", 1.0, 0.000012178),
    )

    def _retained_records(self) -> list[te.E2ECheckpointRecord]:
        return [
            self._record(epoch, cast(te.E2EPhaseName, phase), auprc, residual)
            for epoch, phase, auprc, residual in self._RETAINED
        ]

    def _fp32_records(self) -> list[te.E2ECheckpointRecord]:
        return [
            self._record(epoch, cast(te.E2EPhaseName, phase), auprc, residual)
            for epoch, phase, auprc, residual in self._FP32_RETAINED
        ]

    def test_pre_ramp_epochs_never_qualify(self) -> None:
        assert not te.e2e_overfit_epoch_qualified(self._record(5, "A", 1.0, 0.01))
        assert not te.e2e_overfit_epoch_qualified(self._record(7, "B", 1.0, 0.01))

    def test_phase_c_requires_both_registered_inequalities(self) -> None:
        assert te.e2e_overfit_epoch_qualified(self._record(9, "C", 0.95, 1e-6))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 0.949, 1e-6))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, 9.9e-7))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, 0.0))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, None))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", float("nan"), 1e-6))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, float("nan")))

    def test_retained_passing_trajectory_selects_the_same_final_epoch(self) -> None:
        """The run that passed the old final-epoch rule selects the identical epoch."""
        assert te.select_e2e_overfit_epoch(self._retained_records()) == 30

    def test_knife_edge_final_epoch_no_longer_invalidates_the_run(self) -> None:
        """The latest qualifying epoch wins despite a below-floor final epoch.

        This is the attempt-005 final-epoch-only failure mode, recast at the
        fp32-calibrated boundary.
        """
        records = self._retained_records()
        records[-2], records[-1] = (
            self._record(29, "C", 1.0, 1.2e-5),
            self._record(30, "C", 1.0, 5e-7),
        )
        assert te.select_e2e_overfit_epoch(records) == 29

    def test_fp32_trajectory_selects_the_final_epoch(self) -> None:
        """The retained fp32 gatefix trajectory qualifies under the recalibrated floor."""
        assert te.select_e2e_overfit_epoch(self._fp32_records()) == 30

    def test_no_qualifying_epoch_returns_zero(self) -> None:
        records = [self._record(epoch, "C", 1.0, 5e-7) for epoch in range(9, 31)]
        assert te.select_e2e_overfit_epoch(records) == 0
        assert te.select_e2e_overfit_epoch([]) == 0

    def test_failed_selection_history_is_retained(self, tmp_path: Path) -> None:
        history: list[dict[str, object]] = [
            {"epoch": 1.0, "auprc": 0.5, "fidelity": {"topology_delta_ratio": 0.0}}
        ]
        te._write_failed_run_history(
            tmp_path / "out", run_kind="overfit", arm="full", history=history
        )
        payload = json.loads((tmp_path / "out" / "failed_run_history.json").read_text())
        assert payload["run_kind"] == "overfit"
        assert payload["arm"] == "full"
        assert payload["history"] == history

    def test_failed_selection_history_write_never_masks_the_failure(self, tmp_path: Path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the output directory should be")
        te._write_failed_run_history(blocked, run_kind="overfit", arm="full", history=[])
        assert blocked.is_file()
