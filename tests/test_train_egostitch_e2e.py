"""E2E training contracts for the EgoStitch worker."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
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
from src.data.packed_features import (
    PACK_FORMAT,
    PackedFeatureManifest,
    PackedFeatureTable,
    PackedNodeRecord,
    PackedShardRecord,
    sha256_file,
    write_packed_manifest,
)
from src.model.egostitch import EgoStitchConfig
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
                        manifest=manifest, steps=3, step_offset=5
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
                        manifest=manifest, steps=3, step_offset=5
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
            "pair_only": joint_weight == 0.0,
            "real_ssl_scale": torch.tensor(joint_weight),
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

        assert set(family_norms["pair_encoder_head"]) == {"edge"}
        assert set(family_norms["generator"]) == {"recon"}
        assert family_norms["topology_content_conditioning"] == {}
        assert set(submodule_rms) == {
            "grad_rms_trunk",
            "grad_rms_ste",
            "grad_rms_content",
        }
        assert payload["collect_diagnostics"] is False

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
        original_from_pack = PackedFeatureTable.from_pack.__func__

        def _float_cpu_pack(
            cls: type[PackedFeatureTable], root: Path, device: torch.device
        ) -> PackedFeatureTable:
            table = original_from_pack(cls, root, device)
            table.tokens = table.tokens.float()
            return table

        monkeypatch.setattr(PackedFeatureTable, "from_pack", classmethod(_float_cpu_pack))
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
        assert result.best_epoch == 1
        assert guard_persistence == [False] * len(optimizer_steps)

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
# Family `egostitch_e2e` forces the internal generator's pinned n_ground=20
# (`EgoStitchConfig()`'s spec default, not configurable via `E2EConfig`), and
# `build_grounding_pool` requires `n_ground <= len(train_nodes) - 1` -- so this
# fixture needs strictly more nodes than the 8-node `_NODES` toy universe used
# elsewhere in this file.

_E2E_PIPELINE_NODES = [f"g{i}" for i in range(25)]


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
            model=ModelConfig(family="egostitch_e2e", config={}),
            data=replace(cfg.data, pack_dir=tmp_path / "raw-token-pack"),
        )

    def test_prepare_pack_uses_generator_pinned_n_ground(
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
        assert manifest["n_ground"] == EgoStitchConfig().n_ground
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

        # Warm-path re-validation must agree with the same pinned n_ground.
        rebuilt = te.prepare_pack(e2e_cfg, pack_dir, cold_cache=False)
        assert cast(dict[str, object], rebuilt["pack_manifest"])["n_ground"] == (
            EgoStitchConfig().n_ground
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
            EgoStitchConfig().n_ground,
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
    # 2026-07-21): AUPRC saturates at 1.0 from epoch 7; Phase-C residual
    # ratios oscillate around the 1e-3 floor.
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

    def _retained_records(self) -> list[te.E2ECheckpointRecord]:
        return [
            self._record(epoch, cast(te.E2EPhaseName, phase), auprc, residual)
            for epoch, phase, auprc, residual in self._RETAINED
        ]

    def test_pre_ramp_epochs_never_qualify(self) -> None:
        assert not te.e2e_overfit_epoch_qualified(self._record(5, "A", 1.0, 0.01))
        assert not te.e2e_overfit_epoch_qualified(self._record(7, "B", 1.0, 0.01))

    def test_phase_c_requires_both_registered_inequalities(self) -> None:
        assert te.e2e_overfit_epoch_qualified(self._record(9, "C", 0.95, 1e-3))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 0.949, 1e-3))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, 0.000999))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, None))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", float("nan"), 1e-3))
        assert not te.e2e_overfit_epoch_qualified(self._record(9, "C", 1.0, float("nan")))

    def test_retained_passing_trajectory_selects_the_same_final_epoch(self) -> None:
        """The run that passed the old final-epoch rule selects the identical epoch."""
        assert te.select_e2e_overfit_epoch(self._retained_records()) == 30

    def test_knife_edge_final_epoch_no_longer_invalidates_the_run(self) -> None:
        """A benign trajectory shift ending below the floor keeps a qualifying epoch.

        This is the exact attempt-005 failure mode: swapping the last two
        retained epochs puts the final residual (0.000992638) under 1e-3, which
        the old final-epoch-only rule rejected outright.
        """
        records = self._retained_records()
        records[-2], records[-1] = (
            self._record(29, "C", 1.0, 0.001278644),
            self._record(30, "C", 1.0, 0.000992638),
        )
        assert te.select_e2e_overfit_epoch(records) == 29

    def test_no_qualifying_epoch_returns_zero(self) -> None:
        records = [self._record(epoch, "C", 1.0, 5e-4) for epoch in range(9, 31)]
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
