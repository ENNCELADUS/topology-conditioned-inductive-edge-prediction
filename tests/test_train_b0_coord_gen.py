"""Trainer plumbing for the Stage II coordinate generator (v3_1_coord_gen)."""

from __future__ import annotations

import functools
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src.data.struct_coords import COORD_DIM
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import (
    COORD_GEN_FAMILY,
    MODEL_FAMILIES,
    TopoPromptRows,
    TrainResult,
    ValidationOutcome,
    _base_loss_kwargs,
    _build_optimizer,
    _coordinate_fit_metrics,
    _evaluate_distributed,
    _run_metadata,
    build_model,
    config_to_dict,
    is_v3_1_family,
    load_config,
    resolve_model_kwargs,
    train_ddp_loop,
)

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0 import _constant_metrics, _tiny_config, _write_yaml_config
from tests.test_train_b0_prefix import _base_checkpoint
from tests.test_train_b0_topo_prompt import _prompt_batches, _tiny_graph


def _reader_checkpoint(tmp_path: Path, *, with_stats: bool = True) -> Path:
    torch.manual_seed(0)
    reader = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"trainable": "all", "width": 8, "slots_per_field": 1, "field_mask_prob": 0.0},
    )
    if with_stats:
        reader.generator.set_coord_stats(
            torch.linspace(-1.0, 1.0, COORD_DIM), torch.linspace(0.5, 2.0, COORD_DIM), 9
        )
    with torch.no_grad():
        reader.generator.gates.fill_(0.3)
    payload = {
        "model_state": reader.state_dict(),
        "model_family": "v3_1_topo_prompt",
        "model_config": {"base": _tiny_base_config(), "topo_prompt": reader.cfg.to_dict()},
        "epoch": 1,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    path = tmp_path / ("reader.pt" if with_stats else "reader_nostats.pt")
    torch.save(payload, path)
    return path


def _gen_yaml(tmp_path: Path, **block: object) -> Path:
    config_path = tmp_path / "gen.yaml"
    full_block: dict[str, object] = {
        "reader_checkpoint": str(_reader_checkpoint(tmp_path)),
        "hidden": 16,
        "layers": 1,
        "dropout": 0.0,
    }
    full_block.update(block)
    _write_yaml_config(
        config_path, {"model": {"family": COORD_GEN_FAMILY, "config": {"coord_gen": full_block}}}
    )
    return config_path


def test_family_is_registered_and_grouped_with_v3_1() -> None:
    assert COORD_GEN_FAMILY in MODEL_FAMILIES
    assert is_v3_1_family(COORD_GEN_FAMILY)


def test_resolve_and_build_from_a_reader_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(_gen_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    reader = cast(dict[str, object], kwargs["reader"])
    assert reader["base"] == _tiny_base_config()
    block = cast(dict[str, object], kwargs["coord_gen"])
    assert block["reader_checkpoint_sha256"] is not None and block["kd_alpha"] == 0.5
    assert _base_loss_kwargs(cfg.model)["positive_weight"] == 5.0
    model = build_model(cfg)
    assert isinstance(model, V3_1CoordGen)
    assert float(model.reader.generator.coord_count) == 9.0
    assert float(model.reader.generator.gates.mean()) == pytest.approx(0.3)
    assert all(not param.requires_grad for param in model.teacher.parameters())
    assert any(param.requires_grad for param in model.reader.parameters())
    optimizer = _build_optimizer(model, cfg)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        model.trainable_parameters()
    )


def test_resolution_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    _write_yaml_config(missing, {"model": {"family": COORD_GEN_FAMILY, "config": {}}})
    with pytest.raises(ValueError, match="coord_gen is required"):
        resolve_model_kwargs(load_config(missing).model)
    sibling = tmp_path / "sibling.yaml"
    _write_yaml_config(
        sibling,
        {
            "model": {
                "family": COORD_GEN_FAMILY,
                "config": {"coord_gen": {"reader_checkpoint": "x.pt"}, "base": {}},
            }
        },
    )
    with pytest.raises(ValueError, match="accepts only"):
        resolve_model_kwargs(load_config(sibling).model)
    no_reader = tmp_path / "no_reader.yaml"
    _write_yaml_config(
        no_reader, {"model": {"family": COORD_GEN_FAMILY, "config": {"coord_gen": {}}}}
    )
    with pytest.raises(ValueError, match="reader_checkpoint is required"):
        resolve_model_kwargs(load_config(no_reader).model)
    with pytest.raises(ValueError, match="v3_1_topo_prompt"):
        resolve_model_kwargs(
            load_config(
                _gen_yaml(tmp_path, reader_checkpoint=str(_base_checkpoint(tmp_path)))
            ).model
        )
    no_stats = _gen_yaml(
        tmp_path, reader_checkpoint=str(_reader_checkpoint(tmp_path, with_stats=False))
    )
    with pytest.raises(ValueError, match="no coordinate statistics"):
        build_model(load_config(no_stats))


def test_run_metadata_records_the_reader_and_weights(tmp_path: Path) -> None:
    cfg = load_config(_gen_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    state = build_model(cfg).state_dict()
    result = TrainResult(
        best_state_dict=state,
        last_state_dict=state,
        best_epoch=1,
        last_epoch=1,
        best_val_metrics=_constant_metrics(),
        last_val_metrics=_constant_metrics(),
        history=[],
        stopped_early=False,
        stop_epoch=None,
    )
    metadata = _run_metadata(result, cfg, kwargs, {}, config_to_dict(cfg))
    block = cast(dict[str, object], metadata["coord_gen"])
    assert block["reader_checkpoint"] == str(tmp_path / "reader.pt")
    assert block["sha256"] is not None
    assert cast(dict[str, float], block["weights"])["kd_alpha"] == 0.5
    assert "run_kind" not in metadata
    assert metadata["arm"] == COORD_GEN_FAMILY
    json.dumps(metadata)


def _rows_and_batches() -> tuple[
    TopoPromptRows, list[dict[str, torch.Tensor]], list[tuple[str, str]]
]:
    train, val, nodes = _tiny_graph()
    rows_per_batch, n_batches = 4, 3
    n_rows = rows_per_batch * n_batches
    rng = np.random.default_rng(0)
    train_pairs = [
        (nodes[int(i)], nodes[int(j)]) for i, j in rng.integers(len(nodes), size=(n_rows, 2))
    ]
    val_pairs = [("v0", "v1"), ("v1", "v3"), ("v2", "v4"), ("v0", "v3")]
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=train_pairs,
        stats_rows=np.arange(n_rows),
        val_graph=val,
        val_cls_pairs=val_pairs,
        universe_pairs=val_pairs,
        device=torch.device("cpu"),
    )
    return rows, _prompt_batches(n_batches, rows_per_batch), val_pairs


def test_coordinate_fit_metrics_are_collective_and_bounded(tmp_path: Path) -> None:
    rows, _, val_pairs = _rows_and_batches()
    model = build_model(load_config(_gen_yaml(tmp_path)))
    accelerator = Accelerator(cpu=True)
    val_batches = _prompt_batches(1, len(val_pairs))
    model.train()
    metrics = _coordinate_fit_metrics(model, val_batches, accelerator, attach=rows.attach_val)
    assert model.training  # restored
    assert set(metrics) == {
        "val_coord_r2_endpoint",
        "val_coord_r2_relation",
        "val_coord_r2_context",
        "val_coord_dist_acc",
        "val_coord_loss",
        "val_kd_loss",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["val_coord_dist_acc"] <= 1.0
    assert max(metrics[f"val_coord_r2_{f}"] for f in ("endpoint", "relation", "context")) <= 1.0
    again = _coordinate_fit_metrics(model, val_batches, accelerator, attach=rows.attach_val)
    assert again == metrics
    plain = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    with pytest.raises(TypeError, match="V3_1CoordGen"):
        _coordinate_fit_metrics(plain, val_batches, accelerator, attach=rows.attach_val)


def test_ddp_loop_preserves_teacher_and_logs_the_fit(tmp_path: Path) -> None:
    rows, batches, val_pairs = _rows_and_batches()
    model = build_model(load_config(_gen_yaml(tmp_path)))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    val_batches = _prompt_batches(1, len(val_pairs))
    cfg = replace(load_config(_gen_yaml(tmp_path)), output_dir=tmp_path)
    cfg = replace(cfg, optim=_tiny_config(epochs=1).optim, eval=_tiny_config(epochs=1).eval)
    accelerator = Accelerator(cpu=True)

    def evaluate(
        model_: torch.nn.Module, loader: object, accelerator_: Accelerator
    ) -> ValidationOutcome:
        return _evaluate_distributed(
            model_,
            cast(list[dict[str, torch.Tensor]], loader),
            accelerator_,
            expected_row_ids=np.arange(len(val_pairs)),
            attach=rows.attach_val,
            diagnostics_fn=functools.partial(_coordinate_fit_metrics, attach=rows.attach_val),
        )

    result = train_ddp_loop(
        model,
        lambda epoch: batches,
        val_batches,
        cfg,
        accelerator,
        warmup_steps=1,
        artifact_dir=tmp_path,
        evaluate_fn=evaluate,
        topo_rows=rows,
    )
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before if key.startswith("teacher."))
    assert any(
        not torch.equal(before[key], after[key]) for key in before if key.startswith("generator.")
    )
    last = result.history[-1]
    assert last["epoch"] == 1 and "val_coord_r2_relation" in last and "val_kd_loss" in last


@pytest.mark.parametrize("generator", ["mlp", "virtual_graph"])
def test_compact_coordinate_fit_reports_only_present_fields(generator: str) -> None:
    from src.data.struct_coords import get_coord_spec

    train, val, nodes = _tiny_graph()
    pairs = [(nodes[0], nodes[1]), (nodes[0], nodes[0])]
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=pairs,
        stats_rows=np.arange(2),
        val_graph=val,
        val_cls_pairs=[("v0", "v1"), ("v0", "v3")],
        universe_pairs=[],
        device=torch.device("cpu"),
        spec="v3",
    )
    reader_config = {
        "base": _tiny_base_config(),
        "topo_prompt": {
            "trainable": "all",
            "width": 8,
            "slots_per_field": 1,
            "field_mask_prob": 0.0,
            "coord_spec": "v3",
        },
    }
    model = V3_1CoordGen(
        reader=reader_config,
        coord_gen={
            "coord_spec": "v3",
            "hidden": 16,
            "generator": generator,
            "virtual_graph": {"k": 2, "d_z": 8, "heads": 2},
        },
    )
    model.reader.generator.set_coord_stats(
        torch.zeros(get_coord_spec("v3").coord_dim), torch.ones(get_coord_spec("v3").coord_dim), 2
    )
    model.initialize_teacher()
    metrics = _coordinate_fit_metrics(
        model, _prompt_batches(1, 2), Accelerator(cpu=True), attach=rows.attach_val
    )
    if generator == "virtual_graph":
        assert 0 <= metrics["val_virtual_attachment_mean"] <= 1
        assert metrics["val_virtual_usage_0"] >= 0
        assert "val_virtual_gate_mean" not in metrics
    assert "val_coord_r2_context" not in metrics
    assert "val_coord_r2_endpoint" in metrics and "val_coord_r2_relation" in metrics
    assert all(np.isfinite(value) for value in metrics.values())


def _virtual_initialisation_fixture() -> tuple[TopoPromptRows, object, V3_1CoordGen]:
    from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable, PackedNodeRecord

    train, val, nodes = _tiny_graph()
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=[(nodes[0], nodes[1]), (nodes[1], nodes[2])],
        stats_rows=np.arange(2),
        val_graph=val,
        val_cls_pairs=[],
        universe_pairs=[],
        device=torch.device("cpu"),
        spec="v3",
    )
    node_ids = [*rows.train_table.nodes, "held_out"]
    manifest = PackedFeatureManifest(
        format="bf16_flat_shards_v1",
        input_dim=4,
        dtype="float32",
        source_metadata_sha256="",
        source_index_sha256="",
        nodes=tuple(PackedNodeRecord(n, 0, i * 3, i * 3, 3) for i, n in enumerate(node_ids)),
        shards=(),
        pack_workers=1,
        build_seconds=0.0,
    )
    table = PackedFeatureTable(
        torch.randn(len(node_ids) * 3, 4),
        torch.arange(len(node_ids)) * 3,
        torch.full((len(node_ids),), 3),
        manifest,
    )
    model = V3_1CoordGen(
        reader={
            "base": _tiny_base_config(),
            "topo_prompt": {"trainable": "all", "coord_spec": "v3"},
        },
        coord_gen={
            "coord_spec": "v3",
            "generator": "virtual_graph",
            "virtual_graph": {"k": 2, "d_z": 8, "heads": 2},
        },
    )
    return rows, table, model


def test_virtual_initialisation_uses_only_legal_training_nodes() -> None:
    from src.data.packed_features import PackedFeatureTable
    from src.train_b0 import initialise_virtual_graph

    rows, packed, model = _virtual_initialisation_fixture()
    table = cast(PackedFeatureTable, packed)
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    metadata = initialise_virtual_graph(model, rows, table, Accelerator(cpu=True), seed=13)
    assert metadata["train_nodes"] == len(rows.train_table.nodes)
    assert sum(cast(list[int], metadata["cluster_sizes"])) == len(rows.train_table.nodes)
    assert metadata["seed"] == 13
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in encoder_before.items())


def _virtual_initialisation_worker(rank: int, root: str) -> None:
    import contextlib
    import os
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist
    from src.data.packed_features import PackedFeatureTable
    from src.train_b0 import initialise_virtual_graph

    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(13 + rank)
        rows, table, model = _virtual_initialisation_fixture()
        model.reader.generator.set_coord_stats(rows.coord_mean, rows.coord_std, rows.coord_count)
        model.initialize_teacher()
        model.install_coordinate_scale(rows.train)
        accelerator = cast(
            Accelerator,
            SimpleNamespace(
                is_main_process=rank == 0,
                num_processes=2,
                device=torch.device("cpu"),
                autocast=contextlib.nullcontext,
            ),
        )
        metadata = initialise_virtual_graph(
            model, rows, cast(PackedFeatureTable, table), accelerator, seed=13
        )
        torch.save(
            {
                "state": model.state_dict(),
                "metadata": metadata,
                "attachment_targets": rows.attachment_targets,
            },
            Path(root) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_virtual_initialisation_broadcast_and_checkpoint_round_trip(tmp_path: Path) -> None:
    from torch.multiprocessing.spawn import spawn

    spawn(_virtual_initialisation_worker, args=(str(tmp_path),), nprocs=2, join=True)  # type: ignore[no-untyped-call]
    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert left["metadata"] == right["metadata"]
    torch.testing.assert_close(
        left["attachment_targets"], right["attachment_targets"], rtol=0, atol=0
    )
    for key, value in left["state"].items():
        if key.startswith("generator.") or key == "coordinate_scale":
            torch.testing.assert_close(value, right["state"][key], rtol=0, atol=0)
    _, _, restored = _virtual_initialisation_fixture()
    restored.load_state_dict(left["state"])
    for key, value in left["state"].items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_virtual_initialisation_omits_featureless_nodes_without_changing_targets() -> None:
    from src.data.packed_features import PackedFeatureTable
    from src.train_b0 import initialise_virtual_graph

    rows, packed, model = _virtual_initialisation_fixture()
    table = cast(PackedFeatureTable, packed)
    omitted = rows.train_table.nodes[0]
    table.manifest = replace(
        table.manifest, nodes=tuple(n for n in table.manifest.nodes if n.node_id != omitted)
    )
    # Rebuild lookup arrays in manifest order; token offsets still reference the original table.
    table.offsets = torch.tensor([n.global_offset for n in table.manifest.nodes])
    table.lengths = torch.tensor([n.length for n in table.manifest.nodes])
    original_adjacency = rows.train_table.adjacency.copy()
    original_targets = rows.train.clone()
    metadata = initialise_virtual_graph(model, rows, table, Accelerator(cpu=True), seed=13)
    assert metadata["omitted_featureless_nodes"] == [omitted]
    assert metadata["omitted_featureless_count"] == 1
    assert metadata["train_nodes"] == len(rows.train_table.nodes) - 1
    assert sum(cast(list[int], metadata["cluster_sizes"])) == len(rows.train_table.nodes) - 1
    expected_edges = int(original_adjacency[1:, 1:].nnz // 2)
    assert metadata["train_edges"] == expected_edges
    assert metadata["mean_degree"] == pytest.approx(
        2 * expected_edges / (len(rows.train_table.nodes) - 1)
    )
    assert (rows.train_table.adjacency != original_adjacency).nnz == 0
    assert torch.equal(rows.train, original_targets)


def test_virtual_initialisation_installs_fixed_graph_and_node_prior_targets() -> None:
    from sklearn.cluster import SpectralClustering
    from src.data.packed_features import PackedFeatureTable
    from src.model.egostitch.classifier.virtual_graph import VirtualGraphGenerator
    from src.train_b0 import initialise_virtual_graph

    rows, packed, model = _virtual_initialisation_fixture()
    table = cast(PackedFeatureTable, packed)
    generator = cast(VirtualGraphGenerator, model.generator)
    initialise_virtual_graph(model, rows, table, Accelerator(cpu=True), seed=13)
    adjacency = rows.train_table.adjacency.astype(np.float64)
    labels = SpectralClustering(
        n_clusters=2,
        n_components=2,
        affinity="precomputed",
        assign_labels="cluster_qr",
        eigen_solver="arpack",
        random_state=13,
    ).fit_predict(adjacency)
    membership = np.eye(2)[labels]
    sizes = membership.sum(0)
    expected = torch.tensor(adjacency @ membership, dtype=torch.float32)
    assert rows.attachment_targets is not None
    torch.testing.assert_close(rows.attachment_targets, expected)
    torch.testing.assert_close(generator.multiplicity, torch.tensor(sizes, dtype=torch.float32))
    denominator = sizes[:, None] * sizes[None, :] - np.diag(sizes)
    density = np.divide(
        membership.T @ adjacency @ membership,
        denominator,
        out=np.zeros((2, 2)),
        where=denominator > 0,
    )
    torch.testing.assert_close(generator.adjacency, torch.tensor(density, dtype=torch.float32))
    assert "multiplicity" in dict(generator.named_buffers())
    assert "adjacency" in dict(generator.named_buffers())
    assert rows.attachment_targets.shape[0] == len(rows.train_table.nodes)
    assert "held_out" not in rows.train_table.nodes

    batch = {"_row_id": torch.tensor([1, 0])}
    rows.attach_train(batch)
    for column, side in enumerate(("u", "v")):
        torch.testing.assert_close(
            batch[f"attachment_target_{side}"],
            expected[rows.train_endpoint_indices[[1, 0], column]],
        )
    validation = {"_row_id": torch.empty(0, dtype=torch.long)}
    rows.attach_val(validation)
    assert not any(key.startswith("attachment_target") for key in validation)


@pytest.mark.parametrize("cut", [1, 2, 4])
def test_composite_ddp_loss_matches_global_gradient_with_unequal_partitions(cut: int) -> None:
    from src.train_b0 import _scale_training_loss
    from torch.nn import functional as F

    torch.manual_seed(73)
    model = torch.nn.Linear(3, 2)
    features = torch.randn(5, 3)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    counts = torch.tensor([0.0, 2.0, 1.0, 0.0, 3.0])
    weights = 1 + 4 * labels

    def output(start: int, stop: int) -> dict[str, torch.Tensor]:
        logits = model(features[start:stop])
        w = weights[start:stop]
        task = (
            F.binary_cross_entropy_with_logits(logits[:, 0], labels[start:stop], reduction="none")
            * w
        ).sum() / w.sum()
        attachment = F.huber_loss(F.softplus(logits[:, 1]).log1p(), counts[start:stop].log1p())
        return {
            "loss": task + attachment,
            "attachment_loss": attachment,
            "loss_weight_sum": w.sum(),
        }

    global_objective = output(0, 5)["loss"]
    expected = torch.autograd.grad(global_objective, tuple(model.parameters()))
    partitioned = torch.stack(
        [
            _scale_training_loss(
                output(start, stop),
                local_count=stop - start,
                global_count=5,
                world_size=2,
                global_weight=weights.sum(),
            )
            / 2
            for start, stop in ((0, cut), (cut, 5))
        ]
    ).sum()
    actual = torch.autograd.grad(partitioned, tuple(model.parameters()))
    torch.testing.assert_close(partitioned, global_objective)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_virtual_resume_reuses_saved_partition_without_reclustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.packed_features import PackedFeatureTable
    from src.train_b0 import initialise_virtual_graph

    rows, table, model = _virtual_initialisation_fixture()
    metadata = initialise_virtual_graph(
        model, rows, cast(PackedFeatureTable, table), Accelerator(cpu=True), seed=13
    )
    torch.save(
        {"model_state": model.state_dict(), "virtual_graph": metadata},
        tmp_path / "training_state.pt",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume must not refit spectral clusters")

    monkeypatch.setattr("sklearn.cluster.SpectralClustering.fit_predict", forbidden)
    restored_rows, restored_table, restored = _virtual_initialisation_fixture()
    restored_metadata = initialise_virtual_graph(
        restored,
        restored_rows,
        cast(PackedFeatureTable, restored_table),
        Accelerator(cpu=True),
        seed=99,
        resume_attempt=tmp_path,
    )
    assert restored_metadata == metadata
    assert rows.attachment_targets is not None
    assert restored_rows.attachment_targets is not None
    torch.testing.assert_close(rows.attachment_targets, restored_rows.attachment_targets)
    for key, value in model.generator.state_dict().items():
        torch.testing.assert_close(value, restored.generator.state_dict()[key])
