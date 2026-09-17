"""Trainer plumbing for the v3_1_motif_prompt family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.motif_template import MotifTemplateTable
from src.data.packed_features import PackedFeatureTable
from src.data.struct_sampler import StructSampler, StructSubgraph
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.motif_prompt import TEMPLATE_KEY, V3_1MotifPrompt
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import Config, ModelConfig, StructStream, TrainResult

from tests.model.test_motif_prompt_model import _model, _open_gates, _weights
from tests.test_prefix_model import _pair_batch, _tiny_base_config
from tests.test_train_b0 import _constant_metrics, _tiny_config, _write_yaml_config
from tests.test_train_b0_struct import (
    _stream,
    _struct_fixture,
    _StructToy,
    _topo_prompt_fixture,
)


def _motif_cfg(config: dict[str, object]) -> Config:
    """A tiny `Config` carrying one ``v3_1_motif_prompt`` model section."""
    cfg = _tiny_config()
    return replace(cfg, model=ModelConfig(family="v3_1_motif_prompt", config=config))


def _empty_train_result() -> TrainResult:
    """A `TrainResult` with the fields `_run_metadata` reads and nothing else."""
    state: dict[str, torch.Tensor] = {"w": torch.zeros(2)}
    return TrainResult(
        best_state_dict=state,
        best_epoch=1,
        best_val_metrics=_constant_metrics(),
        last_state_dict=state,
        last_epoch=1,
        last_val_metrics=_constant_metrics(),
        history=[],
        stopped_early=False,
    )


def _motif_model(*, with_teacher: bool = True, stage: str = "two") -> V3_1MotifPrompt:
    """A motif-prompt model with open gates over the tiny frozen base."""
    torch.manual_seed(0)
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
    }
    if stage == "two":
        block["bundle_checkpoint"] = "bundle.pt"
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block)
    model.install_mean_template(torch.full((96,), 0.1))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))
    if with_teacher and stage == "two":
        model.initialize_teacher()
    return model


def _tiny_struct_stream(
    *, with_teacher: bool = True
) -> tuple[StructStream, V3_1MotifPrompt, StructSubgraph]:
    """A structural stream over a small training graph plus its motif template table."""
    sampler, table, _ = _topo_prompt_fixture()
    stream = _stream(sampler, table, token_budget=64, templates=MotifTemplateTable(sampler.graph))
    subgraph = stream._epoch_plan(epoch=1, steps=1).subgraphs[0]  # noqa: SLF001
    return stream, _motif_model(with_teacher=with_teacher), subgraph


def test_struct_stream_compiles_a_template_for_every_sampled_subgraph_pair() -> None:
    stream, model, subgraph = _tiny_struct_stream()
    logits, target, mask = stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    assert logits.shape == target.shape == mask.shape
    assert stream.last_motif_rows["slot"].numel() > 0
    assert stream.last_motif_rows["topo"].numel() == stream.last_motif_rows["slot"].numel()
    # One compiled template per legal pair of the sampled subgraph, not per node.
    assert stream.last_motif_rows["slot"].numel() == int(torch.triu(mask, diagonal=1).sum())


def test_struct_stream_templates_come_from_the_training_graph_with_the_pair_removed() -> None:
    stream, _, subgraph = _tiny_struct_stream()
    table = stream._motif_table  # noqa: SLF001
    assert table is not None
    u, v = subgraph.nodes[0], subgraph.nodes[1]
    row = table.compile_row(u, v)
    assert v not in row.closure and u not in row.closure
    assert v not in row.left and u not in row.right


def test_struct_stream_topo_rows_are_a_differentiable_zero_without_a_teacher() -> None:
    stream, model, subgraph = _tiny_struct_stream(with_teacher=False)
    stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    rows = stream.last_motif_rows["topo"]
    assert float(rows.abs().sum()) == 0.0
    assert rows.requires_grad
    assert stream.last_motif_rows["slot"].numel() == rows.numel()
    assert float(stream.last_motif_rows["slot"].abs().sum()) > 0.0


def test_motif_rows_survive_activation_checkpointing_with_their_gradient() -> None:
    # The rows are produced inside the checkpointed chunk forward, so they only
    # reach the generator if the tuple return replays on the backward pass.
    stream, model, subgraph = _tiny_struct_stream()
    assert torch.is_grad_enabled()
    stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    rows = stream.last_motif_rows
    assert rows["topo"].requires_grad and float(rows["topo"].abs().sum()) > 0.0
    (rows["slot"].sum() + rows["topo"].sum()).backward()  # type: ignore[no-untyped-call]
    assert model.generator is not None
    grads = [p.grad for _, p in model.generator.named_parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads if g is not None)


def test_a_stream_without_a_template_table_refuses_to_score_the_motif_family() -> None:
    sampler, table, _ = _topo_prompt_fixture()
    stream = _stream(sampler, table, token_budget=64)
    subgraph = stream._epoch_plan(epoch=1, steps=1).subgraphs[0]  # noqa: SLF001
    with pytest.raises(RuntimeError, match="requires a template table"):
        stream._score(_motif_model(), subgraph, sampler)  # noqa: SLF001


# --------------------------------------- structural validation across the V_val boundary


def _node_disjoint_streams() -> tuple[StructSampler, PackedFeatureTable, nx.Graph, nx.Graph]:
    """Training and V_val graphs that share no node, as the split contract requires."""
    sampler, table, _ = _topo_prompt_fixture()
    val_graph = sampler.graph
    train_graph = val_graph.subgraph(sorted(set(val_graph) - {"n0", "n1"})).copy()
    return sampler, table, train_graph, val_graph


def _validation_stream(
    stage: str, *, with_val_table: bool = True
) -> tuple[StructStream, V3_1MotifPrompt, StructSampler, StructSubgraph]:
    """A stream whose validation sampler covers V_val nodes absent from the training table."""
    sampler, table, train_graph, val_graph = _node_disjoint_streams()
    val_sampler = StructSampler(
        val_graph,
        nodes=8,
        background_nodes=2,
        mix={"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        v_val=frozenset(),
        exclude_nodes=frozenset(),
    )
    stream = _stream(
        sampler,
        table,
        token_budget=64,
        val_sampler=val_sampler,
        templates=MotifTemplateTable(train_graph),
        val_templates=MotifTemplateTable(val_graph) if with_val_table else None,
    )
    subgraph = StructSubgraph(kind="bfs", nodes=("n0", "n1", "n2", "n3"), background=0)
    return stream, _motif_model(stage=stage), val_sampler, subgraph


def test_stage_one_structural_validation_reads_the_v_val_template_table() -> None:
    # `_motif_table` is the training table and V_val nodes are node-disjoint from
    # it, so scoring a validation subgraph against it raises `KeyError`.
    stream, model, val_sampler, subgraph = _validation_stream("one")
    with torch.no_grad():
        logits, target, mask = stream._score(model, subgraph, val_sampler)  # noqa: SLF001
    assert logits.shape == target.shape == mask.shape
    assert torch.isfinite(logits).all()
    # The V_val table really was read: one compiled row per legal validation pair.
    assert stream.last_motif_rows["slot"].numel() == int(torch.triu(mask, diagonal=1).sum())


def test_stage_two_structural_validation_predicts_without_any_template() -> None:
    # Stage II is deployable: V_val truth is neither an input nor a target there
    # (spec section 8), so validation subgraphs carry no compiled template at all.
    stream, model, val_sampler, subgraph = _validation_stream("two", with_val_table=False)
    with torch.no_grad():
        logits, _, _ = stream._score(model, subgraph, val_sampler)  # noqa: SLF001
    assert torch.isfinite(logits).all()
    assert stream.last_motif_rows == {}


def test_stage_one_structural_validation_without_a_v_val_table_fails_closed() -> None:
    stream, model, val_sampler, subgraph = _validation_stream("one", with_val_table=False)
    with pytest.raises(RuntimeError, match="V_val template table"), torch.no_grad():
        stream._score(model, subgraph, val_sampler)  # noqa: SLF001


def test_a_skipped_structural_step_clears_the_previous_steps_motif_rows() -> None:
    # `struct.subgraphs_per_epoch: 0.5` leaves steps without a subgraph. The
    # trainer folds `last_motif_rows` into the composite on every step, so a
    # stale row tensor is backwarded a second time through a freed graph.
    sampler, table, _ = _topo_prompt_fixture()
    stream = _stream(
        sampler,
        table,
        token_budget=64,
        templates=MotifTemplateTable(sampler.graph),
        weights={"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1},
    )
    stream.config = replace(stream.config, subgraphs_per_epoch=0.5)
    model = _motif_model()
    scheduled, _ = stream.loss(model, epoch=1, step=0, steps=2)
    rows = stream.last_motif_rows
    assert rows["slot"].numel() > 0
    (scheduled + rows["slot"].sum() + rows["topo"].sum()).backward()  # type: ignore[no-untyped-call]

    skipped, _ = stream.loss(model, epoch=1, step=1, steps=2)

    assert stream.last_motif_rows == {}
    # Without the reset the trainer would backward through the freed graph above.
    skipped.backward()  # type: ignore[no-untyped-call]


# ------------------------------------------------- the other arms are untouched


def test_the_checkpointed_chunks_reproduce_an_unchunked_forwards_gradient() -> None:
    # `_score`'s chunk forward now has a second, tuple-returning variant for the
    # motif family. Every other arm keeps the single-tensor closure: the chunked,
    # checkpointed logits and their gradient must still equal one direct forward.
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, token_budget=3)  # forces many small chunks
    model = _StructToy()
    subgraph = stream._epoch_plan(epoch=1, steps=1).subgraphs[0]  # noqa: SLF001
    logits, _, mask = stream._score(model, subgraph, sampler)  # noqa: SLF001
    assert max(model.batch_sizes) <= 3
    logits.sum().backward()  # type: ignore[no-untyped-call]
    assert model.weight.grad is not None
    chunked = float(model.weight.grad)

    index = table.manifest.node_index()
    n = len(subgraph.nodes)
    rows, cols = torch.triu_indices(n, n, offset=1)
    keep = mask[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    boundary = max(table.manifest.nodes[index[node]].length for node in subgraph.nodes)
    emb_a, len_a = table.gather_nodes(
        torch.tensor([index[subgraph.nodes[i]] for i in rows.tolist()]), boundary
    )
    emb_b, len_b = table.gather_nodes(
        torch.tensor([index[subgraph.nodes[j]] for j in cols.tolist()]), boundary
    )
    direct_model = _StructToy()
    direct = direct_model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b})
    torch.testing.assert_close(logits[rows, cols].detach(), direct["logits"].detach())
    # The assembled matrix carries every pair twice, above and below the diagonal.
    (2.0 * direct["logits"].sum()).backward()
    assert direct_model.weight.grad is not None
    assert chunked == pytest.approx(float(direct_model.weight.grad), rel=1e-6)
    assert stream.last_motif_rows == {}


def test_a_topology_prompt_arm_scores_without_motif_rows_and_keeps_its_gradient() -> None:
    sampler, table, coords = _topo_prompt_fixture()
    model = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"trainable": "all", "width": 8, "slots_per_field": 1},
    )
    coords.install(model)
    stream = _stream(sampler, table, token_budget=128, coordinates=coords)
    subgraph = stream._epoch_plan(epoch=1, steps=1).subgraphs[0]  # noqa: SLF001
    logits, _, _ = stream._score(model, subgraph, sampler)  # noqa: SLF001
    assert stream.last_motif_rows == {}
    logits.sum().backward()  # type: ignore[no-untyped-call]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in trainable)


def test_a_plain_struct_arm_never_allocates_motif_rows_even_with_a_table_present() -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, templates=MotifTemplateTable(sampler.graph))
    subgraph = stream._epoch_plan(epoch=1, steps=1).subgraphs[0]  # noqa: SLF001
    stream._score(_StructToy(), subgraph, sampler)  # noqa: SLF001
    assert stream.last_motif_rows == {}


# ------------------------------------------------------ family registration (task 16)


def _base_checkpoint(path: Path) -> dict[str, object]:
    """Write a tiny published ``v3_1`` base checkpoint and return its model config."""
    base = _tiny_base_config()
    payload = {
        "model_family": "v3_1",
        "model_config": base,
        "model_state": V3_1(**base).state_dict(),
    }
    torch.save(payload, path)
    return base


def _motif_config(**block: object) -> dict[str, object]:
    """A minimal ``model.config.motif_prompt`` block over the tiny base."""
    return {
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
        **block,
    }


def test_the_family_is_accepted_by_the_config_schema_and_the_v3_1_gate() -> None:
    from src.train_b0 import MODEL_FAMILIES, MOTIF_PROMPT_FAMILY, V3_1_FAMILIES, is_v3_1_family

    assert MOTIF_PROMPT_FAMILY == "v3_1_motif_prompt"
    assert MOTIF_PROMPT_FAMILY in MODEL_FAMILIES
    assert MOTIF_PROMPT_FAMILY in V3_1_FAMILIES
    assert is_v3_1_family(MOTIF_PROMPT_FAMILY)


def test_motif_kwargs_take_an_inline_base_and_reject_sibling_keys() -> None:
    from src.train_b0 import ModelConfig, resolve_model_kwargs

    inline = ModelConfig(
        family="v3_1_motif_prompt",
        config={
            "motif_prompt": _motif_config(stage="one", base_checkpoint="b.pt"),
            "base": {"d_model": 8},
        },
    )
    kwargs = resolve_model_kwargs(inline)
    assert set(kwargs) == {"motif_prompt", "base"}
    with pytest.raises(ValueError, match="accepts only model.config"):
        resolve_model_kwargs(
            ModelConfig(family="v3_1_motif_prompt", config={"motif_prompt": {}, "reader": {}})
        )
    with pytest.raises(ValueError, match="model.config.motif_prompt is required"):
        resolve_model_kwargs(ModelConfig(family="v3_1_motif_prompt", config={}))


def test_motif_kwargs_embed_the_base_checkpoints_config_and_digest(tmp_path: Path) -> None:
    from src.train_b0 import ModelConfig, resolve_model_kwargs

    checkpoint = tmp_path / "base.pt"
    base = _base_checkpoint(checkpoint)
    kwargs = resolve_model_kwargs(
        ModelConfig(
            family="v3_1_motif_prompt",
            config={"motif_prompt": _motif_config(stage="one", base_checkpoint=str(checkpoint))},
        )
    )
    assert kwargs["base"] == base
    block = cast(dict[str, object], kwargs["motif_prompt"])
    assert isinstance(block["base_checkpoint_sha256"], str)
    with pytest.raises(ValueError, match="needs model.config.base or"):
        resolve_model_kwargs(
            ModelConfig(family="v3_1_motif_prompt", config={"motif_prompt": _motif_config()})
        )


def test_base_loss_kwargs_unwraps_the_motif_nesting() -> None:
    from src.train_b0 import ModelConfig, _base_loss_kwargs

    cfg = ModelConfig(
        family="v3_1_motif_prompt",
        config={
            "motif_prompt": _motif_config(stage="one", base_checkpoint="b.pt"),
            "base": {"positive_weight": 7.0, "label_smoothing": 0.25},
        },
    )
    resolved = _base_loss_kwargs(cfg)
    assert resolved["positive_weight"] == 7.0
    assert resolved["label_smoothing"] == 0.25


def test_build_model_loads_the_frozen_base_and_the_stage_one_bundle(tmp_path: Path) -> None:
    from src.train_b0 import build_model

    base_path = tmp_path / "base.pt"
    _base_checkpoint(base_path)
    stage_one = build_model(
        _motif_cfg(
            {
                "motif_prompt": _motif_config(stage="one", base_checkpoint=str(base_path)),
            }
        )
    )
    assert isinstance(stage_one, V3_1MotifPrompt)
    assert stage_one.teacher is None
    assert stage_one.generator is None
    reference = torch.load(base_path, map_location="cpu", weights_only=False)["model_state"]
    for key, value in stage_one.base.state_dict().items():
        torch.testing.assert_close(value, reference[key], rtol=0, atol=0)

    bundle_path = tmp_path / "bundle.pt"
    torch.save(
        {
            "model_family": "v3_1_motif_prompt",
            "model_config": {},
            "model_state": stage_one.state_dict(),
        },
        bundle_path,
    )
    stage_two = build_model(
        _motif_cfg(
            {
                "motif_prompt": _motif_config(
                    stage="two",
                    base_checkpoint=str(base_path),
                    bundle_checkpoint=str(bundle_path),
                ),
            }
        )
    )
    assert isinstance(stage_two, V3_1MotifPrompt)
    assert stage_two.generator is not None
    assert stage_two.teacher is not None
    # The teacher is an immutable copy of the loaded Stage I reader.
    torch.testing.assert_close(
        stage_two.reader.node_proj.weight,
        stage_one.reader.node_proj.weight,
        rtol=0,
        atol=0,
    )


def test_a_bundle_checkpoint_of_another_family_is_refused(tmp_path: Path) -> None:
    from src.train_b0 import build_model

    base_path = tmp_path / "base.pt"
    _base_checkpoint(base_path)
    bundle_path = tmp_path / "bundle.pt"
    torch.save({"model_family": "v3_1", "model_config": {}, "model_state": {}}, bundle_path)
    with pytest.raises(ValueError, match="published v3_1_motif_prompt checkpoint"):
        build_model(
            _motif_cfg(
                {
                    "motif_prompt": _motif_config(
                        stage="two",
                        base_checkpoint=str(base_path),
                        bundle_checkpoint=str(bundle_path),
                    ),
                }
            )
        )


def test_run_metadata_records_the_motif_provenance(tmp_path: Path) -> None:
    from src.train_b0 import _run_metadata, resolve_model_kwargs

    base_path = tmp_path / "base.pt"
    _base_checkpoint(base_path)
    cfg = _motif_cfg({"motif_prompt": _motif_config(stage="one", base_checkpoint=str(base_path))})
    metadata = _run_metadata(
        _empty_train_result(),
        cfg,
        resolve_model_kwargs(cfg.model),
        {},
        {},
    )
    block = cast(dict[str, object], metadata["motif_prompt"])
    assert block["stage"] == "one"
    assert block["base_checkpoint"] == str(base_path)
    assert isinstance(block["base_checkpoint_sha256"], str)
    assert block["bundle_checkpoint"] is None
    assert block["families"] == ["closure", "bridge"]


# ------------------------------------------- per-row templates and stage guards (task 17)


def _tiny_training_graph() -> nx.Graph:
    """A small loopless graph carrying closures and bridges."""
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("b", "d")])
    return graph


def test_motif_rows_attach_the_template_and_mask_by_row_id() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = _tiny_training_graph()
    pairs = [("a", "b"), ("a", "c"), ("a", "a")]
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.asarray([0, 1, 2]),
        device=torch.device("cpu"),
        seed=0,
    )
    batch = {"_row_id": torch.tensor([2, 0])}
    rows.attach_train(batch)
    assert batch["motif_weights"].shape == (2, 96)
    torch.testing.assert_close(batch["motif_weights"], rows.train[[2, 0]])
    # The self row is excluded from L_slot and L_topo but keeps task BCE.
    torch.testing.assert_close(batch["motif_mask"], torch.tensor([0.0, 1.0]))


def test_the_mean_template_is_measured_over_the_epoch_one_rows_only() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = _tiny_training_graph()
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.asarray([0, 1]),
        device=torch.device("cpu"),
        seed=0,
    )
    torch.testing.assert_close(rows.mean, rows.train[[0, 1]].mean(dim=0))
    assert rows.summary()["stats_rows"] == 2
    assert rows.summary()["train_rows"] == 3
    assert rows.summary()["val_cls_rows"] == 0


def test_installing_the_mean_publishes_it_into_the_models_buffer() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = _tiny_training_graph()
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=[("a", "b"), ("a", "c")],
        stats_rows=np.asarray([0, 1]),
        device=torch.device("cpu"),
        seed=0,
    )
    model = _motif_model()
    rows.install(model)
    torch.testing.assert_close(model.mean_template, rows.mean)
    with pytest.raises(TypeError, match="V3_1MotifPrompt"):
        rows.install(V3_1(**_tiny_base_config()))


def test_stage_two_rows_compile_no_v_val_template_and_refuse_to_attach_one() -> None:
    # Spec section 8: true V_val template compilation belongs exclusively to
    # labelled Stage I and oracle diagnostics.
    from src.train_b0 import MotifTemplateRows

    rows = MotifTemplateRows(
        train_graph=_tiny_training_graph(),
        train_pairs=[("a", "b"), ("a", "c")],
        stats_rows=np.asarray([0]),
        device=torch.device("cpu"),
        seed=0,
    )
    assert rows.val_cls is None
    assert rows.universe is None
    with pytest.raises(RuntimeError, match="no compiled V_val templates"):
        rows.attach_val({"_row_id": torch.tensor([0])})


def test_stage_one_rows_carry_the_v_val_tables_and_attach_them_by_row_id() -> None:
    from src.train_b0 import MotifTemplateRows

    graph = _tiny_training_graph()
    val_pairs = [("a", "b"), ("b", "c"), ("b", "b")]
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=[("a", "b"), ("a", "c")],
        stats_rows=np.asarray([0]),
        device=torch.device("cpu"),
        seed=0,
        val_graph=graph,
        val_cls_pairs=val_pairs,
        universe_pairs=val_pairs[:2],
    )
    assert rows.val_cls is not None and rows.universe is not None
    assert rows.universe.shape == (2, 96)
    batch = {"_row_id": torch.tensor([2, 1])}
    rows.attach_val(batch)
    torch.testing.assert_close(batch["motif_weights"], rows.val_cls[[2, 1]])
    torch.testing.assert_close(batch["motif_mask"], torch.tensor([0.0, 1.0]))
    assert rows.summary()["universe_rows"] == 2


def test_stage_one_refuses_a_formal_run_and_stage_two_refuses_a_diagnostic_one() -> None:
    from src.train_b0 import _assert_motif_run_kind

    with pytest.raises(RuntimeError, match="--run-kind diagnostic"):
        _assert_motif_run_kind(stage="one", run_kind="formal")
    _assert_motif_run_kind(stage="one", run_kind="diagnostic")
    _assert_motif_run_kind(stage="two", run_kind="formal")
    with pytest.raises(RuntimeError, match="deployable"):
        _assert_motif_run_kind(stage="two", run_kind="diagnostic")


# ------------------------------------------------- stage trainability (task 18)


def test_the_interface_group_is_frozen_for_the_warmup_epochs_and_opens_after() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _model("two", interface_warmup_epochs=2)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    names = [group["name"] for group in optimizer.param_groups]
    assert names == ["generator", "interface"]
    for epoch in (1, 2):
        for group in optimizer.param_groups:
            group["lr"] = 3e-4  # what OneCycleLR would have just written
        _set_motif_prompt_training_stage(model, optimizer, epoch=epoch)
        assert optimizer.param_groups[0]["lr"] == 3e-4
        assert optimizer.param_groups[1]["lr"] == 0.0
        assert model.interface_open is False
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
    _set_motif_prompt_training_stage(model, optimizer, epoch=3)
    assert optimizer.param_groups[1]["lr"] == 3e-4
    assert model.interface_open is True


def test_a_null_warmup_freezes_the_interface_for_every_epoch() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _model("two", interface_warmup_epochs=None)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    for epoch in (1, 3, 15):
        optimizer.param_groups[1]["lr"] = 3e-4
        _set_motif_prompt_training_stage(model, optimizer, epoch=epoch)
        assert optimizer.param_groups[1]["lr"] == 0.0
        assert model.interface_open is False


def test_the_interface_weights_do_not_move_during_the_warmup_epochs() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _model("two", interface_warmup_epochs=2)
    model.initialize_teacher()
    _open_gates(model)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-5, 1e-2))
    before = model.reader.node_proj.weight.detach().clone()
    generator_before = next(model.generator.parameters()).detach().clone()  # type: ignore[union-attr]
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
    _set_motif_prompt_training_stage(model, optimizer, epoch=1)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    output = model(batch)
    total = output["loss"] + output["slot_loss_rows"].mean() + output["topo_loss_rows"].mean()
    total.backward()
    optimizer.step()
    torch.testing.assert_close(model.reader.node_proj.weight, before, rtol=0, atol=0)
    assert model.generator is not None
    assert not torch.equal(next(model.generator.parameters()), generator_before)


def test_stage_one_leaves_the_reader_open_from_epoch_one() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = _model("one")
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-4, 1e-4, 1e-2))
    optimizer.param_groups[0]["lr"] = 2e-4
    _set_motif_prompt_training_stage(model, optimizer, epoch=1)
    assert model.interface_open is True
    assert optimizer.param_groups[0]["lr"] == 2e-4


def test_the_stage_hook_ignores_every_other_family() -> None:
    from src.train_b0 import _set_motif_prompt_training_stage

    model = V3_1(**_tiny_base_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.param_groups[0]["lr"] = 5e-4
    _set_motif_prompt_training_stage(model, optimizer, epoch=1)
    assert optimizer.param_groups[0]["lr"] == 5e-4


# ------------------------------------------- optim.stop_after_epoch (task 19)


def test_stop_after_epoch_parses_validates_and_leaves_the_config_hash_unchanged(
    tmp_path: Path,
) -> None:
    from src.train_b0 import config_to_dict, load_config

    plain_path = tmp_path / "plain.json"
    _write_yaml_config(plain_path)
    plain = load_config(plain_path)

    stopped_path = tmp_path / "stopped.json"
    _write_yaml_config(stopped_path, {"optim.stop_after_epoch": 2})
    stopped = load_config(stopped_path)
    assert stopped.optim.stop_after_epoch == 2
    assert plain.optim.stop_after_epoch is None
    assert config_to_dict(stopped) == config_to_dict(plain)

    for illegal in (0, plain.optim.epochs + 1):
        bad_path = tmp_path / f"bad-{illegal}.json"
        _write_yaml_config(bad_path, {"optim.stop_after_epoch": illegal})
        with pytest.raises(ValueError, match="optim.stop_after_epoch"):
            load_config(bad_path)


def test_a_resume_config_comparison_ignores_stop_after_epoch_only() -> None:
    from src.train_b0 import _resume_comparable_config

    saved = {
        "output_dir": "a",
        "seed": 0,
        "optim": {"epochs": 15, "stop_after_epoch": 2, "lr": 1e-4},
    }
    current = {"output_dir": "b", "seed": 0, "optim": {"epochs": 15, "lr": 1e-4}}
    assert _resume_comparable_config(saved) == _resume_comparable_config(current)
    # The saved mapping is not mutated.
    assert saved["optim"] == {"epochs": 15, "stop_after_epoch": 2, "lr": 1e-4}
    diverged = {"output_dir": "b", "seed": 1, "optim": {"epochs": 15, "lr": 1e-4}}
    assert _resume_comparable_config(saved) != _resume_comparable_config(diverged)
    epochs_changed = {"output_dir": "b", "seed": 0, "optim": {"epochs": 2, "lr": 1e-4}}
    assert _resume_comparable_config(saved) != _resume_comparable_config(epochs_changed)


def test_schedule_total_steps_ignores_stop_after_epoch() -> None:
    # The pilot keeps optim.epochs at 15 so the first two epochs are a true
    # prefix of the full one-cycle: same trainability mask, same LRs, same order.
    from src.train_b0 import _count_single_process_steps

    cfg = _tiny_config(epochs=4)
    stopped = replace(cfg, optim=replace(cfg.optim, stop_after_epoch=2))

    def factory(epoch: int) -> list[dict[str, torch.Tensor]]:
        return [{"label": torch.zeros(2)}] * (epoch + 1)

    assert _count_single_process_steps(factory, stopped) == _count_single_process_steps(
        factory, cfg
    )


# ------------------------------------------- composite loss folding (task 20)


def test_l_slot_and_l_topo_are_added_once_across_the_two_streams() -> None:
    from src.train_b0 import _motif_stream_terms

    anchor = torch.zeros((), requires_grad=True)
    task_rows = {
        "slot": torch.tensor([1.0, 3.0]) + anchor,
        "topo": torch.tensor([2.0, 4.0]) + anchor,
    }
    task_mask = torch.tensor([1.0, 1.0])
    struct_rows = {"slot": torch.tensor([5.0]) + anchor, "topo": torch.tensor([7.0]) + anchor}
    struct_mask = torch.tensor([1.0])
    slot, topo = _motif_stream_terms(
        task=(task_rows, task_mask), struct=(struct_rows, struct_mask), like=anchor
    )
    # Per-stream means 2.0 and 5.0, then the mean across streams.
    assert float(slot) == pytest.approx(3.5)
    assert float(topo) == pytest.approx(5.0)
    assert slot.requires_grad


def test_an_absent_structural_stream_leaves_the_task_stream_alone() -> None:
    from src.train_b0 import _motif_stream_terms

    anchor = torch.zeros((), requires_grad=True)
    rows = {"slot": torch.tensor([1.0, 3.0]) + anchor, "topo": torch.tensor([2.0, 4.0]) + anchor}
    slot, topo = _motif_stream_terms(
        task=(rows, torch.tensor([1.0, 1.0])), struct=None, like=anchor
    )
    assert float(slot) == pytest.approx(2.0)
    assert float(topo) == pytest.approx(3.0)


def test_a_self_only_batch_contributes_a_differentiable_zero() -> None:
    from src.train_b0 import _motif_stream_terms

    anchor = torch.zeros((), requires_grad=True)
    rows = {"slot": torch.tensor([0.0, 0.0]) + anchor, "topo": torch.tensor([0.0, 0.0]) + anchor}
    slot, topo = _motif_stream_terms(task=(rows, torch.zeros(2)), struct=None, like=anchor)
    assert float(slot.detach()) == 0.0
    assert float(topo.detach()) == 0.0
    assert slot.requires_grad and topo.requires_grad


def test_epoch_telemetry_reports_both_terms() -> None:
    from src.train_b0 import _motif_epoch_telemetry

    telemetry = _motif_epoch_telemetry(slot_sum=4.0, topo_sum=2.0, steps=2)
    assert telemetry == {"train_motif_slot_loss": 2.0, "train_motif_topo_loss": 1.0}
    assert _motif_epoch_telemetry(slot_sum=0.0, topo_sum=0.0, steps=0) == {}


def test_the_model_returns_task_bce_only_and_leaves_the_composite_to_the_trainer() -> None:
    model = _model("two")
    model.initialize_teacher()
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    out = model(batch)
    # `loss` is the weighted task BCE alone: the trainer owns the per-stream
    # average of L_slot and L_topo, which only it can see (spec section 7.5).
    torch.testing.assert_close(out["loss"].detach(), out["task_loss"])
    assert float(out["slot_loss_rows"].detach().sum()) > 0.0
    assert out["slot_loss_rows"].requires_grad and out["topo_loss_rows"].requires_grad
    assert float(out["loss_term_slot"].detach()) == pytest.approx(
        model.cfg.w_slot * float(out["slot_loss_rows"].detach().mean())
    )
    assert float(out["loss_term_topo"].detach()) == pytest.approx(
        model.cfg.w_topo * float(out["topo_loss_rows"].detach().mean())
    )


def test_the_trainer_fold_reaches_the_generator_through_both_streams() -> None:
    # The exact composition the DDP step performs: the task stream's rows from
    # the forward output, the structural stream's from `last_motif_rows`.
    from src.train_b0 import _motif_stream_terms

    stream, model, subgraph = _tiny_struct_stream()
    stream._score(model, subgraph, stream._sampler)  # noqa: SLF001
    struct_rows = stream.last_motif_rows
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["motif_mask"] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    output = model(batch)
    slot, topo = _motif_stream_terms(
        task=(
            {"slot": output["slot_loss_rows"], "topo": output["topo_loss_rows"]},
            batch["motif_mask"],
        ),
        struct=(struct_rows, torch.ones_like(struct_rows["slot"])),
        like=output["loss"],
    )
    (output["loss"] + model.cfg.w_slot * slot + model.cfg.w_topo * topo).backward()
    assert model.generator is not None
    grads = [p.grad for p in model.generator.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads if g is not None)
    assert all(p.grad is None for p in model.base.parameters())


def test_a_stage_one_batch_folds_a_differentiable_zero() -> None:
    # Stage I reads the compiled template as its input, so it has no graph
    # supervision: the fold must still stay in the graph on every rank.
    from src.train_b0 import _motif_stream_terms

    model = _model("one")
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["motif_mask"] = torch.ones(4)
    output = model(batch)
    assert "slot_loss_rows" not in output
    slot, topo = _motif_stream_terms(
        task=(
            {
                "slot": output.get("slot_loss_rows", output["loss"].new_zeros(0)),
                "topo": output.get("topo_loss_rows", output["loss"].new_zeros(0)),
            },
            batch["motif_mask"],
        ),
        struct=None,
        like=output["loss"],
    )
    assert float(slot.detach()) == 0.0 and float(topo.detach()) == 0.0
    (output["loss"] + slot + topo).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in model.reader.parameters()
    )


# ------------------------------------------- DDP prerequisites (task 22)


def test_every_rank_builds_bit_identical_templates_and_means() -> None:
    # `MotifTemplateRows` is built independently on every rank and never
    # broadcast, so determinism is what keeps the ranks in agreement.
    from src.train_b0 import MotifTemplateRows

    graph = _tiny_training_graph()
    pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("a", "a")]
    built = [
        MotifTemplateRows(
            train_graph=_tiny_training_graph(),
            train_pairs=list(pairs),
            stats_rows=np.asarray([0, 1, 2]),
            device=torch.device("cpu"),
            seed=47,
        )
        for _ in range(2)
    ]
    assert graph.number_of_nodes() == 4
    torch.testing.assert_close(built[0].train, built[1].train, rtol=0, atol=0)
    torch.testing.assert_close(built[0].mean, built[1].mean, rtol=0, atol=0)
    torch.testing.assert_close(built[0].train_mask, built[1].train_mask, rtol=0, atol=0)


def test_every_trainable_parameter_receives_a_gradient_in_one_step() -> None:
    # `build_ddp_accelerator` pins `find_unused_parameters=False`, so a trainable
    # parameter this arm leaves out of the forward would abort the real run.
    from src.train_b0 import _motif_stream_terms

    model = _model("two")
    model.initialize_teacher()
    _open_gates(model)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = _weights(n=4)
    batch["motif_mask"] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    output = model(batch)
    slot, topo = _motif_stream_terms(
        task=(
            {"slot": output["slot_loss_rows"], "topo": output["topo_loss_rows"]},
            batch["motif_mask"],
        ),
        struct=None,
        like=output["loss"],
    )
    (output["loss"] + model.cfg.w_slot * slot + model.cfg.w_topo * topo).backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []
