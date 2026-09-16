"""Trainer plumbing for the v3_1_motif_prompt family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from src.data.motif_template import MotifTemplateTable
from src.data.struct_sampler import StructSubgraph
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.motif_prompt import V3_1MotifPrompt
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import Config, ModelConfig, StructStream, TrainResult

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0 import _constant_metrics, _tiny_config
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


def _motif_model(*, with_teacher: bool = True) -> V3_1MotifPrompt:
    """A Stage II motif-prompt model with open gates over the tiny frozen base."""
    torch.manual_seed(0)
    model = V3_1MotifPrompt(
        base=_tiny_base_config(),
        motif_prompt={
            "stage": "two",
            "base_checkpoint": "base.pt",
            "bundle_checkpoint": "bundle.pt",
            "width": 16,
            "slots_per_field": 2,
            "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
        },
    )
    model.install_mean_template(torch.full((96,), 0.1))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.adapter.gates.copy_(torch.randn(model.adapter.gates.shape, generator=gen))
    if with_teacher:
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
