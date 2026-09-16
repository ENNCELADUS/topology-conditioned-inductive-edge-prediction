"""Trainer plumbing for the v3_1_motif_prompt family."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import MotifTemplateTable
from src.data.struct_sampler import StructSubgraph
from src.model.egostitch.classifier.motif_prompt import V3_1MotifPrompt
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import StructStream

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0_struct import (
    _stream,
    _struct_fixture,
    _StructToy,
    _topo_prompt_fixture,
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
