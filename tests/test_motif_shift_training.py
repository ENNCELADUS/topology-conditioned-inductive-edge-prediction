"""Trainer contracts for v17 shift-matched motif prompting."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
from src.data.motif_crossfit import MotifCrossfitCache, seeded_hash_node_folds
from src.model.egostitch.classifier.motif_prompt import (
    PREDICTED_MASK_KEY,
    PREDICTED_WEIGHTS_KEY,
    TEMPLATE_KEY,
    TRUE_DIAGNOSTIC_KEY,
    V3_1MotifPrompt,
)
from src.train_b0 import (
    MOTIF_PROMPT_FAMILY,
    MotifTemplateRows,
    _assert_motif_run_kind,
    _load_deployed_motif_generator,
)

from tests.test_prefix_model import _tiny_base_config


def _model(*, stage: str, deployed: str = "") -> V3_1MotifPrompt:
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
    }
    if stage == "two":
        block["bundle_checkpoint"] = "reader.pt"
    if deployed:
        block["deployed_generator_checkpoint"] = deployed
    return V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block)


def test_deployed_generator_loader_copies_only_generator_and_freezes(tmp_path: Path) -> None:
    source = _model(stage="two")
    target = _model(stage="one", deployed="source.pt")
    reader_before = {key: value.clone() for key, value in target.reader.state_dict().items()}
    path = tmp_path / "source.pt"
    torch.save(
        {"model_family": MOTIF_PROMPT_FAMILY, "model_state": source.state_dict()}, path
    )

    _load_deployed_motif_generator(target, path)

    assert target.generator is not None and source.generator is not None
    assert all(
        torch.equal(value, source.generator.state_dict()[key])
        for key, value in target.generator.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in target.generator.parameters())
    assert target.generator.training is False
    assert all(
        torch.equal(value, reader_before[key]) for key, value in target.reader.state_dict().items()
    )


def test_crossfit_rows_attach_predictions_and_keep_true_diagnostic_separate() -> None:
    graph = nx.Graph()
    nodes = ["a", "b", "c", "d"]
    graph.add_edges_from((left, right) for left in nodes for right in nodes if left < right)
    folds = seeded_hash_node_folds(graph.nodes, seed=42)
    pair = next(
        (left, right)
        for left in nodes
        for right in nodes
        if left < right and folds[left] == folds[right]
    )
    cache = MotifCrossfitCache.build(
        [pair],
        torch.ones(1, 96),
        fold_by_node=folds,
        source_folds=[1 - folds[pair[0]]],
    )
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=[pair],
        stats_rows=torch.tensor([0]).numpy(),
        device=torch.device("cpu"),
        seed=42,
        val_graph=graph,
        val_cls_pairs=[pair],
        universe_pairs=[pair],
        predicted_cache=cache,
    )
    batch = {"_row_id": torch.tensor([0]), "label": torch.tensor([1.0])}
    rows.attach_train(batch)
    assert TEMPLATE_KEY in batch
    assert torch.equal(batch[PREDICTED_WEIGHTS_KEY], torch.ones(1, 96))
    assert batch[PREDICTED_MASK_KEY].tolist() == [True]

    predicted = {"_row_id": torch.tensor([0]), "label": torch.tensor([1.0])}
    rows.attach_val_predicted(predicted)
    assert TEMPLATE_KEY not in predicted
    assert predicted[TRUE_DIAGNOSTIC_KEY].tolist() == [False]
    rows.attach_val_diagnostic(predicted)
    assert TEMPLATE_KEY in predicted
    assert predicted[TRUE_DIAGNOSTIC_KEY].tolist() == [True]


def test_shift_matched_stage_one_is_formal_while_legacy_stage_one_is_diagnostic() -> None:
    _assert_motif_run_kind(
        stage="one", run_kind="formal", deployed_generator_checkpoint="generator.pt"
    )
    _assert_motif_run_kind(stage="one", run_kind="diagnostic")
    with pytest.raises(RuntimeError, match="requires a formal run"):
        _assert_motif_run_kind(
            stage="one", run_kind="diagnostic", deployed_generator_checkpoint="generator.pt"
        )
    with pytest.raises(RuntimeError, match="launch it with --run-kind diagnostic"):
        _assert_motif_run_kind(stage="one", run_kind="formal")


def test_reader_corpus_presence_is_saved_without_reinitializing_deployed_generator() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    model = V3_1MotifPrompt(
        base=_tiny_base_config(),
        motif_prompt={
            "stage": "one",
            "base_checkpoint": "base.pt",
            "deployed_generator_checkpoint": "deployed.pt",
            "generator_head": "presence_profile",
            "width": 16,
            "reader": {"layers": 1, "dim": 16, "heads": 4, "rrwp_k": 2},
        },
    )
    assert model.generator is not None
    before = {key: value.clone() for key, value in model.generator.state_dict().items()}
    rows = MotifTemplateRows(
        train_graph=graph,
        train_pairs=[("a", "b"), ("a", "d")],
        stats_rows=torch.tensor([0]).numpy(),
        device=torch.device("cpu"),
        seed=42,
    )
    rows.install(model)
    expected = model.generator.presence_from_weights(rows.train[:1]).mean(0)
    assert torch.equal(model.mean_presence, expected)
    assert torch.equal(model.state_dict()["mean_presence"], expected)
    assert all(
        torch.equal(value, before[key]) for key, value in model.generator.state_dict().items()
    )
