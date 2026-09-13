"""Per-epoch prompt sensitivity and resume-safe admitted-set diagnostics."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src.data.struct_coords import COORD_DIM
from src.eval.checkpoint_selection import TopologyValidationMetrics
from src.eval.val_topology import ValTopologyReference, ValTopologyResult
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import (
    PromptV2Validation,
    TopoPromptRows,
    _evaluate_distributed,
)

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0_topo_prompt import _prompt_batches


def _fixture(stage: int) -> tuple[PromptV2Validation, V3_1TopoPrompt, list[Any]]:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    universe_pairs = list(itertools.combinations_with_replacement("abcd", 2))
    rows = TopoPromptRows(
        train_graph=graph,
        train_pairs=pairs,
        stats_rows=np.arange(4),
        val_graph=graph,
        val_cls_pairs=pairs,
        universe_pairs=universe_pairs,
        device=torch.device("cpu"),
    )
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    rows.install(model)
    reference = ValTopologyReference(tuple("abcd"), graph, {4: [set("abcd"), set("abcd")]})
    calls: list[Any] = []

    def topology(
        _model: torch.nn.Module,
        _accelerator: Accelerator,
        *,
        row_coords: torch.Tensor | None = None,
        logits_sink: Callable[[np.ndarray], None],
    ) -> ValTopologyResult:
        assert not torch.is_grad_enabled()
        calls.append(None if row_coords is None else row_coords.clone())
        scores = (
            np.linspace(-1.0, 2.0, len(universe_pairs))
            if row_coords is None
            else row_coords[:, 0].numpy()
        )
        logits_sink(scores)
        return ValTopologyResult(TopologyValidationMetrics(0.4, 1.0, 2.0, 3.0, 4.0), 0.5)

    cls = functools.partial(
        _evaluate_distributed, expected_row_ids=np.arange(4), attach=rows.attach_val
    )
    evaluator = PromptV2Validation(
        cls_evaluate_fn=cls,
        topology_eval_fn=topology,
        reference=reference,
        u_idx=np.array(["abcd".index(u) for u, _ in universe_pairs]),
        v_idx=np.array(["abcd".index(v) for _, v in universe_pairs]),
        rows=rows,
        expected_row_ids=np.arange(4),
        stage=stage,
    )
    return evaluator, model, calls


def test_stage_one_records_all_perturbations_and_clean_selector() -> None:
    evaluator, model, calls = _fixture(1)
    outcome = evaluator(model, _prompt_batches(1, 4), Accelerator(cpu=True))
    assert outcome.topology is not None and outcome.topology.threshold == 0.5
    assert outcome.diagnostics is not None
    assert len(calls) == 4
    for name in ("shrink", "noise", "both"):
        for key in (
            "cls_auprc",
            "cls_bce",
            "cls_brier",
            "logit_change_mean",
            "logit_change_p95",
            "fixed_gs",
            "fixed_rd",
            "fixed_degree_mmd",
            "fixed_clustering_mmd",
            "fixed_spectral_mmd",
            "reselected_gs",
            "reselected_rd",
            "reselected_degree_mmd",
            "reselected_clustering_mmd",
            "reselected_spectral_mmd",
        ):
            assert f"sensitivity_{name}_{key}" in outcome.diagnostics
    # Whole-universe fixed-seed perturbations are reproducible across epochs.
    evaluator(model, _prompt_batches(1, 4), Accelerator(cpu=True))
    for first, second in zip(calls[1:4], calls[5:8], strict=True):
        torch.testing.assert_close(first, second)
        assert first.shape == (10, COORD_DIM)


def test_stage_two_stability_survives_resume_and_rejects_wrong_epoch() -> None:
    evaluator, model, _ = _fixture(2)
    accelerator = Accelerator(cpu=True)
    first = evaluator(model, _prompt_batches(1, 4), accelerator)
    assert not first.diagnostics
    state = evaluator.state_dict()
    restored, _, _ = _fixture(2)
    restored.load_state_dict(state, completed_epoch=1)
    outcome = restored(model, _prompt_batches(1, 4), accelerator)
    assert outcome.diagnostics is not None
    assert outcome.diagnostics["stability_edge_jaccard"] == 1.0
    assert outcome.diagnostics["affine_residual_rmse"] < 1e-12
    assert outcome.diagnostics["stability_degree_change_a"] == 0.0
    assert restored.state_dict()["epoch"] == 2
    with pytest.raises(ValueError, match="epoch/universe"):
        restored.load_state_dict(state, completed_epoch=2)


def test_diagnostic_metric_failures_are_wrapped_for_all_ranks() -> None:
    def fail() -> dict[str, float]:
        raise ValueError("invalid graph metric")

    with pytest.raises(RuntimeError, match="invalid graph metric"):
        PromptV2Validation._collect_metrics(Accelerator(cpu=True), fail)
