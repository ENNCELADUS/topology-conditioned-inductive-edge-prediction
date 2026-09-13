"""Validation diagnostics and strict probe fold boundaries."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from src.data.struct_coords import COORD_DIM
from src.experiments.coord_probe import fit_metrics, fold_surfaces
from src.experiments.topo_prompt_diagnostics import score_stability, sensitivity_row


def test_monotone_scaling_keeps_admitted_set() -> None:
    before = np.array([-2.0, 0.0, 1.0, 4.0])
    result = score_stability(
        [("a", "a"), ("a", "b"), ("b", "c"), ("a", "c")], before, before * 3 + 2, 0.5, 3.5
    )
    assert result["stability_edge_jaccard"] == 1.0
    assert result["affine_slope"] == pytest.approx(3.0)
    assert result["affine_residual_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert set(cast(dict[str, int], result["degree_changes"]).values()) == {0}


def test_stability_detects_pair_changes_and_counts_loop_once() -> None:
    result = score_stability(
        [("a", "a"), ("a", "b")], np.array([1.0, -1.0]), np.array([-1.0, 1.0]), 0.0, 0.0
    )
    assert result["stability_edge_jaccard"] == 0.0
    assert result["degree_changes"] == {"a": 0, "b": 1}
    with pytest.raises(ValueError, match="duplicate"):
        score_stability([("a", "b"), ("b", "a")], np.zeros(2), np.zeros(2), 0.0, 0.0)


def test_sensitivity_replays_and_reselects() -> None:
    calls = []

    def evaluator(scores: np.ndarray, threshold: float | None) -> dict[str, object]:
        calls.append(threshold)
        return {"threshold": threshold}

    result = sensitivity_row(
        np.zeros(2),
        np.array([0.0, 2.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        0.3,
        evaluator,
    )
    assert calls == [0.3, None]
    assert result["logit_change_p95"] == pytest.approx(1.9)
    assert result["cls_auprc"] == 1.0


def test_probe_fit_fold_excludes_every_heldout_touch() -> None:
    surfaces = fold_surfaces([("a", "b"), ("a", "c"), ("c", "d"), ("c", "c")], {"c", "d"})
    assert surfaces["in_fold"].tolist() == [0]
    assert surfaces["one_heldout"].tolist() == [1]
    assert surfaces["two_heldout"].tolist() == [2, 3]


def test_endpoint_metrics_exclude_familiar_partner() -> None:
    truth = torch.zeros((3, COORD_DIM))
    truth[:, :9] = torch.arange(3.0)[:, None]
    prediction = truth.clone()
    prediction[:, 9:18] = 1000.0
    mask = torch.tensor([[True, False]] * 3)
    result = fit_metrics(prediction, truth, torch.zeros(3, 5), truth, mask)
    assert result["endpoint_r2"] == 1.0
    assert result["endpoint_count"] == 3


def test_probe_heads_use_coordinate_only_gradients() -> None:
    from src.experiments.coord_probe import _continuous
    from src.model.egostitch.classifier.coord_gen import CoordGenConfig, CoordinateGenerator

    torch.manual_seed(42)
    cfg = CoordGenConfig(endpoint_hidden=256, endpoint_dropout=0.3, endpoint_weight_decay=0.1)
    head = CoordinateGenerator(4, cfg)
    parts = head(torch.randn(4, 8), torch.randn(4, 8))
    prediction = _continuous(parts)
    assert prediction.shape == (4, 30)
    loss = torch.nn.functional.smooth_l1_loss(prediction, torch.zeros_like(prediction))
    loss = loss + torch.nn.functional.cross_entropy(
        parts["distance_logits"], torch.zeros(4, dtype=torch.long)
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_explicit_v1_diagnostic_conversion_preserves_logits(tmp_path: Path) -> None:
    from src.experiments.score_coord_gen_v1_diagnostic import load_v1_diagnostic

    from tests.test_coord_gen_model import _model, _reader_config
    from tests.test_prefix_model import _pair_batch

    model = _model()
    old_state = {
        name.replace("generator.endpoint_head.1.", "generator.endpoint_head."): value
        for name, value in model.state_dict().items()
        if not name.startswith("teacher.")
    }
    path = tmp_path / "epoch.pt"
    torch.save(
        {
            "model_family": "v3_1_coord_gen",
            "model_config": {
                "reader": _reader_config(),
                "coord_gen": {
                    "hidden": 16,
                    "layers": 1,
                    "dropout": 0.0,
                    "w_task": 1.0,
                    "w_kd": 0.1,
                },
            },
            "model_state": old_state,
        },
        path,
    )
    restored, _, _ = load_v1_diagnostic(path)
    batch = _pair_batch()
    with torch.no_grad():
        assert torch.equal(model(batch)["logits"], restored(batch)["logits"])
