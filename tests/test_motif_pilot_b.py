"""Pilot B's pure readings: node-disjoint rows, the constant fit, transplants, verdicts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.data.motif_template import ATTACH_L, CLOSURE_U, CLOSURE_V, INTERIOR, N_EDGES
from src.experiments.motif_pilot_b import (
    LEVEL1_MASS_FACTOR,
    LEVEL1_RECONSTRUCTION_MAX,
    LEVEL2_MIN_RELATIVE_RISE,
    GraphLossWeights,
    fit_constant_template,
    gate_logit_summary,
    level1_verdict,
    level2_verdict,
    level3_verdict,
    markdown_summary,
    mass_report,
    rows_inside,
    sorted_profile_dispersion,
    split_training_nodes,
    transplant_permutations,
    transplant_report,
)

WEIGHTS = GraphLossWeights(
    beta_p=1.0, beta_q=1.0, beta_a=1.0, beta_i=1.0, beta_c=1.0, huber_delta=1.0
)
NODES = [f"node_{index:04d}" for index in range(40)]


def _templates(rows: int, *, seed: int) -> torch.Tensor:
    """Synthetic row-dependent templates: a random closure/attachment/interior mix."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.zeros(rows, N_EDGES)
    weights[:, CLOSURE_U] = torch.rand(rows, 8, generator=generator)
    weights[:, CLOSURE_V] = weights[:, CLOSURE_U]
    weights[:, ATTACH_L] = torch.rand(rows, 8, generator=generator)
    weights[:, slice(ATTACH_L.stop, INTERIOR.start)] = torch.rand(rows, 8, generator=generator)
    weights[:, INTERIOR] = (torch.rand(rows, 64, generator=generator) > 0.7).float()
    return weights


# ---------------------------------------------------------------------------
# Node-disjoint row selection
# ---------------------------------------------------------------------------


def test_split_training_nodes_is_a_disjoint_cover_and_is_seeded() -> None:
    fit, held = split_training_nodes(NODES, seed=3)
    assert not (fit & held)
    assert fit | held == frozenset(NODES)
    assert split_training_nodes(NODES, seed=3) == (fit, held)
    assert split_training_nodes(NODES, seed=4) != (fit, held)


def test_split_training_nodes_needs_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        split_training_nodes(["node_0000"], seed=0)


def test_rows_inside_keeps_only_nonself_pairs_of_one_side() -> None:
    fit, held = split_training_nodes(NODES, seed=1)
    positives = [(u, v) for u in NODES for v in NODES if u < v][:200]
    negatives = [(u, v) for u in NODES for v in NODES if u < v][200:400]
    selfish = [(u, u) for u in NODES[:5]]
    fit_pairs, fit_labels = rows_inside(positives + selfish, negatives, fit, limit=0, seed=0)
    held_pairs, _ = rows_inside(positives + selfish, negatives, held, limit=0, seed=0)

    assert fit_pairs, "the fixture must produce at least one fit row"
    assert all(u != v for u, v in fit_pairs)
    assert all(u in fit and v in fit for u, v in fit_pairs)
    assert all(u in held and v in held for u, v in held_pairs)
    assert set(fit_pairs).isdisjoint(held_pairs)
    # No node of a fit row appears in a held-out row: the split is node-disjoint.
    fit_nodes = {node for pair in fit_pairs for node in pair}
    held_nodes = {node for pair in held_pairs for node in pair}
    assert not (fit_nodes & held_nodes)
    assert len(fit_labels) == len(fit_pairs)
    assert set(fit_labels.tolist()) <= {0, 1}


def test_rows_inside_limit_draws_at_the_corpus_ratio_and_is_deterministic() -> None:
    side = frozenset(NODES)
    positives = [(u, v) for u in NODES for v in NODES if u < v][:100]
    negatives = [(u, v) for u in NODES for v in NODES if u < v][100:300]
    pairs, labels = rows_inside(positives, negatives, side, limit=24, seed=5)
    # The training stream is 1:5, so a 24-row draw is 4 positives and 20 negatives.
    assert len(pairs) == 24
    assert int((labels == 1).sum()) == 4
    assert int((labels == 0).sum()) == 20
    again, _ = rows_inside(positives, negatives, side, limit=24, seed=5)
    assert again == pairs
    balanced, balanced_labels = rows_inside(
        positives, negatives, side, limit=20, seed=5, negative_ratio=1
    )
    assert len(balanced) == 20 and int((balanced_labels == 1).sum()) == 10
    with pytest.raises(ValueError, match="negative_ratio"):
        rows_inside(positives, negatives, side, limit=20, seed=5, negative_ratio=0)


# ---------------------------------------------------------------------------
# Level 1: the fitted constant
# ---------------------------------------------------------------------------


def test_fitted_constant_lowers_the_loss_on_its_own_fit_rows() -> None:
    target = _templates(64, seed=11)
    start = torch.full((1, N_EDGES), 0.5).expand(target.size(0), -1)
    fitted = fit_constant_template(target, WEIGHTS, steps=250, batch=32, seed=7)
    assert fitted.shape == (N_EDGES,)
    constant = fitted.unsqueeze(0).expand(target.size(0), -1)
    assert WEIGHTS.mean(constant, target) < WEIGHTS.mean(start, target)


def test_fit_constant_template_rejects_an_empty_table() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_constant_template(torch.zeros(0, N_EDGES), WEIGHTS, steps=1)


def test_mass_report_is_exact_when_the_prediction_is_the_target() -> None:
    target = _templates(32, seed=13)
    report = mass_report(target, target)
    wedge = report["wedge_mass"]
    assert isinstance(wedge, dict)
    assert wedge["normalised_reconstruction_on_positive_rows"] == pytest.approx(0.0)
    assert wedge["predicted_over_true_mean"] == pytest.approx(1.0)


def test_mass_report_records_false_mass_on_zero_rows() -> None:
    target = _templates(16, seed=17)
    target[:, CLOSURE_U] = 0.0
    target[:, CLOSURE_V] = 0.0
    predicted = target.clone()
    predicted[:, CLOSURE_U] = 0.5
    predicted[:, CLOSURE_V] = 0.5
    wedge = mass_report(predicted, target)["wedge_mass"]
    assert isinstance(wedge, dict)
    assert wedge["n_positive_target_rows"] == 0
    assert wedge["false_mass_on_zero_rows"] == pytest.approx(8 * 0.25)


def test_sorted_profile_dispersion_sees_a_constant_generator_as_zero_spread() -> None:
    target = _templates(24, seed=19)
    constant = torch.full((24, N_EDGES), 0.3)
    report = sorted_profile_dispersion(constant, target)
    assert report["closure_raw"]["predicted_coord_std_over_s"] == pytest.approx(0.0)
    true_spread = report["closure_raw"]["true_coord_std_over_s"]
    assert isinstance(true_spread, float) and true_spread > 0.0


def test_gate_logit_summary_reports_saturation_of_a_tiny_closure_block() -> None:
    predicted = torch.full((8, N_EDGES), 0.2)
    predicted[:, CLOSURE_U] = 1e-4
    predicted[:, CLOSURE_V] = 1e-4
    summary = gate_logit_summary(predicted)
    assert summary["closure"]["frac_abs_gt_3"] == pytest.approx(1.0)
    assert summary["closure"]["mean"] < -3.0
    assert summary["interior"]["frac_abs_gt_3"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Level 2: transplants
# ---------------------------------------------------------------------------


def test_transplant_permutations_are_never_the_identity() -> None:
    drawn = transplant_permutations(2, count=6, seed=0)
    assert len(drawn) == 6
    identity = np.arange(2)
    for permutation in drawn:
        assert sorted(permutation.tolist()) == [0, 1]
        assert not np.array_equal(permutation, identity)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(drawn, transplant_permutations(2, count=6, seed=0), strict=True)
    )


def test_transplant_permutations_reject_degenerate_arguments() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        transplant_permutations(1, count=1, seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        transplant_permutations(4, count=-1, seed=0)


def test_transplant_report_matches_the_hand_computed_rise() -> None:
    target = _templates(12, seed=23)
    predicted = target.clone().clamp(0.05, 0.95)
    permutations = transplant_permutations(12, count=3, seed=2)
    report = transplant_report(predicted, target, WEIGHTS, permutations)
    base = WEIGHTS.mean(predicted, target)
    assert report["base_loss"] == pytest.approx(base)
    assert report["permutations"] == 3
    entries = report["per_permutation"]
    assert isinstance(entries, list)
    for entry, permutation in zip(entries, permutations, strict=True):
        shuffled = target.index_select(0, torch.from_numpy(permutation))
        expected = WEIGHTS.mean(predicted, shuffled)
        assert entry["loss"] == pytest.approx(expected)
        assert entry["relative_rise"] == pytest.approx((expected - base) / base)
    assert report["relative_rise_min"] == pytest.approx(
        min(entry["relative_rise"] for entry in entries)
    )


def test_a_constant_generator_raises_nothing_under_transplant() -> None:
    target = _templates(16, seed=29)
    constant = torch.full((16, N_EDGES), 0.3)
    report = transplant_report(
        constant, target, WEIGHTS, transplant_permutations(16, count=4, seed=1)
    )
    entries = report["per_permutation"]
    assert isinstance(entries, list)
    assert all(abs(entry["relative_rise"]) < LEVEL2_MIN_RELATIVE_RISE for entry in entries)


# ---------------------------------------------------------------------------
# Pre-registered verdicts
# ---------------------------------------------------------------------------


def _universe(
    *,
    generator: float,
    constant: float,
    reconstruction: float,
    ratio: float,
    rises: list[float],
) -> dict[str, object]:
    """One hand-built universe payload shaped like `_universe_report`'s output."""
    return {
        "rows": 10,
        "L_G": {"generator": generator, "fitted_constant_template": constant},
        "masses": {
            "wedge_mass": {
                "normalised_reconstruction_on_positive_rows": reconstruction,
                "predicted_over_true_mean": ratio,
            }
        },
        "transplant": {"per_permutation": [{"relative_rise": rise} for rise in rises]},
    }


def test_level1_passes_only_when_every_check_holds_on_every_universe() -> None:
    good = _universe(
        generator=0.05,
        constant=0.09,
        reconstruction=LEVEL1_RECONSTRUCTION_MAX - 0.1,
        ratio=1.5,
        rises=[],
    )
    verdict = level1_verdict({"heldout_train": good, "val_cls": good})
    assert verdict["passed"] is True

    worse_fit = _universe(generator=0.11, constant=0.09, reconstruction=0.1, ratio=1.0, rises=[])
    assert level1_verdict({"heldout_train": good, "val_cls": worse_fit})["passed"] is False


def test_level1_fails_on_reconstruction_and_on_mass_scale() -> None:
    reconstruction = _universe(
        generator=0.01,
        constant=0.5,
        reconstruction=LEVEL1_RECONSTRUCTION_MAX + 0.01,
        ratio=1.0,
        rises=[],
    )
    assert level1_verdict({"u": reconstruction})["passed"] is False
    mass = _universe(
        generator=0.01,
        constant=0.5,
        reconstruction=0.1,
        ratio=LEVEL1_MASS_FACTOR + 0.1,
        rises=[],
    )
    verdict = level1_verdict({"u": mass})
    assert verdict["passed"] is False
    assert verdict["wedge_mass_within_factor"] == {"u": False}


def test_level2_needs_every_permutation_on_every_universe() -> None:
    passing = _universe(
        generator=0.0, constant=0.0, reconstruction=0.0, ratio=1.0, rises=[0.30, 0.42, 0.25]
    )
    failing = _universe(
        generator=0.0, constant=0.0, reconstruction=0.0, ratio=1.0, rises=[0.30, 0.01]
    )
    assert level2_verdict({"a": passing})["passed"] is True
    assert level2_verdict({"a": passing, "b": failing})["passed"] is False
    empty = _universe(generator=0.0, constant=0.0, reconstruction=0.0, ratio=1.0, rises=[])
    assert level2_verdict({"a": empty})["passed"] is False


def test_level3_compares_both_predicted_cells_against_gates_off() -> None:
    cells = {
        "s1_pred": {"auroc": 0.8, "auprc": 0.83},
        "s2_pred": {"auroc": 0.8, "auprc": 0.82},
        "s2_gates_off": {"auroc": 0.78, "auprc": 0.8136},
    }
    assert level3_verdict(cells)["passed"] is True
    cells["s2_pred"] = {"auroc": 0.7, "auprc": 0.80}
    verdict = level3_verdict(cells)
    assert verdict["passed"] is False
    assert verdict["beats_gates_off"] == {"s1_pred": True, "s2_pred": False}


def test_level3_fails_closed_without_a_gates_off_cell() -> None:
    cells = {"s1_pred": {"auroc": 0.8, "auprc": 0.83}, "s2_pred": {"auroc": 0.8, "auprc": 0.82}}
    assert level3_verdict(cells)["passed"] is False


def test_markdown_summary_renders_a_hand_built_report() -> None:
    universe = _universe(
        generator=0.05, constant=0.09, reconstruction=0.2, ratio=1.1, rises=[0.4, 0.5]
    )
    transplant = universe["transplant"]
    assert isinstance(transplant, dict)
    transplant.update(
        {"relative_rise_mean": 0.45, "relative_rise_min": 0.4, "relative_rise_max": 0.5}
    )
    losses = universe["L_G"]
    assert isinstance(losses, dict)
    losses["mean_template"] = 0.12
    cells = {"s2_pred": {"auroc": 0.8, "auprc": 0.82}, "s2_gates_off": {"auroc": 0.7, "auprc": 0.8}}
    report = {
        "checkpoint_id": "abc",
        "stage1_checkpoint_id": "def",
        "level_1_fit": {"val_cls": universe},
        "level_3_downstream": {"cells": cells},
        "verdicts": {
            "level_1": {"passed": True, "rule": "r1"},
            "level_2": {"passed": False, "rule": "r2"},
        },
    }
    rendered = markdown_summary(report)
    assert "# Motif Stage II pilot B" in rendered
    assert "val_cls" in rendered
    assert "**level_1**: PASS" in rendered
    assert "**level_2**: FAIL" in rendered
