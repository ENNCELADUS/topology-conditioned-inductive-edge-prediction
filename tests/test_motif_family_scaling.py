"""The F4 family-scaling arithmetic and its decision rule."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import ATTACH_L, CLOSURE_U, CLOSURE_V, INTERIOR, N_EDGES
from src.experiments.motif_family_scaling import (
    EMPTY_SCALE,
    FAMILY_SLICES,
    FULL_SCALE,
    NEAR_ZERO_FRACTION,
    NEAR_ZERO_SCALE,
    distinguishes_near_zero_from_empty,
    markdown_table,
    scale_family,
)


def _weights(rows: int = 4) -> torch.Tensor:
    """A synthetic bank with a distinct constant on each edge block."""
    weights = torch.zeros(rows, N_EDGES)
    weights[:, CLOSURE_U] = 0.4
    weights[:, CLOSURE_V] = 0.6
    weights[:, ATTACH_L] = 0.5
    weights[:, slice(ATTACH_L.stop, INTERIOR.start)] = 0.7
    weights[:, INTERIOR] = 1.0
    return weights


def test_the_two_family_slices_partition_the_template() -> None:
    closure, bridge = FAMILY_SLICES["closure"], FAMILY_SLICES["bridge"]
    assert closure.start == 0
    assert closure.stop == bridge.start == 16
    assert bridge.stop == N_EDGES


def test_scale_family_touches_only_its_own_block() -> None:
    weights = _weights()
    scaled = scale_family(weights, "closure", 0.1)
    assert torch.allclose(scaled[:, FAMILY_SLICES["closure"]], weights[:, :16] * 0.1)
    assert torch.equal(scaled[:, 16:], weights[:, 16:])

    bridged = scale_family(weights, "bridge", 0.5)
    assert torch.equal(bridged[:, :16], weights[:, :16])
    assert torch.allclose(bridged[:, 16:], weights[:, 16:] * 0.5)


def test_scale_family_at_zero_empties_the_family_and_leaves_the_input_alone() -> None:
    weights = _weights()
    emptied = scale_family(weights, "bridge", EMPTY_SCALE)
    assert float(emptied[:, 16:].abs().max()) == 0.0
    assert float(weights[:, 16:].abs().max()) == 1.0


def test_scale_family_rejects_an_unknown_family_and_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="family must be one of"):
        scale_family(_weights(), "wedge", 1.0)
    with pytest.raises(ValueError, match="shape"):
        scale_family(torch.zeros(4, 12), "closure", 1.0)


def test_decision_rule_fires_only_above_the_fraction_of_the_full_effect() -> None:
    # Full effect 1.0; the near-zero effect must exceed 0.1 to count.
    strong = {EMPTY_SCALE: 0.0, NEAR_ZERO_SCALE: 0.2, FULL_SCALE: 1.0}
    weak = {EMPTY_SCALE: 0.0, NEAR_ZERO_SCALE: NEAR_ZERO_FRACTION, FULL_SCALE: 1.0}
    assert distinguishes_near_zero_from_empty(strong) is True
    assert distinguishes_near_zero_from_empty(weak) is False


def test_decision_rule_uses_absolute_effects_in_both_directions() -> None:
    negative = {EMPTY_SCALE: 0.5, NEAR_ZERO_SCALE: 0.2, FULL_SCALE: -0.5}
    assert distinguishes_near_zero_from_empty(negative) is True


def test_decision_rule_needs_all_three_reference_scales() -> None:
    with pytest.raises(KeyError):
        distinguishes_near_zero_from_empty({EMPTY_SCALE: 0.0, FULL_SCALE: 1.0})


def test_markdown_table_renders_a_hand_built_report() -> None:
    entry = {
        "scale": 0.01,
        "mean_abs_rrwp_delta_vs_empty": 0.001,
        "token_change_norm_vs_empty": {
            "topo_u": 0.1,
            "topo_v": 0.2,
            "topo_rel": 0.3,
            "topo_cnt": 0.4,
        },
        "mean_logit": -0.5,
        "logit_change_vs_empty": {"mean": 0.02, "std": 0.01},
        "auroc": 0.81,
        "auprc": 0.82,
    }
    rendered = markdown_table(
        {
            "family": "closure",
            "rows": 2000,
            "checkpoint_id": "abc",
            "by_scale": {"0.01": entry},
            "decision": {
                "rule": "r",
                "near_zero_effect": 0.02,
                "full_effect": 1.0,
                "distinguishes_near_zero_from_empty": False,
            },
        }
    )
    assert "Motif family-scaling check (F4) -- closure" in rendered
    assert "0.100/0.200/0.300/0.400" in rendered
    assert "distinguishes_near_zero_from_empty = False" in rendered
