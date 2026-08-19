"""`src.distill.config.DistillConfig`: arm patterns and mapping validation."""

from __future__ import annotations

import pytest
from src.distill.config import DistillConfig

pytestmark = pytest.mark.unit


def test_arm_names_follow_the_single_group_pattern() -> None:
    assert DistillConfig().arm == "none"
    assert DistillConfig(targets_path="t", w_label=1.0).arm == "kd_control"
    assert DistillConfig(targets_path="t", w_logit=1.0).arm == "kd_d1"
    assert DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0).arm == "kd_d2"
    assert DistillConfig(targets_path="t", w_gram=1.0).arm == "kd_d3"
    assert DistillConfig(targets_path="t", w_align=1.0).arm == "kd_d4"
    assert DistillConfig(targets_path="t", w_residual=1.0).arm == "kd_d5"
    assert DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0, w_gram=1.0).arm == "kd_d6"
    assert DistillConfig(targets_path="t", w_rowmass=1.0).arm == "kd_d8"


def test_arm_label_overrides_the_mapped_name_when_active() -> None:
    cfg = DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0, arm_label="kd_d7a")
    assert cfg.arm == "kd_d7a"


def test_arm_label_rejects_every_pattern_but_kd_d2() -> None:
    # A free-form label on any other pattern would publish under a false arm
    # name (e.g. a w_gram run reported as kd_d7a).
    for weights in (
        {},
        {"w_gram": 1.0},
        {"w_rank": 1.0, "w_dist": 1.0, "w_gram": 1.0},
        {"w_rowmass": 1.0},
    ):
        with pytest.raises(ValueError, match="arm_label"):
            DistillConfig.from_mapping({"targets_path": "t", "arm_label": "kd_d7a", **weights})


def test_active_reflects_any_nonzero_weight() -> None:
    assert not DistillConfig().active
    assert DistillConfig(targets_path="t", w_label=0.5).active
    assert DistillConfig(targets_path="t", w_align=0.5).active
    assert DistillConfig(targets_path="t", w_residual=0.5).active


def test_mixed_arm_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_label=1.0, w_gram=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_rank=1.0)  # w_rank requires w_dist
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_align=1.0, w_gram=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0, w_align=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_rowmass=1.0, w_logit=1.0)


def test_nonzero_weight_requires_targets_path() -> None:
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_label=1.0)


def test_from_mapping_rejects_unknown_keys_and_bad_types() -> None:
    with pytest.raises(ValueError, match="unknown distill config keys"):
        DistillConfig.from_mapping({"weight_of_destiny": 1.0})
    with pytest.raises(ValueError, match="anchors_per_step"):
        DistillConfig.from_mapping({"anchors_per_step": "two"})
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig.from_mapping({"targets_path": 7})
    with pytest.raises(ValueError, match="arm_label"):
        DistillConfig.from_mapping({"arm_label": 7})


def test_from_mapping_accepts_the_kd_d7a_pattern() -> None:
    cfg = DistillConfig.from_mapping(
        {
            "targets_path": "outputs/distill/kd_targets_heuristic_ra_v1",
            "w_rank": 1.0,
            "w_dist": 1.0,
            "arm_label": "kd_d7a",
        }
    )
    assert cfg.arm == "kd_d7a"


def test_from_mapping_roundtrip() -> None:
    cfg = DistillConfig.from_mapping(
        {
            "targets_path": "outputs/distill/kd_targets",
            "w_rank": 1.0,
            "w_dist": 1.0,
            "temperature": 1.0,
            "margin": 0.1,
            "anchors_per_step": 2,
        }
    )
    assert cfg.arm == "kd_d2"
    assert cfg.anchors_per_step == 2


def test_from_mapping_roundtrips_the_kd_d8_pattern() -> None:
    cfg = DistillConfig.from_mapping(
        {
            "targets_path": "outputs/distill/kd_targets_breadth_first_seed0_v2",
            "w_rowmass": 1.0,
            "anchors_per_step": 2,
        }
    )
    assert cfg.arm == "kd_d8"
    assert cfg.anchors_per_step == 2


def test_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DistillConfig(temperature=0.0)
    with pytest.raises(ValueError, match="margin"):
        DistillConfig(margin=0.0)
    with pytest.raises(ValueError, match="anchors_per_step"):
        DistillConfig(anchors_per_step=0)
    with pytest.raises(ValueError, match="non-negative"):
        DistillConfig(w_label=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        DistillConfig(w_align=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        DistillConfig(w_residual=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        DistillConfig(w_rowmass=-0.1)
