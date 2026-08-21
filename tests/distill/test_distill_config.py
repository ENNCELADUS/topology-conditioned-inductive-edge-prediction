"""`src.distill.config.DistillConfig`: arm patterns and mapping validation."""

from __future__ import annotations

import pytest
from src.distill.config import DistillConfig

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- defaults / arms


def test_defaults_are_inactive_with_arm_none() -> None:
    cfg = DistillConfig()
    assert not cfg.active
    assert cfg.arm == "none"


def test_kd_logit_arm_pattern() -> None:
    cfg = DistillConfig(targets_path="t", w_logit=1.0)
    assert cfg.active
    assert cfg.arm == "kd_logit"


def test_kd_rank_arm_pattern() -> None:
    cfg = DistillConfig(targets_path="t", w_rank=1.0, w_dist=1.0)
    assert cfg.active
    assert cfg.arm == "kd_rank"


def test_kd_gram_arm_pattern() -> None:
    cfg = DistillConfig(targets_path="t", w_gram=1.0)
    assert cfg.active
    assert cfg.arm == "kd_gram"


def test_kd_rep_arm_pattern() -> None:
    cfg = DistillConfig(targets_path="t", w_rep=1.0)
    assert cfg.active
    assert cfg.arm == "kd_rep"


def test_kd_d9_arm_pattern_requires_all_three_weights_together() -> None:
    cfg = DistillConfig(targets_path="t", w_seed=1.0, w_geom=0.3, w_kl=0.05)
    assert cfg.active
    assert cfg.arm == "kd_d9"


# --------------------------------------------------------------------------- illegal patterns


def test_mixed_arm_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_logit=1.0, w_rep=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_logit=1.0, w_seed=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_rank=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_dist=1.0)
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig(targets_path="t", w_gram=1.0, w_rep=1.0)


@pytest.mark.parametrize(
    "weights",
    [
        {"w_seed": 1.0},
        {"w_seed": 1.0, "w_geom": 0.3},
        {"w_seed": 1.0, "w_kl": 0.05},
        {"w_geom": 0.3, "w_kl": 0.05},
        {"w_seed": 1.0, "w_geom": 0.3, "w_kl": 0.05, "w_logit": 1.0},
    ],
)
def test_partial_kd_d9_patterns_are_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="exactly one arm group"):
        DistillConfig.from_mapping({"targets_path": "t", **weights})


# --------------------------------------------------------------------------- bounds


@pytest.mark.parametrize(
    "name", ["w_logit", "w_rank", "w_dist", "w_gram", "w_rep", "w_seed", "w_geom", "w_kl"]
)
def test_negative_weights_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DistillConfig(**{name: -0.1})  # type: ignore[arg-type]


def test_negative_kl_warmup_steps_is_rejected() -> None:
    with pytest.raises(ValueError, match="kl_warmup_steps"):
        DistillConfig(kl_warmup_steps=-1)


def test_negative_joint_start_epoch_is_rejected() -> None:
    with pytest.raises(ValueError, match="joint_start_epoch"):
        DistillConfig(joint_start_epoch=-1)


def test_non_positive_gen_lr_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="gen_lr_scale"):
        DistillConfig(gen_lr_scale=0.0)
    with pytest.raises(ValueError, match="gen_lr_scale"):
        DistillConfig(gen_lr_scale=-0.1)


def test_invalid_rank_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="margin"):
        DistillConfig(margin=-0.1)
    with pytest.raises(ValueError, match="temperature"):
        DistillConfig(temperature=0.0)


def test_nonzero_weight_without_targets_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_logit=1.0)
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_rank=1.0, w_dist=1.0)
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_gram=1.0)
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_rep=1.0)
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig(w_seed=1.0, w_geom=0.3, w_kl=0.05)


# --------------------------------------------------------------------------- from_mapping


def test_from_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown distill config keys"):
        DistillConfig.from_mapping({"weight_of_destiny": 1.0})


def test_from_mapping_rejects_removed_legacy_keys() -> None:
    # anchors_per_step (sampled-context stream) and w_align (kd_d4) belonged
    # to the machinery this rework deleted; pin their removal so a stale
    # config surfaces loudly instead of being silently ignored.
    with pytest.raises(ValueError, match="unknown distill config keys"):
        DistillConfig.from_mapping({"anchors_per_step": 2})
    with pytest.raises(ValueError, match="unknown distill config keys"):
        DistillConfig.from_mapping({"w_align": 1.0})


def test_from_mapping_rejects_bool_for_number_fields() -> None:
    with pytest.raises(ValueError, match="w_logit"):
        DistillConfig.from_mapping({"w_logit": True})
    with pytest.raises(ValueError, match="gen_lr_scale"):
        DistillConfig.from_mapping({"gen_lr_scale": False})


def test_from_mapping_rejects_bool_for_int_fields() -> None:
    with pytest.raises(ValueError, match="kl_warmup_steps"):
        DistillConfig.from_mapping({"kl_warmup_steps": True})
    with pytest.raises(ValueError, match="joint_start_epoch"):
        DistillConfig.from_mapping({"joint_start_epoch": False})


def test_from_mapping_rejects_non_string_targets_path() -> None:
    with pytest.raises(ValueError, match="targets_path"):
        DistillConfig.from_mapping({"targets_path": 7})


def test_from_mapping_rejects_float_for_int_fields() -> None:
    with pytest.raises(ValueError, match="joint_start_epoch"):
        DistillConfig.from_mapping({"joint_start_epoch": 2.5})


def test_from_mapping_accepts_int_for_float_fields() -> None:
    cfg = DistillConfig.from_mapping({"targets_path": "t", "w_logit": 1, "gen_lr_scale": 1})
    assert cfg.w_logit == 1.0
    assert cfg.gen_lr_scale == 1.0


def test_from_mapping_full_kd_d9_roundtrip() -> None:
    cfg = DistillConfig.from_mapping(
        {
            "targets_path": "outputs/distill/kd_row_targets_v4",
            "w_seed": 1.0,
            "w_geom": 0.3,
            "w_kl": 0.05,
            "kl_warmup_steps": 2000,
            "joint_start_epoch": 8,
            "gen_lr_scale": 0.1,
        }
    )
    assert cfg.arm == "kd_d9"
    assert cfg.active
    assert cfg.targets_path == "outputs/distill/kd_row_targets_v4"
    assert cfg.w_seed == 1.0
    assert cfg.w_geom == 0.3
    assert cfg.w_kl == 0.05
    assert cfg.kl_warmup_steps == 2000
    assert cfg.joint_start_epoch == 8
    assert cfg.gen_lr_scale == 0.1
