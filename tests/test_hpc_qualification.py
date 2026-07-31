"""Contracts for the historically named single-stage EgoStitch launcher."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "hpc" / "qualification.sh"
CONFIG_DIR = REPO_ROOT / "configs"
CONFIGS = {
    "full": CONFIG_DIR / "egostitch_e2e_v3_full_breadth_first.yaml",
    "f_only": CONFIG_DIR / "egostitch_e2e_v3_f_only_breadth_first.yaml",
    "pair_topology": CONFIG_DIR / "egostitch_e2e_v3_pair_topology_breadth_first.yaml",
    "p0": CONFIG_DIR / "egostitch_e2e_v3_p0_breadth_first.yaml",
    "no_l_rel": CONFIG_DIR / "egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
    "row_layernorm": CONFIG_DIR / "egostitch_e2e_v3_row_layernorm_breadth_first.yaml",
}


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    section = config[name]
    assert isinstance(section, dict)
    return section


def _model_config(config: dict[str, object]) -> dict[str, object]:
    return _section(_section(config, "model"), "config")


def test_e2e_configs_are_exact_registered_arms() -> None:
    configs = {name: _load_config(path) for name, path in CONFIGS.items()}
    expected = {
        "full": ("none", 0.15, 0.15, "full", 50),
        "f_only": ("all_head", 0.15, 0.15, "f_only", 50),
        "pair_topology": ("content_head", 0.15, 0.15, "pair_topology", 50),
        "p0": ("none", 0.0, 0.0, "p0", 50),
        "no_l_rel": ("none", 0.15, 0.15, "no_l_rel", 50),
        "row_layernorm": ("none", 0.15, 0.15, "row_layernorm", 50),
    }
    for arm, (permanent_null, p_topo, p_cont, output_leaf, n_ground) in expected.items():
        config = configs[arm]
        model = _section(config, "model")
        model_config = _model_config(config)
        assert model["family"] == "egostitch_e2e"
        assert model_config["permanent_null"] == permanent_null
        assert model_config["p_topo"] == p_topo
        assert model_config["p_cont"] == p_cont
        assert model_config["n_ground"] == n_ground
        assert str(config["output_dir"]).endswith(f"/{output_leaf}")
    assert _model_config(configs["no_l_rel"])["w_rel"] == 0
    assert "w_rel" not in _model_config(configs["full"])
    assert _model_config(configs["full"])["feature_standardization"] == "zscore_vfit_v1"
    assert (
        _model_config(configs["row_layernorm"])["feature_standardization"] == "row_layernorm"
    )
    for arm in ("f_only", "pair_topology", "p0", "no_l_rel"):
        assert _model_config(configs[arm])["feature_standardization"] == "zscore_vfit_v1"


def test_e2e_configs_pin_training_contract_and_registration() -> None:
    common: dict[str, object] | None = None
    for path in CONFIGS.values():
        config = _load_config(path)
        training = _section(config, "training")
        assert training["positive_weight"] == 5.0
        assert training["phase_a_fraction"] == 0.2
        assert training["phase_b_fraction"] == 0.1
        assert training["lr_peak"] == 1.0e-4
        assert training["min_lr"] == 1.0e-5
        assert training["warmup_steps"] == 500
        assert training["residual_ratio_min"] == 1.0e-3
        assert config["preregistration"] == (
            "docs/registrations/g5_e2e_stage1_preregistration_v5.json"
        )
        runtime = _section(config, "runtime")
        assert runtime["world_size"] == "auto"
        assert runtime["token_budget"] == 128
        assert "token_budget_candidates" not in runtime
        assert runtime["train_eval_budget_seconds"] == 26400
        if common is None:
            common = training
        else:
            assert training == common


def test_e2e_configs_carry_no_hand_pasted_feature_digest() -> None:
    for path in CONFIGS.values():
        assert "feature_stats_sha256" not in _model_config(_load_config(path))


def test_all_six_arms_share_the_single_ng50_pack_and_grounding_cache() -> None:
    """v5: every trained arm is n_ground=50, so one pack and one grounding cache serve all."""
    pack_dirs: set[str] = set()
    grounding_caches: set[str] = set()
    for path in CONFIGS.values():
        config = _load_config(path)
        runtime = _section(config, "runtime")
        data = _section(config, "data")
        assert _model_config(config)["n_ground"] == 50
        assert isinstance(runtime["pack_dir"], str)
        assert isinstance(data["grounding_cache"], str)
        pack_dirs.add(runtime["pack_dir"])
        grounding_caches.add(data["grounding_cache"])
    assert pack_dirs == {"outputs/feature_packs/egostitch_e2e_v_hold_ng50"}
    assert len(grounding_caches) == 1


def test_launcher_is_strict_bash_and_formal_only() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not found")
    subprocess.run([bash, "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "run_formal" in text
    assert "run_qualification" not in text
    assert "calibrate-tolerance" not in text
    assert "<full|f_only|pair_topology|p0|no_l_rel|row_layernorm>" in text
    assert "cosine_pool" not in text


def test_formal_preflight_binds_plan_and_environment_not_quality() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for required in (
        "assert_clean_checkout",
        "assert_source_resolves_to_repo",
        "assert_registration_unchanged",
        "FORMAL_GPU_COUNT=4",
    ):
        assert required in text
    for retired in (
        "qualification.json",
        "pending_manual_review",
        "latest-pass",
        "selected_checkpoint_eligible",
        "validation_liveness_pass",
        "margin_verdict",
        "assert_full_preflight",
        "validate_e2e_qualification_profile",
        "registration_status",
        "status BINDING",
    ):
        assert retired not in text


def test_launcher_forwards_only_the_registered_formal_schedule() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "--run-kind formal" in text
    assert "--qualification-artifact" not in text
    assert "--epochs" not in text
    assert "--max-steps" not in text


def test_scoring_time_controls_are_not_trained() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "structure_control_6a_v3|structure_control_6e_v1" in text
    assert "scoring-time control that reuses the full checkpoint" in text
