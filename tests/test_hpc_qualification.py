"""Contracts for the fail-closed EgoStitch qualification surface."""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "hpc" / "qualification.sh"
CONFIG_DIR = REPO_ROOT / "configs"
CONFIGS = {
    "full": CONFIG_DIR / "egostitch_e2e_v3_full_breadth_first.yaml",
    "f_only": CONFIG_DIR / "egostitch_e2e_v3_f_only_breadth_first.yaml",
    "pair_topology": CONFIG_DIR / "egostitch_e2e_v3_pair_topology_breadth_first.yaml",
    "p0": CONFIG_DIR / "egostitch_e2e_v3_p0_breadth_first.yaml",
    "cosine_pool": CONFIG_DIR / "egostitch_e2e_v3_cosine_pool_breadth_first.yaml",
    "no_l_rel": CONFIG_DIR / "egostitch_e2e_v3_no_l_rel_breadth_first.yaml",
}


@pytest.fixture(scope="module")
def bash_exe() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash not found on PATH")
    return executable


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def test_e2e_configs_are_exact_registered_arms() -> None:
    configs = {name: _load_config(path) for name, path in CONFIGS.items()}
    expected = {
        "full": ("none", 0.15, 0.15, "full", 50),
        "f_only": ("all_head", 0.15, 0.15, "f_only", 50),
        "pair_topology": ("content_head", 0.15, 0.15, "pair_topology", 50),
        "p0": ("none", 0.0, 0.0, "p0", 50),
        # The vocabulary-attribution ablation pins the status-quo pool at 20.
        "cosine_pool": ("none", 0.15, 0.15, "cosine_pool", 20),
        "no_l_rel": ("none", 0.15, 0.15, "no_l_rel", 50),
    }
    for arm, (permanent_null, p_topo, p_cont, output_leaf, n_ground) in expected.items():
        config = configs[arm]
        model = config["model"]
        assert isinstance(model, dict)
        model_config = model["config"]
        assert isinstance(model_config, dict)
        assert model["family"] == "egostitch_e2e"
        assert model_config["permanent_null"] == permanent_null
        assert model_config["p_topo"] == p_topo
        assert model_config["p_cont"] == p_cont
        assert model_config["n_ground"] == n_ground
        assert str(config["output_dir"]).endswith(f"/{output_leaf}")

    # The no_l_rel arm is defined as full with the relational weight zeroed.
    no_l_rel_config = configs["no_l_rel"]["model"]["config"]
    assert isinstance(no_l_rel_config, dict)
    assert no_l_rel_config["w_rel"] == 0
    assert "w_rel" not in configs["full"]["model"]["config"]


def test_e2e_configs_pin_training_contract_and_registration() -> None:
    common: dict[str, object] | None = None
    for path in CONFIGS.values():
        config = _load_config(path)
        training = config["training"]
        assert isinstance(training, dict)
        assert training["positive_weight"] == 5.0
        assert training["phase_a_fraction"] == 0.2
        assert training["phase_b_fraction"] == 0.1
        assert training["lr_peak"] == 1.0e-4
        assert training["min_lr"] == 1.0e-5
        assert training["warmup_steps"] == 500
        assert training["residual_ratio_min"] == 1.0e-3
        assert config["preregistration"] == (
            "docs/registrations/g5_e2e_stage1_preregistration_v3.json"
        )
        optim = config["optim"]
        assert isinstance(optim, dict)
        assert optim["lr"] == 1.0e-4
        assert optim["warmup_steps"] == 500
        runtime = config["runtime"]
        assert isinstance(runtime, dict)
        assert runtime["world_size"] == "auto"
        assert runtime["token_budget_candidates"] == [128]
        assert runtime["setup_probe_budget_seconds"] == 2100
        assert runtime["train_eval_budget_seconds"] == 26400
        assert runtime["total_budget_seconds"] == 30300
        if common is None:
            common = training
        else:
            assert training == common


def test_e2e_runner_is_executable_strict_bash(bash_exe: str) -> None:
    text = RUNNER.read_text()
    assert text.splitlines()[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in text
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        [bash_exe, "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_e2e_runner_help_is_local_and_distinguishes_gpu_contexts(bash_exe: str) -> None:
    result = subprocess.run(
        [bash_exe, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "auto-detect and use every visible H20" in result.stdout
    assert "exactly 4 visible NVIDIA H20s" in result.stdout
    assert "sanity -> registered 2,000-step overfit -> probes -> gates" in result.stdout
    assert "It never opens V_qual" in result.stdout
    # The six trained arms are selectable; the two scoring-time controls are not.
    assert "full|f_only|pair_topology|p0|cosine_pool|no_l_rel" in result.stdout


def test_e2e_runner_is_fail_closed_and_never_auto_binds() -> None:
    text = RUNNER.read_text()
    prebinding = text[text.index("enter_qualification_attempt()") : text.index("formal_config()")]
    calibration = text[text.index("run_calibration()") : text.index("run_rehearsal()")]
    rehearsal = text[text.index("run_rehearsal()") : text.index("formal_config()")]

    assert calibration.index("stage 1/3: sanity") < calibration.index(
        "stage 2/3: registered 2,000-step overfit"
    )
    assert calibration.index("stage 2/3: registered 2,000-step overfit") < (
        calibration.index("stage 3/3: probes and pre-binding gates")
    )
    assert "--run-kind overfit" in calibration
    assert calibration.index("select_all_visible_h20s") < calibration.index(
        "--run-kind overfit"
    )
    assert "--run-kind rehearsal" in rehearsal
    assert rehearsal.index("select_all_visible_h20s") < rehearsal.index(
        "--run-kind rehearsal"
    )

    assert "OVERFIT_GPU_COUNT" not in text
    assert "OVERFIT_GPU_IDS" not in text
    assert "assert_gpu_selection" not in text
    assert "egostitch_e2e_v_fit" in calibration
    assert "egostitch_e2e_v_qual" in rehearsal
    assert "attempt_number" in prebinding
    assert "10#${attempt_number} >= 1 && 10#${attempt_number} <= 3" in prebinding
    assert "attempt-${attempt_number}" in prebinding
    assert "cannot be replaced" in prebinding
    assert "--max-steps" not in prebinding
    assert "tests/test_train_egostitch_training.py" in text
    assert "tests/test_train_egostitch_e2e.py" in text
    assert "tests/model/test_egostitch_conditioning.py" in text
    assert "tests/experiments/test_prebinding_gates.py" in text
    assert "candidate_test_edges.txt" in text
    assert "test_edges.txt" in text
    assert "v_select" in text.lower()
    assert "test_graph.pkl" in text
    assert "registration_sha256" in text
    assert "registration remains DRAFT" in text
    assert "sed -i" not in text
    assert 'status": "BINDING' not in text


def test_calibration_never_opens_v_qual() -> None:
    text = RUNNER.read_text()
    calibration = text[text.index("run_calibration()") : text.index("run_rehearsal()")]
    assert "egostitch_e2e_v_qual" not in calibration
    assert "qualification_qual" not in calibration
    assert "--run-kind rehearsal" not in calibration
    assert "calibration_fit" in calibration


def test_rehearsal_refuses_unresolved_gate_thresholds() -> None:
    text = RUNNER.read_text()
    rehearsal = text[text.index("run_rehearsal()") : text.index("formal_config()")]
    # The freeze check must run before any V_qual access, not after.
    assert rehearsal.index("assert_prebinding_gates_frozen") < rehearsal.index(
        "--run-kind rehearsal"
    )
    guard = text[
        text.index("assert_prebinding_gates_frozen()") : text.index(
            "assert_prebinding_gates_implementable()"
        )
    ]
    assert "prebinding_qualification" in guard
    assert "REQUIRED-BEFORE-BINDING" in guard


def test_rehearsal_refuses_gates_that_have_no_evaluator() -> None:
    # A frozen threshold is not the same as an implemented gate; V_qual is spent
    # the moment the rehearsal starts, so this must precede it.
    text = RUNNER.read_text()
    rehearsal = text[text.index("run_rehearsal()") : text.index("formal_config()")]
    assert rehearsal.index("assert_prebinding_gates_implementable") < rehearsal.index(
        "--run-kind rehearsal"
    )
    assert "check-implementable" in text


def test_single_v_qual_rehearsal_is_enforced_across_attempts() -> None:
    text = RUNNER.read_text()
    rehearsal = text[text.index("run_rehearsal()") : text.index("formal_config()")]
    # The ledger must be recorded before V_qual is opened, so an interrupted
    # rehearsal still counts as spent.
    assert rehearsal.index("assert_single_v_qual_rehearsal") < rehearsal.index(
        "--run-kind rehearsal"
    )
    # It must live under REPO_ROOT: the attempt roots are separate directories,
    # so a per-root guard would let attempt002 open V_qual again.
    ledger = text[text.index("REHEARSAL_LEDGER=") : text.index("select_all_visible_h20s()")]
    assert "${REPO_ROOT}" in ledger
    assert "already spent" in ledger
    # Calibration must not touch the ledger.
    calibration = text[text.index("run_calibration()") : text.index("run_rehearsal()")]
    assert "REHEARSAL_LEDGER" not in calibration
    assert "assert_single_v_qual_rehearsal" not in calibration


def test_both_prebinding_stages_evaluate_the_registered_gates() -> None:
    text = RUNNER.read_text()
    assert "src.experiments.prebinding_gates evaluate" in text
    assert "src.experiments.probes produce-e2e" in text
    calibration = text[text.index("run_calibration()") : text.index("run_rehearsal()")]
    rehearsal = text[text.index("run_rehearsal()") : text.index("formal_config()")]
    assert "evaluate_stage_gates" in calibration
    assert "evaluate_stage_gates" in rehearsal


def test_formal_path_requires_binding_no_markers_and_four_gpus() -> None:
    text = RUNNER.read_text()
    formal = text[text.index("run_formal()") :]
    assert "select_all_visible_h20s" in formal
    assert '"${DETECTED_GPU_COUNT}" -eq "${FORMAL_GPU_COUNT}"' in formal
    assert "assert_formal_registration" in formal
    assert "assert_full_preflight" in formal
    assert "selected_checkpoint_eligible" in text
    assert 'if [[ "${arm}" != "full" ]]' in formal
    assert "--run-kind formal" in formal
    assert "egostitch_e2e_v_select" in formal
    assert "REQUIRED-BEFORE-BINDING" in text


def test_formal_arm_selector_covers_six_trained_arms_and_rejects_controls() -> None:
    text = RUNNER.read_text()
    selector = text[text.index("formal_config()") : text.index("assert_full_preflight()")]
    for arm in CONFIGS:
        assert f"configs/egostitch_e2e_v3_{'f_only' if arm == 'f_only' else arm}" in selector
    # The scoring-time controls reuse the full arm's checkpoint and must not be
    # launchable as trained arms.
    assert "structure_control_6a_v3" in selector
    assert "structure_control_6e_v1" in selector
    assert "is not trained" in selector
    # No v2 config may be reachable from the v3 launcher.
    assert "configs/egostitch_e2e_breadth_first.yaml" not in text
    assert "preregistration_v2.json" not in text
