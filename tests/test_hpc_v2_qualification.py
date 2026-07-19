"""Contracts for the fail-closed EgoStitch v2 launch surface."""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "hpc" / "v2_qualification.sh"
CONFIG_DIR = REPO_ROOT / "configs"
CONFIGS = {
    "full": CONFIG_DIR / "egostitch_e2e_stability_v2_breadth_first.yaml",
    "f_only": CONFIG_DIR / "egostitch_e2e_stability_v2_f_only_breadth_first.yaml",
    "pair_topology": CONFIG_DIR / "egostitch_e2e_stability_v2_pair_topology_breadth_first.yaml",
    "p0": CONFIG_DIR / "egostitch_e2e_stability_v2_p0_breadth_first.yaml",
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


def test_v2_configs_are_exact_registered_arms() -> None:
    configs = {name: _load_config(path) for name, path in CONFIGS.items()}
    expected = {
        "full": ("none", 0.15, 0.15, "full"),
        "f_only": ("all_head", 0.15, 0.15, "f_only"),
        "pair_topology": ("content_head", 0.15, 0.15, "pair_topology"),
        "p0": ("none", 0.0, 0.0, "p0"),
    }
    for arm, (permanent_null, p_topo, p_cont, output_leaf) in expected.items():
        config = configs[arm]
        model = config["model"]
        assert isinstance(model, dict)
        model_config = model["config"]
        assert isinstance(model_config, dict)
        assert model["family"] == "egostitch_e2e"
        assert model_config["permanent_null"] == permanent_null
        assert model_config["p_topo"] == p_topo
        assert model_config["p_cont"] == p_cont
        assert str(config["output_dir"]).endswith(f"/{output_leaf}")


def test_v2_configs_pin_stability_contract_and_v2_registration() -> None:
    common: dict[str, object] | None = None
    for path in CONFIGS.values():
        config = _load_config(path)
        stability = config["stability_v2"]
        assert isinstance(stability, dict)
        assert stability["positive_weight"] == 5.0
        assert stability["phase_a_fraction"] == 0.2
        assert stability["phase_b_fraction"] == 0.1
        assert stability["lr_peak"] == 1.0e-4
        assert stability["min_lr"] == 1.0e-5
        assert stability["warmup_steps"] == 500
        assert stability["residual_ratio_min"] == 1.0e-3
        assert config["preregistration"] == (
            "docs/registrations/g5_e2e_stage1_preregistration_v2.json"
        )
        optim = config["optim"]
        assert isinstance(optim, dict)
        assert optim["lr"] == 1.0e-4
        assert optim["warmup_steps"] == 500
        runtime = config["runtime"]
        assert isinstance(runtime, dict)
        assert runtime["world_size"] == "auto"
        if common is None:
            common = stability
        else:
            assert stability == common


def test_v2_runner_is_executable_strict_bash(bash_exe: str) -> None:
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


def test_v2_runner_help_is_local_and_distinguishes_gpu_contexts(bash_exe: str) -> None:
    result = subprocess.run(
        [bash_exe, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "exactly 2 visible NVIDIA H20s" in result.stdout
    assert "exactly 4 visible NVIDIA H20s" in result.stdout
    assert "sanity -> registered 2,000-step overfit -> full-arm rehearsal" in result.stdout


def test_v2_runner_is_fail_closed_and_never_auto_binds() -> None:
    text = RUNNER.read_text()
    qualification = text[text.index("run_qualification()") : text.index("formal_config()")]
    assert qualification.index("stage 1/3: sanity") < qualification.index(
        "stage 2/3: registered 2,000-step overfit"
    )
    assert qualification.index("stage 2/3: registered 2,000-step overfit") < (
        qualification.index("stage 3/3: exact full-arm rehearsal")
    )
    assert "--v2-run-kind overfit" in qualification
    assert "--v2-run-kind rehearsal" in qualification
    assert "--max-steps" not in qualification
    assert "candidate_test_edges.txt" in text
    assert "test_edges.txt" in text
    assert "v_select" in text.lower()
    assert "test_graph.pkl" in text
    assert "registration_sha256" in text
    assert "registration remains DRAFT" in text
    assert "sed -i" not in text
    assert "status\": \"BINDING" not in text


def test_formal_path_requires_binding_no_markers_and_four_gpus() -> None:
    text = RUNNER.read_text()
    formal = text[text.index("run_formal()") :]
    assert (
        'assert_gpu_selection "${FORMAL_GPU_COUNT}" "${FORMAL_GPU_IDS}"' in formal
    )
    assert "assert_formal_registration" in formal
    assert "assert_full_preflight" in formal
    assert 'if [[ "${arm}" != "full" ]]' in formal
    assert "--v2-run-kind formal" in formal
    assert "REQUIRED-BEFORE-BINDING" in text
