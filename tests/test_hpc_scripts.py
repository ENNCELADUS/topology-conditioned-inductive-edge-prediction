"""Static contract tests for the auto-sized H20 execution layer."""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HPC_DIR = REPO_ROOT / "hpc"
RUNNER = HPC_DIR / "run.sh"


@pytest.fixture(scope="module")
def bash_exe() -> str:
    """Return a usable bash executable or skip shell-dependent checks."""
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash not found on PATH")
    return executable


def test_hpc_layer_has_only_runner_and_documentation() -> None:
    """The old scheduler layer is replaced by two directly useful files."""
    assert sorted(path.name for path in HPC_DIR.iterdir()) == ["README.md", "run.sh"]
    assert not (REPO_ROOT / "slurm").exists()


def test_runner_is_valid_executable_bash(bash_exe: str) -> None:
    """The direct runner has valid strict-mode Bash syntax and permissions."""
    assert RUNNER.read_text().splitlines()[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in RUNNER.read_text()
    assert RUNNER.stat().st_mode & stat.S_IXUSR

    result = subprocess.run(
        [bash_exe, "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_help_is_available_without_the_remote_container(bash_exe: str) -> None:
    """Usage can be inspected locally before the fixed remote paths are checked."""
    result = subprocess.run(
        [bash_exe, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for command in ("check", "train", "score", "merge", "g1", "g2"):
        assert f"hpc/run.sh {command}" in result.stdout
    # The EgoStitch Stage-1 worker routes through the same train entry.
    assert "--worker-module src.train_egostitch" in result.stdout


def test_runner_discovers_visible_h20s() -> None:
    text = RUNNER.read_text()
    for value in (
        "/2023533015/topology-conditioned-inductive-edge-prediction",
        "/2023533015/.uv/bin/uv",
        "NVIDIA H20",
        "expected at least one visible GPU",
        'CUDA_VISIBLE_DEVICES="${GPU_IDS}"',
    ):
        assert value in text
    assert "-m src.e2_pipeline" in text
    assert "automatically" in (HPC_DIR / "README.md").read_text()


def test_docs_describe_auto_sized_h20_runtime() -> None:
    hpc = (HPC_DIR / "README.md").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    config = (REPO_ROOT / "configs" / "b0_v31_breadth_first.yaml").read_text()

    assert "all visible NVIDIA H20" in " ".join(hpc.split())
    assert "auto-detected" in " ".join(readme.split())
    assert "auto-detected" in " ".join(claude.split())
    assert "verified 2026-07-10" not in hpc
    assert "The verified host" not in readme
    assert "single-H20 container" not in config


def test_runner_dispatches_to_the_implemented_clis() -> None:
    """Each command maps directly to one repository Python module."""
    text = RUNNER.read_text()
    assert "-m src.e2_pipeline" in text
    assert "-m src.score_universe score --device cuda --amp bf16" in text
    assert "-m src.score_universe merge" in text
    assert "-m src.experiments.g1_hardened_e2" in text
    assert "-m src.experiments.g2_ceiling" in text


@pytest.mark.parametrize(
    "config_name",
    ["b0_v31_breadth_first.yaml", "b0_alt_breadth_first.yaml"],
)
def test_hpc_training_configs_pin_bf16(config_name: str) -> None:
    """Both implemented baseline runs use H20-native BF16 in the fixed environment."""
    text = (REPO_ROOT / "configs" / config_name).read_text()
    assert 'mixed_precision: "bf16"' in text


def test_primary_docs_reference_only_the_direct_hpc_layer() -> None:
    """Contributor-facing docs do not route experiments through the removed layer."""
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md", HPC_DIR / "README.md"):
        text = path.read_text()
        assert "slurm/" not in text.lower()
        assert "sbatch" not in text.lower()


def test_specs_pin_auto_sized_h20_e2_training() -> None:
    spec = (REPO_ROOT / "docs" / "05-egostitch-spec.md").read_text()
    protocol = (REPO_ROOT / "docs" / "03-experiment-protocol.md").read_text()

    assert "all visible NVIDIA H20" in spec
    assert "automatically detected" in spec
    assert "GPU-count-independent H20 execution design" in spec
    assert "60 minutes" in spec
    assert "30 epochs" in spec
    assert "validation after every epoch" in spec
    assert "fixed 30-epoch" in protocol
    assert "quality is reported but is not the throughput acceptance gate" in protocol
