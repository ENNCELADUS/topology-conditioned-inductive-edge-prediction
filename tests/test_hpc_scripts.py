"""Static contract tests for the pinned single-H20 execution layer."""

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
    for command in ("check", "train", "score", "metrics", "merge", "g1", "g2", "test-b0-v31"):
        assert f"hpc/run.sh {command}" in result.stdout


def test_runner_pins_the_verified_single_h20_environment() -> None:
    """The shipped runner encodes the verified container rather than placeholders."""
    text = RUNNER.read_text()
    for value in (
        "/2023533015/topology-conditioned-inductive-edge-prediction",
        "/2023533015/.uv/bin/uv",
        "NVIDIA H20",
        "CUDA_VISIBLE_DEVICES=0",
    ):
        assert value in text
    assert "expected exactly 1 visible GPU" in text
    assert "cluster.env" not in text
    assert "sbatch" not in text
    assert ".sbatch" not in text


def test_runner_dispatches_to_the_implemented_clis() -> None:
    """Each command maps directly to one repository Python module."""
    text = RUNNER.read_text()
    assert "-m src.train_b0" in text
    assert "-m src.score_universe score --device cuda --amp bf16" in text
    assert "-m src.score_universe metrics" in text
    assert "-m src.score_universe merge" in text
    assert "-m src.experiments.g1_hardened_e2" in text
    assert "-m src.experiments.g2_ceiling" in text


def test_runner_test_b0_v31_follows_the_frozen_test_contract() -> None:
    """The production test command covers test edges, candidate assembly, then G1/G2."""
    text = RUNNER.read_text()
    case_body = text.split("test-b0-v31)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    required_fragments = (
        "outputs/b0_v31/best.pt",
        "--pairs test",
        "scores/b0_v31_test.npz",
        "outputs/b0_v31/test_metrics.json",
        "--pairs candidate",
        "scores/b0_v31_candidate.npz",
        "outputs/g1",
        "outputs/g2",
        "outputs/g2/g2_ceiling.html",
    )
    for fragment in required_fragments:
        assert fragment in case_body

    assert case_body.index("--pairs test") < case_body.index("--pairs candidate")
    assert case_body.index("outputs/b0_v31/test_metrics.json") < case_body.index("outputs/g1")
    assert case_body.index("outputs/g1") < case_body.index("outputs/g2")

    runbook = (HPC_DIR / "README.md").read_text()
    assert "nohup hpc/run.sh test-b0-v31 > outputs/logs/b0_v31_test.log 2>&1 &" in runbook
    assert "echo $! > outputs/logs/b0_v31_test.pid" in runbook


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


def test_runbook_documents_the_v31_input_pipeline_startup_contract() -> None:
    """Operators can distinguish the expected preload pause from a stalled run."""
    text = (HPC_DIR / "README.md").read_text()
    lower_text = text.lower()

    assert "one-time" in lower_text
    assert "~25 gib" in lower_text
    assert "host memory" in lower_text
    for option in (
        "num_workers: 4",
        "persistent_workers: true",
        "prefetch_factor: 4",
        "pin_memory: true",
    ):
        assert option in text
    assert "no step logs" in lower_text
    assert "preload completes" in lower_text
    assert "descriptor-only" in lower_text
    assert "/dev/shm" in text
    assert "main process" in lower_text
    assert "configs/b0_v31_breadth_first.yaml" in text
