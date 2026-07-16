"""Static contract tests for the auto-sized H20 execution layer."""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HPC_DIR = REPO_ROOT / "hpc"
RUNNER = HPC_DIR / "run.sh"
G5_RUNNER = HPC_DIR / "g5_stage1.sh"


@pytest.fixture(scope="module")
def bash_exe() -> str:
    """Return a usable bash executable or skip shell-dependent checks."""
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash not found on PATH")
    return executable


def test_hpc_layer_has_only_runners_and_documentation() -> None:
    """The old scheduler layer is replaced by direct, tracked runners."""
    assert sorted(path.name for path in HPC_DIR.iterdir()) == [
        "README.md",
        "g5_stage1.sh",
        "run.sh",
    ]
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


def test_g5_runner_is_valid_executable_bash(bash_exe: str) -> None:
    assert G5_RUNNER.read_text().splitlines()[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in G5_RUNNER.read_text()
    assert G5_RUNNER.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        [bash_exe, "-n", str(G5_RUNNER)],
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
    for command in ("check", "train", "s0-score", "score", "merge", "g1", "g2"):
        assert f"hpc/run.sh {command}" in result.stdout
    # The EgoStitch Stage-1 worker routes through the same train entry.
    assert "--worker-module src.train_egostitch" in result.stdout

    g5_result = subprocess.run(
        [bash_exe, str(G5_RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert g5_result.returncode == 0, g5_result.stderr
    assert "hpc/g5_stage1.sh seed <0|1|2>" in g5_result.stdout
    assert "non-binding single-seed topology diagnostic" in g5_result.stdout


def test_g5_runner_completes_each_seed_before_formal_holm() -> None:
    text = G5_RUNNER.read_text()
    run_seed_body = text[text.index("run_seed()") : text.index("run_formal()")]
    assert run_seed_body.index("hpc/run.sh train") < run_seed_body.index("hpc/run.sh score")
    assert run_seed_body.index("hpc/run.sh score") < run_seed_body.index(
        "-m src.experiments.g5_stage1"
    )
    formal_body = text[text.index("run_formal()") :]
    assert formal_body.index('run_seed "${seed}"') < formal_body.index(
        '--output-dir "${FORMAL_ROOT}/formal_gate"'
    )
    assert "model or hyperparameter change" in text


def test_g5_runner_rejects_stale_low_resolution_candidate_before_reuse() -> None:
    text = G5_RUNNER.read_text()
    candidate_check = text[text.index("candidate_is_current()") : text.index("run_seed()")]
    assert "validate_score_resolution" in candidate_check


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
    egostitch_config = (REPO_ROOT / "configs" / "egostitch_stage1_breadth_first.yaml").read_text()

    assert "all visible NVIDIA H20" in " ".join(hpc.split())
    assert "auto-detected" in " ".join(readme.split())
    assert "auto-detected" in " ".join(claude.split())
    assert "verified 2026-07-10" not in hpc
    assert "The verified host" not in readme
    assert "single-H20 container" not in config
    assert "world_size: auto" in egostitch_config


def test_runner_dispatches_to_the_implemented_clis() -> None:
    """Each command maps directly to one repository Python module."""
    text = RUNNER.read_text()
    assert "-m src.e2_pipeline" in text
    assert "-m src.score_universe score --device cuda --amp bf16" in text
    assert "-m src.score_universe merge" in text
    assert "-m src.experiments.g1_hardened_e2" in text
    assert "-m src.experiments.g2_ceiling" in text


def test_s0_scoring_uses_packed_auto_sharded_fast_path() -> None:
    text = RUNNER.read_text()

    assert "s0-score)" in text
    assert "outputs/s0_cache/manifests/all_seeds.tsv" in text
    assert "outputs/feature_packs/b0_v31_bf16" in text
    assert '--pack-dir "${S0_PACK_DIR}"' in text
    assert "--token-budget 1048576" in text
    assert "parallel_score \\\n" in text


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
