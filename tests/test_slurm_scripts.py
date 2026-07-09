"""Static contract tests for the `slurm/` Slurm submission layer.

These are shell scripts, not Python, so there is nothing to unit-test in the
usual sense. What we *can* verify without a real Slurm cluster:

- every script parses as valid bash (``bash -n``);
- every script is executable;
- no real cluster identity (hostname/account) leaks outside
  ``cluster.env.example``, and no ``.sbatch`` file hardcodes the account flag
  (those values must flow through ``submit.sh`` via ``sbatch`` CLI flags,
  since ``#SBATCH`` directives cannot expand environment variables);
- ``submit.sh`` fails fast with a clear fill-in-the-example message when
  ``slurm/cluster.env`` is missing.
"""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SLURM_DIR = REPO_ROOT / "slurm"

SCRIPT_NAMES = [
    "submit.sh",
    "preflight.sbatch",
    "train_b0.sbatch",
    "score_universe.sbatch",
    "sync_code.sh",
    "fetch_outputs.sh",
]

SBATCH_NAMES = [
    "preflight.sbatch",
    "train_b0.sbatch",
    "score_universe.sbatch",
]

NON_SCRIPT_NAMES = [
    "README.md",
    "cluster.env.example",
]


@pytest.fixture(scope="module")
def bash_exe() -> str:
    """Path to a usable ``bash`` binary; skip dependent tests if absent."""
    exe = shutil.which("bash")
    if exe is None:
        pytest.skip("bash not found on PATH")
    return exe


@pytest.mark.parametrize("name", SCRIPT_NAMES + NON_SCRIPT_NAMES)
def test_owned_file_exists(name: str) -> None:
    """Every file this task owns must exist under slurm/."""
    assert (SLURM_DIR / name).is_file(), f"missing slurm/{name}"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_has_valid_bash_syntax(name: str, bash_exe: str) -> None:
    """`bash -n` must accept every .sh/.sbatch script (syntax check only)."""
    script = SLURM_DIR / name
    result = subprocess.run(
        [bash_exe, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n {name} failed:\n{result.stderr}"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_is_executable(name: str) -> None:
    """Every script must have its executable bit set."""
    mode = (SLURM_DIR / name).stat().st_mode
    assert mode & stat.S_IXUSR, f"slurm/{name} is missing the executable bit"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_has_bash_shebang(name: str) -> None:
    """Every script must start with the portable bash shebang."""
    first_line = (SLURM_DIR / name).read_text().splitlines()[0]
    assert first_line == "#!/usr/bin/env bash"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_has_strict_mode(name: str) -> None:
    """Every script must enable `set -euo pipefail` near the top."""
    text = (SLURM_DIR / name).read_text()
    assert "set -euo pipefail" in text


def test_placeholder_hostname_only_in_example_env() -> None:
    """The example hostname must not leak into any real, shipped script."""
    needle = "example-cluster"
    offenders = []
    for name in SCRIPT_NAMES + ["README.md"]:
        text = (SLURM_DIR / name).read_text()
        if needle in text:
            offenders.append(name)
    assert not offenders, f"placeholder hostname leaked into: {offenders}"
    assert needle in (SLURM_DIR / "cluster.env.example").read_text()


@pytest.mark.parametrize("name", SBATCH_NAMES)
def test_sbatch_files_do_not_hardcode_account(name: str) -> None:
    """Account/partition/gres/time must come from submit.sh's sbatch flags.

    `.sbatch` files may only carry generic directives (job-name, output,
    nodes, ntasks); the cluster-specific `-A`/`--account` flag must never be
    hardcoded here since #SBATCH directives cannot expand env vars.
    """
    text = (SLURM_DIR / name).read_text()
    assert "--account" not in text
    assert "-A " not in text


@pytest.mark.parametrize("name", SBATCH_NAMES)
def test_sbatch_files_have_only_generic_directives(name: str) -> None:
    """`.sbatch` files must not embed partition/gres/time directives either."""
    text = (SLURM_DIR / name).read_text()
    assert "--partition" not in text
    assert "-p " not in text
    assert "--gres" not in text
    assert "--time" not in text


def test_submit_sh_fails_fast_without_cluster_env(tmp_path: Path, bash_exe: str) -> None:
    """submit.sh must die with a clear fill-in-the-example message.

    Verified when slurm/cluster.env is absent, before doing anything else.
    """
    tmp_slurm = tmp_path / "slurm"
    tmp_slurm.mkdir()
    for name in SCRIPT_NAMES:
        shutil.copy2(SLURM_DIR / name, tmp_slurm / name)
        (tmp_slurm / name).chmod((tmp_slurm / name).stat().st_mode | stat.S_IXUSR)
    # Deliberately do not copy cluster.env or cluster.env.example.

    result = subprocess.run(
        [bash_exe, str(tmp_slurm / "submit.sh"), "preflight"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "cluster.env" in combined
    assert "cluster.env.example" in combined
    # Fail-fast: no side effects like slurm/logs/ should be created.
    assert not (tmp_slurm / "logs").exists()
