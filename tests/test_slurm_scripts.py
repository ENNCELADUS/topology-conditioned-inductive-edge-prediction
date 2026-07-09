"""Static contract tests for the `slurm/` Slurm submission layer.

These are shell scripts, not Python, so most of what we verify is static
(syntax, permissions, no leaked cluster identity). What we *can* also verify
without a real Slurm cluster:

- every script parses as valid bash (``bash -n``);
- every script is executable;
- no real cluster identity (hostname/account) leaks outside
  ``cluster.env.example``, and no ``.sbatch`` file hardcodes the account flag
  (those values must flow through ``submit.sh`` via ``sbatch`` CLI flags,
  since ``#SBATCH`` directives cannot expand environment variables);
- ``submit.sh`` fails fast with a clear fill-in-the-example message when
  ``slurm/cluster.env`` is missing;
- ``submit.sh``'s *dispatch logic* -- which flags it forwards to `sbatch`,
  which `.sbatch` script it picks per target, and which extra CLI args it
  forwards verbatim after `--` -- is exercised end to end against a stub
  `sbatch` binary on `PATH`, using a throwaway sandbox copy of `slurm/` and a
  synthetic `cluster.env` so the developer's real one is never required or
  touched.
"""

import os
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


# ---------------------------------------------------------------------------
# submit.sh dispatch-logic regression tests
#
# These run the real submit.sh (copied into a throwaway sandbox, with a
# synthetic cluster.env) against a stub `sbatch` on PATH that records its
# argv instead of touching a real Slurm cluster. This lets us assert exactly
# what submit.sh forwards: the sbatch flags it derives from cluster.env, the
# `.sbatch` script it picks per target, and the extra CLI args it forwards
# verbatim after `--`.
# ---------------------------------------------------------------------------

_TEST_CLUSTER_ENV = """\
CLUSTER_HOST=test-host
CLUSTER_USER=test-user
CLUSTER_ACCOUNT=unit-test-account
CLUSTER_PARTITION=unit-test-partition
CLUSTER_GRES=gpu:7
CLUSTER_TIME_TRAIN=01:11:11
CLUSTER_TIME_SCORE=02:22:22
CLUSTER_REPO_DIR=/tmp/unit-test-repo-dir
CLUSTER_DATA_ROOT=/tmp/unit-test-data-root
CLUSTER_SETUP=""
"""

_SBATCH_STUB = r"""#!/usr/bin/env bash
# Stub `sbatch`: records every argument it was called with, one per line, to
# the file named by $SBATCH_CAPTURE_FILE, then exits successfully without
# ever touching a real Slurm cluster.
set -euo pipefail
printf '%s\n' "$@" > "${SBATCH_CAPTURE_FILE}"
exit 0
"""


def _make_submit_sandbox(tmp_path: Path) -> Path:
    """Copy `slurm/` into an isolated tmp dir with a synthetic `cluster.env`.

    Returns the path to the copied ``slurm`` directory. The real
    ``slurm/cluster.env`` (if a developer has filled one in locally) is never
    read or written by these tests: the sandbox gets its own throwaway one.
    """
    sandbox = tmp_path / "sandbox-slurm"
    sandbox.mkdir()
    for name in SCRIPT_NAMES:
        shutil.copy2(SLURM_DIR / name, sandbox / name)
        (sandbox / name).chmod((sandbox / name).stat().st_mode | stat.S_IXUSR)
    (sandbox / "cluster.env").write_text(_TEST_CLUSTER_ENV)
    return sandbox


def _make_sbatch_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Create a stub `sbatch` executable and its argument-capture file.

    Returns ``(bin_dir, capture_file)``: prepend ``bin_dir`` to ``PATH`` so
    the stub shadows any real ``sbatch``, and set ``SBATCH_CAPTURE_FILE`` in
    the environment to ``capture_file`` so the stub knows where to record
    its argv.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_file = tmp_path / "sbatch_args.txt"
    stub = bin_dir / "sbatch"
    stub.write_text(_SBATCH_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir, capture_file


def _run_submit(
    sandbox: Path,
    args: list[str],
    bin_dir: Path,
    capture_file: Path,
    tmp_path: Path,
    bash_exe: str,
) -> subprocess.CompletedProcess[str]:
    """Run the sandboxed `submit.sh` with the stub `sbatch` shadowing PATH."""
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["SBATCH_CAPTURE_FILE"] = str(capture_file)
    return subprocess.run(
        [bash_exe, str(sandbox / "submit.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
        env=env,
    )


def _captured_sbatch_args(capture_file: Path) -> list[str]:
    """Read back the argv the stub `sbatch` was invoked with, one per line."""
    return capture_file.read_text().splitlines()


def test_submit_sh_score_forwards_args_after_double_dash(tmp_path: Path, bash_exe: str) -> None:
    """`submit.sh score -- score --foo bar` forwards `score --foo bar` verbatim.

    The `--` separator is stripped by submit.sh; every argument after it is
    payload that must reach `sbatch` (and, on the cluster, the score_universe
    CLI) completely unmodified.
    """
    sandbox = _make_submit_sandbox(tmp_path)
    bin_dir, capture_file = _make_sbatch_stub(tmp_path)

    result = _run_submit(
        sandbox,
        ["score", "--", "score", "--foo", "bar"],
        bin_dir,
        capture_file,
        tmp_path,
        bash_exe,
    )

    assert result.returncode == 0, result.stderr
    argv = _captured_sbatch_args(capture_file)
    assert argv[-3:] == ["score", "--foo", "bar"]
    assert argv[-4].endswith("score_universe.sbatch")
    assert "--array" not in argv


def test_submit_sh_score_array_passes_array_flag_and_forwards_args(
    tmp_path: Path, bash_exe: str
) -> None:
    """`--array 0-3` reaches `sbatch` as an `--array` flag; args still forward.

    Both effects of `slurm/submit.sh score --array 0-3 -- <args>` must hold
    at once: the array range becomes an `--array` flag to `sbatch` itself
    (so Slurm fans the job out), and the post-`--` payload is still
    forwarded verbatim to the `.sbatch` script.
    """
    sandbox = _make_submit_sandbox(tmp_path)
    bin_dir, capture_file = _make_sbatch_stub(tmp_path)

    result = _run_submit(
        sandbox,
        [
            "score",
            "--array",
            "0-3",
            "--",
            "score",
            "--checkpoint",
            "outputs/b0/best.pt",
            "--output",
            "scores/b0.npz",
        ],
        bin_dir,
        capture_file,
        tmp_path,
        bash_exe,
    )

    assert result.returncode == 0, result.stderr
    argv = _captured_sbatch_args(capture_file)
    assert "--array" in argv
    assert argv[argv.index("--array") + 1] == "0-3"
    assert argv[-5:] == [
        "score",
        "--checkpoint",
        "outputs/b0/best.pt",
        "--output",
        "scores/b0.npz",
    ]
    script_index = next(i for i, a in enumerate(argv) if a.endswith("score_universe.sbatch"))
    assert argv.index("--array") < script_index < argv.index("score")


def test_submit_sh_train_forwards_config_path(tmp_path: Path, bash_exe: str) -> None:
    """`submit.sh train configs/x.yaml` forwards the config path verbatim."""
    sandbox = _make_submit_sandbox(tmp_path)
    bin_dir, capture_file = _make_sbatch_stub(tmp_path)

    result = _run_submit(
        sandbox,
        ["train", "configs/x.yaml"],
        bin_dir,
        capture_file,
        tmp_path,
        bash_exe,
    )

    assert result.returncode == 0, result.stderr
    argv = _captured_sbatch_args(capture_file)
    assert argv[-1] == "configs/x.yaml"
    assert argv[-2].endswith("train_b0.sbatch")
    assert "--array" not in argv


def test_submit_sh_passes_cluster_env_flags_to_sbatch(tmp_path: Path, bash_exe: str) -> None:
    """Account/partition/gres/time from `cluster.env` land on the sbatch CLI.

    `.sbatch` files carry only generic directives (see
    ``test_sbatch_files_do_not_hardcode_account`` /
    ``test_sbatch_files_have_only_generic_directives`` above); the
    cluster-specific values must flow through as `sbatch` CLI flags supplied
    by `submit.sh`, sourced from the (here, synthetic) `cluster.env`.
    """
    sandbox = _make_submit_sandbox(tmp_path)
    bin_dir, capture_file = _make_sbatch_stub(tmp_path)

    result = _run_submit(
        sandbox,
        ["train", "configs/x.yaml"],
        bin_dir,
        capture_file,
        tmp_path,
        bash_exe,
    )

    assert result.returncode == 0, result.stderr
    argv = _captured_sbatch_args(capture_file)
    assert argv[argv.index("-A") + 1] == "unit-test-account"
    assert argv[argv.index("-p") + 1] == "unit-test-partition"
    assert argv[argv.index("--gres") + 1] == "gpu:7"
    assert argv[argv.index("--time") + 1] == "01:11:11"


def test_submit_sh_score_uses_score_time_not_train_time(tmp_path: Path, bash_exe: str) -> None:
    """The `score` target's `--time` comes from CLUSTER_TIME_SCORE, not TRAIN."""
    sandbox = _make_submit_sandbox(tmp_path)
    bin_dir, capture_file = _make_sbatch_stub(tmp_path)

    result = _run_submit(
        sandbox,
        ["score", "--", "score", "--output", "scores/b0.npz"],
        bin_dir,
        capture_file,
        tmp_path,
        bash_exe,
    )

    assert result.returncode == 0, result.stderr
    argv = _captured_sbatch_args(capture_file)
    assert argv[argv.index("--time") + 1] == "02:22:22"
