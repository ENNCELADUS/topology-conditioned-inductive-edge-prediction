"""Static contract tests for the auto-sized H20 execution layer."""

import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HPC_DIR = REPO_ROOT / "hpc"
RUNNER = HPC_DIR / "run.sh"
SWEEP_RUNNER = HPC_DIR / "sweep_kd_hpo.sh"


@pytest.fixture(scope="module")
def bash_exe() -> str:
    """Return a usable bash executable or skip shell-dependent checks."""
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash not found on PATH")
    return executable


def test_hpc_layer_has_only_runners_and_documentation() -> None:
    """The old scheduler layer is replaced by one direct, tracked runner.

    `g5_stage1.sh` orchestrated the frozen-s0 `egostitch` family and was
    deleted with it (design 2026-07-29 Sec 6.2). `qualification.sh` and the
    registration/preregistration machinery it enforced were excised (design
    2026-08-02 Sec 10): `run.sh train` now launches EgoStitch E2E directly.
    """
    assert sorted(path.name for path in HPC_DIR.iterdir()) == [
        "README.md",
        "run.sh",
        "sweep_kd_hpo.sh",
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


def test_sweep_runner_is_valid_executable_bash(bash_exe: str) -> None:
    result = subprocess.run(
        [bash_exe, "-n", str(SWEEP_RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sweep_nohup_guidance_records_and_controls_the_session_pgid(
    tmp_path: Path, bash_exe: str
) -> None:
    text = SWEEP_RUNNER.read_text()
    assert "setsid bash -c 'echo $$" in text
    assert "exec hpc/sweep_kd_hpo.sh 0" in text
    assert "echo $!" not in text
    assert 'kill -- -"$(cat outputs/b1_row_kd_hpo/lane0.pgid)"' in text

    fake_setsid = tmp_path / "setsid"
    _write_executable(
        fake_setsid,
        """#!/usr/bin/env python3
import os
import sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
""",
    )
    fake_lane = tmp_path / "lane.sh"
    _write_executable(
        fake_lane,
        """#!/usr/bin/env bash
set -euo pipefail
sleep 30 &
echo "$!" > "${CHILD_PID}"
wait
""",
    )
    pgid_path = tmp_path / "lane.pgid"
    child_path = tmp_path / "child.pid"
    env = os.environ.copy()
    env["CHILD_PID"] = str(child_path)
    process = subprocess.Popen(
        [
            str(fake_setsid),
            bash_exe,
            "-c",
            f'echo $$ > "{pgid_path}"; exec "{fake_lane}"',
        ],
        env=env,
    )
    pgid: int | None = None
    try:
        for _ in range(200):
            if pgid_path.is_file() and child_path.is_file():
                break
            time.sleep(0.01)
        assert pgid_path.is_file() and child_path.is_file()
        pgid = int(pgid_path.read_text())
        child_pid = int(child_path.read_text())
        assert pgid == process.pid
        assert os.getpgid(child_pid) == pgid
        os.killpg(pgid, signal.SIGTERM)
        process.wait(timeout=2)
        assert process.returncode != 0
    finally:
        if process.poll() is None and pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _mock_sweep_checkout(tmp_path: Path) -> tuple[dict[str, str], Path]:
    configs = tmp_path / "configs" / "sweep" / "b1_kd_hpo"
    configs.mkdir(parents=True)
    for name in ("a first.yaml", "b second.yaml", "c third.yaml", "d fourth.yaml"):
        (configs / name).write_text("{}\n")

    call_log = tmp_path / "calls.log"
    _write_executable(
        tmp_path / "hpc" / "run.sh",
        """#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == train && "$3" == --skip-test ]] || exit 31
printf 'run\\t%s\\t%s\\t%s\\n' "${CUDA_VISIBLE_DEVICES}" "$2" "$3" >> "${CALL_LOG}"
if [[ "${EXPECT_OVERLAP:-}" == 1 ]]; then
  touch "${RUN_BARRIER}.${CUDA_VISIBLE_DEVICES}"
  for _ in {1..100}; do
    [[ -f "${RUN_BARRIER}.0,1" && -f "${RUN_BARRIER}.2,3" ]] && break
    sleep 0.01
  done
  [[ -f "${RUN_BARRIER}.0,1" && -f "${RUN_BARRIER}.2,3" ]] || exit 32
fi
stem="$(basename "$2" .yaml)"
if [[ "${FAIL_CONFIG:-}" == "$2" && "${FAIL_MODE:-}" == exit ]]; then
  exit 7
fi
if [[ "${FAIL_CONFIG:-}" == "$2" && "${FAIL_MODE:-}" == marker ]]; then
  mkdir -p "outputs/b1_row_kd_hpo/${stem}"
  touch "outputs/b1_row_kd_hpo/${stem}/failure.json"
fi
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "CALL_LOG": str(call_log),
            "RUN_BARRIER": str(tmp_path / "run-active"),
        }
    )
    return env, call_log


def test_sweep_mock_contract_covers_lanes_resume_quoting_and_overlap(
    tmp_path: Path, bash_exe: str
) -> None:
    env, call_log = _mock_sweep_checkout(tmp_path)
    env["EXPECT_OVERLAP"] = "1"
    completed = tmp_path / "outputs" / "b1_row_kd_hpo" / "c third"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text("{}\n")

    processes = [
        subprocess.Popen(
            [bash_exe, str(SWEEP_RUNNER), lane],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for lane in ("0", "1")
    ]
    results = [process.communicate() for process in processes]
    assert [process.returncode for process in processes] == [0, 0], results

    calls = call_log.read_text().splitlines()
    run_calls = {line for line in calls if line.startswith("run\t")}
    assert run_calls == {
        "run\t0,1\tconfigs/sweep/b1_kd_hpo/a first.yaml\t--skip-test",
        "run\t2,3\tconfigs/sweep/b1_kd_hpo/b second.yaml\t--skip-test",
        "run\t2,3\tconfigs/sweep/b1_kd_hpo/d fourth.yaml\t--skip-test",
    }


def test_sweep_failure_marker_wins_over_stale_completion(tmp_path: Path, bash_exe: str) -> None:
    env, call_log = _mock_sweep_checkout(tmp_path)
    stale_output = tmp_path / "outputs" / "b1_row_kd_hpo" / "a first"
    stale_output.mkdir(parents=True)
    (stale_output / "complete.json").write_text("{}\n")
    (stale_output / "failure.json").write_text("{}\n")

    result = subprocess.run(
        [bash_exe, str(SWEEP_RUNNER), "0"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert not call_log.exists()
    assert "aborting at a first (failure.json exists)" in result.stderr


@pytest.mark.parametrize("failure_mode", ["exit", "marker"])
def test_sweep_mock_contract_aborts_lane_on_failure(
    tmp_path: Path, bash_exe: str, failure_mode: str
) -> None:
    env, call_log = _mock_sweep_checkout(tmp_path)
    env.update(
        {
            "FAIL_CONFIG": "configs/sweep/b1_kd_hpo/a first.yaml",
            "FAIL_MODE": failure_mode,
        }
    )
    result = subprocess.run(
        [bash_exe, str(SWEEP_RUNNER), "0"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    calls = call_log.read_text()
    assert "a first.yaml\t--skip-test" in calls
    assert "c third.yaml\t--skip-test" not in calls
    assert "aborting at a first" in result.stderr


def test_help_is_available_without_the_remote_container(bash_exe: str) -> None:
    """Usage can be inspected locally before the fixed remote paths are checked."""
    result = subprocess.run(
        [bash_exe, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for command in ("check", "train", "score", "test", "merge", "g1", "g2"):
        assert f"hpc/run.sh {command}" in result.stdout
    # `s0-score` served the retired frozen-s0 family and must not come back.
    assert "s0-score" not in result.stdout
    # `qualification.sh` is gone; EgoStitch E2E now launches through the
    # generic `train` branch via an explicit worker-module override.
    assert "qualification.sh" not in result.stdout
    assert "src.train_egostitch" in result.stdout
    assert "src.train_cazi_mbn" in result.stdout
    # score/test are thin passthroughs; the shell no longer owns sharding or
    # the held-out protocol itself.
    assert "src.score_fanout" in result.stdout
    assert "src.eval.test_protocol" in result.stdout
    assert "--rescore-reason" in result.stdout


def test_cazi_uses_isolated_h20_train_worker() -> None:
    """The external CAZI schedule bypasses only the incompatible E2 pipeline."""
    text = RUNNER.read_text()
    assert '"$2" == "src.train_cazi_mbn"' in text
    assert '-m src.train_cazi_mbn "${CONFIG_PATH}" --device cuda' in text
    # Not exec'd: training must complete before the chained test protocol runs
    # against the checkpoint CAZI itself publishes.
    assert 'exec "${PYTHON_BIN}" -m src.train_cazi_mbn' not in text


def test_cazi_chains_the_held_out_test_protocol() -> None:
    """CAZI has no pipeline `test` stage of its own, so the branch chains one."""
    text = RUNNER.read_text()
    assert "from src.train_cazi_mbn import load_config" in text
    assert "-m src.eval.test_protocol" in text
    assert '--checkpoint "${CAZI_OUTPUT_DIR}/student.pt"' in text
    assert "--arm cazi_mbn" in text
    # CAZI's `student.pt` is a bare state_dict, so the family and its own YAML
    # must be named or `score_universe` cannot build the model at all.
    assert "--model-family cazi_mbn" in text
    assert '--model-config "${CONFIG_PATH}"' in text


def test_cazi_trains_without_opening_held_out_data() -> None:
    """`--stage all` would duplicate test/candidate before the protocol owns them.

    CAZI's default stage runs `score_and_evaluate`, which reads the balanced test
    pairs and the candidate universe. Chaining the test protocol after that would
    both duplicate the held-out result and invert the documented ordering, so the
    training call must be train-only and the protocol must own every held-out read.
    """
    text = RUNNER.read_text()
    assert '-m src.train_cazi_mbn "${CONFIG_PATH}" --device cuda --stage train' in text


def test_runner_discovers_visible_h20s() -> None:
    text = RUNNER.read_text()
    for value in (
        "/2023533015/topology-conditioned-inductive-edge-prediction",
        "/2023533015/.uv/bin/uv",
        "NVIDIA H20",
        "NVIDIA H20-3e",
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
    egostitch_config = (
        REPO_ROOT / "configs" / "egostitch_e2e_v3_full_breadth_first.yaml"
    ).read_text()

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
    assert "-m src.score_fanout" in text
    assert "-m src.eval.test_protocol" in text
    assert "-m src.score_universe merge" in text
    assert "-m src.experiments.g1_hardened_e2" in text
    assert "-m src.experiments.g2_ceiling" in text
    # `check` is environment/data validation only; the test suites run locally,
    # never as an hpc/run.sh recheck.
    assert "-m pytest" not in text


def test_scoring_is_a_thin_passthrough_to_score_fanout() -> None:
    """`score` no longer orchestrates sharding itself: `src.score_fanout` owns it."""
    text = RUNNER.read_text()

    assert "s0-score" not in text
    assert "outputs/s0_cache" not in text
    # The shard/merge guard and orchestration loop moved into the Python
    # module; the shell must not duplicate `--shard`/`--num-shards` handling.
    assert "parallel_score" not in text
    assert "hpc/run.sh score owns sharding" not in text
    assert '--shard "${gpu}"' not in text
    assert '--num-shards "${GPU_COUNT}"' not in text
    assert 'exec "${PYTHON_BIN}" -m src.score_fanout "$@"' in text


@pytest.mark.parametrize(
    "config_name",
    ["b0_v31_breadth_first.yaml"],
)
def test_hpc_training_configs_pin_bf16(config_name: str) -> None:
    """The implemented baseline run uses H20-native BF16 in the fixed environment.

    ``b0_alt_breadth_first.yaml`` (B0-alt) was removed 2026-08-03 by owner decision;
    see ``docs/results/E2-pair-to-topology-gap.md`` for the closed result it produced.
    """
    text = (REPO_ROOT / "configs" / config_name).read_text()
    assert 'mixed_precision: "bf16"' in text


def test_primary_docs_reference_only_the_direct_hpc_layer() -> None:
    """Contributor-facing docs do not route experiments through the removed layer."""
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md", HPC_DIR / "README.md"):
        text = path.read_text()
        assert "slurm/" not in text.lower()
        assert "sbatch" not in text.lower()


# test_specs_pin_auto_sized_h20_e2_training was removed with docs/05-egostitch-spec.md
# (deleted in 60dfc25) and the 03-experiment-protocol rewrite that dropped its pinned
# sentences; the runtime behavior it guarded is asserted by the config tests above.
