"""Contracts for the fail-closed two-stage EgoStitch E2E launcher.

The ladder is `qualify -> formal` (design 2026-07-29 Sec 2): both stages train
on the full V_fit and validate on the single V_hold, and differ only in
``optim.epochs``. The pre-binding ladder these tests used to pin — calibrate,
the threshold freeze manifest, the single-rehearsal ledger, the attempt window
and the sanitized-root sandbox — was deleted with it.
"""

import hashlib
import io
import json
import re
import shutil
import stat
import subprocess
import sys
from contextlib import redirect_stdout
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

# The retired pre-binding ladder, named so a revert cannot land silently.
_RETIRED_SURFACE = (
    "run_calibration",
    "run_rehearsal",
    "write_calibration_freeze_manifest",
    "assert_prebinding_gates_frozen",
    "assert_prebinding_gates_implementable",
    "assert_single_v_qual_rehearsal",
    "assert_implementation_frozen",
    "assert_qualification_boundary",
    "evaluate_stage_gates",
    "REHEARSAL_LEDGER",
    "EGOSTITCH_QUALIFICATION_ROOT",
    "prebinding_gates",
    "calibration_freeze",
    "freeze_eligible_gate_ids",
    "registration_canonical_sha256",
    "attempt_number",
    "--run-kind overfit",
    "--run-kind rehearsal",
    "egostitch_e2e_v_qual",
    "egostitch_e2e_v_select",
    "qualification_margins",
)


@pytest.fixture(scope="module")
def bash_exe() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash not found on PATH")
    return executable


def _strip_comments(block: str) -> str:
    """Drop comment lines so prose cannot satisfy a behavioural assertion."""
    return "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))


def _block(name: str, until: str) -> str:
    text = RUNNER.read_text()
    return text[text.index(name) : text.index(until)]


def _embedded_python(block: str) -> str:
    """Return the first ``python -c '<program>'`` body in ``block``.

    The launcher's checks are embedded Python, so running the real program is
    the only way to test them without the fixed remote container.
    """
    start = block.index("-c '") + len("-c '")
    return block[start : block.index("\n'", start)]


# The embedded programs never contain a single quote, so the closing quote is
# unambiguous; `-c` is written both inline and continued onto the next line.
_EMBEDDED_PYTHON_RE = re.compile(r"-c\s*\\?\s*'([^']*)'", re.S)


def _embedded_python_containing(block: str, needle: str) -> str:
    """Return the embedded ``python -c`` program in ``block`` that mentions ``needle``."""
    for program in _EMBEDDED_PYTHON_RE.findall(block):
        if needle in program:
            return program
    raise AssertionError(f"no embedded python program in the block mentions {needle!r}")


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text())
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
        # The vocabulary-attribution ablation pins the status-quo pool at 20.
        "cosine_pool": ("none", 0.15, 0.15, "cosine_pool", 20),
        "no_l_rel": ("none", 0.15, 0.15, "no_l_rel", 50),
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

    # The no_l_rel arm is defined as full with the relational weight zeroed.
    assert _model_config(configs["no_l_rel"])["w_rel"] == 0
    assert "w_rel" not in _model_config(configs["full"])


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
            "docs/registrations/g5_e2e_stage1_preregistration_v4.json"
        )
        optim = _section(config, "optim")
        assert optim["lr"] == 1.0e-4
        assert optim["warmup_steps"] == 500
        runtime = _section(config, "runtime")
        assert runtime["world_size"] == "auto"
        # Halved from v2's 128 on 2026-07-27: rev-3.1's working set reached
        # 92.71 GiB allocated at 128 and OOM'd a 95 GiB H20 even with
        # fragmentation eliminated. For EgoStitch this key is the per-rank
        # node-stream batch B_n, not a token count. It is a scalar now: the
        # orchestrator's candidate sweep was deleted with the probe stage.
        assert runtime["token_budget"] == 128
        assert "token_budget_candidates" not in runtime
        assert runtime["train_eval_budget_seconds"] == 26400
        if common is None:
            common = training
        else:
            assert training == common


def test_e2e_configs_carry_no_hand_pasted_feature_digest() -> None:
    """Both stages train on the identical universe, so the digest is compared, not pinned.

    Pinning it in the config is what made the old ladder require a hand-pasted
    value from a calibration run (design 2026-07-29 Sec 3).
    """
    for path in CONFIGS.values():
        assert "feature_stats_sha256" not in _model_config(_load_config(path))


def test_qualification_is_refused_after_registration_becomes_binding() -> None:
    guard = _strip_comments(
        _block("assert_qualification_registration_open()", "assert_registration_unchanged()")
    )
    assert 'registration_status)" != "BINDING"' in guard
    assert "post-binding attempts would escape the registered K disclosure" in guard
    launcher = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert "assert_qualification_registration_open" in launcher


def test_arms_that_disagree_on_n_ground_do_not_share_a_pack() -> None:
    """The F0/grounding pack manifest is keyed on `n_ground` and rejects a mismatch.

    A single forced pack built at 50 is what made the cosine-pool arm (20)
    raise; each config now names its own pack directory.
    """
    pack_dirs: dict[str, set[str]] = {}
    for name, path in CONFIGS.items():
        config = _load_config(path)
        runtime = _section(config, "runtime")
        model_config = _model_config(config)
        assert isinstance(runtime["pack_dir"], str)
        pack_dirs.setdefault(runtime["pack_dir"], set()).add(f"{name}:{model_config['n_ground']}")
    for pack_dir, arms in pack_dirs.items():
        n_grounds = {arm.split(":")[1] for arm in arms}
        assert len(n_grounds) == 1, f"{pack_dir} is shared by arms with n_ground {n_grounds}"


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


def test_e2e_runner_help_describes_the_two_stage_ladder(bash_exe: str) -> None:
    result = subprocess.run(
        [bash_exe, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "hpc/qualification.sh qualify" in result.stdout
    assert "hpc/qualification.sh calibrate-tolerance" in result.stdout
    assert "hpc/qualification.sh formal" in result.stdout
    assert "auto-detects and uses every visible H20" in result.stdout
    assert "exactly 4 visible NVIDIA H20s" in result.stdout
    assert "differ only in" in result.stdout and "optim.epochs" in result.stdout
    # The six trained arms are selectable; the two scoring-time controls are not.
    assert "full|f_only|pair_topology|p0|cosine_pool|no_l_rel" in result.stdout
    assert "--max-steps is never substituted" in result.stdout


def test_tolerance_bootstrap_records_the_new_full_attempt_before_calibration() -> None:
    body = _strip_comments(_block("run_tolerance_calibration()", "run_formal()"))
    assert body.index("assert_tracked_clean_checkout calibrate-tolerance") < body.index(
        "run_qualification full"
    )
    assert body.index("assert_tolerance_calibration_not_complete") < body.index(
        "find_immutable_tolerance_source_attempt"
    )
    assert body.index('current_attempt="$(current_qualification_attempt full)"') < body.index(
        "find_immutable_tolerance_source_attempt"
    )
    assert body.index("find_immutable_tolerance_source_attempt") < body.index(
        "run_qualification full"
    )
    assert body.index("run_qualification full") < body.index(
        "src.experiments.auprc_tolerance"
    )
    assert 'if [[ -z "${attempt_dir}" ]]' in body
    assert body.count("run_qualification full") == 1
    assert '--attempt-dir "${attempt_dir}"' in body
    assert '--preregistration "${PREREGISTRATION}"' in body
    assert "assert_registration_unchanged" in body

    qualification = _strip_comments(
        _block("run_qualification()", "assert_tolerance_calibration_not_complete()")
    )
    assert qualification.index("record_qualification_attempt") < qualification.index(
        'update_qualification_pointer "${arm}" "${output_dir}" latest'
    )
    assert "latest-pass" not in qualification


def test_failed_full_qualification_cannot_fabricate_calibration_evidence() -> None:
    body = _strip_comments(_block("run_tolerance_calibration()", "run_formal()"))
    launch = body.index("run_qualification full")
    calibrate = body.index("src.experiments.auprc_tolerance")
    assert launch < calibrate
    # `set -e` therefore stops on any failed qualification before the module is
    # invoked; no masking operator may convert that failure into calibration.
    assert re.search(r"^\s*run_qualification full\s*$", body, re.MULTILINE)


def test_tolerance_bootstrap_is_one_shot_and_ordinary_qualify_does_not_run_it() -> None:
    guard = _strip_comments(
        _block(
            "assert_tolerance_calibration_not_complete()",
            "find_immutable_tolerance_source_attempt()",
        )
    )
    assert "auprc_tolerance_calibration.json" in guard
    assert "ap_replicates.npy" not in guard
    assert "-print -quit" in guard
    assert "one-shot" in guard

    qualification = _strip_comments(
        _block("run_qualification()", "assert_tolerance_calibration_not_complete()")
    )
    assert "src.experiments.auprc_tolerance" not in qualification
    assert "ap_replicates.npy" not in qualification


def test_tolerance_source_selector_uses_earliest_successful_recorded_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import auprc_tolerance as calibration

    arm_root = tmp_path / "qualification" / "full"
    attempts_root = arm_root / "attempts"
    dirty = attempts_root / "attempt-001"
    earliest = attempts_root / "attempt-002"
    later = attempts_root / "attempt-003"
    for attempt, clean in ((dirty, False), (earliest, True), (later, True)):
        attempt.mkdir(parents=True)
        (attempt / "auprc_tolerance_source.npz").write_bytes(b"source")
        (attempt / "run_metadata.json").write_text(json.dumps({"clean": clean}))
    (arm_root / "attempt_history.json").write_text(
        json.dumps(
            {
                "schema_version": "egostitch_e2e_qualification_history_v1",
                "arm": "full",
                "attempts": [
                    {
                        "attempt_dir": str(dirty),
                        "exit_code": 0,
                        "outcome": "success",
                        "verdict": "pass",
                    },
                    {
                        "attempt_dir": str(earliest),
                        "exit_code": 0,
                        "outcome": "success",
                        "verdict": "pass",
                    },
                    {
                        "attempt_dir": str(later),
                        "exit_code": 0,
                        "outcome": "success",
                        "verdict": "pass",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "load_source", lambda path: (None, None, {}))

    def validate(attempt: Path, preregistration_path: Path, metadata: object) -> None:
        del preregistration_path, metadata
        run_metadata = json.loads((attempt / "run_metadata.json").read_text())
        if run_metadata["clean"] is not True:
            raise ValueError("dirty source attempt")

    monkeypatch.setattr(calibration, "_validate_attempt", validate)
    program = _embedded_python(
        _block("find_immutable_tolerance_source_attempt()", "run_tolerance_calibration()")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["python", str(arm_root / "attempt_history.json"), str(preregistration)],
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(program, "<selector>", "exec"), {})

    assert Path(stdout.getvalue().strip()).resolve() == earliest.resolve()


def test_tolerance_source_selector_returns_empty_when_no_clean_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments import auprc_tolerance as calibration

    arm_root = tmp_path / "qualification" / "full"
    attempt = arm_root / "attempts" / "attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "auprc_tolerance_source.npz").write_bytes(b"source")
    history = arm_root / "attempt_history.json"
    history.write_text(
        json.dumps(
            {
                "schema_version": "egostitch_e2e_qualification_history_v1",
                "arm": "full",
                "attempts": [
                    {
                        "attempt_dir": str(attempt),
                        "exit_code": 0,
                        "outcome": "success",
                        "verdict": "pass",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "load_source", lambda path: (None, None, {}))
    monkeypatch.setattr(
        calibration,
        "_validate_attempt",
        lambda attempt_path, preregistration_path, metadata: (_ for _ in ()).throw(
            ValueError("dirty source attempt")
        ),
    )
    program = _embedded_python(
        _block("find_immutable_tolerance_source_attempt()", "run_tolerance_calibration()")
    )
    monkeypatch.setattr(sys, "argv", ["python", str(history), str(preregistration)])
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(program, "<selector>", "exec"), {})

    assert stdout.getvalue() == ""


def test_interrupted_bootstrap_without_outputs_resumes_the_same_source() -> None:
    body = _strip_comments(_block("run_tolerance_calibration()", "run_formal()"))
    assert 'attempt_dir="$(find_immutable_tolerance_source_attempt)"' in body
    assert 'if [[ -z "${attempt_dir}" ]]' in body
    # Existing source selection occurs before the only qualification launch.
    assert body.index('attempt_dir="$(find_immutable_tolerance_source_attempt)"') < body.index(
        "run_qualification full"
    )
    # The selector is history-ordered and never consults mutable latest-pass.
    assert "latest-pass" not in body


def test_pending_latest_qualification_blocks_calibration_before_selection_or_launch() -> None:
    body = _strip_comments(_block("run_tolerance_calibration()", "run_formal()"))
    current = body.index('current_attempt="$(current_qualification_attempt full)"')
    recorded_pass = body.index("assert_current_qualification_recorded_pass full")
    assert current < recorded_pass < body.index("find_immutable_tolerance_source_attempt")
    assert recorded_pass < body.index("run_qualification full")


def test_calibrate_tolerance_is_a_no_arm_command() -> None:
    dispatch = _strip_comments(RUNNER.read_text()[RUNNER.read_text().rindex('case "${1:-}" in') :])
    assert "calibrate-tolerance)" in dispatch
    assert '[[ $# -eq 1 ]]' in dispatch
    assert "run_tolerance_calibration" in dispatch


@pytest.mark.parametrize("retired", _RETIRED_SURFACE)
def test_the_prebinding_ladder_is_gone(retired: str) -> None:
    assert retired not in RUNNER.read_text()


def test_qualification_runs_the_short_schedule_on_the_shared_universe() -> None:
    body = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert "--run-kind qualification" in body
    assert '--epochs "${QUALIFICATION_EPOCHS}"' in body
    assert "readonly QUALIFICATION_EPOCHS=3" in RUNNER.read_text()
    # Hardware selection precedes the launch, and the sanity suite precedes both.
    assert body.index("run_sanity_suite") < body.index("select_all_visible_h20s")
    assert body.index("select_all_visible_h20s") < body.index("--run-kind qualification")
    # Development loop: no clean-checkout requirement, and no held-out sandbox.
    assert "assert_clean_checkout" not in body
    assert "--max-steps" not in body


def test_qualification_attempt_directories_are_unique_and_failed_attempt_is_retained(
    tmp_path: Path, bash_exe: str
) -> None:
    functions = _block("allocate_qualification_attempt_dir()", "select_all_visible_h20s()")
    program = (
        'QUALIFICATION_ROOT_DIR="$1"\n'
        'PYTHON_BIN="$2"\n'
        'fail() { echo "$*" >&2; return 1; }\n'
        + functions
        + '\nfirst="$(allocate_qualification_attempt_dir full)"\n'
        + 'printf success > "${first}/qualification.json"\n'
        + 'printf metadata > "${first}/run_metadata.json"\n'
        + 'printf events > "${first}/v_hold_validation_events.jsonl"\n'
        + 'record_qualification_attempt full "${first}" 0\n'
        + 'update_qualification_pointer full "${first}" latest\n'
        + 'update_qualification_pointer full "${first}" latest-pass\n'
        + 'second="$(allocate_qualification_attempt_dir full)"\n'
        + 'printf failure > "${second}/qualification.json"\n'
        + 'record_qualification_attempt full "${second}" 7\n'
        + 'update_qualification_pointer full "${second}" latest\n'
        + 'printf "%s\\n%s\\n" "${first}" "${second}"\n'
    )
    result = subprocess.run(
        [
            bash_exe,
            "-c",
            program,
            "bash",
            str(tmp_path / "qualification"),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    first_raw, second_raw = result.stdout.splitlines()
    first, second = Path(first_raw), Path(second_raw)
    assert first != second
    assert (first / "qualification.json").read_text() == "success"
    assert (second / "qualification.json").read_text() == "failure"
    arm_root = tmp_path / "qualification" / "full"
    assert (arm_root / "latest-pass").resolve() == first.resolve()
    assert (arm_root / "latest").resolve() == second.resolve()
    history = json.loads((arm_root / "attempt_history.json").read_text())
    assert history["schema_version"] == "egostitch_e2e_qualification_history_v1"
    assert history["arm"] == "full"
    assert [row["attempt_id"] for row in history["attempts"]] == [first.name, second.name]
    assert [row["exit_code"] for row in history["attempts"]] == [0, 7]
    assert history["attempts"][0]["run_metadata"]["sha256"] == hashlib.sha256(
        b"metadata"
    ).hexdigest()
    assert history["attempts"][1]["validation_events"] is None


def test_pending_qualification_history_is_a_successful_pending_attempt(
    tmp_path: Path, bash_exe: str
) -> None:
    functions = _block("allocate_qualification_attempt_dir()", "select_all_visible_h20s()")
    program = (
        'QUALIFICATION_ROOT_DIR="$1"\n'
        'PYTHON_BIN="$2"\n'
        'fail() { echo "$*" >&2; return 1; }\n'
        + functions
        + '\nattempt="$(allocate_qualification_attempt_dir full)"\n'
        + 'printf \'{"verdict":"pending_manual_review"}\\n\' > "${attempt}/qualification.json"\n'
        + 'record_qualification_attempt full "${attempt}" 0\n'
        + 'update_qualification_pointer full "${attempt}" latest\n'
    )
    result = subprocess.run(
        [
            bash_exe,
            "-c",
            program,
            "bash",
            str(tmp_path / "qualification"),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arm_root = tmp_path / "qualification" / "full"
    history = json.loads((arm_root / "attempt_history.json").read_text())
    assert len(history["attempts"]) == 1
    assert history["attempts"][0]["exit_code"] == 0
    assert history["attempts"][0]["outcome"] == "success"
    assert history["attempts"][0]["verdict"] == "pending_manual_review"
    assert (arm_root / "latest").resolve().name == history["attempts"][0]["attempt_id"]
    assert not (arm_root / "latest-pass").exists()


def _write_qualification_authority_state(
    tmp_path: Path,
    rows: list[tuple[str, int, str]],
    *,
    latest_index: int | None,
    latest_pass_index: int | None = None,
) -> Path:
    arm_root = tmp_path / "qualification" / "full"
    attempts_root = arm_root / "attempts"
    attempts_root.mkdir(parents=True)
    attempts = []
    attempt_dirs = []
    for index, (verdict, exit_code, outcome) in enumerate(rows):
        attempt = attempts_root / f"attempt-{index:03d}"
        attempt.mkdir()
        (attempt / "qualification.json").write_text(
            json.dumps({"verdict": verdict}), encoding="utf-8"
        )
        attempt_dirs.append(attempt)
        attempts.append(
            {
                "attempt_id": attempt.name,
                "attempt_dir": str(attempt),
                "exit_code": exit_code,
                "outcome": outcome,
                "verdict": verdict,
            }
        )
    (arm_root / "attempt_history.json").write_text(
        json.dumps(
            {
                "schema_version": "egostitch_e2e_qualification_history_v1",
                "arm": "full",
                "attempts": attempts,
            }
        ),
        encoding="utf-8",
    )
    if latest_index is not None:
        (arm_root / "latest").symlink_to(Path("attempts") / attempt_dirs[latest_index].name)
    if latest_pass_index is not None:
        (arm_root / "latest-pass").symlink_to(
            Path("attempts") / attempt_dirs[latest_pass_index].name
        )
    return tmp_path / "qualification"


def _run_current_qualification_authority_guard(
    qualification_root: Path, bash_exe: str
) -> subprocess.CompletedProcess[str]:
    functions = _block("allocate_qualification_attempt_dir()", "select_all_visible_h20s()")
    program = (
        'QUALIFICATION_ROOT_DIR="$1"\n'
        'PYTHON_BIN="$2"\n'
        'fail() { echo "$*" >&2; return 1; }\n'
        + functions
        + '\nattempt="$(current_qualification_attempt full)" || exit $?\n'
        + '[[ -n "${attempt}" ]] || exit 90\n'
        + 'assert_current_qualification_recorded_pass full "${attempt}"\n'
    )
    return subprocess.run(
        [bash_exe, "-c", program, "bash", str(qualification_root), sys.executable],
        capture_output=True,
        text=True,
        check=False,
    )


def test_current_history_tail_and_matching_latest_exact_pass_are_authoritative(
    tmp_path: Path, bash_exe: str
) -> None:
    root = _write_qualification_authority_state(
        tmp_path, [("pass", 0, "success")], latest_index=0
    )
    result = _run_current_qualification_authority_guard(root, bash_exe)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("verdict", "exit_code", "outcome"),
    [
        ("pending_manual_review", 0, "success"),
        ("fail(persistent_clipping)", 2, "failure"),
    ],
)
def test_current_pending_or_failure_cannot_fall_back_to_stale_latest_pass(
    tmp_path: Path,
    bash_exe: str,
    verdict: str,
    exit_code: int,
    outcome: str,
) -> None:
    root = _write_qualification_authority_state(
        tmp_path,
        [("pass", 0, "success"), (verdict, exit_code, outcome)],
        latest_index=1,
        latest_pass_index=0,
    )
    result = _run_current_qualification_authority_guard(root, bash_exe)
    assert result.returncode != 0
    assert "not a recorded pass" in result.stderr


def test_stale_latest_symlink_is_rejected_even_when_history_tail_passes(
    tmp_path: Path, bash_exe: str
) -> None:
    root = _write_qualification_authority_state(
        tmp_path,
        [("pass", 0, "success"), ("pass", 0, "success")],
        latest_index=0,
        latest_pass_index=0,
    )
    result = _run_current_qualification_authority_guard(root, bash_exe)
    assert result.returncode != 0
    assert "authoritative current qualification" in result.stderr


def test_history_append_without_latest_update_is_rejected(
    tmp_path: Path, bash_exe: str
) -> None:
    root = _write_qualification_authority_state(
        tmp_path, [("pass", 0, "success")], latest_index=None
    )
    result = _run_current_qualification_authority_guard(root, bash_exe)
    assert result.returncode != 0
    assert "authoritative current qualification" in result.stderr


def test_failed_qualification_updates_latest_without_replacing_latest_pass() -> None:
    body = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert body.index('update_qualification_pointer "${arm}" "${output_dir}" latest') < body.index(
        'if [[ "${pipeline_status}" -ne 0 ]]'
    )
    assert "latest-pass" not in body


def test_pending_qualification_is_successful_but_does_not_advance_latest_pass() -> None:
    body = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert '[[ "${verdict}" != "pending_manual_review" ]]' in body
    assert "latest-pass" not in body
    assert body.index("record_qualification_attempt") < body.index(
        'update_qualification_pointer "${arm}" "${output_dir}" latest'
    )


def test_new_qualification_rejects_an_automatic_pass_and_records_failure() -> None:
    body = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert "forbidden automatic verdict" in body
    rejection = body.index('[[ "${verdict}" != "pending_manual_review" ]]')
    assert rejection < body.index("pipeline_status=1", rejection)
    assert rejection < body.index("record_qualification_attempt")
    assert "latest-pass" not in body


def test_neither_stage_forces_a_shared_pack_dir() -> None:
    """`runtime.pack_dir` is the single source of truth for the n_ground-keyed pack."""
    body = _strip_comments(_block("run_qualification()", 'case "${1:-}" in'))
    assert "--pack-dir" not in body


def test_both_stages_refuse_to_edit_or_promote_the_registration() -> None:
    text = RUNNER.read_text()
    for stage, until in (
        ("run_qualification()", "run_formal()"),
        ("run_formal()", 'case "${1:-}" in'),
    ):
        body = _strip_comments(_block(stage, until))
        assert "REGISTRATION_SHA256_BEFORE" in body
        assert "trap assert_registration_unchanged EXIT" in body
    guard = _block("assert_registration_unchanged()", "arm_config()")
    assert "must not edit or promote the registration" in guard
    assert "sed -i" not in text
    assert 'status": "BINDING' not in text


def test_the_sanity_suite_cannot_silently_shrink_when_the_tests_are_split() -> None:
    body = _block("run_sanity_suite()", "run_qualification()")
    # Globbed, not listed: splitting tests/test_train_egostitch.py must not drop
    # the e2e contracts it carries.
    assert "tests/test_train_egostitch*.py" in body
    assert "tests/experiments/test_auprc_tolerance.py" in body
    assert "tests/model/test_egostitch_conditioning.py" in body
    assert "tests/test_e2_pipeline.py" in body
    assert "tests/test_hpc_qualification.py" in body


def test_formal_requires_binding_registration_clean_checkout_and_a_passing_qualification() -> None:
    body = _strip_comments(_block("run_formal()", 'case "${1:-}" in'))
    assert "--run-kind formal" in body
    # Every refusal must precede the launch that would consume GPU hours.
    launch = body.index("--run-kind formal")
    for guard in (
        "assert_clean_checkout formal",
        "assert_source_resolves_to_repo",
        "assert_formal_registration",
        'assert_qualification_passed "${arm}"',
        '"${DETECTED_GPU_COUNT}" -eq "${FORMAL_GPU_COUNT}"',
    ):
        assert body.index(guard) < launch
    # Scientific execution order: the full arm's preflight gates the rest.
    assert 'if [[ "${arm}" != "full" ]]' in body
    assert body.index("assert_full_preflight") < launch
    assert "readonly FORMAL_GPU_COUNT=4" in RUNNER.read_text()


def test_formal_reads_current_latest_and_cannot_fall_back_to_stale_latest_pass() -> None:
    guard = _strip_comments(_block("assert_qualification_passed()", "assert_full_preflight()"))
    assert 'attempt_dir="$(current_qualification_attempt "${arm}")"' in guard
    assert 'assert_current_qualification_recorded_pass "${arm}" "${attempt_dir}"' in guard
    assert 'local report="${attempt_dir}/qualification.json"' in guard
    assert "latest-pass" not in guard
    assert 'verdict != "pass"' in guard


def test_formal_forwards_the_verified_qualification_artifact_to_the_worker() -> None:
    """The worker refuses a formal run without it, and cannot derive the path itself.

    `_bind_feature_standardization` compares its own `feature_stats_sha256`
    against the recorded one instead of reading a config pin, so an unforwarded
    path is a refusal rather than a fail-open -- but it is a refusal after the
    pack stage has already run. The launcher keys directories by the short arm
    name while `_e2e_arm_name_from_config` returns the long one, so a
    convention-based derivation inside the worker would silently miss the file.
    """
    body = _strip_comments(_block("run_formal()", 'case "${1:-}" in'))
    assert '--qualification-artifact "${QUALIFICATION_REPORT_PATH}"' in body
    # The forwarded path is the one the preflight resolved and verified, not a
    # second path built by convention next to it.
    guard = _strip_comments(_block("assert_qualification_passed()", "assert_full_preflight()"))
    assert 'QUALIFICATION_REPORT_PATH="${report}"' in guard
    assert body.index('assert_qualification_passed "${arm}"') < body.index(
        "--qualification-artifact"
    )
    # The qualification stage must never receive one: it is that stage's output.
    qualify = _strip_comments(_block("run_qualification()", "run_formal()"))
    assert "--qualification-artifact" not in qualify


def test_formal_registration_guard_refuses_draft_without_legacy_markers() -> None:
    guard = _block("assert_formal_registration()", "assert_registration_unchanged()")
    assert '== "BINDING"' in guard
    assert "REQUIRED-BEFORE-BINDING" not in RUNNER.read_text()


def test_formal_full_arm_preflight_requires_eligibility_and_liveness() -> None:
    """Checkpoint eligibility is retained in both stages despite manual review."""
    guard = _block("assert_full_preflight()", "produce_formal_probe_artifact()")
    assert "selected_checkpoint_eligible" in guard
    assert "validation_liveness_pass" in guard
    assert 'm.get("run_kind")=="formal"' in guard
    assert "preregistration_sha256" in guard


def test_formal_produces_the_registered_probe_artifact_from_the_full_arm_only() -> None:
    body = _strip_comments(_block("run_formal()", 'case "${1:-}" in'))
    assert "produce_formal_probe_artifact" in body
    assert body.index("--run-kind formal") < body.index("produce_formal_probe_artifact")
    producer = _block("produce_formal_probe_artifact()", "run_sanity_suite()")
    assert "src.experiments.probes produce-e2e" in producer
    assert "--scope formal_train" in producer
    # The path is read out of the registration, never restated here.
    assert "expected_path" in producer
    assert 'probe.get("source_arm") != "full"' in producer


def test_formal_validates_the_clip_family_and_rms_margins_after_training() -> None:
    body = _strip_comments(_block("run_formal()", 'case "${1:-}" in'))
    assert "validate_e2e_qualification_profile" in body
    assert body.index("--run-kind formal") < body.index("validate_e2e_qualification_profile")


# ------------------------------------------------------- clip/family/RMS margin verdict
# The gate necessarily runs after `src.e2_pipeline` has published the run, so
# `run_metadata.json` already reads `status: complete` /
# `formal_artifacts_published: true` when the margins are still unknown.
# Publication cannot be gated from this launcher, so the fail-closed half is
# downstream: formal scoring and the G5 gate both demand the persisted verdict
# through the one shared validator, and it is bound to the run it describes.


def _margin_writer_program() -> str:
    return _embedded_python_containing(
        _block("run_formal()", 'case "${1:-}" in'), "validate_e2e_qualification_profile"
    )


def _shell_margin_verdict_filename() -> str:
    match = re.search(r'readonly MARGIN_VERDICT_FILENAME="([^"]+)"', RUNNER.read_text())
    assert match is not None, "the launcher no longer declares MARGIN_VERDICT_FILENAME"
    return match.group(1)


def test_the_launcher_and_its_consumers_agree_on_the_verdict_filename() -> None:
    """The writer is shell-side and every reader is Python-side; a drift fails open.

    The launcher would keep writing under its own name and every consumer would
    keep looking for a file that is never there -- which refuses, but refuses
    every arm, so the refusal would be read as a broken gate rather than as a
    margin failure.
    """
    from src.score_universe import E2E_MARGIN_VERDICT_FILENAME

    assert _shell_margin_verdict_filename() == E2E_MARGIN_VERDICT_FILENAME


def _run_embedded(program: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one of the launcher's embedded programs against the real repository."""
    return subprocess.run(
        [sys.executable, "-c", program, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )


def _passing_margin_profile() -> dict[str, object]:
    """The smallest profile that clears every registered clip/family/RMS margin.

    Clip p1 = 1.0 clears the highest registered floor (0.15), the minimum clears
    0.0012, no coefficient is below 0.1 so the streak is 0, the family-ratio p99
    of 1.0 clears 40.0, and one complete submodule-RMS probe row is present.
    """
    return {
        "total_optimizer_steps": 4,
        "optimizer_step_gradients": [
            {"optimizer_group_gradients": {"generator": {"active": True, "clip_coefficient": 1.0}}}
            for _ in range(4)
        ],
        "gradient_norm_series": [
            {
                "alpha": 1.0,
                "family_group_ratios": {"generator": 1.0},
                "submodule_gradient_rms": {
                    "grad_rms_trunk": 1.0,
                    "grad_rms_ste": 1.0,
                    "grad_rms_content": 1.0,
                },
            }
        ],
    }


def _published_run(tmp_path: Path, profile: dict[str, object]) -> Path:
    """A run directory as `_publish_staged` leaves it: complete, margins unknown."""
    run_dir = tmp_path / "full"
    run_dir.mkdir()
    (run_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "complete", "formal_artifacts_published": True}), encoding="utf-8"
    )
    return run_dir


def test_margin_verdict_binds_to_the_profile_and_the_run_metadata(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path, _passing_margin_profile())
    verdict_path = run_dir / _shell_margin_verdict_filename()

    result = _run_embedded(_margin_writer_program(), str(run_dir), str(verdict_path))

    assert result.returncode == 0, result.stderr
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["status"] == "pass"
    for key, filename in (
        ("profile_sha256", "profile.json"),
        ("run_metadata_sha256", "run_metadata.json"),
    ):
        assert (
            verdict[key] == hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()
        ), key


def test_failed_margins_record_a_fail_verdict_and_exit_nonzero(tmp_path: Path) -> None:
    """The failed arm must be unusable, not merely un-launchable."""
    run_dir = _published_run(tmp_path, {})
    verdict_path = run_dir / _shell_margin_verdict_filename()
    # A stale `pass` from an earlier attempt must not survive this run.
    verdict_path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    result = _run_embedded(_margin_writer_program(), str(run_dir), str(verdict_path))

    assert result.returncode != 0
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["status"] == "fail"
    assert "gradient coverage" in str(verdict["error"])
    # And the recorded failure is refused by the shared downstream validator.
    from src.score_universe import validate_e2e_margin_verdict

    with pytest.raises(ValueError, match=r"margin verdict is 'fail', not 'pass'"):
        validate_e2e_margin_verdict(run_dir / "run_metadata.json", label="full arm")


def test_full_arm_preflight_uses_the_shared_downstream_validator(tmp_path: Path) -> None:
    """One definition of the rule: the launcher cannot drift from the consumers."""
    guard = _block("assert_full_preflight()", "produce_formal_probe_artifact()")
    program = _embedded_python_containing(guard, "validate_e2e_margin_verdict")
    assert "from src.score_universe import validate_e2e_margin_verdict" in program

    run_dir = _published_run(tmp_path, _passing_margin_profile())
    metadata = run_dir / "run_metadata.json"
    verdict_path = run_dir / _shell_margin_verdict_filename()
    writer = _run_embedded(_margin_writer_program(), str(run_dir), str(verdict_path))
    assert writer.returncode == 0, writer.stderr
    assert _run_embedded(program, str(metadata)).returncode == 0

    stale = json.loads(verdict_path.read_text(encoding="utf-8"))
    stale["run_metadata_sha256"] = "0" * 64
    verdict_path.write_text(json.dumps(stale), encoding="utf-8")
    assert _run_embedded(program, str(metadata)).returncode != 0

    verdict_path.unlink()
    assert _run_embedded(program, str(metadata)).returncode != 0


def test_arm_selector_covers_six_trained_arms_and_rejects_the_scoring_controls() -> None:
    selector = _block("arm_config()", "arm_output_dir()")
    for arm, path in CONFIGS.items():
        assert f"configs/{path.name}" in selector, arm
    # The scoring-time controls reuse the full arm's checkpoint and must not be
    # launchable as trained arms.
    assert "structure_control_6a_v3" in selector
    assert "structure_control_6e_v1" in selector
    assert "is not trained" in selector
    assert "unknown arm" in selector
    # No v2 config may be reachable from the v3 launcher.
    text = RUNNER.read_text()
    assert "configs/egostitch_e2e_breadth_first.yaml" not in text
    assert "preregistration_v2.json" not in text


# ------------------------------------------------------- Stage-2 qualification preflight (Sec 8.2)


def _write_report(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "qualification.json"
    path.write_text(payload, encoding="utf-8")
    return path


def _run_qualification_check(report: Path) -> subprocess.CompletedProcess[str]:
    program = _embedded_python(_block("assert_qualification_passed()", "assert_full_preflight()"))
    return subprocess.run(
        [sys.executable, "-c", program, str(report)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_formal_refuses_a_missing_qualification_report() -> None:
    guard = _strip_comments(_block("assert_qualification_passed()", "assert_full_preflight()"))
    assert '[[ -s "${report}" ]]' in guard
    assert "requires a completed qualification stage" in guard
    assert 'local report="${attempt_dir}/qualification.json"' in guard


def test_formal_accepts_a_passing_qualification_report(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        '{"verdict": "pass", "feature_stats_sha256": "a", "model_config_sha256": "b"}',
    )
    assert _run_qualification_check(report).returncode == 0


@pytest.mark.parametrize(
    "payload",
    [
        '{"verdict": "fail(slot_collapse)", "feature_stats_sha256": "a", '
        '"model_config_sha256": "b"}',
        '{"verdict": "fail(no_eligible_checkpoint)", "feature_stats_sha256": "a", '
        '"model_config_sha256": "b"}',
        '{"feature_stats_sha256": "a", "model_config_sha256": "b"}',
        '{"verdict": "pass", "model_config_sha256": "b"}',
        '{"verdict": "pass", "feature_stats_sha256": "", "model_config_sha256": "b"}',
        '{"verdict": "pass", "feature_stats_sha256": "a", "model_config_sha256": null}',
    ],
)
def test_formal_refuses_a_failing_or_malformed_qualification_report(
    tmp_path: Path, payload: str
) -> None:
    """A `fail` verdict, an absent verdict and an unusable digest are all refusals.

    Digest *equality* against the live config is not checkable from the
    launcher — only the worker can recompute both — so it is asserted in
    `train_egostitch.validate_qualification_artifact`, not here.
    """
    assert _run_qualification_check(_write_report(tmp_path, payload)).returncode != 0
