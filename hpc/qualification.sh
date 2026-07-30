#!/usr/bin/env bash
# Fail-closed launcher for the two-stage E2E ladder (spec §14.4.7): the
# qualification stage and the bound formal stage.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PREREGISTRATION="docs/registrations/g5_e2e_stage1_preregistration_v4.json"
readonly FORMAL_GPU_COUNT=4
readonly EXPECTED_GPU_NAME="NVIDIA H20"
# Both stages train on the full V_fit and validate on the single 512-node
# V_hold; they differ only in optim.epochs. Three epochs is the launcher's short
# qualification schedule, and it still traverses phases A -> B -> C because the curriculum
# scales with schedule_total_steps rather than with a fixed step count.
readonly QUALIFICATION_EPOCHS=3
readonly QUALIFICATION_ROOT_DIR="outputs/egostitch_e2e_stage1_v3/qualification"
# The clip/family/RMS margin gate can only run after the pipeline has published
# the run, so "complete" carries no margin evidence. This is where its verdict
# is persisted, next to the artifacts it was computed from. Publication cannot
# be made to wait for it from here, so the fail-closed half lives downstream:
# `src.score_universe.validate_e2e_margin_verdict` is required by formal scoring
# and by the G5 gate, and a run without a passing, run-bound verdict is
# unusable no matter what its run_metadata.json says.
readonly MARGIN_VERDICT_FILENAME="margin_verdict.json"
DETECTED_GPU_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  hpc/qualification.sh qualify <full|f_only|pair_topology|p0|cosine_pool|no_l_rel>
  hpc/qualification.sh formal  <full|f_only|pair_topology|p0|cosine_pool|no_l_rel>

Both stages run the rev-3.2 trainer over the identical universe: they
train on the full V_fit and validate on the single V_hold, and differ only in
optim.epochs. Neither stage may open a held-out path; that boundary is enforced
inside the worker as a path check on both run kinds, not by an isolated data
root, so both commands run directly in the repository.

qualify is the development loop. It auto-detects and uses every visible H20,
overrides optim.epochs to 3, writes into
outputs/egostitch_e2e_stage1_v3/qualification/<arm>, and its verdict is
guards-only: pass iff no fail-fast guard tripped. Checkpoint eligibility is
unaffected — it is enforced in both stages. qualify never edits or promotes the
registration and deliberately does not require a clean checkout, because
iterating on the model is the point.

formal requires exactly 4 visible NVIDIA H20s, a clean checkout, a fully
resolved BINDING registration (the active v4 file is still DRAFT), and a qualification for the same arm whose
verdict is pass and whose model and feature digests equal the ones this formal
config and its shared F0 pack produce. It launches one registered trained arm;
the two scoring-time controls reuse the full arm's checkpoint and are not
launched here. Scientific execution order remains full first, with the full-arm
eligibility/liveness preflight and its persisted clip/family/RMS margin verdict
both required before the remaining arms are launched. After training it
validates those margins and records the verdict next to the run; that verdict is
required again by formal scoring and by the G5 gate, so an arm whose margins
failed cannot be scored or screened. For the full arm it also produces the
registered formal_train probe artifact that the G5 gate evaluator consumes.

--max-steps is never substituted for either schedule.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

registration_sha256() {
  sha256sum "${PREREGISTRATION}" | cut -d' ' -f1
}

registration_status() {
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${PREREGISTRATION}"
}

allocate_qualification_attempt_dir() {
  local arm="$1"
  local attempts_root="${QUALIFICATION_ROOT_DIR}/${arm}/attempts"
  mkdir -p "${attempts_root}"
  mktemp -d "${attempts_root}/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX"
}

update_qualification_pointer() {
  local arm="$1"
  local attempt_dir="$2"
  local pointer="$3"
  local arm_root="${QUALIFICATION_ROOT_DIR}/${arm}"
  [[ "${attempt_dir}" == "${arm_root}/attempts/"* ]] || \
    fail "qualification attempt is outside the immutable attempt history: ${attempt_dir}"
  [[ "${pointer}" == "latest" || "${pointer}" == "latest-pass" ]] || \
    fail "invalid qualification pointer: ${pointer}"
  ln -sfn "attempts/$(basename "${attempt_dir}")" "${arm_root}/${pointer}"
}

record_qualification_attempt() {
  local arm="$1"
  local attempt_dir="$2"
  local exit_code="$3"
  local index_path="${QUALIFICATION_ROOT_DIR}/${arm}/attempt_history.json"
  "${PYTHON_BIN}" -c '
import hashlib, json, os, sys, tempfile
from datetime import UTC, datetime
from pathlib import Path

index_path, arm, attempt_dir, exit_code = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])

def stable_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)

def artifact(name):
    path = attempt_dir / name
    if not path.is_file():
        return None
    return {"path": stable_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

qualification = artifact("qualification.json")
verdict = None
if qualification is not None:
    try:
        verdict = json.loads((attempt_dir / "qualification.json").read_text()).get("verdict")
    except (OSError, json.JSONDecodeError, AttributeError):
        verdict = "unreadable"
entry = {
    "attempt_id": attempt_dir.name,
    "attempt_dir": stable_path(attempt_dir),
    "recorded_at_utc": datetime.now(UTC).isoformat(),
    "exit_code": exit_code,
    "outcome": "success" if exit_code == 0 else "failure",
    "verdict": verdict,
    "qualification": qualification,
    "run_metadata": artifact("run_metadata.json"),
    "validation_events": artifact("v_hold_validation_events.jsonl"),
}
if index_path.is_file():
    payload = json.loads(index_path.read_text())
else:
    payload = {"schema_version": "egostitch_e2e_qualification_history_v1", "arm": arm, "attempts": []}
if payload.get("schema_version") != "egostitch_e2e_qualification_history_v1" or payload.get("arm") != arm:
    raise SystemExit("qualification history index identity mismatch")
attempts = payload.get("attempts")
if not isinstance(attempts, list):
    raise SystemExit("qualification history attempts must be a list")
if any(row.get("attempt_id") == entry["attempt_id"] or row.get("attempt_dir") == entry["attempt_dir"] for row in attempts):
    raise SystemExit("qualification attempt is already indexed")
attempts.append(entry)
index_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".attempt-history-", suffix=".json", dir=index_path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, index_path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
' "${index_path}" "${arm}" "${attempt_dir}" "${exit_code}" || \
    fail "could not atomically index qualification attempt: ${attempt_dir}"
}

select_all_visible_h20s() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
  mapfile -t gpu_rows < <(
    nvidia-smi --query-gpu=index,name --format=csv,noheader
  )
  [[ "${#gpu_rows[@]}" -ge 1 ]] || fail "no visible NVIDIA GPUs detected"
  local gpu_row
  local gpu_index
  local gpu_name
  local -a gpu_ids=()
  for gpu_row in "${gpu_rows[@]}"; do
    IFS=',' read -r gpu_index gpu_name <<<"${gpu_row}"
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_name="${gpu_name#${gpu_name%%[![:space:]]*}}"
    [[ "${gpu_name}" == "${EXPECTED_GPU_NAME}" ]] || \
      fail "expected ${EXPECTED_GPU_NAME}, found ${gpu_name} at GPU ${gpu_index}"
    gpu_ids+=("${gpu_index}")
  done
  DETECTED_GPU_COUNT="${#gpu_ids[@]}"
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_ids[*]}")"
  export DETECTED_GPU_COUNT CUDA_VISIBLE_DEVICES
  echo "auto-detected ${DETECTED_GPU_COUNT} visible ${EXPECTED_GPU_NAME} GPUs: ${CUDA_VISIBLE_DEVICES}"
}

# `python -m` puts the cwd first on sys.path and this script exports PYTHONPATH,
# so verify what Python really resolves rather than trusting path ordering: an
# implementation executed from anywhere but REPO_ROOT is not the one the guards
# and registration describe.
assert_source_resolves_to_repo() {
  local resolved
  resolved="$("${PYTHON_BIN}" -c 'import src, pathlib; print(pathlib.Path(src.__file__).resolve().parent.parent)')" || \
    fail "could not resolve the src package"
  [[ "${resolved}" == "${REPO_ROOT}" ]] || \
    fail "src resolves to ${resolved}, not ${REPO_ROOT}"
}

# The registration pins the config digest but nothing pins the source state, and
# the calibration freeze manifest that used to do it is gone with the pre-binding
# ladder. Formal evidence therefore has to come from a committed tree.
assert_clean_checkout() {
  local stage="$1"
  local dirty
  dirty="$(git status --porcelain)"
  [[ -z "${dirty}" ]] || \
    fail "${stage} requires a clean checkout; the working tree has uncommitted changes"
}

assert_formal_registration() {
  [[ "$(registration_status)" == "BINDING" ]] || \
    fail "formal E2E training requires registration status BINDING"
}

assert_qualification_registration_open() {
  [[ "$(registration_status)" != "BINDING" ]] || \
    fail "qualification is frozen once the registration is BINDING; post-binding attempts would escape the registered K disclosure"
}

assert_registration_unchanged() {
  [[ "$(registration_sha256)" == "${REGISTRATION_SHA256_BEFORE}" ]] || \
    fail "this stage must not edit or promote the registration"
}

arm_config() {
  case "$1" in
    full) echo "configs/egostitch_e2e_v3_full_breadth_first.yaml" ;;
    f_only) echo "configs/egostitch_e2e_v3_f_only_breadth_first.yaml" ;;
    pair_topology) echo "configs/egostitch_e2e_v3_pair_topology_breadth_first.yaml" ;;
    p0) echo "configs/egostitch_e2e_v3_p0_breadth_first.yaml" ;;
    cosine_pool) echo "configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml" ;;
    no_l_rel) echo "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml" ;;
    structure_control_6a_v3|structure_control_6e_v1)
      fail "$1 is a scoring-time control that reuses the full arm's checkpoint; it is not trained" ;;
    *) fail "unknown arm: $1" ;;
  esac
}

arm_output_dir() {
  "${PYTHON_BIN}" -c '
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = config["output_dir"]
if not isinstance(value, str) or not value:
    raise SystemExit("config output_dir must be a non-empty string")
print(value)
' "$1" || fail "could not read output_dir from $1"
}

# Stage 2 refuses to launch on a missing, failing or mismatched qualification
# for the same arm (design 2026-07-29 Sec 3). Two fail-closed reads: first the
# launcher-local shape/verdict check, which refuses a malformed report without
# importing the worker; then the authoritative one, which recomputes
# `model_config_hash` from this arm's config and compares `feature_stats_sha256`
# against the digest this formal run will bind. That digest is read from the F0
# pack both stages share -- `load_feature_stats` recomputes it from the stored
# constants, and the run itself re-verifies those constants against V_fit -- so
# an absent pack is a refusal, never a skipped comparison.
assert_qualification_passed() {
  local arm="$1"
  local config="$2"
  local report="${QUALIFICATION_ROOT_DIR}/${arm}/latest-pass/qualification.json"
  [[ -s "${report}" ]] || \
    fail "formal training requires a completed qualification stage: ${report}"
  "${PYTHON_BIN}" -c '
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
verdict = report.get("verdict")
if verdict != "pass":
    raise SystemExit("qualification verdict is %r, not %r" % (verdict, "pass"))
for key in ("feature_stats_sha256", "model_config_sha256"):
    value = report.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit("qualification report lacks a usable %s" % key)
' "${report}" || fail "qualification did not pass: ${report}"
  "${PYTHON_BIN}" -c '
import sys
from pathlib import Path

from src.data.feature_stats import load_feature_stats
from src.train_egostitch import (
    _PACK_FEATURE_STATS_FILENAME,
    load_config,
    validate_qualification_artifact,
)

config_path, report = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_config(config_path)
if cfg.runtime is None:
    raise SystemExit("config carries no runtime section: %s" % config_path)
stats_path = cfg.runtime.pack_dir / _PACK_FEATURE_STATS_FILENAME
if not stats_path.is_file():
    raise SystemExit(
        "the shared F0 pack carries no feature-standardization statistics at %s, "
        "so the formal digest cannot be compared; run the qualification stage for "
        "this arm first" % stats_path
    )
validate_qualification_artifact(
    report, cfg, feature_stats_sha256=load_feature_stats(stats_path).digest
)
' "${config}" "${report}" || \
    fail "qualification does not match this formal config: ${report}"
  # Published for the launch site so the worker re-checks the *same* file this
  # preflight just verified, rather than a second path built by convention. The
  # worker's own comparison is not redundant: it runs against the statistics the
  # run actually assembled, where this one runs against the pack on disk.
  QUALIFICATION_REPORT_PATH="${report}"
  echo "qualification verified: ${report}"
}

# A published run says nothing about its margins: the gate below necessarily
# runs after the pipeline has marked the run complete. Requiring the persisted
# verdict is what stops a later arm from launching on a full arm whose margins
# failed, and binding it to the profile and run-metadata digests is what stops a
# verdict left by an earlier full run from standing in for this one.
assert_full_preflight() {
  local full_dir="outputs/egostitch_e2e_stage1_v3/full"
  local metadata="${full_dir}/run_metadata.json"
  local margins="${full_dir}/${MARGIN_VERDICT_FILENAME}"
  [[ -s "${metadata}" ]] || \
    fail "remaining arms require the completed full-arm preflight: ${metadata}"
  "${PYTHON_BIN}" -c \
    'import hashlib,json,sys; from pathlib import Path; m=json.loads(Path(sys.argv[1]).read_text()); r=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(); ok=m.get("status")=="complete" and m.get("run_kind")=="formal" and m.get("selected_checkpoint_eligible") is True and m.get("validation_liveness_pass") is True and m.get("preregistration_sha256")==r; raise SystemExit(0 if ok else 1)' \
    "${metadata}" "${PREREGISTRATION}" || \
    fail "full-arm run is not complete, eligible, live, and registration-matched"
  [[ -s "${margins}" ]] || \
    fail "remaining arms require the full arm's persisted margin verdict: ${margins}"
  # One definition of the rule, shared with formal scoring and the G5 gate, so a
  # launcher-local copy cannot drift away from what actually consumes the run.
  "${PYTHON_BIN}" -c '
import sys
from pathlib import Path

from src.score_universe import validate_e2e_margin_verdict

validate_e2e_margin_verdict(Path(sys.argv[1]), label="full arm preflight")
' "${metadata}" || \
    fail "the full arm's clip/family/RMS margin verdict is failing or from another run"
}

# The G5 gate evaluator refuses to run without this artifact and compares its
# path against the registered probe_artifact.expected_path, so the path is read
# out of the registration rather than restated here. Pure post-processing over an
# already written checkpoint; it opens no new universe.
produce_formal_probe_artifact() {
  local stage_dir="$1"
  local output
  output="$("${PYTHON_BIN}" -c '
import json, sys
from pathlib import Path
registration = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
probe = registration.get("probe_artifact")
if not isinstance(probe, dict):
    raise SystemExit("registration does not bind probe_artifact")
if probe.get("source_arm") != "full":
    raise SystemExit("registered probe source arm is not the full arm")
path = probe.get("expected_path")
if not isinstance(path, str) or not path:
    raise SystemExit("registration does not bind probe_artifact.expected_path")
print(path)
' "${PREREGISTRATION}")" || fail "could not read the registered probe artifact path"
  mkdir -p "$(dirname "${output}")"
  "${PYTHON_BIN}" -m src.experiments.probes produce-e2e \
    --checkpoint "${stage_dir}/best.pt" \
    --run-metadata "${stage_dir}/run_metadata.json" \
    --preregistration "${PREREGISTRATION}" \
    --data-root data \
    --strategy breadth_first \
    --output "${output}" \
    --scope formal_train
  echo "produced the registered formal_train probe artifact: ${output}"
}

# The train_egostitch modules are globbed rather than listed so the suite cannot
# silently shrink when that file is split: any tests/test_train_egostitch*.py is
# picked up automatically.
run_sanity_suite() {
  "${UV_BIN}" run pytest -q \
    tests/test_train_egostitch*.py \
    tests/model/test_egostitch_conditioning.py \
    tests/model/test_egostitch_trunk.py \
    tests/test_e2_pipeline.py \
    tests/test_hpc_qualification.py
}

# Neither stage passes --pack-dir. The F0/grounding pack is keyed on n_ground and
# the pack manifest rejects a mismatch outright, so the single shared pack this
# script used to force made the cosine-pool arm (n_ground 20) raise against a pack
# built at 50. Each config now names its own n_ground-keyed pack under
# runtime.pack_dir; both stages of a given arm share it, and arms that agree on
# n_ground share it too.
run_qualification() {
  local arm="$1"
  local config
  local output_dir
  local pipeline_status
  config="$(arm_config "${arm}")"
  output_dir="$(allocate_qualification_attempt_dir "${arm}")"
  assert_source_resolves_to_repo
  assert_qualification_registration_open
  REGISTRATION_SHA256_BEFORE="$(registration_sha256)"
  export REGISTRATION_SHA256_BEFORE
  trap assert_registration_unchanged EXIT

  echo "qualification stage 1/2: sanity"
  run_sanity_suite
  assert_registration_unchanged

  echo "qualification stage 2/2: ${QUALIFICATION_EPOCHS}-epoch ${arm} run on V_fit, validated on V_hold"
  select_all_visible_h20s
  # `optim.epochs` is the one value the two stages may differ in, so it is the
  # one value a launcher may substitute: --epochs is the orchestrator's
  # qualification-only override, refused for --run-kind formal so the registered
  # schedule stays the only schedule a formal run can have. The orchestrator
  # applies it to `cfg.optim.epochs` and forwards it to the worker, so the
  # staged-artifact and worker-profile validations track the short schedule
  # instead of rejecting it. --max-steps is never substituted for it: a step cap
  # truncates the curriculum where a shorter schedule rescales it.
  if "${PYTHON_BIN}" -m src.e2_pipeline --config "${config}" \
      --worker-module src.train_egostitch \
      --run-kind qualification \
      --epochs "${QUALIFICATION_EPOCHS}" \
      --output-dir "${output_dir}"; then
    pipeline_status=0
  else
    pipeline_status=$?
  fi
  record_qualification_attempt "${arm}" "${output_dir}" "${pipeline_status}"
  update_qualification_pointer "${arm}" "${output_dir}" latest
  assert_registration_unchanged
  if [[ "${pipeline_status}" -ne 0 ]]; then
    echo "qualification attempt failed and was retained: ${output_dir}" >&2
    return "${pipeline_status}"
  fi
  update_qualification_pointer "${arm}" "${output_dir}" latest-pass
  echo "qualification completed; verdict: ${output_dir}/qualification.json"
}

run_formal() {
  local arm="$1"
  local config
  local output_dir
  config="$(arm_config "${arm}")"
  output_dir="$(arm_output_dir "${config}")"
  select_all_visible_h20s
  [[ "${DETECTED_GPU_COUNT}" -eq "${FORMAL_GPU_COUNT}" ]] || \
    fail "formal E2E training requires exactly ${FORMAL_GPU_COUNT} visible H20 GPUs, found ${DETECTED_GPU_COUNT}"
  assert_clean_checkout formal
  assert_source_resolves_to_repo
  assert_formal_registration
  assert_qualification_passed "${arm}" "${config}"
  if [[ "${arm}" != "full" ]]; then
    assert_full_preflight
  fi
  REGISTRATION_SHA256_BEFORE="$(registration_sha256)"
  export REGISTRATION_SHA256_BEFORE
  trap assert_registration_unchanged EXIT

  echo "formal stage 1/3: registered ${arm} training run"
  # The worker refuses a formal run without this artifact: its
  # `feature_stats_sha256` is compared for equality against the recorded one
  # rather than pinned in the config (design 2026-07-29 Sec 3). The path is the
  # one `assert_qualification_passed` resolved and verified above; the worker
  # cannot derive it, because this launcher keys directories by the short arm
  # name while the worker knows the arm only by its config-derived name.
  "${PYTHON_BIN}" -m src.e2_pipeline --config "${config}" \
    --worker-module src.train_egostitch \
    --run-kind formal \
    --qualification-artifact "${QUALIFICATION_REPORT_PATH}"
  assert_registration_unchanged

  # The pipeline has already stamped `status: complete` and
  # `formal_artifacts_published: true` by the time this runs, so completion
  # carries no margin evidence and the verdict is the only thing that does.
  # Every downstream consumer -- formal scoring and the G5 gate -- demands it
  # through `validate_e2e_margin_verdict`, so a failing arm is unusable rather
  # than merely un-launchable. Any stale verdict is removed first; the record
  # written on failure says `fail`, which every consumer refuses exactly as it
  # refuses absence. `profile_sha256` binds it to the artifacts the margins were
  # computed from and `run_metadata_sha256` to the run those consumers read.
  echo "formal stage 2/3: clip/family/RMS margins"
  "${PYTHON_BIN}" -c '
import hashlib, json, sys
from pathlib import Path

from src.train_egostitch import validate_e2e_qualification_profile

run_dir, verdict = Path(sys.argv[1]), Path(sys.argv[2])
profile = run_dir / "profile.json"
verdict.unlink(missing_ok=True)
binding = {
    "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "run_metadata_sha256": hashlib.sha256(
        (run_dir / "run_metadata.json").read_bytes()
    ).hexdigest(),
}
def record(summary):
    verdict.write_text(
        json.dumps({**summary, **binding}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
try:
    summary = validate_e2e_qualification_profile(profile)
except Exception as error:
    record({"status": "fail", "error": "%s: %s" % (type(error).__name__, error)})
    raise
record(summary)
print("margin verdict: %s" % verdict)
' "${output_dir}" "${output_dir}/${MARGIN_VERDICT_FILENAME}" || \
    fail "clip/family/RMS margins failed for ${arm}; the recorded verdict refuses every consumer"
  assert_registration_unchanged

  echo "formal stage 3/3: registered formal_train probe artifact"
  if [[ "${arm}" == "full" ]]; then
    produce_formal_probe_artifact "${output_dir}"
  else
    echo "skipped: the registered probe artifact is produced from the full arm only"
  fi
  assert_registration_unchanged
  echo "formal ${arm} completed: ${output_dir}"
}

case "${1:-}" in
  help|--help|-h|"")
    usage
    exit 0
    ;;
esac

[[ -d "${REPO_ROOT}" ]] || fail "repository not found at ${REPO_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment not found at ${PYTHON_BIN}"
[[ -x "${UV_BIN}" ]] || fail "uv not found at ${UV_BIN}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# rev-3.1's working set is close to the H20's 95 GiB: the 2026-07-27 calibration
# OOM'd on an idle card with 89.51 GiB allocated and a further 4.15 GiB reserved
# but unallocated, failing an 856 MiB request. Expandable segments reclaim that
# fragmentation. This changes allocator behaviour only, never numerics, so it is
# set for every stage rather than tuned per run.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

case "${1:-}" in
  qualify)
    [[ $# -eq 2 ]] || fail "qualify requires exactly one arm"
    run_qualification "$2"
    ;;
  formal)
    [[ $# -eq 2 ]] || fail "formal requires exactly one arm"
    run_formal "$2"
    ;;
  *)
    usage >&2
    fail "unknown command: $1"
    ;;
esac
