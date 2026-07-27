#!/usr/bin/env bash
# Fail-closed launcher for the §13.19/§14.4.7 qualification and bound formal contexts.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PREREGISTRATION="docs/registrations/g5_e2e_stage1_preregistration_v3.json"
readonly FULL_CONFIG="configs/egostitch_e2e_v3_full_breadth_first.yaml"
readonly FORMAL_GPU_COUNT=4
readonly EXPECTED_GPU_NAME="NVIDIA H20"
readonly QUALIFICATION_ROOT="${EGOSTITCH_QUALIFICATION_ROOT:-}"
DETECTED_GPU_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  EGOSTITCH_QUALIFICATION_ROOT=/path/to/safe-root hpc/qualification.sh calibrate
  EGOSTITCH_QUALIFICATION_ROOT=/path/to/safe-root hpc/qualification.sh rehearse
  hpc/qualification.sh formal <full|f_only|pair_topology|p0|cosine_pool|no_l_rel>

calibrate and rehearse are train/validation-only and never edit or promote the
DRAFT registration. They auto-detect and use every visible H20, matching the
formal launch style.

calibrate runs sanity -> registered 2,000-step overfit -> probes -> gates on
V_fit only, and stops at the first failure. It never opens V_qual. Its purpose
is to produce the measurements the owner freezes into the DRAFT registration.

rehearse spends one of the three registered attempts on the single prospective
V_qual rehearsal. It refuses to start until every registered pre-binding gate
threshold is frozen, because a rehearsal against an unresolved gate cannot be
prospective. Freezing thresholds is an owner action, never this script's.

formal requires exactly 4 visible NVIDIA H20s and a fully resolved BINDING
registration. It launches one registered trained arm; the two scoring-time
controls reuse the full arm's checkpoint and are not launched here. Scientific
execution order remains full first, with the full-arm eligibility/liveness
preflight required before the remaining arms are launched.

Both qualification stages execute the registered v3 trainer. --max-steps is
never substituted for either the 2,000-step overfit or the 30-epoch rehearsal.
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

registration_has_unresolved_marker() {
  "${PYTHON_BIN}" -c \
    'import sys; raise SystemExit(0 if "REQUIRED-BEFORE-BINDING" in open(sys.argv[1]).read() else 1)' \
    "${PREREGISTRATION}"
}

# A rehearsal is only prospective if every gate it will be judged against is
# already frozen. This is narrower than the whole-file marker scan above: the
# formal path needs the entire registration resolved, but a rehearsal needs
# exactly the gate thresholds.
assert_prebinding_gates_frozen() {
  "${PYTHON_BIN}" -c '
import json, sys
gates = json.load(open(sys.argv[1]))["prebinding_qualification"]["gates"]
unresolved = [
    g.get("id")
    for g in gates
    if isinstance(g.get("threshold"), str) and "REQUIRED-BEFORE-BINDING" in g["threshold"]
]
if unresolved:
    sys.stderr.write("unresolved pre-binding gate thresholds: %s\n" % ", ".join(map(str, unresolved)))
    raise SystemExit(1)
' "${PREREGISTRATION}" || \
    fail "rehearsal refuses unresolved gate thresholds; freeze them from calibration first"
}

# A frozen threshold is necessary but not sufficient: a rehearsal that cannot
# evaluate every registered gate is not a qualification, and V_qual is spent the
# moment it starts. Checked here rather than read off the report afterwards.
assert_prebinding_gates_implementable() {
  "${PYTHON_BIN}" -m src.experiments.prebinding_gates check-implementable \
    --preregistration "${PREREGISTRATION}" || \
    fail "rehearsal refuses gates with no executable evaluator; implement them first"
}

# The registration permits exactly one prospective V_qual rehearsal
# (prebinding_qualification.protocol.v_qual_rehearsals). The attempt roots are
# separate directories, so a per-root guard would let attempt002 open V_qual
# again after attempt001 already did. The ledger lives under REPO_ROOT, which is
# the one path every attempt shares, and is written *before* V_qual is opened so
# an interrupted rehearsal still counts as spent.
readonly REHEARSAL_LEDGER="${REPO_ROOT}/outputs/egostitch_e2e_stage1_v3/qualification/v_qual_rehearsal_ledger.json"

assert_single_v_qual_rehearsal() {
  local attempt="$1"
  mkdir -p "$(dirname "${REHEARSAL_LEDGER}")"
  # Exclusive create, not exists-then-write: two rehearsals racing would both
  # pass a check-then-act guard and open V_qual twice. open(..., "x") is the
  # atomic primitive; the loser gets FileExistsError and stops here.
  "${PYTHON_BIN}" -c '
import json, sys
path, attempt, root, sha = sys.argv[1:5]
payload = json.dumps(
    {"attempt": attempt, "qualification_root": root, "registration_sha256": sha},
    indent=2,
    sort_keys=True,
) + "\n"
try:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(payload)
except FileExistsError:
    raise SystemExit("the single registered V_qual rehearsal is already spent: %s" % path)
' "${REHEARSAL_LEDGER}" "${attempt}" "${QUALIFICATION_ROOT}" "${REGISTRATION_SHA256_BEFORE}" || \
    fail "refusing to open V_qual: ${REHEARSAL_LEDGER}"
  echo "recorded the single V_qual rehearsal: ${REHEARSAL_LEDGER}"
}

# prebinding_qualification.protocol.threshold_freeze freezes *registration
# content and implementation* before qualification. Checking only the thresholds
# would let source or config drift between calibration and rehearsal, so the
# sole V_qual attempt could be spent on a different implementation than the one
# that produced its thresholds.
assert_implementation_frozen() {
  # Resolved against QUALIFICATION_ROOT *before* any cd: these paths are relative
  # to the isolated root, and the config/registration actually consumed by the
  # qualification commands are the isolated root's copies, not the repository's.
  # Hashing the repository copies would certify files the run never read.
  local calibration_manifest="${QUALIFICATION_ROOT}/$1"
  local config_abs="${QUALIFICATION_ROOT}/${FULL_CONFIG}"
  [[ -s "${calibration_manifest}" ]] || \
    fail "rehearsal requires the calibration freeze manifest: ${calibration_manifest}"
  # Replay the eligible set recorded at calibration. Passing it explicitly is
  # what keeps the already-fixed G3.1-G3.3 thresholds inside the digest: only
  # the ids that were genuinely unresolved back then may move.
  local eligible_args
  eligible_args="$("${PYTHON_BIN}" -c '
import json, shlex, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ids = manifest.get("freeze_eligible_gate_ids")
if not isinstance(ids, list):
    raise SystemExit("calibration manifest lacks freeze_eligible_gate_ids")
print(" ".join(shlex.quote("--freeze-eligible=%s" % i) for i in ids))
' "${calibration_manifest}")" || \
    fail "could not read the freeze-eligible gate set from the calibration manifest"
  local live_digest
  # shellcheck disable=SC2086 -- eligible_args is a shell-quoted flag list
  live_digest="$(eval "${PYTHON_BIN}" -m src.experiments.prebinding_gates canonical-digest \
    --preregistration "${QUALIFICATION_ROOT}/${PREREGISTRATION}" ${eligible_args} \
    | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["digest"])')" || \
    fail "could not compute the canonical registration digest"
  cd "${REPO_ROOT}"
  local live_commit
  live_commit="$(git rev-parse HEAD)"
  "${PYTHON_BIN}" -c '
import hashlib, json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
live_commit, config_path, live_canonical = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
problems = []
if manifest.get("implementation_commit") != live_commit:
    problems.append(
        "implementation commit drifted: calibration=%s live=%s"
        % (manifest.get("implementation_commit"), live_commit)
    )
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
if manifest.get("config_sha256") != digest:
    problems.append("config digest drifted: calibration=%s live=%s"
                    % (manifest.get("config_sha256"), digest))
# Not the raw file digest: freezing the calibrated thresholds is the one edit
# allowed between the stages. The canonical digest masks exactly those fields, so
# a flipped gate operator or a rewritten protocol field is still caught.
if manifest.get("registration_canonical_sha256") != live_canonical:
    problems.append(
        "registration changed beyond the permitted threshold freeze: calibration=%s live=%s"
        % (manifest.get("registration_canonical_sha256"), live_canonical)
    )
if problems:
    sys.stderr.write("\n".join(problems) + "\n")
    raise SystemExit(1)
' "${calibration_manifest}" "${live_commit}" "${config_abs}" "${live_digest}" || \
    fail "implementation/config drifted since calibration; re-calibrate before rehearsing"
  cd "${QUALIFICATION_ROOT}"
  echo "implementation freeze verified against ${calibration_manifest}"
}

# Written at the end of calibration so the rehearsal has something to compare
# against. Records what produced the thresholds, not what will consume them.
write_calibration_freeze_manifest() {
  # Absolute, resolved before the cd, for the same reason as
  # assert_implementation_frozen: record what this run actually consumed.
  local output_abs="${QUALIFICATION_ROOT}/$1"
  local config_abs="${QUALIFICATION_ROOT}/${FULL_CONFIG}"
  local registration_abs="${QUALIFICATION_ROOT}/${PREREGISTRATION}"
  # No --freeze-eligible here: at calibration the markers are still present, so
  # the eligible set is derived from them and recorded for the rehearsal to
  # replay. Re-deriving it at rehearsal would yield the empty set, because by
  # then those thresholds are ordinary numbers.
  local canonical
  canonical="$("${PYTHON_BIN}" -m src.experiments.prebinding_gates canonical-digest \
    --preregistration "${registration_abs}")" || \
    fail "could not compute the canonical registration digest"
  cd "${REPO_ROOT}"
  local live_commit
  live_commit="$(git rev-parse HEAD)"
  "${PYTHON_BIN}" -c '
import hashlib, json, sys
from pathlib import Path
output, commit, config_path, registration_path, canonical_json = (
    Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
)
canonical = json.loads(canonical_json)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "implementation_commit": commit,
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
    "registration_canonical_sha256": canonical["digest"],
    "freeze_eligible_gate_ids": canonical["freeze_eligible_gate_ids"],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "${output_abs}" "${live_commit}" "${config_abs}" "${registration_abs}" "${canonical}" || \
    fail "could not write the calibration freeze manifest"
  cd "${QUALIFICATION_ROOT}"
  echo "recorded the calibration freeze manifest: ${output_abs}"
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

# `python -m` puts the cwd first on sys.path, and both stages run from
# QUALIFICATION_ROOT. A `src/` under the isolated root would therefore shadow the
# REPO_ROOT copy, so the implementation actually executed could drift while a
# freeze that only inspects REPO_ROOT's commit still passes. Verify what Python
# really resolves rather than trusting PYTHONPATH ordering.
assert_source_resolves_to_repo() {
  local resolved
  resolved="$("${PYTHON_BIN}" -c 'import src, pathlib; print(pathlib.Path(src.__file__).resolve().parent.parent)')" || \
    fail "could not resolve the src package"
  [[ "${resolved}" == "${REPO_ROOT}" ]] || \
    fail "src resolves to ${resolved}, not ${REPO_ROOT}; the isolated root must not shadow the implementation"
}

# The freeze must pin the source state, not just its commit id. Checking
# cleanliness only at rehearsal would let calibration run dirty and then pass the
# freeze once the edits were reverted, since HEAD would be identical.
assert_clean_checkout() {
  local stage="$1"
  local dirty
  dirty="$(cd "${REPO_ROOT}" && git status --porcelain)"
  [[ -z "${dirty}" ]] || \
    fail "${stage} requires a clean checkout; the working tree has uncommitted changes"
}

assert_qualification_boundary() {
  [[ "$(registration_status)" == "DRAFT" ]] || \
    fail "qualification requires the registration to remain DRAFT"
  [[ ! -e data/benchmark_2025_neurips/test_graph.pkl ]] || \
    fail "qualification data root exposes forbidden test_graph.pkl"
  if find data \( -name candidate_test_edges.txt -o -name test_edges.txt \
      -o -iname '*v_select*' \) -print -quit | grep -q .; then
    fail "qualification data root exposes forbidden candidate/test/V_select manifests"
  fi
}

assert_formal_registration() {
  [[ "$(registration_status)" == "BINDING" ]] || \
    fail "formal E2E training requires registration status BINDING"
  if registration_has_unresolved_marker; then
    fail "formal E2E training refuses unresolved REQUIRED-BEFORE-BINDING markers"
  fi
}

assert_registration_unchanged() {
  [[ "$(registration_sha256)" == "${REGISTRATION_SHA256_BEFORE}" ]] || \
    fail "qualification must not edit or promote the registration"
}

# Shared entry guard for both pre-binding stages. Resolves the attempt root,
# refuses to overwrite retained evidence, and arms the registration-immutability
# trap. Echoes the attempt directory on stdout.
enter_qualification_attempt() {
  local stage="$1"
  local qualification_basename
  local attempt_number
  local attempt_dir
  [[ -n "${QUALIFICATION_ROOT}" && -d "${QUALIFICATION_ROOT}" ]] || \
    fail "${stage} requires EGOSTITCH_QUALIFICATION_ROOT to name the isolated data root"
  qualification_basename="$(basename "${QUALIFICATION_ROOT}")"
  [[ "${qualification_basename}" =~ attempt([0-9]{3}) ]] || \
    fail "qualification root basename must include attemptNNN for retained evidence"
  attempt_number="${BASH_REMATCH[1]}"
  (( 10#${attempt_number} >= 1 && 10#${attempt_number} <= 3 )) || \
    fail "attempt ${attempt_number} is outside the registered v3 attempt001-attempt003 window"
  attempt_dir="outputs/egostitch_e2e_stage1_v3/qualification/attempt-${attempt_number}"
  cd "${QUALIFICATION_ROOT}"
  [[ ! -e "${attempt_dir}/${stage}" ]] || \
    fail "${stage} output already exists and cannot be replaced: ${attempt_dir}/${stage}"
  assert_qualification_boundary
  REGISTRATION_SHA256_BEFORE="$(registration_sha256)"
  export REGISTRATION_SHA256_BEFORE
  ATTEMPT_DIR="${attempt_dir}"
  ATTEMPT_NUMBER="${attempt_number}"
  export ATTEMPT_DIR ATTEMPT_NUMBER
}

run_sanity_suite() {
  cd "${REPO_ROOT}"
  "${UV_BIN}" run pytest -q \
    tests/test_train_egostitch.py \
    tests/test_train_egostitch_training.py \
    tests/test_train_egostitch_e2e.py \
    tests/model/test_egostitch_conditioning.py \
    tests/model/test_egostitch_trunk.py \
    tests/test_e2_pipeline.py \
    tests/experiments/test_prebinding_gates.py \
    tests/test_hpc_qualification.py
  cd "${QUALIFICATION_ROOT}"
  assert_registration_unchanged
}

# Produce the scope-appropriate probe artifact and evaluate the registered
# pre-binding gates against it. Both are pure post-processing over an already
# written checkpoint; neither opens a new universe.
evaluate_stage_gates() {
  local stage_dir="$1"
  local scope="$2"
  "${PYTHON_BIN}" -m src.experiments.probes produce-e2e \
    --checkpoint "${stage_dir}/best.pt" \
    --run-metadata "${stage_dir}/run_metadata.json" \
    --preregistration "${PREREGISTRATION}" \
    --data-root data \
    --strategy breadth_first \
    --output "${stage_dir}/probes/${scope}.npz" \
    --scope "${scope}"
  "${PYTHON_BIN}" -m src.experiments.prebinding_gates evaluate \
    --probe-artifact "${stage_dir}/probes/${scope}.npz" \
    --preregistration "${PREREGISTRATION}" \
    --config "${FULL_CONFIG}" \
    --scope "${scope}" \
    --output "${stage_dir}/prebinding_gates.json"
  assert_registration_unchanged
}

run_calibration() {
  enter_qualification_attempt calibration
  trap assert_registration_unchanged EXIT
  # Checked here, not only at rehearsal: the freeze manifest records HEAD, so a
  # dirty calibration whose edits are later reverted would present an identical,
  # now-clean HEAD and pass a rehearsal-only check.
  assert_clean_checkout calibration
  assert_source_resolves_to_repo

  echo "calibration stage 1/3: sanity"
  run_sanity_suite

  echo "calibration stage 2/3: registered 2,000-step overfit on V_fit"
  select_all_visible_h20s
  "${PYTHON_BIN}" -m src.e2_pipeline --config "${FULL_CONFIG}" \
    --worker-module src.train_egostitch \
    --run-kind overfit \
    --pack-dir outputs/feature_packs/egostitch_e2e_v_fit \
    --output-dir "${ATTEMPT_DIR}/calibration"
  assert_registration_unchanged

  echo "calibration stage 3/3: probes and pre-binding gates on V_fit"
  evaluate_stage_gates "${ATTEMPT_DIR}/calibration" calibration_fit
  write_calibration_freeze_manifest "${ATTEMPT_DIR}/calibration_freeze.json"
  assert_registration_unchanged
  echo "calibration completed; registration remains DRAFT"
  echo "measurements: ${ATTEMPT_DIR}/calibration/prebinding_gates.json"
}

run_rehearsal() {
  enter_qualification_attempt rehearsal
  # Every precondition — including the hardware preflight — is checked before
  # the ledger is claimed, so a rehearsal rejected for any reason does not burn
  # the one attempt. The claim itself stays immediately before the command that
  # opens V_qual, so an interrupted rehearsal still counts as spent.
  assert_clean_checkout rehearsal
  assert_source_resolves_to_repo
  assert_prebinding_gates_frozen
  assert_prebinding_gates_implementable
  assert_implementation_frozen "${ATTEMPT_DIR}/calibration_freeze.json"
  trap assert_registration_unchanged EXIT
  select_all_visible_h20s

  echo "rehearsal stage 1/3: exact full-arm rehearsal on V_qual"
  assert_single_v_qual_rehearsal "${ATTEMPT_NUMBER}"
  "${PYTHON_BIN}" -m src.e2_pipeline --config "${FULL_CONFIG}" \
    --worker-module src.train_egostitch \
    --run-kind rehearsal \
    --pack-dir outputs/feature_packs/egostitch_e2e_v_qual \
    --output-dir "${ATTEMPT_DIR}/rehearsal"
  assert_registration_unchanged

  echo "rehearsal stage 2/3: clip/family/RMS margins"
  "${PYTHON_BIN}" -c \
    'import sys; from pathlib import Path; from src.train_egostitch import validate_e2e_qualification_profile; validate_e2e_qualification_profile(Path(sys.argv[1]), output_path=Path(sys.argv[2]))' \
    "${ATTEMPT_DIR}/rehearsal/profile.json" \
    "${ATTEMPT_DIR}/qualification_margins.json"
  assert_registration_unchanged

  echo "rehearsal stage 3/3: probes and pre-binding gates on V_qual"
  evaluate_stage_gates "${ATTEMPT_DIR}/rehearsal" qualification_qual
  echo "rehearsal completed; registration remains DRAFT"
  echo "gate outcome: ${ATTEMPT_DIR}/rehearsal/prebinding_gates.json"
}

formal_config() {
  case "$1" in
    full) echo "configs/egostitch_e2e_v3_full_breadth_first.yaml" ;;
    f_only) echo "configs/egostitch_e2e_v3_f_only_breadth_first.yaml" ;;
    pair_topology) echo "configs/egostitch_e2e_v3_pair_topology_breadth_first.yaml" ;;
    p0) echo "configs/egostitch_e2e_v3_p0_breadth_first.yaml" ;;
    cosine_pool) echo "configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml" ;;
    no_l_rel) echo "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml" ;;
    structure_control_6a_v3|structure_control_6e_v1)
      fail "$1 is a scoring-time control that reuses the full arm's checkpoint; it is not trained" ;;
    *) fail "unknown formal arm: $1" ;;
  esac
}

assert_full_preflight() {
  local metadata="outputs/egostitch_e2e_stage1_v3/full/run_metadata.json"
  [[ -s "${metadata}" ]] || \
    fail "remaining arms require the completed full-arm preflight: ${metadata}"
  "${PYTHON_BIN}" -c \
    'import hashlib,json,sys; from pathlib import Path; m=json.loads(Path(sys.argv[1]).read_text()); r=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(); ok=m.get("status")=="complete" and m.get("run_kind")=="formal" and m.get("selected_checkpoint_eligible") is True and m.get("validation_liveness_pass") is True and m.get("preregistration_sha256")==r; raise SystemExit(0 if ok else 1)' \
    "${metadata}" "${PREREGISTRATION}" || \
    fail "full-arm run is not complete, eligible, live, and registration-matched"
}

run_formal() {
  local arm="$1"
  local config
  cd "${REPO_ROOT}"
  select_all_visible_h20s
  [[ "${DETECTED_GPU_COUNT}" -eq "${FORMAL_GPU_COUNT}" ]] || \
    fail "formal E2E training requires exactly ${FORMAL_GPU_COUNT} visible H20 GPUs, found ${DETECTED_GPU_COUNT}"
  assert_formal_registration
  if [[ "${arm}" != "full" ]]; then
    assert_full_preflight
  fi
  config="$(formal_config "${arm}")"
  exec "${PYTHON_BIN}" -m src.e2_pipeline --config "${config}" \
    --worker-module src.train_egostitch \
    --pack-dir outputs/feature_packs/egostitch_e2e_v_select \
    --run-kind formal
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

case "${1:-}" in
  calibrate)
    [[ $# -eq 1 ]] || fail "calibrate takes no arguments"
    run_calibration
    ;;
  rehearse)
    [[ $# -eq 1 ]] || fail "rehearse takes no arguments"
    run_rehearsal
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
