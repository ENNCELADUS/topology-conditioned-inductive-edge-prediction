#!/usr/bin/env bash
# Fail-closed launcher for the §13.19 qualification and bound formal contexts.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PREREGISTRATION="docs/registrations/g5_e2e_stage1_preregistration_v2.json"
readonly FULL_CONFIG="configs/egostitch_e2e_breadth_first.yaml"
readonly FORMAL_GPU_COUNT=4
readonly EXPECTED_GPU_NAME="NVIDIA H20"
readonly QUALIFICATION_ROOT="${EGOSTITCH_QUALIFICATION_ROOT:-}"
DETECTED_GPU_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  EGOSTITCH_QUALIFICATION_ROOT=/path/to/safe-root hpc/qualification.sh qualify
  hpc/qualification.sh formal <full|f_only|pair_topology|p0>

qualify is train/validation-only. The registered 2,000-step overfit and full-arm
rehearsal auto-detect and use every visible H20, matching the formal launch style.
It runs sanity -> overfit -> rehearsal and stops at the first failure.
The DRAFT registration is never edited or promoted.

formal requires exactly 4 visible NVIDIA H20s and a fully resolved BINDING
registration. It launches one registered arm; scientific execution order remains
full first, with the full-arm eligibility/liveness preflight required before the
remaining arms are launched.

Both qualification stages execute the registered v2 trainer. --max-steps is
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

run_qualification() {
  [[ -n "${QUALIFICATION_ROOT}" && -d "${QUALIFICATION_ROOT}" ]] || \
    fail "qualify requires EGOSTITCH_QUALIFICATION_ROOT to name the isolated data root"
  local qualification_basename
  local attempt_number
  local attempt_dir
  qualification_basename="$(basename "${QUALIFICATION_ROOT}")"
  [[ "${qualification_basename}" =~ attempt([0-9]{3}) ]] || \
    fail "qualification root basename must include attemptNNN for retained evidence"
  attempt_number="${BASH_REMATCH[1]}"
  (( 10#${attempt_number} >= 3 && 10#${attempt_number} <= 5 )) || \
    fail "attempt ${attempt_number} is outside the registered v2 attempt003-attempt005 window"
  attempt_dir="outputs/egostitch_e2e_stage1/qualification/attempt-${attempt_number}"
  cd "${QUALIFICATION_ROOT}"
  [[ ! -e "${attempt_dir}" ]] || \
    fail "qualification attempt output already exists and cannot be replaced: ${attempt_dir}"
  assert_qualification_boundary
  REGISTRATION_SHA256_BEFORE="$(registration_sha256)"
  export REGISTRATION_SHA256_BEFORE
  trap assert_registration_unchanged EXIT

  echo "qualification stage 1/3: sanity"
  cd "${REPO_ROOT}"
  "${UV_BIN}" run pytest -q \
    tests/test_train_egostitch.py \
    tests/test_train_egostitch_training.py \
    tests/test_train_egostitch_e2e.py \
    tests/model/test_egostitch_conditioning.py \
    tests/model/test_egostitch_trunk.py \
    tests/test_e2_pipeline.py \
    tests/test_hpc_qualification.py
  cd "${QUALIFICATION_ROOT}"
  assert_registration_unchanged

  echo "qualification stage 2/3: registered 2,000-step overfit"
  select_all_visible_h20s
  "${PYTHON_BIN}" -m src.e2_pipeline --config "${FULL_CONFIG}" \
    --worker-module src.train_egostitch \
    --run-kind overfit \
    --pack-dir outputs/feature_packs/egostitch_e2e_v_fit \
    --output-dir "${attempt_dir}/overfit"
  assert_registration_unchanged

  echo "qualification stage 3/3: exact full-arm rehearsal"
  select_all_visible_h20s
  "${PYTHON_BIN}" -m src.e2_pipeline --config "${FULL_CONFIG}" \
    --worker-module src.train_egostitch \
    --run-kind rehearsal \
    --pack-dir outputs/feature_packs/egostitch_e2e_v_qual \
    --output-dir "${attempt_dir}/rehearsal"
  "${PYTHON_BIN}" -c \
    'import sys; from pathlib import Path; from src.train_egostitch import validate_e2e_qualification_profile; validate_e2e_qualification_profile(Path(sys.argv[1]), output_path=Path(sys.argv[2]))' \
    "${attempt_dir}/rehearsal/profile.json" \
    "${attempt_dir}/qualification_margins.json"
  assert_registration_unchanged
  echo "qualification completed; registration remains DRAFT"
}

formal_config() {
  case "$1" in
    full) echo "configs/egostitch_e2e_breadth_first.yaml" ;;
    f_only) echo "configs/egostitch_e2e_f_only_breadth_first.yaml" ;;
    pair_topology) echo "configs/egostitch_e2e_pair_topology_breadth_first.yaml" ;;
    p0) echo "configs/egostitch_e2e_p0_breadth_first.yaml" ;;
    *) fail "unknown formal arm: $1" ;;
  esac
}

assert_full_preflight() {
  local metadata="outputs/egostitch_e2e_stage1/full/run_metadata.json"
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
  qualify)
    [[ $# -eq 1 ]] || fail "qualify takes no arguments"
    run_qualification
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
