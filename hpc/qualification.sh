#!/usr/bin/env bash
# Fail-closed launcher for the single-stage, plan-bound EgoStitch E2E experiment.
# The historical filename is retained to avoid breaking operator entry points.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PREREGISTRATION="docs/registrations/g5_e2e_stage1_preregistration_v5.json"
readonly FORMAL_GPU_COUNT=4
readonly EXPECTED_GPU_NAME="NVIDIA H20"
DETECTED_GPU_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  hpc/qualification.sh formal <full|f_only|pair_topology|p0|no_l_rel|row_layernorm>

formal executes one complete plan-bound training run. It requires exactly four
visible NVIDIA H20 GPUs, a clean checkout, and an unchanged registration snapshot. No prior
qualification artifact, quality verdict, checkpoint-eligibility result, liveness
predicate, or full-arm margin preflight is consulted. Model-quality predicates are
retained as telemetry only. Data-boundary, exact-coverage, non-finite, DDP, input,
I/O, registration-identity, and artifact-provenance failures still fail closed.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

registration_sha256() {
  sha256sum "${PREREGISTRATION}" | cut -d' ' -f1
}

select_all_visible_h20s() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
  [[ "${#gpu_rows[@]}" -ge 1 ]] || fail "no visible NVIDIA GPUs detected"
  local gpu_row gpu_index gpu_name
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
}

assert_source_resolves_to_repo() {
  local resolved
  resolved="$("${PYTHON_BIN}" -c 'import src, pathlib; print(pathlib.Path(src.__file__).resolve().parent.parent)')" || \
    fail "could not resolve the src package"
  [[ "${resolved}" == "${REPO_ROOT}" ]] || \
    fail "src resolves to ${resolved}, not ${REPO_ROOT}"
}

assert_clean_checkout() {
  local dirty
  dirty="$(git status --porcelain)"
  [[ -z "${dirty}" ]] || fail "formal requires a clean checkout"
}

assert_registration_unchanged() {
  [[ "$(registration_sha256)" == "${REGISTRATION_SHA256_BEFORE}" ]] || \
    fail "formal execution must not edit or replace its registered plan"
}

arm_config() {
  case "$1" in
    full) echo "configs/egostitch_e2e_v3_full_breadth_first.yaml" ;;
    f_only) echo "configs/egostitch_e2e_v3_f_only_breadth_first.yaml" ;;
    pair_topology) echo "configs/egostitch_e2e_v3_pair_topology_breadth_first.yaml" ;;
    p0) echo "configs/egostitch_e2e_v3_p0_breadth_first.yaml" ;;
    no_l_rel) echo "configs/egostitch_e2e_v3_no_l_rel_breadth_first.yaml" ;;
    row_layernorm) echo "configs/egostitch_e2e_v3_row_layernorm_breadth_first.yaml" ;;
    structure_control_6a_v3|structure_control_6e_v1)
      fail "$1 is a scoring-time control that reuses the full checkpoint" ;;
    *) fail "unknown arm: $1" ;;
  esac
}

run_formal() {
  local arm="$1"
  local config
  config="$(arm_config "${arm}")"
  select_all_visible_h20s
  [[ "${DETECTED_GPU_COUNT}" -eq "${FORMAL_GPU_COUNT}" ]] || \
    fail "formal E2E training requires exactly ${FORMAL_GPU_COUNT} visible H20 GPUs, found ${DETECTED_GPU_COUNT}"
  assert_clean_checkout
  assert_source_resolves_to_repo
  REGISTRATION_SHA256_BEFORE="$(registration_sha256)"
  export REGISTRATION_SHA256_BEFORE
  trap assert_registration_unchanged EXIT

  "${PYTHON_BIN}" -m src.e2_pipeline \
    --config "${config}" \
    --worker-module src.train_egostitch \
    --run-kind formal
  assert_registration_unchanged
  echo "formal ${arm} completed"
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
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

case "${1:-}" in
  formal)
    [[ $# -eq 2 ]] || fail "formal requires exactly one arm"
    run_formal "$2"
    ;;
  *)
    usage >&2
    fail "unknown command: $1"
    ;;
esac
