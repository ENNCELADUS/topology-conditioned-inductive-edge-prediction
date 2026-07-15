#!/usr/bin/env bash
# G5 Stage-1 outer orchestration: complete each seed before starting the next.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CONFIG="configs/egostitch_stage1_breadth_first.yaml"
readonly FORMAL_ROOT="outputs/egostitch_stage1"
readonly B0_UNIVERSE="outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz"
readonly B0CAL_RESULTS="outputs/b0_cal/b0cal_results.json"
readonly PREREGISTRATION="docs/registrations/g5_stage1_preregistration.json"
readonly F0_CACHE="outputs/feature_packs/egostitch_f0/f0_matrix.pt"
readonly GROUNDING_CACHE="outputs/feature_packs/egostitch_f0/grounding.npz"
readonly COST_REPORT="${FORMAL_ROOT}/cost_report.json"

usage() {
  cat <<'EOF'
Usage:
  hpc/g5_stage1.sh seed <0|1|2>
  hpc/g5_stage1.sh formal

Each seed is completed as:
  train (or validate an existing complete run)
  -> candidate scoring
  -> explicitly non-binding single-seed topology diagnostic

The formal command runs that sequence for seeds 0, 1, and 2, then opens the
binding three-seed Holm gate. Formal evaluation additionally requires
seedN/fidelity.json for every seed and outputs/egostitch_stage1/cost_report.json.

A single-seed diagnostic is never a G5 pass/cut. If its inspection leads to a
model or hyperparameter change, this registered experiment must stop and a new
experiment ID and pre-registration must be created. With unchanged scientific
configuration, missing seeds may be completed under the existing registration.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_seed() {
  case "$1" in
    0|1|2) ;;
    *) fail "seed must be 0, 1, or 2; got $1" ;;
  esac
}

write_status() {
  local stage="$1"
  local seed="${2:-none}"
  mkdir -p "${FORMAL_ROOT}"
  printf 'state=running stage=%s seed=%s commit=%s updated=%s\n' \
    "${stage}" "${seed}" "$(git rev-parse HEAD)" "$(date -Iseconds)" \
    >"${FORMAL_ROOT}/pipeline.status"
}

candidate_is_current() {
  local candidate="$1"
  local metadata="$2"
  [[ -s "${candidate}" && -s "${metadata}" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import json,sys; from pathlib import Path; from src.score_universe import load_scores; a=load_scores(Path(sys.argv[1])); m=json.loads(Path(sys.argv[2]).read_text()); raise SystemExit(0 if a.meta.get("model_family") == "egostitch" and a.meta.get("checkpoint_id") == m.get("checkpoint_id") else 1)' \
    "${candidate}" "${metadata}"
}

run_seed() {
  local seed="$1"
  local seed_dir="${FORMAL_ROOT}/seed${seed}"
  local candidate="${seed_dir}/scores/candidate.npz"
  local diagnostic_dir="${seed_dir}/topology_diagnostic"

  validate_seed "${seed}"
  if [[ -s "${seed_dir}/complete.json" && -s "${seed_dir}/artifact_manifest.json" ]]; then
    echo "seed ${seed}: reusing atomically completed training artifact"
  else
    write_status train "${seed}"
    hpc/run.sh train "${CONFIG}" \
      --worker-module src.train_egostitch \
      --seed "${seed}" \
      --output-dir "${seed_dir}"
  fi

  if candidate_is_current "${candidate}" "${seed_dir}/run_metadata.json"; then
    echo "seed ${seed}: reusing candidate scores bound to the current checkpoint"
  else
    write_status candidate_score "${seed}"
    mkdir -p "$(dirname "${candidate}")"
    rm -f "${candidate}"
    hpc/run.sh score \
      --checkpoint "${seed_dir}/best.pt" \
      --pairs candidate \
      --data-root data \
      --strategy breadth_first \
      --b0-scores "${B0_UNIVERSE}" \
      --s0-checkpoint-id e092537d8cf1e208 \
      --f0-cache "${F0_CACHE}" \
      --grounding-cache "${GROUNDING_CACHE}" \
      --batch-pairs 1024 \
      --output "${candidate}"
  fi

  write_status topology_diagnostic "${seed}"
  "${PYTHON_BIN}" -m src.experiments.g5_stage1 \
    --egostitch-universe "${candidate}" \
    --run-metadata "${seed_dir}/run_metadata.json" \
    --b0-universe "${B0_UNIVERSE}" \
    --b0cal-results "${B0CAL_RESULTS}" \
    --preregistration "${PREREGISTRATION}" \
    --data-root data \
    --strategy breadth_first \
    --output-dir "${diagnostic_dir}"
  echo "seed ${seed}: non-binding topology diagnostic complete at ${diagnostic_dir}"
}

run_formal() {
  local seed
  for seed in 0 1 2; do
    run_seed "${seed}"
  done

  for seed in 0 1 2; do
    [[ -s "${FORMAL_ROOT}/seed${seed}/fidelity.json" ]] || \
      fail "formal gate requires ${FORMAL_ROOT}/seed${seed}/fidelity.json"
  done
  [[ -s "${COST_REPORT}" ]] || fail "formal gate requires ${COST_REPORT}"

  write_status formal_holm
  "${PYTHON_BIN}" -m src.experiments.g5_stage1 \
    --egostitch-universe \
      "${FORMAL_ROOT}/seed0/scores/candidate.npz" \
      "${FORMAL_ROOT}/seed1/scores/candidate.npz" \
      "${FORMAL_ROOT}/seed2/scores/candidate.npz" \
    --run-metadata \
      "${FORMAL_ROOT}/seed0/run_metadata.json" \
      "${FORMAL_ROOT}/seed1/run_metadata.json" \
      "${FORMAL_ROOT}/seed2/run_metadata.json" \
    --b0-universe "${B0_UNIVERSE}" \
    --b0cal-results "${B0CAL_RESULTS}" \
    --preregistration "${PREREGISTRATION}" \
    --data-root data \
    --strategy breadth_first \
    --fidelity-report \
      "${FORMAL_ROOT}/seed0/fidelity.json" \
      "${FORMAL_ROOT}/seed1/fidelity.json" \
      "${FORMAL_ROOT}/seed2/fidelity.json" \
    --cost-report "${COST_REPORT}" \
    --output-dir "${FORMAL_ROOT}/formal_gate"
  printf 'state=complete stage=formal_holm commit=%s ended=%s\n' \
    "$(git rev-parse HEAD)" "$(date -Iseconds)" >"${FORMAL_ROOT}/pipeline.status"
}

case "${1:-}" in
  help|--help|-h|"")
    usage
    exit 0
    ;;
esac

cd "${REPO_ROOT}" || fail "repository not found at ${REPO_ROOT}"

case "${1:-}" in
  seed)
    [[ $# -eq 2 ]] || fail "seed requires exactly one seed number"
    run_seed "$2"
    ;;
  formal)
    [[ $# -eq 1 ]] || fail "formal takes no arguments"
    run_formal
    ;;
  *)
    usage >&2
    fail "unknown command: $1"
    ;;
esac
