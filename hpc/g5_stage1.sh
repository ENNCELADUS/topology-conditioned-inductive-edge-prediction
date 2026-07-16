#!/usr/bin/env bash
# G5 Stage-1 outer orchestration: one fixed-seed engineering screening gate.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CONFIG="configs/egostitch_stage1_breadth_first.yaml"
readonly FORMAL_ROOT="outputs/egostitch_stage1"
readonly B0_UNIVERSE="outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz"
readonly B0CAL_RESULTS="outputs/b0_cal/b0cal_results.json"
readonly PREREGISTRATION="docs/registrations/g5_stage1_preregistration.json"
readonly F0_CACHE="outputs/feature_packs/egostitch_f0/f0_matrix.pt"
readonly GROUNDING_CACHE="${FORMAL_ROOT}/candidate_grounding.npz"
readonly COST_REPORT="${FORMAL_ROOT}/cost_report.json"

usage() {
  cat <<'EOF'
Usage:
  hpc/g5_stage1.sh seed <0|1|2>
  hpc/g5_stage1.sh formal

Each seed is completed as:
  train (or validate an existing complete run)
  -> candidate scoring

The formal command runs that sequence for fixed Seed 0, then opens the binding
single-seed Stage-1 screening gate. Formal evaluation additionally requires
seed0/fidelity.json and outputs/egostitch_stage1/cost_report.json.

This engineering screen uses deterministic point-estimate dominance and does not
claim statistical significance or cross-seed robustness. E1/E3 still require at
least three seeds and Holm-corrected inference.
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
    'import json,sys; from pathlib import Path; from src.score_universe import load_scores,validate_score_precision; a=load_scores(Path(sys.argv[1])); validate_score_precision(a.logit,meta=a.meta,label=sys.argv[1]); m=json.loads(Path(sys.argv[2]).read_text()); raise SystemExit(0 if a.meta.get("model_family") == "egostitch" and a.meta.get("checkpoint_id") == m.get("checkpoint_id") else 1)' \
    "${candidate}" "${metadata}"
}

run_seed() {
  local seed="$1"
  local seed_dir="${FORMAL_ROOT}/seed${seed}"
  local candidate="${seed_dir}/scores/candidate.npz"

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

  echo "seed ${seed}: training and candidate scoring complete"
}

run_formal() {
  run_seed 0

  [[ -s "${FORMAL_ROOT}/seed0/fidelity.json" ]] || \
    fail "formal gate requires ${FORMAL_ROOT}/seed0/fidelity.json"
  [[ -s "${COST_REPORT}" ]] || fail "formal gate requires ${COST_REPORT}"

  write_status single_seed_screening 0
  "${PYTHON_BIN}" -m src.experiments.g5_stage1 \
    --egostitch-universe \
      "${FORMAL_ROOT}/seed0/scores/candidate.npz" \
    --run-metadata \
      "${FORMAL_ROOT}/seed0/run_metadata.json" \
    --b0-universe "${B0_UNIVERSE}" \
    --b0cal-results "${B0CAL_RESULTS}" \
    --preregistration "${PREREGISTRATION}" \
    --data-root data \
    --strategy breadth_first \
    --fidelity-report \
      "${FORMAL_ROOT}/seed0/fidelity.json" \
    --cost-report "${COST_REPORT}" \
    --output-dir "${FORMAL_ROOT}/formal_gate"
  printf 'state=complete stage=single_seed_screening seed=0 commit=%s ended=%s\n' \
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
