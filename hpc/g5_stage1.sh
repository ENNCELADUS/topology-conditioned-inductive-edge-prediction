#!/usr/bin/env bash
# G5 Stage-1 outer orchestration: one fixed-seed engineering screening gate.
set -euo pipefail

readonly REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CONFIG="configs/egostitch_stage1_breadth_first.yaml"
readonly FORMAL_ROOT="outputs/egostitch_stage1"
readonly B0_CHECKPOINT="outputs/deliverables/b0_v31_breadth_first_20260711/model/best.pt"
readonly B0_UNIVERSE="outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz"
readonly B0_PACK_DIR="outputs/feature_packs/b0_v31_bf16"
readonly B0CAL_RESULTS="outputs/b0_cal/b0cal_results.json"
readonly PREREGISTRATION="docs/registrations/g5_stage1_preregistration.json"
readonly F0_CACHE="outputs/feature_packs/egostitch_f0/f0_matrix.pt"
readonly GROUNDING_CACHE="${FORMAL_ROOT}/candidate_grounding.npz"
readonly COST_REPORT="${FORMAL_ROOT}/cost_report.json"
readonly S0_CANDIDATE="${FORMAL_ROOT}/s0_candidate_fp32.npz"

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
  [[ -s "${candidate}" && -s "${metadata}" && -s "${S0_CANDIDATE}" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import hashlib,json,sys; from pathlib import Path; from src.score_universe import load_scores,validate_score_precision; a=load_scores(Path(sys.argv[1])); validate_score_precision(a.logit,meta=a.meta,label=sys.argv[1]); m=json.loads(Path(sys.argv[2]).read_text()); h=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(); p=a.meta.get("score_profile"); ok=a.meta.get("model_family")=="egostitch" and a.meta.get("checkpoint_id")==m.get("checkpoint_id") and a.meta.get("s0_artifact_sha256")==h and isinstance(p,dict) and p.get("wall_seconds",0)>0; raise SystemExit(0 if ok else 1)' \
    "${candidate}" "${metadata}" "${S0_CANDIDATE}"
}

training_is_current() {
  local seed_dir="$1"
  local seed="$2"
  local expected_prereg
  expected_prereg="$(sha256sum "${PREREGISTRATION}" | cut -d' ' -f1)"
  [[ -s "${seed_dir}/complete.json" && -s "${seed_dir}/artifact_manifest.json" && \
     -s "${seed_dir}/run_metadata.json" && -s "${seed_dir}/best.pt" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import hashlib,json,sys,torch; from dataclasses import replace; from pathlib import Path; from src.train_egostitch import config_to_dict,load_config; m=json.loads(Path(sys.argv[1]).read_text()); payload=torch.load(sys.argv[2],map_location="cpu",weights_only=True); recorded=payload.get("config"); out=Path(str(recorded.get("output_dir",""))) if isinstance(recorded,dict) else Path(); current=config_to_dict(replace(load_config(Path(sys.argv[3])),seed=int(sys.argv[4]),output_dir=out)); digest=hashlib.sha256(json.dumps(recorded,sort_keys=True).encode()).hexdigest(); prefix=Path(sys.argv[5]); ok=m.get("status")=="complete" and m.get("preregistration_sha256")==sys.argv[6] and m.get("config_hash")==digest and recorded==current and out.parent==prefix and out.name.startswith(".e2-run-"); raise SystemExit(0 if ok else 1)' \
    "${seed_dir}/run_metadata.json" "${seed_dir}/best.pt" "${CONFIG}" "${seed}" \
    "${seed_dir}" "${expected_prereg}" || return 1
  # New runs record source_commit before training. It is descriptive rather
  # than a reuse key: a later orchestration-only fix must not invalidate a run
  # already bound to the exact registered config and preregistration.
  if [[ -s "${seed_dir}/source_commit" ]]; then
    [[ -n "$(<"${seed_dir}/source_commit")" ]]
  fi
}

s0_candidate_is_current() {
  [[ -s "${S0_CANDIDATE}" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import sys,numpy as np; from pathlib import Path; from src.score_universe import load_scores; a=load_scores(Path(sys.argv[1])); p=a.meta.get("score_precision",{}); ok=a.meta.get("model_family")=="v3_1" and a.meta.get("checkpoint_id")==sys.argv[2] and a.meta.get("pairs_source")=="candidate" and a.meta.get("strategy")=="breadth_first" and p.get("encode_autocast")=="bf16" and p.get("pair_autocast")=="off" and p.get("logit_storage_dtype")=="float32" and a.logit.dtype==np.float32; raise SystemExit(0 if ok else 1)' \
    "${S0_CANDIDATE}" e092537d8cf1e208
}

run_s0_candidate() {
  if s0_candidate_is_current; then
    echo "formal: reusing fresh-fp32 frozen-s0 candidate scores"
    return
  fi
  write_status s0_candidate_score 0
  rm -f "${S0_CANDIDATE}"
  hpc/run.sh score \
    --checkpoint "${B0_CHECKPOINT}" \
    --pairs candidate \
    --data-root data \
    --strategy breadth_first \
    --pack-dir "${B0_PACK_DIR}" \
    --token-budget 262144 \
    --amp bf16 \
    --pair-amp off \
    --output "${S0_CANDIDATE}"
  s0_candidate_is_current || fail "fresh-fp32 s0 candidate artifact failed validation"
}

run_diagnostics() {
  write_status diagnostics 0
  "${PYTHON_BIN}" -m src.experiments.g5_stage1_diagnostics \
    --checkpoint "${FORMAL_ROOT}/seed0/best.pt" \
    --last-checkpoint "${FORMAL_ROOT}/seed0/last.pt" \
    --candidate "${FORMAL_ROOT}/seed0/scores/candidate.npz" \
    --s0-universe "${S0_CANDIDATE}" \
    --run-metadata "${FORMAL_ROOT}/seed0/run_metadata.json" \
    --config "${CONFIG}" \
    --data-root data \
    --strategy breadth_first \
    --fidelity-output "${FORMAL_ROOT}/seed0/fidelity.json" \
    --cost-output "${COST_REPORT}"
}

run_seed() {
  local seed="$1"
  local seed_dir="${FORMAL_ROOT}/seed${seed}"
  local candidate="${seed_dir}/scores/candidate.npz"

  validate_seed "${seed}"
  if training_is_current "${seed_dir}" "${seed}"; then
    echo "seed ${seed}: reusing atomically completed training artifact"
  else
    [[ ! -e "${seed_dir}/run_metadata.json" ]] || \
      fail "seed ${seed} artifacts exist but do not match the registered config; archive them explicitly before retraining"
    write_status train "${seed}"
    mkdir -p "${seed_dir}"
    printf '%s\n' "$(git rev-parse HEAD)" >"${seed_dir}/source_commit"
    hpc/run.sh train "${CONFIG}" \
      --worker-module src.train_egostitch \
      --seed "${seed}" \
      --output-dir "${seed_dir}"
  fi

  run_s0_candidate

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
      --b0-scores "${S0_CANDIDATE}" \
      --s0-checkpoint-id e092537d8cf1e208 \
      --f0-cache "${F0_CACHE}" \
      --grounding-cache "${GROUNDING_CACHE}" \
      --batch-pairs 1024 \
      --output "${candidate}"
  fi

  printf 'state=complete stage=candidate_score seed=%s commit=%s ended=%s\n' \
    "${seed}" "$(git rev-parse HEAD)" "$(date -Iseconds)" >"${FORMAL_ROOT}/pipeline.status"
  echo "seed ${seed}: training and candidate scoring complete"
}

run_formal() {
  run_seed 0
  run_diagnostics

  write_status single_seed_screening 0
  "${PYTHON_BIN}" -m src.experiments.g5_stage1 \
    --egostitch-universe \
      "${FORMAL_ROOT}/seed0/scores/candidate.npz" \
    --run-metadata \
      "${FORMAL_ROOT}/seed0/run_metadata.json" \
    --s0-universe "${S0_CANDIDATE}" \
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
