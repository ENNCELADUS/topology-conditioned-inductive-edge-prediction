#!/usr/bin/env bash
# Direct runner for H20 container instances with automatic GPU discovery.
set -euo pipefail

# Resolve this checkout, including isolated Git worktrees used for measured rollouts.
readonly EXPECTED_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PYTHON_BIN="${EXPECTED_REPO_ROOT}/.venv/bin/python"
readonly DATA_ROOT="${EXPECTED_REPO_ROOT}/data"
readonly EXPECTED_GPU_NAMES="NVIDIA H20 NVIDIA H20-3e"

usage() {
  cat <<'EOF'
Usage:
  hpc/run.sh check
  hpc/run.sh train <config.yaml> [train args...]
  hpc/run.sh score <score args...>
  hpc/run.sh test <test args...>
  hpc/run.sh merge <merge args...>
  hpc/run.sh g1 <g1 args...>
  hpc/run.sh g2 <g2 args...>
  hpc/run.sh kd-targets <kd-targets args...>
  hpc/run.sh dictionary-build <dictionary builder args...>
  hpc/run.sh dictionary-diagnostic <dictionary oracle args...>
  hpc/run.sh motif-shift-diagnostic <v17 Step 0 args...>
  hpc/run.sh motif-crossfit <v17 cache preparation args...>

The train command drives the full packed-feature DDP training pipeline
(`python -m src.e2_pipeline`) across all visible NVIDIA H20 GPUs via an
automatically sized `accelerate launch`. It defaults to the B0 worker
(`src.train_b0`); pass `--worker-module src.train_egostitch --run-kind formal`
after the config path to train a formal EgoStitch E2E config, or use
`--run-kind diagnostic` only for a config explicitly consuming held-out truth
(`model.family: egostitch_e2e`) instead. Direct `python -m src.train_b0
--max-steps N` remains debug-only (bounded smoke runs); it is never a formal E2
training run. `src.e2_pipeline` runs four stages, `pack -> train -> publish ->
test`: the test stage selects on val_topology and val_cls, then scores held-out test/test_topology
and writes `test_report.json` immediately after publish, so scoring
is never a separate manual follow-up command; `--max-steps` debug runs skip it.
Only `egostitch_e2e` is ledgered against repeat scoring — re-running an
already-scored (arm, seed) fails unless `--rescore-reason <reason>` is passed
through after the config path. The external CAZI-MBN reproduction is not an E2
packed-feature worker; select its isolated runner with `--worker-module
src.train_cazi_mbn`, which this branch runs to completion and then chains the
same held-out test protocol against the checkpoint it publishes.
L3-PPI uses `--worker-module src.train_l3ppi [--stage surrogate|trial]
[--resume] [--skip-test]`. Surrogate pretraining never runs held-out testing;
trial training chains the common test protocol unless --skip-test is supplied.

The score command is a thin passthrough to `python -m src.score_fanout`, which
auto-detects GPU count, pins --device cuda --amp bf16, launches one contiguous
shard per visible GPU, waits for every shard, and strictly merges them into the
requested output; this runner does not duplicate that sharding or validation.
The test command is a thin passthrough to `python -m src.eval.test_protocol`
for one checkpoint: val_topology/val_cls threshold selection (topology cascade, max F1) -> test
edge metrics at the frozen max-F1 threshold -> test_topology at the frozen topology threshold -> `test_report.json`.
`train` already runs this automatically
for every trained arm; call `test` directly only for the two scoring-time
controls (`structure_control_6a_v3`, `structure_control_6e_v1`), which reuse
the `full` arm's checkpoint and have no pipeline run of their own.

The kd-targets command is a thin passthrough to `python -m
src.distill.teacher_targets`. By default it dumps training-corpus row
targets; pass `--contexts` to dump the shared epoch-indexed KD2 context banks.
Both modes take one published `full_ego_oracle` checkpoint and a training
`--config`. Use a fresh unique `--output` for every dump so stale shards cannot
be mixed.

  CUDA_VISIBLE_DEVICES=0 hpc/run.sh kd-targets --contexts --device cuda --config CONFIG --checkpoint CHECKPOINT --output UNIQUE_OUTPUT --row-shard 0/4
  CUDA_VISIBLE_DEVICES=1 hpc/run.sh kd-targets --contexts --device cuda --config CONFIG --checkpoint CHECKPOINT --output UNIQUE_OUTPUT --row-shard 1/4
  CUDA_VISIBLE_DEVICES=2 hpc/run.sh kd-targets --contexts --device cuda --config CONFIG --checkpoint CHECKPOINT --output UNIQUE_OUTPUT --row-shard 2/4
  CUDA_VISIBLE_DEVICES=3 hpc/run.sh kd-targets --contexts --device cuda --config CONFIG --checkpoint CHECKPOINT --output UNIQUE_OUTPUT --row-shard 3/4
  hpc/run.sh kd-targets --contexts --config CONFIG --checkpoint CHECKPOINT --output UNIQUE_OUTPUT --merge --row-shard 0/4

dictionary-build prepares the shared training-only motif dictionary; its encoder
uses the visible GPUs automatically. dictionary-diagnostic orchestrates the V_val
oracle comparison through score fan-out and never invokes held-out testing.

merge/g1/g2/kd-targets remain single-process while train uses all visible
NVIDIA H20 GPUs. Use nohup in the calling shell when a run must survive
disconnects.
EOF
}

COMMAND="${1:-}"
if [[ -z "${COMMAND}" || "${COMMAND}" == "help" || "${COMMAND}" == "--help" ]]; then
  usage
  exit 0
fi
shift

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_runtime() {
  [[ -d "${EXPECTED_REPO_ROOT}" ]] || fail "repository not found at ${EXPECTED_REPO_ROOT}"
  [[ -x "${UV_BIN}" ]] || fail "uv not found at ${UV_BIN}"
  [[ -x "${PYTHON_BIN}" ]] || fail "Python environment not found at ${PYTHON_BIN}"
  [[ -d "${DATA_ROOT}/benchmark_2025_neurips" ]] || fail "benchmark data is missing"
  [[ -d "${DATA_ROOT}/features/frozen_node_features_1024" ]] || fail "feature cache is missing"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"

  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  [[ "${#gpu_names[@]}" -ge 1 ]] || fail "expected at least one visible GPU"
  for gpu_name in "${gpu_names[@]}"; do
    [[ " ${EXPECTED_GPU_NAMES} " == *" ${gpu_name} "* ]] || \
      fail "expected all GPUs to be one of ${EXPECTED_GPU_NAMES}, found ${gpu_name}"
  done
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a lane_ids <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT="${#lane_ids[@]}"
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
  else
    GPU_COUNT="${#gpu_names[@]}"
    GPU_IDS="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  fi
  export GPU_COUNT GPU_IDS CUDA_VISIBLE_DEVICES="${GPU_IDS}"
}

export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="/2023533015/.uv/cache"
# Three containers share the host's 224 cores; uncapped torch/BLAS threads
# spin-wait against each other whenever two jobs overlap. Callers may override.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

assert_runtime
cd "${EXPECTED_REPO_ROOT}"

case "${COMMAND}" in
  check)
    [[ $# -eq 0 ]] || fail "check takes no arguments"
    "${UV_BIN}" --version
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    "${PYTHON_BIN}" -c \
      'import os, torch; n=int(os.environ["GPU_COUNT"]); allowed={"NVIDIA H20", "NVIDIA H20-3e"}; assert torch.cuda.device_count() == n; assert all(torch.cuda.get_device_name(i) in allowed for i in range(n)); print(f"python/torch/cuda={torch.__version__}/{torch.version.cuda}; gpus={[torch.cuda.get_device_name(i) for i in range(n)]}")'
    "${PYTHON_BIN}" -c \
      'from pathlib import Path; from src.data.artifacts import verify_benchmark; print(verify_benchmark(Path("data/benchmark_2025_neurips"), "breadth_first"))'
    "${PYTHON_BIN}" -c \
      'from pathlib import Path; from src.data.features import FeatureStore; shape = tuple(FeatureStore(Path("data/features/frozen_node_features_1024")).load_tokens("node_000001").shape); print(f"feature_shape={shape}"); assert shape == (123, 1536)'
    ;;
  train)
    [[ $# -ge 1 ]] || fail "train requires a config path"
    CONFIG_PATH="$1"
    shift
    [[ -f "${CONFIG_PATH}" ]] || fail "config not found: ${CONFIG_PATH}"
    L3_WORKER=false
    PREVIOUS_ARG=
    for TRAIN_ARG in "$@"; do
      if [[ "${PREVIOUS_ARG}" == --worker-module && "${TRAIN_ARG}" == src.train_l3ppi ]]; then
        L3_WORKER=true
      fi
      PREVIOUS_ARG="${TRAIN_ARG}"
    done
    if [[ "${L3_WORKER}" == true ]]; then
      L3_STAGE=trial
      L3_SKIP_TEST=false
      L3_ARGS=("${CONFIG_PATH}")
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --worker-module)
            [[ $# -ge 2 && "$2" == src.train_l3ppi ]] || fail "invalid L3-PPI worker"
            shift 2
            ;;
          --skip-test) L3_SKIP_TEST=true; shift ;;
          --resume) L3_ARGS+=(--resume); shift ;;
          --stage)
            [[ $# -ge 2 ]] || fail "--stage requires surrogate or trial"
            [[ "$2" == surrogate || "$2" == trial ]] || fail "invalid L3-PPI stage: $2"
            L3_STAGE="$2"
            shift 2
            ;;
          *) fail "unsupported L3-PPI training argument: $1" ;;
        esac
      done
      "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}" \
        -m src.train_l3ppi "${L3_ARGS[@]}" --stage "${L3_STAGE}"
      if [[ "${L3_STAGE}" == surrogate || "${L3_SKIP_TEST}" == true ]]; then
        exit 0
      fi
      read -r L3_OUTPUT L3_PACK L3_SEED L3_DATA < <("${PYTHON_BIN}" -c '
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
print(cfg["output_dir"], cfg["pack_dir"], cfg["seed"], cfg["data_root"])
' "${CONFIG_PATH}")
      exec "${PYTHON_BIN}" -m src.eval.test_protocol \
        --checkpoint "${L3_OUTPUT}/best.pt" --output-dir "${L3_OUTPUT}" \
        --pack-dir "${L3_PACK}" --data-root "${L3_DATA}" --strategy breadth_first \
        --arm l3ppi --seed "${L3_SEED}"
    fi
    if [[ $# -eq 2 && "$1" == "--worker-module" && "$2" == "src.train_official_ppi" ]]; then
      "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}" \
        -m src.train_official_ppi "${CONFIG_PATH}"
      read -r PPI_OUTPUT PPI_PACK PPI_SEED PPI_ARM < <("${PYTHON_BIN}" -c '
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
print(cfg["output_dir"], cfg["pack_dir"], cfg["seed"], cfg["model"]["method"])
' "${CONFIG_PATH}")
      exec "${PYTHON_BIN}" -m src.eval.test_protocol \
        --checkpoint "${PPI_OUTPUT}/best.pt" --output-dir "${PPI_OUTPUT}" \
        --pack-dir "${PPI_PACK}" --data-root "${DATA_ROOT}" --strategy breadth_first \
        --arm "${PPI_ARM}" --seed "${PPI_SEED}"
    fi
    if [[ $# -eq 2 && "$1" == "--worker-module" && "$2" == "src.train_cazi_mbn" ]]; then
      # Not exec: the CAZI runner is not an E2 pack/train/publish/test worker,
      # so this branch chains the same held-out test protocol itself once
      # training publishes its checkpoint.
      # --stage train, not the default --stage all: `all` runs CAZI's own
      # score_and_evaluate, which reads the test pairs and the candidate
      # universe before the chained protocol scores it. The test protocol
      # below is the single owner of every
      # held-out read for this arm.
      "${PYTHON_BIN}" -m src.train_cazi_mbn "${CONFIG_PATH}" --device cuda --stage train
      read -r CAZI_OUTPUT_DIR CAZI_STRATEGY CAZI_SEED < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from src.train_cazi_mbn import load_config
cfg = load_config(Path(sys.argv[1]))
print(cfg.output_dir, cfg.strategy, cfg.seed)
' "${CONFIG_PATH}")
      exec "${PYTHON_BIN}" -m src.eval.test_protocol \
        --checkpoint "${CAZI_OUTPUT_DIR}/student.pt" \
        --model-family cazi_mbn \
        --model-config "${CONFIG_PATH}" \
        --output-dir "${CAZI_OUTPUT_DIR}" \
        --data-root "${DATA_ROOT}" \
        --strategy "${CAZI_STRATEGY}" \
        --arm cazi_mbn \
        --seed "${CAZI_SEED}"
    fi
    exec "${PYTHON_BIN}" -m src.e2_pipeline --config "${CONFIG_PATH}" "$@"
    ;;
  score)
    exec "${PYTHON_BIN}" -m src.score_fanout "$@"
    ;;
  test)
    exec "${PYTHON_BIN}" -m src.eval.test_protocol "$@"
    ;;
  merge)
    exec "${PYTHON_BIN}" -m src.score_universe merge "$@"
    ;;
  g1)
    exec "${PYTHON_BIN}" -m src.experiments.g1_hardened_e2 "$@"
    ;;
  g2)
    exec "${PYTHON_BIN}" -m src.experiments.g2_ceiling "$@"
    ;;
  kd-targets)
    exec "${PYTHON_BIN}" -m src.distill.teacher_targets "$@"
    ;;
  dictionary-build)
    exec "${PYTHON_BIN}" -m src.experiments.motif_dictionary_build "$@"
    ;;
  dictionary-diagnostic)
    exec "${PYTHON_BIN}" -m src.experiments.motif_dictionary_diagnostic "$@"
    ;;
  motif-shift-diagnostic)
    exec "${PYTHON_BIN}" -m src.experiments.motif_shift_diagnostic "$@"
    ;;
  motif-crossfit)
    exec "${PYTHON_BIN}" -m src.experiments.motif_crossfit "$@"
    ;;
  *)
    usage >&2
    fail "unknown command: ${COMMAND}"
    ;;
esac
