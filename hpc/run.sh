#!/usr/bin/env bash
# Direct runner for H20 container instances with automatic GPU discovery.
set -euo pipefail

readonly EXPECTED_REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
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
  hpc/run.sh recover-v-hold <recovery args...>
  hpc/run.sh test <test args...>
  hpc/run.sh merge <merge args...>
  hpc/run.sh g1 <g1 args...>
  hpc/run.sh g2 <g2 args...>
  hpc/run.sh kd-targets <kd-targets args...>

The train command drives the full packed-feature DDP training pipeline
(`python -m src.e2_pipeline`) across all visible NVIDIA H20 GPUs via an
automatically sized `accelerate launch`. It defaults to the B0 worker
(`src.train_b0`); pass `--worker-module src.train_egostitch --run-kind formal`
after the config path to train a formal EgoStitch E2E config, or use
`--run-kind diagnostic` only for a config explicitly consuming held-out truth
(`model.family: egostitch_e2e`) instead. Direct `python -m src.train_b0
--max-steps N` remains debug-only (bounded smoke runs); it is never a formal E2
training run. `src.e2_pipeline` runs four stages, `pack -> train -> publish ->
test`: the test stage scores this checkpoint's held-out test/candidate
universes and writes `test_report.json` immediately after publish, so scoring
is never a separate manual follow-up command; `--max-steps` debug runs skip it.
Only `egostitch_e2e` is ledgered against repeat scoring — re-running an
already-scored (arm, seed) fails unless `--rescore-reason <reason>` is passed
through after the config path. The external CAZI-MBN reproduction is not an E2
packed-feature worker; select its isolated runner with `--worker-module
src.train_cazi_mbn`, which this branch runs to completion and then chains the
same held-out test protocol against the checkpoint it publishes.

The score command is a thin passthrough to `python -m src.score_fanout`, which
auto-detects GPU count, pins --device cuda --amp bf16, launches one contiguous
shard per visible GPU, waits for every shard, and strictly merges them into the
requested output; this runner does not duplicate that sharding or validation.
Scoring has no wall-clock deadline: a pass's cost is set by its pairs universe
(the candidate universe is ~2.0M pairs, ~32x the test universe), so a fixed
budget only discards finished work. Pass --timeout-seconds to impose one.

The test command is a thin passthrough to `python -m src.eval.test_protocol`
for one published checkpoint: score test -> fixed-0.5 edge metrics -> score
candidate -> global density-matched graph metrics -> `test_report.json`.
`train` already runs this automatically
for every trained arm; call `test` directly only for the two scoring-time
controls (`structure_control_6a_v3`, `structure_control_6e_v1`), which reuse
the `full` arm's checkpoint and have no pipeline run of their own.

The kd-targets command is a thin passthrough to `python -m
src.distill.teacher_targets` (B1 plan), which dumps the Full-Ego Pooled Oracle
KD teacher-target artifact over V_fit-only anchor/context pairs from one
published `full_ego_oracle` checkpoint. It scores on a single GPU -- there is
no world-size fan-out to size, unlike `train`/`score` -- so pass `--device
cuda` explicitly; `--data-root`/`--strategy` default to this repository's data
root and `breadth_first` but may be overridden after the command's own args.

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
  GPU_COUNT="${#gpu_names[@]}"
  GPU_IDS="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  export GPU_COUNT GPU_IDS CUDA_VISIBLE_DEVICES="${GPU_IDS}"
}

export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="/2023533015/.uv/cache"

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
    "${PYTHON_BIN}" -m pytest -q -m "not integration"
    "${PYTHON_BIN}" -m pytest -q tests/test_e2_ddp_integration.py -m "integration and not slow"
    ;;
  train)
    [[ $# -ge 1 ]] || fail "train requires a config path"
    CONFIG_PATH="$1"
    shift
    [[ -f "${CONFIG_PATH}" ]] || fail "config not found: ${CONFIG_PATH}"
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
  recover-v-hold)
    exec "${PYTHON_BIN}" -m accelerate.commands.launch \
      --num_processes "${GPU_COUNT}" --mixed_precision bf16 \
      -m src.experiments.v_hold_checkpoint_recovery "$@"
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
    exec "${PYTHON_BIN}" -m src.distill.teacher_targets \
      --data-root "${DATA_ROOT}" --strategy breadth_first "$@"
    ;;
  *)
    usage >&2
    fail "unknown command: ${COMMAND}"
    ;;
esac
