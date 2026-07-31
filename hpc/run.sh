#!/usr/bin/env bash
# Direct runner for H20 container instances with automatic GPU discovery.
set -euo pipefail

readonly EXPECTED_REPO_ROOT="/2023533015/topology-conditioned-inductive-edge-prediction"
readonly UV_BIN="/2023533015/.uv/bin/uv"
readonly PYTHON_BIN="${EXPECTED_REPO_ROOT}/.venv/bin/python"
readonly DATA_ROOT="${EXPECTED_REPO_ROOT}/data"
readonly EXPECTED_GPU_NAME="NVIDIA H20"

usage() {
  cat <<'EOF'
Usage:
  hpc/run.sh check
  hpc/run.sh train <config.yaml> [train args...]
  hpc/run.sh score <score args...>
  hpc/run.sh merge <merge args...>
  hpc/run.sh g1 <g1 args...>
  hpc/run.sh g2 <g2 args...>

The train command is the only formal E2 entry: it runs the full packed-feature
DDP training pipeline (`python -m src.e2_pipeline`) across all visible NVIDIA
H20 GPUs via an automatically sized `accelerate launch`. Direct
`python -m src.train_b0 --max-steps N` remains debug-only (bounded smoke runs);
it is never a formal E2 training run. B0-alt keeps its own direct
`python -m src.train_b0` CLI, unaffected by this distributed routing.

EgoStitch E2E training is not launched from here: its plan-bound formal run goes
through the historically named `hpc/qualification.sh`, which verifies the
registration, configuration, implementation, and input-artifact identities.

The score command pins --device cuda --amp bf16. With multiple visible GPUs it
launches one contiguous shard per GPU, waits for every shard, and strictly merges
them into the requested output. merge/g1/g2 remain single-process while train uses
all visible NVIDIA H20 GPUs. Use nohup in the calling shell when a run must
survive disconnects.
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
    [[ "${gpu_name}" == "${EXPECTED_GPU_NAME}" ]] || \
      fail "expected all GPUs to be ${EXPECTED_GPU_NAME}, found ${gpu_name}"
  done
  GPU_COUNT="${#gpu_names[@]}"
  GPU_IDS="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  export GPU_COUNT GPU_IDS CUDA_VISIBLE_DEVICES="${GPU_IDS}"
}

parallel_score() {
  local -a score_args=("$@")
  local output=""
  local arg_index
  for ((arg_index = 0; arg_index < ${#score_args[@]}; arg_index++)); do
    case "${score_args[arg_index]}" in
      --output)
        ((arg_index + 1 < ${#score_args[@]})) || fail "--output requires a path"
        output="${score_args[arg_index + 1]}"
        ;;
      --shard|--num-shards)
        fail "hpc/run.sh score owns sharding; do not pass ${score_args[arg_index]}"
        ;;
    esac
  done
  [[ -n "${output}" ]] || fail "score requires --output"
  [[ "${output}" == *.npz ]] || fail "score output must end in .npz"

  if [[ "${GPU_COUNT}" -eq 1 ]]; then
    exec "${PYTHON_BIN}" -m src.score_universe score --device cuda --amp bf16 \
      "${score_args[@]}"
  fi

  local stem="${output%.npz}"
  local gpu
  local rc=0
  local -a pids=()
  local -a shard_inputs=()
  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    shard_inputs+=("${stem}.shard-${gpu}.npz")
    echo "launching score shard ${gpu}/${GPU_COUNT} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.score_universe score \
      --device cuda --amp bf16 "${score_args[@]}" --shard "${gpu}" \
      --num-shards "${GPU_COUNT}" >"${stem}.shard-${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  trap 'for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done' INT TERM
  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    if ! wait "${pids[gpu]}"; then
      echo "ERROR: score shard ${gpu}/${GPU_COUNT} failed; see ${stem}.shard-${gpu}.log" >&2
      rc=1
      break
    fi
  done
  if [[ "${rc}" -ne 0 ]]; then
    for pid in "${pids[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    wait || true
    return "${rc}"
  fi
  trap - INT TERM
  "${PYTHON_BIN}" -m src.score_universe merge --inputs "${shard_inputs[@]}" --output "${output}"
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
      'import os, torch; n=int(os.environ["GPU_COUNT"]); assert torch.cuda.device_count() == n; assert all(torch.cuda.get_device_name(i) == "NVIDIA H20" for i in range(n)); print(f"python/torch/cuda={torch.__version__}/{torch.version.cuda}; gpus={[torch.cuda.get_device_name(i) for i in range(n)]}")'
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
    # Stated in the usage text and enforced here: an EgoStitch E2E arm launched
    # from this branch would skip its plan/artifact identity preflight -- the
    # BINDING registration, clean checkout, registered config, and four-GPU pin.
    # The family is read from the config, so
    # naming the worker module by hand does not reopen the bypass.
    MODEL_FAMILY="$("${PYTHON_BIN}" -c \
      'import sys, yaml; from pathlib import Path; config = yaml.safe_load(Path(sys.argv[1]).read_text()); model = config.get("model") or {}; print(model.get("family", ""))' \
      "${CONFIG_PATH}")" || fail "could not read model.family from ${CONFIG_PATH}"
    [[ "${MODEL_FAMILY}" != "egostitch_e2e" ]] || \
      fail "EgoStitch E2E arms are launched only through hpc/qualification.sh"
    exec "${PYTHON_BIN}" -m src.e2_pipeline --config "${CONFIG_PATH}" "$@"
    ;;
  score)
    parallel_score "$@"
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
  *)
    usage >&2
    fail "unknown command: ${COMMAND}"
    ;;
esac
