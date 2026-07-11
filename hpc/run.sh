#!/usr/bin/env bash
# Direct runner for the pinned single-H20 container instance.
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
  hpc/run.sh metrics <metrics args...>
  hpc/run.sh merge <merge args...>
  hpc/run.sh g1 <g1 args...>
  hpc/run.sh g2 <g2 args...>
  hpc/run.sh test-b0-v31

The score command pins --device cuda --amp bf16. All commands run directly in
the foreground on the single visible NVIDIA H20; use nohup in the calling shell
when a run must survive disconnects.
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
  [[ "${#gpu_names[@]}" -eq 1 ]] || fail "expected exactly 1 visible GPU, found ${#gpu_names[@]}"
  [[ "${gpu_names[0]}" == "${EXPECTED_GPU_NAME}" ]] || \
    fail "expected ${EXPECTED_GPU_NAME}, found ${gpu_names[0]}"
}

export CUDA_VISIBLE_DEVICES=0
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
      'import torch; assert torch.cuda.device_count() == 1; assert torch.cuda.get_device_name(0) == "NVIDIA H20"; print(f"python/torch/cuda={torch.__version__}/{torch.version.cuda}; gpu={torch.cuda.get_device_name(0)}")'
    "${PYTHON_BIN}" -c \
      'from pathlib import Path; from src.data.artifacts import verify_benchmark; print(verify_benchmark(Path("data/benchmark_2025_neurips"), "breadth_first"))'
    "${PYTHON_BIN}" -c \
      'from pathlib import Path; from src.data.features import FeatureStore; shape = tuple(FeatureStore(Path("data/features/frozen_node_features_1024")).load_tokens("node_000001").shape); print(f"feature_shape={shape}"); assert shape == (123, 1536)'
    "${PYTHON_BIN}" -m pytest -q -m "not integration"
    ;;
  train)
    [[ $# -ge 1 ]] || fail "train requires a config path"
    CONFIG_PATH="$1"
    shift
    [[ -f "${CONFIG_PATH}" ]] || fail "config not found: ${CONFIG_PATH}"
    exec "${PYTHON_BIN}" -m src.train_b0 --config "${CONFIG_PATH}" "$@"
    ;;
  score)
    exec "${PYTHON_BIN}" -m src.score_universe score --device cuda --amp bf16 "$@"
    ;;
  metrics)
    exec "${PYTHON_BIN}" -m src.score_universe metrics "$@"
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
  test-b0-v31)
    [[ $# -eq 0 ]] || fail "test-b0-v31 takes no arguments"
    CHECKPOINT="outputs/b0_v31/best.pt"
    [[ -f "${CHECKPOINT}" ]] || fail "checkpoint not found: ${CHECKPOINT}"

    "${PYTHON_BIN}" -m src.score_universe score --device cuda --amp bf16 \
      --checkpoint "${CHECKPOINT}" \
      --pairs test --data-root data --strategy breadth_first \
      --output scores/b0_v31_test.npz
    "${PYTHON_BIN}" -m src.score_universe metrics \
      --input scores/b0_v31_test.npz \
      --output outputs/b0_v31/test_metrics.json
    "${PYTHON_BIN}" -m src.score_universe score --device cuda --amp bf16 \
      --checkpoint "${CHECKPOINT}" \
      --pairs candidate --data-root data --strategy breadth_first \
      --output scores/b0_v31_candidate.npz
    "${PYTHON_BIN}" -m src.experiments.g1_hardened_e2 \
      --universe scores/b0_v31_candidate.npz \
      --data-root data --strategy breadth_first --output-dir outputs/g1
    "${PYTHON_BIN}" -m src.experiments.g2_ceiling \
      --universe scores/b0_v31_candidate.npz \
      --data-root data --strategy breadth_first --output-dir outputs/g2 \
      --figure outputs/g2/g2_ceiling.html
    ;;
  *)
    usage >&2
    fail "unknown command: ${COMMAND}"
    ;;
esac
