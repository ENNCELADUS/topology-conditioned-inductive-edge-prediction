#!/usr/bin/env bash
#SBATCH --job-name=tciep-preflight
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Node smoke test: verifies uv, the benchmark artifacts, the feature cache,
# and the non-integration test suite all work on this compute node. Submit
# via `slurm/submit.sh preflight` (account/partition/gres/time are passed as
# sbatch CLI flags there, never hardcoded here). A green preflight means the
# node can run train_b0.sbatch / score_universe.sbatch.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f slurm/cluster.env ]]; then
  echo "ERROR: slurm/cluster.env not found." >&2
  echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your cluster's values." >&2
  exit 1
fi

# shellcheck disable=SC1091
source slurm/cluster.env

if [[ -n "${CLUSTER_SETUP:-}" ]]; then
  eval "${CLUSTER_SETUP}"
fi

echo "== hostname =="
hostname

echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found on this node (no GPU visible?)." >&2
fi

echo "== uv --version =="
uv --version

echo "== uv sync --frozen =="
uv sync --frozen

export TCIEP_DATA_ROOT="${CLUSTER_DATA_ROOT}"

echo "== benchmark artifact verification (breadth_first) =="
uv run python -c "from pathlib import Path; from src.data.artifacts import verify_benchmark; print(verify_benchmark(Path('$CLUSTER_DATA_ROOT')/'benchmark_2025_neurips', 'breadth_first'))"

echo "== feature cache spot-check (node_000001 -> (123, 1536)) =="
uv run python -c "from pathlib import Path; from src.data.features import FeatureStore; store = FeatureStore(Path('$CLUSTER_DATA_ROOT') / 'features' / 'frozen_node_features_1024'); shape = tuple(store.load_tokens('node_000001').shape); print(shape); assert shape == (123, 1536), f'unexpected feature shape {shape}'"

echo "== uv run pytest -q -m \"not integration\" =="
uv run pytest -q -m "not integration"

echo "Preflight OK."
