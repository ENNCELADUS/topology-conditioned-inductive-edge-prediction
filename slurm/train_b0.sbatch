#!/usr/bin/env bash
#SBATCH --job-name=tciep-train-b0
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Runs `uv run python -m src.train_b0 --config <config-path> [extra args...]`.
# Submit via `slurm/submit.sh train configs/<name>.yaml [extra args...]`
# (account/partition/gres/time are passed as sbatch CLI flags there, never
# hardcoded here).
#
# The pinned train CLI reads its data root from the config file's own
# `data: root:` key, not from an env var -- so cluster configs under
# configs/ must point `data.root` at the cluster data root (or callers pass
# an override such as `--data-root $CLUSTER_DATA_ROOT` as an extra CLI arg,
# if/when the CLI supports it). This script forwards the config path and any
# extra CLI args through verbatim rather than guessing at flag names; it
# also exports TCIEP_DATA_ROOT in case the CLI or its config loader falls
# back to it.
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

export TCIEP_DATA_ROOT="${CLUSTER_DATA_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch [sbatch flags] train_b0.sbatch <config-path> [extra train CLI args...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift

echo "== training: config=${CONFIG_PATH} extra_args=$* =="
uv run python -m src.train_b0 --config "${CONFIG_PATH}" "$@"
