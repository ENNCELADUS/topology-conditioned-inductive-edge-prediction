#!/usr/bin/env bash
#SBATCH --job-name=tciep-score
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Passthrough to `uv run python -m src.score_universe <args...>` (the
# subcommand, e.g. `score` or `merge`, is part of <args...>). Submit via
# `slurm/submit.sh score [--array M-N] -- <args...>` (account/partition/gres
# /time are passed as sbatch CLI flags there, never hardcoded here). When run
# as a Slurm array task, appends `--shard $SLURM_ARRAY_TASK_ID --num-shards
# $SLURM_ARRAY_TASK_COUNT` to the forwarded args.
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
  echo "Usage: sbatch [sbatch flags] score_universe.sbatch <score|merge> [CLI args...]" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EXTRA_ARGS=(--shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}")
fi

echo "== score_universe: args=$* extra_args=${EXTRA_ARGS[*]:-} =="
uv run python -m src.score_universe "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
