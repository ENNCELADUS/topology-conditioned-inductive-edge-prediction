#!/usr/bin/env bash
# Submit a job to the Slurm cluster. Run this ON the cluster login node from
# $CLUSTER_REPO_DIR (after slurm/sync_code.sh has copied the repo there and
# slurm/cluster.env has been filled in).
#
# Usage:
#   slurm/submit.sh preflight
#   slurm/submit.sh train configs/<name>.yaml [extra train CLI args...]
#   slurm/submit.sh score [--array M-N] -- <score|merge> [CLI args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/cluster.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your cluster's values." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

usage() {
  echo "Usage: $0 <preflight|train|score> [extra args...]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

TARGET="$1"
shift

SBATCH_ARRAY_ARGS=()

case "${TARGET}" in
  preflight)
    SBATCH_SCRIPT="${SCRIPT_DIR}/preflight.sbatch"
    JOB_TIME="${CLUSTER_TIME_TRAIN}"
    ;;
  train)
    SBATCH_SCRIPT="${SCRIPT_DIR}/train_b0.sbatch"
    JOB_TIME="${CLUSTER_TIME_TRAIN}"
    ;;
  score)
    SBATCH_SCRIPT="${SCRIPT_DIR}/score_universe.sbatch"
    JOB_TIME="${CLUSTER_TIME_SCORE}"
    if [[ "${1:-}" == "--array" ]]; then
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --array requires a range, e.g. --array 0-3" >&2
        exit 1
      fi
      ARRAY_RANGE="$2"
      shift 2
      SBATCH_ARRAY_ARGS=(--array "${ARRAY_RANGE}")
      if [[ "${ARRAY_RANGE}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        LO="${BASH_REMATCH[1]}"
        HI="${BASH_REMATCH[2]}"
        COUNT=$((HI - LO + 1))
        echo "Array ${ARRAY_RANGE} -> ${COUNT} shard(s):" \
          "SLURM_ARRAY_TASK_ID ${LO}..${HI} become --shard <id> --num-shards ${COUNT}."
      else
        echo "Array ${ARRAY_RANGE} -> each SLURM_ARRAY_TASK_ID becomes" \
          "--shard <id> --num-shards \$SLURM_ARRAY_TASK_COUNT."
      fi
    fi
    ;;
  *)
    usage
    ;;
esac

if [[ "${1:-}" == "--" ]]; then
  shift
fi

mkdir -p "${SCRIPT_DIR}/logs"

sbatch \
  -A "${CLUSTER_ACCOUNT}" \
  -p "${CLUSTER_PARTITION}" \
  --gres "${CLUSTER_GRES}" \
  --time "${JOB_TIME}" \
  ${SBATCH_ARRAY_ARGS[@]+"${SBATCH_ARRAY_ARGS[@]}"} \
  "${SBATCH_SCRIPT}" "$@"
