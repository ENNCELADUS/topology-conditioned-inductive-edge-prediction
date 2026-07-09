#!/usr/bin/env bash
# Run LOCALLY (not on the cluster). Rsyncs the repo to the cluster, excluding
# large/generated/local-tool directories -- the data package (incl. the 25 GB
# feature cache) is assumed to already be on cluster storage; only code syncs.
#
# Usage:
#   slurm/sync_code.sh                # sync code only
#   slurm/sync_code.sh --with-env     # also scp slurm/cluster.env across
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/cluster.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your cluster's values." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

WITH_ENV=0
if [[ "${1:-}" == "--with-env" ]]; then
  WITH_ENV=1
  shift
fi

EXCLUDE_PATTERNS=(
  ".git"
  ".venv"
  "data"
  "literature"
  "outputs"
  "scores"
  "slurm/logs"
  ".superpowers"
  ".claude"
  "__pycache__"
  ".mypy_cache"
  ".ruff_cache"
  ".pytest_cache"
  "slurm/cluster.env"
)

RSYNC_EXCLUDES=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  RSYNC_EXCLUDES+=(--exclude "${pattern}")
done

echo "Syncing ${REPO_ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" "${REPO_ROOT}/" "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"

if [[ "${WITH_ENV}" -eq 1 ]]; then
  echo "Copying cluster.env -> ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/slurm/cluster.env"
  ssh "${CLUSTER_HOST}" "mkdir -p ${CLUSTER_REPO_DIR}/slurm"
  scp "${ENV_FILE}" "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/slurm/cluster.env"
fi

echo "Sync complete."
