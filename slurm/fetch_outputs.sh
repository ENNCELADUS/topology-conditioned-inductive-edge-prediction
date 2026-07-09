#!/usr/bin/env bash
# Run LOCALLY (not on the cluster). Rsyncs job outputs back from the cluster
# into the local repo.
#
# Usage:
#   slurm/fetch_outputs.sh                     # fetch outputs/ and scores/
#   slurm/fetch_outputs.sh outputs/run-42       # fetch a specific subpath
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

SUBPATHS=("$@")
if [[ ${#SUBPATHS[@]} -eq 0 ]]; then
  SUBPATHS=("outputs" "scores")
fi

for subpath in "${SUBPATHS[@]}"; do
  echo "Fetching ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/${subpath}/ -> ${REPO_ROOT}/${subpath}/"
  mkdir -p "${REPO_ROOT}/${subpath}"
  rsync -az "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/${subpath}/" "${REPO_ROOT}/${subpath}/"
done

echo "Fetch complete."
