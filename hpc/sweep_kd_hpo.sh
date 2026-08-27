#!/usr/bin/env bash
# Serial two-lane driver for the B1 KD loss-weight sweep (configs/sweep/b1_kd_hpo).
#
# Usage: hpc/sweep_kd_hpo.sh <lane>       lane 0 or 1
#   lane 0 -> CUDA_VISIBLE_DEVICES=0,1 and the even-indexed configs (sorted order)
#   lane 1 -> CUDA_VISIBLE_DEVICES=2,3 and the odd-indexed configs
# Two GPUs per run pins world_size=2 for comparability with the original KD runs.
#
# The operator launches each lane with nohup from the H20 checkout root, e.g.:
#   nohup hpc/sweep_kd_hpo.sh 0 > outputs/b1_row_kd_hpo/lane0.nohup 2>&1 &
# Kill a lane by its PID only (kill <pid>), never pkill -f.
#
# Every run is `hpc/run.sh train <cfg> --skip-test`: sweep selection is
# V_val-only; the held-out test protocol is spent later on per-arm winners.
# A config whose output dir already holds complete.json is skipped (resume).
set -euo pipefail

LANE="${1:?usage: hpc/sweep_kd_hpo.sh <lane 0|1>}"
case "${LANE}" in
  0) export CUDA_VISIBLE_DEVICES=0,1 ;;
  1) export CUDA_VISIBLE_DEVICES=2,3 ;;
  *) echo "ERROR: lane must be 0 or 1, got ${LANE}" >&2; exit 1 ;;
esac
# The 224-core H20 box spin-waits concurrent torch jobs into a 12x slowdown
# without a thread cap.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

SWEEP_ROOT="outputs/b1_row_kd_hpo"
LOG_DIR="${SWEEP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

mapfile -t CONFIGS < <(printf '%s\n' configs/sweep/b1_kd_hpo/*.yaml | sort)
[[ -f "${CONFIGS[0]:-}" ]] || { echo "ERROR: no sweep configs found" >&2; exit 1; }

index=0
for cfg in "${CONFIGS[@]}"; do
  lane_of_cfg=$(( index % 2 ))
  index=$(( index + 1 ))
  (( lane_of_cfg == LANE )) || continue
  stem="$(basename "${cfg}" .yaml)"
  out_dir="${SWEEP_ROOT}/${stem}"
  if [[ -f "${out_dir}/complete.json" ]]; then
    echo "lane ${LANE}: skipping ${stem} (complete.json exists)"
    continue
  fi
  echo "lane ${LANE}: training ${stem}"
  rc=0
  bash hpc/run.sh train "${cfg}" --skip-test 2>&1 | tee "${LOG_DIR}/${stem}.log" || rc=$?
  if (( rc != 0 )) || [[ -f "${out_dir}/failure.json" ]]; then
    echo "ERROR: lane ${LANE} aborting at ${stem} (exit ${rc})" >&2
    exit 1
  fi
done
echo "lane ${LANE}: all assigned configs complete"
