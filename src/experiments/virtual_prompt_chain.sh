#!/usr/bin/env bash
# Approved compact-reader -> virtual-student -> intervention chain.
# The student config is $1 (default: the revised generator, virtual_prompt_d_rev1).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

config=${1:-configs/split_seed42/virtual_prompt_d_rev1.yaml}
arm=$(basename "$config" .yaml)
reader=outputs/split_seed42/topo_prompt_full_v3
student=outputs/split_seed42/$arm

# The reader is the same immutable Stage I ceiling for every student revision.
if [ ! -f "$reader/diagnostic_complete.json" ]; then
  hpc/run.sh train configs/split_seed42/topo_prompt_full_v3.yaml \
    --worker-module src.train_b0 --run-kind diagnostic
  test -f "$reader/diagnostic_complete.json"
fi
test ! -f "$reader/failure.json"
hpc/run.sh train "$config"
test -f "$student/complete.json"
test -f "$student/test_complete.json"
test ! -f "$student/failure.json"

for intervention in gates_off mean mean_relation shuffle slot_gates_open; do
  hpc/run.sh test --checkpoint "$student/best.pt" \
    --output-dir "$student/intervention_$intervention" \
    --data-root data --strategy breadth_first \
    --arm "${arm}_$intervention" --seed 0 \
    --prefix-intervention "$intervention"
done
