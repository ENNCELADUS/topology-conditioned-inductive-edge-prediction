#!/usr/bin/env bash
# Approved compact-reader -> virtual-student -> intervention chain.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

reader=outputs/split_seed42/topo_prompt_full_v3
student=outputs/split_seed42/virtual_prompt_d

hpc/run.sh train configs/split_seed42/topo_prompt_full_v3.yaml \
  --worker-module src.train_b0 --run-kind diagnostic
test -f "$reader/diagnostic_complete.json"
test ! -f "$reader/failure.json"
hpc/run.sh train configs/split_seed42/virtual_prompt_d.yaml
test -f "$student/complete.json"
test -f "$student/test_complete.json"
test ! -f "$student/failure.json"

for intervention in gates_off mean mean_relation shuffle slot_gates_open; do
  hpc/run.sh test --checkpoint "$student/best.pt" \
    --output-dir "$student/intervention_$intervention" \
    --data-root data --strategy breadth_first \
    --arm "virtual_prompt_d_$intervention" --seed 0 \
    --prefix-intervention "$intervention"
done
