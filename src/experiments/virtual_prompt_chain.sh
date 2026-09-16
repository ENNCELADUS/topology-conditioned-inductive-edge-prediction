#!/usr/bin/env bash
# Repair reader -> virtual student -> intervention chain; launch only when requested.
# Run the production-size memory/throughput check before launching this chain.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

reader=outputs/split_seed42/topo_prompt_full_v4
student=outputs/split_seed42/virtual_prompt_v2_d

hpc/run.sh train configs/split_seed42/topo_prompt_full_v4.yaml \
  --worker-module src.train_b0 --run-kind diagnostic
test -f "$reader/diagnostic_complete.json"
test ! -f "$reader/failure.json"
hpc/run.sh train configs/split_seed42/virtual_prompt_v2_d.yaml
test -f "$student/complete.json"
test -f "$student/test_complete.json"
test ! -f "$student/failure.json"

for intervention in gates_off mean mean_relation shuffle; do
  hpc/run.sh test --checkpoint "$student/best.pt" \
    --output-dir "$student/intervention_$intervention" \
    --data-root data --strategy breadth_first \
    --arm "virtual_prompt_v2_d_$intervention" --seed 0 \
    --prefix-intervention "$intervention"
done
