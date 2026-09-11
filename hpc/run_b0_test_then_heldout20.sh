#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /2023533015/topology-conditioned-inductive-edge-prediction
output=outputs/split_seed42_geometric_20260910/b0_v31
if [[ ! -s "$output/test_report.json" ]]; then
  bash hpc/run.sh test --checkpoint "$output/best.pt" --output-dir "$output" \
    --data-root data --strategy breadth_first --arm b0_v31 --seed 0 \
    --pack-dir outputs/feature_packs/b0_v31_bf16
fi
cd /2023533015/topology-heldout20-20260910
bash hpc/run.sh train configs/heldout20_seed42/b0_v31.yaml
