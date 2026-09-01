# KD4 (`kd_rep`): 4-H20 HPO winner curves

**Status (verified 2026-08-31):** this directory records the 4-H20 HPO winner `kd_rep_w0p1`, not the
older published 2-H20 unit-weight KD4 run. The winner completed 25 epochs with `w_rep=0.1`; the
selected-epoch marker is epoch 14. The run used `--skip-test`, so nothing in this directory is a
held-out test result.

## Objective

Each official row joins by `_row_id` to the frozen teacher pooled representation. The student pair
representation is aligned to that target, while the endpoint-only V3.1 student still receives exactly
`(x_u, x_v)` at inference.

```text
L_total = L_task + 0.1 · L_rep
```

`L_task` is smoothed-label BCE and `L_rep` is per-row cosine representation loss. All other KD
weights are zero.

## Learning and validation-topology curves

![KD4 train and validation learning curves](learning_curves.png)
![KD4 validation topology curves](validation_topology_curves.png)
The first figure plots train/validation losses; the second plots BFS-macro GS/RD and all three MMD
ratios on V_val. The dotted line marks selected epoch 14, not a minimum-loss selection rule.
[CSV](learning_curves.csv) contains all exact 25-epoch values, and the
[plot script](plot_learning_curves.py) reproduces both PNGs directly from it.

## Campaign provenance

| field | value |
|---|---|
| campaign/run | `outputs/b1_row_kd_hpo/kd_rep_w0p1` |
| runtime | 4 H20 ranks; bf16; seed 0; 25 epochs; `--skip-test` |
| KD weights | `w_rep=0.1`; all other KD weights `0` |
| selected epoch | 14 |
| V_val surface | AUPRC 0.9158; GS 0.5048; RD 1.0405; degree/clustering/spectral MMD 12.98/2.47/10.04 |

The campaign-wide source is [the HPO grid report](../kd_hpo_grid/README.md). It explicitly separates
these 4-rank V_val selection surfaces from the older 2-rank published KD1–KD4 reports.
