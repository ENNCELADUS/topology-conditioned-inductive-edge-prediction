# KD3 (`kd_gram`): 4-H20 HPO winner curves

**Status (verified 2026-08-31):** this directory records the 4-H20 HPO winner `kd_gram_w1`, not the
older published 2-H20 KD3 run. The winner completed 25 epochs with `w_gram=1`; the selected-epoch
marker is epoch 10. The run used `--skip-test`, so nothing in this directory is a held-out test result.

## Objective

Each official row joins by `_row_id` to the frozen teacher representation. The relational loss
matches off-diagonal cosine-Gram entries across the gathered global task batch; the endpoint-only
V3.1 student still receives exactly `(x_u, x_v)` at inference.

```text
L_total = L_task + 1.0 · L_gram
```

`L_task` is smoothed-label BCE and `L_gram` is cosine-Gram MSE (similarity-preserving KD, Tung & Mori ICCV 2019, with feature-cosine instead of row normalization). All other KD weights are zero.

## Learning and validation-topology curves

![KD3 train and validation learning curves](learning_curves.png)
![KD3 validation topology curves](validation_topology_curves.png)
The first figure plots train/validation losses; the second plots BFS-macro GS/RD and all three MMD
ratios on V_val. The dotted line marks selected epoch 10, not a minimum-loss selection rule.
[CSV](learning_curves.csv) contains all exact 25-epoch values, and the
[plot script](plot_learning_curves.py) reproduces both PNGs directly from it.

## Campaign provenance

| field | value |
|---|---|
| campaign/run | `outputs/b1_row_kd_hpo/kd_gram_w1` |
| runtime | 4 H20 ranks; bf16; seed 0; 25 epochs; `--skip-test` |
| KD weights | `w_gram=1`; all other KD weights `0` |
| selected epoch | 10 |
| V_val surface | AUPRC 0.9180; GS 0.5311; RD 1.0484; degree/clustering/spectral MMD 13.47/2.55/11.27 |

The campaign-wide source is [the HPO grid report](../kd_hpo_grid/README.md). It explicitly separates
these 4-rank V_val selection surfaces from the older 2-rank published KD1–KD4 reports.
