# kd_struct: descriptor-level auxiliary arm (2026-09-02)

**Status:** single run `outputs/b1_row_kd/kd_struct` (commit 5a3d0bf), 2 H20 ranks, bf16, seed 0,
`w_struct=1`. Early-stopped by total val loss (patience 10) at epoch 22; checkpoint selection picked
epoch 11; the held-out test protocol ran automatically (`test_report.json`, numbers in
`docs/03-experiments.md` §2).

## Objective

No teacher. An MLP head on the student's pair representation regresses five z-scored truth-graph
descriptors of the queried row with the queried partner masked from both neighbour sets: log1p
common neighbours, log1p degree sum, |log1p degree difference|, Jaccard, log1p Adamic-Adar. Training
rows read the training graph (no V_val-internal edge); V_val rows read the full train-side substrate.

```text
L_total = L_task + 1.0 · MSE(head(pair_repr), z(descriptors))
```

## Learning and validation-topology curves

![kd_struct train and validation learning curves](learning_curves.png)
![kd_struct validation topology curves](validation_topology_curves.png)
Panel (d) of the first figure is the per-descriptor V_val R² of the head, the nonlinear lower bound
on the structure the student's pair representation can recover from `(x_u, x_v)`.
[CSV](learning_curves.csv) holds all 22-epoch values; the [plot script](plot_learning_curves.py)
reproduces both PNGs from it.

## Selected-epoch V_val surface

| field | value |
|---|---|
| task / descriptor MSE (train, val) | 0.381 / 0.419; 0.269 / 0.536 |
| descriptor R² (CN, deg sum, deg diff, Jaccard, AA) | 0.62, 0.44, 0.39, 0.72, 0.59 (best epochs 0.65, 0.61, 0.50, 0.77, 0.62) |
| AUPRC; GS; RD; degree/clustering/spectral MMD | 0.9226; 0.5326; 0.9948; 13.46 / 2.84 / 10.48 |
