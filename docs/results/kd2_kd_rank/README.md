# KD2 (`kd_rank`): 4-H20 HPO winner curves

**Status (verified 2026-08-31):** this directory records the 4-H20 HPO winner
`kd_rank_wr0p1_wd1`, not the older published 2-H20 `w_rank=1, w_dist=1` KD2 run. The winner completed
25 epochs with `w_rank=0.1` and `w_dist=1`; the selected-epoch marker is epoch 22. The run used
`--skip-test`, so nothing in this directory is a held-out test result.

## Objective

These curves are the retired incidental-batch KD2 winner, not evidence for the new strict-LLP KD2. The new arm uses a separate context bank with K=12 per anchor: 3 independent two-step walks retain all
6 visited nodes, plus 6 global random nodes sampled with replacement. On context probabilities,
`L_rank` uses the LLP delta=0.1 tie band and hinge margin; `L_dist` is per-anchor
`KL(softmax(t_prob)||softmax(s_prob))` at fixed temperature 1. Extra context forwards are KD-only and
separate from the row target bank; deployed task input remains exactly the endpoints `(x_u,x_v)`.

Accepted deviations from the LLP reference:
- Teacher contexts are scored offline in eval mode: dumped scores must be deterministic.
- Task loss stays protocol label-smoothed BCE on official rows: strictness applies to relational KD.
- One context bank is shared across runs: this controls sampler variance.
- V_val-internal pairs and featureless nodes are excluded: the repository data contract requires it.

## Learning and validation-topology curves

![KD2 train and validation learning curves](learning_curves.png)
![KD2 validation topology curves](validation_topology_curves.png)
The first figure plots train/validation losses; the second plots BFS-macro GS/RD and all three MMD
ratios on V_val. The dotted line marks selected epoch 22. [CSV](learning_curves.csv) and the
[plot script](plot_learning_curves.py) contain the exact values and reproduce both PNGs.

## Campaign provenance

| field | value |
|---|---|
| campaign/run | `outputs/b1_row_kd_hpo/kd_rank_wr0p1_wd1` |
| runtime | 4 H20 ranks; bf16; seed 0; 25 epochs; `--skip-test` |
| KD weights | `w_rank=0.1`, `w_dist=1`; all other KD weights `0` |
| selected epoch | 22 |
| V_val surface | AUPRC 0.9205; GS 0.5263; RD 1.0490; degree/clustering/spectral MMD 13.95/2.50/10.09 |

The campaign-wide source is [the HPO grid report](../kd_hpo_grid/README.md). It explicitly separates
these 4-rank V_val selection surfaces from the older 2-rank published KD1–KD4 reports.
