# KD2 (`kd_rank`): provisional strict-LLP Trial 8 curves

**Status (verified 2026-09-03):** this directory records strict-LLP HPO Trial 8. The 16-trial sweep is
unfinished; Trial 8 is diagnostic unless it later wins by V_val alone.

## Objective

Trial 8 uses the `h2ns3` context bank with at most K=24 partners per anchor: 3 independent two-step
walks retain up to 6 visits, plus 18 global random nodes sampled with replacement. On probabilities,
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
The first figure plots train/validation losses; the second plots cadence-2 BFS-macro GS/RD and all
three MMD ratios on V_val, with Oracle references. The dotted line marks selected epoch 18. [CSV](learning_curves.csv) and the
[plot script](plot_learning_curves.py) contain the exact values and reproduce both PNGs.

## Campaign provenance

| field | value |
|---|---|
| campaign/run | `outputs/b1_kd_rank_strict_hpo/trial_008` |
| runtime | 4 H20 training ranks; bf16; seed 0; 25 epochs; direct 2-H20 held-out test |
| KD configuration | `w_rank=0.1`, `w_dist=10`, `h2ns3`, margin `0.1` |
| selected epoch | 18 |
| V_val surface | AUPRC 0.9205; GS 0.5414; RD 1.0133; MMD 14.90/2.50/11.43 |
| held-out edge | AUROC/AUPRC 0.7183/0.7457; Accuracy/F1/MCC 0.6477/0.6656/0.2970 |
| held-out topology | GS/RD 0.4143/0.4399; MMD 14.56/12.41/21.03 |

The held-out result must not influence selection; the final KD2 row is formal only if Trial 8 later
wins the unfinished HPO under the frozen V_val-only rule.
