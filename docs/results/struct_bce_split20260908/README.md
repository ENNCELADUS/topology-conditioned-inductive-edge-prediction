# Structural stream, wave 1 on the 2026-09-08 split: `struct_bce` vs B0, first `struct_grand` trial

**Status (2026-09-08 13:30 UTC):** `struct_bce` and B0 have held-out reports on the node-held-out
split (root `node_002696`, seed 273). The `struct_grand` study has 1 of 10 trials complete (trial 0,
the enqueued prior `gs 0.70 / rd 0.90`; ~80 min per trial on 30846); trial 1 (`gs 0.35 / rd 0.45`)
is running. `struct_new` is queued behind it. Nothing here is a wave-1 verdict yet: the verdict
needs both winners tested.

Sources: [`test_report.json`](test_report.json) (this arm),
[B0 report](../b0_v31_split20260908/test_report.json), and the runs' `metrics.jsonl` under
`outputs/split20260908/{b0_v31,struct_bce,struct_hpo/grand/trial_000}` on the H20 checkout.

## Held-out test (single run each, one frozen V_val-selected threshold per arm)

| Arm | seed | selected epoch | AUROC | AUPRC | Acc | F1 | MCC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 (`b0_v31`) | 47 | 6 | 0.7081 | 0.7413 | 0.6153 | 0.6743 | 0.2474 | 0.2016 | 0.2619 |
| `struct_bce` | 0 | 7 | 0.7017 | 0.7330 | 0.6154 | 0.6736 | 0.2470 | 0.1424 | 0.2408 |

| Arm | topology logit thr | BFS GS ↑ | BFS RD → 1 | Degree ↓ | Clustering ↓ | Spectral ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B0 (`b0_v31`) | 1.125 | 0.4364 | 1.2008 | 3.908 | 3.984 | 5.854 |
| `struct_bce` | 0.443 | 0.4382 | 1.1356 | 3.837 | 3.779 | 6.379 |

V_val at the selected epoch (the numbers the cascade saw): B0 GS 0.4466 / RD 0.969 /
MMD 3.12, 2.25, 5.08; `struct_bce` GS 0.4508 / RD 0.979 / MMD 3.05, 2.29, 4.75.

Per-size test RD (sizes 20 → 200): B0 0.75, 0.85, 0.99, 1.14, 1.22, 1.35, 1.41, 1.39, 1.38, 1.53;
`struct_bce` 0.71, 0.82, 0.94, 1.04, 1.16, 1.31, 1.37, 1.30, 1.28, 1.42. Per-size GS is identical
between the arms to within 0.005 at every size.

## Findings

1. **The sampler-matched baseline is B0 within noise on every headline number.** AUPRC is 0.008 lower,
   GS 0.002 higher, RD 0.065 closer to 1, degree and clustering MMD slightly better, spectral worse.
   The two runs differ in seed (47 vs 0), so none of these gaps isolates the structural stream; they
   set the size of seed-plus-stream noise that any weighted arm must clear (roughly ±0.01 AUPRC,
   ±0.01 GS, ±0.07 RD, ±0.5 spectral MMD).
2. **The one clear difference is calibration:** ECE 0.142 vs 0.202 and Brier 0.241 vs 0.262. The
   extra BCE over sampled subgraphs (positive fraction 0.14 versus the 1:6 task stream) pulls the
   raw probabilities toward the observed rate. It is an extra supervision effect on the same
   labels, not a topology effect, and it does not touch the threshold-based metrics.
3. **Selected epochs are matched (6 vs 7), so the epoch confound that erased the earlier KD gains is
   absent here.** Both runs peak in validation task loss at epochs 3–6 and then overfit steadily
   (val task loss 0.46 → 0.86 for B0, 0.46 → 1.12 for `struct_bce`; ECE rising with it). Patience
   on `val_task_loss` stops them at 13 and 16 of 25 epochs, so the one-cycle schedule never reaches
   its annealing phase. The best checkpoints are taken at near-peak learning rate. This is a recipe
   property of the split campaign, not of either arm.
4. **Density transfers worse than the cascade sees.** Both arms sit at RD ≈ 0.97 on V_val but
   1.14–1.20 on test, and the per-size profile is monotone: under-dense at 20–60 nodes, over-dense
   from 80 nodes up. A single global threshold cannot flatten a size-dependent density error; the
   test bucket bank is denser at large sizes than V_val, which is the residual of the
   test-informed root selection (`validation_density_selection`).
5. **`struct_grand` trial 0 (gs 0.70, rd 0.90) moves the V_val topology numbers, at a task-loss cost.**
   V_val at its selected epoch 12: GS 0.4631, RD 0.986, MMD 2.98 / 1.96 / 4.78 (surface objective
   GS 0.463, geometric-mean MMD 3.04, constraint satisfied). Against `struct_bce` at its own
   selected epoch 7 that is +0.012 GS and −0.33 clustering MMD; against `struct_bce` at the matched
   epoch 12 (GS 0.436, RD 0.951, MMD 3.24 / 1.85 / 4.71) it is +0.027 GS with clustering slightly
   worse. Peak GS inside trial 0 is 0.472 at epoch 8, which the mean-rank selector passes over for
   epoch 12's better MMDs. The price: validation task loss stays at 0.9–1.1 after epoch 5 (min 0.58
   at epoch 4, versus 0.46 for the baseline), validation ECE 0.28, AUPRC 0.866 vs 0.873, and the
   selected topology logit threshold is −1.34 (baseline +0.44): the L1 soft-GS and log-ratio RD terms
   shift the logit scale down. V_val threshold selection absorbs the shift for topology, but raw
   ECE/Brier on test will be worse than the baseline's if this trial is the winner.

## What follows

- Wait for the remaining 9 `grand` trials and the 10 `new` trials before any wave-1 statement.
  Judge trial 0's GS gain against the ±0.01 GS seed band from finding 1, at matched epoch.
- Rerun B0 at seed 0 (config already reset in `f99a40b`) so the control matches every other split
  arm; the seed-47 run stays as a second seed for the noise band. The seed-47 output directory
  must be moved or the config's `output_dir` changed before that launch.
- Report the winners' test ECE/Brier alongside the topology numbers (finding 5); the claim rules
  already require the edge and graph families together.
