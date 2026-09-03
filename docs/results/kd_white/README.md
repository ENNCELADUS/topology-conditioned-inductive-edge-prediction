# kd_white: whitened teacher-axis arm (2026-09-03)

**Status:** single run `outputs/b1_row_kd/kd_white` (commit edbfb6d), 2 H20 ranks, bf16, seed 0,
`w_white=1`. Early-stopped by total val loss (patience 10, best at epoch 5) at epoch 15; checkpoint
selection picked epoch 10 (`a15d791fab3d495f`); the held-out test ran automatically.

## Objective

The kd_struct auxiliary head regresses, by per-axis MSE, the whitened PCA axes 2--8 of the PMA(1)
Full-Ego teacher's topology vector (`python -m src.distill.whiten_targets --axes 2-8`; bank
`outputs/distill/kd_row_targets_pma1_white_breadth_first`). Axis 1 is the teacher logit and is
dropped, so the loss is orthogonal to kd_logit. Axis choice follows the
[whitened-axis audit](../kd_whiten_audit/README.md).

## Per-axis V_val R² of the head against the audit's linear content probe

| axis | linear content probe | head, selected epoch 10 | head, epochs 10--15 | audit reading |
|---|---:|---:|---|---|
| pc2 | 0.62 | 0.63 | 0.63--0.66 | degree sum; matches the linear bound |
| pc3 | 0.24 | 0.24 | 0.24--0.38 | CN vs AA contrast; shifts on V_val (−0.34) |
| pc4 | −0.96 | −1.58 | −1.6 to −2.5 | shifted on V_val (+0.32); diverges as training proceeds |
| pc5 | 0.12 | 0.09 | 0.00--0.09 | Jaccard/AA; not reached |
| pc6 | 0.01 | 0.01 | 0.00--0.06 | degree difference/CN; not reached |
| pc7 | 0.10 | 0.18 | 0.18--0.31 | AA/CN; nonlinear gain +0.2 |
| pc8 | 0.19 | 0.28 | 0.28--0.33 | AA/CN; nonlinear gain +0.1 |

[CSV](learning_curves.csv) holds every epoch's values (losses, topology surface, per-axis R²).

## Selected-epoch V_val surface and held-out test

| field | value |
|---|---|
| V_val AUPRC; GS; RD; degree/clustering/spectral MMD | 0.9252; 0.532; 0.993; 14.94 / 2.45 / 11.19 |
| held-out AUROC/AUPRC; Accuracy/F1/MCC; ECE/Brier | 0.7186/0.7429; 0.6453/0.6754/0.2958; 0.155/0.240 |
| held-out GS/RD; degree/clustering/spectral MMD (frozen logit threshold 2.984) | 0.3939/0.5696; 8.66 / 7.42 / 13.06 |

## Reading

- The head reaches the two tail axes the linear probe called near-unreachable (pc7, pc8) only to
  about 0.3, matches the linear bound on the degree axis, and reads nothing on pc5/pc6; pc4's
  divergence is the V_val block shift, not a fit failure. These are joint-training (w=1) lower bounds.
- V_val AUPRC 0.925--0.927 from epoch 9 on sits above the control's selected 0.9193 by the same
  margin as kd_logit; held-out AUPRC +0.007 over control, GS +0.009, RD 0.57 (nearest to 1 among the
  KD arms bar kd_gen_edm) and MMD ratios on the same density line as every arm.
- Matched-epoch control (`kd_control` epoch-0010 checkpoint through the same protocol, report in
  `outputs/b1_row_kd_hpo/kd_control/matched_epoch_0010`): AUROC/AUPRC 0.7249/0.7515, F1 0.6852, ECE 0.134,
  GS/RD 0.4261/0.5765, MMD 9.16/8.01/14.41. kd_white trails it on every edge metric and on GS; the
  arm's margin over the published control was its epoch. Option 2 is closed at w=1.
