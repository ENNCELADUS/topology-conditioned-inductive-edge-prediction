# Whitened-axis audit of the KD banks (2026-09-03)

Which teacher vector a whitened representation KD should regress, decided per whitened PCA axis
(unit variance on training rows) of each bank. `python -m src.experiments.kd_whiten_audit` over the
86,498 train / 22,804 V_val rows with the control, kd_struct, and kd_logit students' `val_cls`
logits; raw output in [audit.json](audit.json). All R² are on the V_val block: content and
content+logit probes are ridge fits on 80% of training rows; descriptor and logit probes are
gradient-boosted trees (descriptor probes use the five ego-graph descriptors of the truth graph
with the queried partner masked). Student probes fit on one random half of V_val and score the
other half. `shift` is the axis mean on V_val in whitened units; `beyond` is
`max(0, R²content − R²student_logit) · R²desc`, the content-reachable structure the control's own
decision does not already hold. fp16 storage noise is ≥1,500× below every axis std.

## `topo` bank (equal-weight reachable structure 0.158, beyond control 0.058)

| axis | var % | shift | R² desc | R² content | R² teacher logit | R² control logit | beyond control | main loadings |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PC1 | 92.87 | -0.08 | 0.94 | 0.45 | 0.96 | 0.57 | 0.00 | CN -1.5, AA +0.6 |
| PC2 | 5.00 | -0.04 | 0.76 | 0.56 | 0.28 | 0.19 | 0.28 | deg-sum -0.9, AA +0.3 |
| PC3 | 1.40 | -0.19 | 0.50 | 0.29 | 0.41 | 0.20 | 0.05 | CN +1.4, AA -1.0 |
| PC4 | 0.23 | -0.39 | 0.33 | -0.80 | 0.44 | 0.28 | 0.00 | CN +1.8, AA -1.4 |
| PC5 | 0.15 | -0.19 | 0.39 | -0.18 | 0.35 | 0.02 | 0.00 | AA +0.9, CN -0.7 |
| PC6 | 0.09 | +0.10 | 0.70 | 0.21 | 0.03 | 0.03 | 0.12 | AA +0.8, deg-diff +0.4 |
| PC7 | 0.06 | -0.14 | 0.56 | 0.06 | 0.04 | 0.02 | 0.02 | AA +0.8, CN -0.7 |
| PC8 | 0.04 | +0.05 | 0.79 | 0.13 | 0.07 | 0.13 | 0.00 | AA +2.0, CN -1.6 |

## `pma1` bank (equal-weight reachable structure 0.166, beyond control 0.060)

| axis | var % | shift | R² desc | R² content | R² teacher logit | R² control logit | beyond control | main loadings |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PC1 | 92.81 | -0.07 | 0.95 | 0.45 | 0.93 | 0.55 | 0.00 | CN -1.5, AA +0.7 |
| PC2 | 4.55 | +0.05 | 0.85 | 0.62 | 0.12 | 0.21 | 0.35 | deg-sum -1.0, CN +0.3 |
| PC3 | 1.30 | -0.34 | 0.46 | 0.24 | 0.47 | 0.22 | 0.01 | CN +1.8, AA -1.5 |
| PC4 | 0.40 | +0.32 | 0.07 | -0.96 | 0.07 | 0.09 | 0.00 | AA +1.7, CN -1.2 |
| PC5 | 0.25 | -0.13 | 0.71 | 0.12 | 0.14 | 0.19 | 0.00 | AA +1.2, Jac -0.9 |
| PC6 | 0.19 | -0.18 | 0.85 | 0.01 | 0.01 | 0.02 | 0.00 | AA -0.6, CN +0.5 |
| PC7 | 0.12 | +0.03 | 0.75 | 0.10 | 0.11 | 0.04 | 0.05 | AA +2.1, CN -1.9 |
| PC8 | 0.10 | +0.05 | 0.51 | 0.19 | 0.29 | 0.04 | 0.08 | CN -1.3, AA +1.3 |

## `fused` bank (equal-weight reachable structure 0.158, beyond control 0.083)

| axis | var % | shift | R² desc | R² content | R² teacher logit | R² control logit | beyond control | main loadings |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PC1 | 73.30 | -0.03 | 0.87 | 0.48 | 1.00 | 0.67 | 0.00 | CN -1.9, AA +1.0 |
| PC2 | 11.09 | -0.18 | 0.37 | 0.44 | 0.96 | 0.24 | 0.07 | CN +0.9, AA -0.7 |
| PC3 | 1.56 | -0.06 | 0.30 | 0.77 | 0.01 | 0.08 | 0.20 | CN -0.4, deg-sum -0.3 |
| PC4 | 1.51 | -0.00 | 0.29 | 0.42 | 0.45 | 0.15 | 0.08 | AA -1.2, CN +0.9 |
| PC5 | 1.03 | -0.09 | 0.13 | 0.49 | 0.19 | 0.06 | 0.06 | AA -1.5, CN +1.4 |
| PC6 | 0.81 | -0.06 | 0.19 | 0.81 | 0.02 | 0.02 | 0.15 | AA -0.8, CN +0.5 |
| PC7 | 0.66 | +0.02 | 0.10 | 0.71 | 0.13 | 0.10 | 0.06 | CN -0.9, AA +0.7 |
| PC8 | 0.61 | -0.11 | 0.07 | 0.52 | 0.18 | 0.01 | 0.04 | CN +0.3, Jac -0.2 |

## Reading

- In both PMA banks the only axis that is structure, reachable from `(x_u, x_v)`, and not already in
  the students' decisions is PC2, the degree-sum axis (`pma1`: R² desc 0.85, content 0.62, teacher
  logit 0.12, control logit 0.21; beyond 0.35). PC1 is the teacher logit; PC3/PC4 (`pma1`) and
  PC4/PC5 (`topo`) shift on V_val (content R² ≤ −0.8: the teacher scores V_val rows without
  V_val-internal edges); PC5–PC8 are descriptor-explained (0.5–0.85) yet content cannot reach them (≤0.19).
- No PMA axis has low descriptor R² and high content R²: there is no content-reachable latent
  structure beyond the five descriptors in the topology vector.
- The `fused` bank's reachable axes (PC3, PC6, PC7: content 0.71–0.81) are descriptor-poor
  (≤0.30): content self-distillation, not topology.
- Decision: the whitened arm targets `pma1` axes PC2–PC8 (PC1 would re-run KD1); its expected new
  content is the degree axis plus slivers, so a null result closes option 2.

