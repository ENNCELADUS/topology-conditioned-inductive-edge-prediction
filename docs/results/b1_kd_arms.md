# B1 training-time KD arms: definitions and results

**Status:** the nine `kd_control`/D1–D8 anchor-context runs below are retired as confounded and support
no method claim. Current full-row reports: [`KD1`](kd1_kd_logit/README.md), [`KD2`](kd2_kd_rank/README.md), [`KD3`](kd3_kd_gram/README.md), [`KD4`](kd4_kd_rep/README.md); loss-weight HPO grid and campaign plan: [`kd_hpo_grid`](kd_hpo_grid/README.md).

## Retired: sampled anchor-context KD arms (historical)

KD supervision below used a separately sampled, biased near/random anchor-context pair distribution
(k=5 near + 5 random partners per anchor) with its own prevalence and label balance, scored via a
second `KDStream` student forward per step on a CSR context artifact -- disjoint from the task loss's
own batch rows. Arm differences therefore confound the KD mechanism itself with sampling bias,
prevalence shift, and partner-distribution shift, not just the loss term. These nine runs
(`kd_control`, D1–D8) are retired for that reason.

### Shared protocol and threshold (retired)

All nine arms used the V3.1 endpoint-only student, the same 74,692-row anchor-context KD stream,
25-epoch recipe, seed 0, and the held-out test protocol (64,038 rows; 32,019 positives and negatives;
1,891 self rows), diagnostic (`formal:false`) within the simple-B0 protocol -- within-protocol arm
comparison only, never a formal fully inductive or current-`V_val` claim. Edge metrics use threshold
0.5 where required; AUROC/AUPRC are threshold-free. Graph metrics use each arm's density-matched
(`RD-matched`) threshold targeting the reference's 30,128 loopless edges, so global RD is near 1 by
construction and BFS-macro RD is the informative number; GS is global simple-edge and BFS-macro
Dice/F1, reported separately. MMD columns are ratios to the real-vs-real floor (1 = floor, lower is
better).

### Method definitions (retired)

| arm | loss / target | mechanism being tested |
|---|---|---|
| `kd_control` | label BCE on the KD rows (`w_label=1`) | matched sampler/capacity control; no teacher signal |
| D1 | BCE to `sigmoid(teacher_logit)` (`w_logit=1`) | pointwise soft-logit KD (GLNN family) |
| D2 | within-anchor margin rank + `KL(softmax(T)||softmax(S))`, T=1 (`w_rank=w_dist=1`) | LLP listwise partner ordering/distribution |
| D3 | cosine-Gram match of student pair factors to teacher pooled embeddings (`w_gram=1`) | pair-space relational geometry (Graph2Feat/CAZI family) |
| D4 | `1-cos(P(z_a*z_b), 0.5(t_ab+t_ba))` (`w_align=1`) | projected representation alignment; projection is train-only |
| D5 | Huber on student node-factor residual vs `teacher_logit-content_logit` (`w_residual=1`) | beyond-content topology-residual distillation |
| D6 | D2 + D3 (`w_rank=w_dist=w_gram=1`) | additivity/interference between listwise and Gram KD |
| D7a | D2 losses, but `teacher_logit=log1p(resource_allocation)` | parameter-free heuristic teacher (EHDM-style provenance test) |
| D8 | Huber-matches teacher/student `sum(sigmoid(logit))` over each anchor's sampled 5-near + 5-random KD context (`w_rowmass=1`) | sampled anchor-context probability-mass transfer; sensitive to logit shifts, but not full-node degree-budget supervision |

### Edge-level results (retired)

Held-out test; Acc/F1/MCC/ECE/Brier are at 0.5. `AUPRC¬self` removes the all-positive self stratum.

| arm | epoch | AUROC | AUPRC | AUPRC¬self | Acc | F1 | MCC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | — | 0.7154 | 0.7411 | 0.7017 | 0.6254 | 0.4696 | 0.3101 | 0.1925 | 0.2560 |
| D1 | — | 0.7199 | 0.7467 | 0.7082 | 0.6529 | 0.6300 | 0.3081 | 0.1082 | 0.2245 |
| D2 | — | 0.7266 | 0.7478 | 0.7094 | 0.6533 | 0.6018 | 0.3174 | 0.0565 | 0.2126 |
| D3 | — | 0.7151 | 0.7452 | 0.7064 | 0.6478 | 0.5665 | 0.3188 | 0.1758 | 0.2504 |
| D4 | 12 | 0.7177 | 0.7473 | 0.7087 | 0.6559 | 0.6213 | 0.3170 | 0.1511 | 0.2389 |
| D5 | 11 | 0.7216 | **0.7484** | **0.7108** | 0.6109 | 0.3977 | 0.3142 | 0.3230 | 0.3213 |
| D6 | 25 | 0.7260 | 0.7474 | 0.7092 | 0.6535 | 0.6017 | 0.3181 | 0.0544 | 0.2123 |
| D7a | 25 | 0.7302 | 0.7448 | 0.7040 | 0.6371 | 0.5242 | 0.3114 | 0.0981 | 0.2227 |
| D8 | 25 | **0.7337** | 0.7481 | 0.7099 | **0.6622** | **0.6511** | **0.3251** | **0.0538** | **0.2108** |

### Density-matched assembled-graph results (retired)

| arm | threshold | GS global | GS BFS | RD global | RD BFS | degree-MMD ratio | clustering-MMD ratio | spectral-MMD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0.697609 | 0.1864 | 0.4175 | 0.9963 | 0.4342 | 16.82 | 14.26 | 24.23 |
| D1 | 0.893309 | 0.1818 | **0.4192** | 0.9810 | **0.4608** | 13.54 | 11.64 | 20.66 |
| D2 | 0.832237 | 0.1824 | 0.4129 | 0.9986 | 0.4379 | 17.11 | 14.55 | 26.01 |
| D3 | 0.884039 | 0.1820 | 0.4142 | 0.9993 | 0.4605 | **13.37** | **11.54** | **19.65** |
| D4 | 0.926304 | 0.1851 | 0.4137 | 0.9921 | 0.4459 | 14.63 | 12.63 | 21.55 |
| D5 | 0.569853 | 0.1734 | 0.4101 | 0.9969 | 0.4605 | 13.39 | 11.57 | 19.78 |
| D6 | 0.833325 | **0.1886** | 0.4163 | 0.9950 | 0.4401 | 17.11 | 14.39 | 25.85 |
| D7a | 0.545286 | 0.1874 | 0.4099 | 0.9965 | 0.4114 | 22.73 | 18.09 | 30.84 |
| D8 | 0.808067 | 0.1861 | 0.4036 | 0.9965 | 0.4042 | 22.51 | 17.94 | 32.04 |

Historical notes (retired): D8 led AUROC/ECE, D3 led all three MMD ratios; BFS GS never left
0.4036–0.4192 around control 0.4175, so edge-set identity stayed flat under every mechanism. All runs
are seed 0 only. Reports: `outputs/b1_stage_v{2,3}/kd_*/test_report.json`; D8 on branch `s4-s6-budget-probes-d8` at `da6f5a500cfd043a715e7c04ddb16b9d410eb0a5`.

## Full-row KD protocol (current)

Teacher targets are dumped once for every official training row (plus the V_val classification rows,
validation-only) into artifact `kd_row_targets_v1` (`src/distill/teacher_targets.py`): pair identity,
row id by position, teacher logit, and symmetrized PMA(1) pooled latent target `teacher_rep`. Teacher
inference applies query-edge masking -- a
positive training edge is never visible in its own structural context. The trainer computes task and
KD losses from one student forward on identical rows, joined by `_row_id`; there is no KD-only stream.

### Arms

- `kd_control` -- no `distill:` section; matched control that zero KD weight must reproduce.
- `kd_logit` -- `w_logit`: binary soft-target BCE to `sigmoid(teacher_logit)` (GLNN family).
- `kd_rank` -- `w_rank`+`w_dist`: D2 margin ranking and per-anchor distribution KL; each official
  non-self row participates under both endpoint anchors, without adding rows or another forward.
- `kd_gram` -- `w_gram`: D3 cosine-Gram matching across all distinct batch rows, including rows that share an endpoint.
- `kd_rep` -- `w_rep`: per-row cosine alignment of the student pair representation to `teacher_rep`.
- `kd_gen` -- `w_gen`: endpoint-conditioned generation of the PMA(1) pooled topology latent; see `docs/superpowers/specs/2026-08-30-kd-gen-arm-design.md`.
### Telemetry

Task/KD loss split, logit correlation/error, representation cosine, D2/D3 validation-block losses,
per-term gradient norms, val AUPRC/ECE/Brier, and the five-number topology result (GS, RD, and the
degree/clustering/spectral MMD ratios). KD1–KD4 reports are linked in the status line above.
