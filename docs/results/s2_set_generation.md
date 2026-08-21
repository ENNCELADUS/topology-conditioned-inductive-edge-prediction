# S-series topology diagnostics: definitions and results

**Current result:** S2–S6 consistently find learnable topology correlates, but none establishes a
useful set-conditioned or node-budget mechanism beyond strong endpoint pair scoring. S3's pair-only
control beats its set-conditioned model, S4's predicted budgets worsen descriptor MMD, S6 residual
R² stays low, and the related STPD pre-gate returns `edge_identity_killed`.

All numbers are diagnostic (`formal=false`). S0/S0-R/S1 belong to the retired `V_hold` protocol and are archived lineage evidence, not current `V_val` results. S2/S3 use sampled test-node sets, so their density-matched GS/RD are comparable only within their tables, not to the official full-candidate protocol.

## Method map

| stage | definition | status / question answered |
|---|---|---|
| S0 | predict four low-order ego scalars from node features, append endpoint sums/differences to a ridge pair head | archived; only tests a narrow linear summary route |
| S0-R | predict CN/AA from the same linear pair features, append them to another ridge head | archived; algebraically near-redundant linear projection |
| S1 | post-process frozen B0 scores with IPF/greedy degree quotas or CN updates | archived; tests frozen-score allocation, not joint learning |
| S2 | generate an induced set topology from all node features: conditional GEN, unconditional UNC, shuffled-condition SHUF, deterministic DET; AE is reconstruction ceiling, B0 pair-score baseline, MARG quota baseline | completed, seed 0 |
| S3 | learn zero-init residual `logit=B0+delta`: RES uses a Set Transformer; PAIR uses parameter-matched independent node MLPs; DIAG runs the same set backbone on singleton sets; SHUF permutes set context | completed, three seeds |
| S4 | no training: replay frozen B0 candidate scores under hard degree quotas from S5; compare exact-N control and true-degree oracle | completed |
| S5 | full B0 trunk regresses `log1p(loopless degree)` from one node's features; warm vs scratch | completed, three seeds; held-out-node targets share graph edges and are not independent generalization |
| S6 | full V3.1 regresses `teacher_logit-content_logit` on anchor-disjoint KD rows; warm vs scratch, zero-output-head control | completed, three seeds |
| STPD | degree-preserving edge-swap regions; distinguish deleted true edges from inserted false edges using pair features (P), pair+corrupted-graph context (S), B0 and RA/CN/degree-parity controls | completed pre-gate, three seeds |

## Archived S0–S1 results (`V_hold`; invalid for current-result comparison)

| stage | archived mechanism result |
|---|---|
| S0 | pair AUPRC: features 0.1197, +oracle node summaries 0.1434, +predicted summaries 0.1231, oracle summaries+CN/AA 0.3653; degree/clustering predictability R² 0.435/0.051 |
| S0-R | predicted `CN>0` AUPRC 0.298 vs degree-product 0.346; predicted CN/AA changed pair AUPRC by only `2.3e-8`, while oracle CN/AA raised 0.1197→0.3634 |
| S1-R | frozen-B0 exact-N→true node-aligned hard quota: BFS GS 0.390→0.439, BFS RD 0.423→0.622, MMD 13.03/11.86/18.07→4.06/5.44/7.65; oracle shortfall 1.62% |
| S1-H | rank-matched multiset / predicted degree / training prior BFS GS 0.340/0.399/0.329; shortfall 0.08%/8.43%/4.02%. Predicted/prior exceed the 2% validity limit and are lower bounds |

## S2 set generation

Macro over 10 sizes × 50 test-node sets. GS/RD use density-matched in-set assembly. GEN/UNC/SHUF edge/GS/RD use the 32-draw mean probability; their MMD descriptors use draw 0 only. MMD ratios are relative to the deterministic real-vs-real reference floor (ratio 1 is the floor; ↓).

| arm | AUPRC | GS | RD | Spearman | hub recall | degree-MMD ratio | clustering-MMD ratio | spectral-MMD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GEN | 0.263 | 0.242 | 0.839 | 0.185 | 0.127 | 6.69 | 4.17 | 7.24 |
| UNC | 0.185 | 0.171 | 0.839 | −0.009 | 0.105 | 5.48 | 3.59 | 6.25 |
| SHUF | 0.185 | 0.169 | 0.839 | −0.010 | 0.093 | 6.43 | 3.94 | 7.21 |
| DET | 0.204 | 0.108 | 0.513 | 0.094 | 0.108 | 23.98 | 17.35 | 28.31 |
| AE ceiling | **0.969** | **0.845** | 0.839 | **0.983** | **0.899** | 4.45 | **1.12** | 4.75 |
| B0 | 0.304 | 0.282 | 0.837 | 0.307 | 0.190 | **3.55** | 2.78 | 5.98 |
| MARG | 0.304 | 0.245 | 0.828 | — | 0.142 | 15.91 | 11.92 | 5.69 |

GEN separates from UNC/SHUF only at sizes ≥120 (intervals overlap at smaller sizes), showing feature-aligned topology signal; this comparison alone does not isolate third-party set context. B0 dominates GEN on every node-aligned endpoint. AE shows that the latent decoder is expressive and the conditional map is the bottleneck. At threshold 0.5, GS/RD are AE 0.835/0.785, B0 0.232/0.563 and GEN 0.076/0.062. MARG collapsed to constant predictions, so GEN-vs-MARG is not a valid headline comparison.

## S3 set residual

Mean ± sample SD across three seeds; B0 is fixed. MMD triplets are degree/clustering/spectral ratios.

| arm | AUPRC | GS | RD | Spearman | hub recall | MMD triplet |
|---|---:|---:|---:|---:|---:|---|
| B0 | 0.3035 | 0.2821 | 0.8370 | 0.3070 | 0.1902 | 3.55 / 2.78 / 5.98 |
| RES | 0.3120±0.0053 | 0.2868±0.0035 | 0.8388 | 0.3368±0.0125 | 0.1936 | 3.46 / 2.71 / 6.00 |
| PAIR | **0.3154±0.0011** | **0.2907±0.0017** | 0.8387 | **0.3509±0.0013** | **0.2083** | 3.52 / 2.89 / 5.98 |
| DIAG | 0.3122±0.0018 | 0.2897±0.0019 | 0.8388 | 0.3475±0.0003 | 0.2037 | **3.35 / 2.90 / 5.91** |
| RES-SHUF | 0.3119±0.0054 | 0.2870±0.0034 | 0.8388 | 0.3369±0.0135 | 0.1941 | 3.47 / 2.71 / 6.04 |

PAIR outperforms RES, while RES and RES-SHUF are effectively identical; no measurable incremental set-context gain is shown here.

## S4 hard-budget assembly

`P/R` are simple-edge precision/recall; then global/BFS GS, global/BFS RD and the MMD triplet.

| arm | P/R | GS global/BFS | RD global/BFS | degree/clustering/spectral MMD ratios | quota shortfall |
|---|---|---|---|---|---|
| B0 exact-N | .137/.137 | .137/.390 | 1.000/.423 | 13.03/11.86/18.07 | — |
| true-degree oracle | .216/.212 | .214/.439 | .984/.622 | 4.06/5.44/7.65 | 1.62% |
| scratch s0/s1/s2 | .175/.154; .176/.150; .184/.157 | .164/.397; .162/.395; .169/.398 | .882/.399; .852/.397; .849/.390 | 15.33/14.46/19.25; 15.35/14.61/19.16; 17.05/15.34/21.41 | 11.8/14.8/15.1% (LB) |
| warm s0/s1/s2 | .177/.155; .168/.148; .174/.155 | .165/.401; .158/.397; .164/.399 | .879/.402; .883/.400; .891/.402 | 15.49/14.43/19.77; 14.70/13.98/18.53; 15.04/14.15/18.77 | 12.1/11.7/10.9% (LB) |

Predicted quotas slightly improve GS over B0 but worsen RD and all three MMD ratios. Their 10.9–15.1% shortfalls make every predicted row `lower_bound_only=true`: S4 therefore fails to establish a usable predicted-budget method because degree prediction and greedy quota realization are confounded.

## S5/S6 probes and STPD pre-gate

| probe | init | held-out MAE | R² | Spearman |
|---|---|---:|---:|---:|
| S5 degree | scratch | 0.775±0.031 | 0.256±0.043 | 0.474±0.026 |
| S5 degree | warm | **0.766±0.028** | **0.290±0.045** | **0.484±0.035** |
| S6 residual | scratch | 1.918±0.046 | −0.010±0.043 | 0.243±0.074 |
| S6 residual | warm | **1.801±0.027** | **0.071±0.032** | **0.475±0.002** |

Warm representations contain rank signal, but S5's degree accuracy is insufficient for S4 and S6's
residual explains only 7% of held-out variance. The initial S6 smoke failed on spectral-norm state
keys; the corrected six full runs produced the table above.

STPD moderate paired accuracy: B0 0.911, Probe-S 0.728, Probe-P 0.714, RA 0.674, CN 0.672 and degree-parity 0.609. Its rule required Probe-S to beat the best control by 0.03; observed margin was −0.182, so the terminal verdict is **`edge_identity_killed`**. Overall, S2–S6 expose weak structural correlates but do not support a selected topology-conditioned method.

Provenance: H20 reports are `outputs/s2/s2_results.json`, `outputs/s3/*/report.json`, `outputs/s4/pass_b/s4_results.json`, `outputs/s5_degree_probe/*/report.json`, `outputs/s6_residual_probe/*/report.json` and `outputs/stpd_pregate/report.json`. S4–S6 come from `s4-s6-budget-probes-d8` at `da6f5a5`; STPD from `design/2026-08-19-stpd` at `d916d48`.
