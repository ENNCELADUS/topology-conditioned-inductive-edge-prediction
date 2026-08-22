# S-series topology diagnostics: definitions and results

**Current result:** S2 finds feature-aligned topology signal but its standalone generator loses to
B0; S3's pair-only control beats its set-conditioned residual. New S4 now tests the missing route:
generated joint graph latents augment frozen B0 representation instead of replacing B0.

All numbers are diagnostic (`formal=false`). S0/S0-R/S1 belong to the retired `V_hold` protocol and are archived lineage evidence, not current `V_val` results. S2/S3 use sampled test-node sets, so their density-matched GS/RD are comparable only within their tables, not to the official full-candidate protocol.

## Method map

| stage | definition | status / question answered |
|---|---|---|
| S0 | predict four low-order ego scalars from node features, append endpoint sums/differences to a ridge pair head | archived; only tests a narrow linear summary route |
| S0-R | predict CN/AA from the same linear pair features, append them to another ridge head | archived; algebraically near-redundant linear projection |
| S1 | post-process frozen B0 scores with IPF/greedy degree quotas or CN updates | archived; tests frozen-score allocation, not joint learning |
| S2 | generate an induced set topology from all node features: conditional GEN, unconditional UNC, shuffled-condition SHUF, deterministic DET; AE is reconstruction ceiling, B0 pair-score baseline, MARG quota baseline | completed, seed 0 |
| S3 | learn zero-init residual `logit=B0+delta`: RES uses a Set Transformer; PAIR uses parameter-matched independent node MLPs; DIAG runs the same set backbone on singleton sets; SHUF permutes set context | completed, three seeds |
| S4 | Set Transformer OT rectified flow generates GAE node latents; a zero-init adapter adds them to frozen B0 `pair_repr` | planned, three seeds; S2 size/test substrate plus S3 V_val boundary |
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

## S4 graph-latent flow residual

S4 keeps frozen B0 intact and learns one set-conditioned generative residual. It uses S2's
20–200 size/test substrate with S3's V_val boundary. A feature-conditioned Set Transformer
OT rectified flow generates aligned GAE node latents; a symmetric zero-init adapter adds those
latents to B0's pre-head `pair_repr`, and the original B0 output head makes the edge decision.
The complete data, architecture, objective, and terminal rule are fixed in
[`S4 latent-flow residual design`](../superpowers/specs/2026-08-21-s4-latent-flow-residual-design.md).

## STPD pre-gate

STPD moderate paired accuracy: B0 0.911, Probe-S 0.728, Probe-P 0.714, RA 0.674, CN 0.672 and degree-parity 0.609. Its rule required Probe-S to beat the best control by 0.03; observed margin was −0.182, so the terminal verdict is **`edge_identity_killed`**. This verdict does not test S4's generated-latent residual.

Provenance: completed H20 reports are `outputs/s2/s2_results.json`, `outputs/s3/*/report.json`, and `outputs/stpd_pregate/report.json`. STPD comes from `design/2026-08-19-stpd` at `d916d48`; new S4 has no result yet.
