# S2 — Set-Conditioned Joint Latent Topology Generation: results

**Verdict: negative; route closed.** Conditioning on $X_S$ carries a real but weak
node-aligned topology signal (GEN > SHUF/UNC), yet the generative set model is
strictly dominated by the frozen B0 pairwise scorer on every node-aligned metric,
and its draws exhibit no joint structure beyond their own marginals. The Stage-A
autoencoder ceiling is high, so the failure lies in the features→topology
conditional map, not in the latent bottleneck — this is an honest negative, not
an S0-style weak-proxy artifact.

Design: `docs/tmp/s2_set_generation_experiment.md`. Implementation:
`src/experiments/s2_latent_topology/` (commit `730d197`). Run: 2026-08-17 on one
H20, `outputs/s2` (full decoder) and `outputs/s2_act` (activity-only ablation);
`evidence_class=diagnostic`, `formal:false`, seed 0, K=32 draws, 32 flow steps.
Evaluation: `test_node_buckets.pkl`, 10 sizes × 50 node-disjoint test sets;
optional full-region eval not run. B0 scores: published v3.1 candidate universe
(checkpoint `e092537d8cf1e208`), fp32-validated.

## Primary endpoint — node-aligned identification (macro over sizes, full run)

| arm | GS↑ | Spearman↑ | hub recall↑ | AUPRC↑ |
|---|---|---|---|---|
| GEN | 0.242 | 0.185 | 0.127 | 0.263 |
| UNC | 0.171 | −0.009 | 0.105 | 0.185 |
| SHUF | 0.169 | −0.010 | 0.093 | 0.185 |
| DET | 0.108 | 0.094 | 0.108 | 0.204 |
| AE (ceiling) | 0.845 | 0.983 | 0.899 | 0.969 |
| B0 (frozen pair scores) | 0.282 | 0.307 | 0.190 | 0.304 |
| MARG (see caveats) | 0.245 | — | 0.142 | 0.304 |

GEN's GS confidence intervals separate from SHUF's at sizes ≥ 120 (e.g. size 200:
GEN [0.117, 0.191] vs SHUF [0.062, 0.090]); at small sizes they overlap. Per-size
tables: `s2_set_generation_tables.md` (from `outputs/s2*/s2_tables*.md` on H20).

## Topology five-tuple per arm (GS↑, RD→1, MMD ratios↓; macro, full run)

| arm | GS | RD | degree | clustering | spectral |
|---|---|---|---|---|---|
| GEN | 0.242 | 0.839 | 6.69 | 4.17 | 7.24 |
| UNC | 0.171 | 0.839 | 5.48 | 3.59 | 6.25 |
| SHUF | 0.169 | 0.839 | 6.43 | 3.94 | 7.21 |
| DET | 0.108 | 0.513 | 23.98 | 17.35 | 28.31 |
| AE | 0.845 | 0.839 | 4.45 | 1.12 | 4.75 |
| B0 | 0.282 | 0.837 | 3.55 | 2.78 | 5.98 |

RD ≈ 0.839 uniformly (including AE) is the disclosed fixed asymmetry: predicted
graphs assemble non-self pairs only while bucket references keep self-loops; it is
not an arm-quality signal. Free-running density (secondary): GEN implies 0.52× the
true edge count, DET 1.72× (its RD 0.513 reflects that miscalibration surviving
matched assembly).

## Coherence — is the joint model actually joint?

No. Across-draw activity variance stays ≈ 198 at every size (the per-node
posterior over $a_i$ never sharpens; the unconditional prior spans 119–178).
Mean per-draw GS is below the across-draw-marginal GS at every size, and
draw-level triangle / clustering / degree-variance statistics track the
independently-thresholded mean-prob statistics closely. The sampled joint carries
no coherent structure beyond its marginals — the mechanism S1 could not test,
now tested directly and dead.

## Activity-only ablation (`outputs/s2_act`)

Only the ceiling is interpretable: AE-act GS 0.530 / Spearman 0.992 / AUPRC 0.585
vs full-decoder AE 0.845 / 0.983 / 0.969 — the 16-dim role geometry
$z_i^\top z_j$, not the degree field, carries most expressible structure. The
act-run conditional arms diverged in training (prior best epoch 2, DET best
epoch 3, later losses ~10¹⁵), so GEN-act/DET-act numbers are from barely-trained
models and are not evidence.

## Caveats

- **MARG regressor collapsed:** train/val loss flat to four decimals across all
  20 epochs (val 246.4142 in both runs), Spearman undefined (constant
  predictions), free density 0 — the arm degenerates to B0 scores + uniform
  quotas. GEN-vs-MARG is not quotable; the clean B0 arm is the valid
  independence baseline, and it already dominates GEN.
- Bucket GS/RD here use S2's density-matched in-set assembly, not the official
  candidate-universe protocol; compare arms within this table only.

## Interpretation against the pre-registered map

1. GEN > SHUF/UNC — $X_S$ carries joint-topology signal (real, small).
2. GEN < B0 everywhere — set-level joint modeling adds nothing over independent
   pairwise prediction; the route's core promise is refuted, not merely unproven.
3. GEN > DET — sampling beats the deterministic twin, but both sit far below B0.
4. GEN/AE ≈ 29% of expressible GS and 19% of expressible degree alignment —
   the bottleneck is expressive; the conditional map is the failure.
5. act vs full — role geometry dominates the degree field at the ceiling;
   the conditional act comparison is void (training instability).

Together with S0/S0-R/S1, both the pair-conditioned and set-conditioned
generation formulations are now closed at the diagnostic level.
