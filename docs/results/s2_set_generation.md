# S2/S3 — set-conditioned topology route: results

**Verdict: negative at both levels; route closed.** Conditioning on $X_S$ carries a
real but weak node-aligned topology signal (GEN > SHUF/UNC), yet the generative set
model is strictly dominated by the frozen B0 pairwise scorer on every node-aligned
metric, and its draws exhibit no joint structure beyond their own marginals. The
Stage-A autoencoder ceiling is high, so the failure lies in the features→topology
conditional map, not in the latent bottleneck — this is an honest negative, not an
S0-style weak-proxy artifact. **S3** then closed the incremental branch the S2 review
left open: a zero-init set residual *on top of* B0 is matched or beaten by its own
capacity- and pointwise-controls (§ S3 below).

## S2 — standalone latent-flow generator: results

Design: `docs/tmp/s2_set_generation_experiment.md`. Implementation:
`src/experiments/s2_latent_topology/` (commit `730d197`). Run: 2026-08-17 on one
H20, `outputs/s2` (full decoder) and `outputs/s2_act` (activity-only ablation);
`evidence_class=diagnostic`, `formal:false`, seed 0, K=32 draws, 32 flow steps.
Evaluation: `test_node_buckets.pkl`, 10 sizes × 50 node-disjoint test sets — every
forward is one 20–200-node sampled set; the full test region is never forwarded
(the whole-region code path is deleted: it is out of distribution for set models).
B0 scores: published v3.1 candidate universe (checkpoint `e092537d8cf1e208`),
fp32-validated.

### Primary endpoint — node-aligned identification (macro over sizes, full run)

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

### Topology five-tuple per arm (GS↑, RD→1, MMD ratios↓; macro, full run)

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

### Fixed threshold 0.5 per set (no density oracle; rerun 2026-08-19)

Eval-stage rerun on the same frozen checkpoints/seed/draws (`outputs/s2/s2_results.json`;
the prior payload is `s2_results.pre_t05.json`, and every pre-existing number reproduces
to ≤ 1.2e-6 — GPU forward fp32 tail only). Each 20–200-node set assembles its own
probabilities at p ≥ 0.5; MARG assembles by degree quota, not threshold, so it has no
row here (disclosed absence). Hard density = `(p ≥ 0.5)` edge count over the true count.

| arm | GS | RD | degree | clustering | spectral | hard density |
|---|---|---|---|---|---|---|
| GEN | 0.076 | 0.062 | 15.71 | 7.97 | 16.93 | 0.066 |
| UNC | 0.000 | 0.000 | 34.16 | 20.87 | 38.35 | 0.000 |
| SHUF | 0.034 | 0.062 | 16.38 | 8.19 | 17.47 | 0.066 |
| DET | 0.181 | 1.441 | 20.95 | 13.83 | 21.96 | 1.717 |
| AE | 0.835 | 0.785 | 5.20 | 1.03 | 5.07 | 0.936 |
| B0 | 0.232 | 0.563 | 9.06 | 3.98 | 10.70 | 0.669 |

Without the density oracle the generative arms produce almost nothing: GEN realizes
0.066× the true edge count (GS 0.242 → 0.076) and UNC is exactly empty, while AE stays
near truth (0.94×) and B0 degrades gently (0.67×, GS 0.282 → 0.232). DET's GS *rises*
(0.108 → 0.181) because its 1.7× over-density buys Dice recall — miscalibration, not
alignment. The matched-threshold ordering (AE ≫ B0 > GEN > SHUF > UNC) survives at 0.5,
so the density oracle was flattering every conditional arm rather than reordering them.

### Coherence — is the joint model actually joint?

No. Across-draw activity variance stays ≈ 198 at every size (the per-node
posterior over $a_i$ never sharpens; the unconditional prior spans 119–178).
Mean per-draw GS is below the across-draw-marginal GS at every size, and
draw-level triangle / clustering / degree-variance statistics track the
independently-thresholded mean-prob statistics closely. The sampled joint carries
no coherent structure beyond its marginals — the mechanism S1 could not test,
now tested directly and dead.

### Activity-only ablation (`outputs/s2_act`)

Only the ceiling is interpretable: AE-act GS 0.530 / Spearman 0.992 / AUPRC 0.585
vs full-decoder AE 0.845 / 0.983 / 0.969 — the 16-dim role geometry
$z_i^\top z_j$, not the degree field, carries most expressible structure. The
act-run conditional arms diverged in training (prior best epoch 2, DET best
epoch 3, later losses ~10¹⁵), so GEN-act/DET-act numbers are from barely-trained
models and are not evidence.

### Caveats

- **MARG regressor collapsed:** train/val loss flat to four decimals across all
  20 epochs (val 246.4142 in both runs), Spearman undefined (constant
  predictions), free density 0 — the arm degenerates to B0 scores + uniform
  quotas. GEN-vs-MARG is not quotable; the clean B0 arm is the valid
  independence baseline, and it already dominates GEN.
- Bucket GS/RD here use S2's density-matched in-set assembly, not the official
  candidate-universe protocol; compare arms within this table only.

### Interpretation against the pre-registered map

1. GEN > SHUF/UNC — $X_S$ carries joint-topology signal (real, small).
2. GEN < B0 everywhere — set-level joint modeling adds nothing over independent
   pairwise prediction; the route's core promise is refuted, not merely unproven.
3. GEN > DET — sampling beats the deterministic twin, but both sit far below B0.
4. GEN/AE ≈ 29% of expressible GS and 19% of expressible degree alignment —
   the bottleneck is expressive; the conditional map is the failure.
5. act vs full — role geometry dominates the degree field at the ceiling;
   the conditional act comparison is void (training instability).

Together with S0/S0-R/S1, both the pair-conditioned and set-conditioned
generation formulations are now closed as *standalone* generators. What that left
open — whether set context helps *beside* B0 rather than instead of it — is S3.

## S3 — set-conditioned residual on frozen B0: results

**Verdict: negative.** A zero-init set residual does lift the frozen B0 scorer
(+0.009 AUPRC, +0.005 GS, paired CIs excluding zero), but a parameter-matched
residual with the set encoder *deleted* buys the same or more, and the set arm's
residual degenerates to zero variance during training. C1 (uplift) is met
nominally; C2 (uplift attributable to third-party features) fails.

Design: `docs/tmp/s3_set_residual_plan.md`. Implementation:
`src/experiments/s3_set_residual/` (commits `394120c` → `1766525`). Run: 2026-08-18
on the two H20 containers, `outputs/s3/{res,pair,diag}_s{0,1,2}`;
`evidence_class=diagnostic`, `formal:false`, seeds 0–2, 9 arm runs of ~3 min each
after a 5 h shared cache build.

### Arms (one change each; frozen B0 v3.1 base `e092537d8cf1e208` everywhere)

Every arm adds `Δ` to the base logit through a zero-initialised final linear layer,
so at step 0 it reproduces B0 exactly; no arm reads adjacency at inference.

- **RES** — `Δ` from S2's set-transformer backbone; every node's hidden state sees
  the whole conditioning set (3,845,376 params).
- **PAIR** — set encoder deleted, replaced by a parameter-matched per-node MLP
  (3,845,384 params, ratio 1.0000). Capacity and region-density control.
- **DIAG** — RES's module and weights, each node processed as its own size-1 set so
  `h_i = f(x_i)`. The pointwise control S2 lacked.
- **SHUF** (eval-only, on RES checkpoints) — within-set permutation of the rows
  feeding the set encoder; `x_i,x_j` into B0 and into `Δ`'s pair slots stay true.
- **B0** — frozen scores, the γ=0 arm.

Training regions: BFS balls on `train_graph`'s loopless giant component with
V_val-internal pairs masked out of the loss; ≤40 epochs, patience 10, lr 3e-4,
weight decay 0.01, 32 regions/batch, grad clip 1.0. Selection mean-ranks V_val
ΔAUPRC plus the five V_val bucket-topology numbers across the top-3 checkpoints.
Evaluation reuses S2's harness — `test_node_buckets.pkl`, 10 sizes × 50
node-disjoint sets, density-matched in-set assembly, base logits from the
fp32-validated B0 candidate universe — so the B0 row below is bit-identical to the
B0 row in the S2 tables and the two experiments are directly comparable.

### Primary endpoint — arm macro (mean of 3 seeds; GS↑, RD→1, MMD ratios↓)

| arm | AUPRC↑ | GS↑ | RD→1 | degree↓ | clustering↓ | spectral↓ | Spearman↑ | hub recall↑ |
|---|---|---|---|---|---|---|---|---|
| B0 (frozen) | 0.3035 | 0.2821 | 0.837 | 3.55 | 2.78 | 5.98 | 0.307 | 0.190 |
| RES | 0.3120 | 0.2868 | 0.839 | 3.46 | 2.71 | 6.00 | 0.337 | 0.194 |
| PAIR | 0.3154 | 0.2907 | 0.839 | 3.52 | 2.89 | 5.98 | 0.351 | 0.208 |
| DIAG | 0.3122 | 0.2897 | 0.839 | 3.35 | 2.90 | 5.91 | 0.347 | 0.204 |

Per-seed AUPRC: RES 0.3152 / 0.3059 / 0.3149, PAIR 0.3144 / 0.3153 / 0.3165,
DIAG 0.3103 / 0.3137 / 0.3125. The between-seed spread of a single arm (0.0093 for
RES) exceeds every between-arm gap.

### Paired contrasts (pooled over the 500 buckets, 95% CI, seeds 0 / 1 / 2)

| contrast | ΔAUPRC | ΔGS |
|---|---|---|
| RES − B0 | +0.0117 / +0.0024 / +0.0113 — all exclude 0 | +0.0072 / +0.0007 / +0.0061 — all exclude 0 |
| RES − PAIR | +0.0008 ns / −0.0094 / −0.0017 | −0.0031 / −0.0061 / −0.0024 — RES worse 3/3 |
| RES − DIAG | +0.0050 / −0.0078 / +0.0023 | +0.0016 / −0.0086 / −0.0018 |

All nine runs share one bucket sample (the B0 arm is bit-identical across them), so
these delta-of-deltas are paired per bucket. RES never beats both controls in the
same seed, and loses to the set-free PAIR arm on GS in all three.

### Controls and telemetry

- **Zero-init gate passed exactly.** A fresh zero-init model reproduces B0 with max
  abs diff 0.0 at every bucket size (`res_s0/sanity_report.json`) — the residual is
  correctly wired as `base + delta`.
- **SHUF is vacuous here, by construction.** `_shuffle_features_within_sets`
  permutes row order *inside* each set and the backbone is permutation-invariant, so
  SHUF tracks RES to ~4 decimals (AUPRC 0.3153 vs 0.3152 at seed 0). It is a wiring
  check, not a falsification control; PAIR and DIAG carry the anti-claims.
- **RES's residual dies; the controls' does not.** `delta_variance` reaches 0.0000
  by the final epoch in all three RES seeds — published epochs are early (4 / 8 / 3)
  — while PAIR and DIAG end at 0.5–27.3. RES's best V_val ΔAUPRC (0.0099–0.0161) is
  about half PAIR's (0.0307–0.0356) and DIAG's (0.0282–0.0295). Set attention
  optimises *toward reproducing the base scorer*. The plan's C1 asked for a CI
  covering zero **with healthy training**; that branch was not reached, so the
  negative rests on the control contrasts, not on a clean null.
- **Free-running calibration gets worse.** ECE improves (B0 0.106 → 0.088–0.095) but
  the implied edge count falls to 0.29–0.57× true (B0 0.84×) and free-running
  density from 0.891 to 0.25–0.58. Matched-assembly RD stays pinned at ≈0.839 for
  every arm — the same fixed self-loop asymmetry disclosed for S2.

### Interpretation against the claim map

1. **C1 — uplift over B0: nominally yes, mechanically empty.** Every arm, including
   the ones with no set access, clears the same bar; the lift is capacity plus
   recalibration.
2. **C2 — third-party features: refuted.** RES ≤ PAIR and RES ≈ DIAG. Neither the
   set encoder nor third-party rows explain the gain.
3. **Topology does not move.** GS 0.282 → 0.287–0.291, RD 0.837 → 0.839, ratios
   3.55/2.78/5.98 → 3.35–3.52 / 2.71–2.90 / 5.91–6.00. Edge-set identity is as inert
   here as under every KD mechanism in `b1_kd_arms.md`.
4. **Downstream gating.** STOCH was gated on a positive C1 and does not earn a run.
   GRAM (a gauge-invariant retrain of S2's DET twin, answering *why* S2 failed) is
   the only probe this route still has open.
