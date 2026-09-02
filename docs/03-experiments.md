# Experiments: Topology-Conditioned Inductive Edge Prediction

**Status (2026-09-01):** paper-style experiment protocol and evidence record. Section 2
numbers await fresh runs under the current protocol; legacy and diagnostic evidence is
labeled wherever cited.

## 1. Experimental setup

### 1.1 Dataset and splits

| Object | Nodes | Loopless positive graph edges | Role |
|---|---:|---:|---|
| Full reference graph | 10,090 | 122,092 | positive graph truth only |
| Train-side substrate (`train_graph.pkl`) | 8,072 | 47,762 | original train⁺∪val⁺ graph and pair pool, before the internal split |
| ↳ Effective train | within the same 8,072 | 38,234 | excludes `V_val`-internal rows; outside–outside and cross-boundary pairs train |
| ↳ `V_val` region | 2,553-node subset | 9,528 | all internal pairs withheld |
| Test graph | disjoint 2,018 | 30,128 | held out until final evaluation |
| Loopless test candidate universe | same 2,018 | — | all 2,035,153 unordered distinct-node pairs |

Train and test nodes are disjoint; the primary split strategy is `breadth_first`.
`V_val` is grown by K=5 dispersed-seed hashed-frontier BFS on the substrate's loopless giant component, stopped at 20% induced loopless edges; all graph-edge counts are loopless positives.
Raw train-side, `V_val`, and test labeled sets are balanced 1:1; effective train need not be. `V_val`
is only pair-disjoint; test is node-disjoint.

Each node carries a frozen intrinsic token sequence (≤1024 tokens × 1536 dims); F0 is
its fp32 mean-pooled vector. Negatives are the fixed benchmark's balanced samples. Graph truth is observation-biased, so uncertain negatives are disclosed.

### 1.2 Evaluation protocol

**Task contract.** For a queried pair $u,v\in V_{\mathrm{test}}$ the model receives
exactly $(x_u,x_v)$ and returns the symmetric probability
$\widehat A_{uv}=P(Y_{uv}=1\mid x_u,x_v)$. No observed test edge, neighbor identity,
retrieval result, degree, or graph statistic is task input; training topology may
supervise a representation or objective. After scoring a pair universe, predictions are
assembled into $\widehat G_\tau$ only for evaluation. Inferred topology is intermediate
context, never the prediction target and never generic graph generation.

**Evidence classes.** A *comparator* is a frozen model or score artifact evaluated
without changing its checkpoint or opening new test-dependent choices; a *formal result*
follows this fixed pre-test protocol and can carry paper claims.

**Model selection.** Training early-stops on total val loss (patience 10): the val
task BCE plus each active KD term's val counterpart at its training weight. The
published checkpoint is chosen independently, by mean rank over V_val AUPRC plus the
five bucket-topology metrics (§1.3); usability is judged from `metrics.jsonl`.

**Threshold rule.** The single fixed threshold selects on the `V_val` 20--200-node
sampled-set pair union. Define `D_RD(t)` as the size-bucket macro-average of mean
`|log RD|`; among finite atomic candidates, let `t_min=argmin_t D_RD(t)` and retain
exactly those with `D_RD(t) <= D_RD(t_min) + SE_t`, where `SE_t` is the size-stratified
paired-bootstrap SE of the difference. Among survivors minimize
`D_shape=(1/3) sum_s log(max(r_s(t), epsilon))`, `epsilon=1e-12`, then break ties
toward the larger threshold; empty-prediction candidates have `D_RD=+inf`. Freeze
before test.

**Test replay.** Test topology scores only its sampled-set pair union plus support-only
rows for the grounded arm; the frozen threshold replays unchanged as the one reported
operating point (self-loops included). Test logits are calibrated by subtracting the
frozen threshold so it sits at probability 0.5.

**Uncertainty.** Runs are single-seed (disclosed). Arm-versus-control deltas report
size-stratified paired-bootstrap intervals over test sampled sets and test rows.

### 1.3 Metrics

Pairwise: AUROC and AUPRC (threshold-free); Accuracy, F1, and MCC at calibrated
probability 0.5; ECE and Brier for calibration. Detailed edge tables also retain class
balance, uncertain-negative disclosure, and the completed easy, hard, degree-corrected,
full-universe, and PA-null negative-regime controls, which qualify every edge claim.

Topology (five numbers, always reported together, directions in headers): BFS-macro
graph similarity (GS ↑, edge-set Dice/F1), BFS-macro relative density (RD → 1), and
degree / clustering / spectral MMD ratios (↓; the denominator is the deterministic
real-vs-real floor, so ratio 1 is that floor). Descriptors retain self-loops.

### 1.4 Compared methods

| Arm | KD signal | Teacher | Role |
|---|---|---|---|
| control (B0) | none | — | frozen endpoint-only comparator |
| kd_logit | GLNN-style pointwise soft-target BCE | PMA(4) Full-Ego Oracle row bank | attribution control |
| kd_rank | LLP-style rank + distribution matching over context banks | PMA(4) Full-Ego Oracle context bank | primary transfer test (RQ2) |
| kd_rep | pair-representation cosine | PMA(4) Full-Ego Oracle row bank | representation-matching arm |
| kd_gram | Gram-matrix relational match | PMA(4) Full-Ego Oracle row bank | relational-geometry arm |
| kd_gen | pair-latent generative head | PMA(1) Full-Ego Oracle latent bank | generative test (RQ3); det/EDM complete |
| Oracles | observed hidden topology | — | diagnostic ceilings |

All arms share the identical V3.1 student, data, feature packs, and threshold rule;
only the KD signal differs. Teachers read hidden topology at training time only, and
the queried edge is always masked from its own structural context. Ladder status:
B2/B3 latent arms are unrun; the B4 retrieval-grounded scaffold is implemented as a
separately labeled support condition with a disclosed support universe and has no
current formal result.

### 1.5 Training, HPO, and reproducibility

The student is V3.1: d_model 512, 3 encoder + 3 cross-attention layers, 8 heads, rich
pooling (mean/attn/max/gated), pair_context_gated readout with abba_max order
aggregation, label smoothing 0.05. Optimization: AdamW, lr 1e-4, weight decay 0.05,
onecycle schedule, 25 epochs, 1,024 pairs per batch, gradient clip 1.0, bf16 DDP.

HPO: a Phase-0 grid (24 runs) fixed per-arm incumbents (kd_logit_w100,
kd_rank_wr0p1_wd1, kd_gram_w1, kd_rep_w0p1). kd_rank continues with a 16-trial
constrained MO-TPE study (objectives GS ↑ and geometric-mean MMD ↓; soft constraint
`|log RD| <= 0.05`) over w_rank × w_dist × context-bank configuration (hops ×
negative-sampling rate) × margin. The best configuration per arm runs the held-out test
protocol exactly once. Provenance recording and HPC completion requirements are
protocol rules 5--6 (§6).

## 2. Main results — edge and assembled topology

Table 2 (pairwise, best HPO per arm) and Table 3 (the five topology numbers at each
arm's single frozen V_val-selected threshold) are reported together, never one family
alone. Prose reports per-arm deltas against the control and the joint verdict of
protocol rule 7: a selected method must improve assembled topology without an
unacceptable edge-metric loss and survive its coupling ablation.

| Arm | AUROC | AUPRC | Accuracy | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| control (B0) | — | — | — | — | — |
| kd_logit | — | — | — | — | — |
| kd_rank | — | — | — | — | — |
| kd_rep | — | — | — | — | — |
| kd_gram | — | — | — | — | — |

| Arm | BFS GS | BFS RD | Degree | Clustering | Spectral |
|---|---:|---:|---:|---:|---:|
| control (B0) | — | — | — | — | — |
| kd_logit | — | — | — | — | — |
| kd_rank | — | — | — | — | — |
| kd_rep | — | — | — | — | — |
| kd_gram | — | — | — | — | — |

**Status.** All rows await fresh formal runs under the protocol above. The PMA(1)
oracle in §3.1 is a current-protocol diagnostic and never enters these formal tables;
the earlier D1--D8 KD results are retired as confounded by the removed anchor-context
sampler.

## 3. Topology headroom and diagnostics

### 3.1 Oracle ceilings

| Model | AUROC | AUPRC | Accuracy | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Pairwise baseline | 0.7067 | 0.7315 | 0.6083 | 0.3987 | 0.3020 |
| PMA(1) Full-Ego Oracle | 0.9519 | 0.9566 | 0.7783 | 0.7184 | 0.6151 |

| Model | BFS GS | BFS RD | Degree | Clustering | Spectral |
|---|---:|---:|---:|---:|---:|
| Pairwise baseline | 0.3896 | 0.4223 | 13.08 | 11.93 | 18.09 |
| PMA(1) Full-Ego Oracle | 0.6429 | 0.9955 | 2.667 | 1.875 | 5.207 |

**Finding.** PMA(1) preserves strong edge discrimination and assembled-graph fidelity
at the fixed `V_val`-selected threshold, but all three MMD ratios remain above the
real-vs-real floor. Hidden relational topology supplies substantial headroom; this
truth-consuming oracle is neither a formal result nor a fair baseline.

The Pairwise baseline is the legacy frozen V3.1 checkpoint `e092537d8cf1e208` and is
retained only as historical context, not as a protocol-matched comparator. **PMA(1)
Full-Ego Oracle** checkpoint `ea3e3ac37772acb2` (seed 0, selected epoch 9) pools
variable-length full ego topology inside GRIT with one PMA seed. `test_protocol_v6`
replays the fixed `V_val` logit threshold 4.4397 on held-out test and calibrates it to
probability 0.5. Its selected-epoch `V_val` AUPRC/GS/RD was 0.9727/0.7643/0.9980 with
MMD 16.34/1.88/14.81. Report:
`outputs/egostitch_e2e_stage1_v3/full_ego_teacher_pma1/diagnostic_test_report.json` (H20).

### 3.2 Feature-to-topology value (S0, S0-R)

S0 asked whether endpoint features predict per-node structural summaries and whether
those summaries help a pair classifier; S0-R asked the same for relational CN/AA.
Oracle node summaries lift held-out AUPRC by +0.0237 and feature-predicted summaries by
+0.0034 (CI includes 0); oracle CN/AA lifts it by +0.2438 (95% CI [0.2163, 0.2706])
while feature-predicted CN/AA changes it by $2.3\times10^{-8}$. Predictability is
uneven (degree $R^2=0.435$, clustering $R^2=0.051$, ego density ≈ unpredictable).
**Finding:** the relational ceiling is large but is not recoverable as extra per-pair
information from endpoint features — a deterministic predicted statistic is already a
function of $(x_u,x_v)$. This motivates training-time transfer (the KD arms) over
feature-derived statistics. Artifacts:
[s0_results.json](../outputs/s0/s0_results.json),
[s0r_results.json](../outputs/s0r/s0r_results.json).

### 3.3 Assembly coherence (S1-R, S1-H)

S1-R held frozen B0 scores fixed and tested node-coupled assembly: the corrected hard
node-aligned degree oracle reached BFS GS 0.439 / RD 0.622 / MMD 4.06/5.44/7.65 versus
0.390 / 0.423 / 13.03/11.86/18.07 for exact-count B0, closing 73.2% of the
frozen-B0-to-oracle clustering gap without changing pair scores (1.62% quota shortfall,
under the 2% limit); every legal post-hoc arm tested (predicted degrees, training-prior
multiset, CN updates, V_hold-fit coupling) failed. S1-H reran the hard-quota assembler
with no node-aligned identity: all three degree sources worsened the clustering gap
(closure −0.150 / −0.280 / −0.319), so the true degree multiset is insufficient and
node identity is the crux. **Finding:** coherent allocation of existing pairwise
evidence can close much of the pair-to-graph gap, but only with node-specific budgets —
motivating learned, non-factorized structure, not a legal post-hoc transform.
Diagnostics only; jointly trained models remain untested. Artifacts:
[s1_results.json](../outputs/s1/s1_results.json),
[s1_hard_decomposition.json](../outputs/s1/s1_hard_decomposition.json).

### 3.4 Archived evidence

G1 (B0, B0-alt, PA-null, hard/degree-corrected controls): the pair-to-topology gap
survives architecture and negative-regime checks. G2: edge-independent allocation has
measurable structural limits. G3 (`oracle_topo`, `oracle_blend`): hidden relational
topology provides assembled-graph headroom. The E2 report was retired in commit
`bd465f2` and remains in version history. Prior EgoStitch screens were cut and
carry no current formal result. GS provenance: §3.1 and S1 use the in-repo GS
implementation; archived G1 used the official evaluator (B0 GS 0.312151). RD and MMD
reproduce, but GS values must never be mixed across those provenances until the
implementation difference is reconciled.

## 4. Ablations and sensitivity

- **KD components (winning arm):** w_rank vs w_dist, margin, and context-bank
  configuration, harvested from the MO-TPE study surfaces — the LLP-style
  rank-loss/distribution-loss decomposition.
- **Threshold protocol:** the raw probability-0.5 operating point versus the frozen
  V_val-selected threshold, re-measured per sampled set on current arms. Historical
  free-run densities (retired KD arms 5.7--16.2× over-dense at 0.5) are motivation
  only and never comparison evidence.
- **Control equivalence:** the all-zero KD configuration is bit-identical to the
  control (tested), so ablation deltas are exact.

## 5. Analysis

- **Learning curves:** per arm, val AUPRC, `val_total_loss` (the stopping signal) and the
  five val topology metrics per epoch from `metrics.jsonl` — the joint-selection trajectory.
- **Calibration:** ECE and Brier before/after the calibrating logit shift.
- **Inference cost:** the student scores a pair from $(x_u,x_v)$ alone — no graph,
  retrieval, or neighbor access at inference — while every teacher requires hidden ego
  topology; report parameter counts and per-pair scoring throughput.
- **Failure modes:** MMD ratios remain above the real-vs-real floor even for oracles;
  uncertain negatives bound achievable edge metrics.

## 6. Protocol rules

1. Report pairwise and all five topology metrics together; no favorable nearby
   threshold or aggregate substitutes for the fixed rule.
2. Preserve provenance: Oracle, S0, S0-R, S1, and any test-informed follow-up remain
   `formal:false` even when their execution is valid.
3. Select the single fixed threshold on sampled `V_val`, freeze it, calibrate test
   logits so it sits at probability 0.5, then evaluate test once; never tune it on test.
4. Bind comparisons to the same frozen features, split, sampled sets, grounding
   support, self-loop convention, checkpoint, and fixed-threshold rule.
5. Validate score precision before analysis. Record checkpoint ID, artifact hashes,
   threshold/quota policy, random seed, code commit, and metric implementation.
6. Require completion markers, no `failure.json`, exited workers, complete outputs, and
   verified hashes before declaring an HPC experiment complete.
7. A selected method must improve assembled topology without an unacceptable
   edge-metric loss and must survive its topology-context or coupling ablation.

The S0, S0-R, and S1 artifacts retain `evidence_class=diagnostic`, the `breadth_first`
strategy, and their input-manifest digests; these records must remain attached to any
derived table.
