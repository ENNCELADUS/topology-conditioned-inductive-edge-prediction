# Experiments: Topology-Conditioned Inductive Edge Prediction

**Status (2026-09-02):** paper-style protocol and evidence record. Section 2 records the Pairwise baseline, Full-Ego Oracle ceiling, and completed KD1; KD2--KD4 and kd_struct await held-out tests.

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
Raw train-side, `V_val`, and test labeled sets are balanced 1:1; effective train need not be. `V_val` is only pair-disjoint; test is node-disjoint.

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

**Evidence classes.** A *comparator* is a frozen model or score artifact evaluated without changing
its checkpoint or opening new test-dependent choices; a *formal result* follows this fixed pre-test protocol.

**Model selection.** Training early-stops on total val loss (patience 10): the val task BCE plus
each active KD term's val counterpart at its training weight. The published checkpoint is chosen
independently, by mean rank over V_val AUPRC plus the five bucket-topology metrics (§1.3).

**Topology threshold.** Selected on the `V_val` 20--200-node sampled-set pair union. Define
`D_RD(t)` as the size-bucket macro-average of mean `|log RD|`; among finite atomic candidates,
let `t_min=argmin_t D_RD(t)` and retain exactly those with `D_RD(t) <= D_RD(t_min) + SE_t`,
where `SE_t` is the size-stratified paired-bootstrap SE of the difference. Among survivors
minimize `D_shape=(1/3) sum_s log(max(r_s(t), epsilon))`, `epsilon=1e-12`, then break ties
toward the larger threshold; empty-prediction candidates have `D_RD=+inf`. Freeze before test.

**Classification threshold.** Selected separately as the max-F1 logit threshold on the balanced
`V_val` classification rows (`val_cls`) and frozen before test. It serves only Accuracy/F1/MCC;
AUROC/AUPRC use raw logits, ECE/Brier use the raw sigmoid probabilities, and no logit shift
stands in for calibration.

**Test replay.** Test topology scores only its sampled-set pair union plus support-only rows for
the grounded arm; the frozen topology threshold replays unchanged as the one reported operating
point (self-loops included), and the frozen classification threshold replays on the balanced test rows.

**Uncertainty.** Runs are single-seed (disclosed). Arm-versus-baseline deltas report
size-stratified paired-bootstrap intervals over test sampled sets and test rows.

### 1.3 Metrics

Pairwise: AUROC and AUPRC (threshold-free); Accuracy, F1, and MCC at the frozen max-F1
threshold; ECE and Brier on raw probabilities. Detailed edge tables also retain class
balance, uncertain-negative disclosure, and the completed easy, hard, degree-corrected,
full-universe, and PA-null negative-regime controls, which qualify every edge claim.

Topology (five numbers, always reported together, directions in headers): BFS-macro
graph similarity (GS ↑, edge-set Dice/F1), BFS-macro relative density (RD → 1), and
degree / clustering / spectral MMD ratios (↓; the denominator is the deterministic
real-vs-real floor, so ratio 1 is that floor). Descriptors retain self-loops.

### 1.4 Compared methods

| Arm | KD signal | Teacher | Role |
|---|---|---|---|
| Pairwise baseline (B0) | none | — | frozen endpoint-only comparator |
| kd_logit | GLNN-style pointwise soft-target BCE | PMA(4) Full-Ego Oracle row bank | attribution control |
| kd_ranking | LLP-style rank + distribution matching over context banks | PMA(4) Full-Ego Oracle context bank | primary transfer test (RQ2) |
| kd_representation | pair-representation cosine | PMA(4) Full-Ego Oracle row bank | representation-matching arm |
| kd_gram | SPKD-style cosine-Gram relational match (Tung & Mori 2019) | PMA(4) Full-Ego Oracle row bank | relational-geometry arm |
| kd_generation | pair-latent generative head | PMA(1) Full-Ego Oracle latent bank | generative test |
| kd_struct | auxiliary head, MSE to z-scored truth-graph descriptors (CN, degrees, Jaccard, Adamic-Adar; queried partner masked) | none: targets from the training graph | descriptor arm and content→structure ceiling (§4) |
| Oracles | observed topology | Full-Ego graph → GRIT → PMA | diagnostic ceilings |

![Teacher and student architecture](results/kd_rep_audit/teacher_architecture.svg)

### 1.5 Training, HPO, and reproducibility

The student is V3.1: d_model 512, 3 encoder + 3 cross-attention layers, 8 heads, rich pooling
(mean/attn/max/gated), pair_context_gated readout with abba_max order aggregation, label smoothing
0.05. Optimization: AdamW, lr 1e-4, weight decay 0.05, onecycle, 25 epochs, 1,024 pairs per batch, clip 1.0, bf16 DDP.

HPO: a Phase-0 grid (24 runs) fixed per-arm incumbents (kd_logit_w100, kd_rank_wr0p1_wd1,
kd_gram_w1, kd_rep_w0p1); kd_rank continues with a 16-trial constrained MO-TPE study
(GS ↑, geometric-mean MMD ↓, soft constraint `|log RD| <= 0.05`) over w_rank × w_dist ×
context bank (hops × negative-sampling rate) × margin. The best configuration per arm runs
the held-out test protocol exactly once; provenance and HPC completion are rules 5--6 (§5).

## 2. Main results — edge and assembled topology

Table 2 (pairwise, best HPO per arm) and Table 3 (the five topology numbers at each arm's single
frozen V_val-selected threshold) are reported together.

| Arm | AUROC | AUPRC | Accuracy | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Full-Ego Oracle | 0.9519 | 0.9566 | 0.8711 | 0.8776 | 0.7464 |
| Pairwise baseline | 0.7067 | 0.7316 | 0.6261 | 0.4769 | 0.3072 |
| kd_logit (`kd_logit_w100`) | 0.7195 | 0.7417 | 0.6459 | 0.6670 | 0.2941 |
| kd_ranking | — | — | — | — | — |
| kd_representation | — | — | — | — | — |
| kd_gram | — | — | — | — | — |

| Arm | BFS GS | BFS RD | Degree | Clustering | Spectral |
|---|---:|---:|---:|---:|---:|
| Full-Ego Oracle | 0.6429 | 0.9955 | 2.667 | 1.875 | 5.207 |
| Pairwise baseline | 0.3672 | 0.3221 | 21.03 | 18.49 | 28.39 |
| kd_logit (`kd_logit_w100`) | 0.4048 | 0.4248 | 15.63 | 13.07 | 22.06 |
| kd_ranking | — | — | — | — | — |
| kd_representation | — | — | — | — | — |
| kd_gram | — | — | — | — | — |

## 3. Ablations and sensitivity

## 4. Analysis

**Learning curves.** Completed seed-0 HPO-winner `V_val` surfaces; the dotted marker is the selected epoch..

**KD1 `kd_logit_w100` (selected epoch 25)** ![KD1 loss curves](results/kd1_kd_logit/learning_curves.png) ![KD1 V_val topology curves](results/kd1_kd_logit/validation_topology_curves.png)
Faithful to GLNN (Zhang et al. ICLR 2022): BCE to the teacher sigmoid equals the paper's KL up to the teacher entropy.

The teacher is near-perfect on its own rows (AUPRC 0.984 train / 0.978 V_val, mean Bernoulli entropy 0.159 / 0.134 nats).

The student reaches logit correlation 0.913 on train rows but 0.866 on V_val, with KL above the entropy floor of 0.10 vs
0.23 nats; `w_logit` 1 and 100 give the same V_val AUPRC (0.927), so the weight is not the limit. It fits the teacher where
it has seen the rows and cannot elsewhere: GLNN's own low-$I(X;Y\mid E)$ regime, in which soft labels transfer only what
$(x_u,x_v)$ already reveals. Held-out: AUPRC +0.010 and GS +0.02 over control, RD and all MMD worse.

**KD2 `kd_rank_wr0p1_wd1` (selected epoch 22)** ![KD2 loss curves](results/kd2_kd_rank/learning_curves.png) ![KD2 V_val topology curves](results/kd2_kd_rank/validation_topology_curves.png)
Retired incidental-batch objective, not strict LLP; the strict-LLP MO-TPE study (§1.5) supersedes it and has not been audited. No reading is drawn from this surface.

**KD3 `kd_gram_w1` (selected epoch 10)** ![KD3 loss curves](results/kd3_kd_gram/learning_curves.png) ![KD3 V_val topology curves](results/kd3_kd_gram/validation_topology_curves.png)
SPKD-style (Tung & Mori ICCV 2019) with feature-cosine rows and an off-diagonal mean; `w_gram` 0.01--100 spans KD-gradient
dominance from 0.03× to above the task gradient, so the paper's γ has no untested analogue. 

The target $t_{uv}$ (below) has a rank-2 cosine Gram ($R^2=0.996$) that correlates $-0.86$ with teacher probability differences, so KD3 relationally
re-encodes KD1's target. The converged Gram loss (0.066 train / 0.074 V_val) equals a label-block-mean
predictor (0.067 / 0.074; constant predictor 0.119 / 0.138): label-level fit is reached and nothing beyond it is demonstrated.

**KD4 `kd_rep_w0p1` (selected epoch 14)** ![KD4 loss curves](results/kd4_kd_rep/learning_curves.png) ![KD4 V_val topology curves](results/kd4_kd_rep/validation_topology_curves.png)
Per-row cosine to the same $t_{uv}$, so the same loss geometry applies. The cosine loss plateaus at 0.195--0.206 on train
and V_val alike (cosine ≈ 0.80), the score of a constant vector along the teacher mean direction (median row cosine to it
0.82 / 0.83): the student learned the shared offset, not the rows. The fused pre-head vector is no better: more logit-aligned
(top-axis corr 0.99), structural probe equal to content+logit (CN 0.90 vs 0.85), so distilling it is KD1 plus content self-distillation; no PMA(1) or fused rerun.

**What $t_{uv}$ carries, and what the student can reach.** $t_{uv}$ is the teacher's topology-branch pooled vector before
fusion (§1.4 figure). Linear ridge probes on held-out train rows ([audit](results/kd_rep_audit/README.md); V_val within 0.02) answer two questions.

| Probe | Value | Reading |
|---|---:|---|
| variance share of the top axis of $t_{uv}$, and its correlation with the teacher logit | 0.93, $-0.96$ | one direction dominates, and it is the edge decision |
| $R^2$ of $t_{uv}\to$ common neighbours, degree, Jaccard, Adamic-Adar | 0.96--0.99 | structure is still readable, from the low-variance tail |
| $R^2$ of $(x_u,x_v)\to t_{uv}$ | 0.40--0.47 | under half of the vector is a function of the student's input |
| $R^2$ of $(x_u,x_v)\to$ the same descriptors | 0.2--0.7 | lower bound on the structure any student can recover from content |

1. *The structure is there, on axes the losses ignore.* $t_{uv}$ passes the hidden ego graph's descriptors through almost
   losslessly (input pass-through, not learned abstraction), but cosine and Gram losses weight directions by variance, so
   they match the logit axis and miss the tail. KD3/KD4 failed on loss geometry, not on missing information.
2. *Only the content-predictable part can transfer.* The student never sees $t_{uv}$ at test time; representation KD moves
   at most the part of $t_{uv}$ that is a function of $(x_u,x_v)$: linearly 0.40--0.47 of the vector, 0.2--0.7 per descriptor.
   Being linear, these lower-bound the nonlinear ceiling, measured next; near these numbers, no whitened or descriptor-level KD can move topology.
