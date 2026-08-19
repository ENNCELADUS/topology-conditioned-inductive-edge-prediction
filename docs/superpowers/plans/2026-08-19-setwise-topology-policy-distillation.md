# Setwise Topology Policy Distillation (STPD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a whole-universe, recurrent knowledge-distillation method that transfers a privileged teacher's node-aligned topology-repair policy into a deployable student, with a strict feature-only tier and a separately reported training-memory-supported tier.

**Architecture:** STPD processes the entire supplied protein universe once, initializes a sparse graph from a learned content-and-budget prior, and recurrently repairs that graph. The student maintains sparse graph state, node activity budgets, soft role/community state, motif obligations, and global topology tokens; a whole-universe proposer supplies a sparse frontier, and a differentiable budget-constrained transport operator selects coordinated ADD/REMOVE/KEEP actions. A privileged training-only teacher sees clean and corrupted training topology and distills both pre-transport action priorities and the post-transport coordinated repair plan.

**Tech Stack:** Python 3.12, PyTorch, PyTorch sparse tensors, Hugging Face Accelerate/DDP, NumPy, NetworkX for offline target construction, SciPy for Hungarian assignment, the repository's packed ESM feature pipeline, validation-region split, score artifacts, and edge/graph evaluation code.

**Spec:** Sections 1–22 of this document are the approved method and experiment specification. Sections 23–34 are the implementation plan that realizes that specification.

## Global Constraints

- Preserve the benchmark's node-disjoint train/test partition and all existing `V_val` quarantine rules.
- The strict tier receives the complete supplied unseen-node feature set `X_S` and no observed test edge, test degree, hidden edge count, or test-derived graph statistic.
- The memory-supported tier may retrieve only training nodes and training edges using intrinsic features; it is always reported separately from the strict tier.
- The supplied set `S` is the whole prediction universe, not a hidden-topology BFS region and not a query-specific context set.
- The reported output remains a symmetric probability for every candidate pair. Recurrent graph state is an internal inference variable.
- Unknown PPI nonedges remain unlabeled. Only observed positives and deliberately injected, provenance-known negatives may define trusted ADD/REMOVE targets.
- The student may never retain an `n × n × d` learned pair-state tensor. Full-universe pair utilities are scanned in chunks; expensive policy reasoning is sparse.
- All candidate chunks in one repair round must read the same frozen round state so chunking changes memory use, not the mathematical prediction.
- Preserve deterministic seeds and existing bf16/fp32 boundaries. Transport, budget, probability, and final-logit arithmetic runs in fp32.
- Continue reporting edge metrics and the complete topology five-tuple together: GS, RD, degree-MMD ratio, clustering-MMD ratio, and spectral-MMD ratio. Global and BFS-macro GS/RD remain separately named.
- Continue using the repository's validation-only checkpoint selection and frozen held-out test protocol. Do not tune method choices on test results.
- Provenance metadata is required; hash or text-contract gates are not. Fail closed only on non-finite state, DDP disagreement, data-boundary violations, malformed artifacts, and I/O failure.
- STPD is a new model family. Do not retrofit its recurrent state or memory route into `v3_1` or the completed B1 KD arms.

---

# Part I — Research and Method Specification

## 1. Why a New KD Object Is Required

The existing evidence distinguishes topology value from topology transferability:

1. `docs/results/results.html` shows a large Full-Ego privileged-topology ceiling over the endpoint-only baseline at both edge and assembled-graph levels.
2. `docs/results/b1_kd_arms.md` shows that conventional KD transfers selected statistics but not the desired graph realization: listwise KD mainly moves ranking/calibration, pair-geometry KD mainly moves MMD ratios, and edge-set identity barely moves.
3. `docs/results/s2_set_generation.md` shows that whole-set features carry real but weak node-aligned topology signal. A one-shot terminal latent is nevertheless dominated by B0 and its samples carry little dependence beyond their marginals.

Therefore STPD does not distill another static logit or arbitrary embedding. It distills the conditional action rule

\[
(A_t, X_S) \longmapsto \text{which coordinated topology repair should occur next},
\]

including the node budgets and higher-order obligations that make edge decisions dependent.

## 2. Contribution and Claim Boundary

STPD's intended contribution is narrow:

> A privileged graph teacher distills a corruption-conditioned, whole-set topology-repair policy into a student whose inference graph is internally generated, with candidate edges coupled through recurrent graph state and a shared budget-constrained allocation operator.

The paper must not claim the first graph-free KD model, feature-to-topology latent, conditional graph generator, or generative link predictor. The new mechanism is the combination of:

- whole-universe set conditioning;
- exact node-aligned repair support;
- state-dependent teacher priorities;
- explicit activity/community/motif obligations;
- post-allocation plan distillation;
- recurrent sparse graph updates;
- and a separate analogical training-topology memory condition.

## 3. Revised Two-Tier Formulation

Let the supplied prediction universe be

\[
S=\{1,\ldots,n\},\qquad X_S\in\mathbb R^{n\times d_x},
\]

with fixed candidate pairs \(\mathcal Q_S\).

### 3.1 Strict STPD

\[
\widehat P_S,\widehat A_S
=
F_\theta(X_S;R_S=\varnothing).
\]

No observed edge over `S` is an input. The whole set is legal because membership is supplied independently of hidden topology.

### 3.2 Memory-supported STPD

\[
\widehat P_S,\widehat A_S
=
F_\theta(X_S;R_S=\mathcal M_{\mathrm{train}}),
\]

where `M_train` contains only intrinsic features and topology from `G_train`. Target-to-memory alignment uses intrinsic features only and is cross-fitted during training.

### 3.3 Output

The canonical output is the final symmetric pair-probability vector

\[
\widehat p_{ij}=\widehat p_{ji},\qquad (i,j)\in\mathcal Q_S.
\]

The final internal graph is an additional model artifact, not a replacement for pair probabilities.

## 4. Recurrent State

At repair round `t`, the student state is

\[
\mathcal Z_t=(A_t,H_t,B_t,C_t,M_t,G_t,E_t^{\mathrm{meta}}),
\]

where:

- \(A_t\): sparse soft graph over `S`;
- \(H_t\in\mathbb R^{n\times d_h}\): node state;
- \(B_t\in\mathbb R^n\): signed remaining activity budget;
- \(C_t\in[0,1]^{n\times K_c}\): soft community/role assignment;
- \(M_t\in\mathbb R^{n\times4}\): triangle, open-wedge, intra-community, and cross-community obligations;
- \(G_t\in\mathbb R^{K_g\times d_h}\): global topology tokens;
- \(E_t^{\mathrm{meta}}\): sparse edge probability, age, and previous-action metadata.

One shared recurrent block updates the state:

\[
\mathcal Z_{t+1}=F_\theta(\mathcal Z_t,X_S,R_S,t),
\qquad t=0,\ldots,T-1.
\]

## 5. Factorized Corruption Curriculum

Training regions are induced subgraphs sampled only from legal training topology. For each clean region \(A^\star\), sample corrupted states from three independently controlled channels.

### 5.1 Identity corruption

Apply degree-preserving double-edge swaps:

\[
(i,j),(k,l)\rightarrow(i,l),(k,j).
\]

This preserves edge count and node degrees while damaging partner identity, triangles, communities, and spectrum. It is the primary test that the policy learns more than density or degree.

### 5.2 Activity corruption

Add or delete trusted synthetic edges with degree-stratified sampling. This creates signed node deficits

\[
B_i^T=d_i(A^\star)-d_i(A_t).
\]

Sampling must include low-, medium-, and high-degree nodes so the model cannot optimize only common low-degree cases.

### 5.3 Motif/community corruption

Replace triangle-closing or intra-community edges with open-wedge or cross-community alternatives. A degree-preserving version isolates higher-order organization; an add/delete version jointly tests motif and activity repair.

### 5.4 Severity schedule

Use corruption severity `s ∈ {light, moderate, heavy}`. Initial probabilities are:

| Severity | identity swaps | positive deletions | trusted insertions | motif-biased share |
|---|---:|---:|---:|---:|
| light | 5% of edges | 2% | 2% | 25% |
| moderate | 15% | 8% | 8% | 50% |
| heavy | 30% | 15% | 15% | 70% |

A sampled state records every edit and its provenance. Ordinary unobserved pairs are never silently converted into trusted negatives.

## 6. Hybrid Oracle-Policy Teacher

The teacher is training-only and sees both \(A^\star\) and \(A_t\).

### 6.1 Shared clean/corrupted encoder

A shared privileged graph encoder computes

\[
(H^\star,E^\star,G^\star)=T_\psi(X_S,A^\star),
\qquad
(H^t,E^t,G^t)=T_\psi(X_S,A_t).
\]

The initial implementation uses a compact GRIT-style encoder on regions of at most 200 nodes: RRWP initializes pair relations, node and pair states co-evolve, and degree information is injected explicitly. The two branches share every parameter so their difference has a common coordinate system.

### 6.2 Exact action support

For each frontier pair, the trusted action is:

\[
y^T_{ij}=\begin{cases}
\mathrm{ADD},&A^\star_{ij}=1,\ A_{t,ij}=0,\\
\mathrm{REMOVE},&\text{the corruption inserted }(i,j),\\
\mathrm{KEEP},&\text{the current trusted state is already correct},\\
\mathrm{UNKNOWN},&\text{the clean dataset provides no trusted negative evidence}.
\end{cases}
\]

UNKNOWN rows are masked from hard action classification.

### 6.3 Exact local advantage plus learned priority residual

Define a node-aligned structural error

\[
\mathcal E(A,A^\star)=
\lambda_e\mathcal E_{\mathrm{edge}}
+\lambda_d\mathcal E_{\mathrm{degree}}
+\lambda_m\mathcal E_{\mathrm{motif}}
+\lambda_c\mathcal E_{\mathrm{community}}.
\]

For legal action `a`, compute its exact one-step advantage

\[
\mathcal A_{\mathrm{exact}}(a;A_t)
=
\mathcal E(A_t,A^\star)-\mathcal E(A_t\oplus a,A^\star).
\]

The privileged teacher adds a bounded learned residual:

\[
u^T_{ij,a}
=
\mathcal A_{\mathrm{exact}}(a;A_t)
+r_\psi(H^\star,H^t,E^\star,E^t,G^\star,G^t,a),
\]

with `tanh` bounding the residual to the scale of the exact term. The residual learns multi-action interactions that a single-edge delta misses; the exact term prevents arbitrary teacher priorities.

### 6.4 Teacher topology targets

The teacher emits:

- signed degree/activity budget \(B_t^T\);
- soft community assignment \(C_t^T\);
- motif obligations \(M_t^T\);
- scalar state value \(V_t^T=-\mathcal E(A_t,A^\star)\);
- stop target equal to one only when no trusted beneficial repair remains.

Community assignments use edge and triangle conductance with anti-collapse regularization. The assignment is node-aligned but its column labels are permutation-equivalent.

### 6.5 Teacher coordinated plan

The teacher utilities and budgets pass through the same transport solver as the student:

\[
\Gamma_t^T
=
\operatorname{Transport}(U_t^T,B_t^T,A_t,\mathcal F_t).
\]

This post-transport plan is the primary KD object. It records which repairs survive global competition for shared node budgets.

## 7. Student Architecture

### 7.1 Cached content path

Reuse the mature `v3_1` protein encoder and content pair scorer. Encode each protein once, cache its token-derived node vector, and compute content compatibility \(c_{ij}\) in pair chunks. During early STPD stages the content path is frozen; on-policy fine-tuning may unfreeze its final encoder block and head at one tenth of the topology-core learning rate.

### 7.2 Whole-set and sparse graph core

Initialize node states from cached content and exchange information with `K_g` topology tokens. Each recurrent block combines:

1. weighted sparse graph propagation over \(A_t\);
2. node-to-token attention;
3. token-to-node attention;
4. gated fusion with the static content state.

The recurrent core costs approximately \(O(|E_t|d_h+nK_gd_h)\).

### 7.3 Operational topology heads

The student predicts \(B_t,C_t,M_t,V_t,p_t^{\mathrm{stop}}\). These outputs are not disposable auxiliary heads:

- budgets enter transport constraints;
- communities enter proposer and policy compatibility;
- motif obligations score action-induced motif changes;
- value ranks candidate plans and monitors rollout improvement;
- stop probability terminates inference.

### 7.4 Symmetric action decoder

For unique undirected frontier edge `(i,j)` with `i < j`, construct

\[
\phi_{ij}^t=[h_i+h_j,\ h_i\odot h_j,\ |h_i-h_j|,\ c_{ij},\ A_{t,ij},\ B_i+B_j,\ C_i^TW_CC_j,\ \Delta M_{ij},\ g_t,\ K^{\mathrm{mem}}_{ij}].
\]

A shared MLP returns ADD, REMOVE, and KEEP utilities plus an action-value estimate. Symmetric features guarantee \(u_{ij}=u_{ji}\).

## 8. Learned Content-and-Budget Prior

STPD's primary initial graph is learned internally:

\[
(H_0,G_0)=\mathcal E_\theta(X_S,R_S),
\qquad
\widehat d_i^0=\operatorname{softplus}(f_d(h_i^0,G_0)),
\]

\[
u_{ij}^0=c_{ij}+z_i^TW_zz_j+C_i^TW_CC_j+\gamma K^{\mathrm{mem}}_{ij},
\]

\[
A_0=\operatorname{SparseTransport}(U_0,\widehat d^0).
\]

No true edge count is supplied. The implied initial edge mass is \(\frac12\sum_i\widehat d_i^0\). Empty-graph and frozen-B0 initializations remain ablations and use the identical recurrent policy afterward.

## 9. Cross-Fitted Analogical Topology Memory

### 9.1 Sparse feature alignment

For target node `i`, retrieve `k_mem` training analogs and compute sparse row-stochastic weights

\[
P_{ir}=\operatorname{sparsemax}_{r\in\mathcal N_k(i)}q(x_i)^Tk(x_r).
\]

The same `P_i` is reused for every pair containing `i`.

### 9.2 Node and relational transfer

Transfer node priors

\[
b_i^{\mathrm{mem}}=\sum_rP_{ir}d_r,
\quad
C_i^{\mathrm{mem}}=\sum_rP_{ir}C_r,
\quad
M_i^{\mathrm{mem}}=\sum_rP_{ir}M_r,
\]

and the relational prior

\[
K^{\mathrm{mem}}=PA_RP^T.
\]

`K_mem` is evaluated only for pair chunks or frontier pairs; it is not materialized as a dense persistent matrix.

### 9.3 Confidence and abstention

A gate derived from nearest-neighbor distance, retrieval margin, and alignment entropy yields `g_i ∈ [0,1]`. Pair memory is scaled by `g_i g_j`. Poorly supported nodes therefore revert toward strict STPD.

### 9.4 Five-fold cross-fitting

Assign every training node to one deterministic fold. A target region from fold `f` may retrieve only nodes outside `f`, and no edge incident to target-region nodes may enter its memory graph. Test inference may use all legal training nodes.

## 10. Recurrent Whole-Universe Proposer

Every repair round performs a cheap all-pairs scan in fixed-size chunks:

\[
g_{ij}^{+,t}=c_{ij}+\alpha_b(B_i+B_j)+\alpha_cC_i^TW_CC_j+\alpha_z z_i^TW_zz_j+\alpha_gq(h_i,h_j,G_t)+\alpha_rK^{\mathrm{mem}}_{ij}.
\]

A streaming per-node top-`L` operator builds the student frontier without storing the full matrix. The final frontier is the deduplicated union of:

- every current edge;
- top-`L` proposed nonedges per node;
- uncertainty-band pairs;
- high-confidence memory proposals;
- recently changed edges retained for two rounds.

During training, trusted missing/inserted repair edges are injected only into the exact-policy training frontier. Proposal recall is always measured on the unaugmented student frontier.

## 11. Budget-Constrained Parallel Repair

For frontier add/remove variables \(x_{ij}^+,x_{ij}^-\), solve

\[
\max_x\sum_{(i,j)\in\mathcal F_t}(u_{ij}^+x_{ij}^++u_{ij}^-x_{ij}^-)
+\tau H(x)
-\lambda_b\sum_i\left[\sum_j(x_{ij}^+-x_{ij}^-)-B_i\right]^2,
\]

subject to

\[
0\le x_{ij}^+\le1-A_{t,ij},\quad
0\le x_{ij}^-\le A_{t,ij},\quad
x_{ij}^++x_{ij}^-\le1.
\]

Use unique undirected edges, so symmetry is structural rather than post-hoc. The differentiable solver runs a fixed number of fp32 primal-dual iterations and returns plan mass, realized degree delta, and constraint residual. The graph update is

\[
A_{t+1}=\operatorname{clip}(A_t+\rho_tX_t^+-\rho_tX_t^-,0,1).
\]

Formal scoring rounds the final soft graph with a deterministic budget-aware greedy projection. Soft probabilities remain available as the canonical pair-score artifact.

## 12. Distillation and Task Losses

For one state, use:

\[
\mathcal L=
\lambda_{\mathrm{sup}}L_{\mathrm{support}}
+\lambda_{\mathrm{prio}}L_{\mathrm{priority}}
+\lambda_{\mathrm{plan}}L_{\mathrm{plan}}
+\lambda_bL_{\mathrm{budget}}
+\lambda_cL_{\mathrm{community}}
+\lambda_mL_{\mathrm{motif}}
+\lambda_vL_{\mathrm{value}}
+\lambda_sL_{\mathrm{stop}}
+\lambda_pL_{\mathrm{proposal}}
+\lambda_tL_{\mathrm{terminal}}.
\]

Definitions:

- `L_support`: masked action cross-entropy on trusted ADD/REMOVE/KEEP targets;
- `L_priority`: KL from teacher to student utilities within legal action support;
- `L_plan`: edgewise divergence between \(\Gamma_t^T\) and \(\Gamma_t^S\), plus per-node realized-degree-change error;
- `L_budget`: Huber on signed normalized budgets;
- `L_community`: Hungarian-aligned soft-assignment MSE plus anti-collapse entropy/occupancy terms;
- `L_motif`: Huber on the four node-aligned motif obligations;
- `L_value`: Huber on teacher state value and sampled-plan advantage;
- `L_stop`: BCE on the privileged stop target;
- `L_proposal`: trusted-addition ranking, hard-false-proposal ranking, and node proposal-mass coverage;
- `L_terminal`: observed-positive/trusted-negative pair loss plus node-aligned degree, motif, and community errors after rollout.

Normalize each raw loss by a detached EMA of its magnitude before applying fixed stage weights. Log per-loss gradient norm and pairwise gradient cosine. Do not enable gradient surgery by default; run PCGrad only as a pre-registered ablation if a pair of load-bearing losses has mean gradient cosine below `-0.2` for three consecutive validations.

## 13. On-Policy Dataset Aggregation

Training-state distribution at epoch `e` is

\[
\mu_e=\beta_e q_{\mathrm{corr}}+(1-\beta_e)q_{\theta,e}+q_{\mathrm{replay}},
\]

where `beta_e` anneals linearly from `1.0` to `0.25` during the on-policy stage. Replay contains states with hub over-allocation, correct degrees but wrong partners, damaged motifs at matched degree, early stopping, and large strict-vs-memory disagreement.

Each transition is initially a stop-gradient boundary:

\[
A_{t+1}=\operatorname{stopgrad}(\operatorname{Update}(A_t,\Gamma_t^S)).
\]

This matches hard inference states and avoids unstable long-horizon gradients. A final two-round differentiable unroll is an optional ablation, not a dependency of the core method.

## 14. Training Curriculum

1. **Teacher stage:** train and validate privileged state encoding, exact support, learned priority residual, topology targets, and oracle transport repair.
2. **Prior stage:** train learned \(A_0\) initialization with frozen content encoder.
3. **Offline KD stage:** one-step student distillation on factorized corruptions.
4. **On-policy strict stage:** recurrent strict-STPD rollouts with dataset aggregation and replay.
5. **Memory stage:** initialize from strict STPD, enable cross-fitted memory, and train retrieval/alignment plus policy use.
6. **Optional short-unroll stage:** two differentiable rounds at low learning rate.
7. **Formal stage:** three seeds for strict and supported primary arms, followed by one frozen held-out test evaluation per seed.

## 15. Inference Algorithm

```text
INPUT: whole set X_S, candidate universe Q_S, optional legal memory M_train
1. Encode every protein once with the cached content path.
2. Build strict or memory-conditioned set/topology-token state.
3. Predict node activity and role/community priors.
4. Scan Q_S in chunks and construct learned sparse A_0 by budgeted transport.
5. For t = 0..T-1:
   a. update sparse node state and global topology tokens from A_t;
   b. predict budgets, communities, motifs, value, and stop probability;
   c. scan all candidate pairs with the cheap proposer;
   d. build the sparse exact-policy frontier;
   e. score ADD/REMOVE/KEEP and solve coordinated transport;
   f. update A_t and sparse edge metadata;
   g. stop only after two consecutive stop-positive rounds or negligible plan mass.
6. Decode every candidate pair once more from the final shared state.
7. Write symmetric probabilities, final sparse graph, and rollout telemetry.
```

Maximum rounds are fixed before formal runs. The recommended initial value is `T=6`.

## 16. Data and Leakage Controls

- Build training regions only from the legal training-side structural graph.
- Strip self-loops for policy targets; preserve the repository's canonical self-loop conventions in final evaluation.
- Remove the queried/edited edge before computing any heuristic or motif feature used to supervise its action.
- Record synthetic insertions explicitly; only those insertions are trusted REMOVE examples.
- Keep `V_val`-internal pairs quarantined from training supervision.
- Memory cross-fitting excludes target fold nodes and all incident edges, not just the exact target region.
- Construct the whole test set from benchmark membership, never from hidden test BFS traversal.
- Do not feed density-matched target edge count, validation threshold, hidden degrees, or reference statistics into inference.
- Fit retrieval embeddings, calibration, stopping thresholds, and rollout count on training/validation only.

## 17. Metrics and Telemetry

### 17.1 Canonical task metrics

Report:

- AUROC, AUPRC, non-self AUPRC, ECE, Brier, F1/MCC at the declared classification threshold;
- global GS and RD;
- BFS-macro GS and RD;
- degree, clustering, and spectral MMD ratios.

### 17.2 Node-aligned topology diagnostics

Report degree Spearman, top-10% hub recall, per-degree-bucket edge recall, triangle-count correlation, and community edge recall.

### 17.3 Policy diagnostics

Report:

- proposer ADD recall at top-`L` by corruption type/severity/degree bucket;
- action macro-F1 on trusted rows;
- teacher/student plan edge overlap and plan-mass KL;
- budget MAE and realized degree-delta MAE;
- motif-obligation MAE;
- transport constraint residual;
- graph error and teacher/student value after every round;
- stop calibration and premature-stop frequency;
- memory confidence, retrieval entropy, and strict-vs-memory disagreement.

### 17.4 Secondary graph-distribution diagnostics

For stochastic corruption/rollout analyses, add random-GIN embedding MMD-RBF and graph-embedding precision/recall as secondary diagnostics. They do not replace the benchmark five-tuple.

## 18. Checkpoint Selection

Use the existing validation protocol: mean-rank validation AUPRC plus BFS-macro GS/RD and all three MMD ratios. Policy telemetry is diagnostic and must not silently redefine checkpoint selection. If a run violates finite-state or data-boundary contracts, abort it; otherwise publish the selected checkpoint even when weak so negative evidence remains interpretable.

## 19. Experiment Ladder

### Gate 0 — Privileged repair ceiling

Compare exact-support teacher repair against no repair on held-out training regions. Required evidence before student training:

- monotonic mean structural error over six rounds;
- at least 90% recovery of trusted identity swaps on light/moderate corruption;
- no collapse to density-only repair on degree-preserving corruption.

### Gate 1 — Initial prior and proposer

Compare learned \(A_0\), frozen-B0 \(A_0\), and empty \(A_0\). The learned prior must match or beat frozen B0 on validation-region AUPRC or GS while remaining independently trained. Unaugmented proposer recall must reach at least `0.95` on light/moderate and `0.90` on heavy trusted additions at the selected `L`.

### Gate 2 — One-step policy transfer

Run strict one-step KD with and without post-transport plan loss. Success requires that plan KD improves partner identity or node-aligned degree repair beyond static action KD, not merely calibration.

### Gate 3 — Recurrent strict STPD

Compare one-shot prior, independent parallel flips, and full recurrent transport. The recurrent arm must improve both a pair metric and a topology-identity metric over the one-shot prior.

### Gate 4 — Memory mechanism

Compare:

- strict STPD;
- node-summary memory only;
- relational \(PA_RP^T\) memory;
- edge-shuffled relational memory;
- full relational memory with confidence gate.

A supported-tier claim requires relational memory to beat summary-only and shuffled-memory controls.

### Gate 5 — Mechanism ablations

Run the following short validation experiments before full seeds:

1. no activity budget;
2. no community state;
3. no motif obligations;
4. no value/stop heads;
5. no post-transport plan KD;
6. no on-policy aggregation;
7. fixed initial frontier instead of recurrent global proposal;
8. independent threshold update instead of transport;
9. exact-only teacher priority instead of learned residual;
10. frozen B0 vs learned vs empty initialization.

### Gate 6 — Formal comparison

Formal primary arms are B0, best conventional B1 KD arm, strict STPD, and memory-supported STPD, each with three seeds. Test remains untouched until all primary configurations and seeds are frozen.

## 20. Pre-Registered Success Criteria

### 20.1 Strict STPD route pass

Across three seeds, strict STPD must satisfy all of:

- mean non-self AUPRC at least `B0 + 0.005`;
- mean BFS-GS at least `B0 + 0.006`;
- at least two of three MMD ratios improve by 10% or more, with the third no worse than 5%;
- no post-calibration ECE regression greater than `0.02`;
- proposer recall gates remain satisfied on validation rollouts.

### 20.2 Strong pass

A strong result is non-self AUPRC `+0.01`, BFS-GS `+0.02`, and all three MMD ratios at least 15% lower than B0.

### 20.3 Memory-supported increment

Memory support must improve strict STPD by at least `+0.003` non-self AUPRC or `+0.006` BFS-GS while not materially degrading the other metric family. The paper reports this as additional support information, not as the strict method.

### 20.4 Negative-result interpretation

- High teacher ceiling + low proposer recall: candidate retrieval failure.
- High proposer recall + low one-step imitation: topology policy is not feature-transferable.
- Good one-step imitation + rollout collapse: state-distribution or update-operator failure.
- Strict failure + memory success: missing realization information can be supplied by training analogs.
- Strict and memory failure despite a strong teacher: the legal inputs do not identify the privileged topology strongly enough for this architecture.

## 21. Monitoring and Stop Rules

Abort a run on NaN/Inf, DDP step disagreement, malformed sparse indices, negative transport mass, symmetry violation above `1e-6`, or forbidden-node/edge access.

Pause and diagnose before formal continuation when:

- frontier recall stays below `0.90` for two validations;
- transport mean absolute budget residual exceeds `0.25` edges per node after warmup;
- more than 5% of nodes consume over 150% of predicted positive budget;
- community occupancy collapses so fewer than four of 16 slots carry 95% of mass;
- graph error increases for three consecutive repair rounds;
- stop-positive frequency exceeds 50% before round two;
- any load-bearing loss gradient norm is 20× another for three validations;
- memory gates saturate above `0.95` or below `0.05` for more than 90% of nodes.

## 22. Initial Operating Point and Compute Budget

These are starting values for profiling, not paper claims:

| Component | Initial value |
|---|---:|
| training region sizes | 64, 96, 128, 160, 200 |
| content width | 512 |
| recurrent topology width | 256 |
| global topology tokens `K_g` | 32 |
| community slots `K_c` | 16 |
| motif channels | 4 |
| repair rounds `T` | 6 |
| proposer top-`L` per node | 64 |
| all-pairs scan chunk | 131,072 pairs |
| transport iterations | 8 |
| memory analogs per node | 16 |
| memory cross-fit folds | 5 |
| topology-core LR | `1e-4` |
| optional content LR | `1e-5` |
| gradient clip | 1.0 |

The full-universe student must fit one H20 with chunked pair scans; DDP may shard pair scans and training regions. Dense privileged teacher work remains restricted to regions of at most 200 nodes.

---

# Part II — File and Interface Map

## 23. New Files

### Model package

- `src/model/stpd/__init__.py` — public model exports.
- `src/model/stpd/types.py` — immutable typed state, frontier, plan, teacher-target, and output records.
- `src/model/stpd/modules.py` — topology tokens, sparse graph block, heads, and symmetric pair features.
- `src/model/stpd/teacher.py` — privileged shared clean/corrupted encoder and priority/value heads.
- `src/model/stpd/student.py` — learned prior and recurrent student.
- `src/model/stpd/proposer.py` — chunked whole-universe proposal scoring and streaming top-`L` frontier.
- `src/model/stpd/transport.py` — differentiable sparse transport and deterministic rounding.
- `src/model/stpd/memory.py` — sparse cross-fitted retrieval alignment and relational-memory scoring.

### Training package

- `src/stpd/__init__.py` — method-level exports.
- `src/stpd/config.py` — strict schema and dataclasses.
- `src/stpd/regions.py` — legal training-region corpus construction.
- `src/stpd/corruption.py` — factorized corruption and provenance.
- `src/stpd/motifs.py` — exact region targets and sparse frontier motif deltas.
- `src/stpd/targets.py` — exact actions, structural error, advantage, and teacher targets.
- `src/stpd/losses.py` — normalized multi-objective losses.
- `src/stpd/rollout.py` — offline/on-policy/replay state generation.
- `src/stpd/artifacts.py` — checkpoint, memory-index, score, and telemetry serialization.
- `src/stpd/metrics.py` — policy and node-aligned diagnostics.
- `src/train_stpd.py` — teacher, prior, offline-KD, on-policy, memory, and formal training stages.
- `src/score_stpd.py` — whole-universe recurrent inference and canonical score artifact writer.

### Experiments and configs

- `src/experiments/stpd/gates.py` — Gate 0–5 analyses and pass/fail summaries.
- `src/experiments/stpd/make_ablation_configs.py` — deterministic ablation config generation.
- `configs/stpd_strict_breadth_first.yaml` — primary strict config.
- `configs/stpd_memory_breadth_first.yaml` — primary supported config.

### Tests

- `tests/model/stpd/` — model, proposer, memory, and transport contracts.
- `tests/stpd/` — corruption, targets, losses, rollout, artifacts, and metrics.
- `tests/test_train_stpd.py` — trainer integration and resume behavior.
- `tests/test_score_stpd.py` — whole-universe scoring and canonical artifact integration.

## 24. Existing Files to Modify

- `src/e2_pipeline.py` — register STPD train/score/test stages without changing existing families.
- `src/eval/test_protocol.py` — accept canonical STPD score artifacts only if family dispatch is currently closed.
- `hpc/run.sh` — add explicit `stpd` train and score routing.
- `hpc/README.md` — document STPD commands and artifacts.
- `src/README.md` — add the new model-family entry after implementation is operational.
- `docs/02-methodology.md` — record strict whole-set and supported-memory conditions only after Gate 3 establishes a usable method.

---

# Part III — Task-by-Task Implementation Plan

## 25. Task 1: Typed State and Configuration

**Files:**
- Create: `src/model/stpd/types.py`
- Create: `src/stpd/config.py`
- Test: `tests/model/stpd/test_types.py`
- Test: `tests/stpd/test_config.py`

**Interfaces:**
- Produces: `SparseGraphState`, `TopologyState`, `Frontier`, `TransportPlan`, `TeacherTargets`, `STPDOutput`, and `STPDConfig`.
- Consumed by: every later task.

- [ ] **Step 1: Write failing shape, symmetry, and schema tests**

```python
def test_frontier_rejects_noncanonical_undirected_edges() -> None:
    with pytest.raises(ValueError, match="i < j"):
        Frontier(edge_index=torch.tensor([[2], [1]]), current=torch.ones(1))


def test_config_rejects_memory_in_strict_tier() -> None:
    raw = minimal_config_dict()
    raw["tier"] = "strict"
    raw["memory"]["enabled"] = True
    with pytest.raises(ValueError, match="strict tier"):
        STPDConfig.from_mapping(raw)
```

- [ ] **Step 2: Run the tests and confirm missing-type failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_types.py tests/stpd/test_config.py -n0 -v`

- [ ] **Step 3: Implement frozen dataclasses, tensor validation, and unknown-key rejection**

```python
@dataclass(frozen=True)
class SparseGraphState:
    edge_index: torch.Tensor  # (2, E), unique i < j
    edge_prob: torch.Tensor   # (E,), fp32 in [0, 1]
    num_nodes: int

@dataclass(frozen=True)
class TransportPlan:
    add_mass: torch.Tensor
    remove_mass: torch.Tensor
    degree_delta: torch.Tensor
    constraint_residual: torch.Tensor
```

- [ ] **Step 4: Run focused tests, Ruff, and mypy**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_types.py tests/stpd/test_config.py -n0 -v && .venv/bin/python -m ruff check src/model/stpd src/stpd tests/model/stpd tests/stpd && .venv/bin/python -m mypy src/model/stpd src/stpd`

- [ ] **Step 5: Commit**

```bash
git add src/model/stpd/types.py src/stpd/config.py tests/model/stpd/test_types.py tests/stpd/test_config.py
git commit -m "feat: define STPD state and config contracts"
```

## 26. Task 2: Region Corpus, Corruption, and Exact Targets

**Files:**
- Create: `src/stpd/regions.py`
- Create: `src/stpd/corruption.py`
- Create: `src/stpd/motifs.py`
- Create: `src/stpd/targets.py`
- Test: `tests/stpd/test_regions.py`
- Test: `tests/stpd/test_corruption.py`
- Test: `tests/stpd/test_targets.py`

**Interfaces:**
- Consumes: `STPDConfig` and the repository `ValRegionSplit`.
- Produces: `RegionExample`, `CorruptedRegion`, `CorruptionProvenance`, and `build_teacher_targets(...)`.

- [ ] **Step 1: Test exact invariants for all corruption channels**

```python
def test_identity_corruption_preserves_degree_and_edge_count() -> None:
    corrupted = corrupt_identity(clean_graph(), seed=7, fraction=0.25)
    assert sorted(dict(corrupted.graph.degree()).values()) == sorted(dict(clean_graph().degree()).values())
    assert corrupted.graph.number_of_edges() == clean_graph().number_of_edges()


def test_unknown_nonedge_is_not_a_remove_target() -> None:
    target = action_target(clean=False, current=False, provenance=None)
    assert target is Action.UNKNOWN
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `.venv/bin/python -m pytest tests/stpd/test_regions.py tests/stpd/test_corruption.py tests/stpd/test_targets.py -n0 -v`

- [ ] **Step 3: Implement deterministic region sampling and corruption provenance**

Use node-keyed seeds, strip policy self-loops, preserve `V_val` quarantine, and store deleted positives, inserted trusted negatives, swaps, and motif edits as separate arrays.

- [ ] **Step 4: Implement exact degree, triangle, wedge, community, structural-error, and one-step-advantage targets**

```python
def exact_advantage(
    state: SparseGraphState,
    clean: SparseGraphState,
    frontier: Frontier,
    weights: StructuralErrorWeights,
) -> torch.Tensor:
    """Return shape (E, 3) advantages for ADD/REMOVE/KEEP."""
```

- [ ] **Step 5: Run focused and data-boundary tests**

Run: `.venv/bin/python -m pytest tests/stpd/test_regions.py tests/stpd/test_corruption.py tests/stpd/test_targets.py -n0 -v`

- [ ] **Step 6: Commit**

```bash
git add src/stpd/regions.py src/stpd/corruption.py src/stpd/motifs.py src/stpd/targets.py tests/stpd/test_regions.py tests/stpd/test_corruption.py tests/stpd/test_targets.py
git commit -m "feat: add STPD corruption and exact repair targets"
```

## 27. Task 3: Sparse Transport and Hard Projection

**Files:**
- Create: `src/model/stpd/transport.py`
- Test: `tests/model/stpd/test_transport.py`

**Interfaces:**
- Produces: `solve_sparse_transport(...) -> TransportPlan` and `round_budgeted_plan(...) -> SparseGraphState`.

- [ ] **Step 1: Write failing optimality, symmetry, and gradient tests**

```python
def test_competing_edges_share_one_node_budget() -> None:
    frontier = three_edges_incident_to_node_zero()
    plan = solve_sparse_transport(frontier, add_utility=torch.tensor([3., 2., 1.]), remove_utility=torch.zeros(3), budgets=torch.tensor([1., 1., 1., 1.]), steps=32)
    assert plan.add_mass.sum() <= pytest.approx(1.5, abs=0.05)
    assert plan.degree_delta[0] <= 1.05


def test_transport_has_finite_utility_gradients() -> None:
    utility = torch.randn(5, requires_grad=True)
    plan = solve_fixture(utility)
    plan.add_mass.sum().backward()
    assert torch.isfinite(utility.grad).all()
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_transport.py -n0 -v`

- [ ] **Step 3: Implement fixed-iteration fp32 primal-dual transport**

The solver must use canonical undirected edges, mask illegal actions, expose degree residuals, and avoid data-dependent Python iteration counts.

- [ ] **Step 4: Implement deterministic budget-aware greedy rounding**

Tie-break by canonical edge ID after adjusted utility. Verify repeated runs are byte-identical.

- [ ] **Step 5: Run tests and numerical gradient checks**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_transport.py -n0 -v`

- [ ] **Step 6: Commit**

```bash
git add src/model/stpd/transport.py tests/model/stpd/test_transport.py
git commit -m "feat: add budget constrained sparse transport"
```

## 28. Task 4: Privileged Teacher

**Files:**
- Create: `src/model/stpd/modules.py`
- Create: `src/model/stpd/teacher.py`
- Test: `tests/model/stpd/test_teacher.py`

**Interfaces:**
- Consumes: clean/corrupted region tensors and exact targets.
- Produces: `PrivilegedTeacher.forward(...) -> TeacherTargets`.

- [ ] **Step 1: Test shared-branch coordinates, action masking, and bounded residuals**

```python
def test_teacher_clean_and_corrupt_encoders_share_parameters() -> None:
    teacher = PrivilegedTeacher(test_config())
    assert teacher.clean_encoder is teacher.state_encoder
    assert teacher.corrupt_encoder is teacher.state_encoder


def test_teacher_never_assigns_remove_mass_to_unknown_nonedge() -> None:
    targets = teacher_fixture().targets
    assert targets.remove_policy[targets.action == Action.UNKNOWN].sum() == 0
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_teacher.py -n0 -v`

- [ ] **Step 3: Implement compact RRWP teacher encoder and topology heads**

Use dense region pair state only inside the privileged teacher. Implement community assignment, motif/budget/value/stop heads, exact-advantage addition, and bounded learned residual.

- [ ] **Step 4: Feed teacher utilities through `solve_sparse_transport`**

The returned `TeacherTargets` must contain both raw policy distributions and the coordinated plan.

- [ ] **Step 5: Run tests and a synthetic oracle-repair smoke rollout**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_teacher.py -n0 -v`

- [ ] **Step 6: Commit**

```bash
git add src/model/stpd/modules.py src/model/stpd/teacher.py tests/model/stpd/test_teacher.py
git commit -m "feat: add privileged STPD repair teacher"
```

## 29. Task 5: Student Prior and Recurrent Topology Core

**Files:**
- Create: `src/model/stpd/student.py`
- Test: `tests/model/stpd/test_student.py`

**Interfaces:**
- Produces: `STPDStudent.initialize(...)`, `STPDStudent.step(...)`, and `STPDStudent.decode_pairs(...)`.

- [ ] **Step 1: Test whole-set permutation equivariance and pair symmetry**

```python
def test_student_pair_scores_are_symmetric() -> None:
    output = student_fixture().decode_pairs(torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(output[0], output[1])


def test_node_state_permutes_with_whole_set() -> None:
    base, permuted = run_permutation_pair()
    torch.testing.assert_close(base.node_state[permutation], permuted.node_state)
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_student.py -n0 -v`

- [ ] **Step 3: Implement cached content adapters, topology tokens, sparse graph block, and operational heads**

```python
class STPDStudent(nn.Module):
    def initialize(self, content: torch.Tensor, memory: MemoryContext | None) -> TopologyState: ...
    def step(self, state: TopologyState, frontier: Frontier, memory: MemoryContext | None, round_index: int) -> STPDOutput: ...
    def decode_pairs(self, state: TopologyState, pairs: torch.Tensor, memory: MemoryContext | None) -> torch.Tensor: ...
```

- [ ] **Step 4: Implement learned `A_0` through transport and matched empty/B0 initialization hooks**

- [ ] **Step 5: Run tests plus one 2,018-node memory smoke test with synthetic features**

The smoke test must prove no persistent dense pair tensor appears in the state dict or runtime state.

- [ ] **Step 6: Commit**

```bash
git add src/model/stpd/student.py tests/model/stpd/test_student.py
git commit -m "feat: add recurrent STPD student"
```

## 30. Task 6: Cross-Fitted Memory

**Files:**
- Create: `src/model/stpd/memory.py`
- Test: `tests/model/stpd/test_memory.py`

**Interfaces:**
- Produces: `TrainingMemoryIndex`, `MemoryContext`, `retrieve_memory(...)`, and `score_memory_pairs(...)`.

- [ ] **Step 1: Test fold exclusion, row stochasticity, relational scoring, and abstention**

```python
def test_cross_fit_memory_excludes_target_fold_and_incident_edges() -> None:
    ctx = retrieve_fixture(target_fold=2)
    assert not set(ctx.reference_nodes) & set(nodes_in_fold(2))


def test_shuffled_reference_edges_change_relational_not_node_priors() -> None:
    base, shuffled = memory_shuffle_fixture()
    torch.testing.assert_close(base.activity_prior, shuffled.activity_prior)
    assert not torch.allclose(base.pair_prior, shuffled.pair_prior)
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_memory.py -n0 -v`

- [ ] **Step 3: Implement deterministic top-k retrieval, sparsemax alignment, confidence gate, and chunked `PA_RP^T` evaluation**

- [ ] **Step 4: Add strict-tier zero-memory path using the identical modules and shapes**

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_memory.py -n0 -v`

- [ ] **Step 6: Commit**

```bash
git add src/model/stpd/memory.py tests/model/stpd/test_memory.py
git commit -m "feat: add cross fitted topology memory"
```

## 31. Task 7: Whole-Universe Proposer and Frontier

**Files:**
- Create: `src/model/stpd/proposer.py`
- Test: `tests/model/stpd/test_proposer.py`

**Interfaces:**
- Produces: `scan_topk_frontier(...) -> Frontier` and `augment_training_frontier(...) -> Frontier`.

- [ ] **Step 1: Test streaming top-k equivalence and no-oracle inference**

```python
def test_chunked_topk_matches_dense_reference() -> None:
    dense = dense_topk_reference(scores_fixture(), k=3)
    chunked = scan_topk_frontier(score_fn_fixture(), num_nodes=12, chunk_pairs=7, top_l=3)
    assert canonical_edges(chunked) == canonical_edges(dense)


def test_oracle_frontier_rows_are_training_only() -> None:
    with pytest.raises(ValueError, match="training only"):
        augment_training_frontier(base_frontier(), oracle_edges(), training=False)
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/model/stpd/test_proposer.py -n0 -v`

- [ ] **Step 3: Implement chunked scoring, streaming per-node top-k, uncertainty/recent/current-edge unions, symmetrization, and deduplication**

- [ ] **Step 4: Implement proposal-recall telemetry on the unaugmented frontier**

- [ ] **Step 5: Run tests and a full candidate-count smoke scan**

- [ ] **Step 6: Commit**

```bash
git add src/model/stpd/proposer.py tests/model/stpd/test_proposer.py
git commit -m "feat: add recurrent whole universe proposer"
```

## 32. Task 8: Losses, Rollouts, Replay, and Diagnostics

**Files:**
- Create: `src/stpd/losses.py`
- Create: `src/stpd/rollout.py`
- Create: `src/stpd/metrics.py`
- Test: `tests/stpd/test_losses.py`
- Test: `tests/stpd/test_rollout.py`
- Test: `tests/stpd/test_metrics.py`

**Interfaces:**
- Produces: `STPDLoss`, `run_rollout(...)`, `ReplayBuffer`, and `policy_metrics(...)`.

- [ ] **Step 1: Test loss masking, EMA normalization, plan incidence, stop-gradient transitions, and replay determinism**

```python
def test_unknown_rows_have_zero_support_gradient() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    loss = support_loss(logits, actions_with_unknown())
    loss.backward()
    assert logits.grad[3].abs().sum() == 0


def test_rollout_state_is_detached_between_rounds() -> None:
    states = run_fixture_rollout(stop_gradient=True)
    assert states[1].edge_prob.grad_fn is None
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/stpd/test_losses.py tests/stpd/test_rollout.py tests/stpd/test_metrics.py -n0 -v`

- [ ] **Step 3: Implement all losses and detached EMA scaling**

- [ ] **Step 4: Implement corruption/student/replay mixture and hard-state categories**

- [ ] **Step 5: Implement roundwise diagnostics and gradient-cosine logging**

- [ ] **Step 6: Run focused tests**

- [ ] **Step 7: Commit**

```bash
git add src/stpd/losses.py src/stpd/rollout.py src/stpd/metrics.py tests/stpd/test_losses.py tests/stpd/test_rollout.py tests/stpd/test_metrics.py
git commit -m "feat: add STPD rollout distillation"
```

## 33. Task 9: Artifacts and Training CLI

**Files:**
- Create: `src/stpd/artifacts.py`
- Create: `src/train_stpd.py`
- Create: `configs/stpd_strict_breadth_first.yaml`
- Create: `configs/stpd_memory_breadth_first.yaml`
- Test: `tests/stpd/test_artifacts.py`
- Test: `tests/test_train_stpd.py`

**Interfaces:**
- Produces stage-specific checkpoints, `metrics.jsonl`, replay snapshots, memory index, and a published STPD checkpoint consumable by scoring.

- [ ] **Step 1: Write failing stage-transition, resume, and artifact-roundtrip tests**

```python
def test_resume_reproduces_next_rollout_state() -> None:
    uninterrupted = run_three_steps()
    resumed = run_one_save_resume_two_steps()
    assert_state_equal(uninterrupted, resumed)
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/stpd/test_artifacts.py tests/test_train_stpd.py -n0 -v`

- [ ] **Step 3: Implement strict config parsing, optimizer groups, fixed stage order, DDP rank-symmetric failures, and epoch-boundary resume**

- [ ] **Step 4: Implement teacher/prior/offline/on-policy/memory stage commands in one CLI**

Example:

```bash
python -m src.train_stpd --config configs/stpd_strict_breadth_first.yaml --stage teacher
python -m src.train_stpd --config configs/stpd_strict_breadth_first.yaml --stage on_policy
```

- [ ] **Step 5: Run CPU smoke tests and one bounded H20 probe**

- [ ] **Step 6: Commit**

```bash
git add src/stpd/artifacts.py src/train_stpd.py configs/stpd_strict_breadth_first.yaml configs/stpd_memory_breadth_first.yaml tests/stpd/test_artifacts.py tests/test_train_stpd.py
git commit -m "feat: add STPD staged trainer"
```

## 34. Task 10: Whole-Universe Scoring, Evaluation Integration, and Gates

**Files:**
- Create: `src/score_stpd.py`
- Create: `src/experiments/stpd/gates.py`
- Create: `src/experiments/stpd/make_ablation_configs.py`
- Modify: `src/e2_pipeline.py`
- Modify: `src/eval/test_protocol.py` only if family dispatch rejects canonical STPD artifacts
- Modify: `hpc/run.sh`
- Modify: `hpc/README.md`
- Modify: `src/README.md`
- Test: `tests/test_score_stpd.py`
- Test: `tests/experiments/stpd/test_gates.py`

**Interfaces:**
- Produces canonical pair score artifacts and Gate 0–5 reports consumed by the existing evaluation code.

- [ ] **Step 1: Test score symmetry, exact candidate coverage, chunk invariance, strict/memory separation, and artifact compatibility**

```python
def test_score_artifact_covers_each_candidate_once() -> None:
    artifact = score_synthetic_universe(chunk_pairs=11)
    assert np.array_equal(np.sort(artifact.row_id), np.arange(len(artifact.row_id)))
    assert len(np.unique(artifact.row_id)) == len(artifact.row_id)
```

- [ ] **Step 2: Confirm failures**

Run: `.venv/bin/python -m pytest tests/test_score_stpd.py tests/experiments/stpd/test_gates.py -n0 -v`

- [ ] **Step 3: Implement whole-set inference, final decode, and telemetry writer**

- [ ] **Step 4: Reuse existing test protocol by writing its canonical score schema; add only the minimal family dispatch required**

- [ ] **Step 5: Add `hpc/run.sh stpd` routing and deterministic ablation config generation**

- [ ] **Step 6: Run Gate 0–2 smoke suite and the repository fast test suite**

Run: `.venv/bin/python -m pytest -m "not slow and not integration" --dist loadfile`

- [ ] **Step 7: Run Ruff, format check, and strict mypy**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy src tests`

- [ ] **Step 8: Commit**

```bash
git add src/score_stpd.py src/experiments/stpd src/e2_pipeline.py src/eval/test_protocol.py hpc/run.sh hpc/README.md src/README.md tests/test_score_stpd.py tests/experiments/stpd
git commit -m "feat: integrate STPD scoring and evaluation"
```

---

# Part IV — Formal Run Order

1. Run CPU unit tests and one-GPU synthetic transport/proposer profiles.
2. Train teacher seed 0 and run Gate 0 on held-out training regions.
3. Train all three initializers and run Gate 1.
4. Train one-step strict arms with/without coordinated-plan KD and run Gate 2.
5. Train recurrent strict seed 0 and run Gate 3.
6. Train memory summary, relational, shuffled, and gated arms and run Gate 4.
7. Run short mechanism ablations and freeze the primary recipe.
8. Run strict and memory primary recipes for seeds 0, 1, and 2.
9. Freeze configs, checkpoints, thresholds, and expected score-artifact schema.
10. Execute the held-out test protocol once per frozen seed.
11. Report every primary edge metric and topology metric together, plus the policy telemetry needed to interpret failure.

# Part V — Design Self-Review

- Every approved design decision from the brainstorming sequence has an operational component: whole-set input, two tiers, corruption-conditioned policy, factorized corruption, hybrid teacher, sparse recurrent student, learned prior, cross-fitted relational memory, whole-universe proposer, coordinated transport, and on-policy plan distillation.
- Every topology target is node-aligned or action-aligned; no load-bearing claim depends on arbitrary teacher coordinate MSE.
- Edge identity is supervised directly through trusted actions and coordinated plans, while degree, community, and motif targets remain explicit rather than hidden inside MMD losses.
- The strict and supported tiers share the same student and differ only in legal memory inputs.
- The implementation never requires persistent dense pair state and has an explicit proposer-recall diagnostic separating retrieval failure from policy failure.
- Unknown PPI nonedges are masked, and memory cross-fitting prevents graph lookup.
- Formal evaluation remains compatible with the repository's pair-score artifacts and five-number topology contract.
- No section contains unresolved design placeholders; empirical defaults may change only through the validation protocol described above.
