# Experiment Protocol: Topology-Conditioned Inductive Edge Prediction

**Status:** current task/evaluation protocol; method selection remains open.
**Forward plan (2026-08-10):** run topology-distillation baselines first, probe
conditional latent-topology generation in parallel, and treat topology-aware output
losses as a complementary control. EgoStitch is a historical retrieval-grounded arm.
No candidate is `Ours` before matched endpoint-only comparison.

This document defines the benchmark contract, baseline hierarchy, experiment matrix, evaluation
protocol, caveats, and deliverables for this repository. It is intentionally self-contained:
all artifact references are relative to this repository, and all dataset/model names are neutral
placeholders.

> **Locked task; open method.** For `(i,j)`, infer topology context from exactly
> `(x_i,x_j)` and emit a binary label. Aggregate predictions only for graph-level
> evaluation. Distillation and latent generation are endpoint-only candidates;
> retrieval-grounded scaffolds are a separate support condition.
>
> **Known structural consequence (stated, not hidden):** under this contract the assembled
> graph is an edge-independent construction given features. Gate G2 measures the realism
> ceiling this implies (Chanpuriya et al. 2111.00048) before implementation, and every
> assembled-metric table carries the ceiling and Oracle reference rows (§4).

---

## 0. Per-query topology-context contract

The useful comparison is not "better edge classifier" alone. The benchmark asks whether a model
can preserve edge-level quality while assembling predictions into a plausible graph.

### Method boundary

For a query pair `(i, j)`, every context-bearing endpoint-only candidate obeys:

```text
T_ij = transfer_or_generate(x_i, x_j; learned_parameters)
p_ij = classifier(x_i, x_j, T_ij)
```

Training neighborhoods may be encoded by a graph-aware teacher, but inference receives
no observed test topology. Distillation aligns a student to teacher representations or
relations; conditional generation models a distribution over an identity-free topology
latent. Neither consumes neighbor identities. Structured output loss is a separate
training-side control. Pair mode remains primary; any future set/subgraph mode
must be declared as a different inference contract.

Components:

| Component | Status under the contract |
|---|---|
| Frozen feature encoder | Reuse as frozen node features `x_i`; never fine-tune in the core protocol. |
| Frozen pairwise scorer | Reuse as B0 and the matched endpoint-only decision trunk. |
| Topology teacher | Encode training topology for distillation targets or latent-generation supervision. |
| Topology representation | Open: distilled deterministic latent or conditional latent distribution. |
| Structured objective | Complementary training control over grouped edge predictions. |
| Retrieval/explicit scaffold | Historical separate-support arm; never presumed available. |
| Benchmark splits, evaluator, and assembled graph metrics | Reuse consistently for all methods. |

### Why per-query topology context remains the research object

- They answer the inductive query directly: "given two unseen nodes and frozen features, should an
  edge exist?"
- Endpoint-only variants avoid requiring a candidate universe before scoring one pair.
- They make the global-vs-local context comparison explicit.
- They amortize per-node work cleanly; per-pair marginal cost is reported honestly against B0
  (FLOPs/wall-clock table, §6).

---

## 1. Fixed experimental substrate

- **Benchmark:** **Benchmark-A**, the primary node-disjoint split used throughout this repository.
  Additional split families are **Benchmark-B** and **Benchmark-C**.
- **Node features:** frozen feature encoder outputs cached as repository-local artifacts. The
  protocol treats these vectors as fixed inputs and does not update the encoder.
- **Candidate universe:** all candidate node pairs for the held-out benchmark split. Labels define
  the edge prediction task only; graph topology is evaluated after assembling predictions.
- **Splits and integrity:** training and test nodes are disjoint. Any near-duplicate or overlap
  filter used by the benchmark must be reported with the run. The same complete train-side
  positive edges supply both structural topology and edge-loss classification; the
  topology projection alone removes self-pairs (spec §§6, 9.3).
- **Edge-level metrics:** AUROC, AUPRC, accuracy, sensitivity, specificity, precision, recall, F1,
  MCC at the selected operating point; calibration (ECE, Brier) on averaged probabilities.
- **Assembled graph metrics — two families (binding):**
  - *Trained-on family* (may appear inside training losses): degree-distribution MMD,
    clustering-coefficient MMD, code-histogram/motif-conductance distances, random-GIN
    energy distance.
  - *Held-out family* (never used in any training loss; **headlined**): Laplacian-spectrum MMD
    with an untouched kernel, orbit/motif-count distances beyond the trained set, component and
    path-length summaries, a learned-graph-feature distance from an encoder never used in
    training, and the discriminator probe (accuracy of a held-out classifier distinguishing
    assembled from real subgraphs; near-chance = realism).
  - *Official Graph Similarity / Relative Density:* for every fixed sampled node set, take the
    predicted and reference induced subgraphs and compute
    `GS = 1 - ||A_pred - A_ref||_1 / (sum(A_pred) + sum(A_ref))` (equivalently edge
    Dice/F1 for an undirected simple graph) and
    `RD = density(G_pred) / density(G_ref)`. Match the official evaluator's zero-density
    guards (`GS(empty, empty) = 1`, `RD(empty, empty) = 1`, nonempty-over-empty RD = inf),
    retain self-loops in both metrics, and report the unweighted mean over every sampled
    subgraph across all node-size buckets. GS and RD are independent of the MMD metrics;
    no MMD composite may be labeled "graph similarity".
  - *References:* every assembled-metric table carries (i) the real-vs-real **noise floor**
    row, (ii) the **edge-independence ceiling** row (gate G2), and (iii) the **Oracle** row
    (gate G3).
  - *Hygiene:* per statistic, disclose descriptor function, bin count/binning rule, kernel
    family, and every parameter; avoid ad-hoc EMD-/TV-based kernels unless justified;
    parameter sweeps are reported only for non-canonical exploratory alternatives; the
    canonical fixed-`σ=1` metric is not swept; bootstrap variance over node buckets and
    seeds; MMDs never aggregated across statistics naively (O'Bray 2106.01098;
    Thompson 2201.09871).
  - *Canonical MMD definition:* each graph is mapped to a full degree histogram,
    a 100-bin local-clustering histogram on `[0,1]`, or a 200-bin normalized-
    Laplacian spectral histogram on `[-1e-5,2]`. Descriptor induced subgraphs retain
    self-loops exactly as in the benchmark/official evaluator; self-loop counts are
    also disclosed separately. The spectral worker first divides its 200-bin counts
    by their own total to form a PMF; degree and clustering remain raw counts. The
    common MMD routine then normalizes every descriptor by `sum + 1e-6`. MMD² is the
    biased V-statistic under
    `k(x,y)=exp(-(0.5·||x-y||₁)²/2)` (`σ=1`, including within-sample diagonals).
  - *Reference normalization:* within every node-size bucket, reference samples
    retain artifact order and are split as `samples[::2]` versus `samples[1::2]`.
    The reported statistic is
    `mean_size MMD²(pred_size,ref_size) / mean_size MMD²(ref_even,ref_odd)`.
    Numerator, denominator, and ratio are all stored; the ratio is canonical and
    lower is better. A `1e-12` denominator floor is only a numerical guard.
- **Repository artifacts:** this protocol, [E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md),
  [e2-gap.html](../figures/e2-gap.html), [positioning.html](../figures/positioning.html),
  [04-model-proposal.md](04-model-proposal.md), [05-egostitch-spec.md](05-egostitch-spec.md).

---

## 2. Baseline hierarchy

| ID | Baseline | Neutral instantiation | Isolates |
|---|---|---|---|
| **B0** | Independent pairwise scorer | Frozen pairwise scorer over `(x_i, x_j)` | Topology-blind edge prediction |
| **B0+cal** | B0 + calibrated assembly | Temperature/Platt calibration + density- and degree-sequence-matched thresholding of B0 scores | Whether trivial assembly calibration recovers topology; also apply to the selected candidate |
| **B1** | Topology distillation | Graph2Feat/LLP/CAZI-MBN-style graph-aware teacher → endpoint-only student | First runnable topology-transfer baseline |
| **B2** | Deterministic structural latent | Endpoint-only student predicts topology-aware latent | Transfer without stochastic generation |
| **B3** | Conditional latent-topology generator | Predict `p(T_ij | x_i,x_j)`; no neighbor identities | Explicit missing-topology imputation |
| **B4** | Retrieval-grounded explicit scaffold | Historical EgoStitch arm with declared support universe | Extra inference support vs endpoint-only transfer |
| **Controls** | Structured loss, static denoising, calibrated assembly, full auxiliary supervision | Output constraint and simpler explanations |
| **B5** | Neural-SBM residual | Feature-only MLP affiliations `F_u` (NOCD Bernoulli–Poisson) + degree terms: `p_ij = σ(pair_logit + block + degree)` | Block prior alone; the project's null hypothesis. Also the first strict-inductive evaluation of the BP block prior |
| **PA-null** | Preferential-attachment null | `s_ij = k_i·k_j` from training-side degree statistics, reported with each benchmark's degree heterogeneity σ (log-normal fit) | Validity precondition (2405.14985): under uniform negatives this null averages AUC 0.83; any method not clearly beating it has an uninformative edge metric |
| **Odds-product** | Degree-respecting edge-independent baseline | `P_ij = σ(ℓ_i + ℓ_j)` fitted to the expected degree sequence (Chanpuriya 2111.00048 §3) | Cheapest degree-budget-honoring assembly with zero topology conditioning; also supplies the G2 overlap dial `P̃ = (1−ω)P + ωA` |
| **DEAL / Graph2Gauss** | External attribute-only baselines | DEAL 2007.08053; Graph2Gauss 1707.03815 | Independent-scoring SOTA for the setting; falsifiable prediction: both exhibit the E2 failure mode |
| **Method comparison** | B1 vs B2 vs B3, with B0 and Controls | Compare before assigning `Ours` | Method selection remains open |
| **Oracle** | Observed-graph upper bound | Uses true held-out graph neighborhoods at evaluation time (pinned instantiation: §5.0 G3) | Headroom; violates the inductive protocol; **run first (gate G3)** |

All fair baselines share frozen features, splits, query sets, complete training
edges, negative sampling, and an equal HPO budget fixed before held-out metrics open.

---

## 3. Historical EgoStitch experiment matrix (archived)

Here `Ours` means EgoStitch; these rows are provenance, not forward method selection.
Current execution follows §5.1 and names B1/B2/B3 explicitly.

### E1: Main result on Benchmark-A

- **Claim:** `Ours` matches or improves B0 on edge AUROC/AUPRC and improves assembled graph
  topology beyond what calibration and loss shaping achieve.
- **Run:** train `Ours` on Benchmark-A. Evaluate both edge-level predictions and assembled graph
  metrics. Compare against B0, `B0+cal`, `B3-dist`, `B3-full`, B5, and B2-global.
- **Metrics:** AUROC, AUPRC, MCC; both assembled-metric families with ceiling/Oracle/noise-floor
  rows; mechanism-transmission diagnostics (§4).
- **Success (pre-registered, Holm-corrected over the held-out family):** edge AUPRC within run
  noise of B0 or better; the **held-out** assembled family improves over B0, `B0+cal`,
  `B3-dist`, and `B3-full` with Holm-adjusted significance across ≥ 3 seeds; relative density
  moves toward 1. Failure readings per §5.2 decision rules.

### E2: Pair-to-topology gap (closed historical G1 result)

- **Claim:** an independent pairwise scorer can assemble into an implausible graph, and the
  degradation is not explained by threshold choice, negative-sampling regime, or metric
  normalization.
- **Run (G1 requirements):** one frozen scorer family (same as `Ours` uses), one candidate
  universe, one canonical metric normalization, a true threshold sweep (recall/density-vs-MMD
  curves), easy, hard (HeaRT-style), **and degree-corrected** (2405.14985) negatives, a
  second-architecture replication (B0-alt, as actually run — see the retirement note in the
  Result below), the PA-null row, real-vs-real noise floor, bootstrap variance, a defined
  official Graph Similarity and Relative Density over the fixed sampled subgraphs,
  an expressivity/robustness perturbation diagnostic (O'Bray 2106.01098), and the
  full-candidate-universe imbalance view.
- **Execution acceptance:** E2 uses a fixed 30-epoch throughput run across every H20
  visible to the runner; the DDP world size is automatically detected. Validation
  is executed after every epoch; quality is reported but is not the throughput acceptance gate
  for the first systems-optimization pass. The wall-clock gate is 60 minutes from an empty
  derived-cache path through final training artifacts.
- **Result (updated 2026-07-14):** G1 is complete with B0, B0-alt, and PA-null. Official
  BFS-macro GS/RD were formally rerun on the frozen score artifacts over all 500 fixed induced
  subgraphs; the final artifact is `outputs/deliverables/g1_graph_metrics_20260714/`. The B0
  degree-corrected ratio-1 row is AUROC 0.705519 / AUPRC 0.730260; the hard-heuristic
  row is 0.583965 / 0.626649; the hard-feature row is 0.569560 / 0.617475; and the full
  candidate-universe row is 0.690627 / 0.134302. At the quota-calibrated operating point,
  global simple-edge RD is `0.997710`, while official BFS-macro GS/RD are
  `0.312151/0.422345`, with degree/clustering/spectral MMD ratios
  `13.0768/11.9273/18.0931`.
  The ratios use the fixed-`sigma=1` Gaussian-TV biased MMD² numerator divided by the
  deterministic real-vs-real reference floor; `1` is that floor and lower is better.
  B0-alt reaches 0.693603 / 0.732509 on the degree-corrected row, 0.576711 / 0.623517
  on hard heuristic negatives, and 0.467864 / 0.561339 on hard feature negatives. Its
  independently calibrated assembly has global simple-edge RD `0.998739`,
  BFS-macro GS/RD `0.345802/0.450793`, and degree/clustering/spectral MMD ratios
  15.8304/13.4718/23.4734. PA-null has global simple-edge RD `1.000000` and
  BFS-macro GS/RD `0.245377/0.489125`.
  The gap therefore survives an alternate architecture, hard negatives, and calibrated
  thresholds, although PA-null beats B0 in the easy and feature-hard rows and remains a
  mandatory control. B0-alt was G1's architecture-independence replication; by owner
  decision (2026-08-03) it is retired from the forward baseline set (§2 lists B0 only)
  and its implementation was removed from the code tree. The values above are the
  closed, unaltered G1 record; provenance is in
  [E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md).
- **G2 result:** the checkpoint-aligned cached soft scorer has `Ov(P)=0.479886`, while
  `Ov_min=0.054954`
  at matched volume is sufficient to reach the reference triangle count; the full ceiling
  curve and caveats are in the synced G2 artifact.
- **G3 result (canonical metric rerun 2026-07-14):** B0 has global simple-edge RD
  `0.997710` and BFS-macro GS/RD `0.312151/0.422345`. The pinned `oracle_topo`
  arm has global simple-edge RD `1.000000` and BFS-macro GS/RD `0.503048/0.794303`
  (GS ratio `1.61155`) and MMD ratios `14.1148/8.23662/16.0772`, with per-statistic
  headroom `0.926465/1.44808/1.12539`. The disclosed `oracle_blend` arm also has global
  simple-edge RD `1.000000` and reaches BFS-macro GS/RD `0.323649/0.652734` (GS ratio
  `1.03683`) with MMD ratios `7.58890/3.06980/9.05715` and headroom
  `1.72315/3.88537/1.99767`. The Oracle arms are not approximately equal
  to B0 across assembled metrics, so the feature-insufficiency stop rule is not triggered
  and cleared the prerequisite for EgoStitch implementation.

**Latest checkpoint-only evaluation rerun:** the aligned legacy `v3_1` checkpoint was
scored on the same split in run `legacy_v31_s47_20260712T193900Z` (completed 2026-07-12
19:53:13 UTC). It reached balanced test AUROC/AUPRC `0.805170/0.818408`; its G1
degree-corrected ratio-1 row was `0.799577/0.813319`, hard-heuristic
`0.626746/0.663360`, and hard-feature `0.510083/0.602131`. This is an evaluation
rerun of a supplied checkpoint, not a replacement for the formal E2 training
acceptance or the canonical B0/B0-alt/G3 gate artifacts. The same scores were rerun through
the official benchmark evaluator on 2026-07-14: global simple-edge RD is `0.978392`,
BFS-macro GS/RD are `0.381264/0.500179`, and degree/clustering/spectral MMD ratios are
`13.8456/11.6277/19.9774`. The stronger scorer improves GS but does not close the
topology gap.

### E3: Baselines head-to-head on Benchmark-A

- **Claim:** `Ours` beats independent scoring, calibration, global refinement, static denoising,
  and all loss-shaping baselines under one protocol.
- **Run:** B0, `B0+cal`, B1, B2-global, B2-static, B3, `B3-dist`, `B3-full`, B5,
  DEAL/Graph2Gauss, `Ours`, and Oracle on the same split and operating-point policy.
- **Metrics:** the joint edge/topology table from E1.
- **Success:** `Ours` is on the Pareto frontier of edge AUPRC and held-out assembled realism;
  wins over `B3-full` and `B0+cal` per the §5.2 decision rules (Holm-corrected), without a
  meaningful edge AUPRC loss.

### E4: Mechanism ablations

1. **No scaffold:** retrieval features only, no local edges.
2. **Randomized scaffold:** same node count and edge count, shuffled edges (matched capacity).
3. **Retrieval-only adjacency:** feature-neighbor edges without pairwise scorer weights.
4. **Global vs local context:** B2-global against `Ours`.
5. **Scaffold size:** K sweep {8, 16, 32} + α-entmax learned-cardinality variant.
6. **Generator variant:** set decoder vs discrete-diffusion (Approach B) under the identical
   head.
7. **Fake-edge isolation:** remove vs keep queried edge inside `T̂_ij`; anchor-labeling on/off.
8. **Loss terms:** edge-only; + statistic distances; + GIN-space; + seam term; full objective;
   `L_ssl` on/off; CVAE knockout (deterministic decoder).
9. **Residual anchor weight:** vary how strongly the fused residual can move the frozen score.
   *(2026-07-16: applies to the frozen-s0 ablation arm only — the rev-3.0 headline has no
   residual anchor.)*
10. **Retrieved-thresholded scaffold (E4.10):** the original §0 instantiation — bridge baseline.
11. **Generation-only (E4.11):** no queried-edge supervision (subsumes blueprint B4).
12. **Harmonization:** `R = 0` vs rounds sweep; mask-schedule sweep; slot-agreement trajectory;
    hard-budget masking on/off.
13. **Grounding:** grounded vs ungrounded (headline arm); grounding-only vs imagination-only;
    conditioning-dropout rates for both nulls with counterfactual contrasts reported.
14. **Channels:** per-channel knockouts (s1/s2/s3/s4) with the s-channel correlation matrix.
    *(2026-07-16: for the rev-3.0 headline this becomes E4.15–E4.17; the channel form
    survives in the frozen-s0 arm.)*
15. **Pathway attribution (rev 3.0, headline requirement):** pair+topology
    (`∅_content_head` permanent) vs pair+content (`∅_topo_head` permanent) vs matched
    `B0-e2e` (`∅_all_head` permanent); four-logit decomposition (full, `f_logit`,
    pair+content, pair+topology) reported with every headline table; branch-dropout
    `p = 0` and swept-rate arms.
16. **Structure specificity (rev 3.0):** (a) within-pair `Â`/`Π` shuffle; (b) all scaffold
    edges removed (STE degenerates to DeepSets); (c) cross-pair scaffold shuffle;
    (d) matched-capacity randomized context; (e) **degree-preserving rewiring / weight
    permutation** (preserves node count, per-edge-type mass, per-node soft degree, weight
    distribution; destroys higher-order connectivity — the decisive control); (f)
    capacity-matched non-message-passing token bottleneck.
17. **Conditioning depth (rev 3.0):** none (`B0-e2e`) → logit-FiLM → pooled low-rank
    adapter → STE + gated cross-attention (headline); plus `N_inj ∈ {1,2}` and
    cls-token-only vs token-level injection variants.

**Identical-head convention:** all generator comparisons (E4.6, E4.10, Oracle-scaffold) use the
same fused decision head with the input-mapping convention of `05-egostitch-spec.md` (retrieved
neighbors as π=1, m=1 slots; Π from identity matches), head retrained per arm under the matched
tuning budget.

**Success:** the ablations identify which mechanism produces the topology gain without obscuring
edge-level quality; any mechanism that owns no gain is cut from the submitted model (04 §4.6
rule).

### E5: Integrity and leakage gates

- **Node-disjoint gate:** train/test nodes must be disjoint.
- **Near-duplicate gate:** report benchmark-provided duplicate filtering or run a repository-local
  nearest-neighbor overlap check.
- **Shared-edge gate:** topology and classification cover the same train positives;
  the topology projection alone removes self-pairs, and explicit leave-one-out corrections
  are verified for every in-batch positive pair.
- **B0 provenance gate:** the frozen scorer must never have seen validation/test pairs.
- **Seam-reference gate:** realism-loss references sampled label-agnostically.
- **Hard negatives:** compare random negatives with heuristic-related hard negatives.
  **Zero-edge construction rule:** hard negatives for zero-edge test nodes are constructed
  evaluator-side (heuristics computed on the ground-truth held-out graph — evaluation-side
  access only) plus feature-similarity hard negatives that need no graph; degree-corrected
  sampling per 2405.14985. Hard-negative *training* is an additional arm.
- **Memorization controls:** nearest-neighbor, degree-only, and random-vector controls.
- **Counterfactual control:** the `∅_content` contrast reported per spec §2.
- **Success:** `Ours` gains survive hard negatives and are not explained by trivial memorization,
  benchmark shortcuts, or context-prior shortcuts.

### E6: Split and benchmark generalization

- **Claim:** gains are not specific to Benchmark-A.
- **Run:** repeat E1 for Benchmark-B, Benchmark-C, and alternate split samplings when available;
  verify the degree-budget density normalization transfers (spec §1); optional retrieval-pool
  swap transfer probe (RDM pattern).
- **Success:** directionally consistent topology improvements, with any degradation reported
  rather than hidden.

### E7: Downstream graph utility — **load-bearing**

- **Claim:** a topology-realistic predicted graph supports graph-level downstream tasks better
  than a topology-blind pairwise assembly. This experiment carries the significance of the whole
  assembled-realism metric family; it is **not optional**.
- **Run:** feed predicted graphs from B0, `B0+cal`, `B3-full`, B2-global, and `Ours` into
  repository-local downstream graph tasks (community detection, node-label propagation,
  graph-based retrieval probes).
- **Success:** `Ours` improves graph-task metrics over B0/`B0+cal`/`B3-full`, or the document
  clearly reports the failure mode — in which case the assembled-realism claims are downgraded
  accordingly in all documents.

---

## 4. Evaluation protocol

- **Edge-level task:** score held-out candidate pairs and report AUROC/AUPRC plus
  thresholded metrics at the selected operating point; stochastic candidates average
  a fixed, disclosed number of samples.
- **Assembled graph task:** assemble scored pairs into a predicted graph and evaluate both metric
  families of §1 on benchmark buckets, with noise-floor, ceiling, and Oracle rows.
- **Operating point:** primary results use a validation-selected threshold. Secondary results use
  density-matched and threshold-sweep views to distinguish score quality from assembly density.
- **Imbalance view:** report PR curves or operating-point summaries under the full candidate
  universe when it differs from balanced edge evaluation.
- **Stratified reporting:** edge metrics by degree (head/tail), by Topological Concentration
  (2310.04612 — degree strata alone are weakly informative and metric-biased), and by
  community familiarity, plus a harmonic-mean joint number; per-benchmark Feature
  Contribution Ratio (Cold Brew) as a headroom diagnostic; per-benchmark degree
  heterogeneity σ; supervision-starvation statistic as a motivation figure.
- **Mechanism diagnostics:** distillation recovery toward teacher/Oracle, latent-generator
  calibration and diversity, or structured-loss effects, according to the candidate.
  Retrieval/grounding diagnostics apply only to the historical separate-support arm.
- **Reporting rule:** edge and assembled graph metrics are always reported together. No claim
  relies on one metric family alone. The held-out family is headlined.

---

## 5. Historical gates and current run order

### 5.0 Historical pre-implementation gates (completed or retired; non-blocking)

1. **G1 — E2 hardening** (see E2). Stop: gap closes ⇒ pivot to evaluation paper.
2. **G2 — Edge-independence ceiling (curve semantics).** With Chanpuriya 2111.00048's exact
   identities on cached soft scores: compute `V(P) = Σp`, `Ov(P) = Σp²/Σp`,
   `E[Δ] = tr(P³)/6`; plot the ceiling curve `(√2/3)(ω·V)^{3/2}` over overlap ω at matched
   volume (clustering ceiling = exact triangle numerator over the assembly's
   expected-degree denominator; do not use their Thm 6 directly — big-O only); mark the
   scorer's measured `Ov(P_soft)` and the minimum-required overlap
   `Ov_min = (3Δ*/√2)^{2/3}/V` for the reference triangle count. Note: a hard-thresholded
   assembly has `Ov = 1` where the bound is vacuous — the curve constrains stochastic,
   calibrated assemblies at their measured overlap. Stop: `Ov_min` far above what an honest
   feature-conditioned scorer attains on unseen nodes ⇒ flag for a locked-decision
   discussion before proceeding.
3. **G3 — Oracle first.** Run the Oracle row. Stop: Oracle ≈ B0 on assembled metrics ⇒ feature
   insufficiency; conditioning cannot help; pivot.
   **Pinned instantiation (2026-07-13, before implementation):** the canonical arm
   (`oracle_topo`) orders every candidate row by common-neighbor count on
   `strip_self_loops(test_graph)`, ties by Adamic–Adar, remaining ties by canonical pair
   order (the hard-heuristic selector's convention); its scalar score for edge metrics is
   the average-tie normalized rank of the `(CN, AA)` key. A secondary disclosed arm
   (`oracle_blend`) is the parameter-free rank fusion
   `0.5·rank01(p_B0) + 0.5·rank01(s_topo)`. Both arms assemble via the PA-null
   top-`target_edges` convention (non-self pairs, deterministic tie-break, no threshold);
   B0 is re-evaluated side-by-side from the same cached scores artifact, and the
   per-statistic headroom `MMD-ratio(B0) / MMD-ratio(oracle)` is the stop-rule quantity.
   Evaluation-side access only; the row is protocol-violating by design and never a fair
   baseline. `hard_heuristic` regime rows for `oracle_topo` are degenerate by construction
   (negatives are CN/AA-selected) and are disclosed as such.
4. **G4 — Specification freeze.** `05-egostitch-spec.md` reviewed and signed off; it then
   becomes the implementation contract. **Done 2026-07-09** (sign-off recorded in the spec's
   change log; the spec's §9 data contract additionally quarantines the shipped
   `*_ratio5_exclusive.txt` negatives and `train_graph.pkl`-as-target — both leak across the
   split under the strict gate).
5. **G5 — Staged build.** Stage 1: imagination + degree budget + closure only; Stage 2: +
   codebook + s3; Stage 3: + harmonization + seam loss. Each stage must beat the previous on
   the pre-registered criteria or the added mechanism is cut.
   **Stage-gate instantiation (2026-07-14, pre-implementation):** the Stage-1 gate
   comparators are `B0` (frozen candidate-scores artifact) and `B0+cal`; B1/B5 comparison
   rows are deferred to E3 with their implementations. Stage 1 runs at spec-default
   hyperparameters × one fixed seed as an engineering screening gate — the §8-spec
   30-config HPO-parity budget and at-least-three-seed inference apply to the E1/E3
   ladder, not to G5 Stage 1. Stage-1 acceptance criteria (headroom-weighted:
   clustering-MMD and BFS-macro GS/RD primary at matched global simple-edge RD;
   degree-MMD non-regression and matched-AUPRC guards) are pre-registered in
   `docs/registrations/g5_stage1_preregistration.json` before any held-out metric is
   opened (rule §5.2.4); the Stage-1 subset itself is pinned in spec §13.
   **Execution status (2026-07-15):** Stage-1 code and its formal auto-sized DDP path
   are implemented. Seed 0 completed on 2 × H20 with best validation AUROC/AUPRC
   `0.945766/0.951966` at epoch 14. Seed 1 stopped at the artifact performance gate
   (`steady_state_data_wait_fraction > 0.05`), so Seed 2 and all held-out gate rows
   remain unrun. This is an incomplete execution state, not a G5 decision; see
   `docs/results/G5-stage1-seed0-20260715.md`.
   **Stage-1 decision scope (revised 2026-07-16 after inspecting the exact-quota
   Seed-0 diagnostic):** one fixed seed is sufficient for the G5 Stage-1 engineering
   screening decision. The seed must complete training, candidate scoring, required
   fidelity/cost diagnostics, and the held-out topology gate. Acceptance is deterministic
   point-estimate dominance on all three registered primary axes plus both guards;
   p-values, confidence intervals, and Holm decisions are not acceptance evidence with
   one seed and must be reported as not applicable. This screening verdict does not
   support statistical-significance or cross-seed-robustness claims. E1/E3 remain bound
   to §5.2's at-least-three-seed Holm procedure.
   **Registration replacement (2026-07-16, before any held-out topology metric):** an
   fp32 feasibility rescore showed intrinsic boundary ties for which the former atomic
   threshold could not satisfy the unchanged 0.005 matched-global-RD tolerance. The
   current registration therefore selects the comparator's exact non-self quota by
   score and resolves only the boundary tie by canonical pair order, without labels or
   topology targets. The registration file is replaced in place; artifacts carrying
   its former hash are not formal inputs, so the then-registered three-seed run restarted
   from Seed 0.
   **Post-observation registration replacement (2026-07-16):** after that Seed-0
   diagnostic was produced and inspected, Stage 1 was deliberately re-scoped to the
   single-seed screening contract above. This is not a retroactive pre-registration:
   the completed artifact retains its old registration hash and remains diagnostic-only.
   A binding one-seed screening run requires the replacement experiment ID and run
   metadata pinned before training begins.
   **Headline-model revision (2026-07-16, rev 3.0; historical):** the pending frozen-s0 screening
   contract above is unchanged and remains the binding contract for the already-pinned
   frozen-s0 run — its outcome is the motivating arm for the successor. The **next**
   Stage-1 build after that screen publishes is the e2e stitched-topology-conditioned
   pair encoder (`04-model-proposal.md` §4.4 rev 3.0; implementation summary spec §14;
   decision trail `docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`),
   screened under a fresh registration with a deliberately small five-arm scope: full
   model, matched `B0-e2e`/f-only, pair+topology (`∅_content_head` permanent), one
   structure-destroyed control (within-pair `Â`/`Π` shuffle), and branch-dropout
   `p = 0`. E2E `B3-full`/`B5`, the conditioning-depth ladder (E4.17), and the
   remaining structure battery (E4.16 b–f) are E1/E3 scope. The e2e screen's
   registration must additionally pin the four-logit decomposition report, the
   representation-probe protocol (including degree-partialled probes), the
   within-checkpoint `f_logit` liveness reference, and a measured H20 cost re-estimate.
   That five-arm scope governed the completed rev-3.0 screen only; the active rev-3.2
   successor uses the seven-arm direct-run contract described below.
   **Binding frozen-s0 result and disposition (2026-07-17):** the replacement
   fixed-Seed-0 run (commit `60745f2`, checkpoint `56b91c17fa8d3b86`, registration
   `97e61a7d...`) completed all required artifacts and returned `cut`. Both guards
   passed, but clustering-MMD and matched BFS-macro GS/RD each failed strict
   dominance against `b0_cal_selfdensity`. A diagnostic epoch-30 evaluation later
   passed matched GS but still failed matched RD/clustering and regressed degree and
   spectral MMD; it does not replace the binding checkpoint. The locked disposition
   is therefore: frozen-s0 scalar fusion becomes a motivating result and E4 ablation
   rung; the rev-3.0 e2e conditioned encoder became the active G5 build line at that
   time. This
   closes the frozen-s0 screen and satisfies spec §14.3(1), without binding the
   successor registration or weakening E1/E3's multi-seed inference rules. Full
   result: `docs/results/G5-stage1-seed0-20260717.md`.
   **Code retirement (implemented in the current worktree; cleanup commit pending).**
   The two-stage cleanup deletes the frozen-s0 `egostitch` pipeline that produced this
   screen — its Stage-1 training/DDP path, `frozen_s0` scoring mode, and diagnostics.
   The producing code last exists at commit `dcae090`. The Stage-gate instantiation and
   dated entries above therefore govern the completed, published screen only and no
   longer describe a runnable active-tree pipeline. Per the spec-freeze rule, this
   implementation authorizes no execution; the successor v4 registration remains
   `DRAFT`. `docs/results/G5-stage1-seed0-20260717.md`,
   `docs/registrations/g5_stage1_preregistration.{json,md}`, and
   `outputs/egostitch_stage1/` remain the citable record regardless of when
   that retirement lands; the two Markdown evidence records name `dcae090` and state
   that deletion is in the current worktree pending a cleanup commit. The binding JSON
   remains unchanged because altering it would change the registered evidence hash. The
   rule that a Stage-1-descended screen carries engineering evidence regardless
   of seed count — inference reserved for E1/E3 under §5.2's ≥3-seed Holm
   procedure plus the §8-spec 30-config HPO-parity budget — is unaffected by
   the code retirement and applies unchanged to any successor ladder.
   **E2E v1 training outcome and prospective replacement (2026-07-19):** the
   registered v1 `full` arm completed its engineering training pipeline, but its
   validation-selected checkpoint came from the reconstruction-only warm-start; the
   subsequent joint phase showed collapsed validation logits and non-finite fixed-probe
   edge-family gradients. No v1 candidate scores, remaining arms, or held-out G5 gate
   result were produced, so this is a training-validity failure rather than a G5
   scientific verdict. The v1 BINDING registration remains immutable. Binding
   execution for any successor arm is governed by spec §13.19's single-stage,
   plan-bound formal run. A
   single 512-node internal topology holdout `V_hold` (:= the union of the
   former `V_qual` and `V_select`; `V_fit` is unchanged from the two-holdout
   definition) supplies validation and checkpoint selection for the run.
   Measured prevalence, disclosed rather than hidden: `V_hold` is 0.0117
   (1,533 positives / 130,816 pairs), against `V_select` alone at 0.0247 (807
   positives / 32,640 pairs) — the union raises positives 1.9x but halves
   prevalence. `e2e_checkpoint_eligible` — the post-ramp/
   phase-C restriction, the AUPRC floor at `prevalence + 0.02`, and the
   residual / logit-std / topology-gradient conditions — is computed as
   run telemetry and cannot block completion, publication, scoring, or evaluation.
   Because the union halves prevalence, that
   floor's absolute value drops accordingly: 0.0447 on `V_select` alone to
   0.0317 on `V_hold`. Checkpoint selection applies the registered AUPRC/MMD/Brier
   rule to every completed epoch; quality-predicate misses are recorded only. The formal stage
   carries pre-registered
   acceptance thresholds recorded before any held-out read (§5.2.4), and its
   result is engineering evidence, not inference, at any seed count — only
   E1/E3 (≥3 seeds, Holm-corrected, plus the §8-spec 30-config HPO-parity
   budget) carry significance or cross-seed-robustness claims (§5.0.5).
   DRAFT/debug artifacts remain forbidden from candidate/test scoring, not
   merely from the final gate.
   **E2E v2 binding and verdict (2026-07-23/2026-07-24):** the v2 replacement
   qualified and bound under registration
   `g5-e2e-stage1-20260719-conditioned-encoder-stability-screen-v2` (binding SHA-256
   `7937d8bb...`, implementation commit `c878a939...`) on 2026-07-23 after satisfying
   the then-current spec §13.19 requirements. All four registered arms
   (`full`/`b0_e2e_f_only`/`pair_topology`/`p0`, plus the `structure_control_6a`
   shuffle control over `full`'s checkpoint) trained 2026-07-23/2026-07-24 on 4 × H20
   (`world_size=4`). Training was valid — the within-checkpoint liveness death rule
   did not fire — and the formal gate published 2026-07-24 23:51 UTC with verdict
   `cut` (multi-label): of the three primary criteria only BFS-macro GS passed, the
   matched-AUPRC guard failed, pathway attribution established no gain
   (`G_full = clu(b0_e2e_f_only) - clu(full) = -0.1980648`), and the
   structure-destruction control's paired-bootstrap lower bound was `0.0` (fail).
   This closes the e2e Stage-1 screen; the rev-3.0 build line's disposition is an
   owner-side locked-decision discussion and is not resolved by this screen. Full
   result: `docs/results/G5-e2e-stage1-seed0-20260724.md`.
   **Historical rev-3.1 relational-repair proposal (retired):** the disposition discussion resolved
   under owner delegation: the rev-3.0 build line proceeds as the rev-3.1 repair
   (spec §14.4; decision trail
   `docs/superpowers/specs/2026-07-25-egostitch-e2e-relational-repair-design.md` r5,
   P0 audits `outputs/p0_audit_20260725/`). Its screen is governed prospectively by
   the then-fresh v4 registration, with an **eight-arm** scope — six
   trained arms (full, `B0-e2e`/f-only,
   pair+topology, `p = 0`, `cosine_pool`, `no_l_rel`) plus two scoring-time
   controls over the full checkpoint (within-pair rebuild-form shuffle 6a-v3 and
   degree-preserving rewiring 6e-v1) — under one plan-bound formal run that trains on
   `V_fit` and validates/selects on `V_hold`. Formal execution records the exact v4
   registration snapshot and validates arm/config/runtime identity; registration status
   and nullable run-evidence placeholders do not gate execution. Concrete provenance is
   generated by the run and verified during scoring/evaluation; no model-quality
   threshold authorizes or blocks execution. The frozen
   pairwise scorer's §0 role is **unchanged** (the measured grounding-pool audit
   eliminated the candidate-proposer reuse), and this paragraph does not bind the
   v4 registration or weaken E1/E3's multi-seed inference rules. This proposal
   never became the active contract; the registration machinery is deleted.
   **Current rev-3.2 component-ablation schema:** five trained checkpoints
   (`full`, `b0_e2e_f_only`, `p0`, `no_l_rel`, `row_layernorm`) plus two
   scoring-time structure controls. The content pathway is deleted, making
   `pair_topology` identical to `full`, so that arm is retired; `cosine_pool` is
   also retired. Runs execute directly without a registration or plan-identity
   gate, and checkpoints/score artifacts carry no training-data identity field
   (spec §12, 2026-08-03) — keeping shared-edge runs apart from older
   80/20-partition ones is an owner-side responsibility, not a gate.

### 5.1 Forward priority order

1. **E2 (hardened)** — established the gap under G1 conditions.
2. **Distillation baseline** — run B1, including CAZI-MBN, and measure Oracle-gap recovery.
3. **Parallel probes** — compare B2/B3 and the structured-loss control against B0.
4. **Method selection** — assign no `Ours` until the matched comparison is complete.
5. **Selected-method study** — main result, baselines, and mechanism ablations.
6. **Extensions** — downstream utility and split/benchmark generalization.

Within each experiment, per-arm execution is a single sequence, not train followed by a
separate manual scoring/reporting step: `hpc/run.sh train <config> ...` packs, trains,
publishes, then automatically scores V_hold, freezes the max-F1 operating point, scores
test and candidate, and writes the combined edge+graph `test_report.json` before the
command returns (`hpc/README.md`). A debug `--max-steps` run skips this test stage
entirely. CAZI-MBN reaches the same report through `hpc/run.sh test`; two historical
structure controls reuse the `full` EgoStitch checkpoint and also run through `test`.

### 5.2 Decision rules (fixed before held-out metrics are opened)

1. Assign `Ours` only after selection. If it does not beat Controls and `B0+cal` on
   held-out topology at matched AUPRC (Holm, ≥3 seeds), its topology is not load-bearing.
2. If a candidate's topology-context knockout costs nothing, do not select that mechanism.
3. Multiple-comparison control: Holm over the held-out assembled family replaces any
   "wins on k of n metrics" language.
4. All thresholds and rules recorded in run metadata before E1/E3 held-out metrics are opened.

### Caching

- Cache frozen node features once per benchmark split.
- Cache audited, universe-scoped ANN lists only for retrieval arms; disclose support.
- Cache teacher targets or generated latents only for candidates that implement them.
- Cache pairwise outputs for every candidate universe used by E2, static denoising,
  `B0+cal`, and the selected candidate.
- Store run metadata with checkpoint identifier, threshold policy, candidate settings, seed,
  split name, metric normalization, kernel/bandwidth disclosures, HPO budget records, and the
  §5.2 decision rules.

### Reproducibility

- At least three seeds for E1/E3 means.
- Lock the operating-point policy and decision rules before opening final held-out metrics.
- Preserve enough metadata to regenerate [e2-gap.html](../figures/e2-gap.html) and the main
  experiment tables from repository-local artifacts.
- Report FLOPs/wall-clock for B0 and candidates; sweep only applicable hyperparameters.

---

## 6. Deliverables and done criteria

1. **Gate reports (G1–G3):** archived G1 (including historical B0-alt), G2 ceiling, and
   the G3 Oracle row are complete; the alternate architecture preserves the topology gap.
2. **Open model/spec:** finalize `04`/`05` only after matched method selection.
3. **Selected-method table:** compare candidates by edge AUPRC, assembled metrics, ceiling/Oracle/noise
   rows, diagnostics.
4. **Mechanism ablation table:** each mechanism and its edge/topology effect; the pruned submitted
   model.
5. **Integrity appendix:** all gates and controls.
6. **Downstream-utility result:** load-bearing.
7. **Extension results:** split generalization.
8. **Positioning figure:** [positioning.html](../figures/positioning.html), extended with the
   settings-taxonomy axis (transductive / observed-neighborhood inductive / KG-inductive /
   strict zero-edge).

Historical EgoStitch-arm decisions (not forward method-selection decisions):

- That arm uses generation-primary scaffolds with gated ANN grounding (04 §4.2).
- Scaffold edges are learned (generated + harmonized), not fixed from the frozen scorer;
  the frozen-scorer variant is E4.10.
- B2-global is reported as a baseline; an "Ours-global" arm is not planned.
- Canonical metric normalization: fixed by G1 as the ratio of size-mean raw biased MMD²
  to the deterministic real-vs-real reference MMD², and recorded in run metadata.
