# Experiment Protocol: Topology-Conditioned Inductive Edge Prediction

**Status:** repository-local experiment plan for an abstract graph ML benchmark.
**Updated 2026-07-09:** incorporates the approved `[protocol-Δ]` items from
`04-model-proposal.md` revision 2.1 (approved as a design proposal; **gate G4 was
signed off later the same day — `06-egostitch-spec.md` is now the implementation
contract**, including its §9 benchmark binding / data contract and §10–11
batch-sampler and four-H20 E2 production execution design). Changes:
pre-implementation gates G1–G5, baseline ladder extensions (`B0+cal`, `B3-dist`,
`B3-full`, `B5`, external attribute-only baselines), trained-on vs held-out metric
families, ceiling and Oracle reference rows, hard-negative construction for zero-edge
nodes, new integrity gates, Holm-corrected pre-registered decision rules, E7 promoted
to load-bearing, and B4's disposition. The C4 framing claim is scoped throughout as:
*the first **protocol-gated** joint edge + assembled-realism evaluation for **strict
zero-edge inductive** edge prediction* (classical priority: ERGM goodness-of-fit;
deep-learning priority on the dissociation observation: Graph Gestalt 2106.15239).

This document defines the benchmark contract, baseline hierarchy, experiment matrix, evaluation
protocol, caveats, and deliverables for this repository. It is intentionally self-contained:
all artifact references are relative to this repository, and all dataset/model names are neutral
placeholders.

> **Locked contract.** The method is a **per-query local scaffold**. For each queried pair
> `(i, j)`, build a local topological context from frozen node features, condition the pair
> classifier on that scaffold, and emit a binary 0/1 edge label. Evaluation aggregates many
> per-query predictions into a full predicted graph and grades the assembled graph with topology
> metrics. Per-query locality is the model boundary; graph-scale assembly is only an evaluation
> aggregation.
>
> **Known structural consequence (stated, not hidden):** under this contract the assembled
> graph is an edge-independent construction given features. Gate G2 measures the realism
> ceiling this implies (Chanpuriya et al. 2111.00048) before implementation, and every
> assembled-metric table carries the ceiling and Oracle reference rows (§4).

---

## 0. Local scaffold contract

The useful comparison is not "better edge classifier" alone. The benchmark asks whether a model
can preserve edge-level quality while assembling predictions into a plausible graph.

### Method boundary

For a query pair `(i, j)`, the original instantiation of the boundary was:

```text
V_T = {i, j} union ANN_feat(i, k) union ANN_feat(j, k)
E_T = {(a, b) in V_T x V_T : score(a, b) >= tau_pair}, with weights score(a, b)
T_ij = (V_T, E_T without queried edge (i, j))
H = scaffold_encoder(x_V_T, T_ij)
p_ij = sigmoid(pair_logit(i, j) + scaffold_residual(H_i, H_j))
```

**Disposition (2026-07-09):** this retrieved-and-thresholded instantiation is retained as
**ablation arm E4.10** (the bridge baseline). The method under test (`Ours`) is the
generated-and-harmonized scaffold defined in `04-model-proposal.md` §4 with the frozen
algorithm spec in `06-egostitch-spec.md`; the *outer* boundary above (per-query local
context from frozen features → conditioned classifier → binary label) is unchanged and
remains locked.

Components:

| Component | Status under the contract |
|---|---|
| Frozen feature encoder | Reuse as frozen node features `x_i`; never fine-tune in the core protocol. |
| Frozen pairwise scorer | Reuse as B0, as the `s0` anchor, and (E4.10 only) as the edge-weight/candidate proposer. Provenance audited (§E5). |
| Scaffold construction | The method under test: generated + harmonized (`Ours`); retrieved + thresholded (E4.10). |
| Scaffold encoder plus fused decision head | Reuse as the scaffold-conditioned classifier head; identical-head convention for generator comparisons (§3 E4). |
| Global generated graph encoding | Report only as a baseline or ablation on the locality axis. |
| Benchmark splits, evaluator, and assembled graph metrics | Reuse consistently for all methods. |

### Why local scaffolds are the primary method

- They answer the inductive query directly: "given two unseen nodes and frozen features, should an
  edge exist?"
- They avoid requiring a whole candidate universe before scoring one pair.
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
  filter used by the benchmark must be reported with the run. Training-graph edges are further
  split into **message (80%) / supervision (20%)** partitions per seed: structural targets use
  message edges only; edge-loss supervision uses supervision edges (spec §6).
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
  - *Composite:* "graph similarity" may be quoted only after its definition (components,
    weights, normalization, direction) is published in the run metadata (gate G1); components
    are always reported alongside.
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
    also disclosed separately. Histograms are normalized by `sum + 1e-6`. MMD² is
    the biased V-statistic under
    `k(x,y)=exp(-(0.5·||x-y||₁)²/2)` (`σ=1`, including within-sample diagonals).
  - *Reference normalization:* within every node-size bucket, reference samples
    retain artifact order and are split as `samples[::2]` versus `samples[1::2]`.
    The reported statistic is
    `mean_size MMD²(pred_size,ref_size) / mean_size MMD²(ref_even,ref_odd)`.
    Numerator, denominator, and ratio are all stored; the ratio is canonical and
    lower is better. A `1e-12` denominator floor is only a numerical guard.
- **Repository artifacts:** this protocol, [E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md),
  [e2-gap.html](../figures/e2-gap.html), [positioning.html](../figures/positioning.html),
  [04-model-proposal.md](04-model-proposal.md), [06-egostitch-spec.md](06-egostitch-spec.md).

---

## 2. Baseline hierarchy

| ID | Baseline | Neutral instantiation | Isolates |
|---|---|---|---|
| **B0** | Independent pairwise scorer | Frozen pairwise scorer over `(x_i, x_j)` | Topology-blind edge prediction |
| **B0-alt** | Second pairwise scorer | Alternate pairwise architecture trained under the same split | Architecture-independence of the gap |
| **B0+cal** | B0 + calibrated assembly | Temperature/Platt calibration + density- and degree-sequence-matched thresholding of B0 scores | Whether trivial assembly calibration recovers the topology gains. Also applied on top of `Ours` as a diagnostic |
| **B1** | Retrieval, no generated adjacency | Retrieve neighbors around `i` and `j`, pool features, use no scaffold edges | Value of adjacency vs retrieval alone |
| **B2-global** | Global post-hoc refiner | Build one generated graph from pairwise scores, encode globally, decode each pair | Global context vs local context |
| **B2-static** | Static graph denoiser | Top-k, density-matched thresholding, or other fixed sparsification over B0 scores | Whether simple graph cleanup is enough |
| **B3** | Topology-aware loss on the pairwise scorer | Pairwise scorer trained with edge loss plus topology regularizers | Whether a topology loss alone fixes assembly |
| **B3-dist** | Distributional-loss scorer | B0 fine-tuned with a kernel-MMD realism regularizer (Graph Gestalt 2106.15239 kernels) | Whether a one-line distributional loss closes the gap without topology conditioning |
| **B3-full** | The Ockham arm | B0 architecture + all of `Ours`'s auxiliary supervision as multi-task heads (degree NLL, BP-NLL, ego-net statistic prediction, distributional statistic loss) + calibrated assembly | **The decisive control**: every training signal, none of the generative machinery. The E2 §5 cached preview is treated as a first-class hypothesis in this arm's favor and re-run under G1 normalization |
| **B4** | *(disposition)* | Subsumed by ablation arm E4.11 (generation-only, no queried-edge supervision) | Recorded here so the blueprint/methodology B4 row is not silently dropped |
| **B5** | Neural-SBM residual | Feature-only MLP affiliations `F_u` (NOCD Bernoulli–Poisson) + degree terms: `p_ij = σ(pair_logit + block + degree)` | Block prior alone; the project's null hypothesis. Also the first strict-inductive evaluation of the BP block prior |
| **PA-null** | Preferential-attachment null | `s_ij = k_i·k_j` from training-side degree statistics, reported with each benchmark's degree heterogeneity σ (log-normal fit) | Validity precondition (2405.14985): under uniform negatives this null averages AUC 0.83; any method not clearly beating it has an uninformative edge metric |
| **Odds-product** | Degree-respecting edge-independent baseline | `P_ij = σ(ℓ_i + ℓ_j)` fitted to the expected degree sequence (Chanpuriya 2111.00048 §3) | Cheapest degree-budget-honoring assembly with zero topology conditioning; also supplies the G2 overlap dial `P̃ = (1−ω)P + ωA` |
| **DEAL / Graph2Gauss** | External attribute-only baselines | DEAL 2007.08053; Graph2Gauss 1707.03815 | Independent-scoring SOTA for the setting; falsifiable prediction: both exhibit the E2 failure mode |
| **Ours** | Per-query generated local scaffold + conditioning | EgoStitch per `04-model-proposal.md` §4 / `06-egostitch-spec.md` | Conditioning the decision on generated, harmonized local topology |
| **Oracle** | Observed-graph upper bound | Uses true held-out graph neighborhoods at evaluation time | Headroom; violates the inductive protocol; **run first (gate G3)** |

All fair baselines share the same frozen features, splits, query sets, message/supervision
partition, negative-sampling protocol, and an equal, pre-registered HPO budget (30 configs ×
3 seeds each, recorded before held-out metrics are opened).

---

## 3. Experiment matrix

Each experiment lists purpose, claim, run, metrics, and success criterion. The run order is
prioritized in Section 5 (gates first).

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

### E2: Pair-to-topology gap (hardened per gate G1)

- **Claim:** an independent pairwise scorer can assemble into an implausible graph, and the
  degradation is not explained by threshold choice, negative-sampling regime, or metric
  normalization.
- **Run (G1 requirements):** one frozen scorer family (same as `Ours` uses), one candidate
  universe, one canonical metric normalization, a true threshold sweep (recall/density-vs-MMD
  curves), easy, hard (HeaRT-style), **and degree-corrected** (2405.14985) negatives, B0-alt
  replication, the PA-null row, real-vs-real noise floor, bootstrap variance, a defined
  "graph similarity" composite **that passes an expressivity/robustness perturbation check
  (O'Bray 2106.01098) before it ranks anything**, and the full-candidate-universe imbalance
  view.
- **Execution acceptance:** E2 uses a fixed 30-epoch four-H20 throughput run. Validation
  is executed after every epoch; quality is reported but is not the throughput acceptance gate
  for the first systems-optimization pass. The wall-clock gate is 60 minutes from an empty
  derived-cache path through final training artifacts.
- **Result (2026-07-12):** G1 passed its execution requirements. The B0 degree-corrected
  ratio-1 row is AUROC 0.716871 / AUPRC 0.742622; the hard-heuristic row is
  0.583741 / 0.620193; the hard-feature row is 0.406667 / 0.475048; and the full
  candidate-universe row is 0.710776 / 0.123982. At the density-matched operating point,
  relative density is 0.985429 and the perturbation-validated graph-similarity composite
  is 9.64858e-10, with degree/clustering/spectral MMD² of 0.620493/0.729620/0.836370.
  The gap therefore survives hard negatives and calibrated thresholds, although PA-null
  beats B0 in the easy and feature-hard rows and is now a mandatory control.
- **G2 result:** the cached soft scorer has `Ov(P)=0.654778`, while `Ov_min=0.010038`
  at matched volume is sufficient to reach the reference triangle count; the full ceiling
  curve and caveats are in the synced G2 artifact. **G3 (Oracle) remains pending.**

### E3: Baselines head-to-head on Benchmark-A

- **Claim:** `Ours` beats independent scoring, calibration, global refinement, static denoising,
  and all loss-shaping baselines under one protocol.
- **Run:** B0, B0-alt, `B0+cal`, B1, B2-global, B2-static, B3, `B3-dist`, `B3-full`, B5,
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
10. **Retrieved-thresholded scaffold (E4.10):** the original §0 instantiation — bridge baseline.
11. **Generation-only (E4.11):** no queried-edge supervision (subsumes blueprint B4).
12. **Harmonization:** `R = 0` vs rounds sweep; mask-schedule sweep; slot-agreement trajectory;
    hard-budget masking on/off.
13. **Grounding:** grounded vs ungrounded (headline arm); grounding-only vs imagination-only;
    conditioning-dropout rates for both nulls with counterfactual contrasts reported.
14. **Channels:** per-channel knockouts (s1/s2/s3/s4) with the s-channel correlation matrix.

**Identical-head convention:** all generator comparisons (E4.6, E4.10, Oracle-scaffold) use the
same fused decision head with the input-mapping convention of `06-egostitch-spec.md` (retrieved
neighbors as π=1, m=1 slots; Π from identity matches), head retrained per arm under the matched
tuning budget.

**Success:** the ablations identify which mechanism produces the topology gain without obscuring
edge-level quality; any mechanism that owns no gain is cut from the submitted model (04 §4.6
rule).

### E5: Integrity and leakage gates

- **Node-disjoint gate:** train/test nodes must be disjoint.
- **Near-duplicate gate:** report benchmark-provided duplicate filtering or run a repository-local
  nearest-neighbor overlap check.
- **Message/supervision partition gate:** all structural losses on message edges only;
  leave-one-out corrections verified for in-batch supervision pairs.
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

- **Edge-level task:** score held-out candidate pairs and report probability-based AUROC/AUPRC plus
  thresholded metrics at the selected operating point; probabilities are the n_s-averaged values
  per the spec determinism policy.
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
- **Mechanism-transmission diagnostics (required with every headline table):** held-out
  imagined-ego-net fidelity (slot recall@K, degree calibration, slot-adjacency vs true local
  clustering); assembled degree-calibration curve; imagined-diversity vs real neighbor
  diversity; mean grounding gate; s-channel correlation matrix.
- **Reporting rule:** edge and assembled graph metrics are always reported together. No claim
  relies on one metric family alone. The held-out family is headlined.

---

## 5. Run order, gates, caching, and reproducibility

### 5.0 Pre-implementation gates (blocking; before any model code)

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
4. **G4 — Specification freeze.** `06-egostitch-spec.md` reviewed and signed off; it then
   becomes the implementation contract. **Done 2026-07-09** (sign-off recorded in the spec's
   change log; the spec's §9 data contract additionally quarantines the shipped
   `*_ratio5_exclusive.txt` negatives and `train_graph.pkl`-as-target — both leak across the
   split under the strict gate).
5. **G5 — Staged build.** Stage 1: imagination + degree budget + closure only; Stage 2: +
   codebook + s3; Stage 3: + harmonization + seam loss. Each stage must beat the previous on
   the pre-registered criteria or the added mechanism is cut.

### 5.1 Priority order (after gates)

1. **E2 (hardened)** — establish the gap under G1 conditions.
2. **E1 and E3** — main method and head-to-head baselines on Benchmark-A.
3. **E4** — the ablations needed to explain E1/E3.
4. **E5** — integrity gates completed before broad claims.
5. **E7** — downstream utility (load-bearing; runs with, not after, breadth).
6. **E6** — split/benchmark generalization.

### 5.2 Pre-registered decision rules (frozen before held-out metrics are opened)

1. If `Ours` does not beat `B3-full` and `B0+cal` on the held-out assembled family at matched
   edge AUPRC (Holm-corrected, ≥ 3 seeds), the generative apparatus is declared not
   load-bearing; the paper pivots to the benchmark/`B3-full` story.
2. If s1/s2 knockouts cost nothing on every benchmark and stratum, `Ours → B5` is declared and
   the honest small-paper outcome ("calibrated block-model marginals close most of the
   pair-to-topology gap") is written.
3. Multiple-comparison control: Holm over the held-out assembled family replaces any
   "wins on k of n metrics" language.
4. All thresholds and rules recorded in run metadata before E1/E3 held-out metrics are opened.

### Caching

- Cache frozen node features once per benchmark split.
- Cache approximate-nearest-neighbor lists for grounding pools.
- Cache per-node Tokenize/Imagine outputs (spec §1–2) for reuse across queries.
- Cache pairwise scorer outputs for every candidate universe used by E2, B2-global, B2-static,
  `B0+cal`, and `Ours`.
- Store run metadata with scorer checkpoint identifier, threshold policy, scaffold size, seed,
  split name, metric normalization, kernel/bandwidth disclosures, HPO budget records, and the
  §5.2 decision rules.

### Reproducibility

- At least three seeds for E1/E3 means.
- Lock the operating-point policy and decision rules before opening final held-out metrics.
- Preserve enough metadata to regenerate [e2-gap.html](../figures/e2-gap.html) and the main
  experiment tables from repository-local artifacts.
- Report the FLOPs/wall-clock table (per-node cached, per-pair marginal, full-universe assembly)
  for B0, E4.10, and `Ours` at R ∈ {0, 2, 4}.

---

## 6. Deliverables and done criteria

1. **Gate reports (G1–G3):** G1 hardened E2 note and G2 ceiling computation are complete;
   the G3 Oracle row remains outstanding.
2. **Spec freeze (G4):** signed-off `06-egostitch-spec.md`.
3. **E1/E3 main table:** method by edge AUPRC, both assembled families, ceiling/Oracle/noise
   rows, diagnostics.
4. **E4 ablation table:** each mechanism and its edge/topology effect; the pruned submitted
   model.
5. **E5 integrity appendix:** all gates and controls.
6. **E7 downstream-utility result:** load-bearing.
7. **E6 extension results:** split generalization.
8. **Positioning figure:** [positioning.html](../figures/positioning.html), extended with the
   settings-taxonomy axis (transductive / observed-neighborhood inductive / KG-inductive /
   strict zero-edge).

Resolved decisions (formerly open):

- Candidate source for scaffolds: generation-primary with gated ANN grounding (04 §4.2).
- Scaffold edges are learned (generated + harmonized), not fixed from the frozen scorer;
  the frozen-scorer variant is E4.10.
- B2-global is reported as a baseline; an "Ours-global" arm is not planned.
- Canonical metric normalization: to be fixed by gate G1 and recorded in run metadata (the one
  remaining open item, deliberately owned by G1).
