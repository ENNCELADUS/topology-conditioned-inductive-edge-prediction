# Experiment Protocol: Topology-Conditioned Inductive Edge Prediction

**Status:** repository-local experiment plan for an abstract graph ML benchmark.

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

---

## 0. Local scaffold contract

The useful comparison is not "better edge classifier" alone. The benchmark asks whether a model
can preserve edge-level quality while assembling predictions into a plausible graph.

### Method boundary

For a query pair `(i, j)`:

```text
V_T = {i, j} union ANN_feat(i, k) union ANN_feat(j, k)
E_T = {(a, b) in V_T x V_T : score(a, b) >= tau_pair}, with weights score(a, b)
T_ij = (V_T, E_T without queried edge (i, j))
H = scaffold_encoder(x_V_T, T_ij)
p_ij = sigmoid(pair_logit(i, j) + scaffold_residual(H_i, H_j))
```

Components:

| Component | Status under the contract |
|---|---|
| Frozen feature encoder | Reuse as frozen node features `x_i`; never fine-tune in the core protocol. |
| Frozen pairwise scorer | Reuse as B0 and as the edge-weight/candidate proposer inside each scaffold. |
| Scaffold encoder plus residual pair head | Reuse as the scaffold-conditioned classifier head. |
| Global generated graph encoding | Report only as a baseline or ablation on the locality axis. |
| Per-query local scaffold `T_ij` | The method under test. |
| Benchmark splits, evaluator, and assembled graph metrics | Reuse consistently for all methods. |

### Why local scaffolds are the primary method

- They answer the inductive query directly: "given two unseen nodes and frozen features, should an
  edge exist?"
- They avoid requiring a whole candidate universe before scoring one pair.
- They make the global-vs-local context comparison explicit.
- They amortize cleanly: feature-neighbor caches can be reused across many queried pairs.

---

## 1. Fixed experimental substrate

- **Benchmark:** **Benchmark-A**, the primary node-disjoint split used throughout this repository.
  Additional split families are **Benchmark-B** and **Benchmark-C**.
- **Node features:** frozen feature encoder outputs cached as repository-local artifacts. The
  protocol treats these vectors as fixed inputs and does not update the encoder.
- **Candidate universe:** all candidate node pairs for the held-out benchmark split. Labels define
  the edge prediction task only; graph topology is evaluated after assembling predictions.
- **Splits and integrity:** training and test nodes are disjoint. Any near-duplicate or overlap
  filter used by the benchmark must be reported with the run.
- **Edge-level metrics:** AUROC, AUPRC, accuracy, sensitivity, specificity, precision, recall, F1,
  and MCC at the selected operating point.
- **Assembled graph metrics:** graph similarity, relative density, degree-distribution MMD,
  clustering-coefficient MMD, and Laplacian-spectrum MMD over benchmark node buckets.
- **Repository artifacts:** this protocol, [E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md),
  [e2-gap.html](../figures/e2-gap.html), and [positioning.html](../figures/positioning.html).

---

## 2. Baseline hierarchy

| ID | Baseline | Neutral instantiation | Isolates |
|---|---|---|---|
| **B0** | Independent pairwise scorer | Frozen pairwise scorer over `(x_i, x_j)` | Topology-blind edge prediction |
| **B0-alt** | Second pairwise scorer | Alternate pairwise architecture trained under the same split | Architecture-independence of the gap |
| **B1** | Retrieval, no generated adjacency | Retrieve neighbors around `i` and `j`, pool features, use no scaffold edges | Value of adjacency vs retrieval alone |
| **B2-global** | Global post-hoc refiner | Build one generated graph from pairwise scores, encode globally, decode each pair | Global context vs local context |
| **B2-static** | Static graph denoiser | Top-k, density-matched thresholding, or other fixed sparsification over B0 scores | Whether simple graph cleanup is enough |
| **B3** | Topology-aware loss on the pairwise scorer | Pairwise scorer trained with edge loss plus topology regularizers | Whether a topology loss alone fixes assembly |
| **Ours** | Per-query local scaffold + conditioning | Local `T_ij` plus scaffold-conditioned residual head | Conditioning the decision on local topology |
| **Oracle** | Observed-graph upper bound | Uses true held-out graph neighborhoods at evaluation time | Headroom; violates the inductive protocol |

---

## 3. Experiment matrix

Each experiment lists purpose, claim, run, metrics, and success criterion. The run order is
prioritized in Section 5.

### E1: Main result on Benchmark-A

- **Claim:** `Ours` matches or improves B0 on edge AUROC/AUPRC and improves assembled graph
  topology.
- **Run:** train `Ours` on Benchmark-A. Evaluate both edge-level predictions and assembled graph
  metrics. Compare against B0, B2-global, and B3.
- **Metrics:** AUROC, AUPRC, MCC, graph similarity, relative density, degree MMD, clustering MMD,
  and spectral MMD.
- **Success:** edge AUPRC is within run noise of B0 or better; graph similarity and MMDs improve
  over B0 and are competitive with or better than B2-global; relative density moves closer to 1.

### E2: Pair-to-topology gap

- **Claim:** a strong pairwise scorer can assemble into an implausible graph, and the degradation
  tracks operating-point behavior such as recall and density.
- **Run:** evaluate the frozen B0 scorer, assemble its predictions into a graph, compute assembled
  graph metrics, and sweep operating points when cached scores are available.
- **Metrics:** edge AUROC/AUPRC, precision, recall, graph similarity, relative density, degree MMD,
  clustering MMD, and spectral MMD.
- **Success:** a figure and result note show high edge scores coexisting with poor assembled graph
  metrics. The local artifact is [e2-gap.html](../figures/e2-gap.html), with details in
  [E2-pair-to-topology-gap.md](results/E2-pair-to-topology-gap.md).

### E3: Baselines head-to-head on Benchmark-A

- **Claim:** `Ours` beats independent scoring, global refinement, static denoising, and topology
  loss baselines under one protocol.
- **Run:** B0, B0-alt, B1, B2-global, B2-static, B3, `Ours`, and Oracle on the same split and
  operating-point policy.
- **Metrics:** the joint edge/topology table from E1.
- **Success:** `Ours` is on the Pareto frontier of edge AUPRC and graph similarity. B2-global is
  the closest competitor, and `Ours` wins on at least 3 of 5 assembled graph metrics without a
  meaningful edge AUPRC loss.

### E4: Mechanism ablations

1. **No scaffold:** retrieval features only, no local edges.
2. **Randomized scaffold:** same node count and edge count, shuffled edges.
3. **Retrieval-only adjacency:** feature-neighbor edges without pairwise scorer weights.
4. **Global vs local context:** B2-global against `Ours`.
5. **Scaffold size `k`:** sweep neighbors per endpoint, such as 5, 10, 20, and 40.
6. **Scaffold encoder variant:** default scaffold encoder against simpler message-passing and
   masked-autoencoding variants.
7. **Fake-edge isolation:** remove vs keep queried edge `(i, j)` inside `T_ij`.
8. **Loss terms:** edge-only, plus graph similarity, plus density, plus distributional topology,
   and full objective.
9. **Residual anchor weight:** vary how strongly the local scaffold residual can move the frozen
   pairwise score.

**Success:** the ablations identify which part of the local scaffold produces the topology gain
without obscuring edge-level quality.

### E5: Integrity and leakage gates

- **Node-disjoint gate:** train/test nodes must be disjoint.
- **Near-duplicate gate:** report benchmark-provided duplicate filtering or run a repository-local
  nearest-neighbor overlap check.
- **Hard negatives:** compare random negatives with topology-aware hard negatives.
- **Memorization controls:** include nearest-neighbor, degree-only, and random-vector controls.
- **Success:** `Ours` gains survive hard negatives and are not explained by trivial memorization or
  benchmark shortcuts.

### E6: Split and benchmark generalization

- **Claim:** gains are not specific to Benchmark-A.
- **Run:** repeat E1 for Benchmark-B, Benchmark-C, and alternate split samplings when available.
- **Success:** directionally consistent topology improvements, with any degradation reported rather
  than hidden.

### E7: Downstream graph utility

- **Claim:** a topology-realistic predicted graph supports graph-level downstream tasks better than
  a topology-blind pairwise assembly.
- **Run:** feed predicted graphs from B0, B2-global, and `Ours` into repository-local downstream
  graph tasks when available.
- **Success:** `Ours` improves graph-task metrics over B0 and B2-global, or the document clearly
  reports the failure mode.

---

## 4. Evaluation protocol

- **Edge-level task:** score held-out candidate pairs and report probability-based AUROC/AUPRC plus
  thresholded metrics at the selected operating point.
- **Assembled graph task:** assemble scored pairs into a predicted graph and evaluate graph
  similarity, relative density, degree MMD, clustering MMD, and spectral MMD on benchmark buckets.
- **Operating point:** primary results use a validation-selected threshold. Secondary results use
  density-matched and threshold-sweep views to distinguish score quality from assembly density.
- **Imbalance view:** report PR curves or operating-point summaries under the full candidate
  universe when it differs from balanced edge evaluation.
- **Calibration:** include ECE or Brier score when probabilities are used as decision scores.
- **Reporting rule:** edge and assembled graph metrics are always reported together. No claim should
  rely on one metric family alone.

---

## 5. Run order, caching, and reproducibility

Priority order:

1. **E2:** establish the pair-to-topology gap using cached B0 predictions and assembled graph
   metrics.
2. **E1 and E3:** run the main method and head-to-head baselines on Benchmark-A.
3. **E4:** run only the ablations needed to explain E1/E3.
4. **E5:** complete integrity gates before broad claims.
5. **E6 and E7:** add breadth after the primary result is stable.

Caching:

- Cache frozen node features once per benchmark split.
- Cache approximate-nearest-neighbor lists for scaffold construction.
- Cache pairwise scorer outputs for every candidate universe used by E2, B2-global, B2-static, and
  `Ours`.
- Store run metadata with scorer checkpoint identifier, threshold policy, scaffold size, random
  seed, split name, and metric normalization.

Reproducibility:

- Use at least three seeds for E1/E3 means when compute permits.
- Lock the operating-point policy before opening final held-out metrics.
- Preserve enough metadata to regenerate [e2-gap.html](../figures/e2-gap.html) and the main
  experiment tables from repository-local artifacts.

---

## 6. Deliverables and done criteria

1. **E2 gap figure:** high edge AUROC/AUPRC vs poor assembled graph metrics, plus a
   recall/density/topology trend.
2. **E1/E3 main table:** method by edge AUPRC, graph similarity, relative density, degree MMD,
   clustering MMD, and spectral MMD.
3. **E4 ablation table:** each scaffold mechanism and its edge/topology effect.
4. **E5 integrity appendix:** leakage gates and simple controls.
5. **E6/E7 extension results:** split generalization and downstream graph utility when available.
6. **Positioning figure:** a repository-local figure placing `Ours` in the inductive and
   structure-aware cell; current artifact: [positioning.html](../figures/positioning.html).

Open implementation decisions:

- Candidate source for `T_ij`: feature-neighbor retrieval, score-topk retrieval, or a hybrid.
- Whether scaffold edges are fixed from the frozen pairwise scorer or learned by a separate sampler.
- Whether B2-global is reported only as a baseline or also as an "Ours-global" ablation arm.
- Which metric normalization is canonical for assembled graph MMDs.
