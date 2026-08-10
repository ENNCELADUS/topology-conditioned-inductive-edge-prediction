# ICLR 2027 Paper Blueprint

**Task name:** Topology-Conditioned Inductive Edge Prediction

**Method name:** not yet selected; grounding/retrieval is one optional comparison arm.

**Working titles:**

- *Topology-Conditioned Inductive Edge Prediction from Frozen Node Features*
- *Local Graphs as Context for Inductive Edge Classification*
- *Beyond Independent Pair Scores for Unseen Nodes*

**Status:** research blueprint. The paper is an empirical ML method paper, not a
generic graph-generation paper: inferred topology is intermediate context for
binary edge prediction, and its representation method remains open.

---

## 1. Configuration

| Field | Value |
|---|---|
| Venue | ICLR 2027 |
| Paper type | Empirical ML method |
| Primary task | Inductive binary edge prediction over unseen nodes |
| Input | Queried pair and its two frozen endpoint features `(x_u, x_v)` |
| Intermediate object | Generated or inferred topology representation |
| Output | Edge probabilities and assembled predicted graphs |
| Main comparison | Independent pair scoring vs. topology-conditioned scoring |
| Success criterion | Strong edge metrics without implausible assembled graph topology |

---

## 2. Thesis

Inductive edge prediction should be modeled as topology-conditioned binary
classification, not as independent pairwise scoring. For each queried pair, the
model infers topology context from the two endpoints' intrinsic features, then
predicts the 0/1 label. Latent, deterministic, and explicit-scaffold families remain
candidates; retrieval is a separate arm with additional inference support.

What the reader should believe after reading: the missing ingredient in
feature-only edge prediction is not just a better scorer, threshold, or auxiliary
loss. It is local topology as input-side context for each edge decision.

---

## 3. Research Problem

**Task.** Given unseen nodes with frozen feature vectors, predict whether each
queried pair `(u, v)` has an edge. For a set of queries, assemble the predicted
edges into a graph and evaluate that graph as an output object.

**Incumbent formulation.** Independent pairwise scoring predicts
`P(edge | x_u, x_v)`. This is cleanly inductive but treats edge labels as
conditionally independent once endpoint features are known.

**Failure mode.** Edge labels are coupled in the graph being recovered. A set of
individually plausible edge scores can assemble into a graph with unrealistic
density, degree distribution, clustering, components, or spectrum.

**Proposed change.** Keep the final prediction target unchanged, but condition
the edge classifier on a generated or inferred topology representation. That
representation is context, not the final task.

**Non-goals.**

- Not transductive graph completion.
- Not generic graph generation.
- Not post-hoc denoising after independent scoring.
- Not threshold tuning as the main contribution.

---

## 4. Research Questions and Contributions

| RQ | Question | Contribution |
|---|---|---|
| RQ1 | Why can feature-only pair scoring fail as an assembled graph? | Pair-to-topology gap: edge metrics and graph realism can diverge |
| RQ2 | Can topology inferred from endpoint features condition an edge decision without test-graph access? | Topology-context edge classifier |
| RQ3 | Does topology conditioning improve edge metrics and assembled-output metrics? | Main empirical result |
| RQ4 | Which representation family and component matter? | Matched endpoint-only latent/deterministic/generative comparisons, plus a separate retrieval-grounded arm |
| RQ5 | Does the assembled graph help generic graph ML probes? | Secondary utility evaluation |

The load-bearing contributions are RQ2 and RQ3. RQ1 motivates the problem, and
RQ4 prevents the method from being mistaken for a larger feature-only scorer.

---

## 5. Method Summary

1. **Input.** The complete task input is the two frozen endpoint vectors.
2. **Context.** Infer a latent, deterministic, or explicit topology representation.
3. **Decision.** Predict the edge from endpoint features and topology context.
4. **Assembly.** Combine pair scores only for graph-level evaluation.

Retrieval-grounded scaffolds are a separate arm whose support pool is not task input.

Minimal contract:

```text
inputs:  pair (u, v), frozen endpoint features (x_u, x_v)
context: inferred topology representation T_uv = g(x_u, x_v; theta_g)
output:  p_uv = P(edge(u, v) = 1 | x_u, x_v, T_uv)
```

Training uses queried-edge supervision as the primary loss, with optional
auxiliary losses for local graph realism, topology consistency, and masked-edge
reconstruction on training subgraphs.

---

## 6. Baseline Plan

| ID | Baseline | Interpretation |
|---|---|---|
| B0 | Independent pairwise scorer | Topology-blind baseline |
| B1 | Retrieval-augmented scorer without adjacency | Tests whether extra nodes are enough |
| B2 | Post-hoc denoising of independent scores | Tests cleanup after scoring |
| B3 | Independent scorer plus graph-statistic loss | Tests loss shaping without topology context |
| B4 | Latent-topology model without queried-edge conditioning | Tests generation without the final decision module |
| Selected family | Topology-conditioned classifier | Headline method only after matched family selection |
| Oracle | Classifier using observed target topology | Upper bound that violates protocol |

All fair baselines should share the same frozen features, splits, query sets, and
evaluation protocol.

---

## 7. Evaluation Plan

**Edge-level metrics.**

- AUROC and AUPR.
- Fixed-operating-point accuracy, F1, or balanced accuracy.
- Per-node or per-query-group ranking metrics.
- Calibration metrics.
- Robustness across node-disjoint inductive splits.

**Graph-level assembled-output metrics.**

- Relative density.
- Degree-distribution distance.
- Clustering-coefficient distance.
- Spectral distance.
- Motif or orbit statistic distance.
- Component and connectivity summaries.
- Learned graph-feature distance.

Use density-matched thresholds and threshold sweeps so graph improvements cannot
be explained only by a different operating point. Report edge and graph metrics
together.

**Ablations.**

- Remove topology context.
- Compare latent, deterministic, and explicit-scaffold endpoint-only representations.
- Remove or randomize the topology representation.
- Remove queried-pair conditioning from the topology module.
- For retrieval only: remove adjacency, shuffle candidates, and sweep pool size.
- Remove topology-realism loss.
- Remove topology-consistency loss.
- Remove queried-edge masking.

**Downstream probes.** If the main result is strong, test generic graph ML probes
such as community detection, node-label propagation, clustering, or graph-based
retrieval on the assembled graph.

---

## 8. Paper Structure

1. **Introduction.** Define the task, the pair-to-topology gap, and the proposed
   topology-conditioned formulation.
2. **Related Work.** Cover feature-only edge prediction, inductive link
   prediction, structure-aware graph learning, local graph generation, and graph
   evaluation metrics.
3. **Problem Formulation.** Specify frozen features, queried pairs, binary edge
   labels, inductive splits, and assembled-output evaluation.
4. **Method.** Present the selected topology-representation mechanism, edge
   classifier, losses, and inference; report the matched family-selection evidence.
5. **Experiments.** Compare baselines, report edge and graph metrics, and run
   mechanism ablations.
6. **Discussion.** Cover representation identifiability, stochasticity, optional
   retrieval support, calibration, compute cost, and limits of frozen features.

---

## 9. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Topology generation appears to be the main task | High | Anchor every section to queried-edge binary prediction |
| Generated topology does not improve edge metrics | High | Report edge and graph metrics jointly; use ablations to diagnose |
| Graph metrics improve only by threshold changes | High | Use density-matched operating points and threshold sweeps |
| Optional candidate retrieval leaks target information or is mistaken for task input | High | Keep retrieval in a separately labeled arm, freeze and disclose its support universe, and audit every inference input |
| Scaffold contains the queried-edge answer | High | Mask or standardize the queried edge inside the scaffold |
| Method becomes too broad | Medium | Keep the fixed setting: unseen nodes, frozen features, queried pairs, binary labels |

---

## 10. Locked Decisions

1. Primary problem: topology-conditioned inductive edge prediction.
2. Final prediction target: binary edge label or probability for queried pairs.
3. Generated local topology role: intermediate context, not final output.
4. Main comparison: independent pair scoring vs. topology-conditioned scoring.
5. Evaluation requirement: pairwise edge metrics plus graph-level
   assembled-output metrics.
6. Protocol: input is exactly `(x_u,x_v)`; no target graph; representation method
   remains open and grounding/retrieval is optional support, never task input.
