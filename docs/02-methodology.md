# Methodology Plan: Topology-Conditioned Inductive Edge Prediction

**Status:** concise method plan for a general ML problem. Companion to
`01-blueprint.md`.

**Problem.** Given unseen nodes with frozen feature vectors and a queried node
pair `(u, v)`, predict a binary edge label. For a query set, predict many pair
labels and assemble the predicted graph. The method must be judged both by
pairwise edge metrics and by graph-level metrics on the assembled output.

**Core claim.** Independent pair scoring is topology-blind: it can optimize
`P(edge | x_u, x_v)` while producing an assembled graph with implausible degree,
density, clustering, or spectral structure. The proposed fix is to condition each
edge decision on generated local topology, without using the target graph at test
time.

---

## 1. Task Contract

| Object | Definition |
|---|---|
| Node | An entity with a frozen feature vector `x_u` |
| Unseen node | A node held out from training supervision |
| Query | A pair `(u, v)` or a set of candidate pairs |
| Label | Binary edge existence label |
| Context | Generated local topology over queried nodes and retrieved neighbors |
| Output | Pair probability `p_uv`, then an assembled graph after thresholding/ranking |
| Evaluation | Edge-level metrics plus graph-level assembled-output metrics |

The target graph is never an inference input. Candidate retrieval, local topology
generation, and edge classification must use frozen node features and training
supervision only.

---

## 2. Method

The model has three stages.

```text
inputs:  queried pair (u, v), frozen features X
step 1:  retrieve candidate neighbors C(u), C(v) from feature space
step 2:  generate local topology T_uv over {u, v} union C(u) union C(v)
step 3:  predict p_uv = f(x_u, x_v, X_T, T_uv)
output:  pairwise probability, or assembled graph over many queried pairs
```

### 2.1 Candidate Retrieval

Retrieve a small candidate-neighbor set for each queried node using only frozen
features. Approximate nearest-neighbor search is the default implementation
pattern, but the method only requires a graph-free retrieval mechanism. Retrieval
is evaluated through ablations because extra nodes alone may explain some gains.

### 2.2 Local Topology Generation

Generate a sparse adjacency scaffold over the queried nodes and retrieved
neighbors. The scaffold is not the final output; it is input-side context for the
edge classifier.

Preferred mechanism:

- A feature-to-adjacency generator that emits pairwise edge probabilities.
- A sparsification step that keeps the local scaffold small and stable.
- A differentiable relaxation so edge loss and realism loss can train the
  topology module.

Heavier graph-generation mechanisms can be used as ablations, but the default
should stay local, sparse, and cheap enough to run per query.

### 2.3 Topology-Conditioned Edge Classifier

Run a graph encoder over `T_uv` and score the queried pair using the endpoint
features, endpoint embeddings, and local structural features. The queried edge's
presence inside `T_uv` must be standardized or masked so the classifier cannot
read the answer from the scaffold.

For a query set, generate context for each pair or for a shared local query graph,
decode all requested pairs, and assemble the predicted graph from the resulting
scores.

---

## 3. Training Objective

The primary supervised target is the queried edge label.

```text
L = L_edge
  + lambda_real * L_real
  + lambda_ssl * L_ssl
  + lambda_recon * L_recon
```

- `L_edge`: binary cross-entropy or ranking loss on queried edge labels.
- `L_real`: distributional realism loss on local graph statistics, such as
  degree, density, clustering, motif counts, or spectral summaries.
- `L_ssl`: label-free consistency or contrastive loss that keeps generated
  topology stable under feature and neighbor perturbations.
- `L_recon`: optional masked-edge reconstruction pretraining on training
  subgraphs.

The load-bearing point is that edge loss alone under-constrains unqueried edges
inside the scaffold. Auxiliary graph-statistic and self-supervised losses make
the generated topology identifiable enough to be useful context.

---

## 4. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Independent pairwise scorer using only `(x_u, x_v)` | Tests the topology-blind formulation |
| B1 | Retrieval-augmented scorer with neighbor features but no adjacency | Separates retrieval value from topology value |
| B2 | Post-hoc denoising of independently scored outputs | Tests whether cleanup after scoring is enough |
| B3 | Independent scorer with graph-statistic auxiliary loss | Tests whether loss shaping replaces topology context |
| B4 | Latent-graph learner without queried-edge conditioning | Tests generation without decision conditioning |
| Ours | Topology-conditioned edge classifier | Full method |
| Oracle | Classifier conditioned on observed target topology | Upper bound; violates the inductive protocol |

All baselines must use the same frozen features, train/validation/test split, and
query sets unless the baseline definition makes that impossible.

---

## 5. Evaluation

### 5.1 Edge-Level Metrics

- AUROC and AUPR.
- Accuracy, F1, or balanced accuracy at fixed operating points.
- Local ranking metrics per source node or query group.
- Calibration metrics such as expected calibration error and Brier score.
- Robustness across node-disjoint inductive splits.

Report class imbalance and threshold choice explicitly. Do not headline a graph
metric improvement if edge quality collapses.

### 5.2 Graph-Level Assembled-Output Metrics

After scoring a query set, assemble a predicted graph by thresholding or ranking.
Evaluate whether the assembled output matches the reference graph structure.

Metric families:

- Density and relative density.
- Degree-distribution distance.
- Clustering-coefficient distance.
- Spectral distance.
- Motif or orbit statistic distance.
- Component, path-length, and connectivity summaries.
- Learned graph-feature distance from an untrained or held-out graph encoder.

Use density-matched operating points and threshold sweeps to separate topology
quality from trivial threshold effects.

### 5.3 Downstream Probes

If the main result is strong, evaluate whether the assembled graph improves
generic graph ML probes such as community detection, node-label propagation,
clustering quality, or graph-based retrieval.

---

## 6. Integrity Gates

1. Use node-disjoint train/validation/test splits for the primary claim.
2. Ensure candidate retrieval never uses target test edges.
3. Ensure topology generation at test time never reads the target graph.
4. Mask or standardize the queried edge inside the generated scaffold.
5. Report edge metrics and assembled-output graph metrics together.
6. Include simple control baselines for feature shortcuts, degree shortcuts, and
   random topology.

These gates are part of the contribution because the method is only meaningful
under a strict inductive protocol.

---

## 7. Ablations

1. No topology context: collapse to B0 or B1.
2. Retrieved-neighbor features only: remove generated adjacency.
3. Randomized topology with the same scaffold size.
4. Feature-nearest-neighbor topology without learned generation.
5. Scaffold size and retrieval mechanism sweep.
6. Remove `L_real`.
7. Remove `L_ssl`.
8. Remove queried-edge masking.
9. Generation-only variant with no queried-edge supervision.

Each ablation should answer one mechanism question: whether the gain comes from
extra features, local topology, topology realism, or decision conditioning.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Topology generation appears to be the final task | Keep the primary target as queried-edge binary prediction |
| Retrieval explains all gains | Include B1 and retrieval-size sweeps |
| Graph metrics improve only through thresholding | Use density-matched evaluation and threshold sweeps |
| Generated topology hurts edge metrics | Report edge and graph metrics jointly; do not overclaim |
| Scaffold is under-constrained by pair labels | Use graph-statistic and self-supervised auxiliary losses |
| Method is too broad | Keep the task fixed: frozen features, unseen nodes, queried pairs, binary labels |

---

## 9. Deliverables

- A clear problem formulation for topology-conditioned inductive edge prediction.
- A local-scaffold edge classifier trained from frozen node features.
- A baseline suite covering independent scoring, retrieval, post-hoc denoising,
  auxiliary losses, latent topology, and oracle topology.
- An evaluation suite with pairwise edge metrics and graph-level assembled-output
  metrics.
- An ablation suite that isolates retrieval, generated topology, and
  topology-conditioned decision making.
