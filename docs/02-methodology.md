# Methodology Plan: Topology-Conditioned Inductive Edge Prediction

**Status:** concise method-selection plan; companion to `01-blueprint.md`.

**Problem.** Given unseen nodes with frozen feature vectors and a queried node
pair `(u, v)`, predict a binary edge label. For a query set, predict many pair
labels and assemble the predicted graph. The method must be judged both by
pairwise edge metrics and by graph-level metrics on the assembled output.

**Core claim.** Independent pair scoring is topology-blind: it can optimize
`P(edge | x_u, x_v)` while producing an assembled graph with implausible degree,
density, clustering, or spectral structure. The proposed fix is to condition each
edge decision on a generated or inferred topology representation derived from the
queried endpoints, without using the target graph at test time.

---

## 1. Task Contract

| Object | Definition |
|---|---|
| Node | An entity with a frozen feature vector `x_u` |
| Unseen node | A node held out from training supervision |
| Query | A pair `(u, v)` or a set of candidate pairs |
| Label | Binary edge existence label |
| Context | Generated or inferred topology representation derived from `(x_u, x_v)` |
| Output | Pair probability `p_uv`, then an assembled graph after thresholding/ranking |
| Evaluation | Edge-level metrics plus graph-level assembled-output metrics |

The target graph is never an inference input. The primary contract receives only
the two queried endpoints' intrinsic features. Training-graph supervision may shape
learned parameters, but candidate retrieval, grounding pools, and external node
identities are not task inputs and are not assumed by any endpoint-only candidate.

---

## 2. Method

The minimal model has two stages.

```text
inputs:  queried pair (u, v), frozen endpoint features (x_u, x_v)
step 1:  infer topology representation T_uv = g(x_u, x_v; theta_g)
step 2:  predict p_uv = f(x_u, x_v, T_uv; theta_f)
output:  pairwise probability, or assembled graph over many queried pairs
```

### 2.1 Candidate Topology-Representation Families

Method selection compares three distinct families rather than presupposing a
retrieval pipeline:

- **Endpoint-only latent relational topology:** a conditional generator maps
  `(x_u, x_v)` to identity-free neighborhood/topology variables, such as anonymous
  slots, soft adjacency, structural tokens, or a distribution over them.
- **Endpoint-only deterministic topology transfer:** a deterministic encoder maps
  the endpoints to a topology-aware latent, discrete code, or structural summary.
- **Retrieval-grounded explicit scaffold:** candidates or prototypes from a declared
  frozen support universe are retrieved and connected into a local graph. This is
  an optional arm with a different inference-support contract, not the default or
  selected method.

All families must expose an intermediate representation that can be removed,
randomized, or replaced in a matched control. If an explicit scaffold is used, it
must remain local and the queried edge must be masked.

### 2.2 Topology-Conditioned Edge Classifier

Encode `T_uv` with a representation-appropriate encoder and score the queried pair
using the endpoint features and topology context. The queried edge's presence
inside any explicit or implicit adjacency target must be standardized or masked so
the classifier cannot read the answer from the representation.

For a query set, infer context and score each pair through the same two-endpoint
API, then assemble the predicted graph from the resulting scores. Graph-scale
context is not an inference input; assembly is an evaluation operation.

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

For explicit scaffolds, auxiliary losses constrain unqueried edges. Latent or
deterministic candidates instead supervise their declared structural representation;
they must not pretend that unobserved scaffold edges exist.

---

## 4. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Independent pairwise scorer using only `(x_u, x_v)` | Tests the topology-blind formulation |
| B1 | Retrieval-augmented scorer with neighbor features but no adjacency | Separates retrieval value from topology value |
| B2 | Post-hoc denoising of independently scored outputs | Tests whether cleanup after scoring is enough |
| B3 | Independent scorer with graph-statistic auxiliary loss | Tests whether loss shaping replaces topology context |
| B4 | Latent-graph learner without queried-edge conditioning | Tests generation without decision conditioning |
| Selected family | Topology-conditioned edge classifier under the endpoint-only contract | Headline method after matched selection |
| Retrieval-grounded arm | Explicit scaffold from a declared frozen support universe | Tests whether external prototype support is necessary |
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
2. Retrieval arms disclose/freeze support and never use target edges or labels.
3. Ensure topology generation at test time never reads the target graph.
4. Mask or standardize the queried edge inside the generated scaffold.
5. Report edge metrics and assembled-output graph metrics together.
6. Include simple control baselines for feature shortcuts, degree shortcuts, and
   random topology.

These gates are part of the contribution because the method is only meaningful
under a strict inductive protocol.

---

## 7. Ablations

1. No topology context: collapse to B0.
2. Identity-free latent vs deterministic structural vs explicit-scaffold context.
3. Deterministic vs stochastic topology representation.
4. Randomized topology with matched representation capacity.
5. Remove queried-pair conditioning from the topology module.
6. Remove `L_real`.
7. Remove `L_ssl`.
8. Remove queried-edge masking.
9. Generation-only variant with no queried-edge supervision.
10. Retrieval-grounded arm only: retrieval-only, shuffled-candidate, random-pool,
    and pool/scaffold-size controls.

Each ablation should answer one mechanism question: whether the gain comes from
extra features, local topology, topology realism, or decision conditioning.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Topology generation appears to be the final task | Keep the primary target as queried-edge binary prediction |
| External retrieval explains an explicit-scaffold gain | Treat it as a separate arm and include retrieval-only and shuffled/random-pool controls |
| Graph metrics improve only through thresholding | Use density-matched evaluation and threshold sweeps |
| Generated topology hurts edge metrics | Report edge and graph metrics jointly; do not overclaim |
| Scaffold is under-constrained by pair labels | Use graph-statistic and self-supervised auxiliary losses |
| Method is too broad | Keep the task fixed: frozen features, unseen nodes, queried pairs, binary labels |

---

## 9. Deliverables

- A clear problem formulation for topology-conditioned inductive edge prediction.
- A matched selection among endpoint-only topology-representation families before
  a headline method is fixed.
- A baseline suite covering independent scoring, retrieval, post-hoc denoising,
  auxiliary losses, latent topology, and oracle topology.
- An evaluation suite with pairwise edge metrics and graph-level assembled-output
  metrics.
- An ablation suite that isolates topology transfer, stochastic generation,
  optional retrieval support, and topology-conditioned decision making.
