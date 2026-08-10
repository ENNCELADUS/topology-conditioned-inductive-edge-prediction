# Methodology Plan: Topology-Conditioned Inductive Edge Prediction

**Status:** concise method-selection plan; companion to `01-blueprint.md`.

**Problem.** Given unseen nodes with frozen feature vectors and a queried node
pair `(u, v)`, predict a binary edge label. For a query set, predict many pair
labels and assemble the predicted graph. The method must be judged both by
pairwise edge metrics and by graph-level metrics on the assembled output.

**Core hypothesis.** Independent pair scoring is topology-blind: it can optimize
`P(edge | x_u, x_v)` while producing an assembled graph with implausible degree,
density, clustering, or spectral structure. A candidate fix is to condition each
edge decision on a generated or inferred topology representation derived from the
queried endpoints, without using the target graph at test time.

---

## 1. Task Contract

| Object | Definition |
|---|---|
| Node | An entity with a frozen feature vector `x_u` |
| Unseen node | A node held out from training supervision |
| Query | One pair `(u, v)` in primary pair mode; many pairs are only the assembly evaluation loop |
| Label | Binary edge existence label |
| Context | Generated or inferred topology representation derived from `(x_u, x_v)` |
| Output | Pair probability `p_uv`; graph assembly is an evaluation operation |
| Evaluation | Edge-level metrics plus graph-level assembled-output metrics |

The target graph is never an inference input. The primary contract receives only
the two queried endpoints' intrinsic features. Candidate retrieval, grounding pools,
and external identities are not task inputs. Set/subgraph inference is a separate
future contract and must not be mixed into primary pair-mode results.

---

## 2. Method

Each context-bearing candidate has two stages.

```text
inputs:  queried pair (u, v), frozen endpoint features (x_u, x_v)
step 1:  infer topology representation T_uv = g(x_u, x_v; theta_g)
step 2:  predict p_uv = f(x_u, x_v, T_uv; theta_f)
output:  pairwise probability; assembly occurs only in evaluation
```

The structured-objective control keeps a direct pair scorer and changes only training.

### 2.1 Candidate Topology-Representation Families

Method selection follows a staged comparison rather than presupposing a generator
or retrieval pipeline:

- **Topology distillation (first baseline):** a graph-aware teacher encodes
  training neighborhoods; a deterministic endpoint-only student aligns latent/logits.
- **Conditional latent-topology generation (parallel):** model
  `p(T_uv | x_u, x_v)` without generating concrete neighbor identities.
- **Topology-aware structured objective (complement):** constrain groups of edge
  predictions using training-graph statistics; this is not representation transfer.
- **Retrieval-grounded explicit scaffold:** candidates or prototypes from a declared
  frozen support universe are retrieved and connected into a local graph. This is
  an optional arm with a different inference-support contract, not the default or
  selected method.

All context-bearing families must expose an intermediate representation that can be
removed, randomized, or replaced. The structured-objective family instead requires
a matched loss-only control. Any explicit scaffold masks the queried edge.

### 2.2 Topology-Conditioned Edge Classifier

Training neighborhoods may be graph-encoded to define a privileged target `T_uv`;
at inference `T_uv` must be distilled or generated from the endpoints alone. A graph
encoder is an inference component only when the selected representation is a graph.
Any queried edge inside a structural target must be masked.

For a query set, infer context and score each pair through the same two-endpoint
API, then assemble the predicted graph from the resulting scores. Graph-scale
context is not an inference input; assembly is an evaluation operation.

---

## 3. Training Objective

The primary supervised target is the queried edge label.

```text
L = L_edge
  + lambda_kd * L_kd
  + lambda_gen * L_gen
  + lambda_struct * L_struct
```

- `L_edge`: binary cross-entropy or ranking loss on queried edge labels.
- `L_kd`: teacher-student latent, relational, or logit alignment.
- `L_gen`: likelihood, variational, or distributional loss for latent topology.
- `L_struct`: graph-statistic loss over jointly sampled training query groups.

Each candidate activates only its declared terms. Distillation is the first runnable
baseline; latent generation is the parallel research probe. Neither may imply that
unobserved test scaffold edges or neighbor identities are available.

---

## 4. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Independent pairwise scorer using only `(x_u, x_v)` | Tests the topology-blind formulation |
| B1 | Graph2Feat/LLP/CAZI-MBN-style topology distillation | Establishes topology-transfer recovery toward Oracle |
| B2 | Deterministic structural latent | Tests topology transfer without generation |
| B3 | Conditional latent-topology generator | Tests explicit missing-topology imputation |
| B4 | Retrieval-grounded explicit scaffold | Historical extra-support arm, including EgoStitch |
| Selected family | Topology-conditioned edge classifier under the endpoint-only contract | Headline method after matched selection |
| Controls | Structured loss, denoising, and retrieval-only | Tests simpler explanations and extra support |
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

Report five headline values together: BFS-macro GS and RD, plus degree,
clustering, and spectral MMD ratios. Name global simple-edge GS/RD separately.

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
2. Distillation-only vs latent-generation-only vs combined representation transfer.
3. Deterministic expectation vs stochastic latent sampling.
4. Randomized topology latent with matched representation capacity.
5. Remove teacher topology or shuffle teacher targets.
6. Remove `L_kd`.
7. Remove `L_gen`.
8. Remove `L_struct`.
9. Remove queried-edge masking from training targets.
10. Retrieval-grounded arm only: retrieval-only, shuffled-candidate, random-pool,
    and pool/scaffold-size controls.

Each ablation should answer one mechanism question: whether the gain comes from
extra features, local topology, topology realism, or decision conditioning.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Topology modeling appears to be the final task | Keep the primary target as queried-edge binary prediction |
| External retrieval explains an explicit-scaffold gain | Treat it as a separate arm and include retrieval-only and shuffled/random-pool controls |
| Graph metrics improve only through thresholding | Use density-matched evaluation and threshold sweeps |
| Generated topology hurts edge metrics | Report edge and graph metrics jointly; do not overclaim |
| Scaffold is under-constrained by pair labels | Use graph-statistic and self-supervised auxiliary losses |
| Method is too broad | Keep the task fixed: frozen features, unseen nodes, queried pairs, binary labels |

---

## 9. Deliverables

- A clear problem formulation for topology-conditioned inductive edge prediction.
- A staged comparison: distillation first, latent generation in parallel, then
  matched selection before a headline method is fixed.
- A baseline suite covering independent scoring, retrieval, post-hoc denoising,
  auxiliary losses, latent topology, and oracle topology.
- An evaluation suite with pairwise edge metrics and graph-level assembled-output
  metrics.
- An ablation suite isolating distillation, latent generation, structured loss,
  optional retrieval support, and topology-conditioned decisions.
