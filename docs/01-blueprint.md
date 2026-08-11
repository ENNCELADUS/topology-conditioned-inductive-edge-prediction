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
| Primary task | Fully inductive edge prediction between two unseen nodes |
| Input | Only frozen intrinsic endpoint features `(x_u, x_v)` |
| Intermediate object | Inferred query-local topology representation |
| Output | Symmetric $\widehat A_{uv}$; $\widehat G_\tau$ assembled only for evaluation |
| Main comparison | Independent pair scoring vs. topology-conditioned scoring |
| Success criterion | Strong edge metrics without implausible assembled graph topology |

---

## 2. Thesis

Fully inductive edge prediction should be topology-conditioned binary classification,
not independent pair scoring. Training topology may teach transferable structural
regularities, but inference receives only two unseen nodes' intrinsic features.
The representation method remains open; retrieval is a separate support condition.

What the reader should believe: endpoint-only pair accuracy does not guarantee a
plausible assembled graph; topology transfer must improve the queried-edge
decision and the assembled graph together.

---

## 3. Research Problem

**Training graph.** $G_{\mathrm{train}}=(V_{\mathrm{train}},E_{\mathrm{train}})$ and
$V_{\mathrm{train}}\cap V_{\mathrm{test}}=\varnothing$; test endpoints have no observed edges.

**Intrinsic features.** $x_i=E_\eta(a_i)$, where $a_i$ contains the node attributes;
the encoder is frozen before edge prediction.

**Query.** For $u,v\in V_{\mathrm{test}}$, the symmetric predictor receives only $(x_u,x_v)$:
$\widehat A_{uv}=P_\theta(Y_{uv}=1\mid x_u,x_v)=P_\theta(Y_{uv}=1\mid x_v,x_u)$.

**Assembly.** Over $\mathcal Q_{\mathrm{test}}$, define $\widehat E_\tau$ by thresholding
$\widehat A_{uv}$ and $\widehat G_\tau=(V_{\mathrm{test}},\widehat E_\tau)$, then compare
it with hidden $G_{\mathrm{test}}=(V_{\mathrm{test}},E_{\mathrm{test}})$.

**H1 — pair-to-topology gap.** Independent $\widehat A_{uv}$ values may be accurate while
their assembly has implausible degree, clustering, density, or spectrum.

**H2 — topology transfer.** Structural regularities of $E_{\mathrm{train}}$ and their
dependence on $x$ are domain properties that may transfer to unseen nodes.

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
| RQ1 | Does the pair-to-topology gap hold in fully inductive edge prediction? | Joint pair/assembled-graph diagnosis |
| RQ2 | Can $G_{\mathrm{train}}$ topology transfer through endpoint features alone? | Zero-observed-edge topology conditioning |
| RQ3 | Which topology representation helps $\operatorname{edge}(u,v)$? | Matched latent, deterministic, and generative comparison |
| RQ4 | Does extra retrieval support explain an explicit-scaffold gain? | Separately labeled retrieval-grounded arm |
| RQ5 | Does the selected method improve both prediction levels? | Joint edge/topology result |

The load-bearing contributions are RQ2 and RQ3. RQ1 motivates the problem; RQ4
separates topology learning from gains caused by extra inference support.

---

## 5. Method Summary

1. **Input.** Encode only the two frozen node representations.
2. **Context.** Learn from $G_{\mathrm{train}}$; infer context from $(x_u,x_v)$ at test time.
3. **Decision.** Produce a symmetric probability for the queried edge.
4. **Assembly.** Construct $\widehat G_\tau$ only after scoring $\mathcal Q_{\mathrm{test}}$.

Retrieval-grounded scaffolds are a separate arm whose support pool is not task input.

Minimal contract:

```text
inputs:  pair (u, v), frozen endpoint features (x_u, x_v)
context: T_uv = g(x_u, x_v; theta_g), learned with G_train supervision
output:  p_uv = p_vu = P(Y_uv = 1 | x_u, x_v, T_uv)
```

Training uses queried-edge supervision as the primary loss, with optional
auxiliary losses for local graph realism, topology consistency, and masked-edge
reconstruction on training subgraphs.

---

## 6. Baseline Plan

| ID | Baseline | Interpretation |
|---|---|---|
| B0 | Endpoint-only direct scorer | Topology-blind baseline |
| B1 | Training-time topology transfer | Graph2Feat/LLP/CAZI-MBN-style student |
| B2 | Deterministic structural latent | Tests topology transfer without generation |
| B3 | Conditional latent topology generator | Tests stochastic topology context |
| B4 | Retrieval-grounded explicit scaffold | Extra-support arm, including EgoStitch |
| Controls | Loss shaping, denoising, retrieval-only | Tests simpler explanations |
| Oracle | Classifier using observed target topology | Upper bound that violates protocol |

All fair baselines should share the same frozen features, splits, query sets, and
evaluation protocol.

---

## 7. Evaluation Plan

**Edge-level metrics.** Report AUROC, AUPRC, F1, MCC, and calibration under
node-disjoint splits, uncertain-negative disclosure, and hard-negative controls.

**Graph-level assembled-output metrics.**

- BFS-macro GS and RD.
- Degree-MMD ratio.
- Clustering-MMD ratio.
- Spectral-MMD ratio.

Report these five headline values together; name global simple-edge GS/RD separately
and never infer topology success from pair metrics alone.
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
2. **Related Work.** Separate transductive/observed-topology, endpoint-only,
   training-time topology transfer, and synthetic attachment methods.
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
| Topology generation appears to be the main task | High | Anchor every section to `edge(u,v)` |
| Generated topology does not improve edge metrics | High | Report edge and graph metrics jointly; use ablations to diagnose |
| Graph metrics improve only by threshold changes | High | Use density-matched operating points and threshold sweeps |
| Optional candidate retrieval leaks target information or is mistaken for task input | High | Keep retrieval in a separately labeled arm, freeze and disclose its support universe, and audit every inference input |
| Scaffold contains the queried-edge answer | High | Mask or standardize the queried edge inside the scaffold |
| Negative labels or graph truth are biased | High | Disclose uncertain negatives and observation limits |

---

## 10. Locked Decisions

1. Task: fully inductive, node-disjoint, zero-observed-edge prediction.
2. Input/output: exactly $(x_u,x_v)$ to symmetric $\widehat A_{uv}$.
3. Structure: $G_{\mathrm{train}}$ may supervise intermediate topology context only.
4. Assembly: score $\mathcal Q_{\mathrm{test}}$, then form $\widehat G_\tau$.
5. Evaluation: pairwise metrics plus GS, RD, and three MMD ratios together.
6. Method remains open; grounding/retrieval is optional support, never task input.
