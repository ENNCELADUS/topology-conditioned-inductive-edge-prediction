# Literature Review Plan — Topology-Conditioned Inductive Edge Prediction

**Fixed task name:** **Topology-Conditioned Inductive Edge Prediction**

**Scope of this document.** This is a literature-review plan for a specific ML
task: predict the binary edge label for an input node pair under an inductive
split, while conditioning the prediction on generated or inferred local topology.
The review should not drift into a broad graph-generation survey. Graph generation
is covered only as a mechanism for constructing local topology context.

**Hard framing rule.** The final task is **edge prediction / binary edge
classification**. Any generated local graph is an intermediate scaffold used to
add topology information around the queried nodes.

---

## 0. Problem Setup

We are given nodes with **intrinsic feature vectors** from a frozen encoder. At
training time, graphs over train nodes provide edge labels. At test time, the
model receives previously unseen nodes and queried node pairs `(u, v)`. It must
predict `edge(u, v) in {0, 1}` without access to the target test graph.

The proposed method adds a topology-context module: retrieve candidate neighbors
from features, construct a local topology scaffold around the queried nodes, and
condition the edge classifier on that scaffold. Evaluation must report both
edge-level metrics and the topology of the graph assembled from predicted edges.

---

## 1. Claims the Review Must Support

| Alias | Claim |
|---|---|
| **K1** | Independent feature-only edge scoring is inductive but topology-blind. |
| **K2** | Topology context can be constructed without using the target test graph. |
| **K3** | Generated or inferred local topology can serve as useful context for a queried edge. |
| **K4** | The method should beat independent scoring, topology-aware pairwise losses, and post-hoc denoising under the same inductive protocol. |
| **K5** | Edge metrics and assembled-graph topology must be evaluated together. |

---

## 2. Locked Taxonomy

The review spine is a 2x2 map:

```text
                         structure-blind                 structure-aware
inductive from features  independent edge scoring          OUR CELL:
                         (clean but no topology)           topology-conditioned
                                                           inductive edge prediction

needs target graph       feature + observed-graph hybrids  transductive graph-native
at inference                                                link prediction/completion
                         (not strict inductive)            (structure, but leaks)
```

The empty cell is **not** "graph generation" in general. It is:

> edge prediction for unseen nodes, conditioned on local topology constructed
> without the target graph.

---

## 3. Literature Clusters

### C0. Edge Prediction as Binary Classification

**Job:** Establish the incumbent formulation: construct pair features, score each
candidate edge independently, optimize binary classification loss.

**Supports:** K1.

**Reviewer objection handled:** "Is this just link prediction?" Answer: yes, the
task is still link prediction; the novelty is the topology-conditioned context.

### C1. Inductive Representation and Feature-Only Generalization

**Job:** Anchor the strict inductive setting: test nodes are unseen and the test
graph is unavailable.

**Supports:** K1, K2.

**Reviewer objection handled:** "Why not use a graph neural network over the known
graph?" Answer: that changes the protocol by requiring graph access at inference.

### C2. Structure-Aware but Transductive Link Prediction

**Job:** Cover graph-native link prediction, graph completion, masked graph
autoencoders, and graph-structure learning methods that use an observed graph.

**Supports:** K2, K4.

**Reviewer objection handled:** "Existing graph methods already use topology."
Answer: many do, but they rely on an input graph and therefore do not satisfy the
strict inductive task.

### C3. Candidate-Neighbor Retrieval Without Target-Graph Leakage

**Job:** Support the retrieval step that builds local candidate sets from node
features or other graph-free signals.

**Supports:** K2, K3.

**Reviewer objection handled:** "Local context is unavailable for unseen nodes."
Answer: candidate neighborhoods can be proposed without using target edges.

### C4. Local Topology Construction as Context

**Job:** Cover local graph generation, subgraph construction, latent adjacency
prediction, discrete graph diffusion, and autoregressive local growth only insofar
as they provide a topology scaffold for the queried edge.

**Supports:** K3, K4.

**Reviewer objection handled:** "You are solving graph generation instead of edge
prediction." Answer: generated topology is an intermediate representation consumed
by the edge classifier.

### C5. Topology-Aware Losses and Post-Hoc Denoising

**Job:** Position natural fixes to independent scoring: add graph-level losses or
clean the assembled graph after scoring.

**Supports:** K4.

**Reviewer objection handled:** "Why not just add topology loss or denoise after
prediction?" Answer: those approaches do not condition the edge decision itself on
local topology.

### C6. Evaluation: Edge Metrics Plus Assembled-Graph Metrics

**Job:** Ground the dual evaluation: AUROC/AUPR for edge prediction, plus density,
degree, clustering, spectral, and graph-similarity metrics for the assembled graph.

**Supports:** K5.

**Reviewer objection handled:** "Which metric defines success?" Answer: the method
must preserve edge-level quality and avoid implausible assembled topology.

---

## 4. Search Workflow

1. **Seed each cluster** with 3-6 canonical ML graph papers.
2. **Snowball** backward and forward two hops from each seed.
3. **Venue sweep:** ICLR, NeurIPS, ICML, KDD, LOG, WWW, TMLR.
4. **Include** works that address at least one of:
   - inductive edge prediction,
   - feature-only link prediction,
   - graph-free candidate retrieval,
   - local topology/subgraph construction,
   - topology-aware link prediction losses,
   - graph-generation evaluation metrics used for assembled predictions.
5. **Exclude** works whose contribution is only application-specific and whose
   mechanism does not transfer to the fixed task.
6. **Tag every note** with K1-K5 and one reviewer objection it helps answer.

---

## 5. Terminology Guardrail

Use these terms consistently.

| Do use | Do not let it drift into |
|---|---|
| topology-conditioned inductive edge prediction | generic graph generation |
| queried node pair `(u, v)` | whole-graph generation as the final task |
| binary edge label / edge probability | only topology reconstruction |
| candidate-neighbor set | observed test graph neighborhood |
| local topology scaffold / local topology context | target graph input |
| generated topology as intermediate context | generated graph as final prediction |
| assembled graph metrics as secondary evaluation | graph metrics as the only objective |

Self-check before drafting: every section must answer how it helps predict
`edge(u, v)` for unseen nodes.

---

## 6. Outputs / Done Criteria

The review is complete when it produces:

1. **2x2 positioning figure** with the method in the structure-aware + inductive
   edge-prediction cell.
2. **Method comparison table:**
   `method x {inductive?, uses target graph at inference?, topology-conditioned?,
   final output is edge label?, graph metrics evaluated?}`.
3. **One gap paragraph** explaining why independent scoring lacks topology, while
   transductive graph methods use topology but violate the inductive protocol.
4. **One mechanism paragraph** explaining local graph generation as topology
   context for the edge classifier.
5. **Cluster-to-claim map** showing that every cited cluster supports K1-K5.

**Definition of done:** the draft never describes graph generation as the final
task; every contribution is tied back to binary edge prediction for queried pairs.
