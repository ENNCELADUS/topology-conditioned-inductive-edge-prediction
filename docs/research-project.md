# Research Project: Topology-Conditioned Inductive Edge Prediction

## Problem Definition

PPI prediction is formulated as fully inductive attributed link prediction. Let the experimentally observed training network be

$$
G_{\mathrm{train}}=(V_{\mathrm{train}},E_{\mathrm{train}}),
$$

with a node-disjoint test set

$$
V_{\mathrm{train}}\cap V_{\mathrm{test}}=\varnothing.
$$

For each protein $i$, a intrinsic encoder produces

$$
x_i=E_{\eta}(S_i,M_i)\in\mathbb R^d,
$$

where $S_i$ is its sequence and $M_i$ is optional monomer structure. For a queried pair $u,v\in V_{\mathrm{test}}$, the model receives only $x_u$ and $x_v$. A symmetric predictor outputs

$$
\widehat A_{uv}
=P_{\theta}(Y_{uv}=1\mid x_u,x_v)
=P_{\theta}(Y_{uv}=1\mid x_v,x_u).
$$

No observed edge incident to $u$ or $v$ is available at inference. After scoring all queried pairs $\mathcal Q_{\mathrm{test}}$, their predictions are assembled into a graph:

$$
\widehat E_{\tau}=\bigl\{\{u,v\}\in\mathcal Q_{\mathrm{test}}:\widehat A_{uv}\geq\tau\bigr\},
\qquad
\widehat G_{\tau}=(V_{\mathrm{test}},\widehat E_{\tau}).
$$

The predicted graph $\widehat G_{\tau}$ is compared with the hidden reference graph $G_{\mathrm{test}}=(V_{\mathrm{test}},E_{\mathrm{test}})$ using both pairwise edge metrics and assembled-graph topology metrics.

## Hypotheses

The problem definition above leaves the predictor with $x_u,x_v$ alone, yet grades it on a graph
$\widehat G_{\tau}$. The representation method remains open; grounding/retrieval is an optional arm,
not task input or a selected method.

**H1 — pair-to-topology gap.** $\widehat A$ decomposed into independent pair decisions cannot control
the joint statistics of $\widehat E_{\tau}$. Pair scores that are individually accurate can therefore
assemble into a network whose degree, clustering, and spectral profile are implausible under
$G_{\mathrm{test}}$, so edge-level accuracy alone does not bound topology-level error.

**H2 — topology transfer.** $G_{\mathrm{train}}$ and $G_{\mathrm{test}}$ are node-disjoint but are
draws from the same interaction domain, so the structural regularities of $E_{\mathrm{train}}$ —
degree profile, neighborhood density, motif and community organization, and their dependence on
$x$ — are properties of the domain rather than of $V_{\mathrm{train}}$. They are consequently
estimable from $G_{\mathrm{train}}$ at training time and remain valid over $V_{\mathrm{test}}$ at
inference, where no edge is observed.

## Determine the Method Ceiling with Oracle Topology

![Held-out method-ceiling headline results comparing the topology oracle with the pairwise baseline](results/method-ceiling-headline.svg)

## Curated Related Work

### Taxonomy

The first split is the **prediction protocol**, not whether a model architecture is described as inductive:

```text
link prediction
├── transductive: seen–seen; both endpoints belong to the training graph
└── inductive: at least one endpoint is unseen during training
    ├── semi-inductive: unseen–seen
    └── fully inductive: unseen–unseen
        ├── an inference graph or observed neighbors are available
        └── no observed inference topology; context inferred from endpoints or distilled (this project)
```

This project is therefore **fully-inductive, zero-observed-edge, attributed link prediction with inferred query-local topology**. PPI adds important domain constraints—undirected physical interactions, uncertain negatives, ascertainment-biased graph truth, and the need to evaluate the assembled network.

### Transductive Link Prediction

| Family | Representative work | Relevance and boundary |
|---|---|---|
| Autoencoding and message passing | [VGAE](https://arxiv.org/abs/1611.07308); [GraphSAGE](https://arxiv.org/abs/1706.02216) | Learn node embeddings from an observed graph and decode links. GraphSAGE has inductive parameters, but a seen–seen edge split is still a transductive experiment. |
| Query-subgraph models | [SEAL](https://arxiv.org/abs/1802.09691); [Distance Encoding](https://arxiv.org/abs/2009.00142); [Labeling Trick](https://arxiv.org/abs/2010.16103) | Establish that target-conditioned enclosing structure is more expressive than independently computed endpoint embeddings, but require the query nodes to exist in an observed graph. |
| Scalable neighborhood-overlap models | [Neo-GNN](https://arxiv.org/abs/2206.04216); [BUDDY](https://arxiv.org/abs/2209.15486); [NCN/NCNC](https://arxiv.org/abs/2302.00890); [LPFormer](https://arxiv.org/abs/2310.11009) | Encode common-neighbor or subgraph signals efficiently. They provide the strongest evidence that topology matters for link prediction, while their topology is observed rather than inferred for isolated nodes. |
| Evaluation critiques | [HeaRT](https://arxiv.org/abs/2306.10453); [Implicit Degree Bias](https://arxiv.org/abs/2405.14985) | Show that easy negatives and degree-biased sampling can inflate link-prediction results; they motivate hard, topology-stratified evaluation rather than a model component. |
| PPI-specific transductive models and evaluation | [DNE](https://doi.org/10.1126/sciadv.adq4324); [SCMPPI](https://doi.org/10.1016/j.neucom.2026.133428) | DNE and SCMPPI consume a known PPI network. All are important PPI references, but none solves two-unseen-protein prediction without incident edges. |

### Inductive Link Prediction

| Category | High-level idea | What the model does at inference | Topology source | Representative work | Relevance and boundary |
|---|---|---|---|---|---|
| **Observed topology available at inference** | Transfer learned graph operators to unseen nodes or a new graph. | Encodes observed neighbors, paths, or query subgraphs and then scores the candidate link. | An inference graph containing incident edges, paths, few-shot links, or an observed disjoint graph. | [GraphSAGE](https://arxiv.org/abs/1706.02216); [DEAL](https://doi.org/10.24963/ijcai.2020/168); [New Node Prediction](https://arxiv.org/abs/2401.05468); [GraIL](https://proceedings.mlr.press/v119/teru20a.html); [NBFNet](https://arxiv.org/abs/2106.06935); [ULTRA](https://arxiv.org/abs/2310.04562); [GEN](https://papers.nips.cc/paper/2020/hash/0663a4ddceacb40b095eda264a85f15c-Abstract.html); [DEKG-ILP](https://arxiv.org/abs/2209.01397) | These protocols are inductive because the endpoints or inference graph were unseen during training, but they are not equivalent to two unseen proteins with no observed incident edges. |
| **Endpoint features only** | Generalize through transferable node attributes rather than graph neighborhoods. | Encodes the two endpoint features and directly scores the pair. | None at inference. | [UPNA](https://arxiv.org/abs/2307.08877); [PLM-interact](https://doi.org/10.1038/s41467-025-64512-w); [Learning the language of PPIs](https://doi.org/10.1038/s41467-025-67971-3) | UPNA directly studies mutually unseen endpoints. Feature-based PPI models naturally accept unseen proteins, but generally score pairs independently and do not model the topology of the assembled prediction graph. |
| **Training-time topology transfer** | Compress structural knowledge from the training graph into a feature-based predictor. | Runs a feature-only student or attribute-to-embedding mapping; no explicit graph is consumed. | The training graph supplies supervision or teacher signals, which are absorbed into learned parameters. | [Graph2Gauss](https://arxiv.org/abs/1707.03815); [Graph2Feat](https://doi.org/10.1145/3543873.3587596); [Linkless Link Prediction](https://proceedings.mlr.press/v202/guo23f/guo23f.pdf); [CAZI-MBN](https://arxiv.org/abs/2603.06618); [GLNN](https://arxiv.org/abs/2110.08727); [Cold Brew](https://arxiv.org/abs/2111.04840); [Topology Distillation](https://arxiv.org/abs/2106.08700) | Graph2Feat, LLP, and CAZI-MBN are the closest link-prediction precedents. GLNN, Cold Brew, and Topology Distillation are adjacent representation-distillation methods rather than protocol-matched link predictors. |
| **Synthetic or learned topology connected to an observed graph** | Give a cold-start node usable graph context by adding or predicting links to known nodes. | Creates synthetic edges or predicts attachments from the newcomer to an existing graph, then applies graph propagation. | Fixed augmentation or learned attachments, combined with the observed topology on the known side. | [NodeDup](https://arxiv.org/abs/2402.09711); [LEAP](https://arxiv.org/abs/2503.03331) | LEAP is the closest mechanism comparator, but its reported inductive task is unseen–seen and still depends on an observed graph on the seen side. |

### Adjacent Topology-Learning Machinery

[LDS](https://proceedings.mlr.press/v97/franceschi19a.html), [IDGL](https://arxiv.org/abs/2006.13009), [NodeFormer](https://arxiv.org/abs/2206.08320), and [DGM](https://arxiv.org/abs/2209.14734) learn task-dependent graph structure; [NRI](https://proceedings.mlr.press/v80/kipf18a.html) infers latent interaction graphs for downstream dynamics; [FLEX](https://arxiv.org/abs/2507.11710) uses generated subgraphs for OOD link prediction. These are important generator precedents, but they should not be presented as protocol-matched solutions to two isolated unseen endpoints. The current encoder machinery is grounded separately in [GRIT](https://proceedings.mlr.press/v202/ma23c.html) and [Set Transformer/PMA](https://proceedings.mlr.press/v97/lee19d.html).

This taxonomy yields the narrow literature gap: transductive methods consume topology; most zero-edge inductive methods remove topology at inference; LEAP reconstructs topology only from a newcomer into an existing graph. The remaining question is whether query-local topology can be generated for **two isolated unseen nodes** and improve both their binary link decision and the topology of the assembled prediction graph.
