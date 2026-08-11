# Literature Review Findings — Topology-Conditioned Inductive Edge Prediction

**Scope.** Findings, not a plan: classifies every work cited in
`docs/research-project.md` (Curated Related Work) and `docs/01-blueprint.md`
(Baseline Plan) against this task's five claims, hardened by a 2026-08-11
forward-citation, sequence-based-PPI, and adjacent-community sweep folded
into §3–4 below. The task is binary edge prediction from `(x_u,x_v)` alone
under a node-disjoint, zero-observed-edge inductive split, graded jointly on
edge metrics and assembled-graph topology. See the [2026-08-10
litreview](2026-08-10-feature-conditioned-latent-topology-litreview.md) for
fuller per-cluster narrative.

---

## 1. Claims

| Alias | Claim |
|---|---|
| **K1** | Independent feature-only edge scoring is inductive but topology-blind. |
| **K2** | Topology context can be constructed without using the target test graph. |
| **K3** | Generated or inferred local topology can serve as useful context for a queried edge. |
| **K4** | The method should beat independent scoring, topology-aware pairwise losses, and post-hoc denoising under the same inductive protocol. |
| **K5** | Edge metrics and assembled-graph topology must be evaluated together. |

## 2. Taxonomy

```text
                         structure-blind                 structure-aware
inductive from features  independent edge scoring          OUR CELL:
                         (clean but no topology)           topology-conditioned
                                                           inductive edge prediction

needs target graph       feature + observed-graph hybrids  transductive graph-native
at inference                                                link prediction/completion
                         (not strict inductive)            (structure, but leaks)
```

The empty cell is edge prediction for unseen nodes, conditioned on local
topology constructed without the target graph — not graph generation in
general.

## 3. Classified Related Work

### 3.1 Transductive link prediction — bounds K1/K2, motivates K5

| Family | Work | Boundary |
|---|---|---|
| Autoencoding / message passing | [VGAE](https://arxiv.org/abs/1611.07308), [GraphSAGE](https://arxiv.org/abs/1706.02216) | Decode links from an observed graph; a seen–seen split stays transductive even with inductive parameters (K1). |
| Query-subgraph | [SEAL](https://arxiv.org/abs/1802.09691), [Distance Encoding](https://arxiv.org/abs/2009.00142), [Labeling Trick](https://arxiv.org/abs/2010.16103) | Target-conditioned structure beats independent endpoint embeddings, but needs query nodes already in an observed graph. |
| Neighborhood-overlap | [Neo-GNN](https://arxiv.org/abs/2206.04216), [BUDDY](https://arxiv.org/abs/2209.15486), [NCN/NCNC](https://arxiv.org/abs/2302.00890), [LPFormer](https://arxiv.org/abs/2310.11009) | Strongest evidence topology matters for link prediction; topology is observed, not inferred, for isolated nodes. |
| Evaluation critiques | [HeaRT](https://arxiv.org/abs/2306.10453), [Implicit Degree Bias](https://arxiv.org/abs/2405.14985) | Motivate hard, topology-stratified evaluation (K5); not a model contribution. |

### 3.2 Inductive link prediction

**Observed topology available at inference — violates the zero-observed-edge
constraint (K2 boundary).** [GraphSAGE](https://arxiv.org/abs/1706.02216),
[DEAL](https://doi.org/10.24963/ijcai.2020/168), [New Node
Prediction](https://arxiv.org/abs/2401.05468),
[GraIL](https://proceedings.mlr.press/v119/teru20a.html),
[NBFNet](https://arxiv.org/abs/2106.06935),
[ULTRA](https://arxiv.org/abs/2310.04562),
[GEN](https://papers.nips.cc/paper/2020/hash/0663a4ddceacb40b095eda264a85f15c-Abstract.html),
[DEKG-ILP](https://arxiv.org/abs/2209.01397) — inductive because endpoints or
the inference graph were unseen in training, but all consume an inference
graph, paths, or few-shot links; none matches two endpoints with no observed
incident edge. Joint Link Prediction via Inference from a Model (DOI
10.1145/3583780.3614941, CIKM 2023) queries joint links from an inductive
VGAE but still conditions on evidence links, and Deep Network Completion
(DOI 10.3390/e23060771, Entropy 2021) recovers unobserved topology only given
labels and distances to the observed part.

**Endpoint features only — direct K1/K3 precedent.**
[UPNA](https://arxiv.org/abs/2307.08877) is the only general-domain work
studying mutually-unseen endpoints directly; it scores pairs independently
and does not model assembled-graph topology, motivating K3. Chatterjee et
al.'s Isolated Nodes extension (OpenReview DRrSYKNhD1, NeurIPS 2023 TGL
workshop) applies the same idea to static and temporal isolated nodes, but
its intermediate is a pretrained embedding, not a relational object. The
sequence-based PPI literature is a domain-specific, leakage-audited
instantiation of this cluster — reviewed separately in §3.3.

**Training-time topology transfer — closest K2/K4 precedents (blueprint
baseline B1).** [Graph2Gauss](https://arxiv.org/abs/1707.03815),
[Graph2Feat](https://doi.org/10.1145/3543873.3587596), [Linkless Link
Prediction (LLP)](https://proceedings.mlr.press/v202/guo23f/guo23f.pdf),
[CAZI-MBN](https://arxiv.org/abs/2603.06618) compress training-graph
structure into a feature-only predictor's parameters — no per-query topology
object. [GLNN](https://arxiv.org/abs/2110.08727), [Cold
Brew](https://arxiv.org/abs/2111.04840), [Topology
Distillation](https://arxiv.org/abs/2106.08700), CSSL (arXiv:2201.10069,
TKDE 2022), and G-SPARC (arXiv:2411.01532, preprint) are adjacent
representation-distillation methods, not protocol-matched link predictors —
each produces an embedding, and G-SPARC's cold node still joins the observed
graph.

**Synthetic/learned topology tied to an observed graph — K2 boundary.**
[NodeDup](https://arxiv.org/abs/2402.09711), [LEAP](https://arxiv.org/abs/2503.03331)
attach a cold-start node to an existing graph. LEAP is the closest mechanism
comparator, but its inductive task is unseen–seen and the seen side still
needs an observed graph. Edgeless-GNN (arXiv:2104.05225, IEEE TETC 2023)
sits at the same boundary, building a kNN proxy graph from attribute
similarity against observed training nodes.

### 3.3 Sequence-based PPI prediction — domain instantiation of the endpoint-only cluster

Park & Marcotte (*Nature Methods* 2012, DOI 10.1038/nmeth.2259) define the
C1/C2/C3 split family, where **C3 — both proteins unseen — is exactly this
project's setting**. Bernett, Blumenthal & List (*Briefings in
Bioinformatics* 2024, DOI 10.1093/bib/bbae076) show random splits inflate
results because models learn similarity and degree, not interaction; Reim et
al. (*Bioinformatics* 2025, DOI 10.1093/bioinformatics/btaf192) report all
architectures plateau near 0.65 accuracy on the leakage-free benchmark — the
honest ceiling to cite before any absolute number. Szymborski & Emad
(*Nature Machine Intelligence* 2026, DOI 10.1038/s42256-025-01176-7) show
pretrained protein language models themselves leak for pair-input tasks, so
a frozen `x_i` from a pLM must disclose whether its pretraining corpus
overlaps the test proteins.

Every predictor in the lineage is PARTIAL — none builds a relational
intermediate: PIPR (*Bioinformatics* 2019), D-SCRIPT (*Cell Systems* 2021;
its inter-protein contact map is relational but confined to the two query
proteins, with no neighborhood), Topsy-Turvy (*Bioinformatics* 2022; injects
global network structure only as a GLIDE-derived training loss — no
topology variable exists in its forward pass, and its TT-Hybrid variant that
does use the network at inference violates the endpoint-only contract
outright), TT3D (*Bioinformatics* 2023), RAPPPID (*Bioinformatics* 2022,
C3-strict by construction), INTREPPPID (*BiB* 2024; strictness asserted by
lineage but not independently verified), SENSE-PPI (*iScience* 2024; only
thresholds pair scores rather than assembling topology), TUnA (*BiB* 2024;
uncertainty-aware), PLM-interact (*Nature Communications* 2025, the
strongest current pair scorer), and MKGR (arXiv:2607.01627, preprint; reaches
"novel–novel" pairs but retrieves external knowledge graphs at inference —
the same boundary violation kept separate for B4 in §3.6). Assembled-graph
evaluation for this domain is owned by PRING (§4).

### 3.4 Adjacent topology-learning machinery — K3 mechanism precedent, not protocol-matched

[LDS](https://proceedings.mlr.press/v97/franceschi19a.html),
[IDGL](https://arxiv.org/abs/2006.13009),
[NodeFormer](https://arxiv.org/abs/2206.08320),
[DGM](https://arxiv.org/abs/2209.14734) learn task-dependent graph structure;
GraphPatcher (arXiv:2310.00800, NeurIPS 2023) generates virtual neighbors at
test time but stays transductive and node-classification-scoped;
[NRI](https://proceedings.mlr.press/v80/kipf18a.html) infers latent graphs
for downstream dynamics; [FLEX](https://arxiv.org/abs/2507.11710) uses
generated subgraphs for OOD link prediction. Encoder machinery is grounded
separately in [GRIT](https://proceedings.mlr.press/v202/ma23c.html) and [Set
Transformer/PMA](https://proceedings.mlr.press/v97/lee19d.html) — used inside
the method, not comparison points.

### 3.5 Adjacent communities screened for novelty — no closer precedent found

Cold-start recommendation, open-world knowledge-graph completion, and
neural processes were screened (~220 titles) for a structurally analogous
mechanism. **Cold-start recommendation:** the generative branch (GAR, SIGIR
2022; CLCRec, MM 2021; GoRec, MM 2023; taxonomized in ALDI's SIGIR 2023
abstract) generates **embeddings**, never neighborhoods; works that
fabricate interaction context attach cold entities to **real,
identity-bearing** warm users or items — ColdLLM (WSDM 2025, DOI
10.1145/3701551.3703546) simulates interactions with retrieved real users,
Shallow-RHS (arXiv:2606.06225, preprint) retrieves warm surrogate neighbors
— and both-cold support (DropoutNet, NeurIPS 2017; Heater, SIGIR 2020) stays
embedding-level. **Open-world/text KG completion:** DKRL, ConMask, OWE (AAAI
2019), BLP (WWW 2021), StATIK (Findings-NAACL 2022), and RAILD
(arXiv:2211.11407) map text to an entity vector and score triples directly;
VN Network (CIKM 2020, DOI 10.1145/3340531.3411865) is a term collision
worth defusing explicitly — its "virtual neighbors" are rule-inferred
triples over real entities, and the emerging entity must still arrive with
observed links. **Neural processes:** Graph Neural Processes
(arXiv:1902.10042), NPGNN (arXiv:2109.14894), and RawNP (arXiv:2307.01204)
all condition on observed context edges.

### 3.6 Baseline-plan cross-reference (`docs/01-blueprint.md` §6)

B1 (training-time topology transfer) is the Graph2Feat/LLP/CAZI-MBN cluster
above. B4 (retrieval-grounded explicit scaffold) is EgoStitch, the project's
own comparison arm — not external literature; MKGR (§3.3) is the closest
cross-domain retrieval precedent to B4 and is rejected on the same
target-leakage boundary. B0/B2/B3/Oracle are ablation points on the
taxonomy's structure-blind ↔ structure-aware axis with no literature-cluster
mapping of their own.

## 4. Gap Conclusion

Transductive methods (§3.1) consume observed topology. Most zero-edge
inductive methods (§3.2, observed-topology-at-inference) remove topology at
inference by requiring an inference graph instead. Endpoint-only methods
(UPNA, the PPI lineage in §3.3) and training-time transfer (Graph2Feat/LLP/
CAZI-MBN, CSSL, G-SPARC) keep topology out of the per-query decision. LEAP
and Edgeless-GNN reconstruct topology only from a newcomer into an existing
graph, and §3.5's adjacent communities show the same pattern under different
vocabulary. Across 156 forward citations of the six closest prior methods
(including CAZI-MBN, EHDM [arXiv:2504.06193], and FLEX, which have zero
forward citations as of 2026-08-11 and so leave any concurrent successors
invisible to citation search), the full PPI lineage, and the ~220 titles
above, no work generates query-local topology for **two isolated unseen
nodes** and conditions their binary edge decision on it — the narrow gap
this project fills.

The evaluation half of this gap is already closed: [PRING,
arXiv:2507.05101](https://arxiv.org/abs/2507.05101) (preprint, 2025) defines
node-disjoint splits (≤40% sequence identity), BFS/DFS/random-walk-sampled
test graphs, and joint evaluation with GS, RD, and degree/clustering/spectral
MMD — the exact five headline numbers in this project's claim rules. PRING
is the project's benchmark lineage and must be cited as the evaluation
source, not claimed as a novelty; it also documents that all 11 benchmarked
predictors over-predict (best GS < 0.5, structural metrics off by an order
of magnitude) — the reviewer-legible failure this project's method targets.

> Across forward citations of the closest prior methods, the sequence-based
> PPI literature, cold-start recommendation, and open-world knowledge-graph
> completion (reviewed through 2026-08-11), we found no method that, from
> only the intrinsic features of two topology-isolated unseen nodes,
> generates an explicit identity-free relational topology object and uses it
> as input to symmetric binary edge prediction. Feature-only methods score
> pairs directly or through unconstrained embeddings; methods that
> materialize topology tie it to identified observed nodes or require
> observed edges at inference. We evaluate on PRING's joint pair and
> assembled-graph protocol, where existing predictors systematically
> over-predict.

*Verification: three Claude subagents (Semantic Scholar, OpenAlex, arXiv,
and primary-source web checks) ran this sweep on 2026-08-11 and the main
agent reconciled the results; every retained record was checked against a
primary source, unverifiable records were dropped rather than kept as
gray-zone hits, and preprints are labeled throughout.*
