# Feature-Conditioned, Identity-Free Latent Topology for Fully Inductive Edge Prediction

**Literature review date:** 2026-08-10  
**Review mode:** task-specific deep literature review, not a PRISMA systematic review  
**Target task:** node-disjoint, zero-observed-edge binary prediction for two unseen endpoints  
**Route under review:** conditional generation of an intrinsic-feature-conditioned, identity-free latent neighborhood/topology representation, used only as intermediate context for `edge(u,v)`

## Executive conclusion

**There is a strong literature basis for every individual ingredient, but this review found no direct precedent for their complete composition under the project's strict inference contract.** In particular:

1. **Feature-only topology transfer is established.** Graph2Gauss, DEAL, Graph2Feat, LLP, SA-MLP, SimMLP, SALE-MLP, HSAD, EHDM, and CAZI-MBN all learn some form of structure-shaped representation, attribution, or decision rule from attributes while removing some or all graph dependency at inference ([Graph2Gauss](https://openreview.net/forum?id=r1ZdKJ-0W); [DEAL](https://www.ijcai.org/proceedings/2020/0168.pdf); [LLP](https://proceedings.mlr.press/v202/guo23f.html)). Therefore, **“predicting topology-aware latent representations from intrinsic features” is not a defensible novelty claim**.
2. **Conditional topology generation is established.** CondGen generates whole graphs from semantic context; GraphMaker and an attributed extension of EDGE model adjacency conditional on node attributes; LGD, SGDIFF, and FLEX connect generative models to link prediction ([CondGen](https://papers.nips.cc/paper_files/paper/2019/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html); [GraphMaker](https://openreview.net/forum?id=0q4zjGMKoA); [EDGE](https://proceedings.mlr.press/v202/chen23k.html); [LGD](https://proceedings.neurips.cc/paper_files/paper/2024/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html); [SGDIFF](https://proceedings.mlr.press/v269/li25a.html)). Therefore, **“conditional graph generation” and “generative modeling for link prediction” are not defensible novelty claims**.
3. **Identity-independent structural summaries are established.** Anonymous walks discard vertex names; rooted graphlet and GraphWave signatures are root-preserving invariants; quantized graph tokens are candidate discrete vocabularies whose semantics and equivariance must be audited ([Anonymous Walk Embeddings](https://proceedings.mlr.press/v80/ivanov18a.html); [GraphWave](https://www.kdd.org/kdd2018/accepted-papers/view/learning-structural-node-embeddings-via-diffusion-wavelets); [Learning Graph Quantized Tokenizers](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html)). All derive targets from an observed graph and do not solve the endpoint-only edge task.

The remaining gaps are narrower and should be separated:

> **Method gap.** No verified paper in this search takes only the intrinsic features of two topology-isolated unseen endpoints, generates an explicit identity-free endpoint-local relational topology object, and uses that object as input to a symmetric binary edge classifier.
>
> **Evaluation gap.** No close paper in the reviewed corpus jointly evaluates its pair decisions and the topology of the assembled unseen-node graph.

These are **absence-of-found-precedent** statements, not proof of global priority. Stochastic conditional generation is the route being investigated, but is not used to exclude deterministic explicit topology generators from the prior-art test. The scientific question is whether an auditable relational bottleneck—and then its stochastic modeling—adds value beyond matched deterministic distillation and topology-generation controls.

## 1. Task-specific scope

The binding task is defined by the project rather than by the terminology used in a paper. Let

\[
G_{\mathrm{train}}=(V_{\mathrm{train}},E_{\mathrm{train}}),
\qquad V_{\mathrm{train}}\cap V_{\mathrm{test}}=\varnothing.
\]

For a test query \(\{u,v\}\subset V_{\mathrm{test}}\), inference receives only frozen intrinsic features \((x_u,x_v)\), observes no edge incident to either endpoint, and returns a symmetric binary probability. Scores over the fixed query universe are then assembled into \(\widehat G_\tau\) and evaluated at both edge and graph levels. The endpoint-only contract is uniform across [the project definition](../../docs/research-project.md) and [the blueprint](../../docs/01-blueprint.md); candidate/grounding pools belong only to a separately labeled retrieval-grounded arm.

One node-coherent form of the route is

\[
Z_u\sim q_\phi(Z\mid x_u),\qquad Z_v\sim q_\phi(Z\mid x_v),\qquad
T_{uv}=\operatorname{Stitch}(Z_u,Z_v;\epsilon_{uv}),
\]

\[
\widehat A_{uv}=p_\theta\!\left(Y_{uv}=1\mid x_u,x_v,\operatorname{Enc}(T_{uv})\right),
\]

where each \(Z_i\) may be cached within one stochastic assembled-graph draw and \(\epsilon_{uv}\) is optional pair noise. Stitching must be root-swap equivariant; encoding/readout and the final undirected edge probability must be root-swap invariant. \(T_{uv}\) is an **anonymous two-root topology object** or a latent with a verified decoder to one. Its non-root slots may have roles, types, and relations, but cannot be identified with training or test nodes.

Four symmetry terms are kept distinct: **identity-free** means no semantic node identity is encoded; **relabeling-invariant** means a rooted/graph-level value is unchanged by permitted vertex permutations; **equivariant** means node-indexed outputs permute with input vertices; **exchangeable** describes a probability law invariant under permutations. The two query roots remain distinguished from auxiliary slots, while the final undirected score is root-swap invariant.

### Direct-match criteria

A paper counts as a direct route match only if it satisfies all of the following:

- both endpoints are unseen and node-disjoint from training;
- inference has exactly endpoint intrinsic features and frozen parameters—no test edges, enclosing graph, trajectories, support triples, retrieved training nodes, or prototype identities;
- the intermediate variable is relational and identity-free, not merely an unconstrained embedding or direct pair score;
- the relational topology is an explicit classifier input;
- the edge output is symmetric. A stochastic parameterization is desirable for the proposed route but is not required for a paper to count as an architectural precedent.

Joint pairwise and assembled-graph evaluation is assessed separately as an **evaluation match**; its absence cannot make an otherwise identical method novel. Matched ablations are then required to show incremental predictive value, but are a scientific-evidence criterion rather than a prior-art filter.

Retrieving candidate-node features or persistent training-node prototypes violates the endpoint-only interpretation used here. Accordingly, the retrieval-grounded EgoStitch experimental arm is outside this review's direct-route class; its results cannot substantiate the endpoint-only novelty claim unless reported as a separate retrieval condition. The [EgoStitch specification](../../docs/05-egostitch-spec.md) explicitly feeds candidate tokens and initializes latent queries from candidate features, while [the blueprint](../../docs/01-blueprint.md) leaves the project method open.

## 2. Review method

### Research questions

1. Which papers already map intrinsic node features to structure-aware latent representations without needing an inference graph?
2. Which conditional graph generators already learn \(p(A\mid X)\), latent topology distributions, or generative link-prediction objectives?
3. Which anonymous structural representations could serve as identity-free prediction targets?
4. Does any end-to-end paper satisfy the endpoint-only architectural route? Separately, does any close paper jointly evaluate pairwise and assembled-graph outcomes?

### Search and screening protocol

Six GPT-5.6-Sol subagents at medium reasoning effort independently covered: research scoping, feature-to-structure transfer, conditional graph generation, cold-start link prediction, anonymous topology representations, and source/status verification. Technical claims were retained only when they could be tied to a primary paper, official proceedings page, or accepted OpenReview record. Preprints and submissions are explicitly marked; superseded records were not treated as current evidence.

Searches were run on 2026-08-10 through the arXiv API and primary-source web search. Query families combined:

- `latent topology`, `latent structure`, `node features`, `graph-free`, `linkless`;
- `fully inductive link prediction`, `isolated nodes`, `zero observed edges`;
- `feature-conditioned graph generation`, `conditional graph generation`, `p(A|X)`;
- `generated neighborhood`, `virtual neighborhood`, `ego-network generation`;
- `anonymous walk`, `structural role`, `graphlet`, `quantized graph tokenizer`;
- backward/forward name searches from DEAL, Graph2Gauss, Graph2Feat, LLP, SA-MLP, VQGraph, GraphMaker, and LGD.

Four deliberately strict arXiv queries were rerun at `2026-08-10T15:51:06Z` with `start=0`, `max_results=20`, `sortBy=relevance`, and `sortOrder=descending`:

| Exact arXiv `search_query` | Returned IDs | Screening outcome |
|---|---|---|
| `(all:"latent topology" OR all:"latent structure") AND all:"node features" AND all:graph` | `2407.10688`, `2601.20704` | PPGNN refines a generated graph using observed topology; the second paper is about LLM-generated reference lists. Neither matches. |
| `(all:"generated neighborhood" OR all:"virtual neighborhood") AND (all:"unseen node" OR all:"cold start")` | none | No exact-query result. |
| `(all:"ego-network generation" OR all:"local graph generation") AND (all:"link prediction" OR all:"edge prediction")` | none | No exact-query result. |
| `(all:"fully inductive link prediction" OR all:"linkless link prediction") AND (all:"node features" OR all:"graph-free")` | none | No exact-query result. |

The two results from the first query did not match the route: one inferred a graph over an observed node collection, and the other concerned citation-reference detection. Because exact-phrase searches have low recall, the conclusion below relies on the broader query families and citation chaining, not on these zero counts alone.

**Included:** primary methods that match the protocol, transfer topology into feature-only inference, generate attributed/latent graphs, use generative models for link prediction, or define identity-free structural targets.  
**Excluded from architectural direct-match status:** transductive edge splits; observed support topology; one-new-to-existing protocols; training-node retrieval; whole-graph synthesis without queried-edge classification; direct feature-only scoring without a relational latent. Evaluation match was coded separately as pair-only, graph-only, or joint.

## 3. Evidence synthesis

### 3.1 Feature → structure-aware latent or feature-only edge prediction is already a mature line

| Work | Source status | Actual inference input | Learned object / output | Route relevance and boundary |
|---|---|---|---|---|
| [Graph2Gauss, arXiv:1707.03815](https://arxiv.org/abs/1707.03815) | ICLR 2018 | Attributes of an unseen node | Gaussian \((\mu_i,\Sigma_i)\), trained with graph-distance ranking; energy supports links | Strongest early **distributional structural embedding** precedent. It does not sample or decode topology, and its LP evaluation is not the project's node-disjoint assembled-graph protocol. |
| [DEAL](https://www.ijcai.org/proceedings/2020/0168.pdf) | IJCAI 2020 | Endpoint attributes; one or both endpoints may be unseen | Attribute representation aligned with a structure encoder, then direct edge score | Direct feature-only both-unseen LP precedent. The topology is training-time alignment, not a generated local latent. |
| [Graph2Feat](https://doi.org/10.1145/3543873.3587596) | WWW 2023 Companion | New-node features; abstract states no connectivity | Student MLP embedding distilled from a GNN, then direct pair score | Strong graph-free inductive LP precedent. Exact unseen–unseen composition was not fully recoverable from the accessible primary text. Canonical first author is Ahmed E. Samy despite a conflicting KTH repository rendering. |
| [LLP, arXiv:2210.05801](https://proceedings.mlr.press/v202/guo23f.html) | ICML 2023 | Feature-only student; no neighborhood fetch | Anchor-centered rank/distribution relational distillation, then pair decoder | Reports new–new pairs and a strict linkless variant, although the strict table is not exclusively new–new. No explicit topology object or assembled topology evaluation. |
| [UPNA, arXiv:2307.08877](https://arxiv.org/abs/2307.08877) | Preprint | Exactly \((x_u,x_v)\), including both unseen/isolated | Direct edge probability | Exact input/task boundary, but its “latent graph generation” language refers to learning the edge mechanism; it does not generate intermediate topology. |
| [SA-MLP, arXiv:2210.09609](https://openreview.net/forum?id=MZ2kKZc8m7) | TMLR 2024 | Isolated variant maps \(x_i\) to an approximated latent structure embedding | Deterministic feature and structure latents for node classification | The clearest counterexample to a broad “feature → latent structure” novelty claim. It is node-level, deterministic, and has no topology decoder or LP assembly. |
| [VQGraph, arXiv:2308.02117](https://openreview.net/forum?id=h6Tz85BqRI) | ICLR 2024 | MLP is distilled toward soft local-substructure codes | Discrete codebook learned from local graph structure | Strong topology-token precedent, but deployed output is deterministic and node-classification oriented. Its reported inductive protocol transfers positional features using test edges, so it is **not** evidence for strict isolated inference. |
| [SimMLP, arXiv:2412.03864](https://arxiv.org/abs/2412.03864) | WSDM 2025 | Node features only after pretraining | Deterministic MLP representation aligned to a context-aware GNN; includes LP experiments | Strong self-supervised transfer baseline, but its LP protocol is not verified as two-isolated-endpoint node-disjoint. The older [arXiv:2402.08918](https://arxiv.org/abs/2402.08918) is withdrawn and must not be cited as the live paper. |
| [SALE-MLP](https://www.ijcai.org/proceedings/2025/668) | IJCAI 2025 | Graph-free MLP/projector | Topology-aligned continuous latent learned with structural losses; node and link tasks | Direct counterexample to “first structure-aware latent from features,” but deterministic and without a generated anonymous topology object. |
| [EHDM / *Weak Models Can be Good Teachers*, arXiv:2504.06193](https://openreview.net/forum?id=TSwEOuoO00) | LoG 2025 | Endpoint features only | Link predictors distilled from structural heuristics such as common neighbors | Particularly important control: cheap deterministic topology transfer may match the proposed route without generative machinery. The arXiv version retains the earlier title *Heuristic Methods are Good Teachers to Distill MLPs for Graph Link Prediction*. |
| [HSAD](https://doi.org/10.1016/j.neucom.2025.131910) | Neurocomputing 660, 2026 | Graph-independent student features/pairs | High-order structural attribution distilled into an MLP link predictor | Very close task precedent, but the accessible primary record does not establish the same strict node-disjoint split; no explicit topology object or graph assembly. |
| [CAZI-MBN, arXiv:2603.06618](https://arxiv.org/abs/2603.06618) | ICLR 2026 | Intrinsic sequence features for two novel biological entities | Topology-aware teacher latent distilled into a topology-agnostic student and interaction classifier | Strongest domain/protocol threat. It blocks “first topology-aware zero-shot biological interaction predictor,” but it remains deterministic latent transfer and reports no assembled-network topology. Its stated 75/15/15 entity split is internally inconsistent and should be independently audited before use as protocol evidence. |

Supporting graph-free distillation includes [GLNN](https://openreview.net/forum?id=4p6_5HBWPCw), [Graph-MLP](https://arxiv.org/abs/2106.04051), [TINED](https://openreview.net/forum?id=nshtqLv4r4), and [PGKD](https://openreview.net/forum?id=3aMK3rg377). These reinforce the density of deterministic graph-to-MLP transfer, mostly for node classification. [Cold Brew](https://openreview.net/pdf?id=1ugNpm7W6E) is an important negative boundary: its strict cold-start representation uses feature-based prediction plus retrieval of stored training-node teacher embeddings, so it violates this review's identity-free endpoint-only rule.

**Synthesis.** The novelty cannot reside in removing graph access at inference, transferring topology into features, predicting a structure-shaped vector/code, or handling two unseen endpoints. Those capabilities already exist separately and, in DEAL/UPNA/CAZI-MBN, substantially overlap the target protocol. The unresolved architectural step is an **explicitly relational, anonymous, load-bearing topology object**; whether stochastic modeling adds value is a secondary hypothesis.

### 3.2 Conditional graph generation and generative link prediction exist, but under different conditioning regimes

| Work | Source status | Condition | Generated object | Why it is not a direct match |
|---|---|---|---|---|
| [CondGen](https://papers.nips.cc/paper_files/paper/2019/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) | NeurIPS 2019 | Graph-level semantic context such as community or disease descriptors | Whole anonymous graph via a variational generative adversarial model | Canonical permutation-invariant conditional-structure-generation ancestor, but its condition describes an entire graph; there are no queried endpoints or binary edge classifier. |
| [GraphMaker, arXiv:2310.13833](https://openreview.net/forum?id=0q4zjGMKoA) | TMLR 2024 | A complete set of categorical node attributes/labels | Entire attributed graph via discrete diffusion; async variant models edges after attributes | Genuine \(p(A\mid X)\), but the nodes are an explicit whole-graph universe, not anonymous endpoint-local slots used for one edge decision. |
| [EDGE, arXiv:2305.04111](https://proceedings.mlr.press/v202/chen23k.html) | ICML 2023 | Primarily degree sequence; attributed appendix incorporates \(X\) | Entire large graph | Establishes scalable discrete topology diffusion and an attributed factorization, not pair-local endpoint-only inference. |
| [NGG, arXiv:2403.01535](https://arxiv.org/abs/2403.01535) | Preprint | Fifteen **whole-graph statistics**, not intrinsic node features | Entire graph through VAE compression and latent diffusion | “Feature-conditioned” in its title means graph-property-conditioned. It must not be cited as unseen-node feature conditioning. |
| [LGD, arXiv:2402.02518](https://proceedings.neurips.cc/paper_files/paper/2024/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html) | NeurIPS 2024 | Graph properties or a partially observed graph with all node features and some edges | Latent node/edge/graph variables; link prediction as masked graph inpainting | Already unifies generation and link prediction, but link inference consumes observed adjacency and uses random edge splits. |
| [SGDIFF, arXiv:2409.08487](https://proceedings.mlr.press/v269/li25a.html) | LoG 2024 conference; PMLR 269 (2025) | Observed \(k\)-hop enclosing subgraph, features, and candidate class | Class-conditional likelihoods for structure/features; Bayes link score | Strongest accepted “generative link prediction” precedent. It evaluates likelihoods of observed target-edge-masked topology instead of imagining context for isolated endpoints. |
| [FLEX, arXiv:2507.11710](https://arxiv.org/abs/2507.11710) | Preprint / ICLR 2026 submission | Observed target-link enclosing subgraph and its node features | Counterfactual adjacency over those same observed nodes | Closest “generated subgraphs improve LP” threat, but generation is training augmentation from observed topology; inference still uses an ordinary predictor. |
| [Deep Generative Models for Subgraph Prediction, arXiv:2408.04053](https://arxiv.org/abs/2408.04053) | ECAI 2024 | Attributed evidence subgraph with observed links/features/labels | Joint missing links, labels, and features | Conditional subgraph prediction, but its evidence adjacency is forbidden by the target protocol. |
| [GLAD, arXiv:2403.16883](https://ojs.aaai.org/index.php/AAAI/article/view/34169) | AAAI 2025 | No endpoint-feature condition in its generic setting | Whole graph from quantized permutation-equivariant latent node codes | Strong latent-topology machinery, but neither feature-conditioned nor query-local. |
| [GFKD, arXiv:2105.07519](https://www.ijcai.org/proceedings/2021/320) | IJCAI 2021 | Frozen teacher GNN plus optional coarse priors | Sampled pseudo-graphs for data-free distillation | It genuinely models discrete topology, but for teacher inversion—not amortized unseen-endpoint inference. |

[NRI](https://proceedings.mlr.press/v80/kipf18a.html) infers a distribution over latent interaction edges, but requires full object trajectories. [DGM](https://arxiv.org/abs/2002.04999) and [Edgeless-GNN, arXiv:2104.05225](https://arxiv.org/abs/2104.05225) construct proxy/task graphs from a collection of observed feature vectors, but those nodes remain explicit candidates and the methods are not conditional graph distributions. Classical [GraphVAE](https://arxiv.org/abs/1802.03480), [Graphite](https://proceedings.mlr.press/v97/grover19a.html), and [Graph Normalizing Flows](https://papers.neurips.cc/paper_files/paper/2019/hash/1e44fdf9c44d7328fecc02d677ed704d-Abstract.html) establish latent adjacency decoding for full graphs, not the strict endpoint-local task.

**Synthesis.** Graph generation supplies the stochastic modeling machinery, but most of it assumes either an entire explicit node set, a graph-level property vector, or an observed partial graph. The proposed route instead needs an anonymous relational state whose slots are not candidate identities and whose only external condition is the endpoint features.

### 3.3 Relabeling-invariant/equivariant structural summaries are representable, but existing methods observe them rather than predict them

| Target family | Primary precedents | What is invariant / transferable | Limitation for this task |
|---|---|---|---|
| Anonymous walks / rooted tokens | [Anonymous Walk Embeddings](https://proceedings.mlr.press/v80/ivanov18a.html), [Role2Vec](https://arxiv.org/abs/1802.02896) | Rooted values discard vertex names and are invariant under non-root relabeling; node-indexed collections are equivariant | Walks are sampled from an observed graph; the papers do not predict their distribution from an isolated node's intrinsic features. |
| Graphlet/orbit distributions | [Pržulj, 2007](https://pubmed.ncbi.nlm.nih.gov/17237089/) | Exact relabeling-invariant local motif counts | Interpretable but lossy beyond the selected graphlet radius/order. |
| Structural-role signatures | [GraphWave](https://www.kdd.org/kdd2018/accepted-papers/view/learning-structural-node-embeddings-via-diffusion-wavelets), [DeepGL](https://doi.org/10.1145/3184558.3191524) | Rooted GraphWave values are root-preserving invariants; DeepGL learns shared relational functions with equivariant node outputs | Targets require a graph Laplacian or observed destination topology. |
| Rooted WL/computation-tree vocabulary | [graph2vec](https://www.mlgworkshop.org/2017/paper/MLG2017_paper_21.pdf) | Rooted subtree codes and multiset aggregation can be relabeling-invariant | Tree codes may miss cycles unless augmented; original setting embeds observed graphs. |
| Learned quantized topology tokens | [VQGraph](https://openreview.net/forum?id=h6Tz85BqRI), [Learning Graph Quantized Tokenizers](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html) | Assignments are equivariant only if the tokenizer is; codebook indices are arbitrary up to code permutation | Token indices have no guaranteed stable topology semantics; collapse and split-wise code drift require explicit audits. |
| Exchangeable mesoscale latents | [Contextual SBM](https://proceedings.neurips.cc/paper/2018/hash/08fc80de8121419136e443a70489c123-Abstract.html) | The random-graph law is vertex-exchangeable | Community labels are identifiable only up to block-label permutation, and the model jointly observes covariates and graph. |

Observed-topology link predictors such as [SEAL](https://proceedings.neurips.cc/paper_files/paper/2018/hash/53f0d7c537d99b3824f0f99d62ea2428-Abstract.html), [GraIL](https://proceedings.mlr.press/v119/teru20a.html), [CAW](https://iclr.cc/virtual/2021/poster/2651), and [NBFNet](https://proceedings.nips.cc/paper_files/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html) show that entity-ID-independent or relabeling-equivariant structural reasoning can be useful for link prediction. SEAL/GraIL/NBFNet preserve query-root/relation conditioning, while CAW preserves temporal/query roles. All consume observed paths, histories, or enclosing subgraphs.

### 3.4 Evidence convergence

| Claim | Evidence | Assessment |
|---|---|---|
| Feature-only fully inductive edge prediction exists | DEAL, UPNA, Graph2Feat, LLP, CAZI-MBN | **Established; high confidence.** |
| Training topology can be compressed into feature-only latents/scores | Graph2Gauss, SA-MLP, SimMLP, SALE-MLP, EHDM, HSAD | **Established; high confidence.** |
| Node attributes can condition generated adjacency | GraphMaker; attributed EDGE factorization | **Established for whole-graph generation; high confidence.** |
| Generative models can perform or assist link prediction | LGD, SGDIFF, FLEX, ECAI subgraph prediction | **Established when support topology is observed; high confidence.** |
| Intrinsic endpoint features alone can generate an explicit two-root relational topology object used for edge classification | No direct match found; Graph2Gauss/SA-MLP/VQGraph and CondGen/GraphMaker cover separate halves | **Open method intersection; moderate confidence as a literature-gap statement.** |
| Such a latent improves both pair metrics and assembled topology | No verified close precedent reports the joint evaluation | **Open empirical question; high confidence that it is not established by the reviewed corpus.** |

## 4. Fit for the project: what would count as the proposed route?

### 4.1 The latent must be auditably topological

A generic vector \(z=f(x)\), even if supervised by a GNN, is already covered by the distillation literature. To sustain a topology claim, at least one of these must hold:

- \(z\) decodes to an anonymous two-root graph: its auxiliary-slot distribution is invariant under slot permutations, its relational decoder is equivariant under auxiliary-slot permutations and root exchange, and quotient/readout is invariant;
- \(z\) is a distribution over two-root graph isomorphism classes with the root-swap action specified explicitly;
- weaker structural-summary latents—topology tokens, degree/graphlet/orbit distributions, or anonymous-walk counts—are labeled as summaries unless a decoder or graphicality constraint produces valid relational objects.

Raw coordinate matching to an arbitrary teacher embedding is insufficient because rotations, code permutations, and non-identifiability can preserve downstream performance while destroying any stable topology semantics. Graph2Gauss motivates distributional role embeddings, VQGraph motivates discrete codes, and graphlet/anonymous-walk work provides invariant audit targets, but the end-to-end composition is a new design inference rather than a published result.

### 4.2 “Generative” requires more than deterministic distillation

The safest terminology is:

- **deterministic structure transfer:** \(z_i=f_\phi(x_i)\), trained by MSE, contrastive, ranking, or code-classification loss;
- **conditional generative topology:** a graph-valued \(T_i\sim q_\phi(T\mid x_i)\) or \(T_{uv}\sim q_\phi(T\mid x_u,x_v)\), or a latent decoded to a valid relational object, trained by a likelihood, ELBO, score/flow objective, or another distributional criterion.

This distinction matters because structurally different neighborhoods can share similar mean statistics. A distributional objective alone does not establish useful multimodality: experiments must separately test non-collapse/diversity, conditional calibration or coverage, graphical validity, and utility beyond the conditional mean/MAP.

### 4.3 Node-level coherence must be specified explicitly

A symmetric pair-conditioned latent can change pair scores and hence the thresholded graph, but it does not create shared node- or triangle-level stochastic coherence. Caching one \(Z_i\) per assembled-graph draw induces dependence among incident edge decisions, yet still does not ensure that independently predicted degree/motif marginals are jointly realizable. If reported \(p_{uv}\) marginalizes the latent and expected scores are thresholded, caching versus resampling changes only Monte Carlo error. A shared-cache protocol must therefore be specified as a stochastic graph-draw procedure and audited for compatibility/realizability. Edge-independent graph models can remain limited in reproducing high triangle density ([Chanpuriya et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/cc9b3c69b56df284846bf2432f1cba90-Abstract.html)).

### 4.4 Recommended first diagnostic target

The lowest-risk first test is a **multiscale anonymous structural-summary distribution**, not a high-capacity latent graph diffusion model:

\[
q_\phi\bigl(d,\;w,\;t,\;\mathbf g,\;\mathbf a\mid x_i\bigr),
\]

where \(\mathbf g\) contains root-centered graphlet-orbit statistics (potentially subsuming \(d,w,t\)) and \(\mathbf a\) contains rooted anonymous-walk/token frequencies. These are interpretable teacher/diagnostic targets, but the joint vector may be non-graphical and does not determine spectral structure. It becomes a topology object only with a valid decoder/projection and compatibility audit. A second stage can test mixture-Gaussian role latents or frozen quantized tokens if the summaries are predictable.

### 4.5 Minimum controls and baselines

The literature makes the following comparisons mandatory:

1. **Protocol/task baselines:** DEAL, UPNA, Graph2Feat, LLP, Graph2Gauss, CAZI-MBN, and the existing independent pair scorer.
2. **Strong deterministic transfer:** SA-MLP-like latent regression, VQGraph-like code prediction, SALE-MLP/SimMLP alignment, HSAD/EHDM-style structural attribution or heuristic distillation.
3. **Latent semantics controls:** same-size unconstrained latent vector; shuffled structural targets; random topology tokens; label-only distillation; topology targets with the queried training edge masked.
4. **Generative-value controls:** mean/MAP structural summary or decoded graph, as applicable, versus stochastic samples; matched stochastic assembled-graph draws for cached-node versus pair-resampled variants; matched-parameter deterministic decoder; diversity, coverage, and graphical-validity audits.
5. **Architectural-use controls:** provide the topology loss but remove \(\operatorname{Enc}(T)\) from the classifier; provide \(\operatorname{Enc}(T)\) but stop its edge-loss gradient. If either matches the full model, “topology as input-side context” is not load-bearing.
6. **Evaluation:** identical query universe and threshold policy; pairwise discrimination/calibration plus the project's GS, global and BFS-macro RD, and degree/clustering/spectral MMD ratios.

For training nodes, every topology target must exclude the queried partner and the queried edge's contribution to degree/motif statistics. A reusable cached node target cannot silently contain the queried contribution: supervision must remain query-specific, or the reusable target must be constructed without any supervised query edge.

## 5. Claim guidance

### Claims contradicted by verified literature

Do **not** claim:

- first feature-only or fully inductive link predictor;
- first method to transfer graph topology into an MLP or intrinsic-feature latent;
- first topology-aware zero-shot biological interaction predictor;
- first feature-conditioned graph generator;
- first generative model used for link prediction;
- first method to generate subgraphs for link prediction;
- that NGG conditions on intrinsic node features;
- that VQGraph's published inductive result is a strict zero-observed-edge result.

### Claim-safe formulation

> Prior work separately establishes feature-only topology transfer for unseen-node link prediction, relabeling-aware structural summaries, conditional whole-graph generation, and generative link prediction from observed neighborhoods. In the primary literature reviewed through 2026-08-10, we found no end-to-end method that, from only the intrinsic features of two topology-isolated unseen nodes, generates an explicit identity-free two-root relational topology object and uses it as input to symmetric binary edge prediction. Separately, no close method in the reviewed corpus jointly evaluates pairwise accuracy and the topology of the assembled unseen-node graph.

The claim should remain qualified until a broader database search or independent novelty review finds no counterexample. More importantly, experiments must show that the explicit/generative topology arm beats the deterministic topology-transfer baselines; otherwise the contribution reduces to another form of representation distillation.

## 6. Limitations and source-quality notes

- This is a structured deep review, not an exhaustive systematic review. Exact-phrase arXiv queries had low recall, so citation chaining and broader primary-source searches were necessary.
- Absence claims are intrinsically fragile. Different communities use terms such as structural role, graph-free distillation, latent graph inference, cold start, relational distillation, and attributed generation for overlapping ideas.
- Graph2Feat's accessible official abstract establishes no-connectivity new-node inference, but the exact endpoint composition of its original evaluation was not fully primary-source verified.
- HSAD's publisher record supported its high-order structural-attribution and LP claims, but the strictness of its node split was not independently established from an accessible full primary copy.
- UPNA, NGG, FLEX, and [*Principled Latent Diffusion for Graphs via Laplacian Autoencoders*, arXiv:2601.13780](https://arxiv.org/abs/2601.13780) are preprints/submissions rather than settled archival evidence. They are useful for novelty boundaries but should be labeled accordingly.
- CAZI-MBN is highly relevant and domain-matched, but its paper contains an internally inconsistent stated split percentage; its implementation and leakage contract require separate audit.
- The PPI graph is incomplete and ascertainment-biased. Even a well-calibrated latent may learn observed-study topology rather than biological topology; this review establishes ancestry and design boundaries, not scientific validity.

## 7. AI-assisted research disclosure

This report was produced with AI-assisted literature search, source triage, synthesis, and adversarial checking. Six GPT-5.6-Sol subagents at medium reasoning effort performed independent topical sweeps and a primary-source citation audit. The main agent reran arXiv searches, reconciled conflicts, and wrote the final synthesis. Bibliographic identities and technical boundaries were checked against primary papers or official venue records where accessible. Remaining unresolved boundaries are marked in Sections 3 and 6; no unresolved claim is used as direct-match evidence.
