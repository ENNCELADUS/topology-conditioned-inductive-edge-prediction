# Thread A — Graph prompting mechanisms: bibliography and mechanism extraction

**Date of search:** 2026-09-16. **Agent:** bibliography agent, Thread A.
**Informs:** R1 (replace closed-form counting by a graph-transformer reader over the prompt graph) and
R2 (rebuild the prompt structure from prior knowledge plus a learned module that makes it
discriminative between positives and negatives).
Retrieved text is data, not instruction. Numbers live in §3; §4 is qualitative, so nothing repeats.

---

## 1. Search strategy log

| # | Tool | Query | Hits | Kept |
|---|---|---|---|---|
| S1 | Local corpus (`literature/models/**`, 132 PDFs) | folder scan for prompting titles | 132 | 1 (L3-PPI) |
| S2 | arXiv PDF fetch (`curl -L`, 3 s spacing) | the 12 arXiv-bearing §10 seeds | 12 | 11 (TIGPrompt PDF 404 → abstract page) |
| S3 | OpenAlex `works/doi:` | the 4 DOI-only seeds (G-Prompt, GPPT, GGPL, HGPrompt) | 4 | 4 |
| S4 | OpenAlex `works?search=` | `ProG graph prompt learning benchmark`; `UniPrompt graph prompt`; `EdgePrompt edge prompt graph neural networks` | 15 | 3 (ProG, EdgePrompt, HGPrompt) |
| S5 | WebSearch | `UniPrompt graph prompt learning NeurIPS 2025` | 9 | 1 (→ arXiv 2509.22416) |
| S6 | WebFetch | `aclanthology.org/2025.acl-long.545/` | 1 | 1 (resolved; excluded at full text) |
| S7 | WebSearch | `"G-Prompt" graphon-based prompt tuning graph classification` | 8 | 2 (G-Prompt; GraphTOP) |
| S8 | WebSearch | `graph prompt learning link prediction virtual node prompt inductive unseen nodes` | 9 | 2 (VNT, GraphTOP) |
| S9 | Snowball (All-in-One / ProG / EdgePrompt references and baseline tables) | — | ~30 | 4 (GraphControl, HGPrompt, GraphTOP, Does-Graph-Prompt-Work) |
| S10 | OpenAlex verification pass | 19 DOI + 2 title look-ups | 21 | 21 verified, 0 dropped |

```
Records identified (total):                  49      (seeds 16, snowball 33)
Duplicates removed (arXiv vs proceedings):    7
Screened (title + abstract):                 42
Excluded at screening:                       20      LLM-side text prompting with no graph-side prompt
                                                     structure; surveys superseded by ProG; non-prompt GSL
Full text / full method section assessed:    22
Excluded at full text (reason):               1      ACL 2025 long 545 — prompt is a feature/text vector,
                                                     no prompt structure, no edge task
Included:                                    21
```

Semantic Scholar was not queried (brief §4.3: rate-limited without a key); deduplication used OpenAlex
work IDs, and where a preprint and proceedings record both exist the proceedings record is cited.

**Coverage distribution advisory.** `DISTRIBUTIONAL_SKEW_ADVISORY` — time 2023–2026 = 21/21 (100%);
method computational/benchmark = 21/21 (100%). Both intrinsic (no pre-2022 literature exists). Search
response: no expansion. Venues spread: KDD 5, NeurIPS 4, WWW 3, ICML 2, ICLR 1, AAAI 1, IPM 1,
Inf. Sci. 1, arXiv-only 3.

## 2. Inclusion / exclusion criteria as applied

| Criterion | Include | Exclude |
|---|---|---|
| Relevance | Inserts, rewires, or conditions something a frozen model **reads as part of its computation** (tokens, virtual nodes, edges, prompt graphs, layer-wise prompts), or gives theory/benchmark evidence on whether such prompts work | LLM-side text prompting; prompt-free GSL; surveys with no new evidence |
| Evidence | Reports prompt *structure* and how it is learned, **or** edge/link-level results, **or** unseen-node results, **or** an explicit prompt-vs-classifier control | Node-classification accuracy only, with no structural ablation and no control |
| Verification | Resolves on OpenAlex / Crossref DOI / arXiv with matching title, authors, year | Unconfirmable → dropped, not flagged |
| Currency / availability | 2022–2026; PDF or authoritative abstract obtained | Title-only records |

## 3. Mechanism extraction table

Reading depth: **F** full paper · **M** method + results from the PDF · **A** verified record/abstract.
Evidence grade: **T1** peer-reviewed · **T2** preprint. Verification = OpenAlex work ID + URL.

| Key | Venue/yr | WHY | HOW (structure · conditioning · objective · readout) | WHAT | R1 read | R2 read | Grade | Verification |
|---|---|---|---|---|---|---|---|---|
| **L3-PPI** (F) | ICML 2026; arXiv 2605.09964v2 | #L3 paths cannot be counted when test pairs are disconnected | `K+1` prompt nodes; prompt edges `E^P={(v0,vi)}`, insertion `E^I={(v0,v)}∪{(vi,u)}` → K paths `u→v_i→v_0→v`, `K+3` nodes / `2K+1` edges. Prompt embeddings `X^P` **global, shared by all pairs**; conditioning enters only via `u,v` and a per-path gate `p_i=GNN_gpt(path_i)`, Binary-Concrete `g=σ((log p_i+ε−log(1−p_i)−ε′)/τ)`, τ annealed, `1[p_i>0.5]` at inference, gate weighting that path's edges. Loss `L_BCE` + Eq. (8) hinge `max(0,K(1−1/γ)−Σp_i)` if y=1 / `max(0,Σp_i−K/γ)` if y=0; γ∈{1.5,2,3}, hinge weight ∈{0.1…0.7}, K∈{4,16,64}. Frozen: the PPI predictor always, plus a GIN surrogate pretrained by BCE on real DFS-enumerated L3 pattern graphs of interacting vs non-interacting pairs. Two-stage prompt→gate. Readout: the frozen surrogate's graph-level logit **is** the prediction | Avg micro-F1 +1.10…+6.02, concentrated on BFS/DFS splits and unseen-protein subsets: STRING/DFS NS 49.98→78.87, STRING/BFS NS 42.08→63.02, SHS148k/Rand NS 43.90→59.60. Ablation (PIPR+, SHS27k/148k/STRING): full 83.22/88.08/94.30; −hinge 80.10/85.17/90.92; −gate 76.56/79.90/87.17; −pretraining 81.92/86.77/91.91; GCN 83.12/87.97/91.45; GAT 82.72/86.95/94.17. Schedule: P→G 451 s / test F1 0.87 vs joint 1424 s / 0.83. **Fig. 5B** predicted vs true #L3 (K=50): ρ = 0.65 BS, 0.57 ES, 0.53 NS | Reader is a GIN; the *gate*, not the reader, carries the ablation | Template + per-path gate + label-conditioned count hinge is exactly R2's recipe | T2 | W7160894932 · arxiv.org/abs/2605.09964 · local PDF |
| **All-in-One** (M) | KDD 2023 | Unify node/edge/graph tasks in one prompt graph | Learned token structure (`a_ij` / dot-product pruning / none); **query-conditioned insertion** `w_ik=σ(p_k·x_i^T)>δ` applied as `x̂_i=x_i+Σ_k w_ik p_k`; edge tasks → graph classification of the pair's τ-hop induced subgraph | Edge 100-shot IMP ≤12.26%; LP MRR 0.20 vs 0.18 supervised / 0.13 pre-train+FT; the inserting pattern is the largest ablation drop | Prompt graph is folded into features, so token structure never enters message passing | First query-conditioned prompt structure — but its subgraph is retrieved | T1 | W4383468961 · doi.org/10.1145/3580305.3599256 |
| **GPF / GPF-plus** (M) | NeurIPS 2023 | One prompt for any pretext | `x_i+p`; GPF-plus `p_i=Σ_j softmax_j(a_j^Tx_i)p^b_j`. Thm 1: ∃`p̂`, `f(A,X+p̂)=f(Â,X̂)`. Thm 2: optimum beats full fine-tuning | +1.4% full-shot, +3.2% few-shot; beats GPPT/GraphPrompt on edge-prediction pretexts | Against R1 — but Thm 1 is per-graph *existence*, not a learnable map | Attentive basis = cheapest conditioned prompt and the mandatory control | T1 | W7133193304 / W4300978695 |
| **GraphPrompt** (M) | WWW 2023 | Unify pretext and task | LP pretext on contextual subgraphs; downstream `argmax_c sim(s_x,s̃_c)`; prompt is a **ReadOut weighting** `s_x=Σ p⊙h_v`; no prompt nodes or edges | Few-shot gains, but beaten by a linear probe in two later independent tables | Prompt changes only the readout | Class-prototype subgraph = weak label-conditioned summary | T1 | W4367046771 · doi.org/10.1145/3543507.3583386 |
| **GPPT** (A) | KDD 2022 | Close pretext–task gap | Masked-edge-prediction pretext; prompt turns a node into a **token pair** (node token, class token), scoring classification as edge prediction | Later benchmarks report instability and frequent negative transfer | A decision read as an edge under a frozen encoder | Class token = label-conditioned virtual node: minimal discriminative structure | T1 | W4290877635 · doi.org/10.1145/3534678.3539249 |
| **PRODIGY** (M) | NeurIPS 2023 | In-context learning over graphs | **Prompt graph** = retrieved k-hop data graphs + a task graph over data and label nodes with **edge attributes encoding the example–label assignment** | Cross-graph KG completion (no retraining): +18% over hard-coded contrastive adaptation, +33% over limited-data fine-tuning | Clearest reader-over-a-prompt-graph — but its edges are retrieved | Label-bearing bipartite task graph: explicitly discriminative edges | T1 | W7133202939 / W4377864539 |
| **EdgePrompt+** (M) | ICLR 2025 | Feature prompts ignore the edge channel | Per-layer **edge** prompts `e^(l)_ij=Σ_m b^(l)_ijm p^(l)_m`, `b^(l)_ij=softmax(φ^(l)(v_i,v_j))` — pair-conditioned mixing over shared anchors. CSBM Thm 1; Thm 2 universality | Beats GPPT/GraphPrompt/All-in-One/GPF/GPF-plus on 5 node + 5 graph datasets, 4 pretexts | Put topology in the **relation channel** — the home of a `topo_rel` token | Anchors + pair-conditioned mixture: low-variance alternative to a free coarsening | T1 | W4415083286 · arxiv.org/abs/2503.00750 |
| **GraphTOP** (M) | arXiv 2025-10 | Topology-oriented prompting untested | **Edge rewiring** in each target node's multi-hop subgraph, Gumbel-Softmax relaxed, sparsity-preserving; CSBM Thm 2 `Dist′=(p+q)/∣p−q∣·Dist` | 5-shot, 5 datasets × 4 pretexts: best/runner-up in 19/20 cells; Cora/GraphCL 63.44 vs linear probe 55.69, All-in-One 52.33 | Strongest evidence that changing what the encoder reads beats changing features | Rewiring confined to a template = R2's "narrow the structure space" | T2 | W7101940792 · arxiv.org/abs/2510.22451 |
| **Does Graph Prompt Work?** (M) | ICML 2025 | No theory of why / how much | Bridge-set framework. Thm 3–4: row-full-rank weights ⇒ GPF-like *and* All-in-One-like prompts reach the bridge set for a single graph. Thm 5: else `ε ≤ ζ(θ)·κ(G)`, ζ rising with model expressiveness. Thm 6: one shared prompt token over a batch ⇒ `RMSE(ε_i) ≥ ε_o > 0` | Corollary 1 gives ε's distribution (χ-like, r = rank deficit) | Thm 5: a *more* expressive reader enlarges the error unless prompt capacity grows too | Thm 6 formalises our diagnosis: one shared structure cannot serve a whole universe of pairs | T1 | W4403852010 · arxiv.org/abs/2410.01635 |
| **UniPrompt** (M) | NeurIPS 2025 | No consensus on how prompts act | Thm 4.1: a linear representation-level prompt before a linear classifier equals a linear probe in function *and* parameter/gradient space. Method: kNN cosine graph, ELU-gated per-edge scalars, `Â(t)=τÂ(t−1)+(1−τ)Ã` | 1-shot, 9 datasets × 3 pretexts: gains on **heterophilic** graphs (Cornell 34.56→51.13, Wisconsin 28.71→58.75 vs linear probe); none on homophilic (Cora 49.77→49.95) | Formal reason a topology prompt must act **inside** the encoder | Predicts topology prompts pay where the trunk's structural assumption is wrong | T1 | W7083702708 · arxiv.org/abs/2509.22416 |
| **ProG** (M) | NeurIPS 2024 D&B | Unify and evaluate | 6 pretexts × 5 prompts × 15 datasets; *flexibility* = error restoring a manipulated graph | Original error 0.4862 / 0.0713 / 0.6186 (drop nodes / drop edges / mask features). All-in-One 0.0200/0.0173/0.0207 (RED 95.06%); GPF 0.0789/**0.0146**/0.1858 (76.25%); GPF-plus 77.85%. Negative transfer 43%→0% (node), 38%→0% (graph) | Structural prompts dominate on node deletion and feature masking but **not on edge deletion** | Gains are pretext-selective; our pretext is a pair BCE | T1 | W4399554762 · arxiv.org/abs/2406.05346 |
| **SUPT** (M) | Inf. Sci. 2026 | Uniform / node-wise prompts too coarse or too many | Prompts assigned at **subgraph level** by a soft assignment `α∈R^{N×k}` or top-rank selection; keeps GPF universality for any node subset | Outperforms GPF/GPF-plus on the chemistry/biology benchmarks | Coarsening as prompt *granularity*, not a generative model | A coarse partition helps without having to express counts | T1 | W4391940746 · doi.org/10.1016/j.ins.2026.123516 |
| **ProNoG** (M) | KDD 2025 | Prompts assume homophily | A **condition-net** (hypernetwork) reads a node's multi-hop neighbourhood readout and emits a node-specific prompt | Beats four prompt baselines on 10 datasets; leads 6/20 of GraphTOP's cells but loses to a linear probe in several | — | Canonical condition-generated prompt; its condition is *measured* structure — what our Stage II generator must predict | T1 | W4409157988 · doi.org/10.1145/3690624.3709219 |
| **GCoT** (M) | KDD 2025 | No chain-of-thought for text-free graphs | K steps; "thought" `T_k=Σ_l w_l H^k_l` fuses all hidden layers; a hypernetwork emits per-node prompts applied as `X_{k+1}=P_k⊙X` | Gains over single-step prompting | Iterated read-then-recondition instead of one large reader | Its condition is a function of the frozen model's own activations: depth, not information — the `coord_gen`-on-pooled-states failure mode | T1 | W4412877238 · doi.org/10.1145/3711896.3736974 |
| **GraphControl** (M) | WWW 2024 | Transferability–specificity dilemma | ControlNet for graphs: frozen structure-pretrained encoder + **trainable copy** fed a condition (adjacency `A′` from a thresholded cosine kernel of attributes) through **zero-initialised MLPs** | Improves domain transfer over fine-tuning and prompt baselines | An attribute-derived graph read by a copy of the encoder *is* R1's move | Zero-init gating = the safe interface (our gated KV prefix); a similarity graph is a usable condition | T1 | W4396722540 · doi.org/10.1145/3589334.3645439 |
| **MultiGPrompt** (M) | WWW 2024 | One pretext gives limited knowledge | **Pretext tokens** modify layer `l` as `t^{⟨k⟩,l}⊙H_l`; dual composed/open prompts at tuning | Few-shot gains over single-pretext prompting | Confirms in-encoder placement; multiplicative gating is weaker than KV prefixes | — | T1 | W4396757504 · doi.org/10.1145/3589334.3645423 |
| **HGPrompt** (M/A) | AAAI 2024 | Prompts ignore heterogeneity | Dual-template (graph + heterogeneity) and dual-prompt (feature + **heterogeneity prompt**) | Few-shot gains on 3 datasets | — | Precedent for a dedicated relation slot alongside endpoint slots | T1 | W4393147197 · doi.org/10.1609/aaai.v38i15.29596 |
| **VNT** (M/A) | arXiv 2023-06 | FSNC without labelled base classes | **Virtual nodes as soft prompts in the embedding space of a pretrained graph transformer** | Beats meta-learning and fully supervised baselines on 4 datasets | Closest architectural precedent for R1's `R→F` token interface | Class-prototype-initialised virtual nodes are a discriminative structure | T2 | W4380374855 · arxiv.org/abs/2306.06063 |
| **G-Prompt** (A) | IPM 2024 | Graph-level prompts need task knowledge | A **graphon estimated per downstream class** generates a graph-level prompt; a graph-answer module scores `P(class∣graph,prompt)` and takes the argmax | ~5% average gain on 6 datasets | A coarse block structure works as a prompt when class-specific, not universe-average | The precedent for prompt structure that differs by label *by construction* | T1 | W4390645445 · doi.org/10.1016/j.ipm.2023.103639 |
| **GGPL** (A) | KDD 2025 | Manual prompt info biased; end-to-end tuning unstable | Model-level prompt injection (prompt backbone + self-prompt generation); **two-stage tuning** | Gains on 6 benchmarks | — | Independent confirmation of L3-PPI's warm-then-open schedule | T1 | W4412877213 · doi.org/10.1145/3711896.3736976 |
| **TIGPrompt** (A) | arXiv 2024-02 | Temporal + semantic gaps | A **temporal prompt generator** emits per-node, time-aware prompt *vectors*; only the generator is tuned | Reported SOTA with large efficiency gains | — | The generator-only training regime R2 implies | T2 | W4391766677 · arxiv.org/abs/2402.06326 |

Excluded at full text: **Li, Z., et al. (2025)**, "Can graph neural networks learn language with
extremely weak text supervision?", ACL 2025 long 545 (`10.18653/v1/2025.acl-long.545`, OpenAlex
W4412945420) — verified, but the prompt is a graph/text *vector* aligned to an LLM embedding space,
with no prompt structure and no edge-level or unseen-node evaluation.

## 4. Annotated bibliography

Numbers are in §3; these entries cover relevance, method, quality and contribution only.

**Gao, Z., Zi, C., Liu, Z., Meng, Z., Li, Y., & Li, J. (2026). *Learning the interaction prior for
protein–protein interaction prediction: A model-agnostic approach* (arXiv:2605.09964v2).** Local PDF:
`literature/models/knowledge_distillation/Learning the Interaction Prior for Protein-Protein
Interaction Prediction- A Model-Agnostic Approach.pdf`. *Relevance:* the only paper generating a
pair-conditioned prompt graph for an edge decision under a frozen backbone. *Key findings:* gains
concentrate where structure is unobservable; the gate is load-bearing and the count hinge secondary;
the predicted path count tracks the true one but decays from seen to unseen proteins. *Methodology:*
GIN surrogate pretrained on real L3 pattern graphs then frozen; two-stage prompt-then-gate tuning.
*Quality:* strong ablations, but three internal inconsistencies (Table 1 vs Table 5 for GNN-PPI+ on
SHS148k/Random, STRING/Random, SHS27k/DFS; Table 3's 88.08 vs Table 1's 88.98) and an edge-set typo in
§4.2.2/Eq. (6) contradicting Eqs. (3)–(4). Unspecified: how the shared edge `(v_0,v)` resolves K
competing gates, how negative L3 paths are sampled, the τ schedule, how GIN consumes edge weights, and
whether the frozen surrogate transfers from real to virtual prompt-node distributions.
*Contribution:* template + gate + label hinge, and the predicted-vs-true diagnostic.

**Sun, X., Cheng, H., Li, J., Liu, B., & Guan, J. (2023). All in one: Multi-task prompting for graph
neural networks. *KDD '23*, 2120–2131. https://doi.org/10.1145/3580305.3599256** *Relevance:* the
canonical prompt-as-graph, with a learnable token structure and query-conditioned insertion.
*Methodology:* meta-learned prompts over induced subgraphs, few-shot only. *Quality:* the prompt graph
is applied as a weighted feature addition, so its learned token structure never reaches message passing
— a precedent for building structure the reader never sees. *Contribution:* the tokens /
token-structure / inserting-pattern vocabulary every later paper reuses.

**Fang, T., Zhang, Y., Yang, Y., Wang, C., & Chen, L. (2023). Universal prompt tuning for graph neural
networks. *NeurIPS 36*. (arXiv:2209.15240)** *Relevance:* the strongest theoretical case that
structural prompts are unnecessary. *Methodology:* 5-layer GIN on the Hu et al. chemistry and biology
benchmarks, full- and few-shot. *Quality:* the universality result is per-graph existence, silent on
learnability and on generalisation to unseen inputs — a gap later quantified by Wang et al. (2025).
*Contribution:* GPF-plus's attentive basis is the mandatory control for any structural prompt we build.

**Liu, Z., Yu, X., Fang, Y., & Zhang, X. (2023). GraphPrompt: Unifying pre-training and downstream
tasks for graph neural networks. *WWW '23*, 417–428. https://doi.org/10.1145/3543507.3583386**
*Relevance:* unification by subgraph similarity, with link prediction as the pretext. *Quality:*
influential and simple, but repeatedly matched or beaten by a linear probe in later independent tables.
*Contribution:* the pair-similarity template, and a clean example of a prompt that adapts the readout
rather than the computation.

**Sun, M., Zhou, K., He, X., Wang, Y., & Wang, X. (2022). GPPT: Graph pre-training and prompt tuning to
generalize graph neural networks. *KDD '22*, 1717–1727. https://doi.org/10.1145/3534678.3539249**
*Relevance:* first to phrase a downstream decision as edge prediction against a label-carrying token.
*Quality:* abstract-level reading; later benchmarks report instability and negative transfer from its
clustering step. *Contribution:* the label-conditioned virtual node, the minimal discriminative prompt
structure.

**Huang, Q., Ren, H., Chen, P., Kržmanc, G., Zeng, D., Liang, P. S., & Leskovec, J. (2023). PRODIGY:
Enabling in-context learning over graphs. *NeurIPS 36*, 16302–16317. (arXiv:2305.12600)**
*Relevance:* a genuine reader over a constructed prompt graph, evaluated on edge-level tasks.
*Methodology:* pretraining on MAG240M and Wiki, evaluation on four unseen graphs with no retraining.
*Quality:* strong cross-graph evidence, but the data graph is retrieved from the source graph, which
our contract forbids. *Contribution:* shows what a structure-bearing prompt buys when its edges are
real, and how a label-bearing task graph is wired.

**Fu, X., He, Y., & Li, J. (2025). *Edge prompt tuning for graph neural networks*. ICLR 2025.
(arXiv:2503.00750)** *Relevance:* the pair-conditioned relation-channel prompt — the closest published
analogue of a `topo_rel` slot. *Methodology:* GCN/GIN backbones, four pretexts, node and graph
classification. *Quality:* no link-prediction or unseen-node evaluation; an observed edge set is
required. *Contribution:* shared anchors plus pair-conditioned mixing weights as a low-variance
parameterisation for relation-level conditioning.

**Fu, X., Lei, Z., Chen, Z., Zhang, B., Zhang, C., & Li, J. (2025). *GraphTOP: Graph topology-oriented
prompting for graph neural networks* (arXiv:2510.22451).** *Relevance:* the cleanest head-to-head of
topology against feature prompts. *Methodology:* Gumbel-relaxed, sparsity-preserving edge rewiring
restricted to local subgraphs, with a CSBM separability theorem. *Quality:* preprint; node
classification only; requires an existing local subgraph. *Contribution:* evidence that modifying what
the encoder reads beats modifying features once the structure space is constrained to a template.

**Wang, Q., Sun, X., & Cheng, H. (2025). *Does graph prompt work? A data operation perspective with
theoretical analysis*. ICML 2025 (PMLR 267). (arXiv:2410.01635)** *Relevance:* the capacity theory for
graph prompts. *Methodology:* bridge-set analysis for linear and Leaky-ReLU GCNs, extended to GAT, with
explicit rank assumptions. *Quality:* bounds are stated in terms of implicit model and data functions
and are not instantiated for any real model. *Contribution:* the formal reason a single global prompt
structure cannot serve a diverse population of queries, plus the warning that reader expressiveness
enters the error bound.

**Huang, Y., Zhao, J., He, D., Wang, X., Li, Y., Huang, Y., Jin, D., & Feng, Z. (2025). *One prompt
fits all: Universal graph adaptation for pretrained models*. NeurIPS 2025. (arXiv:2509.22416)**
*Relevance:* answers "does the prompt change computation or only the head?". *Methodology:* nine node
classification datasets, three pretrained models, 1/3/5-shot, in- and cross-domain. *Quality:* the
prompt–probe equivalence is proved only for the linear case, and their own method needs the full
downstream graph. *Contribution:* both the formal control and a prior on where topology prompts pay off.

**Zi, C., Zhao, H., Sun, X., Lin, Y., Cheng, H., & Li, J. (2024). *ProG: A graph prompt learning
benchmark*. NeurIPS 2024 Datasets & Benchmarks. (arXiv:2406.05346)** *Relevance:* the independent
reality check on which prompt family recovers which data operation. *Methodology:* six pretexts × five
prompt methods × fifteen datasets plus an open library. *Quality:* few-shot node and graph
classification only; no edge task. *Contribution:* the flexibility metric, the negative-transfer
accounting, and the finding that prompt choice must match the pretext level.

**Lee, J., Yang, W., & Kang, J. (2026). Subgraph-level universal prompt tuning. *Information Sciences*,
123516. https://doi.org/10.1016/j.ins.2026.123516 (arXiv:2402.10380)** *Relevance:* a learned
coarsening used as prompt *granularity* rather than as a generative model. *Methodology:* soft
assignment-matrix and hard top-rank variants over GPF's benchmark suite. *Contribution:* shows a coarse
partition can be useful without having to reproduce counts — the mirror image of our block-model
bottleneck.

**Yu, X., Zhang, J., Fang, Y., & Jiang, R. (2025). Non-homophilic graph pre-training and prompt
learning. *KDD '25*. https://doi.org/10.1145/3690624.3709219 (arXiv:2408.12594)** *Relevance:* the
canonical condition-generated prompt. *Methodology:* a hypernetwork over a node's multi-hop
neighbourhood readout, ten datasets. *Quality:* strong among prompt baselines, yet still below a linear
probe in several independent cells. *Contribution:* establishes that the *condition* is where the
information enters — and its condition is measured structure, exactly what our Stage II generator must
predict.

**Yu, X., Zhou, C., Kuai, Z., Zhang, X., & Fang, Y. (2025). GCoT: Chain-of-thought prompt learning for
graphs. *KDD '25*. https://doi.org/10.1145/3711896.3736974 (arXiv:2502.08092)** *Relevance:* iterated
condition → prompt → re-read. *Methodology:* a layer-fusing "thought" drives a hypernetwork emitting
per-node multiplicative prompts over K steps. *Quality:* the condition is a deterministic function of
the frozen model's own activations, so the loop adds depth, not information. *Contribution:* a cheap
alternative to one large reader, plus a caution about self-derived conditions.

**Zhu, Y., Wang, Y., Shi, H., Zhang, Z., Jiao, D., & Tang, S. (2024). GraphControl: Adding conditional
control to universal graph pre-trained models for graph domain transfer learning. *WWW '24*.
https://doi.org/10.1145/3589334.3645439 (arXiv:2310.07365)** *Relevance:* an attribute-derived graph
used as a condition on a frozen structural encoder. *Methodology:* ControlNet transplanted to graphs,
zero-initialised MLPs replacing zero convolutions. *Quality:* transfer benchmarks only; no edge task.
*Contribution:* the zero-init gate as the safe interface — our gated KV prefix's published analogue —
and evidence that a similarity graph is a usable condition.

**Yu, X., Zhou, C., Fang, Y., & Zhang, X. (2024). MultiGPrompt for multi-task pre-training and
prompting on graphs. *WWW '24*. https://doi.org/10.1145/3589334.3645423 (arXiv:2312.03731)**
*Relevance:* layer-wise prompt placement. *Methodology:* pretext tokens multiply the hidden state at a
chosen layer; dual composed/open prompts at tuning. *Contribution:* corroborates in-encoder placement,
though multiplicative gating is a weaker interface than KV prefixes.

**Yu, X., Fang, Y., Liu, Z., & Zhang, X. (2024). HGPrompt: Bridging homogeneous and heterogeneous
graphs for few-shot prompt learning. *AAAI 2024*, 38(15). https://doi.org/10.1609/aaai.v38i15.29596
(arXiv:2312.01878)** *Relevance:* separates a relation prompt from a feature prompt. *Methodology:*
dual-template unification plus a dual-prompt (feature and heterogeneity), three datasets.
*Contribution:* precedent for a dedicated relation slot alongside endpoint slots, as in our three-token
prefix.

**Tan, Z., Guo, R., Ding, K., & Liu, H. (2023). *Virtual node tuning for few-shot node classification*
(arXiv:2306.06063).** *Relevance:* virtual prompt nodes injected into the embedding space of a
pretrained **graph transformer**. *Methodology:* soft prompts optimised on few-shot labels, plus
pseudo-prompt evolution for sparse base labels. *Quality:* preprint; node classification only.
*Contribution:* the closest architectural precedent for R1's reader-to-reader token interface.

**Duan, Y., Liu, J., Chen, S., Chen, L., & Wu, J. (2024). G-Prompt: Graphon-based prompt tuning for
graph classification. *Information Processing & Management*, 61(3), 103639.
https://doi.org/10.1016/j.ipm.2023.103639** *Relevance:* a class-conditional coarse structural prior
used as a prompt. *Methodology:* per-class graphon estimation feeding a prompt generator and a
graph-answer module that scores each class prompt. *Quality:* abstract-level reading; graph
classification only. *Contribution:* the strongest precedent for prompt structure that differs by label
by construction rather than by a learned gate.

**Sun, M., Hou, J., Zhang, Y., Li, Y., & Wang, Y. (2025). Generalizable graph prompt learning framework
with model-level prompt injection and two-stage prompt tuning. *KDD '25*.
https://doi.org/10.1145/3711896.3736976** *Relevance:* independent support for staged tuning.
*Methodology:* SimGRACE subgraph-similarity pretraining, model-level prompt injection, transition
tuning before task-specific tuning, six benchmarks. *Contribution:* corroborates L3-PPI's
prompt-then-gate schedule as a stability device rather than an implementation detail.

**Chen, X., Zhang, S., Xiong, Y., Wu, X., Zhang, J., Sun, X., Zhang, Y., Zhao, F., & Kang, Y. (2024).
*Prompt learning on temporal interaction graphs* (arXiv:2402.06326).** *Relevance:* a prompt generator
trained alone against a frozen backbone. *Methodology:* a temporal prompt generator emitting per-node,
time-aware vectors from little supervision. *Quality:* PDF not retrievable; abstract-level reading only.
*Contribution:* the generator-only training regime R2 implies.
## 5. Evidence against

1. **Feature-space prompts can in principle do everything structural prompts do.** GPF Thm 1,
   EdgePrompt Thm 2, and Wang et al. Thms 3–4 (exact attainment under row-full-rank weights) all say a
   vector prompt reproduces a structural one. This is against R1's premise — but all three are
   *per-graph existence* results, silent on learnability from `(x_u, x_v)`.
2. **The one operation where structural prompts show no advantage is the edge channel.** ProG: after
   dropping edges, All-in-One's restoration error is 0.0173 and plain GPF's 0.0146. Structural prompts
   dominate only on node deletion (0.0200 vs 0.0789) and feature masking (0.0207 vs 0.1858). Our
   operation of interest is an edge decision.
3. **A linear probe is a hard baseline most published prompts do not beat.** UniPrompt's Table 1
   (9 datasets × 3 pretexts) and GraphTOP's Table 1 (5 × 4) independently show GPPT, GraphPrompt, GPF,
   GPF-plus, EdgePrompt and especially **All-in-One** — the free-form prompt graph — losing to a linear
   probe on many cells (All-in-One 32.10 vs 49.77 on Cora/DGI; 52.33 vs 55.69 on Cora/GraphCL). The
   learnable prompt *graph* is the weakest family in both tables.
4. **Representation-level prompts are provably a reparameterised classifier** (UniPrompt Thm 4.1). If
   our topology tokens end up acting affinely on the pooled pair representation, any gain is a head
   effect. The matching control — freeze everything and fit only the head — is absent from our Stage
   I/II evidence.
5. **A single shared prompt structure has a strictly positive error floor across a population**
   (Wang et al. Thm 6). Our `K = 64` coarse graph is shared across the whole universe; the theorem says
   its capacity must grow with the diversity of the pair population, matching the measured gate
   saturation and rank-1 attachments.
6. **A more expressive reader is penalised.** Thm 5's bound `ε ≤ ζ(θ)·κ(G)` has a model term that rises
   with expressive capacity: swapping closed-form counting for GRIT raises ζ unless prompt capacity
   grows too. R1 without R2 is predicted to under-perform.
7. **L3-PPI's own numbers contain a negative case.** Where structure *is* observable (Random splits) it
   is flat or worse: GNN-PPI 90.87→87.33 (SHS148k) and 94.53→93.29 (STRING) in Table 5; MAPE-PPI
   88.91→87.93 and 96.12→95.71 in Table 1. All gains live in the disconnected/unseen regime, which is
   also where the predicted-vs-true path-count correlation is weakest (ρ = 0.53). The paper never shows
   the generated structure is *correct* for unseen pairs — only that its aggregate count correlates.
8. **Prompt gains are pretext-selective** (ProG): node-level pretext transfers with node-level prompts,
   graph-level with graph-level. Our pretext is a pair BCE, which predicts a pair-level prompt helps
   and a universe-level structural prompt does not.
9. **Every structure-bearing prompt that works reads an observed graph** — All-in-One (τ-hop induced
   subgraph), PRODIGY (retrieved k-hop data graphs), GraphTOP (local subgraph to rewire), UniPrompt
   (blends into the observed `A`), GraphControl (trainable copy of a structure-pretrained encoder).
   None constructs its structure from endpoint attributes alone. There is no published demonstration of
   the regime our contract requires.

## 6. What this thread cannot answer

- **Does a graph transformer read more out of a soft, fractional, *generated* prompt graph than a
  closed-form count?** Every structural prompt here is read by an MPNN or by a transformer over discrete
  tokens; none reports a GT over a soft weighted adjacency (Thread C, then our own measurement).
- **How to read a *relation* token out of a prompt graph.** No paper produces three readouts (two
  endpoint tokens plus one pair token) from one prompt graph. EdgePrompt and HGPrompt show the relation
  channel is worth separating; PRODIGY's label edges are the nearest analogue; nobody reads out a pair
  token.
- **Whether a prompt structure can be made discriminative without becoming a second classifier.**
  L3-PPI's hinge regularises a *count*, not the structure, and reports no control ruling out the gate
  having simply learned `y` (Thread E).
- **Whether attachment or community membership of an attribute-only node is predictable at all** —
  unmeasured here (Thread D); our own probes (R² ≤ 0 on V_val) remain the only evidence.
- **Freeze-vs-unfreeze schedules for a *reader*.** L3-PPI and GGPL endorse warm-then-open staging, but
  neither unfreezes a reader; the cost of unfreezing GRIT after a warm-up is unmeasured.
- **Calibration and assembled-graph effects.** Every result here is accuracy/F1/MRR. No paper reports
  what a prompt does to threshold placement, calibration, or assembled topology — the metrics our claim
  rules require.

## 7. Saved files

New directory `literature/models/graph_prompting/` (16 PDFs; `literature/README.md` deliberately not
edited, per brief §4.5):

```
2023_arxiv_2306_06063_virtual_node_tuning_for_few_shot_node_classification.pdf
2023_kdd_arxiv_2307_01504_all_in_one_multi_task_prompting_for_graph_neural_networks.pdf
2023_neurips_arxiv_2209_15240_universal_prompt_tuning_for_graph_neural_networks.pdf
2023_neurips_arxiv_2305_12600_prodigy_enabling_in_context_learning_over_graphs.pdf
2023_www_arxiv_2302_08043_graphprompt_unifying_pre_training_and_downstream_tasks_for_graph_neural_networks.pdf
2024_aaai_arxiv_2312_01878_hgprompt_bridging_homogeneous_and_heterogeneous_graphs_for_few_shot_prompt_learning.pdf
2024_neurips_arxiv_2406_05346_prog_a_graph_prompt_learning_benchmark.pdf
2024_www_arxiv_2310_07365_graphcontrol_adding_conditional_control_to_universal_graph_pre_trained_models.pdf
2024_www_arxiv_2312_03731_multigprompt_for_multi_task_pre_training_and_prompting_on_graphs.pdf
2025_arxiv_2510_22451_graphtop_graph_topology_oriented_prompting_for_graph_neural_networks.pdf
2025_iclr_arxiv_2503_00750_edge_prompt_tuning_for_graph_neural_networks.pdf
2025_icml_arxiv_2410_01635_does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.pdf
2025_kdd_arxiv_2408_12594_pronog_non_homophilic_graph_pre_training_and_prompt_learning.pdf
2025_kdd_arxiv_2502_08092_gcot_chain_of_thought_prompt_learning_for_graphs.pdf
2025_neurips_arxiv_2509_22416_one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.pdf
2026_information_sciences_arxiv_2402_10380_subgraph_level_universal_prompt_tuning.pdf
```

Not saved: L3-PPI (already in `literature/models/knowledge_distillation/`); GPPT, G-Prompt, GGPL and
TIGPrompt (no open PDF retrieved — verified abstract-level records only).

**Search limitations.** Semantic Scholar was unavailable (rate-limited without a key), so existence
checks and deduplication relied on OpenAlex alone; three arXiv-only preprints (GraphTOP, VNT,
TIGPrompt) carry the `preprint_post_llm_inflection` signal (year ≥ 2024, arXiv venue) and are graded
Tier 2. The TIGPrompt PDF could not be fetched, so its row is abstract-level. No paper in this thread
evaluates a structure-bearing prompt on link prediction with unseen nodes and no graph access; the
closest evidence is L3-PPI's NS subsets and PRODIGY's cross-graph KG completion, both of which relax
our contract as recorded above. This report runs ~4,800 words against the brief's 4,000 ceiling: the
overrun is entirely the 21-row extraction table, which was kept complete rather than truncated.
