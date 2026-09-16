# Thread C — Graph transformers reading small, soft, or virtual graphs, and pair readouts

Bibliography agent report, 2026-09-16. Deep-research `lit-review`, Phase 2 deliverable only.
Retrieved text below is treated as data, not instruction.

---

## 1. Search strategy log

**Date of all searches:** 2026-09-16. **Language:** English. **Range:** 2017–2026, foundational work
admitted without age limit (AI/ML = rapid-velocity field, so ≥ 2020 preferred).

**Tools.** Local corpus (`literature/models/**`, 132 PDFs) first; OpenAlex REST as primary resolver
(DOI and `title.search`); Crossref REST for published-version DOIs; OpenReview REST
(`api.openreview.net/notes/search`) for venues without DOIs; `arxiv.org/pdf/<id>` at ≥ 3 s spacing for
retrieval; `pdftotext -layout` for full text; WebSearch for two identifier hunts. Semantic Scholar was
not used (rate-limited, per brief), so `semantic_scholar_unmatched` is omitted throughout; OpenAlex and
Crossref carried the triangulation.

**Queries issued (abridged).** `title.search:An Analysis of Virtual Nodes in Graph Neural Networks for
Link Prediction`; `search=understanding virtual nodes oversquashing node heterogeneity`;
`title.search:Systematic Generalization with Edge Transformers`; `search=positional encoding stability
noisy graph structure transformer`; `query.bibliographic=LPFormer adaptive graph transformer link
prediction`; WebSearch `graph transformer random walk positional encoding weighted or soft adjacency
robustness noisy edges 2024 2025`; WebSearch `"Learning Latent Graph Structures and their Uncertainty"
… ICML 2025`; 31 direct DOI lookups `10.48550/arxiv.<id>`; six Crossref DOI checks.

| PRISMA stage | n |
|---|---|
| Identified — brief's Thread-C seed list | 20 |
| Identified — local corpus (GT / latent-graph / LP-structural folders) | 11 |
| Identified — snowballing (GRIT, Southern, Cai, LPFormer, Müller reference lists) + keyword search | 9 |
| Total identified | 40 |
| Duplicates removed (arXiv vs proceedings of same work) | 3 |
| Screened on title/abstract | 37 |
| Excluded at title/abstract | 4 |
| Full text assessed (targeted extraction of method, theory, results, ablations) | 33 |
| Excluded at full text | 2 |
| **Included** | **31** |

Full-text exclusions: **GAPE** (arXiv 2212.06898, TMLR 2023) — "weighted" refers to automaton
transition weights, not fractional edges; **PR-MPNN** — superseded for our purposes by its virtual-node
successor (Qian et al., 2024), which is included. Title/abstract exclusions: three rewiring surveys and
one signed-bipartite LP paper, all tangential.

**Coverage distribution advisory.**
```
DISTRIBUTIONAL_SKEW_ADVISORY
- Dimension: methodological distribution
- Concentration: computational/architecture papers with benchmark evaluation = 31/31 (100%)
- Advisory: coverage signal, not a defect; the RQ is architectural by construction.
- Search response: no expansion.
DISTRIBUTIONAL_SKEW_ADVISORY
- Dimension: venue tier distribution
- Concentration: NeurIPS/ICML/ICLR/ACL/EMNLP/KDD/TMLR/TPAMI = 30/31 known (97%)
- Advisory: the one exception (LoG 2022 extended abstract) is graded Tier 2 below.
- Search response: no expansion.
```

**Search limitations.** (a) OpenReview PDFs sit behind a browser check; Hwang et al. (2022) full text
could not be retrieved by `curl` or WebFetch, so that entry is annotated from its verified OpenReview
record and abstract only, and is marked as such. (b) No source was found that measures RRWP on
*fractional* adjacencies; the closest evidence is indirect (§5.4). (c) S2 deduplication skipped by
instruction.

---

## 2. Inclusion / exclusion criteria as applied

**Included** if the paper (i) specifies a mechanism bearing on one of the four extraction questions —
what a transformer computes from a small/soft graph; behaviour of random-walk or spectral encodings
under perturbation; virtual-node theory or practice; multi-token / pair readouts; prefix or
freeze-schedule evidence — **and** (ii) reports a theorem or a measured number, **and** (iii) resolves
to a real record with matching title, authors and year.

**Excluded** if it only surveys without new evidence, uses "graph transformer" for an unrelated
architecture, contributes only a leaderboard entry, or could not be confirmed. Nothing is carried as
"uncertain": two candidates OpenAlex could not match were resolved through OpenReview and Crossref.

**Grading.** Tier 1 = peer-reviewed proceedings/journal; Tier 2 = preprint or non-archival extended
abstract; Tier 3 = grey. Under the field-adjusted rubric (Technology row), the gold standard is a
controlled architectural ablation with seeds at a fixed parameter budget; papers meeting it earn
Grade A on Evidence Level.

---

## 3. Mechanism extraction table

R1 = revision 1 (GT reader replaces closed-form counts); R2 = revision 2 (prior-knowledge prompt
structure). All OpenAlex IDs were confirmed to match title, authors and year.

| Key | Venue/Yr | WHY | HOW | WHAT | R1 / R2 | Tier | Verification |
|---|---|---|---|---|---|---|---|
| GRIT (Ma+) | ICML 2023 | GT bias without message passing | RRWP `[I,M,…,M^{K−1}]`, `M=D⁻¹A`, in the **edge** channel; attention updates node *and* pair states; degree scaler; BatchNorm (LayerNorm cancels degree, Prop. 3.3) | Prop. 3.1: RRWP+MLP ≈ SPD_{K−1} and any `Σθ_k(D⁻¹A)^k` (sum/mean agg, truncated PPR, heat kernel). Prop. 3.2: GD-WL+RRWP ⊐ GD-WL+SPD. Synthetic k-hop propagation MAE .001 vs .043 (Graphormer), .083 (Tr.+RWSE). ZINC ablation RRWP→RWSE 0.059→0.081 | R1: reader *can* learn count-like propagations, but Prop. 3.1 quantifies over `A∈{0,1}^{n×n}` / R2: K is itself a template choice | 1 | OpenAlex W4378766933 · doi:10.48550/arxiv.2305.17589 |
| Graphormer (Ying+) | NeurIPS 2021 | Transformers underperform on graphs | SPD spatial-encoding attention bias; degree centrality encoding; `[VNode]` linked to all nodes with its **own** learnable spatial scalar, used as graph readout | Fact 1: a layer represents GIN/GCN/SAGE aggregate+combine. Fact 2: a layer represents MEAN READOUT — self-attention already simulates a virtual node | R1: a graph token is standard but must be typed apart from real nodes / R2: degree injected, not learned | 1 | OpenAlex W3169622372 · doi:10.48550/arxiv.2106.05234 |
| GraphGPS (Rampášek+) | NeurIPS 2022 | Modular GT recipe | MPNN layer ∥ global attention per block; PE/SE taxonomy | Removing MPNN: ZINC 0.070→0.217, PCQM4Mv2 0.1159→0.3294. Removing global attention: ZINC unchanged. RWSE > LapPE on molecules (0.070 vs 0.116) | R1: on small graphs the local structural computation carries the gain / R2: walk priors beat spectral on molecule-like graphs | 1 | OpenAlex W4281706128 · doi:10.48550/arxiv.2205.12454 |
| TokenGT (Kim+) | NeurIPS 2022 | Can a *pure* transformer learn graphs? | Nodes **and edges** as tokens; orthonormal node identifiers + trainable type identifiers; `[graph]` token | Thm. 1: with those identifiers, `bell(2k)` heads approximate any order-k equivariant linear layer (≥ 2-IGN) | R1: edges-as-tokens is an alternative to RRWP, at `O((n+m)²)` / R2: — | 1 | OpenAlex W4284896159 · doi:10.48550/arxiv.2207.02505 |
| Exphormer (Shirzad+) | ICML 2023 | GT quadratic cost | Sparse attention = expander + local edges + a **constant number of virtual global nodes** | Virtual nodes act as a global "storage sink"; sparse scheme retains universal approximation | R1: a handful of global nodes suffices to route global information / R2: — | 1 | OpenAlex W4324109129 · doi:10.48550/arxiv.2303.06147 |
| Müller+ | TMLR 2024 | What do GTs actually recover? | Taxonomy + probes: Edges, Triangles (small/large), CSL | Plain Transformer: Edges 55.8, Triangles 12.1, CSL 10.0. +RWSE: 97.1 / **99.4** / 100. +LapPE: Tri-small 78.3, **Tri-large 10.6**. All models ≤ 54.8 on larger graphs. "Without expressive SE/PE, GTs … equal DeepSets" | R1: a GT with a walk SE counts triangles **at training size** and collapses off-size / R2: the encoding, not the transformer, is the prior | 1 | OpenAlex W4320537170 · doi:10.48550/arxiv.2302.04181 |
| Chen+ | NeurIPS 2020 | Counting as expressivity | Induced vs non-induced counting for MPNN / 2-WL / 2-IGN / k-WL | Thm. 3.3: 2-WL (hence MPNN, 2-IGN) **cannot** induced-subgraph-count any connected pattern ≥ 3 nodes; star (degree-like) patterns *are* countable | R1: triangles/clustering must be supplied or encoded, not hoped for / R2: closed motifs are genuinely hard prior knowledge | 1 | OpenAlex W3005043394 · doi:10.48550/arxiv.2002.04025 |
| LSPE (Dwivedi+) | ICLR 2022 | Decouple position from structure | RWPE = diag of `(AD⁻¹)^k`; learned positional channel updated with features | RWPE unique per node iff k-hop neighbourhoods unique; beats LapPE, no sign ambiguity | R1: node-level sibling of RRWP; polynomial in `A`, hence defined for weighted `A` / R2: — | 1 | OpenAlex W3205705101 · doi:10.48550/arxiv.2110.07875 |
| SPE (Huang+) | ICLR 2024 | Eigenvector PEs are unstable | Soft, eigenvalue-dependent partition of eigenspaces | Thm. 3.1: Hölder-stable, bound depends on the **eigengap**; domain-generalisation gap bounded by the Hölder constant; explicit stability↔expressivity trade-off | R1: spectral PEs are the wrong family on a *predicted* graph; continuous polynomial encodings are not / R2: — | 1 | OpenAlex W4387389962 · doi:10.48550/arxiv.2310.02579 |
| Black+ | ICML 2024 | APE vs RPE | Interchange constructions between node-level and pair-level encodings; new WL variant | Equivalent distinguishing power on featureless graphs; **with node features, RPEs may be strictly advantageous** | R1: coarse nodes carry features, so the pair channel is the right home for `topo_rel` / R2: — | 1 | OpenAlex W4392120532 · doi:10.48550/arxiv.2402.14202 |
| Gilmer+ | ICML 2017 | Long-range info in MPNNs | "Master node" on a special edge type, own dimension and GRU; `O(\|E\|d²+n d²_master)` | GG-NN 3.47 → +virtual edge 2.90 → **+master node 2.62** → +set2set 2.57 (avg error ratio) | R1: origin of the coarse node; a *typed* virtual edge matters / R2: — | 1 | OpenAlex W2606780347 · doi:10.48550/arxiv.1704.01212 |
| Hwang+ | LoG 2022 (Oral, ext. abstract) | VNs for **link prediction** | **Multiple** VNs per graph; graph **clustering** decides attachments; theoretical analysis; OGB | Abstract: VNs "may yield rather stable performance increases and sometimes considerably boost performance"; clustering cuts under-reaching while sparing nodes from unrelated information | R1: K clustered VNs is published LP practice / R2: cluster-derived attachment is the prior-knowledge form of our learned attachment | 2 | OpenReview note `dI6KBKNRp7`, venueid `logconference.io/LOG/2022/Conference` · https://openreview.net/forum?id=dI6KBKNRp7 |
| Cai+ | ICML 2023 | Is a GT needed at all? | MPNN+VN as a multi-relational graph; constructions via Performer and DeepSets | Thm. 4.1: `O(1)` width/depth approximates a linear-attention layer; Thm. 5.5: `O(n^d)` width for full self-attention. PCQM4Mv2 val MAE **GPS 0.0938 vs MPNN+VN+PE 0.0942** (small), 0.0858 vs 0.0867 (medium); GatedGCN+RWSE+VN beats all transformers on Peptides-func | R1: **against** — a VN-style global aggregator already matches a GT at equal budget / R2: — | 1 | OpenAlex W4318719654 · doi:10.48550/arxiv.2301.11956 |
| Southern+ | ICLR 2025 | When/why do VNs help? | Commute-time and Jacobian sensitivity analysis; `VN_G` = local update → global VN update → combine | **Prop. 4.1: a standard VN's `∂h_i/∂h_k` is independent of `k` beyond 2 hops — uniform, node-independent signal.** Where attention-column std is small (peptides, MNIST), MPNN+VN matches GPS; where large (CIFAR10), GPS wins. `VN_G` beats VN on every task, +8.25 % on CIFAR10 | R1: explains our gate collapse — homogeneity is the VN's *default*, not a bug / R2: the fix is an ordering change, cheap, no transformer needed | 1 | OpenAlex W4398795635 · doi:10.48550/arxiv.2405.13526 |
| IPR-MPNN (Qian+) | NeurIPS 2024 | Rewiring without quadratic cost | Upstream MPNN scores node→virtual-node edges; **differentiable exact-k-subset sampling**; message passing original→virtual, virtual↔virtual, original→original | Beyond 1-WL (Thm. 4.1). Peptides-func **0.7210** vs GRIT 0.6988, GPS 0.6535, Exphormer 0.6527; Peptides-struct 0.2422 (best); **PCQM-Contact link prediction 0.3670 vs Exphormer 0.3637, GPS 0.3498**; linear runtime, far less memory than GPS | R1: **strongest evidence against** — learned virtual nodes beat every GT, incl. GRIT, on the nearest analogue / R2: k-subset sampling replaces rank-1 soft attention | 1 | OpenAlex W4399115081 · doi:10.48550/arxiv.2405.17311 |
| Set Transformer (Lee+) | ICML 2019 | Attention-based set pooling | `PMA_k(Z)=MAB(S,rFF(Z))`, k learnable seeds; **SAB after PMA** to model interaction among the k outputs | Props. 1–2: permutation-invariant, universal approximator; multi-seed PMA is prescribed exactly for "k correlated outputs" | R1: three seeds + post-PMA SAB gives `topo_u/topo_v/topo_rel` — but seeds are **query-independent** / R2: — | 1 | OpenAlex W2894384847 · doi:10.48550/arxiv.1810.00825 |
| Labeling trick (Zhang+) | NeurIPS 2021 | Can pooled node states represent a *pair*? | Label the target set in `A` **before** the encoder runs, then aggregate | **Prop. 1: aggregating independently computed node representations is never a structural link representation, however expressive the encoder.** Thm. 1: labelled-graph aggregation *is*. Thm. 2: ω(n²) link pairs separated only with the trick | R1: the three tokens must be read from a **pair-labelled** virtual graph / R2: labelling is the minimal prior | 1 | OpenAlex W3210188290 · doi:10.48550/arxiv.2010.16103 |
| Distance Encoding (Li+) | NeurIPS 2020 | Structural representation of a node *set* | DE(u\|S) = landing probabilities of walks between u and target set S (generalised PageRank); as features (DE-GNN) **or** as an aggregation controller (DEA-GNN) | Provably separates node sets 1-WL GNNs cannot; defined for any set size p (p = 2 for links) | R1: exact recipe for pair-conditioning RRWP on the virtual graph / R2: DE is a template family parameterised by walk length | 1 | OpenAlex W3094003325 · doi:10.48550/arxiv.2009.00142 |
| LPFormer (Shomer+) | KDD 2024 | One-size-fits-all pairwise encodings | `s(a,b)=Σ_{u∈N̂(a,b)} w(a,b,u)h_u`; GATv2 attention conditioned on `(a,b)`; PPR relative PE `MLP(ppr(a,u),ppr(b,u))`; PPR thresholds per node type; score `σ(MLP(h_a⊙h_b ∥ s(a,b) ∥ \|N̂_CN\| ∥ \|N̂_1\| ∥ \|N̂_{>1}\|))` | SOTA on 5/6 datasets. **ogbl-ppa ablation: full 63.32; w/o counts 44.37 (−18.95); w/o learnable attention 62.77 (−0.55); w/o RPE 61.20.** Citeseer w/o counts 54.39 vs 65.42 | R1: **counts and a learned pair token are complementary; on protein-association data counts carry ~35× more** / R2: CN / 1-hop / >1-hop are the load-bearing templates | 1 | Crossref doi:10.1145/3637528.3672025 · OpenAlex W4387797190 |
| Edge Transformer (Bergen+) | NeurIPS 2021 | Relations are not node properties | A vector state for **every pair** `x_ij`; **triangular attention** updates `x_ij` from all `(i,l),(l,j)` | Stronger systematic generalisation than GAT/Transformer on CLUTRR relational reasoning and compositional parsing | R1: native relation token; `O(n³)`/layer — fine at n≈34, heavy at n≈300 / R2: relation composition is itself the L3-style prior | 1 | OpenAlex W3212682848 · NeurIPS 2021 proceedings PDF `0a4dc6dae338c9cb08947c07581f77a2` |
| DGM (Kazi+) | IEEE TPAMI 2023 | Learn graph and task jointly | Continuous (cDGM) and discrete (dDGM, Gumbel-Top-k) latent graph modules | kNN sampling is non-differentiable, so the task loss gives the graph **no gradient**; they add a *graph loss* rewarding edges whose endpoints are classified correctly | R1: a soft graph fed to a reader does not train itself / R2: an explicit structure objective is required | 1 | Crossref doi:10.1109/TPAMI.2022.3170249 · OpenAlex W6929103953 |
| NodeFormer (Wu+) | NeurIPS 2022 | All-pair message passing at scale | Kernelised Gumbel-Softmax, `O(N)`/layer; relational bias; **edge-level regularisation loss** | Thms. 1–2 give approximation/convergence; explicit remark: **large temperature τ degrades the operator to mean pooling** | R1: soft global attention collapses to a mean without temperature and regularisation control — our gate's failure mode / R2: — | 1 | OpenAlex W4380993339 · doi:10.48550/arxiv.2306.08385 |
| LGI-LS (Lu+) | NeurIPS 2023 | Supervision starvation in latent graph inference | k-hop *starved* nodes (weights that never enter the loss); CUR-style pivot selection; restore destroyed affinities | Starved weights "cannot be semantically optimal, resulting in poor generalization"; removing them improves every LGI method, **+6.12 % on Pubmed at a 0.3 % labeling rate** | R1: names why our coarse-node parameters are gradient-starved / R2: direct structural supervision is the published remedy | 1 | OpenAlex W4389974167 · doi:10.48550/arxiv.2310.04314 |
| Manenti+ | ICML 2025 | Is the learned graph the *right* graph? | Losses on stochastic model outputs rather than point predictions | **Minimising a point-prediction loss does not guarantee the latent graph distribution is learned**; suitable stochastic losses solve recovery and prediction together | R1: task BCE will not identify the virtual graph even as accuracy rises / R2: motivates the attachment loss and a calibrated objective | 1 | OpenAlex W4399252636 · doi:10.48550/arxiv.2405.19933 · https://proceedings.mlr.press/v267/manenti25a.html |
| Prefix-Tuning (Li & Liang) | ACL 2021 | Adapt a frozen LM cheaply | Trainable continuous prefix on keys/values at **every** layer; frozen backbone | 0.1 % params ≈ full FT on full data; **+2.9 BLEU over FT in low-data**; better extrapolation to unseen topics; **direct optimisation of `P_θ` "leads to unstable optimization and a slight drop"**, fixed by `P_θ[i,:]=MLP_θ(P'_θ[i,:])`, dropped after training | R1: our gated KV prefix should be MLP-reparameterised, not optimised directly / R2: — | 1 | Crossref doi:10.18653/v1/2021.acl-long.353 · arXiv 2101.00190 |
| P-Tuning (Liu+) | ACL 2022 (short) | Shallow prompts fail on hard tasks and small models | **Deep** prompts at every layer, 0.1–3 % trainable params | Matches fine-tuning from 300 M to 10 B on NLU **and** hard sequence labeling, where input-only prompting fails | R1: supports prefixes at all nine cross-attention sites / R2: — | 1 | Crossref doi:10.18653/v1/2022.acl-short.8 (published title "P-Tuning: …Across Scales and Tasks"); arXiv 2110.07602 is titled "P-Tuning v2: …Universally…" — distinct records, not merged |
| Lester+ | EMNLP 2021 | How far can a frozen model be prompted? | Input-only soft prompt, frozen backbone | Gap to model tuning closes only at ~11 B; **"large gaps" at smaller sizes**; prompt tuning gives **stronger zero-shot performance than model tuning under domain shift** (MRQA, e.g. TextbookQA) | R1: **against** freezing at our scale, **for** freezing under distribution shift / R2: — | 1 | Crossref doi:10.18653/v1/2021.emnlp-main.243 · arXiv 2104.08691 |
| Flamingo (Alayrac+) | NeurIPS 2022 | Condition a frozen LM on a new modality | `GATED XATTN-DENSE` blocks inserted in a frozen LM, each scaled by `tanh(α)` with **α initialised at 0** | Removing the 0-init gate: **−4.2 % and training instabilities**. Fine-tuning the pretrained LM: **−8.0 %**; from scratch −12.9 %; co-training on the original corpus while unfrozen still worse than frozen. Every-layer insertion > every-4th by 1.9 % | R1: direct support for the zero-init gated prefix and a frozen trunk / R2: — | 1 | OpenAlex W4225323055 · doi:10.48550/arxiv.2204.14198 |
| ULMFiT (Howard & Ruder) | ACL 2018 | Catastrophic forgetting when fine-tuning | **Gradual unfreezing** (top layer first, one per epoch) + discriminative LRs + slanted triangular LRs | Val error IMDb/TREC-6/AG: Full 6.87/6.86/5.81; **"Last" (frozen trunk) 6.49/16.09/8.38, never reaches zero training error**; gradual unfreezing alone 6.37/6.86/5.81; full recipe 5.00/5.69/5.38 | R1: unfreezing needs a schedule *and* layer-wise LRs; a frozen trunk can underfit badly / R2: — | 1 | Crossref doi:10.18653/v1/P18-1031 · arXiv 1801.06146 |
| LP-FT (Kumar+) | ICLR 2022 | Fine-tuning distorts good features | Theory on two-layer linear nets + 10 shift datasets; linear-probe-then-fine-tune | FT vs LP: ID 85 % vs 83 %, **OOD 59 % vs 66 %**; a randomly initialised head drives the distortion; **LP-FT is +1 % ID and +10 % OOD over full FT** | R1: decisive argument for *warm-up then unfreeze* over either extreme / R2: — | 1 | OpenAlex W4221149036 · doi:10.48550/arxiv.2202.10054 |
| Surgical FT (Lee+) | ICLR 2023 | Which layers to tune? | Tune one contiguous block, freeze the rest; nine shifts + auto-selection criterion | Beats full FT by ~3 %; the best block depends on shift type — first block for input-level (82.8, +2.9), middle/late for feature-level (81.2, +2.1), last layer for output-level (86.2, +4.0) | R1: unfreeze a *chosen* block where the topology signal enters / R2: — | 1 | OpenAlex W4307079438 · doi:10.48550/arxiv.2210.11466 |

---

## 4. Annotated bibliography

### 4.1 What a graph transformer computes from a small graph

**Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania, P. K., Coates, M., Torr, P. H. S., & Lim, S.-N. (2023). Graph inductive biases in transformers without message passing. *Proceedings of the 40th International Conference on Machine Learning* (PMLR 202). https://arxiv.org/abs/2305.17589** — local: `literature/models/graph_structure_learning/latent_structure_learning/2023_icml_arxiv_2305_17589_…pdf`
*Relevance*: the encoder the project vendors, so it bounds what our reader can compute. *Key findings*: RRWP+MLP approximates truncated shortest-path distance and any polynomial propagation; strictly stronger than SPD under GD-WL; learns k-hop propagation matrices to MAE 0.001 where Graphormer reaches 0.043. *Methodology*: three propositions with proofs, ZINC/LRGB benchmarks, a synthetic attention-matching task, 4 seeds. *Quality*: Grade A, but the propositions quantify over binary adjacency, which our soft graph violates. *Contribution*: structure lives in the pair channel, not the node channel.

**Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., & Liu, T.-Y. (2021). Do transformers really perform bad for graph representation? *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2106.05234**
*Relevance*: the `[VNode]` readout token and typed virtual edges. *Key findings*: a layer can represent GIN/GCN/SAGE aggregation and a MEAN READOUT; self-attention already simulates a virtual node; `[VNode]` spatial encodings are reset to a distinct learnable scalar. *Methodology*: constructive expressivity facts plus OGB-LSC benchmarks. *Quality*: Grade A. *Contribution*: a readout token must be typed apart from real nodes, which matters if coarse nodes and a readout token share one graph.

**Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a general, powerful, scalable graph transformer. *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2205.12454**
*Relevance*: quantifies how much global attention adds over local structure. *Key findings*: removing the MPNN is catastrophic (ZINC 0.070→0.217); removing global attention costs nothing on ZINC; RWSE beats LapPE on molecules. *Methodology*: four-dataset ablation grid at fixed budgets, multiple seeds. *Quality*: Grade A. *Contribution*: on small graphs the transformer is the cheap part of the gain.

**Müller, L., Galkin, M., Rampášek, L., & Morris, C. (2024). Attending to graph transformers. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2302.04181**
*Relevance*: the only located source that directly probes what GTs recover. *Key findings*: without structural encodings a GT equals DeepSets and cannot detect edges (55.8 %); with RWSE it counts triangles at 99.4 %; every model collapses to ≤ 54.8 % on larger graphs. *Methodology*: controlled probes at matched budgets, five seeds. *Quality*: Grade A; probe graphs are molecules, not coarsened block graphs. *Contribution*: size generalisation is the sharpest risk for a virtual graph of varying size.

**Chen, Z., Chen, L., Villar, S., & Bruna, J. (2020). Can graph neural networks count substructures? *Advances in Neural Information Processing Systems 33*. https://arxiv.org/abs/2002.04025**
*Relevance*: the converse question — when are counts exactly what you want? *Key findings*: MPNNs/2-WL/2-IGNs cannot induced-subgraph-count any connected pattern of ≥ 3 nodes; star (degree-like) patterns are countable. *Methodology*: constructive impossibility proofs plus synthetic counting experiments. *Quality*: Grade A; covers message passing, not GTs with strong relative encodings. *Contribution*: clustering and triangle coordinates are provably hard to learn and cheap to supply.

**Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2022). Graph neural networks with learnable structural and positional representations. *International Conference on Learning Representations*. https://arxiv.org/abs/2110.07875**
*Relevance*: defines RWPE, the node-level sibling of RRWP. *Key findings*: RWPE is the return-probability profile of k-step walks, unique per node when k-hop neighbourhoods are unique, and beats LapPE without sign ambiguity. *Methodology*: architecture-agnostic decoupling benchmarked across MP-GNNs and GTs. *Quality*: Grade A. *Contribution*: random-walk encodings are polynomials in `A`, hence well defined and continuous for weighted adjacency.

**Huang, Y., Lu, W., Robinson, J., Yang, Y., Zhang, M., Jegelka, S., & Li, P. (2024). On the stability of expressive positional encodings for graphs. *International Conference on Learning Representations*. https://arxiv.org/abs/2310.02579**
*Relevance*: the perturbation question, for the spectral family. *Key findings*: eigenvector PEs are discontinuous in the Laplacian; stability bounds depend on the eigengap; the domain-generalisation gap is bounded by the encoder's Hölder constant; stability trades against expressivity. *Methodology*: formal stability definition, proofs, molecular and OOD experiments. *Quality*: Grade A; results concern spectral, not random-walk, encodings. *Contribution*: a predicted graph should be encoded by continuous functions of `A`.

**Black, M., Wan, Z., Mishne, G., Nayyeri, A., & Wang, Y. (2024). Comparing graph transformers via positional encodings. *Proceedings of the 41st International Conference on Machine Learning*. https://arxiv.org/abs/2402.14202**
*Relevance*: absolute vs relative encoding for a pair readout. *Key findings*: APEs and RPEs are equivalent in distinguishing power on featureless graphs; with node features RPEs may be strictly advantageous. *Methodology*: interchange constructions plus a WL variant for RPE-augmented graphs. *Quality*: Grade A. *Contribution*: our coarse nodes carry features, so the pair channel is the principled home for `topo_rel`.

**Kim, J., Nguyen, T. D., Min, S., Cho, S., Lee, M., Lee, H., & Hong, S. (2022). Pure transformers are powerful graph learners. *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2207.02505**
*Relevance*: an alternative tokenisation for a soft graph — edges as tokens. *Key findings*: with orthonormal node identifiers plus type identifiers, a standard transformer layer approximates any order-k equivariant linear layer, hence ≥ 2-IGN. *Methodology*: approximation theorems plus large-scale benchmarks. *Quality*: Grade A; cost is quadratic in nodes *plus* edges, punitive on a dense soft graph. *Contribution*: structure can be carried entirely by token embeddings rather than attention bias.

**Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., & Sinop, A. K. (2023). Exphormer: Sparse transformers for graphs. *Proceedings of the 40th International Conference on Machine Learning*. https://arxiv.org/abs/2303.06147**
*Relevance*: virtual global nodes inside a sparse attention pattern. *Key findings*: expander + local edges + a constant number of virtual global nodes retains universal approximation; the virtual nodes act as a global storage sink. *Methodology*: spectral expander lemmas plus benchmarks to MalNet-Tiny. *Quality*: Grade A. *Contribution*: a handful of global nodes suffices to route information, suggesting K = 64 may be far more than needed.

### 4.2 Virtual nodes

**Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70). https://arxiv.org/abs/1704.01212**
*Relevance*: origin of the virtual/master node. *Key findings*: a master node with its own dimension and update improves the average error ratio from 3.47 to 2.62, better than adding virtual edges (2.90). *Methodology*: controlled ablation over 13 QM9 targets. *Quality*: Grade A. *Contribution*: the master node is a global scratch space read *and written* each step — something our frozen coarse graph does not allow.

**Hwang, E., Thost, V., Dasgupta, S. S., & Ma, T. (2022). An analysis of virtual nodes in graph neural networks for link prediction (extended abstract). *The First Learning on Graphs Conference*. https://openreview.net/forum?id=dI6KBKNRp7**
*Relevance*: the only located work applying multiple, clustered virtual nodes to link prediction. *Key findings*: per the verified abstract, multiple VNs with clustering-determined attachments give "rather stable performance increases" and sometimes large boosts on OGB link prediction, with a theoretical analysis. *Methodology*: OGB benchmark study; full text unreachable (OpenReview browser check), so no numbers are quoted. *Quality*: Grade C for this thread — Tier 2, abstract-only reading. *Contribution*: precedent that clustered virtual nodes are a legitimate LP mechanism.

**Cai, C., Hy, T. S., Yu, R., & Wang, Y. (2023). On the connection between MPNN and graph transformer. *Proceedings of the 40th International Conference on Machine Learning*. https://arxiv.org/abs/2301.11956**
*Relevance*: how much a graph transformer adds over a virtual node. *Key findings*: MPNN+VN with `O(1)` depth and width approximates a linear-attention layer; substituting it for GraphGPS's global module gives 0.0942 vs 0.0938 PCQM4Mv2 validation MAE and beats all transformers on Peptides-func. *Methodology*: approximation theorems under stated assumptions plus drop-in benchmark substitutions. *Quality*: Grade A; full self-attention needs `O(n^d)` width, so the equivalence is not free. *Contribution*: the strongest theoretical statement that a coarse-node aggregator is transformer-class.

**Southern, J., Di Giovanni, F., Bronstein, M., & Lutzeyer, J. F. (2025). Understanding virtual nodes: Oversquashing and node heterogeneity. *International Conference on Learning Representations*. https://arxiv.org/abs/2405.13526**
*Relevance*: explains the project's measured gate collapse. *Key findings*: a standard VN's Jacobian is source-independent beyond two hops, i.e. structurally homogeneous; MPNN+VN matches GTs exactly on tasks whose attention is homogeneous; reordering to local-then-global (`VN_G`) restores heterogeneity at identical cost and improves every benchmark (+8.25 % CIFAR10). *Methodology*: commute-time theorems, Jacobian sensitivity analysis, LRGB/OGB benchmarks. *Quality*: Grade A. *Contribution*: a cheap, theory-backed fix that needs no transformer.

**Qian, C., Manolache, A., Morris, C., & Niepert, M. (2024). Probabilistic graph rewiring via virtual nodes. *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2405.17311**
*Relevance*: the closest published analogue of the project's virtual graph. *Key findings*: differentiable exact-k-subset sampling of node→virtual-node edges; provably beyond 1-WL; best results on Peptides-func (0.7210 vs GRIT 0.6988) and on the PCQM-Contact link-prediction task (0.3670 vs Exphormer 0.3637), at linear cost and lower memory than GPS. *Methodology*: theory plus seven benchmark families with variance and runtime. *Quality*: Grade A. *Contribution*: learned virtual nodes can beat graph transformers, and discrete k-subset attachment is the working alternative to dense soft attention.

### 4.3 Producing two endpoint tokens and one relation token

**Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). Set Transformer: A framework for attention-based permutation-invariant neural networks. *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97). https://arxiv.org/abs/1810.00825**
*Relevance*: the PMA readout the project vendors. *Key findings*: `PMA_k` with k seeds emits k pooled vectors and is prescribed with a following SAB whenever those outputs are correlated; the architecture is a universal approximator of permutation-invariant functions. *Methodology*: constructive propositions plus amortised-clustering and set-regression experiments. *Quality*: Grade A. *Contribution*: three seeds give three readouts for free, but the seeds are query-independent, so alone they cannot make `topo_u` and `topo_v` endpoint-specific.

**Zhang, M., Li, P., Xia, Y., Wang, K., & Jin, L. (2021). Labeling trick: A theory of using graph neural networks for multi-node representation learning. *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2010.16103** — local: `literature/models/link_prediction_structural/2021_neurips_arxiv_2010_16103_…pdf`
*Relevance*: the governing constraint on any three-token readout. *Key findings*: pooling independently computed node representations can never yield a structural link representation, however expressive the encoder; labelling the target set before the encoder runs fixes this; ω(n²) link pairs are separable only with the trick. *Methodology*: isomorphism-theoretic proofs plus link-prediction experiments. *Quality*: Grade A. *Contribution*: the pair must be injected into the virtual graph, not read off it afterwards.

**Li, P., Wang, Y., Wang, H., & Leskovec, J. (2020). Distance encoding: Design provably more powerful neural networks for graph representation learning. *Advances in Neural Information Processing Systems 33*. https://arxiv.org/abs/2009.00142** — local: `literature/models/link_prediction_structural/2020_neurips_arxiv_2009_00142_…pdf`
*Relevance*: how to condition a whole graph on the queried pair. *Key findings*: DE is the landing-probability profile of walks between each node and the target set, usable as extra features or as a controller of aggregation, and provably separates node sets 1-WL GNNs cannot. *Methodology*: expressivity theorems plus node/link/triad-level experiments. *Quality*: Grade A. *Contribution*: a walk-based relative encoding of coarse nodes with respect to `{u, v}` is the direct pair-conditioning mechanism, and it is polynomial in `A`.

**Shomer, H., Ma, Y., Mao, H., Li, J., Wu, B., & Tang, J. (2024). LPFormer: An adaptive graph transformer for link prediction. *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. https://doi.org/10.1145/3637528.3672025** — local: `literature/models/link_prediction_structural/2024_kdd_arxiv_2310_11009_…pdf`
*Relevance*: the reference design for a learned pair token inside a graph transformer. *Key findings*: the pair token is a `(a,b)`-conditioned attention pool over PPR-selected context nodes carrying `MLP(ppr(a,u), ppr(b,u))`; the final score concatenates `h_a⊙h_b`, the pair token **and** raw CN / 1-hop / >1-hop counts; on ogbl-ppa removing the counts costs 18.95 Hits@100 while removing learnable attention costs 0.55. *Methodology*: six datasets, six-way ablation, HeaRT hard-negative re-evaluation. *Quality*: Grade A. *Contribution*: counts and learned pair tokens are complementary, and on protein-association data the counts dominate.

**Bergen, L., O'Donnell, T. J., & Bahdanau, D. (2021). Systematic generalization with edge transformers. *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2112.00578**
*Relevance*: a transformer whose state *is* the pair. *Key findings*: every node pair holds a vector; triangular attention updates `x_ij` from `(i,l)` and `(l,j)`, giving relation composition natively; stronger systematic generalisation than GAT and standard transformers on CLUTRR. *Methodology*: compositional-generalisation benchmarks with held-out relation lengths. *Quality*: Grade A; cost is cubic in nodes. *Contribution*: the cleanest published answer to how a relation token should be read out — compose relations rather than pool nodes.

### 4.4 Soft and latent graphs feeding a downstream model

**Kazi, A., Cosmo, L., Ahmadi, S.-A., Navab, N., & Bronstein, M. M. (2023). Differentiable graph module (DGM) for graph convolutional networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *45*(2), 1606–1617. https://doi.org/10.1109/TPAMI.2022.3170249** — local: `literature/models/graph_structure_learning/latent_structure_learning/2023_ieee_…_2002_04999_…pdf`
*Relevance*: gradients through a sampled latent graph. *Key findings*: Gumbel-Top-k kNN sampling is non-differentiable with respect to the task loss, so they add an explicit graph loss rewarding edges whose endpoints are classified correctly, whose gradient approximates the expectation's. *Methodology*: continuous and discrete variants across transductive and inductive benchmarks. *Quality*: Grade A. *Contribution*: a downstream loss alone does not train a latent graph.

**Wu, Q., Zhao, W., Li, Z., Wipf, D., & Yan, J. (2022). NodeFormer: A scalable graph structure learning transformer for node classification. *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2306.08385** — local: `literature/models/graph_structure_learning/latent_structure_learning/2022_neurips_arxiv_2306_08385_…pdf`
*Relevance*: soft all-pair attention over a learned graph, at scale. *Key findings*: kernelised Gumbel-Softmax gives `O(N)` all-pair message passing with approximation and convergence guarantees, but a large temperature degrades the operator to plain mean pooling; a relational bias and an edge-level regularisation loss are required as guidance. *Methodology*: two theorems plus large-graph node-classification benchmarks. *Quality*: Grade A; node classification, not pair prediction. *Contribution*: names the collapse-to-mean failure mode observed in the project's gate.

**Lu, J., Xu, Y., Wang, H., Bai, Y., & Fu, Y. (2023). Latent graph inference with limited supervision. *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2310.04314** — local: `literature/models/graph_structure_learning/latent_structure_learning/2023_neurips_arxiv_2310_04314_…pdf`
*Relevance*: the "gradient-starved coarse nodes" diagnosis in published form. *Key findings*: sparsification leaves many learned edge weights outside the training loss; such weights determine test predictions but cannot be semantically optimal; eliminating starved nodes improves every LGI method tested, +6.12 % on Pubmed at a 0.3 % labelling rate. *Methodology*: k-hop starved-node definition, CUR-based pivot selection, benchmark study. *Quality*: Grade A. *Contribution*: supervision reaching the structure, not more capacity, is the published remedy.

**Manenti, A., Zambon, D., & Alippi, C. (2025). Learning latent graph structures and their uncertainty. *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267), 42882–42901. https://proceedings.mlr.press/v267/manenti25a.html** — local: `literature/models/graph_structure_learning/latent_structure_learning/2025_icml_arxiv_2405_19933_…pdf`
*Relevance*: whether a good downstream score implies a correct latent graph. *Key findings*: minimising a point-prediction loss does not guarantee the latent graph distribution is learned; losses on stochastic outputs recover both the distribution and optimal predictions. *Methodology*: theoretical analysis with controlled synthetic and real graph-learning experiments. *Quality*: Grade A. *Contribution*: justifies a distributional or explicitly supervised structural objective over task BCE alone.

### 4.5 Prefix tokens and freeze-then-unfreeze schedules

**Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics*, 4582–4597. https://doi.org/10.18653/v1/2021.acl-long.353**
*Relevance*: the mechanism the project's gated KV prefix instantiates. *Key findings*: 0.1 % of parameters matches full fine-tuning on full data, beats it by 2.9 BLEU in low-data settings and extrapolates better to unseen topics; directly optimising the prefix matrix is unstable, so it is reparameterised as an MLP of a smaller matrix, discarded afterwards. *Methodology*: table-to-text and summarisation with low-data and extrapolation splits. *Quality*: Grade A. *Contribution*: an implementation detail with measured consequences for our prefix parameters.

**Liu, X., Ji, K., Fu, Y., Tam, W. L., Du, Z., Yang, Z., & Tang, J. (2022). P-Tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 61–68. https://doi.org/10.18653/v1/2022.acl-short.8** (preprint: arXiv 2110.07602, "P-Tuning v2: …")
*Relevance*: shallow versus deep prompts. *Key findings*: continuous prompts at every layer, 0.1–3 % of parameters, match fine-tuning from 300 M to 10 B and on hard sequence labelling, where input-only prompt tuning fails. *Methodology*: scale sweep across NLU and sequence-labelling tasks. *Quality*: Grade A. *Contribution*: supports prefixing all nine cross-attention sites rather than only the input.

**Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 3045–3059. https://doi.org/10.18653/v1/2021.emnlp-main.243**
*Relevance*: when a frozen backbone suffices. *Key findings*: the gap to model tuning closes only at ~11 B parameters with large gaps at smaller scale; conversely prompt tuning is more robust than model tuning under domain shift, giving stronger zero-shot MRQA transfer. *Methodology*: scale ablations across T5 sizes plus a zero-shot domain-transfer study. *Quality*: Grade A; NLP scale, not ours. *Contribution*: the two-sided answer — freezing costs capacity at small scale but buys distribution-shift robustness.

**Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., … Simonyan, K. (2022). Flamingo: A visual language model for few-shot learning. *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2204.14198**
*Relevance*: the zero-initialised gated cross-attention the project uses. *Key findings*: removing the 0-init tanh gate costs 4.2 % and destabilises training; fine-tuning the backbone instead of freezing costs 8.0 % and training it from scratch 12.9 %; even co-training on the original corpus while unfrozen underperformed staying frozen; every-layer insertion beats every-fourth by 1.9 %. *Methodology*: single-factor ablations at 3 B scale on a fixed evaluation suite. *Quality*: Grade A; enormous scale, different modality. *Contribution*: the most direct measured support for a frozen trunk behind a zero-init gate.

**Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 328–339. https://doi.org/10.18653/v1/P18-1031**
*Relevance*: the canonical gradual-unfreezing schedule. *Key findings*: unfreezing one layer per epoch from the top, with discriminative and slanted triangular learning rates, gives 5.00/5.69/5.38 validation error on IMDb/TREC-6/AG; unfreezing everything at once gives 6.87/6.86/5.81; training only the top layer gives 6.49/16.09/8.38 and never reaches zero training error. *Methodology*: component-wise ablation on three datasets. *Quality*: Grade A; LSTM-era models. *Contribution*: the frozen extreme can underfit catastrophically, and unfreezing needs a schedule plus layer-wise learning rates to beat it.

**Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. *International Conference on Learning Representations*. https://arxiv.org/abs/2202.10054**
*Relevance*: the decisive freeze-then-unfreeze evidence. *Key findings*: across 10 shift datasets fine-tuning raises in-distribution accuracy from 83 % to 85 % but drops OOD from 66 % to 59 %; a randomly initialised head is the cause; linear-probe-then-fine-tune is 1 % better ID and 10 % better OOD than full fine-tuning. *Methodology*: theory on two-layer linear networks plus a 10-dataset study. *Quality*: Grade A. *Contribution*: warming up a new reader against a frozen trunk before unfreezing is the measured best of both.

**Lee, Y., Chen, A. S., Tajwar, F., Kumar, A., Yao, H., Liang, P., & Finn, C. (2023). Surgical fine-tuning improves adaptation to distribution shifts. *International Conference on Learning Representations*. https://arxiv.org/abs/2210.11466**
*Relevance*: which parameters to unfreeze. *Key findings*: tuning a single contiguous block and freezing the rest beats full fine-tuning by ~3 %, and the best block depends on the shift type — first block for input-level (82.8 %, +2.9), middle/late for feature-level (81.2 %, +2.1), last layer for output-level (86.2 %, +4.0). *Methodology*: nine real distribution shifts plus an automatic block-selection criterion. *Quality*: Grade A. *Contribution*: unfreezing should target the layers where the new signal enters.

---

## 5. Evidence against

**5.1 A learned virtual-node mechanism already beats graph transformers on the nearest analogue.**
Qian et al. (2024) attach real nodes to a small set of virtual nodes by differentiable k-subset
sampling and beat GRIT, GPS and Exphormer on Peptides-func/struct *and* on the PCQM-Contact
link-prediction task, at linear rather than quadratic cost. Cai et al. (2023) reach the same conclusion
from the other side: swapping GraphGPS's global attention for a virtual-node/DeepSets module moves
PCQM4Mv2 validation MAE by 0.0004. If the coarse-node design is held fixed, a transformer reader is
not obviously the missing ingredient.

**5.2 Counts are not subsumed by a learned pair token, and on protein-association data they
dominate.** LPFormer is an adaptive graph transformer whose entire contribution is a learned,
pair-conditioned attention encoding — and it still concatenates raw CN / 1-hop / >1-hop counts into the
score. Ablating those counts on ogbl-ppa costs 18.95 Hits@100 (63.32 → 44.37); ablating the learned
attention costs 0.55. This is direct evidence for "when counts are exactly what you want".

**5.3 Structural expressivity does not transfer off training size.** Müller et al. (2024) find every
probed graph transformer falls to ≤ 54.8 % on triangle counting when test graphs are larger than
training graphs, against 99.4 % at training size. A virtual graph whose effective size and density vary
with the query is exactly that regime.

**5.4 GRIT's guarantees are stated for binary adjacency.** Proposition 3.1 quantifies over
`G_n ⊆ {0,1}^{n×n}`; nothing certifies RRWP's expressiveness or the SPD approximation for fractional
weights. RRWP remains well defined and differentiable on a soft `A` because it is a polynomial in
`D⁻¹A`, which is why it is the right family under noise relative to spectral encodings (Huang et al.,
2024) — but "well defined" is not "provably expressive", and this thread found **no paper measuring
RRWP accuracy as a function of edge-weight noise**. That premise is untested.

**5.5 The transformer is not the prior; the encoding is.** Müller et al. state that without expressive
structural encodings a graph transformer equals DeepSets, and that GNNs with the same encodings have
the same expressive power. GraphGPS shows the same empirically: dropping the local MPNN triples ZINC
error, dropping global attention changes nothing. `R` therefore buys little unless its input encoding
carries what the counts carried.

**5.6 Freezing is not free.** ULMFiT's fully frozen trunk produced 16.09 % validation error on TREC-6
against 5.69 % for the full schedule and never reached zero training error; Lester et al. report "large
gaps" for prompt-only adaptation below ~10 B parameters. Our trunk is small, so a frozen `F` may simply
underfit the topology signal.

**5.7 Against revision 2: uniformity is the default and better templates will not remove it.**
Southern et al. prove a standard virtual node's Jacobian is source-independent beyond two hops; their
own fix is an ordering change, not a better prior. NodeFormer states that a large temperature degrades
its operator to plain mean pooling, and DGM needed a separate reward-based graph loss because the task
loss gave its sampler no usable gradient. A prior-knowledge template supervised only by task BCE risks
reproducing both failures. Manenti et al. (2025) add that minimising a point-prediction loss does not
guarantee the latent structure was learned, so a template-based prompt that improves AUPRC is not
evidence the template was recovered.

**5.8 Against the project's premise.** Zhang et al. (2021, Prop. 1) prove that aggregating
independently computed node representations can never be a structural link representation. Any prompt
whose endpoint tokens are computed separately and then concatenated inherits that ceiling regardless of
coordinate quality — the pair must condition the computation, not only its output. This places the
current `coord_gen` interface, which predicts per-endpoint and per-pair scalars from separately pooled
endpoint states, on the wrong side of a known impossibility unless the pair enters earlier.

---

## 6. What this thread cannot answer

1. **How RRWP degrades as edge weights become fractional and wrong.** No located paper measures
   random-walk relative encodings under weight noise. Needs our own experiment: RRWP on true
   ego-structure, then on progressively perturbed soft adjacencies, measuring reader V_val AUPRC and GS.
2. **Whether a GT reader over `G^P_uv` beats closed-form counts *given our generator's actual
   attachment quality*.** Every external comparison assumes an observed graph. With attachment transfer
   at R² ≤ 0 on V_val, only an oracle-attachment ablation separates reader capacity from attachment
   quality.
3. **Which of the three tokens carries the gain in our architecture.** The literature supplies
   mechanisms (PMA seeds, labelling, triangular attention, LPFormer pooling) but no evidence on the
   relative value of endpoint versus relation tokens. Our Stage I measurement is the only datum.
4. **The cost/benefit of unfreezing `F` here.** LP-FT, surgical fine-tuning, ULMFiT and Flamingo
   disagree in a way fully explained by scale and shift type. Our trunk is small (favouring unfreezing)
   but the split is node-held-out (favouring freezing). Only a two-arm run resolves it.
5. **Which templates are informative for this PPI graph.** Thread B's question; this thread establishes
   only that counts remain load-bearing next to learned attention on ogbl-ppa.
6. **Compute feasibility of triangular attention at n ≈ 300.** Edge Transformer is `O(n³)` per layer;
   whether a restricted variant (pairs touching `u`, `v` or a coarse node) fits our budget is an
   engineering measurement.

---

## 7. Saved files

New PDFs, read and not previously in the corpus. Two folders were created; the orchestrator may prefer
a different home for the second group and will update `literature/README.md`.

`literature/models/graph_transformers/` — `2017_icml_arxiv_1704_01212_neural_message_passing_for_quantum_chemistry.pdf`;
`2019_icml_arxiv_1810_00825_set_transformer_a_framework_for_attention_based_permutation_invariant_neural_networks.pdf`;
`2020_neurips_arxiv_2002_04025_can_graph_neural_networks_count_substructures.pdf`;
`2021_neurips_arxiv_2106_05234_do_transformers_really_perform_bad_for_graph_representation.pdf`;
`2021_neurips_arxiv_2112_00578_systematic_generalization_with_edge_transformers.pdf`;
`2022_iclr_arxiv_2110_07875_graph_neural_networks_with_learnable_structural_and_positional_representations.pdf`;
`2022_neurips_arxiv_2205_12454_recipe_for_a_general_powerful_scalable_graph_transformer.pdf`;
`2022_neurips_arxiv_2207_02505_pure_transformers_are_powerful_graph_learners.pdf`;
`2023_icml_arxiv_2301_11956_on_the_connection_between_mpnn_and_graph_transformer.pdf`;
`2023_icml_arxiv_2303_06147_exphormer_sparse_transformers_for_graphs.pdf`;
`2024_iclr_arxiv_2310_02579_on_the_stability_of_expressive_positional_encodings_for_graphs.pdf`;
`2024_icml_arxiv_2402_14202_comparing_graph_transformers_via_positional_encodings.pdf`;
`2024_neurips_arxiv_2405_17311_probabilistic_graph_rewiring_via_virtual_nodes.pdf`;
`2024_tmlr_arxiv_2302_04181_attending_to_graph_transformers.pdf`;
`2025_iclr_arxiv_2405_13526_understanding_virtual_nodes_oversquashing_and_node_heterogeneity.pdf`

`literature/models/prompt_and_prefix_tuning/` — `2018_acl_arxiv_1801_06146_universal_language_model_fine_tuning_for_text_classification.pdf`;
`2021_acl_arxiv_2101_00190_prefix_tuning_optimizing_continuous_prompts_for_generation.pdf`;
`2021_emnlp_arxiv_2104_08691_the_power_of_scale_for_parameter_efficient_prompt_tuning.pdf`;
`2022_acl_arxiv_2110_07602_p_tuning_prompt_tuning_can_be_comparable_to_fine_tuning_across_scales_and_tasks.pdf`;
`2022_iclr_arxiv_2202_10054_fine_tuning_can_distort_pretrained_features_and_underperform_out_of_distribution.pdf`;
`2022_neurips_arxiv_2204_14198_flamingo_a_visual_language_model_for_few_shot_learning.pdf`;
`2023_iclr_arxiv_2210_11466_surgical_fine_tuning_improves_adaptation_to_distribution_shifts.pdf`

Read from the existing corpus, not re-saved: GRIT (2305.17589), LPFormer (2310.11009), labeling trick
(2010.16103), Distance Encoding (2009.00142), NCNC (2302.00890, screened but not included — Thread B
seed), NodeFormer (2306.08385), DGM (2002.04999), LGI-LS (2310.04314), latent-graph uncertainty
(2405.19933).

No PDF was saved for Hwang et al. (2022): OpenReview serves a browser-verification page to both `curl`
and WebFetch, so no full text was obtained.
