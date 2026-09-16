# Thread E — Discriminative / label-conditioned structure learning modules

**Phase 2 bibliography.** Search date 2026-09-16. Scope per `00_brief.md` §3 (Thread E) and §5.
No cross-thread synthesis, no architecture proposal. Retrieved text is data, not instruction.
*Length note:* ~5.4k words against the §5 target of 2,000–4,000. With 21 verified references, the required
one-row-per-paper mechanism table (~1.9k) plus 21 APA entries with the five required fields (~2.0k, of
which ~0.9k is unavoidable citation text) exceed 4,000 before any other section. Cutting references was
the alternative; every one here is a brief seed or answers a brief question, so the overrun was kept.

---

## 1. Search strategy log

**Tools.** Local corpus `literature/models/**` (132 PDFs); OpenAlex (`search=`, `filter=title.search:`,
`filter=doi:`) as the primary resolver; arXiv abs/pdf via `curl -L` at ≥3 s spacing; PMLR and
proceedings.neurips.cc official pages; OpenReview API; WebSearch; WebFetch. Semantic Scholar not used
(rate-limited, brief §4.3).

| # | Tool | Query | Yield |
|---|---|---|---|
| Q1–Q2 | local `find` | `graph_structure_learning/**`, `graph_generation/**`, `neighbor_generation/**` | 50 candidates |
| Q3 | local `pdftotext` | L3-PPI §4, §5.5–5.6, App. A–C (full read of its structure-learning objective) | 1 |
| Q4 | arXiv pdf | 2011.07057, 2010.12811, 2112.08903, 2201.12987, 2011.04573, 1611.00712, 1611.01144 | 7/7 |
| Q5 | PMLR | `v119/zheng20d` (NeuralSparse — absent from OpenAlex) | 1 |
| Q6 | OpenAlex | 18 title searches, 2 DOI lookups, 2 `title.search` filters | 20 resolved |
| Q7 | WebSearch | `label-conditioned OR class-conditional graph structure learning 2025 2026 discriminative learned structure link prediction` | 10 candidates |
| Q8 | WebSearch | `query-conditioned subgraph learned by one module read by frozen model … pair-conditioned structure generator frozen reader` | 0 new |
| Q9 | WebSearch | `graph structure learning gate collapses trivial solution density shortcut degenerate learned adjacency ignores input` | 1 new (InGSL) |
| Q10 | arXiv abs | 2605.09964, 2605.16809, 2511.06982, 2502.12618 | 4 resolved |
| Q11 | OpenReview | MoG venue check | 0 — cited as preprint |

**PRISMA flow.** Identified 68 (local 51, arXiv 7, PMLR 1, WebSearch 10) → 1 duplicate removed (InGSL)
→ 67 screened on title/abstract/folder index → 40 excluded → 27 assessed by full-text or method+results
extraction → 6 excluded after assessment → **21 included**.

Excluded after assessment: UnGSL (2502.12618), node-quality gating only, neither label- nor
pair-conditional; SCGG, conditions on an *observed* subgraph rather than a label or query pair;
subgraph-conditioned GIB (2412.15589), molecular pre-training with no discriminative structure objective;
SE-GSL (2303.09778), structural-entropy objective, not label-conditioned; modularity-aware GAE
(2202.00961) and overlapping community detection (1909.12201), Thread D remit; latent graph inference with
limited supervision (2310.04314) and latent structures + uncertainty (2405.19933), Thread C remit.

**DISTRIBUTIONAL_SKEW_ADVISORY** — Dimension: venue tier. Concentration: peer-reviewed proceedings =
16/21 (76%). Advisory: skew is in the favourable direction; recorded, not a defect. Dimension:
methodological. Concentration: computational method papers = 19/21 (90%); only two sources (OpenGSL,
GSLB) are independent reproductions, and §5 rests heavily on them. Search response: both benchmarks were
read in full; no further expansion, as no third GSL benchmark exists. Time (2017–2026): no skew.

---

## 2. Inclusion / exclusion criteria as applied

| Criterion | Include | Exclude |
|---|---|---|
| Relevance | Learns a graph structure (edge gates, subgraph selection, synthetic edges, prompt sub-structure) driven by a **label/task** signal; or supplies the discreteness machinery those methods use; or independently benchmarks them | Structure learning driven only by reconstruction/proximity; pooling and coarsening (Thread D); graph transformers as readers (Thread C); prompt mechanics without a learned structure (Thread A) |
| Mechanism detail | Objective, conditioning variable, gate parameterisation and evaluation-time behaviour recoverable from the text | Abstract-only; mechanism unrecoverable |
| Evidence | ≥1 measured number on class separation of the learned structure, discreteness handling, transfer to unseen nodes, or frozen-vs-trained reader | Position papers; theory with no measurement |
| Verification | Resolves on OpenAlex / arXiv / Crossref DOI / official proceedings with matching title, authors, year | Unresolved — dropped, never flagged "uncertain" |
| Currency | 2017–2026, ≥2019 preferred | Pre-2017 unless the canonical estimator |

Two brief seeds were retained but reclassified: GraphSMOTE for its hard-vs-soft edge ablation rather than
as augmentation; latent graph diffusion (2402.02518) for its conditioning interface only, since its
structure is not label-gated.

---

## 3. Mechanism extraction table

Tier 1 = peer-reviewed, Tier 2 = preprint. R1 = read on revision 1 (graph-transformer reader over
`G^P_uv`); R2 = read on revision 2 (template + learnable discriminative module).

| Key | Venue/yr | WHY | HOW (structure · conditioning · objective · discreteness · readout) | WHAT (numbers) | R1 / R2 | Tier | Verification |
|---|---|---|---|---|---|---|---|
| **L3PPI** | ICML 2026; arXiv 2605.09964 | Inject the L3 prior into a frozen PPI predictor | **Fixed** star template: K+1 prompt nodes, `E^P={(v0,vi)}`, insert `E^I={(v0,v),(vi,u)}`; only the K+1 prompt embeddings learn, **shared across all pairs**. Gate `p_i=GNN_gpt(path_i)` on the 4-node path `{u,v,v0,vi}` → **K numbers per pair**, a function of the pair and `x_i^P`. `L_BCE+λL_PN`; Eq. 8 = class-conditional hinge on `Σp_i` (`≥K(1−1/γ)` if y=1, `≤K/γ` if y=0), γ∈{1.5,2,3}, λ∈[0.1,0.7]. Gumbel-Softmax, τ annealed, soft backward, **hard `1[p_i>0.5]` at inference**. Gates → edge weights; a **frozen** GIN reads the pattern graph | Ablation (PIPR+, micro-F1 SHS27k/148k/STRING): full **83.22/88.08/94.30**; −`L_PN` 80.10/85.17/90.92; −gate **76.56/79.90/87.17**; −pretrain 81.92/86.77/91.91. Schedule (Table 7): prompt→gate **F1 0.87 / 451 s**; joint 0.83/1424 s; gate→prompt 0.84; prompt-only 0.79; gate-only 0.72. Predicted vs true #L3 paths (K=50): **ρ = 0.65/0.57/0.53**. Unseen-protein NS: 37.81→45.61, 43.90→59.60, 42.08→63.02; BS 91.97→91.57; three dataset **averages fall** (90.87→87.33, 94.53→93.29, 66.52→63.78) | Frozen reader over a tiny gated graph suffices / the reference template+gate+hinge design, gate-dominated | 1 | arXiv abs 2026-05-11 + in-PDF "43rd ICML, PMLR 306, 2026" |
| **GSAT** | ICML 2022; arXiv 2201.12987 | Label-relevant subgraphs without sparsity assumptions | Per-edge Bernoulli attention from `g_φ(G)`; GIB `max I(G_S;Y)−βI(G_S;G)`, second term bounded by KL to a fixed `Ber(r)` prior — **no size/sparsity constraint**; stochasticity retained, its *reduction* is the signal | Spurious-Motif b=0.5, interp. AUC: **79.81±3.98** / β=0 66.00±11.04 / **NoStoch 59.64±5.33**. Accuracy: GIN 39.87±1.30, GSAT **51.86±5.51**, β=0 45.97±8.37, **NoStoch 40.34±2.77** (≈ plain backbone). Best `r∈[0.5,0.9]`; `r→0` degrades. Info-regulariser beats grid-searched ℓ1. Thm 4.1: label-determining subgraph is the GIB optimum ∀β∈[0,1]. §4.6: fine-tuning a pre-trained GNN "almost never hurts" | Favours fine-tune over hard-freeze / **the counter-model to a count hinge** | 1 | OpenAlex W4221152136; in-PDF ICML 39 |
| **PGExplainer** | NeurIPS 2020; arXiv 2011.04573 | Collective, inductive explanation of a trained GNN | Binary-concrete gates with **`ω_ij=MLP_Ψ([z_i;z_j;z_v])`** — one edge, a different gate per **query node**; graph-level variant `MLP_Ψ([z_i;z_j])`. Objective: cross-entropy between the **frozen** GNN's prediction `Y_o` and its prediction on the gated graph — **reader-fidelity, not the label** | Up to **24.7% relative AUC** over per-instance GNNExplainer; amortised Ψ explains unseen nodes **without retraining** | A relation-conditioned readout form / the clearest "pair-conditioned structure + separate frozen reader" | 1 | OpenAlex W3103717137; in-PDF NeurIPS 2020 |
| **PTDNet** | WSDM 2021; arXiv 2011.07057 | Drop task-irrelevant edges, inductively | **`α^l_uv=f^l_θ(h^l_u,h^l_v)`** — MLP on endpoint states, amortised and live at test time; task loss + ℓ0-style sparsity + **nuclear-norm low rank**; hard-concrete gates | Cross-community edge ratio 0.211 → 0.206 (sparsity) → **0.189** (+low rank): a global structural regulariser moves the structure ~10% relative. **LP AUC**: Cora 0.916 vs GAE 0.910, Citeseer 0.918 vs 0.895, Pubmed **0.963 vs 0.964**. Pos/neg edge weights separate on *test* edges; unparameterised DropEdge cannot | Endpoint-state gate transfers to test rows / a global structural regulariser is a weak lever | 1 | OpenAlex W3116239416; DOI 10.1145/3437963.3441734 |
| **NeuralSparse** | ICML 2020 (PMLR v119) | Supervised sparsification under an inductive budget | **k-neighbor subgraphs** (≤k one-hop edges per node); sampler `f_φ` **shared across nodes**, |φ| independent of graph size; task loss over `l` samples; Gumbel-Softmax | Up to **+7.2%** accuracy on inductive Reddit/PPI/Transaction. Stated failure mode: "when τ is small … **high-variance gradients block effective backpropagation**. A high value of τ cannot produce expected sparsification" | — / the budget is **per-row, never global** | 1 | PMLR `v119/zheng20d.html` (title + 8 authors) |
| **GIB** | NeurIPS 2020; arXiv 2010.12811 | Information-theoretic principle over structure + features | Local-dependence assumption ⇒ tractable Markov-chain space; **GIB-Cat** (categorical neighbour sampling), **GIB-Bern** (Bernoulli edges) | Up to **31%** improvement under adversarial structure+feature perturbation. Open question: better instantiations "especially in **capturing discrete structural information**" | — / the principled replacement for Eq. 8 | 1 | OpenAlex W3094193403; proceedings.neurips.cc/paper/2020/…ebc2aa04…-Paper.pdf |
| **VIBGSL** | AAAI 2022 | IB structure learning for **graph-level** tasks | Masks task-irrelevant features, learns a new adjacency; variational IB bound for irregular data | Argues explicit constraints ("community, sparsity, low-rank, smoothness") "**may not be applicable to all datasets and tasks**"; sparsifiers give small and sometimes **negative** gains (COLLAB). **Not reproducible — see GSLB** | — / cautions against hand-fixed structural priors | 1 | OpenAlex W4225977739; DOI 10.1609/aaai.v36i4.20335 |
| **LDS** | ICML 2019; arXiv 1903.11960 | Learn a discrete graph jointly with a GCN | Bernoulli edge matrix; **bilevel**, outer objective = *validation* error; inference samples S graphs or uses probabilities as weights | Learns "highly non-uniform edge probabilities that **reflect the class membership of the nodes**" — with **no class-conditional term**. Expected edge count **exceeds** the original graph; density still <0.2% on Cora | — / class separation can *emerge* from a task objective alone | 1 | OpenAlex W2924719072 |
| **IDGL** | NeurIPS 2020; arXiv 2006.13009 | Iterative graph+embedding learning, inductive-capable | Recasts GSL as **similarity metric learning** on features + anchors; smoothness/connectivity/sparsity regularisers; dynamic stopping | LDS-style methods "optimize edge connectivities by assuming that the graph nodes are known, **they are unable to cope with the inductive setting**" | — / the structure module must be an amortised function of `(x_u,x_v)` | 1 | OpenAlex W3036106327 |
| **ProGNN** | KDD 2020 | Robust GNN via structural priors | Free adjacency parameters + sparsity + nuclear norm + feature smoothness; transductive | The object of SUBLIME's edge-distribution-bias finding | — / prior-driven, label-supervised, transductive: the shape to avoid | 1 | OpenAlex W3081203761; DOI 10.1145/3394486.3403049 |
| **SUBLIME** | WWW 2022 | Label-free GSL via bootstrapped anchors | Learner view vs bootstrapped anchor view; kNN-sparsify → activate → symmetrise; MI maximisation | Names "**the bias of edge distribution**" as a defect of supervised GSL and shows it: Pro-GNN builds "**more connections across labeled nodes than unlabeled nodes**" (Fig. 6 heatmap); the label-free objective does not | — / **shortcut evidence**: label supervision concentrates structure on supervised rows | 1 | OpenAlex W4221166060; DOI 10.1145/3485447.3512186 |
| **InGSL** | arXiv 2605.16809 (2026) | GSL's gains come with edge-count explosion | Learnable **diversity** score `w_ij` × similarity `S_ij`, hard threshold `ψ` at ε; `L=L_GSL+βL_MI` with `L_MI` an InfoNCE between representations from the reduced structure `S̃` and the full candidate set `S` under a teacher `GNN_T` — **reader-fidelity, not label** | Fig. 1: GSL methods "even exhibit **adverse effects** when the additional edge count is less than a certain multiple of the original graph (**e.g. 4×** for GRCN and RWGSL on Cora)". Fig. 2: similarity selection keeps neighbours mutually similar (redundancy). Cora, 30→90% edge reduction: bilinear 85.20→83.67 vs similarity 83.73→81.73 | — / diversity, not more similarity, is what a structure module adds | 2 | arXiv abs (8 authors, 2026-05-16); ACM fields unfilled ⇒ preprint |
| **MoG** | arXiv 2405.14260 (2024) | Per-node, not global, sparsification | Noisy top-k router `Ψ(G^(i))=Softmax(TopK(ψ(x_i),k))` over K **sparsifier experts** with different sparsity levels and criteria, on each node's ego graph; Grassmann-manifold mixture | Premise, tested: prior work relies on "a **single global sparsity setting and uniform pruning criteria**, failing to provide customized schemes for each node's complex local context". 8.67–50.85% sparsity at ≥ dense performance; 1.47–2.62× speed-up | — / replaces one scalar knob with a per-row choice among criteria | 2 | OpenAlex W4398796849. **ICLR 2025 in the local filename not confirmed** |
| **GraphSMOTE** | WSDM 2021 | Class-conditional synthesis of minority nodes *and their edges* | Interpolate minority embeddings; a trained **edge generator** attaches them. `_T` = generator trained by the **edge-prediction loss only**, edges **thresholded to binary**; `_O` = **continuous** edges, classifier gradient into the generator | Cora `_T`/`_O`: ACC **0.713/0.709**, AUC **0.929/0.927**, F **0.720/0.712**; BlogCatalog AUC **0.602/0.591**; Twitter AUC **0.622/0.616**. The structurally-supervised, hard-thresholded generator **matches or beats** the soft joint one on all three | — / direct structural supervision suffices; the task gradient into the generator is not the active ingredient | 1 | OpenAlex W3134509497; DOI 10.1145/3437963.3441720 |
| **Concrete** | ICLR 2017; arXiv 1611.00712 | Reparameterisation for discrete variables | Binary/Concrete relaxation; every discrete variable is the τ→0 limit of a Concrete one | Protocol: train with a Concrete node "at some fixed temperature (or with an **annealing schedule**) … **At test time the original discrete loss is evaluated**" | — / fixes the hard-at-eval convention | 1 | OpenAlex W2548228487; in-PDF ICLR 2017 |
| **GumbelSoftmax** | ICLR 2017; arXiv 1611.01144 | Same, categorical | Gumbel-Softmax + **Straight-Through**: argmax forward, `∇_θ z ≈ ∇_θ y` backward | "**ST Gumbel-Softmax allows samples to be sparse even when the temperature τ is high**" — decouples discreteness from gradient variance, at the cost of a biased gradient | — / the standard fix for a gate that saturates or will not discretise | 1 | OpenAlex W2547875792; in-PDF ICLR 2017 |
| **OpenGSL** | NeurIPS 2023 D&B; arXiv 2306.10280 | Fair re-evaluation, 12 methods × 10 datasets | Uniform splits/processing; effectiveness, homophily, cross-model transfer, cost | ① GSL "**does not consistently outperform vanilla GNN** counterparts". ③ learned homophily is "**scarcely different from the original, and in some cases even lower**" — *including GEN, which explicitly maximises it*. ④ homophily predicts performance on **2 of 10** datasets, anti-predicts on Citeseer and Wiki-cooc. ⑤ **cross-reader transfer is strong**: SUBLIME's BlogCatalog structure gives GCN 95.01 (↑18.89), SGC 94.95 (↑19.58), feature-free LPA 89.04 (↑31.51), LINK 86.45 (↑21.98). ⑥ ProGNN 190× GCN runtime; CoGSL 66× memory | **⑤ = the strongest support for a generator/frozen-reader split** / ③④ warn against optimising a global structural statistic | 1 | OpenAlex W4381551214; in-PDF NeurIPS 2023 D&B |
| **GSLB** | NeurIPS 2023 D&B; arXiv 2310.05174 | 16 algorithms × 20 datasets; effectiveness, robustness, complexity | Transductive and structure-inference scenarios, node- and graph-level | "(2) on **graph-level** tasks, current GSL models bring **limited improvement**"; "(3) most GSL algorithms (especially **unsupervised**) show impressive robustness"; "structure learning based on **label information is insufficient**". **VIB-GSL "exhibits strong instability across different random seeds … unable to surpass the baseline models consistently."** Bottleneck: `O(N²)` pairwise edge modelling | — / independent reproduction failure of the leading IB-GSL method | 1 | OpenAlex W4387559766; in-PDF NeurIPS 2023 D&B |
| **GSLSurvey** | arXiv 2103.03036 (2021) | Taxonomy of GSL | Structure learners (metric/neural/direct) × regularisation (sparsity, smoothness, community) × post-processing (discrete sampling) | With a log-barrier connectivity constraint, "changing its coefficient **does not control the degree of sparsity but only scales the solution**" | — / a regulariser coefficient is not a density control | 2 | OpenAlex W4287282757 |
| **LGD** | NeurIPS 2024; arXiv 2402.02518 | Generation + prediction at all graph levels, one formulation | Encode structure+features to a latent space and diffuse there; **conditional generation through a designed cross-attention mechanism**; LP = reconstruct known features while predicting remaining links | SOTA/competitive across generation and regression; the condition enters through attention rather than as an input feature | Cross-attention conditioning parallels the prefix interface / — | 1 | OpenAlex W4391591129; in-PDF NeurIPS 2024 |
| **CGLE** | arXiv 2511.06982 (2025) | Class-level semantics for GNN link prediction | **Class-conditioned link-probability matrix** (entry = P(link \| class pair)) from labels or clustering pseudo-labels, concatenated to the backbone's structural link embedding, read by an MLP | The one recent paper making the link decision explicitly class-conditional. **Caveat:** needs node class labels, which `(x_u,x_v)` does not provide at inference | — / class-conditional structure as a *prior*, not a gate | 2 | arXiv abs: Mazumder & Bedathur, 2025-11-10 |

---

## 4. Annotated bibliography (APA 7)

**Gao, Z., Zi, C., Liu, Z., Meng, Z., Li, Y., & Li, J. (2026). Learning the interaction prior for
protein–protein interaction prediction: A model-agnostic approach. *Proceedings of the 43rd International
Conference on Machine Learning* (PMLR 306). https://arxiv.org/abs/2605.09964** —
*Local:* `literature/models/knowledge_distillation/Learning the Interaction Prior…pdf`
*Relevance:* the reference design for revision 2 — fixed template, pair-conditioned learnable gate,
class-conditional objective, frozen reader. *Key findings:* removing the gate costs 6.7–8.2 micro-F1
("completely fail"), removing the hinge only 2.9–3.4; prompt-first-then-gate reaches F1 0.87 in 451 s vs
0.83 in 1424 s joint; inferred path counts track true counts at ρ = 0.53–0.65; gains concentrate on unseen
proteins. *Methodology:* two-stage prompt tuning, four PPI datasets, three split regimes, three seeds.
*Quality:* peer-reviewed; ablations adversarial to the authors' own hinge, but three per-dataset averages
fall without discussion. Grade A. *Contribution:* the gate, not the count constraint, carries the mechanism.

**Miao, S., Liu, M., & Li, P. (2022). Interpretable and generalizable graph learning via stochastic
attention mechanism. *Proceedings of the 39th International Conference on Machine Learning*.
https://arxiv.org/abs/2201.12987**
*Relevance:* the strongest argument that a size/count constraint is the wrong lever, with the alternative
measured. *Key findings:* a deterministic gate returns the model to the plain-GIN baseline (40.34 vs 39.87)
while the stochastic gate reaches 51.86; the information constraint beats grid-searched ℓ1; best retention
prior `r ∈ [0.5, 0.9]`; fine-tuning a pre-trained predictor "almost never hurts". *Methodology:* eight
datasets, interpretation AUC and accuracy, three-way ablation with a matching theorem. *Quality:*
peer-reviewed, controlled degradation study. Grade A. *Contribution:* isolates stochasticity, not
sparsity, as load-bearing.

**Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., & Zhang, X. (2020). Parameterized explainer for
graph neural network. *Advances in Neural Information Processing Systems, 33*.
https://arxiv.org/abs/2011.04573**
*Relevance:* the only source that literally implements a query-conditioned structure learned by one module
and read by a separate **frozen** model. *Key findings:* `ω_ij = MLP_Ψ([z_i; z_j; z_v])` makes an edge's
gate depend on the query; the objective is agreement with the frozen reader's own prediction, not the
label; amortisation gives inductive transfer with no retraining and up to 24.7% relative AUC over
per-instance optimisation. *Quality:* peer-reviewed, but explanation AUC is a proxy and the datasets are
largely synthetic. Grade B+. *Contribution:* supplies both the conditioning form and a label-free objective.

**Luo, D., Cheng, W., Yu, W., Zong, B., Ni, J., Chen, H., & Zhang, X. (2021). Learning to drop: Robust graph
neural network via topological denoising. *Proceedings of WSDM '21*, 779–787.
https://doi.org/10.1145/3437963.3441734**
*Relevance:* an amortised endpoint-state gate that survives to test time, and the only measurement of how
far a global structural regulariser actually moves a learned structure. *Key findings:* nuclear-norm low
rank shifts the cross-community edge ratio only 0.211 → 0.189; link-prediction gains over a plain GAE are
+0.006/+0.023/−0.001 AUC, an order of magnitude below its node-classification gains. *Quality:*
peer-reviewed, honest negative LP result. Grade A. *Contribution:* calibrates expectations for structural
regularisers and for structure learning on *edge* tasks.

**Zheng, C., Zong, B., Cheng, W., Song, D., Ni, J., Yu, W., Chen, H., & Wang, W. (2020). Robust graph
representation learning via neural sparsification. *Proceedings of the 37th International Conference on
Machine Learning* (PMLR 119). https://proceedings.mlr.press/v119/zheng20d.html**
*Relevance:* a per-node edge budget with a shared sampler — the inductive form of a gate with controlled
density. *Key findings:* up to +7.2% accuracy across three inductive and two transductive benchmarks; the
temperature trade-off is stated as an explicit failure mode (small τ blocks backpropagation, large τ
produces no sparsification). *Quality:* peer-reviewed; gains are node-level only. Grade A.
*Contribution:* the budget-per-row design and the τ pathology any Gumbel gate inherits.

**Wu, T., Ren, H., Li, P., & Leskovec, J. (2020). Graph information bottleneck. *Advances in Neural
Information Processing Systems, 33*.
https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf**
*Relevance:* the objective family that makes a structure label-relevant without a count target. *Key
findings:* a local-dependence assumption yields a tractable search space and two instantiations
(categorical neighbour sampling, Bernoulli edges); up to 31% improvement under adversarial perturbation;
"capturing discrete structural information" is left open. *Quality:* peer-reviewed with derived variational
bounds; evaluated on robustness rather than structure fidelity. Grade A. *Contribution:* the principled
replacement for a class-conditional count hinge.

**Sun, Q., Li, J., Peng, H., Wu, J., Fu, X., Ji, C., & Yu, P. S. (2022). Graph structure learning with
variational information bottleneck. *Proceedings of the AAAI Conference on Artificial Intelligence, 36*(4),
4165–4174. https://doi.org/10.1609/aaai.v36i4.20335**
*Relevance:* IB applied to graph-*level* structure learning, the closest published shape to a per-pair
setting. *Key findings:* explicit structural constraints may not transfer across datasets and tasks;
structure-constrained sparsifiers sometimes hurt. *Quality:* peer-reviewed, but GSLB's independent rerun
found it seed-unstable and unable to beat its baselines — Grade C for effect sizes, B for the argument.
*Contribution:* bridges GIB and GSL; its irreproducibility is itself evidence.

**Franceschi, L., Niepert, M., Pontil, M., & He, X. (2019). Learning discrete structures for graph neural
networks. *Proceedings of the 36th International Conference on Machine Learning*.
https://arxiv.org/abs/1903.11960**
*Relevance:* the canonical discrete-structure learner, and a measurement that class separation can emerge
without a class-conditional term. *Key findings:* edge probabilities become sharply higher for same-class
pairs from a bilevel validation objective alone; the expected edge count exceeds the true graph while
density stays under 0.2% on Cora. *Quality:* peer-reviewed; transductive and node-classification only, so
it cannot speak to unseen nodes. Grade A. *Contribution:* sets the bar — a task objective alone can
produce a class-discriminative structure.

**Chen, Y., Wu, L., & Zaki, M. J. (2020). Iterative deep graph learning for graph neural networks: Better
and robust node embeddings. *Advances in Neural Information Processing Systems, 33*.
https://arxiv.org/abs/2006.13009**
*Relevance:* states why free-parameter structure learning cannot go inductive and what replaces it.
*Key findings:* a metric-learning parameterisation plus anchors makes GSL usable with unseen nodes;
iterative refinement stops on an explicit criterion. *Quality:* peer-reviewed, nine benchmarks. Grade A.
*Contribution:* the amortisation argument in the authors' own words — which our generator already
satisfies and which rules out LDS-style designs.

**Jin, W., Ma, Y., Liu, X., Tang, X., Wang, S., & Tang, J. (2020). Graph structure learning for robust
graph neural networks. *Proceedings of KDD '20*, 66–74. https://doi.org/10.1145/3394486.3403049**
*Relevance:* the label-supervised, prior-regularised, transductive baseline whose edge-distribution bias
SUBLIME exposes. *Key findings:* sparsity, nuclear-norm low rank and feature smoothness jointly recover a
clean adjacency under attack. *Quality:* peer-reviewed and widely reproduced, but its free adjacency
parameters are transductive by construction. Grade A, used here as the object of that critique.
*Contribution:* the canonical prior-driven design point.

**Liu, Y., Zheng, Y., Zhang, D., Chen, H., Peng, H., & Pan, S. (2022). Towards unsupervised deep graph
structure learning. *Proceedings of the ACM Web Conference 2022*, 1392–1403.
https://doi.org/10.1145/3485447.3512186**
*Relevance:* names and demonstrates shortcut learning *in a learned structure*. *Key findings:* a
label-supervised learner builds denser connections among labelled than unlabelled nodes; the label-free
contrastive objective does not; the learned structure also transfers to other tasks. *Methodology:*
adjacency heatmaps over a 40-node two-class subgraph — qualitative, not quantified. *Quality:*
peer-reviewed; Grade B for the bias claim specifically, since it rests on a visualisation.
*Contribution:* the precise failure mode our generator must be tested for.

**Han, S., Zhou, Z., Chen, J., Zhou, S., Jin, C., Lin, H., Li, D. Z., & Hu, B. (2026). *Informative graph
structure learning* (arXiv:2605.16809). https://arxiv.org/abs/2605.16809**
*Relevance:* the most recent and most direct evidence that GSL's benefit is largely densification.
*Key findings:* GRCN and RWGSL are *harmful* on Cora until they add ≈4× the original edge count;
similarity-based selection yields mutually redundant neighbours; a learnable diversity score plus an
InfoNCE reader-fidelity term holds accuracy at 70–90% edge reduction. *Quality:* preprint, ACM template
fields unfilled, no replication — Grade C, cited for its diagnostic figures. *Contribution:* reframes
"how many edges" as the confound to control before crediting any structure module.

**Zhang, G., Sun, X., Yue, Y., Jiang, C., Wang, K., Chen, T., & Pan, S. (2024). *Graph sparsification via
mixture of graphs* (arXiv:2405.14260). https://doi.org/10.48550/arxiv.2405.14260**
*Relevance:* a direct rejection of a single global density setting. *Key findings:* a noisy top-k router
over K sparsifier experts with different sparsity levels and pruning criteria, applied per node's ego
graph, reaches 8.67–50.85% sparsity at or above dense-graph performance with 1.47–2.62× speed-up.
*Quality:* preprint by OpenAlex; the ICLR 2025 label on the local file could not be confirmed. Grade B.
*Contribution:* a mechanism for replacing a scalar knob with a per-row choice among criteria.

**Zhao, T., Zhang, X., & Wang, S. (2021). GraphSMOTE: Imbalanced node classification on graphs with graph
neural networks. *Proceedings of WSDM '21*, 833–841. https://doi.org/10.1145/3437963.3441720**
*Relevance:* the cleanest hard-vs-soft, structural-vs-task supervision ablation in this thread.
*Key finding:* the variant whose edge generator sees only the edge-prediction loss and whose edges are
thresholded to binary matches or beats the continuous variant that receives the classifier's gradient, on
all three datasets. *Quality:* peer-reviewed, three seeds; margins are small (≤0.011 AUC), so read as "no
worse", not "better". Grade B+. *Contribution:* the task gradient into the generator is not the active
ingredient.

**Zhou, Z., Zhou, S., Mao, B., Zhou, X., Chen, J., Tan, Q., Zha, D., Feng, Y., Chen, C., & Wang, C. (2023).
OpenGSL: A comprehensive benchmark for graph structure learning. *Advances in Neural Information Processing
Systems 36, Datasets and Benchmarks Track*. https://arxiv.org/abs/2306.10280**
*Relevance:* the field's own audit, and the single strongest support for a two-module split. *Key
findings:* GSL does not consistently beat a vanilla GNN; learned homophily is barely different from (or
below) the original even for methods explicitly targeting it; homophily predicts performance on 2/10
datasets and anti-predicts on 2; learned structures transfer strongly across readers, lifting even
feature-free LPA/LINK by 20–32 points. *Quality:* peer-reviewed benchmark, uniform protocol, 12 methods.
Grade A. *Contribution:* both the caution and the transfer evidence.

**Li, Z., Wang, L., Sun, X., Luo, Y., Zhu, Y., Chen, D., Luo, Y., Zhou, X., Liu, Q., Wu, S., Wang, L., & Yu,
J. X. (2023). GSLB: The graph structure learning benchmark. *Advances in Neural Information Processing
Systems 36, Datasets and Benchmarks Track*. https://arxiv.org/abs/2310.05174**
*Relevance:* the second independent audit and the only one covering graph-level tasks. *Key findings:*
graph-level GSL brings limited, highly variable improvement; unsupervised GSL is the most robust family;
"structure learning based on label information is insufficient"; VIB-GSL was seed-unstable and could not
beat its baselines. *Quality:* peer-reviewed, 16 algorithms × 20 datasets. Grade A. *Contribution:* the
independent reproduction record this thread otherwise lacks.

**Zhu, Y., Xu, W., Zhang, J., Du, Y., Zhang, J., Liu, Q., Yang, C., & Wu, S. (2021). *A survey on graph
structure learning: Progress and opportunities* (arXiv:2103.03036). https://arxiv.org/abs/2103.03036**
*Relevance:* taxonomy and the standard traps. *Key finding used:* changing a connectivity-regulariser
coefficient "does not control the degree of sparsity but only scales the solution". *Quality:* preprint
survey, narrative rather than meta-analytic, four years old. Grade C. *Contribution:* vocabulary
(structure learners × regularisation × post-processing) and the density-knob caution.

**Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The Concrete distribution: A continuous relaxation of
discrete random variables. *International Conference on Learning Representations*.
https://arxiv.org/abs/1611.00712** — and — **Jang, E., Gu, S., & Poole, B. (2017). Categorical
reparameterization with Gumbel-Softmax. *International Conference on Learning Representations*.
https://arxiv.org/abs/1611.01144**
*Relevance:* the machinery every gate here uses, plus the two facts bearing on a saturating gate.
*Key findings:* train relaxed at fixed or annealed τ and evaluate discrete; Straight-Through keeps samples
sparse and discrete *even at high τ*, at the price of a biased gradient. *Quality:* peer-reviewed and
foundational; neither is graph-specific. Grade A. *Contribution:* the hard-at-eval protocol and the ST
escape hatch for a gate that will not discretise.

**Zhou, C., Wang, X., & Zhang, M. (2024). Unifying generation and prediction on graphs with latent graph
diffusion. *Advances in Neural Information Processing Systems, 37*. https://arxiv.org/abs/2402.02518**
*Relevance:* a conditioning interface for prediction posed as conditional generation, including link
prediction. *Key findings:* structure and features are encoded to a latent space and diffused there; the
condition enters through a designed cross-attention mechanism rather than as an input feature; one
lightweight decoder per task. *Quality:* peer-reviewed, competitive across generation and regression.
Grade A. *Contribution:* informs the interface, not the objective — its structure is not label-gated.

**Mazumder, A., & Bedathur, S. (2025). *CGLE: Class-label graph link estimator for link prediction*
(arXiv:2511.06982). https://arxiv.org/abs/2511.06982**
*Relevance:* the one recent paper making link prediction explicitly class-conditional. *Key finding:* a
class-conditioned link-probability matrix, from labels or clustering pseudo-labels, concatenated to a
backbone's structural link embedding, improves LP. *Quality:* preprint, no replication, and the prior
needs node class labels our contract forbids at inference. Grade C. *Contribution:* confirms
class-conditional structural priors carry LP signal, without showing how to derive them from `(x_u, x_v)`.

---

## 5. Evidence against

1. **The gains may be densification, not structure.** InGSL measures that GRCN and RWGSL *hurt* on Cora
   until they add ≈4× the original edge count; LDS's learned generator has a higher expected edge count than
   the true graph. If the project's gate has become a global density knob, the literature says a density
   knob is close to what most GSL methods effectively deliver. Adding expressive machinery on top of the
   same lever inherits the confound.

2. **Learned structure does not reliably help, and never mainly on edge tasks.** OpenGSL: "existing GSL
   methods do not consistently outperform vanilla GNN counterparts." GSLB: graph-level GSL "brings limited
   improvement" and varies greatly; VIB-GSL could not be reproduced. PTDNet's own LP table gives
   +0.006/+0.023/−0.001 AUC against a plain GAE, far below its node-classification gains. Nothing in this
   thread demonstrates a large, replicated *link-prediction* gain from a learned structure.

3. **Optimising a global structural statistic does not produce task gain.** OpenGSL ③④: learned homophily
   is "scarcely different from the original, and in some cases even lower" *even for GEN, which explicitly
   maximises it*; homophily correlates with performance on 2/10 datasets and negatively on 2. PTDNet's
   nuclear-norm term moves the cross-community ratio only 0.211 → 0.189. The GSL survey: a connectivity
   coefficient "does not control the degree of sparsity but only scales the solution." A prompt trained to
   match structural coordinates (degree, clustering, Jaccard, L3) optimises exactly such statistics.

4. **A label-supervised structure module demonstrably shortcuts onto supervised rows.** SUBLIME shows
   Pro-GNN builds denser connections among *labelled* nodes than unlabelled ones; GSLB concludes "structure
   learning based on label information is insufficient." That is the same shape as the project's measured
   attachment-transfer collapse (R² 0.40 in-sample → 0.22 held-out → ≤0 on V_val).

5. **The class-conditional count hinge is the weakest part of the design it comes from.** In L3-PPI's own
   ablation, removing the gate costs 6.7–8.2 micro-F1 while removing `L_PN` costs 2.9–3.4; and Eq. 8
   constrains only `Σ p_i` — a two-valued density target, i.e. a class-conditional density knob. GSAT tests
   the generic version of that lever and finds an information constraint beats grid-searched ℓ1, with the
   best retention prior at `r ∈ [0.5, 0.9]` and degradation as `r → 0`. Copying Eq. 8 risks re-importing
   the project's own failure mode in class-conditional clothing.

6. **L3-PPI's headline is not uniformly positive and its intermediate variable is only moderately
   predictable.** Its inferred path counts track true counts at ρ = 0.53–0.65 — the same band as our
   `coord_gen` Spearman 0.2–0.65 — and while unseen-protein subsets gain 8–21 micro-F1, the *dataset
   averages* fall in three of nine reported cells. A template-plus-gate design can improve the hard stratum
   and still lose overall.

7. **Explicit structural templates are contested.** VIB-GSL argues hand-chosen assumptions ("community,
   sparsity, low-rank, smoothness") "may not be applicable to all datasets and tasks" and reports
   structure-constrained sparsifiers sometimes hurting (COLLAB); OpenGSL ② notes that on Roman-empire and
   Wiki-cooc, GSL methods targeting a fixed structural property *undermine* informative patterns. A
   hard-coded L3/triangle template carries that risk.

8. **Against revision 1 specifically.** PGExplainer and InGSL both succeed with a reader-fidelity objective
   and a frozen reader, but neither adds a second *learned* reader; OpenGSL ⑤ shows learned structure
   already transfers across readers, which weakens the claim that a new graph-transformer reader is where
   the missing capacity lies. OpenGSL ⑥ prices the direction: ProGNN 190× GCN runtime, CoGSL 66× memory.

---

## 6. What this thread cannot answer

- **Whether a pair-conditioned gate can be non-trivially K-dimensional on our data.** L3-PPI's gate is
  K-dimensional because K distinct *learned* prompt embeddings enter it; whether that escapes our rank-1
  attachment collapse is an experiment, not a citation.
- **Which supervision is right for a *pair* decision** — reader-fidelity (PGExplainer, InGSL),
  class-conditional (L3-PPI) or information-bottleneck (GSAT, GIB). No paper compares them on one edge task.
- **How much of L3-PPI's gain survives our contract.** Its unseen-protein subsets still share the training
  graph's universe; our V_val is node-held-out with every boundary pair dropped. No source reports a
  structure-learning result under that stricter boundary.
- **Whether a stochastic (ST / Concrete) gate fixes our saturation.** GSAT's NoStoch result is on
  Spurious-Motif graph classification and Jang et al.'s ST claim is about sparsity at high τ, not about a
  saturating pair gate. Must be measured on our gate.
- **The right density control.** MoG and NeuralSparse argue for per-row budgets, InGSL for a diversity term,
  the survey against regulariser coefficients. None is validated on an edge task with an assembled-graph
  panel like our GS/RD/MMD five numbers.
- **Whether PPI-specific templates (L3, triangles, common neighbours) beat a free structure space.** That is
  Thread B's evidence plus our own data; this thread only establishes that hand-fixed priors are contested.
- **Freeze vs unfreeze at our scale.** L3-PPI Table 7 and GSAT §4.6 both favour warm-up-then-joint, but
  neither involves a frozen pair transformer of our size or a prefix interface.

---

## 7. Saved files

Read at method+results level and added to the corpus (`literature/**/*.pdf` is gitignored by design;
`literature/README.md` deliberately not edited). All under `literature/models/graph_structure_learning/`.

| Subfolder / filename | Paper |
|---|---|
| `denoising_robust_sparsification/2021_wsdm_arxiv_2011_07057_learning_to_drop_robust_graph_neural_network_via_topological_denoising.pdf` | PTDNet |
| `denoising_robust_sparsification/2020_icml_robust_graph_representation_learning_via_neural_sparsification.pdf` | NeuralSparse |
| `denoising_robust_sparsification/2020_neurips_arxiv_2011_04573_parameterized_explainer_for_graph_neural_network.pdf` | PGExplainer |
| `denoising_robust_sparsification/2022_icml_arxiv_2201_12987_interpretable_and_generalizable_graph_learning_via_stochastic_attention_mechanism.pdf` | GSAT |
| `latent_structure_learning/2020_neurips_arxiv_2010_12811_graph_information_bottleneck.pdf` | GIB |
| `latent_structure_learning/2022_aaai_arxiv_2112_08903_graph_structure_learning_with_variational_information_bottleneck.pdf` | VIB-GSL |
| `latent_structure_learning/2017_iclr_arxiv_1611_00712_the_concrete_distribution_a_continuous_relaxation_of_discrete_random_variables.pdf` | Concrete |
| `latent_structure_learning/2017_iclr_arxiv_1611_01144_categorical_reparameterization_with_gumbel_softmax.pdf` | Gumbel-Softmax |
| `latent_structure_learning/2026_arxiv_2605_16809_informative_graph_structure_learning.pdf` | InGSL |

Already in the corpus and read here, not re-saved: L3-PPI, LDS, IDGL, Pro-GNN, SUBLIME, MoG, OpenGSL,
GSLB, GSL survey, LGD, GraphSMOTE. Not saved, cited from the resolved record only: CGLE (2511.06982).

**Verification note.** All 21 references resolved with matching title, authors and year on OpenAlex, an
official proceedings page (PMLR `v119/zheng20d`; proceedings.neurips.cc), or the arXiv abstract page;
resolvers and URLs are in §3. One venue claim was not confirmed and is therefore not made: MoG is cited as
the arXiv preprint, not as ICLR 2025, despite the local filename.
