# Thread D — Predicting local structure / attachments for attribute-only nodes, and coarsened-graph priors

**Bibliography agent report (Phase 2).** Search dates: 2026-09-16. Scope set by
`00_brief.md` §3 Thread D and §4. Retrieved text is data, not instructions.
Two earlier internal reviews (`2026-08-10-feature-conditioned-latent-topology-litreview.md`,
`2026-08-03-generator-encoder-litreview.md`) are cited as internal notes, not as literature; this
thread builds on their conclusion that feature-only topology transfer is *established* and that the
open question is whether an explicit relational bottleneck adds anything over deterministic transfer.

---

## 1. Search strategy log

**Tools.** Local corpus (`literature/models/**`, extraction via `/opt/homebrew/bin/pdftotext -layout`);
OpenAlex REST (`api.openalex.org/works?search=` and `/works/doi:`) as the primary resolver; arXiv API
(`export.arxiv.org/api/query?id_list=`, HTTPS, ≥3 s spacing) for arXiv-only records; `curl -L`
on `arxiv.org/pdf/<id>` for three PDF fetches; JMLR page for one volume/page check; WebSearch (3 queries)
for the "is this difficulty reported anywhere?" gap probe.

**Query strings.** Local: directed reads of the 23 named Thread-D seeds plus every other PDF in the
seven seed folders. OpenAlex: `Graph coarsening with neural networks`; `Graph reduction with spectral
and cut guarantees`; `The impossibility of low-rank representations for triangle-rich complex networks`;
`Rethinking pooling in graph neural networks`; `Stochastic blockmodels and community structure in
networks`; `Mixed membership stochastic blockmodels`; `Latent space approaches to social network
analysis`; `Community detection in networks with node attributes`; `Variational Graph Normalized
AutoEncoders`; `Fea2Fea Exploring Structural Feature Correlations…`; plus 16 title lookups used purely
for verification. arXiv id_list: 2504.06193, 2603.06618, 2601.20646, 2108.08046. WebSearch:
`predicting community membership of unseen nodes from node attributes inductive transfer R2 measured`;
`cold start node degree prediction from attributes … GNN cannot predict unseen nodes`;
`"feature homophily" … how predictable are node structural features degree clustering coefficient from
node attributes regression`.

**PRISMA-style flow.**

| Stage | n |
|---|---|
| Records identified — local corpus (7 seed folders, every PDF) | 45 |
| Records identified — OpenAlex / arXiv / WebSearch | 34 |
| External records that duplicate a local PDF (verification lookups) removed | 16 |
| Records screened (title + abstract) | 63 |
| Excluded at screening | 32 |
| Full text / substantive sections assessed | 24 |
| Included on full text | 24 |
| Added on verified abstract + metadata (foundational model definitions) | 7 |
| **Total included** | **31** |

Of the 24 full-text reads, 21 were local PDFs and 3 were newly fetched and saved (§7).

**Coverage distribution advisory.**

```
DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: venue tier distribution
- Concentration: peer-reviewed proceedings/journal (T1) = 27/31 (87%)
- Advisory: a coverage signal, not a defect. The four preprints (graphon autoencoder,
  TGSBM, UPNA, GFT) are flagged T2 in the table and none carries a load-bearing claim alone.
- Search response: no expansion; the thread's decisive claims are theorems and peer-reviewed
  measurements, and preprint-only evidence is deliberately kept non-load-bearing.

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: time distribution
- Concentration: published 2018 or later = 27/31 (87%)
- Advisory: the four pre-2018 entries (Hoff 2002, Airoldi 2008, Karrer & Newman 2011,
  CESNA 2013) are the original statements of the block/latent-position priors this thread
  reasons about; they were added by deliberate snowballing, not found by recency search.
- Search response: no expansion; earlier literature would add history, not mechanism.

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational/ML method papers = 27/31 (87%)
- Advisory: inherent to the question (functional forms and measured transfer). The other
  four are statistical-model papers, added so the block-model limits are stated in their
  original form rather than only as reported by ML papers.
- Search response: no expansion.
```

Geographic distribution is not assessed: the included work reports benchmark datasets, not study sites.

---

## 2. Inclusion / exclusion criteria as applied

**Included** if the paper (a) specifies a *functional form* mapping node attributes to neighbourhood,
community or attachment structure, or (b) states a *limit* on what a block / edge-independent /
low-rank model can express about closed motifs, or (c) combines a block or community prior with a
neural pair scorer for link prediction, or (d) is a learned coarsening / pooling method whose
mechanism or failure mode transfers to a `K`-block virtual graph, or (e) reports a *measured* number
for structure recovered by an attribute-only model on nodes unseen at training.

**Excluded** if the method requires the query node's own edges at inference and has no attribute-only
variant; if it is primarily a Thread A/B/C topic (prompting mechanics, heuristic templates, graph
transformers); if it reports no number and defines no mechanism.

**Screened corpus seeds excluded, with reasons (no silent skips).**
`2103.08826` GraphSMOTE (synthesises minority-class nodes *inside* the observed graph);
`2111.12128` Feature Propagation (recovers missing features *given* the graph — the inverse problem);
`2507.11710` FLEX (conditions on observed structural statistics; Thread B/C);
`1706.02216` GraphSAGE, `2006.10637` TGN, `2307.01026` TGB, `1904.11547` Warm-Up (need incident edges or
interaction history); `1908.07078`, `1911.05954`, `1802.08773`, `2209.14734`, `2305.04111`, `2306.11412`,
`2202.02514`, `1803.00816` NetGAN, `2106.15239`, `2512.14241`, `1802.04687` (whole-graph generation or
realism evaluation with no attribute→attachment map; NetGAN is covered via Chanpuriya et al.);
`2402.02518` LGD (Thread C); the KD survey, SPKD and the two unnamed KD PDFs (covered by the KD entries).
Excluded at *full text*: none outright — `2411.06070` GFT and `2105.14244` graphon autoencoder were
narrowed to their template-vocabulary and limit statements respectively.

---

## 3. Mechanism extraction table

Grades: **T1** peer-reviewed proceedings/journal, **T2** preprint, **T3** grey. "R1" = revision 1
(graph-transformer reader replacing closed-form counting); "R2" = revision 2 (prompt structure from
prior knowledge / templates + a discriminative learned module). Verification column gives the resolver
and the URL actually resolved on 2026-09-16.

| Key | Venue/Year | WHY | HOW (structure / conditioning / objective / readout) | WHAT (numbers) | R1 / R2 | Grade | Verification |
|---|---|---|---|---|---|---|---|
| Chanpuriya et al. | NeurIPS 2021 | Why do neural graph generators under-produce triangles? | Any edge-independent `P ∈ [0,1]^{n×n}` (ER, SBM, VGAE, NetGAN/CELL); *overlap* `Ov(P)` bounds memorisation | Thm 1: `E[Δ] ≤ (2/3)·Ov(P)^{3/2}·V(P)^{3/2}`; Thm 6: `E[C(G)] = O(Ov^{3/2}·n/V^{1/2})`; Cora-ML 2,802 triangles → best model ≤1,461 | R1: a learned reader over `P` is not bound by Thm 1 / R2: closed motifs must enter as templates, not products of marginals | T1 | OpenAlex `doi:10.48550/arxiv.2111.00048` |
| Seshadhri et al. | PNAS 2020 | Can dot-product embeddings be triangle-rich? | `P = f(XX^T)`, rank `k`; jointly requires low degree + high local clustering | Any such embedding reproducing both needs rank **nearly linear in n** | R1/R2: `K = 64…1024` is provably too low-rank for clustering (cf. our measured R² 0.21→0.45 as K: 64→1024) | T1 | OpenAlex `doi:10.1073/pnas.1911030117` |
| Karrer & Newman | Phys. Rev. E 2011 | Plain SBM misfits broad degree distributions | Degree-corrected SBM: `λ_ij = θ_i θ_j ω_{g_i g_j}` | DC version "dramatically outperforms" the uncorrected one on real and synthetic nets | R1/R2: degree correction is the first fix for a `K`-block prior | T1 | OpenAlex `doi:10.1103/physreve.83.016107` |
| Airoldi et al. | JMLR 2008 | Nodes belong to several groups | Mixed-membership SBM: per-node Dirichlet `π_i`, per-interaction group draw, `B` block matrix | Canonical soft-membership generative form | R1/R2: our `a_u` is an MMSBM membership, but unnormalised and not per-interaction | T1 | OpenAlex `doi:10.48550/arxiv.0705.4485` |
| Hoff et al. | JASA 2002 | Blockmodels miss transitivity | Latent *position* model: `P(edge) = f(|z_i − z_j|, covariates)`, MCMC | Distance-based latent space fits the three benchmark networks better than a blockmodel and reproduces transitivity by the triangle inequality | R1/R2: a metric latent space yields transitivity where a block prior does not | T1 | OpenAlex `doi:10.1198/016214502388618906` |
| Xu et al. | arXiv 2021 | Interpretable, size-free graph generation | Graphon autoencoder: Chebyshev graphon filters encode, linear graphon factorisation decodes, Wasserstein/RAML objective | Transferable and size-generalising; the decoder is a step-function graphon — an edge-independent model | R1/R2: graphon priors inherit the Chanpuriya bound | T2 | OpenAlex `doi:10.48550/arxiv.2105.14244` |
| Yang et al. (TGSBM) | arXiv 2026 | Scalable, interpretable LP with overlapping communities | Expander-sparse graph transformer infers variational OSBM posteriors (stick-breaking `v`, binary `B`, Gaussian strengths `R`); **decoder replaces the bilinear block form by a 2-layer MLP then inner product**, "beyond simple bilinear forms" | Mean rank 1.6 under HeaRT; Cora MRR 39.69 vs LPFormer 39.42; ~6× faster training | R1: strong precedent — the leading block-model LP method abandons closed-form bilinear scoring for a learned readout | T2 | arXiv `2601.20646`; OpenAlex search |
| Salha-Galvan et al. | Neural Networks 2022 | Can one GAE do LP *and* community detection? | Modularity-based prior communities injected into message passing + a modularity regulariser alongside the reconstruction loss | Joint CD+LP without sacrificing either; gains largest where features are weak | R1/R2: the community prior is injected, not learned from the task loss | T1 | OpenAlex `doi:10.1016/j.neunet.2022.06.021` |
| Shchur & Günnemann (NOCD) | DLG@KDD 2019 | Overlapping community detection with GNNs | `F = GNN(A, X)` (or `MLP(X)`), Bernoulli–Poisson `p(A_uv)=1−exp(−F_u·F_v)`, balanced NLL | Attribute-only MLP NMI: **1.4 / 1.5** (Facebook 698/686) to **44.5–49.2** (co-authorship); GNN(X) 21.0–59.8. Inductive performance explicitly *not* assessed | R1/R2: attribute→community transfer is dataset-dependent and can be ≈0, even transductively | T1 | OpenAlex `doi:10.48550/arxiv.1909.12201` |
| Yang, McAuley & Leskovec (CESNA) | ICDM 2013 | Communities from edges **and** attributes | Joint generative model: community affiliations generate both edges (BigCLAM) and attributes (logistic per attribute) | More accurate and noise-robust than structure-only or attribute-only; linear runtime | R2: the established form is *joint* generation, not attribute→community regression | T1 | OpenAlex `doi:10.1109/icdm.2013.167` |
| Cai, Wang & Wang (GOREN) | ICLR 2021 | Coarse-graph edge weights are hand-set and suboptimal | **Partition fixed** by a classical coarsening algorithm; a GNN reads the induced subgraph `G_{û,v̂}` spanning two super-node clusters and emits the coarse edge weight; unsupervised Rayleigh-quotient loss | Large Eigenerror reductions across 6 coarsening algorithms; generalises to graphs **25× larger** and to unseen graphs; mean-pool+MLP baseline on the same input is sometimes catastrophic (−403 %, −230 %, −157 % at ratio 0.3) where GOREN is positive | R1: closest precedent — a **GNN over local structure beats pooled counting on identical input**, with a *fixed* partition | T1 | OpenAlex `doi:10.48550/arxiv.2102.01350`; PDF saved |
| Loukas | JMLR 2019, 20(116), 1–42 | Reduce a graph without destroying its properties | Restricted spectral approximation; sufficient conditions for a coarse graph to approximate a large one, nearly-linear algorithms | Coarse graphs of improved quality "often by a large margin" vs standard and advanced reduction | R1/R2: a principled fixed partition exists (cf. our spectral-64 ≈ k-means-1024) | T1 | JMLR page + OpenAlex `doi:10.48550/arxiv.1808.10650` |
| Ying et al. (DiffPool) | NeurIPS 2018 | Hierarchical graph representation | Soft assignment `S`; coarse graph `A^{(l+1)} = S^T A^{(l)} S`; **auxiliary link-prediction loss `L_LP = ‖A^{(l)} − S S^T‖_F`** + entropy regulariser on rows of `S` | Task loss alone is "difficult to push away from spurious local minima"; the side objective converges slower but gives better and more interpretable clusters | R1/R2: direct structural supervision of the assignment is a published necessity | T1 | arXiv `1806.08804` (https://arxiv.org/abs/1806.08804) + OpenAlex NeurIPS 2018 record |
| Bianchi et al. (MinCutPool) | ICML 2020 | Differentiable spectral clustering for pooling | Relaxed min-cut loss `L_c` + **orthogonality loss `L_o`**; `A_pool` de-diagonalised and degree-normalised | `L_c` alone has two degenerate minima: **all nodes to one cluster**, and **all nodes uniformly to all clusters**; `L_o` penalises both and enforces balanced sizes | R1/R2: our rank-1, near-uniform attachments are the textbook degenerate minima; the fix is an orthogonality/balance term | T1 | OpenAlex `doi:10.48550/arxiv.1907.00481` |
| Duval & Malliaros (HoscPool) | CIKM 2022 | Pooling ignores higher-order connectivity | Probabilistic **motif** spectral clustering: cluster on the triangle adjacency `A_M = A² ⊙ A`, relaxed motif normalised cut, mixed with the edge term | Cora NMI 0.502 vs MinCutPool 0.391 / DiffPool 0.308; on triangle-structured synthetics HoscPool/MSC reach 1.000 where SC=0.000, DiffPool=0.035, MinCutPool=0.043. Observes DiffPool/MinCutPool "do not really learn to optimise their cluster assignment matrix… producing degenerate solutions", which "would explain why random pooling performs on par" | R1/R2: motif adjacency is the concrete fix for a block model blind to triangles | T1 | OpenAlex `doi:10.1145/3511808.3557353` |
| Mesquita, Souza & Kaski | NeurIPS 2020 | Does local pooling matter? | Replace cluster-assignment pooling by randomisation or clustering on the *complement* graph | **No decrease in performance**; convolutions dominate the learned representation; "local pooling is not responsible for the success of GNNs" | R1/R2: strongest evidence against a learned coarsening being load-bearing | T1 | OpenAlex `doi:10.48550/arxiv.2010.11418` |
| Bojchevski & Günnemann (Graph2Gauss) | ICLR 2018 | Inductive attributed embeddings | Deep encoder attributes → Gaussian `(μ_i, Σ_i)`; **personalised ranking over hop-distance sets** `N_i^1 ≺ N_i^2 ≺ …` with a KL energy; unseen nodes embedded from attributes alone | `K ≤ 2` hops suffice; inductive for nodes with *no* connections | R2: supervise *ranking*, not magnitude — what our transfer measurement recommends | T1 | OpenAlex `doi:10.48550/arxiv.1707.03815` |
| Hao et al. (DEAL) | IJCAI 2020 | LP for nodes with attributes only | Two encoders (attribute-oriented, structure-oriented) + an alignment mechanism (tight/loose) with a ranking-motivated loss | Attribute-only inference after training; evaluated on PPI among others | R2: the established attribute-only baseline our arms must beat | T1 | OpenAlex `doi:10.24963/ijcai.2020/168` |
| Chatterjee et al. (UPNA) | arXiv 2023 | GNNs take topological shortcuts under power-law degrees | Learn `f(x_u, x_v) → P(edge)` from *pre-trained* node attributes only; usable to grow a graph by adding incoming nodes | 3×–34× improvement over SOTA on benchmark inductive splits | R2: a pair-attribute function learns part of the generative mechanism | T2 | OpenAlex `doi:10.48550/arxiv.2307.08877` |
| Zanardini & Serrano | Neurocomputing 2024 | Predict *all* links of a new isolated node | "Zero-shot out-of-graph all-links prediction": embed a candidate set `S` with a GNN, concatenate mean(`h_S`) with `h_t`, MLP binary classifier | MRR 0.04–0.16 on one 10 K citation sample; 0.29–0.89 on another, strongly dependent on candidate-set "purity" | R2: closest framing of our assembled-graph target; still needs identified candidates | T1 | OpenAlex `doi:10.1016/j.neucom.2024.128474` |
| Ahn & Kim (VGNAE) | CIKM 2021 | GAE embeddings for isolated nodes | Observes GAE/VGAE shrink the **norm** of isolated-node embeddings toward zero regardless of content; fixes it with `L2`-normalised graph convolution | Isolated-node embeddings "go close to zero regardless of their content features" | R1/R2: a zero-degree scale pathology — matches our V_val collapse in level and spread, not ranking | T1 | OpenAlex `doi:10.1145/3459637.3482215`; PDF saved |
| Liu et al. (LA-GNN) | ICML 2022 | Nodes with few neighbours aggregate too little | **One CVAE for all nodes** learns `p(x_neighbour | x_centre)`; at inference the graph is not needed to sample neighbour features; decoupled pre-training | Plug-and-play gains; authors state the model "can be applied to inductive settings" | R2: simplest published attribute→neighbourhood generator; node classification only | T1 | OpenAlex `doi:10.48550/arxiv.2109.03856` |
| Zhang et al. (GLNN) | ICLR 2022 | Remove graph dependency at inference | MLP student matches GNN soft labels | Under a production setting mixing transductive and inductive nodes, +12.36 % over stand-alone MLP on 7 datasets; inductive performance drops slightly as the inductive share grows | R1/R2: the deterministic-transfer floor to beat | T1 | OpenAlex `doi:10.48550/arxiv.2110.08727` |
| Guo et al. (LLP) | ICML 2023 | Logit/representation KD fails for LP | Relational KD: rank-based + distribution-based matching of an anchor node's context list | **Cold-start (isolated) Hits@20: teacher GNN 0.87–11.04, MLP 10.72–29.33, LLP 14.64–46.83** (ΔMLP +2.76…+19.15) | R1/R2: relational KD does transfer structure to isolated nodes, but from a low base and by a small margin | T1 | OpenAlex `doi:10.48550/arxiv.2210.05801` |
| Zheng et al. (Cold Brew) | ICLR 2022 | Strict cold start (no neighbours) | Teacher GNN + node-wise Structural Embedding; student MLP `ξ_1` predicts the teacher embedding, then **`ẽ_i = softmax(Θ_K(ê_i Ē^T)) Ē`** — top-`K` hard-thresholded *retrieval over the bank of all N training-node embeddings* — and `ξ_2` maps `[x_i, ẽ_i]` to the target | "Latent/virtual neighbourhood discovery"; FCR metric for architecture screening | R1/R2: the field's "invent the neighbourhood" solution is **retrieval**, which our contract forbids | T1 | OpenAlex `doi:10.48550/arxiv.2111.04840` |
| Qin et al. (EHDM) | LoG 2025 (arXiv 2504.06193v3) | Do stronger teachers give stronger MLP students? | Defines *teachable knowledge* `E[F | x_i, x_j]`; decomposition `KL(p‖E_F) = KL(p‖F) + information lost`; ensembles heuristic-distilled MLPs | Toy case: a near-perfect structure-dependent teacher has `NLL(F₁)=0.07` but its teachable knowledge scores `0.61`, **worse than a constant teacher's 0.60**. GNN4LP teachers are provably no more teachable than plain GNNs (Thm 3.2) | R1/R2: picking a teacher for accuracy on true structure is the wrong criterion | T1 | arXiv `2504.06193` (v3 PDF title verified) |
| Samy et al. (Graph2Feat) | WWW 2023 Companion | Fully inductive LP by KD | Student MLP matches teacher GNN representations; separates *semi*-inductive (new–existing) from *fully* inductive (new–new) | Fully inductive AUC: MLP 70.3–94.3, DEAL 73.8–96.2, Graph2Feat 79.1–96.0; transductive Graph2Feat 97.6–99.6 | R1/R2: attribute-only students already carry most of the achievable AUC; KD adds ~1–9 points | T1 | OpenAlex `doi:10.1145/3543873.3587596` |
| Yang et al. (VQGraph) | ICLR 2024 | Class space is too coarse a KD target | VQ-VAE structure-aware tokenizer labels each node's local substructure with a discrete code; student matches **soft code assignments** | SOTA GNN-to-MLP distillation in transductive *and* inductive settings over 7 datasets | R1/R2: a discrete structural vocabulary is a viable prompt target | T1 | OpenAlex `doi:10.48550/arxiv.2308.02117` |
| Deng et al. (CAZI-MBN) | ICLR 2026 | Zero-shot interaction prediction in multiplex biological networks | Sequence foundation-model embeddings + topology-aware Unified Graph Tokenizer teacher; **topology-agnostic sequence-only student** trained by MSE on latents + task loss | Transductive → zero-shot AUROC 0.715 → 0.671 (DGIdb), 0.812 → 0.791 (ChEMBL); 3.1–20.4 % over 13 baselines | R1/R2: closest task analogue; small zero-shot drop, but ceiling 0.67–0.79 AUROC and no graph-level metric | T1 | arXiv `2603.06618` (ICLR 2026) |
| Wang et al. (GFT) | NeurIPS 2024 | A transferable graph "vocabulary" | Computation trees as tokens; quantised prototypical tree tokens learned by a tree-reconstruction pre-training task; tasks unified as tree classification | Strong correlation between computation-tree similarity and transfer performance across graphs | R2: precedent for a fixed quantised template vocabulary instead of a free `K`-block space | T2 | OpenAlex `doi:10.52202/079017-3412`; arXiv 2411.06070 |
| Xie & Ying (Fea2Fea) | ICONIP 2021 (CCIS) | How correlated are structural features? | GNN pipelines predicting one structural feature from others in low dimension | **Degree is the easiest to predict; clustering coefficient is the most difficult**, reproduced on Planetoid, TUDataset and synthetic geometric graphs | R1/R2: corroborates our per-coordinate difficulty ordering, but structure→structure | T1 | OpenAlex `doi:10.1007/978-3-030-93736-2_19`; PDF saved |

---

## 4. Annotated bibliography (APA 7)

**Ahn, S. J., & Kim, M. (2021). Variational graph normalized auto-encoders. *CIKM '21*, 2827–2831.** https://doi.org/10.1145/3459637.3482215 — local: `literature/models/inductive_cold_start/2021_cikm_arxiv_2108_08046_…pdf`
*Relevance:* isolated-node pathology. *Key findings:* GAEs minimise similarity between unconnected pairs by shrinking the embedding **norm** of zero-degree nodes toward zero, discarding their content features; `L2`-normalised convolution fixes it. *Methodology:* analysis + LP benchmarks. *Quality:* short paper, small benchmarks. *Contribution:* names the failure mode our V_val collapse looks like — level and scale, not ranking.

**Airoldi, E. M., Blei, D. M., Fienberg, S. E., & Xing, E. P. (2008). Mixed membership stochastic blockmodels. *JMLR, 9*, 1981–2014.** https://doi.org/10.48550/arXiv.0705.4485
*Relevance:* the canonical soft-attachment prior. *Key findings:* per-node Dirichlet membership with a per-interaction group draw and a `K×K` block matrix. *Methodology:* variational EM. *Quality:* foundational; verified from abstract + metadata. *Contribution:* our `a_u ∈ [0,1]^K` is an unnormalised MMSBM membership; the published form normalises memberships and draws a block per interaction, which is precisely what removes the "global density knob" degree of freedom our gate collapsed into.

**Bianchi, F. M., Grattarola, D., & Alippi, C. (2020). Spectral clustering with graph neural networks for graph pooling. *ICML 2020*, PMLR 119, 874–883.** https://doi.org/10.48550/arXiv.1907.00481 — local: `graph_structure_learning/pooling_hierarchy_subgraph/2020_icml_…1907_00481….pdf`
*Relevance:* degenerate coarsenings. *Key findings:* the relaxed min-cut loss admits two degenerate optima — all nodes in one cluster, and all nodes uniformly in all clusters — which message passing *exacerbates*; an orthogonality term `L_o` penalises both and balances cluster sizes. *Methodology:* relaxation + clustering/classification benchmarks. *Quality:* strong, widely reproduced. *Contribution:* the project's uniform-at-init attention and rank-1 attachments are these textbook minima; the published remedy is an orthogonality/balance loss, not more capacity.

**Bojchevski, A., & Günnemann, S. (2018). Deep Gaussian embedding of graphs: Unsupervised inductive learning via ranking. *ICLR 2018*.** https://doi.org/10.48550/arXiv.1707.03815 — local: `inductive_cold_start/2018_iclr_…1707_03815….pdf`
*Relevance:* attribute→structure with a rank objective. *Key findings:* a personalised ranking over hop-distance sets, with `K ≤ 2` sufficient; unseen nodes are embedded from attributes alone, including nodes with no edges. *Methodology:* energy-based ranking with KL dissimilarity. *Quality:* strong, heavily cited. *Contribution:* the objective matches what actually transfers in our probes (within-node ranking, Spearman ≈ 0.2) and avoids the magnitude target that does not.

**Cai, C., Wang, D., & Wang, Y. (2021). Graph coarsening with neural networks. *ICLR 2021*.** https://doi.org/10.48550/arXiv.2102.01350 — saved: `graph_structure_learning/pooling_hierarchy_subgraph/2021_iclr_arxiv_2102_01350_….pdf`
*Relevance:* the closest precedent for revision 1. *Key findings:* the partition is **fixed** by a classical coarsening algorithm and only the coarse edge weights are learned, by a GNN reading the induced subgraph spanning the two super-node clusters; this generalises to unseen graphs 25× larger, and a mean-pool + MLP on the same input is sometimes catastrophically worse (−403 %). *Methodology:* unsupervised Rayleigh-quotient proxy for Eigenerror, six coarsening backends. *Quality:* rigorous, with convergence analysis. *Contribution:* separates "which blocks" (prior knowledge) from "how they relate" (learned reader) — the exact split our diagnosis implies.

**Chanpuriya, S., Musco, C., Sotiropoulos, K., & Tsourakakis, C. E. (2021). On the power of edge independent graph models. *NeurIPS 2021*.** https://doi.org/10.48550/arXiv.2111.00048 — local: `graph_generation_realism/2021_neurips_…2111_00048….pdf`
*Relevance:* the hard limit on block-model motifs. *Key findings:* for any edge-independent `P`, `E[Δ(G)] ≤ (2/3)·Ov(P)^{3/2}·V(P)^{3/2}`, with matching constructions; analogous bounds for `k`-cycles and for the global clustering coefficient; empirically, no tested model reproduces Cora-ML's 2,802 triangles (best ≤1,461). *Methodology:* proofs plus a simple overlap-tunable baseline. *Quality:* Tier 1, tight bounds. *Contribution:* our K-block mean-field model is edge-independent, so its inability to express clustering and Jaccard is a theorem, not a tuning failure.

**Chatterjee, A., Walters, R., Menichetti, G., & Eliassi-Rad, T. (2023). Disentangling node attributes from graph topology for improved generalizability in link prediction (UPNA).** arXiv:2307.08877. https://doi.org/10.48550/arXiv.2307.08877 — local: `inductive_cold_start/2023_arxiv_2307_08877_….pdf`
*Relevance:* pair-attribute functions as a generative mechanism. *Key findings:* learning `f(x_u, x_v)` from pre-trained attributes avoids GNNs' topological shortcuts under power-law degrees and yields 3×–34× improvements on inductive benchmarks; the learned function can grow a graph node by node. *Methodology:* unsupervised attribute pre-training + pairwise supervised learning. *Quality:* preprint; large claimed gains need the original splits to interpret. *Contribution:* supports the premise that the *pair* function, not a node-wise structural vector, is where inductive signal lives.

**Deng, A., Janarthanan, S., Sun, Y., Jing, Z., & Hu, P. (2026). Distilling and adapting: A topology-aware framework for zero-shot interaction prediction in multiplex biological networks. *ICLR 2026*.** arXiv:2603.06618 — local: `knowledge_distillation/2026_iclr_…2603_06618….pdf`
*Relevance:* nearest task analogue (proteins, sequence-only student). *Key findings:* teacher uses sequence + topology through a unified graph tokenizer; the student is topology-agnostic and trained by latent MSE; zero-shot AUROC 0.671 (DGIdb) and 0.791 (ChEMBL) against transductive 0.715 / 0.812. *Methodology:* five multiplex networks, 13 baselines, three seeds. *Quality:* Tier 1, but "zero-shot" pairing (one vs both endpoints unseen) is under-specified and no graph-level metric is reported. *Contribution:* sets the realistic ceiling for a sequence-only student and shows the field does not evaluate assembled topology.

**Duval, A., & Malliaros, F. D. (2022). Higher-order clustering and pooling for graph neural networks. *CIKM '22*, 426–435.** https://doi.org/10.1145/3511808.3557353 — local: `graph_structure_learning/pooling_hierarchy_subgraph/2022_cikm_…2209_03473….pdf`
*Relevance:* the concrete fix for a block model blind to triangles. *Key findings:* clustering on the triangle adjacency `A_M = A² ⊙ A` via a relaxed motif normalised cut raises Cora NMI to 0.502 (MinCutPool 0.391, DiffPool 0.308) and solves triangle-structured synthetics (1.000 vs 0.035–0.052) that edge-based clustering cannot see; the authors observe DiffPool/MinCutPool "do not really learn to optimise their cluster assignment matrix". *Methodology:* motif spectral relaxation + clustering and classification benchmarks. *Quality:* Tier 1, honest negative analysis. *Contribution:* if closed motifs must survive coarsening, coarsen on the motif adjacency.

**Guo, Z., Shiao, W., Zhang, S., Liu, Y., Chawla, N. V., Shah, N., & Zhao, T. (2023). Linkless link prediction via relational distillation. *ICML 2023*, PMLR 202.** https://doi.org/10.48550/arXiv.2210.05801 — local: `knowledge_distillation/2023_icml_…2210_05801….pdf`
*Relevance:* how much structure survives into an attribute-only student for isolated nodes. *Key findings:* on truly isolated cold-start nodes, the teacher GNN collapses to Hits@20 0.87–11.04 while the plain MLP reaches 10.72–29.33 and LLP 14.64–46.83; rank- and distribution-matching each contribute. *Methodology:* production/cold-start splits over 7 datasets. *Quality:* Tier 1, clean ablations. *Contribution:* relational distillation does transfer structure to attribute-only nodes — but the increment over a feature MLP is modest, which bounds what any Stage-II prompt can add.

**Hao, Y., Cao, X., Fang, Y., Xie, X., & Wang, S. (2020). Inductive link prediction for nodes having only attribute information. *IJCAI 2020*, 1209–1215.** https://doi.org/10.24963/ijcai.2020/168 — local: `inductive_cold_start/2020_ijcai_…2007_08053….pdf`
*Relevance:* the attribute-only inductive baseline. *Key findings:* two encoders (attribute-oriented, structure-oriented) plus tight/loose alignment trained with a ranking-motivated loss; at inference only the attribute encoder runs. *Methodology:* alignment objectives on attributed benchmarks including PPI. *Quality:* Tier 1. *Contribution:* alignment-to-a-structural-space is the established competitor; any prompt-based arm must beat it on the same split.

**Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *JASA, 97*(460), 1090–1098.** https://doi.org/10.1198/016214502388618906
*Relevance:* the non-block alternative that yields transitivity. *Key findings:* edge probability depends on distance in an unobserved social space; fits three standard networks better than a blockmodel and gives interpretable positions. *Methodology:* ML and Bayesian inference with MCMC. *Quality:* foundational; verified from abstract + metadata. *Contribution:* metric latent positions produce transitivity by the triangle inequality, so a distance-class coordinate is a principled companion to block-style counts — which is also the only relation coordinate our reader gets for free.

**Karrer, B., & Newman, M. E. J. (2011). Stochastic blockmodels and community structure in networks. *Physical Review E, 83*(1), 016107.** https://doi.org/10.1103/PhysRevE.83.016107
*Relevance:* the first fix for a plain block prior. *Key findings:* plain SBMs mis-handle broad degree distributions; adding node-level degree parameters "dramatically outperforms" the uncorrected model. *Methodology:* likelihood + heuristic inference on real and synthetic networks. *Quality:* foundational; verified from abstract + metadata. *Contribution:* a `K`-block prior for a PPI network with a power-law degree profile is mis-specified without degree correction, independently of attachment quality.

**Liu, S., Ying, R., Dong, H., Li, L., Xu, T., Rong, Y., Zhao, P., Huang, J., & Wu, D. (2022). Local augmentation for graph neural networks. *ICML 2022*, PMLR 162.** https://doi.org/10.48550/arXiv.2109.03856 — local: `neighbor_generation/2022_icml_…2109_03856….pdf`
*Relevance:* the simplest published attribute→neighbourhood generator. *Key findings:* a **single** CVAE learns `p(x_neighbour | x_centre)` across all nodes and, at generation time, needs only the node's own features; pre-training is decoupled from GNN training and the authors state it applies to inductive settings. *Methodology:* CVAE pre-training + plug-in augmentation; node-classification benchmarks. *Quality:* Tier 1, but no link-prediction or unseen-node structural evaluation. *Contribution:* shows the functional form is trainable; leaves the transfer question we care about untested.

**Loukas, A. (2019). Graph reduction with spectral and cut guarantees. *JMLR, 20*(116), 1–42.** https://jmlr.org/papers/v20/18-680.html
*Relevance:* a principled fixed partition. *Key findings:* restricted spectral approximation gives sufficient conditions under which a coarse graph preserves spectral and cut properties, with nearly-linear algorithms that beat standard reduction "often by a large margin". *Methodology:* theory + algorithms. *Quality:* Tier 1. *Contribution:* supplies the partition GOREN presumes and explains our measurement that spectral-`K=64` matches embedding k-means at `K=1024`.

**Mesquita, D., Souza, A. H., & Kaski, S. (2020). Rethinking pooling in graph neural networks. *NeurIPS 2020*.** https://doi.org/10.48550/arXiv.2010.11418
*Relevance:* the null result for learned coarsening. *Key findings:* replacing cluster-assignment pooling with randomisation or clustering on the *complement* graph causes **no** performance loss; convolutions dominate the learned representation. *Methodology:* controlled variants of representative pooling GNNs. *Quality:* Tier 1, well-controlled. *Contribution:* the sharpest warning that a learned `K`-block coarsening can be decorative; any design must include the random-partition control.

**Qin, Z., Zhang, S., Ju, M., Zhao, T., Shah, N., & Sun, Y. (2026). Weak models can be good teachers: A case study on link prediction with MLPs.** arXiv:2504.06193v3 (LoG 2025). — local: `knowledge_distillation/2025_log_…2504_06193….pdf`
*Relevance:* why a strong true-structure teacher need not help a feature-only student. *Key findings:* "teachable knowledge" `E[F | x_i, x_j]` and the decomposition `KL(p‖E_F) = KL(p‖F) + information lost`; a toy teacher with NLL 0.07 has teachable knowledge scoring 0.61, worse than a constant teacher (0.60); Thm 3.2 shows GNN4LP teachers are no more teachable than plain GNNs. *Methodology:* theory + benchmarks; ensembles of heuristic-distilled MLPs. *Quality:* Tier 1 (arXiv metadata title differs from the v3 PDF title; same ID, same authors — verified directly). *Contribution:* the general statement of our Stage I → Stage II failure.

**Salha-Galvan, G., Lutzeyer, J. F., Dasoulas, G., Hennequin, R., & Vazirgiannis, M. (2022). Modularity-aware graph autoencoders for joint community detection and link prediction. *Neural Networks, 153*, 474–495.** https://doi.org/10.1016/j.neunet.2022.06.021 — local: `graph_structure_learning/community_graph_modules/2022_neural_networks_…2202_00961….pdf`
*Relevance:* community prior + pair scorer. *Key findings:* injecting modularity-based prior communities into message passing, plus a modularity regulariser alongside reconstruction, improves community detection without hurting link prediction — especially where node features are weak. *Methodology:* theory of the message-passing scheme + benchmarks. *Quality:* Tier 1 journal. *Contribution:* the community prior is supplied, not discovered from the task loss — the opposite of our end-to-end coarsening.

**Samy, A. E., Kefato, Z. T., & Girdzijauskas, Š. (2023). Graph2Feat: Inductive link prediction via knowledge distillation. *WWW '23 Companion*, 805–812.** https://doi.org/10.1145/3543873.3587596 — local: `knowledge_distillation/2023_www_graph2feat_….pdf`
*Relevance:* the new–new (fully inductive) number. *Key findings:* fully inductive AUC — MLP 70.3–94.3, DEAL 73.8–96.2, Graph2Feat 79.1–96.0 — against transductive 97.6–99.6 for the same model. *Methodology:* seven homo/heterogeneous graphs, explicit semi- vs fully-inductive splits. *Quality:* Tier 1 companion track. *Contribution:* an attribute-only MLP already carries most of the achievable AUC; KD contributes ~1–9 points, which calibrates how large a *structural* contribution is plausible.

**Seshadhri, C., Sharma, A., Stolman, A., & Goel, A. (2020). The impossibility of low-rank representations for triangle-rich complex networks. *PNAS, 117*(11), 5631–5637.** https://doi.org/10.1073/pnas.1911030117
*Relevance:* the rank limit on attachments. *Key findings:* any dot-product embedding that produces both low degree and large clustering coefficients must have rank nearly linear in the number of vertices; SVD and node2vec empirically fail to capture triangle structure. *Methodology:* proof + empirical study of several embeddings. *Quality:* Tier 1, PNAS. *Contribution:* `K = 64…1024` is provably far too small for clustering, which is exactly the coordinate our block model cannot reproduce at any `K` we can afford.

**Shchur, O., & Günnemann, S. (2019). Overlapping community detection with graph neural networks. *DLG@KDD 2019*.** https://doi.org/10.48550/arXiv.1909.12201 — local: `graph_structure_learning/community_graph_modules/2019_dlg_…1909_12201….pdf`
*Relevance:* the only directly comparable attribute-only community number. *Key findings:* Bernoulli–Poisson `p(A_uv)=1−exp(−F_u·F_v)` with `F = GNN(A,X)`; the attribute-only ablation `F = MLP(X)` ranges from **NMI 1.4/1.5** (Facebook 698/686) to **44.5–49.2** (co-authorship). The authors explicitly defer inductive evaluation. *Methodology:* 11 datasets, 50 initialisations, MLP/free-variable ablations. *Quality:* workshop paper but carefully ablated. *Contribution:* even *transductively*, attribute→community membership can be near-zero; the literature reports no held-out-node number, so our Spearman ≈ 0.2 has no published counterexample.

**Wang, Z., Zhang, Z., Chawla, N. V., Zhang, C., & Ye, Y. (2024). GFT: Graph foundation model with transferable tree vocabulary. *NeurIPS 2024*.** arXiv:2411.06070. https://doi.org/10.52202/079017-3412 — local: `graph_structure_learning/community_graph_modules/2024_neurips_…2411_06070….pdf`
*Relevance:* a fixed template vocabulary. *Key findings:* computation trees as transferable tokens, quantised into a prototypical tree vocabulary by a reconstruction pre-training task; tree similarity correlates with transfer performance. *Methodology:* cross-domain pre-training + fine-tuning; LP protocol is transductive. *Quality:* Tier 1 venue, preprint copy. *Contribution:* precedent for replacing a free `K`-block space with a quantised template vocabulary — the structural half of revision 2.

**Xie, J., & Ying, R. (2021). Fea2Fea: Exploring structural feature correlations via graph neural networks. *ICONIP 2021 / CCIS*, 238–249.** https://doi.org/10.1007/978-3-030-93736-2_19 — saved: `link_prediction_structural/2021_iconip_arxiv_2106_13061_….pdf`
*Relevance:* per-coordinate difficulty ordering. *Key findings:* degree is the easiest structural feature to predict and the clustering coefficient the hardest, reproduced on Planetoid, TUDataset and synthetic geometric graphs. *Methodology:* low-dimensional GNN feature-to-feature regression. *Quality:* modest venue, small models. *Contribution:* independent corroboration of our coordinate ranking — but structure→structure, so it is an upper bound on what attribute→structure could achieve.

**Xu, H., Zhao, P., Huang, J., & Luo, D. (2021). Learning graphon autoencoders for generative graph modeling.** arXiv:2105.14244. https://doi.org/10.48550/arXiv.2105.14244 — local: `graph_generation_realism/2021_arxiv_2105_14244_….pdf`
*Relevance:* the size-free block prior. *Key findings:* Chebyshev graphon filters encode, a linear graphon factorisation decodes, trained by a Wasserstein/RAML objective; transfers across graph sizes. *Methodology:* functional-space autoencoding. *Quality:* preprint. *Contribution:* a graphon *is* an edge-independent kernel, so graphon-style priors inherit the Chanpuriya bound; size-transferability is what a coarse-graph prior buys, not motif fidelity.

**Yang, J., McAuley, J., & Leskovec, J. (2013). Community detection in networks with node attributes (CESNA). *ICDM 2013*, 1151–1156.** https://doi.org/10.1109/ICDM.2013.167
*Relevance:* the statistical form of "attributes indicate community". *Key findings:* a joint generative model in which community affiliations generate both edges and attributes is more accurate and more noise-robust than either modality alone, with linear runtime. *Methodology:* block-coordinate ascent on a joint likelihood. *Quality:* Tier 1; verified from abstract + metadata. *Contribution:* the established form is *joint* (communities → attributes), not a regression attributes → communities — a modelling choice our generator does not currently make.

**Yang, L., Tian, Y., Xu, M., Liu, Z., Hong, S., Qu, W., Zhang, W., Cui, B., Zhang, M., & Leskovec, J. (2024). VQGraph: Rethinking graph representation space for bridging GNNs and MLPs. *ICLR 2024*.** https://doi.org/10.48550/arXiv.2308.02117 — local: `knowledge_distillation/2024_iclr_…2308_02117….pdf`
*Relevance:* discrete structural targets for an attribute-only student. *Key findings:* a VQ-VAE structure-aware tokenizer assigns each node's local substructure a discrete code; distilling **soft code assignments** beats class-space distillation in both transductive and inductive settings on seven datasets. *Methodology:* tokenizer pre-training + KD. *Quality:* Tier 1. *Contribution:* evidence that a structural *vocabulary* is a better prompt target than a continuous coordinate vector — relevant to both revisions.

**Yang, Z., Zhao, S., Zhao, Z., & Chen, H. (2026). TGSBM: Transformer-guided stochastic block model for link prediction.** arXiv:2601.20646. — local: `graph_structure_learning/community_graph_modules/2026_arxiv_2601_20646_….pdf`
*Relevance:* block prior + neural pair scorer, current SOTA-competitive. *Key findings:* an expander-sparse graph transformer infers overlapping-SBM posteriors; the decoder **replaces the bilinear block form with a two-layer MLP followed by an inner product**, explicitly "beyond simple bilinear forms"; mean rank 1.6 under HeaRT, Cora MRR 39.69 vs LPFormer 39.42. *Methodology:* SGVB with Kumaraswamy/stick-breaking relaxations; five datasets, 13 baselines. *Quality:* preprint, not yet peer-reviewed. *Contribution:* the strongest positive read on revision 1 — the best current block-model link predictor keeps the community *prior* but abandons closed-form counting for a learned readout.

**Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). Hierarchical graph representation learning with differentiable pooling. *NeurIPS 2018*, 4805–4815.** arXiv:1806.08804. https://arxiv.org/abs/1806.08804 — local: `graph_structure_learning/pooling_hierarchy_subgraph/2018_neurips_…1806_08804….pdf`
*Relevance:* the coarse-graph construction our generator uses. *Key findings:* `A^{(l+1)} = S^T A^{(l)} S`; the task gradient alone cannot escape spurious minima, so an auxiliary link-prediction loss `‖A − S S^T‖_F` and a row-entropy regulariser are required; training is slower but clusters become interpretable. *Methodology:* end-to-end hierarchical pooling on graph classification. *Quality:* Tier 1, foundational. *Contribution:* direct supervision of the assignment matrix is a published necessity, matching the project's `w_attach` fix.

**Zanardini, D., & Serrano, E. (2024). Introducing new node prediction in graph mining: Predicting all links from isolated nodes with graph neural networks. *Neurocomputing, 602*, 128474.** https://doi.org/10.1016/j.neucom.2024.128474 — local: `inductive_cold_start/2024_arxiv_2401_05468_….pdf`
*Relevance:* the assembled-neighbourhood target. *Key findings:* defines "zero-shot out-of-graph all-links prediction" and solves it as set-to-node classification (mean of candidate-set embeddings concatenated with the target); MRR 0.04–0.16 on one 10 K citation sample and 0.29–0.89 on another, strongly dependent on candidate-set purity. *Methodology:* 5-layer SAGE, synthetic and citation samples. *Quality:* Tier 1 journal, but the "purity" design confounds difficulty with set composition. *Contribution:* the only framing that predicts a whole neighbourhood for an unseen node; it still needs identified candidate nodes, so it does not meet our contract.

**Zhang, S., Liu, Y., Sun, Y., & Shah, N. (2022). Graph-less neural networks: Teaching old MLPs new tricks via distillation. *ICLR 2022*.** https://doi.org/10.48550/arXiv.2110.08727 — local: `knowledge_distillation/2022_iclr_…2110_08727….pdf`
*Relevance:* the deterministic-transfer floor. *Key findings:* soft-label distillation lifts MLPs by 12.36 % on average in a production setting mixing transductive and inductive nodes; inductive accuracy degrades only slightly as the inductive share rises to 50 %. *Methodology:* seven datasets, noise and split-rate ablations. *Quality:* Tier 1. *Contribution:* fixes the baseline our structural arms must exceed, and shows that plain soft labels already capture much of the transferable teacher signal.

**Zheng, W., Huang, E. W., Rao, N., Katariya, S., Wang, Z., & Subbian, K. (2022). Cold Brew: Distilling graph node representations with incomplete or missing neighborhoods. *ICLR 2022*.** https://doi.org/10.48550/arXiv.2111.04840 — local: `knowledge_distillation/2022_iclr_…2111_04840….pdf`
*Relevance:* "invent the neighbourhood" for a strict cold-start node. *Key findings:* the teacher GNN is augmented with node-wise Structural Embeddings; the student's first MLP predicts the teacher embedding and then performs **latent/virtual neighbourhood discovery** `ẽ_i = softmax(Θ_K(ê_i Ē^T)) Ē` — a top-`K` hard-thresholded retrieval over the bank of all training-node embeddings — before a second MLP maps `[x_i, ẽ_i]` to the target; the FCR metric screens architectures for SCS. *Methodology:* public + proprietary e-commerce benchmarks. *Quality:* Tier 1. *Contribution:* the field's best answer to our problem is *retrieval over training nodes*, which our contract forbids — making a parametric virtual graph a genuinely under-explored alternative, and explaining why published cold-start numbers are not a fair reference point for us.

---

## 5. Evidence against

**(a) The block-model half of the current design cannot express the coordinates that matter — by theorem, not by tuning.** Chanpuriya et al. bound expected triangles and the global clustering coefficient of *any* edge-independent `P` by its overlap; Seshadhri et al. show any dot-product kernel producing both low degree and high clustering needs rank nearly linear in `n`. Our `deg / common / L3 / Jaccard / clustering` counts are exactly such an edge-independent mean-field construction over `K` blocks. The project's own measurement — oracle attachments at `K = 64` give clustering R² 0.21–0.27 and Jaccard 0.43–0.54, and `K = 1024` lifts clustering only to 0.45 — is the predicted behaviour. Raising `K` to the rank these theorems demand is not affordable.

**(b) Learned coarsening is frequently decorative.** Mesquita et al. show that replacing cluster-assignment pooling with *random* or complement-graph clustering costs nothing on standard benchmarks; Duval & Malliaros independently observe DiffPool and MinCutPool "do not really learn to optimise their cluster assignment matrix", producing degenerate solutions (MinCutPool NMI 0.086 / 0.026 on Photo / PC, with all nodes assigned to <10 % of clusters) and conclude this is *why* random pooling matches them. Bianchi et al. name the two degenerate optima — one cluster, or uniform assignment over all clusters — as intrinsic to the relaxed objective. This is the project's rank-1, near-uniform attachment field, and it argues that revision 1 must include a **random-partition and a shuffled-attachment control**, or the reader's gain cannot be attributed to the virtual graph.

**(c) The clearest statement that a strong true-structure teacher need not be teachable.** Qin et al.'s decomposition `KL(p‖E_F) = KL(p‖F) + information lost` and their toy case — a teacher with NLL 0.07 whose feature-conditioned projection scores 0.61, *worse than a constant teacher's 0.60* — is the general form of our Stage I (V_val AUPRC 0.958 on true coordinates) → Stage II (relation-coordinate label AUROC ≈ 0.5 on V_val) collapse. Their Thm 3.2 adds that a structurally more powerful teacher class is provably no more teachable. A graph-transformer reader `R` trained on true structure raises teacher accuracy; nothing in this literature suggests it raises teachability.

**(d) Attribute→community transfer is reported as weak or absent where it has been measured at all.** NOCD's attribute-only MLP reaches NMI **1.4 and 1.5** on two Facebook ego-networks and 11.7–22.1 on four more, all *transductive*; the authors defer inductive evaluation entirely. Fea2Fea reports clustering coefficient as the hardest structural feature to predict even from other structural features. No paper found in this thread reports an attachment R² or a within-node community-ranking correlation for **held-out nodes**, so the project's Spearman ≈ 0.2 / R² ≤ 0 has neither a counterexample nor a published floor — but the transductive low end is consistent with it.

**(e) Attribute-only students already capture most of the achievable pair signal, leaving little room for a structural prompt.** Graph2Feat's fully inductive AUC for a plain MLP is 70.3–94.3 against 79.1–96.0 for the distilled student; LLP's cold-start gain over a stand-alone MLP is +2.76 to +19.15 Hits@20 from an absolute base of 10.72–29.33; CAZI-MBN's sequence-only student on PPI-style multiplex data tops out at 0.67–0.79 zero-shot AUROC. None of these reports an assembled-graph metric, so the *topology* headline the project chases is unmeasured in the literature — but the edge-level headroom is demonstrably small.

**(f) The only published solution to "discover the neighbourhood of an attribute-only node" violates our contract.** Cold Brew's latent-neighbourhood discovery is a top-`K` retrieval over the bank of all training-node embeddings. Zanardini & Serrano's all-links prediction needs an identified candidate set. That the field solves this by retrieval is weak evidence that a purely parametric virtual graph is hard — and strong evidence that published cold-start numbers cannot be used as our reference point.

---

## 6. What this thread cannot answer

1. **Held-out-node attachment transfer.** No paper reports R² or Spearman for predicting a node's community/neighbourhood membership from attributes on nodes unseen at training. Only our own probes (`docs/tmp/virtual_prompt_diagnosis/`) measure it. Whether PPI sequence features are unusually weak, or whether this is universal, needs a cross-dataset replication of our own ridge probe.
2. **Whether a graph transformer over a soft `K`-node virtual graph escapes the edge-independence bound in practice.** The theorems bound the *generative* model's motif densities, not what a reader can compute from `P` itself; Thread C must supply the expressivity side, and only an experiment can say whether the reader recovers clustering/Jaccard information that the closed-form counts destroy.
3. **How to read out a *pair-relation* token from a coarse graph.** GOREN's per-super-edge GNN is the closest published readout, but it reads an observed induced subgraph between two clusters, not a pair of query endpoints attached to a virtual graph. No verified paper produces two endpoint tokens plus one relation token from one graph.
4. **Whether motif-adjacency coarsening (HoscPool) fixes our clustering/Jaccard ceiling.** It is measured for clustering NMI on observed graphs, never for a predicted-attachment block model. A cheap local rerun of the §3.4 oracle-attachment ceiling with blocks derived from `A² ⊙ A` would settle it without GPU time.
5. **Whether any of this survives the project's assembled-graph metrics.** No included paper reports GS/RD or degree/clustering/spectral MMD ratios; the KD and cold-start literature is edge-level only. The claim rules therefore cannot be satisfied from literature at all.
6. **Random-partition and shuffled-attachment controls.** Mesquita et al. make these mandatory, but no paper reports them for a *pair-conditioned* virtual graph feeding a frozen reader.

---

## 7. Saved files

Three PDFs were newly fetched, read, and saved (none was already in the corpus). `literature/README.md` was not edited.

- `literature/models/graph_structure_learning/pooling_hierarchy_subgraph/2021_iclr_arxiv_2102_01350_graph_coarsening_with_neural_networks.pdf` — Cai, Wang & Wang (ICLR 2021).
- `literature/models/inductive_cold_start/2021_cikm_arxiv_2108_08046_variational_graph_normalized_autoencoders.pdf` — Ahn & Kim (CIKM 2021).
- `literature/models/link_prediction_structural/2021_iconip_arxiv_2106_13061_fea2fea_exploring_structural_feature_correlations_via_graph_neural_networks.pdf` — Xie & Ying (ICONIP 2021).

Verification note: every included reference resolved on 2026-09-16 through OpenAlex (`api.openalex.org`), the arXiv API over HTTPS, a publisher DOI, or the JMLR volume page, with matching title, authors and year; the resolver and URL for each are in the table's last column. `2504.06193` is recorded with the title printed on the verified v3 PDF, which differs from the current arXiv metadata title for the same ID and authors. No reference is carried as "uncertain".
