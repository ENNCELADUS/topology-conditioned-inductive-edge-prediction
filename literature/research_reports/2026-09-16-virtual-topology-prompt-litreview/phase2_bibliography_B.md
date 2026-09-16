# Phase 2 Bibliography — Thread B: Structural templates and inductive biases for (PPI) link prediction

**Date:** 2026-09-16. **Deliverable:** annotated bibliography + search log. **Scope:** which templates carry signal
for PPI edge decisions; what is known about *learning* template weights/activity rather than counting; how methods
read a **pair** representation out of local structure; what happens when structure is **predicted or noisy**. No
cross-thread synthesis, no architecture proposal.

---

## 1. Search strategy log

**Tools:** local corpus `literature/models/**`; OpenAlex REST (search + `doi:` resolver, `mailto=` set); arXiv API
and `arxiv.org/pdf` (≥3 s spacing); Europe PMC (`search`, `fullTextXML`); WebSearch/WebFetch for paywalled venues.
Semantic Scholar unused (429 without key). All searches run 2026-09-16.

| # | Query / source | Tool | Inspected | Taken |
|---|---|---|---|---|
| 1 | `ls link_prediction_structural`, `kg_inductive_lineage` | local | 16 | 10 |
| 2 | "Network-based prediction of protein interactions" | OpenAlex | 3 | 1 |
| 3 | "Normalized L3-based link prediction in PPI networks" | OpenAlex | 3 | 1 |
| 4 | "Revisiting link prediction: a data perspective" | OpenAlex + arXiv | 3 | 1 |
| 5 | "Higher-order organization of complex networks" | OpenAlex | 3 | 1 |
| 6 | "Link prediction via higher-order motif features" | OpenAlex + WebSearch + arXiv 1902.06679 | 3 | 1 |
| 7 | "Stealing fire or stacking knowledge … link prediction" | OpenAlex + EPMC PMC9771718 | 2 | 1 |
| 8 | "Disentangling homophily, community structure, triadic closure" | OpenAlex + arXiv 2101.02510 | 2 | 1 |
| 9 | "Heuristic learning with graph neural networks" | OpenAlex + arXiv 2406.07979 | 2 | 1 |
| 10 | "Missing and spurious interactions … reconstruction" | OpenAlex | 2 | 1 |
| 11 | "Cracking the black box of deep sequence-based PPI prediction" | OpenAlex + WebFetch PMC10939362 | 3 | 1 |
| 12 | "Flaws in evaluation schemes for pair-input predictions" | OpenAlex + EPMC PMC3531800 | 3 | 1 |
| 13 | GNN-LP robustness to noisy / perturbed / missing edges | WebSearch → OpenAlex | 8 | 2 |
| 14 | MPLP; graphlet LP; PPI network bias; structural consistency | OpenAlex | 14 | 0 |

**PRISMA counts.** Identified 68 (local 16, OpenAlex 44, WebSearch 8). Duplicates (preprint↔proceedings) removed 9.
Screened 59; excluded at pass 1: 24 (node-classification robustness, generic GSL surveys, Thread C/D/E scope).
Full text assessed 35; excluded at pass 2: 11 (7 duplicative with a stronger row — Distance Encoding, P-GNN, MPLP,
GraIL, IGMC, Lü–Zhou survey, ULTRA; 4 not retrievable in full text). **Included 24.**

**Distributional skew advisory.** Venue tier: top-tier ML proceedings 12/24 (50 %) — below the 70 % threshold, no
advisory. Time: 18/24 from 2018–2026; the four pre-2018 entries are the field's foundational evaluation and
network-reconstruction results and are retained deliberately.

## 2. Inclusion / exclusion criteria (fixed before searching)

**Include** if the work (a) measures or theorises *which* structural pattern predicts edges, with PPI or a
biological network in scope; (b) specifies a mechanism turning local structure into a **pair-level** representation;
(c) reports what happens to a structural predictor when structure is incomplete, noisy, or itself predicted; or
(d) defines the protocol under which such claims are credible. **Exclude** node-/graph-level-only robustness,
GSL surveys (Thread E), coarsening/cold-start attribute→membership work (Thread D), prompt mechanisms (Thread A),
and any work whose full text could not be obtained where my claim would depend on numbers inside it.
**Verification (IRON RULE):** every entry resolves through OpenAlex (`doi:` or search) or the official proceedings
DOI with matching title/authors/year; resolver + URL recorded. Nothing carried as "uncertain". Peer-reviewed version
cited where both exist, with the arXiv ID given.

## 3. Mechanism extraction table

**T1** peer-reviewed, **T2** preprint. "R1" = graph-transformer reader over the virtual graph; "R2" =
prior-knowledge template structure + learned discriminative module. All DOIs resolved via OpenAlex, 2026-09-16.

| Key | Venue/Year | WHY | HOW | WHAT (numbers) | R1 | R2 | Grade | Verification |
|---|---|---|---|---|---|---|---|---|
| Kovács 2019 | Nat. Commun. | Triadic closure is the wrong PPI prior | Degree-normalised ℓ=3 path count `p_XY=Σ a_XU a_UV a_VY/√(k_U k_V)`; no learning | Jaccard/CN **anti**-correlates with interaction in all PPI nets; L3 precision 2–3× CN over 4 human data classes, 7× CN/PA and >2× CRA experimentally; ℓ=3 best of ℓ≤8; precision stable to removing 60–70 % of edges and to adding more false edges than true ones | justifies a small prompt graph | strongest single PPI prior | T1 | 10.1038/s41467-019-09177-y |
| Yuen 2023 | BMC Bioinf. | L3 under-uses the L3 principle | Decompose L3 into neighbourhood-similarity comparisons per intermediate pair (`f1/f2`); L3N/L3N′ normalise by interface compatibility not just degree | L3N′ most sensitive to perturbations of an ideal L3 graph, plain L3 **least**; CN- vs L3-predicted PPI pools overlap only 4 % (MINT-H) – 72 % (STRING-Y) vs 58–92 % within family | — | templates are complementary | T1 | 10.1186/s12859-023-05178-3 |
| Muscoloni 2022 | iScience | Is stacking better than one rule? | Unsupervised Cannistraci–Hebb automaton over L2/L3 local-community paths vs supervised stack of ~100 predictors, 550 networks | CHA beats stacking 14/18 mean-performance comparisons (median +64.5 %), 16/18 win-rate (median +383 %); adding CH+SPM+SBM to the stack gives "no clear advantage" | — | **against** naive stacking | T1 | 10.1016/j.isci.2022.105697 |
| Peixoto 2022 | Phys. Rev. X | Triangles: community or closure? | SBM + explicit triadic-closure layer; posterior labels each edge seminal vs closure; MDL selection | Separates the two at AUC-ROC 0.92 on a controlled synthetic; SBM/TC beats pure SBM on edge prediction for most empirical nets "sometimes substantially", never degrades; pure SBM invents spurious communities from triangles | — | closure template adds to a block prior | T1 | 10.1103/physrevx.12.011004 |
| Benson 2016 | Science | Higher-order organisation | Motif conductance; spectral clustering of a motif adjacency, with optimality guarantee | Exposes motif-level organisation invisible at edge level (abstract-level screening; no numeric claim drawn) | motif-weighted adjacency is a valid reader input | principled motif vocabulary | T1 | 10.1126/science.aad9029 |
| AbuOda 2020 | ECML PKDD | Do motifs beyond triangles help? | 29 normalised size-3/4/5 motif counts for the pair → GB/RF classifier | Up to **+10 pp accuracy** over CN/AA/Katz/PageRank/embeddings; inserting negatives beats removing positives by ≈3 pp; difficulty varies with pair shortest-path distance, so negatives must be distance-stratified | counts are the tabular baseline a reader must beat | multi-size motif vocabulary works | T1 | 10.1007/978-3-030-46150-8_25 |
| Mao 2024 | ICLR | Which factor governs which graph? | Latent-space theory + heuristic-overlap empirics over local (LSP) / global (GSP) structural and feature proximity (FP) | GSP helps only where LSP is deficient; LSP and FP are **incompatible** (high-CN pairs have low feature similarity); on **ogbl-ppa** CN dominates, feature heuristics are weak; GNN4LP models consistently lose to plain GraphSAGE on FP-dominated edges | reader must not fix one factor | motivates pair-conditional weighting | T1 | 10.48550/arXiv.2310.00793 |
| Wang 2023 (TC) | arXiv 2310.04612 | Which *nodes* predict well? | Topological Concentration = overlap of a node's local subgraph with its neighbours'; ATC approximation; TC-reweighted message passing | TC correlates **+82.1 %** more with node LP performance than degree, ~200 % wider under-performer gap; uncovers Topological Distribution Shift (new neighbours interact less with old ones) degrading test-time LP, varying per node | per-row context quality varies | favours pair-conditioned structure | T2 | 10.48550/arXiv.2310.04612 |
| Li 2023 (HeaRT) | NeurIPS D&B | Are LP gains real? | Re-tuned unified comparison + heuristic-related hard negative sampling | Under HeaRT simple heuristics/embeddings more often beat GNN4LP; citation2 best MRR ~90 → ~20; Shortest Path Hits@K = 0 with AUC > 99 % on ogbl-ppa; per-seed std up to 15.56 Hits@20 under the old setting | — | **against** gains measured on easy negatives | T1 | 10.48550/arXiv.2306.10453 |
| Aiyappa 2024 | arXiv 2405.14985 | Is the LP benchmark valid? | Uniform edge sampling favours high-degree positives; proposes degree-matched negatives | A degree-only null predictor reaches near-optimal AUC-ROC; degree correction cuts degree overfitting and improves learned structure | — | **against** crediting degree-bearing templates | T2 | 10.48550/arXiv.2405.14985 |
| Zhang 2018 (SEAL) | NeurIPS | Can heuristics be learned? | γ-decaying heuristic theory: any such heuristic is approximable from the h-hop enclosing subgraph, error decaying ≥ exponentially in h; DRNL labels + GNN | "the effective order of these high-order heuristics is not that high"; beats heuristics, latent-feature and embedding methods | small prompt graphs are theoretically sufficient | pair readout = labelled subgraph | T1 | 10.48550/arXiv.1802.09691 |
| Zhang 2021 (labeling trick) | NeurIPS | Why does GAE fail on pairs? | Label the target set *before* message passing; proves node-most-expressive GNN + labeling trick ⇒ most expressive node-set representation | Pooled node embeddings cannot separate automorphic nodes: `p(u,v)=p(u,w)` when `v≅w`, even at different distances | **relation readout must be target-conditioned inside the computation** | — | T1 | 10.48550/arXiv.2010.16103 |
| Yun 2021 (Neo-GNN) | NeurIPS | Learn overlap, don't count it | Learned structural-feature generator `F_θ(A)` + learned per-hop weights `Σ_l β^{l-1}A^l`; combined with a feature GNN | Generalises CN/AA/RA to multi-hop overlap; SOTA on four OGB LP datasets | — | canonical "learn the weights" | T1 | 10.48550/arXiv.2206.04216 |
| Chamberlain 2023 (ELPH/BUDDY) | ICLR | Subgraph GNNs are redundant | MinHash/HyperLogLog **sketches** as messages; structure feature = node counts per (d_u,d_v) hop-pair | Provably more expressive than MPNNs; matches/beats subgraph GNNs orders of magnitude faster | reader's job reduces to estimating a small count vector | (d_u,d_v) table *is* a template vocabulary | T1 | 10.48550/arXiv.2209.15486 |
| Wang 2024 (NCN/NCNC) | ICLR | Structure is incomplete at test time | MPNN-then-SF: pool MPNN states over common neighbours; NCNC *predicts* missing CN links and re-runs NCN on the **soft-weighted** completed graph (`P_uij=Â_iu`) | Incompleteness cuts CN counts and shifts train/test CN distributions, degrading even non-learnable CN; completion lifts Pubmed 79.05→81.29, DDI 82.32→84.11, Citeseer 91.56→93.47, PPA 61.19→61.42; authors warn "weak models may not accurately recover the unobserved common neighbor structure" | **precedent for reading a soft predicted structure** | completion is itself a learned module | T1 | 10.48550/arXiv.2302.00890 |
| Shomer 2024 (LPFormer) | KDD | One pairwise encoding does not fit all links | Attention over a PPR-thresholded context set `N̂(a,b)`; symmetrised RPE `MLP(ppr(a,u),ppr(b,u))`; score = `h_a⊙h_b ‖ s(a,b) ‖ counts(CN,1-hop,>1-hop)` | SOTA on 5/6 datasets and most consistent across them; beats NCNC/BUDDY where the dominant LP factor differs; faster than NCNC on dense graphs | **graph transformer emitting an explicit relation token beside endpoint states** | attention = learned per-pair template activity | T1 | 10.1145/3637528.3672025 |
| Zhang 2024 (HL-GNN) | KDD | Unify local and global heuristics | All heuristics = weighted adjacency products; learn `β^(l)` over ~20 layers, few parameters | Recovers the right template: synthetic triangular graph → `β^(2)` largest, hexagonal → `β^(5)`; Cora/Citeseer → `β^(0)` (features) dominant; ogbl-collab/ddi → `β^(2)` dominant with **negative** weights at large `l`; orders of magnitude faster than SEAL/NBFNet/Neo-GNN | — | **best evidence that template weights are learnable and interpretable** | T1 | 10.1145/3637528.3671946 |
| Zhu 2021 (NBFNet) | NeurIPS | Path counting as a learned semiring | Generalised Bellman–Ford with learned INDICATOR/MESSAGE/AGGREGATE (boundary, ×, +); representations conditioned on the source node | Covers Katz, PPR, graph distance; large margins transductive **and inductive** | source-conditioned readout is a second pair-readout design | learned path weighting generalises L3 | T1 | 10.48550/arXiv.2106.06935 |
| Zhou 2023 (RGIB) | NeurIPS | Edge noise is bilateral | Noise perturbs input topology **and** label → degradation + representation collapse; Robust Graph Information Bottleneck decouples I(topology;label;representation); RGIB-SSL / RGIB-REP | Verified on 6 datasets × 3 GNNs across noise regimes | names the failure mode of a predicted-structure reader | — | T1 | 10.48550/arXiv.2311.01196 |
| Zhu 2024 (SpotTarget) | WSDM | Target-link inclusion distorts LP | Exclude training target edges incident to a low-degree node; exclude all test target edges | Identifies overfitting, distribution shift and implicit test leakage, concentrated on low-degree nodes | — | validates dropping the queried partner from structural targets | T1 | 10.1145/3616855.3635786 |
| Guimerà 2009 | PNAS | Observed networks are wrong | SBM ensemble → posterior reliability of every observed/absent pair | Identifies missing **and** spurious interactions; reconstructions give true-network property estimates "more accurate than those provided by the observations themselves" (abstract-level screening) | an inferred graph can beat the observed one as input | — | T1 | 10.1073/pnas.0908366106 |
| Zheng 2025 (PRING) | NeurIPS D&B | Pairwise AUC ≠ network recovery | 21,484 proteins / 186,818 interactions, redundancy- and leakage-controlled; BFS/DFS/RW subgraphs; GS↑, RD→1, degree/clustering/spectral MMD↓ | All four model families reconstruct **overly dense** graphs; Human BFS GS 0.18–0.45, RD 1.13–4.39, clustering MMD 11.4–40.5; "classification metrics cannot completely reflect a model's ability to recover network structure" | — | the metric set any template claim must report | T1 | 10.52202/085713-3131 |
| Bernett 2024 | Brief. Bioinform. | Do sequence PPI models learn biology? | KaHIP partition of a SIMAP2 similarity network into blocks; INTRA₀/INTRA₁/INTER splits; CD-HIT 30 %; degree-preserving negatives; rewiring controls | "models learn solely from sequence similarities and node degrees"; degree-only prediction reaches ~87–92 % on HUANG/PAN; on the unbiased gold standard "performances were random for all methods" (best 56 %) | — | **degree is a shortcut, not a template** | T1 | 10.1093/bib/bbae076 |
| Park 2012 | Nat. Methods | Pair-input evaluation is biased | Partition test pairs by component overlap: C1 (both proteins seen), C2 (one), C3 (neither) | All 7 PPI methods differ significantly across classes; C1 is **>99 %** of typical CV test sets but only **19.2 %** of the human pair population (C2 49.2 %, C3 31.6 %) | — | template claims must be reported per C-class | T1 | 10.1038/nmeth.2259 |

## 4. Annotated bibliography (APA 7)

**AbuOda, G., De Francisci Morales, G., & Aboulnaga, A. (2020). Link prediction via higher-order motif features.
*LNCS, 11906*, 412–429.** https://doi.org/10.1007/978-3-030-46150-8_25 (arXiv:1902.06679). PDF:
`literature/models/link_prediction_structural/2020_ecml_pkdd_arxiv_1902_06679_link_prediction_via_higher_order_motif_features.pdf`
*Relevance:* cleanest test of "more templates than triangles". *Findings:* 29 normalised size-3/4/5 motif counts
give up to +10 pp accuracy over CN/AA/Katz/PageRank and embeddings; inserting negatives into the graph beats
removing positives by ≈3 pp; difficulty varies with pair shortest-path distance. *Method:* supervised classification,
balanced sets, 10-fold CV, Arabesque counting. *Quality:* strong mechanism, no PPI graph. *Contribution:* motif
vocabulary plus two dataset-construction rules any template study must respect.

**Aiyappa, R., Wang, X., Kim, M., Seckin, O. C., Yoon, J., Ahn, Y.-Y., & Kojaku, S. (2024). *Implicit degree bias in
the link prediction task* (arXiv:2405.14985).** https://doi.org/10.48550/arXiv.2405.14985. PDF:
`…/link_prediction_structural/2024_arxiv_2405_14985_….pdf` *Relevance:* how much of a "structural" gain is just
degree. *Findings:* uniform edge sampling makes a degree-*k* node *k*× likelier among positives, so a degree-only
null predictor is near-optimal; a degree-corrected benchmark aligns better with recommendation and reduces degree
overfitting in training. *Method:* sampling analysis + benchmark reconstruction. *Quality:* Tier 2, analytic and
reproducible. *Contribution:* the control every degree-bearing coordinate needs.

**Benson, A. R., Gleich, D. F., & Leskovec, J. (2016). Higher-order organization of complex networks. *Science,
353*(6295), 163–166.** https://doi.org/10.1126/science.aad9029 *Relevance:* reference point for motif-level
structure as a first-class object. *Findings:* a motif-conductance framework with optimality guarantees scales to
billions of edges and exposes organisation invisible at the edge level. *Method:* spectral clustering on a motif
adjacency. *Quality:* Tier 1 but **abstract-level screening only** (paywalled); no numeric claim drawn.
*Contribution:* legitimises treating a motif-weighted adjacency, not the raw one, as the object a model reads.

**Bernett, J., Blumenthal, D. B., & List, M. (2024). Cracking the black box of deep sequence-based protein–protein
interaction prediction. *Briefings in Bioinformatics, 25*(2), bbae076.** https://doi.org/10.1093/bib/bbae076
*Relevance:* strongest negative result for sequence-only PPI prediction — our exact input contract. *Findings:*
random pair splits leak; models "learn solely from sequence similarities and node degrees"; a degree-only rule
reaches ~87–92 % on HUANG/PAN; on their leakage-controlled gold standard every method is at chance (best 56 %).
*Method:* KaHIP block partition of a sequence-similarity network, CD-HIT 30 %, degree-preserving negatives, rewiring
ablations. *Quality:* Tier 1, open access, code released. *Contribution:* the bar an attribute-only PPI predictor
must clear before any topology claim is credible.

**Chamberlain, B. P., Shirobokov, S., Rossi, E., Frasca, F., Markovich, T., Hammerla, N., Bronstein, M. M., &
Hansmire, M. (2023). Graph neural networks for link prediction with subgraph sketching. *ICLR 2023*
(arXiv:2209.15486).** https://doi.org/10.48550/arXiv.2209.15486. PDF: `…/2023_iclr_arxiv_2209_15486_….pdf`
*Relevance:* what a pair reader minimally needs. *Findings:* MPNNs fail at LP because they cannot count triangles
and cannot distinguish automorphic nodes; ELPH recovers subgraph-GNN accuracy by passing MinHash/HyperLogLog
sketches, BUDDY precomputes them, both orders of magnitude faster. *Method:* expressiveness proofs + LP benchmarks.
*Quality:* Tier 1. *Contribution:* node counts per (d_u,d_v) hop-pair form a compact, learnable template vocabulary.

**Guimerà, R., & Sales-Pardo, M. (2009). Missing and spurious interactions and the reconstruction of complex
networks. *PNAS, 106*(52), 22073–22078.** https://doi.org/10.1073/pnas.0908366106 *Relevance:* founding result that
an inferred graph can beat the measured one. *Findings:* an SBM-ensemble reliability score identifies both missing
and spurious interactions, and reconstructions give true-network property estimates more accurate than the
observations themselves. *Method:* Bayesian model averaging over block partitions. *Quality:* Tier 1;
**abstract-level screening only** (PNAS and PMC both blocked), so no internal numbers cited. *Contribution:*
motivates predicting structure rather than trusting it.

**Kovács, I. A., Luck, K., Spirohn, K., Wang, Y., Pollis, C., Schlabach, S., … Barabási, A.-L. (2019).
Network-based prediction of protein interactions. *Nature Communications, 10*, 1240.**
https://doi.org/10.1038/s41467-019-09177-y. PDF:
`…/link_prediction_structural/2019_nature_communications_network_based_prediction_of_protein_interactions.pdf`
*Relevance:* the L3 principle itself. *Findings:* Jaccard/CN similarity *anti*-correlates with interaction in every
PPI network tested (7 organisms, 4 assay types) — paralogs share interfaces rather than complementary ones;
degree-normalised L3 gives 2–3× CN precision computationally, 7× CN/PA experimentally and >2× CRA; ℓ=3 is optimal
among ℓ≤8; precision is stable to removing 60–70 % of edges and to adding more false edges than true ones. *Method:*
Monte-Carlo CV on four human interactome classes plus ~3,000 Y2H tests and an independent HI-III screen. *Quality:*
Tier 1, wet-lab validated. *Contribution:* the best-evidenced PPI template prior and the only direct robustness curve
for it.

**Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang, J., & Yin, D. (2023). Evaluating graph neural
networks for link prediction: Current pitfalls and new benchmarking. *NeurIPS 2023 D&B* (arXiv:2306.10453).**
https://doi.org/10.48550/arXiv.2306.10453. PDF: `…/2023_neurips_arxiv_2306_10453_….pdf` *Relevance:* calibrates
every template-gain number here. *Findings:* retuning changes rankings (Neo-GNN collab 57.52→66.13); per-seed std
reaches 15.56 Hits@20; AUC contradicts ranking metrics (Shortest Path: Hits@K = 0, AUC > 99 % on ogbl-ppa); under
hard negatives (HeaRT) simple baselines more often win, citation2 MRR collapses ~90 → ~20, variance drops > 85 % on
the Planetoid graphs. *Method:* unified splits, shared HPO, new sampler. *Quality:* Tier 1. *Contribution:* hard
negatives are the honest setting for template comparison.

**Mao, H., Li, J., Shomer, H., Li, B., Fan, W., Ma, Y., Zhao, T., Shah, N., & Tang, J. (2024). Revisiting link
prediction: A data perspective. *ICLR 2024* (arXiv:2310.00793).** https://doi.org/10.48550/arXiv.2310.00793. PDF:
`…/link_prediction_structural/2024_iclr_arxiv_2310_00793_….pdf` *Relevance:* the theory of which heuristic suits
which graph, with a protein dataset in the panel. *Findings:* three factors (local structural, global structural,
feature proximity); GSP helps only where LSP is deficient; LSP and FP are incompatible — high-common-neighbour pairs
have low feature similarity; on ogbl-ppa CN dominates and feature heuristics are weak; GNN4LP models consistently
underperform plain GNNs on FP-dominated edges. *Method:* latent-space model + heuristic-overlap analysis on six
benchmarks. *Quality:* Tier 1. *Contribution:* a principled reason to make template weighting pair-conditional.

**Muscoloni, A., & Cannistraci, C. V. (2022). "Stealing fire or stacking knowledge" by machine intelligence to model
link prediction in complex networks. *iScience, 25*(12), 105697.** https://doi.org/10.1016/j.isci.2022.105697
*Relevance:* the sharpest warning against template stacking. *Findings:* the unsupervised Cannistraci–Hebb automaton
over L2/L3 local-community paths beats a supervised stack of ~100 predictors in 14/18 mean-performance comparisons
(median +64.5 %) and 16/18 win-rate comparisons (median +383 %) over 550 networks; adding CH/SPM/SBM features to the
stack gives no clear advantage. *Method:* 550-network re-analysis, 10 repetitions, precision/AUC-PR/AUC-mROC/NDCG/MCC,
permutation tests. *Quality:* Tier 1, reproducible. *Contribution:* one good rule can dominate learned combination.

**Park, Y., & Marcotte, E. M. (2012). Flaws in evaluation schemes for pair-input computational predictions. *Nature
Methods, 9*(12), 1134–1136.** https://doi.org/10.1038/nmeth.2259 *Relevance:* defines the split regime a
node-held-out protocol sits in. *Findings:* test pairs partition into C1 (both proteins seen), C2 (one), C3
(neither); all seven PPI methods differ significantly across classes; C1 is > 99 % of typical CV test sets but only
19.2 % of the human pair population (C2 49.2 %, C3 31.6 %). *Method:* large-scale yeast and human PPI experiments
plus a population count over HIPPIE. *Quality:* Tier 1 with extensive supplement. *Contribution:* performance must
be reported per C-class or it does not generalise.

**Peixoto, T. P. (2022). Disentangling homophily, community structure, and triadic closure in networks. *Physical
Review X, 12*(1), 011004.** https://doi.org/10.1103/physrevx.12.011004 (arXiv:2101.02510). PDF:
`…/graph_structure_learning/community_graph_modules/2022_physical_review_x_arxiv_2101_02510_….pdf` *Relevance:*
whether a block model can stand in for closed motifs. *Findings:* closure-generated triangles are routinely mistaken
for community structure by a pure SBM; an SBM with an explicit triadic-closure layer separates seminal from closure
edges (AUC-ROC 0.92 on a controlled synthetic) and improves edge prediction on most empirical networks "sometimes
substantially", never degrading it. *Method:* generative model + MDL inference, precision/recall reconstruction
including a yeast interactome (N=1004, E=8319). *Quality:* Tier 1. *Contribution:* an explicit closure template is
worth adding on top of a block prior and is cheap when unused.

**Shomer, H., Ma, Y., Mao, H., Li, J., Wu, B., & Tang, J. (2024). LPFormer: An adaptive graph transformer for link
prediction. *KDD '24*, 2686–2698.** https://doi.org/10.1145/3637528.3672025 (arXiv:2310.11009). PDF:
`…/2024_kdd_arxiv_2310_11009_….pdf` *Relevance:* closest published design to a graph transformer emitting a relation
token. *Findings:* attention over a PPR-thresholded context set with a symmetrised `MLP(ppr(a,u),ppr(b,u))`
positional encoding learns a *per-link* pairwise encoding; SOTA on 5/6 datasets and the most consistent method; the
score concatenates `h_a⊙h_b`, the pair encoding `s(a,b)`, and explicit CN/1-hop/>1-hop counts. *Method:* six
benchmarks, PPR-threshold ablations, factor-stratified analysis. *Quality:* Tier 1. *Contribution:* endpoint product,
learned relation token and raw counts are complementary — the relation readout is not derivable from pooled endpoint
states.

**Wang, X., Yang, H., & Zhang, M. (2024). Neural common neighbor with completion for link prediction. *ICLR 2024*
(arXiv:2302.00890).** https://doi.org/10.48550/arXiv.2302.00890. PDF: `…/2024_iclr_arxiv_2302_00890_….pdf`
*Relevance:* the only entry that *predicts* structural context and then reads it. *Findings:* incompleteness reduces
common-neighbour counts and shifts their train/test distribution, degrading even the non-learnable CN baseline; NCNC
softly completes the CN set with predicted edge probabilities (`P_uij=Â_iu`) and re-applies NCN, lifting Pubmed
79.05→81.29, DDI 82.32→84.11, Citeseer 91.56→93.47, PPA 61.19→61.42; the authors caution that weak completion models
will not recover the structure. *Method:* MPNN-then-SF architecture, distribution visualisations, ablations.
*Quality:* Tier 1. *Contribution:* soft, fractional, model-predicted structure is a working input to a pair readout,
bounded by the completion model's quality.

**Wang, Y., Zhao, T., Zhao, Y., Liu, Y., Cheng, X., Shah, N., & Derr, T. (2023). *A topological perspective on
demystifying GNN-based link prediction performance* (arXiv:2310.04612).** https://doi.org/10.48550/arXiv.2310.04612.
PDF: `…/2023_arxiv_2310_04612_….pdf` *Relevance:* why per-row structural context quality varies. *Findings:*
Topological Concentration correlates 82.1 % more with node LP performance than degree and separates under-performing
nodes ~200 % better; Topological Distribution Shift — new neighbours interacting less with old ones — degrades
test-time LP by a node-dependent amount; TC-guided edge reweighting helps but is flagged as non-causal. *Method:*
metric definition, scalable approximation, correlation study. *Quality:* Tier 2. *Contribution:* a per-node
diagnostic for when structural context is informative.

**Yun, S., Kim, S., Lee, J., Kang, J., & Kim, H. J. (2021). Neo-GNNs: Neighborhood overlap-aware graph neural
networks for link prediction. *NeurIPS 2021* (arXiv:2206.04216).** https://doi.org/10.48550/arXiv.2206.04216. PDF:
`…/2021_neurips_arxiv_2206_04216_….pdf` *Relevance:* canonical "learn the template weights". *Findings:* a learned
structural-feature generator over the adjacency plus learned per-hop coefficients `Σ_l β^{l-1}A^l` generalises
CN/AA/RA to multi-hop overlap and reaches SOTA on four OGB LP datasets; scores are adaptively combined with a feature
GNN. *Method:* OGB benchmarks, hop-depth ablations. *Quality:* Tier 1. *Contribution:* the normalisation of overlap
statistics is itself learnable.

**Yuen, H. Y., & Jansson, J. (2023). Normalized L3-based link prediction in protein–protein interaction networks.
*BMC Bioinformatics, 24*, 59.** https://doi.org/10.1186/s12859-023-05178-3. PDF:
`…/link_prediction_structural/2023_bmc_bioinformatics_normalized_l3_based_link_prediction_….pdf` *Relevance:*
degree-corrected L3 variants and template complementarity. *Findings:* plain degree-normalised L3 is the *least*
sensitive predictor to perturbations of an ideal L3 subgraph while L3N′ is the most sensitive; CN-based and L3-based
predictors select overlapping PPI pools in only 4–72 % of cases (MINT-Human 4 %, HuRI 24 %, BioGRID-Yeast 30 %)
against 58–92 % within each family. *Method:* controlled synthetic L3-graph perturbation plus Monte-Carlo sampling
on BioGRID/STRING/MINT/HuRI, yeast and human. *Quality:* Tier 1, open access, complexity analysis. *Contribution:*
quantitative proof that CN and L3 are different templates selecting different PPIs.

**Zhang, J., Wei, L., Xu, Z., & Yao, Q. (2024). Heuristic learning with graph neural networks: A unified framework
for link prediction. *KDD '24*, 4223–4234.** https://doi.org/10.1145/3637528.3671946 (arXiv:2406.07979). PDF:
`…/link_prediction_structural/2024_kdd_arxiv_2406_07979_….pdf` *Relevance:* clearest demonstration that template
weights are learnable *and* readable. *Findings:* local and global heuristics are all weighted adjacency products,
so learning `β^(l)` over ~20 layers subsumes CN, RA, Katz, LHN and LPI; on a synthetic triangular graph `β^(2)`
dominates and on a hexagonal graph `β^(5)` dominates; Cora/Citeseer put mass on `β^(0)` (features) while
ogbl-collab/ddi put it on `β^(2)` with negative weights at large `l`. *Method:* matrix-formulation proof,
Planetoid/Amazon/OGB benchmarks, interpretability and timing studies. *Quality:* Tier 1, few parameters, far faster
than SEAL/NBFNet. *Contribution:* the reference recipe for a learned, interpretable template-weight module.

**Zhang, M., & Chen, Y. (2018). Link prediction based on graph neural networks. *NeurIPS 2018* (arXiv:1802.09691).**
https://doi.org/10.48550/arXiv.1802.09691. PDF: `…/2018_neurips_arxiv_1802_09691_….pdf` *Relevance:* the licence for
reading a small local graph instead of the whole one. *Findings:* the γ-decaying heuristic theory proves Katz,
rooted PageRank, SimRank and similar are approximable from an h-hop enclosing subgraph with error decaying at least
exponentially in h, so "the effective order of these high-order heuristics is not that high"; SEAL beats heuristics,
latent-feature and embedding methods. *Method:* theory plus benchmarks. *Quality:* Tier 1, foundational.
*Contribution:* justifies compact pair-local structures as reader input.

**Zhang, M., Li, P., Xia, Y., Wang, K., & Jin, L. (2021). Labeling trick: A theory of using graph neural networks for
multi-node representation learning. *NeurIPS 2021* (arXiv:2010.16103).** https://doi.org/10.48550/arXiv.2010.16103.
PDF: `…/2021_neurips_arxiv_2010_16103_….pdf` *Relevance:* the formal statement of the pair-readout problem.
*Findings:* aggregating independently computed node embeddings cannot represent a node *set* — automorphic nodes get
identical representations, so `p(u,v)=p(u,w)` whenever `v≅w` regardless of distance; labelling the target set before
message passing yields the most expressive node-set representation. *Method:* expressiveness theory + LP experiments.
*Quality:* Tier 1. *Contribution:* the relation readout must be conditioned on the target pair inside the
computation, not assembled afterwards.

**Zheng, X., Du, H., Xu, F., Li, J., Liu, Z., Wang, W., … Zhang, Y. (2025). PRING: Rethinking protein–protein
interaction prediction from pairs to graphs. *NeurIPS 2025 Datasets and Benchmarks*.**
https://doi.org/10.52202/085713-3131 (arXiv:2507.05101). PDF: `…/2026_iclr_arxiv_2507_05101_….pdf` (local filename
says ICLR; the venue is NeurIPS 2025 D&B). *Relevance:* graph-level PPI evaluation matching our claim rules.
*Findings:* 21,484 proteins / 186,818 interactions with redundancy and leakage control; on Human BFS reconstruction
GS ranges 0.18–0.45 with RD 1.13–4.39 and clustering MMD 11.4–40.5 — every family produces graphs that are too dense
and mis-clustered; predicted modules align poorly with Reactome complexes. *Method:* BFS/DFS/RW sampling,
GS/RD/degree/clustering/spectral MMD, three function-oriented tasks. *Quality:* Tier 1, reproducible pipeline.
*Contribution:* fixes the metric set and the current state of the art for PPI network recovery.

**Zhou, Z., Yao, J., Liu, J., Guo, X., Yao, Q., He, L., Wang, L., Zheng, B., & Han, B. (2023). Combating bilateral
edge noise for robust link prediction. *NeurIPS 2023* (arXiv:2311.01196).** https://doi.org/10.48550/arXiv.2311.01196
*Relevance:* the most direct study of a link predictor whose input topology is wrong. *Findings:* edge noise is
*bilateral* — it perturbs input topology and supervision label simultaneously, causing severe degradation and
representation collapse; RGIB decouples the mutual dependence between topology, label and representation, as
RGIB-SSL and RGIB-REP; verified on six datasets, three GNNs, multiple noise regimes. *Method:* empirical noise study
+ information-bottleneck objectives. *Quality:* Tier 1. *Contribution:* names the failure mode a predicted-structure
reader will meet and gives an objective-level remedy.

**Zhu, J., Zhou, Y., Ioannidis, V. N., Qian, S., Ai, W., Song, X., & Koutra, D. (2024). Pitfalls in link prediction
with graph neural networks: Understanding the impact of target-link inclusion and better practices. *WSDM '24*,
994–1002.** https://doi.org/10.1145/3616855.3635786 *Relevance:* the leakage rule for structural supervision.
*Findings:* including the edge being predicted in the message-passing graph causes overfitting, distribution shift
and implicit test leakage, concentrated on low-degree nodes; SpotTarget excludes training target edges incident to a
low-degree node and all test target edges. *Method:* theoretical and empirical degree-stratified analysis. *Quality:*
Tier 1. *Contribution:* validates dropping the queried partner (and decrementing its degree) when building
structural targets, and warns the effect is degree-dependent.

**Zhu, Z., Zhang, Z., Xhonneux, L.-P., & Tang, J. (2021). Neural Bellman–Ford networks: A general graph neural
network framework for link prediction. *NeurIPS 2021* (arXiv:2106.06935).**
https://doi.org/10.48550/arXiv.2106.06935. PDF: `…/kg_inductive_lineage/2021_neurips_arxiv_2106_06935_….pdf`
*Relevance:* the learned generalisation of path templates such as L3. *Findings:* a pair score is a generalised sum
over path representations, each a generalised product of edge representations; a generalised Bellman–Ford with three
learned operators solves it, covering Katz, PPR and graph distance, and wins by large margins transductively and
inductively. *Method:* semiring formulation, homogeneous and knowledge-graph benchmarks. *Quality:* Tier 1.
*Contribution:* path-length weighting can be learned rather than fixed at ℓ=3, and source-conditioned
representations are a viable pair readout.

## 5. Evidence against

**Against "more templates, learned freely, beats one template."** Muscoloni & Cannistraci (2022): a supervised stack
of ~100 predictors loses to one unsupervised local rule in 14/18 mean-performance comparisons (median +64.5 % for
the rule) and 16/18 win-rate comparisons (median +383 %), and adding 16 CH features plus SPM and SBM produced "no
clear advantage". Kovács et al. (2019) agrees: L3 alone beat 20 published methods, and CRA lost *because* it mixed
ℓ=2 paths into ℓ=3 counting — a wrong template actively injected false positives. A learned weighting over a large
template bank is not automatically safer than one well-chosen prior.

**Against crediting degree- and density-bearing coordinates.** Aiyappa et al. (2024): a degree-only null predictor is
near-optimal under standard edge sampling. Bernett et al. (2024): sequence-based PPI models learn "solely from
sequence similarities and node degrees", reach ~87–92 % with degree alone on HUANG/PAN, and fall to chance (best
56 %) once leakage is removed. Li et al. (2023): Shortest Path scores AUC > 99 % with Hits@K = 0 on ogbl-ppa. Any
gain attributable to a degree coordinate or a global density knob must be assumed to be this artefact until a
degree-matched control says otherwise.

**Against the premise that a predicted pair structure is usable.** Wang et al. (2024) is a supportive precedent that
is also a warning: soft completion works, but the authors state that "weak models may not accurately recover the
unobserved common neighbor structure", and their gains are +0.2 to +2.2 points, not transformative. Zhou et al.
(2023): when input topology is noisy the noise is bilateral and produces representation collapse, not graceful
degradation. Yuen & Jansson (2023) add a subtler point — degree normalisation makes the plain L3 score *insensitive*
to real changes inside an L3 subgraph, so a well-normalised template can look stable while carrying no information
about the specific pair; and CN- and L3-family predictors agree on as little as 4 % of predicted PPIs, so a generator
that gets the "average" template right can still select entirely the wrong pairs.

**Against optimistic readings of any template gain.** Under HeaRT's hard negatives (Li et al., 2023) simple
heuristics more often beat GNN4LP and ogbl-citation2 MRR collapses from ~90 to ~20. Park & Marcotte (2012): C1 pairs
are > 99 % of typical test sets but 19.2 % of the population, so a gain measured on component-overlapping pairs does
not transfer to unseen proteins. PRING (2025): current PPI models reconstruct networks 1.13–4.39× too dense with
clustering MMD 11.4–40.5, i.e. pairwise accuracy and network fidelity come apart.

**Against assuming closed motifs are cheap to represent.** Peixoto (2022): triangles produced by closure are
systematically mis-attributed to community structure by a pure block model; only an explicit closure layer recovers
them. And Mao et al. (2024): structural and feature proximity are *incompatible*, and GNN4LP models consistently
lose to a plain GNN on feature-dominated edges — adding structural conditioning to a feature-strong trunk can cost
accuracy on exactly the pairs the trunk was good at.

## 6. What this thread cannot answer

1. **Which templates carry signal on *our* interactome at *our* V_val boundary.** Every PPI number above is on
   HuRI/BioGRID/STRING/MINT/HI-II-14 under transductive or C1-heavy splits; the CN-vs-L3 overlap (4–72 %) and the L3
   robustness curve (60–70 % edge removal) must be re-measured node-held-out.
2. **Whether an L3/motif template survives being *predicted from features only*.** Kovács's robustness is to random
   edge removal/addition; nothing here measures a template's value when the paths are hallucinated by a generator
   conditioned on `(x_u, x_v)`. NCNC is closest and completes from an *observed* graph.
3. **How to read a relation token out of a small soft-weighted graph.** LPFormer attends over real nodes with PPR
   encodings; the analogous readout over fractional edges of a ≈10–300-node virtual graph is Thread C territory.
4. **Whether learned template weights transfer to unseen nodes.** HL-GNN's interpretability study is transductive;
   no source reports `β^(l)` stability under a node-disjoint split.
5. **The right template-activity objective.** Muscoloni & Cannistraci argue against stacking, Neo-GNN/HL-GNN for
   learned weights; no included work resolves this for a *pair-conditioned, label-discriminative* module (Thread E).
6. **Degree-matched controls for our own numbers.** Aiyappa et al. and Bernett et al. specify the control; running it
   on our split is our experiment.

## 7. Saved files

New PDFs read in full and added to the corpus (`literature/README.md` untouched, per brief §4.5):

1. `literature/models/link_prediction_structural/2019_nature_communications_network_based_prediction_of_protein_interactions.pdf`
2. `literature/models/link_prediction_structural/2023_bmc_bioinformatics_normalized_l3_based_link_prediction_in_protein_protein_interaction_networks.pdf`
3. `literature/models/link_prediction_structural/2024_iclr_arxiv_2310_00793_revisiting_link_prediction_a_data_perspective.pdf`
4. `literature/models/link_prediction_structural/2024_kdd_arxiv_2406_07979_heuristic_learning_with_graph_neural_networks_a_unified_framework_for_link_prediction.pdf`
5. `literature/models/link_prediction_structural/2020_ecml_pkdd_arxiv_1902_06679_link_prediction_via_higher_order_motif_features.pdf`
6. `literature/models/graph_structure_learning/community_graph_modules/2022_physical_review_x_arxiv_2101_02510_disentangling_homophily_community_structure_and_triadic_closure_in_networks.pdf`

Read in full text but **not** saved as PDF (publisher PDF blocked; read via Europe PMC / PMC HTML): Muscoloni &
Cannistraci (2022, PMC9771718), Park & Marcotte (2012, PMC3531800), Bernett et al. (2024, PMC10939362). Screened at
abstract level only, with no numeric claim drawn: Benson et al. (2016), Guimerà & Sales-Pardo (2009). Already in the
corpus and read locally: SEAL, labeling trick, Neo-GNN, ELPH/BUDDY, NCNC, LPFormer, NBFNet, HeaRT, implicit degree
bias, topological perspective, PRING.
