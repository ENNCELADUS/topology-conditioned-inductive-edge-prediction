# Candidate Model Record: EgoStitch — Community-Conditioned Ego-Network Imagination with Consensus Stitching

**Status:** **revision 3.0 historical EgoStitch candidate, not the selected project method.**
Approved 2026-07-09; G4 signed off the arm-specific reproduction contract in
`05-egostitch-spec.md`. Companion to `03-experiment-protocol.md` (updated 2026-07-09 with the
approved [protocol-Δ] items) and `02-methodology.md`. The frozen-s0 EgoStitch
Stage-1 screen completed on 2026-07-17 with the binding verdict `cut`; revision 3.0
subsequently became the G5 build line before the 2026-08-10 method-selection reset.
**Revision 3.0 (2026-07-16):** the §4.4 decision mechanism is replaced — the frozen-B0
`s0` anchor and logit-level gated-residual fusion give way to an **end-to-end
stitched-topology-conditioned pair encoder** (jointly trained V3.1-class trunk,
structure-only stitched-topology encoder, zero-init tanh-gated cross-attention, and a
three-null decomposition). Design record and decision trail:
`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md` (rev 3).
Spec §§1–13 remain the historical contract for the completed frozen-s0 run and its
retained implementation. The successor contract is summarized in spec §14; its Phase-0
§5/§13 rewrite and fresh registration are now unblocked but are not yet normative.

**Provenance.** Revision 1 was grounded in a three-track literature review run on
2026-07-07. Revision 2 (2026-07-08) followed a full fan-out review of the local vault
(~90 PDFs) plus targeted external novelty-risk searches on the arXiv API. Revision 2.1
(2026-07-08) incorporates a simulated 5-persona design review (Editor-in-Chief,
methodology, domain, cross-disciplinary/CV, Devil's Advocate) — the panel's
design-stage findings are fixed in this text, and its experiment-stage findings are
registered as pre-implementation gates (§6.0).
Revision 2.2 (2026-07-09) follows full-text verification of the 21 papers previously
cited from abstracts only (now in the vault); all corrections are listed in the 2.2
change block below. All cited papers were verified against local PDFs or the arXiv
API; papers found only by external search are marked ⊕.

**One-line summary of this candidate arm.** Replace the retrieved-and-thresholded scaffold with a *generated
and harmonized* one: for each queried pair `(i, j)`, each endpoint **imagines its own
ego-network** (a set of latent neighbor slots with existence probabilities, slot–slot
adjacency, and a degree budget) conditioned on its frozen features and a learned
**community codebook**; the two imagined ego-networks are **stitched and jointly
re-decoded to consensus** under scaffold-level degree-budget constraints; and the edge
decision fuses the node-intrinsic pairwise logit with membership, closure,
community/capacity, and scaffold-readout evidence.

**Honest interpretation.** The strict task input is `(x_i,x_j)`. Grounded EgoStitch
also reads retrieved feature rows, so it is a separate retrieval-grounded arm and
requires retrieval-only controls. Only its ungrounded configuration can claim no
test-time information beyond B0; there the scaffold injects **structural priors
distilled from training-graph supervision**. The remaining claim is about inductive
bias and supervision: the ego-net
bottleneck is hypothesized to be a better vehicle for learning topology-aware,
better-calibrated per-pair marginals than direct multi-task heads on a pairwise scorer
— and that hypothesis is falsifiable against the `B3-full` control (§6.2), which
receives all of EgoStitch's auxiliary supervision with none of its generative
machinery and matched endpoint-only candidates. Under the per-query contract the assembled graph is an
**edge-independent construction given features**; mechanisms in this proposal
constrain the *scaffold* exactly and influence the *assembly* through shaped marginals
(§4.6, §6.0-G2).

**What changed in revision 2** (each item traceable to a reviewed source, §5/§6):

1. **Consensus stitching (Module 3b):** one-shot stitching leaves the queried edge on
   an unharmonized *seam* between two independently generated ego-nets — the
   documented failure locus of one-shot conditioning in inpainting (RePaint
   2201.09865). A confidence-scheduled parallel refinement loop (MaskGIT 2202.04200
   pattern) makes the two ego-nets reach consensus by cross-conditioned re-decoding.
2. **Degree-budget masking at stitch time** (EDGE 2305.04111 pattern) — scaffold-level
   constraint (see change 2.1-a below for scope correction).
3. **Conditioning dropout** yielding counterfactual controls (scope corrected in
   2.1-c).
4. **Realism loss aligned with a defensible metric:** `L_real` adds an energy-distance
   term in untrained random-GIN embedding space (Thompson et al. 2201.09871).
5. **Narrowed, qualifier-scoped novelty claims** and the §5 threat matrix.
6. **Evaluation hardening package** (§6.4).

**What changed in revision 2.1** (panel findings; `R1`–`R4` / `DA` tags identify the
reviewer persona that raised each one — EIC, methodology, domain, cross-disciplinary,
Devil's Advocate):

- **(a) Scaffold/assembly scope correction** [R1-W3, R3-W3, DA-C1, R2-W3 — unanimous]:
  all "hard guarantee" language is now scoped to the scaffold; assembly-level effects
  are claimed only as mechanism-backed shaping of marginals, with a degree-calibration
  diagnostic (§6.4.8) measuring the transmission directly.
- **(b) Edge-independence ceiling made explicit** [DA-C2, R2-W3]: Chanpuriya et al.
  (NeurIPS 2021, 2111.00048 ⊕) is adopted in both roles — the strongest theoretical
  backbone for the E2 gap *and* a bound on what any locked-contract method can repair;
  a pre-implementation feasibility check (§6.0-G2) quantifies the reachable frontier
  before any code is written.
- **(c) Two distinct dropout nulls** [R3-W7]: `∅_content` (drop `[x,z,r]`, keep
  `G(u)`) implements the TDE-style content-wiped control; `∅_all` (drop both)
  implements the CFG-style population prior. The previous "CFG = TDE" unification is
  retracted as imprecise; guidance-style extrapolation is scoped to logit space.
- **(d) Harmonization is a trained module** [R3-W4/W6, R1-W2]: a joint two-ego
  cross-conditioned masked training task is specified so inference-time refinement
  queries in-distribution conditionals; gradient estimators for the hard operations
  are specified; slot *content* confidence comes from codebook quantization; kept
  slots are re-maskable (Token-Critic pattern).
- **(e) Hub policy for set decoding** [R3-W1/W2]: importance-weighted target
  subsampling with community-multiplicity counts, compound matching costs, denoising
  queries, and a budget trigger redefined in K-representable terms.
- **(f) Shared interactions with leave-one-out** [R1-W5]: structural targets (BP-NLL,
  degree NLL, reconstruction) and `L_edge` use the same complete train positives;
  the queried partner is explicitly removed from per-pair reconstruction targets and
  degree counts. Seam references are sampled label-agnostically; B0 provenance is an E5 gate.
- **(g) Goodhart split** [R1-W7]: assembled-graph metrics are partitioned into
  trained-on and held-out families; the held-out family is headlined.
- **(h) Full loss tree** [R1-W1]: every sub-loss assigned a weight under the four
  locked lambdas; balancing strategy and HPO-parity protocol stated.
- **(i) Positioning corrections** [R2-W1/W4/W5/W6/W7]: KG-inductive lineage
  (GraIL/NBFNet/ULTRA ⊕) added with a settings taxonomy; VQGraph's distilled student
  acknowledged; labeling-trick and γ-decay claims scoped; C4 claim scoped to
  "protocol-gated joint evaluation under strict zero-edge inductive LP"; NRI,
  Graph2Gauss, GraphMAE, DC-SBM/ERGM classical priority added.
- **(j) Ladder and run-order changes** [DA-C5, DA-M10, EIC]: `B3-full` Ockham arm
  added; Oracle promoted to first run; E7 promoted from optional to load-bearing;
  minimum-viable-model staging milestone; pre-registered decision rules.

**What changed in revision 2.2** (full-text verification of the 21 abstract-cited
papers; two shard reports in the session record):

- **(k) Ceiling semantics corrected [substantive].** Chanpuriya 2111.00048's bound is
  a quantitative overlap–volume tradeoff with *exact* computable identities
  (`E[Δ] = tr(P³)/6`; `Ov(P)·V(P) = Σ p²`), not a binary memorization dichotomy — and
  a hard-thresholded assembly has overlap 1, where the bound is vacuous. Gate G2 is
  restated as a **ceiling curve over overlap**, evaluated at the soft scorer's
  measured overlap, with the clustering ceiling built from the exact triangle
  numerator + the assembly's expected-degree denominator (Thm 6 is big-O only). The
  "cannot reproduce triangles without memorization" gloss is retired.
- **(l) Quote-backed taxonomy.** GraIL/NBFNet/ULTRA/IGMC one-liners are now grounded
  in the papers' own text (IGMC states verbatim it "does not address the extreme
  cold-start problem"); the KG-inductive row and ours are information-theoretic
  complements (structure-only/features-absent vs features-only/structure-absent).
- **(m) Family split.** P-GNN is anchor-based *absolute positional*, not a labeling
  trick — §5.4/§5.5 wording split; the §5.4 fidelity qualifier now covers Distance
  Encoding as well.
- **(n) Stratification correction.** Topological Concentration (2310.04612) *undercuts*
  naive degree strata (degree is weakly and bias-prone correlated with LP
  performance); TC/ATC stratification added alongside degree, and a TDS-style
  train/test ego-net drift diagnostic added to §6.4.8.
- **(o) Ladder additions.** Mandatory **PA-null** baseline (`s_ij = k_i·k_j`, with
  degree-heterogeneity σ reported per benchmark) and the Chanpuriya **odds-product**
  degree-respecting edge-independent baseline; G1 re-verifies E2 under
  degree-corrected negatives explicitly (2405.14985).
- **(p) Metric-validation additions (O'Bray 2106.01098).** Per-statistic bin-count
  disclosure; no ad-hoc EMD/TV kernels without justification; MMD-based diagnostics
  must pass an expressivity/robustness perturbation check. Official Graph
  Similarity and Relative Density remain separately reported metrics.
- **(q) Scope qualifiers.** GraphMAE's feature-target evidence is for classification
  (its own text concedes GAEs are strong at LP) — R4 rests on the shortcut argument
  and our ablation, with GraphMAE as representation-learning precedent (its
  scaled-cosine + re-mask stabilizers imported); UPNA's "learns the graph generation
  mechanism" rhetoric distinguished explicitly (independent pairwise scoring, no
  per-query generated context, no assembled grading); New Node Prediction (2401.05468)
  sharpened to *semi-inductive* (its isolated node links into a fully observed graph);
  HiGGs' stage-three community stitching distinguished from per-query ego-net
  stitching; FLEX's documented dense-generation failure recorded as independent
  evidence for hard degree budgets.
- **(r) Venue statuses updated** (§8): Meta-Embedding SIGIR 2019 confirmed (tag
  cleared); NBFNet/ULTRA/P-GNN/DE/LPFormer/O'Bray/TGB confirmed from PDFs; GraIL and
  IGMC local PDFs are arXiv preprints (camera-ready check before quoting pages);
  2310.04612, 2405.14985, UPNA, FLEX are preprints; HiGGs and 2401.05468 cited as
  arXiv without venue claims.

---

## 1. Critique of the current scaffold contract

> **Metric note (2026-07-14):** official BFS-macro GS/RD were formally rerun on the frozen
> score artifacts over all 500 fixed induced subgraphs. The final G1/G3 artifacts are
> `outputs/deliverables/g1_graph_metrics_20260714/` and `outputs/deliverables/g3_graph_metrics_20260714/`;
> the MMD component ratios are unchanged canonical-run values.

Current contract (`03-experiment-protocol.md` §0):

```text
V_T  = {i, j} ∪ ANN_feat(i, k) ∪ ANN_feat(j, k)
E_T  = {(a, b) ∈ V_T × V_T : score(a, b) ≥ τ_pair}, weighted by score(a, b)
T_ij = (V_T, E_T \ {(i, j)})
p_ij = σ(pair_logit(i, j) + scaffold_residual(H_i, H_j))
```

### 1.1 The scaffold is B0's echo — a supervision critique, stated precisely

Scaffold edges are thresholded outputs of the same frozen pairwise scorer whose
assembled output G1 proved structurally implausible (global simple-edge RD `0.997710`,
official BFS-macro GS/RD `0.312151/0.422345`, and degree/clustering/spectral MMD ratios
`13.0768/11.9273/18.0931`). `T_ij` is a local patch of exactly the pathological graph
the method is supposed to fix, and systematic B0 errors — hub over-prediction,
similarity–adjacency conflation — are inherited by the context and then "corrected" by a
residual conditioned on those same errors.

A separate checkpoint-only evaluation of the aligned legacy `v3_1` scorer used the frozen
scores from run `legacy_v31_s47_20260712T193900Z`; its final evaluator artifact is
`outputs/deliverables/legacy_g1_graph_metrics_20260714/`. The official metric rerun gives
global simple-edge RD `0.978392`, BFS-macro GS/RD `0.381264/0.500179`, and
degree/clustering/spectral MMD ratios
`13.8456/11.6277/19.9774` despite degree-corrected AUROC/AUPRC
`0.799577/0.813319`; stronger edge ranking therefore does not remove the topology gap.
This checkpoint-only rerun does not replace the formal E2 training record.

Stated carefully: *any* per-query scaffold — retrieved or generated — is a function of
the same test-time inputs available to B0, so no scaffold "adds information" at test
time. The relevant axis is **what supervision shaped the scaffold's parameters**. The
retrieved-thresholded scaffold is trained by nothing: its edges re-encode B0's biases
with no independent signal. A generated scaffold can instead be trained against **real
ego-network topology** (masked reconstruction, degree likelihoods, block likelihoods,
distributional realism), distilling structural priors the pairwise scorer never
receives. That — not test-time information — is the deficiency this proposal fixes, and
it is why the decisive control is `B3-full` (§6.2): the same auxiliary supervision
delivered without the topological bottleneck.

### 1.2 Retrieval conflates similarity with adjacency

`ANN_feat` retrieves *peers* (nodes similar to `i`), not *plausible partners* (nodes
likely adjacent to `i`). In heterophilous or interaction-type graphs these are different
sets. The scaffold therefore tends toward two near-cliques of look-alikes around each
endpoint — a nearly query-invariant topology carrying little discriminative structural
signal. This predicts collapse toward B1 (retrieval features, no adjacency) in ablations
E4.1/E4.3.

### 1.3 The topology module is not actually learned

`E_T` comes from a frozen scorer plus a hard threshold `τ_pair`: non-differentiable, no
gradient path to improve scaffold construction. Only the encoder and residual head learn.
The methodology plan's requirement of "a differentiable relaxation so edge loss and
realism loss can train the topology module" is not realized by the protocol
instantiation. Worse, score distributions shift across feature-space regions, so one
global `τ_pair` yields wildly varying scaffold density per query — the encoder sees noise
in exactly the statistics that matter.

### 1.4 No generative prior; moment-matching `L_real` is a weak fix

Queried-edge loss under-constrains unqueried scaffold edges (acknowledged in the plan).
Moment-matching losses on degree/clustering statistics are non-identifiable: many wrong
graphs match the moments, producing "mean" topologies rather than samples from the
distribution of real neighborhoods. Manenti et al. (ICML 2025, 2405.19933, vault) prove
the sharper version in their setting: point-prediction losses do not grant a calibrated
latent-graph distribution even with the true downstream map (their Prop. 4.1), while a
distributional loss on outputs is sufficient under mild conditions (their Thm. 5.2, with
MMD recommended and energy distance named as a feasible alternative). Our `L_real` is
*motivated by* this impossibility/sufficiency pair — not "guaranteed" by the theorem,
whose assumptions (expressive families, injectivity, and a per-instance downstream
signal far richer than one binary pair label) do not literally hold here and are stated
honestly wherever the argument is used.

### 1.5 Test-pool dependence

ANN retrieval runs over whichever nodes co-occur in the held-out split. Predictions for
`(i, j)` therefore depend on the composition of the rest of the test pool: non-i.i.d.
behavior, inconsistency across query batches, and failure for isolated pairs with no good
candidates. The design resolution (fixed in §4.2): generation is the *primary* pathway
and answers from `(x_i, x_j)` alone; retrieval is optional grounding whose influence is
gated, trained for pool-robustness by `L_ssl`, and ablated as a headline arm
(grounded vs ungrounded, §6.3). The old design's flaw was *hard* dependency on the pool;
the new design's residual pool sensitivity is measured, not assumed away.

### 1.6 A local residual cannot repair distributional failures — and the honest limit of any local fix

G1's failure is distributional (global simple-edge RD `0.997710`, BFS-macro
GS/RD `0.312151/0.422345`, and degree/clustering/spectral MMD ratios 13.0768, 11.9273,
and 18.0931). Independent per-query residuals have no channel for node-level degree budgets or community-level edge
budgets. EgoStitch's answer is to inject those budgets as *evidence within each
decision* — shaping every marginal by the same distilled priors — while acknowledging
the structural limit that survives any locked-contract method: predictions remain
independent across queries, so the assembled graph is an edge-independent construction
whose triangle/clustering realism is bounded by an overlap–volume tradeoff, tight up to
constants (Chanpuriya et al. 2111.00048 ⊕: `E[Δ] ≤ (√2/3)(Ov·V)^{3/2}`; §6.0-G2
evaluates this as a ceiling *curve over overlap* on the benchmark before
implementation). The CV literature reached the analogous verdict on
inpainting seams: no per-unit loss fixes assembly-level implausibility without either a
whole-assembly objective or non-independent decoding (Context Encoders 1604.07379,
RePaint 2201.09865, MaskGIT 2202.04200) — within a single query's scaffold we *can* and
do decode non-independently (§4.3b); across queries the locked contract forbids it, and
the proposal claims only marginal-shaping there.

### 1.7 No mesoscale (community) representation

The scaffold is ~`2k+2` nodes: micro-scale. Clustering and spectral MMD failures are
mesoscale phenomena — community structure, degree mixing. The contract has no channel by
which community structure, structural roles (hub vs leaf), or block densities enter the
decision. "Fuse community topology with node-intrinsic representation" is exactly the
missing piece.

### 1.8 Soft answer leakage bounds the residual's contribution

The queried edge is masked, but scaffold edges `(i, c)` and `(j, c)` for shared
candidates `c` are scores from the same function; the encoder can largely reconstruct
`score(i, j)` by transitivity. The residual risks being a smoothed re-reading of B0 —
consistent with a "marginal AUROC gain, no topology gain" outcome.

### 1.9 Expressiveness limits — what they do and do not license

The labeling-trick theory (Zhang et al., 2010.16103) proves that a GNN over a subgraph
cannot represent common-neighbor or other pair-relative structural evidence unless nodes
carry target-relative labels; the *automorphic node problem* (ELPH/BUDDY 2209.15486)
shows equivariant independent node embeddings assign equal link probability to
automorphic pairs at any distance. The E2 gap is therefore **consistent with a known
expressiveness limit** of independent pairwise scoring — a motivation, not a proof of
the measured MMD magnitudes.

Honesty requires the converse statement too [DA-M6]: any feature-only model — EgoStitch
included, when ungrounded — assigns identical scores to feature-identical pairs. The
labeling trick on a *generated* scaffold buys the most expressive readout *of the
generated context* (see §5.4 for the scoped claim), not access to target-graph
structure. What the architecture changes is not the test-time function class's
information content but (i) which functions are *easy to learn* under topological
supervision, and (ii) with grounding, the injection of genuine test-pool
context — which is why grounded-vs-ungrounded is a headline ablation rather than a
footnote.

### What survives

The locked outer contract is right and is kept unchanged: per-query locality, frozen
features, masked queried edge, dual edge+graph evaluation, the baseline ladder. The
weakness is entirely in *how `T_ij` is constructed and what supervision shapes it*:
retrieved + thresholded (unsupervised echo) ≠ generated + realism-trained + harmonized.

---

## 2. Design requirements (derived from §1 and the two-pass literature review)

| # | Requirement | Source |
|---|---|---|
| R1 | Scaffold must be trained against **real ego-network topology**, not B0's outputs | §1.1, §1.4 |
| R2 | Neighbor proposals must be **generative**, with retrieval as optional, gated grounding | §1.2, §1.5; GAR (SIGIR 2022), Cold Brew 2111.04840, RDM 2204.11824 |
| R3 | Scaffold construction must be **differentiable / learned end-to-end**, with every hard operation given an explicit gradient estimator | §1.3; DGM 2002.04999 (machinery); §4.5 loss tree |
| R4 | Reconstruct neighbor **features**, not raw adjacency bits, as the primary generative target | DINOSAUR 2209.14860; GraphMAE 2205.10803 ⊕ as representation-learning precedent only — its evidence is classification-side, and its own text concedes structure-reconstructing GAEs are strong at LP, so R4's edge-prediction justification rests on the §1.8 shortcut argument + our E4 ablation; GraphMAE's scaled-cosine error + re-mask decoding are imported as stabilizers; Graffe 2505.04956 |
| R5 | Include a **distributional realism loss** (energy distance/MMD), not only point losses | Manenti et al. 2405.19933 (motivating theory, §1.4 caveats); NetGAN 1803.00816; Graph Gestalt 2106.15239 |
| R6 | Carry explicit **degree budgets** and **community/block priors** into every decision; enforce the budget as a hard constraint **at the scaffold level** and measure its transmission to assembly | §1.6, §1.7; NOCD 1909.12201; EDGE 2305.04111 (degree-first factorization); MaskGAE degree decoder 2205.10053; DC-SBM (Karrer & Newman 2011) as the classical ancestor of the `d̂`+block channel |
| R7 | Apply the **labeling trick** on the scaffold (anchor labels for `i`,`j`) — the most expressive pair-aware readout *of the generated context* (§5.4) and the theoretically grounded implementation of the queried-edge masking gate | §1.9; 2010.16103; Distance Encoding 2009.00142 ⊕ |
| R8 | Generation must be **label-agnostic** (no edge-hypothesis conditioning), including all realism-loss reference sampling | TDE 2002.11949; SGG bias evidence 1711.06640; §4.5 seam-reference rule |
| R9 | Negative sampling **after** masking; queried endpoint explicitly removed from reconstruction targets; topology and classification otherwise use the **same complete training interactions** | vault masked-edge report; [R1-W5] |
| R10 | Per-node computation should be **cacheable across queries** (amortization), with the per-pair marginal cost stated honestly against B0, not against the old scaffold | protocol §0; [R3-W9] |
| R11 | The stitched scaffold must be **seam-consistent**, via a *trained* joint refinement procedure (train/test-matched conditionals) | RePaint 2201.09865; MaskGIT 2202.04200; [R3-W4] |
| R12 | The reconstruction pretext must be **train/test-matched**: mask *entire* ego-networks (the zero-edge inference condition), mask structurally coherent units, and train all conditioning patterns used at inference (including cross-ego conditionals and dropout nulls) | pix2gestalt 2401.14398; MAE 2111.06377; MaskGAE Prop. 1; [R3-W4] |
| R13 | Set supervision must use **Hungarian matching with an existence (∅) class** and a **compound matching cost** (feature + structural descriptors); Chamfer is inadmissible for multisets; an explicit **hub policy** covers degree ≫ K | DSPN 1906.06565; DETR 2005.12872; [R3-W1/W2] |
| R14 | Evaluation must survive the known attacks on LP benchmarks and MMD metrics, and must separate **trained-on** from **held-out** metric families | HeaRT 2306.10453; Thompson et al. 2201.09871; Beyond-MMD 2512.14241; [R1-W7] — see §6.4 |
| R15 | Feasibility before construction: the **edge-independence realism ceiling** and the **Oracle headroom** must be measured before implementation | Chanpuriya et al. 2111.00048 ⊕; [DA-C2, DA-M10] — §6.0 |

---

## 3. Candidate architectures considered

### Approach A (recommended): EgoStitch — dual ego-network imagination + community codebook + consensus stitching

Each endpoint generates its own ego-network (latent neighbor set + degree + local
adjacency) conditioned on frozen features and a quantized community code; the two
imagined ego-nets are stitched, jointly re-decoded to consensus under scaffold-level
degree budgets, and the decision head fuses four evidence channels. Detailed spec in §4.

- **Pros:** attacks all three E2 failure axes by an explicit mechanism-to-marginal
  pathway (§4.6); per-node generation is cacheable (R10); subsumes the current design
  as an ablation (generator off, grounding on); novelty combination verified unoccupied
  after a two-pass review (§5).
- **Cons:** set-generation training (Hungarian matching) adds engineering complexity
  and converges slowly early (documented by DETR itself); refinement rounds add
  per-pair cost that must be reported against B0 (§4.7); several auxiliary heads to
  balance (§4.5 loss tree).

### Approach B: per-query discrete graph diffusion scaffold

Condition a DiGress/EDGE-style discrete denoiser (2209.14734, 2305.04111) on
`{x_i, x_j} ∪ candidates` to denoise a local adjacency; optionally score the edge as a
diffusion classifier (ELBO gap between hypotheses, 2303.16203).

- **Pros:** strongest generative fidelity; principled likelihood-based decision.
- **Cons:** T denoising steps *per query* (not per node) — poor amortization; restricted
  to retrieved candidates, so §1.5 pool-dependence survives; adjacent to Latent Graph
  Diffusion (2402.02518), weakening the novelty claim. **Verdict:** keep as the
  generator-variant ablation arm (E4.6), not the main method. Note the protocol
  asymmetry to state loudly: LGD-style inpainting *conditions on observed test
  adjacency*, which our protocol forbids — the honest version of this baseline can
  condition only on frozen features and retrieved candidates, which is exactly what
  E4.6 instantiates.

### Approach C: neural-SBM/graphon residual (lightweight)

Infer NOCD-style community affiliations `F_u` and a degree correction from features
alone; `p_ij = σ(pair_logit + block_affinity(F_i, F_j) + degree_terms)`. No neighbor
generation. Classically: a feature-conditioned degree-corrected SBM (Karrer & Newman
2011) in modern dress.

- **Pros:** cheap; directly repairs density and degree distribution.
- **Cons:** no local closure evidence; no neighbor generation. **Verdict:** embed as
  EgoStitch's community/capacity channel (§4.4, channel `s3`) and register as baseline
  **B5**. NOCD asserts inductiveness of amortized affiliations in one sentence and
  defers evaluation; B5 is therefore itself a small claimable contribution — *the
  first evaluation of the Bernoulli–Poisson block prior under a strict inductive,
  frozen-feature, no-test-graph protocol*. Per NOCD's own Table 2, the feature-only
  MLP form of the affiliation encoder is not a strawman (it beats the GCN form on
  attribute-strong datasets), and it is the only form admissible under our gates.
  The Devil's Advocate's observation is accepted and recorded: **B5 (+ calibration)
  is the natural null hypothesis of this whole project** — if `Ours ≈ B5+cal`, the
  honest paper is about calibrated block-model marginals, and §6.5's decision rules
  say so in advance.

---

## 4. Historical candidate model: EgoStitch

### 4.0 Contract compliance

```text
inputs:  queried pair (i, j), frozen features X (raw token sequences + pooled x_u),
         optional ANN candidate pools G(i), G(j)
step 1:  community coding      z_u, r_u, F_u, d̂_u    = Tokenize(x_u)              (per node, cached)
step 2:  ego-net imagination   S_u = {(h_u^k, π_u^k)} = Imagine(x_u, z_u, r_u, G(u)) (per node, cached)
step 3a: stitch                T̂⁰_ij = Stitch(S_i, S_j, {i, j})                   (per pair)
step 3b: consensus             T̂_ij  = Harmonize(T̂⁰_ij | d̂_i, d̂_j, R rounds)     (per pair, trained module)
step 4:  encode + decide       t = STE(T̂_ij)   # structure-only token states (§4.4)
         p_ij = σ(head(Trunk(tok_i, tok_j | t)))                                  (per pair)
```

The per-query local scaffold boundary is unchanged: `T̂_ij` is a local topological
context built from frozen features; the classifier is conditioned on `T̂_ij`; the output
is the binary edge label. Steps 1–2 are per-*node* and cached across all queries sharing
an endpoint (R10); steps 3a/3b/4 are per-pair. No target-graph access anywhere.

**Inference determinism policy** [R1-W6, DA-M14]: the CVAE ego-net latent is sampled
`n_s` times per node (default `n_s = 4`, cached samples shared across that node's
queries with a per-pair random pairing of samples to avoid systematically correlated
errors); `p_ij` is the average over sample pairs; calibration metrics (ECE/Brier) are
computed on the averaged probability; `n_s = 1` (mode) is the reported fast variant.
Seeds fix the sample set, making inference reproducible.

### 4.1 Module 1 — Topology tokenizer (community + capacity channel)

- **Community code:** `e_u = MLP_enc(x_u)`; vector-quantize against a codebook
  `C = {c_1..c_M}` of neighborhood prototypes (VQ with EMA updates, straight-through;
  VQ-VAE 1711.00937). `z_u = c_{m(u)}`. Codebook quantization additionally supplies the
  **content-confidence** signal used by harmonization (§4.3b): a slot embedding's
  distance to its nearest code is a calibrated proxy for how prototypical the imagined
  neighbor is.
- **Overlapping community affiliation:** `F_u = softplus(MLP_F([x_u; z_u])) ≥ 0`,
  trained with the NOCD Bernoulli–Poisson likelihood on the training graph's
  full loopless training topology (R9), with class-balanced negatives. The encoder is the
  feature-only MLP form (never graph-conditioned), so affiliations exist for zero-edge
  unseen nodes by construction — the property the whole channel depends on (§3C).
- **Degree budget:** `d̂_u = softplus(MLP_deg([x_u; z_u]))`, trained with NLL against
  full training-topology degrees, **normalized as expected degree per unit candidate-universe
  density** so budgets transfer across evaluation scales (train graph vs 20–200-node
  buckets vs Benchmark-B/C; [R1-W11-iii, R2-Q8]). MaskGAE's degree decoder 2205.10053
  is the cited prior for auxiliary degree supervision; EDGE for degree-first
  factorization; DC-SBM for the classical degree-correction reading.
- **Code supervision:** an auxiliary head predicts ego-net statistics — degree, local
  clustering, neighbor-code histogram, and *triangle/motif conductance* (HoscPool
  2209.03473) — from `z_u`, so codes tile the space of *neighborhood shapes* (hub/leaf,
  dense/sparse, community identity), not feature space.
- **Continuous residual:** `r_u = e_u − z_u` is passed downstream (DIAMOND 2405.12399
  documents decision-relevant information lost to discrete-latent compression); the
  no-residual variant is an ablation.

**Novelty scoping (binding for the paper text).** Discrete codebooks over graph
structure exist: VQGraph (2308.02117) quantizes teacher-GNN node embeddings into codes
used as distillation targets; GFT (2411.06070) quantizes computation-tree embeddings
into a transferable task vocabulary. VQGraph's *tokenizer* needs the observed
neighborhood, but its **distilled student MLP predicts soft code assignments from
features alone** — that is the point of GNN-to-MLP distillation, and the earlier
"cannot code a zero-edge node" phrasing is withdrawn as overbroad [R2-W5a]. The claim
that survives rests on the remaining qualifiers: *a VQ codebook of neighborhood
prototypes used to condition **generation** of ego-net context, with codes supervised by
**ego-net statistics**, inside a strict zero-edge inductive edge-prediction protocol.*
Neither VQGraph nor GFT generates topology, supervises codes with ego-net statistics,
or addresses edge prediction as the task.

This module *is* the community-topology representation, fused downstream with the
node-intrinsic representation `x_u` — the fusion the current design lacks (§1.7).

### 4.2 Module 2 — Ego-network imagination (neighbor node generation)

A conditional set decoder generates `K` latent neighbor slots per node. Lineage naming
(anti-grab-bag): this is a **DETR-style set decoder** (2005.12872) — Hungarian-matched
queries with an existence class — with Slot Attention (2006.15055) and DSPN (1906.06565)
cited as related set-decoding lineages, not as additional components.

- **Slots:** `K` slot queries cross-attend to `[x_u; z_u; r_u]` and (when available) to
  grounding candidates `G(u)` from feature-ANN. **Dynamic query initialization**
  (PoinTr 2108.08839): slot queries are initialized from the grounded candidates and
  the community code rather than fixed learned vectors. Output per slot: neighbor
  embedding `h_u^k` (in a projection of frozen-feature space), existence probability
  `π_u^k`, a **multiplicity weight** `m_u^k ≥ 1` (hub policy, below), and a grounding
  gate `g^k ∈ [0,1]` with a pointer over `G(u)`. **The grounding gate is trained as a
  partner-vs-peer discriminator** [R3-W8]: against the full training topology, `g^k` is
  supervised to predict *adjacency plausibility* of the candidate (is this candidate
  actually a neighbor type, not merely feature-similar), so grounded slots earn trust
  from evidence rather than retrieval rank. Parametric conditioning remains primary
  (RDM 2204.11824's "NNs only" ablation shows retrieval-only conditioning fails to
  generalize). A per-ego-net stochastic latent (CVAE, LA-GNN 2109.03856) provides
  diversity — a node that could belong to two communities should imagine two different
  ego-nets (semi-implicit posterior precedent: SIG-VAE 1908.07078) — with **free-bits
  KL and annealing named as the posterior-collapse mitigations** and an imagined-ego-net
  **diversity metric** (dispersion across CVAE samples vs real per-node neighbor
  diversity) registered as a training diagnostic [R3-W10]. (f-VAEGAN 1903.10132 is
  imported in its *inductive* reading only; its D2 transductive variant uses unlabeled
  test data and is inadmissible under our gates.)
- **Hub policy** [R3-W1]: for nodes with `|N(u)| > K` the reconstruction target is an
  importance-weighted subsample of `N(u)` (stratified by neighbor community and degree)
  paired with per-slot multiplicity supervision: slot `k` predicts `m_u^k`, the number
  of near-equivalent neighbors it represents, so `Σ_k π_u^k · m_u^k` — not `Σπ ≤ K` —
  is tied to `d̂_u`. This makes the cardinality tie feasible for hubs and keeps the
  budget trigger (§4.3b) non-vacuous exactly on the degree axis. Matching stability
  measures: **compound matching cost** (feature distance + structural descriptor
  agreement: degree bucket, code, adjacency-overlap with currently matched set)
  [R3-W2], **denoising queries** (noised true neighbors with fixed assignments — the
  DN-DETR-style fix for assignment flapping, implemented directly rather than cited,
  as no graph version exists), and a monitored **assignment-flip-rate** diagnostic.
- **Slot–slot adjacency:** logits `Â_u^{kk'}` over slot pairs model neighbor–neighbor
  edges (the source of local clustering), supervised at **group level** (expected
  connectivity between matched community groups) to be robust to near-duplicate
  assignment flips [R3-W2]; a Bandana-style continuous bandwidth form (2402.03814) is
  the registered alternative.
- **Training — masked ego-net reconstruction** on the full training topology (R9, R12):
  hide node `u`'s **entire** neighbor set — the total-masking limit matches the
  zero-edge inference condition, which no masked graph autoencoder trains (§5) — and
  mask structurally coherent units in curricula (pix2gestalt lesson). Hungarian
  matching of slots to `{proj(x_v)}` with the compound cost; matched slots: feature
  regression + `π → 1` + multiplicity NLL; unmatched slots: `π → 0`; degree NLL;
  group-level adjacency loss. Masking severity is swept (PoinTr). When the pair loss
  for `(u, v)` is active, `v` is removed from `u`'s reconstruction targets and
  generation never sees the edge label (R8).
- **Conditioning dropout — two distinct nulls** [R3-W7]: during training the decoder
  sees, with scheduled probabilities, (i) `∅_content`: `[x_u; z_u; r_u]` replaced by a
  null token while `G(u)` is retained — the *content-wiped, context-held* pattern whose
  inference-time contrast with the full condition is the **TDE-style counterfactual
  control** (2002.11949) on the fused logit; and (ii) `∅_all`: both content and
  grounding dropped — the population-prior null whose contrast supports **CFG-style
  guidance, applied in logit space only** (extrapolated continuous slot embeddings are
  not guaranteed on-manifold, so `h` is never extrapolated). Both patterns are trained,
  so both inference-time contrasts are in-distribution (R12). The earlier claim that
  one mechanism "is" the TDE counterfactual is retracted as conflating the two nulls;
  counterfactual-subtraction debiasing is established in SGG/VQA and we claim only the
  *amortized, trained-null* implementation in this setting.

### 4.3 Module 3a — Stitching

```text
V̂ = {i, j} ∪ S_i ∪ S_j          (≈ 2K + 2 nodes, K ≈ 10–20)
Π = SoftAlign(S_i, S_j)          (entropic-OT alignment over compound slot descriptors:
                                  which imagined neighbors coincide; ε and cost defined
                                  in the §6.0-G4 algorithm box)
Ê  = { (u, k): π-weighted star edges } ∪ { slot–slot edges Â } ∪ { aligned-slot merge
       edges weighted by Π }     (soft merge: aligned slots are kept as distinct nodes
                                  joined by high-weight alignment edges, so s2's closure
                                  mass and the s4 readout see the same object)
labels: target-relative labeling-trick features (endpoint-i / endpoint-j / slot-of-i /
        slot-of-j / grounded-node identity match)                                (R7)
```

`Π` receives gradients from three sources: the seam realism term (§4.5), the closure
channel `s2` (through `L_edge`), and the joint harmonization task below — it is not an
unsupervised afterthought [R2-Q6].

### 4.3b Module 3b — Consensus harmonization (a trained module)

One-shot stitching is the failure mode RePaint documents for one-shot conditioned
inpainting: locally plausible, globally wrong, with the error concentrated at the
*seam* — and the queried edge `(i, j)` lies precisely on the seam between the two
generated ego-nets.

**Joint training task (the fix for the untrained-conditional gap [R3-W4/W6]).** During
training, sample from the `V_fit` pair universe `(u, v)` (50% from `E_topo`, 50%
random train-node pairs, label-agnostically; R8), build both ego-nets, mask each
side at a scheduled ratio, and
train the decoder to re-decode its masked slots conditioned on `[x_u; z_u; r_u]`, its
own kept slots, *and the partner's kept slots* (a dedicated cross-ego input channel,
with `Π` computed on the fly). This makes inference-time harmonization query
in-distribution conditionals at every round and makes the two per-endpoint conditionals
consistent by construction with one shared decoder — addressing the pseudo-Gibbs
objection: rounds are alternating conditionals of a single trained model, and a
slot-agreement trajectory across rounds is reported as a standard diagnostic, with `Π`
frozen after round 1 within each harmonization run.

**Inference procedure** — `R` rounds (default 2–4; `R = 0` recovers one-shot stitching
so the harmonization gain is directly measurable):

1. **Keep** slots and slot–slot edges by a cosine-scheduled confidence quantile
   (population: per scaffold), where confidence combines existence (`π`, temperature-
   calibrated on held-out real ego-nets — calibration is a requirement, not an option)
   and **content confidence** from codebook quantization distance (§4.1) [R3-W5].
   Grounded slots (`g^k` high) enter as low-re-mask-probability anchors — *reduced*
   probability, not exemption, so a peer-not-partner anchor can still be revised
   [R3-W8].
2. **Re-mask and re-decode** the remainder with the jointly trained cross-conditioned
   decoder.
3. **Scaffold-level budget constraint:** once `Σ_k π_u^k · m_u^k` reaches the
   K-representable budget `min(d̂_u, budget representable in K slots)` within tolerance
   τ_b, the scheduler masks further slot activations and closures for that endpoint
   (EDGE masking pattern). **Scope [unanimous panel finding]:** this guarantees budget
   respect *within the scaffold*; at assembly level the budget acts only through shaped
   marginals (`s3` budget-pressure features), and its transmission is *measured* by the
   degree-calibration diagnostic (§6.4.8), never asserted.
4. A lightweight critic (SID/CID 2503.21592 pattern) modulates re-mask probability of
   individually implausible elements and **may re-mask previously kept slots**
   (Token-Critic-style revisiting) [R3-W5].

**Gradient estimators** [R1-W2]: keep/re-mask decisions use straight-through estimators
during the joint training task; the budget mask is applied with a detached trigger and
a soft penalty mirror (so the constraint surface is felt in gradients); harmonization
runs in the training forward pass (matched to inference, R12) after generator
warm-start. A half-page algorithm box with tensor shapes is a deliverable of gate
§6.0-G4 before implementation.

### 4.4 Module 4 — Stitched-topology-conditioned pair encoder **[rev 3.0, 2026-07-16]**

> **Historical rev-3.0 architecture, superseded 2026-08-02.** The current
> three-component model deletes the separate content pathway; see the active
> successor update in §6.0 and spec §14.4.6–14.4.7. The details below are retained
> only to interpret the completed 2026-07-24 screen.

The rev-2.2 head — frozen B0 `pair_logit` anchor plus logit-level gated-residual
fusion of scalar channels `s1..s4` — is replaced. Motivations: (i) the frozen
anchor made the model read as an extension of an existing scorer, and the Seed-0
exact-quota diagnostic showed the anchored residual collapsing onto B0 (s0-logit
correlation ≈ 1.0, assembled metrics indistinguishable); (ii) logit-level late
fusion is weak methodological novelty for a paper titled *topology-conditioned*.
The full decision trail, reviewer rounds, and literature sweeps are recorded in
`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`.

**Architecture (end-to-end; no pretrained checkpoint anywhere in the model):**

```text
z_pair = Trunk(tok_i, tok_j)          # V3.1-class pair encoder trained FROM SCRATCH:
                                      # raw token sequences → Siamese encoder → pair
                                      # cross-attention → pair_context_gated / abba_max
c_topo = STE(T̂_ij)                    # structure-only stitched-topology encoder:
                                      # 2–3 layer edge-weighted MP over the scaffold
                                      # (star edges, Â_i, Â_j, Π; 4-type anchor labels,
                                      # π, m, soft degrees — NO h, NO g), token-level output
c_cont = ContentTokens(S_i, S_j)      # separate pathway: slot content h, (π, g) tags,
                                      # grounded-identity-match, membership signal
conditioning: in the last N_inj ∈ {1,2} trunk pair-cross-attention blocks, the CLS
              token cross-attends to c_topo and (separately gated) c_cont through
              zero-initialized tanh gates, per direction (AB / BA with swapped anchor
              labels) BEFORE abba_max; AB/BA share STE and attention parameters
p_ij = σ( head(z'_pair) )
```

**Three-null decomposition (checkpoint-exact; all gates residual and zero-init):**
`∅_all_head` (skip STE + both attentions) yields the pair-only `f_logit`;
`∅_topo_head` yields pair+content; `∅_content_head` yields pair+topology. All four
logits come from one checkpoint and are reported with every headline table; training
uses per-pair multiplicative branch masks (Modality-Dropout lineage 2005.13616 ⊕;
branch-competition analysis Huang et al. ICML 2022 ⊕ — mechanism novelty not claimed),
evaluation uses hard bypasses, and their exact equality is a required unit test, as is
`p(i,j) = p(j,i)` under every null.

**Channel mapping (rev 2.2 → rev 3.0):**

| rev 2.2 channel | rev 3.0 role |
|---|---|
| `s0` frozen anchor | retired from the headline; the jointly-trained trunk's `f_logit` is the node-intrinsic evidence. The **frozen-s0 variant is retained as an ablation arm**, where the SHOT frozen-hypothesis reading (2002.08546) now exclusively lives |
| `s1` membership | content pathway (`c_cont`) input — pair/content compatibility, deliberately excluded from the topology claim |
| `s2` closure (SEAL/NCN signal 1802.09691, 2302.00890; BUDDY fallback 2209.15486) | registered diagnostic + representation-probe target (alignment consistency); the STE sees the same `Π, π, m` structure and may learn it |
| `s3` community/capacity | Stage-2 structural inputs to the STE (`F_u` block features, `d̂` budget-pressure), unchanged in spirit |
| `s4` scaffold GNN readout | **promoted into the STE**: same edge-weighted-GNN lineage, token-level states instead of a pooled scalar, now the headline conditioning source |

**Attribution is pre-registered** [supersedes the DA-M9/R3-W10 collinearity block]:
(i) *pathway attribution* — the topology-representation claim must survive
content-pathway removal (pair+topology retains a registered share of the full-model
gain over the matched `B0-e2e` arm), else the honest conclusion is content-side
information; (ii) *structure specificity* — a battery of edge-shuffle,
edge-removal-to-DeepSets, cross-pair scaffold shuffle, matched-capacity
non-message-passing, and **degree-preserving rewiring** controls (the decisive answer
to "is the STE encoding topology or a continuous latent code disguised as a graph",
given π/m/Â/Π are all feature-derived); (iii) *representation evidence* — frozen-STE
linear probes to real degree / ego density / clustering / `Π`-consistency on held-out
train-side nodes, **with degree-partialled variants** (degree bias dominates LP
signals: 2405.14985, 2310.04612); (iv) the pre-registered prediction that topology
conditioning adds value precisely on low-FCR benchmarks and tail-degree strata
(Cold Brew FCR, §6.4.9) stands unchanged — if it does not, `Ours → B5` and §6.5's
decision rules apply. **Anti-shortcut controls:** the `∅_content`
generator-conditioning contrast (§4.2 — a different mechanism from the `_head` nulls
above) remains an E5 integrity control (TDE 2002.11949; Neural Motifs 1711.06640).

**Novelty scoping (binding for the paper text; 2026-07-16 sweep, local vault +
verified external arXiv).** The claim is **novel overall composition** — never
per-component unprecedentedness, and never "first to generate structural context for
unseen nodes" (Leap ⊕ 2503.03331 already grafts predicted edges for inductive LP):

> *Dual imagined ego-nets are differentiably aligned and stitched into a generated
> local scaffold whose structure-only token representation conditions a queried-edge
> pair encoder under the strict zero-edge inductive protocol.*

Component ancestry (each reported as ancestry / prior usage / difference): set-decoder
imagination (DETR 2005.12872 — detection queries → neighbor slots with existence,
multiplicity, adjacency); anchor labeling (labeling trick 2010.16103 — observed
subgraphs → generated slots); OT alignment (GOAT ⊕ 2111.05366, SLOTAlign ⊕ 2301.12721
— align two *observed* graphs as the end task → internal differentiable stitch of two
*imagined* ego-nets, the most distinctive element); zero-init gated cross-attention
(Flamingo ⊕ 2204.14198 — modality injection into a frozen LLM → structure-token
injection into a from-scratch pair encoder with a checkpoint-exact bypass); graph-token
conditioning (GraphToken ⊕ 2402.05862, GraphGPT ⊕ 2310.13023 — LLM reasoning → binary
edge logit; CAM tokens ⊕ 2405.19375 — cross-attentive modulation for linkset
prediction over *observed* tokens, so cross-attention itself is not claimed);
FiLM-style modulation (GNN-FiLM 1906.12192). The SEAL family (1802.09691, 2010.16103,
2302.00890, 2209.15486) reads observed subgraphs — the defining protocol delta. No
exact match for the composition was identified in the reviewed corpus; a limited
search cannot prove absence.

### 4.5 Training objective (maps 1:1 onto the locked objective)

```text
L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon
```

**Full loss tree with owners [R1-W1]** — every sub-loss lives under exactly one locked
lambda with an interior weight; interior weights are pre-registered and swept under the
HPO-parity protocol (§6.5):

- `L_edge` (master): BCE on queried pairs (negatives after masking; hard-negative
  protocol per §6.4). **Gradient routing:** `L_edge` gradients reach the ego-decoder —
  the hallucination literature's criterion is *usefulness to the classifier*
  (1801.05401; f-CLSWGAN's L_CLS 1712.00981). **Covert-channel watchdog [R1-W4]:**
  because `L_edge` can turn slots into a discriminative side-channel that `L_real`'s
  batch-level marginal matching cannot prevent (marginal matching does not identify the
  *conditional* p(ego|x)), imagined-ego-net **fidelity on held-out training nodes**
  (slot recall@K of true neighbors, degree calibration, slot-adjacency vs true local
  clustering) is a required report (§6.4.8) — the direct evidence for the §4.6
  mechanism story.
- `λ_recon · L_recon`: Hungarian feature matching + existence BCE + multiplicity NLL +
  degree NLL + group-level slot-adjacency loss + CVAE KL (free-bits) + **NOCD BP-NLL**
  + **VQ commitment/codebook losses** + **code-supervision head** (ego-net statistics
  from `z_u`) + **code-usage entropy regularizer** (the two former orphans, now owned
  here) + the joint two-ego harmonization task (§4.3b). All structural targets on the
  shared training interactions, using their loopless topology projection (R9).
- `λ_real · L_real` (distributional): energy distance between generated and real
  ego-net statistic vectors (degree, clustering, code histogram, motif conductance);
  energy distance in untrained random-GIN embedding space (2201.09871); the **seam
  term** on stitched overlap regions vs real 2-ego-net unions, with references sampled
  **label-agnostically** (random node pairs stratified by feature similarity, labels
  marginalized — never adjacent-only pairs [R1-W5-ii]); optional small GNN
  discriminator (DANN 1505.07818 gradient-reversal variant).
- `λ_ssl · L_ssl`: consistency of imagined ego-nets under feature noise and
  candidate-pool resampling. **Known tension [R1-W9]:** pool-resampling consistency
  penalizes reliance on `G(u)` and can collapse the grounding gate; the resolution is
  to apply pool-consistency only to *ungrounded* slots and monitor mean `g^k` as a
  registered diagnostic with its own risk row (§7).
- **Balancing strategy:** fixed interior weights from a pre-registered sweep, with
  uncertainty weighting as the registered alternative; gradient-norm monitoring per
  loss family; a documented tuning budget equal across `Ours` and all baselines
  (HeaRT's central complaint; §6.5).
- **Curriculum:** warm-start Modules 1–2 with `L_recon` only (decoupled generator
  pre-training per LA-GNN), then add the joint harmonization task, then full joint
  training; model selection / early stopping of generative components on
  edge-prediction validation metrics (NetGAN's VAL-CRITERION).

### 4.6 Mechanism-to-failure-axis map — with the transmission step stated

Every mechanism below constrains the **scaffold** exactly and reaches the **assembled
graph** only by shaping per-pair marginals; the assembled effect is a mechanism-backed,
*measured* tendency (diagnostics §6.4.8), bounded above by the edge-independence
ceiling (§6.0-G2). "Guarantee" refers to the scaffold level only [unanimous panel
finding].

| E2 failure (assembly-level) | Scaffold-level mechanism | Transmission to assembly | Measured by |
|---|---|---|---|
| BFS-macro RD 0.422345 despite global simple-edge RD 0.997710 | hard K-representable budget masking in harmonization | budget-pressure features in `s3` shift marginals | degree-calibration diagnostic (E[d̂] vs realized assembled degree) |
| Degree MMD ratio 13.0768 | cardinality tie `Σπ·m ↔ d̂` (doubly supervised) | degree-aware marginals across each node's queries | assembled degree MMD + per-node calibration curve |
| Clustering MMD ratio 11.9273 | slot–slot adjacency + closure channel `s2` + motif-conductance code supervision | closure evidence raises/lowers triangle-completing marginals | assembled clustering MMD + edge-independence ceiling comparison |
| Spectral MMD ratio 18.0931 | community codebook + block prior | block-consistent marginals shape mesoscale spectrum | assembled spectral MMD (held-out kernel per §6.4.2) |
| Pair-to-topology gap | decision conditioned on realism-trained, seam-harmonized context | better-calibrated marginals from topological supervision | joint edge+assembled table vs `B0+cal`, `B3-dist`, `B3-full`, B5 |

Anti-grab-bag rule (binding): a mechanism stays in the model only if it owns a row here
and an ablation arm in §6.3; everything else is cited once in related work.

### 4.7 Complexity — honest comparator and commitments

Per node (cached): one tokenizer pass + one slot decode — `O(K·(d + |G(u)|))`.
Per pair: OT stitch + `R` partial re-decodes + GNN over `≈ 2K+2` nodes.

Three commitments replace the previous qualitative claims [R3-W9, R1-W10, DA-M20]:
(i) the deployment comparator is **B0** (a dot product on cached embeddings), not the
old scaffold; EgoStitch's per-pair marginal cost is plausibly 10²–10⁴× B0 and will be
reported as a **FLOPs and wall-clock table** (per-node cached; per-pair marginal; full
candidate-universe assembly for Benchmark-A with its size stated) at R ∈ {0, 2, 4};
(ii) the `R = 0` cached one-shot variant is a full metric row, exposing the
realism-vs-cost frontier; (iii) if a B0-prefilter cascade is the deployment mode,
assembled metrics are reported *under the cascade* — a cascade dilutes topology gains
because most assembled edges become B0 decisions, and hiding that would be
self-deception. Amortization claim, corrected: per-*node* work amortizes; per-*pair*
work strictly increases over both B0 and the §0 contract. The canonical attack to
pre-empt [rev 2.2]: NBFNet's Table 1 reports ≈1 month of wall time for per-query
subgraph methods (SEAL/GraIL) on a single KG test set — the paper must show the
per-endpoint cache plus `R = 0` row keeps EgoStitch out of that regime.

---

## 5. Novelty positioning

### 5.1 The claim, scoped

Review-based claim (two-pass: full-vault fan-out review + targeted arXiv novelty-risk
queries + 5-persona design review, 2026-07-08; an absence claim, so stated as "not
found" rather than "does not exist"): **our review found no model that, for an unseen
node pair with no observed edges and no target-graph access, generates latent
ego-network structure (neighbor slots + intra-neighborhood adjacency + degree budget)
conditioned on frozen features, harmonizes the two generated neighborhoods into a
per-query scaffold, fuses a community-topology representation with node-intrinsic
features for a binary edge decision, and is graded jointly on edge metrics and
assembled-graph realism.** A final automated pass (`/novelty-check`, including its
cross-model verification phase, which failed on infrastructure this run) must be rerun
before submission.

The unifying structural defense, stated once and reused everywhere: **every close prior
requires the query node's observed edges at inference; none can score a pair of
zero-edge nodes; none grades the graph its predictions assemble into.** The final G1
result is AUROC 0.705519 / AUPRC 0.730260 on degree-corrected negatives, falling to
0.583965 / 0.626649 on hard heuristic negatives and 0.569560 / 0.617475 on hard feature
negatives, while global simple-edge RD is `0.997710` but official BFS-macro
GS/RD are `0.312151/0.422345`. The failure survives
G1. B0-alt independently reaches 0.693603 / 0.732509 on degree-corrected negatives but
assembles with global simple-edge RD `0.998739`, BFS-macro GS/RD
`0.345802/0.450793`, and MMD ratios
`15.8304/13.4718/23.4734`, closing the required architecture-independence arm. PA-null
wins in some edge regimes and remains a mandatory control.

Known weakness of the claim's *shape* [EIC, DA-M21]: it is a conjunction of qualifiers,
and conjunction-novelty weakens with each conjunct. The mitigation is §6's controls:
the composition must be shown load-bearing (ablations) and the task protocol must be
shown meaningful (E7, promoted to load-bearing), or the novelty reduces to system
assembly.

### 5.2 Settings taxonomy (place the task before defending the model) [R2-W1]

| Setting | Test-time structural input | Representative work |
|---|---|---|
| Transductive LP | full observed graph incl. test endpoints | VGAE 1611.07308; SEAL 1802.09691; NCN 2302.00890 |
| Inductive-with-neighborhood | unseen nodes arrive **with observed edges** | GraphSAGE 1706.02216; temporal/new-node streams TGN 2006.10637 ⊕, TGB 2307.01026 ⊕ |
| KG-inductive | unseen entities, but an **observed inference graph** around them | GraIL 1911.06962 ⊕; NBFNet 2106.06935 ⊕; ULTRA 2310.04562 ⊕ (zero-shot across KGs, still requires the inference graph); IGMC 1904.12058 ⊕ |
| Attribute-only inductive (independent scoring) | frozen features only; per-pair similarity | Graph2Gauss 1707.03815 ⊕; DEAL 2007.08053; UPNA ⊕ 2307.08877 |
| **This work: strict zero-edge inductive, topology-conditioned** | frozen features only; **generated** local context; assembled output graded for realism | — |

The KG-inductive community owns the phrase "inductive link prediction"; the table's
third row is the pre-emptive answer to "how is this not GraIL/ULTRA?" — in that setting
structural context is *given* at inference and methods reason over it; here it does not
exist and must be generated. Quote-backed anchors (full-text verified, rev 2.2):
GraIL assumes "the local graph neighborhood of a particular triplet in the KG will
contain the logical evidence needed" and uses no node attributes — a zero-edge pair
yields an empty enclosing subgraph and no input at all; NBFNet's pair representation
is defined over the set of observed paths `P_uv` ("generalize to entirely new graphs
*without node features*" — for a zero-edge pair, `P_uv = ∅`); ULTRA predicts "based on
the incomplete inference graph `G_inf`" and its first step lifts `G_inf` to a graph of
relations, which does not exist without observed edges; IGMC states verbatim that it
"does not address the extreme cold-start problem, as it still requires an unseen
user-item pair's enclosing subgraph." Note the symmetry: that lineage is
structure-only/features-absent; ours is features-only/structure-absent — the two
regimes are information-theoretic complements, which makes "why not just run ULTRA?"
self-answering. On the attribute-only row, UPNA's abstract claims to learn "a
significant part of the latent graph generation mechanism"; the distinction is that
UPNA still scores pairs independently with no per-query generated context and no
assembled-graph grading. The zero-edge setting is strictly harder in test-time
information and is the regime where all observed-context machinery is undefined.

### 5.3 Threat matrix (closest prior work, and the exact delta)

| Closest prior | What it does | Why it is not EgoStitch |
|---|---|---|
| **Graph Gestalt** (2106.15239) | diagnoses local link-reconstruction vs global realism divergence; fixes GVAEs with a kernel-MMD ELBO regularizer; realism co-training also improves LP | **priority on the dissociation observation — cited in the E2 framing**; transductive GVAE on an observed graph: no unseen nodes, no per-query scaffold, no generation. Neutralized structurally *and* empirically: its remedy is imported as the `B3-dist` arm and inside `B3-full` (§6.2) |
| **Chanpuriya et al.** ⊕ (NeurIPS 2021, 2111.00048) | proves `E[Δ] ≤ (√2/3)(Ov·V)^{3/2}` for any edge-independent model, tight up to constants — matching real triangle density at fixed volume forces overlap toward the memorization end of the tradeoff; exact identities `E[Δ] = tr(P³)/6`, `Ov·V = Σp²` make the bound computable for any concrete assembly matrix | not a competitor but the **boundary of the playing field**: simultaneously the strongest theoretical backbone for RQ1/E2 (their own CELL/CORA-ML experiment — ≤1,461 generated vs 2,802 real triangles — is an external companion to E2) and the ceiling on what any locked-contract method (ours included) can repair — used in both roles (§1.6, §6.0-G2). Caveat that changes the gate: overlap is *self-resampling* agreement, so a thresholded (deterministic) assembly has `Ov = 1` and a vacuous bound — the ceiling is a curve over overlap, evaluated at the soft scorer's measured `Ov(P)`; informative features legitimately raise sustainable overlap on unseen nodes, so the ceiling scales with how deterministic the feature→edge map is allowed to be |
| **NCNC** (ICLR 2024, 2302.00890) | completes unobserved common-neighbor structure with a link predictor, then scores | transductive completion among observed nodes; no latent neighbor generation, budget, community prior, or assembled evaluation. Its incompleteness finding is *evidence for* our premise |
| **Cold Brew** (ICLR 2022, 2111.04840) | virtual neighborhoods for zero-edge nodes = attention over existing training-node embeddings | no generated structure (no slot adjacency, no existence probabilities, no budget); node-level embeddings, not per-pair decisions; no realism objective |
| **DEAL** (IJCAI 2020, 2007.08053) / **Graph2Gauss** ⊕ (ICLR 2018, 1707.03815) | attribute-only inductive LP; independent pairwise similarity at inference | the B0 template family (Graph2Gauss predates DEAL); no scaffold, no generation, no assembled metrics; mandatory baselines with the falsifiable prediction that they exhibit the E2 failure mode |
| **NRI** ⊕ (ICML 2018, 1802.04687) | infers interaction graphs over entity sets from observed *trajectories*, evaluated as a graph | the canon's closest "imagine topology from features" relative; but it requires per-instance dynamical observations (trajectories) as test-time evidence, infers one latent graph per system rather than per-query context for a binary pair decision, and has no inductive zero-edge pair protocol |
| **Latent Graph Diffusion** (NeurIPS 2024, 2402.02518) | "unifies generation and prediction": prediction = conditional generation (graph inpainting) in a whole-graph latent space | conditions on *observed test-graph adjacency* (inadmissible under our protocol); one whole-graph latent, "predict once per graph"; prediction experiments in the main text are graph regression and transductive node classification [verify against camera-ready before submission — flagged, not yet page-verified] |
| **FLEX** ⊕ (2507.11710, preprint) | GGM for OOD link prediction: SEAL-labeled k-hop subgraphs around *training* links generated by a SIGVAE, counterfactually shifted (KL-target penalty), adversarially co-training a GNN — verbatim "we apply graph generation as a data augmentation method"; GGM not invoked at inference | full-text verified: generation is train-time augmentation from *observed* neighborhoods; test queries answered by the tuned GNN on an observed (shifted) graph; Hits@K only, no assembled-graph grading. Bonus: FLEX documents that naive generated subgraphs are uniformly over-dense and fixes it with a probability threshold — independent evidence for our hard degree budgets |
| **EDGE** (ICML 2023, 2305.04111) | degree-first factorization `p(A,d)=p(d)p(A|d)` with hard degree masking | whole-graph unconditional generation; our budget is **EDGE's principle amortized**: feature-conditional, per-endpoint, per-query, inductive, at ego scale, used as classifier evidence |
| **TGSBM** (2601.20646) | amortized variational OSBM latents decoded into edge probabilities | encoder attends over the node's *observed edges* (+ expander edges): a zero-edge node has no attention neighborhood; transductive evaluation; block prior is the whole likelihood, not a fused channel |
| **VQGraph** (ICLR 2024, 2308.02117) / **GFT** (NeurIPS 2024, 2411.06070) | discrete codebooks over graph structure (node-code distillation targets; computation-tree vocabulary) | recognition/distillation codebooks; VQGraph's student MLP *can* code a zero-edge node from features [R2-W5a], but neither work generates topology, supervises codes with ego-net statistics, nor addresses edge prediction — see the scoped claim in §4.1 |
| **DGM** (TPAMI 2022, 2002.04999) / **NodeFormer** (2306.08385) / **DCM** (ICLR 2024, 2305.16174) | differentiable, *inductive* graph construction (Gumbel-Top-k kNN; kernelized Gumbel-Softmax; α-entmax over cell complexes) | selection among **observed** nodes for node-level tasks; one global graph, classification-only supervision, no realism grading, no per-query scaffold. Differentiable construction per se is solved technology — the claim is never "we learn structure differentiably". (DCM is a cell-complex paper; its account of DGM's fixed-k topology should be quote-checked before citation in print) |
| **MaskGAE / S2GAE / Bandana / GraphMAE** ⊕ (2205.10053, WSDM'23, 2402.03814, 2205.10803) | masked structure/feature reconstruction (per-edge Bernoullis; degree decoder; continuous bandwidths; feature-space targets) | all reconstruct *observed* structure from a *partially visible* graph: none emits a variable-size neighbor set, none uses matching losses, none trains the total-masking zero-edge limit, all are transductive representation learners. Concede the fragments; claim the composition + protocol |
| **MoG** (ICLR 2025, 2405.14260) | per-node ego-graph decisions re-assembled globally (sparsifier experts) | deletion of observed edges for node-level efficiency; the ego-decompose-and-reassemble skeleton exists here, so our delta rests on generation + the inductive edge protocol |
| **GSR** (WSDM 2023, 2211.06545) | "GSL is essentially link prediction"; ego-subgraph contrastive pretraining, one-shot refinement | pretext for refining one observed global graph for node classification; transductive. Its MoCo-style multi-view edge pretraining is imported as an encoder pre-training recipe |
| **NetGAN** (ICML 2018, 1803.00816) / **SIG-VAE** (1908.07078) | early generation-realism/LP couplings | transductive, single-graph, not per-query; NetGAN's edge-independence critique (Rendsburg et al., "NetGAN without GAN," 2020 ⊕ [cite by name]) generalizes into the Chanpuriya bound above. SIG-VAE has priority on "BP decoder → realistic statistics": cite, never claim that connection as new |
| **LA-GNN** (ICML 2022, 2109.03856) / **GraphSMOTE** (2103.08826) / **GAR** (SIGIR 2022) + cold-start recsys (DropoutNet; Heater [venue-verify]; Meta-Embedding — SIGIR 2019, 1904.11547 ⊕, venue confirmed rev 2.2: a meta-learned feature→initial-ID-embedding generator for zero-history ads) | conditional neighbor-*feature* generation as augmentation; synthetic-node edges for class balance; cold-item embedding generation | features not structure / augmentation not inference context / single embedding not neighborhood. Cold-start recommendation *is* zero-edge bipartite edge prediction: either a bipartite benchmark joins E6 or the scope exclusion is stated explicitly [R2-W9] |
| **New Node Prediction** ⊕ (2401.05468, arXiv preprint) | zero-shot out-of-graph all-links prediction for one isolated node | task-adjacent but **semi-inductive**: the isolated node is linked *into a fully observed graph* available at prediction time — unlike our node-disjoint, no-test-graph regime; no generated scaffold, no realism grading. (Its related work also records the VGNAE observation that GAE embeddings of zero-degree nodes collapse toward zero — a one-line empirical answer to "why not plain GAEs?") |
| **HiGGs** ⊕ (2306.11412, cite as arXiv) | hierarchical large-graph generation: community-level graph sampled first, per-community subgraphs generated, then inter-community edges filled by an edge-predictive diffusion model | whole-graph *synthesis*; its stage-three "stitch sampled communities into one big graph" is the mechanical cousin of our stitching, but it stitches sampled communities to synthesize output, whereas we stitch imagined ego-nets to answer `edge(u, v)` under a no-test-graph protocol; no queries, no inductive nodes, realism graded on sampled graphs only |

### 5.4 Theory usage — scoped statements [R2-W4, DA-M13]

- **Labeling trick (2010.16103) and Distance Encoding (2009.00142):** grant the most
  expressive pair-aware readout *of the generated context `T̂_ij`*. Both theories are
  proven with respect to actual graph structure; neither certifies that the generated
  context carries true structural evidence — that is an empirical property of
  `L_recon`/`L_real` fidelity, measured by the §6.4.8 diagnostics. The paper never
  writes "provably most expressive" without this qualifier. **P-GNN (1906.04817) is a
  separate family** — anchor-based *absolute positional* encodings, sampled
  independently of the target pair, not a target-conditioned labeling scheme; its
  structure-aware vs position-aware distinction also scopes what an imagined ego-net
  can restore: local structural evidence, not global position.
- **γ-decaying locality (SEAL 1802.09691):** applies to observed enclosing subgraphs.
  The transferred claim is conditional: *if* generated ego-nets match the true local
  distribution, then local (1-hop-stitched, K-slot) context suffices for CN/AA-type
  evidence. The antecedent is exactly what the fidelity diagnostics test.
- **Automorphic-node limit (2209.15486):** motivates the gap as consistent with a known
  expressiveness limit; it binds every feature-only scorer including ungrounded
  EgoStitch (§1.9) — the architecture's answer is ease-of-learning under topological
  supervision plus optional grounding, not a claim to exceed the feature-function
  class.
- **Manenti et al. (2405.19933):** motivates distributional losses; its identifiability
  assumptions do not literally hold here (§1.4).
- **Joint-evaluation priority (C4):** graded network-model realism is classical (ERGM
  goodness-of-fit: Hunter, Goodreau & Handcock 2008, JASA ⊕; latent-space models: Hoff
  et al. 2002 ⊕); Graph Gestalt and NetGAN's VAL-CRITERION are the deep-learning
  instances. The scoped claim, used in all four repository documents: *the first
  **protocol-gated** joint edge + assembled-realism evaluation for **strict zero-edge
  inductive** link prediction* — [protocol-Δ: docs 01–03 currently state the unscoped
  version and need this qualifier].

### 5.5 Mechanism imports (cited, none changing the novelty of the combination)

Labeling trick (2010.16103) + Distance Encoding (2009.00142 ⊕) [target-conditioned
structural]; P-GNN (1906.04817 ⊕) [anchor-based positional — related, distinct family];
automorphic-node motivation + sketching (2209.15486); γ-decaying locality (1802.09691,
conditional form); NCN closure form (2302.00890); NOCD block prior (1909.12201);
DC-SBM degree correction (Karrer & Newman 2011 ⊕); EDGE degree factorization + masking
(2305.04111); MaskGAE degree decoder (2205.10053); set decoding with matching
(2005.12872, 1906.06565); dynamic queries (2108.08839); feature-space set targets
(2209.14860; GraphMAE 2205.10803 ⊕); retrieval-grounded generation (2204.11824);
confidence-scheduled parallel refinement (2202.04200) with critic revisiting
(Token-Critic pattern); seam harmonization motive (2201.09865); critic-guided
re-masking (2503.21592); distributional latent-structure motivation (2405.19933);
realism metric alignment (2201.09871; companion pitfalls paper O'Bray et al.
2106.01098 ⊕); whole-assembly critics (1604.07379, 1803.00816, 2106.15239);
counterfactual control (2002.11949); usefulness-over-realism criterion (1801.05401,
1712.00981); frozen-hypothesis anchoring (2002.08546).

Honest disanalogies of the "amodal completion / inpainting for graphs" framing, stated
preemptively [R3]: (i) in image inpainting the surrounding context is observed ground
truth, whereas at our test time *both* ego-nets are hallucinated — grounded slots are
the only "known pixels," and they are retrieval-quality-limited (§4.2's partner-vs-peer
gate is the mitigation); (ii) pix2gestalt's zero-shot power comes from an
internet-scale pretrained prior that graphs lack; (iii) the conditioning
signal-to-entropy ratio is far worse here — one frozen feature vector conditions a
high-entropy neighborhood, closer to unconditional generation than any CV completion
task, which is why the diversity diagnostic and the mode-seeking risk row exist; (iv)
images carry generic low-level regularity (smoothness, equivariance) with no
dataset-agnostic graph analogue; and (v) graphs lack a cheap perceptual oracle for
realism — the discriminator probe (§6.4.4) is the compensation and is promoted
accordingly. The analogy is used as mechanism transfer, not decoration.

---

## 6. Fit to the experiment protocol

The **[protocol-Δ]** items below were approved and merged into
`03-experiment-protocol.md` on 2026-07-09; the tags are retained as provenance markers
for which protocol clauses originate here.

### 6.0 Pre-implementation gates (run before any EgoStitch code) **[protocol-Δ]**

Ordered; each gate has a stop condition. These convert the panel's fatal-vs-fixable
verdict into process [DA verdict, EIC concern 1, R1-W8].

- **G1 — E2 hardening.** Re-run E2 with: one frozen scorer family (same as `Ours` will
  use), one candidate universe, one canonical metric normalization, a true threshold
  sweep (recall/density-vs-MMD curves), easy *and* hard negatives, a second-architecture
  replication (B0-alt, as actually run — see the retirement note in the Result below),
  a real-vs-real MMD noise floor, bootstrap variance over buckets/seeds, and official
  official Graph Similarity / Relative Density over the fixed induced subgraphs. Also record the full-candidate-universe
  imbalance view [DA-m17]. *Stop condition:* if the gap substantially closes under hard
  negatives or calibrated thresholds, the motivation is dead as stated and the project
  pivots to the evaluation/benchmark paper. **Result (2026-07-13): passed and closed;**
  B0-alt preserves the topology gap under the canonical evaluator. B0-alt was G1's
  architecture-independence replication, run once for that closed gate; by owner
  decision (2026-08-03) it is retired from the forward baseline set (B0 is the sole
  baseline) and its implementation was removed from the code tree — provenance in
  `docs/results/E2-pair-to-topology-gap.md`.
- **G2 — Edge-independence ceiling check (rev 2.2 semantics).** Using the reference
  graph and cached B0 scores, with Chanpuriya 2111.00048's exact identities:
  (i) compute the assembly's operating point exactly — `V(P) = Σ p_ij`,
  `Ov(P) = Σ p_ij² / Σ p_ij`, `E[Δ] = tr(P³)/6`; (ii) plot the **ceiling curve**
  `E[Δ]_max(ω) = (√2/3)(ω·V)^{3/2}` over overlap ω at matched volume, with the
  clustering ceiling from the exact triangle numerator over the assembly's
  expected-degree denominator `Σ d̂_i(d̂_i − 1)` (Theorem 6 is big-O only — do not use
  it directly); (iii) mark B0's measured `Ov(P_soft)` on the curve, plus the
  minimum-required overlap to reach the reference triangle count,
  `Ov_min = (3Δ*/√2)^{2/3} / V`; the odds-product model's convex-combination dial
  `P̃ = (1−ω)P + ωA` sweeps the curve empirically. **Semantics:** overlap is
  self-resampling agreement — a hard-thresholded assembly has `Ov = 1` and the bound
  is vacuous there; the honest reading is that the ceiling constrains *stochastic,
  calibrated* assemblies at their measured overlap, and informative features
  legitimately raise the overlap sustainable on unseen nodes without training-graph
  memorization. Report the curve next to all assembled metrics thereafter. *Stop
  condition:* if the reference statistics require `Ov_min` far above what any honest
  feature-conditioned scorer attains on unseen nodes (calibration-checked), the locked
  per-query contract cannot express the fix; flag for a locked-decision discussion
  (assembly-time coupling would be a different paper) rather than building a model
  that cannot succeed.
- **G3 — Oracle first [DA-M10].** Run the Oracle row (observed-neighborhood scaffold)
  before implementation: it calibrates whether the G1 BFS-macro GS `0.312151` is poor, bounds all possible
  gains, and separates "conditioning is the missing ingredient" from "features are
  insufficient." *Stop condition:* Oracle ≈ B0 on assembled metrics ⇒ feature
  insufficiency; topology conditioning cannot help; pivot.
- **G4 — Specification freeze.** The algorithm box (Stitch + Harmonize pseudocode
  with tensor shapes, OT cost/ε, quantile schedule, budget tolerance τ_b, gradient
  estimators) and the loss tree with interior weights are **delivered in
  `05-egostitch-spec.md` and signed off (2026-07-09)**; the
  spec is the implementation contract and deviations require a spec edit first.
  *Stop condition:* none — this is a deliverable.
- **G5 — Minimum-viable-model milestone [EIC].** Stage 1: imagination + degree budget
  + closure channel only (no codebook, no harmonization, no CVAE) vs B0/B1/B5/`B0+cal`.
  Stage 2 adds codebook + s3; Stage 3 adds harmonization + seam loss. Each stage must
  beat the previous on the pre-registered criteria or the added mechanism is cut per
  the §4.6 rule.
  *Update 2026-07-14 (pre-implementation):* the Stage-1 gate runs against B0 and
  `B0+cal` only — B1/B5 comparison rows are deferred to E3 with their implementations;
  the Stage-1 subset is pinned in spec §13 and the acceptance criteria are
  pre-registered in `docs/registrations/g5_stage1_preregistration.json`
  (protocol §5.0.5).
  *Execution update 2026-07-15:* Seed 0 completed formal training (best validation
  AUROC/AUPRC `0.945766/0.951966`); Seed 1 stopped at the artifact performance gate
  and Seed 2 was not run. Candidate scoring, fidelity, assembled-graph evaluation,
  and the then-required three-seed decision remained pending. This was not a G5 result.
  *Protocol update 2026-07-16:* after inspecting the later exact-quota Seed-0
  diagnostic, Stage 1 was re-scoped to a fixed-one-seed engineering screen using
  deterministic point-estimate dominance. The inspected artifact is not retroactively
  rebound; a new-hash run plus fidelity/cost reports is required for the screening verdict.
  E1/E3 retain multi-seed Holm inference.
  *Headline revision 2026-07-16 (rev 3.0; historical):* the pending frozen-s0 screen keeps its
  registered contract unchanged and becomes the motivating arm for the §4.4 e2e
  redesign. The **next** binding Stage-1 build is the stitched-topology-conditioned
  pair encoder with a five-arm screen scope — full model, matched `B0-e2e`/f-only,
  pair+topology (`∅_content_head` permanent), one structure-destroyed control
  (within-pair `Â`/`Π` shuffle), and branch-dropout `p = 0` — everything else (E2E
  `B3-full`/`B5`, conditioning-depth rungs, the remaining structure battery) is E1/E3
  scope. Landing sequence and registration requirements: design doc §8 and spec §14.
  *Binding result 2026-07-17:* the replacement fixed-Seed-0 frozen-s0 screen
  completed and returned `cut`: all three topology-dominance criteria failed while
  both guards passed. The selected epoch-1 checkpoint stayed near S0; a diagnostic
  epoch-30 checkpoint moved ranking and passed matched GS, but still failed matched
  RD/clustering and worsened degree/spectral MMD. Per the registered failure reading,
  frozen-s0 scalar fusion is retired to motivating-arm + ablation status. The rev-3.0
  conditioned encoder then became the active G5 build line; see
  `docs/results/G5-stage1-seed0-20260717.md`.
  *Successor implementation update (rev 3.2):* the five-arm rev-3.0 scope above is
  historical; before the 2026-08-10 reset, this arm moved to the seven-arm
  direct-run screen in spec §14.4.6–14.4.7: five trained checkpoints plus two
  scoring-time controls. The content pathway is deleted, so `pair_topology` is
  retired as identical to `full`. There is no registration or plan-identity gate, and
  no training-data identity field on checkpoints or score artifacts (spec §12,
  2026-08-03); separating shared-interaction runs from older 80/20-partition ones is
  owner-side.

### 6.1 Method rows

- **Historical `Ours` row:** now named the retrieval-grounded EgoStitch arm, not the selected method.
- **Earlier design becomes an ablation:** generator off + grounding on + thresholded
  scorer edges reproduces the §0 contract exactly — register as ablation arm
  `E4.10: retrieved-thresholded scaffold` (also the natural bridge baseline).
- **Frozen-s0 EgoStitch becomes an ablation [rev 3.0]:** the rev-2.2 anchored
  residual head (frozen B0 `s0` + gated scalar fusion) is retained as an E4 arm; the
  matched-training pairwise-only control is `B0-e2e` (trunk with all conditioning
  permanently bypassed under identical data/negatives/schedule/optimizer/seed/HPO) —
  canonical B0 and `B0-e2e` are distinct rows and are never conflated.
- **B4 disposition [R1-W11-i]:** blueprint/methodology B4 (latent-topology model
  without queried-edge conditioning) is subsumed by the E4 "generation-only, no edge
  supervision" arm (methodology ablation 9), now explicitly registered as `E4.11`;
  the protocol's baseline table should say so rather than silently dropping B4
  **[protocol-Δ]**.

### 6.2 Baseline ladder extensions **[protocol-Δ]**

- **B5 — neural-SBM residual** (from Approach C): isolates "block prior alone"; also
  the project's null hypothesis (§3C).
- **B0+cal — B0 + calibrated assembly:** post-hoc temperature/Platt calibration plus
  density- and degree-sequence-matched thresholding of B0 scores. Answers "would
  trivial assembly calibration recover the MMD gains?" Applied *also on top of `Ours`*
  [R3-W3]: if global calibration still helps `Ours`, the per-pair budget did not do
  the assembly-level work.
- **B3-dist — B0 fine-tuned with a Gestalt-style kernel-MMD realism regularizer**
  (2106.15239's differentiable kernels): answers "does a one-line distributional loss
  close the gap without topology conditioning?"
- **B3-full — the Ockham arm [DA-C5]:** B0 architecture + *all* of EgoStitch's
  auxiliary supervision as multi-task heads (degree NLL, BP-NLL, ego-net statistic
  prediction, distributional statistic loss) + calibrated assembly — every training
  signal, none of the generative machinery. **This is the decisive control for the
  §Thesis inductive-bias claim.** The E2 §5 cached preview (topology-aware loss alone
  reaching degree MMD 7.18 / clustering MMD 8.28 at AUROC 0.686) is hereby treated as
  a first-class hypothesis in `B3-full`'s favor — it is re-run under G1's canonical
  normalization with the same priority as the headline gap numbers, not deflated by a
  caveat the headline numbers share [DA-C5's asymmetry objection, accepted].
- **DEAL** (2007.08053) and **Graph2Gauss** ⊕ (1707.03815) as external attribute-only
  baselines, with the falsifiable prediction that they reproduce the E2 failure mode.
- **PA-null** [rev 2.2, from 2405.14985]: the preferential-attachment null
  `s_ij = k_i·k_j` (degrees from training-side statistics/d̂), reported with each
  benchmark's degree heterogeneity σ (log-normal fit). "Beats PA under the stated
  negative regime" is a validity precondition for any edge-metric claim — the null
  averages AUC 0.83 and beats half the field under uniform negatives.
- **Odds-product** [rev 2.2, from 2111.00048 §3]: the degree-sequence-respecting
  edge-independent baseline (`P_ij = σ(ℓ_i + ℓ_j)` fitted to the expected degree
  sequence) — the cheapest model that honors degree budgets with zero topology
  conditioning; also supplies the G2 overlap dial.
- **TGSBM** (2601.20646): discussed in related work; if a feature-only-encoder
  re-implementation is feasible it strengthens the B5 family — otherwise document why
  its observed-edge encoder is protocol-inadmissible.

### 6.3 Ablation arms (each mechanism must own one)

No-codebook (`z` removed); no-residual (`r_u`); no-imagination (slots off → B1);
**grounded vs ungrounded (headline arm — §1.9's automorphic honesty and §1.5's pool
question hang on it)**; grounding-only vs imagination-only slots; per-channel knockouts
(s1/s2/s3/s4) with the **s1–s4 correlation matrix** reported alongside [DA-M9];
CVAE-latent knockout (deterministic decoder) [R1-W9]; anchor-labeling (R7) on/off
[R1-W9]; `L_ssl` on/off; `L_real` component knockouts (statistic-space vs GIN-space vs
seam term separately); ∅-class vs multiplicity vs degree-NLL cardinality signals;
generator variant (set decoder vs discrete-diffusion, Approach B / E4.6); K sweep +
α-entmax learned-cardinality variant (DCM); **harmonization off (`R = 0`) vs rounds
sweep + mask-schedule sweep + slot-agreement trajectory**; hard-budget masking on/off;
binary vs bandwidth slot-adjacency supervision; conditioning-dropout rates for both
nulls (with both counterfactual contrasts reported); seam realism on/off;
retrieved-thresholded scaffold (E4.10); generation-only/no-edge-supervision (E4.11);
randomized-scaffold at matched capacity (E4.2).

**The credibility-critical experiment** (ZSL evaluation lesson, 1712.00981): all
generator variants compared under the *identical* fused decision head — retrieved-
thresholded vs generated vs generated+harmonized vs Oracle scaffold — **with an
explicit input-mapping convention** [R1-W9]: retrieved neighbors enter as `π=1, m=1`
slots, `Π` from identity matches, and the head is retrained per arm under a matched
tuning budget (otherwise the comparison measures channel availability, not generator
quality).

### 6.4 Evaluation hardening **[protocol-Δ]**

1. **Hard negatives:** HeaRT-style heuristic-related negatives (2306.10453) alongside
   random negatives; **re-verify the E2 gap under them (gate G1)**. For zero-edge test
   nodes, hard negatives are constructed **evaluator-side** [R2-W8]: heuristics
   computed on the ground-truth held-out graph (legitimate — the evaluator may see it;
   models never do), plus feature-similarity hard negatives that need no graph;
   the construction is specified in the protocol, not left to implementation.
   Degree-corrected negative sampling per the implicit degree bias result ⊕
   (2405.14985): evaluation negatives sampled with the positives' degree bias
   (`p(k) ∝ k·p(k)/⟨k⟩`), and gate G1 re-verifies the E2 gap under degree-corrected
   negatives explicitly, alongside HeaRT negatives; degree-corrected *training*
   negatives (their finding: reduces degree overfitting) fold into the hard-negative
   training arm [DA alt-4].
2. **MMD hygiene** (2201.09871, O'Bray 2106.01098 ⊕, 2512.14241): disclose per
   statistic the descriptor function, **bin count/binning rule**, kernel family, and
   every parameter (O'Bray's Table 1: three SOTA papers used three different kernels
   with different σ and bins — numbers are incomparable otherwise); avoid ad-hoc
   EMD-/TV-based "kernels" unless justified; parameter sweeps over the O'Bray ranges;
   never aggregate MMDs naively; MMD diagnostics must pass an
   **expressivity/robustness perturbation check** (metric
   increases monotonically under controlled perturbation of real graphs; bounded
   response to small perturbations) before they rank anything. Official BFS-macro GS/RD
   are reported separately; MMD components are always
   reported; bootstrap variance over buckets/seeds; the real-vs-real noise floor as
   the zero line. MMD is a *ported* two-sample statistic never validated as a GGM
   evaluator — one more reason the discriminator probe and held-out family stay
   headline.
3. **Trained-on vs held-out metric split [R1-W7]:** metrics used inside `L_real`
   (degree/clustering/code-histogram/motif energy distances, GIN-space distance) are
   reported as *trained-on*; the **held-out family is headlined**: orbit/motif counts
   beyond the trained set, component and path-length summaries, spectral MMD with an
   untouched kernel, and a learned-graph-feature distance from an encoder never used in
   training (restoring the blueprint §7 families the protocol dropped — flagged as an
   inconsistency to resolve **[protocol-Δ]**).
4. **Discriminator probe** (2512.14241 pattern), promoted per R3: accuracy of a
   held-out classifier distinguishing assembled from real subgraphs; near-chance is a
   bandwidth-independent realism claim.
5. **Stratified reporting:** edge metrics by degree (head/tail), by **Topological
   Concentration** (2310.04612 — added rev 2.2: TC correlates far better with LP
   performance than degree, and some node-centric metrics carry intrinsic
   degree-related bias, so degree strata alone mislead), and by community familiarity;
   harmonic-mean joint number; calibration (ECE/Brier on the §4.0 averaged
   probability).
6. **Diagnostics:** Cold Brew FCR per benchmark (pre-registers where imagination has
   headroom — and where s1/s2 are predicted to beat s3, §4.4); supervision-starvation
   statistic (2310.04314) as a motivation figure.
7. **Transfer probe (E6/E7 extension):** RDM-style retrieval-pool swap (2204.11824) —
   zero-shot cross-graph transfer no ladder baseline can run.
8. **Mechanism-transmission diagnostics [the §4.6 evidence]:** (a) imagined-ego-net
   fidelity on held-out *training* nodes — slot recall@K of true neighbors, `d̂`
   calibration against true degrees, slot-adjacency vs true local clustering [R1-W4];
   (b) assembled degree-calibration curve (E[d̂_u] vs realized assembled degree)
   [R1-W3]; (c) imagined-ego-net diversity vs real neighbor diversity [R3-W10];
   (d) mean grounding gate `g^k` trajectory [R1-W9]; (e) **TDS-style ego-net drift**
   [rev 2.2, 2310.04612's Topological Distribution Shift]: distributional distance
   between imagined ego-net statistics for training-time vs test-time nodes — detects
   the feature→ego-net map degrading under node-population shift.
9. **Ceiling reporting:** every assembled-metric table carries the G2 edge-independence
   ceiling row and the G3 Oracle row, so all gains are read against the reachable
   frontier, not against zero.

### 6.5 Integrity, parity, and decision rules **[protocol-Δ]**

- All five existing gates hold; gate 4 is strengthened (label-agnostic generation +
  endpoint removal + anchor labeling). New gates: **shared training-interaction identity
  plus per-query leave-one-out** for all structural losses [R1-W5-i]; **label-agnostic seam references**
  [R1-W5-ii]; **B0 provenance audit** (the frozen scorer must never have seen
  validation/test pairs) [R1-W5-iii].
- **HPO parity:** a pre-registered, equal tuning budget per baseline, recorded before
  final held-out metrics are opened (HeaRT's central complaint).
- **Pre-registered decision rules [DA-M12]:** (i) if `Ours` does not beat `B3-full`
  and `B0+cal` on the *held-out* assembled-metric family at matched edge AUPRC, the
  generative apparatus is declared not load-bearing and the paper pivots to the
  benchmark/`B3-full` story; (ii) if the topology-pathway knockout (`∅_content_head`
  vs `∅_all_head` attribution) and the closure diagnostics cost nothing on every
  benchmark and stratum, `Ours → B5` is declared; (iii) multiple-comparison control (Holm) over
  the assembled-metric family replaces "3 of 5 metrics" language; (iv) all rules and
  thresholds are frozen in the protocol before E1/E3 held-out metrics are opened.

### 6.6 Predicted headline — and its failure reading

Prediction: B0-level or better edge AUPRC (the jointly-trained trunk's `f_logit`
floor, reported via the four-logit decomposition against the matched `B0-e2e` arm),
with the **held-out**
assembled-metric family improved over B0, `B0+cal`, `B3-dist`, and `B3-full` by the
§4.6 mechanisms, within the G2 ceiling and read against the G3 Oracle headroom. E7
(downstream graph utility) is **promoted to load-bearing** [EIC, DA-M8]: the claim
that assembled realism matters is established there (community detection, label
propagation, retrieval probes on the assembled graph), because without it the paper
asks the community to value an intrinsic metric family on faith. Failure readings are
pre-registered in §6.5 — including the honest small-paper outcome ("calibrated
block-model marginals close most of the pair-to-topology gap") if the controls win.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Slot matching unstable early; hubs break Hungarian matching | warm-start with `L_recon`; dynamic + denoising queries; hub subsampling with multiplicity supervision (§4.2); assignment-flip-rate monitored; expect slow early convergence (DETR) |
| Loss stack unoptimizable (~15 sub-losses) | full loss tree with owners (§4.5); fixed pre-registered interior weights; gradient-norm monitoring; uncertainty-weighting fallback; staged curriculum (G5) |
| Harmonization train/inference mismatch | joint two-ego training task (§4.3b) — harmonization conditionals trained by construction; R12 |
| CVAE posterior collapse / mode-seeking imagination | free-bits + KL annealing; diversity diagnostic; SIG-VAE-style richer posterior as fallback; s1–s3 correlation watch |
| Generator becomes `L_edge`'s covert channel (realistic-looking but discriminatively-warped ego-nets) | held-out ego-net fidelity diagnostics (§6.4.8a) reported alongside headline metrics; `L_recon` weight floor; G5 staging isolates when it happens |
| Imagined neighbors leak the queried label | label-agnostic generation; queried endpoint explicitly removed from targets; `∅_content` counterfactual reported |
| s3 train-time quasi-oracle (the classified interaction is present in the shared topology) | explicit per-query leave-one-out targets and degree correction (R9); E5 gate |
| Community prior shortcut (context overrides pair) | gated fusion + counterfactual control; hard negatives |
| Content pathway carries the win / topology gates stay dead [rev 3.0] | pathway-attribution rule (§4.4) is a headline requirement; gate-magnitude + per-branch gradient telemetry; branch dropout; FCR-stratified pre-registered prediction; §6.5 decision rule (ii) |
| STE encodes a latent code disguised as a graph (π/m/Â/Π are feature-derived) [rev 3.0] | degree-preserving rewiring is the decisive control; edge-shuffle / DeepSets-reduction / matched-capacity-no-MP battery; degree-partialled representation probes |
| Gains explained by extra parameters or operating point | B1/B5/`B0+cal`/`B3-dist`/`B3-full` + randomized scaffold at matched capacity + identical-head comparison + density-matched thresholds |
| Assembled-graph realism capped by edge independence | G2 ceiling computed first; all tables read against it; stop condition defined |
| E2 numbers fragile (weak scorer, easy negatives, undefined composite, mixed normalization) | G1 rerun completed with hard negatives and official BFS-macro GS/RD; current claims route only to the 2026-07-14 formal artifacts, and "strong scorer" wording remains retired [DA-m16] |
| Budget prior breaks under benchmark density shift | `d̂` normalized per candidate-universe density (§4.1); E6 checks |
| Grounding collapse under `L_ssl` | pool-consistency applied to ungrounded slots only; mean `g^k` monitored |
| Stochastic inference irreproducible | §4.0 determinism policy (fixed samples, averaged `p_ij`, seeds) |
| Harmonization schedule sensitivity | mask-schedule sweep; `R = 0` fallback row |
| MMD metric attacks / Goodhart on trained metrics | §6.4 hygiene + trained-on/held-out split + discriminator probe + ceiling and Oracle rows |
| Codebook collapse / underuse | EMA + code-usage entropy (owned by `L_recon`); M sweep; continuous residual path |
| Grab-bag perception | §4.6 rule; single decoder lineage name; G5 staging means the submitted model is the pruned one |
| Reviewer community mismatch (KG-inductive expectations) | §5.2 settings taxonomy + GraIL/NBFNet/ULTRA positioning |

---

## 8. Key references (verified against local PDFs or the arXiv API; ⊕ = external-search addition; [venue-verify] = verify against version of record before print)

Subgraph/structural LP: SEAL 1802.09691; labeling trick 2010.16103; Distance Encoding
2009.00142 ⊕ (NeurIPS 2020, PDF-confirmed); P-GNN 1906.04817 ⊕ (ICML 2019,
PDF-confirmed; positional family, distinct from labeling tricks); BUDDY/ELPH
2209.15486; NCN/NCNC 2302.00890; Neo-GNN 2206.04216; HeaRT 2306.10453; LPFormer ⊕
2310.11009 (KDD 2024, ACM DOI confirmed); Topological Concentration ⊕ 2310.04612
(preprint — venue-check before print).
KG-inductive lineage: GraIL ⊕ 1911.06962 (ICML 2020; local PDF is the arXiv preprint —
check page-level quotes against PMLR); NBFNet ⊕ 2106.06935 (NeurIPS 2021,
PDF-confirmed); ULTRA ⊕ 2310.04562 (ICLR 2024, PDF-confirmed); IGMC ⊕ 1904.12058
(ICLR 2020; local PDF is arXiv v3 — check quotes against OpenReview).
Inductive/cold-start: DEAL 2007.08053; Graph2Gauss ⊕ 1707.03815 (ICLR 2018); Cold Brew
2111.04840; GLNN 2110.08727; DropoutNet (NeurIPS 2017); GAR (SIGIR 2022); Heater
[venue-verify]; Meta-Embedding ⊕ 1904.11547 (SIGIR 2019, PDF-confirmed — tag cleared);
GraphSAGE 1706.02216; UPNA ⊕ 2307.08877 (preprint, under review — no venue); New Node
Prediction ⊕ 2401.05468 (arXiv preprint, submitted to Information Sciences — no
venue); TGN ⊕ 2006.10637 (cite as arXiv); TGB ⊕ 2307.01026 (NeurIPS 2023 D&B,
PDF-confirmed).
Structure learning: LDS 1903.11960; IDGL 2006.13009; DGM 2002.04999; SUBLIME
2201.06367; NodeFormer 2306.08385; DCM 2305.16174; LGI-LS 2310.04314; SE-GSL
2303.09778; GSR 2211.06545; MoG 2405.14260; Manenti et al. 2405.19933; Pro-GNN
2005.10203; STABLE 2207.00012.
Generation: GraphRNN 1802.08773; GDSS 2202.02514; DiGress 2209.14734; EDGE 2305.04111;
Latent Graph Diffusion 2402.02518 [venue-verify the main-text task coverage claim];
SID/CID 2503.21592; HiGGs ⊕ 2306.11412 (cite as arXiv — main-track NeurIPS acceptance
unconfirmed); FLEX ⊕ 2507.11710 (preprint, under review — full-text verified rev 2.2);
NRI ⊕ 1802.04687 (ICML 2018, full-text verified).
Edge-independence bound: Chanpuriya et al. ⊕ 2111.00048 (NeurIPS 2021, full-text
verified rev 2.2 — exact identities and tightness theorems as quoted in §6.0-G2);
Rendsburg et al. "NetGAN without GAN" ⊕ [venue-verify].
Neighbor generation: LA-GNN 2109.03856; GraphSMOTE 2103.08826; Feature Propagation
2111.12128.
Community priors/codebooks: NOCD 1909.12201; Modularity-Aware GAE 2202.00961; TGSBM
2601.20646; VQGraph 2308.02117; GFT 2411.06070; SIG-VAE 1908.07078; HoscPool
2209.03473 (venue: CIKM 2022);
MinCutPool 1907.00481; DiffPool 1806.08804; DC-SBM: Karrer & Newman 2011 ⊕; latent
space models: Hoff, Raftery & Handcock 2002 ⊕; ERGM goodness-of-fit: Hunter, Goodreau
& Handcock 2008 ⊕ (classical, pre-arXiv — cite from JASA).
Masked autoencoding: VGAE 1611.07308; MGAE 2201.02534 + S2GAE (WSDM 2023 — same line,
cite the version of record); MaskGAE 2205.10053; GraphMAE ⊕ 2205.10803; SeeGera
2301.12458; Bandana 2402.03814; HGMAE 2208.09957.
Realism/evaluation: NetGAN 1803.00816; Graph Gestalt 2106.15239; Beyond-MMD
2512.14241; graphon AE 2105.14244; Thompson et al. 2201.09871; O'Bray et al. ⊕
2106.01098; implicit degree bias ⊕ 2405.14985.
CV mechanisms: Context Encoders 1604.07379; MAE 2111.06377; LaMa 2109.07161; RePaint
2201.09865; MaskGIT 2202.04200; pix2gestalt 2401.14398; diffusion classifier
2303.16203; JEM 1912.03263 (analysis lens only); low-shot hallucination 1801.05401;
f-CLSWGAN 1712.00981; f-VAEGAN 1903.10132; DETR 2005.12872; Slot Attention 2006.15055;
DSPN 1906.06565; PoinTr 2108.08839; VQ-VAE 1711.00937; VQGAN 2012.09841; RAC
2202.11233; semi-parametric synthesis (RDM) 2204.11824; IMP 1701.02426; Neural Motifs
1711.06640; TDE 2002.11949; DINOSAUR 2209.14860; Dreamer 1912.01603 (one motivating
sentence at most); DIAMOND 2405.12399.
Domain adaptation (mechanism sources): DANN 1505.07818; MCD 1712.02560; SHOT
2002.08546.
E2E conditioning & alignment (rev 3.0 additions; IDs verified against arXiv abstract
pages 2026-07-16, ⊕ external-search, [venue-verify] before print): Leap ⊕ 2503.03331
[venue-verify]; CAM tokens ⊕ 2405.19375 [venue-verify]; GOAT ⊕ 2111.05366
[venue-verify]; SLOTAlign ⊕ 2301.12721 [venue-verify]; GraphToken ⊕ 2402.05862
[venue-verify]; GraphGPT ⊕ 2310.13023 [venue-verify]; Flamingo ⊕ 2204.14198 (NeurIPS
2022 [venue-verify]); GNN-FiLM ⊕ 1906.12192 (ICML 2019 [venue-verify]); subgraph-VGAE
prediction ⊕ 2408.04053 [venue-verify]; Top-N set/graph generation ⊕ 2110.02096
[venue-verify]; NodeDup ⊕ 2402.09711 [venue-verify]; Modality Dropout ⊕ 2005.13616
[venue-verify]; modality competition Huang et al. ⊕ (ICML 2022, PMLR v162
[venue-verify]).
