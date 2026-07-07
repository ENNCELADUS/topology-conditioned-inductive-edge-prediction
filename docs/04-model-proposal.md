# Model Proposal: EgoStitch — Community-Conditioned Ego-Network Imagination for Topology-Conditioned Inductive Edge Prediction

**Status:** design proposal awaiting approval. Companion to `03-experiment-protocol.md` and
`02-methodology.md`. Grounded in a three-track literature review (graph/GNN, CV
generative mechanisms, local vault) run on 2026-07-07; all cited papers verified against
the arXiv API or local PDFs.

**One-line summary.** Replace the retrieved-and-thresholded scaffold with a *generated*
one: for each queried pair `(i, j)`, each endpoint **imagines its own ego-network** (a set
of latent neighbor nodes with existence probabilities, local adjacency, and a degree
budget) conditioned on its frozen features and a learned **community codebook**; the two
imagined ego-networks are **stitched** into the local scaffold `T_ij`, and the edge
decision fuses the node-intrinsic pairwise logit with membership, closure, and
community/capacity evidence computed on the stitched scaffold. The per-query local
boundary of the locked contract is preserved exactly.

---

## 1. Critique of the current scaffold contract

Current contract (`03-experiment-protocol.md` §0):

```text
V_T  = {i, j} ∪ ANN_feat(i, k) ∪ ANN_feat(j, k)
E_T  = {(a, b) ∈ V_T × V_T : score(a, b) ≥ τ_pair}, weighted by score(a, b)
T_ij = (V_T, E_T \ {(i, j)})
p_ij = σ(pair_logit(i, j) + scaffold_residual(H_i, H_j))
```

### 1.1 Circularity: the scaffold is B0's echo

Scaffold edges are thresholded outputs of the same frozen pairwise scorer whose assembled
output E2 proved structurally implausible (graph similarity 0.235, degree MMD 17.2,
relative density 0.684). `T_ij` is a local patch of exactly the pathological graph the
method is supposed to fix. Information-theoretically, `T_ij` is a deterministic function
of `(X, B0)`: the scaffold adds no information beyond re-encoding B0's biases as a graph.
Systematic B0 errors — hub over-prediction, similarity–adjacency conflation — are
inherited by the context and then "corrected" by a residual conditioned on those same
errors.

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
distribution of real neighborhoods. The vault's ICML 2025 result (Manenti et al.,
latent-graph uncertainty) proves the sharper version: point-prediction/reconstruction
losses alone cannot calibrate a latent graph distribution; a distributional objective
(MMD/energy distance on outputs) is required. Nothing in the current design makes `T_ij`
look like a real ego-network.

### 1.5 Test-pool dependence

ANN retrieval runs over whichever nodes co-occur in the held-out split. Predictions for
`(i, j)` therefore depend on the composition of the rest of the test pool: non-i.i.d.
behavior, inconsistency across query batches, and failure for isolated pairs with no good
candidates. A generative neighbor model answers from `(x_i, x_j)` alone; candidates
become optional grounding, not a hard dependency.

### 1.6 A local residual cannot repair distributional failures

E2's failure is distributional (relative density 0.684, degree MMD 17.2). Independent
per-query residuals have no channel for node-level degree budgets or community-level edge
budgets, so there is no *mechanism* by which the assembled degree distribution improves —
any gain would be accidental. The CV literature reached the same verdict on the analogous
problem (inpainting seams): no per-unit loss fixes assembly-level implausibility; you
need a whole-assembly objective or non-independent decoding (Context Encoders 1604.07379,
RePaint 2201.09865, MaskGIT 2202.04200).

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

### 1.9 Expressiveness: no target-relative labeling

The labeling-trick theory (Zhang et al., 2010.16103) proves that a GNN over a subgraph
cannot represent common-neighbor or other pair-relative structural evidence unless nodes
carry target-relative labels. The current contract encodes `T_ij` without any such
labeling, so even the structural signal that *is* present in the scaffold is partly
inexpressible to the encoder.

### What survives

The locked outer contract is right and is kept unchanged: per-query locality, frozen
features, masked queried edge, dual edge+graph evaluation, the baseline ladder. The
weakness is entirely in *how `T_ij` is constructed and what information it can carry*:
retrieved + thresholded ≠ generated + realistic.

---

## 2. Design requirements (derived from §1 and the literature review)

| # | Requirement | Source |
|---|---|---|
| R1 | Scaffold must be trained against **real ego-network topology**, not B0's outputs | §1.1, §1.4 |
| R2 | Neighbor proposals must be **generative**, with retrieval as optional grounding | §1.2, §1.5; GAR (SIGIR 2022), Cold Brew 2111.04840 |
| R3 | Scaffold construction must be **differentiable / learned end-to-end** | §1.3; DGM 2002.04999 |
| R4 | Reconstruct neighbor **features**, not raw adjacency bits, as the primary generative target | Graffe (vault), DINOSAUR 2209.14860 |
| R5 | Include **distributional realism loss** (energy distance/MMD), not only point losses | Manenti et al. (vault, ICML 2025); NetGAN 1803.00816; Graph Gestalt 2106.15239 |
| R6 | Carry explicit **degree budgets** and **community/block priors** into every decision | §1.6, §1.7; NOCD (vault); EDGE 2305.04111 |
| R7 | Apply the **labeling trick** on the scaffold | §1.9; 2010.16103 |
| R8 | Generation must be **label-agnostic** (no edge-hypothesis conditioning) to avoid self-fulfilling hallucination | CV warning: 2312.15540; SGG shortcut fix TDE 2002.11949 |
| R9 | Negative sampling **after** masking; queried endpoint removed from reconstruction targets | vault masked-edge report |
| R10 | Per-node computation should be **cacheable across queries** (amortization) | protocol §0 "amortize cleanly" |

---

## 3. Candidate architectures considered

### Approach A (recommended): EgoStitch — dual ego-network imagination + community codebook

Each endpoint generates its own ego-network (latent neighbor set + degree + local
adjacency) conditioned on frozen features and a quantized community code; the two
imagined ego-nets are stitched and the decision head fuses four evidence channels.
Detailed spec in §4.

- **Pros:** attacks all three E2 failure axes by mechanism (degree budget → density/degree
  MMD; slot–slot adjacency + closure channel → clustering; codebook/block prior →
  mesoscale spectrum); per-node generation is cacheable (R10); subsumes the current
  design as an ablation (generator off, grounding on); novelty combination verified
  unoccupied.
- **Cons:** set-generation training (Hungarian matching) adds engineering complexity;
  three auxiliary heads to balance.

### Approach B: per-query discrete graph diffusion scaffold

Condition a DiGress/EDGE-style discrete denoiser (2209.14734, 2305.04111) on
`{x_i, x_j} ∪ candidates` to denoise a local adjacency; optionally score the edge as a
diffusion classifier (ELBO gap between hypotheses, 2303.16203).

- **Pros:** strongest generative fidelity; principled likelihood-based decision.
- **Cons:** T denoising steps *per query* (not per node) — poor amortization; restricted
  to retrieved candidates, so §1.5 pool-dependence survives; adjacent to Latent Graph
  Diffusion (2402.02518), weakening the novelty claim. **Verdict:** keep as the
  generator-variant ablation arm (E4.6), not the main method.

### Approach C: neural-SBM/graphon residual (lightweight)

Infer NOCD-style community affiliations `F_u` and a degree correction from features
alone; `p_ij = σ(pair_logit + block_affinity(F_i, F_j) + degree_terms)`. No neighbor
generation.

- **Pros:** cheap; directly repairs density and degree distribution.
- **Cons:** no local closure evidence; no neighbor generation (misses the core
  innovation); close to B3/B4 baseline territory. **Verdict:** embed as EgoStitch's
  community/capacity channel (§4.4, channel `s3`) and additionally register it as a new
  baseline **B5** — it strengthens the ladder by isolating "block prior alone".

---

## 4. Recommended model: EgoStitch

### 4.0 Contract compliance

```text
inputs:  queried pair (i, j), frozen features X, optional ANN candidate pools G(i), G(j)
step 1:  community coding      z_u, F_u, d̂_u        = Tokenize(x_u)              (per node, cached)
step 2:  ego-net imagination   S_u = {(h_u^k, π_u^k)} = Imagine(x_u, z_u, G(u))    (per node, cached)
step 3:  stitch                T̂_ij = Stitch(S_i, S_j, {i, j})                    (per pair)
step 4:  decide                p_ij = σ(pair_logit(i,j) + g·Fuse(s1, s2, s3, s4))  (per pair)
```

The per-query local scaffold boundary is unchanged: `T̂_ij` is a local topological
context built from frozen features; the classifier is conditioned on `T̂_ij`; the output
is the binary edge label. Steps 1–2 are per-*node* and cached across all queries sharing
an endpoint (R10). No target-graph access anywhere.

### 4.1 Module 1 — Topology tokenizer (community + capacity channel)

- **Community code:** `e_u = MLP_enc(x_u)`; vector-quantize against a codebook
  `C = {c_1..c_M}` of neighborhood prototypes (VQ with EMA updates, straight-through;
  VQGraph 2308.02117 / GFT 2411.06070 pattern). `z_u = c_{m(u)}`.
- **Overlapping community affiliation:** `F_u = softplus(MLP_F([x_u; z_u])) ≥ 0`,
  trained with the NOCD Bernoulli–Poisson likelihood on the training graph:
  `P(A_uv = 1) = 1 − exp(−F_u · F_v)`, class-balanced negatives (vault: NOCD is
  inductive to unseen nodes — the key property we need).
- **Degree budget:** `d̂_u = softplus(MLP_deg([x_u; z_u]))`, trained with NLL against
  observed train-graph degrees.
- **Code supervision:** an auxiliary head predicts ego-net statistics (degree, local
  clustering, neighbor-code histogram) from `z_u`, so codes tile the space of
  *neighborhood shapes* (hub/leaf, dense/sparse, community identity), not feature space.

This module *is* the community-topology representation, fused downstream with the
node-intrinsic representation `x_u` — the fusion the current design lacks (§1.7).

### 4.2 Module 2 — Ego-network imagination (neighbor node generation)

A conditional set decoder generates `K` latent neighbor slots per node (DETR 2005.12872 /
Slot Attention 2006.15055 / point-cloud completion 2108.08839 template):

- **Slots:** `K` learned queries, initialized with `z_u`, iteratively cross-attend to
  `[x_u; z_u]` and (when available) to grounding candidates `G(u)` from feature-ANN.
  Output per slot: neighbor embedding `h_u^k` (in a projection of frozen-feature space),
  existence probability `π_u^k`, and a grounding gate `g^k ∈ [0,1]` with a pointer
  (attention) over `G(u)` — retrieval-augmented generation (2204.11824): slots *ground*
  in real candidates when good ones exist, otherwise imagine. Soft cardinality
  `Σ_k π_u^k` is tied to `d̂_u`. A per-ego-net stochastic latent (CVAE, LA-GNN
  2109.03856; VAE term counters mode collapse per f-VAEGAN 1903.10132) provides
  diversity.
- **Slot–slot adjacency:** logits `Â_u^{kk'}` over slot pairs model neighbor–neighbor
  edges (the source of local clustering).
- **Training — masked ego-net reconstruction** on the training graph: hide node `u`'s
  true neighbor set `N(u)`; generate slots; Hungarian/Chamfer matching of slots to
  `{proj(x_v) : v ∈ N(u)}` (feature-space targets per R4). Matched slots: feature
  regression + `π → 1`; unmatched slots: `π → 0`; degree NLL on `|N(u)|`; BCE on
  slot–slot adjacency against `A_{vv'}` for matched pairs. Negatives sampled after
  masking (R9). When the pair loss for `(u, v)` is active, `v` is removed from `u`'s
  reconstruction targets and generation never sees the edge label (R8: queried-edge
  standardization at the generative level, strictly stronger than edge masking alone).

### 4.3 Module 3 — Stitching

```text
V̂ = {i, j} ∪ S_i ∪ S_j          (≈ 2K + 2 nodes, K ≈ 10–20)
Π = SoftAlign(S_i, S_j)          (OT/attention alignment: which imagined neighbors coincide)
Ê  = { (u, k): π-weighted star edges } ∪ { slot–slot edges Â } ∪ { merged nodes via Π }
labels: target-relative labeling-trick features (endpoint-i / endpoint-j / slot-of-i /
        slot-of-j / grounded-node identity match)                                (R7)
```

The alignment matrix `Π` soft-merges slots of `i` with slots of `j` that describe the
same latent node — the stitched scaffold is where the two "dreams" must agree.

### 4.4 Module 4 — Decision head (fusion of evidence channels)

- `s0` **node-intrinsic:** frozen `pair_logit(i, j)` (B0-compatible anchor, unchanged).
- `s1` **membership:** likelihood of the partner under each imagined neighborhood —
  `logsumexp_k [κ(h_i^k, proj(x_j)) + log π_i^k]`, symmetrized. ("Is `j` one of the
  neighbors `i` dreamed?") This is analysis-by-synthesis scoring (diffusion-classifier
  pattern 2303.16203) at set level.
- `s2` **closure:** soft common-neighbor mass
  `CN̂_ij = Σ_{k,k'} Π_{kk'} π_i^k π_j^{k'}` plus an Adamic–Adar-weighted variant —
  the SEAL/NCN structural signal (1802.09691, 2302.00890) computed on *imagined*
  neighborhoods, which no observed-graph method can provide in this setting.
- `s3` **community/capacity:** block rate `1 − exp(−F_i·F_j)`, degree budgets
  `d̂_i, d̂_j`, and within-scaffold budget-pressure features (e.g., total imagined
  neighbor mass `Σ_k π^k` relative to `d̂`) — Approach C embedded. All features are
  computed inside the single query's scaffold; no state is carried across queries, so
  the per-query boundary is preserved.
- `s4` **scaffold readout:** 2–3 layer edge-weighted GNN over labeled `T̂_ij`,
  readout `MLP(H_i, H_j, H_T)`.

```text
p_ij = σ( s0 + g_θ(s1, s2, s3, s4) · w )
```

Gated residual fusion keeps B0 anchoring per the protocol and makes channel ablations
clean. Anti-shortcut control (TDE 2002.11949): report the counterfactual scaffold-only
score (context with endpoint features replaced by prototypes) as an E5 integrity control,
so community priors cannot silently override node-intrinsic evidence.

### 4.5 Training objective (maps 1:1 onto the locked objective)

```text
L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon
```

- `L_edge`: BCE on queried pairs (negatives after masking).
- `L_recon`: masked ego-net set reconstruction (Hungarian feature matching + degree NLL +
  slot-adjacency BCE + KL term for the CVAE ego-net latent) + NOCD Bernoulli–Poisson NLL
  + VQ commitment/codebook losses.
  *This is where topology realism is actually learned* — against real ego-networks, not
  B0 outputs (R1).
- `L_real`: **distributional** — energy distance/MMD between generated ego-net statistic
  vectors (degree, clustering, code histogram) and real ego-net statistics per batch
  (R5); optional small GNN discriminator on stitched scaffolds vs real ego-net unions
  (Context Encoders / NetGAN pattern).
- `L_ssl`: consistency of imagined ego-nets under feature noise and candidate-pool
  resampling — directly neutralizes §1.5 pool sensitivity.

### 4.6 Why each E2 failure axis is repaired by mechanism, not accident

| E2 failure | EgoStitch mechanism |
|---|---|
| Relative density 0.684 | per-node degree budgets `d̂` + block rates modulate every decision |
| Degree MMD 17.2 | soft cardinality tied to learned degree distribution; capacity features |
| Clustering MMD 11.8 | slot–slot adjacency + closure channel `s2` encode triangles explicitly |
| Spectral MMD 22.1 | community codebook + block prior shape mesoscale spectrum |
| Pair-to-topology gap | decision conditioned on a *realistic generated* context, trained with a distributional realism loss |

### 4.7 Complexity

Per node (cached): one tokenizer pass + one slot decode — `O(K·(d + |G(u)|))`.
Per pair: stitch + GNN over `≈ 2K+2` nodes — comparable to the current scaffold encoder.
Strictly better amortization than the current design (per-pair ANN + threshold) because
imagination is per-node.

---

## 5. Novelty positioning

Review-based claim (graph-track review, arXiv API + targeted novelty-risk queries; an
absence claim, so stated as "not found" rather than "does not exist"): **our review
found no model that, for an unseen node pair with no target graph, generates latent
neighbor nodes/ego-topology conditioned on the query, fuses a community-topology
representation with node-intrinsic features, and makes a binary edge decision under an
assembled-graph realism objective.** A final novelty pass (e.g., `/novelty-check`)
should be rerun before submission.

| Closest prior | What it does | Why it is not EgoStitch |
|---|---|---|
| LA-GNN (ICML 2022, 2109.03856) | CVAE generates neighbor *features* to augment an observed neighborhood | node classification; observed graph required; no query conditioning, no community prior, no realism objective |
| Cold Brew (ICLR 2022, 2111.04840) | hallucinates a virtual latent neighborhood for isolated nodes | produces one node embedding for node classification; no per-pair decision, no scaffold, no graph realism |
| Latent Graph Diffusion (NeurIPS 2024, 2402.02518) | one diffusion framework for graph generation and prediction | not per-query, not inductive-from-frozen-features, no community/intrinsic fusion |
| GAR (SIGIR 2022) | adversarially generates a cold item's warm embedding | single embedding, recsys ranking; no neighbor set, no topology |
| Modularity-Aware GAE (2202.00961) / TGSBM (2601.20646) | community/SBM prior inside an edge decoder | transductive, observed graph, no neighbor generation |

Mechanism imports (properly cited, none of which changes the novelty of the
combination): labeling trick (2010.16103), NCN-style closure evidence (2302.00890),
NOCD block prior (Shchur & Günnemann 2019, vault), degree-guided generation (EDGE
2305.04111), set decoding with matching losses (2005.12872, 2006.15055),
retrieval-grounded generation (2204.11824), distributional latent-structure calibration
(vault ICML 2025), whole-assembly realism critics (1604.07379, 1803.00816, 2106.15239).

---

## 6. Fit to the experiment protocol

- **`Ours` row:** EgoStitch drops into the existing E1/E3 table unchanged.
- **Current design becomes an ablation:** generator off + grounding on + thresholded
  scorer edges reproduces the §0 contract exactly — register as ablation arm
  `E4.10: retrieved-thresholded scaffold` (it is also the natural bridge baseline).
- **New baseline B5 (from Approach C):** neural-SBM residual — isolates "block prior
  alone", strengthening the mechanism story against reviewers.
- **New ablation arms:** no-codebook (z removed), no-imagination (slots off → B1),
  grounding-only vs imagination-only slots, per-channel knockouts (s1/s2/s3/s4),
  generator variant (slot decoder vs discrete-diffusion, Approach B), K sweep (matches
  E4.5), counterfactual scaffold-only control (E5 addition).
- **Integrity gates:** all five gates hold; gate 4 is strengthened (label-agnostic
  generation + endpoint removal from reconstruction targets, §4.2).
- **Predicted headline:** B0-level or better edge AUPRC (anchored `s0`), with graph
  similarity and all three MMDs improved by the §4.6 mechanisms — the E1 success
  criterion, now with a causal story per metric.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Slot matching unstable early in training | warm-start Modules 1–2 with `L_recon` only; freeze B0 anchor throughout |
| Imagined neighbors leak the queried label (self-fulfilling prior) | generation never conditioned on edge hypothesis; endpoint removed from targets; TDE counterfactual control reported |
| Community prior shortcut (context overrides pair) | gated fusion + scaffold-only integrity control; hard negatives from E5 |
| Mode collapse of generated neighborhoods | CVAE latent + distributional `L_real` (energy distance), f-VAEGAN lesson |
| Gains explained by extra parameters, not topology | B1/B5 baselines + randomized-scaffold ablation (E4.2) at matched capacity |
| Topology-reconstruction objective degrades representations (Graffe warning) | feature-space targets primary; adjacency losses secondary and down-weighted |
| Codebook collapse | EMA updates + code-usage entropy regularizer; M sweep in ablations |

---

## 8. Key references (verified)

Subgraph/structural LP: SEAL 1802.09691; labeling trick 2010.16103; BUDDY 2209.15486;
NCNC 2302.00890; Neo-GNN 2206.04216; HeaRT 2306.10453.
Inductive/cold-start: DEAL 2007.08053; Cold Brew 2111.04840; GLNN 2110.08727;
DropoutNet (NeurIPS 2017); GAR (SIGIR 2022); GraphSAGE 1706.02216.
Structure learning: LDS 1903.11960; IDGL 2006.13009; DGM 2002.04999; SUBLIME 2201.06367;
NodeFormer 2306.08385; latent-graph uncertainty (ICML 2025, vault).
Generation: GraphRNN 1802.08773; GDSS 2202.02514; DiGress 2209.14734; EDGE 2305.04111;
Latent Graph Diffusion 2402.02518.
Neighbor generation: LA-GNN 2109.03856; GraphSMOTE 2103.08826; Feature Propagation
2111.12128.
Community priors/codebooks: NOCD (vault); Modularity-Aware GAE 2202.00961; TGSBM
2601.20646; VQGraph 2308.02117; GFT 2411.06070; SIG-VAE 1908.07078.
Realism: VGAE 1611.07308; NetGAN 1803.00816; Graph Gestalt 2106.15239; Beyond-MMD
2512.14241; graphon AE 2105.14244.
CV mechanisms: Context Encoders 1604.07379; MAE 2111.06377; LaMa 2109.07161; RePaint
2201.09865; MaskGIT 2202.04200; pix2gestalt 2401.14398; diffusion classifier
2303.16203; JEM 1912.03263; low-shot hallucination 1801.05401; f-CLSWGAN 1712.00981;
f-VAEGAN 1903.10132; DETR 2005.12872; Slot Attention 2006.15055; DSPN 1906.06565;
PoinTr 2108.08839; VQ-VAE 1711.00937; VQGAN 2012.09841; RAC 2202.11233;
semi-parametric synthesis 2204.11824; IMP 1701.02426; Neural Motifs 1711.06640;
TDE 2002.11949; DINOSAUR 2209.14860; Dreamer 1912.01603; DIAMOND 2405.12399.
