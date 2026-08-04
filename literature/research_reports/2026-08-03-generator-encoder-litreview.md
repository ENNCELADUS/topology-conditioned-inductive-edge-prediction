# Literature Review — Generator and Graph Encoder Components

**Date:** 2026-08-03 (restructured per owner directive: no architecture
contracts; ranked suggestions).
**Task framing (binding, `docs/lit-review-plan.md`):** the final task is binary
edge prediction for a queried pair `(u, v)` of unseen nodes, with no access to
the target graph at inference. Any generated local topology is an intermediate
scaffold that conditions the edge decision — never the final output. The
training graph is *incomplete* (an unknown fraction of true edges is missing),
so negatives are PU-contaminated and structural targets are biased low.
Evaluation is dual and inseparable: edge-level metrics plus the topology of the
graph assembled from predicted edges.

**Method.** Five parallel bibliography sweeps (generator; encoder;
incomplete-supervision; discrete-output training routes; conditional discrete
diffusion), each excluding the ~150 citations already verified in
`docs/04-model-proposal.md` §8. Every reported arXiv ID was independently
re-verified against the arXiv API on 2026-08-03 (ID ↔ title match);
unverifiable candidates were dropped. 83 new papers survived. Convergent
finds across sweeps (DeepNC, PULL) are reported once.

---

## 1. Generator — ranked suggestions

Ranking criteria: (a) expected gain on the dual objective (edge metrics +
assembled-graph realism), (b) robustness to the incomplete training graph,
(c) strength of published precedent, (d) engineering risk — **weighted, per
the owner's bold-design directive, with conditional steerability by the
downstream edge loss counted into (a)**. Stated plainly: on raw published
topology-statistics evidence alone, G2 (GruM) leads; G1 outranks it because
it is the only family with published mechanisms for per-query conditioning
and downstream reward steering, which the task needs and GruM lacks. The
noise toolkit in §3 composes with every rank.

*Domain-transfer flags used below:* [text], [image], [bio] mark evidence
obtained outside graphs; transfer to graph generation is **unproven** unless
a graph-native citation accompanies it.

### Rank G1 — Conditional discrete flow/diffusion generator, reward-steered
**The bold default.** Pretrain a discrete flow-matching or diffusion model of
*local* subgraphs conditioned on endpoint features + retrieved candidates,
sample a (binary or soft-marginal) scaffold per query, and steer it with the
downstream edge loss.

- Backbone objective: **DeFoG** (`2410.04263`, ICML 2025) — discrete flow
  matching; train once, tune sampling later; near-full quality at 5–10% of
  diffusion steps, which is what makes per-query sampling affordable at
  millions of queries. Continuous-time alternatives: **DisCo** (`2405.11416`,
  NeurIPS 2024), **Cometh** (`2406.06449`) — Cometh also shows node/edge
  denoising schedules should be decoupled (decide node existence first, then
  edges).
- Downstream steering, three published mechanisms, by increasing cost:
  (i) inference-time derivative-free value guidance, no retraining
  (`2408.08252`) [text/bio]; (ii) gradient-based reward fine-tuning specific
  to discrete diffusion (`2410.13643`, ICLR 2025) [bio]; (iii)
  policy-gradient on graph diffusion — **GDPO** (`2402.16302`, NeurIPS 2024),
  the *only graph-native* steering precedent: a discrete graph-diffusion
  sampler steered by an arbitrary reward (here: edge-loss improvement +
  realism terms).
- If gradients through the scaffold are wanted instead of RL: truncate
  sampling and hand the encoder the expected rate-matrix marginals (soft), or
  keep hard samples with a Route-A estimator (§1.7).
- Conditioning mechanics (discrete-guidance sweep): D3PM (`2107.03006`,
  NeurIPS 2021) is the foundational 2-state (edge/no-edge) contract. Exact
  guided CTMC rate matrices give classifier-based/free guidance *without
  retraining the denoiser* (`2406.01572`, ICLR 2025); a lightweight CFG/CBG
  derivation plus the key design finding that **uniform-state (editable)
  diffusion guides better than mask-absorbing** — absorbing chains lock early
  edge mistakes (`2412.10193`, ICLR 2025) [text/image; the edge-mistake
  reading is our transfer]. FreeGress (`2312.17397`, ECML
  2024) is the closest template for baking an external condition vector into
  a graph denoiser (CFG beats classifier guidance there); GGDiff
  (`2505.19685`) unifies guidance under stochastic control, including
  *non-differentiable* rewards — e.g. "does the frozen classifier's
  confidence improve on this scaffold".
- Step-cost reduction (distillation sweep — **all non-graph evidence**):
  self-distillation through time cuts discrete-diffusion sampling 32–64× at
  equal quality (`2410.21035`, ICLR 2025) [text]; **di4c** (`2410.08709`,
  ICML 2025) [text/image] preserves cross-dimension correlations during
  distillation — *we infer* this matters most for graphs, where naive
  parallel few-step edge sampling should destroy triangle/degree coherence,
  but di4c never tests graphs; the Diffusion Duality ports consistency-style
  few-step samplers into exactly the uniform-state regime that guides best
  (`2506.10892`, ICML 2025) [text]; the extreme end is one-step distillation
  (Di[M]O `2503.15457`, ICCV 2025 [image]; DMD `2311.18828`, CVPR 2024
  [image] as the mechanism anchor).
- Honest cost accounting (reconciling with §4 Gap 5): even distilled to one
  step, a per-query scaffold costs a full denoiser forward over the scaffold
  plus an encoder forward — versus an MLP on two embeddings for pairwise
  scoring; reward steering additionally requires sampling scaffolds
  throughout *training*, and one-step students only exist after paying full
  teacher training. Distillation makes per-query diffusion *plausible*, not
  free; no published work amortizes denoiser computation across related
  queries (Gap 5). A FLOP-level train/inference cost table against the
  pairwise baseline is a prerequisite for adopting this rank.
- Compatibility constraint (the three ingredients do not compose freely):
  trajectory-based guidance (`2406.01572`, `2408.08252`, GGDiff) needs an
  iterative sampler and so is incompatible with a one-step student; the
  uniform-state editability advantage (`2412.10193`) conflicts with Di[M]O's
  *masked*-diffusion distillation route. Jointly realizable configurations:
  (i) uniform-state + CTMC guidance + moderate step cut (SDTT/di4c-style);
  (ii) conditioning baked into the denoiser (FreeGress-style CFG at training
  time) + aggressive one-step distillation, forgoing inference-time
  guidance; (iii) full-step sampler + reward fine-tuning (GDPO), highest
  cost. Pick per compute budget; do not assume all three properties at once.
- Closest existing system: **RADD** (`2604.25693`, preprint 2026,
  **unreviewed**, single-entity KG completion — adjacent setting, not ours) —
  a retrieval-augmented *conditional discrete denoiser over a retrieved
  candidate pool used for link prediction*, with the retriever distilling
  into the denoiser. Prime comparison target; the graph-native steering
  precedent otherwise rests on GDPO + SaGess.
- Novelty demarcation vs latent-space diffusion: Laplacian-AE latent graph
  diffusion (`2601.13780`, preprint 2026), GLAD's quantized graph latents
  (`2403.16883`, AAAI 2025), and feature-conditioned latent generation (NGG
  `2403.01535`) are the post-LGD latent line — this rank deliberately
  diffuses in the raw binary adjacency state space per query (no lossy
  autoencoder), conditioned on *two specific endpoints*, not a global
  property vector.
- Why rank 1: strongest generative fidelity family; all three objections that
  previously shelved per-query diffusion now have direct published answers —
  step cost (distillation, above), training-through-samples (reward steering,
  above), and novelty adjacency to Latent Graph Diffusion (the demarcation is
  now sharp, above); SaGess (`2306.16827`) is evidence that a distribution
  over small local subgraphs captures large-graph structure.
- Risk: conditioning on *two endpoint feature vectors + retrieved pool* is
  still unoccupied (RADD is single-entity, KG-specific; FreeGress/NGG are
  global-property) — novel, hence unproven.
- Optional generative-scorer branch: the generator's own likelihoods can
  score the edge — ELBO ratio p(scaffold | edge=1) vs p(scaffold | edge=0)
  (RDC `2305.15241`, ICML 2024 [image]) — and generative classifiers resist
  shortcut features under distribution shift (`2512.25034`, ICLR 2025
  [image]); *treating inductive unseen-node prediction as such a shift, and
  hence expecting the same benefit, is this review's conjecture*, not a
  published graph result. Competes with, not replaces, a separate
  classifier: likelihood estimation needs multiple denoiser calls per query.

### Rank G2 — Endpoint-predicting diffusion mixture (topology-first objective)
**GruM** (`2302.03596`, ICML 2024): parameterize generation as
endpoint-conditioned bridges that predict the *final* graph at every step
rather than a denoising residual. Best published results precisely on the
statistics the assembled-graph arm penalizes (degree/clustering/orbit). The
endpoint prediction is a soft adjacency at every step — usable directly as
conditioning without discrete sampling, or sampled at the end. Lower risk than
G1 (single coherent objective, no RL loop), slightly weaker fidelity ceiling.

### Rank G3 — Seed-rooted local expansion (grow the neighborhood around the query)
**Iterative Local Expansion** (`2312.11529`, ICLR 2024): generate by expanding
a single node through inverted spectrum-preserving coarsenings, each step a
local denoising — the closest published mechanism to "grow an ego-network
around a queried endpoint, stop at radius 2". **Pard** (`2402.03687`, NeurIPS
2024) supplies the permutation-invariant block-autoregressive machinery
(condition on partial graph, diffuse the next block: seed pair → 1-hop
candidates → closure edges). Most natural inductive-bias fit for local
scaffolds; more sequential machinery to engineer than G1/G2.

### Rank G4 — GFlowNet edge-wise scaffold builder (diversity + uncertainty)
**DAG-GFlowNet** (`2202.13903`, UAI 2022) is nearly the template: sample graph
structures edge-by-edge in proportion to a reward, with calibrated edge-level
uncertainty; swap the Bayesian score for edge-loss improvement + realism.
Foundations `2111.09266`; trajectory balance `2201.13259` for convergence.
Unique capability no other rank offers: *diverse* scaffold ensembles per query
with principled uncertainty — attractive for the noisy-graph setting.
Higher training risk (credit assignment, reward shaping) and least
graph-generation precedent at scale.

### Rank G5 — One-shot set-style decoder with spectral conditioning
The cheapest family: decode all edges in one shot, but condition the decoder
on a small set of generated spectral coordinates — **SPECTRE** (`2204.01613`,
ICML 2022) — so triangles/communities are coordinated rather than
pairwise-independent (the canonical remedy to the Chanpuriya `2111.00048`
edge-independence ceiling). **SwinGNN** (`2307.01646`, TMLR 2024) licenses a
canonical node ordering (theory: permutation-invariant score targets are
*harder* to learn), and **IFH** (`2408.13194`, ECAI 2024) contributes a
learned halt head so node count is modeled rather than padded — which matters
for degree-distribution realism. Lowest cost and risk; lowest fidelity
ceiling.

### Rank G6 — Autoregressive sequence generator with goal-oriented fine-tuning
**G2PT** (`2501.01073`, ICML 2025): next-token graph transformer with
published recipes for goal-oriented fine-tuning and representation
extraction; exact likelihoods on small graphs enable likelihood-based
rejection of degenerate scaffolds. Ranked last mainly for ordering
sensitivity and weaker topology-statistics evidence than G1–G2.

### 1.7 Training-signal toolbox for discrete outputs (composes with G1–G6)
If the scaffold is binary, choose the estimator by bias/variance budget:
- Cheap default: Gumbel-Softmax/ST (`1611.01144`; Concrete `1611.00712`),
  variance-reduced by Rao-Blackwellization (`2010.04838`).
- Bounded scaffolds ("exactly k neighbors"): **SIMPLE** (`2210.01941`,
  ICLR 2023) dominates SoftSub (`1901.10517`) on bias-variance.
- Structured samplers (trees/matchings/subgraphs): Stochastic Softmax Tricks
  (`2006.08063`).
- Through a black-box combinatorial step: **I-MLE** (`2106.01798`,
  NeurIPS 2021).
- Unbiased fallback: RELAX (`1711.00123`; REBAR `1703.07370` lineage).
- Latent-variable alternative (no relaxation at all): Reweighted Wake-Sleep
  (`1406.2751`) with VIMCO (`1602.06725`) — scaffold as discrete latent,
  classifier as likelihood.
- Reward fine-tuning for differentiable rewards: DRaFT (`2309.17400`,
  ICLR 2024) — backprop through the sampling chain, truncated.

No paper benchmarks these families head-to-head on graph-structure
objectives — that comparison is itself a claimable ablation.

---

## 2. Graph encoder — ranked suggestions

The encoder's representation is good exactly insofar as it improves
`edge(u, v)` and the assembled graph's realism. One upstream decision gates
the field: **does the generator emit a soft or a binary scaffold?** Soft
scaffolds require every structural computation to be continuous in the
adjacency; binary scaffolds re-admit the discrete toolbox (shortest-path
biases, subgraph extraction, exact motif counts, edge-list tokenization).
Ranks below note the regime.

### Rank E1 — GRIT: RRWP-based graph transformer (both regimes)
`2305.17589` (ICML 2023). Relative random-walk probabilities as initial pair
representations, updated by attention that mixes node and pair channels.
Regime-agnostic (matrix powers work on weighted or binary adjacency), dense by
construction, cheap on small scaffolds, and the pair channel is intrinsically
link-aware — aligned with a task whose output is a pair decision. Provably
generalizes shortest-path and RWSE encodings. Typed relations enter as extra
pair channels. **Conditionality of this rank:** `2407.11764` shows
structure-consuming transformers can be catastrophically fragile to structure
perturbation, and a generated scaffold is perturbed by construction — E1
above E4 presumes the generator reaches reasonable scaffold fidelity. Gate
the E1-vs-E4 choice on a measured scaffold-quality checkpoint; if scaffold
noise dominates the error analysis, E4 outranks E1 under criterion (b).

### Rank E2 — Edge-channel transformers: EGT / Relational Attention (both; best for typed relations)
EGT `2108.03348` (KDD 2022) keeps an `(N,N,d)` edge channel that biases *and
gates* attention and is updated each layer — reading the gate as a learned
edge-reliability estimate that reweighs unreliable generated edges is *this
review's hypothesis to test*: EGT was designed and evaluated on clean
benchmark graphs, never on noisy generated inputs. Relational Attention `2210.05062` (ICLR 2023) is the fully-dense
variant with directed per-pair relation vectors and principled node↔edge
co-updates. Choose E2 over E1 when relation typing and explicit edge-state
readout matter more than random-walk structure.

### Rank E3 — Graphormer-style SPD-bias transformer (binary regime only)
`2106.05234` (NeurIPS 2021). With a binary scaffold, exact shortest-path
biases, integer degrees, and the virtual readout node work as designed, and
all-pairs BFS on a small scaffold is free. The strongest classic template in
the binary regime; in the soft regime its discrete parts need the continuous
relaxations published in `2407.11764`. SAT (`2202.03036`, ICML 2022) and
TokenGT/ESA (`2207.02505` / `2402.10793`) similarly re-enter in the binary
regime (subgraph extraction and edge-list tokenization become legal and
cheap).

### Rank E4 — DIFFormer: energy-derived attention (both; best under scaffold distrust)
`2301.09474` (ICLR 2023, spotlight). Attention derived as the descent step of
an energy where the input graph is one term — the scaffold acts as a *prior*
the encoder can override when features disagree; explicit experiments with
partial/missing/wrong structure. The principled choice if scaffold noise
turns out to dominate error analysis. SGFormer (`2306.10759`, NeurIPS 2023)
is the same line's minimalism warning: one attention layer + a local branch
often suffices.

### Rank E5 — GraphGPS as harness (both; not a bet, a platform)
`2205.12454` (NeurIPS 2022). Parallel local-MPNN + global-attention layer with
plug-in positional/structural encodings — the right *experimental platform*
for ablating E1–E4 choices under one roof rather than a distinct bet.

### Mandatory baseline (any regime)
Typed-edge MPNN + virtual node, pooled state as conditioning: `2301.11956`
(ICML 2023) proves virtual-node MPNNs approximate self-attention. If a graph
transformer cannot beat this, encoder capacity is not the binding constraint.
NRI's decoder (`1802.04687`) is the canonical minimal per-relation-type
message-passing encoder over a soft typed adjacency.

### Positional/structural encoding guidance (applies across ranks)
- Default: RWSE/RRWP random-walk features — continuous in the adjacency, no
  stability pathologies.
- Spectral, if used: **SPE** (`2310.02579`, ICLR 2024) only — soft eigenbasis
  aggregation, provably Lipschitz under edge perturbation (doubles as a
  robustness guarantee against scaffold noise); SignNet (`2202.13013`) is the
  sign-invariance predecessor; Specformer (`2303.01028`) upgrades this to a
  *learned* spectral filter that can denoise the scaffold in-encoder —
  directly serving the spectral half of assembled-graph realism.
- For link tasks specifically: **PEG** (`2203.00199`, ICLR 2022) — inject PEs
  only via pairwise PE-distances (symmetric in `(u,v)`, perturbation-stable);
  the canonical "PEs for edges done right".
- Robustness caveat: `2407.11764` shows graph transformers can be
  catastrophically fragile to structure perturbation — PE choice is the main
  lever; its relaxation toolbox is also the reference for any
  differentiable-in-adjacency structural computation.

### Readout for conditioning
**GMT** (`2102.11533`, ICLR 2021): attention pooling onto k learnable seeds →
a multi-token graph summary, strictly richer than one pooled vector, with
reconstruction/generation evidence that attention pooling preserves topology
statistics. Cheap alternative: virtual-node state. RelGT (`2505.10960`,
preprint 2025) offers learnable-centroid summaries and is the 2025 reference
for typed-graph transformers generally.

---

## 3. Noise toolkit — training and evaluating under an incomplete graph
(composes with every generator/encoder rank)

1. **PU-correct the edge loss.** PULL (`2405.11911`, AAAI 2025) — link
   prediction as PU learning, iterative latent-edge expectation; PUDA
   (`2205.00904`, IJCAI 2022) — unbiased PU risk with a class prior
   (= edge-observation rate, sweep it); prior-estimation caveat under
   structure: `2405.19919` (ICML 2024).
2. **Corruption-matched generative training.** Ambient Diffusion
   (`2305.19256`, NeurIPS 2023): further-mask observed edges and train to
   reconstruct the observed (not further-masked) neighborhood. Its
   clean-distribution guarantee is proven for the paper's image/linear
   corruption operators; *the transfer to edge-missingness is conjectured
   here, and establishing it is precisely Gap 2 in §4* — do not cite the
   guarantee as settled for graphs. Follow-ups `2510.12691`, `2407.01014`,
   `2404.10177`. Cheap version: ANFM's prefix corruption (`2502.02415`).
   Framing: DeepNC (`1907.07381`, TPAMI) — the generated neighborhood must
   explain observed edges *as a subset*, not match them exactly.
3. **Statistically correct targets.** Young/Cantwell/Newman (`2008.03334`) —
   posterior over the true adjacency under an explicit observation model
   (soft structural targets); Zhang/Kolaczyk/Spencer (`1305.4977`) — degree
   distribution under sampling as a linear inverse problem: prefer
   *forward-applying* the sampling operator to generated graphs over
   debiasing the reference.
4. **Robust supervised training mechanics.** RTGNN (`2211.06614`, WSDM 2023)
   co-teaching for benign noise; consult NoisyGL (`2406.04299`, NeurIPS 2024
   D&B) before adopting any robustness component (many don't replicate);
   missingness-symmetric repair (impute, don't only delete): `2606.03462`.
5. **Evaluation under an incomplete reference.** Non-uniform missingness
   shifts LP accuracy *and method rankings*: `2401.15140` (PLoS ONE 2024) —
   stress-test under 2–3 non-uniform patterns; the probe/validation split is
   itself a biased sample: `2606.19775`; bounded classifier-based
   per-descriptor discrepancy: PolyGraph (`2510.06122`, ICLR 2026);
   perturbation-stable descriptors with provable bounds: Curvature
   Filtrations (`2301.12906`, NeurIPS 2023).
6. **The null hypothesis.** `2411.07672`: learned structure often adds no
   mutual information beyond the representations that built it — the
   generated scaffold must demonstrably add information beyond pairwise
   features, or the topology-conditioning claim collapses. Their MI-gain
   probe is a ready-made ablation. Inductive-transfer precedent for structure
   inference on unseen graphs: GraphGLOW (`2306.11264`, KDD 2023); IB-guided
   structure-for-LP under noise: CORE (`2404.11032`).

---

## 4. Gaps the paper can claim (sweeps converge)

1. **Feature-conditioned ego-network generation is unoccupied** — no found
   paper generates a local neighborhood from a seed node's features plus a
   retrieved candidate pool. (SGDiff `2409.08487` is the nearest neighbor and
   the key novelty-demarcation citation: it uses the enclosing-subgraph
   generative model *as the scorer*, not as context for a classifier.)
2. **Graph generative training under incomplete adjacency is unpublished** —
   PU work is discriminative; completion is pre-diffusion; corrupted-data
   theory is image-domain. Porting ambient-style training to a topology
   generator is claimable.
3. **Realism metrics with an incomplete reference are unstudied** — nothing
   answers how descriptor-MMD shifts when the reference graph misses x% of
   edges; triangulate `2401.15140` + `2301.12906` + `2510.06122`.
4. **No head-to-head benchmark of discrete-gradient families (ST vs I-MLE vs
   GFlowNet vs reward-FT) on graph-structure objectives.**
5. **No cross-query amortization of conditional diffusion** — published
   caching is within one sampling trajectory (DeepCache `2312.00858`), never
   across distinct conditions/queries; for a per-query scaffold generator at
   millions of queries, denoiser-feature sharing across related queries is an
   open contribution.

Protocol-adjacent suggestions (owner decisions, flagged only): adding
PolyGraph per-descriptor scores alongside the MMD ratios; non-uniform
missingness stress tests; the MI-gain ablation.

---

## Appendix A. Verified bibliography by sweep

Every ID re-verified against the arXiv API 2026-08-03. One-line takeaways;
full annotations live in the sweep transcripts.

**A.1 Generator sweep.** DisCo `2405.11416` (CT discrete diffusion,
quality/steps dial); Cometh `2406.06449` (RW encodings suffice; decoupled
node/edge schedules); DeFoG `2410.04263` (discrete flow matching, 5–10%
steps); Pard `2402.03687` (perm-invariant AR-diffusion blocks); GruM
`2302.03596` (endpoint-prediction bridges, best topology MMD); ANFM
`2502.02415` (filtration AR, prefix corruption, edge deletion); G2PT
`2501.01073` (sequence AR + goal-oriented FT); HDFM `2604.00236`
(coarse-to-fine flow); IFH `2408.13194` (sequentiality dial, halt head);
SPECTRE `2204.01613` (spectral conditioning vs edge-independence); SwinGNN
`2307.01646` (non-invariant generation license); ILE `2312.11529`
(seed-rooted expansion); SGDiff `2409.08487` (enclosing-subgraph likelihood
as LP scorer); GraphMaker `2310.13833` (async attribute→structure); SaGess
`2306.16827` (small-subgraph cover training); DeepNC `1907.07381`
(generative completion of partial graphs); Ambient Diffusion `2305.19256`
(clean distributions from corrupted data).
Rejected: SeaDAG `2410.16119`, FLAGG `2606.05067`, SparseDiff `2311.02142`,
LayerDAG `2411.02322`, ARROW-Diff `2408.04461`, `2307.08849` (superseded by
Pard), `2202.10824` (scene graphs), HOG-Diff `2502.04308`, `2511.03015`,
`2512.01190`, `2506.16404`, `2405.13094`.

**A.2 Encoder sweep.** Graphormer `2106.05234`; SAN `2106.03893`; GraphGPS
`2205.12454`; TokenGT `2207.02505`; GRIT `2305.17589`; EGT `2108.03348`;
Relational Attention `2210.05062`; RelGT `2505.10960`; SignNet `2202.13013`;
SPE `2310.02579`; PEG `2203.00199`; Specformer `2303.01028`; DIFFormer
`2301.09474`; SGFormer `2306.10759`; GT adversarial robustness `2407.11764`;
GMT `2102.11533`; MPNN≈GT via virtual node `2301.11956`; NRI `1802.04687`
(already in §8; new angle: decoder as soft-typed-adjacency encoder).
Rejected: Exphormer `2303.06147`, Polynormer `2403.01232`, GraphViT
`2212.13350`, SAT `2202.03036` (re-admitted in binary regime, §2 Rank E3),
LSPE `2110.07875`, MoSE `2410.18676`, GARNET `2201.12741`, ESA `2402.10793`
(re-admitted in binary regime), LRGB reassessment `2309.00367`.

**A.3 Incomplete-supervision sweep.** PULL `2405.11911`; PUDA `2205.00904`;
GPL heterophily-PU `2405.19919`; RTGNN `2211.06614`; NoisyGL `2406.04299`;
Gaussian graph repair `2606.03462`; DeepNC `1907.07381`; Young et al.
`2008.03334`; Zhang et al. `1305.4977`; non-uniform missingness `2401.15140`;
sampling strategy in LP `2606.19775`; PolyGraph Discrepancy `2510.06122`;
Curvature Filtrations `2301.12906`; Rethinking GSL `2411.07672`; GraphGLOW
`2306.11264`; CORE `2404.11032`.
Rejected: `2306.07512`, ProGCL `2110.02027`, NML-GCL `2505.10307`,
`1411.6081`, `2010.01916`, GraphCleaner `2306.00015`, Newman `1703.07376`,
Peixoto `1806.07956`, `2402.08893`, `2412.06173`, UGSL `2308.10737`,
`2503.21223`, `1909.07578`, DeMix `2310.09781`, RS-GNN `2201.00232`, NRGNN
`2106.04714`, `2408.00700`. (Kossinets 2006, DOI
10.1016/j.socnet.2005.07.002, standard missing-data sensitivity citation —
not arXiv-verified.)

**A.4 Discrete-output training sweep.** Gumbel-Softmax `1611.01144` (Concrete
`1611.00712` inline); RB-ST-GS `2010.04838`; SoftSub `1901.10517`; SIMPLE
`2210.01941`; Stochastic Softmax Tricks `2006.08063`; I-MLE `2106.01798`;
RELAX `1711.00123` (REBAR `1703.07370` lineage); GFlowNet Foundations
`2111.09266`; Trajectory Balance `2201.13259`; DAG-GFlowNet `2202.13903`;
DDPO `2305.13301` (DPOK `2305.16381` adjacent); DRaFT `2309.17400`;
discrete-diffusion reward optimization `2410.13643`; GDPO `2402.16302`; RWS
`1406.2751`; VIMCO `1602.06725`; soft-value derivative-free guidance
`2408.08252`. Rejected: `2110.01515` (survey), `2305.17010` (CO rewards),
`2106.04399` (subsumed).

**A.5 Conditional discrete diffusion sweep.** D3PM `2107.03006` (discrete
state-space foundations); CTMC exact guidance `2406.01572`; simple discrete
CFG/CBG + uniform-state editability `2412.10193`; FreeGress `2312.17397`
(CFG conditional DiGress); GGDiff `2505.19685` (stochastic-control guidance,
non-differentiable rewards); SDTT `2410.21035` (32–64× step cut); Di[M]O
`2503.15457` (one-step masked-diffusion distillation); Duo `2506.10892`
(consistency few-step for uniform-state); di4c `2410.08709`
(correlation-preserving distillation); Laplacian-AE latent graph diffusion
`2601.13780`; GLAD `2403.16883`; NGG `2403.01535`; DMD `2311.18828`; RADD
`2604.25693` (retrieval-conditioned discrete denoiser for link prediction —
closest existing system); RDC `2305.15241` (diffusion-ELBO classifier);
generative classifiers avoid shortcuts `2512.25034`.
Rejected: `2409.07359` (workshop note, superseded), Jump Your Steps
`2410.07761` + `2509.19962` (backup few-step cites), DeepCache `2312.00858`
(intra-trajectory caching only), Shortcut Models `2410.12557`, `2303.15233`
(redundant with cited diffusion classifier), hyperbolic latent `2405.03188`,
`2403.17259` (diffusion only for negative sampling), `2603.17677` (text-RAG
specific).
