# EgoStitch E2E Redesign: Stitched-Topology-Conditioned Pair Encoder

**Date:** 2026-07-16 (rev 2, same day — after the second user-as-reviewer round)
**Status:** PENDING — this design lands only **after** the currently registered
frozen-s0 Stage-1 screening run completes and publishes. Until then, no edits to
`docs/03/04/05`, `configs/`, or `docs/registrations/` derive from this note; the
pending screening registration binds to the current spec state and must not be
edited out from under the run.

**Rev 2 delta:** the rev-1 headline (pooled topology context `c` fused by a
zero-init low-rank adapter) was assessed by the user-as-reviewer as readable as
"B0 pair encoder + pooled topology-side-information adapter," not a learned
topology representation. Rev 2 promotes an **explicit stitched-topology encoder
with zero-init gated cross-attention** to the headline; the pooled adapter is
demoted to a conditioning-depth rung. The innovation direction this design must
serve, verbatim: *the model learns a topology representation, conditioned on two
node features, where each node imagines its ego-neighborhood.*

## 1. Problem

The approved EgoStitch decision head anchors on a **pretrained frozen B0 logit**
(`s0`, spec §5/§13.10). Two independent problems:

1. **Reviewer-facing:** the model requires an externally pretrained checkpoint and
   composes evidence by logit-level late fusion — it reads as an extension of an
   existing scorer, not a standalone topology-conditioned model, and the late-fusion
   delta itself is weak methodological novelty. Rev-1's pooled-context adapter only
   partially fixed this: pooling the stitched scaffold into summary statistics
   before fusion still reads as side information, not a topology representation.
2. **Empirical:** the 2026-07-16 Seed-0 exact-quota diagnostic (diagnostic-only,
   superseded registration) showed the gated residual collapsing to B0: s0-logit
   correlation ≈ 1.0, validation AUPRC flat from epoch 1, assembled metrics
   indistinguishable from B0. A strong frozen anchor leaves the residual BCE
   gradient nothing easy to add.

**Constraint check (verified):** Blueprint §10 locked decisions are all satisfied by
this redesign (binary edge label target, generated topology as intermediate context,
independent-vs-topology-conditioned comparison, dual metric families, no test-graph
access). Frozen **features** remain frozen (integrity gate, untouched). Protocol §0's
*outer* boundary (per-query local context from frozen features → conditioned
classifier → binary label) is unchanged; the §0 component-table row assigning the
frozen pairwise scorer the `s0`-anchor role requires a dated disposition at landing
time (same pattern as the 2026-07-09 scaffold disposition). This note flags that
conflict; it does not resolve it unilaterally.

## 2. Decisions (with trail)

1. **E2E:** no pretrained checkpoint anywhere inside the model. The pairwise pathway
   is trained from scratch jointly with everything else under the locked objective.
   B0 survives only as an external baseline.
2. **Sequencing:** the frozen-s0 screen runs first under its existing registration.
   Decision branch, stated in advance: if it shows the expected dead residual, that
   is the citable motivation for this redesign; if it unexpectedly shows a live,
   assembled-metric-improving residual, this redesign's motivation narrows to
   "removes the pretraining dependency" and the frozen-s0 arm gains weight.
3. **Representation-level fusion** is the binding requirement (round 1): topology
   must enter the pair encoder's representation, not the logit.
4. **Round 2 (this rev): the topology object must be encoded explicitly, at token
   level.** The stitched scaffold's actual structure — imagined slots on both sides,
   intra-side adjacencies `Â_i, Â_j`, cross-side alignment `Π`, anchor/endpoint
   identity labels — is encoded by a **stitched-topology encoder (STE)** whose
   *token-level* outputs condition the trunk via **zero-init tanh-gated
   cross-attention**. Pooling-first designs are demoted to ladder rungs. Structure
   inputs and content inputs are **separated into independently ablatable
   pathways**, and the paper's topology-representation claim must survive removal
   of the content pathway.

## 3. Architecture

### 3.1 Objects

```text
T̂_ij:  V̂ = {i, j} ∪ S_i ∪ S_j                       (stitched scaffold, ≈ 2K+2 nodes)
       edges = star(u→slots, weight π·m)  ∪  intra-side slot–slot Â_i, Â_j
             ∪ cross-side alignment edges Π
       labels = target-relative anchor labels (R7), 4 types: endpoint-i /
                endpoint-j / slot-of-i / slot-of-j
                (grounded-identity-match is EXCLUDED from the structural label
                set — it derives from grounding/pointer information, which
                belongs to the content pathway; see §3.2)
```

### 3.2 Two token sets (the attribution split)

- **`c_topo` — structure-only tokens (the headline pathway).** STE node inputs are
  *exclusively structural*: the 4-type anchor label, existence `π`, multiplicity
  `m`, and soft degree features (weighted incident mass per edge type). Edge
  inputs: edge type (star / `Â` / `Π`) and weight. **No slot content embeddings
  `h`, no grounding gate `g`, and no grounded-identity-match label** — grounding
  is content evidence, and leaving its identity-match label in the structural set
  would let reviewers argue content leaks back into `c_topo`.
- **`c_content` — content tokens (separate, ablatable).** Slot content embeddings
  `h` tagged with `(π, g)`, grounding/pointer features including the
  grounded-identity-match label, and the membership signal (the former `s1`).
  This pathway may help; the paper's claim may not rest on it.

### 3.3 Stitched-topology encoder (STE)

2–3 layer edge-weighted message passing / light graph-transformer over `T̂_ij`
(the spec §5 `s4` GNN, **promoted**: same lineage, but its per-node states `{t_v}`
are the output — no pooled readout, no scalar channel). The STE *is* the learned
topology representation; the former `s4` scalar channel is retired at landing.

### 3.4 Conditioning (zero-init gated cross-attention, direction-symmetric)

Mapped to the actual V3.1 implementation: each `CrossAttentionLayer`
(`src/model/B0.py:523`) maintains and updates `(h_a, h_b, cls_token)`. The
Stage-1 pin is the simplest sufficient injection:

```text
for each direction stream (AB: anchor labels source=i, target=j; BA: swapped):
    after each of the final N_inj pair-cross-attention blocks:
        cls ← cls + tanh(g_topo) · XAttn(q=cls, kv={t_v}^{dir})       # STE tokens
        cls ← cls + tanh(g_cont) · XAttn(q=cls, kv={c_content}^{dir}) # separate gate
z'_AB, z'_BA → abba_max → head → p_ij = σ(head(z'))
```

Pins (all registered):

- **Queries: `cls_token` only** (Stage-1 default). `h_a`/`h_b` token-level
  injection is a registered deeper variant, not the screen headline.
- **Injection point: after** the pair-cross-attention block, for the final
  `N_inj` blocks; default `N_inj = 1`, sweep `{1, 2}`.
- **Parameter sharing:** AB and BA share the same STE and XAttn parameters;
  only the anchor labeling swaps.
- **Branch masks are per pair, shared across AB/BA** — otherwise `abba_max`
  would mix conditioned and unconditioned streams and break the decomposition.
- **Required unit test:** `p(i,j) = p(j,i)` (exact within fp tolerance) for the
  full model and for every null condition of §3.5; exact wiring into
  `PairContextGatedReadout` is an implementation-plan item with this test as
  the acceptance criterion.
- **Endpoint-exchange symmetry is constructed, not assumed:** conditioning is
  applied per direction *before* `abba_max`; the STE runs per labeling (cheap —
  ~2K+2 tokens). Pooled-`c` symmetry assumptions are explicitly disallowed.
- Trunk = V3.1 pair encoder from scratch (raw token sequences → encoder → pair
  cross-attn → `pair_context_gated`/`abba_max` readout, d_model 512). All gates
  `g_topo, g_cont` are zero-initialized: at init the model computes exactly the
  pair-only function.

### 3.5 Null taxonomy (three mutually exclusive head nulls; checkpoint-exact)

The rev-2 draft used one overloaded `∅_topology`; that is replaced by three
mutually exclusive null conditions (the `_head` suffix also resolves the naming
collision with the proposal §4.2 `∅_content` *generator-conditioning* dropout,
which is a different, unchanged mechanism):

| Null | Skips | Yields |
|---|---|---|
| `∅_all_head` | STE + topology XAttn + content XAttn | checkpoint-exact pair-only `f_logit` |
| `∅_topo_head` | STE + topology XAttn only | pair + content logit |
| `∅_content_head` | content XAttn only | pair + topology logit |

Because every conditioning sublayer is residual (`cls + tanh(g)·XAttn`), a
skipped sublayer is an exact identity at any checkpoint, so all four logits
(full, and the three nulls) come from the **same checkpoint** with no separate
comparator artifact.

**Training-time vs evaluation-time nulls are defined separately:**

- *Training* uses **per-sample multiplicative branch masks** — the sublayer runs
  for the whole batch and its residual update is zeroed for masked samples.
  For a residual sublayer this is numerically identical to the bypass (only
  wasted FLOPs differ), so mixed batches never require per-sample control flow.
- *Evaluation* uses **batch-level hard bypasses** — the skipped sublayers (and
  the STE, when unused by every sample) are literally not executed. A required
  unit test asserts train-mask ≡ eval-bypass logit equality per null condition.

### 3.6 Conditioning-depth ladder (study axis)

none (`B0-e2e`) → logit-FiLM `σ(a_θ(c)·f + b_θ(c))` → pooled low-rank adapter
(rev-1 design) → **STE + gated cross-attention (headline)**. The ladder
operationalizes the paper title: *where* and *in what form* topology conditions
the pair encoder is a measured axis.

## 4. Training

- Locked objective unchanged: `L = L_edge + λ_real·L_real + λ_ssl·L_ssl +
  λ_recon·L_recon`. No auxiliary loss attaches to trunk, STE, or gate parameters
  (the STE learns its representation from `L_edge` through the conditioning
  pathway); Modules 1–3 keep their auxiliary losses, and `L_edge` gradients flow
  through the STE and `c_content` into the generative modules — the proposal's
  gradient-routing requirement (usefulness-to-the-classifier) is preserved, and
  the covert-channel watchdog stays in force.
- **Curriculum:** trunk, STE, and gates train exactly when `L_edge` is active —
  i.e. not during the `L_recon`-only warm-start — so the pairwise trunk and the
  topology pathway enter joint training together.
- **Trained nulls, per pair (per-sample masks, §3.5):** the topology pathway is
  masked with probability `p_topo` (applying `∅_topo_head`) and the content
  pathway independently with `p_cont` (applying `∅_content_head`); their joint
  event realizes `∅_all_head`, so all three §3.5 null conditions are trained and
  in-distribution. Both probabilities registered (defaults 0.1–0.2, swept, plus
  `p = 0` arms); each pair's mask is shared across the AB/BA streams (§3.4).
  Because the null path *is* the trunk, the topology side never has to relearn
  class prior or calibration. Terminology: **branch dropout**; precedents cited,
  novelty not claimed for the mechanism: Modality Dropout (arXiv 2005.13616);
  modality competition in joint multimodal training (Huang et al., ICML 2022).
  No TDE claim attaches here; the TDE-style `∅_content`
  generator-conditioning dropout in proposal §4.2 is a different, unchanged
  mechanism — the `_head` suffix keeps the two namespaces disjoint.

## 5. Controls and ablation ladder (registered minimum)

The claim "–topology reproduces B0" from the first design round is **withdrawn**:
canonical B0 trains on raw token sequences / `train_plus` / 1:1 negatives /
lr 1e-4 / seed 47, while the Ours regime is `e_sup` / 1:5 negatives / lr 3e-4 /
seed 0 with `L_edge` active for only the post-warm-start 80% of budget
(`configs/b0_v31_breadth_first.yaml` vs `configs/egostitch_stage1_breadth_first.yaml`).
Three distinct objects:

- **B0-canonical** — the existing frozen external baseline (unchanged).
- **B0-e2e / f-only** — the trunk trained with `∅_all_head` permanent, under
  **exactly** the Ours data/negatives/edge-active-steps/optimizer/seed/HPO. The
  matched control; "–topology = B0-e2e" is the only claim made.
- **Exact-B0 reproduction** — trunk trained standalone under the canonical B0
  recipe; implementation sanity check only, never a paper arm.

Ladder (identical-head convention, matched budgets):

1. Full E2E (STE + gated cross-attention headline)
2. B0-e2e / f-only (matched-training pairwise-only arm)
3. E2E **B3-full** — trunk + all of Ours' auxiliary supervision as multi-task
   heads, none of the generative machinery. The protocol-defined Ockham arm; a
   live topology pathway is not evidence until it beats this.
4. E2E B5 (trunk + block + degree terms, jointly trained null hypothesis)
5. Frozen-s0 EgoStitch (the current design, retained as an arm)
6. **Structure-specificity controls** (the "live branch ≠ topology" battery):
   a. slot tokens kept, `Â`/`Π` shuffled within pair;
   b. identical tokens, all scaffold edges removed — STE degenerates to DeepSets;
   c. cross-pair topology shuffle (scaffold structure swapped between random
      pairs, endpoint tokens kept);
   d. same-capacity randomized-context arm (rev-1 arm 6, retained);
   e. **degree-preserving rewiring / weight permutation** — preserve node count,
      per-edge-type mass, per-node soft degree, and the weight distribution
      while destroying higher-order connectivity. The decisive answer to "is
      the STE encoding topology, or a continuous latent code disguised as a
      graph?" — a/b/c perturb degree too; only e isolates structure beyond
      degree;
   f. capacity-matched non-topological bottleneck — tokens of identical
      dimensionality and parameter count, no adjacency and no message passing
      (the parameter-matched sharpening of b).
7. **Pathway attribution:** pair + topology (`∅_content_head` permanent) vs
   pair + content (`∅_topo_head` permanent). **Headline requirement:** the
   topology-representation claim must survive content-pathway removal — the
   registered decision rule (set at registration time) requires the
   pair+topology arm to retain a defined share of the full-model gain over
   B0-e2e; otherwise the honest conclusion is content-side information.
8. No-direct-pair-context arm (head sees only conditioning tokens; *not* called
   "topology-only" — slots still derive from endpoint features)
9. Branch dropout: `p = 0` and the registered sweep, for both pathways
10. Conditioning-depth rungs: logit-FiLM; pooled low-rank adapter (rev-1)
11. Per-checkpoint decomposition, every headline table: all four §3.5 logits —
    full, `f_logit` (`∅_all_head`), pair+content (`∅_topo_head`), pair+topology
    (`∅_content_head`) — plus `topology_delta = full − pair+content` and
    `content_delta = full − pair+topology`

**Stage-1 screen scope (single-seed engineering screen — deliberately small):**
arms 1 (full), 2 (B0-e2e), 7 pair+topology (`∅_content_head` permanent), one
structure-destroyed control (6a, the simplest to implement correctly at screen
time), and 9 `p = 0`. Everything else — E2E B3-full, E2E B5, the
conditioning-depth rungs, the remaining structure battery including 6e/6f, and
the full shuffle set — is reserved for the formal multi-seed E1/E3 experiments.

## 6. Instrumentation, reporting, registration corrections

- **Correct spec anchors:** §13.16 is the fp32 score-precision pin; the
  dead-residual instrumentation lives in **§13.17**. Nothing is "migrated
  verbatim": at landing the §13.17 liveness signals (std ratio, Spearman, top-k
  overlap, conjunctive death rule) are **re-registered** against the
  within-checkpoint `f_logit` reference, replacing the fresh-frozen-s0 comparator
  artifact and its alignment machinery. The §13.16 fp32 pair-pass pin extends to
  the trunk pair pass, STE, gates, and head.
- **Representation evidence (new, required):** frozen-encoder linear probes from
  STE token/scaffold states to *real* local-topology quantities on held-out
  message-partition nodes — degree, ego density, clustering coefficient — plus an
  alignment-consistency probe (agreement of `Π` with real shared-neighbor
  structure on message-partition pairs). **Degree-partialled variants are
  required alongside the raw probes** (ego density and clustering probed after
  residualizing against degree): the link-prediction degree-bias literature
  (2405.14985, 2310.04612) shows topology signals in LP are heavily entangled
  with degree, so a probe suite that only recovers degree would not evidence a
  topology representation. Reported with every headline table; probe protocol
  and thresholds fixed at registration. Without this, the evidence shows only
  "topology branch helps prediction," not "a topology representation was
  learned."
- **Branch competition telemetry** (extends the §13.17 probe): per-pathway gate
  magnitudes `tanh(g)` per block over training, per-branch RMS gradient norms and
  relative update norms (trunk vs STE vs content pathway vs Modules 1–3),
  `topology_delta` scale per epoch.
- **Cost honesty:** parameter counts per arm, active edge examples seen, FLOPs,
  GPU-hours, and candidate-universe scoring latency required in every arm's
  report. HPO statement corrected: 30 configs × 3 seeds establishes **trial-count
  parity only**; GPU-hours per arm are reported so compute/search density is
  inspectable.

## 7. Engineering deltas (this is not "remove a cache")

- The EgoStitch worker gains a **packed-token edge stream**: raw variable-length
  token tensors for both endpoints, token bucketing, and its interaction with the
  node-stream/edge-stream batch sampler (spec §10/§13.13). Reuses the
  worker-generic `e2_pipeline` pack → probe → projection machinery that already
  drives B0 V3.1 DDP training.
- New modules: STE (small — ~2K+2 tokens, 2–3 layers, promoted s4 lineage) and
  2×N gated cross-attention sublayers in the trunk; per-pair scoring now runs
  trunk + STE + both directional conditioning passes over ~2.04 M candidate rows,
  on top of the cached per-node encode pass. Expected budget class: the E2 B0 run
  (3,600 s total-budget pin), **not** the Stage-1 profile (673 s / 2.04 GiB),
  which does not extrapolate. A measured re-estimate on the H20 shape is a
  required deliverable before the replacement registration is written.
- Retired at landing: the `s0_cache` / `s0_checkpoint_id` config keys, the
  frozen-B0 logit cache (spec §13.10), the fresh-frozen-s0 comparator scoring
  step in the gate, and the separate `s4` scalar channel (absorbed by the STE).

## 8. Landing sequence (after the frozen-s0 screen publishes)

1. `docs/04-model-proposal.md` rev: §4.4 rewritten from anchored late fusion to
   the stitched-topology-conditioned encoder; §4.4's `s4` promoted into the STE;
   SHOT frozen-hypothesis citation rescoped to the frozen-s0 ablation arm;
   conditioning-depth ladder added; novelty-scoping paragraph per §10 below
   (verified citations only, per repository convention).
2. `docs/05-egostitch-spec.md` edits, each with a change-log line per the freeze
   rule: §5 (head: STE + gated cross-attention, pathway split, hard bypass), §7
   (no new lambda), §8 (curriculum), §13.1 (Stage-1 head — Stitch is already
   retained in Stage 1, so the STE is Stage-1-runnable), §13.10 (retired),
   §13.16 (fp32 scope extension), §13.17 (re-registered liveness reference,
   thresholds, gate/probe telemetry). The head-null names adopt the §3.5
   `_head`-suffixed taxonomy verbatim, keeping them disjoint from the proposal
   §4.2 `∅_content` generator-conditioning dropout.
3. `docs/03-experiment-protocol.md`: dated §0 component-table disposition (frozen
   pairwise scorer loses the `s0`-anchor role; keeps B0-baseline and
   E4.10-proposer roles); baseline table gains B0-e2e and the E2E B3-full/B5
   instantiations; E4 gains the structure-specificity battery.
4. New Stage-1 screening registration binding the e2e architecture, scoped to
   the **Stage-1 minimum arm set pinned in §5** (full, B0-e2e, pair+topology,
   structure control 6a, `p = 0`) — not the full ladder — plus the §6 probe
   protocol and thresholds and the pathway-attribution decision rule. The
   remaining arms enter the E1/E3 registrations.

## 9. Risks

| Risk | Owner control |
|---|---|
| Topology gates stay dead (branch never used) | Gate-magnitude telemetry; branch dropout; conditioning-depth rungs isolate where signal dies |
| Live branch ≠ topology content (covert channel) | Structure-specificity battery (arm 6) + E2E B3-full (arm 3) are the deciding controls |
| **Content pathway carries the win** | Pathway attribution (arm 7) is a headline requirement with a registered decision rule, not an optional ablation |
| STE uses tokens but ignores structure ("a latent code disguised as a graph" — π, m, Â, Π are all feature-derived) | Arm 6e (degree-preserving rewiring) is the decisive control; 6a/6b/6f support; degree-partialled representation probes give direct evidence |
| Edge-metric floor: joint model under B0-canonical AUPRC | B0-e2e matched arm separates regime effects from architecture effects; protocol dual-metric reporting unchanged |
| Cost blowout on token-stream training/scoring | §7 re-estimate gate before registration; budget class re-anchored to E2 pin |
| HPO fairness challenge | Trial-count vs compute parity distinction registered; GPU-hours per arm reported |
| Frozen-s0 screen returns a live residual | Pre-stated decision branch (§2.2); redesign proceeds with narrowed motivation |

## 10. Novelty scoping (2026-07-16 sweep: local vault + external arXiv)

Two sweeps were run before committing this rev: the local `literature/` vault
(full PDF manifest, the four `research_reports/` synthesis notes, GSL taxonomy,
plus text extraction of the three most decisive PDFs) and an external arXiv
sweep in which every ID below was verified against its abstract page on
2026-07-16. Repo convention still applies: re-verify each citation before it is
quoted as fact in `docs/04-model-proposal.md` §8 at landing time.

### 10.1 Direct-anticipation verdicts

- **(i) Encoder over *generated* local topology conditioning a pair scorer — no
  exact match identified in the reviewed corpus.** Nearest: Leap (2503.03331)
  proposes inductive topology augmentation — so the paper must **never** claim
  "the first method to generate structural context for unseen nodes"; the
  distinction rests on *two* unseen endpoints, self-contained ego-nets, and
  OT-based stitching. Leap grafts MLP-predicted edges onto the
  *observed* graph via real anchor nodes; NCNC (2302.00890) *completes*
  common-neighbor structure over the observed adjacency; FLEX (2507.11710)
  generates link-subgraphs as training augmentation, not per-query inference
  context; VGAE subgraph prediction (2408.04053) is transductive and the
  generator *is* the scorer; LA-GNN (2109.03856) generates neighbor *features*
  only. None fabricates a self-contained ego-net scaffold from frozen features
  under a zero-edge protocol and encodes it with a structure-only encoder to
  condition a separate pair encoder.
- **(ii) Gated cross-attention from graph-structure tokens into a pair encoder —
  no exact match identified in the reviewed corpus; cross-attention itself is
  NOT claimable as an independent contribution.** Nearest: CAM tokens
  (2405.19375, cross-attentive modulation
  for linkset prediction — but over *observed* node/edge tokens, no zero-init
  gate, no pair-only bypass); GraphToken (2402.05862) and GraphGPT (2310.13023)
  condition a *pretrained LLM* for reasoning, not a binary edge logit; Flamingo
  (2204.14198) is the zero-init tanh-gate lineage (method-transfer source, not
  anticipation); GNN-FiLM (1906.12192) modulates messages on the observed graph.
- **(iii) OT-stitching of two *generated* ego-nets — no exact match identified
  in the reviewed corpus; the most defensibly novel element.** Nearest: GOAT
  (2111.05366) and SLOTAlign
  (2301.12721) OT-align two *observed* graphs where alignment is the end task;
  no found work aligns two independently imagined ego-neighborhoods as an
  internal differentiable step feeding an edge decision.

### 10.2 Claim discipline (binding for the proposal rev)

A limited literature search cannot prove no prior work exists, so the claim is
**novel overall composition**, never per-component unprecedentedness. The
registered contribution statement:

> *Dual imagined ego-nets are differentiably aligned and stitched into a
> generated local scaffold whose structure-only token representation conditions
> a queried-edge pair encoder under the strict zero-edge inductive protocol.*

Every ingredient has named ancestry — DETR set decoding (2005.12872), the
labeling trick (2010.16103), entropic OT matching, Flamingo-style zero-init
gating (2204.14198), FiLM/graph-token conditioning — and the proposal rev must
carry a per-component table reporting, for each mechanism: **ancestry** (where
it comes from), **prior usage** (what it was used for), and **difference**
(what this work changes). The proposal rev must name and distinguish Leap and
CAM tokens explicitly (the two closest works on axes (i) and (ii)), alongside
the SEAL/labeling-trick family (1802.09691, 2010.16103, 2302.00890, 2209.15486)
whose defining delta is that they read observed subgraphs.

### 10.3 Adversarial evidence the paper must own (not hide)

- **Edge-independent ceiling** (2111.00048): if the topology pathway collapses
  to the pair-only logit, the model regresses to the G2 edge-independent ceiling
  — exactly the Seed-0 failure mode. The liveness telemetry and pathway
  attribution rule (§5 arm 7, §6) are the pre-registered answer.
- **Hard-negative fragility** (HeaRT, 2306.10453): structural LP gains often
  evaporate under realistic negatives — mirrors our G1 result; the gate's
  hard-negative rows already report this axis for every arm.
- **Degree entanglement** (2405.14985, 2310.04612): generated topology may
  re-encode a degree prior. Answered by the degree-partialled probe suite (§6),
  the PA-null and Odds-product baseline rows already in the protocol, and the
  structure-specificity battery (§5 arm 6).
- **Attribute–topology entanglement** (2307.08877): topology signals can hurt
  inductive generalization when not disentangled from attributes — direct
  motivation for the `c_topo`/`c_content` pathway split being a headline
  requirement rather than an ablation.
