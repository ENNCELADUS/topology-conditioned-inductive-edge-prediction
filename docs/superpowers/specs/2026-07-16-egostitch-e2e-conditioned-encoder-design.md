# EgoStitch E2E Redesign: Topology-Conditioned Pair Encoder

**Date:** 2026-07-16
**Status:** PENDING — this design lands only **after** the currently registered
frozen-s0 Stage-1 screening run completes and publishes. Until then, no edits to
`docs/03/04/05`, `configs/`, or `docs/registrations/` derive from this note; the
pending screening registration binds to the current spec state and must not be
edited out from under the run.

## 1. Problem

The approved EgoStitch decision head anchors on a **pretrained frozen B0 logit**
(`s0`, spec §5/§13.10). Two independent problems:

1. **Reviewer-facing:** the model requires an externally pretrained checkpoint and
   composes evidence by logit-level late fusion — it reads as an extension of an
   existing scorer, not a standalone topology-conditioned model, and the late-fusion
   delta itself is weak methodological novelty.
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
   is the citable motivation for this redesign ("anchored residual collapses; hence
   joint training"); if it unexpectedly shows a live, assembled-metric-improving
   residual, this redesign's motivation narrows to "removes the pretraining
   dependency" and the frozen-s0 arm gains weight in the ladder.
3. **Binding architectural requirement:** the topology representation is **fused
   into the pair encoder's representation** (representation-level conditioning), not
   added at the logit. Registered headline mechanism: **zero-initialized low-rank
   adapter** on the final pair representation. Gated cross-attention to slot tokens
   is the registered deeper-fusion extension arm; logit-FiLM is a ladder rung; the
   previous late additive fusion is demoted to a rung. Late fusion is explicitly
   **not** the headline.

## 3. Architecture

```text
z    = trunk_pair_repr(tokens_i, tokens_j)      # V3.1 pair encoder, from scratch:
                                                # raw token sequences → encoder →
                                                # pair cross-attn → pair_context_gated
                                                # / abba_max readout (d_model 512)
c    = topo_context(S_i, S_j, Π, s-features)    # from Imagine + Stitch (below)
z'   = z + B·A·[z; c]                           # low-rank adapter, B zero-init,
                                                # rank r registered (sweep 16–64)
p_ij = σ(head(z'))
```

- **Topology context `c` (Stage-1 form):** pooled slot embeddings of both endpoints
  weighted by `(π, m, g)`, an alignment-plan summary (Π mass, entropy, top-k
  agreement), the closure features `s2`/`s2_aa`, and the membership feature `s1`.
  The full model appends `s3` (affiliations/degree budgets) and the scaffold-GNN
  states behind `s4`. The s-channels stop being late-fused scalars and become
  conditioning inputs.
- **Exact recovery is a checkpoint property:** with the adapter nulled
  (`∅_topology`), the same checkpoint computes exactly `σ(head(z))` — the pure
  pair-encoder logit `f_logit`. `topology_delta = logit − f_logit` is exact and
  per-pair, from one checkpoint, with no frozen-cache alignment machinery.
- **Conditioning-depth ladder (study axis, operationalizing the paper title):**
  none (`B0-e2e`) → logit-FiLM `σ(a_θ(c)·f + b_θ(c))` → representation adapter
  (headline) → zero-init tanh-gated cross-attention from the last trunk blocks to
  slot tokens (extension arm, run only if the adapter shows liveness).

## 4. Training

- Locked objective unchanged: `L = L_edge + λ_real·L_real + λ_ssl·L_ssl +
  λ_recon·L_recon`. No auxiliary loss attaches to trunk or adapter parameters
  (mirroring B0's pure-BCE training); Modules 1–3 keep their auxiliary losses, and
  `L_edge` gradients still flow through `c` into the generative modules — the
  proposal's gradient-routing requirement (usefulness-to-the-classifier) is
  preserved, and the covert-channel watchdog therefore stays in force.
- **Curriculum:** trunk + adapter train exactly when `L_edge` is active — i.e. not
  during the `L_recon`-only warm-start — so the pairwise trunk and topology
  channels enter joint training together.
- **`∅_topology` is a trained null** with a registered dropout probability
  (default 0.1–0.2, swept, plus a registered `p = 0` arm): during training the
  adapter update is zeroed **per pair** with probability p, keeping the null path
  in-distribution and calibrated. Because the null path *is* the trunk, the
  topology side never has to relearn class prior or calibration.
  Terminology: **topology-branch dropout**. Precedents cited, novelty not claimed
  for the mechanism: Modality Dropout (arXiv 2005.13616); modality competition in
  joint multimodal training (Huang et al., ICML 2022). No TDE claim attaches to
  this mechanism; the TDE-style `∅_content` contrast in proposal §4.2 is a separate,
  unchanged control.

## 5. Controls and ablation ladder (registered minimum)

The claim "–topology reproduces B0" from the first design round is **withdrawn**:
canonical B0 trains on raw token sequences / `train_plus` / 1:1 negatives /
lr 1e-4 / seed 47, while the Ours regime is `e_sup` / 1:5 negatives / lr 3e-4 /
seed 0 with `L_edge` active for only the post-warm-start 80% of budget
(`configs/b0_v31_breadth_first.yaml` vs `configs/egostitch_stage1_breadth_first.yaml`).
Three distinct objects:

- **B0-canonical** — the existing frozen external baseline (unchanged).
- **B0-e2e / f-only** — the trunk with the adapter permanently nulled, trained
  under **exactly** the Ours data/negatives/edge-active-steps/optimizer/seed/HPO.
  This is the matched control; "–topology = B0-e2e" is the only claim made.
- **Exact-B0 reproduction** — trunk trained standalone under the canonical B0
  recipe; implementation sanity check only, never a paper arm.

Ladder (all arms under the identical-head convention and matched budgets):

1. Full E2E (adapter headline)
2. B0-e2e / f-only (matched-training pairwise-only arm)
3. E2E **B3-full** — trunk + all of Ours' auxiliary supervision as multi-task heads
   (degree NLL, BP-NLL, ego-net statistics, distributional loss), none of the
   generative machinery. Already protocol-defined as the decisive Ockham control;
   a live adapter is not evidence for topology until it beats this arm.
4. E2E B5 (trunk + block + degree terms, jointly trained null hypothesis)
5. Frozen-s0 EgoStitch (the current design, retained as an arm)
6. **Same-capacity shuffled/randomized topology** — adapter kept, context content
   destroyed (slots shuffled across pairs / resampled). If the adapter still helps,
   the improvement is not topology content. Direct control on the covert-channel
   risk (proposal §4.5 watchdog).
7. No-direct-pair-context arm (head sees only `c`; *not* called "topology-only" —
   it still observes endpoint features through the slots)
8. Topology-branch dropout: `p = 0` and the registered sweep
9. Conditioning-depth rungs: logit-FiLM; gated cross-attention extension
10. Per-checkpoint decomposition, reported for every headline table:
    `f_logit`, `topology_delta`, fused logit

## 6. Instrumentation, reporting, registration corrections

- **Correct spec anchors:** §13.16 is the fp32 score-precision pin; the
  dead-residual instrumentation lives in **§13.17**. Nothing is "migrated
  verbatim": at landing time the §13.17 liveness signals (std ratio, Spearman,
  top-k overlap, conjunctive death rule) are **re-registered** against the
  within-checkpoint `f_logit` reference, replacing the fresh-frozen-s0 comparator
  artifact and its alignment machinery. The §13.16 fp32 pair-pass pin extends to
  the trunk pair pass, adapter, and head.
- **Branch competition telemetry** (extends the §13.17 probe): per-branch RMS
  gradient norms and relative update norms (trunk vs adapter vs Modules 1–3),
  adapter output norm trajectory, `topology_delta` scale per epoch.
- **Cost honesty:** parameter counts per arm, active edge examples seen, FLOPs,
  GPU-hours, and candidate-universe scoring latency are required in every arm's
  report. The HPO statement is corrected: 30 configs × 3 seeds establishes
  **trial-count parity only**; GPU-hours per arm are reported so compute/search
  density is inspectable.

## 7. Engineering deltas (this is not "remove a cache")

- The EgoStitch worker gains a **packed-token edge stream**: raw variable-length
  token tensors for both endpoints, token bucketing, and its interaction with the
  node-stream/edge-stream batch sampler (spec §10/§13.13). Reuses the
  worker-generic `e2_pipeline` pack → probe → projection machinery that already
  drives B0 V3.1 DDP training.
- Candidate-universe scoring runs the full trunk per pair (~2.04 M candidate rows)
  plus the cached per-node encode pass. Expected budget class: the E2 B0 run
  (3,600 s total-budget pin), **not** the Stage-1 profile (673 s / 2.04 GiB), which
  does not extrapolate. A measured re-estimate on the H20 shape is a required
  deliverable before the replacement registration is written.
- Retired at landing time: the `s0_cache` / `s0_checkpoint_id` config keys, the
  frozen-B0 logit cache (spec §13.10), and the fresh-frozen-s0 comparator scoring
  step in the gate.

## 8. Landing sequence (after the frozen-s0 screen publishes)

1. `docs/04-model-proposal.md` rev: §4.4 rewritten from anchored late fusion to the
   conditioned-encoder head; SHOT frozen-hypothesis citation rescoped to the
   frozen-s0 ablation arm; conditioning-depth ladder added; novelty-scoping
   paragraph for the adapter (zero-init adapter/ControlNet-LoRA lineage cited;
   claimed contribution is the conditioning of a pair encoder on *generated* local
   topology under the strict zero-edge protocol, not the adapter mechanism itself).
2. `docs/05-egostitch-spec.md` edits, each with a change-log line per the freeze
   rule: §5 (head), §7 (no new lambda; trunk under `L_edge`), §8 (curriculum),
   §13.1 (Stage-1 head), §13.10 (retired), §13.16 (fp32 scope extension),
   §13.17 (re-registered liveness reference and thresholds).
3. `docs/03-experiment-protocol.md`: dated §0 component-table disposition (frozen
   pairwise scorer loses the `s0`-anchor role; keeps B0-baseline and E4.10-proposer
   roles); baseline table gains B0-e2e and the E2E B3-full/B5 instantiations.
4. New Stage-1 screening registration binding the e2e architecture, with the §6
   corrections and the §5 ladder arms that fall inside Stage-1 scope.

## 9. Risks

| Risk | Owner control |
|---|---|
| Adapter stays at zero (dead topology branch) | §6 liveness telemetry; topology-branch dropout; conditioning-depth rungs isolate where signal dies |
| Live adapter ≠ topology content (covert channel) | Arm 6 (shuffled/randomized context) + Arm 3 (E2E B3-full) are the deciding controls |
| Edge-metric floor: joint model under B0-canonical AUPRC | B0-e2e matched arm separates regime effects from architecture effects; protocol dual-metric reporting unchanged |
| Cost blowout on token-stream training/scoring | §7 re-estimate gate before registration; budget class re-anchored to E2 pin |
| HPO fairness challenge | Trial-count vs compute parity distinction registered; GPU-hours per arm reported |
| Frozen-s0 screen returns a live residual | Pre-stated decision branch (§2.2); redesign proceeds with narrowed motivation |
