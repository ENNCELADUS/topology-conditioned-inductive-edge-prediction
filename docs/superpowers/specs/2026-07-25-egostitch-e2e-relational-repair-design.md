# EgoStitch E2E Relational Repair: rev 3.1 Fix Design

**Status: DECIDED (owner-delegated, 2026-07-25).** This document specifies the
rev-3.1 repair of the rev-3.0 e2e conditioned encoder after the binding
2026-07-24 `cut` (`docs/results/G5-e2e-stage1-seed0-20260724.md`). The owner
delegated the final decisions on the §7 decision points to the working agent in
the 2026-07-25 session; the resolved decisions are recorded in §7 with that
provenance and may be revisited by the owner at any time. This document is a
decision trail, not a contract: per the spec freeze rule, every code-facing
item below lands only through `docs/05-egostitch-spec.md` §14 /
`docs/03-experiment-protocol.md` §0 amendments with change-log lines, followed
by a fresh v3 BINDING registration. All registered-pin conflicts are enumerated
in §7.1 (five items at decision time; the A1/protocol-§0 conflict became moot
at r5 when the measured P0.2 curve eliminated the reranker, leaving four
active pin edits).

Revision history: r1 drafted 2026-07-25 from the adversarially verified failure
diagnosis; r2 same day, after an independent adversarial review of r1 that
surfaced 20 defects (3 blocker / 10 major / 7 minor) — all resolved, including
two corrected factual premises about the current code (the edge stream builds
no ego targets; two of three proposed scaffold feature channels already
existed) and a previously unflagged locked-protocol conflict
(comparator-scorer reuse for grounding pools vs `docs/03-experiment-protocol.md`
§0). r3 (this text) same day, after a Codex adversarial review of r2 returned
five high-severity findings, all verified against code/spec and all accepted:
(1) P0 audits could have tuned on sealed `V_qual`/`V_select` topology — all
pre-freeze topology measurements are now `G_fit`-scoped (§13.19.4 boundary);
(2) `L_rel` assigned structurally false "no shared structure" labels to
negatives (the `NegativeSampler` excludes only true positives — non-edges
freely share neighbors); targets are now computed from `G_fit` for every pair;
(3) the proposed rank-seeded edge-stream target RNG contradicted the
world-size-invariance acceptance test — now pair-identity-keyed; (4) the
promised `no_l_rel` ablation was absent from the arm set and the gate fails
closed on the pinned five-arm schema (`g5_stage1.py:1205-1217`) — the v3
screen is now an explicit eight-arm schema with a WS7 gate change and
re-derived cost; (5) the sequencing calibrated qualification thresholds from
the same run they gate and ran code before spec edits — reordered to
spec-first, `V_fit`-calibrated, frozen, then a single sealed `V_qual`
rehearsal (≤3 attempts) before BINDING. r4 (same day) reconciles the second
Codex re-review (4 findings, all verified and accepted; no §7 decision
overturned): per-endpoint node-identity target keying replaces r3's
single pair-keyed RNG (traversal-order/AB-BA hazard; `L_rel` targets need no
RNG; negative sampling stays rank-drawn — invariance is per-pair, not
stream-composition); a masked, all-reduced reduction contract for
`L_rel`/`L_align` (padded duplicate rows, §13.19.1 class); the eight-arm
migration expanded from the gate tuples to every enforcement point
(`train_egostitch.py:1227,1315-1388`, `score_universe.py:98,991-1110,
2155-2439`) plus a pinned deterministic `6e-v1` checkerboard-swap algorithm;
and the pool cache hash widened to bind the F0/source-pack digest and
role-universe identity. r5 (same day) records the **measured P0.2/P0.4
results** (Phase 0 results block) and the delegated resolution of the fired
A-branch stop rule: no pool variant reaches even a 0.3 pair ceiling (max
0.179 at cosine-100 / rerank-50), so the grounded-identity chain is
**ceiling-limited by feature retrieval itself** and is re-scoped to a
secondary channel; cosine top-50 is adopted (§7.1.2 resolution), the B0-alt
reranker is measured-and-rejected (≈ cosine at 2.5× pool size, unreachable
target either way, and the §0 amendment becomes unnecessary), primary
relational weight shifts to the pool-independent channels (B3 + F.1 + D,
whose P0.4 teacher density measured healthy at `P(|S|>0) = 0.663`), and G3
gate 1 is recalibrated from an absolute slot-recall floor (falsified by the
measured pool ceiling) to a pool-ceiling-relative form. r6 (same day):
container P0.1/P0.3 autopsy measured — full collapse confirmed (`h` cosine
0.9997, Â constant 0.5014, pointer exactly uniform, gate never above 0.5);
C3's Π row-entropy arm measured blind to full collapse and replaced with a
rank-1 marginal-residual criterion (spec §14.4.8 refined, second §12
change-log entry). Phase 0 is complete.

Diagnosis basis: adversarially verified code audit, 2026-07-25 (all claims
traced to file:line; one correction — Π is not gradient-free, it receives
`L_edge` gradient via scaffold→STE→trunk but has no identity target).

---

## 1. Failure ledger (what must be fixed)

| # | Verified failure | Evidence |
|---|---|---|
| F1 | Pointer head receives zero gradient from any loss; only consumer is `grounded_identity_match` via non-differentiable argmax | `losses.py` (no pointer term), `scaffold.py:56` |
| F2 | Grounding pools (F0-cosine top-20) contain essentially no true neighbors: slot recall@k = 0.0 train and test | `grounding.py:26`, `G5-stage1-seed0-20260717.md:102` |
| F3 | `target_in_pool` labels ≈ all-False → `L_gate` drives gate→0 → Π-consistency probe (`gate > 0.5` both sides) structurally forced to exactly 0 | `losses.py:92-103`, `probes.py:616` |
| F4 | Π has content-only cost, no identity/alignment target; shared-neighbor correspondence never learned | `stitch.py:24-41` |
| F5 | Slot collapse: `structure_control_6a` (adj-only slot permutation, feats fixed) inert to six decimals ⇒ Â near-constant, Π near-uniform, slots interchangeable | `score_universe.py:122-146`, result note |
| F6 | With Â/Π collapsed, the scaffold reduces to a degree code; no explicit triangle/closure signal exists anywhere in the structural inputs (probe: degree R² 0.382 vs partialled clustering 0.021) | `scaffold.py:106-114`, gate JSON probes |
| F7 | Conditioning pathway converged to a near-constant calibration offset: `topology_delta = +0.144 ± 0.085` vs `f_logit` std 2.85 | gate JSON `decomposition.arms.full` |
| F8 | The surviving degree-like residual actively degrades degree-corrected AUPRC (guard fail: 0.705903 vs floor 0.710260) | gate JSON `guards` |
| F9 | Dynamics starve the pathway: Phase A (20%) trains the trunk pair-only before conditioning activates; zero-init gates; branch dropout keeps the trunk self-sufficient; aux:edge gradient family ratios ~500×/1,682× (frozen-s0 measurement) | `train_egostitch.py:748-759, 2752-2757`, frozen-s0 note §3 |
| F10 | No pair-level relational training pressure exists anywhere: nothing asks any module to predict shared-neighbor structure for a *pair* | design §4 ("no aux loss on trunk/STE/gates") |
| F11 | The Π-consistency probe conflates alignment quality with grounding quality (reads 0 whenever gates are shut, regardless of Π) | `probes.py:608-628` |
| F12 | The registered control battery could not detect collapse: an inert 6a control was reported as a *failed check*, not as the collapse alarm it actually was; 6e (degree-preserving rewiring) was deferred | design §5 ladder |

Design rule for §3: one mechanism per failure where a mechanism exists (Occam
during construction); F8 is honestly mapped to E1/E2 with E4 as its watchdog
rather than claiming a dedicated mechanism. §6 then attacks the composed system
with Murphy-style qualification gates.

## 2. Constraints honored

- **Integrity gates (protocol §E5):** all new supervision is train-side only and
  E_msg-derived; retrieval and grounding for test nodes remain feature-only
  (frozen features; never test-graph edges). Node-disjointness untouched.
  Probe measurement targets derive from the train-side `G_struct` (E_msg over
  operative train nodes) only, and are evaluation-side-only quantities.
- **Attribution split (design §3.2):** `c_topo` stays structure-only at
  inference — every new loss shapes *training*; no content/grounding channel is
  added to the eval-time STE inputs. (Π was always `h`-derived; rev 3.1 does not
  change Π's eval-time inputs, and the D-channel additions are functions of
  `{star, Â, Π}` only.)
- **Locked objective (blueprint §10 / methodology):** outer form
  `L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon` is preserved.
  Every new term — `{L_ptr, L_align, L_div, L_rel}` — folds into the registered
  `L_recon` decomposition by spec change-log (the same mechanism that defined
  `L_feat…L_gate`); `L_rel` is classified as relational reconstruction of real
  local-topology quantities. No fifth top-level term is created.
- **Terminology guardrail:** the generated/stitched scaffold remains
  intermediate context for predicting `edge(u, v)`; nothing below is a
  graph-generation objective.
- **One-seed screen rules:** the rev-3.1 screen stays a fixed-Seed-0
  engineering screen; no significance claims; E1/E3 remain ≥3-seed Holm.
- **Locked-decision discipline:** conflicts with registered pins are *flagged*
  (§7.1), never resolved unilaterally here.

## 3. Fixes

### Phase 0 — Autopsies before code (calibrates everything downstream)

No retraining; existing checkpoint `a471010f57e495f0` + data only.

**Data-role boundary (r3 fix — binding for every P0 item and every downstream
decision rule):** all topology-based measurements are computed on
`G_fit = E_msg[V_fit]` **only**. `E_msg[V_qual]` and `E_msg[V_select]` are
evaluation-only topology under spec §13.19.4's sealed-selection boundary:
`V_qual` is reserved for the single post-freeze qualification rehearsal
(≤3 attempts total per §13.19.4), `V_select` stays sealed until the first
bound run, and the test side is **never** topology-audited — test-side pool
caches are validated feature-only (node-set + schema + `pool_method_hash`).
A2's `n_ground` rule, A3's `L_gate` pos-weight constant, `w_align` freezing,
and every G3 threshold consume `G_fit` curves exclusively.

- **P0.1 Slot-dispersion autopsy** (discriminates F5's severity): per-node std
  of π across slots, mean pairwise `h` cosine, Â off-diagonal std, Π row
  entropy vs `log K`, over `V_fit` nodes. Output: the collapse baseline the §6
  gates must beat.
- **P0.2 Grounding-ceiling audit** (sizes A1/A2), `G_fit`-scoped: for pool
  variants {cosine top-20 (status quo), cosine top-{50, 100}, two-stage
  reranked top-{20, 50} per A1}, measure (a) slot-recall ceiling: fraction of a
  `V_fit` node's true `G_fit` neighbors present in its own pool; (b) pair
  ceiling: fraction of `G_fit` pairs with ≥1 true common neighbor in
  `G(u) ∩ G(v)`. Pool-build cost is itemized per role universe (spec §13.12
  isolates V_fit / V_qual / V_select / test — one cache may never serve
  another), with the reranker cost model of A1; the V_qual/V_select/test pool
  *builds* are mechanical cache generation with feature-only validation — no
  topology metric is computed on them pre-binding.
- **P0.3 Gate/pointer histograms**: confirm gate mass < 0.5 and pointer ≈
  uniform (untrained). Before/after evidence for the repair note.
- **P0.4 Teacher-density audit** (sizes B3; added in r2), `G_fit`-scoped: on
  edge-stream positive pairs within `V_fit`, measure `P(|S| > 0)` and
  `E[|S|]` for the B3 teacher-cell set `S`, stratified by endpoint degree. Two
  compounding sparsifiers must be measured, not assumed: (a) each endpoint
  independently subsamples at most K = 16 targets, so a common neighbor
  survives on both sides with probability ≈ (16/deg_u)(16/deg_v); (b) a common
  neighbor `w` counts only if **both** `(u,w)` and `(v,w)` landed in E_msg
  (the partition removes the E_sup fraction). High-degree pairs — where
  clustering-MMD signal concentrates — are the most teacher-starved; if
  stratified density is unusably low, B3 switches to matching against the full
  common-neighbor set rather than the K-capped subsample (decision recorded
  before `w_align` is frozen).

### Phase 0 results (measured 2026-07-25, local run; archived under `outputs/p0_audit_20260725/`)

P0.2 and P0.4 ran locally (41 s; deterministic under seed 0 / msg_fraction 0.8
/ holdout 256). Population: 8,070 operative train nodes, `|V_fit| = 7,558`,
`|E_msg[V_fit]| = 22,708`, `|E_sup[V_fit]| = 6,612`, mean/median `G_fit`
degree 6.0/3.

| Pool variant | Slot-recall ceiling | Pair ceiling (e_sup_fit, uncond.) | Pair ceiling (cond. on ≥1 common) |
|---|---:|---:|---:|
| cosine top-20 (status quo) | 0.107 | 0.095 | 0.133 |
| cosine top-50 | 0.140 | 0.134 | 0.187 |
| cosine top-100 | 0.173 | 0.179 | 0.250 |
| B0-alt rerank top-20 | 0.122 | 0.134 | 0.187 |
| B0-alt rerank top-50 | 0.160 | 0.179 | 0.250 |

`P(≥1 common G_fit neighbor) = 0.717` for e_sup positives — the graph carries
abundant triangle signal; **feature retrieval simply cannot reach it** (the
reranker ≈ doubles effective pool size but tracks the cosine curve; no
variant approaches the 0.5 target or the 0.3 stop floor). Two consequences:
the rev-3.0 grounded-identity chain was ceiling-limited **by construction**
(≤ 9.5% of positives could ever fire under status-quo pools), and no legal
vocabulary fixes this (test-side pools are inherently feature-based; graph-
expanded train pools would create a train/test retrieval mismatch).

P0.4 (pool-independent `L_align` teacher, K-cap 16): overall
`P(|S| > 0) = 0.663`, `E[|S|] = 2.12`; by degmax quartile (edges 7/16/38):
0.507 / 0.645 / 0.688 / 0.798. The r2 starvation hypothesis was **inverted**
— high-degree pairs are the *best*-served stratum (more common neighbors
outweigh the (16/deg)² subsampling); the K-cap costs only ~5 pp versus
uncapped (0.663 vs 0.717). **B3's teacher is healthy; K-capped matching is
retained; the uncapped fallback is not needed.**

P0.1/P0.3 (measured on the container, checkpoint `a471010f57e495f0`, 261 s;
r6): the collapse diagnosis is confirmed quantitatively and near-totally —
mean pairwise `h` cosine **0.9997** (p05 0.9995), Â off-diagonal **constant
0.5014 ± 0.0004**, π std across slots 0.042, plan max-cell fraction 0.045;
the pointer is **exactly uniform** (max prob 0.052 ≈ 1/n_g, normalized
entropy 0.9999 — the untrained-head prediction to four decimals) and the
gate **never crosses 0.5 anywhere** (mean 0.088, fraction above 0.5 = 0.0
over all 7,558 × 16 slots — the structural-zero explanation of
Π-consistency exact). One calibration correction fell out (r6): a Π
**row-entropy** collapse criterion is blind to full collapse (measured 0.624
normalized, because the `π·m` marginals concentrate rows regardless of
cost-blindness); C3's second arm is therefore a rank-1 marginal-residual
`‖Π − r c^T/m‖_F / ‖Π‖_F < 0.05` instead (spec §14.4.8, second change-log
entry). The `h`-cosine arm (> 0.95) catches the v2 collapse outright.

**Delegated resolution of the fired stop rule (recorded in §7.1.2):** adopt
**cosine top-50** (no §0 amendment, single §13.12 `n_g` change; ceiling
0.134 vs 0.095 status quo, +41% relative); reject the reranker (measured ≈
cosine at 2.5× k; the §0 conflict is not worth +4.6 pp toward an unreachable
target); re-scope A3/B1/B2 (the grounded-identity chain) from co-primary to
**secondary** — they repair a dead mechanism into a weak-but-alive one and
stay in the plan, but the headline relational repair routes through the
pool-independent channels **B3 (`L_align`) + F.1 (`L_rel`) + D (closure
features)**, all of which P0 leaves fully viable. Paper claims about the
grounding/content identity channel must be scoped to its measured ceiling.

### Fix A — Grounding vocabulary repair (F2, F3) — resolved, no ⚠ remaining

- **A1 Two-stage learned retrieval for pools.** `G(u)` = top-`n_g` of a cosine
  top-M shortlist (M = 200) re-ranked by the **frozen B0-alt scorer**
  (`src/model/b0_alt.py`, MLP over F0 mean-pool features — degree-corrected
  AUPRC 0.7325, comparable to B0 V3.1 at a small fraction of the scoring cost;
  the V3.1 token-sequence scorer would make 8M-pair pool builds a
  `score_universe`-class job and is rejected on cost). Legality: B0-alt
  consumes only frozen features; test-side pools are built within the test
  split's own operative nodes, touching no test-graph edges — the same
  legality class as the existing cosine pool.
  - **⚠ PIN (two registered conflicts, not resolvable here):**
    (i) `docs/03-experiment-protocol.md` §0 pins the frozen pairwise scorer's
    reuse to "B0 and (E4.10 only) as the edge-weight/candidate proposer; the
    s0-anchor role is retired with the rev-3.0 headline" — A1 reinstalls a
    comparator-family scorer as a grounding-candidate proposer inside the
    headline model and requires a §0 component-table amendment;
    (ii) spec §13.12 pins the pool definition to exact top-`n_g` cosine.
  - **Attribution defense (required):** the reranker is a comparator-family
    baseline, so a `cosine_pool` ablation arm (identical model, status-quo
    pools) is registered in the v3 screen to separate "better vocabulary" from
    "B0-family leakage" in the attribution story. The reranker checkpoint
    SHA-256 becomes a pinned data-contract field.
  - **Cache safety (r2 blocker fix; hash widened in r4):** `grounding.py:94-106`
    validates caches only on node set + `n_ground`, so a redefined pool read
    through an existing cache path would silently serve stale cosine pools —
    corrupting `L_gate`, `L_ptr`, and probe labels with no error. The cache
    `.npz` schema gains a pinned `pool_method_hash` field covering **method id
    + reranker checkpoint SHA-256 + M + `n_g` + the ordered F0/source-
    feature-pack digest + the role-universe identity** (r4: the r2 hash
    omitted the feature inputs, so mutated feature contents under unchanged
    node ids and method settings would still silently reuse stale pools —
    both cosine shortlisting and reranking depend on the features, and §13.12
    requires a SHA-bound cache); the loader rejects mismatches. WS1 carries
    two regression tests — stale-method and mutated-features-same-ids. The
    full cache-regeneration set (all four role universes × both strategies in
    use) is enumerated in the spec edit.
- **A2 `n_ground` sweep — RESOLVED (r5).** The registered decision rule
  (smallest `n_g` with pair ceiling ≥ 0.5; stop below 0.3) **fired its stop
  clause**: the measured curve (Phase 0 results) tops out at 0.179 with no
  legal vocabulary above it. Delegated resolution: `n_g = 50` cosine
  (ceiling 0.134), with cosine top-100 measured-and-rejected (0.179 but 2×
  grounding-tensor footprint toward a target that stays unreachable) and the
  identity chain re-scoped to secondary per the Phase 0 results block.
- **A3 Class-rebalanced `L_gate`.** Replace plain BCE with pos-weighted BCE
  (weight = registered constant from P0.2's post-repair in-pool rate) so the
  now-nonzero positive labels are not drowned; gates can cross 0.5 exactly when
  evidence exists.

### Fix B — Supervise the identity chain (F1, F4)

- **B1 `L_ptr` (pointer supervision — the missing gradient).** For each
  Hungarian-matched slot whose target neighbor is in the pool:
  `L_ptr = CE(pointer_k, pool_index(t_k))`; unmatched/out-of-pool slots masked.
  Folded into `L_recon` with `w_ptr = 0.25`. Depends on A for label coverage
  (without pool coverage there are no labels — exactly why this loss could not
  have worked in rev 3.0).
- **B2 Soft differentiable matched flags.** Replace argmax identity matching in
  the content pathway: with per-pair shared-id indicator
  `I[g_a, g_b] = 1[id_a(g_a) = id_b(g_b)]` (n_g × n_g),
  `M[k, l] = (p_a I p_b^T)[k, l]`, and
  `matched_a[k] = gate_a[k] · max_l (M[k, l] · gate_b[l])` (r2 fix: max over
  the *product*, removing the argmax-switching discontinuity of the r1 form;
  the (b,a) direction uses `M^T`, so AB/BA consistency holds by construction).
  Disjoint pools give `I = 0` ⇒ matched ≡ 0 with zero gradient — correct
  behavior, stated explicitly. Train and eval both use the soft form
  (train-mask ≡ eval-bypass equality test re-run). Claim discipline (r2): with
  near-uniform untrained pointers, `M ≈ |pool overlap| / n_g²`, so the
  `L_edge`-through-content gradient to the pointer is *vanishing at init* —
  B2 is a plumbing repair that lets gradient exist; B1 and A3 do the real
  training work.
- **B3 `L_align` (Π identity target — the core relational fix).** On
  edge-stream **positive** pairs (both endpoints train-side with E_msg egos):
  run the per-endpoint Hungarian matching, define teacher cells
  `S = {(k, l) : target_id_a(match(k)) = target_id_b(match(l))}` — slot pairs
  generated from the *same real shared neighbor* — and (r2 fix, replacing the
  r1 globally-normalized form, which had an irreducible ≈ log(#matched) floor
  and pushed gradient into inflating `π·m` against `L_exist`/`L_mult`):
  `L_align = −(1/|S|) Σ_{(k,l)∈S} ½·[ log(Π[k,l] / Σ_{l'} Π[k,l']) +
  log(Π[k,l] / Σ_{k'} Π[k',l]) ]` — row- and column-conditional concentration
  only, orthogonal to the Sinkhorn marginals; pairs with `S = ∅` skipped.
  Folded into `L_recon` with `w_align = 0.5` (frozen only after P0.4).
  - B3 does **not** depend on pools (Hungarian matches slots to true neighbors
    directly), so it repairs the topology pathway's relational channel even
    before A converges. Eval-time Π stays content-cost-only: the loss shapes
    `h` so same-neighbor slots become mutual nearest neighbors — the
    attribution split is intact.
  - **Implementation reality (r2 blocker fix):** the edge stream currently
    builds *no ego targets* — `_edge_tensors` (`train_egostitch.py:2504-2545`)
    assembles only features/grounding/labels; targets exist only in
    `_node_tensors` (:2436-2479). WS2 therefore includes explicit edge-stream
    ego-target assembly for positive pairs: per-endpoint target gathers
    (`(B, K, d)` features + multiplicity + adjacency + mask) and
    memory-manifest accounting mirroring `:2566-2616`. Targets are capped at
    T ≤ K = 16 (`EgoTargetBuilder(slots=…)`, `ego_targets.py:149-171`) — not
    20 as r1 stated. Hungarian at 16×16 is CPU-trivial; positives are 1/6 of
    the edge stream.
  - **Seeding (r4 — r3's pair-keyed RNG was still underspecified):** the
    existing node-stream target RNG includes the DDP rank in its seed
    (`rng((seed, epoch, step, rank, 0x7A))`, `train_egostitch.py:2564`), so
    mirroring it would make a pair's sampled ego targets — and therefore
    `L_align`'s teacher cells, gradients, and potentially checkpoint selection
    — depend on which rank/shard the pair lands on. r3 keyed one RNG by
    canonical pair, which still left endpoint-traversal-order dependence (two
    endpoints consuming one stream) and could break AB/BA equivalence. r4
    pins **per-endpoint node-identity keying**: each endpoint's ego-target
    subsample is drawn from `rng(blake2b(node_id) ⊕ (seed, epoch))` — never
    rank, never step, never pair — so a node's target subset is fixed within
    an epoch regardless of which pairs, ranks, or directions it appears in
    (AB/BA symmetry and cross-pair consistency by construction). `L_rel`
    targets need **no RNG at all**: CN/Jaccard are deterministic functions of
    the pair under `G_fit`. Stream-composition scope (r4 clarification): the
    positive manifest is already built pre-shard (epoch-shuffled globally),
    but `enumerate_edge_stream` draws negatives per `(seed, epoch, rank)`
    **after** sharding (`train_egostitch.py:1722-1758`) — that registered
    stream behavior is unchanged; the invariance contract is **per-pair**
    (any given pair receives identical targets/losses at any world size), not
    stream-composition invariance. Acceptance tests: identical target ids,
    masks, `L_align`/`L_rel` values, and gradients under world sizes 1 vs 2,
    with reversed pair order (BA), unequal tail batches, and padded filler
    rows.

### Fix C — Anti-collapse (F5)

- **C1 Slot diversity term.** `L_div = mean_{k≠l} max(0, cos(h_k, h_l) −
  τ_div)²`, `w_div = 0.1`, in `L_recon`, **restricted to slot pairs where at
  most one slot is Hungarian-matched** (r2 fix: two matched slots may
  legitimately be similar when a node's true neighbors share a community —
  penalizing that fights `L_feat`/B3). `τ_div = 0.5` initial, interaction
  measured in the first dev run. (B1/B3 are themselves the strongest
  anti-collapse forces — identity anchors pull slots apart; C1 is the
  backstop.)
- **C2 Â gradient repair.** Compute `L_slotadj` in logit space
  (`BCEWithLogits` on `adj_logits`) with a registered temperature
  `τ_adj < 1`. This requires exposing `adj_logits` alongside the sigmoid —
  a `SlotSet` extension to the pinned §2 head contract (r2: named explicitly;
  it is an API change in `imagine.py:218-223`, not a losses-only edit).
- **C3 Collapse telemetry + abort rule.** Log P0.1's four dispersion statistics
  every validation; registered death-style rule: mean pairwise `h` cosine
  > 0.95, or Π rank-1 marginal residual `‖Π − r c^T/m‖_F / ‖Π‖_F < 0.05`
  (r6 — the r2 row-entropy arm was measured blind to full collapse: a fully
  collapsed plan reads 0.624 normalized entropy because the `π·m` marginals
  concentrate rows), for 2 consecutive validations after conditioning
  activates ⇒ `training_invalid(slot_collapse)` abort. The 6a inertness of
  the v2 run becomes a *pre-scoring* alarm instead of a post-hoc discovery.

### Fix D — Structure-only closure signal (F6)

r2 correction: r1 proposed three additions, two of which already exist —
`feats[:, :, 6:9]` already carries per-edge-type incident mass including the
alignment row sum `Σ_l Π[k,l]` (`scaffold.py:114`), and STE row-normalization
never touched `feats` (`ste.py:46-47`), so "unnormalized mass channels" was a
no-op. What is genuinely missing is *closure* signal. Retained additions,
functions of `{star, Â, Π}` only:

- **D1 Cross-triangle node feature** (FEAT_DIM 9 → 10): per-slot closed-wedge
  mass `t_k = [Π Â̊_other Π^T]_{kk}` computed on the **zero-diagonal**
  `Â̊ = Â − diag(Â)` (r2 fix: `sigmoid(h·h^T)` has a large diagonal, and
  `Σ_l Π[k,l]²·Â[l,l]` is alignment-concentration — another degree-flavored
  scalar, the exact failure mode under repair; `generated_ego_graph` already
  zeroes the diagonal for the same reason, `losses.py:302-305`).
- **D2 Closure edge type** (EDGE_TYPES 3 → 4), direction pinned (r2 fix — the
  r1 form was direction-ambiguous): the src↔dst closure block is the
  symmetrized wedge
  `C = ½ (Â̊_src Π + Π Â̊_dst)`, with
  `adj[CLOSE, s_src, s_dst] = C` and `adj[CLOSE, s_dst, s_src] = C^T`.
  The degree-feature slice extends to cover the new edge type (6:9 → 6:10).
  The required symmetry test is **rebuild-symmetry** — the scaffold built for
  `(j, i)` (with `Π' = Π^T`) must equal the side-permuted `(i, j)` scaffold —
  not the trivially passing relabel-only test (`swap_direction` only swaps
  anchor channels).
- **6a redefinition under D (r2 fix):** D routes Π/Â-derived signal into
  `feats`, which the current shuffle (adj-only, feats fixed,
  `score_universe.py:122-146`) would bypass — shrinking the control's coverage
  exactly when G3.4 makes inertness binding. The 6a-v3 control is redefined to
  shuffle **at scaffold-build input** (slot-axis permutations of Â_src, Â_dst,
  Π against the fixed per-slot (π, m) features) and then rebuild the scaffold,
  so every structure-derived channel — including `t_k`, `C`, and the 6:10
  degree slice — is recomputed consistently from the shuffled structure. The
  control's meaning ("destroy the binding between slot identity/mass and
  relational structure") is preserved and its perturbation surface again
  covers everything structural.

### Fix E — Conditioning dynamics (F7, F8, F9)

- **E1 Centered gated residual (kills the calibration knob). ⚠ PIN** —
  this edits the §3.4 registered injection equation
  (design doc `2026-07-16…md:117-127`, "Pins (all registered)").
  `cls ← cls + active · tanh(g) · (XAttn(...) − μ)`.
  DDP/masking semantics pinned (r2 fix — r1 left them unstated behind a
  "determinism preserved" claim): μ is the mean over rows that are (a)
  pathway-**active** and (b) real samples (`edge_mask = 1`; padded filler rows
  from `_edge_tensors:2508-2513` excluded), **all-reduced across ranks** so
  training math is independent of the auto-detected world size; the eval-time
  μ is a single synchronized EMA updated post-all-reduce (identical on every
  rank by construction, so checkpoint content does not depend on which rank
  saves). Frozen EMA at eval ⇒ determinism and the checkpoint-exact null
  bypass both hold: inactive rows still receive exact identity. A constant
  offset can no longer buy loss; the pathway must earn per-pair variance.
- **E2 Curriculum repair. ⚠ PIN** — conditioning pathways activate at the
  **first edge-active step**; the rev-3.0 Phase-A `pair_only` head start is
  removed (warm-start remains recon-only with no `L_edge` at all). This edits
  the registered §13.19 curriculum (`train_egostitch.py:748-759`), but note it
  *restores* design §4's own stated intent — "the pairwise trunk and the
  topology pathway enter joint training together" — which the Phase-A
  `pair_only` implementation deviated from. F9's head start let the trunk
  converge before the gates ever opened.
- **E3 Aux/edge family rebalancing — scoped (r2 fix).** The registered anneal
  schedule 1.0 → 0.25 across the edge-active phase applies **only to the
  per-node fidelity components measured in the 500×/1,682× imbalance:
  `{L_feat, L_exist, L_mult, L_deg}`**. The repair terms
  `{L_slotadj, L_gate, L_ptr, L_align, L_div, L_rel}` keep weight 1.0
  throughout — r1's blanket `λ_recon` anneal would have multiplied the core
  relational fixes down 4× exactly when they matter. Implemented as
  per-component schedule factors inside the `L_recon` decomposition (outer
  `λ_recon` untouched). Adaptive gradient surgery is deliberately not proposed
  (hard to register deterministically).
- **E4 Degree-decorrelation telemetry** (F8 watchdog, telemetry-only): per
  validation, correlation of the `full − f_logit` residual with endpoint
  degree; reported in every table. F8's *repair* is E1 + E2 (offset removed,
  head start removed); E4 makes any resurgent degree-flavored pathway visible
  before scoring.

### Fix F — Pair-level relational pressure (F10) ⚠ PIN

- **F.1 `L_rel`:** from the STE's pair state (mean over scaffold tokens of the
  AB-direction output), a 2-layer head predicts train-pair relational targets:
  `log1p(common-neighbor count)` and neighborhood Jaccard, **computed from
  `G_fit` independently for every pair — positive and negative alike** (r3
  fix). r2 assigned negatives "no shared structure" labels, which is
  structurally false: the `NegativeSampler` (`pairs.py:261-268`) rejects only
  canonicalized true positives, so sampled non-edges freely share neighbors —
  and those wedge-bearing non-edges are precisely the hard cases whose
  relational signal the repair needs. Zero-labeling them would train the STE
  to *erase* closure structure on hard negatives and would turn `L_rel` into
  an edge-label shortcut inside an allegedly structural task. Huber loss,
  `w_rel = 0.25`. Train-side, `G_fit`-only ⇒ integrity-legal. Folds into the
  `L_recon` decomposition as relational reconstruction (§2); the head is
  discarded at inference (never part of the scored logit), so the four-logit
  decomposition and null taxonomy are unchanged. Required regression motif: a
  non-edge sharing ≥1 neighbor must receive nonzero relational targets.
- **Reduction contract (r4 — applies to `L_rel` and `L_align` alike):**
  `_edge_tensors` fills rank-local tail batches by duplicating the first row
  and exposes `edge_mask` (`train_egostitch.py:2504-2545`); an ordinary local
  mean would count duplicated targets and weight ranks unequally, making the
  loss and its gradients depend on world size and tail shape — the failure
  class §13.19.1 already forbids for the BCE. Both losses are pinned as
  `edge_mask`-weighted (for `L_align`: positive-and-real-row-weighted) global
  means with an **all-reduced real-row denominator** and matching DDP scaling.
  Acceptance test: per-parameter gradient and loss-value equality at world
  sizes 1 vs 2 with unequal tails and padded filler rows.
- **Why ⚠:** design §4 pins "no auxiliary loss attaches to trunk, STE, or gate
  parameters." The v2 screen is direct evidence this pin failed: `L_edge`
  through a zero-init gate against a self-sufficient trunk taught the STE a
  constant. **Adopted — see §7.1 item 1** — without pair-level pressure, A–E
  still leave the risk that the pathway relearns a variance-carrying degree
  code and produces a second degree-flavored `cut`.
- Scope containment: `L_rel` attaches to the STE only (not trunk, not gates),
  is its own telemetry family, and its removal is a registered ablation arm
  (`no_l_rel`). Shortcut risk (degree-correlated targets) is mitigated by the
  Jaccard target (degree-normalized) and gated by G3.3.

### Fix G — Probe and protocol repair (F11, F12)

- **G1 Π-consistency v2:** plan mass on double-Hungarian same-identity cells
  (`S` from B3) divided by total plan mass — measures *alignment* directly,
  independent of gates/pointers. Data source pinned (r2): for the **formal
  post-run probe artifact**, probe pairs and matching targets derive from the
  train-side `G_struct` (E_msg over operative train nodes), the same
  evaluation-side-only measurement basis the current probe already uses
  (`probes.py:611`); no test-side structure is touched. Pre-binding instances
  of the same probes are scoped by the Phase-0 boundary instead: calibration
  computes them on `G_fit`, the qualification rehearsal on `E_msg[V_qual]`.
  v1 kept alongside for continuity (its honest scope is the grounding chain).
- **G2 New registered probes:** slot recall@`n_g` (per e2e run, not only
  frozen-s0); shared-neighbor-count R² from STE pair states (the direct
  measure of "did a relational representation form"); the P0.1 dispersion
  statistics.
- **G3 Pre-binding qualification gates** (r3 fix — r2 finalized thresholds
  from the same dev run that satisfied them, making a pass post-hoc rather
  than prospective, and ran code tranches before spec edits in violation of
  the freeze rule). Corrected two-stage protocol:
  - **Calibration (V_fit only):** thresholds are derived from P0 plus dev
    training evaluated exclusively on `G_fit` quantities, then **frozen** in
    the draft v3 registration together with the implementation.
  - **Qualification (V_qual, once):** after freezing, a single §13.19.4-class
    rehearsal on the previously untouched `V_qual` universe evaluates the
    frozen gates prospectively (≤3 rehearsal attempts total; a fourth requires
    a new registration version). `V_select` stays sealed until the bound run.
  Frozen gate set (initial values; V_fit-calibrated before freezing):
  1. slot recall@`n_g` ≥ **0.5 × the measured pool ceiling** (r5
     recalibration: the r3 absolute floor of 0.2 was falsified by P0.2 —
     the top-50 pool's own recall ceiling is 0.140, so no model can exceed
     it; the gate now asks whether the model captures at least half of what
     its vocabulary permits, i.e. ≥ 0.070 at `n_g = 50`);
  2. Π-consistency v2 > 0.05 (pool-independent — unaffected by P0.2);
  3. degree-partialled clustering probe R² ≥ 0.10;
  4. **shuffle sensitivity:** the 6a-v3 control must move clustering-MMD by
     more than the evaluator's bootstrap noise floor — an inert control now
     *blocks binding* instead of surfacing post hoc;
  5. matched edge-AUPRC guard passes.
  Fail any at qualification ⇒ no v3 binding; return to Phases A–F (and a new
  registration version if the rehearsal budget is exhausted).
- **G4 Arm schema (r3 fix — the r2 seven-arm list omitted the `no_l_rel`
  ablation F.1 itself promised, and the gate fails closed on the pinned
  five-arm schema `_E2E_ARMS`/`_E2E_FORMAL_ARMS`, `g5_stage1.py:1205-1217`).**
  The v3 screen is an explicit **eight-arm schema**:
  - **Six trained checkpoints:** `full`, `b0_e2e_f_only`, `pair_topology`,
    `p0`, `cosine_pool` (A1 attribution defense: identical model, status-quo
    cosine pools), `no_l_rel` (F.1 attribution defense: identical model,
    `w_rel = 0`).
  - **Two scoring-time controls over `full`'s checkpoint:** `6a-v3`
    (rebuild-form shuffle, §D) and `6e` (degree-preserving rewiring — the
    confirmed failure mode, "degree code disguised as a graph", is exactly
    what 6e isolates; 6a alone cannot distinguish it from collapse).
  Each arm's checkpoint provenance, scoring semantics (trained vs
  scoring-control), execution order, and per-arm config hash are pinned in the
  v3 registration. This supersedes §4's v2 five-arm shape, and the §7 budget
  is re-derived for it.
  - **Full provenance migration (r4 — arm enforcement is not one tuple):**
    formal-arm validation is duplicated across the stack — the worker's
    binding-evidence/config checks (`train_egostitch.py:1227, 1315-1388`),
    the scoring CLI's arm/provenance/permanent-null enforcement
    (`score_universe.py:98, 991-1110, 2155-2439`), and the gate tuples
    (`g5_stage1.py:1205-1217`) — and each rejects unknown arms. WS7/WS8
    therefore migrate **every** formal-arm constant, binding-evidence
    validator, scoring CLI input and provenance enum, run-metadata schema
    field, and test fixture to the eight-arm schema. Required end-to-end
    tests: a complete six-trained-plus-two-control package is accepted, and a
    v2 five-arm package is rejected, at every enforcement point.
  - **6e algorithm pin (r4 — a control without a deterministic algorithm is
    not registrable):** `6e-v1` = canonical-pair-keyed **checkerboard swaps**
    on the scaffold inputs before rebuild: for `N_swap = 8·K²` keyed draws,
    select a row pair and column pair, transfer
    `δ = u·min(w_il, w_kj)` (keyed `u ∈ (0,1)`) across the 2×2 checkerboard
    (`w_ij, w_kl += δ`; `w_il, w_kj −= δ`), applied to `Â̊_src`, `Â̊_dst`
    (symmetrized) and `Π`. Checkerboard swaps preserve every row and column
    sum exactly — per-node, per-edge-type soft degree and total mass are
    invariant — while destroying higher-order connectivity, which is
    precisely 6e's registered purpose (isolating structure beyond degree).
    Keying mirrors 6a's blake2b canonical-pair scheme; scaffold rebuild then
    recomputes all derived channels, exactly as in 6a-v3. Acceptance tests:
    row/column-sum preservation to fp32 tolerance; determinism across
    processes; the control measurably moves STE output on a random
    non-collapsed model.

## 4. What is deliberately kept

The from-scratch trunk (f-only ≈ B0 AUPRC 0.7289, beats B0 BFS-macro GS),
branch dropout at p = 0.15 (the p0 arm regresses on every MMD axis), the
liveness/death rule, the fp32 pair-pass pins (§13.16/§14), the four-logit
decomposition, and the screen's gate structure. All performed correctly in v2.
(The v2 *five-arm shape* is superseded by G4's eight-arm schema.)

## 5. Implementation plan (workstreams, tests, touch list)

| WS | Touches | Tests (acceptance) |
|---|---|---|
| 1. Grounding (A) | `src/data/grounding.py` (two-stage rerank + `pool_method_hash` cache schema), pool caches ×4 role universes, config | pool determinism; own-split-side legality; **stale-method and mutated-features-same-ids rejection** (r2 blocker fix, hash widened in r4); P0.2 harness |
| 2. Losses (B1, B3, C1, C2, F.1) | `losses.py`, `config.py`, `model.py`, `imagine.py` (`SlotSet` + `adj_logits` exposure), `train_egostitch.py` (**edge-stream ego-target assembly**: node-identity-keyed target subsampling per the B3 r4 seeding pin — never rank-seeded — plus memory-manifest accounting; the r2-identified missing machinery) | teacher-cell correctness on a hand-built graph; `L_align` invariance under AB/BA; per-pair world-size invariance of targets, losses, and gradients (1 vs 2 ranks, reversed pairs, unequal tails, padded rows); masked all-reduced reduction contract for `L_rel`/`L_align`; loss-family telemetry names |
| 3. Soft matching (B2) | `scaffold.py`, `e2e_model.py` | `p(i,j)=p(j,i)` under every null; train-mask ≡ eval-bypass re-run; gradient reaches `head_pointer` (explicit grad-flow test — the F1 regression test) |
| 4. Scaffold/STE (D) | `scaffold.py` (FEAT_DIM 9→10, EDGE_TYPES 3→4, zero-diag closure), `ste.py`, `score_universe.py` (6a-v3 rebuild-form control) | shape contract fail-closed; **rebuild-symmetry** (not relabel-symmetry); 6a-v3 moves STE output on a random non-collapsed model (F12 regression test) |
| 5. Conditioning (E1) | `conditioning.py` (masked, all-reduced μ; synchronized EMA) | checkpoint-exact bypass with EMA-μ; determinism across two eval runs; world-size-invariance of μ (1 vs 2 ranks on CPU) |
| 6. Trainer (E2, E3, C3, E4) | `train_egostitch.py` | phase-state unit tests; collapse-abort fires on a synthetic collapsed run; per-component anneal factors correct |
| 7. Probes/gate (G) | `probes.py`, `g5_stage1.py` (incl. replacing the pinned `_E2E_ARMS`/`_E2E_FORMAL_ARMS` five-arm tuples with the G4 eight-arm schema) | v2-consistency on hand-built alignment; gate parses new fields and **fails closed on the exact eight-arm set** (five-arm v2 inputs rejected); **probe/scores artifact schema-version bumps** (`egostitch_e2e_probe_v2`, scores-npz meta version) with old-version rejection |
| 8. Docs | spec §14 edits + change-log (incl. §13.12 pool definition, §13.19 curriculum, §3.4 injection equation, §0 component-table amendment via protocol edit), v3 registration draft | — |

Sequencing (r3 fix — r2 ran code before spec edits and calibrated G3 from the
run it gated; the freeze rule is spec-first, and qualification must be
prospective):

1. **P0 audits** (`G_fit`-scoped; ≈1 day; P0.2's reranked variants need the
   B0-alt batch-scoring harness — itemized, not assumed free).
2. **Owner decisions** (§7) resolved.
3. **Spec/protocol amendments** land with change-log lines (§14, §13.12,
   §13.19, §3.4-equivalent, protocol §0 component table). Per the freeze rule
   this *authorizes implementation, not execution*.
4. **Implementation** — WS tranche 1 (B3 including its edge-stream target
   machinery, C, E) then tranche 2 (A, B1, B2, D), with the WS test suites.
5. **Dev calibration on V_fit only** (~2.5 h per training arm on 4×H20);
   G3 thresholds derived and **frozen** into the draft v3 registration.
6. **Single V_qual qualification rehearsal** against the frozen gates
   (§13.19.4, ≤3 attempts).
7. **v3 registration → BINDING** → formal eight-arm screen. Full candidate
   scoring is spent **only** after qualification passes.

## 6. Expected mapping to the registered failure labels

| v2 label | Fixes that target it | Post-fix expectation |
|---|---|---|
| Primary: clustering-MMD, BFS-macro RD | B3, D, F.1 (+A/B1 via content pathway) | the first mechanism that can *represent* shared neighbors; success = beating `b0_cal_selfdensity` 10.20 clustering-MMD, which remains the bar |
| Guard: matched AUPRC | E1, E2 (E4 watchdog) | pathway carries variance or nothing; offset trick removed |
| Attribution `G_full ≤ 0` | B3 + F.1 (topology pathway gains its own signal) | `G_full > 0` with `G_pt ≥ 0.25·G_full` re-tested under the same rule |
| Structure control inert | C (collapse prevention), 6a-v3 + G3.4 (inertness blocks binding), G4/6e | control informative again; degree-code vs collapse now distinguishable |
| Π-consistency = 0 | A, B1, B2 (grounding chain), B3 + G1 (alignment measured directly) | nonzero by construction if learning occurs; still-zero after repair is *evidence*, not artifact |

## 7. Decisions (recorded 2026-07-25 under owner delegation)

**Provenance.** The owner delegated these final decisions to the working agent
("make final decisions by yourself", 2026-07-25 session), after three review
rounds (internal verification pass, 20-defect internal adversarial review,
5-finding Codex adversarial review — all resolved). The second Codex re-review
of r3 completed after these decisions were recorded and returned 4 findings
(3 high / 1 medium), all specification-tightening; they were reconciled as r4
(revision history) and **no decision below was overturned**.

**Disposition (delegated):** the rev-3.0 e2e build line **proceeds as the
rev-3.1 repair** specified here, rather than being cut to the E2E B3-full
Ockham line. The §8 kill criterion stands: a G3-qualified rev-3.1 screen that
still fails the primary clustering criterion is evidence against the
conditioned-encoder headline itself, and B3-full becomes the priority
comparison at that point.

### 7.1 Registered-pin inventory — five items, all DECIDED

1. ⚠ **F.1** — `L_rel` on the STE vs design §4 "no aux loss on trunk/STE/gate
   parameters". **DECIDED: adopt.** The v2 screen empirically falsified the
   pin (`L_edge` through a zero-init gate against a self-sufficient trunk
   taught the STE a constant); without pair-level pressure the dominant
   residual risk is a second degree-flavored `cut`. Contained: STE-only
   attachment, own telemetry family, formal `no_l_rel` ablation arm —
   anti-grab-bag compliant (owns the clustering/triangle-closure failure axis
   and an ablation arm).
2. ⚠ **A1** — comparator-family reranker for grounding pools vs protocol §0's
   pinned B0 role ("B0 and (E4.10 only)…") **and** spec §13.12's exact-cosine
   pool pin. **DECIDED: adopt conditionally, plain cosine preferred.**
   Selection order pinned before P0 runs: (a) if plain cosine top-{20, 50}
   meets the A2 pair-ceiling target on `G_fit`, use it — no reranker, the §0
   amendment is moot, and only the §13.12 `n_g` value changes; (b) otherwise
   land the B0-alt rerank with the §0 component-table amendment, the pinned
   reranker hash, and the `cosine_pool` ablation arm; (c) if no variant
   reaches a 0.3 pair ceiling, stop before tranche 2 — the vocabulary needs a
   redesign, which reopens this decision. The eight-arm schema is identical in
   (a) and (b) (`cosine_pool` = status-quo top-20 cosine pools in both
   branches, serving as the vocabulary-attribution arm).
   **RESOLVED (r5, measured):** clause (c) fired — no variant reached 0.3
   (max 0.179; Phase 0 results). Delegated resolution of the reopened
   decision: **cosine top-50, no reranker** — the reranker measured ≈ cosine
   at 2.5× pool size, so the §0 amendment buys +4.6 pp toward an unreachable
   target and is dropped; the protocol §0 conflict is therefore **moot** and
   A1's remaining spec surface is the §13.12 `n_g`/cache-schema edit only.
   The vocabulary is not redesigned further: P0.2 shows the ceiling belongs
   to feature retrieval itself (no legal alternative exists at test time),
   so the identity chain is re-scoped to secondary and the primary relational
   repair routes through the pool-independent channels (B3 + F.1 + D). The
   `cosine_pool` (top-20) arm keeps its role as the vocabulary-attribution
   arm. Active ⚠ pin count drops from five to **four** (items 1, 3, 4 below
   plus the §13.12 pool-definition edit, now conflict-free).
3. ⚠ **E1** — centered residual vs the §3.4 registered injection equation.
   **DECIDED: adopt**, with the r2-pinned masked/all-reduced μ + synchronized
   EMA semantics. The `+0.144 ± 0.085` constant-offset telemetry is direct
   evidence the uncentered form funds a calibration shortcut.
4. ⚠ **E2** — removal of Phase-A `pair_only` vs the registered §13.19
   curriculum. **DECIDED: adopt.** It restores design §4's stated joint-entry
   intent, which the Phase-A head start deviated from; the v2 screen showed
   the head start let the trunk converge before the gates opened.
5. **New-loss folding** — `{L_ptr, L_align, L_div, L_rel}` into the registered
   `L_recon` decomposition + per-component E3 anneal factors (outer locked
   objective unchanged). **DECIDED: adopt** via spec change-log.

### 7.2 Other approvals — DECIDED

6. Scaffold format change (FEAT_DIM 9→10, EDGE_TYPES 3→4) — §14.1 edit; plus
   probe/scores artifact schema-version bumps. **DECIDED: adopt.**
7. Budget (r3 re-costing for the eight-arm schema): P0 ≈ 1 GPU-day incl. the
   B0-alt pool-scoring harness; dev calibration ≈ 1–2 × (up to 6 × 2.5 h)
   training + V_fit-side evaluation only; one V_qual rehearsal within the
   registered rehearsal envelope; formal v3 screen: **6 trained arms ≈ 15 h**
   training (v2 per-arm envelope ≈2.5 h × 6), and scoring re-derived from v2
   measurements — v2's five passes cost ≈24.5 h with 6a's keyed-permutation
   pass ≈4× a plain pass (≈3 h plain / ≈12 h perturbed); v3 has **6 plain +
   2 perturbed passes (6a-v3, 6e) ≈ 42 h** scoring on 4×H20, stated here so
   the registration's cost report is derivation-based, not aspirational.
   **DECIDED: run the full eight-arm screen.** The trimmed fallback (demoting
   `cosine_pool` and `no_l_rel` to dev-only, ≈36 h scoring / ≈10 h training)
   is **rejected**: the attribution defenses are the point of the repair, and
   under the anti-grab-bag rule F.1 without its formal ablation arm could not
   keep its mechanism row in the paper story.

## 8. Risks and kill criteria

- **Pool ceiling too low (A) — MEASURED, FIRED, RESOLVED (r5):** every
  variant landed below the 0.3 stop floor (max 0.179). Resolution recorded in
  §7.1.2 and the Phase 0 results block: cosine top-50, identity chain
  re-scoped to secondary, primary repair through pool-independent channels.
  Residual risk: paper claims about the grounding/content identity channel
  must carry its measured ceiling (~13% of positives at top-50) or a
  reviewer will correctly call the mechanism decorative.
- **Teacher starvation (B3) — MEASURED, DID NOT FIRE (r5):** P0.4 measured
  `P(|S|>0) = 0.663`, `E[|S|] = 2.12`, thinnest stratum 0.507 (and it is the
  *low*-degree quartile — the r2 hypothesis was inverted). K-capped matching
  retained; the uncapped fallback is unnecessary. F.1 stays adopted on its
  own merits (§7.1.1), not as a starvation fallback.
- **F.1 shortcutting:** `L_rel` satisfiable by degree if targets correlate with
  degree — mitigated by the Jaccard target and gated by G3.3.
- **C1/L_feat interaction:** diversity vs faithful same-community slots —
  bounded by the matched-pair exclusion; measured in the first dev run.
- **Threshold transfer V_fit → V_qual (r3):** gates calibrated on `G_fit` may
  miss on the disjoint `V_qual` topology; the rehearsal budget is **≤3
  attempts total** (§13.19.4), and exhausting it forces a new registration
  version. Mitigation: calibrate with margin (freeze thresholds below the
  observed V_fit values, not at them) and treat the first rehearsal as the
  transfer measurement.
- **Still `cut` after repair:** if a G3-passing rev-3.1 run still fails the
  primary clustering criterion, that is honest evidence against the
  conditioned-encoder headline itself (not its plumbing), and the E2E B3-full
  Ockham arm (design §5.3) becomes the priority comparison.
- The rev-3.1 screen decides nothing about disposition; it feeds the same
  owner-side locked-decision process as v2.
