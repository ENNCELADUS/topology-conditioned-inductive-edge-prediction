# EgoStitch E2E rev-3.2: Slot-Collapse Fix Design (D0 feature-geometry repair)

**Status: PROPOSED (2026-07-27). Owner decision required.** Diagnosis and
fix design for the 2026-07-27 `training_invalid(slot_collapse)` V_fit
calibration failure (attempt-001, rev-3.1). This document proposes; it
does not decide. Per the spec freeze rule, every code-facing item lands
only through `docs/05-egostitch-spec.md` edits with §12 change-log
lines, then implementation, then a fresh v3.x BINDING registration.

Review trail (all same-day): r1 drafted from CPU probes of the model
code at iid-random init (Sinkhorn-scale mechanism primary). r2:
independent adversarial design review (2 blockers, 6 major) hardened the
proposal. r3: independent adversarial diagnosis review re-ran the probes
with **real F0 features and real cosine-top-50 pools** and overturned
the causal ordering — the feature-anisotropy floor, not the OT scale, is
primary. r4 (this text): Codex adversarial review (gpt-5.6-sol, high)
returned 5 blockers / 5 major against the r3 draft, refuting the
guard-replacement arm (an aggregation error in our own argument), the
slot-ID embedding arm (universal-template escape), and the unit-sphere
arm as drafted (incompatible normalization sites; wrong affine-Gram
rescale) — and demonstrated that the minimal intervention (D0 alone)
removes the measured floor. The final design is **D0-only**, with the
r1–r3 mechanisms recorded as a deferred, evidence-triggered ledger.

---

## 1. What happened

First rev-3.1 V_fit calibration past the OOM (see the memory record of
commit `0ec5551`) trained 15.3 min of a projected ~83 (overfit T=2000,
30 epochs, ~66 steps/epoch, B_n=128, 4×H20) and raised
`RuntimeError: training_invalid(slot_collapse)` on all ranks
(`E2ESlotCollapseGuard`, spec §14.4.8: `h_pairwise_cosine_mean > 0.95`
OR `plan_rank1_marginal_residual < 0.05` for 2 consecutive
conditioning-active validations). `phase_a_end = 400` falls inside
epoch 6, so epoch 6 is the first conditioning-active validation and
epoch 7 the earliest possible trip; 15.3 min ≈ 7 epochs at the observed
pace. The guard tripped at (or within one epoch of) the earliest
arithmetically possible validation. Which clause fired is unrecoverable
for this run (the telemetry line landed one commit later), but the
measured init state (§2) makes the cosine clause the presumptive
trigger: **it is above threshold at initialization**.

## 2. Diagnosis: verified contributors (three adversarial rounds)

Probes: (a) iid-random inputs through `ImagineDecoder` +
`sinkhorn_log_plan` + `_e2e_dispersion_rows`; (b) real F0 features
(`frozen_node_features_1024`, input_dim 1536) and real cosine-top-50
pools through `EgoStitchStage1(standardize_features=True).encode_nodes`
(300-node sample, conservative — true top-50-of-universe pools are more
similar); (c) Codex re-derivations on 100–300-node real-F0 probes.
Config: K=16, d_p=256, eps=0.1, tau=1.0, n_g=50.

### R1 (primary, measured): an anisotropy floor the architecture never removes — the run is born above the guard threshold

- Raw F0 rows: mean pairwise cosine **0.949** (p95 0.981). The applied
  standardization is per-row `LayerNorm(elementwise_affine=False)`
  (`model.py:93-97`), which removes each row's own mean/variance but
  **cannot remove the shared cross-row mean direction**. After
  LayerNorm+proj, the L_feat target projections still have mean
  pairwise cosine **0.941**.
- All 16 slot queries are `W_q[proj(x_{g_k}); e_u]` over the
  top-16-by-cosine pool candidates (`n_dynamic = min(K, n_g) = 16` —
  the learned per-slot `q_base` branch at `imagine.py:139-145` is dead
  code at every n_g ever configured); every query shares `e_u`; the
  decoder is permutation-equivariant. Measured at random init with real
  data: **h_pairwise_cosine = 0.9897 > 0.95**, with a *live* plan
  (mass 0.069, rank-1 residual 0.87). The h clause of the guard is
  primed at step 0; a healthy-but-untrained model reads "collapsed".
- Correction recorded from r4: the healthy *target* operating point
  (0.941) is **below** the 0.95 threshold, and the guard aggregates the
  batch mean — so the existing guard is passable once the init floor is
  removed. The r3 claim "unpassable by construction" was an aggregation
  error and is withdrawn. The guard is retained unchanged.

### R2 (contributing): weak separating forces over a cone of targets

- `L_feat` (w 1.0, the largest recon term) pulls all slots toward
  targets that lie in a 0.94-cosine cone — a pull *into* the cone.
- `L_div` (w 0.1, hinge²@τ=0.5) excludes matched-matched pairs
  (`losses.py:311-322`) and is **exactly zero for any node with ≥ K
  matched targets** — high-degree nodes get no diversity pressure at
  all. `L_ptr` covers ~13% of matched slots (measured pool ceiling).
- v2's 0.9997 endpoint is explained by the same start plus **no**
  diversity force at all (L_div/L_ptr/L_align are rev-3.1 additions;
  v2 Phase A was `pair_only` with hard-bypassed conditioning — no plan
  gradient in Phase A). Two v2 endpoint details remain attributed
  elsewhere or open: the exactly-uniform pointer is the separately
  verified dead pointer/gate chain (2026-07-25 diagnosis), and Â
  settling at 0.5014 against a true neighbor-neighbor adjacency rate
  of ~0.22–0.28 is *consistent with* a flat bias-dominated Gram but
  not fully derived. The chain is a set of supported contributors, not
  a closed-form proof.

### R3 (latent hazard, real but not what tripped this run): the Sinkhorn stage has two degenerate regimes and no scale control

Function-level (verified twice): with C = (5/4)‖h_i−h_j‖² + (5/16)|Δπ|
on unnormalized h, eps=0.1, tau=1.0, the plan is numerically dead at
dist² ≫ eps (iid init: total mass ~1e-26, residual reads exactly 0 for
*diverse* slots) and diffuse-rank-1 at dist² ≪ eps; the healthy band is
narrow and unpinned. Real-data init happens to sit in-band, so this is
not the trigger; but the v2 collapsed endpoint sits in the diffuse
regime, so the eps-band is a plausible terminal attractor for late
contraction. `L_align` is row/column-conditional and therefore
mass-invariant; with real (near-argmin) teacher cells its init
magnitude is small (~4 nats), not the 215-nat random-teacher figure —
its saturation is a hazard only at out-of-band scales. Monitored via
telemetry (§3.4); acted on only if measured (§5).

## 3. Fix design: D0 — remove the anisotropy at the source (the only rev-3.2 model change)

### 3.1 The change

Replace the per-row LayerNorm feature standardization inside the
generator with registered **per-dimension z-scoring**:
`x̃ = (x − μ)/σ`, applied everywhere the generator currently applies
`standardize_features` (anchor features, pool candidate features,
target features — one transformation for all consumers).

Measured effect (deterministic 300-node real-F0 probe, r4):

| Geometry                    | Target cosine | Init slot cosine | Init ‖h‖ |
|-----------------------------|--------------:|-----------------:|---------:|
| Current per-row LayerNorm   |        0.9405 |           0.9897 |     9.12 |
| Per-dim z-score, raw pools  |       −0.0013 |           0.6204 |     9.29 |

One scientific change removes the measured floor (init cosine drops
0.9897 → 0.62, far below the 0.95 trip), preserves activation scale
(‖x̃‖ ≈ √1536; ‖h‖ ≈ 9.3 vs 9.1), leaves Tokenize-lite/`w_x` in-regime,
and leaves the raw-token pair encoder — hence B0-e2e comparability —
untouched. The existing guard, probe schema, and artifact formats are
all retained; attribution of the next run's outcome is clean.

### 3.2 The μ/σ statistics contract (r4 finding 5 — required, not optional)

- **Scope**: computed over the **ordered V_fit universe only** (the
  loaded feature matrix in rehearsal/formal assembly contains sealed
  universes; moments must be restricted to `fit_rows` — computing over
  the full matrix would leak V_qual/V_select).
- **Provenance**: bind ordered V_fit identity + source-feature-pack
  digest; fp64 accumulation; pinned variance estimator and per-dim
  variance floor; μ/σ digest recorded.
- **Storage**: μ/σ are checkpoint buffers (scoring constructs the model
  from checkpoint state before loading the scored universe's F0
  matrix — runtime recomputation is not an option) and registered
  constants; new config fields (standardization mode + stats digest) so
  `config_hash` binds the preprocessing identity. Loaders fail closed
  on any mismatch.
- The same frozen constants apply in every universe (a train-side
  statistic; no universe-conditional preprocessing).

### 3.3 Pinned non-changes (r4 findings 6, 7, 9)

- **Grounding pools stay raw-F0 cosine.** Retrieval in z-scored space
  would silently invalidate every pool cache, the P0 ceilings, and the
  recalibrated G3 gate 1 (measured raw-vs-z-scored top-50 overlap:
  0.653 mean, min 0.36). The retrieval/representation mismatch is
  accepted and recorded; `pool_method_hash` and caches are untouched.
- **SSL noise moves to standardized coordinates.** Today's noise is
  added in raw F0 space before standardization; under z-scoring a raw
  σ_noise=0.05 becomes ~5e-5…1.9e-3 per standardized coordinate
  (feature σ ranges 26.1–1023.3) — the augmentation would silently
  vanish. Sample noise in standardized space (or scale by registered
  per-dim σ); this is part of the D0 spec edit, not a loss-weight knob.
- **Guard, probe ABI, arm schema unchanged.** §14.4.8 criteria,
  thresholds, arming, `egostitch_e2e_probe_v2`, and the v3
  registration's named arrays are not touched. New diagnostics (plan
  total mass, plan max-cell fraction, pre-activation h-norm, pairwise
  distance stats) go to the **training-log telemetry line only** —
  never into the probe artifact schema.

### 3.4 Verification before any GPU time

- Committed pinned-seed regression test: under D0, random-init
  `h_pairwise_cosine_mean` < 0.9 (margin below the trip), plan mass ≥
  floor, rank-1 residual ≥ 0.3; target-projection cosine spread
  recorded.
- Read-only init probe **on the actual guard population** (V_fit
  validation pairs through `_validate_epoch`'s dispersion path), not a
  synthetic sample, before launching calibration.

## 4. Deferred ledger (evidence-triggered; NOT part of rev-3.2)

Each item below was part of earlier drafts, was refuted-as-drafted or
made unnecessary by review, and is retained only as a named next move
with its trigger and its recorded defect:

- **L-1: L_div matched-matched hole.** For nodes with ≥ K matched
  targets, L_div ≡ 0. Trigger: post-D0 trajectories show dispersion
  decaying fastest on high-degree nodes. Candidate: per-node coverage
  floor or reduced-weight inclusion of matched-matched pairs.
- **L-2: unit-sphere slot geometry + bounded OT cost (r1 "D1").**
  Defects recorded at r4: normalization sites as drafted are mutually
  incompatible (matching/recon are per-node, before the pair island;
  `SlotSet` carries one h — needs explicit h_raw/h_unit semantics), and
  the `exp(s)`·Gram adjacency rescale is wrong for the affine
  `head_adj` (bias does not scale; measured logit error 1.31, sigmoid
  std 0.0072 → 0.0022). Trigger: telemetry shows plan mass/residual
  drifting out of band during otherwise-healthy training. Any revival
  must also resolve §13.7 ("raw proj(x) semantics remain intact for
  Stitch/matching…"), the Hungarian both-sides normalization, w_feat /
  L_ssl / denoise rescales, and proj weight-decay anchoring.
- **L-3: persistent slot-ID embeddings (r1 "D2").** Defect recorded at
  r4: a universal regular-simplex template (identical slots for every
  node) reaches ρ=1.0 and residual 0.97 — passing every geometric
  criterion with zero node-conditioning; global q_id embeddings
  *encourage* that solution, and calibrating their scale "so init
  passes" is a tautological gate. Trigger: post-D0 evidence that query
  noise-diversity is insufficient. Prerequisite: an across-node
  conditionality statistic (e.g., matched-slot-set variation across
  nodes, or sensitivity to grounding-candidate shuffle) added to
  telemetry *first*, so template collapse is visible.
- **L-4: guard-criterion replacement (r3 "D4").** Withdrawn: premised
  on an aggregation error; for unit vectors the proposed dispersion
  ratio is a reparameterization of mean pairwise cosine. The r6
  precedent (replace only criteria *measured blind on the collapsed
  checkpoint*) cuts against replacement — the cosine arm caught v2.
  Revisit only if a healthy post-D0 run measurably re-floors the
  cosine above 0.95 through decoder-side effects (e.g., head biases).
- **L-5: structural conditioning of the generator (r1 "D5").** The
  capacity fix for the clustering gate; blocked on a
  **non-spec-editable** conflict: blueprint §10.6 (no target-graph
  access at test time) — test-universe pool-internal adjacency is
  target-side structure. Any design must route around §10.6 (e.g.,
  G_fit-only structural profiles of fit-universe candidates). Trigger:
  rev-3.2 passes the guard but fails §14.4.7 gate (3)/(4).

## 5. Spec pins touched by D0 (flag, don't resolve)

- The §13 standardization sentence (per-row LayerNorm → registered
  per-dimension z-scoring with the §3.2 statistics contract) + §12
  change-log entry.
- The SSL noise contract sentence (noise in standardized coordinates).
- New registered constants/config fields: standardization mode, μ/σ
  digest (+ checkpoint buffers). `config_hash` changes as a
  consequence — **rev-3.1 calibration artifacts become inadmissible
  for rev-3.2 threshold freezing** (correct; state it in the v3.x
  draft — both revisions share `egostitch_e2e_probe_v2`, so
  registration-SHA binding is the discriminator).
- Blueprint §10 / protocol §0: no conflict (checked r2–r4). Pool
  contract §13.12 untouched (raw-F0 retrieval pinned).
- §14.4.8 untouched.

## 6. Sequencing and open blockers

spec edits (§12) → D0 implementation + §3.4 tests/probe → V_fit
calibration under the **existing** guard → read trajectory telemetry →
freeze thresholds → v3.x BINDING → single V_qual rehearsal.

- The **v3 pre-binding gate circularity** (owner decision deferred
  2026-07-27: G3.4/G3.5 need scoring, scoring needs BINDING, BINDING
  needs those thresholds) is inherited unchanged and must be resolved
  by the owner before the freeze step. D0 adds **no** jointly-searched
  constants (μ/σ are data statistics, not tuned knobs), so it does not
  widen the circularity the way the r3 draft would have.
- Calibration attempts: window 001–003, `max_attempts_total: 3`;
  V_fit calibration burns no `v_qual_rehearsals`.

## 7. Expected outcomes and kill criterion

- Expected: init reads healthy on the true guard population; the guard
  becomes a genuine *dynamics* guard. The scientific question moves to
  whether training keeps slots dispersed and the §14.4.7 gates pass.
  Residual gate risk ranking (r2): clustering R² gate (3) > Π-
  consistency (2) > 6a movement (4) > slot recall (1) — gate (3) is
  capacity-limited (L-5's territory), not geometry-limited.
- Kill criterion: if under D0 the cosine climbs back through 0.95
  across consecutive validations with a live plan, that is the first
  *true* measurement of collapse dynamics in this program — escalate
  to the deferred ledger item the trajectory implicates (L-1 for
  degree-stratified decay, L-3-with-prerequisite for query-noise
  insufficiency, L-2 for scale drift), one item at a time.

## 8. Experiment suite (unchanged protocol, new registration version)

**D0 changes preprocessing, not the experiment design.** The eight-arm
v3 screen, the five G3 gates, the two-stage qualification, and the
`calibrate → rehearse → formal` harness all carry over verbatim. What
must change is the registration *version*, because `config_hash` and
the μ/σ constants move.

### S0 — local, no GPU (hours)

New tests, all committed before any spec edit lands in code:
D0 transform correctness; **μ/σ leakage test** (statistics identical
whether or not sealed V_qual/V_select rows are present in the loaded
matrix); fail-closed on stats-digest / feature-pack mismatch;
checkpoint-buffer roundtrip through save→load→score; SSL noise sampled
in standardized coordinates; pinned-seed init-health test (cos < 0.9,
plan mass ≥ floor, rank-1 residual ≥ 0.3). Re-run the existing required
suite (world-size 1v2 reduction equality, rebuild-symmetry, 6a/6e
non-inertness) — 6a/6e non-inertness is now genuinely satisfiable,
since a random rev-3.2 model has dispersed slots and a live plan.
Then the **read-only init probe on the actual guard population**
(V_fit validation pairs through the `_validate_epoch` dispersion path),
not a synthetic sample. Gate: if init cosine on the true population is
not comfortably below 0.95, stop — D0 has not done its job and no GPU
time is justified.

### S1 — V_fit calibration (`hpc/qualification.sh calibrate`, ~1.5 h train)

Single `full` arm: sanity → registered 2,000-step overfit → probes →
gates on V_fit only. Never opens V_qual; burns no `v_qual_rehearsals`.
Produces exactly what is missing today: the **G3.4** evaluator
bootstrap noise floor and the **G3.5** matched edge-AUPRC guard (the
two `REQUIRED-BEFORE-BINDING` placeholders), plus the collapse
telemetry trajectory that tells us whether the guard is now measuring
dynamics instead of an init floor. G3.1 (0.0698) needs no
recalibration: it is 0.5 × the measured raw-F0 top-50 pool ceiling
(0.1395), and D0 pins raw-F0 retrieval — the ceiling is untouched.

### S2 — freeze + registration v3.1 (owner action)

Replace the two placeholder thresholds with measured values; add the
μ/σ digest, the new implementation SHA, and six regenerated v3.1
config digests. **This is where the recorded pre-binding gate
circularity must be resolved by the owner** — G3.4/G3.5 are the
circular pair. D0 does not widen the circularity (μ/σ are data
statistics, not tuned knobs), but it does not resolve it either.

Attempt budget: a new registration version resets the ≤3-attempt
window (`fourth_attempt: requires a new registration version`), so the
failed rev-3.1 attempt-001 does not consume rev-3.2's budget.

### S3 — single V_qual rehearsal (`rehearse`, ~1.5 h train)

Prospective evaluation of the frozen gates on the previously untouched
V_qual universe. Refuses to start while any threshold is unfrozen.
Spends one of the three attempts. V_select stays sealed.

### S4 — the formal screen (`formal <arm>`, scoring ≈ 42 h)

Six trained arms — `full`, `b0_e2e_f_only`, `pair_topology`, `p0`,
`cosine_pool`, `no_l_rel` — plus the two scoring-time controls
(`structure_control_6a_v3`, `structure_control_6e_v1`) reusing
`full`'s checkpoint. `full` runs first with the eligibility/liveness
preflight before the rest launch. This produces the G5 Stage-1 verdict.

### Claim discipline

Unchanged and non-negotiable: this remains a fixed-Seed-0 engineering
screen. No significance, no cross-seed robustness, p-values/CIs/Holm
stay `null`. Edge-level and assembled-graph metrics are reported
together; the three MMD ratios are never aggregated.

## 9. Review provenance

- r2 design attack (general agent): spec-edit completeness (§13.7),
  Hungarian cost, adj-Gram scale, w_feat arithmetic, eps registration,
  proj decay, q_id scale, OR-retention, gate-risk ranking,
  registration circularity; verified no-ops (membership/s1 normalize
  internally; pointer/pool caches unaffected by query-seed choice).
- r3 diagnosis attack (general agent, real-data probes): F0 anisotropy
  0.949/0.941; init h-cos 0.9897 with live plan (mass 0.069, residual
  0.87); L_align mass-invariance and near-argmin teacher cells; v2
  pair_only Phase A; OT mechanism demoted to latent hazard.
- r4 Codex attack (gpt-5.6-sol high, clean CODEX_HOME): D4 aggregation
  error + ρ reparameterization; D2 universal-template escape
  (ρ=1.0/residual 0.97 construction); D1 incompatible normalization
  sites + affine-Gram rescale error (measured); D0 statistics-artifact
  contract, raw-pool pinning (overlap 0.653), SSL-noise vanishing
  (σ 26–1023); probe-ABI protection; **D0-only minimality verdict**
  with the §3.1 measurement table.
