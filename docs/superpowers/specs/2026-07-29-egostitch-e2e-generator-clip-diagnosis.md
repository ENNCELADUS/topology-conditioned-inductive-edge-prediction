# EgoStitch E2E: generator clip-guard diagnosis and next-step design

**Status: PROPOSED (2026-07-29). Owner decision required.** Diagnosis of the
2026-07-28 15:15 UTC `persistent clipping in active E2E group 'generator'`
failure (`qualification/attempt-001/calibration/failure.json`) and design of
the next diagnostic step. This document proposes; it does not decide. Per the
spec freeze rule, every code-facing item lands only through
`docs/05-egostitch-spec.md` edits with §12 change-log lines, then
implementation, then a fresh v3.x BINDING registration.

This is a **different failure** from the 2026-07-27 `training_invalid(slot_collapse)`
run analysed in `2026-07-27-egostitch-e2e-rev32-slot-collapse-fix-design.md`.
Conflating them is the first trap.

**Review trail.** r1 (2026-07-29): CPU probes of the loss code, drafted an
ordering that put the degree head first on a σ-collapse mechanism. r2: three
independent adversarial reviews run in parallel against r1 — **all three
returned refutations**, two of them of r1's own central claims. r1's §2.1
(L_align exonerated) and §2.2 (σ-collapse) are both **withdrawn**; the text
below is r2. Every number in §2 is a measured parameter-space quantity, not an
activation-space one — that unit error was r1's principal defect.
r3 (2026-07-29, this text): Codex adversarial review (gpt-5.6-sol, high)
**rejected r2**, confirming its period-4 indexing and Phase-A activation audits
but refuting the "two independently sufficient causes" synthesis, the
prediction table, and the residual-headroom claim. The decisive fact Codex
surfaced — the manifest's positive/negative *layout* — is developed in §2.6 and
closes the attribution completely. r2's §3 is withdrawn.

---

## 1. The observed failure

| field | value |
|---|---|
| guard | `E2EClipGuard`, persistent arm (`train_egostitch.py:1213-1216`) |
| group | `generator` (5,563,207 params, `train_egostitch.py:1063-1072`) |
| step | 10, Phase A |
| streak | 10 |
| step-10 preclip norm | 59.2730 |
| clip max norm | 3.0 |
| step-10 coefficient | 0.05061 |
| `nonfinite_elements` | 0 throughout |
| reported peaks | step 1: 525.87 · step 4: 2601.15 · step 5: 519.88 · step 8: 2594.05 · step 9: 506.68 |

### 1.1 What the guard constants mean

`persistent_threshold = 0.1`, `persistent_steps = 10`,
`immediate_threshold = 1e-3` (`train_egostitch.py:1165-1167`), against
`generator_clip_norm: 3.0` (`configs/egostitch_e2e_v3_full_breadth_first.yaml:51-53`,
wired at `:4554`). The norm is the fp64 L2 over every `.grad` in the group's
parameter tuple (`:1130-1136`). So:

- **persistent abort ⇔ ‖∂L/∂θ_generator‖ > 30**
- **immediate abort ⇔ ‖∂L/∂θ_generator‖ > 3000**

Two consequences the raw report understates:

- **`streak = 10` at step 10 means every one of steps 1–10 exceeded norm 30.**
  The run is *born* clipping. Any explanation must hold at initialization; an
  explanation that requires a trajectory is thereby disqualified.
- **Steps 4 and 8 (2601, 2594) came within 1.15× of immediate abort.**

### 1.2 What is actually active in Phase A

`E2EPhaseState("A", 0.0, False, 0.0)` gives `edge_active = False` and
`real_ssl_scale = 0.0`, so the `generator` group's expected family reduces to
`{"recon"}` (`:4194`), with `e2e_recon_component_factors` returning 1.0 for
every component while `step < edge_start = 400` (`:808-822`).

**Correction to r1.** r1 inferred from this that only the *node* stream feeds
the generator in Phase A. That is wrong, and the error mattered.
`train_egostitch.py:3223` calls `self.model(edge_view, masks=...)`
**unconditionally** — only the BCE `edge_loss` is gated by `edge_active`
(`:3245-3254`) — and `e2e_model.py:193` runs
`self.generator.encode_nodes(x, ground)` on the edge endpoints. **So `L_align`
and `L_rel` are live in Phase A, on the edge stream, and backprop into
`generator`.** §4 depends on this.

Also: `e2e_recon_component_factors` anneals only `{feat, exist, mult, deg}`
(`:805, 819-822`). **`L_align` is never damped, at any phase, for the whole
run.**

---

## 2. Measured evidence (parameter-space, CPU, real modules at real init)

All probes instantiate the real `EgoStitchStage1` / `TokenizeLite` at default
`EgoStitchConfig` with the v3 `n_ground=50`, K=16, d_p=256, B_n=128, z-scored
inputs (‖x‖ = √1536 = 39.2 exactly). Threshold for reference: **30**.

### 2.1 `L_align` — r1's exoneration is REFUTED

r1 measured ‖∂L/∂h‖ ≈ 5.9–10.0 and concluded L_align was two orders below the
guard. **The guard does not measure ∂L/∂h.** `h` is an activation five layers
deep, not a parameter.

Measured amplification from `h` to generator parameters: **63.6–65.4** across
four seeds, ~50 for random directions on `h` (so it is a generic Jacobian
property, not an artifact of synthetic teacher cells), rising to **~89** when
‖h‖ reaches the observed late-epoch 12.5.

| r1's ‖∂L/∂h‖ | ×50 (floor) | ×64 (init) | ×89 (‖h‖≈12.9) |
|---|---|---|---|
| 5.9 | 295 | 378 | 525 |
| 10.0 | 500 | 640 | 890 |

Directly measured configurations (not extrapolated):

| B | n_pos | teach/row | ‖h‖ | ‖∂L/∂h‖ | **‖∂L/∂θ_gen‖** | clip coeff |
|---|---|---|---|---|---|---|
| 128 | 21 | 1 | 9.26 | 47.85 | **3043.5** | 9.86e-04 |
| 128 | 64 | 1 | 9.26 | 26.83 | **1726.2** | 1.74e-03 |
| 128 | 128 | 4 | 9.26 | 9.995 | **654.7** | 4.58e-03 |
| 128 | 21 | 4 | 12.87 | 34.64 | **3135.8** | 9.57e-04 |

**A configuration at r1's own upper bound (‖∂L/∂h‖ = 9.995) measures 654.7 —
21.8× the threshold.** The sparse-teacher rows sit below the 1e-3 *immediate*
abort coefficient: L_align alone can trigger the harsher guard.

Per-submodule at the 3043.5 row: `imagine.proj` 2567.76, `imagine.w_q` 1048.60,
`tokenize.encoder` 760.69, `imagine.layers` 735.21, `imagine.head_h` 670.82,
`imagine.w_g` 20.42. Three compounding causes:

1. `imagine.proj` is `Linear(1536→256)` applied to all 50 grounding candidates
   (`imagine.py:108, 197-198`), so one backward accumulates B×n_g = 6400 outer
   products against activations of norm 39.2. The *target* side is detached
   (`e2e_model.py:402-403`); the **generation** side is not, so L_align writes
   into the shared projection.
2. Sinkhorn divides the cost by `eps = 0.1` (`stitch.py:101-111`) — a flat 10×
   on everything reaching `h`.
3. `∂C/∂h = 2·(5/4)·(h_i − h_j)` (`stitch.py:38-41`) scales with ‖h‖, which is
   why amplification tracks 64 → 89 as `h_norm_mean` goes 9.3 → 12.9.

### 2.2 `L_deg` — r1's σ-collapse mechanism is REFUTED; the loss survives via μ

r1 claimed the observed norms implied `log σ ∈ [−2.2, −4.2]`. Three independent
kills:

- **Sign.** `∂L/∂log σ = 1 − ((log d − μ)/σ)²`. Measured on the real
  `TokenizeLite` at real init: mean `−1.297e-01`, **negative for 92% of
  samples**. Descent *raises* log σ. L_deg inflates σ; it cannot collapse it.
- **Support.** `build_mlp` places `LayerNorm(d_h)` immediately before the final
  `Linear` (`layers.py:33-38`), giving `Var(out) ≈ 1/3`. Over 20 seeds × 256
  rows: `deg_log_sigma` mean `+0.0024`, within-batch sd `0.5716`, global min
  `−2.0552`. **The claimed range lies outside the entire init support**; −4.2
  is a 7.3σ event, and it would have to hold for the batch *mean*.
- **Movement budget.** `lr(k) = 2e-7·(k+1)` (`:4067-4068`, `warmup_steps: 500`,
  `lr_peak: 1.0e-4`), so total AdamW parameter movement over 10 steps is
  `Σlr = 1.10e-5`. Measured drift on a fixed probe batch: log σ moved
  **+7.1e-4 nats, upward**. The claim needed −2.2 to −4.2. Short by
  **~3,000–6,000×** and wrong in direction.

**What survives.** L_deg's *parameter-space* norm at perfectly healthy init:

```
correlated_F0=False:  mean  71.58   min  45.38   max  141.50
correlated_F0=True :  mean 209.13   min  72.03   max  788.70
```

**L_deg alone exceeds the streak threshold at every step at healthy init**, with
no σ pathology of any kind. Step 10's 59.27 sits inside the measured band.

**The culpable parameter is μ, not σ.** μ is born at ≈0 while `mean(log d) ≈ 3.1`,
so the residual² ≈ 10 at σ ≈ 1:

| μ₀ | log σ₀ | L_deg | ‖∂L/∂θ‖ | |
|---|---|---|---|---|
| 0.000 | 0.000 | 2.6655 | **9.89** | shipped |
| 3.145 | 0.000 | 0.1924 | **1.19** | μ matched only |
| 0.000 | −0.127 | 3.3732 | **12.99** | log σ matched only — *worse* |
| 3.145 | −0.127 | 0.1845 | 1.50 | both |

Matching μ alone cuts the norm 8.3×. Matching log σ alone makes it worse.
r1 §2.3's framing (an unfloored `log_σ` and "a factor of 10⁶") attacked the
wrong parameter and is withdrawn.

**Batch-composition sensitivity** (independent measurement): `true_degree` is
the raw unbounded `|N(u)|` (`src/data/ego_targets.py:210`). At fixed h-cos 0.84,
‖∂L/∂θ‖ scales with the batch maximum degree — 56 → **179**, 223 → **548**,
1115 → **1443** — with ~90% landing in `tokenize.degree_dist_head.0.weight`
(`tokenize.py:56`), whose input `[x; e]` carries a large shared component that
accumulates coherently over B=128.

### 2.3 The period-4 signature is an arithmetic artifact of the overfit manifest

This is the strongest new finding and it was not anticipated by any prior draft.

Calibration runs `run_kind="overfit"` (`:6088-6090`, `:4414-4425`) over the
510-row manifest (`build_overfit_manifest`, `:2229` asserts exactly 510 rows).
`e2e_overfit_step_rows` (`:2182-2189`) takes `rows[(step·128 + i) % 510]` with
`edge_batch: 128`.

**4 × 128 = 512 ≡ 2 (mod 510).**

Every 4 steps the edge batch repeats, shifted by exactly two rows —
**126/128 = 98.4% identical**:

| reported step | global step | start | start % 510 |
|---|---|---|---|
| 5 | 4 | 512 | 2 |
| 8 | 7 | 896 | 386 |
| 9 | 8 | 1024 | 4 |

That is the observed signature exactly: steps 4/8 agree to **0.27%**, steps
1/5/9 to ≤2.6% (the slow decay being the two swapped rows plus micro parameter
drift). Random batch draws cannot produce 0.27% repeatability.

Ruled out as causes: DDP world size (norms are all-reduced, and
`e2e_assert_replicated_squared_norms` asserts cross-rank equality, `:1148-1158`);
gradient accumulation (one backward per step); `sample_branch_masks`
(`conditioning.py:42-44`, seeded per step, iid); the family probe (interval 50);
and the **node** stream, which draws a fresh permutation per cycle
(`_shuffled_nodes`, `:2746-2750`) and so carries no 4-step structure at all.

**Because the period-4 batch is the *edge* stream, and the edge stream reaches
`generator` in Phase A (§1.2), the periodicity implicates `L_align`/`L_rel` and
the Sinkhorn/Stitch path — not `L_deg`.**

### 2.4 The other eight `L_recon` components are innocent

Measured in isolation at random init across five regimes spanning init slot
h-cos 0.32 → 0.996 and batch max-degree 56 → 1115. None of the eight ever
exceeded **23.4**; all eight *summed* reach **14.8** at realistic init (h-cos
0.58) and **33.0** at total collapse. `deg` and `align` own **96–99%** of the
norm in every regime.

Strongest of the eight: `exist` (22.8 worst case, probability-space BCE
`1/(p(1−p))` at `losses.py:304`, neutralized at init because `pi = sigmoid(·)`
starts at 0.5) and `slotadj` (23.4, the only one carrying a real structural
amplifier — `adj_pred / tau_adj` with τ=0.5, exactly 2×, at `losses.py:252`).

### 2.5 The OT mass collapse — real, but a different failure

Recorded because it drove the prior diagnosis. Marginals are normalized to unit
mass (`stitch.py:89-90`), so balanced transport is mass ≈ 1. At dist² ≈ 67,
`tau` alone moves plan mass across thirteen orders: **1.3e-14** at the shipped
`tau=1` (max cell fraction 0.530), 5.4e-01 at `tau=16`, 9.4e-01 at `tau=66`.
Unit-sphere h (L-2 as drafted) lands at 0.32 — the same band. `tau` and a cost
temperature are the same lever, and `tau` is already an exposed parameter of
`sinkhorn_log_plan`.

**But it does not fix this failure.** `alignment_loss` uses row/column
*conditionals* (`losses.py:127-131`), invariant to total mass, so sweeping
`tau` 1 → 66 moves ‖∂L/∂h‖ only 5.90 → 6.44. This belongs to the deferred-ledger
L-2 decision, not to this one.

### 2.6 The manifest's positive/negative layout closes the attribution

`build_overfit_manifest` lays the rows out as **85 positives first, then 425
negatives** (`train_egostitch.py:2228-2230`, cardinality asserted at `:2229`).
`L_align` is masked to positive real rows —
`positive_real_mask = edge_mask * (label == 1)` (`e2e_model.py:426`) — and masked
again for teacher-bearing rows (`losses.py:136`). **On any batch containing no
positive rows, `L_align`'s loss and generator gradient are exactly zero.**

Combining that with the §2.3 indexing gives the positive count per step, which
reproduces every reported norm as a three-level step function:

| reported step | start % 510 | n_pos | reported norm |
|---|---|---|---|
| 2, 3, 6, 7 | 128, 256, 130, 258 | **0** | (unreported) |
| **10** | 132 | **0** | **59.27** |
| 4 | 384 | **2** | **2601.15** |
| 8 | 386 | **4** | **2594.05** |
| 9 | 4 | 81 | 506.68 |
| 5 | 2 | 83 | 519.88 |
| 1 | 0 | 85 | 525.87 |

The ordering is *inverse* in the positive count, which is exactly the
independently measured `L_align` dose-response (`_masked_global_mean` divides by
the effective row count, so fewer positive rows concentrate the per-row weight):
n_pos 21 → 3043.5, n_pos 64 → 1726.2, n_pos 128 → 654.7. And step 10's 59.27 —
the sole reported zero-positive step, where `L_align` contributes nothing — sits
inside the measured `L_deg`-alone band of 45.38–141.50.

**The decisive inference, and it does not depend on any probe magnitude or on
whether the registered interior weights were applied.** The persistent guard
requires **ten consecutive** steps above 30 (`:1206`, `:1213`). Steps 2, 3, 6, 7
and 10 carry zero `L_align` gradient. The run nevertheless reached `streak = 10`.
**Therefore some non-`L_align` component exceeded 30 on every one of those five
steps** — this is proven by the guard semantics, not inferred from a probe.

It follows that:

- **`L_align` cannot be the cause of this abort.** It is absent on half the
  steps, so it cannot form a consecutive streak, and its maximum (2601) stays
  under the 3000 immediate threshold.
- **The floor component is necessary and, given the observed streak, sufficient.**
  `L_deg` is the only candidate: the other eight components sum to 14.8 at
  realistic init (§2.4), and `L_rel`'s generator path measures 0.04162.

---

## 3. Synthesis (r3): one cause of the abort, one latent hazard

| | floor on zero-positive steps | the 520/2600 peaks | caused *this* abort |
|---|---|---|---|
| `L_deg` | **yes** — 45–141, μ-driven, every step | no | **yes** |
| `L_align` | **no** — exactly zero on 5 of 10 steps | yes | no |

**Fixing the degree head alone would very likely have prevented this abort.**
Not because `L_align` is small — it is the largest single contributor on
positive-bearing steps — but because it is **intermittent**, and a
consecutive-streak detector cannot be tripped by an intermittent term.

`L_align` remains a serious latent hazard and must still be fixed: 2601 is only
13% below the immediate-abort threshold, the margin is batch-composition
dependent, and §2.1 measured 3043.5 (coefficient 9.86e-04, *below* 1e-3) at
n_pos = 21 — a positive count this manifest would produce under a different
`edge_batch`. It is a fix for the next failure, not this one.

### 3.1 Methodological defects r3 inherits and does not yet resolve

Raised by Codex; recorded rather than papered over. None of them touch §2.6's
streak argument, which is why that argument now carries the conclusion.

- **Registered interior weights.** `w_deg = 0.5` (`config.py:147`,
  `losses.py:624`) and `w_align = 0.5` (`config.py:151`, `losses.py:632`). It is
  not established that every probe applied them. If any reported a raw component
  norm, halve it: `L_deg` 71.58 → 35.79 (min 45.38 → 22.69, which would fall
  *below* 30), and `L_align` 3043.5 → 1521.75 (no longer near immediate abort).
  **Must be resolved before any magnitude in §2 is quoted.**
- **Unexplained internal disagreement.** §2.2 reports a healthy-init `L_deg`
  norm of 71.58 in one table and 9.89 for the shipped initialization in the
  next. Different scopes or batches may explain it; this document does not say
  which, and until it does both numbers are unreliable.
- **Norms do not add.** `g_total = g_deg + g_align + g_rest`. Isolated norms
  show only that a component *could* violate the guard alone. No same-batch
  gradient vectors or pairwise cosines were measured.
- **The 33.0 residual headroom is an upper bound**, not a violation: the
  triangle inequality permits a combined norm as low as 13.8. It needs one
  backward on the weighted sum of the eight, plus pairwise cosines.
- **The "flat 10× from `1/eps`" causal language in §2.1 is invalid.** `eps` also
  enters the iterative potentials and `phi = tau/(tau+eps)` (`stitch.py:94`), so
  the total derivative through the unrolled iterations is not a flat multiplier.
  Establishing it needs an eps-controlled gradient sweep.
- **98.4% row overlap does not by itself explain 0.27% peak repeatability.**
  Only positives drive `L_align`, and steps 4 and 8 share positives
  `{0,1}` vs `{0,1,2,3}` — a Jaccard of 2/4. The overlap establishes an
  edge-stream fingerprint; §2.6's positive-count step function is what actually
  explains the magnitudes.

## 3.2 Withdrawn claims

r2 asserted "two independently sufficient causes" and "therefore no single fix
rescues the run", on the strength of isolated norms of 71.6 (`L_deg`) and
295–890 (`L_align`). **Both the claim and its conclusion are withdrawn.**
`L_align` is exactly zero on five of the ten steps (§2.6), so it was never
capable of the streak; and isolated norms cannot establish a claim about the
combined gradient in any case (§3.1).

The residual-headroom warning is likewise downgraded to an upper bound (§3.1),
and its "collapse regime" (h-cos 0.996) has no evidence of being reachable in a
repaired run.

---

## 4. Open questions

1. **`failure.json` has not been read.** It is not in this repo (checked
   `outputs/`, `docs/`, `scores/`) and lives on the cluster. `E2EClipGuard`
   retains exactly `persistent_steps = 10` trail entries (`:1202`), so it
   contains all ten steps. §2.6 makes a **sharp falsifiable prediction**: steps
   2, 3, 6 and 7 all carry zero positive rows, exactly as step 10 does, so all
   five should read ≈59 — a flat floor, not a decaying trail. If steps 2, 3, 6
   or 7 instead show peaks, §2.6 is refuted and `L_align` is back in
   contention. Cheapest decisive evidence available; still unread.
2. **All probes are at random init on synthetic or statistically-faithful
   inputs**, not on the real checkpoint, the real F0 pack, or real
   `alignment_teacher_cells` output. The amplification factor is bounded by
   measurement (50–89) but its run-time value depends on trained weight norms;
   the ‖h‖ sweep suggests it moves up, not down.
3. **Cancellation.** The guard sees the *sum*. Formally the components could
   cancel, but the others would have to cancel a 378–890 contribution to within
   ~3% to hold the total under 30.
4. **bf16.** All probes ran fp32 outside autocast. The fp32 island
   (`stitch.py:82`) covers the Sinkhorn, but not the rest of the backward.

---

## 5. Proposed next step

**Free, before any GPU time:**

- **F1.** Read all ten trail entries from `failure.json` and test the §2.3
  prediction. Confirms or kills the edge-stream fingerprint outright.
- **F2.** Instrument `deg_mu` / `deg_log_sigma` per step — now as *confirmation*,
  not discovery. Predicted: mean ≈ 0, drift < 1e-3 nats over 10 steps.

**The 2×2, retained and now sharply predictive.** §2.3 and §3 turn the
factorial from an attribution exercise into a falsification test with
pre-registered expectations:

| arm | prediction |
|---|---|
| `(off, off)` | passes — matches the completed 30-epoch diagnostic |
| `(on, off)` — align only | peaks of ~2600 on steps 4/8 and ~520 on 1/5/9, but **norm below 30 on steps 2, 3, 6, 7 → the streak breaks at step 2 → NO ABORT** |
| `(off, on)` — deg only | a flat floor near 59 on every step, no peaks → **aborts at step 10, exactly as observed** |
| `(on, on)` | as observed |

**The `(on, off)` prediction is the sharp one.** §3 says it should *pass* ten
steps despite producing the largest gradients in the entire experiment. Anyone
holding an `L_align`-first theory should expect it to fail. One arm decides
between the two accounts.

Caveat on `(off, on)`: it is not strictly "deg only". `L_rel` stays enabled at
`w_rel = 0.25` (`config.py:280`) and its generator path is live and undetached
(`e2e_model.py:381` → `:317` → `scaffold.py:367`), so the period-4 edge stream
still reaches `generator`. Measured contribution 0.04162 — negligible against a
~59 floor, so the prediction stands, but "no periodicity" should be read as
"periodic component below measurement noise", not zero.

Amendments:

- **The `(on, on)` cell must be re-run under current HEAD** so all cells share
  one build.
- **Fifth arm: `(on, on)` + μ-matched degree-head init.** Per §3 this alone is
  predicted to clear the persistent guard, with the ~2600 peaks surviving. If it
  passes, it is both the fix and the confirmation.
- **Before any of this, resolve the §3.1 weight question** — it costs one probe
  re-run and it determines whether any §2 magnitude is quotable.

**Scope discipline.** Passing 10 steps establishes only that the immediate
gradient pathology is gone. It is not a screen — `w_deg = 0` / `w_align = 0`
disable registered objectives, and G3 gate (2) Π-consistency *is* `L_align`.
The repaired `(on, on)` configuration must still complete a full calibration,
and §2.5's OT dynamics remain untested until it does.

---

## 6. Spec surface (flag, do not resolve)

- **Degree-head initialization** (μ matched to `mean(log d)`): check whether
  §13.2's degree-head definition pins initialization. Note the statistic is
  data-dependent, so if it is pinned it likely needs the same registered-digest
  treatment as the D0 standardization statistics. **Owner call.**
- **Any align-side fix** touches a registered objective. The `/eps` factor
  (§2.1 cause 2) is a registered constant; `w_align` is a registered weight;
  the `imagine.proj` sharing (cause 1) is architectural. Three different
  blast radii, one owner decision.
- **`tau` / cost temperature** belongs to deferred-ledger L-2, not here. The
  temperature variant avoids every defect r4 recorded against L-2 as drafted
  (no h-semantics change, so §13.7's raw `proj(x)` requirement, the Hungarian
  both-sides renormalization, and the affine `head_adj` Gram rescale are all
  untouched).

## 7. Incidental defects found during review (neither explains the abort)

- **Unreachable multiplicity target.** `ego_targets.py:177` sets
  `label = len(members)/count`, unbounded above (measured up to 94), while
  `slots.mult` is hard-clamped at `m_max=32` (`imagine.py:212`). `L_mult`
  therefore carries a permanent irreducible floor of `0.5·(log(target/32))²` on
  hub nodes that no parameter setting can reach.
- **`L_div` weakens exactly when it is needed.** The squared hinge
  (`losses.py:320`) gets *smaller* as slots collapse (0.048 → 0.007 from h-cos
  0.58 → 0.996), because `∂cos/∂h` is suppressed by 1/‖h‖ and averaged over
  ~2000 eligible pairs. At `w_div = 0.1` it cannot counteract slot collapse.
  Relevant to the `e2e-stage1-slot-collapse-rev31` line of work, not to this
  failure.
