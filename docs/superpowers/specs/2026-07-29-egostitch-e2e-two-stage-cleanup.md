# EgoStitch E2E: two-stage cleanup

**Status: IMPLEMENTED IN CURRENT WORKTREE (2026-07-30), rev 3; cleanup commit pending.
Owner-decided via grill-me interview.** Collapses the e2e ladder from five stages to two, removes
the legacy frozen-s0 `egostitch` family, and adopts the data-derived degree-head
initialization as standard.

Per the spec freeze rule, the code-facing items landed through
`docs/05-egostitch-spec.md` edits with §12 change-log lines, then implementation.
This implementation status authorizes no execution: the v4 registration remains
`DRAFT`. §7 records the required documentation edits.

Two delete-list items are sanctioned retentions rather than incomplete cleanup:
the non-publishing `probe`/`epoch-probe` dispatch used to measure
`feature_stats_sha256`, and `runtime.total_budget_seconds` as the invariant sum of
the remaining stage budgets. The authoritative rationale is recorded in
`docs/05-egostitch-spec.md` §12 (2026-07-29, second entry); §4 below reflects it.

**Review trail.** r1 (2026-07-29) drafted a ~2,000-node BFS-ball Stage 1 with a
guards-only ladder and a design-only registration. Three independent adversarial
reviews were run in parallel against r1; all three returned blocking findings,
two of them independently. r1's central preflight was arithmetically incoherent
(§B1 below), its Stage 1 could not run in the repo data root, its degree
statistic carried a units error, and its removal of every pre-registered decision
rule violated protocol §5.2.4. **r1 §2 (BFS ball), §2.2 (seed-count evidence
promotion), §3 (design-only registration and digest propagation), and acceptance
item 6 are withdrawn.** The text below is r2. §10 records what r1 got wrong so
the same ground is not re-walked.

**r3 owner amendment (2026-07-30).** The first full qualification confirmed the
§5.1 persistent-gradient prediction. Qualification now keeps the registered clipping
operation and complete telemetry but does not abort merely because a clip coefficient
stays below `0.1` for ten consecutive steps. A complete otherwise-valid qualification
writes `pending_manual_review`, never automatic `pass`; formal rejects that pending
state. All other hard failures remain unchanged. This design trail defines no approval
artifact and never edits an immutable attempt in place.

**r4 owner amendment (2026-07-30; supersedes r3's qualification-verdict
boundary).** Qualification is exactly three epochs and every finite model-quality
threshold is telemetry-only: initial/during slot statistics, finite zero
gradient/family norms, immediate/persistent clipping, family ratio, warm-reference
and validation dispersion floors, precision thresholds, validation collapse, and
checkpoint eligibility. Actual clipping is unchanged. Only non-finite, DDP, boundary,
coverage, and I/O/infrastructure failures abort qualification. Every complete
three-epoch run writes `pending_manual_review`; with no quality-eligible checkpoint,
the existing `best.pt`/`last.pt` compatibility aliases expose the final epoch for
diagnostic manual review only and metadata/profile mark it non-eligible. Formal guards,
eligibility, selection, and fail-closed preflight remain unchanged.

---

## 1. Why

The previous ladder was `init-probe → calibrate(overfit) → threshold-freeze →
rehearse → formal`, wrapped in seven independent blocking devices. Every model
iteration paid that five-stage tax, and the fastest available signal cost a
15-minute calibration run plus a hand-pasted digest.

The cleanup buys one thing: **a fast development loop**. Stage 1 is where model
work happens; Stage 2 produces results.

## 2. The ladder

**Naming.** "Stage 1" and "Stage 2" are used below only as local shorthand for
the **qualification stage** and the **formal stage**. They are *not* the G5
Stage-1/2/3 architecture ladder (imagination / codebook / harmonization) and not
the G5 Stage-1 screen. In `docs/05-egostitch-spec.md` and
`docs/03-experiment-protocol.md` the unabbreviated forms are used throughout,
because all three senses coexist there.

```
train nodes (8,072; 2 featureless)
 ├─ V_hold       512 nodes = V_qual ∪ V_select   → validation, BOTH stages
 └─ V_fit   unchanged from today            → training, BOTH stages
      ├─ STAGE 1  QUALIFICATION   full V_fit, short schedule
      └─ STAGE 2  FORMAL          full V_fit, full schedule
test (2,018 nodes) — sealed; see §2.4 for what "opened once" actually means
```

**Both stages train on the identical universe.** They differ only in
`optim.epochs`. This is the single most important property of the design and it
is what makes the whole thing cheap:

- No third role-universe, no Stage-1 pack dir, no Stage-1 grounding cache, no
  Stage-1 F0 cache. `src/data/grounding.py:126-133` fails closed on any universe
  drift; there is now no drift.
- `feature_stats_sha256` is *identical* between stages, so the Stage-2 preflight
  compares equal values rather than propagating an incompatible digest. This is
  what killed r1 (§10, B1).
- `mean(log d)` is identical between stages, so the degree-head initialization
  Stage 1 certifies is the one Stage 2 runs under (§5).
- The degree regime, isolate fraction and ego-net statistics are identical by
  construction, not by a sampler argument.

Stage 1 at exactly `epochs=3` costs ~1/10 of Stage 2. The three-phase curriculum scales
with `schedule_total_steps = steps_per_epoch × epochs`
(`_epoch_step_plan`, `train_egostitch.py:2585-2600`), so a short run still
traverses A → B → C. Eligibility is recorded as telemetry and may be absent;
Phase C opens at 30% of the schedule regardless of its length.

### 2.1 V_hold is the union of the two existing holdouts

`V_qual` and `V_select` are already node-disjoint with an asserted all-zero
`OverlapProof` (`internal_holdout.py:174-175`) and are *already* both subtracted
from `V_fit` (`:156`). Defining `V_hold := V_qual ∪ V_select` therefore leaves
`V_fit`, `e_msg_fit`, `e_sup_fit`, `G_fit`, `rho_train`, `feature_stats_sha256`,
`pool_method_hash` and every pack manifest **bit-identical to today**. No cold
rebuild. Stage 2 remains comparable to the v2/v3 baselines.

Selection positives rise from 807 (`V_select`, the manifest the formal run
actually selected on) to **1,533** = 456 + 807 + 270 cross-side, over
C(512,2) = 130,816 pairs — **1.9×, not the "roughly triples" r1 claimed**, which
compared against the wrong baseline. The 270 cross-side edges are genuinely new
topology gold: today they are quarantined into `quarantine_counts` and used by
nothing.

**Two consequences, disclosed not hidden.** Pairs grow 4× while positives grow
1.9×, so prevalence *halves*, `0.0247 → 0.0117`:

- AP sampling SD shrinks only ≈`sqrt(1.9)` ≈ 1.4×, so enlarging the holdout
  helps materially less than a naive area argument suggests.
- The retained eligibility floor `prevalence + 0.02` drops in absolute terms from
  `0.0447` to `0.0317` — **a 29% reduction in the very floor credited with
  preventing the 2026-07-19 v1 degenerate-checkpoint failure.** Eligibility is
  retained unchanged in form; its bar is lower. If that is unacceptable, the
  additive `0.02` must be re-pinned, and that is an owner decision.

`auprc_tolerance` (`train_egostitch.py:1331`, currently `0.02`) must be
**re-derived from V_hold's measured AP sampling SD** rather than left at a value that
happened to equal ~1 SD of the old, smaller manifest — otherwise the AUPRC band
admits nearly every post-ramp epoch and selection falls entirely to a
single-sample MMD² argmin. See §9 for the residual risk that survives this.

**Known limitation, recorded not fixed:** V_hold's gold graph is a BFS truncation —
its nodes carry degree-sum 9,390 in full `G_msg` but only 3,066 inside V_hold (33%
retained). Selection against `clustering_histogram(gold)` therefore selects
partly for fidelity to the cut. This is pre-existing; the union mitigates it
(cross-side edges are retained) but does not remove it.

### 2.2 What each stage does

| | Stage 1 — qualification | Stage 2 — formal |
|---|---|---|
| trains on | full V_fit | full V_fit |
| validates on | V_hold | V_hold |
| epochs | exactly 3 | registered full schedule |
| purpose | development loop | results |
| verdict | complete run → `pending_manual_review`; never automatic `pass` | registered thresholds, §2.3 |
| arms | whichever is under development | 6 trained + 2 scoring-time controls |
| seeds | 1 | `--seeds`, default `0` |
| test | never opened | §2.4 |

**Selection — arm *and* checkpoint — happens on V_hold, never on test.** Ranking arms
on test and promoting a winner is test-set selection bias and would void the
ablation table.

### 2.3 Decision rules

**Stage 1: complete run, then manual review.** The registered global-norm clipping
remains active (`3.0` for pair/generator, `1.0` for conditioning), and every pre-clip
norm, clip coefficient, and consecutive-low-coefficient streak remains telemetry.
Every finite model-quality threshold is non-aborting in qualification: initial/during
slot statistics, finite zero gradient/family norms, immediate/persistent clipping,
family ratio, warm-reference and validation dispersion floors, precision thresholds,
validation collapse, and checkpoint eligibility. Only non-finite, DDP, boundary,
coverage, and I/O/infrastructure failures remain hard. Every complete three-epoch run
writes `pending_manual_review`, never `pass`, regardless of quality eligibility.
`pending_manual_review` does not authorize Stage 2, and this revision defines no
approval artifact or conversion to `pass`.

Two consequences that must be implemented, not assumed:

- `e2e_checkpoint_eligible` (`:1284`) computes an AUPRC floor of
  `prevalence + 0.02` plus `residual_ratio`, `active_logit_std` and
  `topology_gradient_norm` conditions. These are qualification telemetry and remain
  hard formal eligibility — they prevent formal selection from repeating the
  documented 2026-07-19 v1 failure where selection landed on a reconstruction-only
  warm-start checkpoint.
- If `select_e2e_checkpoint` finds no quality-eligible epoch, Stage 1 exposes the final
  epoch through the existing `best.pt`/`last.pt` compatibility aliases for diagnostic
  manual review. Metadata/profile mark it non-eligible; neither alias is Stage 2
  authorization, and formal still forbids fallback.

**Stage 2: pre-registered thresholds, pinned before test opens.** Registration v4
re-pins the v2 acceptance thresholds — clustering-MMD and BFS-macro GS/RD
dominance at matched global RD, plus the AUPRC and degree-MMD guards. These were
never part of the 2026-07-27 G3.4/G3.5 circularity (that loop was
scoring→BINDING→thresholds for the *pre-binding* gates); they were satisfiable in
v2 and are what produced the `cut`. Protocol §5.2.4 requires them recorded before
held-out metrics are opened.

### 2.4 Test access

This ladder **cannot** promise "test opened exactly once" and the spec must not
claim it. Measured reality: scoring shards across every visible GPU
(`hpc/run.sh:78-131`), each shard independently reading `candidate_test_edges.txt`
(`score_universe.py:1425-1428`); `load_test_graph` is called at three sites
(`g5_stage1.py:1036`, `:2010`, `:2333`); the v2 screen spent ≈24.5 h across five
candidate-scoring passes.

The enforceable property is **one scoring epoch per (arm, seed), no re-scoring
after seeing results.** Implement it as an append-only test-access ledger written
at each `_resolve_pairs` call on a held-out source, checked into the run
artifacts. Re-scoring is permitted only with a recorded reason and it is visible.

**Arm count is 8, not 6.** The 6 trained arms plus the two scoring-time controls
`structure_control_6a_v3` and `structure_control_6e_v1`. r1 said "6 arms" and
thereby silently dropped 6e-v1 — the degree-preserving rewiring control that
protocol §E4.16(e) calls decisive and whose paired-bootstrap lower bound of `0.0`
was one of the four legs of the v2 `cut`.

### 2.5 Seeds and evidence class

`--seeds` sets the seed list. **This ladder never emits `evidence_class:
inference`, at any seed count.** CLAUDE.md is explicit — only E1/E3 carry
inference — and protocol §5.0.5 adds that inference additionally requires the §8
30-config HPO-parity budget and Holm over the pre-registered held-out assembled
family, neither of which a Stage-1-descended screen has. r1's mechanical
"3 seeds ⇒ inference" rule is withdrawn.

- `--seeds 0` → `evidence_class: engineering`; `p_value`, `ci`, `holm` written
  `null`, artifact refused if not.
- `--seeds 0,1,2` → still `engineering`; adds **cross-seed variance reporting**,
  not significance. Same null enforcement.

Multi-seed additionally requires relaxing four hard seed-0 pins:
`g5_stage1.py:1005-1007`, `:1028`, `:1499`, `probes.py:868-869`.

## 3. Gates

Stage 1 writes one artifact:

```json
{ "verdict": "pending_manual_review" | "fail(<named_guard>)",
  "epochs", "hparams",
  "feature_stats_sha256",
  "model_config_sha256" }
```

Stage 2's preflight is three assertions: the file exists, `verdict == "pass"`,
and both digests match. Because §2 makes the universes identical, the feature
digest comparison is a genuine equality — no propagation, no hand-pasting, and
`train_egostitch.py:2422`'s stats-universe == training-universe assertion holds
in both stages unchanged.
`pending_manual_review` therefore fails closed. r3 intentionally defines no approval
artifact or in-place mutation capable of changing an immutable attempt to `pass`.
The existing profile, metrics, and run metadata carry the finite-threshold telemetry,
per-predicate eligibility status, and diagnostic-selection identity. Existing
checkpoint aliases are compatibility outputs, not formal authorization artifacts.

`model_config_sha256` **must be defined; no such function exists today.**
`_config_hash` (`:5523`) bakes in `output_dir`, `data.root` and `preregistration`
(a documented CLAUDE.md trap), so it cannot serve — the two stages necessarily
differ in `output_dir`. Define it over the model-defining keys only, explicitly
excluding `output_dir`, `optim.epochs` and `feature_stats_sha256`.

Deleted: `calibration_freeze.json` and its three-way sha assertion
(`hpc/qualification.sh:133-235`), the exclusive-create rehearsal ledger
(`:102-126`), the `attempt00[1-3]` window (`:317-334`),
`src/experiments/prebinding_gates.py` and `tests/experiments/test_prebinding_gates.py`,
the `REQUIRED-BEFORE-BINDING` markers, `qualification_margins.json`, and the
manual `feature_stats_sha256` field in the six v3 configs.

`validate_e2e_qualification_profile` (`:5539`) is **retained** and moves to
Stage 2 — it is the repo's only clip-coefficient / family-ratio / submodule-RMS
margin gate, and nothing else replaces it.

### 3.1 The test-boundary guard must be strengthened, not deleted

`train_egostitch.py:2426-2431` raises when held-out files are present **and**
`run_kind != "formal"`. Two problems: Stage 1 cannot run in the repo data root
at all (all three forbidden files exist there), and the check is *disabled for
Stage 2*, so it has never guarded the run that matters.

The sanitized-root sandbox that made this work today
(`assert_qualification_boundary`, `hpc/qualification.sh:285-294`) is reachable
only from the deleted `calibrate`/`rehearse` subcommands.

**Replace the run-kind condition with a path condition applied to both stages:**
training may not open a held-out path, in any run kind. This is strictly stronger
than today, removes the sandbox friction that would otherwise make the "fast
loop" not fast, and is the only thing that makes acceptance item 4 implementable.
Mere presence of candidate, test, or `test_graph.pkl` files in the shared repo data
root is allowed; an attempted open, including through an alias or symlink, raises
before the held-out path is read.

## 4. In-run sub-stages

`pack → train → publish`. Deleted: the token-budget probe pipeline stage and
`runtime.token_budget_candidates`; the projection stage; `--ddp-mode init-probe`
and `_run_init_probe`; `select_probe_result`,
`conservative_e2e_epoch_seconds`, and `project_total_seconds`.

Two load-bearing pieces are retained under the §12 change-log exception. The
`probe`/`epoch-probe` entries in `_PROBE_DISPATCH_MODES` and
`--ddp-mode epoch-probe` survive only as a non-publishing measurement dispatch for
the pre-run `feature_stats_sha256`; they are not pipeline sub-stages. Likewise,
`runtime.total_budget_seconds` remains the declared wall-clock total whose value
must equal the sum of the remaining stage budgets; the projection sub-stage that
formerly consumed part of it is deleted.

The budget probe goes because it did not prevent the OOM it exists to prevent —
the rev-3.1 OOM was `_e2e_family_probe` double-buffering autograd graphs, a path
the probe never measured. `runtime.token_budget` becomes a scalar config value;
note this is a `RuntimeConfig` schema change plus edits to all six configs, since
`token_budget` is a required positional of `build_accelerate_command`
(`e2_pipeline.py:343`) forwarded to the worker's required `--token-budget-per-rank`.

**Pack reuse is narrower than r1 claimed.** The raw-token pack is genuinely
universe/arm/seed-independent. The F0/grounding pack (`--pack-dir`) is keyed by
`run_kind`, `validation_role` *and* `n_ground` (`:1907-1908`, `:1940-1945`). The
first two must be dropped from the manifest for the two stages to share a pack.
The third cannot be: **`configs/egostitch_e2e_v3_cosine_pool_breadth_first.yaml:13`
pins `n_ground: 20` while `hpc/qualification.sh:478` forces one shared pack built
at 50, so that arm raises at `:1940` today.** This is a pre-existing live bug, not
something the cleanup introduces; it needs two packs, one per `n_ground`.

### 4.1 Step-0 guards

r1 claimed h-cos and slot/plan statistics are "computed at step 0 and fail fast".
Partly fiction. What is true:

- `h_pairwise_cosine_mean > 0.95` is computable at step 0 and is the check
  `_run_init_probe` already performs (`:6210`). It remains telemetry in
  qualification and a hard condition in formal.
- `E2ESlotCollapseGuard` is **inert at step 0**: it short-circuits on
  `conditioning_active`, and `e2e_phase_state(0, N)` is Phase A with
  `edge_active=False` (`:775-776`, `:926-937`). It also needs two consecutive
  validations. It cannot be a step-0 guard; it stays a during-training guard.
- Clip telemetry and non-finite-loss checks need a forward/backward, so they are
  available at step 1 at the earliest; finite clip thresholds do not abort
  qualification.

The raise must follow the loop's rank-synchronization pattern (`accelerator.reduce`
then raise on all ranks, `:4823-4828`) — `_validate_epoch` returns `None` on
non-main ranks, so a rank-0-only raise is a DDP hang. Placement is inside
`_train_e2e_stability_loop` after `accelerator.prepare` (`:4433`), passing the
**unwrapped** model per the existing `[P2]` convention.

Step 0 also costs a full validation pass over C(512,2) = 130,816 pairs on every
run, at the same time as §4 deletes the projection stage that used to account for
runtime. Budget it explicitly.

## 5. Model changes landing with this

**`e2e_degree_prior_init` becomes standard** (`train_egostitch.py:969+`,
currently uncommitted): the degree head's output bias is set to `mean(log d)`
over `G_fit`. `deg_mu` is a raw linear output born near 0; `degree_nll`'s `1/σ²`
turns the standing residual into a generator gradient over the clip threshold on
every step from step 1. `log σ` is deliberately left alone — matching it instead
of μ measures worse.

**The fix is already validated empirically.** Unregistered engineering run
`engineering/degfix-20260729` (4,326 s, 4×H20): 30/30 epochs, eligible from
epoch 10, `validation_liveness_pass`; zero-positive floor 59–68 → 21–27; **max
consecutive steps above 30 went 10 → 2**. Slot guards comfortable (h-cos max
0.879 vs 0.95; plan residual min 0.128 vs 0.05). AUPRC 1.0 is on 510 memorized
rows — overfit behaving as designed, not generalization and not a screen.

**Two corrections before it lands as standard:**

- **The residual is 1.1538 nats, not "~3".** Measured on `G_fit`; 3.17 is
  `exp(1.1538)`, the *geometric mean degree* — a units error inherited from the
  diagnosis doc. This does not invalidate the fix (the code computes the value
  it needs), only the narrative magnitude quoted around it.
- **Cross-rank determinism is not established**, contrary to the docstring. The
  function iterates `graph.nodes()`, and `build_g_fit` adds nodes from a
  `frozenset[str]` (`internal_holdout.py:71`, `:81-86`); iteration order depends
  on `PYTHONHASHSEED`, pinned nowhere in this repo, and `np.log(...).mean()` uses
  pairwise summation. Fix: `sorted(graph.nodes())`.

The 2×2 factorial proposed in
`2026-07-29-egostitch-e2e-generator-clip-diagnosis.md` §5 is **cancelled** — its
sharp `(on, off)` prediction depends on the zero-positive steps produced by
`build_overfit_manifest`'s 85-positives-first layout, which this cleanup deletes.

### 5.1 Recorded prediction (tripwire, not a fix)

`negative_ratio: 5`, `edge_batch: 128` ⇒ ~21 positive rows in **every** batch once
the 510-row manifest is gone. `L_align` was ruled innocent of the 2026-07-28
abort *because it was intermittent* — zero positives on 5 of 10 steps — and that
intermittency was an artifact of the manifest layout.

**Prediction: under the new ladder `L_align` becomes a persistent gradient
source, and the deg fix will not carry the run.** This now rests on the degfix
run's own telemetry, not on CPU probes:

- That run cleared the guard by dropping **max consecutive steps above 30 from 10
  to 2**. The surviving 2 are precisely the positive-bearing steps. Make every
  step positive-bearing and the streak returns to 10 directly.
- Its own dose-response: ~520 at 81–85 positives, ~2600 at 2–4 positives. n_pos
  ≈ 21 sits between — one to two orders above the threshold of 30, on every step.
- Its margins are already thin under *intermittent* align: step 11 hit 29.218
  against the 30 streak threshold (2.6%), and max norm over 2,000 steps was
  2835.5 against the 3000 immediate-abort threshold (5.5%).
- The residual floor is mostly the *other eight* `L_recon` components (~15–20
  summed), not `L_deg`, so the degree fix cannot widen it further.

**No align-side fix lands now.** The first Stage-1 run is the test — cheap, and
more honest than a factorial built on CPU probes the diagnosis itself flags as
unreliable. If it aborts, the fix lands then, informed by run telemetry. If it
does not, the prediction is refuted and the hazard closes. Narrowest candidate
for that eventuality: `e2e_recon_component_factors` anneals
`{feat, exist, mult, deg}` but not `align` (`:805, 819-822`), so align runs at
full weight from step 1 for the whole run.

## 6. Deletions

### 6.1 Ladder machinery

`build_overfit_manifest` (`:2247-2266`) and its co-dependents, which r1 did not
enumerate: `OverfitManifest` (`:2207`), `EgoStitchData.overfit_manifest` (`:2181`),
`e2e_overfit_target_seed` (`:2240`), `_BatchFactory.fixed_row_batches`
(`:3042-3097`), `overfit_manifest_sha256` (`:5039`);
`e2e_overfit_epoch_step_counts`, `e2e_overfit_step_rows`,
`e2e_overfit_rank_step_rows`, `e2e_overfit_epoch_qualified`,
`select_e2e_overfit_epoch`, the 2,000-step pin; the ten `run_kind == "overfit"`
branches at `:4415, 4446, 4475, 4528, 4562, 4691, 4750, 4805, 4809, 4865`;
`hpc/qualification.sh`'s `calibrate` and `rehearse` subcommands.

`E2EProbeScope` loses `calibration_fit` and `qualification_qual`. **`formal_train`
survives and reads `holdout.v_select` (`probes.py:1005-1006`)** — it must be
repointed at V_hold, along with `select_manifest`, `_cross_partition_counts`'s
hardcoded `{"fit__qual","fit__select","qual__select"}` (`internal_holdout.py:241`)
and `_pairwise_overlap_counts` (`:251-256`), whose outputs land in the published
`access_audit`.

`hpc/qualification.sh:355-374` (`evaluate_stage_gates`) is the **only** invocation
anywhere of `python -m src.experiments.probes produce-e2e`, and
`g5_stage1.py:2699-2703` errors without its output. A producer for the
`formal_train` probe artifact must be re-homed into the Stage-2 path before those
subcommands are deleted.

**`run_kind` → `{qualification, formal, debug}`.** `debug` is a real fourth value
today (`:5649`, `:5712`, `e2_pipeline.py:1239`, read at `probes.py:877`,
`g5_stage1.py:157`) that never appears in the `Literal`. Three whitelists fail
**open** under a bare rename and must be rewritten, not just extended — this is
the same fail-open class commit `8d24a08` closed:

- `:6122-6132` — `effective_run_kind in ("rehearsal","formal")` is what forces the
  digest pin; a new `qualification` kind would be unguarded.
- `:5747`, `:5751` — `expected_run_kind != "overfit"` sets `checkpoint_eligible`
  and `selected_checkpoint_eligible`; with `overfit` gone this is unconditionally
  true and a Stage-1 run would publish an "eligible" checkpoint.
- `e2_pipeline.py:443` — `qualification` must be **added** to `choices`.

The `epochs != 30` pin (`:679`) is removed from the equality block. Note
`_validate_staged_artifacts` and `_validate_worker_profile` key off
`cfg.optim.epochs` and track automatically, but changing `optim.epochs` in a
config changes `_sha256_file(config_path)`, which `_validate_e2e_formal_binding`
compares against registered `binding_evidence` (`:1651-1655`) — so registration
v4 must re-pin the six config digests.

### 6.2 Legacy frozen-s0 `egostitch` family

**Safe whole-file deletions:** `src/experiments/g5_stage1_diagnostics.py`,
`src/eval/ego_fidelity.py`, `hpc/g5_stage1.sh`,
`configs/egostitch_stage1_breadth_first.yaml`, `tests/test_g5_stage1.py`,
`tests/test_g5_stage1_diagnostics.py`, `tests/eval/test_ego_fidelity.py`.

**Excisions:** `write_s0_manifest`, `build_s0_manifest`,
`EgoDataConfig.s0_cache`/`s0_checkpoint_id`, `optim.warmstart_fraction`, the
legacy body of `train_egostitch_ddp_loop` from `:5137` (**not** `:5125-5136`,
which is the live e2e dispatch), the legacy `isinstance(model, EgoStitchStage1)`
branches of `_run_probe_mode`; `_build_egostitch`, `_align_s0_logits`,
`_score_egostitch`, the `egostitch` branch of `_run_score` and the
`"egostitch": _build_egostitch` entry at `score_universe.py:880`;
`--mode frozen_s0` and its pipeline; `s0-score` (`hpc/run.sh:161-178`).

**Corrections to r1's delete list — these are NOT safe:**

| item | why |
|---|---|
| `enumerate_edge_stream` | live e2e: `_BatchFactory.epoch_batches` (`:2999`) |
| `S0Cache` | `_assemble_e2e_data` constructs one at `:2432-2433` to fill the required field `EgoStitchData.s0` (`:2172`); the field must be excised in the same change |
| legacy body of `assemble_egostitch_data` | the `cfg.training is None` branch is e2e-aware (`:2560`) and is exercised by `tests/test_train_egostitch_e2e.py:2143-2179` |
| `tests/test_train_egostitch.py` | **split, do not delete** — it holds the only tests of the live e2e BINDING gate (`:391`, `:399`, `:414`), the active-config contract (`:431`), `_epoch_step_plan`/`_step_global_count` (`:591-603`), `_BatchFactory` determinism (`:703`), `_GradientImbalanceMonitor`, `_enforce_probe_s1_scale` and the `feature_stats_sha256` metadata contract |
| `tests/helpers/egostitch_ddp_smoke.py` | subprocess-run by `tests/test_e2_ddp_integration.py:164-190`, which is not on the delete list |
| `enforce_frozen_inputs`, `_resolve_b0cal_results_path` | reached from `enforce_e2e_frozen_inputs`, called by `run_g5_e2e_stage1_pipeline:2307` |
| `decision.py`, `model.py` | e2e reads `tau_kappa` (`e2e_model.py:341`); `model.py` **is** `EgoStitchE2E.generator` |

**Removing `s0_cache`/`s0_checkpoint_id`/`warmstart_fraction` changes
`_config_hash` for every e2e arm**, because it is `sha256(json(asdict(cfg)))`
(`:5523`, `:721`). That breaks the `config_hash` equality gates at
`probes.py:928-929` and `g5_stage1.py:1516-1517` against every existing
`run_metadata.json` and probe artifact. Either keep the fields as deprecated
no-ops or re-issue those artifacts; decide before coding.

**Evidence kept.** `docs/results/G5-stage1-seed0-20260717.md`,
`docs/registrations/g5_stage1_preregistration.{json,md}` and
`outputs/egostitch_stage1/` survive. The Markdown evidence files state that the
producing code last exists at `dcae090` and that its deletion is present only in
the current worktree pending a cleanup commit. The binding JSON remains byte-for-byte
unchanged because adding a note would alter the registered evidence hash.

**Test files r1 undercounted:** `tests/test_hpc_scripts.py` reads `hpc/g5_stage1.sh`
at module level, so `:26-33`, `:49-59`, `:64`, `:85`, `:89`, `:102`, `:118`,
`:127`, `:143`, `:151`, `:171` all break — not just `:73-177`.
`tests/test_hpc_qualification.py:146-269` statically asserts the deleted bodies.
`tests/test_score_universe.py:877-1130` calls `_score_egostitch`.
`tests/test_e2_pipeline.py:298-322` asserts `--run-kind rehearsal`.
`tests/experiments/test_probes.py:541-621` and
`tests/test_train_egostitch_e2e.py:1325, 1447` are parametrized on the removed
run kinds.

**Observed and dispositioned 2026-07-30.** The first full qualification produced
generator norms `540–618` and tripped the ten-step persistent-clipping condition.
That confirms the tripwire but does not by itself establish divergence or scientific
failure. The owner chose to keep actual clipping and telemetry, remove only this
qualification-time automatic abort, run the complete three epochs, and require manual
review of the immutable result. The formal-stage hard failure and margin checks remain
unchanged.

The subsequent r4 owner decision broadens that qualification-only rationale to all
finite model-quality thresholds. Non-finite, DDP, boundary, coverage, and
I/O/infrastructure failures remain hard; formal is unchanged.

## 7. Doc edits

Rewritten in place — no `§15`, no SUPERSEDED strata:

| section | change |
|---|---|
| §9.3 (`:379-396`) | two holdouts → `V_hold := V_qual ∪ V_select`; `V_fit` unchanged |
| §13.12 (`:1259-1262`) | role universes: `V_fit`-side, V_hold, test |
| §13.19.1 (`:1550-1595`) | **added by r2** — `zscore_vfit_v1` semantics are unchanged by this cleanup precisely because both stages use V_fit; state that explicitly so the method id is not re-litigated |
| §13.19.3 (`:1666-1689`) | strike "`V_select` unread until the first formal bound run"; V_hold serves both stages; re-derive `auprc_tolerance`; correct the metric name (§10) |
| §13.19.4 (`:1720-1766`) | five pre-binding items → complete qualification with `pending_manual_review`; no automatic pass |
| §13.19.6 (`:1797-1818`) | acceptance matrix retargeted at the two-stage ladder |
| §14.4.7 (`:2070+`) | calibrate→freeze→rehearse→formal → the §2 ladder |
| §13.1–13.17 (`:1064-1420`) | frozen-s0 carve-out retired with its code |
| protocol §5.0.5, `:506-508` | drop the v2-DRAFT/§13.19 pointer; keep the E1/E3-only-inference rule intact |

The §12 change-log line **names the pre-rewrite commit sha** — that is the whole
mitigation for editing in place, letting the published `cut` verdicts resolve
their citations through git.

CLAUDE.md needs updating: three data-contract traps retire with their code
(never-shard-the-Stage-1-scorer, the s0 manifest binding, the s0 cache).

## 8. Acceptance

1. Stage 1 runs end-to-end on full V_fit for exactly three epochs and writes
   `qualification.json` with `pending_manual_review`, regardless of quality
   eligibility. Every finite quality-threshold miss must continue under unchanged
   clipping with complete telemetry. No eligible checkpoint means final-epoch
   diagnostic selection through the existing compatibility aliases, with metadata and
   profile marking it non-eligible; neither alias is formal authorization. Named hard
   verdicts remain `fail(<named_guard>)`, limited to non-finite values, DDP,
   boundary, coverage, and I/O/infrastructure failures; an
   interrupted, incomplete, or inexact-coverage run cannot write pending.
2. Stage 2 refuses to launch on a missing, `pending_manual_review`, `fail`, or
   digest-mismatched `qualification.json`. It may not fall back from the latest
   pending attempt to a stale historical pass.
3. Any seed count produces `evidence_class: engineering`; a non-null
   `p_value`/`ci`/`holm` is refused. No path emits `inference`.
4. The path-scoped held-out guard (§3.1) raises in **both** run kinds. Tested by
   attempting to open candidate/test/`test_graph.pkl` through direct, alias, and
   symlink paths in both run kinds. Mere presence in the shared data root is allowed.
5. `V_hold ∩ V_fit = ∅`; `|V_hold| = 512`; `V_fit` digest equals the pre-cleanup value.
6. **Stage 1 and Stage 2 produce the *identical* `feature_stats_sha256` and the
   identical `mean(log d)`.** (r1 asserted these must *differ*; that was the
   hazard, not the criterion.)
7. Stage 2's registered thresholds are present in run metadata before any
   held-out read, and the test-access ledger (§2.4) records one scoring epoch per
   (arm, seed).
8. The full suite passes with legacy tests deleted and the e2e tests from
   `tests/test_train_egostitch.py` preserved in a new file — not skipped.

## 9. Residual risk, accepted

Enlarging V_hold and re-deriving `auprc_tolerance` reduces but does not eliminate
selection inflation. With ~20 eligible epochs × 8 arms per formal run, plus an
uncapped pre-binding number of Stage-1 development iterations on the same V_hold,
the accumulated argmin count K can be large and selection noise is **not exchangeable
across arms** — a higher-capacity arm draws a larger max-of-K. This biases the
`full` vs `b0_e2e_f_only` contrast, which is the headline ablation.

This is the price of the Q2 decision to share one holdout across both stages, and
it is accepted knowingly. Every qualification attempt is durably retained under its
arm's immutable `attempts/attempt-*` history, including failures; the launcher freezes
new attempts once the registration becomes `BINDING`. The formal artifact binds the
cumulative count of V_hold evaluations per arm to that history. Thus K is recorded and
bounded at binding rather than hidden, but the resulting selection inflation can still
be large and nonexchangeable across arms.

## 10. What r1 got wrong (do not re-walk)

- **BFS ball Stage 1.** Measured: ball mean degree 11.66 vs V_fit 6.88 (1.70×
  dense), zero isolates vs 16.4%. Uniform sampling is 4× sparse. Neither is
  representative; the subset idea is withdrawn entirely in favour of a short
  full-V_fit run, which also dissolves blockers B1 (feature-stats universe) and
  the third-role-universe cache problem.
- **"~3.1 nats" degree residual.** Real value 1.1538 nats on `G_fit`; 3.17 is the
  geometric mean degree.
- **"clustering RD".** The computed quantity is raw single-graph clustering MMD²
  (`:3655-3658`), not RD (`density(G_pred)/density(G_ref)`). CLAUDE.md forbids
  the conflation; spec §13.19.3 is careful about it and r1 undid that care.
  The canonical MMD *ratio* and macro GS/RD are not computable on V_hold at all — one
  gold graph gives no real-vs-real floor and no bucket structure.
- **"V_fit grows +3.4%".** Under r1's single-256-holdout it would have grown
  `e_msg_fit` by +18.3%, invalidating every cache and breaking v2/v3
  comparability. §2.1's union construction makes the change zero.
- **"Test opened exactly once."** Unenforceable and already false; replaced by
  §2.4's ledger.
- **"3 seeds ⇒ inference."** Violates CLAUDE.md and protocol §5.0.5.
- **Design-only registration.** Left the ladder with no pre-registered decision
  rule while test was opened; violates protocol §5.2.4. The v3 circularity did
  not force it.
- **"Step-0 slot-collapse guard."** Inert at step 0 by construction.
- **"Pack built once, reused across all arms."** False for the F0/grounding pack.

## 11. Deferred

- **`L_align`** — §5.1. Reopened or closed by the first Stage-1 run.
- **`n_ground` pack collision** — pre-existing live bug (§4), fix independently.
- **OT mass collapse / `tau` as cost temperature** — deferred-ledger L-2.
- **Unreachable multiplicity target** — `ego_targets.py:177` sets
  `label = len(members)/count` unbounded (measured to 94) while `slots.mult`
  clamps at `m_max=32`, leaving `L_mult` an irreducible floor on hub nodes.
- **`L_div` weakens as slots collapse** — hinge shrinks 0.048 → 0.007 across
  h-cos 0.58 → 0.996; at `w_div=0.1` it cannot counteract collapse.
- **V_hold's gold graph is a truncated BFS prefix** (§2.1) — 33% of true degree
  retained; and the eligibility floor is 29% lower than on `V_select` alone.
