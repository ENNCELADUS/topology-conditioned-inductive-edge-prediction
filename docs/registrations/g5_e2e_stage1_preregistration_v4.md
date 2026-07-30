# G5 E2E Stage-1 two-stage-ladder pre-registration v4 — DRAFT

**Status: DRAFT. This document authorizes qualification training on the registered
`V_fit`/`V_hold` universes only. It does not authorize a formal run, candidate/test
scoring, an external held-out read, or any scientific claim.**

The machine-readable registration is
`docs/registrations/g5_e2e_stage1_preregistration_v4.json`. This Markdown twin is
explanatory only. **If the two disagree, the JSON governs.**

## What v4 is

v4 records the **design** of the rev-3.2 e2e screen — arms, metrics, seeds, test
protocol, and the two-stage ladder — and **re-pins the v2 acceptance thresholds**
for the formal stage. It invents no threshold. Every acceptance number below is
copied from the immutable BINDING v2 registration
`docs/registrations/g5_e2e_stage1_preregistration_v2.json`
(SHA-256 `7937d8bb…`), which is named as v4's predecessor and whose digest v4
pins.

The predecessor DRAFT `g5_e2e_stage1_preregistration_v3.json` (SHA-256
`1275c446…`) stays on disk as history. It was never bound, so it authorized
nothing. v4 supersedes it.

## What v4 deletes from v3

v3 declared a ladder the code can no longer run:

| v3 field | why it is gone |
|---|---|
| `prebinding_qualification.gates` (G3.1–G3.5) | the calibrate → freeze → rehearse ladder that computed them is deleted; G3.4/G3.5 were unsatisfiable as drafted (thresholds needed scoring, scoring needed `BINDING`, `BINDING` needed those thresholds) |
| `protocol.max_attempts_total = 3` | the attempt window is deleted |
| `protocol.v_qual_rehearsals = 1` | the rehearsal stage is deleted |
| `protocol.v_select = "sealed until first bound run"` | `V_select` is no longer a universe; `V_hold := V_qual ∪ V_select` serves **both** stages |
| `probe_artifact.prebinding_scopes` | `E2EProbeScope` is `formal_train` only |
| the seven unresolved-marker strings | unresolved binding evidence is now explicit JSON `null` |

**No pre-binding numeric threshold survives.** The measured Phase-0 grounding
ceilings (`top50 = 0.13952495387963418`, `top20 = 0.10728125418065595`) are
retained as *reported measurements*; the half-ceiling gate derived from them is
not.

The unresolved-marker convention is replaced by explicit `null`. That is not a
weakening: `_validate_e2e_formal_binding` (`src/train_egostitch.py`),
`_validate_e2e_binding_evidence` (`src/experiments/g5_stage1.py`) and
`_validate_e2e_scoring_provenance` (`src/score_universe.py`) each fail closed on
a `null` section, so a DRAFT cannot be mistaken for a resolved BINDING state by
any consumer.

This v4 draft uses `egostitch_e2e_binding_evidence_v2`. Version 2 preserves the v1
fields and adds mandatory `auprc_tolerance_calibration` evidence. Historical v1
registrations and artifacts remain unchanged under their original schema; they are not
rewritten or treated as v2.

## The two-stage ladder

"Qualification stage" and "formal stage" are the spec §13.19 ladder stages. They
are **not** the G5 Stage-1/2/3 architecture ladder and not the frozen-s0 G5
Stage-1 screen.

| | qualification | formal |
|---|---|---|
| trains on | full `V_fit` | full `V_fit` |
| validates on | `V_hold` | `V_hold` |
| `optim.epochs` | reduced; recorded in `qualification.json`, **not registered** | `30` |
| purpose | development loop ending in manual review | results |
| verdict | `pending_manual_review` after a complete otherwise-valid run; never automatic `pass` | the re-pinned v2 thresholds |
| test access | never opened | one scoring epoch per (arm, seed) |

Both stages train on the **identical** universe and differ only in
`optim.epochs`. That is what makes `feature_stats_sha256` and the `mean(log d)`
degree prior identical between stages by construction, so the formal preflight
compares equal values rather than propagating a digest.

**Qualification verdict — complete run, then manual review.** The actual clipping
operation is unchanged: global L2 norm `3.0` for pair/generator and `1.0` for
conditioning, with every pre-clip norm, coefficient, and low-coefficient streak
recorded. Clip coefficient `< 0.1` for ten consecutive steps no longer aborts
qualification; it remains visible telemetry and remains a hard failure in formal.
A qualification that finishes the complete reduced schedule, selects an eligible
checkpoint, and hits no other hard failure writes `pending_manual_review`, never
automatic `pass`. The one-step `< 1e-3` extreme, non-finite, family-imbalance,
logit-collapse, slot-collapse, and no-eligible-checkpoint failures are unchanged.
Named failure outcomes remain `fail(<named_guard>)`,
`training_invalid(slot_collapse)`, and `fail(no_eligible_checkpoint)`.
`pending_manual_review` fails formal preflight. This DRAFT defines neither a manual-
approval artifact nor any conversion to `pass`; an immutable attempt is not edited in
place. Formal preflight must inspect the current immutable attempt and may not fall
back from a latest pending result to an earlier `pass` artifact.

**Manual review does not weaken checkpoint eligibility.** The §13.19.3
eligibility conditions — post-ramp/Phase-C restriction, the
`prevalence + 0.02` AUPRC floor, and the residual / logit-std /
topology-gradient conditions — are retained in full at **both** stages. That
retention is what prevents selection from landing on a reconstruction-only
warm-start checkpoint, the documented 2026-07-19 v1 failure.

**Boundary audit.** Training may not open a held-out path **in any run kind** —
a path condition, not a `run_kind` condition. The prior form gated on
`run_kind != "formal"`, which left the formal stage unguarded. Mere presence of
candidate, test, or `test_graph.pkl` files in the shared repository data root is
allowed; an attempted open, including through an alias or symlink, raises before
the held-out path is read.

## Data contract (verified, must not move)

`V_fit` and every digest keyed on it are **bit-identical** to the two-holdout
construction the union replaces:

| quantity | value |
|---|---|
| `\|V_fit\|` | 7,558 |
| `\|e_msg_fit\|` | 22,708 |
| `\|e_sup_fit\|` | 6,612 |
| `v_fit_nodes_sha256` | `32f07189…` |
| `e_msg_fit_sha256` | `ca92d2eb…` |
| `e_sup_fit_sha256` | `e66ca8f6…` |

`V_hold := V_qual ∪ V_select`, recomputed through
`src/data/internal_holdout.py::derive_internal_holdout`:

| quantity | value |
|---|---|
| `\|V_hold\|` | 512 |
| positives | 1,533 = 456 + 807 + 270 cross-side |
| complete non-self pairs | 130,816 |
| prevalence | 0.01171875 |
| `nodes_sha256` | `41ba2aa6…` |
| `positive_edges_sha256` | `8b1c8ca9…` |
| `pair_labels_sha256` | `089c92ca…` |

Two consequences are **disclosed, not hidden**:

- Positives grow 1.9× while pairs grow 4×, so AP sampling SD shrinks only
  ≈`sqrt(1.9)` ≈ 1.4×.
- The `prevalence + 0.02` eligibility floor keeps its inequality form but its
  *absolute* value drops from `0.0447` to `0.0317` — a 29% reduction. Re-pinning
  the additive `0.02` would be an owner decision; v4 does not re-pin it.

`V_hold`'s gold graph is a truncated BFS prefix (degree-sum 9,390 in full
`G_msg`, 3,066 inside `E_msg[V_hold]`, 33% retained). Pre-existing; mitigated by
the union, not removed.

`auprc_tolerance` is `null` in v4 by design. The method is now frozen, but its
numerical output does not exist yet:

- **Immutable source:** the first `full`, Seed-0 qualification attempt under the
  first implementation containing this method that reaches the first validation
  after the conditioning ramp plus one complete Phase-C epoch and successfully
  writes the complete source artifact. Failed, incomplete, later, or manually
  preferred attempts cannot replace it.
- **No extra exposure:** reuse that existing validation; do not launch another
  validation and do not add another `V_hold` evaluation to K. The source attempt and
  validation remain in complete `attempt_history.json` / cumulative K exactly once.
- **Inputs:** the complete canonical non-self `V_hold` manifest — 130,816 rows,
  1,533 positives, 129,283 negatives — and the active full-model fp32 logits, with
  no sigmoid or other transform.
- **Bootstrap:** iterate replicate-major from replicate 0 through 9,999. Within each
  replicate, use the same sequential
  `numpy.random.Generator(numpy.random.PCG64(0))` to draw 1,533 positive-row indices
  with replacement first, then 129,283 negative-row indices with replacement. Build
  `y_true` by concatenating the positive sample before the negative sample, and build
  `y_score` in the identical order from the corresponding raw logits; then call
  `sklearn.metrics.average_precision_score(y_true, y_score)`.
- **Pin:** sample SD across replicate AP values with `ddof=1`, then
  `ceil(10000 * sd) / 10000`, without clamp, floor, or cap. The result is shared by
  all six trained arms.
- **Access boundary:** pair labels and active logits only. No `V_hold` topology/MMD,
  candidate/test pairs or scores, or test graph may be read.

The fixed `0.02` eligibility constants and `1e-6` MMD tie tolerance do not move.
Before binding, `binding_evidence.auprc_tolerance_calibration` must bind the complete
method/source/output artifact by path and SHA-256, including the replicate-major,
positive-draw-first, positive-concatenation-first order; the resulting scalar must be written
to `checkpoint_selection.auprc_tolerance` and all six configs, whose digests are then
re-pinned. Until that evidence exists, the value and evidence field remain `null`.

## Eight arms

Six trained checkpoints — `full`, `b0_e2e_f_only`, `pair_topology`, `p0`,
`cosine_pool` (`n_ground = 20`), `no_l_rel` (`w_rel = 0`) — plus two
scoring-time controls over `full`'s selected checkpoint:
`structure_control_6a_v3` (`shuffle_within_pair_v3`) and
`structure_control_6e_v1` (`rewire_checkerboard_v1`).

The arm count is **8**. Dropping `structure_control_6e_v1` would silently delete
the degree-preserving rewiring control that protocol §E4.16(e) calls decisive.

The six v3 config SHA-256 digests are re-pinned in `binding_evidence.configs`
as they now stand after the two-stage cleanup. Any further byte change — including
retargeting a config's `preregistration:` key at this v4 file, or re-pinning
`selection_auprc_tolerance` — moves the digest and must be re-pinned before
binding.

## Probes and artifact versions

The probe artifact is `egostitch_e2e_probe_v2` at scope `formal_train`, produced
from `full`. Π-consistency v1 and v2, per-run slot recall at `n_ground`,
shared-neighbour-count R², degree-partialled clustering R², and the four
slot-dispersion statistics are **telemetry reported at both stages**; none is a
binding gate. Older probe versions are rejected, not upgraded.

The scores-`.npz` metadata version is `egostitch_e2e_scores_v3` and the per-pair
precision contract is `egostitch_e2e_pair_fp32_v1`; both are read from the code
rather than restated by hand.

## Re-pinned v2 acceptance thresholds (formal stage)

Recorded here **before any held-out read**, per protocol §5.2.4. These were never
part of the 2026-07-27 G3.4/G3.5 circularity: that loop was
scoring → `BINDING` → thresholds for the deleted *pre-binding* gates. These were
satisfiable under v2 and are what produced the published 2026-07-24 `cut`.

**Primary criteria** (arm `full`, all must pass, at the canonical density-matched
operating point):

1. clustering-MMD ratio strictly lower than every comparator;
2. BFS-macro GS strictly higher than every comparator at that comparator's
   matched-global-RD quota;
3. BFS-macro RD strictly higher than every comparator at the same matched quota.

**Guards:**

- full-arm degree-MMD ratio ≤ `1.10` × B0;
- full-arm degree-corrected candidate AUPRC ≥ B0 − `0.02`.

**Comparators:** `b0`, `b0_cal_density`, `b0_cal_selfdensity`, `b0_cal_degseq`.
`b0_cal_selfdensity` remains the bar.

**Evaluator:** seed `0`, 500 fixed subgraphs.

**Pathway attribution:** `G_full = clu(b0_e2e_f_only) − clu(full)`,
`G_pt = clu(b0_e2e_f_only) − clu(pair_topology)`; pass requires `G_full > 0` and
`G_pt ≥ 0.25 · G_full`.

**Structure control:** paired bootstrap over the 500 fixed subgraphs with shared
resample indices, `B = 1000`, seed `0`, `alpha = 0.05`; require the 2.5th
percentile of `clu(structure_control_6a_v3) − clu(full) > 0`. This applies to
`6a-v3` **only** — v2 registered exactly one structure-control condition and v4
re-pins that one. `6e-v1` is scored, published and read alongside it but carries
no verdict inequality; adding one would be a new decision rule, which is an owner
action, not a re-pin. The single edit to v2's text is the control identity
(`structure_control_6a` → `structure_control_6a_v3`) under the spec §14.4.6
arm-schema migration.

**Verdict rule and failure reading** are carried verbatim from v2.

## Evidence class

`engineering`, **at every seed count**. `p_value`, `ci` and `holm` are `null` and
the artifact is refused before it is written if any is non-null at any nesting
depth (`_enforce_engineering_evidence_class`). Only E1/E3 carry inference
(CLAUDE.md; protocol §5.0.5), which additionally requires the spec §8 30-config
HPO-parity budget and Holm over the pre-registered held-out assembled family.
Screening several registered seeds adds cross-seed **variance reporting** only.

## Test protocol

This ladder **cannot** promise "test opened exactly once" and does not claim it:
scoring shards across every visible GPU, each shard independently reading
`candidate_test_edges.txt`, and `load_test_graph` is called at three sites. The
v2 screen spent ≈24.5 h across five candidate-scoring passes.

The enforceable property is **one scoring epoch per (arm, seed), no re-scoring
after seeing results**, recorded in an append-only test-access ledger written at
each held-out `_resolve_pairs` call and checked into the run artifacts.
Re-scoring is permitted only with a recorded reason, and it is visible.

Qualification exposure is separately exact-set bound. Each trained arm has an
`attempt_history.json` with schema `egostitch_e2e_qualification_history_v1`. At
binding, `binding_evidence.qualification_history_indexes` must map exactly the six
trained arms to `{path, sha256}` references for those indexes, and
`binding_evidence.qualification_attempts` must map the same arms to non-empty attempt
lists exactly equal to the referenced indexes' complete `attempts` arrays.
The `full` Seed-0 attempt supplying the AUPRC-tolerance calibration must occur in that
complete history and its source validation contributes to K exactly once; calibration
reuses it and creates no additional K event. No `attempt_history` schema change is
introduced.

Arm **and** checkpoint selection happen on `V_hold`, never on test.

## Disclosed residual risk

With ~20 eligible epochs × 8 arms per formal run, plus an unbounded number of
qualification-stage iterations on the same `V_hold`, the accumulated argmin count
K is large and selection noise is **not exchangeable across arms** — a
higher-capacity arm draws a larger max-of-K, which biases the
`full` vs `b0_e2e_f_only` headline contrast. This is the accepted price of
sharing one holdout across both stages. The mitigation is to **record K**: the
cumulative count of `V_hold` evaluations per arm is logged into the formal-stage
artifact, so the inflation is disclosed rather than hidden.

## Binding boundary

The governing `binding_evidence.schema_version` is
`egostitch_e2e_binding_evidence_v2`; its additional mandatory field relative to v1 is
`auprc_tolerance_calibration`. Historical v1 evidence remains unchanged.

`binding_evidence.implementation`, `parameter_group_manifests`,
`packs_and_validation_manifests`, `qualification_attempts`,
`qualification_history_indexes`, `boundary_access_audit`,
`runtime_and_peak_memory`, `auprc_tolerance_calibration`, and
`checkpoint_policy_version` are `null`;
`checkpoint_selection.auprc_tolerance` is `null`. `binding_prerequisites` in the
JSON lists exactly what must be resolved.
The formal worker rejects this DRAFT. **Only the owner may promote a resolved
successor content state to BINDING** — no agent, screen, or note may do it.
Even after a qualification completes, its `pending_manual_review` verdict is only
review evidence and formal remains fail-closed. This DRAFT deliberately specifies no
approval artifact or mutation that could turn the immutable attempt into `pass`.
