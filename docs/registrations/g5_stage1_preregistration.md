# G5 Stage-1 Membership-Normalized Screening Registration (human-readable twin)

> **Retained evidence, retired code.** The frozen-s0 `egostitch` implementation this
> registration governed last exists at commit `dcae090`; it is deleted in the current
> two-stage-cleanup worktree pending a cleanup commit
> (`docs/superpowers/specs/2026-07-29-egostitch-e2e-two-stage-cleanup.md` §6.2). The
> registration is kept verbatim as the evidence record for
> `docs/results/G5-stage1-seed0-20260717.md`; it can no longer be executed, and the
> source paths it names resolve through git history at `dcae090`.

> Machine-checked source of truth: [`g5_stage1_preregistration.json`](g5_stage1_preregistration.json).
> The training worker records that file's sha256 in `run_metadata.json` at run start;
> the gate evaluator (`src/experiments/g5_stage1.py`) refuses to open held-out metrics
> on any mismatch. This `.md` twin is explanatory only — if they ever disagree, the
> JSON governs.

## What is being gated

Stage 1 of the G5 staged EgoStitch build (spec §13): imagination + degree budget +
closure channel only — no codebook, no harmonization, no CVAE. Benchmark-A
(`breadth_first`), spec-default hyperparameters, fixed Seed 0.

This replacement was adopted after the earlier exact-quota Seed-0 diagnostic was
inspected. It is a post-observation engineering-screen contract, not a retroactive
pre-registration; the completed diagnostic remains bound to its superseded hash.
The new experiment ID is `g5-stage1-20260716-membership-normalized-screen-v2`.

## Registered repair and validity instrumentation

Only the `s1` membership kernel operands are L2-normalized; raw `proj(x)` remains
unchanged elsewhere and `s2/s2_aa` are not redesigned. A fixed post-warm-start probe
logs weighted family gradient norms every 50 steps and activates pre-instantiated
Kendall weights only after a >10×-median imbalance persists for 1,000 steps. The
runtime probe aborts at `|mean(s1)| > 1000`; every validation epoch records channel
scales, residual/s0 scale, Kendall rank mobility, and top-1% overlap.

Before topology evaluation, a run is invalid as dead only when residual/s0 std ratio
is below `1e-5`, Spearman versus s0 exceeds `0.9999`, and top-1% overlap exceeds
`0.9999` simultaneously. This is not a success criterion. Validation AUPRC remains
the checkpoint-selection primary; within `1e-4`, residual/s0 std ratio breaks the tie.
Hard-negative exposure is explicitly excluded from this revision.
The residual guard takes a separately supplied fresh-fp32 B0 candidate artifact from
checkpoint `e092537d8cf1e208`; it does not subtract the historical quantized comparator
deliverable.

## Comparators

`B0` (frozen candidate-scores artifact, checkpoint `e092537d8cf1e208`) and the three
`B0+cal` arms (`density`, `selfdensity`, `degseq`). The bar on every axis is the
**best comparator on that axis**. B1/B5 rows are deferred to E3 (protocol §5.0.5
instantiation, 2026-07-14).

## Primary criteria (single-seed point estimates, all must pass)

Headroom-weighted from the completed G3 gate — clustering-MMD carries the largest
measured headroom (Oracle-blend 3.885×), GS/local-RD the clearest Oracle-topo gains:

1. **Clustering MMD ratio** strictly below every comparator.
2. **BFS-macro GS** strictly above every comparator at matched global simple-edge RD.
3. **BFS-macro RD** strictly above every comparator at matched global simple-edge RD.

No inferential acceptance procedure is used with one seed. P-values, CI pass flags,
and Holm decisions are reported as not applicable. This Stage-1 screen cannot support
statistical-significance or cross-seed-robustness claims; E1/E3 retain at least three
seeds and Holm correction.

## Guards (both must hold)

- **Degree-MMD non-regression:** ≤ 1.10× B0's degree-MMD ratio. (Oracle-topo's 0.926
  degree headroom shows pure topology signal can regress this axis.)
- **Matched edge AUPRC:** degree-corrected ratio-1 AUPRC within 0.02 of B0's
  recomputed value. Hard-negative rows disclosed, not gated at Stage 1.

## Non-binding diagnostics (required in the report)

Degree-calibration curve (E[d̂·ρ̂_eval/ρ_train] vs realized assembled degree), slot
recall@K, slot-adjacency-vs-clustering correlation, (s0, s1, s2) correlation matrix,
self/non-self split + self-loop-rate row, proj-variance trajectory, and the §4.7
FLOPs/wall-clock table (R = 0 commitment), plus the per-epoch channel/rank-mobility
series and per-family gradient-norm/Kendall state embedded in run metadata.

## Verdict

`pass` ⇒ Stage 2 (+ codebook + s3) proceeds. `cut` ⇒ the registered failure reading
in the JSON is written into the gate results verbatim and the §4.6 rule cuts the
mechanisms that own no gain. All three primary point-estimate dominance checks and
both guards must pass.
