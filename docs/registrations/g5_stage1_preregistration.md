# G5 Stage-1 Pre-registration (human-readable twin)

> Machine-checked source of truth: [`g5_stage1_preregistration.json`](g5_stage1_preregistration.json).
> The training worker records that file's sha256 in `run_metadata.json` at run start;
> the gate evaluator (`src/experiments/g5_stage1.py`) refuses to open held-out metrics
> on any mismatch. This `.md` twin is explanatory only — if they ever disagree, the
> JSON governs.

## What is being gated

Stage 1 of the G5 staged EgoStitch build (spec §13): imagination + degree budget +
closure channel only — no codebook, no harmonization, no CVAE. Benchmark-A
(`breadth_first`), spec-default hyperparameters, seeds {0, 1, 2}.

## Comparators

`B0` (frozen candidate-scores artifact, checkpoint `e092537d8cf1e208`) and the three
`B0+cal` arms (`density`, `selfdensity`, `degseq`). The bar on every axis is the
**best comparator on that axis**. B1/B5 rows are deferred to E3 (protocol §5.0.5
instantiation, 2026-07-14).

## Primary criteria (Holm-corrected family, all must pass)

Headroom-weighted from the completed G3 gate — clustering-MMD carries the largest
measured headroom (Oracle-blend 3.885×), GS/local-RD the clearest Oracle-topo gains:

1. **Clustering MMD ratio** strictly below every comparator (3-seed mean), with a
   95% CI on the difference vs the best comparator excluding zero.
2. **BFS-macro GS** strictly above every comparator at matched global simple-edge RD.
3. **BFS-macro RD** strictly above every comparator at matched global simple-edge RD.

Statistics: per-seed differences, normal-approximation one-sided p-values
(bootstrap variance included for the MMD member), Holm step-down at α = 0.05.

## Guards (both must hold)

- **Degree-MMD non-regression:** ≤ 1.10× B0's degree-MMD ratio. (Oracle-topo's 0.926
  degree headroom shows pure topology signal can regress this axis.)
- **Matched edge AUPRC:** degree-corrected ratio-1 AUPRC within 0.02 of B0's
  recomputed value. Hard-negative rows disclosed, not gated at Stage 1.

## Non-binding diagnostics (required in the report)

Degree-calibration curve (E[d̂·ρ̂_eval/ρ_train] vs realized assembled degree), slot
recall@K, slot-adjacency-vs-clustering correlation, (s0, s1, s2) correlation matrix,
self/non-self split + self-loop-rate row, proj-variance trajectory, and the §4.7
FLOPs/wall-clock table (R = 0 commitment).

## Verdict

`pass` ⇒ Stage 2 (+ codebook + s3) proceeds. `cut` ⇒ the pre-registered failure
reading in the JSON is written into the gate results verbatim and the §4.6 rule cuts
the mechanisms that own no gain. Criteria are never edited after held-out metrics
are opened; §5.2 decision rules are copied verbatim in the JSON.
