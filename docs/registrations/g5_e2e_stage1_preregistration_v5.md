# G5 E2E Stage-1 rev-3.2 registration v5 (component-ablation schema)

Status: `DRAFT` (descriptive only). The JSON twin is authoritative. Status and
nullable run-evidence placeholders do not authorize or block formal execution.

## What changed from v4

The trained-arm schema migrates to the spec §14.4.6 v5 component-ablation set.
`cosine_pool` retires from the trained set (the Phase-0 measured slot-recall
ceilings — top-50 `0.1395` vs top-20 `0.1073` — already bound the pool-width
effect; its top-20 pack and caches stay on disk as historical evidence). A new
`row_layernorm` arm — identical to `full` except
`feature_standardization: row_layernorm` — gives the rev-3.2 D0 per-dimension
z-scoring mechanism the ablation arm the proposal §4.6 anti-grab-bag rule
requires. Every acceptance threshold, guard, comparator, evaluator setting,
and claim-limit rule is carried over unchanged (v4 → v2 lineage). This v5 is
the owner-directed 2026-07-31 replacement snapshot: it also binds the repaired
validation/cache/precision execution path described below; no earlier v5 bytes
remain authoritative.

## Active execution contract

One formal training stage. It trains the six registered arms on `V_fit`,
selects checkpoints on `V_hold`, then scores the two registered scoring-time
controls over `full`'s checkpoint. There is no qualification stage, calibration
prerequisite, or preliminary-run authorization path. Formal launch verifies
only the exact experiment plan and runtime boundary:

- registration ID and byte-unchanged SHA-256;
- clean implementation checkout and exact arm/config path plus SHA-256;
- correct repository/runtime boundary and exactly four visible H20s.

## Validation and cache execution contract

Each rank keeps the existing rank-strided, padded validation pair shard, but
encodes every unique endpoint node only once per validation event. Endpoint
states are reconstructed for `AB`, `BA`, and self pairs from an event-local
cache. Raw-token and generator encoding may use BF16 autocast; every floating
cached `E2ENodeState` field is promoted to FP32, while lengths and global
grounding IDs retain their integer identity. Pair-context construction and all
full/null/active-arm logits run with autocast disabled in FP32.

Validation uses `torch.no_grad()` in eval mode and restores the model's previous
training mode. `torch.inference_mode()` is forbidden here: under an enclosing
autocast context it can seed the autocast weight cache with inference tensors
that make the following training backward pass fail. `step_0`, `phase_a_end`,
and every `epoch_end` remain separate real validation executions and ledger
events even when two events share an optimizer step.

The DDP contract requires identical collective order on every rank, exact
ordered global `V_hold` coverage after padded-row deduplication, and timing for
the slowest-rank node-cache encode, pair scoring, and rank-zero gather/metrics
phases. The Linux gate runs a real two-rank, five-row non-divisible shard and
compares it with serial validation within FP32 numerical tolerance.

F0 caches are exact-universe artifacts. Training requires the exact ordered
`V_fit ∪ V_hold` universe; scoring requires the exact ordered requested scoring
universe. Superset or reordered caches fail closed. Validation endpoint and
grounding reads remain `V_hold`-only, and training remains `V_fit`-only.

The two warm-reference quantities have distinct registered snapshot points:
`warm_reference_std` is measured immediately after the final Phase-A update;
`warm_reference_auprc` is measured at the first validation after conditioning
activates, as frozen spec §14.4.3 requires. Its threshold remains
`>= prevalence + 0.02`.

## Registered arms (each owns one mechanism axis)

| Arm | Delta from `full` | Mechanism it ablates |
| --- | --- | --- |
| `full` | — | complete rev-3.2 model |
| `b0_e2e_f_only` | `permanent_null: all_head` | conditioning as a whole (matched content-only baseline) |
| `pair_topology` | `permanent_null: content_head` | content pathway (isolates topology conditioning) |
| `p0` | `p_topo = p_cont = 0` | stochastic branch dropout |
| `no_l_rel` | `w_rel = 0` | train-only relational auxiliary loss |
| `row_layernorm` | `feature_standardization: row_layernorm` | D0 per-dimension V_fit z-scoring |

Scoring-time controls over the `full` checkpoint: `structure_control_6a_v3`
(within-pair slot permutation) and `structure_control_6e_v1` (degree-preserving
checkerboard rewiring) — together they cover necessity of intact generated
structure without extra training runs.

## Observation plan for the formal `full` run (telemetry, never gates)

1. **Generator clipping.** The 2026-07-29 degree-prior-init fix left two
   razor-thin margins: worst persistent-streak step `29.218` against the
   norm-30-equivalent streak threshold (2.6%), peak pre-clip norm `2835.5`
   against the norm-3000-equivalent immediate threshold (5.5%). Read
   `profile.json → optimizer_step_gradients` (per-step per-group pre-clip norm
   + `clip_coefficient`), `gradient_norm_series` (fixed-replay family norms and
   ratios), and `quality_guard_events` kinds `optimizer_gradient` /
   `family_gradient`.
2. **Slot collapse.** rev-3.2's D0 fix should hold init h-cosine near `0.62`
   (row-LayerNorm measured `0.9897`, above the `0.95` predicate). Read
   `profile.json → initial_slot_health`, per-validation `history[].fidelity`
   (`h_pairwise_cosine_mean`, `plan_rank1_marginal_residual`),
   `history[].quality_thresholds.slot_collapse`, and `quality_guard_events`
   kinds `initial_slot_collapse` / `slot_collapse`. The `row_layernorm` arm is
   the registered end-to-end contrast.
3. **End-ramp precision differential.** BF16-trunk/fp32-island logits must sit
   inside relative-L2 `0.05` on `full`, `f_logit`, and their residual, with
   residual correlation ≥ `0.999`. Read
   `profile.json → precision_differential.end_ramp` and `.selected`, plus
   `quality_guard_events` kinds `end_ramp_precision` / `selected_precision`.
   Only a non-finite measurement aborts (truthfulness); finite misses record
   and continue.

`python -m src.experiments.observe_e2e_formal <output_dir>` prints all three
reports from a completed run's artifacts (read-only analysis, never a gate).

## Launch sequence (operator, on the H20 container)

```bash
hpc/qualification.sh formal full           # the one full-model run; observe 1-3
hpc/qualification.sh formal f_only
hpc/qualification.sh formal pair_topology
hpc/qualification.sh formal p0
hpc/qualification.sh formal no_l_rel
hpc/qualification.sh formal row_layernorm  # D0 ablation; expect collapse telemetry
```

`full` first is an operational recommendation only (its telemetry decides
whether the razor-thin clip margins deserve an owner-side plan amendment
before burning the other five arms); it is not an execution gate.

## Quality telemetry and fail-closed boundary

Eligibility, liveness, slot-collapse indicators, clipping/family/submodule-RMS
margins, AUPRC floors, dispersion, and precision differentials are telemetry
only. Truthfulness and artifact-integrity failures still fail closed:
non-finite tensors or optimizer state, DDP disagreement, incomplete or
duplicate coverage, data-boundary violations, malformed or hash-mismatched
plan inputs/outputs, checkpoint/score provenance mismatch, and I/O or
infrastructure failure.

## Evidence class and reporting

This Stage-1 screen is engineering evidence at every seed count. No
significance or cross-seed-robustness claims. Edge-level and assembled-graph
metrics are always reported together; the three MMD ratios are never
aggregated and never called "graph similarity".
