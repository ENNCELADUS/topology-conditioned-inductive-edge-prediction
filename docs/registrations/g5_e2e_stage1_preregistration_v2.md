# G5 E2E Stage-1 Stability Screening Pre-registration v2

**Registration ID:** `g5-e2e-stage1-20260719-conditioned-encoder-stability-screen-v2`  
**Status:** `DRAFT`  
**Bindings:** `docs/05-egostitch-spec.md` §5, §13.16–§13.19, §14; and
`docs/03-experiment-protocol.md` §5 G5 / §5.2.

This is a prospective, post-v1 replacement. It does not amend the v1 BINDING
registration or relabel v1 artifacts. The v1 `full` arm completed only the
engineering training pipeline: its selected checkpoint came from the
reconstruction-only warm-start, later joint training produced collapsed validation
logits and non-finite fixed-replay edge-family gradients, and no candidate scores,
remaining arms, or formal G5 gate result were produced. Those train/validation
diagnostics motivate v2; they are not a held-out scientific result.

## Claim and scope

The primary claim is that a **numerically valid** E2E topology-conditioned encoder
improves assembled-graph topology over calibrated independent scoring without
more than the registered `0.02` candidate AUPRC degradation. The supporting claim is
that the topology pathway, rather than only pair/content capacity, contributes
materially to any clustering-MMD gain.

This remains a fixed-Seed-0 engineering screen. It supports neither statistical-
significance nor cross-seed-robustness claims; E1/E3 retain their multi-seed Holm
requirements. A random, collapsed, warm-start-only, or numerically unstable
checkpoint is classified as an invalid training run, not evidence for or against the
architecture.

### Disclosed qualification-infrastructure amendment — 2026-07-21

DRAFT rehearsal attempt 003 used two H20s. Its token-budget probes measured about
`80.46`, `75.81`, and `70.44` pairs/s for candidates `128`, `256`, and `512`; the
subsequent single-epoch probe had only `1,823.6` seconds left and was stopped by the
watchdog before completing an epoch. It produced no eligible checkpoint or scientific
result and remains one of the three allowed rehearsal attempts.

Because the first full-arm rehearsal attempt (qualification attempt 003) exposed a
launcher/time-accounting defect rather than a
model result, this still-DRAFT registration is explicitly amended before binding to
permit only the following failure-recovery changes: auto-detect and use every visible
H20 for the overfit test and full-arm rehearsal; rebalance the unchanged total runtime budget; remove
slower token-budget candidates while retaining measured candidate `128`; correct
batch-construction timing; and make semantically equivalent in-memory reuse changes
whose outputs are locked by exact regression tests. Model, loss, optimizer, schedule,
precision, data roles, sampler, guards, checkpoint eligibility and selection, gates,
and verdict rules remain frozen. This is a disclosed post-attempt infrastructure
amendment, not a claim that attempt 003 was prospectively registered in its corrected
form. Any change outside this list, a fourth full-arm rehearsal attempt, or any post-binding change
requires a new versioned registration.

The registered `prefetch_factor=2` is therefore applied to the manual packed-token
batch iterator with one bounded deterministic producer thread per rank. This overlaps
unchanged CPU batch construction with GPU execution; it does not change batch order,
rows, padding, seeds, tensor values, losses, or optimizer steps. Rejected completed
JSON-object worker profiles are also retained as failure evidence so the measured
bottleneck is auditable.

The consolidation port briefly omitted the already-calibrated V2 group-specific
ceilings and therefore serialized `1.0/1.0/1.0`. This DRAFT correction restores the
pair-encoder/generator/topology ceilings `3.0/3.0/1.0` actually executed by the
successful attempt-003 overfit. The clip-coefficient abort thresholds remain unchanged.

## Arms and frozen evaluation contract

The scientific comparison remains unchanged:

| Arm | Proposed v2 training config | Primary logit in `logits` artifact |
|---|---|---|
| `full` | `egostitch_e2e_breadth_first.yaml` | `full` |
| `b0_e2e_f_only` | `…_training_f_only_…` | `f_logit` |
| `pair_topology` | `…_training_pair_topology_…` | `pair_topology` |
| `structure_control_6a` | selected full checkpoint + `shuffle_within_pair` | `full` |
| `p0` | `…_training_p0_…` | `full` |

The comparator set, frozen candidate/B0 digests, operating points, five-arm
provenance, four-logit fp32 decomposition, representation probe, primary criteria,
guards, pathway-attribution rule, and structure-control bootstrap remain the v1
scientific comparison. The JSON is the machine authority: it repeats the frozen
input digests and full probe/control definitions, while unresolved v2 validation,
cost, config, pack, and qualification hashes remain explicit
`REQUIRED-BEFORE-BINDING` fields.

## Stable training contract

### Phase A — dual-track warm-start (20%)

- Train Tokenize-lite/Imagine with `L_recon`.
- Simultaneously train the pair encoder/head using `L_edge(f_logit)` while topology
  and content conditioning are hard-bypassed.
- Retain the 1:5 sampled rows but use positive weight `5.0`, normalized by the exact
  global sum of effective row weights.
- Keep `L_real` and `L_ssl` off.

This prevents model selection from comparing a randomly initialized pair scorer
against trained joint checkpoints and removes the all-negative optimum induced by the
sampling prior.

### Phase B — conditioning ramp (10%)

Use
`alpha = clip((step - ramp_start + 1) / ceil(0.10 * total_steps), 0, 1)`.
The full-logit edge loss and `L_recon` are fully active. Multiply the
topology/content parameter-group learning rate and `L_real`/`L_ssl` weights by
`alpha`. Apply the registered `p_topo = p_cont = 0.15` branch masks.

### Phase C — full joint (70%)

Use the complete §13.5 objective, registered branch masks, and full learning rates.

The pair encoder/head, generator, and topology/content conditioning modules are
disjoint and exhaustive optimizer groups; their sorted parameter names and hashes are
run metadata. Kendall scalars are frozen outside the optimizer. AdamW uses
`betas=(0.9,0.999)`, `eps=1e-8`, and weight decay `0.01`. Neural groups use peak LR
`1e-4`; pair/generator use a 500-step linear warm-up followed by cosine decay to
`1e-5`, while conditioning multiplies that base schedule by Phase-B `alpha`. Each
pair-encoder/head and generator groups are independently clipped to L2 norm `3.0`;
topology/content conditioning is clipped to `1.0` after BF16 unscale and DDP reduction.

For weighted BCE, each rank backpropagates
`world_size × sum_local(m_i w_i BCE_i) / all_reduce_sum(sum_local m_i w_i)`, with
`w_i=5y_i+(1-y_i)` and real-row mask `m_i=0` on DDP padding. This preserves the exact
global weighted mean with unequal tail batches and must match 1-GPU gradients
parameter by parameter.

Training remains BF16 for large blocks, but Sinkhorn, the gated conditioning residual
multiplication and addition, logits/BCE, and finite checks are fp32. The residual must
survive a pure-fp32 differential check before any cast.

## Mandatory numerical validity

Every real backward records the norm of each unscaled, DDP-averaged optimizer-group
gradient, clip coefficients, and non-finite element counts. Replicated DDP ranks
all-gather the fp64 local squared sums to assert equality and record their mean; they
must not sum identical replicated norms. Parameters and optimizer state are checked
after every step. Every 50 steps, separate replay backwards measure each active loss
family within the optimizer groups it actually shares, then clear replay gradients
without changing optimizer state. Disabled/null families are recorded but excluded.

Abort synchronously if:

- any loss, logit, parameter, optimizer state, gradient, family norm, or submodule RMS
  is non-finite;
- any group clip coefficient is below `1e-3` once or below `0.1` for ten consecutive
  steps;
- within a shared group, `max(active-family norm) / median(active-family norm) > 50`
  for four consecutive probes after `alpha=1` (a non-positive median is invalid);
- fixed stability-split pair-logit standard deviation stays below
  `max(25% of warm-reference std, 1e-4)` for two validations. The reference is
  measured after the final Phase-A update and must be finite and at least `1e-4`.

An aborted run publishes failure metadata only. It may not publish `best.pt`,
`complete.json`, candidate scores, or gate inputs. Any available `last.pt` is
diagnostic-only under a failure-specific directory. Non-finite values may not be
serialized away in a manner that permits success.

## Checkpoint eligibility and selection

Before binding, two deterministic connected 256-node BFS holdouts are carved from
`E_msg` as node-disjoint `V_qual/V_select`; `V_fit` is the remainder. Seeds and
frontiers use registered hash prefixes, with `V_qual` removed before constructing
`V_select`. Training uses only induced `E_msg[V_fit]` and
`E_sup` pairs wholly inside `V_fit`; cross-bucket edges are quarantined. Complete
non-self pair universes and gold graphs for qualification/formal selection come only
from the held-out `E_msg[V_qual]` and `E_msg[V_select]`, respectively—not
`train_graph.pkl`, `val_edges.txt`, or `E_sup`. Counts, prevalence, hashes, and zero
node/label-edge overlap are binding. Rehearsal measures its reference and topology
metric only on `V_qual`; the first bound run opens untouched `V_select`. These are
internal model-selection metrics, never external held-out evidence.

Warm-start and ramp epochs are never eligible. Eligibility begins only after one
complete full-joint epoch.

The full and `p0` arms must satisfy:

1. all finite/gradient guards pass;
2. `std(f_logit) >= max(0.25 × warm-reference std, 1e-4)`;
3. warm-reference AUPRC is at least prevalence `+0.02`, and
   `AUPRC(full) >= warm-reference AUPRC(f_logit) − 0.02`;
4. `std(full−f_logit) / max(std(f_logit), 1e-12) >= 1e-3`;

The existing conjunctive liveness rule remains a separately recorded preflight; item
4 already excludes its `<1e-5` residual-death branch.

`b0_e2e_f_only` requires finite gradients, `std(f_logit)>=1e-4`, and AUPRC at least
prevalence `+0.02`. `pair_topology` uses the analogous conditions on its active logit
plus a topology-conditioning fixed-replay gradient norm `>=1e-8`; it is not compared with the
full arm's warm-reference AUPRC because removing content is the intended ablation.

For each arm, retain eligible epochs within `0.02` AUPRC of its best eligible epoch,
then choose the lowest train-side validation clustering metric. With one 256-node gold
graph per side, this metric is the raw biased squared MMD between the single predicted
and gold clustering-coefficient histograms under the pinned existing histogram bins
and RBF-kernel configuration; it is not the external multi-reference normalized MMD
ratio. Within raw `MMD^2` tolerance `1e-6`, choose lower unweighted Brier score and
then the earlier epoch. This makes selection topology-aware without silently trading
away edge quality. If no epoch is eligible,
the run is invalid. There is no fallback to epoch 1, `last.pt`, or the numerically
least-bad checkpoint.

## Pre-binding qualification — train/validation only

Before changing this registration to `BINDING`:

1. A deterministic 510-row 1:5 overfit set—85 smallest-hash `E_sup[V_fit]` positives
   and 425 registered-sampler negatives with both endpoints in `V_fit`—is cycled for exactly 2,000 optimizer steps under
   the formal weighting/schedule (Phase A/B/C=`400/200/1400`). Its pair manifest is
   created once before sharding and is rank/world-size invariant. It uses every
   auto-detected visible H20 (four on the current target host), matching the
   rehearsal/formal launch style, and records the detected world size. It must reach
   train AUPRC `>=0.95`, residual ratio `>=1e-3` after the ramp, and pass every
   applicable stability guard.
2. The exact full-arm 30-epoch config must complete using every auto-detected visible
   H20 (four on the current target host), matching the formal launch style, using only
   the qualification pair/topology manifests and selecting an eligible post-ramp
   checkpoint. The detected world size is recorded and the formal
   checkpoint-selection manifests remain unread.
3. At end-ramp and the selected checkpoint, the same eval-mode replay and hard masks
   must compare BF16+fp32 islands with pure fp32. Full/f-only meet the elementwise
   tolerance; residual relative L2 is `<=1e-3`, correlation is `>=0.999`, and neither
   residual is all zero.
4. Qualification runs use a data root without candidate/test manifests or
   `test_graph.pkl`. An access log proves training endpoints are within `V_fit`, no
   `V_qual`/`V_select`/`V_test` feature row is read by a training step, structural
   training targets equal loop-stripped `E_msg[V_fit]`, held-out message edges and
   `E_sup`/validation positives never enter topology-training targets, and no
   candidate/test score or assembly occurs. The unchanged
   global-positive rejection set may use validation-positive membership only to avoid
   false-negative sampling; it is disclosed and identical across arms.
   Grounding caches are separately hashed for `V_fit`, `V_qual`, `V_select`, and the
   external test side; rehearsal cannot read a `V_select` feature row.
5. A schema-validated `binding_evidence` object records implementation, four configs,
   parameter groups, packs, validation manifests, every qualification attempt,
   artifacts, access audit, runtime, peak memory, and checkpoint-policy hashes.
6. An independent review must confirm JSON/prose parity and that no v1 artifact is
   presented as v2 evidence.

The clipping and family-ratio thresholds cannot bind from intuition alone: a passing
rehearsal requires clip-coefficient `p1>0.12`, minimum `>0.0012`, and fewer than ten
consecutive steps below `0.1`, plus family-ratio `p99<40`. At most three full-arm
rehearsal attempts are allowed and every attempt is
retained. Except for the disclosed 2026-07-21 failure-recovery amendment above, the
first full-arm rehearsal pass freezes implementation/config; a fourth full-arm
rehearsal attempt or any later change outside
that amendment requires v3. No candidate/test artifact may be read during
qualification.

## Cost-aware formal execution order

1. Qualify and bind v2.
2. Train `full` only.
3. Run its checkpoint-eligibility and validation-liveness preflight.
4. Stop if full is invalid.
5. Train `b0_e2e_f_only`, `pair_topology`, and `p0`.
6. Require an eligible checkpoint and matching registration/config hashes for every arm.
7. Only then score candidate artifacts, produce the representation probe, and run G5.

This changes compute exposure, not the scientific comparison or verdict inequalities.
Candidate/test scoring must verify a BINDING registration plus formal, complete,
eligible run metadata and exact registration/config/checkpoint hashes. DRAFT/debug
checkpoints are limited to train/validation sources even if a checkpoint file exists.
Each formal config path must equal its JSON `arms.<arm>.training`; v1 configs are
historical reproduction only.

## Verdict and failure readings

The final pass rule remains: all training arms valid; full strictly dominates every
comparator on clustering-MMD, matched BFS-macro GS, and matched BFS-macro RD; both
guards pass; liveness passes; `G_full>0`; the 25% pathway-attribution share and
structure control pass.

- **Training invalid:** the optimization process did not produce a numerically valid,
  eligible checkpoint; no held-out architecture conclusion is permitted.
- **Primary failure:** the stable topology-conditioned encoder does not beat calibrated
  independent scoring at this stage.
- **Guard failure:** the model exceeds the registered AUPRC-degradation or degree-MMD
  safety guard.
- **Attribution failure:** no gain over matched f-only was established, or the
  topology share was insufficient; withdraw the topology-representation headline.
- **Structure-control failure:** the registered control did not establish that intact
  relational connectivity is necessary beyond retained topology-derived node features.

Classification is deterministic: `training invalid` is evaluated first and permits
no scientific verdict; otherwise every applicable primary, guard, attribution, and
structure-control failure is reported, and any such label yields `cut`.

## REQUIRED-BEFORE-BINDING

- Implement and test every §13.19 schedule, precision, finite-check, checkpoint-
  eligibility, and publication rule.
- Enforce candidate/test provenance checks and schema-validate structured binding
  evidence before DDP startup.
- Create and hash all four v2 training configs.
- Create and hash the disjoint validation and train-side topology-selection manifests.
- Complete the deterministic overfit test and full-arm rehearsal with auto-detected
  all-visible-H20 execution in a data root where candidate/test inputs are not mounted.
- Record implementation, optimizer-group, profile, pack, validation, access-audit,
  config, and every qualification-attempt digest.
- Complete independent JSON/prose and provenance review.

Formal workers must reject this file while it is `DRAFT` or any
`REQUIRED-BEFORE-BINDING` marker remains.
