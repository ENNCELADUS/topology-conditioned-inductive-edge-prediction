# G5 E2E Stage-1 Stability Screening Pre-registration v2

**Registration ID:** `g5-e2e-stage1-20260719-conditioned-encoder-stability-screen-v2`  
**Status:** `BINDING` (promoted 2026-07-23; owner decision, all pre-binding items resolved)  
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
result and is retained as engineering evidence; it predates and does not count
against the 2026-07-22 replacement's three-attempt allowance.

Because the first full-arm rehearsal attempt (qualification attempt 003) exposed a
launcher/time-accounting defect rather than a
model result, this still-DRAFT registration is explicitly amended before binding to
permit only the following failure-recovery changes: auto-detect and use every visible
H20 for the overfit test and full-arm rehearsal; rebalance the unchanged total runtime budget; remove
slower token-budget candidates while retaining measured candidate `128`; correct
batch-construction timing; and make semantically equivalent in-memory reuse changes
whose outputs are locked by exact regression tests. Model, loss, optimizer, schedule,
data roles, sampler, guards, checkpoint eligibility and selection, gates, and verdict
rules remain frozen; precision is changed only by the separate disclosed replacement
below. This is a disclosed post-attempt infrastructure
amendment, not a claim that attempt 003 was prospectively registered in its corrected
form.

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

### Disclosed pre-binding precision replacement — 2026-07-22

The attempt-005 full-arm rehearsal reached the end-ramp differential and failed before
checkpoint selection or any held-out/candidate/test scoring. Root-cause replay used
only the retained Stage-2 `V_fit` checkpoint and the same fixed `V_fit` batch. It
showed that the conditioned pair readout and final linear head still ran under BF16
autocast, despite the registered fp32-logit contract, and the mixed-path residual was
rounded entirely to zero. After moving the conditioned pair readout and logits into
fp32 islands, the exact four-H20 replay passes full/f-only elementwise tolerances and
measures residual relative L2 `0.0127555` with correlation `0.999752`.

Because no V2 qualification has passed and this registration remains DRAFT, this file
is directly replaced before binding: the residual relative-L2 ceiling is `0.05`,
calibrated only from that V_fit engineering replay. The full/f-only elementwise
tolerances, correlation floor `0.999`, non-zero rule, architecture, losses, optimizer,
production schedule, data roles, sampler, other guards, checkpoint policy, gates, and
verdict rules are unchanged. Earlier failed attempts remain retained as engineering
evidence but do not count against the replacement's at-most-three full-arm rehearsal
allowance. Any fourth replacement rehearsal, change after its first pass, or
post-binding change requires a new versioned registration.

The first four-H20 acceptance probe of the corrected fp32 readout measured `96.6 GiB`
peak memory and `285.04 s` for the production-prefix epoch because autograd retained
that island's activations. Training therefore activation-checkpoints only the same
fp32 readout and recomputes it during backward; evaluation remains direct. Exact
output/gradient tests lock numerical equivalence, so this is an implementation-memory
correction rather than a model, loss, optimizer, schedule, or guard change.

### Disclosed elementwise recalibration and vector-tolerance replacement — 2026-07-22

Replacement rehearsal attempt 1 failed only the end-ramp elementwise logit check
(max abs error `~0.0176` against `atol 1e-5 + rtol 1e-3·|logit|`) while the residual
contract passed with wide margin. The bound was rtol-dominated and
logit-magnitude-dependent, so the elementwise `atol` was recalibrated to the
end-ramp-measured `0.05`, and the overfit residual floor was aligned with the
same-day fp32-calibrated `1e-6` post-ramp floor with latest-qualifying-epoch
retention (the BF16-era `1e-3` floor measured readout quantization noise).
Replacement rehearsal attempt 2 then completed all 30 epochs with an eligible
checkpoint and passed the end-ramp differential, but failed the selected-checkpoint
per-element conjunct alone (max abs `0.1045` vs `0.05`) with a healthy residual
(relative L2 `0.0161`, correlation `0.99983`). Two successive single-point `atol`
calibrations were each invalidated by the next measurement context: per-element
max-abs against a BF16 trunk is an extreme-value statistic that grows with training
scale, so the contract form — not the constant — was wrong. The per-element
full/f-only tolerance is therefore replaced by vector relative-L2 `<= 0.05` per
logit stream versus pure fp32, provably slack whenever the residual bound holds
while still catching common-mode and gross single-element corruption; per-element
max-abs errors remain logged diagnostics. Attempts 1 and 2 consumed two of the
three allowed replacement rehearsals.

### Disclosed clip-margin calibration — 2026-07-22

Replacement rehearsal attempt 3 completed the full 30-epoch schedule with every
in-run gate green (no stability guard fired, eligible epoch-16 checkpoint with
liveness pass, both precision differentials passed under the vector bounds). The
post-run margins validator failed solely on the scaffold-era global
clip-coefficient `p1 > 0.12` band, which predates any completed V2 rehearsal and
was pinned DRAFT in the spec until a passing rehearsal recorded its empirical
distribution. The calibrated per-group `p1` floors are `pair_encoder_head > 0.04`,
`generator > 0.01`, `topology_content_conditioning > 0.15` (unlisted groups keep
`0.12`), set roughly 2.5–3× below the measured distribution; all other margins and
every in-run abort threshold are unchanged. Re-validating the retained attempt-3
profile under the calibrated floors passes (`qualification_margins.json`
status `pass`); no new rehearsal was launched and no attempt was consumed. The
generator trains in a persistently-clipped regime (median coefficient `0.226`
against ceiling `3.0`) — a disclosed trajectory property protected in-run by the
unchanged immediate/persistent clip aborts and family-ratio guard.

### Disclosed binding-mechanics amendment — 2026-07-23

The binding review found the formal worker's commit-identity check mechanically
impossible to satisfy as written: it required a clean checkout whose HEAD begins
with the recorded implementation commit, while this registration is itself a
tracked file — a tracked registration cannot contain its own promotion commit's
hash. The check now also accepts a clean HEAD that descends from the recorded
implementation commit exclusively through commits touching `docs/registrations/`
paths. The qualification code path never invokes this check; no training,
precision, guard, eligibility, scoring, or verdict semantics change; unit tests
lock the equal-HEAD, registration-only-descent, non-ancestor, and
non-registration-diff cases. No rehearsal was launched and no attempt consumed.

### Disclosed branch-dropout rank correlation — 2026-07-23

External branch review (2026-07-22) found that the frozen V2 per-pair
branch-dropout mask realization derives its randomness without the DDP rank, so
per-step mask draws are correlated (identical per local batch index) across
ranks. Expected per-pair rates `p_topo = p_cont = 0.15` are unchanged and the
property is identical across all four training arms. Owner decision 2026-07-23:
accepted as-is for this screen — correcting it would invalidate the qualified
attempt-3 trajectory and consume the remaining rehearsal allowance; any fix is
deferred to a new versioned registration (v3) or the E1/E3 multi-seed builds.

## Arms and frozen evaluation contract

The scientific comparison remains unchanged:

| Arm | v2 training config | Primary logit in `logits` artifact |
|---|---|---|
| `full` | `configs/egostitch_e2e_breadth_first.yaml` | `full` |
| `b0_e2e_f_only` | `configs/egostitch_e2e_f_only_breadth_first.yaml` | `f_logit` |
| `pair_topology` | `configs/egostitch_e2e_pair_topology_breadth_first.yaml` | `pair_topology` |
| `structure_control_6a` | selected full checkpoint + `shuffle_within_pair` | `full` |
| `p0` | `configs/egostitch_e2e_p0_breadth_first.yaml` | `full` |

The comparator set, frozen candidate/B0 digests, operating points, five-arm
provenance, four-logit fp32 decomposition, representation probe, primary criteria,
guards, pathway-attribution rule, and structure-control bootstrap remain the v1
scientific comparison. The JSON is the machine authority: it repeats the frozen
input digests and full probe/control definitions, and — as of 2026-07-23 — pins
every previously unresolved validation, cost, config, pack, and qualification
hash in its `binding_evidence` and `checkpoint_selection` blocks.

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
multiplication and addition, the final conditioned pair readout, logits/BCE, and finite
checks are fp32. The residual must survive a pure-fp32 differential check before any
cast.

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
   train AUPRC `>=0.95`, residual ratio `>=1e-6` at one or more post-ramp validation
   epochs (fp32-calibrated 2026-07-22; the latest qualifying epoch supplies the
   retained checkpoint), and pass every applicable stability guard.
2. The exact full-arm 30-epoch config must complete using every auto-detected visible
   H20 (four on the current target host), matching the formal launch style, using only
   the qualification pair/topology manifests and selecting an eligible post-ramp
   checkpoint. The detected world size is recorded and the formal
   checkpoint-selection manifests remain unread.
3. At end-ramp and the selected checkpoint, the same eval-mode replay and hard masks
   must compare BF16+fp32 islands with pure fp32. Full and f-only each meet vector
   relative-L2 `<=0.05` versus pure fp32 (per-element max-abs errors are logged
   diagnostics only); residual relative L2 is `<=0.05`, correlation is `>=0.999`,
   and neither residual is all zero.
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
rehearsal requires per-group clip-coefficient `p1` floors calibrated from the first
completed replacement rehearsal (`pair_encoder_head>0.04`, `generator>0.01`,
`topology_content_conditioning>0.15`, unlisted groups `>0.12`), minimum `>0.0012`,
and fewer than ten
consecutive steps below `0.1`, plus family-ratio `p99<40`. At most three full-arm
rehearsal attempts are allowed for the 2026-07-22 replacement and every attempt is
retained. Earlier failed V2 attempts remain retained but do not count against this
replacement allowance. Except for the disclosed 2026-07-21 infrastructure amendment
and 2026-07-22 precision replacement above, the first replacement rehearsal pass
freezes implementation/config; a fourth replacement rehearsal or any later change
outside those amendments requires v3. No candidate/test artifact may be read during
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

## Pre-binding qualification record (all items resolved 2026-07-23)

Every previously required pre-binding item is resolved; the JSON
`binding_evidence` and `required_before_binding` blocks are the machine record.
Summary of the passing evidence (qualification attempt `attempt005-vectol`,
implementation commit `928763af…`, auto-detected 4 × H20):

- **Overfit test:** 2,000 steps over the fixed 510-row manifest; train AUPRC
  `~1.0` (floor `0.95`); post-ramp residual ratio `~0.10` (floor `1e-6`); latest
  qualifying epoch 30 retained; no guard fired. Total `4,043.6 s`.
- **Full-arm rehearsal:** all 30 epochs, `2,340/2,340` optimizer steps, no
  stability guard fired, eligible epoch-16 checkpoint (`424e6024d2e3b893`) with
  liveness pass. Total `8,642.6 s`; peak memory `64.45 GiB` per rank (of
  `95.58 GiB`).
- **Precision differentials:** end-ramp full/f-only vector relative L2
  `0.00278/0.00278`, residual L2 `0.00193`, correlation `0.999988`;
  selected-checkpoint `0.00231/0.00226`, residual L2 `0.00494`, correlation
  `0.999966` — all within the registered bounds.
- **Margins:** re-validated retained profile passes the calibrated per-group
  floors (`qualification_margins.json` status `pass`); family-ratio p99
  `16.73 < 40`.
- **Boundary audit:** access log proves training endpoints ⊂ `V_fit`, structural
  targets equal loop-stripped `E_msg[V_fit]`, forbidden candidate/test files
  absent, zero node/label-edge overlap across `V_fit`/`V_qual`/`V_select`.
- **Manifests:** `V_fit` 7,558 nodes / 22,708 message / 6,612 supervision edges;
  `V_qual` 256 nodes, 32,640 pairs, 456 positives; `V_select` (untouched) 256
  nodes, 32,640 pairs, 807 positives. Recomputed digests equal the retained
  audit digests.
- **Provenance:** the v1 formal full-arm artifacts were archived unmodified to
  `outputs/egostitch_e2e_stage1/full_v1_20260718/`; no v1 artifact is presented
  as v2 evidence. Independent adversarial review (separate agent context,
  2026-07-23) confirmed JSON/prose parity and provenance separation.

Formal workers reject this file unless its status is `BINDING`, every pre-binding
marker string is resolved, and the live configs, artifacts, and implementation
commit match `binding_evidence` from a clean checkout (registration-document-only
descent permitted per the 2026-07-23 binding-mechanics amendment).
