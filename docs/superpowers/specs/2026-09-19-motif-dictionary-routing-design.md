# Shared motif dictionary, sequence routing and task-head adaptation

**Date:** 2026-09-19. **Version:** v1.
**Status:** approved and implemented on 2026-09-19; local review/checks complete, three-container launch in progress.
Implementation: [plan](../plans/2026-09-19-motif-dictionary-routing.md).
**Source checkout:** `63061a9`, branch `codex/motif-graph-grit-prompt`.

This is the next architecture experiment after the three motif-prompt repair waves. It defines three
parallel investigations, not a claim that the dictionary is already the selected method. The owner
selected: dictionary-oracle, sequence-routing and head-only lanes in parallel; frozen-reader token
space for the dictionary targets; and provisional adoption on validation evidence followed by replication.
Numerical settings below are first-round design defaults, not literature-derived optima.

This document supplements the [motif-graph specification](2026-09-16-motif-graph-grit-prompt-design.md)
for the new experiments. Historical models, checkpoints and results retain their meanings. The task
remains exactly `(x_u, x_v) -> edge(u, v)`: a shared structural prior is intermediate context for deciding
the queried edge, never an additional deployment input or a graph-generation deliverable.

## 1. Evidence and the question this round answers

Sources: [wave-1 verdict](../../results/motif_prompt_verdict/README.md),
[wave-2 repair design](../../tmp/2026-09-18-motif-stage2-fix-plan.md),
[wave-3 results, especially section 7](../../results/motif_prompt_wave3/README.md), and Claude CLI
session `0955283d-6bb4-4463-8318-b321f7c7c832` through its September 19 hand-back. The wave-2 document is a
historical plan, not evidence that all its proposed continuations ran. Wave-3 section 7 supersedes the
earlier statements in that result note that the density control was missing.

| Wave | Observed change | What the observation establishes |
|---|---|---|
| 1 | True-template Stage I: test AUPRC 0.810 vs `prefix_base` 0.744; deployable Stage II about 0.745 | The true structural channel can help; the generator was close to a constant with saturated closure gates and weak structural gradients. |
| 2 | Closure magnitude initialization helped; raw closure supervision plus that initialization encouraged collapse; graph-only warm-up isolated the structural gradient | Initialization alone was insufficient. Sparse targets and indistinguishable slots remained problems. |
| 3 | Query residual, unnormalised value path and closure-balanced graph-loss rows worked together | At selected epoch 2, graph transplantation raises validation graph loss by 39.1%; predicted prompts raise V_val AUPRC from 0.8136 with gates off to 0.8176. |
| 3 attribution | Test AUPRC 0.7418 vs gates off 0.7446; at matched output density, `prefix_base` has higher GS and comparable or better MMD | Pair dependence is now measurable, but held-out edge or shape benefit has not been demonstrated. |

The joint and detached continuations were performed on the **previous warm8 generator**. They warn
against that training handoff, but do not establish that every adaptation of the final repaired
generator fails. Nor does using frozen sequence states establish an information-theoretic ceiling:
different supervision and decision functions can improve prediction with the same task input.

The new hypothesis is that predicting a mixture of fixed, graph-derived prompt representations gives
a more useful handoff than predicting anonymous edge-weight profiles and interpreting the resulting
soft graph through a nonlinear reader. Head adaptation is a separate hypothesis and is tested on the
final repaired graph generator before combining it with the dictionary route.

## 2. Alternatives and the selected comparison

Three approaches were considered: dictionary replacement before any head adaptation; completing a
final-wave3 adaptation experiment first; and replacing the handoff while simultaneously opening the
head. The owner instead chose three container lanes that separate representation compression,
sequence-to-prompt prediction, and task readout. Dictionary construction is their only shared setup
dependency; its measured success is not a prerequisite for starting the router.

| Lane | Input to the existing four-field prefix | Parameters trained | Main question |
|---|---|---|---|
| D: dictionary oracle | True-template-derived weights over the fixed dictionary | None | How much oracle utility survives this compression and mixture interface? |
| R: sequence router | Sequence-derived weights over the same dictionary | Repaired slot encoder and new router | Can endpoints alone select useful structural prompts? |
| H: head adaptation | Final wave3 epoch-2 predicted graph, read by its original frozen interface | Only `base.output_head` | Can the task head make better use of the already repaired generator? |

Lane D is an oracle diagnostic, never a deployable arm. R and H use endpoints alone when scoring.
The first round holds the original prefix interface fixed in all lanes; it does **not** implement the
later dictionary-conditioned Stage I interface retraining proposed in the initial discussion. This
choice isolates the initial handoff. If D loses utility, compression and mixture-interface mismatch
remain competing explanations; the result does not close the dictionary route.

## 3. Shared dictionary and fixed target coordinates

### 3.1 Training boundary and reusable source models

Use the headline split, seed 42/root `node_007630`, and the legal graph returned by the current training
split builder. All training pairs touching V_val remain excluded. Compile targets with
`MotifTemplateTable`, which removes the queried partner and adjusts degrees. Retain the existing
26-slot, 96-edge compiler and degree-normalised weights unchanged.

The source reader/count head and prefix interface are from
`outputs/split_seed42/motif_prompt_stage1/best.pt`. The repaired slot encoder and lane-H source are
from `outputs/split_seed42/motif_prompt_stage2_v3_prefix/best.pt`, selected epoch 2, recorded checkpoint
`d2bb4fc7a55a10d3`. Record the actual loaded source identities and split in the experiment artifacts;
do not silently substitute the earlier warm8 checkpoint or the secondary epoch-4 checkpoint.

Build the seed-0, 15-epoch dynamic training corpus with the existing 1:5 sampler. Templates and routing
targets cover its deduplicated pair rows, keyed by canonical pair with orientation restored on lookup.
No validation or test row enters dictionary selection, target normalization, temperature estimation or
the training mean. Later seed replicas keep this dictionary fixed and compile targets for their own
legal training rows against it.

### 3.2 Representation and distance

For a true compiled template A, freeze the source reader and count head in evaluation mode and compute

\[
Z(A)=[t_u,t_v,t_{\mathrm{rel}},t_{\mathrm{cnt}}]\in\mathbb R^{4\times128}.
\]

The reader receives only the template's topology, role embeddings and existing structural features;
no sequence features or source node identities enter the dictionary token. Keep the four raw output
fields as the prompt values. Normalize **distances**, not individual prompt vectors.

Estimate each field's scalar variance on nonself epoch-1 training rows, including their swapped
orientations with equal weight. Use one pooled variance for the two endpoint fields. For field f,
let `s_f^2 = max(mean((Z_f - mean(Z_f))^2), 1e-12)`, averaging over rows and coordinates. Define

\[
d(Z,Z')=\frac14\sum_f\frac{\|Z_f-Z'_f\|_2^2}{128s_f^2}.
\]

This gives fields equal distance weight while preserving per-row magnitude differences. Store the
scales; never estimate them from a scored universe. Swapping endpoints leaves the metric unchanged.

### 3.3 Sixteen real representatives

The default is **16 template representatives before endpoint-swap expansion**, not 16 final routing
logits. Reserve one true empty template and split the other 15 representatives as follows:

| Nonempty template category | Representatives |
|---|---:|
| No closure and no complete bridge, but nonzero attachments | 3 |
| Closure present, no complete bridge | 4 |
| Complete bridge present, no closure | 4 |
| Both present | 4 |

Closure presence means positive wedge mass; bridge presence means positive complete three-hop path
mass. A template with neither can still carry endpoint attachments in the current compiler and must
not be labelled empty. The empty representative is an actual compiled training template, including a
legal training self-pair if needed; self rows otherwise remain outside structural target statistics.

For bounded offline construction, take up to 5,000 nonself candidate rows per nonempty category from
the corpus, uniformly without replacement with construction seed 42 and canonical ordering before
sampling. Candidate sampling does not change the training stream. Encode candidates, remove exact
duplicate template weight vectors, and choose real representatives within each category:

1. Start with the candidate closest to the category's mean in scaled token space, then initialize
   the remaining centers by maximum distance to the nearest chosen center.
2. Assign candidates to their nearest center. Replace each center with the real member nearest the
   assigned cluster's mean, which minimizes its within-cluster sum of squared distances.
3. Stop when representatives do not change or after 20 rounds. Keep an empty cluster's previous center.
   Break ties by canonical source pair and then candidate order.

If a category lacks enough distinct candidates, allocate its remaining places by farthest distance
over all other unused candidates. If fewer than 16 distinct templates exist overall, retain the
available representatives and record the actual size. Do not manufacture averaged adjacencies.

Append each representative's endpoint-swapped template using the compiler's `SWAP_PERM`; deduplicate
exact weight vectors and retain an involutive index map `swap_index`. The effective dictionary size
`K_eff` is at most 32. For each two-entry swap orbit, encode one orientation and generate its partner
token by exchanging `topo_u` and `topo_v`. Check this against the reader's direct swapped-template
output within FP32 tolerance. For a self-swapping template, symmetrize the endpoint tokens to remove
roundoff asymmetry. These operations preserve the source reader's equivariant representation.

### 3.4 Fixed target weights and constant control

Set temperature to the median of the strictly positive nearest-dictionary distances on nonempty,
nonself epoch-1 training rows and their swaps. Use `1e-6` if that set is empty and otherwise floor the
median at `1e-6`. This is a training-side default, not a validation temperature search.

\[
q_k^*(A)=\operatorname{softmax}_k[-d(Z(A),Z_k)/\tau],\qquad
T^*(A)=\sum_k q_k^*(A)Z_k.
\]

Use this rule for the empty template too; no oracle-only hard assignment is introduced. The constant
control uses the mean q* over the same nonself epoch-1 rows and their swaps under their natural
frequency, not the balanced candidate sample or closure-loss weighting. This mean is swap invariant.
It is a mean of routing weights and encoded prompts, **not** the old mean-adjacency intervention.

The dictionary, tokens, scales, temperature, mean and swap map are fixed across D and R. Routing
targets are compiled from training truth only during R training. V_val truth supplies separately
labelled oracle/target diagnostics; it never supplies optimization gradients.

## 4. Lane R: sequence-conditioned routing

Reuse the final wave3 residue projection, shared attention, query-residual bridge/witness reads,
endpoint projection and witness mix, including their learned epoch-2 parameters. Preserve
`slot_read: residual_block` and `slot_read_value_norm: false`. Their output is the ordered
`S=[u,v,C_1..C_8,L_1..L_8,R_1..R_8]` tensor of shape `B x 26 x 96`.

In the new branch, replace the two typed message-passing layers and three edge heads with
`flatten -> Linear(2496,256) -> GELU -> Linear(256,K_eff)`, using ordinary linear-layer initialization.
Let h be that shared MLP, P_S exchange the endpoint and left/right slots, and P_K apply `swap_index`.
Define the equivariant router by

\[
\ell(S)=\tfrac12[h(S)+P_Kh(P_SS)],\qquad
\pi(S)=\operatorname{softmax}(\ell(S)),\qquad
\widehat T(S)=\sum_k\pi_k(S)Z_k.
\]

P_S leaves the symmetric closure slots unchanged. The existing AB/BA prediction aggregation stays
unchanged. No predicted adjacency, RRWP or per-query GRIT pass exists on this deployment path.

Train only the reused slot encoder and router on `KL(q* || pi)` over the dynamic task stream's
nonself rows. Use the existing globally counted closure-balanced row weights, nonempty share 0.5,
including its behavior for batches missing one category. Use FP32 `log_softmax` and KL arithmetic.
No task BCE, subgraph objective or token KD gradient reaches this path in the first round. Those
objectives are not silently retained as extra training signals. Self-pairs still receive sequence
predictions and remain in official evaluation; non-self metrics expose that unsupervised category.

Budget: 15 epochs, no early stopping, seed 0, AdamW peak LR `1e-4`, weight decay `1e-2`, gradient clip
1.0, one-cycle `pct_start=0.1`, `div_factor=25`, `final_div_factor=10000`, cosine annealing. Keep the
current motif family's data and runtime batching defaults. Evaluate every epoch through the frozen
Stage I prefix/trunk/head and select with the existing five-metric V_val rule, not minimum route KL.

## 5. Lanes D and H

### 5.1 Dictionary oracle

Score the same V_val classification and topology universes with the frozen Stage I interface under:
original true-template tokens; dictionary q*-mixture tokens; constant training-mean routing tokens;
and `gates_off`. Run the dictionary mixture through the actual prefix LayerNorm, projections,
attention and PPI head. Token reconstruction alone does not measure retained oracle utility.

There is no optimizer or artificial training schedule in this lane. Report the retained AUPRC gap
relative to gates off as a descriptive ratio when the original oracle gap is positive, alongside
absolute edge and topology metrics. No numerical retention cutoff becomes a runtime gate.

A convex combination after the reader avoids interpreting an averaged graph, but is not guaranteed
to be on the reader's output distribution. Failure here leaves open whether a larger dictionary or
an interface trained on these combinations would work. Those are subsequent isolated experiments.

### 5.2 Head-only adaptation and its paired control

Initialize both H runs from the final wave3 epoch-2 model. In the prompted run, freeze the generator,
reader, count head, prefix adapter, encoder and cross-attention in evaluation mode. Train only
`base.output_head`, in training mode. In the content-only control, remove the prompt contribution
throughout training and scoring while training the identically initialized output head with the same
batches and budget.
This is a training condition, not misuse of a scoring-only intervention inside a training forward.

Use 15 epochs, no early stopping, seed 0, and the lane-R optimizer settings with peak LR `1e-5`.
Both H runs use the existing task BCE and subgraph objective: subgraph BCE 1.0, rank 1.0, degree 0.1,
motif 0.1; GS, RD and degree-MMD training weights remain zero. Preserve the current structural sampler
and stream reductions. Omit constant graph-fit and teacher losses from the optimization objective.

Evaluate and select both runs independently with the current five-metric rule. Compare the prompted
run with both its unadapted source and the newly trained content-only head. Its scoring-time
`gates_off` result uses its adapted head and is **not** the historical `prefix_base` result. Do not
open prefix parameters in this lane; that would add a second changed training component.

## 6. Integration boundaries and experiment artifacts

Introduce a separate dictionary model family, `v3_1_motif_dictionary`, with explicit `oracle` and
`sequence` modes. Oracle mode must use the existing diagnostic run-kind/scoring opt-in boundary.
Sequence mode's forward takes only endpoint features and lengths; any training target is consumed by
the loss, not by the scoring forward. Do not encode a dictionary prompt as a fake 96-edge graph.

Factor the existing four-token-to-logit path for reuse: the token fields remain `topo_u`, `topo_v`,
`topo_rel`, `topo_cnt`, width 128, two prefix rows per field, at all nine existing cross-attention sites.
Keep the source graph model's normal behavior and checkpoint format intact. Extract the slot-read
component only as needed for reuse; no unrelated refactoring or legacy migration is part of this work.

Keep head-only adaptation in the current motif family with an explicit training policy that chooses
exactly the output head and, for its control, disables the prompt. Freezing must include eval mode as
well as optimizer membership; a frozen encoder or dropout-bearing interface must not become
stochastic because its parent enters training mode. In head-only mode gradients need not traverse the
frozen feature producer. Lane R can encode residues without gradients, but must retain the backward
path through the slot encoder, router and route loss. Its downstream prefix/trunk pass is evaluation
only in this round and needs no backward graph.

Save one shared dictionary artifact with representative weights, source pairs for provenance only,
source checkpoints, raw tokens, scales, temperature, swap map, training mean and construction settings.
Embed the inference constants in each dictionary checkpoint. Scoring must work with the construction
artifact and all target caches unavailable. Source node identifiers never become router inputs.

Cache training q* by canonical pair and recover its orientation explicitly, covering each run's
corpus. Validation target caches are separate. Cache mismatches, illegal data access, non-finite
values and inconsistent distributed reductions fail closed; entropy, collapse or poor quality are
reported telemetry and do not abort a run.

For sequence routing, support `gates_off`, `mean` (fixed training-mean routing weights), and
`shuffle_route` (one seeded permutation over the whole scored universe). Shards and batches must not
define the permutation. Transplant routing weights/tokens, keeping this row's endpoints fixed.
Closure/bridge rewiring interventions have no meaning on cached mixture tokens and must be rejected.

## 7. Evaluation, provisional adoption and later controls

The [current evaluation protocol](../../03-experiments.md#32-evaluation-protocol) remains authoritative:
select each checkpoint's topology threshold by closest geometric RD on the V_val sampled pair union;
rank checkpoints by AUPRC, GS and the three MMD ratios; keep RD reporting-only. Classification uses
its separate frozen max-F1 threshold. Test never selects a checkpoint, dictionary, temperature or repair.

Report AUROC, AUPRC, non-self AUPRC, classification/calibration metrics and all five topology metrics.
Alongside the protocol threshold, compare GS/MMD at matched output density on the same pair union,
retaining atomic ties and reporting realized counts and residual per-subgraph RD differences. AUPRC
does not change when thresholds change. Do not interpret lower MMD alone as improved topology.

For R, add route KL versus the training-mean predictor, raw four-field token error, token norms,
mean routing weights, per-row entropy, between-row dispersion and transplantation effects. Apply
ten V_val shuffle seeds 0-9 to the selected checkpoint and report mean and range. KL and norm
diagnostics complement, rather than replace, downstream metrics. Fixed tokens give

\[
\|\widehat T-T^*\|\leq\max_k\|Z_k\|\,\|\pi-q^*\|_1,
\]

but this does not guarantee a bound on useful PPI decisions after nonlinear prefix processing.

The owner chose **provisional adoption, then replication**. Read each lane against its matched
comparator on V_val and its mechanism checks; record the full metric tradeoff and an explicit
adopt/retain/revise decision. Do not invent a new numerical eligibility gate. In particular:

- D can support retaining the dictionary for investigation, not adopting an endpoint-only method.
- R needs downstream evidence beyond lower KL or a better training fit; mean and shuffle controls
  establish whether useful pair-specific selection is present.
- H improving only as much as its content-only control supports generic head adaptation, not a
  graph-prior claim. Keep that distinction in the main-architecture update.
- If R and H separately help, their combination is a new comparison against both components;
  separate improvements do not establish additivity.

After a useful seed-0 candidate, replicate seeds 1-2 and run the appropriate next-round controls:
same-shape free learned prompt vectors; mean routing; count-only and degree-only representations;
and a random frozen reader. These controls are not launched in the first round. Give each its own
matched interface training rather than feeding incompatible random tokens into a pretrained adapter.
A count-only claim requires targets/routing to use only that allowed information: retaining q* from
full GRIT tokens would leak richer structure into that control. Report parameter and training-budget
differences for the free-vector control rather than calling unequal trainable capacity identical.

Only after fixing the candidate and comparisons, run held-out evaluation from the V_val-selected
checkpoints. A seed-0 improvement is a working-architecture choice, not a validated generalization
claim. This round does not redefine the previous task contract or establish that GRIT is necessary.

## 8. Parallel execution, verification and deliverables

After implementation and local checks, build the shared dictionary once, then allocate:

| Container | Work |
|---|---|
| `30838`, 4 H20 | R training and its selected-checkpoint route interventions |
| `30846`, 4 H20 | H prompted training, then its matched content-only training |
| `30030`, 4 H20 | D oracle scoring and cached-score comparisons |

The execution-time inspection on 2026-09-19 found four H20s on each container; the earlier
two-GPU description of `30030` was stale. Confirm actual occupancy again before launch; these are allocations, not observations of current idle
resources. All containers share the checkout/data/output filesystem. Use Git to synchronize working
code, cap concurrent Torch CPU threads per the HPC runbook, and use the production `hpc/run.sh`
training/scoring paths with automatic visible-GPU sizing. Preserve batching across the paired H
experiments. Do not keep a completed diagnostic GPU job alive merely to fill a lane.

Use separate output directories and attempts for `motif_dict_oracle`, `motif_dict_router`,
`motif_wave3_head`, and `motif_wave3_head_content`. Training runs initially use `--skip-test`; oracle
artifacts are labelled diagnostic. Record actual source identity, loaded trainable modules, losses,
learning-rate schedule, per-epoch validation and selections. Completion follows the runbook's
run-kind-specific artifacts and terminal logs; direct testing does not create a pipeline sentinel.

Required implementation verification:

- Training templates/targets exclude V_val endpoints and remove the queried edge correctly;
  attachment-only templates remain a distinct dictionary category.
- Swap is an involution for templates, tokens, q*, router outputs and final symmetric logits;
  role relabellings of a compiled template leave reader tokens unchanged within numerical tolerance.
- Lane R changes only the slot encoder/router; lane H changes only the output head. Confirm gradient
  reachability and unchanged frozen tensors across a real optimizer step, including AdamW decay.
- D/R use the same constants; zero prompt contribution reproduces the appropriate source head
  path. Test the dictionary-weighted token sum directly, including a single active dictionary entry.
- Endpoint-only scoring succeeds without truth graphs or target caches. Oracle scoring requires
  the diagnostic opt-in. Invalid graph-only interventions on the dictionary family are refused.
- Mean routing is fixed to training; whole-universe shuffle agrees across batch sizes and fan-out.
  Self-only batches and batches lacking one closure category have defined finite loss behavior.
- Checkpoint round trips retain constants and training policy. FP32 mixtures, losses and score
  artifacts satisfy the existing precision contract; run relevant local tests and changed-path lint.

Deliver a compact result note with the three hypotheses, exact comparisons, epoch curves, all metric
groups, mechanism diagnostics, and provisional adoption decisions. Update architecture/config
documentation only when an implemented behavior or accepted working architecture actually changes.

## 9. Review boundary and change log

This spec is the approved architectural brainstorming deliverable. The owner explicitly requested
GPT-5.6-sol medium subagents for implementation, primary-agent review, and parallel launch after
integration. Authorization is not evidence that implementation, remote launch or the proposed
architecture has succeeded. The implementation plan preserves the three-lane comparison and the
adopted-versus-confirmed distinction.

| Version | Date | Change |
|---|---|---|
| v1 | 2026-09-19 | Initial three-lane design following owner decisions. Source inspection corrected the conversational 1+5+5+5 dictionary allocation to 1 empty + 3 attachment-only + 4 closure-only + 4 bridge-only + 4 combined, preserving the 16-representative budget. |
