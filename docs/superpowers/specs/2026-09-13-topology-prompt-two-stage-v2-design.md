# Topology prompt v2: a two-stage method with KD and structural supervision inside the stages

Date: 2026-09-13, revised the same day after the owner's review. Status: approved for implementation on 2026-09-13; no v2 study has been launched. Follows the curve reading of `docs/results/topo_prompt_stage1.md`,
`docs/results/topo_prompt_stage2.md` and `docs/results/topo_prompt_stage2_curves/README.md`.
Supersedes the four-stage route table in
`docs/superpowers/specs/2026-09-12-topology-prompt-stage1-design.md` §1: Stages III and IV are
retired as concepts; what they were meant to do is placed inside Stage I or Stage II. The
hyperparameter space is declared in `docs/03-experiments.md` §6.

Revision record (review of 2026-09-13): the logit-shift explanation of the V_val MMD swings is
withdrawn (§0.1); the internal held-out-node fold is removed from training and stopping and
replaced by the standalone probe (§3.1); the Stage II objective is reduced to task, coordinate,
pointwise KD, anchor KD and the hard structural terms, with representation KD and teacher-soft
degrees deferred to ablations (§3.3); anchor KD gets an explicit masked-softmax definition
(§3.3); the schedule runs to completion with no early stopping (§3.5); the Stage II wave becomes
a 2×2 factorial (§5); test rescoring is descriptive only (§5).

## 0. What the curves establish, and what they do not

### 0.1 Established

1. **Coordinate fit, classification loss and assembled topology favour different checkpoints.**
   Logged values of `coord_gen_full` (not the later rescores):

   | Epoch | val BCE | val coord loss | AUPRC | degree MMD ratio | topology threshold |
   |---:|---:|---:|---:|---:|---:|
   | 3 | 0.730 | **1.328** | 0.793 | 13.03 | 2.875 |
   | 6 (selected) | 0.867 | 1.632 | 0.806 | **5.83** | 1.359 |
   | 10 | 0.678 | 1.521 | 0.808 | 7.34 | 1.977 |

   The selector chose the favourable end of an unstable topology curve. Ranking (AUPRC ±0.01)
   is stable across epochs; coordinate predictions and graph reconstruction are not.
2. **The MMD swings are changes in the admitted edge set, not a scale effect.** The threshold
   is re-selected per checkpoint by searching admitted-edge sets, so any strictly increasing
   transformation of the logits (up to ties) leaves the admitted set and every graph metric
   unchanged. A scale or offset change can move val BCE and the numerical threshold; it cannot
   move GS or the MMD ratios. The earlier reading that "all logits shifted down by one nat"
   may describe part of the BCE swing, but it is not the mechanism of the topology swing and
   has not been separated from admitted-set changes. The diagnostics that would separate them
   (§0.3) have not been run.
3. **Val coordinate loss bottoms at epoch 3 in both lanes and rises afterwards** (+17 % full,
   +61 % frozen) while the composite training loss falls. Only the composite training loss was
   logged, so the training coordinate term is not known to fall monotonically.
4. **The predicted coordinates affect the assembled test graph.** Mean coordinates move test RD
   from 0.91 to 0.52 and the relation field alone to 0.71 (`docs/results/topo_prompt_stage2.md` §1).
5. **Stage II ran at a declining, never-annealed learning rate**: 1e-3 near epoch 3 down to
   6.2e-4 at epoch 12, where patience 10 from the epoch-2 minimum stopped it. The low-LR tail
   of the 25-epoch cycle was never reached (`_build_scheduler`, `train_b0.py:517-586`).

### 0.2 Hypotheses (to be tested, not assumed)

- **Memorisation of node-specific structure.** The endpoint head is pair-conditioned
  (`[self | partner]`) and its targets are recomputed with the queried edge removed, so endpoint
  coordinates are not per-node constants; they do carry a large repeated node-specific
  component (about 60 rows per node per epoch). Training-row R² of 0.4–0.6 against V_val R² ≤ 0
  is consistent with poor generalisation of the head; it does not identify the mechanism or
  separate it from the train→V_val universe shift (mean log1p degree 1.83 vs 2.15).
- **Region-level density transfer.** The interventions show the coordinates matter on test;
  they do not show that the endpoint field learned transferable regional density. Training-row
  degree R² says nothing about unseen test nodes.

### 0.3 Diagnostics to run on V_val before the redesign is read as confirmed

On the existing `coord_gen_full` per-epoch checkpoints (`checkpoints/epoch-XXXX.pt` on H20):
for consecutive epochs, the least-squares affine fit between their logits on the V_val topology
universe and its residual; the Jaccard overlap of the admitted edge sets at each epoch's own
threshold; and the per-node change in predicted degree. These say how much of the swing is
scale and how much is admitted-set change. They are V_val diagnostics and select nothing.

## 1. Design principle: two stages, mirrored objectives

| Stage | Structure in the forward | Trains | Loss | Role |
|---|---|---|---|---|
| I, **reader** | true `s*` (query edge removed), clean or corrupted per row | trunk + prompt interface | task BCE + hard structural terms (optional arm) | ceiling diagnostic; the immutable teacher for II |
| II, **student** | predicted `ŝ = g(x_u, x_v)` | generator + prompt interface + output head; encoder and cross-attention frozen | task BCE + coordinate + pointwise KD + anchor KD + hard structural terms | the deployable arm |

Both stages use the two data streams the repo already has: the 1:5 task stream and the
sampled-subgraph structural stream (`StructStream`, `train_b0.py:3708+`). KD in Stage II is
online: the teacher is an immutable, eval-mode deep copy of the Stage I checkpoint fed the true
coordinates in the same forward; no bank is dumped. The structural subgraph is the KD context.
Inference stays `(x_u, x_v)` → one symmetric logit; coordinates keep their fixed semantics and
every intervention still applies.

## 2. Stage I: a reader that tolerates imperfect coordinates

### 2.1 Corruption training (stationary; not a curriculum)

The reader has only seen exact coordinates plus a 0.1 field mask. Stage I `gates_off` and
`shuffle` show it relies on the coordinate path almost exclusively. Train it on imperfect
coordinates, with a fixed corruption distribution for the whole run (no severity schedule; a
schedule is a second knob with no evidence behind it).

Per training row, with probability `prob` (default 0.5; the other rows see exact coordinates so
the ceiling stays a ceiling), after standardisation and before the existing field mask:

- **continuous fields** (30 coordinates): shrink toward the training mean, `z ← λ z`,
  `λ ~ U(shrink_min, 1)`, then add `σ ε`, `ε ~ N(0, I)`, `σ ~ U(0, sigma_max)`;
- **distance class** (4 one-hots, 5-way with the self-pair convention): build a valid softened
  distribution *before* standardisation, `p̃ = λ e_{d*} + (1 − λ) π_train` with the training class
  prior `π_train` and the same `λ`, then standardise it as the generator's soft one-hots are
  standardised (`coord_gen.py:317-336`). Gaussian noise on standardised indicator components is
  not applied: it produces inputs that are not probabilities.

Defaults `prob 0.5, shrink_min 0.3, sigma_max 0.5` are starting points, not validated values.
The generator is not an exact conditional-expectation estimator (Huber plus cross-entropy plus
task and KD gradients), and even for one the reader's nonlinearity means
`f(X, E[s*|X]) ≠ E[f(X, s*)|X]`; corruption buys tolerance, and Stage II's interface
adaptation handles the particular errors the learned generator makes. Once the standalone probe
(§3.1) exists, its residuals say whether the corruption magnitude and correlation resemble real
generator error; adjust then, not before.

**Sensitivity diagnostics** logged per epoch under fixed perturbations (all rows at `λ = 0.5`;
all rows at `σ = 0.5`; both): AUPRC, BCE and Brier on val_cls; mean and 95th-percentile
absolute logit change on the V_val topology universe; and the five topology numbers both at
the clean checkpoint's threshold (fixed) and at the perturbation's own re-selected threshold.
AUPRC alone can stay flat while logits move a great deal; the logit-change and fixed-threshold
rows measure what actually concerns Stage II.

### 2.2 Structural stream on the reader (second teacher arm)

The full lane's V_val shape ratios (11.1 / 4.9 / 13.8) are worse than the frozen lane's
(4.2 / 2.0 / 6.0) and rise while GS is flat. `topo_prompt_full_struct` adds the structural
stream: `struct.weights = {bce 1.0, rank 1.0, degree 0.1, motif 0.1}`, the `struct_new` seed-42
winner, **subgraph BCE included**. Subgraph pairs need coordinates for this family
(`topo_prompt.py:609` refuses a batch without them): add `StructCoordinateTable.coords_for_pairs`
on the *full training table* (dense products for non-edges, `_exact_pair` for edges), so the
coordinates keep their training-graph meaning. They are never recomputed inside the sampled
40-node graph. Lift the family gate at `train_b0.py:5786-5790`.

### 2.3 Teacher choice and what stays fixed

lr 1e-4, weight decay 0.05, one-cycle over 25 epochs, patience 10 on task BCE, five-rank
selection: unchanged. Its rising val BCE with flat Brier/ECE is benign. Between the two teacher
runs, the V_val five-rank on clean coordinates is the protocol's selector, but the §2.1
sensitivity rows are evidence too: the better clean-coordinate score does not by itself identify
the better teacher for a generator. The chosen checkpoint is copied once and never trained again.

## 3. Stage II: generator and interface adaptation

### 3.1 The generalisation question: a standalone probe, not a training fold

The earlier proposal to withhold coordinate supervision from 10 % of training nodes inside the
Stage II run is withdrawn. Those nodes would still train the generator through task BCE, through
KD (whose targets are computed from their true coordinates and whose gradient reaches `g_ψ`
through the student logit) and through the structural stream, so their coordinate R² would not
measure generalisation to unseen nodes. Stage II keeps all training supervision and selects
checkpoints on V_val; coordinate fit is a diagnostic.

**The probe** (separate script, CPU or one GPU, hours):

- Hold out 10 % of training nodes (seeded). Fit a *separate copy* of the two coordinate heads
  on all training rows that touch no held-out node, with the coordinate loss only.
- Report per-field R² and distance accuracy on: in-fold pairs; pairs with exactly one held-out
  endpoint; pairs with two held-out endpoints; V_val. Endpoint metrics are reported on the
  held-out endpoints themselves, never pooled with familiar endpoints.
- Two input feature sets: the Stage I frozen encoder's pooled states (the deployed input; the
  encoder has already seen the held-out nodes during Stage I training, so this measures
  generalisation of new heads inside an already trained representation, and the note says so),
  and the raw intrinsic F0 features as a complementary check that carries no such exposure.
- The probe is also where endpoint-head regularisation (dropout, weight decay, width) is
  compared, on held-out-node R², before any Stage II run uses it (`docs/03-experiments.md` §6.3, P).

Readings, deliberately soft: in-fold high and held-out low supports poor generalisation of the
head (mechanism unidentified); in-fold ≈ held-out ≫ V_val points at the universe shift, in
which case V_val R² is reported but not optimised and a context-relative coordinate spec
(`docs/tmp/2026-09-12-structure-semantic-prompt-design.md` §3) becomes the follow-up; all low
could be limited features, head capacity, optimisation or target scale, and needs its own
follow-up before anything is built on it.

### 3.2 Trainable scope

Generator, prompt interface (tokenizer, per-layer projections, offsets, 72 gates) and
`base.output_head` train; encoder and the three cross-attention layers stay frozen. Freezing the
encoder is a compute choice: the frozen-trunk reader's 0.940 on *true* coordinates shows it can
use supplied structure, not that its pooled states contain everything a generator needs.

Implementation consequences: the teacher is a deep copy made at construction, eval mode,
`requires_grad=False`, never updated; the student's `train()` override (`coord_gen.py:300-311`)
must put the trainable modules in train mode while frozen modules stay in eval; frozen
parameters keep `requires_grad=False` but remain on the autograd path from the prompt input to
the generator (they do today). Two parameter groups (generator; interface + head) with their own
peak learning rates, which requires `_build_scheduler` to pass a per-group `max_lr` list instead
of replicating one scalar (`train_b0.py:517+`, the `[max_lr] * len(param_groups)` line).

### 3.3 The objective

$$
\mathcal L_{II} = \mathcal L_{\mathrm{task}} + \lambda_c \mathcal L_{\mathrm{coord}}
 + \lambda_a \mathcal L_{\mathrm{anchor}} + \lambda_H \mathcal L_{\mathrm{struct}}
$$

- **Task with pointwise KD as a normalised mixture.**
  `L_task = (1 − α) BCE(ℓ_S, y) + α BCE(ℓ_S, q_T)`, `q_T = σ(ℓ_T)` from the teacher on true
  coordinates, positive row weights as today. The classification weight stays 1 whatever `α`
  is, so raising KD cannot silently raise the classification weight against the coordinate
  term; `α = 0` is the no-pointwise-KD control at matched weight. Default `α = 0.5`. Logged on
  the same rows and weighting: teacher entropy `H(q_T)` (the KD floor; it is not the teacher's
  BCE against the label) and the entropy-subtracted KD, i.e. the Bernoulli KL.
- **Coordinate term** unchanged in wave 1: Huber averaged over the 30 continuous standardised
  coordinates plus distance cross-entropy. This is equal weighting of coordinates, not of fields
  (endpoints hold 18 of 30). Per-field reweighting is an ablation, not a default.
- **Anchor KD on the structural subgraph**, new function `struct_anchor_kl`: for anchor `i` with
  legal candidate set `C_i`, `q^T_i = softmax_{j∈C_i}(z^T_{ij} / T_a)`, `q^S_i` likewise from
  the student, loss = mean over anchors of `KL(q^T_i ‖ q^S_i)`. Invariant to a per-anchor
  additive shift; `T_a` controls concentration (default 1). This is *not* `kd_dist_loss`
  (`losses.py:88`), which applies sigmoid before the softmax and therefore bounds any two
  candidates' probability ratio by `e`; that function stays as it is for the `kd_rank` baseline.
  Before its weight is chosen, log teacher anchor entropy and the term's gradient norm.
- **Hard structural terms** on the same subgraph: `{bce 1.0, rank 1.0, degree 0.1, motif 0.1}`
  against the induced adjacency. `λ_H` scales rank, degree and motif only; subgraph BCE stays at weight 1.0 in every factorial arm. Hard degree
  says how many neighbours; anchor KD says which ones; the roles do not overlap.
- **Deferred to ablations (weight 0 in wave 1):** representation KD (`kd_rep` is a cosine
  distance before a trainable head; it constrains neither norm nor the head and can compete with
  adaptation to predicted coordinates) and teacher-soft degree matching (the teacher's sigmoids
  are not shown to be calibrated adjacency probabilities on sampled subgraphs, so its soft
  degrees can conflict with the hard degree target). Each earns a weight only through a matched
  run. KD supplies a constraint on the unfrozen interface; it does not make unfreezing "safe".

### 3.4 Cost

Per step: all legal pairs of one 40-node subgraph through the student with gradients into
generator, interface and head (checkpointed as `StructStream._score` does) plus one no-grad
teacher pass on the same pairs with true coordinates from `coords_for_pairs`. The stream assigns
the whole subgraph to one rank per global step while the others only synchronise, so the wall
cost depends on the active rank's workload, not on 780 / global pairs. **Profile one epoch
before allocating the wave**; the earlier "2.2×, 7–8 h" figure is unverified.
`struct.subgraphs_per_epoch` is the compute lever if needed.

### 3.5 Schedule and selection

- Full 15-epoch one-cycle (`pct_start 0.1`, `final_div_factor 100`), **no early stopping**
  except non-finite state. Patience 5 on any monitor could still stop a run at epoch 7 with the
  schedule mid-anneal; a fixed budget gives every arm the same optimisation and removes the
  composite monitor and its mixed populations.
- Checkpoint selection unchanged (`geometric_rd_five_rank_v1`) on V_val after the run.
- The result note shows the per-epoch spread of the three MMD ratios next to the selected value
  and the §0.3 admitted-set diagnostics for the selected epoch's neighbours.
- No field masking on predicted coordinates (unchanged).

## 4. Where Stages III/IV went

| Planned in III/IV | Now |
|---|---|
| unfreeze interface, readout, upper encoder under task BCE | interface + head trainable in Stage II under KD and structural terms (§3.2, §3.3); encoder frozen as a compute choice |
| topology loss on the student | hard structural terms in Stage II (§3.3) |
| weak coordinate anchor | full coordinate supervision kept (§3.1) |
| adapt the reader to predicted structure | corruption training in Stage I (§2.1) plus interface adaptation in Stage II |

## 5. Runs

**Before training (no test involvement):** the §0.3 admitted-set diagnostics on V_val; the
§3.1 probe; seeds 1–2 of `coord_gen_full` as already planned. Test rescoring of old epochs does
*not* decide the redesign's focus. After the method and selection rule are locked, a
predeclared sensitivity analysis on the selected epoch's neighbouring checkpoints is reported
descriptively; threshold–RD correlation is not read as causal, since each checkpoint changes
both its scoring function and its threshold.

**Stage I (two runs, diagnostic):** `topo_prompt_full_v2` (corruption) and
`topo_prompt_full_struct` (corruption + structural stream). One teacher is chosen (§2.3) and
frozen as the reference; the other is the teacher ablation.

**Stage II wave: a 2×2 factorial on the KD and structural contribution.** All four share the
teacher, the trainable scope, the sampled subgraph stream and pairs, the coordinate term, the
schedule and the selector; the structural stream runs in every row so the pairs are matched.

| Run | task + coordinate + subgraph BCE | pointwise + anchor KD | rank + degree + motif |
|---|---|---|---|
| A | yes | off (`α = 0`, `λ_a = 0`) | off |
| B | yes | on | off |
| C | yes | off | on |
| D | yes | on | on |

A is the sampler-matched control the earlier wave lacked. B vs A and D vs C read KD with and
without structural terms; C vs A and D vs B read the structural terms with and without KD. The
factorial does not isolate unfreezing; a fifth run (D with generator-only training) does, if
the budget allows, and is the lower priority. The optimisation changes (§3.5) are the common
recipe, not a contribution under test. Read on V_val first, edge and topology families
together, against `prefix_base` and each row's teacher ceiling; single-seed differences inside
±0.01 GS / ±0.5 MMD ratio are not read; the winner gets seeds 1–2, then one test.

## 6. Implementation contract

- `topo_prompt.py`: corruption in the training path of `logits_from_encoded` (continuous:
  shrink + noise; distance: prior mixture before standardisation; seeded per step); the §2.1
  sensitivity diagnostics under `no_grad`; `trainable: interface_head`; `pair_repr` exposed for
  the later `kd_rep` ablation only.
- `coord_gen.py`: immutable teacher deep copy; trainable scope and `train()` override; two
  parameter groups; `α` mixture task term; teacher entropy and KL logging; a pair-batch entry
  returning student and teacher logits for the structural stream.
- `struct_coords.py`: `coords_for_pairs(u_idx, v_idx)` on the training table.
- `train_b0.py`: lift the family gates (:5786-5790, :5816-5820); `StructStream._score` passes
  true coordinates to the teacher pass and nothing to the student; `struct_anchor_kl` assembled
  next to `struct_total` with its own weight and temperature; per-group `max_lr` list in
  `_build_scheduler`; `eval.patience: null` semantics for a fixed budget; the §0.3 diagnostics
  as a scoring utility on saved checkpoints.
- `struct_losses.py`: `struct_anchor_kl(z_student, z_teacher, mask, temperature)`.
- `src/experiments/coord_probe.py`: the §3.1 probe (two feature sets, four surfaces,
  regularisation grid).
- Configs: `topo_prompt_full_v2.yaml`, `topo_prompt_full_struct.yaml`,
  `coord_gen_v2_{a,b,c,d}.yaml`.
- Tests: corruption is training-only, swap-symmetric, and yields valid distance distributions;
  teacher copy bit-identical after a training step; frozen modules stay in eval while trainable
  ones train; anchor KL is shift-invariant per anchor and zero at equality; per-group peak LRs
  reach the scheduler; 2-rank DDP agreement for the new stream terms (follow
  `tests/test_struct_stream.py`).
- Docs in the same change: this specification is canonical; `AGENTS.md` active set; `docs/03-experiments.md` §5.4 "Next" and §6.

Implemented entry points (no study has been launched):

- `src.experiments.coord_probe --reader-checkpoint <reader.pt> --feature-root <features>
  --data-root data --output <probe.json>` fits the two feature sets and six regularisation
  settings; only held-out endpoint R² selects the pooled-feature winner.
- `src.experiments.topo_prompt_diagnostics <previous.npz> <current.npz>
  --previous-threshold <value> --threshold <value> --output <stability.json>` joins canonical
  V_val pairs before comparing saved scores. New training runs log these rows every epoch
  and preserve the previous scores in their resume state.
- `src.experiments.score_coord_gen_v1_diagnostic score --checkpoint <old-epoch.pt>
  --pairs val_topology <normal score options>` explicitly reconstructs the original Stage II
  forward for the historical analysis in §0.3. It accepts only V_val sources, keeps the original
  checkpoint identity, and never rewrites or publishes a converted checkpoint. Current training
  and production scoring use the v2 contract; old loss keys are not silently reinterpreted.
- `src.experiments.topology_prompt_hpo --study <T1|T2|S1|S2|S3_size|S3_frequency>
  --base-config <inherited-winner.yaml> --sweep-dir <study-dir>` runs the §6 budget through
  `hpc/run.sh` with `--skip-test`; S2 additionally needs `--prefix-base-auprc <V_val value>`.
  T2 adds the fixed structural-stream defaults when inheriting the stream-free T1 config.
  The result includes the winner's epoch-wise MMD ranges. Replicas, explicit immutable teacher
  choice, profiling, factorial runs and the final test remain the declared operator sequence.

Run each module with `.venv/bin/python -m`. These commands are interfaces, not launch records.

## 7. What each outcome would mean

- A alone (new recipe, no KD, no structural terms) already removes the epoch-to-epoch
  admitted-set instability (§0.3 overlap high, MMD spread inside ±0.5): the instability was the
  reader's brittleness plus the schedule; B–D are then about the edge cost.
- B or D recovers test AUROC to ≥ `prefix_base` while holding RD ≈ 0.9 and the MMD ratios:
  the edge cost was the frozen reader's misreading of predicted coordinates, and KD-anchored
  interface adaptation is the method.
- C or D lifts V_val GS above the base (today 0.387 vs 0.401) with the anchor/rank terms and
  the probe's held-out relation R² rises: the structural stream supplies the pair-level signal
  the pointwise generator could not learn.
- None moves V_val while test topology holds: the gain is region-level density transfer only;
  write it as such and move to a context-relative coordinate spec.
