# Motif-graph GRIT prompt: verdict on the first wave (2026-09-18)

Headline split (seed 42, root `node_007630`), training seed 0, code `686627d`
(`codex/motif-graph-grit-prompt`). Spec: `docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md`.
Runs read here: `motif_prompt_stage1{,_bridge_only,_closure_only}` (ceiling diagnostics, true templates),
`motif_prompt_stage2` and `motif_prompt_bridge_only` (deployable, `(x_u,x_v)` only). `motif_prompt_closure_only`
was still training when this was written; the other section 8 controls and the seed-1/2 replicas were not launched.
The spec's pilots A and B (section 0.2) were never implemented or run; the Stage II bundle is the five-metric
Stage I winner, "not selected for teachability" (section 7.4). Evidence files beside this README:
`interventions_and_counterfactuals.md` (scoring-time interventions, reader x graph counterfactual, dose-response,
row-level mediation; artifacts under `outputs/analysis/motif_verdict_20260918/interventions/` on the H20) and
`gradients_and_generator.md` (generator output statistics, gate saturation, gradient flow, reader Jacobians,
prefix gates; `outputs/analysis/motif_verdict_20260918/gradients/`). No linear probe is used anywhere as evidence.

## 1. Verdict

**The architecture works; the generator failed to learn any effective structure.** Stage II is `prefix_base`
with a harmful constant offset, and every mechanism-level measurement locates the failure in the
residue-conditioned generator G, not in the reader, the interface, the trunk or the losses' ability to carry
signal. G's parameters did move (`L_topo` fell 0.70 -> 0.34), but never towards the graph target.

Held-out test, `test_protocol_v8`, one V_val-selected threshold:

| Row | sel. ep | AUROC | AUPRC | non-self AUPRC | GS | RD | deg / clus / spec MMD |
|---|---|---|---|---|---|---|---|
| `prefix_base` (frozen trunk; `gates_off` reproduces it) | 10 | 0.720 | 0.744 | 0.704 | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 |
| B0 | 8 | 0.701 | 0.736 | 0.697 | 0.429 | 0.550 | 9.9 / 8.4 / 15.0 |
| Stage I, true templates (ceiling) | 11 | 0.785 | 0.810 | 0.783 | 0.492 | 0.483 | 10.8 / 9.7 / 15.2 |
| Stage I bridge-only | 12 | 0.781 | 0.811 | 0.784 | 0.502 | 0.621 | 7.4 / 6.4 / 11.2 |
| Stage I closure-only | 10 | 0.722 | 0.766 | 0.735 | 0.451 | 1.222 | 3.7 / 4.5 / 5.0 |
| **Stage II (deployable)** | 10 | 0.722 | 0.745 | 0.707 | 0.416 | 0.491 | 10.6 / 9.0 / 15.4 |
| Stage II bridge-only control | 1 | 0.717 | 0.741 | 0.702 | 0.409 | 0.467 | 11.2 / 9.5 / 16.3 |

At matched output density (RD ~ 1.05 on test, `outputs/split_seed42/motif_rd1_diagnostic.json`) Stage II,
`prefix_base` and B0 sit at GS 0.431 / 0.427 / 0.431 with MMD 3.0-3.5-5.0 / 3.1-3.9-5.2 / 3.2-3.4-5.7: no
shape gain is hiding behind threshold placement. Single-seed margins (+-0.01 GS, +-0.5 MMD) are not read.

### What the ablative arms say

- **Stage I proves the interface and the reader.** True compiled templates through the frozen trunk give
  +0.066 test AUPRC and +0.08 GS over the same trunk. The bridge family alone carries the whole edge gain and
  the better topology; closure-only matches bridge-only on V_val (0.874 vs 0.876 AUPRC) but loses 0.045 more
  on test, so closure evidence transfers less well across the train/test density gap.
- **Stage II showed no sustained improvement.** V_val AUPRC 0.8055 / 0.8013 / 0.8039 / 0.8027 at epochs
  1 / 5 / 10 / 15 (prefix_base 0.813 at its selected epoch), train slot loss flat at 0.070, and the bridge-only
  control selected epoch 1 because all epochs are within noise of each other. The control is uninformative as
  an ablation: both arms are the base.

### Where the signal dies (mechanism)

1. **The predicted graph carries no row information.** `shuffle_graph` (every row reads another row's predicted
   graph) leaves V_val AUPRC at 0.8039 to four decimals and GS at 0.396 vs 0.394; `permute_closure` and
   `rewire_bridge` change no logit by more than 0.001 because the generator's output is slot-symmetric. On Stage I
   the same interventions cost -0.100, -0.023 and -0.002 AUPRC and up to -0.15 GS.
   Held-out test agrees (AUPRC, intervention = none 0.7452): gates_off 0.7446, mean 0.7454, permute_closure /
   rewire_bridge 0.7452, shuffle_graph 0.7375. The test transplant is the only non-null graph-content effect
   anywhere on Stage II (-0.008; V_val 0.000). The spec's +-0.01 margin is a GS margin, not an AUPRC one, so
   this is a small single-seed, single-permutation signal that needs replication, not a null.
2. **The channel is active and net-harmful.** `gates_off` raises V_val AUPRC 0.8039 -> 0.8136, `mean` (corpus
   mean template) to 0.8162; the prediction pushes every row down (mean -1.30 logits vs mean template), positives
   hardest (-1.68 vs -0.91), and self rows by -1.88 where the truth moves them 0.03. It inverts the useful contrast.
3. **No reader drift.** Reader x graph 2x2 on `val_cls`: Stage II reader on the TRUE template = 0.8867 AUPRC
   (Stage I bundle 0.8872); Stage I reader on the PREDICTED graph = 0.8010 (Stage II 0.8039). The reader swap is
   worth 0.0005, the graph swap 0.086. On `val_topology` the Stage II interface on the true graph reaches GS 0.527
   with MMD 8.6 / 3.2 / 7.2 (published Stage I: 0.538, 8.9 / 3.3 / 7.8).
4. **The interface is starved, not saturated.** Mixing 25% of the true template into the prediction recovers 71%
   of the AUPRC gap (0.804 -> 0.862 of 0.804 -> 0.887); 50% recovers 86%. Prefix gates did not shrink
   (mean tanh|alpha| 0.0612 -> 0.0618), and the reader's token Jacobian at the predicted graphs is 3x *larger*
   than at true graphs (208 vs 66 per row), so the reader is not flat where G's outputs live.
5. **G collapsed to a near-constant template, and the collapse was set up at initialisation.** Predicted
   wedge mass 0.0017 vs true 0.149 on V_val (1.2%); bridge mass 0.18 vs 0.89; the 64 interior weights of a row are
   one number (within-row std 8e-5); row-wise centred cosine to the truth 0.03; sorted-profile dispersion 1/117
   of the truth for wedge products. Its `L_slot` (0.135 train / 0.143 V_val) is *worse* than a single Adam-fitted
   96-weight constant (0.105 / 0.099), the spec 0.2 comparator, and the row-transplant rise is +3% / +1%: the
   pilot-B criterion for "not using row information". Cause: `init_biases` sets each gate-head bias to
   `logit(per-edge corpus mean)`; the closure mean 0.0185 is a *density* (58% of rows have no closure edge; the
   mean non-zero weight is 0.196), so the closure heads start at z = -3.97 and sit at mean -4.34 (sigmoid derivative
   0.015, 100% beyond |z|>3) for the whole run. `L_slot` reaches the 16 closure weights only through the 8
   products `w(u,c) w(c,v)`, the same quadratic attenuation section 7.3 diagnosed for the interior family and fixed
   there with a raw term; closure has no raw term. Measured slot gradient into the closure head is 1/1800 of the
   interior head's. The bias moved +0.004 logits in 5,413 steps.
6. **The composite has a cheap no-structure minimum.** Gradient into G at the trained checkpoint: task 5-18x the
   slot term (170-860x at initialisation); `L_topo` fell 0.70 -> 0.34 but a LayerNormed direction match is satisfied
   by a constant prompt (transplant rise +15%). Yet 200 AdamW steps on `L_slot` alone from `best.pt` at the run's
   own LR take it 0.247 -> 0.031 on a fixed batch: the generator can fit when the objective lets it.
7. **Output space.** 94.5% of training negatives have an empty closure family vs 21.4% of positives (interior
   74.8% vs 15.8%): the label is carried by family *presence*, which a dense sigmoid-gated 96-edge template cannot
   emit (global minimum weight 0.0025 on every row).

The spec's own rule (section 8) applies: Stage I gained and Stage II did not, so the remaining problem is
transfer, not the oracle. The mechanism analysis narrows "transfer" to a fixable training defect, not to the
attribute-to-structure ceiling measured on 2026-09-16 (kNN transfer 0.71-0.75 V_val AUROC): that ceiling was never
tested, because G never left its initialisation.

## 2. Decision: architecture revision of G and its objective, then one rerun; no HPO first

HPO over `w_slot`, `w_topo`, the interface LR or the warm-up cannot help while the closure gradient is
attenuated 1800x by a saturated initialisation and a product-only loss; every such trial would reproduce the
flat curve. The revision is small and targeted:

1. **Fix the gate-head initialisation.** Initialise each edge type's bias from the mean *non-zero* training
   weight (closure logit -1.41, not -3.97), or from a target family mass; keep the `[0.01, 0.99]` clip but apply
   it to the magnitude, not the density. Report the pre-activation distribution at init as telemetry.
2. **Add a raw closure term to `L_slot`** (`beta_c`, symmetric with `beta_a` / `beta_i`), or supervise
   `wedge_mass` directly beside the sorted products, so the label-carrying family receives an O(1) gradient.
3. **Give G a way to say "empty".** Either a per-family presence gate (one logit per family, multiplied into the
   family's edges, supervised by the compiled family's non-emptiness) or hard-concrete / sparsified edge gates
   with a straight-through estimator; the sorted-profile loss already pins the number of non-zero entries.
4. **Rebalance the composite** so the graph objective is not a rounding error: warm-start G for the two
   held-interface epochs on `L_slot` (+ `L_topo`) alone with the task loss detached from G, then open the
   task path; or weight `w_slot` by the measured task/slot gradient ratio at init. Log per-group gradient norms
   (closure / attach / interior heads) every epoch as telemetry.
5. **Run pilot B before the rerun** (spec 0.2; now cheap because `a1_collapse.py` already computes its
   quantities): the revised head must beat the asymmetric fitted constant on V_val `L_slot` with a transplant
   rise well above 3%, and a positive-row wedge-mass reconstruction well below 0.99. If the revised G still
   cannot beat the constant, the attribute-to-structure ceiling is the binding limit and the arm is closed on that
   evidence, which the current run does not provide.
6. Then rerun `motif_prompt_stage2` (one seed first) and read it against `prefix_base` at the protocol
   threshold and at matched density; the seed replicas and the section 8 controls only after the main arm
   leaves the base.

Not recommended: more Stage I lanes, a stronger oracle, the teachability pilots (the reader swap is worth
0.0005 AUPRC), or any change to the reader / adapter / trunk.

## 3. Evidence limits

**Integrated-gradients attribution is not usable as reported.** The completeness check in `a2_grad.json` does
not close: mean attribution sum vs mean actual logit change is 1.72 vs 1.26 (Stage I, mean -> true), -1.48 vs
-1.36 (Stage II, mean -> predicted) and 0.16 vs -0.63 (Stage II, mean -> true; the signs differ). The per-family
split (closure +2.07, interior -1.15) in `gradients_and_generator.md` therefore cannot be read as a mechanistic
attribution until the path integration (steps, determinism, AB/BA aggregation, baseline) is re-verified; nothing
in section 1 depends on it. Mean logit attribution would in any case not equal a contribution to AUPRC.

Single seed everywhere. The V_val true templates used by the crossed cells and the dose-response are a labelled
oracle diagnostic (spec section 8) and appear in no deployable row. The crossed `val_topology` cells are read at
Stage II's own frozen threshold, so their RD drifts; GS and the MMD ratios are the comparison. Gradient norms are
per-stream unscaled terms on 64-row batches; the trainer's stream averaging makes the slot share smaller, not
larger. The structural stream's gradient into G was read from `metrics.jsonl` only. `gates_off` reproduces
`prefix_base` to fp32 tolerance (published prefix_base scored its pair pass under bf16 autocast), not bit for bit.
The five `test_topology` intervention artifacts were scored but their fixed-threshold replay was not run
(`analyze_interventions.py` in `/2023533015/scratch_motif_verdict/interventions/` completes it on CPU).
