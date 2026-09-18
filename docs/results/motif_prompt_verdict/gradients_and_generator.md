# Why the motif-prompt Stage II generator learned nothing

Gradient / Jacobian / gate / generator-output analysis of `motif_prompt_stage2`
(`outputs/split_seed42/motif_prompt_stage2/best.pt`), against `motif_prompt_stage1/best.pt`
and `prefix_base/best.pt`. Run 2026-09-18 on H20 container 30030, single GPU, fp32 pair pass
(the published scorer's precision contract), bf16 encoder.

**Verdict in one line.** The failure is **generator output collapse to a near-constant,
strictly-positive template**, caused by **closure-gate saturation installed at initialisation**
plus a **task/`L_topo` gradient that outweighs `L_slot` by 18-858x**. The reader is *not* flat,
the prefix gates were *not* shrunk, and the `L_slot` gradient is *not* dead — 200 AdamW steps at
the run's own LR drop `L_slot` 8x from the trained checkpoint.

---

## 0. Instruments and rows

| item | value |
|---|---|
| rows, generator statistics | 5,000 training (2,500 pos / 2,500 neg, V_val-excluded, self rows dropped) + 5,000 V_val `val_cls` (2,500/2,500) |
| rows, gradients | 64-row batches, exactly 50/50, length-banded (30th-70th pct) so padding is bounded |
| true templates | `MotifTemplateTable(split.build_training_graph())`, `randomise=True, seed=0, epoch=1` (as the trainer compiles them); V_val on `build_g_val_simple()` (labelled diagnostic) |
| precision | encoder bf16 autocast then `.float()`; generator / reader / count head / trunk fp32, autocast disabled |
| scripts | `motif_common.py`, `a1_collapse.py`, `a2_grad.py`, `a3_tokens.py`, `a4_dynamics.py`, `a5_density.py` |
| remote outputs | `outputs/analysis/motif_verdict_20260918/gradients/{a1_collapse,a2_grad,a3_tokens,a5_density}.json` (`a4_dynamics` did not finish, §10) |

### What the target actually looks like (`a5_density.json`)

| family | nonzero fraction of its edges | mean nonzero value | rows with the **whole family empty** (train) | ... among positives | ... among negatives |
|---|---|---|---|---|---|
| closure (16) | 0.309 | 0.196 | 0.580 | 0.214 | **0.945** |
| attach (16) | 0.864 | 0.2415 | 0.0008 | 0.0008 | 0.0008 |
| interior (64) | 0.283 | 1.000 (binary) | 0.453 | 0.158 | **0.748** |
| all 96 | 0.384 | — | 0.0008 | | |

**The label lives in family presence/absence, not in edge magnitude.** 94.5% of training negatives
have an entirely empty closure family against 21.4% of positives (interior: 74.8% vs 15.8%).
True `wedge_mass` correlates with the label at r = 0.597 (train) / 0.670 (V_val).

---

## 1. Generator output collapse (A1) — **confirmed, and it is the primary locus**

### 1.1 The predicted 96 weights are a near-constant, dense template

Per edge type, training rows (V_val in brackets):

| type | predicted mean | std **across** rows | std **within** a row | corpus mean weight `Abar` |
|---|---|---|---|---|
| closure | 0.01554 [0.01539] | 0.00822 [0.00756] | 0.00489 [0.00537] | 0.01853 |
| attach | 0.23210 [0.23287] | 0.05624 [0.06134] | 0.04403 [0.04875] | 0.19454 |
| interior | 0.05874 [0.05487] | 0.01753 [0.01546] | **7.65e-05** [7.39e-05] | 0.09501 |

The 64 interior weights of a row are one number repeated (within-row std 7.6e-5 on a mean of
0.0587). Every one of the 96 weights is strictly positive on every row (global min 0.00246).

### 1.2 Sorted-profile dispersion, in `s_family` units (spec §0.2)

`s_family` = mean positive target coordinate of that sorted profile. Values are the mean over the
profile's coordinates of `std_across_rows / s_family`.

| profile | predicted (train) | true (train) | ratio | predicted (V_val) | true (V_val) |
|---|---|---|---|---|---|
| wedge products `p` | **0.0051** | 0.597 | 1 : 117 | 0.0029 | 0.527 |
| bridge path products `q` | **0.0269** | 0.553 | 1 : 21 | 0.0150 | 0.420 |
| raw attachments `a` | 0.1407 | 0.468 | 1 : 3.3 | 0.1400 | 0.349 |
| raw interior `b` | **0.0175** | 0.436 | 1 : 25 | 0.0155 | 0.407 |

### 1.3 Family masses, correlations and the §0.2 strata

Training rows (nonself):

| mass | pred mean | true mean | Pearson(pred,true) | Spearman | Pearson(pred,label) | Pearson(true,label) | false mass on zero-target rows | normalised recon. on positive-target rows | `s_family` | n0 / n+ |
|---|---|---|---|---|---|---|---|---|---|---|
| wedge | **0.00195** | 0.1117 | 0.221 | 0.369 | 0.312 | 0.597 | 0.00125 | **0.989** | 0.2658 | 2898 / 2102 |
| bridge | 0.1909 | 0.5683 | 0.172 | 0.254 | 0.196 | 0.574 | 0.1757 | 0.846 | 1.040 | 2267 / 2733 |
| deg_u | 1.991 | 2.193 | **0.014** | 0.017 | -0.107 | 0.367 | 2.090 | 0.375 | 2.230 | 82 / 4918 |
| deg_v | 1.971 | 2.116 | **0.029** | 0.041 | 0.044 | 0.399 | 2.087 | 0.387 | 2.165 | 111 / 4889 |

V_val: wedge 0.270 / 0.323, bridge 0.107 / 0.093, deg_u **-0.192** / -0.154 (sign flips).

The generator emits **1.7% of the true wedge mass** and reconstructs essentially none of it
(normalised reconstruction 0.989 ≈ "all of the mass missing"). The two endpoint degrees — the
easiest quantity in the whole target, since they are just `sum of deg^-0.5` over neighbours — are
**uncorrelated with truth** (r = 0.014 / 0.029) while truth correlates with the label at r ≈ 0.37-0.40.

### 1.4 `L_slot` against the pilot-B comparators — the generator **loses to a fitted constant**

| `L_slot` | train rows | V_val `val_cls` |
|---|---|---|
| generator (Stage II best.pt) | 0.1351 | 0.1425 |
| installed mean template `Abar` | 0.1334 | 0.1369 |
| **asymmetric fixed 96-weight template**, Adam-fitted on training targets (§0.2 comparator) | **0.1052** | **0.0987** |
| margin (generator - fitted constant) | **+0.0299 (worse)** | **+0.0438 (worse)** |
| seeded row-transplant `L_slot` | 0.1395 | 0.1444 |
| **transplant rise** | **+0.0044 (+3.2%)** | **+0.0019 (+1.4%)** |

Both pre-registered pilot-B readings fail: the margin over the fitted template is negative, and
"a head whose loss does not rise under transplant is not using row information" — the rise is 1-3%.
After 15 epochs the generator is, in `L_slot` terms, **exactly the constant template it was
initialised to emit**, and a plain fitted constant would have been 22-31% better.

> Note on the logged curve. `train_motif_slot_loss` (0.0726 -> 0.070) is the trainer's
> `stream_mean` over the task and structural streams (`_motif_stream_terms` -> `stream_mean`), i.e.
> the task stream enters at weight 1/2. The task-stream value measured directly here is **0.135**,
> and `0.135 / 2 = 0.0675`, which is where the logged curve sits. `L_slot(Abar) = 0.133`. Either way
> the quantity never moved: it is the mean template's loss from the first epoch to the last.

---

## 2. Gate-head saturation (A2) — **confirmed for the label-carrying family**

Gate pre-activations `z` (before the sigmoid), pooled over 5,000 rows x that type's edges:

| type | mean `z` | std | frac `|z| > 3` | frac `|z| > 4` | mean sigmoid derivative |
|---|---|---|---|---|---|
| **closure** | **-4.337** | 0.675 | **1.000** | **0.561** | **0.0152** |
| interior | -2.820 | 0.325 | 0.312 | 0.000 | 0.0550 |
| attach | -1.221 | 0.306 | 0.000 | 0.000 | 0.1751 |

V_val is identical to three digits (closure -4.326, 57.1% beyond |z| > 4).

### 2.1 The saturation is installed at initialisation, by `init_biases`

`install_mean_template` -> `MotifGenerator.init_biases` sets each head's output bias to
`logit(mean training weight of that edge type)`, clipped to [0.01, 0.99]:

| type | corpus mean weight | clip applied? | bias at init | sigma' at init | **trained bias** | drift over 15 epochs / 5,413 steps |
|---|---|---|---|---|---|---|
| closure | 0.018528 | no (above 0.01) | **-3.9698** | **0.0182** | -3.9661 | **+0.0037** |
| interior | 0.095015 | no | -2.2539 | 0.0860 | -2.2541 | -0.0002 |
| attach | 0.194544 | no | -1.4207 | 0.1567 | -1.4131 | +0.0077 |

The clip `[0.01, 0.99]` never fires, and it is the wrong guard. The closure family's *per-edge
corpus mean* is 0.0185 **because 58% of rows have no closure edges at all** — it is a density,
not a typical magnitude (the mean *nonzero* closure weight is 0.196, `logit = -1.41`). Initialising
the shared closure head at `logit(density)` puts every closure gate 4 logits deep in the flat
region that spec §4 says the reparameterisation exists to avoid. The biases then move by 0.004
logits in the entire run.

The spec's *predecessor* failure mode did **not** recur: LayerNorm input norms are 6.9-7.6
(closure 7.20, attach 7.64, interior 7.52) and slot-state norms average 3.61 — no norm blow-up.
Head output weights grew from `std 1e-3` to `|w|mean` 0.0069-0.0134, so the heads are not frozen;
they are flat.

### 2.2 The consequence in gradient terms

`||dL_slot/d theta||` restricted to each head (64-row balanced train batch, best.pt):

| head | `L_slot` grad norm | `L_task` grad norm | `0.1*L_topo` grad norm |
|---|---|---|---|
| closure | **3e-05** | 0.1876 | 0.0307 |
| attach | 0.0199 | 0.4339 | 0.0175 |
| interior | 0.0546 | 0.9504 | 0.0088 |

The closure head — the only family that carries the label — receives ~**1/1800** of the interior
head's `L_slot` gradient. `L_slot` is effectively blind to closure at this operating point.

### 2.3 Why the closure gradient is ~0: a double attenuation `L_slot` has no term to repair

`slot_profiles` supervises the 16 **closure edge weights only through their 8 products**
`p_c = w(u,c)*w(c,v)`. The other two families each additionally carry a *raw* term
(`a` = the 16 raw attachments, `b` = the 64 raw interior weights). Spec §7.3 introduced the raw
interior term precisely because "for `q = a b c`, `dq/db = a c`, so the interior gradient is
attenuated quadratically by small attachments" — **the closure family has exactly that defect and
no corresponding raw term.**

With the measured operating point (`w_closure = 0.0155`, `sigma'(z) = 0.0152`, Huber in its
quadratic region since `|p - p*| < 1`):

```
dL/dz_closure = dL/dp * dp/dw       * dw/dz
              = O(1e-2) * w_partner * sigma'(z)
              = O(1e-2) * 0.0155    * 0.0152   ~= 2e-6 per edge
dL/dz_interior= dL/db               * sigma'(z)
              = O(3e-3) * 1         * 0.0550   ~= 2e-4 per edge
```

Two multiplicative attenuations (`0.0155 * 0.0152 = 2.4e-4`) apply to the closure family and one
(`0.055`) to the interior family — matching the measured head gradient ratio of 3e-5 : 5.5e-2.
Worse, the two attenuations are **self-reinforcing**: the product term can only grow the closure
weight by a factor proportional to the closure weight itself, so a head initialised at
`w = 0.0155` cannot escape, and `wedge_mass` stays at `8 * 0.0155^2 = 0.0019` against the target
0.112. For comparison, the Adam-fitted constant template sets the closure weights to 0.114,
i.e. `wedge_mass = 8 * 0.114^2 = 0.104`, which matches the corpus mean almost exactly — the correct
constant is reachable in principle and `L_slot` simply cannot walk there from `logit(density)`.

---

---

## 3. Prefix gates and token fields (A3) — **gates were NOT shrunk; tokens lose ~half their spread**

### 3.1 `tanh(alpha)` per layer/site/head, Stage I bundle vs Stage II best.pt

| | mean `|tanh a|` | max | layer 0 / 1 / 2 | site A<-B / B<-A / CLS |
|---|---|---|---|---|
| Stage I | 0.06121 | 0.1632 | 0.0493 / 0.0603 / 0.0740 | 0.0567 / 0.0492 / 0.0777 |
| Stage II | 0.06180 | 0.1657 | 0.0497 / 0.0606 / 0.0751 | 0.0576 / 0.0499 / 0.0779 |

Stage II **grew** the gates by 0.9%. Shrunken prefix gates are ruled out.

### 3.2 Token fields across 3,000 training rows

`std_norm` = L2 norm of the per-dimension across-row std; `cos` = mean pairwise cosine between rows.

| source | topo_u | topo_v | topo_rel | topo_cnt |
|---|---|---|---|---|
| Stage II @ **predicted** graphs | 5.18 / cos 0.796 | 5.02 / 0.782 | 4.89 / 0.734 | **0.59 / 0.995** |
| Stage II @ true graphs | 7.64 / 0.430 | 7.73 / 0.472 | 9.78 / 0.331 | 2.72 / 0.928 |
| Stage I @ true graphs | 6.64 / 0.562 | 6.77 / 0.598 | 9.98 / 0.316 | 2.72 / 0.928 |
| @ mean template (reference) | 0.00 / 1.000 | 0.00 / 1.000 | 0.00 / 1.000 | 0.00 / 1.000 |

Prefix rows (across-row std of the projected rows, layers 0/1/2): Stage II @ predicted
37.9 / 44.6 / 54.1 against Stage II @ true 71.7 / 81.5 / 93.6 and Stage I @ true 67.8 / 77.4 / 90.2.

The tokens are **not** constant — so "the prompt cannot condition anything" is too strong — but they
carry ~50-65% of the true-graph across-row spread for the three reader fields and only 22% for
`topo_cnt` (inter-row cosine 0.995: the count token is essentially one vector for every pair).

### 3.3 `L_topo`, the term that *did* move

| | value (train rows, through the immutable teacher) |
|---|---|
| generator's predicted graph | 0.4841 |
| installed mean template | 0.7823 |
| true graph (sanity) | 3.5e-14 |
| generator under seeded row transplant | 0.5555 (**+14.8%**) |

`train_motif_topo_loss` fell 0.699 -> 0.342 across the run. So the generator *did* optimise —
it moved the LayerNormed token directions 38% of the way from `Abar` towards the teacher, but only
15% of that is row-specific. `L_topo` compares directions after a non-affine LayerNorm, so it is
satisfiable by a nearly row-independent shift; it is the one graph-side term the collapse is
compatible with, and it is the one that moved.

---

## 4. Gradient flow into G (A4) — **the task gradient dominates; `L_slot` is the negligible term**

Per-term gradient norms restricted to the generator's parameters, 64-row 50/50 batches, fp32,
`model.train()`, per-stream unscaled terms as `_add_composite_loss` emits them.

| model / batch | `||∇_G L_task||` | `||∇_G L_slot||` | `||∇_G 0.1·L_topo||` | task/slot | task/topo |
|---|---|---|---|---|---|
| **best.pt, train** | **1.388** | **0.0772** | 0.1011 | **18.0** | 13.7 |
| best.pt, V_val | 6.160 | 0.1008 | 0.1944 | 61.1 | 31.7 |
| **initialisation** (Stage I bundle + fresh G) | **45.47** | **0.0530** | 6.922 | **858** | 6.6 |

Per group at best.pt (train), `L_task` / `L_slot`:
residue_proj 0.693 / 0.038, attention 0.195 / 0.013, mpnn 0.451 / 0.026,
head_closure 0.188 / **0.00003**, head_attach 0.434 / 0.020, head_interior 0.950 / 0.055,
slot_queries ~1e-4 / ~1e-6.

**The answer to the pre-registered question is the opposite of the hypothesis.** The task gradient
into G is not negligible; it is 18x the slot gradient at the trained checkpoint and 858x at
initialisation (where it is also clipped from 45.5 to 1.0, so the first epochs are almost purely a
clipped task direction). The graph-content term is the one that is negligible — and within it, the
label-carrying closure family is zero.

`L_task` can be reduced by *any* row-dependent perturbation of the four tokens, and from epoch 3 the
interface group (reader final block, count head, adapter/gates) trains at 0.1x and can re-read
whatever G emits. The composite therefore has a cheap no-structure minimum, and that is the one it
found: `train_loss` 0.2158 -> 0.1873, `L_topo` 0.699 -> 0.342, `L_slot` flat.

### 4.1 Saliency `d logit / d w` (per-row L2 over the 96 weights)

| | median | mean | p05 | rows with `||.|| < 1e-3` |
|---|---|---|---|---|
| Stage II @ predicted graphs (train) | 13.78 | 13.99 | 0.022 | **0.000** |
| Stage II @ predicted graphs (V_val) | 12.57 | 11.55 | 2.77 | 0.000 |
| Stage I @ true graphs (train) | 12.60 | 3.2e4 | 0.174 | 0.000 |
| Stage II reader @ true graphs (train) | 15.86 | 2.9e4 | 0.207 | 0.000 |

By family @ predicted graphs (per-row L2, median): closure 11.90, interior 5.49, attach 3.81.
No row has a vanishing saliency. (The huge *means* on true graphs come from near-empty templates,
where `dense_rrwp`'s `clamp_min(1e-6)` degree normalisation is numerically singular — a real
distribution difference between the compiled and the generated graphs, discussed in §6.)

---

## 5. Reader sensitivity (A5) — **the reader is NOT in a flat region; refuted**

Exact per-row Frobenius norm of the Jacobian of the four token fields w.r.t. the 96 weights
(96 rows, all 512 output dimensions differentiated):

| graph the reader sits on | median `||J||_F` | closure cols | attach cols | interior cols |
|---|---|---|---|---|
| Stage II @ **predicted** graphs | **208.1** | 191.0 | 65.4 | 54.9 |
| Stage II @ mean template | 515.2 | 341.0 | 346.5 | 170.5 |
| Stage II @ true graphs | 65.8 | 46.6 | 25.6 | 0.52 |
| Stage I @ true graphs | 61.7 | 44.9 | 25.6 | 0.51 |

The predicted graphs sit in a region **3x more sensitive** than the true graphs. Flatness is not the
problem.

### 5.1 Integrated gradients of the logit, mean template -> graph (24 steps, completeness checked)

| path | total attribution | closure | attach | interior | mean logit shift |
|---|---|---|---|---|---|
| Stage I: `Abar` -> **true** graph | **+1.723** | **+2.070** | +0.149 | -0.497 | +1.258 |
| Stage II: `Abar` -> true graph | +0.156 | +0.727 | +0.026 | -0.597 | -0.633 |
| Stage II: `Abar` -> **predicted** graph | **-1.481** | **+0.080** (std 0.179) | -0.408 | -1.153 | **-1.359** |

This is the cleanest single statement of the failure. Stage I's gain is carried by the closure
channel: it moves the logit by **+2.07** on true templates. The Stage II generator's closure channel
moves it by **+0.08**, a 26x reduction with a cross-row spread of 0.18. What the predicted prompt
*does* transmit is a large, mostly row-independent **negative** shift dominated by the interior
family (-1.15) — i.e. an offset, not a discriminant.

---

## 6. Optimisation sanity (A6) — **the gradient is alive; training simply never used it**

From the trained `best.pt` generator, on one fixed 40-row training batch, `L_slot` alone:

| optimiser | step 0 | 30 | 60 | 90 | 120 | 199 |
|---|---|---|---|---|---|---|
| AdamW lr **1e-4** (the run's peak LR), clip 1.0 | 0.2473 | 0.2162 | 0.1690 | 0.0825 | 0.0374 | **0.0306** |
| AdamW lr 1e-3 | 0.2473 | 0.0321 | 0.0111 | 0.0044 | 0.0037 | **0.0012** |

Reference floors on that batch: fitted constant template 0.1222; free per-row weights 7.5e-05.

**200 steps at the run's own learning rate take `L_slot` 8x below its trained value and below the
constant floor.** The loss is not at a floor and the gradient is not dead. The generator did not fit
the graph because, inside the composite, the `L_slot` direction is ~5% of the gradient it receives
and the closure part of it is ~0.

---

## 7. What the training curves say, with these numbers attached

| epoch | Stage II V_val AUPRC | GS | `L_slot` (2-stream) | `L_topo` | prefix_base V_val AUPRC | Stage I V_val AUPRC / GS |
|---|---|---|---|---|---|---|
| 1 | 0.8055 | 0.3966 | 0.0726 | 0.699 | 0.7671 | 0.8163 / 0.4174 |
| 5 | 0.8013 | 0.3886 | 0.0705 | 0.373 | 0.8104 | 0.8687 / 0.5186 |
| 10 | 0.8039 | 0.3943 | 0.0711 | 0.353 | **0.8130 / 0.4013 (selected)** | 0.8848 / 0.5461 |
| 15 | 0.8027 | 0.3926 | 0.0700 | 0.342 | — | **0.8859 / 0.5463** |

Stage II starts at its frozen trunk's level and stays there, at or slightly **below** `prefix_base`'s
own selected V_val row, for all 15 epochs. `grad_norm_struct_degree` = 149.5 at epoch 1 against
`grad_clip: 1.0`, so the first epoch's updates are scaled by ~1/150 and are a near-pure structural /
task direction.

---

## 8. Failure locus, ranked

1. **Generator output collapse** (primary). The 96 predicted weights are a dense, near-constant
   template: sorted-profile dispersion 21-117x below truth on the two product families, 64 interior
   weights identical within a row to 1e-4, `L_slot` equal to the installed mean template and 22-31%
   *worse* than a plain fitted constant, transplant rise 1-3%, endpoint degrees uncorrelated with
   truth (r = 0.01-0.03).
2. **Closure-gate saturation, installed by `init_biases`** (the mechanism). `logit(mean training
   weight)` for a family that is *absent* on 58% of rows is `logit(density) = -3.97`; 100% of closure
   gates sit beyond `|z| > 3`, 56% beyond `|z| > 4`, mean sigma' = 0.015; the `L_slot` gradient into the
   closure head is 3e-5, 1/1800 of the interior head's; the closure bias moved 0.004 logits in 5,413
   steps. The `[0.01, 0.99]` clip guards against 0/1, not against this.
3. **`L_slot` is outweighed 18:1 (858:1 at init) by `L_task`, and 1.3:1 by `0.1*L_topo`** — and `L_topo`
   is a LayerNormed direction match that a near-constant prompt can satisfy (it fell 0.699 -> 0.342
   with only a 15% transplant rise). The composite's cheapest descent direction never passes through
   the graph.
4. **A reachability problem under the collapse**: the label is carried by family *presence*
   (94.5% of negatives have zero closure mass vs 21.4% of positives). A sigmoid cannot emit 0, so the
   deployable generator puts positive mass on all 96 edges of every row; the binary contrast Stage I
   reads is outside its output set. Its closure channel transmits +0.08 logit against Stage I's +2.07.

**Ruled out:** flat reader (Jacobian at predicted graphs is 3x *larger* than at true graphs,
zero rows with vanishing saliency); shrunken prefix gates (`tanh|alpha|` 0.0612 -> 0.0618, +0.9%);
dead `L_slot` gradient (200 steps at lr 1e-4 drop it 8x); LayerNorm input-norm blow-up
(norms 6.9-7.6, the predecessor's 23 -> 162 failure did not recur).

---

## 9. Commands run

All analysis scripts were written locally, copied to `/2023533015/scratch_motif_verdict/gradients/`
on H20 container 30030 and run with the checkout as cwd. Nothing in the repository working tree
(local or remote) was modified; no process was started or stopped other than these scripts.

```bash
ssh -o BatchMode=yes -p 30030 root@10.15.171.204 \
  'mkdir -p /2023533015/scratch_motif_verdict/gradients \
            /2023533015/topology-conditioned-inductive-edge-prediction/outputs/analysis/motif_verdict_20260918/gradients'
scp -P 30030 motif_common.py a1_collapse.py a2_grad.py a3_tokens.py a4_dynamics.py a5_density.py \
  root@10.15.171.204:/2023533015/scratch_motif_verdict/gradients/

# each run, from /2023533015/topology-conditioned-inductive-edge-prediction:
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
PYTHONPATH=/2023533015/topology-conditioned-inductive-edge-prediction CUDA_VISIBLE_DEVICES=<0|1|2|3> \
  .venv/bin/python /2023533015/scratch_motif_verdict/gradients/<script>.py

# curves
.venv/bin/python -c "import json; [print(json.loads(l)['epoch'], ...) for l in open('outputs/split_seed42/<arm>/metrics.jsonl')]"
```

Script -> analysis map: `a1_collapse.py` = A1 + A2 + the A3 gate table; `a2_grad.py` = A4 + A5 + A6;
`a3_tokens.py` = A3 tokens/prefix rows, `L_topo` comparators, corrected label-balanced A4;
`a5_density.py` = the target-sparsity table; `a4_dynamics.py` = cross-batch gradient consistency and
a 300-step composite-vs-slot-only optimisation (supplementary).

## 10. Caveats and what was not run

- Gradient norms are measured in **fp32 with `model.train()`** on a 64-row batch and use the
  per-stream *unscaled* terms `loss_term_{task,slot,topo}` as `_add_composite_loss` emits them. The
  trainer additionally averages each graph term over the task and structural streams (a further
  factor ~0.5 on `L_slot`) and clips the *combined* gradient — including the structural stream — to
  1.0, so the slot share of a real update is smaller than the 1:18 measured here, not larger.
- The **structural stream was not reproduced** in any of these measurements; its contribution to the
  gradient into G is known only through the logged `grad_norm_struct_*` (0.3-5 after epoch 2,
  27.6/149.5 at epoch 1) against `grad_clip: 1.0`.
- V_val true templates were compiled from the V_val gold graph, which is a **labelled oracle
  diagnostic**; they are used only to measure the generator's fit, never to select anything.
- Predicted-vs-true mass correlations are computed on **nonself rows only**; self rows are excluded
  from `L_slot`/`L_topo` by `motif_mask` and would inflate every correlation.
- Integrated gradients use 24 midpoint steps; the completeness residual was checked against the
  measured logit shift on every path (agreement to 3 decimals).
- **One planned supplementary run did not complete.** `a4_dynamics.py` (cross-batch gradient
  consistency over 24 independent batches, and a 300-step `composite` vs `slot_only` vs `task_only`
  optimisation of G from `best.pt`) was killed by its own 90-minute `timeout` before writing output:
  it re-probes sequence lengths from the feature store on every step, which dominates the wall clock.
  It would have answered "is the `L_slot` gradient consistent enough across batches for Adam to
  follow it inside the composite?"; nothing in the verdict above depends on it, and §6 already shows
  the gradient is followable when it is the only term. Re-run with precomputed lengths if wanted.
- No linear probe was used anywhere in this analysis.
