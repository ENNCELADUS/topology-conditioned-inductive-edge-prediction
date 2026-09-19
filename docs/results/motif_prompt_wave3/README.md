# Motif-graph GRIT prompt, wave 3: the adopted generator and its held-out test row

Date: 2026-09-19. Headline split (seed 42, root `node_007630`), training seed 0, branch
`codex/motif-graph-grit-prompt` at `39ffdd0`. Spec:
[`2026-09-16-motif-graph-grit-prompt-design.md`](../../superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md)
(v16). Wave-1 verdict and the mechanism that closed it:
[`motif_prompt_verdict/README.md`](../motif_prompt_verdict/README.md).

Subject: `motif_prompt_stage2_v3_prefix`, the eight-epoch graph-only warm-up of the adopted Stage II
generator (`configs/split_seed42/motif_prompt_stage2_v3_prefix.yaml`, attempt
`5e801f7e6dc84e42b9253bed1b2418da`, launched `--skip-test`). Because the joint phase is closed (below),
this warm-up generator read through the frozen Stage I interface **is** the deployable wave-3 model, and
this note reports its held-out test row.

## 1. The adopted configuration, and why

Wave 1 died in the generator, not the reader: `G` emitted a near-constant, slot-symmetric graph whose
closure heads were initialised into saturation by `logit(density)` and supervised only through the
quadratic `w(u,c) w(c,v)` products, inside a composite whose task gradient outweighed the graph term
5-18x. Wave 2 fixed the initialisation (`closure_bias_init: nonzero_mean`, F1) and detached the task,
structural and `L_topo` gradients from `G` during the warm-up (`warmup_losses: graph_only`, F3), which was
not enough on its own. Wave 3 adds three keys that **pass only together**: `slot_read: residual_block`
(F4 — the bare read returned the attention output alone, so a near-uniform attention gave every slot the
same mean of `V`), `slot_read_value_norm: false` (the residual read over raw states, restoring the
magnitude information `LN_h` throws away), and `graph_row_weighting: closure_balanced` at
`graph_row_positive_share: 0.5` (about 82% of the 1:5 task stream's rows have an empty closure family, so
the uniform `L_G` drifts to that majority). Each alone leaves `G` on the empty-closure constant, measured
as two-epoch prefixes read by `src.experiments.motif_pilot_b`:

| two-epoch prefix | keys beyond F1 + F3 | level 1 | level 2 | level 3 |
|---|---|---|---|---|
| `motif_prompt_stage2_v3_prefix` (superseded) | F4 residual read, `LN_h` values | FAIL | FAIL | FAIL |
| `motif_prompt_stage2_v3_novln_prefix` | F4 + value norm off | FAIL | FAIL | FAIL |
| `motif_prompt_stage2_v3_balanced_{full,initonly}_prefix` | closure-balanced rows alone | FAIL | FAIL | PASS |
| `motif_prompt_stage2_v3_novln_balanced_prefix` | all three | **PASS** | **PASS** | **PASS** |

The interface-open regime is closed by a causal test rather than by preference. On the *previous*
generator's eight-epoch warm-up, opening the interface jointly lost the fit within one epoch (V_val AUPRC
0.816 -> 0.800 at epoch 9, closure gates re-saturated, task gradient into the closure head 1400x the graph
gradient by epoch 10), and five-rank selection fell back to warm-up epoch 6. The detached continuation
(`warmup_losses: graph_only_always`, `G` protected while the interface adapts) gained nothing either: it
reached 0.8105 at epoch 9 and settled at 0.8080, never above the warm-up's own 0.8159, so its five-rank
selection also returned warm-up epoch 6 and `motif_prompt_stage2_v3_initonly_warm8` and
`motif_prompt_stage2_v3_initonly_warm8_detached` published the **same checkpoint**
(`0b3c5aab6724f7ae`) and have identical test rows. The deployable wave-3 model is therefore the warm-up
generator itself, published at its five-rank-selected epoch.

## 2. Per-epoch pilot B over the eight-epoch warm-up

`outputs/analysis/motif_pilot_b_v3_main_warm8/epoch-000N/pilot_b.md` on the H20, one read per epoch
checkpoint of the warm-up. Level 1 compares `L_G` against the asymmetric constant template re-fitted by
Adam under the same `L_G` (**0.05945** on 4,999 held-out training rows, **0.10808** on the 10,050 `val_cls`
rows; the corpus mean template is 0.06779 / 0.13409) and requires positive-row wedge reconstruction < 0.7
and predicted mean wedge mass within 3x of the truth's. Level 2 requires every one of ten seeded row
transplants to raise `L_G` by at least +25% on every universe. Level 3 requires the predicted graph,
through both the frozen Stage I bundle and the prefix's own interface, to beat `gates_off` on `val_cls`
AUPRC (**0.8136**; the true-template ceiling is 0.8872).

| ep | `L_G` train | `L_G` val_cls | recon train / val_cls | pred/true wedge mass train / val_cls | transplant rise train / val_cls | `s2_pred` AUPRC | L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.06255 | 0.09761 | 0.648 / 0.683 | 3.038 / 0.845 | +35.3% / +11.0% | 0.8161 | FAIL | FAIL | PASS |
| 2 | 0.05094 | 0.09341 | 0.562 / 0.648 | 2.066 / 0.752 | +93.1% / +39.1% | **0.8176** | PASS | PASS | PASS |
| 3 | 0.03515 | 0.09645 | 0.576 / 0.683 | 1.608 / 0.626 | +133.2% / +32.7% | 0.8127 | PASS | PASS | FAIL |
| 4 | 0.03834 | **0.08431** | 0.522 / 0.612 | 2.219 / 0.867 | +138.8% / +44.8% | 0.8151 | PASS | PASS | PASS |
| 5 | 0.03629 | 0.09004 | 0.511 / 0.616 | 2.235 / 0.880 | +160.3% / +40.3% | 0.8150 | PASS | PASS | PASS |
| 6 | 0.02982 | 0.10186 | 0.515 / 0.675 | 1.763 / 0.652 | +199.3% / +30.0% | 0.8121 | PASS | PASS | FAIL |
| 7 | 0.02907 | 0.10170 | 0.506 / 0.665 | 1.808 / 0.676 | +211.3% / +29.3% | 0.8124 | PASS | PASS | FAIL |
| 8 | 0.02863 | 0.09839 | 0.508 / 0.678 | 1.742 / 0.647 | +222.1% / +31.5% | 0.8116 | PASS | PASS | FAIL |

`s2_pred` and `s1_pred` are equal at every epoch (the reader swap is worth nothing, as in wave 1), and
`s2_true` = `s1_true` = 0.8872 throughout. The generator keeps improving on the training-side `L_G` for
all eight epochs while the V_val fit turns at epoch 4: the held-out graph fit, not the training fit, is
what the warm-up length buys. Level 3 passes only at epochs 2, 4 and 5 and is never worth more than
+0.004 AUPRC over `gates_off`.

The run's own V_val selection stream (`metrics.jsonl`) agrees on epoch 2 and 4 as the two candidates:

| ep | V_val AUPRC | GS | RD | degree / clustering / spectral MMD | threshold |
|---|---|---|---|---|---|
| 1 | 0.8161 | 0.3914 | 1.077 | 7.72 / 3.64 / 7.05 | 3.255 |
| 2 | 0.8176 | 0.3948 | 1.073 | 6.63 / 2.94 / 6.68 | 2.821 |
| 3 | 0.8127 | 0.3926 | 1.125 | 8.76 / 3.91 / 8.11 | 2.926 |
| 4 | 0.8151 | 0.3958 | 1.101 | 7.34 / 3.25 / 7.23 | 3.212 |
| 5 | 0.8150 | 0.3984 | 1.107 | 7.49 / 3.35 / 7.28 | 3.121 |
| 6 | 0.8121 | 0.3951 | 1.129 | 7.88 / 3.70 / 7.43 | 2.816 |
| 7 | 0.8124 | 0.3983 | 1.120 | 8.32 / 3.67 / 7.64 | 2.857 |
| 8 | 0.8116 | 0.3966 | 1.125 | 8.69 / 3.70 / 7.78 | 2.822 |

Five-rank selection (`geometric_rd_five_rank_v1`) published epoch 2 as `best.pt`.

## 3. Held-out test

`test_protocol_v8`, one V_val-selected fixed threshold (closest geometric RD) replayed on every test
subgraph; Accuracy/F1/MCC at the separate max-F1 threshold frozen on `val_cls`; ECE/Brier on raw
probabilities. Scored on four H20 GPUs from the published checkpoints alone — no truth graph, no bank, no
universe statistic — with `hpc/run.sh test`. Reference rows are the wave-1 verdict's, re-read from their
own `test_report.json`.

| Row | sel. ep | AUROC | AUPRC | non-self AUPRC | F1 | MCC | ECE | Brier | GS | RD | deg / clus / spec MMD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 8 | 0.701 | 0.736 | 0.697 | 0.671 | 0.228 | 0.202 | 0.266 | 0.429 | 0.550 | 9.9 / 8.4 / 15.0 |
| `prefix_base` (the frozen trunk) | 10 | 0.720 | 0.744 | 0.704 | 0.679 | 0.279 | 0.293 | 0.315 | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 |
| Stage I, true templates (ceiling) | 11 | 0.785 | 0.810 | 0.783 | 0.709 | 0.406 | 0.269 | 0.280 | 0.492 | 0.483 | 10.8 / 9.7 / 15.2 |
| Wave-1 Stage II | 10 | 0.722 | 0.745 | 0.707 | 0.679 | 0.282 | 0.302 | 0.322 | 0.416 | 0.491 | 10.6 / 9.0 / 15.4 |
| Wave-3 warm8 joint / detached (same ckpt) | 6 | 0.720 | 0.744 | 0.704 | 0.675 | 0.289 | 0.220 | 0.271 | 0.405 | 0.481 | 10.4 / 9.0 / 15.8 |
| **Wave-3 deployable, epoch 2 (published)** | 2 | 0.717 | 0.742 | 0.701 | 0.675 | 0.281 | 0.214 | 0.268 | 0.404 | 0.529 | **8.4 / 7.3 / 13.3** |
| Wave-3 epoch 4 (secondary row) | 4 | 0.717 | 0.741 | 0.699 | 0.672 | 0.287 | 0.219 | 0.270 | 0.400 | 0.473 | 10.6 / 9.2 / 15.9 |

V_val replay at the same frozen thresholds (the operating point each row was selected at):

| Row | GS | RD | deg / clus / spec MMD |
|---|---|---|---|
| B0 | 0.400 | 1.081 | 12.5 / 4.8 / 10.7 |
| `prefix_base` | 0.401 | 1.102 | 10.6 / 4.6 / 9.0 |
| Stage I (ceiling) | 0.547 | 1.062 | 9.8 / 3.9 / 8.9 |
| Wave-1 Stage II | 0.394 | 1.148 | 12.8 / 5.5 / 9.7 |
| Wave-3 warm8 joint / detached | 0.391 | 1.108 | 8.0 / 3.5 / 7.4 |
| **Wave-3 deployable, epoch 2** | 0.395 | 1.073 | 6.6 / 2.9 / 6.7 |
| Wave-3 epoch 4 | 0.396 | 1.101 | 7.3 / 3.3 / 7.2 |

**Five-number topology line (test, the V_val-selected fixed threshold).** Wave-3 deployable epoch 2:
GS 0.404, RD 0.529, degree / clustering / spectral MMD ratio 8.4 / 7.3 / 13.3, against `prefix_base`
GS 0.409, RD 0.456, 11.4 / 9.7 / 16.8 and B0 GS 0.429, RD 0.550, 9.9 / 8.4 / 15.0. Read together with the
edge block above, never alone.

## 4. Reading

**Edge: the wave-3 deployable row does not leave the trunk.** AUROC 0.717 / AUPRC 0.742 / non-self AUPRC
0.701 against `prefix_base`'s 0.720 / 0.744 / 0.704 — at or a few thousandths below the frozen trunk it
prompts, and far below the Stage I ceiling (0.785 / 0.810 / 0.783) that the same interface reaches on true
templates. The epoch-4 secondary row is the same (0.717 / 0.741 / 0.699). F1 and MCC are likewise inside
the wave-1 band. ECE and Brier are better than `prefix_base` (0.214 / 0.268 and 0.219 / 0.270 against
0.293 / 0.315), but the warm8 row shares that (0.220 / 0.271), so it is a property of the graph-only
warm-up schedule, not of the adopted generator.

**Topology: one movement, and it tracks output density.** The published epoch-2 row is the only Stage II
row whose test MMD ratios leave the `prefix_base` band — 8.4 / 7.3 / 13.3 against 11.4 / 9.7 / 16.8, well
beyond the pre-registered ±0.5 MMD reporting margin — with GS 0.404 vs 0.409 (inside the ±0.01 GS margin,
so not read) and RD 0.529 vs 0.456. But it comes with a looser operating point: epoch 2's V_val-frozen
threshold is 2.821 against epoch 4's 3.212, and the epoch-4 row, at RD 0.473, is back in the band at
10.6 / 9.2 / 15.9 with GS 0.400. Within this family the MMD ratios move with RD and nothing else moves,
which is the same finding the wave-1 matched-density diagnostic reported for Stage II. Against B0, whose
RD 0.550 is the nearest density among the reference rows, the epoch-2 row has better MMD ratios
(8.4 / 7.3 / 13.3 vs 9.9 / 8.4 / 15.0) and better edge metrics but worse GS (0.404 vs 0.429). No
density-matched control was trained for wave 3, so shape-versus-placement is not settled here; the
spec §8 output-density control remains the standing evidence gap.

**Verdict.** The wave-3 fix does what it was designed to do at the generator: this is the first Stage II
generator to beat the asymmetric fitted constant on both universes, to depend on the row (transplant rise
+93% / +39% at the selected epoch against wave 1's +3% / +1%), and to carry that through to `val_cls`
AUPRC above `gates_off`. None of it converts into a held-out edge gain, and the topology movement is
confounded with threshold placement. A single seed, one permutation, and no density-matched control.

## 5. Artifacts and how these rows were produced

All on the H20 shared checkout `/2023533015/topology-conditioned-inductive-edge-prediction`, four GPUs.

```bash
# Published best.pt (epoch 2) -- the deployable wave-3 row
hpc/run.sh test --checkpoint outputs/split_seed42/motif_prompt_stage2_v3_prefix/best.pt \
  --output-dir outputs/split_seed42/motif_prompt_stage2_v3_prefix \
  --pack-dir outputs/feature_packs/b0_v31_bf16 --data-root data \
  --strategy breadth_first --arm v3_1_motif_prompt --seed 0

# Epoch-4 secondary row
hpc/run.sh test --checkpoint outputs/split_seed42/motif_prompt_stage2_v3_prefix_epoch4/best.pt \
  --output-dir outputs/split_seed42/motif_prompt_stage2_v3_prefix_epoch4 \
  --pack-dir outputs/feature_packs/b0_v31_bf16 --data-root data \
  --strategy breadth_first --arm v3_1_motif_prompt_epoch4 --seed 0
```

`hpc/run.sh test` is a passthrough to `src.eval.test_protocol` and writes `test_report.json` only; the
`test_complete.json` sentinel is written by `src.e2_pipeline`, so neither row has one; both logs end at
`wrote ... test_report.json` with no error.

The epoch-4 checkpoint is not a published artifact, so it was republished into its **own** directory,
`outputs/split_seed42/motif_prompt_stage2_v3_prefix_epoch4/`, leaving the source run untouched:
`attempts/5e801f7e6dc84e42b9253bed1b2418da/checkpoints/epoch-0004.pt` copied to `best.pt` with the two
fields training publication adds — `selection_rule: geometric_rd_five_rank_v1` and
`val_threshold_transfer: {n_val: 869, threshold: 3.2118051052093506}` — taken verbatim from that epoch's
own `selection_metrics["val_threshold"]`, which is the same quantity
`src/train_b0.py` transfers for the selected epoch (and `src/experiments/reselect_campaign.py` for an
offline reselection). Weights are unmodified; the directory records this in its `provenance.json`.

Reports: `outputs/split_seed42/motif_prompt_stage2_v3_prefix/test_report.json` (checkpoint
`d2bb4fc7a55a10d3`) and `outputs/split_seed42/motif_prompt_stage2_v3_prefix_epoch4/test_report.json`
(checkpoint `4be56952089d1da7`). Logs: `outputs/logs/motif_v3_prefix_test.log`,
`outputs/logs/motif_v3_prefix_epoch4_test.log`.

## 6. Evidence limits

Single seed (training seed 0) everywhere; the spec's three-seed rule for the main arm is unmet. The
pre-registered reporting margins are ±0.01 GS and ±0.5 MMD ratio and apply to those metrics only — no
margin is claimed for AUROC or AUPRC, whose single-seed differences are reported as read. The epoch-4 row
is a labelled secondary row selected by inspection of the V_val graph fit, not by the five-rank rule, and
must never be reported as the arm's result. `s2_true` / `s1_true` in the pilot table are a labelled oracle
diagnostic (true compiled templates on V_val) and appear in no deployable row. The output-density control
of spec §8 was not trained for wave 3, so no matched-density comparison exists for these rows; the
wave-1 diagnostic (`outputs/split_seed42/motif_rd1_diagnostic.json`) covers wave-1 Stage II only.

## 7. Attribution

Date 2026-09-19, same checkpoint (`outputs/split_seed42/motif_prompt_stage2_v3_prefix/best.pt`, epoch 2,
checkpoint `d2bb4fc7a55a10d3`), same split, single seed. Two reads, run with the wave-1 recipe
(`motif_prompt_verdict/interventions_and_counterfactuals.md` §1): the spec §8 output-density control, which
asks whether the held-out MMD movement is shape or threshold placement, and the five scoring-time prefix
interventions, which ask whether the V_val gain is attributable to the row's *own* predicted graph.
Artifacts on the H20 shared checkout under `outputs/analysis/motif_wave3_attribution/`
(`density_control.json`, `interventions_summary_vval.json`, `interventions/stage2_v3/<intervention>/<pairs>.npz`);
drivers in `/2023533015/scratch_motif_wave3/`.

### 7.1 Output-density control (spec §8, held-out test)

One union of 943,685 scored `test_topology` pairs, self-pairs included and assembling as self-loops.
The target is the arm's **own** realised edge count at its V_val-selected threshold — 29,992 union pairs at
logit 2.8211 — and every row is re-thresholded to that count with `density_matched_threshold`, which admits
tie groups atomically; all three rows land within 0.3% of the target, so none is marked approximate. Matching
the union density does not equalise density inside each BFS subgraph, so RD here is a diagnostic of the
residual spread, not a matched quantity. Reported **alongside** the protocol's V_val-selected threshold, never
instead of it.

| Row | logit threshold | realised edges | GS ↑ | RD →1 | degree / clustering / spectral MMD ↓ |
|---|---|---|---|---|---|
| **Wave-3 deployable, epoch 2** (its own protocol threshold) | 2.8211 | 29,992 | 0.404 | 0.529 | 8.38 / 7.31 / 13.31 |
| `prefix_base`, density-matched | 2.3281 | 29,931 | **0.417** | 0.525 | **8.38 / 7.30 / 12.99** |
| B0, density-matched | 1.7734 | 29,939 | **0.428** | 0.530 | 10.77 / 9.05 / 16.11 |
| `prefix_base`, its own protocol threshold (published, for contrast) | — | — | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 |
| B0, its own protocol threshold (published, for contrast) | — | — | 0.429 | 0.550 | 9.9 / 8.4 / 15.0 |

The arm's row reproduces its §3 test row exactly, as it must: the control re-thresholds it to the count it
already realises.

### 7.2 Scoring-time prefix interventions (V_val)

`score_universe --prefix-intervention <mode> --prefix-intervention-seed 0` on the same checkpoint, scored
through the repo's own four-GPU fan-out and read at the arm's ONE frozen protocol threshold
(logit 2.8211100101470947) on `val_topology`. `gates_off` deletes the prompt channel; `mean` replaces the
prediction with the corpus-mean template (the marginal-preserving null); `permute_closure` and
`rewire_bridge` permute within a family of the row's own predicted graph; `shuffle_graph` gives every row
another row's predicted graph under one seeded permutation of the whole universe. Every artifact passed
`validate_artifact_precision` with the family's fp32 pair contract.

| Intervention | val_cls AUROC | val_cls AUPRC | Δ AUPRC vs none | max abs Δ logit | GS ↑ | RD →1 | degree / clustering / spectral MMD ↓ |
|---|---|---|---|---|---|---|---|
| none (published) | 0.7984 | **0.8176** | – | – | 0.395 | 1.073 | 6.63 / 2.94 / 6.68 |
| `gates_off` | 0.7925 | 0.8136 | −0.0040 | 6.89 | 0.399 | 1.040 | 10.94 / 5.08 / 9.57 |
| `mean` | 0.7819 | 0.8017 | **−0.0159** | 6.74 | 0.228 | 0.192 | 80.12 / 40.09 / 54.45 |
| `permute_closure` | 0.7976 | 0.8167 | −0.0009 | 1.19 | 0.389 | 0.969 | 7.06 / 3.42 / 7.40 |
| `rewire_bridge` | 0.7983 | 0.8175 | −0.0001 | 1.17 | 0.395 | 1.082 | 6.63 / 2.94 / 6.51 |
| `shuffle_graph` | 0.7922 | 0.8079 | **−0.0097** | 5.92 | 0.375 | 0.825 | 10.75 / 8.41 / 9.61 |
| `prefix_base` (reference, replayed at the same threshold) | 0.7925 | 0.8130 | – | – | 0.400 | 1.040 | 10.95 / 5.10 / 9.56 |

`gates_off` reproduces `prefix_base` to fp32 tolerance on both families of metrics, as in wave 1. On held-out
test the two completed intervention rows read: `gates_off` AUROC 0.7205 / AUPRC 0.7446 and `mean`
0.7132 / 0.7385, against the arm's own 0.7168 / 0.7418 and `prefix_base`'s 0.7205 / 0.7441.

### 7.3 Reading

**Edge, and the graph it comes from, together.** The +0.004 V_val `val_cls` AUPRC the arm holds over
`gates_off` is attributable to the row's own predicted graph: transplanting another row's graph costs more
than deleting the channel outright (0.8079 against `gates_off`'s 0.8136, −0.0097 from the unintervened
0.8176), the marginal-preserving `mean` null removes the gain and a further 0.012 (0.8017), and the two
within-family permutations — which move the output by at most ~1.2 logits, against wave 1's exactly-null
0.001 — cost only −0.0009 and −0.0001; the V_val topology row says the same thing with its five numbers,
the arm's GS 0.395 / RD 1.073 / MMD 6.63 / 2.94 / 6.68 falling to GS 0.375 / RD 0.825 / 10.75 / 8.41 / 9.61
under `shuffle_graph` and collapsing under `mean` (GS 0.228, RD 0.192), while `gates_off` returns the frozen
trunk's row (0.399 / 1.040 / 10.94 / 5.08 / 9.57 against `prefix_base`'s 0.400 / 1.040 / 10.95 / 5.10 / 9.56).
**The held-out topology movement is threshold placement, not shape.** At the arm's own realised density —
29,992 of 943,685 union pairs — `prefix_base` reaches degree / clustering / spectral MMD 8.38 / 7.30 / 12.99
against the arm's 8.38 / 7.31 / 13.31 at an all-but-identical RD (0.525 vs 0.529) and a *better* GS
(0.417 vs 0.404), and B0 at the same density is 10.77 / 9.05 / 16.11 at GS 0.428, so the 8.4 / 7.3 / 13.3
against 11.4 / 9.7 / 16.8 of §3 is the arm's looser operating point rather than a better assembled shape.
**Together these close the wave-3 evidence gap in opposite directions**: the generator's predicted graph is
now genuinely row-conditioned and its condition is read by the interface — the mechanism wave 1 lacked — but
that row conditioning buys +0.004 AUPRC on V_val, nothing on held-out test (the channel is net-harmful there,
`gates_off` 0.7446 against the arm's 0.7418), and no shape gain at matched density, so the §4 verdict is
unchanged and now rests on a density-matched control rather than on its absence.

### 7.4 Limits of this read

Single seed, single permutation, one `--prefix-intervention-seed 0`; the pre-registered margins are ±0.01 GS
and ±0.5 MMD ratio and no margin is claimed for AUROC or AUPRC, so the −0.0097 `shuffle_graph` AUPRC cost and
the +0.004 gain it attributes are single-seed differences reported as read. The V_val comparison between the
arm and `gates_off` is **not** density-matched: their RDs differ (1.073 vs 1.040), and §7.1 shows how much a
density difference alone can move MMD ratios on this family, so the V_val MMD contrast should not be read as a
pure shape statement either. `mean` shifts every logit down by ~1.9 on test and ~1.5 more than `gates_off` on
V_val, so its topology row is far off the frozen operating point and its RD 0.192 is a placement artefact, not
a shape verdict. The test-universe interventions were run for completeness beyond the V_val question:
`gates_off` and `mean` had completed when this was written and are quoted above; `permute_closure`,
`rewire_bridge` and `shuffle_graph` on `test` / `test_topology` were still scoring, and land under
`outputs/analysis/motif_wave3_attribution/interventions/stage2_v3/`. Finally, the shipped entry point
`python -m src.experiments.motif_density_control` **cannot run against a current report**: it reads
`graph.fixed_threshold.validation_selection.logit_threshold`, a flat key the `test_protocol_v8`
`geometric_rd_five_rank_v1` schema does not write (the threshold is at `…validation_selection.selected.logit_threshold`),
and raises `KeyError`. The control here was therefore run through that module's own
`target_edge_count` / `density_matched_report` functions unchanged, with the threshold read from the key the
schema does write; the module was not edited.
