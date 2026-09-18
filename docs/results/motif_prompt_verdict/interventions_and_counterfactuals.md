# Motif-graph GRIT prompt — causal-intervention and counterfactual verdict

Date 2026-09-18. Branch `codex/motif-graph-grit-prompt`. Split seed 42, root `node_007630`.
Everything was scored on H20 container 30838 from the shared checkout
`/2023533015/topology-conditioned-inductive-edge-prediction`. Every artifact this analysis
produced lives under `outputs/analysis/motif_verdict_20260918/` in that shared outputs tree;
no published arm directory was written into, no file in the H20 working tree was edited, and
no `git` command was run there.

## 0. Verdict

**Generator failure. Not reader drift, and not an inert interface.**

1. The Stage II *prompt pathway* is causally active and **net-harmful**: turning the gates off
   (`gates_off`) *raises* V_val `val_cls` AUPRC from 0.8039 to 0.8136 and moves the median row's
   logit by order 1 (mean Δ −0.541, max |Δ| 6.41, r = 0.980). Replacing the prediction with the
   constant training-mean template (`mean`) raises it further, to 0.8162.
2. The *row-specific content* of the predicted graph is causally inert. `shuffle_graph`
   (every row reads another row's predicted graph, one seeded whole-universe permutation)
   leaves V_val `val_cls` AUPRC at 0.8039 — unchanged to four decimals — and V_val GS at
   0.396 vs 0.394. `permute_closure` and `rewire_bridge` are *exactly* null: max |Δ logit| =
   0.001 and 0.001, Pearson r = 1.00000, and the recorded degree marginals change by 0.0000
   because the generator emits a slot-symmetric graph there is nothing to permute in.
   On the Stage I checkpoint the same three interventions cost −0.100, −0.023 and −0.002 AUPRC.
3. The trained interface **can still deliver Stage I's gain**. In the 2×2 reader × graph
   counterfactual on `val_cls`, the Stage II reader+interface reading the *true* compiled
   template scores AUROC 0.8707 / AUPRC 0.8867 — statistically indistinguishable from the
   Stage I bundle on the same graph (0.8710 / 0.8872, Δ AUPRC = 0.0005). Swapping the reader
   is worth ~0.0005 AUPRC; swapping the graph is worth ~0.086.
4. The dose–response is steep and monotone: mixing only 25 % of the true template into the
   prediction recovers 71 % of the AUPRC gap (0.8039 → 0.8624 of a 0.8039→0.8867 range).
   The interface is not saturated, not gated shut and not mis-scaled — it is starved of signal.

The generator has collapsed to a near-constant graph carrying the wrong mass by two orders of
magnitude: predicted `wedge_mass` 0.0017 against a true 0.1488 on nonself `val_cls` rows, and a
row-to-row dispersion 5.6× smaller than the truth's (relative per-row deviation from the column
mean 0.24 vs 1.36). Its row-wise centred cosine against the true template is 0.030.

---

## 1. What was run

All scoring used the repo's own path, `python -m src.score_fanout` → `src.score_universe score`
(`--device cuda --amp bf16`, `--pack-dir outputs/feature_packs/b0_v31_bf16`), 4-way GPU fan-out
with the strict merge, exactly as the published `none` artifacts were produced. The existing
`scores/{val_cls,val_topology,test,test_topology}.npz` of `motif_prompt_stage1`,
`motif_prompt_stage2` and `prefix_base` were reused as the `intervention = none` baselines and
never rescored.

```bash
# analysis 1 (stage 2, deployable) and 2 (stage 1, ceiling reference), per intervention iv in
# {gates_off, mean, permute_closure, rewire_bridge, shuffle_graph} and pairs in
# {val_cls, val_topology, test, test_topology}:
.venv/bin/python -m src.score_fanout \
  --checkpoint outputs/split_seed42/motif_prompt_stage2/best.pt \
  --pairs <pairs> --data-root data --strategy breadth_first \
  --pack-dir outputs/feature_packs/b0_v31_bf16 \
  --output outputs/analysis/motif_verdict_20260918/interventions/stage2/<iv>/<pairs>.npz \
  --prefix-intervention <iv> --prefix-intervention-seed 0
# stage 1 adds --allow-oracle-diagnostic and writes under .../interventions/stage1/<iv>/
```

Driver: `/2023533015/scratch_motif_verdict/interventions/run_interventions.sh {A|B|C|all}`.

Analyses 3–4 (the crossed cells and the λ dose–response) ran through
`/2023533015/scratch_motif_verdict/interventions/crossed_cells.py`, which loads both
checkpoints, compiles the true templates through `MotifTemplateTable` on the same
`_oracle_truth_graph_for_scoring` V_val truth graph the Stage I scorer uses, and calls the
repo's own `src.score_universe._score_v3_1_packed` with an explicit `row_templates` bank — so
the trunk drive, the AB/BA max-symmetrisation, the length-bucketed batching and the fp32 pair
pass are byte-for-byte the formal scoring path. The Stage II generator's own predictions were
captured from that same path by running it with an identity `shuffle_sources` map, which makes
its transplant bank each row's own predicted graph.

```bash
.venv/bin/python /2023533015/scratch_motif_verdict/interventions/crossed_cells.py \
  --pairs val_cls --pack-dir outputs/feature_packs/b0_v31_bf16 \
  --stage1-checkpoint outputs/split_seed42/motif_prompt_stage1/best.pt \
  --stage2-checkpoint outputs/split_seed42/motif_prompt_stage2/best.pt \
  --output-dir outputs/analysis/motif_verdict_20260918/interventions/crossed \
  --lambdas 0,0.25,0.5,0.75,1.0
# val_topology: 4-way sharded, cells s1_pred,s2_true (s1_true and s2_pred are the published
# artifacts), via run_crossed_topology.sh
```

Readers: `analyze_interventions.py` (deltas, AUROC/AUPRC, the topology five replayed at the
checkpoint-frozen threshold **1.1287425756454468**, degree marginals from the artifact meta) and
`analyze_crossed.py` (cells, dose–response, mediation strata). Outputs:
`interventions_summary_vval.json`, `interventions_summary.json`, `crossed_val_cls.json`,
`crossed_val_topology.json`.

## 2. Precision checks (analysis 5)

* `src.score_universe.validate_artifact_precision` was called on **every** artifact this
  analysis loaded, including every intervention artifact. All passed. Each motif artifact
  carries `score_precision = {contract: v3_1_motif_prompt_pair_fp32_v1, encode_autocast: bf16,
  pair_autocast: False, pair_compute_dtype: float32, logit_storage_dtype: float32}`, i.e. the
  count head and the RRWP stack ran in fp32 as spec §5.2 requires, and no logit column lies on
  the bf16 grid.
* The crossed-cell logits carry the same fp32 pair pass (autocast is disabled inside
  `_score_v3_1_packed` for the motif family) and their `score_resolution` diagnostics were
  recorded next to each `.npz`.
* **`gates_off` reproduces `prefix_base` to floating-point tolerance, not bit-for-bit.**
  On `val_cls`: mean |Δ| = 0.0110, max |Δ| = 0.1577, Pearson r = 0.9999926, AUROC 0.7925 vs
  0.7925, AUPRC 0.8136 vs 0.8130. The residual is explained and expected: `prefix_base`'s own
  published artifact ran its pair pass under **bf16 autocast** (`pair_autocast: "bf16"`), while
  the motif family's pair pass is fp32 by construction. `gates_off` is therefore `prefix_base`
  recomputed in fp32. On `val_topology` the two agree on the five topology numbers to
  3 decimals except RD (GS 0.399 / 0.399, RD 1.844 / 1.846, MMD 11.43 / 11.39, 5.28 / 5.32,
  6.68 / 6.63). The Stage I and Stage II `gates_off` artifacts agree with each other to the same
  tolerance, as they must — both reduce to the same frozen trunk.
* My crossed-cell path reproduces the published artifacts it should: `s2_pred` vs published
  `motif_prompt_stage2/scores/val_cls.npz` max |Δ| = **8.1e-6**; `s1_true` vs published
  `motif_prompt_stage1/scores/val_cls.npz` max |Δ| = **8.2e-6** (fp32 reduction-order
  nondeterminism from a different batch partition, not a different computation).

---

## 3. Analysis 1 — scoring-time interventions on the deployable Stage II checkpoint

### 3.1 `val_cls` (10,694 rows, balanced) and `test` (64,038 rows)

| pairs | intervention | AUROC | AUPRC | mean Δ | std Δ | max abs Δ | r(none, iv) | frac Δ>0 on positives | mean Δ pos | mean Δ neg |
|---|---|---|---|---|---|---|---|---|---|---|
| val_cls | none (published) | 0.7847 | 0.8039 | – | – | – | – | – | – | – |
| val_cls | gates_off | 0.7925 | 0.8136 | −0.541 | 1.121 | 6.405 | 0.97991 | 0.263 | −1.020 | −0.062 |
| val_cls | mean | 0.7968 | 0.8162 | −1.298 | 1.086 | 6.872 | 0.96642 | 0.041 | −1.684 | −0.912 |
| val_cls | permute_closure | 0.7847 | 0.8039 | 0.000 | 0.000 | **0.001** | **1.00000** | 0.399 | 0.000 | −0.000 |
| val_cls | rewire_bridge | 0.7847 | 0.8039 | −0.000 | 0.000 | **0.001** | **1.00000** | 0.412 | −0.000 | −0.000 |
| val_cls | shuffle_graph | 0.7841 | **0.8039** | −0.045 | 0.541 | 4.501 | 0.98644 | 0.457 | −0.070 | −0.019 |

Δ is `logit_none − logit_intervened`.

### 3.2 Topology at the one frozen protocol threshold (logit 1.1287425756454468)

| pairs | intervention | GS ↑ | RD →1 | deg MMD ↓ | clus MMD ↓ | spec MMD ↓ | mean Δ | max abs Δ | r |
|---|---|---|---|---|---|---|---|---|---|
| val_topology | none (published) | 0.394 | 1.148 | 12.79 | 5.49 | 9.71 | – | – | – |
| val_topology | gates_off | 0.399 | 1.844 | 11.43 | 5.28 | 6.68 | −0.042 | 6.405 | 0.96590 |
| val_topology | mean | 0.385 | 2.397 | 17.70 | 8.65 | 7.66 | −0.963 | 6.872 | 0.94982 |
| val_topology | permute_closure | 0.394 | 1.148 | 12.79 | 5.49 | 9.71 | 0.000 | 0.001 | 1.00000 |
| val_topology | rewire_bridge | 0.394 | 1.148 | 12.79 | 5.49 | 9.71 | −0.000 | 0.004 | 1.00000 |
| val_topology | shuffle_graph | 0.396 | 1.225 | 11.29 | 5.26 | 8.55 | −0.024 | 5.375 | 0.97912 |

`prefix_base`, same threshold, `val_topology`: GS 0.399 / RD 1.846 / 11.39 / 5.32 / 6.63.

### 3.3 Degree marginals recorded by the three graph substitutions (spec §8)

Sorted-within-role weighted slot degrees, per-slot means over every scored row.

| intervention | pairs | role | slots | mean own | mean read | mean abs change | max abs change |
|---|---|---|---|---|---|---|---|
| permute_closure | val_cls | u / v | 1 / 1 | 2.0145 / 1.9594 | 2.0145 / 1.9594 | 0.0000 | 0.0000 |
| permute_closure | val_cls | closure | 8 | 0.0311 | 0.0311 | **0.0000** | **0.0000** |
| permute_closure | val_cls | left / right | 8 / 8 | 0.6797 / 0.6720 | same | 0.0000 | 0.0000 |
| rewire_bridge | val_cls | closure | 8 | 0.0311 | 0.0311 | 0.0000 | 0.0000 |
| rewire_bridge | val_cls | left / right | 8 / 8 | 0.6797 / 0.6720 | same | **0.0001** | 0.0019 / 0.0021 |
| shuffle_graph | val_cls | u / v | 1 / 1 | 2.0145 / 1.9594 | 2.0145 / 1.9594 | **0.4969 / 0.4668** | 1.7319 / 1.6486 |
| shuffle_graph | val_cls | closure | 8 | 0.0311 | 0.0311 | 0.0099 | 0.0490 |
| shuffle_graph | val_cls | left / right | 8 / 8 | 0.6797 / 0.6720 | same | 0.1464 / 0.1506 | 0.5924 / 0.6349 |
| shuffle_graph | val_topology | u / v | 1 / 1 | 2.0314 / 1.9817 | same | 0.5552 / 0.5356 | 1.7371 / 1.7968 |

Reading: `permute_closure` and `rewire_bridge` move **nothing** on the predicted graph — the
generator's eight closure slots and its 8×8 interior block are effectively slot-symmetric, so a
within-role permutation is the identity map on its output. `shuffle_graph` genuinely swaps
graphs (an endpoint's weighted degree changes by 0.50 on a mean of 2.01, i.e. 25 %), and the
decision does not notice: AUPRC 0.8039 → 0.8039, GS 0.394 → 0.396.

**Answer to "is the predicted graph causally inert, or does it move logits without helping?"**
Both, in different senses, and the distinction is the whole result. The *graph channel as a
whole* is far from inert — deleting it (`gates_off`) moves logits by mean −0.54 / max 6.4 and
*improves* AUPRC by +0.010. The *row-conditioning* inside that channel is inert: the three
interventions that preserve the marginal distribution of predicted graphs while destroying the
row↔graph pairing produce a change of exactly zero (two of them) or zero to four decimals
(the transplant).

---

## 4. Analysis 2 — the same interventions on Stage I (reference scale)

Stage I reads the true compiled template; these runs carry `--allow-oracle-diagnostic` and are
ceiling diagnostics, never formal results.

| pairs | intervention | AUROC | AUPRC | Δ AUPRC vs its own none | mean Δ | max abs Δ | r |
|---|---|---|---|---|---|---|---|
| val_cls | none (published diagnostic) | 0.8710 | 0.8872 | – | – | – | – |
| val_cls | gates_off | 0.7925 | 0.8136 | **−0.0736** | 0.178 | 6.050 | 0.94995 |
| val_cls | mean | 0.7819 | 0.8017 | **−0.0855** | 0.779 | 6.654 | 0.94556 |
| val_cls | permute_closure | 0.8437 | 0.8640 | **−0.0232** | 0.222 | 6.656 | 0.98288 |
| val_cls | rewire_bridge | 0.8689 | 0.8852 | −0.0020 | 0.016 | 1.468 | 0.99947 |
| val_cls | shuffle_graph | 0.7742 | 0.7873 | **−0.0999** | 0.196 | 7.766 | 0.89774 |

| pairs | intervention | GS | RD | deg | clus | spec |
|---|---|---|---|---|---|---|
| val_topology | none (published diagnostic) | 0.538 | 1.183 | 8.91 | 3.31 | 7.82 |
| val_topology | gates_off | 0.399 | 1.844 | 11.43 | 5.28 | 6.68 |
| val_topology | mean | 0.387 | 0.957 | 16.78 | 8.17 | 12.91 |
| val_topology | permute_closure | 0.509 | 1.116 | 10.77 | 3.89 | 8.90 |
| val_topology | rewire_bridge | 0.538 | 1.175 | 9.65 | 3.74 | 8.25 |
| val_topology | shuffle_graph | 0.389 | 1.032 | 12.32 | 6.68 | 9.89 |

This is the effect size the reader + interface **can** deliver when the graph carries
information: 0.023–0.100 AUPRC and 0.03–0.15 GS. Note that on Stage I, `shuffle_graph` is worse
than `gates_off` (0.7873 vs 0.8136; GS 0.389 vs 0.399) — another row's *true* graph is actively
misleading, which is the correct behaviour of a reader that is using the graph.

Against that scale, every Stage II graph-content intervention is 0.000.
`rewire_bridge` is small on Stage I too (−0.002 AUPRC, GS 0.538 → 0.538): the interior
correspondence carries little on its own, so the Stage II null there is uninformative; the
informative Stage II nulls are `permute_closure` (Stage I −0.023) and above all
`shuffle_graph` (Stage I −0.100).

---

## 5. Analysis 3 — the reader × graph 2×2 counterfactual

Both cells scored through the identical trunk, identical batching and the identical AB/BA
symmetrisation; only the reader/interface weights and the 96-edge bank differ.

### 5.1 `val_cls` (edge)

| reader \ graph | TRUE compiled template | Stage II PREDICTED graph |
|---|---|---|
| **Stage I bundle** (`motif_prompt_stage1`) | AUROC 0.8710 / AUPRC **0.8872** *(= published diagnostic)* | AUROC 0.7810 / AUPRC **0.8010** |
| **Stage II student** (`motif_prompt_stage2`) | AUROC 0.8707 / AUPRC **0.8867** | AUROC 0.7847 / AUPRC **0.8039** *(= published formal)* |

Row effect (reader swap, holding the graph): **+0.0005** AUPRC on truth, **+0.0029** on the
prediction. Column effect (graph swap, holding the reader): **−0.0862** for Stage I,
**−0.0828** for Stage II.

Both predicted-graph cells sit *below* `gates_off` (0.8136) and below `mean` (0.8162).

### 5.2 `val_topology` at the frozen threshold (297,209 rows)

| reader \ graph | TRUE | PREDICTED |
|---|---|---|
| **Stage I** | AUPRC 0.377* / GS 0.538 / RD 1.183 / 8.91, 3.31, 7.82 *(published)* | AUPRC 0.204 / GS 0.376 / RD 0.781 / 20.06, 10.69, 16.01 |
| **Stage II** | AUPRC 0.377 / GS **0.527** / RD 1.312 / 8.62, 3.15, 7.19 | AUPRC 0.204 / GS 0.394 / RD 1.148 / 12.79, 5.49, 9.71 *(published)* |

\* the Stage I `val_topology` AUPRC is not printed in the published report; the measured
crossed-cell value for `s2_true` is 0.3774 and `s1_pred` 0.2043.

Caveat, stated plainly: the threshold 1.1287 was selected on Stage II's own `none` V_val logits,
so the crossed cells are read at a *borrowed* operating point and their RD drifts (1.31 for
`s2_true`). GS and the MMD ratios are still the comparison of interest and they say the same
thing as the edge table: give the Stage II interface the true graph and its assembled V_val
graph goes from GS 0.394 to **0.527** with every MMD ratio roughly halved (12.79→8.62,
5.49→3.15, 9.71→7.19) — essentially the Stage I row. Give the Stage I reader the predicted
graph and it falls to GS 0.376, *below* the frozen trunk.

**Conclusion of analysis 3: there is no reader drift.** The interface Stage II trained (final
reader block, count head, adapter and gates, opened at epoch 3 at 0.1× LR) still converts a
true motif template into Stage I's full gain, on the edge metric and on the assembled graph.

---

## 6. Analysis 4 — dose–response and row-level mediation (`val_cls`)

Stage II model, graph = `(1−λ)·predicted + λ·true`.

| λ | AUROC | AUPRC | mean logit | std logit | fraction of the 0→1 AUPRC gap recovered |
|---|---|---|---|---|---|
| 0.00 | 0.7847 | 0.8039 | −2.402 | 3.244 | 0 % |
| 0.25 | 0.8461 | 0.8624 | −1.664 | 3.778 | **71 %** |
| 0.50 | 0.8602 | 0.8754 | −1.610 | 3.833 | 86 % |
| 0.75 | 0.8662 | 0.8813 | −1.595 | 3.866 | 93 % |
| 1.00 | 0.8707 | 0.8867 | −1.641 | 3.895 | 100 % |

Monotone, steep at the origin, saturating. The interface responds strongly to small
improvements in graph quality — the opposite of a dead or mis-scaled prompt.

### 6.1 Which rows the true graph helps, against `mean` as the common reference

`Δ_pred = logit_pred − logit_mean`, `Δ_true = logit_true − logit_mean`
(`logit_mean` from the Stage II `mean` intervention).

| stratum | n | mean Δ_pred | mean Δ_true | mean abs Δ_pred | mean abs Δ_true | sign agreement | r(Δ_pred, Δ_true) |
|---|---|---|---|---|---|---|---|
| all | 10,694 | −1.298 | −0.537 | 1.328 | 0.963 | 0.676 | 0.295 |
| positives | 5,347 | **−1.684** | **−0.051** | 1.703 | 0.847 | 0.422 | 0.176 |
| negatives | 5,347 | −0.912 | −1.023 | 0.953 | 1.078 | 0.929 | 0.884 |
| self (u = v) | 644 | −1.881 | +0.027 | 1.881 | 0.040 | 0.264 | −0.059 |
| nonself | 10,050 | −1.261 | −0.573 | 1.292 | 1.022 | 0.702 | 0.323 |
| nonself, true wedge_mass > 0 | 5,101 | −1.615 | −0.153 | 1.640 | 1.012 | **0.468** | 0.219 |
| nonself, true wedge_mass = 0 | 4,949 | −0.895 | −1.007 | 0.934 | 1.033 | **0.944** | 0.908 |
| nonself, true bridge_mass > 0 | 7,705 | −1.393 | −0.465 | 1.422 | 1.040 | 0.632 | 0.273 |
| nonself, true bridge_mass = 0 | 2,345 | −0.827 | −0.930 | 0.866 | 0.962 | 0.934 | 0.884 |

Reading. The true graph earns its gain by *sparing* the rows that have closure evidence
(positives: mean Δ_true −0.05; wedge-bearing nonself rows: −0.15) while suppressing the rows
that have none (wedge_mass = 0: −1.01). The predicted graph suppresses **everything**, and
suppresses positives hardest (−1.68 vs −0.91 on negatives) — it inverts the useful contrast.
Prediction and truth agree on a row's fate (sign agreement 0.94, r = 0.91) exactly where the
true graph is *empty*, i.e. where there is nothing to predict, and agree at chance (0.47) where
the true graph carries a wedge. Self-rows, whose template is empty by construction, are pushed
down 1.88 logits by the prediction and 0.03 by the truth: the generator hallucinates structure
on rows that have none.

### 6.2 What the predicted graph actually is (nonself `val_cls` rows)

| quantity | predicted | true | corpus mean template `Abar` |
|---|---|---|---|
| wedge_mass (mean) | **0.00174** | 0.14875 | 0.00275 |
| bridge_mass (mean) | 0.17749 | 0.89382 | 0.23002 |
| deg_u (mean) | 2.016 | 2.444 | 1.741 |
| relative per-row deviation from the column mean | **0.242** | **1.357** | 0 by definition |
| relative distance of each row to `Abar` | 0.379 | – | – |
| row-wise centred cos(pred, true) | **0.030** mean / 0.044 median | – | – |
| Pearson(pred, true) over wedge_mass | **0.277** | | |
| Pearson(pred, true) over bridge_mass | 0.096 | | |
| Pearson(pred, true) over deg_u | **−0.194** | | |

The generator sits close to its own initialisation: its per-type output biases were set to
`logit(mean weight of that edge type)` by `init_biases`, and after 15 epochs its rows still
deviate from the corpus mean by only 0.38 in relative norm, with 5.6× less row-to-row variation
than the truth. It carries a faint closure signal (wedge-mass Pearson 0.28) at ~1 % of the right
magnitude, no bridge signal (0.10), and an *anti*-correlated endpoint degree (−0.19). Training
telemetry agrees: `train_motif_slot_loss` is 0.0726 at epoch 1 and 0.0700 at epoch 15 — flat —
while `train_motif_topo_loss` falls 0.699 → 0.342, i.e. the generator learned to match the
teacher's *token representation* without ever matching the graph, which is the failure mode
spec §7.4 warned about ("minimising token reconstruction error is optimised by constant or
low-variance tokens").

---

## 7. Analysis 1 continued — held-out test side (partial)

Scored through the identical path; read at the same frozen thresholds. The classification
threshold is the protocol's own max-F1 value from `val_cls`; the AUROC/AUPRC below use raw
logits and need no threshold.

### 7.1 `test` (64,038 rows) — complete

| intervention | AUROC | AUPRC | mean Δ | std Δ | max abs Δ | r(none, iv) | frac Δ>0 pos | mean Δ pos | mean Δ neg |
|---|---|---|---|---|---|---|---|---|---|
| none (published) | 0.7219 | 0.7452 | – | – | – | – | – | – | – |
| gates_off | 0.7205 | 0.7446 | −0.261 | 0.961 | 5.388 | 0.97277 | 0.398 | −0.467 | −0.054 |
| mean | 0.7221 | 0.7454 | −1.125 | 1.043 | 7.334 | 0.96053 | 0.085 | −1.313 | −0.938 |
| permute_closure | 0.7219 | 0.7452 | −0.000 | 0.000 | **0.002** | **1.00000** | 0.393 | +0.000 | −0.000 |
| rewire_bridge | 0.7219 | 0.7452 | −0.000 | 0.000 | **0.006** | **1.00000** | 0.414 | −0.000 | −0.000 |
| shuffle_graph | 0.7131 | 0.7375 | +0.017 | 0.739 | 6.093 | 0.97061 | 0.550 | +0.074 | −0.040 |

`prefix_base` on `test`: AUROC 0.7205 / AUPRC 0.7441 — again `gates_off` (0.7205 / 0.7446)
reproduces it to fp tolerance.

Degree marginals on `test`: `permute_closure` mean |Δ| = 0.0000 in every role;
`rewire_bridge` 0.0001 on the bridge-intermediate roles and 0.0000 elsewhere; `shuffle_graph`
0.4428 on the endpoint degree (mean own 1.9927, i.e. 22 %), 0.0138 closure, 0.1661 left.

Same reading as V_val. The two slot-permutation interventions are exactly null because the
predicted graph is slot-symmetric. `shuffle_graph` is the one test-side row where the
transplant costs something (AUPRC 0.7452 → 0.7375, −0.008) — still an order of magnitude below
Stage I's V_val transplant cost (−0.100) and in the same range as `gates_off` itself, so it
does not overturn the V_val reading; it is one seed of one permutation and inside the ±0.01
practical margin the spec sets for single-seed differences.

### 7.2 `test_topology` (943,685 rows) — NOT COMPUTED

The five `test_topology` intervention artifacts were scored (four complete, `shuffle_graph`
still in flight at hand-off), but the fixed-threshold topology replay was not run on them: the
coordinator stopped the work before that CPU pass. To finish:

```bash
cd /2023533015/topology-conditioned-inductive-edge-prediction
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$PWD .venv/bin/python \
  /2023533015/scratch_motif_verdict/interventions/analyze_interventions.py \
  --root outputs/analysis/motif_verdict_20260918/interventions \
  --output outputs/analysis/motif_verdict_20260918/interventions_summary.json
```
That one command also refreshes every V_val row in this report; it is pure CPU and reads only
already-written artifacts.

---

## 8. What this does and does not license

* It is a mechanism result on one seed of one checkpoint. Spec §8 pre-registers three seeds and
  ±0.01 GS / ±0.5 MMD reporting margins; several differences here (the 0.0005 reader-swap gap,
  the 0.396 vs 0.394 `shuffle_graph` GS) are inside those margins and are read as "no change",
  which is the direction the argument needs.
* The architecture reading is made on **V_val and the crossed cells only**. The test-side
  interventions are reported for completeness at the one frozen threshold and were not used to
  choose or judge any design.
* "The interface can deliver Stage I's gain" is established for *this* interface against *the
  true compiled template*. It does not establish that any endpoint-only generator can reach a
  graph good enough to matter; §6 shows only that the λ ≥ 0.25 neighbourhood of the truth is
  enough, which is a statement about proximity to truth, not about attainability from
  `(x_u, x_v)`.
* `rewire_bridge` is a weak instrument on this benchmark (−0.002 AUPRC even on the true graph),
  so its Stage II null carries little weight. `permute_closure` and `shuffle_graph` carry the
  argument.
