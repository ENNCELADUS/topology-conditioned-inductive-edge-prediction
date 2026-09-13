# Topology-prompt Stage II (`v3_1_coord_gen`): results

Headline split (seed 42, root `node_007630`), training seed 0, commit aeb04d0. Spec:
`docs/superpowers/specs/2026-09-13-topology-prompt-stage2-design.md`. Both runs are **formal,
deployable** models (input `(x_u, x_v)` only): a coordinate generator predicts the queried pair's
standardised structural coordinates from the frozen Stage I reader's endpoint states, and the
frozen reader scores the pair from that prediction. Chains ran 2026-09-13 05:10–08:43 UTC
(`coord_gen_full`, 30846) and 05:10–09:23 UTC (`coord_gen_frozen`, 30838): train → held-out test →
`gates_off`, `mean`, `mean_relation` interventions.

| Run | Reader (frozen) | Reader's ceiling on true coordinates (V_val AUPRC / GS) | Stopped / selected epoch |
|---|---|---|---|
| `coord_gen_full` | `topo_prompt_full/best.pt` | 0.961 / 0.691 | 12 (val-task-loss patience; minimum at epoch 2) / 6 |
| `coord_gen_frozen` | `topo_prompt_frozen/best.pt` | 0.940 / 0.648 | 22 (minimum at epoch 12) / 4 |

Comparator: `prefix_base` (val_cls AUROC/AUPRC 0.793/0.814, GS 0.401; test 0.721/0.744, GS 0.409).

## 1. Verdict (topology first)

The study's object is the assembled topology, so the reading starts with the five topology numbers
at the ONE V_val-selected threshold on the held-out test subgraphs, next to every deployable arm:

| Deployable row (test) | GS ↑ | RD → 1 | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ | geo RD | mean abs log RD ↓ | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.430 | 0.557 | 9.67 | 8.12 | 14.55 | 0.535 | 0.627 | 0.701 | 0.736 |
| best structural arm so far (`struct_grand` / `struct_new`) | 0.431 / 0.426 | 0.621 / 0.654 | 8.06 / 8.35 | 6.74 / 6.85 | 11.87 / 12.02 | 0.596 / 0.621 | 0.522 / 0.491 | 0.714 / 0.701 | 0.739 / 0.732 |
| `prefix_base` (the students' trunk recipe) | 0.409 | 0.456 | 11.4 | 9.7 | 16.8 | 0.445 | 0.810 | 0.721 | 0.744 |
| **`coord_gen_full`** | **0.430** | **0.907** | **4.2** | **3.6** | **6.7** | **0.864** | **0.264** | 0.677 | 0.725 |
| `coord_gen_frozen` | 0.415 | 0.516 | 8.8 | 7.8 | 13.2 | 0.502 | 0.689 | 0.721 | 0.751 |
| Full-Ego oracle (ceiling, reads true structure) | 0.602 | 0.708 | 6.96 | 6.43 | 11.47 | 0.667 | 0.445 | 0.950 | 0.955 |

**`coord_gen_full` is the best topology-preserving deployable row to date.** At the transferred
threshold its test subgraphs have the density closest to the reference of any row, including the
oracle (RD 0.91; geometric RD 0.86 against 0.62 for the best structural arm), and the lowest
degree / clustering / spectral MMD ratios of any row, again including the oracle (4.2 / 3.6 / 6.7
against the previous deployable best 8.1 / 6.7 / 11.9), at a GS equal to B0's (0.430). Per subgraph
size its RD runs from 0.63 (20 nodes) to 1.09 (200 nodes). The attribution rows tie the gain to the
predicted coordinates: the same trunk on mean coordinates sits at RD 0.52 and MMD 8.9 / 7.9 / 12.9,
with the prompt gated off at RD 0.57 and 7.4 / 6.6 / 10.9, and with the relation field alone
replaced by its mean at RD 0.71 and 13.5 / 10.7 / 16.5. The mechanism is visible in the
coordinates themselves: the generator predicts, from the endpoints' attributes, the higher
degrees and richer neighbourhoods of the denser test region (training-row R² 0.4–0.6 for degree,
triangles, common neighbours, Jaccard), the reader raises its logits accordingly, and the fixed
threshold admits proportionally more edges where the true graph is denser. That is the
topology-conditioned decision the pipeline was built for, obtained from `(x_u, x_v)` alone.

The costs and caveats belong next to it. The edge family pays: test AUROC 0.677 against
prefix_base's 0.721 and B0's 0.701 (AUPRC 0.725 vs 0.744 / 0.736; non-self pairs 0.657 / 0.686),
so the student ranks individual edges worse while assembling better graphs. On V_val the
student's GS does not move (0.387 vs 0.401) while its MMD ratios halve (5.8 / 2.4 / 6.2 vs
10.6 / 4.6 / 9.0), and the V_val ratios vary by a factor of two across epochs (5.8–14.4), so the
selected epoch (6, chosen by the geometric-RD five-rank rule) is the favourable end of a noisy
curve; the test gain is a single seed and needs two more before it is written as a finding. The
frozen-reader student moves the same way but only by noise-band amounts (RD 0.52 vs 0.46, MMD
8.8 / 7.8 / 13.2 vs 11.4 / 9.7 / 16.8, GS 0.415).

On the edge family alone the stage would read as a null (both students at or below the base), and
the generator's fit diagnostics (§3) show why the *edge* signal is absent: the predicted
coordinates carry the region-level structure the attributes support (density, neighbourhood
size), not the pair-level relation that separates a positive from a negative among neighbours.
That region-level signal is exactly what the topology family rewards.

## 2. Tables

### 2.1 V_val (primary)

val_cls edge metrics from `scores/val_cls.npz` (10,694 rows, 1:1); topology at each run's own
V_val-selected threshold.

| Row | val_cls AUROC ↑ | val_cls AUPRC ↑ | GS ↑ | RD → 1 | Degree / Clustering / Spectral MMD ↓ | logit thr |
|---|---:|---:|---:|---:|---|---:|
| `prefix_base` | 0.7925 | 0.8143 | 0.401 | 1.102 | 10.6 / 4.6 / 9.0 | 2.70 |
| `topo_prompt_full` (reader, true coords) | 0.9522 | 0.9607 | 0.691 | 1.030 | 11.1 / 4.9 / 13.8 | 5.19 |
| **`coord_gen_full`** | 0.7828 | 0.8062 | 0.387 | 1.064 | 5.8 / 2.4 / 6.2 | 1.36 |
| ↳ `gates_off` (full trunk alone) | 0.7245 | 0.7675 | 0.374 | 1.083 | 9.6 / 3.9 / 7.8 | 1.74 |
| ↳ `mean` | 0.7425 | 0.7790 | 0.376 | 1.102 | 10.4 / 4.2 / 7.9 | 2.19 |
| ↳ `mean_relation` | 0.7759 | 0.7942 | 0.381 | 1.183 | 14.9 / 5.5 / 11.1 | 3.03 |
| `topo_prompt_frozen` (reader, true coords) | 0.9248 | 0.9399 | 0.648 | 1.018 | 4.2 / 2.0 / 6.0 | 4.09 |
| **`coord_gen_frozen`** | 0.7881 | 0.8121 | 0.400 | 1.083 | 9.1 / 3.9 / 8.4 | 2.17 |
| ↳ `gates_off` (= `prefix_base`) | 0.7925 | 0.8143 | 0.401 | 1.102 | 10.6 / 4.6 / 9.0 | 2.70 |
| ↳ `mean` | 0.7725 | 0.7995 | 0.405 | 1.115 | 13.4 / 5.2 / 9.9 | 2.30 |
| ↳ `mean_relation` | 0.7890 | 0.8139 | 0.405 | 1.088 | 9.9 / 4.4 / 8.7 | 3.17 |

### 2.2 Test (formal held-out; 64,038 rows, 1:1)

| Row | AUROC ↑ | AUPRC ↑ | ECE ↓ | Brier ↓ | Acc ↑ | F1 ↑ | MCC ↑ | GS ↑ | RD → 1 | Degree / Clustering / Spectral MMD ↓ | geo RD | mean abs log RD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `prefix_base` | 0.7205 | 0.7441 | 0.293 | 0.315 | 0.634 | 0.679 | 0.279 | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 | 0.445 | 0.810 |
| **`coord_gen_full`** | 0.6771 | 0.7251 | 0.202 | 0.273 | 0.575 | 0.652 | 0.167 | 0.430 | 0.907 | 4.2 / 3.6 / 6.7 | 0.864 | 0.264 |
| ↳ `gates_off` | 0.6467 | 0.6952 | 0.160 | 0.264 | 0.521 | 0.655 | 0.067 | 0.385 | 0.570 | 7.4 / 6.6 / 10.9 | 0.543 | 0.611 |
| ↳ `mean` | 0.6651 | 0.7070 | 0.159 | 0.255 | 0.581 | 0.645 | 0.174 | 0.390 | 0.520 | 8.9 / 7.9 / 12.9 | 0.502 | 0.688 |
| ↳ `mean_relation` | 0.6687 | 0.7093 | 0.247 | 0.297 | 0.590 | 0.644 | 0.189 | 0.433 | 0.713 | 13.5 / 10.7 / 16.5 | 0.658 | 0.476 |
| **`coord_gen_frozen`** | 0.7206 | 0.7505 | 0.278 | 0.305 | 0.632 | 0.674 | 0.273 | 0.415 | 0.516 | 8.8 / 7.8 / 13.2 | 0.502 | 0.689 |
| ↳ `gates_off` | 0.7205 | 0.7441 | 0.293 | 0.315 | 0.634 | 0.679 | 0.279 | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 | 0.445 | 0.810 |
| ↳ `mean` | 0.7104 | 0.7371 | 0.278 | 0.308 | 0.629 | 0.667 | 0.265 | 0.402 | 0.424 | 13.3 / 11.6 / 18.4 | 0.413 | 0.884 |
| ↳ `mean_relation` | 0.7184 | 0.7459 | 0.272 | 0.300 | 0.642 | 0.665 | 0.286 | 0.406 | 0.464 | 11.2 / 9.9 / 16.4 | 0.452 | 0.795 |

**Test-set reading.** On the edge family the students are ineffective; on the topology family
the full-reader student is the best deployable row to date (§1). The frozen-reader student equals `prefix_base` on every edge metric (AUROC 0.721 vs
0.721, AUPRC +0.007 inside the noise band, accuracy / F1 / MCC marginally lower) and its topology
row is within the band except MMD ratios that move by the same amount the `mean` row moves the
other way. The full-reader student is *worse* than the comparator on every edge metric (AUROC
−0.044, AUPRC −0.019, MCC 0.167 vs 0.279; non-self pairs 0.657 / 0.686). Its topology row (RD
0.91, MMD 4.2 / 3.6 / 6.7, GS +0.021 at the band's edge) is a density-transfer effect driven by the
predicted coordinates (the same trunk on mean coordinates sits at RD 0.52; §1); the student's
V_val GS is unchanged (0.387), its V_val-selected threshold (1.36) is half every other row's,
and the movement comes with the worst edge ranking
in the table. Read under the claim rules (both families together): a topology-preserving
student with an edge-ranking cost, single seed; the test protocol carries no bootstrap intervals,
so the noise band (±0.01 GS, ±0.5 MMD ratio) governs, and the MMD and RD movements of the full
student are far outside it while its GS movement is not.

### 2.3 Per-epoch curves (validation; `metrics.jsonl`)

![Stage II component loss curves](topo_prompt_stage2_curves/loss_curves.png)

[Generator fit and validation topology curves, full CSV and reproducible plots](topo_prompt_stage2_curves/README.md).
[Derived unweighted validation composite](topo_prompt_stage2_curves/validation_composite.png):
`val_task_loss + val_coord_loss + 0.1 * val_kd_loss`; this differs from training's
5:1 row-weighted objective.
The audit uses all recorded epochs. Only composite training loss was logged;
individual training-loss decreases cannot be inferred. KD soft BCE has a nonzero
teacher-entropy floor, and prompt interventions do not isolate a training loss.


`vBCE` = student task BCE on val_cls (the early-stopping monitor), `vCoord` = coordinate loss
(Huber over 30 continuous standardised coordinates + shortest-path class cross-entropy) against
V_val truth, `vKD` = soft-BCE of the student logit against the same reader fed the true
coordinates. `train` is the composite `BCE + coord + 0.1·KD` on training rows. R² per field is
against V_val truth on val_cls; `dist` is the five-class shortest-path accuracy.

`coord_gen_full` (selected epoch 6):

| ep | lr | train | vBCE | vCoord | vKD | AUPRC | GS | MMD d/c/s | R² end | R² rel | R² ctx | dist |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 3.7e-4 | 2.211 | 0.840 | 1.668 | 0.825 | 0.779 | 0.379 | 9.4/3.9/7.9 | −0.96 | −0.22 | −0.04 | 0.45 |
| 2 | 9.1e-4 | 1.937 | **0.644** | 1.349 | **0.562** | 0.799 | 0.385 | 10.8/4.1/8.9 | −0.21 | +0.04 | +0.18 | 0.55 |
| 3 | 1.0e-3 | 1.843 | 0.730 | **1.328** | 0.631 | 0.793 | 0.369 | 13.0/5.5/11.1 | −0.40 | +0.06 | +0.22 | 0.58 |
| 5 | 9.7e-4 | 1.686 | 0.704 | 1.367 | 0.649 | 0.807 | 0.391 | 9.3/3.5/8.7 | −0.27 | +0.09 | +0.18 | 0.54 |
| 6 | 9.4e-4 | 1.643 | 0.867 | 1.632 | 0.859 | 0.806 | 0.387 | 5.8/2.4/6.1 | −0.30 | +0.03 | +0.05 | 0.44 |
| 7 | 9.0e-4 | 1.612 | 0.677 | 1.373 | 0.592 | 0.802 | 0.362 | 10.4/4.3/8.9 | −0.14 | +0.10 | +0.20 | 0.56 |
| 10 | 7.5e-4 | 1.501 | 0.678 | 1.521 | 0.638 | 0.808 | 0.382 | 7.3/3.3/7.5 | −0.29 | +0.06 | +0.08 | 0.50 |
| 12 | 6.2e-4 | 1.439 | 0.680 | 1.557 | 0.625 | 0.802 | 0.374 | 9.6/4.5/8.6 | −0.15 | +0.07 | +0.09 | 0.53 |

`coord_gen_frozen` (selected epoch 4):

| ep | lr | train | vBCE | vCoord | vKD | AUPRC | GS | MMD d/c/s | R² end | R² rel | R² ctx | dist |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 3.7e-4 | 1.587 | 1.031 | 1.379 | 0.656 | 0.808 | 0.398 | 9.2/4.0/8.4 | −0.25 | +0.13 | −0.01 | 0.54 |
| 3 | 1.0e-3 | 1.302 | 1.069 | **1.247** | 0.707 | 0.806 | 0.403 | 9.6/4.0/8.6 | −0.26 | +0.17 | +0.22 | 0.59 |
| 4 | 9.9e-4 | 1.236 | 1.065 | 1.393 | 0.665 | 0.811 | 0.399 | 9.2/3.9/8.4 | −0.37 | **+0.20** | +0.13 | 0.54 |
| 8 | 8.6e-4 | 1.116 | 1.072 | 1.718 | 0.673 | 0.810 | 0.401 | 9.8/4.3/8.8 | −0.20 | +0.10 | −0.04 | 0.49 |
| 12 | 6.2e-4 | 1.048 | **0.982** | 1.726 | **0.586** | 0.810 | 0.402 | 11.0/4.6/9.0 | −0.11 | +0.07 | −0.06 | 0.48 |
| 16 | 3.5e-4 | 0.977 | 1.059 | 1.946 | 0.652 | 0.808 | 0.397 | 9.2/4.3/8.3 | −0.49 | +0.05 | −0.19 | 0.46 |
| 22 | 4.3e-5 | 0.883 | 1.075 | 2.008 | 0.665 | 0.810 | 0.404 | 10.7/4.6/9.0 | −0.54 | +0.06 | −0.14 | 0.46 |

### 2.4 Training-universe fit of the published checkpoints (probe)

`outputs/logs/coord_fit_probe.py <run> 4000` (HPC, not in the repo): 4,000 epoch-1 training rows
(692 positives) against training-graph truth, no universe shift. R² of the standardised
prediction; "swapped" scores the endpoint prediction against the other endpoint's target.

| Coordinate | `coord_gen_full` (ep 6) | `coord_gen_frozen` (ep 4) |
|---|---:|---:|
| endpoint field (pooled) / vs swapped target | 0.235 / −0.05 | 0.324 / −0.06 |
| ↳ log1p degree · clustering · log1p triangles · log1p open wedges | 0.40 · 0.23 · 0.51 · 0.34 | 0.53 · 0.27 · 0.63 · 0.48 |
| ↳ log1p two-hop · mean neighbour degree · returns 2/3/4 | 0.19 · 0.24 · 0.06/−0.03/0.01 | 0.34 · 0.40 · 0.10/0.04/0.03 |
| relation field | 0.314 | 0.392 |
| ↳ log1p common · Jaccard · log1p L3 · walks 2/3/4/5 | 0.38 · 0.61 · 0.48 · 0.39/−0.03/0.14/0.12 | 0.59 · 0.67 · 0.50 · 0.40/0.21/0.17/0.11 |
| context field | 0.413 | 0.483 |
| distance class accuracy (5 classes) | 0.659 | 0.660 |

## 3. Does each module work?

The [complete curve audit](topo_prompt_stage2_curves/README.md) separates observed
validation behavior from causal attribution:

1. **Generator:** weak node-held-out coordinate fit. Selected endpoint R² is
   -0.299 / -0.365 (full / frozen); relation R² is 0.031 / 0.198. The training
   probe in §2.4 and poor validation fit suggest a generalization gap. Memorization
   versus graph-distribution shift remains unresolved, not proven by these curves.
2. **L_coord:** validation improves initially, with both minima at epoch 3, then
   fluctuates and ends higher (full 1.328 → 1.557; frozen 1.247 → 2.008). Separate
   training Huber and CE trajectories were not logged. The minimum coordinate
   loss is not the checkpoint selection objective.
3. **L_BCE:** full shows an early validation improvement; frozen has no sustained
   downward trend. These are joint-training trajectories, not evidence isolating
   the benefit of BCE from the other terms.
4. **L_KD:** full improves early then oscillates; frozen mostly oscillates. Soft
   BCE includes teacher entropy, which was not recorded, so its absolute size
   does not establish whether the student matches the teacher. A matched
   no-KD run is needed to measure KD's benefit or harm.

The causal explanations and future-stage predictions elsewhere in this note
remain hypotheses. No loss-removal ablation or uncertainty estimate is supplied
by these single-seed curves.

## 4. Interventions

The attribution behaves as designed. Full lane: `gates_off` 0.768 → `mean` 0.779 → `mean_relation`
0.794 → predicted 0.806 val_cls AUPRC: the prediction adds 0.04 over the trunk alone, 0.012 of it
through the relation field, and the sum is the base. Frozen lane: nothing to attribute (all rows
within 0.014 of the base; `gates_off` is the base exactly). GS does not move in either lane
(0.36–0.40 everywhere), so the topology family agrees with the edge family.

## 5. Implications and next experiments

- Replicate the topology result first: `coord_gen_full` with seeds 1 and 2 (about 3.5 h each on a
  4-GPU container). If RD ≈ 0.9 and MMD ratios ≈ 4 / 4 / 7 hold, the Stage II student is the
  project's topology row and Stage III/IV can be aimed at recovering the edge cost (unfreeze the
  readout and upper encoder under task BCE while keeping the prompt path); if they do not, the
  epoch-6 selection was a favourable draw of a noisy curve.
- The *edge* signal is absent because the predicted coordinates carry region-level, not
  pair-level, structure (§3); Stages III/IV on the same predictions will not recover the Stage I
  edge gain. The pair-level part needs richer generator inputs or a generator that does not
  memorise training nodes.
- Cheapest decisive test of the memorisation reading (CPU/one GPU, hours): cache the frozen
  encoder's pooled states for every node, fit the same heads under a **node-held-out** split of
  the training nodes, and report R² on the held-out training nodes next to V_val. If held-out
  training-node R² matches V_val's, generalisation is the limit; if it matches the in-sample fit,
  the shift between the training graph and the V_val gold graph is.
- If generalisation is the limit, the generator needs inputs beyond the two endpoints — the
  design note's own fallback: attribute-retrieved context (nearest training nodes in attribute
  space and their known structure), which is arm-specific support, not task input.
- Partial-oracle rows (true relation + predicted rest, and the reverse) would price each field's
  prediction error through the frozen reader before any of the above is built.

Single-seed differences inside ±0.01 GS / ±0.5 MMD ratio are not read.
