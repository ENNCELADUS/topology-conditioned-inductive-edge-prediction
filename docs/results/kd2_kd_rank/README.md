# KD2 (`kd_rank`): model design and learning curves

**Status (verified 2026-08-23):** KD2 is the current full-row anchor-relational arm. Its 25-epoch run completed, published epoch 24, completed held-out testing, has no `failure.json`, and had released its two H20 GPUs when inspected. This is not the retired sampled-stream D2 protocol.

## 1. Arm identity and data flow

Each official row `(x_u, x_v, y, row_id)` receives the frozen teacher logit for that exact row. Every non-self row participates twice in relational KD, once under endpoint `u` and once under endpoint `v`; self rows participate once. Relational comparisons use only rows in the normal global task batch, with no partner sampler or second student forward.

```text
(x_u, x_v, y, row_id) ──> V3.1 student ──> student logit s
               row_id ──> frozen target bank ──> teacher logit t

L_total = L_task + 1.0 · L_rank + 1.0 · L_dist
```

The shared `kd_row_targets_v1` artifact contains 86,498 training rows, 22,804 V_val classification rows, 42,238 positive training rows, and 8,070 nodes. Teacher inference masks the queried edge; V_val-internal pairs are validation-only.

## 2. Teacher

The frozen teacher is the diagnostic-only Full-Ego oracle checkpoint `outputs/egostitch_e2e_stage1_v3/full_ego_teacher_kd/best.pt` (checkpoint `c390709fb7070c62`, epoch 6). It is used only to dump exact-row targets and is not available to the deployed student.

| component | design |
|---|---|
| generator | `full_ego_oracle`; truth source `training_structure`; `n_ground=50`; row LayerNorm feature standardization |
| encoder | `grit_gmt`; width 96; 4 layers; RRWP `k=8`; 8 heads; 4 GMT seeds; `w_rel=0` |
| classifier | `b0_v31`; 8 cross-attention heads; 1 injection layer; `p_topo=0.15`; `pooled_adapter` |
| KD2 target | one fp32 teacher logit per official row; stored teacher representations and PMA seeds are unused |

The target manifest binds teacher checkpoint SHA-256 `8d7cdad619f130db8df19413692005919bde66a66fdfeb1763a08321c40f7469`; target NPZ SHA-256 is `ef65e905da9da0a78a83c9a4c6f198ac73e1c65d11b1d96dccf0cfa9d5b4d82b`.

## 3. Student and training hyperparameters

The endpoint-only V3.1 student receives exactly `(x_u, x_v)` at inference.

| group | hyperparameters |
|---|---|
| encoder | input 1536; width 512; 3 encoder layers; 3 cross-attention layers; 8 heads |
| pooling/readout | mean + attention + max + gated pooling; `pair_context_gated`; AB/BA max aggregation; no mixing |
| MLP head | `[512, 256, 128]`; GELU; LayerNorm; dropout 0.2; no spectral norm |
| regularization | dropout 0.1; token/cross-attention dropout 0.1; stochastic depth 0.1; label smoothing 0.05 |
| optimizer | AdamW path; max LR `1e-4`; weight decay 0.05; gradient clip 1.0 |
| schedule | OneCycle cosine; `pct_start=0.1`; `div_factor=25`; `final_div_factor=10000`; 25 epochs |
| runtime | global token budget 524,288; max 4,096 pairs/rank; 2 H20 ranks; bf16; seed 0 |

All 86,498 training rows were processed once per epoch (`training_coverage_exact=true`) for 2,550 optimizer steps. Attempt id is `d4d77cdb53fa4539b24f054c9ce5609b`; published checkpoint id is `887788d88d02422c`.

## 4. Loss terms and weights

Let `G_a` be rows sharing anchor `a`. For ordered pairs `j,k∈G_a` with distinct teacher logits, ranking uses margin `m=0.1`. Distribution KD uses `T=1.0` and excludes singleton groups.

| term | definition | weight |
|---|---|---:|
| supervised task | smoothed-label `BCEWithLogits`, smoothing `ε=0.05` | 1.0 |
| anchor ranking | `mean ReLU[m - sign(t_j-t_k)(s_j-s_k)]` over eligible within-anchor ordered comparisons | `w_rank=1.0` |
| anchor distribution | mean per-anchor `KL(softmax(t/T) || softmax(s/T))` | `w_dist=1.0` |
| total | `L_task + L_rank + L_dist` | — |

All other KD weights are zero. The ranking and distribution losses are formed after the two-rank relational gather, so cross-rank rows in the same global task batch can interact.

## 5. Learning curves

![KD2 train and validation learning curves](learning_curves.png)

Figure: task loss, combined relational KD loss, the two atomic validation KD losses, and the derived total. The dotted line marks published epoch 24. [Vector PDF](learning_curves.pdf); [CSV](learning_curves.csv); [plot script](plot_learning_curves.py).

Train `L_task` and combined `L_rank+L_dist` come directly from `metrics.jsonl`. The original run did **not** persist the two atomic online train losses separately; `train_rank_loss` and `train_dist_loss` are therefore intentionally blank in the CSV and are not fabricated in the figure. Exact validation block losses for both atomic terms were persisted and are shown separately. Validation task loss was replayed on 2026-08-23 from every saved checkpoint over the exact 22,804-row V_val block using the original bf16 forward and loss definition.

| point | train task | train KD combined | derived train total | val task | val rank | val dist | derived val total |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimum derived val total, epoch 9 | 0.406933 | 0.329198 | 0.736131 | 0.415093 | 0.186667 | 0.235354 | **0.837114** |
| published epoch 24 | 0.281685 | 0.199403 | 0.481088 | 0.405449 | 0.204990 | 0.273504 | 0.883942 |
| final epoch 25 | **0.281302** | **0.197633** | **0.478935** | 0.405866 | 0.205098 | 0.273797 | 0.884761 |

The online training terms fall throughout training, while the derived validation summary bottoms at epoch 9 and later rises. This diagnoses a train/validation objective gap; it does not override this classification-only run's checkpoint rule, which selected epoch 24 by validation AUPRC rather than by minimum loss. Topology metrics were computed only after checkpoint publication.

## 6. Held-out result

| edge metric | value |
|---|---:|
| AUPRC / AUROC | 0.741062 / 0.719392 |
| ECE / Brier | 0.175073 / 0.247090 |

| topology operating point | GS ↑ | RD → 1 | degree MMD ↓ | clustering MMD ↓ | spectral MMD ↓ |
|---|---:|---:|---:|---:|---:|
| primary fixed V_val threshold (`logit=2.53125`) | 0.382678 | 0.443259 | 11.939408 | 10.950879 | 16.632809 |
| per-subgraph RD=1 diagnostic | 0.432714 | 1.000000 | 2.102057 | 2.551019 | 3.751541 |

The per-subgraph row is oracle-calibrated and nondeployable; the fixed-threshold row is the protocol result. The low fixed-threshold RD and large MMD ratios show that the diagnostic improvement cannot be claimed at the deployable operating point.

## 7. Artifact provenance

| artifact | verified identity/state |
|---|---|
| config | source name `configs/b1_kd_rank_breadth_first.yaml`; artifact-recorded serialized config hash `8034ee330781454c009e518a6ad68ed3e3f78787e89c3378239aa6619d4b934c` (not asserted as the current working-tree file hash) |
| run | `outputs/b1_row_kd/kd_rank`; epochs 1–25; selected epoch 24; checkpoint `887788d88d02422c` |
| terminal state | `complete.json` and `test_complete.json` present; no `failure.json`; GPUs released |
| validation replay | 25 checkpoints × 22,804 exact V_val rows; H20 bf16; completed 2026-08-23 |
