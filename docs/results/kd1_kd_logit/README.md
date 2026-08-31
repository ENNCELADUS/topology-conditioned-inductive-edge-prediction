# KD1 (`kd_logit`): model design and learning curves

**Status (verified 2026-08-23):** KD1 is the current full-row `kd_logit` arm, not the retired sampled-stream D1. Its 25-epoch student run completed, published epoch 7, completed held-out testing, has no `failure.json`, and had released its two H20 GPUs when inspected. This document records design and training dynamics; it does not make a four-arm comparison claim.

## 1. Arm identity and data flow

For every official training row `(x_u, x_v, y)`, one frozen teacher target is joined to the same row by `_row_id`. The student performs one forward pass; both supervised and KD losses use its single output logit. There is no KD-only sampler, extra pair distribution, or second student forward.

```text
(x_u, x_v, y, row_id) ──> V3.1 student ──> student logit s
               row_id ──> frozen target bank ──> teacher logit t

L_total = 1.0 · L_task(s, y) + 1.0 · L_KD(s, t)
```

The frozen target artifact is `kd_row_targets_v1`: 86,498 training rows, 22,804 V_val classification rows, 42,238 positive training rows, and 8,070 nodes. Teacher inference masks the queried edge; V_val-internal pairs are validation-only, while the teacher structural source is the V_val-quarantined training graph.

## 2. Teacher

The teacher is the diagnostic-only Full-Ego oracle checkpoint at `outputs/egostitch_e2e_stage1_v3/full_ego_teacher_kd/best.pt` (checkpoint `c390709fb7070c62`, epoch 6). It is used only to dump targets; it is frozen during student training and is not a deployable task input.

| component | design |
|---|---|
| generator | `full_ego_oracle`; truth source `training_structure`; `n_ground=50`; row LayerNorm feature standardization |
| encoder | `grit_gmt`; width 96; 4 layers; RRWP `k=8`; 8 heads; 4 GMT seeds; `w_rel=0` |
| classifier | `b0_v31`; 8 cross-attention heads; 1 injection layer; `p_topo=0.15`; `pooled_adapter`; no permanent-null arm |
| KD1 target | one fp32 teacher logit `t_i` per official row; KD1 does not consume the stored teacher representations or PMA seeds |

The target manifest binds the teacher checkpoint SHA-256 `8d7cdad619f130db8df19413692005919bde66a66fdfeb1763a08321c40f7469` and reports the same checkpoint id. The published teacher checkpoint belongs to successful attempt `e62c9008d1144177a691a71203b4aff2`; an older failed attempt remains archived and is not the source checkpoint.

## 3. Student

KD1 uses the endpoint-only V3.1 pair scorer. Its strict task input remains exactly `(x_u, x_v)`; teacher logits exist only during training.

| group | hyperparameters |
|---|---|
| encoder | input 1536; model width 512; 3 encoder layers; 3 cross-attention layers; 8 heads |
| pooling/readout | mean + attention + max + gated pooling; `pair_context_gated`; AB/BA max aggregation; no mixing |
| MLP head | `[512, 256, 128]`; GELU; LayerNorm; dropout 0.2; no spectral norm |
| regularization | dropout 0.1; token dropout 0.1; cross-attention dropout 0.1; stochastic depth 0.1; symmetric label smoothing 0.05 |
| optimizer | AdamW path; configured optimizer LR and OneCycle max LR `1e-4`; weight decay 0.05; gradient clip 1.0 |
| schedule | OneCycle; effective initial LR `4e-6`; `pct_start=0.1`; `div_factor=25`; `final_div_factor=10000`; cosine annealing; 25 epochs |
| batching/runtime | global token budget 524,288; max 4,096 pairs/rank; 2 H20 ranks; bf16; seed 0 |

The run processed every one of the 86,498 training rows once per epoch (`training_coverage_exact=true`) for 2,550 optimizer steps. The artifact-native run id is attempt `da164bba9c514105a547f7ff77a57a5d`; published student checkpoint id is `231a7f83886861f1`.

## 4. Loss terms and weights

Let `s_i` be the student logit, `t_i` the frozen teacher logit, `y_i∈{0,1}`, and label-smoothing coefficient `ε=0.05`. The smoothed label is `ỹ_i=(1-ε)y_i+ε/2`.

| term | definition | weight |
|---|---|---:|
| supervised task | `L_task = mean(BCEWithLogits(s_i, ỹ_i))` | 1.0 |
| pointwise logit KD | `L_KD = mean(BCEWithLogits(s_i, sigmoid(t_i)))` | `w_logit=1.0` |
| total | `L_total = L_task + L_KD` | — |

All other distillation weights are exactly zero: `w_rank=w_dist=w_gram=w_rep=w_seed=w_geom=w_kl=0`. The generic defaults `margin=0.1`, `temperature=1.0`, and `kl_warmup_steps=2000` are inactive in KD1 and do not affect its objective.

## 5. Learning curves

![KD1 train and validation learning curves](learning_curves.png)

Figure: training and validation curves for supervised task loss (left), pointwise KD loss (middle), and their weighted total summary (right). The dotted vertical line marks the published checkpoint at epoch 7. [Vector PDF](learning_curves.pdf).

The exact epoch-level source table is [available as CSV](learning_curves.csv); [the Python plotting script](plot_learning_curves.py) reads this file directly and reproduces both the PDF and PNG.

The figure contains exact primitive-term values at every epoch: train/validation `L_task` and train/validation `L_KD`. Train terms come directly from `metrics.jsonl`. Because the original run did not persist validation objective losses, the validation terms were replayed on 2026-08-23 from all 25 saved `epoch-XXXX.pt` checkpoints over the exact 22,804-row V_val classification block, using the same bf16 forward and loss definitions. `val_brier` and `val_kd_prob_mae` were not substituted for losses.

The total curves/tables are deterministic sums of the displayed term summaries because both weights are 1.0. Validation terms share one row-weighted aggregation, so their sum is the validation objective mean. The artifact aggregated train `L_task` by rows but `train_kd_loss` by steps/ranks; consequently, the displayed train total is a convenient derived summary, not an artifact-native mean optimizer loss.

| checkpoint/point | train `L_task` | train `L_KD` | derived train total | val `L_task` | val `L_KD` | val `L_total` |
|---|---:|---:|---:|---:|---:|---:|
| published epoch 7 | 0.400545 | 0.359821 | 0.760366 | 0.442098 | 0.363260 | 0.805359 |
| minimum validation total, epoch 11 | 0.339152 | 0.301548 | 0.640700 | **0.416763** | **0.338672** | **0.755435** |
| final epoch 25 | **0.237211** | **0.237375** | **0.474587** | 0.448988 | 0.348961 | 0.797949 |

The training terms decrease through epoch 25, while both validation loss terms bottom at epoch 11 and then rise. This is evidence of a widening train/validation loss gap after epoch 11. It does **not** imply that epoch 11 should replace the published checkpoint: production checkpoint selection mean-ranks validation AUPRC together with BFS-macro GS/RD and degree/clustering/spectral MMD ratios, and selected epoch 7. The loss curve is diagnostic, not the selection rule.

## 6. Artifact provenance

| artifact | verified identity/state |
|---|---|
| student config | `configs/b1_kd_logit_breadth_first.yaml`; config hash `023d63960dbea56fb524d67eeaaffafee3c5aa5336bfcc9807903052dece0d7b` |
| student metrics/checkpoints | `outputs/b1_row_kd/kd_logit`; epochs 1–25 present; checkpoint `231a7f83886861f1`; selected epoch 7 |
| student terminal state | `complete.json=status:complete`; `test_complete.json=status:test_complete`; no `failure.json`; GPUs released |
| teacher targets | `outputs/distill/kd_row_targets_breadth_first`; format `kd_row_targets_v1`; target NPZ SHA-256 `ef65e905da9da0a78a83c9a4c6f198ac73e1c65d11b1d96dccf0cfa9d5b4d82b` |
| teacher checkpoint | `outputs/egostitch_e2e_stage1_v3/full_ego_teacher_kd/best.pt`; id `c390709fb7070c62`; epoch 6; diagnostic-only |
| validation-loss replay | all 25 student checkpoints; exact V_val row ids; 22,804 rows/checkpoint; H20 bf16 inference; completed 2026-08-23; persisted in `learning_curves.csv` |
