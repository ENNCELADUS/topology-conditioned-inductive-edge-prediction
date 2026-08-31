# KD3 (`kd_gram`): model design and learning curves

**Status (verified 2026-08-23):** KD3 is the current full-row cosine-Gram arm. Training stopped after epoch 24 under patience 10, published epoch 14, completed held-out testing, has no `failure.json`, and had released its two H20 GPUs when inspected.

## 1. Arm identity and teacher

The strict student input is `(x_u, x_v)`. Each official row joins by `_row_id` to the frozen teacher pooled representation from `kd_row_targets_v1`; all distinct row pairs in the normal global task batch contribute to Gram matching, including pairs that share an endpoint. There is no KD-only sampler or second student forward.

```text
(x_u, x_v, y, row_id) ──> V3.1 student ──> logit s, pair representation z_s
               row_id ──> frozen target bank ──> teacher representation z_t
L_total = L_task + 1.0 · L_gram
```

The frozen teacher is the diagnostic-only Full-Ego oracle checkpoint `outputs/egostitch_e2e_stage1_v3/full_ego_teacher_kd/best.pt` (id `c390709fb7070c62`, epoch 6): `full_ego_oracle` generator (`n_ground=50`), 4-layer width-96 `grit_gmt` encoder with RRWP `k=8`, and `b0_v31` classifier. The shared target bank has 86,498 train rows, 22,804 V_val rows, 42,238 positive train rows, and 8,070 nodes. Teacher checkpoint SHA-256 is `8d7cdad619f130db8df19413692005919bde66a66fdfeb1763a08321c40f7469`; target NPZ SHA-256 is `ef65e905da9da0a78a83c9a4c6f198ac73e1c65d11b1d96dccf0cfa9d5b4d82b`.

## 2. Student and training hyperparameters

| group | hyperparameters |
|---|---|
| encoder | input 1536; width 512; 3 encoder layers; 3 cross-attention layers; 8 heads |
| pooling/readout | mean + attention + max + gated; `pair_context_gated`; AB/BA max; no mixing |
| head/regularization | MLP `[512,256,128]`; GELU; LayerNorm; head dropout 0.2; dropout/token/cross-attention/stochastic-depth 0.1; label smoothing 0.05 |
| optimizer/schedule | AdamW path; max LR `1e-4`; weight decay 0.05; clip 1.0; OneCycle cosine (`pct_start=0.1`, div 25, final div 10000) |
| runtime | configured 25 epochs, stopped at 24; 524,288 global tokens; 4,096 pairs/rank; 2 H20 ranks; bf16; seed 0 |

All 86,498 rows were covered exactly per completed epoch for 2,448 steps. Attempt id is `7a4f699e1bfa436dbb0c894574622dec`; published checkpoint id is `acf10c51d1ab373c`.

## 3. Loss terms and weights

For row-normalized representation matrices `Z_s` and `Z_t`, let `G_s=Z_s Z_s^T` and `G_t=Z_t Z_t^T`. Diagonal elements are excluded.

| term | definition | weight |
|---|---|---:|
| supervised task | smoothed-label `BCEWithLogits`, `ε=0.05` | 1.0 |
| cosine-Gram KD | mean `(G_s[j,k]-G_t[j,k])²` over all `j≠k` in the gathered global batch | `w_gram=1.0` |
| total | `L_task + L_gram` | — |

All other KD weights are zero. The relational gather lets rows on different ranks interact within the same global batch.

## 4. Learning curves

![KD3 train and validation learning curves](learning_curves.png)

Figure: train/validation task, Gram, and derived total losses. The dotted line marks published epoch 14. [Vector PDF](learning_curves.pdf); [CSV](learning_curves.csv); [plot script](plot_learning_curves.py).

Train task and Gram losses are artifact-native. Validation Gram block loss was persisted by the run. Validation task loss was replayed on 2026-08-23 from all 24 checkpoints over the exact 22,804-row V_val block with the original bf16 forward. Derived totals sum the displayed unit-weight terms; because train and relational validation terms use their native aggregations, these totals are diagnostic summaries rather than a newly recovered per-step optimizer trace.

| point | train task | train Gram | derived train total | val task | val Gram | derived val total |
|---|---:|---:|---:|---:|---:|---:|
| minimum derived val total, epoch 8 | 0.387081 | 0.077277 | 0.464358 | **0.421743** | 0.072948 | **0.494691** |
| published epoch 14 | 0.289333 | 0.062344 | 0.351677 | 0.457294 | **0.069141** | 0.526436 |
| final epoch 24 | **0.212990** | **0.054831** | **0.267821** | 0.510952 | 0.070036 | 0.580988 |

The Gram term remains stable on validation while validation task loss rises after the early epochs; the widening gap is driven mainly by supervised generalization. This classification-only run selected epoch 14 by validation AUPRC, not by minimum loss; topology metrics were computed only after publication.

## 5. Held-out result

| edge metric | value |
|---|---:|
| AUPRC / AUROC | 0.736020 / 0.710838 |
| ECE / Brier | 0.215131 / 0.267979 |

| topology operating point | GS ↑ | RD → 1 | degree MMD ↓ | clustering MMD ↓ | spectral MMD ↓ |
|---|---:|---:|---:|---:|---:|
| primary fixed V_val threshold (`logit=3.296875`) | 0.393731 | 0.509612 | 8.842574 | 7.879546 | 13.693224 |
| per-subgraph RD=1 diagnostic | 0.435947 | 1.000000 | 1.827655 | 2.432006 | 3.706364 |

The per-subgraph row is oracle-calibrated and nondeployable. The fixed-threshold topology result remains under-dense and poorly calibrated in all three MMD ratios.

## 6. Artifact provenance

| artifact | verified identity/state |
|---|---|
| config | source name `configs/b1_kd_gram_breadth_first.yaml`; artifact-recorded serialized config hash `bbea5ee56a928f96cb5301ba219478f852118148293f4cfc3cdd773c6c16d23b` (not asserted as the current working-tree file hash) |
| run | `outputs/b1_row_kd/kd_gram`; epochs 1–24; selected epoch 14; checkpoint `acf10c51d1ab373c` |
| terminal state | `complete.json` and `test_complete.json` present; no `failure.json`; GPUs released |
| validation replay | 24 checkpoints × 22,804 exact V_val rows; H20 bf16; completed 2026-08-23 |
