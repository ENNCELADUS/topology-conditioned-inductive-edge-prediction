# Experiments: Topology-Conditioned Inductive Edge Prediction

**Scope (2026-09-11):** the 10% positive-budget, seed-42 node-held-out split only.
All results below use `geometric_rd_five_rank_v1` and `test_protocol_v8`.
The ablation study is incomplete: B0 and GRAND have test results; KD and NEW
selection is ongoing. PMA1 is a separate true-structure oracle reference.

## 1. Experimental setup

### 1.1 Dataset and split

| Partition | Nodes | Loopless positive edges | Role |
|---|---:|---:|---|
| Training substrate (`train_graph.pkl`) | 8,072 | 47,762 | Original train and validation positives |
| Effective training | 7,203 | 31,623 | All pairs touching V_val removed |
| V_val | 869 | 4,703 | Node-held-out selection and validation |
| Excluded boundary | — | 11,436 | Cross-boundary positives excluded from training |
| Test | 2,018 | 30,128 | Disjoint held-out nodes |

A single sorted-neighbor FIFO BFS starts at `node_007630`, drawn uniformly from
five-neighbor roots with split seed 42. It stops before exceeding 10% of the
substrate's positive pairs, counting self-loops once: 5,347 validation positives
(including 644 self-loops), within the 5,364 cap. Training retains 36,857 positive
pairs including self-pairs. Neither test labels nor test statistics select the
root or budget. The [split manifest](../data/val_region/breadth_first.json) fixes
membership and the validation buckets.

Validation uses 50 BFS samples at each size 20, 40, ..., 200 (500 subgraphs;
bucket seed 43), plus 5,347 positive and 5,347 negative classification pairs.
Test classification contains 32,019 positive and 32,019 negative pairs; test
topology uses its own fixed sampled-set pair union. Official metrics retain
self-loops. Negative pairs are sampled non-edges, not experimentally confirmed
non-interactions.

### 1.2 Training and information boundaries

The deployed interface is exactly `(x_u, x_v) -> edge(u,v)`. Frozen intrinsic
features have at most 1,024 tokens of dimension 1,536 per node. Any inferred
structural context must help decide that queried edge; target graph structure
is unavailable to deployable models.

B0 and KD variants share the V3.1 backbone: dimension 512, three encoder layers,
eight heads, `mixing.mode: none`, and pair-context-gated readout with `abba_max`
aggregation. AdamW uses learning rate 1e-4, weight decay 0.05, a one-cycle schedule,
at most 25 epochs, gradient clipping 1.0, and BF16 DDP. All reported runs use
training seed 0. Early stopping monitors validation task BCE; per-trial configs
record patience and topology evaluation cadence.

Teacher and students use the shared dynamic 1:5 positive-to-negative training
sampler. No training pair touches V_val. Students sample globally before DDP
batch distribution; the teacher samples per rank. Positive BCE weight is 5,
negative weight is 1, and label smoothing is zero. Thus the sampler is shared,
while the teacher's exact multi-GPU pair stream need not match the student's.

KD uses fresh PMA1 training banks for this split. Row banks cover the deduplicated
multi-epoch corpus with exact pair/label joins; each epoch consumes its own 1:5
subset. Context banks provide the separate strict-LLP objective. Teacher targets
are training-only; optional true-G_val targets are diagnostics and never supply
training targets or selection losses. Teacher training excludes V_val nodes;
its oracle evaluation reads the corresponding true validation/test structure.

Structural variants use no teacher. Each step additionally scores legal pairs
in a sampled 40-node training subgraph, with 32 locally expanded and eight
uniform background nodes. The loss supervises edge decisions within this
subgraph; inference still receives only the queried endpoints.

The topology-prompt Stage I runs (`v3_1_topo_prompt`; spec
`superpowers/specs/2026-09-12-topology-prompt-stage1-design.md`) are true-structure
diagnostics like the oracle, not deployable arms: the queried pair's structural
coordinates are measured on the universe's true graph with the queried edge
removed (training graph for training rows, V_val gold graph for validation, the
labelled test graph for held-out scoring) and read by the prefix_base trunk
through gated key/value prefixes. `topo_prompt_full` trains trunk and prompt from
scratch under the prefix_base recipe; `topo_prompt_frozen` trains only the prompt
on the frozen prefix_base. Their `gates_off`, `mean` and `shuffle` interventions
attribute any gain to the structural content. Results, when complete, are
reported as a ceiling row beside the oracle in §3.2 and never in the
endpoint-only table.

## 2. Selection and evaluation

For each checkpoint, enumerate distinct V_val sampled-union logit boundaries
with atomic ties and `logit >= threshold`. Let
`L(t) = mean_size(mean_subgraph(log RD_i(t)))`. Choose the threshold minimizing
`abs(L(t))`; geometric RD is `exp(L(t))`. Exact ties prefer higher GS, lower
geometric-mean MMD, then the larger threshold. A threshold emptying a
positive-reference subgraph has infinite density error.

At each checkpoint's own threshold, rank AUPRC and GS descending and degree,
clustering and spectral MMD ratios ascending. Select the lowest equal mean of
these five ranks; ties prefer higher GS, lower geo-MMD, then the earlier epoch.
RD is reporting-only. Optuna optimizes AUPRC, GS and geo-MMD and selects its final
trial by the same five-metric ranking, breaking remaining ties by earlier trial
number. There is no RD eligibility band. Test results never select a trial.

The selected checkpoint carries its frozen topology threshold into test.
Classification separately freezes the max-F1 threshold on `val_cls` for
Accuracy/F1/MCC. AUROC/AUPRC use raw logits; ECE/Brier use raw sigmoid probabilities.
Teacher oracle threshold selection is a separate true-G_val diagnostic surface.

Report BFS-macro GS (edge-set Dice/F1, higher), arithmetic RD (toward 1), and all
three MMD ratios (lower). Also report geometric RD and mean absolute log RD;
geometric RD near 1 can hide opposing density errors across subgraphs. MMD ratios
are normalized by the deterministic real-vs-real floor. All current results are
single-seed point estimates; bootstrap intervals and seed replication are not
yet available, so small differences are descriptive, not significance claims.

Existing saved epochs were rescored and reselected under this rule without
retraining the completed runs. These observations do not reconstruct an HPO
trajectory that used the new objectives from its first trial. The original
PMA1 teacher remains the frozen bank source; its test below reuses its scores
with the new oracle threshold, not a newly selected or retrained teacher.

## 3. Ablation study

The degree/motif follow-up uses task BCE plus sampled-subgraph BCE, degree and motif
losses, with RD explicitly off or on. Its 12-trial HPO searches degree/motif in
[0.01, 1] and active RD in [0.03, 1] (log scales), starting with two matched RD-off/on
pairs. A separate BCE-only run matches its sampler, architecture and stopping policy.
Both retain positive weight 5, `mixing:none`, and the split-seed-42 training universe.

Structural scheduling was corrected on 2026-09-12: the default now assigns one
subgraph to every global optimizer step, rotating ranks and compensating for DDP
gradient averaging. The per-epoch subgraph budget is unchanged. Earlier four-rank
runs concentrated supervision in the final quarter of each epoch; their results
remain observations under that schedule. The correction's metric and wall-time
effects require matched reruns; existing results do not measure them.

For these new configs, `eval.early_stop_metric: val_total_loss` monitors validation task
BCE plus the weighted per-subgraph mean of all active structural terms on a fixed
validation sample. These terms are evaluated every epoch; patience is 10 epochs,
with stopping deferred to a topology-evaluation epoch (cadence 2, maximum 25 epochs).
Checkpoint and HPO selection retain the five-metric ranking; total loss is not compared
across trials. Other configs continue to monitor validation task BCE. HPO skips test.

Soft adjacency is the masked symmetric sigmoid-score matrix, with zero diagonal.
Degree and open/closed two-hop targets describe the sampled legal subgraph, not
full-network degrees. These are differentiable structural surrogates; class-weighted
BCE does not establish calibrated interaction probabilities. No sigmoid shift or
structural BCE reweighting is introduced in this study.

### 3.1 Endpoint-only variants

Each row is an independent variant of the common backbone, not a cumulative
addition of the preceding row. B0 is the zero-KD, zero-structural-loss control.
Only completed tests of validation-selected checkpoints receive numbers.
`—` means no test result, not zero performance or evidence of failure.

| Variant | Added training signal | Test AUROC ↑ | Test AUPRC ↑ | GS ↑ | RD → 1 | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | Task BCE only | 0.7006 | 0.7363 | 0.4296 | 0.5574 | 9.667 | 8.118 | 14.549 |
| + Logit KD (`kd_logit`) | Teacher soft-target BCE | 0.7024 | 0.7357 | 0.4157 | 0.5325 | 8.485 | 7.490 | 12.815 |
| + Rank KD (`kd_rank`) | Context rank and distribution matching | 0.7290 | 0.7543 | 0.4037 | 0.3829 | 19.597 | 16.234 | 28.092 |
| + Representation KD (`kd_rep`) | Per-row teacher representation cosine | 0.7037 | 0.7365 | 0.4293 | 0.4863 | 13.818 | 11.569 | 20.370 |
| + Rank and representation KD (`kd_rank_rep`) | Joint context and representation losses | 0.7050 | 0.7364 | 0.4217 | 0.4609 | 15.068 | 12.726 | 22.382 |
| + Logit and representation KD (`kd_logit_rep`) | Joint soft-target BCE and per-row cosine losses | — | — | — | — | — | — | — |
| + Gram KD (`kd_gram`) | Teacher cosine-Gram matching | 0.7175 | 0.7428 | 0.4304 | 0.5524 | 11.631 | 9.516 | 16.227 |
| + Structural-stream BCE (`struct_bce`) | Additional subgraph BCE, no topology terms | — | — | — | — | — | — | — |
| + GRAND (`struct_grand`) | Subgraph BCE, soft GS and RD losses | 0.7140 | 0.7392 | 0.4312 | 0.6210 | 8.063 | 6.740 | 11.872 |
| + NEW (`struct_new`) | Subgraph BCE, neighbor rank, degree and motif losses | 0.7013 | 0.7324 | 0.4258 | 0.6540 | 8.348 | 6.846 | 12.021 |

Each arm uses its V_val-selected checkpoint, not the best held-out score. The completed
KD grids select Logit weight 100, Rep weight 0.01 and Gram weight 100; Rank selects
trial 008 and Rank+Rep trial 004. GRAND is trial 006 / epoch 14, NEW is trial 009 /
epoch 10, and B0 is epoch 8. Logit+Rep HPO has completed but has no test report;
its row remains blank, as does the untested structural-stream BCE control.

Rank improves test AUPRC but worsens GS and all three MMD ratios. Logit improves
all three MMD ratios but lowers GS; Gram improves AUPRC and slightly improves GS
while worsening MMD. NEW improves MMD but lowers AUPRC and GS. No tested KD arm
improves all reported edge and topology metrics over B0.

GRAND exceeds B0 by 0.0029 AUPRC and 0.0016 GS and improves RD and all three MMD
ratios in this seed. However, B0 versus GRAND changes both the structural sample
stream and the loss: the missing `struct_bce` row is required to isolate the
GRAND topology terms. This is an incomplete ablation, not proof of a standalone
loss effect or a finalized method. No related-work baseline result is claimed
by this table.

### 3.2 Oracle reference and calibration

The teacher consumes true graph structure and has a different architecture.
It is an information-access reference, not an endpoint-only ablation or a
candidate deployable model.

| Oracle reference | Test AUROC ↑ | Test AUPRC ↑ | GS ↑ | RD → 1 | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-Ego PMA1 | 0.9498 | 0.9547 | 0.6019 | 0.7079 | 6.962 | 6.430 | 11.472 |
| Topology prompt, Stage I (`topo_prompt_full`, true coordinates) | 0.9295 | 0.9392 | 0.4608 | 0.3442 | 33.345 | 25.267 | 44.234 |

The topology-prompt row is a second true-structure ceiling, not a comparator: the
prefix_base trunk reads the queried pair's structural coordinates measured on the
test graph with the query removed. Its V_val-selected threshold under-densifies the
denser test region (RD 0.34), so its test topology numbers are dominated by the
density shift; the V_val readout and the interventions that attribute the gain to
the structural content are in `docs/results/topo_prompt_stage1.md`.

| Model | Accuracy ↑ | F1 ↑ | MCC ↑ | ECE ↓ | Brier ↓ | Test geometric RD → 1 | Test mean absolute log RD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6044 | 0.6709 | 0.2283 | 0.2017 | 0.2663 | 0.5348 | 0.6266 |
| + GRAND | 0.6321 | 0.6794 | 0.2766 | 0.3092 | 0.3289 | 0.5956 | 0.5217 |
| + Rank | 0.6530 | 0.6803 | 0.3105 | 0.1713 | 0.2424 | 0.3613 | 1.0181 |
| + Rank+Rep | 0.6166 | 0.6692 | 0.2459 | 0.2232 | 0.2748 | 0.4385 | 0.8259 |
| + Logit | 0.6116 | 0.6730 | 0.2408 | 0.2669 | 0.2991 | 0.5169 | 0.6603 |
| + Rep | 0.6088 | 0.6736 | 0.2371 | 0.2150 | 0.2726 | 0.4635 | 0.7698 |
| + Gram | 0.6228 | 0.6843 | 0.2667 | 0.1537 | 0.2429 | 0.5260 | 0.6447 |
| + NEW | 0.6159 | 0.6695 | 0.2450 | 0.3168 | 0.3293 | 0.6211 | 0.4913 |
| PMA1 oracle | 0.8762 | 0.8794 | 0.7534 | 0.0993 | 0.1130 | 0.6670 | 0.4451 |
| Topology prompt, Stage I (ceiling) | 0.7909 | 0.7414 | 0.6299 | 0.0774 | 0.1151 | 0.3000 | 1.2042 |

GRAND's calibration is worse than B0 despite its edge/topology gains. Even the
oracle's validation geometric RD of 1.0000 falls to 0.6670 on test: the frozen
operating point does not eliminate density transfer error.

## 4. Result provenance

The added KD and NEW reports were verified on H20 on 2026-09-12 against their
published checkpoint IDs and merged score artifacts, with no run-level failure
marker. The KD follow-up queue reached `complete` and its container released all
GPUs. B0, GRAND and the oracle retain their 2026-09-11 verification. Direct tests
do not write the pipeline's `test_complete.json`; terminal execution and reports
establish their completion. Full per-size metrics and score provenance are
preserved in the linked raw reports.

| Model | Checkpoint ID | Frozen topology logit threshold | Raw report |
|---|---|---:|---|
| B0, epoch 8 | `7e78609f8c904284` | 1.687500 | [B0](results/split_seed42_geometric_20260910/b0_test_report.json) |
| GRAND, trial 006 / epoch 14 | `47e62521e7491bee` | 0.992188 | [GRAND](results/split_seed42_geometric_20260910/grand_trial006_test_report.json) |
| PMA1 oracle, fixed bank-source checkpoint | `94fa2e50d9fc6c46` | 5.431674 | [PMA1 diagnostic](results/split_seed42_geometric_20260910/teacher_pma1_diagnostic_test_report.json) |
| Rank, trial_008 / epoch 7 | `384d8d93ef40f347` | 3.640625 | [Rank](results/split_seed42_geometric_20260910/rank_trial008_test_report.json) |
| Rank+Rep, trial_004 / epoch 7 | `d13ac2b88ebf36ac` | 1.820312 | [Rank+Rep](results/split_seed42_geometric_20260910/rank_rep_trial004_test_report.json) |
| Logit, kd_logit_w100 / epoch 12 | `00f9a6ec6f660144` | 0.636719 | [Logit](results/split_seed42_geometric_20260910/logit_w100_test_report.json) |
| Rep, kd_rep_w0p01 / epoch 8 | `f717370ad937ad30` | 1.523438 | [Rep](results/split_seed42_geometric_20260910/rep_w0p01_test_report.json) |
| Gram, kd_gram_w100 / epoch 8 | `47fb1066898c95b8` | 0.863281 | [Gram](results/split_seed42_geometric_20260910/gram_w100_test_report.json) |
| NEW, trial_009 / epoch 10 | `f3aa183d4a422f24` | -0.640625 | [NEW](results/split_seed42_geometric_20260910/new_trial009_test_report.json) |

H20 sources are under `outputs/split_seed42_geometric_20260910/`: `b0_v31/`,
`struct_hpo/{grand/trial_006,new/trial_009}/`, `kd_hpo/{rank,rank_rep,grid}/`,
and `teacher_pma1/`. The teacher checkpoint remains
`outputs/split_seed42/teacher_pma1/best.pt`. Base configs and teacher banks are in
`configs/split_seed42/` and `outputs/distill/split_seed42/`; resumed trial configs
remain alongside their outputs. See [the HPC runbook](../hpc/README.md) for execution.

## 5. External CAZI-MBN baseline (protocol adaptation, 2026-09-12)

`configs/cazi_mbn_breadth_first.yaml` now uses the same seed-42 node holdout,
dynamic 1:5 supervised negative stream, positive BCE weight 5, V_val five-metric
checkpoint selection and frozen closest-geometric-RD test threshold. Validation
classification and early stopping use val_cls; topology uses the sampled pair union.
The endpoint classifier averages both orders. Frozen pooled FP32 features and
train-only normalization remain CAZI-specific, unlike the token B0 backbone.

The teacher's consensus table is defined only for training nodes. Its convergence
and checkpoint use training total loss, with no V_val teacher targets; only the
feature-only student is a deployable, V_val-selected model. Adam/StepLR and the
original 10,000-epoch caps/patience 30 are retained; student topology cadence is 10,
plus the first/final epochs. This is a benchmark-adapted CAZI run, not an unchanged
upstream reproduction. Publication alone is not a completed test; no new result
is claimed here. Execution details are in `hpc/README.md`.

### 5.1 Official TUnA / PPITrans feature-controlled baselines

The author implementations are vendored and run through `src.train_official_ppi`
and `hpc/run.sh`; see the source provenance in `src/baselines/vendor/README.md`.
Both consume the same frozen 1536-dimensional token features as our task, not the
papers' original ESM-2/ProtT5 embeddings. No supervised upstream PPI checkpoints
or upstream train/test splits are used. This is a classifier comparison with fixed
features; it does not reproduce either paper's native-embedding result.

The current node-held-out split, dynamic 1:5 negative stream, positive weight 5,
val_cls BCE stopping, five-metric checkpoint selection and test threshold replay
are shared. Architecture and method-specific optimizer defaults are retained;
TUnA's GP precision is fitted only on training rows at each evaluated network state,
and its mean-field logits are frozen consistently for validation and test. The
execution runbook records budgets and cadence. No completed results are claimed.
