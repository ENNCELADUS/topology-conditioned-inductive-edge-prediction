# KD loss-weight HPO grid (Phase 0): results and campaign plan

**Status (verified 2026-08-31):** all 24 grid runs under `outputs/b1_row_kd_hpo/` completed on the
4-GPU H20 container with no `failure.json`. Design and literature grounding: `docs/tmp/kd_hpo_grid.md`;
harness spec: `docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md`. All numbers here are
**V_val selection surfaces** (AUPRC plus the five topology metrics at each run's selected-epoch V_val
threshold); no grid run has held-out test results — winners are tested once, at campaign finalization.

## 1. Protocol

Every point used the V3.1 student and full-row `kd_row_targets_v1` bank: 25 epochs, seed 0, 4 H20
ranks, bf16, `--skip-test`. The nine `kd_rank*` rows are the retired incidental-batch KD2 objective;
they are historical and non-comparable to the new strict-LLP KD2 family. Of 24 points, 22 validated
topology every epoch; `kd_rep_w10`/`kd_rep_w100` used `eval.topology_every: 2`. Every surface is the
**cadence-2 reselection** (`src.autoresearch.metrics_io.read_run(..., topology_every=2)`). The grid is
also not comparable to the published 2-rank KD1–KD4 reports (`../b1_kd_arms.md`).

## 2. Control anchor

Per operator decision (2026-08-31), the zero-KD control is the **published B0 V3.1 run** (checkpoint
`e092537d8cf1e208`); the never-launched `kd_control` sweep point is retired. B0 predates per-epoch
topology validation, so it has no judge-comparable V_val surface; it anchors at the protocol level
via `outputs/deployable_topology_v4/b0_v31_e092537d/test_report.json`: test AUPRC/AUROC
0.7315/0.7067, ECE/Brier 0.3396/0.3520; fixed V_val threshold (logit 2.125): GS 0.3675, RD 0.3236,
degree/clustering/spectral MMD ratios 20.92/18.38/28.25. Caveats: different rank count and training
vintage — mechanism-isolation claims against this anchor carry both confounds.

## 3. Cadence-2 historical surfaces (selected epoch; AUPRC telemetry, never optimized)

| run | ep | AUPRC | GS ↑ | RD → 1 | deg MMD ↓ | clu MMD ↓ | spec MMD ↓ |
|---|--:|--:|--:|--:|--:|--:|--:|
| kd_logit_w0p01 | 12 | 0.9180 | 0.5266 | 1.0098 | 14.50 | 2.68 | 11.58 |
| kd_logit_w0p1 | 18 | 0.9165 | 0.5089 | 1.0645 | 13.47 | 2.49 | 10.13 |
| kd_logit_w1 | 24 | 0.9273 | 0.5411 | 1.0379 | 15.57 | 2.92 | 11.32 |
| kd_logit_w10 | 22 | 0.9272 | 0.5455 | **0.9963** | 16.05 | 2.90 | 11.83 |
| **kd_logit_w100** | 25 | 0.9274 | **0.5459** | 1.0318 | 15.96 | 2.58 | 11.65 |
| kd_rank_w0p01 | 14 | 0.9181 | 0.5000 | 1.0116 | 13.19 | **2.23** | 10.40 |
| kd_rank_w0p1 | 12 | 0.9184 | 0.5112 | 1.0532 | 14.98 | 2.31 | 10.93 |
| kd_rank_w1 | 14 | 0.9192 | 0.5374 | 1.0684 | 13.67 | 2.69 | 10.11 |
| kd_rank_w10 | 22 | 0.9171 | 0.4935 | 1.0902 | 13.01 | 3.61 | 9.51 |
| kd_rank_w100 | 16 | 0.9144 | 0.4726 | 1.0941 | **12.46** | 3.46 | **9.38** |
| kd_rank_wr0p01_wd10 | 22 | 0.9139 | 0.5167 | 1.0758 | 13.06 | 3.54 | 9.42 |
| **kd_rank_wr0p1_wd1** | 22 | 0.9205 | 0.5263 | 1.0490 | 13.95 | 2.50 | 10.09 |
| kd_rank_wr0p1_wd10 | 18 | 0.9146 | 0.5128 | 1.0736 | 12.85 | 3.48 | 9.42 |
| kd_rank_wr1_wd0p1 | 14 | 0.9189 | 0.5196 | 1.0524 | 14.42 | 2.52 | 10.73 |
| kd_gram_w0p01 | 14 | 0.9170 | 0.5042 | 1.0532 | 15.77 | 2.65 | 11.31 |
| kd_gram_w0p1 | 14 | 0.9172 | 0.5099 | 1.0403 | 14.29 | 2.47 | 10.84 |
| **kd_gram_w1** | 10 | 0.9180 | 0.5311 | 1.0484 | 13.47 | 2.55 | 11.27 |
| kd_gram_w10 | 14 | 0.9145 | 0.5125 | 1.0225 | 14.43 | 3.28 | 10.64 |
| kd_gram_w100 | 18 | 0.9113 | 0.5154 | 1.0836 | 14.86 | 3.94 | 10.28 |
| kd_rep_w0p01 | 18 | 0.9177 | 0.5106 | 1.0630 | 14.18 | 2.49 | 10.38 |
| **kd_rep_w0p1** | 14 | 0.9158 | 0.5048 | 1.0405 | 12.98 | 2.47 | 10.04 |
| kd_rep_w1 | 16 | 0.9137 | 0.4963 | 1.0323 | 13.31 | 2.69 | 10.17 |
| kd_rep_w10 | 18 | 0.9144 | 0.5170 | 1.0430 | 14.94 | 3.47 | 10.73 |
| kd_rep_w100 | 18 | 0.9099 | 0.5153 | 1.0402 | 15.26 | 2.96 | 10.76 |

## 4. Winner selection

Pareto-undominated sets on the five oriented topology metrics (`src.autoresearch.verdict.undominated`):
kd_logit 5/5, kd_rank 7/9 (dropping `w10`, `wr1_wd0p1`), kd_gram 4/5 (dropping `w0p01`), kd_rep 5/5.
The metrics trade off, so the human pick among survivors (bold rows) used a select_checkpoint-style
mean rank of AUPRC plus the five oriented metrics across each arm's points. Findings: weight
sensitivity is term-specific, matching the literature extraction — `kd_logit` trades ~+0.04 GS and
best AUPRC/RD for ~+2 degree MMD as w→100; `kd_rank`'s heavy weights buy the grid's best degree and
spectral MMD at GS/clustering cost, with the LLP-style dist-heavy `wr0p1_wd1` most balanced;
`kd_gram` peaks at w=1; `kd_rep` at w=0.1.

## 5. Autoresearch campaigns

Protocol: `autoresearch/program.md` (operator loop, judge-only keep/revert on the five topology
metrics, append-only `autoresearch/ledger.jsonl`). State as of 2026-08-31: baselines recorded as
ledger trials 1–2; campaign surfaces materialized at `configs/autoresearch/kd_logit.yaml`
(commit `beac042f4f95`) and `configs/autoresearch/kd_rank.yaml` (commit `4424d5038459`).

- **kd_logit** (first), then **kd_rank**: launch per campaign once GPUs free —
  `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 bash hpc/run.sh train configs/autoresearch/<arm>.yaml --skip-test`;
  judge with `--incumbent-topology-every 2` while a grid winner is incumbent.
- **kd_gram / kd_rep**: gated on the PMA(1) Stage-0 teacher
  (`outputs/egostitch_e2e_stage1_v3/full_ego_teacher_pma1`, diagnostic-complete 2026-08-31, 10.1 h).
  Open question to settle first: does the single-PMA-seed teacher beat the 4-seed GMT teacher for
  representation-family KD? Plan: (a) teacher parity check vs `full_ego_teacher_kd`; (b) if it holds,
  dump a PMA1 target bank; (c) rerun each arm's grid winner (`kd_gram_w1`, `kd_rep_w0p1`) on the new
  bank and compare surfaces — the winning bank becomes that campaign's frozen `distill.targets_path`
  before its baseline row is recorded.
