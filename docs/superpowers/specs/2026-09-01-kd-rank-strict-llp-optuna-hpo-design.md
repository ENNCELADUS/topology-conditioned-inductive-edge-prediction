# KD2 Strict-LLP Optuna HPO — Design

One Optuna study, run unattended on the H20 container, jointly searches the strict-LLP `kd_rank`
loss weights, context-bank composition, and margin. LLP-paper priors enter as search-box widths and
enqueued first trials, never as staged manual screening. The study only steers sampling: official
winner selection stays the frozen five-metric rule, and AUPRC stays telemetry.

## Frozen decisions

- **Trial protocol:** one grid-protocol run per trial — base `configs/autoresearch/kd_rank.yaml`,
  25 epochs, seed 0, bf16, auto world size, `--skip-test`, `eval.topology_every: 2`. Only
  `distill.w_rank`, `distill.w_dist`, `distill.margin`, `distill.context_targets_path`, and
  `output_dir` vary per trial. Frozen: LR and schedule, KD temperature (=1), `rw_step=3`, row
  `targets_path`, the dump teacher, and the single seed — no multi-seed replicates.
- **Objectives:** maximize BFS-macro GS; minimize the geometric mean of the three MMD ratios. Both
  come from the cadence-2 selected-epoch row via
  `src.autoresearch.metrics_io.read_run(run_dir, topology_every=2)`, the Phase-0 reselection rule.
- **Constraint:** RD soft feasibility `|log RD| <= 0.05` through the sampler's `constraints_func`
  (band is a driver flag; 0.05 splits the observed Phase-0 RD range 1.01–1.09).
- **Sampler:** `TPESampler(multivariate=True, seed=0, n_startup_trials=6, constraints_func=...)`
  with `directions=["maximize", "minimize"]`, sqlite RDB storage, ask-and-tell loop.
- **Budget:** 16 trials (6 enqueued priors + 10 TPE-guided) plus 3 context-bank dumps.
- **Final selection:** the feasible Pareto front (`study.best_trials`) is advisory. The recorded
  winner comes from the frozen five-metric undominated verdict over all completed trials plus the
  human pick, then becomes kd_rank's first active campaign baseline per `autoresearch/program.md`;
  the human records that ledger row, never the driver.

## Search space

| param | space | prior rationale (LLP paper) |
|---|---|---|
| `w_dist` | log-float [0.1, 100] | most sensitive axis; normalized optimum ~10, collapse near 1000 caps the box |
| context bank | categorical {h2ns1, h2ns3, h2ns5, h3ns3} | global context second-most sensitive; covers 6–9 local + 6–30 global (K = 12/24/36/36) |
| `w_rank` | log-float [0.01, 1] | flat across ~5 decades; narrow box |
| `margin` | categorical {0.05, 0.1, 0.2} | mechanically load-bearing, numerically tame; prior anchored at 0.1 |

If the front piles against a box edge, the human widens the box and relaunches; `load_if_exists`
keeps every completed trial.

## Enqueued prior trials

| # | w_rank | w_dist | bank | margin | role |
|---|--:|--:|---|--:|---|
| 1 | 1 | 1 | h2ns1 | 0.1 | required from-scratch strict baseline (current default) |
| 2 | 0.1 | 10 | h2ns1 | 0.1 | paper optimum on the current bank; isolates the weight effect |
| 3 | 0.1 | 10 | h2ns3 | 0.1 | headline bet: distribution-heavy weights + more global context |
| 4 | 0.01 | 10 | h2ns5 | 0.1 | headline bet, maximal global context |
| 5 | 0.1 | 100 | h2ns3 | 0.1 | top of the w_dist box, where more context may push the optimum |
| 6 | 0.1 | 10 | h3ns3 | 0.1 | RW-length (hops=3) probe |

## Context banks

`h2ns1` is the existing `outputs/distill/kd_ctx_targets_breadth_first`. The driver dumps the three
missing banks before the study starts, reusing the exact dump invocation and teacher checkpoint of
the existing artifact and varying only new `--rw-step/--hops/--ns-rate` flags on
`python -m src.distill.teacher_targets --contexts`; the flags land in the artifact's
`sampler_params`. Naming: `outputs/distill/kd_ctx_targets_breadth_first_{h2ns3,h2ns5,h3ns3}`.
Dump cost and kd_rank's KD-only forwards both scale linearly in K, so ns5/h3 trials run longer
than the ~1 h Phase-0 pace; expect ~1.5–2 days total on the 4-GPU container.

## Driver

`src/experiments/kd_rank_strict_hpo.py`, launched once in tmux on the container:

1. Dump any missing context bank (skip those already on disk).
2. Create-or-load the study (`kd_rank_strict_llp`,
   `sqlite:///outputs/b1_kd_rank_strict_hpo/optuna.db`); enqueue the prior trials with
   `skip_if_exists=True`.
3. Loop `ask()` → materialize `outputs/b1_kd_rank_strict_hpo/trial_NNN/config.yaml` from the base
   config (validated through `DistillConfig.from_mapping`) → subprocess
   `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 bash hpc/run.sh train <config> --skip-test` → read the
   surface → `tell()` objectives with the RD constraint in `user_attrs`, until 16 completed trials.
4. Print the feasible Pareto front and the full trial table (all six metrics per trial).

Trial configs live under `outputs/` (provenance via `run_metadata.json` + the study db + the final
results doc); code and the base config reach the container by git pull as always.

## Failure and resume semantics

- `failure.json` → the trial is told FAILED and the sweep continues; no auto-repair. Repeated
  FAILs in one region are a human signal, not a driver problem.
- Non-finite objective or constraint values from a *completed* run stop the sweep (fail-closed).
- On startup the driver reconciles stale RUNNING trials: run dir has `complete.json` → tell its
  real result; has `failure.json` → tell FAILED; neither → mark FAILED and let TPE re-propose.
- No sha/digest pinning or artifact-contract gates anywhere; matching banks to the current split
  is the operator's job.

## Testing (CPU-only)

- Trial-config materialization: legal `DistillConfig` arm pattern, only the five whitelisted keys
  differ from base, bank name → path mapping.
- Objective/constraint extraction from a fixture `metrics.jsonl` (geo-mean MMD, `|log RD|` band).
- Enqueue determinism and `skip_if_exists` idempotence; startup reconciliation of a fake
  interrupted run dir (all three cases).
- Dump-CLI flag threading into `sampler_params` (unit level; no GPU, no teacher).

## Out of scope

- LR, temperature, `rw_step`, epochs, seeds, and every non-`distill` config key.
- Changes to `autoresearch/program.md`, the judge, or the ledger — this sweep precedes the
  campaign and runs outside the operator contract.
- Multi-container parallelism; the sweep is sequential on one container.
- Any eligibility predicate on trials; usability is judged from surfaces, per repository rules.

## Hand-off

Publish a Phase-0-style results README under `docs/results/kd_rank_strict_hpo/` (protocol, full
table, Pareto front, winner rationale). The human records the winner as the kd_rank campaign
baseline and proceeds under `autoresearch/program.md`.
