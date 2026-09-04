# KD Double Arm (`kd_rank_rep`) + Strict Optuna HPO — Design

One new B1 arm combines the strongest logit-level KD (strict-LLP `kd_rank`: anchor ranking +
per-anchor distribution KL over context banks) with the strongest representation-level KD
(`kd_rep`: per-row cosine of the student pair representation to the teacher pooled vector
`t_uv`). A second Optuna study, run the way the kd_rank strict-LLP sweep runs, searches the
three loss weights while inheriting the kd_rank winner's context bank and margin. Winner
selection stays the frozen five-metric undominated rule plus the human pick; the study front is
advisory.

Stated concern, not a blocker: the 2026-09-02 audit found cosine losses are variance-weighted
onto the teacher-logit axis of `t_uv`. If `w_rep` earns no weight on the V_val surface, that is
the arm's result.

## Loss

Per optimizer step, on top of the protocol label-smoothed task BCE:

    w_rank * L_rank + w_dist * L_dist          (KD-only context forwards, `KDContextStream`,
                                               count-scaled exactly as kd_rank)
    + w_rep * mean_rows (1 - cos(pair_repr, t_uv))   (task batch's official rows, shared forward,
                                               `KDRowBank`, DDP-mean-scaled as kd_rep)

No new forward and no new artifact: the row bank (`kd_row_targets_v1`) supplies `teacher_rep`;
the context bank (`kd_ctx_targets_v1`) supplies context logits. Rep KD never touches context
pairs, which carry no teacher vector. The task input stays exactly `(x_u, x_v)`.

## Trainer changes

- `src/distill/config.py`: one new legal pattern `{w_rank, w_dist, w_rep}` (all three > 0) →
  `arm == "kd_rank_rep"`; `targets_path` required as for every teacher arm;
  `context_targets_path` required exactly when `w_rank` is active (kd_rank or kd_rank_rep).
- `src/train_b0.py`: `_REP_COS_ARMS = frozenset({"kd_rep", "kd_rank_rep"})` replaces every
  `arm == "kd_rep"` gate (train/val `teacher_rep` staging, the cosine loss term, `kd_rep_cos`
  train telemetry, `val_kd_rep_cos`/`val_kd_rep_loss` val diagnostics). `KDContextStream` and its
  construction in `main` gate on `distill.w_rank > 0` instead of `arm == "kd_rank"`. The
  `kd_rep_head` width check already keys on `w_rep` and is untouched. The gradient-norm probe,
  `_scale_kd_loss`, and `epoch_telemetry` need no change: the row bank returns the rep term and the
  context stream returns its own count-scaled sum, as today.
- `src/eval/early_stopping.py`: `_VAL_TERMS["kd_rank_rep"] = (("w_rank", "val_kd_rank_loss"),
  ("w_dist", "val_kd_dist_loss"), ("w_rep", "val_kd_rep_loss"))`, so patience counts all three at
  their training weights.
- Every existing arm stays bit-identical; the matched-control invariant is unchanged.

## Optuna driver

Parametrize the three hard-coded pieces of `src/experiments/kd_rank_strict_hpo.py` — study name,
whitelisted param→`distill` mapping in `materialize_trial_config`, and the report's param
columns — and add `src/experiments/kd_rank_rep_hpo.py`, which imports the shared ask/tell loop,
`trial_outcome`, `reconcile_running`, `enqueue_priors`, and the H20 thread caps. Objectives
(GS ↑, geometric-mean MMD ↓), the `|log RD| <= 0.05` soft constraint, the multivariate
constrained TPE sampler (seed 0), `MAX_CONSECUTIVE_FAILURES = 3`, and the reconcile/re-enqueue
semantics are identical to the kd_rank sweep.

| item | value |
|---|---|
| study / storage | `kd_rank_rep_strict`, `sqlite:///outputs/b1_kd_rank_rep_hpo/optuna.db` |
| base config | `configs/autoresearch/kd_rank_rep.yaml` = kd_rank autoresearch base + `w_rep`, `eval.topology_every: 2` |
| per-trial run | `hpc/run.sh train <trial cfg> --skip-test`, output `outputs/b1_kd_rank_rep_hpo/trial_NNN` |
| varied keys | `distill.w_rank`, `distill.w_dist`, `distill.w_rep`, `distill.margin`, `distill.context_targets_path`, `output_dir` |
| frozen | bank + margin (driver flags `--bank`, `--margin`, defaults `h2ns3` / 0.1 = Trial 8), LR, schedule, temperature 1, row `targets_path`, seed 0, 25 epochs |
| budget | 12 completed trials: 4 enqueued priors + 8 guided (`n_startup_trials = 4`) |

The driver dumps no banks: the bank named by `--bank` must already exist (the kd_rank sweep
dumped all four); a missing manifest fails closed before any training budget is spent.

## Search space and priors

| param | space |
|---|---|
| `w_rank` | log-float [0.01, 1] |
| `w_dist` | log-float [0.1, 100] |
| `w_rep` | log-float [0.01, 100] |

| # | w_rank | w_dist | w_rep | role |
|---|--:|--:|--:|---|
| 1 | 0.1 | 10 | 0.1 | kd_rank Trial-8 weights + kd_rep Phase-0 incumbent |
| 2 | 0.1 | 10 | 1 | rep one decade up |
| 3 | 0.1 | 10 | 10 | rep dominant |
| 4 | 1 | 1 | 1 | flat weights |

If the kd_rank sweep finishes with a different bank/margin winner, relaunch with the flags; the
study keeps completed trials (`load_if_exists`). If the front piles against a box edge, widen and
relaunch.

## Winner and test

Frozen five-metric undominated verdict over completed trials plus human pick; the winner's
checkpoint runs the held-out protocol once via `hpc/run.sh test` on its trial directory. Compare
against `kd_control` and the matched-epoch controls, never the published B0 (seed confound).

## Testing (CPU-only)

- `DistillConfig`: the new pattern is legal and names `kd_rank_rep`; `{w_rank, w_rep}` without
  `w_dist` and `{w_logit, w_rep}` stay rejected; `context_targets_path` required/forbidden rules.
- `KDRowBank` with a `kd_rank_rep` config stages `teacher_rep`, emits the weighted cosine term and
  `kd_rep_cos` telemetry, and still produces the kd_rank logit-correlation telemetry.
- `KDContextStream` accepts a `kd_rank_rep` config and rejects one without `w_rank`.
- `_evaluate_distributed` reports `val_kd_rep_cos`/`val_kd_rep_loss` and the context-bank
  rank/dist diagnostics in one outcome; `compose_val_total` sums the three weighted terms and
  raises on a missing one.
- Driver: trial-config materialization touches only the six whitelisted keys and validates
  through `DistillConfig.from_mapping`; priors enqueue idempotently; unknown `--bank` fails.
- Existing kd_rank and kd_rep tests pass unchanged.

## Documentation

- `docs/03-experiments.md`: a §1.4 row for `kd_rank_rep` (signal, teacher, searched
  hyperparameters, role "joint logit + representation transfer") and one §1.5 sentence naming
  the 12-trial study; results land in §2/§4 only after the held-out test.
- `src/distill/config.py` module docstring lists the new arm group.
- Hand-off README under `docs/results/kd_rank_rep_hpo/` in the kd_rank sweep's format.

## Out of scope

- Any other pairing (kd_logit + kd_rep, kd_rank + kd_gram); arbitrary weight compositions.
- Re-searching bank, margin, LR, temperature, epochs, seeds.
- Rep KD on context pairs; a new teacher dump; changes to the judge, ledger, or program.
