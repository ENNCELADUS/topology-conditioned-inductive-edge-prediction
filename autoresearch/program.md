# KD Autoresearch Program

Human-owned research organization code. The operator follows this file and never edits it; proposed program changes go to `ideas.md`. Spec:
`docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md`.

## Objective

Improve the five V_val topology numbers of the active campaign's KD arm — BFS-macro GS (higher),
RD (closer to 1, judged as |log RD|), degree/clustering/spectral MMD ratios (lower) — measured at
each run's selected epoch. AUPRC is telemetry: always logged, never optimized, never a
keep/revert input.

## Campaigns

kd_logit, kd_rank, kd_gram, kd_rep — one at a time; order set by the human from grid results.
Each campaign starts from its arm's grid winner (human-recorded `baseline` ledger row, picked
from `src.autoresearch.verdict.undominated`). No campaign begins until Phase 0 is complete and the
human materializes that winner as `configs/autoresearch/<arm>.yaml` and records its baseline row; the operator never invents it.

## The trial loop

1. Cold start: read this file, then run
   `.venv/bin/python -m src.autoresearch.summary autoresearch/ledger.jsonl`.
2. Propose exactly one hypothesis, citing the previous trial's fit diagnosis, the ledger, and
   `ideas.md`.
3. Edit only whitelisted keys in `configs/autoresearch/<arm>.yaml`; bump `output_dir` to
   `outputs/b1_row_kd_ar/<arm>/trial_NNN`.
4. Commit `ar(<arm>) trial NNN: <hypothesis>` BEFORE launching; push; pull on the container.
5. Launch on the container with the sweep thread caps:
   `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 bash hpc/run.sh train configs/autoresearch/<arm>.yaml --skip-test`.
6. Poll `complete.json` / `failure.json`; pull back `metrics.jsonl`, `run_metadata.json`,
   `complete.json`, `profile.json` into a local mirror of the run directory.
7. Run `.venv/bin/python -m src.autoresearch.curves <run_dir> autoresearch/runs/<arm>/trial_NNN`.
8. Diagnose fit from the CSV and PNG — train vs val task loss, KD terms, grad norms, topology
   trajectory, selected epoch vs loss minimum. Verdict one of overfit / underfit / healthy with
   one sentence of evidence; this goes into the ledger row's `asi`.
9. Run `.venv/bin/python -m src.autoresearch.judge --incumbent <dir> --trial <dir>`
   (add `--bands autoresearch/bands.json` only if the human has created that file).
10. Append the ledger row via `.venv/bin/python -m src.autoresearch.ledger autoresearch/ledger.jsonl <row.json>`;
    commit the row, curve artifacts, and any changed `ideas.md` together (separate from the
    trial's config commit); every per-trial state commit includes changed `ideas.md`.
11. keep → the incumbent becomes this trial's run. revert → `git revert` the trial's config
    commit.
12. Crash (`failure.json`, or the judge raises): one repair commit maximum, then relaunch the same
    trial. On a second failure, log `crash`, record proposal/repair SHAs in `asi`, then revert repair and proposal commits in reverse order before the next trial.
13. Post a one-paragraph digest to the human; continue with the next trial.

## Stage-1 whitelist (config keys the agent may edit)

- `distill.*` (weights, `margin`, `temperature`)
- arm-specific KD model keys (`model.config.kd_rep_dim` for kd_rep)
- all `optim.*` EXCEPT `epochs`
- model regularization keys (`model.config.regularization.*`, `model.config.mlp_head.dropout`,
  `model.config.label_smoothing`)
- `output_dir` (mandatory bump every trial)

Frozen — never edit: `seed`, `optim.epochs`, `eval.*` (including `patience`), `data.*`,
`runtime.*`, `distill.targets_path`, `model.family`, `mixed_precision`.

## Named duties

- The per-trial fit diagnosis (step 8) is mandatory; every hypothesis must cite the latest one.
- Maintain `ideas.md`: deferred hypotheses go in; spent or rejected ones come out.

## Contract

- The verdict comes only from the judge CLI; never argue a trial into a keep.
- V_val only. Test files, `test_report.json`, and the test protocol are out of contract in-loop.
- Only the five frozen topology metrics decide keeps; all other telemetry informs proposals only.
- One hypothesis per trial; no compound edits.
- Never edit: `src/autoresearch/`, this file, `bands.json`, `configs/sweep/`, any frozen key.
- Every trial — keep, revert, crash — gets a ledger row; the ledger is append-only.
- After 5 consecutive non-keeps: post a stall advisory and wait for human re-steer.
- Do not pause to ask whether to continue; stop only at stall advisories and human interrupts.
