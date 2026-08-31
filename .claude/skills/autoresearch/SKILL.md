---
name: autoresearch
description: Use when the user invokes /autoresearch in this repository or asks to run, resume, judge, or report a KD campaign trial (kd_logit, kd_rank, kd_gram, kd_rep) — including "next trial", "campaign standings", keep/revert verdicts, stall advisories, or baseline/incumbent questions. This project skill replaces the generic global autoresearch skill here.
---

# Project autoresearch — KD campaign operator

This repo has its own frozen-protocol harness. The loop body is `autoresearch/program.md`
(human-owned — read it in full and follow its numbered trial steps exactly; NEVER edit it).
Design rationale: `docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md`.

The generic autoresearch dispatch does not apply here: no mode banner, no classic/orchestrator/
wizard classification, no success-predicate derivation, no confirmation question, no iteration
cap, no `autoresearch/<goal>-<timestamp>/` run dirs. `program.md` already defines the loop.

## Dispatch

| Invocation | Action |
|---|---|
| `/autoresearch <arm>` | Run that campaign's trial loop per `program.md` |
| `/autoresearch` bare or `status` | Cold start only: summary CLI, report standings, stop |

## Cold start (every session, before any proposal)

1. Read `autoresearch/program.md`.
2. `.venv/bin/python -m src.autoresearch.summary autoresearch/ledger.jsonl`

## Preconditions — stop and report if missing (human-only steps)

- `configs/autoresearch/<arm>.yaml` exists (materialized grid winner, `eval.topology_every: 2`).
- The ledger holds that arm's `baseline` row (from `python -m src.autoresearch.baseline
  <grid_run_dir> --topology-every 2`).

The operator never picks incumbents, records baselines, sets campaign order, or scaffolds the
missing config "to unblock". The incumbent is the ledger's baseline/last-keep run dir — never
`docs/results/kd*` (published KD1–KD4 are 2-rank runs, incomparable to the 4-rank grid).

## CLI quick reference

| Purpose | Command |
|---|---|
| Standings digest | `.venv/bin/python -m src.autoresearch.summary autoresearch/ledger.jsonl` |
| Curves + fit CSV | `.venv/bin/python -m src.autoresearch.curves <run_dir> autoresearch/runs/<arm>/trial_NNN` |
| Verdict | `.venv/bin/python -m src.autoresearch.judge --incumbent <dir> --trial <dir>` |
| Ledger append | `.venv/bin/python -m src.autoresearch.ledger autoresearch/ledger.jsonl <row.json>` |

Judge flags: `--incumbent-topology-every 2` only while the incumbent is an every-epoch Phase-0
grid run; `--bands autoresearch/bands.json` only if the human created that file.

## Launch

**REQUIRED SUB-SKILL:** hpc-execution (container, paths, runner). Trials edit
`configs/autoresearch/<arm>.yaml` only — `configs/sweep/b1_kd_hpo/` is test-pinned and never
touched. Commit `ar(<arm>) trial NNN: <hypothesis>`, push, pull on the container, then:
`OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 bash hpc/run.sh train configs/autoresearch/<arm>.yaml --skip-test`

## Rationalizations vs reality

| Excuse | Reality |
|---|---|
| "Free-form goal → orchestrator/classic mode" | `program.md` replaces every generic mode. |
| "The judge/program is silent, so default to mean-rank or AUPRC" | The judge CLI is never silent — run it. No fallback verdict exists; AUPRC never decides a keep. |
| "The deltas obviously improve; mark keep" | Only the judge CLI's `decision` field is a verdict. |
| "Config/baseline missing — I'll create it and continue" | Campaign not started. Stop and report. |
| "Verify with the test protocol to be sure" | Test files and `test_report.json` are out of contract in-loop; V_val only. |
| "Bundle two small edits into one trial" | One hypothesis per trial, whitelisted keys only. |

## Red flags — stop, you are off-protocol

- Keep/revert stated without a judge CLI run; a ledger row skipped for any trial.
- Editing anything under `configs/sweep/`, `src/autoresearch/`, `program.md`, `bands.json`.
- Launching before the trial commit is pushed and pulled on the container.
- Pausing to ask whether to continue (stop only at preconditions, stall advisory after 5
  consecutive non-keeps, or human interrupt).
