# KD Autoresearch HPO — Design

Karpathy-autoresearch-style optimization loop for the B1 KD arms. The human optimizes the
research organization code (`autoresearch/program.md`); a local Claude Code operator session
executes one frozen-protocol trial at a time on the H20 container; a frozen judge decides
keep/revert; an append-only ledger records every proposal. Patterns adopted from
karpathy/autoresearch (three-layer split: frozen judge / one modifiable surface / human-owned
program), pi-autoresearch (replay-tolerant JSONL ledger, deterministic cold-start summary,
refuse-keep enforcement), AIDE (debug-depth cap), and the Cerebras retrospective
(single-experiment discipline, frequent human checkpoints, frozen measurable objective).

## Frozen decisions

- **Objective (topology-first):** the five V_val topology numbers at the run's selected epoch —
  BFS-macro GS, RD, degree/clustering/spectral MMD ratios. AUPRC is logged every trial and never
  enforced or optimized.
- **Verdict:** strict no-regression, computed only by frozen code. Keep iff at least one topology
  metric strictly improves versus the incumbent and none degrades. Zero-width tolerance bands by
  default (fixed-seed runs; no replicate measurement). An optional `autoresearch/bands.json` can
  widen per-metric tolerances later — written only by the human, never by the agent.
- **Trial:** one full grid-protocol run — 25 epochs, seed 0, `--skip-test`, topology-aware
  selection every epoch — sequential, auto world size, one container; grid winners = incumbents.
- **Substrate:** local Claude Code operator via git + ssh. All loop state lives in git plus the
  ledger; any fresh session cold-starts from files alone.
- **Scope:** four campaigns — kd_logit, kd_rank, kd_gram, kd_rep — prioritized by grid results.
  kd_d9 is out until it has a grid of its own.
- **Budget:** no pre-set hard cap. The operator raises a stall advisory after 5 consecutive
  non-keeps and waits for human re-steer; only the human ends a campaign.
- **Test protocol:** forbidden inside the loop. Finalization is human-triggered, once per campaign.

## Phases

- **Phase 0 — grid + control (no agent modification of code):** run the 24-point grid via
  `hpc/sweep_kd_hpo.sh all`; add the matched control as a 25th sweep point; identify per-arm
  incumbents with the frozen judge's own metric reader.
- **Phase 1 — toolkit:** build and test `src/autoresearch/`, write `autoresearch/program.md`.
- **Phase 2 — campaigns:** one arm at a time, protocol below.

## Layout

State, committed to git:

- `autoresearch/program.md` — human-owned research organization code: objective, protocol,
  stage-1 key whitelist, named agent duties (per-trial fit diagnosis), campaign order, stall
  rule, contract clauses (below). The agent never edits it; program changes go via `ideas.md`.
- `autoresearch/ledger.jsonl` — append-only, one JSON object per trial across all campaigns.
- `autoresearch/bands.json` — optional per-metric tolerances; absent means zero-width.
- `autoresearch/ideas.md` — agent-maintained backlog of deferred hypotheses; survives reverts.

Code, `src/autoresearch/` (mypy strict, ruff, tested):

- `metrics_io.py` — read one run dir: `run_metadata.json` → `selected_epoch`, then that row of
  `metrics.jsonl` → the six-tuple (`val_auprc`, `val_gs_bfs`, `val_rd_bfs`, three
  `val_*_mmd_ratio`) plus `val_threshold`, `total_seconds` from `complete.json`. Raises on
  non-finite values, missing rows, or a present `failure.json`.
- `verdict.py` + `python -m src.autoresearch.judge` — incumbent dir, trial dir, optional bands →
  verdict JSON: per-metric deltas, `keep` / `revert`, reasons. Improvement directions: GS higher,
  |log RD| lower, MMD ratios lower. AUPRC appears in the output as telemetry only.
- `ledger.py` + append CLI — validates schema, global trial-number monotonicity, unique
  `output_dir` and commit per row, and refuses a `keep` row whose embedded verdict says revert.
  Replay-tolerant reader (skip unparseable lines, recompute derived state).
- `summary.py` — deterministic cold-start digest: campaign standings, incumbent metrics, last N
  ledger rows one line each (hypothesis → status → deltas), open ideas. No LLM content.

Trial surface: `configs/autoresearch/<arm>.yaml`, one file per campaign, edited in place; each
trial bumps `output_dir` to `outputs/b1_row_kd_ar/<arm>/trial_NNN`. The test-pinned
`configs/sweep/b1_kd_hpo/` directory is never touched during campaigns.

## Ledger row

`trial` (global int), `campaign`, `commit` (post-commit sha), `config_hash`, `output_dir`,
`hypothesis` (one sentence), `status` ∈ {`baseline`, `keep`, `revert`, `crash`}, six metrics,
`selected_epoch`, `total_seconds`, `verdict` (deltas + reasons; null for baseline/crash),
`asi` (free-form diagnostics worth remembering), `timestamp`. Each campaign opens with a
`baseline` row for its grid-winner incumbent.

## Trial protocol (one hypothesis per trial)

1. Cold start: read `program.md`, run `summary`.
2. Propose one hypothesis; edit only whitelisted keys in `configs/autoresearch/<arm>.yaml`;
   bump `output_dir`.
3. Commit `ar(<arm>) trial NNN: <hypothesis>` before launch; push; pull on the container.
4. Launch over ssh: `hpc/run.sh train configs/autoresearch/<arm>.yaml --skip-test` with the sweep
   script's thread caps (`OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`).
5. Poll `complete.json` / `failure.json`; on completion pull back the small JSON artifacts.
6. Run `judge`; diagnose fit from the full per-epoch curve (train vs val task loss, KD terms,
   grad norms, topology trajectory, selected epoch vs loss minimum) — verdict overfit /
   underfit / healthy, written into `asi`; append the ledger row; commit the ledger.
7. Keep → incumbent becomes this trial's run dir. Revert → `git revert` the trial commit
   (history preserved; every proposal stays countable).
8. Crash (`failure.json` or judge gate): one config-level fix commit maximum, relaunched as the
   same trial (the row records the final commit, both shas in `asi`); a second failure logs `crash`.
9. Post a one-paragraph digest to the human after every trial; stall advisory after 5 consecutive
   non-keeps, then wait for re-steer.

## Modifiable surface

- **Stage 1 (config only):** `distill.*` (weights, `margin`, `temperature`), arm-specific KD
  model keys (`model.config.kd_rep_dim`), all `optim.*` except `epochs`/`patience`, and model
  regularization keys (e.g. dropout) — so a fit diagnosis can be acted on directly. Frozen:
  `seed`, `epochs`, `patience`, data, eval, `targets_path`, model family — the budget and the
  measurement, never the recipe.
- **Stage 2 (after plateau, human sign-off):** widen to that arm's loss implementation in
  `src/distill/losses.py` (and its wiring), same budget and verdict, normal review gates
  (codex review per wave).

## Guardrails (contract clauses in program.md)

- The verdict comes only from `judge`; the agent never argues a trial into a keep.
- V_val only; test files, `test_report.json`, and the test protocol are out of contract in-loop.
- Only the frozen six enter the verdict; other telemetry may inform proposals, never keeps.
- One hypothesis per trial; no compound edits; no editing `src/autoresearch/`, `program.md`,
  `bands.json`, the sweep configs, or anything outside the whitelist during campaigns.
- Every trial — including crashes and reverts — gets a ledger row; the ledger is append-only.
- No pausing to ask whether to continue; stop only at stall advisories and human interrupts.

## Phase 0 details

- Control config: copy of `configs/b1_kd_control_breadth_first.yaml` with
  `eval.classification_only` removed and `output_dir: outputs/b1_row_kd_hpo/kd_control`; extend
  `EXPECTED_SWEEPS` in `tests/test_sweep_configs.py` and give the per-stem test a control branch
  (no `distill:` section to compare).
- Incumbent identification: for each arm, take the Pareto-undominated grid points on the five
  topology metrics (GS↑, |log RD|↓, ratios↓); a sole survivor is the incumbent, otherwise the
  human picks among survivors; the choice lands in the ledger's baseline row.

## Finalization (per campaign, human-triggered)

Rerun the campaign winner without `--skip-test` for its single `test_report.json`
(`test_protocol_v6`), then generate the campaign report from the ledger: every proposal counted
(keeps, reverts, crashes), final config diff vs baseline, and both metric families reported
together per the claim rules — AUPRC included even though never optimized.

## Risks

- Zero-width bands assume fixed-seed runs are reproducible enough that any regression is real.
  If keeps/reverts thrash on tiny deltas, the human sets `bands.json`; the loop never measures
  or widens bands itself.
- The 4-rank grid is not comparable to the published 2-rank KD1–KD4 references; campaign claims
  compare only against the grid, the control, and other campaign trials.
- With `optim.*` open, campaign winners differ from the control by more than the KD term; the
  campaign report must present the full config diff vs baseline (the ledger captures it), and
  mechanism-isolation claims stay pinned to the grid + control, never to campaign winners.

## Testing

Toolkit is TDD'd: `metrics_io` against fixture run dirs (including failure/non-finite cases),
`verdict` against hand-built metric tables (all keep/revert branches, band widening), `ledger`
append validation + replay (torn final line, duplicate output_dir, contradicted keep),
`summary` golden output. No test touches the network or the container.
