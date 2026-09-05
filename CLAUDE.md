# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`AGENTS.md` is the Codex-facing copy; update it only when a constraint here changes.

## What this is

Research code for an ICLR 2027 paper, *Topology-Conditioned Inductive Edge Prediction*. `README.md`
orients; `docs/01-project-definition.md` and `docs/03-experiments.md` govern task and evaluation,
`docs/02-methodology.md` the open method-selection constraints. `docs/results/` holds per-arm result
notes (`b1_kd_arms.md` indexes the KD arms), `docs/superpowers/{specs,plans}` design records,
`docs/tmp/` working plans, `docs/iclr2027/` the LaTeX paper (`main.tex` + `sections/`, latexmk).
When a change alters behavior a doc describes, update that doc in the same change; otherwise leave docs alone.

**Core thesis — do not let it drift:** the strict task input is exactly `(x_u,x_v)` and the output
is the binary decision for `edge(u,v)`. Inferred topology is intermediate context, not graph
generation; grounding/retrieval/prototypes are optional arm-specific support, never task input or a
selected method. Every piece of writing must explain how its context helps decide the queried edge.

**Active method set (2026-09):** one teacher, four KD students, one control.
- Teacher: the Full-Ego Oracle (`model.family: egostitch_e2e`, generator `full_ego_oracle`, encoder
  `grit_gmt`). It reads true training structure, so it is a ceiling and a bank source, never a
  deployable arm.
- Students (`model.family: v3_1`, endpoint-only): `kd_logit` (GLNN soft logits), `kd_rank`
  (strict-LLP rank + distribution matching over context banks), `kd_gram` (SPKD cosine-Gram),
  `kd_rep` (per-row representation cosine); `kd_rank_rep` is the joint variant. Control: `b1_kd_control`.
- Retired, history only: the EgoStitch imagination arm (`egostitch_imagine`, G5 screens), the S-series
  (`docs/results/s_series.md`), `kd_struct`, `kd_white`, `kd_gen`, and the D1–D8 anchor-context arms.
  Do not revive them or compare new results against them.

## Project rules beyond the global ones

- No formalism gates: never add digest pinning, artifact- or text-contract verifiers,
  eligibility/promotion ceremony, or any check that blocks a run. Record provenance and proceed.
- Sync code between machines with git only (commit → push → pull on the H20 checkout); never
  rsync/scp/tar working trees.
- GPU work needs the H20 container. When a task reaches that point, finish everything local first
  (config, tests, commit, push) and end with the exact launch command; do not wait on a GPU you cannot reach.
- One independent Codex review per implementation wave that changes `src/` (none for docs or config
  edits). Run the recipe under Commands yourself, wait for it, fix the blockers, finish. The user may
  instead invoke `/codex:review` or `/codex:adversarial-review`. No second round without new changes.
- `autoresearch/program.md` is human-owned. KD campaign trials (`/autoresearch`, the project skill)
  follow it exactly and never edit it, `src/autoresearch/`, `configs/sweep/`, or its frozen keys.
- `src/vendor/` is the official GRIT and Set Transformer code; do not refactor or lint-fix it.

## Commands

Local (macOS, CPU only): use `.venv/bin/python -m …` — `rtk` garbles `uv run` output.

```bash
.venv/bin/python -m pytest                                          # full; add -m "not slow and not integration" for fast local
.venv/bin/python -m pytest tests/test_score_universe.py -n0 -k density   # one file / one test
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests                                  # strict, tests included
uv sync                                                             # dependencies
```

Use `-n0` when debugging — xdist swallows breakpoints. Tests in one file share state, so
`--dist loadfile` is required, never `--dist load`. Real-artifact tests read `TCIEP_DATA_ROOT`
(default `data/`) and skip when it is absent. Ruff enforces Google docstrings and full annotations
in `src/` and bans `print`.

GPU work runs only in the H20 container via `hpc/run.sh`
(`check | train | score | test | merge | g1 | g2 | kd-targets`); world size and score shards are
auto-detected from visible GPUs, and there is no scheduler. `train` runs pack → train → publish →
held-out test unless `--max-steps` (debug) or `--skip-test` (sweep points) is passed; `test` also runs
directly against a published checkpoint. Config keys change meaning per `model.family`; direct
`python -m src.train_b0` is debug-only. Runbook: `hpc/README.md` and the `hpc-execution` skill.

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_ego_teacher_pma1_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind diagnostic       # teacher (reads true structure)
hpc/run.sh kd-targets --config <student.yaml> --checkpoint <teacher best.pt> --output outputs/distill/<row bank>
hpc/run.sh kd-targets --contexts --config <student.yaml> --checkpoint <teacher best.pt> \
  --output outputs/distill/<ctx bank> --rw-step N --hops H --ns-rate R   # context bank (kd_rank family)
hpc/run.sh train configs/b1_kd_logit_breadth_first.yaml             # a KD student; distill.* keys name the banks
hpc/run.sh train configs/b0_v31_breadth_first.yaml                  # B0 baseline
```
KD sweeps go through `hpc/sweep_kd_hpo.sh` or the Optuna drivers `src/experiments/kd_rank_*_hpo.py`
(which dump missing banks), never a hand-launched grid. When more than one torch job shares the
box, export `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` as the sweep script does, or the jobs spin-wait.

Codex review recipe: `CODEX_HOME=<scratch>/codex-home codex review --base <sha> > wave-review.txt 2>&1`,
backgrounded — ~200 KB of output must never land in context; read the file when it finishes. That
home needs only `auth.json` and a `config.toml` pinning model/effort; `-c 'mcp_servers={}'` merges
rather than overrides, leaving the MCP tables alive, which hangs the review with zero output.

## Architecture

Teacher flow: benchmark artifacts → partition → packed features → DDP teacher training
(`train_egostitch`) → row and context banks under `outputs/distill/` (`src/distill/teacher_targets.py`,
`context_sampler.py`). Student flow: packed features → `train_b0` with `distill.*` losses → publish →
`test_protocol`. Students never see graph structure; only the banks carry it.

- `src/data/` — `artifacts.py` verified benchmark loader; `partition.py:build_g_struct` is the only
  legal structural graph; `val_region.py` derives V_val; `packed_features.py`/`features.py` bf16 pack
  and F0 cache; `pairs.py` batching; `grounding.py` is arm-specific support.
- `src/model/egostitch/` — the teacher's model tree: generator, encoder, and classifier slots behind
  `registry.py`, composed by `composite.py`, talking through `graph.py` dataclasses. Each slot is
  chosen by the `name` field of its config section (generators `full_ego_oracle`, `full_ego_features`,
  `oracle_struct`, `null`; encoders `grit_gmt`, `ste_typed`; classifier `b0_v31`), so a new component
  is one registry entry, never a `composite.py` change. No encoder may read `ImaginedGraph.aux`: it is
  generator-private and a generator swap invalidates it.
- `src/distill/` — `config.py` legal KD weight patterns per arm, `losses.py` the KD terms,
  `artifacts.py` the strict bank loaders. `train_b0` adds KD-only forwards for the context stream
  (`kd_rank`, `kd_rank_rep`) under the token budget with activation checkpointing.
- `src/train_{b0,egostitch,cazi_mbn}.py` DDP workers (`cazi_mbn` + `src/baselines/` is the isolated
  external baseline); `src/e2_pipeline.py` pack → train → publish → test; `src/score_universe.py` and
  `src/score_fanout.py` score-once artifacts and their multi-GPU fan-out; `src/eval/` metrics,
  assembly, calibration, checkpoint selection, and `test_protocol.py`; `src/experiments/` KD audits,
  Optuna drivers, and the legacy G1–G3 gates; `src/autoresearch/` the KD-campaign judge, ledger, curves.

Experiments run directly; there is no plan, registration, or qualification gate. Model-quality
signals (liveness, slot collapse, margins, dispersion) are telemetry in `profile.json`/`metrics.jsonl`
and never block a run; non-finite state, DDP disagreement, data-boundary violations, and I/O failures
stay fail-closed. `complete.json` means published, not evaluated: held-out evidence is
`test_report.json`/`test_complete.json`, or the `diagnostic_*` pair for true-structure runs.

## Claim rules

- Report edge-level and assembled-graph metrics together — never one family alone.
- Each topology operating point reports five numbers together: BFS-macro GS/RD and degree /
  clustering / spectral MMD ratios (GS↑, RD→1, ratios↓). The primary deployable result is the ONE
  V_val-selected fixed threshold (density-first cascade) on every test subgraph; Accuracy/F1/MCC use
  a separate max-F1 threshold frozen on `val_cls`; ECE/Brier use raw probabilities. GS is edge-set Dice/F1.
- Compare KD arms against `b1_kd_control` and check the selected epoch before crediting a KD term:
  matched-epoch control runs (epochs 10/11) erased the earlier `kd_struct`/`kd_white` gains. The
  teacher is a ceiling row, never a comparator arm.
- Baseline and benchmark names are deliberately neutral (`Benchmark-A/B/C`, `B0`, `B1`, …, `Oracle`);
  never substitute real dataset names unless asked. `Ours` is unassigned until method selection.

## Data-contract traps

- `*_ratio5_exclusive.txt` is quarantined: its negatives leak across the node split; no loader reads it.
- `train_graph.pkl` (train⁺∪val⁺) is the V_val substrate; `train_edges.txt`/`val_edges.txt` are retired.
- `exclude_nodes` filters only the pair lists, so featureless nodes survive in every graph and bucket.
- Self-loops: training structural targets strip them; canonical MMD descriptors and official GS/RD
  subgraphs keep them, as the benchmark evaluator does.
- Grounding, when used, is universe-scoped (`train`, `V_val`, test): no cache crosses universes;
  training may not read V_val-internal pairs (cross-boundary edges do train).
- Topology and classification share the same train positives (no message/supervision split):
  loopless projection for topology, self-pairs kept for classification, and edge-stream structural
  targets must drop the queried partner and decrement its degree.
- `V_val` (`val_region.py`): K=5 dispersed-seed hashed-frontier BFS on `train_graph.pkl`'s loopless
  giant component, stopped at 20% induced loopless edges — pair-disjoint only, never "fully
  inductive". It invalidated every V_hold-keyed cache, pack, threshold, and result.
- KD banks are keyed to the teacher checkpoint and the split: a re-trained teacher or a moved V_val
  boundary invalidates `outputs/distill/*` wholesale. `src/distill/artifacts.py` loads only the
  `kd_row_targets_v1` / `kd_ctx_targets_v1` formats and raises on row-coverage mismatch.

## Traps that corrupt results silently instead of raising

- `load_scores` does no precision validation, so a bf16-contaminated artifact analyses cleanly. Call
  `validate_artifact_precision(artifact, label=…)`; `validate_score_precision` on an `egostitch_e2e`
  artifact spuriously raises "missing arrays".
- Checkpoint selection (`src/eval/checkpoint_selection.py`) mean-ranks AUPRC plus the five V_val
  topology metrics and can still return a weak checkpoint. There is no eligibility predicate and none
  should be added — judge usability from `metrics.jsonl`.
- The fp32 islands in `generator/assemble.py` must promote inputs *before* cost and marginal products
  are formed; casting afterwards keeps the bf16 ulp grid and silently quantizes logits.
- `allow_cache_subset=True` (live in `score_universe.py`) gathers a superset F0 cache into a different
  node set with no content check. Exact-order mismatches raise; this path stays quiet.
- Packed-feature manifests depend on `index.json` insertion order; sorting or reserializing it
  invalidates the pack. Its F0 cache holds fp32 means taken before bf16 conversion, so it is not
  reproducible from the shards.
- Legacy G1 only: density-matched thresholds use non-self rows and a self-loop-stripped reference
  edge count, yet self-pairs still assemble as self-loops (`g1_hardened_e2.py`); changing either side
  moves every operating point.
