# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`AGENTS.md` is a symlink to this file.

## What this is

Research code for an ICLR 2027 paper, *Topology-Conditioned Inductive Edge Prediction*. `README.md`
orients, `docs/` holds the specs, this file holds the constraints. Doc authority, later refining
earlier: `01-blueprint` → `02-methodology` → `03-experiment-protocol` → `04-model-proposal` →
`05-egostitch-spec` → `results/*`; the model's current design is
`docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`. Keep code and spec in step
within one change; flag genuine conflicts rather than resolving them silently.

**Core thesis — do not let it drift:** inductive edge prediction is topology-conditioned binary
classification, not independent pairwise scoring and not graph generation. Generated local topology
is always intermediate context; the output is always a 0/1 decision for a queried pair. Every piece
of writing must answer "how does this help predict `edge(u, v)` for unseen nodes?"

## Engineering rules

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility
  layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirement. Avoid speculative
  abstraction, configuration, and indirection.
- Grow the system in layers: start from the smallest version that works end to end, and add each
  capability on top of something that already works. Never trade a working product for unfinished
  complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce complexity or improve reliability.
  Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding
  packages; check a library's docs and types before assuming it lacks a capability.
- Make architectural decisions for the long term. No stopgap meant to be replaced later.

## Commands

Local (macOS, CPU only): use `.venv/bin/python -m …` — `rtk` garbles `uv run` output.

```bash
.venv/bin/python -m pytest                                          # full suite (xdist, loadfile)
.venv/bin/python -m pytest tests/test_score_universe.py -n0 -k density   # one file / one test
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests                                  # strict
uv sync                                                             # dependencies
```

Use `-n0` when debugging — xdist swallows breakpoints. Tests in one file share state, so
`--dist loadfile` is required, never `--dist load`.

GPU work runs only in the H20 container via `hpc/run.sh`
(`check | train | score | test | merge | g1 | g2`); world size and score shards are
auto-detected from visible GPUs, and there is no scheduler. `train` runs the held-out
test protocol automatically after publish (skipped by `--max-steps` debug runs); `test`
is also callable directly for the two scoring-time controls, which have no training run
of their own. Config keys change meaning per `model.family`; direct `python -m
src.train_b0` is debug-only. Runbook: `hpc/README.md` and the `hpc-execution` skill.

```bash
hpc/run.sh train configs/b0_v31_breadth_first.yaml                  # baseline
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal             # EgoStitch e2e
```

Review each implementation wave yourself: `CODEX_HOME=<scratch>/codex-home codex review --base <sha>
> wave-review.txt 2>&1`, backgrounded — ~200 KB of output must never land in context. That home needs
only `auth.json` and a `config.toml` pinning model/effort; `-c 'mcp_servers={}'` merges rather than
overrides, leaving the MCP tables alive, which hangs the review with zero output.

## Architecture

Flow: benchmark artifacts → node partition + grounding pools → packed bf16 features → DDP training →
score-once artifacts → gate and eval analyses.

- `src/data/` — `artifacts.py` verified benchmark loader; `partition.py:build_g_struct` is the only
  legal structural graph; `grounding.py` universe-scoped candidate pools; `internal_holdout.py`
  V_hold; `packed_features.py`/`features.py` bf16 pack and F0 cache; `pairs.py` batching.
- `src/model/egostitch/` — three independently swappable components behind `registry.py`, composed by
  `composite.py`, talking through `graph.py`'s dataclasses (`ImaginedGraph`, `GraphEmbedding`,
  `PairInputs`). `generator/` imagines a graph (`egostitch_imagine`, `null`, `oracle_struct`),
  `encoder/` encodes one (`ste_typed`, `grit_gmt`), `classifier/` decides the edge (`b0_v31`). Each
  slot is chosen by the `name` field of its config section, so adding a component is one registry
  entry, never a change to `composite.py`. No encoder may read `ImaginedGraph.aux`: it is
  generator-private, and a generator swap invalidates it wholesale.
- `src/train_{egostitch,b0,cazi_mbn}.py` DDP workers; `src/e2_pipeline.py` orchestrates the four
  stages pack → train → publish → test; `src/score_universe.py` writes score-once artifacts;
  `src/score_fanout.py` fans a score pass across visible GPUs and merges the shards (owns
  `hpc/run.sh score`); `src/eval/test_protocol.py` runs one published checkpoint's held-out
  V_hold → test → candidate sequence into `test_report.json`; `src/experiments/` holds the gate
  analyses (`g1_hardened_e2`, `g2_ceiling`, `g3_oracle`, `g5_stage1`, `probes`,
  `observe_e2e_formal`) and `src/eval/` the edge/graph metrics, assembly, and calibration.

Experiments run directly; there is no plan, registration, or qualification gate. Model-quality signals
(liveness, slot collapse, margins, dispersion) are telemetry in `profile.json`/`metrics.jsonl` and
never block a run; non-finite state, DDP disagreement, data-boundary violations, and I/O failures stay
fail-closed.

## Claim rules

- Report edge-level and assembled-graph metrics together — never one family alone.
- The topology result is five numbers, always together and never aggregated: GS, RD, and the degree /
  clustering / spectral MMD ratios (GS↑, RD→1, ratios↓). GS is edge-set Dice/F1
  (`src/eval/graph_metrics.py`), not an MMD summary; global simple-edge and BFS-macro RD (and GS) are
  always named separately.
  
## Data-contract traps

- `*_ratio5_exclusive.txt` is quarantined: its negatives leak across the node split, and no loader in
  `src/` reads them.
- `train_graph.pkl` is train⁺ ∪ val⁺ — split audits only. Structure comes from `build_g_struct`;
  `val_edges.txt` is model selection, never a training target.
- `exclude_nodes` filters only `train/val/test_pairs`, so featureless nodes survive in `graph`,
  `train_graph`, `test_graph`, and `buckets`.
- Self-loops: training structural targets strip them; canonical MMD descriptors and official GS/RD
  subgraphs keep them, as the benchmark evaluator does.
- Grounding pools are universe-scoped (`V_fit` train, `V_hold` validation, test): no cache crosses
  universes, and a training pass may not read a `V_hold` row.
- Topology and classification share the same train positives (no message/supervision split): loopless
  projection for topology, self-pairs kept for classification, and edge-stream structural targets must
  drop the queried partner and decrement its degree.
- `V_hold` is the union of two deterministic 256-node BFS draws (`internal_holdout.py`); the 2026-08-03
  shared-interaction correction changed that graph and invalidated every earlier cache, pack,
  threshold, and result.

## Traps that corrupt results silently instead of raising

- `load_scores` does no precision validation, so a bf16-contaminated EgoStitch artifact loads and
  analyses cleanly. Call `validate_artifact_precision(artifact, label=…)`; `validate_score_precision`
  called directly on an `egostitch_e2e` artifact spuriously raises "missing arrays".
- `select_e2e_checkpoint` ranks by AUPRC → clustering MMD → Brier and will return a degenerate
  checkpoint. There is no eligibility predicate and none should be added — judge usability from the
  per-epoch rows in `metrics.jsonl`.
- The fp32 islands in `generator/assemble.py` must promote their inputs *before* cost and marginal
  products are formed; casting afterwards keeps the bf16 ulp grid and silently quantizes logits.
- `allow_cache_subset=True` (live in `probes.py`, `score_universe.py`) gathers a superset F0 cache
  into a different node set with no content check. Exact-order mismatches raise; this path stays quiet.
- Packed-feature manifests depend on `index.json` insertion order; sorting or reserializing it
  invalidates the pack. Its F0 cache holds fp32 means taken before bf16 conversion, so it is not
  reproducible from the shards.
- Density-matched thresholds are computed on non-self rows against a self-loop-stripped reference edge
  count, yet self-pairs still assemble as self-loops at that threshold (`g1_hardened_e2.py`); changing
  either side moves every operating point.