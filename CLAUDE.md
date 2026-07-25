# CLAUDE.md

Guidance for Claude Code and Codex working in this repository.

## What this is

A research-paper project targeting ICLR 2027: *Topology-Conditioned Inductive Edge
Prediction*. `README.md` is the human entry point (orientation, status, structure map).
This file holds the *constraints*.

**`docs/` are binding contracts, not documentation.** `docs/05-egostitch-spec.md` was
signed off at G4 and is frozen: code may not silently deviate — edit the spec first
with a §12 change-log line, then the code. A spec rewrite authorizes *implementation*,
not *execution* (§14). Authority order: `01-blueprint` → `02-methodology` →
`03-experiment-protocol` → `04-model-proposal` → `05-egostitch-spec` → `results/*`.
Later refines earlier, but blueprint §10 Locked Decisions and protocol §0 override
casual changes — **flag conflicts, don't resolve them unilaterally**.

## Skills — load before acting

- `.claude/skills/formal-run-protocol/` — pre-registration binding, BINDING vs
  diagnostic-only artifacts, what claims a one-seed screen may make. **Load before
  touching `docs/registrations/`, running a formal arm, or writing up any result.**
- `.claude/skills/hpc-execution/` — `hpc/run.sh`, `g5_stage1.sh`, `qualification.sh`,
  world-size auto-detection, and the config keys that change meaning per model family.
  **Load before any training/scoring/gate command.**

## The core thesis (do not let it drift)

Inductive edge prediction is **topology-conditioned binary classification** — not
independent pairwise scoring, not graph generation. Generated local topology is always
*intermediate context*; the final task is always binary edge prediction for queried
pairs. Every section must answer "how does this help predict `edge(u, v)` for unseen
nodes?" (`docs/lit-review-plan.md` §5, binding for all writing.)

## Integrity gates and claim rules (non-negotiable — the strict protocol is the contribution)

- **Edge-level and assembled-graph metrics are always reported together.** No
  single-metric-family claims. Protocol §E5; methodology §6.
- **Never claim significance or cross-seed robustness from a G5 Stage-1 screen** — it
  is fixed-Seed-0 engineering evidence; p-values/CIs/Holm must be `null`. Only E1/E3
  (≥3 seeds + Holm) carry inference.
- **Never call an MMD composite "graph similarity" and never aggregate the three MMD
  ratios.** Global simple-edge RD and BFS-macro RD are named separately, always.
- Dispositions are **owner-side decisions** — never settled by a screen, a note, or an
  agent.

## Data contract traps

- **`*_ratio5_exclusive.txt` is quarantined** (all strategies): its negatives leak
  across the node split. No loader in `src/` reads them — keep it that way.
  `src/data/artifacts.py:9`, spec §9.3.
- **`train_graph.pkl` contains every val positive** (`artifacts.py:258` asserts
  `train⁺ ∪ val⁺`). It is for split audits only. Everything structural must come from
  `build_g_struct` (`src/data/partition.py:69`). `val_edges.txt` is model selection
  only, never a training target.
- **`exclude_nodes` filters only `train/val/test_pairs`** (`artifacts.py:367`) —
  `graph`, `train_graph`, `test_graph`, and `buckets` come back unfiltered, so
  featureless nodes silently survive into structural computations.
- **Self-loop policy is asymmetric**: training structural targets strip self-loops;
  canonical MMD descriptors and official GS/RD induced subgraphs *retain* them, exactly
  as the benchmark evaluator does. Spec §9.4.
- **Grounding pools are universe-scoped** (`V_fit`/`V_qual`/`V_select`/test, separately
  hashed). One cache may never serve another; rehearsal may not read a `V_select` row.
  Spec §13.12.

## Traps that silently corrupt results rather than raising

- **`load_scores` does no precision validation** (`src/score_universe.py:665`). A
  bf16-contaminated EgoStitch artifact loads and analyses cleanly. Call
  `validate_artifact_precision(artifact, label=...)` yourself — it is the only correct
  entry point; calling `validate_score_precision` directly on an `egostitch_e2e`
  artifact spuriously raises "missing arrays".
- **Never shard the Stage-1 scorer.** `_score_egostitch` derives its grounding-pool
  universe from *this call's* `pairs` (`score_universe.py:1600`), so `--shard` yields
  different logits per shard. The e2e path is safe only because `_run_score` forwards
  `universe_pairs`/`row_start`.
- **The fp32 island in `stitch.py:79` must promote `h`/`pi`/`m` before** cost and
  marginal products are formed. Casting afterward keeps the bf16 ulp grid (0.03125
  spacing at |logit| ∈ [4,8)) and silently quantizes logits. Spec §13.16.
- **Null-head naming is inverted** at `src/model/egostitch/e2e_model.py:416`:
  `pair_content` comes from `NULL_TOPO_HEAD`, `pair_topology` from `NULL_CONTENT_HEAD`.
  Swapping them mislabels two published arms.
- **`data.partition_seed` is not `seed`** (`train_egostitch.py:366`). Changing it
  silently changes `G_struct`, the internal holdouts, and the whole s0 pair universe.
- **`_config_hash` bakes in paths** (`train_egostitch.py:708` — `output_dir`,
  `data.root`, `preregistration`), so overriding `--output-dir` breaks the
  `config_hash` equality gate at `src/experiments/probes.py:505`.
- **The s0 manifest is tied to `(world_size, epochs, negative_ratio, seed)`**
  (`train_egostitch.py:1786`). Launching with a different `--num_processes` than the
  manifest build leaves the s0 cache incomplete.
- **Stale caches warn instead of failing**: mismatched grounding (`grounding.py:58`)
  and F0 (`features.py:130`) caches log a warning, recompute in memory, and are *never
  rewritten* — so the mismatch recurs on every run.
- **Packed-feature manifests depend on `index.json` insertion order**
  (`packed_features.py:571`); sorting or reserializing it invalidates the pack. The F0
  cache written during packing holds exact fp32 means taken *before* bf16 conversion,
  so it is not reproducible from the shards.
- **Density-matched thresholds** are computed on non-self rows (`u_idx != v_idx`)
  against a self-loop-*stripped* reference edge count, yet self-pairs still assemble as
  self-loops at that same threshold (`g1_hardened_e2.py:822`). Including self-pairs in
  the quota, or dropping them from assembly, shifts every operating point.

## Numbers and naming

The load-bearing E2/G1/G2/G3 values are duplicated across
`docs/results/E2-pair-to-topology-gap.md` (canonical), `docs/03-experiment-protocol.md`,
`docs/04-model-proposal.md`, `README.md`, and `figures/e2-gap.html`. Read the canonical
file rather than trusting a remembered number, and update all five together.
Current G5 verdicts live in `docs/results/G5-stage1-seed0-20260717.md` (frozen-s0,
`cut`) and `docs/results/G5-e2e-stage1-seed0-20260724.md` (rev-3.0 e2e, `cut`).

Benchmark and baseline names (`Benchmark-A/B/C`, `B0`, `B0-alt`, `B0-e2e`, `B1`,
`B2-*`, `B3`, `B5`, `Ours`, `Oracle`, `PA-null`) are deliberate dataset-agnostic
placeholders. Don't substitute real dataset names unless asked, never conflate `B0`
with `B0-e2e`, and keep `PA-null` as a mandatory control.

`literature/` is a hand-curated, gitignored reference collection. Citations in
`docs/04-model-proposal.md` §8 are verified against arXiv or a local PDF — verify new
ones the same way before quoting them as fact.
