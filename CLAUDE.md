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

**Active design:** `docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`
governs the current EgoStitch component-interface refactor and the excision of the
preregistration/formal-run-registration machinery (its §10). The owner has withdrawn
the registration mechanism; experiments are run directly, without a plan/artifact
identity gate.

## After each implementation wave

**You run the review, not the user.** Background it to a file and read the findings
when it finishes — output runs to ~200 KB, so never let it land in context:

```bash
CH=<scratch>/codex-home; mkdir -p "$CH"; cp ~/.codex/auth.json "$CH/"
printf 'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\napproval_policy = "never"\nsandbox_mode = "workspace-write"\n' > "$CH/config.toml"
CODEX_HOME="$CH" codex review --base <WAVE_BASE_SHA> > <wave>-review.txt 2>&1
```

`/codex:review` is `disable-model-invocation`, but that blocks only the slash command
— the `codex` CLI beneath it is yours to run.

Runtime, for any training/scoring/gate command: the world size is **auto-detected**
from all visible NVIDIA H20 devices via `hpc/run.sh`, and config keys change meaning
per model family. EgoStitch e2e trains through the same `hpc/run.sh train` branch as
the baselines, naming the worker explicitly:
`hpc/run.sh train <config.yaml> --worker-module src.train_egostitch --run-kind formal`.

**`-c 'mcp_servers={}'` does NOT disable MCP** — `-c` merges into the config, so the
`[mcp_servers.*]` sub-tables in `~/.codex/config.toml` survive it. Verified 2026-07-25:
a review launched with that override still started `gitnexus/detect_changes` and hung
there with zero output growth, which is the same failure that killed two Phase B
reviews. A clean `CODEX_HOME` (no `[mcp_servers]` table at all) is the fix; it also
carries model and effort, so no `-c` flags are needed. `--effort` is not a valid flag
either — `review` takes only `--base/--scope/--model/--cwd`, and a stray value is
parsed as focus text, which is rejected.

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
  `build_g_struct` (`src/data/partition.py:71`). `val_edges.txt` is model selection
  only, never a training target.
- **`exclude_nodes` filters only `train/val/test_pairs`** (`artifacts.py:367`) —
  `graph`, `train_graph`, `test_graph`, and `buckets` come back unfiltered, so
  featureless nodes silently survive into structural computations.
- **Self-loop policy is asymmetric**: training structural targets strip self-loops;
  canonical MMD descriptors and official GS/RD induced subgraphs *retain* them, exactly
  as the benchmark evaluator does. Spec §9.4.
- **Grounding pools are universe-scoped** (`V_fit` = training, `V_hold` = validation,
  test; separately hashed, and both ladder stages share the same three). One cache may
  never serve another; a training pass may not read a `V_hold` row. Spec §13.12.
- **`V_hold` must stay the union of the two former holdouts** (`V_qual ∪ V_select`:
  512 nodes, 1,533 positives, 130,816 pairs — `internal_holdout.py:171-178`). That union
  is the only reason `V_fit`, `e_msg_fit`, `feature_stats_sha256` and every pack manifest
  are bit-identical to the two-holdout era. Re-deriving `V_hold` from a single BFS draw
  silently changes `V_fit` and every digest. Spec §9.3.

## Traps that silently corrupt results rather than raising

- **`load_scores` does no precision validation** (`src/score_universe.py:648`). A
  bf16-contaminated EgoStitch artifact loads and analyses cleanly. Call
  `validate_artifact_precision(artifact, label=...)` yourself — it is the only correct
  entry point; calling `validate_score_precision` directly on an `egostitch_e2e`
  artifact spuriously raises "missing arrays".
- **There is no checkpoint-eligibility predicate, by owner decision (2026-08-02).**
  `e2e_checkpoint_eligible` is deleted and nothing replaces it: `select_e2e_checkpoint`
  returns the best record by AUPRC → clustering MMD → Brier and will happily return a
  degenerate one. No code decides whether a checkpoint is scientifically usable —
  judge it yourself from the per-epoch rows in `metrics.jsonl`. Do not reintroduce an
  automated eligibility gate.
- **The fp32 island in `stitch.py:80` must promote `h`/`pi`/`m` before** cost and
  marginal products are formed. Casting afterward keeps the bf16 ulp grid (0.03125
  spacing at |logit| ∈ [4,8)) and silently quantizes logits. Spec §13.16.
- **Null-head naming is inverted** at `src/model/egostitch/e2e_model.py:596`:
  `pair_content` comes from `NULL_TOPO_HEAD`, `pair_topology` from `NULL_CONTENT_HEAD`.
  Swapping them mislabels two published arms.
- **`data.partition_seed` is not `seed`** (`train_egostitch.py:401`). Changing it
  silently changes `G_struct`, the internal holdouts, and the whole training pair
  universe.
- **F0 caches serve supersets silently**: with `allow_cache_subset=True`
  (`features.py:130`; live at `train_egostitch.py:2245`, `probes.py:892`,
  `score_universe.py:1624`/`:1723`) a superset cache is gathered into a *different* node
  set with no content digest. Exact-order F0 mismatches (`features.py:140`) and grounding
  `pool_method_hash` mismatches now raise — this subset path is the one that stays quiet.
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
Current G5 verdicts live in `docs/results/G5-stage1-seed0-20260717.md` (frozen-s0, `cut`
— evidence retained; its producing code last exists at `dcae090` and is deleted in the
current cleanup worktree pending commit) and
`docs/results/G5-e2e-stage1-seed0-20260724.md` (rev-3.0 e2e, `cut`). The active line is
rev-3.2 on a single formal stage, run directly (`hpc/run.sh train`) without a
preregistration plan or artifact-identity gate — see
`docs/superpowers/specs/2026-08-02-three-component-refactor-design.md` §10.

Benchmark and baseline names (`Benchmark-A/B/C`, `B0`, `B0-alt`, `B0-e2e`, `B1`,
`B2-*`, `B3`, `B5`, `Ours`, `Oracle`, `PA-null`) are deliberate dataset-agnostic
placeholders. Don't substitute real dataset names unless asked, never conflate `B0`
with `B0-e2e`, and keep `PA-null` as a mandatory control.

`literature/` is a hand-curated, gitignored reference collection. Citations in
`docs/04-model-proposal.md` §8 are verified against arXiv or a local PDF — verify new
ones the same way before quoting them as fact.
