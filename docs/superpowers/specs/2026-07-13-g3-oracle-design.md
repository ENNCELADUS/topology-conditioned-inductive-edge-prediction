# G3 Oracle Gate — Design Spec (approved 2026-07-13)

> **Historical design record:** the composite/global-density output schema below was
> superseded by official PRING GS/RD on 2026-07-14. It is not a current result source;
> use `outputs/deliverables/g3_pring_20260714/`.

**Deliverable:** `src/experiments/g3_oracle.py` + `tests/test_g3_oracle.py`, implementing
the pre-implementation gate G3 ("Oracle first") of `docs/03-experiment-protocol.md` §5.0,
plus the pinned-instantiation contract edit in docs/03 (done in the same change set,
before code, per the freeze-rule convention).

## 1. Purpose and gate semantics

G3 answers: *if an edge scorer could condition on the TRUE held-out graph neighborhoods
(the best possible version of the intermediate context EgoStitch would generate), how much
better would the assembled graph be than B0's?*

- Stop rule (docs/03 §5.0): **Oracle ≈ B0 on assembled metrics ⇒ feature insufficiency;
  conditioning cannot help; pivot.** The artifact reports per-statistic headroom
  `mmd_ratio(B0) / mmd_ratio(oracle)`; the pivot decision is human.
- The Oracle row is **protocol-violating by design** (evaluation-side access to the true
  test graph). It is a reference row, never a fair baseline.
- Out of scope: E4's identical-head "Oracle-scaffold" arm (post-implementation); any
  training; any change to G1/G2 behavior or outputs; new dependencies.

## 2. CLI contract (mirrors G1/G2)

```bash
python -m src.experiments.g3_oracle \
    --universe scores/b0_v31_candidate.npz \
    --data-root data --strategy breadth_first \
    --output-dir outputs/g3 [--seed 0]
```

Inputs: the pinned candidate-universe scores `.npz` (`src/score_universe.load_scores`)
and the benchmark artifacts. Validation fails fast, reusing G1's checks
(`validate_universe_artifact`: strategy match, expected row count
`C(n,2) + n`, node-set equality, `pairs_source == "candidate"`). Default `--seed 0`
(matches G1). Outputs: `g3_results.json` + `g3_tables.md` under `--output-dir`
(create the directory if needed).

## 3. Scorer rows (three)

Let `g_simple = strip_self_loops(test_graph)`, `node_ids` = the universe's node ids,
`(CN, AA)` = `common_neighbor_and_adamic_adar(g_simple, node_ids)` (existing G1
function; dense symmetric matrices). Row-aligned values: `cn_i = CN[u_i, v_i]`,
`aa_i = AA[u_i, v_i]` for every universe row *including self-pairs* (diagonal values
used as-is; disclosed in metadata — PA-null precedent scores all rows too).

1. **`b0`** — `probs = artifact.probs()`; density-matched threshold assembly via G1's
   exact code path (`density_matched_threshold` on non-self rows against
   `target_edges = |E(g_simple)|`; self-pairs assemble at the same threshold, reported
   in the self-loop row, never counted toward the quota). Its assembled numbers must
   reproduce G1's B0 row (same seed/config) — this is a required cross-check.
2. **`oracle_topo`** (canonical; drives the stop rule) — two related objects:
   - *Assembly ordering:* lexicographic (CN desc, AA desc, canonical pair order asc) —
     the exact convention of G1's `select_hard_heuristic` / `_lexsort_top_k`.
   - *Scalar score* `s_topo` (for edge metrics and the blend): the **average-tie
     normalized rank** of the `(CN, AA)` key over all universe rows — rows with equal
     `(CN, AA)` share the mean rank; ranks scaled to `[0, 1]` with 1 = most edge-like.
     Deterministic; implement via `np.lexsort` + run-length averaging over equal keys
     (or an equivalent deterministic method), n ≈ 2.04M rows so O(n log n) only.
3. **`oracle_blend`** (secondary, disclosed) —
   `s_blend = 0.5 · rank01(probs_b0) + 0.5 · rank01(s_topo)`, where `rank01` is the
   same average-tie normalized rank in `[0, 1]`. Parameter-free; no fitting on test
   labels.

## 4. Assembly and assembled-row evaluation

- `target_edges = |E(g_simple)|`.
- Oracle arms assemble by the **PA-null convention**: exact top-`target_edges` rows
  among **non-self pairs only**, deterministic tie-break, `threshold = None`. For
  `oracle_topo` the top-N uses the lexicographic ordering above; for `oracle_blend`,
  `assemble_top_n_by_score` on `s_blend` (its built-in `(u, v)` tie-break).
  Self-pairs are excluded from oracle top-N pools (CN diagonal degenerate; same
  rationale/treatment as PA-null) and appear only in the self-pair rows.
- Each arm gets G1's full `assemble_and_evaluate`: canonical MMD ratios with
  raw/reference disclosure, bootstrap mean/std, relative density, self-loop counts,
  composite. The O'Bray `perturbation_check` runs **once** (reference-graph property,
  as in G1) and gates `composite` for all arms. `noise_floor` is computed and included
  (G1 pattern). Same `MMDConfig` / `CompositeDefinition` defaults as G1.

## 5. Edge-level reporting (integrity gate: always with assembled rows)

- Full G1-style regime table for all three scorers: `easy_uniform`,
  `degree_corrected`, `hard_heuristic`, `hard_feature` at ratios {1, 5}, plus
  `full_universe` — reuse `select_negative_indices` / `evaluate_regime_table` with the
  scalar scores as probabilities. Regime index selection is scorer-independent: select
  once per regime, evaluate all scorers on the same rows.
- `compute_self_pair_edge_metrics` per scorer.
- Metadata note (verbatim caveat): `hard_heuristic` negatives are CN/AA-selected, so
  `oracle_topo`'s hard-heuristic row is degenerate by construction — disclosed, not
  hidden.

## 6. Outputs

`g3_results.json` mirrors `g1_results.json`'s shape where applicable:

- `metadata`: universe artifact summary (checkpoint id, family, rows, strategy),
  `oracle` block with formula strings for `oracle_topo` (ordering + scalarization) and
  `oracle_blend` (fusion formula), access =
  `"evaluator_side_oracle_reference (protocol-violating by design; G3 reference row)"`,
  seed, `mmd_config`, composite definition + `composite_valid`, threshold policy
  string, regime-construction strings, notes (hard-heuristic degeneracy, self-pair
  diagonal scoring).
- `regime_table`: `{b0, oracle_topo, oracle_blend} × regimes`.
- `self_pair_edge_metrics`: per scorer.
- `assembled`: per scorer `AssembledRow` dict (b0 with its threshold; oracle arms with
  `threshold: null`).
- `headroom`: per oracle arm, per statistic `b0.mmd_ratio[stat] / arm.mmd_ratio[stat]`,
  plus `composite_ratio = arm.composite / b0.composite` when both composites are
  computed (else null). Guard division: if an arm's `mmd_ratio[stat]` is 0, headroom is
  `null` (disclosed), not inf.
- `perturbation_check`, `noise_floor`, `degree_heterogeneity_sigma`, `positive_rate`
  (as in G1).

`g3_tables.md` renders: regime table, assembled-graph rows, MMD-ratio component table,
**headroom table** (the stop-rule view), noise floor — following `g1_tables.md` style
(`render_tables_markdown` precedent; a G3-local renderer is fine).

Determinism: identical inputs ⇒ byte-identical outputs.

## 7. Code-reuse policy

Import from `src.experiments.g1_hardened_e2` (same package, stable module):
`load_test_graph`, `load_test_node_buckets`, `validate_universe_artifact`,
`common_neighbor_and_adamic_adar`, `select_negative_indices` (and/or the individual
selectors), `evaluate_regime_table`, `compute_self_pair_edge_metrics`,
`assemble_and_evaluate`, `assemble_top_n_by_score`, `build_threshold_grid` (not
needed unless reused), `degree_heterogeneity_sigma`, plus `src.eval` primitives
(`density_matched_threshold`, `noise_floor`, `perturbation_check`, `MMDConfig`,
`CompositeDefinition`, `graph_similarity`, `strip_self_loops`). Do **not** copy-paste
these; do not modify G1/G2 modules except (if strictly necessary) zero-behavior-change
visibility tweaks.

## 8. Tests (`tests/test_g3_oracle.py`, mirroring G1/G2 test patterns)

- Oracle scalarization: average-tie rank correctness on hand-computed small cases
  (ties share mean rank; range [0,1]; ordering matches (CN, AA) lex).
- Determinism: repeated runs byte-identical; tie-break by canonical pair order in
  top-N assembly.
- Blend fusion math on hand-computed cases.
- Self-pair exclusion from oracle top-N pools; diagonal scores present in scalar
  arrays.
- Headroom math incl. zero-guard.
- b0 row cross-check: on a synthetic fixture, G3's b0 assembled row equals G1's
  pipeline output for the same inputs/seed.
- JSON/markdown rendering snapshots; CLI end-to-end on a tiny synthetic benchmark +
  synthetic scores artifact (reuse existing test fixture helpers where available).
- Validation failures (wrong strategy, wrong row count, non-candidate source) raise
  clear errors.

Quality gates: `uv run pytest`, `uv run ruff check . && uv run ruff format --check .`,
`uv run mypy src tests` (strict; heed the CLAUDE.md mypy-cache gotcha) all pass.

## 9. Contract edit (done with this spec, before code)

docs/03-experiment-protocol.md: (a) §2 Oracle row cell gains a pointer
"(pinned instantiation: §5.0 G3)"; (b) §5.0 G3 bullet gains the pinned-instantiation
paragraph (ordering, scalarization, blend formula, PA-null assembly convention,
headroom stop-rule quantity, evaluation-side access note); (c) header change-log line
"Updated 2026-07-13: pinned the G3 Oracle instantiation". G3 **result** numbers enter
docs only after the gate is actually run.

## 10. Final verification (after implementation)

Run the real gate locally against
`outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz`
(present, 10.1 MB) with `--output-dir outputs/g3`, confirm the b0 row reproduces the
G1 deliverable numbers (MMD ratios 13.0768/11.9273/18.0931, relative density
0.997710, composite 5.76802e-7), and report the oracle rows + headroom. Doc updates
with the resulting numbers are a separate follow-up owned by the user unless
requested.
