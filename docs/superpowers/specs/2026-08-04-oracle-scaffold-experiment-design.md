# Oracle-Scaffold Experiment — Design (Wave 1)

**Date:** 2026-08-04
**Status:** DRAFT (Wave-1 configs authored; not yet run)
**Scope:** `configs/egostitch_e2e_v3_oracle_*.yaml`,
`src/model/egostitch/{generator,encoder}/` (concurrent), `src/train_egostitch.py`
(concurrent), `src/score_universe.py` (wave 2)
**Builds on:** `docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`
(the `NeighborhoodGenerator` / `GraphEncoder` / `PairClassifier` interfaces this
experiment instantiates two new members of). Does not amend that design; this is a
new experiment defined against its interfaces.

---

## 1. Goal

Bound the value of topology conditioning before investing further in a learned
generator. The three-component refactor made the generator substitutable
(`generator.name: "null"` reproduces the pairwise baseline by config alone); this
experiment substitutes it the other direction, with a generator that cannot be
wrong about the graph it emits, and asks whether a conditioning **architecture** can
turn that perfect signal into predictive lift at all.

If a zero-parameter oracle generator feeding a real encoder and conditioning ladder
does not measurably beat the null (pairwise) baseline, the ceiling on any *learned*
generator is the same or lower, and further generator investment (`egostitch_imagine`,
its slot decoder, its Sinkhorn alignment) is not justified until the conditioning
architecture itself is fixed. If it does beat the baseline, the gap between R1's oracle
result and today's `egostitch_imagine` arms (`full`, `row_layernorm`, etc.,
`docs/results/E2-pair-to-topology-gap.md`) is the headroom actually available to close
by improving generation quality, separated from headroom lost to a conditioning
pathway that cannot consume good structure even when handed it.

**Disposition is owner-side** (CLAUDE.md, non-negotiable): this document defines the
run matrix and metrics, not the go/no-go call. §7 states the threshold shape so the
owner can apply it once results exist; it does not pre-commit an outcome.

## 2. Relationship to the G3 evaluation-side Oracle

`docs/results/E2-pair-to-topology-gap.md` §4 already ran a G3 Oracle gate: **Oracle-topo**
and **Oracle-blend**, evaluation-side arms that re-rank B0's candidate universe using
true-graph common-neighbor / Adamic-Adar counts computed directly from the test graph,
with no model in the loop at all. That gate confirmed the gap is real and
architecture-independent (Oracle-topo GS is `1.611×` B0; no single Oracle arm dominates
every metric) — it is evidence that *the signal exists in the graph*, not evidence that
*a trained architecture can extract it*.

This experiment tests the second claim. `oracle_struct` is a generator, not an
evaluation-time re-ranker: it participates in the same `NeighborhoodGenerator` protocol
as `egostitch_imagine` and `null` (`encode_node`/`stitch`/`forward`,
three-component-refactor design §3.3), feeds a real `GraphEncoder`
(`grit_gmt` or `ste_typed`), and its output is consumed by a trained `PairClassifier`
through one of four conditioning pathways. Where G3 asked "is there headroom," R1 asks
"can this architecture reach it." A weak R1 result despite a strong G3 result would
localize the failure to the conditioning/encoder side rather than the generation side —
the opposite of what the 2026-07-24/07-27/07-29 slot-collapse and dead-pointer
diagnoses (`e2e-stage1-failure-diagnosis`, `rev32-slot-collapse-diagnosis-design`)
found for the learned generator, which is exactly the ambiguity this experiment is
designed to resolve.

## 3. The two oracle rows

| Row | Source graph | Partner handling | Status |
|---|---|---|---|
| **R1** | `G_struct` (training structural graph, spec §9.3's loopless projection) | Leave-one-out: the queried partner is explicitly masked out of the emitted scaffold at stitch time, matching the "edge-stream structural targets must explicitly remove the queried partner and decrement its degree" rule (CLAUDE.md data-contract traps) | **Wave 1 — this document, configs below** |
| **R2** | The labeled test graph (validation/test positives included) | None — a diagnostic ceiling, not a protocol-clean arm | **Wave 2 — deferred, `score_universe.py`-side, no training config** |

R1 is the only row with a training config in this wave. It answers "what does the
architecture do with the true graph under the same information constraint every other
arm respects" — no test-graph leakage, no shortcut through the label itself. R2 answers
a different question ("what is the absolute ceiling if leakage were not a constraint")
and is explicitly a diagnostic upper bound, not a comparable arm — it must never be
plotted or tabled alongside B0/R1 without that caveat attached, and per CLAUDE.md it
carries no significance claim of any kind regardless of wave.

## 4. Architecture summary

**Generator — `oracle_struct`.** Zero-parameter. Reads the ground-truth local scaffold
for a node directly from `G_struct` rather than imagining one; `oracle_seed` (new
`GeneratorConfig` field, default 0) seeds any tie-breaking needed when truncating a
node's true neighborhood to the grounding budget. Emits the same `ImaginedGraph`
contract as `egostitch_imagine` (`x`/`adj`/`mask`/`aux`) built by the same
`build_scaffold` structural-channel assembly, so nothing downstream needs to know it
isn't a learned graph.

**Encoder — `grit_gmt`.** Vendored, not reimplemented from a paper description:
official GRIT transformer layers (MIT-licensed upstream) and the Set-Transformer PMA
pooling layer (MIT-licensed upstream), with any shim diffs against the vendored source
recorded at the vendoring site. The GMT (Graph Multiset Transformer) reference repo was
evaluated and **rejected for vendoring — it ships no LICENSE file**, so its pooling
variant is not used even though it is the more directly relevant paper; PMA from the
Set-Transformer repo (which does carry a permissive license) is used instead, both to
stay license-clean and per the owner's design pin.

Shape: dense RRWP (relative random-walk positional encoding) at `K = 8`, four
`GritTransformerLayer`s at hidden width 96, then PMA pooling with 4 seed vectors.
Token count follows the graph's own `N = 2 + 2K_slot = 34` (the same slot-count
identity as `egostitch_imagine`'s emitted graph, three-component-refactor design §3.1)
plus the 4 PMA seed tokens, so `GraphEmbedding.tokens` carries 38 rows; `pooled` is the
mean of the 4 seed tokens, not a learned readout, matching `GraphEmbedding`'s existing
two-form contract (design §3.2) without requiring a classifier-side change to consume
it.

**Conditioning ladder.** Four `ClassifierConfig.conditioning_mode` values, each
otherwise sharing the same `oracle_struct` × `grit_gmt` pair:

| Mode | Consumes | Notes |
|---|---|---|
| `film_logit` | `pooled` | FiLM-style modulation applied at the logit, the cheapest pathway |
| `pooled_adapter` | `pooled` | An adapter block ahead of the trunk rather than a logit-level scale/shift |
| `xattn_cls` | `tokens` (all 38, CLS included) | Today's `b0_v31` gated cross-attention pathway, generalized to the new token count |
| `xattn_tokens` | `tokens` **excluding the CLS query** | Per the owner's CLS-may-be-noise finding: tests whether dropping the CLS query from the attention target set removes a noise source `xattn_cls` carries |

Every mode carries a **zero-init null-identity guarantee**: at initialization, the
conditioning pathway must contribute nothing, so an untrained conditioned arm reduces
to the unconditioned baseline rather than starting from an arbitrary random
perturbation of it. This is the same invariant the existing conditioning-dropout
nulls (`∅_content`/`∅_all`) already rely on elsewhere in the model; it is being
generalized to a fourth pathway shape, not invented fresh here.

## 5. Run matrix

| Arm | Config | Generator | Encoder | Conditioning | Wave |
|---|---|---|---|---|---|
| B0 historical reference | Existing completed B0 result; no Wave-1 training config | n/a | n/a | pairwise | external context only |
| R1 × film_logit | `configs/egostitch_e2e_v3_oracle_grit_film_logit_breadth_first.yaml` | `oracle_struct` | `grit_gmt` | `film_logit` | 1 |
| R1 × pooled_adapter | `configs/egostitch_e2e_v3_oracle_grit_pooled_adapter_breadth_first.yaml` | `oracle_struct` | `grit_gmt` | `pooled_adapter` | 1 |
| R1 × xattn_cls | `configs/egostitch_e2e_v3_oracle_grit_xattn_cls_breadth_first.yaml` | `oracle_struct` | `grit_gmt` | `xattn_cls` | 1 |
| R1 × xattn_tokens | `configs/egostitch_e2e_v3_oracle_grit_xattn_tokens_breadth_first.yaml` | `oracle_struct` | `grit_gmt` | `xattn_tokens` | 1 |
| R1 ste-typed reference | `configs/egostitch_e2e_v3_oracle_ste_ref_breadth_first.yaml` | `oracle_struct` | `ste_typed` | placeholder, re-pin before launch (§6) | 1 |
| Local smoke | `configs/egostitch_e2e_v3_oracle_smoke.yaml` | `oracle_struct` | `grit_gmt` | `xattn_cls` | 1 (verification only, not scientific) |
| Perturbation ladder: edge-drop | *(no config yet — wave 2)* | `oracle_struct` + new perturbation | `grit_gmt` (ladder winner) | ladder winner | 2 |
| Perturbation ladder: add-matched | *(no config yet — wave 2)* | `oracle_struct` + new perturbation | " | " | 2 |
| Perturbation ladder: degree-preserving rewire | Existing `--scaffold-control rewire_checkerboard_v1` (`score_universe.py`, arm `structure_control_6e_v1`) | `oracle_struct` | " | " | 2 |
| Perturbation ladder: neighbor substitution | *(no config yet — wave 2)* | `oracle_struct` + new perturbation | " | " | 2 |
| Eval-side full shuffle | Existing `--scaffold-control shuffle_within_pair_v3` (`score_universe.py`, arm `structure_control_6a_v3`) | `oracle_struct` | " | " | 2 |
| R2 diagnostic ceiling | *(`score_universe.py`-side, no training config)* | `oracle_struct` on the labeled test graph | " | " | 2 |

Wave 1 is the full R1 ladder (five training configs) and a local smoke test. The
already completed B0 result is historical external context only: its seed, sampling
ratio, epoch count, optimizer schedule, weight decay, and label smoothing differ, so
no B0-to-R1 delta is a controlled causal comparison. The attempted trainable-null R0 arm was
withdrawn by the owner on 2026-08-04 because it duplicates B0; its incomplete run and
dedicated config are not evidence and are not part of this matrix. Every remaining
Wave-1 training config uses `training.phase_a_fraction: 0.0`: the oracle generator is
zero-parameter and has no generator-owned auxiliary loss, so a generator-only Phase A
would execute zero-gradient forward/backward steps and advance optimizer state without
training any component. Phase B remains the shared 10% conditioning ramp.
Wave 2 — the perturbation ladder that degrades the oracle's signal in controlled ways
(edge-drop, add-matched, degree-preserving rewire, neighbor substitution) plus the
eval-side full-shuffle control and the R2 ceiling — is scoped but not configured here;
`edge-drop`/`add-matched`/`neighbor substitution` need new `stitch`-time perturbation
kinds alongside the two that already exist (`shuffle_within_pair_v3`,
`rewire_checkerboard_v1`), consistent with the "`stitch` must accept a scaffold-control
perturbation" amendment already load-bearing in the three-component-refactor design
(§3.3 Amendment 2, 2026-08-03).

The ste-typed reference arm exists to separate "does the true graph help at all" from
"does `grit_gmt` specifically help" — see its config header for why its
`conditioning_mode` is an explicit placeholder pending the R1 ladder's winner.

## 6. Metrics

Per CLAUDE.md's non-negotiable integrity gates, unchanged by this experiment:

- **Edge-level and assembled-graph metrics are always reported together.** Every arm's
  per-epoch AUPRC (edge-level) is reported alongside its clustering MMD
  (assembled-graph family) — never one without the other, and the full assembled-graph
  family (global simple-edge RD, BFS-macro RD, the three MMD ratios reported
  individually, never aggregated into a composite "graph similarity" number) is run on
  selected checkpoints, not just clustering MMD in isolation.
- **Seed-0 Stage-1 screen.** Every wave-1 run in §5 is a single-seed (`seed: 0`)
  engineering screen, exactly like the G5 Stage-1 family it follows. No p-values,
  confidence intervals, or Holm correction are computed or implied; any such field in
  a result record is `null`. Cross-arm or cross-seed robustness claims are out of
  scope for this wave regardless of how clean a result looks — that inference is
  reserved for a future ≥3-seed E1/E3-style run, not this screen.
- Checkpoint selection follows the existing owner-judged process: there is no
  `e2e_checkpoint_eligible` predicate (deleted 2026-08-02, not reintroduced here) —
  the owner reads per-epoch `metrics.jsonl` rows directly, as for every other e2e arm.

## 7. Owner decisions flagged, not made

This document does not settle any of the following; it exists so they can be decided
with the run matrix and metrics already fixed, not improvised mid-analysis.

1. **Go/no-go threshold shape.** Wave 1 may compare the five R1 conditioning pathways
   to one another and judge their absolute edge/topology metrics, but must not treat
   the historical B0-to-R1 difference as a controlled effect. This document takes no
   position beyond noting both metric families must move together (§6) — a result that
   improves AUPRC while regressing every MMD ratio, or vice versa, is not a clean
   "go" under the standing edge+graph joint-reporting rule and needs an owner call on
   how to weigh the tradeoff, not a formula baked into this design.
2. **Protocol addendum for oracle rows.** R1's leave-one-out masking and R2's
   test-graph ceiling are new run *kinds* that do not fit neatly into the existing
   G5/rev-3.x Stage-1 vocabulary (`docs/03-experiment-protocol.md` predates this
   generator family entirely). Whether R1 needs its own named protocol addendum
   before its numbers can be cited anywhere outside this screen, or whether it stays
   inside the existing Stage-1 screen umbrella, is an owner call.
3. **Whether a weak R1 result kills the generator program or just this conditioning
   ladder.** §1 frames the binary "no lift here means no lift from a better
   generator" reading, but a narrower reading — "these four conditioning modes
   specifically can't consume it, try others before concluding the generator ceiling
   is the bottleneck" — is equally defensible from the same data. This document does
   not pick between them.
4. **R2's status once measured.** Whether the R2 ceiling number is ever reported
   outside an internal diagnostic context (it is a leakage-permitting upper bound, not
   a comparable arm per §3) is an owner call, not a default-open one.

## 8. Verified integration and runtime contracts

Separate from §7: these are live implementation facts verified before launch, not
judgment calls. Each wave-1 config's header comment repeats the relevant subset.

- `GeneratorConfig.oracle_seed: int = 0` and an `"oracle_struct"` entry in
  `GENERATOR_REGISTRY`.
- `EncoderConfig.{rrwp_k, n_heads, seeds}: int` and a `"grit_gmt"` entry in
  `ENCODER_REGISTRY`; `dim`/`layers` are reused fields, not new ones.
- `ClassifierConfig.conditioning_mode: str` over the four values in §4, with the
  zero-init null-identity guarantee actually implemented per mode.
- Trainer support for an empty generator optimizer group, because every R1 oracle
  generator is zero-parameter while its encoder and classifier remain trainable.
- Component-aware batch construction for the zero-parameter oracle generator: omit
  the generator node stream, grounding-pool transfer, and generator-only targets.
  The four GRIT arms also omit unused relational targets (`w_rel: 0`); the STE
  reference retains them. This changes data movement only, not any arm's loss.
- The five retained 30-epoch configs use a 12-hour train/eval hard budget instead
  of the inherited 7-hour-20-minute cap. The previous cap was below the observed
  full-run extrapolation and could terminate a healthy run before epoch 30.
- `_e2e_arm_name_from_config` recognizes `generator.name: oracle_struct` before legacy
  ablation labels, so all five arms use oracle telemetry/checkpoint semantics.
- The training loop passes `training.phase_a_fraction`/`phase_b_fraction` into the
  phase resolver; all five configs therefore skip Phase A exactly as declared.
- Feature-standardization choice (config-level, not code): every oracle config pins
  `feature_standardization: row_layernorm`, mirroring the null-generator arm rather
  than the real generator's `zscore_vfit_v1` default, on the inference that
  `oracle_struct` — like `NullGenerator` — has no `EgoStitchConfig`/decoder of its own
  to bind V_fit statistics onto and reads only the ground-truth adjacency table. This
  is verified against the live zero-parameter `OracleStructGenerator` implementation.
