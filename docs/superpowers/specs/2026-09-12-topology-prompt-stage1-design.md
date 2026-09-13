# Topology Prompt, Stage I: a reader that uses true structure through a prefix

**Design spec and implementation record.** Date: 2026-09-12. Status: implemented
(`model.family: v3_1_topo_prompt`); both diagnostic runs complete; results in
`docs/results/topo_prompt_stage1.md` (§9 below summarises).
Supersedes the attribute-conditioned prefix arms of
`2026-09-10-prefix-tuning-frozen-trunk-design.md` as the prefix design under study; those
arms remain as comparison rows.

## 1. Why a new prefix design

The 2026-09-10 prefix arms learn a prefix from the endpoint attributes under structural
losses on a frozen trunk. Their most informative run, `prefix_static` trial 006, moved neither
the validation degree/motif losses nor V_val AUPRC over 25 epochs (degree loss 0.5455 → 0.5477,
AUPRC 0.8127 → 0.8098). That does not show the prefix mechanism is useless: architecture,
initialisation and trainable scope changed together, and the prefix never received an input
that carried structure the trunk did not already have.

The owner's revised route is a topology-representation-transfer pipeline in four stages:

| Stage | Structure in the forward | Trains | Loss |
|---|---|---|---|
| I | true coordinates `s*` from the training graph | trunk + prompt interface | task BCE |
| II | predicted `ŝ = g(x_u, x_v)` | the generator `g` | coordinate supervision, task BCE, light logit KD |
| III | `ŝ` only | generator, prompt, readout, upper encoder | task/subgraph BCE, coordinate loss, topology loss |
| IV | `ŝ` only | the whole student | task/subgraph BCE, topology loss, weak coordinate anchor |

This document implements Stage I and the question it must settle before anything else is
built: **can this trunk, reading true structure through the prefix interface, decide the
queried edge better than the same trunk without it, and does the gain disappear when the
structure is shuffled or replaced by its mean?** If a reader given true structure cannot use
it, no generator can rescue the interface; if it can, Stage II has a well-defined target.

## 2. Task contract and why Stage I is a diagnostic

The strict task input is `(x_u, x_v)`. Stage I adds true structure at inference, so every
Stage I run is a ceiling diagnostic in the same class as the Full-Ego Oracle: it is launched
with `--run-kind diagnostic`, publishes `diagnostic_complete.json` /
`diagnostic_test_report.json`, and its score artifacts carry `formal: False` with an
`oracle_diagnostic` block. It is never a deployable arm and never compared as one. Its two
deployable descendants are the later stages, where the prompt is fed `ŝ` computed from
`(x_u, x_v)` alone.

## 3. Structural coordinates `s*_uv` (`src/data/struct_coords.py`, spec `v1`)

For a queried pair, take the universe's true simple graph (training graph for training rows,
the V_val gold graph for validation rows, the labelled test graph for held-out rows), **remove
the queried edge first**, then measure. Every quantity has a fixed meaning; nothing is a free
latent. 34 numbers in four fields:

| Field | Dim | Content |
|---|---|---|
| `endpoint_u`, `endpoint_v` | 9 each | `log1p` degree; local clustering; `log1p` triangles and open wedges at the node; `log1p` two-hop reach; `log1p` mean neighbour degree; walk returns `(S^k)_xx`, `k = 2, 3, 4`, `S = D^{-1/2} A D^{-1/2}` |
| `relation` | 11 | `log1p` common neighbours; Jaccard; `log1p` simple L3 path count; walk kernels `(S^k)_uv`, `k = 2..5`; one-hot shortest-path class {2, 3, ≥ 4, disconnected} |
| `context` | 5 | `log1p` one-hop union; `log1p` two-hop shell `{x : min(d(u,x), d(v,x)) = 2}`; fraction of the shell at distance two from both roots; `log1p` cross-ring links; L3 link density `L3 / (d_u d_v)` |

Properties that matter downstream:

- Swapping `u` and `v` swaps the endpoint fields and fixes the rest, so the reader can be
  made exactly swap-symmetric (§4).
- With the edge removed and the graph loopless, `(A^3)_uv` is exactly the simple L3 count and
  `(S^3)_uv` its degree-normalised version; no walk backtracks through the queried edge.
- Self-pairs use the same formulas literally (Jaccard 1, no distance class); no self flag.
- Non-edges are read from dense all-pairs products (`n ≤ 7,203`, about ten `n×n` float32
  matrices, 2 GB); every edge is recomputed exactly on the edge-deleted graph, and that exact
  routine is the reference the dense path is tested against (`tests/test_struct_coords.py`
  also checks both against a NetworkX brute-force implementation).
- Cost, measured locally: training graph 5.5 s build + 3.2 ms per positive edge (100 s for
  31,623), V_val universe 297,208 rows in 3 s, test graph 65 s. Each DDP rank builds its own
  copy at startup.

Standardisation uses the mean and standard deviation over epoch 1's 1:5 training rows (the
distribution one epoch presents; the multi-epoch union over-weights negatives 25-fold), floored
so constant columns standardise to zero. The statistics are buffers of the published checkpoint.

**Known shift.** Absolute local structure differs across universes: mean `log1p` degree is
1.83 on the training graph, 2.15 on the V_val graph and 3.26 on the test graph (the BFS test
region is dense). The V_val readout is therefore the primary Stage I evidence; the test
diagnostic is reported with this caveat and is the same shift the Full-Ego Oracle faces.

## 4. The reader (`src/model/egostitch/classifier/topo_prompt.py`)

Base: the `prefix_base` recipe, the headline B0 with `mixing.mode: bidirectional_cross`
(three shared-weight bidirectional cross-attention layers ahead of the `pair_context_gated`
readout, halved micro-batch 262144/2048). The prompt path:

1. `standardise(s*)`, then optional field masking during training: with probability
   `field_mask_prob` (0.1) a field of a row is replaced by its training mean (standardised
   zero). This is the only perturbation; it keeps the reader from keying on exact values and
   makes the scoring-time `mean` interventions in-distribution.
2. Four tokens of width 128: `LayerNorm(GELU(Linear(field)) + role)`. The two endpoint
   tokens carry **self / partner roles relative to the attending stream**: the stream encoding
   `u` reads `[E(u) as self, E(v) as partner, R, C]`, the stream encoding `v` reads the mirror.
3. Per layer, one linear map expands the four tokens to `4 × slots_per_field` (= 8) key/value
   rows plus a learned static offset `p0`.
4. Each of the nine attention sites (A←B, B←A, CLS in three layers) adds the
   separately-softmaxed, tanh-gated prefix attention of the prefix arm
   (`prefix_branch`), gates zero-initialised: at initialisation the model is exactly the base.

Because the AB and BA passes receive mirrored prefixes and `abba_max` is symmetric, the logit
of `(u, v)` equals that of `(v, u)` with swapped coordinates (tested).

Two trainability modes, selected by `model.config.topo_prompt.trainable`:

- `all` (**Stage I proper**): trunk and prompt train, from scratch under the prefix_base
  recipe (or from a `base_checkpoint` warm start). The trunk must learn how a given structure
  should alter the interaction and the decision; a trunk trained only on task BCE would put the
  whole burden on the prefix.
- `prompt` (**frozen-trunk control**): the published `prefix_base` trunk frozen and in eval
  mode; only the tokenizer, per-layer projections, offsets and the 72 gate scalars train.

Every forward requires `batch["struct_coords"]`; a missing tensor raises. `StructStream` and
the KD banks are refused for this family (Stage I is task BCE only).

## 5. Training and evaluation plumbing (`src/train_b0.py`, `src/score_universe.py`)

- `TopoPromptRows` measures, once per rank at startup, the coordinates of every training-union
  row (training graph), every `val_cls` row and every V_val ball-union row (V_val gold graph),
  installs the statistics into the model, and attaches `struct_coords` per batch by `_row_id`.
- `--run-kind` now reaches the B0 worker (`Config.run_kind`, execution context only, excluded
  from the config hash). The family refuses to start outside `diagnostic`; `run_metadata.json`
  records `run_kind`, `checkpoint_role: diagnostic_only`, `formal_artifacts_published: false`
  and a `topo_prompt` provenance block.
- Scoring (`score_universe score`, and therefore the test protocol) requires
  `--allow-oracle-diagnostic` for this family, measures coordinates on
  `_oracle_truth_graph_for_scoring(pairs)` (V_val gold graph or labelled test graph, query
  edge removed), and marks the artifact `formal: False`.
- Interventions, `--prefix-intervention`: `gates_off` (the base function), `mean`
  (every field at its training mean), `mean_endpoint` / `mean_relation` / `mean_context`
  (one field at a time), `shuffle` (every row reads the coordinates of another row drawn by one seeded permutation
  of the whole universe, `--prefix-intervention-seed`; identical under fan-out and never within a
  batch or a shard: the 1:1 `val_cls`/`test` lists are label-sorted, so the within-shard
  permutation used until 2026-09-12 left every substitute with the row's own label there). Each intervention run re-selects its own V_val thresholds, as
  the prefix arms do.

## 6. Runs and comparison rows (headline split, seed 0)

| Row | Config | Trainable | Structure | Role |
|---|---|---|---|---|
| `prefix_base` | existing | all | none | the base; every prompt row is read against it |
| `topo_prompt_full` | `configs/split_seed42/topo_prompt_full.yaml` | trunk + prompt, from scratch | true `s*` | **Stage I**; becomes the Stage II teacher / initialisation |
| `topo_prompt_frozen` | `configs/split_seed42/topo_prompt_frozen.yaml` | prompt only on frozen `prefix_base` | true `s*` | freezing control; read against `prefix_pair_bce` (same trainable scope, attribute condition) |
| interventions on both | scoring only | — | shuffled / mean / off | attribution |

Launch (2026-09-12; 30030 runs the old `prefix_static`/`prefix_pair` studies throughout; 30838
carried another session's CAZI-MBN and official-PPI baselines, so the frozen lane took 30846 as
soon as the full lane had freed it):

```bash
# 30846, 09:51-15:12 UTC at c75dddf (train, test diagnostic, six interventions)
hpc/run.sh train configs/split_seed42/topo_prompt_full.yaml --run-kind diagnostic
# 30846, started 16:02 UTC
hpc/run.sh train configs/split_seed42/topo_prompt_frozen.yaml --run-kind diagnostic
# afterwards, per run (each writes its own output-dir)
hpc/run.sh test --checkpoint outputs/split_seed42/<run>/best.pt \
  --output-dir outputs/split_seed42/<run>/intervention_<name> \
  --data-root data --strategy breadth_first --arm <run>_<name> --seed 0 \
  --allow-oracle-diagnostic --prefix-intervention <name>
```

## 7. Reading the result

Report edge (AUROC/AUPRC) and assembled-graph (GS, RD, three MMD ratios) families together,
on V_val first, then the test diagnostic with the §3 shift caveat.

1. `topo_prompt_full` ≫ `prefix_base` on V_val, and `shuffle` / `mean` return it to
   `prefix_base` level: the interface transmits structure and the reader uses it. Stage II
   proceeds with `topo_prompt_full` as teacher and initialisation.
2. `topo_prompt_full` ≈ `prefix_base`: the trunk cannot exploit even true structure through
   this interface; redesign the interface before building any generator.
3. `topo_prompt_frozen` ≈ `prefix_base` while `topo_prompt_full` gains: freezing was the
   bottleneck of the earlier round; Stage III must unfreeze at least the interaction path.
4. `topo_prompt_frozen` ≫ `prefix_pair_bce`: the frozen interface can use structure when it is
   given; the earlier failure was the absence of structural information in an
   attribute-derived condition, which is exactly what Stage II must supply.
5. Per-field `mean_*` drops rank the fields by contribution and tell Stage II which
   coordinates must be predicted well; a field whose removal costs nothing can be dropped.

Single-seed differences inside the split's noise band (about ±0.01 GS, ±0.5 MMD ratio) are not
read; the paired bootstrap in the test protocol governs claims.

## 8. Out of scope here

Stage II–IV code (the coordinate generator, its supervision, joint fine-tuning), any struct-
stream supervision for this family, and a fixed-size context variant of the coordinates.

## 9. Result (2026-09-12)

Numbers and tables: `docs/results/topo_prompt_stage1.md`. On V_val, `topo_prompt_full` reads
val_cls AUROC/AUPRC 0.952/0.961 against `prefix_base`'s 0.793/0.814 and GS 0.691 against 0.401
at RD ≈ 1.03; `mean` returns it to 0.743/0.779 and GS 0.376, `gates_off` to 0.725/0.768 and
0.374, and the universe-level `shuffle` drops it to near chance (val_cls AUROC 0.579, test 0.553,
ball-union 0.641, GS 0.265), far below the base.
This is outcome 1 of §7: the interface transmits structure and the reader uses it. Per field,
`mean_relation` costs the most (GS 0.542), `mean_endpoint` little (0.674), `mean_context`
nothing (0.703). The three V_val MMD ratios do not improve with GS. On the test universe the
edge family transfers (AUPRC 0.939 vs 0.744) but the V_val-selected threshold under-densifies
the dense test region (RD 0.34) and the shape ratios triple — the §3 shift, read with that caveat.

Correction found while reading the result: the scorer's `shuffle` permuted within each fan-out
shard, and the 1:1 `val_cls`/`test` lists are label-sorted, so on those two universes every
substitute carried the row's own label (shuffle scored *above* the true coordinates there). The
scorer now draws one permutation of the whole universe (§5); the ball-union V_val topology
readout was unaffected (0.647 → 0.641 AUROC), and the rerun 1:1 rows fall to near chance.

Frozen lane (`topo_prompt_frozen`, prompt only on the frozen `prefix_base`, selected epoch 7):
val_cls AUROC/AUPRC 0.925/0.940, GS 0.648 at RD 1.02 with MMD ratios 4.2/2.0/6.0 (the best of any
row); `gates_off` reproduces `prefix_base` exactly, `mean` returns to it (0.773/0.800, GS 0.405),
the universe-level `shuffle` falls below it (0.718, GS 0.359). Outcome 3 of §7 does not hold:
freezing was not the earlier round's bottleneck. The attribute-conditioned `prefix_static` best
trial, the same trainable scope, sits at the base (AUPRC 0.813, GS 0.401), so the earlier failure
was the absence of structural information in the condition — what Stage II must supply. The
`prefix_pair_bce` row (outcome 4 proper) is still pending on 30030.
