# Topology Prompt, Stage II: a generator that predicts the coordinates from attributes

**Design spec and implementation record.** Date: 2026-09-13. Status: implemented
(`model.family: v3_1_coord_gen`); two formal runs launched on the H20 containers. Follows
`2026-09-12-topology-prompt-stage1-design.md` (Stage I) and its result
(`docs/results/topo_prompt_stage1.md`).

## 1. What Stage I established and what Stage II must do

Stage I showed that the prefix_base trunk, reading the queried pair's true structural
coordinates through gated KV prefixes, decides the edge far better than without them
(V_val AUPRC 0.961 vs 0.814, GS 0.691 vs 0.401), that the gain vanishes under `mean` and
inverts under the universe-level `shuffle`, and that a *frozen* prefix_base trunk with only
the prompt trained recovers most of it (0.940 / 0.648) with the best shape ratios of any row.
The interface transmits structure and both readers use it; the relation field carries the
most, the context field nothing.

Stage II replaces the true coordinates with a prediction from the strict task input. Let
`E` be the reader's frozen per-node encoder. A generator

    ŝ_uv = g_ψ(E(x_u), E(x_v))

predicts the *standardised* coordinates, and the frozen reader scores the pair through the
same prompt interface `Π_ω` it learned on true structure:

    ℓ_uv = F_θ(E(x_u), E(x_v); Π_ω(ŝ_uv)).

Everything but `g_ψ` is frozen and comes from a published Stage I checkpoint, so the
deployed model is a function of `(x_u, x_v)` alone. The Stage I network fed the true
coordinates is the teacher; in this stage it shares every weight with the student's reader.

The question Stage II settles: **how much of the Stage I gain survives when the coordinates
must be predicted from the endpoint attributes, and which coordinates are predictable?**
Per the design note, the target is the structural expectation the attributes support,
`E[s* | x_u, x_v]`; the irreducible part is `E[Var(s* | X)]`, and fixed semantics let us
measure per field how much of that expectation is learned.

## 2. The generator (`src/model/egostitch/classifier/coord_gen.py`)

- **Inputs.** Each endpoint's frozen encoder tokens pooled to `[masked mean | masked max]`
  (`2 d_model = 1024`). No structure, no neighbours: only the two endpoints' attributes.
- **Heads.** Two MLPs (hidden 512, two GELU/LayerNorm/dropout blocks each). The *endpoint
  head* reads `[p_self | p_partner]` and is applied as `e(p_u, p_v)` and `e(p_v, p_u)` for
  the two endpoint fields; the *pair head* reads `[p_u + p_v | p_u ⊙ p_v | |p_u − p_v|]` and
  emits the seven continuous relation coordinates, four shortest-path class logits and the
  five context coordinates. Swapping `u` and `v` therefore swaps the endpoint fields and
  fixes the rest — the symmetry of the true coordinates and of the reader.
- **Output space.** Standardised coordinates, i.e. the space the reader consumes after its
  published `(mean, std)`; the reader's statistics are its own checkpoint buffers. The four
  shortest-path one-hots enter as softmax probabilities standardised with the same
  statistics (soft one-hots on the trained scale). The reader gets a new entry
  `V3_1TopoPrompt.logits_from_standardized` for this; its Stage I path is unchanged.
- **Trainable.** `g_ψ` only (~1.6 M parameters). The reader is frozen and kept in eval mode
  by `train()`; `trainable_parameters()` returns the generator, which `_build_optimizer`
  already prefers.

## 3. Losses

Training rows carry their true coordinates (`TopoPromptRows`, training graph with the
query edge removed — exactly Stage I's targets). Per row `i`, with the base's positive row
weight `w_i` so the trainer's DDP scaling applies unchanged:

    L = Σ_i w_i [ w_task · BCE(ℓ_i, y_i) + w_coord · C_i + w_kd · KD_i ] / Σ_i w_i

- `C_i`: Huber (δ = 1) averaged over the 30 continuous standardised coordinates, plus the
  cross-entropy of the shortest-path class. Fields are weighted equally; Stage I's
  attribution says the relation field matters most, and the per-field R² report tells
  whether it is also the hardest to predict.
- `KD_i`: soft-target BCE of the student logit against `sigmoid` of the same reader's
  logit on the true coordinates (computed under `no_grad` in the same forward). "Light":
  `w_kd = 0.1`.
- Defaults `w_task = w_coord = 1`, `w_kd = 0.1` (`model.config.coord_gen`).

Field masking is not applied to predictions; the reader's scoring-time interventions
(`gates_off`, `mean`, `mean_endpoint`, `mean_relation`, `mean_context`) apply to the
predicted coordinates as they did to the true ones. `shuffle` has no null here (the model
predicts its own coordinates) and fails closed.

## 4. Validation and scoring

- **Formal run.** The V_val universe, `val_cls` and every test pair are scored from the
  prediction alone; no truth graph is consulted at scoring time, so the family runs under
  the ordinary pipeline (`complete.json`, `test_report.json`) without `--allow-oracle-diagnostic`.
- **Selection and stopping** are the standard rules on the student's own V_val metrics
  (`geometric_rd_five_rank_v1`; val task loss patience).
- **Fit diagnostics** (`metrics.jsonl`, every epoch, collective across ranks): per-field R²
  of the standardised prediction against V_val's true coordinates on `val_cls`
  (`val_coord_r2_{endpoint,relation,context}`), the shortest-path class accuracy
  (`val_coord_dist_acc`), and the mean coordinate and KD losses. These read V_val truth
  only as validation metrics, as GS/RD already do.
- **Scoring.** `score_universe` scores the family on both packed and unpacked paths from the
  frozen encoder's cached node states; the checkpoint embeds the reader's config, state and
  statistics, so no reader file is needed at scoring time.

## 5. Runs (headline split, seed 0)

| Row | Config | Reader (frozen) | Role |
|---|---|---|---|
| `coord_gen_full` | `configs/split_seed42/coord_gen_full.yaml` | `topo_prompt_full/best.pt` (Stage I proper, retrained trunk) | **Stage II** |
| `coord_gen_frozen` | `configs/split_seed42/coord_gen_frozen.yaml` | `topo_prompt_frozen/best.pt` (prompt-only on prefix_base) | Stage II on the frozen-trunk reader: better shape ratios, prefix_base's own trunk |
| interventions on both | scoring only | — | `gates_off`, `mean`, `mean_relation` |

Recipe: only the generator trains, lr 1e-3 onecycle, 25 epochs, patience 10, micro-batch
196608 / 1536 per rank (the gradient flows through the frozen trunk into the prompt input,
so activations are retained as in the frozen Stage I lane; the teacher pass is transient).

```bash
hpc/run.sh train configs/split_seed42/coord_gen_full.yaml      # 30846
hpc/run.sh train configs/split_seed42/coord_gen_frozen.yaml    # 30838
hpc/run.sh test --checkpoint outputs/split_seed42/coord_gen_full/best.pt \
  --output-dir outputs/split_seed42/coord_gen_full/intervention_gates_off \
  --data-root data --strategy breadth_first --arm coord_gen_full_gates_off --seed 0 \
  --prefix-intervention gates_off
```

## 6. Reading the result

Report edge and assembled-graph families together, V_val first. Comparators: `prefix_base`
(the deployable comparator: same trunk for the frozen lane, same recipe for the full lane)
and each row's own reader on true coordinates (its ceiling: `topo_prompt_full` 0.961 / 0.691,
`topo_prompt_frozen` 0.940 / 0.648).

1. `coord_gen_*` ≫ `prefix_base` on V_val and `gates_off` / `mean` return it to the base:
   the predicted coordinates carry structure the trunk did not have, delivered through the
   interface. The fraction of the ceiling gap closed is the headline number of this stage.
2. `coord_gen_*` ≈ `prefix_base` with high per-field R²: the coordinates are predictable but
   the frozen reader does not tolerate prediction error (the `R(E[s*|X]) ≠ E[R(s*)|X]` shift);
   Stage III must adapt the reader to predicted structure.
3. `coord_gen_*` ≈ `prefix_base` with low R² on the relation field: the relation coordinates
   are not predictable from endpoint attributes; the generator needs richer inputs
   (attribute-retrieved context) before any reader adaptation is worth doing.
4. Per-field R² against Stage I's per-field attribution tells which coordinates are both
   useful and predictable.

Single-seed differences inside the split's noise band (±0.01 GS, ±0.5 MMD ratio) are not read.

## 7. Out of scope here

Stage III/IV (unfreezing the interface, readout and upper encoder; topology losses),
generator inputs beyond the two endpoints, and the "coordinates through a plain MLP head"
control that would attribute the gain between the representation and the prefix.
