# Phase 5 — Feasibility audit of the motif-graph GRIT prompt

Audit of `docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md` against the code at
`1817ae2`. Read-only; every claim was checked in the current file, not the graph index.

## 1. Verdict

**Buildable with named changes.** Nothing in the repo blocks the design: the vendored GRIT layer
already computes and returns the final-layer edge state, the prefix interface already carries three
role-swapped fields at nine sites with exact zero-gate identity, and the arm is a classifier-side
`v3_1_*` family that never touches the EgoStitch registry other arms depend on.

The biggest obstacle is not the model but the **sampled-subgraph structural stream** the spec keeps
in both stages (§7). `StructStream._score` (`src/train_b0.py:4259–4400`) builds per-pair conditioning
for *every legal pair of a 40-node subgraph, on every rank, every optimizer step* — ~211k motif
compilations per rank per epoch, CPU work serialized with the GPU step, with no dense-lookup
shortcut for top-k witness *identities*. Measured struct arms already sit at **81.7 GiB/rank against
the 85 GiB cap** (`outputs/analysis/prefix_20260912/struct_comparison.json`): ~3 GiB of headroom.

## 2. Per-item findings

### 2.1 GRIT pair states — confirmed, no vendor edit

`encoder/grit_gmt.py:302–331` builds the stack with `norm_e = O_e = (index < layers - 1)`, and its
inline comment gives the spec's reason (an unused edge output is a DDP hard error). **Spec assertion
true.**

No `src/vendor/` change is needed. `GritTransformerLayer.forward`
(`vendor/grit_official/grit_layer.py:397–464`) computes `e = O_e(dropout(e_attn_out.flatten(1)))`,
adds the residual, applies `layer_norm1_e`, and writes `batch.edge_attr = e` whenever `cfg.update_e`
is true (it is, `_grit_layer_cfg()`). `O_e`/`norm_e` are **constructor flags**: the new adapter
passes `True` on the last layer and reads `flat.edge_attr` after the loop. Row `(u,v)` of graph `b`
is a gather — with all 26 slots valid (spec §2), `_flat_index` lays edges out row-major per graph,
so the offset is `b*676 + i_u*26 + i_v`. A fixed-26 adapter should precompute that index as a buffer
rather than call `_flat_index`, which does a `nonzero` + `.item()` host sync every forward.

`GritGmtEncoder`, `ENCODER_REGISTRY` and every `egostitch_e2e` checkpoint stay untouched.
**Size:** new `classifier/motif_reader.py`, ~250–300 lines, reusing `dense_rrwp` and `_GritBatch`.
`dense_rrwp` computes in `adj.dtype`, i.e. bf16 under autocast: the spec's fp32 requirement is new
code, not existing behaviour.

### 2.2 Prefix interface — confirmed

`prefix.py:31` `SITES = 3` × `cross_attn_layers: 3` = **9 sites**. `TopoPromptGenerator.__init__` sets
`slots = len(spec.fields) * slots_per_field` → **6 rows** per layer for three fields.
`gates = nn.Parameter(torch.zeros(...))`, and `prefix_branch` (`prefix.py:219–261`) applies
`tanh(gate)*gate_scale` and returns `F.linear(out, out_proj.weight)` **without bias**, added outside
dropout — exactly zero, base reproduced bit-for-bit. `TopoPromptGenerator.tokens` builds
`view_u`/`view_v` with `ROLE_SELF`/`ROLE_PARTNER` and a shared relation token;
`logits_from_standardized` feeds `(prefixes_b, prefixes_a)` to the BA pass and `abba_max`-combines —
exactly `[topo_self, topo_partner, topo_rel]`. All four sub-claims hold.

Consequence for cost: **the graph reader runs once per row, not once per orientation.**

**Registration.** `registry.py`/`composite.py` are *not* touchpoints: they serve `EgoStitchModel`
only, and `GraphEncoder` returns `GraphEmbedding(tokens, pooled)` with no edge channel
(`graph.py:112–138`), so the motif reader cannot live there. The real list:

| File | Change | Size |
|---|---|---|
| `classifier/motif_prompt.py` (new) | `V3_1MotifPrompt`; `TopoPromptCrossAttentionLayer` and `prefix_branch` reused verbatim | ~400 |
| `classifier/motif_gen.py` (new) | residue attention, 2-layer gate MP, typed gate heads | ~350 |
| `train_b0.py:152–155`, `:1119–1145`, `:6513–6600` | family lists, kwargs resolver, rows object, run-kind gate | ~145 |
| `train_b0.py:4299–4311` | `StructStream` branch | ~20 |
| `score_universe.py:1205, 1321, 3695–3735, 3839` | builder, `MODEL_BUILDERS`, oracle gate (Stage I only), family branch | ~60 |
| `eval/test_protocol.py` | help text only — it reads `model_family` off the checkpoint (`:217–235`) | ~2 |

Shared paths: `StructStream._score` and the `train_b0` family lists, both `isinstance` dispatch;
other arms are unaffected if the branch is additive.

### 2.3 Compiler and the data boundary

`data/struct_coords.py:224–311` is the reusable precedent: one table per universe from a loopless
graph, caching CSR `adjacency`, `degree`, `index` and dense products, with the queried-edge-removal
rule the spec repeats. `train_b0.py:6534` already passes `build_training_graph()` and
`build_g_val_simple()`.

What does not carry over: the compiler needs witness *identities* and truncation, not counts. The
left-candidate score `Σ_j A[i,j]/√(d_i d_j)` over `j ∈ N(v)` is the `(i,v)` entry of
`D^{-1/2}AD^{-1/2}·A` — one more 7203² fp32 product (208 MB) beside the existing ones, after which
per-pair work is `O(d_u + d_v + 64)`. Positives follow the existing `_exact_pair` recomputation.

**Row-id slicing extends cleanly.** `TopoPromptRows._attach` (`:3863`) does
`table.index_select(0, batch["_row_id"])`; a fixed 96-slot template makes each row a fixed 96-vector,
so the same call works unchanged.

**Memory — the brief's premise is off by an order of magnitude.** `TopoPromptRows` is built over
`corpus.pairs`, which `build_training_corpus` (`data/training_sampler.py:62–107`) defines as the
**union of unique rows over all epochs**. Measured on the real seed-42 split: **2,509,789 rows at 15
epochs, 3,916,450 at 25** — not 221,142. At 96 fp32 weights that is **964 MB/rank** (482 MB fp16),
×4 ranks, since every rank builds the table independently with no broadcast. RAM is fine; *build
time* is the risk: 2.51M top-k compilations at startup per rank, ~8 min at 0.2 ms/pair, ~84 min at
2 ms/pair. Cache it (fp16); per-epoch recomputation is 15× worse in total.

### 2.4 Cost

Measured, split-seed-42, 4×H20, 221,142 task pairs/epoch:

| Run | Train s/ep | Val s/ep | Peak GiB/rank |
|---|---|---|---|
| `b0_v31` (138 steps) | 102 | 186 | 59.7 |
| `prefix_base` (271 steps, bidirectional) | **316.6** | 240.4 | — |
| `struct_bce`/`grand`/`new` (+40-node struct) | 245–270 | 137–418 | **81.7** |
| `topo_prompt_full` (trunk trainable) | 490–507 | 352–362 | — |
| `coord_gen_full` (trunk + teacher frozen) | 387–403 | 372–382 | — |
| teacher (GRIT, egostitch_e2e) | 722 compute | 398 | 72.5 |

Derived (assumption → value; no *new* row is grounded in a profile):

| Term | Stage I | Stage II | Assumption |
|---|---|---|---|
| Task stream, frozen F | ~410 s | ~410 s | `coord_gen_full`/`prefix_base` = 1.23× |
| Struct stream | ~600 s | ~600 s | +150 s measured on a 102 s base, ×3.1 for the bidirectional trunk |
| GRIT-26, 1 fwd/row (I) vs 3 (II: student, `R_T(Â)` with grad, detached `R_T(A*)`) | +100–400 s | +300–900 s | **ungrounded**; 0.08 GFLOP/row vs the trunk's ~27 (0.3%), so launch/bandwidth-bound |
| Motif compile, struct stream | +40–400 s | +40–400 s | 211k compiles/rank/epoch at 0.2–2 ms; **ungrounded** |
| Generator (16 queries/endpoint, 2 MP layers @96) | — | <20 s | FLOP-negligible |
| **Train total** | **1,150–1,800 s** | **1,350–2,300 s** | |
| **Validation** | 350–450 s | ~380 s | V_val cascade + per-universe compile |
| **Peak GiB/rank** | **80–85** | **80–85** | struct arms measured 81.7; GRIT edge activations (~0.3–0.6 GB/chunk) live only inside `checkpoint(...)` |

15 epochs ≈ **6.5–9.5 h** (I) and **7.5–11 h** (II) per lane, plus 8–84 min startup and the test pass.

**Dominant term: the structural stream — ~600 s/epoch, about half the train wall, and +22 GiB of
peak (59.7 → 81.7).** The 26-node graph *is* cheap in FLOPs, so the spec's implicit assumption
survives at the task micro-batch; but it is multiplied by 780 pairs per optimizer step in the struct
stream, and at production micro-batch the memory ceiling, not the arithmetic, binds.
`max_pairs_per_rank` must drop from 4096/2048 toward the 1536 of `virtual_prompt_d_rev1.yaml`.

The brief's "arm-D ≈ 50 min/epoch at 61.8 GiB" could not be grounded: no `coord_gen_v2_d` profile
exists in this checkout, and 61.8 GiB appears only as prose in the 2026-09-15 spec (`:43`). The
measured struct-arm peak is higher, so 61.8 must not be used as the Stage II ceiling.

### 2.5 Structural-stream interaction

A new family must supply: an `isinstance` branch at `:4299–4311` materialising per-pair conditioning
for **all** legal subgraph pairs on **every** rank (the tensor is sliced by chunk only afterwards,
`:4323`); a forward taking `{emb_a, emb_b, len_a, len_b, …}` and returning `logits`; a trainable
parameter for the empty-rank dependency (`:4378`). Everything runs under
`checkpoint(..., use_reentrant=False)`, so each chunk's GRIT forward executes twice. So yes — every
pair in the sampled 40-node subgraph needs its own compiled motif graph every step. That is the term
quantified above and the reason to defer the struct block (§4).

### 2.6 Landmines

1. **Precision validation is a no-op here.** `validate_score_precision` (`score_universe.py:436–449`)
   returns immediately unless `model_family == "egostitch_e2e"`, so a bf16-contaminated
   `v3_1_motif_prompt` artifact passes `validate_artifact_precision` silently. The only defence is
   the spec's fp32 rule inside the adapter — which `dense_rrwp` does not implement. Same class as the
   `assemble.py` fp32 islands: promote *before* the degree division and the walk products.
2. **`allow_cache_subset=True`** is live at `score_universe.py:2481`. The arm adds no new node set, so
   it is not newly exposed, but any diagnostic scoring a different universe inherits the silent gather.
3. **Packed manifests** depend on `index.json` insertion order and are digested by `sha256_file`
   (`packed_features.py:241, 567`). Reuse `outputs/feature_packs/b0_v31_bf16`; do not regenerate.
4. **Self-pairs.** Test carries 1,891 self rows of 64,038, assembling as self-loops. The spec gives
   self rows an *empty* Stage I template but no Stage II self flag — yet `N(u) ∩ N(u) = N(u)` would
   make the closure family *full*. "Empty" is an explicit override to implement and report as a
   stage mismatch, not an assumption.
5. **V_val gating.** Stage I compiles V_val/test templates from truth and must inherit the
   `run_kind == "diagnostic"` refusal (`train_b0.py:6520`) and `--allow-oracle-diagnostic`
   (`score_universe.py:3695–3710`). Stage II must **not** join `is_topo_prompt`, or it is marked
   non-formal.

## 3. Required spec corrections

1. **§9 misroutes the implementation.** `generator/` and `encoder/` are `EgoStitchModel` registry
   slots whose contracts (`ImaginedGraph` → `GraphEmbedding`) have no pair channel and no residue
   input. Every new module belongs in `classifier/`.
2. **§5 "enable that update".** The final edge state is already produced and written to
   `batch.edge_attr`; only the `O_e`/`norm_e` constructor flags are off. Nothing needs re-enabling in
   vendored code.
3. **§5 FP32.** The `1e-6` floor exists; fp32 promotion in `dense_rrwp` does not. Say "add".
4. **§3/§7 row count.** The cached corpus is the union over epochs — 2.51M rows at 15 epochs.
5. **AB/BA.** §6 is right that the swap is at token level; state explicitly that the reader runs once
   per row, since the interface makes the opposite reading easy.

## 4. Implementation order

1. **Motif reader alone** — no generator, no struct, task BCE, closure family only (8 witnesses, 16
   edges), templates cached for the task stream. Assert gates-off equals published `prefix_base`
   bit-for-bit, within-role permutation invariance, swap equivariance, gradient into every gate head.
   This proves the pair-state readout and the six-row interface in one run.
2. **Add the L3 bridge family** and the corruption schedule; measure per-pair compile time and one
   epoch's wall/peak with `--ddp-mode epoch-probe`. That number gates everything after.
3. **Add the struct stream**, retuning `max_pairs_per_rank` against the measured 81.7 GiB peak.
4. **Stage II generator**, reusing `V3_1CoordGen`'s frozen-teacher pattern (`coord_gen.py:376–383`)
   with `R_T` as the GRIT reader rather than a trunk copy.
5. **Controls** (direct six-row prefix, one weight per type, fixed training-mean graph) — config
   changes on the step-3 artifact, not new code.
