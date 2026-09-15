# Is the generator effective? A causal decomposition of `ŝ_uv` (Stage II, `coord_gen_full`)

Date: 2026-09-15. Subject: the published `coord_gen_full` checkpoint (`4a743373f0b9e4d0`, selected
epoch 6) of `docs/results/topo_prompt_stage2.md`, headline split seed 42, training seed 0. Inputs:
that run's own saved score artifacts for `none`, `mean`, `gates_off` and `mean_relation` on
`val_topology`, `test_topology` and `test`. No re-scoring and no GPU: every number here is
recomputed on CPU from those artifacts with
`python -m src.experiments.coord_prompt_mediation`. Record:
[`topo_prompt_stage2_causal/coord_gen_full.json`](topo_prompt_stage2_causal/coord_gen_full.json).

The published note establishes that the predicted coordinates *matter* (`mean` and `gates_off`
move every number). It does not establish *how* they matter, and the claim it draws — "that is the
topology-conditioned decision the pipeline was built for" — turns out not to be supported by its
own artifacts.

## 1. Verdict

**`ŝ_uv` is causally effective, but its effect on the assembled test graph is one-dimensional.**
The headline topology result (test RD 0.907, MMD 4.2 / 3.6 / 6.7 against `mean`'s RD 0.520 and
8.9 / 7.9 / 12.9) is reproduced — and overshot — by replacing the whole prediction with **one scalar
per universe**: the mean logit shift it induces. Two controls carry the finding.

1. **The offset surrogate.** Write `Δ_i = logit_none_i − logit_mean_i`. Its mean is −2.904 on
   V_val and −2.243 on test: the predicted coordinates depress V_val logits **0.661 nats more**
   than test logits. Since the reported operating point selects one threshold on V_val and replays
   it on test, that differential alone loosens the transferred threshold. Feeding
   `logit_mean + mean(Δ)` — no node structure, no pair structure, a constant on each universe —
   through the ordinary protocol gives RD **1.089** and MMD **3.3 / 4.0 / 4.8**: a better density and
   better degree and spectral ratios than the real prediction achieves.
2. **Matched density.** MMD ratios move strongly with density, so two rows at RD 0.52 and RD 0.91
   are not comparable. Re-selecting each row's threshold on the test universe itself (test-informed,
   a diagnostic only) puts every row at RD ≈ 1.06, and the published shape advantage disappears:
   predicted 4.0 / 3.5 / 6.0 against `mean`'s **3.3 / 3.9 / 4.9** and `gates_off`'s 3.3 / 3.9 / 4.8.
   The prediction is *worse* on degree and spectral, better on clustering, and better on GS
   (0.428 vs 0.399).

So the mechanism is a V_val→test calibration difference, not per-pair topology conditioning. It
was obtained without any truth graph and is in that sense real and deployable; it is also a single
number, a single seed, and a favourable sign that nothing in the method controls.

**What the generator does contribute** is smaller and sits elsewhere than the note claims: about
+0.03 GS (edge-set overlap, three times the stated noise band), a lower clustering ratio, and — in
its *pair-specific* component alone — the best edge ranking of any row in the table (AUROC 0.693
against the real model's 0.677 and `mean`'s 0.665). Its node-level component buys the (illusory)
density effect and costs edge ranking (AUROC 0.623).

## 2. Method

The computation is `x → E → ŝ = g_ψ(·) → Π_ω(ŝ) → gated KV prefix → ℓ`, with a second path from
`E` straight into the frozen trunk. `mean` and `gates_off` are already interventions on the prompt
node; what they lack is a decomposition of the effect they measure. On each universe,

    Δ_i = offset + node_i + pair_i,

where `offset = mean(Δ)`, `node` is the mean-centred symmetric least-squares fit
`Δ_i ≈ c + a_u + a_v` (a self-pair loads its node twice), and `pair` is the residual. The three
parts are orthogonal by construction and sum to `Δ`. Each is priced by feeding
`logit_mean + <part>` through the unchanged protocol: select one threshold on V_val by the
geometric-RD five-rank rule, replay it on every test subgraph.

These surrogates are built from the run's own logits and the node effects are fitted on the very
universe they are evaluated on. **Every surrogate row is a diagnostic attribution, never a
deployable arm and never a reportable operating point.** The same applies to the matched-density
column, which is test-informed by construction.

| Universe | rows | nodes | mean Δ | sd Δ | sd node | sd pair | node-additive R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `val_topology` | 297,209 | 869 | −2.904 | 1.925 | 1.097 | 1.582 | 0.324 |
| `test_topology` | 943,685 | 2,018 | −2.243 | 2.014 | 1.439 | 1.410 | 0.510 |

Correlation of the fitted test node effect with the node's true test degree: **+0.239** — the node
component is only weakly a density knob, which is why forcing it through alone overshoots.

## 3. Rows

Transferred threshold (the reported protocol) on the left; the test-informed matched-density
diagnostic on the right. Edge metrics are on the held-out 1:1 `test` list and are threshold-free,
so a constant offset leaves them unchanged (visible in the `offset only` row, which reproduces
`mean`'s AUROC exactly — a consistency check on the pipeline).

| Row | thr | GS ↑ | RD → 1 | MMD d/c/s ↓ | AUROC | matched GS | matched RD | matched MMD d/c/s |
|---|---:|---:|---:|---|---:|---:|---:|---|
| predicted (no intervention) | +1.359 | 0.430 | 0.907 | 4.2 / 3.6 / 6.7 | 0.677 | 0.428 | 1.058 | 4.0 / 3.5 / 6.0 |
| mean coordinates | +2.188 | 0.390 | 0.520 | 8.9 / 7.9 / 12.9 | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| gates off | +1.742 | 0.385 | 0.570 | 7.4 / 6.6 / 10.9 | 0.647 | 0.392 | 1.075 | 3.3 / 3.9 / 4.8 |
| mean relation field | +3.031 | 0.433 | 0.713 | 13.5 / 10.7 / 16.5 | 0.669 | 0.436 | 1.085 | 7.8 / 6.4 / 9.5 |
| **offset only** | −0.716 | 0.399 | **1.089** | **3.3 / 4.0 / 4.8** | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| offset + node | −0.368 | 0.386 | 1.734 | 11.8 / 9.9 / 9.9 | 0.623 | 0.387 | 1.104 | 11.5 / 9.4 / 11.8 |
| offset + pair | +1.179 | 0.376 | 0.656 | 6.2 / 5.4 / 9.7 | **0.693** | 0.390 | 1.097 | **3.4 / 3.4 / 4.4** |
| offset + permuted structure | +0.396 | 0.333 | 1.220 | 7.8 / 17.3 / 5.1 | 0.606 | 0.332 | 1.092 | 6.1 / 16.3 / 5.0 |

Reading:

- **offset only** reproduces the whole reported gain. Nothing in the remaining 100 % of `Δ`'s
  variance improves on it under the transferred threshold.
- **offset + node** overshoots density badly (RD 1.73) and is the worst shape row but one; the node
  component is not a clean density signal.
- **offset + pair** carries the edge signal: the best AUROC in the table, and at matched density the
  best clustering and spectral ratios. Pair-specific content is real; it just does not produce the
  density transfer the note attributes to it.
- **offset + permuted structure** (the same values reassigned to random rows, seed 0) keeps a large
  density movement — the offset survives permutation — while clustering MMD blows up to 17.3. The
  assignment of structure to pairs matters for clustering and for ranking, and not for density.
  This is a logit-level proxy for a transplant; the true `do(ŝ_i ← ŝ_σ(i))` intervention is §5.

## 4. Per-universe mean logits (the mechanism in one table)

| Row | V_val mean | test mean | gap |
|---|---:|---:|---:|
| `coord_gen_full` predicted | −3.332 | −2.430 | **−0.902** |
| `coord_gen_full` mean | −0.428 | −0.187 | −0.241 |
| `coord_gen_full` gates_off | −1.404 | −1.174 | −0.230 |
| `coord_gen_full` mean_relation | −1.354 | −0.915 | −0.439 |
| `coord_gen_frozen` predicted | −3.888 | −3.676 | −0.212 |
| `coord_gen_frozen` mean | −3.511 | −3.386 | −0.125 |

Every ablated row sits near a −0.23 universe gap; only the full lane's predicted row opens it to
−0.90. The frozen lane, whose published topology row moves only by noise-band amounts, shows no
differential at all (−0.21 under its own predictions). The gap, not the structure, tracks the
result.

This also refines the two-stage v2 note's §0.1.2 ("a scale or offset change cannot move GS or the
MMD ratios"). That is true of an offset applied to both universes alike — the V_val selection
absorbs it — and false of a *differential* offset between the selection universe and the test
universe, which is exactly what is present here.

## 5. What this does not settle, and the run that would

The decomposition intervenes on the logit, not on `ŝ` itself: it prices the parts of the effect the
prompt already had. It cannot say whether the generator's own output is pair-specific in a way the
frozen reader consumes, because the reader is nonlinear and the permutation row perturbs the logit
distribution as well as the assignment.

The intervention that settles it is a **coordinate transplant**, `do(ŝ_i ← ŝ_σ(i))` for one seeded
permutation of the whole universe: it leaves the universe's distribution of predicted coordinates
— and therefore the offset — intact and destroys only the pairing. It was previously refused for
this family on the reading that a self-predicting model "has no null"; that gate was wrong and was
removed (`src/score_universe.py`, `_coord_gen_source_coords`). Prediction under this note's
reading: the transplant keeps RD near 0.9 and loses the GS and clustering advantages.

```bash
# 4-GPU container, ~25 min per universe chain; diagnostic attribution, selects nothing.
hpc/run.sh test --checkpoint outputs/split_seed42/coord_gen_full/best.pt \
  --output-dir outputs/split_seed42/coord_gen_full/intervention_shuffle \
  --data-root data --strategy breadth_first --arm coord_gen_full_shuffle --seed 0 \
  --prefix-intervention shuffle --prefix-intervention-seed 0
python -m src.experiments.coord_prompt_mediation \
  --run-dir outputs/split_seed42/coord_gen_full \
  --output docs/results/topo_prompt_stage2_causal/coord_gen_full.json
```

## 6. Consequences for the write-up

- The sentence "the best topology-preserving deployable row to date … that is the
  topology-conditioned decision the pipeline was built for" is not supported. What is supported:
  the predicted coordinates shift the student's confidence differentially between the selection
  and test universes, and that shift lands test density near the reference.
- Any claim that reads MMD ratios across rows at different relative densities is confounded.
  Report the matched-density column beside the transferred-threshold column, or compare only rows
  at comparable RD.
- The replication called for in `topo_prompt_stage2.md` §5 (seeds 1–2) should read the V_val−test
  logit gap as its primary quantity: it is the thing that has to replicate.
- The pair-level component is the part worth developing — it is the only one that improves ranking
  and clustering — and it is the part the current objective does not select for.

Single-seed result; the noise band (±0.01 GS, ±0.5 MMD ratio) governs every comparison above.
