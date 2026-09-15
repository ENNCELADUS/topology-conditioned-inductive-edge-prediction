# Stage II (`v3_1_coord_gen`): does the generator work? Combined verdict

Date: 2026-09-15. Subject: the published `coord_gen_full` checkpoint (`4a743373f0b9e4d0`, selected
epoch 6) of [`topo_prompt_stage2.md`](../topo_prompt_stage2.md), headline split seed 42, training
seed 0. This note replaces the two separate audits it merges (the per-statistic generalisation
audit and the causal decomposition); they are Parts A and B below.

Two independent questions, two independent inputs, no re-scoring and no GPU in either — every
number is recomputed on CPU from artifacts already on disk:

| | Part A — what `ŝ_uv` **contains** | Part B — what `ŝ_uv` **causes** |
|---|---|---|
| Input | the three-universe probe dump [`coord_fit_universes.npz`](../topo_prompt_stage2_curves/coord_gen_full/coord_fit_universes.npz) (generator predictions and true coordinates on label-balanced 3,000 + 3,000 rows of the training graph, the V_val gold graph and the test graph) | that run's own score artifacts for `none`, `mean`, `gates_off`, `mean_relation` on `val_topology`, `test_topology` and `test` |
| Produced by | [`coord_fit_universes.py`](../topo_prompt_stage2_curves/coord_fit_universes.py) on H20 | the ordinary intervention chain of 2026-09-13 |
| Analysed by | [`coord_fit_analysis.py`](../topo_prompt_stage2_curves/coord_fit_analysis.py), which prints Part A's tables verbatim | `python -m src.experiments.coord_prompt_mediation`, record [`coord_gen_full.json`](coord_gen_full.json) |

Code and artifacts of both are bundled in [`stage2_analysis_bundle.zip`](stage2_analysis_bundle.zip).

```bash
# Part A (CPU, seconds)
python docs/results/topo_prompt_stage2_curves/coord_fit_analysis.py
# Part B (CPU, ~8 min)
python -m src.experiments.coord_prompt_mediation \
  --run-dir outputs/split_seed42/coord_gen_full \
  --output docs/results/topo_prompt_stage2_verdict/coord_gen_full.json
```

---

## 1. Combined verdict

**The generator learns a narrow, low-rank, positive-class-biased signal, and the one effect of that
signal on the reported result is a single scalar.** The two analyses meet exactly:

- Part A: the 30-number prompt carries about **three effective dimensions** (3.1–3.6 against the
  truth's 5.6–7.8), only **three of 21 statistics** survive on both unseen universes, the
  correlation lives almost entirely in the positive class (within-positive ρ 0.66–0.77 against
  within-negative 0.15–0.33), every informative coordinate **loses 0.11–0.30 AUROC** for the edge
  label relative to the truth, and the walk coordinates are **sign-inverted** (true `walk_2` AUROC
  0.92/0.89, predicted 0.35/0.41).
- Part B: the published topology headline (test RD 0.907, geo 0.864, MMD 4.2 / 3.6 / 6.7) is
  reproduced and overshot by **one scalar per universe** — `logit_mean + mean(Δ)` gives RD 1.089,
  geo 1.027, MMD 3.3 / 4.0 / 4.8 — and at matched density the shape advantage disappears entirely.

A prompt with three effective degrees of freedom, whose within-class variation sits in the
positives and whose dominant axis on test is orthogonal to the label (η² 0.007), is precisely the
kind of signal whose downstream effect one per-universe scalar can reproduce. The two findings are
not two problems; they are the same problem measured from the input side and the output side.

**What is therefore not supported:** the Stage II note's reading that `coord_gen_full` delivers
"the topology-conditioned decision the pipeline was built for". What *is* supported: the predicted
coordinates shift the student's confidence differentially between the threshold-selection universe
and the test universe, and that shift lands test density near the reference. That was obtained
without any truth graph and is in that sense real and deployable; it is also one number, one seed,
and a favourable sign that nothing in the method controls.

**What the generator does contribute,** and where effort should go: ≈ +0.03 GS (three times the
noise band), a lower clustering ratio, and — in its *pair-specific* component alone — the best edge
ranking of any row measured (AUROC 0.693 against 0.677 for the real model and 0.665 for `mean`).
The three coordinates that survive are all normalised pair-overlap quantities. The node-level
component buys the illusory density effect and costs ranking (0.623).

---

# Part A — What `ŝ_uv` contains

This part asks what §5.3 of `docs/03-experiments.md` and its Figure 5 leave open: the figure shows
a *fair* correlation for some coordinates, and the question is which of those correlations are real
graded predictions, which are an artefact of the label-balanced sample, and what the reader
actually receives through the prompt.

Four findings, in descending order of consequence for the arm:

1. **Only three of 21 statistics hold a within-label correlation ≥ 0.5 on *both* unseen
   universes:** `relation.jaccard`, `relation.walk_2`, `context.shell_shared_fraction`. Ten more
   are moderate on test and weak on V_val; the remaining eight are weak or dead on test.
2. **The correlation is carried by the positive class.** On test, within-positive ρ is 0.66–0.77
   for the relation and context pair statistics while within-negative ρ is 0.15–0.33 — and −0.07
   for `walk_2`. Negative rows, five sixths of any deployment batch, receive a nearly flat
   prediction.
3. **Several predicted coordinates are anti-informative about the label.** True `walk_2` has AUROC
   0.92 / 0.89 (V_val / test) for the edge label; the *predicted* `walk_2` has 0.35 / 0.41 — it is
   inverted. Every relation and context coordinate loses label information (jaccard 0.92 → 0.77 and
   0.88 → 0.65).
4. **The 30-number prompt carries about three numbers.** The predicted coordinate matrix has PC1 at
   0.55 / 0.43 / 0.64 of its variance and PC1–3 at 0.91–0.92, effective dimension 3.1–3.6, against
   5.6–7.8 for the truth.

## A.1 Is the reported correlation just positive-versus-negative separation?

Mostly not, except on V_val. Because the probe rows are label-balanced, any statistic that
separates positives from negatives gets a free contribution to the pooled ρ. The size of that
contribution is the label's η² on the truth: **≈ 0 for every endpoint statistic** (they do not
separate classes at all — AUROC 0.48–0.50 in §A.4), ≤ 0.09 for the context size statistics on the
unseen universes, and **as much as 0.53 for the relation family** and for the two shared-overlap
context statistics (`shell_shared_fraction`, `l3_density`, η² 0.21–0.39).

Removing it (label-centring both sides before correlating) costs little on test but a lot on V_val:

| statistic | pooled ρ val → test | within-label ρ val → test |
|---|---|---|
| `relation.jaccard` | 0.75 → 0.76 | 0.74 → 0.75 |
| `relation.log1p_common` | 0.47 → 0.57 | **0.32** → 0.56 |
| `relation.log1p_l3` | 0.44 → 0.58 | **0.33** → 0.56 |
| `context.l3_density` | 0.35 → 0.59 | **0.22** → 0.58 |
| `relation.walk_5` | 0.35 → 0.43 | **0.14** → 0.35 |

So §5.3's caveat ("part of every relation ρ is label separation") is quantified: it is worth up to
0.21 ρ on V_val (`walk_5`) and ≤ 0.08 on test. All tables below report the within-label number.

## A.2 Which statistics transfer

Tiered by within-label ρ on the two unseen universes:

| tier | statistics | within-label ρ train / val / test |
|---|---|---|
| **holds on both** (≥ 0.5 everywhere unseen) | `relation.jaccard` | 0.70 / 0.74 / 0.75 |
| | `context.shell_shared_fraction` | 0.68 / 0.69 / 0.66 |
| | `relation.walk_2` | 0.57 / 0.60 / 0.60 |
| **test only** (≥ 0.35 test, ≤ 0.45 V_val) | `endpoint.log1p_triangles` | 0.74 / 0.27 / 0.65 |
| | `endpoint.clustering` | 0.50 / 0.10 / 0.58 |
| | `context.l3_density` | 0.56 / 0.22 / 0.58 |
| | `relation.log1p_common`, `relation.log1p_l3` | 0.69–0.73 / 0.32–0.33 / 0.56 |
| | `endpoint.log1p_mean_neighbor_degree` | 0.61 / 0.18 / 0.56 |
| | `context.log1p_hop1_union` | 0.71 / 0.45 / 0.50 |
| | `endpoint.log1p_degree` | 0.67 / 0.24 / 0.48 |
| | `endpoint.return_2`, `relation.walk_5` | 0.12–0.35 / 0.14–0.18 / 0.35–0.43 |
| **decays or dead** (< 0.35 on test) | `endpoint.log1p_open_wedges` | 0.62 / 0.22 / 0.30 |
| | `context.log1p_cross_ring` | 0.66 / 0.34 / 0.22 |
| | `context.log1p_shell` | 0.60 / 0.36 / **0.07** |
| | `endpoint.log1p_two_hop` | 0.50 / 0.26 / **0.04** |
| | `endpoint.return_3/4`, `relation.walk_3`, `relation.walk_4` | ≤ 0.34 / ≤ 0.33 / ≤ 0.28 |

Two readings matter here. First, the five endpoint statistics that work at all are *better*
predicted on the further universe (test 0.43–0.65) than on the nearer one (V_val 0.10–0.27). A
quantity that generalises does not behave that way; this pattern says the endpoint head tracks
each universe's degree profile rather than the queried node. Second, the three that hold on both
unseen universes are all normalised, ratio-like overlap quantities, while the coordinates that
fail worst are unnormalised counts (`walk_3`, `walk_4`, `log1p_shell`, `log1p_two_hop`) — the
ones whose true spread also shrinks most from one universe to the next (§A.3).

Within-class asymmetry on test (the positives carry the signal):

| statistic | ρ within positives | ρ within negatives |
|---|---:|---:|
| `relation.jaccard` | 0.77 | 0.21 |
| `context.shell_shared_fraction` | 0.74 | 0.15 |
| `relation.log1p_common` | 0.68 | 0.22 |
| `context.l3_density` | 0.66 | 0.22 |
| `relation.log1p_l3` | 0.66 | 0.33 |
| `relation.walk_2` | 0.61 | **−0.07** |
| endpoint family (all nine) | 0.05–0.67 | 0.02–0.62 (within 0.03 of the positives) |

## A.3 Why the level fails even where the ranking holds

An oracle per-universe affine recalibration would leave R² = ρ². On test that ceiling is 0.56 for
Jaccard, 0.43 shared shell, 0.42 triangles, 0.37 `walk_2`, 0.34 clustering and `l3_density`, 0.32
common neighbours and mean neighbour degree — and ≤ 0.10 for eight of the 21. So for most
coordinates the negative R² is **not** one recalibration away.

The mechanism is visible in the spreads: the prediction keeps roughly the training scale in every
universe, while the truth's spread collapses on the unseen graphs.

| statistic | sd(pred) train / val / test | sd(true) train / val / test | R² test |
|---|---|---|---:|
| `endpoint.return_4` | 0.17 / 0.14 / 0.19 | 0.87 / 0.27 / **0.05** | −33.85 |
| `context.log1p_shell` | 0.52 / 0.40 / 0.47 | 0.99 / 0.52 / **0.32** | −21.29 |
| `endpoint.log1p_two_hop` | 0.42 / 0.31 / 0.45 | 0.95 / 0.50 / **0.36** | −11.81 |
| `relation.walk_4` | 0.46 / 0.42 / 0.31 | 1.78 / 0.65 / **0.15** | −3.40 |
| `relation.jaccard` | 1.19 / 1.07 / 0.84 | 1.56 / 1.48 / 1.32 | **+0.51** |

On top of that sits a negative mean offset on test — −1.87 sd for `log1p_cross_ring`, −1.41
`log1p_shell`, −1.23 `log1p_l3`, −1.14 `log1p_two_hop`, −1.01 shared shell, −0.97 `hop1_union` and
`open_wedges` — on the denser test graph, where the true statistics are themselves shifted. A
coordinate can therefore have ρ = 0.08 and R² = −33.9 at the same time.

## A.4 What the reader actually receives: the label direction is degraded, sometimes inverted

The prompt's job is to tell the frozen reader something about the queried pair. Measuring each
coordinate as a one-number classifier of the true label isolates exactly that:

| statistic | val: true → predicted | test: true → predicted |
|---|---|---|
| `relation.walk_2` | 0.923 → **0.347** | 0.891 → **0.415** |
| `relation.walk_3` | 0.902 → **0.368** | 0.923 → 0.515 |
| `relation.jaccard` | 0.919 → 0.773 | 0.883 → 0.654 |
| `relation.walk_4` | 0.929 → 0.729 | 0.907 → 0.625 |
| `context.l3_density` | 0.889 → 0.728 | 0.907 → 0.603 |
| `relation.log1p_common` | 0.908 → 0.737 | 0.859 → 0.605 |
| `context.shell_shared_fraction` | 0.863 → 0.749 | 0.768 → 0.629 |
| `context.log1p_hop1_union` | 0.375 → 0.517 | 0.380 → 0.483 |
| endpoint family (all nine) | 0.466–0.500 → 0.481–0.499 | 0.496–0.502 → 0.478–0.510 |

Every informative coordinate loses 0.11–0.30 AUROC in prediction, and the walk coordinates are
**sign-flipped**: the token says "fewer 2-walks" precisely where the true structure says "more".
The frozen reader can of course learn around a consistent inversion — it was trained on these
predictions — but as a statement about predicted topology this is the opposite of the truth, and it
is why the arm's edge metrics are flat while §5.3's per-coordinate ρ looks respectable. The endpoint
coordinates are uninformative about the label in truth as well as in prediction, so nothing is lost
there; they cannot be the source of any edge-level gain.

## A.5 Rank of the prompt

| universe | source | PC1 | PC1–3 | effective dim | η²(PC1, label) |
|---|---|---:|---:|---:|---:|
| train | predicted | 0.552 | 0.911 | 3.31 | 0.110 |
| train | true | 0.370 | 0.690 | 7.77 | 0.398 |
| val | predicted | 0.429 | 0.908 | 3.63 | 0.100 |
| val | true | 0.479 | 0.717 | 6.61 | 0.464 |
| test | predicted | 0.635 | 0.922 | 3.13 | 0.007 |
| test | true | 0.480 | 0.757 | 5.62 | 0.303 |

The generator compresses a 5.6–7.8-dimensional object into a 3.1–3.6-dimensional one. Worse, on
test its dominant axis is almost orthogonal to the label (η² 0.007) while the truth's dominant axis
is not (0.303): the one direction the prompt varies along most is the direction that carries the
least task information.

## A.6 The distance head is a near/far binary

| universe | accuracy | majority | recall d=2 | recall d=3 | recall d≥4 | recall d=∞ | recall self | rows per true class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | 0.669 | 0.379 | 0.692 | 0.014 | 0.914 | 0.007 | 1.000 | 2142/578/2274/563/443 |
| val | 0.447 | 0.477 | 0.382 | 0.003 | 0.949 | 0.000 | 1.000 | 2862/1439/1288/49/362 |
| test | 0.322 | 0.646 | 0.430 | 0.003 | 0.901 | 0.000 | 1.000 | 3875/1851/91/3/180 |

Distance 3 is never recovered (recall ≤ 0.014 even on training rows) and ∞ is never predicted on
unseen rows. Self-pairs are perfect in all three universes, which is trivial — the two token sets
are identical — and they are the only reason the training accuracy clears the majority rate by as
much as it does. On test, where 96% of rows are truly at distance 2 or 3, the head assigns ≥ 4 to
3,758 of 5,820 non-self rows: accuracy 0.322 against a 0.646 majority. The head is worse than
constant prediction on the universe that matters.

Single-universe differences inside ±0.03 ρ are not read; every number in Part A is one probe seed
(seed 0) of one checkpoint.

---

# Part B — What `ŝ_uv` causes

The published note establishes that the predicted coordinates *matter* (`mean` and `gates_off` move
every number). It does not establish *how* they matter.

## B.1 Method

The computation is `x → E → ŝ = g_ψ(·) → Π_ω(ŝ) → gated KV prefix → ℓ`, with a second path from
`E` straight into the frozen trunk. `mean` and `gates_off` are already interventions on the prompt
node; what they lack is a decomposition of the effect they measure. On each universe, with
`Δ_i = logit_none_i − logit_mean_i`,

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

## B.2 The two controls

1. **The offset surrogate.** `Δ`'s mean is −2.904 on V_val and −2.243 on test: the predicted
   coordinates depress V_val logits **0.661 nats more** than test logits. Since the reported
   operating point selects one threshold on V_val and replays it on test, that differential alone
   loosens the transferred threshold. Feeding `logit_mean + mean(Δ)` — no node structure, no pair
   structure, a constant on each universe — through the ordinary protocol gives RD **1.089**
   (geometric RD **1.027**) and MMD **3.3 / 4.0 / 4.8**: a better density and better degree and
   spectral ratios than the real prediction achieves (0.907, geometric 0.864, 4.2 / 3.6 / 6.7).
   The surrogate is a *constant* added to `mean`, so it leaves `mean`'s ranking of pairs completely
   intact: the two rows admit edges in exactly the same order and differ only in where the
   transferred threshold lands. The matched-density column confirms it — `offset only` and `mean`
   are numerically identical there (GS 0.399, RD 1.061, MMD 3.28 / 3.86 / 4.86). **All of the
   published RD and MMD movement is threshold placement, not a change in which edges the model
   prefers.**
2. **Matched density.** MMD ratios move strongly with density, so two rows at RD 0.52 and RD 0.91
   are not comparable. Re-selecting each row's threshold on the test universe itself (test-informed,
   a diagnostic only) puts every row at RD ≈ 1.06, and the published shape advantage disappears:
   predicted 4.0 / 3.5 / 6.0 against `mean`'s **3.3 / 3.9 / 4.9** and `gates_off`'s 3.3 / 3.9 / 4.8.
   The prediction is *worse* on degree and spectral, better on clustering, and better on GS
   (0.428 vs 0.399).

## B.3 Rows

Transferred threshold (the reported protocol) on the left; the test-informed matched-density
diagnostic on the right. Edge metrics are on the held-out 1:1 `test` list and are threshold-free,
so a constant offset leaves them unchanged (visible in the `offset only` row, which reproduces
`mean`'s AUROC exactly — a consistency check on the pipeline).

| Row | thr | GS ↑ | RD → 1 | geo RD | mean abs log RD ↓ | MMD d/c/s ↓ | AUROC | matched GS | matched RD | matched MMD d/c/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| predicted (no intervention) | +1.359 | 0.430 | 0.907 | 0.864 | 0.264 | 4.2 / 3.6 / 6.7 | 0.677 | 0.428 | 1.058 | 4.0 / 3.5 / 6.0 |
| mean coordinates | +2.188 | 0.390 | 0.520 | 0.502 | 0.688 | 8.9 / 7.9 / 12.9 | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| gates off | +1.742 | 0.385 | 0.570 | 0.543 | 0.611 | 7.4 / 6.6 / 10.9 | 0.647 | 0.392 | 1.075 | 3.3 / 3.9 / 4.8 |
| mean relation field | +3.031 | 0.433 | 0.713 | 0.658 | 0.476 | 13.5 / 10.7 / 16.5 | 0.669 | 0.436 | 1.085 | 7.8 / 6.4 / 9.5 |
| **offset only** | −0.716 | 0.399 | **1.089** | **1.027** | 0.289 | **3.3 / 4.0 / 4.8** | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| offset + node | −0.368 | 0.386 | 1.734 | 1.551 | 0.562 | 11.8 / 9.9 / 9.9 | 0.623 | 0.387 | 1.104 | 11.5 / 9.4 / 11.8 |
| offset + pair | +1.179 | 0.376 | 0.656 | 0.595 | 0.556 | 6.2 / 5.4 / 9.7 | **0.693** | 0.390 | 1.097 | **3.4 / 3.4 / 4.4** |
| offset + permuted structure | +0.396 | 0.333 | 1.220 | 1.109 | 0.387 | 7.8 / 17.3 / 5.1 | 0.606 | 0.332 | 1.092 | 6.1 / 16.3 / 5.0 |

The four measured rows reproduce `topo_prompt_stage2.md` §2.2 exactly (`predicted` geo RD 0.864 /
mean abs log RD 0.264; `mean` 0.502 / 0.688), which is the check that this pipeline is the
protocol and not a re-implementation of it.

Reading:

- **offset only** reproduces the whole reported gain. Nothing in the remaining 100 % of `Δ`'s
  variance improves on it under the transferred threshold.
- **offset + node** overshoots density badly (RD 1.73) and is the worst shape row but one; the node
  component is not a clean density signal.
- **offset + pair** carries the edge signal: the best AUROC in the table, and at matched density the
  best clustering and spectral ratios. Pair-specific content is real; it just does not produce the
  density transfer the note attributes to it. Part A says what that content is: three normalised
  overlap coordinates, read mostly on positive rows.
- **offset + permuted structure** (the same values reassigned to random rows, seed 0) keeps a large
  density movement — the offset survives permutation — while clustering MMD blows up to 17.3. The
  assignment of structure to pairs matters for clustering and for ranking, and not for density.
  This is a logit-level proxy for a transplant; the true `do(ŝ_i ← ŝ_σ(i))` intervention is §B.5.

## B.4 Per-universe mean logits (the mechanism in one table)

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

## B.5 What Part B does not settle, and the run that would

The decomposition intervenes on the logit, not on `ŝ` itself: it prices the parts of the effect the
prompt already had. It cannot say whether the generator's own output is pair-specific in a way the
frozen reader consumes, because the reader is nonlinear and the permutation row perturbs the logit
distribution as well as the assignment.

The intervention that settles it is a **coordinate transplant**, `do(ŝ_i ← ŝ_σ(i))` for one seeded
permutation of the whole universe: it leaves the universe's distribution of predicted coordinates
— and therefore the offset — intact and destroys only the pairing. It was previously refused for
this family on the reading that a self-predicting model "has no null"; that gate was wrong and was
removed (`src/score_universe.py`, `_coord_gen_source_coords`). Prediction under this note's
reading: the transplant keeps RD near 0.9 and loses the GS and clustering advantages. Part A adds a
second, sharper prediction to test on the same run: because the predicted walk coordinates are
sign-inverted with respect to the label, the transplant should *improve* the rows those coordinates
dominate.

```bash
# 4-GPU container, ~25 min per universe chain; diagnostic attribution, selects nothing.
hpc/run.sh test --checkpoint outputs/split_seed42/coord_gen_full/best.pt \
  --output-dir outputs/split_seed42/coord_gen_full/intervention_shuffle \
  --data-root data --strategy breadth_first --arm coord_gen_full_shuffle --seed 0 \
  --prefix-intervention shuffle --prefix-intervention-seed 0
```

Single-seed result; the noise band (±0.01 GS, ±0.5 MMD ratio) governs every comparison in Part B.

---

## 2. Two statements in `docs/03-experiments.md` §5.3 that these numbers contradict

Not yet applied to that file:

- *"Return probabilities and walk 3 are unpredictable even on training rows (ρ ≤ 0.2)."*
  `endpoint.return_2` has ρ 0.36 on train and 0.43 on test. The claim holds for `return_3`,
  `return_4` and `walk_3` only.
- *"the predictions are compressed towards the training mean (about half the true spread)."* True on
  train and V_val; on test several coordinates are **over**-dispersed relative to that universe's
  truth (sd ratio 3.79 `return_4`, 2.12 `walk_4`, 1.97 `return_2`, 1.83 `walk_2`). The accurate
  statement is that the prediction spread stays near the training scale while the unseen universes'
  true spread shrinks — which is the same failure, but the figure's caption as written is wrong for
  the test panel.

## 3. Consequences for the write-up and for Stage III/IV

Reporting:

- The sentence "the best topology-preserving deployable row to date … that is the
  topology-conditioned decision the pipeline was built for" is not supported. What is supported:
  the predicted coordinates shift the student's confidence differentially between the selection
  and test universes, and that shift lands test density near the reference.
- Any claim that reads MMD ratios across rows at different relative densities is confounded.
  Report the matched-density column beside the transferred-threshold column, or compare only rows
  at comparable RD.
- The replication called for in `topo_prompt_stage2.md` §5 (seeds 1–2) should read the V_val−test
  logit gap as its primary quantity: it is the thing that has to replicate.

Method:

- The three surviving coordinates are all *normalised pair overlap* quantities. If the prompt is to
  be narrowed, narrow it to those rather than to a field; predicting raw counts across universes of
  different density is the part that is failing, and log1p does not fix it.
- Fixing the level is worth little on its own: the recalibration ceiling is ≤ 0.10 R² for eight of
  21 coordinates. Per-universe standardisation of the *target* (so the generator predicts a
  within-universe rank or z-score rather than a training-scale value) attacks the actual failure.
- The negative-class flatness is the sharpest deficiency for a deployable arm, and no current
  supervision term addresses it: coordinate Huber is dominated by the positives' larger dynamic
  range. A label-balanced or negative-reweighted coordinate loss is a cheap, targeted ablation.
- The distance head should either be dropped or retargeted as a binary d = 2 versus d > 2 head; as a
  five-class head it contributes a constant on unseen rows.
- The pair-level component is the part worth developing — it is the only one that improves ranking
  and clustering — and it is the part the current objective does not select for.
