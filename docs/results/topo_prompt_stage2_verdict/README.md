# Stage II (`v3_1_coord_gen`): does the generator work? Combined verdict

Date: 2026-09-15, **revised the same day after an independent audit** (see §2 and §7). Subject: the
published `coord_gen_full` checkpoint (`4a743373f0b9e4d0`, selected epoch 6) of
[`topo_prompt_stage2.md`](../topo_prompt_stage2.md), headline split seed 42, training seed 0. This
note replaces the two separate audits it merges; they are Parts A and B below.

Two questions, two inputs, no re-scoring and no GPU in either — every number is recomputed on CPU
from artifacts already on disk:

| | Part A — what `ŝ_uv` **contains** | Part B — what `ŝ_uv` **causes** |
|---|---|---|
| Input | the three-universe probe dump [`coord_fit_universes.npz`](../topo_prompt_stage2_curves/coord_gen_full/coord_fit_universes.npz) (generator predictions and true coordinates on label-balanced 3,000 + 3,000 rows of the training graph, the V_val gold graph and the test graph), SHA-256 `09358ac7…83a49cf` | that run's own score artifacts for `none`, `mean`, `gates_off`, `mean_relation` on `val_topology`, `test_topology` and `test` |
| Produced by | [`coord_fit_universes.py`](../topo_prompt_stage2_curves/coord_fit_universes.py) on H20 | the ordinary intervention chain of 2026-09-13 |
| Analysed by | [`coord_fit_analysis.py`](../topo_prompt_stage2_curves/coord_fit_analysis.py), which prints Part A's tables verbatim | `python -m src.experiments.coord_prompt_mediation`, record [`coord_gen_full.json`](coord_gen_full.json) |

Code and artifacts of both, including the Part B score artifacts, are bundled in
[`stage2_analysis_bundle.zip`](stage2_analysis_bundle.zip).

```bash
# Part A (CPU, seconds). --stratum nonself is the stratum that speaks to distinct proteins.
PYTHONPATH=. python docs/results/topo_prompt_stage2_curves/coord_fit_analysis.py --stratum nonself
# Part B (CPU, ~8 min)
python -m src.experiments.coord_prompt_mediation \
  --run-dir outputs/split_seed42/coord_gen_full \
  --output docs/results/topo_prompt_stage2_verdict/coord_gen_full.json
```

---

## 1. Combined verdict

**The checkpoint carries some transferable structural signal. The evidence does not establish that
it reconstructs a broad set of topology statistics for distinct unseen proteins, and it does not
establish that statistic fidelity — rather than density calibration — causes the reported
assembled-graph improvement.** Both of the stronger readings in this note's first version were
overturned by the audit in §2 and are withdrawn.

What the two parts establish, stated at the strength the evidence supports:

1. **Between distinct proteins, coordinate fidelity is partial and uneven** (Part A). Applying this
   note's own descriptive criterion — within-label ρ ≥ 0.5 on *both* unseen universes — **zero of
   21 statistics** qualify once self-pairs are removed; the three that appeared to qualify were
   carried by the self-pair stratum, where Jaccard is one by construction. Several statistics do
   retain moderate test association on distinct pairs (nonself test ρ: triangles 0.65, Jaccard
   0.63, L3 density 0.60, clustering 0.59, L3 count 0.58, common neighbours 0.57). Their validation
   behaviour and their calibration are much weaker, so *partial signal transfer* is supported and
   *broad accurate statistics prediction* is not.
2. **The predicted coordinates change more than a threshold** (Part B). The reported topology
   improvement (test RD 0.907, geo 0.864, MMD 4.2 / 3.6 / 6.7 against `mean`'s 0.520 / 0.502 /
   8.9 / 7.9 / 12.9) is largely, but **not wholly**, reproduced by a single per-universe scalar.
   The scalar reproduces density and the degree and spectral ratios; it does not reproduce GS
   (0.399 against 0.430), the clustering ratio (3.97 against 3.56), or any ranking metric — a
   constant cannot move AUROC, yet predicted 0.6771 > `mean` 0.6651 > `gates_off` 0.6467.
3. **A large part of the reported topology gain is threshold placement.** `Δ = logit_none −
   logit_mean` has mean −2.904 on V_val and −2.243 on test; that 0.661-nat differential alone moves
   the transferred operating point, and the offset-only surrogate — which keeps `mean`'s ranking
   exactly — lands at RD 1.089. Density is itself part of topology quality, so explaining a benefit
   through calibration does not make it fictitious; it does mean the benefit is not evidence of
   pair-level topology conditioning.
4. **The costs are real and single-seed.** The full branch loses edge ranking against both
   independent baselines (AUROC 0.677 against B0's 0.701 and prefix_base's 0.721) and does not
   improve GS against B0 (0.430 both).

**What remains open, and is the actual gap:** no density-matched row exists for B0 or prefix_base,
so none of these diagnostics compares shape against an *independently trained* baseline at a common
density. `mean` and `gates_off` intervene on a reader that was trained on true coordinates; beating
them shows dependence on the prompt, not an improvement over a content-only model. §5 lists the
confirmatory runs.

## 2. Correction log — what the independent audit overturned

Findings of the audit of 2026-09-15, each verified here against the same dump before being applied.

| First-version claim | Status | What replaces it |
|---|---|---|
| "Three of 21 statistics hold within-label ρ ≥ 0.5 on both unseen universes" | **Overturned** | Self-pairs carry them. On the nonself stratum: **zero of 21**. Jaccard within-label ρ val 0.74 → **0.32**; `walk_2` 0.60 → **−0.04**; shared-shell fraction 0.69 → **0.29** (§A.1, §A.2) |
| "The only downstream effect is one scalar" | **Withdrawn** | The scalar reproduces density, degree and spectral ratios only. Ranking metrics differ (a constant cannot change them) and GS, clustering and mean abs log RD are not reproduced (§B.2) |
| "An oracle per-universe affine recalibration would leave R² = ρ²" | **Corrected** | The column squares a *within-label-centred* correlation; the ordinary affine fit on all rows uses the pooled Pearson squared. Neither bounds a nonlinear or multivariate readout. Column renamed (§A.4) |
| "The coordinates that fail worst are unnormalised counts (`walk_3`, `walk_4`…)" | **Corrected** | `walk_k` are normalised walk kernels `(S^k)_uv`, not raw path counts (§A.2) |
| "The frozen reader can of course learn around a consistent inversion — it was trained on these predictions" | **Factually wrong** | Stage II freezes every reader parameter and the reader was trained on **true** coordinates. The generator adapts to the frozen reader, not the reverse (§A.5) |
| "A quantity that generalises does not behave that way" (better on test than V_val) | **Softened** | The two universes may differ in task difficulty; the ordering is a fact to explain, not proof of non-generalisation (§A.2) |
| Effective dimension ≈ "the prompt carries about three numbers" | **Softened** | Entropy-based effective dimension is variance concentration, not algebraic rank nor a count of informative numbers; low ED does not prove task irrelevance (§A.6) |
| "The pair-level component is the part worth developing … its node-level component" | **Corrected** | The node and pair terms are components of the *reader's* output, not the generator's endpoint and relation heads. Endpoint coordinates can produce pair residuals and relation coordinates node-additive effects; no head-level conclusion follows (§B.5) |
| Predicted transplant result: "keeps RD near 0.9" | **Demoted to hypothesis** | A transplant preserves `P(S)`, not `E[F(X,S)]`: the reader consumes endpoints and coordinates jointly, so the mean logit may move. Measure it (§B.7) |
| Matched density described as comparing "like with like" | **Qualified** | The procedure re-runs the same threshold selector on test, matching geometric-mean RD near one — not per-subgraph edge counts. Macro RD and the density distribution still differ across rows (§B.1) |
| Negative R² read as "no useful information" | **Qualified** | The comparator is the evaluation subset's own true mean, which is not a deployable predictor; negative R² is a calibration statement, not an information one (§A.4) |
| Sign-inverted coordinates read as harming the decision | **Qualified** | A one-coordinate AUROC below 0.5 is a reversed *marginal* association; no causal claim about the multivariate nonlinear reader follows (§A.5) |

Two further limits of the audit itself, recorded because they bound what §1 can say: the auditor
reproduced Part A's 21 per-statistic Pearson and R² values on all three universes to a maximum
absolute discrepancy of 7.11e-15 against the supplied JSON, but Part B could not be independently
recomputed because the bundle then lacked the score artifacts (now included, §6), and the dump
carries no node identifiers, so node-disjointness, repeated-node weighting and node-cluster
uncertainty remain unchecked in Part A.

---

# Part A — What `ŝ_uv` contains

This part asks what §5.3 of `docs/03-experiments.md` and its Figure 5 leave open: the figure shows
a *fair* correlation for some coordinates, and the question is which of those correlations are real
graded predictions between distinct proteins, which are strata that make the statistic true by
construction, and what the reader actually receives.

## A.1 The self-pair stratum

Each universe holds 6,000 sampled rows. Self-pairs (`u == v`, the distance head's own fifth class,
whose four raw one-hot coordinates are all zero — both definitions checked against each other)
number **443 / 362 / 180** on train / V_val / test, and are almost entirely positive: 442 of 443
training self-pairs, and *all* V_val and test self-pairs.

For a non-isolated self-pair, Jaccard is one by construction, the shared-shell fraction is one on a
non-empty shell, and the normalised two-step walk becomes a return quantity. Recognising that two
token sets are identical is a far easier problem than inferring the neighbourhood relationship of
two distinct proteins, and because the stratum is label-pure, **label centring does not remove its
contribution**: self and non-self rows stay mixed inside the positive class.

| statistic | universe | all ρ | nonself ρ | all within-label ρ | nonself within-label ρ |
|---|---|---:|---:|---:|---:|
| `relation.jaccard` | val | 0.754 | 0.434 | 0.738 | **0.321** |
| `relation.jaccard` | test | 0.756 | 0.632 | 0.746 | 0.608 |
| `relation.walk_2` | val | 0.621 | **−0.251** | 0.604 | **−0.041** |
| `relation.walk_2` | test | 0.599 | **−0.185** | 0.605 | **−0.100** |
| `context.shell_shared_fraction` | val | 0.712 | 0.392 | 0.691 | **0.291** |
| `context.shell_shared_fraction` | test | 0.686 | 0.434 | 0.656 | 0.376 |

Self-interactions legitimately belong in the benchmark and are not discarded here; the nonself
stratum is simply the one that bears on the distinct-protein claim. Under this note's own
descriptive criterion — within-label ρ ≥ 0.5 on both unseen universes — **zero of the 21 statistics
qualify on the nonself stratum**. The criterion is descriptive, not a definition of generalisation.

## A.2 Which statistics transfer between distinct proteins

Nonself stratum, the numbers `--stratum nonself` prints:

| statistic | ρ val | ρ test | within-label ρ val | within-label ρ test | R² val | R² test |
|---|---:|---:|---:|---:|---:|---:|
| `endpoint.log1p_triangles` | 0.27 | 0.65 | 0.27 | 0.65 | −0.50 | −0.30 |
| `relation.jaccard` | 0.43 | 0.63 | 0.32 | 0.61 | −0.03 | **+0.28** |
| `context.l3_density` | 0.35 | 0.60 | 0.24 | 0.59 | −0.03 | **+0.30** |
| `endpoint.clustering` | 0.09 | 0.59 | 0.09 | 0.59 | −0.24 | **+0.33** |
| `relation.log1p_l3` | 0.45 | 0.58 | 0.34 | 0.56 | −0.09 | −1.15 |
| `relation.log1p_common` | 0.43 | 0.57 | 0.32 | 0.57 | **+0.13** | **+0.24** |
| `endpoint.log1p_mean_neighbor_degree` | 0.19 | 0.57 | 0.19 | 0.57 | −0.58 | −4.39 |
| `endpoint.log1p_degree` | 0.23 | 0.47 | 0.23 | 0.47 | −0.42 | −1.50 |
| `relation.walk_5` | 0.36 | 0.45 | 0.17 | 0.38 | −0.09 | +0.02 |
| `endpoint.return_2` | 0.18 | 0.44 | 0.18 | 0.44 | −0.11 | −7.99 |
| `context.shell_shared_fraction` | 0.39 | 0.43 | 0.29 | 0.38 | −0.18 | −1.34 |
| `context.log1p_hop1_union` | 0.19 | 0.38 | 0.22 | 0.40 | −0.46 | −3.83 |
| `relation.walk_4` | 0.31 | 0.35 | 0.16 | 0.29 | −0.60 | −1.18 |
| `endpoint.log1p_open_wedges` | 0.21 | 0.29 | 0.21 | 0.29 | −0.34 | −2.35 |
| `endpoint.return_3` | 0.06 | 0.27 | 0.06 | 0.27 | −0.39 | −0.83 |
| `context.log1p_cross_ring` | 0.32 | 0.16 | 0.22 | 0.14 | −0.37 | −5.30 |
| `endpoint.return_4` | 0.14 | 0.09 | 0.14 | 0.09 | −0.19 | −34.22 |
| `relation.walk_3` | −0.25 | 0.03 | −0.07 | −0.02 | −0.34 | −0.14 |
| `endpoint.log1p_two_hop` | 0.26 | 0.02 | 0.26 | 0.02 | −0.12 | −12.46 |
| `relation.walk_2` | −0.25 | −0.19 | −0.04 | −0.10 | −0.32 | −0.71 |
| `context.log1p_shell` | 0.21 | −0.16 | 0.20 | −0.17 | −0.12 | −28.68 |

Readings, at the strength the table supports:

- **Partial transfer is real.** Six statistics reach nonself test ρ ≥ 0.55, and five have positive
  test R² (clustering 0.33, L3 density 0.30, Jaccard 0.28, common neighbours 0.24, `walk_5` 0.02).
  Only common neighbours is positive on V_val (0.13).
- **It is uneven across universes.** The endpoint statistics that work at all are better predicted
  on the further universe (test 0.44–0.65) than on the nearer one (V_val 0.09–0.27). That ordering
  needs explaining — the two universes may differ in task difficulty, and V_val is much smaller —
  but it is not by itself proof of non-generalisation.
- **The walk kernels are the clearest failures**, and `walk_2` and `context.log1p_shell` are
  *negatively* correlated with the truth on distinct pairs. These are normalised walk kernels
  `(S^k)_uv` with `S = D^{-1/2} A D^{-1/2}`, not raw path counts, so "unnormalised counts fail" is
  not the explanation; the earlier version of this note said so and was wrong.

## A.3 How much of the combined-stratum correlation is label separation

On the combined stratum, because the probe rows are label-balanced, any statistic that separates
positives from negatives gains a free contribution to the pooled ρ, of size the label's η² on the
truth: ≈ 0 for every endpoint statistic (AUROC 0.48–0.50 in §A.5), ≤ 0.09 for the context size
statistics, and up to 0.53 for the relation family and the two shared-overlap context statistics.
Label-centring costs little on test and a great deal on V_val:

| statistic | pooled ρ val → test | within-label ρ val → test |
|---|---|---|
| `relation.jaccard` | 0.75 → 0.76 | 0.74 → 0.75 |
| `relation.log1p_common` | 0.47 → 0.57 | **0.32** → 0.56 |
| `relation.log1p_l3` | 0.44 → 0.58 | **0.33** → 0.56 |
| `context.l3_density` | 0.35 → 0.59 | **0.22** → 0.58 |
| `relation.walk_5` | 0.35 → 0.43 | **0.14** → 0.35 |

This quantifies §5.3's caveat ("part of every relation ρ is label separation") at up to 0.21 ρ on
V_val and ≤ 0.08 on test. It is a *separate* effect from the self-pair stratum of §A.1, which
label-centring cannot remove.

## A.4 Level and spread

The prediction keeps roughly the training scale in every universe while the truth's spread
collapses on the unseen graphs (combined stratum):

| statistic | sd(pred) train / val / test | sd(true) train / val / test | R² test |
|---|---|---|---:|
| `endpoint.return_4` | 0.17 / 0.14 / 0.19 | 0.87 / 0.27 / **0.05** | −33.85 |
| `context.log1p_shell` | 0.52 / 0.40 / 0.47 | 0.99 / 0.52 / **0.32** | −21.29 |
| `endpoint.log1p_two_hop` | 0.42 / 0.31 / 0.45 | 0.95 / 0.50 / **0.36** | −11.81 |
| `relation.walk_4` | 0.46 / 0.42 / 0.31 | 1.78 / 0.65 / **0.15** | −3.40 |
| `relation.jaccard` | 1.19 / 1.07 / 0.84 | 1.56 / 1.48 / 1.32 | **+0.51** |

On top sits a negative mean offset on test — −1.87 sd for `log1p_cross_ring`, −1.41 `log1p_shell`,
−1.23 `log1p_l3`, −1.14 `log1p_two_hop`, −1.01 shared shell, −0.97 `hop1_union` and `open_wedges` —
on the denser test graph, where the true statistics are themselves shifted. A coordinate can
therefore have ρ = 0.08 and R² = −33.9 at the same time.

Two qualifications on how to read this. **Negative R² compares against the evaluation subset's own
true mean**, which is not a predictor anything could deploy; it is a statement about calibration,
not about information content. And the tool's `within rho^2` column is *not* a recalibration
ceiling: it squares a within-label-centred correlation, while the best ordinary affine fit on all
rows would use the pooled Pearson squared, and neither bounds what a nonlinear or multivariate
readout could recover. The first version of this note called that column an "oracle recalibration
ceiling"; it is not one.

## A.5 What the reader receives: the label direction is degraded, sometimes reversed

Measuring each coordinate as a one-number classifier of the true label (combined stratum):

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
**marginally reversed**: the token's association with the label runs opposite to the truth's.

Two things follow and one does not. Stage II freezes every reader parameter, and that reader was
trained on **true** coordinates, so it cannot have learned to compensate for an inversion during
Stage II — the adaptation, if any, runs the other way, with the generator emitting whatever
satisfies the frozen reader. (The first version of this note asserted the opposite and was wrong.)
What does *not* follow is that the inversion harms the decision: an AUROC below 0.5 is a reversed
*marginal* association of one coordinate, and the reader consumes thirty of them jointly and
nonlinearly. The endpoint coordinates are uninformative about the label in truth as well as in
prediction, so nothing is lost there; they cannot be the source of any edge-level gain.

## A.6 Variance concentration of the prompt

| universe | source | PC1 | PC1–3 | effective dim | η²(PC1, label) |
|---|---|---:|---:|---:|---:|
| train | predicted | 0.762 | 0.963 | 2.38 | 0.132 |
| train | true | 0.467 | 0.716 | 6.83 | 0.400 |
| val | predicted | 0.652 | 0.952 | 2.97 | 0.014 |
| val | true | 0.503 | 0.717 | 6.34 | 0.457 |
| test | predicted | 0.730 | 0.971 | 2.43 | 0.010 |
| test | true | 0.522 | 0.776 | 5.20 | 0.282 |

(Nonself stratum; on the combined stratum the predicted effective dimension is 3.1–3.6 against the
truth's 5.6–7.8.) The predicted coordinate vector's variance is far more concentrated than the
truth's, and on the unseen universes its dominant axis is nearly unrelated to the label (η² 0.010
on test) while the truth's dominant axis is not (0.282).

This is **variance concentration, not algebraic rank**, and it is not a test of usefulness: a
low-variance direction can still carry the information a downstream reader uses, and η² measures
mean separation, not general dependence. Read it as a description of the prompt's geometry, not as
proof that the prompt is uninformative.

## A.7 The distance head

Nonself stratum, where the self class is trivially perfect (identical token sets) and is the only
reason the combined training accuracy clears its majority rate by as much as it does:

| universe | accuracy | majority | recall d=2 | recall d=3 | recall d≥4 | recall d=∞ | rows per true class |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 0.643 | 0.409 | 0.692 | 0.014 | 0.914 | 0.007 | 2142/578/2274/563 |
| val | **0.411** | 0.508 | 0.382 | 0.003 | 0.949 | 0.000 | 2862/1439/1288/49 |
| test | **0.301** | 0.666 | 0.430 | 0.003 | 0.901 | 0.000 | 3875/1851/91/3 |

Distance 3 is never recovered (recall ≤ 0.014 even on training rows) and ∞ is never predicted on
unseen rows. On both unseen universes the head is **worse than constant prediction of the majority
class**. No conclusion about the sign of its downstream effect follows from accuracy alone.

## A.8 The target distribution itself moves

From the probe's own metadata, the loopless graph each universe's coordinates are measured on:

| universe | nodes | edges | mean degree |
|---|---:|---:|---:|
| train | 7,203 | 31,623 | 8.78 |
| V_val | 869 | 4,703 | 10.82 |
| test | 2,018 | 30,128 | 29.86 |

The test graph is roughly three times denser than the training graph, so a target shift is
certainly present. That does not identify which of target shift, feature insufficiency, or
training-objective trade-offs dominates the prediction error; all three are consistent with these
tables.

Single-universe differences inside ±0.03 ρ are not read; every number in Part A is one probe seed
(seed 0) of one checkpoint, with no interval estimate.

---

# Part B — What `ŝ_uv` causes

## B.1 Method and its limits

The computation is `x → E → ŝ = g_ψ(·) → Π_ω(ŝ) → gated KV prefix → ℓ`, with a second path from
`E` straight into the frozen trunk. `mean` and `gates_off` already intervene on the prompt node;
what they lack is a decomposition of the effect they measure. On each universe, with
`Δ_i = logit_none_i − logit_mean_i`,

    Δ_i = offset + node_i + pair_i,

where `offset = mean(Δ)`, `node` is the mean-centred symmetric least-squares fit
`Δ_i ≈ c + a_u + a_v` (a self-pair loads its node twice), and `pair` is the residual. The three
parts are orthogonal by construction and sum to `Δ`. Each is priced by feeding
`logit_mean + <part>` through the unchanged protocol: select one threshold on V_val by the
geometric-RD five-rank rule, replay it on every test subgraph.

| Universe | rows | nodes | mean Δ | sd Δ | sd node | sd pair | node-additive R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `val_topology` | 297,209 | 869 | −2.904 | 1.925 | 1.097 | 1.582 | 0.324 |
| `test_topology` | 943,685 | 2,018 | −2.243 | 2.014 | 1.439 | 1.410 | 0.510 |

Three limits govern every row below.

- **The surrogates are diagnostics, never arms.** They are built from the run's own logits, the
  node effects are fitted on the universe they are evaluated on, and the offset constants require
  the evaluated collection. None is a deployable predictor under the strict single-pair contract.
- **The matched-density column is test-informed** and re-runs the same threshold selector on the
  test universe. It matches *geometric-mean RD near one*, not each subgraph's edge count: macro RD
  and the distribution of subgraph densities still differ between rows, so it is an approximate
  density control, not an exact one.
- **`mean` and `gates_off` are not content-only baselines.** They intervene on a reader trained
  with true coordinates. Beating them shows the prompt path is load-bearing inside *this* reader;
  it says nothing about an independently trained model.

## B.2 What the interventions establish

**The effect is not solely a constant.** AUROC / AUPRC on the held-out 1:1 test list are 0.6771 /
0.7251 for the prediction, 0.6651 / 0.7070 for `mean` and 0.6467 / 0.6952 for `gates_off`. A
constant logit shift cannot move a ranking metric, so the predicted coordinates change the ordering
of pairs, not only the operating point.

**A large part of the topology movement is nonetheless threshold placement.** `Δ`'s mean is −2.904
on V_val and −2.243 on test — the predicted coordinates depress V_val logits 0.661 nats more than
test logits — and because one threshold is selected on V_val and replayed on test, that
differential alone loosens the operating point. The offset-only surrogate adds that constant to
`mean`, leaving `mean`'s ranking of pairs exactly intact, and reaches RD 1.089 (geometric 1.027),
MMD 3.3 / 4.0 / 4.8. Its matched-density row is numerically identical to `mean`'s, as an unchanged
ranking must be.

**But it does not reproduce the whole measured effect.** Against the prediction it gives GS 0.399
versus 0.430, clustering ratio 3.97 versus 3.56, mean abs log RD 0.289 versus 0.264, and AUROC
0.6651 versus 0.6771. The honest summary is that one scalar accounts for the density and for the
degree and spectral ratios, and does not account for edge-set overlap, clustering shape, or
ranking.

**Some shape evidence survives approximate density control.** At matched density the prediction
improves GS by 0.029 and the clustering ratio by 0.383 over `mean`, while degree and spectral get
worse — mixed, and with no interval estimate; the ±0.01 GS / ±0.5 MMD band quoted throughout this
project is a stated convention, not an estimated uncertainty. More pointedly, replacing **only the
relation field** by its mean worsens all three ratios substantially even after density matching
(7.8 / 6.4 / 9.5 against the prediction's 4.0 / 3.5 / 6.0). That is evidence that relation-field
variation contributes to graph shape in this reader. It does not identify which statistic is
responsible, nor isolate it from interactions with the other coordinates.

The same point appears on V_val, where the rows already sit at similar macro RD: `coord_gen_full`
RD 1.064 with MMD 5.8 / 2.4 / 6.2, against `prefix_base` RD 1.102 with 10.6 / 4.6 / 9.0 and the full
reader's own `mean` intervention RD 1.102 with 10.4 / 4.2 / 7.9. Shape differences at comparable
density are worth investigating rather than attributing wholesale to the V_val→test mean shift.

## B.3 Rows

Transferred threshold (the reported protocol) on the left; the test-informed matched-density
diagnostic on the right.

| Row | thr | GS ↑ | RD → 1 | geo RD | mean abs log RD ↓ | MMD d/c/s ↓ | AUROC | matched GS | matched RD | matched MMD d/c/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| predicted (no intervention) | +1.359 | 0.430 | 0.907 | 0.864 | 0.264 | 4.2 / 3.6 / 6.7 | 0.677 | 0.428 | 1.058 | 4.0 / 3.5 / 6.0 |
| mean coordinates | +2.188 | 0.390 | 0.520 | 0.502 | 0.688 | 8.9 / 7.9 / 12.9 | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| gates off | +1.742 | 0.385 | 0.570 | 0.543 | 0.611 | 7.4 / 6.6 / 10.9 | 0.647 | 0.392 | 1.075 | 3.3 / 3.9 / 4.8 |
| mean relation field | +3.031 | 0.433 | 0.713 | 0.658 | 0.476 | 13.5 / 10.7 / 16.5 | 0.669 | 0.436 | 1.085 | 7.8 / 6.4 / 9.5 |
| offset only | −0.716 | 0.399 | 1.089 | 1.027 | 0.289 | 3.3 / 4.0 / 4.8 | 0.665 | 0.399 | 1.061 | 3.3 / 3.9 / 4.9 |
| offset + node | −0.368 | 0.386 | 1.734 | 1.551 | 0.562 | 11.8 / 9.9 / 9.9 | 0.623 | 0.387 | 1.104 | 11.5 / 9.4 / 11.8 |
| offset + pair | +1.179 | 0.376 | 0.656 | 0.595 | 0.556 | 6.2 / 5.4 / 9.7 | **0.693** | 0.390 | 1.097 | **3.4 / 3.4 / 4.4** |
| offset + permuted structure | +0.396 | 0.333 | 1.220 | 1.109 | 0.387 | 7.8 / 17.3 / 5.1 | 0.606 | 0.332 | 1.092 | 6.1 / 16.3 / 5.0 |

The four measured rows reproduce `topo_prompt_stage2.md` §2.2 exactly (`predicted` geo RD 0.864 /
mean abs log RD 0.264; `mean` 0.502 / 0.688), which is the check that this pipeline is the protocol
and not a re-implementation of it.

Note that no single surrogate dominates: the pair-residual surrogate has the best AUROC in the
table (0.693) and the best matched-density clustering and spectral ratios, yet its matched-density
GS is 0.390 against the prediction's 0.428. The best edge-ranking surrogate is not the best
assembled-graph-overlap surrogate.

## B.4 Per-universe mean logits

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
differential at all. The gap tracks the density result.

This refines the two-stage v2 note's §0.1.2 ("a scale or offset change cannot move GS or the MMD
ratios"). That is true of an offset applied to both universes alike — the V_val selection absorbs
it — and false of a *differential* offset between the selection universe and the test universe.

## B.5 What the decomposition does not imply

The `node` and `pair` terms are components of the **reader's nonlinear output**, not of the
generator's heads. The fitted node effect is not the endpoint field and the pair residual is not
the relation field: endpoint coordinates can produce pair-specific residuals, and relation
coordinates can induce node-additive logit effects. Nothing here says which generator head to keep,
change, or remove — the earlier version of this note drew exactly that conclusion and should not
have.

The decomposition also intervenes on the logit, not on `ŝ`. It prices parts of an effect the prompt
already had; it cannot say whether the generator's own output is pair-specific in a way the frozen
reader consumes, because the permutation row perturbs the logit distribution as well as the
assignment.

## B.6 What is not settled

- **No density-matched row exists for B0 or prefix_base.** Every matched-density comparison here is
  internal to the full reader, so shape performance against an independently trained baseline at a
  common density is simply not measured.
- **Reported operating points, for reference** (from `topo_prompt_stage2.md`, each row at its own
  validation-selected threshold; not recomputed here):

  | Row | AUROC | AUPRC | GS | RD | Degree / Clustering / Spectral MMD |
  |---|---:|---:|---:|---:|---|
  | B0 | 0.701 | 0.736 | 0.430 | 0.557 | 9.67 / 8.12 / 14.55 |
  | prefix_base | 0.7205 | 0.7441 | 0.409 | 0.456 | 11.4 / 9.7 / 16.8 |
  | `coord_gen_full` | 0.6771 | 0.7251 | 0.430 | 0.907 | 4.2 / 3.6 / 6.7 |
  | `coord_gen_frozen` | 0.7206 | 0.7505 | 0.415 | 0.516 | 8.8 / 7.8 / 13.2 |

  The full branch has a genuine reported improvement in density and in all three MMD ratios, loses
  edge ranking against both independent baselines, and does not improve GS against B0. Single seed.
- **No uncertainty estimate** accompanies any row: one checkpoint, one seed, one permutation, and a
  conventional noise band rather than an interval.

## B.7 The transplant run

The intervention that would test pair-specificity at the level of `ŝ` is a **coordinate
transplant**, `do(ŝ_i ← ŝ_σ(i))` for one seeded permutation of the universe. It was previously
refused for this family on the reading that a self-predicting model "has no null"; that gate was
wrong and was removed (`src/score_universe.py`, `_coord_gen_source_coords`).

What it does and does not guarantee: a transplant preserves the universe's *marginal* distribution
of predicted coordinates `P(S)`. It does **not** guarantee that the mean logit is preserved, because
the reader consumes endpoints and coordinates jointly — changing `P(X, S)` while holding `P(S)` can
move `E[F(X, S)]`. The first version of this note predicted that the transplant "keeps RD near 0.9";
that is a hypothesis, not a consequence of the intervention. Measure the output mean shift rather
than assuming it, stratify by self/nonself status (§A.1), permute the whole coordinate vector
including the distance outputs, and compare at controlled density as well as at the transferred
threshold.

```bash
# 4-GPU container, ~25 min per universe chain; diagnostic attribution, selects nothing.
hpc/run.sh test --checkpoint outputs/split_seed42/coord_gen_full/best.pt \
  --output-dir outputs/split_seed42/coord_gen_full/intervention_shuffle \
  --data-root data --strategy breadth_first --arm coord_gen_full_shuffle --seed 0 \
  --prefix-intervention shuffle --prefix-intervention-seed 0
```

---

## 3. Comparison with L3-PPI

Taken from the independent audit's reading of Gao et al., *Learning the Interaction Prior for
Protein-Protein Interaction Prediction: A Model-Agnostic Approach* (arXiv:2605.09964v1), Eq. (8),
§5.5 and Figure 5B; not independently re-derived here.

L3-PPI's Eq. (8) regularises *activated virtual-path counts according to the binary label*; it does
not fit each pair's numerical observed L3 count with a coordinate-regression loss. Figure 5B sets
the path budget `K` to 50 while the actual counts on the x-axis reach roughly 1,000, and the
displayed Pearson values are 0.65 / 0.57 / 0.53 for its BS / ES / NS strata of the Random partition.
That is evidence for a useful *associated surrogate*, not for calibrated equality to true path
counts, and its prediction benchmarks and metrics differ from this project's assembled-graph
evaluation.

The precedent therefore cuts in a specific direction: a low-dimensional structural surrogate can be
useful without being an accurate reconstruction. It does not establish that precise multivariate
statistics reconstruction is required, that either system achieves it, or that the resulting
assembled networks preserve degree, clustering and spectral structure. Pearson values are not
comparable across datasets, splits, and raw versus log-transformed targets.

## 4. Two statements in `docs/03-experiments.md` §5.3 that these numbers contradict

Not yet applied to that file:

- *"Return probabilities and walk 3 are unpredictable even on training rows (ρ ≤ 0.2)."*
  `endpoint.return_2` has ρ 0.36 on train and 0.43 on test (0.37 / 0.44 nonself). The claim holds
  for `return_3`, `return_4` and `walk_3` only.
- *"the predictions are compressed towards the training mean (about half the true spread)."* True on
  train and V_val; on test several coordinates are **over**-dispersed relative to that universe's
  truth (sd ratio 3.79 `return_4`, 2.12 `walk_4`, 1.97 `return_2`, 1.83 `walk_2`). The accurate
  statement is that the prediction spread stays near the training scale while the unseen universes'
  true spread shrinks.

To these the self-pair stratum adds a third: any per-coordinate correlation reported for this arm
should state its stratum, because the combined-stratum figure for Jaccard, shared-shell fraction
and `walk_2` is substantially an artefact of rows where `u == v`.

## 5. Evidence needed to establish the proposed mechanism

The gap is the link between statistic fidelity and improved assembled structure *against an
independently trained baseline*. A compact confirmatory set:

1. **Common-protocol baseline comparison.** B0 and prefix_base under frozen validation thresholds
   as the formal result, plus a separate graph diagnostic at equal per-subgraph edge count. Report
   GS and all three MMD statistics, with repeated training seeds and appropriate treatment of
   overlapping sampled subgraphs.
2. **A coordinate transplant through the actual reader** (§B.7), stratified by self/nonself, using
   one consistent permutation of the entire coordinate vector including the distance outputs,
   inspecting the output mean shift rather than assuming it, and compared at controlled density.
   This tests endpoint–coordinate alignment, not structural semantics alone.
3. **Field-specific oracle replacement, or a pre-declared interpolation from predicted to true
   coordinates**, used only for diagnosis: does improved coordinate fidelity improve the specific
   graph metrics at controlled density? This tests the missing step of the causal chain directly.
4. **Method changes Part A points at**, each as a cheap ablation rather than a conclusion: per-universe
   standardisation of the coordinate *target* (predicting a within-universe rank or z-score rather
   than a training-scale value); a label-balanced or negative-reweighted coordinate loss, since the
   Huber term is dominated by the positives' larger dynamic range; and retargeting the distance head
   as `d = 2` versus `d > 2`, since as a five-class head it is worse than a majority constant on
   both unseen universes.

Truth-informed diagnostics stay out of model selection and out of deployment claims, and
architecture or checkpoints must not be chosen by repeatedly optimising these held-out test
diagnostics. Replication follows decisions made on validation.

## 6. Reproduction and sources

```bash
PYTHONPATH=. python docs/results/topo_prompt_stage2_curves/coord_fit_analysis.py \
  --npz docs/results/topo_prompt_stage2_curves/coord_gen_full/coord_fit_universes.npz \
  --stratum {all,self,nonself}                      # Part A, CPU, seconds
python -m src.experiments.coord_prompt_mediation \
  --run-dir outputs/split_seed42/coord_gen_full \
  --output docs/results/topo_prompt_stage2_verdict/coord_gen_full.json   # Part B, CPU, ~8 min
```

`stage2_analysis_bundle.zip` carries both analyses' code, the Part A dump
(`coord_fit_universes.npz`, SHA-256 `09358ac7386483bcd45806deb00ff51eb357a6b332c5d1cb5eea4305983a49cf`),
the Part A record JSON, the Part B record JSON, **and the Part B per-pair score artifacts** for all
four intervention rows on `val_topology`, `test_topology` and `test` — which the first version of
the bundle omitted, making Part B impossible to recompute independently. Rerunning Part B from the
bundle still requires the benchmark data package (`test_graph.pkl`, `test_node_buckets.pkl`, and
the V_val substrate), which is not redistributed here; node identifiers in the score artifacts are
the benchmark's own anonymised `node_XXXXXX` labels.

What is *not* reproducible from any of this: the original GPU inference, and the graph-coordinate
extraction that produced the dump. Both parts re-analyse saved predictions.

Sources for the numbers in this note:

- Part A: recomputed here from the dump named above (all 21 statistics, all three universes, all
  three strata).
- Part B: recomputed here from the run's score artifacts by
  `src/experiments/coord_prompt_mediation.py`.
- Reported baseline and Stage II operating points in §B.6: copied from
  [`topo_prompt_stage2.md`](../topo_prompt_stage2.md) (original run commit `aeb04d0`), not
  recomputed.
- §3: the independent audit's reading of arXiv:2605.09964v1, not re-derived here.
- Independent audit of 2026-09-15, whose findings are applied throughout and logged in §2. It
  reproduced Part A's per-statistic Pearson and R² on all three universes to within 7.11e-15 of the
  supplied JSON, working from the NPZ alone without this repository or a checkpoint.
