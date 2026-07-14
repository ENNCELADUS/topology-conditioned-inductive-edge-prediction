# E2 Pair-to-Topology Gap: G1/G2/G3 Gate Results

**Experiment:** repository-local hardened E2 on Benchmark-A (`breadth_first`).

**Status:** G1, G2, and G3 are complete. The combined canonical G1 rerun completed on
2026-07-13 with frozen V3.1 B0 checkpoint `e092537d8cf1e208` and F0-MLP B0-alt
checkpoint `f3b8afc0b0781c43`, each scored over the same 2,037,171-row candidate
universe (2,035,153 non-self pairs plus 2,018 self-pairs). The mandatory O'Bray
perturbation check passed, so both graph-similarity composites are valid. B0-alt
preserves—and enlarges—the density-matched topology gap, closing G1's required
architecture-independence arm. G3 does not trigger the feature-insufficiency stop rule:
the parameter-free Oracle blend has substantial per-statistic headroom over B0.
EgoStitch can proceed to implementation.

### Latest checkpoint-only evaluation rerun

The aligned legacy checkpoint `pair_context_gated_abba_no_cross_s47_best_model_v3_1.pt`
was evaluated separately on the same `breadth_first` split in run
`legacy_v31_s47_20260712T193900Z`, completed 2026-07-12 19:53:13 UTC. This was a
test+G1/G2 rerun, not a new formal four-H20 training acceptance. It produced balanced
test AUROC/AUPRC `0.805170/0.818408`; G1 degree-corrected ratio-1
`0.799577/0.813319`, hard-heuristic `0.626746/0.663360`, and hard-feature
`0.510083/0.602131`; G2 reports `Ov(P)=0.579126` versus `Ov_min=0.010196`.
Its scores were rerun through the canonical evaluator on 2026-07-13. At relative density
`0.978392`, graph similarity is `2.63231e-7` and degree/clustering/spectral MMD ratios
are `13.8456/11.6277/19.9774`. The stronger edge scorer therefore does not shrink the
topology gap. The copied closeout package is
[`outputs/deliverables/g1_closeout_20260713/`](../../outputs/deliverables/g1_closeout_20260713/).

The final run confirms the pair-to-topology gap under the hardened protocol, but it
also exposes an important control result: the PA-null baseline is stronger than B0 in
some easy and feature-hard edge regimes. The claim is therefore a topology/robustness
failure of independent scoring, not a claim that this B0 checkpoint is uniformly strong.

## 1. G1 hardened-E2 result

### Edge-level regimes

The canonical balanced degree-corrected row is reported together with the two hard
negative regimes and the full-candidate imbalance view. All rows use 32,019 positives;
ratio-1 has 32,019 negatives and ratio-5 has 160,095 negatives.

| Scorer | Regime | AUROC | AUPRC | MCC |
|---|---|---:|---:|---:|
| B0 | degree-corrected, ratio-1 | 0.705519 | 0.730260 | 0.298489 |
| B0 | hard heuristic, ratio-1 | 0.583965 | 0.626649 | 0.172118 |
| B0 | hard feature, ratio-1 | 0.569560 | 0.617475 | 0.173379 |
| B0 | full candidate universe | 0.690627 | 0.134302 | 0.138845 |
| B0-alt | degree-corrected, ratio-1 | 0.693603 | 0.732509 | 0.319179 |
| B0-alt | hard heuristic, ratio-1 | 0.576711 | 0.623517 | 0.160893 |
| B0-alt | hard feature, ratio-1 | 0.467864 | 0.561339 | 0.094384 |
| B0-alt | full candidate universe | 0.745022 | 0.187195 | 0.193642 |
| PA-null | easy uniform, ratio-1 | 0.814711 | 0.824781 | 0.005302 |
| PA-null | hard feature, ratio-1 | 0.858646 | 0.869196 | 0.003226 |

The full regime table, including ratio-5 rows, is preserved in the combined
[`g1_tables.md`](../../outputs/runs/g1_b0_b0_alt_20260713T165714Z/g1_tables.md).

### Density-matched assembled graph

At the density-matched operating point (`threshold = 0.794385`), B0 assembles a graph
with relative density 0.997710 and a valid composite similarity of
`5.76802e-7` (higher is better; 1 means identical). The canonical topology metric is
the ratio of the size-mean raw biased MMD² to the deterministic real-vs-real reference
MMD². A ratio of `1` is the reference floor; lower is better.

| Component | Raw MMD² | Reference MMD² | MMD ratio |
|---|---:|---:|---:|
| Degree | 0.150219 | 0.0114874 | 13.0768 |
| Clustering | 0.139977 | 0.0117358 | 11.9273 |
| Spectral | 0.172248 | 0.00952007 | 18.0931 |

B0-alt uses its independently density-matched threshold `0.679179`. Its relative
density is `0.998739`, graph similarity is `2.29059e-8`, and all three canonical MMD
ratios are worse than B0:

| Component | Raw MMD² | Reference MMD² | MMD ratio |
|---|---:|---:|---:|
| Degree | 0.181850 | 0.0114874 | 15.8304 |
| Clustering | 0.158103 | 0.0117358 | 13.4718 |
| Spectral | 0.223468 | 0.00952007 | 23.4734 |

The assembled graph contains 2,018 predicted self-loops versus 1,891 reference
self-loops. The PA-null density-matched control has relative density 1.0,
degree/clustering/spectral MMD ratios of 30.1230/22.4118/37.8654, and composite
`8.18897e-14`.

The full threshold sweep is in the synced `g1_tables.md`; it shows that the gap is not
an artifact of one arbitrary threshold. The sweep includes the operating point and
the 50, 80, 90, 95, 97.5, 99, 99.5, and 99.9 probability percentiles.

### Composite validation and construction

- O'Bray degree-preserving-swap and uniform-rewire perturbation checks both passed;
  similarities decrease monotonically across the tested perturbation fractions.
- Noise-floor calibration, MMD configuration, threshold policy, negative construction,
  PA-null formula, and the breadth-first missing-feature note are recorded in
  `g1_results.json`.
- Degree heterogeneity is `sigma = 1.098430`; candidate positive rate is `0.0157174`.
- The scorer artifact contains 2,037,171 rows and was produced from the packed-feature
  pipeline on the four-H20 run.

## 2. G2 edge-independence ceiling

G2 evaluates Chanpuriya et al.'s exact identities on the cached soft score matrix,
excluding self-pairs from the dense probability matrix:

| Quantity | Value |
|---|---:|
| Test nodes | 2,018 |
| Non-self scored pairs | 2,035,153 |
| Simple reference edges | 30,128 |
| Reference triangles (`delta_star`) | 263,164 |
| Expected soft-edge volume `V(P)` | 123,373.982 |
| Measured overlap `Ov(P)` | 0.479886 |
| Minimum overlap `Ov_min` at matched volume | 0.054954 |
| Expected triangles `E[Delta]` | 1,342,118.384 |
| Triangle headroom (`E[Delta] / delta_star`) | 5.100 |
| Overlap headroom (`Ov / Ov_min`) | 8.733 |

Thus the measured soft scorer is well above the minimum overlap required by the
edge-independence ceiling to reach the reference triangle count. This is a feasibility
check, not evidence that the assembled graph is realistic: hard-thresholded assemblies
have `Ov = 1`, where the bound is vacuous, and the exact curve is the relevant object.
The complete curve is in [`g2_results.json`](../../outputs/e2_resubmit_retry/g2/g2_results.json).

## 3. G3 Oracle gate

G3 re-evaluated B0 from the same cached candidate universe and compared it with the pinned
evaluation-side Oracle arms. The B0 assembled row reproduces the canonical G1 values exactly:

| scorer | threshold | relative density | degree MMD ratio | clustering MMD ratio | spectral MMD ratio | composite |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.794385 | 0.997710 | 13.0768 | 11.9273 | 18.0931 | 5.76802e-7 |
| Oracle-topo | top-N | 1.000000 | 14.1148 | 8.23662 | 16.0772 | 2.73458e-6 |
| Oracle-blend | top-N | 1.000000 | 7.58890 | 3.06980 | 9.05715 | 0.00139907 |

The stop-rule headroom is `MMD-ratio(B0) / MMD-ratio(Oracle)`:

| Oracle arm | degree | clustering | spectral | composite ratio |
|---|---:|---:|---:|---:|
| Oracle-topo | 0.926465 | 1.44808 | 1.12539 | 4.74093 |
| Oracle-blend | 1.72315 | 3.88537 | 1.99767 | 2425.56 |

Oracle-topo is mixed on degree, but Oracle-blend improves all three canonical MMD ratios and
the composite by a large margin. Oracle is therefore not approximately equal to B0, so the G3
feature-insufficiency stop rule is not triggered. The hard-heuristic Oracle-topo rows are
degenerate by construction because their negatives are selected by the same CN/AA signal.

The complete G3 artifacts are preserved in
[`outputs/deliverables/b0_v31_breadth_first_20260711/g3/`](../../outputs/deliverables/b0_v31_breadth_first_20260711/g3/),
including `g3_results.json` and `g3_tables.md`.

## 4. Interpretation and gate outcome

The G1 B0 arm survives degree-corrected, heuristic-hard, feature-hard, and full-universe views:
the topology gap remains at a density-matched operating point and the composite passes
its expressivity check. However, the PA-null control wins on easy and feature-hard edge
ranking, so subsequent model comparisons must report PA-null alongside B0 and must not
call this checkpoint a uniformly strong edge scorer. G2 leaves substantial theoretical
headroom under the locked edge-independent contract. G3 confirms that the observed B0 topology
failure is not caused by an absence of true-topology signal: the Oracle-blend arm has clear
room over B0. B0-alt independently reproduces the failure, with degree/clustering/spectral
MMD ratios `15.8304/13.4718/23.4734` despite degree-corrected AUROC/AUPRC
`0.693603/0.732509`; G1 is therefore closed under its required alternate architecture.
Relative to B0, that is an AUROC/AUPRC change of `-0.011917/+0.002249` alongside
degree/clustering/spectral MMD-ratio increases of `21.06%/12.95%/29.74%`. Conversely,
the stronger legacy scorer improves AUROC/AUPRC by `+0.094058/+0.083059` but changes
the three topology ratios by only `+5.88%/-2.51%/+10.41%`, reinforcing that better
edge ranking does not close the assembled-topology gap.
EgoStitch implementation can now begin under the approved spec.

## 5. Synced artifacts

- [`g1_results.json`](../../outputs/e2_resubmit_retry/g1/g1_results.json)
- [`g1_tables.md`](../../outputs/e2_resubmit_retry/g1/g1_tables.md)
- [`g2_results.json`](../../outputs/e2_resubmit_retry/g2/g2_results.json)
- [`test_edge_metrics.json`](../../outputs/e2_resubmit_retry/test_edge_metrics.json)
- [`complete.json`](../../outputs/e2_resubmit_retry/complete.json)
- [`g3_results.json`](../../outputs/deliverables/b0_v31_breadth_first_20260711/g3/g3_results.json)
- [`g3_tables.md`](../../outputs/deliverables/b0_v31_breadth_first_20260711/g3/g3_tables.md)
- [Combined B0/B0-alt `g1_results.json`](../../outputs/runs/g1_b0_b0_alt_20260713T165714Z/g1_results.json)
- [B0-alt candidate scores](../../outputs/runs/b0_alt_20260713T164214Z/scores/candidate.npz)
- [B0-alt checkpoint metadata](../../outputs/b0_alt/run_metadata.json)
- [G1 closeout package](../../outputs/deliverables/g1_closeout_20260713/)

The latest checkpoint-only rerun is preserved at
[`outputs/runs/legacy_v31_s47_20260712T193900Z/`](../../outputs/runs/legacy_v31_s47_20260712T193900Z/);
its classification and G2 results remain robustness context, and its canonical G1 rerun
is under [`g1_canonical_20260713/`](../../outputs/runs/legacy_v31_s47_20260712T193900Z/g1_canonical_20260713/).
