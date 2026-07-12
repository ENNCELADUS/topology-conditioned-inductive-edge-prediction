# E2 Pair-to-Topology Gap: G1/G2 Gate Results

**Experiment:** repository-local hardened E2 on Benchmark-A (`breadth_first`).

**Status:** G1 and G2 completed on 2026-07-12. The final artifacts use the frozen
V3.1 B0 scorer, checkpoint `8a8ebe823e476bed`, and the full 2,037,171-row candidate
universe (2,035,153 non-self pairs plus 2,018 self-pairs). The mandatory O'Bray
perturbation check passed, so the graph-similarity composite is valid for this run.
G3 (Oracle) remains pending.

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
| B0 | degree-corrected, ratio-1 | 0.716871 | 0.742622 | 0.306863 |
| B0 | hard heuristic, ratio-1 | 0.583741 | 0.620193 | 0.092256 |
| B0 | hard feature, ratio-1 | 0.406667 | 0.475048 | -0.149638 |
| B0 | full candidate universe | 0.710776 | 0.123982 | 0.087632 |
| PA-null | easy uniform, ratio-1 | 0.814711 | 0.824781 | 0.005302 |
| PA-null | hard feature, ratio-1 | 0.858646 | 0.869196 | 0.003226 |

The full regime table, including ratio-5 rows, is preserved in
[`g1_tables.md`](../../outputs/e2_resubmit_retry/g1/g1_tables.md).

### Density-matched assembled graph

At the density-matched operating point (`threshold = 0.993710`), B0 assembles a graph
with relative density 0.985429 and a valid composite similarity of
`9.64858e-10` (higher is better; 1 means identical). The component MMD² values are:

| Component | MMD² | Ideal |
|---|---:|---:|
| Degree | 0.620493 | 0 |
| Clustering | 0.729620 | 0 |
| Spectral | 0.836370 | 0 |

The assembled graph contains 2,018 predicted self-loops versus 1,891 reference
self-loops. The PA-null density-matched control has relative density 1.0,
degree/clustering/spectral MMD² of 0.645286/0.734427/0.753003, and composite
`1.53185e-9`.

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
| Expected soft-edge volume `V(P)` | 675,446.012 |
| Measured overlap `Ov(P)` | 0.654778 |
| Minimum overlap `Ov_min` at matched volume | 0.010038 |
| Expected triangles `E[Delta]` | 82,752,832.477 |
| Triangle headroom (`E[Delta] / delta_star`) | 314.453 |
| Overlap headroom (`Ov / Ov_min`) | 65.232 |

Thus the measured soft scorer is well above the minimum overlap required by the
edge-independence ceiling to reach the reference triangle count. This is a feasibility
check, not evidence that the assembled graph is realistic: hard-thresholded assemblies
have `Ov = 1`, where the bound is vacuous, and the exact curve is the relevant object.
The complete curve is in [`g2_results.json`](../../outputs/e2_resubmit_retry/g2/g2_results.json).

## 3. Interpretation and remaining gate

G1 survives degree-corrected, heuristic-hard, feature-hard, and full-universe views:
the topology gap remains at a density-matched operating point and the composite passes
its expressivity check. However, the PA-null control wins on easy and feature-hard edge
ranking, so subsequent model comparisons must report PA-null alongside B0 and must not
call this checkpoint a uniformly strong edge scorer. G2 leaves substantial theoretical
headroom under the locked edge-independent contract. G3's Oracle row is still required
before EgoStitch implementation begins.

## 4. Synced artifacts

- [`g1_results.json`](../../outputs/e2_resubmit_retry/g1/g1_results.json)
- [`g1_tables.md`](../../outputs/e2_resubmit_retry/g1/g1_tables.md)
- [`g2_results.json`](../../outputs/e2_resubmit_retry/g2/g2_results.json)
- [`test_edge_metrics.json`](../../outputs/e2_resubmit_retry/test_edge_metrics.json)
- [`complete.json`](../../outputs/e2_resubmit_retry/complete.json)
