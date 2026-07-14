# E2 Pair-to-Topology Gap: G1/G2/G3 Gate Results

> **Metric scope (2026-07-14):** official benchmark Graph Similarity (GS) and Relative
> Density (RD) were formally rerun on frozen score artifacts over all 500 fixed
> `breadth_first` induced subgraphs. MMD ratios remain canonical-run values.

## Executive summary

**Experiment:** hardened E2 on Benchmark-A (`breadth_first`).

**Status:** G1, G2, and G3 are complete. G1 closes the architecture-independence
requirement; neither G2 nor G3 triggers its stop rule. EgoStitch may proceed to
implementation under the approved specification.

| Gate | Headline evidence | Decision |
|---|---|---|
| G1 | B0 AUROC/AUPRC `0.705519/0.730260`; BFS-macro GS/RD `0.312151/0.422345`; MMD ratios `13.0768/11.9273/18.0931` | Pair-to-topology gap supported |
| G1 B0-alt | AUROC/AUPRC `0.693603/0.732509`; BFS-macro GS/RD `0.345802/0.450793`; MMD ratios `15.8304/13.4718/23.4734` | Architecture-independence arm closed |
| G2 | `Ov(P)=0.479886`, above `Ov_min=0.054954` | Edge-independence ceiling does not force a stop |
| G3 | Oracle-topo GS is `1.61155×` B0; Oracle-blend improves all three MMD ratios | Feature-insufficiency stop rule not triggered |

The result supports a narrow claim: independent scoring can retain reasonable edge
ranking while failing to recover the topology of its assembled graph.

It does not establish that this B0 checkpoint is uniformly strong across edge regimes.

This is a scientific gate closeout, not a strict cold-start systems acceptance. The
verified four-H20 timing evidence remains warm-cache only.

## 1. Run identity and evaluation scope

The canonical benchmark-aligned G1 closeout completed on 2026-07-14. B0 and B0-alt were
scored over the same candidate universe.

| Item | B0 | B0-alt |
|---|---|---|
| Model | V3.1 pairwise scorer | F0-MLP |
| Checkpoint ID | `e092537d8cf1e208` | `f3b8afc0b0781c43` |
| Candidate rows | 2,037,171 | 2,037,171 |
| Non-self rows | 2,035,153 | 2,035,153 |
| Self-pair rows | 2,018 | 2,018 |

The primary report uses the canonical B0/B0-alt artifacts. The stronger legacy
checkpoint is retained only as a separately labeled robustness analysis in Section 5.

## 2. G1 hardened-E2

### 2.1 Edge-level regimes

The balanced degree-corrected row is reported with both hard-negative regimes and
the imbalanced full candidate universe. Ratio-1 rows contain 32,019 positives and
32,019 negatives.

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

**Observation:** hard negatives substantially reduce B0 and B0-alt discrimination.
PA-null wins selected easy and feature-hard rows.

**Implication:** subsequent comparisons must retain PA-null and must not describe B0
as uniformly strong. The G1 claim rests on the edge/topology mismatch, not on a
leaderboard claim about this checkpoint.

The ratio-5 and full regime rows are available in the final combined
[`g1_tables.md`](../../outputs/deliverables/g1_graph_metrics_20260714/g1_tables.md).

### 2.2 Assembly calibration and topology

Two quantities called “relative density” must be kept separate:

- **Global simple-edge RD** is the assembled/reference non-self edge-count ratio. It
  checks the operating-point calibration over the whole test graph.
- **BFS-macro RD** is computed on each fixed reference-derived induced subgraph,
  then averaged over all 500 samples. It measures recovery of dense local topology.

B0 and B0-alt use thresholds chosen under a non-self edge quota. Equal-score ties are
included or excluded atomically, so realized global RD can be slightly below `1`.

| Scorer | Assembly policy | Realized non-self edges | Global simple-edge RD | BFS-macro GS | BFS-macro RD |
|---|---:|---:|---:|---:|---:|
| B0 | threshold `0.794385` | 30,059 | 0.997710 | 0.312151 | 0.422345 |
| B0-alt | threshold `0.679179` | 30,090 | 0.998739 | 0.345802 | 0.450793 |
| PA-null | exact top-N | 30,128 | 1.000000 | 0.245377 | 0.489125 |

The reference has 30,128 non-self edges. B0 is globally density calibrated but locally
under-dense inside the reference-derived BFS subgraphs.

The low local RD does not mean the full assembled graph contains only 42% as many edges.

The same assemblies remain far from the real-vs-real MMD floor:

| Scorer | Degree MMD ratio | Clustering MMD ratio | Spectral MMD ratio |
|---|---:|---:|---:|
| B0 | 13.0768 | 11.9273 | 18.0931 |
| B0-alt | 15.8304 | 13.4718 | 23.4734 |
| PA-null | 30.1230 | 22.4118 | 37.8654 |

Relative to B0, B0-alt changes degree-corrected AUROC/AUPRC by
`-0.011917/+0.002249`. Its GS and BFS-macro RD rise by `10.78%/6.74%`, while its
three MMD ratios worsen by `21.06%/12.95%/29.74%`.

The metric directions are mixed, so B0-alt supports persistence of the gap but not a
claim that every topology metric becomes worse.

B0 predicts 2,018 self-loops and B0-alt predicts 2,012, versus 1,891 in the
reference. Official GS/RD retain these loops inside each induced subgraph; global
simple-edge RD excludes them by definition.

The formal 2026-07-14 threshold sweep reports official BFS-macro GS/RD together with
MMD/recall at every evaluated threshold; the operating-point row above is the headline.

### 2.3 Metric provenance

- Official GS/RD use the fixed sampled node sets and macro-average every sample.
- MMD ratios use the canonical fixed-`sigma=1` Gaussian-TV biased MMD² evaluator.
- The historical O'Bray diagnostic validated the retired MMD-ratio composite. It is
  MMD provenance, not part of the official GS definition.
- Degree heterogeneity is `sigma=1.098430`; candidate positive rate is `0.0157174`.

## 3. G2 edge-independence ceiling

G2 evaluates the exact edge-independence identities on the cached soft score matrix.
Self-pairs are excluded from the dense probability matrix.

| Quantity | Value |
|---|---:|
| Test nodes | 2,018 |
| Non-self scored pairs | 2,035,153 |
| Simple reference edges | 30,128 |
| Reference triangles (`delta_star`) | 263,164 |
| Expected soft-edge volume `V(P)` | 123,373.982 |
| Measured overlap `Ov(P)` | 0.479886 |
| Minimum required overlap `Ov_min` | 0.054954 |
| Expected triangles `E[Delta]` | 1,342,118.384 |
| Triangle headroom `E[Delta]/delta_star` | 5.100 |
| Overlap headroom `Ov/Ov_min` | 8.733 |

**Observation:** measured overlap is well above the minimum needed to reach the
reference triangle count under the soft edge-independent ceiling.

**Interpretation:** this is a feasibility result, not evidence that the assembled
graph is realistic. For a hard-thresholded graph, `Ov=1` and the bound becomes vacuous.

**Decision:** G2 does not trigger a stop. The exact ceiling curve remains the relevant
comparison for future calibrated assemblies.

The curve is preserved in
[`results.json`](../../outputs/deliverables/b0_v31_breadth_first_20260711/g2/results.json).

## 4. G3 Oracle gate

G3 compares B0 with two evaluation-side Oracle arms assembled from the same candidate
universe. Oracle arms use exact top-N non-self assembly.

| Scorer | Global simple-edge RD | BFS-macro GS | BFS-macro RD | Degree MMD ratio | Clustering MMD ratio | Spectral MMD ratio |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.997710 | 0.312151 | 0.422345 | 13.0768 | 11.9273 | 18.0931 |
| Oracle-topo | 1.000000 | 0.503048 | 0.794303 | 14.1148 | 8.23662 | 16.0772 |
| Oracle-blend | 1.000000 | 0.323649 | 0.652734 | 7.58890 | 3.06980 | 9.05715 |

Headroom is `MMD-ratio(B0)/MMD-ratio(Oracle)` for MMD and
`GS(Oracle)/GS(B0)` for graph similarity.

| Oracle arm | Degree | Clustering | Spectral | GS ratio |
|---|---:|---:|---:|---:|
| Oracle-topo | 0.926465 | 1.44808 | 1.12539 | 1.61155 |
| Oracle-blend | 1.72315 | 3.88537 | 1.99767 | 1.03683 |

**Observation:** Oracle-topo gives the clearest GS and local-RD improvement but is
mixed on degree MMD. Oracle-blend improves every MMD ratio but only slightly improves GS.

**Implication:** no single Oracle dominates every metric, yet the Oracle set is not
approximately equal to B0 across assembled metrics. The feature-insufficiency stop
rule is therefore not triggered.

Hard-heuristic Oracle-topo edge rows are degenerate because their negatives are
selected by the same common-neighbor/Adamic-Adar signal used by the Oracle.

## 5. Stronger-checkpoint robustness context

The aligned legacy V3.1 checkpoint was evaluated separately in
`legacy_v31_s47_20260712T193900Z`. It is not the primary B0 and not a new formal
four-H20 training acceptance.

| Metric | Canonical B0 | Legacy checkpoint |
|---|---:|---:|
| Degree-corrected AUROC | 0.705519 | 0.799577 |
| Degree-corrected AUPRC | 0.730260 | 0.813319 |
| Global simple-edge RD | 0.997710 | 0.978392 |
| BFS-macro GS | 0.312151 | 0.381264 |
| BFS-macro RD | 0.422345 | 0.500179 |
| Degree/clustering/spectral MMD ratio | 13.0768/11.9273/18.0931 | 13.8456/11.6277/19.9774 |

The stronger checkpoint improves BFS-macro GS by `22.14%` and BFS-macro RD by `18.43%`.
Its MMD ratios do not consistently improve, so it narrows part of the gap without
closing the assembled-topology failure.

Its balanced test AUROC/AUPRC is `0.805170/0.818408`; hard-heuristic and hard-feature
G1 rows are `0.626746/0.663360` and `0.510083/0.602131`, respectively.

## 6. Conclusions and reporting requirements

1. **G1 is closed.** B0-alt independently reproduces the topology failure.
2. **The claim remains narrow.** PA-null wins selected edge regimes, so B0 is not a
   uniformly strong scorer.
3. **Density calibration is not topology recovery.** Global simple-edge RD is near
   `1`, while BFS-macro RD remains `0.422345` for B0.
4. **G2 leaves theoretical room.** The soft scorer exceeds the required overlap.
5. **G3 leaves empirical room.** Oracle gains appear on different metric axes.

Future assembled-graph tables must name global simple-edge RD and BFS-macro RD
separately.

The benchmark-aligned threshold sweep is preserved in the formal G1 artifact. Any claim
away from the documented operating point must identify its exact threshold and recall.

The next research stage is EgoStitch implementation under
[`docs/05-egostitch-spec.md`](../05-egostitch-spec.md). Cold-start four-H20 acceptance
remains a separate systems-validation item.

## 7. Source artifacts

The three 2026-07-14 directories below are the only active final evaluator artifacts.
Each includes a manifest with input, source, and output SHA-256 values. Older evaluator
outputs use the retired schema and are not current result sources.

### Primary artifacts

- [Formal B0 candidate scores](../../outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz)
- [B0-alt candidate scores](../../outputs/runs/b0_alt_20260713T164214Z/scores/candidate.npz)
- [Combined B0/B0-alt G1 results](../../outputs/deliverables/g1_graph_metrics_20260714/g1_results.json)
- [Combined G1 manifest](../../outputs/deliverables/g1_graph_metrics_20260714/manifest.json)
- [G2 results](../../outputs/deliverables/b0_v31_breadth_first_20260711/g2/results.json)
- [G3 results](../../outputs/deliverables/g3_graph_metrics_20260714/g3_results.json)
- [G3 manifest](../../outputs/deliverables/g3_graph_metrics_20260714/manifest.json)
- [Legacy G1 results](../../outputs/deliverables/legacy_g1_graph_metrics_20260714/g1_results.json)
- [Legacy G1 manifest](../../outputs/deliverables/legacy_g1_graph_metrics_20260714/manifest.json)

### Provenance and robustness artifacts

- [Formal B0 deliverable](../../outputs/deliverables/b0_v31_breadth_first_20260711/)
- [Legacy checkpoint rerun](../../outputs/runs/legacy_v31_s47_20260712T193900Z/)
