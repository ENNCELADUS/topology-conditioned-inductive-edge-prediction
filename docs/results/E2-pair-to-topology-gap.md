# E2: The Pair-to-Topology Gap

**Experiment:** motivating result for the abstract graph ML benchmark.

**Claim:** a strong independent pairwise scorer can achieve reasonable edge-level AUROC/AUPRC while
assembling into a graph with poor topology. This creates the need for per-query local scaffold
conditioning rather than independent edge scoring alone.

**Verdict:** supported by the current repository-local result note and figure. The frozen pairwise
scorer reaches **AUROC 0.676 / AUPRC 0.691**, but the assembled Benchmark-A graph has
**graph similarity 0.235** with high degree, clustering, and spectral MMDs. The scorer is usable at
the pair level and weak at the assembled graph level.

Figure: [e2-gap.html](../../figures/e2-gap.html).

---

## 1. What was run

- **Model:** frozen pairwise scorer, used as B0 in the protocol. It scores each candidate pair
  independently from frozen node features and has no topology objective.
- **Benchmark:** Benchmark-A primary split. The held-out candidate universe is assembled into a
  predicted graph after pair scores are thresholded.
- **Operating point:** fixed threshold 0.5 for the canonical result.
- **Graph metrics:** assembled graph metrics over benchmark node buckets with sizes 20 through 200.
- **Artifact boundary:** this document and [e2-gap.html](../../figures/e2-gap.html) are the local
  artifacts to cite from this repository. No external local folder is required to understand the
  result.

---

## 2. The gap

Canonical summary metrics:

| Level | Metric | Value | Reading |
|---|---|---:|---|
| Edge | AUROC | **0.676** | reasonable |
| Edge | AUPRC | **0.691** | reasonable |
| Edge | precision / recall at 0.5 | 0.731 / 0.321 | high precision, low recall |
| Edge | MCC | 0.245 | modest thresholded separation |
| Assembled graph | graph similarity | **0.235** | poor under the benchmark scale |
| Assembled graph | relative density | 0.684 | sparse at this operating point |
| Assembled graph | degree MMD | 17.17 | high; ideal is near 0 |
| Assembled graph | clustering MMD | 11.81 | high; ideal is near 0 |
| Assembled graph | spectral MMD | 22.12 | high; ideal is near 0 |

Paper-ready statement:

> The frozen pairwise scorer reaches AUROC 0.676 and AUPRC 0.691, yet its assembled Benchmark-A
> graph has graph similarity 0.235 and high degree, clustering, and spectral MMDs. Stronger
> pair-level scores do not by themselves guarantee plausible graph assembly.

---

## 3. The gap widens with graph size

Per-bucket graph similarity for the frozen baseline:

| Node bucket | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph similarity | 0.320 | 0.260 | 0.252 | 0.255 | 0.232 | 0.208 | 0.211 | 0.213 | 0.220 | 0.179 |

The assembled graph gets less plausible as the evaluated subgraph grows: graph similarity falls
from 0.320 at bucket size 20 to 0.179 at bucket size 200. This is the size-trend panel in
[e2-gap.html](../../figures/e2-gap.html).

---

## 4. Operating-point evidence

The canonical B0 operating point is threshold 0.5. Additional cached variants show the same broad
failure mode across different recall and density regimes, but they are not a clean threshold sweep
because the variants differ in more than the decision threshold.

| Variant | recall | relative density | graph similarity | degree MMD | spectral MMD |
|---|---:|---:|---:|---:|---:|
| variant-20 | 0.448 | 1.07 | 0.262 | 24.18 | 22.88 |
| variant-40 | 0.314 | 0.53 | 0.242 | 15.20 | 24.14 |
| variant-60 | 0.254 | 0.40 | 0.228 | 17.09 | 29.07 |
| variant-60-100 | 0.176 | 0.25 | 0.192 | 28.58 | 44.51 |

Reading: both denser and sparser assemblies can be topologically poor. This motivates conditioning
the edge decision on local topology rather than treating the graph problem as threshold selection
alone.

---

## 5. Preview for the main experiment

A topology-aware training objective narrows part of the gap in cached evidence: one exploratory
variant reaches degree MMD 7.18 and clustering MMD 8.28 at AUROC 0.686. This should be treated as
directional evidence only because metric normalization differs from the canonical B0 result above.

The E3 baseline table must rerun B0, topology-loss, global-refiner, static-denoiser, and local
scaffold methods under the same split, scorer family, threshold policy, and metric normalization.

---

## 6. Caveats and open decisions

1. **Baseline family:** E2 should use the same frozen pairwise scorer family as the main local
   scaffold method. If the current cached run is older than the main scorer, rerun E2 before making
   final claims.
2. **Metric normalization:** assembled graph MMDs must use one canonical normalization. Existing
   evidence includes mixed scales, so cross-run comparisons should be regenerated before quoting
   them as final.
3. **No true threshold sweep yet:** the clean recall-to-MMD curve requires one frozen scorer,
   one candidate universe, and multiple thresholds. Current operating-point evidence is useful but
   not a pure threshold sweep.
4. **Density direction is operating-point dependent:** state the robust claim as low graph
   similarity plus high MMDs, not as a universal over-dense or under-dense failure.
5. **Head-to-head baselines:** this E2 result motivates the benchmark. It is not a substitute for
   the E3 table that compares all baselines under one protocol.

---

## 7. Deliverables produced

- [e2-gap.html](../../figures/e2-gap.html): edge-vs-topology contrast, MMD panel, and gap-vs-size
  trend.
- This document: self-contained E2 result summary, caveats, and follow-up requirements.
- [03-experiment-protocol.md](../03-experiment-protocol.md): protocol context for where E2 sits in the full
  experiment matrix.
