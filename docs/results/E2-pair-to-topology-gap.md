# E2 Expectation: The Pair-to-Topology Gap

**Experiment:** motivating result and reproduction target for the abstract graph ML benchmark.

**Claim:** a strong independent pairwise scorer can achieve reasonable edge-level AUROC/AUPRC while
assembling into a graph with poor topology. This creates the need for per-query local scaffold
conditioning rather than independent edge scoring alone.

**Status:** expectation corrected 2026-07-11. The correct legacy raw-scorer reference reaches
**AUROC 0.793 / AUPRC 0.792** on the final balanced test file, but
its assembled Benchmark-A graph still has only **graph similarity 0.337**, with relative density
**3.134** and large degree, clustering, and spectral MMDs. This is the result the repository-local
E2 rerun is expected to reproduce before it replaces the legacy reference.

Figure: [e2-gap.html](../../figures/e2-gap.html).

---

## 1. What was run

- **Model:** frozen V3.1 pairwise scorer, used as the B0 reference in the protocol. It scores each
  candidate pair independently from frozen node features and has no topology objective.
- **Legacy source run:** `pair_context_gated_abba_no_cross_s47`, checkpoint
  `models/v3.1/train/pair_context_gated_abba_no_cross_s47/best_model.pth`.
- **Benchmark:** Benchmark-A primary split. The held-out candidate universe is assembled into a
  predicted graph after pair scores are thresholded.
- **Operating point:** fixed threshold 0.5 for the canonical result.
- **Graph metrics:** assembled graph metrics over benchmark node buckets with sizes 20 through 200.
- **Audited legacy pairwise artifact:**
  `/Users/richardwang/Documents/grand/logs/tccig/02_balanced_subset/pairwise_test/raw_metrics.json`.
- **Audited legacy topology artifact:**
  `/Users/richardwang/Documents/grand/artifacts/tccig_01_20260626/logs/tccig/pairwise_baseline/topology_test/topology_metrics.json`.
- **Artifact boundary:** the external paths above establish the corrected expectation, but they are
  not repository-local deliverables. E2 remains provisional until the same final benchmark inputs
  and scorer are rerun and packaged in this repository.

---

## 2. The gap

Canonical summary metrics:

| Level | Metric | Value | Reading |
|---|---|---:|---|
| Edge | AUROC | **0.793** | strong pairwise ranking |
| Edge | AUPRC | **0.792** | above the published comparison rows used by the legacy study |
| Edge | accuracy at 0.5 | 0.712 | balanced-test operating point |
| Edge | precision / recall at 0.5 | 0.718 / 0.668 | comparatively balanced operating point |
| Edge | F1 / MCC at 0.5 | 0.692 / 0.423 | thresholded separation |
| Assembled graph | graph similarity | **0.337** | poor despite strong pairwise metrics |
| Assembled graph | relative density | 3.134 | strongly over-dense; target is approximately 1 |
| Assembled graph | degree MMD | 26.99 | high; ideal is near 0 |
| Assembled graph | clustering MMD | 21.14 | high; ideal is near 0 |
| Assembled graph | spectral MMD | 20.99 | high; ideal is near 0 |

Paper-ready statement:

> The frozen pairwise scorer reaches AUROC 0.793 and AUPRC 0.792, yet its assembled Benchmark-A
> graph has graph similarity 0.337, relative density 3.134, and high degree, clustering, and
> spectral MMDs. Strong pair-level ranking does not by itself guarantee plausible graph assembly.

---

## 3. The gap widens with graph size

Per-bucket graph similarity for the frozen baseline:

| Node bucket | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph similarity | 0.431 | 0.414 | 0.407 | 0.335 | 0.366 | 0.296 | 0.308 | 0.292 | 0.295 | 0.224 |

The assembled graph gets less plausible overall as the evaluated subgraph grows: graph similarity
falls from 0.431 at bucket size 20 to 0.224 at bucket size 200, although the trend is not strictly
monotone at every intermediate bucket. The existing [e2-gap.html](../../figures/e2-gap.html) still
contains the invalidated intermediate-file numbers and must be regenerated separately.

---

## 4. Invalidated operating-point evidence

The previous variant table was derived from the same invalid intermediate benchmark surface and
must not be used. The corrected legacy reference establishes only the threshold-0.5 point above.
A clean operating-point curve must be regenerated from one frozen scorer and the final candidate
universe, with edge metrics and assembled-graph metrics reported together at every threshold.

---

## 5. Preview for the main experiment

No topology-aware comparison should inherit the invalidated intermediate-file rows. The E3
baseline table must rerun B0, topology-loss, global-refiner, static-denoiser, and local-scaffold
methods under the same final split artifacts, scorer family, threshold policy, and metric
normalization.

---

## 6. Caveats and open decisions

1. **Reproduction required:** the corrected numbers are an audited legacy expectation, not yet a
   repository-local E2 deliverable.
2. **Final benchmark files only:** intermediate benchmark files are invalid for E2 training,
   checkpoint comparison, evaluation, and paper tables.
3. **Baseline identity:** the reproduction must use the pinned V3.1 raw scorer above, or explicitly
   declare and justify a different frozen B0 before comparing results.
4. **Metric normalization:** assembled graph MMDs must use one canonical implementation and scale.
5. **No true threshold sweep yet:** the clean recall-to-topology curve requires one frozen scorer,
   one final candidate universe, and multiple thresholds.
6. **Head-to-head baselines:** this E2 expectation motivates the benchmark; it is not a substitute
   for the E3 table that compares all baselines under one protocol.

---

## 7. Deliverables produced

- [e2-gap.html](../../figures/e2-gap.html): currently stale; regenerate from corrected artifacts
  before citing it.
- This document: corrected E2 expectation, provenance, invalidation notice, and rerun requirements.
- [03-experiment-protocol.md](../03-experiment-protocol.md): protocol context for where E2 sits in the full
  experiment matrix.
