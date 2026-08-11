# Scalable Edge Prediction and Benchmarking Notes

## Scope

This report reframes the prior benchmark-oriented retrieval pass as a general
ML note on scalable edge prediction, graph-level evaluation, data leakage,
negative sampling, and uncertainty. It is intended to support a neutral
literature directory where the application domain is not part of the visible
surface.

The papers below are grouped by the methodological issue they address rather
than by domain. Application-specific papers from the previous report are not
listed by title here when the title itself exposes the old framing.

## What This Note Is For

Use this note when designing or reviewing an edge-prediction benchmark that must
work at graph scale:

- all-pairs scoring must be replaced or accelerated by retrieval;
- train/test splits must avoid near-duplicate or shortcut leakage;
- negative examples must be sampled with care;
- graph-level quality must be evaluated alongside pairwise metrics;
- uncertainty and selective prediction should be reported for deployment-like
  settings.

## Method Clusters

| Cluster | Question | Useful design pattern |
|---|---|---|
| Retrieval-scale edge scoring | How can a model avoid exhaustive all-pairs scoring? | Factorize pair scoring into node-side or substructure-side representations, then use approximate nearest-neighbor retrieval. |
| edge-aware representation learning | How can a model represent pair context rather than independent endpoints only? | Use cross-attention, paired pretraining, or context-conditioned encoders before edge scoring. |
| Leakage-resistant evaluation | Do reported gains survive strict split rules? | Split by entities, near-duplicates, groups, or graph regions; audit shortcut features before claiming generalization. |
| Negative sampling | Are negatives realistic, too easy, or contaminated? | Compare random, hard, topology-aware, and positive-unlabeled protocols. |
| Graph-level evaluation | Does the predicted edge set preserve global structure? | Report density, degree, clustering, component, spectral, and modularity statistics in addition to AUROC or AUPR. |
| Uncertainty and abstention | Can the model identify low-confidence predictions? | Add calibrated confidence, selective risk curves, or conformal-style coverage checks. |

## High-Value General References

| Priority | Reference | Use |
|---|---|---|
| A | *On Evaluation Metrics for Graph Generative Models* (ICLR 2022) | Metric design for graph-level fidelity. |
| A | OpenGSL, *A Comprehensive Benchmark for Graph Structure Learning* (NeurIPS 2023) | Standardized comparison of graph structure learning methods. |
| A | GSLB, *The Graph Structure Learning Benchmark* (NeurIPS 2023) | Cross-check for effectiveness, robustness, and complexity claims. |
| A | *Learning Discrete Structures for Graph Neural Networks* (ICML 2019) | Foundational edge-distribution learning setup. |
| A | IDGL, *Iterative Deep Graph Learning for Graph Neural Networks* (NeurIPS 2020) | Practical iterative refinement baseline. |
| B | Graph pooling benchmark work (2024) | Useful when graph-level outputs rely on hierarchy or subgraph summarization. |
| B | Modern uncertainty-aware edge prediction work | Relevant for selective prediction and calibrated confidence. |

## Benchmark Design Checklist

1. Define the target graph construction task before choosing metrics.
2. Separate candidate generation, edge scoring, and graph refinement.
3. Report pairwise metrics and graph-level metrics side by side.
4. Audit leakage from duplicated entities, high-similarity examples, or shared
   metadata.
5. Compare at least one random-negative protocol against one hard-negative or
   topology-aware protocol.
6. Measure retrieval cost, memory use, and inference latency for large graphs.
7. Include calibration or selective prediction when predictions will be ranked
   for review.

## Recommended Neutral Framing

Avoid presenting the task as isolated pair classification. The stronger general
ML framing is:

> Learn a sparse, calibrated edge set over a large node universe such that the
> resulting graph is accurate, well calibrated, computationally feasible, and
> structurally faithful.

This framing keeps the benchmark aligned with graph construction, inductive edge
prediction, and structure refinement.

## Remaining Gap

The prior pass had many domain-specific benchmark entries. For a fully neutral
research corpus, replace those with general papers on retrieval-based edge
prediction, leakage-resistant graph splits, graph generation metrics, and
positive-unlabeled or hard-negative learning.
