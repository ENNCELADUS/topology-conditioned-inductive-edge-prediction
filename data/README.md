# benchmark_2025_neurips Data Contract

This repository does not store concrete benchmark artifacts. The benchmark is
defined as a neutral graph ML task interface so the paper and implementation
remain dataset-agnostic.

## Task

Given two unseen nodes with frozen feature vectors, predict a binary edge label
for the queried pair. Predictions over a held-out candidate-pair universe are
assembled into a graph and evaluated with both edge-level and graph-level
metrics.

## Required Inputs

An implementation should provide the following artifacts outside this repository:

| Artifact | Format | Role |
|---|---|---|
| Node features | tensor or array store keyed by opaque node ID | Frozen input representation for every node used by the benchmark. |
| Split metadata | structured file | Train, validation, and test node membership. |
| Train pairs | tabular edge list | Labeled pairs for model fitting. |
| Validation pairs | tabular edge list | Labeled pairs for threshold selection and early stopping. |
| Test candidate pairs | tabular edge list | Held-out candidate universe for final edge scoring. |
| Train reference graph | graph object or edge list | Training-only topology available for topology-aware supervision. |
| Test reference graph | graph object or edge list | Held-out graph used only by the evaluator. |
| Evaluation node buckets | structured file | Node subsets used for assembled-graph metrics. |

Node IDs must be opaque benchmark IDs, for example `node_000001`, and must not
encode domain-specific source identifiers.

## Expected Pair File Schema

Pair files use one record per line:

```text
node_id_a<TAB>node_id_b<TAB>label
```

`label` is `1` for a positive edge and `0` for a negative edge. Candidate-pair
files used only for inference may omit `label` when labels are stored separately
for the evaluator.

## Model Output

Models should emit one score per queried pair:

```text
node_id_a<TAB>node_id_b<TAB>score
```

`score` is a calibrated or rankable edge probability in `[0, 1]`. The evaluator
may additionally consume a thresholded prediction column when reporting
operating-point metrics.

## Evaluation Outputs

Every benchmark run must report:

- Edge-level metrics: AUROC, AUPRC, accuracy, precision, recall, F1, MCC, and
  calibration when probabilities are used.
- Assembled-graph metrics: graph similarity, relative density, degree-distribution
  MMD, clustering-coefficient MMD, and Laplacian-spectrum MMD.
- Metadata: split name, node-feature source, scorer checkpoint, threshold policy,
  random seed, and candidate-universe definition.

## Integrity Rules

- Train and test nodes must be disjoint.
- Retrieval, scaffold construction, and training must not access held-out graph
  structure.
- The queried edge must be masked from any local scaffold used to score that
  query.
- Edge-level and assembled-graph metrics must be reported together.
- Concrete source data, domain IDs, and domain-specific filenames should not be
  committed to this repository.
