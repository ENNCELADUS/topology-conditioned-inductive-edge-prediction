# benchmark_2025_neurips Data

This directory contains the local artifact package for `benchmark_2025_neurips`,
a neutral graph ML benchmark for inductive binary edge prediction.

All model-facing node IDs are opaque IDs in the form `node_000001`. The original
source identifiers are not stored in this repository.

## Layout

```text
benchmark_2025_neurips/
  graph.pkl
  positive_edges.txt
  breadth_first/
  depth_first/
  random_walk/
features/
  frozen_node_features_1024/
    index.json
```

`graph.pkl` is the full reference graph for offline split construction and audit
work. `positive_edges.txt` is the global positive edge list used by negative
sampling and split-level consistency checks.

`features/frozen_node_features_1024/index.json` maps neutral node IDs to cached
feature tensor paths. The tensor paths are left unchanged because they are
content-addressed cache paths and do not expose source node IDs.

Each split-strategy directory contains:

| File | Role |
|---|---|
| `split.pkl` | Node membership for the split strategy. |
| `train_edges.txt` | Labeled train pairs. |
| `val_edges.txt` | Labeled validation pairs. |
| `test_edges.txt` | Labeled test pairs. |
| `train_edges_ratio5_exclusive.txt` | Explicit train supervision with a 1:5 positive-to-negative ratio. |
| `val_edges_ratio5_exclusive.txt` | Explicit validation supervision with a 1:5 positive-to-negative ratio. |
| `test_edges_ratio5_exclusive.txt` | Explicit test supervision with a 1:5 positive-to-negative ratio. |
| `candidate_test_edges.txt` | Candidate-pair universe for held-out assembled-graph evaluation. |
| `train_graph.pkl` | Reference graph over train nodes for topology-aware supervision. |
| `test_graph.pkl` | Held-out reference graph for final assembled-graph evaluation. |
| `test_node_buckets.pkl` | Node buckets used for assembled-graph metrics. |

## Contract

- Treat node IDs as opaque strings.
- Train and test nodes are disjoint under each split strategy.
- Training, retrieval, and scaffold construction must not access held-out graph
  structure.
- Report edge-level metrics and assembled-graph metrics together.
- Keep model-facing file and folder names domain-neutral.
