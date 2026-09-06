# PRING-style node-disjoint validation

Status: sampling design approved in conversation on 2026-09-06; written specification
awaiting review. This document specifies future behavior, not the current runtime.

## Purpose

Select checkpoints and freeze decision thresholds under a validation setting that
resembles the benchmark's unseen-node deployment condition. Replace the current
five-region, edge-budget, pair-disjoint V_val with a single FIFO BFS node holdout
inside the existing train-side substrate. Then sample validation subgraphs using
the PRING inner BFS procedure.

The task remains a binary edge decision from exactly `(x_u, x_v)`. Validation graph
truth defines offline splits, labels and evaluation samples; it is not model input.
This design does not add test-time adaptation or tune validation to observed test
density. It does not promise removal of distribution shift or RD near one on test.

## Evidence and reference behavior

Primary sources inspected on 2026-09-06:

- [PRING paper, Appendix D.1](https://arxiv.org/html/2507.05101v2#A4.SS1):
  node-disjoint outer train/test split, followed by topology subgraph evaluation;
  the paper's original validation instead partitions training-side labeled pairs.
- [graph_split.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/graph_split.py):
  the executed main uses a 0.2 node fraction and a root with exactly five neighbors.
  Traversal uses a FIFO queue and sorted neighbors. Both partitions are induced graphs.
- [graph_sample.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/graph_sample.py):
  uniform random roots, FIFO BFS, 50 draws for each size in 20, 40, ..., 200.
- [negative_sample.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/negative_sample.py):
  negative endpoints are drawn from the positive-edge endpoint multiset, with
  rejection of positives and self-pairs.

The reference scripts were read from upstream main; the links are not immutable
artifact provenance. Reproduce their stated algorithm, not original node membership:
our opaque node IDs differ, and sorting them changes traversal tie ordering.

Read-only inspection of the actual `breadth_first/train_graph.pkl` on H20 host
`5bn369682drog-0`, checkout `3a36b94512a2608549cbc5fcbed0150d6ff193d6`:

| Quantity | Value |
|---|---:|
| Nodes, exactly matching split.pkl train membership | 8,072 |
| Non-self positive edges | 47,762 |
| Self-loop positives | 5,878 |
| Connected components after removing loops | 728 |
| Largest component nodes / non-self edges | 7,237 / 47,643 |
| Second-largest component nodes | 6 |
| Roots with five neighbors in original graph | 445 |
| Such roots in the largest component | 443 |

Current `src/data/val_region.py` differs in both isolation and traversal:
`_grow_regions` uses hash-priority heaps rather than a FIFO BFS queue;
`_bfs_ball` orders entire distance layers by hash; training retains cross-boundary
pairs. These paths must not silently remain the new protocol's implementation.

## Sampling contract

### 1. Substrate and outer holdout

Start from the complete existing train-side substrate, original train positives
plus original validation positives, not the effective graph after the old V_val
quarantine. Keep all substrate nodes, including small components and isolated nodes.
The existing official test split and sampled node buckets are unchanged.

Set the holdout node count to `floor(0.2 * N_substrate)`: 1,614 for this dataset.
Do not use a fraction of giant-component nodes or a target edge count.

Candidate roots have `len(neighbors(root)) == 5` on the substrate with its recorded
self-loops retained. A self-loop contributes one neighbor here, unlike NetworkX
degree, where it contributes two. Restrict candidates to components large enough
to reach the holdout size. The current dataset has 443 eligible roots, all in its
largest component.

Use a local Python random generator seeded with 42 and select uniformly from the
sorted eligible-root list. This is a reproducible version of the original script's
shuffled-first-eligible root selection. Do not search roots by density, model scores,
validation quality or resemblance to official test.

Run FIFO BFS from that root: pop the head, admit a node on its first visit, enqueue
its sorted unvisited neighbors, and stop after exactly 1,614 unique nodes. Duplicate
queue entries may be skipped on pop. Do not reorder a complete frontier by a global
hash, use a priority queue, or combine multiple roots. Self-loops cannot add nodes.

Define `V_fit = V_substrate - V_val`; retain the full induced `G_val` and `G_fit`.
For this substrate `|V_fit| = 6,458`. Exact induced and boundary edge counts are
outputs to measure during implementation, not assumed from the node fraction.

### 2. Complete node isolation

Training positives and negatives must have both endpoints in V_fit. Exclude every
cross-boundary pair, as well as every V_val-internal pair, from all training streams.
Build the training structural graph over V_fit only, removing structural self-loops
as required by the existing model contract. Fit-side classification loops remain.

Apply the same boundary to teachers, KD training banks, topology supervision,
negative samplers, grounding/retrieval pools and fitted feature statistics. Merely
removing edges while allowing held-out nodes to enter training buckets is insufficient.
Frozen per-node intrinsic features can be loaded for validation inference; learned
normalization or other fitted statistics must use V_fit only.

Validation labels and graph descriptors are available to the evaluator and selector.
Any teacher outputs used in validation must come from the fit-trained teacher under
its declared information boundary. True V_val structure overlays remain separate
oracle diagnostics and must not feed the deployable selection path. Training KD
records may not involve V_val nodes.

### 3. Inner topology sampling

Use a separate local Python random generator seeded with 43, so inner draws do not
depend on how many random operations outer selection consumed. For ascending sizes
20, 40, ..., 200, draw 50 roots uniformly with replacement from sorted V_val.
Use the same sorted-neighbor FIFO BFS on G_val for each draw and retain the induced
subgraph on the resulting nodes. The root-neighbor-count condition applies only to
the outer split, not these inner draws.

All sizes are reachable because the outer BFS produces a connected 1,614-node
induced graph. Preserve overlapping samples, repeated roots and any repeated node
sets. Do not filter, balance or resample by density, descriptors or model performance.
Record the final 500 node sets and seeds for reproducibility.

Build the exact union of unordered within-sample pairs, including self-pairs under
the current evaluation contract. Score each unique pair once; evaluate each sample
occurrence with its original weight. Do not downsample topology negatives or confuse
unique-union metrics with sample-macro metrics.

### 4. Classification samples

Construct a separate fixed validation classification set inside V_val: all induced
positive pairs including positive self-pairs, and equally many unique non-self
negative pairs. Use a local negative-sampling seed of 0. Draw endpoints from the
multiset obtained by appending both endpoints of each canonical positive pair;
a positive self-pair therefore contributes twice. Reject self-pairs, known positives
and duplicate canonical unordered pairs. Restrict positive rejection truth to this
validation node universe; official test structure is unnecessary.

Generate this validation negative set afresh, rather than retaining a mixture of
old benchmark negatives and a different top-up distribution. Fit-side fixed negatives
can be filtered from the existing train/val negative pool; existing training-side
sampling ratios remain unchanged, with all dynamic endpoints restricted to V_fit.

### 5. Selection and threshold replay

Keep the current checkpoint selection rule, topology density/shape threshold rule,
and separate max-F1 classification threshold rule. All three use the new validation
surface. Freeze the selected checkpoint and both thresholds before official test.
Report edge metrics and all five topology metrics, with additional non-self density
diagnostics to expose self-loop effects.

Do not refit on all substrate nodes and carry over thresholds from a different model.
Old checkpoints, structural caches, teacher banks and score artifacts cannot establish
unseen-node performance on the new split. Regenerate split-dependent artifacts and
retrain teachers/students before making new-protocol claims. Immutable intrinsic
per-node feature caches need not be recomputed; their downstream membership views do.

## Integration and scope

Use the existing split module and shared graph construction/data-loading paths; keep
outer membership and inner BFS sampling as explicit, independently testable concerns.
Every consumer must distinguish substrate membership from fit membership, especially
current consumers of `ValRegionSplit.train_nodes`. Audit train workers, KD-bank builders,
score-universe loading, grounding, feature-stat fitting and pipeline pack construction.
Preserve the current public evaluation outputs where their semantics are unchanged.

Update project protocol documentation alongside the future implementation. Retain
historical reports with their original protocol labels; do not relabel old results.
Retire the replaced multi-region/edge-budget runtime path rather than adding a silent
compatibility fallback. Formal HPO program changes require their own explicit scope;
this design does not alter the human-owned campaign program or launch training.

A single outer split is the main protocol. Multiple independently trained BFS splits
are a possible later robustness study, not part of this implementation. Uniform random
node holdout and merged multi-region validation were considered and rejected for the
primary BFS setting because they change the target region-generation mechanism.

## Verification and diagnostic outputs

Use behavior tests that distinguish FIFO from priority/layer ordering, check root
neighbor-count semantics with self-loops, and verify exact node budgets, connectivity,
reproducibility and disjoint membership. Check fit/internal/boundary positive accounting
and ensure positive and negative training streams never admit a validation endpoint.
Exercise structural and bank consumers, not only pair-list filters.

Verify the 10 sizes and 50 draws per size, exact pair-union coverage, classification
balance and negative rejection. With fixed scores, verify the existing selector and
threshold replay still use validation only. Missing eligible roots, unreachable sizes,
non-finite scores, boundary violations and I/O failures must fail explicitly; do not
silently change the root rule, reduce sample size or fall back to the old split.

The CPU dataset audit must report induced fit/validation edges, discarded boundary
edges, component sizes, fit isolated nodes, and per-size density, degree, clustering,
spectral descriptors, node coverage, unique sample counts and sample overlap. Include
both loopless diagnostics and the official loop-retaining evaluation convention.
These statistics describe the split; they are not quality gates or seed-search targets.
Do not claim 500 overlapping samples are independent replicates.

The new sampling mechanism is closer to PRING's deployment experiment, but the
substrate already excludes the original held-out BFS region. Nested sampling cannot
guarantee identical feature, label or structural distributions. Actual training-data
loss and threshold transfer quality remain empirical outcomes to report.
