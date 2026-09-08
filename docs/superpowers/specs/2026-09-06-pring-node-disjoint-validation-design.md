# PRING-style BFS validation with a positive-edge budget

Status: superseded on 2026-09-08 by
[node-held-out validation](2026-09-08-node-held-out-validation-design.md): the
budget is now 10% of positives and every pair touching V_val is held out.
The BFS growth, bucket, and negative-sampling rules below still apply.

Original status: implemented on 2026-09-06. The user revised the budget from 20% of nodes
to an upper bound of approximately 20% of positive pairs. Cross-boundary training
samples remain retained. The historical filename is unchanged; this is
**pair-disjoint validation**, not node-disjoint validation.

## Purpose and evidence

Reproduce PRING's two-stage FIFO BFS sampling while retaining approximately the
original amount of positive training supervision. The former 20%-node variant
held out 18,296 of 53,640 positive pairs (34.1%), compared with the official
10,760 validation positives. Node fraction is therefore replaced by a positive-pair
cap. This is a deliberate adaptation of PRING's outer split, not its exact recipe.

Sources inspected for the original design:

- [PRING Appendix D.1](https://arxiv.org/html/2507.05101v2#A4.SS1).
- [graph_split.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/graph_split.py): five-neighbor root, sorted-neighbor FIFO BFS, original node budget.
- [graph_sample.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/graph_sample.py): uniform roots with replacement, 50 samples per size 20–200.
- [negative_sample.py](https://github.com/SophieSarceau/PRING/blob/main/data_process/graph_gen/negative_sample.py): positive-endpoint-frequency negative sampling.

## Outer region and strict upper bound

Use the complete, unchanged train-side `train_graph.pkl`, including original
train and validation positives: 8,072 nodes and 53,640 canonical positive pairs,
including 5,878 self-loops. Official test membership and buckets remain unchanged.

Set `positive_edge_fraction=0.20` and the integer cap to
`floor(0.20 * 53,640) = 10,728` positive pairs. Count each unordered pair once and
each positive self-loop once. This matches the labeled-positive accounting used
for the official train/validation comparison, rather than loopless topology counts.

Select uniformly from sorted roots with `len(neighbors(root)) == 5`, using seed 42.
A loop contributes one neighbor here. Candidate components must reach the largest
requested bucket size (200); on this substrate the eligible 443 roots all belong
to the giant component. No test statistics or model scores influence selection.

Grow one sorted-neighbor FIFO BFS prefix. Before admitting a node, count its
positive edges to admitted nodes plus its self-loop, if present. **Stop before the
first node that would exceed the cap.** Do not skip that node or search another
root to fill the remaining budget. The final count may be slightly below the cap.
If the result cannot support the requested buckets, fail explicitly without
increasing the budget or changing seeds. Preserve the full induced validation graph.

## Training boundary

Keep all substrate nodes in `train_nodes`. Hold out only pairs whose two endpoints
are in V_val. Retain every complement-internal and cross-boundary positive and
benchmark negative pair as training samples. Thus train and validation positives
form a complete, disjoint partition of substrate positives.

Training structure, KD banks and dynamic negative sampling reject V_val-internal
pairs; they allow cross-boundary pairs. Structural self-loops are removed, while
classification positives retain them. Fitted feature statistics use the training
universe. V_val nodes can be exposed through training boundary samples; official
test nodes remain unseen. True V_val topology overlays remain oracle diagnostics.

This revision does not change KD negative-sampling behavior or launch training.

## Validation sampling and selection

Within G_val, use seed 43 and uniform roots with replacement. For each size
20, 40, 60, 80, 100, 120, 140, 160, 180 and 200, take **50** sorted-neighbor FIFO
BFS samples: **500 sample occurrences total**. Retain overlaps and repeated sets;
report unique-set counts separately. No density-based filtering or resampling.

For classification, use every internal positive pair and equally many unique
non-self negatives. Seed 0 draws each endpoint from the positive-edge endpoint
multiset; reject self-pairs, positives and duplicate unordered pairs. A positive
self-loop contributes twice to endpoint frequency. No test truth is needed.

Topology evaluates all within-sample pairs, including evaluation self-pairs;
score the deduplicated union and preserve each sample occurrence's weight.
Checkpoint selection, density-first topology threshold, separate max-F1
classification threshold and frozen test replay rules remain unchanged.

## Verification and artifacts

Verify the positive cap, FIFO-prefix stopping including loops, positive partition
completeness, boundary retention for both labels, negative validity, deterministic
replay and exact bucket sizes/counts. Report nodes, loopless edges, loops, positive
and negative rows, boundary edges and each size's sample and unique-set counts.

Keep the current exact manifest at `data/val_region/breadth_first.json`. Workers
rederive the same split from the raw substrate and fixed parameters. To regenerate:

```bash
rtk proxy .venv/bin/python -m src.data.val_region \
  --data-root data/benchmark_2025_neurips --strategy breadth_first \
  --positive-edge-fraction 0.20 --output data/val_region/breadth_first.json
```

The retained split contains 1,250 nodes, 10,719 validation positives (including
946 loops), 10,719 validation negatives and 500 BFS samples (50 per size).
Training retains 42,921 positives and all boundary samples. Obsolete split
manifests and exploratory analysis reports were removed at the user's request.
