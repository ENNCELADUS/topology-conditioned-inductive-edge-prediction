# Node-held-out V_val with a 10% positive-edge budget

Status: implemented on 2026-09-08; replaces the pair-disjoint boundary of the
[2026-09-06 design](2026-09-06-pring-node-disjoint-validation-design.md). BFS growth,
inner buckets, and classification negatives are unchanged from that design.

## Decision

1. **Boundary.** V_val leaves the training universe. `ValRegionSplit.train_nodes` is the
   substrate minus V_val; `training_positives` and `training_negatives` keep only pairs with
   both endpoints in `train_nodes`. Dynamic negatives, G_struct, KD row and context banks,
   feature statistics, and the teacher's oracle structure are all built on `train_nodes`.
   Packs and F0 caches still cover the whole substrate so V_val rows can be scored.
2. **Budget.** `positive_edge_fraction = 0.10`: cap `floor(0.10 * 53,640) = 5,364` positives,
   FIFO prefix stops at 5,347 (644 self-loops), 869 nodes (10.8% of the substrate).

## Evidence

- The 2026-09-07 diagnostic (branch `codex/node-heldout-diagnostic-20260907`, run in
  `/2023533015/topology-node-heldout-20260907`) trained KD1 with every V_val-touching pair
  dropped at the 20% budget. Its V_val-selected threshold replayed on held-out test gave
  AUROC 0.7222, AUPRC 0.7494, F1 0.6879, GS 0.4255, RD 0.6779, MMD ratios 5.78/5.14/9.92.
  Pair-disjoint selection had not transferred. The node-held-out boundary is what makes V_val
  mirror the official train/test boundary, where every cross-boundary edge is also absent.
- Budget sweep on the real substrate (same root, seed 42), node-held-out accounting:

  | Budget | V_val nodes | Val positives | Boundary dropped | Training positives |
  |---|---:|---:|---:|---:|
  | 20% | 1,250 | 10,719 | 12,327 | 30,594 |
  | 12.5% | 1,019 | 6,703 | 12,572 | 34,365 |
  | 10% | 869 | 5,347 | 11,436 | 36,857 |
  | 7.5% | 695 | 4,015 | 9,555 | 40,070 |

  Above roughly 800 nodes the boundary loss is flat (the BFS core carries the hubs), so
  val positives trade against training positives almost 1:1. At 10% the region is one
  component with density 0.0125 and clustering 0.387 (20%: 0.0125 / 0.394); MMD² to the
  full-giant ball bank is 0.027 / 0.028 / 0.024 for degree / clustering / spectral against
  floors of 0.009 / 0.010 / 0.011 (20%: 0.025 / 0.028 / 0.027); 46–50 unique balls per size.
  Balanced classification validation keeps 5,347 rows per side (AUROC SE ≈ 0.005 at 0.72).

## Consequences

Every split-keyed artifact is invalid: packs' feature statistics, G_struct caches, KD banks,
thresholds, checkpoints, and all results. The teacher must be retrained under this split
before banks are dumped; the diagnostic reused a teacher that had seen the V_val nodes.
