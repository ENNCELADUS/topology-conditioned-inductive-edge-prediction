# Density-oriented validation root selection (10% positive budget)

Selected on 2026-09-08: **node_002696**, reproduced by **split_seed=273**.
This is an explicitly **test-informed split redesign** authorized by the user.
It is not an untouched-test result, and actual threshold transfer has not been measured.

**Retired as headline on 2026-09-09.** Selecting the validation region on test label
statistics tunes the deployable threshold on test, so the headline split reverted to the
pre-registered random draw (`split_seed=42`, root `node_007630`, `configs/split_seed42/`).
Results on this split (`configs/split20260908/`, §2.2 of `docs/03-experiments.md`) remain
as a labeled upper bound on threshold transfer.

## Decision rule

The goal is to reduce density mismatch relevant to freezing a topology threshold.
For each size s, compare distributions of `log(number of positive edges)` in
validation and the fixed test buckets, including self-loops as the evaluator does.
Average Wasserstein-1 distance equally over the ten sizes. This measures relative
edge-count mismatch and relates to the selector's absolute-log-RD objective; it
is a proxy, not an estimate of the model's test RD. Matching density alone cannot
correct changes in positive/negative conditional score distributions.

Screen every eligible five-neighbor root under the unchanged 5,364-positive cap:
443 roots, 438 large enough to support 200-node buckets. Each screen uses 50 balls
per size, seed 43. Refine the 12 lowest-distance roots and previous reference
candidates with 50 balls per size and seeds 43–47. Choose minimum mean distance;
fit-edge retention breaks exact ties only. This is the winner of this two-stage
search, not a claim of global optimality over every possible split or all sampler seeds.

The production inner bank remains seed 43; it was not replaced by the most favorable
replicate. Negative seed 0, FIFO traversal, budget and complete node isolation are unchanged.

## Comparison

Distances below are five-seed means to the same fixed test bank. Log-edge distance
includes loops; density distance excludes loops. All figures are graph-truth
comparisons, not prediction quality.

| Split | Nodes | Mean log-edge W1 ↓ | Mean density W1 ↓ | Fit positive pairs |
|---|---:|---:|---:|---:|
| Previous 10%, root 007630 | 869 | 0.36966 | 0.07297 | 36,857 |
| Previous 20% positive budget, root 007630 | 1250 | 0.17643 | 0.04156 | 30,594 |
| **Selected 10%, root 002696** | **642** | **0.15574** | **0.03942** | **38,916** |

Log-edge distance falls 57.9% versus old 10% and
11.7% versus old 20%. Replicate variation and all shortlisted
candidates are in [selection.json](selection.json). Overlapping balls and repeated
banks from one graph are not independent graph replicates or confidence intervals.

## Frozen artifact and accounting

[data/val_region/breadth_first.json](../../../data/val_region/breadth_first.json)
is regenerated and verified against the new default:

- Validation: 642 nodes; 5,354 positives, including 528 self-loops; 5,354 negatives.
- Fit: 7,430 nodes; 38,916 positive pairs, including 5,350 self-loops.
- Boundary: 9,370 positive pairs discarded from both fit and validation.
- Non-self edge accounting: 33,566 fit + 4,826 validation + 9,370 boundary = 47,762.
- Fixed benchmark negative count after filtering: 35,477. Active dynamic training
  retains its existing 1:5 sampler, restricted to the new fit universe.

Compared with old 10%, this retains 2,059 more training positives; compared with
old 20%, it retains 8,322 more. All validation-touching training pairs remain excluded.
The new validation is smaller than test and overlapping balls have limited effective
coverage; this design does not assert full feature or topology distribution equality.

## Reproduce

Run from the repository root:

```bash
rtk proxy env PYTHONPATH=. .venv/bin/python docs/results/validation_density_selection/select_root.py
rtk proxy .venv/bin/python -m src.data.val_region \
  --data-root data/benchmark_2025_neurips --strategy breadth_first \
  --output data/val_region/breadth_first.json
```

The search uses true test topology by design; do not describe the selected seed as
randomly chosen independently of test. The generator itself uses only train-side
inputs and the frozen seed. Previous split-dependent checkpoints, structural packs,
fitted statistics and KD banks must be regenerated for new training. No training,
HPO launch, remote synchronization, or score/threshold replay was performed here.

Checks: 250 focused data/loader/scoring/teacher-bank/pack tests passed; Ruff and
`git diff --check` passed. The production scoring loader independently reproduced
the frozen membership, labels and buckets. The generated
manifest was checked for cap compliance, node isolation, complete bucket sizes,
classification-negative legality, and exact fit/validation/boundary accounting.
