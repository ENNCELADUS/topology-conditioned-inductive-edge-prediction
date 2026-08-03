# src/

This package implements the benchmark/evaluation stack, frozen B0-family baselines,
G1–G3 analyses, and the pre-registered EgoStitch Stage-1 model and gate.

- `data/`: verified benchmark artifacts, frozen features, packed caches, grounding,
  pair streams, and ego targets.
- `model/`: `egostitch/`'s three independently swappable components --
  `generator/` (imagines a local graph around a queried pair), `encoder/`
  (encodes that graph into an embedding), and `classifier/` (the pairwise
  scorer; B0-V3.1 lives here as `classifier/b0_v31.py` + `classifier/layers.py`,
  since B0-V3.1 is itself the mature pairwise classifier). B0-alt (`b0_alt.py`,
  family `f0_mlp`) was removed 2026-08-03 by owner decision; see
  [`docs/results/E2-pair-to-topology-gap.md`](../docs/results/E2-pair-to-topology-gap.md)
  for the closed result it produced.
- `eval/`: edge metrics, assembled-graph metrics, calibration, and ego fidelity.
- `experiments/`: G1/G2/G3, `B0+cal`, and the G5 Stage-1 gate.
- `train_b0.py` / `train_egostitch.py`: baseline and auto-sized DDP training workers.
- `score_universe.py`: shardable scoring and strict artifact merge.

The retained Seed-0 diagnostic was produced under a superseded registration and is not
a current G5 verdict. The replacement Stage-1 gate is a one-fixed-seed engineering
screen; see [`docs/results/G5-stage1-seed0-20260715.md`](../docs/results/G5-stage1-seed0-20260715.md).
The binding contracts remain [`CLAUDE.md`](../CLAUDE.md), the
[experiment protocol](../docs/03-experiment-protocol.md), and the
[EgoStitch specification](../docs/05-egostitch-spec.md).
