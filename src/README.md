# src/

This package implements the benchmark/evaluation stack, frozen B0-family baselines,
G1–G3 analyses, and the pre-registered EgoStitch Stage-1 model and gate.

- `data/`: verified benchmark artifacts, frozen features, packed caches, grounding,
  pair streams, and ego targets.
- `model/`: B0/B0-alt plus `egostitch/` Stage-1 modules.
- `eval/`: edge metrics, assembled-graph metrics, calibration, and ego fidelity.
- `experiments/`: G1/G2/G3, `B0+cal`, and the G5 Stage-1 gate.
- `train_b0.py` / `train_egostitch.py`: baseline and auto-sized DDP training workers.
- `score_universe.py`: shardable scoring and strict artifact merge.

The current formal G5 Stage-1 run has one completed seed and no three-seed verdict;
see [`docs/results/G5-stage1-seed0-20260715.md`](../docs/results/G5-stage1-seed0-20260715.md).
The binding contracts remain [`CLAUDE.md`](../CLAUDE.md), the
[experiment protocol](../docs/03-experiment-protocol.md), and the
[EgoStitch specification](../docs/05-egostitch-spec.md).
