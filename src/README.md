# src/

The implemented pipeline verifies benchmark artifacts and inductive partitions in
`data/`, trains the frozen B0 V3.1 and B0-alt scorers, scores pair universes once, and
runs the G1/G2 gate analyses over cached score artifacts. The V3.1 F1 path preloads the
operative raw-token features into a host-memory cache before using the frozen
length-bucketed loader contract; the F0 path uses its pooled matrix cache. Shared edge-
and graph-level metrics live in `eval/`. EgoStitch itself remains gated and is not yet
implemented; see the repository [`README.md`](../README.md) and
[`CLAUDE.md`](../CLAUDE.md) for status and contracts.
