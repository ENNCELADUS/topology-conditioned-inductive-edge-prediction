# src/

Implementation code lives here. **Nothing is implemented yet** — this project is
currently in the design phase (see the repository [`README.md`](../README.md) and
[`CLAUDE.md`](../CLAUDE.md)).

## Implementation contract

When code lands, it must implement the experiment contract defined in
[`../docs/03-experiment-protocol.md`](../docs/03-experiment-protocol.md):

- the locked per-query local-scaffold method boundary (§0),
- the baseline ladder B0–B3, B5, `Ours`, `Oracle` (§2),
- the E1–E7 experiment matrix (§3),
- the joint edge-level + assembled-graph evaluation protocol (§4), and
- the integrity gates (§E5) — node-disjoint splits, no target-graph access at test
  time, queried-edge masking.

The proposed model to build is **EgoStitch**
([`../docs/04-model-proposal.md`](../docs/04-model-proposal.md)), **pending approval**.
Until it is approved, treat its architecture as a proposal rather than a spec.
