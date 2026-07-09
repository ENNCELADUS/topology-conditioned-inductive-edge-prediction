<div align="center">

# Topology-Conditioned Inductive Edge Prediction

### Predict edges for *unseen* nodes by conditioning each decision on generated local topology — then grade the assembled graph, not just the pairs.

<p>
  <img alt="Venue" src="https://img.shields.io/badge/target-ICLR%202027-b31b1b">
  <img alt="Type" src="https://img.shields.io/badge/paper-empirical%20ML%20method-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-design%20phase-yellow">
  <img alt="Code" src="https://img.shields.io/badge/implementation-pending-lightgrey">
</p>

<p>
  <a href="#the-idea">The Idea</a>
  ◆ <a href="#the-motivating-result-e2">Motivating Result</a>
  ◆ <a href="#the-proposed-method-egostitch">Method</a>
  ◆ <a href="#repository-map">Repository Map</a>
  ◆ <a href="#reading-order">Reading Order</a>
  ◆ <a href="#status--roadmap">Status &amp; Roadmap</a>
</p>

</div>

---

## TL;DR

This repository holds the design of an ICLR 2027 paper. **There is no implementation
code yet** — it currently contains the problem framing, methodology, a self-contained
experiment protocol, a model proposal awaiting approval, one motivating result, and a
curated literature library.

- **Task:** given two *unseen* nodes with frozen feature vectors, predict whether an
  edge exists between them (binary classification), under a strict inductive protocol
  (no access to the target graph at test time).
- **Core claim:** independent pairwise scoring is *topology-blind* — it can score pairs
  well yet assemble into a structurally implausible graph.
- **Fix:** condition each edge decision on **generated local topology** built only from
  frozen features, and evaluate **edge-level metrics and assembled-graph metrics together**.

> **Agents/new contributors:** read [`CLAUDE.md`](CLAUDE.md) first — it holds the binding
> constraints (the terminology guardrail, locked decisions, integrity gates, and the
> neutral naming conventions this project deliberately uses).

## The Idea

Inductive edge prediction is framed as **topology-conditioned binary classification**,
*not* independent pairwise scoring and *not* graph generation. For a queried pair
`(u, v)`, the model constructs local topological context from feature-only candidate
neighborhoods, then predicts the `0/1` edge label *in that context*. Predictions over a
query set are assembled into a graph and graded with topology metrics.

> **Guardrail:** generated local topology is always **intermediate context**, never the
> final output. The final task is always **binary edge prediction for queried pairs**.
> See [`docs/lit-review-plan.md`](docs/lit-review-plan.md) §5.

The positioning is a 2×2 map — the target cell is *inductive-from-features* **and**
*structure-aware* (see [`figures/positioning.html`](figures/positioning.html)):

```
                          structure-blind              structure-aware
  inductive from          independent edge scoring      ★ THIS WORK:
  features                (clean, but no topology)       topology-conditioned
                                                         inductive edge prediction

  needs target graph      feature + observed-graph       transductive graph-native
  at inference            hybrids (not strict inductive) link prediction (leaks)
```

## The Motivating Result (E2)

A strong independent pairwise scorer (`B0`) achieves reasonable *edge-level* scores while
assembling into a poor *graph*. This "pair-to-topology gap" is the reason the method
exists. Full note: [`docs/results/E2-pair-to-topology-gap.md`](docs/results/E2-pair-to-topology-gap.md);
figure: [`figures/e2-gap.html`](figures/e2-gap.html).

| Level | Metric | Value | Reading |
|---|---|---:|---|
| Edge | AUROC / AUPRC | **0.676 / 0.691** | reasonable |
| Assembled graph | graph similarity | **0.235** | poor |
| Assembled graph | relative density | 0.684 | sparse at this operating point |
| Assembled graph | degree MMD | 17.2 | high (ideal ≈ 0) |
| Assembled graph | clustering MMD | 11.8 | high (ideal ≈ 0) |
| Assembled graph | spectral MMD | 22.1 | high (ideal ≈ 0) |

The gap **widens with graph size** (graph similarity falls from 0.320 at 20 nodes to
0.179 at 200), and shows up at both denser and sparser operating points — so it is not
fixable by threshold tuning alone.

## The Proposed Method (EgoStitch)

> **Status: awaiting approval.** Treat the architecture as a proposal, not settled design,
> until sign-off. Spec: [`docs/04-model-proposal.md`](docs/04-model-proposal.md).

For each queried pair `(i, j)`, each endpoint **imagines its own ego-network** (latent
neighbor nodes with existence probabilities, local adjacency, and a degree budget)
conditioned on its frozen features and a learned **community codebook**. The two imagined
ego-nets are **stitched** into a local scaffold `T̂_ij`, and the edge decision fuses four
evidence channels: the node-intrinsic pairwise logit, membership, closure, and
community/capacity. Each channel maps to a specific E2 failure axis it is meant to repair
(degree budgets → density/degree; slot–slot adjacency + closure → clustering; community
codebook → spectrum).

```text
step 1  community coding      z_u, F_u, d̂_u       = Tokenize(x_u)            (per node, cached)
step 2  ego-net imagination   S_u = {(h_u^k, π_u^k)} = Imagine(x_u, z_u, G(u)) (per node, cached)
step 3  stitch                T̂_ij = Stitch(S_i, S_j, {i, j})                 (per pair)
step 4  decide                p_ij = σ(pair_logit(i,j) + g·Fuse(s1,s2,s3,s4)) (per pair)
```

Training objective (maps onto the methodology plan's four terms):
`L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon`.

## Repository Map

```
README.md                        you are here — orientation hub
CLAUDE.md                        binding constraints for agents (read this first)
docs/
  01-blueprint.md                top-level paper blueprint + locked decisions
  02-methodology.md              abstract method contract + training objective
  03-experiment-protocol.md      run/eval contract: baselines, E1–E7, metrics  ← implementation spec
  04-model-proposal.md           EgoStitch model  (AWAITING APPROVAL)
  lit-review-plan.md             review plan, claims K1–K5, terminology guardrail
  results/
    E2-pair-to-topology-gap.md   the motivating result note
figures/
  e2-gap.html                    edge-vs-topology contrast (open in a browser)
  positioning.html               2×2 taxonomy positioning figure
data/
  README.md                      neutral benchmark input/output contract (no concrete data)
src/
  README.md                      implementation-contract stub (no code yet)
literature/                      curated reference library (gitignored)
  models/                        PDFs by topic; filename YEAR_venue_short_title.pdf
  research_reports/              markdown synthesis notes
  data/                          dataset / negative-sampling references
```

**Baseline & benchmark names are deliberately neutral placeholders** (`Benchmark-A/B/C`;
`B0`, `B1`, `B2-global/static`, `B3`, `B5`, `Ours`, `Oracle`) so the protocol reads as a
general graph-ML benchmark. Don't substitute real dataset names unless asked.

## Reading Order

| # | Document | Role |
|---|---|---|
| 1 | [`docs/01-blueprint.md`](docs/01-blueprint.md) | Paper blueprint; **§10 Locked Decisions** are fixed |
| 2 | [`docs/02-methodology.md`](docs/02-methodology.md) | Method contract + training objective |
| 3 | [`docs/03-experiment-protocol.md`](docs/03-experiment-protocol.md) | **Source of truth** for what to run and how to grade it |
| 4 | [`docs/04-model-proposal.md`](docs/04-model-proposal.md) | EgoStitch (awaiting approval) |
| 5 | [`docs/results/E2-pair-to-topology-gap.md`](docs/results/E2-pair-to-topology-gap.md) | Motivating result |
| 6 | [`docs/lit-review-plan.md`](docs/lit-review-plan.md) | Review plan, claims, terminology guardrail |

When documents conflict, the more specific/later one refines the earlier — but locked
decisions and the locked §0 method boundary override casual changes.

## Evaluation & Integrity Gates

Every claim reports **edge-level metrics** (AUROC, AUPRC, F1/MCC, calibration) **and
assembled-graph metrics** (graph similarity, relative density, degree/clustering/spectral
MMD) *together* — never one family alone. Non-negotiable gates (part of the contribution):
node-disjoint train/test splits; retrieval and topology construction never touch the
target test graph; the queried edge is masked/standardized inside the scaffold. See
[`docs/03-experiment-protocol.md`](docs/03-experiment-protocol.md) §E5.

## Status & Roadmap

- [x] **E2 — pair-to-topology gap** established (motivating result + figure).
- [x] **Blueprint, methodology, and experiment protocol** written and locked.
- [ ] **EgoStitch proposal** — awaiting approval ([`docs/04-model-proposal.md`](docs/04-model-proposal.md)).
- [ ] **Implementation** under `src/`, per the [experiment protocol](docs/03-experiment-protocol.md).
- [ ] **Experiments** in priority order: E2 (rerun) → E1/E3 main + baselines → E4 ablations → E5 integrity gates → E6/E7 breadth.

## Literature

`literature/` is a hand-curated reference collection (gitignored, not a dependency). PDFs
follow `YEAR_venue_short_title.pdf` and are filed by topic; see
`literature/models/graph_structure_learning/README.md` for the taxonomy. Citations in the
model proposal are by arXiv ID and were verified against arXiv or local PDFs.
