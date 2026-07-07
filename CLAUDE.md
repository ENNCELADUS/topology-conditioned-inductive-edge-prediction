# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **research-paper project** targeting ICLR 2027: *Topology-Conditioned Inductive
Edge Prediction*. It contains design documents, result notes, self-contained HTML
figures, and a curated literature library. There is **no implementation code and no
test suite yet** — the work so far is problem framing, methodology, an experiment
protocol, and a model proposal awaiting approval.

`README.md` is the human-facing entry point (orientation, status, structure map,
reading order). This file (`CLAUDE.md`) holds the binding *constraints* an agent
must respect. When implementation begins, code goes under `src/`, and
`docs/03-experiment-protocol.md` is the contract it must implement.

## Repository layout

```
README.md                        entry point / navigation hub
CLAUDE.md                        this file — agent guardrails
docs/
  01-blueprint.md                top-level paper blueprint (locked decisions)
  02-methodology.md              abstract method contract + training objective
  03-experiment-protocol.md      run/eval contract: baselines, E1–E7, metrics
  04-model-proposal.md           EgoStitch model (AWAITING APPROVAL)
  lit-review-plan.md             review plan, claims K1–K5, terminology guardrail
  results/
    E2-pair-to-topology-gap.md   motivating result note
figures/
  e2-gap.html                    edge-vs-topology contrast figure
  positioning.html               2×2 taxonomy positioning figure
src/
  README.md                      stub: implementation contract pointer (no code yet)
literature/                      curated reference library (gitignored, unchanged)
```

## The core thesis (do not let it drift)

Inductive edge prediction is framed as **topology-conditioned binary
classification**, *not* independent pairwise scoring and *not* graph generation.
For an unseen node pair `(u, v)` with frozen features, the model builds local
topological context and predicts the 0/1 edge label; predictions are then assembled
into a graph and graded with topology metrics.

`docs/lit-review-plan.md` §5 (Terminology Guardrail) is binding for all writing:
generated local topology is always **intermediate context**, never the final
output. The final task is always **binary edge prediction for queried pairs**.
Every section must answer "how does this help predict `edge(u, v)` for unseen
nodes?" If a draft starts describing graph generation as the task, it has drifted.

## Document reading order (know which is authoritative)

1. `docs/01-blueprint.md` — top-level paper blueprint. Its §10 **Locked Decisions**
   are fixed constraints; do not silently renegotiate them.
2. `docs/02-methodology.md` — the abstract method contract (retrieval → local
   topology → topology-conditioned classifier) and the training objective
   `L = L_edge + λ_real·L_real + λ_ssl·L_ssl + λ_recon·L_recon`.
3. `docs/03-experiment-protocol.md` — the **repository-local, self-contained**
   experiment contract: the locked per-query local-scaffold method boundary (§0),
   baseline ladder B0–B3/Oracle (§2), experiment matrix E1–E7 (§3), evaluation
   protocol, and run order. Source of truth for what to run and how to grade it.
4. `docs/04-model-proposal.md` — **EgoStitch**, the concrete proposed model (dual
   ego-net imagination + community codebook, replacing the retrieved-and-thresholded
   scaffold). **Status: awaiting approval.** It adds baseline B5 and reframes the
   original §0 scaffold as an ablation arm. Treat it as a proposal, not settled
   design, until the user approves it.
5. `docs/results/E2-pair-to-topology-gap.md` — the motivating result note (numbers below).
6. `docs/lit-review-plan.md` — literature-review plan, claims K1–K5, the 2×2
   taxonomy, and the terminology guardrail.

When these conflict, the more specific/later document refines the earlier one, but
locked decisions and the locked §0 contract override casual changes — flag conflicts
rather than resolving them unilaterally.

## Load-bearing facts (the E2 result)

The whole motivation rests on one measured gap: the frozen B0 pairwise scorer
reaches **AUROC 0.676 / AUPRC 0.691** yet assembles into a Benchmark-A graph with
**graph similarity 0.235**, degree MMD 17.2, clustering MMD 11.8, spectral MMD 22.1,
relative density 0.684. `docs/04-model-proposal.md` §4.6 maps each of these failure
axes to a specific EgoStitch mechanism. If you touch these numbers, keep them
consistent across `docs/results/E2-pair-to-topology-gap.md`,
`docs/03-experiment-protocol.md`, `docs/04-model-proposal.md`, and
`figures/e2-gap.html` — they are quoted in all four.

## Naming conventions (neutral placeholders — keep them neutral)

The experiment docs deliberately use abstract, dataset-agnostic names so the
protocol reads as a general graph-ML benchmark:

- **Benchmarks:** `Benchmark-A` (primary node-disjoint split), `Benchmark-B`, `Benchmark-C`.
- **Baselines:** `B0` (independent scorer), `B0-alt`, `B1` (retrieval, no adjacency),
  `B2-global` / `B2-static`, `B3` (topology-aware loss), `B5` (neural-SBM residual,
  from the EgoStitch proposal), `Ours`, `Oracle` (upper bound, violates protocol).
- **Method internals:** `T_ij` (local scaffold), `C(u)`/`G(u)` (candidate sets),
  `x_u` (frozen features), `p_ij` (edge probability).

Don't substitute real dataset names into these documents unless the user asks.

## Integrity gates (non-negotiable, part of the contribution)

The strict inductive protocol is itself a contribution. Any experiment or claim must
honor: node-disjoint train/test splits; retrieval and topology construction never
touch the target test graph; the queried edge is masked/standardized inside the
scaffold; and **edge-level metrics and assembled-graph metrics are always reported
together** (no single-metric-family claims). See `docs/03-experiment-protocol.md`
§E5 and `docs/02-methodology.md` §6.

## Literature library conventions

`literature/` is a hand-curated reference collection, not a dependency, and is
gitignored:

- `literature/models/<topic>/<subtopic>/` — PDFs, filename pattern
  `YEAR_venue_short_title.pdf` (e.g. `2019_icml_learning_discrete_structures_...pdf`).
  `graph_structure_learning/README.md` documents the subfolder taxonomy — keep new
  PDFs consistent with it and update that README when adding a subfolder.
- `literature/research_reports/` — markdown synthesis notes, one topic per file.
- `literature/data/` — dataset/negative-sampling references.

Papers are cited in `docs/04-model-proposal.md` by arXiv ID; that file's §8 is the
verified reference list. New citations should be verified against arXiv or a local
PDF before being quoted as fact (the proposal notes all its citations were so verified).

## Figures

`figures/e2-gap.html` (edge-vs-topology contrast) and `figures/positioning.html`
(2×2 taxonomy placing the method in the inductive + structure-aware cell) are
standalone HTML — open directly in a browser, no server. Regenerate them from
repository-local data only; the figures must stay reproducible from the numbers in
the markdown notes.
