<div align="center">

# Topology-Conditioned Inductive Edge Prediction

### Predict edges for *unseen* nodes by conditioning each decision on generated local topology — then grade the assembled graph, not just the pairs.

<p>
  <img alt="Venue" src="https://img.shields.io/badge/target-ICLR%202027-b31b1b">
  <img alt="Type" src="https://img.shields.io/badge/paper-empirical%20ML%20method-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-design%20phase-yellow">
  <img alt="Code" src="https://img.shields.io/badge/implementation-baselines%20%2B%20gates-green">
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

This repository holds the design and pre-implementation gate pipeline for an ICLR 2027
paper. The benchmark loaders, evaluation library, B0/B0-alt baselines, score-once
artifact pipeline, and gates G1–G3 are implemented and tested. **EgoStitch itself is not
implemented**; G1 is closed with the B0-alt architecture-independence arm, and G3
(Oracle) has passed, so the next stage is model implementation.

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
| Edge | AUROC / AUPRC (degree-corrected, ratio-1) | **0.7055 / 0.7303** | final G1 B0 row |
| Edge | AUROC / AUPRC (hard feature, ratio-1) | **0.5696 / 0.6175** | hard-negative stress test |
| Assembled graph | graph similarity | **5.77e-7** | poor; composite passed perturbation check |
| Assembled graph | relative density | 0.9977 | density-matched operating point |
| Assembled graph | degree / clustering / spectral MMD ratio | **13.0768 / 11.9273 / 18.0931** | reference floor = 1; lower is better |
| Edge (B0-alt) | AUROC / AUPRC (degree-corrected, ratio-1) | **0.6936 / 0.7325** | alternate F0-MLP architecture |
| Assembled graph (B0-alt) | graph similarity | **2.29e-8** | architecture-independence arm |
| Assembled graph (B0-alt) | degree / clustering / spectral MMD ratio | **15.8304 / 13.4718 / 23.4734** | gap persists and is larger |

The final two-architecture G1 threshold sweep and all negative-regime rows are recorded in
[`outputs/runs/g1_b0_b0_alt_20260713T165714Z/g1_tables.md`](outputs/runs/g1_b0_b0_alt_20260713T165714Z/g1_tables.md).
G2 reports measured soft-score overlap 0.4799 versus the minimum 0.0550 required to
reach the reference triangle count. G3's Oracle-blend arm reports MMD-ratio headroom
1.723 / 3.885 / 1.998 for degree / clustering / spectral, with composite ratio 2425.56;
the feature-insufficiency stop rule is not triggered. The B0-alt result closes G1's
architecture-independence requirement. PA-null is reported
alongside B0 because it wins some easy and feature-hard edge regimes.

The latest checkpoint-only evaluation rerun (`legacy_v31_s47_20260712T193900Z`, not a
new formal training acceptance) reached balanced test AUROC/AUPRC **0.8052 / 0.8184**
and its G1 degree-corrected ratio-1 row was **0.7996 / 0.8133**. Its archived assembled
metrics were recomputed with the canonical evaluator: graph similarity **2.63e-7** and
degree / clustering / spectral MMD ratios **13.8456 / 11.6277 / 19.9774** at relative
density 0.9784. Thus the stronger edge scorer does not shrink the topology gap. The
complete closeout package is
[`outputs/deliverables/g1_closeout_20260713/`](outputs/deliverables/g1_closeout_20260713/).

## The Proposed Method (EgoStitch)

> **Status: approved (2026-07-09).** Design proposal approved and gate G4 signed off —
> [`docs/06-egostitch-spec.md`](docs/06-egostitch-spec.md) is the active implementation
> contract (algorithm spec + benchmark/data contract + four-H20 execution design). Design
> rationale: [`docs/04-model-proposal.md`](docs/04-model-proposal.md); review record:
> [`docs/05-review-report.md`](docs/05-review-report.md). G1 (including B0-alt), G2, and
> G3 (Oracle) are complete and support proceeding to implementation.

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
  03-experiment-protocol.md      run/eval contract: baselines, E1–E7, metrics, gates G1–G5
  04-model-proposal.md           EgoStitch model rationale  (APPROVED 2026-07-09)
  05-review-report.md            novelty check + 5-persona review record
  06-egostitch-spec.md           algorithm + data + 4×H20 spec  ← implementation contract (G4 signed off)
  lit-review-plan.md             review plan, claims K1–K5, terminology guardrail
  results/
    E2-pair-to-topology-gap.md   the motivating result note
figures/
  e2-gap.html                    edge-vs-topology contrast (open in a browser)
  positioning.html               2×2 taxonomy positioning figure
data/
  README.md                      neutral benchmark artifact manifest
  benchmark_2025_neurips/        graph, edge lists, split artifacts, evaluation buckets
  features/                      neutral node-feature index
src/
  data/                          verified benchmark/features + pair batching
  eval/                          edge and assembled-graph metrics
  model/                         frozen B0 and B0-alt scorers
  experiments/                   implemented G1/G2/G3 cached-score analyses
  train_b0.py                    baseline training CLI
  score_universe.py              score-once artifact CLI
hpc/
  run.sh                         direct fixed 4×H20 runner
  README.md                      required target environment + exact experiment runbook
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
| 4 | [`docs/04-model-proposal.md`](docs/04-model-proposal.md) | EgoStitch rationale (approved 2026-07-09) |
| 5 | [`docs/06-egostitch-spec.md`](docs/06-egostitch-spec.md) | **Implementation contract**: algorithm, data/batch contract, four-H20 execution |
| 6 | [`docs/results/E2-pair-to-topology-gap.md`](docs/results/E2-pair-to-topology-gap.md) | Motivating result |
| 7 | [`docs/lit-review-plan.md`](docs/lit-review-plan.md) | Review plan, claims, terminology guardrail |

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
- [x] **EgoStitch proposal** — approved 2026-07-09; reviewed via novelty check + 5-persona panel ([`docs/05-review-report.md`](docs/05-review-report.md)).
- [x] **G4 spec freeze** — [`docs/06-egostitch-spec.md`](docs/06-egostitch-spec.md) signed off (algorithm + benchmark data contract + batch sampler + four-H20 execution).
- [x] **Baseline + gate pipeline** — benchmark/features, B0/B0-alt training, cached scoring, G1/G2/G3 analyses, and tests are implemented.
- [x] **G1 + G2** — B0/PA-null, B0-alt architecture replication, and the checkpoint-aligned edge-independence ceiling are complete; G1 closeout artifacts are under `outputs/deliverables/g1_closeout_20260713/`.
- [x] **Latest legacy v3.1 robustness rerun** — canonical G1 confirms that the stronger scorer preserves the topology gap; artifacts are included in `outputs/deliverables/g1_closeout_20260713/`.
- [x] **G3 gate** — Oracle row passed; Oracle-blend shows substantial headroom over B0 and the feature-insufficiency stop rule is not triggered. Results are under [`outputs/deliverables/b0_v31_breadth_first_20260711/g3/`](outputs/deliverables/b0_v31_breadth_first_20260711/g3/).
- [ ] **EgoStitch implementation** under `src/`, per [`docs/06-egostitch-spec.md`](docs/06-egostitch-spec.md) and the [experiment protocol](docs/03-experiment-protocol.md).
- [ ] **Experiments** in priority order: EgoStitch implementation → E1/E3 main + baselines → E4 ablations → E5 integrity gates → E7 (load-bearing) → E6 breadth.

## HPC execution

The required target environment is a fixed container with 4× NVIDIA H20 GPUs. Run
`hpc/run.sh check`, then use the same runner for cached scoring and G1/G2; run G3 directly
over the cached candidate universe. Formal E2
(B0 V3.1) training runs **only** through `hpc/run.sh train
configs/b0_v31_breadth_first.yaml`, which drives the production
`python -m src.e2_pipeline` entry (pack → probe → projection → 30-epoch DDP train via
`accelerate launch --num_processes 4` across all 4 GPUs); direct
`python -m src.train_b0 --max-steps N` is debug-only, and `B0-alt` keeps its own direct
`python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml` training CLI,
outside this E2-only optimization. The host, repository/data paths, required software
versions, and the `nohup` form are pinned in [`hpc/README.md`](hpc/README.md). The
live four-H20 cold-run acceptance is pending, so the 60-minute target is not yet a
verified result. There is no job scheduler (e.g. Slurm).

## Literature

`literature/` is a hand-curated reference collection (gitignored, not a dependency). PDFs
follow `YEAR_venue_short_title.pdf` and are filed by topic; see
`literature/models/graph_structure_learning/README.md` for the taxonomy. Citations in the
model proposal are by arXiv ID and were verified against arXiv or local PDFs.
