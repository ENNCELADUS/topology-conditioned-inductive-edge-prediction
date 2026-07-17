<div align="center">

# Topology-Conditioned Inductive Edge Prediction

### Predict edges for *unseen* nodes by conditioning each decision on generated local topology — then grade the assembled graph, not just the pairs.

<p>
  <img alt="Venue" src="https://img.shields.io/badge/target-ICLR%202027-b31b1b">
  <img alt="Type" src="https://img.shields.io/badge/paper-empirical%20ML%20method-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-G5%20rev--3.0%20build-orange">
  <img alt="Code" src="https://img.shields.io/badge/implementation-EgoStitch%20Stage--1-green">
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

This repository holds the implementation and experiment pipeline for an ICLR 2027
paper. The benchmark loaders, evaluation library, B0/B0-alt baselines, score-once
artifact pipeline, gates G1–G3, and the frozen-s0 EgoStitch Stage-1 model are
implemented and tested. The replacement fixed-Seed-0 screen completed on 2026-07-16
under its bound registration and produced a formal **`cut`** verdict: both guards
passed, but EgoStitch failed all three required topology-dominance checks against
the calibrated-B0 ladder. The frozen-s0 scalar head is now a motivating result and
ablation rung; the active G5 build line is the approved rev-3.0 end-to-end
stitched-topology-conditioned pair encoder. This one-seed engineering decision does
not replace E1/E3's multi-seed Holm inference.

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

A frozen independent pairwise scorer (`B0`) achieves reasonable *edge-level* scores while
assembling into a poor *graph*. This "pair-to-topology gap" is the reason the method
exists. Full note: [`docs/results/E2-pair-to-topology-gap.md`](docs/results/E2-pair-to-topology-gap.md);
figure: [`figures/e2-gap.html`](figures/e2-gap.html).

> **Canonical metric rerun (2026-07-14):** GS/RD now match the official benchmark evaluation and were
> formally rerun on the frozen score artifacts over all 500 fixed induced subgraphs. The separately
> reported MMD ratios remain the canonical-run values.

| Level | Metric | Value | Reading |
|---|---|---:|---|
| Edge | AUROC / AUPRC (degree-corrected, ratio-1) | **0.7055 / 0.7303** | final G1 B0 row |
| Edge | AUROC / AUPRC (hard feature, ratio-1) | **0.5696 / 0.6175** | hard-negative stress test |
| Assembled graph | BFS-macro GS / RD | **0.3122 / 0.4223** | local topology; higher GS and RD closer to 1 are better |
| Assembled graph | global simple-edge RD | **0.9977** | whole-graph non-self edge-count calibration |
| Assembled graph | degree / clustering / spectral MMD ratio | **13.0768 / 11.9273 / 18.0931** | reference floor = 1; lower is better |
| Edge (B0-alt) | AUROC / AUPRC (degree-corrected, ratio-1) | **0.6936 / 0.7325** | alternate F0-MLP architecture |
| Assembled graph (B0-alt) | BFS-macro GS / RD | **0.3458 / 0.4508** | architecture-independence arm |
| Assembled graph (B0-alt) | global simple-edge RD | **0.9987** | independently calibrated threshold |
| Assembled graph (B0-alt) | degree / clustering / spectral MMD ratio | **15.8304 / 13.4718 / 23.4734** | gap persists and is larger |

The final two-architecture G1 threshold sweep and all negative-regime rows are recorded in
[`outputs/deliverables/g1_graph_metrics_20260714/g1_tables.md`](outputs/deliverables/g1_graph_metrics_20260714/g1_tables.md).
G2 reports measured soft-score overlap 0.4799 versus the minimum 0.0550 required to
reach the reference triangle count. G3's Oracle-blend arm reports MMD-ratio headroom
1.723 / 3.885 / 1.998 for degree / clustering / spectral; Oracle-topo reaches BFS-macro GS
0.5030, 1.612 times B0, so the feature-insufficiency stop rule is not triggered. The
B0-alt result closes G1's architecture-independence requirement. PA-null is reported
alongside B0 because it wins some easy and feature-hard edge regimes.

The latest checkpoint-only evaluation rerun (`legacy_v31_s47_20260712T193900Z`, not a
new formal training acceptance) reached balanced test AUROC/AUPRC **0.8052 / 0.8184**
and its G1 degree-corrected ratio-1 row was **0.7996 / 0.8133**. Its archived assembled
metrics were recomputed with the official evaluator: BFS-macro GS/RD
**0.3813 / 0.5002** at global simple-edge RD **0.9784**, alongside MMD ratios
**13.8456 / 11.6277 / 19.9774**. Thus the stronger edge scorer improves GS but does not
close the topology gap. The complete closeout package is
[`outputs/deliverables/legacy_g1_graph_metrics_20260714/`](outputs/deliverables/legacy_g1_graph_metrics_20260714/).

## The Proposed Method (EgoStitch)

> **Status: frozen-s0 Stage-1 formally cut; rev-3.0 successor build unblocked
> (2026-07-17).** Design proposal approved and gate G4 signed off —
> [`docs/05-egostitch-spec.md`](docs/05-egostitch-spec.md) is the active implementation
> contract (algorithm spec + benchmark/data contract + auto-sized H20 execution design); its
> §14 records the approved e2e successor headline. Design
> rationale: [`docs/04-model-proposal.md`](docs/04-model-proposal.md) (rev 3.0). G1 (including
> B0-alt), G2, and G3 (Oracle) are complete. The binding frozen-s0 result is recorded in
> [`docs/results/G5-stage1-seed0-20260717.md`](docs/results/G5-stage1-seed0-20260717.md):
> matched edge AUPRC and degree-MMD guards pass, while clustering-MMD and matched
> BFS-macro GS/RD all fail. The rev-3.0 e2e screen now becomes the next G5 build.

For each queried pair `(i, j)`, each endpoint **imagines its own ego-network** (latent
neighbor nodes with existence probabilities, local adjacency, and a degree budget)
conditioned on its frozen features and a learned **community codebook**. The two imagined
ego-nets are **stitched** into a local scaffold `T̂_ij`. Since rev 3.0 the edge decision
is **end-to-end**: a from-scratch pair encoder over the two endpoints' raw token
sequences is *conditioned on the scaffold* — a structure-only **stitched-topology
encoder** (anchor labels, existence, multiplicity, degrees, and the scaffold's edge
structure; no content embeddings) produces token-level topology states that enter the
pair encoder through zero-initialized gated cross-attention, with content evidence on a
separate, independently ablatable pathway. No pretrained frozen classifier appears
anywhere in the model; the same checkpoint emits a pair-only logit, pair+content,
pair+topology, and the full logit for attribution.

```text
step 1  community coding      z_u, F_u, d̂_u       = Tokenize(x_u)            (per node, cached)
step 2  ego-net imagination   S_u = {(h_u^k, π_u^k)} = Imagine(x_u, z_u, G(u)) (per node, cached)
step 3  stitch                T̂_ij = Stitch(S_i, S_j, {i, j})                 (per pair)
step 4  encode + decide       t = STE(T̂_ij);  p_ij = σ(head(Trunk(tok_i, tok_j | t, c_content)))
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
  05-egostitch-spec.md           algorithm + data + auto-sized H20 spec  ← implementation contract (G4 signed off)
  lit-review-plan.md             review plan, claims K1–K5, terminology guardrail
  results/
    E2-pair-to-topology-gap.md   the motivating result note
    G5-stage1-seed0-20260715.md  superseded-registration training record
    G5-stage1-seed0-20260717.md  binding frozen-s0 screen (`cut`) + last.pt diagnostic
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
  model/                         frozen B0/B0-alt scorers + EgoStitch Stage-1 modules
  experiments/                   G1/G2/G3 analyses + pre-registered G5 Stage-1 gate
  train_b0.py                    baseline training CLI
  train_egostitch.py             auto-sized DDP EgoStitch training worker
  score_universe.py              score-once artifact CLI
hpc/
  run.sh                         direct auto-sized H20 runner
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
| 5 | [`docs/05-egostitch-spec.md`](docs/05-egostitch-spec.md) | **Implementation contract**: algorithm, data/batch contract, auto-sized H20 execution |
| 6 | [`docs/results/E2-pair-to-topology-gap.md`](docs/results/E2-pair-to-topology-gap.md) | Motivating result |
| 7 | [`docs/results/G5-stage1-seed0-20260717.md`](docs/results/G5-stage1-seed0-20260717.md) | Binding frozen-s0 Stage-1 result (`cut`) and diagnostic interpretation |
| 8 | [`docs/results/G5-stage1-seed0-20260715.md`](docs/results/G5-stage1-seed0-20260715.md) | Historical superseded-registration training record |
| 9 | [`docs/lit-review-plan.md`](docs/lit-review-plan.md) | Review plan, claims, terminology guardrail |

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
- [x] **EgoStitch proposal** — approved 2026-07-09; reviewed via novelty check + 5-persona panel.
- [x] **G4 spec freeze** — [`docs/05-egostitch-spec.md`](docs/05-egostitch-spec.md) signed off (algorithm + benchmark data contract + batch sampler + auto-sized H20 execution).
- [x] **Baseline + gate pipeline** — benchmark/features, B0/B0-alt training, cached scoring, G1/G2/G3 analyses, and tests are implemented.
- [x] **G1 + G2** — B0/PA-null, B0-alt architecture replication, and the checkpoint-aligned edge-independence ceiling are complete; the final benchmark-aligned G1 artifacts are under [`outputs/deliverables/g1_graph_metrics_20260714/`](outputs/deliverables/g1_graph_metrics_20260714/).
- [x] **Latest legacy v3.1 robustness rerun** — canonical G1 confirms that the stronger scorer preserves the topology gap; final artifacts are under [`outputs/deliverables/legacy_g1_graph_metrics_20260714/`](outputs/deliverables/legacy_g1_graph_metrics_20260714/).
- [x] **G3 gate** — Oracle row passed; Oracle-blend shows substantial headroom over B0 and the feature-insufficiency stop rule is not triggered. Final artifacts are under [`outputs/deliverables/g3_graph_metrics_20260714/`](outputs/deliverables/g3_graph_metrics_20260714/).
- [x] **EgoStitch Stage-1 implementation (frozen-s0 form)** — model, losses, data targets, auto-sized DDP worker, scoring, fidelity diagnostics, calibrated comparator, G5 runner, and tests are implemented under `src/`.
- [x] **E2E headline redesign (rev 3.0)** — approved 2026-07-16 after two user-as-reviewer rounds + a vault/arXiv novelty sweep: stitched-topology-conditioned pair encoder (no frozen anchor). Design record: [`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`](docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md); implementation summary: spec §14; Phase 0 is unblocked by the completed frozen-s0 screen: [`docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`](docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md).
- [x] **G5 frozen-s0 Stage-1 screening gate** — binding fixed-Seed-0 verdict `cut`: all three primary topology-dominance checks fail and both guards pass. The selected warm-start checkpoint is near S0; a diagnostic `last.pt` rerun proves later joint training moves ranking and passes matched GS, but still fails matched RD/clustering and regresses degree/spectral MMD. See the [result note](docs/results/G5-stage1-seed0-20260717.md) and [presentation-only status snapshot](docs/artifacts/2026-07-17-egostitch-status.html).
- [x] **Locked-decision disposition** — frozen-s0 scalar fusion is retired to motivating-arm + ablation status; rev 3.0 is the active G5 build line. Successor landing condition §14.3(1) is satisfied.
- [ ] **Experiments** in priority order: complete the rev-3.0 Phase-0 spec/protocol/config work and fresh five-arm registration → e2e conditioned-encoder Stage-1 screen → accepted later G5 stages → E1/E3 multi-seed main + baselines → E4 ablations (incl. E4.15–E4.17 attribution/structure/conditioning-depth) → E5 integrity gates → E7 (load-bearing) → E6 breadth.

## HPC execution

The required target environment is a container with one or more NVIDIA H20 GPUs. Run
`hpc/run.sh check`, then use the same runner for cached scoring and G1/G2; run G3 directly
over the cached candidate universe. Formal E2
(B0 V3.1) training runs **only** through `hpc/run.sh train
configs/b0_v31_breadth_first.yaml`, which drives the production
`python -m src.e2_pipeline` entry (pack → probe → projection → 30-epoch DDP train via
an auto-detected `accelerate launch --num_processes N` across all visible GPUs); direct
`python -m src.train_b0 --max-steps N` is debug-only, and `B0-alt` keeps its own direct
`python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml` training CLI,
outside this E2-only optimization. The host, repository/data paths, required software
versions, and the `nohup` form are pinned in [`hpc/README.md`](hpc/README.md). The
GPU count is recorded with each run, so throughput evidence is interpreted against
that exact hardware shape. There is no job scheduler (e.g. Slurm).

Formal EgoStitch Stage-1 training uses the same auto-detected multi-GPU orchestrator
with `configs/egostitch_stage1_breadth_first.yaml` and `src.train_egostitch`; it must
not be replaced by a hard-coded single-GPU launch. The validated Seed-0 run used two
visible H20s (`world_size=2`).

## Literature

`literature/` is a hand-curated reference collection (gitignored, not a dependency). PDFs
follow `YEAR_venue_short_title.pdf` and are filed by topic; see
`literature/models/graph_structure_learning/README.md` for the taxonomy. Citations in the
model proposal are by arXiv ID and were verified against arXiv or local PDFs.
