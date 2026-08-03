<div align="center">

# Topology-Conditioned Inductive Edge Prediction

### Predict edges for *unseen* nodes by conditioning each decision on generated local topology — then grade the assembled graph, not just the pairs.

<p>
  <img alt="Venue" src="https://img.shields.io/badge/target-ICLR%202027-b31b1b">
  <img alt="Type" src="https://img.shields.io/badge/paper-empirical%20ML%20method-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-G5%20e2e%20rev--3.2%20build-orange">
  <img alt="Code" src="https://img.shields.io/badge/implementation-EgoStitch%20E2E%20single--stage-green">
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
artifact pipeline, and gates G1–G3 are implemented and tested. Two G5 screens have
completed, and both returned a formal **`cut`**:

- the **frozen-s0** EgoStitch screen (2026-07-16/17) — both guards passed, but all
  three required topology-dominance checks against the calibrated-B0 ladder failed:
  [`docs/results/G5-stage1-seed0-20260717.md`](docs/results/G5-stage1-seed0-20260717.md);
- the **rev-3.0 end-to-end** screen (2026-07-24) — multi-label; training was valid and
  liveness passed:
  [`docs/results/G5-e2e-stage1-seed0-20260724.md`](docs/results/G5-e2e-stage1-seed0-20260724.md).

Both result notes stay citable as history. The frozen-s0 *implementation* that produced
the first last exists at commit `dcae090` and is deleted in the current two-stage-cleanup
worktree pending a cleanup commit. The verdict and `outputs/egostitch_stage1/` are kept
unchanged, and the source paths those notes cite resolve at `dcae090`; the registration
snapshot that governed it is deleted along with the rest of `docs/registrations/` (the
owner has withdrawn the registration mechanism —
[`docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`](docs/superpowers/specs/2026-08-02-three-component-refactor-design.md)
§10). Both are fixed-Seed-0 engineering screens: neither replaces E1/E3's multi-seed
Holm inference.

The active build line is **rev-3.2** of the end-to-end stitched-topology-conditioned
pair encoder, run directly (the owner has withdrawn the preregistration/formal-run
gating mechanism):

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal   # or another trained-arm config
```

The run trains on `V_fit`, validates on the single 512-node `V_hold`, and executes
`pack → train → publish` through the shared orchestrator. Quality telemetry
(eligibility, liveness, slot collapse, margins) is recorded but never blocks
completion, publication, scoring, or evaluation.

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

> **Provenance note (2026-08-03):** the B0-alt implementation (`src/model/b0_alt.py`,
> `F0PairMLP`) and its config were removed from the code tree by owner decision. Every
> B0-alt value above is retained unchanged; reproducing them requires checking out a
> commit at or before `7842684`.

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

> **Status: frozen-s0 screen formally cut (2026-07-17); rev-3.0 e2e screen also
> formally cut (2026-07-24); rev-3.2 is the active single-stage build line.**
> Design proposal approved and gate G4 signed off —
> [`docs/05-egostitch-spec.md`](docs/05-egostitch-spec.md) is the active implementation
> contract (algorithm spec + benchmark/data contract + auto-sized H20 execution design); its
> §13.19 defines plan-bound execution and its §14 records the approved e2e
> headline. Design rationale:
> [`docs/04-model-proposal.md`](docs/04-model-proposal.md) (rev 3.0). G1 (including
> B0-alt), G2, and G3 (Oracle) are complete. The binding frozen-s0 result is recorded in
> [`docs/results/G5-stage1-seed0-20260717.md`](docs/results/G5-stage1-seed0-20260717.md):
> matched edge AUPRC and degree-MMD guards pass, while clustering-MMD and matched
> BFS-macro GS/RD all fail. That note is retained evidence — its implementing code last
> exists at `dcae090` and is deleted in the current cleanup worktree pending commit. The
> rev-3.0 e2e screen's historical four-trained-arm-plus-control result is
> recorded in [`docs/results/G5-e2e-stage1-seed0-20260724.md`](docs/results/G5-e2e-stage1-seed0-20260724.md):
> training was valid (liveness passed), BFS-macro GS passes but clustering-MMD and
> BFS-macro RD fail, the matched-AUPRC guard fails, and pathway attribution and the
> structure-destruction control both fail to establish a topology-conditioning gain.
> The owner-side disposition subsequently advanced the build through rev-3.1 to the
> active rev-3.2 eight-arm contract; it does not alter the historical `cut` verdict.

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
    G5-stage1-seed0-20260717.md  binding frozen-s0 screen (`cut`) — retained evidence, retired code
    G5-e2e-stage1-seed0-20260724.md  binding rev-3.0 e2e screen (`cut`)
figures/
  e2-gap.html                    edge-vs-topology contrast (open in a browser)
  positioning.html               2×2 taxonomy positioning figure
data/
  README.md                      neutral benchmark artifact manifest
  benchmark_2025_neurips/        graph, edge lists, split artifacts, evaluation buckets
  features/                      neutral node-feature index
src/
  data/                          verified benchmark/features + pair batching + V_fit/V_hold partition
  eval/                          edge and assembled-graph metrics
  model/                         B0-V3.1 pairwise baseline + EgoStitch three-component model
                                  (generator/encoder/classifier, each independently swappable)
  experiments/                   G1/G2/G3 analyses + G5 e2e gate + probes
  train_b0.py                    baseline training CLI
  train_egostitch.py             auto-sized DDP EgoStitch e2e training worker
  e2_pipeline.py                 production orchestrator: pack → train → publish
  score_universe.py              score-once artifact CLI
hpc/
  run.sh                         direct auto-sized H20 runner (check / B0 + EgoStitch e2e train / score / merge / G1 / G2)
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
| 7 | [`docs/results/G5-stage1-seed0-20260717.md`](docs/results/G5-stage1-seed0-20260717.md) | Binding frozen-s0 result (`cut`) and diagnostic interpretation; retained evidence, retired code |
| 8 | [`docs/results/G5-e2e-stage1-seed0-20260724.md`](docs/results/G5-e2e-stage1-seed0-20260724.md) | Binding rev-3.0 e2e screen result (`cut`); historical evidence for the rev-3.2 successor |
| 9 | [`hpc/README.md`](hpc/README.md) | Target environment and the experiment runbook |
| 10 | [`docs/lit-review-plan.md`](docs/lit-review-plan.md) | Review plan, claims, terminology guardrail |

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
- [x] **EgoStitch implementation (frozen-s0 form)** — model, losses, data targets, auto-sized DDP worker, scoring, fidelity diagnostics, calibrated comparator, G5 runner, and tests were implemented under `src/`. The producing code last exists at `dcae090` and is deleted in the current cleanup worktree pending commit. The result note and `outputs/egostitch_stage1/` are retained; the registration snapshot that governed it is deleted with the rest of `docs/registrations/`.
- [x] **E2E headline redesign (rev 3.0)** — approved 2026-07-16 after two user-as-reviewer rounds + a vault/arXiv novelty sweep: stitched-topology-conditioned pair encoder (no frozen anchor). Design record: [`docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md`](docs/superpowers/specs/2026-07-16-egostitch-e2e-conditioned-encoder-design.md); implementation summary: spec §14; Phase 0 is unblocked by the completed frozen-s0 screen: [`docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md`](docs/superpowers/plans/2026-07-16-egostitch-e2e-conditioned-encoder.md).
- [x] **G5 frozen-s0 screening gate** — binding fixed-Seed-0 verdict `cut`: all three primary topology-dominance checks fail and both guards pass. The selected warm-start checkpoint is near S0; a diagnostic `last.pt` rerun proves later joint training moves ranking and passes matched GS, but still fails matched RD/clustering and regresses degree/spectral MMD. See the [result note](docs/results/G5-stage1-seed0-20260717.md).
- [x] **Locked-decision disposition (2026-07-17)** — frozen-s0 scalar fusion is retired to motivating evidence; rev 3.0 was made the active G5 build line, which has since advanced to rev-3.2. Successor landing condition §14.3(1) is satisfied. The frozen-s0 code is deleted in the current cleanup worktree, so after the cleanup commit the arm is citable from published artifacts and `dcae090`, not re-runnable from the active tree.
- [x] **G5 e2e screening gate** — binding fixed-Seed-0 v2 registration completed training (valid; liveness passed), held-out fp32 scoring, and the formal gate on 2026-07-24: BFS-macro GS passes but clustering-MMD and BFS-macro RD fail, the matched-AUPRC guard fails, and pathway attribution / the structure-destruction control both fail to establish a topology-conditioning gain. Verdict `cut` (multi-label). See the [result note](docs/results/G5-e2e-stage1-seed0-20260724.md).
- [x] **Single-stage plan-bound cleanup (current worktree; commit pending)** — qualification and every qualification-to-formal or model-quality authorization gate are removed. Formal execution is coupled only to the owner-bound experiment plan and exact artifact identities; model-quality signals are telemetry. `V_fit` and the single 512-node `V_hold := V_qual ∪ V_select` remain unchanged. Contract: spec §13.19 and §12.
- [x] **Rev-3.0 disposition** — the owner-side discussion advanced the line through rev-3.1 to the active rev-3.2 eight-arm contract; the 2026-07-24 `cut` remains historical engineering evidence.
- [x] **Component-ablation arm set** — six trained arms (`full`, `f_only`, `pair_topology`, `p0`, `no_l_rel`, `row_layernorm`), each owning one mechanism axis; `cosine_pool` is retired (Phase-0 pool-recall ceilings carry the width attribution) and `row_layernorm` ablates the D0 z-scoring fix. `python -m src.experiments.observe_e2e_formal <output_dir>` reports the three run-observation targets (generator clipping margins, slot collapse, end-ramp precision differential) from `profile.json` telemetry.
- [x] **Registration excision (2026-08-02)** — the preregistration/formal-run-registration/provenance-gating machinery (`docs/registrations/`, `hpc/qualification.sh`) is removed in full; the owner runs experiments directly against the trained-arm configs via `hpc/run.sh train`. Design: [`docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`](docs/superpowers/specs/2026-08-02-three-component-refactor-design.md) §10.
- [ ] **Experiments** in priority order: the rev-3.2 eight-arm formal screen, then E1/E3 multi-seed main + baselines → E4 ablations (incl. E4.15–E4.17 attribution/structure/conditioning-depth) → E5 integrity gates → E7 (load-bearing) → E6 breadth.

## HPC execution

The required target environment is a container with one or more NVIDIA H20 GPUs. Run
`hpc/run.sh check`, then use the same runner for cached scoring and G1/G2; run G3 directly
over the cached candidate universe. Formal E2
(B0 V3.1) training runs **only** through `hpc/run.sh train
configs/b0_v31_breadth_first.yaml`, which drives the production
`python -m src.e2_pipeline` entry (pack → train → publish, with the 30-epoch DDP train
launched by an auto-detected `accelerate launch --num_processes N` across all visible
GPUs at the config's pinned `runtime.token_budget`); direct
`python -m src.train_b0 --max-steps N` is debug-only. `B0-alt` no longer has a training
CLI: its implementation and config were removed from the tree on 2026-08-03 (see the
provenance note below). The host, repository/data paths, required software
versions, and the `nohup` form are pinned in [`hpc/README.md`](hpc/README.md). The
GPU count is recorded with each run, so throughput evidence is interpreted against
that exact hardware shape. There is no job scheduler (e.g. Slurm).

**EgoStitch e2e training runs through the same `hpc/run.sh train` branch as the
baselines**, over the six configured `configs/egostitch_e2e_v3_*.yaml` trained arms
(`full`, `f_only`, `pair_topology`, `p0`, `no_l_rel`, `row_layernorm`). It uses the
auto-detected multi-GPU orchestrator and `src.train_egostitch`, runs `pack → train →
publish`, trains on `V_fit`, validates on `V_hold`, and may not open a held-out path.

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal
```

There is no preregistration plan, artifact-identity gate, or clean-checkout precondition
— the owner has withdrawn that mechanism (design
[`docs/superpowers/specs/2026-08-02-three-component-refactor-design.md`](docs/superpowers/specs/2026-08-02-three-component-refactor-design.md)
§10). Non-finite state, DDP disagreement, coverage/data-boundary violations, and
I/O/infrastructure failures remain fail-closed inside the worker and pipeline.
Model-quality signals (eligibility, liveness, slot collapse, clip/family/RMS margins,
AUPRC, dispersion, precision quality) remain telemetry, recorded in `profile.json` and
`run_metadata.json`, and a quality miss cannot suppress a completed artifact. The two
scoring-time controls (`structure_control_6a_v3`, `structure_control_6e_v1`) reuse the
`full` arm's checkpoint and are not launched here. Historical hardware shapes, for
interpreting the published throughput evidence: the frozen-s0 Seed-0 run used
`world_size=2`; the rev-3.0 e2e screen used `world_size=4` (4 × H20).

## Literature

`literature/` is a hand-curated reference collection (gitignored, not a dependency). PDFs
follow `YEAR_venue_short_title.pdf` and are filed by topic; see
`literature/models/graph_structure_learning/README.md` for the taxonomy. Citations in the
model proposal are by arXiv ID and were verified against arXiv or local PDFs.
