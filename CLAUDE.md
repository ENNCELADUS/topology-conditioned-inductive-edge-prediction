# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Codex when working with code in this repository.

## What this repository is

A **research-paper project** targeting ICLR 2027: *Topology-Conditioned Inductive
Edge Prediction*. It contains design documents, result notes, self-contained HTML
figures, benchmark data artifacts, and a curated literature library. `src/` holds a
typed, tested Python package covering the benchmark/evaluation stack, B0/B0-alt,
G1–G3, and the pre-registered **EgoStitch Stage-1 implementation**. The direct HPC
layer auto-detects all visible H20s for formal E2 and EgoStitch DDP execution.
Gates G1–G3 are complete; G5 Stage 1 is not. Its formal run produced one valid Seed-0
training artifact on 2026-07-15, then Seed 1 stopped at the artifact performance gate
and Seed 2 did not run. Do not report a G5 verdict until all three seeds, held-out
scoring, fidelity diagnostics, and pre-registered gate evaluation are complete.

`README.md` is the human-facing entry point (orientation, status, structure map,
reading order). This file (`CLAUDE.md`) holds the binding *constraints* an agent
must respect. Code lives under `src/`; `docs/05-egostitch-spec.md` (algorithm + data
contract + auto-sized H20 execution design) and `docs/03-experiment-protocol.md` (what to run, how to
grade) are the contracts it must implement.

## Repository layout

```
README.md                        entry point / navigation hub
CLAUDE.md                        this file — agent guardrails
pyproject.toml                   package/deps (uv) + ruff, mypy, pytest config
uv.lock                          pinned dependency lockfile
docs/
  01-blueprint.md                top-level paper blueprint (locked decisions)
  02-methodology.md              abstract method contract + training objective
  03-experiment-protocol.md      run/eval contract: baselines, E1–E7, metrics, gates G1–G5
  04-model-proposal.md           EgoStitch model rationale (APPROVED 2026-07-09)
  05-egostitch-spec.md           implementation contract (G4 signed off): algorithm,
                                 benchmark/data contract, batch sampler, auto-sized H20 design
  lit-review-plan.md             review plan, claims K1–K5, terminology guardrail
  results/E2-pair-to-topology-gap.md   motivating result note
  results/G5-stage1-seed0-20260715.md  completed Seed-0 training result; gate incomplete
figures/                         e2-gap.html, positioning.html (standalone, open in browser)
data/
  README.md                      benchmark artifact manifest (layout + usage contract)
  benchmark_2025_neurips/        graph, splits, edge lists, eval buckets (3 strategies)
  features/                      frozen per-node token-sequence features (~25 GB, gitignored)
src/
  data/        artifacts.py (benchmark load+verify), partition.py (message/supervision
               split, spec §9.3), pairs.py (datasets, bucketed batching, neg sampler),
               features.py (frozen feature store + F0 mean-pool matrix)
  eval/        edge_metrics.py, graph_metrics.py (official BFS-macro GS/RD + assembled-graph MMD),
               assembly.py (assemble + threshold sweep), composite.py (perturbation diagnostic)
  model/       B0.py, b0_alt.py, egostitch/ (Stage-1 model/loss/decision modules)
  train_b0.py        CLI: train a frozen B0-family scorer
  train_egostitch.py auto-sized Accelerate DDP worker for formal EgoStitch training
  score_universe.py  CLI: score pair lists -> pinned .npz scores artifact (shardable)
  experiments/ G1/G2/G3 analyses, b0_cal.py, g5_stage1.py
configs/        B0/B0-alt and egostitch_stage1_breadth_first.yaml
hpc/            auto-sized H20 container runner (run.sh + environment/runbook README)
outputs/        run artifacts: checkpoints, metrics, cached score matrices (gitignored)
tests/          pytest suite mirroring src/ (data/, eval/, experiments/, CLIs, hpc)
literature/     curated reference library (gitignored, unchanged)
```

## Commands

Dependencies and tools are managed with **uv** (Python 3.11, pinned in `uv.lock`).

```bash
uv sync --group dev              # install runtime + dev deps into .venv

uv run pytest                    # full test suite
uv run pytest tests/eval/test_assembly.py             # one test file
uv run pytest tests/test_g1_hardened_e2.py -k regime  # tests matching a name
uv run ruff check . && uv run ruff format --check .   # lint + format check
uv run mypy src tests            # strict type check
```

Research CLIs are `python -m` modules — run under `uv run` (or
`.venv/bin/python`, which avoids the rtk proxy garbling `uv run` output locally):

```bash
# 1. Train a frozen B0-family scorer (--max-steps N is DEBUG-ONLY: bounded smoke run)
python -m src.train_b0 --config configs/b0_v31_breadth_first.yaml

# 2. Score pair lists ONCE into the pinned scores artifact the gates consume
python -m src.score_universe score --checkpoint outputs/b0_v31/best.pt \
    --pairs candidate --data-root data --strategy breadth_first \
    --output scores/b0_v31_candidate.npz          # `merge` subcommand joins shards

# 3. Gate analyses read the cached scores artifact (no model scoring here)
python -m src.experiments.g1_hardened_e2 --universe scores/b0_v31_candidate.npz \
    --data-root data --strategy breadth_first --output-dir outputs/g1
python -m src.experiments.g2_ceiling    --universe scores/b0_v31_candidate.npz \
    --data-root data --strategy breadth_first --output-dir outputs/g2
```

HPC runs target containers with one or more NVIDIA H20 GPUs; `hpc/README.md` pins
the SSH endpoint, repository/data paths, Python/PyTorch/CUDA versions, and the exact
`check → train → score → G1/G2` sequence; G3 is a direct cached-score analysis command.
Formal E2 (B0 V3.1) training runs **only**
through `hpc/run.sh train configs/b0_v31_breadth_first.yaml`, which drives
`python -m src.e2_pipeline` (pack → probe → projection → 30-epoch DDP train via
an auto-detected `accelerate launch` across all visible GPUs); direct
`python -m src.train_b0 --max-steps N` remains debug-only. `B0-alt` keeps its
existing direct `python -m src.train_b0 --config ...` training CLI, outside this
E2-only optimization. EgoStitch Stage-1 uses the same orchestrator with the EgoStitch
worker and an auto-detected Accelerate world size. `merge`/`g1`/`g2` remain
single-process; G3 is also single-process and CPU/memory-bound; there is
no job scheduler (e.g. Slurm). Every run records its detected world size; throughput
claims are hardware-shape-specific.

**Tooling gotcha:** mypy is strict with `warn_unused_ignores = true`. Don't run two
`mypy` invocations against the same `.mypy_cache` concurrently — it corrupts the cache
and surfaces phantom `unused-ignore` errors; re-run cold before believing them.

## Code architecture (the implemented pipeline)

The code implements the baseline/gate stack and EgoStitch Stage 1. The evaluation flow
remains deliberately **score-once, analyze-many**:

1. **Data** (`src/data/`) loads and *verifies* the frozen `benchmark_2025_neurips`
   package (`artifacts.py`), enforces the node-disjoint split and message/supervision
   partition (`partition.py`, spec §9.3), builds token-pair datasets with a negative
   sampler (`pairs.py`), and serves frozen per-node features (`features.py`). The 25 GB
   feature cache is gitignored and must already exist locally / in the fixed container checkout.
2. **Train** (`src/train_b0.py`) fits a frozen B0-family scorer (`model/B0.py` V3.1, or
   `model/b0_alt.py` MLP) and writes a Task-4-format checkpoint under `outputs/`. Formal
   E2 (B0 V3.1) training is driven by `src/e2_pipeline.py` (`hpc/run.sh train`), which
   wraps `train_b0.py` in a pack → probe → projection → 30-epoch `accelerate launch
   --num_processes N` DDP run across all visible H20s; B0-alt keeps its own direct
   `train_b0.py` invocation.
3. **EgoStitch Stage 1** (`src/model/egostitch/`, `src/train_egostitch.py`) implements
   Tokenize-lite, imagination, matching, stitching, the `(s0,s1,s2)` decision head,
   the frozen-S0 contract, pre-registered losses, auto-sized DDP training, and formal
   artifact publication. `src/experiments/g5_stage1.py` performs the Stage-1 gate.
4. **Score** (`src/score_universe.py`) runs a checkpoint over the candidate universe /
   val / test pairs *once* and writes a single self-contained `.npz` scores artifact
   (the CLI is shardable and strict merge restores manifest order; the HPC runner uses
   one contiguous shard per visible GPU).
5. **Gate analyses** (`src/experiments/`) are pure row-selections + graph math over that
   cached artifact — **no model scoring happens here**. `g1_hardened_e2.py` builds the
   hardened-E2 gate table (negative regimes, threshold/density-matched operating point,
   assembled-graph rows); `g2_ceiling.py` computes the edge-independence triangle ceiling;
   `g3_oracle.py` evaluates the pinned Oracle arms and headroom stop rule.
6. **Eval** (`src/eval/`) is the shared metric library used by all of the above:
   `edge_metrics.py` (AUROC/AUPRC…), `graph_metrics.py` (official BFS-macro GS/RD plus
   degree/clustering/spectral MMD on the assembled graph), `assembly.py` (assemble +
   threshold sweep), `composite.py` (MMD perturbation diagnostic). Edge- and
   graph-level metrics are always reported together (see Integrity gates).

The spec's freeze rule is binding: `docs/05-egostitch-spec.md` is the contract, and code
must not silently deviate — edit the spec first (with a change-log line), then the code.

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
4. `docs/04-model-proposal.md` — **EgoStitch**, the concrete model (dual ego-net
   imagination + community codebook, replacing the retrieved-and-thresholded
   scaffold). **Status: approved 2026-07-09** (design rationale). It adds baseline B5
   and reframes the original §0 scaffold as an ablation arm.
5. `docs/05-egostitch-spec.md` — the **implementation contract** (gate G4 signed off
   2026-07-09): pinned algorithm/shapes/losses, §9 benchmark binding and data
   contract (including quarantined artifacts and the self-loop policy), §10 batch
   sampler, §11 auto-sized H20 execution design. Its freeze rule is binding: code may not
   silently deviate — edit the spec first, with a change-log line.
6. `docs/results/E2-pair-to-topology-gap.md` — the motivating result note with completed
   G1 B0/PA-null/B0-alt, G2 ceiling, and G3 Oracle artifacts.
7. `docs/results/G5-stage1-seed0-20260715.md` — completed Seed-0 training evidence and
   the explicit boundary that the three-seed G5 gate remains incomplete.
8. `docs/lit-review-plan.md` — literature-review plan, claims K1–K5, the 2×2
   taxonomy, and the terminology guardrail.

When these conflict, the more specific/later document refines the earlier one, but
locked decisions and the locked §0 contract override casual changes — flag conflicts
rather than resolving them unilaterally.

## Load-bearing facts (the E2 result)

**Canonical metric rerun note (2026-07-14):** official BFS-macro GS/RD were formally rerun on
the frozen candidate-score artifacts over all 500 fixed induced subgraphs and macro-averaged
over all samples. Final G1, G3, and legacy artifacts live under
`outputs/deliverables/{g1_graph_metrics_20260714,g3_graph_metrics_20260714,legacy_g1_graph_metrics_20260714}`.
The MMD component ratios remain the recorded canonical-run values.

The current load-bearing gap is the completed G1 result: the frozen B0 pairwise scorer
reaches **AUROC 0.705519 / AUPRC 0.730260** on degree-corrected ratio-1 negatives,
falls to **0.583965 / 0.626649** on hard heuristic negatives and **0.569560 / 0.617475**
on hard feature negatives, yet its quota-calibrated assembly has global simple-edge RD
**0.997710** but official BFS-macro **GS/RD 0.312151/0.422345**, alongside
degree/clustering/spectral MMD ratios **13.0768/11.9273/18.0931**. B0-alt reaches
**0.693603 / 0.732509** on the same degree-corrected row and has global simple-edge RD
**0.998739**, BFS-macro GS/RD **0.345802/0.450793**, and MMD ratios
**15.8304/13.4718/23.4734**, closing the required
architecture-independence arm. A stronger legacy scorer reaches **0.799577 / 0.813319**
yet retains MMD ratios **13.8456/11.6277/19.9774** at global simple-edge RD **0.978392**
and BFS-macro GS/RD **0.381264/0.500179**, so higher edge quality does not close
the topology gap.
G2 reports `Ov(P)=0.479886` versus `Ov_min=0.054954`. G3 reports Oracle-blend
MMD-ratio headroom `1.72315/3.88537/1.99767` for degree/clustering/spectral; Oracle-topo
raises graph similarity to `0.503048` (`1.61155×` B0), so the feature-insufficiency stop
rule is not triggered.
`docs/04-model-proposal.md` §4.6 maps these failure axes to specific EgoStitch
mechanisms. Keep the current values consistent across `docs/results/E2-pair-to-topology-gap.md`,
`docs/03-experiment-protocol.md`, `docs/04-model-proposal.md`, `README.md`, and
`figures/e2-gap.html`.

## Current G5 Stage-1 evidence boundary

The only completed formal EgoStitch training result is Seed 0 at commit `31afbe6`:
2 × H20, `world_size=2`, 30 epochs, best epoch 14, validation AUROC/AUPRC
`0.945766/0.951966`, and total time `807.389 s`. Its training curve is nearly flat
(AUPRC range `0.951582–0.951966`), so it does not by itself establish a learned gain
over frozen S0. Seed 1 failed only the artifact performance check
(`steady_state_data_wait_fraction > 0.05`); Seed 2 did not run. These are training
artifacts, not held-out candidate-universe or assembled-graph results. Never call G5
passed or failed from Seed 0 alone.

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
