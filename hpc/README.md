# Auto-sized H20 container runner

This directory is the only HPC execution layer. It runs the implemented baseline, gate,
and EgoStitch CLIs directly inside the pinned container; there is no scheduler, job
array, or cluster-specific environment file. One tracked runner:

- `hpc/run.sh` — the single launcher: baselines, EgoStitch E2E training, cached
  scoring, and the G1/G2 gates all go through it.

Formal E2 training (`B0`, `configs/b0_v31_breadth_first.yaml`) runs **only** through
`hpc/run.sh train configs/b0_v31_breadth_first.yaml`, the runner's single `train` branch,
which always drives the production `python -m src.e2_pipeline` entry. That pipeline has
three sub-stages, `pack -> train -> publish`: build or strictly validate the BF16 feature
pack, launch one clean `accelerate launch` at the configured `runtime.token_budget` whose
process count is automatically set to all visible NVIDIA H20 GPUs, then validate and
atomically publish the staging tree. Direct `python -m src.train_b0 --max-steps N` remains
debug-only (bounded smoke runs) and must never be used for a reported E2 experiment.
`B0-alt` (`configs/b0_alt_breadth_first.yaml`) is outside this E2-only optimization: its
config is not the V3.1 shape `src.e2_pipeline` expects, so it is **not** trained through
`hpc/run.sh train` — it keeps its existing direct
`.venv/bin/python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml`
invocation, run directly (bypassing the runner's `train` branch) inside the same
container.

`hpc/run.sh train` also drives EgoStitch E2E: the same branch always execs
`python -m src.e2_pipeline`, which defaults to the B0 worker (`src.train_b0`). Pass
`--worker-module src.train_egostitch --run-kind formal` after the config path to select
the EgoStitch worker instead (see "EgoStitch E2E" below).

The external CAZI-MBN reproduction is intentionally isolated from the E2
`pack -> train -> publish` workers because its released teacher/student schedule is
not that pipeline. It still goes through the fixed H20 runner and its environment,
data, and GPU checks:

```bash
hpc/run.sh train configs/cazi_mbn_breadth_first.yaml \
  --worker-module src.train_cazi_mbn
```

This direct branch preserves CAZI's early-stopped teacher-then-student schedule and
writes its checkpoints, cached candidate scores, and pairwise/topology report under
the config's `output_dir`.

## Required target environment

Formal execution requires one or more NVIDIA H20 GPUs. GPU count is intentionally
node-dependent and is recorded in runtime/run metadata; throughput claims remain tied
to the exact count in their retained artifacts.

| Item | Fixed value |
|---|---|
| SSH | `ssh -p 30838 root@10.15.171.204` |
| Repository | `/2023533015/topology-conditioned-inductive-edge-prediction` |
| GPU | 1 or more NVIDIA H20, 97,871 MiB each |
| NVIDIA driver | 550.144.03 |
| Python | 3.11.15 from the repository `.venv` |
| PyTorch / CUDA runtime | `2.10.0+cu128` / `12.8` |
| uv | `0.11.28` at `/2023533015/.uv/bin/uv` |
| Data | repository-local `data/` (26 GB; benchmark + frozen feature cache) |

Do not store the SSH password in this repository. The runner fails before executing any
command unless at least one visible GPU is named `NVIDIA H20`, the fixed paths exist, and
both benchmark and feature directories are present. The runner automatically exports
all detected GPU indices. `score` creates one contiguous shard per visible GPU and
strictly merges them into the requested artifact; `merge`, `g1`, and `g2` are
single-process. `train` uses all visible GPUs via a matching Accelerate world size.
G3 is a direct single-process cached-score analysis command outside `run.sh`.

## Run order

Connect, enter the fixed checkout, and verify the container before any experiment:

```bash
ssh -p 30838 root@10.15.171.204
cd /2023533015/topology-conditioned-inductive-edge-prediction
hpc/run.sh check
```

The check runs the lightweight suite plus the three CPU DDP smoke contracts on Linux;
the four-H20 cold-run acceptance test remains an explicit opt-in.

## EgoStitch E2E

EgoStitch E2E trains on `V_fit` and validates on the single 512-node `V_hold`. It may
not open a held-out path; that boundary is checked inside the worker, and the command
runs directly in the repository checkout.

It runs through the same `hpc/run.sh train` branch as the baselines, naming the
EgoStitch worker and run kind explicitly and pointing at one of the trained-arm
configs (`full`, `f_only`, `pair_topology`, `p0`, `no_l_rel`, `row_layernorm`):

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal
```

The two scoring-time controls (`structure_control_6a_v3`, `structure_control_6e_v1`)
are not trained arms — they reuse the `full` arm's checkpoint at scoring time.

Non-finite state, DDP disagreement, coverage/data-boundary violations, and
I/O/infrastructure failures remain fail-closed inside the worker and pipeline. Model-
quality predicates (eligibility, liveness, slot collapse, clip/family/RMS margins,
AUPRC, dispersion, precision quality) are telemetry, recorded in `profile.json` and
`run_metadata.json`, and do not block completion, publication, scoring, or evaluation.

## Baselines

Train both frozen baselines. `B0` runs through the auto-sized H20 E2 production pipeline;
`B0-alt` is outside this optimization and is trained directly, without the runner's
`train` branch. The shipped configs pin BF16 and the repository-local data root:

```bash
hpc/run.sh train configs/b0_v31_breadth_first.yaml                       # B0: auto-sized H20 pipeline

.venv/bin/python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml  # B0-alt: direct train_b0
```

`hpc/run.sh train configs/b0_v31_breadth_first.yaml` writes, under the pipeline's
output directory, `best.pt`, `last.pt`, `metrics.jsonl`, `run_metadata.json`,
`profile.json` (per-stage timings and the configured token budget), and
`artifact_manifest.json` (sha256 + byte size of the above). A successful atomic
publication writes `complete.json` last; its `total_seconds` is the authoritative
post-publication 60-minute acceptance time. It returns exit code `0` on success and `2`
on a gated failure (for example a pack or training stage exceeding its
`runtime.*_budget_seconds` deadline), naming the stage in `failure.json`; the runner does
not mask this exit code.

## Scoring and gates

Score the candidate universe once per checkpoint. `run.sh score` defaults to
`--device cuda --amp bf16`; on a multi-GPU node it launches one contiguous shard per GPU
and publishes the final output only after strict `score_universe merge` validation. For
V3.1, pass `--pack-dir` to keep the BF16 token table GPU-resident and avoid repeated
per-pair feature-file reads.

```bash
hpc/run.sh score \
  --checkpoint outputs/b0_v31/best.pt \
  --pairs candidate --data-root data --strategy breadth_first \
  --output scores/b0_v31_candidate.npz

hpc/run.sh score \
  --checkpoint outputs/b0_alt/best.pt \
  --pairs candidate --data-root data --strategy breadth_first \
  --output scores/b0_alt_candidate.npz
```

Run the implemented gates over the cached scores:

```bash
hpc/run.sh g1 \
  --universe scores/b0_v31_candidate.npz \
  --alt-universe scores/b0_alt_candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g1

hpc/run.sh g2 \
  --universe scores/b0_v31_candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g2
```

Run the G3 Oracle gate directly over the cached candidate universe. It performs no model
scoring and writes the regime, assembled-graph, and headroom tables under `outputs/g3/`:

```bash
python -m src.experiments.g3_oracle \
  --universe outputs/deliverables/b0_v31_breadth_first_20260711/scores/candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g3
```

## Disconnect-safe runs

The commands above run in the foreground. For a disconnect-safe run, keep the same
runner and add only shell-level logging/backgrounding:

```bash
mkdir -p outputs/logs
nohup hpc/run.sh train configs/b0_v31_breadth_first.yaml \
  > outputs/logs/b0_v31_train.log 2>&1 &
```

`--max-steps` remains debug-only and must not be used for a reported experiment.
