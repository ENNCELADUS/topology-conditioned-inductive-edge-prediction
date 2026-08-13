# Auto-sized H20 container runner

This directory is the only HPC execution layer. It runs the implemented baseline, gate,
and EgoStitch CLIs directly inside the pinned container; there is no scheduler, job
array, or cluster-specific environment file. One tracked runner:

- `hpc/run.sh` — the single launcher: baselines, EgoStitch E2E training, held-out
  testing, cached scoring, and the G1/G2 gates all go through it.

Formal E2 training (`B0`, `configs/b0_v31_breadth_first.yaml`) runs **only** through
`hpc/run.sh train configs/b0_v31_breadth_first.yaml`, the runner's single `train` branch,
which always drives the production `python -m src.e2_pipeline` entry. That pipeline has
four sub-stages, `pack -> train -> publish -> test`: build or strictly validate the BF16
feature pack, launch one clean `accelerate launch` at the configured `runtime.token_budget`
whose process count is automatically set to all visible NVIDIA H20 GPUs, validate and
atomically publish the staging tree, then immediately run the held-out test protocol
against the published checkpoint (see "Held-out testing" below). Direct
`python -m src.train_b0 --max-steps N` remains debug-only (bounded smoke runs), skips the
test stage entirely so it never spends a held-out scoring epoch, and must never be used
for a reported E2 experiment.

`hpc/run.sh train` also drives EgoStitch E2E: the same branch always execs
`python -m src.e2_pipeline`, which defaults to the B0 worker (`src.train_b0`). Pass
`--worker-module src.train_egostitch --run-kind formal` after the config path to select
the EgoStitch worker instead (see "EgoStitch E2E" below).

The external CAZI-MBN reproduction is intentionally isolated from the E2
`pack -> train -> publish -> test` workers because its released teacher/student schedule is
not that pipeline. It still goes through the fixed H20 runner and its environment,
data, and GPU checks, and the same `train` branch chains the held-out test protocol
itself once training publishes a checkpoint:

```bash
hpc/run.sh train configs/cazi_mbn_breadth_first.yaml \
  --worker-module src.train_cazi_mbn
```

This direct branch preserves CAZI's early-stopped teacher-then-student schedule, writes
its checkpoints and topology report under the config's `output_dir`, then runs
`python -m src.eval.test_protocol` against the `student.pt` checkpoint it just published,
writing `test_report.json`/`test_complete.json` under the same `output_dir`.

## Required target environment

Formal execution requires one or more NVIDIA H20 GPUs. GPU count is intentionally
node-dependent and is recorded in runtime/run metadata; throughput claims remain tied
to the exact count in their retained artifacts.

| Item | Fixed value |
|---|---|
| SSH | `ssh -p 30838 root@10.15.171.204` |
| Repository | `/2023533015/topology-conditioned-inductive-edge-prediction` |
| GPU | 1 or more NVIDIA H20/H20-3e |
| NVIDIA driver | 550.144.03 |
| Python | 3.11.15 from the repository `.venv` |
| PyTorch / CUDA runtime | `2.10.0+cu128` / `12.8` |
| uv | `0.11.28` at `/2023533015/.uv/bin/uv` |
| Data | repository-local `data/` (26 GB; benchmark + frozen feature cache) |

Do not store the SSH password in this repository. The runner fails before executing any
command unless at least one visible GPU is named `NVIDIA H20` or `NVIDIA H20-3e`, the fixed paths exist, and
both benchmark and feature directories are present. The runner automatically exports
all detected GPU indices. `score` is a thin passthrough to `python -m src.score_fanout`,
which owns GPU-count detection, `--device cuda --amp bf16`, sharding, and the strict merge;
`merge`, `g1`, and `g2` are single-process. `test` is a thin passthrough to
`python -m src.eval.test_protocol`. `train` uses all visible GPUs via a matching
Accelerate world size. G3 is a direct single-process cached-score analysis command
outside `run.sh`.

## Run order

Connect, enter the fixed checkout, and verify the container before any experiment:

```bash
ssh -p 30838 root@10.15.171.204
cd /2023533015/topology-conditioned-inductive-edge-prediction
hpc/run.sh check
```

The check is fast environment/data validation only (uv, GPUs, benchmark, feature
shapes); test suites run locally, never as an HPC recheck.

Every arm's formal run is now one sequence, not train followed by separate manual
scoring commands: `hpc/run.sh train <config> ...` packs, trains, publishes, and then
scores test/candidate and writes the combined edge+graph `test_report.json` before
the command returns. Nothing further is required to get an arm's held-out numbers.

Only `egostitch_e2e` is ledgered against repeat scoring. Each e2e arm run performs two
scoring passes in one ledger epoch: fixed-0.5 test classification and density-controlled
candidate topology.
Re-running an already-scored `(arm, seed)` fails with "repeat scoring requires
--rescore-reason"; pass `--rescore-reason "<why>"` after the config path (and any
`--worker-module`/`--run-kind` flags) to `hpc/run.sh train` to intentionally re-open it.
There is no default reason and none is auto-generated — an operator must state one, for
example when a checkpoint is rescored after a scoring-code fix rather than a new training
run. `B0` and `CAZI-MBN` are not ledgered, since neither publishes an `egostitch_e2e`
checkpoint.

## EgoStitch E2E

EgoStitch E2E trains on `V_fit` and validates on the single 512-node `V_hold`. It may
not open a held-out path during training; that boundary is checked inside the worker, and
the command runs directly in the repository checkout.

The sole exception is an explicitly configured true-Oracle diagnostic:
`generator.oracle_truth_source: g_fit_plus_v_hold`. It consumes internal `V_hold`
positive structure and therefore must use `--run-kind diagnostic`; it writes
`diagnostic_complete.json`, never formal artifacts or `complete.json`, and its test stage
correspondingly writes `diagnostic_test_report.json`/`diagnostic_test_complete.json`,
never the formal `test_report.json`/`test_complete.json` names.

It runs through the same `hpc/run.sh train` branch as the baselines, naming the
EgoStitch worker and run kind explicitly and pointing at one of the trained-arm
configs (`full`, `f_only`, `pair_topology`, `p0`, `no_l_rel`, `row_layernorm`):

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal
```

For a true-Oracle diagnostic, use its dedicated config/output directory:

```bash
hpc/run.sh train configs/egostitch_e2e_v3_oracle_grit_film_logit_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind diagnostic
```

The two scoring-time controls (`structure_control_6a_v3`, `structure_control_6e_v1`)
are not trained arms and have no `train` invocation of their own — they reuse the `full`
arm's published checkpoint (`--checkpoint` only; nothing about the control changes what
was trained), so their held-out numbers come from calling `test` directly.

`--arm` and `--scaffold-control` are not the same name and must not be set to the same
value: `--arm` is the ledger/report identity (`structure_control_6a_v3` /
`structure_control_6e_v1`), while `--scaffold-control` is the perturbation *mode* that
`src.score_universe` actually knows how to parse (`shuffle_within_pair_v3` /
`rewire_checkerboard_v1`). Passing an arm name to `--scaffold-control` fails argument
parsing before any scoring happens, because that flag's choices are mode names only.

Each control also needs its own `--output-dir`, distinct from the `full` arm's own
directory and from each other's. `run_test_protocol` unconditionally (over)writes
`test_protocol_run_metadata.json`, `test_report.json`, and
`scores/{test,candidate}.npz` plus their per-universe `f0_cache_*`/
`grounding_cache_*` files into whatever `--output-dir` it is given. It never writes a
published `run_metadata.json` — it validates an existing one instead — but pointing a
control at `outputs/egostitch_e2e_v3_full` would clobber the trained `full` arm's
published evidence, and running the second control after the first would then clobber
the first control's evidence too. Only `--checkpoint` points back at `full`:

```bash
hpc/run.sh test \
  --checkpoint outputs/egostitch_e2e_v3_full/best.pt \
  --output-dir outputs/egostitch_e2e_v3_full/structure_control_6a_v3 \
  --data-root data --strategy breadth_first \
  --arm structure_control_6a_v3 --seed 0 \
  --timeout-seconds 1800 \
  --scaffold-control shuffle_within_pair_v3 \
  --rescore-reason "scoring-time structure control over the full checkpoint"

hpc/run.sh test \
  --checkpoint outputs/egostitch_e2e_v3_full/best.pt \
  --output-dir outputs/egostitch_e2e_v3_full/structure_control_6e_v1 \
  --data-root data --strategy breadth_first \
  --arm structure_control_6e_v1 --seed 0 \
  --timeout-seconds 1800 \
  --scaffold-control rewire_checkerboard_v1 \
  --rescore-reason "scoring-time structure control over the full checkpoint"
```

Scoring has no wall-clock deadline. A pass's cost is set by its pairs universe — the
candidate universe is 2,037,171 pairs, ~32x the test universe's 64,038, and takes hours
per shard — so any fixed budget is a guess that discards finished work when it is too
small. `--timeout-seconds` remains available on `score`/`test` to impose one deliberately.
A rerun after a later pass failed can pass `--reuse-existing-scores` to keep the
already-written `scores/<pairs>.npz` files instead of rescoring those universes; reused
artifacts are still cross-checked against the others for checkpoint identity.

Non-finite state, DDP disagreement, coverage/data-boundary violations, and
I/O/infrastructure failures remain fail-closed inside the worker and pipeline. Model-
quality predicates (eligibility, liveness, slot collapse, clip/family/RMS margins,
AUPRC, dispersion, precision quality) are telemetry, recorded in `profile.json` and
`run_metadata.json`, and do not block completion, publication, scoring, or evaluation.

## Baselines

Train the frozen `B0` baseline through the auto-sized H20 E2 production pipeline. The
shipped config pins BF16 and the repository-local data root:

```bash
hpc/run.sh train configs/b0_v31_breadth_first.yaml
```

`hpc/run.sh train configs/b0_v31_breadth_first.yaml` writes, under the pipeline's
output directory, `best.pt`, `last.pt`, `metrics.jsonl`, `run_metadata.json`,
`profile.json` (per-stage timings and the configured token budget), and
`artifact_manifest.json` (sha256 + byte size of the above). A successful atomic
publication writes `complete.json` last; its `total_seconds` is the authoritative
post-publication 60-minute acceptance time. The subsequent test stage then writes
`test_report.json` and `test_complete.json` (or the `diagnostic_*` pair for a diagnostic
run). It returns exit code `0` on success and `2` on a gated failure (for example a pack,
training stage exceeding its `runtime.*_budget_seconds` deadline; the test stage has no
deadline), naming the stage in `failure.json`; the published training artifacts survive a test-stage failure, and the
runner does not mask this exit code.

## Scoring and gates

`hpc/run.sh score` is a thin passthrough to `python -m src.score_fanout` for scoring one
pairs universe outside the automatic per-arm test protocol above — for example
regenerating a candidate-universe artifact for a G1/G2 gate rerun. It requires
`--timeout-seconds` (the hard wall-clock deadline for each shard and the merge), then
auto-detects GPU count, pins `--device cuda --amp bf16`, and on a multi-GPU node launches
one contiguous shard per GPU before strictly merging them; the shell no longer performs
any of that orchestration itself. For V3.1, pass `--pack-dir` to keep the BF16 token
table GPU-resident and avoid repeated per-pair feature-file reads.

```bash
hpc/run.sh score \
  --timeout-seconds 1800 \
  --checkpoint outputs/b0_v31/best.pt \
  --pairs candidate --data-root data --strategy breadth_first \
  --output scores/b0_v31_candidate.npz
```

Run the implemented gates over the cached scores:

```bash
hpc/run.sh g1 \
  --universe scores/b0_v31_candidate.npz \
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

`--max-steps` remains debug-only, skips the test stage, and must not be used for a
reported experiment.
