# Auto-sized H20 container runner

This directory is the only HPC execution layer. It runs the implemented baseline, gate,
and EgoStitch CLIs directly inside the pinned container; there is no scheduler, job
array, or cluster-specific environment file. Two tracked runners:

- `hpc/run.sh` — the baselines, cached scoring, and the G1/G2 gates.
- `hpc/qualification.sh` — the EgoStitch E2E two-stage ladder (`qualify` then `formal`),
  which owns the registration and preflight checks. `hpc/run.sh train` **refuses** a
  config whose `model.family` is `egostitch_e2e`: launching an arm from there would skip
  every preflight the ladder owns.

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

## Required target environment

Formal execution requires one or more NVIDIA H20 GPUs. GPU count is intentionally
node-dependent and is recorded in runtime/run metadata; throughput claims remain tied
to the exact count in their retained artifacts. The EgoStitch `formal` stage is the one
exception: it pins exactly four.

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

## The EgoStitch E2E ladder: qualify, then formal

Two stages, no sandbox and no threshold-freeze step between them. Both train on the full
`V_fit` and validate on the single 512-node `V_hold`, and they differ **only** in
`optim.epochs`; that identity is what lets them share one F0/grounding pack, one
grounding cache, and one `feature_stats_sha256`. Neither stage may open a held-out path —
that boundary is a path check inside the worker on both run kinds, not an isolated data
root, so both commands run directly in the repository checkout.

Neither command passes `--pack-dir`. The pack manifest is keyed on `n_ground`, so each
config names its own `runtime.pack_dir`; both stages of an arm share it, and arms that
agree on `n_ground` share it too (`cosine_pool` pins 20, the other five 50).

`qualify` is the development loop. It runs the trainer/conditioning/pipeline sanity
tests, then a 3-epoch run of the requested arm on every auto-detected visible H20. The
short schedule still traverses curriculum phases A -> B -> C, because the curriculum
scales with `schedule_total_steps` rather than with a fixed step count. Its verdict is
guards-only — `pass` iff no fail-fast guard tripped. Each invocation receives an
immutable directory under
`outputs/egostitch_e2e_stage1_v3/qualification/<arm>/attempts/attempt-*/` and writes
its `qualification.json` there together with the `feature_stats_sha256` and
`model_config_sha256` the formal stage compares against. The arm-local `latest`
symlink advances after every attempt, including a retained failure; `latest-pass`
advances only after a successful attempt. The formal preflight reads
`latest-pass/qualification.json`, never a mutable canonical report.

This durable history is part of the selection-exposure audit: every qualification
attempt evaluates the shared `V_hold`, so retaining all attempts makes the cumulative
evaluation count `K` recoverable per arm instead of hiding repeated development
selection. Qualification is frozen once v4 becomes `BINDING`; allowing further
attempts would escape the registered `K` disclosure. Each arm's
`attempt_history.json` uses schema `egostitch_e2e_qualification_history_v1`. Binding
must map exactly the six trained arms to `{path, sha256}` index references and register
the same exact, non-empty attempt lists under `qualification_attempts`; any omission,
extra arm, stale index, or list mismatch is a refusal.

Checkpoint eligibility is unaffected: it is enforced in both stages. `qualify` never
edits or promotes the active v4 registration, which remains `DRAFT`, and deliberately
does not require a clean checkout, because iterating on the model is the point:

```bash
hpc/qualification.sh qualify full   # or f_only|pair_topology|p0|cosine_pool|no_l_rel
```

`formal` produces results. It refuses to start unless there are exactly four visible
H20s, the checkout is clean, the registration is `BINDING` with every required
binding-evidence field resolved, and this arm's qualification verdict is `pass` with
both digests equal to the ones this formal config and its shared pack produce — a stale
report from before a model change is a refusal, not a pass. Six trained arms are
selectable; the two scoring-time controls (`structure_control_6a_v3`,
`structure_control_6e_v1`) reuse the full arm's checkpoint and are rejected here:

```bash
hpc/qualification.sh formal full   # or f_only|pair_topology|p0|cosine_pool|no_l_rel
```

The governing file is
`docs/registrations/g5_e2e_stage1_preregistration_v4.json`. It is currently `DRAFT`,
so `formal` is intentionally launch-blocked until the owner resolves its real binding
evidence and promotes a successor content state to `BINDING`.

Scientific execution order is `full` first. After training, the stage validates the
registered clip-coefficient, family-ratio, and submodule-RMS margins and persists that
verdict to `<output_dir>/margin_verdict.json`, bound to the digest of the `profile.json`
it was computed from. That gate necessarily runs *after* the pipeline has published the
run, so a completed run carries no margin evidence on its own: the remaining arms require
the full arm's persisted verdict alongside its eligibility and liveness preflight, and a
verdict left behind by an earlier full run cannot stand in for the current one. For the
full arm the stage then produces the registered `formal_train` probe artifact that the G5
gate evaluator consumes, at the path bound in the registration.

Neither stage substitutes `--max-steps` for its schedule, and neither promotes the
registration; both stop immediately on failure.

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
