---
name: hpc-execution
description: Use before running or modifying training, scoring, merge, or gate commands for this repository, locally or on the H20 container. Covers the fixed SSH checkout, hpc/run.sh, automatic world-size and score-shard sizing, direct EgoStitch execution, cached-score analysis, and model-family-specific runtime keys.
---

# HPC execution

Treat `hpc/README.md` and `hpc/run.sh` as the live execution sources. The repository
has one HPC launcher, `hpc/run.sh`, and no scheduler or qualification ladder. Run
experiments directly; do not add preregistration, contract-identity, hash, eligibility,
promotion, or qualification preflights.

## Connect to the H20 container

```bash
ssh -p 30838 root@10.15.171.204
cd /2023533015/topology-conditioned-inductive-edge-prediction
```

The pinned checkout uses:

- repository: `/2023533015/topology-conditioned-inductive-edge-prediction`
- Python: `/2023533015/topology-conditioned-inductive-edge-prediction/.venv/bin/python`
- uv: `/2023533015/.uv/bin/uv`
- data: repository-local `data/`
- hardware: one or more visible `NVIDIA H20` GPUs

Do not store the SSH password in the repository.

## Use `hpc/run.sh`

Available subcommands are `check`, `train`, `score`, `merge`, `g1`, and `g2`.
The runner validates the fixed paths, data directories, Python environment, and every
visible GPU before dispatching any subcommand.

```bash
hpc/run.sh check
```

`check` runs environment/data checks, the full non-integration test suite, and the Linux
DDP integration tests. Do not describe it as a lightweight smoke test.

## Let the runner size the job

Keep `runtime.world_size: auto` in EgoStitch configs. `hpc/run.sh` counts all visible
H20s, exports contiguous GPU IDs, and the pipeline resolves `auto` to that count.
Do not hard-code a GPU count or launch a separate single-GPU substitute.

For scoring, do not pass `--shard` or `--num-shards`. The runner creates one contiguous
shard per visible GPU, waits for all shards, and merges them. `merge`, `g1`, and `g2`
remain single-process.

## Train directly

All production training through the runner uses `src.e2_pipeline` and its
`pack -> train -> publish` flow.

Train B0 with its config:

```bash
hpc/run.sh train configs/b0_v31_breadth_first.yaml
```

Train EgoStitch through the same branch by naming its worker and run kind:

```bash
hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal
```

Choose the current arm config under `configs/`; do not route EgoStitch through a
`qualification.sh` script. `--max-steps` is debug-only and must not be used for a
reported experiment.

For disconnect-safe execution, add only shell-level logging/backgrounding around the
same command:

```bash
mkdir -p outputs/logs
nohup hpc/run.sh train configs/egostitch_e2e_v3_full_breadth_first.yaml \
  --worker-module src.train_egostitch --run-kind formal \
  > outputs/logs/egostitch_full.log 2>&1 &
```

Treat a live process as execution evidence only. Confirm completion from the process
exit, logs, GPU state, required artifacts, and `complete.json`; treat `failure.json` as
terminal failure.

## Score once, analyze many times

Score a pair universe once per checkpoint into `.npz`, then run analyses over that
cached artifact. Gate modules must not load a checkpoint or rescore pairs.

```bash
hpc/run.sh score \
  --checkpoint outputs/example/best.pt \
  --pairs candidate --data-root data --strategy breadth_first \
  --output scores/example_candidate.npz

hpc/run.sh g1 --universe scores/example_candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g1

hpc/run.sh g2 --universe scores/example_candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g2
```

Run G3 directly over cached scores because it is outside `run.sh`:

```bash
.venv/bin/python -m src.experiments.g3_oracle \
  --universe scores/example_candidate.npz \
  --data-root data --strategy breadth_first --output-dir outputs/g3
```

## Config keys with family-specific meaning

- Keep EgoStitch `runtime.world_size` as the literal string `auto`.
- Read `runtime.token_budget` by family: B0 uses a global token budget; EgoStitch uses
  the per-rank node-stream batch size `B_n`.
- Keep `data.pack_dir` and `runtime.pack_dir` distinct. The former is the raw-token
  pack; the latter is the pipeline F0/grounding pack. CLI `--pack-dir` overrides only
  `runtime.pack_dir`.
- Expect unknown config keys to fail closed. Before changing config semantics, inspect
  the current loader in `src/train_b0.py` or `src/train_egostitch.py`.
