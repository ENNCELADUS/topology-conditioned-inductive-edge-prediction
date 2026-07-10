# Single-H20 container runner

This directory is the only HPC execution layer. It runs the implemented baseline and
gate CLIs directly inside the pinned container; there is no Slurm scheduler, job array,
or cluster-specific environment file.

## Pinned environment (verified 2026-07-10)

| Item | Fixed value |
|---|---|
| SSH | `ssh -p 30838 root@10.15.171.204` |
| Repository | `/2023533015/topology-conditioned-inductive-edge-prediction` |
| GPU | 1 × NVIDIA H20, 97,871 MiB |
| NVIDIA driver | 550.144.03 |
| Python | 3.11.15 from the repository `.venv` |
| PyTorch / CUDA runtime | `2.10.0+cu128` / `12.8` |
| uv | `0.11.28` at `/2023533015/.uv/bin/uv` |
| Data | repository-local `data/` (26 GB; benchmark + frozen feature cache) |

Do not store the SSH password in this repository. The runner fails before executing an
experiment unless exactly one visible GPU is named `NVIDIA H20`, the fixed paths exist,
and both benchmark and feature directories are present.

## Run order

Connect, enter the fixed checkout, and verify the container before any experiment:

```bash
ssh -p 30838 root@10.15.171.204
cd /2023533015/topology-conditioned-inductive-edge-prediction
hpc/run.sh check
```

Train both frozen baselines. The shipped configs pin BF16 and the repository-local data
root:

```bash
hpc/run.sh train configs/b0_v31_breadth_first.yaml
hpc/run.sh train configs/b0_alt_breadth_first.yaml
```

### V3.1/F1 startup and host memory

The production F1 launch is
`hpc/run.sh train configs/b0_v31_breadth_first.yaml`. Before its DataLoader workers
start, the process performs a one-time ~25 GiB preload of the operative raw-token
tensors into CPU host memory. Before launching, run
`free -h` and verify that the `available` column is at least 25 GiB, with additional
headroom for the training process, workers, and pinned batches.

The production config uses the frozen F1 loader contract:

- `num_workers: 4`
- `persistent_workers: true`
- `prefetch_factor: 4`
- `pin_memory: true`

Training creates one DataLoader and one persistent four-worker pool. At each exhausted
epoch boundary, the parent process replaces only the shared endpoint-index/label rows
and the length-bucket sampler state before reusing that same pool for the next epoch.
Worker IPC is descriptor-only: workers prefetch integer row ids, so padded token tensors
never pass through the container's `/dev/shm`. The main process uses the preloaded cache
to materialize each padded batch directly in pinned host memory before non-blocking H2D.

Preload progress is logged every 1,000 new tensors, followed by a final
`preloaded ... operative node tensors (... GiB) into host memory` message. During
this startup pause, no step logs are expected. Once preload completes, step logs begin
every 50 optimizer steps; the pause does not by itself indicate a stalled run.

Score the candidate universe once per checkpoint. `run.sh score` always injects
`--device cuda --amp bf16`; the single-H20 reference run is intentionally unsharded.

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

The commands above run in the foreground. For a disconnect-safe run, keep the same
runner and add only shell-level logging/backgrounding:

```bash
mkdir -p outputs/logs
nohup hpc/run.sh train configs/b0_v31_breadth_first.yaml \
  > outputs/logs/b0_v31_train.log 2>&1 &
```

After launch, verify all of the following before leaving the run unattended:

- `tail -f outputs/logs/b0_v31_train.log` shows preload progress and the final
  `preloaded ...` summary; do not expect `epoch ... step ...` lines before that summary.
- After preload, `epoch ... step ...` values advance rather than remaining fixed.
- `nvidia-smi` shows the training process and nonzero GPU memory use.
- The log contains neither `NaN` nor `Traceback`.

`--max-steps` remains debug-only and must not be used for a reported experiment.
