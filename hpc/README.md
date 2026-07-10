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

`--max-steps` remains debug-only and must not be used for a reported experiment.
