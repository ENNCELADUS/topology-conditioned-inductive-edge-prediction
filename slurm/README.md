# Slurm submission layer

Everything here flows through a user-filled `cluster.env` (git-ignored) since
the cluster hostname/partition/account are unknown at build time. `.sbatch`
files carry only generic directives; account/partition/gres/time are passed
by `submit.sh` as `sbatch` CLI flags, since `#SBATCH` cannot expand env vars.

## One-time setup

```bash
cp slurm/cluster.env.example slurm/cluster.env
# edit slurm/cluster.env with your cluster's host/account/partition/paths
```

The data package (including the 25 GB feature cache) must already exist on
cluster storage at `$CLUSTER_DATA_ROOT` — only code is synced by this layer.

## Command sequence

```bash
# 1. From your local checkout: push code to the cluster (run locally)
slurm/sync_code.sh                 # add --with-env to also scp cluster.env

# 2. On the cluster login node, from $CLUSTER_REPO_DIR: sanity-check the node
slurm/submit.sh preflight

# 3. Train
slurm/submit.sh train configs/b0.yaml

# 4. Score (optionally sharded via a Slurm array)
slurm/submit.sh score -- score --checkpoint outputs/b0/best.pt \
  --pairs data/benchmark_2025_neurips/breadth_first/candidate_test_edges.txt \
  --data-root "$CLUSTER_DATA_ROOT" --strategy breadth_first --output scores/b0.npz

# Sharded variant: pass the SAME --output to every array task. The CLI itself
# derives each shard's filename from --output plus its --shard index (e.g.
# --output scores/b0_v31_candidate.npz with --shard 2 writes
# scores/b0_v31_candidate.shard-2.npz) — do not hand-shard the --output path.
slurm/submit.sh score --array 0-3 -- score --checkpoint outputs/b0/best.pt \
  --pairs data/benchmark_2025_neurips/breadth_first/candidate_test_edges.txt \
  --data-root "$CLUSTER_DATA_ROOT" --strategy breadth_first \
  --output scores/b0_v31_candidate.npz
slurm/submit.sh score -- merge --inputs scores/b0_v31_candidate.shard-*.npz \
  --output scores/b0_v31_candidate.npz

# 5. Back on your local machine: pull results down
slurm/fetch_outputs.sh             # fetches outputs/ and scores/ by default
```

Logs land in `slurm/logs/%x-%j.out` (created by `submit.sh`, git-ignored).
