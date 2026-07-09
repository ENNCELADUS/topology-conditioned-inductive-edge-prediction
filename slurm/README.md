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

# 3. Train. The shipped configs pin `data.root: data` (relative), and the train
# CLI has no data-root flag — so first make the repo-local `data/` resolve to
# the cluster data root (once, in $CLUSTER_REPO_DIR):
ln -s "$CLUSTER_DATA_ROOT" data     # or edit data.root in the config
slurm/submit.sh train configs/b0_v31_breadth_first.yaml
slurm/submit.sh train configs/b0_alt_breadth_first.yaml

# 4. Score (optionally sharded via a Slurm array). --pairs takes a SOURCE NAME
# (candidate | test | val | file:<path.tsv>), never a raw benchmark path —
# analysis pipelines reject artifacts whose pairs_source is not "candidate".
slurm/submit.sh score -- score --checkpoint outputs/b0_v31/best.pt \
  --pairs candidate \
  --data-root "$CLUSTER_DATA_ROOT" --strategy breadth_first --output scores/b0.npz

# Sharded variant: pass the SAME --output to every array task. The CLI itself
# derives each shard's filename from --output plus its --shard index (e.g.
# --output scores/b0_v31_candidate.npz with --shard 2 writes
# scores/b0_v31_candidate.shard-2.npz) — do not hand-shard the --output path.
# Sharding is for the v3_1 checkpoint; run the cheap b0_alt scoring unsharded.
slurm/submit.sh score --array 0-3 -- score --checkpoint outputs/b0_v31/best.pt \
  --pairs candidate \
  --data-root "$CLUSTER_DATA_ROOT" --strategy breadth_first \
  --output scores/b0_v31_candidate.npz
# Wait for ALL array tasks to finish (squeue) before merging — merge validates
# gap-free coverage and will fail loudly on a missing shard.
slurm/submit.sh score -- merge --inputs scores/b0_v31_candidate.shard-*.npz \
  --output scores/b0_v31_candidate.npz

# 5. Back on your local machine: pull results down
slurm/fetch_outputs.sh             # fetches outputs/ and scores/ by default
```

Logs land in `slurm/logs/%x-%j.out` (created by `submit.sh`, git-ignored).
