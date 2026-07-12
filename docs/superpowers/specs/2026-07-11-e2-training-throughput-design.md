# E2 Four-H20 Training Throughput Design

**Date:** 2026-07-11  
**Status:** Approved design; implementation not started  
**Scope:** The implemented E2 B0 V3.1 training path only

## 1. Objective

Replace the production E2 training path with one four-GPU Hugging Face Accelerate
DDP pipeline that completes, from an empty derived-cache directory, within 60 minutes
on 4 x NVIDIA H20 GPUs.

The timed interval includes:

1. reading and validating the original per-node feature files;
2. building the first BF16 packed-feature cache;
3. initializing four DDP ranks and loading the packed features onto every GPU;
4. selecting the large-batch operating point with a bounded throughput probe;
5. training for exactly 30 epochs with validation after every epoch; and
6. writing the best/final checkpoints, history, metadata, profile, and checksums.

Universe scoring and the G1/G2 analyses remain downstream consumers of the existing
checkpoint contract. They are not included in the 60-minute training budget and must
not change as part of this work.

The first acceptance run is throughput-first. Validation AUROC and AUPRC are recorded
and compared with the baseline, but quality does not select or reject the fastest
batch configuration in this phase. Any quality regression is handled after the
throughput design is validated.

## 2. Motivation and constraints

The observed E2 run spends approximately 1.67 seconds per optimizer step, with a
21-22 minute training phase and a 5-6 minute validation phase per epoch. Sampled H20
SM utilization averages about 9.5%, with long zero-utilization intervals and brief
peaks. The current production configuration uses `num_workers: 0` and the training
dataset loads two separate per-node `.pt` tensors for every pair before padding on the
CPU and synchronously moving the resulting batch to the GPU.

Changing `num_workers` alone cannot meet the target. The design must eliminate the
per-pair small-file path, move token assembly onto the GPU, increase the effective
batch size, and distribute model computation across four GPUs.

This is a deliberate specification change. The current repository contract pins a
single H20 and prohibits DDP. The implementation sequence must therefore update
`docs/06-egostitch-spec.md`, with a change-log entry, before changing the code. The E2
task, model, split, metrics, and checkpoint interface remain unchanged.

## 3. Architecture

The production data flow is:

`per-node .pt files -> parallel BF16 feature pack -> GPU-resident feature table ->`
`distributed compact pair batches -> GPU gather/padding -> four-rank DDP ->`
`distributed validation -> rank-zero artifacts`

### 3.1 Cold-start packed-feature builder

Sixteen CPU processes read the source feature files in deterministic node order. Each
source tensor is loaded once, validated against the existing shape/dtype contract,
converted to BF16, and written directly to a worker-owned contiguous shard. Workers
return only compact manifest metadata; they do not transfer token tensors through
Python multiprocessing queues or `/dev/shm`.

The builder writes into a run-scoped temporary directory. After all shards and the
manifest pass validation, the directory is atomically renamed to its final cache
path. The manifest contains:

- hashes of source `metadata.json` and `index.json`;
- the exact node ordering;
- each node's shard, offset, and sequence length;
- feature dimension and stored dtype;
- per-shard byte size and checksum; and
- pack-worker count and stage timing.

An incomplete or stale pack is never read. A source-hash, node-coverage, shape,
length, dtype, size, or checksum mismatch is a hard error.

### 3.2 GPU-resident feature table

After the pack is complete, all four ranks map the same shards and copy the complete
BF16 token store to their local H20. The original cache is about 25 GB in FP32, so
the packed BF16 payload is expected to occupy about 12.5 GB per GPU. Each rank also
stores compact offset and length tensors indexed by integer node ID.

The model and optimizer are replicated normally by DDP. Replicating the feature table
is intentional: every rank can assemble an arbitrary pair locally without feature
communication between GPUs.

### 3.3 Multi-worker compact batch planning

Each DDP rank uses four persistent loader workers, for 16 workers total. These workers
operate only on integer endpoint IDs, labels, lengths, bucket boundaries, and row IDs.
They never load source feature files and never own CUDA tensors.

The fixed settings are:

- `num_workers = 4` per rank;
- `persistent_workers = true`;
- `prefetch_factor = 4`; and
- compact CPU tensors only across the worker boundary.

This keeps multiprocessing traffic small despite the container's limited shared
memory and prevents unsafe CUDA multiprocessing behavior.

### 3.4 GPU batch assembler

For each endpoint list, the rank gathers offsets and lengths from the local metadata
tensors. A vectorized position matrix is formed from `offset + arange(bucket_length)`.
Advanced indexing gathers all valid tokens from the flattened GPU feature buffer, and
the invalid positions are masked to zero. The assembler returns the existing
`emb_a`, `emb_b`, `len_a`, `len_b`, and `label` batch contract.

No production training step performs per-pair `torch.load`, CPU token padding, or a
feature-tensor host-to-device copy.

## 4. Distributed batching and training semantics

### 4.1 Deterministic global batch plan

The E2 legacy-reproduction arm continues to train on the fixed balanced training
rows. At cold start, pairs, labels, endpoint IDs, lengths, and row IDs are converted
once into compact arrays.

For every epoch, `(seed, epoch)` deterministically shuffles rows within the existing
length buckets. The sampler creates a global batch for one bucket at a time and then
splits that batch across four ranks. All ranks execute the same number of optimizer
steps, and the union of the four local streams covers every training row exactly once
per epoch.

No tail rows are dropped. When the last global batch cannot be split into equal local
batch sizes, each local mean loss is multiplied by:

`local_pair_count * world_size / global_pair_count`

After DDP gradient averaging, this is equivalent to a global sample mean.

### 4.2 Large-batch throughput probe

Before the formal run, the pipeline evaluates these per-rank token budgets:

- 262,144;
- 524,288;
- 1,048,576; and
- 1,572,864.

`max_pairs_per_rank` is fixed at 4,096 so a short-sequence bucket cannot create an
unbounded pair batch. Each candidate runs in a fresh four-rank process group with the
real model, optimizer, forward pass, backward pass, and optimizer step. The probe
contains warm-up and timed batches representative of the length-bucket distribution.

The chosen candidate maximizes global pairs per second subject to all of the
following:

- peak memory is at most 85 GiB on every H20;
- all ranks complete the same number of steps;
- no OOM or non-finite loss occurs;
- the GPU feature-cache hit rate is 100%; and
- increasing the token budget still improves measured throughput.

The probe is capped at four minutes and is the only place where batch fallback is
allowed. The selected token budget is frozen before the formal 30-epoch run.

### 4.3 Optimizer schedule

The formal run uses BF16, `lr = 1e-4`, no gradient accumulation, and the existing
optimizer/weight-decay settings. Large batches reduce the number of optimizer steps,
so the fixed `warmup_steps = 500` is replaced by a sample-based warm-up target. The
pipeline computes the number of pairs covered by the old loader's first 500 steps and
converts that target into the required number of new global-batch steps.

The first throughput run does not use linear learning-rate scaling. This prevents the
batch-system change and an optimizer change from being confounded in the initial
quality comparison.

Training always executes all 30 epochs. The current patience rule is evaluated and
reported counterfactually, but it does not stop this acceptance run early.

### 4.4 DDP configuration

The production launcher uses four processes and four GPUs through Hugging Face
Accelerate. The DDP configuration is:

- BF16 mixed precision;
- `find_unused_parameters = false`;
- `gradient_as_bucket_view = true`;
- `broadcast_buffers = false`; and
- no gradient accumulation.

The local `--max-steps` path remains available as a single-process smoke/debug mode.
It is not an acceptable source of a formal E2 result.

## 5. Distributed validation and artifacts

The fixed validation rows are partitioned by stable row ID so that every row appears
on exactly one rank. Each rank scores its rows from the same local GPU feature table
and returns only row IDs, labels, and logits.

Rank zero sorts the gathered records by row ID and verifies the expected row count,
uniqueness, and complete coverage before computing AUROC and AUPRC. A missing or
duplicate row is a hard error; metrics are never computed from a partial gather.

Validation runs after every epoch. Rank zero alone updates the best checkpoint and
writes artifacts. The final output contains:

- best and final checkpoints;
- per-epoch training loss and validation metrics;
- selected token budget and complete probe results;
- stage and per-epoch timing profiles;
- per-rank pair counts, batch counts, throughput, data-wait time, and peak memory;
- source and packed-feature manifest hashes; and
- the counterfactual early-stopping epoch.

The checkpoint format consumed by `score_universe` remains unchanged.

## 6. Failure handling and observability

The formal run does not silently change batch size, worker count, precision, or world
size. Any of the following terminates all ranks and writes a rank-zero `failure.json`:

- OOM after the probe has frozen the operating point;
- a non-finite loss;
- different optimizer-step counts across ranks;
- a feature-pack integrity failure;
- training or validation coverage mismatch;
- duplicate or missing validation rows; or
- violation of the configured stage deadline.

The structured profile records wall time for pack, GPU load, probe, training,
validation, and artifact writing. It also records global pairs/s, tokens/s, estimated
data-wait fraction, GPU compute time, cache hit rate, and per-rank peak memory.

If the probe extrapolates the full run above 60 minutes, the pipeline fails before
formal training and preserves the stage profile. It does not spend multiple hours to
confirm an already-predicted budget miss.

## 7. Time and resource acceptance budget

| Stage | Maximum wall time |
|---|---:|
| Parallel source read, validation, and BF16 pack | 5 minutes |
| DDP initialization, GPU cache load, and batch probe | 5 minutes |
| 30 training epochs with 30 distributed validations | 47 minutes |
| Checkpoints, metrics, profile, and checksums | 1 minute |
| Reserved margin | 2 minutes |
| **Total** | **60 minutes** |

The 4 x H20 acceptance criteria are:

- the timed run begins with an empty, new derived-cache directory;
- all 30 epochs and all 30 validations complete in at most 60 minutes;
- peak memory is at most 85 GiB per GPU;
- steady-state data-wait time is at most 5%;
- training and validation pair coverage exactly matches their inputs;
- the feature-cache hit rate during training and validation is 100%; and
- all required artifacts and manifests are complete and internally consistent.

Validation quality is reported against the existing baseline but is not a throughput
acceptance condition for this first phase.

## 8. Code boundaries

The implementation is divided into focused units:

- `src/data/packed_features.py`: pack construction, validation, manifests, GPU table,
  and vectorized gather/padding;
- `src/data/distributed_pairs.py`: deterministic global batch planning, rank
  partitioning, coverage accounting, and tail-batch metadata;
- `src/train_b0.py`: model/optimizer construction, sample-based warm-up, DDP
  train/eval loops, and checkpoint production;
- `src/e2_pipeline.py`: `pack -> probe -> train` orchestration, time-budget
  enforcement, profiles, and failure reports; and
- `hpc/run.sh`: the single production entry that invokes the E2 pipeline and its
  four-process Accelerate launches.

The existing `configs/b0_v31_breadth_first.yaml` gains one `runtime` section for the
world size, worker counts, probe candidates, memory limit, and time budget. No second
production E2 configuration is introduced.

## 9. Verification strategy

### 9.1 Unit tests

- BF16 packed features remain within the specified conversion error of the source.
- GPU gather/padding matches the legacy collate output after dtype conversion.
- Global batch plans are deterministic and have exact, duplicate-free coverage.
- Rank partitions have synchronized step counts.
- tail-batch loss scaling equals a single global sample mean.
- corrupt manifests, checksums, shapes, and source hashes are rejected.

### 9.2 Multi-process integration tests

A small synthetic feature store exercises pack construction, two- and four-rank DDP,
distributed validation gathering, and rank-zero checkpoint writing. Fault injection
covers OOM handling, checksum corruption, non-finite loss, and missing validation
rows. Tests verify that non-main ranks never write final artifacts.

### 9.3 Four-H20 acceptance test

The formal acceptance test runs from an empty derived-cache directory under the
budget and checks every criterion in Section 7. The profile, rather than an estimated
speedup, is the source of truth for whether the design meets the goal.

## 10. Required specification and documentation changes

The first implementation step must update the frozen contract before code changes:

1. `docs/06-egostitch-spec.md`: replace the single-H20/no-DDP production design with
   the four-H20 Accelerate DDP design and add a dated change-log entry.
2. `docs/03-experiment-protocol.md`: record the E2 execution budget and fixed
   30-epoch throughput acceptance without changing metrics or gates.
3. `hpc/README.md`, `README.md`, and `CLAUDE.md`: document the four-GPU environment,
   the single production entry point, and the status of single-process execution as
   debug-only.

The downstream score-once/analyze-many design and the experiment's integrity gates
remain binding.
