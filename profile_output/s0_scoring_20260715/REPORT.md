# S0 scoring system profile

## Part A — Results and ranked recommendations

### Outcome

The full 3,364,731-pair S0 universe completed successfully on two NVIDIA H20s in
1,853 seconds (30 minutes 53 seconds). The observed aggregate throughput was
1,815.83 rows/s. The legacy partial run projected to 18,188 seconds (about 5.05
hours), so the production path is 9.815x faster and cuts projected wall time by
89.81%.

The merged artifact contains exactly 3,364,731 rows over 8,070 node IDs, all
logits are finite, and its checkpoint ID is `e092537d8cf1e208`. Its SHA-256 is
`7bfa8c281508c4f3c7a552f7e85c0479120342c4969e37f21b934f1302b6af4e`.

### Bottlenecks, ranked

1. **Legacy unpacked feature I/O and repeated endpoint work.** The baseline
   spent about 236 seconds probing 8,070 individual feature files, then loaded
   endpoint feature files repeatedly while scoring pairs. It reached only about
   185 rows/s including startup.
2. **The production command bypassed multi-GPU sharding.** The monitored second
   GPU remained idle throughout the baseline; post-probe mean utilization on
   the active GPU was only 8.85%.
3. **The Siamese encoder was recomputed per pair.** S0 has only 8,070 unique
   nodes across 3.36 million pairs, so caching each node encoding once removes
   redundant encoder calls. The cache must remain FP32 because the V3.1 encoder
   output is FP32 under BF16 autocast.
4. **Remaining cost is long-sequence pair-head computation.** In the completed
   run both H20s stayed near 100% utilization. Later, longer batches used up to
   roughly 87 GB per GPU and determined the final shard runtimes.

### Implemented optimizations

- `hpc/run.sh s0-score` now auto-detects visible GPUs, launches one strict shard
  per GPU, uses the packed BF16 feature table, and strictly merges both shards.
- Packed scoring gathers every used node once, computes its encoder output once,
  preserves that cache in FP32, and reuses it in the pair head.
- The S0 token budget is 1,048,576. A fourfold token-budget probe reduced batch
  count from 983 to 248 but improved the old packed scorer by only 1.2%, showing
  that batch-launch overhead was not the main bottleneck.
- Stage-1 runtime sizing now follows the detected world size instead of the
  legacy single-process or hardcoded-GPU path.

### Correctness and production validation

- Cached FP32 packed scoring and the previous packed implementation produced
  identical pairs and logits on 52,574 rows: maximum and mean absolute logit
  differences were both `0.0`.
- The remote focused suite passed 41 tests before the formal run.
- The full runner launched shard `0/2` on GPU 0 and shard `1/2` on GPU 1.
- Shard 0 wrote 1,682,366 rows; shard 1 wrote 1,682,365 rows; strict merge wrote
  3,364,731 rows.
- The synchronized remote tracked tree remained identical to commit `10d5436`
  after the run.

### Recommendations

1. **Keep the packed, auto-sharded, FP32 encoder-cache path as the only formal
   S0 entry point.** This is the measured 9.815x production improvement.
2. **Retain the timing logs.** Pack load, encoder-cache construction, scoring,
   and merge timings are low-overhead operational checks against regression.
3. **Do not prioritize further token-budget tuning.** The controlled probe
   showed only a 1.2% gain from four times fewer batches.
4. **If another large gain is required, profile or redesign the long-sequence
   pair head.** It is now the dominant GPU-bound stage; feature I/O and encoder
   duplication are no longer the limiting factors.

## Part B — Profiling instrumentation changelog

| Files | Profiling purpose | Final state |
|---|---|---|
| `src/score_universe.py` | Pack-load, encoder-cache, and scoring timing/throughput logs. | Retained for production observability. |
| `full_dual_10d5436.log`, `full_dual_10d5436.status` | Formal dual-H20 launch, merge, exit status, and wall time. | Retained as latest-run evidence. |
| `full_dual_10d5436_shard-0.log`, `full_dual_10d5436_shard-1.log` | Final per-GPU shard progress and throughput. | Retained as latest-run evidence. |
| `full_run_summary.json`, `REPORT.md` | Machine-readable result and final human-readable report. | Retained as latest-run evidence. |
| `04_s0_score_ws2.log`, `04_s0_score_ws2.status`, `04_s0_score_ws2_gpu.csv`, `baseline_summary.json` | Legacy unpacked baseline evidence. | Deleted after final-run acceptance. |
| `compare_logits.py`, `inspect_encoder_dtype.py` | Temporary correctness and dtype probes. | Deleted after final-run acceptance. |
| `probe_tb262144.log`, `probe_tb262144.time`, `probe_long_tb262144.log`, `probe_long_tb262144.time`, `probe_long_tb1048576.log`, `probe_long_tb1048576.time` | Legacy and packed token-budget probes. | Deleted after final-run acceptance. |
| `probe_cached_tb262144.log`, `probe_cached_tb262144.time`, `probe_cached_fp32_tb262144.log`, `probe_cached_fp32_tb262144.time`, `probe_summary.json` | Encoder-cache probes and summary. | Deleted after final-run acceptance. |
| `run_full_remote.sh`, `full_dual_10d5436_launcher.log`, five temporary probe `.npz` files | Failed launcher and intermediate remote artifacts. | Deleted from local/HPC work areas. |

The final run used a foreground SSH command with an explicit status file. Raw
legacy/probe artifacts have been removed; only the latest formal-run evidence and
the low-overhead production timing logs remain.
