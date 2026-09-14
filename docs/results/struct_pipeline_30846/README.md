# Structural-stream pipeline benchmark — 30846

Engineering measurements for `topo_prompt_full_struct` on four H20s. All nine
benchmark runs completed successfully on 2026-09-13. Each restored epoch 1,
global step 271 from attempt `eb4fcc982eb943d2b2f1ba3a9c8e8dae`, used 10 warmup
updates and 40 measured updates, and discarded the benchmark updates. Per-step
task and structural pair counts match across all nine reports. Variant order
was rotated across three rounds.

| Implementation | Round 1 s/step | Round 2 | Round 3 | Median of round medians | Reduction vs legacy |
|---|---:|---:|---:|---:|---:|
| Legacy, one rank per subgraph | 14.085 | 14.350 | 14.389 | 14.350 | — |
| Coordinates prepared outside checkpoint | 13.156 | 13.326 | 13.215 | 13.215 | 7.9% |
| Coordinate reuse + distributed subgraph scoring | 5.596 | 5.607 | 5.638 | **5.607** | **60.9%** |

Each step uses the slowest rank's synchronized wall time. Peak allocated memory
was 86.88 GiB for all variants; peak reserved memory was 87.17 GiB for legacy and
coordinate reuse, and 87.30 GiB for full parallel scoring. Full parallel scoring
was selected for its repeated end-to-end performance, exceeding the planned
10% engineering adoption threshold. Sampling, loss weights, task batches,
optimizer schedule, and validation protocol were retained.

GPU 1 had substantially different thermal conditions across variants. Its
thermal-slowdown flag was active for 44.7–67.3% of legacy timed samples,
46.6–72.3% of coordinate-only samples, and 90.3–92.1% of full-parallel samples.
Its mean SM clock was approximately 1828–1864 MHz for legacy and 1523–1557 MHz
for full parallel scoring. These are observed timings on this host with its
thermal limitation; they do not establish a temperature-controlled software-only
speedup. GPU power limits and masks were unchanged.

## Provenance and verification

- Endpoint: `ssh -p 30846 root@10.15.171.204`, host `gglu55qtu8i5-0`.
- Isolated checkouts: `/2023533015/struct-pipeline-bench/{legacy,coordinate-only,full}`.
- Measured HEADs: legacy `7d567a2`, coordinate-only `1b15204`, full `ebc306e`.
  Untracked `.venv` and `outputs` symlinks supply shared resources; tracked source
  had no local diff. Tracked split metadata remained in each checkout.
- Reports, logs, and per-second GPU CSVs:
  `/2023533015/struct-pipeline-bench/results/20260913/`.
  `campaign_status.json` records all nine completions; `*-r*.json` contains
  per-step measurements and `*.json.gpu.csv` the thermal evidence.
- Local targeted suites passed: 209 tests, including real 2/4-rank gradient
  equivalence, empty scoring ranks, actual prompt/coordinate-generator models,
  rebuilt DDP buckets, and isolated snapshot recovery. Ruff and targeted mypy
  passed. Six pre-existing failures in the full HPC-script test file were
  reproduced on the original implementation; the 10 relevant runner checks passed.

## Training recovery

The original run saved one complete epoch: training 3893.0 s and validation
1502.7 s. It was stopped for the benchmark with that snapshot preserved.
The benchmark campaign did not automatically resume training after the
interactive implementation turn was interrupted.

Training was resumed on 2026-09-14 at 03:01 UTC from the complete epoch-1 state,
including optimizer, scheduler and all four rank RNG states. The selected
checkout is at `9d75ee9`, with the same measured implementation and additional
documentation/tests. New attempt: `e61a4817ad2e458dabd3eed588cb417c`.
At the user's request, acceptance was narrowed to confirming the full pipeline
launch, with no continuing monitoring. At 03:19 UTC, the four-rank worker had
reached epoch 2/25, global step 381, with no failure marker. The live pipeline
has neither `--skip-test` nor a debug step limit: it will publish and run the
diagnostic test stage after normal training termination. The training/test run
has not yet completed.
