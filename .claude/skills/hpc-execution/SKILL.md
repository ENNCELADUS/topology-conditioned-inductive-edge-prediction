---
name: hpc-execution
description: Use before running or modifying any training, scoring, or gate command — locally or on the H20 container. Covers hpc/run.sh, g5_stage1.sh and qualification.sh semantics, world-size auto-detection, the score-once/analyze-many flow, and the config keys that silently change meaning per model family.
---

# HPC execution

`hpc/README.md` pins the SSH endpoint, paths, and versions. There is **no job
scheduler** (no Slurm). Every run records its detected world size; throughput
claims are hardware-shape-specific.

## The flow is score-once, analyze-many

Train → score the pair universe **once** into a pinned `.npz` → run gate
analyses over that cached artifact. Gate analyses (`src/experiments/g1_*`,
`g2_*`, `g3_*`, `g5_*`) are pure row-selection plus graph math; **no model
scoring happens in them**. If you find yourself loading a checkpoint inside a
gate, stop.

## `hpc/run.sh`

Subcommands: `check`, `train`, `s0-score`, `score`, `merge`, `g1`, `g2`.

- `assert_runtime` (`hpc/run.sh:59`) runs before **every** subcommand and hard-fails
  unless the repo is literally at `/2023533015/topology-conditioned-inductive-edge-prediction`,
  uv is at `/2023533015/.uv/bin/uv`, both data dirs exist, and **every** visible
  GPU is named exactly `NVIDIA H20`. One non-H20 device aborts the whole run.
- World size is auto-detected by counting `nvidia-smi` rows and exporting
  `CUDA_VISIBLE_DEVICES=0..N-1` (`hpc/run.sh:67`). `score` re-overrides it per
  shard; passing `--shard`/`--num-shards` yourself is explicitly rejected
  (`hpc/run.sh:88`).
- `train` **never** calls `src.train_b0` directly — it always execs
  `python -m src.e2_pipeline` (pack → probe → projection → 30-epoch
  `accelerate launch`). Formal E2 and EgoStitch training must go through this
  orchestrator; a hard-coded single-GPU launch is not a substitute.
- `check` is not a smoke test: it runs the full non-integration suite plus
  `tests/test_e2_ddp_integration.py -m "integration and not slow"`. It is the
  **only** place the DDP contracts actually execute (they are skipped on macOS).
- Exit code `2` from `train` is a **gated** failure (e.g. projected 30-epoch time
  over the 60-minute budget), not a crash. The runner deliberately does not mask
  it. `complete.json` is written last; its `total_seconds` is authoritative.
- `s0-score` must not be replaced with a direct `src.score_universe` call — the
  unpacked path re-reads feature files per pair and does not fan out across GPUs
  (`hpc/README.md:148`). It keeps `--amp bf16` but sets `--pair-amp off` so pair
  logits stay fp32 (`hpc/g5_stage1.sh:109`).
- **`B0-alt` is the exception**: its config schema is not the V3.1 shape
  `src.e2_pipeline` expects, so it keeps a direct
  `python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml`.

## `hpc/qualification.sh`

- `qualify` gates on five things: `EGOSTITCH_QUALIFICATION_ROOT` exists, its
  basename matches `attempt00[3-5]`, the target attempt dir does not already
  exist, the isolated root contains no test-side artifacts
  (`test_graph.pkl`, `candidate_test_edges.txt`, `test_edges.txt`, `*v_select*`),
  and the registration SHA-256 is byte-identical before and after (an `EXIT` trap).
- `formal` additionally requires **exactly 4** H20s (`qualify` accepts any count),
  registration `status == "BINDING"` with zero `REQUIRED-BEFORE-BINDING` markers,
  and for non-`full` arms a completed full-arm preflight whose `run_metadata.json`
  shows `status=complete`, `run_kind=formal`, `selected_checkpoint_eligible`,
  `validation_liveness_pass`, and a matching `preregistration_sha256`.
- The three stages use **three different feature packs** — `egostitch_e2e_v_fit`
  (overfit), `egostitch_e2e_v_qual` (rehearsal), `egostitch_e2e_v_select` (formal
  only). Reusing one pack across stages leaks `V_select` into qualification.

## `hpc/g5_stage1.sh`

- Hardcodes the audited B0 checkpoint id `e092537d8cf1e208` in **two** places
  (`:92` and `:164`) plus a dated deliverable path
  `outputs/deliverables/b0_v31_breadth_first_20260711/`. Rotating the B0
  checkpoint means editing both literals.
- `training_is_current` (`:77`) reuses a seed only when the checkpoint's recorded
  `output_dir` basename starts with `.e2-run-`, the staging temp dir created by
  `src/e2_pipeline.py:867`. A checkpoint written outside the orchestrator is
  always judged stale, and a mismatched config aborts rather than retraining.

## Config keys that bite

All loaders are **fail-closed on unknown keys** (`_check_no_unknown_keys`,
`src/train_b0.py:286`) — a typo raises rather than being ignored. Beyond that:

- `runtime.world_size` must be the literal string `"auto"` for EgoStitch; any
  integer raises (`src/train_egostitch.py:499`). `train_b0.py` accepts `auto` or a
  positive int. `auto` is stored internally as `0` and resolved by
  `detect_visible_gpu_count()`.
- `runtime.token_budget_candidates` means **token counts** for B0 but **per-rank
  node-stream batch sizes `B_n`** for EgoStitch. Same key, different unit.
- There are two distinct `pack_dir` keys — `data.pack_dir` (raw-token pack) and
  `runtime.pack_dir` — pointing at different directories. The CLI `--pack-dir`
  overrides only the `runtime` one.
- Keys are family-conditional: `data.s0_cache` is required only for
  `family: egostitch` and forced empty for `egostitch_e2e`;
  `optim.warmstart_fraction` is legal only for `egostitch`; a `training:` block is
  legal only for `egostitch_e2e`.
- `expected_missing_features: [node_004764, node_007050]` is load-bearing: those
  two nodes are known-missing from the feature cache and **any other** missing
  node is a hard error. Copying a config to new data without updating this fails
  at load.
- The four e2e arm configs are byte-identical except three fields —
  `permanent_null`, `p_topo`/`p_cont`, and the `output_dir` leaf — and the whole
  `training:` block must be identical across all four. `tests/test_hpc_qualification.py:36`
  asserts this exhaustively, so edit arms together.
- Identity keys that must match downstream: `s0_checkpoint_id`, and the score
  `meta` tuple `checkpoint_id`/`model_family`/`pairs_source`/`strategy`
  (`src/score_universe.py:715`, re-validated on merge).
