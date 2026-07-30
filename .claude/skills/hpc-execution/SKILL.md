---
name: hpc-execution
description: Use before running or modifying any training, scoring, or gate command — locally or on the H20 container. Covers hpc/run.sh and the two-stage qualification.sh ladder, world-size auto-detection, the score-once/analyze-many flow, and the config keys that silently change meaning per model family.
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

Subcommands: `check`, `train`, `score`, `merge`, `g1`, `g2`. EgoStitch E2E training is
**not** launched from here — both stages go through `hpc/qualification.sh`.

- `assert_runtime` (`hpc/run.sh:52`) runs before **every** subcommand and hard-fails
  unless the repo is literally at `/2023533015/topology-conditioned-inductive-edge-prediction`,
  uv is at `/2023533015/.uv/bin/uv`, both data dirs exist, and **every** visible
  GPU is named exactly `NVIDIA H20`. One non-H20 device aborts the whole run.
- World size is auto-detected by counting `nvidia-smi` rows and exporting
  `CUDA_VISIBLE_DEVICES=0..N-1` (`hpc/run.sh:60-68`). `score` re-overrides it per
  shard; passing `--shard`/`--num-shards` yourself is explicitly rejected
  (`hpc/run.sh:81-83`).
- `train` **never** calls `src.train_b0` directly — it always execs
  `python -m src.e2_pipeline`, whose sub-stages are `pack → train → publish`. The
  token-budget `probe` and runtime `projection` sub-stages are gone, so nothing
  pre-estimates runtime any more. Formal E2 training must go through this orchestrator;
  a hard-coded single-GPU launch is not a substitute.
- `check` is not a smoke test: it runs the full non-integration suite plus
  `tests/test_e2_ddp_integration.py -m "integration and not slow"`. It is the
  **only** place the DDP contracts actually execute (they are skipped on macOS).
- Exit code `2` from `train` is a **gated** failure, not a crash: the offending
  sub-stage is named in `failure.json`. The runner deliberately does not mask it.
  `complete.json` is written last; its `total_seconds` is authoritative.
- **`B0-alt` is the exception**: its config schema is not the V3.1 shape
  `src.e2_pipeline` expects, so it keeps a direct
  `python -m src.train_b0 --config configs/b0_alt_breadth_first.yaml`.

## `hpc/qualification.sh` — the two-stage E2E ladder

`qualify <arm>` then `formal <arm>`, over the six trained arms
(`full`, `f_only`, `pair_topology`, `p0`, `cosine_pool`, `no_l_rel`).
`structure_control_6a_v3` and `structure_control_6e_v1` are **scoring-time controls
that reuse the full arm's checkpoint** — the arm selector rejects them by name.

- **Both stages train on the identical universe**: full `V_fit`, validated on the
  single 512-node `V_hold`. They differ only in `optim.epochs` (qualify runs the
  registered short schedule, 3 epochs). That identity is what makes the Stage-2
  `feature_stats_sha256` check a genuine *equality* rather than a hand-pasted pin, so
  never give the two stages different training universes, samplers, or pack contents.
- There is **no sanitized data root and no attempt window** any more. The held-out
  boundary is a *path* check inside the worker that applies to **both** run kinds, so
  both commands run directly in the repository checkout.
- `qualify` is the development loop: any visible-H20 count, no clean checkout required,
  and no **BINDING** registration required (it may run against the active v4 `DRAFT`).
  Its verdict is **guards-only** — `pass` iff no fail-fast guard tripped. Every
  invocation is retained under
  `outputs/egostitch_e2e_stage1_v3/qualification/<arm>/attempts/attempt-*/`; `latest`
  points to the newest attempt, including a failure, while `latest-pass` advances only
  after success. This durable history makes the cumulative `V_hold` evaluation count
  `K` auditable, and qualification is frozen once the registration becomes `BINDING`.
  Named failures include `training_invalid(slot_collapse)`,
  `training_invalid(initial_slot_collapse)` and `fail(no_eligible_checkpoint)`.
- **Guards-only never means eligibility-off.** `e2e_checkpoint_eligible` runs in both
  stages; a qualify run that finds no eligible checkpoint fails, it does not fall back.
- `formal` requires **exactly 4** H20s, a clean checkout, registration
  `status == "BINDING"` with every required binding-evidence field resolved, a passing
  `latest-pass/qualification.json` for the same arm, and — for non-`full` arms — a completed
  full-arm preflight whose `run_metadata.json` shows `status=complete`,
  `run_kind=formal`, `selected_checkpoint_eligible`, `validation_liveness_pass` and a
  matching `preregistration_sha256`. Execution order is `full` first, always.
- After training, `formal` runs `validate_e2e_qualification_profile` on `profile.json`
  (the only clip-coefficient / family-ratio / submodule-RMS margin gate in the repo)
  and, for `full` only, produces the registered `formal_train` probe artifact. The G5
  gate evaluator refuses to run without that artifact, and its path is read out of the
  registration rather than restated in the script.
- Both stages install an `EXIT` trap asserting the registration SHA-256 is byte-identical
  before and after. Neither stage may edit or promote the registration.
- **Packs are keyed by `n_ground`, not by stage.** Neither stage passes `--pack-dir`;
  each config names its own `runtime.pack_dir`, and both stages of an arm share it.
  `cosine_pool` pins `n_ground: 20` and therefore gets its own pack — forcing one shared
  pack across arms makes it raise on the pack manifest.

## Config keys that bite

All loaders are **fail-closed on unknown keys** (`_check_no_unknown_keys`,
`src/train_b0.py:294`) — a typo raises rather than being ignored. Beyond that:

- `runtime.world_size` must be the literal string `"auto"` for EgoStitch; any
  integer raises (`src/train_egostitch.py:530-532`). `train_b0.py` accepts `auto` or a
  positive int. `auto` is stored internally as `0` and resolved by
  `detect_visible_gpu_count()`.
- `runtime.token_budget` is a **scalar** (the candidate-list form is gone). It means
  **tokens per global batch** for B0 but the **per-rank node-stream batch size `B_n`**
  for EgoStitch. Same key, different unit.
- There are two distinct `pack_dir` keys — `data.pack_dir` (raw-token pack, shared by
  every arm) and `runtime.pack_dir` (F0/grounding pack, keyed by `n_ground`) — pointing
  at different directories. The CLI `--pack-dir` overrides only the `runtime` one.
- `data.s0_cache`, `data.s0_checkpoint_id` and `optim.warmstart_fraction` are
  **deprecated no-ops**, retained only so `_config_hash` stays stable against already
  published `run_metadata.json` and probe artifacts. Do not set them in an
  `egostitch_e2e` config: `s0_cache` is forced empty, but `s0_checkpoint_id` still lands
  in `config_to_dict` and would move the hash. A `training:` block is legal only for
  `egostitch_e2e`.
- `expected_missing_features: [node_004764, node_007050]` is load-bearing: those
  two nodes are known-missing from the feature cache and **any other** missing
  node is a hard error. Copying a config to new data without updating this fails
  at load.
- The six e2e arm configs differ only in `permanent_null`, `p_topo`/`p_cont`, the
  `output_dir` leaf, `n_ground` (20 for `cosine_pool`, else 50), `runtime.pack_dir`, and
  `w_rel: 0` on `no_l_rel`. The whole `training:` block and the `preregistration` path
  must be identical across all six; `tests/test_hpc_qualification.py` asserts this
  exhaustively, so edit arms together.
- Identity keys that must match downstream: the two ladder digests
  `feature_stats_sha256` and `model_config_sha256` (`model_config_hash`, **not**
  `_config_hash`, which bakes in `output_dir` and `optim.epochs` and therefore differs
  between the stages by construction), and the score `meta` tuple
  `checkpoint_id`/`model_family`/`pairs_source`/`strategy` (`src/score_universe.py`,
  re-validated on merge).
