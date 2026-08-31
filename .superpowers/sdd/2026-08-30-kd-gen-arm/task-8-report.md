# Task 8 report — configs, docs, and D9 cleanup

## Status

DONE.

## RED evidence

- After adding the config-parse tests but before adding either run config, `uv run --offline python -m pytest tests/test_train_b0_kd_gen.py -n0 -q -k kd_gen_configs_parse` failed in both parametrized cases with `FileNotFoundError` for `configs/b1_kd_gen_{edm,det}_breadth_first.yaml`.
- The new PMA(1) invariant passed immediately against the existing encoder: one appended seed token and `pooled == tokens[:, -1, :]`.

## Implementation

- Added `configs/b1_kd_gen_edm_breadth_first.yaml` and `configs/b1_kd_gen_det_breadth_first.yaml` as exact `b1_kd_control` deltas: the `topo_gen` block, kd_gen output directory, and PMA(1) target/distill block are the only semantic additions or replacements. The deterministic control uses `name: det_mse`, `mc_samples: 1`, and has no `sampler_steps` key.
- Verified, without editing, the existing `configs/egostitch_e2e_v3_full_ego_teacher_pma1_breadth_first.yaml`: `grit_gmt`, `seeds: 1`, `conditioning_mode: pooled_adapter`, and `outputs/egostitch_e2e_stage1_v3/full_ego_teacher_pma1`; relative to the PMA(4) teacher its parsed config differs only in seed count and output directory.
- Deleted the retired `configs/b1_kd_d9_breadth_first.yaml` and `tests/test_b1_kd_d9_config.py`. The obsolete `docs/tmp` D9 file was already absent.
- Added run-config parsing/build checks to `tests/test_train_b0_kd_gen.py` and the PMA(1) pooled-seed invariant to `tests/model/test_grit_gmt_encoder.py`.
- Replaced the retired seed-artifact and D9 arm lines in `docs/results/b1_kd_arms.md` while preserving its 97-line count.
- Added `kd_gen` to the enumerated autoresearch `CAMPAIGNS` set and acceptance test, changed the invalid example to `kd_unknown`, and added no grid or sweep surface.
- Preserved the explicit `pair_latent_gen` removed-key guard and regression required by the ledger ruling.

## Verification

- Focused GREEN: `uv run --offline python -m pytest tests/test_train_b0_kd_gen.py tests/autoresearch/test_ledger.py tests/model/test_grit_gmt_encoder.py -n0 -q` — passed all selected tests; only the existing PyTorch JIT deprecation warning appeared.
- `uv run --offline ruff check` on the four changed Python test/source files — passed.
- `uv run --offline ruff format --check` on the four changed Python test/source files — all already formatted.
- `uv run --offline mypy` on the four changed Python test/source files — success, no issues.
- Parsed-dictionary comparison verified each new run config matches `b1_kd_control` outside the exact `topo_gen`, `output_dir`, and `distill` deltas.
- PMA(1)/PMA(4) parsed comparison verified only `encoder.seeds` and `output_dir` differ semantically.
- `rg -n 'kd_d9' src tests configs hpc` — no hits.
- `rg -n 'pair_latent_gen' src tests configs hpc` — only the intentional fail-fast source guard and its regression test.
- `wc -l docs/results/b1_kd_arms.md` — 97 before and after.
- `git diff --check` — passed before report creation and rerun at handoff.

## Concerns

- No Task 8 implementation concerns. No DDP or H20 work was run.
- `CONTEXT.md` is absent from this worktree and its sibling worktree tree; the checked-in `AGENTS.md`, full plan/spec, Task 8 brief/drift, and current SDD progress ledger/rulings were used instead.
