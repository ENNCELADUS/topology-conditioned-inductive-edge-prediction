# Task 10 Report — Topology Probe + Final Gate

## Result

Implemented the paper-name topology probe at the dated-plan path, added strict
RED-to-GREEN coverage, and applied the ledger-ruling mypy baseline correction.
No H20 or remote command was run. `CONTEXT.md` is absent from this worktree;
`AGENTS.md`, the full kd_gen design spec, Task 10 plan section, brief, drift map,
and progress-ledger rulings were read before editing.

## Changed files

- `src/experiments/seed_topology_probe.py`
- `tests/test_seed_topology_probe.py`
- `tests/test_sweep_configs.py`
- `.superpowers/sdd/2026-08-30-kd-gen-arm/task-10-report.md`

The probe implements NumPy-only deterministic five-fold OOF ridge with
train-fold z-scoring, an unregularized intercept, and `1e-3` feature ridge. It
computes degree, common-neighbor, and clustering targets from a fresh graph copy
per row after removing that row's query edge. The generated control is one
fixed seed-0 row shuffle shared by all five statistics. The CLI consumes Task 7
validation rows, a generated `latents` NPZ, and the exact
`load_benchmark`/`derive_val_region_split`/`truth_graph_for_kd` training graph
convention before writing the exact nested JSON schema.

## TDD and focused verification

- RED: `uv run --offline python -m pytest tests/test_seed_topology_probe.py -n0 -q`
  reached collection and failed with the expected missing
  `src.experiments.seed_topology_probe` import.
- Focused GREEN: probe plus sweep-config tests — `43 passed`, 2 existing
  PyTorch JIT deprecation warnings.
- Focused ruff check: passed.
- Focused ruff format: all three assigned Python files formatted.
- Focused mypy: success in 3 source files.
- `git diff --check`: passed.

## Final gates

- Required exact pytest command,
  `uv run --offline python -m pytest --ignore=tests/test_e2_ddp_integration.py`:
  `1853 passed, 6 skipped, 5 failed, 21 warnings` in 57.05 s. Four failures are
  local Gloo subprocess cases in `tests/test_train_b0_kd.py` (outside the
  documented excluded file) failing to resolve `127.0.0.1`; the fifth is the
  pre-existing `tests/test_score_universe_e2e.py` fixture omitting the Task 6
  required `topo_gen_control` shard metadata. These files are outside Task 10
  ownership and were not changed.
- Non-DDP diagnostic rerun, deselecting those four Gloo cases and the one stale
  Task 6 fixture: `1853 passed, 6 skipped, 21 warnings` in 53.65 s.
- Repo-wide ruff check: passed.
- Repo-wide ruff format check: `195 files already formatted`; 8 unassigned,
  pre-existing files would be reformatted. They were preserved.
- Repo-wide mypy: `Found 98 errors in 14 files (checked 203 source files)`,
  exactly meeting the ledger ceiling; changed surfaces have zero errors.

## Concerns

The literal full pytest and repo-wide format gates are not globally green for
pre-existing, unassigned reasons recorded above. The Task 10 surface and the
non-DDP remainder are green. The first literal full-suite run unexpectedly
entered four local Gloo tests despite the documented single-file exclusion;
subsequent verification explicitly deselected them.

## Parent final-gate correction

The worker's first full command incorrectly allowed four local Gloo/DDP cases in
`tests/test_train_b0_kd.py`; they attempted and failed on macOS loopback resolution. This violated
the user's local no-DDP boundary. No H20 or remote command ran. The parent then fixed the one real
non-DDP regression by stamping `topo_gen_control: null` in the legacy score fixture, applied the
plan-required repository-wide Ruff formatting, and committed both as `396dceb`.

All post-cleanup gates used the required `.venv/bin/python -m ...` surface:

- Full non-DDP pytest with `tests/test_e2_ddp_integration.py` ignored and the three parameterized
  relational-Gloo cases plus one production-Gloo case explicitly deselected:
  `1854 passed, 6 skipped, 21 warnings` in 60.72 s.
- Ruff: `All checks passed!`; format check: `203 files already formatted`.
- Mypy: `Found 98 errors in 14 files (checked 203 source files)`, meeting the required ceiling;
  no changed kd_gen/probe surface appears in the output.
- `git diff --check` and worktree status: clean.
