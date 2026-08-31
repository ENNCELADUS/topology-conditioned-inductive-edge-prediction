# Task 6 report — packed scorer controls

Date: 2026-08-31

## Scope delivered

- Preserved Task 2's existing shared `_score_v3_1_packed` topology-generator branch; no duplicate
  scoring path was added.
- Added direct `V3_1.forward` / `TopoGenBase.marginal_forward` equivalence coverage.
- Added a production `_score_v3_1_packed` regression with nonzero AdaLN modulation and adapter-up
  weights. The test proves the generator reference differs from the retired output-head-only path,
  then requires exact equality from the production packed scorer.
- Added `--topo-gen-control` with the existing `branch_zero` / `shuffle` vocabulary.
- Applied the control to the checkpoint-loaded model before `model.to(device)` and pair access.
  A checkpoint without `model.config.topo_gen` exits clearly instead of silently ignoring the flag.
- Kept the flag scoped to direct `src.score_universe` / `hpc/run.sh score`; no test-protocol or
  `hpc/run.sh` changes were made.

## TDD evidence

RED command:

`UV_CACHE_DIR=/tmp/kd-gen-task6-uv-cache uv run python -m pytest tests/test_score_universe.py -n0 -q -k 'topo_gen_control or live_topo_gen or direct_marginal'`

Result before implementation: 2 passed, 5 failed. Both generator-path equivalence tests passed on
Task 2's existing scorer branch; parser, assignment, and no-generator tests failed because argparse
rejected the missing `--topo-gen-control` option. An earlier invocation did not start pytest because
the sandbox denied uv's global cache; rerunning with the task-local cache above established RED.

Focused GREEN: the same command passed 7 tests.

## Final verification

- `UV_CACHE_DIR=/tmp/kd-gen-task6-uv-cache uv run python -m pytest tests/test_score_universe.py -n0 -q`
  — 73 passed; only two pre-existing PyTorch `torch.jit.script` deprecation warnings.
- `UV_CACHE_DIR=/tmp/kd-gen-task6-uv-cache uv run ruff check src/score_universe.py tests/test_score_universe.py`
  — all checks passed.
- `UV_CACHE_DIR=/tmp/kd-gen-task6-uv-cache uv run mypy src/score_universe.py tests/test_score_universe.py`
  — success, no issues in 2 source files.
- `git diff --check` — clean.

## Concerns / exclusions

- No DDP or H20 execution was performed, as required.
- The control is intentionally unavailable through `src.eval.test_protocol` / `hpc/run.sh test`;
  the progress-ledger ruling limits Task 6 to direct score-universe scoring.

## Fix round 1 — scoring-control provenance

- Added score-artifact provenance for `topo_gen_control`: controlled runs record the selected mode,
  while live runs record explicit JSON `null`.
- Added `topo_gen_control` to shard merge agreement. Existing artifacts without the key retain the
  prior backfill behavior because merge comparison continues to use `meta.get(...)`.
- RED: focused metadata/merge tests failed 3 tests before implementation (both missing metadata
  cases and the missing mismatch rejection).
- GREEN: the same focused selection passed 3 tests after implementation.
- Final verification: `tests/test_score_universe.py -n0 -q` passed 76 tests; Ruff passed; focused
  mypy passed; `git diff --check` passed. No DDP or H20 execution was performed.

## Fix round 2 — reject unstamped scoring-control shards

- Added a regression proving a shard missing `topo_gen_control` cannot merge with a shard carrying
  explicit JSON `null`.
- Merge now requires every shard to contain `topo_gen_control` and compares the recorded values
  directly; pre-fix unstamped shards fail closed.
- Current EgoStitch scoring also records explicit `null`, preserving the common score-metadata
  convention for newly written shards.
- RED: the focused missing-key test failed because merge did not raise. The first invocation was
  blocked by uv's global-cache permissions; rerunning with `UV_CACHE_DIR` under `/tmp` established
  RED.
- GREEN: the missing-key and controlled-value focused tests passed (2 tests), and the broader merge
  regression selection passed (6 tests).
- Final verification: `tests/test_score_universe.py -n0 -q` passed 77 tests; Ruff passed; focused
  mypy passed. No DDP or H20 execution was performed.
