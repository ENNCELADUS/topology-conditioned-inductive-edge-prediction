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
