# Task 7 report — BINDING enforcement and paired-bootstrap control

## Status

DONE.

## RED evidence

- `tests/test_train_egostitch.py::TestRegistrationRunMode::test_formal_worker_refuses_draft_registration` initially failed because `PreregistrationNotBinding` did not exist.
- `tests/test_e2_pipeline.py::test_pipeline_forwards_debug_max_steps_to_worker` initially failed because `src.e2_pipeline` rejected `--max-steps` as unknown.

## Implementation

- `src/train_egostitch.py`: formal DDP entry rejects a registration unless `status == "BINDING"`; positive `--max-steps` redirects to a sibling `*_debug` directory, marks metadata `run_kind: debug`, and marks it non-formal. The gate rejects debug/non-formal metadata.
- `src/e2_pipeline.py`: accepts and forwards `--max-steps` to every worker invocation, preserving the bounded debug flag through the documented HPC entry point.
- `src/experiments/g5_stage1.py`: requires `BINDING`, verifies every formal registration SHA, requires completed explicitly formal E2E metadata, and computes the 6a-minus-full clustering-MMD paired lower bound from identical, one-stream, bucket-preserving resamples (`B=1000`, seed `0`, alpha `.05`). The condition passes only when the lower bound is positive and is explicitly scoped to fixed-seed evaluator stability.
- `src/experiments/g5_stage1.py --mode e2e`: accepts exactly the registered five artifacts, four run metadata paths, frozen B0/B0-cal inputs, registration, and output directory; it emits the E2E result/table artifacts with primary criteria, guards, liveness, pathway-attribution, structure-control, and `pass|cut` verdict.

## GREEN evidence

- `pytest tests/test_g5_stage1.py -q` — 46 passed after the E2E CLI and verdict implementation.
- Earlier combined regression: `pytest tests/test_train_egostitch.py tests/test_g5_stage1.py tests/test_score_universe.py tests/test_e2_pipeline.py -k 'not test_run_command_timeout_kills_the_whole_process_group' -q` — 100% passed before the final E2E CLI extension; its directly affected G5 suite was rerun afterward.
- Focused max-steps pass-through regression passed.
- `ruff check` and `ruff format --check` passed for all six owned files.
- Isolated-cache `mypy --strict` passed for `src/train_egostitch.py`, `src/experiments/g5_stage1.py`, `src/e2_pipeline.py`, and the three corresponding test files.
- `git diff --check` passed.

## Gates and scope

- Formal training and held-out gate publication fail closed without a BINDING registration.
- The four formal metadata records must all carry the exact registration SHA and declare `run_kind=formal`, `formal_artifacts_published=true`, and `status=complete`; the 6a control still shares the full checkpoint.
- Debug checkpoint artifacts remain in `*_debug` only and cannot be consumed by the gate as formal metadata.
- The paired bootstrap is evaluator-stability evidence at one fixed training seed, not a significance or cross-seed claim.

## Self-review

- The worker boundary is enforced inside `_run_ddp_worker`, not solely at CLI dispatch, so direct worker invocation cannot bypass it.
- The bootstrap reuses one captured descriptor set per arm and reuses identical resampled indices across arms; it does not independently re-evaluate a different random subgraph universe per arm.
- The E2E CLI evaluates the registered primary/guard/pathway/control verdict rather than only returning a summary helper payload.
- No changes were made to `src/model/B0.py`.

## Concerns

- The complete `tests/test_e2_pipeline.py` has one unrelated sandbox failure: its process-group timeout test cannot invoke `ps` (`PermissionError: Operation not permitted`). The Task 7 pass-through regression and the complete relevant suite excluding that sandbox-only check passed.
- The full repository suite was not run; the documented local DDP hang was avoided.

## Commit

`74cd63e` — `feat(g5-e2e): machine-enforced BINDING status + paired-bootstrap structure-control condition`

## Review-fix addendum (2026-07-17)

- Corrected the debug path: `e2_pipeline --max-steps N` chooses `<formal>_debug`
  before execution, sends all DDP modes there, accepts the bounded completed-epoch
  scope, and leaves the formal output root untouched. A real fake-worker
  `run_pipeline` regression now completes that path and verifies debug-only output.
- Restored historical frozen-s0 registration compatibility: BINDING is an E2E-only
  status rule; the real no-status frozen-s0 registration shape remains valid.
- E2E now binds four distinct formal metadata files and checkpoint identities, and
  validates the registered `permanent_null` / `p_topo` / `p_cont` semantics for
  full, F-only, pair-topology, and P0. Worker metadata records these fields.
- Registration JSON is parsed and hashed from one byte snapshot. The captured hash
  is carried through worker finalization, preventing mixed-version status/SHA use.
- B0-cal accepts the Task-11 deliverable directory only when it resolves exactly
  one `b0cal_results.json`; E2E publication stages both verdict files and swaps the
  directory atomically after clearing stale authoritative files on a failed attempt.
- The formal breadth-first bootstrap path asserts its fixed evaluator has exactly
  500 induced subgraphs before resampling.

Verification after review fixes: focused Task-7 tests excluding only the documented
sandbox process-group timeout check passed; ruff check/format, diff check, and
isolated strict mypy all passed.
