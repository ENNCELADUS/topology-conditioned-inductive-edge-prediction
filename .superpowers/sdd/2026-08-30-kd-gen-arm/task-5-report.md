# Task 5 Report — Warmup stage and generator optimizer group

Date: 2026-08-31

Status: COMPLETE

## Scope and files

- `src/train_b0.py`
  - Restored a kd_gen-specific two-group AdamW optimizer using the existing unwrapped-model helper.
  - Added `_set_topo_gen_training_stage`.
  - Wired the stage update at the start of both production epoch loops.
- `tests/test_train_b0_kd_gen.py`
  - Added focused optimizer partition, epoch-boundary, stop-gradient, joint-gradient, and all-parameter liveness coverage.

No Task 6/7, configuration, documentation, or unrelated source changes were made.

## Contract implemented

- Optimizer groups are named `base` and `topo_gen` for active kd_gen.
- The `topo_gen` group is exactly `V3_1.topo_gen_parameters()`; fusion parameters such as `topo_gen.gate` remain in `base`.
- Models without generator parameters retain the ordinary single-group AdamW path.
- A kd_gen partition with no base parameters fails closed.
- Warmup length is `ceil(joint_warmup_frac * total_epochs)` using 1-based epochs.
- `joint_stage` becomes true only on the first epoch after warmup.
- The generator LR equals the current base LR in warmup and `gen_lr_scale * base LR` in joint stage.
- Missing named optimizer groups fail closed.

## Epoch-numbering verification

Verified before wiring:

- Single-process training uses `for epoch in range(1, cfg.optim.epochs + 1)` (`src/train_b0.py:1342` after this change).
- DDP training initializes at epoch 1, resumes with `start_epoch = completed_epoch + 1` (`src/train_b0.py:3551`), and loops through `range(start_epoch, cfg.optim.epochs + 1)` (`src/train_b0.py:3557`).

Both stage calls therefore receive 1-based epoch numbers, including resumed DDP runs.

## Strict TDD evidence

RED command:

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py -n0 -q -k 'optimizer_groups or warmup_stops or task_plus_generator_loss'
```

Expected failure observed (exit 2 during collection):

```text
ImportError: cannot import name '_set_topo_gen_training_stage' from 'src.train_b0'
```

Focused GREEN command after implementation:

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py -n0 -q -k 'optimizer_groups or warmup_stops or task_plus_generator_loss'
```

Result: exit 0, four focused tests passed.

Full requested non-DDP GREEN command:

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py tests/test_train_b0.py -n0 -q
```

Result: exit 0; all collected Task 4/5 and ordinary trainer tests passed. The only output beyond test progress was the pre-existing TorchScript deprecation warning.

## Static checks and diff check

```text
rtk proxy .venv/bin/python -m ruff format src/train_b0.py tests/test_train_b0_kd_gen.py
```

Result: two touched files formatted.

```text
rtk proxy .venv/bin/python -m ruff check src/train_b0.py tests/test_train_b0_kd_gen.py
```

Result: `All checks passed!`

```text
rtk proxy .venv/bin/python -m mypy src/train_b0.py tests/test_train_b0_kd_gen.py
```

Result: `Success: no issues found in 2 source files`.

```text
rtk git diff --check
```

Result: exit 0 with no whitespace errors.

## Self-review

- Confirmed the generator group matches `topo_gen_parameters()` by parameter identity.
- Confirmed the fusion gate remains in the base group.
- Confirmed the 25-epoch, 0.1-fraction boundary: epoch 3 is warmup and epoch 4 is joint.
- Confirmed task loss alone cannot reach generator-core parameters during warmup and can reach them in joint stage. The joint-path assertion initializes the zero-initialized adapter-up weight nonzero to represent the post-warmup pathway and avoid mistaking zero initialization for a disconnected graph.
- Confirmed task plus generator loss gives every trainable parameter a non-`None` gradient in both phases.
- Confirmed ordinary trainer tests remain green.

## Concerns / limitations

- Per instruction, no local DDP tests were run. The DDP loop was verified by source inspection and receives the same tested stage helper, but multi-rank runtime behavior remains for the authorized environment.
- No Task 5 blocker or API ambiguity remains.

## Fix round 1 — scheduler ratio resynchronization

Independent review identified that both supported per-step schedulers write every parameter-group
LR during `scheduler.step()`, undoing the kd_gen generator/base ratio set at epoch start. The
single-process and DDP loops now call `_set_topo_gen_training_stage` immediately after every
scheduler step, with the current 1-based epoch and configured total epoch count unchanged.

Strict RED used the production single-process loop and its supported default `LambdaLR`. The test
observed the scheduler overwrite the generator/base ratio to 1.0 in both warmup and joint epochs,
then failed because no post-step stage resynchronization occurred:

```text
rtk proxy uv run python -m pytest tests/test_train_b0_kd_gen.py::test_kd_gen_train_loop_resyncs_lr_ratio_after_lambda_scheduler_step -n0 -q
FAILED: synced ratios contained only epoch-start values [(1, 1.0), (2, 0.1)], not post-step values
```

Focused GREEN passed after the two loop call-site changes. The regression verifies ratio 1.0 after
the warmup scheduler step and ratio `gen_lr_scale == 0.1` after the joint-stage scheduler step:

```text
rtk proxy uv run python -m pytest tests/test_train_b0_kd_gen.py::test_kd_gen_train_loop_resyncs_lr_ratio_after_lambda_scheduler_step -n0 -q
1 passed; 2 pre-existing TorchScript deprecation warnings
```

Task 5 non-DDP suites and focused checks:

```text
rtk proxy uv run python -m pytest tests/test_train_b0_kd_gen.py tests/test_train_b0.py -n0 -q
all collected tests passed; 2 pre-existing TorchScript deprecation warnings

rtk proxy uv run ruff format --check src/train_b0.py tests/test_train_b0_kd_gen.py
2 files already formatted

rtk proxy uv run ruff check src/train_b0.py tests/test_train_b0_kd_gen.py
All checks passed!

rtk proxy uv run mypy src/train_b0.py tests/test_train_b0_kd_gen.py
Success: no issues found in 2 source files

rtk git diff --check
exit 0 with no whitespace errors
```

Per instruction, no local DDP or H20 run was performed. The DDP loop receives the same tested stage
helper immediately after its scheduler step; runtime multi-rank behavior remains unexecuted here.
The deferred `no_grad` minor was not changed because the shared differentiable sample is required
for generator-loss liveness and the detached task branch already enforces warmup stop-gradient.
