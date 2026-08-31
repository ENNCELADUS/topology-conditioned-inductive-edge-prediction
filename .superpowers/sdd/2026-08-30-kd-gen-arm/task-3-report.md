# Task 3 report: DistillConfig and losses teardown; `{w_gen}` arm group

## Result

Task 3 is complete on branch `kd-gen-arm`, based on `a544298`. The distillation
config now exposes the legal, non-stackable `{w_gen}` arm as `kd_gen`, retains
`gen_lr_scale`, adds `joint_warmup_frac`, and removes the legacy D9 config,
losses, trainer behavior, diagnostics, optimizer staging, and epoch KL plumbing.
The existing non-D9 KD arms and both `kd_set_*` losses remain intact.

## RED evidence

Tests were changed before implementation, then run with:

```text
.venv/bin/python -m pytest tests/distill/test_distill_config.py -n0 -q
```

Result: RED, exit 1, 7 failures. The first required failure was:

```text
TypeError: DistillConfig.__init__() got an unexpected keyword argument 'w_gen'
```

The mapping round-trip also failed with the expected unknown-interface evidence:

```text
ValueError: unknown distill config keys: ['joint_warmup_frac', 'w_gen']
```

The `w_seed` removal test also correctly failed before implementation because
`w_seed` was still recognized rather than reported as an unknown config key.

## Implementation and exact teardown

### `src/distill/config.py`

- Added only `w_gen: float = 0.0` and `joint_warmup_frac: float = 0.1`.
- Retained `gen_lr_scale: float = 0.1`.
- Removed `w_seed`, `w_geom`, `w_kl`, `kl_warmup_steps`,
  `joint_start_epoch`, and `_INT_FIELDS`, including integer parsing.
- Added the `0 <= joint_warmup_frac < 1` validation.
- Replaced the D9 legality pattern with `frozenset({"w_gen"}) -> "kd_gen"`.
- Kept the one-arm-only rule, so `w_gen` cannot stack with any other KD arm.
- Rewrote module and class documentation for generator loss weight, adapter
  stop-gradient warmup fraction, and joint-phase generator LR scaling.

### `src/distill/losses.py`

- Deleted only `kd_seed_loss`, `kd_seed_gram_loss`, and `kd_kl_loss`, plus
  their `__all__` entries.
- Preserved `kd_set_seed_loss`, `kd_set_gram_loss`, and all remaining B1 KD
  losses unchanged.

### `src/train_b0.py`

- Removed the three retired loss imports.
- Simplified `_build_optimizer` to ordinary `AdamW(model.parameters(), ...)`.
- Deleted `_sync_pair_latent_generator_lr` and
  `_set_pair_latent_training_stage`, including both stage calls and both
  post-scheduler LR-sync calls.
- Removed D9 validation seed injection, RNG forking, prior-seed diagnostics,
  and the `val_kd_prior_cos` path from `_evaluate_distributed`.
- Removed D9 config reads, architecture/target validation, training seed
  staging, validation seed staging, batch seed injection, KD loss terms,
  telemetry sums, and epoch telemetry.
- Reduced `KDRowBank.attach` to a no-op.
- Removed `epoch_kd_kl_dim_sum`, `kd_kl_dims` accumulation, and the KL tensor
  passed into production epoch telemetry.
- Kept the temporarily retained `KDValDiagnostics.teacher_seeds` field, set
  always to `None`, with no behavior or compatibility branch; Task 4 owns its
  explicit rename.
- Kept `load_kd_targets(..., load_seeds=False)` exactly as the temporary Task 7
  boundary.

### Tests

- Replaced D9 config cases with `kd_gen` arm, no-stacking, removed-`w_seed`,
  warmup-bound, negative-weight, missing-target, and mapping-round-trip cases.
- Removed imports and coverage for the three deleted D9 losses.
- Did not create or modify Task 4 tests.

## GREEN evidence

Focused config GREEN:

```text
.venv/bin/python -m pytest tests/distill/test_distill_config.py -n0 -q
25 passed
```

Required affected suite:

```text
GLOO_SOCKET_IFNAME=lo0 .venv/bin/python -m pytest \
  tests/distill tests/test_train_b0_kd.py tests/test_b0_topo_gen.py -n0 -q
113 passed
```

The first sandboxed run reached all non-Gloo tests but four internal
two-process Gloo tests failed before application code with `Cannot resolve
127.0.0.1 to a (local) address`. Binding `lo0` inside the sandbox confirmed
the restriction as `uv_bind: operation not permitted`. The final identical
pytest selection was therefore run outside the socket-restricted sandbox with
`GLOO_SOCKET_IFNAME=lo0` and passed. Warnings were limited to the existing
TorchScript deprecation and `float(loss)` warning in `test_b0_topo_gen.py`.

## Static evidence

```text
.venv/bin/python -m ruff format \
  src/distill/config.py src/distill/losses.py src/train_b0.py \
  tests/distill/test_distill_config.py tests/distill/test_kd_losses.py
2 files reformatted, 3 files left unchanged

.venv/bin/python -m ruff check src tests
All checks passed!

.venv/bin/python -m mypy src/distill src/train_b0.py
Success: no issues found in 6 source files

git diff --check
PASS
```

## Files changed

- `src/distill/config.py`
- `src/distill/losses.py`
- `src/train_b0.py`
- `tests/distill/test_distill_config.py`
- `tests/distill/test_kd_losses.py`
- `.superpowers/sdd/2026-08-30-kd-gen-arm/task-3-report.md`

No scorer, artifacts/teacher-target loader, configs, docs, plan/spec, Task 4
code, or Task 4 tests were edited.

## Self-review

- Confirmed the worktree started at the requested `a544298` base.
- Searched the scoped source for all retired D9 symbols, helpers, seed
  injection/staging names, optimizer staging names, and KL accumulator names;
  none remain.
- Reviewed the final diff for scope: 422 deletions and 77 insertions across the
  five code/test files before adding this report, with no unrelated edits.
- Confirmed the legal arm set still contains the four existing simple arms,
  `kd_rep`, and the new `kd_gen`, with no stacking.
- Confirmed `kd_set_*` definitions and exports remain.
- Confirmed ordinary AdamW is used for every arm until Task 5 restores the
  `kd_gen` parameter groups.

## Concerns and staged boundaries

No blocking Task 3 concern remains. As explicitly staged by the brief,
`kd_gen` is config-legal before its KDRowBank replacement (Task 4), generator
optimizer groups/warmup behavior (Task 5), and loader keyword removal (Task 7)
land. This commit does not claim a runnable standalone `kd_gen` training arm
until those later tasks are integrated.
