# Task 4 report: KDRowBank `kd_gen` arm and validation diagnostics

## Result

Task 4 is complete on branch `kd-gen-arm`, based on `9a150e4`. `KDRowBank`
now validates and stages the `kd_gen` arm, stamps the artifact-wide fp64
teacher-representation RMS into the generator checkpoint buffer, injects
normalized row-aligned teacher latents, adds the weighted generator loss once,
and reports train/validation generator diagnostics. The clean
`loss(batch, output) -> (total, stats)` API remains unchanged.

## RED evidence

The new Task 4 tests and the required `KDValDiagnostics` constructor renames
were written before production implementation. Command:

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py -n0 -q
```

Result: exit 1, eight failures. The requested missing-bank-validation failure
was explicit:

```text
FAILED tests/test_train_b0_kd_gen.py::test_kd_gen_requires_topo_gen_both_directions
E       Failed: DID NOT RAISE <class 'RuntimeError'>
```

The same RED run also showed the absent `_topo_gen`, latent-width guard, RMS
stamp/validation, `kd_teacher_latent` attachment, generator telemetry, and
`val_kd_latent_cos` behavior.

## Implementation

### `src/train_b0.py`

- Imported `TopoGenBase` and renamed the legacy-specific unwrap helper to the
  reusable `_unwrapped_model`; the bank accepts prepared/DDP-style `.module`
  wrappers.
- Added the two-way `topo_gen` / positive-`w_gen` guard and exact
  `topo_gen.latent_dim == teacher_rep.shape[1]` validation.
- Computed RMS over every training `teacher_rep` element after fp64 conversion,
  rejected non-finite/non-positive scales, and stamped the generator's
  checkpointed `latent_rms_scale` buffer.
- Staged training and validation `teacher_rep` tensors for `kd_gen`; `attach`
  joins by `_row_id`, converts to fp32, and normalizes without mutating the bank.
- Added `w_gen * gen_loss` exactly once to the bank scalar and emitted row-sum
  telemetry for generator loss, latent cosine, MC probability spread, sample
  dispersion, branch ratio, and optional EDM sigma-bin loss/count pairs.
- Added epoch telemetry after the existing cross-rank sum, including
  `kd_gen_sigma_q1..q4` only for positive reduced bin counts and the live
  tanh-transformed generator gate.
- Renamed `KDValDiagnostics.teacher_seeds` to `teacher_latent`, added
  `latent_scale`, normalized only at validation injection, preserved RNG with
  `torch.random.fork_rng`, and emitted `val_kd_latent_cos` without a rep-loss
  alias.
- Retained `load_kd_targets(..., load_seeds=False)` and the existing row-local
  DDP scaling path. The generic first-step term-gradient probe consumes the
  returned KD scalar directly, so no arm-specific mapping was needed.

### Tests

- Added `tests/test_train_b0_kd_gen.py` for both-direction guards, prepared-model
  unwrapping, latent-width mismatch, exact fp64 RMS stamping, invalid-scale
  failure, normalized attachment, one-time weighting, required sums, EDM
  quartile aggregation, generator gradient-probe visibility, validation RNG
  preservation, and `val_kd_latent_cos`.
- Updated only the four `KDValDiagnostics` constructors in
  `tests/test_train_b0_kd.py` from `teacher_seeds=None` to
  `teacher_latent=None`; ordinary KD behavior remained unchanged.

## GREEN evidence

Final Task 4 plus ordinary-KD command, excluding exactly the four authorized
local Gloo/DDP cases (three parameterizations plus one production seam):

```text
rtk proxy .venv/bin/python -m pytest \
  tests/test_train_b0_kd_gen.py tests/test_train_b0_kd.py -n0 \
  -k 'not test_relational_kd_gathers_cross_rank_rows_with_exact_ddp_gradient and not test_production_kd_row_bank_keeps_global_loss_unscaled_and_one_pass'
```

Exact result:

```text
39 passed, 4 deselected, 2 warnings in 1.53s
```

The two warnings are the existing TorchScript deprecation warnings.

## Formatting and static evidence

Formatting was run on all three touched code/test files. Ruff initially
reformatted two files; two unrelated pre-existing line wraps in the ordinary-KD
test were then restored to keep its change constructor-only. Final checks:

```text
rtk proxy .venv/bin/python -m ruff format \
  src/train_b0.py tests/test_train_b0_kd_gen.py
2 files left unchanged

rtk proxy .venv/bin/python -m ruff check \
  src/train_b0.py tests/test_train_b0_kd.py tests/test_train_b0_kd_gen.py
All checks passed!

rtk proxy .venv/bin/python -m mypy \
  src/train_b0.py tests/test_train_b0_kd_gen.py tests/test_train_b0_kd.py
Success: no issues found in 3 source files

rtk proxy git diff --check
PASS
```

## Files changed

- `src/train_b0.py`
- `tests/test_train_b0_kd_gen.py`
- `tests/test_train_b0_kd.py`
- `.superpowers/sdd/2026-08-30-kd-gen-arm/task-4-report.md`

No Task 5 optimizer/stage logic, Task 7 loader-signature removal, model/scorer,
configs, general docs, plan/spec, or H20 state was changed.

## Self-review

- Confirmed the worktree was clean at exact base
  `9a150e4b5297528b2d45f86f0c5126c040dbf679` before edits.
- Confirmed the generator loss is weighted once in `KDRowBank.loss` and added
  once by the existing trainer path; the dedicated test checks the exact scalar.
- Confirmed ordinary KD tests remain green and the `global_relational` decision
  is unchanged, preserving row-exact DDP scaling.
- Confirmed validation normalizes from staged fp16 source data at use time and
  its generator diagnostic forward leaves the caller RNG state unchanged.
- Confirmed EDM sigma sums/counts are reduced before division, with empty
  quartiles omitted.
- Confirmed no compatibility shim or legacy seed path was introduced and the
  final diff contains only the four owned files.

## Concerns and staged boundaries

No blocking Task 4 concern remains. The four socket-based local DDP cases were
excluded exactly as authorized and were not claimed as freshly run. Task 5's
generator optimizer groups and warmup/joint-stage switch remain intentionally
unimplemented, so this report does not claim the later full staged-training
contract or any H20 execution.

---

## Fix round 1/5: enforce the production pre-DDP arm contract

### Important finding

The finding was valid: the `topo_gen` iff positive-`w_gen` guard existed only
inside `KDRowBank`, so production paths that built a model without a bank could
bypass it. In particular, `ddp-mode probe` prepared a `topo_gen` model without
the teacher-latent attachment or `gen_loss` path, leaving generator-core
parameters unused during warmup.

### TDD RED

Three focused tests were added first: production build with `topo_gen` but no
positive `w_gen`, production build with positive `w_gen` but no `topo_gen`, and
valid `kd_gen` probe rejection before `accelerator.prepare`.

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py \
  -n0 -q -k 'production_model_build or kd_gen_ddp_probe'
```

Exact result: exit 1, three failures.

```text
FAILED test_production_model_build_rejects_topo_gen_without_w_gen
E       Failed: DID NOT RAISE <class 'RuntimeError'>

FAILED test_production_model_build_rejects_w_gen_without_topo_gen
E       Failed: DID NOT RAISE <class 'RuntimeError'>

FAILED test_kd_gen_ddp_probe_rejects_before_accelerator_prepare
E       AssertionError: accelerator.prepare must not run for kd_gen probe mode
```

The third failure proves the old path crossed the preparation boundary.

### Implementation

- Added `_validate_topo_gen_distill_contract(model, distill)`, which unwraps a
  prepared model and requires `model.topo_gen` exactly when
  `distill.w_gen > 0`.
- Called the preflight from `build_model`, making it apply to every production
  model build before DDP preparation or optional bank construction.
- Reused the same preflight in `KDRowBank`; retained the bank-owned latent-width,
  finite positive fp64 RMS, and checkpoint-buffer stamping checks.
- Called the preflight at `_run_probe_mode` entry and rejected valid `kd_gen`
  with a clear instruction to use `epoch-probe` or `train`, before optimizer
  preparation. Those two supported modes continue through bank construction and
  teacher-latent attachment.
- Did not implement or modify any of the three Minor findings.

### GREEN and regression evidence

Focused bypass tests:

```text
rtk proxy .venv/bin/python -m pytest tests/test_train_b0_kd_gen.py \
  -n0 -q -k 'production_model_build or kd_gen_ddp_probe'
...                                                                      [100%]
```

Final Task 4 plus ordinary-KD non-DDP selection:

```text
rtk proxy .venv/bin/python -m pytest \
  tests/test_train_b0_kd_gen.py tests/test_train_b0_kd.py -n0 \
  -k 'not test_relational_kd_gathers_cross_rank_rows_with_exact_ddp_gradient and not test_production_kd_row_bank_keeps_global_loss_unscaled_and_one_pass'
42 passed, 4 deselected, 2 warnings in 1.47s
```

The two warnings remain the existing TorchScript deprecations.

### Formatting and static evidence

```text
rtk proxy .venv/bin/python -m ruff format \
  src/train_b0.py tests/test_train_b0_kd_gen.py
2 files left unchanged

rtk proxy .venv/bin/python -m ruff check \
  src/train_b0.py tests/test_train_b0_kd_gen.py tests/test_train_b0_kd.py
All checks passed!

rtk proxy .venv/bin/python -m mypy \
  src/train_b0.py tests/test_train_b0_kd_gen.py
Success: no issues found in 2 source files

rtk proxy git diff --check
PASS
```

### Files, self-review, and concerns

- Changed `src/train_b0.py`, `tests/test_train_b0_kd_gen.py`, and this report.
- Confirmed valid matched `kd_gen` builds successfully; only plain probe mode is
  rejected, and rejection occurs before `accelerator.prepare`.
- Confirmed the bank has one source of truth for the two-direction arm contract
  and still owns target-dependent latent validation/stamping.
- Confirmed ordinary KD and row-exact scaling code is untouched.
- No blocking concern remains. The four authorized socket-based DDP cases were
  excluded and are not claimed as freshly run; no H20 work was performed.
