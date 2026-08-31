# Task 2 report — V3_1 topo_gen integration

## Scope and implementation

Implemented only the assigned V3_1 and test surfaces in the `kd-gen-arm` worktree (base `cbdb5c6`). `V3_1` now imports and builds `TopoGenBase` through `build_topo_gen`, exposing `topo_gen` and `topo_gen_parameters()`. A supplied legacy `pair_latent_gen` config is explicitly rejected. The generator is built after the trunk and optional `kd_rep_head`, preserving trunk initialization RNG ordering. Its marginal path provides logits and generator telemetry; optional `kd_teacher_latent` provides `gen_loss` and family statistics. `pair_repr`, `kd_rep`, and the pre-existing label-loss path remain intact.

Removed the obsolete `PairLatentGenerator`, `_SeedRecognition`, generator constants, and private masked-token helper from `b0_v31.py`. Deleted the two legacy D9 test files, the legacy packed-scoring test, D9 guards, and the D9 shipped-config inventory item. The generic all-zero distillation tests remain, with only their obsolete D9 weights removed.

## RED evidence

After appending the four specified V3_1 tests, ran:

```bash
.venv/bin/python -m pytest tests/test_b0_topo_gen.py -n0 -q -k v31
```

Result: expected RED, `4 failed`.

- `pair_latent_gen` was accepted instead of rejected.
- `topo_gen` emitted none of the required forward outputs.
- `gen_loss` was absent for `kd_teacher_latent`.
- `topo_gen_parameters()` did not exist.

## GREEN and static evidence

```bash
.venv/bin/python -m pytest tests/test_b0_topo_gen.py -n0 -q -k v31
```

Result: `4 passed`.

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src/model/egostitch/classifier/b0_v31.py
git diff --check
```

Results: Ruff `All checks passed`; mypy `Success: no issues found in 1 source file`; `git diff --check` exited successfully with no output.

The required exact focused command was also run:

```bash
.venv/bin/python -m pytest tests/test_b0_topo_gen.py tests/test_train_b0_kd.py tests/test_score_universe.py -n0 -q
```

Result: not green, with five out-of-scope failures.

- `tests/test_score_universe.py::test_packed_scoring_caches_encoder_without_changing_logits` fails because unowned `src/score_universe.py` still dereferences the removed `model.pair_latent_gen` attribute in `_score_v3_1_packed`.
- Four existing DDP tests in `test_train_b0_kd.py` fail before their assertions because this local environment cannot resolve `127.0.0.1` to a local Gloo address. Retrying with `GLOO_SOCKET_IFNAME=lo` also fails because `lo` is not an available Gloo address here.

As a non-DDP control, the rest of `tests/test_score_universe.py` passed when the one legacy-attribute-dependent packed-scoring test was excluded.

## Changed and deleted files

- Modified: `src/model/egostitch/classifier/b0_v31.py`
- Modified: `tests/test_b0_topo_gen.py`
- Modified: `tests/test_score_universe.py`
- Modified: `tests/test_train_b0_kd.py`
- Modified: `tests/test_train_b0.py`
- Deleted: `tests/test_b0_pair_latent_gen.py`
- Deleted: `tests/test_train_b0_d9.py`

## Self-review

- Legacy configuration is rejected explicitly because V3_1 intentionally has no global unknown-key guard.
- `topo_gen` construction occurs after all trunk and optional-head construction; the seeded branch-zero control proves unchanged trunk logits at initialization.
- The new forward path never computes a separate trunk logit when `topo_gen` is active; `marginal_forward` is the single source of logits.
- Only the legacy D9/w_seed test portions were removed; no trainer, scorer, config, spec, or other production source was changed.

## Concerns

The integration deliberately has no compatibility shim, as required. The remaining packed scorer reference must be migrated in its owning task before the exact focused suite can be fully green. The four local DDP failures are environmental Gloo address-resolution failures, not test assertion failures.

## Scope-ruling update: packed scorer

The Task 2 scope was minimally expanded to `src/score_universe.py` after the
initial exact focused run demonstrated that deleting `V3_1.pair_latent_gen`
left `_score_v3_1_packed` crashing. The packed scorer now uses the same shared
path as `V3_1.forward`: the plain case calls `model.output_head(pair_repr)`;
the topology-generator case calls `model.topo_gen.marginal_forward(encoded_a,
encoded_b, len_a, len_b, pair_repr, model.output_head)["logits"]`. No
`pair_latent_gen` compatibility access or CLI control was added.

The preceding packed-scorer failure is retained as RED evidence and is
superseded by the focused GREEN evidence recorded after this update.

### GREEN evidence after scope ruling

```bash
.venv/bin/python -m pytest tests/test_b0_topo_gen.py tests/test_train_b0_kd.py tests/test_score_universe.py -n0 -q -k "not relational_kd_gathers_cross_rank_rows_with_exact_ddp_gradient and not production_kd_row_bank_keeps_global_loss_unscaled_and_one_pass"
```

Result: passed (exit code 0). All non-DDP cases in the required focused suite
passed; only the four explicitly excluded local macOS Gloo DDP cases were not
run.

```bash
.venv/bin/python -m ruff check src/model/egostitch/classifier/b0_v31.py src/score_universe.py tests/test_b0_topo_gen.py tests/test_score_universe.py tests/test_train_b0_kd.py tests/test_train_b0.py
.venv/bin/python -m mypy src/model/egostitch/classifier/b0_v31.py src/score_universe.py
git diff --check
```

Results: Ruff `All checks passed`; mypy `Success: no issues found in 2 source
files`; `git diff --check` exited successfully with no output.
