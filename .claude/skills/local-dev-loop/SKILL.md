---
name: local-dev-loop
description: Use when writing or running tests, debugging a suite that passes locally but fails on the container, or fixing ruff/mypy complaints. Covers the synthetic-fixture rule, what silently skips, DDP smoke-test mechanics, and the lint/type config traps.
---

# Local dev loop

```bash
uv sync --group dev                                   # install into .venv
uv run pytest                                         # full suite
uv run pytest tests/eval/test_assembly.py             # one file
uv run pytest tests/test_g1_hardened_e2.py -k regime  # by name
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
```

Locally, prefer `.venv/bin/python -m ...` over `uv run` for the research CLIs —
the rtk proxy garbles `uv run` output.

## A green local run may have tested nothing

- `pythonpath = ["."]` plus `[tool.uv] package = false` means `src` is importable
  **only when pytest is invoked from the repo root**. Any other cwd gives
  `ModuleNotFoundError: src`.
- The only sanctioned real-data entrypoints are the session-scoped
  `benchmark_root` / `features_root` fixtures (`tests/conftest.py:20`, `:28`),
  which **silently `pytest.skip`** when `data/` is absent. On a bare checkout the
  real-data assertions never execute and the suite still reports green.
- The data root is overridable only via `TCIEP_DATA_ROOT`, read at *import* time
  (`tests/conftest.py:15`) — setting it inside a test or after collection does
  nothing.
- The whole DDP integration module is skipped on macOS by a module-level
  `pytestmark` (`tests/test_e2_ddp_integration.py:14`). `torch.distributed`
  contracts are only ever exercised on Linux, via `hpc/run.sh check`.
- The four-H20 cold-run acceptance test is further gated behind
  `RUN_E2_H20_ACCEPTANCE=1` with a 3700s timeout. Never set that off the container.

## Everything else must be synthetic

Do not reach for the 25 GB corpus in a new test. Use the existing builders:

- `_build_synthetic_benchmark` (`tests/test_train_b0.py:710`) — 7 nodes, writes
  `graph.pkl` + split + edge files, deliberately bypasses `verify_benchmark`.
- `_write_feature_store` (`tests/test_train_b0.py:768`)
- `_build_synthetic_package` (`tests/data/test_artifacts.py:87`)
- `_write_benchmark` (`tests/test_g1_hardened_e2.py:132`)

Real-data tests hardcode corpus-specific magic numbers that will fail loudly if
the artifact package ever changes: token shape `(123, 1536)`
(`tests/data/test_features.py:174`, also asserted by `hpc/run.sh:150`) and
`g_struct.number_of_nodes() == 8_072` (`tests/data/test_partition.py:89`).

## DDP smoke tests

Two ranks are launched for real via
`python -m torch.distributed.run --standalone --nproc_per_node=2` against
`tests/helpers/e2_ddp_smoke.py` and `tests/helpers/egostitch_ddp_smoke.py`.

- Both helpers set `ACCELERATE_USE_CPU=true` **before** importing accelerate
  (`e2_ddp_smoke.py:14`, `egostitch_ddp_smoke.py:27`). Moving that line below the
  import breaks the CPU path.
- `_low_thread_env` (`tests/test_e2_ddp_integration.py:20`) forces single-thread
  BLAS; without it each rank spawns a full OpenMP pool and the 60s timeout trips.
- Known hang: if a VPN or security client hijacks hostname resolution, the c10d
  TCPStore rendezvous dies with `DistNetworkError: Failed to recv, got 0 bytes`.
  Fix with `export PET_LOCAL_ADDR=localhost` (documented at
  `tests/test_e2_ddp_integration.py:31`).
- `egostitch_ddp_smoke.py:125` builds `S0Cache` via `__new__` and pokes `_logits`
  directly, so refactoring `S0Cache`'s private state breaks the DDP smoke test
  rather than the unit tests.

## Slow tests are not deselected

`addopts = ["-q", "-ra", "--strict-markers"]` has no default `-m` filter, so a
bare `pytest` runs the 300-node Watts-Strogatz perturbation check
(`tests/eval/test_composite.py:26`) and the 3-subprocess `PYTHONHASHSEED`
determinism test (`tests/eval/test_graph_metrics.py:367`). Only `unit`,
`integration`, and `slow` markers exist, and `--strict-markers` rejects others.
There is no `pytest-timeout` plugin — every timeout is a hand-rolled
`subprocess` timeout, so a hung rendezvous outside those calls hangs forever.

## Lint and type traps

- Ruff enables `ANN` + `D` + `T201` repo-wide. `tests/**` is exempt only from
  `D100-D104` and `ANN201` (`pyproject.toml:78`) — test helpers still need full
  argument annotations, and `print()` is banned.
- Mypy is `strict = true` with `ignore_missing_imports = true` **and**
  `warn_unused_ignores = true`. That combination means untyped third-party
  imports need an explicit `# type: ignore[import-untyped]` (e.g. `yaml`, see
  `tests/helpers/egostitch_ddp_smoke.py:33`).
- Never run two `mypy` invocations against the same `.mypy_cache` concurrently —
  it corrupts the cache and surfaces phantom `unused-ignore` errors. Re-run cold
  before believing them.
