# KD2 Strict-LLP Optuna HPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An unattended container-side Optuna driver that jointly searches the strict-LLP `kd_rank` loss weights, context-bank composition, and margin over 16 trials, plus the context-dump CLI flags the new banks need.

**Architecture:** One new driver module (`src/experiments/kd_rank_strict_hpo.py`) runs an ask-and-tell TPE loop over a sqlite-backed study; each trial materializes a YAML config from `configs/autoresearch/kd_rank.yaml` and subprocesses `bash hpc/run.sh train <cfg> --skip-test`, then reads the cadence-2 surface via `src.autoresearch.metrics_io.read_run`. Three sampler flags added to `python -m src.distill.teacher_targets` let the driver dump the three missing context banks through `hpc/run.sh kd-targets`.

**Tech Stack:** Python 3.12, optuna 4.7.0 (already pinned in `pyproject.toml`), PyYAML, numpy, existing repo modules (`src.distill.config.DistillConfig`, `src.autoresearch.metrics_io`).

**Spec:** `docs/superpowers/specs/2026-09-01-kd-rank-strict-llp-optuna-hpo-design.md`

## Global Constraints

- Lint/type gates before every commit: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests` and `.venv/bin/python -m mypy src tests` (strict; optuna ships type stubs — wrap `suggest_categorical` results in `float()`/`str()`).
- Run single test files with `-n0`: `.venv/bin/python -m pytest tests/<file> -n0 -v`. Never use `--dist load`.
- All new tests are CPU-only: no GPU, no teacher checkpoints, no real `outputs/` paths — monkeypatch bank paths and command seams into `tmp_path`.
- No sha256/digest pinning or artifact-contract verifiers in new code paths beyond what existing functions already do. Fail closed only on non-finite state, missing sentinels, and I/O failures.
- Objectives: maximize BFS-macro GS, minimize geometric mean of the three MMD ratios; RD soft constraint `|log RD| - rd_band <= 0`, `rd_band` default `0.05`.
- Budget/search space (verbatim from spec): 16 trials; `w_rank` log-float [0.01, 1]; `w_dist` log-float [0.1, 100]; bank categorical {h2ns1, h2ns3, h2ns5, h3ns3}; margin categorical {0.05, 0.1, 0.2}; sampler `TPESampler(multivariate=True, seed=0, n_startup_trials=6, constraints_func=...)`; study name `kd_rank_strict_llp`.
- Per-trial config: only `distill.w_rank`, `distill.w_dist`, `distill.margin`, `distill.context_targets_path`, `output_dir` may differ from the base config.
- Do not touch `autoresearch/program.md`, `src/autoresearch/`, `configs/sweep/`, or any frozen config key.
- Commit after each task with the message given in the task.

---

### Task 1: Context-dump sampler flags

**Files:**
- Modify: `src/distill/teacher_targets.py` (parser ~line 697, `main` contexts branch ~line 822, `_finalize_context_artifact` ~line 758)
- Test: `tests/distill/test_teacher_targets.py` (append new tests)

**Interfaces:**
- Consumes: `DEFAULT_RW_STEP`/`DEFAULT_HOPS`/`DEFAULT_NS_RATE` and the `rw_step`/`hops`/`ns_rate` keyword parameters of `sample_context_banks` / `sample_v_val_context_bank` (`src/distill/context_sampler.py:76-78`, already exist).
- Produces: CLI flags `--rw-step`, `--hops`, `--ns-rate` (int, defaults 3/2/1) on `python -m src.distill.teacher_targets`, honored in `--contexts` mode (shard and merge invocations alike) and recorded in the artifact's `sampler_params`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/distill/test_teacher_targets.py`:

```python
def test_parser_sampler_flags_default_to_reference_values() -> None:
    args = teacher_targets.build_parser().parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pt", "--output", "out"]
    )
    assert (args.rw_step, args.hops, args.ns_rate) == (3, 2, 1)


def test_parser_sampler_flags_accept_overrides() -> None:
    args = teacher_targets.build_parser().parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pt", "--output", "out",
         "--contexts", "--rw-step", "3", "--hops", "3", "--ns-rate", "5"]
    )
    assert (args.rw_step, args.hops, args.ns_rate) == (3, 3, 5)


def test_finalize_context_artifact_records_cli_sampler_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"fixture")
    captured: dict[str, object] = {}

    def fake_write(output: Path, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(teacher_targets, "write_kd_context_targets", fake_write)
    args = teacher_targets.build_parser().parse_args(
        ["--config", "cfg.yaml", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "o"),
         "--contexts", "--rw-step", "3", "--hops", "3", "--ns-rate", "5"]
    )
    graph = nx.Graph([("a", "b")])
    teacher_targets._finalize_context_artifact(
        args,
        ["a", "b"],
        graph,
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.float32),
        [],
        None,
        "cafe0000",
    )
    assert captured["sampler_params"] == {"rw_step": 3, "hops": 3, "ns_rate": 5}
```

Match the existing file's import style (`from src.distill import teacher_targets` vs direct symbol imports) — read the top of the file first and reuse whatever alias it already has; add `import networkx as nx`, `import numpy as np`, `import pytest`, `from pathlib import Path` only if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/distill/test_teacher_targets.py -n0 -k "sampler_flags or records_cli" -v`
Expected: FAIL — `argparse` errors on unknown `--rw-step` and `sampler_params == {"rw_step": 3, "hops": 2, ...}` mismatch.

- [ ] **Step 3: Implement**

In `build_parser()` (after the `--contexts` argument):

```python
    parser.add_argument("--rw-step", type=int, default=DEFAULT_RW_STEP)
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS)
    parser.add_argument("--ns-rate", type=int, default=DEFAULT_NS_RATE)
```

In `main()`, thread the flags into both sampling calls in the `args.contexts` branch:

```python
        banks = sample_context_banks(
            truth_graph,
            anchor_ids=node_ids,
            node_ids=node_ids,
            forbidden_internal=split.v_val,
            seed=_CONTEXT_SEED,
            n_banks=cfg.optim.epochs,
            rw_step=args.rw_step,
            hops=args.hops,
            ns_rate=args.ns_rate,
        )
        val_bank = sample_v_val_context_bank(
            truth_graph,
            v_val=split.v_val,
            node_ids=node_ids,
            rw_step=args.rw_step,
            hops=args.hops,
            ns_rate=args.ns_rate,
        )
```

In `_finalize_context_artifact`, replace the constant dict:

```python
        sampler_params={
            "rw_step": args.rw_step,
            "hops": args.hops,
            "ns_rate": args.ns_rate,
        },
```

Docstring/help: the `--contexts` help line already says "strict-LLP context targets"; extend the three new arguments with `help="Context-sampler override; shard and merge invocations of one dump must pass identical values"`.

- [ ] **Step 4: Run tests to verify they pass, plus the file's existing tests**

Run: `.venv/bin/python -m pytest tests/distill/test_teacher_targets.py tests/distill/test_context_sampler.py -n0`
Expected: PASS (existing dump tests keep passing because defaults are unchanged).

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/distill/teacher_targets.py tests/distill/test_teacher_targets.py
git commit -m "feat(kd2): sampler flags on the context-target dump CLI"
```

---

### Task 2: Driver scaffold — bank registry, priors, config materialization

**Files:**
- Create: `src/experiments/kd_rank_strict_hpo.py`
- Test: `tests/experiments/test_kd_rank_strict_hpo.py` (create)

**Interfaces:**
- Consumes: `src.distill.config.DistillConfig.from_mapping(mapping) -> DistillConfig` (raises `ValueError` on illegal weight patterns).
- Produces (used by Tasks 3–6):
  - `BankSpec(rw_step: int, hops: int, ns_rate: int, path: str)` frozen dataclass.
  - `BANKS: dict[str, BankSpec]` with keys `h2ns1|h2ns3|h2ns5|h3ns3`.
  - `ENQUEUED_PRIORS: tuple[dict[str, object], ...]` (6 entries).
  - `materialize_trial_config(base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/test_kd_rank_strict_hpo.py`:

```python
"""CPU-only tests for the strict-LLP kd_rank Optuna sweep driver."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.experiments import kd_rank_strict_hpo as hpo

BASE_CONFIG = Path("configs/autoresearch/kd_rank.yaml")


def test_bank_registry_matches_spec() -> None:
    assert set(hpo.BANKS) == {"h2ns1", "h2ns3", "h2ns5", "h3ns3"}
    assert hpo.BANKS["h2ns1"].path == "outputs/distill/kd_ctx_targets_breadth_first"
    assert (hpo.BANKS["h3ns3"].rw_step, hpo.BANKS["h3ns3"].hops, hpo.BANKS["h3ns3"].ns_rate) == (3, 3, 3)


def test_enqueued_priors_match_spec() -> None:
    assert len(hpo.ENQUEUED_PRIORS) == 6
    assert hpo.ENQUEUED_PRIORS[0] == {"w_rank": 1.0, "w_dist": 1.0, "bank": "h2ns1", "margin": 0.1}
    assert hpo.ENQUEUED_PRIORS[4] == {"w_rank": 0.1, "w_dist": 100.0, "bank": "h2ns3", "margin": 0.1}


def test_materialize_changes_only_whitelisted_keys(tmp_path: Path) -> None:
    params = {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns3", "margin": 0.05}
    config_path = hpo.materialize_trial_config(BASE_CONFIG, params, 7, tmp_path)
    assert config_path == tmp_path / "configs" / "trial_007.yaml"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    trial = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert trial["output_dir"] == str(tmp_path / "trial_007")
    assert trial["distill"]["w_rank"] == 0.1
    assert trial["distill"]["w_dist"] == 10.0
    assert trial["distill"]["margin"] == 0.05
    assert trial["distill"]["context_targets_path"] == hpo.BANKS["h2ns3"].path
    trial["output_dir"] = base["output_dir"]
    trial["distill"] = base["distill"]
    assert trial == base


def test_materialize_rejects_unknown_bank(tmp_path: Path) -> None:
    params = {"w_rank": 0.1, "w_dist": 10.0, "bank": "h9ns9", "margin": 0.1}
    with pytest.raises(KeyError):
        hpo.materialize_trial_config(BASE_CONFIG, params, 1, tmp_path)


def test_materialize_rejects_illegal_weight_pattern(tmp_path: Path) -> None:
    params = {"w_rank": 0.0, "w_dist": 0.0, "bank": "h2ns1", "margin": 0.1}
    with pytest.raises(ValueError):
        hpo.materialize_trial_config(BASE_CONFIG, params, 1, tmp_path)
```

(The last test works because zero weights with a nonempty `context_targets_path` violate `DistillConfig`'s "context_targets_path only valid when kd_rank is active" rule — `src/distill/config.py:114-115`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: FAIL with `ModuleNotFoundError: src.experiments.kd_rank_strict_hpo`.

- [ ] **Step 3: Implement the module scaffold**

Create `src/experiments/kd_rank_strict_hpo.py`:

```python
"""Unattended Optuna sweep for the strict-LLP ``kd_rank`` arm.

Runs on the H20 container: an ask-and-tell TPE loop proposes
``(w_rank, w_dist, context bank, margin)``, launches one grid-protocol
training per trial through ``hpc/run.sh train --skip-test``, and scores the
cadence-2 V_val surface as (GS max, geometric-mean MMD ratio min) with an
``|log RD|`` soft constraint. The feasible Pareto front is advisory: the
recorded winner comes from the frozen five-metric undominated verdict.
Spec: ``docs/superpowers/specs/2026-09-01-kd-rank-strict-llp-optuna-hpo-design.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.distill.config import DistillConfig


@dataclass(frozen=True)
class BankSpec:
    """One frozen context bank: sampler composition and artifact path."""

    rw_step: int
    hops: int
    ns_rate: int
    path: str


BANKS: dict[str, BankSpec] = {
    "h2ns1": BankSpec(3, 2, 1, "outputs/distill/kd_ctx_targets_breadth_first"),
    "h2ns3": BankSpec(3, 2, 3, "outputs/distill/kd_ctx_targets_breadth_first_h2ns3"),
    "h2ns5": BankSpec(3, 2, 5, "outputs/distill/kd_ctx_targets_breadth_first_h2ns5"),
    "h3ns3": BankSpec(3, 3, 3, "outputs/distill/kd_ctx_targets_breadth_first_h3ns3"),
}

ENQUEUED_PRIORS: tuple[dict[str, object], ...] = (
    {"w_rank": 1.0, "w_dist": 1.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns1", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.01, "w_dist": 10.0, "bank": "h2ns5", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 100.0, "bank": "h2ns3", "margin": 0.1},
    {"w_rank": 0.1, "w_dist": 10.0, "bank": "h3ns3", "margin": 0.1},
)


def materialize_trial_config(
    base_config: Path, params: Mapping[str, object], trial_number: int, sweep_dir: Path
) -> Path:
    """Write trial ``trial_number``'s config; only the five whitelisted keys differ.

    Raises:
        KeyError: On an unknown bank name.
        ValueError: If the resulting ``distill`` section is illegal.
    """
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(sweep_dir / f"trial_{trial_number:03d}")
    distill = dict(cfg["distill"])
    distill["w_rank"] = float(params["w_rank"])  # type: ignore[arg-type]
    distill["w_dist"] = float(params["w_dist"])  # type: ignore[arg-type]
    distill["margin"] = float(params["margin"])  # type: ignore[arg-type]
    distill["context_targets_path"] = BANKS[str(params["bank"])].path
    cfg["distill"] = distill
    DistillConfig.from_mapping(distill)
    config_path = sweep_dir / "configs" / f"trial_{trial_number:03d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path
```

(`argparse` import is used in Task 6; if ruff flags it as unused now, add it in Task 6 instead.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git commit -m "feat(kd2): strict-LLP sweep scaffold — banks, priors, trial configs"
```

---

### Task 3: Trial outcome extraction

**Files:**
- Modify: `src/experiments/kd_rank_strict_hpo.py`
- Test: `tests/experiments/test_kd_rank_strict_hpo.py`

**Interfaces:**
- Consumes: `src.autoresearch.metrics_io.read_run(run_dir: Path, topology_every: int | None) -> RunMetrics` (raises `RunFailure` on `failure.json`, `ValueError` on missing/non-finite state); `RunMetrics.topology` is `TopologyValidationMetrics` with fields `gs, rd, degree_mmd, clustering_mmd, spectral_mmd`; `RunMetrics.auprc`, `.selected_epoch`.
- Produces: `TrialOutcome(gs: float, geo_mmd: float, constraint: float, surface: dict[str, float])` frozen dataclass and `trial_outcome(run_dir: Path, rd_band: float) -> TrialOutcome`. `surface` holds JSON-serializable floats: `auprc, gs, rd, degree_mmd, clustering_mmd, spectral_mmd, selected_epoch`.

- [ ] **Step 1: Write the failing tests**

Append to the test file (imports: `import math`, `import json`, `from tests.autoresearch.conftest import make_cadence_rows`, `from src.autoresearch.metrics_io import RunFailure`):

```python
def _publish_run(run_dir: Path, rows: list[dict[str, object]], *, failure: bool = False) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"selected_epoch": 2, "arm": "kd_rank", "config_hash": "d", "checkpoint_id": "c"}),
        encoding="utf-8",
    )
    (run_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "fixture", "total_seconds": 60.0}),
        encoding="utf-8",
    )
    if failure:
        (run_dir / "failure.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    return run_dir


def test_trial_outcome_reads_cadence2_surface(tmp_path: Path) -> None:
    run_dir = _publish_run(tmp_path / "trial_000", make_cadence_rows())
    outcome = hpo.trial_outcome(run_dir, rd_band=0.05)
    # make_cadence_rows: epoch 4 dominates the cadence-2 due set (gs .80, rd 1.02, mmds .60).
    assert outcome.gs == pytest.approx(0.80)
    assert outcome.geo_mmd == pytest.approx(0.60)
    assert outcome.constraint == pytest.approx(abs(math.log(1.02)) - 0.05)
    assert outcome.constraint < 0.0
    assert outcome.surface["selected_epoch"] == 4.0


def test_trial_outcome_flags_rd_outside_band(tmp_path: Path) -> None:
    rows = make_cadence_rows()
    rows[3]["val_rd_bfs"] = 1.20
    run_dir = _publish_run(tmp_path / "trial_001", rows)
    assert hpo.trial_outcome(run_dir, rd_band=0.05).constraint > 0.0


def test_trial_outcome_propagates_run_failure(tmp_path: Path) -> None:
    run_dir = _publish_run(tmp_path / "trial_002", make_cadence_rows(), failure=True)
    with pytest.raises(RunFailure):
        hpo.trial_outcome(run_dir, rd_band=0.05)


def test_trial_outcome_rejects_nonpositive_mmd_ratio(tmp_path: Path) -> None:
    rows = make_cadence_rows()
    rows[3]["val_degree_mmd_ratio"] = 0.0
    run_dir = _publish_run(tmp_path / "trial_003", rows)
    with pytest.raises(ValueError):
        hpo.trial_outcome(run_dir, rd_band=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -k trial_outcome -v`
Expected: FAIL with `AttributeError: ... has no attribute 'trial_outcome'`.

- [ ] **Step 3: Implement**

Add to the driver module (new imports: `import math`, `from src.autoresearch.metrics_io import RunMetrics, read_run`):

```python
@dataclass(frozen=True)
class TrialOutcome:
    """Objectives, constraint, and telemetry surface of one completed trial."""

    gs: float
    geo_mmd: float
    constraint: float
    surface: dict[str, float]


def trial_outcome(run_dir: Path, rd_band: float) -> TrialOutcome:
    """Score one run at its cadence-2 selected epoch.

    Raises:
        RunFailure: If the run wrote ``failure.json``.
        ValueError: On missing/non-finite metrics or a non-positive MMD ratio.
    """
    run: RunMetrics = read_run(run_dir, topology_every=2)
    topo = run.topology
    ratios = (topo.degree_mmd, topo.clustering_mmd, topo.spectral_mmd)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"{run_dir}: MMD ratios must be positive, got {ratios}")
    geo_mmd = math.exp(sum(math.log(ratio) for ratio in ratios) / 3.0)
    surface = {
        "auprc": run.auprc,
        "gs": topo.gs,
        "rd": topo.rd,
        "degree_mmd": topo.degree_mmd,
        "clustering_mmd": topo.clustering_mmd,
        "spectral_mmd": topo.spectral_mmd,
        "selected_epoch": float(run.selected_epoch),
    }
    return TrialOutcome(topo.gs, geo_mmd, abs(math.log(topo.rd)) - rd_band, surface)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git commit -m "feat(kd2): sweep trial outcome — objectives, RD constraint, surface"
```

---

### Task 4: Study construction and search space

**Files:**
- Modify: `src/experiments/kd_rank_strict_hpo.py`
- Test: `tests/experiments/test_kd_rank_strict_hpo.py`

**Interfaces:**
- Consumes: `BANKS`, `ENQUEUED_PRIORS` (Task 2).
- Produces: `build_study(db_path: Path) -> optuna.Study` (name `kd_rank_strict_llp`, directions `["maximize", "minimize"]`, sqlite storage, priors enqueued idempotently) and `suggest_params(trial: optuna.Trial) -> dict[str, object]`. Constraint convention: completed trials carry `user_attrs["constraint"] == [float]`; `_constraints` returns `(inf,)` when absent (conservative infeasible).

- [ ] **Step 1: Write the failing tests**

Append (imports: `import optuna`, `from optuna.trial import TrialState`):

```python
def test_build_study_directions_and_priors(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    assert study.study_name == "kd_rank_strict_llp"
    assert [d.name.lower() for d in study.directions] == ["maximize", "minimize"]
    assert len(study.get_trials(deepcopy=False)) == 6


def test_build_study_enqueue_is_idempotent(tmp_path: Path) -> None:
    hpo.build_study(tmp_path / "optuna.db")
    study = hpo.build_study(tmp_path / "optuna.db")
    assert len(study.get_trials(deepcopy=False)) == 6


def test_suggest_params_consumes_priors_in_order(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    first = hpo.suggest_params(study.ask())
    second = hpo.suggest_params(study.ask())
    assert first == hpo.ENQUEUED_PRIORS[0]
    assert second == hpo.ENQUEUED_PRIORS[1]


def test_constraints_default_to_infeasible() -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.add_trial(
        optuna.trial.create_trial(
            params={}, distributions={}, values=[0.5, 1.0], user_attrs={}
        )
    )
    assert hpo._constraints(study.get_trials(deepcopy=False)[0]) == (float("inf"),)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -k "build_study or suggest_params or constraints_default" -v`
Expected: FAIL with `AttributeError: ... 'build_study'`.

- [ ] **Step 3: Implement**

Add (new imports: `import optuna`, `from collections.abc import Sequence`; `optuna.logging.set_verbosity(optuna.logging.WARNING)` is NOT set — keep default logging):

```python
STUDY_NAME = "kd_rank_strict_llp"
N_STARTUP_TRIALS = 6


def _constraints(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
    constraint = trial.user_attrs.get("constraint")
    if not isinstance(constraint, list) or len(constraint) != 1:
        return (float("inf"),)
    return (float(constraint[0]),)


def build_study(db_path: Path) -> optuna.Study:
    """Create-or-load the sweep study and (re-)enqueue the prior trials."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=0,
        multivariate=True,
        n_startup_trials=N_STARTUP_TRIALS,
        constraints_func=_constraints,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db_path}",
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )
    for params in ENQUEUED_PRIORS:
        study.enqueue_trial(dict(params), skip_if_exists=True)
    return study


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Draw one point of the spec's search space (enqueued values pass through)."""
    return {
        "w_rank": float(trial.suggest_float("w_rank", 0.01, 1.0, log=True)),
        "w_dist": float(trial.suggest_float("w_dist", 0.1, 100.0, log=True)),
        "bank": str(trial.suggest_categorical("bank", sorted(BANKS))),
        "margin": float(trial.suggest_categorical("margin", [0.05, 0.1, 0.2])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: PASS. If `suggest_categorical([0.05, 0.1, 0.2])` upsets mypy, annotate the choice list as `list[float]` or cast the return; do not change the values.

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git commit -m "feat(kd2): sweep study — constrained MO-TPE, prior enqueue"
```

---

### Task 5: Startup reconciliation of interrupted trials

**Files:**
- Modify: `src/experiments/kd_rank_strict_hpo.py`
- Test: `tests/experiments/test_kd_rank_strict_hpo.py`

**Interfaces:**
- Consumes: `build_study`, `trial_outcome`, `_publish_run` test helper.
- Produces: `reconcile_running(study: optuna.Study, sweep_dir: Path, rd_band: float) -> None`. Contract: every stale RUNNING trial is told FAIL; when its run dir has `complete.json`, a COMPLETE twin trial with the same params, real values, and `user_attrs["constraint"]`/`user_attrs["surface"]` is `add_trial`-ed (public API — a FrozenTrial cannot receive user attrs after `tell`).

- [ ] **Step 1: Write the failing tests**

```python
def _ask_running_trial(study: optuna.Study) -> optuna.Trial:
    trial = study.ask()
    hpo.suggest_params(trial)
    return trial


def test_reconcile_completed_run_is_retold_with_values(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    trial = _ask_running_trial(study)
    _publish_run(tmp_path / f"trial_{trial.number:03d}", make_cadence_rows())
    hpo.reconcile_running(study, tmp_path, rd_band=0.05)
    trials = study.get_trials(deepcopy=False)
    assert [t.state for t in trials if t.number == trial.number] == [TrialState.FAIL]
    twins = [t for t in trials if t.state == TrialState.COMPLETE]
    assert len(twins) == 1
    assert twins[0].params == dict(hpo.ENQUEUED_PRIORS[0])
    assert twins[0].values == pytest.approx([0.80, 0.60])
    assert twins[0].user_attrs["constraint"][0] < 0.0
    assert twins[0].user_attrs["surface"]["selected_epoch"] == 4.0


def test_reconcile_failed_and_vanished_runs_are_failed(tmp_path: Path) -> None:
    study = hpo.build_study(tmp_path / "optuna.db")
    failed = _ask_running_trial(study)
    _publish_run(tmp_path / f"trial_{failed.number:03d}", make_cadence_rows(), failure=True)
    vanished = _ask_running_trial(study)
    hpo.reconcile_running(study, tmp_path, rd_band=0.05)
    states = {t.number: t.state for t in study.get_trials(deepcopy=False)}
    assert states[failed.number] == TrialState.FAIL
    assert states[vanished.number] == TrialState.FAIL
    assert TrialState.COMPLETE not in states.values()
```

(`read_run` raises `RunFailure` when `failure.json` exists even next to a `complete.json`, so the failed case needs no special ordering.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -k reconcile -v`
Expected: FAIL with `AttributeError: ... 'reconcile_running'`.

- [ ] **Step 3: Implement**

Add (new import: `from optuna.trial import TrialState`; also `from src.autoresearch.metrics_io import RunFailure` — extend the existing import line):

```python
def reconcile_running(study: optuna.Study, sweep_dir: Path, rd_band: float) -> None:
    """Resolve trials left RUNNING by an interrupted driver.

    The stale trial is always failed; a run that actually completed is
    re-added as a COMPLETE twin with its real objectives and constraint.
    """
    for stale in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        run_dir = sweep_dir / f"trial_{stale.number:03d}"
        twin: optuna.trial.FrozenTrial | None = None
        if (run_dir / "complete.json").exists():
            try:
                outcome = trial_outcome(run_dir, rd_band)
            except RunFailure:
                outcome = None
            if outcome is not None:
                twin = optuna.trial.create_trial(
                    params=dict(stale.params),
                    distributions=dict(stale.distributions),
                    values=[outcome.gs, outcome.geo_mmd],
                    user_attrs={"constraint": [outcome.constraint], "surface": outcome.surface},
                )
        study.tell(stale.number, state=TrialState.FAIL)
        if twin is not None:
            study.add_trial(twin)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git commit -m "feat(kd2): sweep resume — reconcile interrupted trials"
```

---

### Task 6: Sweep loop, bank dumps, CLI, report

**Files:**
- Modify: `src/experiments/kd_rank_strict_hpo.py`
- Test: `tests/experiments/test_kd_rank_strict_hpo.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_command(cmd: list[str]) -> int` and `run_commands_parallel(commands: list[tuple[list[str], dict[str, str]]]) -> list[int]` (test seams), `dump_missing_banks(args: argparse.Namespace) -> None`, `run_sweep(args: argparse.Namespace) -> None`, `print_report(study: optuna.Study) -> None`, `build_parser() -> argparse.ArgumentParser`, `main(argv: Sequence[str] | None = None) -> None`, and a `python -m src.experiments.kd_rank_strict_hpo` entrypoint.

- [ ] **Step 1: Write the failing tests**

Append (new test-file import: `import argparse`):

```python
def _fabricate_run(config_path: Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _publish_run(Path(cfg["output_dir"]), make_cadence_rows())


def _sweep_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    argv = [
        "--teacher-checkpoint", str(tmp_path / "teacher.pt"),
        "--sweep-dir", str(tmp_path),
        "--n-trials", "2",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return hpo.build_parser().parse_args(argv)


def test_run_sweep_completes_n_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, spec in list(hpo.BANKS.items()):  # pre-existing banks: no dumps expected
        bank = tmp_path / Path(spec.path).name
        bank.mkdir()
        monkeypatch.setitem(
            hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank))
        )
    launched: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        launched.append(cmd)
        assert cmd[:3] == ["bash", "hpc/run.sh", "train"]
        assert cmd[-1] == "--skip-test"
        _fabricate_run(Path(cmd[3]))
        return 0

    monkeypatch.setattr(hpo, "run_command", fake_run)
    monkeypatch.setattr(
        hpo, "run_commands_parallel", lambda commands: pytest.fail("no dumps expected")
    )
    hpo.run_sweep(_sweep_args(tmp_path))
    study = hpo.build_study(tmp_path / "optuna.db")
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    assert len(complete) == 2
    assert [t.params for t in complete] == [dict(p) for p in hpo.ENQUEUED_PRIORS[:2]]
    assert all(t.user_attrs["surface"]["gs"] == pytest.approx(0.80) for t in complete)
    assert len(launched) == 2


def test_run_sweep_marks_failed_run_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, spec in list(hpo.BANKS.items()):
        bank = tmp_path / Path(spec.path).name
        bank.mkdir()
        monkeypatch.setitem(hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(bank)))
    calls = {"n": 0}

    def fake_run(cmd: list[str]) -> int:
        calls["n"] += 1
        cfg = yaml.safe_load(Path(cmd[3]).read_text(encoding="utf-8"))
        _publish_run(Path(cfg["output_dir"]), make_cadence_rows(), failure=calls["n"] == 1)
        return 0

    monkeypatch.setattr(hpo, "run_command", fake_run)
    monkeypatch.setattr(hpo, "run_commands_parallel", lambda commands: [])
    hpo.run_sweep(_sweep_args(tmp_path, n_trials=1))
    states = [t.state for t in hpo.build_study(tmp_path / "optuna.db").get_trials(deepcopy=False)]
    assert states.count(TrialState.FAIL) == 1
    assert states.count(TrialState.COMPLETE) == 1


def test_dump_missing_banks_shards_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "kd_ctx_targets_breadth_first"
    existing.mkdir()
    monkeypatch.setitem(hpo.BANKS, "h2ns1", hpo.BankSpec(3, 2, 1, str(existing)))
    monkeypatch.setitem(hpo.BANKS, "h2ns3", hpo.BankSpec(3, 2, 3, str(tmp_path / "b_h2ns3")))
    monkeypatch.setitem(hpo.BANKS, "h2ns5", hpo.BankSpec(3, 2, 5, str(existing)))
    monkeypatch.setitem(hpo.BANKS, "h3ns3", hpo.BankSpec(3, 3, 3, str(existing)))
    parallel_calls: list[list[tuple[list[str], dict[str, str]]]] = []
    merges: list[list[str]] = []
    monkeypatch.setattr(
        hpo, "run_commands_parallel",
        lambda commands: (parallel_calls.append(commands), [0] * len(commands))[1],
    )
    monkeypatch.setattr(hpo, "run_command", lambda cmd: (merges.append(cmd), 0)[1])
    hpo.dump_missing_banks(_sweep_args(tmp_path, dump_shards=2))
    assert len(parallel_calls) == 1 and len(parallel_calls[0]) == 2
    shard_cmd, shard_env = parallel_calls[0][0]
    assert shard_cmd[:3] == ["bash", "hpc/run.sh", "kd-targets"]
    assert "--contexts" in shard_cmd and "--row-shard" in shard_cmd
    for flag, value in (("--rw-step", "3"), ("--hops", "2"), ("--ns-rate", "3")):
        assert shard_cmd[shard_cmd.index(flag) + 1] == value
    assert shard_env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert len(merges) == 1 and "--merge" in merges[0]


def test_dump_missing_banks_raises_on_shard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(hpo.BANKS, "h2ns3", hpo.BankSpec(3, 2, 3, str(tmp_path / "missing")))
    for name in ("h2ns1", "h2ns5", "h3ns3"):
        spec = hpo.BANKS[name]
        existing = tmp_path / f"bank_{name}"
        existing.mkdir()
        monkeypatch.setitem(hpo.BANKS, name, hpo.BankSpec(spec.rw_step, spec.hops, spec.ns_rate, str(existing)))
    monkeypatch.setattr(hpo, "run_commands_parallel", lambda commands: [0, 1])
    with pytest.raises(RuntimeError):
        hpo.dump_missing_banks(_sweep_args(tmp_path, dump_shards=2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -k "run_sweep or dump_missing" -v`
Expected: FAIL with `AttributeError: ... 'build_parser'`.

- [ ] **Step 3: Implement**

Add (new imports: `import os`, `import subprocess`):

```python
_THREAD_CAPS = {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}


def run_command(cmd: list[str]) -> int:
    """Run one foreground container command with the H20 thread caps."""
    return subprocess.run(cmd, env={**os.environ, **_THREAD_CAPS}, check=False).returncode


def run_commands_parallel(commands: list[tuple[list[str], dict[str, str]]]) -> list[int]:
    """Run commands concurrently; each tuple is (argv, extra env)."""
    procs = [
        subprocess.Popen(cmd, env={**os.environ, **_THREAD_CAPS, **extra})
        for cmd, extra in commands
    ]
    return [proc.wait() for proc in procs]


def _dump_cmd(args: argparse.Namespace, spec: BankSpec) -> list[str]:
    return [
        "bash", "hpc/run.sh", "kd-targets", "--contexts",
        "--config", str(args.base_config),
        "--checkpoint", str(args.teacher_checkpoint),
        "--output", spec.path,
        "--rw-step", str(spec.rw_step),
        "--hops", str(spec.hops),
        "--ns-rate", str(spec.ns_rate),
    ]


def dump_missing_banks(args: argparse.Namespace) -> None:
    """Dump every context bank whose artifact is absent (sharded, then merged).

    Raises:
        RuntimeError: If any shard or merge exits nonzero (fail-closed
            before any training budget is spent).
    """
    for name in sorted(BANKS):
        spec = BANKS[name]
        if Path(spec.path).exists():
            continue
        shards = [
            (
                _dump_cmd(args, spec)
                + ["--device", "cuda", "--row-shard", f"{index}/{args.dump_shards}"],
                {"CUDA_VISIBLE_DEVICES": str(index)},
            )
            for index in range(args.dump_shards)
        ]
        codes = run_commands_parallel(shards)
        if any(code != 0 for code in codes):
            raise RuntimeError(f"bank {name}: shard exit codes {codes}")
        merge_code = run_command(
            _dump_cmd(args, spec) + ["--merge", "--row-shard", f"0/{args.dump_shards}"]
        )
        if merge_code != 0:
            raise RuntimeError(f"bank {name}: merge exited {merge_code}")


def _n_complete(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


def run_sweep(args: argparse.Namespace) -> None:
    """Drive the whole sweep: reconcile, dump banks, ask/tell until budget."""
    study = build_study(args.sweep_dir / "optuna.db")
    reconcile_running(study, args.sweep_dir, args.rd_band)
    dump_missing_banks(args)
    while _n_complete(study) < args.n_trials:
        trial = study.ask()
        params = suggest_params(trial)
        config_path = materialize_trial_config(
            args.base_config, params, trial.number, args.sweep_dir
        )
        run_command(["bash", "hpc/run.sh", "train", str(config_path), "--skip-test"])
        run_dir = args.sweep_dir / f"trial_{trial.number:03d}"
        try:
            outcome = trial_outcome(run_dir, args.rd_band)
        except RunFailure:
            study.tell(trial, state=TrialState.FAIL)
            continue
        trial.set_user_attr("constraint", [outcome.constraint])
        trial.set_user_attr("surface", outcome.surface)
        study.tell(trial, values=[outcome.gs, outcome.geo_mmd])
    print_report(study)


def print_report(study: optuna.Study) -> None:
    """Print the full trial table, then the feasible Pareto front."""
    columns = "auprc gs rd degree_mmd clustering_mmd spectral_mmd selected_epoch".split()
    print("number state w_rank w_dist bank margin " + " ".join(columns))
    for t in study.get_trials(deepcopy=False):
        surface = t.user_attrs.get("surface", {})
        values = " ".join(f"{surface[c]:.4f}" if c in surface else "-" for c in columns)
        print(
            f"{t.number} {t.state.name} {t.params.get('w_rank', '-')} "
            f"{t.params.get('w_dist', '-')} {t.params.get('bank', '-')} "
            f"{t.params.get('margin', '-')} {values}"
        )
    front = ", ".join(str(t.number) for t in study.best_trials)
    print(f"feasible Pareto front (advisory): trials [{front}]")


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m src.experiments.kd_rank_strict_hpo` parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("configs/autoresearch/kd_rank.yaml")
    )
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/b1_kd_rank_strict_hpo"))
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--rd-band", type=float, default=0.05)
    parser.add_argument("--dump-shards", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unattended container sweep."""
    run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
```

Note the trailing `_n_complete` check semantics: FAILed trials never count toward `--n-trials`, matching the spec's "until 16 completed trials".

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_kd_rank_strict_hpo.py -n0 -v`
Expected: PASS (all tasks' tests).

- [ ] **Step 5: Gates and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
git add src/experiments/kd_rank_strict_hpo.py tests/experiments/test_kd_rank_strict_hpo.py
git commit -m "feat(kd2): unattended strict-LLP Optuna sweep driver"
```

---

### Task 7: Runbook line, full-suite gate, wave review

**Files:**
- Modify: `hpc/README.md` (usage section)

**Interfaces:**
- Consumes: the finished driver CLI (Task 6).
- Produces: an operator-visible launch line; a fully green fast suite.

- [ ] **Step 1: Add the runbook entry**

In `hpc/README.md`, next to the other launch examples, add (three lines — keep the addition minimal per the doc-conciseness rule):

```markdown
Strict-LLP kd_rank HPO sweep (unattended; run inside tmux, dumps missing context banks first):

    .venv/bin/python -m src.experiments.kd_rank_strict_hpo --teacher-checkpoint <full_ego_oracle best.pt>
```

- [ ] **Step 2: Run the fast local suite and gates**

```bash
.venv/bin/python -m pytest -m "not slow and not integration"
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
.venv/bin/python -m mypy src tests
```
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add hpc/README.md
git commit -m "docs(hpc): strict-LLP kd_rank sweep launch line"
```

- [ ] **Step 4: Wave review**

Per `CLAUDE.md`: run the Codex review over the whole wave, backgrounded, output to a scratch file — never into context:

```bash
CODEX_HOME=<scratch>/codex-home codex review --base <sha-before-task-1> > <scratch>/wave-review.txt 2>&1 &
```

Read only the findings summary afterwards; fix anything real, then re-run gates and commit fixes.
