# KD Autoresearch Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen toolkit and state files for the KD autoresearch HPO loop — the kd_control sweep point (Phase 0) and the `src/autoresearch/` judge/ledger/curves/summary toolkit plus `autoresearch/program.md` (Phase 1).

**Architecture:** A thin frozen toolkit: `metrics_io` reads one published run's six-metric surface at its selected epoch; `verdict`/`judge` computes the strict topology no-regression keep/revert decision; `curves` distills `metrics.jsonl` into a per-trial CSV + PNG; `ledger` is a validated append-only JSONL record; `summary` renders a deterministic cold-start digest. The operator agent (a Claude Code session) does git/ssh/launch itself per `autoresearch/program.md`; nothing in this plan automates that.

**Tech Stack:** Python 3.11, dataclasses, matplotlib (Agg), pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md`

## Global Constraints

- Local commands use `.venv/bin/python -m …` (never bare `python`, never `uv run`).
- Tests: `.venv/bin/python -m pytest <file> -n0 -v` while iterating; every new test module starts with `pytestmark = pytest.mark.unit`.
- Lint/type gates per task: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests` and `.venv/bin/python -m mypy src tests` (strict; tests included).
- Ruff enforces `T201` (no `print` — CLIs use `sys.stdout.write`), `ANN` (full annotations), `D` google-style docstrings on all public `src/` modules/functions (tests exempt from D100–D104/ANN201), line length 100, `UP038` (`isinstance(x, int | float)` form).
- Every `src/` module begins with a docstring and `from __future__ import annotations`.
- Do NOT touch: `configs/sweep/b1_kd_hpo/` beyond Task 1, `src/eval/checkpoint_selection.py`, `hpc/`, any `docs/results/` content.
- Commit after each task; end commit messages with the repo's required trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JCvN9Ne1Qn8kqSjruXtD5m`.
- Out of scope (operational, not implementation): running the 25-point grid on the H20, recording baseline ledger rows, and campaign execution. Those follow `autoresearch/program.md` after this plan lands.

---

### Task 1: kd_control sweep point (Phase 0)

**Files:**
- Create: `configs/sweep/b1_kd_hpo/kd_control.yaml`
- Modify: `tests/test_sweep_configs.py`

**Interfaces:**
- Consumes: `configs/b1_kd_control_breadth_first.yaml` (the base control config; has NO `distill:` section and NO `eval.classification_only` key).
- Produces: the 25th sweep config the grid runner (`hpc/sweep_kd_hpo.sh`, which globs `configs/sweep/b1_kd_hpo/*.yaml`) picks up automatically. No code interface for later tasks.

- [ ] **Step 1: Extend the sweep test to expect kd_control**

In `tests/test_sweep_configs.py`, make these exact edits.

Replace the module docstring (lines 1–7) with:

```python
"""Drift guard for the checked-in KD loss-weight sweep configs.

Every YAML in ``configs/sweep/b1_kd_hpo/`` must be a copy of its base arm
config differing only in the ``distill:`` weight/temperature values, the
removal of ``eval.classification_only`` (uniform topology-aware validation),
and its sweep ``output_dir``. ``kd_control`` has no ``distill:`` section and
differs from its base in ``output_dir`` alone.
"""
```

Add to `_BASE_CONFIGS` (after the `"kd_rep"` entry):

```python
    "kd_control": _REPO_ROOT / "configs" / "b1_kd_control_breadth_first.yaml",
```

Add to `EXPECTED_SWEEPS` (after the `"kd_rep_w100"` entry):

```python
    "kd_control": ("kd_control", {}),
```

In `test_sweep_config_differs_from_base_only_in_distill_eval_and_output_dir`, replace

```python
    if arm == "kd_logit":
        assert expected_eval == base_eval  # kd_logit's base eval stays byte-identical
```

with

```python
    if arm in {"kd_logit", "kd_control"}:
        assert expected_eval == base_eval  # these base evals stay byte-identical
```

and wrap the distill assertions (the block from `base_distill_mapping = …` through
`assert distill == replace(base_distill, **expected_weights)`) in a control branch:

```python
    if arm == "kd_control":
        assert "distill" not in base
        assert "distill" not in sweep
    else:
        base_distill_mapping = cast(dict[str, object], base["distill"])
        expected_distill_mapping = base_distill_mapping | expected_weights
        assert sweep["distill"] == expected_distill_mapping

        base_distill = DistillConfig.from_mapping(base_distill_mapping)
        distill = DistillConfig.from_mapping(cast(dict[str, object], sweep["distill"]))
        assert distill.arm == arm
        assert distill == replace(base_distill, **expected_weights)
    assert sweep["output_dir"] == f"outputs/b1_row_kd_hpo/{stem}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py -n0 -v`
Expected: `test_all_expected_sweep_stems_exist_exactly` FAILS (stem set lacks `kd_control`); the parametrized `kd_control` case errors on the missing file.

- [ ] **Step 3: Create the sweep config**

Create `configs/sweep/b1_kd_hpo/kd_control.yaml` — a byte-faithful copy of
`configs/b1_kd_control_breadth_first.yaml` except the header comment and `output_dir`:

```yaml
# B1 matched control on the HPO axis: identical to the base control config
# (V3.1 student, no `distill:` section, topology-aware selection) except for
# the sweep output_dir. Anchors the KD weight grid at zero KD influence.
model:
  family: v3_1
  config:
    input_dim: 1536
    d_model: 512
    encoder_layers: 3
    cross_attn_layers: 3
    n_heads: 8
    mlp_head:
      hidden_dims: [512, 256, 128]
      dropout: 0.2
      activation: gelu
      norm: layernorm
      spectral_norm: false
    regularization:
      dropout: 0.1
      token_dropout: 0.1
      cross_attention_dropout: 0.1
      stochastic_depth: 0.1
    rich_pooling:
      components: [mean, attn, max, gated]
    pair_readout:
      mode: pair_context_gated
      order_aggregation: abba_max
      spectral_norm: false
    mixing:
      mode: none
    label_smoothing: 0.05 # symmetric BCE smoothing, fixed V3.1 recipe
data:
  root: data
  strategy: breadth_first
  negative_ratio: 1 # F0-only; V3.1 trains on fixed balanced train_edges.txt rows
  token_budget: 131072
  batch_pairs: 1024
  num_workers: 0
  f0_cache: outputs/f0_cache/f0_matrix.pt
  expected_missing_features: [node_004764, node_007050]
optim:
  lr: 1.0e-4
  weight_decay: 0.05
  epochs: 25
  warmup_steps: 500 # unused while a onecycle scheduler is configured
  grad_clip: 1.0
  scheduler:
    type: onecycle
    max_lr: 1.0e-4
    pct_start: 0.1
    div_factor: 25
    final_div_factor: 10000
    anneal_strategy: cos
eval:
  patience: 10
  eval_every: 1
runtime:
  world_size: auto
  pack_dir: outputs/feature_packs/b0_v31_bf16
  pack_workers: 16
  loader_workers_per_rank: 4
  prefetch_factor: 4
  token_budget: 524288
  max_pairs_per_rank: 4096
  memory_limit_gib: 85.0
  probe_warmup_steps: 10
  probe_timed_steps: 30
seed: 0
output_dir: outputs/b1_row_kd_hpo/kd_control
mixed_precision: "bf16" # pinned for the H20 E2 target
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sweep_configs.py tests/test_hpc_scripts.py -n0 -q`
Expected: ALL PASS (the hpc-script tests use a mock config checkout and are unaffected).

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add configs/sweep/b1_kd_hpo/kd_control.yaml tests/test_sweep_configs.py
git commit -m "feat(sweep): add kd_control point to the b1 KD HPO grid"
```

---

### Task 2: package skeleton + metrics_io

**Files:**
- Create: `src/autoresearch/__init__.py`
- Create: `src/autoresearch/metrics_io.py`
- Create: `tests/autoresearch/__init__.py`
- Create: `tests/autoresearch/conftest.py`
- Test: `tests/autoresearch/test_metrics_io.py`

**Interfaces:**
- Consumes: `src.eval.checkpoint_selection.TopologyValidationMetrics` (frozen dataclass with float fields `gs`, `rd`, `degree_mmd`, `clustering_mmd`, `spectral_mmd`).
- Produces (used by Tasks 3–4):
  - `RunFailure(RuntimeError)` — raised when `failure.json` exists in the run dir.
  - `@dataclass(frozen=True) RunMetrics(run_dir: Path, selected_epoch: int, auprc: float, topology: TopologyValidationMetrics, threshold: float, total_seconds: float)`
  - `read_run(run_dir: Path) -> RunMetrics`
  - `read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]` (strict parse — published runs are pipeline-validated).
  - Test helpers `tests.autoresearch.conftest.make_metric_row(epoch, **overrides) -> dict[str, Any]` and fixture `make_run_dir` (factory `Callable[..., Path]`, alias `RunDirFactory`).

- [ ] **Step 1: Write conftest and the failing tests**

Create `tests/autoresearch/__init__.py` (empty file).

Create `tests/autoresearch/conftest.py`:

```python
"""Fixture helpers that synthesize minimal published run directories."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

RunDirFactory = Callable[..., Path]


def make_metric_row(epoch: int, **overrides: object) -> dict[str, Any]:
    """One synthetic metrics.jsonl row in the B1 KD schema."""
    row: dict[str, Any] = {
        "epoch": epoch,
        "attempt_id": "fixture",
        "global_step": epoch * 10,
        "timestamp": f"2026-08-30T00:{epoch:02d}:00+00:00",
        "learning_rate": 1e-4,
        "train_loss": 1.0 / epoch,
        "train_kd_loss": 0.5 / epoch,
        "val_task_loss": 1.2 / epoch,
        "val_auroc": 0.9,
        "val_auprc": 0.80 + 0.01 * epoch,
        "val_ece": 0.05,
        "val_brier": 0.10,
        "val_gs_bfs": 0.50 + 0.01 * epoch,
        "val_rd_bfs": 1.10,
        "val_degree_mmd_ratio": 0.90,
        "val_clustering_mmd_ratio": 0.85,
        "val_spectral_mmd_ratio": 0.80,
        "val_threshold": 2.5,
        "grad_norm_task": 1.0,
        "grad_norm_kd": 0.3,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def make_run_dir(tmp_path: Path) -> RunDirFactory:
    """Factory building a published-run directory under ``tmp_path``."""

    def _make(
        name: str = "run",
        epochs: int = 3,
        selected_epoch: int = 2,
        rows: list[dict[str, Any]] | None = None,
        total_seconds: float = 123.0,
        failure: dict[str, Any] | None = None,
    ) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        actual = rows if rows is not None else [make_metric_row(e) for e in range(1, epochs + 1)]
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in actual), encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "selected_epoch": selected_epoch,
                    "arm": "kd_logit",
                    "config_hash": "deadbeef",
                    "checkpoint_id": "cafe0000",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "complete.json").write_text(
            json.dumps(
                {"status": "complete", "attempt_id": "fixture", "total_seconds": total_seconds}
            ),
            encoding="utf-8",
        )
        if failure is not None:
            (run_dir / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
        return run_dir

    return _make
```

Create `tests/autoresearch/test_metrics_io.py`:

```python
from pathlib import Path

import pytest

from src.autoresearch.metrics_io import RunFailure, read_metric_rows, read_run
from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def test_read_run_returns_selected_epoch_surface(make_run_dir: RunDirFactory) -> None:
    run = read_run(make_run_dir(selected_epoch=2))
    assert run.selected_epoch == 2
    assert run.auprc == pytest.approx(0.82)
    assert run.topology.gs == pytest.approx(0.52)
    assert run.topology.rd == pytest.approx(1.10)
    assert run.topology.degree_mmd == pytest.approx(0.90)
    assert run.threshold == pytest.approx(2.5)
    assert run.total_seconds == pytest.approx(123.0)


def test_read_run_raises_on_failure_marker(make_run_dir: RunDirFactory) -> None:
    run_dir = make_run_dir(failure={"stage": "train", "message": "boom"})
    with pytest.raises(RunFailure, match="boom"):
        read_run(run_dir)


def test_read_run_rejects_non_finite_metric(make_run_dir: RunDirFactory) -> None:
    rows = [make_metric_row(1), make_metric_row(2, val_gs_bfs=float("nan"))]
    with pytest.raises(ValueError, match="val_gs_bfs"):
        read_run(make_run_dir(rows=rows, selected_epoch=2))


def test_read_run_rejects_non_positive_rd(make_run_dir: RunDirFactory) -> None:
    rows = [make_metric_row(1, val_rd_bfs=0.0)]
    with pytest.raises(ValueError, match="val_rd_bfs"):
        read_run(make_run_dir(rows=rows, selected_epoch=1))


def test_read_run_rejects_missing_selected_epoch_row(make_run_dir: RunDirFactory) -> None:
    with pytest.raises(ValueError, match="selected epoch 9"):
        read_run(make_run_dir(selected_epoch=9))


def test_read_metric_rows_rejects_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        read_metric_rows(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_metrics_io.py -n0 -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.autoresearch'`.

- [ ] **Step 3: Implement the package and metrics_io**

Create `src/autoresearch/__init__.py`:

```python
"""Frozen toolkit for the KD autoresearch HPO loop.

Spec: docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md. These
modules are read-only during campaigns: the operator agent invokes them but
never edits them.
"""
```

Create `src/autoresearch/metrics_io.py`:

```python
"""Read the frozen objective surface of one published training run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.checkpoint_selection import TopologyValidationMetrics


class RunFailure(RuntimeError):
    """The run directory carries a ``failure.json`` marker."""


@dataclass(frozen=True)
class RunMetrics:
    """Judge-facing summary of one published run at its selected epoch."""

    run_dir: Path
    selected_epoch: int
    auprc: float
    topology: TopologyValidationMetrics
    threshold: float
    total_seconds: float


def read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """Parse every ``metrics.jsonl`` row strictly (published runs are validated).

    Raises:
        ValueError: If a line is not a JSON object.
    """
    rows: list[dict[str, Any]] = []
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{metrics_path}:{number}: row is not an object")
        rows.append(parsed)
    return rows


def read_run(run_dir: Path) -> RunMetrics:
    """Load the six-metric surface at the selected epoch of ``run_dir``.

    Raises:
        RunFailure: If ``failure.json`` is present (the run must count as a crash).
        ValueError: On missing rows/keys, non-finite metrics, or RD <= 0.
    """
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        detail = failure_path.read_text(encoding="utf-8").strip()
        raise RunFailure(f"{run_dir} failed: {detail}")
    metadata = _load_json(run_dir / "run_metadata.json")
    selected_epoch = metadata["selected_epoch"]
    if not isinstance(selected_epoch, int):
        raise ValueError(f"{run_dir}: selected_epoch must be an int")
    row = _selected_row(run_dir / "metrics.jsonl", selected_epoch)
    auprc = _finite(row, "val_auprc", run_dir)
    gs = _finite(row, "val_gs_bfs", run_dir)
    rd = _finite(row, "val_rd_bfs", run_dir)
    degree_mmd = _finite(row, "val_degree_mmd_ratio", run_dir)
    clustering_mmd = _finite(row, "val_clustering_mmd_ratio", run_dir)
    spectral_mmd = _finite(row, "val_spectral_mmd_ratio", run_dir)
    threshold = _finite(row, "val_threshold", run_dir)
    if rd <= 0.0:
        raise ValueError(f"{run_dir}: val_rd_bfs must be positive, got {rd}")
    complete = _load_json(run_dir / "complete.json")
    return RunMetrics(
        run_dir=run_dir,
        selected_epoch=selected_epoch,
        auprc=auprc,
        topology=TopologyValidationMetrics(
            gs=gs,
            rd=rd,
            degree_mmd=degree_mmd,
            clustering_mmd=clustering_mmd,
            spectral_mmd=spectral_mmd,
        ),
        threshold=threshold,
        total_seconds=float(complete["total_seconds"]),
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object; reject any other JSON top-level type."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return parsed


def _selected_row(metrics_path: Path, selected_epoch: int) -> dict[str, Any]:
    """Return the metrics row whose epoch matches ``selected_epoch``."""
    for row in read_metric_rows(metrics_path):
        if row.get("epoch") == selected_epoch:
            return row
    raise ValueError(f"{metrics_path}: no row for selected epoch {selected_epoch}")


def _finite(row: Mapping[str, Any], key: str, run_dir: Path) -> float:
    """Extract ``row[key]`` as a finite float."""
    if key not in row:
        raise ValueError(f"{run_dir}: metrics row missing {key!r}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{run_dir}: non-finite {key}={value}")
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_metrics_io.py -n0 -v`
Expected: 6 PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add src/autoresearch/__init__.py src/autoresearch/metrics_io.py tests/autoresearch/
git commit -m "feat(autoresearch): published-run metrics reader"
```

---

### Task 3: verdict + judge CLI + Pareto helper

**Files:**
- Create: `src/autoresearch/verdict.py`
- Create: `src/autoresearch/judge.py`
- Test: `tests/autoresearch/test_verdict.py`

**Interfaces:**
- Consumes: `RunMetrics`, `read_run`, `RunFailure` from Task 2; `TopologyValidationMetrics`.
- Produces:
  - `verdict.METRIC_NAMES: tuple[str, ...] = ("gs", "log_rd", "degree_mmd", "clustering_mmd", "spectral_mmd")`
  - `@dataclass(frozen=True) Verdict(decision: str, improved: tuple[str, ...], regressed: tuple[str, ...], deltas: dict[str, float], auprc_delta: float)` — `decision` is `"keep"` or `"revert"`; `deltas` are oriented (negative = better).
  - `oriented(topology: TopologyValidationMetrics) -> dict[str, float]`
  - `judge_runs(incumbent: RunMetrics, trial: RunMetrics, bands: Mapping[str, float] | None = None) -> Verdict`
  - `undominated(runs: Sequence[RunMetrics]) -> list[RunMetrics]` — Pareto filter for Phase 0 incumbent identification.
  - `judge.main(argv: list[str] | None = None) -> int` — CLI `python -m src.autoresearch.judge --incumbent DIR --trial DIR [--bands FILE]`; writes one JSON object to stdout with the `Verdict` fields plus `incumbent`/`trial` surface dicts (keys: `run_dir`, `selected_epoch`, `auprc`, `gs`, `rd`, `degree_mmd`, `clustering_mmd`, `spectral_mmd`, `threshold`, `total_seconds`).

- [ ] **Step 1: Write the failing tests**

Create `tests/autoresearch/test_verdict.py`:

```python
import json
from pathlib import Path

import pytest

from src.autoresearch.judge import main as judge_main
from src.autoresearch.metrics_io import RunMetrics
from src.autoresearch.verdict import judge_runs, undominated
from src.eval.checkpoint_selection import TopologyValidationMetrics
from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def run_with(
    gs: float = 0.50,
    rd: float = 1.10,
    degree_mmd: float = 0.90,
    clustering_mmd: float = 0.85,
    spectral_mmd: float = 0.80,
    auprc: float = 0.82,
) -> RunMetrics:
    return RunMetrics(
        run_dir=Path("fixture"),
        selected_epoch=1,
        auprc=auprc,
        topology=TopologyValidationMetrics(
            gs=gs,
            rd=rd,
            degree_mmd=degree_mmd,
            clustering_mmd=clustering_mmd,
            spectral_mmd=spectral_mmd,
        ),
        threshold=2.5,
        total_seconds=1.0,
    )


def test_keep_when_one_metric_strictly_improves() -> None:
    verdict = judge_runs(run_with(), run_with(gs=0.60))
    assert verdict.decision == "keep"
    assert verdict.improved == ("gs",)
    assert verdict.regressed == ()


def test_revert_on_exact_tie() -> None:
    verdict = judge_runs(run_with(), run_with())
    assert verdict.decision == "revert"
    assert verdict.improved == ()
    assert verdict.regressed == ()


def test_revert_when_any_metric_regresses() -> None:
    verdict = judge_runs(run_with(), run_with(gs=0.60, degree_mmd=0.95))
    assert verdict.decision == "revert"
    assert verdict.improved == ("gs",)
    assert verdict.regressed == ("degree_mmd",)


def test_rd_is_judged_by_absolute_log_distance_from_one() -> None:
    # |log 0.90| ~= 0.105 > |log 1.10| ~= 0.095: moving RD from 1.10 to 0.90 regresses.
    verdict = judge_runs(run_with(rd=1.10), run_with(rd=0.90))
    assert verdict.regressed == ("log_rd",)
    assert verdict.decision == "revert"


def test_bands_absorb_small_moves_in_both_directions() -> None:
    bands = {"gs": 0.02, "degree_mmd": 0.02}
    small_both_ways = run_with(gs=0.51, degree_mmd=0.91, spectral_mmd=0.70)
    verdict = judge_runs(run_with(), small_both_ways, bands)
    assert verdict.improved == ("spectral_mmd",)
    assert verdict.regressed == ()
    assert verdict.decision == "keep"


def test_auprc_never_enters_the_decision() -> None:
    verdict = judge_runs(run_with(auprc=0.82), run_with(gs=0.60, auprc=0.10))
    assert verdict.decision == "keep"
    assert verdict.auprc_delta == pytest.approx(-0.72)


def test_negative_band_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        judge_runs(run_with(), run_with(), {"gs": -0.1})


def test_undominated_filters_pareto_dominated_runs() -> None:
    best = run_with(gs=0.60)
    dominated = run_with(gs=0.40, spectral_mmd=0.90)
    tradeoff = run_with(gs=0.40, spectral_mmd=0.10)
    survivors = undominated([best, dominated, tradeoff])
    assert survivors == [best, tradeoff]


def test_judge_cli_emits_verdict_json(
    make_run_dir: RunDirFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    incumbent = make_run_dir(name="incumbent", selected_epoch=1)
    trial_rows = [make_metric_row(1, val_gs_bfs=0.60)]
    trial = make_run_dir(name="trial", rows=trial_rows, selected_epoch=1)
    assert judge_main(["--incumbent", str(incumbent), "--trial", str(trial)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "keep"
    assert payload["improved"] == ["gs"]
    assert payload["trial"]["gs"] == pytest.approx(0.60)
    assert payload["incumbent"]["auprc"] == pytest.approx(0.81)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_verdict.py -n0 -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.autoresearch.judge'`.

- [ ] **Step 3: Implement verdict.py and judge.py**

Create `src/autoresearch/verdict.py`:

```python
"""Frozen keep/revert verdict over the five topology metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.autoresearch.metrics_io import RunMetrics
from src.eval.checkpoint_selection import TopologyValidationMetrics

METRIC_NAMES = ("gs", "log_rd", "degree_mmd", "clustering_mmd", "spectral_mmd")


@dataclass(frozen=True)
class Verdict:
    """Outcome of judging one trial against the incumbent."""

    decision: str
    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    deltas: dict[str, float]
    auprc_delta: float


def oriented(topology: TopologyValidationMetrics) -> dict[str, float]:
    """Orient the five metrics so that lower is always better."""
    return {
        "gs": -topology.gs,
        "log_rd": abs(math.log(topology.rd)),
        "degree_mmd": topology.degree_mmd,
        "clustering_mmd": topology.clustering_mmd,
        "spectral_mmd": topology.spectral_mmd,
    }


def judge_runs(
    incumbent: RunMetrics,
    trial: RunMetrics,
    bands: Mapping[str, float] | None = None,
) -> Verdict:
    """Keep iff >=1 metric improves beyond its band and none regresses beyond its band.

    Bands default to zero width (strict no-regression). AUPRC is telemetry
    only and never enters the decision.

    Raises:
        ValueError: If any band width is negative.
    """
    widths = {name: float((bands or {}).get(name, 0.0)) for name in METRIC_NAMES}
    if any(width < 0.0 for width in widths.values()):
        raise ValueError("tolerance bands must be non-negative")
    incumbent_oriented = oriented(incumbent.topology)
    trial_oriented = oriented(trial.topology)
    deltas = {name: trial_oriented[name] - incumbent_oriented[name] for name in METRIC_NAMES}
    improved = tuple(name for name in METRIC_NAMES if deltas[name] < -widths[name])
    regressed = tuple(name for name in METRIC_NAMES if deltas[name] > widths[name])
    decision = "keep" if improved and not regressed else "revert"
    return Verdict(
        decision=decision,
        improved=improved,
        regressed=regressed,
        deltas=deltas,
        auprc_delta=trial.auprc - incumbent.auprc,
    )


def undominated(runs: Sequence[RunMetrics]) -> list[RunMetrics]:
    """Return runs not Pareto-dominated on the five oriented topology metrics."""
    surfaces = [oriented(run.topology) for run in runs]

    def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
        return all(a[name] <= b[name] for name in METRIC_NAMES) and any(
            a[name] < b[name] for name in METRIC_NAMES
        )

    return [
        run
        for run, surface in zip(runs, surfaces, strict=True)
        if not any(dominates(other, surface) for other in surfaces if other is not surface)
    ]
```

Create `src/autoresearch/judge.py`:

```python
"""CLI: judge one trial run directory against the incumbent."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.autoresearch.metrics_io import RunMetrics, read_run
from src.autoresearch.verdict import judge_runs


def main(argv: list[str] | None = None) -> int:
    """Write the verdict JSON for ``--trial`` vs ``--incumbent`` to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-judge")
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--bands", type=Path, default=None)
    args = parser.parse_args(argv)
    bands: dict[str, float] | None = None
    if args.bands is not None:
        loaded = json.loads(args.bands.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{args.bands}: bands file must be a JSON object")
        bands = {str(key): float(value) for key, value in loaded.items()}
    incumbent = read_run(args.incumbent)
    trial = read_run(args.trial)
    verdict = judge_runs(incumbent, trial, bands)
    payload = asdict(verdict) | {"incumbent": _surface(incumbent), "trial": _surface(trial)}
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _surface(run: RunMetrics) -> dict[str, Any]:
    """Flatten one run's judge-facing surface for the JSON payload."""
    return {
        "run_dir": str(run.run_dir),
        "selected_epoch": run.selected_epoch,
        "auprc": run.auprc,
        "gs": run.topology.gs,
        "rd": run.topology.rd,
        "degree_mmd": run.topology.degree_mmd,
        "clustering_mmd": run.topology.clustering_mmd,
        "spectral_mmd": run.topology.spectral_mmd,
        "threshold": run.threshold,
        "total_seconds": run.total_seconds,
    }


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_verdict.py -n0 -v`
Expected: 9 PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add src/autoresearch/verdict.py src/autoresearch/judge.py tests/autoresearch/test_verdict.py
git commit -m "feat(autoresearch): frozen topology verdict, judge CLI, Pareto helper"
```

---

### Task 4: learning-curve CSV + plot distiller

**Files:**
- Modify: `pyproject.toml:18-28` (dependencies list)
- Create: `src/autoresearch/curves.py`
- Test: `tests/autoresearch/test_curves.py`

**Interfaces:**
- Consumes: `read_metric_rows` from Task 2; fixture `make_run_dir` / `make_metric_row`.
- Produces:
  - `curves.CSV_COLUMNS: tuple[str, ...]` — exact CSV header order (below).
  - `write_curves(run_dir: Path, out_dir: Path) -> tuple[Path, Path]` — writes `learning_curves.csv` and `learning_curves.png` into `out_dir` (created if missing), returns `(csv_path, png_path)`.
  - CLI `python -m src.autoresearch.curves RUN_DIR OUT_DIR` via `main(argv) -> int`.

- [ ] **Step 1: Declare matplotlib**

In `pyproject.toml`, insert into `dependencies` after `"accelerate==1.13.0",`:

```toml
    "matplotlib==3.11.1",
```

Run: `uv sync`
Expected: resolves cleanly (3.11.1 is already the installed version).

- [ ] **Step 2: Write the failing tests**

Create `tests/autoresearch/test_curves.py`:

```python
import csv

import pytest

from src.autoresearch.curves import CSV_COLUMNS, main as curves_main, write_curves
from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def test_write_curves_emits_csv_in_column_order(make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory) -> None:
    out_dir = tmp_path_factory.mktemp("curves")
    run_dir = make_run_dir(epochs=2, selected_epoch=1)
    csv_path, png_path = write_curves(run_dir, out_dir)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert float(rows[1]["val_gs_bfs"]) == pytest.approx(0.52)
    assert float(rows[0]["train_kd_loss"]) == pytest.approx(0.5)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_missing_keys_become_empty_cells(make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory) -> None:
    out_dir = tmp_path_factory.mktemp("curves-control")
    row = make_metric_row(1)
    del row["train_kd_loss"]
    del row["grad_norm_kd"]
    run_dir = make_run_dir(rows=[row], selected_epoch=1)
    csv_path, _ = write_curves(run_dir, out_dir)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["train_kd_loss"] == ""
    assert parsed[0]["grad_norm_kd"] == ""


def test_curves_cli_writes_both_artifacts(make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory) -> None:
    out_dir = tmp_path_factory.mktemp("curves-cli")
    run_dir = make_run_dir(epochs=2, selected_epoch=2)
    assert curves_main([str(run_dir), str(out_dir)]) == 0
    assert (out_dir / "learning_curves.csv").exists()
    assert (out_dir / "learning_curves.png").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_curves.py -n0 -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.autoresearch.curves'`.

- [ ] **Step 4: Implement curves.py**

Create `src/autoresearch/curves.py`:

```python
"""Distill one run's metrics.jsonl into a learning-curve CSV and plot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.autoresearch.metrics_io import read_metric_rows  # noqa: E402

CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "train_kd_loss",
    "val_task_loss",
    "val_auprc",
    "val_gs_bfs",
    "val_rd_bfs",
    "val_degree_mmd_ratio",
    "val_clustering_mmd_ratio",
    "val_spectral_mmd_ratio",
    "learning_rate",
    "grad_norm_task",
    "grad_norm_kd",
)

_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("train_loss", "train_kd_loss", "val_task_loss")),
    ("topology (GS, RD)", ("val_gs_bfs", "val_rd_bfs")),
    (
        "MMD ratios",
        ("val_degree_mmd_ratio", "val_clustering_mmd_ratio", "val_spectral_mmd_ratio"),
    ),
    ("val AUPRC", ("val_auprc",)),
)


def write_curves(run_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write ``learning_curves.csv`` and ``learning_curves.png`` for one run."""
    rows = read_metric_rows(run_dir / "metrics.jsonl")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    selected_epoch = int(metadata["selected_epoch"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "learning_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    png_path = out_dir / "learning_curves.png"
    _plot(rows, selected_epoch, png_path)
    return csv_path, png_path


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """Collect (epoch, value) pairs for rows that carry ``key``."""
    epochs = [int(row["epoch"]) for row in rows if key in row]
    values = [float(row[key]) for row in rows if key in row]
    return epochs, values


def _plot(rows: list[dict[str, Any]], selected_epoch: int, png_path: Path) -> None:
    """Render the four-panel learning-curve figure with the selected epoch marked."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    flat_axes = (axes[0][0], axes[0][1], axes[1][0], axes[1][1])
    for ax, (title, keys) in zip(flat_axes, _PANELS, strict=True):
        for key in keys:
            epochs, values = _series(rows, key)
            if epochs:
                ax.plot(epochs, values, marker="o", markersize=2.5, linewidth=1.4, label=key)
        ax.axvline(selected_epoch, color="#666666", linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.4, linewidth=0.5)
        ax.legend(fontsize=7)
    for ax in (axes[1][0], axes[1][1]):
        ax.set_xlabel("epoch")
    fig.suptitle(f"selected epoch {selected_epoch}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: distill RUN_DIR's metrics.jsonl into OUT_DIR's CSV + PNG."""
    parser = argparse.ArgumentParser(prog="autoresearch-curves")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    csv_path, png_path = write_curves(args.run_dir, args.out_dir)
    sys.stdout.write(f"{csv_path}\n{png_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_curves.py -n0 -v`
Expected: 3 PASS.

- [ ] **Step 6: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add pyproject.toml uv.lock src/autoresearch/curves.py tests/autoresearch/test_curves.py
git commit -m "feat(autoresearch): learning-curve CSV and plot distiller"
```

---

### Task 5: append-only ledger

**Files:**
- Create: `src/autoresearch/ledger.py`
- Test: `tests/autoresearch/test_ledger.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone by design — the ledger must replay without the rest of the toolkit).
- Produces (used by Task 6):
  - `ledger.STATUSES = frozenset({"baseline", "keep", "revert", "crash"})`
  - `ledger.METRIC_KEYS = frozenset({"auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"})`
  - `ledger.REQUIRED_KEYS` — the 13 row keys (below).
  - `read_rows(path: Path) -> list[dict[str, Any]]` — replay-tolerant (skips torn/unparseable lines; missing file → `[]`).
  - `append_row(path: Path, row: Mapping[str, Any]) -> None` — validates against existing rows, then appends one sorted-key JSON line.
  - CLI `python -m src.autoresearch.ledger LEDGER_PATH ROW_JSON_PATH` via `main(argv) -> int`.

Row schema (all 13 keys required on every row): `trial` (int, strictly `max(existing)+1`, first row is 1), `campaign` (str), `commit` (str, unique), `config_hash` (str), `output_dir` (str, unique), `hypothesis` (str), `status` (one of `STATUSES`), `metrics` (dict with exactly `METRIC_KEYS`, all finite; `None` required for `crash`), `selected_epoch` (int or `None`), `total_seconds` (number or `None`), `verdict` (dict whose `decision` equals the status, required for `keep`/`revert`; `None` required for `baseline`/`crash`), `asi` (free-form), `timestamp` (str).

- [ ] **Step 1: Write the failing tests**

Create `tests/autoresearch/test_ledger.py`:

```python
import json
from pathlib import Path
from typing import Any

import pytest

from src.autoresearch.ledger import append_row, read_rows

pytestmark = pytest.mark.unit


def valid_row(trial: int = 1, **overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial": trial,
        "campaign": "kd_logit",
        "commit": f"c{trial:07d}",
        "config_hash": "deadbeef",
        "output_dir": f"outputs/b1_row_kd_ar/kd_logit/trial_{trial:03d}",
        "hypothesis": "baseline" if trial == 1 else f"hypothesis {trial}",
        "status": "baseline" if trial == 1 else "keep",
        "metrics": {
            "auprc": 0.82,
            "gs": 0.52,
            "rd": 1.10,
            "degree_mmd": 0.90,
            "clustering_mmd": 0.85,
            "spectral_mmd": 0.80,
        },
        "selected_epoch": 2,
        "total_seconds": 123.0,
        "verdict": None if trial == 1 else {"decision": "keep", "improved": ["gs"]},
        "asi": "healthy fit",
        "timestamp": "2026-08-30T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    append_row(path, valid_row(2))
    rows = read_rows(path)
    assert [row["trial"] for row in rows] == [1, 2]
    assert rows[1]["status"] == "keep"


def test_missing_key_rejected(tmp_path: Path) -> None:
    row = valid_row(1)
    del row["hypothesis"]
    with pytest.raises(ValueError, match="hypothesis"):
        append_row(tmp_path / "ledger.jsonl", row)


def test_non_monotonic_trial_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="trial must be 2"):
        append_row(path, valid_row(3))


def test_duplicate_output_dir_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    duplicate = valid_row(2, output_dir=valid_row(1)["output_dir"])
    with pytest.raises(ValueError, match="duplicate output_dir"):
        append_row(path, duplicate)


def test_duplicate_commit_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with pytest.raises(ValueError, match="duplicate commit"):
        append_row(path, valid_row(2, commit="c0000001"))


def test_keep_row_contradicting_verdict_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    contradicted = valid_row(2, verdict={"decision": "revert"})
    with pytest.raises(ValueError, match="decision"):
        append_row(path, contradicted)


def test_crash_row_requires_null_metrics(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    crash = valid_row(
        2, status="crash", verdict=None, metrics=None, selected_epoch=None, total_seconds=None
    )
    append_row(path, crash)
    with pytest.raises(ValueError, match="crash rows"):
        append_row(path, valid_row(3, status="crash", verdict=None))


def test_replay_skips_torn_line_and_append_continues(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, valid_row(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"trial": 2, "camp')  # torn tail write
    assert [row["trial"] for row in read_rows(path)] == [1]
    append_row(path, valid_row(2))
    assert [row["trial"] for row in read_rows(path)] == [1, 2]


def test_non_finite_metric_rejected(tmp_path: Path) -> None:
    bad = valid_row(1)
    bad["metrics"] = dict(bad["metrics"], gs=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        append_row(tmp_path / "ledger.jsonl", bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_ledger.py -n0 -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.autoresearch.ledger'`.

- [ ] **Step 3: Implement ledger.py**

Create `src/autoresearch/ledger.py`:

```python
"""Append-only autoresearch trial ledger with validation and tolerant replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STATUSES = frozenset({"baseline", "keep", "revert", "crash"})
METRIC_KEYS = frozenset({"auprc", "gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd"})
REQUIRED_KEYS = frozenset(
    {
        "trial",
        "campaign",
        "commit",
        "config_hash",
        "output_dir",
        "hypothesis",
        "status",
        "metrics",
        "selected_epoch",
        "total_seconds",
        "verdict",
        "asi",
        "timestamp",
    }
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Replay the ledger, skipping unparseable lines (torn tail writes)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Validate ``row`` against the existing ledger, then append it durably.

    A torn tail (a final line without a newline, from an interrupted write)
    is healed by prefixing a newline so the new row starts its own line.

    Raises:
        ValueError: On any schema, monotonicity, uniqueness, or consistency
            violation. A rejected row writes nothing.
    """
    existing = read_rows(path)
    _validate(row, existing)
    raw = path.read_bytes() if path.exists() else b""
    with path.open("a", encoding="utf-8") as handle:
        if raw and not raw.endswith(b"\n"):
            handle.write("\n")
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _validate(row: Mapping[str, Any], existing: list[dict[str, Any]]) -> None:
    """Enforce the ledger row contract against the replayed history."""
    missing = REQUIRED_KEYS - set(row)
    if missing:
        raise ValueError(f"ledger row missing keys: {sorted(missing)}")
    status = row["status"]
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    trials = [entry["trial"] for entry in existing if isinstance(entry.get("trial"), int)]
    expected_trial = (max(trials) + 1) if trials else 1
    if row["trial"] != expected_trial:
        raise ValueError(f"trial must be {expected_trial}, got {row['trial']!r}")
    if row["output_dir"] in {entry.get("output_dir") for entry in existing}:
        raise ValueError(f"duplicate output_dir {row['output_dir']!r}")
    if row["commit"] in {entry.get("commit") for entry in existing}:
        raise ValueError(f"duplicate commit {row['commit']!r}")
    verdict = row["verdict"]
    if status in {"keep", "revert"}:
        if not isinstance(verdict, Mapping) or verdict.get("decision") != status:
            raise ValueError(f"status {status!r} requires a verdict with decision={status!r}")
    elif verdict is not None:
        raise ValueError(f"status {status!r} must carry verdict=None")
    metrics = row["metrics"]
    if status == "crash":
        if metrics is not None:
            raise ValueError("crash rows must carry metrics=None")
        return
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_KEYS:
        raise ValueError(f"metrics must carry exactly {sorted(METRIC_KEYS)}")
    for key in sorted(METRIC_KEYS):
        value = metrics[key]
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"metric {key} must be finite, got {value!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI: append one validated row (a JSON object file) to the ledger."""
    parser = argparse.ArgumentParser(prog="autoresearch-ledger")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("row", type=Path)
    args = parser.parse_args(argv)
    row = json.loads(args.row.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"{args.row}: row file must contain one JSON object")
    append_row(args.ledger, row)
    sys.stdout.write(f"appended trial {row['trial']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_ledger.py -n0 -v`
Expected: 9 PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add src/autoresearch/ledger.py tests/autoresearch/test_ledger.py
git commit -m "feat(autoresearch): validated append-only trial ledger"
```

---

### Task 6: cold-start summary

**Files:**
- Create: `src/autoresearch/summary.py`
- Test: `tests/autoresearch/test_summary.py`

**Interfaces:**
- Consumes: `read_rows`, `append_row` from Task 5 (tests build ledgers through `append_row` so fixtures stay contract-valid).
- Produces: `render_summary(ledger_path: Path, last: int = 10) -> str` (deterministic, ends with `\n`) and CLI `python -m src.autoresearch.summary LEDGER_PATH [--last N]` via `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/autoresearch/test_summary.py`:

```python
from pathlib import Path
from typing import Any

import pytest

from src.autoresearch.ledger import append_row
from src.autoresearch.summary import render_summary

pytestmark = pytest.mark.unit


def row(trial: int, status: str, campaign: str = "kd_logit", **overrides: object) -> dict[str, Any]:
    verdict: dict[str, Any] | None = None
    if status in {"keep", "revert"}:
        verdict = {"decision": status, "improved": ["gs"] if status == "keep" else []}
    base: dict[str, Any] = {
        "trial": trial,
        "campaign": campaign,
        "commit": f"c{trial:07d}",
        "config_hash": "deadbeef",
        "output_dir": f"outputs/b1_row_kd_ar/{campaign}/trial_{trial:03d}",
        "hypothesis": f"hypothesis {trial}",
        "status": status,
        "metrics": {
            "auprc": 0.82,
            "gs": 0.52,
            "rd": 1.10,
            "degree_mmd": 0.90,
            "clustering_mmd": 0.85,
            "spectral_mmd": 0.80,
        },
        "selected_epoch": 2,
        "total_seconds": 123.0,
        "verdict": verdict,
        "asi": "healthy fit",
        "timestamp": "2026-08-30T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_summary_reports_standings_and_recent_trials(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, row(1, "baseline"))
    append_row(path, row(2, "keep", metrics=dict(row(1, "baseline")["metrics"], gs=0.60)))
    append_row(path, row(3, "revert"))
    text = render_summary(path)
    lines = text.splitlines()
    assert lines[0] == "# autoresearch summary"
    assert lines[1].startswith("campaign kd_logit: trials=3 keeps=1 incumbent=")
    assert "trial_002" in lines[1]
    assert "gs=0.6" in lines[1]
    assert lines[2] == "last 3 trials:"
    assert lines[3].startswith("#1 [kd_logit] baseline | hypothesis 1")
    assert "improved:gs" in lines[4]
    assert lines[5].startswith("#3 [kd_logit] revert")


def test_summary_honors_last_limit(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_row(path, row(1, "baseline"))
    append_row(path, row(2, "revert"))
    append_row(path, row(3, "revert"))
    text = render_summary(path, last=1)
    assert "last 1 trials:" in text
    assert "#3 " in text
    assert "#2 " not in text


def test_summary_of_empty_ledger(tmp_path: Path) -> None:
    assert render_summary(tmp_path / "missing.jsonl") == "ledger empty; no trials recorded\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_summary.py -n0 -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.autoresearch.summary'`.

- [ ] **Step 3: Implement summary.py**

Create `src/autoresearch/summary.py`:

```python
"""Deterministic cold-start digest of the autoresearch ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.autoresearch.ledger import read_rows

_INCUMBENT_METRIC_ORDER = ("gs", "rd", "degree_mmd", "clustering_mmd", "spectral_mmd", "auprc")


def render_summary(ledger_path: Path, last: int = 10) -> str:
    """Render campaign standings plus the last ``last`` trials, one line each."""
    rows = read_rows(ledger_path)
    if not rows:
        return "ledger empty; no trials recorded\n"
    lines = ["# autoresearch summary"]
    campaigns: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        campaigns.setdefault(str(row.get("campaign", "?")), []).append(row)
    for campaign in sorted(campaigns):
        entries = campaigns[campaign]
        keeps = [entry for entry in entries if entry.get("status") in {"baseline", "keep"}]
        keep_count = sum(1 for entry in entries if entry.get("status") == "keep")
        line = f"campaign {campaign}: trials={len(entries)} keeps={keep_count}"
        if keeps:
            incumbent = keeps[-1]
            metrics = incumbent.get("metrics") or {}
            surface = " ".join(f"{key}={metrics.get(key)}" for key in _INCUMBENT_METRIC_ORDER)
            line += f" incumbent={incumbent.get('output_dir')} {surface}"
        lines.append(line)
    recent = rows[-last:] if last > 0 else []
    lines.append(f"last {len(recent)} trials:")
    for row in recent:
        verdict = row.get("verdict") or {}
        improved = ",".join(verdict.get("improved", [])) or "-"
        regressed = ",".join(verdict.get("regressed", [])) or "-"
        lines.append(
            f"#{row.get('trial')} [{row.get('campaign')}] {row.get('status')}"
            f" | {row.get('hypothesis')} | improved:{improved} regressed:{regressed}"
            f" | asi:{row.get('asi') or '-'}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI: write the ledger digest to stdout."""
    parser = argparse.ArgumentParser(prog="autoresearch-summary")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args(argv)
    sys.stdout.write(render_summary(args.ledger, last=args.last))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/autoresearch/test_summary.py -n0 -v`
Expected: 3 PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Then:

```bash
git add src/autoresearch/summary.py tests/autoresearch/test_summary.py
git commit -m "feat(autoresearch): cold-start ledger summary"
```

---

### Task 7: program.md and state files

**Files:**
- Create: `autoresearch/program.md`
- Create: `autoresearch/ideas.md`

**Interfaces:**
- Consumes: every CLI from Tasks 3–6 (the loop steps below reference them by exact invocation).
- Produces: the human-owned research organization code the operator session follows. `autoresearch/ledger.jsonl` is NOT created here — the first `append_row` creates it. `autoresearch/bands.json` is NOT created — absent means zero-width bands. `autoresearch/runs/` is created per trial at runtime.

- [ ] **Step 1: Write program.md**

Create `autoresearch/program.md` with exactly this content:

```markdown
# KD Autoresearch Program

Human-owned research organization code. The operator agent follows this file exactly and never
edits it; proposed program changes go to `ideas.md`. Spec:
`docs/superpowers/specs/2026-08-30-kd-autoresearch-hpo-design.md`.

## Objective

Improve the five V_val topology numbers of the active campaign's KD arm — BFS-macro GS (higher),
RD (closer to 1, judged as |log RD|), degree/clustering/spectral MMD ratios (lower) — measured at
each run's selected epoch. AUPRC is telemetry: always logged, never optimized, never a
keep/revert input.

## Campaigns

kd_logit, kd_rank, kd_gram, kd_rep — one at a time; order set by the human from grid results.
Each campaign starts from its arm's grid winner (human-recorded `baseline` ledger row, picked
from the Pareto-undominated grid points via `src.autoresearch.verdict.undominated`).

## The trial loop

1. Cold start: read this file, then run
   `.venv/bin/python -m src.autoresearch.summary autoresearch/ledger.jsonl`.
2. Propose exactly one hypothesis, citing the previous trial's fit diagnosis, the ledger, and
   `ideas.md`.
3. Edit only whitelisted keys in `configs/autoresearch/<arm>.yaml`; bump `output_dir` to
   `outputs/b1_row_kd_ar/<arm>/trial_NNN`.
4. Commit `ar(<arm>) trial NNN: <hypothesis>` BEFORE launching; push; pull on the container.
5. Launch on the container with the sweep thread caps:
   `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 bash hpc/run.sh train configs/autoresearch/<arm>.yaml --skip-test`.
6. Poll `complete.json` / `failure.json`; pull back `metrics.jsonl`, `run_metadata.json`,
   `complete.json`, `profile.json` into a local mirror of the run directory.
7. Run `.venv/bin/python -m src.autoresearch.curves <run_dir> autoresearch/runs/<arm>/trial_NNN`.
8. Diagnose fit from the CSV and PNG — train vs val task loss, KD terms, grad norms, topology
   trajectory, selected epoch vs loss minimum. Verdict one of overfit / underfit / healthy with
   one sentence of evidence; this goes into the ledger row's `asi`.
9. Run `.venv/bin/python -m src.autoresearch.judge --incumbent <dir> --trial <dir>`
   (add `--bands autoresearch/bands.json` only if the human has created that file).
10. Append the ledger row via `.venv/bin/python -m src.autoresearch.ledger autoresearch/ledger.jsonl <row.json>`;
    commit the row and the curve artifacts together (separate commit from the trial's config
    commit).
11. keep → the incumbent becomes this trial's run. revert → `git revert` the trial's config
    commit.
12. Crash (`failure.json`, or the judge raises): at most one config-level fix commit, relaunched
    as the same trial; a second failure is logged as `crash` (metrics null, both shas in `asi`).
13. Post a one-paragraph digest to the human; continue with the next trial.

## Stage-1 whitelist (config keys the agent may edit)

- `distill.*` (weights, `margin`, `temperature`)
- arm-specific KD model keys (`model.config.kd_rep_dim` for kd_rep)
- all `optim.*` EXCEPT `epochs`
- model regularization keys (`model.config.regularization.*`, `model.config.mlp_head.dropout`,
  `model.config.label_smoothing`)
- `output_dir` (mandatory bump every trial)

Frozen — never edit: `seed`, `optim.epochs`, `eval.*` (including `patience`), `data.*`,
`runtime.*`, `distill.targets_path`, `model.family`, `mixed_precision`.

## Named duties

- The per-trial fit diagnosis (step 8) is mandatory; every hypothesis must cite the latest one.
- Maintain `ideas.md`: deferred hypotheses go in; spent or rejected ones come out.

## Contract

- The verdict comes only from the judge CLI; never argue a trial into a keep.
- V_val only. Test files, `test_report.json`, and the test protocol are out of contract in-loop.
- Only the five frozen topology metrics decide keeps; all other telemetry informs proposals only.
- One hypothesis per trial; no compound edits.
- Never edit: `src/autoresearch/`, this file, `bands.json`, `configs/sweep/`, any frozen key.
- Every trial — keep, revert, crash — gets a ledger row; the ledger is append-only.
- After 5 consecutive non-keeps: post a stall advisory and wait for human re-steer.
- Do not pause to ask whether to continue; stop only at stall advisories and human interrupts.
```

- [ ] **Step 2: Write ideas.md**

Create `autoresearch/ideas.md`:

```markdown
# Ideas backlog

Deferred hypotheses and proposed program.md changes. Maintained by the operator agent; consumed
at proposal time. Nothing here is a commitment.
```

- [ ] **Step 3: Verify the referenced CLIs actually run**

Run each (against no args or `--help`) to confirm the module paths in program.md are real:

```bash
.venv/bin/python -m src.autoresearch.summary --help
.venv/bin/python -m src.autoresearch.judge --help
.venv/bin/python -m src.autoresearch.curves --help
.venv/bin/python -m src.autoresearch.ledger --help
```

Expected: each prints usage and exits 0.

- [ ] **Step 4: Commit**

```bash
git add autoresearch/program.md autoresearch/ideas.md
git commit -m "docs(autoresearch): program.md research organization code and ideas backlog"
```

---

### Task 8: full-suite verification

**Files:** none new.

**Interfaces:** none — this task gates the whole plan.

- [ ] **Step 1: Run the fast test suite**

Run: `.venv/bin/python -m pytest -m "not slow and not integration"`
Expected: ALL PASS (pre-existing failures, if any, must be shown to the human — do not fix unrelated tests in this plan).

- [ ] **Step 2: Run the full gates**

Run: `.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests && .venv/bin/python -m mypy src tests`
Expected: clean. If `ruff format` changed files, re-run the affected test files, then:

```bash
git add -u
git commit -m "style(autoresearch): formatting from full-suite verification"
```

(Skip the commit if nothing changed.)
