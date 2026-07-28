# EgoStitch e2e rev-3.2 — D0 feature-standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generator's per-row `LayerNorm` F0 standardization with registered per-dimension z-scoring computed over the ordered V_fit universe only, so the §14.4.8 slot-collapse guard measures training dynamics instead of an initialization floor.

**Architecture:** One scientific change. `EgoStitchStage1.feature_norm` becomes a `FeatureStandardizer` module holding persistent `mu`/`sigma`/`digest` buffers; the statistics are computed once from the ordered V_fit rows (fp64 accumulation, fp32 canonical values), cached in the feature pack, bound by digest into the config, the access audit, and the checkpoint. SSL feature noise is scaled by the registered σ so it keeps its meaning in standardized coordinates. Everything else — the guard, the probe ABI, raw-F0 pool retrieval, the eight-arm screen, the five G3 gates, the `calibrate → rehearse → formal` harness — is unchanged.

**Tech Stack:** Python 3.11, PyTorch + Accelerate (DDP), NumPy, pytest, ruff, mypy (strict).

**Source of truth:** `docs/superpowers/specs/2026-07-27-egostitch-e2e-rev32-slot-collapse-fix-design.md` (owner-signed 2026-07-27). §3 is the change, §3.2 the statistics contract, §3.3 the pinned non-changes, §4 the deferred ledger (do **not** implement any of it), §8 the experiment suite. This plan implements **S0 only**; S1–S4 are owner-run and appear as a runbook at the end.

## Global Constraints

- **Spec first.** `docs/05-egostitch-spec.md` is frozen; Task 1 lands the spec edit with a §12 change-log line *before* any code change. No code may deviate from the spec text written in Task 1.
- **Do not touch:** §14.4.8 guard criteria/thresholds/arming; `egostitch_e2e_probe_v2` and every array in it; `_e2e_dispersion_rows` and `e2e_dispersion_statistics` return keys (`src/experiments/probes.py:1144` consumes them); `fidelity` dict keys (they flow into `fidelity_series`, checked by `src/experiments/g5_stage1.py:706`); grounding-pool retrieval (stays raw-F0 cosine), `POOL_METHOD_ID`, `_pool_method_hash`, and every pool cache.
- **Do not implement anything from design-doc §4** (L-1 … L-5): no unit-sphere slots, no slot-ID embeddings, no guard replacement, no L_div change, no structural conditioning.
- **Fail closed, never warn-and-recompute.** Follow `src/data/grounding.py:174-192`, not `src/data/features.py:130`.
- Statistics are computed from **V_fit rows only**, and the computation must be provably independent of whether sealed V_qual/V_select rows are present in the loaded matrix.
- Canonical statistics precision: **fp64 accumulation, fp32 canonical output**. The digest is taken over the fp32 bytes that the model actually uses, so a checkpoint's buffers are directly verifiable against the digest.
- Style: `line-length = 100`, ruff `E,W,F,I,B,C4,UP,SIM,ANN,D,T201` with google docstrings, `mypy --strict` over `src` and `tests`. No `print`.
- Tests: `pytestmark = pytest.mark.unit` in every new test file. Do **not** add pytest markers (`--strict-markers` is on). Everything in this plan runs on CPU in seconds.
- Commands (never `uv run` — it garbles output through the shell proxy):
  - `.venv/bin/python -m pytest <paths> -q`
  - `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .`
  - `.venv/bin/python -m mypy src tests`
- Commit after every task. Branch: `g5/e2e-stage1` (current).

## File Structure

| File | Responsibility |
|---|---|
| `docs/05-egostitch-spec.md` | §13.19.1 standardization paragraph, §7/§13.5 SSL-noise sentences, §12 change-log entry (Task 1) |
| `src/data/feature_stats.py` | **New.** Compute / digest / cache / fail-closed load of the registered μ,σ (Task 2) |
| `src/model/egostitch/model.py` | `FeatureStandardizer`, the `feature_standardization` mode kwarg, `set_feature_stats`, SSL-noise scaling (Task 3) |
| `src/model/egostitch/config.py` | `E2EConfig.feature_standardization`, `E2EConfig.feature_stats_sha256`, validation, checkpoint backfill (Task 4) |
| `src/model/egostitch/e2e_model.py` | Thread the mode into the generator; delegate `set_feature_stats` (Task 4) |
| `src/train_egostitch.py` | Pack build + assembly wiring, audit digests, model binding, scale telemetry, `--ddp-mode init-probe` (Tasks 6–9) |
| `configs/egostitch_e2e_v3_*_breadth_first.yaml` | Six arms declare the mode (Task 10) |
| `tests/data/test_feature_stats.py` | **New.** Statistics unit + leakage + fail-closed tests (Task 2) |
| `tests/model/test_egostitch_feature_standardization.py` | **New.** Transform, buffers, SSL noise, pinned-seed init health (Tasks 3, 5) |

---

## Task 1: Spec edit (§13.19.1, §7, §13.5, §12 change log)

**Files:**
- Modify: `docs/05-egostitch-spec.md:1513-1521` (standardization paragraph inside §13.19.1)
- Modify: `docs/05-egostitch-spec.md:259` (§7 `L_ssl` line), `docs/05-egostitch-spec.md:1120` (§13.5 `L_ssl` line)
- Modify: `docs/05-egostitch-spec.md:513` (top of §12 change log — newest entry first)

**Interfaces:**
- Produces: the normative text every later task implements — mode names `row_layernorm` / `zscore_vfit_v1`, the estimator pin, the digest rule, the SSL-noise rule.

- [ ] **Step 1: Replace the §13.19.1 standardization paragraph**

Replace the paragraph beginning "The cached F0 matrix remains the raw fp32 mean pool defined in §9.2." (line 1513) through "...the registered optimizer-group name manifests are unchanged." (line 1521) with:

```markdown
The cached F0 matrix remains the raw fp32 mean pool defined in §9.2. Inside the active
E2E model only, immediately before every trainable generator path, each F0 row is
standardized by registered per-dimension z-scoring `x̃ = (x - mu) / sigma`
(`feature_standardization: zscore_vfit_v1`). The rev-3.1 per-row
`LayerNorm(d, elementwise_affine=False, eps=1e-5)` is retained under the name
`row_layernorm` for replay of rev-3.1 checkpoints only; it may not be selected for a
rev-3.2 or later run. Rationale: per-row LayerNorm cannot remove the shared cross-row
mean direction of F0, so the generator's slot set is born at mean pairwise cosine
0.9897 — above the §14.4.8 trip line — before any training step (design record
2026-07-27 §2).

`mu` and `sigma` are registered constants, not learned parameters:

1. **Scope.** Computed over the ordered V_fit universe only. When the loaded feature
   matrix contains sealed V_qual/V_select rows, the statistics are computed after
   gathering the V_fit rows, and are bit-identical to the statistics of the same rows
   loaded alone.
2. **Estimator.** Two-pass, fp64 accumulation: `mu_j = mean_i x_ij`,
   `var_j = mean_i (x_ij - mu_j)^2` (population, ddof = 0),
   `sigma_j = sqrt(max(var_j, 1e-12))`. The canonical values are the fp32 casts of
   `mu` and `sigma`; a zero or non-finite fp32 `sigma_j` is a hard error.
3. **Identity.** `feature_stats_sha256` = SHA-256 over the method id, the SHA-256 of
   the ordered V_fit node-id list, the dimension, and the fp32 `mu` then `sigma`
   bytes. It is recorded in the pack manifest, in the run access audit, in
   `run_metadata.json`, and as a model buffer; the model config field of the same
   name pins it. A non-empty config value that disagrees with the computed statistics
   aborts the run before the first step.
4. **Storage.** `mu`, `sigma`, and the digest are persistent model buffers, so scoring
   reconstructs the transform from the checkpoint alone — scoring never sees V_fit.
   The same frozen constants apply in every universe; there is no universe-conditional
   preprocessing.

The transform is applied to node features, grounding candidates, reconstruction
targets, denoising targets, and the generator's shared `d -> d_p` projection — one
transformation for all consumers. The raw-token pair encoder and cosine grounding-pool
construction remain on raw F0 and are unchanged; the retrieval/representation mismatch
is deliberate (it preserves every pool cache and the measured P0 pool ceiling that
G3.1 is derived from). Historical frozen-s0 Stage-1 checkpoints retain their original
raw-F0 behavior. Because the transform has no learned parameters, the registered
optimizer-group name manifests are unchanged.
```

- [ ] **Step 2: Amend both `L_ssl` lines for standardized-coordinate noise**

At `docs/05-egostitch-spec.md:259` (§7) and `:1120` (§13.5), keep the σ = 0.05 constant and append the coordinate pin. §7 becomes:

```
L_ssl   = 0.5·consistency(feature noise σ=0.05, in standardized coordinates for the
          E2E family — §13.19.1) + 0.5·pool-resample consistency
          (applied to ungrounded slots only; mean g^k logged)
```

Make the matching edit at §13.5 line 1120. Add one sentence after the §13.5 block:

```markdown
Under `zscore_vfit_v1` the SSL feature perturbation is sampled in standardized
coordinates (equivalently: sampled in raw coordinates and scaled by the registered
per-dimension `sigma` before addition). Adding a raw σ = 0.05 perturbation to raw F0
would be a 5e-5 … 1.9e-3 perturbation per standardized coordinate, since the measured
per-dimension F0 σ spans 26.1 … 1023.3 — the augmentation would silently vanish.
```

- [ ] **Step 3: Add the §12 change-log entry** (insert immediately after the `## 12. Change log` heading at line 513, above the 2026-07-26 entry)

```markdown
- 2026-07-27 (seventh entry): §13.19.1 replaces the E2E generator's per-row
  `LayerNorm` F0 standardization with registered per-dimension z-scoring over
  the ordered V_fit universe (`zscore_vfit_v1`), with a pinned two-pass fp64
  estimator, an fp32 canonical value, a `feature_stats_sha256` identity bound
  into the config, the pack manifest, the access audit, and the checkpoint
  buffers; the rev-3.1 transform is retained as `row_layernorm` for replay
  only. §7/§13.5: the SSL feature perturbation is applied in standardized
  coordinates. Found by the 2026-07-27 diagnosis of the rev-3.1 attempt-001
  `training_invalid(slot_collapse)`: raw F0 rows have mean pairwise cosine
  0.949, per-row LayerNorm cannot remove a shared cross-row mean direction,
  and a randomly initialized generator therefore reads
  `h_pairwise_cosine_mean = 0.9897` — above the §14.4.8 trip line — while its
  transport plan is healthy (mass 0.069, rank-1 residual 0.87). The guard was
  measuring a feature-geometry floor, not training dynamics. Measured effect
  of the edit on a deterministic 300-node real-F0 probe: target-projection
  cosine 0.9405 → −0.0013, init slot cosine 0.9897 → 0.62, ‖h‖ 9.12 → 9.29.
  §14.4.8 (criteria, thresholds, arming), the `egostitch_e2e_probe_v2` schema,
  the eight-arm v3 screen, the five G3 gates, and the §13.12 raw-F0 pool
  contract are all unchanged; new scale diagnostics are training-log only.
  Consequence: `config_hash` changes, so rev-3.1 calibration artifacts are
  inadmissible for rev-3.2 threshold freezing and a new registration version
  is required. Owner-confirmed 2026-07-27.
```

- [ ] **Step 4: Verify no other spec sentence contradicts the new text**

Run: `grep -n "LayerNorm\|per-row\|standardiz" docs/05-egostitch-spec.md`
Expected: only §0/§1 module-internal LayerNorms (lines ~37, decoder blocks), the new §13.19.1 text, line 664 and line 790 (historical change-log prose describing the *prior* transform — leave them; they are dated records). If any *normative* sentence outside §13.19.1 still mandates per-row standardization, fix it here and note it in the change-log entry.

- [ ] **Step 5: Commit**

```bash
git add docs/05-egostitch-spec.md
git commit -m "spec(13.19.1): register per-dimension V_fit z-scoring for the e2e generator"
```

---

## Task 2: Registered statistics module (`src/data/feature_stats.py`)

**Files:**
- Create: `src/data/feature_stats.py`
- Test: `tests/data/test_feature_stats.py`

**Interfaces:**
- Produces:
  - `FEATURE_STATS_METHOD_ID: str = "zscore_vfit_v1"`, `VARIANCE_FLOOR: float = 1e-12`
  - `@dataclass(frozen=True) FeatureStats(mu: NDArray[np.float32], sigma: NDArray[np.float32], method_id: str, node_ids_sha256: str, n_rows: int, digest: str)`
  - `node_ids_sha256(node_ids: Sequence[str]) -> str`
  - `feature_stats_digest(mu, sigma, *, method_id: str, node_ids_sha256: str) -> str`
  - `compute_feature_stats(rows: NDArray[np.float32], node_ids: Sequence[str]) -> FeatureStats`
  - `feature_stats_for_universe(matrix: NDArray[np.float32], node_index: Mapping[str, int], node_ids: Sequence[str], *, cache_path: Path | None = None) -> FeatureStats`
  - `save_feature_stats(stats: FeatureStats, path: Path) -> None`, `load_feature_stats(path: Path, *, expected_node_ids_sha256: str | None = None) -> FeatureStats`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_feature_stats.py`:

```python
"""Tests for src.data.feature_stats: the registered V_fit standardization constants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.feature_stats import (
    FEATURE_STATS_METHOD_ID,
    compute_feature_stats,
    feature_stats_for_universe,
    load_feature_stats,
    node_ids_sha256,
    save_feature_stats,
)

pytestmark = pytest.mark.unit


def _rows(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    gen = np.random.default_rng(seed)
    base = 40.0 * gen.standard_normal(d)
    return (base + 3.0 * gen.standard_normal((n, d))).astype(np.float32)


class TestComputeFeatureStats:
    def test_matches_population_moments(self) -> None:
        rows = _rows(64, 12)
        stats = compute_feature_stats(rows, [f"n{i}" for i in range(64)])

        expected_mu = rows.astype(np.float64).mean(axis=0).astype(np.float32)
        expected_sigma = rows.astype(np.float64).std(axis=0, ddof=0).astype(np.float32)
        np.testing.assert_allclose(stats.mu, expected_mu, rtol=0, atol=1e-4)
        np.testing.assert_allclose(stats.sigma, expected_sigma, rtol=0, atol=1e-4)
        assert stats.method_id == FEATURE_STATS_METHOD_ID
        assert stats.n_rows == 64
        assert stats.node_ids_sha256 == node_ids_sha256([f"n{i}" for i in range(64)])
        assert len(stats.digest) == 64

    def test_canonical_values_are_float32(self) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        assert stats.mu.dtype == np.float32
        assert stats.sigma.dtype == np.float32

    def test_constant_dimension_is_rejected_not_silently_floored(self) -> None:
        rows = _rows(8, 4)
        rows[:, 2] = 7.0
        with pytest.raises(ValueError, match="degenerate"):
            compute_feature_stats(rows, [f"n{i}" for i in range(8)])

    def test_digest_is_sensitive_to_universe_identity(self) -> None:
        rows = _rows(8, 4)
        a = compute_feature_stats(rows, [f"n{i}" for i in range(8)])
        b = compute_feature_stats(rows, [f"m{i}" for i in range(8)])
        assert a.digest != b.digest

    def test_rejects_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="node ids"):
            compute_feature_stats(_rows(8, 4), ["n0", "n1"])


class TestUniverseIsolation:
    def test_sealed_rows_in_the_matrix_do_not_change_the_statistics(self) -> None:
        fit_ids = [f"fit{i}" for i in range(16)]
        fit_rows = _rows(16, 6, seed=1)
        sealed_rows = 500.0 + _rows(24, 6, seed=2)

        fit_only = {node: i for i, node in enumerate(fit_ids)}
        alone = feature_stats_for_universe(fit_rows, fit_only, fit_ids)

        interleaved = np.zeros((40, 6), dtype=np.float32)
        index: dict[str, int] = {}
        for i, node in enumerate(fit_ids):
            interleaved[2 * i] = fit_rows[i]
            index[node] = 2 * i
        sealed_slots = [r for r in range(40) if r not in set(index.values())]
        for slot, row in zip(sealed_slots, sealed_rows, strict=False):
            interleaved[slot] = row
        mixed = feature_stats_for_universe(interleaved, index, fit_ids)

        assert mixed.digest == alone.digest
        np.testing.assert_array_equal(mixed.mu, alone.mu)
        np.testing.assert_array_equal(mixed.sigma, alone.sigma)

    def test_row_order_of_the_universe_list_is_load_bearing_for_identity(self) -> None:
        ids = [f"fit{i}" for i in range(8)]
        rows = _rows(8, 4, seed=3)
        index = {node: i for i, node in enumerate(ids)}
        forward = feature_stats_for_universe(rows, index, ids)
        reversed_ = feature_stats_for_universe(rows, index, list(reversed(ids)))

        np.testing.assert_allclose(forward.mu, reversed_.mu, rtol=0, atol=1e-5)
        assert forward.digest != reversed_.digest


class TestCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        loaded = load_feature_stats(path, expected_node_ids_sha256=stats.node_ids_sha256)

        assert loaded.digest == stats.digest
        np.testing.assert_array_equal(loaded.mu, stats.mu)
        np.testing.assert_array_equal(loaded.sigma, stats.sigma)

    def test_load_fails_closed_on_a_foreign_universe(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        with pytest.raises(ValueError, match="universe"):
            load_feature_stats(path, expected_node_ids_sha256="0" * 64)

    def test_load_fails_closed_on_a_tampered_payload(self, tmp_path: Path) -> None:
        stats = compute_feature_stats(_rows(8, 4), [f"n{i}" for i in range(8)])
        path = tmp_path / "feature_stats.npz"
        save_feature_stats(stats, path)
        payload = dict(np.load(path, allow_pickle=False))
        payload["mu"] = payload["mu"] + np.float32(1.0)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="digest"):
            load_feature_stats(path)

    def test_universe_helper_reuses_a_matching_cache(self, tmp_path: Path) -> None:
        ids = [f"n{i}" for i in range(8)]
        rows = _rows(8, 4)
        index = {node: i for i, node in enumerate(ids)}
        path = tmp_path / "feature_stats.npz"
        first = feature_stats_for_universe(rows, index, ids, cache_path=path)
        assert path.is_file()
        second = feature_stats_for_universe(rows, index, ids, cache_path=path)
        assert second.digest == first.digest

    def test_universe_helper_fails_closed_on_a_stale_cache(self, tmp_path: Path) -> None:
        ids = [f"n{i}" for i in range(8)]
        rows = _rows(8, 4)
        index = {node: i for i, node in enumerate(ids)}
        path = tmp_path / "feature_stats.npz"
        feature_stats_for_universe(rows, index, ids, cache_path=path)
        other = [f"z{i}" for i in range(8)]
        with pytest.raises(ValueError, match="universe"):
            feature_stats_for_universe(rows, {n: i for i, n in enumerate(other)}, other, cache_path=path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_feature_stats.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.feature_stats'`

- [ ] **Step 3: Implement `src/data/feature_stats.py`**

```python
"""Registered per-dimension F0 standardization statistics (spec Sec 13.19.1).

The constants are computed over the ordered V_fit universe only, in fp64, and
are canonically fp32 -- the exact values the model divides by, so a checkpoint's
buffers are directly verifiable against `feature_stats_sha256`.

Fail-closed like `src.data.grounding`, not warn-and-recompute like the F0 cache:
a cached payload that disagrees with the caller's universe or with its own digest
raises. A silently recomputed statistic would change the preprocessing identity
without changing the recorded digest.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FEATURE_STATS_METHOD_ID = "zscore_vfit_v1"
VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class FeatureStats:
    """The registered standardization constants and their identity.

    Attributes:
        mu: Shape ``(d,)`` fp32 per-dimension mean.
        sigma: Shape ``(d,)`` fp32 per-dimension standard deviation.
        method_id: The pinned estimator id.
        node_ids_sha256: Digest of the ordered universe node-id list.
        n_rows: Number of rows the statistics were computed from.
        digest: The `feature_stats_sha256` identity.
    """

    mu: NDArray[np.float32]
    sigma: NDArray[np.float32]
    method_id: str
    node_ids_sha256: str
    n_rows: int
    digest: str


def node_ids_sha256(node_ids: Sequence[str]) -> str:
    """Digest an ordered node-id list exactly as the training access audit does.

    Args:
        node_ids: The ordered universe node ids.

    Returns:
        The 64-hex SHA-256 digest.
    """
    return hashlib.sha256("".join(f"{node}\n" for node in node_ids).encode()).hexdigest()


def feature_stats_digest(
    mu: NDArray[np.float32],
    sigma: NDArray[np.float32],
    *,
    method_id: str,
    node_ids_sha256: str,
) -> str:
    """Compute the `feature_stats_sha256` identity over the fp32 canonical values.

    Args:
        mu: Shape ``(d,)`` fp32 mean.
        sigma: Shape ``(d,)`` fp32 standard deviation.
        method_id: The pinned estimator id.
        node_ids_sha256: Digest of the ordered universe node-id list.

    Returns:
        The 64-hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(f"method\t{method_id}\n".encode())
    digest.update(f"node_ids\t{node_ids_sha256}\n".encode())
    digest.update(f"dim\t{int(mu.shape[0])}\n".encode())
    digest.update(np.ascontiguousarray(mu, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(sigma, dtype=np.float32).tobytes())
    return digest.hexdigest()


def compute_feature_stats(
    rows: NDArray[np.float32], node_ids: Sequence[str]
) -> FeatureStats:
    """Compute the registered constants from one universe's rows.

    Args:
        rows: Shape ``(n, d)`` fp32 F0 rows, aligned with `node_ids`.
        node_ids: The ordered universe node ids.

    Returns:
        The `FeatureStats` bundle.

    Raises:
        ValueError: On a shape/alignment mismatch, fewer than two rows, or a
            dimension whose fp32 sigma is zero or non-finite.
    """
    if rows.ndim != 2:
        raise ValueError("feature stats require a (n, d) row matrix")
    if rows.shape[0] != len(node_ids):
        raise ValueError("feature stats rows and node ids disagree")
    if rows.shape[0] < 2:
        raise ValueError("feature stats require at least two rows")
    accumulator = np.asarray(rows, dtype=np.float64)
    mu64 = accumulator.mean(axis=0)
    var64 = np.square(accumulator - mu64).mean(axis=0)
    sigma64 = np.sqrt(np.maximum(var64, VARIANCE_FLOOR))
    mu = mu64.astype(np.float32)
    sigma = sigma64.astype(np.float32)
    if not bool(np.isfinite(mu).all()) or not bool(np.isfinite(sigma).all()):
        raise ValueError("feature stats are not finite")
    if not bool((sigma > 0.0).all()):
        degenerate = int(np.flatnonzero(sigma <= 0.0)[0])
        raise ValueError(f"degenerate feature dimension {degenerate}: fp32 sigma is zero")
    identity = node_ids_sha256(node_ids)
    return FeatureStats(
        mu=mu,
        sigma=sigma,
        method_id=FEATURE_STATS_METHOD_ID,
        node_ids_sha256=identity,
        n_rows=int(rows.shape[0]),
        digest=feature_stats_digest(
            mu, sigma, method_id=FEATURE_STATS_METHOD_ID, node_ids_sha256=identity
        ),
    )


def save_feature_stats(stats: FeatureStats, path: Path) -> None:
    """Atomically write the constants next to the feature pack.

    Args:
        stats: The computed constants.
        path: Destination ``.npz`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    np.savez(
        temp,
        mu=stats.mu,
        sigma=stats.sigma,
        method_id=np.asarray(stats.method_id),
        node_ids_sha256=np.asarray(stats.node_ids_sha256),
        n_rows=np.asarray(stats.n_rows, dtype=np.int64),
        digest=np.asarray(stats.digest),
    )
    temp.replace(path)


def load_feature_stats(
    path: Path, *, expected_node_ids_sha256: str | None = None
) -> FeatureStats:
    """Load and verify cached constants, failing closed on any disagreement.

    Args:
        path: The ``.npz`` path written by `save_feature_stats`.
        expected_node_ids_sha256: The caller's ordered-universe digest, when known.

    Returns:
        The verified `FeatureStats`.

    Raises:
        ValueError: On a method, digest, or universe mismatch.
    """
    payload = np.load(path, allow_pickle=False)
    method_id = str(payload["method_id"])
    if method_id != FEATURE_STATS_METHOD_ID:
        raise ValueError(
            f"cached feature stats at {path} use method {method_id!r}, "
            f"expected {FEATURE_STATS_METHOD_ID!r}"
        )
    mu = np.ascontiguousarray(payload["mu"], dtype=np.float32)
    sigma = np.ascontiguousarray(payload["sigma"], dtype=np.float32)
    identity = str(payload["node_ids_sha256"])
    stored = str(payload["digest"])
    recomputed = feature_stats_digest(
        mu, sigma, method_id=method_id, node_ids_sha256=identity
    )
    if recomputed != stored:
        raise ValueError(f"cached feature stats at {path} fail their own digest check")
    if expected_node_ids_sha256 is not None and identity != expected_node_ids_sha256:
        raise ValueError(
            f"cached feature stats at {path} were computed over a different universe"
        )
    return FeatureStats(
        mu=mu,
        sigma=sigma,
        method_id=method_id,
        node_ids_sha256=identity,
        n_rows=int(payload["n_rows"]),
        digest=stored,
    )


def feature_stats_for_universe(
    matrix: NDArray[np.float32],
    node_index: Mapping[str, int],
    node_ids: Sequence[str],
    *,
    cache_path: Path | None = None,
) -> FeatureStats:
    """Compute (or load) the constants for one ordered universe of a shared matrix.

    The rows are gathered before any accumulation, so the result is bit-identical
    whether or not sealed universes share the loaded matrix.

    Args:
        matrix: Shape ``(N, d)`` fp32 matrix that contains the universe's rows.
        node_index: Node id -> row of `matrix`.
        node_ids: The ordered universe node ids.
        cache_path: Optional ``.npz`` cache; verified on read, written on miss.

    Returns:
        The `FeatureStats` bundle.
    """
    identity = node_ids_sha256(node_ids)
    if cache_path is not None and cache_path.is_file():
        return load_feature_stats(cache_path, expected_node_ids_sha256=identity)
    rows = np.ascontiguousarray(
        np.asarray(matrix, dtype=np.float32)[[node_index[node] for node in node_ids]],
        dtype=np.float32,
    )
    stats = compute_feature_stats(rows, node_ids)
    if cache_path is not None:
        save_feature_stats(stats, cache_path)
    return stats
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_feature_stats.py -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `.venv/bin/python -m ruff check src/data/feature_stats.py tests/data/test_feature_stats.py && .venv/bin/python -m mypy src/data/feature_stats.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/data/feature_stats.py tests/data/test_feature_stats.py
git commit -m "feat(data): registered V_fit per-dimension standardization constants"
```

---

## Task 3: `FeatureStandardizer` and the generator's standardization mode

**Files:**
- Modify: `src/model/egostitch/model.py:93-97` (the `feature_norm` construction), `:110-116` (`normalize_features` / `project_features`), `:323-350` (`ssl_losses`)
- Modify: `tests/model/test_egostitch_model.py:74-94` (the two existing standardization tests)
- Create: `tests/model/test_egostitch_feature_standardization.py`

**Interfaces:**
- Consumes: `src.data.feature_stats.FeatureStats` (Task 2).
- Produces:
  - `FeatureStandardizationMode = Literal["none", "row_layernorm", "zscore_vfit_v1"]`
  - `class FeatureStandardizer(nn.Module)` with persistent buffers `feature_mu (d,)`, `feature_sigma (d,)`, `feature_stats_ready ()` int64, `feature_stats_digest (32,)` uint8; methods `load_stats(mu, sigma, digest)`, `scale_perturbation(noise)`, property `digest_hex: str`.
  - `EgoStitchStage1.__init__(config, *, feature_standardization: FeatureStandardizationMode = "none", loss_family=...)` — **replaces** the `standardize_features: bool` kwarg.
  - `EgoStitchStage1.set_feature_stats(stats: FeatureStats) -> None`
  - `EgoStitchStage1.feature_stats_digest_hex -> str` (`""` when the mode carries no statistics)
  - `EgoStitchStage1.scale_feature_perturbation(noise: Tensor) -> Tensor`

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_egostitch_feature_standardization.py`:

```python
"""Tests for the rev-3.2 registered per-dimension F0 standardization (spec Sec 13.19.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.data.feature_stats import compute_feature_stats
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.model import EgoStitchStage1, FeatureStandardizer

pytestmark = pytest.mark.unit

_TINY = EgoStitchConfig(
    input_dim=8,
    d_p=4,
    d_z=4,
    d_h=8,
    slots=4,
    m_max=8,
    n_ground=3,
    decoder_layers=2,
    n_heads=2,
    gin_hidden=8,
    gin_layers=2,
    sinkhorn_iters=5,
)


def _stats(seed: int = 0):
    gen = np.random.default_rng(seed)
    rows = (30.0 + 4.0 * gen.standard_normal((32, _TINY.input_dim))).astype(np.float32)
    return compute_feature_stats(rows, [f"n{i}" for i in range(32)])


class TestModes:
    def test_none_is_identity(self) -> None:
        model = EgoStitchStage1(_TINY)
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)
        torch.testing.assert_close(model.normalize_features(x), x)
        assert model.feature_stats_digest_hex == ""

    def test_row_layernorm_preserves_the_rev31_transform(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)
        normalized = model.normalize_features(x)
        torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(3), atol=1e-6, rtol=0.0)
        assert tuple(model.feature_norm.parameters()) == ()

    def test_zscore_applies_registered_constants(self) -> None:
        stats = _stats()
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(stats)
        x = torch.randn(5, _TINY.input_dim)

        expected = (x - torch.from_numpy(stats.mu)) / torch.from_numpy(stats.sigma)
        torch.testing.assert_close(model.normalize_features(x), expected)
        torch.testing.assert_close(
            model.project_features(x), model.proj(model.normalize_features(x))
        )

    def test_zscore_broadcasts_over_grounding_candidates(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(_stats())
        ground = torch.randn(2, _TINY.n_ground, _TINY.input_dim)
        assert model.normalize_features(ground).shape == ground.shape

    def test_zscore_fails_closed_before_statistics_are_registered(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        with pytest.raises(RuntimeError, match="feature standardization statistics"):
            model.normalize_features(torch.randn(2, _TINY.input_dim))

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature standardization"):
            EgoStitchStage1(_TINY, feature_standardization="zscore_v9")  # type: ignore[arg-type]


class TestCheckpointBuffers:
    def test_statistics_survive_a_state_dict_roundtrip(self) -> None:
        stats = _stats(1)
        source = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        source.set_feature_stats(stats)

        target = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        target.load_state_dict(source.state_dict())

        assert target.feature_stats_digest_hex == stats.digest
        x = torch.randn(3, _TINY.input_dim)
        torch.testing.assert_close(target.normalize_features(x), source.normalize_features(x))

    def test_buffers_are_persistent_and_named(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        keys = set(model.state_dict())
        assert {
            "feature_norm.feature_mu",
            "feature_norm.feature_sigma",
            "feature_norm.feature_stats_ready",
            "feature_norm.feature_stats_digest",
        } <= keys

    def test_row_layernorm_checkpoints_carry_no_statistics_buffers(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        assert not [key for key in model.state_dict() if key.startswith("feature_norm.")]

    def test_a_rev31_checkpoint_cannot_be_loaded_as_zscore(self) -> None:
        legacy = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        with pytest.raises(RuntimeError):
            model.load_state_dict(legacy.state_dict())


class TestSslNoiseCoordinates:
    def test_perturbation_is_scaled_into_standardized_coordinates(self) -> None:
        stats = _stats(2)
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(stats)
        x = torch.randn(4, _TINY.input_dim)
        raw_noise = 0.05 * torch.randn(4, _TINY.input_dim)

        perturbed = model.normalize_features(x + model.scale_feature_perturbation(raw_noise))
        expected = model.normalize_features(x) + raw_noise
        torch.testing.assert_close(perturbed, expected, atol=1e-5, rtol=0.0)

    def test_legacy_modes_leave_the_perturbation_untouched(self) -> None:
        for mode in ("none", "row_layernorm"):
            model = EgoStitchStage1(_TINY, feature_standardization=mode)  # type: ignore[arg-type]
            noise = torch.randn(2, _TINY.input_dim)
            torch.testing.assert_close(model.scale_feature_perturbation(noise), noise)


class TestStandardizerUnit:
    def test_load_stats_rejects_a_dimension_mismatch(self) -> None:
        standardizer = FeatureStandardizer(_TINY.input_dim)
        with pytest.raises(ValueError, match="dimension"):
            standardizer.load_stats(torch.zeros(3), torch.ones(3), "ab" * 32)

    def test_load_stats_rejects_a_non_positive_sigma(self) -> None:
        standardizer = FeatureStandardizer(_TINY.input_dim)
        sigma = torch.ones(_TINY.input_dim)
        sigma[0] = 0.0
        with pytest.raises(ValueError, match="sigma"):
            standardizer.load_stats(torch.zeros(_TINY.input_dim), sigma, "ab" * 32)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_egostitch_feature_standardization.py -q`
Expected: FAIL — `ImportError: cannot import name 'FeatureStandardizer'`

- [ ] **Step 3: Implement the module and the mode switch**

In `src/model/egostitch/model.py`, add imports (`from typing import Literal`, `from src.data.feature_stats import FeatureStats`) and insert above `class EgoStitchStage1`:

```python
FeatureStandardizationMode = Literal["none", "row_layernorm", "zscore_vfit_v1"]


class FeatureStandardizer(nn.Module):
    """Registered per-dimension F0 z-scoring (spec Sec 13.19.1).

    The constants are persistent buffers so scoring reconstructs the transform
    from the checkpoint alone; the run aborts rather than standardizing with
    unregistered statistics.
    """

    def __init__(self, dim: int) -> None:
        """Register empty, not-ready statistics buffers.

        Args:
            dim: The F0 dimension.
        """
        super().__init__()
        self.register_buffer("feature_mu", torch.zeros(dim))
        self.register_buffer("feature_sigma", torch.ones(dim))
        self.register_buffer("feature_stats_ready", torch.zeros((), dtype=torch.int64))
        self.register_buffer("feature_stats_digest", torch.zeros(32, dtype=torch.uint8))

    def load_stats(self, mu: torch.Tensor, sigma: torch.Tensor, digest: str) -> None:
        """Pin the registered constants and their identity.

        Args:
            mu: Shape ``(d,)`` per-dimension mean.
            sigma: Shape ``(d,)`` strictly positive per-dimension standard deviation.
            digest: The 64-hex `feature_stats_sha256`.

        Raises:
            ValueError: On a dimension mismatch, a non-positive or non-finite
                sigma, or a malformed digest.
        """
        buffers = (self.feature_mu, self.feature_sigma)
        assert isinstance(buffers[0], torch.Tensor) and isinstance(buffers[1], torch.Tensor)
        if mu.shape != buffers[0].shape or sigma.shape != buffers[1].shape:
            raise ValueError("feature statistics dimension does not match the model")
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
            raise ValueError("feature statistics are not finite")
        if not bool((sigma > 0).all()):
            raise ValueError("feature statistics sigma must be strictly positive")
        if len(digest) != 64:
            raise ValueError("feature statistics digest must be a 64-hex sha256")
        buffers[0].copy_(mu.to(buffers[0].dtype))
        buffers[1].copy_(sigma.to(buffers[1].dtype))
        ready = self.feature_stats_ready
        assert isinstance(ready, torch.Tensor)
        ready.fill_(1)
        recorded = self.feature_stats_digest
        assert isinstance(recorded, torch.Tensor)
        recorded.copy_(torch.frombuffer(bytearray.fromhex(digest), dtype=torch.uint8))

    @property
    def digest_hex(self) -> str:
        """The registered `feature_stats_sha256`, or ``""`` when unset."""
        ready = self.feature_stats_ready
        recorded = self.feature_stats_digest
        assert isinstance(ready, torch.Tensor) and isinstance(recorded, torch.Tensor)
        if int(ready) != 1:
            return ""
        return bytes(recorded.cpu().numpy().tobytes()).hex()

    def scale_perturbation(self, noise: torch.Tensor) -> torch.Tensor:
        """Scale a raw-coordinate perturbation into standardized coordinates.

        Args:
            noise: Raw-space noise broadcastable to ``(..., d)``.

        Returns:
            ``noise * sigma`` -- adding it before standardization is exactly a
            `noise`-sized perturbation of the standardized row.
        """
        sigma = self.feature_sigma
        assert isinstance(sigma, torch.Tensor)
        return noise * sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize an F0 tensor in fp32.

        Args:
            features: Shape ``(..., d)`` raw F0 values.

        Returns:
            The standardized fp32 tensor.

        Raises:
            RuntimeError: When the statistics have not been registered.
        """
        ready = self.feature_stats_ready
        assert isinstance(ready, torch.Tensor)
        if int(ready) != 1:
            raise RuntimeError(
                "feature standardization statistics are not registered; "
                "the trainer must call set_feature_stats before the first forward"
            )
        mu = self.feature_mu
        sigma = self.feature_sigma
        assert isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor)
        return (features.float() - mu) / sigma
```

Replace the `__init__` kwarg and the `feature_norm` construction (`model.py:75-97`):

```python
    def __init__(
        self,
        config: EgoStitchConfig,
        *,
        feature_standardization: FeatureStandardizationMode = "none",
        loss_family: LossFamily = "egostitch",
    ) -> None:
        """Build every Stage-1 module.

        Args:
            config: The pinned Stage-1 configuration.
            feature_standardization: ``"none"`` for the legacy frozen-s0 raw-F0
                semantics, ``"row_layernorm"`` for the rev-3.1 stateless per-row
                transform (replay only), ``"zscore_vfit_v1"`` for the registered
                rev-3.2 per-dimension constants (spec Sec 13.19.1).
            loss_family: Select the frozen Stage-1 or rev-3.1 E2E loss path.
        """
        super().__init__()
        self.config = config
        self.loss_family = loss_family
        self.feature_standardization = feature_standardization
        self.feature_norm: nn.Module
        if feature_standardization == "none":
            self.feature_norm = nn.Identity()
        elif feature_standardization == "row_layernorm":
            self.feature_norm = nn.LayerNorm(config.input_dim, elementwise_affine=False)
        elif feature_standardization == "zscore_vfit_v1":
            self.feature_norm = FeatureStandardizer(config.input_dim)
        else:
            raise ValueError(
                f"unknown feature standardization mode {feature_standardization!r}"
            )
```

Add the three accessors after `normalize_features`:

```python
    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered standardization constants on this model.

        Args:
            stats: The V_fit statistics bundle.

        Raises:
            TypeError: When the configured mode carries no statistics.
        """
        if not isinstance(self.feature_norm, FeatureStandardizer):
            raise TypeError(
                f"feature standardization mode {self.feature_standardization!r} "
                "does not accept registered statistics"
            )
        self.feature_norm.load_stats(
            torch.from_numpy(stats.mu), torch.from_numpy(stats.sigma), stats.digest
        )

    @property
    def feature_stats_digest_hex(self) -> str:
        """The registered `feature_stats_sha256`, or ``""`` for stateless modes."""
        if not isinstance(self.feature_norm, FeatureStandardizer):
            return ""
        return self.feature_norm.digest_hex

    def scale_feature_perturbation(self, noise: torch.Tensor) -> torch.Tensor:
        """Map a raw-coordinate SSL perturbation into the active coordinates.

        Args:
            noise: Raw-space noise shaped ``(B, d)``.

        Returns:
            The noise the caller should add before standardization (spec Sec 7).
        """
        if not isinstance(self.feature_norm, FeatureStandardizer):
            return noise
        return self.feature_norm.scale_perturbation(noise)
```

In `ssl_losses` (`model.py:345`), replace `enc_noise = self.encode_nodes(x + noise, ground_x)` with:

```python
        enc_noise = self.encode_nodes(x + self.scale_feature_perturbation(noise), ground_x)
```

and extend that method's `noise` docstring line with "(raw F0 coordinates; scaled into standardized coordinates by `scale_feature_perturbation`)".

- [ ] **Step 4: Update the two existing standardization tests**

In `tests/model/test_egostitch_model.py`, rename `test_e2e_generator_inputs_use_stateless_per_row_normalization` to `test_row_layernorm_mode_keeps_the_rev31_per_row_transform` and change its first line to `model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")`. Leave `test_legacy_generator_keeps_raw_feature_semantics` as is (default mode is still `"none"`).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/model/test_egostitch_feature_standardization.py tests/model/test_egostitch_model.py -q`
Expected: PASS

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src/model/egostitch/model.py
git add src/model/egostitch/model.py tests/model/test_egostitch_feature_standardization.py tests/model/test_egostitch_model.py
git commit -m "feat(model): registered per-dimension feature standardization with fail-closed stats"
```

---

## Task 4: Config fields, checkpoint back-compat, and the E2E thread-through

**Files:**
- Modify: `src/model/egostitch/config.py:18-32` (`e2e_checkpoint_config`), `:210-262` (`E2EConfig` fields), `:264-312` (`__post_init__`), `:314-328` (`from_mapping` string fields)
- Modify: `src/model/egostitch/e2e_model.py:113-117` (generator construction), plus a `set_feature_stats` delegate
- Modify: `tests/model/test_egostitch_modules.py` (`TestConfig`), `tests/model/test_egostitch_e2e_model.py`

**Interfaces:**
- Consumes: `FeatureStandardizationMode`, `EgoStitchStage1.set_feature_stats` (Task 3).
- Produces:
  - `E2EConfig.feature_standardization: str = "zscore_vfit_v1"` (allowlist `{"row_layernorm", "zscore_vfit_v1"}`)
  - `E2EConfig.feature_stats_sha256: str = ""` (empty or 64-hex lowercase)
  - `EgoStitchE2E.set_feature_stats(stats: FeatureStats) -> None`, `EgoStitchE2E.feature_stats_digest_hex -> str`

- [ ] **Step 1: Write the failing tests**

Add to `tests/model/test_egostitch_modules.py` inside `class TestConfig`:

```python
    def test_feature_standardization_defaults_to_the_rev32_mode(self) -> None:
        assert E2EConfig().feature_standardization == "zscore_vfit_v1"
        assert E2EConfig().feature_stats_sha256 == ""

    def test_feature_standardization_allowlist(self) -> None:
        assert E2EConfig(feature_standardization="row_layernorm").feature_standardization == (
            "row_layernorm"
        )
        with pytest.raises(ValueError, match="feature_standardization"):
            E2EConfig(feature_standardization="layer_norm")

    def test_feature_stats_sha256_must_be_empty_or_64_hex(self) -> None:
        assert E2EConfig(feature_stats_sha256="ab" * 32).feature_stats_sha256 == "ab" * 32
        with pytest.raises(ValueError, match="feature_stats_sha256"):
            E2EConfig(feature_stats_sha256="deadbeef")

    def test_from_mapping_accepts_the_string_fields(self) -> None:
        cfg = E2EConfig.from_mapping(
            {"feature_standardization": "row_layernorm", "feature_stats_sha256": "cd" * 32}
        )
        assert cfg.feature_standardization == "row_layernorm"

    def test_checkpoint_config_backfills_rev31_as_row_layernorm(self) -> None:
        restored = e2e_checkpoint_config({"n_ground": 50}, has_rel_head=False)
        assert restored["feature_standardization"] == "row_layernorm"
        assert restored["feature_stats_sha256"] == ""

    def test_checkpoint_config_preserves_an_explicit_mode(self) -> None:
        restored = e2e_checkpoint_config(
            {"feature_standardization": "zscore_vfit_v1"}, has_rel_head=False
        )
        assert restored["feature_standardization"] == "zscore_vfit_v1"
```

Add to `tests/model/test_egostitch_e2e_model.py`:

```python
    def test_generator_uses_the_configured_standardization_mode(self) -> None:
        model = EgoStitchE2E(E2EConfig(feature_standardization="zscore_vfit_v1"))
        assert model.generator.feature_standardization == "zscore_vfit_v1"

    def test_set_feature_stats_reaches_the_generator(self) -> None:
        gen = np.random.default_rng(0)
        cfg = E2EConfig(feature_standardization="zscore_vfit_v1")
        model = EgoStitchE2E(cfg)
        rows = (30.0 + 4.0 * gen.standard_normal((32, model.generator_cfg.input_dim))).astype(
            np.float32
        )
        stats = compute_feature_stats(rows, [f"n{i}" for i in range(32)])
        model.set_feature_stats(stats)
        assert model.feature_stats_digest_hex == stats.digest
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/model/test_egostitch_modules.py tests/model/test_egostitch_e2e_model.py -q`
Expected: FAIL — `TypeError: E2EConfig.__init__() got an unexpected keyword argument 'feature_standardization'`

- [ ] **Step 3: Add the config fields**

In `src/model/egostitch/config.py`, add to `E2EConfig` (after `permanent_null`, keeping the docstring convention):

```python
    feature_standardization: str = "zscore_vfit_v1"
    feature_stats_sha256: str = ""
```

Append to `E2EConfig.__post_init__`:

```python
        if self.feature_standardization not in _FEATURE_STANDARDIZATION_MODES:
            raise ValueError(
                "feature_standardization must be one of "
                f"{sorted(_FEATURE_STANDARDIZATION_MODES)}"
            )
        if self.feature_stats_sha256 and (
            len(self.feature_stats_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.feature_stats_sha256)
        ):
            raise ValueError("feature_stats_sha256 must be empty or a 64-hex lowercase sha256")
```

with the module-level constant `_FEATURE_STANDARDIZATION_MODES = frozenset({"row_layernorm", "zscore_vfit_v1"})`.

Extend `E2EConfig.from_mapping`'s `string_fields` to
`frozenset({"permanent_null", "feature_standardization", "feature_stats_sha256"})`.

Extend `e2e_checkpoint_config` with the back-compat defaults (rev-3.1 checkpoints replay as `row_layernorm`; document that in the function docstring):

```python
    restored.setdefault("feature_standardization", "row_layernorm")
    restored.setdefault("feature_stats_sha256", "")
```

- [ ] **Step 4: Thread the mode through `EgoStitchE2E`**

In `src/model/egostitch/e2e_model.py`, change the generator construction (`:113-117`) to pass
`feature_standardization=cfg.feature_standardization` in place of `standardize_features=True`, then add:

```python
    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered F0 standardization constants on the generator.

        Args:
            stats: The V_fit statistics bundle.
        """
        self.generator.set_feature_stats(stats)

    @property
    def feature_stats_digest_hex(self) -> str:
        """The generator's registered `feature_stats_sha256`, or ``""``."""
        return self.generator.feature_stats_digest_hex
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/model/ -q`
Expected: PASS

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src/model tests/model
git add src/model/egostitch/config.py src/model/egostitch/e2e_model.py tests/model
git commit -m "feat(config): pin the e2e feature-standardization mode and stats digest"
```

---

## Task 5: Pinned-seed init-health regression test

**Files:**
- Modify: `tests/model/test_egostitch_feature_standardization.py` (add the class below)

**Interfaces:**
- Consumes: Task 3's modes; `src.model.egostitch.stitch.sinkhorn_plan`; `src.train_egostitch.e2e_dispersion_statistics` (unchanged ABI).

This is the committed §3.4 regression: it reproduces the failure geometry and pins the fix, measuring **both** modes on the same fixture so a future regression is attributable.

- [ ] **Step 1: Write the test**

```python
class TestInitHealthUnderAnisotropicFeatures:
    """Spec Sec 13.19.1: a random model must not be born above the guard line."""

    @staticmethod
    def _config() -> EgoStitchConfig:
        return EgoStitchConfig(
            input_dim=64,
            d_p=32,
            d_z=16,
            d_h=32,
            slots=16,
            m_max=8,
            n_ground=50,
            decoder_layers=2,
            n_heads=4,
            gin_hidden=16,
            gin_layers=2,
            sinkhorn_iters=20,
        )

    @staticmethod
    def _features(cfg: EgoStitchConfig, *, seed: int = 0) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        gen = np.random.default_rng(seed)
        direction = np.abs(gen.standard_normal(cfg.input_dim)) + 1.0
        rows = (40.0 * direction + 4.0 * gen.standard_normal((256, cfg.input_dim))).astype(
            np.float32
        )
        x = torch.from_numpy(rows[:8])
        pool = torch.from_numpy(
            np.stack([rows[gen.choice(256, cfg.n_ground, replace=False)] for _ in range(8)])
        )
        return rows, x, pool

    def _dispersion(self, mode: str, *, seed: int = 0) -> dict[str, float]:
        from src.model.egostitch.stitch import sinkhorn_plan
        from src.train_egostitch import e2e_dispersion_statistics

        cfg = self._config()
        rows, x, pool = self._features(cfg, seed=seed)
        torch.manual_seed(11)
        model = EgoStitchStage1(cfg, feature_standardization=mode)  # type: ignore[arg-type]
        if mode == "zscore_vfit_v1":
            model.set_feature_stats(
                compute_feature_stats(rows, [f"n{i}" for i in range(rows.shape[0])])
            )
        model.eval()
        with torch.no_grad():
            enc = model.encode_nodes(x, pool)
            plan = sinkhorn_plan(
                enc.slots.h,
                enc.slots.h,
                enc.slots.pi,
                enc.slots.pi,
                enc.slots.mult,
                enc.slots.mult,
                eps=cfg.sinkhorn_eps,
                iters=cfg.sinkhorn_iters,
                tau=cfg.sinkhorn_tau,
            )
            stats = e2e_dispersion_statistics(enc.slots.pi, enc.slots.h, enc.slots.adj, plan)
            stats["plan_total_mass"] = float(plan.float().sum(dim=(1, 2)).mean())
        return stats

    def test_the_fixture_reproduces_the_measured_f0_anisotropy(self) -> None:
        cfg = self._config()
        rows, _, _ = self._features(cfg)
        normalized = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        gram = normalized @ normalized.T
        off_diagonal = gram[~np.eye(rows.shape[0], dtype=bool)]
        assert float(off_diagonal.mean()) > 0.9

    def test_row_layernorm_is_born_above_the_guard_line(self) -> None:
        assert self._dispersion("row_layernorm")["h_pairwise_cosine_mean"] > 0.95

    def test_zscore_is_born_healthy_on_every_guard_criterion(self) -> None:
        stats = self._dispersion("zscore_vfit_v1")
        assert stats["h_pairwise_cosine_mean"] < 0.9
        assert stats["plan_rank1_marginal_residual"] > 0.3
        assert stats["plan_total_mass"] > 1e-6
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/model/test_egostitch_feature_standardization.py::TestInitHealthUnderAnisotropicFeatures -q`
Expected: PASS.

**If `test_row_layernorm_is_born_above_the_guard_line` fails**, the fixture is not anisotropic enough to reproduce the real-F0 geometry — raise the `40.0` shared-direction scale (or lower the `4.0` per-row noise) until the raw-cosine precondition test reports > 0.94, and re-run. Do **not** weaken the two `zscore` thresholds: they are the spec's §3.4 numbers.

**If `test_zscore_is_born_healthy...` fails**, stop and report — that is D0 failing its own bench, and it is a finding for the owner, not a threshold to tune.

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_egostitch_feature_standardization.py
git commit -m "test(model): pin rev-3.2 init health against the measured F0 anisotropy"
```

---

## Task 6: Compute the statistics in the pack build and the data assembly

**Files:**
- Modify: `src/train_egostitch.py:103-108` (pack filename constants), `:1775-1798` (pack build), `:2194-2207` (assembly, immediately after `fit_rows`), `:2263-2295` (access audit), `:2021-2060` (`EgoStitchData`)
- Test: `tests/test_train_egostitch_e2e.py` (new tests near the existing assembly tests)

**Interfaces:**
- Consumes: `feature_stats_for_universe`, `FeatureStats` (Task 2).
- Produces:
  - `_PACK_FEATURE_STATS_FILENAME = "feature_stats.npz"`
  - `EgoStitchData.feature_stats: FeatureStats | None = None`
  - audit keys `training_feature_stats_sha256`, `training_feature_stats_universe_sha256`, `training_feature_stats_rows`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train_egostitch_e2e.py` (reuse whatever fixture the file already uses to call `assemble_egostitch_data`; grep for `assemble_egostitch_data(` in that file and copy the closest existing test's setup):

```python
def test_assembly_registers_v_fit_feature_statistics(tmp_path: Path) -> None:
    cfg = _e2e_config(tmp_path)  # existing helper in this file
    data = te.assemble_egostitch_data(cfg)

    assert data.feature_stats is not None
    audit = data.access_audit or {}
    assert audit["training_feature_stats_sha256"] == data.feature_stats.digest
    # The statistics universe is exactly the audited V_fit id list.
    assert (
        audit["training_feature_stats_universe_sha256"]
        == audit["training_feature_nodes_sha256"]
    )
    assert data.feature_stats.n_rows == len(data.train_nodes)


def test_assembly_statistics_ignore_sealed_validation_rows(tmp_path: Path) -> None:
    """The loaded matrix carries V_select rows; the constants must not see them."""
    cfg = _e2e_config(tmp_path)
    data = te.assemble_egostitch_data(cfg)
    assert data.feature_stats is not None
    assert data.validation_nodes  # precondition: sealed rows really are in the matrix

    expected = compute_feature_stats(
        np.asarray(
            data.f0.numpy()[[data.node_index[node] for node in data.train_nodes]],
            dtype=np.float32,
        ),
        data.train_nodes,
    )
    assert data.feature_stats.digest == expected.digest
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py -q -k feature_statistics`
Expected: FAIL — `AttributeError: 'EgoStitchData' object has no attribute 'feature_stats'`

- [ ] **Step 3: Add the field and the constant**

`src/train_egostitch.py`: import `from src.data.feature_stats import FeatureStats, feature_stats_for_universe`; add
`_PACK_FEATURE_STATS_FILENAME = "feature_stats.npz"` beside the other `_PACK_*` constants; add to `EgoStitchData`
(after `overfit_manifest`, with a docstring line):

```python
    feature_stats: FeatureStats | None = None
```

- [ ] **Step 4: Compute in the assembly**

Immediately after `fit_rows` is materialized (`:2195-2197`) and **before** `build_grounding_pool`:

```python
    feature_stats_cache = (
        (pack_dir / _PACK_FEATURE_STATS_FILENAME)
        if pack_dir is not None
        else cfg.data.f0_cache.with_name(_PACK_FEATURE_STATS_FILENAME)
    )
    feature_stats = feature_stats_for_universe(
        np.asarray(matrix.numpy(), dtype=np.float32),
        node_index,
        fit_nodes,
        cache_path=feature_stats_cache,
    )
```

Pass `feature_stats=feature_stats` in the `EgoStitchData(...)` construction.

- [ ] **Step 5: Audit the identity**

In the `audit` dict (`:2263-2295`) add:

```python
        "training_feature_stats_sha256": feature_stats.digest,
        "training_feature_stats_universe_sha256": feature_stats.node_ids_sha256,
        "training_feature_stats_rows": feature_stats.n_rows,
```

and immediately after the dict literal, add the V_fit-only proof:

```python
    if audit["training_feature_stats_universe_sha256"] != audit["training_feature_nodes_sha256"]:
        raise RuntimeError(
            "feature standardization statistics were computed over a universe other than V_fit"
        )
```

- [ ] **Step 6: Write the statistics into the pack**

In the pack build (`:1775-1798`), after `train_rows` is built and before the `files` list is closed:

```python
        feature_stats_for_universe(
            np.asarray(matrix.numpy(), dtype=np.float32),
            index,
            train_nodes,
            cache_path=pack_dir / _PACK_FEATURE_STATS_FILENAME,
        )
        files = [
            _PACK_F0_FILENAME,
            _PACK_GROUNDING_FILENAME,
            _PACK_FEATURE_STATS_FILENAME,
        ]
```

(The existing `files` list at `:1785` gains the new name so the manifest's per-file sha256 drift check and the `packs_and_validation_manifests` binding section cover it.)

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py tests/test_train_egostitch.py -q`
Expected: PASS. A pack-manifest test that hardcodes the old two-file list will fail — update its expectation to include `feature_stats.npz`; do not remove the assertion.

- [ ] **Step 8: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src tests
git add src/train_egostitch.py tests
git commit -m "feat(train): compute and audit the V_fit standardization statistics"
```

---

## Task 7: Bind the statistics to the model, fail closed on the digest

**Files:**
- Modify: `src/train_egostitch.py:5904-5915` (model construction in `_run_ddp_worker`), `:5480-5495` (run-start metadata)
- Test: `tests/test_train_egostitch_e2e.py`

**Interfaces:**
- Consumes: `EgoStitchData.feature_stats` (Task 6), `EgoStitchE2E.set_feature_stats` (Task 4).
- Produces: `_bind_feature_standardization(model: EgoStitchE2E, cfg: EgoConfig, data: EgoStitchData) -> str` — returns the bound digest (`""` for `row_layernorm`); `run_metadata.json` key `feature_stats_sha256`.

- [ ] **Step 1: Write the failing tests**

```python
class TestFeatureStandardizationBinding:
    def test_binding_pins_the_statistics_and_returns_the_digest(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        data = te.assemble_egostitch_data(cfg)
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))

        digest = te._bind_feature_standardization(model, cfg, data)

        assert data.feature_stats is not None
        assert digest == data.feature_stats.digest
        assert model.feature_stats_digest_hex == digest

    def test_binding_fails_closed_on_a_pinned_digest_mismatch(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                config={**cfg.model.config, "feature_stats_sha256": "ab" * 32},
            ),
        )
        data = te.assemble_egostitch_data(cfg)
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))

        with pytest.raises(RuntimeError, match="feature_stats_sha256"):
            te._bind_feature_standardization(model, cfg, data)

    def test_binding_fails_closed_when_statistics_are_absent(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        data = replace(te.assemble_egostitch_data(cfg), feature_stats=None)
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))

        with pytest.raises(RuntimeError, match="statistics"):
            te._bind_feature_standardization(model, cfg, data)

    def test_row_layernorm_binds_nothing(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                config={**cfg.model.config, "feature_standardization": "row_layernorm"},
            ),
        )
        data = te.assemble_egostitch_data(cfg)
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))

        assert te._bind_feature_standardization(model, cfg, data) == ""
```

(`EgoStitchData` is a non-frozen dataclass, so `replace` works; if the helper `_e2e_config` in that file has a different name, use the one the file already defines.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py -q -k FeatureStandardizationBinding`
Expected: FAIL — `AttributeError: module 'src.train_egostitch' has no attribute '_bind_feature_standardization'`

- [ ] **Step 3: Implement the binder**

Add near the other `_run_ddp_worker` helpers in `src/train_egostitch.py`:

```python
def _bind_feature_standardization(
    model: EgoStitchE2E, cfg: EgoConfig, data: EgoStitchData
) -> str:
    """Pin the registered F0 statistics on the model before the first step.

    Args:
        model: The freshly constructed E2E model.
        cfg: The loaded run configuration.
        data: The assembled training data, carrying the V_fit statistics.

    Returns:
        The bound `feature_stats_sha256`, or ``""`` for the replay-only
        `row_layernorm` mode.

    Raises:
        RuntimeError: When the registered mode has no statistics available, or
            when the config pins a digest that disagrees with them.
    """
    mode = str(cfg.model.config.get("feature_standardization", "zscore_vfit_v1"))
    if mode != "zscore_vfit_v1":
        return ""
    stats = data.feature_stats
    if stats is None:
        raise RuntimeError(
            "feature standardization statistics are unavailable; "
            "rebuild the feature pack before training"
        )
    pinned = str(cfg.model.config.get("feature_stats_sha256", ""))
    if pinned and pinned != stats.digest:
        raise RuntimeError(
            "feature_stats_sha256 mismatch: config pins "
            f"{pinned}, assembled statistics are {stats.digest}"
        )
    model.set_feature_stats(stats)
    logger.info(
        "registered feature standardization mode=%s rows=%d feature_stats_sha256=%s pinned=%s",
        mode,
        stats.n_rows,
        stats.digest,
        "yes" if pinned else "no",
    )
    return stats.digest
```

Call it immediately after the model is built (`:5911-5914`):

```python
    feature_stats_sha256 = ""
    if isinstance(model, EgoStitchE2E):
        feature_stats_sha256 = _bind_feature_standardization(model, cfg, data)
```

- [ ] **Step 4: Record it in run metadata**

Thread `feature_stats_sha256` into `write_run_start_metadata` (`:5480-5495`) beside `config_hash`, as a top-level `"feature_stats_sha256"` key. Do not add it to `fidelity` or to any probe artifact.

- [ ] **Step 5: Run tests, lint, typecheck**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src tests`
Expected: PASS / clean

- [ ] **Step 6: Commit**

```bash
git add src/train_egostitch.py tests/test_train_egostitch_e2e.py
git commit -m "feat(train): fail-closed binding of the registered standardization statistics"
```

---

## Task 8: Scale telemetry on the training-log line only

**Files:**
- Modify: `src/train_egostitch.py:823-873` (add `_e2e_scale_rows` beside `_e2e_dispersion_rows`), `:3491-3494` (`_ValidationResult`), `:3684-3721` (per-row stack), `:3770` (`n_cols`), `:3808-3830` (summaries), `:4598-4606` (the telemetry log line)
- Test: `tests/test_train_egostitch_training.py`, `tests/test_train_egostitch_e2e.py`

**Interfaces:**
- Produces: `_e2e_scale_rows(h: Tensor, plan: Tensor) -> dict[str, Tensor]` with keys `plan_total_mass`, `plan_max_cell_fraction`, `h_norm_mean`, `h_pairwise_sqdist_mean`; `_ValidationResult.scale_telemetry: dict[str, float]`.

**Hard constraint:** `_e2e_dispersion_rows`, `e2e_dispersion_statistics`, and the `fidelity` dict keep **exactly** their current keys (`src/experiments/probes.py:1144` and `src/experiments/g5_stage1.py:706` bind them). The new numbers ride a separate field and reach only the log.

- [ ] **Step 1: Write the failing tests**

```python
class TestScaleTelemetry:
    def test_scale_rows_report_plan_scale_and_slot_geometry(self) -> None:
        torch.manual_seed(0)
        h = torch.randn(3, 4, 6)
        plan = torch.rand(3, 4, 4)
        rows = te._e2e_scale_rows(h, plan)

        assert set(rows) == {
            "plan_total_mass",
            "plan_max_cell_fraction",
            "h_norm_mean",
            "h_pairwise_sqdist_mean",
        }
        torch.testing.assert_close(rows["plan_total_mass"], plan.sum(dim=(1, 2)))
        torch.testing.assert_close(
            rows["h_norm_mean"], torch.linalg.vector_norm(h, dim=-1).mean(dim=-1)
        )
        assert bool((rows["plan_max_cell_fraction"] <= 1.0).all())

    def test_pairwise_squared_distance_matches_the_direct_computation(self) -> None:
        h = torch.randn(2, 5, 3)
        direct = ((h[:, :, None, :] - h[:, None, :, :]) ** 2).sum(-1)
        upper = torch.triu_indices(5, 5, offset=1)
        expected = direct[:, upper[0], upper[1]].mean(dim=-1)
        torch.testing.assert_close(
            te._e2e_scale_rows(h, torch.rand(2, 5, 5))["h_pairwise_sqdist_mean"],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_dispersion_keys_are_unchanged(self) -> None:
        """The probe ABI and fidelity_series bind these five names exactly."""
        rows = te._e2e_dispersion_rows(
            torch.rand(2, 4), torch.randn(2, 4, 6), torch.rand(2, 4, 4), torch.rand(2, 4, 4)
        )
        assert set(rows) == {
            "pi_slot_std",
            "h_pairwise_cosine_mean",
            "adj_offdiag_std",
            "plan_row_entropy",
            "plan_rank1_marginal_residual",
        }
```

Plus a column-indexing regression on the validation packing — extend the existing
`test_profile_loop_executes_real_optimizer_and_validation`
(`tests/test_train_egostitch_e2e.py:1225`) with:

```python
        fidelity = history[0]["fidelity"]
        assert set(fidelity) == _EXPECTED_FIDELITY_KEYS  # define from the current keys, verbatim
        assert 0.0 <= fidelity["h_pairwise_cosine_mean"] <= 1.0
```

`_EXPECTED_FIDELITY_KEYS` must be written out literally from the current implementation — this is
the test that catches the `ordered[:, 3:]` slice picking up the four new columns.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_training.py -q -k ScaleTelemetry`
Expected: FAIL — `AttributeError: ... has no attribute '_e2e_scale_rows'`

- [ ] **Step 3: Implement `_e2e_scale_rows`**

Insert after `_e2e_dispersion_rows` in `src/train_egostitch.py`:

```python
def _e2e_scale_rows(h: torch.Tensor, plan: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-row OT-scale diagnostics (training log only, never an artifact).

    The Sinkhorn stage is numerically dead when squared slot distances greatly
    exceed `eps` and diffuse-rank-1 when they are far below it (design record
    2026-07-27 R3). These four numbers say which regime a run is in.

    Args:
        h: Shape ``(B, K, D)`` slot embeddings.
        plan: Shape ``(B, K, K)`` transport plan.

    Returns:
        Per-row ``plan_total_mass``, ``plan_max_cell_fraction``, ``h_norm_mean``,
        ``h_pairwise_sqdist_mean``.
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        h32 = h.float()
        plan32 = plan.float()
        slots = h32.shape[1]
        mass = plan32.sum(dim=(1, 2))
        max_cell = plan32.amax(dim=(1, 2)) / mass.clamp_min(1e-30)
        square = h32.square().sum(dim=-1)
        distance = (
            square[:, :, None] + square[:, None, :] - 2.0 * torch.bmm(h32, h32.transpose(1, 2))
        ).clamp_min(0.0)
        upper = torch.triu_indices(slots, slots, offset=1, device=h.device)
        return {
            "plan_total_mass": mass,
            "plan_max_cell_fraction": max_cell,
            "h_norm_mean": torch.linalg.vector_norm(h32, dim=-1).mean(dim=-1),
            "h_pairwise_sqdist_mean": distance[:, upper[0], upper[1]].mean(dim=-1),
        }
```

- [ ] **Step 4: Carry four extra columns through validation**

In `_validate_epoch`'s e2e branch (`:3684-3721`): compute `scale_a = _e2e_scale_rows(state_a.slots.h, context.plan)` and `scale_b = _e2e_scale_rows(state_b.slots.h, context.plan)`; average the two `h_*` entries, take `scale_a` for the two `plan_*` entries, and NaN-mask both `plan_*` entries on `is_self` exactly as the dispersion rows are masked. Append the four columns to the `torch.stack([...])` **after** `plan_rank1_marginal_residual`, in the fixed order `plan_total_mass, plan_max_cell_fraction, h_norm_mean, h_pairwise_sqdist_mean`.

Set `n_cols = 12 if is_e2e else 5` (`:3770`).

In the summary block (`:3808-3815`) change the dispersion slice from `ordered[:, 3:]` to **`ordered[:, 3:8]`** — this is load-bearing; leaving `3:` silently zips five names against nine columns. Then add:

```python
        scale_names = (
            "plan_total_mass",
            "plan_max_cell_fraction",
            "h_norm_mean",
            "h_pairwise_sqdist_mean",
        )
        scale_telemetry = {
            name: (
                float(np.mean(values[np.isfinite(values)]))
                if bool(np.isfinite(values).any())
                else float("nan")
            )
            for name, values in zip(scale_names, ordered[:, 8:12].T, strict=True)
        }
```

(Note the `nan` fallback, not `0.0`: these are diagnostics, and a fabricated zero would read as a dead plan.)

Add `scale_telemetry: dict[str, float] = field(default_factory=dict)` to `_ValidationResult` (`:3491-3494`, keep it last since the dataclass is frozen) and pass it at `:3836`. The non-e2e branch leaves it empty.

- [ ] **Step 5: Extend the telemetry log line**

At `:4598-4606`, extend the existing `logger.log(...)` call with the four values from `validation.scale_telemetry` (use `.get(name, float("nan"))`), keeping the existing fields and their order first:

```python
                    "e2e slot telemetry epoch=%d h_pairwise_cosine_mean=%.6f "
                    "plan_rank1_marginal_residual=%.6f streak=%d plan_total_mass=%.6g "
                    "plan_max_cell_fraction=%.6f h_norm_mean=%.4f h_pairwise_sqdist_mean=%.4f",
```

- [ ] **Step 6: Run the full trainer test files**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py tests/test_train_egostitch_training.py tests/experiments -q`
Expected: PASS — in particular `tests/experiments/test_probes.py` must be untouched by this change.

- [ ] **Step 7: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src tests
git add src/train_egostitch.py tests
git commit -m "feat(train): log OT-scale telemetry without touching the probe or fidelity ABI"
```

---

## Task 9: `--ddp-mode init-probe` — the read-only pre-GPU gate

**Files:**
- Modify: `src/train_egostitch.py:1334` region (the argparse `--ddp-mode` choices — grep `epoch-probe` for the exact tuple), `:5917-5960` (the mode dispatch, beside the `probe` and `epoch-probe` branches)
- Test: `tests/test_train_egostitch_e2e.py`

**Interfaces:**
- Consumes: `_validate_epoch`, `_bind_feature_standardization` (Task 7), `_ValidationResult.scale_telemetry` (Task 8).
- Produces: `_run_init_probe(model, cfg, data, accelerator, *, edge_batch, topk_fraction) -> dict[str, float]` — runs one validation pass at initialization, takes no optimizer step, writes no artifact.

This is the §8 S0 hard gate: it measures the guard's own population before any GPU budget is committed.

- [ ] **Step 1: Write the failing test**

```python
class TestInitProbe:
    def test_init_probe_reports_guard_telemetry_without_training(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        data = te.assemble_egostitch_data(cfg)
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))
        te._bind_feature_standardization(model, cfg, data)
        accelerator = Accelerator()
        before = [p.detach().clone() for p in model.parameters()]

        report = te._run_init_probe(
            model, cfg, data, accelerator, edge_batch=8, topk_fraction=0.1
        )

        assert "h_pairwise_cosine_mean" in report
        assert "plan_rank1_marginal_residual" in report
        assert "plan_total_mass" in report
        for original, current in zip(before, model.parameters(), strict=True):
            torch.testing.assert_close(original, current.detach())

    def test_init_probe_fails_closed_on_an_empty_guard_population(self, tmp_path: Path) -> None:
        cfg = _e2e_config(tmp_path)
        data = replace(te.assemble_egostitch_data(cfg), val_pairs=[])
        model = EgoStitchE2E(E2EConfig.from_mapping(cfg.model.config))
        te._bind_feature_standardization(model, cfg, data)

        with pytest.raises(RuntimeError, match="guard population"):
            te._run_init_probe(
                model, cfg, data, Accelerator(), edge_batch=8, topk_fraction=0.1
            )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py -q -k InitProbe`
Expected: FAIL — `AttributeError: ... has no attribute '_run_init_probe'`

- [ ] **Step 3: Implement**

```python
def _run_init_probe(
    model: EgoStitchE2E,
    cfg: EgoConfig,
    data: EgoStitchData,
    accelerator: Accelerator,
    *,
    edge_batch: int,
    topk_fraction: float,
) -> dict[str, float]:
    """Measure the collapse guard's own population at initialization.

    Read-only: no optimizer step, no checkpoint, no artifact. Spec Sec 13.19.1
    landed a preprocessing change whose whole claim is that a random model is
    born healthy; this is where that claim is checked before GPU time is spent.

    Args:
        model: The bound, freshly initialized E2E model.
        cfg: The loaded run configuration.
        data: The assembled training data.
        accelerator: The live accelerator.
        edge_batch: Validation pair batch size.
        topk_fraction: The registered top-k fidelity fraction.

    Returns:
        The guard telemetry plus the scale diagnostics, on the main process
        (an empty dict on other ranks).

    Raises:
        RuntimeError: When there are no validation pairs to measure.
    """
    if not data.val_pairs:
        raise RuntimeError(
            "init probe has an empty guard population: this config gives "
            "_validate_epoch no validation pairs, so the slot-collapse guard "
            "would never evaluate"
        )
    validation = _validate_epoch(
        model, data, accelerator, edge_batch=edge_batch, topk_fraction=topk_fraction
    )
    if validation is None:
        return {}
    report = {
        "h_pairwise_cosine_mean": validation.fidelity["h_pairwise_cosine_mean"],
        "plan_rank1_marginal_residual": validation.fidelity["plan_rank1_marginal_residual"],
        "pi_slot_std": validation.fidelity["pi_slot_std"],
        "adj_offdiag_std": validation.fidelity["adj_offdiag_std"],
        **validation.scale_telemetry,
    }
    logger.info(
        "e2e init probe rows=%d feature_stats_sha256=%s %s",
        len(data.val_pairs),
        model.feature_stats_digest_hex,
        " ".join(f"{name}={value:.6g}" for name, value in sorted(report.items())),
    )
    if report["h_pairwise_cosine_mean"] > 0.95:
        logger.error(
            "init probe reads above the Sec 14.4.8 cosine trip line (%.6f > 0.95): "
            "the run would be born collapsed",
            report["h_pairwise_cosine_mean"],
        )
    return report
```

Wire the CLI: add `"init-probe"` to the `--ddp-mode` choices tuple, and add the dispatch branch beside `epoch-probe` (copy the `edge_batch` / `topk_fraction` expressions from the per-epoch `_validate_epoch` call at `:4547`):

```python
    if args.ddp_mode == "init-probe":
        _run_init_probe(
            model,
            cfg,
            data,
            accelerator,
            edge_batch=<same expression as the epoch-loop call site>,
            topk_fraction=<same expression as the epoch-loop call site>,
        )
        return
```

Update the module docstring's CLI contract line (`src/train_egostitch.py:4`) to
``--ddp-mode {probe,epoch-probe,init-probe,train}``.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_train_egostitch_e2e.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src tests
git add src/train_egostitch.py tests/test_train_egostitch_e2e.py
git commit -m "feat(train): read-only init probe on the slot-collapse guard population"
```

---

## Task 10: Scoring round-trip and the six v3 configs

**Files:**
- Modify: `configs/egostitch_e2e_v3_full_breadth_first.yaml`, `..._f_only_...`, `..._pair_topology_...`, `..._p0_...`, `..._cosine_pool_...`, `..._no_l_rel_...` (the `model.config` block, lines 3-14)
- Test: `tests/test_score_universe.py`

**Interfaces:**
- Consumes: everything above. Verifies the checkpoint → `_load_checkpoint` → score path reconstructs the transform without any V_fit access.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_score_universe.py` (reuse the file's existing e2e checkpoint fixture; grep `_build_egostitch_e2e` / `egostitch_e2e` in that file for the closest one):

```python
def test_e2e_checkpoint_restores_the_feature_standardization(tmp_path: Path) -> None:
    gen = np.random.default_rng(0)
    cfg = E2EConfig(feature_standardization="zscore_vfit_v1")
    trained = EgoStitchE2E(cfg)
    rows = (30.0 + 4.0 * gen.standard_normal((32, trained.generator_cfg.input_dim))).astype(
        np.float32
    )
    stats = compute_feature_stats(rows, [f"n{i}" for i in range(32)])
    trained.set_feature_stats(stats)

    path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": trained.state_dict(),
            "model_family": "egostitch_e2e",
            "model_config": {
                **asdict(cfg),
                "feature_stats_sha256": stats.digest,
            },
        },
        path,
    )

    restored, family, _ = su._load_checkpoint(path)
    assert family == "egostitch_e2e"
    assert restored.feature_stats_digest_hex == stats.digest
    x = torch.randn(4, trained.generator_cfg.input_dim)
    torch.testing.assert_close(
        restored.generator.normalize_features(x), trained.generator.normalize_features(x)
    )


def test_rev31_checkpoints_still_load_as_row_layernorm(tmp_path: Path) -> None:
    legacy_cfg = E2EConfig(feature_standardization="row_layernorm")
    legacy = EgoStitchE2E(legacy_cfg)
    payload = {k: v for k, v in asdict(legacy_cfg).items() if k != "feature_standardization"}
    payload.pop("feature_stats_sha256", None)
    path = tmp_path / "legacy.pt"
    torch.save(
        {"model_state": legacy.state_dict(), "model_family": "egostitch_e2e",
         "model_config": payload},
        path,
    )

    restored, _, _ = su._load_checkpoint(path)
    assert restored.generator.feature_standardization == "row_layernorm"
```

- [ ] **Step 2: Run to verify failure, then confirm no source change is needed**

Run: `.venv/bin/python -m pytest tests/test_score_universe.py -q -k feature_standardization`

Both tests should pass **without** touching `src/score_universe.py` — the buffers ride the strict state-dict load and `e2e_checkpoint_config` supplies the legacy default (Task 4). If either fails, the fix belongs in `src/model/egostitch/config.py:e2e_checkpoint_config` or in the buffer registration, **not** in a `strict=False` load. Never relax strict loading.

- [ ] **Step 3: Declare the mode in the six v3 configs**

Add to each `model.config` block (after `n_ground: 50`):

```yaml
    feature_standardization: zscore_vfit_v1
    feature_stats_sha256: ""
```

Also update each file's header comment from "rev-3.1" to "rev-3.2 (D0 feature standardization; spec §13.19.1)".

`feature_stats_sha256` stays empty until S1 measures it — the runbook below pins it before the rehearsal. Leaving it empty is legal (the trainer records the computed digest either way) but a BINDING run must have it pinned.

- [ ] **Step 4: Run the affected suites**

Run: `.venv/bin/python -m pytest tests/test_score_universe.py tests/test_g5_stage1_e2e.py -q`
Expected: PASS. Tests that assert a config digest or `config_hash` literal will fail — regenerate the expected values from the new files (that invalidation is exactly what the §12 change-log entry records). Do not weaken those assertions.

- [ ] **Step 5: Commit**

```bash
git add configs tests/test_score_universe.py
git commit -m "feat(configs): declare rev-3.2 feature standardization on the six v3 arms"
```

---

## Task 11: Whole-suite verification and the adversarial review wave

**Files:** none (verification only), then `docs/results/` status note if the review finds nothing.

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q -m "not integration"`
Expected: PASS. Investigate every failure; do not `xfail` anything.

- [ ] **Step 2: Lint and strict typecheck from cold caches**

```bash
rm -rf .mypy_cache
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src tests
```
(Cold cache per the recorded gotcha: concurrent mypy runs produce phantom `unused-ignore` errors.)

- [ ] **Step 3: Confirm the untouched contracts, mechanically**

```bash
git diff --stat main...HEAD
git diff main...HEAD -- src/experiments/probes.py src/experiments/prebinding_gates.py hpc/
```
Expected: **empty** for the second command. The probe evaluator, the gate evaluator, and the HPC harness are unchanged by D0 (design doc §8). Also confirm `git diff main...HEAD -- src/data/grounding.py` is empty — raw-F0 retrieval is pinned.

- [ ] **Step 4: Independent adversarial review (CLAUDE.md wave rule)**

```bash
CH=/private/tmp/claude-501/-Users-richardwang-Documents-topology-conditioned-inductive-edge-prediction/e4042827-87ff-429d-b39a-0575bb09bb2e/scratchpad/codex-home
mkdir -p "$CH"; cp ~/.codex/auth.json "$CH/"
printf 'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "high"\napproval_policy = "never"\nsandbox_mode = "workspace-write"\n' > "$CH/config.toml"
CODEX_HOME="$CH" codex review --base 6bb4b82 > /private/tmp/.../scratchpad/rev32-d0-review.txt 2>&1
```

Run it in the background and read the file when it finishes — never let ~200 KB of review output land in context. A clean `CODEX_HOME` is mandatory (`-c 'mcp_servers={}'` does not disable MCP and will hang the review).

- [ ] **Step 5: Triage and fix**

Use `superpowers:receiving-code-review`. Blockers get fixed and re-tested; anything that argues for a *design* change (not an implementation bug) goes back to the owner as a design-doc §4 ledger candidate — do not silently expand D0's scope.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore(rev-3.2): address D0 review findings"
```

---

## Self-Review (run this against the spec before handing off)

- **Spec coverage:** design-doc §3.1 → Tasks 1, 3; §3.2 scope/provenance/storage → Tasks 2, 4, 6, 7, 10; §3.3 pinned non-changes → Task 11 Step 3 (mechanical check), Task 8 (ABI protection), Task 3 (SSL noise); §3.4 verification → Tasks 5, 9; §8 S0 test list → Tasks 2 (leakage, fail-closed), 3 (transform, SSL), 5 (init health), 10 (checkpoint round-trip), 9 (init probe on the real population).
- **Not implemented, deliberately:** everything in design-doc §4 (L-1 … L-5); the registration v3.1 edit and the two measured thresholds (S2, owner).
- **Known invalidations, all intended and recorded in the §12 entry:** `config_hash` changes for all six arms; rev-3.1 calibration artifacts become inadmissible; the pack manifest gains a file; `_checkpoint_id` changes for rev-3.2 checkpoints.

---

## Runbook after S0 (owner-run; not implementation tasks)

The experiment suite is unchanged from rev-3.1 — the eight-arm v3 screen, the five G3 gates, and `hpc/qualification.sh` all carry over verbatim. What follows is the sequence, not new work.

1. **S0 gate (local, no GPU).** Full suite green, then rebuild the feature pack and run
   `--ddp-mode init-probe` with the exact calibration config. **Hard gate:** `h_pairwise_cosine_mean`
   comfortably below 0.95 with a live plan (`plan_total_mass` well above the 1e-30 clamp,
   `plan_rank1_marginal_residual` ≥ 0.3). If it is not, D0 did not do its job — stop and report;
   no GPU time is justified. Record the probe's `feature_stats_sha256`.
   - If the probe reports **an empty guard population**, that is a finding, not a probe bug: the
     calibration config would give the guard nothing to evaluate. Report it before launching.
2. **Pin the digest.** Paste the recorded `feature_stats_sha256` into `feature_stats_sha256` in
   all six `configs/egostitch_e2e_v3_*_breadth_first.yaml`. From here the trainer fails closed on
   any pack rebuild that changes the statistics.
3. **S1 — `hpc/qualification.sh calibrate`** (~1.5 h train, `full` arm only). Produces the G3.4
   evaluator bootstrap noise floor and the G3.5 matched edge-AUPRC guard — the two
   `REQUIRED-BEFORE-BINDING` placeholders — plus the collapse-telemetry trajectory. V_fit only;
   never opens V_qual; burns no `v_qual_rehearsals`. G3.1 (0.0698) is *not* recalibrated: it is
   0.5 × the raw-F0 top-50 pool ceiling and D0 pins raw-F0 retrieval.
   - Read the per-epoch `e2e slot telemetry` line. Its purpose is now diagnostic: a run that
     *starts* healthy and *drifts* into collapse is the first true dynamics measurement in this
     program, and the design doc §7 kill criterion says which §4 ledger item the trajectory
     implicates (L-1 degree-stratified decay, L-3 query-noise insufficiency, L-2 scale drift) —
     **one at a time, with evidence.**
4. **S2 — freeze + registration v3.1 (owner).** Measured thresholds, the μ/σ digest, the new
   implementation SHA, six regenerated config digests. **The pre-binding gate circularity must be
   resolved here** (G3.4/G3.5 are the circular pair). D0 adds no jointly-searched constants, so it
   neither widens nor resolves it. A new registration version resets the ≤3-attempt window, so
   rev-3.1 attempt-001 does not consume rev-3.2's budget.
5. **S3 — `rehearse`** (~1.5 h): one V_qual attempt, refuses to start with any threshold unfrozen,
   V_select stays sealed.
6. **S4 — `formal <arm>`** (scoring ≈ 42 h): six trained arms + the two scoring-time controls over
   `full`'s checkpoint, `full` first with the eligibility preflight. This is the G5 Stage-1 verdict.

**Claim discipline (unchanged):** fixed-Seed-0 engineering screen; p-values/CIs/Holm stay `null`;
edge-level and assembled-graph metrics reported together; the three MMD ratios are never
aggregated; dispositions are owner-side.

