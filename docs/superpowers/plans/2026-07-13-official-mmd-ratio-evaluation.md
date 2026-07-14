# Official MMD Ratio Evaluation Implementation Plan

> **Historical implementation record:** this dated plan predates the 2026-07-14
> migration from the MMD-ratio composite/global-density schema to official PRING GS/RD.
> Current results live under `outputs/deliverables/*_pring_20260714/`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current assembled-graph MMD implementation and output schema with the fixed-sigma Gaussian-TV, split-reference normalized MMD ratio needed to produce PRING-paper-table-scale degree, clustering, and spectral values.

**Architecture:** `src/eval/graph_metrics.py` becomes the single source of truth for descriptor construction, raw biased MMD2, deterministic reference-floor construction, and ratio-of-aggregate normalization. The evaluator exposes the numerator, denominator, and normalized ratio, while downstream threshold sweeps, G1 rows, bootstrap summaries, metadata, and Markdown tables use the normalized ratio as the canonical metric. There is no compatibility mode and no retained L2-RBF/median-bandwidth path.

**Tech Stack:** Python 3.11, NetworkX, NumPy, SciPy, pytest, mypy, ruff, uv.

## Global Constraints

- Replace the old evaluator; do not add a mode switch, compatibility branch, or second named metric profile.
- Use neutral names: `raw_mmd2`, `reference_mmd2`, and `mmd_ratio`. Do not encode a paper or table name in Python identifiers or output keys.
- Use `d_TV(x,y) = 0.5 * ||x-y||_1`, `k(x,y) = exp(-d_TV(x,y)^2 / (2 * sigma^2))`, and fixed `sigma = 1.0`.
- Use the biased V-statistic MMD2 with all Cartesian-product terms, including within-sample diagonal terms.
- Normalize each descriptor histogram by `sum + 1e-6` inside `mmd_squared`, matching the official evaluator. Before that common normalization, the spectral worker converts its 200-bin counts to a PMF; degree and clustering remain raw counts.
- Degree descriptors use the complete `networkx.degree_histogram` with pairwise zero-padding; do not clip degrees.
- Clustering descriptors use 100 bins on `[0, 1]`; spectral descriptors use all normalized-Laplacian eigenvalues with 200 bins on `[-1e-5, 2]`.
- For every node-size bucket, split reference descriptor samples deterministically by original artifact order: `samples[::2]` versus `samples[1::2]`.
- Aggregate with ratio-of-means: mean raw MMD2 over node sizes divided by mean reference MMD2 over node sizes. Do not average per-size ratios.
- Require at least two reference samples in each bucket and use `reference_epsilon = 1e-12` only as a division guard.
- Retain self-loops in canonical MMD descriptor induced subgraphs, matching the official evaluator, and continue reporting their counts separately. Relative density, recall, and other structural metrics still use simple graphs.
- The canonical downstream metric is `mmd_ratio`; `raw_mmd2` and `reference_mmd2` remain in artifacts solely as disclosed components of that metric.
- Existing checked-in result numbers must not be reinterpreted or mechanically renamed. They remain old-run artifacts until a new G1 evaluation is executed and reviewed.
- Preserve unrelated working-tree changes. `README.md`, `docs/03-experiment-protocol.md`, `docs/04-model-proposal.md`, `docs/06-egostitch-spec.md`, `docs/results/E2-pair-to-topology-gap.md`, and `figures/e2-gap.html` are already modified before this plan.
- Every shell command is prefixed with `rtk`; every command-chain segment is prefixed separately.

## File Structure

- Modify `docs/03-experiment-protocol.md`: bind the new canonical MMD formula and disclosure requirements.
- Modify `docs/06-egostitch-spec.md`: bind assembled-evaluation normalization in §10.3 and add a dated change-log entry before code changes.
- Modify `src/eval/graph_metrics.py`: replace descriptors, kernel, MMD estimator, reference floor, bucket aggregation, and bootstrap semantics.
- Modify `src/eval/assembly.py`: replace threshold-sweep `mmd2` output with `mmd_ratio`.
- Modify `src/eval/composite.py`: make the composite consume already-normalized ratios directly, with no scale or compatibility field.
- Modify `src/experiments/g1_hardened_e2.py`: replace result fields, remove `tau`/`scales`, and update metadata and Markdown rendering.
- Modify `tests/eval/test_graph_metrics.py`: pin the official formula, descriptor rules, split rule, and ratio-of-means aggregation.
- Modify `tests/eval/test_assembly.py`: assert threshold rows expose ratios.
- Modify `tests/eval/test_composite.py`: assert the composite consumes ratios directly.
- Modify `tests/test_g1_hardened_e2.py`: assert the new JSON/Markdown schema and remove old configuration arguments.

## Verified execution corrections

Live exact-value verification supersedes any conflicting example snippet later in
this plan: canonical descriptor subgraphs retain self-loops; spectral histograms are
PMFs before the common `sum + 1e-6` normalization; bootstrap replicates aggregate
raw and reference MMD2 across sizes before taking their ratio; and the final schema
contains no `CompositeDefinition.scales`, top-level `tau`, or metadata `scales`.

---

### Task 1: Freeze the replacement metric contract

**Files:**
- Modify: `docs/03-experiment-protocol.md:97-115`
- Modify: `docs/06-egostitch-spec.md:358-366`
- Modify: `docs/06-egostitch-spec.md:382-398`

**Interfaces:**
- Consumes: The repository freeze rule and the formula established in the global constraints.
- Produces: Binding prose that Tasks 2 and 3 implement exactly.

- [ ] **Step 1: Inspect and preserve the pre-existing documentation changes**

Run:

```bash
rtk git diff -- docs/03-experiment-protocol.md docs/06-egostitch-spec.md
```

Expected: both files may already contain user changes; identify the exact surrounding paragraphs before applying only the additions below.

- [ ] **Step 2: Add the canonical normalization block to the experiment protocol**

Insert the following bullets immediately after the existing MMD hygiene bullet in `docs/03-experiment-protocol.md`:

```markdown
  - *Canonical MMD definition:* each graph is mapped to a full degree histogram,
    a 100-bin local-clustering histogram on `[0,1]`, or a 200-bin normalized-
    Laplacian spectral histogram on `[-1e-5,2]`. Histograms are normalized by
    `sum + 1e-6`. MMD² is the biased V-statistic under
    `k(x,y)=exp(-(0.5·||x-y||₁)²/2)` (`σ=1`, including within-sample diagonals).
  - *Reference normalization:* within every node-size bucket, reference samples
    retain artifact order and are split as `samples[::2]` versus `samples[1::2]`.
    The reported statistic is
    `mean_size MMD²(pred_size,ref_size) / mean_size MMD²(ref_even,ref_odd)`.
    Numerator, denominator, and ratio are all stored; the ratio is canonical and
    lower is better. A `1e-12` denominator floor is only a numerical guard.
```

- [ ] **Step 3: Bind the same rule in the EgoStitch evaluation-loader contract**

Add this paragraph after the assembled-graph-eval bullet in `docs/06-egostitch-spec.md` §10.3:

```markdown
- The assembled evaluator uses the protocol's single canonical MMD ratio: fixed
  Gaussian-TV (`σ=1`) raw MMD² divided by the deterministic even/odd reference
  floor after separately averaging numerator and denominator across node-size
  buckets. Run artifacts disclose `raw_mmd2`, `reference_mmd2`, and `mmd_ratio`;
  only `mmd_ratio` is used in result tables and the topology composite.
```

Append this exact change-log item in §12:

```markdown
- 2026-07-13: replaced the assembled evaluator's L2-RBF/median-bandwidth raw MMD²
  with the fixed-`σ=1` Gaussian-TV biased MMD² ratio defined in protocol §1;
  removed degree clipping and bound deterministic even/odd reference splitting,
  ratio-of-size-means aggregation, and numerator/denominator disclosure.
```

- [ ] **Step 4: Verify the documentation expresses one metric only**

Run:

```bash
rtk proxy rg -n "Canonical MMD definition|Reference normalization|Gaussian-TV|mmd_ratio" docs/03-experiment-protocol.md docs/06-egostitch-spec.md
```

Expected: the new protocol block, §10.3 paragraph, and change-log entry appear; none describes a legacy or optional evaluator.

- [ ] **Step 5: Stage only the plan-owned documentation hunks and commit**

Run:

```bash
rtk git add -p docs/03-experiment-protocol.md docs/06-egostitch-spec.md
rtk git diff --cached --check
rtk git commit -m "docs: bind normalized topology MMD evaluation"
```

Expected: select only the three additions above during `git add -p`; the commit excludes all pre-existing user hunks.

---

### Task 2: Replace the core descriptors and MMD evaluator

**Files:**
- Modify: `src/eval/graph_metrics.py:18-429`
- Modify: `tests/eval/test_graph_metrics.py:11-302`

**Interfaces:**
- Consumes: `networkx.Graph`, `dict[int, list[set[str]]]`, and the Task 1 metric contract.
- Produces: `MMDConfig`, `mmd_squared`, `BucketedMMDReport`, `evaluate_assembled_graph`, `noise_floor`, and `bootstrap_mmd` with the signatures shown below.

- [ ] **Step 1: Replace the old unit tests with formula-pinning failing tests**

Remove tests for degree clipping, median bandwidth, and bandwidth scales. Add these imports and tests to `tests/eval/test_graph_metrics.py`:

```python
from src.eval.graph_metrics import (
    STATISTICS,
    MMDConfig,
    bootstrap_mmd,
    clustering_histogram,
    degree_histogram,
    evaluate_assembled_graph,
    laplacian_spectrum_histogram,
    mmd_squared,
    noise_floor,
)


@pytest.mark.unit
class TestOfficialMmdSquared:
    def test_singleton_total_variation_formula(self) -> None:
        a = [np.array([1.0, 0.0])]
        b = [np.array([0.0, 1.0])]
        expected = 2.0 - 2.0 * np.exp(-0.5)
        assert mmd_squared(a, b, MMDConfig()) == pytest.approx(expected)

    def test_identical_sets_are_zero(self) -> None:
        samples = [np.array([3.0, 1.0]), np.array([1.0, 3.0])]
        assert mmd_squared(samples, samples, MMDConfig()) == pytest.approx(0.0, abs=1e-12)

    def test_histograms_are_normalized_inside_mmd(self) -> None:
        a = [np.array([3.0, 1.0]), np.array([1.0, 3.0])]
        b = [np.array([6.0, 2.0]), np.array([2.0, 6.0])]
        assert mmd_squared(a, b, MMDConfig()) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
class TestOfficialDescriptors:
    def test_degree_histogram_keeps_full_support(self) -> None:
        hist = degree_histogram(nx.star_graph(70))
        assert hist.shape == (71,)
        assert hist[1] == 70
        assert hist[70] == 1

    def test_clustering_uses_one_hundred_bins(self) -> None:
        hist = clustering_histogram(nx.complete_graph(3))
        assert hist.shape == (100,)
        assert hist[-1] == 3

    def test_spectral_uses_two_hundred_bins(self) -> None:
        hist = laplacian_spectrum_histogram(nx.path_graph(5))
        assert hist.shape == (200,)
        assert hist.sum() == pytest.approx(1.0)
```

Add this aggregation test after `_seeded_er_graph_and_buckets`:

```python
@pytest.mark.unit
def test_bucket_report_uses_ratio_of_aggregate_means() -> None:
    g_ref, buckets = _seeded_er_graph_and_buckets()
    g_pred = nx.Graph()
    g_pred.add_nodes_from(g_ref.nodes())
    report = evaluate_assembled_graph(g_pred, g_ref, buckets, MMDConfig())

    for stat in STATISTICS:
        raw_mean = float(np.mean([report.per_size_raw_mmd2[size][stat] for size in buckets]))
        ref_mean = float(
            np.mean([report.per_size_reference_mmd2[size][stat] for size in buckets])
        )
        assert report.raw_mmd2[stat] == pytest.approx(raw_mean)
        assert report.reference_mmd2[stat] == pytest.approx(ref_mean)
        assert report.mmd_ratio[stat] == pytest.approx(raw_mean / max(ref_mean, 1e-12))
```

- [ ] **Step 2: Run the focused tests and verify they fail for the old evaluator**

Run:

```bash
rtk proxy uv run pytest tests/eval/test_graph_metrics.py -q
```

Expected: failures show the old `degree_histogram(..., max_degree=...)` API, `MMDResult` return type, L2 kernel, and missing ratio report fields.

- [ ] **Step 3: Replace the configuration, descriptors, and raw MMD implementation**

Replace the current descriptor/config/kernel section in `src/eval/graph_metrics.py` with:

```python
STATISTICS = ("degree", "clustering", "spectral")


def degree_histogram(g: nx.Graph) -> np.ndarray:
    """Return the complete, unnormalized NetworkX degree histogram."""
    return np.asarray(nx.degree_histogram(g), dtype=float)


def clustering_histogram(g: nx.Graph) -> np.ndarray:
    """Return the official 100-bin local-clustering histogram on [0, 1]."""
    coeffs = list(nx.clustering(g).values())
    counts, _ = np.histogram(coeffs, bins=100, range=(0.0, 1.0), density=False)
    return counts.astype(float)


def laplacian_spectrum_histogram(g: nx.Graph) -> np.ndarray:
    """Return the official 200-bin normalized-Laplacian spectral PMF."""
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(g).todense())
    except Exception:
        eigs = np.zeros(g.number_of_nodes())
    counts, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2.0), density=False)
    hist = counts.astype(float)
    return hist / max(1.0, float(hist.sum()))


@dataclass(frozen=True)
class MMDConfig:
    """Fixed parameters for the canonical normalized MMD evaluation."""

    sigma: float = 1.0
    reference_epsilon: float = 1e-12


def _pad_histograms(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support_size = max(len(x), len(y))
    x_values = np.pad(np.asarray(x, dtype=float), (0, support_size - len(x)))
    y_values = np.pad(np.asarray(y, dtype=float), (0, support_size - len(y)))
    return x_values, y_values


def _gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    x_values, y_values = _pad_histograms(x, y)
    distance = float(np.abs(x_values - y_values).sum() / 2.0)
    return float(np.exp(-(distance * distance) / (2.0 * sigma * sigma)))


def _mean_kernel(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    *,
    sigma: float,
) -> float:
    total = sum(_gaussian_tv(x, y, sigma) for x in samples1 for y in samples2)
    return float(total / (len(samples1) * len(samples2)))


def mmd_squared(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    config: MMDConfig,
) -> float:
    """Return the biased Gaussian-TV MMD2 used by the canonical evaluator."""
    if not samples1 or not samples2:
        raise ValueError("mmd_squared requires two non-empty sample sets")
    normalized1 = [sample / (float(np.sum(sample)) + 1e-6) for sample in samples1]
    normalized2 = [sample / (float(np.sum(sample)) + 1e-6) for sample in samples2]
    return float(
        _mean_kernel(normalized1, normalized1, sigma=config.sigma)
        + _mean_kernel(normalized2, normalized2, sigma=config.sigma)
        - 2.0 * _mean_kernel(normalized1, normalized2, sigma=config.sigma)
    )
```

Add `from scipy.linalg import eigvalsh` and remove `from scipy.spatial.distance import cdist`.

- [ ] **Step 4: Replace the bucket report and evaluator**

Replace `BucketedMMDReport`, `_descriptors`, and `evaluate_assembled_graph` with:

```python
@dataclass(frozen=True)
class BucketedMMDReport:
    per_size_raw_mmd2: dict[int, dict[str, float]]
    per_size_reference_mmd2: dict[int, dict[str, float]]
    raw_mmd2: dict[str, float]
    reference_mmd2: dict[str, float]
    mmd_ratio: dict[str, float]
    relative_density: float
    self_loops_pred: int
    self_loops_ref: int


def _descriptors(g_simple: nx.Graph) -> dict[str, np.ndarray]:
    return {
        "degree": degree_histogram(g_simple),
        "clustering": clustering_histogram(g_simple),
        "spectral": laplacian_spectrum_histogram(g_simple),
    }


def evaluate_assembled_graph(
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> BucketedMMDReport:
    self_loops_pred = nx.number_of_selfloops(g_pred)
    self_loops_ref = nx.number_of_selfloops(g_ref)
    pred_simple_full = strip_self_loops(g_pred)
    ref_simple_full = strip_self_loops(g_ref)
    ref_edge_count = ref_simple_full.number_of_edges()
    pred_edge_count = pred_simple_full.number_of_edges()
    relative_density = (
        pred_edge_count / ref_edge_count
        if ref_edge_count > 0
        else (0.0 if pred_edge_count == 0 else float("inf"))
    )

    per_size_raw: dict[int, dict[str, float]] = {}
    per_size_reference: dict[int, dict[str, float]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        pred_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        ref_descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            pred_d = _descriptors(_simple_subgraph(g_pred, nodes))
            ref_d = _descriptors(_simple_subgraph(g_ref, nodes))
            for stat in STATISTICS:
                pred_descs[stat].append(pred_d[stat])
                ref_descs[stat].append(ref_d[stat])
        per_size_raw[size] = {
            stat: mmd_squared(pred_descs[stat], ref_descs[stat], config)
            for stat in STATISTICS
        }
        per_size_reference[size] = {
            stat: mmd_squared(ref_descs[stat][::2], ref_descs[stat][1::2], config)
            for stat in STATISTICS
        }

    raw_mmd2 = {
        stat: float(np.mean([per_size_raw[size][stat] for size in per_size_raw]))
        for stat in STATISTICS
    }
    reference_mmd2 = {
        stat: float(np.mean([per_size_reference[size][stat] for size in per_size_reference]))
        for stat in STATISTICS
    }
    mmd_ratio = {
        stat: raw_mmd2[stat] / max(reference_mmd2[stat], config.reference_epsilon)
        for stat in STATISTICS
    }
    return BucketedMMDReport(
        per_size_raw_mmd2=per_size_raw,
        per_size_reference_mmd2=per_size_reference,
        raw_mmd2=raw_mmd2,
        reference_mmd2=reference_mmd2,
        mmd_ratio=mmd_ratio,
        relative_density=relative_density,
        self_loops_pred=self_loops_pred,
        self_loops_ref=self_loops_ref,
    )
```

- [ ] **Step 5: Replace noise-floor and bootstrap semantics**

Change `noise_floor` to return the deterministic reference denominator used by the evaluator, removing `seed` and `n_splits`. Change `bootstrap_mmd` so each resample reports the normalized ratio, not raw MMD2:

```python
def noise_floor(
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        descs: dict[str, list[np.ndarray]] = {stat: [] for stat in STATISTICS}
        for nodes in node_sets:
            values = _descriptors(_simple_subgraph(g_ref, nodes))
            for stat in STATISTICS:
                descs[stat].append(values[stat])
        result[size] = {
            stat: mmd_squared(descs[stat][::2], descs[stat][1::2], config)
            for stat in STATISTICS
        }
    return result


def bootstrap_mmd(
    g_pred: nx.Graph,
    g_ref: nx.Graph,
    buckets: dict[int, list[set[str]]],
    config: MMDConfig,
    *,
    seed: int,
    n_boot: int = 200,
) -> dict[int, dict[str, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for size, node_sets in buckets.items():
        if len(node_sets) < 2:
            raise ValueError(f"bucket {size} requires at least two reference samples")
        pred = [_descriptors(_simple_subgraph(g_pred, nodes)) for nodes in node_sets]
        ref = [_descriptors(_simple_subgraph(g_ref, nodes)) for nodes in node_sets]
        values: dict[str, list[float]] = {stat: [] for stat in STATISTICS}
        for _ in range(n_boot):
            indices = rng.integers(0, len(node_sets), size=len(node_sets))
            for stat in STATISTICS:
                pred_samples = [pred[i][stat] for i in indices]
                ref_samples = [ref[i][stat] for i in indices]
                raw = mmd_squared(pred_samples, ref_samples, config)
                denominator = mmd_squared(ref_samples[::2], ref_samples[1::2], config)
                values[stat].append(raw / max(denominator, config.reference_epsilon))
        result[size] = {
            stat: (float(np.mean(values[stat])), float(np.std(values[stat])))
            for stat in STATISTICS
        }
    return result
```

- [ ] **Step 6: Update all graph-metric tests for the new APIs**

Make these exact mechanical changes in `tests/eval/test_graph_metrics.py`:

```python
config = MMDConfig()
floor = noise_floor(g_ref, buckets, config)
```

For identical-graph evaluation, assert `report.raw_mmd2[stat] == 0`, `report.mmd_ratio[stat] == 0`, and `report.reference_mmd2[stat] > 0`. For bootstrap, keep the shape/type assertions and assert the identical-graph ratio mean and standard deviation are both approximately zero.

- [ ] **Step 7: Run the core tests**

Run:

```bash
rtk proxy uv run pytest tests/eval/test_graph_metrics.py -q
```

Expected: all graph-metric tests pass.

- [ ] **Step 8: Commit the core evaluator replacement**

Run:

```bash
rtk git add src/eval/graph_metrics.py tests/eval/test_graph_metrics.py
rtk git diff --cached --check
rtk git commit -m "feat: replace topology MMD with normalized TV ratio"
```

Expected: the commit contains no L2-distance kernel, median bandwidth, bandwidth sweep, or degree clipping.

---

### Task 3: Replace downstream result semantics and rendering

**Files:**
- Modify: `src/eval/assembly.py:104-164`
- Modify: `src/eval/composite.py:63-77`
- Modify: `src/experiments/g1_hardened_e2.py:770-955`
- Modify: `src/experiments/g1_hardened_e2.py:1067-1149`
- Modify: `src/experiments/g1_hardened_e2.py:1389-1402`
- Modify: `tests/eval/test_assembly.py`
- Modify: `tests/eval/test_composite.py`
- Modify: `tests/test_g1_hardened_e2.py`

**Interfaces:**
- Consumes: `BucketedMMDReport.raw_mmd2`, `.reference_mmd2`, and `.mmd_ratio` from Task 2.
- Produces: threshold rows, assembled rows, JSON metadata, and Markdown tables whose canonical topology columns are ratios.

- [ ] **Step 1: Write failing downstream-schema tests**

In `tests/eval/test_assembly.py`, replace assertions that inspect `point.mmd2` with:

```python
assert set(point.mmd_ratio) == set(STATISTICS)
```

In `tests/test_g1_hardened_e2.py`, add:

```python
def test_pipeline_exposes_only_normalized_mmd_schema(tmp_path: Path) -> None:
    g = _make_reference_graph()
    buckets = _small_buckets(_NODES, size=5, n_samples=4, seed=12)
    data_root = _write_benchmark(tmp_path, "toy", g, buckets)
    universe_path = _reference_universe_path(tmp_path)
    payload = g1.run_g1_pipeline(
        universe_path=universe_path,
        alt_universe_path=None,
        data_root=data_root,
        strategy="toy",
        output_dir=tmp_path / "out",
        seed=0,
        skip_perturbation_check=True,
    )
    row = _d(_d(payload["assembled"])["b0"])
    assert set(_d(row["mmd_ratio"])) == set(STATISTICS)
    assert set(_d(row["raw_mmd2"])) == set(STATISTICS)
    assert set(_d(row["reference_mmd2"])) == set(STATISTICS)
    assert "aggregate_mmd2" not in row
    metadata = _d(payload["metadata"])
    assert metadata["metric_normalization"] == "ratio_of_size_mean_mmd2"
```

Add a renderer assertion that `g1_tables.md` contains `degree MMD ratio`, `raw numerator`, and `reference denominator`, and does not contain the old `degree MMD2` header.

- [ ] **Step 2: Run downstream tests and verify old-field failures**

Run:

```bash
rtk proxy uv run pytest tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py -q
```

Expected: failures identify `mmd2`, `aggregate_mmd2`, old `MMDConfig` arguments, and old metadata/rendering.

- [ ] **Step 3: Replace threshold-sweep fields**

In `src/eval/assembly.py`, replace the `SweepPoint` metric field and constructor assignment:

```python
@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    recall: float
    relative_density: float
    mmd_ratio: dict[str, float]
```

```python
SweepPoint(
    threshold=float(t),
    recall=recall,
    relative_density=report.relative_density,
    mmd_ratio=dict(report.mmd_ratio),
)
```

- [ ] **Step 4: Make the composite consume ratios directly**

Replace `graph_similarity` in `src/eval/composite.py` with:

```python
def graph_similarity(
    mmd_ratio: Mapping[str, float],
    definition: CompositeDefinition,
) -> float:
    """Compute exp(-sum_k w_k * normalized_mmd_ratio_k)."""
    total = 0.0
    for stat in definition.statistics:
        total += definition.weights[stat] * mmd_ratio[stat]
    return float(np.exp(-total))
```

Update `tests/eval/test_composite.py` so `definition.scales` remains present for dataclass compatibility during this task but does not affect the result; assert ratios `{degree: 0.1, clustering: 0.2, spectral: 0.3}` with equal weights yield `exp(-0.2)` even if scales are not all one. Remove old `MMDConfig(degree_max=...)` arguments.

- [ ] **Step 5: Replace G1 row dataclasses and assembly wiring**

Replace the metric fields of `SweepRow` and `AssembledRow` in `src/experiments/g1_hardened_e2.py`:

```python
@dataclass(frozen=True)
class SweepRow:
    threshold: float
    recall: float
    relative_density: float
    mmd_ratio: dict[str, float]
    self_loop_count: int
    self_loop_rate: float


@dataclass(frozen=True)
class AssembledRow:
    threshold: float | None
    mmd_ratio: dict[str, float]
    raw_mmd2: dict[str, float]
    reference_mmd2: dict[str, float]
    relative_density: float
    self_loops_pred: int
    self_loops_ref: int
    bootstrap_mean: dict[str, float]
    bootstrap_std: dict[str, float]
    composite: float | None
```

In `run_threshold_sweep`, assign `mmd_ratio=dict(point.mmd_ratio)`.

In `assemble_and_evaluate`, replace the composite and return block with:

```python
    composite = graph_similarity(report.mmd_ratio, definition) if composite_valid else None
    return AssembledRow(
        threshold=threshold,
        mmd_ratio=dict(report.mmd_ratio),
        raw_mmd2=dict(report.raw_mmd2),
        reference_mmd2=dict(report.reference_mmd2),
        relative_density=report.relative_density,
        self_loops_pred=report.self_loops_pred,
        self_loops_ref=report.self_loops_ref,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
        composite=composite,
    )
```

- [ ] **Step 6: Remove double calibration and disclose the fixed metric**

Delete `calibrate_tau`. In `run_g1_pipeline`, replace its call with:

```python
    nf = noise_floor(g_ref, buckets, config)
    tau = {stat: 1.0 for stat in STATISTICS}
```

Keep `CompositeDefinition.scales=tau` until a later cleanup because this plan does not restructure that dataclass. Add these metadata entries alongside `mmd_config`:

```python
        "metric_normalization": "ratio_of_size_mean_mmd2",
        "reference_split": "artifact_order_even_vs_odd_within_each_node_size",
        "canonical_metric": "mmd_ratio",
        "component_disclosure": ["raw_mmd2", "reference_mmd2", "mmd_ratio"],
```

Change the composite calibration metadata to:

```python
"calibration_rule": "no second calibration: MMD ratios are already reference-normalized"
```

Remove `_SMALL_CONFIG_KWARGS["degree_max"]` and every `MMDConfig(degree_max=...)` argument in the touched tests.

- [ ] **Step 7: Replace Markdown table headers and fields**

In `render_tables_markdown`, make the threshold table use `row["mmd_ratio"]` and the assembled table use `row_dict["mmd_ratio"]`. Replace their metric headers with `degree MMD ratio`, `clustering MMD ratio`, and `spectral MMD ratio`.

After the assembled table, append this component table:

```python
    lines.append("## MMD ratio components")
    lines.append("")
    lines.append(
        "| scorer | statistic | raw numerator | reference denominator | normalized ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for scorer, assembled_row in assembled.items():
        if assembled_row is None:
            continue
        row_dict = cast(dict[str, object], assembled_row)
        raw = cast(dict[str, float], row_dict["raw_mmd2"])
        reference = cast(dict[str, float], row_dict["reference_mmd2"])
        ratio = cast(dict[str, float], row_dict["mmd_ratio"])
        for stat in STATISTICS:
            lines.append(
                f"| {scorer} | {stat} | {_fmt(raw[stat])} | "
                f"{_fmt(reference[stat])} | {_fmt(ratio[stat])} |"
            )
    lines.append("")
```

- [ ] **Step 8: Run downstream tests**

Run:

```bash
rtk proxy uv run pytest tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py -q
```

Expected: all selected tests pass and no serialized result contains `aggregate_mmd2` or threshold-row `mmd2`.

- [ ] **Step 9: Commit downstream replacement**

Run:

```bash
rtk git add src/eval/assembly.py src/eval/composite.py src/experiments/g1_hardened_e2.py tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py
rtk git diff --cached --check
rtk git commit -m "refactor: make normalized MMD ratio canonical"
```

Expected: the commit removes the old result keys rather than retaining aliases.

---

### Task 4: Verify official-data scale and repository integrity

**Files:**
- Modify only if verification exposes a defect in Tasks 2 or 3; corrections stay within the files already listed.

**Interfaces:**
- Consumes: The completed evaluator and the read-only official Arath graph/bucket artifacts under `/Users/richardwang/Documents/grand/PRING`.
- Produces: Evidence that reference denominators match the independently measured scale and that the full repository remains type-, lint-, and test-clean.

- [ ] **Step 1: Verify the official Arath reference denominators**

Run this read-only verification from the current repository:

```bash
rtk proxy uv run python -c 'import pickle; from pathlib import Path; from src.eval.graph_metrics import MMDConfig, noise_floor; base=Path("/Users/richardwang/Documents/grand/PRING/data_process/pring_dataset/arath"); g=pickle.loads((base/"arath_test_graph.pkl").read_bytes()); files={"BFS":"arath_BFS_sampled_nodes.pkl","DFS":"arath_DFS_sampled_nodes.pkl","RW":"arath_RANDOM_WALK_sampled_nodes.pkl"}; expected={"BFS":{"degree":0.015331448015001858,"clustering":0.013107044351804676,"spectral":0.014301673896181399},"DFS":{"degree":0.002210707375096832,"clustering":0.0032868321040904203,"spectral":0.011222664293332163},"RW":{"degree":0.007537813277395444,"clustering":0.009945100977063647,"spectral":0.00986206743851339}}; import numpy as np; [(lambda buckets, name: [np.testing.assert_allclose(np.mean([floor[stat] for floor in noise_floor(g,buckets,MMDConfig()).values()]),expected[name][stat],rtol=1e-10,atol=1e-12) for stat in expected[name]])(pickle.loads((base/file).read_bytes()),name) for name,file in files.items()]; print("official Arath reference floors match")'
```

Expected: `official Arath reference floors match`.

- [ ] **Step 2: Scan for removed evaluator concepts and output keys**

Run:

```bash
rtk proxy rg -n "median_bandwidth|bandwidth_scales|degree_max|gaussian_rbf|aggregate_mmd2|\bmmd2\b" src tests
```

Expected: no production-code matches. Test names may contain `mmd_squared`, which is the intended raw component function.

- [ ] **Step 3: Run focused formatting and static checks**

Run:

```bash
rtk proxy uv run ruff format --check src/eval/graph_metrics.py src/eval/assembly.py src/eval/composite.py src/experiments/g1_hardened_e2.py tests/eval/test_graph_metrics.py tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py
rtk proxy uv run ruff check src/eval/graph_metrics.py src/eval/assembly.py src/eval/composite.py src/experiments/g1_hardened_e2.py tests/eval/test_graph_metrics.py tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py
rtk proxy uv run mypy src tests
```

Expected: all three commands exit successfully. Run mypy once; do not start a concurrent invocation against the same cache.

- [ ] **Step 4: Run the full test suite**

Run:

```bash
rtk proxy uv run pytest
```

Expected: the full suite passes.

- [ ] **Step 5: Run a cached-score G1 evaluation into a new output directory**

On the fixed execution host, run:

```bash
rtk proxy hpc/run.sh g1 --universe scores/b0_v31_candidate.npz --alt-universe scores/b0_alt_candidate.npz --data-root data --strategy breadth_first --output-dir outputs/g1_mmd_ratio
```

Expected: `outputs/g1_mmd_ratio/g1_results.json` and `g1_tables.md` are newly written; existing `outputs/g1` artifacts are untouched.

- [ ] **Step 6: Validate the new result artifact schema**

Run:

```bash
rtk proxy rg -n '"mmd_ratio"|"raw_mmd2"|"reference_mmd2"|"metric_normalization"' outputs/g1_mmd_ratio/g1_results.json
rtk proxy rg -n 'aggregate_mmd2|"mmd2"' outputs/g1_mmd_ratio/g1_results.json
```

Expected: the first command finds all four disclosures; the second command returns no matches.

- [ ] **Step 7: Inspect the final diff and commit any verification-only corrections**

Run:

```bash
rtk git status --short
rtk git diff --check
rtk git diff --stat
```

Expected: only the planned evaluator, tests, and plan-owned documentation hunks are present. If Tasks 2 or 3 required corrections, stage only those files and commit with:

```bash
rtk git add src/eval/graph_metrics.py src/eval/assembly.py src/eval/composite.py src/experiments/g1_hardened_e2.py tests/eval/test_graph_metrics.py tests/eval/test_assembly.py tests/eval/test_composite.py tests/test_g1_hardened_e2.py
rtk git commit -m "fix: verify normalized topology evaluation"
```

Expected: no generated `outputs/` files are staged.

## Self-Review Results

- Spec coverage: the plan covers the official descriptors, Gaussian-TV kernel, fixed sigma, biased estimator, deterministic reference split, ratio-of-means aggregation, output disclosure, bootstrap semantics, composite semantics, rendering, and live-data validation.
- Scope control: the old evaluator is removed directly; there is no compatibility mode, alternate named profile, or unrelated model/training change.
- Result integrity: existing E2 values are not renamed. The new evaluator writes to a fresh run directory, after which result-document synchronization requires a separate current-run-only review.
- Type consistency: `mmd_squared` returns `float`; `BucketedMMDReport` exposes `raw_mmd2`, `reference_mmd2`, and `mmd_ratio`; `SweepPoint`, `SweepRow`, and `AssembledRow` consistently use `mmd_ratio`.
- Placeholder scan: every code-editing step includes exact replacement content, commands, and expected outcomes.
