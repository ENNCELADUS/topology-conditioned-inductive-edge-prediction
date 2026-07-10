# Single-H20 Loader Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the measured host-input stalls in B0 V3.1 training and enable PyTorch's optimized scaled-dot-product-attention path without changing the experiment.

**Architecture:** Preload the operative raw token tensors once into the existing `FeatureStore` on the host before any DataLoader worker is created, then serve cached tensors through the frozen four-worker/pinned-memory loader contract and non-blocking device copies. Separately, mark every custom `nn.MultiheadAttention` call whose weights are discarded with `need_weights=False`. Keep sampling, batches, losses, model configuration, and the single-H20 execution contract unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.10, HF Accelerate 1.13, pytest, Ruff, mypy, uv-managed lockfile.

## Global Constraints

- `docs/06-egostitch-spec.md` is frozen: code must align to it; do not change the contract to fit the implementation.
- The F1 path remains length-bucketed with boundaries `{128, 256, 384, 512, 768, 1024}` and token budget `131,072`.
- The production F1 loader must use exactly `num_workers = 4`, `persistent_workers = True`, `prefetch_factor = 4`, and pinned memory.
- Preload only `assembled.operative_node_ids`, keep tensors on CPU, and perform the preload before DataLoader workers start.
- Host-to-device copies must request `non_blocking=True`; batch keys, dtypes, padding, labels, and model inputs remain unchanged.
- Training remains single-process, single-H20, BF16, with `negative_ratio = 5`, `train_positives = train_plus`, seed 47, and no OHEM.
- Do not increase the token budget, change the optimizer/scheduler, add `torch.compile`, alter checkpoint format, or add resume behavior.
- Every custom `nn.MultiheadAttention` invocation that discards its returned weights must pass `need_weights=False`; no architecture or dropout changes.
- Use TDD, surgical changes, and one task commit. Prefix shell commands with `rtk`. Run no concurrent mypy processes.
- Work from `/Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.worktrees/loader-throughput` and invoke tools through `/Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv`.

---

### Task 1: Cached FeatureStore and Frozen F1 Loader Contract

**Files:**
- Modify: `src/data/features.py`
- Modify: `src/train_b0.py`
- Modify: `configs/b0_v31_breadth_first.yaml`
- Modify: `tests/data/test_features.py`
- Modify: `tests/test_train_b0.py`

**Interfaces:**
- Produces: `FeatureStore.preload(node_ids: Iterable[str] | None = None) -> int`.
- Produces: `FeatureStore.cached_node_count: int` and `FeatureStore.cached_bytes: int` read-only properties.
- Produces: `_v3_loader_options(num_workers: int) -> dict[str, object]` in `src.train_b0`.
- Preserves: `FeatureStore.load_tokens(node_id: str) -> torch.Tensor`; it becomes cache-aware but retains all existing validation and exceptions.

- [ ] **Step 1: Add failing FeatureStore cache tests**

Add tests that wrap `torch.load` with a counting function, preload two synthetic nodes, then call `load_tokens` again. Assert exactly two loads, `cached_node_count == 2`, `cached_bytes == sum(tensor.numel() * tensor.element_size())`, and object identity on repeated access. Add a subset test proving `preload([one_node])` does not load unrelated nodes and a missing-node test preserving `KeyError`.

- [ ] **Step 2: Add failing loader-contract and non-blocking-copy tests**

Import `_to_device` and the new `_v3_loader_options` in `tests/test_train_b0.py`. Assert:

```python
assert _v3_loader_options(4) == {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
assert _v3_loader_options(0) == {
    "num_workers": 0,
    "pin_memory": True,
    "persistent_workers": False,
}
```

Use `Mock(spec=torch.Tensor)` plus `cast(torch.Tensor, mock)` to verify `_to_device` calls `.to(device, non_blocking=True)`. Add a focused `_build_v3_1_loaders` test whose monkeypatched `store.preload` records `("preload", tuple(node_ids))` and whose monkeypatched `DataLoader` records `("loader",)`, then assert the preload event is first and receives exactly `assembled.operative_node_ids`. Add a shipped-config test asserting `configs/b0_v31_breadth_first.yaml` loads with four workers.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/data/test_features.py tests/test_train_b0.py -q
```

Expected: failures because the cache API, loader options, preload call, production worker count, and non-blocking copy do not yet exist.

- [ ] **Step 4: Implement the in-memory host cache**

Initialize a private `dict[str, torch.Tensor]` in `FeatureStore`. Refactor the existing disk-load and validation logic into one cache-miss path so validation still happens before insertion. `preload()` must iterate a deterministic node list, use `load_tokens()`, log progress every 1,000 newly cached nodes, and return the total cached count. Compute `cached_bytes` from the cached tensors without maintaining a second mutable counter.

- [ ] **Step 5: Implement the loader contract and preload placement**

Set the shipped V3.1 config to `num_workers: 4`. Add `_v3_loader_options()` with the exact dictionaries above and spread it into both V3.1 DataLoaders. At the beginning of `_build_v3_1_loaders`, before either DataLoader is created, call:

```python
cached_nodes = assembled.store.preload(assembled.operative_node_ids)
logger.info(
    "preloaded %d operative node tensors (%.2f GiB) into host memory",
    cached_nodes,
    assembled.store.cached_bytes / float(1024**3),
)
```

Change `_to_device()` to pass `non_blocking=True`. Do not alter F0 loaders.

- [ ] **Step 6: Run focused and real-data tests**

Run the focused command from Step 3, then:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/data/test_features.py tests/test_train_b0.py::TestRealDataV31Assembly -q
rtk proxy /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/ruff check src/data/features.py src/train_b0.py tests/data/test_features.py tests/test_train_b0.py
```

Expected: all selected tests pass and Ruff reports no errors. The real-data test must not preload the full cache unless it explicitly calls `_build_v3_1_loaders`.

- [ ] **Step 7: Run the full suite once and commit**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest -q
```

Commit only the five task files:

```bash
rtk git add src/data/features.py src/train_b0.py configs/b0_v31_breadth_first.yaml tests/data/test_features.py tests/test_train_b0.py
rtk git commit -m "perf: preload V3.1 features and pipeline batches"
```

### Task 2: Optimized Custom Attention Calls

**Files:**
- Modify: `src/model/B0.py`
- Create: `tests/test_b0_attention.py`

**Interfaces:**
- Consumes: existing `CrossAttentionLayer` and `BlockSelfMixingLayer` forward contracts.
- Preserves: all tensor shapes, masks, residual paths, dropout modules, AB/BA aggregation, and output dictionary format.

- [ ] **Step 1: Add failing call-contract tests**

Create small recording `nn.Module` attention doubles that accept `need_weights`, record its value, and return `(query, None)`. Cover these three discarded-weight call sites:

1. `CrossAttentionLayer._attend()` through a forward call.
2. `CrossAttentionLayer.attn_cls` through the same forward call.
3. `BlockSelfMixingLayer._self_attend()` through a forward call.

Assert every recorded value is exactly `False`, and assert the public output shapes remain unchanged.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/test_b0_attention.py -q
```

Expected: the recording double observes the default/missing `need_weights`, not `False`.

- [ ] **Step 3: Add `need_weights=False` to all three call sites**

Use explicit keyword arguments on `self.attn(...)` and `self.attn_cls(...)`. Do not replace the attention modules, refactor `block_self`, cache AB/BA results, or change masks.

- [ ] **Step 4: Run model-focused checks**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/test_b0_attention.py tests/data/test_pairs.py tests/test_train_b0.py::TestTrainLoopTinyV3_1 tests/test_score_universe.py -q
rtk proxy /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/ruff check src/model/B0.py tests/test_b0_attention.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 5: Run the full suite once and commit**

Run the full pytest command from Task 1 Step 7, then commit:

```bash
rtk git add src/model/B0.py tests/test_b0_attention.py
rtk git commit -m "perf: skip unused V3.1 attention weights"
```

### Task 3: Runtime Documentation and Integration Verification

**Files:**
- Modify: `hpc/README.md`
- Modify: `docs/06-egostitch-spec.md`
- Modify: `src/README.md`
- Test: `tests/test_hpc_scripts.py`

**Interfaces:**
- Consumes: Task 1's preload log/API and exact loader options.
- Consumes: Task 2's `need_weights=False` implementation.
- Produces: operator-facing expectations for preload startup, memory use, training launch, and first-step verification.

- [ ] **Step 1: Add or update documentation assertions**

In `tests/test_hpc_scripts.py`, add focused text assertions for the H20 runbook: it must mention the one-time host preload, four workers, pinned/prefetched batches, the expected startup pause before step logs, and the exact production config path. Keep assertions semantic and avoid pinning prose paragraphs verbatim.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/test_hpc_scripts.py -q
```

Expected: new runbook assertions fail.

- [ ] **Step 3: Update the relevant docs**

Update `hpc/README.md` with:

- the one-time ~25 GiB CPU preload and the verified host-memory requirement;
- exact loader values `4 / True / 4 / True` for workers/persistence/prefetch/pinning;
- a warning that no step logs are expected until preload completes;
- the existing disconnect-safe command unchanged;
- post-launch checks for the preload log, advancing steps, GPU memory, and absence of NaN/traceback.

Add a 2026-07-10 change-log line to `docs/06-egostitch-spec.md` stating that the B0 F1 implementation was aligned to the already-frozen loader contract and unused custom attention weights were disabled; do not change normative values. Replace the stale `src/README.md` sentence claiming nothing is implemented with a concise current pipeline map that mentions the cached F1 path.

- [ ] **Step 4: Run documentation and repository gates**

Run:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest tests/test_hpc_scripts.py -q
rtk proxy /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/ruff check .
rtk proxy /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/ruff format --check .
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/mypy src tests
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/richardwang/Documents/topology-conditioned-inductive-edge-prediction/.venv/bin/python -m pytest -q
```

Expected: all gates pass. Run mypy alone, never concurrently with another mypy process.

- [ ] **Step 5: Commit**

Commit the four task files:

```bash
rtk git add hpc/README.md docs/06-egostitch-spec.md src/README.md tests/test_hpc_scripts.py
rtk git commit -m "docs: document optimized H20 input pipeline"
```
