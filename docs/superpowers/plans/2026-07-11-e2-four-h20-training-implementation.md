# E2 Four-H20 Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved cold-start E2 B0 V3.1 pipeline that packs frozen features, selects a large-batch operating point, and completes 30 training/validation epochs in at most 60 minutes on 4 x NVIDIA H20 GPUs.

**Architecture:** Sixteen CPU pack workers convert the per-node FP32 `.pt` cache into validated BF16 flat shards. Four Accelerate DDP ranks each load the complete packed feature table, while four lightweight DataLoader workers per rank prefetch compact pair indices; token gather and padding happen on the GPU. A single `src.e2_pipeline` entry performs `pack -> candidate probes -> one-epoch projection -> fixed 30-epoch train`, then preserves the existing checkpoint contract for `score_universe` and G1/G2.

**Tech Stack:** Python 3.11, PyTorch 2.10.0, CUDA 12.8, Hugging Face Accelerate 1.13.0, NumPy 2.3.5, PyYAML 6.0.3, pytest 9.0.2, ruff 0.15.0, mypy 1.19.1, Bash, 4 x NVIDIA H20 (97,871 MiB each).

## Global Constraints

- Update the frozen contract in `docs/06-egostitch-spec.md`, including a dated change-log line, before modifying runtime code.
- Scope is the implemented E2 B0 V3.1 path; do not implement EgoStitch or change universe scoring, G1/G2 logic, thresholds, metrics, or checkpoint payload keys.
- The formal run uses the fixed balanced E2 training rows, BF16, `lr = 1e-4`, 30 epochs, validation every epoch, and no early termination.
- Report validation AUROC/AUPRC, but do not use quality to accept or reject the first throughput configuration.
- The timed cold run starts with a new empty derived-cache path and includes first pack construction, probes, 30 train/validation epochs, and artifacts.
- Wall-time allocation is pack 300 s, setup/cache/probe 300 s, train plus validation 2,820 s, artifacts 60 s, reserve 120 s, total 3,600 s.
- Use 16 pack workers, 4 loader workers per rank, `persistent_workers=True`, `prefetch_factor=4`, and no CUDA tensors in DataLoader workers.
- Probe per-rank token budgets `[262144, 524288, 1048576, 1572864]`, cap each rank at 4,096 pairs, and reject candidates above 85 GiB peak memory.
- Formal E2 execution uses exactly 4 Accelerate processes and 4 H20 GPUs with `find_unused_parameters=False`, `gradient_as_bucket_view=True`, and `broadcast_buffers=False`.
- Every training and validation row must appear exactly once; no row may be silently dropped or duplicated.
- Preserve the existing `best.pt` / `last.pt` payload consumed by `src.score_universe`.
- Preserve the user's pre-existing untracked `docs/results/G1-G2-b0-v31-breadth-first-20260711.md`; no task may stage, edit, or delete it.
- All shell commands in this plan use `rtk` as required by the repository instructions.

---

## File and ownership map

| File | Responsibility |
|---|---|
| `docs/06-egostitch-spec.md` | Frozen four-H20/DDP execution contract and change log |
| `docs/03-experiment-protocol.md` | E2 fixed-30-epoch throughput acceptance rule |
| `src/data/packed_features.py` | Manifest schema, atomic BF16 pack construction, validation, GPU table, gather/padding |
| `src/data/distributed_pairs.py` | Deterministic global length-bucket plans, rank partitioning, compact batch dataset |
| `src/train_b0.py` | Runtime config, packed loaders, sample-based warm-up, DDP train/eval, rank-zero checkpoints |
| `src/e2_pipeline.py` | Probe selection, deadlines, subprocess orchestration, profiles, failure reports |
| `configs/b0_v31_breadth_first.yaml` | The sole production E2 runtime configuration |
| `hpc/run.sh` | Fixed-container four-H20 check and single E2 production entry |
| `hpc/README.md`, `README.md`, `CLAUDE.md` | User/contributor execution guidance |
| `tests/data/test_packed_features.py` | Pack schema, corruption, BF16 conversion, gather/padding tests |
| `tests/data/test_distributed_pairs.py` | Coverage, determinism, rank-step, token-cap, compact dataset tests |
| `tests/test_train_b0.py` | Runtime schema, loaders, warm-up, loss scaling, DDP loop, artifacts |
| `tests/test_e2_pipeline.py` | Probe selection, projection, deadlines, subprocess command tests |
| `tests/test_e2_ddp_integration.py` | CPU multi-process and opt-in four-H20 acceptance checks |
| `tests/test_hpc_scripts.py` | Runner and documentation contract checks |

## Parallel execution map

Use a separate implementation worktree/branch per task when tasks share a wave. Merge only after every task in the wave passes its own tests and review gate.

| Wave | Tasks | Dependency rule |
|---|---|---|
| 0 | Task 1 | Governance gate; merge before code tasks |
| 1 | Tasks 2, 3, 5 in parallel | They modify disjoint production/test files |
| 2 | Tasks 4 and 7 in parallel | Task 4 requires Task 3; Task 7 requires Task 2 |
| 3 | Task 6 | Requires Task 4 |
| 4 | Task 8 | Requires Tasks 2, 5, 6 |
| 5 | Task 9 | Requires Task 8 |
| 6 | Task 10 | Requires Tasks 7 and 9 |
| 7 | Tasks 11 and 12 in parallel | Both require Task 10 and modify disjoint files |
| 8 | Final merge verification | Run after Tasks 11 and 12 are merged |

---

### Task 1: Authorize the four-H20 runtime in the frozen specifications

**Files:**
- Modify: `docs/06-egostitch-spec.md:385-483`
- Modify: `docs/03-experiment-protocol.md:1-12,167-190,322-331`
- Modify: `tests/test_hpc_scripts.py`

**Interfaces:**
- Consumes: Approved design `docs/superpowers/specs/2026-07-11-e2-training-throughput-design.md`.
- Produces: Binding permission for Tasks 2-12 to replace single-H20 E2 execution with four-H20 Accelerate DDP.

- [ ] **Step 1: Add the failing specification test**

Append this test to `tests/test_hpc_scripts.py`:

```python
def test_frozen_specs_pin_four_h20_e2_training() -> None:
    spec = (REPO_ROOT / "docs" / "06-egostitch-spec.md").read_text()
    protocol = (REPO_ROOT / "docs" / "03-experiment-protocol.md").read_text()

    assert "4 × NVIDIA H20" in spec
    assert "accelerate launch --num_processes 4" in spec
    assert "60 minutes" in spec
    assert "30 epochs" in spec
    assert "validation after every epoch" in spec
    assert "fixed 30-epoch" in protocol
    assert "quality is reported but is not the throughput acceptance gate" in protocol
```

- [ ] **Step 2: Run the test and verify the old contract fails**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/test_hpc_scripts.py::test_frozen_specs_pin_four_h20_e2_training -q
```

Expected: FAIL because the current spec pins one H20 and prohibits DDP.

- [ ] **Step 3: Replace the execution section and add the protocol rule**

Replace spec section 11 with text that contains these exact contract statements:

```markdown
## 11. E2 production execution design (4 × H20, Hugging Face Accelerate DDP)

The formal E2 B0 V3.1 run uses 4 × NVIDIA H20 and is launched with
`accelerate launch --num_processes 4`. A cold acceptance run includes first BF16
feature-pack construction, bounded batch probes, exactly 30 epochs, validation after
every epoch, and final artifacts. The complete interval must be at most 60 minutes.

Each rank owns one model/optimizer replica and one complete GPU-resident BF16 feature
table. DataLoader workers transfer compact endpoint indices only. Training and
validation coverage are exact; tail-batch loss is weighted by local/global pair count.
The checkpoint payload consumed by `score_universe` is unchanged.
```

Add this E2 protocol paragraph:

```markdown
**Execution acceptance:** E2 uses a fixed 30-epoch four-H20 throughput run. Validation
is executed after every epoch. Quality is reported but is not the throughput acceptance
gate for the first systems-optimization pass; the wall-clock gate is 60 minutes from an
empty derived-cache path through final training artifacts.
```

Add this dated spec change-log entry:

```markdown
- 2026-07-11: replaced the formal E2 single-H20 path with the approved 4 × H20
  Accelerate DDP packed-feature pipeline; fixed the cold-run budget at 60 minutes for
  30 epochs with validation after every epoch. The scorer and checkpoint contracts did
  not change.
```

- [ ] **Step 4: Run the specification test**

Run:

```bash
rtk proxy .venv/bin/python -m pytest tests/test_hpc_scripts.py::test_frozen_specs_pin_four_h20_e2_training -q
```

Expected: PASS.

- [ ] **Step 5: Commit the governance change**

```bash
rtk git add docs/06-egostitch-spec.md docs/03-experiment-protocol.md tests/test_hpc_scripts.py
rtk git commit -m "docs: authorize four-H20 E2 training"
```

---

### Task 2: Add the validated E2 runtime configuration

**Files:**
- Modify: `src/train_b0.py:61-164,234-350`
- Modify: `configs/b0_v31_breadth_first.yaml:29-51`
- Modify: `tests/test_train_b0.py:38-145`

**Interfaces:**
- Consumes: Task 1 four-H20 contract.
- Produces: `RuntimeConfig`, optional `Config.runtime`, and the sole production runtime values used by Tasks 7-10.

- [ ] **Step 1: Write failing runtime-schema tests**

Add these imports/assertions and tests to `tests/test_train_b0.py`:

```python
def _runtime_dict() -> dict[str, object]:
    return {
        "world_size": 4,
        "pack_dir": "outputs/feature_packs/b0_v31_bf16",
        "pack_workers": 16,
        "loader_workers_per_rank": 4,
        "prefetch_factor": 4,
        "token_budget_candidates": [262144, 524288, 1048576, 1572864],
        "max_pairs_per_rank": 4096,
        "memory_limit_gib": 85.0,
        "total_budget_seconds": 3600,
        "pack_budget_seconds": 300,
        "setup_probe_budget_seconds": 300,
        "train_eval_budget_seconds": 2820,
        "artifact_budget_seconds": 60,
        "reserve_seconds": 120,
        "probe_warmup_steps": 10,
        "probe_timed_steps": 30,
    }


def test_loads_four_h20_runtime_config(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"runtime": _runtime_dict()})

    cfg = load_config(config_path)

    assert cfg.runtime is not None
    assert cfg.runtime.world_size == 4
    assert cfg.runtime.token_budget_candidates == [262144, 524288, 1048576, 1572864]
    assert cfg.runtime.total_budget_seconds == 3600


def test_runtime_budget_must_sum_to_total(tmp_path: Path) -> None:
    runtime = _runtime_dict()
    runtime["reserve_seconds"] = 119
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"runtime": runtime})

    with pytest.raises(ValueError, match="runtime stage budgets must sum to 3600"):
        load_config(config_path)
```

Update `_write_yaml_config` so a top-level override replaces the whole value:

```python
if key:
    cast_section = base[section]
    assert isinstance(cast_section, dict)
    cast_section[key] = value
else:
    base[dotted_key] = value
```

- [ ] **Step 2: Verify the tests fail**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py -k "runtime_config or runtime_budget" -q
```

Expected: FAIL because `runtime` is currently an unknown top-level key.

- [ ] **Step 3: Implement the runtime dataclass and parser**

Add this exact public shape to `src/train_b0.py`:

```python
@dataclass(frozen=True)
class RuntimeConfig:
    world_size: int
    pack_dir: Path
    pack_workers: int
    loader_workers_per_rank: int
    prefetch_factor: int
    token_budget_candidates: list[int]
    max_pairs_per_rank: int
    memory_limit_gib: float
    total_budget_seconds: int
    pack_budget_seconds: int
    setup_probe_budget_seconds: int
    train_eval_budget_seconds: int
    artifact_budget_seconds: int
    reserve_seconds: int
    probe_warmup_steps: int
    probe_timed_steps: int
```

Add `runtime: RuntimeConfig | None = None` as the last field of `Config`. Permit
`runtime` at the top level, parse it only when present, reject unknown runtime keys,
and enforce:

```python
def _as_int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"config key '{name}' must be a list of integers")
    return [int(item) for item in value]


runtime = RuntimeConfig(
    world_size=_as_int(_require(runtime_raw, "world_size", "runtime."), "runtime.world_size"),
    pack_dir=Path(_as_str(_require(runtime_raw, "pack_dir", "runtime."), "runtime.pack_dir")),
    pack_workers=_as_int(_require(runtime_raw, "pack_workers", "runtime."), "runtime.pack_workers"),
    loader_workers_per_rank=_as_int(
        _require(runtime_raw, "loader_workers_per_rank", "runtime."),
        "runtime.loader_workers_per_rank",
    ),
    prefetch_factor=_as_int(
        _require(runtime_raw, "prefetch_factor", "runtime."), "runtime.prefetch_factor"
    ),
    token_budget_candidates=_as_int_list(
        _require(runtime_raw, "token_budget_candidates", "runtime."),
        "runtime.token_budget_candidates",
    ),
    max_pairs_per_rank=_as_int(
        _require(runtime_raw, "max_pairs_per_rank", "runtime."), "runtime.max_pairs_per_rank"
    ),
    memory_limit_gib=_as_float(
        _require(runtime_raw, "memory_limit_gib", "runtime."), "runtime.memory_limit_gib"
    ),
    total_budget_seconds=_as_int(
        _require(runtime_raw, "total_budget_seconds", "runtime."),
        "runtime.total_budget_seconds",
    ),
    pack_budget_seconds=_as_int(
        _require(runtime_raw, "pack_budget_seconds", "runtime."), "runtime.pack_budget_seconds"
    ),
    setup_probe_budget_seconds=_as_int(
        _require(runtime_raw, "setup_probe_budget_seconds", "runtime."),
        "runtime.setup_probe_budget_seconds",
    ),
    train_eval_budget_seconds=_as_int(
        _require(runtime_raw, "train_eval_budget_seconds", "runtime."),
        "runtime.train_eval_budget_seconds",
    ),
    artifact_budget_seconds=_as_int(
        _require(runtime_raw, "artifact_budget_seconds", "runtime."),
        "runtime.artifact_budget_seconds",
    ),
    reserve_seconds=_as_int(
        _require(runtime_raw, "reserve_seconds", "runtime."), "runtime.reserve_seconds"
    ),
    probe_warmup_steps=_as_int(
        _require(runtime_raw, "probe_warmup_steps", "runtime."), "runtime.probe_warmup_steps"
    ),
    probe_timed_steps=_as_int(
        _require(runtime_raw, "probe_timed_steps", "runtime."), "runtime.probe_timed_steps"
    ),
)

stage_total = (
    runtime.pack_budget_seconds
    + runtime.setup_probe_budget_seconds
    + runtime.train_eval_budget_seconds
    + runtime.artifact_budget_seconds
    + runtime.reserve_seconds
)
if runtime.world_size != 4:
    raise ValueError("runtime.world_size must be 4 for formal E2 training")
if stage_total != runtime.total_budget_seconds:
    raise ValueError(
        f"runtime stage budgets must sum to {runtime.total_budget_seconds}; got {stage_total}"
    )
if runtime.token_budget_candidates != [262144, 524288, 1048576, 1572864]:
    raise ValueError("runtime.token_budget_candidates must match the frozen E2 probe set")
```

Add the exact `runtime:` mapping from `_runtime_dict()` to
`configs/b0_v31_breadth_first.yaml`. Keep B0-alt valid without a runtime section.

- [ ] **Step 4: Run schema and regression tests**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py -k "LoadConfig or runtime" -q
rtk proxy .venv/bin/ruff check src/train_b0.py tests/test_train_b0.py
rtk proxy .venv/bin/mypy src/train_b0.py tests/test_train_b0.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/train_b0.py configs/b0_v31_breadth_first.yaml tests/test_train_b0.py
rtk git commit -m "feat: add E2 runtime configuration"
```

---

### Task 3: Define and validate the packed-feature manifest

**Files:**
- Create: `src/data/packed_features.py`
- Create: `tests/data/test_packed_features.py`

**Interfaces:**
- Consumes: Existing source `metadata.json`, `index.json`, and FP32 per-node `.pt` files.
- Produces: `PackedNodeRecord`, `PackedShardRecord`, `PackedFeatureManifest`, `load_packed_manifest()`, `write_packed_manifest()`, and `validate_packed_manifest()` for Tasks 4 and 6.

- [ ] **Step 1: Write failing manifest round-trip and corruption tests**

Create `tests/data/test_packed_features.py` with a synthetic feature-root helper and these tests:

```python
def _write_minimal_pack(pack_root: Path, source_root: Path) -> Path:
    pack_root.mkdir()
    shard_path = pack_root / "shard-000.bin"
    shard_path.write_bytes(bytes(24))
    manifest = PackedFeatureManifest(
        format="bf16_flat_shards_v1",
        input_dim=4,
        dtype="bfloat16",
        source_metadata_sha256=sha256_file(source_root / "metadata.json"),
        source_index_sha256=sha256_file(source_root / "index.json"),
        nodes=(PackedNodeRecord("node_a", 0, 0, 0, 3),),
        shards=(
            PackedShardRecord(
                "shard-000.bin", 3, 24, sha256_file(shard_path)
            ),
        ),
        pack_workers=1,
        build_seconds=0.0,
    )
    write_packed_manifest(pack_root, manifest)
    return pack_root


def test_manifest_round_trip_preserves_node_order(tmp_path: Path) -> None:
    manifest = PackedFeatureManifest(
        format="bf16_flat_shards_v1",
        input_dim=4,
        dtype="bfloat16",
        source_metadata_sha256="a" * 64,
        source_index_sha256="b" * 64,
        nodes=(PackedNodeRecord("node_a", 0, 0, 0, 3),),
        shards=(PackedShardRecord("shard-000.bin", 3, 24, "c" * 64),),
        pack_workers=1,
        build_seconds=0.25,
    )
    pack_root = tmp_path / "pack"
    pack_root.mkdir()

    write_packed_manifest(pack_root, manifest)

    assert load_packed_manifest(pack_root) == manifest
    assert manifest.node_index() == {"node_a": 0}


def test_manifest_rejects_changed_source_hash(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    (source_root / "metadata.json").write_text('{"format":"changed"}')

    with pytest.raises(ValueError, match="source metadata hash"):
        validate_packed_manifest(pack_root, source_root)
```

- [ ] **Step 2: Verify the module is missing**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py -q
```

Expected: collection ERROR with `ModuleNotFoundError: src.data.packed_features`.

- [ ] **Step 3: Implement the manifest types and strict validation**

Implement these exact types and functions:

```python
PACK_FORMAT = "bf16_flat_shards_v1"


@dataclass(frozen=True)
class PackedNodeRecord:
    node_id: str
    shard_index: int
    shard_offset: int
    global_offset: int
    length: int


@dataclass(frozen=True)
class PackedShardRecord:
    filename: str
    num_tokens: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class PackedFeatureManifest:
    format: str
    input_dim: int
    dtype: str
    source_metadata_sha256: str
    source_index_sha256: str
    nodes: Sequence[PackedNodeRecord]
    shards: Sequence[PackedShardRecord]
    pack_workers: int
    build_seconds: float

    def node_index(self) -> dict[str, int]:
        return {record.node_id: index for index, record in enumerate(self.nodes)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Serialize with `dataclasses.asdict`, write `manifest.json.tmp`, then `os.replace` it
with `manifest.json`. `load_packed_manifest()` must reconstruct tuples and dataclasses.
`validate_packed_manifest(pack_root: Path, source_root: Path | None)` must always check
format, dtype, unique node IDs, contiguous global offsets, shard file sizes, and shard
checksums. When `source_root` is provided it must also check both source hashes before
returning the manifest.

- [ ] **Step 4: Run unit checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py -q
rtk proxy .venv/bin/ruff check src/data/packed_features.py tests/data/test_packed_features.py
rtk proxy .venv/bin/mypy src/data/packed_features.py tests/data/test_packed_features.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/packed_features.py tests/data/test_packed_features.py
rtk git commit -m "feat: define packed feature manifest"
```

---

### Task 4: Build BF16 shards in parallel and publish atomically

**Files:**
- Modify: `src/data/packed_features.py`
- Modify: `tests/data/test_packed_features.py`

**Interfaces:**
- Consumes: Task 3 manifest API.
- Produces: `build_packed_features(source_root: Path, pack_root: Path, workers: int) -> PackedFeatureManifest` for Tasks 6 and 10.

- [ ] **Step 1: Write failing build tests**

```python
def test_parallel_build_reads_once_and_writes_bf16_shards(tmp_path: Path) -> None:
    source_root = _write_feature_root(
        tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4), "node_c": (5, 4)}
    )
    pack_root = tmp_path / "pack"

    manifest = build_packed_features(source_root, pack_root, workers=2)

    assert manifest.pack_workers == 2
    assert [node.node_id for node in manifest.nodes] == ["node_a", "node_b", "node_c"]
    assert sum(shard.num_tokens for shard in manifest.shards) == 10
    assert validate_packed_manifest(pack_root, source_root) == manifest


def test_failed_build_never_publishes_final_directory(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    torch.save(torch.ones(3, 3), source_root / "embeddings/node_a.pt")
    pack_root = tmp_path / "pack"

    with pytest.raises(ValueError, match="input_dim"):
        build_packed_features(source_root, pack_root, workers=1)

    assert not pack_root.exists()
```

- [ ] **Step 2: Run and verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py -k "parallel_build or failed_build" -q
```

Expected: FAIL because `build_packed_features` is not defined.

- [ ] **Step 3: Implement one-read worker-owned shard construction**

Use a top-level picklable worker with this interface:

```python
@dataclass(frozen=True)
class _ShardJob:
    shard_index: int
    source_root: Path
    temp_root: Path
    input_dim: int
    entries: Sequence[tuple[str, str]]


def _write_shard(job: _ShardJob) -> tuple[PackedShardRecord, Sequence[PackedNodeRecord]]:
    shard_path = job.temp_root / f"shard-{job.shard_index:03d}.bin"
    records: list[PackedNodeRecord] = []
    token_offset = 0
    with shard_path.open("wb") as handle:
        for node_id, relative_path in job.entries:
            tensor = cast(
                torch.Tensor,
                torch.load(job.source_root / relative_path, map_location="cpu", weights_only=True),
            )
            if tensor.ndim != 2 or tensor.size(1) != job.input_dim:
                raise ValueError(f"feature {node_id} does not match input_dim={job.input_dim}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"feature {node_id} must be float32")
            bf16 = tensor.to(torch.bfloat16).contiguous().view(torch.uint16)
            handle.write(bf16.numpy().tobytes())
            records.append(
                PackedNodeRecord(node_id, job.shard_index, token_offset, 0, tensor.size(0))
            )
            token_offset += int(tensor.size(0))
    shard = PackedShardRecord(
        filename=shard_path.name,
        num_tokens=token_offset,
        byte_size=shard_path.stat().st_size,
        sha256=sha256_file(shard_path),
    )
    return shard, tuple(records)
```

`build_packed_features()` must partition deterministic index entries into `workers`
contiguous chunks, submit them with `ProcessPoolExecutor`, sort results by shard index,
assign global offsets in source node order, write the manifest, validate the temporary
pack, and atomically rename it. On failure, remove only the run-owned temporary path.

- [ ] **Step 4: Run pack tests and static checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py -q
rtk proxy .venv/bin/ruff check src/data/packed_features.py tests/data/test_packed_features.py
rtk proxy .venv/bin/mypy src/data/packed_features.py tests/data/test_packed_features.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/packed_features.py tests/data/test_packed_features.py
rtk git commit -m "feat: build BF16 feature shards in parallel"
```

---

### Task 5: Plan exact distributed pair batches

**Files:**
- Create: `src/data/distributed_pairs.py`
- Create: `tests/data/test_distributed_pairs.py`

**Interfaces:**
- Consumes: Existing `BUCKET_BOUNDARIES` from `src.data.pairs`.
- Produces: `PairBatchSpec`, `CompactPairBatch`, `build_distributed_epoch_plan()`, `CompactPairBatchDataset`, and `identity_compact_batch()` for Task 8.

- [ ] **Step 1: Write failing coverage, determinism, and tail tests**

```python
def test_plan_has_exact_coverage_and_equal_step_counts() -> None:
    lengths = [(100, 100)] * 19 + [(300, 200)] * 17
    plan = build_distributed_epoch_plan(
        lengths,
        token_budget_per_rank=1024,
        max_pairs_per_rank=8,
        world_size=4,
        seed=47,
        epoch=2,
        shuffle=True,
    )

    assert len({len(rank_plan) for rank_plan in plan}) == 1
    seen = [index for rank_plan in plan for spec in rank_plan for index in spec.indices]
    assert sorted(seen) == list(range(len(lengths)))
    assert len(seen) == len(set(seen))


def test_plan_is_reproducible() -> None:
    kwargs = dict(
        lengths=[(100, 100)] * 32,
        token_budget_per_rank=2048,
        max_pairs_per_rank=8,
        world_size=4,
        seed=7,
        epoch=3,
        shuffle=True,
    )
    assert build_distributed_epoch_plan(**kwargs) == build_distributed_epoch_plan(**kwargs)


def test_tail_batch_records_global_pair_count() -> None:
    plan = build_distributed_epoch_plan(
        [(100, 100)] * 21,
        token_budget_per_rank=1024,
        max_pairs_per_rank=8,
        world_size=4,
        seed=0,
        epoch=0,
        shuffle=False,
    )
    final_specs = [rank_plan[-1] for rank_plan in plan]
    assert sum(len(spec.indices) for spec in final_specs) == final_specs[0].global_pair_count
```

- [ ] **Step 2: Verify missing-module failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_distributed_pairs.py -q
```

Expected: collection ERROR for `src.data.distributed_pairs`.

- [ ] **Step 3: Implement the planner and compact dataset**

Define these exact types:

```python
@dataclass(frozen=True)
class PairBatchSpec:
    indices: Sequence[int]
    bucket_boundary: int
    global_pair_count: int


@dataclass(frozen=True)
class CompactPairBatch:
    row_ids: torch.Tensor
    node_a: torch.Tensor
    node_b: torch.Tensor
    labels: torch.Tensor
    bucket_boundary: int
    global_pair_count: int


def identity_compact_batch(batch: CompactPairBatch) -> CompactPairBatch:
    return batch
```

`build_distributed_epoch_plan()` must bucket indices, shuffle with
`np.random.default_rng((seed, epoch))`, cap a local batch by
`min(max_pairs_per_rank, token_budget_per_rank // (2 * boundary))`, build global
chunks of `local_cap * world_size`, merge a final chunk smaller than `world_size`
into the prior chunk, and split every global chunk with `np.array_split`. Reject a
non-empty bucket with fewer rows than `world_size`; the real-data integration test
must prove the E2 data does not hit that condition.

`CompactPairBatchDataset.__getitem__()` must gather the selected rows from four
prebuilt tensors and return `CompactPairBatch`; it must not load feature tensors.

- [ ] **Step 4: Run tests and static checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_distributed_pairs.py -q
rtk proxy .venv/bin/ruff check src/data/distributed_pairs.py tests/data/test_distributed_pairs.py
rtk proxy .venv/bin/mypy src/data/distributed_pairs.py tests/data/test_distributed_pairs.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/distributed_pairs.py tests/data/test_distributed_pairs.py
rtk git commit -m "feat: add distributed pair batch planning"
```

---

### Task 6: Load packed features onto a device and assemble padded batches

**Files:**
- Modify: `src/data/packed_features.py`
- Modify: `tests/data/test_packed_features.py`

**Interfaces:**
- Consumes: Task 4 packed shards and Task 5 `CompactPairBatch`.
- Produces: `PackedFeatureTable.from_pack()` and `PackedFeatureTable.assemble()` for Task 8.

- [ ] **Step 1: Write the failing gather/padding equivalence test**

```python
def test_packed_table_assembles_the_legacy_batch_contract(tmp_path: Path) -> None:
    source_root = _write_feature_root(
        tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4)}
    )
    pack_root = tmp_path / "pack"
    build_packed_features(source_root, pack_root, workers=1)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    compact = CompactPairBatch(
        row_ids=torch.tensor([9]),
        node_a=torch.tensor([0]),
        node_b=torch.tensor([1]),
        labels=torch.tensor([1.0]),
        bucket_boundary=4,
        global_pair_count=1,
    )

    batch = table.assemble(compact)

    assert batch["emb_a"].shape == (1, 4, 4)
    assert batch["emb_b"].shape == (1, 4, 4)
    assert batch["emb_a"].dtype == torch.bfloat16
    assert torch.equal(batch["len_a"], torch.tensor([3]))
    assert torch.equal(batch["len_b"], torch.tensor([2]))
    assert torch.count_nonzero(batch["emb_a"][0, 3:]) == 0
    assert int(batch["_row_id"][0]) == 9
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py -k packed_table -q
```

Expected: FAIL because `PackedFeatureTable` is not defined.

- [ ] **Step 3: Implement device loading and vectorized assembly**

Implement this interface:

```python
class PackedFeatureTable:
    def __init__(
        self,
        tokens: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        manifest: PackedFeatureManifest,
    ) -> None:
        self.tokens = tokens
        self.offsets = offsets
        self.lengths = lengths
        self.manifest = manifest

    @classmethod
    def from_pack(cls, pack_root: Path, device: torch.device) -> PackedFeatureTable:
        manifest = load_packed_manifest(pack_root)
        total_tokens = sum(shard.num_tokens for shard in manifest.shards)
        tokens = torch.empty((total_tokens, manifest.input_dim), dtype=torch.bfloat16, device=device)
        cursor = 0
        for shard in manifest.shards:
            mapped = torch.from_file(
                str(pack_root / shard.filename),
                shared=False,
                size=shard.num_tokens * manifest.input_dim,
                dtype=torch.bfloat16,
            ).reshape(shard.num_tokens, manifest.input_dim)
            tokens[cursor : cursor + shard.num_tokens].copy_(mapped, non_blocking=False)
            cursor += shard.num_tokens
        offsets = torch.tensor([node.global_offset for node in manifest.nodes], device=device)
        lengths = torch.tensor([node.length for node in manifest.nodes], device=device)
        return cls(tokens, offsets, lengths, manifest)
```

Add a private `_gather_endpoint(node_indices, boundary)` that constructs positions as
`offsets[:, None] + arange(boundary)`, masks positions beyond each length, gathers
from `tokens`, and zeroes padded rows. `assemble()` must return existing model keys plus
`_row_id`, `_local_pair_count`, and `_global_pair_count`, all on the table device.

- [ ] **Step 4: Run equivalence and regression tests**

```bash
rtk proxy .venv/bin/python -m pytest tests/data/test_packed_features.py tests/data/test_pairs.py -q
rtk proxy .venv/bin/ruff check src/data/packed_features.py tests/data/test_packed_features.py
rtk proxy .venv/bin/mypy src/data/packed_features.py tests/data/test_packed_features.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/packed_features.py tests/data/test_packed_features.py
rtk git commit -m "feat: assemble token batches from packed features"
```

---

### Task 7: Define probe selection, projections, profiles, and failure artifacts

**Files:**
- Create: `src/e2_pipeline.py`
- Create: `tests/test_e2_pipeline.py`

**Interfaces:**
- Consumes: Task 2 `RuntimeConfig`.
- Produces: `ProbeResult`, `PipelineProfile`, `select_probe_result()`, `project_total_seconds()`, and `write_failure()` for Task 10.

- [ ] **Step 1: Write failing pure-function tests**

```python
def test_selects_fastest_valid_probe_below_memory_limit() -> None:
    results = [
        ProbeResult(262144, True, 1200.0, 40.0, None),
        ProbeResult(524288, True, 1900.0, 70.0, None),
        ProbeResult(1048576, True, 1800.0, 82.0, None),
        ProbeResult(1572864, False, 0.0, 90.0, "memory limit"),
    ]
    assert select_probe_result(results, memory_limit_gib=85.0).token_budget == 524288


def test_projection_includes_all_thirty_epochs() -> None:
    projected = project_total_seconds(
        pack_seconds=240.0,
        setup_probe_seconds=240.0,
        epoch_seconds=90.0,
        epochs=30,
        artifact_seconds=60.0,
    )
    assert projected == pytest.approx(3240.0)


def test_failure_json_is_atomic_and_structured(tmp_path: Path) -> None:
    path = write_failure(tmp_path, stage="probe", message="projected budget miss")
    payload = json.loads(path.read_text())
    assert payload["stage"] == "probe"
    assert payload["message"] == "projected budget miss"
    assert not path.with_suffix(".json.tmp").exists()
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_pipeline.py -q
```

Expected: collection ERROR for `src.e2_pipeline`.

- [ ] **Step 3: Implement the pure pipeline contracts**

```python
@dataclass(frozen=True)
class ProbeResult:
    token_budget: int
    valid: bool
    global_pairs_per_second: float
    peak_memory_gib: float
    failure: str | None


@dataclass
class PipelineProfile:
    cold_cache: bool
    stage_seconds: dict[str, float]
    probe_results: list[ProbeResult]
    selected_token_budget: int | None
    projected_total_seconds: float | None


def select_probe_result(results: Sequence[ProbeResult], memory_limit_gib: float) -> ProbeResult:
    valid = [result for result in results if result.valid and result.peak_memory_gib <= memory_limit_gib]
    if not valid:
        raise RuntimeError("no valid E2 token-budget probe")
    return max(valid, key=lambda result: result.global_pairs_per_second)


def project_total_seconds(
    *, pack_seconds: float, setup_probe_seconds: float, epoch_seconds: float,
    epochs: int, artifact_seconds: float
) -> float:
    return pack_seconds + setup_probe_seconds + epoch_seconds * epochs + artifact_seconds
```

Implement failure writing with this exact signature:

```python
def write_failure(
    output_dir: Path,
    *,
    stage: str,
    message: str,
    extra: Mapping[str, object] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": stage, "message": message}
    if extra is not None:
        payload.update(extra)
    temp_path = output_dir / "failure.json.tmp"
    final_path = output_dir / "failure.json"
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, final_path)
    return final_path
```

Add JSON conversion methods for both dataclasses without external serialization
dependencies.

- [ ] **Step 4: Run tests and static checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_pipeline.py -q
rtk proxy .venv/bin/ruff check src/e2_pipeline.py tests/test_e2_pipeline.py
rtk proxy .venv/bin/mypy src/e2_pipeline.py tests/test_e2_pipeline.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/e2_pipeline.py tests/test_e2_pipeline.py
rtk git commit -m "feat: add E2 probe and profile contracts"
```

---

### Task 8: Build packed multi-worker loaders without feature I/O

**Files:**
- Modify: `src/train_b0.py:56-58,889-1012`
- Modify: `tests/test_train_b0.py:740-905`

**Interfaces:**
- Consumes: Task 2 runtime config, Task 5 compact plans, Task 6 `PackedFeatureTable`.
- Produces: `PackedLoaderFactory`, `GpuBatchIterable`, `compute_sample_warmup_steps()`, and `_build_packed_v3_1_loaders()` for Task 9.

- [ ] **Step 1: Write failing loader and warm-up tests**

```python
def test_sample_warmup_preserves_pair_exposure() -> None:
    assert compute_sample_warmup_steps([10, 10], [25], baseline_steps=5) == 2


def test_packed_loader_does_not_call_feature_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, assembled, pack_root = _synthetic_v31_pack_fixture(tmp_path)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))
    monkeypatch.setattr(
        assembled.store,
        "load_tokens",
        Mock(side_effect=AssertionError("unexpected feature I/O")),
    )

    factory, val_loader, warmup_steps = _build_packed_v3_1_loaders(
        cfg,
        assembled,
        table,
        token_budget_per_rank=1024,
        process_index=0,
        world_size=1,
    )

    batch = next(iter(factory(1)))
    assert set(batch) >= {"emb_a", "emb_b", "len_a", "len_b", "label"}
    assert next(iter(val_loader))["_row_id"].numel() > 0
    assert warmup_steps >= 1
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py -k "sample_warmup or packed_loader" -q
```

Expected: FAIL because the packed loader API is absent.

- [ ] **Step 3: Implement the adapter and loader factory**

Add these exact interfaces:

```python
PackedLoaderFactory = Callable[[int], Iterable[Batch]]


class GpuBatchIterable:
    def __init__(
        self, source: Iterable[CompactPairBatch], table: PackedFeatureTable
    ) -> None:
        self._source = source
        self._table = table

    def __iter__(self) -> Iterator[Batch]:
        for compact in self._source:
            yield self._table.assemble(compact)


def compute_sample_warmup_steps(
    baseline_batch_sizes: Sequence[int],
    new_global_batch_sizes: Sequence[int],
    *,
    baseline_steps: int,
) -> int:
    if not baseline_batch_sizes or not new_global_batch_sizes:
        raise ValueError("warmup batch-size sequences must be non-empty")
    target = sum(islice(cycle(baseline_batch_sizes), baseline_steps))
    seen = 0
    steps = 0
    for batch_size in cycle(new_global_batch_sizes):
        seen += batch_size
        steps += 1
        if seen >= target:
            return steps
    raise AssertionError("cycle over non-empty batch sizes must terminate")
```

Add `from unittest.mock import Mock` to the test imports and define the synthetic
fixture using existing helpers in `tests/test_train_b0.py`:

```python
def _synthetic_v31_pack_fixture(
    tmp_path: Path,
) -> tuple[Config, AssembledData, Path]:
    data_root = tmp_path / "data"
    benchmark_root = data_root / "benchmark_2025_neurips"
    benchmark_root.mkdir(parents=True)
    _build_synthetic_benchmark(benchmark_root, "synthetic")
    source_root = data_root / "features" / "frozen_node_features_1024"
    _write_feature_store(source_root, [f"node_{index:06d}" for index in range(1, 7)])
    cfg = load_config(Path("configs/b0_v31_breadth_first.yaml"))
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            root=data_root,
            strategy="synthetic",
            expected_missing_features=["node_000007"],
        ),
        output_dir=tmp_path / "outputs",
    )
    assembled = assemble_data(cfg, verify=False)
    pack_root = tmp_path / "pack"
    build_packed_features(source_root, pack_root, workers=1)
    return cfg, assembled, pack_root
```

`_build_packed_v3_1_loaders()` must derive endpoint integer IDs and lengths from the
manifest, build one `CompactPairBatchDataset` per rank/epoch, and wrap a DataLoader
configured as follows:

```python
DataLoader(
    dataset,
    batch_size=None,
    num_workers=cfg.runtime.loader_workers_per_rank,
    persistent_workers=True,
    prefetch_factor=cfg.runtime.prefetch_factor,
    collate_fn=identity_compact_batch,
)
```

Use a non-persistent single-process DataLoader only when `world_size == 1` in unit
tests. Compute baseline pair exposure from the legacy `LengthBucketedBatchSampler`
without loading feature tensors.

- [ ] **Step 4: Run loader and legacy regression tests**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py -k "packed_loader or sample_warmup or V31TrainingLoader" -q
rtk proxy .venv/bin/ruff check src/train_b0.py tests/test_train_b0.py
rtk proxy .venv/bin/mypy src/train_b0.py tests/test_train_b0.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/train_b0.py tests/test_train_b0.py
rtk git commit -m "feat: load E2 batches from GPU feature cache"
```

---

### Task 9: Add fixed-epoch DDP train/eval semantics and rank-zero artifacts

**Files:**
- Modify: `src/train_b0.py:166-180,600-886,1048-1165`
- Modify: `tests/test_train_b0.py:180-620`

**Interfaces:**
- Consumes: Task 8 packed loaders.
- Produces: `build_ddp_accelerator()`, `scale_ddp_mean_loss()`, `validate_gathered_validation()`, `train_ddp_loop()`, and internal worker CLI modes `probe`, `epoch-probe`, and `train` for Task 10.

- [ ] **Step 1: Write failing loss, coverage, and fixed-epoch tests**

```python
def test_ddp_loss_scaling_matches_global_sample_mean() -> None:
    local_mean = torch.tensor(2.0)
    scaled = scale_ddp_mean_loss(local_mean, local_count=3, global_count=10, world_size=4)
    assert scaled.item() == pytest.approx(2.4)


def test_validation_rejects_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="duplicate validation row IDs"):
        validate_gathered_validation(
            row_ids=np.array([0, 0, 1]),
            labels=np.array([0, 0, 1]),
            logits=np.array([-1.0, -1.0, 1.0]),
            expected_row_ids=np.array([0, 1]),
        )


def test_ddp_loop_records_counterfactual_stop_but_runs_all_epochs(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    _write_yaml_config(config_path, {"optim.epochs": 4, "eval.patience": 1})
    cfg = load_config(config_path)
    model = F0PairMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
    batch = _batch_of(_make_synthetic_pair_dataset(8))
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    batch["_row_id"] = torch.arange(8)

    def factory(epoch: int) -> list[dict[str, torch.Tensor]]:
        assert 1 <= epoch <= 4
        return [batch]

    metrics = EdgeMetrics(
        auroc=0.5, auprc=0.5, accuracy=0.5, sensitivity=0.5,
        specificity=0.5, precision=0.5, recall=0.5, f1=0.5, mcc=0.0,
        ece=0.0, brier=0.25, threshold=0.5, n_pos=4, n_neg=4,
    )
    result = train_ddp_loop(
        model,
        factory,
        [batch],
        cfg,
        Accelerator(),
        warmup_steps=1,
        evaluate_fn=lambda model, loader, accelerator: metrics,
    )
    assert result.last_epoch == 4
    assert result.stopped_early is False
    assert result.counterfactual_stop_epoch == 2
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py -k "ddp_loss or duplicate_rows or counterfactual" -q
```

Expected: FAIL because the DDP helpers and result field are absent.

- [ ] **Step 3: Implement DDP helpers and distributed validation**

Use the installed Accelerate API exactly as follows:

```python
def build_ddp_accelerator(mixed_precision: str) -> Accelerator:
    kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[kwargs])


def scale_ddp_mean_loss(
    loss: torch.Tensor, *, local_count: int, global_count: int, world_size: int
) -> torch.Tensor:
    if local_count < 1 or global_count < local_count or world_size < 1:
        raise ValueError("invalid DDP loss-scaling counts")
    return loss * (float(local_count * world_size) / float(global_count))
```

For validation, pad row IDs with `-1` using
`accelerator.pad_across_processes()`, gather row IDs/labels/logits with
`accelerator.gather()`, mask padded rows on rank zero, sort by row ID, and call
`validate_gathered_validation()`. Broadcast `asdict(metrics)` with
`accelerate.utils.broadcast_object_list` so all ranks make the same best-epoch
decision.

Add `counterfactual_stop_epoch: int | None = None` to `TrainResult`. In
`train_ddp_loop()`, keep training after patience is reached, set that field once, and
always stop at `cfg.optim.epochs`. Only the main rank snapshots checkpoint state or
executes the incremental artifact callback.

Add `runtime_profile: dict[str, object] = field(default_factory=dict)` after the
counterfactual field. Measure each batch's loader wait with `time.monotonic()`, measure
GPU compute with paired `torch.cuda.Event` objects, reset/read peak allocated memory
per epoch, and all-reduce rank pair/batch/step counts. The rank-zero profile must
contain these exact keys used by Task 12:

```python
runtime_profile = {
    "epochs_completed": cfg.optim.epochs,
    "validations_completed": cfg.optim.epochs,
    "peak_memory_gib_per_rank": peak_memory_gib_per_rank,
    "steady_state_data_wait_fraction": data_wait_seconds / train_wall_seconds,
    "training_coverage_exact": training_coverage_exact,
    "validation_coverage_exact": validation_coverage_exact,
    "feature_cache_hit_rate": 1.0,
    "per_epoch": per_epoch_profiles,
}
```

Before backward, all-reduce a finite-loss flag with the minimum reduction and fail all
ranks when it is zero. At every epoch boundary, gather local step counts and fail all
ranks when their minimum and maximum differ.

Use this callable injection point so unit tests can supply deterministic metrics while
production uses distributed validation:

```python
EvaluateFn = Callable[[nn.Module, Iterable[Batch], Accelerator], EdgeMetrics]


def train_ddp_loop(
    model: nn.Module,
    train_loader_factory: PackedLoaderFactory,
    val_loader: Iterable[Batch],
    cfg: Config,
    accelerator: Accelerator,
    *,
    warmup_steps: int,
    evaluate_fn: EvaluateFn = _evaluate_distributed,
    on_eval: OnEval | None = None,
) -> TrainResult:
    """Run fixed-epoch E2 DDP training and return rank-consistent metrics."""
```

- [ ] **Step 4: Add internal worker CLI arguments**

Extend `CliArgs` and `parse_args()` with:

```python
parser.add_argument("--ddp-mode", choices=("probe", "epoch-probe", "train"), default=None)
parser.add_argument("--pack-dir", type=Path, default=None)
parser.add_argument("--token-budget-per-rank", type=int, default=None)
parser.add_argument("--profile-output", type=Path, default=None)
```

Require all four flags when `ddp_mode` is set. `probe` runs configured warm-up/timed
steps and writes one rank-zero `ProbeResult` JSON; `epoch-probe` runs one complete
train epoch plus validation and writes elapsed seconds; `train` runs all 30 epochs and
writes formal artifacts. Keep the existing no-`ddp_mode` path for B0-alt and local
`--max-steps` debugging.

- [ ] **Step 5: Run train, checkpoint-contract, lint, and type checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_train_b0.py tests/test_score_universe.py -q
rtk proxy .venv/bin/ruff check src/train_b0.py tests/test_train_b0.py
rtk proxy .venv/bin/mypy src/train_b0.py tests/test_train_b0.py
```

Expected: PASS; the score-universe checkpoint tests prove the payload contract is unchanged.

- [ ] **Step 6: Commit**

```bash
rtk git add src/train_b0.py tests/test_train_b0.py
rtk git commit -m "feat: train E2 with fixed-epoch DDP semantics"
```

---

### Task 10: Orchestrate pack, probes, projection, formal train, and deadlines

**Files:**
- Modify: `src/e2_pipeline.py`
- Modify: `tests/test_e2_pipeline.py`

**Interfaces:**
- Consumes: Task 4 `build_packed_features`, Task 7 profile contracts, Task 9 internal worker CLI.
- Produces: `PipelineArgs`, `BudgetExceeded`, `enforce_projection()`, `build_accelerate_command()`, `run_pipeline()`, and `python -m src.e2_pipeline` for Task 11.

- [ ] **Step 1: Write failing command and fail-fast tests**

```python
def test_accelerate_command_pins_four_processes(tmp_path: Path) -> None:
    command = build_accelerate_command(
        accelerate_bin=Path("/venv/bin/accelerate"),
        config_path=Path("configs/b0_v31_breadth_first.yaml"),
        mode="probe",
        pack_dir=tmp_path / "pack",
        output_dir=tmp_path / "outputs",
        token_budget=524288,
        profile_output=tmp_path / "probe.json",
    )
    assert command[:4] == [
        "/venv/bin/accelerate", "launch", "--num_processes", "4"
    ]
    assert command[-2:] == ["--profile-output", str(tmp_path / "probe.json")]


def test_projection_budget_failure_writes_artifact(tmp_path: Path) -> None:
    with pytest.raises(BudgetExceeded, match="3600"):
        enforce_projection(projected_seconds=6200.0, limit_seconds=3600, output_dir=tmp_path)
    payload = json.loads((tmp_path / "failure.json").read_text())
    assert payload["stage"] == "projection"
    assert payload["projected_seconds"] == 6200.0
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_pipeline.py -k "accelerate_command or projection_budget" -q
```

Expected: FAIL because orchestration functions are absent.

- [ ] **Step 3: Implement the production orchestration**

Define:

```python
@dataclass(frozen=True)
class PipelineArgs:
    config: Path
    pack_dir: Path | None
    output_dir: Path | None


class BudgetExceeded(RuntimeError):
    """Raised before formal training when projected wall time exceeds the budget."""


def enforce_projection(
    *, projected_seconds: float, limit_seconds: int, output_dir: Path
) -> None:
    if projected_seconds <= limit_seconds:
        return
    write_failure(
        output_dir,
        stage="projection",
        message=f"projected {projected_seconds:.1f}s exceeds {limit_seconds}s",
        extra={"projected_seconds": projected_seconds, "limit_seconds": limit_seconds},
    )
    raise BudgetExceeded(f"projected runtime exceeds {limit_seconds} seconds")


def build_accelerate_command(
    *, accelerate_bin: Path, config_path: Path, mode: str, pack_dir: Path,
    output_dir: Path, token_budget: int, profile_output: Path
) -> list[str]:
    return [
        str(accelerate_bin),
        "launch",
        "--num_processes",
        "4",
        "--mixed_precision",
        "bf16",
        "-m",
        "src.train_b0",
        "--config",
        str(config_path),
        "--ddp-mode",
        mode,
        "--pack-dir",
        str(pack_dir),
        "--output-dir",
        str(output_dir),
        "--token-budget-per-rank",
        str(token_budget),
        "--profile-output",
        str(profile_output),
    ]
```

Parse the public pipeline CLI with this concrete shape:

```python
def parse_pipeline_args(argv: Sequence[str] | None = None) -> PipelineArgs:
    parser = argparse.ArgumentParser(prog="python -m src.e2_pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    namespace = parser.parse_args(argv)
    return PipelineArgs(namespace.config, namespace.pack_dir, namespace.output_dir)
```

Use this command-runner seam and public orchestration signature:

```python
CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def run_command(
    command: Sequence[str], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def run_pipeline(
    args: PipelineArgs, *, command_runner: CommandRunner = run_command
) -> int:
    """Execute the cold E2 pipeline and return 0 on success or 2 on a gated failure."""
```

`run_pipeline()` must:

1. load and validate the V3.1 config/runtime;
2. record whether the pack path was initially absent;
3. build or strictly validate the pack within 300 seconds;
4. launch one fresh `probe` process group per candidate;
5. select the fastest valid result at or below 85 GiB;
6. launch one `epoch-probe` with the selected budget;
7. project 30 epochs plus completed setup/artifact allowance;
8. write `failure.json` and return 2 when projected total exceeds 3,600 seconds;
9. launch one clean `train` process group with the frozen budget;
10. merge stage/profile data into `outputs/b0_v31/profile.json`; and
11. write `artifact_manifest.json` with SHA-256 and byte size for `best.pt`, `last.pt`,
    `metrics.jsonl`, `run_metadata.json`, and `profile.json`.

The merged JSON must copy Task 9's runtime fields and add pipeline fields with this
shape:

```python
final_profile = {
    **worker_runtime_profile,
    "cold_cache": cold_cache,
    "total_seconds": time.monotonic() - pipeline_started,
    "stage_seconds": stage_seconds,
    "probe_results": [asdict(result) for result in probe_results],
    "selected_token_budget": selected.token_budget,
    "projected_total_seconds": projected_total_seconds,
}
```

Every subprocess must use `check=False`, capture its return code, and convert a failed
rank group into one rank-zero failure artifact. Do not retry or change the selected
configuration after formal training begins.

Pass an explicit timeout to every subprocess: the remaining setup/probe budget for
probe modes and `runtime.train_eval_budget_seconds` for formal training. Convert
`subprocess.TimeoutExpired` into `failure.json` with the stage and configured deadline.
After artifact writing, reject a total elapsed time above `runtime.total_budget_seconds`.

- [ ] **Step 4: Run orchestration tests and static checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_pipeline.py -q
rtk proxy .venv/bin/ruff check src/e2_pipeline.py tests/test_e2_pipeline.py
rtk proxy .venv/bin/mypy src/e2_pipeline.py tests/test_e2_pipeline.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/e2_pipeline.py tests/test_e2_pipeline.py
rtk git commit -m "feat: orchestrate the E2 throughput pipeline"
```

---

### Task 11: Make the fixed HPC runner use the four-H20 E2 entry

**Files:**
- Modify: `hpc/run.sh:1-79`
- Modify: `hpc/README.md:1-79`
- Modify: `README.md:95,187-205`
- Modify: `CLAUDE.md:8-18,102-126,150-170`
- Modify: `tests/test_hpc_scripts.py:24-99`

**Interfaces:**
- Consumes: Task 10 `python -m src.e2_pipeline`.
- Produces: `hpc/run.sh train configs/b0_v31_breadth_first.yaml` as the only formal E2 entry.

- [ ] **Step 1: Replace the single-H20 runner test with a failing four-H20 test**

```python
def test_runner_pins_the_verified_four_h20_environment() -> None:
    text = RUNNER.read_text()
    for value in (
        "/2023533015/topology-conditioned-inductive-edge-prediction",
        "/2023533015/.uv/bin/uv",
        "NVIDIA H20",
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "expected exactly 4 visible GPUs",
    ):
        assert value in text
    assert "-m src.e2_pipeline" in text
    assert "--num_processes 4" in (HPC_DIR / "README.md").read_text()
```

- [ ] **Step 2: Verify failure**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_hpc_scripts.py -q
```

Expected: FAIL on the current one-GPU assertions.

- [ ] **Step 3: Update the runner and docs**

Set:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Change runtime validation to require exactly four entries from `nvidia-smi` and verify
each is `NVIDIA H20`. Change the E2 train branch to:

```bash
train)
  [[ $# -ge 1 ]] || fail "train requires a config path"
  CONFIG_PATH="$1"
  shift
  [[ -f "${CONFIG_PATH}" ]] || fail "config not found: ${CONFIG_PATH}"
  exec "${PYTHON_BIN}" -m src.e2_pipeline --config "${CONFIG_PATH}" "$@"
  ;;
```

Keep `score`, `merge`, `g1`, and `g2` behavior unchanged. Document that direct
`python -m src.train_b0 --max-steps N` remains debug-only and that B0-alt continues
to use its existing direct training CLI outside this E2-only optimization.

- [ ] **Step 4: Run runner, docs, and shell checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_hpc_scripts.py -q
rtk proxy bash -n hpc/run.sh
rtk proxy .venv/bin/ruff check tests/test_hpc_scripts.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add hpc/run.sh hpc/README.md README.md CLAUDE.md tests/test_hpc_scripts.py
rtk git commit -m "feat: route formal E2 training through four H20s"
```

---

### Task 12: Add multi-process integration and opt-in four-H20 acceptance tests

**Files:**
- Create: `tests/test_e2_ddp_integration.py`
- Create: `tests/helpers/e2_ddp_smoke.py`

**Interfaces:**
- Consumes: Tasks 5-10 Python pipeline APIs.
- Produces: A CPU/Gloo smoke test for normal integration runs and an explicit H20 acceptance gate.

- [ ] **Step 1: Write the CPU multi-process smoke test**

Create `tests/test_e2_ddp_integration.py` with these imports/constants before the
tests:

```python
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
```

Add this test:

```python
@pytest.mark.integration
def test_two_rank_cpu_plan_has_exact_global_coverage(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/helpers/e2_ddp_smoke.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = []
    for rank in range(2):
        rows.extend(json.loads((tmp_path / f"rank-{rank}.json").read_text()))
    assert sorted(rows) == list(range(64))
    assert len(rows) == len(set(rows))
```

- [ ] **Step 2: Run the CPU multi-process test and verify it fails**

Run before creating the helper:

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_ddp_integration.py -k two_rank -q
```

Expected: FAIL because `tests/helpers/e2_ddp_smoke.py` is missing.

- [ ] **Step 3: Implement the two-rank helper**

Create `tests/helpers/e2_ddp_smoke.py` with this complete helper:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.data.distributed_pairs import build_distributed_epoch_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    plan = build_distributed_epoch_plan(
        [(100, 100)] * 64,
        token_budget_per_rank=2048,
        max_pairs_per_rank=8,
        world_size=world_size,
        seed=47,
        epoch=1,
        shuffle=True,
    )
    row_ids = [index for spec in plan[rank] for index in spec.indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = args.output_dir / f"rank-{rank}.json.tmp"
    final_path = args.output_dir / f"rank-{rank}.json"
    temp_path.write_text(json.dumps(row_ids))
    os.replace(temp_path, final_path)
    count = torch.tensor([len(row_ids)], dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if int(count.item()) != 64:
        raise RuntimeError(f"expected 64 globally covered rows, got {count.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the CPU multi-process test and verify it passes**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_ddp_integration.py -k two_rank -q
```

Expected: PASS.

- [ ] **Step 5: Add the opt-in four-H20 acceptance test**

```python
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_E2_H20_ACCEPTANCE") != "1",
    reason="set RUN_E2_H20_ACCEPTANCE=1 only on the fixed four-H20 container",
)
def test_cold_four_h20_run_meets_budget(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "hpc/run.sh",
            "train",
            "configs/b0_v31_breadth_first.yaml",
            "--pack-dir",
            str(tmp_path / "cold-pack"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3700,
    )
    assert result.returncode == 0, result.stderr
    profile = json.loads((tmp_path / "outputs/profile.json").read_text())
    assert profile["cold_cache"] is True
    assert profile["total_seconds"] <= 3600
    assert profile["epochs_completed"] == 30
    assert profile["validations_completed"] == 30
    assert max(profile["peak_memory_gib_per_rank"]) <= 85.0
    assert profile["steady_state_data_wait_fraction"] <= 0.05
    assert profile["training_coverage_exact"] is True
    assert profile["validation_coverage_exact"] is True
    for filename in (
        "best.pt",
        "last.pt",
        "metrics.jsonl",
        "run_metadata.json",
        "profile.json",
        "artifact_manifest.json",
    ):
        assert (tmp_path / "outputs" / filename).exists()
```

- [ ] **Step 6: Run local integration/static checks**

```bash
rtk proxy .venv/bin/python -m pytest tests/test_e2_ddp_integration.py -k two_rank -q
rtk proxy .venv/bin/ruff check tests/test_e2_ddp_integration.py tests/helpers/e2_ddp_smoke.py
rtk proxy .venv/bin/mypy tests/test_e2_ddp_integration.py tests/helpers/e2_ddp_smoke.py
```

Expected: CPU smoke PASS; H20 acceptance SKIPPED unless explicitly enabled.

- [ ] **Step 7: Commit**

```bash
rtk git add tests/test_e2_ddp_integration.py tests/helpers/e2_ddp_smoke.py
rtk git commit -m "test: add E2 distributed acceptance gates"
```

---

## Final merge verification

After Tasks 11 and 12 are merged, run this sequence from a clean implementation worktree:

```bash
rtk proxy .venv/bin/python -m pytest -m "not integration and not slow" -q
rtk proxy .venv/bin/python -m pytest tests/test_e2_ddp_integration.py -k two_rank -q
rtk proxy .venv/bin/ruff check src tests
rtk proxy .venv/bin/ruff format --check src tests
rtk proxy .venv/bin/mypy src tests
rtk git diff --check
rtk git status --short --branch
```

Expected:

- all unit and CPU multi-process tests PASS;
- the opt-in H20 acceptance test remains skipped locally;
- ruff, formatting, mypy, and diff checks PASS;
- only intentional implementation changes appear in git status.

On the fixed four-H20 container, use a new cache/output directory and run:

```bash
rtk proxy env RUN_E2_H20_ACCEPTANCE=1 .venv/bin/python -m pytest \
  tests/test_e2_ddp_integration.py::test_cold_four_h20_run_meets_budget -q -s
```

Expected: PASS within the test's 3,700-second outer timeout, with
`profile.total_seconds <= 3600` and all coverage/memory/data-wait assertions satisfied.
