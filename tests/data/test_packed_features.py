from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
import src.data.packed_features as packed_features
import torch
from src.data.distributed_pairs import CompactPairBatch
from src.data.packed_features import (
    PackedFeatureManifest,
    PackedFeatureTable,
    PackedNodeRecord,
    PackedShardRecord,
    _ShardJob,
    build_packed_features,
    load_packed_manifest,
    sha256_file,
    validate_packed_manifest,
    write_packed_manifest,
)

pytestmark = pytest.mark.unit

_ShardResult = tuple[PackedShardRecord, Sequence[PackedNodeRecord]]


class _SynchronousExecutor:
    def __init__(self, max_workers: int, after_map: Callable[[], None] | None = None) -> None:
        self.max_workers = max_workers
        self.after_map = after_map

    def __enter__(self) -> _SynchronousExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def map(
        self,
        function: Callable[[_ShardJob], _ShardResult],
        jobs: Iterable[_ShardJob],
    ) -> list[_ShardResult]:
        results = [function(job) for job in jobs]
        if self.after_map is not None:
            self.after_map()
        return results


@pytest.fixture(autouse=True)
def _avoid_process_pool_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise packing deterministically without spawning Python worker processes."""
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", _SynchronousExecutor)


def _write_feature_root(root: Path, node_shapes: dict[str, tuple[int, int]]) -> Path:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    index: dict[str, str] = {}
    for node_id, (length, dim) in node_shapes.items():
        relative_path = f"embeddings/{node_id}.pt"
        torch.save(torch.zeros(length, dim, dtype=torch.float32), root / relative_path)
        index[node_id] = relative_path
    (root / "metadata.json").write_text(json.dumps({"format": "torch_pt_per_node", "input_dim": 4}))
    (root / "index.json").write_text(json.dumps(index))
    return root


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
        shards=(PackedShardRecord("shard-000.bin", 3, 24, sha256_file(shard_path)),),
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


def _read_manifest_json(pack_root: Path) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((pack_root / "manifest.json").read_text(encoding="utf-8")),
    )


def _write_manifest_json(pack_root: Path, raw: object) -> None:
    (pack_root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")


@pytest.mark.parametrize(
    ("location", "change"),
    [
        ("top", ("unexpected", 1)),
        ("node", ("unexpected", 1)),
        ("shard", ("unexpected", 1)),
    ],
)
def test_manifest_rejects_unknown_schema_keys(
    tmp_path: Path, location: str, change: tuple[str, object]
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    raw = _read_manifest_json(pack_root)
    target = raw
    if location == "node":
        target = cast(dict[str, object], cast(list[object], raw["nodes"])[0])
    elif location == "shard":
        target = cast(dict[str, object], cast(list[object], raw["shards"])[0])
    target[change[0]] = change[1]
    _write_manifest_json(pack_root, raw)

    with pytest.raises(ValueError, match="malformed"):
        load_packed_manifest(pack_root)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("top", "format"),
        ("node", "node_id"),
        ("shard", "filename"),
    ],
)
def test_manifest_rejects_missing_schema_keys(tmp_path: Path, location: str, field: str) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    raw = _read_manifest_json(pack_root)
    target = raw
    if location == "node":
        target = cast(dict[str, object], cast(list[object], raw["nodes"])[0])
    elif location == "shard":
        target = cast(dict[str, object], cast(list[object], raw["shards"])[0])
    del target[field]
    _write_manifest_json(pack_root, raw)

    with pytest.raises(ValueError, match="malformed"):
        load_packed_manifest(pack_root)


@pytest.mark.parametrize(
    ("location", "field", "bad_value"),
    [
        ("top", "format", 1),
        ("top", "input_dim", True),
        ("top", "dtype", 1),
        ("top", "source_metadata_sha256", 1),
        ("top", "source_index_sha256", 1),
        ("top", "nodes", {}),
        ("top", "shards", {}),
        ("top", "pack_workers", True),
        ("top", "build_seconds", True),
        ("node", "node_id", 1),
        ("node", "shard_index", True),
        ("node", "shard_offset", True),
        ("node", "global_offset", True),
        ("node", "length", True),
        ("shard", "filename", 1),
        ("shard", "num_tokens", True),
        ("shard", "byte_size", True),
        ("shard", "sha256", 1),
    ],
)
def test_manifest_rejects_wrong_schema_types_cleanly(
    tmp_path: Path, location: str, field: str, bad_value: object
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    raw = _read_manifest_json(pack_root)
    target = raw
    if location == "node":
        target = cast(dict[str, object], cast(list[object], raw["nodes"])[0])
    elif location == "shard":
        target = cast(dict[str, object], cast(list[object], raw["shards"])[0])
    target[field] = bad_value
    _write_manifest_json(pack_root, raw)

    with pytest.raises(ValueError, match="malformed"):
        load_packed_manifest(pack_root)


def test_manifest_rejects_changed_source_hash(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    (source_root / "metadata.json").write_text('{"format":"changed"}')

    with pytest.raises(ValueError, match="source metadata hash"):
        validate_packed_manifest(pack_root, source_root)


def test_manifest_rejects_changed_source_index_hash(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    (source_root / "index.json").write_text("{}")

    with pytest.raises(ValueError, match="source index hash"):
        validate_packed_manifest(pack_root, source_root)


def test_manifest_rejects_unsupported_format(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    manifest = load_packed_manifest(pack_root)
    write_packed_manifest(pack_root, replace(manifest, format="other"))

    with pytest.raises(ValueError, match="format"):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_unsupported_dtype(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    manifest = load_packed_manifest(pack_root)
    write_packed_manifest(pack_root, replace(manifest, dtype="float32"))

    with pytest.raises(ValueError, match="dtype"):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_duplicate_node_ids(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    manifest = load_packed_manifest(pack_root)
    duplicate = PackedNodeRecord("node_a", 0, 24, 3, 1)
    write_packed_manifest(pack_root, replace(manifest, nodes=(*manifest.nodes, duplicate)))

    with pytest.raises(ValueError, match="duplicate node ID"):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_noncontiguous_global_offsets(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    manifest = load_packed_manifest(pack_root)
    bad_node = replace(manifest.nodes[0], global_offset=1)
    write_packed_manifest(pack_root, replace(manifest, nodes=(bad_node,)))

    with pytest.raises(ValueError, match="global offset"):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_node_order_different_from_source_index(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (1, 4), "node_b": (2, 4)})
    pack_root = tmp_path / "pack"
    monkey_manifest = PackedFeatureManifest(
        format="bf16_flat_shards_v1",
        input_dim=4,
        dtype="bfloat16",
        source_metadata_sha256=sha256_file(source_root / "metadata.json"),
        source_index_sha256=sha256_file(source_root / "index.json"),
        nodes=(
            PackedNodeRecord("node_b", 0, 0, 0, 2),
            PackedNodeRecord("node_a", 0, 2, 2, 1),
        ),
        shards=(),
        pack_workers=1,
        build_seconds=0.0,
    )
    pack_root.mkdir()
    shard_path = pack_root / "shard-000.bin"
    shard_path.write_bytes(bytes(24))
    write_packed_manifest(
        pack_root,
        replace(
            monkey_manifest,
            shards=(PackedShardRecord(shard_path.name, 3, 24, sha256_file(shard_path)),),
        ),
    )

    with pytest.raises(ValueError, match="node order"):
        validate_packed_manifest(pack_root, source_root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: replace(manifest, input_dim=0), "input_dim"),
        (
            lambda manifest: replace(manifest, nodes=(replace(manifest.nodes[0], length=0),)),
            "length",
        ),
        (
            lambda manifest: replace(manifest, nodes=(replace(manifest.nodes[0], shard_index=1),)),
            "shard index",
        ),
        (
            lambda manifest: replace(manifest, nodes=(replace(manifest.nodes[0], shard_offset=1),)),
            "shard offset",
        ),
        (
            lambda manifest: replace(manifest, nodes=(replace(manifest.nodes[0], length=4),)),
            "token total",
        ),
        (
            lambda manifest: replace(manifest, shards=(replace(manifest.shards[0], byte_size=32),)),
            "byte size formula",
        ),
        (
            lambda manifest: replace(
                manifest, shards=(replace(manifest.shards[0], filename="../shard.bin"),)
            ),
            "filename",
        ),
    ],
)
def test_manifest_rejects_structural_corruption(
    tmp_path: Path,
    mutate: Callable[[PackedFeatureManifest], PackedFeatureManifest],
    message: str,
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    write_packed_manifest(pack_root, mutate(load_packed_manifest(pack_root)))

    with pytest.raises(ValueError, match=message):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_duplicate_shard_filenames(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    manifest = load_packed_manifest(pack_root)
    write_packed_manifest(pack_root, replace(manifest, shards=tuple(manifest.shards) * 2))

    with pytest.raises(ValueError, match="duplicate shard filename"):
        validate_packed_manifest(pack_root, None)


def test_manifest_rejects_malformed_json_types_cleanly(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    (pack_root / "manifest.json").write_text('{"nodes": "not-a-list", "shards": []}')

    with pytest.raises(ValueError, match="malformed"):
        load_packed_manifest(pack_root)


@pytest.mark.parametrize("corruption", ["size", "checksum"])
def test_manifest_rejects_corrupt_shard(tmp_path: Path, corruption: str) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)
    shard_path = pack_root / "shard-000.bin"
    if corruption == "size":
        shard_path.write_bytes(bytes(23))
        message = "size"
    else:
        shard_path.write_bytes(bytes([1]) + bytes(23))
        message = "checksum"

    with pytest.raises(ValueError, match=message):
        validate_packed_manifest(pack_root, None)


def test_parallel_build_reads_once_and_writes_bf16_shards(tmp_path: Path) -> None:
    source_root = _write_feature_root(
        tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4), "node_c": (5, 4)}
    )
    expected_a = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 0.125
    expected_b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100.25
    expected_c = torch.arange(20, dtype=torch.float32).reshape(5, 4) - 50.5
    torch.save(expected_a, source_root / "embeddings/node_a.pt")
    torch.save(expected_b, source_root / "embeddings/node_b.pt")
    torch.save(expected_c, source_root / "embeddings/node_c.pt")
    pack_root = tmp_path / "pack"

    manifest = build_packed_features(source_root, pack_root, workers=2)

    assert manifest.pack_workers == 2
    assert manifest.nodes == (
        PackedNodeRecord("node_a", 0, 0, 0, 3),
        PackedNodeRecord("node_b", 0, 3, 3, 2),
        PackedNodeRecord("node_c", 1, 0, 5, 5),
    )
    assert [shard.filename for shard in manifest.shards] == [
        "shard-000.bin",
        "shard-001.bin",
    ]
    assert sum(shard.num_tokens for shard in manifest.shards) == 10
    shard_0 = torch.frombuffer(
        bytearray((pack_root / "shard-000.bin").read_bytes()), dtype=torch.uint16
    ).view(torch.bfloat16)
    shard_1 = torch.frombuffer(
        bytearray((pack_root / "shard-001.bin").read_bytes()), dtype=torch.uint16
    ).view(torch.bfloat16)
    assert torch.equal(
        shard_0.reshape(-1, 4),
        torch.cat((expected_a, expected_b)).to(torch.bfloat16),
    )
    assert torch.equal(shard_1.reshape(-1, 4), expected_c.to(torch.bfloat16))
    assert validate_packed_manifest(pack_root, source_root) == manifest


def test_build_loads_each_source_feature_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _write_feature_root(
        tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4), "node_c": (5, 4)}
    )
    original_load = torch.load
    load_counts: dict[str, int] = {}

    def counting_load(path: Path, *, map_location: str, weights_only: bool) -> object:
        load_counts[path.name] = load_counts.get(path.name, 0) + 1
        return original_load(path, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", _SynchronousExecutor)
    monkeypatch.setattr(torch, "load", counting_load)

    build_packed_features(source_root, tmp_path / "pack", workers=2)

    assert load_counts == {"node_a.pt": 1, "node_b.pt": 1, "node_c.pt": 1}


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/tmp/outside.pt",
        "../outside.pt",
        "embeddings/../../outside.pt",
    ],
)
def test_build_rejects_unsafe_source_index_paths_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    (source_root / "index.json").write_text(json.dumps({"node_a": unsafe_path}))

    def unexpected_executor(*args: object, **kwargs: object) -> None:
        pytest.fail("ProcessPoolExecutor started before source paths were validated")

    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("torch.load ran before source paths were validated")

    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", unexpected_executor)
    monkeypatch.setattr(torch, "load", unexpected_load)

    with pytest.raises(ValueError, match="Unsafe source feature path"):
        build_packed_features(source_root, tmp_path / "pack", workers=1)

    assert not (tmp_path / "pack").exists()


def test_build_rejects_source_index_symlink_escape_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    outside = tmp_path / "outside.pt"
    torch.save(torch.zeros(3, 4), outside)
    link = source_root / "embeddings/escape.pt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    (source_root / "index.json").write_text(json.dumps({"node_a": "embeddings/escape.pt"}))

    def unexpected_executor(*args: object, **kwargs: object) -> None:
        pytest.fail("ProcessPoolExecutor started before source paths were validated")

    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("torch.load ran before source paths were validated")

    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", unexpected_executor)
    monkeypatch.setattr(torch, "load", unexpected_load)

    with pytest.raises(ValueError, match="Unsafe source feature path"):
        build_packed_features(source_root, tmp_path / "pack", workers=1)

    assert not (tmp_path / "pack").exists()


def test_manifest_source_identity_rejects_unsafe_index_path(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    (source_root / "index.json").write_text(json.dumps({"node_a": "../outside.pt"}))
    pack_root = _write_minimal_pack(tmp_path / "pack", source_root)

    with pytest.raises(ValueError, match="Unsafe source feature path"):
        validate_packed_manifest(pack_root, source_root)


@pytest.mark.parametrize(
    ("source_filename", "message"),
    [
        ("metadata.json", "source metadata hash"),
        ("index.json", "source index hash"),
    ],
)
def test_build_rejects_source_manifest_mutation_during_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_filename: str,
    message: str,
) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    source_path = source_root / source_filename

    def mutate_source() -> None:
        source_path.write_text(source_path.read_text() + "\n")

    def executor_factory(max_workers: int) -> _SynchronousExecutor:
        return _SynchronousExecutor(max_workers, after_map=mutate_source)

    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", executor_factory)

    with pytest.raises(ValueError, match=message):
        build_packed_features(source_root, tmp_path / "pack", workers=1)

    assert not (tmp_path / "pack").exists()


def test_failed_build_never_publishes_final_directory(tmp_path: Path) -> None:
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4)})
    torch.save(torch.ones(3, 3), source_root / "embeddings/node_a.pt")
    pack_root = tmp_path / "pack"
    sentinel = tmp_path / ".pack.tmp-unrelated"
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep")

    with pytest.raises(ValueError, match="input_dim"):
        build_packed_features(source_root, pack_root, workers=1)

    assert not pack_root.exists()
    assert list(tmp_path.glob(".pack.tmp-*")) == [sentinel]
    assert (sentinel / "keep.txt").read_text() == "keep"


def test_packed_table_assembles_the_legacy_batch_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", _SynchronousExecutor)
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4)})
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


def test_packed_table_gathers_arbitrary_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packed_features, "ProcessPoolExecutor", _SynchronousExecutor)
    source_root = _write_feature_root(tmp_path / "source", {"node_a": (3, 4), "node_b": (2, 4)})
    pack_root = tmp_path / "pack"
    build_packed_features(source_root, pack_root, workers=1)
    table = PackedFeatureTable.from_pack(pack_root, torch.device("cpu"))

    tokens, lengths = table.gather_nodes(torch.tensor([1, 0]), boundary=4)

    assert lengths.tolist() == [2, 3]
    assert tokens.shape == (2, 4, 4)
    assert torch.count_nonzero(tokens[0, 2:]) == 0
    assert torch.count_nonzero(tokens[1, 3:]) == 0
