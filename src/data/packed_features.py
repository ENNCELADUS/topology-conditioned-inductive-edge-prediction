"""Manifest schema, persistence, hashing, and validation for packed node features."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import cast

import torch

from src.data.distributed_pairs import CompactPairBatch

PACK_FORMAT = "bf16_flat_shards_v1"
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "input_dim",
        "dtype",
        "source_metadata_sha256",
        "source_index_sha256",
        "nodes",
        "shards",
        "pack_workers",
        "build_seconds",
    }
)
_NODE_FIELDS = frozenset({"node_id", "shard_index", "shard_offset", "global_offset", "length"})
_SHARD_FIELDS = frozenset({"filename", "num_tokens", "byte_size", "sha256"})


@dataclass(frozen=True)
class PackedNodeRecord:
    """Location and length of one node's token sequence in a packed shard."""

    node_id: str
    shard_index: int
    shard_offset: int
    global_offset: int
    length: int


@dataclass(frozen=True)
class PackedShardRecord:
    """Integrity metadata for one packed feature shard."""

    filename: str
    num_tokens: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class PackedFeatureManifest:
    """Complete metadata contract for a packed feature store."""

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
        """Map each node ID to its ordered position in the manifest."""
        return {record.node_id: index for index, record in enumerate(self.nodes)}


class PackedFeatureTable:
    """Packed token sequences and node locations resident on one device."""

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
        """Load all packed shards into a contiguous token table on ``device``."""
        manifest = load_packed_manifest(pack_root)
        total_tokens = sum(shard.num_tokens for shard in manifest.shards)
        tokens = torch.empty(
            (total_tokens, manifest.input_dim),
            dtype=torch.bfloat16,
            device=device,
        )
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

    def _gather_endpoint(
        self, node_indices: torch.Tensor, boundary: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather node token rows and zero positions beyond each true length."""
        node_indices = node_indices.to(self.tokens.device)
        offsets = self.offsets.index_select(0, node_indices)
        lengths = self.lengths.index_select(0, node_indices)
        steps = torch.arange(boundary, device=self.tokens.device)
        mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
        positions = offsets.unsqueeze(1) + steps.unsqueeze(0)
        positions = positions.masked_fill(~mask, 0)
        gathered = self.tokens[positions]
        return gathered.masked_fill(~mask.unsqueeze(2), 0), lengths

    def gather_nodes(
        self, node_indices: torch.Tensor, boundary: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather padded token sequences for arbitrary packed node indices."""
        return self._gather_endpoint(node_indices, boundary)

    def assemble(self, compact: CompactPairBatch) -> dict[str, torch.Tensor]:
        """Assemble a compact pair batch into the padded ``V3_1`` contract."""
        emb_a, len_a = self._gather_endpoint(compact.node_a, compact.bucket_boundary)
        emb_b, len_b = self._gather_endpoint(compact.node_b, compact.bucket_boundary)
        device = self.tokens.device
        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": compact.labels.to(device),
            "_row_id": compact.row_ids.to(device),
            "_local_pair_count": torch.tensor(compact.row_ids.numel(), device=device),
            "_global_pair_count": torch.tensor(compact.global_pair_count, device=device),
        }


@dataclass(frozen=True)
class _ShardJob:
    shard_index: int
    source_root: Path
    temp_root: Path
    input_dim: int
    entries: Sequence[tuple[str, str]]
    f0_node_ids: frozenset[str] = frozenset()


def _write_shard(
    job: _ShardJob,
) -> tuple[PackedShardRecord, Sequence[PackedNodeRecord], Sequence[str], torch.Tensor | None]:
    shard_path = job.temp_root / f"shard-{job.shard_index:03d}.bin"
    records: list[PackedNodeRecord] = []
    f0_node_ids: list[str] = []
    f0_rows: list[torch.Tensor] = []
    token_offset = 0
    digest = hashlib.sha256()
    with shard_path.open("wb") as handle:
        for node_id, relative_path in job.entries:
            tensor = cast(
                torch.Tensor,
                torch.load(
                    job.source_root / relative_path,
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            if tensor.ndim != 2 or tensor.size(1) != job.input_dim:
                raise ValueError(f"feature {node_id} does not match input_dim={job.input_dim}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"feature {node_id} must be float32")
            if node_id in job.f0_node_ids:
                f0_node_ids.append(node_id)
                f0_rows.append(tensor.mean(dim=0))
            bf16 = tensor.to(torch.bfloat16).contiguous().view(torch.uint16)
            payload = bf16.numpy().tobytes()
            handle.write(payload)
            digest.update(payload)
            records.append(
                PackedNodeRecord(
                    node_id,
                    job.shard_index,
                    token_offset,
                    0,
                    tensor.size(0),
                )
            )
            token_offset += int(tensor.size(0))
    shard = PackedShardRecord(
        filename=shard_path.name,
        num_tokens=token_offset,
        byte_size=shard_path.stat().st_size,
        sha256=digest.hexdigest(),
    )
    f0_matrix = torch.stack(f0_rows).float() if f0_rows else None
    return shard, tuple(records), tuple(f0_node_ids), f0_matrix


def build_packed_features(
    source_root: Path,
    pack_root: Path,
    workers: int,
    *,
    temp_prefix: str | None = None,
    f0_node_ids: Sequence[str] | None = None,
    f0_cache_path: Path | None = None,
) -> PackedFeatureManifest:
    """Pack per-node float32 tensors into validated BF16 shards atomically.

    When ``f0_node_ids`` and ``f0_cache_path`` are supplied, the workers also
    retain the exact float32 means from their one source-tensor load and publish
    the standard F0 cache without reopening the per-node ``.pt`` files.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if pack_root.exists():
        raise FileExistsError(f"packed feature directory already exists: {pack_root}")
    if (f0_node_ids is None) != (f0_cache_path is None):
        raise ValueError("f0_node_ids and f0_cache_path must be supplied together")
    resolved_temp_prefix = temp_prefix or f".{pack_root.name}.tmp-"
    if not resolved_temp_prefix or Path(resolved_temp_prefix).name != resolved_temp_prefix:
        raise ValueError("temp_prefix must be a non-empty filename prefix")

    source_metadata_sha256 = sha256_file(source_root / "metadata.json")
    source_index_sha256 = sha256_file(source_root / "index.json")
    metadata = cast(
        dict[str, object],
        json.loads((source_root / "metadata.json").read_text(encoding="utf-8")),
    )
    index = _validated_source_index(
        source_root,
        json.loads((source_root / "index.json").read_text(encoding="utf-8")),
    )
    input_dim = cast(int, metadata["input_dim"])
    entries = tuple(index.items())
    resolved_f0_node_ids = tuple(f0_node_ids or ())
    if len(resolved_f0_node_ids) != len(set(resolved_f0_node_ids)):
        raise ValueError("f0_node_ids must not contain duplicates")
    missing_f0_nodes = set(resolved_f0_node_ids) - set(index)
    if missing_f0_nodes:
        raise ValueError(
            f"F0 nodes are missing from the source index: {sorted(missing_f0_nodes)!r}"
        )
    requested_f0_nodes = frozenset(resolved_f0_node_ids)
    job_count = min(workers, len(entries))
    base_size, extra = divmod(len(entries), job_count) if job_count else (0, 0)

    pack_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=resolved_temp_prefix, dir=pack_root.parent))
    started = time.monotonic()
    try:
        jobs: list[_ShardJob] = []
        start = 0
        for shard_index in range(job_count):
            size = base_size + (1 if shard_index < extra else 0)
            jobs.append(
                _ShardJob(
                    shard_index=shard_index,
                    source_root=source_root,
                    temp_root=temp_root,
                    input_dim=input_dim,
                    entries=entries[start : start + size],
                    f0_node_ids=requested_f0_nodes,
                )
            )
            start += size

        if workers == 1 and jobs:
            results = [_write_shard(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_write_shard, jobs))
        results.sort(key=lambda result: result[1][0].shard_index)

        shards = tuple(result[0] for result in results)
        nodes: list[PackedNodeRecord] = []
        global_offset = 0
        for _, records, _, _ in results:
            for record in records:
                nodes.append(replace(record, global_offset=global_offset))
                global_offset += record.length

        f0_matrix: torch.Tensor | None = None
        if resolved_f0_node_ids:
            f0_by_node: dict[str, torch.Tensor] = {}
            for _, _, result_node_ids, result_matrix in results:
                if result_matrix is None:
                    continue
                if len(result_node_ids) != result_matrix.size(0):
                    raise ValueError("packed F0 worker result has inconsistent row metadata")
                f0_by_node.update(
                    (node_id, result_matrix[row_index])
                    for row_index, node_id in enumerate(result_node_ids)
                )
            f0_matrix = torch.stack(
                [f0_by_node[node_id] for node_id in resolved_f0_node_ids]
            ).float()

        manifest = PackedFeatureManifest(
            format=PACK_FORMAT,
            input_dim=input_dim,
            dtype="bfloat16",
            source_metadata_sha256=source_metadata_sha256,
            source_index_sha256=source_index_sha256,
            nodes=tuple(nodes),
            shards=shards,
            pack_workers=workers,
            build_seconds=time.monotonic() - started,
        )
        write_packed_manifest(temp_root, manifest)
        validate_packed_manifest(temp_root, source_root, verify_shard_sha256=False)
        temp_root.rename(pack_root)
        if f0_cache_path is not None and f0_matrix is not None:
            temporary_f0_path = f0_cache_path.with_name(f"{f0_cache_path.name}.tmp-{os.getpid()}")
            try:
                f0_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"matrix": f0_matrix, "node_ids": list(resolved_f0_node_ids)},
                    temporary_f0_path,
                )
                os.replace(temporary_f0_path, f0_cache_path)
            except BaseException:
                temporary_f0_path.unlink(missing_ok=True)
                shutil.rmtree(pack_root, ignore_errors=True)
                raise
        return manifest
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_packed_manifest(pack_root: Path, manifest: PackedFeatureManifest) -> None:
    """Atomically write ``manifest`` to ``pack_root/manifest.json``."""
    manifest_path = pack_root / "manifest.json"
    temporary_path = pack_root / "manifest.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)


def load_packed_manifest(pack_root: Path) -> PackedFeatureManifest:
    """Load a packed feature manifest and reconstruct its immutable records."""
    try:
        raw = json.loads((pack_root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("top level must be an object")
        _require_exact_keys(raw, "top level", _MANIFEST_FIELDS)
        node_rows = _object_list(raw["nodes"], "nodes")
        shard_rows = _object_list(raw["shards"], "shards")
        for row in node_rows:
            _require_exact_keys(row, "node record", _NODE_FIELDS)
        for row in shard_rows:
            _require_exact_keys(row, "shard record", _SHARD_FIELDS)
        return PackedFeatureManifest(
            format=_string(raw["format"], "format"),
            input_dim=_integer(raw["input_dim"], "input_dim"),
            dtype=_string(raw["dtype"], "dtype"),
            source_metadata_sha256=_string(raw["source_metadata_sha256"], "source_metadata_sha256"),
            source_index_sha256=_string(raw["source_index_sha256"], "source_index_sha256"),
            nodes=tuple(
                PackedNodeRecord(
                    node_id=_string(row["node_id"], "node_id"),
                    shard_index=_integer(row["shard_index"], "shard_index"),
                    shard_offset=_integer(row["shard_offset"], "shard_offset"),
                    global_offset=_integer(row["global_offset"], "global_offset"),
                    length=_integer(row["length"], "length"),
                )
                for row in node_rows
            ),
            shards=tuple(
                PackedShardRecord(
                    filename=_string(row["filename"], "filename"),
                    num_tokens=_integer(row["num_tokens"], "num_tokens"),
                    byte_size=_integer(row["byte_size"], "byte_size"),
                    sha256=_string(row["sha256"], "sha256"),
                )
                for row in shard_rows
            ),
            pack_workers=_integer(raw["pack_workers"], "pack_workers"),
            build_seconds=_number(raw["build_seconds"], "build_seconds"),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Packed feature manifest is malformed: {exc}") from exc


def _require_exact_keys(value: Mapping[str, object], field: str, expected: frozenset[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise TypeError(f"{field} has missing keys {missing} and unknown keys {unknown}")


def _object_list(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"{field} must be a list of objects")
    return cast(list[Mapping[str, object]], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    return float(value)


def _validated_source_index(source_root: Path, value: object) -> dict[str, str]:
    """Validate source feature paths before any feature file is opened."""
    if not isinstance(value, dict) or not all(isinstance(node_id, str) for node_id in value):
        raise ValueError("Packed feature source index is malformed")

    root = source_root.resolve()
    validated: dict[str, str] = {}
    for node_id, raw_path in value.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Packed feature source index is malformed")
        path = Path(raw_path)
        windows_path = PureWindowsPath(raw_path)
        normalized = str(path)
        unsafe = (
            path.is_absolute()
            or windows_path.is_absolute()
            or normalized != raw_path
            or ".." in path.parts
        )
        if not unsafe:
            try:
                (source_root / path).resolve().relative_to(root)
            except ValueError:
                unsafe = True
        if unsafe:
            raise ValueError(f"Unsafe source feature path for node {node_id!r}: {raw_path!r}")
        validated[node_id] = raw_path
    return validated


def validate_packed_manifest(
    pack_root: Path,
    source_root: Path | None,
    *,
    verify_shard_sha256: bool = True,
) -> PackedFeatureManifest:
    """Load and strictly validate a packed manifest and its referenced files.

    When ``source_root`` is given, its metadata and index hashes must also match the
    source hashes captured in the manifest.
    """
    manifest = load_packed_manifest(pack_root)
    if manifest.format != PACK_FORMAT:
        raise ValueError(
            f"Unexpected packed feature format {manifest.format!r}; expected {PACK_FORMAT!r}"
        )
    if manifest.dtype != "bfloat16":
        raise ValueError(f"Unexpected packed feature dtype {manifest.dtype!r}; expected 'bfloat16'")
    if manifest.input_dim <= 0:
        raise ValueError("Packed feature input_dim must be positive")
    if manifest.pack_workers <= 0:
        raise ValueError("Packed feature pack_workers must be positive")
    if manifest.build_seconds < 0:
        raise ValueError("Packed feature build_seconds must be non-negative")

    node_ids = [record.node_id for record in manifest.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Packed feature manifest contains a duplicate node ID")

    filenames = [shard.filename for shard in manifest.shards]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Packed feature manifest contains a duplicate shard filename")
    for filename in filenames:
        path = Path(filename)
        if not filename or path.is_absolute() or path.name != filename:
            raise ValueError(f"Unsafe packed shard filename: {filename!r}")

    expected_global_offset = 0
    shard_offsets = [0] * len(manifest.shards)
    for record in manifest.nodes:
        if record.length <= 0:
            raise ValueError(f"Packed node {record.node_id!r} has non-positive length")
        if not 0 <= record.shard_index < len(manifest.shards):
            raise ValueError(f"Invalid shard index for node {record.node_id!r}")
        expected_shard_offset = shard_offsets[record.shard_index]
        if record.shard_offset != expected_shard_offset:
            raise ValueError(
                f"Noncontiguous shard offset for node {record.node_id!r}: "
                f"found {record.shard_offset}, expected {expected_shard_offset}"
            )
        if record.global_offset != expected_global_offset:
            raise ValueError(
                f"Noncontiguous global offset for node {record.node_id!r}: "
                f"found {record.global_offset}, expected {expected_global_offset}"
            )
        expected_global_offset += record.length
        shard_offsets[record.shard_index] += record.length

    for shard_index, shard in enumerate(manifest.shards):
        if shard.num_tokens <= 0:
            raise ValueError(f"Packed shard {shard.filename!r} has non-positive num_tokens")
        if shard_offsets[shard_index] != shard.num_tokens:
            raise ValueError(
                f"Packed shard token total mismatch for {shard.filename!r}: "
                f"nodes contain {shard_offsets[shard_index]}, manifest says {shard.num_tokens}"
            )
        expected_byte_size = shard.num_tokens * manifest.input_dim * 2
        if shard.byte_size != expected_byte_size:
            raise ValueError(
                f"Packed shard byte size formula mismatch for {shard.filename!r}: "
                f"found {shard.byte_size}, expected {expected_byte_size}"
            )
        shard_path = pack_root / shard.filename
        if not shard_path.is_file():
            raise ValueError(f"Packed shard file is missing: {shard.filename!r}")
        actual_size = shard_path.stat().st_size
        if actual_size != shard.byte_size:
            raise ValueError(
                f"Packed shard size mismatch for {shard.filename!r}: "
                f"found {actual_size}, expected {shard.byte_size}"
            )
        if verify_shard_sha256:
            actual_sha256 = sha256_file(shard_path)
            if actual_sha256 != shard.sha256:
                raise ValueError(f"Packed shard checksum mismatch for {shard.filename!r}")

    if expected_global_offset != sum(shard.num_tokens for shard in manifest.shards):
        raise ValueError("Packed feature global token total does not match shard totals")

    if source_root is not None:
        metadata_sha256 = sha256_file(source_root / "metadata.json")
        if metadata_sha256 != manifest.source_metadata_sha256:
            raise ValueError("Packed feature source metadata hash does not match")
        index_sha256 = sha256_file(source_root / "index.json")
        if index_sha256 != manifest.source_index_sha256:
            raise ValueError("Packed feature source index hash does not match")
        source_index = _validated_source_index(
            source_root,
            json.loads((source_root / "index.json").read_text(encoding="utf-8")),
        )
        if node_ids != list(source_index):
            raise ValueError("Packed feature manifest node order does not match source index")

    return manifest
