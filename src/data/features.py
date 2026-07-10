"""Frozen per-node token-sequence feature store and F0 mean-pool matrix builder."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

logger = logging.getLogger(__name__)

_EXPECTED_FORMAT = "torch_pt_per_node"


class FeatureStore:
    """Read-only accessor for the frozen per-node token-sequence feature cache.

    Reads ``metadata.json`` (format/input_dim) and ``index.json`` (node_id to relative
    path) from ``root`` at construction time. Individual node tensors are loaded lazily
    on demand via :meth:`load_tokens`.
    """

    def __init__(self, root: Path) -> None:
        """Load and validate the feature store metadata and index.

        Args:
            root: Directory containing ``metadata.json``, ``index.json``, and the
                per-node ``.pt`` tensor files referenced by the index.

        Raises:
            ValueError: If the metadata format is unexpected or required metadata
                fields are missing.
        """
        self._root = Path(root)
        metadata: dict[str, object] = json.loads((self._root / "metadata.json").read_text())
        declared_format = metadata.get("format")
        if declared_format != _EXPECTED_FORMAT:
            raise ValueError(
                f"Unexpected feature store format {declared_format!r}; "
                f"expected {_EXPECTED_FORMAT!r}"
            )
        if "input_dim" not in metadata:
            raise ValueError("metadata.json is missing required field 'input_dim'")
        self._input_dim = int(cast(int, metadata["input_dim"]))
        self._index: dict[str, str] = json.loads((self._root / "index.json").read_text())

    @property
    def node_ids(self) -> frozenset[str]:
        """Return the set of node ids indexed by this feature store."""
        return frozenset(self._index)

    @property
    def input_dim(self) -> int:
        """Return the per-token feature dimensionality declared in metadata."""
        return self._input_dim

    def load_tokens(self, node_id: str) -> torch.Tensor:
        """Load the raw token-sequence tensor for a single node.

        Args:
            node_id: Opaque node id to look up.

        Returns:
            A ``(L, input_dim)`` float32 tensor.

        Raises:
            KeyError: If ``node_id`` is not present in the feature index.
            ValueError: If the loaded tensor's shape or dtype does not match the
                expected ``(L, input_dim)`` float32 contract.
        """
        if node_id not in self._index:
            raise KeyError(node_id)
        path = self._root / self._index[node_id]
        tensor = cast(torch.Tensor, torch.load(path, map_location="cpu", weights_only=True))
        if tensor.ndim != 2:
            raise ValueError(
                f"Feature tensor for {node_id!r} has ndim={tensor.ndim}, expected 2 (L, input_dim)"
            )
        if tensor.size(1) != self._input_dim:
            raise ValueError(
                f"Feature tensor for {node_id!r} has dim {tensor.size(1)}, expected "
                f"input_dim={self._input_dim}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"Feature tensor for {node_id!r} has dtype {tensor.dtype}, expected torch.float32"
            )
        return tensor


def build_f0_matrix(
    store: FeatureStore, node_ids: Sequence[str], *, cache_path: Path | None = None
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build the F0 mean-pooled feature matrix for a set of nodes.

    Args:
        store: Feature store to read raw token sequences from.
        node_ids: Ordered node ids to build rows for.
        cache_path: Optional path to a cached matrix. If it exists, it is loaded and
            validated to match ``node_ids`` exactly (order included); otherwise the
            matrix is computed and saved there (when given).

    Returns:
        A tuple ``(matrix, index)`` where ``matrix`` is a ``(N, input_dim)`` float32
        tensor and ``index`` maps each node id to its row in ``matrix``.

    Raises:
        ValueError: If a cache exists at ``cache_path`` but its stored node ordering
            does not match ``node_ids``.
    """
    resolved_node_ids = list(node_ids)

    if cache_path is not None and cache_path.exists():
        cached = cast(
            dict[str, object],
            torch.load(cache_path, map_location="cpu", weights_only=True),
        )
        cached_node_ids = list(cast(list[str], cached["node_ids"]))
        if cached_node_ids != resolved_node_ids:
            raise ValueError(
                f"Cached F0 matrix at {cache_path} has a node ordering that does not "
                "match the requested node_ids"
            )
        matrix = cast(torch.Tensor, cached["matrix"])
        index = {node_id: i for i, node_id in enumerate(resolved_node_ids)}
        return matrix, index

    rows: list[torch.Tensor] = []
    for i, node_id in enumerate(resolved_node_ids):
        rows.append(store.load_tokens(node_id).mean(dim=0))
        if (i + 1) % 1000 == 0:
            logger.info("build_f0_matrix: processed %d/%d nodes", i + 1, len(resolved_node_ids))

    matrix = (
        torch.stack(rows, dim=0).float()
        if rows
        else torch.zeros(0, store.input_dim, dtype=torch.float32)
    )
    index = {node_id: i for i, node_id in enumerate(resolved_node_ids)}

    if cache_path is not None:
        # Concurrent writers (e.g. parallel shard processes sharing a default cache path) must
        # never expose a partially-written file to readers: write to a per-process temp
        # file in the same directory, then atomically replace.
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
        torch.save({"matrix": matrix, "node_ids": resolved_node_ids}, tmp_path)
        os.replace(tmp_path, cache_path)

    return matrix, index
