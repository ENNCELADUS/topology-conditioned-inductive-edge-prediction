"""Attribute-only endpoint caching and scoring for the frozen L3-PPI encoder."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.baselines.l3ppi import L3PPI
from src.data.packed_features import PackedFeatureTable

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_nodes(
    model: L3PPI,
    pack_dir: Path,
    nodes: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Encode unique nodes once in FP32, keeping the large token pack on CPU."""
    if batch_size < 1:
        raise ValueError("encode batch size must be positive")
    if not nodes:
        return {}
    table = PackedFeatureTable.from_pack(pack_dir, torch.device("cpu"))
    position = table.manifest.node_index()
    ordered = sorted(set(nodes))
    missing = set(ordered) - position.keys()
    if missing:
        raise ValueError(f"L3-PPI endpoints lack intrinsic features: {sorted(missing)}")
    result: dict[str, torch.Tensor] = {}
    for start in range(0, len(ordered), batch_size):
        names = ordered[start : start + batch_size]
        index = torch.tensor([position[node] for node in names])
        bound = min(model.max_length, int(table.lengths[index].max()))
        tokens, lengths = table.gather_nodes(index, bound)
        pooled = model.encode(tokens.to(device), lengths.to(device)).cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("non-finite L3-PPI endpoint features")
        result.update(zip(names, pooled.unbind(), strict=True))
        if start % (batch_size * 100) == 0:
            logger.info("encoded %d/%d endpoints", start + len(names), len(ordered))
    return result


@torch.no_grad()
def score_cached(
    model: L3PPI,
    features: Mapping[str, torch.Tensor],
    pairs: Sequence[tuple[str, str]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Score deterministically; return logits and mean soft/hard path counts."""
    if batch_size < 1:
        raise ValueError("score batch size must be positive")
    batch_size = min(batch_size, 64)  # The generic scorer defaults to 8192 pair rows.
    model.eval()
    output = np.empty((3, len(pairs)), dtype=np.float32)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        a = torch.stack([features[u] for u, _ in batch]).to(device)
        b = torch.stack([features[v] for _, v in batch]).to(device)
        logits, probabilities, activations = model(a, b)
        values = torch.stack((logits, probabilities.sum(-1).mean(-1), activations.sum(-1).mean(-1)))
        output[:, start : start + len(batch)] = values.cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("non-finite L3-PPI scores")
    return output[0], output[1], output[2]


def score_pairs(
    model: L3PPI,
    pack_dir: Path,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 64,
) -> NDArray[np.float32]:
    """Deploy from checkpoint weights and intrinsic attributes, without a graph."""
    nodes = sorted({node for pair in pairs for node in pair})
    features = encode_nodes(model, pack_dir, nodes, device)
    return score_cached(model, features, pairs, device, batch_size)[0]
