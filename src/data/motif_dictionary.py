"""Training-only motif dictionary artifacts and routing mathematics.

The dictionary stores graph-derived prompt tokens in the frozen Stage-I reader's
coordinate system.  This module deliberately contains no model or benchmark
loading: construction lives in :mod:`src.experiments.motif_dictionary_build`,
while training and scoring consume the same small, self-validating artifact.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from src.data.motif_template import ATTACH_L, ATTACH_R, INTERIOR, N_EDGES, SWAP_PERM

FORMAT_VERSION: Final[int] = 1
FIELD_ORDER: Final[tuple[str, ...]] = ("topo_u", "topo_v", "topo_rel", "topo_cnt")
CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "empty",
    "attachment_only",
    "closure_only",
    "bridge_only",
    "combined",
)


@dataclass(frozen=True)
class DictionaryArtifact:
    """Fixed constants shared by dictionary-oracle and sequence-router lanes."""

    weights: Tensor
    tokens: Tensor
    scales: Tensor
    temperature: float
    mean_weights: Tensor
    swap_index: Tensor
    metadata: dict[str, object]


def _cpu_tensor(value: Tensor, *, dtype: torch.dtype) -> Tensor:
    return value.detach().to(device="cpu", dtype=dtype).contiguous()


def validate_dictionary(artifact: DictionaryArtifact) -> DictionaryArtifact:
    """Return a canonical CPU artifact after checking its complete contract."""
    weights = _cpu_tensor(artifact.weights, dtype=torch.float32)
    tokens = _cpu_tensor(artifact.tokens, dtype=torch.float32)
    scales = _cpu_tensor(artifact.scales, dtype=torch.float32)
    mean_weights = _cpu_tensor(artifact.mean_weights, dtype=torch.float32)
    swap_index = _cpu_tensor(artifact.swap_index, dtype=torch.int64)
    if weights.ndim != 2 or weights.shape[1] != N_EDGES or weights.shape[0] == 0:
        raise ValueError(f"dictionary weights must have shape (K, {N_EDGES}) with K > 0")
    k = weights.shape[0]
    if tokens.ndim != 3 or tokens.shape[:2] != (k, len(FIELD_ORDER)):
        raise ValueError("dictionary tokens must have shape (K, 4, width)")
    if scales.shape != (len(FIELD_ORDER),) or not bool(torch.all(scales > 0)):
        raise ValueError("dictionary scales must be four positive scalars")
    if not torch.equal(scales[0], scales[1]):
        raise ValueError("dictionary endpoint scales must be tied")
    if mean_weights.shape != (k,):
        raise ValueError("dictionary mean_weights must have shape (K,)")
    if swap_index.shape != (k,):
        raise ValueError("dictionary swap_index must have shape (K,)")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("dictionary weights contain non-finite values")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("dictionary tokens contain non-finite values")
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("dictionary scales contain non-finite values")
    if not bool(torch.isfinite(mean_weights).all()):
        raise ValueError("dictionary mean_weights contain non-finite values")
    if not torch.isfinite(torch.tensor(float(artifact.temperature))) or artifact.temperature <= 0:
        raise ValueError("dictionary temperature must be finite and positive")
    expected = torch.arange(k, dtype=torch.int64)
    if bool(((swap_index < 0) | (swap_index >= k)).any()):
        raise ValueError("dictionary swap_index is out of range")
    if not torch.equal(swap_index.index_select(0, swap_index), expected):
        raise ValueError("dictionary swap_index must be an involution")
    perm = torch.as_tensor(SWAP_PERM, dtype=torch.int64)
    if not torch.equal(weights.index_select(0, swap_index), weights.index_select(1, perm)):
        raise ValueError("dictionary weights disagree with swap_index")
    swapped_tokens = tokens[:, (1, 0, 2, 3)]
    if not torch.allclose(
        tokens.index_select(0, swap_index), swapped_tokens, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("dictionary tokens disagree with swap_index")
    total = mean_weights.sum()
    if bool((mean_weights < 0).any()) or not torch.isclose(total, total.new_tensor(1.0), atol=1e-5):
        raise ValueError("dictionary mean_weights must be a probability vector")
    if not torch.allclose(
        mean_weights, mean_weights.index_select(0, swap_index), rtol=1e-5, atol=1e-6
    ):
        raise ValueError("dictionary mean_weights must be swap invariant")
    return DictionaryArtifact(
        weights=weights,
        tokens=tokens,
        scales=scales,
        temperature=float(artifact.temperature),
        mean_weights=mean_weights,
        swap_index=swap_index,
        metadata=dict(artifact.metadata),
    )


def save_dictionary(artifact: DictionaryArtifact, path: Path) -> None:
    """Atomically save a validated dictionary artifact."""
    value = validate_dictionary(artifact)
    payload = {
        "format_version": FORMAT_VERSION,
        "weights": value.weights,
        "tokens": value.tokens,
        "scales": value.scales,
        "temperature": value.temperature,
        "mean_weights": value.mean_weights,
        "swap_index": value.swap_index,
        "metadata": value.metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_dictionary(path: Path) -> DictionaryArtifact:
    """Load and validate a dictionary artifact."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported motif dictionary artifact")
    required = {
        "weights",
        "tokens",
        "scales",
        "temperature",
        "mean_weights",
        "swap_index",
        "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: motif dictionary is missing {missing}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: motif dictionary metadata must be a mapping")
    return validate_dictionary(
        DictionaryArtifact(
            weights=payload["weights"],
            tokens=payload["tokens"],
            scales=payload["scales"],
            temperature=float(payload["temperature"]),
            mean_weights=payload["mean_weights"],
            swap_index=payload["swap_index"],
            metadata=dict(metadata),
        )
    )


def routing_targets(tokens: Tensor, artifact: DictionaryArtifact) -> Tensor:
    """Return fixed soft routing targets for true Stage-I token rows."""
    if tokens.ndim != 3 or tokens.shape[1:] != artifact.tokens.shape[1:]:
        raise ValueError(
            "target tokens must have shape (B, 4, width) matching dictionary tokens"
        )
    target = tokens.float()
    bank = artifact.tokens.to(device=target.device, dtype=torch.float32)
    scales = artifact.scales.to(device=target.device, dtype=torch.float32)
    distance = ((target[:, None] - bank[None]) / scales[None, None, :, None]).square().mean(
        dim=(-1, -2)
    )
    routes = torch.softmax(-distance / float(artifact.temperature), dim=-1)
    if not bool(torch.isfinite(routes).all()):
        raise ValueError("routing targets contain non-finite values")
    return routes


def mix_dictionary_tokens(routes: Tensor, tokens: Tensor) -> Tensor:
    """Mix dictionary prompt fields in FP32."""
    if routes.ndim != 2 or tokens.ndim != 3 or routes.shape[1] != tokens.shape[0]:
        raise ValueError("routes (B, K) and tokens (K, 4, width) have incompatible shapes")
    # einsum participates in autocast, so merely casting its inputs does not
    # guarantee the dictionary mixture itself remains FP32.
    with torch.autocast(device_type=routes.device.type, enabled=False):
        return torch.einsum("bk,kfd->bfd", routes.float(), tokens.to(routes.device).float())


def closure_nonempty_from_weights(weights: Tensor) -> Tensor:
    """Identify rows with positive complete two-hop wedge mass."""
    if weights.shape[-1] != N_EDGES:
        raise ValueError(f"template weights must end in {N_EDGES} entries")
    values = weights.float()
    return (values[..., :8] * values[..., 8:16]).sum(dim=-1) > 0


def classify_template_weights(weights: Tensor) -> Tensor:
    """Classify compiled templates into the five dictionary strata.

    Category indices follow :data:`CATEGORY_NAMES`.  A complete bridge requires
    both endpoint attachments and an interior edge; attachment-only rows are
    therefore distinct from true empty templates.
    """
    if weights.ndim != 2 or weights.shape[1] != N_EDGES:
        raise ValueError(f"template weights must have shape (B, {N_EDGES})")
    values = weights.float()
    closure = closure_nonempty_from_weights(values)
    left = values[:, ATTACH_L]
    right = values[:, ATTACH_R]
    interior = values[:, INTERIOR].reshape(-1, 8, 8)
    bridge = torch.einsum("bi,bij,bj->b", left, interior, right) > 0
    nonzero = values.abs().sum(dim=-1) > 0
    result = torch.zeros(values.shape[0], dtype=torch.int64, device=values.device)
    result[nonzero & ~closure & ~bridge] = 1
    result[closure & ~bridge] = 2
    result[~closure & bridge] = 3
    result[closure & bridge] = 4
    return result


__all__ = [
    "CATEGORY_NAMES",
    "DictionaryArtifact",
    "classify_template_weights",
    "closure_nonempty_from_weights",
    "load_dictionary",
    "mix_dictionary_tokens",
    "routing_targets",
    "save_dictionary",
    "validate_dictionary",
]
