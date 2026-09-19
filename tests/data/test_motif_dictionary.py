"""Artifact, fixed routing geometry and compiler strata for motif dictionaries."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
from src.data.motif_dictionary import (
    DictionaryArtifact,
    classify_template_weights,
    closure_nonempty_from_weights,
    load_dictionary,
    mix_dictionary_tokens,
    routing_targets,
    save_dictionary,
)
from src.data.motif_template import SWAP_PERM, MotifTemplateTable

pytestmark = pytest.mark.unit


def _artifact() -> DictionaryArtifact:
    first = torch.zeros(96)
    first[16] = 0.5
    second = first[torch.as_tensor(SWAP_PERM)]
    token = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    swapped = token.index_select(0, torch.tensor([1, 0, 2, 3]))
    return DictionaryArtifact(
        weights=torch.stack((first, second)),
        tokens=torch.stack((token, swapped)),
        scales=torch.tensor([2.0, 2.0, 3.0, 4.0]),
        temperature=0.75,
        mean_weights=torch.tensor([0.5, 0.5]),
        swap_index=torch.tensor([1, 0]),
        metadata={"source": "synthetic"},
    )


def test_routing_targets_are_endpoint_swap_equivariant() -> None:
    artifact = _artifact()
    targets = torch.stack((artifact.tokens[0] + 0.1, artifact.tokens[1] - 0.2))
    swapped = targets[:, (1, 0, 2, 3)]
    routes = routing_targets(targets, artifact)
    swapped_routes = routing_targets(swapped, artifact)
    torch.testing.assert_close(
        swapped_routes,
        routes.index_select(1, artifact.swap_index),
        rtol=1e-6,
        atol=1e-7,
    )


def test_one_hot_mixture_is_exact_fp32_under_autocast() -> None:
    artifact = _artifact()
    routes = torch.tensor([[0.0, 1.0]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = mix_dictionary_tokens(routes, artifact.tokens)
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed[0], artifact.tokens[1], rtol=0, atol=0)


def test_artifact_round_trip_validates_swap_contract(tmp_path: Path) -> None:
    path = tmp_path / "dictionary.pt"
    save_dictionary(_artifact(), path)
    loaded = load_dictionary(path)
    assert loaded.metadata == {"source": "synthetic"}
    torch.testing.assert_close(loaded.tokens, _artifact().tokens)
    broken = _artifact()
    broken.tokens[1, 0, 0] += 1
    with pytest.raises(ValueError, match="tokens disagree"):
        save_dictionary(broken, path)


def test_attachment_only_compiler_row_is_neither_empty_nor_bridge() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(("u", "v"))
    graph.add_edge("u", "left")
    graph.add_edge("v", "right")
    weights = torch.from_numpy(MotifTemplateTable(graph).compile_row("u", "v").weights)[None]
    assert int(classify_template_weights(weights)[0]) == 1
    assert not bool(closure_nonempty_from_weights(weights)[0])
    assert float(weights.abs().sum()) > 0


def test_complete_bridge_requires_attachments_and_interior() -> None:
    weights = torch.zeros(3, 96)
    weights[0, 16] = 1  # attachment-only
    weights[1, 32] = 1  # incomplete interior alone remains a nonempty residual stratum
    weights[2, 16] = weights[2, 24] = weights[2, 32] = 1
    assert classify_template_weights(weights).tolist() == [1, 1, 3]
