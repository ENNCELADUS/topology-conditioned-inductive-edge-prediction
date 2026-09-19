"""Deterministic bounded construction helpers for the motif dictionary builder."""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import networkx as nx
import torch
from src.data.motif_dictionary import validate_dictionary
from src.data.motif_template import MotifTemplateTable
from src.experiments.motif_dictionary_build import (
    CandidateSet,
    _expand_swap_bank,
    _load_stage1_encoder,
    _pairwise_scaled_distance,
    _scaled_distance,
    build_dictionary,
    estimate_scales,
    select_representatives,
)


def _encoder(weights: torch.Tensor) -> torch.Tensor:
    """Small exactly swap-equivariant four-field graph-token encoder."""
    left = torch.stack((weights[:, :8].sum(1), weights[:, 16:24].sum(1)), dim=1)
    right = torch.stack((weights[:, 8:16].sum(1), weights[:, 24:32].sum(1)), dim=1)
    relation = torch.stack((weights[:, 32:].sum(1), weights.sum(1)), dim=1)
    count = torch.stack((weights.square().sum(1), weights.abs().sum(1)), dim=1)
    return torch.stack((left, right, relation, count), dim=1)


def test_tied_scale_estimation_is_positive_and_deterministic() -> None:
    weights = torch.zeros(4, 96)
    weights[:, 16] = torch.arange(1, 5)
    weights[:, 24] = torch.arange(4, 0, -1)
    weights[:, 32] = torch.arange(4)
    first = estimate_scales(weights, _encoder, batch_size=2)
    second = estimate_scales(weights, _encoder, batch_size=3)
    torch.testing.assert_close(first, second)
    assert first[0] == first[1]
    assert bool((first > 0).all())


def test_bounded_pairwise_distance_matches_direct_formula() -> None:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(17, 4, 6, generator=generator)
    scales = torch.tensor([2.0, 2.0, 0.5, 3.0])
    torch.testing.assert_close(
        _pairwise_scaled_distance(tokens, scales),
        _scaled_distance(tokens, tokens, scales),
        rtol=2e-5,
        atol=2e-6,
    )


def test_real_representative_allocation_and_swap_expansion() -> None:
    rows = []
    categories = []
    pairs = []
    for category, count in enumerate((1, 3, 4, 4, 4)):
        for ordinal in range(count):
            row = torch.zeros(96)
            row[16] = category + ordinal / 10 + 0.1
            row[24] = ordinal / 20 + 0.2
            rows.append(row)
            categories.append(category)
            pairs.append((f"u{category}_{ordinal}", f"v{category}_{ordinal}"))
    weights = torch.stack(rows)
    tokens = _encoder(weights)
    candidates = CandidateSet(pairs, weights, torch.tensor(categories))
    selected = select_representatives(candidates, tokens, torch.ones(4))
    assert len(selected) == 16 and len(set(selected)) == 16
    bank_weights, bank_tokens, swap_index, sources = _expand_swap_bank(
        weights[selected], tokens[selected], [pairs[index] for index in selected]
    )
    assert bank_weights.shape[0] <= 32
    assert swap_index.index_select(0, swap_index).tolist() == list(range(len(sources)))
    torch.testing.assert_close(
        bank_tokens.index_select(0, swap_index), bank_tokens[:, (1, 0, 2, 3)]
    )


def test_insufficient_category_uses_farthest_unused_real_templates() -> None:
    weights = torch.zeros(6, 96)
    for index in range(6):
        weights[index, 16] = index + 1
    candidates = CandidateSet(
        [(f"a{index}", f"b{index}") for index in range(6)],
        weights,
        torch.ones(6, dtype=torch.int64),
    )
    selected = select_representatives(candidates, _encoder(weights), torch.ones(4))
    assert len(selected) == 6
    assert len(set(selected)) == 6


def test_representative_indices_refer_to_original_rows_after_deduplication() -> None:
    weights = torch.zeros(4, 96)
    weights[:, 16] = torch.tensor([1.0, 1.0, 2.0, 3.0])
    candidates = CandidateSet(
        [("a0", "b0"), ("duplicate", "duplicate"), ("a2", "b2"), ("a3", "b3")],
        weights,
        torch.ones(4, dtype=torch.int64),
    )
    selected = select_representatives(candidates, _encoder(weights), torch.ones(4))
    assert 1 not in selected
    assert len({weights[index].numpy().tobytes() for index in selected}) == len(selected)


def test_stage_two_checkpoint_is_rejected_as_dictionary_token_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "model_family": "v3_1_motif_prompt",
            "model_config": {
                "motif_prompt": {
                    "stage": "two",
                    "base_checkpoint": "base.pt",
                    "bundle_checkpoint": "stage1.pt",
                    "width": 8,
                    "reader": {"layers": 1, "dim": 8, "heads": 1, "rrwp_k": 1},
                }
            },
            "model_state": {},
        },
        checkpoint,
    )
    try:
        _load_stage1_encoder(checkpoint, torch.device("cpu"))
    except ValueError as error:
        assert "require a Stage I" in str(error)
    else:
        raise AssertionError("Stage II checkpoint was accepted as a dictionary token source")


def test_small_end_to_end_build_produces_a_valid_swap_complete_artifact() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("u", "w"),
            ("w", "v"),
            ("u", "left"),
            ("left", "right"),
            ("right", "v"),
            ("u", "orphan_u"),
            ("v", "orphan_v"),
        ]
    )
    table = MotifTemplateTable(graph)
    pairs = list(combinations_with_replacement(table.nodes, 2))
    artifact = build_dictionary(
        table=table,
        corpus_pairs=pairs,
        epoch1_pairs=pairs,
        encoder=_encoder,
        batch_size=5,
        metadata={"test": True},
    )
    validated = validate_dictionary(artifact)
    assert 1 <= validated.weights.shape[0] <= 32
    count = validated.metadata["real_representatives"]
    assert isinstance(count, int) and count <= 16
