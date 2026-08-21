"""Contracts for the Gate A features-only oracle-candidate-set generator.

Pins the properties `FullEgoFeaturesGenerator` adds on top of the
`FullOracleGenerator` layout it inherits: the emitted node set and padding are
*identical* to the structural oracle's on the same context, the adjacency is
exactly zero, the per-node channels are ``[f0 | has_f0 | root_u | root_v |
exists]`` with features gathered by node id (missing ids zero-filled with the
indicator down), and the stashed teacher view reproduces the structural
oracle's own emission bit for bit.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder
from src.model.egostitch.generator.egostitch import GeneratorNodeState
from src.model.egostitch.generator.full_oracle import (
    FeatureEgoGraph,
    FullEgoFeaturesGenerator,
    FullEgoGraph,
    FullOracleGenerator,
)

pytestmark = pytest.mark.unit

_DIM = 6


def _feature_matrix(count: int) -> torch.Tensor:
    return torch.arange(count * _DIM, dtype=torch.float32).reshape(count, _DIM) / 7.0


def _generator(
    truth: nx.Graph[str],
    node_ids: list[str],
    *,
    feature_ids: list[str] | None = None,
    features: torch.Tensor | None = None,
) -> FullEgoFeaturesGenerator:
    generator = FullEgoFeaturesGenerator(input_dim=_DIM)
    generator.set_oracle_context(truth, node_ids)
    resolved_ids = node_ids if feature_ids is None else feature_ids
    generator.set_node_features(
        _feature_matrix(len(resolved_ids)) if features is None else features, resolved_ids
    )
    return generator


def _encode(generator: FullOracleGenerator, rows: list[int]) -> GeneratorNodeState:
    batch = len(rows)
    return generator.encode_node(
        torch.zeros(batch, 3),
        torch.zeros(batch, 1, 3),
        node_rows=torch.tensor(rows, dtype=torch.int64),
    )


def _stitch(generator: FullOracleGenerator, rows_a: list[int], rows_b: list[int]) -> FullEgoGraph:
    return generator.stitch(
        _encode(generator, rows_a),
        _encode(generator, rows_b),
        torch.tensor([a == b for a, b in zip(rows_a, rows_b, strict=True)]),
    )


def _oracle_reference(truth: nx.Graph[str], node_ids: list[str]) -> FullOracleGenerator:
    generator = FullOracleGenerator()
    generator.set_oracle_context(truth, node_ids)
    return generator


_TRUTH_EDGES = [
    ("0", "1"),
    ("0", "2"),
    ("0", "3"),
    ("1", "2"),
    ("1", "4"),
    ("2", "3"),
    ("3", "4"),
    ("4", "5"),
]


def test_emits_oracle_node_set_as_features_with_zero_adjacency() -> None:
    truth = nx.Graph(_TRUTH_EDGES)
    node_ids = [str(node) for node in range(6)]
    graph = _stitch(_generator(truth, node_ids), [0], [1])
    oracle = _stitch(_oracle_reference(truth, node_ids), [0], [1])

    assert isinstance(graph, FeatureEgoGraph)
    assert graph.x.shape == (1, 5, _DIM + 4)
    assert torch.equal(graph.mask, oracle.mask)
    assert torch.count_nonzero(graph.adj) == 0
    assert graph.adj.shape == oracle.adj.shape

    # Node order is the oracle's: endpoints [0, 1], then sorted union [2, 3, 4].
    expected = torch.nn.functional.layer_norm(_feature_matrix(len(node_ids)), (_DIM,))
    for slot, node in enumerate([0, 1, 2, 3, 4]):
        torch.testing.assert_close(graph.x[0, slot, :_DIM], expected[node])
    assert graph.x[0, :, _DIM].tolist() == [1.0] * 5  # has_f0
    torch.testing.assert_close(graph.x[0, :, _DIM + 1 : _DIM + 3], oracle.x[0, :, 0:2])
    torch.testing.assert_close(graph.x[0, :, _DIM + 3], oracle.x[0, :, 4])


def test_featureless_truth_node_gathers_zeros_with_indicator_down() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2")])
    # Node "2" exists only in the truth graph: no context row, no feature row.
    graph = _stitch(_generator(truth, ["0", "1"]), [0], [1])

    assert graph.num_nodes == 3
    assert torch.count_nonzero(graph.x[0, 2, :_DIM]) == 0
    assert graph.x[0, 2, _DIM].item() == 0.0  # has_f0
    assert graph.x[0, 2, _DIM + 3].item() == 1.0  # exists


def test_feature_table_may_cover_non_context_truth_nodes() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2")])
    features = _feature_matrix(3)
    graph = _stitch(
        _generator(truth, ["0", "1"], feature_ids=["0", "1", "2"], features=features),
        [0],
        [1],
    )

    expected = torch.nn.functional.layer_norm(features, (_DIM,))
    torch.testing.assert_close(graph.x[0, 2, :_DIM], expected[2])
    assert graph.x[0, 2, _DIM].item() == 1.0


def test_swapped_exchanges_only_root_channels_and_shares_structure() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2"), ("1", "3")])
    graph = _stitch(_generator(truth, ["0", "1", "2", "3"]), [0], [1])

    swapped = graph.swapped()

    width = _DIM + 4
    order = list(range(width))
    order[width - 3], order[width - 2] = order[width - 2], order[width - 3]
    assert torch.equal(swapped.x, graph.x[..., order])
    assert swapped.adj is graph.adj
    assert swapped.mask is graph.mask
    assert swapped.aux is graph.aux
    assert swapped.directed is False
    assert torch.equal(swapped.swapped().x, graph.x)


def test_teacher_view_stash_reproduces_the_structural_oracle() -> None:
    truth = nx.Graph(_TRUTH_EDGES)
    node_ids = [str(node) for node in range(6)]
    generator = _generator(truth, node_ids)
    oracle = _stitch(_oracle_reference(truth, node_ids), [0, 4], [1, 5])

    default_graph = _stitch(generator, [0, 4], [1, 5])
    assert "teacher_x" not in default_graph.aux
    assert "teacher_adj" not in default_graph.aux

    generator.set_stash_teacher_view(True)
    stashed = _stitch(generator, [0, 4], [1, 5])
    torch.testing.assert_close(stashed.aux["teacher_x"], oracle.x)
    torch.testing.assert_close(stashed.aux["teacher_adj"], oracle.adj)
    teacher_graph = FullEgoGraph(
        x=stashed.aux["teacher_x"],
        adj=stashed.aux["teacher_adj"],
        mask=stashed.mask,
        aux={"plan": stashed.aux["plan"], "log_plan": stashed.aux["log_plan"]},
        directed=True,
    )
    torch.testing.assert_close(teacher_graph.swapped().x, oracle.swapped().x)


def test_feature_binding_is_fail_closed() -> None:
    truth = nx.Graph([("0", "1")])
    generator = FullEgoFeaturesGenerator(input_dim=_DIM)

    with pytest.raises(RuntimeError, match="context is not installed"):
        generator.set_node_features(_feature_matrix(2), ["0", "1"])

    generator.set_oracle_context(truth, ["0", "1"])
    with pytest.raises(ValueError, match="must be"):
        generator.set_node_features(torch.zeros(2, _DIM + 1), ["0", "1"])
    with pytest.raises(ValueError, match="duplicates"):
        generator.set_node_features(_feature_matrix(2), ["0", "0"])
    with pytest.raises(ValueError, match="empty"):
        generator.set_node_features(torch.zeros(0, _DIM), [])
    with pytest.raises(ValueError, match="row count"):
        generator.set_node_features(_feature_matrix(3), ["0", "1"])

    with pytest.raises(RuntimeError, match="features are not installed"):
        _stitch(generator, [0], [1])

    generator.set_node_features(_feature_matrix(2), ["0", "1"])
    _stitch(generator, [0], [1])
    # Re-installing the context invalidates the bound features.
    generator.set_oracle_context(truth, ["0", "1"])
    with pytest.raises(RuntimeError, match="features are not installed"):
        _stitch(generator, [0], [1])


def test_zero_parameters_dims_and_empty_plans() -> None:
    truth = nx.Graph([("0", "1")])
    generator = _generator(truth, ["0", "1"])
    graph = _stitch(generator, [0], [1])

    assert list(generator.parameters()) == []
    assert generator.state_dict() == {}
    assert generator.graph_dims() == (_DIM + 4, 1)
    assert generator.auxiliary_losses(graph, {}) == {}
    assert graph.aux["plan"].shape == (1, 0, 0)
    assert graph.aux["log_plan"].shape == (1, 0, 0)
    with pytest.raises(ValueError, match="must be positive"):
        FullEgoFeaturesGenerator(input_dim=0)


def test_grit_over_edgeless_graph_is_finite_and_pooled_ignores_batch_padding() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2"), ("4", "6"), ("4", "7"), ("5", "8")])
    truth.add_node("3")
    node_ids = [str(node) for node in range(9)]
    generator = _generator(truth, node_ids)
    small = _stitch(generator, [0], [1])
    padded_batch = _stitch(generator, [0, 4], [1, 5])
    assert small.num_nodes == 3
    assert padded_batch.num_nodes == 5

    torch.manual_seed(7)
    encoder = GritGmtEncoder(
        in_dim=_DIM + 4,
        num_relations=1,
        d_model=8,
        dim=8,
        layers=1,
        w_rel=0.0,
        rrwp_k=3,
        n_heads=2,
        seeds=2,
    )
    encoder.eval()
    embedding = encoder(small)
    loss = embedding.pooled.square().sum()
    loss.backward()

    assert torch.isfinite(embedding.tokens).all()
    assert torch.isfinite(embedding.pooled).all()
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    with torch.no_grad():
        padded_embedding = encoder(padded_batch)
    torch.testing.assert_close(
        embedding.pooled[0], padded_embedding.pooled[0], atol=1e-6, rtol=1e-6
    )
