from __future__ import annotations

import networkx as nx
import pytest
import torch
from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder
from src.model.egostitch.generator.egostitch import GeneratorNodeState
from src.model.egostitch.generator.full_oracle import FullEgoGraph, FullOracleGenerator


def _generator(graph: nx.Graph[str], node_ids: list[str] | None = None) -> FullOracleGenerator:
    generator = FullOracleGenerator()
    generator.set_oracle_context(graph, list(graph.nodes) if node_ids is None else node_ids)
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


def _edge_set(graph: FullEgoGraph, batch_row: int) -> set[tuple[int, int]]:
    count = int(graph.mask[batch_row].sum().item())
    matrix = graph.adj[batch_row, 0, :count, :count]
    return {
        (left, right)
        for left in range(count)
        for right in range(left + 1, count)
        if bool(matrix[left, right])
    }


def _reference_query(
    truth: nx.Graph[str], src: str, dst: str
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    src_neighbors = set(truth.neighbors(src)) - {dst}
    dst_neighbors = set(truth.neighbors(dst)) - {src}
    endpoints = [src] if src == dst else [src, dst]
    nodes = endpoints + sorted((src_neighbors | dst_neighbors) - set(endpoints))
    index = {node_id: offset for offset, node_id in enumerate(nodes)}
    x = torch.zeros((len(nodes), 5))
    x[:, 4] = 1.0
    x[index[src], 0] = 1.0
    x[index[dst], 1] = 1.0
    for node_id in src_neighbors:
        x[index[node_id], 2] = 1.0
    for node_id in dst_neighbors:
        x[index[node_id], 3] = 1.0
    adj = torch.zeros((len(nodes), len(nodes)))
    for left, right in truth.subgraph(nodes).edges:
        if {left, right} == {src, dst}:
            continue
        left_i, right_i = index[left], index[right]
        adj[left_i, right_i] = 1.0
        adj[right_i, left_i] = 1.0
    return nodes, x, adj


def test_emits_exact_induced_edges_after_removing_query_edge() -> None:
    truth = nx.Graph(
        [
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("1", "2"),
            ("1", "4"),
            ("2", "3"),
            ("3", "4"),
            ("4", "5"),
        ]
    )
    graph = _stitch(_generator(truth, [str(node) for node in range(6)]), [0], [1])

    # Deterministic order is endpoints [0, 1], then sorted union [2, 3, 4].
    assert graph.mask.tolist() == [[True, True, True, True, True]]
    assert _edge_set(graph, 0) == {(0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)}
    assert graph.adj[0, 0, 0, 1].item() == 0.0
    # Node 5 is two hops from the endpoints and is outside U_q.
    assert graph.num_nodes == 5


def test_retains_every_neighbor_above_sixteen() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(str(node) for node in range(22))
    truth.add_edge("0", "1")
    truth.add_edges_from(("0", str(neighbor)) for neighbor in range(2, 22))

    graph = _stitch(_generator(truth, [str(node) for node in range(22)]), [0], [1])

    assert graph.mask.sum().item() == 22
    assert graph.num_nodes == 22
    assert graph.x[0, 2:, 2].sum().item() == 20.0


def test_shared_neighbor_is_deduplicated_and_has_both_roles() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2")])
    # The neighbor has no F0 row but must remain visible in the full truth graph.
    graph = _stitch(_generator(truth, ["0", "1"]), [0], [1])

    assert graph.num_nodes == 3
    assert graph.x[0, 2].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_batch_padding_uses_only_current_batch_max_and_masks_padding() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(str(node) for node in range(8))
    truth.add_edges_from([("0", "2"), ("0", "3"), ("1", "4"), ("5", "7")])
    generator = _generator(truth, [str(node) for node in range(8)])

    graph = _stitch(generator, [0, 5], [1, 6])

    assert graph.x.shape == (2, 5, 5)
    assert graph.mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    assert torch.count_nonzero(graph.x[1, 3:]).item() == 0
    assert torch.count_nonzero(graph.adj[1, :, 3:, :]).item() == 0


def test_batched_cpu_construction_matches_independent_reference() -> None:
    truth = nx.Graph(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "d"),
            ("b", "e"),
            ("c", "d"),
            ("d", "e"),
            ("f", "g"),
            ("f", "h"),
            ("g", "h"),
        ]
    )
    truth.add_nodes_from(["i", "j"])
    node_ids = ["j", "a", "f", "b", "i", "c", "d", "e", "g", "h"]
    pairs = [("a", "b"), ("f", "f"), ("i", "j"), ("c", "d")]
    row = {node_id: offset for offset, node_id in enumerate(node_ids)}

    graph = _stitch(
        _generator(truth, node_ids),
        [row[src] for src, _ in pairs],
        [row[dst] for _, dst in pairs],
    )

    references = [_reference_query(truth, src, dst) for src, dst in pairs]
    assert graph.x.shape == (len(pairs), max(len(nodes) for nodes, _, _ in references), 5)
    for batch_row, (nodes, expected_x, expected_adj) in enumerate(references):
        count = len(nodes)
        assert graph.mask[batch_row].tolist() == [True] * count + [False] * (
            graph.num_nodes - count
        )
        torch.testing.assert_close(graph.x[batch_row, :count], expected_x)
        torch.testing.assert_close(graph.adj[batch_row, 0, :count, :count], expected_adj)
        assert torch.count_nonzero(graph.x[batch_row, count:]) == 0
        assert torch.count_nonzero(graph.adj[batch_row, :, count:, :]) == 0


def test_stitch_does_not_extract_cuda_style_scalars_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = nx.Graph([("0", "2"), ("1", "2"), ("3", "4")])
    generator = _generator(truth, [str(node) for node in range(5)])
    state_a = _encode(generator, [0, 3, 2])
    state_b = _encode(generator, [1, 4, 2])
    is_self = torch.tensor([False, False, True])

    def fail_item(_tensor: torch.Tensor) -> object:
        raise AssertionError("stitch must resolve the batch without Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    graph = generator.stitch(state_a, state_b, is_self)

    assert graph.mask.tolist() == [
        [True, True, True],
        [True, True, False],
        [True, True, True],
    ]


def test_swapped_exchanges_roles_and_shares_structure() -> None:
    truth = nx.Graph([("0", "2"), ("1", "2"), ("1", "3")])
    graph = _stitch(_generator(truth, ["0", "1", "2", "3"]), [0], [1])

    swapped = graph.swapped()

    assert torch.equal(swapped.x, graph.x[..., [1, 0, 3, 2, 4]])
    assert swapped.adj is graph.adj
    assert swapped.mask is graph.mask
    assert swapped.aux is graph.aux
    assert swapped.directed is False
    assert torch.equal(swapped.swapped().x, graph.x)


def test_self_pair_emits_one_endpoint_and_complete_single_ego() -> None:
    truth = nx.Graph([("0", "1"), ("0", "2"), ("1", "2"), ("2", "3")])
    graph = _stitch(_generator(truth, ["0", "1", "2", "3"]), [0], [0])

    assert graph.num_nodes == 3
    assert graph.x[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert graph.x[0, 1:, 2:4].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert _edge_set(graph, 0) == {(0, 1), (0, 2), (1, 2)}


def test_context_and_row_failures_are_fail_closed() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(["0", "1"])
    generator = FullOracleGenerator()
    x = torch.zeros(1, 3)
    ground = torch.zeros(1, 1, 3)

    with pytest.raises(RuntimeError, match="context is not installed"):
        generator.encode_node(x, ground, node_rows=torch.tensor([0]))
    with pytest.raises(ValueError, match="duplicates"):
        generator.set_oracle_context(truth, ["0", "0"])
    with pytest.raises(ValueError, match="absent from the truth graph"):
        generator.set_oracle_context(truth, ["0", "missing"])

    generator.set_oracle_context(truth, ["0", "1"])
    with pytest.raises(ValueError, match="requires node_rows"):
        generator.encode_node(x, ground)
    with pytest.raises(ValueError, match="rows are missing"):
        generator.encode_node(x, ground, node_rows=torch.tensor([2]))


def test_rejects_non_loopless_or_non_undirected_context() -> None:
    looped = nx.Graph()
    looped.add_edge("0", "0")
    with pytest.raises(ValueError, match="loopless"):
        FullOracleGenerator().set_oracle_context(looped, ["0"])

    directed = nx.DiGraph()
    directed.add_nodes_from(["0", "1"])
    with pytest.raises(ValueError, match="undirected"):
        FullOracleGenerator().set_oracle_context(directed, ["0", "1"])


def test_zero_parameters_losses_dims_and_empty_plans() -> None:
    truth = nx.Graph()
    truth.add_nodes_from(["0", "1"])
    generator = _generator(truth, ["0", "1"])
    graph = _stitch(generator, [0], [1])

    assert list(generator.parameters()) == []
    assert generator.state_dict() == {}
    assert generator.graph_dims() == (5, 1)
    assert generator.auxiliary_losses(graph, {}) == {}
    assert graph.aux["plan"].shape == (1, 0, 0)
    assert graph.aux["log_plan"].shape == (1, 0, 0)


def test_grit_forward_backward_is_finite_and_pooled_output_ignores_batch_padding() -> None:
    truth = nx.Graph(
        [
            ("0", "2"),
            ("1", "2"),
            ("4", "6"),
            ("4", "7"),
            ("5", "8"),
        ]
    )
    truth.add_node("3")
    node_ids = [str(node) for node in range(9)]
    generator = _generator(truth, node_ids)
    small = _stitch(generator, [0], [1])
    padded_batch = _stitch(generator, [0, 4], [1, 5])
    assert small.num_nodes == 3
    assert padded_batch.num_nodes == 5

    torch.manual_seed(7)
    encoder = GritGmtEncoder(
        in_dim=5,
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
