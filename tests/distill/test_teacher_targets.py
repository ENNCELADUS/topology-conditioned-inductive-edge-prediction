"""Contracts for `src.distill.teacher_targets` (full-row KD teacher dumper).

Covers row-universe legality (training-side-only refusal, V_val-internal
row quarantine, off-universe endpoint refusal), the query-edge masking
property the whole KD design leans on, shard/merge round-tripping, the
pre-check stats report, and the `kd_row_targets_v1` artifact round-trip.

The CLI (`main`) itself -- checkpoint I/O, `--row-shard` process sharding,
F0 caching -- is not covered by a synthetic end-to-end fixture here: it is
thin glue over the unit-tested pieces below (`assert_training_side_only`,
`encode_all_nodes`, `score_rows`, `_write_shard`/`_merge_shards`,
`write_kd_targets`) plus checkpoint/benchmark I/O that is exercised
operationally (`hpc/run.sh`), not economically reproduced as a pytest
fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from src.data.artifacts import canonical_pair
from src.data.features import FeatureStore
from src.data.val_region import Pair, ValRegionParams, ValRegionSplit, derive_val_region_split
from src.distill import teacher_targets as tt
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.generator.egostitch import GeneratorNodeState
from src.model.egostitch.generator.full_oracle import FullEgoGraph, FullOracleGenerator
from src.model.egostitch.generator.null import NullGenerator
from src.score_universe import _install_oracle_context, _shard_range

pytestmark = pytest.mark.unit

# EgoStitchModel's internal Stage-1 generator keeps its own pinned spec
# default (EgoStitchConfig().input_dim, spec Sec 13) regardless of which
# generator is selected, so the feature store backing these tests must use
# that exact node-token width (mirrors tests/test_score_universe.py's
# `_E2E_NODE_DIM`).
_NODE_DIM = 1536
_SEEDS = 4
# The KD dump requires the PMA (grit_gmt) encoder: its pooled embedding is
# the kd_gen latent target.
_TINY_FULL_ORACLE_CONFIG: dict[str, object] = {
    "generator": {
        "name": "full_ego_oracle",
        "feature_standardization": "row_layernorm",
    },
    "encoder": {"name": "grit_gmt", "dim": 16, "layers": 2, "seeds": _SEEDS},
    "classifier": {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 4,
        "n_inj": 1,
        "xattn_heads": 4,
    },
}


def _tiny_model() -> EgoStitchModel:
    torch.manual_seed(0)
    model = EgoStitchModel(E2EConfig.from_mapping(_TINY_FULL_ORACLE_CONFIG))
    model.eval()
    return model


def _write_feature_store(root: Path, node_tokens: dict[str, torch.Tensor]) -> None:
    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for node_id, tokens in node_tokens.items():
        rel_path = f"embeddings/{node_id}.pt"
        torch.save(tokens, root / rel_path)
        index[node_id] = rel_path
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "torch_pt_per_node", "input_dim": _NODE_DIM, "max_sequence_length": 1024}
        )
    )
    (root / "index.json").write_text(json.dumps(index))


# --------------------------------------------------------------------------- row-universe legality


def _grid_nodes(rows: int, cols: int) -> list[str]:
    return [f"n{r:02d}_{c:02d}" for r in range(rows) for c in range(cols)]


def _grid_edges(rows: int, cols: int) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            node = f"n{r:02d}_{c:02d}"
            if c + 1 < cols:
                edges.append((node, f"n{r:02d}_{c + 1:02d}"))
            if r + 1 < rows:
                edges.append((node, f"n{r + 1:02d}_{c:02d}"))
    return edges


def _toy_split() -> ValRegionSplit:
    """A tiny grid-graph split whose V_val region has >=2 nodes and cross-boundary edges."""
    nodes = _grid_nodes(5, 5)
    edges = _grid_edges(5, 5)
    global_positives = frozenset(canonical_pair(u, v) for u, v in edges)
    params = ValRegionParams(
        edge_fraction=0.15,
        n_regions=1,
        salt="teacher-targets-toy|",
        bucket_sizes=(2, 3),
        buckets_per_size=1,
    )
    split = derive_val_region_split(nodes, edges, [], global_positives, params=params)
    assert len(split.v_val) >= 2, "test fixture must produce a multi-node V_val region"
    return split


def test_assert_no_val_internal_training_rows_raises_for_a_v_val_internal_row() -> None:
    split = _toy_split()
    a, b = sorted(split.v_val)[:2]
    with pytest.raises(ValueError, match="V_val"):
        tt.assert_no_val_internal_training_rows([(a, b)], split.v_val)


def test_assert_no_val_internal_training_rows_passes_for_a_cross_boundary_row() -> None:
    split = _toy_split()
    v_val_node = next(iter(split.v_val))
    outside_node = next(iter(split.train_nodes - split.v_val))
    # Must not raise: only both-endpoints-inside-V_val rows are quarantined.
    tt.assert_no_val_internal_training_rows([(v_val_node, outside_node)], split.v_val)


def test_row_positions_maps_rows_to_node_positions() -> None:
    position = {"a": 0, "b": 1, "c": 2}
    rows: list[Pair] = [("a", "b"), ("b", "c")]
    a_idx, b_idx = tt._row_positions(rows, position)
    np.testing.assert_array_equal(a_idx, [0, 1])
    np.testing.assert_array_equal(b_idx, [1, 2])


def test_row_positions_raises_on_an_off_universe_endpoint() -> None:
    position = {"a": 0, "b": 1}
    rows: list[Pair] = [("a", "missing-node")]
    with pytest.raises(ValueError, match="outside the training node universe"):
        tt._row_positions(rows, position)


def test_refuses_a_truth_graph_node_outside_the_training_universe() -> None:
    split = _toy_split()
    node_ids = sorted(split.train_nodes)
    truth = tt.truth_graph_for_kd(split)
    truth.add_node("phantom-outside-training-universe")

    with pytest.raises(ValueError, match="outside the training universe"):
        tt.assert_training_side_only(node_ids, truth, split, frozenset())


def test_refuses_a_node_universe_overlapping_the_test_split() -> None:
    split = _toy_split()
    truth = tt.truth_graph_for_kd(split)
    node_ids = [*sorted(split.train_nodes), "held-out-test-node"]

    with pytest.raises(ValueError, match="test-split"):
        tt.assert_training_side_only(node_ids, truth, split, frozenset({"held-out-test-node"}))


def test_refuses_a_truth_graph_containing_a_v_val_internal_edge() -> None:
    split = _toy_split()
    node_ids = sorted(split.train_nodes)
    truth = tt.truth_graph_for_kd(split)
    a, b = sorted(split.v_val)[:2]
    truth.add_edge(a, b)

    with pytest.raises(ValueError, match="V_val-internal edge"):
        tt.assert_training_side_only(node_ids, truth, split, frozenset())


def test_accepts_a_clean_training_universe_that_legitimately_contains_v_val_nodes() -> None:
    split = _toy_split()
    node_ids = sorted(split.train_nodes)
    truth = tt.truth_graph_for_kd(split)

    assert split.v_val & set(node_ids), "V_val nodes must remain part of the training universe"
    tt.assert_training_side_only(node_ids, truth, split, frozenset())  # must not raise


def test_require_full_ego_oracle_refuses_a_null_generator() -> None:
    model = EgoStitchModel(E2EConfig.from_mapping({"generator": {"name": "null"}}))
    assert isinstance(model.generator, NullGenerator)
    with pytest.raises(ValueError, match="full_ego_oracle"):
        tt.require_full_ego_oracle(model)


def test_require_full_ego_oracle_refuses_the_features_subclass() -> None:
    model = EgoStitchModel(E2EConfig.from_mapping({"generator": {"name": "full_ego_features"}}))

    with pytest.raises(ValueError, match="full_ego_oracle"):
        tt.require_full_ego_oracle(model)


# --------------------------------------------------------------------------- query-edge masking


def _oracle_generator(graph: nx.Graph[str], node_ids: list[str]) -> FullOracleGenerator:
    generator = FullOracleGenerator()
    generator.set_oracle_context(graph, node_ids)
    return generator


def _oracle_encode(generator: FullOracleGenerator, rows: list[int]) -> GeneratorNodeState:
    batch = len(rows)
    return generator.encode_node(
        torch.zeros(batch, 3),
        torch.zeros(batch, 1, 3),
        node_rows=torch.tensor(rows, dtype=torch.int64),
    )


def _oracle_stitch(
    generator: FullOracleGenerator, rows_a: list[int], rows_b: list[int]
) -> FullEgoGraph:
    return generator.stitch(
        _oracle_encode(generator, rows_a),
        _oracle_encode(generator, rows_b),
        torch.tensor([a == b for a, b in zip(rows_a, rows_b, strict=True)]),
    )


def test_query_edge_is_invisible_in_its_own_structural_context() -> None:
    """The load-bearing KD-legality property.

    Per the module docstring: a positive training edge (u, v) is never
    visible in its own structural context.
    `tests/model/test_full_oracle_generator.py` already asserts the query
    edge is dropped from the adjacency matrix
    (`test_emits_exact_induced_edges_after_removing_query_edge`); this test
    adds the dump-level *identity* check the KD dumper's masking-safety
    claim actually depends on -- stitching (u, v) against a graph that
    still HAS the edge must be byte-identical to stitching it against a
    graph that never had the edge at all. A full `score_rows` scoring-path
    variant would need a real checkpoint to be meaningful; this
    structural, weight-independent identity check is what the masking
    guarantee actually rests on, so it suffices on its own.
    """
    node_ids = [str(n) for n in range(6)]
    edges_with_query = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
    truth_with_edge = nx.Graph([(str(a), str(b)) for a, b in edges_with_query])
    truth_with_edge.add_nodes_from(node_ids)

    generator = _oracle_generator(truth_with_edge, node_ids)
    graph = _oracle_stitch(generator, [0], [1])

    src_local, dst_local = 0, 1
    # Adjacency: no (src, dst) edge in either direction.
    assert graph.adj[0, 0, src_local, dst_local].item() == 0.0
    assert graph.adj[0, 0, dst_local, src_local].item() == 0.0
    # Neighbor channels: the partner endpoint is not flagged as a neighbor.
    assert graph.x[0, dst_local, 2].item() == 0.0  # dst not flagged as a src-neighbor
    assert graph.x[0, src_local, 3].item() == 0.0  # src not flagged as a dst-neighbor

    # Rebind to the truth graph with the query edge removed entirely: the
    # stitched query graph for (u, v) must be identical either way.
    truth_without_edge = truth_with_edge.copy()
    truth_without_edge.remove_edge("0", "1")
    generator_without_edge = _oracle_generator(truth_without_edge, node_ids)
    graph_without_edge = _oracle_stitch(generator_without_edge, [0], [1])

    torch.testing.assert_close(graph.x, graph_without_edge.x)
    torch.testing.assert_close(graph.adj, graph_without_edge.adj)
    torch.testing.assert_close(graph.mask, graph_without_edge.mask)


# --------------------------------------------------------------------------- score_rows


def test_score_rows_refuses_a_non_pma_encoder() -> None:
    config = dict(_TINY_FULL_ORACLE_CONFIG)
    config["encoder"] = {"name": "ste_typed", "dim": 16, "layers": 2}
    torch.manual_seed(0)
    model = EgoStitchModel(E2EConfig.from_mapping(config))
    model.eval()
    with pytest.raises(RuntimeError, match="grit_gmt"):
        tt.score_rows(
            model,
            {},
            ["a", "b"],
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            device=torch.device("cpu"),
        )


def test_score_rows_handles_an_empty_row_list() -> None:
    scored = tt.score_rows(
        _tiny_model(),
        {},
        ["a", "b"],
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        device=torch.device("cpu"),
    )
    assert scored.teacher_logit.shape == (0,)
    assert scored.teacher_rep.shape[0] == 0


def test_score_rows_is_invariant_to_the_batch_pairs_chunk_size(tmp_path: Path) -> None:
    """`score_rows` must reproduce the same per-row values however the row list is chunked.

    The trainer-side artifact contract depends on this (`verify_sample`
    itself leans on the same batch-invariance).
    """
    torch.manual_seed(0)
    nodes = [f"n{i}" for i in range(5)]
    node_tokens = {node: torch.randn(3 + i, _NODE_DIM) for i, node in enumerate(nodes)}
    features_root = tmp_path / "features"
    _write_feature_store(features_root, node_tokens)
    store = FeatureStore(features_root)

    truth = nx.Graph([("n0", "n1"), ("n1", "n2"), ("n2", "n3")])
    truth.add_nodes_from(nodes)

    model = _tiny_model()
    device = torch.device("cpu")
    _install_oracle_context(model, nodes, truth_graph=truth)
    node_cache = tt.encode_all_nodes(
        model, store, nodes, device=device, token_budget=4096, f0_cache=tmp_path / "f0.pt"
    )

    a_positions = np.array([0, 0, 1, 3], dtype=np.int32)
    b_positions = np.array([1, 2, 2, 3], dtype=np.int32)  # row 3 is a self-pair

    batched = tt.score_rows(
        model, node_cache, nodes, a_positions, b_positions, device=device, batch_pairs=3
    )
    single = tt.score_rows(
        model, node_cache, nodes, a_positions, b_positions, device=device, batch_pairs=1
    )

    n_rows = len(a_positions)
    assert batched.teacher_logit.shape == (n_rows,)
    assert batched.teacher_rep.shape[0] == n_rows
    np.testing.assert_allclose(batched.teacher_logit, single.teacher_logit, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        batched.teacher_rep.astype(np.float32), single.teacher_rep.astype(np.float32), atol=1e-3
    )


# --------------------------------------------------------------------------- shard/merge round-trip


def _synthetic_scored_rows(n_rows: int, *, rep_dim: int, rng: np.random.Generator) -> tt.ScoredRows:
    return tt.ScoredRows(
        teacher_logit=rng.standard_normal(n_rows).astype(np.float32),
        teacher_rep=rng.standard_normal((n_rows, rep_dim)).astype(np.float16),
    )


def test_write_shard_and_merge_shards_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_total, num_shards = 7, 3
    a_idx = np.arange(n_total, dtype=np.int32)
    b_idx = ((np.arange(n_total, dtype=np.int32) + 1) % n_total).astype(np.int32)
    full = _synthetic_scored_rows(n_total, rep_dim=4, rng=rng)

    output = tmp_path / "kd_targets"
    for shard in range(num_shards):
        start, end = _shard_range(n_total, shard, num_shards)
        shard_scored = tt.ScoredRows(
            teacher_logit=full.teacher_logit[start:end],
            teacher_rep=full.teacher_rep[start:end],
        )
        tt._write_shard(
            output,
            shard,
            num_shards,
            a_idx=a_idx[start:end],
            b_idx=b_idx[start:end],
            scored=shard_scored,
        )

    merged = tt._merge_shards(output, a_idx, b_idx, num_shards)
    np.testing.assert_array_equal(merged.teacher_logit, full.teacher_logit)
    np.testing.assert_array_equal(merged.teacher_rep, full.teacher_rep)


def test_merge_shards_raises_on_a_row_identity_mismatch(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_total, num_shards = 4, 2
    a_idx = np.arange(n_total, dtype=np.int32)
    b_idx = (np.arange(n_total, dtype=np.int32) + 10).astype(np.int32)

    output = tmp_path / "kd_targets"
    for shard in range(num_shards):
        start, end = _shard_range(n_total, shard, num_shards)
        shard_scored = _synthetic_scored_rows(end - start, rep_dim=2, rng=rng)
        write_a_idx = a_idx[start:end].copy()
        if shard == 1:
            write_a_idx[0] = 999  # corrupt row identity in the second shard
        tt._write_shard(
            output,
            shard,
            num_shards,
            a_idx=write_a_idx,
            b_idx=b_idx[start:end],
            scored=shard_scored,
        )

    with pytest.raises(ValueError, match="row identity"):
        tt._merge_shards(output, a_idx, b_idx, num_shards)


# --------------------------------------------------------------------------- stats report


def test_build_stats_report_shape_and_keys() -> None:
    train_logit = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    train_label = np.array([1, 0, 1], dtype=np.int8)
    val_logit = np.array([0.2], dtype=np.float32)
    val_label = np.array([0], dtype=np.int8)

    report = tt.build_stats_report(train_logit, train_label, val_logit, val_label)

    assert set(report) == {"train", "val"}
    for block_name, expected_n_rows in (("train", 3), ("val", 1)):
        block = report[block_name]
        assert isinstance(block, dict)
        assert block["n_rows"] == expected_n_rows
        assert "label_prevalence" in block
        for hist_key in (
            "teacher_logit_overall",
            "teacher_logit_positives",
            "teacher_logit_negatives",
            "sigmoid_entropy",
        ):
            histogram = block[hist_key]
            assert isinstance(histogram, dict)
            assert set(histogram) == {"count", "mean", "std", "p10", "p50", "p90"}


def test_build_stats_report_histograms_are_empty_safe() -> None:
    train_logit = np.array([1.0, -1.0], dtype=np.float32)
    train_label = np.array([0, 0], dtype=np.int8)  # no positives at all
    val_logit = np.array([], dtype=np.float32)
    val_label = np.array([], dtype=np.int8)

    report = tt.build_stats_report(train_logit, train_label, val_logit, val_label)

    train_block = report["train"]
    assert isinstance(train_block, dict)
    assert train_block["teacher_logit_positives"] == {
        "count": 0,
        "mean": 0.0,
        "std": 0.0,
        "p10": 0.0,
        "p50": 0.0,
        "p90": 0.0,
    }

    val_block = report["val"]
    assert isinstance(val_block, dict)
    assert val_block["n_rows"] == 0
    assert val_block["label_prevalence"] == 0.0
    val_overall = val_block["teacher_logit_overall"]
    assert isinstance(val_overall, dict)
    assert val_overall["count"] == 0


# --------------------------------------------------------------------------- artifact round-trip


def _write_toy_artifact(path: Path) -> None:
    rng = np.random.default_rng(0)
    write_kd_targets(
        path,
        node_ids=["a", "b", "c"],
        pair_a_idx=np.array([0, 0, 1], dtype=np.int32),
        pair_b_idx=np.array([1, 2, 2], dtype=np.int32),
        pair_label=np.array([1, 0, 1], dtype=np.int8),
        teacher_logit=np.array([0.5, -0.3, 1.2], dtype=np.float32),
        teacher_rep=rng.standard_normal((3, 4)).astype(np.float16),
        val_pair_a_idx=np.array([0], dtype=np.int32),
        val_pair_b_idx=np.array([2], dtype=np.int32),
        val_pair_label=np.array([0], dtype=np.int8),
        val_teacher_logit=np.array([0.1], dtype=np.float32),
        val_teacher_rep=rng.standard_normal((1, 4)).astype(np.float16),
        truth_graph_sha256="deadbeef",
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_sha256="cafebabe",
        checkpoint_id="abc123",
    )


def test_load_kd_targets_has_no_seeds_surface(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_toy_artifact(artifact_dir)

    loaded = load_kd_targets(artifact_dir)
    assert loaded.node_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(loaded.pair_a_idx, [0, 0, 1])
    np.testing.assert_array_equal(loaded.pair_b_idx, [1, 2, 2])
    np.testing.assert_array_equal(loaded.pair_label, [1, 0, 1])
    np.testing.assert_array_equal(
        loaded.teacher_logit, np.array([0.5, -0.3, 1.2], dtype=np.float32)
    )
    assert loaded.teacher_rep.shape == (3, 4)
    np.testing.assert_array_equal(loaded.val_pair_a_idx, [0])
    np.testing.assert_array_equal(loaded.val_pair_b_idx, [2])
    np.testing.assert_array_equal(loaded.val_pair_label, [0])
    np.testing.assert_array_equal(loaded.val_teacher_logit, np.array([0.1], dtype=np.float32))
    assert loaded.val_teacher_rep.shape == (1, 4)
    assert loaded.manifest["format"] == "kd_row_targets_v1"
    assert loaded.manifest["truth_source"] == "training_structure"
    assert loaded.manifest["checkpoint_id"] == "abc123"
    assert not hasattr(loaded, "teacher_seeds")
    assert not hasattr(loaded, "val_teacher_seeds")
    for key in ("seed_count", "seed_dim", "teacher_seeds_dtype", "seed_symmetry"):
        assert key not in loaded.manifest


def test_load_rejects_a_missing_npz(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_toy_artifact(artifact_dir)
    (artifact_dir / "targets.npz").unlink()

    with pytest.raises(ValueError, match="missing"):
        load_kd_targets(artifact_dir)


def test_load_rejects_an_npz_carrying_a_legacy_anchor_offsets_array(tmp_path: Path) -> None:
    """Pin the no-obsolete-artifact-path requirement.

    A v2-era sampled-context array (`anchor_offsets`, deleted with the
    anchor-context stream) must not silently pass the exactly-these-arrays
    check, even alongside a complete and otherwise-valid v1 array set.
    """
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    rng = np.random.default_rng(2)
    np.savez(
        artifact_dir / "targets.npz",
        pair_a_idx=np.array([0], dtype=np.int32),
        pair_b_idx=np.array([1], dtype=np.int32),
        pair_label=np.array([1], dtype=np.int8),
        teacher_logit=np.array([0.1], dtype=np.float32),
        teacher_rep=rng.standard_normal((1, 4)).astype(np.float16),
        val_pair_a_idx=np.array([0], dtype=np.int32),
        val_pair_b_idx=np.array([1], dtype=np.int32),
        val_pair_label=np.array([0], dtype=np.int8),
        val_teacher_logit=np.array([0.1], dtype=np.float32),
        val_teacher_rep=rng.standard_normal((1, 4)).astype(np.float16),
        anchor_offsets=np.array([0, 1], dtype=np.int64),  # obsolete v2 array
    )
    (artifact_dir / "manifest.json").write_text(json.dumps({"format": "kd_row_targets_v1"}))
    (artifact_dir / "node_ids.json").write_text(json.dumps(["a", "b"]))

    with pytest.raises(ValueError, match="arrays must be exactly"):
        load_kd_targets(artifact_dir)


# --------------------------------------------------------------------------- shard spec parsing


def test_parse_shard_accepts_a_valid_spec() -> None:
    assert tt._parse_shard("1/4") == (1, 4)


@pytest.mark.parametrize("spec", ["4/4", "-1/4", "not-a-shard", "1/0"])
def test_parse_shard_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="row-shard"):
        tt._parse_shard(spec)
