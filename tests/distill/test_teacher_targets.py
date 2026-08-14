"""Contracts for `src.distill.teacher_targets`.

Covers the scoring core against direct recomputation, training-side legality
refusal (test-split leakage, off-universe nodes, V_val-internal truth edges),
and pair-label correctness.
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
from src.data.val_region import ValRegionParams, ValRegionSplit, derive_val_region_split
from src.distill import teacher_targets as tt
from src.distill.artifacts import load_kd_targets, write_kd_targets
from src.distill.context_sampler import ContextSets, sample_context_sets
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.generator.null import NullGenerator
from src.score_universe import _install_oracle_context

pytestmark = pytest.mark.unit

# EgoStitchModel's internal Stage-1 generator keeps its own pinned spec default
# (EgoStitchConfig().input_dim, spec Sec 13) regardless of which generator is
# selected, so the feature store backing these tests must use that exact
# node-token width (mirrors tests/test_score_universe.py's `_E2E_NODE_DIM`).
_NODE_DIM = 1536
_TINY_FULL_ORACLE_CONFIG: dict[str, object] = {
    "generator": {
        "name": "full_ego_oracle",
        "feature_standardization": "row_layernorm",
    },
    "encoder": {"dim": 16, "layers": 2},
    "classifier": {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 4,
        "n_inj": 1,
        "xattn_heads": 4,
    },
}


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


def _tiny_model() -> EgoStitchModel:
    torch.manual_seed(0)
    model = EgoStitchModel(E2EConfig.from_mapping(_TINY_FULL_ORACLE_CONFIG))
    model.eval()
    return model


# --------------------------------------------------------------------------- scoring core


def test_scoring_core_matches_direct_recomputation_on_a_tiny_fixture(tmp_path: Path) -> None:
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

    context = sample_context_sets(truth, nodes, seed=0, k_near=2, k_rand=2)
    assert len(context.anchor_idx) > 0
    teacher_logit, pooled_ab, pooled_ba = tt.score_kd_context(
        model, node_cache, nodes, context, device=device, batch_pairs=3
    )
    assert pooled_ab.shape[0] == len(context.anchor_idx)
    assert pooled_ba.shape == pooled_ab.shape

    # Recompute every row one pair at a time via the same
    # build_pair_context_from_states/score_pair_context path -- the ground
    # truth `score_kd_context`'s batched loop must reproduce exactly.
    for row in range(len(context.anchor_idx)):
        node_a = nodes[int(context.anchor_idx[row])]
        node_b = nodes[int(context.partner_idx[row])]
        state_a = tt._move_state(tt._stack_states([node_cache[node_a]]), device)
        state_b = tt._move_state(tt._stack_states([node_cache[node_b]]), device)
        is_self = torch.tensor([node_a == node_b], dtype=torch.bool)
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            pair_context = model.build_pair_context_from_states(state_a, state_b, is_self)
            expected_logit = model.score_pair_context(pair_context)
        assert teacher_logit[row] == pytest.approx(float(expected_logit.item()), abs=1e-5)
        assert pair_context.pooled_ab is not None
        np.testing.assert_allclose(
            pooled_ab[row],
            pair_context.pooled_ab[0].detach().to(dtype=torch.float16).numpy(),
        )


def test_scoring_core_handles_an_empty_context() -> None:
    empty = ContextSets(
        anchor_idx=np.array([], dtype=np.int32),
        partner_idx=np.array([], dtype=np.int32),
        anchor_offsets=np.array([0, 0, 0], dtype=np.int64),
        is_near=np.array([], dtype=np.uint8),
        k_near=1,
        k_rand=1,
        seed=0,
    )
    teacher_logit, pooled_ab, pooled_ba = tt.score_kd_context(
        _tiny_model(), {}, ["a", "b"], empty, device=torch.device("cpu")
    )
    assert teacher_logit.shape == (0,)
    assert pooled_ab.shape[0] == 0
    assert pooled_ba.shape[0] == 0


def test_require_full_ego_oracle_refuses_a_null_generator() -> None:
    model = EgoStitchModel(E2EConfig.from_mapping({"generator": {"name": "null"}}))
    assert isinstance(model.generator, NullGenerator)
    with pytest.raises(ValueError, match="full_ego_oracle"):
        tt.require_full_ego_oracle(model)


# --------------------------------------------------------------------------- artifact roundtrip


def _write_toy_artifact(path: Path) -> None:
    rng = np.random.default_rng(0)
    write_kd_targets(
        path,
        node_ids=["a", "b", "c"],
        pair_anchor_idx=np.array([0, 0, 1], dtype=np.int32),
        pair_partner_idx=np.array([1, 2, 2], dtype=np.int32),
        anchor_offsets=np.array([0, 2, 3, 3], dtype=np.int64),
        teacher_logit=np.array([0.5, -0.3, 1.2], dtype=np.float32),
        teacher_pooled_ab=rng.standard_normal((3, 4)).astype(np.float16),
        teacher_pooled_ba=rng.standard_normal((3, 4)).astype(np.float16),
        is_near=np.array([1, 0, 1], dtype=np.uint8),
        pair_label=np.array([1, 0, 0], dtype=np.int8),
        truth_graph_sha256="deadbeef",
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_sha256="cafebabe",
        checkpoint_id="abc123",
        k_near=2,
        k_rand=2,
        seed=0,
    )


def test_write_load_roundtrip(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_toy_artifact(artifact_dir)

    loaded = load_kd_targets(artifact_dir)
    assert loaded.node_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(loaded.pair_anchor_idx, [0, 0, 1])
    np.testing.assert_array_equal(loaded.pair_partner_idx, [1, 2, 2])
    np.testing.assert_array_equal(loaded.anchor_offsets, [0, 2, 3, 3])
    np.testing.assert_array_equal(loaded.pair_label, [1, 0, 0])
    assert loaded.manifest["truth_source"] == "training_structure"
    assert loaded.manifest["format"] == "kd_targets_v2"
    assert loaded.manifest["checkpoint_id"] == "abc123"


def test_load_rejects_a_missing_npz(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_toy_artifact(artifact_dir)
    (artifact_dir / "targets.npz").unlink()

    with pytest.raises(ValueError, match="missing"):
        load_kd_targets(artifact_dir)


def test_write_rejects_mismatched_array_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same pair count"):
        write_kd_targets(
            tmp_path / "bad",
            node_ids=["a", "b"],
            pair_anchor_idx=np.array([0], dtype=np.int32),
            pair_partner_idx=np.array([1, 0], dtype=np.int32),
            anchor_offsets=np.array([0, 1, 1], dtype=np.int64),
            teacher_logit=np.array([0.1], dtype=np.float32),
            teacher_pooled_ab=np.zeros((1, 2), dtype=np.float16),
            teacher_pooled_ba=np.zeros((1, 2), dtype=np.float16),
            is_near=np.array([1], dtype=np.uint8),
            pair_label=np.array([1], dtype=np.int8),
            truth_graph_sha256="x",
            checkpoint_path=Path("c.pt"),
            checkpoint_sha256="y",
            checkpoint_id=None,
            k_near=1,
            k_rand=1,
            seed=0,
        )


# --------------------------------------------------------------------------- training-side legality


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


# --------------------------------------------------------------------------- pair labels


def test_pair_labels_match_truth_graph_edges() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c")])
    node_ids = ["a", "b", "c"]
    context = ContextSets(
        anchor_idx=np.array([0, 0, 1], dtype=np.int32),
        partner_idx=np.array([1, 2, 2], dtype=np.int32),
        anchor_offsets=np.array([0, 2, 3, 3], dtype=np.int64),
        is_near=np.array([1, 1, 1], dtype=np.uint8),
        k_near=2,
        k_rand=0,
        seed=0,
    )

    labels = tt.pair_labels(node_ids, context, graph)

    np.testing.assert_array_equal(labels, [1, 0, 1])  # a-b edge, a-c not, b-c edge


# --------------------------------------------------------------------------- shard parsing


def test_parse_shard_accepts_a_valid_spec() -> None:
    assert tt._parse_shard("1/4") == (1, 4)


@pytest.mark.parametrize("spec", ["4/4", "-1/4", "not-a-shard", "1/0"])
def test_parse_shard_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="anchor-shard"):
        tt._parse_shard(spec)
