from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import yaml
from src.baselines.cazi_mbn import CAZIStudent, CAZITeacher
from src.train_cazi_mbn import (
    PreparedData,
    _validate_data_provenance,
    compute_ugt_projection,
    load_config,
    load_or_build_ugt,
)


def test_teacher_and_student_contracts() -> None:
    torch.manual_seed(0)
    sequence = torch.randn(8, 10)
    topology = torch.randn(8, 4)
    positive = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    negative = torch.tensor([[0, 2, 4, 6, 7, 3], [2, 4, 6, 7, 3, 0]])
    teacher = CAZITeacher(8, 10, topology_dim=4, latent_dim=3, heads=2)
    discriminator, consensus = teacher.graph_objective(topology, positive, negative)
    assert discriminator.ndim == 0
    assert consensus.ndim == 0
    assert teacher.pair_logits(sequence, positive[0], positive[1]).shape == (6,)
    assert teacher.distilled_latent().shape == (8, 3)

    student = CAZIStudent(10, latent_dim=3)
    assert student.node_latent(sequence).shape == (8, 3)
    assert student.pair_logits(sequence, positive[0], positive[1]).shape == (6,)


def test_sparse_ugt_matches_released_dense_operator_subspace() -> None:
    nodes = [f"n{i}" for i in range(12)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    sparse_projection = compute_ugt_projection(
        nodes,
        edges,
        order=3,
        feature_length=4,
        seed=0,
    ).astype(np.float64)

    row: list[int] = []
    col: list[int] = []
    for i in range(len(nodes) - 1):
        j = i + 1
        row.extend((i, j))
        col.extend((j, i))
    adjacency = sp.coo_matrix(
        (np.ones(len(row)), (row, col)), shape=(len(nodes), len(nodes))
    ).toarray()
    degree = adjacency.sum(axis=1)
    normalized = np.diag(degree**-0.5) @ adjacency @ np.diag(degree**-0.5)
    dense_operator = sum(np.linalg.matrix_power(normalized, k) for k in range(1, 4))
    dense_u, dense_s, _ = np.linalg.svd(dense_operator, full_matrices=False)
    dense_projection = dense_u[:, :4] * dense_s[:4]
    dense_projection = (dense_projection - dense_projection.mean(axis=0)) / dense_projection.std(
        axis=0
    )
    np.testing.assert_allclose(
        sparse_projection @ sparse_projection.T,
        dense_projection @ dense_projection.T,
        atol=1e-4,
    )


def test_ugt_cache_rejects_pre_contract_node_only_payload(tmp_path: Path) -> None:
    path = tmp_path / "ugt_projection.npz"
    np.savez_compressed(
        path,
        node_ids=np.asarray(["a", "b", "c"]),
        projection=np.zeros((3, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="stale UGT cache"):
        load_or_build_ugt(
            path,
            ["a", "b", "c"],
            [("a", "b"), ("b", "c")],
            order=1,
            feature_length=1,
            seed=0,
        )


def test_ugt_cache_rejects_same_nodes_with_different_topology(tmp_path: Path) -> None:
    path = tmp_path / "ugt_projection.npz"
    nodes = [f"n{i}" for i in range(6)]
    load_or_build_ugt(
        path,
        nodes,
        [("n0", "n1"), ("n1", "n2"), ("n2", "n3")],
        order=1,
        feature_length=2,
        seed=0,
    )

    with pytest.raises(ValueError, match="topology mismatch"):
        load_or_build_ugt(
            path,
            nodes,
            [("n0", "n1"), ("n1", "n2"), ("n4", "n5")],
            order=1,
            feature_length=2,
            seed=0,
        )


def test_load_config_rejects_removed_partition_seed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((repo_root / "configs/cazi_mbn_breadth_first.yaml").read_text())
    payload["data"]["partition_seed"] = 0
    config_path = tmp_path / "cazi.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="data has unknown keys"):
        load_config(config_path)


def test_checkpoint_provenance_rejects_old_contract_payload() -> None:
    data = cast(
        PreparedData,
        SimpleNamespace(
            training_interactions_sha256="interactions",
            training_topology_sha256="topology",
        ),
    )

    with pytest.raises(ValueError, match="data provenance mismatch"):
        _validate_data_provenance({"state_dict": {}}, data, label="checkpoint")
