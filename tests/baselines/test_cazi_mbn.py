from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from src.baselines.cazi_mbn import CAZIStudent, CAZITeacher
from src.train_cazi_mbn import compute_ugt_projection


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
