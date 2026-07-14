"""Tests for src.eval.ego_fidelity: Stage-1 imagined-ego-net diagnostics."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
import torch
from src.eval.ego_fidelity import (
    degree_calibration_curve,
    slot_adjacency_clustering_correlation,
    slot_recall_at_k,
)

pytestmark = pytest.mark.unit

_NODES = [f"n{i}" for i in range(6)]


def _toy_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(_NODES)
    # n0-n1-n2 triangle (clustering 1.0); n3-n4 edge; n5 isolated.
    g.add_edges_from([("n0", "n1"), ("n0", "n2"), ("n1", "n2"), ("n3", "n4")])
    return g


class TestSlotRecallAtK:
    def _proj(self, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return 10.0 * torch.randn(len(_NODES), 3, generator=gen)

    def test_perfect_slots_full_recall(self) -> None:
        proj = self._proj()
        index = {node: i for i, node in enumerate(_NODES)}
        g = _toy_graph()
        k = 3
        slot_h = torch.zeros(len(_NODES), k, 3)
        for node in _NODES:
            neighbors = sorted(v for v in g.neighbors(node) if v != node)[:k]
            for slot, v in enumerate(neighbors):
                slot_h[index[node], slot] = proj[index[v]]
        report = slot_recall_at_k(slot_h, g, _NODES, proj)
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.n_nodes == 5  # n5 has no neighbors
        assert report.tau > 0.0

    def test_far_slots_zero_recall(self) -> None:
        proj = self._proj(seed=1)
        slot_h = torch.full((len(_NODES), 3, 3), 1e6)
        report = slot_recall_at_k(slot_h, _toy_graph(), _NODES, proj)
        assert report.recall_at_k == 0.0
        assert report.n_neighbors > 0

    def test_rejects_misaligned_rows(self) -> None:
        with pytest.raises(ValueError, match="row-aligned"):
            slot_recall_at_k(torch.zeros(3, 2, 3), _toy_graph(), _NODES, torch.zeros(6, 3))

    def test_deterministic(self) -> None:
        proj = self._proj(seed=2)
        slot_h = torch.randn(len(_NODES), 3, 3, generator=torch.Generator().manual_seed(3))
        a = slot_recall_at_k(slot_h, _toy_graph(), _NODES, proj, seed=7)
        b = slot_recall_at_k(slot_h.clone(), _toy_graph(), _NODES, proj.clone(), seed=7)
        assert a == b


class TestDegreeCalibrationCurve:
    def test_perfectly_calibrated(self) -> None:
        d_hat = np.linspace(1.0, 10.0, 100)
        rows = degree_calibration_curve(d_hat, d_hat.copy(), n_bins=5)
        assert len(rows) == 5
        for row in rows:
            assert row["mean_expected"] == pytest.approx(row["mean_realized"])
        assert sum(int(row["n"]) for row in rows) == 100

    def test_systematic_overprediction_visible(self) -> None:
        d_hat = np.linspace(1.0, 10.0, 50)
        realized = d_hat / 2.0
        rows = degree_calibration_curve(d_hat, realized, n_bins=4)
        for row in rows:
            assert row["mean_expected"] > row["mean_realized"]

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError, match="equal-shaped"):
            degree_calibration_curve(np.ones(3), np.ones(4))
        with pytest.raises(ValueError, match="non-empty"):
            degree_calibration_curve(np.empty(0), np.empty(0))


class TestSlotAdjacencyClusteringCorrelation:
    def _inputs(
        self, generated_mass: list[float]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        n, k = len(generated_mass), 2
        # With two slots and pi = 1, the normalized off-diagonal mass equals
        # adj[0, 1] exactly, so generated_mass is directly controllable.
        adj = torch.zeros(n, k, k)
        for i, mass in enumerate(generated_mass):
            adj[i, 0, 1] = adj[i, 1, 0] = mass
        pi = torch.ones(n, k)
        return adj, pi, {}

    def test_perfect_correlation(self) -> None:
        clustering = {"n0": 0.1, "n1": 0.5, "n2": 0.9, "n3": 0.3}
        adj, pi, _ = self._inputs([0.1, 0.5, 0.9, 0.3])
        out = slot_adjacency_clustering_correlation(adj, pi, ["n0", "n1", "n2", "n3"], clustering)
        assert out["pearson"] == pytest.approx(1.0)
        assert out["spearman"] == pytest.approx(1.0)
        assert out["n"] == 4.0

    def test_anticorrelated(self) -> None:
        clustering = {"n0": 0.1, "n1": 0.5, "n2": 0.9}
        adj, pi, _ = self._inputs([0.9, 0.5, 0.1])
        out = slot_adjacency_clustering_correlation(adj, pi, ["n0", "n1", "n2"], clustering)
        assert out["pearson"] == pytest.approx(-1.0)

    def test_constant_side_yields_nan(self) -> None:
        clustering = {"n0": 0.5, "n1": 0.5}
        adj, pi, _ = self._inputs([0.2, 0.8])
        out = slot_adjacency_clustering_correlation(adj, pi, ["n0", "n1"], clustering)
        assert math.isnan(out["pearson"])

    def test_nodes_missing_from_mapping_skipped(self) -> None:
        clustering = {"n0": 0.1}
        adj, pi, _ = self._inputs([0.1, 0.9])
        out = slot_adjacency_clustering_correlation(adj, pi, ["n0", "n_unknown"], clustering)
        assert out["n"] == 1.0
        assert math.isnan(out["pearson"])
