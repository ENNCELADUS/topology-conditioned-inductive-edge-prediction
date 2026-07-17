"""Tests for src.experiments.probes: closed-form ridge representation probes."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from src.experiments import probes

pytestmark = pytest.mark.unit


def _linearly_predictable(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build states with a target that is an exact linear function of them plus tiny noise."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, d))
    weights = rng.normal(size=d)
    targets = states @ weights + rng.normal(scale=0.01, size=n)
    return states, targets


class TestLinearProbeR2:
    def test_near_one_on_linearly_predictable_target(self) -> None:
        states, targets = _linearly_predictable(300, 5, seed=0)
        r2 = probes.linear_probe_r2(states, targets)
        assert r2 == pytest.approx(1.0, abs=0.02)

    def test_near_zero_on_shuffled_target(self) -> None:
        states, targets = _linearly_predictable(300, 5, seed=0)
        rng = np.random.default_rng(1)
        shuffled = rng.permutation(targets)
        r2 = probes.linear_probe_r2(states, shuffled)
        assert abs(r2) < 0.2

    def test_deterministic_given_fixed_seed(self) -> None:
        states, targets = _linearly_predictable(120, 4, seed=2)
        first = probes.linear_probe_r2(states, targets)
        second = probes.linear_probe_r2(states, targets)
        assert first == second

    def test_accepts_one_dimensional_states(self) -> None:
        rng = np.random.default_rng(3)
        states = rng.normal(size=200)
        targets = states * 3.0 + rng.normal(scale=0.01, size=200)
        r2 = probes.linear_probe_r2(states, targets)
        assert r2 == pytest.approx(1.0, abs=0.02)

    def test_requires_at_least_n_folds_samples(self) -> None:
        states = np.zeros((2, 3))
        targets = np.zeros(2)
        with pytest.raises(ValueError, match="at least"):
            probes.linear_probe_r2(states, targets)


class TestDegreePartialledR2:
    def test_near_zero_when_target_is_degree(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = rng.normal(size=(n, 4))
        r2 = probes.degree_partialled_r2(states, degrees, degrees)
        assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_recovers_signal_beyond_degree(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = rng.normal(size=(n, 3))
        residual_signal = states[:, 0] * 2.0
        targets = degrees * 0.5 + residual_signal + rng.normal(scale=0.01, size=n)
        r2 = probes.degree_partialled_r2(states, targets, degrees)
        assert r2 == pytest.approx(1.0, abs=0.05)

    def test_zero_when_states_are_pure_degree_confound(self) -> None:
        # states carry ONLY the degree signal (up to noise): once both sides
        # are partialled against degree, nothing predictive should remain.
        rng = np.random.default_rng(6)
        n = 250
        degrees = rng.uniform(1.0, 50.0, size=n)
        states = (degrees * 2.0 + rng.normal(scale=0.01, size=n)).reshape(-1, 1)
        targets = degrees * 0.3 + rng.normal(scale=1.0, size=n)
        r2 = probes.degree_partialled_r2(states, targets, degrees)
        assert abs(r2) < 0.2


class TestProbeTargets:
    """Hand-verified spec Sec 13.6 targets on a triangle-plus-pendant graph."""

    @staticmethod
    def _graph() -> nx.Graph:
        # Triangle a-b-c plus pendant d attached to a.
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("a", "c"), ("b", "c"), ("a", "d")])
        return g

    def test_hand_computed_values(self) -> None:
        targets = probes.probe_targets(self._graph(), ["a", "b", "d"])
        # Degrees: a=3 (b, c, d), b=2, d=1.
        np.testing.assert_array_equal(targets["degree"], [3.0, 2.0, 1.0])
        # Clustering (networkx convention): a has 1 of C(3,2)=3 neighbor
        # pairs connected; b's single neighbor pair (a, c) is connected; d
        # has < 2 neighbors -> 0.
        np.testing.assert_allclose(targets["clustering"], [1 / 3, 1.0, 0.0])
        # ego(a) is the whole graph (4 edges); ego(b) is the triangle
        # (3 edges); ego(d) is the single edge a-d.
        np.testing.assert_array_equal(targets["ego_edges"], [4.0, 3.0, 1.0])
        # nx.density: 4 edges / C(4,2)=6; 3/3; 1/1.
        np.testing.assert_allclose(targets["ego_density"], [4 / 6, 1.0, 1.0])

    def test_row_alignment_follows_input_order(self) -> None:
        forward = probes.probe_targets(self._graph(), ["a", "d"])
        reversed_order = probes.probe_targets(self._graph(), ["d", "a"])
        np.testing.assert_array_equal(forward["degree"], reversed_order["degree"][::-1])

    def test_missing_node_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            probes.probe_targets(self._graph(), ["a", "zzz"])
