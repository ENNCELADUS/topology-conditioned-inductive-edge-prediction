"""Decomposition arithmetic behind the topology-prompt causal attribution."""

from __future__ import annotations

import numpy as np
from src.experiments.coord_prompt_mediation import decompose


def _pairs(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Every unordered pair of ``num_nodes`` nodes, self-pairs included."""
    u, v = np.triu_indices(num_nodes)
    return u.astype(np.int64), v.astype(np.int64)


def test_decomposition_is_exact_and_recovers_a_purely_additive_effect() -> None:
    u, v = _pairs(8)
    rng = np.random.default_rng(0)
    node_effect = rng.normal(size=8)
    delta = 1.5 + node_effect[u] + node_effect[v]

    part = decompose(u, v, delta, 8)

    np.testing.assert_allclose(part.offset + part.node + part.pair, delta, atol=1e-8)
    # A self-pair loads its node twice, which the design matrix must reproduce.
    assert part.additive_r2 is not None
    assert part.additive_r2 > 1.0 - 1e-8
    np.testing.assert_allclose(part.pair, np.zeros_like(delta), atol=1e-6)
    np.testing.assert_allclose(part.node.mean(), 0.0, atol=1e-8)


def test_pair_specific_effect_survives_the_node_fit() -> None:
    u, v = _pairs(8)
    rng = np.random.default_rng(1)
    delta = rng.normal(size=len(u))

    part = decompose(u, v, delta, 8)

    np.testing.assert_allclose(part.offset + part.node + part.pair, delta, atol=1e-8)
    assert part.additive_r2 is not None
    assert 0.0 < part.additive_r2 < 1.0
    assert part.pair.std() > 0.0
    # Orthogonality: the residual carries nothing the node fit could have taken.
    assert abs(float(np.dot(part.node, part.pair))) < 1e-6 * len(delta)


def test_a_constant_effect_reports_an_undefined_r2_instead_of_dividing_by_zero() -> None:
    """An offset-only (or no-effect) intervention has no variance to explain."""
    u, v = _pairs(6)
    delta = np.full(len(u), -2.5)

    part = decompose(u, v, delta, 6)

    assert part.additive_r2 is None
    assert part.offset == -2.5
    np.testing.assert_allclose(part.node, np.zeros_like(delta), atol=1e-8)
    np.testing.assert_allclose(part.offset + part.node + part.pair, delta, atol=1e-8)
