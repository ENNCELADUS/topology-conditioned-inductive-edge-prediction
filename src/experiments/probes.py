"""Closed-form ridge-regression representation probes (spec Sec 14.3(3)).

Diagnostics-only linear probes over frozen encoder states (STE token states,
in the registered representation-probe protocol): out-of-fold ``R^2`` from a
closed-form ridge fit, implemented in plain numpy (no sklearn dependency, per
the pinned protocol). ``degree_partialled_r2`` additionally residualizes both
``states`` and ``targets`` against node degree before probing, isolating any
predictive signal beyond what a trivial degree confound would already give
away for free (registration ``diagnostics_nonbinding``: "linear probes on
frozen STE token states: R2 to degree / ego-density / clustering (ridge
lambda 1e-3, 5-fold), plus degree-partialled variants and Pi-consistency").

Both probes are diagnostics only — they never gate the Stage-1 verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

_RIDGE_LAMBDA = 1e-3
_N_FOLDS = 5
_MIN_VARIANCE = 1e-10


def probe_targets(graph: nx.Graph, nodes: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Per-node probe regression targets via the spec Sec 13.6 evaluator convention.

    Computes the pinned ego-stat quantities with the NetworkX implementations
    (the same evaluator convention the Stage-1 ego-stat targets bind to):
    ``degree = deg(u)``, ``clustering = nx.clustering(G, u)``,
    ``ego_edges = |E(ego(u))|`` and ``ego_density = nx.density(ego(u))`` with
    ``ego(u) = G.subgraph(N(u) | {u})`` on the simple graph. Callers pass the
    message-partition structural graph (``G_struct``, spec Sec 9.3) so probe
    targets never touch the target test graph.

    Args:
        graph: The simple structural graph (no self-loops).
        nodes: Probe node ids, in the row order of the probe states.

    Returns:
        ``{"degree", "clustering", "ego_edges", "ego_density"}`` -> shape
        ``(len(nodes),)`` float64 arrays, row-aligned with `nodes`.

    Raises:
        ValueError: If any probe node is missing from `graph`.
    """
    missing = [node for node in nodes if node not in graph]
    if missing:
        raise ValueError(f"probe nodes missing from the graph: {missing[:5]}")
    clustering = nx.clustering(graph, nodes)
    degree = np.array([float(graph.degree(node)) for node in nodes], dtype=np.float64)
    clustering_arr = np.array([float(clustering[node]) for node in nodes], dtype=np.float64)
    ego_edges = np.empty(len(nodes), dtype=np.float64)
    ego_density = np.empty(len(nodes), dtype=np.float64)
    for i, node in enumerate(nodes):
        ego = graph.subgraph(set(graph.neighbors(node)) | {node})
        ego_edges[i] = float(ego.number_of_edges())
        ego_density[i] = float(nx.density(ego))
    return {
        "degree": degree,
        "clustering": clustering_arr,
        "ego_edges": ego_edges,
        "ego_density": ego_density,
    }


def _as_2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reshape a 1-D feature array to a single-column 2-D array (no-op if already 2-D)."""
    reshaped: NDArray[np.float64] = values.reshape(-1, 1) if values.ndim == 1 else values
    return reshaped


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[NDArray[np.int64]]:
    """Partition ``range(n)`` into `n_folds` shuffled, near-equal folds."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [fold.astype(np.int64) for fold in np.array_split(order, n_folds)]


def _ridge_fit_predict(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Closed-form (mean-centered) ridge regression: fit on train, predict on test.

    ``w = (Xc^T Xc + lam I)^-1 Xc^T yc`` on centered train data; the intercept
    is recovered by re-adding the train means at predict time.
    """
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean()
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered.T @ x_centered + lam * np.eye(x_centered.shape[1])
    weights = np.linalg.solve(gram, x_centered.T @ y_centered)
    predictions: NDArray[np.float64] = (x_test - x_mean) @ weights + y_mean
    return predictions


def linear_probe_r2(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    lam: float = _RIDGE_LAMBDA,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
) -> float:
    """Cross-validated closed-form ridge probe ``R^2``.

    Args:
        states: Shape ``(n,)`` or ``(n, d)`` frozen representation features.
        targets: Shape ``(n,)`` regression targets.
        lam: Ridge penalty (spec-pinned ``1e-3``).
        n_folds: Number of cross-validation folds (spec-pinned ``5``).
        seed: Fold-assignment seed (deterministic given fixed inputs).

    Returns:
        The out-of-fold ``R^2``. Returns ``0.0`` (rather than an undefined
        ``0/0``) when the held-out target has ~zero variance.

    Raises:
        ValueError: If there are fewer samples than `n_folds`.
    """
    states64 = _as_2d(np.asarray(states, dtype=np.float64))
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    n = states64.shape[0]
    if n < n_folds:
        raise ValueError(f"linear_probe_r2 requires at least {n_folds} samples, got {n}")
    folds = _kfold_indices(n, n_folds, seed)
    predictions = np.empty(n, dtype=np.float64)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        predictions[test_idx] = _ridge_fit_predict(
            states64[train_idx], targets64[train_idx], states64[test_idx], lam
        )
    residual = targets64 - predictions
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((targets64 - targets64.mean()) ** 2))
    if ss_tot < _MIN_VARIANCE:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _partial_out(values: NDArray[np.float64], degrees: NDArray[np.float64]) -> NDArray[np.float64]:
    """OLS-residualize `values` (1-D or 2-D, one row per sample) against `degrees`."""
    design = np.column_stack([np.ones_like(degrees), degrees])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted: NDArray[np.float64] = design @ coefficients
    residual: NDArray[np.float64] = values - fitted
    return residual


def degree_partialled_r2(
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    degrees: NDArray[np.float64],
    *,
    lam: float = _RIDGE_LAMBDA,
    n_folds: int = _N_FOLDS,
    seed: int = 0,
) -> float:
    """Degree-partialled probe ``R^2``: residualize `states`/`targets` against `degrees` first.

    Args:
        states: Shape ``(n,)`` or ``(n, d)`` frozen representation features.
        targets: Shape ``(n,)`` regression targets.
        degrees: Shape ``(n,)`` node degrees — the confound to partial out.
        lam: Ridge penalty (spec-pinned ``1e-3``).
        n_folds: Number of cross-validation folds (spec-pinned ``5``).
        seed: Fold-assignment seed (deterministic given fixed inputs).

    Returns:
        The `linear_probe_r2` of the degree-residualized target from the
        degree-residualized states — signal beyond a pure degree confound.
        ``0.0`` when the degree-residualized target has ~zero variance (e.g.
        the target IS degree).
    """
    degrees64 = np.asarray(degrees, dtype=np.float64).reshape(-1)
    states64 = _as_2d(np.asarray(states, dtype=np.float64))
    targets64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    target_residual = _partial_out(targets64, degrees64)
    state_residual = _partial_out(states64, degrees64)
    return linear_probe_r2(state_residual, target_residual, lam=lam, n_folds=n_folds, seed=seed)


__all__ = ["degree_partialled_r2", "linear_probe_r2", "probe_targets"]
