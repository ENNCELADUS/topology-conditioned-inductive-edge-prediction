"""Stage-1 imagined-ego-net fidelity diagnostics (prereg non-binding set).

The three per-mechanism diagnostics the G5 Stage-1 pre-registration requires in
the gate report (docs/registrations/g5_stage1_preregistration.json; proposal
Sec 4.6 transmission evidence):

- `slot_recall_at_k`: do the generated slots land near true neighbors?
- `degree_calibration_curve`: does ``E[d_hat]`` track realized assembled degree?
- `slot_adjacency_clustering_correlation`: does the slot-slot adjacency carry
  true local clustering?

Pure tensor/graph math — no model forward happens here; callers supply cached
per-node outputs (the scorer's per-node pass or a training-side dump).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class SlotRecallReport:
    """Slot recall@K over one node set.

    Attributes:
        recall_at_k: Fraction of true neighbors (capped at K per node) whose
            Hungarian-matched slot lies within `tau` in projected space.
        tau: The distance threshold used (percentile-calibrated, disclosed).
        tau_percentile: The calibration percentile.
        n_nodes: Nodes with at least one in-set neighbor.
        n_neighbors: Total capped neighbor count evaluated.
    """

    recall_at_k: float
    tau: float
    tau_percentile: float
    n_nodes: int
    n_neighbors: int


def slot_recall_at_k(
    slot_h: torch.Tensor,
    g_eval: nx.Graph,
    node_ids: Sequence[str],
    proj: torch.Tensor,
    *,
    tau_percentile: float = 10.0,
    seed: int = 0,
    n_calibration_pairs: int = 10_000,
) -> SlotRecallReport:
    """Hungarian slot-recall of true neighbors in projected feature space.

    A neighbor counts as recovered when its Hungarian-matched slot's L2
    distance is at or below the `tau_percentile`-th percentile of random-pair
    projection distances over `node_ids` (a scale-free threshold, disclosed in
    the report). Neighbors outside `node_ids` (no projection available) are
    skipped.

    Args:
        slot_h: Shape ``(N, K, d_p)`` cached slot embeddings, row-aligned with
            `node_ids`.
        g_eval: The evaluation graph (train-side ``G_struct`` for the
            protocol-clean row; the test graph for the disclosed row).
        node_ids: Node ids aligned with `slot_h` / `proj` rows.
        proj: Shape ``(N, d_p)`` projected node features (same projection the
            decision head uses).
        tau_percentile: Threshold percentile of the random-pair distances.
        seed: Seed of the random calibration pairs.
        n_calibration_pairs: Number of random pairs for the threshold.

    Returns:
        The `SlotRecallReport`.

    Raises:
        ValueError: On shape mismatches or fewer than two nodes.
    """
    n = len(node_ids)
    if slot_h.shape[0] != n or proj.shape[0] != n:
        raise ValueError("slot_h and proj must be row-aligned with node_ids")
    if n < 2:
        raise ValueError("slot recall needs at least two nodes")
    index = {node: i for i, node in enumerate(node_ids)}
    k = slot_h.shape[1]

    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=n_calibration_pairs)
    right = rng.integers(0, n, size=n_calibration_pairs)
    keep = left != right
    distances = torch.linalg.vector_norm(
        proj[torch.from_numpy(left[keep])] - proj[torch.from_numpy(right[keep])], dim=-1
    )
    tau = float(np.percentile(distances.detach().cpu().numpy(), tau_percentile))

    recovered = 0
    total = 0
    n_nodes = 0
    for node in node_ids:
        if node not in g_eval:
            continue
        neighbors = [v for v in g_eval.neighbors(node) if v != node and v in index]
        if not neighbors:
            continue
        neighbors = sorted(neighbors)[:k]
        n_nodes += 1
        slots = slot_h[index[node]]
        targets = proj[torch.tensor([index[v] for v in neighbors], dtype=torch.long)]
        cost = torch.cdist(slots, targets).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            total += 1
            if cost[row, col] <= tau:
                recovered += 1
    return SlotRecallReport(
        recall_at_k=recovered / total if total else 0.0,
        tau=tau,
        tau_percentile=tau_percentile,
        n_nodes=n_nodes,
        n_neighbors=total,
    )


def degree_calibration_curve(
    d_hat: NDArray[np.float64],
    realized_degree: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Binned ``E[d_hat]`` vs realized assembled degree (proposal Sec 4.6 rows 1-2).

    Nodes are binned by `d_hat` quantiles; each row reports the bin's mean
    expected and mean realized degree (the direct transmission evidence for the
    degree-budget mechanism).

    Args:
        d_hat: Shape ``(N,)`` density-normalized degree budgets
            (``d_hat_raw * rho_hat_eval / rho_train`` at evaluation).
        realized_degree: Shape ``(N,)`` degrees of the same nodes in the
            assembled graph.
        n_bins: Quantile bin count.

    Returns:
        One row per non-empty bin:
        ``{"bin", "n", "mean_expected", "mean_realized"}``.

    Raises:
        ValueError: On shape mismatch or empty input.
    """
    if d_hat.shape != realized_degree.shape or d_hat.size == 0:
        raise ValueError("d_hat and realized_degree must be equal-shaped and non-empty")
    edges = np.quantile(d_hat, np.linspace(0.0, 1.0, n_bins + 1))
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        low, high = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (d_hat >= low) & (d_hat <= high)
        else:
            mask = (d_hat >= low) & (d_hat < high)
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "bin": float(b),
                "n": float(mask.sum()),
                "mean_expected": float(d_hat[mask].mean()),
                "mean_realized": float(realized_degree[mask].mean()),
            }
        )
    return rows


def slot_adjacency_clustering_correlation(
    adj: torch.Tensor,
    pi: torch.Tensor,
    node_ids: Sequence[str],
    clustering: Mapping[str, float],
) -> dict[str, float]:
    """Correlation of pi-weighted slot-adjacency mass with true local clustering.

    The per-node generated statistic is the pi-weighted mean off-diagonal
    slot-slot adjacency ``sum_{k<k'} adj·pi·pi / max(sum_{k<k'} pi·pi, eps)`` —
    a clustering-like score in [0, 1] — correlated against
    ``nx.clustering(G_struct)`` values (proposal Sec 4.6 clustering row).

    Args:
        adj: Shape ``(N, K, K)`` cached slot-slot adjacencies.
        pi: Shape ``(N, K)`` cached slot existence probabilities.
        node_ids: Node ids aligned with the tensor rows.
        clustering: Node id -> true local clustering coefficient (nodes absent
            from the mapping are skipped).

    Returns:
        ``{"pearson", "spearman", "n"}`` (correlations are ``nan`` when fewer
        than two nodes overlap or either side is constant).
    """
    generated: list[float] = []
    real: list[float] = []
    pair_pi = pi[:, :, None] * pi[:, None, :]
    weighted = adj * pair_pi
    for i, node in enumerate(node_ids):
        if node not in clustering:
            continue
        upper = torch.triu(weighted[i], diagonal=1).sum()
        norm = torch.clamp(torch.triu(pair_pi[i], diagonal=1).sum(), min=1e-8)
        generated.append(float(upper / norm))
        real.append(float(clustering[node]))
    if len(generated) < 2 or np.std(generated) == 0.0 or np.std(real) == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": float(len(generated))}
    pearson = float(pearsonr(generated, real).statistic)
    spearman = float(spearmanr(generated, real).statistic)
    return {"pearson": pearson, "spearman": spearman, "n": float(len(generated))}
