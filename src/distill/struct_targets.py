"""Training-only structural descriptors for the historical auxiliary arm.

Each training pair gets query-masked graph descriptors, standardized using
training-row statistics. Validation labels and structure are not KD targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.val_region import Pair
from src.distill.artifacts import KDRowTargets

STRUCT_NAMES = ("log1p_cn", "log1p_deg_sum", "log1p_deg_absdiff", "jaccard", "log1p_adamic_adar")
STRUCT_REP_SOURCE = "struct_descriptors"
_STD_FLOOR = 1e-6


def structural_targets(
    graph: nx.Graph, node_ids: list[str], a_idx: NDArray[np.int32], b_idx: NDArray[np.int32]
) -> NDArray[np.float64]:
    """Per-row ``(n, 5)`` structural descriptors with the queried partner masked out."""
    neigh = {node: set(graph.neighbors(node)) for node in node_ids}
    degree = {node: len(members) for node, members in neigh.items()}
    out = np.zeros((len(a_idx), len(STRUCT_NAMES)), dtype=np.float64)
    for row, (a, b) in enumerate(zip(a_idx.tolist(), b_idx.tolist(), strict=True)):
        u, v = node_ids[a], node_ids[b]
        n_u = neigh[u] - {v}
        n_v = neigh[v] - {u}
        common = n_u & n_v
        union = len(n_u | n_v)
        aa = sum(1.0 / np.log1p(degree[w]) for w in common if degree[w] > 0)
        out[row] = (
            np.log1p(len(common)),
            np.log1p(len(n_u)) + np.log1p(len(n_v)),
            abs(np.log1p(len(n_u)) - np.log1p(len(n_v))),
            len(common) / union if union else 0.0,
            np.log1p(aa),
        )
    return out


def _index_pairs(
    node_index: dict[str, int], pairs: Sequence[Pair]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    a_idx = np.fromiter((node_index[a] for a, _ in pairs), dtype=np.int32, count=len(pairs))
    b_idx = np.fromiter((node_index[b] for _, b in pairs), dtype=np.int32, count=len(pairs))
    return a_idx, b_idx


def structural_row_targets(
    *,
    train_graph: nx.Graph,
    train_pairs: Sequence[Pair],
    train_labels: Sequence[int],
) -> KDRowTargets:
    """Z-scored descriptor targets for the trainer's own rows, in row order."""
    node_ids = sorted(train_graph.nodes)
    node_index = {node: position for position, node in enumerate(node_ids)}
    a_idx, b_idx = _index_pairs(node_index, train_pairs)
    train_raw = structural_targets(train_graph, node_ids, a_idx, b_idx)
    mean = train_raw.mean(axis=0)
    std = np.maximum(train_raw.std(axis=0), _STD_FLOOR)
    return KDRowTargets(
        node_ids=node_ids,
        pair_a_idx=a_idx,
        pair_b_idx=b_idx,
        pair_label=np.asarray(train_labels, dtype=np.int8),
        teacher_logit=np.zeros(len(train_pairs), dtype=np.float32),
        teacher_rep=((train_raw - mean) / std).astype(np.float16),
        manifest={
            "rep_source": STRUCT_REP_SOURCE,
            "descriptors": list(STRUCT_NAMES),
            "train_mean": mean.tolist(),
            "train_std": std.tolist(),
        },
    )


__all__ = [
    "STRUCT_NAMES",
    "STRUCT_REP_SOURCE",
    "structural_row_targets",
    "structural_targets",
]
