"""Descriptor-level structural targets for the ``kd_struct`` auxiliary arm.

No teacher is involved. Each official row gets the five ego-graph descriptors
the representation audit (`src/experiments/kd_rep_audit.py`) found linearly
readable from the teacher's topology vector, computed straight from the truth
graph with the queried partner dropped from both neighbour sets: log1p common
neighbours, log1p degree sum, |log1p degree difference|, Jaccard, and log1p
Adamic-Adar. Training rows read the training-side graph (no V_val-internal
edge); V_val rows read the full train-side substrate, so the validation
counterpart scores recovery of the true structure. Targets are z-scored with
training-row statistics and packed as an in-memory `KDRowTargets` whose
``teacher_rep`` holds the descriptors, so the trainer's row-exact join and
staging are reused unchanged; ``teacher_logit`` is zero and never read.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.data.val_region import Pair, ValRegionSplit
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


def substrate_graph(split: ValRegionSplit, train_graph: nx.Graph) -> nx.Graph:
    """Loopless full train-side substrate (V_val-internal edges kept) over all train nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(split.train_nodes)
    graph.add_edges_from((u, v) for u, v in train_graph.edges() if u != v)
    return graph


def _index_pairs(
    node_index: dict[str, int], pairs: Sequence[Pair]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    a_idx = np.fromiter((node_index[a] for a, _ in pairs), dtype=np.int32, count=len(pairs))
    b_idx = np.fromiter((node_index[b] for _, b in pairs), dtype=np.int32, count=len(pairs))
    return a_idx, b_idx


def structural_row_targets(
    *,
    train_graph: nx.Graph,
    val_graph: nx.Graph,
    train_pairs: Sequence[Pair],
    train_labels: Sequence[int],
    val_pairs: Sequence[Pair],
    val_labels: Sequence[int],
) -> KDRowTargets:
    """Z-scored descriptor targets for the trainer's own rows, in row order.

    Raises:
        ValueError: If the two graphs do not share one node set.
    """
    node_ids = sorted(train_graph.nodes)
    if set(node_ids) != set(val_graph.nodes):
        raise ValueError("training and validation structural graphs must share one node set")
    node_index = {node: position for position, node in enumerate(node_ids)}
    a_idx, b_idx = _index_pairs(node_index, train_pairs)
    val_a_idx, val_b_idx = _index_pairs(node_index, val_pairs)
    train_raw = structural_targets(train_graph, node_ids, a_idx, b_idx)
    val_raw = structural_targets(val_graph, node_ids, val_a_idx, val_b_idx)
    mean = train_raw.mean(axis=0)
    std = np.maximum(train_raw.std(axis=0), _STD_FLOOR)
    return KDRowTargets(
        node_ids=node_ids,
        pair_a_idx=a_idx,
        pair_b_idx=b_idx,
        pair_label=np.asarray(train_labels, dtype=np.int8),
        teacher_logit=np.zeros(len(train_pairs), dtype=np.float32),
        teacher_rep=((train_raw - mean) / std).astype(np.float16),
        val_pair_a_idx=val_a_idx,
        val_pair_b_idx=val_b_idx,
        val_pair_label=np.asarray(val_labels, dtype=np.int8),
        val_teacher_logit=np.zeros(len(val_pairs), dtype=np.float32),
        val_teacher_rep=((val_raw - mean) / std).astype(np.float16),
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
    "substrate_graph",
]
