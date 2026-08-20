"""Features-only full-ego candidate sets for the Gate A set-student diagnostic.

`FullEgoFeaturesGenerator` emits the *same node set* as `FullOracleGenerator`
-- both endpoints and every true one-hop neighbor in the installed truth graph,
queried edge removed -- but strips every edge from what the encoder may see:
``adj`` is exactly zero and the structural role channels (u-neighbor /
v-neighbor, which are endpoint-candidate edges) are withheld. What remains per
node is its content feature row plus root identity, so the downstream encoder
is a pure relational *set* student over the oracle's candidate nodes.

The structural view the oracle would have emitted (5-channel role features plus
the truth adjacency) is optionally stashed in ``aux`` -- generator-private by
contract (`graph.py`), read only by the KD trainer to run the frozen teacher
encoder on the identical node layout, never by any encoder. Diagnostic-only,
exactly like the parent: scoring pairs through this generator consumes truth
*membership* for the queried pairs and stays behind the same run-kind and
``--allow-oracle-diagnostic`` fences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from .generator import FullEgoGraph, FullOracleGenerator

__all__ = ["FeatureEgoGraph", "FullEgoFeaturesGenerator"]


@dataclass(frozen=True)
class FeatureEgoGraph(FullEgoGraph):
    """One padded batch of edgeless, feature-carrying oracle candidate sets.

    ``x`` is ``[f0 | has_f0 | root_u | root_v | exists]`` (width ``D + 4``):
    the row-layernormed content features, a featureless-candidate indicator
    (truth-graph nodes without an F0 row are legitimate -- CLAUDE.md's
    ``exclude_nodes`` trap -- and gather zeros), and the two root tags plus
    node existence. ``adj`` is exactly zero: candidate membership is input,
    edges are not.
    """

    def swapped(self) -> FeatureEgoGraph:
        """Exchange the two root-tag channels while sharing everything else."""
        width = int(self.x.shape[-1])
        order = torch.arange(width, device=self.x.device)
        order[width - 3] = width - 2
        order[width - 2] = width - 3
        return FeatureEgoGraph(
            x=self.x.index_select(-1, order),
            adj=self.adj,
            mask=self.mask,
            aux=self.aux,
            directed=not self.directed,
        )


class FullEgoFeaturesGenerator(FullOracleGenerator):
    """Emit the oracle candidate node set with content features and no edges."""

    def __init__(self, *, input_dim: int) -> None:
        """Pin the frozen content-feature width the emitted ``x`` will carry.

        Args:
            input_dim: F0 mean-pool feature width ``d`` (spec Sec 9.2),
                threaded from `EgoStitchConfig.input_dim` by the registry so
                `graph_dims` is known before any context is installed.

        Raises:
            ValueError: If `input_dim` is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self._input_dim = input_dim
        self._graph_row_to_feature_row: NDArray[np.int64] | None = None
        self._feature_matrix: torch.Tensor | None = None
        self._features_device: torch.Tensor | None = None
        self._stash_teacher_view = False

    def set_oracle_context(self, graph: nx.Graph[str], node_ids: Sequence[str]) -> None:
        """Bind the truth graph and invalidate any previously bound features."""
        super().set_oracle_context(graph, node_ids)
        self._graph_row_to_feature_row = None
        self._feature_matrix = None
        self._features_device = None

    def set_node_features(self, features: torch.Tensor, node_ids: Sequence[str]) -> None:
        """Bind row-layernormed content features for truth-graph nodes by id.

        Keyed by node id rather than by context row because the feature
        universe and the queryable-context universe genuinely differ: at
        scoring time candidate egos routinely reach truth-graph nodes outside
        the scored pair universe, and featureless truth-graph nodes exist in
        every universe. Nodes absent from `node_ids` gather zeros with
        ``has_f0 = 0``.

        Args:
            features: Shape ``(M, input_dim)`` raw F0 rows; row-layernormed
                here once, in fp32, so per-stitch work is a pure gather.
            node_ids: Length-``M`` unique node ids aligned with `features`.

        Raises:
            RuntimeError: If no truth context is installed yet.
            ValueError: On a shape/width mismatch or duplicate ids.
        """
        if self._graph is None:
            raise RuntimeError("full oracle context is not installed")
        if features.ndim != 2 or int(features.shape[1]) != self._input_dim:
            raise ValueError(
                f"features must be (M, {self._input_dim}), got {tuple(features.shape)}"
            )
        if int(features.shape[0]) == 0:
            raise ValueError(
                "feature table is empty: no truth-graph node has features, which "
                "means the wrong feature store or truth graph is bound"
            )
        ids = tuple(node_ids)
        if len(ids) != int(features.shape[0]):
            raise ValueError("features and node_ids disagree on row count")
        if len(set(ids)) != len(ids):
            raise ValueError("feature node_ids contain duplicates")
        graph_row = {node_id: row for row, node_id in enumerate(sorted(self._graph.nodes))}
        mapping = np.full(len(graph_row), -1, dtype=np.int64)
        for row, node_id in enumerate(ids):
            graph_index = graph_row.get(node_id)
            if graph_index is not None:
                mapping[graph_index] = row
        self._graph_row_to_feature_row = mapping
        self._feature_matrix = F.layer_norm(features.detach().float(), (self._input_dim,))
        self._features_device = None

    def set_stash_teacher_view(self, stash: bool) -> None:
        """Toggle stashing the structural teacher view in ``aux``.

        Off by default: only the KD trainer reads ``teacher_x``/``teacher_adj``,
        and scoring must not pay the device transfer for tensors nobody reads.
        """
        self._stash_teacher_view = stash

    def graph_dims(self) -> tuple[int, int]:
        """Return the feature-plus-tag node channels and single (zero) relation."""
        return self._input_dim + 4, 1

    def _features_on(self, device: torch.device) -> torch.Tensor:
        """Return the bound feature matrix cached on `device`."""
        matrix = self._feature_matrix
        if matrix is None:
            raise RuntimeError("full ego features are not installed")
        cached = self._features_device
        if cached is None or cached.device != device:
            cached = matrix.to(device=device)
            self._features_device = cached
        return cached

    def _finalize_graph(
        self,
        *,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
        node_owners: NDArray[np.int64],
        node_locals: NDArray[np.int64],
        node_values: NDArray[np.int64],
    ) -> FullEgoGraph:
        """Swap the structural channels for gathered features; zero the edges."""
        if self._graph_row_to_feature_row is None:
            raise RuntimeError("full ego features are not installed")
        features = self._features_on(device)
        batch, max_nodes = mask.shape
        feature_rows = self._graph_row_to_feature_row[node_values]
        slot_rows = np.zeros((batch, max_nodes), dtype=np.int64)
        slot_has = np.zeros((batch, max_nodes), dtype=np.bool_)
        slot_rows[node_owners, node_locals] = np.maximum(feature_rows, 0)
        slot_has[node_owners, node_locals] = feature_rows >= 0
        rows = torch.from_numpy(slot_rows).to(device=device)
        has = torch.from_numpy(slot_has).to(device=device)
        has_channel = has.to(dtype=features.dtype).unsqueeze(-1)
        gathered = features[rows] * has_channel
        x_struct = x.to(device=device)
        student_x = torch.cat(
            (gathered, has_channel, x_struct[..., 0:2], x_struct[..., 4:5]), dim=-1
        )
        empty_plan = torch.empty((batch, 0, 0), device=device)
        aux: dict[str, torch.Tensor] = {"plan": empty_plan, "log_plan": empty_plan}
        if self._stash_teacher_view:
            aux["teacher_x"] = x_struct
            aux["teacher_adj"] = adj.to(device=device)
        return FeatureEgoGraph(
            x=student_x,
            adj=torch.zeros((batch, 1, max_nodes, max_nodes), device=device, dtype=features.dtype),
            mask=mask.to(device=device),
            aux=aux,
            directed=True,
        )
