"""Fixed-semantics structural coordinates of a queried pair.

The topology-prompt arm (``model.family: v3_1_topo_prompt``) reads, for every
queried pair ``(u, v)``, a short vector of structural quantities measured on the
universe's true simple graph **with the queried edge removed**. The vector has
four fields with fixed meaning, so a generator trained later to predict it from
endpoint attributes has verifiable targets rather than a free latent:

- ``endpoint_u`` / ``endpoint_v`` (9 each): the local role of one endpoint --
  ``log1p`` degree, local clustering, ``log1p`` closed and open two-hop counts,
  ``log1p`` two-hop reach, ``log1p`` mean neighbour degree, and the normalised
  walk returns ``(S^k)_xx`` for ``k = 2, 3, 4`` with ``S = D^-1/2 A D^-1/2``.
- ``relation`` (11): ``log1p`` common neighbours, Jaccard, ``log1p`` simple L3
  path count, the walk kernels ``(S^k)_uv`` for ``k = 2..5``, and a one-hot
  shortest-path category (2, 3, >= 4, disconnected).
- ``context`` (5): ``log1p`` one-hop union size, ``log1p`` size of the
  two-hop shell ``{x : min(d(u,x), d(v,x)) = 2}``, the fraction of that shell
  at distance two from both roots, ``log1p`` cross-ring links (distance one from
  one root and two from the other), and the L3 link density ``L3 / (d_u d_v)``.

Swapping ``u`` and ``v`` swaps the two endpoint fields and leaves the relation
and context fields unchanged. Self-pairs use the same formulas literally
(``u == v``): no edge is removed, Jaccard is one, and every distance category is
zero. The queried edge is removed **before** anything is measured, so a
positive pair never sees itself; with the edge gone and the graph loopless,
``(A^3)_uv`` counts exactly the simple ``u-a-b-v`` paths.

Non-edges are read from dense all-pairs products; edges are recomputed exactly
on the edge-deleted graph (`StructCoordinateTable._exact_pair`), which is also
the reference implementation the tests compare the dense path against.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path

COORD_SPEC = "v1"
ENDPOINT_NAMES: tuple[str, ...] = (
    "log1p_degree",
    "clustering",
    "log1p_triangles",
    "log1p_open_wedges",
    "log1p_two_hop",
    "log1p_mean_neighbor_degree",
    "return_2",
    "return_3",
    "return_4",
)
RELATION_NAMES: tuple[str, ...] = (
    "log1p_common",
    "jaccard",
    "log1p_l3",
    "walk_2",
    "walk_3",
    "walk_4",
    "walk_5",
    "dist_2",
    "dist_3",
    "dist_4plus",
    "dist_inf",
)
CONTEXT_NAMES: tuple[str, ...] = (
    "log1p_hop1_union",
    "log1p_shell",
    "shell_shared_fraction",
    "log1p_cross_ring",
    "l3_density",
)
FIELDS: tuple[str, ...] = ("endpoint_u", "endpoint_v", "relation", "context")
ENDPOINT_DIM = len(ENDPOINT_NAMES)
RELATION_DIM = len(RELATION_NAMES)
CONTEXT_DIM = len(CONTEXT_NAMES)
COORD_DIM = 2 * ENDPOINT_DIM + RELATION_DIM + CONTEXT_DIM
FIELD_SLICES: dict[str, slice] = {
    "endpoint_u": slice(0, ENDPOINT_DIM),
    "endpoint_v": slice(ENDPOINT_DIM, 2 * ENDPOINT_DIM),
    "relation": slice(2 * ENDPOINT_DIM, 2 * ENDPOINT_DIM + RELATION_DIM),
    "context": slice(2 * ENDPOINT_DIM + RELATION_DIM, COORD_DIM),
}
COORD_NAMES: tuple[str, ...] = (
    *(f"u_{name}" for name in ENDPOINT_NAMES),
    *(f"v_{name}" for name in ENDPOINT_NAMES),
    *RELATION_NAMES,
    *CONTEXT_NAMES,
)
_MAX_WALK = 5


def _log1p(values: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.log1p(np.asarray(values, dtype=np.float64))


def _safe_divide(
    numerator: NDArray[np.floating], denominator: NDArray[np.floating]
) -> NDArray[np.float64]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _distance_one_hot(dist: NDArray[np.floating]) -> NDArray[np.float64]:
    """``(n, 4)`` one-hot over {2, 3, >= 4, disconnected}; zero rows for distances 0/1."""
    dist = np.asarray(dist, dtype=np.float64)
    finite = np.isfinite(dist)
    return np.stack(
        [
            (dist == 2.0).astype(np.float64),
            (dist == 3.0).astype(np.float64),
            (finite & (dist >= 4.0)).astype(np.float64),
            (~finite).astype(np.float64),
        ],
        axis=1,
    )


class StructCoordinateTable:
    """Dense all-pairs structural products of one loopless simple graph.

    Build once per universe graph, then call `coords` for any pair list over
    that graph's nodes. Memory is ``O(n^2)`` (about ten ``n x n`` float32
    matrices); the largest universe in this project has 7,203 nodes.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Precompute the dense products.

        Args:
            graph: A simple, loopless `networkx.Graph`; isolated nodes are kept.

        Raises:
            ValueError: If the graph has a self-loop or no nodes.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("struct coordinates need a non-empty graph")
        if nx.number_of_selfloops(graph) > 0:
            raise ValueError("struct coordinates need a loopless graph; strip self-loops first")
        self.nodes: tuple[str, ...] = tuple(sorted(str(node) for node in graph.nodes))
        self.index: dict[str, int] = {node: i for i, node in enumerate(self.nodes)}
        n = len(self.nodes)
        rows: list[int] = []
        cols: list[int] = []
        for node_u, node_v in graph.edges:
            i, j = self.index[str(node_u)], self.index[str(node_v)]
            rows += [i, j]
            cols += [j, i]
        adjacency = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n), dtype=np.float32
        )
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0
        self.adjacency: sp.csr_matrix = adjacency
        self.degree: NDArray[np.float64] = (
            np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64)
        )
        self._inv_sqrt_degree = _safe_divide(np.ones(n), np.sqrt(self.degree))
        scale = sp.diags(self._inv_sqrt_degree.astype(np.float32))
        normalised = (scale @ adjacency @ scale).tocsr()

        # Walk kernels S^2..S^5 and the raw products A^2 (common neighbours) and
        # A^3 (length-3 walks) as dense float32 matrices.
        power = (normalised @ normalised).toarray().astype(np.float32)
        self._walks: list[NDArray[np.float32]] = [power]
        for _ in range(3, _MAX_WALK + 1):
            power = np.asarray(power @ normalised, dtype=np.float32)
            self._walks.append(power)
        common = (adjacency @ adjacency).toarray().astype(np.float32)
        self._common: NDArray[np.float32] = common
        self._l3: NDArray[np.float32] = np.asarray(common @ adjacency, dtype=np.float32)

        distances = shortest_path(adjacency, directed=False, unweighted=True)
        self._distance: NDArray[np.float32] = np.asarray(distances, dtype=np.float32)
        two_hop_mask = (self._distance == 2.0).astype(np.float32)
        self._two_hop_count: NDArray[np.float64] = two_hop_mask.sum(axis=1).astype(np.float64)
        self._shell_shared: NDArray[np.float32] = np.asarray(
            two_hop_mask @ two_hop_mask.T, dtype=np.float32
        )
        self._cross_ring: NDArray[np.float32] = np.asarray(
            two_hop_mask @ adjacency, dtype=np.float32
        )
        del two_hop_mask

        self._triangles: NDArray[np.float64] = np.diagonal(self._l3).astype(np.float64) / 2.0
        neighbour_degree_sum = np.asarray(adjacency @ self.degree.astype(np.float32)).reshape(-1)
        self._mean_neighbour_degree = _safe_divide(neighbour_degree_sum, self.degree)
        self._returns: NDArray[np.float64] = np.stack(
            [np.diagonal(self._walks[k]).astype(np.float64) for k in range(3)], axis=1
        )

    # ------------------------------------------------------------------ helpers

    @property
    def size(self) -> int:
        """Node count of the universe graph."""
        return len(self.nodes)

    def is_edge(self, u: int, v: int) -> bool:
        """Whether ``(u, v)`` is an edge of the universe graph."""
        return bool(self.adjacency[u, v] != 0)

    def _endpoint_block(
        self,
        degree: NDArray[np.float64],
        triangles: NDArray[np.float64],
        two_hop: NDArray[np.float64],
        mean_neighbour_degree: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        open_wedges = degree * (degree - 1.0) / 2.0 - triangles
        clustering = _safe_divide(2.0 * triangles, degree * (degree - 1.0))
        return np.stack(
            [
                _log1p(degree),
                clustering,
                _log1p(triangles),
                _log1p(np.maximum(open_wedges, 0.0)),
                _log1p(two_hop),
                _log1p(mean_neighbour_degree),
                returns[:, 0],
                returns[:, 1],
                returns[:, 2],
            ],
            axis=1,
        )

    # ------------------------------------------------------------------ dense path

    def _bulk(self, u: NDArray[np.int64], v: NDArray[np.int64]) -> NDArray[np.float64]:
        """Coordinates of non-edge (or self) pairs straight from the dense products."""
        degree_u = self.degree[u]
        degree_v = self.degree[v]
        common = self._common[u, v].astype(np.float64)
        union = degree_u + degree_v - common
        l3 = self._l3[u, v].astype(np.float64)
        distance = self._distance[u, v].astype(np.float64)
        shared = self._shell_shared[u, v].astype(np.float64)
        cross = self._cross_ring[u, v].astype(np.float64) + self._cross_ring[v, u].astype(
            np.float64
        )
        shell = (
            self._two_hop_count[u]
            + self._two_hop_count[v]
            - shared
            - cross
            - 2.0 * (distance == 2.0)
        )
        endpoint_u = self._endpoint_block(
            degree_u,
            self._triangles[u],
            self._two_hop_count[u],
            self._mean_neighbour_degree[u],
            self._returns[u],
        )
        endpoint_v = self._endpoint_block(
            degree_v,
            self._triangles[v],
            self._two_hop_count[v],
            self._mean_neighbour_degree[v],
            self._returns[v],
        )
        relation = np.concatenate(
            [
                np.stack(
                    [
                        _log1p(common),
                        _safe_divide(common, union),
                        _log1p(l3),
                        *(self._walks[k][u, v].astype(np.float64) for k in range(4)),
                    ],
                    axis=1,
                ),
                _distance_one_hot(distance),
            ],
            axis=1,
        )
        context = np.stack(
            [
                _log1p(union),
                _log1p(np.maximum(shell, 0.0)),
                _safe_divide(shared, shell),
                _log1p(cross),
                _safe_divide(l3, degree_u * degree_v),
            ],
            axis=1,
        )
        return np.asarray(
            np.concatenate([endpoint_u, endpoint_v, relation, context], axis=1), dtype=np.float64
        )

    # ------------------------------------------------------------------ exact path

    def _exact_pair(self, u: int, v: int) -> NDArray[np.float64]:
        """Coordinates of one pair measured on the graph with the edge ``(u, v)`` removed.

        Also correct for non-edges (nothing is removed), which is how the tests
        pin the dense path to this reference.
        """
        n = self.size
        adjacency = self.adjacency
        removed = u != v and self.is_edge(u, v)
        if removed:
            adjacency = adjacency.copy()
            adjacency[u, v] = 0.0
            adjacency[v, u] = 0.0
            adjacency.eliminate_zeros()
        degree = self.degree.copy()
        if removed:
            degree[u] -= 1.0
            degree[v] -= 1.0
        inv_sqrt = _safe_divide(np.ones(n), np.sqrt(degree))

        def walk_step(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            propagated = np.asarray(adjacency @ (inv_sqrt * vector)).reshape(-1)
            return np.asarray(inv_sqrt * propagated, dtype=np.float64)

        def adjacency_step(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(adjacency @ vector).reshape(-1)

        distances = shortest_path(adjacency, directed=False, unweighted=True, indices=[u, v])
        dist_u = np.asarray(distances[0], dtype=np.float64)
        dist_v = np.asarray(distances[1], dtype=np.float64)

        def endpoint(x: int, dist_x: NDArray[np.float64]) -> NDArray[np.float64]:
            unit = np.zeros(n)
            unit[x] = 1.0
            a1 = adjacency_step(unit)
            a3 = adjacency_step(adjacency_step(a1))
            triangles = a3[x] / 2.0
            s_k = walk_step(unit)
            returns = []
            for _ in range(3):
                s_k = walk_step(s_k)
                returns.append(s_k[x])
            neighbours = np.nonzero(a1)[0]
            mean_neighbour_degree = float(degree[neighbours].mean()) if len(neighbours) else 0.0
            block = self._endpoint_block(
                np.asarray([degree[x]]),
                np.asarray([triangles]),
                np.asarray([float((dist_x == 2.0).sum())]),
                np.asarray([mean_neighbour_degree]),
                np.asarray([returns]),
            )
            return np.asarray(block[0], dtype=np.float64)

        endpoint_u = endpoint(u, dist_u)
        endpoint_v = endpoint(v, dist_v)

        unit_u = np.zeros(n)
        unit_u[u] = 1.0
        unit_v = np.zeros(n)
        unit_v[v] = 1.0
        a1 = adjacency_step(unit_v)
        a3 = adjacency_step(adjacency_step(a1))
        common = float((a1 * adjacency_step(unit_u)).sum())
        l3 = float(a3[u])
        walks = []
        s_k = walk_step(unit_v)
        for _ in range(4):
            s_k = walk_step(s_k)
            walks.append(float(s_k[u]))
        degree_u, degree_v = degree[u], degree[v]
        union = degree_u + degree_v - common
        distance = dist_u[v] if u != v else 0.0
        relation = np.concatenate(
            [
                [
                    np.log1p(common),
                    common / union if union > 0 else 0.0,
                    np.log1p(l3),
                    *walks,
                ],
                _distance_one_hot(np.asarray([distance]))[0],
            ]
        )
        both_two = (dist_u == 2.0) & (dist_v == 2.0)
        shell = float((np.minimum(dist_u, dist_v) == 2.0).sum())
        shared = float(both_two.sum())
        cross = float(((dist_u == 2.0) & (dist_v == 1.0)).sum()) + float(
            ((dist_u == 1.0) & (dist_v == 2.0)).sum()
        )
        context = np.asarray(
            [
                np.log1p(union),
                np.log1p(shell),
                shared / shell if shell > 0 else 0.0,
                np.log1p(cross),
                l3 / (degree_u * degree_v) if degree_u * degree_v > 0 else 0.0,
            ]
        )
        return np.asarray(
            np.concatenate([endpoint_u, endpoint_v, relation, context]), dtype=np.float64
        )

    # ------------------------------------------------------------------ public API

    def coords_by_index(self, u: NDArray[np.int64], v: NDArray[np.int64]) -> NDArray[np.float32]:
        """Coordinates for index pairs; edges are recomputed on the edge-deleted graph.

        Args:
            u: ``(n,)`` node indices into `nodes`.
            v: ``(n,)`` node indices aligned with ``u``.

        Returns:
            ``(n, COORD_DIM)`` float32 coordinates in row order.

        Raises:
            ValueError: On mismatched or out-of-range indices.
        """
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("u and v must be aligned 1-D index arrays")
        if len(u) and (u.min() < 0 or v.min() < 0 or u.max() >= self.size or v.max() >= self.size):
            raise ValueError("pair index outside the coordinate table")
        out = self._bulk(u, v) if len(u) else np.zeros((0, COORD_DIM))
        if len(u):
            edge_rows = np.nonzero((u != v) & (np.asarray(self.adjacency[u, v]).reshape(-1) != 0))[
                0
            ]
            for row in edge_rows.tolist():
                out[row] = self._exact_pair(int(u[row]), int(v[row]))
        return np.asarray(out, dtype=np.float32)

    def coords(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        """Coordinates for node-id pairs, in order (see `coords_by_index`).

        Raises:
            KeyError: If a pair names a node outside the universe graph.
        """
        if not pairs:
            return np.zeros((0, COORD_DIM), dtype=np.float32)
        u = np.fromiter((self.index[str(a)] for a, _ in pairs), dtype=np.int64, count=len(pairs))
        v = np.fromiter((self.index[str(b)] for _, b in pairs), dtype=np.int64, count=len(pairs))
        return self.coords_by_index(u, v)


def coordinate_statistics(
    coords: NDArray[np.floating], *, std_floor: float = 1e-6
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-coordinate mean and standard deviation for standardisation.

    The two endpoint fields share one set of statistics, pooled over both
    endpoints of every row: training pairs are canonically ordered, so their
    marginals differ, and per-field statistics would make the standardised
    representation of a pair depend on which endpoint came first -- breaking
    the reader's swap symmetry. Constant coordinates get a unit scale so they
    standardise to zero instead of blowing up.

    Args:
        coords: ``(n, COORD_DIM)`` raw coordinates of the reference rows.
        std_floor: Standard deviations at or below this become ``1.0``.

    Returns:
        ``(mean, std)`` float32 vectors of length ``COORD_DIM``; the
        ``endpoint_u`` and ``endpoint_v`` blocks are identical.

    Raises:
        ValueError: On an empty or mis-shaped input.
    """
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != COORD_DIM or array.shape[0] == 0:
        raise ValueError(f"expected a non-empty (n, {COORD_DIM}) coordinate matrix")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    u_slice, v_slice = FIELD_SLICES["endpoint_u"], FIELD_SLICES["endpoint_v"]
    pooled = np.concatenate([array[:, u_slice], array[:, v_slice]], axis=0)
    mean[u_slice] = mean[v_slice] = pooled.mean(axis=0)
    std[u_slice] = std[v_slice] = pooled.std(axis=0)
    std = np.where(std <= std_floor, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def reference_pair_coords(graph: nx.Graph, u: str, v: str) -> NDArray[np.float64]:
    """Slow, obviously-correct coordinates of one pair for tests.

    Rebuilds the edge-deleted graph in NetworkX and measures everything with
    library calls or brute force (simple L3 paths by enumeration, walk kernels
    by dense matrix powers).
    """
    work = nx.Graph(graph)
    if u != v and work.has_edge(u, v):
        work.remove_edge(u, v)
    nodes = sorted(str(node) for node in work.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    dense = nx.to_numpy_array(work, nodelist=nodes, dtype=np.float64)
    degree = dense.sum(axis=1)
    inv_sqrt = _safe_divide(np.ones(len(nodes)), np.sqrt(degree))
    normalised = inv_sqrt[:, None] * dense * inv_sqrt[None, :]
    powers = {1: normalised}
    for k in range(2, _MAX_WALK + 1):
        powers[k] = powers[k - 1] @ normalised
    lengths = dict(nx.all_pairs_shortest_path_length(work))
    i, j = index[u], index[v]

    def endpoint(x: str) -> list[float]:
        xi = index[x]
        d = degree[xi]
        triangles = float(nx.triangles(work, x))
        two_hop = float(sum(1 for value in lengths[x].values() if value == 2))
        neighbours = list(work.neighbors(x))
        mean_neighbour_degree = float(np.mean([work.degree(w) for w in neighbours])) if d else 0.0
        clustering = float(nx.clustering(work, x)) if d >= 2 else 0.0
        return [
            float(np.log1p(d)),
            clustering,
            float(np.log1p(triangles)),
            float(np.log1p(d * (d - 1) / 2 - triangles)),
            float(np.log1p(two_hop)),
            float(np.log1p(mean_neighbour_degree)),
            float(powers[2][xi, xi]),
            float(powers[3][xi, xi]),
            float(powers[4][xi, xi]),
        ]

    n_u = set(work.neighbors(u))
    n_v = set(work.neighbors(v))
    common = len(n_u & n_v)
    union = len(n_u | n_v)
    if u == v:
        l3 = 2.0 * nx.triangles(work, u)
    else:
        l3 = float(
            sum(
                1
                for a in n_u
                for b in work.neighbors(a)
                if b in n_v and a not in (u, v) and b not in (u, v) and a != b
            )
        )
    distance = 0.0 if u == v else float(lengths[u].get(v, np.inf))
    relation = [
        float(np.log1p(common)),
        common / union if union else 0.0,
        float(np.log1p(l3)),
        *(float(powers[k][i, j]) for k in range(2, _MAX_WALK + 1)),
        *_distance_one_hot(np.asarray([distance]))[0].tolist(),
    ]
    dist_u = {node: lengths[u].get(node, np.inf) for node in nodes}
    dist_v = {node: lengths[v].get(node, np.inf) for node in nodes}
    shell = [node for node in nodes if min(dist_u[node], dist_v[node]) == 2]
    shared = [node for node in shell if dist_u[node] == 2 and dist_v[node] == 2]
    cross = sum(
        1
        for node in nodes
        if (dist_u[node] == 2 and dist_v[node] == 1) or (dist_u[node] == 1 and dist_v[node] == 2)
    )
    du, dv = degree[i], degree[j]
    context = [
        float(np.log1p(union)),
        float(np.log1p(len(shell))),
        len(shared) / len(shell) if shell else 0.0,
        float(np.log1p(cross)),
        l3 / (du * dv) if du * dv else 0.0,
    ]
    return np.asarray([*endpoint(u), *endpoint(v), *relation, *context], dtype=np.float64)


__all__ = [
    "CONTEXT_NAMES",
    "COORD_DIM",
    "COORD_NAMES",
    "COORD_SPEC",
    "ENDPOINT_NAMES",
    "FIELDS",
    "FIELD_SLICES",
    "RELATION_NAMES",
    "StructCoordinateTable",
    "coordinate_statistics",
    "reference_pair_coords",
]
