"""Fixed motif-template geometry and the deterministic per-pair compiler.

Design: ``docs/superpowers/specs/2026-09-16-motif-graph-grit-prompt-design.md``
sections 2 and 3. The template is a fixed 26-slot, 96-edge graph:
``{u, v} union C union L union R`` with ``|C| = |L| = |R| = 8``. Closure edges
``u-c`` and ``c-v`` carry a shared witness, attachment edges ``u-l`` and ``r-v``
carry the endpoints' bridge attachments, and the 64 interior edges ``l-r``
carry the bridge crossing. Every other adjacency entry, the diagonal and the
queried ``u-v`` entry are exactly zero.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray

from src.data.artifacts import canonical_pair
from src.distill.context_sampler import _anchor_rng

N_SLOTS = 26
N_EDGES = 96

SLOT_U = 0
SLOT_V = 1
C_SLOTS: tuple[int, ...] = tuple(range(2, 10))
L_SLOTS: tuple[int, ...] = tuple(range(10, 18))
R_SLOTS: tuple[int, ...] = tuple(range(18, 26))

ROLE_ENDPOINT = 0
ROLE_CLOSURE = 1
ROLE_BRIDGE = 2
N_ROLES = 3

TYPE_CLOSURE = 0
TYPE_ATTACH = 1
TYPE_INTERIOR = 2
N_EDGE_TYPES = 3

SLOT_ROLES: tuple[int, ...] = (ROLE_ENDPOINT,) * 2 + (ROLE_CLOSURE,) * 8 + (ROLE_BRIDGE,) * 16

CLOSURE_U = slice(0, 8)
CLOSURE_V = slice(8, 16)
ATTACH_L = slice(16, 24)
ATTACH_R = slice(24, 32)
INTERIOR = slice(32, 96)

EDGE_ENDPOINTS: tuple[tuple[int, int], ...] = (
    tuple((SLOT_U, c) for c in C_SLOTS)
    + tuple((c, SLOT_V) for c in C_SLOTS)
    + tuple((SLOT_U, left) for left in L_SLOTS)
    + tuple((right, SLOT_V) for right in R_SLOTS)
    + tuple((L_SLOTS[i], R_SLOTS[j]) for i in range(8) for j in range(8))
)

EDGE_TYPES: tuple[int, ...] = (TYPE_CLOSURE,) * 16 + (TYPE_ATTACH,) * 16 + (TYPE_INTERIOR,) * 64


def _interior_index(left: int, right: int) -> int:
    """Return the edge index of interior slot pair ``(left, right)``.

    Args:
        left: Left-bridge slot ordinal in ``range(8)``.
        right: Right-bridge slot ordinal in ``range(8)``.

    Returns:
        The index into `EDGE_ENDPOINTS` of the interior edge ``l-r``.
    """
    return 32 + 8 * left + right


def _build_swap_perm() -> tuple[int, ...]:
    """Return the edge permutation realising ``u<->v`` with ``L<->R`` (spec section 4).

    Returns:
        A length-96 permutation ``perm`` with ``perm[i]`` the new index of edge ``i``.
    """
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = 8 + k
        perm[8 + k] = k
        perm[16 + k] = 24 + k
        perm[24 + k] = 16 + k
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(right, left)
    return tuple(perm)


SWAP_PERM: tuple[int, ...] = _build_swap_perm()


def role_permutation(
    sigma_c: tuple[int, ...], sigma_l: tuple[int, ...], sigma_r: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the edge permutation induced by relabelling slots within each role.

    Args:
        sigma_c: Destination closure slot of each closure slot.
        sigma_l: Destination left-bridge slot of each left slot.
        sigma_r: Destination right-bridge slot of each right slot.

    Returns:
        A length-96 permutation ``perm`` with ``perm[i]`` the new index of edge ``i``.
    """
    perm = [0] * N_EDGES
    for k in range(8):
        perm[k] = sigma_c[k]
        perm[8 + k] = 8 + sigma_c[k]
        perm[16 + k] = 16 + sigma_l[k]
        perm[24 + k] = 24 + sigma_r[k]
    for left in range(8):
        for right in range(8):
            perm[_interior_index(left, right)] = _interior_index(sigma_l[left], sigma_r[right])
    return tuple(perm)


@dataclass(frozen=True)
class CompiledTemplate:
    """One pair's compiled motif graph.

    Attributes:
        weights: The 96 edge weights against `EDGE_ENDPOINTS`, in ``[0, 1]``.
        closure: Selected shared witnesses, in slot order.
        left: Selected left bridge intermediates, in slot order.
        right: Selected right bridge intermediates, in slot order.
    """

    weights: NDArray[np.float32]
    closure: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]


def _pair_key(u: str, v: str) -> int:
    """Return a swap-invariant 128-bit tie-break key for the pair ``(u, v)``.

    Args:
        u: First endpoint.
        v: Second endpoint.

    Returns:
        A deterministic integer that is identical for ``(u, v)`` and ``(v, u)``.
    """
    a, b = canonical_pair(u, v)
    digest = hashlib.blake2b(f"{a}|{b}".encode(), digest_size=16).digest()
    return int.from_bytes(digest, "little")


def _node_key(node: str) -> int:
    """Return a stable, label-independent ordering token for ``node``.

    Args:
        node: Node id.

    Returns:
        A deterministic 128-bit integer derived from the node id alone.
    """
    return int.from_bytes(hashlib.blake2b(node.encode(), digest_size=16).digest(), "little")


def _relabel(names: tuple[str, ...], sigma: tuple[int, ...]) -> tuple[str, ...]:
    """Move each selected name to its permuted slot, dropping the empty slots.

    Args:
        names: Selected node ids, in slot order.
        sigma: Destination slot of each of the eight slots.

    Returns:
        The selected node ids in the permuted slot order.
    """
    placed = [""] * 8
    for slot, node in enumerate(names):
        placed[sigma[slot]] = node
    return tuple(name for name in placed if name)


def _randomise_roles(
    weights: NDArray[np.float32],
    closure: tuple[str, ...],
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    key: int,
    seed: int,
    epoch: int,
) -> tuple[NDArray[np.float32], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Permute slots within each role; no training identity is an input feature.

    Args:
        weights: The compiled ``(96,)`` weight vector.
        closure: Selected witnesses, in slot order.
        left: Selected left intermediates, in slot order.
        right: Selected right intermediates, in slot order.
        key: The pair's swap-invariant tie-break key.
        seed: Seed lane.
        epoch: Epoch lane.

    Returns:
        The permuted weights and the three relabelled selection tuples.
    """
    rng = _anchor_rng(f"motif:{key:032x}", seed=seed, epoch=epoch)
    sigma_c = tuple(int(x) for x in rng.permutation(8))
    sigma_l = tuple(int(x) for x in rng.permutation(8))
    sigma_r = tuple(int(x) for x in rng.permutation(8))
    perm = np.asarray(role_permutation(sigma_c, sigma_l, sigma_r))
    permuted = np.zeros_like(weights)
    permuted[perm] = weights
    return (
        permuted,
        _relabel(closure, sigma_c),
        _relabel(left, sigma_l),
        _relabel(right, sigma_r),
    )


class MotifTemplateTable:
    """Compile the fixed motif template of any pair over one loopless graph.

    Build once per universe graph; the adjacency and degrees are shared by every
    row. The queried edge is removed before neighbourhoods, candidates, weights
    and truncation are computed (spec section 3).
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Snapshot the loopless adjacency and degrees.

        Args:
            graph: A simple, loopless `networkx.Graph`; isolated nodes are kept.

        Raises:
            ValueError: If the graph is empty or carries a self-loop.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("motif templates need a non-empty graph")
        if nx.number_of_selfloops(graph) > 0:
            raise ValueError("motif templates need a loopless graph; strip self-loops first")
        self._neighbors: dict[str, frozenset[str]] = {
            str(node): frozenset(str(other) for other in neighbours)
            for node, neighbours in graph.adj.items()
        }
        self.nodes: tuple[str, ...] = tuple(sorted(self._neighbors))
        self.index: dict[str, int] = {node: i for i, node in enumerate(self.nodes)}
        self._degree: dict[str, int] = {
            node: len(members) for node, members in self._neighbors.items()
        }
        self._order: dict[str, int] = {node: _node_key(node) for node in self.nodes}

    def _removed_degree(self, u: str, v: str, candidates: frozenset[str]) -> dict[str, int]:
        """Return candidate and endpoint degrees after deleting the queried edge.

        Args:
            u: First endpoint.
            v: Second endpoint.
            candidates: Every node that can occupy a witness or bridge slot.

        Returns:
            Degrees measured on the graph with the queried ``{u, v}`` edge gone.
        """
        degree = {node: self._degree[node] for node in candidates}
        removed = 1 if u != v and v in self._neighbors[u] else 0
        degree[u] = self._degree[u] - removed
        degree[v] = self._degree[v] - removed
        return degree

    def compile_row(
        self, u: str, v: str, *, seed: int = 0, epoch: int = 0, randomise: bool = False
    ) -> CompiledTemplate:
        """Compile one pair's template.

        Args:
            u: First endpoint (slot ``u``).
            v: Second endpoint (slot ``v``).
            seed: Slot-randomisation seed lane.
            epoch: Slot-randomisation epoch lane.
            randomise: Randomise slot order within each role (training only).

        Returns:
            The compiled template.

        Raises:
            KeyError: If an endpoint is not a node of this table's graph.
        """
        weights = np.zeros(N_EDGES, dtype=np.float32)
        if u == v:
            return CompiledTemplate(weights=weights, closure=(), left=(), right=())
        key = _pair_key(u, v)
        n_u = self._neighbors[u] - {v}
        n_v = self._neighbors[v] - {u}
        degree = self._removed_degree(u, v, n_u | n_v)
        order = self._order

        common = sorted(n_u & n_v, key=lambda w: (degree[w], order[w] ^ key))[:8]
        for slot, witness in enumerate(common):
            value = np.float32(degree[witness] ** -0.5) if degree[witness] else np.float32(0.0)
            weights[slot] = value
            weights[8 + slot] = value

        # Both pools are iterated in a fixed, label-independent hash order so
        # the float score sums below are bit-identical across processes.
        left_pool = sorted(n_u, key=lambda w: order[w])
        right_pool = sorted(n_v, key=lambda w: order[w])
        left_score: dict[str, float] = dict.fromkeys(left_pool, 0.0)
        right_score: dict[str, float] = dict.fromkeys(right_pool, 0.0)
        bridges: list[tuple[str, str]] = []
        for i in left_pool:
            if not degree[i]:
                continue
            for j in sorted(self._neighbors[i] & n_v, key=lambda w: order[w]):
                if i == j or not degree[j]:
                    continue
                contribution = float((degree[i] * degree[j]) ** -0.5)
                left_score[i] += contribution
                right_score[j] += contribution
                bridges.append((i, j))
        left = tuple(sorted(left_pool, key=lambda w: (-left_score[w], order[w] ^ key))[:8])
        right = tuple(sorted(right_pool, key=lambda w: (-right_score[w], order[w] ^ key))[:8])
        left_slot = {node: slot for slot, node in enumerate(left)}
        right_slot = {node: slot for slot, node in enumerate(right)}
        for node, slot in left_slot.items():
            weights[16 + slot] = (
                np.float32(degree[node] ** -0.5) if degree[node] else np.float32(0.0)
            )
        for node, slot in right_slot.items():
            weights[24 + slot] = (
                np.float32(degree[node] ** -0.5) if degree[node] else np.float32(0.0)
            )
        for i, j in bridges:
            if i in left_slot and j in right_slot:
                weights[_interior_index(left_slot[i], right_slot[j])] = np.float32(1.0)

        closure = tuple(common)
        if randomise:
            weights, closure, left, right = _randomise_roles(
                weights, closure, left, right, key=key, seed=seed, epoch=epoch
            )
        return CompiledTemplate(weights=weights, closure=closure, left=left, right=right)

    def weights_by_index(
        self,
        u_idx: NDArray[np.int64],
        v_idx: NDArray[np.int64],
        *,
        seed: int = 0,
        epoch: int = 0,
        randomise: bool = False,
    ) -> NDArray[np.float32]:
        """Compile every index pair into an ``(n, 96)`` weight table, in row order.

        Args:
            u_idx: ``(n,)`` indices into `nodes`.
            v_idx: ``(n,)`` indices aligned with ``u_idx``.
            seed: Slot-randomisation seed lane.
            epoch: Slot-randomisation epoch lane.
            randomise: Randomise slot order within each role.

        Returns:
            ``(n, 96)`` float32 weights.

        Raises:
            ValueError: On mismatched or out-of-range index arrays.
        """
        left_idx = np.asarray(u_idx, dtype=np.int64)
        right_idx = np.asarray(v_idx, dtype=np.int64)
        if left_idx.shape != right_idx.shape or left_idx.ndim != 1:
            raise ValueError("u_idx and v_idx must be aligned 1-D index arrays")
        size = len(self.nodes)
        if left_idx.size and (
            left_idx.min() < 0
            or right_idx.min() < 0
            or left_idx.max() >= size
            or right_idx.max() >= size
        ):
            raise ValueError("pair index outside the motif template table")
        out = np.zeros((left_idx.size, N_EDGES), dtype=np.float32)
        rows = zip(left_idx.tolist(), right_idx.tolist(), strict=True)
        for row, (a, b) in enumerate(rows):
            out[row] = self.compile_row(
                self.nodes[a], self.nodes[b], seed=seed, epoch=epoch, randomise=randomise
            ).weights
        return out

    def summary(self) -> dict[str, object]:
        """Provenance and the measured compile rate that decides caching (spec section 3).

        The rate is measured on up to 10,000 pairs drawn round-robin from this
        table's node list, so a caller can record it in ``profile.json`` and pick
        between recomputing per epoch (1.34x the compilations) and a 0.95 GB
        cached table, as spec section 3 requires the choice to be made.

        Measured once on the real seed-42 training graph (7,203 nodes, 36,857
        positives), single-threaded CPU: 18.5 us/row on the round-robin mix and
        235.5 us/row on the denser positive rows. The whole 15-epoch corpus
        therefore recompiles in about 13 minutes at worst, and caching saves
        about 3.3 minutes of it, so this arm recomputes per epoch and
        ``motif_prompt.cache_templates`` defaults to false.

        Returns:
            The node count, the fixed edge count and the measured seconds per
            10,000 compiled rows.
        """
        sample = min(10_000, max(1, len(self.nodes) * (len(self.nodes) - 1) // 2))
        u_idx = np.arange(sample, dtype=np.int64) % len(self.nodes)
        v_idx = (u_idx + 1) % len(self.nodes)
        started = time.monotonic()
        self.weights_by_index(u_idx, v_idx)
        elapsed = time.monotonic() - started
        return {
            "nodes": len(self.nodes),
            "edges": N_EDGES,
            "compile_seconds_per_10k": float(elapsed * 10_000.0 / sample),
        }

    def weights(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        seed: int = 0,
        epoch: int = 0,
        randomise: bool = False,
    ) -> NDArray[np.float32]:
        """Compile node-id pairs into an ``(n, 96)`` weight table, in row order.

        Args:
            pairs: ``(u, v)`` node-id pairs.
            seed: Slot-randomisation seed lane.
            epoch: Slot-randomisation epoch lane.
            randomise: Randomise slot order within each role.

        Returns:
            ``(n, 96)`` float32 weights, aligned with ``pairs``.
        """
        u_idx = np.fromiter((self.index[u] for u, _ in pairs), dtype=np.int64, count=len(pairs))
        v_idx = np.fromiter((self.index[v] for _, v in pairs), dtype=np.int64, count=len(pairs))
        return self.weights_by_index(u_idx, v_idx, seed=seed, epoch=epoch, randomise=randomise)


def apply_families(weights: NDArray[np.float32], families: Sequence[str]) -> NDArray[np.float32]:
    """Zero every edge of a family that is not active (spec section 8 family ablations).

    Args:
        weights: ``(..., 96)`` weights.
        families: Active families, drawn from ``("closure", "bridge")``.

    Returns:
        A copy with the inactive families' edges set to zero.
    """
    out = np.array(weights, dtype=np.float32, copy=True)
    if "closure" not in families:
        out[..., :16] = 0.0
    if "bridge" not in families:
        out[..., 16:] = 0.0
    return out


def mean_template(weights: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the training-corpus mean adjacency ``Abar`` (spec section 3).

    Args:
        weights: ``(n, 96)`` compiled weights over the training corpus.

    Returns:
        The ``(96,)`` float32 mean weight vector.
    """
    mean = np.asarray(weights, dtype=np.float32).mean(axis=0)
    return np.asarray(mean, dtype=np.float32)


def slot_profiles(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the four section 7.3 supervised multisets of a weight table.

    Args:
        weights: ``(B, 96)`` edge weights against `EDGE_ENDPOINTS`.

    Returns:
        ``p`` the 8 wedge products, ``q`` the 64 path products, ``a`` the 16 raw
        attachment weights and ``b`` the 64 raw interior weights, in float32.

    Raises:
        ValueError: If ``weights`` is not ``(B, 96)``.
    """
    if weights.dim() != 2 or weights.size(-1) != N_EDGES:
        raise ValueError(
            f"motif weights must have shape (B, {N_EDGES}), got {tuple(weights.shape)}"
        )
    value = weights.float()
    closure_u = value[:, CLOSURE_U]
    closure_v = value[:, CLOSURE_V]
    attach_l = value[:, ATTACH_L]
    attach_r = value[:, ATTACH_R]
    interior = value[:, INTERIOR].reshape(-1, 8, 8)
    products = attach_l.unsqueeze(2) * interior * attach_r.unsqueeze(1)
    return {
        "p": closure_u * closure_v,
        "q": products.reshape(-1, 64),
        "a": torch.cat([attach_l, attach_r], dim=1),
        "b": interior.reshape(-1, 64),
    }


def count_statistics(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the closed-form count-head statistics of a weight table (spec section 5.1).

    Args:
        weights: ``(B, 96)`` edge weights against `EDGE_ENDPOINTS`.

    Returns:
        ``wedge_mass`` and ``bridge_mass``, the sums of the supervised wedge and
        path products, and the weighted endpoint degrees ``deg_u`` and ``deg_v``,
        each ``(B,)`` in float32.
    """
    parts = slot_profiles(weights)
    value = weights.float()
    return {
        "wedge_mass": parts["p"].sum(dim=1),
        "bridge_mass": parts["q"].sum(dim=1),
        "deg_u": value[:, CLOSURE_U].sum(dim=1) + value[:, ATTACH_L].sum(dim=1),
        "deg_v": value[:, CLOSURE_V].sum(dim=1) + value[:, ATTACH_R].sum(dim=1),
    }
