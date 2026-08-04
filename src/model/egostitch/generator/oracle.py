"""``OracleStructGenerator``: the ground-truth scaffold, smuggled through the slot API.

Wave 1 of the oracle-scaffold experiment. This generator *imagines nothing*.
It reads the true local topology of both endpoints out of a precomputed table
and emits the same `StitchedGraph` the real generator emits, so the downstream
encoder + classifier see a scaffold whose structure is correct by construction.
That makes it the upper bound the learned generator is measured against: any
gap between `oracle_struct` and `egostitch_imagine` is generator error, not
encoder or classifier error.

READ THIS BEFORE TOUCHING ANYTHING BELOW -- the slot fields are reinterpreted
==========================================================================

`EgoStitchImagineGenerator` caches its per-node state as a `SlotSet` +
`projected_x`, and *four separate call sites outside this package* know how to
slice, `index_copy`, `float()` and concatenate exactly those fields:
`composite._merge_slots`, `train_egostitch._fp32_cached_node_state`,
`train_egostitch._stack_cached_node_states` and `score_universe._stack_cached`.
Introducing an oracle-specific state type would mean teaching all four about a
second layout, for no gain.

So this generator reuses `GeneratorNodeState`/`SlotSet` verbatim and *smuggles*
the oracle table through their fields. The reinterpretation is total and
deliberate:

===================  ============================================================
`SlotSet` field      What it means here
===================  ============================================================
``pi``      (B, K)   ``mask.float()`` -- 1.0 for a real selected neighbor, 0.0
                     for padding. Binary, never soft: the scaffold is real.
``mult``    (B, K)   the hub-subsample multiplicity, straight from the table.
``adj``     (B,K,K)  the *true* binary adjacency among the selected neighbors.
``pointer`` (B,K,1)  **not a pointer distribution.** ``neighbor_row.float()``:
                     the table-row identity of the slot's neighbor, ``-1`` for
                     padding. This is what makes the alignment plan exact --
                     two slots align iff they name the same node.
``h``       (B,K,1)  zeros. Unread: `build_scaffold` never touches ``h``, and
                     this generator never calls Sinkhorn, which is ``h``'s only
                     other consumer. Width 1 rather than ``d_p`` because
                     allocating a real embedding width here would cost memory
                     for a tensor nothing reads.
``gate``    (B, K)   zeros. Unread (grounding is a learned-generator concept).
``adj_logits``       zeros. Unread (no adjacency BCE is trained here).
===================  ============================================================

and ``GeneratorNodeState.projected_x`` is ``(B, 1)`` holding the endpoint's own
table row as a float, not a projected feature -- `stitch` needs each endpoint's
identity to do leave-one-out partner masking, and this is the only field in the
cacheable state that survives every caching call site with per-row semantics.

Both float smugglings are exact: `build_oracle_table` refuses to build a table
whose largest row id reaches ``2**24``, above which consecutive integers stop
being representable in fp32 and two distinct nodes could compare equal.

This is a documented lie, not an accident. If you are here because a slot field
looks wrong, it is: read the table above, then read `build_oracle_table`.

What the oracle does *not* fake
-------------------------------

- **Leave-one-out.** `stitch` removes any slot naming the *other* endpoint,
  which is exactly spec Sec 6's rule: for a training positive ``(u, v)``, ``v``
  must not appear inside ``u``'s scaffold. It is a no-op for negatives and for
  a test-graph table that already excludes the queried edge, so one code path
  serves all three cases.
- **The hub policy.** Neighbor selection reuses `EgoTargetBuilder`'s own
  log2-degree-stratified largest-remainder subsample (`src/data/ego_targets.py`)
  rather than a re-implementation, so the oracle's ``K``-slot view of a hub is
  the same view the learned generator is trained to reproduce.
- **Determinism across ranks.** Each node's subsample rng is seeded from
  ``blake2b(f"{node_id}|{seed}|oracle")``, so the table is bit-identical on
  every DDP rank and independent of `node_ids` ordering, batching, or world
  size.

Exact leave-one-out for hub nodes (2026-08-04 P1 fix)
------------------------------------------------------

`EgoTargetBuilder._select_targets(u, rng, exclude_neighbor=v)` removes ``v``
from ``u``'s neighbor list *before* stratifying and subsampling
(`src/data/ego_targets.py:173-203`). For ``deg(u) <= K`` that is equivalent to
subsampling first and then deleting whichever slot happens to hold ``v`` --
every surviving multiplicity is already ``1.0``, so nothing about the
remaining slots depends on which one was removed. For ``deg(u) > K`` it is
*not* equivalent: removing ``v`` from the pool before allocation can change
the largest-remainder allocation across every degree stratum (not just
``v``'s own), and can change the population handed to `rng.choice` for a
stratum even when ``v`` itself was never one of the ``K`` originally-selected
slots (fewer members in a stratum shifts every later member's index, so the
same rng draw resolves to a different node). A generator that just zeroed the
one slot naming ``v`` (the pre-fix behavior of `_drop_partner_slots`) would
therefore hand a hub's positive-edge scaffold a permanently short table (only
``K - 1`` live slots, no replacement draw) with the wrong multiplicity label
on top -- silently understating represented degree and breaking the "the
oracle is an exact upper bound on the learned generator" claim this whole
experiment rests on.

The fix precomputes, at table-build time, one **exact override row** per
``(hub node, true incident edge)`` -- i.e. for every node ``u`` with
``deg(u) > K`` and every real neighbor ``v`` of ``u``, the row
`EgoTargetBuilder._select_targets(u, _node_rng(u, seed), exclude_neighbor=v)`
actually returns. This is a direct call to the canonical selection method
with the canonical exclusion argument, not a re-derivation, so parity is
exact by construction rather than by careful bookkeeping. `stitch` looks the
override row up by ``(own_row, partner_row)`` (a sorted `torch.searchsorted`
over a flat int64 key, vectorized across the batch) and substitutes the
*entire* slot row -- not a per-slot patch -- when a match is found, falling
back to the original `_drop_partner_slots` zeroing (still exact for the
``deg(u) <= K`` case, and a correct no-op for negatives, self-pairs, and any
pair that isn't a real edge) otherwise. Each override row's rng is
`_node_rng(u, seed)` -- the identical per-node seed the base row already
uses, restarted fresh for every distinct ``v`` -- so an override row is a
pure, order- and rank-independent function of ``(u, v, seed)``, exactly like
every other table entry.

**Why eager precomputation over hubs only (not a lazy per-query cache, not
every node):** the population needing an exact row is bounded by
``sum(deg(u) for u in G if deg(u) > K)`` -- only hub-incident edges, because
non-hub exclusion is provably identical to the existing zeroing (previous
paragraph). Measured on this repository's real training graph
(`breadth_first` strategy, ``V_fit`` = 7,560 nodes, 30,820 edges, ``K = 16``,
the default `EgoStitchConfig.slots` every wave-1 oracle config uses): 897 hub
nodes, summing to 33,572 incident edges needing an override row. At
``K = 16`` each row costs ``K`` int64 (128 B) + ``K`` float32 (64 B) + ``K``
bool (16 B) + ``K x K`` float32 adjacency (1024 B) = 1,232 B, so the whole
override table is **~41 MB** -- negligible next to the base table's own
``V x K x K`` adjacency block and utterly dominated by model/activation
memory. A lazy per-query cache would save at most this same ~41 MB while
adding thread-safety and re-entrancy questions inside a DDP training step,
for no measurable gain; precomputing at table-build time (already a one-time,
off-the-hot-path cost) is simpler and cannot introduce order-dependence.
Restricting to hub nodes only (rather than every node) is exact, not an
approximation -- see the previous paragraph.

Zero parameters. `state_dict()` is empty; the table is held as plain attributes,
never as buffers -- see `set_oracle_context`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import networkx as nx
import numpy as np
import torch

from src.data.ego_targets import EgoTargetBuilder
from src.model.egostitch.generator.assemble import (
    EDGE_TYPES,
    FEAT_DIM,
    ScaffoldInputPerturbation,
    build_scaffold,
)
from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.generator.egostitch import (
    GeneratorNodeState,
    StitchedGraph,
    _slotset_to_aux,
)
from src.model.egostitch.generator.imagine import SlotSet

__all__ = ["OracleScaffoldTable", "OracleStructGenerator", "build_oracle_table"]

# Above this, consecutive integers are no longer exactly representable in fp32,
# so two distinct table rows could compare equal after the `pointer`/
# `projected_x` smuggling. 2**24 == 16_777_216 nodes is far beyond any universe
# this project handles; the check exists so the failure is a loud `ValueError`
# and not a silently wrong alignment plan.
_MAX_EXACT_FP32_INTEGER = 2**24

_PAD_ROW = -1

# Stride used to pack an (own_row, partner_row) pair into one int64 lookup
# key for the override table (`own_row * _ROW_ID_STRIDE + partner_row`). Must
# exceed every possible row id -- reusing the fp32-exactness bound is
# convenient (both are "row ids must fit under this ceiling" constraints) and
# leaves enormous headroom: the largest key is under 2**48, far inside int64.
# This lookup is plain int64 arithmetic, never smuggled through fp32, so it
# does not itself need the fp32-exactness property -- only a ceiling.
_ROW_ID_STRIDE = _MAX_EXACT_FP32_INTEGER

# `x`'s node-existence channel, written by `build_scaffold` as
# `feats[:, :, 4] = cat([1, 1, pi_src, pi_dst])`. Read back out rather than
# recomputed so `ImaginedGraph.mask` can never disagree with `x` -- the same
# reasoning (and the same constant) as `generator/egostitch.py`.
_EXISTENCE_CHANNEL = 4


class OracleScaffoldTable(NamedTuple):
    """The precomputed ground-truth local scaffold of every node in one universe.

    Row order is the order of the ``node_ids`` sequence passed to
    `build_oracle_table`, so a caller that built the table over its own F0 row
    order can use the identity map as ``f0_row_to_table_row``.

    Attributes:
        neighbor_row: Shape ``(V, K)`` int64 -- the table-row id of each
            selected neighbor, ``-1`` for padding. Ids ``>= V`` are possible
            and legal: a neighbor present in `graph` but absent from
            ``node_ids`` still needs a *distinct identity* for the alignment
            plan even though it owns no table row of its own (see
            `build_oracle_table`).
        mult: Shape ``(V, K)`` float32 hub-subsample multiplicity labels
            (``stratum_size / allocated_count``; ``1.0`` when the node's degree
            fits in ``K``). Zero for padding.
        adj: Shape ``(V, K, K)`` float32 binary -- the true adjacency among the
            selected neighbors on `graph`, zero on padded rows/columns.
        mask: Shape ``(V, K)`` bool -- slot validity.
        override_key: Shape ``(E,)`` int64, ascending and unique -- one entry
            per ``(hub node, true incident edge)``, encoded as
            ``own_row * _ROW_ID_STRIDE + partner_row``. Looked up with
            `torch.searchsorted` at `OracleStructGenerator.stitch` time. Empty
            when `graph` has no node with ``deg > K``.
        override_neighbor_row: Shape ``(E, K)`` int64 -- the *entire* exact
            leave-one-out row `EgoTargetBuilder._select_targets(u, rng,
            exclude_neighbor=v)` returns for the edge named by the matching
            `override_key` entry (same padding convention as `neighbor_row`).
        override_mult: Shape ``(E, K)`` float32 -- the override row's
            multiplicity labels, recomputed over the partner-excluded strata
            (not sliced out of `mult`).
        override_adj: Shape ``(E, K, K)`` float32 -- the true adjacency among
            *that* row's selected neighbors (generally a different node set
            than `adj`'s row for ``u``, hence recomputed, not resliced).
        override_mask: Shape ``(E, K)`` bool -- the override row's slot
            validity.
    """

    neighbor_row: torch.Tensor
    mult: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor
    override_key: torch.Tensor
    override_neighbor_row: torch.Tensor
    override_mult: torch.Tensor
    override_adj: torch.Tensor
    override_mask: torch.Tensor


def _node_rng(node_id: str, seed: int) -> np.random.Generator:
    """Return the per-node subsample rng, identical on every rank.

    Seeded from ``blake2b(f"{node_id}|{seed}|oracle")`` rather than from a
    stream advanced per node, so the draw for one node cannot depend on how
    many nodes preceded it -- which is what makes the table independent of
    `node_ids` ordering, DDP sharding, and batch size.

    Args:
        node_id: The node's string id.
        seed: The table-wide seed (`GeneratorConfig.oracle_seed`).

    Returns:
        A freshly seeded generator.
    """
    digest = hashlib.blake2b(f"{node_id}|{seed}|oracle".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _fill_row(
    neighbor_row: torch.Tensor,
    mult: torch.Tensor,
    mask: torch.Tensor,
    adj: torch.Tensor,
    row: int,
    selected: Sequence[str],
    multiplicity: Sequence[float],
    row_of: Mapping[str, int],
    graph: nx.Graph,
) -> None:
    """Write one `_select_targets` result into row `row` of four parallel tensors.

    Shared between `build_oracle_table`'s base-row pass and its hub
    override-row pass -- both are "take an `EgoTargetBuilder._select_targets`
    result and pack it into the table's slot layout," differing only in which
    tensors and which row index they write to.

    Args:
        neighbor_row: ``(rows, K)`` int64, written in place.
        mult: ``(rows, K)`` float32, written in place.
        mask: ``(rows, K)`` bool, written in place.
        adj: ``(rows, K, K)`` float32, written in place.
        row: The row index to write.
        selected: Neighbor ids from `_select_targets`, in slot order.
        multiplicity: Parallel multiplicity labels.
        row_of: Node id -> table row (including overflow rows).
        graph: The graph `selected`'s adjacency is read from.
    """
    count = len(selected)
    if count == 0:
        return
    neighbor_row[row, :count] = torch.tensor(
        [row_of[neighbor] for neighbor in selected], dtype=torch.int64
    )
    mult[row, :count] = torch.tensor(list(multiplicity), dtype=torch.float32)
    mask[row, :count] = True
    for i in range(count):
        for j in range(i + 1, count):
            if graph.has_edge(selected[i], selected[j]):
                adj[row, i, j] = 1.0
                adj[row, j, i] = 1.0


def build_oracle_table(
    graph: nx.Graph,
    node_ids: Sequence[str],
    *,
    slots: int,
    seed: int = 0,
) -> OracleScaffoldTable:
    """Materialize the ground-truth ``K``-slot scaffold of every node in `node_ids`.

    Neighbor selection is `EgoTargetBuilder`'s own hub policy, reused rather
    than reimplemented: when ``deg(u) > K`` the neighbors are stratified by
    ``log2(deg)`` bucket, allocated across strata by largest remainder, and
    subsampled within each stratum, with the multiplicity label recording how
    many real neighbors each surviving slot stands for (spec Sec 13.4,
    `src/data/ego_targets.py`). That is the *same* view of a hub the learned
    generator is trained against, which is the only way the oracle is an upper
    bound on that generator rather than a different task. For every node with
    ``deg > K``, this also precomputes an *exact* leave-one-out override row
    per true incident edge (module docstring, "Exact leave-one-out for hub
    nodes"), consumed by `OracleStructGenerator.stitch` rather than by this
    function's own per-node rows.

    Args:
        graph: The structural graph the scaffold is read from -- ``G_struct``
            for the training/validation table, the test-graph projection for a
            test-time table. Must be self-loop-free (`EgoTargetBuilder`
            enforces this; spec Sec 9.3).
        node_ids: The table's row order. Duplicates are rejected. Ids absent
            from `graph` (isolated or featureless nodes) get an all-padding
            row rather than an error -- an isolated node genuinely has an
            empty scaffold.
        slots: ``K`` -- slots per node. Must match the model's
            `EgoStitchConfig.slots`.
        seed: Table-wide seed folded into every per-node rng.

    Returns:
        The `OracleScaffoldTable`.

    Raises:
        ValueError: On duplicate `node_ids`, a non-positive `slots`, or an
            identity space large enough to break fp32 exactness.
    """
    if slots <= 0:
        raise ValueError(f"slots must be positive, got {slots}")

    row_of: dict[str, int] = {}
    for row, node_id in enumerate(node_ids):
        if node_id in row_of:
            raise ValueError(f"node_ids contains a duplicate: {node_id!r}")
        row_of[node_id] = row
    num_rows = len(row_of)

    # A neighbor that lives in `graph` but not in `node_ids` owns no table row,
    # yet the alignment plan still has to be able to tell it apart from every
    # other node -- two endpoints that share such a neighbor must align on it.
    # Extending the identity space past `num_rows` (deterministically, by sorted
    # id) is what lets that work without silently dropping the neighbor, which
    # would fabricate a *less* connected scaffold than the truth. Rows
    # `0 .. num_rows-1` still mean exactly what `node_ids` says they mean.
    overflow = sorted(node for node in graph.nodes() if node not in row_of)
    for offset, node_id in enumerate(overflow):
        row_of[node_id] = num_rows + offset
    if len(row_of) >= _MAX_EXACT_FP32_INTEGER:
        raise ValueError(
            "oracle identity space "
            f"({len(row_of)} nodes) reaches 2**24, past which fp32 cannot "
            "represent consecutive integers exactly and the smuggled "
            "pointer/projected_x identities would start colliding"
        )

    # `EgoTargetBuilder` wants an F0 matrix and a grounding pool it only reads
    # inside `build()`; the hub policy in `_select_targets` reads neither.
    # Passing placeholders is what lets the policy be *reused* instead of
    # copied. (A public `select_targets` on that class would be tidier, but
    # `src/data/ego_targets.py` is owned by another workstream this pass --
    # flagged rather than edited.)
    builder = EgoTargetBuilder(
        graph,
        np.zeros((graph.number_of_nodes(), 1), dtype=np.float32),
        {node: index for index, node in enumerate(graph.nodes())},
        {},
        slots=slots,
    )

    neighbor_row = torch.full((num_rows, slots), _PAD_ROW, dtype=torch.int64)
    mult = torch.zeros((num_rows, slots), dtype=torch.float32)
    adj = torch.zeros((num_rows, slots, slots), dtype=torch.float32)
    mask = torch.zeros((num_rows, slots), dtype=torch.bool)

    # Nodes needing an exact per-edge override row: exactly the nodes whose
    # *base* selection can be truncated (`deg > K`), collected here so the
    # override pass below doesn't need a second `graph.degree` scan. See the
    # module docstring's "Exact leave-one-out for hub nodes" section for why
    # this set -- not "every node" -- is the exact population, not an
    # approximation of it.
    hub_edges: list[tuple[int, str, str]] = []

    for row, node_id in enumerate(node_ids):
        if node_id not in graph:
            continue
        selected, multiplicity = builder._select_targets(node_id, _node_rng(node_id, seed))
        _fill_row(neighbor_row, mult, mask, adj, row, selected, multiplicity, row_of, graph)
        if graph.degree(node_id) > slots:
            for neighbor in sorted(graph.neighbors(node_id)):
                hub_edges.append((row, node_id, neighbor))

    num_overrides = len(hub_edges)
    override_neighbor_row = torch.full((num_overrides, slots), _PAD_ROW, dtype=torch.int64)
    override_mult = torch.zeros((num_overrides, slots), dtype=torch.float32)
    override_adj = torch.zeros((num_overrides, slots, slots), dtype=torch.float32)
    override_mask = torch.zeros((num_overrides, slots), dtype=torch.bool)
    override_key = torch.zeros((num_overrides,), dtype=torch.int64)

    for idx, (own_row, node_id, excluded) in enumerate(hub_edges):
        # Identical seed to the base row's -- `_node_rng` is re-derived fresh
        # from `(node_id, seed)` every call, never advanced across calls, so
        # this is a pure function of `(node_id, excluded, seed)` regardless of
        # `hub_edges` iteration order (module docstring).
        selected, multiplicity = builder._select_targets(
            node_id, _node_rng(node_id, seed), exclude_neighbor=excluded
        )
        _fill_row(
            override_neighbor_row,
            override_mult,
            override_mask,
            override_adj,
            idx,
            selected,
            multiplicity,
            row_of,
            graph,
        )
        override_key[idx] = own_row * _ROW_ID_STRIDE + row_of[excluded]

    # `hub_edges` is grouped by `own_row` but not globally key-sorted (a
    # node's neighbors are visited in sorted-id order, not sorted-row-id
    # order), and `OracleStructGenerator.stitch` needs ascending keys for
    # `torch.searchsorted`.
    order = torch.argsort(override_key)
    override_key = override_key[order]
    override_neighbor_row = override_neighbor_row[order]
    override_mult = override_mult[order]
    override_adj = override_adj[order]
    override_mask = override_mask[order]

    return OracleScaffoldTable(
        neighbor_row=neighbor_row,
        mult=mult,
        adj=adj,
        mask=mask,
        override_key=override_key,
        override_neighbor_row=override_neighbor_row,
        override_mult=override_mult,
        override_adj=override_adj,
        override_mask=override_mask,
    )


class OracleStructGenerator(
    NeighborhoodGenerator[GeneratorNodeState, ScaffoldInputPerturbation, StitchedGraph]
):
    """Emits the ground-truth scaffold instead of an imagined one. Zero parameters.

    Binds the same three generic parameters as `EgoStitchImagineGenerator` --
    `GeneratorNodeState` / `ScaffoldInputPerturbation` / `StitchedGraph` -- so
    the composite, the caching call sites and `swapped()`'s AB/BA symmetry all
    work unchanged. What differs is only *where the numbers come from*: a
    lookup table rather than a decoder.
    """

    def __init__(self, *, slots: int) -> None:
        """Build the (parameter-free) generator.

        Args:
            slots: ``K`` -- must match the table's slot count and the model's
                `EgoStitchConfig.slots`.

        Raises:
            ValueError: If `slots` is not positive.
        """
        super().__init__()
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        self.slots = int(slots)
        self._table: OracleScaffoldTable | None = None
        self._f0_row_to_table_row: torch.Tensor | None = None

    # ------------------------------------------------------------------ context

    def set_oracle_context(
        self, table: OracleScaffoldTable, f0_row_to_table_row: torch.Tensor
    ) -> None:
        """Install the scaffold table and the F0-row -> table-row lookup.

        Deliberately **not** `register_buffer`, and deliberately not
        `nn.Parameter`. Two reasons, both load-bearing:

        1. DDP broadcasts registered buffers from rank 0 on every forward pass
           by default (``broadcast_buffers=True``). ``table.adj`` alone is
           ``V * K * K`` float32 -- hundreds of megabytes for a realistic
           universe -- so registering it would push that over the interconnect
           once per step to re-send a constant.
        2. A buffer, persistent or not, is state the checkpoint machinery and
           `state_dict` consumers have to reason about. This generator must
           have an empty `state_dict`; the table is *data*, reproducible from
           the graph and the seed, not model state.

        Device placement is therefore handled lazily in `encode_node`, which
        moves (and caches) the table on first use for a given device. That also
        makes call order irrelevant: `set_oracle_context` may be called before
        or after `.to(device)`.

        Args:
            table: The `OracleScaffoldTable`, built with the same ``K``.
            f0_row_to_table_row: Shape ``(n_f0,)`` int64 mapping each
                *context-local* F0 row to its table row, or ``-1`` for an F0
                row the table does not cover. `encode_node` raises on a ``-1``
                hit rather than emitting an empty scaffold, because a node the
                oracle cannot see is a data-plumbing bug, not a legitimately
                isolated node (isolated nodes do have table rows -- empty ones).

        Raises:
            ValueError: On a slot-count mismatch or a malformed lookup tensor.
        """
        if table.neighbor_row.shape[1] != self.slots:
            raise ValueError(
                f"table slot count {table.neighbor_row.shape[1]} != generator slots {self.slots}"
            )
        num_overrides = table.override_key.shape[0]
        if table.override_neighbor_row.shape != (num_overrides, self.slots):
            raise ValueError(
                "table override rows are malformed: override_key has "
                f"{num_overrides} entries but override_neighbor_row has shape "
                f"{tuple(table.override_neighbor_row.shape)}"
            )
        if f0_row_to_table_row.ndim != 1 or f0_row_to_table_row.dtype != torch.int64:
            raise ValueError(
                "f0_row_to_table_row must be a 1-D int64 tensor, got "
                f"{tuple(f0_row_to_table_row.shape)} / {f0_row_to_table_row.dtype}"
            )
        self._table = table
        self._f0_row_to_table_row = f0_row_to_table_row
        self._device_cache: dict[torch.device, tuple[OracleScaffoldTable, torch.Tensor]] = {}

    def _context_for(
        self, device: torch.device
    ) -> tuple[OracleScaffoldTable, torch.Tensor]:
        """Return the table and lookup on `device`, moving and caching on first use."""
        if self._table is None or self._f0_row_to_table_row is None:
            raise RuntimeError(
                "OracleStructGenerator.set_oracle_context has not been called: "
                "this generator has no scaffold to emit"
            )
        cache = getattr(self, "_device_cache", None)
        if cache is None:  # pragma: no cover - only if set_oracle_context was bypassed
            cache = {}
            self._device_cache = cache
        hit = cache.get(device)
        if hit is None:
            table = OracleScaffoldTable(*(value.to(device) for value in self._table))
            lookup = self._f0_row_to_table_row.to(device)
            hit = (table, lookup)
            cache[device] = hit
        return hit

    # ------------------------------------------------------------------ per-node

    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
        *,
        node_rows: torch.Tensor | None = None,
    ) -> GeneratorNodeState:
        """Look each node's true scaffold up and pack it into the slot layout.

        `x` and `ground` are ignored entirely -- that is the point: the oracle
        is not a function of the node's *features*, only of its *topology*.
        They stay in the signature because every caller in the composite passes
        them positionally.

        Args:
            x: Shape ``(B, d)`` frozen node features. Used only for device and
                batch size.
            ground: Unused.
            ground_ids: Carried through unchanged, as the base contract
                requires.
            node_rows: Shape ``(B,)`` int64 context-local F0 row ids. Required.

        Returns:
            A `GeneratorNodeState` whose fields carry the reinterpretation
            documented in this module's header.

        Raises:
            RuntimeError: If `set_oracle_context` was never called.
            ValueError: If `node_rows` is absent, misshaped, or names an F0 row
                the table does not cover.
        """
        del ground
        batch_size = int(x.shape[0])
        if node_rows is None:
            raise ValueError(
                "OracleStructGenerator.encode_node requires node_rows: the oracle "
                "scaffold is looked up by node identity, not derived from features"
            )
        if node_rows.shape != (batch_size,):
            raise ValueError(
                f"node_rows shape must be {(batch_size,)}, got {tuple(node_rows.shape)}"
            )
        table, lookup = self._context_for(x.device)
        rows = node_rows.to(device=x.device, dtype=torch.int64)
        table_rows = lookup.index_select(0, rows)
        if bool((table_rows < 0).any()):
            missing = rows[table_rows < 0][:5].tolist()
            raise ValueError(
                f"F0 rows {missing} have no oracle table row; the table was built "
                "over a different node universe than this batch"
            )

        neighbor_row = table.neighbor_row.index_select(0, table_rows)
        mask = table.mask.index_select(0, table_rows)
        slots = SlotSet(
            # `h` is genuinely unread -- width 1, not `d_p`, to keep the cached
            # state small. See the module header's field table.
            h=torch.zeros(batch_size, self.slots, 1, device=x.device, dtype=torch.float32),
            pi=mask.to(dtype=torch.float32),
            mult=table.mult.index_select(0, table_rows),
            gate=torch.zeros(batch_size, self.slots, device=x.device, dtype=torch.float32),
            pointer=neighbor_row.to(dtype=torch.float32).unsqueeze(-1),
            adj=table.adj.index_select(0, table_rows),
            adj_logits=torch.zeros(
                batch_size, self.slots, self.slots, device=x.device, dtype=torch.float32
            ),
        )
        return GeneratorNodeState(
            slots=slots,
            projected_x=table_rows.to(dtype=torch.float32).unsqueeze(-1),
            ground_ids=ground_ids,
        )

    # ------------------------------------------------------------------ pair-level

    @staticmethod
    def _drop_partner_slots(
        slots: SlotSet, partner_row: torch.Tensor
    ) -> tuple[SlotSet, torch.Tensor]:
        """Remove any slot naming `partner_row` -- spec Sec 6's leave-one-out.

        **Exact only when `slots`' owner has `deg <= K`** (or the pair isn't a
        real edge at all): the row is already every true neighbor with
        multiplicity ``1.0``, so deleting the one slot naming ``partner_row``
        and leaving the rest untouched reproduces
        `EgoTargetBuilder._select_targets(..., exclude_neighbor=partner)`
        exactly -- removing one node from a list of ``<= K`` never triggers
        stratified subsampling either before or after the removal. For a hub
        (``deg > K``) it is *not* exact on its own: excluding a neighbor
        before allocation can change which neighbors are selected and their
        multiplicities, not just delete one slot. `_apply_leave_one_out` is
        the entry point that gets this right in both regimes -- it uses this
        method only as the correct-for-its-regime fallback when no
        precomputed exact override row applies. Do not call this method
        directly from `stitch`.

        For a training positive ``(u, v)``, ``v`` is a real neighbor of ``u`` on
        ``G_struct`` and would otherwise sit inside ``u``'s own scaffold, making
        the answer readable off the conditioning. Removing it is not an
        approximation of the learned generator's behaviour -- it *is* the rule
        the edge-stream structural targets already follow
        (`EgoTargetBuilder.build(..., exclude_neighbors=...)`, spec Sec 9.3),
        applied here at stitch time because the table is built once, without
        knowing which pairs will be queried.

        Naturally a no-op for negatives (``v`` is not a neighbor), for
        self-pairs (a node is never its own neighbor: ``G_struct`` is
        self-loop-free), and for a test-graph table that already excludes the
        queried edge.

        Args:
            slots: One endpoint's slot bundle.
            partner_row: Shape ``(B,)`` float32 table row of the *other*
                endpoint.

        Returns:
            ``(slots_without_partner, dropped)`` where ``dropped`` is the
            ``(B, K)`` bool mask of removed slots.
        """
        neighbor_row = slots.pointer[..., 0]
        dropped = (neighbor_row == partner_row.unsqueeze(1)) & (neighbor_row >= 0.0)
        keep = (~dropped).to(dtype=slots.pi.dtype)
        return (
            SlotSet(
                h=slots.h,
                pi=slots.pi * keep,
                mult=slots.mult * keep,
                gate=slots.gate,
                # The identity is cleared too, not just the existence flag:
                # `aux` is read by telemetry and by this generator's own future
                # losses, and a dropped slot that still advertises its neighbor
                # would be a trap for both.
                pointer=torch.where(
                    dropped.unsqueeze(-1),
                    torch.full_like(slots.pointer, float(_PAD_ROW)),
                    slots.pointer,
                ),
                adj=slots.adj * keep.unsqueeze(1) * keep.unsqueeze(2),
                adj_logits=slots.adj_logits,
            ),
            dropped,
        )

    @staticmethod
    def _apply_leave_one_out(
        table: OracleScaffoldTable,
        slots: SlotSet,
        own_row: torch.Tensor,
        partner_row: torch.Tensor,
    ) -> SlotSet:
        """Exact leave-one-out: substitute the precomputed override row when one exists.

        Falls back to `_drop_partner_slots`'s per-slot zeroing otherwise. An
        override row exists exactly when ``own_row`` names a hub
        (``deg > K``) and ``partner_row`` names one of its *true* neighbors --
        the only regime in which `_drop_partner_slots`'s zeroing is not
        already exact (its own docstring). The lookup is a batched
        `torch.searchsorted` over `table.override_key`
        (``own_row * _ROW_ID_STRIDE + partner_row``), so it costs one sorted
        search over ``E`` entries per call, not a Python loop over the batch.

        When a match is found, the **entire** row is replaced (pi, mult,
        pointer, adj) -- not just the one slot matching `partner_row` --
        because the override row's selected neighbor set is, in general, a
        different subset of ``u``'s neighborhood than the base row's (a
        replacement slot the base row never sampled can now be filled).

        Args:
            table: The table `own_row`/`partner_row` index into.
            slots: One endpoint's slot bundle (the side being masked).
            own_row: Shape ``(B,)`` int64 table row of `slots`'s owner.
            partner_row: Shape ``(B,)`` float32 table row of the other
                endpoint (`_drop_partner_slots`'s expected dtype, passed
                through unchanged for the fallback path).

        Returns:
            The leave-one-out `SlotSet`, exact in both the hub and non-hub
            regimes.
        """
        fallback, _ = OracleStructGenerator._drop_partner_slots(slots, partner_row)

        override_key = table.override_key
        if override_key.numel() == 0:
            return fallback

        partner_row_i = partner_row.round().to(dtype=torch.int64)
        keys = own_row * _ROW_ID_STRIDE + partner_row_i
        idx = torch.searchsorted(override_key, keys)
        idx_safe = idx.clamp(max=override_key.numel() - 1)
        found = override_key.index_select(0, idx_safe) == keys
        if not bool(found.any()):
            return fallback

        override_pi = table.override_mask.index_select(0, idx_safe).to(dtype=slots.pi.dtype)
        override_mult = table.override_mult.index_select(0, idx_safe)
        override_pointer = (
            table.override_neighbor_row.index_select(0, idx_safe)
            .to(dtype=torch.float32)
            .unsqueeze(-1)
        )
        override_adj = table.override_adj.index_select(0, idx_safe)

        found_row = found.unsqueeze(-1)  # (B, 1) -- broadcasts over pi/mult's (B, K)
        found_cell = found_row.unsqueeze(-1)  # (B, 1, 1) -- broadcasts over pointer/adj
        return SlotSet(
            h=slots.h,
            pi=torch.where(found_row, override_pi, fallback.pi),
            mult=torch.where(found_row, override_mult, fallback.mult),
            gate=slots.gate,
            pointer=torch.where(found_cell, override_pointer, fallback.pointer),
            adj=torch.where(found_cell, override_adj, fallback.adj),
            adj_logits=slots.adj_logits,
        )

    def stitch(
        self,
        state_a: GeneratorNodeState,
        state_b: GeneratorNodeState,
        is_self: torch.Tensor,
        *,
        perturbation: ScaffoldInputPerturbation | None = None,
    ) -> StitchedGraph:
        """Assemble the joint scaffold with an *exact* alignment plan.

        Three steps, in order:

        1. Leave-one-out partner masking on both sides
           (`_apply_leave_one_out`): the precomputed exact override row for a
           hub endpoint whose partner is a true neighbor, else
           `_drop_partner_slots`'s per-slot zeroing (exact for every other
           case -- see that method's docstring).
        2. The alignment plan, which needs no Sinkhorn at all:
           ``plan[b, i, j] = pi_a[b, i] * pi_b[b, j] *
           1[neighbor_row_a[b, i] == neighbor_row_b[b, j] != -1]``.
           Because neighbor ids are distinct within one node's slots, this is a
           partial permutation matrix -- the true shared-neighbor
           correspondence, which is precisely what the learned generator's
           unbalanced OT is trying to approximate.
        3. `build_scaffold`, unchanged, so the emitted `StitchedGraph` is
           structurally identical to the learned generator's and inherits
           `swapped()`'s AB/BA relabelling.

        ``is_self`` rows take the same exact-identity plan the learned
        generator's `stitch` installs (`generator/egostitch.py`): a full
        ``eye(K)``, not the ``pi``-weighted diagonal the formula above would
        give. Matching the existing contract matters more here than internal
        elegance -- self-pair rows are the spec Sec 13.9 single-ego path and
        every downstream consumer was written against that exact form.

        Args:
            state_a: Endpoint-A state from `encode_node`.
            state_b: Endpoint-B state from `encode_node`.
            is_self: Shape ``(B,)`` boolean self-pair mask.
            perturbation: Optional deterministic scaffold-structure control,
                forwarded to `build_scaffold` unchanged (the §14.4.5 arms work
                against the oracle exactly as against the learned generator).

        Returns:
            The `StitchedGraph`.

        Raises:
            ValueError: On disagreeing batch size / slot count, or a
                misshaped `is_self`.
        """
        slots_a, slots_b = state_a.slots, state_b.slots
        batch_size, num_slots = slots_a.pi.shape
        if slots_b.pi.shape != (batch_size, num_slots):
            raise ValueError(
                "state_a and state_b must have matching batch size and slot count: "
                f"{tuple(slots_a.pi.shape)} != {tuple(slots_b.pi.shape)}"
            )
        if is_self.shape != (batch_size,):
            raise ValueError(f"is_self shape must be {(batch_size,)}, got {tuple(is_self.shape)}")

        row_a = state_a.projected_x[:, 0]
        row_b = state_b.projected_x[:, 0]
        table, _ = self._context_for(slots_a.pi.device)
        row_a_i = row_a.round().to(dtype=torch.int64)
        row_b_i = row_b.round().to(dtype=torch.int64)
        slots_a = self._apply_leave_one_out(table, slots_a, row_a_i, row_b)
        slots_b = self._apply_leave_one_out(table, slots_b, row_b_i, row_a)

        neighbor_a = slots_a.pointer[..., 0]
        neighbor_b = slots_b.pointer[..., 0]
        same = (neighbor_a.unsqueeze(2) == neighbor_b.unsqueeze(1)) & (
            neighbor_a.unsqueeze(2) >= 0.0
        )
        plan = (
            slots_a.pi.unsqueeze(2).float()
            * slots_b.pi.unsqueeze(1).float()
            * same.to(dtype=torch.float32)
        )

        self_rows = torch.nonzero(is_self, as_tuple=False).squeeze(-1)
        if self_rows.numel() > 0:
            identity = torch.eye(num_slots, device=plan.device, dtype=plan.dtype)
            plan = plan.index_copy(0, self_rows, identity.expand(self_rows.numel(), -1, -1))

        # `log(0) == -inf` exactly, which is the value the alignment loss and
        # every plan consumer expect for an impossible cell; `torch.where`
        # keeps the log off the zeros so no `-inf * 0 == nan` can appear in a
        # backward pass through a branch that is not taken.
        log_plan = torch.where(
            plan > 0.0,
            torch.log(plan.clamp(min=torch.finfo(torch.float32).tiny)),
            torch.full_like(plan, -torch.inf),
        )

        scaffold = build_scaffold(slots_a, slots_b, plan, perturbation=perturbation)
        aux: dict[str, torch.Tensor] = {
            **_slotset_to_aux("slots_a", slots_a),
            **_slotset_to_aux("slots_b", slots_b),
            "plan": plan,
            "log_plan": log_plan,
        }
        return StitchedGraph(
            x=scaffold.feats,
            adj=scaffold.adj,
            mask=scaffold.feats[..., _EXISTENCE_CHANNEL],
            aux=aux,
            directed=True,
        )

    def auxiliary_losses(
        self, graph: StitchedGraph | None, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No losses: the oracle has nothing to learn and nothing to supervise.

        Every generator-owned loss in `EgoStitchImagineGenerator` exists to pull
        an imagined scaffold towards the true one. Here the scaffold *is* the
        true one, and there are no parameters for a loss to reach anyway.

        Args:
            graph: Unused.
            batch: Unused.

        Returns:
            An empty dict.
        """
        del graph, batch
        return {}

    def graph_dims(self) -> tuple[int, int]:
        """Return `build_scaffold`'s pinned ``(feature_dim, num_relations)``.

        No throwaway probe is needed (unlike `EgoStitchImagineGenerator`): this
        generator calls `build_scaffold` directly and nothing about its output
        shape depends on a learned configuration.
        """
        return FEAT_DIM, EDGE_TYPES

