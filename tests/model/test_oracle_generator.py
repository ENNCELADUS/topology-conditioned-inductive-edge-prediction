"""Tests for the `oracle_struct` neighborhood generator.

Oracle-scaffold design,
`docs/superpowers/specs/2026-08-04-oracle-scaffold-experiment-design.md` §4.

Target module `src/model/egostitch/generator/oracle.py` is being written
concurrently with this test file (Wave 1). The whole file degrades to a clean
skip via `pytest.importorskip` below if it has not landed yet.

Contract pinned for this file (see the Wave-1 task brief; also confirmed by
the design doc):

- `build_oracle_table(graph, node_ids, *, slots, seed=0) -> OracleScaffoldTable`
  is a pure function over an `nx.Graph` -- fully specified, so the tests below
  hold it to an exact contract.
- `OracleStructGenerator` is zero-parameter and "[e]mits the same
  `ImaginedGraph` contract as `egostitch_imagine` (x/adj/mask/aux) built by
  the same `build_scaffold` structural-channel assembly" (design doc §4) --
  this is the load-bearing fact that licenses testing its `StitchedGraph`
  output against `generator/assemble.py`'s documented FEAT_DIM=11 channel
  layout (channel 4 = pi, channel 5 = mult; node order
  `[src, dst, slots_src(K), slots_dst(K)]`; EDGE_TYPES order star/intra/
  align/closure = 0/1/2/3) exactly as `generator/egostitch.py` and
  `generator/assemble.py` already document for the neural generator. This is
  an inference from the design doc's explicit "same build_scaffold assembly"
  statement, not a byte-level read of the (at-write-time-unwritten)
  implementation -- flagged in the final report.
- Leave-one-out masking happens at `stitch` time (design doc §3 R1 row), not
  inside `encode_node` (which is cacheable across many different pair
  partners and therefore cannot know the counterpart's identity). The
  differential tests below isolate this effect by holding one endpoint fixed
  across two `stitch` calls and varying only whether the *other* endpoint is
  the counterpart occupying one of the fixed endpoint's own slots.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

pytest.importorskip("src.model.egostitch.generator.oracle")

from src.data.ego_targets import EgoTargetBuilder
from src.model.egostitch.generator.egostitch import GeneratorNodeState, StitchedGraph
from src.model.egostitch.generator.oracle import (
    _ROW_ID_STRIDE,
    OracleScaffoldTable,
    OracleStructGenerator,
    _node_rng,
    build_oracle_table,
)

pytestmark = pytest.mark.unit

# EDGE_TYPES ordering documented in generator/assemble.py's module docstring
# ("star / intra-side slot-slot / alignment / closure") and re-affirmed by
# `_STAR, _INTRA, _ALIGN, _CLOSE = 0, 1, 2, 3` there (private names, not
# imported -- these are the same values, spelled out for this file's own
# readability).
_STAR, _INTRA, _ALIGN, _CLOSE = 0, 1, 2, 3
# `feats[:, :, 4] = cat([1, 1, pi_src, pi_dst])` / `feats[:, :, 5] = cat([1, 1,
# mult_src, mult_dst])` (generator/assemble.py `build_scaffold`) -- channel 4
# is existence (pi), channel 5 is multiplicity, for ANY generator that reuses
# `build_scaffold`, per this module's own docstring above.
_PI_CHANNEL = 4
_MULT_CHANNEL = 5

_SLOTS = 4


# --------------------------------------------------------------------------- build_oracle_table


def _tiny_graph() -> nx.Graph:
    """Hub (degree 6 > slots), isolate (degree 0), and a triangle wired to a hub neighbor."""
    g = nx.Graph()
    g.add_nodes_from(_NODE_IDS)
    g.add_edges_from(
        [
            ("h", "a"),
            ("h", "b"),
            ("h", "c"),
            ("h", "d"),
            ("h", "e"),
            ("h", "f"),
            ("t1", "t2"),
            ("t2", "t3"),
            ("t1", "t3"),
            ("t1", "a"),
        ]
    )
    return g


_NODE_IDS = ["h", "a", "b", "c", "d", "e", "f", "iso", "t1", "t2", "t3"]


def test_table_shapes_and_dtypes() -> None:
    table = build_oracle_table(_tiny_graph(), _NODE_IDS, slots=_SLOTS, seed=0)
    v = len(_NODE_IDS)
    assert table.neighbor_row.shape == (v, _SLOTS)
    assert table.mult.shape == (v, _SLOTS)
    assert table.adj.shape == (v, _SLOTS, _SLOTS)
    assert table.mask.shape == (v, _SLOTS)
    assert table.neighbor_row.dtype == torch.long
    assert table.mult.dtype == torch.float32
    assert table.adj.dtype == torch.float32
    assert table.mask.dtype == torch.bool


def test_isolate_row_is_fully_masked_and_padded() -> None:
    table = build_oracle_table(_tiny_graph(), _NODE_IDS, slots=_SLOTS, seed=0)
    row = _NODE_IDS.index("iso")
    assert not bool(table.mask[row].any())
    assert torch.all(table.neighbor_row[row] == -1)


def test_low_degree_row_selects_every_true_neighbor_and_pads_the_rest() -> None:
    """`t1` has exactly 3 true neighbors (t2, t3, a) and 4 slots: no truncation."""
    table = build_oracle_table(_tiny_graph(), _NODE_IDS, slots=_SLOTS, seed=0)
    row = _NODE_IDS.index("t1")
    valid = table.mask[row]
    assert int(valid.sum()) == 3
    selected_ids = {_NODE_IDS[i] for i in table.neighbor_row[row][valid].tolist()}
    assert selected_ids == {"t2", "t3", "a"}
    assert bool((table.mult[row][valid] >= 1.0).all())
    pad_idx = (~valid).nonzero(as_tuple=True)[0]
    assert torch.all(table.neighbor_row[row][pad_idx] == -1)


def test_true_sub_adjacency_matches_the_real_graph_edges() -> None:
    """`t1`'s selected neighbors {t2, t3, a}: only t2-t3 are actually adjacent."""
    g = _tiny_graph()
    table = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=0)
    row = _NODE_IDS.index("t1")
    valid_slots = table.mask[row].nonzero(as_tuple=True)[0].tolist()
    for si in valid_slots:
        for sj in valid_slots:
            if si == sj:
                continue
            u = _NODE_IDS[int(table.neighbor_row[row, si])]
            w = _NODE_IDS[int(table.neighbor_row[row, sj])]
            expected = 1.0 if g.has_edge(u, w) else 0.0
            assert table.adj[row, si, sj].item() == expected, (u, w)


def test_hub_row_truncates_to_exactly_slots_with_no_duplicate_picks() -> None:
    """`h` has 6 true neighbors > 4 slots: exactly `slots` are selected."""
    table = build_oracle_table(_tiny_graph(), _NODE_IDS, slots=_SLOTS, seed=0)
    row = _NODE_IDS.index("h")
    valid = table.mask[row]
    assert int(valid.sum()) == _SLOTS
    selected = table.neighbor_row[row][valid]
    assert selected.numel() == len(set(selected.tolist()))
    assert bool((table.mult[row][valid] >= 1.0).all())
    pad_idx = (~valid).nonzero(as_tuple=True)[0]
    assert torch.all(table.neighbor_row[row][pad_idx] == -1)


def test_same_seed_gives_bit_identical_tables() -> None:
    g = _tiny_graph()
    t1 = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=7)
    t2 = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=7)
    assert torch.equal(t1.neighbor_row, t2.neighbor_row)
    assert torch.equal(t1.mult, t2.mult)
    assert torch.equal(t1.adj, t2.adj)
    assert torch.equal(t1.mask, t2.mask)
    # The hub-edge override rows must be just as bit-identical as the base
    # table -- they are read every stitch call on every DDP rank.
    assert torch.equal(t1.override_key, t2.override_key)
    assert torch.equal(t1.override_neighbor_row, t2.override_neighbor_row)
    assert torch.equal(t1.override_mult, t2.override_mult)
    assert torch.equal(t1.override_adj, t2.override_adj)
    assert torch.equal(t1.override_mask, t2.override_mask)


def _decode_override_rows(
    table: OracleScaffoldTable, node_ids: list[str], row_of: dict[str, int]
) -> dict[tuple[str, str], frozenset[tuple[str, float]]]:
    """Turn a table's override rows into an id-space mapping for order-independent comparison.

    Maps ``(hub_node_id, excluded_neighbor_id) -> frozenset of (selected_id, mult)``,
    stripping away the row-index numbering that a `node_ids` permutation changes.
    """
    id_of = {row: node for node, row in row_of.items()}
    decoded: dict[tuple[str, str], frozenset[tuple[str, float]]] = {}
    for i in range(table.override_key.shape[0]):
        key = int(table.override_key[i].item())
        own_row, partner_row = divmod(key, _ROW_ID_STRIDE)
        own_id = node_ids[own_row]
        partner_id = id_of[partner_row]
        valid = table.override_mask[i]
        selected = frozenset(
            (id_of[int(r)], float(m))
            for r, m, keep in zip(
                table.override_neighbor_row[i].tolist(),
                table.override_mult[i].tolist(),
                valid.tolist(),
                strict=True,
            )
            if keep
        )
        decoded[(own_id, partner_id)] = selected
    return decoded


def test_override_rows_are_independent_of_node_ids_order() -> None:
    """Reordering `node_ids` permutes row numbers but must not change any override row's content.

    `_node_rng` is seeded purely from `(node_id, seed)`, never from a row
    index or draw order, so every override row is a pure function of
    `(hub_id, excluded_id, seed)` -- this is the same order-independence
    `test_non_hub_rows_are_seed_invariant` already pins for the base table,
    extended to the hub-edge overrides the P1 fix adds.
    """
    g = _tiny_graph()
    forward = list(_NODE_IDS)
    reversed_ids = list(reversed(_NODE_IDS))

    t_forward = build_oracle_table(g, forward, slots=_SLOTS, seed=11)
    t_reversed = build_oracle_table(g, reversed_ids, slots=_SLOTS, seed=11)

    row_of_forward = {node: i for i, node in enumerate(forward)}
    row_of_reversed = {node: i for i, node in enumerate(reversed_ids)}

    decoded_forward = _decode_override_rows(t_forward, forward, row_of_forward)
    decoded_reversed = _decode_override_rows(t_reversed, reversed_ids, row_of_reversed)

    assert decoded_forward, "test fixture expected to produce at least one override row"
    assert decoded_forward == decoded_reversed


def test_non_hub_rows_are_seed_invariant() -> None:
    """Rows with |N(u)| <= slots need no tie-breaking, so the seed cannot matter."""
    g = _tiny_graph()
    t_a = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=0)
    t_b = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=999)
    for node in ("t1", "t2", "t3", "a", "iso", "b", "c", "d", "e", "f"):
        row = _NODE_IDS.index(node)
        assert torch.equal(t_a.neighbor_row[row], t_b.neighbor_row[row]), node
        assert torch.equal(t_a.mask[row], t_b.mask[row]), node


def test_different_seed_can_change_hub_truncation() -> None:
    """Across 10 seeds, the hub's 6-choose-4 tie-break should not be a constant."""
    g = _tiny_graph()
    row = _NODE_IDS.index("h")
    seen = set()
    for seed in range(10):
        table = build_oracle_table(g, _NODE_IDS, slots=_SLOTS, seed=seed)
        seen.add(tuple(sorted(table.neighbor_row[row][table.mask[row]].tolist())))
    assert len(seen) > 1, "hub truncation never varied across 10 seeds"


# --------------------------------------------------------------------------- OracleStructGenerator


def _build_generator_and_table(
    graph: nx.Graph, node_ids: list[str], *, slots: int = _SLOTS, seed: int = 0
) -> tuple[OracleStructGenerator, OracleScaffoldTable]:
    generator = OracleStructGenerator(slots=slots)
    table = build_oracle_table(graph, node_ids, slots=slots, seed=seed)
    # Identity map sidesteps any ambiguity in whether `node_rows` passed to
    # `encode_node` is a raw F0 row id needing translation or already a table
    # row -- both readings coincide here.
    f0_row_to_table_row = torch.arange(len(node_ids), dtype=torch.long)
    generator.set_oracle_context(table, f0_row_to_table_row)
    return generator, table


def _encode(
    generator: OracleStructGenerator, rows: list[int], *, feature_dim: int = 8, n_ground: int = 3
) -> GeneratorNodeState:
    batch = len(rows)
    x = torch.zeros(batch, feature_dim)
    ground = torch.zeros(batch, n_ground, feature_dim)
    node_rows = torch.tensor(rows, dtype=torch.long)
    return generator.encode_node(x, ground, node_rows=node_rows)


def test_zero_parameters() -> None:
    generator = OracleStructGenerator(slots=_SLOTS)
    assert sum(p.numel() for p in generator.parameters()) == 0


def test_graph_dims_matches_the_shared_scaffold_layout() -> None:
    generator = OracleStructGenerator(slots=_SLOTS)
    assert generator.graph_dims() == (11, 4)


def test_auxiliary_losses_is_always_empty() -> None:
    generator = OracleStructGenerator(slots=_SLOTS)
    assert generator.auxiliary_losses(None, {}) == {}


def test_registered_under_oracle_struct_if_registry_updated() -> None:
    """Soft check: `registry.py`'s side of this wave is owned by another agent."""
    from src.model.egostitch.registry import GENERATOR_REGISTRY

    if "oracle_struct" not in GENERATOR_REGISTRY:
        pytest.skip("GENERATOR_REGISTRY['oracle_struct'] not yet wired in registry.py")
    assert GENERATOR_REGISTRY["oracle_struct"] is OracleStructGenerator


def test_non_adjacent_pair_leaves_both_sides_fully_unmasked() -> None:
    """`t1` and `b` share no edge and no leave-one-out condition triggers."""
    g = _tiny_graph()
    generator, table = _build_generator_and_table(g, _NODE_IDS)
    row_t1, row_b = _NODE_IDS.index("t1"), _NODE_IDS.index("b")
    state_t1 = _encode(generator, [row_t1])
    state_b = _encode(generator, [row_b])
    graph = generator.stitch(state_t1, state_b, torch.zeros(1, dtype=torch.bool))
    k = _SLOTS
    src_pi_nonzero = int((graph.x[0, 2 : 2 + k, _PI_CHANNEL] != 0).sum())
    dst_pi_nonzero = int((graph.x[0, 2 + k : 2 + 2 * k, _PI_CHANNEL] != 0).sum())
    assert src_pi_nonzero == int(table.mask[row_t1].sum())
    assert dst_pi_nonzero == int(table.mask[row_b].sum())


def test_leave_one_out_zeroes_the_dst_side_slot_holding_the_src_counterpart() -> None:
    """Isolate dst-side masking: fix dst="a", vary src between adjacent/not.

    `a`'s true neighbors are {h, t1}. Pairing `a` with `f` (not adjacent to
    `a`) is the control; pairing `a` with `h` (adjacent to `a`) must zero
    exactly `a`'s own slot that holds `h`, and its intra-slot adjacency row
    and column, while every other dst-side slot stays identical between the
    two calls.
    """
    g = _tiny_graph()
    generator, table = _build_generator_and_table(g, _NODE_IDS)
    row_a, row_f, row_h = _NODE_IDS.index("a"), _NODE_IDS.index("f"), _NODE_IDS.index("h")
    state_a = _encode(generator, [row_a])
    state_f = _encode(generator, [row_f])
    state_h = _encode(generator, [row_h])

    graph_control = generator.stitch(state_f, state_a, torch.zeros(1, dtype=torch.bool))
    graph_test = generator.stitch(state_h, state_a, torch.zeros(1, dtype=torch.bool))

    k = _SLOTS
    dst_slice = slice(2 + k, 2 + 2 * k)
    valid = table.mask[row_a]
    i_h = int((table.neighbor_row[row_a] == row_h).nonzero(as_tuple=True)[0].item())
    assert valid[i_h], "test setup assumes h is a selected (non-padded) neighbor of a"

    control_pi = graph_control.x[0, dst_slice, _PI_CHANNEL]
    test_pi = graph_test.x[0, dst_slice, _PI_CHANNEL]

    assert control_pi[i_h].item() != 0.0, "control baseline must be unmasked at slot i_h"
    assert test_pi[i_h].item() == 0.0, "leave-one-out must zero a's own slot holding h"

    unaffected = torch.ones(k, dtype=torch.bool)
    unaffected[i_h] = False
    assert torch.equal(control_pi[unaffected], test_pi[unaffected])

    control_intra = graph_control.adj[0, _INTRA, dst_slice, dst_slice]
    test_intra = graph_test.adj[0, _INTRA, dst_slice, dst_slice]
    assert torch.all(test_intra[i_h, :] == 0.0)
    assert torch.all(test_intra[:, i_h] == 0.0)
    assert torch.equal(
        control_intra[unaffected][:, unaffected], test_intra[unaffected][:, unaffected]
    )


def test_leave_one_out_zeroes_the_src_side_slot_holding_the_dst_counterpart() -> None:
    """Symmetric counterpart of the dst-side test: fix src="t1", vary dst."""
    g = _tiny_graph()
    generator, table = _build_generator_and_table(g, _NODE_IDS)
    row_t1, row_b, row_a = _NODE_IDS.index("t1"), _NODE_IDS.index("b"), _NODE_IDS.index("a")
    state_t1 = _encode(generator, [row_t1])
    state_b = _encode(generator, [row_b])
    state_a = _encode(generator, [row_a])

    graph_control = generator.stitch(state_t1, state_b, torch.zeros(1, dtype=torch.bool))
    graph_test = generator.stitch(state_t1, state_a, torch.zeros(1, dtype=torch.bool))

    k = _SLOTS
    src_slice = slice(2, 2 + k)
    valid = table.mask[row_t1]
    i_a = int((table.neighbor_row[row_t1] == row_a).nonzero(as_tuple=True)[0].item())
    assert valid[i_a], "test setup assumes a is a selected (non-padded) neighbor of t1"

    control_pi = graph_control.x[0, src_slice, _PI_CHANNEL]
    test_pi = graph_test.x[0, src_slice, _PI_CHANNEL]
    assert control_pi[i_a].item() != 0.0
    assert test_pi[i_a].item() == 0.0

    unaffected = torch.ones(k, dtype=torch.bool)
    unaffected[i_a] = False
    assert torch.equal(control_pi[unaffected], test_pi[unaffected])


def _hub_replacement_graph() -> nx.Graph:
    """Hub `u` with 5 single-degree neighbors and 3 slots.

    Every neighbor's only edge is to `u`, so all five sit in the same
    log2-degree stratum -- excluding any one of them is pure "K of the
    remaining N-1" resampling, with no cross-stratum reallocation to
    complicate the arithmetic. `deg(u) - 1 == 4 > slots == 3`, so excluding
    *any* neighbor -- sampled or not -- still requires a fresh subsample
    rather than a full pass-through, which is what makes this fixture
    decisive for the "only K-1 slots survive, no replacement drawn" defect.
    """
    g = nx.Graph()
    g.add_edges_from([("u", f"n{i}") for i in range(5)])
    return g


_HUB_NODE_IDS = ["u", "n0", "n1", "n2", "n3", "n4"]
_HUB_SLOTS = 3
# Picked (by direct computation against the fixture, not by exhaustive
# search) so `u`'s base K-sample is a strict 3-of-5 subset -- the seed value
# has no other significance.
_HUB_SEED = 3


def test_hub_leave_one_out_matches_ego_target_builder_exactly() -> None:
    """Decisive parity test for the P1 hub leave-one-out defect.

    `EgoTargetBuilder._select_targets(u, rng, exclude_neighbor=v)` removes
    `v` from `u`'s neighbor pool *before* stratifying and subsampling
    (`src/data/ego_targets.py`). For a hub (`deg(u) > K`), a generator that
    instead subsamples once and then merely zeroes whichever slot happens to
    hold `v` produces a *different* answer whenever `v` was one of the
    sampled slots: it leaves the scaffold `K - 1` slots full with a stale
    multiplicity label, instead of drawing a replacement from the
    partner-excluded pool with recomputed multiplicities. This test pins the
    oracle's emitted scaffold to the exact `_select_targets` output, not to
    "the base sample minus one slot".
    """
    g = _hub_replacement_graph()
    generator, table = _build_generator_and_table(
        g, _HUB_NODE_IDS, slots=_HUB_SLOTS, seed=_HUB_SEED
    )
    row_u = _HUB_NODE_IDS.index("u")
    base_valid = table.mask[row_u]
    assert int(base_valid.sum()) == _HUB_SLOTS, "test setup expects u's base row truncated to K"
    base_selected = {_HUB_NODE_IDS[i] for i in table.neighbor_row[row_u][base_valid].tolist()}

    # `v` is a neighbor the *base* K-sample actually selected -- exactly the
    # regime `_drop_partner_slots` alone gets wrong (it has a slot to zero,
    # and stops there instead of refilling from the excluded-pool resample).
    v = sorted(base_selected)[0]
    row_v = _HUB_NODE_IDS.index(v)
    assert v not in {"u"}

    state_u = _encode(generator, [row_u])
    state_v = _encode(generator, [row_v])
    graph = generator.stitch(state_u, state_v, torch.zeros(1, dtype=torch.bool))

    got_pointer = graph.aux["slots_a_pointer"][0, :, 0]
    got_pi = graph.aux["slots_a_pi"][0]
    got_mult = graph.aux["slots_a_mult"][0]
    got: dict[str, float] = {}
    for slot in range(_HUB_SLOTS):
        if got_pi[slot].item() != 0.0:
            neighbor_id = _HUB_NODE_IDS[int(round(got_pointer[slot].item()))]
            got[neighbor_id] = got_mult[slot].item()

    node_index = {node: index for index, node in enumerate(g.nodes())}
    builder = EgoTargetBuilder(
        g,
        np.zeros((g.number_of_nodes(), 1), dtype=np.float32),
        node_index,
        {},
        slots=_HUB_SLOTS,
    )
    expected_selected, expected_mult = builder._select_targets(
        "u", _node_rng("u", _HUB_SEED), exclude_neighbor=v
    )
    expected = dict(zip(expected_selected, expected_mult, strict=True))

    # Decisive assertions: exact set/multiplicity match against the canonical
    # builder, all K slots filled (a replacement was drawn), and the dropped
    # partner is genuinely gone -- not merely a mult=1.667-labelled 2-of-3
    # remnant of the never-excluded base sample.
    assert got.keys() == expected.keys(), (got, expected)
    for neighbor_id, mult in expected.items():
        assert got[neighbor_id] == pytest.approx(mult), (neighbor_id, got, expected)
    assert len(got) == _HUB_SLOTS
    assert v not in got


def test_self_pair_yields_an_unconditional_identity_alignment_plan() -> None:
    """`is_self=True`: the plan must be exactly `eye(slots)`.

    Matches the "identity contract" `EgoStitchImagineGenerator.stitch`
    enforces explicitly for self rows (`generator/egostitch.py`'s
    `self_rows` branch sets ``plan = eye(num_slots)`` unconditionally --
    diagonal 1.0 for *every* slot, including padded ones, never gated by
    `pi`/`mask`).
    """
    g = _tiny_graph()
    generator, _ = _build_generator_and_table(g, _NODE_IDS)
    row_t1 = _NODE_IDS.index("t1")
    state = _encode(generator, [row_t1])
    graph = generator.stitch(state, state, torch.ones(1, dtype=torch.bool))
    plan = graph.aux["plan"][0]
    torch.testing.assert_close(plan, torch.eye(_SLOTS))


def _shared_neighbor_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(
        [
            ("u", "n1"),
            ("u", "n2"),
            ("u", "only_u"),
            ("v", "n1"),
            ("v", "n2"),
            ("v", "only_v"),
        ]
    )
    return g


_SHARED_NODE_IDS = ["u", "v", "n1", "n2", "only_u", "only_v"]


def test_identity_plan_matches_exactly_the_shared_neighbor_slots() -> None:
    """The alignment plan must match exactly at shared-neighbor slot pairs.

    `u` and `v` are not adjacent but share neighbors n1, n2 (degree 3 <= 4
    slots each, so no hub truncation and no seed dependence). The plan must
    be nonzero exactly where a slot pair holds the same real node, with
    magnitude pi_a[i] * pi_b[j] there, and exactly zero (with -inf log-plan)
    everywhere else.
    """
    g = _shared_neighbor_graph()
    generator, table = _build_generator_and_table(g, _SHARED_NODE_IDS)
    row_u, row_v = _SHARED_NODE_IDS.index("u"), _SHARED_NODE_IDS.index("v")
    state_u = _encode(generator, [row_u])
    state_v = _encode(generator, [row_v])
    graph = generator.stitch(state_u, state_v, torch.zeros(1, dtype=torch.bool))

    k = _SLOTS
    plan = graph.aux["plan"][0]
    log_plan = graph.aux["log_plan"][0]
    src_pi = graph.x[0, 2 : 2 + k, _PI_CHANNEL]
    dst_pi = graph.x[0, 2 + k : 2 + 2 * k, _PI_CHANNEL]

    for i in range(k):
        nid_u = int(table.neighbor_row[row_u, i].item())
        for j in range(k):
            nid_v = int(table.neighbor_row[row_v, j].item())
            if nid_u != -1 and nid_u == nid_v:
                expected = (src_pi[i] * dst_pi[j]).item()
                assert plan[i, j].item() == pytest.approx(expected), (i, j)
                assert plan[i, j].item() > 0.0
                assert torch.isfinite(log_plan[i, j])
            else:
                assert plan[i, j].item() == 0.0, (i, j, nid_u, nid_v)
                assert torch.isneginf(log_plan[i, j])


def test_swapped_round_trips_and_shares_adjacency_by_reference() -> None:
    g = _tiny_graph()
    generator, _ = _build_generator_and_table(g, _NODE_IDS)
    row_t1, row_b = _NODE_IDS.index("t1"), _NODE_IDS.index("b")
    state_t1 = _encode(generator, [row_t1])
    state_b = _encode(generator, [row_b])
    graph = generator.stitch(state_t1, state_b, torch.zeros(1, dtype=torch.bool))
    assert isinstance(graph, StitchedGraph)

    once = graph.swapped()
    assert once.adj is graph.adj
    assert once.directed != graph.directed
    twice = once.swapped()
    assert torch.equal(twice.x, graph.x)
    assert torch.equal(twice.adj, graph.adj)


def test_cache_round_trip_matches_direct_stitch_if_node_state_is_a_generator_node_state() -> None:
    """Confirm the shared per-node cache helpers round-trip this generator's state.

    Push `encode_node`'s output through the shared per-node cache helpers
    every other generator's caller (`score_universe.py`, `train_egostitch.py`)
    relies on, and confirm the reassembled state stitches identically to the
    direct path -- the same property
    `test_generator_component.py::
    test_generator_node_state_survives_index_select_index_copy_and_restack`
    proves for `EgoStitchImagineGenerator`.

    This can only be exercised if `OracleStructGenerator.encode_node` returns
    an actual `GeneratorNodeState` (a real `SlotSet`) -- the design doc
    confirms `build_scaffold` (which requires `SlotSet`-shaped inputs) is
    reused, which is highly suggestive but not a guarantee about the
    generator's own cacheable `NodeStateT`. If it returns something else, the
    hardcoded-to-`SlotSet` cache helpers
    (`_fp32_cached_node_state`/`_stack_cached_node_states`,
    `src/train_egostitch.py:2960-3049`, referenced by the task brief as
    living in `composite.py`) cannot consume it without further changes; this
    test skips cleanly in that case and reports what it found, rather than
    asserting a shape it cannot verify from the pinned contract alone.
    """
    try:
        from src.model.egostitch.composite import (  # type: ignore[attr-defined]
            E2ENodeState,
            _fp32_cached_node_state,
            _stack_cached_node_states,
        )
    except ImportError:
        try:
            from src.train_egostitch import (  # type: ignore[attr-defined]
                E2ENodeState,
                _fp32_cached_node_state,
                _stack_cached_node_states,
            )
        except ImportError:
            pytest.skip(
                "_fp32_cached_node_state/_stack_cached_node_states/E2ENodeState "
                "not importable from composite.py or train_egostitch.py"
            )

    g = _tiny_graph()
    generator, _ = _build_generator_and_table(g, _NODE_IDS)
    rows = {"t1": _NODE_IDS.index("t1"), "a": _NODE_IDS.index("a"), "b": _NODE_IDS.index("b")}
    states = {name: _encode(generator, [row]) for name, row in rows.items()}

    if not all(isinstance(state, GeneratorNodeState) for state in states.values()):
        pytest.skip(
            "OracleStructGenerator.encode_node does not return a GeneratorNodeState; "
            f"got {type(next(iter(states.values())))!r} -- the SlotSet-hardcoded cache "
            "helpers do not apply to this generator's NodeStateT as written"
        )

    length = 3
    cache: dict[str, object] = {}
    for name, state in states.items():
        wrapped = E2ENodeState(
            encoded=torch.randn(1, length, 8),
            length=torch.tensor([length]),
            slots=state.slots,
            projected_x=state.projected_x,
            ground_ids=state.ground_ids,
        )
        cache[name] = _fp32_cached_node_state(wrapped, 0)

    stacked_a = _stack_cached_node_states(cache, ["t1", "t1"])
    stacked_b = _stack_cached_node_states(cache, ["a", "b"])
    generator_state_a = GeneratorNodeState(
        slots=stacked_a.slots, projected_x=stacked_a.projected_x, ground_ids=stacked_a.ground_ids
    )
    generator_state_b = GeneratorNodeState(
        slots=stacked_b.slots, projected_x=stacked_b.projected_x, ground_ids=stacked_b.ground_ids
    )
    is_self = torch.zeros(2, dtype=torch.bool)
    cached_graph = generator.stitch(generator_state_a, generator_state_b, is_self)

    direct_state_t1_a = _encode(generator, [rows["t1"]])
    direct_state_t1_b = _encode(generator, [rows["t1"]])
    direct_state_a = _encode(generator, [rows["a"]])
    direct_state_b = _encode(generator, [rows["b"]])
    direct_graph_row0 = generator.stitch(
        direct_state_t1_a, direct_state_a, torch.zeros(1, dtype=torch.bool)
    )
    direct_graph_row1 = generator.stitch(
        direct_state_t1_b, direct_state_b, torch.zeros(1, dtype=torch.bool)
    )

    torch.testing.assert_close(cached_graph.x[0:1], direct_graph_row0.x)
    torch.testing.assert_close(cached_graph.x[1:2], direct_graph_row1.x)
    torch.testing.assert_close(cached_graph.adj[0:1], direct_graph_row0.adj)
    torch.testing.assert_close(cached_graph.adj[1:2], direct_graph_row1.adj)
