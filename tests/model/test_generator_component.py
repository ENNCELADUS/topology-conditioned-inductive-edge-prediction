"""Tests for the neighborhood-generator component (three-component refactor design §3.3, §5).

The key test is numerical equivalence with the live `EgoStitchModel` path:
capturing the exact `ScaffoldTokens` `build_pair_context_from_states` builds
internally and checking `EgoStitchImagineGenerator` reproduces it to within
floating-point equivalence for the same weights and inputs is the proof this
is a behaviour-preserving port, not a reimplementation that happens to look
similar. The comparison is `assert_close`, not bit-identity: the two routes
order their ops differently, so the last ulp tracks the platform's BLAS/FMA
contraction (see `_EQUIV_RTOL` below).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.testing
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import (
    ClassifierConfig,
    E2EConfig,
    EgoStitchConfig,
    EncoderConfig,
    GeneratorConfig,
)
from src.model.egostitch.generator import (
    EgoStitchImagineGenerator,
    GeneratorNodeState,
    NeighborhoodGenerator,
    NullGenerator,
    StitchedGraph,
)
from src.model.egostitch.generator.assemble import (
    EDGE_TYPES,
    FEAT_DIM,
    make_scaffold_input_perturbation,
    match_slots,
)
from src.model.egostitch.generator.egostitch import _slotset_from_aux
from src.model.egostitch.generator.imagine import FeatureStandardizationMode, SlotSet
from src.model.egostitch.generator.losses import alignment_loss, alignment_teacher_cells

pytestmark = pytest.mark.unit

# Component-vs-live equivalence tolerance; see the identically named constants
# in `tests/model/test_classifier_component.py` for the full rationale. Kept at
# 1e-5 so the assertion is portable between Apple silicon and the x86 H20
# container while staying far tighter than any real regression.
_EQUIV_RTOL = 1e-5
_EQUIV_ATOL = 1e-5

_TINY = EgoStitchConfig(
    input_dim=8,
    d_p=4,
    d_z=4,
    d_h=8,
    slots=4,
    m_max=8,
    n_ground=3,
    decoder_layers=2,
    n_heads=2,
    gin_hidden=8,
    gin_layers=2,
    sinkhorn_iters=5,
)


def _tiny_generator(*, seed: int = 0) -> EgoStitchImagineGenerator:
    torch.manual_seed(seed)
    return EgoStitchImagineGenerator(_TINY, feature_standardization="row_layernorm")


def _tiny_node(*, seed: int, n_ground: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, _TINY.input_dim, generator=gen)
    ground = torch.randn(1, n_ground, _TINY.input_dim, generator=gen)
    return x, ground


def _tiny_endpoints(
    batch: int = 4, *, n_ground: int = 3, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x_a = torch.randn(batch, _TINY.input_dim, generator=gen)
    x_b = torch.randn(batch, _TINY.input_dim, generator=gen)
    ground_a = torch.randn(batch, n_ground, _TINY.input_dim, generator=gen)
    ground_b = torch.randn(batch, n_ground, _TINY.input_dim, generator=gen)
    return x_a, x_b, ground_a, ground_b


def _node_stream_batch(batch: int = 4, *, seed: int = 7) -> dict[str, torch.Tensor]:
    """A node-stream batch shaped like `EgoStitchStage1.node_losses`/`ssl_losses` expect."""
    gen = torch.Generator().manual_seed(seed)
    t = _TINY.slots
    target_in_pool = torch.rand(batch, t, generator=gen) > 0.5
    return {
        "x": torch.randn(batch, _TINY.input_dim, generator=gen),
        "ground_x": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "target_features": torch.randn(batch, t, _TINY.input_dim, generator=gen),
        "target_mult": 1.0 + torch.rand(batch, t, generator=gen),
        "target_adj": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
        "target_mask": torch.ones(batch, t, dtype=torch.bool),
        "target_in_pool": target_in_pool,
        "target_pool_index": torch.where(
            target_in_pool,
            torch.arange(t).remainder(_TINY.n_ground).expand(batch, -1),
            torch.full((batch, t), -1),
        ),
        "true_degree": torch.randint(1, 6, (batch,), generator=gen).float(),
        "real_ego_stats": torch.rand(batch, 4, generator=gen),
        "ground_resampled": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "ssl_noise": 0.05 * torch.randn(batch, _TINY.input_dim, generator=gen),
    }


def _edge_align_batch(batch: int = 4, *, seed: int = 9) -> dict[str, torch.Tensor]:
    """An edge-stream batch shaped like Task-7 `L_align` supervision expects."""
    gen = torch.Generator().manual_seed(seed)
    t = _TINY.slots
    return {
        "target_features_a": torch.randn(batch, t, _TINY.input_dim, generator=gen),
        "target_features_b": torch.randn(batch, t, _TINY.input_dim, generator=gen),
        "target_mult_a": 1.0 + torch.rand(batch, t, generator=gen),
        "target_mult_b": 1.0 + torch.rand(batch, t, generator=gen),
        "target_adj_a": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
        "target_adj_b": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
        "target_mask_a": torch.ones(batch, t, dtype=torch.bool),
        "target_mask_b": torch.ones(batch, t, dtype=torch.bool),
        "target_node_index_a": torch.randint(0, 100, (batch, t), generator=gen),
        "target_node_index_b": torch.randint(0, 100, (batch, t), generator=gen),
        "label": torch.randint(0, 2, (batch,), generator=gen),
        "edge_mask": torch.ones(batch),
    }


# --------------------------------------------------------------------------- shape / aux contract


def test_stitched_graph_shapes_and_aux_contents() -> None:
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.tensor([False, False, False, False])

    graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)

    n_nodes = 2 + 2 * _TINY.slots
    assert isinstance(graph, StitchedGraph)
    assert graph.x.shape == (4, n_nodes, FEAT_DIM)
    assert graph.adj.shape == (4, EDGE_TYPES, n_nodes, n_nodes)
    assert graph.mask.shape == (4, n_nodes)
    assert graph.num_nodes == n_nodes
    assert graph.num_relations == EDGE_TYPES
    assert graph.feature_dim == FEAT_DIM
    assert graph.directed is True
    # mask is exactly the node-existence channel smuggled into x (graph.py contract).
    assert torch.equal(graph.mask, graph.x[..., 4])

    slot_shapes = {
        "h": (4, _TINY.slots, _TINY.d_p),
        "pi": (4, _TINY.slots),
        "mult": (4, _TINY.slots),
        "gate": (4, _TINY.slots),
        "pointer": (4, _TINY.slots, _TINY.n_ground),
        "adj": (4, _TINY.slots, _TINY.slots),
        "adj_logits": (4, _TINY.slots, _TINY.slots),
    }
    for prefix in ("slots_a", "slots_b"):
        for field, shape in slot_shapes.items():
            key = f"{prefix}_{field}"
            assert key in graph.aux, key
            assert graph.aux[key].shape == shape, key
    assert graph.aux["plan"].shape == (4, _TINY.slots, _TINY.slots)
    assert graph.aux["log_plan"].shape == (4, _TINY.slots, _TINY.slots)


# --------------------------------------------------------------------------- swapped()


def test_swapped_is_an_involution_and_shares_state_by_reference() -> None:
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.tensor([True, False, True, False])

    graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)
    swapped = graph.swapped()

    assert swapped.adj is graph.adj
    assert swapped.mask is graph.mask
    assert swapped.aux is graph.aux
    assert swapped.directed is False
    assert not torch.equal(swapped.x, graph.x)  # anchor channels were actually relabelled

    twice = swapped.swapped()
    assert twice.adj is graph.adj
    assert twice.directed is True
    assert torch.equal(twice.x, graph.x)


# --------------------------------------------------------------------------- null generator


def test_null_generator_returns_none_and_no_losses() -> None:
    generator = NullGenerator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.zeros(4, dtype=torch.bool)

    assert generator(x_a, x_b, ground_a, ground_b, is_self=is_self) is None
    assert generator.auxiliary_losses(None, {}) == {}
    assert isinstance(generator, NeighborhoodGenerator)


def test_neighborhood_generator_is_abstract() -> None:
    with pytest.raises(TypeError):
        NeighborhoodGenerator()  # type: ignore[abstract]


# --------------------------------------------------------------------------- self-pair identity


def test_self_pair_rows_get_the_identity_plan_without_running_sinkhorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`stitch` never runs Sinkhorn on a self-pair row -- the plan is forced identity.

    (The complementary "B is never re-encoded for a self row" optimization
    now lives in the *caller*'s per-node cache -- a self pair reuses the
    same cache entry for both endpoints because it is the same node id --
    not inside the generator itself; see
    `test_encode_node_once_then_stitch_against_several_counterparts_matches_fused_forward`.)
    """
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.ones(4, dtype=torch.bool)
    state_a = generator.encode_node(x_a, ground_a)
    state_b = generator.encode_node(x_b, ground_b)

    import src.model.egostitch.generator.egostitch as generator_module

    sinkhorn_calls = 0
    original_sinkhorn = cast(Callable[..., torch.Tensor], generator_module.sinkhorn_log_plan)

    def counted_sinkhorn(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        nonlocal sinkhorn_calls
        sinkhorn_calls += 1
        return original_sinkhorn(*args, **kwargs)

    monkeypatch.setattr(generator_module, "sinkhorn_log_plan", counted_sinkhorn)

    graph = generator.stitch(state_a, state_b, is_self)

    assert sinkhorn_calls == 0
    expected = torch.eye(_TINY.slots).expand(4, -1, -1)
    assert torch.equal(graph.aux["plan"], expected)


def test_fp32_island_preserved_through_sinkhorn_under_bf16_autocast() -> None:
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.tensor([True, False, True, False])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)

    assert graph.aux["plan"].dtype == torch.float32
    assert graph.aux["log_plan"].dtype == torch.float32
    assert graph.x.dtype == torch.float32
    assert graph.adj.dtype == torch.float32
    expected_identity = torch.eye(_TINY.slots)
    assert torch.equal(graph.aux["plan"][is_self], expected_identity.expand(2, -1, -1))


# ----------------------------------------------------------------------- equivalence w/ e2e path


def test_matches_live_e2e_build_pair_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves `EgoStitchImagineGenerator` is a behaviour-preserving port.

    Intercepts the exact `ScaffoldTokens` `EgoStitchModel.build_pair_context`
    builds internally (via `build_pair_context_from_states` ->
    `generator.stitch` -> `build_scaffold`) and checks a standalone generator
    sharing the same live weights reproduces it bit-for-bit for the same
    inputs, including a mixed self/non-self batch so both the identity-plan
    short-circuit and the Sinkhorn path are exercised.
    """
    torch.manual_seed(0)
    cfg = E2EConfig(
        generator=GeneratorConfig(feature_standardization="row_layernorm"),
        encoder=EncoderConfig(dim=8, layers=1),
        classifier=ClassifierConfig(
            d_model=16,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=2,
            n_inj=1,
            xattn_heads=2,
        ),
    )
    model = EgoStitchModel(cfg).eval()

    import src.model.egostitch.generator.egostitch as generator_module

    captured: dict[str, object] = {}
    original_build_scaffold = generator_module.build_scaffold

    def capturing_build_scaffold(*args: object, **kwargs: object) -> object:
        result = original_build_scaffold(*args, **kwargs)  # type: ignore[arg-type]
        captured["scaffold"] = result
        return result

    monkeypatch.setattr(generator_module, "build_scaffold", capturing_build_scaffold)

    b, n_ground = 4, 5
    x_a = torch.randn(b, model.node_feature_dim)
    x_b = torch.randn(b, model.node_feature_dim)
    ground_a = torch.randn(b, n_ground, model.node_feature_dim)
    ground_b = torch.randn(b, n_ground, model.node_feature_dim)
    is_self = torch.tensor([True, False, True, False])
    # The production data contract guarantees x_a == x_b (and hence the same
    # grounding pool) on self-pair rows; the live path never even encodes B
    # there (`_pair_node_states` reuses A's own slots), while the new
    # two-phase generator always encodes both endpoints independently, so
    # this equality is what makes the two paths agree numerically on those
    # rows -- not an accident of a shared cache.
    x_b = torch.where(is_self[:, None], x_a, x_b)
    ground_b = torch.where(is_self[:, None, None], ground_a, ground_b)

    batch = {
        "emb_a": torch.randn(b, 6, model.input_dim),
        "emb_b": torch.randn(b, 6, model.input_dim),
        "len_a": torch.full((b,), 6, dtype=torch.long),
        "len_b": torch.full((b,), 6, dtype=torch.long),
        "x_a": x_a,
        "x_b": x_b,
        "ground_a": ground_a,
        "ground_b": ground_b,
        "is_self": is_self,
    }
    context = model.build_pair_context(batch)
    assert "scaffold" in captured
    live_scaffold = captured["scaffold"]
    assert context.plan is not None
    assert context.log_plan is not None

    generator = EgoStitchImagineGenerator(
        model.generator_cfg,
        # `GeneratorConfig.feature_standardization` is a runtime-validated
        # plain `str` (`config.py`'s `__post_init__`), not the narrower
        # `FeatureStandardizationMode` literal `EgoStitchImagineGenerator`
        # expects; same cast `registry.build_generator` uses for this exact
        # value.
        feature_standardization=cast(
            FeatureStandardizationMode, cfg.generator.feature_standardization
        ),
        loss_family="egostitch_e2e",
    )
    # `model.generator`'s static type is the `NeighborhoodGenerator` ABC now
    # that construction is registry-driven (design §12 P3); narrow to the
    # concrete class this test always builds to reach `.stage1`.
    assert isinstance(model.generator, EgoStitchImagineGenerator)
    generator.stage1 = model.generator.stage1  # share the exact live weights
    generator.eval()

    graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)

    # Numerically equivalent, not bit-identical: the standalone component and
    # the live model reach the same scaffold by different op orders, so the
    # last ulp depends on the platform's BLAS/FMA contraction. `torch.equal`
    # held on Apple silicon and failed on the x86 H20 container (verified
    # failing at the pre-Wave-1 base `73d44ba` -- a portability defect in the
    # assertion, not a regression in the generator). `assert_close` compares
    # `-inf` cells in `log_plan` as equal, which is what the alignment loss
    # requires of an impossible plan cell.
    torch.testing.assert_close(
        graph.x,  # type: ignore[attr-defined]
        live_scaffold.feats,
        rtol=_EQUIV_RTOL,
        atol=_EQUIV_ATOL,
    )
    torch.testing.assert_close(
        graph.adj,  # type: ignore[attr-defined]
        live_scaffold.adj,
        rtol=_EQUIV_RTOL,
        atol=_EQUIV_ATOL,
    )
    torch.testing.assert_close(graph.aux["plan"], context.plan, rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL)
    torch.testing.assert_close(
        graph.aux["log_plan"], context.log_plan, rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL
    )


# --------------------------------------------------------------------------- two-phase caching


def test_encode_node_once_then_stitch_against_several_counterparts_matches_fused_forward() -> None:
    """Proves the property that makes per-node caching safe.

    A caller like `score_universe.py` encodes one shared node exactly once
    and reuses that state to `stitch` against many different counterparts;
    this must give bit-identical graphs to calling the fused `forward` fresh
    for every pair (which re-encodes the shared endpoint every time). If
    this did not hold, caching `encode_node`'s output across pairs would be
    unsound.
    """
    generator = _tiny_generator()
    x_shared, ground_shared = _tiny_node(seed=11)
    shared_state = generator.encode_node(x_shared, ground_shared)

    for counterpart_seed in (101, 202, 303):
        x_counterpart, ground_counterpart = _tiny_node(seed=counterpart_seed)
        is_self = torch.zeros(1, dtype=torch.bool)

        counterpart_state = generator.encode_node(x_counterpart, ground_counterpart)
        cached_graph = generator.stitch(shared_state, counterpart_state, is_self)
        fused_graph = generator(
            x_shared, x_counterpart, ground_shared, ground_counterpart, is_self=is_self
        )

        assert torch.equal(cached_graph.x, fused_graph.x), counterpart_seed
        assert torch.equal(cached_graph.adj, fused_graph.adj), counterpart_seed
        assert torch.equal(cached_graph.mask, fused_graph.mask), counterpart_seed
        assert torch.equal(cached_graph.aux["plan"], fused_graph.aux["plan"]), counterpart_seed
        assert torch.equal(cached_graph.aux["log_plan"], fused_graph.aux["log_plan"]), (
            counterpart_seed
        )


def test_generator_node_state_survives_index_select_index_copy_and_restack() -> None:
    """`GeneratorNodeState` supports the exact caching pattern callers use.

    Mirrors `E2ENodeState`'s cache lifecycle in `score_universe.py:1859-1882`
    / `train_egostitch.py:2944-2968`: index-select individual node rows out
    of a batch, re-stack a subset of them into a fresh pair-endpoint batch by
    concatenation, and index-copy one cached row over another.
    """
    generator = _tiny_generator()
    x, ground = _tiny_node(seed=21, n_ground=_TINY.n_ground)
    x = x.expand(4, -1).clone()
    ground = ground.expand(4, -1, -1).clone()
    x = x + 0.1 * torch.randn_like(x)  # make the four rows distinguishable
    ground_ids = torch.randint(0, 1000, (4, _TINY.n_ground), dtype=torch.long)

    state = generator.encode_node(x, ground, ground_ids)

    def _row(source: GeneratorNodeState, row: int) -> GeneratorNodeState:
        return GeneratorNodeState(
            slots=SlotSet(*(value[row : row + 1] for value in source.slots)),
            projected_x=source.projected_x[row : row + 1],
            ground_ids=(None if source.ground_ids is None else source.ground_ids[row : row + 1]),
        )

    row0 = _row(state, 0)
    row2 = _row(state, 2)

    # re-stack: rebuild a 2-row batch by concatenation, mirroring `_stack_cached`.
    restacked = GeneratorNodeState(
        slots=SlotSet(*(torch.cat(values) for values in zip(row0.slots, row2.slots, strict=True))),
        projected_x=torch.cat([row0.projected_x, row2.projected_x]),
        ground_ids=(
            None
            if row0.ground_ids is None or row2.ground_ids is None
            else torch.cat([row0.ground_ids, row2.ground_ids])
        ),
    )

    rows = torch.tensor([0, 2])
    expected_slots = SlotSet(*(value.index_select(0, rows) for value in state.slots))
    for field_name, actual, expected in zip(
        SlotSet._fields, restacked.slots, expected_slots, strict=True
    ):
        assert torch.equal(actual, expected), field_name
    assert torch.equal(restacked.projected_x, state.projected_x.index_select(0, rows))
    assert state.ground_ids is not None and restacked.ground_ids is not None
    assert torch.equal(restacked.ground_ids, state.ground_ids.index_select(0, rows))

    # index-copy: overwrite row 0 of the original batch with row 2's state
    # (the `_merge_node_states` pattern), leaving every other row untouched.
    merged_slots = SlotSet(
        *(
            a.index_copy(0, torch.tensor([0]), b)
            for a, b in zip(state.slots, row2.slots, strict=True)
        )
    )
    for field_name, merged_field, row2_field, original_field in zip(
        SlotSet._fields, merged_slots, row2.slots, state.slots, strict=True
    ):
        assert torch.equal(merged_field[0:1], row2_field), field_name
        assert torch.equal(merged_field[1:], original_field[1:]), field_name


def test_perturbation_reaches_build_scaffold_and_none_reproduces_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-`None` perturbation reaches `build_scaffold` and changes the graph.

    `perturbation=None` (the default) must reproduce the unperturbed
    baseline exactly -- the same value already proven bit-exact against the
    live `EgoStitchModel` path in
    `test_matches_live_e2e_build_pair_context`.
    """
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.tensor([False, False, False, False])
    state_a = generator.encode_node(x_a, ground_a)
    state_b = generator.encode_node(x_b, ground_b)

    import src.model.egostitch.generator.egostitch as generator_module

    captured_kwargs: dict[str, object] = {}
    original_build_scaffold = generator_module.build_scaffold

    def capturing_build_scaffold(*args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return original_build_scaffold(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(generator_module, "build_scaffold", capturing_build_scaffold)

    baseline = generator.stitch(state_a, state_b, is_self)
    assert captured_kwargs.get("perturbation") is None

    pairs = [(f"u{i}", f"v{i}") for i in range(4)]
    perturbation = make_scaffold_input_perturbation("shuffle_within_pair_v3", pairs)
    perturbed = generator.stitch(state_a, state_b, is_self, perturbation=perturbation)

    assert captured_kwargs.get("perturbation") is perturbation
    assert not torch.equal(perturbed.x, baseline.x)
    assert not torch.equal(perturbed.adj, baseline.adj)

    reproduced = generator.stitch(state_a, state_b, is_self, perturbation=None)
    assert torch.equal(reproduced.x, baseline.x)
    assert torch.equal(reproduced.adj, baseline.adj)


# --------------------------------------------------------------------------- auxiliary_losses


def test_auxiliary_losses_returns_every_generator_owned_component() -> None:
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.zeros(4, dtype=torch.bool)
    graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)
    batch = {**_node_stream_batch(), **_edge_align_batch()}

    losses = generator.auxiliary_losses(graph, batch)

    expected_keys = {
        "feat",
        "exist",
        "mult",
        "slotadj",
        "gate",
        "ptr",
        "div",
        "deg",
        "real_egostat",
        "real_gin",
        "ssl_noise",
        "ssl_pool",
        "align",
    }
    assert set(losses) == expected_keys
    for name, value in losses.items():
        assert value.dim() == 0, name
        assert bool(torch.isfinite(value)), name

    total = torch.stack(list(losses.values())).sum()
    total.backward()  # type: ignore[no-untyped-call]
    first_linear = generator.stage1.tokenize.encoder[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.weight.grad is not None
    assert bool(torch.isfinite(first_linear.weight.grad).all())


def test_auxiliary_losses_align_matches_a_direct_computation() -> None:
    """Cross-checks `L_align`'s wiring against the same primitives called directly."""
    generator = _tiny_generator()
    x_a, x_b, ground_a, ground_b = _tiny_endpoints()
    is_self = torch.zeros(4, dtype=torch.bool)
    graph = generator(x_a, x_b, ground_a, ground_b, is_self=is_self)
    batch = {**_node_stream_batch(), **_edge_align_batch()}

    losses = generator.auxiliary_losses(graph, batch)

    slots_a = _slotset_from_aux(graph.aux, "slots_a")
    slots_b = _slotset_from_aux(graph.aux, "slots_b")
    target_proj_a = generator.stage1.project_features(batch["target_features_a"]).detach()
    target_proj_b = generator.stage1.project_features(batch["target_features_b"]).detach()
    assignment_a = match_slots(
        slots_a,
        target_proj=target_proj_a,
        target_mult=batch["target_mult_a"],
        target_adj=batch["target_adj_a"],
        target_mask=batch["target_mask_a"],
    )
    assignment_b = match_slots(
        slots_b,
        target_proj=target_proj_b,
        target_mult=batch["target_mult_b"],
        target_adj=batch["target_adj_b"],
        target_mask=batch["target_mask_b"],
    )
    teacher = alignment_teacher_cells(
        assignment_a,
        assignment_b,
        target_ids_a=batch["target_node_index_a"],
        target_ids_b=batch["target_node_index_b"],
        slots=_TINY.slots,
    )
    expected_align = alignment_loss(
        graph.aux["log_plan"],
        teacher,
        positive_real_mask=batch["edge_mask"] * (batch["label"] == 1).float(),
        world_size=1,
    )
    torch.testing.assert_close(losses["align"], expected_align)
