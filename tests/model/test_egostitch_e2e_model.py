"""E2E model property tests (design rev 3 §3.4–§3.5 acceptance criteria)."""

from collections.abc import Callable
from typing import cast

import pytest
import torch
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_CONTENT_HEAD,
    NULL_TOPO_HEAD,
    GatedCrossAttention,
    masks_for_null,
)
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import (
    E2EPairContext,
    EgoStitchE2E,
    counterpart_membership,
    grounded_identity_match,
)
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.model import NodeEncoding


def _tiny_model_and_batch() -> tuple[EgoStitchE2E, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    cfg = E2EConfig(
        d_model=32,
        encoder_layers=1,
        cross_attn_layers=2,
        n_heads=4,
        n_inj=1,
        ste_dim=16,
        ste_layers=2,
        xattn_heads=4,
    )
    model = EgoStitchE2E(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    batch = {
        "emb_a": torch.randn(b, t, d_in),
        "emb_b": torch.randn(b, t, d_in),
        "len_a": torch.full((b,), t, dtype=torch.long),
        "len_b": torch.full((b,), t, dtype=torch.long),
        "x_a": torch.randn(b, model.node_feature_dim),
        "x_b": torch.randn(b, model.node_feature_dim),
    }
    return model, batch


def test_pair_symmetry_all_conditions() -> None:
    model, batch = _tiny_model_and_batch()
    topo_xattn, cont_xattn = model.trunk.topo_xattn[0], model.trunk.cont_xattn[0]
    assert isinstance(topo_xattn, GatedCrossAttention)
    assert isinstance(cont_xattn, GatedCrossAttention)
    with torch.no_grad():
        for p in (topo_xattn.gate, cont_xattn.gate):
            p.fill_(0.4)  # open gates: symmetry must hold with live conditioning
    swapped = dict(batch)
    swapped["emb_a"], swapped["emb_b"] = batch["emb_b"], batch["emb_a"]
    swapped["len_a"], swapped["len_b"] = batch["len_b"], batch["len_a"]
    swapped["x_a"], swapped["x_b"] = batch["x_b"], batch["x_a"]
    for null in (None, NULL_ALL_HEAD, NULL_TOPO_HEAD, NULL_CONTENT_HEAD):
        masks = None if null is None else masks_for_null(null, 4, torch.device("cpu"))
        out_ij = model(batch, masks=masks)["logits"]
        out_ji = model(swapped, masks=masks)["logits"]
        assert torch.allclose(out_ij, out_ji, atol=1e-5), f"asymmetry under {null}"


def test_train_mask_equals_eval_bypass() -> None:
    model, batch = _tiny_model_and_batch()
    topo_xattn, cont_xattn = model.trunk.topo_xattn[0], model.trunk.cont_xattn[0]
    assert isinstance(topo_xattn, GatedCrossAttention)
    assert isinstance(cont_xattn, GatedCrossAttention)
    with torch.no_grad():
        topo_xattn.gate.fill_(0.4)
        cont_xattn.gate.fill_(0.4)
    dec = model.decompose(batch)  # eval-time hard bypasses
    for null, key in (
        (NULL_ALL_HEAD, "f_logit"),
        (NULL_TOPO_HEAD, "pair_content"),
        (NULL_CONTENT_HEAD, "pair_topology"),
    ):
        masked = model(batch, masks=masks_for_null(null, 4, torch.device("cpu")))["logits"]
        assert torch.allclose(masked, dec[key], atol=1e-6), f"mask!=bypass for {null}"


def test_f_logit_invariant_to_scaffold() -> None:
    model, batch = _tiny_model_and_batch()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["x_a"] = torch.randn_like(batch["x_a"])  # changes slots => scaffold
    dec_b = model.decompose(batch2)
    # f_logit depends on token streams only — x_a feeds imagination, not trunk
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)


# --------------------------------------------------------------------------- Task 13b:
# grounded-identity-match semantics + real grounding wiring (spec Sec 13.18 pinned)


def test_matched_flags_shared_candidate() -> None:
    """Direct unit test of the pure `grounded_identity_match` helper.

    Four rows, one slot per side, two grounding candidates: (0) same argmax
    id + both gates open => matched on both sides; (1) disjoint argmax ids =>
    unmatched on both sides even though both gates are open; (2) own gate
    <= 0.5 on side a => side a unmatched (and side b unmatched too, since the
    only a-side slot that could support it isn't gated); (3) ids match and
    side a's own gate is open, but side b's gate is <= 0.5 => side a
    unmatched (the OTHER endpoint's gate clause fails) and side b unmatched
    (its own gate clause fails).
    """
    ids_a = torch.tensor([[10, 11], [10, 11], [10, 11], [10, 11]], dtype=torch.long)
    ids_b = torch.tensor([[10, 11], [20, 21], [10, 11], [10, 11]], dtype=torch.long)
    # Single slot per side; pointer argmax always lands on grounding-pool index 0.
    pointer_a = torch.zeros(4, 1, 2)
    pointer_a[:, :, 0] = 1.0
    pointer_b = torch.zeros(4, 1, 2)
    pointer_b[:, :, 0] = 1.0
    gate_a = torch.tensor([[0.9], [0.9], [0.3], [0.9]])
    gate_b = torch.tensor([[0.9], [0.9], [0.9], [0.3]])

    matched_a, matched_b = grounded_identity_match(
        pointer_a, gate_a, ids_a, pointer_b, gate_b, ids_b
    )
    assert torch.equal(matched_a, torch.tensor([[1.0], [0.0], [0.0], [0.0]]))
    assert torch.equal(matched_b, torch.tensor([[1.0], [0.0], [0.0], [0.0]]))


def _tiny_model_and_batch_with_grounding() -> tuple[EgoStitchE2E, dict[str, torch.Tensor]]:
    model, batch = _tiny_model_and_batch()
    b = batch["emb_a"].size(0)
    n_ground = 5
    torch.manual_seed(1)
    batch["ground_a"] = torch.randn(b, n_ground, model.node_feature_dim)
    batch["ground_b"] = torch.randn(b, n_ground, model.node_feature_dim)
    batch["ground_id_a"] = torch.randint(0, 1000, (b, n_ground), dtype=torch.long)
    batch["ground_id_b"] = torch.randint(0, 1000, (b, n_ground), dtype=torch.long)
    return model, batch


def test_pair_symmetry_with_real_grounding() -> None:
    """Task-11's symmetry test, extended with real grounding-pool batch keys."""
    model, batch = _tiny_model_and_batch_with_grounding()
    topo_xattn, cont_xattn = model.trunk.topo_xattn[0], model.trunk.cont_xattn[0]
    assert isinstance(topo_xattn, GatedCrossAttention)
    assert isinstance(cont_xattn, GatedCrossAttention)
    with torch.no_grad():
        for p in (topo_xattn.gate, cont_xattn.gate):
            p.fill_(0.4)  # open gates: symmetry must hold with live conditioning
    swapped = dict(batch)
    swapped["emb_a"], swapped["emb_b"] = batch["emb_b"], batch["emb_a"]
    swapped["len_a"], swapped["len_b"] = batch["len_b"], batch["len_a"]
    swapped["x_a"], swapped["x_b"] = batch["x_b"], batch["x_a"]
    swapped["ground_a"], swapped["ground_b"] = batch["ground_b"], batch["ground_a"]
    swapped["ground_id_a"], swapped["ground_id_b"] = (
        batch["ground_id_b"],
        batch["ground_id_a"],
    )
    for null in (None, NULL_ALL_HEAD, NULL_TOPO_HEAD, NULL_CONTENT_HEAD):
        masks = None if null is None else masks_for_null(null, 4, torch.device("cpu"))
        out_ij = model(batch, masks=masks)["logits"]
        out_ji = model(swapped, masks=masks)["logits"]
        assert torch.allclose(out_ij, out_ji, atol=1e-5), f"asymmetry under {null}"


def test_f_logit_invariant_to_grounding() -> None:
    model, batch = _tiny_model_and_batch_with_grounding()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["ground_a"] = torch.randn_like(batch["ground_a"])
    batch2["ground_b"] = torch.randn_like(batch["ground_b"])
    batch2["ground_id_a"] = torch.randint_like(batch["ground_id_a"], 0, 1000)
    batch2["ground_id_b"] = torch.randint_like(batch["ground_id_b"], 0, 1000)
    dec_b = model.decompose(batch2)
    # f_logit nulls both topo and content pathways, so grounding never reaches the trunk.
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)


# --------------------------------------------------------------------------- Task 13b review
# follow-up: direct AB/BA-swap unit assertion on the pure helper


def test_matched_flags_swap_sides_swaps_outputs() -> None:
    """Swapping which endpoint is passed as `a`/`b` swaps the matched outputs.

    Direct check of spec Sec 13.18's "symmetric across AB/BA by construction"
    claim on the pure `grounded_identity_match` helper itself: calling it with
    the two endpoints' (pointer, gate, ids) arguments swapped must return the
    same pair of per-slot flags with `matched_a`/`matched_b` themselves
    swapped. Reuses the fixture from `test_matched_flags_shared_candidate`.
    """
    ids_a = torch.tensor([[10, 11], [10, 11], [10, 11], [10, 11]], dtype=torch.long)
    ids_b = torch.tensor([[10, 11], [20, 21], [10, 11], [10, 11]], dtype=torch.long)
    pointer_a = torch.zeros(4, 1, 2)
    pointer_a[:, :, 0] = 1.0
    pointer_b = torch.zeros(4, 1, 2)
    pointer_b[:, :, 0] = 1.0
    gate_a = torch.tensor([[0.9], [0.9], [0.3], [0.9]])
    gate_b = torch.tensor([[0.9], [0.9], [0.9], [0.3]])

    matched_a, matched_b = grounded_identity_match(
        pointer_a, gate_a, ids_a, pointer_b, gate_b, ids_b
    )
    matched_b_swapped, matched_a_swapped = grounded_identity_match(
        pointer_b, gate_b, ids_b, pointer_a, gate_a, ids_a
    )
    assert torch.equal(matched_a, matched_a_swapped)
    assert torch.equal(matched_b, matched_b_swapped)


# --------------------------------------------------------------------------- Task 15:
# `probe_states` — read-only STE token-state export for representation probes


def test_probe_states_returns_ste_token_states() -> None:
    """`probe_states` exports the STE token states that condition the trunk."""
    model, batch = _tiny_model_and_batch()
    states = model.probe_states(batch)
    b = batch["x_a"].size(0)
    assert states.shape[0] == b
    assert states.shape[-1] == model.cfg.d_model
    assert states.requires_grad is False
    again = model.probe_states(batch)
    assert torch.equal(states, again)  # deterministic in eval mode (no dropout)


def test_probe_states_reflects_probe_node_features() -> None:
    """Changing the probe node's own features changes its exported token states."""
    model, batch = _tiny_model_and_batch()
    baseline = model.probe_states(batch)
    perturbed = dict(batch)
    perturbed["x_a"] = torch.randn_like(batch["x_a"])
    changed = model.probe_states(perturbed)
    assert not torch.allclose(baseline, changed)


def test_probe_states_accepts_self_pair_batch() -> None:
    """`probe_states` runs on a self-pair batch (spec Sec 13.9 single-ego path)."""
    model, batch = _tiny_model_and_batch()
    batch["emb_b"] = batch["emb_a"]
    batch["len_b"] = batch["len_a"]
    batch["x_b"] = batch["x_a"]
    batch["is_self"] = torch.ones(batch["x_a"].size(0), dtype=torch.bool)
    states = model.probe_states(batch)
    assert states.shape[0] == batch["x_a"].size(0)


def test_counterpart_membership_matches_pinned_formula_and_is_scale_safe() -> None:
    slots = model_slots = (
        _tiny_model_and_batch()[0]
        .generator.encode_nodes(torch.randn(2, 1536), torch.randn(2, 20, 1536))
        .slots
    )
    other = torch.randn(2, 256)
    tau = torch.tensor(1.7)
    actual = counterpart_membership(slots, other, tau)
    expected = -(
        torch.nn.functional.normalize(slots.h, dim=-1)
        - torch.nn.functional.normalize(other, dim=-1)[:, None]
    ).square().sum(dim=-1) / tau + torch.log((slots.pi * slots.mult).clamp_min(1e-8))
    assert torch.allclose(actual, expected)
    scaled = counterpart_membership(model_slots._replace(h=model_slots.h * 11.0), other * 7.0, tau)
    assert torch.allclose(actual, scaled)


def test_decompose_builds_pair_context_once_and_matches_explicit_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _tiny_model_and_batch_with_grounding()
    with torch.no_grad():
        cast(GatedCrossAttention, model.trunk.topo_xattn[0]).gate.data.fill_(0.4)
        cast(GatedCrossAttention, model.trunk.cont_xattn[0]).gate.data.fill_(0.4)
    calls = 0
    original = model.build_pair_context

    def counted(batch_arg: dict[str, torch.Tensor]) -> E2EPairContext:
        nonlocal calls
        calls += 1
        return original(batch_arg)

    monkeypatch.setattr(model, "build_pair_context", counted)
    decomposed = model.decompose(batch)
    assert calls == 1
    context = original(batch)
    expected = {
        "full": model.score_pair_context(context),
        "f_logit": model.score_pair_context(
            context, masks=masks_for_null(NULL_ALL_HEAD, 4, torch.device("cpu"))
        ),
        "pair_content": model.score_pair_context(
            context, masks=masks_for_null(NULL_TOPO_HEAD, 4, torch.device("cpu"))
        ),
        "pair_topology": model.score_pair_context(
            context, masks=masks_for_null(NULL_CONTENT_HEAD, 4, torch.device("cpu"))
        ),
    }
    for key, value in expected.items():
        assert torch.allclose(decomposed[key], value, atol=1e-6)


def test_self_pairs_encode_one_ego_and_use_exact_identity_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _tiny_model_and_batch_with_grounding()
    batch["emb_b"] = batch["emb_a"]
    batch["len_b"] = batch["len_a"]
    batch["x_b"] = batch["x_a"]
    batch["ground_b"] = batch["ground_a"]
    batch["ground_id_b"] = batch["ground_id_a"]
    batch["is_self"] = torch.ones(4, dtype=torch.bool)

    encode_calls = 0
    original_encode = model.generator.encode_nodes

    def counted_encode(*args: torch.Tensor, **kwargs: torch.Tensor) -> NodeEncoding:
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(model.generator, "encode_nodes", counted_encode)

    import src.model.egostitch.e2e_model as e2e_module

    sinkhorn_calls = 0
    original_sinkhorn = cast(
        Callable[..., torch.Tensor],
        e2e_module.sinkhorn_plan,  # type: ignore[attr-defined]
    )

    def counted_sinkhorn(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        nonlocal sinkhorn_calls
        sinkhorn_calls += 1
        return original_sinkhorn(*args, **kwargs)

    monkeypatch.setattr(e2e_module, "sinkhorn_plan", counted_sinkhorn)
    context = model.build_pair_context(batch)
    assert encode_calls == 1
    assert sinkhorn_calls == 0
    expected = torch.eye(model.generator_cfg.slots).expand(4, -1, -1)
    assert context.plan is not None
    assert torch.equal(context.plan, expected)


def test_bf16_autocast_combines_self_and_sinkhorn_plans_in_fp32() -> None:
    """The fp32 Sinkhorn island owns the assembled plan dtype under autocast."""
    model, batch = _tiny_model_and_batch_with_grounding()
    is_self = torch.tensor([True, False, True, False])
    state_a = model.encode_node_state(
        batch["emb_a"],
        batch["len_a"],
        batch["x_a"],
        batch["ground_a"],
        batch["ground_id_a"],
    )
    state_b = model.encode_node_state(
        batch["emb_b"],
        batch["len_b"],
        batch["x_b"],
        batch["ground_b"],
        batch["ground_id_b"],
    )
    state_a = state_a._replace(
        slots=SlotSet(*(value.to(torch.bfloat16) for value in state_a.slots))
    )
    state_b = state_b._replace(
        slots=SlotSet(*(value.to(torch.bfloat16) for value in state_b.slots))
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = model.build_pair_context_from_states(state_a, state_b, is_self, need_cont=False)

    assert context.plan is not None
    assert context.plan.dtype == torch.float32
    expected_identity = torch.eye(model.generator_cfg.slots)
    assert torch.equal(context.plan[is_self], expected_identity.expand(2, -1, -1))


def test_membership_is_content_only_and_content_null_ablates_it() -> None:
    model, batch = _tiny_model_and_batch_with_grounding()
    with torch.no_grad():
        cast(GatedCrossAttention, model.trunk.cont_xattn[0]).gate.data.fill_(0.7)
    context = model.build_pair_context(batch)
    assert context.cont is not None
    changed = context._replace(cont=context.cont + torch.randn_like(context.cont))
    full_a = model.score_pair_context(context)
    full_b = model.score_pair_context(changed)
    assert not torch.allclose(full_a, full_b)
    mask = masks_for_null(NULL_CONTENT_HEAD, 4, torch.device("cpu"))
    null_a = model.score_pair_context(context, masks=mask)
    null_b = model.score_pair_context(changed, masks=mask)
    assert torch.equal(null_a, null_b)
    assert context.topo_ab is not None and changed.topo_ab is not None
    assert torch.equal(context.topo_ab, changed.topo_ab)
