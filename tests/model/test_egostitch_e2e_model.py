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
from src.model.egostitch.e2e_model import E2EPairContext, EgoStitchE2E
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.model import NodeEncoding
from src.model.egostitch.scaffold import counterpart_membership, grounded_identity_match


def _tiny_model_and_batch(
    *, p_topo: float = 0.15, p_cont: float = 0.15
) -> tuple[EgoStitchE2E, dict[str, torch.Tensor]]:
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
        p_topo=p_topo,
        p_cont=p_cont,
    )
    model = EgoStitchE2E(cfg).eval()
    b, t, d_in = 4, 6, model.input_dim
    n_ground = 5
    batch = {
        "emb_a": torch.randn(b, t, d_in),
        "emb_b": torch.randn(b, t, d_in),
        "len_a": torch.full((b,), t, dtype=torch.long),
        "len_b": torch.full((b,), t, dtype=torch.long),
        "x_a": torch.randn(b, model.node_feature_dim),
        "x_b": torch.randn(b, model.node_feature_dim),
        "ground_a": torch.randn(b, n_ground, model.node_feature_dim),
        "ground_b": torch.randn(b, n_ground, model.node_feature_dim),
        "ground_id_a": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
        "ground_id_b": torch.randint(0, 1000, (b, n_ground), dtype=torch.long),
    }
    return model, batch


def test_pair_symmetry_all_conditions() -> None:
    model, batch = _tiny_model_and_batch()
    batch["ground_id_b"] = batch["ground_id_a"].roll(shifts=1, dims=1)
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

    p0_model, p0_batch = _tiny_model_and_batch(p_topo=0.0, p_cont=0.0)
    p0_batch["ground_id_b"] = p0_batch["ground_id_a"].roll(shifts=1, dims=1)
    p0_topo_xattn = p0_model.trunk.topo_xattn[0]
    p0_cont_xattn = p0_model.trunk.cont_xattn[0]
    assert isinstance(p0_topo_xattn, GatedCrossAttention)
    assert isinstance(p0_cont_xattn, GatedCrossAttention)
    with torch.no_grad():
        p0_topo_xattn.gate.fill_(0.4)
        p0_cont_xattn.gate.fill_(0.4)
    p0_swapped = dict(p0_batch)
    for key_a, key_b in (
        ("emb_a", "emb_b"),
        ("len_a", "len_b"),
        ("x_a", "x_b"),
        ("ground_a", "ground_b"),
        ("ground_id_a", "ground_id_b"),
    ):
        p0_swapped[key_a], p0_swapped[key_b] = p0_batch[key_b], p0_batch[key_a]
    assert torch.allclose(
        p0_model(p0_batch)["logits"], p0_model(p0_swapped)["logits"], atol=1e-5
    ), "asymmetry under p0"


def test_train_mask_equals_eval_bypass() -> None:
    model, batch = _tiny_model_and_batch()
    batch["ground_id_b"] = batch["ground_id_a"].roll(shifts=1, dims=1)
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


def test_missing_grounding_fails_closed() -> None:
    model, batch = _tiny_model_and_batch()
    batch.pop("ground_a")
    with pytest.raises(ValueError, match="requires real grounding"):
        model(batch)


def test_matched_flags_shared_candidate() -> None:
    """Soft identity matching is symmetric and requires shared pool ids."""
    ids_a = torch.tensor([[10, 11], [10, 11], [10, 11], [10, 11]], dtype=torch.long)
    ids_b = torch.tensor([[20, 10], [20, 21], [20, 10], [20, 10]], dtype=torch.long)
    # Single slot per side; the shared id is at different pool positions.
    pointer_a = torch.zeros(4, 1, 2)
    pointer_a[:, :, 0] = 1.0
    pointer_b = torch.zeros(4, 1, 2)
    pointer_b[:, :, 1] = 1.0
    gate_a = torch.tensor([[0.9], [0.9], [0.3], [0.9]])
    gate_b = torch.tensor([[0.9], [0.9], [0.9], [0.3]])

    matched_a, matched_b = grounded_identity_match(
        pointer_a, gate_a, ids_a, pointer_b, gate_b, ids_b
    )
    expected = torch.tensor([[0.81], [0.0], [0.27], [0.27]])
    assert torch.allclose(matched_a, expected)
    assert torch.allclose(matched_b, expected)
    matched_b_swapped, matched_a_swapped = grounded_identity_match(
        pointer_b, gate_b, ids_b, pointer_a, gate_a, ids_a
    )
    assert torch.equal(matched_a, matched_a_swapped)
    assert torch.equal(matched_b, matched_b_swapped)


def test_matched_flags_gradient_reaches_head_pointer() -> None:
    """An overlapping grounding pool carries matched-flag gradient to the pointer head."""
    model, batch = _tiny_model_and_batch()
    slots_a = model.generator.encode_nodes(batch["x_a"][:1], batch["ground_a"][:1]).slots
    slots_b = model.generator.encode_nodes(batch["x_b"][:1], batch["ground_b"][:1]).slots
    ids_a = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    ids_b = torch.tensor([[21, 10, 22, 23, 24]], dtype=torch.long)

    matched_a, _ = grounded_identity_match(
        slots_a.pointer,
        slots_a.gate,
        ids_a,
        slots_b.pointer,
        slots_b.gate,
        ids_b,
    )
    matched_a.sum().backward()

    pointer_parameters = tuple(model.generator.imagine.head_pointer.parameters())
    assert pointer_parameters
    for parameter in pointer_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_matched_flags_maximizes_probability_gate_product() -> None:
    """The counterpart slot maximizes ``M[k,l] * gate_b[l]`` as one product."""
    pointer_a = torch.tensor([[[1.0, 0.0]]])
    pointer_b = torch.tensor([[[0.1, 0.9], [0.4, 0.6]]])
    gate_a = torch.tensor([[0.8]])
    gate_b = torch.tensor([[0.1, 0.9]])
    ids_a = torch.tensor([[10, 11]], dtype=torch.long)
    ids_b = torch.tensor([[11, 10]], dtype=torch.long)

    matched_a, _ = grounded_identity_match(
        pointer_a, gate_a, ids_a, pointer_b, gate_b, ids_b
    )

    assert torch.allclose(matched_a, torch.tensor([[0.432]]))


def test_matched_flags_disjoint_pools_are_exact_zero_with_zero_gradient() -> None:
    """Disjoint grounding pools produce exact zeros, including through backward."""
    pointer_a = torch.softmax(torch.randn(1, 2, 3), dim=-1).requires_grad_()
    pointer_b = torch.softmax(torch.randn(1, 2, 3), dim=-1).requires_grad_()
    gate_a = torch.sigmoid(torch.randn(1, 2)).requires_grad_()
    gate_b = torch.sigmoid(torch.randn(1, 2)).requires_grad_()
    ids_a = torch.tensor([[10, 11, 12]], dtype=torch.long)
    ids_b = torch.tensor([[20, 21, 22]], dtype=torch.long)

    matched_a, matched_b = grounded_identity_match(
        pointer_a, gate_a, ids_a, pointer_b, gate_b, ids_b
    )
    assert torch.equal(matched_a, torch.zeros_like(matched_a))
    assert torch.equal(matched_b, torch.zeros_like(matched_b))

    (matched_a.sum() + matched_b.sum()).backward()
    for tensor in (pointer_a, pointer_b, gate_a, gate_b):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_f_logit_invariant_to_grounding() -> None:
    model, batch = _tiny_model_and_batch()
    dec_a = model.decompose(batch)
    batch2 = dict(batch)
    batch2["ground_a"] = torch.randn_like(batch["ground_a"])
    batch2["ground_b"] = torch.randn_like(batch["ground_b"])
    batch2["ground_id_a"] = torch.randint_like(batch["ground_id_a"], 0, 1000)
    batch2["ground_id_b"] = torch.randint_like(batch["ground_id_b"], 0, 1000)
    dec_b = model.decompose(batch2)
    # f_logit nulls both topo and content pathways, so grounding never reaches the trunk.
    assert torch.allclose(dec_a["f_logit"], dec_b["f_logit"], atol=1e-6)


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
    model, batch = _tiny_model_and_batch()
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
    model, batch = _tiny_model_and_batch()
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
    model, batch = _tiny_model_and_batch()
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


def test_bf16_autocast_returns_registered_fp32_readout_and_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered BF16 path keeps the pair readout and logits in fp32."""
    model, batch = _tiny_model_and_batch()
    head_input_dtypes: list[torch.dtype] = []
    original_forward = model.head.forward

    def capture_head_input(feat: torch.Tensor) -> torch.Tensor:
        head_input_dtypes.append(feat.dtype)
        return original_forward(feat)

    monkeypatch.setattr(model.head, "forward", capture_head_input)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)["logits"]

    assert head_input_dtypes == [torch.float32]
    assert output.dtype == torch.float32


def test_membership_is_content_only_and_content_null_ablates_it() -> None:
    model, batch = _tiny_model_and_batch()
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
