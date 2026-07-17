"""E2E model property tests (design rev 3 §3.4–§3.5 acceptance criteria)."""

import torch
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_CONTENT_HEAD,
    NULL_TOPO_HEAD,
    GatedCrossAttention,
    masks_for_null,
)
from src.model.egostitch.config import E2EConfig
from src.model.egostitch.e2e_model import EgoStitchE2E, grounded_identity_match


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
