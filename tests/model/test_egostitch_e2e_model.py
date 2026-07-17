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
from src.model.egostitch.e2e_model import EgoStitchE2E


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
