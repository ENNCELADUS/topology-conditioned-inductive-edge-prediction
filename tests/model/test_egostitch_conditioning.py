"""Tests for e2e head-null branch masks (design rev 3 §3.5/§4)."""

import pytest
import torch
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_CONTENT_HEAD,
    NULL_NONE,
    NULL_TOPO_HEAD,
    GatedCrossAttention,
    HeadNullMasks,
    masks_for_null,
    sample_branch_masks,
)


def test_sample_branch_masks_shapes_and_dtype() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(64, 0.15, 0.15, generator=gen, device=torch.device("cpu"))
    assert isinstance(masks, HeadNullMasks)
    assert masks.topo.shape == (64,) and masks.topo.dtype == torch.bool
    assert masks.cont.shape == (64,) and masks.cont.dtype == torch.bool


def test_sample_branch_masks_p_zero_all_active() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(32, 0.0, 0.0, generator=gen, device=torch.device("cpu"))
    assert bool(masks.topo.all()) and bool(masks.cont.all())


def test_sample_branch_masks_deterministic_given_seed() -> None:
    a = sample_branch_masks(
        128, 0.5, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    b = sample_branch_masks(
        128, 0.5, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    assert torch.equal(a.topo, b.topo) and torch.equal(a.cont, b.cont)


@pytest.mark.parametrize(
    ("null", "topo_on", "cont_on"),
    [
        (NULL_NONE, True, True),
        (NULL_ALL_HEAD, False, False),
        (NULL_TOPO_HEAD, False, True),
        (NULL_CONTENT_HEAD, True, False),
    ],
)
def test_masks_for_null(null: str, topo_on: bool, cont_on: bool) -> None:
    masks = masks_for_null(null, 4, torch.device("cpu"))
    assert bool(masks.topo.all()) is topo_on and bool((~masks.topo).all()) is not topo_on
    assert bool(masks.cont.all()) is cont_on and bool((~masks.cont).all()) is not cont_on


def test_masks_for_null_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown head-null"):
        masks_for_null("nope", 4, torch.device("cpu"))


def test_gated_xattn_identity_at_init() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    cls = torch.randn(3, 1, 32)
    tokens = torch.randn(3, 7, 32)
    active = torch.ones(3, dtype=torch.bool)
    out = m(cls, tokens, None, active)
    assert torch.equal(out, cls)  # gate zero-init => exact identity


def test_gated_xattn_per_sample_mask_equals_bypass() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    with torch.no_grad():
        m.gate.fill_(0.7)  # open the gate so the pathway is live
    m.eval()
    cls = torch.randn(5, 1, 32)
    tokens = torch.randn(5, 9, 32)
    active = torch.tensor([True, False, True, False, True])
    out = m(cls, tokens, None, active)
    # masked samples: exact identity (== hard bypass)
    assert torch.equal(out[~active], cls[~active])
    # active samples: actually conditioned
    assert not torch.allclose(out[active], cls[active])


def test_gated_xattn_respects_token_mask() -> None:
    torch.manual_seed(0)
    m = GatedCrossAttention(d_model=32, n_heads=4, dropout=0.0)
    with torch.no_grad():
        m.gate.fill_(0.7)
    m.eval()
    cls = torch.randn(2, 1, 32)
    tokens = torch.randn(2, 6, 32)
    active = torch.ones(2, dtype=torch.bool)
    mask_all = torch.ones(2, 6, dtype=torch.bool)
    out_full = m(cls, tokens, mask_all, active)
    # zero out the padded tail AND mask it: masked tokens must not matter
    tokens2 = tokens.clone()
    tokens2[:, 3:, :] = 999.0
    mask_head = mask_all.clone()
    mask_head[:, 3:] = False
    out_head_a = m(cls, tokens, mask_head, active)
    out_head_b = m(cls, tokens2, mask_head, active)
    assert torch.allclose(out_head_a, out_head_b, atol=1e-6)
    assert not torch.allclose(out_full, out_head_a)
