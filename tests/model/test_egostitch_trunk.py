"""Tests: conditioned trunk equals audited PairCrossAttention under bypass."""

from typing import Any

import pytest
import torch
from src.model.B0 import PairCrossAttention
from src.model.egostitch.conditioning import GatedCrossAttention
from src.model.egostitch.trunk import ConditionedPairCrossAttention

_KW: dict[str, Any] = {
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.0,
    "pair_readout_mode": "pair_context_gated",
    "mixing_mode": "bidirectional_cross",
}


def _inputs(b: int = 3, la: int = 5, lb: int = 7) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(b, la, 32, generator=g),
        torch.randn(b, lb, 32, generator=g),
        torch.full((b,), la, dtype=torch.long),
        torch.full((b,), lb, dtype=torch.long),
    )


def test_bypass_matches_parent_exactly() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW)
    parent = PairCrossAttention(**_KW)
    # load the shared submodule weights parent<-cond (cond has extra sublayers)
    parent.load_state_dict({k: v for k, v in cond.state_dict().items() if k in parent.state_dict()})
    cond.eval(), parent.eval()
    h_a, h_b, la, lb = _inputs()
    out_cond = cond(h_a, h_b, la, lb)  # no tokens => hard bypass
    out_parent = parent(h_a, h_b, la, lb)
    assert torch.equal(out_cond, out_parent)


def test_zero_gate_is_identity_even_when_active() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=2, xattn_heads=4, **_KW).eval()
    h_a, h_b, la, lb = _inputs()
    topo = torch.randn(3, 10, 32)
    active = torch.ones(3, dtype=torch.bool)
    out_off = cond(h_a, h_b, la, lb)
    out_on = cond(h_a, h_b, la, lb, topo_tokens=topo, topo_active=active)
    assert torch.equal(out_on, out_off)  # gates zero-init


def test_open_gate_conditions_output() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW).eval()
    xattn = cond.topo_xattn[0]
    assert isinstance(xattn, GatedCrossAttention)
    with torch.no_grad():
        xattn.gate.fill_(0.5)
    h_a, h_b, la, lb = _inputs()
    topo = torch.randn(3, 10, 32)
    active = torch.ones(3, dtype=torch.bool)
    out_off = cond(h_a, h_b, la, lb)
    out_on = cond(h_a, h_b, la, lb, topo_tokens=topo, topo_active=active)
    assert not torch.allclose(out_on, out_off)


def test_conditioned_pair_readout_is_fp32_under_bf16_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW).eval()
    h_a, h_b, la, lb = _inputs()
    observed: list[tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]] = []
    original_forward = cond.pair_context_readout.forward

    def capture_readout(
        readout_a: torch.Tensor,
        readout_b: torch.Tensor,
        cls_vec: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        output = original_forward(readout_a, readout_b, cls_vec, mask_a, mask_b)
        observed.append((readout_a.dtype, readout_b.dtype, cls_vec.dtype, output.dtype))
        return output

    monkeypatch.setattr(cond.pair_context_readout, "forward", capture_readout)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = cond(h_a, h_b, la, lb)

    assert observed == [(torch.float32, torch.float32, torch.float32, torch.float32)]
    assert output.dtype == torch.float32


def test_malformed_inputs_raise_valueerror_like_parent() -> None:
    torch.manual_seed(1)
    cond = ConditionedPairCrossAttention(n_inj=1, xattn_heads=4, **_KW).eval()
    h_a, h_b, la, lb = _inputs()

    with pytest.raises(ValueError):
        cond(h_a[:, 0, :], h_b, la, lb)  # h_a is 2-D, not (batch, seq_len, d_model)

    with pytest.raises(ValueError):
        cond(h_a, h_b[:-1], la, lb[:-1])  # mismatched batch dimension
