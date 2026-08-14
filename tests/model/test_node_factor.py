"""Tests for `NodeFactorBottleneck` (B1 training-time topology-distillation plan).

Pins the properties the plan's design rationale depends on: the residual is
exactly zero at init (bit-identical to the `node_factor_dim=0` baseline,
proven again at the model level in `tests/model/test_v31_node_factor.py`), `r`
is symmetric in the two endpoints by construction, and the zero-init `diag_w`
gives every coordinate a direct, nonzero step-0 gradient while `w_z` gets
none through that same path -- the "no scalar-gate saddle" claim
(`node_factor.py` module docstring).
"""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.classifier.node_factor import NodeFactorBottleneck, NodeFactorOutputs

_D_MODEL = 8
_DIM = 6


def _tokens(seed: int, batch: int = 4, length: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, length, _D_MODEL, generator=gen)
    lengths = torch.full((batch,), length, dtype=torch.long)
    return tokens, lengths


def test_forward_returns_correctly_shaped_outputs() -> None:
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1)

    out = model(tokens_a, tokens_b, len_a, len_b)

    assert isinstance(out, NodeFactorOutputs)
    batch = tokens_a.size(0)
    assert out.z_a.shape == (batch, _DIM)
    assert out.z_b.shape == (batch, _DIM)
    assert out.r.shape == (batch,)
    assert out.b_a.shape == (batch,)
    assert out.b_b.shape == (batch,)
    assert out.residual.shape == (batch,)


def test_z_is_unit_norm() -> None:
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1)

    out = model(tokens_a, tokens_b, len_a, len_b)

    torch.testing.assert_close(out.z_a.norm(dim=-1), torch.ones(tokens_a.size(0)))
    torch.testing.assert_close(out.z_b.norm(dim=-1), torch.ones(tokens_b.size(0)))


def test_r_is_symmetric_in_the_two_endpoints() -> None:
    """`r = (z_a * diag_w * z_b).sum() / sqrt(dim)` is symmetric by construction.

    Away from init (`diag_w != 0`), so the property being pinned is the
    formula's symmetry, not the trivial init-time zero.
    """
    torch.manual_seed(0)
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    with torch.no_grad():
        model.diag_w.normal_()
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1, length=7)

    forward = model(tokens_a, tokens_b, len_a, len_b)
    swapped = model(tokens_b, tokens_a, len_b, len_a)

    torch.testing.assert_close(forward.r, swapped.r)


def test_residual_is_exactly_zero_at_init() -> None:
    """`diag_w` and `bias_head` zero-init: `r`, `b_a`, `b_b`, and the residual are all exactly 0."""
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1)

    out = model(tokens_a, tokens_b, len_a, len_b)

    assert torch.equal(out.r, torch.zeros_like(out.r))
    assert torch.equal(out.b_a, torch.zeros_like(out.b_a))
    assert torch.equal(out.b_b, torch.zeros_like(out.b_b))
    assert torch.equal(out.residual, torch.zeros_like(out.residual))


def test_diag_w_gradient_nonzero_and_w_z_gradient_zero_at_init() -> None:
    """No scalar-gate saddle: `diag_w` gets a direct step-0 gradient, `w_z` does not (yet).

    `dr/dz_k = diag_w_k * z_other_k`, so with `diag_w == 0` at init the
    gradient reaching `w_z` through `r` is exactly zero -- `w_z` has no other
    path to the loss (`b_a`/`b_b` never read `z_a`/`z_b`). `dr/dw_k =
    z_ak * z_bk / sqrt(dim)` has no such dependency on `diag_w` and is
    generically nonzero at a random init.
    """
    torch.manual_seed(0)
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1)
    labels = torch.randint(0, 2, (tokens_a.size(0),)).float()

    out = model(tokens_a, tokens_b, len_a, len_b)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.residual, labels)
    loss.backward()  # type: ignore[no-untyped-call]

    assert model.diag_w.grad is not None
    assert torch.count_nonzero(model.diag_w.grad) > 0
    assert model.w_z.weight.grad is not None
    assert torch.equal(model.w_z.weight.grad, torch.zeros_like(model.w_z.weight.grad))
    assert model.w_z.bias.grad is not None
    assert torch.equal(model.w_z.bias.grad, torch.zeros_like(model.w_z.bias.grad))


def test_output_stays_fp32_under_bf16_autocast() -> None:
    model = NodeFactorBottleneck(_D_MODEL, _DIM)
    tokens_a, len_a = _tokens(0)
    tokens_b, len_b = _tokens(1)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(tokens_a, tokens_b, len_a, len_b)

    for tensor in (out.z_a, out.z_b, out.r, out.b_a, out.b_b, out.residual):
        assert tensor.dtype == torch.float32


def test_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        NodeFactorBottleneck(_D_MODEL, 0)
