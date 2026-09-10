"""Prefix-tuning module tests: generator shapes, symmetry, and the slot-specific shift."""

from __future__ import annotations

import torch
from src.model.egostitch.classifier.layers import CrossAttentionLayer
from src.model.egostitch.classifier.prefix import (
    PrefixConfig,
    PrefixCrossAttentionLayer,
    PrefixGenerator,
    prefix_branch,
)


def test_static_generator_has_no_condition_and_shares_one_prefix() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="static", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=2, n_heads=2, cfg=cfg)
    assert gen.condition(torch.zeros(3, 5, 8), torch.zeros(3, 5, 8), None, None) is None
    prefix = gen.prefix(1, None, batch_size=3)
    assert prefix.shape == (3, 4, 8)
    assert torch.equal(prefix[0], prefix[2])
    assert not any(
        name.startswith("shift") or name.startswith("cond") for name, _ in gen.named_parameters()
    )


def test_pair_generator_is_symmetric_and_slot_specific() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=2, n_heads=2, cfg=cfg)
    a = torch.randn(3, 5, 8)
    b = torch.randn(3, 4, 8)
    mask_a = torch.tensor([[False] * 5, [False] * 4 + [True], [False] * 3 + [True] * 2])
    mask_b = torch.tensor([[False] * 4, [False] * 4, [False] * 3 + [True]])
    z_ab = gen.condition(a, b, mask_a, mask_b)
    z_ba = gen.condition(b, a, mask_b, mask_a)
    assert z_ab is not None and z_ab.shape == (3, 6)
    assert z_ba is not None
    assert torch.allclose(z_ab, z_ba)
    prefix = gen.prefix(0, z_ab, batch_size=3)
    delta = prefix - gen.p0[0].unsqueeze(0)
    # slot-specific: rows of the shift differ within one pair
    assert not torch.allclose(delta[0, 0], delta[0, 1])
    # pair-specific: the shift differs across pairs
    assert not torch.allclose(delta[0], delta[1])


def test_gates_start_at_zero_and_shift_out_is_small_but_nonzero() -> None:
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=3, n_heads=2, cfg=cfg)
    assert gen.gates.shape == (3, 3, 2)
    assert torch.count_nonzero(gen.gates) == 0
    assert all(p.abs().sum() > 0 for p in gen.shift_out)
    assert all(p.abs().max() < 0.2 for p in gen.shift_out)


def test_init_static_from_tokens_copies_rows_and_z_mean_accumulates() -> None:
    cfg = PrefixConfig(tokens=2, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    rows = torch.arange(40, dtype=torch.float32).view(5, 8)
    gen.init_static_from_tokens(rows, seed=3)
    assert all(any(torch.equal(p_row, r) for r in rows) for p_row in gen.p0[0])
    gen.train()
    z = gen.condition(torch.randn(4, 3, 8), torch.randn(4, 3, 8), None, None)
    assert z is not None
    assert float(gen.z_count) == 4.0
    assert torch.allclose(gen.z_mean, z.mean(dim=0))


def test_config_round_trip_and_validation() -> None:
    raw = {
        "tokens": 16,
        "rank": 8,
        "conditioning": "pair",
        "bottleneck": 128,
        "base_checkpoint": "outputs/x/best.pt",
        "base_checkpoint_sha256": None,
    }
    cfg = PrefixConfig.from_mapping(raw)
    assert cfg.to_dict() == raw
    try:
        PrefixConfig.from_mapping({**raw, "conditioning": "graph"})
    except ValueError as err:
        assert "conditioning" in str(err)
    else:
        raise AssertionError("bad conditioning must raise")


def _frozen_layer(seed: int = 0) -> CrossAttentionLayer:
    torch.manual_seed(seed)
    layer = CrossAttentionLayer(d_model=8, n_heads=2, dropout=0.1)
    layer.eval()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def test_prefix_branch_is_exactly_zero_at_zero_gate_and_nonzero_otherwise() -> None:
    layer = _frozen_layer()
    query = torch.randn(3, 5, 8)
    prefix = torch.randn(3, 4, 8)
    zero = prefix_branch(layer.attn, query, prefix, torch.zeros(2), gate_scale=1.0)
    assert torch.count_nonzero(zero) == 0
    scaled_off = prefix_branch(layer.attn, query, prefix, torch.ones(2), gate_scale=0.0)
    assert torch.count_nonzero(scaled_off) == 0
    live = prefix_branch(layer.attn, query, prefix, torch.ones(2), gate_scale=1.0)
    assert live.shape == (3, 5, 8) and torch.count_nonzero(live) > 0


def test_prefix_layer_reproduces_frozen_layer_bitwise_at_zero_gates() -> None:
    layer = _frozen_layer()
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    wrapped = PrefixCrossAttentionLayer(layer, 0, gen)
    h_a, h_b = torch.randn(3, 5, 8), torch.randn(3, 4, 8)
    cls = torch.randn(3, 1, 8)
    mask_a = torch.tensor([[False] * 5, [False] * 4 + [True], [False] * 3 + [True] * 2])
    mask_b = torch.tensor([[False] * 4, [False] * 4, [False] * 3 + [True]])
    z = gen.condition(h_a, h_b, mask_a, mask_b)
    base = layer(h_a, h_b, cls, mask_a, mask_b)
    ours = wrapped(h_a, h_b, cls, mask_a, mask_b, z, gate_scale=1.0)
    for got, want in zip(ours, base, strict=True):
        assert torch.equal(got, want)


def test_prefix_layer_changes_output_once_a_gate_opens_and_grads_stay_on_prefix() -> None:
    layer = _frozen_layer()
    cfg = PrefixConfig(tokens=4, rank=2, conditioning="pair", bottleneck=6)
    gen = PrefixGenerator(d_model=8, n_layers=1, n_heads=2, cfg=cfg)
    with torch.no_grad():
        gen.gates[0, 0, 0] = 0.5
    wrapped = PrefixCrossAttentionLayer(layer, 0, gen)
    h_a, h_b, cls = torch.randn(3, 5, 8), torch.randn(3, 4, 8), torch.randn(3, 1, 8)
    z = gen.condition(h_a, h_b, None, None)
    out_a, _, _ = wrapped(h_a, h_b, cls, None, None, z, gate_scale=1.0)
    base_a, _, _ = layer(h_a, h_b, cls, None, None)
    assert not torch.equal(out_a, base_a)
    out_a.sum().backward()
    assert all(p.grad is None for p in layer.parameters())
    assert gen.p0[0].grad is not None and gen.gates.grad is not None
