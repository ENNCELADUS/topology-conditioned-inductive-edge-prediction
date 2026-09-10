"""Prefix-tuning module tests: generator shapes, symmetry, and the slot-specific shift."""

from __future__ import annotations

import torch
from src.model.egostitch.classifier.prefix import PrefixConfig, PrefixGenerator


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
