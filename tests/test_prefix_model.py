"""Prefix-tuning module tests: generator shapes, symmetry, and the slot-specific shift."""

from __future__ import annotations

import torch
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.layers import CrossAttentionLayer
from src.model.egostitch.classifier.prefix import (
    PrefixConfig,
    PrefixCrossAttentionLayer,
    PrefixGenerator,
    V3_1Prefix,
    prefix_branch,
)


def _tiny_base_config(mixing: str = "bidirectional_cross") -> dict[str, object]:
    """A 4-dim, 8-wide V3_1 with two cross-attention layers and every dropout on."""
    return {
        "input_dim": 4,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 2,
        "mlp_head": {
            "hidden_dims": [8],
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "spectral_norm": False,
        },
        "regularization": {
            "dropout": 0.1,
            "token_dropout": 0.1,
            "cross_attention_dropout": 0.1,
            "stochastic_depth": 0.1,
        },
        "rich_pooling": {"components": ["mean", "attn", "max", "gated"]},
        "pair_readout": {
            "mode": "pair_context_gated",
            "order_aggregation": "abba_max",
            "spectral_norm": False,
        },
        "mixing": {"mode": mixing},
        "label_smoothing": 0.0,
        "positive_weight": 5.0,
    }


def _pair_batch(n: int = 6, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random (B, L, 4) token pairs with ragged lengths and binary labels."""
    gen = torch.Generator().manual_seed(seed)
    emb_a = torch.randn(n, 7, 4, generator=gen)
    emb_b = torch.randn(n, 5, 4, generator=gen)
    len_a = torch.tensor([7, 6, 5, 7, 4, 7][:n])
    len_b = torch.tensor([5, 5, 3, 4, 5, 3][:n])
    label = torch.tensor([1, 0, 0, 1, 0, 1][:n])
    return {"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b, "label": label}


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


def _trained_base(seed: int = 1) -> V3_1:
    torch.manual_seed(seed)
    base = V3_1(**_tiny_base_config())
    with torch.no_grad():  # perturb so the frozen function is not the init
        for param in base.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return base


def _wrapped(conditioning: str = "pair", seed: int = 1) -> tuple[V3_1, V3_1Prefix]:
    base = _trained_base(seed)
    model = V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": conditioning, "bottleneck": 6},
    )
    model.base.load_state_dict(base.state_dict())
    return base, model


def test_wrapper_rejects_a_base_without_cross_attention_layers() -> None:
    try:
        V3_1Prefix(base=_tiny_base_config(mixing="none"), prefix={"tokens": 2})
    except ValueError as err:
        assert "bidirectional_cross" in str(err)
    else:
        raise AssertionError("mixing none must be rejected")


def test_null_identity_against_the_frozen_base_in_both_modes() -> None:
    for conditioning in ("static", "pair"):
        base, model = _wrapped(conditioning)
        base.eval()
        batch = _pair_batch()
        with torch.no_grad():
            want = base(batch)["logits"]
            model.train()
            got_train = model(batch)["logits"]
            model.eval()
            got_eval = model(batch)["logits"]
        assert torch.equal(got_train, want), conditioning
        assert torch.equal(got_eval, want), conditioning


def test_base_stays_in_eval_mode_and_frozen_after_wrapper_train() -> None:
    _, model = _wrapped()
    model.train()
    assert model.training
    assert not model.base.training
    assert all(not m.training for m in model.base.modules())
    assert all(not p.requires_grad for p in model.base.parameters())
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all(name.startswith("generator.") for name in trainable)
    assert {id(p) for p in model.prefix_parameters()} == {
        id(p) for p in model.generator.parameters()
    }


def test_loss_backward_reaches_only_prefix_parameters() -> None:
    _, model = _wrapped()
    model.train()
    with torch.no_grad():
        model.generator.gates.fill_(0.3)
    out = model(_pair_batch())
    assert "loss" in out and "loss_weight_sum" in out
    out["loss"].backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is not None for p in model.generator.parameters())


def test_pair_symmetry_holds_with_open_gates() -> None:
    _, model = _wrapped()
    model.eval()
    with torch.no_grad():
        model.generator.gates.fill_(0.4)
    batch = _pair_batch()
    swapped = {
        "emb_a": batch["emb_b"],
        "emb_b": batch["emb_a"],
        "len_a": batch["len_b"],
        "len_b": batch["len_a"],
    }
    with torch.no_grad():
        assert torch.allclose(model(batch)["logits"], model(swapped)["logits"], atol=1e-5)


def test_interventions() -> None:
    base, model = _wrapped()
    base.eval()
    model.eval()
    with torch.no_grad():
        model.generator.gates.fill_(0.4)
        model.generator.z_sum.copy_(torch.randn(6))
        model.generator.z_count.fill_(10.0)
    batch = _pair_batch()
    with torch.no_grad():
        live = model(batch)["logits"]
        model.intervention = "gates_off"
        assert torch.equal(model(batch)["logits"], base(batch)["logits"])
        model.intervention = "mean"
        mean_logits = model(batch)["logits"]
        assert not torch.equal(mean_logits, live)
        model.intervention = "shuffle"
        model.intervention_seed = 5
        shuffled_1 = model(batch)["logits"]
        model.intervention_seed = 5
        shuffled_2 = model(batch)["logits"]
        assert torch.equal(shuffled_1, shuffled_2)
        assert not torch.equal(shuffled_1, live)
        # the same permutation is applied to AB and BA (abba_max stays symmetric)
        swapped = {
            "emb_a": batch["emb_b"],
            "emb_b": batch["emb_a"],
            "len_a": batch["len_b"],
            "len_b": batch["len_a"],
        }
        model.intervention_seed = 5
        assert torch.allclose(model(swapped)["logits"], shuffled_1, atol=1e-5)
    model.intervention = "bogus"
    try:
        with torch.no_grad():
            model(batch)
    except ValueError as err:
        assert "intervention" in str(err)
    else:
        raise AssertionError("unknown intervention must raise")


def test_init_static_prefix_from_a_batch_and_state_dict_round_trip() -> None:
    _, model = _wrapped()
    before = model.generator.p0[0].clone()
    model.init_static_prefix(_pair_batch(), seed=0)
    assert not torch.equal(before, model.generator.p0[0])
    rebuilt = V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": "pair", "bottleneck": 6},
    )
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt(_pair_batch())["logits"], model(_pair_batch())["logits"])
