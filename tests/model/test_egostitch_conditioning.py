"""Tests for e2e head-null branch masks (design rev 3 §3.5/§4)."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.model.egostitch.classifier import b0_v31 as conditioning_module
from src.model.egostitch.classifier.b0_v31 import (
    NULL_ALL_HEAD,
    NULL_NONE,
    GatedCrossAttention,
    masks_for_null,
    sample_branch_masks,
)
from src.model.egostitch.classifier.base import HeadNullMasks
from src.model.egostitch.config import E2EConfig
from torch import nn


class _FixedAttention(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
    ) -> tuple[torch.Tensor, None]:
        del query, key, value, key_padding_mask, need_weights
        return self.output, None


def _loopback_bind_available() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _world_size_centering_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = "lo0" if "lo0" in interfaces else "lo"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        attention_outputs = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0, 6.0]],
                [[5.0, 6.0, 7.0, 8.0]],
                [[7.0, 8.0, 9.0, 10.0]],
            ]
        )
        start = rank * 2
        local_attn = attention_outputs[start : start + 2]
        module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
        module.attn = _FixedAttention(local_attn)
        with torch.no_grad():
            module.gate.fill_(0.7)
        module.train()
        cls = torch.zeros_like(local_attn)
        active = torch.full((2,), rank == 0, dtype=torch.bool)
        real = torch.ones(2, dtype=torch.bool)
        output = module(cls, local_attn, None, active, real)
        loss_weights = torch.arange(1, 17, dtype=torch.float32).view(4, 1, 4)
        (output * loss_weights[start : start + 2]).sum().backward()
        assert module.gate.grad is not None
        torch.save(
            {
                "mu": module.ema_mu.clone(),
                "output": output.detach(),
                "gate_grad": module.gate.grad.clone(),
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_sample_branch_masks_shapes_and_dtype() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(64, 0.15, generator=gen, device=torch.device("cpu"))
    assert isinstance(masks, HeadNullMasks)
    assert masks.topo.shape == (64,) and masks.topo.dtype == torch.bool


def test_sample_branch_masks_p_zero_all_active() -> None:
    gen = torch.Generator().manual_seed(0)
    masks = sample_branch_masks(32, 0.0, generator=gen, device=torch.device("cpu"))
    assert bool(masks.topo.all())


def test_sample_branch_masks_deterministic_given_seed() -> None:
    a = sample_branch_masks(
        128, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    b = sample_branch_masks(
        128, 0.5, generator=torch.Generator().manual_seed(7), device=torch.device("cpu")
    )
    assert torch.equal(a.topo, b.topo)


def test_sample_branch_masks_topo_stream_is_bit_identical_to_a_direct_draw() -> None:
    """Content-path removal deleted the `cont` draw that used to follow `topo`.

    `topo` was always drawn first from the shared generator (b0_v31.py
    §2 mask-stream ordering), so removing the second (`cont`) draw must not
    perturb it: the mask equals the first ``torch.rand(...) >= p_topo`` draw
    from an identically seeded generator, exactly as it did before the `cont`
    draw existed.
    """
    p_topo = 0.3
    batch_size = 256
    masks = sample_branch_masks(
        batch_size,
        p_topo,
        generator=torch.Generator().manual_seed(1234),
        device=torch.device("cpu"),
    )
    direct = torch.rand(batch_size, generator=torch.Generator().manual_seed(1234)) >= p_topo
    assert torch.equal(masks.topo, direct)


@pytest.mark.parametrize(
    ("null", "topo_on"),
    [
        (NULL_NONE, True),
        (NULL_ALL_HEAD, False),
    ],
)
def test_masks_for_null(null: str, topo_on: bool) -> None:
    masks = masks_for_null(null, 4, torch.device("cpu"))
    assert bool(masks.topo.all()) is topo_on and bool((~masks.topo).all()) is not topo_on


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


def test_conditioning_ema_decay_is_configurable_and_validated() -> None:
    assert E2EConfig().conditioning_ema_decay == 0.99
    with pytest.raises(ValueError, match="conditioning_ema_decay must be in \\(0, 1\\)"):
        E2EConfig(conditioning_ema_decay=1.0)

    module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0, ema_decay=0.8)
    module._update_ema(torch.ones(1, 1, 4))
    module._update_ema(torch.zeros(1, 1, 4))
    torch.testing.assert_close(module.ema_mu, torch.full((1, 1, 4), 0.8))


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


def test_gated_xattn_inactive_row_is_checkpoint_exact_with_ema_mu() -> None:
    module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    module.attn = _FixedAttention(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    with torch.no_grad():
        module.gate.fill_(0.7)
    cls = torch.randn(1, 1, 4)
    module.train()
    module(cls, cls, None, torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    module.eval()
    output = module(
        cls,
        cls,
        None,
        torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )
    assert torch.equal(output, cls)


def test_gated_xattn_eval_is_deterministic() -> None:
    attention_output = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[3.0, 4.0, 5.0, 6.0]]]
    )
    module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    module.attn = _FixedAttention(attention_output)
    with torch.no_grad():
        module.gate.fill_(0.7)
    cls = torch.randn(2, 1, 4)
    active = torch.ones(2, dtype=torch.bool)
    real = torch.ones(2, dtype=torch.bool)
    module.train()
    module(cls, cls, None, active, real)
    module.eval()
    first = module(cls, cls, None, active, real)
    second = module(cls, cls, None, active, real)
    assert torch.equal(first, second)


def test_gated_xattn_world_size_invariant_mu_and_activations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attention_outputs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[3.0, 4.0, 5.0, 6.0]],
            [[5.0, 6.0, 7.0, 8.0]],
            [[7.0, 8.0, 9.0, 10.0]],
        ]
    )
    single = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    single.attn = _FixedAttention(attention_outputs)
    with torch.no_grad():
        single.gate.fill_(0.7)
    single.train()
    single_output = single(
        torch.zeros_like(attention_outputs),
        attention_outputs,
        None,
        torch.tensor([True, True, False, False]),
        torch.ones(4, dtype=torch.bool),
    )
    loss_weights = torch.arange(1, 17, dtype=torch.float32).view(4, 1, 4)
    (single_output * loss_weights).sum().div(2).backward()
    assert single.gate.grad is not None

    if _loopback_bind_available():
        init_file = tmp_path / "gloo-init"
        mp.spawn(
            _world_size_centering_worker,
            args=(2, str(init_file), str(tmp_path)),
            nprocs=2,
            join=True,
        )
        rank_results = [
            torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
        ]
    else:
        active = torch.tensor([True, True, False, False])
        included = active.view(-1, 1, 1).float()
        global_count = included.sum()
        global_reference = attention_outputs[active].amin(dim=0, keepdim=True)
        global_deviation = ((attention_outputs - global_reference) * included).sum(
            dim=0, keepdim=True
        )
        rank_results = []
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        for rank in range(2):
            start = rank * 2
            local_attn = attention_outputs[start : start + 2]
            local_active = active[start : start + 2]
            local_weight = local_active.view(-1, 1, 1).float()
            remote_deviation = global_deviation - (
                (local_attn - global_reference) * local_weight
            ).sum(dim=0, keepdim=True)
            remote_count = global_count - local_weight.sum()

            def emulated_sum(
                tensor: torch.Tensor,
                *,
                op: dist.ReduceOp,
                addend: torch.Tensor = remote_deviation,
            ) -> torch.Tensor:
                assert op == dist.ReduceOp.SUM
                return tensor + addend

            def emulated_count(
                tensor: torch.Tensor,
                op: dist.ReduceOp,
                addend: torch.Tensor = remote_count,
            ) -> None:
                if op == dist.ReduceOp.SUM:
                    tensor.add_(addend)
                else:
                    assert op == dist.ReduceOp.MIN
                    tensor.copy_(global_reference)

            monkeypatch.setattr(conditioning_module.dist_functional, "all_reduce", emulated_sum)
            monkeypatch.setattr(dist, "all_reduce", emulated_count)
            module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
            module.attn = _FixedAttention(local_attn)
            with torch.no_grad():
                module.gate.fill_(0.7)
            module.train()
            output = module(
                torch.zeros_like(local_attn),
                local_attn,
                None,
                local_active,
                torch.ones(2, dtype=torch.bool),
            )
            (output * loss_weights[start : start + 2]).sum().backward()
            assert module.gate.grad is not None
            rank_results.append(
                {
                    "mu": module.ema_mu.clone(),
                    "output": output.detach(),
                    "gate_grad": module.gate.grad.clone(),
                }
            )
    for result in rank_results:
        torch.testing.assert_close(result["mu"], single.ema_mu, rtol=0.0, atol=0.0)
    distributed_output = torch.cat([result["output"] for result in rank_results])
    torch.testing.assert_close(distributed_output, single_output, rtol=0.0, atol=0.0)
    distributed_gate_grad = torch.stack([result["gate_grad"] for result in rank_results]).mean()
    torch.testing.assert_close(distributed_gate_grad, single.gate.grad, rtol=1e-6, atol=1e-6)


def test_gated_xattn_padded_filler_rows_do_not_contribute_to_mu() -> None:
    attention_output = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0, 3.0]], [[999.0] * 4]]
    )
    module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    module.attn = _FixedAttention(attention_output)
    module.train()
    module(
        torch.zeros_like(attention_output),
        attention_output,
        None,
        torch.ones(3, dtype=torch.bool),
        torch.tensor([True, True, False]),
    )
    assert torch.equal(module.ema_mu, torch.full((1, 1, 4), 2.0))


def test_gated_xattn_constant_attention_has_exactly_zero_injection() -> None:
    attention_output = torch.full((7, 1, 4), 0.3)
    module = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    module.attn = _FixedAttention(attention_output)
    with torch.no_grad():
        module.gate.fill_(0.7)
    module.train()
    cls = torch.randn(7, 1, 4)
    output = module(
        cls,
        cls,
        None,
        torch.ones(7, dtype=torch.bool),
        torch.ones(7, dtype=torch.bool),
    )
    assert torch.equal(output, cls)


def test_gated_xattn_ema_round_trips_and_is_frozen_in_eval() -> None:
    torch.manual_seed(0)
    source = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    cls = torch.randn(2, 1, 4)
    tokens = torch.randn(2, 3, 4)
    active = torch.ones(2, dtype=torch.bool)
    real = torch.ones(2, dtype=torch.bool)
    source.train()
    source(cls, tokens, None, active, real)

    restored = GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0)
    restored.load_state_dict(source.state_dict())
    assert torch.equal(restored.ema_mu, source.ema_mu)
    restored.eval()
    before = restored.ema_mu.clone()
    restored(cls, tokens * 10.0, None, active, real)
    assert torch.equal(restored.ema_mu, before)


def test_gated_xattn_loads_legacy_checkpoint_without_ema() -> None:
    source = nn.Sequential(GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0))
    legacy_state = source.state_dict()
    del legacy_state["0.ema_mu"]
    del legacy_state["0.ema_updates"]
    restored = nn.Sequential(GatedCrossAttention(d_model=4, n_heads=1, dropout=0.0))
    restored.load_state_dict(legacy_state)
    restored_layer = restored[0]
    assert isinstance(restored_layer, GatedCrossAttention)
    assert torch.count_nonzero(restored_layer.ema_mu) == 0
    assert int(restored_layer.ema_updates.item()) == 0
