"""Motif-prompt model tests: config, generator, reader, prefix interface, identity."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import SWAP_PERM
from src.model.egostitch.classifier.motif_prompt import (
    FIELD_ORDER,
    GATE_MODES,
    MotifCountHead,
    MotifPromptConfig,
    ReaderConfig,
    dense_adjacency,
    motif_rrwp,
    typed_adjacency,
)
from src.model.egostitch.encoder.grit_gmt import dense_rrwp


def test_config_round_trip_and_defaults() -> None:
    cfg = MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "base.pt"})
    assert cfg.stage == "one"
    assert cfg.width == 128
    assert cfg.slots_per_field == 2
    assert cfg.fields == FIELD_ORDER
    assert cfg.families == ("closure", "bridge")
    assert cfg.token_source == "graph"
    assert cfg.gate_mode == "learned"
    assert cfg.interface_warmup_epochs == 2
    assert (cfg.w_slot, cfg.w_topo) == (1.0, 0.1)
    assert (cfg.beta_p, cfg.beta_q, cfg.beta_a, cfg.beta_i) == (1.0, 1.0, 1.0, 1.0)
    assert cfg.huber_delta == 1.0
    assert cfg.reader == ReaderConfig(layers=3, dim=96, heads=4, rrwp_k=4)
    assert MotifPromptConfig.from_mapping(cfg.to_dict()) == cfg


def test_config_validation_rejects_illegal_blocks() -> None:
    with pytest.raises(ValueError, match="unknown motif_prompt keys"):
        MotifPromptConfig.from_mapping({"stage": "one", "base_checkpoint": "b.pt", "tokens": 4})
    with pytest.raises(ValueError, match="motif_prompt.stage"):
        MotifPromptConfig(stage="three", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="requires motif_prompt.base_checkpoint"):
        MotifPromptConfig(stage="one")
    with pytest.raises(ValueError, match="requires motif_prompt.bundle_checkpoint"):
        MotifPromptConfig(stage="two", base_checkpoint="b.pt")
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=("topo_self", "topo_self"))
    with pytest.raises(ValueError, match="motif_prompt.fields"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", fields=())
    with pytest.raises(ValueError, match="motif_prompt.families"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", families=())
    with pytest.raises(ValueError, match="motif_prompt.gate_mode"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", gate_mode="hard_topk")
    with pytest.raises(ValueError, match="reader.dim"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(dim=97, heads=4))
    with pytest.raises(ValueError, match="reader.rrwp_k"):
        MotifPromptConfig(stage="one", base_checkpoint="b.pt", reader=ReaderConfig(rrwp_k=0))


def test_gate_modes_are_the_three_spec_controls() -> None:
    assert GATE_MODES == ("learned", "per_type", "mean_graph")


def _weights(n: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 96, generator=gen)


def test_count_head_is_exactly_swap_invariant() -> None:
    torch.manual_seed(0)
    head = MotifCountHead(width=16)
    weights = _weights()
    swapped = weights[:, list(SWAP_PERM)]
    torch.testing.assert_close(head(weights), head(swapped), rtol=0, atol=1e-6)


def test_count_head_runs_in_fp32_under_autocast_and_stays_finite_at_zero() -> None:
    head = MotifCountHead(width=16)
    zero = torch.zeros(3, 96, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = head(zero)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert zero.grad is not None and torch.isfinite(zero.grad).all()


def test_dense_adjacency_is_symmetric_with_a_zero_diagonal_and_no_uv_entry() -> None:
    adj = dense_adjacency(_weights())
    assert adj.shape == (4, 26, 26)
    torch.testing.assert_close(adj, adj.transpose(1, 2))
    assert float(torch.diagonal(adj, dim1=1, dim2=2).abs().sum()) == 0.0
    assert float(adj[:, 0, 1].abs().sum()) == 0.0
    assert float(adj[:, 1, 0].abs().sum()) == 0.0


def test_typed_adjacency_splits_the_three_edge_types_and_sums_to_the_untyped_one() -> None:
    typed = typed_adjacency(_weights())
    assert typed.shape == (4, 26, 26, 3)
    torch.testing.assert_close(typed.sum(dim=-1), dense_adjacency(_weights()))


def test_rrwp_arithmetic_returns_fp32_inside_an_active_autocast_context() -> None:
    adj = dense_adjacency(_weights()).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stack = motif_rrwp(adj, k=4)
    assert stack.dtype == torch.float32
    assert stack.shape == (4, 26, 26, 4)
    torch.testing.assert_close(stack[..., 0], torch.eye(26).expand(4, 26, 26), rtol=0, atol=0)


def test_rrwp_under_autocast_is_bit_identical_to_the_autocast_free_arithmetic() -> None:
    # An fp32 *input* is not enough: `torch.bmm` returns bf16 from fp32 operands
    # inside an active autocast context, `torch.stack` then promotes the mixed
    # list back to fp32, and the walk powers are silently quantised. Only the
    # disabled-autocast context reproduces the fp32 arithmetic (spec section 5.2).
    adj = dense_adjacency(_weights(seed=11))
    reference = motif_rrwp(adj, k=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        guarded = motif_rrwp(adj, k=4)
        naive = dense_rrwp(adj.float(), 4).float()
    assert torch.equal(guarded, reference)
    assert not torch.equal(naive, reference)
    assert float((naive - reference).abs().max()) > 1e-4


def test_rrwp_is_finite_at_a_zero_adjacency_and_keeps_the_gradient_connection() -> None:
    weights = torch.zeros(2, 96, requires_grad=True)
    stack = motif_rrwp(dense_adjacency(weights), k=4)
    assert torch.isfinite(stack).all()
    stack.sum().backward()  # type: ignore[no-untyped-call]
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
