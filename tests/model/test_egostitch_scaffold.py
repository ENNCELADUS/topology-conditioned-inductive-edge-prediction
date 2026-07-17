"""Tests for the structure-only stitched scaffold (design rev 3 §3.1–§3.2)."""

import torch
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.scaffold import (
    EDGE_TYPES,
    FEAT_DIM,
    N_ANCHOR_TYPES,
    ScaffoldTokens,
    build_scaffold,
    swap_direction,
)


def _slots(b: int = 2, k: int = 4, d_p: int = 8, seed: int = 0) -> SlotSet:
    g = torch.Generator().manual_seed(seed)
    adj = torch.rand(b, k, k, generator=g)
    adj = 0.5 * (adj + adj.transpose(1, 2))
    return SlotSet(
        h=torch.randn(b, k, d_p, generator=g),
        pi=torch.rand(b, k, generator=g),
        mult=1.0 + torch.rand(b, k, generator=g),
        gate=torch.rand(b, k, generator=g),
        pointer=torch.rand(b, k, 3, generator=g),
        adj=adj,
    )


def test_scaffold_shapes_and_layout() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    out = build_scaffold(si, sj, plan)
    assert isinstance(out, ScaffoldTokens)
    v = 2 + 2 * 4
    assert out.feats.shape == (2, v, FEAT_DIM)
    assert out.adj.shape == (2, EDGE_TYPES, v, v)
    # anchor one-hots: exactly one label per node, correct blocks
    onehot = out.feats[..., :N_ANCHOR_TYPES]
    assert torch.equal(onehot.sum(-1), torch.ones(2, v))
    assert bool(onehot[:, 0, 0].all()) and bool(onehot[:, 1, 1].all())
    assert bool(onehot[:, 2 : 2 + 4, 2].all()) and bool(onehot[:, 2 + 4 :, 3].all())


def test_scaffold_contains_no_content_features() -> None:
    # identical structure, different content (h/gate/pointer) => identical scaffold
    si_a, sj = _slots(seed=0), _slots(seed=1)
    si_b = si_a._replace(
        h=torch.randn_like(si_a.h),
        gate=torch.rand_like(si_a.gate),
        pointer=torch.rand_like(si_a.pointer),
    )
    plan = torch.rand(2, 4, 4)
    a, b = build_scaffold(si_a, sj, plan), build_scaffold(si_b, sj, plan)
    assert torch.equal(a.feats, b.feats) and torch.equal(a.adj, b.adj)


def test_scaffold_adj_symmetric_and_star_weights() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    out = build_scaffold(si, sj, plan)
    assert torch.allclose(out.adj, out.adj.transpose(2, 3), atol=1e-6)
    # star edge endpoint_src -> its slot k carries pi*mult
    expected = si.pi * si.mult
    assert torch.allclose(out.adj[:, 0, 0, 2 : 2 + 4], expected, atol=1e-6)


def test_swap_direction_is_involution_and_relabels() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    fwd = build_scaffold(si, sj, plan)
    rev = swap_direction(fwd)
    # anchor channels swapped: src<->dst, slot-of-src<->slot-of-dst
    assert torch.equal(rev.feats[..., 0], fwd.feats[..., 1])
    assert torch.equal(rev.feats[..., 2], fwd.feats[..., 3])
    # non-label features and structure unchanged
    assert torch.equal(rev.feats[..., 4:], fwd.feats[..., 4:])
    assert torch.equal(rev.adj, fwd.adj)
    back = swap_direction(rev)
    assert torch.equal(back.feats, fwd.feats) and torch.equal(back.adj, fwd.adj)
