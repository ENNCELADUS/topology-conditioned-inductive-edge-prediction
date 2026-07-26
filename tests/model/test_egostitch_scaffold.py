"""Tests for the structure-only stitched scaffold (design rev 3 §3.1–§3.2)."""

import json
import subprocess
import sys

import pytest
import src.model.egostitch.scaffold as scaffold_module
import torch
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.scaffold import (
    EDGE_TYPES,
    FEAT_DIM,
    N_ANCHOR_TYPES,
    ContentProjector,
    ScaffoldTokens,
    _checkerboard_rewire,
    build_content_tokens,
    build_scaffold,
    make_scaffold_input_perturbation,
    swap_direction,
)
from src.model.egostitch.ste import STEncoder


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
    assert FEAT_DIM == 11
    assert EDGE_TYPES == 4
    v = 2 + 2 * 4
    assert out.feats.shape == (2, v, FEAT_DIM)
    assert out.adj.shape == (2, EDGE_TYPES, v, v)
    # anchor one-hots: exactly one label per node, correct blocks
    onehot = out.feats[..., :N_ANCHOR_TYPES]
    assert torch.equal(onehot.sum(-1), torch.ones(2, v))
    assert bool(onehot[:, 0, 0].all()) and bool(onehot[:, 1, 1].all())
    assert bool(onehot[:, 2 : 2 + 4, 2].all()) and bool(onehot[:, 2 + 4 :, 3].all())
    assert torch.allclose(out.feats[..., 6:10], out.adj.sum(dim=-1).permute(0, 2, 1))


def test_scaffold_rebuild_symmetry() -> None:
    assert (FEAT_DIM, EDGE_TYPES) == (11, 4)
    si, sj = _slots(seed=0), _slots(seed=1)
    plan = torch.rand(2, 4, 4)
    ij = build_scaffold(si, sj, plan)
    ji = build_scaffold(sj, si, plan.transpose(1, 2))
    side_perm = torch.tensor([1, 0, 6, 7, 8, 9, 2, 3, 4, 5])
    anchor_perm = torch.tensor([1, 0, 3, 2])
    expected_feats = ij.feats[:, side_perm].clone()
    expected_feats[..., :N_ANCHOR_TYPES] = expected_feats[..., anchor_perm]
    expected_adj = ij.adj[:, :, side_perm][:, :, :, side_perm]
    assert torch.allclose(ji.feats, expected_feats, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ji.adj, expected_adj, atol=1e-6, rtol=1e-6)


def test_closed_wedge_feature_ignores_adjacency_diagonal() -> None:
    si, sj = _slots(b=1, k=3, seed=0), _slots(b=1, k=3, seed=1)
    plan = torch.tensor([[[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.4, 0.2]]])
    diagonal_only_i = si._replace(adj=torch.diag_embed(torch.tensor([[2.0, 3.0, 4.0]])))
    diagonal_only_j = sj._replace(adj=torch.diag_embed(torch.tensor([[5.0, 6.0, 7.0]])))
    zero_off_diag = build_scaffold(diagonal_only_i, diagonal_only_j, plan)
    assert torch.equal(zero_off_diag.feats[..., 10], torch.zeros(1, 8))

    baseline = build_scaffold(si, sj, plan)
    adj_i_zero = si.adj - torch.diag_embed(torch.diagonal(si.adj, dim1=-2, dim2=-1))
    adj_j_zero = sj.adj - torch.diag_embed(torch.diagonal(sj.adj, dim1=-2, dim2=-1))
    expected_i = torch.diagonal(plan @ adj_j_zero @ plan.transpose(1, 2), dim1=-2, dim2=-1)
    expected_j = torch.diagonal(plan.transpose(1, 2) @ adj_i_zero @ plan, dim1=-2, dim2=-1)
    assert torch.allclose(baseline.feats[:, 2:5, 10], expected_i, atol=1e-6)
    assert torch.allclose(baseline.feats[:, 5:8, 10], expected_j, atol=1e-6)
    changed_i = si._replace(
        adj=si.adj + torch.diag_embed(torch.tensor([[10.0, 20.0, 30.0]]))
    )
    changed_j = sj._replace(
        adj=sj.adj + torch.diag_embed(torch.tensor([[40.0, 50.0, 60.0]]))
    )
    changed = build_scaffold(changed_i, changed_j, plan)
    assert torch.allclose(changed.feats[..., 10], baseline.feats[..., 10], atol=1e-6)


def test_closure_edge_matches_hand_built_three_slot_example() -> None:
    si, sj = _slots(b=1, k=3, seed=0), _slots(b=1, k=3, seed=1)
    si = si._replace(
        adj=torch.tensor([[[9.0, 1.0, 2.0], [1.0, 8.0, 3.0], [2.0, 3.0, 7.0]]])
    )
    sj = sj._replace(
        adj=torch.tensor([[[6.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 4.0]]])
    )
    plan = torch.diag_embed(torch.tensor([[1.0, 2.0, 3.0]]))
    out = build_scaffold(si, sj, plan)
    expected = torch.tensor([[[0.0, 3.0, 5.5], [4.5, 0.0, 10.5], [8.5, 12.0, 0.0]]])
    assert torch.allclose(out.adj[:, 3, 2:5, 5:8], expected, atol=1e-6)
    assert torch.allclose(out.adj[:, 3, 5:8, 2:5], expected.transpose(1, 2), atol=1e-6)


def test_closure_fp32_island_preserves_values_and_gradients_under_bf16() -> None:
    si, sj = _slots(b=1, k=3, seed=0), _slots(b=1, k=3, seed=1)
    adj_i = si.adj.to(torch.bfloat16).detach().requires_grad_()
    adj_j = sj.adj.to(torch.bfloat16).detach().requires_grad_()
    plan = (
        torch.tensor([[[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.4, 0.2]]])
        .to(torch.bfloat16)
        .requires_grad_()
    )
    si = si._replace(
        pi=si.pi.to(torch.bfloat16),
        mult=si.mult.to(torch.bfloat16),
        adj=adj_i,
    )
    sj = sj._replace(
        pi=sj.pi.to(torch.bfloat16),
        mult=sj.mult.to(torch.bfloat16),
        adj=adj_j,
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = build_scaffold(si, sj, plan)

    assert out.feats.dtype == torch.float32
    assert out.adj.dtype == torch.float32
    plan32, adj_i32, adj_j32 = plan.float(), adj_i.float(), adj_j.float()
    adj_i_zero = adj_i32 - torch.diag_embed(torch.diagonal(adj_i32, dim1=-2, dim2=-1))
    adj_j_zero = adj_j32 - torch.diag_embed(torch.diagonal(adj_j32, dim1=-2, dim2=-1))
    expected_c = 0.5 * (adj_i_zero @ plan32 + plan32 @ adj_j_zero)
    expected_t_i = torch.diagonal(
        plan32 @ adj_j_zero @ plan32.transpose(1, 2), dim1=-2, dim2=-1
    )
    expected_t_j = torch.diagonal(
        plan32.transpose(1, 2) @ adj_i_zero @ plan32, dim1=-2, dim2=-1
    )
    assert torch.allclose(out.adj[:, 3, 2:5, 5:8], expected_c, atol=1e-6)
    assert torch.allclose(out.feats[:, 2:5, 10], expected_t_i, atol=1e-6)
    assert torch.allclose(out.feats[:, 5:8, 10], expected_t_j, atol=1e-6)

    (out.adj[:, 3].sum() + out.feats[..., 10].sum()).backward()
    for grad in (plan.grad, adj_i.grad, adj_j.grad):
        assert grad is not None
        assert bool(torch.isfinite(grad).all())
        assert bool((grad != 0).any())


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


def test_rewire_checkerboard_preserves_rebuilt_visible_degrees_and_moves_close() -> None:
    si, sj = _slots(b=2, k=4, seed=10), _slots(b=2, k=4, seed=11)
    plan = torch.rand(2, 4, 4, generator=torch.Generator().manual_seed(12))
    baseline = build_scaffold(si, sj, plan)
    controlled = build_scaffold(
        si,
        sj,
        plan,
        perturbation=make_scaffold_input_perturbation(
            "rewire_checkerboard_v1", [("node-a", "node-b"), ("node-c", "node-d")]
        ),
    )

    assert torch.allclose(
        controlled.feats[..., 6:9],
        baseline.feats[..., 6:9],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.max(torch.abs(controlled.feats[..., 9] - baseline.feats[..., 9])).item() > 1e-4


def test_rewire_checkerboard_keeps_adjacency_symmetric_with_zero_diagonal() -> None:
    si, sj = _slots(b=2, k=4, seed=13), _slots(b=2, k=4, seed=14)
    plan = torch.rand(2, 4, 4, generator=torch.Generator().manual_seed(15))
    rewired_i, rewired_j, rewired_plan = make_scaffold_input_perturbation(
        "rewire_checkerboard_v1", [("node-a", "node-b"), ("node-c", "node-d")]
    )(si.adj, sj.adj, plan, si.pi, sj.pi)

    for rewired in (rewired_i, rewired_j):
        assert torch.allclose(rewired, rewired.transpose(1, 2), atol=1e-6, rtol=1e-6)
        assert rewired.min().item() >= -torch.finfo(torch.float32).eps
        assert torch.equal(
            torch.diagonal(rewired, dim1=-2, dim2=-1),
            torch.zeros_like(si.pi),
        )
    assert rewired_plan.min().item() >= -torch.finfo(torch.float32).eps


def test_rewire_checkerboard_vectorizes_cell_disjoint_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_shapes: list[tuple[torch.Size, torch.Size, int]] = []
    original_scan = scaffold_module.scan

    def scan_spy(
        combine_fn: object,
        init: torch.Tensor,
        xs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draw_indices, units = xs
        scan_shapes.append(
            (draw_indices.shape, units.shape, int(torch.count_nonzero(units).item()))
        )
        return original_scan(combine_fn, init, xs)

    monkeypatch.setattr(scaffold_module, "scan", scan_spy)
    generator = torch.Generator().manual_seed(16)
    _checkerboard_rewire(torch.rand(1, 16, 16, generator=generator), [generator])

    # K=16: 2,048 keyed swaps become 37 sequential rounds of up to 56
    # cell-disjoint transfers, rather than 2,048 scalar scan/Python steps.
    assert scan_shapes == [(torch.Size([37, 1, 56, 4]), torch.Size([37, 1, 56]), 2048)]


def test_scaffold_controls_are_cross_process_deterministic_and_pair_keyed() -> None:
    code = """
import json
import torch
from src.model.egostitch.scaffold import make_scaffold_input_perturbation

adj = torch.tensor([[
    [0.0, 0.2, 0.7, 0.4],
    [0.2, 0.0, 0.4, 0.8],
    [0.7, 0.4, 0.0, 0.3],
    [0.4, 0.8, 0.3, 0.0],
]])
plan = torch.tensor([[
    [0.1, 0.2, 0.3, 0.4],
    [0.4, 0.5, 0.6, 0.7],
    [0.7, 0.8, 0.9, 1.0],
    [0.3, 0.5, 0.7, 0.9],
]])
pi = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
out = {}
for mode in ("shuffle_within_pair_v3", "rewire_checkerboard_v1"):
    values = make_scaffold_input_perturbation(mode, [("node-a", "node-b")])(
        adj, adj, plan, pi, pi
    )
    other = make_scaffold_input_perturbation(mode, [("node-a", "node-c")])(
        adj, adj, plan, pi, pi
    )
    out[mode] = {
        "same_pair": [value.tolist() for value in values],
        "other_pair": [value.tolist() for value in other],
    }
print(json.dumps(out, sort_keys=True))
"""
    first = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    assert json.loads(first) == json.loads(second)
    payload = json.loads(first)
    for mode in ("shuffle_within_pair_v3", "rewire_checkerboard_v1"):
        assert payload[mode]["same_pair"] != payload[mode]["other_pair"]


@pytest.mark.parametrize("mode", ["shuffle_within_pair_v3", "rewire_checkerboard_v1"])
def test_scaffold_control_measurably_moves_noncollapsed_ste_output(mode: str) -> None:
    torch.manual_seed(21)
    si, sj = _slots(b=2, k=4, seed=22), _slots(b=2, k=4, seed=23)
    plan = torch.rand(2, 4, 4)
    ste = STEncoder(d_model=16, ste_dim=8, n_layers=2).eval()
    baseline = ste(build_scaffold(si, sj, plan))
    controlled = ste(
        build_scaffold(
            si,
            sj,
            plan,
            perturbation=make_scaffold_input_perturbation(
                mode, [("node-a", "node-b"), ("node-c", "node-d")]
            ),
        )
    )
    assert torch.max(torch.abs(controlled - baseline)).item() > 1e-6


def test_shuffle_v3_rebuilds_closed_wedge_and_degree_features() -> None:
    si, sj = _slots(b=2, k=4, seed=31), _slots(b=2, k=4, seed=32)
    plan = torch.rand(2, 4, 4, generator=torch.Generator().manual_seed(33))
    baseline = build_scaffold(si, sj, plan)
    controlled = build_scaffold(
        si,
        sj,
        plan,
        perturbation=make_scaffold_input_perturbation(
            "shuffle_within_pair_v3", [("node-a", "node-b"), ("node-c", "node-d")]
        ),
    )
    assert not torch.equal(controlled.feats[..., 6:10], baseline.feats[..., 6:10])
    assert not torch.equal(controlled.feats[..., 10], baseline.feats[..., 10])


def test_content_tokens_shape_and_content_sensitivity() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    matched = torch.zeros(2, 4)
    membership = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = build_content_tokens(si, sj, matched, matched, membership, -membership)
    assert out.shape == (2, 8, 8 + 4)
    assert torch.equal(out[:, :4, -1], membership)
    assert torch.equal(out[:, 4:, -1], -membership)
    si_b = si._replace(h=torch.randn_like(si.h))
    out_b = build_content_tokens(si_b, sj, matched, matched, membership, -membership)
    assert not torch.equal(out, out_b)


def test_content_projector_shape() -> None:
    torch.manual_seed(0)
    proj = ContentProjector(d_p=8, d_model=64)
    si, sj = _slots(seed=0), _slots(seed=1)
    zeros = torch.zeros(2, 4)
    tokens = build_content_tokens(si, sj, zeros, zeros, zeros, zeros)
    assert proj(tokens).shape == (2, 8, 64)


def test_content_tokens_swap_sides_is_a_token_permutation() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    matched_i, matched_j = torch.zeros(2, 4), torch.ones(2, 4)
    membership_i = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    membership_j = -membership_i
    ij = build_content_tokens(si, sj, matched_i, matched_j, membership_i, membership_j)
    ji = build_content_tokens(sj, si, matched_j, matched_i, membership_j, membership_i)
    assert torch.equal(ij[:, :4], ji[:, 4:])
    assert torch.equal(ij[:, 4:], ji[:, :4])


def test_scaffold_and_content_reject_shape_drift() -> None:
    si, sj = _slots(seed=0), _slots(seed=1)
    with pytest.raises(ValueError, match="plan shape"):
        build_scaffold(si, sj, torch.rand(2, 4, 3))
    with pytest.raises(ValueError, match="equal slot counts"):
        build_scaffold(si, _slots(k=3, seed=2), torch.rand(2, 4, 3))
    zeros = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="matched_src shape"):
        build_content_tokens(si, sj, zeros[:, :3], zeros, zeros, zeros)
    with pytest.raises(ValueError, match="membership_dst shape"):
        build_content_tokens(si, sj, zeros, zeros, zeros, zeros[:, :3])
