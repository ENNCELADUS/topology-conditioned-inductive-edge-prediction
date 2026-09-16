"""L_slot order statistics and mass pinning; L_topo directions and stream averaging."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import (
    CLOSURE_U,
    CLOSURE_V,
    SWAP_PERM,
    count_statistics,
    role_permutation,
)
from src.distill.motif_losses import slot_loss, slot_loss_rows, stream_mean, topo_loss_rows

_BETAS = {"beta_p": 1.0, "beta_q": 1.0, "beta_a": 1.0, "beta_i": 1.0, "huber_delta": 1.0}
_FAMILIES = ("beta_p", "beta_q", "beta_a", "beta_i")


def _only(name: str) -> dict[str, float]:
    betas = dict.fromkeys(_FAMILIES, 0.0)
    betas[name] = 1.0
    betas["huber_delta"] = 1.0
    return betas


def test_a_matched_profile_scores_zero_and_has_zero_gradient() -> None:
    weights = torch.rand(3, 96, requires_grad=True)
    loss = slot_loss(weights, weights.detach().clone(), **_BETAS)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    loss.backward()  # type: ignore[no-untyped-call]
    assert weights.grad is not None
    assert float(weights.grad.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_a_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        slot_loss_rows(torch.zeros(2, 96), torch.zeros(3, 96), **_BETAS)


def test_wedge_mass_mismatch_is_penalised_although_the_edge_multisets_agree() -> None:
    target = torch.zeros(1, 96)
    target[0, 0] = 0.5  # w(u, c0)
    target[0, 8] = 0.5  # w(c0, v)
    predicted = torch.zeros(1, 96)
    predicted[0, 0] = 0.5  # w(u, c0)
    predicted[0, 1] = 0.5  # w(u, c1)
    assert sorted(predicted[0, :16].tolist()) == sorted(target[0, :16].tolist())
    masses = count_statistics(predicted)["wedge_mass"], count_statistics(target)["wedge_mass"]
    assert (float(masses[0]), float(masses[1])) == (0.0, 0.25)
    loss = slot_loss(predicted, target, **_only("beta_p"))
    assert float(loss) > 0.0
    assert float(loss) == pytest.approx(0.00390625, rel=1e-9)


def test_the_slot_misalignment_fixture_of_the_withdrawn_collapse_control() -> None:
    target = torch.zeros(1, 96)
    target[0, 0] = target[0, 8] = 0.5
    predicted = torch.zeros(1, 96)
    predicted[0, 0] = predicted[0, 8] = 0.25
    predicted[0, 1] = predicted[0, 9] = 0.25
    loss = slot_loss(predicted, target, **_only("beta_p"))
    assert float(loss) == pytest.approx(0.00244140625, rel=1e-9)


def test_a_bridge_mass_mismatch_is_penalised_although_the_raw_multisets_agree() -> None:
    target = torch.zeros(1, 96)
    target[0, 16] = 1.0  # w(u, l0)
    target[0, 24] = 1.0  # w(r0, v)
    target[0, 32] = 1.0  # w(l0, r0)
    predicted = torch.zeros(1, 96)
    predicted[0, 16] = 1.0  # w(u, l0)
    predicted[0, 25] = 1.0  # w(r1, v), so no path closes
    predicted[0, 32] = 1.0  # w(l0, r0)
    for family in ("beta_a", "beta_i"):
        assert float(slot_loss(predicted, target, **_only(family))) == 0.0
    assert float(count_statistics(predicted)["bridge_mass"]) == 0.0
    assert float(count_statistics(target)["bridge_mass"]) == 1.0
    loss = slot_loss(predicted, target, **_only("beta_q"))
    assert float(loss) == pytest.approx(0.0078125, rel=1e-9)


def test_matching_sorted_products_pin_both_count_head_masses() -> None:
    gen = torch.Generator().manual_seed(7)
    target = torch.rand(4, 96, generator=gen)
    perm = torch.as_tensor(
        role_permutation(
            (3, 4, 5, 6, 7, 0, 1, 2), (1, 0, 3, 2, 5, 4, 7, 6), (7, 6, 5, 4, 3, 2, 1, 0)
        )
    )
    predicted = torch.zeros_like(target)
    predicted[:, perm] = target
    # Move every wedge's weight across its two sides: the products, and so the
    # supervised multisets, are untouched while the edge assignment is not.
    left = predicted[:, CLOSURE_U].clone()
    predicted[:, CLOSURE_U] = predicted[:, CLOSURE_V]
    predicted[:, CLOSURE_V] = left
    assert not torch.allclose(predicted, target)
    rows = slot_loss_rows(predicted, target, **_BETAS)
    assert float(rows.abs().max()) == pytest.approx(0.0, abs=1e-12)
    ours = count_statistics(predicted)
    theirs = count_statistics(target)
    for mass in ("wedge_mass", "bridge_mass"):
        torch.testing.assert_close(ours[mass], theirs[mass], rtol=1e-6, atol=1e-7)


def test_loss_is_invariant_to_a_within_role_permutation_of_either_argument() -> None:
    predicted = torch.rand(2, 96)
    target = torch.rand(2, 96)
    perm = torch.as_tensor(
        role_permutation(
            (1, 0, 3, 2, 5, 4, 7, 6), (4, 5, 6, 7, 0, 1, 2, 3), (2, 3, 4, 5, 6, 7, 0, 1)
        )
    )
    shuffled = torch.zeros_like(predicted)
    shuffled[:, perm] = predicted
    shuffled_target = torch.zeros_like(target)
    shuffled_target[:, perm] = target
    base = slot_loss_rows(predicted, target, **_BETAS)
    torch.testing.assert_close(
        slot_loss_rows(shuffled, target, **_BETAS), base, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        slot_loss_rows(shuffled, shuffled_target, **_BETAS), base, rtol=1e-6, atol=1e-7
    )


def test_loss_is_invariant_under_the_endpoint_swap() -> None:
    perm = torch.as_tensor(SWAP_PERM)
    gen = torch.Generator().manual_seed(11)
    predicted = torch.rand(100, 96, generator=gen)
    target = torch.rand(100, 96, generator=gen)
    swapped = torch.zeros_like(predicted)
    swapped[:, perm] = predicted
    swapped_target = torch.zeros_like(target)
    swapped_target[:, perm] = target
    base = slot_loss_rows(predicted, target, **_BETAS)
    after = slot_loss_rows(swapped, swapped_target, **_BETAS)
    # The swap transposes the interior matrix, so the three factors of a path
    # product are multiplied in a different order; in float32 that costs an ulp.
    assert float((base - after).abs().max()) < 1e-8
    # On a dyadic grid every triple product is exact and the swap is bit exact.
    grid = torch.randint(0, 17, (100, 96), generator=gen).float() / 16.0
    grid_target = torch.randint(0, 17, (100, 96), generator=gen).float() / 16.0
    swapped_grid = torch.zeros_like(grid)
    swapped_grid[:, perm] = grid
    swapped_grid_target = torch.zeros_like(grid_target)
    swapped_grid_target[:, perm] = grid_target
    assert torch.equal(
        slot_loss_rows(grid, grid_target, **_BETAS),
        slot_loss_rows(swapped_grid, swapped_grid_target, **_BETAS),
    )


def test_each_family_weight_switches_off_exactly_its_own_term() -> None:
    gen = torch.Generator().manual_seed(9)
    predicted = torch.rand(2, 96, generator=gen)
    target = torch.rand(2, 96, generator=gen)
    singles = {name: slot_loss_rows(predicted, target, **_only(name)) for name in _FAMILIES}
    for rows in singles.values():
        assert float(rows.min()) > 0.0
    torch.testing.assert_close(
        slot_loss_rows(predicted, target, **_BETAS),
        torch.stack(list(singles.values())).sum(dim=0),
        rtol=1e-6,
        atol=1e-7,
    )
    # The closure-supervised fallback of spec section 7.3: beta_q = beta_i = 0.
    torch.testing.assert_close(
        slot_loss_rows(
            predicted, target, beta_p=1.0, beta_q=0.0, beta_a=1.0, beta_i=0.0, huber_delta=1.0
        ),
        singles["beta_p"] + singles["beta_a"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_rows_are_independent_so_self_rows_can_be_masked_out() -> None:
    gen = torch.Generator().manual_seed(5)
    predicted = torch.rand(3, 96, generator=gen, requires_grad=True)
    target = torch.rand(3, 96, generator=gen)
    rows = slot_loss_rows(predicted, target, **_BETAS)
    assert rows.shape == (3,)
    for index in range(3):
        one = slice(index, index + 1)
        torch.testing.assert_close(
            slot_loss_rows(predicted[one].detach(), target[one], **_BETAS),
            rows[one].detach(),
        )
    (rows * torch.tensor([1.0, 0.0, 1.0])).sum().backward()  # type: ignore[no-untyped-call]
    assert predicted.grad is not None
    assert float(predicted.grad[1].abs().max()) == 0.0
    assert float(predicted.grad[0].abs().max()) > 0.0
    assert float(predicted.grad[2].abs().max()) > 0.0


@pytest.mark.parametrize("scale", [0.0, 1e-8])
def test_gradients_are_finite_at_zero_and_at_tiny_weights(scale: float) -> None:
    logits = torch.full((2, 96), -20.0, requires_grad=True)
    live = torch.sigmoid(logits)
    predicted = live - live.detach() + scale  # exactly ``scale``, with a live gradient path
    target = torch.full((2, 96), 0.5)
    loss = slot_loss(predicted, target, **_BETAS)
    loss.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gradients_reach_all_96_edges_on_a_non_degenerate_fixture() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = (torch.rand(4, 96, generator=gen) * 2 - 1).requires_grad_(True)
    target = torch.rand(4, 96, generator=gen)
    slot_loss(torch.sigmoid(logits), target, **_BETAS).backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    reached = (logits.grad.abs() > 0).any(dim=0)
    assert int(reached.sum()) == 96


def _interior_fixture(beta_i: float) -> float:
    """Return ``|dL/d(interior logit)|`` on the fixture of spec section 7.3.

    Every attachment is 0.1, every predicted interior 0.1 and every true
    interior 1, so the predicted and compiled bridge masses are 0.064 and 0.64.

    Args:
        beta_i: Weight of the raw interior term.

    Returns:
        The largest absolute gradient reaching an interior logit.
    """
    interior_logit = torch.full((1, 64), float(torch.logit(torch.tensor(0.1))), requires_grad=True)
    predicted = torch.zeros(1, 96)
    predicted[0, 16:32] = 0.1
    predicted = predicted.clone()
    predicted[0, 32:] = torch.sigmoid(interior_logit)[0]
    target = torch.zeros(1, 96)
    target[0, 16:32] = 0.1
    target[0, 32:] = 1.0
    assert float(count_statistics(predicted.detach())["bridge_mass"]) == pytest.approx(
        0.064, rel=1e-4
    )
    assert float(count_statistics(target)["bridge_mass"]) == pytest.approx(0.64, rel=1e-4)
    loss = slot_loss(
        predicted, target, beta_p=1.0, beta_q=1.0, beta_a=1.0, beta_i=beta_i, huber_delta=1.0
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert interior_logit.grad is not None
    return float(interior_logit.grad.abs().max())


def test_the_analytic_interior_derivative_rises_by_exactly_1e4_with_beta_i() -> None:
    attachment, interior, true_interior = 0.1, 0.1, 1.0
    sigmoid_slope = interior * (1.0 - interior)
    err_q = attachment * interior * attachment - attachment * true_interior * attachment
    err_b = interior - true_interior
    # Spec section 9: (1/64) * err_q * a * c from the product term against
    # (1/64) * err_b from the interior term, both through the sigmoid slope.
    product_term = abs(err_q * attachment * attachment) / 64.0 * sigmoid_slope
    interior_term = abs(err_b) / 64.0 * sigmoid_slope
    without = _interior_fixture(0.0)
    with_interior = _interior_fixture(1.0)
    assert without == pytest.approx(product_term, rel=1e-5)
    assert with_interior == pytest.approx(product_term + interior_term, rel=1e-5)
    assert without == pytest.approx(1.266e-7, rel=2e-3)
    assert with_interior == pytest.approx(1.266e-3, rel=2e-3)
    assert with_interior / without == pytest.approx(1e4, rel=5e-3)


def test_the_product_term_alone_reproduces_the_documented_mean_of_4_05e_minus_5() -> None:
    predicted = torch.zeros(1, 96)
    predicted[0, 16:32] = 0.1
    predicted[0, 32:] = 0.1
    target = torch.zeros(1, 96)
    target[0, 16:32] = 0.1
    target[0, 32:] = 1.0
    loss = slot_loss(predicted, target, **_only("beta_q"))
    assert float(loss) == pytest.approx(4.05e-5, rel=1e-3)


_FIELDS = ("topo_u", "topo_v", "topo_rel", "topo_cnt")


def _tokens(n: int = 3, width: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.randn(n, width, generator=gen) for name in _FIELDS}


def test_topo_loss_is_zero_on_identical_tokens_and_ignores_a_global_shift_and_scale() -> None:
    tokens = _tokens()
    rows = topo_loss_rows(tokens, tokens)
    assert rows.shape == (3,)
    assert float(rows.abs().max()) == pytest.approx(0.0, abs=1e-6)
    scaled = {name: 3.0 * value + 5.0 for name, value in tokens.items()}
    assert float(topo_loss_rows(scaled, tokens).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_topo_loss_detaches_the_teacher_but_not_the_student() -> None:
    student = {name: value.clone().requires_grad_(True) for name, value in _tokens(seed=1).items()}
    teacher = {name: value.clone().requires_grad_(True) for name, value in _tokens(seed=2).items()}
    topo_loss_rows(student, teacher).sum().backward()  # type: ignore[no-untyped-call]
    assert all(value.grad is not None for value in student.values())
    assert all(value.grad is None for value in teacher.values())


def test_topo_loss_covers_all_four_fields() -> None:
    student = _tokens(seed=1)
    teacher = _tokens(seed=1)
    for name in _FIELDS:
        perturbed = dict(student)
        perturbed[name] = student[name] + 1.5
        assert float(topo_loss_rows(perturbed, teacher).sum()) > 0.0


def test_topo_loss_raises_when_a_field_is_missing() -> None:
    tokens = _tokens()
    partial = {name: value for name, value in tokens.items() if name != "topo_cnt"}
    with pytest.raises(KeyError, match="topo_cnt"):
        topo_loss_rows(partial, tokens)


def test_an_empty_stream_contributes_a_differentiable_zero() -> None:
    anchor = torch.zeros((), requires_grad=True)
    rows = torch.stack([torch.tensor(0.5), torch.tensor(1.5)]) + anchor
    mask = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(stream_mean([rows], [mask], like=anchor), torch.tensor(0.5))
    empty = stream_mean([], [], like=anchor)
    assert float(empty) == 0.0
    assert empty.requires_grad
    torch.testing.assert_close(
        stream_mean([rows, rows.new_zeros(0)], [mask, mask.new_zeros(0)], like=anchor),
        torch.tensor(0.5),
    )


def test_stream_mean_weights_streams_equally_regardless_of_their_row_counts() -> None:
    anchor = torch.zeros((), requires_grad=True)
    big = torch.full((10,), 2.0) + anchor
    small = torch.full((1,), 0.0) + anchor
    combined = stream_mean([big, small], [torch.ones(10), torch.ones(1)], like=anchor)
    torch.testing.assert_close(combined, torch.tensor(1.0))
    combined.backward()  # type: ignore[no-untyped-call]
    assert anchor.grad is not None


def test_a_fully_masked_stream_drops_out_rather_than_dividing_by_zero() -> None:
    anchor = torch.zeros((), requires_grad=True)
    rows = torch.tensor([1.0, 3.0]) + anchor
    masked = stream_mean([rows, rows], [torch.ones(2), torch.zeros(2)], like=anchor)
    torch.testing.assert_close(masked, torch.tensor(2.0))
    assert float(stream_mean([rows], [torch.zeros(2)], like=anchor)) == 0.0
