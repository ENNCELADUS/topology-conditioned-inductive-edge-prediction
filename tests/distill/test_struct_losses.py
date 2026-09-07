from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
import torch
from src.distill.struct_config import StructConfig
from src.distill.struct_losses import (
    hard_struct_errors,
    struct_bce,
    struct_deg_mmd,
    struct_degree,
    struct_gs,
    struct_motif,
    struct_rank,
    struct_rd,
    struct_total,
)


def _graph(
    n: int, edges: list[tuple[int, int]], illegal: list[tuple[int, int]] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(n, n)
    for u, v in edges:
        target[u, v] = target[v, u] = 1.0
    mask = 1.0 - torch.eye(n)
    for u, v in illegal or []:
        mask[u, v] = mask[v, u] = 0.0
    return target * mask, mask


def _logits_from(target: torch.Tensor, scale: float = 12.0) -> torch.Tensor:
    return ((target * 2.0 - 1.0) * scale).requires_grad_(True)


def _brute_motifs(a: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = a.size(0)
    closed = torch.zeros(n, n)
    open_ = torch.zeros(n, n)
    for u, v, w in itertools.permutations(range(n), 3):
        if mask[u, w] and mask[w, v] and a[u, w] and a[w, v]:
            if a[u, v]:
                closed[u, v] += 1
            else:
                open_[u, v] += 1
    return closed, open_


def test_motif_matches_brute_force_on_hard_graph() -> None:
    target, mask = _graph(
        6, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)], illegal=[(1, 4)]
    )
    logits = _logits_from(target, scale=40.0)
    closed, open_ = _brute_motifs(target, mask)
    p = torch.sigmoid(logits.detach()) * mask
    c_soft = p * (p @ p)
    o_soft = (1.0 - p) * (p @ p)
    sel = torch.triu(mask, 1) > 0
    torch.testing.assert_close(c_soft[sel], closed[sel], atol=1e-4, rtol=0.0)
    torch.testing.assert_close(o_soft[sel], open_[sel], atol=1e-4, rtol=0.0)
    assert float(struct_motif(logits, target, mask, huber_delta=1.0)) < 1e-6


def test_motif_balanced_mean_weights_classes_equally() -> None:
    # One closed pair, one open pair, many zero pairs: the loss is the mean of three class means.
    target, mask = _graph(8, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)])
    logits = torch.zeros(8, 8, requires_grad=True)
    p = torch.sigmoid(logits.detach()) * mask
    a = target
    c_p, o_p = p * (p @ p), (1 - p) * (p @ p)
    c_a, o_a = a * (a @ a), (1 - a) * (a @ a)
    sel = torch.triu(mask, 1) > 0
    huber = torch.nn.functional.smooth_l1_loss
    err = huber(torch.log1p(c_p[sel]), torch.log1p(c_a[sel]), reduction="none") + huber(
        torch.log1p(o_p[sel]), torch.log1p(o_a[sel]), reduction="none"
    )
    ca, oa = c_a[sel], o_a[sel]
    classes = [ca > 0, (ca == 0) & (oa > 0), (ca == 0) & (oa == 0)]
    expected = torch.stack([err[c].mean() for c in classes]).mean()
    torch.testing.assert_close(struct_motif(logits, target, mask, huber_delta=1.0), expected)


def test_rank_vanishes_for_separated_logits_and_grows_when_inverted() -> None:
    target, mask = _graph(5, [(0, 1), (0, 2), (3, 4)])
    good = _logits_from(target, scale=10.0)
    bad = _logits_from(1.0 - target - torch.eye(5), scale=10.0)
    assert float(struct_rank(good, target, mask, margin=0.1, temperature=1.0)) < 1e-3
    assert float(struct_rank(bad, target, mask, margin=0.1, temperature=1.0)) > 5.0


def test_rank_weights_anchors_equally() -> None:
    # Anchor 0 has 3 positives x 1 negative, anchor 4 has 1 x 3: with all-zero logits and no
    # margin every (j, k) pair costs log 2, so equal anchor weighting gives exactly log 2.
    target, mask = _graph(5, [(0, 1), (0, 2), (0, 3), (4, 1)])
    logits = torch.zeros(5, 5)
    loss_zero = struct_rank(logits, target, mask, margin=0.0, temperature=1.0)
    torch.testing.assert_close(loss_zero, torch.log(torch.tensor(2.0)))


def test_gs_and_rd_reproduce_grand_hand_values() -> None:
    # 4 nodes, edges (0,1),(2,3); predict p=0.5 on every legal pair (logit 0).
    target, mask = _graph(4, [(0, 1), (2, 3)])
    logits = torch.zeros(4, 4, requires_grad=True)
    # GS = sum|p-a| / (sum p + sum a): 6 pairs -> |0.5-1|*2 + |0.5-0|*4 = 3; denominator 3 + 2 = 5.
    torch.testing.assert_close(
        struct_gs(logits, target, mask), torch.tensor(3.0 / 5.0), atol=1e-6, rtol=0.0
    )
    # RD: log(3/2) = 0.405 < delta -> SmoothL1 = 0.5 * 0.405^2.
    expected = 0.5 * float(torch.log(torch.tensor(1.5))) ** 2
    torch.testing.assert_close(
        struct_rd(logits, target, mask, huber_delta=1.0),
        torch.tensor(expected),
        atol=1e-6,
        rtol=0.0,
    )


def test_degree_uses_in_subgraph_legal_degrees() -> None:
    # Node 0's legal degree is 2 because pair (0,3) is masked; confident legal edges hit zero loss.
    target, mask = _graph(4, [(0, 1), (0, 2), (0, 3)], illegal=[(0, 3)])
    logits = _logits_from(target, scale=40.0)
    assert float(struct_degree(logits, target, mask, huber_delta=1.0)) < 1e-5
    # Predicting the masked pair as an edge must not change the degree target or the loss.
    with_masked = logits.detach().clone()
    with_masked[0, 3] = with_masked[3, 0] = 40.0
    assert float(struct_degree(with_masked, target, mask, huber_delta=1.0)) < 1e-5


def test_deg_mmd_vanishes_at_target_and_grows_off_target() -> None:
    target, mask = _graph(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)])
    at_target = _logits_from(target, scale=40.0)
    assert float(struct_deg_mmd(at_target, target, mask, sigma=1.25, bins=48)) < 1e-5
    empty = torch.full((7, 7), -40.0)
    assert float(struct_deg_mmd(empty, target, mask, sigma=1.25, bins=48)) > 0.05


def test_bce_uses_positive_weight_over_legal_upper_triangle() -> None:
    target, mask = _graph(3, [(0, 1)])
    logits = torch.zeros(3, 3, requires_grad=True)
    # Three legal pairs, all at p=0.5: weighted mean of log 2 is log 2 whatever the weights are.
    torch.testing.assert_close(
        struct_bce(logits, target, mask, positive_weight=5.0), torch.log(torch.tensor(2.0))
    )
    # Logit 4 everywhere: the positive costs softplus(-4) at weight 5, each negative
    # softplus(4) at weight 1, normalised by the weight sum 7.
    tilted = torch.tensor([[0.0, 4.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 0.0]])
    softplus = torch.nn.functional.softplus
    expected = (5.0 * softplus(torch.tensor(-4.0)) + 2.0 * softplus(torch.tensor(4.0))) / 7.0
    torch.testing.assert_close(struct_bce(tilted, target, mask, positive_weight=5.0), expected)


_ALL_TERMS = [
    (struct_bce, {"positive_weight": 5.0}),
    (struct_gs, {}),
    (struct_rd, {"huber_delta": 1.0}),
    (struct_deg_mmd, {"sigma": 1.25, "bins": 48}),
    (struct_rank, {"margin": 0.1, "temperature": 1.0}),
    (struct_degree, {"huber_delta": 1.0}),
    (struct_motif, {"huber_delta": 1.0}),
]


@pytest.mark.parametrize(("fn", "kwargs"), _ALL_TERMS[1:])
def test_terms_vanish_when_prediction_equals_target(
    fn: Callable[..., torch.Tensor], kwargs: dict[str, float]
) -> None:
    target, mask = _graph(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)])
    logits = _logits_from(target, scale=40.0)
    assert float(fn(logits, target, mask, **kwargs)) < 1e-5


@pytest.mark.parametrize(("fn", "kwargs"), _ALL_TERMS)
def test_masked_entries_receive_zero_gradient(
    fn: Callable[..., torch.Tensor], kwargs: dict[str, float]
) -> None:
    target, mask = _graph(6, [(0, 1), (1, 2), (2, 0), (3, 4)], illegal=[(0, 3), (2, 5)])
    raw = torch.randn(6, 6, generator=torch.Generator().manual_seed(0))
    logits = ((raw + raw.T) / 2).fill_diagonal_(0.0).requires_grad_(True)
    fn(logits, target, mask, **kwargs).backward()  # type: ignore[no-untyped-call]
    grad = logits.grad
    assert grad is not None
    masked = grad[0, 3].abs() + grad[3, 0].abs() + grad[2, 5].abs() + grad[5, 2].abs()
    assert float(masked) == 0.0
    assert float(grad.diagonal().abs().sum()) == 0.0


@pytest.mark.parametrize(("fn", "kwargs"), _ALL_TERMS)
def test_empty_reduction_sets_return_differentiable_zero(
    fn: Callable[..., torch.Tensor], kwargs: dict[str, float]
) -> None:
    target, mask = _graph(3, [], illegal=[(0, 1), (0, 2), (1, 2)])
    logits = torch.zeros(3, 3, requires_grad=True)
    value = fn(logits, target, mask, **kwargs)
    assert value.requires_grad
    assert float(value) == 0.0


def test_total_computes_only_active_terms_and_weights_them() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0, "motif": 0.5}})
    target, mask = _graph(5, [(0, 1), (1, 2), (0, 2)])
    logits = torch.zeros(5, 5, requires_grad=True)
    total, terms = struct_total(logits, target, mask, cfg, positive_weight=5.0, label_smoothing=0.0)
    assert set(terms) == {"bce", "motif"}
    torch.testing.assert_close(total, terms["bce"] * 1.0 + terms["motif"] * 0.5)
    assert total.requires_grad


def test_hard_errors_are_zero_at_perfect_threshold_and_count_misses() -> None:
    target, mask = _graph(5, [(0, 1), (1, 2), (0, 2), (3, 4)])
    logits = _logits_from(target, scale=4.0).detach()
    assert hard_struct_errors(logits, target, mask, threshold=0.0) == {
        "hard_degree_mae": 0.0,
        "hard_triangle_mae": 0.0,
        "hard_wedge_mae": 0.0,
    }
    # Threshold above every logit: nothing predicted, so degree error is the mean true degree.
    missed = hard_struct_errors(logits, target, mask, threshold=10.0)
    assert missed["hard_degree_mae"] == pytest.approx(8.0 / 5.0)
    assert missed["hard_triangle_mae"] == pytest.approx(3.0 / 5.0)
