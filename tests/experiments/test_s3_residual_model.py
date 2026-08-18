"""Tests for `src/experiments/s3_set_residual/model.py`'s set-conditioned residual.

All CPU, tiny dims throughout: ``d_in=8, d_model=16, sab_layers=2, heads=2,
pma_seeds=2, p_dim=6, head_hidden=12``, node counts ``n <= 6``.
"""

from __future__ import annotations

import pytest
import torch
from src.experiments.s2_latent_topology.model import _SetBackbone
from src.experiments.s3_set_residual.model import (
    ResidualConfig,
    SetResidualModel,
    _matched_node_mlp,
    _param_count,
)

pytestmark = pytest.mark.unit

_D_IN = 8
_D_MODEL = 16
_SAB_LAYERS = 2
_HEADS = 2
_PMA_SEEDS = 2
_P_DIM = 6
_HEAD_HIDDEN = 12


def _cfg(mode: str) -> ResidualConfig:
    return ResidualConfig(
        mode=mode,  # type: ignore[arg-type]
        d_in=_D_IN,
        d_model=_D_MODEL,
        sab_layers=_SAB_LAYERS,
        heads=_HEADS,
        pma_seeds=_PMA_SEEDS,
        p_dim=_P_DIM,
        head_hidden=_HEAD_HIDDEN,
    )


def _inputs(batch: int, n: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(batch, n, _D_IN)
    mask = torch.ones(batch, n)
    return x, mask


def _all_pairs(batch: int, n: int) -> torch.Tensor:
    """Every ``i < j`` pair in every set of a `batch`-set, `n`-node batch."""
    rows = []
    for b in range(batch):
        for i in range(n):
            for j in range(i + 1, n):
                rows.append((b, i, j))
    return torch.tensor(rows, dtype=torch.long)


class _ZeroBackbone(torch.nn.Module):
    """Test double: always returns zeros, whatever the (tokens, mask) input."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self._d_model = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        return torch.zeros(tokens.shape[0], tokens.shape[1], self._d_model)


# --------------------------------------------------------------------------- ResidualConfig


def test_config_defaults_match_s2_backbone_convention() -> None:
    cfg = ResidualConfig(mode="res")
    assert cfg.d_in == 1536
    assert cfg.d_model == 384
    assert cfg.sab_layers == 4
    assert cfg.heads == 4
    assert cfg.pma_seeds == 4


# --------------------------------------------------------------------------- _matched_node_mlp


def test_matched_node_mlp_hits_target_param_count_closely() -> None:
    reference = _SetBackbone(
        in_dim=_D_MODEL,
        out_dim=_D_MODEL,
        width=_D_MODEL,
        sab_layers=_SAB_LAYERS,
        heads=_HEADS,
        pma_seeds=_PMA_SEEDS,
    )
    target = _param_count(reference)

    mlp = _matched_node_mlp(_D_MODEL, target)

    assert _param_count(mlp) == pytest.approx(target, rel=0.1)


def test_matched_node_mlp_shape_is_pointwise() -> None:
    mlp = _matched_node_mlp(_D_MODEL, target_params=500)
    x = torch.randn(2, 5, _D_MODEL)

    out = mlp(x)

    assert out.shape == (2, 5, _D_MODEL)


# ------------------------------------------------------------------- construction / param counts


def test_res_and_diag_have_identical_total_parameter_counts() -> None:
    res = SetResidualModel(_cfg("res"))
    diag = SetResidualModel(_cfg("diag"))

    assert _param_count(res) == _param_count(diag)


def test_pair_total_param_count_within_10_percent_of_res() -> None:
    res = SetResidualModel(_cfg("res"))
    pair = SetResidualModel(_cfg("pair"))

    res_total = _param_count(res)
    pair_total = _param_count(pair)

    assert pair_total == pytest.approx(res_total, rel=0.1)


def test_pair_mode_has_no_set_backbone_module() -> None:
    pair = SetResidualModel(_cfg("pair"))
    res = SetResidualModel(_cfg("res"))

    assert not isinstance(pair.backbone, _SetBackbone)
    assert isinstance(res.backbone, _SetBackbone)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        SetResidualModel(_cfg("bogus"))


# --------------------------------------------------------------------------- zero-init contract


@pytest.mark.parametrize("mode", ["res", "diag", "pair"])
def test_zero_init_delta_is_exactly_zero(mode: str) -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg(mode)).eval()
    n = 5
    x, mask = _inputs(1, n, seed=1)
    pairs = _all_pairs(1, n)

    with torch.no_grad():
        delta = model(x, mask, pairs)

    assert torch.equal(delta, torch.zeros_like(delta))


# --------------------------------------------------------------------------- symmetry


@pytest.mark.parametrize("mode", ["res", "diag", "pair"])
def test_delta_is_symmetric_under_i_j_swap(mode: str) -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg(mode))
    # Break zero-init so the symmetry check is non-trivial.
    for p in model.pair_out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    n = 5
    x, mask = _inputs(1, n, seed=2)
    pairs = _all_pairs(1, n)
    swapped = pairs.clone()
    swapped[:, 1], swapped[:, 2] = pairs[:, 2].clone(), pairs[:, 1].clone()

    with torch.no_grad():
        delta = model(x, mask, pairs)
        delta_swapped = model(x, mask, swapped)

    torch.testing.assert_close(delta, delta_swapped)


# ------------------------------------------------------------------- diag: pointwise invariance


def test_diag_mode_delta_is_invariant_to_other_nodes_features() -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg("diag"))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    n = 6
    x, mask = _inputs(1, n, seed=3)
    pairs = torch.tensor([[0, 0, 1]], dtype=torch.long)

    x_changed = x.clone()
    x_changed[:, 2:] = torch.randn(1, n - 2, _D_IN)
    x_changed[:, 5] = torch.randn(_D_IN) * 100.0  # large, arbitrary perturbation

    with torch.no_grad():
        delta = model(x, mask, pairs)
        delta_changed = model(x_changed, mask, pairs)

    torch.testing.assert_close(delta, delta_changed)


def test_res_mode_delta_is_not_invariant_to_other_nodes_features() -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg("res"))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    n = 6
    x, mask = _inputs(1, n, seed=3)
    pairs = torch.tensor([[0, 0, 1]], dtype=torch.long)

    x_changed = x.clone()
    x_changed[:, 2:] = torch.randn(1, n - 2, _D_IN)
    x_changed[:, 5] = torch.randn(_D_IN) * 100.0

    with torch.no_grad():
        delta = model(x, mask, pairs)
        delta_changed = model(x_changed, mask, pairs)

    assert not torch.allclose(delta, delta_changed)


# ------------------------------------------------------------------- SHUF: p-side unaffected


def test_x_set_override_only_touches_backbone_not_raw_feature_projection() -> None:
    """With the backbone silenced, `x_set` permutation cannot move delta.

    `p = Linear(x)` never reads `x_set`, so if the backbone's contribution is
    forced to zero regardless of its input, delta must be identical whatever
    `x_set` says -- this is what "``x_set`` overrides only the backbone input"
    means, verified by construction rather than by reading the source.
    """
    torch.manual_seed(0)
    model = SetResidualModel(_cfg("res"))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    model.backbone = _ZeroBackbone(_D_MODEL)
    n = 5
    x, mask = _inputs(1, n, seed=4)
    pairs = _all_pairs(1, n)
    x_set_a = torch.randn(1, n, _D_IN)
    x_set_b = x_set_a[:, torch.randperm(n)]

    with torch.no_grad():
        delta_a = model(x, mask, pairs, x_set=x_set_a)
        delta_b = model(x, mask, pairs, x_set=x_set_b)

    torch.testing.assert_close(delta_a, delta_b)


def test_pair_mode_ignores_x_set() -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg("pair"))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    n = 5
    x, mask = _inputs(1, n, seed=5)
    pairs = _all_pairs(1, n)
    x_set = torch.randn(1, n, _D_IN)

    with torch.no_grad():
        delta_no_override = model(x, mask, pairs)
        delta_with_override = model(x, mask, pairs, x_set=x_set)

    torch.testing.assert_close(delta_no_override, delta_with_override)


# ------------------------------------------------------------------- gradients / determinism


@pytest.mark.parametrize("mode", ["res", "diag", "pair"])
def test_gradients_are_finite(mode: str) -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg(mode))
    n = 5
    x, mask = _inputs(2, n, seed=6)
    pairs = _all_pairs(2, n)

    delta = model(x, mask, pairs)
    loss = (delta**2).sum()
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("mode", ["res", "diag", "pair"])
def test_forward_is_deterministic(mode: str) -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg(mode)).eval()
    n = 5
    x, mask = _inputs(1, n, seed=7)
    pairs = _all_pairs(1, n)

    with torch.no_grad():
        delta_a = model(x, mask, pairs)
        delta_b = model(x, mask, pairs)

    torch.testing.assert_close(delta_a, delta_b)


# --------------------------------------------------------------------------- output dtype / shape


@pytest.mark.parametrize("mode", ["res", "diag", "pair"])
def test_output_shape_and_dtype(mode: str) -> None:
    torch.manual_seed(0)
    model = SetResidualModel(_cfg(mode)).eval()
    n = 4
    x, mask = _inputs(3, n, seed=8)
    pairs = _all_pairs(3, n)

    with torch.no_grad():
        delta = model(x, mask, pairs)

    assert delta.shape == (pairs.shape[0],)
    assert delta.dtype == torch.float32
