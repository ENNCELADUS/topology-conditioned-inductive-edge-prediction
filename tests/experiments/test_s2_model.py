"""Tests for `src/experiments/s2_latent_topology/model.py`'s S2 model components.

All CPU, tiny dims throughout: ``in_dim=8, z_dim=4, dim=16, n_heads=4, rrwp_k=3,
seeds=2, d_model=32, width=32, sab_layers=2, heads=2, pma_seeds=2``, node
counts ``n <= 10``.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from src.experiments.s2_latent_topology.model import (
    DetTwin,
    MargDegreeRegressor,
    RegionAutoencoder,
    RegionGraph,
    SetDenoiser,
    ae_losses,
    flow_matching_loss,
    pair_logits,
    sample_prior,
    sinusoidal_time_embedding,
)

pytestmark = pytest.mark.unit

_IN_DIM = 8
_Z_DIM = 4
_DIM = 16
_N_HEADS = 4
_RRWP_K = 3
_SEEDS = 2
_D_MODEL = 32
_WIDTH = 32
_SAB_LAYERS = 2
_HEADS = 2
_PMA_SEEDS = 2


def _region_graph(batch: int, n: int) -> RegionGraph:
    return RegionGraph(
        x=torch.randn(batch, n, _IN_DIM),
        adj=torch.zeros(batch, 1, n, n),
        mask=torch.ones(batch, n),
        aux={},
    )


def _build_ae() -> RegionAutoencoder:
    return RegionAutoencoder(
        in_dim=_IN_DIM,
        z_dim=_Z_DIM,
        dim=_DIM,
        layers=2,
        rrwp_k=_RRWP_K,
        n_heads=_N_HEADS,
        seeds=_SEEDS,
        d_model=_D_MODEL,
    )


def _symmetric_adj(n: int, *, seed: int, p: float = 0.5) -> torch.Tensor:
    graph = nx.gnp_random_graph(n, p, seed=seed)
    return torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32)


def _build_denoiser(*, time_dim: int = 8) -> SetDenoiser:
    return SetDenoiser(
        in_dim=_IN_DIM,
        z_dim=_Z_DIM,
        width=_WIDTH,
        sab_layers=_SAB_LAYERS,
        heads=_HEADS,
        pma_seeds=_PMA_SEEDS,
        time_dim=time_dim,
    )


def _build_det_twin() -> DetTwin:
    return DetTwin(
        in_dim=_IN_DIM,
        z_dim=_Z_DIM,
        width=_WIDTH,
        sab_layers=_SAB_LAYERS,
        heads=_HEADS,
        pma_seeds=_PMA_SEEDS,
    )


# --------------------------------------------------------------------------- RegionGraph


def test_swapped_returns_the_same_object() -> None:
    graph = _region_graph(1, 5)
    assert graph.swapped() is graph


def test_mismatched_shapes_raise_via_inherited_validation() -> None:
    with pytest.raises(ValueError, match="adj"):
        RegionGraph(
            x=torch.randn(1, 5, _IN_DIM),
            adj=torch.zeros(1, 1, 7, 7),
            mask=torch.ones(1, 5),
            aux={},
        )


# --------------------------------------------------------------------------- pair_logits


def test_pair_logits_matches_a_hand_computed_three_node_example() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0]])
    z = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    bias = torch.tensor(0.5)

    logits = pair_logits(a, z, bias)

    # z_0.z_1=0, z_0.z_2=1, z_1.z_2=1
    expected = torch.tensor(
        [
            [
                [1 + 1 + 1 + 0.5, 1 + 2 + 0 + 0.5, 1 + 3 + 1 + 0.5],
                [1 + 2 + 0 + 0.5, 2 + 2 + 1 + 0.5, 2 + 3 + 1 + 0.5],
                [1 + 3 + 1 + 0.5, 2 + 3 + 1 + 0.5, 3 + 3 + 2 + 0.5],
            ]
        ]
    )
    torch.testing.assert_close(logits, expected)


def test_pair_logits_is_symmetric() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 6)
    z = torch.randn(2, 6, _Z_DIM)
    bias = torch.tensor(-0.3)

    logits = pair_logits(a, z, bias)

    torch.testing.assert_close(logits, logits.transpose(-1, -2))


def test_pair_logits_zero_z_dim_equals_the_pure_activity_value() -> None:
    torch.manual_seed(1)
    a = torch.randn(2, 6)
    z = torch.zeros(2, 6, 0)
    bias = torch.tensor(0.25)

    logits = pair_logits(a, z, bias)

    expected = a.unsqueeze(-1) + a.unsqueeze(-2) + bias
    torch.testing.assert_close(logits, expected)


# --------------------------------------------------------------------------- RegionAutoencoder


def test_encode_shape_and_padded_rows_exactly_zero() -> None:
    ae = _build_ae()
    n = 10
    x = torch.randn(2, n, _IN_DIM)
    adj = torch.zeros(2, n, n)
    mask = torch.ones(2, n)
    mask[1, 6:] = 0.0

    latents = ae.encode(x, adj, mask)

    assert latents.shape == (2, n, 1 + _Z_DIM)
    assert torch.equal(latents[1, 6:], torch.zeros(n - 6, 1 + _Z_DIM))


def test_decode_shape() -> None:
    ae = _build_ae()
    n = 7
    latents = torch.randn(2, n, 1 + _Z_DIM)
    mask = torch.ones(2, n)

    logits = ae.decode(latents, mask)

    assert logits.shape == (2, n, n)


def test_z_dim_property_matches_constructor_argument() -> None:
    ae = _build_ae()
    assert ae.z_dim == _Z_DIM


# --------------------------------------------------------------------------- ae_losses


def test_triangle_and_bce_near_zero_for_confident_correct_logits() -> None:
    n = 8
    adj = _symmetric_adj(n, seed=3).unsqueeze(0)
    mask = torch.ones(1, n)
    logits = 30.0 * (2.0 * adj - 1.0)

    losses = ae_losses(logits, adj, mask, lambda_tri=1.0)

    assert losses["triangle"].item() < 1e-6
    assert losses["bce"].item() < 1e-6
    torch.testing.assert_close(
        losses["total"], losses["bce"] + 1.0 * losses["triangle"], atol=1e-6, rtol=1e-6
    )


def test_soft_p_equals_hard_a_triangle_moment_matches_networkx_triangle_count() -> None:
    n = 9
    graph = nx.gnp_random_graph(n, 0.5, seed=7)
    adj = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, n)
    # sigmoid(logits) must equal `adj` exactly (0/1), so drive logits to +-inf.
    logits = 60.0 * (2.0 * adj - 1.0)

    losses = ae_losses(logits, adj, mask, lambda_tri=1.0)

    expected_triangles = sum(nx.triangles(graph).values()) / 3.0
    # triangle penalty == 0 here confirms E_T == T_true == expected_triangles.
    assert losses["triangle"].item() == pytest.approx(0.0, abs=1e-8)
    assert expected_triangles >= 1.0  # sanity: the graph actually has triangles


def test_padding_and_diagonal_are_excluded_from_both_terms() -> None:
    n = 8
    adj = _symmetric_adj(n, seed=11).unsqueeze(0)
    mask = torch.ones(1, n)
    torch.manual_seed(2)
    logits = torch.randn(1, n, n)
    logits = (logits + logits.transpose(-1, -2)) / 2.0
    reference = ae_losses(logits, adj, mask, lambda_tri=0.7)

    n_pad = 12
    adj_pad = torch.zeros(1, n_pad, n_pad)
    adj_pad[:, :n, :n] = adj
    mask_pad = torch.zeros(1, n_pad)
    mask_pad[:, :n] = 1.0
    logits_pad = torch.zeros(1, n_pad, n_pad)
    logits_pad[:, :n, :n] = logits
    # Garbage in the padded region and on the diagonal must not leak in.
    logits_pad[:, n:, :] = 999.0
    logits_pad[:, :, n:] = -999.0
    logits_pad += 500.0 * torch.eye(n_pad).unsqueeze(0)

    padded = ae_losses(logits_pad, adj_pad, mask_pad, lambda_tri=0.7)

    torch.testing.assert_close(padded["bce"], reference["bce"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(padded["triangle"], reference["triangle"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(padded["total"], reference["total"], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- time embedding


def test_sinusoidal_time_embedding_shape() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    embedding = sinusoidal_time_embedding(t, 8)
    assert embedding.shape == (3, 8)
    assert torch.isfinite(embedding).all()


# --------------------------------------------------------------------------- SetDenoiser


def test_denoiser_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser().eval()
    n = 10
    z_t = torch.randn(2, n, 1 + _Z_DIM)
    t = torch.rand(2)
    x = torch.randn(2, n, _IN_DIM)
    mask = torch.ones(2, n)
    mask[1, 8:] = 0.0

    perm = torch.randperm(n)
    with torch.no_grad():
        out = denoiser(z_t, t, x, mask)
        out_perm = denoiser(z_t[:, perm], t, x[:, perm], mask[:, perm])

    assert torch.allclose(out[:, perm], out_perm, atol=1e-5)


def test_denoiser_output_is_sensitive_to_conditioning_features() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser().eval()
    n = 6
    z_t = torch.randn(1, n, 1 + _Z_DIM)
    t = torch.rand(1)
    x = torch.randn(1, n, _IN_DIM)
    mask = torch.ones(1, n)

    with torch.no_grad():
        out = denoiser(z_t, t, x, mask)
        out_zeroed = denoiser(z_t, t, torch.zeros_like(x), mask)

    assert not torch.allclose(out, out_zeroed)


def test_denoiser_padded_rows_are_zero() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser().eval()
    n = 10
    z_t = torch.randn(1, n, 1 + _Z_DIM)
    t = torch.rand(1)
    x = torch.randn(1, n, _IN_DIM)
    mask = torch.ones(1, n)
    mask[0, 7:] = 0.0

    with torch.no_grad():
        out = denoiser(z_t, t, x, mask)

    assert torch.equal(out[0, 7:], torch.zeros(n - 7, 1 + _Z_DIM))


# --------------------------------------------------------------------------- DetTwin


def test_det_twin_shape_and_padded_rows_zero() -> None:
    torch.manual_seed(0)
    det = _build_det_twin().eval()
    n = 9
    x = torch.randn(2, n, _IN_DIM)
    mask = torch.ones(2, n)
    mask[0, 5:] = 0.0

    with torch.no_grad():
        out = det(x, mask)

    assert out.shape == (2, n, 1 + _Z_DIM)
    assert torch.equal(out[0, 5:], torch.zeros(n - 5, 1 + _Z_DIM))


# --------------------------------------------------------------------------- MargDegreeRegressor


def test_marg_degree_regressor_is_nonnegative() -> None:
    torch.manual_seed(0)
    reg = MargDegreeRegressor(in_dim=_IN_DIM, hidden=16)
    x = torch.randn(2, 5, _IN_DIM)
    n = torch.tensor([5, 3])

    out = reg(x, n)

    assert out.shape == (2, 5)
    assert torch.all(out >= 0.0)


def test_marg_degree_regressor_has_strict_per_node_independence() -> None:
    torch.manual_seed(0)
    reg = MargDegreeRegressor(in_dim=_IN_DIM, hidden=16)
    x = torch.randn(1, 5, _IN_DIM)
    n = torch.tensor([5])

    out = reg(x, n)
    x_changed = x.clone()
    x_changed[:, 2] = torch.randn(1, _IN_DIM)
    out_changed = reg(x_changed, n)

    other_rows = torch.tensor([True, True, False, True, True])
    torch.testing.assert_close(out[:, other_rows], out_changed[:, other_rows])
    assert not torch.allclose(out[:, 2], out_changed[:, 2])


# --------------------------------------------------------------------------- flow_matching_loss


def test_flow_matching_loss_is_finite_and_deterministic_under_reseeded_generator() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser()
    n = 6
    z1_std = torch.randn(2, n, 1 + _Z_DIM)
    x = torch.randn(2, n, _IN_DIM)
    mask = torch.ones(2, n)

    gen_a = torch.Generator().manual_seed(42)
    loss_a = flow_matching_loss(denoiser, z1_std, x, mask, p_drop=0.1, generator=gen_a)
    gen_b = torch.Generator().manual_seed(42)
    loss_b = flow_matching_loss(denoiser, z1_std, x, mask, p_drop=0.1, generator=gen_b)

    assert torch.isfinite(loss_a)
    torch.testing.assert_close(loss_a, loss_b)


def test_flow_matching_loss_decreases_with_optimization() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser()
    n = 6
    z1_std = torch.randn(3, n, 1 + _Z_DIM)
    x = torch.randn(3, n, _IN_DIM)
    mask = torch.ones(3, n)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-2)

    def _step(seed: int) -> float:
        gen = torch.Generator().manual_seed(seed)
        loss = flow_matching_loss(denoiser, z1_std, x, mask, p_drop=0.0, generator=gen)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return float(loss.item())

    first = _step(0)
    for step in range(1, 30):
        last = _step(step)

    assert last < first


# --------------------------------------------------------------------------- sample_prior


def test_sample_prior_shape_and_determinism() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser().eval()
    n = 6
    x = torch.randn(2, n, _IN_DIM)
    mask = torch.ones(2, n)
    mask[1, 4:] = 0.0

    gen_a = torch.Generator().manual_seed(7)
    samples_a = sample_prior(denoiser, x, mask, z_dim=_Z_DIM, n_draws=3, steps=4, generator=gen_a)
    gen_b = torch.Generator().manual_seed(7)
    samples_b = sample_prior(denoiser, x, mask, z_dim=_Z_DIM, n_draws=3, steps=4, generator=gen_b)

    assert samples_a.shape == (3, 2, n, 1 + _Z_DIM)
    assert torch.isfinite(samples_a).all()
    torch.testing.assert_close(samples_a, samples_b)


def test_sample_prior_different_draws_differ() -> None:
    torch.manual_seed(0)
    denoiser = _build_denoiser().eval()
    n = 6
    x = torch.randn(1, n, _IN_DIM)
    mask = torch.ones(1, n)

    gen = torch.Generator().manual_seed(3)
    samples = sample_prior(denoiser, x, mask, z_dim=_Z_DIM, n_draws=2, steps=3, generator=gen)

    assert not torch.allclose(samples[0], samples[1])
