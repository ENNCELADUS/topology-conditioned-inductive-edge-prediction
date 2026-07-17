"""Tests for the stitched-topology encoder (design rev 3 §3.3)."""

import torch
from src.model.egostitch.scaffold import ScaffoldTokens, build_scaffold
from src.model.egostitch.ste import STEncoder

from tests.model.test_egostitch_scaffold import _slots


def _scaffold(seed_i: int = 0, seed_j: int = 1) -> ScaffoldTokens:
    torch.manual_seed(42)
    return build_scaffold(_slots(seed=seed_i), _slots(seed=seed_j), torch.rand(2, 4, 4))


def test_ste_output_shape() -> None:
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2)
    out = enc(_scaffold())
    assert out.shape == (2, 10, 64)


def test_ste_permutation_equivariance() -> None:
    """Permuting scaffold nodes permutes STE tokens identically."""
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2).eval()
    sc = _scaffold()
    perm = torch.randperm(sc.feats.size(1))
    sc_p = ScaffoldTokens(feats=sc.feats[:, perm], adj=sc.adj[:, :, perm][:, :, :, perm])
    out, out_p = enc(sc), enc(sc_p)
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_ste_is_structure_sensitive() -> None:
    """Rewiring adjacency (same tokens) must change the output."""
    torch.manual_seed(0)
    enc = STEncoder(d_model=64, ste_dim=32, n_layers=2).eval()
    sc = _scaffold()
    rewired = ScaffoldTokens(feats=sc.feats, adj=sc.adj.flip(dims=[2]))
    assert not torch.allclose(enc(sc), enc(rewired), atol=1e-4)
