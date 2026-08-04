"""Tests for the `grit_gmt` graph encoder and its vendored `torch_scatter` shim.

Oracle-scaffold design,
`docs/superpowers/specs/2026-08-04-oracle-scaffold-experiment-design.md` §4.

`src/vendor/grit_official/scatter_shim.py` already exists (real, not
speculative) and is fully specified by its own module docstring: it
implements exactly the four call patterns `grit_layer.py` issues, matching
`torch_scatter` semantics including the empty-group fill values it documents
explicitly (`scatter_add`/`reduce='add'` -> 0 for empty groups; `scatter_max`
-> 0 for empty groups, `-1` for the empty-group argmax). Those tests below
are therefore exact, not best-effort.

`src/model/egostitch/encoder/grit_gmt.py` (`GritGmtEncoder`) does not exist
yet at the time this file was written; the whole file degrades to a clean
skip via `pytest.importorskip` if it is still absent. Per the design doc:
dense RRWP at K=8, four `GritTransformerLayer`s, PMA pooling with 4 seeds;
"Token count follows the graph's own N = 2 + 2*K_slot = 34 ... plus the 4 PMA
seed tokens, so `GraphEmbedding.tokens` carries 38 rows; `pooled` is the mean
of the 4 seed tokens" -- the design doc's own wording, which the
shape/pooled-definition tests below hold literally. The seed-tokens-appended-
after-node-tokens ordering (`tokens[:, :N]` = node tokens, `tokens[:, N:]` =
seed tokens) is inferred from that same "34 node + 4 seed" phrasing (nodes
named first) and is *not* independently confirmed -- flagged in the final
report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import torch
from src.model.egostitch.generator.assemble import EDGE_TYPES, FEAT_DIM
from src.model.egostitch.graph import ImaginedGraph
from src.vendor.grit_official.scatter_shim import scatter, scatter_add, scatter_max

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- scatter_shim


def _ref_scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype)
    for i in range(src.shape[0]):
        out[index[i]] += src[i]
    return out


def _ref_scatter_max(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((dim_size,), float("-inf"), dtype=src.dtype)
    arg = torch.full((dim_size,), -1, dtype=torch.long)
    for i in range(src.shape[0]):
        if src[i] > out[index[i]]:
            out[index[i]] = src[i]
            arg[index[i]] = i
    # torch_scatter / the vendored shim fill empty groups with 0, not -inf.
    out = torch.where(torch.isneginf(out), torch.zeros_like(out), out)
    return out, arg


def test_scatter_add_matches_reference_including_empty_groups() -> None:
    torch.manual_seed(0)
    src = torch.randn(20, 5)
    index = torch.randint(0, 8, (20,))
    index[0] = 7  # guarantee group 7 is hit
    dim_size = 10  # groups 8, 9 stay empty
    expected = _ref_scatter_add(src, index, dim_size)
    got = scatter_add(src, index, dim=0, dim_size=dim_size)
    torch.testing.assert_close(got, expected)
    got_generic = scatter(src, index, dim=0, dim_size=dim_size, reduce="add")
    torch.testing.assert_close(got_generic, expected)
    assert torch.equal(got[8], torch.zeros(5))
    assert torch.equal(got[9], torch.zeros(5))


def test_scatter_add_out_kwarg_accumulates_in_place_matching_torch_scatter() -> None:
    torch.manual_seed(1)
    src = torch.randn(6, 3)
    index = torch.tensor([0, 1, 0, 2, 1, 2])
    out = torch.ones(3, 3)
    expected = out.clone()
    for i in range(6):
        expected[index[i]] += src[i]
    got = scatter(src, index, dim=0, out=out, reduce="add")
    torch.testing.assert_close(got, expected)
    assert got is out  # torch_scatter's out= form accumulates into and returns `out`


def test_scatter_max_matches_reference_values_and_argmax() -> None:
    torch.manual_seed(2)
    src = torch.randn(20)
    index = torch.randint(0, 6, (20,))
    dim_size = 6
    expected_val, expected_arg = _ref_scatter_max(src, index, dim_size)
    got_val, got_arg = scatter_max(src, index, dim=0, dim_size=dim_size)
    torch.testing.assert_close(got_val, expected_val)
    torch.testing.assert_close(got_arg, expected_arg)


def test_scatter_max_empty_group_fill_matches_documented_torch_scatter_semantics() -> None:
    """Empty groups must fill values=0, argmax=-1, per the shim's own docstring.

    Groups 3 and 4 never receive an element here, matching a freshly
    allocated `torch_scatter` output.
    """
    src = torch.tensor([1.0, -2.0, 3.0])
    index = torch.tensor([0, 1, 2])
    values, argmax = scatter_max(src, index, dim=0, dim_size=5)
    assert values[3].item() == 0.0
    assert values[4].item() == 0.0
    assert argmax[3].item() == -1
    assert argmax[4].item() == -1


def _grouped_softmax_via_shim(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Mirror `grit_layer.py`'s vendored `pyg_softmax` (subtract-max, exp, normalize)."""
    src_max = scatter_max(src, index, dim=0, dim_size=dim_size)[0]
    shifted = (src - src_max[index]).exp()
    denom = scatter_add(shifted, index, dim=0, dim_size=dim_size)[index] + 1e-16
    return shifted / denom


def _ref_grouped_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros_like(src)
    for group in range(dim_size):
        member_idx = (index == group).nonzero(as_tuple=True)[0]
        if member_idx.numel() == 0:
            continue
        values = src[member_idx]
        weights = (values - values.max()).exp()
        out[member_idx] = weights / weights.sum()
    return out


def test_pyg_softmax_via_shim_matches_reference_with_a_genuinely_empty_group() -> None:
    """Only grouped rows matter for `pyg_softmax`'s correctness.

    Empty groups' scatter_max fill value is never gathered back for any real
    row, so this test asserts equality only where members exist -- per the
    task brief's framing of this exact numeric edge case.
    """
    torch.manual_seed(3)
    dim_size = 5
    index = torch.randint(0, dim_size, (15,))
    index = index[index != 2]  # group 2 is now empty
    src = torch.randn(index.numel())

    got = _grouped_softmax_via_shim(src, index, dim_size)
    expected = _ref_grouped_softmax(src, index, dim_size)
    torch.testing.assert_close(got, expected)

    for group in range(dim_size):
        member_idx = (index == group).nonzero(as_tuple=True)[0]
        if member_idx.numel() == 0:
            continue
        assert torch.allclose(got[member_idx].sum(), torch.tensor(1.0), atol=1e-5)


# --------------------------------------------------------------------------- GritGmtEncoder

pytest.importorskip("src.model.egostitch.encoder.grit_gmt")

from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder  # noqa: E402

_D_MODEL = 32
_SEEDS = 4
_RRWP_K = 8
_N_HEADS = 8


def _build_encoder(*, d_model: int = _D_MODEL, layers: int = 2) -> GritGmtEncoder:
    return GritGmtEncoder(
        in_dim=FEAT_DIM,
        num_relations=EDGE_TYPES,
        d_model=d_model,
        dim=d_model,
        layers=layers,
        w_rel=0.0,
        rrwp_k=_RRWP_K,
        n_heads=_N_HEADS,
        seeds=_SEEDS,
    )


def _random_graph(batch: int, n: int, *, all_pad_row: int | None = None) -> ImaginedGraph:
    torch.manual_seed(0)
    x = torch.randn(batch, n, FEAT_DIM)
    adj = torch.rand(batch, EDGE_TYPES, n, n)
    mask = torch.ones(batch, n)
    if all_pad_row is not None:
        mask[all_pad_row] = 0.0
    return ImaginedGraph(x=x, adj=adj, mask=mask, aux={})


def test_shapes_and_finiteness_with_one_all_pad_graph() -> None:
    encoder = _build_encoder().eval()
    graph = _random_graph(3, 34, all_pad_row=2)
    with torch.no_grad():
        out = encoder(graph)
    assert out.tokens.shape == (3, 34 + _SEEDS, _D_MODEL)
    assert out.pooled.shape == (3, _D_MODEL)
    assert torch.isfinite(out.tokens).all()
    assert torch.isfinite(out.pooled).all()
    assert torch.isfinite(out.tokens[2]).all()


def test_pooled_is_exactly_the_mean_of_the_seed_tokens() -> None:
    """`pooled` must equal the mean of the trailing `seeds` token positions.

    Design doc §4: "pooled is the mean of the 4 seed tokens, not a learned
    readout" -- assumes seed tokens are the *last* `seeds` positions of
    `tokens` (the "34 node + 4 seed" ordering); flagged as an assumption in
    the module docstring above.
    """
    encoder = _build_encoder().eval()
    graph = _random_graph(2, 12)
    with torch.no_grad():
        out = encoder(graph)
    seed_tokens = out.tokens[:, -_SEEDS:, :]
    torch.testing.assert_close(out.pooled, seed_tokens.mean(dim=1))


def test_permutation_of_valid_nodes_leaves_pooled_and_seed_tokens_invariant() -> None:
    """Permutation invariance of pooled/seed tokens; node tokens permute along.

    GMT-style PMA pooling attends over the node token *set*, so both the
    pooled summary and the seed tokens themselves should be permutation
    invariant, while the permuted node tokens permute accordingly.
    """
    torch.manual_seed(0)
    encoder = _build_encoder().eval()
    n = 12
    x = torch.randn(1, n, FEAT_DIM)
    adj = torch.rand(1, EDGE_TYPES, n, n)
    mask = torch.ones(1, n)
    graph = ImaginedGraph(x=x, adj=adj, mask=mask, aux={})

    perm = torch.randperm(n)
    permuted = ImaginedGraph(
        x=x[:, perm],
        adj=adj[:, :, perm][:, :, :, perm],
        mask=mask[:, perm],
        aux={},
    )
    with torch.no_grad():
        out = encoder(graph)
        out_p = encoder(permuted)

    assert torch.allclose(out.pooled, out_p.pooled, atol=1e-4)
    assert torch.allclose(out.tokens[:, :n][:, perm], out_p.tokens[:, :n], atol=1e-4)
    assert torch.allclose(out.tokens[:, n:], out_p.tokens[:, n:], atol=1e-4)


def test_dense_rrwp_matches_the_official_grit_definition() -> None:
    """`dense_rrwp` reproduces GRIT's `[I, P, ..., P^{K-1}]` stack exactly.

    The official `add_full_rrwp` (upstream grit/transform/rrwp.py,
    `add_identity=True` -- the setting every upstream configs/GRIT/*.yaml uses) puts the
    identity in channel 0 and the (K-1)-th transition power in the last
    channel. A numpy matrix-power reference pins that convention here.
    """
    from src.model.egostitch.encoder.grit_gmt import dense_rrwp

    adj = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    k = _RRWP_K
    got = dense_rrwp(adj.unsqueeze(0), k)
    assert got.shape == (1, 4, 4, k)

    deg = adj.sum(-1, keepdim=True).clamp(min=1e-6)
    p = (adj / deg).numpy()
    reference = np.stack(
        [np.linalg.matrix_power(p, order) for order in range(k)], axis=-1
    )  # (N, N, K), channel 0 = identity
    np.testing.assert_allclose(got.squeeze(0).numpy(), reference, atol=1e-5)

    # Masked variant: an isolated padded node must neither send nor receive
    # walk mass in any non-identity channel.
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    masked = dense_rrwp(adj.unsqueeze(0), k, mask=mask).squeeze(0)
    assert torch.all(masked[3, :, 1:] == 0.0)
    assert torch.all(masked[:, 3, 1:] == 0.0)
