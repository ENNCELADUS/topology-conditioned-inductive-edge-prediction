"""Tests for the graph-encoder component (three-component refactor design §3.3, §5).

`TypedMessagePassingEncoder` is a behaviour-preserving port of the original
`STEncoder` (`src/model/egostitch/ste.py`, deleted once the port landed --
three-component refactor design §5): numerical equivalence via a
`state_dict` transfer between the two was the proof of that at port time,
and it has served its purpose (no live module remains to transfer weights
from or to). What this file pins going forward is the ported encoder's own
behavioural contract: permutation equivariance, structure sensitivity,
shape-contract failure, mask handling, `aux` privacy, substitutability
(dims read from the graph, not hardcoded), and `L_rel` ownership.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import pytest
import torch
from src.model.egostitch.encoder import GraphEncoder, TypedMessagePassingEncoder
from src.model.egostitch.generator.assemble import (
    EDGE_TYPES,
    FEAT_DIM,
    ScaffoldTokens,
    build_scaffold,
)
from src.model.egostitch.generator.losses import relational_loss
from src.model.egostitch.graph import ImaginedGraph

from tests.model.test_egostitch_scaffold import _slots


def _scaffold(seed_i: int = 0, seed_j: int = 1) -> ScaffoldTokens:
    torch.manual_seed(42)
    return build_scaffold(_slots(seed=seed_i), _slots(seed=seed_j), torch.rand(2, 4, 4))


def _graph_from_scaffold(
    sc: ScaffoldTokens, *, aux: Mapping[str, torch.Tensor] | None = None
) -> ImaginedGraph:
    batch, nodes, _ = sc.feats.shape
    mask = torch.ones(batch, nodes)
    return ImaginedGraph(x=sc.feats, adj=sc.adj, mask=mask, aux=aux if aux is not None else {})


class _ForbiddenAux(Mapping[str, torch.Tensor]):
    """An `aux` mapping that fails the test the moment the encoder touches it."""

    def __getitem__(self, key: str) -> torch.Tensor:
        raise AssertionError(f"encoder read graph.aux[{key!r}]; aux is generator-private")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("encoder iterated graph.aux; aux is generator-private")

    def __len__(self) -> int:
        raise AssertionError("encoder measured len(graph.aux); aux is generator-private")


# --------------------------------------------------------------------------- shape


def test_output_shape_matches_ste_encoder_convention() -> None:
    torch.manual_seed(0)
    enc = TypedMessagePassingEncoder(
        in_dim=FEAT_DIM, num_relations=EDGE_TYPES, d_model=64, dim=32, layers=2, w_rel=0.0
    )
    graph = _graph_from_scaffold(_scaffold())
    out = enc(graph)
    assert out.tokens.shape == (2, 10, 64)
    assert out.pooled.shape == (2, 64)


# ---------------------------------------------------------------- equivariance / sensitivity
# `STEncoder`'s own behavioural tests (`tests/model/test_egostitch_ste.py`),
# ported onto `TypedMessagePassingEncoder`/`ImaginedGraph`.


def test_permutation_equivariance() -> None:
    """Permuting graph nodes permutes encoder tokens identically."""
    torch.manual_seed(0)
    enc = TypedMessagePassingEncoder(
        in_dim=FEAT_DIM, num_relations=EDGE_TYPES, d_model=64, dim=32, layers=2, w_rel=0.0
    ).eval()
    graph = _graph_from_scaffold(_scaffold())
    perm = torch.randperm(graph.num_nodes)
    permuted = ImaginedGraph(
        x=graph.x[:, perm],
        adj=graph.adj[:, :, perm][:, :, :, perm],
        mask=graph.mask[:, perm],
        aux={},
    )
    out, out_p = enc(graph).tokens, enc(permuted).tokens
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_is_structure_sensitive() -> None:
    """Rewiring adjacency (same node features) must change the output."""
    torch.manual_seed(0)
    enc = TypedMessagePassingEncoder(
        in_dim=FEAT_DIM, num_relations=EDGE_TYPES, d_model=64, dim=32, layers=2, w_rel=0.0
    ).eval()
    graph = _graph_from_scaffold(_scaffold())
    rewired = ImaginedGraph(x=graph.x, adj=graph.adj.flip(dims=[2]), mask=graph.mask, aux={})
    assert not torch.allclose(enc(graph).tokens, enc(rewired).tokens, atol=1e-4)


@pytest.mark.parametrize(
    ("x", "adj", "message"),
    [
        (torch.zeros(2, 10, FEAT_DIM - 1), torch.zeros(2, EDGE_TYPES, 10, 10), "feature"),
        (torch.zeros(2, 10, FEAT_DIM), torch.zeros(2, EDGE_TYPES - 1, 10, 10), "adjacency"),
    ],
)
def test_shape_contract_fails_closed(x: torch.Tensor, adj: torch.Tensor, message: str) -> None:
    enc = TypedMessagePassingEncoder(
        in_dim=FEAT_DIM, num_relations=EDGE_TYPES, d_model=64, dim=32, layers=2, w_rel=0.0
    )
    graph = ImaginedGraph(x=x, adj=adj, mask=torch.zeros(2, 10), aux={})
    with pytest.raises(ValueError, match=message):
        enc(graph)


# --------------------------------------------------------------------------- substitutability


def test_accepts_a_graph_with_different_feature_and_relation_counts() -> None:
    """The property today's STEncoder cannot have: F != 11, R != 4."""
    torch.manual_seed(0)
    batch, nodes, feature_dim, relations, d_model = 3, 5, 7, 2, 16
    enc = TypedMessagePassingEncoder(
        in_dim=feature_dim, num_relations=relations, d_model=d_model, dim=8, layers=2, w_rel=0.0
    )
    graph = ImaginedGraph(
        x=torch.randn(batch, nodes, feature_dim),
        adj=torch.rand(batch, relations, nodes, nodes),
        mask=torch.ones(batch, nodes),
        aux={},
    )
    out = enc(graph)
    assert out.tokens.shape == (batch, nodes, d_model)
    assert out.pooled.shape == (batch, d_model)


# --------------------------------------------------------------------------- pooled / mask


def test_pooled_ignores_masked_out_nodes() -> None:
    """Perturbing a masked-out node's features must not move `pooled`."""
    torch.manual_seed(0)
    batch, nodes, feature_dim, relations, d_model = 2, 4, 6, 3, 8
    enc = TypedMessagePassingEncoder(
        in_dim=feature_dim, num_relations=relations, d_model=d_model, dim=8, layers=1, w_rel=0.0
    ).eval()
    # zero adjacency isolates masking to the pooling step: with no cross-node
    # messages, each token depends only on its own row of x.
    adj = torch.zeros(batch, relations, nodes, nodes)
    mask = torch.ones(batch, nodes)
    mask[:, -1] = 0.0
    x = torch.randn(batch, nodes, feature_dim)
    graph = ImaginedGraph(x=x, adj=adj, mask=mask, aux={})
    embedding = enc(graph)

    x_perturbed = x.clone()
    x_perturbed[:, -1] = torch.randn(batch, feature_dim)
    graph_perturbed = ImaginedGraph(x=x_perturbed, adj=adj, mask=mask, aux={})
    embedding_perturbed = enc(graph_perturbed)

    assert torch.allclose(embedding.pooled, embedding_perturbed.pooled, atol=1e-6)
    assert not torch.allclose(
        embedding.tokens[:, -1], embedding_perturbed.tokens[:, -1], atol=1e-6
    )


def test_pooled_finite_when_a_mask_row_is_all_zero() -> None:
    torch.manual_seed(0)
    batch, nodes, feature_dim, relations, d_model = 2, 4, 6, 3, 8
    enc = TypedMessagePassingEncoder(
        in_dim=feature_dim, num_relations=relations, d_model=d_model, dim=8, layers=1, w_rel=0.0
    ).eval()
    mask = torch.ones(batch, nodes)
    mask[1] = 0.0
    graph = ImaginedGraph(
        x=torch.randn(batch, nodes, feature_dim),
        adj=torch.rand(batch, relations, nodes, nodes),
        mask=mask,
        aux={},
    )
    pooled = enc(graph).pooled
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled[1], torch.zeros(d_model), atol=1e-6)


# --------------------------------------------------------------------------- aux privacy


def test_never_reads_graph_aux() -> None:
    """`aux` is generator-private; the encoder must not touch it, even to check membership."""
    torch.manual_seed(0)
    batch, nodes, feature_dim, relations, d_model = 2, 4, 6, 3, 8
    enc = TypedMessagePassingEncoder(
        in_dim=feature_dim, num_relations=relations, d_model=d_model, dim=8, layers=1, w_rel=0.0
    ).eval()
    graph = ImaginedGraph(
        x=torch.randn(batch, nodes, feature_dim),
        adj=torch.rand(batch, relations, nodes, nodes),
        mask=torch.ones(batch, nodes),
        aux=_ForbiddenAux(),
    )
    enc(graph)  # must not raise


# --------------------------------------------------------------------------- L_rel ownership


def test_auxiliary_losses_matches_direct_relational_loss_computation() -> None:
    """`GraphEncoder.auxiliary_losses` must compute exactly the `relational_loss` reference."""
    torch.manual_seed(0)
    batch, nodes, feature_dim, relations, d_model = 3, 4, 5, 2, 8
    enc = TypedMessagePassingEncoder(
        in_dim=feature_dim, num_relations=relations, d_model=d_model, dim=8, layers=1, w_rel=0.5
    ).eval()
    graph = ImaginedGraph(
        x=torch.randn(batch, nodes, feature_dim),
        adj=torch.rand(batch, relations, nodes, nodes),
        mask=torch.ones(batch, nodes),
        aux={},
    )
    embedding = enc(graph)
    batch_dict = {
        "rel_target": torch.rand(batch, 2),
        "edge_mask": torch.ones(batch),
        "loss_world_size": torch.tensor(1),
    }

    losses = enc.auxiliary_losses(embedding, batch_dict)

    assert set(losses) == {"rel_loss"}
    expected = relational_loss(
        enc.relational_predictions(embedding),
        batch_dict["rel_target"],
        batch_dict["edge_mask"],
        world_size=1,
    )
    assert torch.allclose(losses["rel_loss"], expected)


def test_auxiliary_losses_zero_when_relational_head_disabled() -> None:
    torch.manual_seed(0)
    enc = TypedMessagePassingEncoder(
        in_dim=5, num_relations=2, d_model=8, dim=8, layers=1, w_rel=0.0
    ).eval()
    graph = ImaginedGraph(
        x=torch.randn(2, 3, 5), adj=torch.rand(2, 2, 3, 3), mask=torch.ones(2, 3), aux={}
    )
    embedding = enc(graph)
    losses = enc.auxiliary_losses(embedding, {})
    assert float(losses["rel_loss"].detach()) == 0.0
    assert enc.rel_head is None
    with pytest.raises(RuntimeError, match="w_rel"):
        enc.relational_predictions(embedding)


def test_graph_encoder_is_abstract() -> None:
    with pytest.raises(TypeError):
        GraphEncoder(d_model=8, w_rel=0.0)  # type: ignore[abstract]


def test_no_l_rel_arm_shares_initialization_with_the_full_arm() -> None:
    """`w_rel=0` must change only `L_rel`, never the initial weights.

    The `no_l_rel` ablation is interpretable only if it differs from `full` by
    the loss term alone. `GraphEncoder.__init__` allocates `rel_head`
    conditionally and runs before a subclass's own layers and before the
    classifier, so an RNG-consuming conditional would give the two arms
    different initial weights everywhere downstream and confound the ablation
    with an initialization difference.
    """
    from src.model.egostitch.composite import EgoStitchModel
    from src.model.egostitch.config import E2EConfig

    shared = {
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 2,
        "n_heads": 2,
        "n_inj": 1,
        "ste_dim": 16,
        "ste_layers": 2,
        "xattn_heads": 2,
    }
    torch.manual_seed(0)
    # `**dict[str, int]` unpacking can't be proven to avoid `E2EConfig`'s
    # other, non-`int`-typed fields (e.g. `feature_standardization: str`),
    # even though none of `shared`'s keys target one; same limitation
    # `tests/model/test_composite.py`'s `_tiny_e2e_config` hits.
    full = EgoStitchModel(E2EConfig(**shared, w_rel=0.25))  # type: ignore[arg-type]
    torch.manual_seed(0)
    no_l_rel = EgoStitchModel(E2EConfig(**shared, w_rel=0.0))  # type: ignore[arg-type]

    assert full.encoder.rel_head is not None
    assert no_l_rel.encoder.rel_head is None

    full_state = full.state_dict()
    shared_names = [
        name
        for name in no_l_rel.state_dict()
        if not name.startswith("encoder.rel_head")
    ]
    assert shared_names, "expected shared parameters to compare"

    def _mismatch_msg(name: str) -> Callable[[str], str]:
        """Bind `name` by closure so each iteration's message is correct.

        A `lambda m, name=name: ...` default-argument capture is the usual
        late-binding fix for this, but its extra defaulted parameter defeats
        mypy's inference against `assert_close`'s `Callable[[str], str]`
        `msg` type; a nested closure has the same one-binding-per-iteration
        behavior with an inferrable signature.
        """
        return lambda m: f"{name} differs between arms: {m}"

    for name in shared_names:
        torch.testing.assert_close(
            no_l_rel.state_dict()[name],
            full_state[name],
            rtol=0,
            atol=0,
            msg=_mismatch_msg(name),
        )
