"""Tests for src.model.egostitch.generator.imagine: the composed Stage-1 model."""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.imagine import (
    NULL_MODE_CONTENT,
    NULL_MODE_FULL,
    EgoStitchStage1,
)
from src.model.egostitch.generator.losses import stage1_total

pytestmark = pytest.mark.unit

# n_ground < slots so the learned base queries participate (gradient coverage).
_TINY = EgoStitchConfig(
    input_dim=8,
    d_p=4,
    d_z=4,
    d_h=8,
    slots=4,
    m_max=8,
    n_ground=3,
    decoder_layers=2,
    n_heads=2,
    gin_hidden=8,
    gin_layers=2,
    sinkhorn_iters=5,
)

# The frozen random GIN and the rev-3.1-only pointer head legitimately receive
# no gradient under the standalone family's frozen Section-13 objective.
# `tokenize.degree_mean_head` (fed `d_hat_raw` into the retired
# `EgoStitchStage1.d_hat()`, itself only consumed by the `pair_outputs`/
# `self_outputs` decision-fusion methods deleted alongside `decision.py`,
# three-component refactor design §9) was removed entirely in the P2a
# dead-code sweep, so there are no more parameters under that prefix to
# exempt.
_NO_GRAD_PREFIXES = ("random_gin", "imagine.head_pointer")


def _model(seed: int = 0) -> EgoStitchStage1:
    torch.manual_seed(seed)
    return EgoStitchStage1(_TINY)


def _node_batch(batch: int = 4, *, seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    t = _TINY.slots
    target_in_pool = torch.rand(batch, t, generator=gen) > 0.5
    return {
        "x": torch.randn(batch, _TINY.input_dim, generator=gen),
        "ground_x": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "target_features": torch.randn(batch, t, _TINY.input_dim, generator=gen),
        "target_mult": 1.0 + torch.rand(batch, t, generator=gen),
        "target_adj": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
        "target_mask": torch.ones(batch, t, dtype=torch.bool),
        "target_in_pool": target_in_pool,
        "target_pool_index": torch.where(
            target_in_pool,
            torch.arange(t).remainder(_TINY.n_ground).expand(batch, -1),
            torch.full((batch, t), -1),
        ),
        "true_degree": torch.randint(1, 6, (batch,), generator=gen).float(),
        "real_ego_stats": torch.rand(batch, 4, generator=gen),
    }


class TestFeatureNormalization:
    """Feature-normalization surface of `EgoStitchStage1`.

    Its own `(s0, s1, s2)` forward contract (`pair_outputs`/`self_outputs`/
    `forward`) belonged to the retired frozen-s0 `egostitch` scorer and was
    removed with `decision.py` (three-component refactor design §9); only
    the feature-normalization surface below survives.
    """

    def test_legacy_generator_keeps_raw_feature_semantics(self) -> None:
        model = _model()
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)

        torch.testing.assert_close(model.normalize_features(x), x)
        torch.testing.assert_close(model.project_features(x), model.proj(x))

    def test_row_layernorm_mode_keeps_the_rev31_per_row_transform(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)

        normalized = model.normalize_features(x)

        torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(3), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            normalized.var(dim=-1, unbiased=False), torch.ones(3), atol=1e-5, rtol=0.0
        )
        assert tuple(model.feature_norm.parameters()) == ()
        torch.testing.assert_close(model.project_features(x), model.proj(normalized))


class TestNodeLosses:
    def test_all_families_finite(self) -> None:
        model = _model()
        node = _node_batch()
        losses, enc = model.node_losses(
            node["x"],
            node["ground_x"],
            target_features=node["target_features"],
            target_mult=node["target_mult"],
            target_adj=node["target_adj"],
            target_mask=node["target_mask"],
            target_in_pool=node["target_in_pool"],
            target_pool_index=node["target_pool_index"],
            true_degree=node["true_degree"],
            real_ego_stats=node["real_ego_stats"],
        )
        for term_t in (losses.deg, losses.real_egostat, losses.real_gin):
            assert bool(torch.isfinite(term_t))
        for term in losses.recon.values():
            assert bool(torch.isfinite(term))
        assert enc.slots.h.shape == (4, _TINY.slots, _TINY.d_p)

    def test_denoise_requires_mask_and_noise(self) -> None:
        model = _model()
        node = _node_batch()
        with pytest.raises(ValueError, match="denoise"):
            model.node_losses(
                node["x"],
                node["ground_x"],
                target_features=node["target_features"],
                target_mult=node["target_mult"],
                target_adj=node["target_adj"],
                target_mask=node["target_mask"],
                target_in_pool=node["target_in_pool"],
                target_pool_index=node["target_pool_index"],
                true_degree=node["true_degree"],
                real_ego_stats=node["real_ego_stats"],
                denoise_features=node["target_features"][:, :2],
            )


class TestGradientFlow:
    def test_every_trainable_parameter_receives_gradient(self) -> None:
        model = _model()
        node = _node_batch()
        batch = 4
        gen = torch.Generator().manual_seed(3)
        null_mode = torch.tensor([NULL_MODE_FULL, NULL_MODE_CONTENT, 2, NULL_MODE_FULL])
        denoise_features = node["target_features"][:, :2]
        denoise_mask = torch.ones(batch, 2, dtype=torch.bool)
        denoise_noise = 0.1 * torch.randn(batch, 2, _TINY.d_p, generator=gen)

        losses, _ = model.node_losses(
            node["x"],
            node["ground_x"],
            target_features=node["target_features"],
            target_mult=node["target_mult"],
            target_adj=node["target_adj"],
            target_mask=node["target_mask"],
            target_in_pool=node["target_in_pool"],
            target_pool_index=node["target_pool_index"],
            true_degree=node["true_degree"],
            real_ego_stats=node["real_ego_stats"],
            null_mode=null_mode,
            denoise_features=denoise_features,
            denoise_mask=denoise_mask,
            denoise_noise=denoise_noise,
        )
        ssl = model.ssl_losses(
            node["x"],
            node["ground_x"],
            torch.randn_like(node["ground_x"]),
            noise=_TINY.ssl_noise_sigma * torch.randn_like(node["x"]),
        )
        # `EgoStitchStage1` no longer scores an edge stream itself (its own
        # `forward`/decision fusion was removed with `decision.py`, design
        # §9); `L_edge` is now entirely the pair classifier's responsibility
        # (`EgoStitchE2E`/`train_egostitch.py`). A constant, non-grad zero
        # stands in for `stage1_total`'s required `edge` argument so this
        # test still exercises gradient flow through every *other* family.
        total, _ = stage1_total(
            _TINY,
            edge=torch.zeros(()),
            recon=losses.recon,
            deg=losses.deg,
            real_egostat=losses.real_egostat,
            real_gin=losses.real_gin,
            ssl_noise=ssl["noise"],
            ssl_pool=ssl["pool"],
        )
        total.backward()  # type: ignore[no-untyped-call]

        missing: list[str] = []
        for name, param in model.named_parameters():
            if not param.requires_grad or name.startswith(_NO_GRAD_PREFIXES):
                continue
            if param.grad is None or not bool(torch.isfinite(param.grad).all()):
                missing.append(name)
        assert missing == []
