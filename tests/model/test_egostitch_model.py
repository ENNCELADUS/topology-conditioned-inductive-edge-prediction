"""Tests for src.model.egostitch.model: the composed Stage-1 model."""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.imagine import NULL_MODE_CONTENT, NULL_MODE_FULL
from src.model.egostitch.losses import stage1_total
from src.model.egostitch.model import EgoStitchStage1

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

# Heads with no Stage-1 loss consumer (pointer grounds slot identity for the
# Stage-3 scaffold) and the frozen random GIN legitimately receive no gradient.
_NO_GRAD_PREFIXES = ("imagine.head_pointer", "random_gin")


def _model(seed: int = 0) -> EgoStitchStage1:
    torch.manual_seed(seed)
    return EgoStitchStage1(_TINY)


def _pair_batch(batch: int = 3, *, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "x_i": torch.randn(batch, _TINY.input_dim, generator=gen),
        "x_j": torch.randn(batch, _TINY.input_dim, generator=gen),
        "ground_i": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "ground_j": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "s0": torch.randn(batch, generator=gen),
        "label": (torch.rand(batch, generator=gen) > 0.5).long(),
    }


def _node_batch(batch: int = 4, *, seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    t = _TINY.slots
    return {
        "x": torch.randn(batch, _TINY.input_dim, generator=gen),
        "ground_x": torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen),
        "target_features": torch.randn(batch, t, _TINY.input_dim, generator=gen),
        "target_mult": 1.0 + torch.rand(batch, t, generator=gen),
        "target_adj": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
        "target_mask": torch.ones(batch, t, dtype=torch.bool),
        "target_in_pool": torch.rand(batch, t, generator=gen) > 0.5,
        "true_degree": torch.randint(1, 6, (batch,), generator=gen).float(),
        "real_ego_stats": torch.rand(batch, 4, generator=gen),
    }


class TestForwardContract:
    def test_pair_batch_logits_and_loss(self) -> None:
        model = _model()
        out = model(_pair_batch())
        assert out["logits"].shape == (3,)
        assert bool(torch.isfinite(out["logits"]).all())
        assert "loss" in out
        assert float(out["loss"].detach()) > 0.0

    def test_self_batch_routes_single_ego_path(self) -> None:
        model = _model()
        batch = _pair_batch()
        self_batch = {"x_i": batch["x_i"], "ground_i": batch["ground_i"], "s0": batch["s0"]}
        out = model(self_batch)
        assert out["logits"].shape == (3,)
        assert "loss" not in out

    def test_deterministic_in_eval(self) -> None:
        model = _model()
        model.eval()
        batch = _pair_batch()
        with torch.no_grad():
            first = model(batch)["logits"]
            second = model(batch)["logits"]
        torch.testing.assert_close(first, second)

    def test_density_ratio_scales_budget(self) -> None:
        model = _model()
        x = torch.randn(2, _TINY.input_dim)
        tok = model.tokenize(x)
        base = model.d_hat(tok)
        model.set_density_ratio(11.6)
        torch.testing.assert_close(model.d_hat(tok), base * 11.6)


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
                true_degree=node["true_degree"],
                real_ego_stats=node["real_ego_stats"],
                denoise_features=node["target_features"][:, :2],
            )


class TestGradientFlow:
    def test_pair_score_backpropagates_through_projection(self) -> None:
        model = _model()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("imagine.proj."))
        batch = _pair_batch()
        with torch.no_grad():
            enc_i = model.encode_nodes(batch["x_i"], batch["ground_i"])
            enc_j = model.encode_nodes(batch["x_j"], batch["ground_j"])
        model.pair_logits(enc_i, enc_j, batch["x_i"], batch["x_j"], batch["s0"]).sum().backward()  # type: ignore[no-untyped-call]
        grad = model.proj.weight.grad
        assert grad is not None
        assert bool(torch.isfinite(grad).all())
        assert bool(torch.count_nonzero(grad))

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
        edge = model(_pair_batch())["loss"]
        total, _ = stage1_total(
            _TINY,
            edge=edge,
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

    def test_s0_receives_no_gradient(self) -> None:
        model = _model()
        batch = _pair_batch()
        batch["s0"] = batch["s0"].clone().requires_grad_(True)
        out = model(batch)
        out["loss"].backward()
        # s0 is a graph leaf here, so it *would* get a pass-through gradient of
        # d loss/d logit — the cache design (spec Sec 13.10) never exposes a
        # trainable s0; this documents that the head adds, not transforms, s0.
        assert batch["s0"].grad is not None
        assert bool(torch.isfinite(batch["s0"].grad).all())
