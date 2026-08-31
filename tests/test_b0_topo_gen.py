"""kd_gen TopoLatentGenerator unit tests.

Spec: docs/superpowers/specs/2026-08-30-kd-gen-arm-design.md.
"""

import pytest
import torch
from src.model.egostitch.classifier.topo_gen import (
    TopoGenBase,
    build_topo_gen,
    marginal_logit,
)

CFG = {
    "name": "edm",
    "latent_dim": 16,
    "cond_dim": 8,
    "blocks": 2,
    "adapter_dim": 8,
    "mc_samples": 3,
    "sampler_steps": 2,
}
D_MODEL = 12


def _module(name: str = "edm") -> TopoGenBase:
    torch.manual_seed(0)
    return build_topo_gen({**CFG, "name": name}, D_MODEL)


def _inputs(batch: int = 4) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    enc_a = torch.randn(batch, 5, D_MODEL)
    enc_b = torch.randn(batch, 6, D_MODEL)
    len_a = torch.full((batch,), 5, dtype=torch.long)
    len_b = torch.full((batch,), 6, dtype=torch.long)
    pair_repr = torch.randn(batch, D_MODEL)
    return enc_a, enc_b, len_a, len_b, pair_repr


def test_build_rejects_unknown_family_and_keys() -> None:
    with pytest.raises(ValueError, match="topo_gen.name"):
        build_topo_gen({**CFG, "name": "cvae"}, D_MODEL)
    with pytest.raises(ValueError, match="unknown topo_gen keys"):
        build_topo_gen({**CFG, "z_dim": 4}, D_MODEL)
    assert _module("edm").family == "edm"
    assert _module("det_mse").family == "det_mse"


def test_marginal_logit_identities() -> None:
    logits = torch.randn(6, 1, dtype=torch.float64).expand(6, 5)
    # All samples equal => marginal equals the per-sample logit exactly.
    assert torch.equal(marginal_logit(logits), logits[:, 0])
    # M=1 reduces to the single logit.
    single = torch.randn(6, 1, dtype=torch.float64)
    assert torch.equal(marginal_logit(single), single[:, 0])
    # Matches naive fp64 computation on random logits.
    mixed = torch.randn(6, 5, dtype=torch.float64) * 4
    naive = torch.logit(torch.sigmoid(mixed).mean(dim=1))
    torch.testing.assert_close(marginal_logit(mixed), naive, atol=1e-10, rtol=1e-10)


def test_marginal_logit_equal_samples_preserves_gradient() -> None:
    logits = torch.full((2, 3), 0.3, dtype=torch.float64, requires_grad=True)
    marginal_logit(logits).sum().backward()
    torch.testing.assert_close(logits.grad, torch.full_like(logits, 1.0 / 3.0))


def test_public_condition_and_sampling_are_fp32_under_autocast() -> None:
    module = _module().eval()
    enc_a, enc_b, len_a, len_b, _ = _inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cond = module.condition(enc_a, enc_b, len_a, len_b)
        latents = module.sample_latents(cond)
    assert cond.dtype == torch.float32
    assert latents.dtype == torch.float32


def test_adapter_saddle_guard() -> None:
    module = _module()
    assert float(module.gate.detach()) == pytest.approx(1.0)  # g init 1.0
    up = module.adapter_up  # the zero-init projection back to d_model
    assert torch.all(up.weight.detach() == 0) and torch.all(up.bias.detach() == 0)


def test_branch_zero_identity_and_gradients_nonzero_at_init() -> None:
    module = _module().eval()
    head = torch.nn.Linear(D_MODEL, 1)
    enc_a, enc_b, len_a, len_b, pair_repr = _inputs()
    out = module.marginal_forward(enc_a, enc_b, len_a, len_b, pair_repr, head)
    trunk = head(pair_repr.float())
    # W_up = 0 at init => functionally identical to the trunk-only model.
    assert torch.equal(out["logits"], trunk)
    # Saddle guard: task gradient into W_up is nonzero at init (g=1, delta pre-act live).
    module.train()
    out = module.marginal_forward(enc_a, enc_b, len_a, len_b, pair_repr, head)
    out["logits"].sum().backward()
    grad = module.adapter_up.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_eval_sampling_is_deterministic_and_train_is_not() -> None:
    module = _module().eval()
    enc_a, enc_b, len_a, len_b, _ = _inputs()
    cond = module.condition(enc_a, enc_b, len_a, len_b)
    first = module.sample_latents(cond)
    second = module.sample_latents(cond)
    assert first.shape == (4, CFG["mc_samples"], CFG["latent_dim"])
    torch.testing.assert_close(first, second)
    module.train()
    torch.manual_seed(2)
    third = module.sample_latents(cond)
    torch.manual_seed(3)
    fourth = module.sample_latents(cond)
    assert not torch.allclose(third, fourth)


@pytest.mark.parametrize("family", ["edm", "det_mse"])
def test_gen_loss_finite_and_decreases(family: str) -> None:
    module = _module(family).train()
    enc_a, enc_b, len_a, len_b, _ = _inputs(8)
    target = torch.randn(8, CFG["latent_dim"])
    opt = torch.optim.Adam(module.generator_parameters(), lr=1e-2)
    torch.manual_seed(4)
    first_loss, stats = module.gen_loss(module.condition(enc_a, enc_b, len_a, len_b), target)
    assert torch.isfinite(first_loss)
    if family == "edm":
        assert stats["sigma_bin_loss"].shape == (4,)
        assert float(stats["sigma_bin_count"].sum()) == 8.0
    losses = []
    for step in range(60):
        torch.manual_seed(100 + step)
        loss, _ = module.gen_loss(module.condition(enc_a, enc_b, len_a, len_b), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert sum(losses[-10:]) < sum(losses[:10])


def test_det_mse_forces_single_sample() -> None:
    module = _module("det_mse")
    assert module.mc_samples == 1


def test_controls_branch_zero_and_shuffle() -> None:
    module = _module().eval()
    head = torch.nn.Linear(D_MODEL, 1)
    assert not hasattr(module.core, "cond_in")
    torch.nn.init.normal_(module.core.blocks[0].mod.weight, std=0.1)
    # Give the branch real weight so controls have something to remove.
    torch.nn.init.normal_(module.adapter_up.weight, std=0.5)
    enc_a, enc_b, len_a, len_b, pair_repr = _inputs()
    live = module.marginal_forward(enc_a, enc_b, len_a, len_b, pair_repr, head)["logits"]
    module.control = "branch_zero"
    zeroed = module.marginal_forward(enc_a, enc_b, len_a, len_b, pair_repr, head)["logits"]
    assert torch.equal(zeroed, head(pair_repr.float()))
    module.control = "shuffle"
    shuffled = module.marginal_forward(enc_a, enc_b, len_a, len_b, pair_repr, head)["logits"]
    assert not torch.allclose(shuffled, live)
    module.control = None


def test_generator_parameters_exclude_fusion() -> None:
    module = _module()
    gen_ids = {id(p) for p in module.generator_parameters()}
    assert id(module.gate) not in gen_ids
    assert id(module.adapter_up.weight) not in gen_ids
    assert gen_ids  # condition + core are present
