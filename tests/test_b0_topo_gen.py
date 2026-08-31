"""kd_gen TopoLatentGenerator unit tests.

Spec: docs/superpowers/specs/2026-08-30-kd-gen-arm-design.md.
"""

import math
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from src.model.egostitch.classifier.b0_v31 import BEST_V3_1_CONFIG, V3_1
from src.model.egostitch.classifier.topo_gen import (
    ImfTopoGen,
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
    assert _module("imf").family == "imf"
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
    marginal_logit(logits).sum().backward()  # type: ignore[no-untyped-call]
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
    out["logits"].sum().backward()  # type: ignore[no-untyped-call]
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


def test_imf_one_nfe_sampling_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = cast(ImfTopoGen, _module("imf")).eval()
    enc_a, enc_b, len_a, len_b, _ = _inputs()
    cond = module.condition(enc_a, enc_b, len_a, len_b)
    original_u = module._u
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def counted_u(
        x: torch.Tensor, r: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        calls.append((r.detach().clone(), t.detach().clone()))
        return original_u(x, r, t, condition)

    monkeypatch.setattr(module, "_u", counted_u)
    first = module.sample_latents(cond)
    assert first.shape == (4, CFG["mc_samples"], CFG["latent_dim"])
    assert len(calls) == 1
    assert torch.count_nonzero(calls[0][0]) == 0
    assert torch.all(calls[0][1] == 1)
    second = module.sample_latents(cond)
    assert len(calls) == 2
    torch.testing.assert_close(first, second)


def test_imf_u_uses_ordered_time_and_interval_fourier_tail() -> None:
    module = cast(ImfTopoGen, _module("imf"))
    x = torch.zeros(2, cast(int, CFG["latent_dim"]))
    cond = torch.zeros(2, cast(int, CFG["cond_dim"]))
    r = torch.tensor([0.1, 0.6])
    t = torch.tensor([0.8, 0.9])
    captured: list[torch.Tensor] = []
    hook = module.cond_embed.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    _ = module._u(x, r, t, cond)
    hook.remove()

    def fourier(values: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            torch.arange(half, dtype=values.dtype) * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = values.unsqueeze(-1) * frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    t_embed = fourier(t, 32)
    interval_embed = fourier(t - r, 32)
    time_tail = captured[0][:, -64:]
    expected = torch.cat((t_embed, interval_embed), dim=-1)
    assert torch.equal(time_tail, expected)
    averaged = 0.5 * (t_embed + interval_embed)
    assert not torch.allclose(time_tail, torch.cat((averaged, averaged), dim=-1))


def test_imf_gen_loss_pins_sampling_jvp_target_and_adaptive_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(ImfTopoGen, _module("imf")).train()
    batch = 3
    latent_dim = cast(int, CFG["latent_dim"])
    cond = torch.arange(batch * cast(int, CFG["cond_dim"]), dtype=torch.float32).reshape(batch, -1)
    target = torch.arange(batch * latent_dim, dtype=torch.float32).reshape(batch, -1) / 10.0
    eps = torch.flip(target, dims=(1,)) + 0.5
    pair_draws = torch.tensor([[0.2, 0.8], [0.7, 0.1], [0.4, 0.6]])
    branch_draws = torch.tensor([0.1, 0.25, 0.9])
    rand_calls: list[tuple[int, ...]] = []

    def fake_randn_like(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == target.shape
        return eps.clone()

    def fake_rand(*size: int, **kwargs: object) -> torch.Tensor:
        assert kwargs == {"device": target.device}
        rand_calls.append(size)
        if size == (batch, 2):
            return pair_draws.clone()
        assert size == (batch,)
        return branch_draws.clone()

    u = torch.linspace(-0.3, 0.6, batch * latent_dim).reshape(batch, -1).requires_grad_()
    dudt = torch.linspace(0.2, 0.8, batch * latent_dim).reshape(batch, -1).requires_grad_()
    jvp_call: dict[str, object] = {}

    def fake_jvp(
        function: object,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert callable(function)
        jvp_call.update(inputs=inputs, tangents=tangents, create_graph=create_graph)
        return u, dudt

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch.autograd.functional, "jvp", fake_jvp)

    loss, stats = module.gen_loss(cond, target)

    assert rand_calls == [(batch, 2), (batch,)]
    expected_t = torch.tensor([0.8, 0.7, 0.6])
    expected_r = torch.tensor([0.8, 0.1, 0.4])
    expected_v = eps - target
    expected_x_t = (1.0 - expected_t).unsqueeze(-1) * target + expected_t.unsqueeze(-1) * eps
    x_t, r, t = cast(tuple[torch.Tensor, ...], jvp_call["inputs"])
    tangent_x, tangent_r, tangent_t = cast(tuple[torch.Tensor, ...], jvp_call["tangents"])
    assert torch.equal(t, expected_t)
    assert torch.equal(r, expected_r)
    assert torch.equal(x_t, expected_x_t)
    assert torch.equal(tangent_x, expected_v)
    assert torch.equal(tangent_r, torch.zeros_like(expected_r))
    assert torch.equal(tangent_t, torch.ones_like(expected_t))
    assert jvp_call["create_graph"] is True

    expected_target = expected_v - (expected_t - expected_r).unsqueeze(-1) * dudt.detach()
    expected_sq = (u - expected_target).square().mean(dim=-1)
    expected_weight = (expected_sq.detach() + 1e-3).pow(-0.5)
    expected_loss = (expected_weight * expected_sq).mean()
    assert stats == {}
    assert torch.equal(loss, expected_loss)
    assert not torch.equal(loss, expected_sq.mean())
    loss.backward()  # type: ignore[no-untyped-call]
    assert u.grad is not None and torch.count_nonzero(u.grad) > 0
    assert dudt.requires_grad and dudt.grad is None


def test_imf_fp32_autocast_loss_and_jvp_gradients() -> None:
    module = _module("imf").train()
    enc_a, enc_b, len_a, len_b, _ = _inputs()
    target = torch.randn(4, cast(int, CFG["latent_dim"]))
    torch.manual_seed(7)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cond = module.condition(enc_a, enc_b, len_a, len_b)
        latents = module.sample_latents(cond)
        loss, stats = module.gen_loss(cond, target)
    assert cond.dtype == torch.float32
    assert latents.dtype == torch.float32 and torch.isfinite(latents).all()
    assert loss.dtype == torch.float32 and torch.isfinite(loss) and loss.requires_grad
    assert stats == {}
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [parameter.grad for parameter in module.generator_parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    core_gradients = [parameter.grad for parameter in module.core.parameters()]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in core_gradients
    )


def test_imf_config_is_exact_edm_delta() -> None:
    edm = yaml.safe_load(Path("configs/b1_kd_gen_edm_breadth_first.yaml").read_text())
    imf = yaml.safe_load(Path("configs/b1_kd_gen_imf_breadth_first.yaml").read_text())
    expected = deepcopy(edm)
    expected["model"]["config"]["topo_gen"]["name"] = "imf"
    del expected["model"]["config"]["topo_gen"]["sampler_steps"]
    expected["output_dir"] = "outputs/b1_row_kd/kd_gen_imf"
    assert imf == expected
    topo_cfg = imf["model"]["config"]["topo_gen"]
    assert topo_cfg["name"] == "imf"
    assert "sampler_steps" not in topo_cfg
    assert build_topo_gen(topo_cfg, int(imf["model"]["config"]["d_model"])).family == "imf"


@pytest.mark.parametrize("family", ["edm", "det_mse"])
def test_gen_loss_finite_and_decreases(family: str) -> None:
    module = _module(family).train()
    enc_a, enc_b, len_a, len_b, _ = _inputs(8)
    target = torch.randn(8, cast(int, CFG["latent_dim"]))
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
        loss.backward()  # type: ignore[no-untyped-call]
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
    modulation = cast(torch.nn.Linear, module.core.blocks[0].get_submodule("mod"))
    torch.nn.init.normal_(modulation.weight, std=0.1)
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


def _v31(topo: bool) -> V3_1:
    torch.manual_seed(0)
    config = {**BEST_V3_1_CONFIG, "input_dim": 24, "d_model": 16, "n_heads": 2}
    if topo:
        config["topo_gen"] = {
            "name": "edm",
            "latent_dim": 8,
            "cond_dim": 8,
            "blocks": 1,
            "adapter_dim": 4,
            "mc_samples": 2,
            "sampler_steps": 2,
        }
    return V3_1(**config)


def _v31_batch(batch: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    return {
        "emb_a": torch.randn(batch, 4, 24),
        "emb_b": torch.randn(batch, 4, 24),
        "len_a": torch.full((batch,), 4, dtype=torch.long),
        "len_b": torch.full((batch,), 4, dtype=torch.long),
    }


def test_v31_rejects_pair_latent_gen_key() -> None:
    with pytest.raises(ValueError):
        _ = V3_1(
            **{**BEST_V3_1_CONFIG, "input_dim": 24, "d_model": 16, "n_heads": 2},
            pair_latent_gen={"z_dim": 4},
        )


def test_v31_topo_gen_eval_branch_zero_equals_control() -> None:
    fused = _v31(topo=True).eval()
    control = _v31(topo=False).eval()
    # Same seed + build-last ordering => identical trunk weights.
    batch = _v31_batch()
    with torch.no_grad():
        fused_out = fused(dict(batch))
        control_out = control(dict(batch))
    torch.testing.assert_close(
        fused_out["logits"].float(), control_out["logits"].float(), atol=1e-5, rtol=1e-5
    )
    assert "gen_prob_std" in fused_out and "gen_latent_sample" in fused_out


def test_v31_topo_gen_kd_outputs_present_with_teacher_latent() -> None:
    model = _v31(topo=True).train()
    batch = _v31_batch()
    batch["kd_teacher_latent"] = torch.randn(3, 8)
    out = model(batch)
    assert out["gen_loss"].requires_grad
    assert out["gen_sigma_bin_loss"].shape == (4,)


def test_v31_topo_gen_parameters_nonempty() -> None:
    assert _v31(topo=True).topo_gen_parameters()
    assert _v31(topo=False).topo_gen_parameters() == []
