"""Contracts for `PairLatentGenerator` inside `V3_1` (the D9 kd_d9 arm).

Covers: baseline containment (absent config and the zero-init alpha gate),
eval determinism vs training stochasticity, the KD-stream output keys, the
stage-1/stage-2 gradient routing, checkpoint roundtrip, and conditioning
symmetry.
"""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.classifier.b0_v31 import V3_1

pytestmark = pytest.mark.unit

_Z_DIM = 8
_SEEDS = 3
_SEED_DIM = 12
_MC = 2


def _config(*, with_gen: bool = True, mc_samples: int = _MC) -> dict[str, object]:
    config: dict[str, object] = {
        "input_dim": 16,
        "d_model": 32,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 4,
        "mlp_head": {"hidden_dims": [16], "dropout": 0.0},
        "regularization": {"dropout": 0.0},
    }
    if with_gen:
        config["pair_latent_gen"] = {
            "z_dim": _Z_DIM,
            "cond_dim": 16,
            "hidden": 32,
            "seed_count": _SEEDS,
            "seed_dim": _SEED_DIM,
            "mc_samples": mc_samples,
            "kl_free_bits": 0.05,
        }
    return config


def _batch(batch_size: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "emb_a": torch.randn(batch_size, 5, 16, generator=generator),
        "emb_b": torch.randn(batch_size, 7, 16, generator=generator),
        "len_a": torch.tensor([5, 3, 4, 5]),
        "len_b": torch.tensor([7, 3, 7, 5]),
    }


def _teacher_seeds(batch_size: int = 4, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, _SEEDS, _SEED_DIM, generator=generator)


def test_absent_config_builds_no_generator_and_emits_no_gen_keys() -> None:
    model = V3_1(**_config(with_gen=False))
    model.eval()
    output = model(_batch())
    assert model.pair_latent_gen is None
    assert not any(key.startswith("gen_") for key in output)


def test_zero_init_alpha_keeps_logits_identical_to_the_bare_trunk() -> None:
    torch.manual_seed(0)
    with_gen = V3_1(**_config(with_gen=True))
    torch.manual_seed(0)
    without = V3_1(**_config(with_gen=False))
    with_gen.eval()
    without.eval()

    batch = _batch()
    logits_gen = with_gen(batch)["logits"]
    logits_bare = without(batch)["logits"]
    torch.testing.assert_close(logits_gen, logits_bare, atol=0.0, rtol=0.0)
    assert with_gen.pair_latent_gen is not None
    assert float(with_gen.pair_latent_gen.alpha.detach()) == 0.0


def test_eval_is_deterministic_and_training_samples_the_prior() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    assert model.pair_latent_gen is not None
    with torch.no_grad():
        model.pair_latent_gen.alpha.fill_(1.0)

    batch = _batch()
    model.eval()
    with torch.no_grad():
        eval_first = model(batch)["logits"]
        eval_second = model(batch)["logits"]
    torch.testing.assert_close(eval_first, eval_second, atol=0.0, rtol=0.0)

    model.train()
    with torch.no_grad():
        train_first = model(batch)["logits"]
        train_second = model(batch)["logits"]
    assert not torch.allclose(train_first, train_second)


def test_kd_stream_forward_returns_posterior_seeds_kl_and_prior_mean_decode() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    model.train()
    output = model({**_batch(), "kd_teacher_seeds": _teacher_seeds()})

    assert output["gen_seeds_q"].shape == (4, _SEEDS, _SEED_DIM)
    assert output["gen_seeds_prior_mean"].shape == (4, _SEEDS, _SEED_DIM)
    assert output["gen_kl"].shape == (4, _Z_DIM)
    assert output["gen_prior_dispersion"].shape == (4,)
    assert torch.isfinite(output["gen_seeds_q"]).all()
    assert torch.isfinite(output["gen_kl"]).all()
    assert (output["gen_kl"] >= 0.0).all()
    assert (output["gen_prior_dispersion"] >= 0.0).all()
    assert output["gen_delta_std"].shape == (4,)


def test_single_mc_sample_reports_zero_prior_dispersion() -> None:
    model = V3_1(**_config(with_gen=True, mc_samples=1))
    output = model({**_batch(), "kd_teacher_seeds": _teacher_seeds()})
    torch.testing.assert_close(output["gen_prior_dispersion"], torch.zeros(4), atol=0.0, rtol=0.0)


def _grad_is_live(parameters: list[torch.nn.Parameter]) -> bool:
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters)


def test_stage1_blocks_task_gradient_into_the_generator_and_stage2_opens_it() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    gen = model.pair_latent_gen
    assert gen is not None
    with torch.no_grad():
        gen.alpha.fill_(1.0)
    model.train()

    assert gen.joint_stage is False
    model(_batch())["logits"].sum().backward()
    decoder_params = list(gen.decoder.parameters())
    assert not _grad_is_live(decoder_params)
    assert gen.alpha.grad is not None
    assert _grad_is_live(list(gen.delta_head.parameters()))
    assert _grad_is_live(list(model.encoder.parameters()))

    model.zero_grad()
    gen.joint_stage = True
    model(_batch())["logits"].sum().backward()
    assert _grad_is_live(decoder_params)


def test_kd_losses_reach_the_generator_but_never_the_trunk() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    gen = model.pair_latent_gen
    assert gen is not None
    model.train()

    output = model({**_batch(), "kd_teacher_seeds": _teacher_seeds()})
    (output["gen_seeds_q"].sum() + output["gen_kl"].sum()).backward()

    assert _grad_is_live(list(gen.decoder.parameters()))
    assert _grad_is_live(list(gen.recognition.parameters()))
    assert _grad_is_live(list(gen.prior.parameters()))
    assert _grad_is_live(list(gen.condition.parameters()))
    assert not _grad_is_live(list(model.encoder.parameters()))
    assert not _grad_is_live(list(model.cross_attention.parameters()))


def test_checkpoint_roundtrip_restores_the_generator_and_eval_logits() -> None:
    torch.manual_seed(0)
    source = V3_1(**_config(with_gen=True))
    assert source.pair_latent_gen is not None
    with torch.no_grad():
        source.pair_latent_gen.alpha.fill_(0.7)
    source.eval()
    batch = _batch()
    with torch.no_grad():
        expected = source(batch)["logits"]

    torch.manual_seed(99)  # rebuild under a different RNG stream on purpose
    rebuilt = V3_1(**_config(with_gen=True))
    rebuilt.load_state_dict(source.state_dict(), strict=True)
    rebuilt.eval()
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(batch)["logits"], expected, atol=0.0, rtol=0.0)


def test_generated_latent_is_symmetric_under_endpoint_swap() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    model.eval()
    batch = _batch()
    swapped = {
        "emb_a": batch["emb_b"],
        "emb_b": batch["emb_a"],
        "len_a": batch["len_b"],
        "len_b": batch["len_a"],
    }
    seeds = _teacher_seeds()
    with torch.no_grad():
        forward = model({**batch, "kd_teacher_seeds": seeds})
        backward = model({**swapped, "kd_teacher_seeds": seeds})
    torch.testing.assert_close(
        forward["gen_seeds_prior_mean"], backward["gen_seeds_prior_mean"], atol=1e-6, rtol=1e-5
    )


def test_prior_dispersion_uses_fixed_samples_in_training_mode() -> None:
    model = V3_1(**_config(with_gen=True))
    batch = {**_batch(), "kd_teacher_seeds": _teacher_seeds()}
    first = model(batch)["gen_prior_dispersion"]
    second = model(batch)["gen_prior_dispersion"]
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_generator_parameters_helper_excludes_fusion_branch() -> None:
    torch.manual_seed(0)
    model = V3_1(**_config(with_gen=True))
    assert model.pair_latent_gen is not None
    helper_ids = {id(p) for p in model.pair_latent_gen_parameters()}
    generative_ids = {
        id(p)
        for component in (
            model.pair_latent_gen.condition,
            model.pair_latent_gen.prior,
            model.pair_latent_gen.recognition,
            model.pair_latent_gen.decoder,
        )
        for p in component.parameters()
    }
    fusion_ids = {
        id(model.pair_latent_gen.alpha),
        *(id(p) for p in model.pair_latent_gen.adapter.parameters()),
        *(id(p) for p in model.pair_latent_gen.delta_head.parameters()),
    }
    assert helper_ids == generative_ids
    assert helper_ids.isdisjoint(fusion_ids)
    assert V3_1(**_config(with_gen=False)).pair_latent_gen_parameters() == []
