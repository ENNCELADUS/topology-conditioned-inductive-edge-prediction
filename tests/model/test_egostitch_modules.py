"""Tests for the EgoStitch Stage-1 modules: config, tokenize, imagine, stitch."""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.config import E2EConfig, EgoStitchConfig, GeneratorConfig
from src.model.egostitch.generator.assemble import sinkhorn_plan, stitch_cost
from src.model.egostitch.generator.imagine import (
    NULL_MODE_ALL,
    NULL_MODE_CONTENT,
    NULL_MODE_FULL,
    ImagineDecoder,
    SlotSet,
    TokenizeLite,
)

pytestmark = pytest.mark.unit

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
)


def _tiny_inputs(batch: int = 3, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, _TINY.input_dim, generator=gen)
    ground = torch.randn(batch, _TINY.n_ground, _TINY.input_dim, generator=gen)
    return x, ground


class TestConfig:
    def test_defaults_are_spec_values(self) -> None:
        config = EgoStitchConfig()
        assert config.input_dim == 1536
        assert config.d_p == 256
        assert config.slots == 16
        assert config.m_max == 32
        assert config.n_ground == 20
        assert config.sinkhorn_eps == 0.1
        assert config.sinkhorn_iters == 20
        assert config.lambda_recon == 1.0
        assert config.lambda_real == 0.5
        assert config.lambda_ssl == 0.1
        assert config.tau_adj == 0.5
        assert config.tau_div == 0.5
        assert config.l_gate_pos_weight == 6.17
        assert config.w_egostat == pytest.approx(2.0 / 3.0)
        assert config.w_gin == pytest.approx(1.0 / 3.0)

    def test_from_mapping_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown EgoStitch config keys"):
            EgoStitchConfig.from_mapping({"codebook_size": 256})

    def test_from_mapping_applies_overrides(self) -> None:
        config = EgoStitchConfig.from_mapping({"slots": 8, "sinkhorn_eps": 0.2})
        assert config.slots == 8
        assert config.sinkhorn_eps == 0.2

    def test_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            EgoStitchConfig(slots=0)
        with pytest.raises(ValueError, match="divisible"):
            EgoStitchConfig(d_h=10, n_heads=4)
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            EgoStitchConfig(null_dropout=1.5)
        with pytest.raises(ValueError, match="must be an int"):
            EgoStitchConfig.from_mapping({"slots": 8.5})

    @pytest.mark.parametrize("tau_adj", [1.0, 1.5])
    def test_tau_adj_must_be_strictly_less_than_one(self, tau_adj: float) -> None:
        with pytest.raises(ValueError, match="tau_adj must be in \\(0, 1\\)"):
            EgoStitchConfig(tau_adj=tau_adj)

    def test_e2e_repair_calibration_fields_propagate_to_generator(self) -> None:
        from src.model.egostitch.composite import EgoStitchModel

        config = E2EConfig.from_mapping(
            {"generator": {"tau_adj": 0.4, "tau_div": 0.3, "l_gate_pos_weight": 5.0}}
        )
        model = EgoStitchModel(config)
        assert model.generator_cfg.tau_adj == 0.4
        assert model.generator_cfg.tau_div == 0.3
        assert model.generator_cfg.l_gate_pos_weight == 5.0

    def test_feature_standardization_defaults_to_the_rev32_mode(self) -> None:
        assert E2EConfig().generator.feature_standardization == "zscore_vfit_v1"
        assert E2EConfig().generator.feature_stats_sha256 == ""
        assert E2EConfig().generator.oracle_truth_source == "g_fit"

    def test_held_out_oracle_truth_is_explicit_and_oracle_only(self) -> None:
        cfg = E2EConfig.from_mapping(
            {
                "generator": {
                    "name": "oracle_struct",
                    "oracle_truth_source": "g_fit_plus_v_hold",
                }
            }
        )
        assert cfg.generator.oracle_truth_source == "g_fit_plus_v_hold"
        with pytest.raises(ValueError, match="oracle_truth_source"):
            GeneratorConfig(oracle_truth_source="test_graph")
        with pytest.raises(ValueError, match="requires name='oracle_struct'"):
            GeneratorConfig(oracle_truth_source="g_fit_plus_v_hold")

    def test_feature_standardization_allowlist(self) -> None:
        cfg = E2EConfig(generator=GeneratorConfig(feature_standardization="row_layernorm"))
        assert cfg.generator.feature_standardization == "row_layernorm"
        with pytest.raises(ValueError, match="feature_standardization"):
            GeneratorConfig(feature_standardization="layer_norm")

    def test_feature_stats_sha256_must_be_empty_or_64_hex(self) -> None:
        cfg = E2EConfig(generator=GeneratorConfig(feature_stats_sha256="ab" * 32))
        assert cfg.generator.feature_stats_sha256 == "ab" * 32
        with pytest.raises(ValueError, match="feature_stats_sha256"):
            GeneratorConfig(feature_stats_sha256="deadbeef")
        with pytest.raises(ValueError, match="feature_stats_sha256"):
            GeneratorConfig(feature_stats_sha256="AB" * 32)

    def test_from_mapping_accepts_the_string_fields(self) -> None:
        cfg = E2EConfig.from_mapping(
            {
                "generator": {
                    "feature_standardization": "row_layernorm",
                    "feature_stats_sha256": "cd" * 32,
                }
            }
        )
        assert cfg.generator.feature_standardization == "row_layernorm"
        assert cfg.generator.feature_stats_sha256 == "cd" * 32


class TestTokenizeLite:
    def test_shapes(self) -> None:
        module = TokenizeLite(_TINY)
        x, _ = _tiny_inputs()
        out = module(x)
        assert out.e.shape == (3, _TINY.d_z)
        assert out.deg_mu.shape == (3,)
        assert out.deg_log_sigma.shape == (3,)

    def test_deterministic(self) -> None:
        torch.manual_seed(0)
        module = TokenizeLite(_TINY)
        x, _ = _tiny_inputs()
        first = module(x)
        second = module(x)
        torch.testing.assert_close(first.e, second.e)


class TestImagineDecoder:
    def _decode(self, *, null_mode: torch.Tensor | None = None, seed: int = 0) -> SlotSet:
        torch.manual_seed(seed)
        module = ImagineDecoder(_TINY)
        x, ground = _tiny_inputs(seed=seed)
        tok_e = torch.randn(3, _TINY.d_z, generator=torch.Generator().manual_seed(seed))
        slots, _ = module(x, tok_e, ground, null_mode=null_mode)
        assert isinstance(slots, SlotSet)
        return slots

    def test_shapes_and_bounds(self) -> None:
        slots = self._decode()
        b, k, n_g = 3, _TINY.slots, _TINY.n_ground
        assert slots.h.shape == (b, k, _TINY.d_p)
        assert slots.pi.shape == (b, k)
        assert slots.mult.shape == (b, k)
        assert slots.gate.shape == (b, k)
        assert slots.pointer.shape == (b, k, n_g)
        assert slots.adj.shape == (b, k, k)
        assert slots.adj_logits.shape == (b, k, k)
        assert bool(((slots.pi >= 0) & (slots.pi <= 1)).all())
        assert bool(((slots.gate >= 0) & (slots.gate <= 1)).all())
        assert bool((slots.mult >= 1).all())
        assert bool((slots.mult <= _TINY.m_max).all())
        assert bool(((slots.adj >= 0) & (slots.adj <= 1)).all())
        torch.testing.assert_close(slots.adj, torch.sigmoid(slots.adj_logits))

    def test_adjacency_symmetric(self) -> None:
        slots = self._decode(seed=2)
        torch.testing.assert_close(slots.adj, slots.adj.transpose(1, 2))

    def test_pointer_rows_sum_to_one(self) -> None:
        slots = self._decode(seed=3)
        torch.testing.assert_close(
            slots.pointer.sum(dim=-1), torch.ones(3, _TINY.slots), atol=1e-6, rtol=0
        )

    def test_null_modes_change_outputs(self) -> None:
        full = self._decode(null_mode=torch.full((3,), NULL_MODE_FULL), seed=4)
        content = self._decode(null_mode=torch.full((3,), NULL_MODE_CONTENT), seed=4)
        both = self._decode(null_mode=torch.full((3,), NULL_MODE_ALL), seed=4)
        assert not torch.allclose(full.h, content.h)
        assert not torch.allclose(content.h, both.h)

    def test_denoise_queries_returned_separately(self) -> None:
        torch.manual_seed(5)
        module = ImagineDecoder(_TINY)
        x, ground = _tiny_inputs(seed=5)
        tok_e = torch.randn(3, _TINY.d_z)
        extra = torch.randn(3, 2, _TINY.d_p)
        slots, denoise = module(x, tok_e, ground, extra_proj=extra)
        assert slots.h.shape[1] == _TINY.slots
        assert denoise is not None
        assert denoise.h.shape == (3, 2, _TINY.d_p)
        assert bool((denoise.mult >= 1).all())

    def test_deterministic_given_weights(self) -> None:
        torch.manual_seed(6)
        module = ImagineDecoder(_TINY)
        x, ground = _tiny_inputs(seed=6)
        tok_e = torch.randn(3, _TINY.d_z, generator=torch.Generator().manual_seed(6))
        first, _ = module(x, tok_e, ground)
        second, _ = module(x, tok_e, ground)
        torch.testing.assert_close(first.h, second.h)
        torch.testing.assert_close(first.adj, second.adj)


class TestSinkhornPlan:
    def test_bfloat16_inputs_return_fp32_plan(self) -> None:
        h_i = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        h_j = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        pi = torch.rand(2, 4, dtype=torch.bfloat16)
        mult = torch.ones(2, 4, dtype=torch.bfloat16)
        plan = sinkhorn_plan(h_i, h_j, pi, pi, mult, mult, iters=3)
        assert plan.dtype == torch.float32
        assert bool(torch.isfinite(plan).all())

    def test_one_hot_pi_concentrates_on_matching_slots(self) -> None:
        # Side i has one live slot (index 0), side j one live slot (index 1):
        # the plan's mass must concentrate on the (0, 1) cell.
        k, d_p = 3, 2
        h_i = torch.zeros(1, k, d_p)
        h_j = torch.zeros(1, k, d_p)
        h_i[0, 0] = torch.tensor([1.0, 0.0])
        h_j[0, 1] = torch.tensor([1.0, 0.0])
        pi_i = torch.tensor([[1.0, 1e-6, 1e-6]])
        pi_j = torch.tensor([[1e-6, 1.0, 1e-6]])
        ones = torch.ones(1, k)
        plan = sinkhorn_plan(h_i, h_j, pi_i, pi_j, ones, ones, eps=0.05, iters=50)
        assert plan[0, 0, 1] == plan[0].max()
        assert plan[0, 0, 1] > 0.5

    def test_degenerate_all_zero_pi_stays_finite(self) -> None:
        k = 4
        h = torch.randn(2, k, 3)
        zeros = torch.zeros(2, k)
        ones = torch.ones(2, k)
        plan = sinkhorn_plan(h, h, zeros, zeros, ones, ones)
        assert bool(torch.isfinite(plan).all())
        assert bool((plan >= 0).all())

    def test_marginal_mass_bounded_by_unbalanced_relaxation(self) -> None:
        gen = torch.Generator().manual_seed(0)
        h_i = torch.randn(2, 4, 3, generator=gen)
        h_j = torch.randn(2, 4, 3, generator=gen)
        pi = torch.rand(2, 4, generator=gen)
        m = 1.0 + torch.rand(2, 4, generator=gen)
        plan = sinkhorn_plan(h_i, h_j, pi, pi, m, m)
        assert bool(torch.isfinite(plan).all())
        # Unbalanced OT: total mass is at most ~1 (unit marginals, KL slack).
        assert float(plan.sum(dim=(1, 2)).max()) <= 1.0 + 1e-4

    def test_differentiable(self) -> None:
        h_i = torch.randn(1, 3, 2, requires_grad=True)
        h_j = torch.randn(1, 3, 2)
        pi = torch.full((1, 3), 0.5)
        m = torch.ones(1, 3)
        plan = sinkhorn_plan(h_i, h_j, pi, pi, m, m)
        plan.sum().backward()  # type: ignore[no-untyped-call]
        assert h_i.grad is not None
        assert bool(torch.isfinite(h_i.grad).all())

    def test_cost_hand_case(self) -> None:
        h_i = torch.tensor([[[0.0, 0.0]]])
        h_j = torch.tensor([[[3.0, 4.0]]])
        pi_i = torch.tensor([[1.0]])
        pi_j = torch.tensor([[0.5]])
        cost = stitch_cost(h_i, h_j, pi_i, pi_j)
        assert cost[0, 0, 0] == pytest.approx(5.0 / 4.0 * 25.0 + 5.0 / 16.0 * 0.5)
