"""Tests for the rev-3.2 registered per-dimension F0 standardization (spec Sec 13.19.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.data.feature_stats import FeatureStats, compute_feature_stats
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.imagine import EgoStitchStage1, FeatureStandardizer

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
    sinkhorn_iters=5,
)


def _stats(seed: int = 0) -> FeatureStats:
    gen = np.random.default_rng(seed)
    rows = (30.0 + 4.0 * gen.standard_normal((32, _TINY.input_dim))).astype(np.float32)
    return compute_feature_stats(rows, [f"n{i}" for i in range(32)])


class TestModes:
    def test_none_is_identity(self) -> None:
        model = EgoStitchStage1(_TINY)
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)
        torch.testing.assert_close(model.normalize_features(x), x)
        assert model.feature_stats_digest_hex == ""

    def test_row_layernorm_preserves_the_rev31_transform(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        x = 25.0 + 7.0 * torch.randn(3, _TINY.input_dim)
        normalized = model.normalize_features(x)
        torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(3), atol=1e-6, rtol=0.0)
        assert tuple(model.feature_norm.parameters()) == ()

    def test_zscore_applies_registered_constants(self) -> None:
        stats = _stats()
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(stats)
        x = torch.randn(5, _TINY.input_dim)

        expected = (x - torch.from_numpy(stats.mu)) / torch.from_numpy(stats.sigma)
        torch.testing.assert_close(model.normalize_features(x), expected)
        torch.testing.assert_close(
            model.project_features(x), model.proj(model.normalize_features(x))
        )

    def test_zscore_broadcasts_over_grounding_candidates(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(_stats())
        ground = torch.randn(2, _TINY.n_ground, _TINY.input_dim)
        assert model.normalize_features(ground).shape == ground.shape

    def test_zscore_fails_closed_before_statistics_are_registered(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        with pytest.raises(RuntimeError, match="feature standardization statistics"):
            model.normalize_features(torch.randn(2, _TINY.input_dim))

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature standardization"):
            EgoStitchStage1(_TINY, feature_standardization="zscore_v9")  # type: ignore[arg-type]


class TestCheckpointBuffers:
    def test_statistics_survive_a_state_dict_roundtrip(self) -> None:
        stats = _stats(1)
        source = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        source.set_feature_stats(stats)

        target = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        target.load_state_dict(source.state_dict())

        assert target.feature_stats_digest_hex == stats.digest
        x = torch.randn(3, _TINY.input_dim)
        torch.testing.assert_close(target.normalize_features(x), source.normalize_features(x))

    def test_buffers_are_persistent_and_named(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        keys = set(model.state_dict())
        assert {
            "feature_norm.feature_mu",
            "feature_norm.feature_sigma",
            "feature_norm.feature_stats_ready",
            "feature_norm.feature_stats_digest",
        } <= keys

    def test_row_layernorm_checkpoints_carry_no_statistics_buffers(self) -> None:
        model = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        assert not [key for key in model.state_dict() if key.startswith("feature_norm.")]

    def test_a_rev31_checkpoint_cannot_be_loaded_as_zscore(self) -> None:
        legacy = EgoStitchStage1(_TINY, feature_standardization="row_layernorm")
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        with pytest.raises(RuntimeError):
            model.load_state_dict(legacy.state_dict())


class TestSslNoiseCoordinates:
    def test_perturbation_is_scaled_into_standardized_coordinates(self) -> None:
        stats = _stats(2)
        model = EgoStitchStage1(_TINY, feature_standardization="zscore_vfit_v1")
        model.set_feature_stats(stats)
        x = torch.randn(4, _TINY.input_dim)
        raw_noise = 0.05 * torch.randn(4, _TINY.input_dim)

        perturbed = model.normalize_features(x + model.scale_feature_perturbation(raw_noise))
        expected = model.normalize_features(x) + raw_noise
        torch.testing.assert_close(perturbed, expected, atol=1e-5, rtol=0.0)

    def test_legacy_modes_leave_the_perturbation_untouched(self) -> None:
        for mode in ("none", "row_layernorm"):
            model = EgoStitchStage1(_TINY, feature_standardization=mode)
            noise = torch.randn(2, _TINY.input_dim)
            torch.testing.assert_close(model.scale_feature_perturbation(noise), noise)


class TestInitHealthUnderAnisotropicFeatures:
    """Spec Sec 13.19.1: a random model must not be born above the guard line."""

    @staticmethod
    def _config() -> EgoStitchConfig:
        return EgoStitchConfig(
            input_dim=64,
            d_p=32,
            d_z=16,
            d_h=32,
            slots=16,
            m_max=8,
            n_ground=50,
            decoder_layers=2,
            n_heads=4,
            gin_hidden=16,
            gin_layers=2,
            sinkhorn_iters=20,
        )

    @staticmethod
    def _features(
        cfg: EgoStitchConfig, *, seed: int = 0
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = np.random.default_rng(seed)
        direction = np.abs(gen.standard_normal(cfg.input_dim)) + 1.0
        rows = (40.0 * direction + 4.0 * gen.standard_normal((256, cfg.input_dim))).astype(
            np.float32
        )
        # Two disjoint endpoint batches so the transport plan below aligns two
        # distinct nodes, matching what production actually scores (a self-plan
        # gets an exact-zero cost diagonal from `stitch_cost` regardless of
        # collapse, which would make the plan-based assertions free passes).
        x_a = torch.from_numpy(rows[:8])
        x_b = torch.from_numpy(rows[8:16])
        pool_a = torch.from_numpy(
            np.stack([rows[gen.choice(256, cfg.n_ground, replace=False)] for _ in range(8)])
        )
        pool_b = torch.from_numpy(
            np.stack([rows[gen.choice(256, cfg.n_ground, replace=False)] for _ in range(8)])
        )
        return rows, x_a, pool_a, x_b, pool_b

    def _dispersion(self, mode: str, *, seed: int = 0) -> dict[str, float]:
        # Local imports: keep the trainer's heavy dependencies (accelerate, scipy,
        # networkx) out of collection for the tests in this file that don't need them.
        from src.model.egostitch.generator.assemble import sinkhorn_plan
        from src.train_egostitch import e2e_dispersion_statistics

        cfg = self._config()
        rows, x_a, pool_a, x_b, pool_b = self._features(cfg, seed=seed)
        torch.manual_seed(11)
        model = EgoStitchStage1(cfg, feature_standardization=mode)  # type: ignore[arg-type]
        if mode == "zscore_vfit_v1":
            model.set_feature_stats(
                compute_feature_stats(rows, [f"n{i}" for i in range(rows.shape[0])])
            )
        model.eval()
        with torch.no_grad():
            enc_a = model.encode_nodes(x_a, pool_a)
            enc_b = model.encode_nodes(x_b, pool_b)
            plan = sinkhorn_plan(
                enc_a.slots.h,
                enc_b.slots.h,
                enc_a.slots.pi,
                enc_b.slots.pi,
                enc_a.slots.mult,
                enc_b.slots.mult,
                eps=cfg.sinkhorn_eps,
                iters=cfg.sinkhorn_iters,
                tau=cfg.sinkhorn_tau,
            )
            stats = e2e_dispersion_statistics(enc_a.slots.pi, enc_a.slots.h, enc_a.slots.adj, plan)
            stats["plan_total_mass"] = float(plan.float().sum(dim=(1, 2)).mean())
        return stats

    def test_the_fixture_reproduces_the_measured_f0_anisotropy(self) -> None:
        cfg = self._config()
        rows, _, _, _, _ = self._features(cfg)
        normalized = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        gram = normalized @ normalized.T
        off_diagonal = gram[~np.eye(rows.shape[0], dtype=bool)]
        assert float(off_diagonal.mean()) > 0.9

    def test_row_layernorm_is_born_above_the_guard_line(self) -> None:
        assert self._dispersion("row_layernorm")["h_pairwise_cosine_mean"] > 0.95

    def test_zscore_is_born_healthy_on_every_guard_criterion(self) -> None:
        stats = self._dispersion("zscore_vfit_v1")
        assert stats["h_pairwise_cosine_mean"] < 0.9
        assert stats["plan_rank1_marginal_residual"] > 0.3
        assert stats["plan_total_mass"] > 1e-6


class TestStandardizerUnit:
    def test_load_stats_rejects_a_dimension_mismatch(self) -> None:
        standardizer = FeatureStandardizer(_TINY.input_dim)
        with pytest.raises(ValueError, match="dimension"):
            standardizer.load_stats(torch.zeros(3), torch.ones(3), "ab" * 32)

    def test_load_stats_rejects_a_non_positive_sigma(self) -> None:
        standardizer = FeatureStandardizer(_TINY.input_dim)
        sigma = torch.ones(_TINY.input_dim)
        sigma[0] = 0.0
        with pytest.raises(ValueError, match="sigma"):
            standardizer.load_stats(torch.zeros(_TINY.input_dim), sigma, "ab" * 32)
