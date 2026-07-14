"""Tests for src.model.egostitch.losses: the Stage-1 loss tree."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.losses import (
    RandomGIN,
    degree_nll,
    energy_distance,
    generated_ego_graph,
    generated_ego_stats,
    real_ego_graph,
    recon_losses,
    ssl_consistency,
    stage1_total,
    standardized_energy_distance,
)
from src.model.egostitch.matching import Assignment

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


def _slots(batch: int = 1, k: int = 4, d_p: int = 4, *, seed: int = 0) -> SlotSet:
    gen = torch.Generator().manual_seed(seed)
    adj_raw = torch.rand(batch, k, k, generator=gen)
    return SlotSet(
        h=torch.randn(batch, k, d_p, generator=gen),
        pi=torch.rand(batch, k, generator=gen),
        mult=1.0 + torch.rand(batch, k, generator=gen),
        gate=torch.rand(batch, k, generator=gen),
        pointer=torch.softmax(torch.randn(batch, k, 2, generator=gen), dim=-1),
        adj=0.5 * (adj_raw + adj_raw.transpose(1, 2)),
    )


def _full_assignment(batch: int, n: int) -> Assignment:
    idx = [np.arange(n, dtype=np.int64) for _ in range(batch)]
    return Assignment(idx, [i.copy() for i in idx])


class TestReconLosses:
    def _targets(self, batch: int = 1, t: int = 4, d_p: int = 4) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(1)
        return {
            "target_proj": torch.randn(batch, t, d_p, generator=gen),
            "target_mult": torch.ones(batch, t),
            "target_adj": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
            "target_in_pool": torch.rand(batch, t, generator=gen) > 0.5,
        }

    def test_all_terms_present_and_finite(self) -> None:
        slots = _slots()
        out = recon_losses(slots, _full_assignment(1, 4), **self._targets())
        assert set(out) == {"feat", "exist", "mult", "slotadj", "gate"}
        for term in out.values():
            assert bool(torch.isfinite(term))
            assert float(term) >= 0.0

    def test_perfect_slots_zero_feat_loss(self) -> None:
        targets = self._targets()
        slots = _slots()
        slots = slots._replace(h=targets["target_proj"].clone(), mult=torch.ones(1, 4))
        out = recon_losses(slots, _full_assignment(1, 4), **targets)
        assert float(out["feat"]) == pytest.approx(0.0, abs=1e-9)
        assert float(out["mult"]) == pytest.approx(0.0, abs=1e-9)

    def test_exist_balances_matched_and_unmatched(self) -> None:
        # 1 matched slot with pi ~ 1 and 3 unmatched with pi ~ 0: near-zero loss.
        slots = _slots()
        pi = torch.tensor([[0.999, 0.001, 0.001, 0.001]])
        slots = slots._replace(pi=pi)
        assignment = Assignment([np.array([0], dtype=np.int64)], [np.array([0], dtype=np.int64)])
        out = recon_losses(slots, assignment, **self._targets())
        assert float(out["exist"]) < 0.01

    def test_empty_assignment_yields_zero_matched_terms(self) -> None:
        slots = _slots()
        assignment = Assignment([np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)])
        out = recon_losses(slots, assignment, **self._targets())
        assert float(out["feat"]) == 0.0
        assert float(out["mult"]) == 0.0
        assert float(out["gate"]) == 0.0

    def test_gradients_reach_slots_not_targets(self) -> None:
        targets = self._targets()
        target_proj = targets["target_proj"].requires_grad_(True)
        h = torch.randn(1, 4, 4, requires_grad=True)
        slots = _slots()._replace(h=h)
        out = recon_losses(
            slots,
            _full_assignment(1, 4),
            target_proj=target_proj,
            target_mult=targets["target_mult"],
            target_adj=targets["target_adj"],
            target_in_pool=targets["target_in_pool"],
        )
        out["feat"].backward()  # type: ignore[no-untyped-call]
        assert h.grad is not None
        assert target_proj.grad is None  # stop-gradient targets (spec Sec 13.7)


class TestDegreeNll:
    def test_minimized_at_true_log_degree(self) -> None:
        true_degree = torch.tensor([8.0])
        log_sigma = torch.tensor([0.0])
        at_truth = degree_nll(torch.log(true_degree), log_sigma, true_degree)
        off = degree_nll(torch.log(true_degree) + 1.0, log_sigma, true_degree)
        assert float(at_truth) < float(off)

    def test_degree_zero_clamped(self) -> None:
        out = degree_nll(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        assert bool(torch.isfinite(out))


class TestEnergyDistance:
    def test_identical_distributions_near_zero(self) -> None:
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(400, 3, generator=gen)
        y = torch.randn(400, 3, generator=gen)
        assert float(energy_distance(x, y)) == pytest.approx(0.0, abs=0.05)

    def test_shifted_distributions_positive(self) -> None:
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(200, 3, generator=gen)
        y = torch.randn(200, 3, generator=gen) + 5.0
        assert float(energy_distance(x, y)) > 1.0

    def test_empty_input_zero(self) -> None:
        assert float(energy_distance(torch.empty(0, 3), torch.randn(4, 3))) == 0.0

    def test_standardized_uses_real_side_scale(self) -> None:
        gen = torch.Generator().manual_seed(2)
        real = torch.randn(100, 2, generator=gen) * 100.0
        generated = real.clone()
        assert float(standardized_energy_distance(generated, real)) == pytest.approx(0.0, abs=1e-6)


class TestGeneratedEgoStats:
    def test_hand_case(self) -> None:
        # Two slots: pi = [1, 0.5], m = [2, 4]; adj[0,1] = 0.5.
        adj = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]])
        slots = SlotSet(
            h=torch.zeros(1, 2, 3),
            pi=torch.tensor([[1.0, 0.5]]),
            mult=torch.tensor([[2.0, 4.0]]),
            gate=torch.zeros(1, 2),
            pointer=torch.full((1, 2, 1), 1.0),
            adj=adj,
        )
        stats = generated_ego_stats(slots)
        d_soft = 1.0 * 2.0 + 0.5 * 4.0  # 4.0
        e_nn = 0.5 * (1.0 * 2.0) * (0.5 * 4.0)  # adj * w0 * w1 = 2.0
        choose2 = d_soft * (d_soft - 1.0) / 2.0  # 6.0
        choose2_ego = (d_soft + 1.0) * d_soft / 2.0  # 10.0
        expected = torch.tensor(
            [[d_soft, e_nn / choose2, d_soft + e_nn, (d_soft + e_nn) / choose2_ego]]
        )
        torch.testing.assert_close(stats, expected)


class TestRandomGin:
    def test_frozen_and_deterministic(self) -> None:
        gin_a = RandomGIN(_TINY)
        gin_b = RandomGIN(_TINY)
        assert all(not p.requires_grad for p in gin_a.parameters())
        features = torch.randn(2, 5, 4)
        adjacency = torch.rand(2, 5, 5)
        torch.testing.assert_close(gin_a(features, adjacency), gin_b(features, adjacency))

    def test_output_shape(self) -> None:
        gin = RandomGIN(_TINY)
        out = gin(torch.randn(3, 5, 4), torch.rand(3, 5, 5))
        assert out.shape == (3, _TINY.gin_hidden)

    def test_graph_builders_shapes(self) -> None:
        slots = _slots(batch=2)
        gen_feat, gen_adj = generated_ego_graph(slots)
        assert gen_feat.shape == (2, 5, 4)
        assert gen_adj.shape == (2, 5, 5)
        torch.testing.assert_close(gen_adj, gen_adj.transpose(1, 2))
        real_feat, real_adj = real_ego_graph(
            torch.ones(2, 4),
            torch.zeros(2, 4, 4),
            torch.ones(2, 4, dtype=torch.bool),
            torch.zeros(2, 4, dtype=torch.bool),
        )
        assert real_feat.shape == (2, 5, 4)
        assert real_adj.shape == (2, 5, 5)


class TestSslConsistency:
    def test_identical_passes_zero(self) -> None:
        slots = _slots()
        mask = torch.ones(1, 4, dtype=torch.bool)
        assert float(ssl_consistency(slots, slots, ungrounded=mask)) == 0.0

    def test_no_ungrounded_slots_zero(self) -> None:
        slots_a = _slots(seed=1)
        slots_b = _slots(seed=2)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        assert float(ssl_consistency(slots_a, slots_b, ungrounded=mask)) == 0.0

    def test_perturbed_pass_positive(self) -> None:
        slots_a = _slots(seed=1)
        slots_b = _slots(seed=2)
        mask = torch.ones(1, 4, dtype=torch.bool)
        assert float(ssl_consistency(slots_a, slots_b, ungrounded=mask)) > 0.0


class TestStage1Total:
    def test_pinned_weights(self) -> None:
        one = torch.tensor(1.0)
        recon = {"feat": one, "exist": one, "mult": one, "slotadj": one, "gate": one}
        total, parts = stage1_total(
            EgoStitchConfig(),
            edge=one,
            recon=recon,
            deg=one,
            real_egostat=one,
            real_gin=one,
            ssl_noise=one,
            ssl_pool=one,
        )
        # L_recon = 1 + 0.5 + 0.25 + 0.5 + 0.5 + 0.25 = 3.0 (deg folded at 0.5)
        # L_real = 2/3 + 1/3 = 1.0 ; L_ssl = 1.0
        # total = 1 + 0.5*1 + 0.1*1 + 1.0*3.0 = 4.6
        assert float(total) == pytest.approx(4.6)
        assert parts["recon"] == pytest.approx(3.0)
        assert parts["real"] == pytest.approx(1.0)
        assert parts["total"] == pytest.approx(4.6)

    def test_parts_are_floats(self) -> None:
        zero = torch.tensor(0.0)
        recon = {"feat": zero, "exist": zero, "mult": zero, "slotadj": zero, "gate": zero}
        _, parts = stage1_total(
            _TINY,
            edge=zero,
            recon=recon,
            deg=zero,
            real_egostat=zero,
            real_gin=zero,
            ssl_noise=zero,
            ssl_pool=zero,
        )
        assert all(isinstance(v, float) for v in parts.values())
