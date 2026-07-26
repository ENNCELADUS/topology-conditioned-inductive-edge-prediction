"""Tests for src.model.egostitch.losses: the Stage-1 loss tree."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.imagine import DenoiseSlots, SlotSet
from src.model.egostitch.losses import (
    RandomGIN,
    alignment_loss,
    degree_nll,
    denoise_losses,
    energy_distance,
    generated_ego_graph,
    generated_ego_stats,
    real_ego_graph,
    recon_losses,
    relational_loss,
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
    adj = 0.5 * (adj_raw + adj_raw.transpose(1, 2))
    return SlotSet(
        h=torch.randn(batch, k, d_p, generator=gen),
        pi=torch.rand(batch, k, generator=gen),
        mult=1.0 + torch.rand(batch, k, generator=gen),
        gate=torch.rand(batch, k, generator=gen),
        pointer=torch.softmax(torch.randn(batch, k, 3, generator=gen), dim=-1),
        adj=adj,
        adj_logits=torch.logit(adj.clamp(1e-6, 1.0 - 1e-6)),
    )


def _full_assignment(batch: int, n: int) -> Assignment:
    idx = [np.arange(n, dtype=np.int64) for _ in range(batch)]
    return Assignment(idx, [i.copy() for i in idx])


class TestReconLosses:
    def _targets(self, batch: int = 1, t: int = 4, d_p: int = 4) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(1)
        in_pool = torch.rand(batch, t, generator=gen) > 0.5
        return {
            "target_proj": torch.randn(batch, t, d_p, generator=gen),
            "target_mult": torch.ones(batch, t),
            "target_adj": (torch.rand(batch, t, t, generator=gen) > 0.5).float(),
            "target_in_pool": in_pool,
            "target_pool_index": torch.where(
                in_pool,
                torch.arange(t).remainder(_TINY.n_ground).expand(batch, -1),
                torch.full((batch, t), -1),
            ),
        }

    def test_all_terms_present_and_finite(self) -> None:
        slots = _slots()
        out = recon_losses(
            slots,
            _full_assignment(1, 4),
            config=_TINY,
            family="egostitch_e2e",
            **self._targets(),
        )
        assert set(out) == {"feat", "exist", "mult", "slotadj", "gate", "ptr", "div"}
        for term in out.values():
            assert bool(torch.isfinite(term))
            assert float(term) >= 0.0

    def test_legacy_family_preserves_pre_rev31_reconstruction(self) -> None:
        slots = _slots(k=3)
        targets = self._targets(t=2)
        targets["target_in_pool"] = torch.tensor([[True, False]])
        targets["target_pool_index"] = torch.tensor([[1, -1]])
        assignment = Assignment(
            [np.array([0, 1], dtype=np.int64)],
            [np.array([0, 1], dtype=np.int64)],
        )

        out = recon_losses(
            slots,
            assignment,
            config=_TINY,
            family="egostitch",
            **targets,
        )

        off_diag = ~torch.eye(2, dtype=torch.bool)
        expected_slotadj = F.binary_cross_entropy(
            slots.adj[0, :2, :2][off_diag].clamp(1e-6, 1.0 - 1e-6),
            targets["target_adj"][0, :2, :2][off_diag],
        )
        expected_gate = F.binary_cross_entropy(
            slots.gate[0, :2].clamp(1e-6, 1.0 - 1e-6),
            targets["target_in_pool"][0].float(),
        )
        torch.testing.assert_close(out["slotadj"], expected_slotadj)
        torch.testing.assert_close(out["gate"], expected_gate)
        assert float(out["ptr"]) == 0.0
        assert float(out["div"]) == 0.0

    def test_probability_bce_runs_outside_autocast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_bce = F.binary_cross_entropy

        def guarded_bce(
            input: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor | None = None,
            size_average: bool | None = None,
            reduce: bool | None = None,
            reduction: str = "mean",
        ) -> torch.Tensor:
            if torch.is_autocast_enabled("cpu"):
                raise RuntimeError("probability BCE called under autocast")
            return original_bce(input, target, weight, size_average, reduce, reduction)

        monkeypatch.setattr(F, "binary_cross_entropy", guarded_bce)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = recon_losses(
                _slots(),
                _full_assignment(1, 4),
                config=_TINY,
                family="egostitch_e2e",
                **self._targets(),
            )
            denoise = DenoiseSlots(
                h=torch.randn(1, 2, 4), pi=torch.rand(1, 2), mult=torch.ones(1, 2)
            )
            denoise_out = denoise_losses(
                denoise, target_proj=torch.randn(1, 2, 4), mask=torch.ones(1, 2, dtype=torch.bool)
            )
        assert all(bool(torch.isfinite(term)) for term in out.values())
        assert all(bool(torch.isfinite(term)) for term in denoise_out.values())

    def test_perfect_slots_zero_feat_loss(self) -> None:
        targets = self._targets()
        slots = _slots()
        slots = slots._replace(h=targets["target_proj"].clone(), mult=torch.ones(1, 4))
        out = recon_losses(
            slots,
            _full_assignment(1, 4),
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )
        assert float(out["feat"]) == pytest.approx(0.0, abs=1e-9)
        assert float(out["mult"]) == pytest.approx(0.0, abs=1e-9)

    def test_exist_balances_matched_and_unmatched(self) -> None:
        # 1 matched slot with pi ~ 1 and 3 unmatched with pi ~ 0: near-zero loss.
        slots = _slots()
        pi = torch.tensor([[0.999, 0.001, 0.001, 0.001]])
        slots = slots._replace(pi=pi)
        assignment = Assignment([np.array([0], dtype=np.int64)], [np.array([0], dtype=np.int64)])
        out = recon_losses(
            slots,
            assignment,
            config=_TINY,
            family="egostitch_e2e",
            **self._targets(),
        )
        assert float(out["exist"]) < 0.01

    def test_empty_assignment_yields_zero_matched_terms(self) -> None:
        slots = _slots()
        assignment = Assignment([np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)])
        out = recon_losses(
            slots,
            assignment,
            config=_TINY,
            family="egostitch_e2e",
            **self._targets(),
        )
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
            target_pool_index=targets["target_pool_index"],
            config=_TINY,
            family="egostitch_e2e",
        )
        out["feat"].backward()  # type: ignore[no-untyped-call]
        assert h.grad is not None
        assert target_proj.grad is None  # stop-gradient targets (spec Sec 13.7)

    def test_slotadj_uses_temperature_scaled_logits(self) -> None:
        logits = torch.tensor([[[0.0, 0.8], [0.8, 0.0]]])
        slots = _slots(k=2)._replace(adj=torch.sigmoid(logits), adj_logits=logits)
        targets = {
            "target_proj": torch.zeros(1, 2, 4),
            "target_mult": torch.ones(1, 2),
            "target_adj": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            "target_in_pool": torch.zeros(1, 2, dtype=torch.bool),
            "target_pool_index": torch.full((1, 2), -1),
        }
        out = recon_losses(
            slots,
            _full_assignment(1, 2),
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )
        expected = F.binary_cross_entropy_with_logits(
            torch.tensor([0.8, 0.8]) / _TINY.tau_adj, torch.ones(2)
        )
        torch.testing.assert_close(out["slotadj"], expected)

    @pytest.mark.parametrize("pos_weight", [6.17, 1.0])
    def test_gate_pos_weight_matches_hand_computation(self, pos_weight: float) -> None:
        config = EgoStitchConfig(
            input_dim=8,
            d_p=4,
            d_z=4,
            d_h=8,
            slots=4,
            n_ground=3,
            n_heads=2,
            l_gate_pos_weight=pos_weight,
        )
        gate = torch.tensor([[0.8, 0.2, 0.1, 0.3]])
        slots = _slots()._replace(gate=gate)
        targets = self._targets()
        targets["target_in_pool"] = torch.tensor([[True, False, False, False]])
        targets["target_pool_index"] = torch.tensor([[0, -1, -1, -1]])
        out = recon_losses(
            slots,
            _full_assignment(1, 4),
            config=config,
            family="egostitch_e2e",
            **targets,
        )
        labels = targets["target_in_pool"].float()
        expected = -(
            pos_weight * labels * torch.log(gate) + (1.0 - labels) * torch.log1p(-gate)
        ).mean()
        torch.testing.assert_close(out["gate"], expected)
        if pos_weight == 1.0:
            torch.testing.assert_close(out["gate"], F.binary_cross_entropy(gate, labels))

    def test_diversity_threshold_and_matched_pair_exclusion(self) -> None:
        orthogonal = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        )
        targets = self._targets()
        one_match = Assignment([np.array([0])], [np.array([0])])
        below = recon_losses(
            _slots()._replace(h=orthogonal),
            one_match,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["div"]
        assert float(below) == 0.0

        similar = orthogonal.clone()
        similar[0, 1] = similar[0, 0]
        unmatched = Assignment([np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)])
        above = recon_losses(
            _slots()._replace(h=similar),
            unmatched,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["div"]
        assert float(above) > 0.0

        two_matches = Assignment([np.array([0, 1])], [np.array([0, 1])])
        excluded = recon_losses(
            _slots()._replace(h=similar),
            two_matches,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["div"]
        assert float(excluded) == 0.0

        matched_unmatched = recon_losses(
            _slots()._replace(h=similar),
            one_match,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["div"]
        assert float(matched_unmatched) > 0.0

    def test_pointer_masks_out_of_pool_and_matches_hand_ce(self) -> None:
        pointer = torch.tensor([[[0.1, 0.7, 0.2], [0.4, 0.5, 0.1]]])
        slots = _slots(k=2)._replace(pointer=pointer)
        targets = {
            "target_proj": torch.zeros(1, 2, 4),
            "target_mult": torch.ones(1, 2),
            "target_adj": torch.zeros(1, 2, 2),
            "target_in_pool": torch.zeros(1, 2, dtype=torch.bool),
            "target_pool_index": torch.full((1, 2), -1),
        }
        masked = recon_losses(
            slots,
            _full_assignment(1, 2),
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["ptr"]
        assert bool(torch.isfinite(masked))
        assert float(masked) == 0.0

        targets["target_in_pool"][0, 0] = True
        targets["target_pool_index"][0, 0] = 1
        supervised = recon_losses(
            slots,
            _full_assignment(1, 2),
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["ptr"]
        torch.testing.assert_close(supervised, -torch.log(torch.tensor(0.7)))

        partial = Assignment([np.array([0])], [np.array([0])])
        targets["target_in_pool"][:] = True
        targets["target_pool_index"][:] = torch.tensor([[1, 0]])
        first = recon_losses(
            slots,
            partial,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["ptr"]
        changed_unmatched = slots._replace(pointer=pointer.clone())
        changed_unmatched.pointer[0, 1] = torch.tensor([0.99, 0.005, 0.005])
        second = recon_losses(
            changed_unmatched,
            partial,
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )["ptr"]
        torch.testing.assert_close(first, -torch.log(torch.tensor(0.7)))
        torch.testing.assert_close(second, first)

    def test_pointer_loss_reaches_pointer_head(self) -> None:
        torch.manual_seed(7)
        from src.model.egostitch.imagine import ImagineDecoder

        decoder = ImagineDecoder(_TINY)
        x = torch.randn(1, _TINY.input_dim)
        e = torch.randn(1, _TINY.d_z)
        ground = torch.randn(1, _TINY.n_ground, _TINY.input_dim)
        slots, _ = decoder(x, e, ground)
        targets = self._targets()
        targets["target_in_pool"] = torch.tensor([[True, False, False, False]])
        targets["target_pool_index"] = torch.tensor([[1, -1, -1, -1]])
        out = recon_losses(
            slots,
            _full_assignment(1, 4),
            config=_TINY,
            family="egostitch_e2e",
            **targets,
        )
        out["ptr"].backward()  # type: ignore[no-untyped-call]
        assert decoder.head_pointer.weight.grad is not None
        assert bool((decoder.head_pointer.weight.grad != 0).any())


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
            adj_logits=torch.logit(adj.clamp(1e-6, 1.0 - 1e-6)),
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
    @pytest.mark.parametrize(
        ("component", "expected"),
        [
            ("feat", 1.0),
            ("exist", 0.5),
            ("mult", 0.25),
            ("deg", 0.5),
            ("slotadj", 0.5),
            ("gate", 0.25),
            ("ptr", 0.25),
            ("align", 0.5),
            ("div", 0.1),
            ("rel", 0.25),
        ],
    )
    def test_each_recon_component_uses_its_pinned_weight(
        self, component: str, expected: float
    ) -> None:
        zero = torch.tensor(0.0)
        recon = {
            name: torch.tensor(float(name == component))
            for name in ("feat", "exist", "mult", "slotadj", "gate", "ptr", "align", "div", "rel")
        }
        total, _ = stage1_total(
            EgoStitchConfig(),
            family="egostitch_e2e",
            edge=zero,
            recon=recon,
            deg=torch.tensor(float(component == "deg")),
            real_egostat=zero,
            real_gin=zero,
            ssl_noise=zero,
            ssl_pool=zero,
        )
        assert float(total) == pytest.approx(expected)

    def test_pinned_weights(self) -> None:
        config = EgoStitchConfig()
        assert {
            "L_feat": config.w_feat,
            "L_exist": config.w_exist,
            "L_mult": config.w_mult,
            "L_deg": config.w_deg,
            "L_slotadj": config.w_slotadj,
            "L_gate": config.w_gate,
            "L_ptr": config.w_ptr,
            "L_align": config.w_align,
            "L_div": config.w_div,
            "L_rel": config.w_rel,
        } == {
            "L_feat": 1.0,
            "L_exist": 0.5,
            "L_mult": 0.25,
            "L_deg": 0.5,
            "L_slotadj": 0.5,
            "L_gate": 0.25,
            "L_ptr": 0.25,
            "L_align": 0.5,
            "L_div": 0.1,
            "L_rel": 0.25,
        }
        one = torch.tensor(1.0)
        align = alignment_loss(
            torch.log(torch.full((1, 2, 2), 0.25)),
            torch.tensor([[[True, False], [False, False]]]),
            positive_real_mask=torch.ones(1),
        )
        rel = relational_loss(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.ones(1),
        )
        recon = {
            "feat": one,
            "exist": one,
            "mult": one,
            "slotadj": one,
            "gate": one,
            "ptr": one,
            "align": align,
            "div": one,
            "rel": rel,
        }
        total, parts = stage1_total(
            config,
            family="egostitch_e2e",
            edge=one,
            recon=recon,
            deg=one,
            real_egostat=one,
            real_gin=one,
            ssl_noise=one,
            ssl_pool=one,
        )
        expected_recon = 3.35 + 0.5 * float(align) + 0.25 * float(rel)
        expected_total = 1.0 + 0.5 + 0.1 + expected_recon
        assert float(total) == pytest.approx(expected_total)
        assert parts["recon"] == pytest.approx(expected_recon)
        assert parts["recon_align"] == pytest.approx(float(align))
        assert parts["recon_rel"] == pytest.approx(float(rel))
        assert parts["real"] == pytest.approx(1.0)
        assert parts["total"] == pytest.approx(expected_total)

    def test_legacy_total_ignores_rev31_components(self) -> None:
        zero = torch.tensor(0.0)
        recon = {
            "feat": zero,
            "exist": zero,
            "mult": zero,
            "slotadj": zero,
            "gate": zero,
            "ptr": torch.tensor(3.0),
            "align": torch.tensor(4.0),
            "div": torch.tensor(5.0),
            "rel": torch.tensor(6.0),
        }
        total, parts = stage1_total(
            EgoStitchConfig(),
            family="egostitch",
            edge=zero,
            recon=recon,
            deg=zero,
            real_egostat=zero,
            real_gin=zero,
            ssl_noise=zero,
            ssl_pool=zero,
        )
        assert float(total) == 0.0
        assert parts["recon"] == 0.0

    def test_e2e_component_factors_do_not_anneal_repair_losses(self) -> None:
        one = torch.tensor(1.0)
        recon = dict.fromkeys(
            ("feat", "exist", "mult", "slotadj", "gate", "ptr", "align", "div", "rel"),
            one,
        )
        factors = {
            "feat": 0.25,
            "exist": 0.25,
            "mult": 0.25,
            "deg": 0.25,
            "slotadj": 1.0,
            "gate": 1.0,
            "ptr": 1.0,
            "align": 1.0,
            "div": 1.0,
            "rel": 1.0,
        }
        _, parts = stage1_total(
            EgoStitchConfig(),
            family="egostitch_e2e",
            edge=torch.tensor(0.0),
            recon=recon,
            deg=one,
            real_egostat=torch.tensor(0.0),
            real_gin=torch.tensor(0.0),
            ssl_noise=torch.tensor(0.0),
            ssl_pool=torch.tensor(0.0),
            recon_factors=factors,
        )
        assert parts["recon"] == pytest.approx(2.4125)

    def test_parts_are_floats(self) -> None:
        zero = torch.tensor(0.0)
        recon = {
            name: torch.tensor(float(index))
            for index, name in enumerate(
                ("feat", "exist", "mult", "slotadj", "gate", "ptr", "align", "div", "rel"),
                start=1,
            )
        }
        deg = torch.tensor(10.0)
        _, parts = stage1_total(
            _TINY,
            edge=zero,
            recon=recon,
            deg=deg,
            real_egostat=zero,
            real_gin=zero,
            ssl_noise=zero,
            ssl_pool=zero,
        )
        assert set(parts) == {
            "edge",
            "recon",
            "recon_feat",
            "recon_exist",
            "recon_mult",
            "recon_deg",
            "recon_slotadj",
            "recon_gate",
            "recon_ptr",
            "recon_align",
            "recon_div",
            "recon_rel",
            "real",
            "real_egostat",
            "real_gin",
            "ssl",
            "total",
        }
        for name, value in recon.items():
            assert parts[f"recon_{name}"] == float(value)
        assert parts["recon_deg"] == float(deg)
        assert all(isinstance(v, float) for v in parts.values())
