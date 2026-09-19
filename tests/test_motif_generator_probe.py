"""The generator probe's pure readings: spread, strata and the decomposition's fidelity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.data.motif_template import EDGE_ENDPOINTS
from src.experiments.motif_generator_probe import (
    STRATA,
    CachedInputs,
    _short_enough,
    attention_side,
    check_decomposition,
    closure_counts,
    decompose,
    slot_spread,
    spread_summary,
    stage_groups,
    stratified_sample,
    stratum_of,
)
from src.model.egostitch.classifier.motif_prompt import MotifGenerator, MotifPromptConfig


def _generator(seed: int = 0, **extra: object) -> MotifGenerator:
    torch.manual_seed(seed)
    cfg = MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt", **extra}
    )
    return MotifGenerator(d_model=32, cfg=cfg).eval()


def _states(n: int = 5, length: int = 9, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.full((n,), length, dtype=torch.long)
    lengths[0] = length - 2
    return torch.randn(n, length, 32, generator=generator), lengths


def test_slot_spread_is_zero_exactly_when_the_slots_are_identical() -> None:
    same = torch.randn(3, 1, 7).expand(3, 8, 7).contiguous()
    assert float(slot_spread(same).max()) == pytest.approx(0.0, abs=1e-7)
    distinct = torch.randn(3, 8, 7)
    assert float(slot_spread(distinct).min()) > 0.0


def test_slot_spread_scales_with_the_states() -> None:
    states = torch.randn(4, 8, 6)
    base = slot_spread(states)
    scaled = slot_spread(states * 3.0)
    assert torch.allclose(scaled, base * 3.0, atol=1e-5)


def test_slot_spread_rejects_a_non_block_input() -> None:
    with pytest.raises(ValueError, match="state block"):
        slot_spread(torch.randn(4, 8))


def test_spread_summary_reports_the_identity_fractions() -> None:
    summary = spread_summary(
        np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0], dtype=np.float64)
    )
    assert summary["rows"] == 2.0
    assert summary["frac_identical_below_1e-06"] == pytest.approx(0.5)
    assert summary["frac_identical_below_1e-02"] == pytest.approx(0.5)
    assert summary["D"] == pytest.approx(0.5)


def test_closure_counts_and_strata_read_the_witnesses() -> None:
    targets = torch.zeros(3, 96)
    targets[1, 0] = 0.5
    targets[1, 8 + 1] = 0.5
    targets[2, :8] = 0.5
    assert closure_counts(targets).tolist() == [0, 2, 8]
    assert stratum_of(closure_counts(targets)) == ["0", "1-2", "8"]


def test_stratified_sample_is_even_when_every_stratum_is_full() -> None:
    counts = np.array([0] * 40 + [1] * 40 + [5] * 40 + [8] * 40, dtype=np.int64)
    picked = stratified_sample(counts, rows=40, seed=3)
    assert picked.size == 40
    assert np.all(np.diff(picked) > 0)
    names = stratum_of(counts[picked])
    assert [names.count(name) for name, _, _ in STRATA] == [10, 10, 10, 10]


def test_stratified_sample_redistributes_a_short_stratum() -> None:
    counts = np.array([0] * 50 + [1] * 50 + [5] * 50 + [8] * 2, dtype=np.int64)
    picked = stratified_sample(counts, rows=40, seed=5)
    assert picked.size == 40
    names = stratum_of(counts[picked])
    assert names.count("8") == 2
    assert min(names.count(name) for name, _, _ in STRATA[:3]) >= 12


def test_stratified_sample_takes_everything_it_can() -> None:
    counts = np.array([0, 0, 3], dtype=np.int64)
    assert stratified_sample(counts, rows=0, seed=1).tolist() == [0, 1, 2]
    assert stratified_sample(counts, rows=99, seed=1).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="non-negative"):
        stratified_sample(counts, rows=-1, seed=1)


@pytest.mark.parametrize("slot_read", ["bare", "residual_block"])
def test_decomposition_reproduces_the_generator(slot_read: str) -> None:
    generator = _generator(slot_read=slot_read)
    states_u, lengths_u = _states(seed=1)
    states_v, lengths_v = _states(seed=2)
    assert check_decomposition(generator, states_u, states_v, lengths_u, lengths_v) < 1e-5


@pytest.mark.parametrize("slot_read", ["bare", "residual_block"])
def test_the_replay_matches_the_generator_under_bf16_autocast_too(slot_read: str) -> None:
    # The trainer runs the pair pass under bf16 autocast, and the generator
    # promotes both operands before forming the head features. A replay that adds
    # in bf16 and promotes afterwards measures a rounding grid the generator does
    # not use, so the decomposition has to hold in that regime as well.
    generator = _generator(slot_read=slot_read)
    states_u, lengths_u = _states(seed=1)
    states_v, lengths_v = _states(seed=2)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stages = decompose(generator, states_u, states_v, lengths_u, lengths_v)
        reference = generator(states_u, states_v, lengths_u, lengths_v)
    assert float((stages.weights - reference).abs().max()) < 1e-5
    # The head features must be the promoted sum and difference, not a bf16 sum
    # widened afterwards -- which differs by a whole bf16 ulp of the slot states.
    final = stages.slots[-1]
    rows = torch.as_tensor([i for i, _ in EDGE_ENDPOINTS], dtype=torch.long)
    cols = torch.as_tensor([j for _, j in EDGE_ENDPOINTS], dtype=torch.long)
    h_i, h_j = final[:, rows].float(), final[:, cols].float()
    expected = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
    assert stages.features.dtype == torch.float32
    torch.testing.assert_close(stages.features, expected, rtol=0.0, atol=0.0)


def test_stage_groups_cover_the_chain_with_the_right_slot_counts() -> None:
    generator = _generator()
    states_u, lengths_u = _states(seed=1)
    states_v, lengths_v = _states(seed=2)
    with torch.no_grad():
        stages = decompose(generator, states_u, states_v, lengths_u, lengths_v)
    groups = stage_groups(stages)
    assert set(groups) == {"reads", "mix", "mp1", "mp2", "head_features", "logits", "weights"}
    assert groups["mix"]["C"].shape == (5, 8, 96)
    assert groups["mp2"]["L"].shape == (5, 8, 96)
    assert groups["head_features"]["closure"].shape == (5, 16, 192)
    assert groups["weights"]["interior"].shape == (5, 64, 1)
    assert stages.weights.shape == (5, 96)


def test_replay_is_independent_of_the_chunk_padding_width() -> None:
    generator = _generator()
    torch.manual_seed(11)
    rows = [(torch.randn(7, 32), torch.randn(9, 32)), (torch.randn(11, 32), torch.randn(5, 32))]
    cached = CachedInputs(
        states_u=[row[0] for row in rows],
        states_v=[row[1] for row in rows],
        lengths_u=torch.tensor([row[0].size(0) for row in rows]),
        lengths_v=torch.tensor([row[1].size(0) for row in rows]),
        dtype="torch.float32",
    )
    device = torch.device("cpu")
    with torch.no_grad():
        alone = generator(*cached.batch([0], device))
        together = generator(*cached.batch([0, 1], device))
    assert torch.allclose(alone[0], together[0], atol=1e-5)
    assert cached.rows() == 2
    assert cached.megabytes() > 0.0


def test_short_enough_keeps_only_pairs_under_the_cap() -> None:
    lengths = {"a": 100, "b": 500, "c": 900}
    pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("a", "missing")]
    mask = _short_enough(pairs, lengths, 512)
    assert mask.tolist() == [True, False, False, False]
    assert _short_enough(pairs, lengths, 0).tolist() == [True] * 4


# ---------------------------------------------------------------------------
# The attention measurement follows the read the generator actually performs
# ---------------------------------------------------------------------------


def _measure(generator: MotifGenerator, *, through_block: bool) -> dict[str, float]:
    """Measure the witness read, with or without the residual block's norms."""
    states, lengths = _states()
    pad = torch.arange(states.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
    projected = generator.residue_proj(states)
    block = generator.witness_read if through_block else None
    measured = attention_side(generator, states, projected, pad, generator.witness_queries, block)
    return {key: float(np.mean(series)) for key, series in measured.items()}


def test_the_bare_read_is_measured_without_any_normalisation() -> None:
    generator = _generator(slot_read="bare")
    assert generator.witness_read is None
    # No block to pass: the two calls are the same measurement.
    assert _measure(generator, through_block=False) == _measure(generator, through_block=True)


def test_the_residual_read_is_measured_through_its_own_norms() -> None:
    generator = _generator(slot_read="residual_block")
    assert generator.witness_read is not None
    # `norm_h` rescales every residue state to unit RMS, so the keys the real
    # attention sees have a different variance from the raw projection's.
    through = _measure(generator, through_block=True)
    bare = _measure(generator, through_block=False)
    assert through["K"] != pytest.approx(bare["K"], rel=1e-3)
    assert through["entropy_ratio"] != pytest.approx(bare["entropy_ratio"], rel=1e-4)
    # `proj` is the raw projection either way: it is the denominator of the
    # reported K/proj and V/proj ratios.
    assert through["proj"] == pytest.approx(bare["proj"], rel=1e-6)
    assert through["H"] == pytest.approx(bare["H"], rel=1e-6)


def test_an_un_normalised_value_path_is_measured_on_the_raw_states() -> None:
    normed = _generator(slot_read="residual_block", slot_read_value_norm=True)
    raw_values = _generator(slot_read="residual_block", slot_read_value_norm=False)
    # Same seed, so the two generators differ only in the value path.
    assert _measure(normed, through_block=True)["K"] == pytest.approx(
        _measure(raw_values, through_block=True)["K"], rel=1e-6
    )
    assert _measure(normed, through_block=True)["V"] != pytest.approx(
        _measure(raw_values, through_block=True)["V"], rel=1e-3
    )
    # The un-normalised values are the bare read's values.
    assert _measure(raw_values, through_block=True)["V"] == pytest.approx(
        _measure(raw_values, through_block=False)["V"], rel=1e-6
    )
