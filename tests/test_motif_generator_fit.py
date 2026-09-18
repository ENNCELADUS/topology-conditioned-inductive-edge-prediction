"""Units of the fixed-set generator-only fit harness."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.experiments.motif_generator_fit import (
    GROUPS,
    STRATUM_NAMES,
    RowInputs,
    allocate_strata,
    assert_replay_matches,
    build_generator,
    build_parser,
    collate,
    dispersion_chain,
    identical_closure_fraction,
    replay,
    slot_dispersion,
    strata_of,
    stratified_draw,
    wedge_mass_auroc,
    witness_counts,
)
from src.model.egostitch.classifier.motif_prompt import MotifPromptConfig


def _cfg(**extra: object) -> MotifPromptConfig:
    return MotifPromptConfig.from_mapping(
        {"stage": "two", "base_checkpoint": "b.pt", "bundle_checkpoint": "s.pt", **extra}
    )


def _corpus(counts: list[int]) -> torch.Tensor:
    """A compiled table whose rows carry exactly ``counts[i]`` witnesses."""
    table = torch.zeros(len(counts), 96)
    for row, count in enumerate(counts):
        table[row, :count] = 0.5
        table[row, 8 : 8 + count] = 0.5
        table[row, 16:24] = 0.3
    return table


def test_witness_counts_and_strata_follow_the_closure_block() -> None:
    table = _corpus([0, 1, 2, 3, 7, 8])
    np.testing.assert_array_equal(witness_counts(table), [0, 1, 2, 3, 7, 8])
    np.testing.assert_array_equal(strata_of(witness_counts(table)), [0, 1, 1, 2, 2, 3])


def test_allocate_strata_hits_the_target_composition_when_supply_allows() -> None:
    assert allocate_strata([1000, 1000, 1000, 1000], 100) == [40, 20, 20, 20]
    assert sum(allocate_strata([1000, 1000, 1000, 1000], 97)) == 97


def test_allocate_strata_redistributes_a_shortfall_and_never_oversubscribes() -> None:
    want = allocate_strata([1000, 1000, 5, 0], 100)
    assert want[2] == 5 and want[3] == 0
    assert sum(want) == 100
    assert want[0] >= 40 and want[1] >= 20
    short = allocate_strata([10, 5, 2, 1], 100)
    assert short == [10, 5, 2, 1]


def test_allocate_strata_rejects_a_malformed_request() -> None:
    with pytest.raises(ValueError, match="expected 4 strata"):
        allocate_strata([1, 2, 3], 10)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_strata([1, 2, 3, 4], -1)


def test_stratified_draw_reaches_the_forty_sixty_split_on_a_rich_pool() -> None:
    pool = _corpus([0] * 400 + [1] * 200 + [4] * 200 + [8] * 200)
    indices, composition = stratified_draw(pool, total=100, seed=3)
    assert indices.size == 100
    assert composition == {"0": 40, "1-2": 20, "3-7": 20, "8": 20}
    assert sorted(set(indices.tolist())) == sorted(indices.tolist())
    again, _ = stratified_draw(pool, total=100, seed=3)
    np.testing.assert_array_equal(indices, again)


def test_stratified_draw_records_what_a_thin_pool_actually_supplied() -> None:
    pool = _corpus([0] * 500 + [8] * 3)
    _, composition = stratified_draw(pool, total=100, seed=1)
    assert composition["8"] == 3
    assert composition["1-2"] == 0 and composition["3-7"] == 0
    assert sum(composition.values()) == 100


def _row(length: int, width: int, seed: int) -> np.ndarray:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(length, width, generator=gen)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .copy()
    )


def test_collate_pads_to_the_batch_maximum_and_keeps_the_true_lengths() -> None:
    rows = [
        RowInputs(u=_row(5, 8, 1), v=_row(3, 8, 2)),
        RowInputs(u=_row(2, 8, 3), v=_row(7, 8, 4)),
    ]
    encoded_u, encoded_v, lengths_u, lengths_v = collate(rows, torch.device("cpu"))
    assert encoded_u.shape == (2, 5, 8) and encoded_v.shape == (2, 7, 8)
    assert encoded_u.dtype == torch.float32
    assert lengths_u.tolist() == [5, 2] and lengths_v.tolist() == [3, 7]
    assert float(encoded_u[1, 2:].abs().max()) == 0.0
    torch.testing.assert_close(
        encoded_u[0], torch.from_numpy(rows[0].u).view(torch.bfloat16).float(), rtol=0, atol=0
    )


def test_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        collate([], torch.device("cpu"))


def test_slot_dispersion_is_zero_on_identical_slots_and_positive_otherwise() -> None:
    same = torch.ones(3, 8, 4)
    assert slot_dispersion(same) == pytest.approx(0.0, abs=1e-7)
    spread = same.clone()
    spread[:, 0] = 0.0
    assert slot_dispersion(spread) > 0.1
    scalars = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert slot_dispersion(scalars) == pytest.approx(float(scalars.std(unbiased=False)), abs=1e-6)


def test_identical_closure_fraction_counts_flat_closure_rows() -> None:
    weights = torch.zeros(4, 96)
    weights[1, :16] = 0.5
    weights[2, 0] = 0.9
    weights[3, 8] = 0.9
    assert identical_closure_fraction(weights) == pytest.approx(0.5)


def test_wedge_mass_auroc_needs_both_classes() -> None:
    target = _corpus([0, 0, 3, 5])
    good = torch.zeros(4, 96)
    good[2:, :16] = 0.6
    assert wedge_mass_auroc(good, target) == pytest.approx(1.0)
    assert wedge_mass_auroc(good, _corpus([3, 3, 3, 3])) is None


def _states(
    n: int = 3, length: int = 6, width: int = 32, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.full((n,), length, dtype=torch.long)
    lengths[0] = max(length - 2, 1)
    return (
        torch.randn(n, length, width, generator=gen),
        torch.randn(n, length, width, generator=gen),
        lengths,
        lengths.flip(0).contiguous(),
    )


@pytest.mark.parametrize("slot_read", ["bare", "residual_block"])
def test_the_step_by_step_replay_reproduces_the_generator_forward(slot_read: str) -> None:
    generator = build_generator(
        _cfg(slot_read=slot_read),
        d_model=32,
        group="baseline" if slot_read == "bare" else "residual",
        fit_target=_corpus([0, 2, 5, 8]),
        seed=5,
    ).eval()
    encoded_u, encoded_v, lengths_u, lengths_v = _states(seed=7)
    assert_replay_matches(generator, encoded_u, encoded_v, lengths_u, lengths_v)
    with torch.no_grad():
        stages = replay(generator, encoded_u, encoded_v, lengths_u, lengths_v)
    assert stages["message_2"].shape == (3, 26, 96)
    assert stages["head_features"].shape == (3, 96, 192)
    assert stages["weights"].shape == (3, 96)


def test_the_replay_refuses_a_gate_mode_it_does_not_cover() -> None:
    generator = build_generator(
        _cfg(gate_mode="per_type"),
        d_model=32,
        group="baseline",
        fit_target=_corpus([0, 2]),
        seed=1,
    )
    with pytest.raises(ValueError, match="gate_mode"):
        replay(generator, *_states(seed=1))


def test_the_dispersion_chain_covers_every_stage_and_role() -> None:
    generator = build_generator(
        _cfg(slot_read="residual_block"),
        d_model=32,
        group="residual",
        fit_target=_corpus([0, 3, 8]),
        seed=2,
    ).eval()
    with torch.no_grad():
        chain = dispersion_chain(replay(generator, *_states(seed=2)))
    assert set(chain) == {
        "read_bridge_u",
        "read_bridge_v",
        "read_witness_u",
        "read_witness_v",
        "witness_mix",
        "message_1",
        "message_2",
        "head_features",
        "logits",
        "weights",
    }
    assert set(chain["message_1"]) == {"C", "L", "R"}
    assert set(chain["logits"]) == {"closure", "attach", "interior"}
    assert all(value >= 0.0 for stage in chain.values() for value in stage.values())
    assert chain["read_witness_u"]["C"] > 0.0


def test_build_generator_applies_the_group_keys_and_the_bias_initialisation() -> None:
    target = _corpus([0, 0, 4, 8])
    for group, (slot_read, head_std) in GROUPS.items():
        generator = build_generator(_cfg(), d_model=32, group=group, fit_target=target, seed=0)
        assert generator.cfg.slot_read == slot_read
        assert generator.cfg.head_output_init_std == head_std
        assert (generator.bridge_read is not None) is (slot_read == "residual_block")
        assert set(generator.bias_init_record) == {"closure", "attach", "interior"}
    with pytest.raises(ValueError, match="group must be one of"):
        build_generator(_cfg(), d_model=32, group="nope", fit_target=target, seed=0)


def test_the_parser_defaults_match_the_pre_registered_protocol() -> None:
    args = build_parser().parse_args(
        ["--checkpoint", "c.pt", "--pack-dir", "p", "--output-dir", "o"]
    )
    assert args.group == "baseline"
    assert (args.fit_rows, args.heldout_rows, args.val_rows) == (4000, 1500, 1500)
    assert (args.steps, args.batch_rows, args.lr) == (1500, 256, 1e-4)
    assert (args.weight_decay, args.grad_clip, args.eval_every) == (1e-2, 1.0, 100)
    assert args.seed == 0
    assert sorted(GROUPS) == ["baseline", "combined", "head_gain", "residual"]
    assert STRATUM_NAMES == ("0", "1-2", "3-7", "8")
