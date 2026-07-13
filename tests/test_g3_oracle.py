"""Tests for src.experiments.g3_oracle: the G3 Oracle gate pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from src.experiments import g3_oracle as g3
from src.experiments.g1_hardened_e2 import AssembledRow

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- rank01 / rank01_lex


class TestRank01:
    def test_distinct_values(self) -> None:
        out = g3.rank01(np.array([3.0, 1.0, 2.0]))
        np.testing.assert_allclose(out, [1.0, 0.0, 0.5])

    def test_average_ties(self) -> None:
        # ascending ranks: the two 1.0s occupy ranks 0 and 1 -> mean 0.5
        out = g3.rank01(np.array([1.0, 2.0, 1.0, 3.0]))
        np.testing.assert_allclose(out, [0.5 / 3, 2.0 / 3, 0.5 / 3, 1.0])

    def test_all_equal_is_uninformative_half(self) -> None:
        out = g3.rank01(np.array([7.0, 7.0, 7.0]))
        np.testing.assert_allclose(out, [0.5, 0.5, 0.5])

    def test_single_and_empty(self) -> None:
        np.testing.assert_allclose(g3.rank01(np.array([42.0])), [0.5])
        assert g3.rank01(np.array([], dtype=np.float64)).size == 0

    def test_idempotent_on_its_own_output(self) -> None:
        values = np.array([1.0, 5.0, 5.0, 2.0, 1.0, 9.0])
        once = g3.rank01(values)
        np.testing.assert_allclose(g3.rank01(once), once)


class TestRank01Lex:
    def test_secondary_breaks_primary_ties(self) -> None:
        primary = np.array([1.0, 1.0, 0.0])
        secondary = np.array([5.0, 9.0, 9.0])
        # keys ascending: (0,9) -> rank 0, (1,5) -> rank 1, (1,9) -> rank 2
        out = g3.rank01_lex(primary, secondary)
        np.testing.assert_allclose(out, [0.5, 1.0, 0.0])

    def test_full_ties_share_mean_rank(self) -> None:
        primary = np.array([1.0, 1.0, 0.0])
        secondary = np.array([5.0, 5.0, 9.0])
        # keys: (0,9) -> 0; (1,5) twice -> ranks 1,2 -> mean 1.5
        out = g3.rank01_lex(primary, secondary)
        np.testing.assert_allclose(out, [0.75, 0.75, 0.0])

    def test_matches_rank01_of_strict_encoding(self) -> None:
        rng = np.random.default_rng(0)
        primary = rng.integers(0, 4, size=200).astype(np.float64)
        secondary = rng.integers(0, 4, size=200).astype(np.float64)
        encoded = primary * 10.0 + secondary  # strictly monotone in the lex key
        np.testing.assert_allclose(g3.rank01_lex(primary, secondary), g3.rank01(encoded))


# --------------------------------------------------------------------------- headroom


def _assembled_row(
    mmd_ratio: dict[str, float],
    composite: float | None,
    threshold: float | None = None,
) -> AssembledRow:
    zeros = dict.fromkeys(("degree", "clustering", "spectral"), 0.0)
    return AssembledRow(
        threshold=threshold,
        mmd_ratio=mmd_ratio,
        raw_mmd2=dict(zeros),
        reference_mmd2=dict(zeros),
        relative_density=1.0,
        self_loops_pred=0,
        self_loops_ref=0,
        bootstrap_mean=dict(zeros),
        bootstrap_std=dict(zeros),
        composite=composite,
    )


class TestComputeHeadroom:
    def test_ratios_and_composite(self) -> None:
        b0 = _assembled_row({"degree": 12.0, "clustering": 9.0, "spectral": 18.0}, 1e-6)
        arm = _assembled_row({"degree": 3.0, "clustering": 4.5, "spectral": 2.0}, 1e-2)
        row = g3.compute_headroom(b0, arm)
        assert row.mmd_ratio_headroom == {"degree": 4.0, "clustering": 2.0, "spectral": 9.0}
        assert row.composite_ratio == pytest.approx(1e4)

    def test_zero_arm_ratio_yields_none(self) -> None:
        b0 = _assembled_row({"degree": 12.0, "clustering": 9.0, "spectral": 18.0}, None)
        arm = _assembled_row({"degree": 0.0, "clustering": 4.5, "spectral": 2.0}, None)
        row = g3.compute_headroom(b0, arm)
        assert row.mmd_ratio_headroom["degree"] is None
        assert row.mmd_ratio_headroom["clustering"] == 2.0
        assert row.composite_ratio is None

    def test_composite_none_when_either_missing_or_b0_zero(self) -> None:
        ratios = {"degree": 1.0, "clustering": 1.0, "spectral": 1.0}
        assert (
            g3.compute_headroom(
                _assembled_row(ratios, None), _assembled_row(ratios, 0.5)
            ).composite_ratio
            is None
        )
        assert (
            g3.compute_headroom(
                _assembled_row(ratios, 0.0), _assembled_row(ratios, 0.5)
            ).composite_ratio
            is None
        )
