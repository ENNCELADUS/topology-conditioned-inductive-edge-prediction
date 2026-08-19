"""Tests for src.experiments.fixed_threshold_replay: fixed-threshold candidate replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from src.experiments import fixed_threshold_replay as ftr

from tests.experiments.test_s4_budget_assembly import _toy_inputs

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- spec parsing


def test_parse_universe_spec_splits_on_first_equals() -> None:
    arm, path = ftr.parse_universe_spec("d1=outputs/a=b/candidate.npz")
    assert arm == "d1"
    assert path == Path("outputs/a=b/candidate.npz")


@pytest.mark.parametrize("spec", ["no-equals", "=path.npz", "arm="])
def test_parse_universe_spec_rejects_malformed(spec: str) -> None:
    with pytest.raises(ValueError, match="arm=path"):
        ftr.parse_universe_spec(spec)


# --------------------------------------------------------------------------- pipeline


def _run(tmp_path: Path, *, threshold: float, output_name: str = "out.json") -> dict[str, Any]:
    universe_path, data_root = _toy_inputs(tmp_path)
    payload = ftr.run_fixed_threshold_replay(
        universes=[("toy_arm", universe_path)],
        data_root=data_root,
        strategy="toy",
        threshold=threshold,
        output_path=tmp_path / output_name,
    )
    return cast(dict[str, Any], payload)


def test_replay_at_half_recovers_the_nice_toy_graph_exactly(tmp_path: Path) -> None:
    """Toy scores: true edges 0.9, noise 0.1, n6 self-pair 0.95 -- 0.5 splits them exactly."""
    payload = _run(tmp_path, threshold=0.5)

    arm = payload["arms"]["toy_arm"]
    assert arm["predicted_edges_simple"] == 5
    assert arm["self_loops_pred"] == 1
    assert arm["self_loops_ref"] == 1
    assert arm["edge_precision"] == 1.0
    assert arm["edge_recall"] == 1.0
    assert arm["graph_similarity"]["global_simple_edge"] == pytest.approx(1.0)
    assert arm["relative_density"]["global_simple_edge"] == pytest.approx(1.0)
    assert arm["graph_similarity"]["bfs_macro"] == pytest.approx(1.0)
    for stat in ("degree", "clustering", "spectral"):
        assert arm["mmd_ratio"][stat] >= 0.0

    metadata = payload["metadata"]
    assert metadata["threshold"] == 0.5
    assert metadata["target_edges"] == 5
    assert metadata["arms"] == ["toy_arm"]
    assert metadata["artifacts"]["toy_arm"]["pairs_source"] == "candidate"


def test_replay_threshold_moves_the_operating_point(tmp_path: Path) -> None:
    """At 0.92 only the n6 self-pair (0.95) clears: no simple edges, one self-loop."""
    payload = _run(tmp_path, threshold=0.92)

    arm = payload["arms"]["toy_arm"]
    assert arm["predicted_edges_simple"] == 0
    assert arm["self_loops_pred"] == 1
    assert arm["edge_recall"] == 0.0
    assert arm["relative_density"]["global_simple_edge"] == pytest.approx(0.0)


def test_replay_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    universe_path, data_root = _toy_inputs(tmp_path)
    for name in ("a.json", "b.json"):
        ftr.run_fixed_threshold_replay(
            universes=[("toy_arm", universe_path)],
            data_root=data_root,
            strategy="toy",
            threshold=0.5,
            output_path=tmp_path / name,
        )
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_replay_rejects_duplicate_arm_names(tmp_path: Path) -> None:
    universe_path, data_root = _toy_inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicate arm names"):
        ftr.run_fixed_threshold_replay(
            universes=[("toy_arm", universe_path), ("toy_arm", universe_path)],
            data_root=data_root,
            strategy="toy",
            threshold=0.5,
            output_path=tmp_path / "out.json",
        )
