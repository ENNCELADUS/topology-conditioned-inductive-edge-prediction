from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from src import train_b0
from src.experiments.struct_pipeline_benchmark import (
    _aggregate_rank_reports,
    _BenchmarkComplete,
    _Collector,
    _install_worker_hooks,
    _parse_args,
    _WorkerComplete,
)

from tests.test_train_b0 import (
    _batch_of,
    _make_synthetic_pair_dataset,
    _tiny_config,
    _TinyPairMLP,
)


def _rank_report(rank: int, walls: list[float]) -> dict[str, Any]:
    return {
        "rank": rank,
        "world_size": 2,
        "warmup_steps": 1,
        "timed_steps": 2,
        "steps": [
            {
                "step": index + 1,
                "started_at_unix_seconds": 100.0 + index,
                "finished_at_unix_seconds": 101.0 + index,
                "local_task_pairs": 3,
                "global_task_pairs": 6,
                "struct_pairs_contribution": float(rank + 1),
                "step_wall_seconds": wall,
                "data_wait_seconds": 0.01 * (rank + 1),
                "coordinate_prepare_seconds": 0.02 * (rank + 1),
                "coordinate_prepare_forward_seconds": 0.01 * (rank + 1),
                "coordinate_prepare_backward_seconds": 0.01 * (rank + 1),
                "coordinate_prepare_forward_calls": rank + 1,
                "coordinate_prepare_backward_calls": rank,
                "struct_forward_seconds": 0.03 * (rank + 1),
                "shared_backward_seconds": 0.04 * (rank + 1),
                "memory_allocated_gib": 10.0 + rank,
                "memory_reserved_gib": 12.0 + rank,
                "memory_current_allocated_gib": 9.0 + rank,
                "memory_current_reserved_gib": 11.0 + rank,
            }
            for index, wall in enumerate(walls)
        ],
    }


def test_aggregate_uses_slowest_rank_and_global_pair_counts() -> None:
    result = _aggregate_rank_reports([_rank_report(0, [2.0, 4.0]), _rank_report(1, [3.0, 2.0])])

    assert result["world_size"] == 2
    steps = result["steps"]
    assert isinstance(steps, list)
    assert steps[0]["slowest_rank"] == 1
    assert steps[0]["slowest_rank_step_seconds"] == 3.0
    assert steps[0]["global_struct_pairs"] == 3.0
    assert steps[0]["coordinate_prepare_forward_calls"] == 3
    assert steps[0]["coordinate_prepare_backward_calls"] == 1
    assert steps[1]["slowest_rank"] == 0
    assert result["timed_window"] == {
        "started_at_unix_seconds": 100.0,
        "finished_at_unix_seconds": 102.0,
    }
    summary = result["summary"]
    assert summary["median_slowest_rank_step_seconds"] == 3.5
    assert summary["global_task_pairs_per_second"] == pytest.approx(12.0 / 7.0)


def test_aggregate_rejects_divergent_global_pair_counts() -> None:
    reports = [_rank_report(0, [1.0, 1.0]), _rank_report(1, [1.0, 1.0])]
    reports[1]["steps"][0]["global_task_pairs"] = 7

    with pytest.raises(ValueError, match="diverged"):
        _aggregate_rank_reports(reports)


def test_cli_rejects_empty_timed_window(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--resume-attempt",
                str(tmp_path / "attempt"),
                "--output",
                str(tmp_path / "result.json"),
                "--variant",
                "full",
                "--round-index",
                "1",
                "--timed-steps",
                "0",
            ]
        )


def test_collector_excludes_warmup_and_stops_after_timed_update() -> None:
    collector = _Collector(warmup_steps=1, timed_steps=1, device=torch.device("cpu"))
    batch = {"label": torch.ones(2)}

    for _ in range(2):
        started = collector.begin_batch()
        collector.set_batch(batch, waited=0.0, world_size=2)
        assert started > 0.0
        collector.mark_update_finished()

    with pytest.raises(_BenchmarkComplete):
        collector.begin_batch()

    assert len(collector.records) == 1
    assert collector.records[0].global_task_pairs == 4


def test_hooks_bound_actual_production_loop_without_checkpoint(tmp_path: Path) -> None:
    rank_output = tmp_path / "ranks"
    restore = _install_worker_hooks(
        warmup_steps=1,
        timed_steps=2,
        rank_output_dir=rank_output,
    )
    batch = _batch_of(_make_synthetic_pair_dataset(4))
    batch["_row_id"] = torch.arange(4)
    batch["_local_pair_count"] = torch.tensor(4)
    batch["_global_pair_count"] = torch.tensor(4)
    artifact_dir = tmp_path / "attempt"
    try:
        with pytest.raises(_WorkerComplete):
            train_b0.train_ddp_loop(
                _TinyPairMLP(input_dim=4, dropout=0.0),
                lambda epoch: [batch, batch, batch, batch],
                [batch],
                _tiny_config(epochs=1),
                Accelerator(cpu=True),
                warmup_steps=0,
                artifact_dir=artifact_dir,
            )
    finally:
        restore()

    report = rank_output / "rank-0.json"
    assert report.is_file()
    assert len(report.read_text(encoding="utf-8").split('"step":')) - 1 == 2
    assert not (artifact_dir / "training_state.pt").exists()
