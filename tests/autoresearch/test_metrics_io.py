from pathlib import Path

import pytest
from src.autoresearch.metrics_io import RunFailure, read_metric_rows, read_run

from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def test_read_run_returns_selected_epoch_surface(make_run_dir: RunDirFactory) -> None:
    run = read_run(make_run_dir(selected_epoch=2))
    assert run.selected_epoch == 2
    assert run.auprc == pytest.approx(0.82)
    assert run.topology.gs == pytest.approx(0.52)
    assert run.topology.rd == pytest.approx(1.10)
    assert run.topology.degree_mmd == pytest.approx(0.90)
    assert run.threshold == pytest.approx(2.5)
    assert run.total_seconds == pytest.approx(123.0)


def test_read_run_raises_on_failure_marker(make_run_dir: RunDirFactory) -> None:
    run_dir = make_run_dir(failure={"stage": "train", "message": "boom"})
    with pytest.raises(RunFailure, match="boom"):
        read_run(run_dir)


def test_read_run_rejects_non_finite_metric(make_run_dir: RunDirFactory) -> None:
    rows = [make_metric_row(1), make_metric_row(2, val_gs_bfs=float("nan"))]
    with pytest.raises(ValueError, match="val_gs_bfs"):
        read_run(make_run_dir(rows=rows, selected_epoch=2))


def test_read_run_rejects_non_finite_total_seconds(make_run_dir: RunDirFactory) -> None:
    with pytest.raises(ValueError, match="total_seconds"):
        read_run(make_run_dir(total_seconds=float("inf")))


def test_read_run_rejects_non_positive_rd(make_run_dir: RunDirFactory) -> None:
    rows = [make_metric_row(1, val_rd_bfs=0.0)]
    with pytest.raises(ValueError, match="val_rd_bfs"):
        read_run(make_run_dir(rows=rows, selected_epoch=1))


def test_read_run_rejects_missing_selected_epoch_row(make_run_dir: RunDirFactory) -> None:
    with pytest.raises(ValueError, match="selected epoch 9"):
        read_run(make_run_dir(selected_epoch=9))


def test_read_metric_rows_rejects_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        read_metric_rows(path)
