from pathlib import Path

import pytest
from src.autoresearch.metrics_io import RunFailure, read_metric_rows, read_run

from tests.autoresearch.conftest import RunDirFactory, make_cadence_rows, make_metric_row

pytestmark = pytest.mark.unit


def test_read_run_reselects_at_campaign_cadence(make_run_dir: RunDirFactory) -> None:
    run_dir = make_run_dir(rows=make_cadence_rows(), selected_epoch=3)
    assert read_run(run_dir).selected_epoch == 3
    reselected = read_run(run_dir, topology_every=2)
    assert reselected.selected_epoch == 4
    assert reselected.auprc == pytest.approx(0.90)
    assert reselected.topology.degree_mmd == pytest.approx(0.60)


def test_read_run_cadence_one_reselects_over_all_epochs(make_run_dir: RunDirFactory) -> None:
    run_dir = make_run_dir(rows=make_cadence_rows(), selected_epoch=1)
    assert read_run(run_dir, topology_every=1).selected_epoch == 3


def test_read_run_rejects_non_positive_topology_every(make_run_dir: RunDirFactory) -> None:
    with pytest.raises(ValueError, match="topology_every must be >= 1"):
        read_run(make_run_dir(), topology_every=0)


def test_read_run_reselect_fails_on_due_epoch_without_topology(
    make_run_dir: RunDirFactory,
) -> None:
    rows = make_cadence_rows()
    del rows[1]["val_gs_bfs"]
    with pytest.raises(ValueError, match="val_gs_bfs"):
        read_run(make_run_dir(rows=rows), topology_every=2)


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


@pytest.mark.parametrize("status", ["running", None, True])
def test_read_run_requires_complete_status(make_run_dir: RunDirFactory, status: object) -> None:
    with pytest.raises(ValueError, match="status must be 'complete'"):
        read_run(make_run_dir(complete_status=status))


@pytest.mark.parametrize("selected_epoch", [0, -1, True, 2.0])
def test_read_run_requires_positive_integer_selected_epoch(
    make_run_dir: RunDirFactory, selected_epoch: object
) -> None:
    with pytest.raises(ValueError, match="selected_epoch must be a positive int"):
        read_run(make_run_dir(selected_epoch=selected_epoch))


@pytest.mark.parametrize("epoch", [True, 1.0])
def test_read_run_requires_genuine_integer_epoch_on_selected_row(
    make_run_dir: RunDirFactory, epoch: object
) -> None:
    row = make_metric_row(1)
    row["epoch"] = epoch
    with pytest.raises(ValueError, match="selected row epoch must be a positive int"):
        read_run(make_run_dir(rows=[row], selected_epoch=1))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("val_auprc", True),
        ("val_gs_bfs", "0.5"),
        ("val_rd_bfs", None),
        ("val_degree_mmd_ratio", False),
        ("val_clustering_mmd_ratio", "0.5"),
        ("val_spectral_mmd_ratio", True),
        ("val_threshold", "2.5"),
    ],
)
def test_read_run_rejects_non_numeric_surface_values(
    make_run_dir: RunDirFactory, key: str, value: object
) -> None:
    rows = [make_metric_row(1, **{key: value})]
    with pytest.raises(ValueError, match=key):
        read_run(make_run_dir(rows=rows, selected_epoch=1))


@pytest.mark.parametrize("total_seconds", [-1.0, True, "123"])
def test_read_run_requires_nonnegative_numeric_timing(
    make_run_dir: RunDirFactory, total_seconds: object
) -> None:
    with pytest.raises(ValueError, match="total_seconds"):
        read_run(make_run_dir(total_seconds=total_seconds))


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
