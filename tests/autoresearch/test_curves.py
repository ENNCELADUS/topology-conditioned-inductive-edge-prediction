import csv

import pytest
from src.autoresearch.curves import CSV_COLUMNS, write_curves
from src.autoresearch.curves import main as curves_main

from tests.autoresearch.conftest import RunDirFactory, make_metric_row

pytestmark = pytest.mark.unit


def test_write_curves_emits_csv_in_column_order(
    make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_dir = tmp_path_factory.mktemp("curves")
    run_dir = make_run_dir(epochs=2, selected_epoch=1)
    csv_path, png_path = write_curves(run_dir, out_dir)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert float(rows[1]["val_gs_bfs"]) == pytest.approx(0.52)
    assert float(rows[0]["train_kd_loss"]) == pytest.approx(0.5)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_missing_keys_become_empty_cells(
    make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_dir = tmp_path_factory.mktemp("curves-control")
    row = make_metric_row(1)
    del row["train_kd_loss"]
    del row["grad_norm_kd"]
    run_dir = make_run_dir(rows=[row], selected_epoch=1)
    csv_path, _ = write_curves(run_dir, out_dir)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["train_kd_loss"] == ""
    assert parsed[0]["grad_norm_kd"] == ""


def test_curves_cli_writes_both_artifacts(
    make_run_dir: RunDirFactory, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_dir = tmp_path_factory.mktemp("curves-cli")
    run_dir = make_run_dir(epochs=2, selected_epoch=2)
    assert curves_main([str(run_dir), str(out_dir)]) == 0
    assert (out_dir / "learning_curves.csv").exists()
    assert (out_dir / "learning_curves.png").exists()
