import json

import pytest
from src.autoresearch.baseline import main as baseline_main

from tests.autoresearch.conftest import RunDirFactory, make_cadence_rows

pytestmark = pytest.mark.unit


def test_baseline_cli_prints_reselected_surface(
    make_run_dir: RunDirFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = make_run_dir(rows=make_cadence_rows(), selected_epoch=3)
    assert baseline_main([str(run_dir), "--topology-every", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_epoch"] == 4
    assert payload["auprc"] == pytest.approx(0.90)
    assert payload["run_dir"] == str(run_dir)


def test_baseline_cli_defaults_to_published_selection(
    make_run_dir: RunDirFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = make_run_dir(rows=make_cadence_rows(), selected_epoch=3)
    assert baseline_main([str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["selected_epoch"] == 3
