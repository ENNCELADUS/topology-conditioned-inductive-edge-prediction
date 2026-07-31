"""Regression tests for removal of qualification orchestration."""

from pathlib import Path

import pytest
from src.e2_pipeline import parse_pipeline_args

pytestmark = pytest.mark.unit


def test_pipeline_cli_rejects_qualification_run_kind(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pipeline_args(
            ["--config", str(tmp_path / "config.yaml"), "--run-kind", "qualification"]
        )


@pytest.mark.parametrize("flag", ["--qualification-artifact", "--epochs"])
def test_pipeline_cli_rejects_retired_qualification_flags(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_pipeline_args(["--config", str(tmp_path / "config.yaml"), flag, "value"])
