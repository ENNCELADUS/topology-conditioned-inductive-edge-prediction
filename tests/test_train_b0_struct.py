from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.train_b0 import config_to_dict, load_config


def _control_yaml() -> dict[str, object]:
    raw = yaml.safe_load(Path("configs/b1_kd_control_breadth_first.yaml").read_text())
    return dict(raw)


def test_struct_block_parses_and_serialises(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg.struct is not None
    assert cfg.struct.arm == "bce+degree+motif+rank"
    payload = config_to_dict(cfg)
    assert payload["struct"]["weights"]["rank"] == 1.0
    assert payload["struct"]["mix"] == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}


def test_absent_struct_block_is_none(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_control_yaml()))
    assert load_config(path).struct is None


def test_struct_block_rejects_unknown_key(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0}, "gradnorm": True}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown struct config keys"):
        load_config(path)
