from __future__ import annotations

import pytest
from src.distill.struct_config import KINDS, WEIGHT_KEYS, StructConfig


def test_defaults_match_spec() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0}})
    assert cfg.nodes == 40
    assert cfg.background_nodes == 8
    assert cfg.mix == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}
    assert cfg.subgraphs_per_epoch is None
    assert cfg.rank_margin == 0.1
    assert cfg.rank_temperature == 1.0
    assert cfg.huber_delta == 1.0
    assert cfg.mmd_sigma == 1.25
    assert cfg.mmd_bins == 48
    assert cfg.val_subgraphs == 32
    assert set(cfg.weights) == set(WEIGHT_KEYS)
    assert cfg.active_weights == {"bce": 1.0}
    assert cfg.arm == "bce"


def test_arm_name_is_sorted_nonzero_keys() -> None:
    cfg = StructConfig.from_mapping({"weights": {"bce": 1.0, "motif": 0.1, "rank": 1.0}})
    assert cfg.arm == "bce+motif+rank"
    assert cfg.active_weights == {"bce": 1.0, "motif": 0.1, "rank": 1.0}


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"weights": {}}, "at least one"),
        ({"weights": {"bce": 0.0}}, "at least one"),
        ({"weights": {"bce": -1.0}}, "non-negative"),
        ({"weights": {"bce": 1.0, "clustering": 1.0}}, "unknown struct weight"),
        ({"weights": {"bce": 1.0}, "mix": {"bfs": 1.0}}, "mix must name exactly"),
        (
            {"weights": {"bce": 1.0}, "mix": {"bfs": 0.6, "motif": 0.3, "bridge": 0.3}},
            "sum to 1",
        ),
        ({"weights": {"bce": 1.0}, "nodes": 8, "background_nodes": 8}, "background_nodes"),
        ({"weights": {"bce": 1.0}, "nodes": 3}, "nodes must be"),
        ({"weights": {"bce": 1.0}, "subgraphs_per_epoch": 0}, "subgraphs_per_epoch"),
        ({"weights": {"bce": 1.0}, "rank_temperature": 0.0}, "rank_temperature"),
        ({"weights": {"bce": 1.0}, "mmd_bins": 1}, "mmd_bins"),
        ({"weights": {"bce": 1.0}, "bogus": 1}, "unknown struct config keys"),
        ({"weights": {"bce": True}}, "must be a number"),
    ],
)
def test_rejects_illegal_blocks(mapping: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StructConfig.from_mapping(mapping)


def test_kinds_and_weight_keys_are_frozen() -> None:
    assert KINDS == ("bfs", "motif", "bridge")
    assert WEIGHT_KEYS == ("bce", "gs", "rd", "deg_mmd", "rank", "degree", "motif")
