"""Feature caching is attribute-only and independent of scoring batch composition."""

from pathlib import Path
from typing import Any

import pytest
import torch
from src.baselines.l3ppi import build_l3ppi
from src.baselines.l3ppi_features import encode_nodes, score_cached
from src.data.packed_features import (
    PackedFeatureManifest,
    PackedNodeRecord,
    PackedShardRecord,
    write_packed_manifest,
)
from src.train_l3ppi import load_features

from tests.test_l3ppi_model import config


def test_real_pack_pooling_and_cache_identity(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = build_l3ppi(config()).eval()
    pack = tmp_path / "pack"
    pack.mkdir()
    tokens = torch.randn(5, 6).bfloat16()
    raw = tokens.view(torch.uint8).numpy().tobytes()
    (pack / "shard.bin").write_bytes(raw)
    write_packed_manifest(
        pack,
        PackedFeatureManifest(
            "bf16_flat_shards_v1",
            6,
            "bfloat16",
            "unused",
            "unused",
            (PackedNodeRecord("a", 0, 0, 0, 2), PackedNodeRecord("b", 0, 2, 2, 3)),
            (PackedShardRecord("shard.bin", 5, len(raw), "unused"),),
            1,
            0.0,
        ),
    )
    device = torch.device("cpu")
    features = encode_nodes(model, pack, ["b", "a", "b"], device, 2)
    assert set(features) == {"a", "b"}
    torch.testing.assert_close(features["a"], model.encode(tokens[:2][None], torch.tensor([2]))[0])
    torch.testing.assert_close(features["b"], model.encode(tokens[2:][None], torch.tensor([3]))[0])
    cfg: dict[str, Any] = {
        "feature_cache": str(tmp_path / "cache.pt"),
        "pack_dir": str(pack),
        "encode_batch_size": 2,
    }
    load_features(model, cfg, ["a", "b"], "encoder1", 0, 1, device)
    # Reusing the cache needs neither the source pack nor any graph.
    cfg["pack_dir"] = "does-not-exist"
    cached = load_features(model, cfg, ["a", "b"], "encoder1", 0, 1, device)
    torch.testing.assert_close(cached["a"], features["a"])
    with pytest.raises(ValueError, match="different encoder"):
        load_features(model, cfg, ["a", "b"], "encoder2", 0, 1, device)
    pairs = [("a", "b"), ("b", "a"), ("a", "a")]
    scores = score_cached(model, cached, pairs, device, 8192)[0]
    individual = score_cached(model, cached, pairs, device, 1)[0]
    torch.testing.assert_close(torch.from_numpy(scores), torch.from_numpy(individual))
    assert scores[0] == pytest.approx(scores[1])
