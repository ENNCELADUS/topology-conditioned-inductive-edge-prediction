"""Scorer paths for the v3_1_coord_gen family: builder, packed parity, interventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src import score_universe
from src.data.distributed_pairs import CompactPairBatch
from src.data.features import FeatureStore
from src.data.packed_features import PackedFeatureTable
from src.data.struct_coords import COORD_DIM
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen

from tests.test_prefix_model import _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture


def _reader_config() -> dict[str, object]:
    return {
        "base": _tiny_base_config(),
        "topo_prompt": {
            "trainable": "all",
            "width": 8,
            "slots_per_field": 1,
            "field_mask_prob": 0.0,
        },
    }


def _tiny_coord_gen() -> V3_1CoordGen:
    torch.manual_seed(0)
    model = V3_1CoordGen(
        reader=_reader_config(), coord_gen={"hidden": 16, "layers": 1, "dropout": 0.0}
    )
    model.reader.generator.set_coord_stats(torch.zeros(COORD_DIM), torch.ones(COORD_DIM), 5)
    with torch.no_grad():
        model.reader.generator.gates.fill_(0.4)
    model.eval()
    return model


def test_model_builder_round_trips_the_checkpoint_config_and_statistics() -> None:
    model = _tiny_coord_gen()
    rebuilt = score_universe.MODEL_BUILDERS["v3_1_coord_gen"](
        {"reader": _reader_config(), "coord_gen": model.cfg.to_dict()}
    )
    rebuilt.load_state_dict(model.state_dict())
    assert isinstance(rebuilt, V3_1CoordGen)
    assert float(rebuilt.reader.generator.coord_count) == 5.0
    assert all(not param.requires_grad for param in rebuilt.reader.parameters())


def test_packed_and_unpacked_scoring_need_no_coordinates_and_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _tiny_coord_gen()
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=5)
    store = FeatureStore(tmp_path / "features")
    device = torch.device("cpu")
    packed = score_universe._score_v3_1_packed(
        model, pairs, pack_root, device=device, amp="off", token_budget=512
    )
    unpacked = score_universe._score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=512
    )
    table = PackedFeatureTable.from_pack(pack_root, device)
    node_index = table.manifest.node_index()
    compact = CompactPairBatch(
        row_ids=torch.arange(len(pairs)),
        node_a=torch.tensor([node_index[u] for u, _ in pairs]),
        node_b=torch.tensor([node_index[v] for _, v in pairs]),
        labels=torch.zeros(len(pairs)),
        bucket_boundary=128,
        global_pair_count=len(pairs),
    )
    packed_batch = table.assemble(compact)
    packed_batch["emb_a"] = packed_batch["emb_a"].float()
    packed_batch["emb_b"] = packed_batch["emb_b"].float()
    with torch.inference_mode():
        reference = model(packed_batch)["logits"].numpy().reshape(-1)
    np.testing.assert_allclose(packed, reference, rtol=0.0, atol=1e-5)
    # The pack stores bf16 tokens, so the two paths agree only loosely.
    np.testing.assert_allclose(packed, unpacked, rtol=0.0, atol=5e-3)

    base = V3_1(**_tiny_base_config())
    base.load_state_dict(model.reader.base.state_dict())
    base.eval()
    base_scores = score_universe._score_v3_1(
        base, pairs, store, device=device, amp="off", token_budget=512
    )
    assert not np.allclose(unpacked, base_scores)
    model.intervention = "gates_off"
    gated_off = score_universe._score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=512
    )
    np.testing.assert_allclose(gated_off, base_scores, rtol=0.0, atol=1e-5)
    model.intervention = "none"
    with pytest.raises(SystemExit, match="v3_1_prefix"):
        score_universe._score_v3_1(
            model,
            pairs,
            store,
            device=device,
            amp="off",
            token_budget=512,
            shuffle_sources=list(pairs),
        )
