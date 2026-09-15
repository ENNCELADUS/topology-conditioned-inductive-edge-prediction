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
from src.data.struct_coords import get_coord_spec
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen

from tests.test_prefix_model import _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture


def _reader_config(spec: str = "v1") -> dict[str, object]:
    return {
        "base": _tiny_base_config(),
        "topo_prompt": {
            "trainable": "all",
            "width": 8,
            "slots_per_field": 1,
            "field_mask_prob": 0.0,
            "coord_spec": spec,
        },
    }


def _tiny_coord_gen(generator: str = "mlp") -> V3_1CoordGen:
    torch.manual_seed(0)
    spec = "v2" if generator == "virtual_graph" else "v1"
    model = V3_1CoordGen(
        reader=_reader_config(spec),
        coord_gen={
            "hidden": 16,
            "layers": 1,
            "dropout": 0.0,
            "generator": generator,
            "coord_spec": spec,
            "virtual_graph": {"k": 4, "d_z": 8, "heads": 2},
        },
    )
    dim = get_coord_spec(spec).coord_dim
    model.reader.generator.set_coord_stats(torch.zeros(dim), torch.ones(dim), 5)
    with torch.no_grad():
        model.reader.generator.gates.fill_(0.4)
    model.initialize_teacher()
    model.eval()
    return model


@pytest.mark.parametrize("generator", ["mlp", "virtual_graph"])
def test_model_builder_round_trips_the_checkpoint_config_and_statistics(generator: str) -> None:
    model = _tiny_coord_gen(generator)
    rebuilt = score_universe.MODEL_BUILDERS["v3_1_coord_gen"](
        {"reader": _reader_config(model.cfg.coord_spec), "coord_gen": model.cfg.to_dict()}
    )
    rebuilt.load_state_dict(model.state_dict())
    assert isinstance(rebuilt, V3_1CoordGen)
    assert float(rebuilt.reader.generator.coord_count) == 5.0
    assert all(not param.requires_grad for param in rebuilt.teacher.parameters())
    assert any(param.requires_grad for param in rebuilt.reader.parameters())
    for key, value in model.teacher.state_dict().items():
        assert torch.equal(value, rebuilt.teacher.state_dict()[key])


@pytest.mark.parametrize("generator", ["mlp", "virtual_graph"])
def test_packed_and_unpacked_scoring_need_no_coordinates_and_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generator: str
) -> None:
    model = _tiny_coord_gen(generator)
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


def _encoded_pairs(
    model: V3_1CoordGen,
    table: PackedFeatureTable,
    node_index: dict[str, int],
    pairs: list[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one pair list end to end through the reader's frozen encoder."""
    compact = CompactPairBatch(
        row_ids=torch.arange(len(pairs)),
        node_a=torch.tensor([node_index[u] for u, _ in pairs]),
        node_b=torch.tensor([node_index[v] for _, v in pairs]),
        labels=torch.zeros(len(pairs)),
        bucket_boundary=128,
        global_pair_count=len(pairs),
    )
    batch = table.assemble(compact)
    with torch.inference_mode():
        encoded_a = model.reader.encoder(batch["emb_a"].float(), batch["len_a"])
        encoded_b = model.reader.encoder(batch["emb_b"].float(), batch["len_b"])
    return encoded_a, encoded_b, batch["len_a"], batch["len_b"]


@pytest.mark.parametrize("generator", ["mlp", "virtual_graph"])
def test_coordinate_transplant_substitutes_the_source_pair_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generator: str
) -> None:
    """`shuffle` on a coord_gen checkpoint scores own endpoints under another row's coordinates."""
    model = _tiny_coord_gen(generator)
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=5)
    store = FeatureStore(tmp_path / "features")
    device = torch.device("cpu")
    sources = list(reversed(pairs))

    identity = score_universe._score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=512, shuffle_sources=list(pairs)
    )
    plain = score_universe._score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=512
    )
    # Transplanting each row's own prediction is the identity intervention.
    np.testing.assert_allclose(identity, plain, rtol=0.0, atol=1e-4)

    transplanted = score_universe._score_v3_1(
        model, pairs, store, device=device, amp="off", token_budget=512, shuffle_sources=sources
    )
    assert not np.allclose(transplanted, plain, atol=1e-3)

    table = PackedFeatureTable.from_pack(pack_root, device)
    node_index = table.manifest.node_index()
    own = _encoded_pairs(model, table, node_index, pairs)
    source = _encoded_pairs(model, table, node_index, sources)
    with torch.inference_mode():
        source_coords, _ = model.predict(*source)
        reference = model.reader.logits_from_standardized(*own, source_coords)
    np.testing.assert_allclose(transplanted, reference.numpy().reshape(-1), rtol=0.0, atol=5e-3)

    packed = score_universe._score_v3_1_packed(
        model,
        pairs,
        pack_root,
        device=device,
        amp="off",
        token_budget=512,
        shuffle_sources=sources,
    )
    np.testing.assert_allclose(packed, reference.numpy().reshape(-1), rtol=0.0, atol=1e-5)


def test_cli_accepts_virtual_graph_slot_gate_intervention() -> None:
    args = score_universe.build_parser().parse_args(
        [
            "score",
            "--checkpoint",
            "virtual.pt",
            "--pairs",
            "val_cls",
            "--output",
            "scores.npz",
            "--prefix-intervention",
            "slot_gates_open",
        ]
    )
    assert args.prefix_intervention == "slot_gates_open"


def test_shuffle_bank_preserves_compact_coordinates_and_row_order() -> None:
    predictions = torch.arange(44, dtype=torch.float32).reshape(4, 11)
    bank = score_universe._coord_gen_source_coords(
        [[2, 0], [3, 1]], lambda rows: predictions[rows], num_rows=4, coord_dim=11
    )
    torch.testing.assert_close(bank, predictions)


def test_slot_gate_intervention_refuses_mlp_checkpoint() -> None:
    model = _tiny_coord_gen()
    with pytest.raises(ValueError, match="virtual_graph"):
        model.intervention = "slot_gates_open"
