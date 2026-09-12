"""Scorer paths for the v3_1_topo_prompt family: coordinates, shuffle, packed parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from src import score_universe
from src.data.distributed_pairs import CompactPairBatch
from src.data.features import FeatureStore
from src.data.packed_features import PackedFeatureTable
from src.data.struct_coords import COORD_DIM
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt

from tests.test_prefix_model import _tiny_base_config
from tests.test_score_universe import _build_prefix_packed_fixture


def _tiny_topo_prompt(*, gates: float = 0.4) -> V3_1TopoPrompt:
    torch.manual_seed(0)
    model = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"trainable": "all", "width": 8, "slots_per_field": 1, "field_mask_prob": 0.0},
    )
    model.generator.set_coord_stats(torch.zeros(COORD_DIM), torch.ones(COORD_DIM), 5)
    with torch.no_grad():
        model.generator.gates.fill_(gates)
    model.eval()
    return model


def _coords(n: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(n, COORD_DIM, generator=torch.Generator().manual_seed(seed)) * 2.0


def test_shuffle_source_rows_is_a_seeded_permutation_of_the_whole_universe() -> None:
    first = score_universe._shuffle_source_rows(6, 0)
    assert first.dtype == np.int64 and sorted(first.tolist()) == list(range(6))
    np.testing.assert_array_equal(first, score_universe._shuffle_source_rows(6, 0))
    assert not np.array_equal(first, score_universe._shuffle_source_rows(6, 3))
    with pytest.raises(SystemExit, match="at least 2"):
        score_universe._shuffle_source_rows(1, 0)
    score_universe._check_row_coords(None, 7)
    score_universe._check_row_coords(_coords(6), 6)
    with pytest.raises(SystemExit, match="cover 6 rows, expected 7"):
        score_universe._check_row_coords(_coords(6), 7)


def test_unpacked_and_packed_scoring_agree_and_respond_to_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _tiny_topo_prompt()
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=5)
    store = FeatureStore(tmp_path / "features")
    coords = _coords(len(pairs))
    device = torch.device("cpu")

    def unpacked(row_coords: torch.Tensor) -> NDArray[np.float32]:
        return score_universe._score_v3_1(
            model, pairs, store, device=device, amp="off", token_budget=512, row_coords=row_coords
        )

    def packed(row_coords: torch.Tensor) -> NDArray[np.float32]:
        return score_universe._score_v3_1_packed(
            model,
            pairs,
            pack_root,
            device=device,
            amp="off",
            token_budget=512,
            row_coords=row_coords,
        )

    baseline = unpacked(coords)
    # The pack stores bf16 tokens, so packed scoring is compared against a forward
    # over the packed table's own assembled batch, as the prefix-arm test does.
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
        packed_reference = model({**packed_batch, "struct_coords": coords})["logits"]
        # `shuffle` reaches this family as substituted coordinates: the scorer
        # measures each row's universe-level source pair and passes the result
        # as `row_coords`, so the scoring functions see plain row-aligned banks.
        sources = torch.from_numpy(score_universe._shuffle_source_rows(len(pairs), 0))
        packed_shuffled_reference = model(
            {**packed_batch, "struct_coords": coords.index_select(0, sources)}
        )["logits"]
    np.testing.assert_allclose(
        packed(coords), packed_reference.numpy().reshape(-1), rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        packed(coords.index_select(0, sources)),
        packed_shuffled_reference.numpy().reshape(-1),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(packed(coords), baseline, rtol=0.0, atol=5e-3)
    other_coords = score_universe._score_v3_1(
        model,
        pairs,
        store,
        device=device,
        amp="off",
        token_budget=512,
        row_coords=_coords(len(pairs), seed=9),
    )
    assert not np.allclose(baseline, other_coords)

    shuffled = unpacked(coords.index_select(0, sources))
    assert not np.allclose(baseline, shuffled)
    np.testing.assert_array_equal(shuffled, unpacked(coords.index_select(0, sources)))
    assert model.intervention == "none"
    # The prefix arm's source-condition path is not this family's shuffle.
    with pytest.raises(SystemExit, match="v3_1_prefix"):
        score_universe._score_v3_1(
            model,
            pairs,
            store,
            device=device,
            amp="off",
            token_budget=512,
            shuffle_sources=list(pairs),
            row_coords=coords,
        )

    # gates_off is the base function on the same (unpacked) path.
    base = V3_1(**_tiny_base_config())
    base.load_state_dict(model.base.state_dict())
    base.eval()
    reference = score_universe._score_v3_1(
        base, pairs, store, device=device, amp="off", token_budget=512
    )
    model.intervention = "gates_off"
    np.testing.assert_allclose(unpacked(coords), reference, rtol=0.0, atol=1e-5)
    model.intervention = "none"


def test_packed_scoring_requires_coordinates_for_the_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _tiny_topo_prompt()
    pack_root, pairs = _build_prefix_packed_fixture(tmp_path, monkeypatch, node_count=3)
    with pytest.raises(ValueError, match="needs row_coords"):
        score_universe._score_v3_1_packed(
            model, pairs, pack_root, device=torch.device("cpu"), amp="off", token_budget=512
        )


def test_model_builder_round_trips_the_checkpoint_config_and_statistics() -> None:
    model = _tiny_topo_prompt()
    with torch.no_grad():
        model.generator.coord_mean.fill_(0.25)
    rebuilt = score_universe.MODEL_BUILDERS["v3_1_topo_prompt"](
        {"base": _tiny_base_config(), "topo_prompt": model.cfg.to_dict()}
    )
    rebuilt.load_state_dict(model.state_dict())
    assert isinstance(rebuilt, V3_1TopoPrompt)
    torch.testing.assert_close(rebuilt.generator.coord_mean, model.generator.coord_mean)
    assert float(rebuilt.generator.coord_count) == 5.0
