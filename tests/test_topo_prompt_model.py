"""Topology-prompt model tests: null identity, swap symmetry, freezing, interventions."""

from __future__ import annotations

import pytest
import torch
from src.data.struct_coords import COORD_DIM, FIELD_SLICES
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.topo_prompt import (
    TopoPromptConfig,
    TopoPromptGenerator,
    V3_1TopoPrompt,
)

from tests.test_prefix_model import _pair_batch, _tiny_base_config


def _coords(n: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, COORD_DIM, generator=gen) * 3.0


def _swap(coords: torch.Tensor) -> torch.Tensor:
    out = coords.clone()
    out[:, FIELD_SLICES["endpoint_u"]] = coords[:, FIELD_SLICES["endpoint_v"]]
    out[:, FIELD_SLICES["endpoint_v"]] = coords[:, FIELD_SLICES["endpoint_u"]]
    return out


def _model(trainable: str, seed: int = 0, **extra: object) -> V3_1TopoPrompt:
    torch.manual_seed(seed)
    block: dict[str, object] = {"trainable": trainable, "width": 16, "slots_per_field": 2}
    if trainable == "prompt":
        block["base_checkpoint"] = "unused.pt"
    block.update(extra)
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt=block)
    model.generator.set_coord_stats(
        torch.full((COORD_DIM,), 1.5), torch.full((COORD_DIM,), 0.9), 10
    )
    return model


def _open_gates(model: V3_1TopoPrompt, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.generator.gates.copy_(torch.randn(model.generator.gates.shape, generator=gen))


def test_config_validation_and_round_trip() -> None:
    cfg = TopoPromptConfig.from_mapping({"trainable": "all", "width": 32})
    assert TopoPromptConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown topo_prompt keys"):
        TopoPromptConfig.from_mapping({"tokens": 4})
    with pytest.raises(ValueError, match="requires topo_prompt.base_checkpoint"):
        TopoPromptConfig(trainable="prompt")
    with pytest.raises(ValueError, match="trainable"):
        TopoPromptConfig(trainable="half")
    with pytest.raises(ValueError, match="field_mask_prob"):
        TopoPromptConfig(field_mask_prob=1.0)
    with pytest.raises(ValueError, match="coord_spec"):
        TopoPromptConfig(coord_spec="v0")


def test_wrapper_rejects_a_base_without_cross_attention_layers() -> None:
    with pytest.raises(ValueError, match="bidirectional_cross"):
        V3_1TopoPrompt(base=_tiny_base_config("none"), topo_prompt={"trainable": "all"})


@pytest.mark.parametrize("trainable", ["prompt", "all"])
def test_zero_gates_reproduce_the_base_in_eval_mode(trainable: str) -> None:
    model = _model(trainable)
    base = V3_1(**_tiny_base_config())
    base.load_state_dict(model.base.state_dict())
    base.eval()
    model.eval()
    batch = _pair_batch()
    if trainable == "prompt":
        # The deployed frozen base: PyTorch's attention kernel path depends on
        # requires_grad, so bitwise identity is asserted against that state.
        for param in base.parameters():
            param.requires_grad_(False)
    with torch.no_grad():
        expected = base(batch)["logits"]
        observed = model({**batch, "struct_coords": _coords(6)})["logits"]
    if trainable == "prompt":
        assert torch.equal(observed, expected)
    else:
        torch.testing.assert_close(observed, expected, rtol=0, atol=1e-6)


def test_forward_requires_coordinates_and_set_statistics() -> None:
    model = _model("all")
    with pytest.raises(ValueError, match="struct_coords"):
        model(_pair_batch())
    fresh = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all"})
    with pytest.raises(ValueError, match="never set"):
        fresh({**_pair_batch(), "struct_coords": _coords(6)})
    with pytest.raises(ValueError, match="shape"):
        model({**_pair_batch(), "struct_coords": torch.zeros(6, 3)})


def test_swapping_the_pair_and_its_coordinates_leaves_logits_unchanged() -> None:
    model = _model("all")
    _open_gates(model)
    model.eval()
    batch = _pair_batch()
    coords = _coords(6)
    swapped = {
        "emb_a": batch["emb_b"],
        "emb_b": batch["emb_a"],
        "len_a": batch["len_b"],
        "len_b": batch["len_a"],
        "struct_coords": _swap(coords),
    }
    with torch.no_grad():
        forward = model({**batch, "struct_coords": coords})["logits"]
        backward = model(swapped)["logits"]
    torch.testing.assert_close(forward, backward, rtol=0, atol=1e-6)


def test_open_gates_change_logits_and_respond_to_coordinates() -> None:
    model = _model("all")
    _open_gates(model)
    model.eval()
    batch = _pair_batch()
    with torch.no_grad():
        a = model({**batch, "struct_coords": _coords(6, seed=0)})["logits"]
        b = model({**batch, "struct_coords": _coords(6, seed=1)})["logits"]
        model.intervention = "gates_off"
        off = model({**batch, "struct_coords": _coords(6, seed=0)})["logits"]
        model.intervention = "none"
    assert not torch.allclose(a, b)
    base = V3_1(**_tiny_base_config())
    base.load_state_dict(model.base.state_dict())
    base.eval()
    with torch.no_grad():
        torch.testing.assert_close(off, base(batch)["logits"], rtol=0, atol=1e-6)


def test_prompt_mode_freezes_the_base_and_all_mode_trains_it() -> None:
    frozen = _model("prompt")
    frozen.train()
    assert not frozen.base.training
    out = frozen({**_pair_batch(), "struct_coords": _coords(6)})
    out["loss"].backward()
    assert all(param.grad is None for param in frozen.base.parameters())
    assert frozen.generator.gates.grad is not None
    assert len(frozen.trainable_parameters()) == len(list(frozen.generator.parameters()))

    full = _model("all")
    full.train()
    assert full.base.training
    out = full({**_pair_batch(), "struct_coords": _coords(6)})
    out["loss"].backward()
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in full.base.parameters()
    )
    assert len(full.trainable_parameters()) == len(list(full.parameters()))


def test_interventions_mask_only_their_fields_and_are_scoring_only() -> None:
    model = _model("all")
    _open_gates(model)
    model.train()
    model.intervention = "mean"
    with pytest.raises(ValueError, match="scoring-time only"):
        model({**_pair_batch(), "struct_coords": _coords(6)})
    model.eval()
    z = model.generator.standardize(_coords(6))
    for name, fields in (
        ("mean", ("endpoint_u", "endpoint_v", "relation", "context")),
        ("mean_endpoint", ("endpoint_u", "endpoint_v")),
        ("mean_relation", ("relation",)),
        ("mean_context", ("context",)),
    ):
        model.intervention = name
        masked, scale = model._apply_intervention(z)  # noqa: SLF001
        assert scale == 1.0
        for field, sl in FIELD_SLICES.items():
            if field in fields:
                assert torch.equal(masked[:, sl], torch.zeros_like(masked[:, sl]))
            else:
                assert torch.equal(masked[:, sl], z[:, sl])
    model.intervention = "bogus"
    with pytest.raises(ValueError, match="unknown topo_prompt intervention"):
        model({**_pair_batch(), "struct_coords": _coords(6)})


def test_field_masking_only_in_training_and_with_positive_probability() -> None:
    generator = TopoPromptGenerator(8, 2, 2, TopoPromptConfig(field_mask_prob=0.999, width=8))
    generator.set_coord_stats(torch.zeros(COORD_DIM), torch.ones(COORD_DIM), 1)
    z = torch.ones(64, COORD_DIM)
    generator.eval()
    assert torch.equal(generator.training_field_mask(z), z)
    generator.train()
    torch.manual_seed(0)
    masked = generator.training_field_mask(z)
    assert (masked == 0).float().mean() > 0.9
    quiet = TopoPromptGenerator(8, 2, 2, TopoPromptConfig(field_mask_prob=0.0, width=8))
    quiet.train()
    assert torch.equal(quiet.training_field_mask(z), z)


def test_state_dict_round_trip_keeps_statistics_and_has_no_duplicate_keys() -> None:
    model = _model("all")
    keys = list(model.state_dict())
    assert len(keys) == len(set(keys))
    assert not any(key.startswith("prompt_layers.") for key in keys)
    clone = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 16})
    clone.load_state_dict(model.state_dict())
    assert float(clone.generator.coord_count) == 10.0
    torch.testing.assert_close(clone.generator.coord_std, model.generator.coord_std)
    with pytest.raises(ValueError, match="positive row count"):
        clone.generator.set_coord_stats(torch.zeros(COORD_DIM), torch.ones(COORD_DIM), 0)
    with pytest.raises(ValueError, match="positive scale"):
        clone.generator.set_coord_stats(torch.zeros(COORD_DIM), torch.zeros(COORD_DIM), 3)
