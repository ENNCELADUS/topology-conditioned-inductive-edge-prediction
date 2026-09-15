"""Topology-prompt model tests: null identity, swap symmetry, freezing, interventions."""

from __future__ import annotations

import pytest
import torch
from src.data.struct_coords import COORD_DIM, FIELD_SLICES, coordinate_statistics
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


def test_swap_symmetry_survives_measured_statistics_with_asymmetric_endpoint_marginals() -> None:
    # Canonically ordered training pairs give endpoint_u and endpoint_v different
    # marginals; pooled endpoint statistics keep the standardised pair swap-symmetric.
    gen = torch.Generator().manual_seed(4)
    reference = torch.rand(200, COORD_DIM, generator=gen) * 3.0
    reference[:, FIELD_SLICES["endpoint_u"]] *= 4.0
    reference[:, FIELD_SLICES["endpoint_u"]] += 2.0
    mean, std = coordinate_statistics(reference.numpy())
    model = _model("all")
    model.generator.set_coord_stats(torch.from_numpy(mean), torch.from_numpy(std), 200)
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


def test_corruption_is_seeded_swap_equivariant_and_keeps_distance_probabilities() -> None:
    from src.model.egostitch.classifier.coord_gen import DISTANCE_INDEX
    from src.model.egostitch.classifier.topo_prompt import TopoPromptConfig, TopoPromptGenerator

    cfg = TopoPromptConfig.from_mapping({"corruption": {"prob": 1.0}})
    generator = TopoPromptGenerator(8, 1, 2, cfg)
    mean = torch.zeros(COORD_DIM)
    mean[list(DISTANCE_INDEX)] = torch.tensor([0.2, 0.1, 0.2, 0.3])
    generator.set_coord_stats(mean, torch.ones(COORD_DIM), 10)
    coords = torch.randn(4, COORD_DIM)
    coords[:, list(DISTANCE_INDEX)] = 0
    coords[0, DISTANCE_INDEX[0]] = 1
    z = generator.standardize(coords)
    swapped = z.clone()
    swapped[:, :9], swapped[:, 9:18] = z[:, 9:18], z[:, :9]
    generator.train()
    observed = generator.training_corruption(z, seed=123)
    other = generator.training_corruption(swapped, seed=123)
    torch.testing.assert_close(observed[:, :9], other[:, 9:18])
    torch.testing.assert_close(observed[:, 9:18], other[:, :9])
    torch.testing.assert_close(observed[:, 18:], other[:, 18:])
    probabilities = observed[:, list(DISTANCE_INDEX)] + mean[list(DISTANCE_INDEX)]
    assert (probabilities >= 0).all() and (probabilities.sum(1) <= 1).all()
    generator.eval()
    assert torch.equal(generator.training_corruption(z, seed=123), z)


def test_compact_reader_has_three_tokens_six_rows_and_round_trips() -> None:
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"coord_spec": "v2", "width": 16})
    model.generator.set_coord_stats(torch.zeros(11), torch.ones(11), 10)
    model.eval()
    coords = torch.randn(6, 11)
    view_u, view_v = model.generator.tokens(coords)
    assert view_u.shape == view_v.shape == (6, 3, 16)
    assert model.generator.prefix(0, view_u).shape == (6, 6, model.d_model)
    batch = {**_pair_batch(), "struct_coords": coords}
    with torch.no_grad():
        expected = model(batch)["logits"]
    clone = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"coord_spec": "v2", "width": 16})
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(clone(batch)["logits"], expected, rtol=0, atol=0)
    clone.intervention = "mean_context"
    with pytest.raises(ValueError, match="context"):
        clone(batch)


def test_compact_corruption_and_reader_are_swap_equivariant() -> None:
    model = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"coord_spec": "v2", "width": 16, "corruption": {"prob": 1.0}},
    )
    model.generator.set_coord_stats(torch.zeros(11), torch.ones(11), 10)
    coords = torch.randn(6, 11)
    coords[:, 8:] = 0
    coords[0, 8] = 1
    swapped = coords.clone()
    swapped[:, :2], swapped[:, 2:4] = coords[:, 2:4], coords[:, :2]
    noisy = model.generator.training_corruption(coords, seed=13)
    reverse = model.generator.training_corruption(swapped, seed=13)
    torch.testing.assert_close(noisy[:, :2], reverse[:, 2:4])
    torch.testing.assert_close(noisy[:, 2:4], reverse[:, :2])
    torch.testing.assert_close(noisy[:, 4:], reverse[:, 4:])
    assert (noisy[:, 8:] >= 0).all() and (noisy[:, 8:].sum(1) <= 1).all()
    _open_gates(model)
    model.eval()
    batch = _pair_batch()
    with torch.no_grad():
        original = model({**batch, "struct_coords": coords})["logits"]
        other = model(
            {
                "emb_a": batch["emb_b"],
                "emb_b": batch["emb_a"],
                "len_a": batch["len_b"],
                "len_b": batch["len_a"],
                "struct_coords": swapped,
            }
        )["logits"]
    torch.testing.assert_close(original, other, rtol=0, atol=1e-6)
