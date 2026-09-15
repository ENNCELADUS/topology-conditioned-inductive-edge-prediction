"""The Stage II coordinate generator: symmetry, frozen reader, composite loss, scoring."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest
import torch
from src.data.struct_coords import COORD_DIM, FIELD_SLICES
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.coord_gen import (
    CONTINUOUS_INDEX,
    DISTANCE_CLASSES,
    DISTANCE_INDEX,
    SELF_DISTANCE_CLASS,
    CoordGenConfig,
    V3_1CoordGen,
    distance_class_targets,
)

from tests.test_prefix_model import _pair_batch, _tiny_base_config


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


def _model(seed: int = 0, **extra: object) -> V3_1CoordGen:
    torch.manual_seed(seed)
    model = V3_1CoordGen(
        reader=_reader_config(),
        coord_gen={
            "hidden": 16,
            "layers": 1,
            "dropout": 0.0,
            "endpoint_dropout": 0.0,
            "endpoint_hidden": 16,
            **extra,
        },
    )
    model.reader.generator.set_coord_stats(
        torch.linspace(-1.0, 1.0, COORD_DIM), torch.linspace(0.5, 2.0, COORD_DIM), 7
    )
    with torch.no_grad():
        model.reader.generator.gates.fill_(0.4)
    model.initialize_teacher()
    model.eval()
    return model


def _true_coords(n: int, seed: int = 0) -> torch.Tensor:
    """Raw coordinates with a valid distance one-hot; the last row is a self-pair (all zero)."""
    gen = torch.Generator().manual_seed(seed)
    coords = torch.rand(n, COORD_DIM, generator=gen) * 3.0
    coords[:, list(DISTANCE_INDEX)] = 0.0
    classes = torch.randint(0, len(DISTANCE_INDEX), (n,), generator=gen)
    coords[torch.arange(n - 1), torch.tensor(DISTANCE_INDEX)[classes[:-1]]] = 1.0
    return coords


def test_distance_class_targets_keep_self_pairs_as_their_own_class() -> None:
    coords = _true_coords(6)
    targets = distance_class_targets(coords)
    assert targets.dtype == torch.int64 and targets.shape == (6,)
    assert int(targets[-1]) == SELF_DISTANCE_CLASS
    one_hot = coords[:-1, list(DISTANCE_INDEX)]
    assert torch.equal(targets[:-1], one_hot.argmax(dim=1))
    assert len(DISTANCE_INDEX) + 1 == DISTANCE_CLASSES


def _swapped(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["emb_a"], out["emb_b"] = batch["emb_b"], batch["emb_a"]
    out["len_a"], out["len_b"] = batch["len_b"], batch["len_a"]
    return out


def test_config_validation_and_round_trip() -> None:
    cfg = CoordGenConfig.from_mapping({"hidden": 32, "kd_alpha": 0.0})
    assert CoordGenConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown coord_gen keys"):
        CoordGenConfig.from_mapping({"width": 8})
    with pytest.raises(ValueError, match="non-negative"):
        CoordGenConfig.from_mapping({"w_coord": -1.0})
    with pytest.raises(ValueError, match="dropout"):
        CoordGenConfig.from_mapping({"dropout": 1.0})
    with pytest.raises(ValueError, match="coord_spec"):
        CoordGenConfig.from_mapping({"coord_spec": "v0"})


def test_reader_is_frozen_and_only_the_generator_trains() -> None:
    model = _model(trainable="generator")
    assert all(not param.requires_grad for param in model.reader.parameters())
    model.train()
    assert model.training and model.generator.training and not model.reader.training
    trainable = {id(param) for param in model.trainable_parameters()}
    assert trainable == {id(param) for param in model.generator.parameters()}
    assert next(model.parameters()) is next(model.generator.parameters())


def test_swapping_the_pair_swaps_endpoint_fields_and_fixes_the_rest() -> None:
    model = _model()
    batch = _pair_batch()
    with torch.no_grad():
        forward = model(batch)
        swapped = model(_swapped(batch))
    z, z_swapped = forward["predicted_coords"], swapped["predicted_coords"]
    torch.testing.assert_close(
        z_swapped[:, FIELD_SLICES["endpoint_u"]], z[:, FIELD_SLICES["endpoint_v"]]
    )
    torch.testing.assert_close(
        z_swapped[:, FIELD_SLICES["endpoint_v"]], z[:, FIELD_SLICES["endpoint_u"]]
    )
    torch.testing.assert_close(
        z_swapped[:, FIELD_SLICES["relation"]], z[:, FIELD_SLICES["relation"]]
    )
    torch.testing.assert_close(z_swapped[:, FIELD_SLICES["context"]], z[:, FIELD_SLICES["context"]])
    torch.testing.assert_close(swapped["logits"], forward["logits"], rtol=0.0, atol=1e-5)


def test_distance_classes_enter_as_standardised_soft_one_hots() -> None:
    model = _model()
    with torch.no_grad():
        out = model(_pair_batch())
    z = out["predicted_coords"]
    assert z.dtype == torch.float32 and out["distance_logits"].shape[1] == DISTANCE_CLASSES
    stats = model.reader.generator
    dist = list(DISTANCE_INDEX)
    probs = z[:, dist] * stats.coord_std[dist] + stats.coord_mean[dist]
    assert (probs >= 0).all() and (probs <= 1).all()
    assert (probs.sum(dim=1) <= 1.0 + 1e-6).all()
    # A confident self-pair prediction reproduces the true all-zero one-hot.
    parts = {
        "endpoint_u": torch.zeros(2, 9),
        "endpoint_v": torch.zeros(2, 9),
        "relation": torch.zeros(2, 7),
        "context": torch.zeros(2, 5),
        "distance_logits": torch.full((2, DISTANCE_CLASSES), -40.0),
    }
    parts["distance_logits"][:, SELF_DISTANCE_CLASS] = 40.0
    z_self = model.assemble(parts)
    truth = torch.zeros(2, COORD_DIM)
    torch.testing.assert_close(z_self[:, dist], stats.standardize(truth)[:, dist])


def test_bf16_autocast_assembles_float32_coordinates_and_trains() -> None:
    model = _model()
    model.train()
    batch = _pair_batch()
    coords = _true_coords(batch["emb_a"].size(0))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model({**batch, "struct_coords": coords})
    assert out["predicted_coords"].dtype == torch.float32
    assert out["distance_logits"].dtype == torch.float32
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(param.grad is not None for param in model.generator.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        scored = model({key: value for key, value in batch.items() if key != "label"})
    assert scored["predicted_coords"].dtype == torch.float32


def test_forward_scores_without_truth_and_supervises_with_it() -> None:
    model = _model()
    batch = _pair_batch()
    scoring = {key: value for key, value in batch.items() if key != "label"}
    with torch.no_grad():
        out = model(scoring)
    assert set(out) == {"logits", "predicted_coords", "distance_logits"}
    with torch.no_grad():
        labelled = model(batch)
    assert "loss" not in labelled and "logits" in labelled
    coords = _true_coords(batch["emb_a"].size(0))
    supervised = model({**batch, "struct_coords": coords})
    for key in ("loss", "loss_weight_sum", "task_loss", "coord_loss", "kd_loss", "teacher_logits"):
        assert key in supervised
    # All-negative rows weigh 1, so the composite is the plain sum of the means.
    negatives = {**batch, "label": torch.zeros_like(batch["label"]), "struct_coords": coords}
    out = model(negatives)
    expected = 0.5 * out["task_loss"] + out["coord_loss"] + 0.5 * out["kd_loss"]
    torch.testing.assert_close(out["loss"], expected, rtol=1e-5, atol=1e-6)
    assert float(out["loss_weight_sum"]) == float(batch["label"].numel())
    out["loss"].backward()
    assert all(param.grad is not None for param in model.generator.parameters())
    assert all(param.grad is None for param in model.teacher.parameters())
    assert all(param.grad is None for param in model.encoder.parameters())
    no_kd = _model(kd_alpha=0.0)
    out_no_kd = no_kd({**batch, "struct_coords": coords})
    assert "teacher_logits" not in out_no_kd and "kd_loss" not in out_no_kd


def test_coordinate_loss_is_zero_at_the_truth() -> None:
    model = _model()
    coords = _true_coords(5)
    z_star = model.reader.generator.standardize(coords)
    parts = {"distance_logits": torch.full((5, DISTANCE_CLASSES), -30.0)}
    parts["distance_logits"][torch.arange(5), distance_class_targets(coords)] = 30.0
    total, continuous, distance = model.coordinate_loss_rows(parts, z_star, coords)
    assert float(continuous.abs().max()) == 0.0
    assert float(distance.max()) < 1e-6 and float(total.max()) < 1e-6
    assert len(CONTINUOUS_INDEX) == COORD_DIM - len(DISTANCE_INDEX)


def test_coordinate_supervision_ignores_self_rows_but_keeps_nonself_distance() -> None:
    model = _model()
    coords = _true_coords(4)
    z = model.reader.generator.standardize(coords) + 2.0
    parts = {"distance_logits": torch.zeros(4, DISTANCE_CLASSES)}
    total, continuous, distance = model.coordinate_loss_rows(parts, z, coords)
    assert total[-1] == 0 and continuous[-1] == 0 and distance[-1] == 0
    assert (continuous[:-1] > 0).all()
    torch.testing.assert_close(distance[:-1], torch.full((3,), float(torch.log(torch.tensor(5.0)))))


def test_coordinate_loss_uses_nonself_training_units_and_saves_scale() -> None:
    model = _model()
    coords = _true_coords(4)
    coords[:3, list(CONTINUOUS_INDEX)] = torch.tensor([0.0, 1.0, 2.0])[:, None]
    coords[-1, list(CONTINUOUS_INDEX)] = 1000
    model.install_coordinate_scale(coords)
    z = model.reader.generator.standardize(coords)
    prediction = z.clone()
    prediction[:, list(CONTINUOUS_INDEX)] += model.coordinate_scale
    parts = {"distance_logits": torch.zeros(4, DISTANCE_CLASSES)}
    _, continuous, _ = model.coordinate_loss_rows(parts, prediction, coords)
    torch.testing.assert_close(continuous[:3], torch.full((3,), 0.5))
    rebuilt = _model()
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(rebuilt.coordinate_scale, model.coordinate_scale)


def test_compact_student_scores_with_three_fields_and_four_distance_classes() -> None:
    reader = _reader_config()
    reader["topo_prompt"] = {**cast(dict[str, object], reader["topo_prompt"]), "coord_spec": "v2"}
    model = V3_1CoordGen(reader=reader, coord_gen={"coord_spec": "v2", "hidden": 16, "layers": 1})
    model.reader.generator.set_coord_stats(torch.zeros(11), torch.ones(11), 10)
    model.initialize_teacher()
    model.eval()
    out = model(_pair_batch())
    assert out["predicted_coords"].shape == (len(_pair_batch()["label"]), 11)
    assert out["distance_logits"].shape[1] == 4
    assert "loss" not in out


def test_virtual_student_round_trip_and_training_keep_teacher_frozen() -> None:
    reader = _reader_config()
    reader["topo_prompt"] = {**cast(dict[str, object], reader["topo_prompt"]), "coord_spec": "v2"}
    config = {
        "coord_spec": "v2",
        "generator": "virtual_graph",
        "virtual_graph": {"k": 3, "d_z": 8, "heads": 2},
    }
    model = V3_1CoordGen(reader=reader, coord_gen=config)
    model.reader.generator.set_coord_stats(torch.zeros(11), torch.ones(11), 10)
    model.initialize_teacher()
    with torch.no_grad():
        model.reader.generator.gates.fill_(0.2)
    batch = _pair_batch()
    coords = torch.rand(len(batch["label"]), 11)
    coords[:, 8:] = 0
    coords[:, 8] = 1
    model.install_coordinate_scale(coords)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch | {"struct_coords": coords})
    output["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.generator.parameters()
    )
    assert all(p.grad is None for p in model.teacher.parameters())
    groups = model.optimizer_parameter_groups(3e-4, 1e-4, 0.05)
    optimized = {id(p) for g in groups for p in cast(Iterable[torch.nn.Parameter], g["params"])}
    assert optimized == {id(p) for p in model.trainable_parameters()}
    rebuilt = V3_1CoordGen(reader=reader, coord_gen=config)
    rebuilt.load_state_dict(model.state_dict())
    model.eval()
    rebuilt.eval()
    torch.testing.assert_close(model(batch)["logits"], rebuilt(batch)["logits"])
    model.intervention = "slot_gates_open"
    assert model.intervention == "slot_gates_open"
    assert torch.isfinite(model(batch)["logits"]).all()
    with pytest.raises(ValueError, match="virtual"):
        _model().intervention = "slot_gates_open"


def test_logits_from_encoded_matches_forward_and_gates_off_is_the_reader_base() -> None:
    model = _model()
    batch = _pair_batch()
    with torch.no_grad():
        encoded_a = model.encoder(batch["emb_a"], batch["len_a"])
        encoded_b = model.encoder(batch["emb_b"], batch["len_b"])
        from_encoded = model.logits_from_encoded(
            encoded_a, encoded_b, batch["len_a"], batch["len_b"]
        )
        forward = model(batch | {"struct_coords": _true_coords(batch["emb_a"].size(0))})["logits"]
    torch.testing.assert_close(from_encoded, forward, rtol=0.0, atol=1e-6)
    base = V3_1(**_tiny_base_config())
    base.load_state_dict(model.reader.base.state_dict())
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    model.intervention = "gates_off"
    assert model.reader.intervention == "gates_off"
    with torch.no_grad():
        observed = model({key: value for key, value in batch.items() if key != "label"})["logits"]
        expected = base(batch)["logits"]
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1e-6)


def test_state_dict_round_trip_keeps_reader_statistics() -> None:
    model = _model()
    rebuilt = V3_1CoordGen(reader=_reader_config(), coord_gen=model.cfg.to_dict())
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        rebuilt.reader.generator.coord_mean, model.reader.generator.coord_mean
    )
    assert float(rebuilt.reader.generator.coord_count) == 7.0
    keys = list(model.state_dict())
    assert len(keys) == len(set(keys))


def test_teacher_stays_bit_identical_while_interface_trains() -> None:
    model = _model()
    snapshot = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    model.train()
    assert model.reader.generator.training and model.reader.base.output_head.training
    assert not model.encoder.training and not model.reader.base.cross_attention.training
    assert not model.teacher.training
    groups = model.optimizer_parameter_groups(3e-4, 1e-4, 0.05)
    assert [g["max_lr"] for g in groups] == [3e-4, 3e-4, 1e-4]
    optimized = {
        id(p) for group in groups for p in cast(Iterable[torch.nn.Parameter], group["params"])
    }
    assert not optimized.intersection(id(p) for p in model.teacher.parameters())
    assert optimized == {id(p) for p in model.trainable_parameters()}
    optimizer = torch.optim.AdamW(groups)
    batch = _pair_batch()
    output = model(batch | {"struct_coords": _true_coords(len(batch["label"]))})
    output["loss"].backward()
    optimizer.step()
    for key, value in snapshot.items():
        assert torch.equal(value, model.teacher.state_dict()[key])
    torch.testing.assert_close(output["kd_loss"] - output["teacher_entropy"], output["kd_kl"])
