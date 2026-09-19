"""Focused model contracts for the optional v17 motif-prompt path."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_template import EDGE_TYPES
from src.model.egostitch.classifier.motif_prompt import (
    PREDICTED_MASK_KEY,
    PREDICTED_PRESENCE_KEY,
    PREDICTED_WEIGHTS_KEY,
    TEMPLATE_KEY,
    TRUE_DIAGNOSTIC_KEY,
    MotifGenerator,
    MotifPromptConfig,
    V3_1MotifPrompt,
)

from tests.model.test_motif_prompt_model import _model, _open_gates, _statistics, _weights
from tests.test_prefix_model import _pair_batch, _tiny_base_config


def _v17(stage: str = "two", *, deployed: bool = False) -> V3_1MotifPrompt:
    block: dict[str, object] = {
        "stage": stage,
        "base_checkpoint": "base.pt",
        "width": 16,
        "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
        "generator_head": "presence_profile",
        "fields": ["topo_self", "topo_partner", "topo_rel", "topo_cnt", "topo_conf"],
        "row_gate": True,
    }
    if stage == "two":
        block["bundle_checkpoint"] = "bundle.pt"
    if deployed:
        block["deployed_generator_checkpoint"] = "deployed.pt"
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=block)
    model.install_mean_template(_statistics(torch.full((96,), 0.1)))
    model.install_presence_rates(torch.tensor([0.4, 0.6, 0.8]))
    return model


def test_presence_profile_has_gradient_and_renders_presence_times_profile() -> None:
    model = _v17()
    batch = _pair_batch(n=4)
    encoded_u = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_v = model.base.encoder(batch["emb_b"], batch["len_b"])
    generator = model.generator
    assert generator is not None
    weights, presence = generator.forward_with_presence(
        encoded_u, encoded_v, batch["len_a"], batch["len_b"]
    )
    assert presence is not None and presence.shape == (4, 3)
    target = torch.tensor([[1.0, 0.0, 1.0]]).expand_as(presence)
    loss = torch.nn.functional.binary_cross_entropy(presence, target) + weights.mean()
    loss.backward()
    assert generator.presence_heads is not None
    assert all(parameter.grad is not None for parameter in generator.presence_heads.parameters())
    assert torch.all(weights <= presence[:, torch.as_tensor(EDGE_TYPES)])


def test_row_gate_starts_near_one_with_zero_output_layer() -> None:
    model = _v17().eval()
    output = model(_pair_batch(n=3))
    assert model.row_gate_head is not None
    final = model.row_gate_head[-1]
    assert isinstance(final, torch.nn.Linear)
    assert torch.count_nonzero(final.weight) == 0
    torch.testing.assert_close(output["row_gate"], torch.sigmoid(torch.tensor(4.0)).expand(3))


def test_row_gate_requires_the_confidence_field() -> None:
    with pytest.raises(ValueError, match="requires fields to include 'topo_conf'"):
        MotifPromptConfig.from_mapping(
            {
                "stage": "one",
                "base_checkpoint": "base.pt",
                "generator_head": "presence_profile",
                "row_gate": True,
            }
        )


def test_gates_off_is_exact_and_confidence_cannot_change_it() -> None:
    model = _v17().eval().requires_grad_(False)
    _open_gates(model)
    batch = _pair_batch(n=3)
    encoded_u = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_v = model.base.encoder(batch["emb_b"], batch["len_b"])
    weights = _weights(3)
    model.intervention = "gates_off"
    low = model.logits_from_encoded(
        encoded_u,
        encoded_v,
        batch["len_a"],
        batch["len_b"],
        weights=weights,
        presence=torch.zeros(3, 3),
    )
    high = model.logits_from_encoded(
        encoded_u,
        encoded_v,
        batch["len_a"],
        batch["len_b"],
        weights=weights,
        presence=torch.ones(3, 3),
    )
    base = model.base(batch)["logits"]
    assert torch.equal(low, base) and torch.equal(high, base)


def test_mean_intervention_uses_persisted_presence_without_row_leakage() -> None:
    model = _v17().eval().requires_grad_(False)
    _open_gates(model)
    model.intervention = "mean"
    batch = _pair_batch(n=3)
    encoded_u = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_v = model.base.encoder(batch["emb_b"], batch["len_b"])
    weights = _weights(3)
    low = model.logits_from_encoded(
        encoded_u,
        encoded_v,
        batch["len_a"],
        batch["len_b"],
        weights=weights,
        presence=torch.zeros(3, 3),
    )
    high = model.logits_from_encoded(
        encoded_u,
        encoded_v,
        batch["len_a"],
        batch["len_b"],
        weights=weights,
        presence=torch.ones(3, 3),
    )
    assert torch.equal(low, high)
    output = model(batch)
    torch.testing.assert_close(
        output["presence_probabilities"], model.mean_presence.unsqueeze(0).expand(3, -1)
    )


def test_graph_only_warmup_detaches_presence_from_task_path() -> None:
    model = _v17()
    model.cfg = type(model.cfg).from_mapping({**model.cfg.to_dict(), "warmup_losses": "graph_only"})
    _open_gates(model)
    model.train()
    assert model.graph_only_warmup
    batch = _pair_batch(n=3)
    model(batch)["logits"].sum().backward()
    generator = model.generator
    assert generator is not None
    assert all(parameter.grad is None for parameter in generator.parameters())


def test_confidence_parameters_belong_to_interface_optimizer_group() -> None:
    model = _v17()
    groups = model.optimizer_parameter_groups(1e-3, 1e-4, 0.0)
    interface = next(group for group in groups if group["name"] == "interface")
    ids = {id(parameter) for parameter in interface["params"]}
    assert model.conf_proj is not None and model.row_gate_head is not None
    assert all(id(parameter) in ids for parameter in model.conf_proj.parameters())
    assert all(id(parameter) in ids for parameter in model.row_gate_head.parameters())
    assert model.row_gate_bias is not None and id(model.row_gate_bias) in ids


def test_v17_state_dict_reloads_exactly() -> None:
    source = _v17().eval()
    restored = _v17().eval()
    restored.load_state_dict(source.state_dict())
    batch = _pair_batch(n=3)
    first = source(batch)
    second = restored(batch)
    assert torch.equal(first["logits"], second["logits"])
    assert torch.equal(first["presence_probabilities"], second["presence_probabilities"])


def test_flat_v16_state_has_no_new_presence_buffer_and_strictly_reloads() -> None:
    source = _model("one")
    state = source.state_dict()
    assert "mean_presence" not in state
    restored = _model("one")
    restored.load_state_dict(state, strict=True)


def test_stage_one_crossfit_rows_and_deployed_eval_sources() -> None:
    model = _v17("one", deployed=True)
    model.cfg = type(model.cfg).from_mapping(
        {
            **model.cfg.to_dict(),
            "corruption": {**model.cfg.corruption.__dict__, "source": "predicted"},
        }
    )
    batch = _pair_batch(n=3)
    truth = _weights(3, seed=1)
    predicted = _weights(3, seed=2)
    batch[TEMPLATE_KEY] = truth
    batch[PREDICTED_WEIGHTS_KEY] = predicted
    batch[PREDICTED_PRESENCE_KEY] = MotifGenerator.presence_from_weights(predicted)
    batch[PREDICTED_MASK_KEY] = torch.tensor([1, 0, 1])
    model.train()
    encoded_u = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_v = model.base.encoder(batch["emb_b"], batch["len_b"])
    weights, _ = model.resolve_graph(batch, encoded_u, encoded_v, batch["len_a"], batch["len_b"])
    torch.testing.assert_close(weights[[0, 2]], predicted[[0, 2]])
    model.eval()
    deployed = model(batch)["predicted_weights"]
    batch[TRUE_DIAGNOSTIC_KEY] = torch.ones(3, dtype=torch.bool)
    diagnostic = model(batch)["predicted_weights"]
    assert not torch.equal(deployed, truth)
    torch.testing.assert_close(diagnostic, truth)


def test_predicted_confidence_cache_is_required() -> None:
    model = _v17("one", deployed=True)
    model.cfg = type(model.cfg).from_mapping(
        {
            **model.cfg.to_dict(),
            "corruption": {**model.cfg.corruption.__dict__, "source": "predicted"},
        }
    )
    batch = _pair_batch(n=2)
    batch[TEMPLATE_KEY] = _weights(2, seed=1)
    batch[PREDICTED_WEIGHTS_KEY] = _weights(2, seed=2)
    model.train()
    encoded_u = model.base.encoder(batch["emb_a"], batch["len_a"])
    encoded_v = model.base.encoder(batch["emb_b"], batch["len_b"])
    with torch.no_grad(), pytest.raises(ValueError, match="motif_predicted_presence"):
        model.resolve_graph(batch, encoded_u, encoded_v, batch["len_a"], batch["len_b"])
