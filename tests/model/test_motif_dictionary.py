"""Dictionary routing, checkpoint, and head-only adaptation regressions."""

from __future__ import annotations

import pytest
import torch
from src.data.motif_dictionary import DictionaryArtifact
from src.data.motif_template import SWAP_PERM, MotifTemplateStatistics, template_statistics
from src.model.egostitch.classifier.motif_dictionary import (
    ROUTE_TARGET_KEY,
    V3_1MotifDictionary,
)
from src.model.egostitch.classifier.motif_prompt import (
    TEMPLATE_KEY,
    TEMPLATE_MASK_KEY,
    V3_1MotifPrompt,
)

from tests.test_prefix_model import _pair_batch, _tiny_base_config


def _motif_block(**extra: object) -> dict[str, object]:
    block: dict[str, object] = {
        "stage": "two",
        "base_checkpoint": "absent-base.pt",
        "bundle_checkpoint": "absent-stage1.pt",
        "width": 16,
        "slots_per_field": 2,
        "reader": {"layers": 2, "dim": 16, "heads": 4, "rrwp_k": 4},
        "slot_read": "residual_block",
        "slot_read_value_norm": False,
    }
    block.update(extra)
    return block


def _artifact(width: int = 16) -> DictionaryArtifact:
    gen = torch.Generator().manual_seed(4)
    tokens = torch.randn(4, 4, width, generator=gen)
    # Each swap partner exchanges only endpoint fields.
    tokens[1, 0] = tokens[0, 1]
    tokens[1, 1] = tokens[0, 0]
    tokens[1, 2:] = tokens[0, 2:]
    tokens[3, 0] = tokens[2, 1]
    tokens[3, 1] = tokens[2, 0]
    tokens[3, 2:] = tokens[2, 2:]
    weights = torch.rand(4, 96, generator=gen)
    weights[1] = weights[0, list(SWAP_PERM)]
    weights[3] = weights[2, list(SWAP_PERM)]
    return DictionaryArtifact(
        weights=weights,
        tokens=tokens,
        scales=torch.tensor([1.0, 1.0, 2.0, 3.0]),
        temperature=0.7,
        mean_weights=torch.full((4,), 0.25),
        swap_index=torch.tensor([1, 0, 3, 2]),
        metadata={},
    )


def _dictionary(mode: str = "sequence", seed: int = 0) -> V3_1MotifDictionary:
    torch.manual_seed(seed)
    model = V3_1MotifDictionary(
        base=_tiny_base_config(),
        motif_prompt=_motif_block(),
        motif_dictionary={
            "mode": mode,
            "artifact_path": "absent-dictionary.pt",
            "dictionary_size": 4,
            "slot_checkpoint": "absent-wave3.pt",
        },
    )
    model.install_dictionary(_artifact())
    with torch.no_grad():
        model.adapter.gates.fill_(0.7)
    return model


def _stats() -> MotifTemplateStatistics:
    return template_statistics(torch.full((1, 96), 0.1).numpy())


def _swapped(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["emb_a"], out["emb_b"] = batch["emb_b"], batch["emb_a"]
    out["len_a"], out["len_b"] = batch["len_b"], batch["len_a"]
    return out


def test_routes_and_logits_are_endpoint_swap_equivariant_and_symmetric() -> None:
    model = _dictionary().eval().requires_grad_(False)
    batch = _pair_batch(n=4)
    encoded_a = model.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.encoder(batch["emb_b"], batch["len_b"])
    routes = model.predict_routes(encoded_a, encoded_b, batch["len_a"], batch["len_b"])
    swapped = model.predict_routes(encoded_b, encoded_a, batch["len_b"], batch["len_a"])
    torch.testing.assert_close(
        swapped, routes[:, model.dictionary_swap_index], rtol=1e-5, atol=1e-6
    )
    logits = model.logits_from_routes(
        encoded_a, encoded_b, batch["len_a"], batch["len_b"], routes=routes
    )
    swapped_logits = model.logits_from_routes(
        encoded_b, encoded_a, batch["len_b"], batch["len_a"], routes=swapped
    )
    torch.testing.assert_close(swapped_logits, logits, rtol=1e-5, atol=1e-6)
    model.intervention = "gates_off"
    gates_off = model.logits_from_routes(
        encoded_a, encoded_b, batch["len_a"], batch["len_b"], routes=routes
    )
    assert not torch.allclose(logits, gates_off, rtol=1e-5, atol=1e-6)


def test_sequence_forward_masks_self_route_rows_but_keeps_autograd_connected() -> None:
    model = _dictionary().train()
    batch = _pair_batch(n=4)
    batch[ROUTE_TARGET_KEY] = torch.full((4, 4), 0.25)
    batch[TEMPLATE_MASK_KEY] = torch.zeros(4)
    output = model(batch)
    assert torch.equal(output["route_loss_rows"], torch.zeros(4))
    output["route_loss_rows"].sum().backward()
    assert all(
        param.grad is not None
        for param in model.router.parameters()
        if param.requires_grad
    )


def test_extreme_route_logits_have_finite_kl_and_a_correcting_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _dictionary().train()
    extreme = torch.nn.Parameter(torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]]))

    def fixed_logits(*_args: torch.Tensor) -> torch.Tensor:
        return extreme.expand(2, -1)

    monkeypatch.setattr(model, "_route_logits", fixed_logits)
    batch = _pair_batch(n=2)
    batch[ROUTE_TARGET_KEY] = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    batch[TEMPLATE_MASK_KEY] = torch.ones(2)
    loss = model(batch)["route_loss_rows"].mean()
    assert torch.isfinite(loss) and float(loss.detach()) > 1000.0
    loss.backward()
    assert extreme.grad is not None and torch.isfinite(extreme.grad).all()
    assert float(extreme.grad[0, 0]) > 0.0
    assert float(extreme.grad[0, 1]) < 0.0


def test_sequence_real_adamw_step_changes_only_slot_encoder_and_router() -> None:
    model = _dictionary().train()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any(name.startswith("router.") for name in trainable)
    assert any(name.startswith("generator.attention.") for name in trainable)
    assert not any(name.startswith("generator.message_layers.") for name in trainable)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2, weight_decay=0.2)
    batch = _pair_batch(n=4)
    batch[ROUTE_TARGET_KEY] = torch.eye(4)
    batch[TEMPLATE_MASK_KEY] = torch.ones(4)
    output = model(batch)
    optimizer.zero_grad()
    output["route_loss_rows"].mean().backward()
    optimizer.step()
    after = dict(model.named_parameters())
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert any(name.startswith("router.") for name in changed)
    assert any(name.startswith("generator.") for name in changed)
    assert changed <= trainable
    assert all(torch.equal(before[name], after[name]) for name in set(before) - trainable)


def test_oracle_derives_routes_from_templates_and_fails_closed_without_them() -> None:
    model = _dictionary(mode="oracle").eval()
    weights = torch.rand(3, 96)
    routes = model.routes_from_weights(weights)
    assert routes.shape == (3, 4)
    assert routes.dtype == torch.float32
    torch.testing.assert_close(routes.sum(dim=-1), torch.ones(3))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_routes = model.routes_from_weights(weights)
    assert autocast_routes.dtype == torch.float32
    torch.testing.assert_close(autocast_routes, routes, rtol=0, atol=0)
    batch = _pair_batch(n=3)
    with pytest.raises(ValueError, match="requires motif_route_targets or motif_weights"):
        model(batch)


def test_checkpoint_round_trip_needs_no_source_or_artifact_files() -> None:
    model = _dictionary(seed=0).eval().requires_grad_(False)
    batch = _pair_batch(n=3)
    before = model(batch)["logits"]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    restored = V3_1MotifDictionary(
        base=_tiny_base_config(),
        motif_prompt=_motif_block(),
        motif_dictionary={
            "mode": "sequence",
            "artifact_path": "/does/not/exist",
            "dictionary_size": 4,
            "slot_checkpoint": "/also/absent",
        },
    ).eval()
    restored.load_state_dict(state, strict=True)
    after = restored(batch)["logits"]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert bool(restored.dictionary_installed)


def test_legacy_graph_logits_equal_the_factored_token_path() -> None:
    torch.manual_seed(2)
    model = V3_1MotifPrompt(base=_tiny_base_config(), motif_prompt=_motif_block()).eval()
    model.install_mean_template(_stats())
    with torch.no_grad():
        model.adapter.gates.fill_(0.7)
    batch = _pair_batch(n=3)
    weights = torch.rand(3, 96, generator=torch.Generator().manual_seed(7))
    encoded_a = model.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.encoder(batch["emb_b"], batch["len_b"])
    tokens = model.tokens_from_weights(
        weights, encoded_a, encoded_b, batch["len_a"], batch["len_b"]
    )
    graph = model.logits_from_encoded(
        encoded_a, encoded_b, batch["len_a"], batch["len_b"], weights=weights
    )
    direct = model.logits_from_tokens(
        encoded_a, encoded_b, batch["len_a"], batch["len_b"], tokens=tokens
    )
    torch.testing.assert_close(direct, graph, rtol=0, atol=0)


def test_head_only_real_adamw_step_changes_only_output_head() -> None:
    torch.manual_seed(8)
    model = V3_1MotifPrompt(
        base=_tiny_base_config(),
        motif_prompt=_motif_block(
            training_policy="head_only",
            head_prompt_enabled=True,
            init_checkpoint="absent-wave3.pt",
        ),
    )
    model.install_mean_template(_stats())
    model.train()
    assert model.training and model.base.output_head.training
    assert not model.base.encoder.training
    assert not model.base.cross_attention.training
    assert not model.generator.training  # type: ignore[union-attr]
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all(name.startswith("base.output_head.") for name in trainable)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2, weight_decay=0.2)
    batch = _pair_batch(n=4)
    batch[TEMPLATE_KEY] = torch.rand(4, 96)
    output = model(batch)
    assert "slot_loss_rows" not in output and "topo_loss_rows" not in output
    optimizer.zero_grad()
    output["loss"].backward()
    optimizer.step()
    changed = {
        name
        for name, param in model.named_parameters()
        if not torch.equal(before[name], param.detach())
    }
    assert changed and changed <= trainable
    frozen = set(before) - trainable
    assert all(torch.equal(before[name], dict(model.named_parameters())[name]) for name in frozen)


def test_head_only_content_control_disables_prompt_in_train_and_eval() -> None:
    model = V3_1MotifPrompt(
        base=_tiny_base_config(),
        motif_prompt=_motif_block(
            training_policy="head_only",
            head_prompt_enabled=False,
            init_checkpoint="absent-wave3.pt",
        ),
    )
    model.install_mean_template(_stats())
    with torch.no_grad():
        model.adapter.gates.fill_(2.0)
    batch = _pair_batch(n=3)
    batch[TEMPLATE_KEY] = torch.rand(3, 96)
    model.eval()
    torch.testing.assert_close(model(batch)["logits"], model.base(_pair_batch(n=3))["logits"])
    model.train()
    # The head has dropout in train mode, so compare the deterministic feature
    # producer rather than two independently sampled head evaluations.
    encoded_a = model.encoder(batch["emb_a"], batch["len_a"])
    encoded_b = model.encoder(batch["emb_b"], batch["len_b"])
    weights = model.resolve_weights(batch, encoded_a, encoded_b, batch["len_a"], batch["len_b"])
    tokens = model.tokens_from_weights(
        weights, encoded_a, encoded_b, batch["len_a"], batch["len_b"]
    )
    prompted = model.logits_from_tokens(
        encoded_a,
        encoded_b,
        batch["len_a"],
        batch["len_b"],
        tokens=tokens,
        return_pair_repr=True,
    )
    base = model.base._pair_representation(  # noqa: SLF001
        encoded_a, encoded_b, batch["len_a"], batch["len_b"]
    )
    torch.testing.assert_close(prompted, base, rtol=0, atol=0)
