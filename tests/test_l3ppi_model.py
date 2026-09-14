"""Behavioral checks for the paper-based virtual-path head."""

import io
from typing import Any

import torch
from src.baselines.l3ppi import WeightedGIN, build_l3ppi, path_number_loss


def config() -> dict[str, Any]:
    return {
        "backbone_config": {
            "input_dim": 6,
            "d_model": 8,
            "encoder_layers": 1,
            "n_heads": 2,
            "regularization": {"dropout": 0.1, "token_dropout": 0.1},
        },
        "k": 3,
        "surrogate_hidden": 8,
        "gate_hidden": 8,
    }


def test_template_and_gate_weights() -> None:
    model = build_l3ppi(config()).eval()
    assert model.template_edges.shape == (2, 14)
    assert len(set(map(tuple, model.template_edges.T.tolist()))) == 14
    assert model.template_edges.max() == 5
    seen: list[tuple[Any, ...]] = []

    def capture(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        seen.append(args)

    hook = model.surrogate.register_forward_pre_hook(capture)
    a, b = torch.randn(2, 16), torch.randn(2, 16)
    model(a, b, all_open=True)
    assert seen[0][0].shape == (12, 16)
    assert torch.equal(seen[0][2], torch.ones(28))
    with torch.no_grad():
        model.gate.readout.weight.zero_()
        model.gate.readout.bias.fill_(-10)
    _, p, g = model(a, b)
    assert (p < 0.5).all() and (g == 0).all()
    assert seen[-1][0].shape == (12, 16)  # closed nodes are retained
    assert (seen[-1][2] == 0).all()

    def single_path(
        _module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        return output.new_tensor([10.0, -10.0, -10.0]).repeat(2)

    gate_hook = model.gate.register_forward_hook(single_path)
    model(a, b)
    expected = torch.tensor([1, 0, 0, 1, 0, 0, 1]).repeat(4).float()
    torch.testing.assert_close(seen[-1][2], expected)
    gate_hook.remove()
    hook.remove()


def test_eval_symmetry_batching_and_standalone_reload() -> None:
    torch.manual_seed(12)
    model = build_l3ppi(config()).eval()
    model.initialize_prompt(torch.randn(7, 16))
    a, b = torch.randn(5, 16), torch.randn(5, 16)
    expected = model(a, b)[0]
    torch.testing.assert_close(model(b, a)[0], expected)
    torch.testing.assert_close(model(a, b)[0], expected, rtol=0, atol=0)
    shards = torch.cat([model(a[i : i + 1], b[i : i + 1])[0] for i in range(5)])
    torch.testing.assert_close(shards, expected)
    buf = io.BytesIO()
    torch.save({"model_config": config(), "model_state": model.state_dict()}, buf)
    buf.seek(0)
    checkpoint = torch.load(buf, weights_only=True)
    loaded = build_l3ppi(checkpoint["model_config"]).eval()
    loaded.load_state_dict(checkpoint["model_state"])
    torch.testing.assert_close(loaded(a, b)[0], expected)


def test_frozen_modules_allow_prompt_and_gate_input_gradients() -> None:
    torch.manual_seed(5)
    model = build_l3ppi(config()).train()
    model.initialize_prompt(torch.randn(7, 16))
    assert not model.encoder.training and not model.surrogate.training
    a, b = torch.randn(8, 16), torch.randn(8, 16)
    logits, _, _ = model(a, b)
    logits.sum().backward()
    assert model.prompt.grad is not None and model.prompt.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.surrogate.parameters())
    model.zero_grad(set_to_none=True)
    model(a, b, all_open=True)[0].sum().backward()
    assert model.prompt.grad is not None
    assert all(p.grad is None for p in model.gate.parameters())
    model.surrogate.requires_grad_(True)
    model.train()
    assert model.surrogate.training


def test_encoder_pooling_ignores_padding_and_backbone_load() -> None:
    torch.manual_seed(4)
    model = build_l3ppi(config())
    other = build_l3ppi(config())
    other.load_backbone({"model_state": model.state_dict()})
    x = torch.randn(2, 4, 6)
    lengths = torch.tensor([2, 4])
    actual = model.encode(x, lengths)
    assert actual.shape == (2, 16) and actual.dtype == torch.float32
    assert not actual.requires_grad
    x[0, 2:] = 99
    torch.testing.assert_close(model.encode(x, lengths), actual)
    torch.testing.assert_close(other.encode(x, lengths), actual)


def test_path_regularizer_gradient_direction_and_dead_zone() -> None:
    p = torch.full((2, 2, 3), 0.5, requires_grad=True)
    loss = path_number_loss(p, torch.tensor([1, 0]))
    torch.testing.assert_close(loss, torch.tensor([0.5, 0.5]))
    torch.autograd.backward(loss.sum())
    assert p.grad is not None
    assert (p.grad[0] < 0).all() and (p.grad[1] > 0).all()
    p = torch.stack((torch.ones(2, 3), torch.zeros(2, 3)))
    assert (path_number_loss(p, torch.tensor([1, 0])) == 0).all()


def test_weighted_gin_zero_edges_retain_self_features() -> None:
    model = WeightedGIN(1, 1, layers=1, dropout=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    x = torch.tensor([[1.0], [2.0]])
    edges = torch.tensor([[0, 1], [1, 0]])
    batch = torch.tensor([0, 0])
    closed = model(x, edges, torch.zeros(2), batch, 1)
    opened = model(x, edges, torch.ones(2), batch, 1)
    torch.testing.assert_close(closed, torch.tensor([8.0]))
    torch.testing.assert_close(opened, torch.tensor([11.0]))
