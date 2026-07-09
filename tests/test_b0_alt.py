"""Tests for the F0PairMLP model (B0-alt: MLP over F0 mean-pooled features)."""

from __future__ import annotations

import pytest
import torch
from src.model.b0_alt import F0PairMLP

pytestmark = pytest.mark.unit


class TestF0PairMLPShapes:
    def test_forward_returns_logits_of_expected_shape(self) -> None:
        model = F0PairMLP(input_dim=8, hidden_dims=(16, 8), dropout=0.0)
        batch = {"x_a": torch.randn(4, 8), "x_b": torch.randn(4, 8)}

        output = model(batch)

        assert output["logits"].shape == (4, 1)
        assert "loss" not in output

    def test_forward_accepts_kwargs_instead_of_batch(self) -> None:
        model = F0PairMLP(input_dim=8, hidden_dims=(16,), dropout=0.0)

        output = model(x_a=torch.randn(2, 8), x_b=torch.randn(2, 8))

        assert output["logits"].shape == (2, 1)

    def test_missing_keys_raises_key_error(self) -> None:
        model = F0PairMLP(input_dim=8, hidden_dims=(16,), dropout=0.0)

        with pytest.raises(KeyError):
            model(x_a=torch.randn(2, 8))


class TestF0PairMLPLoss:
    def test_loss_present_when_label_given(self) -> None:
        model = F0PairMLP(input_dim=8, hidden_dims=(16,), dropout=0.0)
        batch = {
            "x_a": torch.randn(4, 8),
            "x_b": torch.randn(4, 8),
            "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }

        output = model(batch)

        assert torch.isfinite(output["loss"]).item()
        assert output["loss"].dim() == 0

    def test_loss_absent_without_label(self) -> None:
        model = F0PairMLP(input_dim=8, hidden_dims=(16,), dropout=0.0)
        batch = {"x_a": torch.randn(4, 8), "x_b": torch.randn(4, 8)}

        output = model(batch)

        assert "loss" not in output


class TestF0PairMLPSymmetry:
    def test_swapping_x_a_and_x_b_gives_identical_logits(self) -> None:
        torch.manual_seed(0)
        model = F0PairMLP(input_dim=8, hidden_dims=(16, 8), dropout=0.0)
        model.eval()
        x_a = torch.randn(5, 8)
        x_b = torch.randn(5, 8)

        with torch.no_grad():
            out_ab = model({"x_a": x_a, "x_b": x_b})
            out_ba = model({"x_a": x_b, "x_b": x_a})

        assert torch.allclose(out_ab["logits"], out_ba["logits"], atol=1e-6)


class TestF0PairMLPDefaults:
    def test_default_input_dim_is_1536(self) -> None:
        model = F0PairMLP()
        batch = {"x_a": torch.randn(2, 1536), "x_b": torch.randn(2, 1536)}

        output = model(batch)

        assert output["logits"].shape == (2, 1)

    def test_name_attribute_is_f0_mlp(self) -> None:
        assert F0PairMLP.name == "f0_mlp"
