"""`V3_1` node-factor path (B1 student): zero-init parity and pair feature."""

from __future__ import annotations

import pytest
import torch
from src.model.egostitch.classifier.b0_v31 import V3_1

pytestmark = pytest.mark.unit

_TINY_CONFIG: dict[str, object] = {
    "input_dim": 8,
    "d_model": 16,
    "encoder_layers": 1,
    "cross_attn_layers": 1,
    "n_heads": 2,
    "mlp_head": {"hidden_dims": [16], "dropout": 0.0},
    "regularization": {"dropout": 0.0},
}


def _batch(batch_size: int = 3, seq: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "emb_a": torch.randn(batch_size, seq, 8, generator=generator),
        "emb_b": torch.randn(batch_size, seq, 8, generator=generator),
        "len_a": torch.full((batch_size,), seq, dtype=torch.long),
        "len_b": torch.full((batch_size,), seq, dtype=torch.long),
    }


def test_zero_init_node_factor_keeps_baseline_logits() -> None:
    torch.manual_seed(0)
    baseline = V3_1(**_TINY_CONFIG)
    torch.manual_seed(0)
    student = V3_1(**_TINY_CONFIG, node_factor_dim=4)
    baseline.eval()
    student.eval()
    batch = _batch()
    with torch.no_grad():
        base_logits = baseline(batch)["logits"]
        student_out = student(batch)
    assert torch.equal(base_logits, student_out["logits"])
    assert "node_factor_pair" in student_out
    assert student_out["node_factor_pair"].shape == (3, 4)


def test_baseline_has_no_node_factor_output() -> None:
    model = V3_1(**_TINY_CONFIG)
    model.eval()
    with torch.no_grad():
        output = model(_batch())
    assert model.node_factor is None
    assert "node_factor_pair" not in output


def test_node_factor_residual_becomes_live_after_diag_w_moves() -> None:
    torch.manual_seed(0)
    student = V3_1(**_TINY_CONFIG, node_factor_dim=4)
    student.eval()
    batch = _batch()
    with torch.no_grad():
        before = student(batch)["logits"].clone()
        assert student.node_factor is not None
        student.node_factor.diag_w.add_(1.0)
        after = student(batch)["logits"]
    assert not torch.equal(before, after)
