"""Call-contract tests for V3.1 attention layers."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from src.model.B0 import BlockSelfMixingLayer, CrossAttentionLayer


class _RecordingAttention(nn.Module):
    """Record whether callers explicitly discard attention weights."""

    def __init__(self) -> None:
        super().__init__()
        self.need_weights: list[bool | None] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        self.need_weights.append(need_weights)
        return query, None


def test_cross_attention_skips_discarded_weights_and_preserves_shapes() -> None:
    layer = CrossAttentionLayer(d_model=4, n_heads=2, dropout=0.0)
    cross_attention = _RecordingAttention()
    cls_attention = _RecordingAttention()
    layer.attn = cast(nn.MultiheadAttention, cross_attention)
    layer.attn_cls = cast(nn.MultiheadAttention, cls_attention)

    h_a = torch.randn(2, 3, 4)
    h_b = torch.randn(2, 5, 4)
    cls_token = torch.randn(2, 1, 4)
    mask_a = torch.zeros(2, 3, dtype=torch.bool)
    mask_b = torch.zeros(2, 5, dtype=torch.bool)

    output_a, output_b, output_cls = layer(h_a, h_b, cls_token, mask_a, mask_b)

    assert len(cross_attention.need_weights) == 2
    assert all(value is False for value in cross_attention.need_weights)
    assert len(cls_attention.need_weights) == 1
    assert all(value is False for value in cls_attention.need_weights)
    assert output_a.shape == h_a.shape
    assert output_b.shape == h_b.shape
    assert output_cls.shape == cls_token.shape


def test_block_self_attention_skips_discarded_weights_and_preserves_shapes() -> None:
    layer = BlockSelfMixingLayer(d_model=4, n_heads=2, dropout=0.0)
    attention = _RecordingAttention()
    layer.attn = cast(nn.MultiheadAttention, attention)

    h_a = torch.randn(2, 3, 4)
    h_b = torch.randn(2, 5, 4)
    cls_token = torch.randn(2, 1, 4)
    mask_a = torch.zeros(2, 3, dtype=torch.bool)
    mask_b = torch.zeros(2, 5, dtype=torch.bool)

    output_a, output_b, output_cls = layer(h_a, h_b, cls_token, mask_a, mask_b)

    assert len(attention.need_weights) == 2
    assert all(value is False for value in attention.need_weights)
    assert output_a.shape == h_a.shape
    assert output_b.shape == h_b.shape
    assert output_cls.shape == cls_token.shape
