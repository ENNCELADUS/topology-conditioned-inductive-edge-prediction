"""Conditioned V3.1 trunk: cls-token gated cross-attention injection.

PairCrossAttention subclass adding gated topo/content conditioning after the
final n_inj blocks (design rev 3 §3.4). B0.py is never modified; the parent
forward loop is re-stated here with the injection.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils import checkpoint as checkpoint_module

from src.model.B0 import PairCrossAttention, _build_padding_mask
from src.model.egostitch.conditioning import GatedCrossAttention


class ConditionedPairCrossAttention(PairCrossAttention):
    """PairCrossAttention + zero-init gated cls conditioning (§3.4 pins)."""

    def __init__(
        self,
        *,
        n_inj: int = 1,
        xattn_heads: int = 8,
        xattn_dropout: float = 0.0,
        conditioning_ema_decay: float = 0.99,
        d_model: int,
        **kwargs: object,
    ) -> None:
        super().__init__(d_model=d_model, **kwargs)  # type: ignore[arg-type]
        if not 1 <= n_inj <= len(self.layers):
            raise ValueError("n_inj must be in [1, n_layers]")
        self.n_inj = n_inj
        self.topo_xattn = nn.ModuleList(
            GatedCrossAttention(
                d_model,
                xattn_heads,
                xattn_dropout,
                ema_decay=conditioning_ema_decay,
            )
            for _ in range(n_inj)
        )
        self.cont_xattn = nn.ModuleList(
            GatedCrossAttention(
                d_model,
                xattn_heads,
                xattn_dropout,
                ema_decay=conditioning_ema_decay,
            )
            for _ in range(n_inj)
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        topo_tokens: torch.Tensor | None = None,
        cont_tokens: torch.Tensor | None = None,
        topo_active: torch.Tensor | None = None,
        cont_active: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the pair trunk with optional gated topo/content cls conditioning.

        Args:
            h_a: Item A hidden states ``(batch, seq_len_a, d_model)``.
            h_b: Item B hidden states ``(batch, seq_len_b, d_model)``.
            lengths_a: Sequence lengths for A ``(batch,)``.
            lengths_b: Sequence lengths for B ``(batch,)``.
            topo_tokens: Optional scaffold tokens ``(batch, T, d_model)`` for
                the topo cross-attention pathway; ``None`` is a hard bypass.
            cont_tokens: Optional content tokens ``(batch, T, d_model)`` for
                the content cross-attention pathway; ``None`` is a hard bypass.
            topo_active: Optional ``(batch,)`` bool mask gating the topo
                pathway per-sample; required alongside ``topo_tokens``.
            cont_active: Optional ``(batch,)`` bool mask gating the content
                pathway per-sample; required alongside ``cont_tokens``.
            edge_mask: Optional ``(batch,)`` real-row mask used to exclude DDP
                filler rows from conditioning means.

        Returns:
            Fused pair representation ``(batch, d_model)``.
        """
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Item pair batches must have matching batch dimension")

        batch_size = h_a.size(0)
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        n_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)
            inj = idx - (n_layers - self.n_inj)
            if inj >= 0:
                if topo_tokens is not None and topo_active is not None:
                    cls_token = self.topo_xattn[inj](
                        cls_token, topo_tokens, None, topo_active, edge_mask
                    )
                if cont_tokens is not None and cont_active is not None:
                    cls_token = self.cont_xattn[inj](
                        cls_token, cont_tokens, None, cont_active, edge_mask
                    )
        cls_vec = cls_token.squeeze(1)
        if self.pair_readout_mode == "pair_context_gated":
            # Full and hard-null heads differ only through the conditioned cls
            # stream. Keep their final pair readout in fp32 so BF16 does not
            # quantize away that small, scientifically load-bearing residual.
            def fp32_readout(
                readout_a: torch.Tensor,
                readout_b: torch.Tensor,
                readout_cls: torch.Tensor,
            ) -> torch.Tensor:
                with torch.autocast(device_type=readout_cls.device.type, enabled=False):
                    return cast(
                        torch.Tensor,
                        self.pair_context_readout(
                            readout_a.float(),
                            readout_b.float(),
                            readout_cls.float(),
                            mask_a,
                            mask_b,
                        ),
                    )

            if self.training and torch.is_grad_enabled():
                return cast(
                    torch.Tensor,
                    checkpoint_module.checkpoint(
                        fp32_readout, h_a, h_b, cls_vec, use_reentrant=False
                    ),
                )
            return fp32_readout(h_a, h_b, cls_vec)
        base_repr = self._rich_pooling_readout(h_a, h_b, cls_vec, mask_a, mask_b)
        if self.pair_readout_mode == "grid_sketch_fusion":
            return cast(
                torch.Tensor,
                self.grid_sketch_readout(base_repr, h_a, h_b, mask_a, mask_b),
            )
        return base_repr
