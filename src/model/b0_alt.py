"""B0-alt: an MLP over F0 mean-pooled node features (architecture-independence arm)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.B0 import MLPHead


class F0PairMLP(nn.Module):
    """MLP pairwise scorer over concatenated F0 mean-pooled node-pair features.

    Builds a symmetric pair feature ``concat[x_a + x_b, |x_a - x_b|, x_a * x_b]``
    (invariant to swapping ``x_a``/``x_b``) and feeds it through an :class:`MLPHead`
    to produce a single edge logit. Mirrors ``V3_1``'s forward contract: a batch
    dict (or kwargs) in, ``{"logits": ..., optional "loss": ...}`` out.
    """

    name = "f0_mlp"

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        """Build the pair-feature MLP head.

        Args:
            input_dim: Dimensionality of each node's F0 mean-pooled feature vector.
            hidden_dims: Hidden layer widths of the MLP head.
            dropout: Dropout probability applied after each hidden layer.
            activation: Activation name (``"gelu"``, ``"relu"``, ``"silu"``, ``"tanh"``).
            norm: Normalization name (``"layernorm"``, ``"batchnorm"``, ``"none"``).
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.head = MLPHead(
            input_dim=3 * self.input_dim,
            hidden_dims=list(hidden_dims),
            output_dim=1,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score a batch of node pairs from their F0 mean-pooled features.

        Args:
            batch: Optional batch dictionary with keys ``x_a`` and ``x_b`` (each
                ``(B, input_dim)``) and optional ``label`` ``(B,)``.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            Dict with ``logits`` ``(B, 1)`` and, when ``label`` is present, ``loss``
            (BCE-with-logits, scalar).

        Raises:
            KeyError: If ``x_a`` or ``x_b`` is missing from the merged batch.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)

        if "x_a" not in merged or "x_b" not in merged:
            raise KeyError("Batch must contain 'x_a' and 'x_b' tensors")

        x_a = merged["x_a"]
        x_b = merged["x_b"]
        pair_features = torch.cat([x_a + x_b, torch.abs(x_a - x_b), x_a * x_b], dim=-1)
        logits = self.head(pair_features)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            output["loss"] = F.binary_cross_entropy_with_logits(logits_for_loss, labels_for_loss)

        return output


__all__ = ["F0PairMLP"]
