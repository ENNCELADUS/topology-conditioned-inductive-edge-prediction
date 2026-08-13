"""``NodeFactorBottleneck``: zero-init low-rank node-factor path (B1 KD plan).

Distillation cannot add per-pair information (`(x_u, x_v)` is still the
strict task input); this module installs a topology-supervised *structured
inductive bias* instead -- a low-rank per-node factorization whose row
ordering, activity biases, and pair-embedding geometry are shaped by teacher
targets in `src/distill/` (Wave 2+). It is additive on the fused pair logit
and lives entirely on the content path: built by `B0V31PairClassifier`
whenever `ClassifierConfig.node_factor_dim > 0`, independent of the topo
conditioning ladder (`CONDITIONING_MODES`) and never frozen by
`freeze_unreachable_conditioning`.

``diag_w`` is zero-initialized and there is no scalar gate. A scalar gate
``gamma`` at ``0`` is a saddle: ``E[dL/dgamma] ~= 0`` under random ``z``, so
escaping it rides on minibatch noise alone. Zero-initializing ``diag_w``
instead keeps the residual exactly ``0`` at step 0 (bit-identical to the
``node_factor_dim=0`` baseline) while giving every coordinate its own direct
gradient from step 0, ``dL/dw_k = z_ak * z_bk / sqrt(dim)`` -- nonzero
whenever the two endpoint embeddings are not orthogonal on that coordinate,
which is the generic case at random init. That removes both the saddle and
the asymmetry a scalar gate would create between logit-space KD arms (which
see gradient through `r` immediately) and embedding-space KD arms (which
would not, until the gate first moves).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from src.model.egostitch.classifier.layers import (
    _build_padding_mask,
    inner_token_mask,
    masked_mean,
)


class NodeFactorOutputs(NamedTuple):
    """One pair batch's node-factor decomposition.

    Attributes:
        z_a: Shape ``(B, dim)`` unit-norm endpoint-A factor embedding.
        z_b: Shape ``(B, dim)`` unit-norm endpoint-B factor embedding.
        r: Shape ``(B,)`` symmetric bilinear factor term.
        b_a: Shape ``(B,)`` endpoint-A activity bias.
        b_b: Shape ``(B,)`` endpoint-B activity bias.
        residual: Shape ``(B,)`` ``r + b_a + b_b``, added to the fused logit.
    """

    z_a: torch.Tensor
    z_b: torch.Tensor
    r: torch.Tensor
    b_a: torch.Tensor
    b_b: torch.Tensor
    residual: torch.Tensor


class NodeFactorBottleneck(nn.Module):
    """Zero-init low-rank node-factor residual on the fused pair logit.

    ``z_i = normalize(w_z(pool(H_i)))``; ``r = (z_a * diag_w * z_b).sum() /
    sqrt(dim)`` (symmetric in ``a``/``b`` by construction); ``b_i =
    bias_head(pool(H_i))``. Pooling reuses `classifier.layers.masked_mean`
    over `classifier.layers.inner_token_mask` -- the same inner-token mean
    the ``mean`` rich-pooling component uses -- so the factor path reads the
    same per-node summary the rest of the trunk does.
    """

    def __init__(self, d_model: int, dim: int) -> None:
        """Build the bottleneck, with `diag_w` and `bias_head` zero-initialized.

        Args:
            d_model: Width of the encoded per-token endpoint states.
            dim: Factor-embedding width `dim`.

        Raises:
            ValueError: If `dim` is not positive.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.w_z = nn.Linear(d_model, dim)
        self.diag_w = nn.Parameter(torch.zeros(dim))
        self.bias_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def _pool(self, tokens: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Masked mean over inner tokens (excludes BOS/EOS/padding)."""
        mask = _build_padding_mask(length, tokens.size(1))
        return masked_mean(tokens, inner_token_mask(tokens, mask))

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        len_a: torch.Tensor,
        len_b: torch.Tensor,
    ) -> NodeFactorOutputs:
        """Compute the node-factor decomposition for one pair batch.

        Runs its own fp32 island regardless of the caller's autocast state
        (the same discipline `FilmLogitConditioner`/`GatedCrossAttention`
        follow, `b0_v31.py`): `diag_w` starts at exactly `0`, and a bf16
        matmul could quantize that exact zero away.

        Args:
            tokens_a: Shape ``(B, T_a, d_model)`` endpoint-A token states.
            tokens_b: Shape ``(B, T_b, d_model)`` endpoint-B token states.
            len_a: Shape ``(B,)`` unpadded token counts for A.
            len_b: Shape ``(B,)`` unpadded token counts for B.

        Returns:
            The `NodeFactorOutputs` for this batch.
        """
        with torch.autocast(device_type=tokens_a.device.type, enabled=False):
            pooled_a = self._pool(tokens_a.float(), len_a)
            pooled_b = self._pool(tokens_b.float(), len_b)

            z_a = F.normalize(self.w_z(pooled_a), dim=-1)
            z_b = F.normalize(self.w_z(pooled_b), dim=-1)
            r = (z_a * self.diag_w * z_b).sum(dim=-1) / (float(self.dim) ** 0.5)
            b_a = self.bias_head(pooled_a).squeeze(-1)
            b_b = self.bias_head(pooled_b).squeeze(-1)
            residual = r + b_a + b_b
            return NodeFactorOutputs(z_a=z_a, z_b=z_b, r=r, b_a=b_a, b_b=b_b, residual=residual)


__all__ = ["NodeFactorBottleneck", "NodeFactorOutputs"]
