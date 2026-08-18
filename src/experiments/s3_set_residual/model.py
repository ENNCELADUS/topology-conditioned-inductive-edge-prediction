"""S3 model: a set-conditioned residual over frozen B0 pair logits.

Spec: ``docs/tmp/s3_set_residual_plan.md``. Three modes probe whether *set*
context (as opposed to pointwise-per-node or pairwise-only context) helps
predict ``edge(u,v)`` beyond the frozen base scorer:

- ``"res"``: full set attention (`_SetBackbone`, reused from
  `src.experiments.s2_latent_topology.model`) -- every node's hidden state can
  depend on the whole conditioning set.
- ``"diag"``: architecturally identical to ``"res"`` (same module, same
  parameter count) but attention is restricted so each node only ever attends
  to itself. `_SetBackbone`'s own `MAB`/`PMA` mask argument is a *key-padding*
  mask shared across all query rows (see
  ``src/vendor/set_transformer_official/modules.py``) -- it cannot express a
  per-query diagonal restriction, so this is *not* implemented by passing a
  different mask. Instead each node is processed as its own size-1 "set": the
  backbone is called on a ``(B*N, 1, D)`` batch of singleton sets, so the only
  key any query can ever see is itself. Same weights, same forward path, zero
  new modules -- just a different batching of the same call.
- ``"pair"``: no set attention at all. The backbone is replaced by a per-node
  MLP (`_matched_node_mlp`) sized so its parameter count sits within 10% of
  the backbone's, so any "res" vs "pair" gap is attributable to set attention,
  not to model capacity.

All three modes share an input projection (`d_in -> d_model`) feeding whichever
per-node encoder is active, a raw-feature projection ``p = Linear(x)`` that
never depends on ``x_set``, and a pair head that reads a symmetric function of
``(h_i, h_j, p_i, p_j)``. The pair head's final linear layer is zero-initialized
(weight and bias), so at construction ``forward`` returns exactly zero for
every pair -- the zero-init contract: total logit (base + delta) starts out
equal to the base logit alone.

For the default config (``d_in=1536, d_model=384, sab_layers=4, heads=4,
pma_seeds=4, p_dim=128, head_hidden=256``): the `_SetBackbone` used by
``"res"``/``"diag"`` has 3,845,376 parameters; the parameter-matched per-node
MLP used by ``"pair"`` has 3,845,384 (ratio 1.0000, well inside the 10%
contract asserted in `test_s3_residual_model.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.experiments.s2_latent_topology.model import _SetBackbone

__all__ = ["ResidualConfig", "SetResidualModel"]


@dataclass(frozen=True)
class ResidualConfig:
    """Hyperparameters for `SetResidualModel`.

    Backbone dims (`d_model`, `sab_layers`, `heads`, `pma_seeds`) mirror
    `src.experiments.s2_latent_topology.model.SetDenoiser`/`DetTwin`'s
    defaults, which size the same `_SetBackbone`.

    Attributes:
        mode: ``"res"`` (full set attention), ``"pair"`` (no set attention,
            parameter-matched per-node MLP), or ``"diag"`` (same module and
            parameter count as ``"res"``, attention restricted to self).
        d_in: Raw node feature width.
        d_model: Width of the input projection and of the set backbone /
            matched per-node MLP.
        sab_layers: Number of `MAB` self-attention layers in the backbone.
        heads: Attention heads per `MAB`/`PMA`.
        pma_seeds: Number of PMA seed vectors in the backbone's global read.
        p_dim: Width of the raw-feature projection ``p = Linear(x)``.
        head_hidden: Hidden width of the pair head MLP.
    """

    mode: Literal["res", "pair", "diag"]
    d_in: int = 1536
    d_model: int = 384
    sab_layers: int = 4
    heads: int = 4
    pma_seeds: int = 4
    p_dim: int = 128
    head_hidden: int = 256


def _param_count(module: nn.Module) -> int:
    """Total element count over all parameters of `module`."""
    return sum(p.numel() for p in module.parameters())


def _matched_node_mlp(d_model: int, target_params: int) -> nn.Module:
    """Per-node MLP whose parameter count is closest to `target_params`.

    Fixed shape ``Linear(d_model, h) -> GELU -> Linear(h, d_model)``; solving
    ``2*h*d_model + h + d_model = target_params`` for the bottleneck width
    ``h`` (rounded to the nearest positive integer) lands within a handful of
    parameters of `target_params` for any ``d_model``, comfortably inside the
    10% contract `test_s3_residual_model.py` asserts.

    Args:
        d_model: Input and output width (matches the backbone it replaces).
        target_params: Parameter count to match (the backbone's).

    Returns:
        A two-layer `nn.Sequential` mapping ``(..., d_model) -> (..., d_model)``.
    """
    denom = 2 * d_model + 1
    hidden = max(1, round((target_params - d_model) / denom))
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))


class SetResidualModel(nn.Module):
    """Set-conditioned residual over frozen base logits, in three modes.

    See the module docstring for the ``"res"``/``"diag"``/``"pair"`` designs
    and the default-config parameter counts.
    """

    def __init__(self, cfg: ResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.d_in, cfg.d_model)
        self.p_proj = nn.Linear(cfg.d_in, cfg.p_dim)

        if cfg.mode in ("res", "diag"):
            self.backbone: nn.Module = _SetBackbone(
                in_dim=cfg.d_model,
                out_dim=cfg.d_model,
                width=cfg.d_model,
                sab_layers=cfg.sab_layers,
                heads=cfg.heads,
                pma_seeds=cfg.pma_seeds,
            )
        elif cfg.mode == "pair":
            reference = _SetBackbone(
                in_dim=cfg.d_model,
                out_dim=cfg.d_model,
                width=cfg.d_model,
                sab_layers=cfg.sab_layers,
                heads=cfg.heads,
                pma_seeds=cfg.pma_seeds,
            )
            self.backbone = _matched_node_mlp(cfg.d_model, _param_count(reference))
        else:
            raise ValueError(f"unknown mode: {cfg.mode!r}")

        combo_dim = 2 * cfg.d_model + 2 * cfg.p_dim
        self.pair_hidden = nn.Sequential(
            nn.Linear(combo_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.GELU(),
        )
        self.pair_out = nn.Linear(cfg.head_hidden, 1)
        nn.init.zeros_(self.pair_out.weight)
        nn.init.zeros_(self.pair_out.bias)

    def _encode(
        self, x: torch.Tensor, x_set: torch.Tensor | None, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-node hidden states ``h``, dispatched by `cfg.mode`.

        ``"pair"`` consumes `x` alone (its per-node MLP has no set context to
        override); ``"res"``/``"diag"`` consume `x_set` when given, else `x`.
        """
        if self.cfg.mode == "pair":
            h: torch.Tensor = self.backbone(self.input_proj(x))
            return h
        backbone_input = self.input_proj(x if x_set is None else x_set)
        if self.cfg.mode == "res":
            out: torch.Tensor = self.backbone(backbone_input, mask)
            return out
        return self._diag_encode(backbone_input, mask)

    def _diag_encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the backbone on each node as its own size-1 set.

        A size-1 set has exactly one key -- itself -- so every node's output
        is a pointwise function of only its own (projected) features,
        regardless of what any other node's features are. Same backbone
        weights and call as ``"res"``; only the batching differs.

        Args:
            tokens: Shape ``(B, N, d_model)`` projected backbone input.
            mask: Shape ``(B, N)`` real node existence in ``{0, 1}``.

        Returns:
            Shape ``(B, N, d_model)``, padded rows zeroed.
        """
        b, n, d = tokens.shape
        flat = tokens.reshape(b * n, 1, d)
        flat_mask = torch.ones(b * n, 1, device=tokens.device, dtype=mask.dtype)
        out: torch.Tensor = self.backbone(flat, flat_mask).reshape(b, n, -1)
        return out * mask.unsqueeze(-1).to(out.dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pairs: torch.Tensor,
        x_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the residual logit delta for each queried pair.

        Args:
            x: Shape ``(B, N, d_in)`` raw node features; always feeds the
                raw-feature projection ``p``, and feeds the backbone too
                unless overridden by `x_set`.
            mask: Shape ``(B, N)`` node existence in ``{0, 1}``.
            pairs: Shape ``(P, 3)`` long rows ``[set_row, i, j]`` indexing
                into `x`/`mask`'s batch and node axes.
            x_set: Shape ``(B, N, d_in)`` or ``None``. When given, overrides
                only the backbone input (e.g. a SHUF ablation) -- ``p`` still
                reads `x`. Ignored in ``"pair"`` mode, which has no backbone.

        Returns:
            Shape ``(P,)`` fp32 residual logit delta, zero at construction
            (zero-init contract).
        """
        h = self._encode(x, x_set, mask)
        p = self.p_proj(x)

        set_row, i, j = pairs[:, 0], pairs[:, 1], pairs[:, 2]
        h_i, h_j = h[set_row, i], h[set_row, j]
        p_i, p_j = p[set_row, i], p[set_row, j]
        combo = torch.cat((h_i + h_j, h_i * h_j, p_i + p_j, p_i * p_j), dim=-1)
        delta: torch.Tensor = self.pair_out(self.pair_hidden(combo)).squeeze(-1)
        return delta.to(torch.float32)
