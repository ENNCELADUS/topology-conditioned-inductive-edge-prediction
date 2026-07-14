"""Module 1 (Stage-1 form): Tokenize-lite — encoder + degree budget (spec Sec 1, Sec 13.2).

No VQ codebook, no BP affiliations, no code-stats head in Stage 1; ``e_u``
replaces ``(z_u, r_u)`` everywhere downstream. The degree head is
density-normalized by the caller (``rho_eval / rho_train`` lives in the model,
spec Sec 9.3); this module outputs the raw softplus mean head plus the
lognormal NLL parameters (mu, log sigma).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.layers import build_mlp


class TokenizeOut(NamedTuple):
    """Per-node Tokenize-lite outputs.

    Attributes:
        e: Shape ``(B, d_z)`` encoder embeddings ``e_u``.
        d_hat_raw: Shape ``(B,)`` raw (density-unnormalized) degree budget mean.
        deg_mu: Shape ``(B,)`` lognormal location parameter.
        deg_log_sigma: Shape ``(B,)`` lognormal log-scale parameter.
    """

    e: torch.Tensor
    d_hat_raw: torch.Tensor
    deg_mu: torch.Tensor
    deg_log_sigma: torch.Tensor


class TokenizeLite(nn.Module):
    """Encoder ``e_u = MLP_2(d -> d_z)(x_u)`` + degree-budget heads.

    ``d_hat_raw = softplus(MLP_2(d + d_z -> 1)([x_u; e_u]))`` (spec Sec 13.2 —
    ``e_u`` substitutes the codebook code) and the 2-head lognormal
    parameterization ``(mu, log sigma) = MLP_2(d + d_z -> 2)([x_u; e_u])``
    (spec Sec 1: "the table above shows the mean head").
    """

    def __init__(self, config: EgoStitchConfig) -> None:
        """Build the encoder and degree heads.

        Args:
            config: The pinned Stage-1 configuration.
        """
        super().__init__()
        self.encoder = build_mlp(config.input_dim, config.d_h, config.d_z)
        self.degree_mean_head = build_mlp(config.input_dim + config.d_z, config.d_h, 1)
        self.degree_dist_head = build_mlp(config.input_dim + config.d_z, config.d_h, 2)

    def forward(self, x: torch.Tensor) -> TokenizeOut:
        """Encode a node batch.

        Args:
            x: Shape ``(B, d)`` frozen F0 features.

        Returns:
            The `TokenizeOut` bundle.
        """
        e = self.encoder(x)
        xe = torch.cat([x, e], dim=-1)
        d_hat_raw = F.softplus(self.degree_mean_head(xe)).squeeze(-1)
        dist = self.degree_dist_head(xe)
        return TokenizeOut(
            e=e,
            d_hat_raw=d_hat_raw,
            deg_mu=dist[:, 0],
            deg_log_sigma=dist[:, 1],
        )
