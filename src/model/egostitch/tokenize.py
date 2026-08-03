"""Module 1 (Stage-1 form): Tokenize-lite — encoder + degree budget (spec Sec 1, Sec 13.2).

No VQ codebook, no BP affiliations, no code-stats head in Stage 1; ``e_u``
replaces ``(z_u, r_u)`` everywhere downstream. This module outputs the
lognormal NLL parameters (mu, log sigma) that `degree_nll` (`losses.py`)
supervises. The density-normalized raw softplus mean head
(``degree_mean_head`` / ``d_hat_raw``) was removed (three-component refactor
design §12 P2a dead-code sweep): its only consumer was
``EgoStitchStage1.d_hat()``, itself only called by the retired frozen-s0
``pair_outputs``/``self_outputs`` decision-fusion methods deleted with
`decision.py` (design §9). ``deg_mu``/``deg_log_sigma`` are unrelated and
stay -- they are `degree_nll`'s live inputs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.layers import build_mlp


class TokenizeOut(NamedTuple):
    """Per-node Tokenize-lite outputs.

    Attributes:
        e: Shape ``(B, d_z)`` encoder embeddings ``e_u``.
        deg_mu: Shape ``(B,)`` lognormal location parameter.
        deg_log_sigma: Shape ``(B,)`` lognormal log-scale parameter.
    """

    e: torch.Tensor
    deg_mu: torch.Tensor
    deg_log_sigma: torch.Tensor


class TokenizeLite(nn.Module):
    """Encoder ``e_u = MLP_2(d -> d_z)(x_u)`` + degree-distribution head.

    The 2-head lognormal parameterization
    ``(mu, log sigma) = MLP_2(d + d_z -> 2)([x_u; e_u])`` (spec Sec 1:
    "the table above shows the mean head") feeds `degree_nll` directly.
    """

    def __init__(self, config: EgoStitchConfig) -> None:
        """Build the encoder and degree-distribution head.

        Args:
            config: The pinned Stage-1 configuration.
        """
        super().__init__()
        self.encoder = build_mlp(config.input_dim, config.d_h, config.d_z)
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
        dist = self.degree_dist_head(xe)
        return TokenizeOut(
            e=e,
            deg_mu=dist[:, 0],
            deg_log_sigma=dist[:, 1],
        )
