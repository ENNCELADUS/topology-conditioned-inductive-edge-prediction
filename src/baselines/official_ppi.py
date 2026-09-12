"""Adapters around the vendored official TUnA and PPITrans model classes."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src.data.packed_features import PackedFeatureTable


class OfficialPPI(nn.Module):
    """Feature-controlled original classifiers with one scalar edge logit."""

    def __init__(
        self,
        method: str,
        input_dim: int = 1536,
        max_length: int = 1024,
        hidden_dim: int = 64,
        layers: int = 1,
        heads: int = 8,
        dropout: float = 0.2,
        rffs: int = 4096,
        seed: int = 0,
        batch_size: int = 4,
    ) -> None:
        super().__init__()
        self.core: Any
        self.method = method
        self.max_length = max_length
        self.batch_size = batch_size
        if method == "tuna":
            from src.baselines.vendor.tuna.model import (
                InterEncoder,
                IntraEncoder,
                ProteinInteractionNet,
            )
            from src.baselines.vendor.uadl.classic_rffs import VanillaRFFLayer

            args = (
                input_dim,
                hidden_dim,
                layers,
                heads,
                hidden_dim * 4,
                dropout,
                "swish",
                torch.device("cpu"),
            )
            gp = VanillaRFFLayer(
                hidden_dim,
                rffs,
                gp_cov_momentum=-1,
                gp_ridge_penalty=1.0,
                likelihood="binary_logistic",
                random_seed=seed,
            )
            self.core = ProteinInteractionNet(  # type: ignore[no-untyped-call]
                IntraEncoder(*args),  # type: ignore[no-untyped-call]
                InterEncoder(*args),  # type: ignore[no-untyped-call]
                gp, torch.device("cpu")
            )
            # Match the official Trainer's initialization of trainable matrices.
            for parameter in self.core.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        elif method == "ppitrans":
            from src.baselines.vendor.ppitrans.decoder import Decoder
            from src.baselines.vendor.ppitrans.encoder import Encoder

            ppi_args = SimpleNamespace(
                emb_dim=input_dim,
                hid_dim=hidden_dim,
                max_len=max_length,
                dropout=dropout,
                trans_layers=layers,
            )
            self.encoder = Encoder(ppi_args)  # type: ignore[no-untyped-call]
            self.decoder = Decoder(ppi_args)  # type: ignore[no-untyped-call]
        else:
            raise ValueError(f"unknown official PPI method: {method}")

    def train(self, mode: bool = True) -> OfficialPPI:
        """Use the explicitly train-fitted covariance, including after reload."""
        if self.method == "tuna" and not mode:
            self.core.gp_layer.fitted = True
        super().train(mode)
        return self

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        la: torch.Tensor,
        lb: torch.Tensor,
        *,
        update_precision: bool = False,
    ) -> torch.Tensor:
        """Return original mean-field logits at inference, unadjusted during fitting."""
        if self.method == "ppitrans":
            x, lx, y, ly = self.encoder(a, la, b, lb)
            logits = self.decoder(x, lx, y, ly)["logits"]
            return cast(torch.Tensor, logits[:, 1] - logits[:, 0])
        self.core.device = a.device
        if self.training or update_precision:
            return cast(
                torch.Tensor,
                self.core.forward(a, b, la, lb, a.shape[1], b.shape[1], update_precision, True),
            ).reshape(-1)
        logits, variance = self.core.forward(a, b, la, lb, a.shape[1], b.shape[1], True, False)
        # Official mean-field rule, with explicit per-pair shape (no N x N broadcasting).
        return cast(
            torch.Tensor, logits.reshape(-1) / torch.sqrt(1 + math.pi / 8 * variance.reshape(-1))
        )


class PairFeatures:
    """Use the same existing BF16 token pack as the current benchmark models."""

    def __init__(self, pack: Path, device: torch.device, max_length: int) -> None:
        self.table = PackedFeatureTable.from_pack(pack, device)
        self.position = self.table.manifest.node_index()
        self.max_length = max_length

    def batch(self, pairs: list[tuple[str, str]]) -> tuple[torch.Tensor, ...]:
        """Gather token batches without fitting anything on the scoring universe."""
        u = torch.tensor([self.position[a] for a, _ in pairs], device=self.table.tokens.device)
        v = torch.tensor([self.position[b] for _, b in pairs], device=self.table.tokens.device)
        bound_a = min(self.max_length, int(self.table.lengths[u].max()))
        bound_b = min(self.max_length, int(self.table.lengths[v].max()))
        a, la = self.table.gather_nodes(u, bound_a)
        b, lb = self.table.gather_nodes(v, bound_b)
        return a.float(), b.float(), la.clamp_max(bound_a), lb.clamp_max(bound_b)


def score_pairs(
    model: OfficialPPI, features: PairFeatures, pairs: list[tuple[str, str]]
) -> NDArray[np.float32]:
    """Score a row universe without touching GP precision or fitting parameters."""
    model.eval()
    output = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(pairs), model.batch_size):
            batch = pairs[start : start + model.batch_size]
            output[start : start + len(batch)] = model(*features.batch(batch)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("non-finite official PPI scores")
    return output


def build_official_ppi(config: dict[str, Any]) -> OfficialPPI:
    """Construct a classifier from its embedded checkpoint configuration."""
    return OfficialPPI(**config)
