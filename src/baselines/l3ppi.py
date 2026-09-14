"""Paper-based L3-PPI head; real graphs are needed only to pretrain the surrogate.

Undirected edges are supplied in both directions. Missing implementation details
(shared-edge max, straight-through gates, sum readout) are reproduction choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from src.model.egostitch.classifier.layers import SiameseEncoder


class WeightedGIN(nn.Module):
    """Sparse epsilon-zero GIN with sum graph readout and scalar logits."""

    def __init__(
        self, input_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, layers) < 1:
            raise ValueError("GIN dimensions and depth must be positive")
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(layers)
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Score a disjoint batch of weighted graphs."""
        source, target = edge_index
        for layer in self.layers:
            neighbors = torch.zeros_like(x).index_add(
                0, target, x[source] * edge_weight[:, None]
            )
            x = layer(x + neighbors)
        pooled = x.new_zeros((num_graphs, x.shape[-1])).index_add(0, batch, x)
        return cast(torch.Tensor, self.readout(pooled).squeeze(-1))


def path_number_loss(
    probabilities: torch.Tensor, labels: torch.Tensor, gamma: float = 3.0
) -> torch.Tensor:
    """Equation (8), averaged over the AB/BA directions, without class weights."""
    if gamma <= 1:
        raise ValueError("gamma must exceed one")
    k = probabilities.shape[-1]
    counts = probabilities.sum(-1)
    positive = F.relu(k * (1 - 1 / gamma) - counts)
    negative = F.relu(counts - k / gamma)
    return torch.where(labels[:, None].bool(), positive, negative).mean(-1)


class L3PPI(nn.Module):
    """Frozen independent B0 encoder and a symmetric virtual L3 graph head."""

    template_edges: torch.Tensor
    path_edges: torch.Tensor

    def __init__(
        self,
        backbone_config: dict[str, Any],
        k: int = 16,
        surrogate_hidden: int = 64,
        gate_hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        max_length: int = 1024,
    ) -> None:
        super().__init__()
        if k < 1 or max_length < 1:
            raise ValueError("k and max_length must be positive")
        self.k = k
        self.max_length = max_length
        self.feature_dim = 2 * int(backbone_config["d_model"])
        reg = backbone_config["regularization"]
        self.encoder = SiameseEncoder(
            input_dim=int(backbone_config["input_dim"]),
            d_model=int(backbone_config["d_model"]),
            n_layers=int(backbone_config["encoder_layers"]),
            n_heads=int(backbone_config["n_heads"]),
            dropout=float(reg["dropout"]),
            token_dropout=float(reg.get("token_dropout", 0.0)),
            stochastic_depth=float(reg.get("stochastic_depth", 0.0)),
        )
        self.encoder.requires_grad_(False)
        self.surrogate = WeightedGIN(self.feature_dim, surrogate_hidden, layers, dropout)
        self.surrogate.requires_grad_(False)
        self.prompt = nn.Parameter(torch.zeros(k + 1, self.feature_dim))
        self.gate = WeightedGIN(self.feature_dim, gate_hidden, layers, dropout)
        # Nodes: u=0, private prompts=1..K, shared prompt=K+1, v=K+2.
        private = torch.arange(1, k + 1)
        edges = torch.stack(
            (
                torch.cat((torch.zeros(k, dtype=torch.long), private, torch.tensor([k + 1]))),
                torch.cat((private, torch.full((k,), k + 1), torch.tensor([k + 2]))),
            )
        )
        self.register_buffer("template_edges", torch.cat((edges, edges.flip(0)), dim=1))
        path = torch.tensor([[0, 1, 2], [1, 2, 3]])
        self.register_buffer("path_edges", torch.cat((path, path.flip(0)), dim=1))
        self.train(self.training)

    def train(self, mode: bool = True) -> L3PPI:
        """Keep frozen modules deterministic while training the prompt interface."""
        super().train(mode)
        self.encoder.eval()
        if not any(p.requires_grad for p in self.surrogate.parameters()):
            self.surrogate.eval()
        return self

    def load_backbone(self, checkpoint_payload: Mapping[str, Any]) -> None:
        """Copy only independent encoder weights from a published B0 checkpoint."""
        state = checkpoint_payload["model_state"]
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state, strict=True)

    @torch.no_grad()
    def encode(self, emb: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Cacheable masked mean/max pooling of frozen independent token states."""
        emb = emb[:, : self.max_length].float()
        length = length.clamp(max=emb.shape[1])
        if (length <= 0).any():
            raise ValueError("Endpoint sequences must contain at least one valid token")
        self.encoder.eval()
        x = self.encoder(emb, length).float()
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None]
        mean = (x * valid[..., None]).sum(1) / length[:, None]
        maximum = x.masked_fill(~valid[..., None], -torch.inf).amax(1)
        return torch.cat((mean, maximum), dim=-1)

    @torch.no_grad()
    def initialize_prompt(self, training_features: torch.Tensor) -> None:
        """Initialize global prompt nodes from training-only feature moments."""
        if training_features.ndim != 2 or training_features.shape[1] != self.feature_dim:
            raise ValueError("training_features must be [nodes, feature_dim]")
        mean = training_features.float().mean(0)
        std = training_features.float().std(0, correction=0)
        self.prompt.copy_(mean + 0.01 * std * torch.randn_like(self.prompt))

    @staticmethod
    def _batch_edges(edges: torch.Tensor, count: int, nodes: int) -> torch.Tensor:
        offsets = torch.arange(count, device=edges.device) * nodes
        return (edges[:, None, :] + offsets[None, :, None]).reshape(2, -1)

    def _ordered(
        self, a: torch.Tensor, b: torch.Tensor, all_open: bool, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = a.shape[0]
        if all_open:
            probabilities = a.new_ones((count, self.k))
            gates = probabilities
        else:
            private = self.prompt[: self.k].expand(count, -1, -1)
            shared = self.prompt[self.k].expand(count, self.k, -1)
            paths = torch.stack(
                (a[:, None].expand_as(private), private, shared, b[:, None].expand_as(private)),
                dim=2,
            ).reshape(-1, self.feature_dim)
            edges = self._batch_edges(self.path_edges, count * self.k, 4)
            path_batch = torch.arange(count * self.k, device=a.device).repeat_interleave(4)
            gate_logits = self.gate(
                paths, edges, a.new_ones(edges.shape[1]), path_batch, count * self.k
            ).reshape(count, self.k)
            probabilities = gate_logits.sigmoid()
            if self.training:
                uniform = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
                noisy_logits = gate_logits + uniform.log() - torch.log1p(-uniform)
                relaxed = (noisy_logits / temperature).sigmoid()
                gates = (relaxed > 0.5).to(relaxed.dtype) - relaxed.detach() + relaxed
            else:
                gates = (probabilities > 0.5).to(a.dtype)
        graph_nodes = torch.cat(
            (a[:, None], self.prompt[None].expand(count, -1, -1), b[:, None]), dim=1
        ).reshape(-1, self.feature_dim)
        weights = torch.cat((gates, gates, gates.amax(-1, keepdim=True)), dim=1)
        weights = torch.cat((weights, weights), dim=1).reshape(-1)
        edges = self._batch_edges(self.template_edges, count, self.k + 3)
        batch = torch.arange(count, device=a.device).repeat_interleave(self.k + 3)
        logits = self.surrogate(graph_nodes, edges, weights, batch, count)
        return logits, probabilities, gates

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        all_open: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return symmetric logits and per-direction path probabilities/activations."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        ab = self._ordered(a.float(), b.float(), all_open, temperature)
        ba = self._ordered(b.float(), a.float(), all_open, temperature)
        return (ab[0] + ba[0]) / 2, torch.stack((ab[1], ba[1]), 1), torch.stack((ab[2], ba[2]), 1)


def build_l3ppi(config: Mapping[str, Any]) -> L3PPI:
    """Build from the checkpoint's self-contained model.config mapping."""
    return L3PPI(**dict(config))
