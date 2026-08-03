"""Executable CAZI-MBN teacher/student modules.

The layer definitions mirror the released CAZI-MBN repository.  This module is
kept separate because the release has no runnable experiment entry point and its
teacher/student glue contains incompatible tensor and constructor contracts.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGPooling, TransformerConv, global_mean_pool


class GraphTransformer(nn.Module):
    """The three-layer TransformerConv block released by CAZI-MBN."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 3,
        heads: int = 6,
        dropout: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphTransformer requires at least two layers")
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(
            TransformerConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
                bias=bias,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        for _ in range(1, num_layers - 1):
            self.layers.append(
                TransformerConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    bias=bias,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.layers.append(
            TransformerConv(
                hidden_channels * heads,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
                bias=bias,
            )
        )
        self.norms.append(nn.LayerNorm(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes over one CAZI network layer."""
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = F.relu(norm(layer(x, edge_index)), inplace=True)
        return x


class SAGPoolReadoutEdge(nn.Module):
    """Released CAZI edge-level SAGPool readout."""

    def __init__(self, in_channels: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.sagpool = SAGPooling(in_channels, ratio)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Pool concatenated endpoint states to one graph summary."""
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, batch=batch)
        edge_attr = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        edge_batch = batch[edge_index[0]]
        return cast(torch.Tensor, global_mean_pool(edge_attr, edge_batch))


class Discriminator(nn.Module):
    """Released CAZI bilinear graph-summary discriminator."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.PReLU()
        nn.init.kaiming_uniform_(self.bilinear.weight.data, a=0.01)
        if self.bilinear.bias is not None:
            self.bilinear.bias.data.zero_()

    def forward(
        self,
        summary: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive then negative discriminator logits."""
        positive = self.activation(self.norm(positive))
        negative = self.activation(self.norm(negative))
        pos_summary = summary.expand_as(positive)
        neg_summary = summary.expand_as(negative)
        pos_score = self.bilinear(positive, pos_summary).squeeze(1)
        neg_score = self.bilinear(negative, neg_summary).squeeze(1)
        return torch.cat((pos_score, neg_score), dim=0)


class MoEClassifier(nn.Module):
    """Released CAZI mixture-of-experts classifier."""

    def __init__(
        self,
        fused_dim: int,
        output_dim: int,
        *,
        num_experts: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gating_network = nn.Linear(fused_dim, num_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Mix expert outputs with a learned softmax gate."""
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        gate_weights = F.softmax(self.gating_network(h), dim=1).unsqueeze(-1)
        return (expert_outputs * gate_weights).sum(dim=1)


class MoE(nn.Module):
    """Endpoint-fusion wrapper around the released MoE classifier."""

    def __init__(
        self,
        in1_dim: int,
        in2_dim: int,
        out_dim: int,
        *,
        num_experts: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in1_dim = in1_dim
        self.in2_dim = in2_dim
        self.classifier = MoEClassifier(
            in1_dim + in2_dim,
            out_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Classify one batch of endpoint pairs."""
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError("MoE endpoint tensors must be two-dimensional")
        if x1.shape[1] != self.in1_dim or x2.shape[1] != self.in2_dim:
            raise ValueError(
                f"MoE endpoint dimensions must be ({self.in1_dim}, {self.in2_dim}), "
                f"got ({x1.shape[1]}, {x2.shape[1]})"
            )
        return cast(torch.Tensor, self.classifier(torch.cat((x1, x2), dim=1)))


class CAZITeacher(nn.Module):
    """Topology-aware CAZI teacher for one local benchmark interaction layer."""

    def __init__(
        self,
        num_nodes: int,
        sequence_dim: int,
        *,
        topology_dim: int = 128,
        latent_dim: int = 32,
        network_layers: int = 1,
        heads: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = GraphTransformer(
            topology_dim,
            topology_dim * 2,
            topology_dim,
            heads=heads,
        )
        self.readout = SAGPoolReadoutEdge(topology_dim)
        self.discriminator = Discriminator(topology_dim * 2)
        self.consensus = nn.Parameter(torch.empty(num_nodes, topology_dim))
        nn.init.xavier_normal_(self.consensus)
        self.latent_projection = nn.Linear(topology_dim, latent_dim)
        endpoint_dim = latent_dim + sequence_dim
        self.classifier = MoE(
            endpoint_dim,
            endpoint_dim,
            network_layers,
            num_experts=network_layers,
        )

    def graph_objective(
        self,
        topology: torch.Tensor,
        positive_edge_index: torch.Tensor,
        negative_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return released discriminator and consensus losses."""
        positive_h = self.encoder(topology, positive_edge_index)
        negative_h = self.encoder(topology, negative_edge_index)
        positive_edge_h = torch.cat(
            (positive_h[positive_edge_index[0]], positive_h[positive_edge_index[1]]), dim=1
        )
        negative_edge_h = torch.cat(
            (negative_h[negative_edge_index[0]], negative_h[negative_edge_index[1]]), dim=1
        )
        summary = torch.sigmoid(self.readout(positive_h, positive_edge_index))
        logits = self.discriminator(summary, positive_edge_h, negative_edge_h)
        labels = torch.cat(
            (
                torch.ones(positive_edge_h.shape[0], device=topology.device),
                torch.zeros(negative_edge_h.shape[0], device=topology.device),
            )
        )
        discriminator_loss = F.binary_cross_entropy_with_logits(logits, labels)
        consensus_loss = (
            1.0
            - F.cosine_similarity(self.consensus, positive_h, dim=1).mean()
            + F.cosine_similarity(self.consensus, negative_h, dim=1).mean()
        )
        return discriminator_loss, consensus_loss

    def pair_logits(
        self,
        sequence: torch.Tensor,
        u_idx: torch.Tensor,
        v_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Classify pairs from sequence and learned topology consensus states."""
        topology_latent = self.distilled_latent()
        u = torch.cat((sequence[u_idx], topology_latent[u_idx]), dim=1)
        v = torch.cat((sequence[v_idx], topology_latent[v_idx]), dim=1)
        return cast(torch.Tensor, self.classifier(u, v).squeeze(1))

    def distilled_latent(self) -> torch.Tensor:
        """Return the teacher's 32-dimensional node target."""
        return cast(torch.Tensor, self.latent_projection(self.consensus))


class CAZIStudent(nn.Module):
    """Sequence-only CAZI student used for held-out-node inference."""

    def __init__(
        self,
        sequence_dim: int,
        *,
        latent_dim: int = 32,
        network_layers: int = 1,
    ) -> None:
        super().__init__()
        self.latent_projection = nn.Linear(sequence_dim, latent_dim)
        self.classifier = MoE(
            latent_dim,
            latent_dim,
            network_layers,
            num_experts=network_layers,
        )

    def node_latent(self, sequence: torch.Tensor) -> torch.Tensor:
        """Project sequence-only node inputs to the distilled latent space."""
        return cast(torch.Tensor, self.latent_projection(sequence))

    def pair_logits(
        self,
        sequence: torch.Tensor,
        u_idx: torch.Tensor,
        v_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Classify pairs without graph inputs."""
        latent = self.node_latent(sequence)
        return cast(torch.Tensor, self.classifier(latent[u_idx], latent[v_idx]).squeeze(1))


__all__ = ["CAZIStudent", "CAZITeacher", "GraphTransformer", "MoE"]
