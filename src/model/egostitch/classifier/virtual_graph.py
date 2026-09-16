"""Fixed training-graph coarsening with supervised, endpoint-only attachments."""

from __future__ import annotations

from typing import cast

import torch
from scipy.sparse import csr_matrix
from torch import nn
from torch.nn import functional as F


class VirtualGraphGenerator(nn.Module):
    """Match protein residues to fixed block prototypes and count pair topology."""

    coord_mean: torch.Tensor
    coord_std: torch.Tensor
    prototypes: torch.Tensor
    multiplicity: torch.Tensor
    adjacency: torch.Tensor

    def __init__(
        self,
        d_model: int,
        *,
        k: int = 256,
        d_z: int = 128,
        heads: int = 4,
        coord_mean: torch.Tensor | None = None,
        coord_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if k < 1 or d_z < 1 or heads < 1 or d_z % heads:
            raise ValueError("positive dimensions and d_z divisible by heads are required")
        self.k, self.heads = k, heads
        self.register_buffer("prototypes", torch.zeros(k, d_model))
        self.register_buffer("multiplicity", torch.ones(k))
        self.register_buffer("adjacency", torch.zeros(k, k))
        self.residue_projection = nn.Linear(d_model, d_z)
        self.attachment = nn.Sequential(nn.Linear(4 * d_z, d_z), nn.GELU(), nn.Linear(d_z, 1))
        nn.init.zeros_(cast(nn.Linear, self.attachment[-1]).weight)
        nn.init.zeros_(cast(nn.Linear, self.attachment[-1]).bias)
        self.slot_bias = nn.Parameter(torch.full((k,), float(torch.tensor(0.01).expm1().log())))
        self.distance_head = nn.Linear(3, 4)
        self.register_buffer(
            "coord_mean",
            torch.zeros(9) if coord_mean is None else coord_mean.detach().float().clone(),
        )
        self.register_buffer(
            "coord_std", torch.ones(9) if coord_std is None else coord_std.detach().float().clone()
        )
        if self.coord_mean.shape != (9,) or self.coord_std.shape != (9,):
            raise ValueError("virtual graph requires nine v3 coordinate statistics")
        if not torch.isfinite(self.coord_mean).all() or not torch.isfinite(self.coord_std).all():
            raise ValueError("coordinate statistics must be finite")
        if (self.coord_std <= 0).any():
            raise ValueError("coordinate standard deviations must be positive")
        self._telemetry: dict[str, torch.Tensor] = {}

    def reset_telemetry(self) -> None:
        """Start a new evaluation-row telemetry window."""
        self._telemetry.clear()

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp(1e-7, 1 - 1e-7)
        return -(p * p.log() + (1 - p) * (1 - p).log())

    @torch.no_grad()
    def telemetry(self, *, reset: bool = False) -> dict[str, torch.Tensor]:
        """Additive attachment statistics; reduce ranks before taking means."""
        result = {name: value.clone() for name, value in self._telemetry.items()}
        for suffix in ("sum", "entropy_sum"):
            result.setdefault(f"attachment_{suffix}", self.prototypes.new_zeros(self.k))
        result.setdefault("attachment_count", self.prototypes.new_zeros(()))
        if reset:
            self.reset_telemetry()
        return result

    @torch.no_grad()
    def _record(self, a: torch.Tensor, b: torch.Tensor) -> None:
        values = torch.cat((a, b))
        additions = {
            "attachment_sum": values.sum(0),
            "attachment_entropy_sum": self._entropy(values).sum(0),
            "attachment_count": values.new_tensor(values.size(0)),
        }
        for key, value in additions.items():
            self._telemetry[key] = self._telemetry.get(key, torch.zeros_like(value)) + value

    @torch.no_grad()
    def initialise(
        self,
        pooled_states: torch.Tensor,
        assignments: torch.Tensor,
        adjacency: torch.Tensor | csr_matrix,
    ) -> None:
        """Install masked-mean prototypes, loopless densities and count biases.

        Input rows cover precisely the feature-present legal training graph.
        Per-block mean neighbor counts determine the initial count bias.
        """
        device = self.prototypes.device
        pools = pooled_states.to(device=device, dtype=torch.float32)
        labels = assignments.to(device=device, dtype=torch.long)
        if pools.shape != (labels.numel(), self.prototypes.size(1)) or labels.ndim != 1:
            raise ValueError("prototype states and assignments must have matching rows")
        if labels.numel() == 0 or (labels < 0).any() or (labels >= self.k).any():
            raise ValueError("assignments must index the configured blocks")
        sizes = torch.bincount(labels, minlength=self.k).float()
        if (sizes == 0).any():
            raise ValueError("virtual graph initialisation requires every cluster to be nonempty")
        if adjacency.shape != (labels.numel(), labels.numel()):
            raise ValueError("training adjacency must match prototype rows")
        centroids = torch.zeros_like(self.prototypes)
        centroids.index_add_(0, labels, pools)
        self.prototypes.copy_(centroids / sizes[:, None])
        self.multiplicity.copy_(sizes)
        if isinstance(adjacency, torch.Tensor):
            row, col = adjacency.nonzero(as_tuple=True)
        else:
            row_np, col_np = adjacency.nonzero()
            row, col = torch.as_tensor(row_np), torch.as_tensor(col_np)
        row, col = row.to(device), col.to(device)
        keep = row != col
        ids = labels[row[keep]] * self.k + labels[col[keep]]
        edge_counts = torch.bincount(ids, minlength=self.k**2).reshape(self.k, self.k).float()
        if not torch.equal(edge_counts, edge_counts.T):
            raise ValueError("training adjacency must be undirected")
        pair_counts = sizes[:, None] * sizes[None, :] - torch.diag(sizes)
        self.adjacency.copy_(edge_counts / pair_counts.clamp_min(1))
        mean_counts = (edge_counts.sum(0) / labels.numel()).clamp_min(0.01)
        self.slot_bias.copy_(mean_counts + torch.log(-torch.expm1(-mean_counts)))

    @staticmethod
    def pool(encoded: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Masked-mean encoder states for fixed semantic prototype construction."""
        keep = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
        if lengths is not None:
            keep = torch.arange(encoded.size(1), device=encoded.device)[None] < lengths[:, None]
        return (
            encoded.float().masked_fill(~keep[..., None], 0).sum(1)
            / keep.sum(1).clamp_min(1)[:, None]
        )

    def forward(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor | None,
        lengths_b: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Emit six standardized continuous coordinates and distance logits."""
        with torch.autocast(device_type=encoded_a.device.type, enabled=False):
            nu = self.attachment_counts(encoded_a, lengths_a)
            nv = self.attachment_counts(encoded_b, lengths_b)
            a, b = nu / self.multiplicity, nv / self.multiplicity
            if not self.training:
                self._record(a, b)
            c = self.count(a, b)
            raw = torch.stack(
                (
                    c["degree_u"].log1p(),
                    c["degree_v"].log1p(),
                    c["common"].log1p(),
                    c["jaccard"],
                    c["l3"].log1p(),
                    c["l3_density"],
                ),
                -1,
            )
            z = (raw - self.coord_mean[:6]) / self.coord_std[:6]
            distance = self.distance_head(
                torch.stack(
                    (
                        c["common"].log1p(),
                        c["l3"].log1p(),
                        c["degree_u"].log1p() + c["degree_v"].log1p(),
                    ),
                    -1,
                )
            )
            return {
                "endpoint_u": z[:, :1],
                "endpoint_v": z[:, 1:2],
                "relation": z[:, 2:6],
                "distance_logits": distance,
                "attachment_counts_u": nu,
                "attachment_counts_v": nv,
            }

    def count(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        """Loopless lifted-graph counts with distinct within-block node pairs."""
        with torch.autocast(device_type=a.device.type, enabled=False):
            a, b = a.float(), b.float()
            m = self.multiplicity.float()
            pairs = m[:, None] * m[None, :] - torch.diag(m)
            edges = pairs * self.adjacency.float()
            du, dv = a @ m, b @ m
            common = (a * b * m).sum(-1)
            l3 = (a @ edges * b).sum(-1)
            return {
                "degree_u": du,
                "degree_v": dv,
                "common": common,
                "jaccard": common / (du + dv - common).clamp_min(1e-8),
                "l3": l3,
                "l3_density": l3 / (du * dv).clamp_min(1e-8),
            }

    def attach(self, encoded: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Per-protein block probabilities, independent of the queried partner."""
        return self.attachment_counts(encoded, lengths) / self.multiplicity

    def attachment_counts(
        self, encoded: torch.Tensor, lengths: torch.Tensor | None
    ) -> torch.Tensor:
        """Shared normalized Q/K matching, with no independent Q/K projections."""
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            states = self.residue_projection(encoded.float())
            prototypes = self.residue_projection(self.prototypes.float())
            batch, residues, width = states.shape
            q = F.normalize(prototypes.reshape(self.k, self.heads, -1), dim=-1)
            kv = F.normalize(states.reshape(batch, residues, self.heads, -1), dim=-1)
            scores = 4.0 * torch.einsum("khd,blhd->bhkl", q, kv)
            if lengths is not None:
                if (lengths <= 0).any() or (lengths > residues).any():
                    raise ValueError("attachment requires nonempty valid residue lengths")
                pad = torch.arange(residues, device=encoded.device)[None] >= lengths[:, None]
                scores = scores.masked_fill(pad[:, None, None, :], -torch.inf)
            output = torch.einsum("bhkl,blhd->bkhd", scores.softmax(-1), kv).reshape(
                batch, self.k, width
            )
            p = q.reshape(self.k, width)[None].expand(batch, -1, -1)
            features = torch.cat((output, p, output * p, (output - p).abs()), -1)
            raw = cast(torch.Tensor, self.attachment(features)).squeeze(-1) + self.slot_bias
            return torch.minimum(F.softplus(raw), self.multiplicity)

    @staticmethod
    def attachment_loss_rows(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Equal-weight nonzero/zero block Huber means for each endpoint row."""
        errors = F.huber_loss(predicted.float().log1p(), target.float().log1p(), reduction="none")
        positive = target > 0
        positive_n, zero_n = positive.sum(-1), (~positive).sum(-1)
        positive_mean = (errors * positive).sum(-1) / positive_n.clamp_min(1)
        zero_mean = (errors * ~positive).sum(-1) / zero_n.clamp_min(1)
        groups = (positive_n > 0).float() + (zero_n > 0).float()
        return (positive_mean + zero_mean) / groups.clamp_min(1)
