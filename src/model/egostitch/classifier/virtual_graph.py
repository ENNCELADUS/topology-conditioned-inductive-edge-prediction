"""A learned training-graph coarsening behind the compact coordinate interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from scipy.sparse import csr_matrix
from torch import nn
from torch.nn import functional as F


class VirtualGraphGenerator(nn.Module):
    """Attend from shared coarse nodes to residues and count pair topology."""

    coord_mean: torch.Tensor
    coord_std: torch.Tensor

    def __init__(
        self,
        d_model: int,
        *,
        k: int = 64,
        d_z: int = 128,
        heads: int = 4,
        gate_bias: float = 3.0,
        coord_mean: torch.Tensor | None = None,
        coord_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.P = nn.Parameter(torch.randn(k, d_z) / d_z**0.5)
        self.residue_projection = nn.Linear(d_model, d_z)
        self.attention = nn.MultiheadAttention(d_z, heads, batch_first=True)
        self.readout_bias = nn.Parameter(torch.zeros(k))
        self.m_raw = nn.Parameter(torch.ones(k))
        self.B_logits = nn.Parameter(torch.zeros(k, k))
        self.W = nn.Linear(2 * d_model, d_z)
        self.phi = nn.Sequential(nn.Linear(4 * d_z, d_z), nn.GELU(), nn.Linear(d_z, 1))
        nn.init.constant_(cast(nn.Linear, self.phi[-1]).bias, gate_bias)
        # Degree/clustering share their calibration across endpoint roles.
        self.cal_scale = nn.Parameter(torch.ones(6))
        self.cal_shift = nn.Parameter(torch.zeros(6))
        self.distance_head = nn.Linear(3, 4)
        self.register_buffer(
            "coord_mean",
            torch.zeros(11) if coord_mean is None else coord_mean.detach().float().clone(),
        )
        self.register_buffer(
            "coord_std", torch.ones(11) if coord_std is None else coord_std.detach().float().clone()
        )
        self.intervention = "none"
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
        """Return additive eval sums (reduce across ranks before taking means).

        Attachment sums, entropy sums and logit moments have one entry per
        coarse node, with ``attachment_count`` endpoint observations;
        ``attachment_within_var_sum`` is one scalar over those observations.
        Gate sums use ``gate_count`` pairs. ``adjacency_entropy`` is a parameter
        statistic, identical across ranks, and must not be summed across ranks.
        """
        result = {name: value.clone() for name, value in self._telemetry.items()}
        for prefix in ("attachment", "gate"):
            for suffix in ("sum", "entropy_sum"):
                result.setdefault(f"{prefix}_{suffix}", self.P.new_zeros(self.k))
            result.setdefault(f"{prefix}_count", self.P.new_zeros(()))
        for suffix in ("logit_sum", "logit_sq_sum"):
            result.setdefault(f"attachment_{suffix}", self.P.new_zeros(self.k))
        result.setdefault("attachment_within_var_sum", self.P.new_zeros(()))
        result["adjacency_entropy"] = self._entropy(self.adjacency).mean()
        if reset:
            self.reset_telemetry()
        return result

    @torch.no_grad()
    def _record(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        gates: torch.Tensor,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> None:
        additions: dict[str, torch.Tensor] = {}
        for prefix, values in (("attachment", torch.cat((a, b))), ("gate", gates)):
            additions[f"{prefix}_sum"] = values.sum(0)
            additions[f"{prefix}_entropy_sum"] = self._entropy(values).sum(0)
            additions[f"{prefix}_count"] = values.new_tensor(values.size(0))
        # Selectivity: how much of the logit variance is within a protein,
        # across coarse nodes, rather than a per-protein scalar.
        logits = torch.cat((logits_a, logits_b))
        additions["attachment_logit_sum"] = logits.sum(0)
        additions["attachment_logit_sq_sum"] = logits.square().sum(0)
        additions["attachment_within_var_sum"] = logits.var(dim=1, correction=0).sum()
        for key, value in additions.items():
            self._telemetry[key] = self._telemetry.get(key, torch.zeros_like(value)) + value

    @torch.no_grad()
    def initialise(
        self,
        assignments: torch.Tensor,
        adjacency: torch.Tensor | csr_matrix,
        cluster_mean_states: torch.Tensor,
    ) -> None:
        """Install the coarsening, seed the attention, and freeze the coarse graph.

        ``m`` and ``B`` are measured on the training graph and then frozen: they
        are data, not parameters. The queries ``P`` are seeded in the key map's
        own space and the attention's query projection is tied to its key
        projection, so a score is a similarity in one random projection rather
        than a dot product of two unrelated random maps. Each block's readout
        bias reproduces that block's mean training attachment, which puts every
        predicted count at the right order of magnitude at step zero.

        Args:
            assignments: Cluster label per training node ``(N,)``.
            adjacency: The loopless training adjacency over those nodes, dense
                or scipy sparse, in the same row order.
            cluster_mean_states: Per-cluster mean residue state ``(K, d_model)``.

        Raises:
            ValueError: If any cluster is empty.
        """
        device = self.P.device
        labels = assignments.to(device=device, dtype=torch.long)
        sizes = torch.bincount(labels, minlength=self.k).float()
        if sizes.numel() != self.k or (sizes == 0).any():
            raise ValueError("virtual graph initialisation requires every cluster to be nonempty")
        self.P.copy_(self.residue_projection(cluster_mean_states.to(device, torch.float32)))
        d_z = self.P.size(-1)
        self.attention.in_proj_weight[:d_z].copy_(self.attention.in_proj_weight[d_z : 2 * d_z])
        if self.attention.in_proj_bias is not None:
            self.attention.in_proj_bias[: 2 * d_z].zero_()
        self.m_raw.copy_(sizes + torch.log(-torch.expm1(-sizes)))
        if isinstance(adjacency, torch.Tensor):
            row, col = adjacency.nonzero(as_tuple=True)
        else:
            row_np, col_np = adjacency.nonzero()
            row, col = torch.as_tensor(row_np), torch.as_tensor(col_np)
        row, col = row.to(device), col.to(device)
        keep = row != col
        ids = labels[row[keep]] * self.k + labels[col[keep]]
        edge_counts = torch.bincount(ids, minlength=self.k**2).reshape(self.k, self.k).float()
        pair_counts = sizes[:, None] * sizes[None, :] - torch.diag(sizes)
        density = (edge_counts / pair_counts.clamp_min(1)).clamp(1e-4, 1 - 1e-4)
        self.B_logits.copy_(torch.logit(density))
        # Column mean of the node-to-block neighbour counts, per block size.
        block_counts = torch.zeros(self.k, device=device)
        block_counts.index_add_(0, labels[col[keep]], torch.ones(int(keep.sum()), device=device))
        mean_attachment = block_counts / labels.numel() / sizes
        self.readout_bias.copy_(torch.logit(mean_attachment.clamp(1e-6, 1 - 1e-6)))
        self.freeze_coarse_graph()

    def freeze_coarse_graph(self) -> None:
        """Stop training ``B`` and ``m``; idempotent, and called on every rank.

        Only the main process runs `initialise`; the other ranks receive the
        coarse graph by broadcast and must freeze it themselves, or DDP and the
        optimizer groups disagree across ranks.
        """
        self.B_logits.requires_grad_(False)
        self.m_raw.requires_grad_(False)

    @staticmethod
    def pool(encoded: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Masked mean/max summaries used only by the query gate."""
        keep = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
        if lengths is not None:
            keep = torch.arange(encoded.size(1), device=encoded.device)[None] < lengths[:, None]
        states = encoded.float()
        mean = states.masked_fill(~keep[..., None], 0).sum(1) / keep.sum(1).clamp_min(1)[:, None]
        maximum = states.masked_fill(~keep[..., None], -torch.inf).amax(1)
        return torch.cat((mean, maximum), -1)

    def forward(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor | None,
        lengths_b: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Emit standardised compact fields and four distance-class logits."""
        with torch.autocast(device_type=encoded_a.device.type, enabled=False):
            logits_a = self.attachment_logits(encoded_a, lengths_a)
            logits_b = self.attachment_logits(encoded_b, lengths_b)
            a, b = logits_a.sigmoid(), logits_b.sigmoid()
            u, v = self.W(self.pool(encoded_a, lengths_a)), self.W(self.pool(encoded_b, lengths_b))
            pair = torch.cat((u + v, (u - v).abs(), u * v), -1)
            gates = (
                self.phi(
                    torch.cat(
                        (
                            pair[:, None].expand(-1, self.k, -1),
                            self.P[None].expand(pair.size(0), -1, -1),
                        ),
                        -1,
                    )
                )
                .squeeze(-1)
                .sigmoid()
            )
            if self.intervention == "slot_gates_open":
                gates = torch.ones_like(gates)
            elif self.intervention != "none":
                raise ValueError(f"unsupported virtual graph intervention {self.intervention!r}")
            if not self.training:
                self._record(a, b, gates, logits_a, logits_b)
            c = self.count(a * gates, b * gates)
            raw = torch.stack(
                (
                    c["degree_u"].log1p(),
                    c["clustering_u"],
                    c["degree_v"].log1p(),
                    c["clustering_v"],
                    c["common"].log1p(),
                    c["jaccard"],
                    c["l3"].log1p(),
                    c["l3_density"],
                ),
                -1,
            )
            scale = torch.cat((self.cal_scale[:2], self.cal_scale[:2], self.cal_scale[2:]))
            shift = torch.cat((self.cal_shift[:2], self.cal_shift[:2], self.cal_shift[2:]))
            z = (raw * scale + shift - self.coord_mean[:8]) / self.coord_std[:8]
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
                "endpoint_u": z[:, :2],
                "endpoint_v": z[:, 2:4],
                "relation": z[:, 4:8],
                "distance_logits": distance,
                "attach_a": a,
                "attach_b": b,
            }

    def attachment_loss_rows(
        self, parts: Mapping[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        r"""Per-row Huber on predicted against true per-block neighbour counts.

        Supervision is in count space, where ``d log1p(m sigma(s)) / ds`` is
        order one at ``n ~ 1``; the eight aggregate coordinates alone leave the
        attachment logits with 1e-5 gradients. The caller masks self rows and
        rows touching featureless nodes.

        Args:
            parts: A `forward` output, read for ``attach_a`` and ``attach_b``.
            targets: ``(B, 2, K)`` counts ``|N(x) \ {partner} ^ C_j|``.

        Returns:
            The ``(B,)`` mean loss over the two endpoints' ``K`` blocks.
        """
        attachments = torch.stack((parts["attach_a"], parts["attach_b"]), dim=1).float()
        predicted = (attachments * self.multiplicity.float()).log1p()
        loss = F.smooth_l1_loss(predicted, targets.to(predicted).log1p(), reduction="none")
        return loss.flatten(1).mean(dim=1)

    @property
    def multiplicity(self) -> torch.Tensor:
        """Positive effective block sizes."""
        return F.softplus(self.m_raw)

    @property
    def adjacency(self) -> torch.Tensor:
        """Symmetric block probabilities, including within-block density."""
        return ((self.B_logits + self.B_logits.T) * 0.5).sigmoid()

    def count(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        """Count loopless lifted-graph motifs from effective attachments.

        Fractional degrees need a bounded clustering extension: zero below
        degree two's singular boundary, and at most one above it. Fractional
        blocks below one likewise contain zero distinct within-block pairs.
        Both agree with ordinary graph counts for integer binary lifts.
        """
        with torch.autocast(device_type=a.device.type, enabled=False):
            a, b = a.float(), b.float()
            m = self.multiplicity.float()
            pairs = m[:, None] * m[None, :]
            pairs = pairs - torch.diag(m.square()) + torch.diag((m * (m - 1)).clamp_min(0))
            edges = pairs * self.adjacency.float()
            du, dv = a @ m, b @ m
            tu, tv = (a @ edges * a).sum(-1) * 0.5, (b @ edges * b).sum(-1) * 0.5
            cu = torch.where(du > 1, 2 * tu / (du * (du - 1)).clamp_min(1e-8), 0.0).clamp(0, 1)
            cv = torch.where(dv > 1, 2 * tv / (dv * (dv - 1)).clamp_min(1e-8), 0.0).clamp(0, 1)
            common = (a * b * m).sum(-1)
            l3 = (a @ edges * b).sum(-1)
            return {
                "degree_u": du,
                "degree_v": dv,
                "triangles_u": tu,
                "triangles_v": tv,
                "clustering_u": cu,
                "clustering_v": cv,
                "common": common,
                "jaccard": common / (du + dv - common).clamp_min(1e-8),
                "l3": l3,
                "l3_density": l3 / (du * dv).clamp_min(1e-8),
            }

    def attach(self, encoded: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Return per-protein attachments, ignoring padded residues."""
        return self.attachment_logits(encoded, lengths).sigmoid()

    def attachment_logits(
        self, encoded: torch.Tensor, lengths: torch.Tensor | None
    ) -> torch.Tensor:
        """Return ``(B, K)`` attachment logits, one readout per coarse node.

        Coarse node ``j`` reads its own attended state with its own query
        direction, ``s_uj = readout_bias[j] + <P_j, o_uj> / sqrt(d_z)``. A
        readout shared across coarse nodes makes ``a_uj`` a per-protein scalar
        whatever the attention does, which is what starved the graph
        parameters of gradient in v0.5.
        """
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            pad = (
                None
                if lengths is None
                else (
                    torch.arange(encoded.size(1), device=encoded.device)[None] >= lengths[:, None]
                )
            )
            states = self.residue_projection(encoded.float())
            output, _ = self.attention(
                self.P[None].expand(encoded.size(0), -1, -1),
                states,
                states,
                key_padding_mask=pad,
                need_weights=False,
            )
            scores = (output * self.P[None]).sum(-1) / self.P.size(-1) ** 0.5
            return cast(torch.Tensor, self.readout_bias[None] + scores)
