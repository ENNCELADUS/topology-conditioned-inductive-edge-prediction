"""S2 model components: Stage A graph AE and Stage B conditional flow prior.

Spec: ``docs/tmp/s2_set_generation_experiment.md``. Stage A (`RegionAutoencoder`)
encodes the *true* region graph with the existing GRIT encoder into per-node
latents ``Z_i = (a_i, z_i)`` and decodes them with an inner-product edge
decoder (`pair_logits`); its reconstruction quality is the latent ceiling
Stage B is judged against. Stage B (`SetDenoiser`) is a permutation-equivariant
Set Transformer velocity field for a rectified-flow prior over ``Z_{1:n}``
conditioned on node features alone -- no positional encodings, so every node's
prediction can depend on the whole set, which is where the node-aligned
jointness this diagnostic tests for has to live. `DetTwin` shares the same
backbone as a deterministic regressor (isolates generation from set encoding);
`MargDegreeRegressor` is the strictly-per-node baseline (isolates joint
modeling from independent per-node prediction).

Depends only on torch/numpy and the existing `src.model.egostitch` /
`src.vendor.set_transformer_official` modules -- never on the sibling
`data.py`, so every interface here is plain tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder
from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph
from src.vendor.set_transformer_official import MAB, PMA

__all__ = [
    "RegionGraph",
    "pair_logits",
    "RegionAutoencoder",
    "ae_losses",
    "sinusoidal_time_embedding",
    "SetDenoiser",
    "DetTwin",
    "MargDegreeRegressor",
    "flow_matching_loss",
    "sample_prior",
]


@dataclass(frozen=True)
class RegionGraph(ImaginedGraph):
    """Undirected region graph; no pair roles.

    Stage A's autoencoder runs over one region at a time, not an ordered
    pair, so there is no AB/BA orientation to flip.
    """

    def swapped(self) -> RegionGraph:
        """Return ``self`` -- the region graph carries no direction to flip."""
        return self


def pair_logits(a: torch.Tensor, z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Inner-product edge decoder: ``a_i + a_j + z_i . z_j + bias``.

    Args:
        a: Shape ``(B, N)`` per-node activity.
        z: Shape ``(B, N, z_dim)`` per-node role vectors. ``z_dim == 0`` is the
            activity-only ablation: `torch.matmul` over an empty contracted
            dimension is exactly zero, so the role term drops out on its own.
        bias: Scalar decoder bias.

    Returns:
        Shape ``(B, N, N)`` symmetric logits. Diagonal entries are computed
        but not meaningful; consumers exclude them.
    """
    role = torch.matmul(z, z.transpose(-2, -1))
    return a.unsqueeze(-1) + a.unsqueeze(-2) + role + bias


class RegionAutoencoder(nn.Module):
    """Stage A graph autoencoder: GRIT encoder + inner-product edge decoder.

    Encodes the true region graph into per-node latents and decodes them back
    to edge logits. Reconstruction quality bounds what Stage B's prior can
    ever recover from features alone.
    """

    def __init__(
        self,
        *,
        in_dim: int = 1536,
        z_dim: int = 16,
        dim: int = 96,
        layers: int = 4,
        rrwp_k: int = 8,
        n_heads: int = 8,
        seeds: int = 4,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        self._z_dim = z_dim
        self.encoder = GritGmtEncoder(
            in_dim=in_dim,
            num_relations=1,
            d_model=d_model,
            dim=dim,
            layers=layers,
            w_rel=0.0,
            rrwp_k=rrwp_k,
            n_heads=n_heads,
            seeds=seeds,
        )
        self.latent_head = nn.Linear(d_model, 1 + z_dim)
        self.bias = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the true region graph into per-node latents.

        Args:
            x: Shape ``(B, N, in_dim)`` node features.
            adj: Shape ``(B, N, N)`` 0/1 undirected adjacency.
            mask: Shape ``(B, N)`` 0/1 node existence.

        Returns:
            Shape ``(B, N, 1 + z_dim)`` latents, padded rows exactly zero.
        """
        graph = RegionGraph(x=x, adj=adj.unsqueeze(1), mask=mask, aux={}, directed=False)
        embedding: GraphEmbedding = self.encoder(graph)
        node_tokens = embedding.tokens[:, : x.shape[1]]
        latents: torch.Tensor = self.latent_head(node_tokens)
        return latents * mask.unsqueeze(-1).to(latents.dtype)

    def decode(self, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode latents to edge logits via `pair_logits`.

        Args:
            latents: Shape ``(B, N, 1 + z_dim)``.
            mask: Unused for the decoded values themselves -- padded rows of
                the returned logits are garbage the consumer masks.

        Returns:
            Shape ``(B, N, N)`` edge logits.
        """
        del mask
        return pair_logits(latents[..., 0], latents[..., 1:], self.bias)

    @property
    def z_dim(self) -> int:
        """The latent role width ``z_dim``."""
        return self._z_dim


def _triangle_moment(mat: torch.Tensor) -> torch.Tensor:
    """Zero-diagonal ``(B, N, N)`` soft/hard adjacency -> per-set ``(B,)`` triangle count.

    ``trace(A^3) / 6`` over an undirected, self-loop-free adjacency: each
    triangle contributes exactly 6 to the trace (3 starting nodes, 2
    directions), which is the closed-walk identity `ae_losses` uses for both
    the soft expectation and the hard ground truth.
    """
    cube = torch.matmul(torch.matmul(mat, mat), mat)
    return torch.diagonal(cube, dim1=-2, dim2=-1).sum(-1) / 6.0


def ae_losses(
    logits: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, *, lambda_tri: float
) -> dict[str, torch.Tensor]:
    """Reconstruction BCE plus a soft-triangle-count moment penalty.

    Args:
        logits: Shape ``(B, N, N)`` decoder output.
        adj: Shape ``(B, N, N)`` 0/1 true adjacency, self-loop-free.
        mask: Shape ``(B, N)`` 0/1 node existence.
        lambda_tri: Triangle-penalty weight.

    Returns:
        ``{"bce": ..., "triangle": ..., "total": ...}``, all scalars;
        ``total = bce + lambda_tri * triangle``. ``bce`` is
        `F.binary_cross_entropy_with_logits` over valid, strict-upper-triangle
        pairs only (both endpoints real). ``triangle`` is the per-set penalty
        ``((E_T - T_true) / (T_true + 1)) ** 2``, meaned over the batch, where
        ``E_T`` comes from the soft edge probabilities (zero diagonal, padded
        pairs excluded) and ``T_true`` from the hard adjacency the same way.
    """
    n = logits.shape[-1]
    valid = mask > 0.0
    pairmask = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=logits.device), diagonal=1)
    upper = pairmask & triu
    bce = F.binary_cross_entropy_with_logits(logits, adj, reduction="none")[upper].mean()

    eye = torch.eye(n, device=logits.device, dtype=logits.dtype)
    zero_diag_pairmask = pairmask.to(logits.dtype) * (1.0 - eye)
    soft_p = torch.sigmoid(logits) * zero_diag_pairmask
    hard_a = adj * zero_diag_pairmask

    e_t = _triangle_moment(soft_p)
    t_true = _triangle_moment(hard_a)
    triangle = (((e_t - t_true) / (t_true + 1.0)) ** 2).mean()

    return {"bce": bce, "triangle": triangle, "total": bce + lambda_tri * triangle}


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a per-set flow-matching time value.

    Args:
        t: Shape ``(B,)``, typically in ``[0, 1]``.
        dim: Embedding width.

    Returns:
        Shape ``(B, dim)``, ``[sin(...), cos(...)]`` concatenated (zero-padded
        by one trailing column when ``dim`` is odd).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if dim % 2 == 1:
        pad = torch.zeros(*t.shape, 1, device=t.device, dtype=t.dtype)
        embedding = torch.cat((embedding, pad), dim=-1)
    return embedding


class _SetBackbone(nn.Module):
    """Permutation-equivariant SAB stack + PMA global read + broadcast-back MAB.

    Shared implementation for `SetDenoiser` and `DetTwin`; each owns its own
    instance, sized for its own per-node input width. No positional encoding
    is used anywhere, so equivariance under a node permutation is exact: `SAB`
    is `MAB(X, X, mask)`, the PMA global read is permutation-invariant over
    its keys, and the final broadcast `MAB(X, global)` lets every node attend
    to that shared, order-independent summary of the whole set.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        width: int,
        sab_layers: int,
        heads: int,
        pma_seeds: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.sabs = nn.ModuleList(
            [MAB(width, width, width, heads) for _ in range(sab_layers)]  # type: ignore[no-untyped-call]
        )
        self.pma = PMA(width, heads, pma_seeds)  # type: ignore[no-untyped-call]
        self.broadcast = MAB(width, width, width, heads)  # type: ignore[no-untyped-call]
        self.output_head = nn.Linear(width, out_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Refine per-node ``tokens`` through the shared set-attention stack.

        Args:
            tokens: Shape ``(B, N, in_dim)`` per-node input.
            mask: Shape ``(B, N)`` node existence in ``{0, 1}``.

        Returns:
            Shape ``(B, N, out_dim)``, padded rows zeroed.
        """
        key_mask = mask > 0.0
        hidden: torch.Tensor = self.input_proj(tokens)
        for sab in self.sabs:
            hidden = sab(hidden, hidden, key_mask)
        global_tokens = self.pma(hidden, key_mask)
        hidden = self.broadcast(hidden, global_tokens)
        out: torch.Tensor = self.output_head(hidden)
        return out * mask.unsqueeze(-1).to(out.dtype)


class SetDenoiser(nn.Module):
    """Rectified-flow velocity field over ``Z_{1:n}`` conditioned on ``X_S``.

    No positional encodings: every node attends to the whole conditioning
    set through the shared `_SetBackbone`, which is where node-aligned
    jointness lives. Classifier-free conditioning dropout (zeroing a whole
    set's ``x``) is the caller's -- `flow_matching_loss`'s -- responsibility;
    this module only ever does what it is told with the ``x`` it receives.
    """

    def __init__(
        self,
        *,
        in_dim: int = 1536,
        z_dim: int = 16,
        width: int = 384,
        sab_layers: int = 4,
        heads: int = 4,
        pma_seeds: int = 4,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.feature_proj = nn.Linear(in_dim, width)
        self.backbone = _SetBackbone(
            in_dim=(1 + z_dim) + time_dim + width,
            out_dim=1 + z_dim,
            width=width,
            sab_layers=sab_layers,
            heads=heads,
            pma_seeds=pma_seeds,
        )

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict the rectified-flow velocity at ``(z_t, t)`` given ``x``.

        Args:
            z_t: Shape ``(B, N, 1 + z_dim)`` noisy latents.
            t: Shape ``(B,)`` per-set flow time in ``[0, 1]``.
            x: Shape ``(B, N, in_dim)`` conditioning features.
            mask: Shape ``(B, N)`` node existence in ``{0, 1}``.

        Returns:
            Shape ``(B, N, 1 + z_dim)`` predicted velocity, padded rows zeroed.
        """
        n = z_t.shape[1]
        time_embed = sinusoidal_time_embedding(t, self.time_dim).unsqueeze(1).expand(-1, n, -1)
        feature: torch.Tensor = self.feature_proj(x)
        tokens = torch.cat((z_t, time_embed, feature), dim=-1)
        velocity: torch.Tensor = self.backbone(tokens, mask)
        return velocity


class DetTwin(nn.Module):
    """Same backbone as `SetDenoiser`, features only -> standardized latent prediction.

    Isolates generation from set encoding: a deterministic regressor sharing
    the exact set-attention architecture, trained on the same conditioning.
    """

    def __init__(
        self,
        *,
        in_dim: int = 1536,
        z_dim: int = 16,
        width: int = 384,
        sab_layers: int = 4,
        heads: int = 4,
        pma_seeds: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = _SetBackbone(
            in_dim=in_dim,
            out_dim=1 + z_dim,
            width=width,
            sab_layers=sab_layers,
            heads=heads,
            pma_seeds=pma_seeds,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict standardized latents directly from features.

        Args:
            x: Shape ``(B, N, in_dim)`` node features.
            mask: Shape ``(B, N)`` node existence in ``{0, 1}``.

        Returns:
            Shape ``(B, N, 1 + z_dim)`` predicted latents, padded rows zeroed.
        """
        prediction: torch.Tensor = self.backbone(x, mask)
        return prediction


class MargDegreeRegressor(nn.Module):
    """Strictly per-node degree regressor -- the independence baseline.

    Deliberately no set attention: each node's expected degree is a function
    of only its own features and the set size, so any node-to-node coherence
    `GEN` exhibits beyond this arm is attributable to joint modeling, not to
    context shared across nodes.
    """

    def __init__(self, *, in_dim: int = 1536, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Predict softplus expected in-set degree, independently per node.

        Args:
            x: Shape ``(B, N, in_dim)`` node features.
            n: Shape ``(B,)`` long set sizes.

        Returns:
            Shape ``(B, N)`` nonnegative expected degree.
        """
        num_nodes = x.shape[1]
        size_frac = (n.to(x.dtype) / 200.0).view(-1, 1, 1).expand(-1, num_nodes, 1)
        raw: torch.Tensor = self.mlp(torch.cat((x, size_frac), dim=-1))
        return F.softplus(raw.squeeze(-1))


def flow_matching_loss(
    denoiser: SetDenoiser,
    z1_std: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    p_drop: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Rectified-flow regression loss with classifier-free conditioning dropout.

    ``z0 ~ N(0, I)``; ``t ~ U(0, 1)`` per set; ``z_t = (1 - t) z0 + t z1_std``;
    loss is the mean squared error between the denoiser's predicted velocity
    and ``z1_std - z0`` over valid nodes only. With probability ``p_drop``,
    independently per set, that set's ``x`` is zeroed before conditioning.

    Args:
        denoiser: The velocity-field model to score.
        z1_std: Shape ``(B, N, 1 + z_dim)`` standardized target latents.
        x: Shape ``(B, N, in_dim)`` conditioning features.
        mask: Shape ``(B, N)`` node existence in ``{0, 1}``.
        p_drop: Per-set conditioning-dropout probability.
        generator: Source of all randomness (noise, time, dropout draws).

    Returns:
        Scalar loss.
    """
    batch, n, dim = z1_std.shape
    z0 = torch.randn(batch, n, dim, generator=generator, device=z1_std.device, dtype=z1_std.dtype)
    t = torch.rand(batch, generator=generator, device=z1_std.device, dtype=z1_std.dtype)
    z_t = (1.0 - t).view(batch, 1, 1) * z0 + t.view(batch, 1, 1) * z1_std

    drop = torch.rand(batch, generator=generator, device=x.device) < p_drop
    x_cond = torch.where(drop.view(batch, 1, 1), torch.zeros_like(x), x)

    velocity: torch.Tensor = denoiser(z_t=z_t, t=t, x=x_cond, mask=mask)
    target = z1_std - z0
    valid = (mask > 0.0).to(velocity.dtype).unsqueeze(-1)
    count = valid.sum() * dim
    return ((velocity - target) ** 2 * valid).sum() / count.clamp(min=1.0)


def sample_prior(
    denoiser: SetDenoiser,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    z_dim: int,
    n_draws: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw samples from the flow prior by Euler-integrating ``t: 0 -> 1``.

    Args:
        denoiser: The trained velocity-field model.
        x: Shape ``(B, N, in_dim)`` conditioning features.
        mask: Shape ``(B, N)`` node existence in ``{0, 1}``.
        z_dim: Latent role width, so the sampled dimension is ``1 + z_dim``.
        n_draws: Number of independent draws ``K``.
        steps: Number of Euler steps.
        generator: Source of all randomness. Must live on a device compatible
            with ``x`` -- the initial noise is created on ``x.device``.

    Returns:
        Shape ``(K, B, N, 1 + z_dim)`` standardized-scale samples; the caller
        de-standardizes. Padded rows are finite but not necessarily zero.
    """
    batch, n, _ = x.shape
    dim = 1 + z_dim
    dt = 1.0 / steps
    draws = []
    with torch.no_grad():
        for _ in range(n_draws):
            z = torch.randn(batch, n, dim, generator=generator, device=x.device, dtype=x.dtype)
            time = 0.0
            for _ in range(steps):
                t = torch.full((batch,), time, device=x.device, dtype=x.dtype)
                velocity = denoiser(z_t=z, t=t, x=x, mask=mask)
                z = z + dt * velocity
                time += dt
            draws.append(z)
        return torch.stack(draws, dim=0)
