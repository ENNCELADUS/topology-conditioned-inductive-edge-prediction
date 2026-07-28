"""EgoStitch Stage-1 model: Tokenize-lite + Imagine + Stitch + decision head.

Follows the repository forward contract (``model(batch) -> {"logits", "loss"?}``,
the V3_1 / F0PairMLP convention) so `src.score_universe` and the training worker
consume it uniformly. ``s0`` always arrives as a cached frozen-B0 logit
(spec Sec 13.10); the model never computes it.

The per-node pass (`encode_nodes`) is the cacheable unit (spec Sec 10.3): the
scorer runs it once per node, then scores pair batches from the cache.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.feature_stats import FeatureStats
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.decision import DecisionHead
from src.model.egostitch.imagine import DenoiseSlots, ImagineDecoder, SlotSet
from src.model.egostitch.losses import (
    LossFamily,
    RandomGIN,
    degree_nll,
    denoise_losses,
    generated_ego_graph,
    generated_ego_stats,
    real_ego_graph,
    recon_losses,
    ssl_consistency,
    standardized_energy_distance,
)
from src.model.egostitch.matching import match_slots
from src.model.egostitch.stitch import sinkhorn_plan
from src.model.egostitch.tokenize import TokenizeLite, TokenizeOut


class NodeEncoding(NamedTuple):
    """The cacheable per-node pass outputs.

    Attributes:
        tok: Tokenize-lite outputs.
        slots: Generated slot set.
        denoise: Denoise-slot outputs (training only, else ``None``).
    """

    tok: TokenizeOut
    slots: SlotSet
    denoise: DenoiseSlots | None


class NodeLosses(NamedTuple):
    """Node-stream loss bundle (spec Sec 13.5 families minus SSL).

    Attributes:
        recon: The `recon_losses` dict with any denoise terms folded in.
        deg: The lognormal degree NLL.
        real_egostat: Ego-stat energy distance.
        real_gin: Random-GIN energy distance.
    """

    recon: dict[str, torch.Tensor]
    deg: torch.Tensor
    real_egostat: torch.Tensor
    real_gin: torch.Tensor


FeatureStandardizationMode = Literal["none", "row_layernorm", "zscore_vfit_v1"]


class FeatureStandardizer(nn.Module):
    """Registered per-dimension F0 z-scoring (spec Sec 13.19.1).

    The constants are persistent buffers so scoring reconstructs the transform
    from the checkpoint alone; the run aborts rather than standardizing with
    unregistered statistics.
    """

    def __init__(self, dim: int) -> None:
        """Register empty, not-ready statistics buffers.

        Args:
            dim: The F0 dimension.
        """
        super().__init__()
        self.register_buffer("feature_mu", torch.zeros(dim))
        self.register_buffer("feature_sigma", torch.ones(dim))
        self.register_buffer("feature_stats_ready", torch.zeros((), dtype=torch.int64))
        self.register_buffer("feature_stats_digest", torch.zeros(32, dtype=torch.uint8))

    def load_stats(self, mu: torch.Tensor, sigma: torch.Tensor, digest: str) -> None:
        """Pin the registered constants and their identity.

        Args:
            mu: Shape ``(d,)`` per-dimension mean.
            sigma: Shape ``(d,)`` strictly positive per-dimension standard deviation.
            digest: The 64-hex `feature_stats_sha256`.

        Raises:
            ValueError: On a dimension mismatch, a non-positive or non-finite
                sigma, or a malformed digest.
        """
        buffers = (self.feature_mu, self.feature_sigma)
        assert isinstance(buffers[0], torch.Tensor) and isinstance(buffers[1], torch.Tensor)
        if mu.shape != buffers[0].shape or sigma.shape != buffers[1].shape:
            raise ValueError("feature statistics dimension does not match the model")
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
            raise ValueError("feature statistics are not finite")
        if not bool((sigma > 0).all()):
            raise ValueError("feature statistics sigma must be strictly positive")
        if len(digest) != 64:
            raise ValueError("feature statistics digest must be a 64-hex sha256")
        buffers[0].copy_(mu.to(buffers[0].dtype))
        buffers[1].copy_(sigma.to(buffers[1].dtype))
        ready = self.feature_stats_ready
        assert isinstance(ready, torch.Tensor)
        ready.fill_(1)
        recorded = self.feature_stats_digest
        assert isinstance(recorded, torch.Tensor)
        recorded.copy_(torch.frombuffer(bytearray.fromhex(digest), dtype=torch.uint8))

    @property
    def digest_hex(self) -> str:
        """The registered `feature_stats_sha256`, or ``""`` when unset."""
        ready = self.feature_stats_ready
        recorded = self.feature_stats_digest
        assert isinstance(ready, torch.Tensor) and isinstance(recorded, torch.Tensor)
        if int(ready) != 1:
            return ""
        return bytes(recorded.cpu().numpy().tobytes()).hex()

    def scale_perturbation(self, noise: torch.Tensor) -> torch.Tensor:
        """Scale a raw-coordinate perturbation into standardized coordinates.

        Args:
            noise: Raw-space noise broadcastable to ``(..., d)``.

        Returns:
            ``noise * sigma`` -- adding it before standardization is exactly a
            `noise`-sized perturbation of the standardized row.
        """
        sigma = self.feature_sigma
        assert isinstance(sigma, torch.Tensor)
        return noise * sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize an F0 tensor in fp32.

        Args:
            features: Shape ``(..., d)`` raw F0 values.

        Returns:
            The standardized fp32 tensor.

        Raises:
            RuntimeError: When the statistics have not been registered.
        """
        ready = self.feature_stats_ready
        assert isinstance(ready, torch.Tensor)
        if int(ready) != 1:
            raise RuntimeError(
                "feature standardization statistics are not registered; "
                "the trainer must call set_feature_stats before the first forward"
            )
        mu = self.feature_mu
        sigma = self.feature_sigma
        assert isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor)
        return (features.float() - mu) / sigma


class EgoStitchStage1(nn.Module):
    """The Stage-1 minimum-viable EgoStitch model (spec Sec 13)."""

    name = "egostitch"

    def __init__(
        self,
        config: EgoStitchConfig,
        *,
        feature_standardization: FeatureStandardizationMode = "none",
        loss_family: LossFamily = "egostitch",
    ) -> None:
        """Build every Stage-1 module.

        Args:
            config: The pinned Stage-1 configuration.
            feature_standardization: ``"none"`` for the legacy frozen-s0 raw-F0
                semantics, ``"row_layernorm"`` for the rev-3.1 stateless per-row
                transform (replay only), ``"zscore_vfit_v1"`` for the registered
                rev-3.2 per-dimension constants (spec Sec 13.19.1).
            loss_family: Select the frozen Stage-1 or rev-3.1 E2E loss path.
        """
        super().__init__()
        self.config = config
        self.loss_family = loss_family
        self.feature_standardization = feature_standardization
        self.feature_norm: nn.Module
        if feature_standardization == "none":
            self.feature_norm = nn.Identity()
        elif feature_standardization == "row_layernorm":
            self.feature_norm = nn.LayerNorm(config.input_dim, elementwise_affine=False)
        elif feature_standardization == "zscore_vfit_v1":
            self.feature_norm = FeatureStandardizer(config.input_dim)
        else:
            raise ValueError(
                f"unknown feature standardization mode {feature_standardization!r}"
            )
        self.tokenize = TokenizeLite(config)
        self.imagine = ImagineDecoder(config)
        self.decision = DecisionHead(config)
        self.random_gin = RandomGIN(config)
        # rho_eval / rho_train (spec Sec 9.3); the trainer sets it at load.
        self.register_buffer("density_ratio", torch.tensor(1.0))

    @property
    def proj(self) -> nn.Linear:
        """The shared ``d -> d_p`` projection (spec Sec 13.7)."""
        return self.imagine.proj

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the registered stateless per-row F0 standardization."""
        return cast(torch.Tensor, self.feature_norm(features))

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the shared projection to standardized frozen F0 features."""
        return cast(torch.Tensor, self.proj(self.normalize_features(features)))

    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered standardization constants on this model.

        Args:
            stats: The V_fit statistics bundle.

        Raises:
            TypeError: When the configured mode carries no statistics.
        """
        if not isinstance(self.feature_norm, FeatureStandardizer):
            raise TypeError(
                f"feature standardization mode {self.feature_standardization!r} "
                "does not accept registered statistics"
            )
        self.feature_norm.load_stats(
            torch.from_numpy(stats.mu), torch.from_numpy(stats.sigma), stats.digest
        )

    @property
    def feature_stats_digest_hex(self) -> str:
        """The registered `feature_stats_sha256`, or ``""`` for stateless modes."""
        if not isinstance(self.feature_norm, FeatureStandardizer):
            return ""
        return self.feature_norm.digest_hex

    def scale_feature_perturbation(self, noise: torch.Tensor) -> torch.Tensor:
        """Map a raw-coordinate SSL perturbation into the active coordinates.

        Args:
            noise: Raw-space noise shaped ``(B, d)``.

        Returns:
            The noise the caller should add before standardization (spec Sec 7).
        """
        if not isinstance(self.feature_norm, FeatureStandardizer):
            return noise
        return self.feature_norm.scale_perturbation(noise)

    def set_density_ratio(self, ratio: float) -> None:
        """Pin the density normalization ``rho_eval / rho_train``."""
        buffer = self.density_ratio
        assert isinstance(buffer, torch.Tensor)
        buffer.fill_(float(ratio))

    def d_hat(self, tok: TokenizeOut) -> torch.Tensor:
        """Density-normalized degree budget ``d_hat_u`` (spec Sec 13.2)."""
        buffer = self.density_ratio
        assert isinstance(buffer, torch.Tensor)
        return tok.d_hat_raw * buffer

    def encode_nodes(
        self,
        x: torch.Tensor,
        ground_x: torch.Tensor,
        *,
        null_mode: torch.Tensor | None = None,
        extra_proj: torch.Tensor | None = None,
    ) -> NodeEncoding:
        """Run the cacheable per-node pass (Tokenize-lite + Imagine).

        Args:
            x: Shape ``(B, d)`` frozen node features.
            ground_x: Shape ``(B, n_g, d)`` grounding-candidate features.
            null_mode: Optional conditioning-dropout modes.
            extra_proj: Optional denoising-query initializations.

        Returns:
            The `NodeEncoding` bundle.
        """
        x = self.normalize_features(x)
        ground_x = self.normalize_features(ground_x)
        tok = self.tokenize(x)
        slots, denoise = self.imagine(
            x, tok.e, ground_x, null_mode=null_mode, extra_proj=extra_proj
        )
        return NodeEncoding(tok=tok, slots=slots, denoise=denoise)

    # ------------------------------------------------------------------ scoring

    def pair_outputs(
        self,
        enc_i: NodeEncoding,
        enc_j: NodeEncoding,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        s0: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return fused logits and the three Stage-1 channels for diagnostics."""
        plan = sinkhorn_plan(
            enc_i.slots.h,
            enc_j.slots.h,
            enc_i.slots.pi,
            enc_j.slots.pi,
            enc_i.slots.mult,
            enc_j.slots.mult,
            eps=self.config.sinkhorn_eps,
            iters=self.config.sinkhorn_iters,
            tau=self.config.sinkhorn_tau,
        )
        proj_i = self.project_features(x_i)
        proj_j = self.project_features(x_j)
        channels = self.decision.channels(
            enc_i.slots,
            enc_j.slots,
            plan,
            proj_i,
            proj_j,
            self.d_hat(enc_i.tok),
            self.d_hat(enc_j.tok),
        )
        return self.decision.fuse(s0, channels), channels

    def self_outputs(
        self, enc: NodeEncoding, x: torch.Tensor, s0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return self-pair logits and channels without a second imagination pass."""
        channels = self.decision.self_channels(
            enc.slots, self.project_features(x), self.d_hat(enc.tok)
        )
        return self.decision.fuse(s0, channels), channels

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Score one pair batch (the repository forward contract).

        A batch with ``x_j`` present is a non-self pair batch (keys ``x_i``,
        ``x_j``, ``ground_i``, ``ground_j``, ``s0``); a batch without ``x_j``
        is a self-pair batch (keys ``x_i``, ``ground_i``, ``s0``). An optional
        ``label`` adds the BCE ``loss`` entry.

        Args:
            batch: The batch dictionary.

        Returns:
            ``{"logits": (B,)}`` plus ``"loss"`` when labels are present.
        """
        s0 = batch["s0"]
        if "x_j" in batch:
            enc_i = self.encode_nodes(batch["x_i"], batch["ground_i"])
            enc_j = self.encode_nodes(batch["x_j"], batch["ground_j"])
            logits, channels = self.pair_outputs(enc_i, enc_j, batch["x_i"], batch["x_j"], s0)
        else:
            enc = self.encode_nodes(batch["x_i"], batch["ground_i"])
            logits, channels = self.self_outputs(enc, batch["x_i"], s0)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "residual": logits - s0,
            **channels,
        }
        if "label" in batch:
            out["loss"] = F.binary_cross_entropy_with_logits(logits, batch["label"].float())
        return out

    # ------------------------------------------------------------------ training losses

    def node_losses(
        self,
        x: torch.Tensor,
        ground_x: torch.Tensor,
        *,
        target_features: torch.Tensor,
        target_mult: torch.Tensor,
        target_adj: torch.Tensor,
        target_mask: torch.Tensor,
        target_in_pool: torch.Tensor,
        target_pool_index: torch.Tensor,
        true_degree: torch.Tensor,
        real_ego_stats: torch.Tensor,
        null_mode: torch.Tensor | None = None,
        denoise_features: torch.Tensor | None = None,
        denoise_mask: torch.Tensor | None = None,
        denoise_noise: torch.Tensor | None = None,
    ) -> tuple[NodeLosses, NodeEncoding]:
        """Node-stream losses: reconstruction + degree NLL + realism EDs.

        Args:
            x: Shape ``(B, d)`` node features.
            ground_x: Shape ``(B, n_g, d)`` grounding-candidate features.
            target_features: Shape ``(B, T, d)`` ego-net target features.
            target_mult: Shape ``(B, T)`` multiplicity labels.
            target_adj: Shape ``(B, T, T)`` adjacency among targets.
            target_mask: Shape ``(B, T)`` target validity mask.
            target_in_pool: Shape ``(B, T)`` grounding-pool membership.
            target_pool_index: Shape ``(B, T)`` ordered grounding-pool indices,
                with ``-1`` for out-of-pool targets and padding.
            true_degree: Shape ``(B,)`` true simple degrees.
            real_ego_stats: Shape ``(B, 4)`` real-side ego-stat vectors
                (spec Sec 13.6, computed by the target builder on G_struct).
            null_mode: Optional conditioning-dropout modes.
            denoise_features: Optional ``(B, K_d, d)`` denoise-source features.
            denoise_mask: Optional ``(B, K_d)`` denoise validity mask.
            denoise_noise: Optional ``(B, K_d, d_p)`` pre-sampled query noise
                (seeded by the trainer for determinism).

        Returns:
            ``(losses, encoding)`` — see `NodeLosses` for the loss bundle.
        """
        extra_proj: torch.Tensor | None = None
        denoise_target: torch.Tensor | None = None
        if denoise_features is not None:
            if denoise_mask is None or denoise_noise is None:
                raise ValueError("denoise_features requires denoise_mask and denoise_noise")
            denoise_target = self.project_features(denoise_features).detach()
            extra_proj = denoise_target + denoise_noise

        enc = self.encode_nodes(x, ground_x, null_mode=null_mode, extra_proj=extra_proj)

        target_proj = self.project_features(target_features).detach()
        assignment = match_slots(
            enc.slots,
            target_proj=target_proj,
            target_mult=target_mult,
            target_adj=target_adj,
            target_mask=target_mask,
        )
        recon = recon_losses(
            enc.slots,
            assignment,
            target_proj=target_proj,
            target_mult=target_mult,
            target_adj=target_adj,
            target_in_pool=target_in_pool,
            target_pool_index=target_pool_index,
            config=self.config,
            family=self.loss_family,
        )
        if enc.denoise is not None and denoise_target is not None and denoise_mask is not None:
            extra = denoise_losses(enc.denoise, target_proj=denoise_target, mask=denoise_mask)
            recon = {
                **recon,
                "feat": recon["feat"] + extra["feat"],
                "exist": recon["exist"] + extra["exist"],
            }
        deg = degree_nll(enc.tok.deg_mu, enc.tok.deg_log_sigma, true_degree)

        real_egostat = standardized_energy_distance(generated_ego_stats(enc.slots), real_ego_stats)
        gen_feat, gen_adj = generated_ego_graph(enc.slots)
        real_feat, real_adj = real_ego_graph(target_mult, target_adj, target_mask, target_in_pool)
        real_gin = standardized_energy_distance(
            self.random_gin(gen_feat, gen_adj), self.random_gin(real_feat, real_adj)
        )
        losses = NodeLosses(recon=recon, deg=deg, real_egostat=real_egostat, real_gin=real_gin)
        return losses, enc

    def ssl_losses(
        self,
        x: torch.Tensor,
        ground_x: torch.Tensor,
        ground_x_resampled: torch.Tensor,
        *,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """SSL consistency terms (spec Sec 7; ungrounded slots only).

        Args:
            x: Shape ``(B, d)`` node features.
            ground_x: Shape ``(B, n_g, d)`` grounding-candidate features.
            ground_x_resampled: Shape ``(B, n_g, d)`` resampled pool features.
            noise: Shape ``(B, d)`` pre-sampled feature noise (scaled by the
                trainer to ``ssl_noise_sigma``; seeded for determinism)
                (raw F0 coordinates; scaled into standardized coordinates by
                `scale_feature_perturbation`).

        Returns:
            ``{"noise", "pool"}`` scalar consistency losses.
        """
        enc_clean = self.encode_nodes(x, ground_x)
        ungrounded = enc_clean.slots.gate < 0.5
        enc_noise = self.encode_nodes(x + self.scale_feature_perturbation(noise), ground_x)
        enc_pool = self.encode_nodes(x, ground_x_resampled)
        return {
            "noise": ssl_consistency(enc_clean.slots, enc_noise.slots, ungrounded=ungrounded),
            "pool": ssl_consistency(enc_clean.slots, enc_pool.slots, ungrounded=ungrounded),
        }
