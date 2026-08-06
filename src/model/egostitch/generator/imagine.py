"""Modules 1-2 (Stage-1 form): Tokenize-lite + Imagine, and the composed Stage-1 model.

Merges three former standalone modules (three-component refactor design §5
consolidation pass): ``tokenize.py`` (Tokenize-lite: encoder + degree budget),
``imagine.py`` (the DETR-style set decoder), and ``model.py``
(``EgoStitchStage1``, which composes them). They are one cohesive unit --
`EgoStitchStage1` is nothing but `TokenizeLite` + `ImagineDecoder` + the
frozen random-GIN realism head -- so they now live in one file rather than
three that always change together.

Tokenize-lite: no VQ codebook, no BP affiliations, no code-stats head in
Stage 1; ``e_u`` replaces ``(z_u, r_u)`` everywhere downstream. It outputs the
lognormal NLL parameters (mu, log sigma) that `degree_nll`
(`src.model.egostitch.generator.losses`) supervises. The density-normalized
raw softplus mean head (``degree_mean_head`` / ``d_hat_raw``) was removed
(three-component refactor design §12 P2a dead-code sweep): its only consumer
was ``EgoStitchStage1.d_hat()``, itself only called by the retired frozen-s0
``pair_outputs``/``self_outputs`` decision-fusion methods deleted with
`decision.py` (design §9). ``deg_mu``/``deg_log_sigma`` are unrelated and
stay -- they are `degree_nll`'s live inputs.

Imagine: codebook-free conditioning ``T_cond = [W_x x_u; W_e e_u]`` (2
tokens); query init from ``e_u``; no CVAE token. Owns the shared projection
``proj: d -> d_p`` consumed by the matching cost, ``L_feat``, and ``s1``
(spec Sec 13.7 — those consumers stop-gradient its target-side output). The
generated ego-net is **intermediate context** for the binary edge decision,
never a final output (docs/lit-review-plan.md Sec 5).

``EgoStitchStage1``: the per-node pass (`encode_nodes`) is the cacheable unit
(spec Sec 10.3): the E2E generator
(`src.model.egostitch.generator.egostitch.EgoStitchImagineGenerator`) runs it
once per node, then builds pair batches (Stitch + STE) from the cache. The
standalone frozen-s0 ``(s0, s1, s2)`` decision-head fusion
(`pair_outputs`/`self_outputs`/`forward`) belonged to the retired frozen-s0
``egostitch`` scorer and was removed with `decision.py` (three-component
refactor design §9); this model class is imagination-only.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import torch
import torch.nn as nn

from src.data.feature_stats import FeatureStats
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.assemble import match_slots
from src.model.egostitch.generator.losses import (
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
from src.model.egostitch.layers import build_mlp

NULL_MODE_FULL = 0
NULL_MODE_CONTENT = 1
NULL_MODE_ALL = 2


# --------------------------------------------------------------------------- Tokenize-lite


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


# --------------------------------------------------------------------------- Imagine


class SlotSet(NamedTuple):
    """Per-node generated slot set (spec Sec 2 heads).

    Attributes:
        h: Shape ``(B, K, d_p)`` slot embeddings in projected feature space.
        pi: Shape ``(B, K)`` slot existence probabilities in ``[0, 1]``.
        mult: Shape ``(B, K)`` slot multiplicities in ``[1, m_max]``.
        gate: Shape ``(B, K)`` grounding-gate probabilities in ``[0, 1]``.
        pointer: Shape ``(B, K, n_g)`` pointer softmax over grounding candidates.
        adj: Shape ``(B, K, K)`` symmetric slot-slot adjacency in ``[0, 1]``.
        adj_logits: Shape ``(B, K, K)`` pre-sigmoid adjacency logits.
    """

    h: torch.Tensor
    pi: torch.Tensor
    mult: torch.Tensor
    gate: torch.Tensor
    pointer: torch.Tensor
    adj: torch.Tensor
    adj_logits: torch.Tensor


class DenoiseSlots(NamedTuple):
    """Denoising-query outputs (fixed-assignment supervision only, spec Sec 2).

    Attributes:
        h: Shape ``(B, K_d, d_p)`` denoise-slot embeddings.
        pi: Shape ``(B, K_d)`` existence probabilities.
        mult: Shape ``(B, K_d)`` multiplicities.
    """

    h: torch.Tensor
    pi: torch.Tensor
    mult: torch.Tensor


class _DecoderLayer(nn.Module):
    """One Imagine decoder layer: slot self-attn -> memory cross-attn -> FFN."""

    def __init__(self, d_h: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d_h)
        self.norm_cross = nn.LayerNorm(d_h)
        self.norm_ffn = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(nn.Linear(d_h, 4 * d_h), nn.GELU(), nn.Linear(4 * d_h, d_h))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm_self(queries + attn_out)
        attn_out, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.norm_cross(queries + attn_out)
        out: torch.Tensor = self.norm_ffn(queries + self.ffn(queries))
        return out


class ImagineDecoder(nn.Module):
    """DETR-style slot decoder with codebook-free conditioning (spec Sec 13.2).

    Query init: ``Q_k = W_q [proj(x_{g_k}); e_u]`` for ``k <= min(K, n_g)``, else
    ``Q_k = q_k^base + W_q' e_u``. Conditioning dropout replaces ``T_cond``
    (``∅_content``) or both ``T_cond`` and ``T_g`` (``∅_all``) with learned null
    tokens.
    """

    def __init__(self, config: EgoStitchConfig) -> None:
        """Build the decoder, its conditioning projections, and the slot heads.

        Args:
            config: The pinned Stage-1 configuration.
        """
        super().__init__()
        self.config = config
        d, d_p, d_z, d_h, k = (
            config.input_dim,
            config.d_p,
            config.d_z,
            config.d_h,
            config.slots,
        )
        # Shared projection (spec Sec 13.7): matching / L_feat / s1 all consume it.
        self.proj = nn.Linear(d, d_p)

        self.w_x = nn.Linear(d, d_h)
        self.w_e = nn.Linear(d_z, d_h)
        self.w_g = nn.Linear(d_p, d_h)
        self.w_q = nn.Linear(d_p + d_z, d_h)
        self.w_q_base = nn.Linear(d_z, d_h)
        self.q_base = nn.Parameter(torch.randn(k, d_h) * 0.02)
        self.null_cond = nn.Parameter(torch.randn(1, d_h) * 0.02)
        self.null_ground = nn.Parameter(torch.randn(1, d_h) * 0.02)

        self.layers = nn.ModuleList(
            _DecoderLayer(d_h, config.n_heads) for _ in range(config.decoder_layers)
        )

        self.head_h = nn.Linear(d_h, d_p)
        self.head_pi = nn.Linear(d_h, 1)
        self.head_mult = nn.Linear(d_h, 1)
        self.head_gate = nn.Linear(d_h, 1)
        self.head_pointer = nn.Linear(d_h, d_h)
        self.head_adj = nn.Linear(d_p, d_p)

    def _build_queries(
        self, e: torch.Tensor, ground_proj: torch.Tensor, extra_proj: torch.Tensor | None
    ) -> torch.Tensor:
        """Assemble the K (+ optional denoise) slot queries."""
        batch, k = e.shape[0], self.config.slots
        n_dynamic = min(k, ground_proj.shape[1])
        dynamic = self.w_q(
            torch.cat([ground_proj[:, :n_dynamic], e[:, None, :].expand(-1, n_dynamic, -1)], dim=-1)
        )
        if n_dynamic < k:
            base = (
                self.q_base[None, n_dynamic:k].expand(batch, -1, -1) + self.w_q_base(e)[:, None, :]
            )
            queries = torch.cat([dynamic, base], dim=1)
        else:
            queries = dynamic
        if extra_proj is not None:
            extra = self.w_q(
                torch.cat([extra_proj, e[:, None, :].expand(-1, extra_proj.shape[1], -1)], dim=-1)
            )
            queries = torch.cat([queries, extra], dim=1)
        return queries

    def _build_memory(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        ground_tokens: torch.Tensor,
        null_mode: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble ``[T_cond; T_g]`` with conditioning-dropout null substitution."""
        cond = torch.stack([self.w_x(x), self.w_e(e)], dim=1)  # (B, 2, d_h)
        drop_cond = (null_mode >= NULL_MODE_CONTENT)[:, None, None]
        cond = torch.where(drop_cond, self.null_cond[None].expand_as(cond), cond)
        drop_ground = (null_mode == NULL_MODE_ALL)[:, None, None]
        ground_tokens = torch.where(
            drop_ground, self.null_ground[None].expand_as(ground_tokens), ground_tokens
        )
        return torch.cat([cond, ground_tokens], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        ground_x: torch.Tensor,
        *,
        null_mode: torch.Tensor | None = None,
        extra_proj: torch.Tensor | None = None,
    ) -> tuple[SlotSet, DenoiseSlots | None]:
        """Generate the slot set for a node batch.

        Args:
            x: Shape ``(B, d)`` frozen node features.
            e: Shape ``(B, d_z)`` Tokenize-lite embeddings.
            ground_x: Shape ``(B, n_g, d)`` grounding-candidate features.
            null_mode: Optional ``(B,)`` int tensor in ``{0, 1, 2}``
                (full / ``∅_content`` / ``∅_all``); defaults to full.
            extra_proj: Optional ``(B, K_d, d_p)`` noised true-neighbor
                projections for denoising queries (fixed assignments; caller
                supervises them, spec Sec 2).

        Returns:
            ``(slots, denoise)`` — `denoise` is ``None`` when `extra_proj` is.
        """
        if null_mode is None:
            null_mode = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        ground_proj = self.proj(ground_x)
        ground_tokens = self.w_g(ground_proj)
        memory = self._build_memory(x, e, ground_tokens, null_mode)
        queries = self._build_queries(e, ground_proj, extra_proj)

        decoded = queries
        for layer in self.layers:
            decoded = layer(decoded, memory)

        k = self.config.slots
        main, extra = decoded[:, :k], decoded[:, k:]

        h = self.head_h(main)
        pi = torch.sigmoid(self.head_pi(main)).squeeze(-1)
        mult = 1.0 + torch.nn.functional.softplus(self.head_mult(main)).squeeze(-1)
        mult = torch.clamp(mult, max=float(self.config.m_max))
        gate = torch.sigmoid(self.head_gate(main)).squeeze(-1)
        pointer_scores = (
            torch.einsum("bkd,bgd->bkg", self.head_pointer(main), ground_tokens)
            / float(self.config.d_h) ** 0.5
        )
        pointer = torch.softmax(pointer_scores, dim=-1)
        adj_proj = self.head_adj(h)
        adj_logits = (
            torch.einsum("bkd,bld->bkl", adj_proj, adj_proj) / float(self.config.d_p) ** 0.5
        )
        adj = torch.sigmoid(adj_logits)

        slots = SlotSet(
            h=h,
            pi=pi,
            mult=mult,
            gate=gate,
            pointer=pointer,
            adj=adj,
            adj_logits=adj_logits,
        )

        denoise: DenoiseSlots | None = None
        if extra_proj is not None:
            d_h_extra = self.head_h(extra)
            d_pi = torch.sigmoid(self.head_pi(extra)).squeeze(-1)
            d_mult = 1.0 + torch.nn.functional.softplus(self.head_mult(extra)).squeeze(-1)
            d_mult = torch.clamp(d_mult, max=float(self.config.m_max))
            denoise = DenoiseSlots(h=d_h_extra, pi=d_pi, mult=d_mult)
        return slots, denoise


# --------------------------------------------------------------------------- Stage-1 model


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
            raise ValueError(f"unknown feature standardization mode {feature_standardization!r}")
        self.tokenize = TokenizeLite(config)
        self.imagine = ImagineDecoder(config)
        self.random_gin = RandomGIN(config)

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
