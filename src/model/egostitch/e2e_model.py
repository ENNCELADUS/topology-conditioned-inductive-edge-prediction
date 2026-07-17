"""EgoStitchE2E: stitched-topology-conditioned pair encoder (design rev 3).

The rev-3.0 end-to-end conditioned encoder: an internally *trainable* Stage-1
generator (Tokenize-lite + Imagine — no frozen s0, no s0 cache) feeds a
stitched-topology scaffold and content tokens into the conditioned pair trunk
via zero-init gated cross-attention (Tasks 5-10). AB/BA order is handled
internally: the trunk runs both orders through the same
`ConditionedPairCrossAttention` (topo tokens relabeled per side via
`swap_direction`) and the two pair representations are fused by a
feature-wise max before the head, mirroring `V3_1._pair_representation`
(`B0.py:1093`). `decompose` realizes the four-logit decomposition
(`full`/`f_logit`/`pair_content`/`pair_topology`) via the eval-time hard
bypasses in `masks_for_null` (design rev 3 Sec 3.5) — one checkpoint, no
retraining per arm.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.B0 import MLPHead, SiameseEncoder
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_CONTENT_HEAD,
    NULL_NONE,
    NULL_TOPO_HEAD,
    HeadNullMasks,
    masks_for_null,
)
from src.model.egostitch.config import E2EConfig, EgoStitchConfig
from src.model.egostitch.model import EgoStitchStage1
from src.model.egostitch.scaffold import (
    ContentProjector,
    build_content_tokens,
    build_scaffold,
    swap_direction,
)
from src.model.egostitch.ste import STEncoder
from src.model.egostitch.stitch import sinkhorn_plan
from src.model.egostitch.trunk import ConditionedPairCrossAttention


class EgoStitchE2E(nn.Module):
    """The rev-3.0 end-to-end topology-conditioned pair encoder.

    Composes a trainable Stage-1 generator (per-node imagination), the
    stitched-topology scaffold encoder, the content-token projector, and the
    conditioned pair trunk into a single model supporting the four-logit
    decomposition (`full`, `f_logit`, `pair_content`, `pair_topology`).
    """

    def __init__(self, cfg: E2EConfig) -> None:
        """Build every E2E submodule.

        Args:
            cfg: The pair-trunk / conditioning hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        # Stage-1 generator keeps its own pinned spec defaults (spec Sec 13);
        # E2EConfig sizes only the trunk and its conditioning pathways.
        self.generator_cfg = EgoStitchConfig()
        self.input_dim = self.generator_cfg.input_dim  # frozen feature dim (spec Sec 0 table)
        self.node_feature_dim = self.generator_cfg.input_dim
        self.generator = EgoStitchStage1(self.generator_cfg)
        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=cfg.d_model,
            n_layers=cfg.encoder_layers,
            n_heads=cfg.n_heads,
            dropout=0.1,
            token_dropout=0.0,
        )
        self.trunk = ConditionedPairCrossAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.cross_attn_layers,
            dropout=0.1,
            pair_readout_mode="pair_context_gated",
            mixing_mode="bidirectional_cross",
            n_inj=cfg.n_inj,
            xattn_heads=cfg.xattn_heads,
        )
        self.ste = STEncoder(cfg.d_model, cfg.ste_dim, cfg.ste_layers)
        self.content_proj = ContentProjector(d_p=self.generator_cfg.d_p, d_model=cfg.d_model)
        self.head = MLPHead(
            input_dim=cfg.d_model,
            hidden_dims=[cfg.d_model // 2],
            output_dim=1,
            dropout=0.1,
            activation="gelu",
            norm="layernorm",
        )

    def _ground(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder grounding-candidate pool for the generator.

        WARNING: every row of the returned pool is an identical copy of `x`,
        so every primary slot query in `ImagineDecoder._build_queries` is also
        identical (with spec defaults, slots=16 <= n_ground=20, this holds for
        ALL primary slots) -- `encode_nodes` therefore emits structurally
        collapsed, non-diverse slots per node, not merely reduced-fidelity
        ones. Real grounding-candidate retrieval arrives with the Phase-4
        worker integration (Task 13); until then, any full-pipeline output
        has degenerate topology and must not be read as evidence of
        meaningful topology generation.

        Args:
            x: Shape ``(B, d)`` per-node features.

        Returns:
            Shape ``(B, n_ground, d)`` grounding-candidate features.
        """
        ground: torch.Tensor = x.unsqueeze(1).expand(-1, self.generator_cfg.n_ground, -1)
        return ground

    def _context(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the topo (AB/BA) and content conditioning tokens.

        Args:
            batch: Batch dict with ``x_a``/``x_b`` per-node feature tensors.

        Returns:
            ``(topo_ab, topo_ba, cont)`` conditioning token tensors.
        """
        enc_a = self.generator.encode_nodes(batch["x_a"], self._ground(batch["x_a"]))
        enc_b = self.generator.encode_nodes(batch["x_b"], self._ground(batch["x_b"]))
        slots_a, slots_b = enc_a.slots, enc_b.slots
        plan = sinkhorn_plan(
            slots_a.h,
            slots_b.h,
            slots_a.pi,
            slots_b.pi,
            slots_a.mult,
            slots_b.mult,
            eps=self.generator_cfg.sinkhorn_eps,
            iters=self.generator_cfg.sinkhorn_iters,
            tau=self.generator_cfg.sinkhorn_tau,
        )
        scaffold = build_scaffold(slots_a, slots_b, plan)
        topo_ab: torch.Tensor = self.ste(scaffold)
        topo_ba: torch.Tensor = self.ste(swap_direction(scaffold))
        # Grounded-identity-match signal: wired to the pointer-match output
        # in a later task; zeros are a neutral (AB/BA-symmetric) placeholder.
        matched_a = torch.zeros_like(slots_a.pi)
        matched_b = torch.zeros_like(slots_b.pi)
        cont: torch.Tensor = self.content_proj(
            build_content_tokens(slots_a, slots_b, matched_a, matched_b)
        )
        return topo_ab, topo_ba, cont

    def forward(
        self, batch: dict[str, torch.Tensor], *, masks: HeadNullMasks | None = None
    ) -> dict[str, torch.Tensor]:
        """Score one pair batch under an optional head-null condition.

        Args:
            batch: Pair batch with ``emb_a``/``emb_b`` token streams,
                ``len_a``/``len_b`` lengths, and the ``x_a``/``x_b`` per-node
                features the Stage-1 generator's `encode_nodes` consumes.
            masks: Optional per-pair topo/content activity masks; ``None``
                defaults to both pathways fully active (``NULL_NONE``).

        Returns:
            ``{"logits": (B,)}`` fused edge logits.
        """
        batch_size = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        need_topo = bool(masks.topo.any())
        need_cont = bool(masks.cont.any())
        topo_ab: torch.Tensor | None = None
        topo_ba: torch.Tensor | None = None
        cont: torch.Tensor | None = None
        if need_topo or need_cont:
            topo_ab, topo_ba, cont = self._context(batch)
        h_a = self.encoder(batch["emb_a"], batch["len_a"])
        h_b = self.encoder(batch["emb_b"], batch["len_b"])
        feat_ab = self.trunk(
            h_a,
            h_b,
            batch["len_a"],
            batch["len_b"],
            topo_tokens=topo_ab if need_topo else None,
            cont_tokens=cont if need_cont else None,
            topo_active=masks.topo if need_topo else None,
            cont_active=masks.cont if need_cont else None,
        )
        feat_ba = self.trunk(
            h_b,
            h_a,
            batch["len_b"],
            batch["len_a"],
            topo_tokens=topo_ba if need_topo else None,
            cont_tokens=cont if need_cont else None,
            topo_active=masks.topo if need_topo else None,
            cont_active=masks.cont if need_cont else None,
        )
        feat: torch.Tensor = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        logits: torch.Tensor = self.head(feat).squeeze(-1)
        return {"logits": logits}

    def decompose(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the four-logit decomposition via eval-time hard bypasses.

        Args:
            batch: Pair batch (see :meth:`forward`).

        Returns:
            ``{"full", "f_logit", "pair_content", "pair_topology"}`` logits.
        """
        batch_size = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        with torch.no_grad():
            return {
                "full": self(batch, masks=None)["logits"],
                "f_logit": self(batch, masks=masks_for_null(NULL_ALL_HEAD, batch_size, device))[
                    "logits"
                ],
                "pair_content": self(
                    batch, masks=masks_for_null(NULL_TOPO_HEAD, batch_size, device)
                )["logits"],
                "pair_topology": self(
                    batch, masks=masks_for_null(NULL_CONTENT_HEAD, batch_size, device)
                )["logits"],
            }
