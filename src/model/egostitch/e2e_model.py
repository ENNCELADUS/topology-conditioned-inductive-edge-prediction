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


def grounded_identity_match(
    pointer_a: torch.Tensor,
    gate_a: torch.Tensor,
    ids_a: torch.Tensor,
    pointer_b: torch.Tensor,
    gate_b: torch.Tensor,
    ids_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grounded-identity-match binary flags (spec Sec 13.18 pinned definition).

    Slot `k` of endpoint a (resp. `k'` of endpoint b) is grounded-identity-
    matched iff (1) its own gate exceeds 0.5, (2) its pointer argmax selects a
    grounding-pool candidate with some global node id `c`, and (3) the OTHER
    endpoint has at least one slot with gate > 0.5 whose pointer argmax
    selects that same global id `c`. Symmetric across AB/BA by construction:
    the underlying shared-candidate relation (matching global ids between a
    gated slot on each side) is undirected.

    Args:
        pointer_a: Shape ``(B, K, n_g)`` side-a pointer softmax over its
            grounding pool.
        gate_a: Shape ``(B, K)`` side-a grounding-gate probabilities.
        ids_a: Shape ``(B, n_g)`` int64 global node ids of side-a's grounding
            pool candidates (indexed by `pointer_a`'s last dimension).
        pointer_b: Shape ``(B, K, n_g)`` side-b pointer softmax.
        gate_b: Shape ``(B, K)`` side-b grounding-gate probabilities.
        ids_b: Shape ``(B, n_g)`` int64 global node ids of side-b's grounding
            pool candidates.

    Returns:
        ``(matched_a, matched_b)``, each shape ``(B, K)`` binary floats in
        ``{0.0, 1.0}``.
    """
    gated_a = gate_a > 0.5
    gated_b = gate_b > 0.5
    sel_id_a = torch.gather(ids_a, 1, pointer_a.argmax(dim=-1))
    sel_id_b = torch.gather(ids_b, 1, pointer_b.argmax(dim=-1))
    shared = sel_id_a[:, :, None] == sel_id_b[:, None, :]  # (B, K, K); [b, k, k']
    other_gated_for_a = (shared & gated_b[:, None, :]).any(dim=-1)  # (B, K), indexed by k
    other_gated_for_b = (shared & gated_a[:, :, None]).any(dim=1)  # (B, K), indexed by k'
    matched_a = (gated_a & other_gated_for_a).float()
    matched_b = (gated_b & other_gated_for_b).float()
    return matched_a, matched_b


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
        """Degenerate fallback grounding-candidate pool for the generator.

        WARNING: every row of the returned pool is an identical copy of `x`,
        so every primary slot query in `ImagineDecoder._build_queries` is also
        identical (with spec defaults, slots=16 <= n_ground=20, this holds for
        ALL primary slots) -- `encode_nodes` therefore emits structurally
        collapsed, non-diverse slots per node, not merely reduced-fidelity
        ones. This path engages ONLY when the batch omits the ``ground_a``/
        ``ground_b`` keys (e.g. the tiny unit fixtures in
        ``tests/model/test_egostitch_e2e_model.py``). Every worker/scorer
        entry point (``src.score_universe._score_egostitch_e2e``) always
        supplies real grounding-pool candidates (spec Sec 13.12; tested), so
        production full-pipeline output never takes this branch.

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
            batch: Batch dict with ``x_a``/``x_b`` per-node feature tensors,
                and optionally ``ground_a``/``ground_b`` (real grounding-pool
                features, ``(B, n_g, d)``) and ``ground_id_a``/``ground_id_b``
                (their global node ids, ``(B, n_g)`` int64). Missing feature
                keys fall back to the degenerate `_ground` placeholder;
                missing id keys fall back to zero matched flags (see
                `_ground` and `grounded_identity_match`).

        Returns:
            ``(topo_ab, topo_ba, cont)`` conditioning token tensors.
        """
        ground_a = batch.get("ground_a")
        ground_b = batch.get("ground_b")
        ground_x_a = ground_a if ground_a is not None else self._ground(batch["x_a"])
        ground_x_b = ground_b if ground_b is not None else self._ground(batch["x_b"])
        enc_a = self.generator.encode_nodes(batch["x_a"], ground_x_a)
        enc_b = self.generator.encode_nodes(batch["x_b"], ground_x_b)
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
        ground_id_a = batch.get("ground_id_a")
        ground_id_b = batch.get("ground_id_b")
        if ground_id_a is not None and ground_id_b is not None:
            matched_a, matched_b = grounded_identity_match(
                slots_a.pointer,
                slots_a.gate,
                ground_id_a,
                slots_b.pointer,
                slots_b.gate,
                ground_id_b,
            )
        else:
            # Neutral (AB/BA-symmetric) placeholder when ids are unavailable
            # (mirrors the `_ground` degenerate path — tiny unit fixtures only).
            matched_a = torch.zeros_like(slots_a.pi)
            matched_b = torch.zeros_like(slots_b.pi)
        cont: torch.Tensor = self.content_proj(
            build_content_tokens(slots_a, slots_b, matched_a, matched_b)
        )
        return topo_ab, topo_ba, cont

    @torch.no_grad()
    def probe_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Export STE token states for a fixed probe batch (read-only diagnostic).

        Runs only the Stage-1 generator + Stitch + STE pathway used to build
        the ``topo_ab`` conditioning tensor `_context` computes internally for
        `forward` — no trunk, no head, no gradient. Intended for the
        registered representation-probe protocol (`src.experiments.probes`,
        spec Sec 14.3(3)): callers typically pass a batch of self-pairs
        (``x_a == x_b``, the Sec 13.9 single-ego path) so each row's tokens
        describe one probe node's own generated ego-net, then reduce over the
        token axis (e.g. mean-pool) before probing.

        Args:
            batch: Pair batch with the `_context` keys (``x_a``/``x_b`` and
                optionally ``ground_a``/``ground_b``/``ground_id_a``/
                ``ground_id_b``).

        Returns:
            Shape ``(B, V, d_model)`` AB-direction STE token states.
        """
        topo_ab, _topo_ba, _cont = self._context(batch)
        return topo_ab

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
