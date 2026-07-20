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

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

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
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.model import EgoStitchStage1
from src.model.egostitch.scaffold import (
    ContentProjector,
    build_content_tokens,
    build_scaffold,
    counterpart_membership,
    grounded_identity_match,
    swap_direction,
)
from src.model.egostitch.ste import STEncoder
from src.model.egostitch.stitch import sinkhorn_plan
from src.model.egostitch.trunk import ConditionedPairCrossAttention


class E2ENodeState(NamedTuple):
    """Cacheable per-node trunk and imagination state."""

    encoded: torch.Tensor
    length: torch.Tensor
    slots: SlotSet
    projected_x: torch.Tensor
    ground_ids: torch.Tensor | None


class E2EPairContext(NamedTuple):
    """One shared pair context consumed by every hard-bypass head."""

    encoded_a: torch.Tensor
    encoded_b: torch.Tensor
    len_a: torch.Tensor
    len_b: torch.Tensor
    topo_ab: torch.Tensor | None
    topo_ba: torch.Tensor | None
    cont: torch.Tensor | None
    plan: torch.Tensor | None


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

    @staticmethod
    def _select_slots(slots: SlotSet, rows: torch.Tensor) -> SlotSet:
        """Index every tensor in one slot bundle by batch row."""
        return SlotSet(*(value.index_select(0, rows) for value in slots))

    @staticmethod
    def _merge_slots(base: SlotSet, replacement: SlotSet, rows: torch.Tensor) -> SlotSet:
        """Replace selected batch rows while preserving autograd to both inputs."""
        return SlotSet(*(a.index_copy(0, rows, b) for a, b in zip(base, replacement, strict=True)))

    def encode_node_state(
        self,
        emb: torch.Tensor,
        length: torch.Tensor,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> E2ENodeState:
        """Run the cacheable raw-token and generator pass for one node batch."""
        generated = self.generator.encode_nodes(x, ground)
        return E2ENodeState(
            encoded=self.encoder(emb, length),
            length=length,
            slots=generated.slots,
            projected_x=self.generator.imagine.proj(x),
            ground_ids=ground_ids,
        )

    def _merge_node_states(
        self, base: E2ENodeState, replacement: E2ENodeState, rows: torch.Tensor
    ) -> E2ENodeState:
        """Use `base` for self rows and `replacement` for non-self rows."""
        width = max(base.encoded.size(1), replacement.encoded.size(1))
        encoded_base = F.pad(base.encoded, (0, 0, 0, width - base.encoded.size(1)))
        encoded_replacement = F.pad(
            replacement.encoded, (0, 0, 0, width - replacement.encoded.size(1))
        )
        if base.ground_ids is None or replacement.ground_ids is None:
            ground_ids = None
        else:
            ground_ids = base.ground_ids.index_copy(0, rows, replacement.ground_ids)
        return E2ENodeState(
            encoded=encoded_base.index_copy(0, rows, encoded_replacement),
            length=base.length.index_copy(0, rows, replacement.length),
            slots=self._merge_slots(base.slots, replacement.slots, rows),
            projected_x=base.projected_x.index_copy(0, rows, replacement.projected_x),
            ground_ids=ground_ids,
        )

    def _pair_node_states(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[E2ENodeState, E2ENodeState, torch.Tensor]:
        """Encode pair endpoints, using exactly one encode for every self row."""
        missing = {"ground_a", "ground_b"} - batch.keys()
        if missing:
            raise ValueError(f"EgoStitchE2E requires real grounding tensors: {sorted(missing)}")
        batch_size = batch["emb_a"].size(0)
        is_self = batch.get(
            "is_self", torch.zeros(batch_size, dtype=torch.bool, device=batch["emb_a"].device)
        )
        state_a = self.encode_node_state(
            batch["emb_a"],
            batch["len_a"],
            batch["x_a"],
            batch["ground_a"],
            batch.get("ground_id_a"),
        )
        non_self = torch.nonzero(~is_self, as_tuple=False).squeeze(-1)
        if non_self.numel() == 0:
            return state_a, state_a, is_self
        state_b_non_self = self.encode_node_state(
            batch["emb_b"].index_select(0, non_self),
            batch["len_b"].index_select(0, non_self),
            batch["x_b"].index_select(0, non_self),
            batch["ground_b"].index_select(0, non_self),
            (batch["ground_id_b"].index_select(0, non_self) if "ground_id_b" in batch else None),
        )
        if non_self.numel() == batch_size:
            return state_a, state_b_non_self, is_self
        return state_a, self._merge_node_states(state_a, state_b_non_self, non_self), is_self

    def build_pair_context_from_states(
        self,
        state_a: E2ENodeState,
        state_b: E2ENodeState,
        is_self: torch.Tensor,
        *,
        need_topo: bool = True,
        need_cont: bool = True,
    ) -> E2EPairContext:
        """Build Stitch/STE/content once from cacheable endpoint states."""
        slots_a, slots_b = state_a.slots, state_b.slots
        batch_size, slots = slots_a.pi.shape
        if is_self.shape != (batch_size,):
            raise ValueError(f"is_self shape must be {(batch_size,)}, got {tuple(is_self.shape)}")
        plan: torch.Tensor | None = None
        topo_ab: torch.Tensor | None = None
        topo_ba: torch.Tensor | None = None
        if need_topo:
            # Sinkhorn is an fp32 island, so its assembled destination must also be fp32.
            plan = slots_a.pi.new_zeros((batch_size, slots, slots), dtype=torch.float32)
            self_rows = torch.nonzero(is_self, as_tuple=False).squeeze(-1)
            if self_rows.numel() > 0:
                identity = torch.eye(slots, device=plan.device, dtype=plan.dtype)
                plan = plan.index_copy(0, self_rows, identity.expand(self_rows.numel(), -1, -1))
            non_self = torch.nonzero(~is_self, as_tuple=False).squeeze(-1)
            if non_self.numel() > 0:
                selected_a = self._select_slots(slots_a, non_self)
                selected_b = self._select_slots(slots_b, non_self)
                non_self_plan = sinkhorn_plan(
                    selected_a.h,
                    selected_b.h,
                    selected_a.pi,
                    selected_b.pi,
                    selected_a.mult,
                    selected_b.mult,
                    eps=self.generator_cfg.sinkhorn_eps,
                    iters=self.generator_cfg.sinkhorn_iters,
                    tau=self.generator_cfg.sinkhorn_tau,
                )
                plan = plan.index_copy(0, non_self, non_self_plan)
            scaffold = build_scaffold(slots_a, slots_b, plan)
            topo_ab = self.ste(scaffold)
            topo_ba = self.ste(swap_direction(scaffold))

        cont: torch.Tensor | None = None
        if need_cont:
            if state_a.ground_ids is not None and state_b.ground_ids is not None:
                matched_a, matched_b = grounded_identity_match(
                    slots_a.pointer,
                    slots_a.gate,
                    state_a.ground_ids,
                    slots_b.pointer,
                    slots_b.gate,
                    state_b.ground_ids,
                )
            else:
                matched_a = torch.zeros_like(slots_a.pi)
                matched_b = torch.zeros_like(slots_b.pi)
            membership_a = counterpart_membership(
                slots_a, state_b.projected_x, self.generator.decision.tau_kappa
            )
            membership_b = counterpart_membership(
                slots_b, state_a.projected_x, self.generator.decision.tau_kappa
            )
            cont = self.content_proj(
                build_content_tokens(
                    slots_a,
                    slots_b,
                    matched_a,
                    matched_b,
                    membership_a,
                    membership_b,
                )
            )
        return E2EPairContext(
            encoded_a=state_a.encoded,
            encoded_b=state_b.encoded,
            len_a=state_a.length,
            len_b=state_b.length,
            topo_ab=topo_ab,
            topo_ba=topo_ba,
            cont=cont,
            plan=plan,
        )

    def build_pair_context(
        self,
        batch: dict[str, torch.Tensor],
        *,
        need_topo: bool = True,
        need_cont: bool = True,
    ) -> E2EPairContext:
        """Encode endpoints and build one reusable pair context."""
        state_a, state_b, is_self = self._pair_node_states(batch)
        return self.build_pair_context_from_states(
            state_a, state_b, is_self, need_topo=need_topo, need_cont=need_cont
        )

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
            batch: Pair batch with endpoint features plus required
                ``ground_a``/``ground_b`` tensors. Grounding ids are optional.

        Returns:
            Shape ``(B, V, d_model)`` AB-direction STE token states.
        """
        context = self.build_pair_context(batch, need_topo=True, need_cont=False)
        assert context.topo_ab is not None
        return context.topo_ab

    def score_pair_context(
        self, context: E2EPairContext, *, masks: HeadNullMasks | None = None
    ) -> torch.Tensor:
        """Evaluate one hard-bypass head from a shared pair context."""
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        need_topo = bool(masks.topo.any())
        need_cont = bool(masks.cont.any())
        if need_topo and (context.topo_ab is None or context.topo_ba is None):
            raise ValueError("pair context does not contain topology tokens")
        if need_cont and context.cont is None:
            raise ValueError("pair context does not contain content tokens")
        feat_ab = self.trunk(
            context.encoded_a,
            context.encoded_b,
            context.len_a,
            context.len_b,
            topo_tokens=context.topo_ab if need_topo else None,
            cont_tokens=context.cont if need_cont else None,
            topo_active=masks.topo if need_topo else None,
            cont_active=masks.cont if need_cont else None,
        )
        feat_ba = self.trunk(
            context.encoded_b,
            context.encoded_a,
            context.len_b,
            context.len_a,
            topo_tokens=context.topo_ba if need_topo else None,
            cont_tokens=context.cont if need_cont else None,
            topo_active=masks.topo if need_topo else None,
            cont_active=masks.cont if need_cont else None,
        )
        feat = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        logits: torch.Tensor = self.head(feat).squeeze(-1)
        return logits

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
        context = self.build_pair_context(batch, need_topo=need_topo, need_cont=need_cont)
        return {"logits": self.score_pair_context(context, masks=masks)}

    def decompose(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the four-logit decomposition via eval-time hard bypasses.

        Args:
            batch: Pair batch (see :meth:`forward`).

        Returns:
            ``{"full", "f_logit", "pair_content", "pair_topology"}`` logits.
        """
        with torch.no_grad():
            context = self.build_pair_context(batch)
            return self.decompose_pair_context(context)

    def decompose_pair_context(self, context: E2EPairContext) -> dict[str, torch.Tensor]:
        """Evaluate all four logits without rebuilding node or pair state."""
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        return {
            "full": self.score_pair_context(context),
            "f_logit": self.score_pair_context(
                context, masks=masks_for_null(NULL_ALL_HEAD, batch_size, device)
            ),
            "pair_content": self.score_pair_context(
                context, masks=masks_for_null(NULL_TOPO_HEAD, batch_size, device)
            ),
            "pair_topology": self.score_pair_context(
                context, masks=masks_for_null(NULL_CONTENT_HEAD, batch_size, device)
            ),
        }
