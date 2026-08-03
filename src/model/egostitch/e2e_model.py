"""EgoStitchE2E: stitched-topology-conditioned pair encoder (design rev 3).

The rev-3.0 end-to-end conditioned encoder: an internally *trainable* Stage-1
generator (Tokenize-lite + Imagine — no frozen s0, no s0 cache) feeds a
stitched-topology scaffold into the conditioned pair trunk via zero-init
gated cross-attention (Tasks 5-10). AB/BA order is handled internally: the
trunk runs both orders through the same `ConditionedPairCrossAttention` (topo
tokens relabeled per side via `swap_direction`) and the two pair
representations are fused by a feature-wise max before the head, mirroring
`V3_1._pair_representation` (`B0.py:1093`). `decompose` realizes the
two-logit decomposition (`full`/`f_logit`) via the eval-time hard bypass in
`masks_for_null` (design rev 3 Sec 3.5) — one checkpoint, no retraining per
arm. The content pathway (pair-content tokens routed around the graph
encoder straight into the trunk) was removed with the content path
(three-component refactor design §9).
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from src.data.feature_stats import FeatureStats
from src.model.B0 import MLPHead, SiameseEncoder
from src.model.egostitch.conditioning import (
    NULL_ALL_HEAD,
    NULL_NONE,
    HeadNullMasks,
    masks_for_null,
)
from src.model.egostitch.config import E2EConfig, EgoStitchConfig
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.losses import (
    alignment_loss,
    alignment_teacher_cells,
    relational_loss,
)
from src.model.egostitch.matching import match_slots
from src.model.egostitch.model import EgoStitchStage1, FeatureStandardizationMode
from src.model.egostitch.scaffold import (
    ScaffoldInputPerturbation,
    build_scaffold,
    swap_direction,
)
from src.model.egostitch.ste import STEncoder
from src.model.egostitch.stitch import sinkhorn_log_plan
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
    plan: torch.Tensor | None
    log_plan: torch.Tensor | None


class EgoStitchE2E(nn.Module):
    """The rev-3.0 end-to-end topology-conditioned pair encoder.

    Composes a trainable Stage-1 generator (per-node imagination), the
    stitched-topology scaffold encoder, and the conditioned pair trunk into a
    single model supporting the two-logit decomposition (`full`, `f_logit`).
    """

    def __init__(self, cfg: E2EConfig) -> None:
        """Build every E2E submodule.

        Args:
            cfg: The pair-trunk / conditioning hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        # Stage-1 generator keeps its own pinned spec defaults (spec Sec 13);
        # E2EConfig sizes the trunk and carries the registered rev-3.1
        # grounding/loss calibration fields into the internal generator.
        self.generator_cfg = replace(
            EgoStitchConfig(),
            n_ground=cfg.n_ground,
            tau_adj=cfg.tau_adj,
            tau_div=cfg.tau_div,
            l_gate_pos_weight=cfg.l_gate_pos_weight,
            w_rel=cfg.w_rel,
        )
        self.input_dim = self.generator_cfg.input_dim  # frozen feature dim (spec Sec 0 table)
        self.node_feature_dim = self.generator_cfg.input_dim
        self.generator = EgoStitchStage1(
            self.generator_cfg,
            # E2EConfig.__post_init__ restricts this to the
            # _FEATURE_STANDARDIZATION_MODES allowlist, a subset of
            # FeatureStandardizationMode; the field itself stays plain `str`
            # so YAML/checkpoint round-tripping needs no special-casing.
            feature_standardization=cast(FeatureStandardizationMode, cfg.feature_standardization),
            loss_family="egostitch_e2e",
        )
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
            conditioning_ema_decay=cfg.conditioning_ema_decay,
        )
        self.ste = STEncoder(cfg.d_model, cfg.ste_dim, cfg.ste_layers)
        self.head = MLPHead(
            input_dim=cfg.d_model,
            hidden_dims=[cfg.d_model // 2],
            output_dim=1,
            dropout=0.1,
            activation="gelu",
            norm="layernorm",
        )
        self.rel_head: nn.Sequential | None = None
        if cfg.w_rel > 0.0:
            rel_hidden = max(1, cfg.d_model // 2)
            self.rel_head = nn.Sequential(
                nn.Linear(cfg.d_model, rel_hidden),
                nn.GELU(),
                nn.Linear(rel_hidden, 2),
            )

    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered F0 standardization constants on the generator.

        Args:
            stats: The V_fit statistics bundle.
        """
        self.generator.set_feature_stats(stats)

    @property
    def feature_stats_digest_hex(self) -> str:
        """The generator's registered `feature_stats_sha256`, or ``""``."""
        return self.generator.feature_stats_digest_hex

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
            projected_x=self.generator.project_features(x),
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
        scaffold_input_perturbation: ScaffoldInputPerturbation | None = None,
    ) -> E2EPairContext:
        """Build Stitch/STE state once from cacheable endpoint states."""
        slots_a, slots_b = state_a.slots, state_b.slots
        batch_size, slots = slots_a.pi.shape
        if is_self.shape != (batch_size,):
            raise ValueError(f"is_self shape must be {(batch_size,)}, got {tuple(is_self.shape)}")
        plan: torch.Tensor | None = None
        log_plan: torch.Tensor | None = None
        topo_ab: torch.Tensor | None = None
        topo_ba: torch.Tensor | None = None
        if need_topo:
            # Sinkhorn is an fp32 island, so its assembled destination must also be fp32.
            plan = slots_a.pi.new_zeros((batch_size, slots, slots), dtype=torch.float32)
            log_plan = slots_a.pi.new_full(
                (batch_size, slots, slots),
                -torch.inf,
                dtype=torch.float32,
            )
            self_rows = torch.nonzero(is_self, as_tuple=False).squeeze(-1)
            if self_rows.numel() > 0:
                identity = torch.eye(slots, device=plan.device, dtype=plan.dtype)
                plan = plan.index_copy(0, self_rows, identity.expand(self_rows.numel(), -1, -1))
                log_identity = torch.diag_embed(
                    torch.zeros(
                        self_rows.numel(),
                        slots,
                        device=log_plan.device,
                        dtype=log_plan.dtype,
                    )
                )
                log_identity = log_identity.masked_fill(
                    ~torch.eye(slots, device=log_plan.device, dtype=torch.bool).unsqueeze(0),
                    -torch.inf,
                )
                log_plan = log_plan.index_copy(0, self_rows, log_identity)
            non_self = torch.nonzero(~is_self, as_tuple=False).squeeze(-1)
            if non_self.numel() > 0:
                selected_a = self._select_slots(slots_a, non_self)
                selected_b = self._select_slots(slots_b, non_self)
                non_self_log_plan = sinkhorn_log_plan(
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
                non_self_plan = torch.exp(non_self_log_plan)
                plan = plan.index_copy(0, non_self, non_self_plan)
                log_plan = log_plan.index_copy(0, non_self, non_self_log_plan)
            scaffold = build_scaffold(
                slots_a,
                slots_b,
                plan,
                perturbation=scaffold_input_perturbation,
            )
            topo_ab = self.ste(scaffold)
            topo_ba = self.ste(swap_direction(scaffold))

        return E2EPairContext(
            encoded_a=state_a.encoded,
            encoded_b=state_b.encoded,
            len_a=state_a.length,
            len_b=state_b.length,
            topo_ab=topo_ab,
            topo_ba=topo_ba,
            plan=plan,
            log_plan=log_plan,
        )

    def build_pair_context(
        self,
        batch: dict[str, torch.Tensor],
        *,
        need_topo: bool = True,
    ) -> E2EPairContext:
        """Encode endpoints and build one reusable pair context."""
        state_a, state_b, is_self = self._pair_node_states(batch)
        return self.build_pair_context_from_states(
            state_a, state_b, is_self, need_topo=need_topo
        )

    def relational_predictions(self, context: E2EPairContext) -> torch.Tensor:
        """Predict train-only relational targets from the AB STE state."""
        if context.topo_ab is None:
            raise ValueError("relational predictions require AB-direction STE tokens")
        if self.rel_head is None:
            raise RuntimeError("relational head is absent when w_rel == 0")
        with torch.autocast(device_type=context.topo_ab.device.type, enabled=False):
            pair_state = context.topo_ab.float().mean(dim=1)
            prediction: torch.Tensor = self.rel_head(pair_state)
            return prediction

    def _training_relational_losses(
        self,
        batch: dict[str, torch.Tensor],
        state_a: E2ENodeState,
        state_b: E2ENodeState,
        context: E2EPairContext,
    ) -> dict[str, torch.Tensor]:
        """Compute Task-7 train-only losses without entering any scored logit."""
        if context.plan is None or context.log_plan is None:
            raise RuntimeError("Task-7 alignment requires Sinkhorn plan and log-plan tensors")
        target_proj_a = self.generator.project_features(batch["target_features_a"]).detach()
        target_proj_b = self.generator.project_features(batch["target_features_b"]).detach()
        assignment_a = match_slots(
            state_a.slots,
            target_proj=target_proj_a,
            target_mult=batch["target_mult_a"],
            target_adj=batch["target_adj_a"],
            target_mask=batch["target_mask_a"],
        )
        assignment_b = match_slots(
            state_b.slots,
            target_proj=target_proj_b,
            target_mult=batch["target_mult_b"],
            target_adj=batch["target_adj_b"],
            target_mask=batch["target_mask_b"],
        )
        teacher_cells = alignment_teacher_cells(
            assignment_a,
            assignment_b,
            target_ids_a=batch["target_node_index_a"],
            target_ids_b=batch["target_node_index_b"],
            slots=self.generator.config.slots,
        )
        world_size = int(batch["loss_world_size"].item())
        positive_real_mask = batch["edge_mask"] * (batch["label"] == 1).float()
        rel = (
            relational_loss(
                self.relational_predictions(context),
                batch["rel_target"],
                batch["edge_mask"],
                world_size=world_size,
            )
            if self.rel_head is not None
            else context.plan.sum() * 0.0
        )
        return {
            "align_loss": alignment_loss(
                context.log_plan,
                teacher_cells,
                positive_real_mask=positive_real_mask,
                world_size=world_size,
            ),
            "rel_loss": rel,
        }

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
        context = self.build_pair_context(batch, need_topo=True)
        assert context.topo_ab is not None
        return context.topo_ab

    def score_pair_context(
        self,
        context: E2EPairContext,
        *,
        masks: HeadNullMasks | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate one hard-bypass head from a shared pair context.

        AB and BA are one trunk batch so every conditioning layer centers both
        directions with one synchronized statistic and updates its single EMA
        exactly once.
        """
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        distributed_training = self.training and dist.is_available() and dist.is_initialized()
        need_topo = bool(masks.topo.any()) or distributed_training
        if need_topo and (context.topo_ab is None or context.topo_ba is None):
            raise ValueError("pair context does not contain topology tokens")
        topo_tokens = None
        if need_topo:
            assert context.topo_ab is not None and context.topo_ba is not None
            topo_tokens = torch.cat((context.topo_ab, context.topo_ba))
        max_tokens = max(context.encoded_a.size(1), context.encoded_b.size(1))
        encoded_a = F.pad(
            context.encoded_a,
            (0, 0, 0, max_tokens - context.encoded_a.size(1)),
        )
        encoded_b = F.pad(
            context.encoded_b,
            (0, 0, 0, max_tokens - context.encoded_b.size(1)),
        )
        pair_features = self.trunk(
            torch.cat((encoded_a, encoded_b)),
            torch.cat((encoded_b, encoded_a)),
            torch.cat((context.len_a, context.len_b)),
            torch.cat((context.len_b, context.len_a)),
            topo_tokens=topo_tokens,
            topo_active=torch.cat((masks.topo, masks.topo)) if need_topo else None,
            edge_mask=torch.cat((edge_mask, edge_mask)) if edge_mask is not None else None,
        )
        feat_ab, feat_ba = pair_features.chunk(2)
        feat = torch.max(torch.stack([feat_ab, feat_ba], dim=-1), dim=-1).values
        # The registered mixed-precision contract keeps logits in fp32. Casting
        # after the head is too late: autocast has already quantized the linear
        # outputs, which can destroy the small full-minus-f-only residual.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            logits: torch.Tensor = self.head(feat.float()).squeeze(-1)
        return logits

    def forward(
        self, batch: dict[str, torch.Tensor], *, masks: HeadNullMasks | None = None
    ) -> dict[str, torch.Tensor]:
        """Score one pair batch under an optional head-null condition.

        Args:
            batch: Pair batch with ``emb_a``/``emb_b`` token streams,
                ``len_a``/``len_b`` lengths, and the ``x_a``/``x_b`` per-node
                features the Stage-1 generator's `encode_nodes` consumes.
            masks: Optional per-pair topo activity mask; ``None`` defaults to
                the topo pathway fully active (``NULL_NONE``).

        Returns:
            ``{"logits": (B,)}`` fused edge logits.
        """
        batch_size = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        has_relational_targets = "rel_target" in batch
        distributed_training = self.training and dist.is_available() and dist.is_initialized()
        need_topo = bool(masks.topo.any()) or has_relational_targets or distributed_training
        state_a, state_b, is_self = self._pair_node_states(batch)
        context = self.build_pair_context_from_states(
            state_a,
            state_b,
            is_self,
            need_topo=need_topo,
        )
        output = {
            "logits": self.score_pair_context(
                context,
                masks=masks,
                edge_mask=batch.get("edge_mask"),
            )
        }
        if has_relational_targets:
            output.update(self._training_relational_losses(batch, state_a, state_b, context))
        return output

    def decompose(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the two-logit decomposition via the eval-time hard bypass.

        Args:
            batch: Pair batch (see :meth:`forward`).

        Returns:
            ``{"full", "f_logit"}`` logits.
        """
        with torch.no_grad():
            context = self.build_pair_context(batch)
            return self.decompose_pair_context(context)

    def decompose_pair_context(self, context: E2EPairContext) -> dict[str, torch.Tensor]:
        """Evaluate both logits without rebuilding node or pair state."""
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        return {
            "full": self.score_pair_context(context),
            "f_logit": self.score_pair_context(
                context, masks=masks_for_null(NULL_ALL_HEAD, batch_size, device)
            ),
        }
