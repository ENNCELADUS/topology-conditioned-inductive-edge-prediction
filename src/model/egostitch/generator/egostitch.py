"""``EgoStitchImagineGenerator``: today's Tokenize-lite + Imagine + Stitch pipeline.

Design §1, §3.3, §5: assembling the joint stitched scaffold -- Sinkhorn
alignment plus scaffold assembly -- is imagination, not encoding, so this
component owns the stitching. It stops at the scaffold: encoding that
structure into token states is the graph encoder's job
(`src.model.egostitch.encoder.TypedMessagePassingEncoder`, the
behaviour-preserving port of the original `STEncoder`, which is deleted),
which consumes the `StitchedGraph` this module emits.

Split into `encode_node` (cacheable per-node) and `stitch` (pair-level)
per `NeighborhoodGenerator`'s two-phase contract (design correction,
2026-08-03): `score_universe.py`/`train_egostitch.py` encode each node in a
scored universe exactly once and reuse that state across every pair it
appears in, so a fused per-pair call would re-encode both endpoints on every
pair. Numerically, `stitch` reproduces
`EgoStitchE2E.build_pair_context_from_states`'s topology-token construction
(`e2e_model.py`) up to (not including) the STE call: same self-row
identity-plan short-circuit, same fp32 Sinkhorn island, same `build_scaffold`
assembly including the optional scaffold-structure `perturbation`
(`e2e_model.py:312`). `tests/model/test_generator_component.py` proves the
equivalence directly against the live `EgoStitchE2E` path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from src.data.feature_stats import FeatureStats
from src.model.egostitch.config import EgoStitchConfig
from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.graph import ImaginedGraph
from src.model.egostitch.imagine import SlotSet
from src.model.egostitch.losses import (
    LossFamily,
    alignment_loss,
    alignment_teacher_cells,
)
from src.model.egostitch.matching import match_slots
from src.model.egostitch.model import EgoStitchStage1, FeatureStandardizationMode
from src.model.egostitch.scaffold import (
    ScaffoldInputPerturbation,
    ScaffoldTokens,
    build_scaffold,
    swap_direction,
)
from src.model.egostitch.stitch import sinkhorn_log_plan

# `mask` is smuggled into `x`'s channel 4 today (scaffold.py: node-existence
# channel, `scaffold.feats[:, :, 4] = cat([1, 1, pi_src, pi_dst])`) --
# `ImaginedGraph.mask` reads it back out rather than recomputing it, so the
# two can never silently disagree.
_EXISTENCE_CHANNEL = 4


def _slotset_to_aux(prefix: str, slots: SlotSet) -> dict[str, torch.Tensor]:
    """Flatten a `SlotSet` into `aux`'s ``{prefix}_{field}`` keys."""
    return {
        f"{prefix}_{field}": value
        for field, value in zip(slots._fields, slots, strict=True)
    }


def _slotset_from_aux(aux: Mapping[str, torch.Tensor], prefix: str) -> SlotSet:
    """Reconstruct a `SlotSet` from `aux`'s ``{prefix}_{field}`` keys."""
    return SlotSet(*(aux[f"{prefix}_{field}"] for field in SlotSet._fields))


class GeneratorNodeState(NamedTuple):
    """The cacheable per-node half of `EgoStitchImagineGenerator`'s imagination.

    Mirrors `E2ENodeState` (`e2e_model.py`) minus the raw-token-encoder
    fields (`encoded`/`length`), which now belong to the classifier
    component: every field here index-selects, index-copies, and
    concatenates the same way `E2ENodeState`'s generator-owned fields do
    today (`score_universe.py:1859-1882`, `train_egostitch.py:2944-2968`),
    so existing per-node caching call sites port over field-by-field.

    Attributes:
        slots: The generated `SlotSet` (spec Sec 2 heads).
        projected_x: Shape ``(B, d_p)`` the node's own frozen features,
            projected into the shared ``d -> d_p`` space
            (`EgoStitchStage1.project_features`).
        ground_ids: Optional shape ``(B, n_g)`` grounding-candidate global
            ids, carried through unchanged; `None` when the caller does not
            track them.
    """

    slots: SlotSet
    projected_x: torch.Tensor
    ground_ids: torch.Tensor | None


@dataclass(frozen=True)
class StitchedGraph(ImaginedGraph):
    """The joint stitched scaffold `EgoStitchImagineGenerator` emits.

    Structurally identical to `ImaginedGraph` (``x``: ``(B, 34, 11)``,
    ``adj``: ``(B, 4, 34, 34)``); the only override is `swapped`.
    """

    def swapped(self) -> StitchedGraph:
        """Relabel the four anchor one-hot channels for the BA stream.

        Delegates to the existing `swap_direction` logic (`scaffold.py`)
        instead of rebuilding anything: the joint graph is built exactly
        once per pair (design §4). `adj` is passed straight through
        `swap_direction` untouched, so it is literally the same tensor
        object on the returned graph; `mask` and `aux` are shared by
        reference here directly.

        Returns:
            A new `StitchedGraph` with relabelled ``x``, shared ``adj`` /
            ``mask`` / ``aux``, and ``directed`` flipped.
        """
        relabelled = swap_direction(ScaffoldTokens(feats=self.x, adj=self.adj))
        return StitchedGraph(
            x=relabelled.feats,
            adj=relabelled.adj,
            mask=self.mask,
            aux=self.aux,
            directed=not self.directed,
        )


class EgoStitchImagineGenerator(NeighborhoodGenerator):
    """Wraps today's Tokenize-lite + Imagine + Stitch + scaffold pipeline.

    Internally owns a trainable `EgoStitchStage1` (Tokenize-lite + Imagine +
    the frozen random-GIN realism head), exactly as `EgoStitchE2E.generator`
    does today (`e2e_model.py:107-115`). Everything downstream of
    `encode_nodes` -- Sinkhorn alignment, scaffold assembly -- moves here
    from `e2e_model.py`, unchanged.
    """

    def __init__(
        self,
        generator_cfg: EgoStitchConfig,
        *,
        feature_standardization: FeatureStandardizationMode = "zscore_vfit_v1",
        loss_family: LossFamily = "egostitch_e2e",
    ) -> None:
        """Build the internal Stage-1 generator.

        Args:
            generator_cfg: The (already-merged) Stage-1 hyperparameters --
                callers translate a nested ``generator:`` config section into
                this the same way `EgoStitchE2E.__init__` does today
                (``replace(EgoStitchConfig(), n_ground=..., tau_adj=...)``).
            feature_standardization: Registered F0 preprocessing mode.
            loss_family: Selects the frozen Stage-1 or rev-3.1 E2E loss
                decomposition inside `EgoStitchStage1`.
        """
        super().__init__()
        self.cfg = generator_cfg
        self.stage1 = EgoStitchStage1(
            generator_cfg,
            feature_standardization=feature_standardization,
            loss_family=loss_family,
        )

    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered F0 standardization constants (spec Sec 13.19.1).

        Args:
            stats: The V_fit statistics bundle.
        """
        self.stage1.set_feature_stats(stats)

    @property
    def feature_stats_digest_hex(self) -> str:
        """The registered `feature_stats_sha256`, or ``""`` for stateless modes."""
        return self.stage1.feature_stats_digest_hex

    @staticmethod
    def _select_slots(slots: SlotSet, rows: torch.Tensor) -> SlotSet:
        """Index every tensor in one slot bundle by batch row."""
        return SlotSet(*(value.index_select(0, rows) for value in slots))

    def encode_node(
        self,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
    ) -> GeneratorNodeState:
        """Run the cacheable per-node pass: `EgoStitchStage1.encode_nodes` + `project_features`.

        Args:
            x: Shape ``(B, d)`` frozen node features.
            ground: Shape ``(B, n_g, d)`` grounding-candidate features.
            ground_ids: Optional shape ``(B, n_g)`` grounding-candidate
                global ids, carried through unchanged.

        Returns:
            The `GeneratorNodeState` bundle.
        """
        generated = self.stage1.encode_nodes(x, ground)
        return GeneratorNodeState(
            slots=generated.slots,
            projected_x=self.stage1.project_features(x),
            ground_ids=ground_ids,
        )

    def stitch(
        self,
        state_a: GeneratorNodeState,
        state_b: GeneratorNodeState,
        is_self: torch.Tensor,
        *,
        perturbation: ScaffoldInputPerturbation | None = None,
    ) -> StitchedGraph:
        """Imagine the joint stitched scaffold from two already-encoded endpoint states.

        Mirrors `EgoStitchE2E.build_pair_context_from_states` (`e2e_model.py`)
        exactly: self-pair rows (``is_self``) get an exact identity alignment
        plan without running Sinkhorn on that row at all; non-self rows run
        the unbalanced Sinkhorn plan (an fp32 island regardless of the
        ambient autocast state -- see `stitch.sinkhorn_log_plan`) and
        `build_scaffold` promotes to fp32 before any product is formed.
        ``perturbation``, when given, is forwarded to `build_scaffold`
        unchanged -- the deterministic §14.4.5 scaffold-structure control
        (e.g. the mandatory 6a-shuffle / 6e-rewire arms) applied to
        ``(adj_src, adj_dst, plan)`` before any graph channel is derived.

        Args:
            state_a: Endpoint-A state from `encode_node`.
            state_b: Endpoint-B state from `encode_node`.
            is_self: Shape ``(B,)`` boolean self-pair mask.
            perturbation: Optional deterministic scaffold-structure control.

        Returns:
            The `StitchedGraph` (``x``: ``(B, 34, 11)``, ``adj``:
            ``(B, 4, 34, 34)``, ``mask``: ``(B, 34)``, ``aux`` carrying both
            `SlotSet`s and the Sinkhorn plan).

        Raises:
            ValueError: If ``state_a``/``state_b`` disagree on batch size or
                slot count, or if ``is_self``'s shape disagrees with the
                batch size.
        """
        slots_a, slots_b = state_a.slots, state_b.slots
        batch_size, num_slots = slots_a.pi.shape
        if slots_b.pi.shape != (batch_size, num_slots):
            raise ValueError(
                "state_a and state_b must have matching batch size and slot count: "
                f"{tuple(slots_a.pi.shape)} != {tuple(slots_b.pi.shape)}"
            )
        if is_self.shape != (batch_size,):
            raise ValueError(f"is_self shape must be {(batch_size,)}, got {tuple(is_self.shape)}")

        # fp32 island: the assembled plan destination must already be fp32
        # before either the identity block or the Sinkhorn result (itself an
        # fp32 island, stitch.py) is copied in.
        plan = slots_a.pi.new_zeros((batch_size, num_slots, num_slots), dtype=torch.float32)
        log_plan = slots_a.pi.new_full(
            (batch_size, num_slots, num_slots), -torch.inf, dtype=torch.float32
        )
        self_rows = torch.nonzero(is_self, as_tuple=False).squeeze(-1)
        if self_rows.numel() > 0:
            identity = torch.eye(num_slots, device=plan.device, dtype=plan.dtype)
            plan = plan.index_copy(0, self_rows, identity.expand(self_rows.numel(), -1, -1))
            log_identity = torch.diag_embed(
                torch.zeros(
                    self_rows.numel(), num_slots, device=log_plan.device, dtype=log_plan.dtype
                )
            )
            log_identity = log_identity.masked_fill(
                ~torch.eye(num_slots, device=log_plan.device, dtype=torch.bool).unsqueeze(0),
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
                eps=self.cfg.sinkhorn_eps,
                iters=self.cfg.sinkhorn_iters,
                tau=self.cfg.sinkhorn_tau,
            )
            plan = plan.index_copy(0, non_self, torch.exp(non_self_log_plan))
            log_plan = log_plan.index_copy(0, non_self, non_self_log_plan)

        scaffold = build_scaffold(slots_a, slots_b, plan, perturbation=perturbation)
        aux: dict[str, torch.Tensor] = {
            **_slotset_to_aux("slots_a", slots_a),
            **_slotset_to_aux("slots_b", slots_b),
            "plan": plan,
            "log_plan": log_plan,
        }
        return StitchedGraph(
            x=scaffold.feats,
            adj=scaffold.adj,
            mask=scaffold.feats[..., _EXISTENCE_CHANNEL],
            aux=aux,
            directed=True,
        )

    def auxiliary_losses(
        self, graph: StitchedGraph, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute every generator-owned loss component (design §6), unweighted.

        Two independent streams flow through ``batch``, matching the field
        names `train_egostitch.py`'s ``node``/``edge`` dicts already use
        today:

        - Node-stream keys supervise ``L_recon``
          (feat/exist/mult/slotadj/gate/ptr/div), ``L_deg`` and ``L_real``
          (egostat + GIN) and ``L_ssl`` via a fresh `encode_nodes` pass over
          whatever node batch the trainer samples that step -- independent
          of ``graph``, which was built from a different (pair) batch:
          ``x``, ``ground_x``, ``target_features``, ``target_mult``,
          ``target_adj``, ``target_mask``, ``target_in_pool``,
          ``target_pool_index``, ``true_degree``, ``real_ego_stats``,
          ``ground_resampled``, ``ssl_noise``, and the optional
          ``null_mode`` / ``denoise_features`` / ``denoise_mask`` /
          ``denoise_noise`` (mirrors `EgoStitchStage1.node_losses` /
          `.ssl_losses`).
        - Edge-stream keys supervise ``L_align`` against ``graph``'s own
          slots and Sinkhorn ``log_plan`` -- no re-encoding:
          ``target_features_a`` / ``_b``, ``target_mult_a`` / ``_b``,
          ``target_adj_a`` / ``_b``, ``target_mask_a`` / ``_b``,
          ``target_node_index_a`` / ``_b``, ``label``, ``edge_mask``, and
          the optional ``loss_world_size`` (defaults to 1). Mirrors
          `EgoStitchE2E._training_relational_losses`'s align half
          (`e2e_model.py`), minus ``rel_loss`` -- that belongs to the
          encoder now.

        Every returned value is an *unweighted* scalar component; the
        composite applies the registered weights (`losses._recon_total`,
        `losses.stage1_total`).

        Args:
            graph: This generator's own most recent output for the pair
                half of ``batch``.
            batch: The combined node+edge training batch (see above).

        Returns:
            ``{"feat", "exist", "mult", "slotadj", "gate", "ptr", "div",
            "deg", "real_egostat", "real_gin", "ssl_noise", "ssl_pool",
            "align"}``.
        """
        node_losses, _ = self.stage1.node_losses(
            batch["x"],
            batch["ground_x"],
            target_features=batch["target_features"],
            target_mult=batch["target_mult"],
            target_adj=batch["target_adj"],
            target_mask=batch["target_mask"],
            target_in_pool=batch["target_in_pool"],
            target_pool_index=batch["target_pool_index"],
            true_degree=batch["true_degree"],
            real_ego_stats=batch["real_ego_stats"],
            null_mode=batch.get("null_mode"),
            denoise_features=batch.get("denoise_features"),
            denoise_mask=batch.get("denoise_mask"),
            denoise_noise=batch.get("denoise_noise"),
        )
        ssl = self.stage1.ssl_losses(
            batch["x"],
            batch["ground_x"],
            batch["ground_resampled"],
            noise=batch["ssl_noise"],
        )

        slots_a = _slotset_from_aux(graph.aux, "slots_a")
        slots_b = _slotset_from_aux(graph.aux, "slots_b")
        target_proj_a = self.stage1.project_features(batch["target_features_a"]).detach()
        target_proj_b = self.stage1.project_features(batch["target_features_b"]).detach()
        assignment_a = match_slots(
            slots_a,
            target_proj=target_proj_a,
            target_mult=batch["target_mult_a"],
            target_adj=batch["target_adj_a"],
            target_mask=batch["target_mask_a"],
        )
        assignment_b = match_slots(
            slots_b,
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
            slots=self.cfg.slots,
        )
        world_size = int(batch["loss_world_size"].item()) if "loss_world_size" in batch else 1
        positive_real_mask = batch["edge_mask"] * (batch["label"] == 1).float()
        align = alignment_loss(
            graph.aux["log_plan"],
            teacher_cells,
            positive_real_mask=positive_real_mask,
            world_size=world_size,
        )

        return {
            "feat": node_losses.recon["feat"],
            "exist": node_losses.recon["exist"],
            "mult": node_losses.recon["mult"],
            "slotadj": node_losses.recon["slotadj"],
            "gate": node_losses.recon["gate"],
            "ptr": node_losses.recon["ptr"],
            "div": node_losses.recon["div"],
            "deg": node_losses.deg,
            "real_egostat": node_losses.real_egostat,
            "real_gin": node_losses.real_gin,
            "ssl_noise": ssl["noise"],
            "ssl_pool": ssl["pool"],
            "align": align,
        }
